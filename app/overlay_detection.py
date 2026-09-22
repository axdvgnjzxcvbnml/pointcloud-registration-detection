# ============================================================
# app/overlay_detection.py —— 点云上叠加 3D 检测框与类别文本
# ------------------------------------------------------------
# 功能：
#   1. 加载点云（.npz/.ply）与检测结果 JSON
#      （格式对齐 detection/evaluate_detection.py 的保存输出：
#       {"boxes": [[cx,cy,cz,sx,sy,sz,heading], ...],
#        "labels": [int, ...], "scores": [float, ...]}）；
#   2. 用 Open3D LineSet 绘制每个 3D 边界框（12 条边，按类别着色），
#      框心加小圆球标记；
#   3. 离屏渲染 PNG，并把“类别 分数”文本标注投影到框顶部中心
#      （numpy 手工视锥投影 + PIL 绘制，headless 服务器可用）；
#   4. 支持交互模式查看 3D 场景。
#
# 用法示例：
#   python app/overlay_detection.py --point-cloud results/det/frame_000.npz \
#       --detections results/det/frame_000_pred.json --out-dir results/vis
#
# 依赖：numpy / open3d / scipy / matplotlib（调色板）/ PIL（纯 CPU）
# ============================================================
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

from render_compat import make_offscreen_renderer, setup_camera as setup_camera_compat

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("overlay_detection")

# SUN RGB-D 检测 10 类（顺序与 configs/default.yaml 一致）
CLASS_NAMES = ["bathtub", "bed", "bookshelf", "chair", "desk",
               "dresser", "night_stand", "sofa", "table", "toilet"]

# 3D 框 12 条边（顶点索引，8 个角点按小盒子顺序编号）
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),   # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7),   # 竖直边
]


# ---------------------------------------------------------------
# 3D 框几何
# ---------------------------------------------------------------
def box7_to_corners(box7):
    """由 (cx, cy, cz, sx, sy, sz, heading) 计算 8 个角点 (8, 3)。

    heading 为绕 z 轴的旋转角（rad），先构建轴对齐盒再绕 z 旋转，
    与 evaluate_detection.py 的 3D IoU 口径保持一致。
    """
    cx, cy, cz, sx, sy, sz, heading = np.asarray(box7, dtype=np.float64)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    cos_t, sin_t = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_t, -sin_t, 0.0],
                    [sin_t, cos_t, 0.0],
                    [0.0, 0.0, 1.0]])
    return corners @ rot.T + np.array([cx, cy, cz])


def box_lineset(box7, color=(1.0, 0.3, 0.1)):
    """将 3D 框转为 Open3D LineSet（12 条边）。"""
    corners = box7_to_corners(box7)
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    lines.lines = o3d.utility.Vector2iVector(np.asarray(BOX_EDGES, dtype=np.int32))
    lines.paint_uniform_color(color)
    return lines


def class_color(label):
    """按类别取稳定颜色（matplotlib tab20 调色板，RGB 0~1）。"""
    import matplotlib
    idx = int(label) % 20
    # matplotlib >=3.9 移除了 cm.get_cmap 顶层接口，优先用新 API
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps["tab20"](idx)[:3]
    from matplotlib import cm
    return cm.get_cmap("tab20")(idx)[:3]


# ---------------------------------------------------------------
# 手工视锥投影（用于把 3D 标注锚点投到 2D 图像坐标）
# ---------------------------------------------------------------
def build_view_proj(center, eye, up, fov_deg, width, height, near=0.01, far=100.0):
    """构造 view/projection 矩阵（OpenGL 约定），返回 (V, P)。

    用于把 3D 框中心投影到渲染图像坐标，绘制类别文本。
    注意：与 Open3D 内部相机可能略有偏差，若文本位置偏移，
    可通过微调 near/far 或 eye 位置校正（骨架约定）。
    """
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=1)          # (3,3)
    t = -R.T @ eye
    V = np.eye(4)
    V[:3, :3] = R.T
    V[:3, 3] = t

    fovy = np.deg2rad(fov_deg)
    f_py = 1.0 / np.tan(fovy / 2.0)
    aspect = width / height
    P = np.zeros((4, 4))
    P[0, 0] = f_py / aspect
    P[1, 1] = f_py
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2.0 * far * near / (near - far)
    P[3, 2] = -1.0
    return V, P


def project_to_pixels(points3d, V, P, width, height):
    """把 (M,3) 场景坐标点投影为 (M,2) 像素坐标与可见性。

    返回 (uv, visible)：uv 为 (M,2) 浮点像素（u 右 / v 下），
    visible 为布尔（位于视锥内、z 在 [near, far] 内）。
    """
    pts = np.hstack([np.asarray(points3d, dtype=np.float64), np.ones((len(points3d), 1))])
    cam = pts @ V.T          # 相机坐标
    clip = cam @ P.T         # 裁剪坐标
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / clip[:, 3:4]
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    visible = (clip[:, 3] > 0) & (np.abs(ndc[:, 0]) <= 1.0) & \
              (np.abs(ndc[:, 1]) <= 1.0) & (np.abs(ndc[:, 2]) <= 1.0)
    return np.stack([px, py], axis=1), visible


# ---------------------------------------------------------------
# 渲染与标注
# ---------------------------------------------------------------
def load_pointcloud(path):
    """加载 .npz 或 .ply 为 open3d PointCloud（与 compare_registration 一致）。"""
    from compare_registration import load_pointcloud as _lp
    return _lp(path)


def load_detections(path):
    """读取检测结果 JSON → [{"box7": ndarray(7,), "label": int, "score": float}]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = np.asarray(data["boxes"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    scores = np.asarray(data["scores"], dtype=np.float64)
    dets = []
    for b, l, s in zip(boxes, labels, scores):
        dets.append({"box7": b, "label": int(l), "score": float(s)})
    return dets


def build_scene_geometry(pcd, dets, score_threshold=0.5):
    """构建 [点云, 框 LineSet..., 框心小球...] 的渲染几何列表。"""
    geos = [pcd]
    kept = []
    for d in dets:
        if d["score"] < score_threshold:
            continue
        color = class_color(d["label"])
        geos.append(box_lineset(d["box7"], color=color))
        center = d["box7"][:3]
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sph.translate(center)
        sph.paint_uniform_color(color)
        geos.append(sph)
        kept.append(d)
    return geos, kept


def render_with_text(pcd, dets, out_path, width=1280, height=720,
                     fov=60.0, elevation=25.0, azimuth=30.0,
                     dist_scale=1.6, score_threshold=0.5):
    """离屏渲染点云+检测框，并把类别文本标注叠加到 PNG。

    步骤：
      1) 用 OffscreenRenderer 渲染（点云+框+框心小球）；
      2) 用同一相机参数构建 V/P，把每个框顶部中心 (cx, cy, cz+sz/2)
         投影到图像坐标；
      3) PIL 绘制“类别 分数”文本。
    """
    # --- 1. 渲染底图 ---
    geos, kept = build_scene_geometry(pcd, dets, score_threshold)
    pts = np.concatenate([np.asarray(p.points) for p in geos
                          if isinstance(p, o3d.geometry.PointCloud)], axis=0)
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) or 1.0
    dist = radius * dist_scale
    eye = np.array([
        center[0] + dist * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth)),
        center[1] + dist * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth)),
        center[2] + dist * np.sin(np.deg2rad(elevation)),
    ])
    up = np.array([0.0, 0.0, 1.0])

    renderer = make_offscreen_renderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 2.0
    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    mat_sph = rendering.MaterialRecord()
    mat_sph.shader = "defaultLit"
    for i, g in enumerate(geos):
        mat = mat_pcd if isinstance(g, o3d.geometry.PointCloud) else \
              (mat_line if isinstance(g, o3d.geometry.LineSet) else mat_sph)
        renderer.scene.add_geometry(f"g_{i}", g, mat)
    setup_camera_compat(renderer, fov, center, eye, up)
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)

    # --- 2. 文本标注叠加 ---
    V, P = build_view_proj(center, eye, up, fov, width, height)
    anchors = np.array([d["box7"][:3] + np.array([0.0, 0.0, d["box7"][5] / 2.0])
                        for d in kept]) if kept else np.zeros((0, 3))
    uv, visible = project_to_pixels(anchors, V, P, width, height)

    canvas = Image.open(out_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    for d, (u, v), ok in zip(kept, uv, visible):
        if not ok:
            continue
        label = CLASS_NAMES[d["label"]] if 0 <= d["label"] < len(CLASS_NAMES) \
            else f"cls_{d['label']}"
        text = f"{label} {d['score']:.2f}"
        r, g, b = [int(c * 255) for c in class_color(d["label"])]
        draw.rectangle([u + 2, v + 2, u + 2 + draw.textlength(text, font=font),
                        v + 22], fill=(255, 255, 255))
        draw.text((u + 4, v + 4), text, font=font, fill=(r, g, b))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    log.info("检测可视化已保存：%s（%d 个框）", out_path, len(kept))


def _load_font(size=18):
    """加载字体：优先 matplotlib 内置 DejaVuSans，失败用默认字体。"""
    try:
        from matplotlib import get_data_path
        font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return ImageFont.truetype(str(font_path), size)
    except Exception:
        return ImageFont.load_default()


def visualize_interactive(pcd, dets, window_name="Detection Overlay",
                          score_threshold=0.5):
    """交互模式：Open3D 视窗查看 3D 场景（无文本，文本见离屏输出）。"""
    geos, _ = build_scene_geometry(pcd, dets, score_threshold)
    o3d.visualization.draw_geometries(geos, window_name=window_name,
                                      width=1280, height=720)


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="在点云上叠加 3D 检测框（Open3D LineSet）与类别文本标注")
    p.add_argument("--point-cloud", type=str, required=True,
                   help="点云文件 .npz/.ply")
    p.add_argument("--detections", type=str, required=True,
                   help="检测结果 JSON（boxes/labels/scores，见文件头说明）")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="分数过滤阈值（默认 0.5）")
    p.add_argument("--interactive", action="store_true",
                   help="交互模式（有显示器时使用）")
    p.add_argument("--out-dir", type=str, default="results/vis",
                   help="渲染输出目录（默认 results/vis）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=30.0)
    p.add_argument("--elevation", type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()
    pcd = load_pointcloud(args.point_cloud)
    dets = load_detections(args.detections)
    log.info("加载点云 %s（%d 点），检测框 %d 个",
             args.point_cloud, len(pcd.points), len(dets))

    if args.interactive:
        visualize_interactive(pcd, dets, score_threshold=args.score_threshold)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.point_cloud).stem}_detection.png"
    render_with_text(pcd, dets, out_path, width=args.width, height=args.height,
                     azimuth=args.azimuth, elevation=args.elevation,
                     score_threshold=args.score_threshold)


if __name__ == "__main__":
    main()
