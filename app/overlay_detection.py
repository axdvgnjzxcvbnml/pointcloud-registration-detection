# ============================================================
# app/overlay_detection.py —— 在点云上叠加 3D 检测框与类别标注
# ------------------------------------------------------------
# 功能：
#   1. 加载点云（.npz：point_cloud/xyz + colors/rgb，可选）；
#   2. 加载检测结果 JSON（见“检测结果格式”）；
#   3. 用 Open3D LineSet 绘制 12 条棱边的 3D 框（带朝向），
#      框心画小球，框顶叠加“类别 分数”文本
#      （视锥投影到 2D，用 PIL 绘制后贴回，规避 open3d 文本限制）；
#   4. 输出叠加后的 PNG（离屏渲染，服务器可用）。
#
# 检测结果格式（JSON）：
#   {
#     "boxes":  [[cx,cy,cz,sx,sy,sz,heading], ...],   # (M,7)
#     "labels": ["chair", ...],
#     "scores": [0.92, ...]
#   }
#
# 用法示例：
#   python app/overlay_detection.py --point-cloud frame_000.npz \
#       --detections pred_000.json --out-dir results/vis
#
# 依赖：numpy / open3d / Pillow（纯 CPU）
# ============================================================
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("overlay_detection")


# ---------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------
def load_pointcloud(path):
    """加载 .npz 点云（point_cloud/xyz + colors/rgb）为 open3d PointCloud。"""
    data = np.load(path)
    key = "point_cloud" if "point_cloud" in data.files else "xyz"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(data[key], dtype=np.float64))
    for ckey in ("colors", "rgb", "color"):
        if ckey in data.files:
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(np.asarray(data[ckey], dtype=np.float32) / 255.0, 0, 1))
            break
    return pcd


def load_detections(path):
    """加载检测结果 JSON，返回 [{box7, label, score}, ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for b, l, s in zip(data["boxes"], data["labels"], data["scores"]):
        out.append({"box7": np.asarray(b, dtype=np.float64),
                    "label": str(l), "score": float(s)})
    return out


# ---------------------------------------------------------------
# 3D 框几何（12 棱边 LineSet）
# ---------------------------------------------------------------
def box_corners(box7):
    """由 [cx,cy,cz,sx,sy,sz,heading] 计算 8 个角点（绕 z 轴旋转）。"""
    cx, cy, cz, sx, sy, sz, heading = box7
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    return (R @ corners.T).T + np.array([cx, cy, cz])


def box_lineset(box7, color=(0.9, 0.2, 0.2)):
    """构建 3D 框 LineSet（12 条棱边 + 8 个小球可另行叠加）。"""
    corners = box_corners(box7)
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),   # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),   # 竖棱
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


def box_center_sphere(box7, radius=0.02, color=(0.9, 0.2, 0.2)):
    """框心小球（便于在点云中定位）。"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(box7[:3])
    sphere.paint_uniform_color(color)
    return sphere


def class_color(label):
    """按类别哈希取稳定颜色（RGB 0~1）。"""
    import hashlib
    h = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------
# 文本标签（视锥投影 → PIL 绘制）
# ---------------------------------------------------------------
def project_to_image(points3d, K, width, height):
    """3D 点投影到图像坐标（仅保留 z>0）。"""
    pts = np.asarray(points3d, dtype=np.float64)
    z = pts[:, 2]
    u = K[0, 0] * pts[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * pts[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), ok


def render_with_labels(pcd, detections, out_path, K=None, width=1280,
                       height=720, label_font_size=18):
    """渲染点云+3D 框，并把标签以 PIL 文本绘制到结果图上。

    K 缺省时给一个与视口近似的伪内参（仅用于标签投影，框本身
    由 Open3D 直接渲染在 3D 场景中，投影误差不影响主视觉）。
    """
    if K is None:
        K = np.array([[width, 0, width / 2],
                      [0, height, height / 2],
                      [0, 0, 1.0]], dtype=np.float64)

    renderer = rendering.OffscreenRenderer(width, height, headless=True)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 2.0
    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"

    renderer.scene.add_geometry("pcd", pcd, mat_pcd)
    for i, d in enumerate(detections):
        color = class_color(d["label"])
        renderer.scene.add_geometry(f"box_{i}", box_lineset(d["box7"], color), mat_line)
        renderer.scene.add_geometry(f"sph_{i}",
                                    box_center_sphere(d["box7"], color=color), mat_sphere)

    # 相机：看向所有框中心
    centers = np.array([d["box7"][:3] for d in detections]) if detections else None
    all_pts = np.asarray(pcd.points)
    if centers is not None and len(centers):
        center = centers.mean(axis=0)
    else:
        center = all_pts.mean(axis=0)
    radius = float(np.linalg.norm(all_pts - center, axis=1).max()) or 1.0
    eye = center + np.array([0.0, -1.4 * radius, 0.7 * radius])
    renderer.scene.setup_camera(60.0, center, eye, [0.0, 0.0, 1.0])

    img = renderer.render_to_image()
    tmp = "__overlay_base.png"
    o3d.io.write_image(tmp, img)
    pil = Image.open(tmp).convert("RGB")
    Path(tmp).unlink()

    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_font_size)
    except OSError:
        font = ImageFont.load_default()

    for d in detections:
        top = box_corners(d["box7"])[[4, 5, 6, 7]].mean(axis=0)
        (u, v), ok = project_to_image(top[None], K, width, height)
        if not ok[0]:
            continue
        label = f"{d['label']} {d['score']:.2f}"
        box_px = draw.textbbox((0, 0), label, font=font)
        w, h = box_px[2] - box_px[0], box_px[3] - box_px[1]
        x, y = int(u[0] - w / 2), int(v[0] - h - 6)
        r, g, b = class_color(d["label"])
        draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2],
                       fill=(int(r * 255), int(g * 255), int(b * 255)))
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    log.info("叠加结果已保存：%s（%d 个检测框）", out_path, len(detections))


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="在点云上叠加 3D 检测框与类别标注并渲染 PNG")
    p.add_argument("--point-cloud", type=str, required=True, help="点云 .npz")
    p.add_argument("--detections", type=str, required=True,
                   help="检测结果 JSON（见文件头格式）")
    p.add_argument("--out-dir", type=str, default="results/vis")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--label-size", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    pcd = load_pointcloud(args.point_cloud)
    detections = load_detections(args.detections)
    out_path = Path(args.out_dir) / f"{Path(args.point_cloud).stem}_det.png"
    render_with_labels(pcd, detections, out_path,
                       width=args.width, height=args.height,
                       label_font_size=args.label_size)


if __name__ == "__main__":
    main()
