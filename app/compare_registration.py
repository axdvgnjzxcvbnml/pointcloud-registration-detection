# ============================================================
# app/compare_registration.py —— 左右视图对比配准前后点云
# ------------------------------------------------------------
# 功能：
#   1. 加载源/目标点云（.npz 或 .ply）；
#   2. 变换来源二选一：
#        a) --transform 提供 4×4 矩阵（.npy / 文本文件，来自
#           registration/evaluate_registration.py 的输出）；
#        b) --run-pipeline 现场跑配准流水线（预处理→FPFH+RANSAC
#           粗配准→改进 ICP 精配准，纯 CPU）；
#   3. 输出“左右视图”对比图：左=配准前（源红/目标灰），
#      右=配准后（源经变换转绿/目标灰），左右用同一相机视角，
#      保证视觉可比；同时支持交互模式查看。
#
# 用法示例：
#   python app/compare_registration.py --source data/pairs/s_001.npz \
#       --target data/pairs/t_001.npz --transform results/pairs/T_001.npy \
#       --out-dir results/vis
#   python app/compare_registration.py --source a.ply --target b.ply \
#       --run-pipeline --interactive
#
# 依赖：numpy / open3d / scipy / PIL（纯 CPU）
# ============================================================
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image

# 使 `import registration.*` 可用（无论从项目根还是 app/ 运行）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("compare_registration")

# 配色（RGB 0~1）
COLOR_SOURCE_RAW = (0.85, 0.25, 0.20)   # 红：源点云（未对齐）
COLOR_TARGET     = (0.62, 0.62, 0.62)   # 灰：目标点云
COLOR_ALIGNED    = (0.20, 0.75, 0.25)   # 绿：对齐后的源点云


# ---------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------
def load_pointcloud(path):
    """加载 .npz（point_cloud/xyz + colors/rgb）或 .ply 为 open3d PointCloud。"""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        key = "point_cloud" if "point_cloud" in data.files else "xyz"
        xyz = np.asarray(data[key], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        for ckey in ("colors", "rgb", "color"):
            if ckey in data.files:
                colors = np.clip(np.asarray(data[ckey], dtype=np.float32) / 255.0, 0, 1)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                break
        return pcd
    return o3d.io.read_point_cloud(str(path))


def load_transform(path):
    """读取 4×4 变换矩阵：.npy 或文本（空格/换行分隔的 16 个数字）。"""
    if Path(path).suffix.lower() == ".npy":
        return np.load(path)
    return np.loadtxt(path).reshape(4, 4)


# ---------------------------------------------------------------
# 配准流水线（可选路径，纯 CPU，参数与 configs/default.yaml 对齐）
# ---------------------------------------------------------------
def run_registration_pipeline(source_pcd, target_pcd, voxel_size=0.02):
    """预处理 → FPFH+RANSAC 粗配准 → 改进 ICP 精配准。

    复用 registration/ 模块（保持单一实现，避免本文件重复逻辑）。
    参数与 configs/default.yaml 的 registration 段一致：
      统计去噪 k=20/std=2.0；体素 0.02；FPFH 半径 0.25；
      RANSAC 迭代 100000、距离阈值 0.03（1.5×voxel）；
      ICP 最大迭代 50、阈值下限 0.02、系数 0.5。
    """
    from registration.preprocess_pointcloud import preprocess_pointcloud
    from registration.coarse_registration import coarse_registration
    from registration.fine_registration import fine_registration

    src = preprocess_pointcloud(source_pcd, voxel_size=voxel_size,
                                nb_neighbors=20, std_ratio=2.0, radius_normal=0.1)
    tgt = preprocess_pointcloud(target_pcd, voxel_size=voxel_size,
                                nb_neighbors=20, std_ratio=2.0, radius_normal=0.1)
    T_coarse, _ = coarse_registration(src, tgt, voxel_size=voxel_size)
    T_fine, info = fine_registration(src, tgt, T_coarse, voxel_size=voxel_size)
    log.info("配准完成：RMSE=%.4f, fitness=%.4f", info["rmse"], info["fitness"])
    return T_fine


# ---------------------------------------------------------------
# 场景构建与渲染
# ---------------------------------------------------------------
def build_comparison_geometries(source_pcd, target_pcd, T):
    """构建左右两半的场景几何。

    返回:
        left  : [源(红), 目标(灰)]           —— 配准前
        right : [源经 T 变换(绿), 目标(灰)]   —— 配准后
    """
    source_raw = o3d.geometry.PointCloud(source_pcd)
    source_raw.paint_uniform_color(COLOR_SOURCE_RAW)
    source_aligned = o3d.geometry.PointCloud(source_pcd)
    source_aligned.transform(T)
    source_aligned.paint_uniform_color(COLOR_ALIGNED)
    target = o3d.geometry.PointCloud(target_pcd)
    target.paint_uniform_color(COLOR_TARGET)
    return [source_raw, target], [source_aligned, target]


def _scene_camera(pcds, fov=60.0, elevation=25.0, azimuth=0.0, dist_scale=1.5):
    """由点云集合计算观察相机（center/eye/up），左右两半共用。"""
    pts = np.concatenate([np.asarray(p.points) for p in pcds], axis=0)
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    if radius <= 1e-9:
        radius = 1.0
    dist = radius * dist_scale
    eye = np.array([
        center[0] + dist * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth)),
        center[1] + dist * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth)),
        center[2] + dist * np.sin(np.deg2rad(elevation)),
    ])
    return {"center": center, "eye": eye, "up": np.array([0.0, 0.0, 1.0])}


def render_offscreen(pcds, out_path, camera, width=1280, height=720, fov=60.0,
                     point_size=2.0):
    """渲染一组点云为 PNG（离屏）。"""
    renderer = rendering.OffscreenRenderer(width, height, headless=True)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]
    mat.point_size = point_size
    for i, pcd in enumerate(pcds):
        renderer.scene.add_geometry(f"g_{i}", pcd, mat)
    renderer.scene.setup_camera(fov, camera["center"], camera["eye"], camera["up"])
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)


def render_side_by_side(left, right, out_path, camera,
                        width=1280, height=720, fov=60.0, point_size=2.0):
    """左右两半各渲染一次（同一相机），横向拼接成一张对比图。"""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="reg_compare_"))
    l_path = tmp / "left.png"
    r_path = tmp / "right.png"
    render_offscreen(left, l_path, camera, width, height, fov, point_size)
    render_offscreen(right, r_path, camera, width, height, fov, point_size)

    l_img = Image.open(l_path)
    r_img = Image.open(r_path)
    canvas = Image.new("RGB", (l_img.width + r_img.width, l_img.height),
                       (255, 255, 255))
    canvas.paste(l_img, (0, 0))
    canvas.paste(r_img, (l_img.width, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    log.info("对比图已保存：%s（左=配准前 右=配准后）", out_path)


def visualize_interactive(left, right, window_name="Registration Compare"):
    """交互模式：依次弹出两个视窗（配准前 → 配准后）。"""
    for title, geos in (("Before", left), ("After", right)):
        o3d.visualization.draw_geometries(geos, window_name=f"{window_name} - {title}",
                                          width=1280, height=720)


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="左右视图对比点云配准前后（左=前/右=后，同一相机）")
    p.add_argument("--source", type=str, required=True, help="源点云 .npz/.ply")
    p.add_argument("--target", type=str, required=True, help="目标点云 .npz/.ply")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--transform", type=str, default=None,
                     help="4×4 变换矩阵文件（.npy 或文本）")
    src.add_argument("--run-pipeline", action="store_true",
                     help="现场跑配准流水线（预处理→粗配准→精配准）")
    p.add_argument("--out-dir", type=str, default="results/vis",
                   help="对比图输出目录（默认 results/vis）")
    p.add_argument("--interactive", action="store_true",
                   help="交互模式（有显示器时使用）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=30.0,
                   help="相机方位角（度，默认 30）")
    p.add_argument("--elevation", type=float, default=25.0,
                   help="相机俯仰角（度，默认 25）")
    return p.parse_args()


def main():
    args = parse_args()
    source_pcd = load_pointcloud(args.source)
    target_pcd = load_pointcloud(args.target)

    if args.run_pipeline:
        T = run_registration_pipeline(source_pcd, target_pcd)
    else:
        T = load_transform(args.transform)
        log.info("已加载变换矩阵（shape=%s）", T.shape)

    left, right = build_comparison_geometries(source_pcd, target_pcd, T)
    camera = _scene_camera(left + right)   # 左右两半共用一个相机

    if args.interactive:
        visualize_interactive(left, right)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.source).stem}_vs_{Path(args.target).stem}_compare.png"
    render_side_by_side(left, right, out_path, camera,
                        width=args.width, height=args.height,
                        azimuth=args.azimuth, elevation=args.elevation)


if __name__ == "__main__":
    main()
