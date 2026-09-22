# ============================================================
# app/load_scene.py —— 加载 SUN3D 场景多帧点云并用 Open3D 展示
# ------------------------------------------------------------
# 功能：
#   1. 读取场景目录下的帧文件（.npz：point_cloud/xyz 必选，
#      colors/rgb 可选），可体素下采样；
#   1. 交互模式：Open3D Visualizer 多窗口/单窗口展示（--interactive）；
#   2. 离屏模式（默认，无显示器服务器也可用）：渲染一帧
#      静态俯视/侧视 PNG 到 --out-dir，便于快速检查。
#
# 用法示例：
#   python app/load_scene.py --scene-dir data/SUN3D/scene_001 \
#       --out-dir results/vis          # 离屏渲染（默认）
#   python app/load_scene.py --scene-dir data/SUN3D/scene_001 --interactive
#
# 依赖：numpy / open3d（纯 CPU，不需要 GPU）
# ============================================================
import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_compat import (make_offscreen_renderer, setup_camera as setup_camera_compat,
                           orbit_eye)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("load_scene")


# ---------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------
def load_frames(scene_dir, pattern="*.npz", frame_limit=None):
    """加载场景目录下的帧数据。

    每个 .npz 文件约定包含：
        point_cloud : (N, 3) float32 相机坐标点云（必选）
        colors      : (N, 3) float32 0~255 颜色（可选）
        rgb         : (N, 3) float32 0~255 颜色（可选，与 colors 二选一）

    返回: [{"path": Path, "xyz": ndarray, "rgb": ndarray|None}, ...]
    """
    files = sorted(glob.glob(str(Path(scene_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"场景目录无匹配帧：{scene_dir}/{pattern}")
    if frame_limit:
        files = files[: frame_limit]
    log.info("加载 %d 帧（%s）", len(files), scene_dir)

    frames = []
    for fp in files:
        data = np.load(fp, allow_pickle=True)
        xyz = data["point_cloud"] if "point_cloud" in data.files else data["xyz"]
        rgb = None
        for key in ("colors", "rgb", "color"):
            if key in data.files:
                rgb = data[key]
                break
        frames.append({"path": Path(fp), "xyz": np.asarray(xyz, dtype=np.float32),
                       "rgb": None if rgb is None else np.asarray(rgb, dtype=np.float32)})
    return frames


# ---------------------------------------------------------------
# Open3D 几何构建
# ---------------------------------------------------------------
def build_pointcloud(xyz, rgb=None, voxel_size=None, color=None):
    """由 numpy 点云构建 open3d PointCloud。

    参数:
        xyz        : (N, 3) 坐标
        rgb        : (N, 3) 0~255 颜色，可选
        voxel_size : 体素下采样边长（米），None 表示不下采样
        color      : 统一颜色 [r, g, b] 0~1（无顶点颜色时使用）
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if rgb is not None:
        colors = np.clip(np.asarray(rgb, dtype=np.float32) / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color is not None:
        n = len(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64), (n, 1)))
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


# ---------------------------------------------------------------
# 渲染（离屏）
# ---------------------------------------------------------------

def render_offscreen(pcds, out_path, width=1280, height=720, fov=60.0,
                     elevation=25.0, azimuth=0.0, dist_scale=1.4):
    """用 OffscreenRenderer 渲染一组点云为单张 PNG（headless 可用）。

    相机：环绕目标中心（所有点云质心），由方位角/俯仰角/距离决定。

    参数:
        pcds       : open3d PointCloud 列表
        out_path   : 输出 PNG 路径
        width/height: 渲染分辨率
        fov        : 垂直视场角（度）
        elevation  : 相机俯仰角（度，>0 从上方俯视）
        azimuth    : 相机方位角（度，绕 Y 轴旋转）
        dist_scale : 相机距离 = 场景半径 × dist_scale
    """
    center, radius = _scene_bounds(pcds)
    dist = radius * dist_scale

    eye = orbit_eye(center, dist, elevation, azimuth)
    renderer = make_offscreen_renderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = [0.7, 0.7, 0.7, 1.0]
    mat.point_size = 2.0
    for i, pcd in enumerate(pcds):
        renderer.scene.add_geometry(f"pcd_{i}", pcd, mat)
    setup_camera_compat(renderer, fov, center, eye, [0.0, 0.0, 1.0])
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    log.info("已渲染: %s (%dx%d)", out_path, width, height)


def _scene_bounds(pcds):
    """合并所有点云，返回 (center, radius)；空点云抛 ValueError。"""
    pts_list = [np.asarray(p.points) for p in pcds if len(p.points) > 0]
    if not pts_list:
        raise ValueError("所有点云均为空，无法计算相机视角")
    pts = np.concatenate(pts_list, axis=0)
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    if radius <= 1e-9:
        radius = 1.0
    return center, radius


# ---------------------------------------------------------------
# 交互展示
# ---------------------------------------------------------------
def visualize_interactive(pcds, window_name="SUN3D Scene"):
    """交互模式：Open3D 视窗展示点云列表。"""
    o3d.visualization.draw_geometries(
        pcds, window_name=window_name, width=1280, height=720)


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="加载 SUN3D 场景多帧点云并展示（交互或离屏渲染 PNG）")
    p.add_argument("--scene-dir", type=str, required=True,
                   help="场景目录（含 *.npz 帧文件）")
    p.add_argument("--pattern", type=str, default="*.npz",
                   help="帧文件通配符（默认 *.npz）")
    p.add_argument("--frame-limit", type=int, default=None,
                   help="最多加载帧数（默认全部）")
    p.add_argument("--voxel-size", type=float, default=0.02,
                   help="展示用体素下采样边长（米，默认 0.02）")
    p.add_argument("--interactive", action="store_true",
                   help="交互模式（有显示器时使用）")
    p.add_argument("--out-dir", type=str, default="results/vis",
                   help="离屏渲染输出目录（默认 results/vis）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()
    frames = load_frames(args.scene_dir, args.pattern, args.frame_limit)
    pcds = [build_pointcloud(f["xyz"], f["rgb"], voxel_size=args.voxel_size)
            for f in frames]

    if args.interactive:
        visualize_interactive(pcds, window_name=Path(args.scene_dir).name)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.scene_dir).name}_scene.png"
    render_offscreen(pcds, out_path, width=args.width, height=args.height)
    log.info("离屏渲染完成：%s", out_path)


if __name__ == "__main__":
    main()
