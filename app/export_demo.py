# ============================================================
# app/export_demo.py —— 将可视化结果导出为 GIF 动图与视频
# ------------------------------------------------------------
# 功能：
#   1. 加载场景多帧点云（复用 load_scene.load_frames/build_pointcloud），
#      可选叠加检测框（复用 overlay_detection.box_lineset）；
#   2. 相机环绕动画：围绕场景中心旋转俯拍，逐帧离屏渲染
#      （OffscreenRenderer，headless 服务器可用）；
#   3. 导出：
#        GIF —— PIL（Pillow）逐帧写入；
#        MP4 —— imageio + imageio-ffmpeg（libx264）。
#
# 用法示例：
#   python app/export_demo.py --scene-dir data/SUN3D/scene_001 \
#       --out-dir results/demo --n-frames 60 --fps 15 --gif --video
#   python app/export_demo.py --scene-dir data/SUN3D/scene_001 \
#       --detections results/det/frame_000_pred.json --video
#
# 依赖：numpy / open3d / Pillow / imageio / imageio-ffmpeg（纯 CPU）
# ============================================================
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_compat import (make_offscreen_renderer, setup_camera as setup_camera_compat,
                           scene_center_radius)
from load_scene import load_frames, build_pointcloud   # 同目录复用

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("export_demo")


# ---------------------------------------------------------------
# 相机环绕
# ---------------------------------------------------------------
def make_orbit_camera(center, radius, n_frames=60, elevation=20.0):
    """生成环绕动画的相机序列（纯 numpy，可单测）。

    每帧 eye 在水平圆周上，保持俯仰角 elevation 与半径 radius 不变，
    从方位角 0° 均匀旋转 360°。
    返回: {"center": ndarray(3,), "eyes": (n_frames,3), "up": ndarray(3,)}
    """
    elev = np.deg2rad(elevation)
    angles = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    r_h = radius * np.cos(elev)
    eyes = np.stack([
        center[0] + r_h * np.sin(angles),
        center[1] + r_h * np.cos(angles),
        np.full(n_frames, center[2] + radius * np.sin(elev)),
    ], axis=1)
    return {"center": np.asarray(center, dtype=np.float64),
            "eyes": eyes, "up": np.array([0.0, 0.0, 1.0])}


# ---------------------------------------------------------------
# 逐帧离屏渲染
# ---------------------------------------------------------------

def render_orbit_frames(geometries, camera, n_frames=60, width=1280,
                        height=720, fov=60.0, point_size=2.0):
    """按相机序列逐帧渲染，返回 PIL RGB 图像列表（全程内存，不写临时文件）。"""
    renderer = make_offscreen_renderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = point_size
    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    for i, g in enumerate(geometries):
        mat = mat_pcd if isinstance(g, o3d.geometry.PointCloud) else mat_line
        renderer.scene.add_geometry(f"g_{i}", g, mat)

    frames = []
    center, up = camera["center"], camera["up"]
    for eye in camera["eyes"]:
        setup_camera_compat(renderer, fov, center, eye, up)
        img = renderer.render_to_image()
        # render_to_image -> numpy -> PIL，全程内存（避免逐帧写盘残留临时文件）
        frames.append(Image.fromarray(np.asarray(img)).convert("RGB"))
    return frames


# ---------------------------------------------------------------
# 导出 GIF / MP4
# ---------------------------------------------------------------
def write_gif(frames, out_path, fps=15.0):
    """PIL 写 GIF 动图（循环播放）。"""
    if not frames:
        raise ValueError("没有可写入的帧（frames 为空），无法导出 GIF")
    duration = int(1000.0 / fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)
    log.info("GIF 已导出：%s（%d 帧 @ %g fps）", out_path, len(frames), fps)


def write_video(frames, out_path, fps=15.0):
    """imageio + imageio-ffmpeg 写 MP4（libx264）。"""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise RuntimeError("缺少 imageio，请安装：pip install imageio imageio-ffmpeg")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()
    log.info("MP4 已导出：%s（%d 帧 @ %g fps）", out_path, len(frames), fps)


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="将点云/检测可视化导出为 GIF 动图与 MP4 视频（相机环绕动画）")
    p.add_argument("--scene-dir", type=str, required=True,
                   help="场景目录（含 *.npz 帧文件）")
    p.add_argument("--pattern", type=str, default="*.npz")
    p.add_argument("--frame-limit", type=int, default=20,
                   help="最多加载帧数（默认 20，控制体积）")
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--detections", type=str, default=None,
                   help="可选：检测结果 JSON，叠加 3D 框（见 overlay_detection 格式）")
    p.add_argument("--out-dir", type=str, default="results/demo")
    p.add_argument("--n-frames", type=int, default=60, help="动画总帧数")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--elevation", type=float, default=20.0,
                   help="相机俯仰角（度）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--gif", action="store_true", help="导出 GIF")
    p.add_argument("--video", action="store_true", help="导出 MP4")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.gif and not args.video:
        raise SystemExit("请至少指定 --gif 或 --video 之一")

    frames_in = load_frames(args.scene_dir, args.pattern, args.frame_limit)
    pcds = [build_pointcloud(f["xyz"], f["rgb"], voxel_size=args.voxel_size)
            for f in frames_in]

    # 可选：叠加检测框（用检测 JSON 的第一帧或逐帧——骨架按全场景复用同一框集）
    geometries = list(pcds)
    if args.detections:
        from overlay_detection import load_detections, box_lineset, class_color
        dets = load_detections(args.detections)
        for d in dets:
            if d["score"] >= 0.5:
                geometries.append(box_lineset(d["box7"], color=class_color(d["label"])))
        log.info("叠加 %d 个检测框", len(dets))

    # 场景中心与半径（空点云时 scene_center_radius 抛 ValueError）
    center, radius = scene_center_radius(pcds)

    camera = make_orbit_camera(center, radius * 1.5, args.n_frames,
                               elevation=args.elevation)
    log.info("开始逐帧渲染（%d 帧）...", args.n_frames)
    frames = render_orbit_frames(geometries, camera, n_frames=args.n_frames,
                                 width=args.width, height=args.height)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.scene_dir).name
    if args.gif:
        write_gif(frames, out_dir / f"{stem}_demo.gif", fps=args.fps)
    if args.video:
        write_video(frames, out_dir / f"{stem}_demo.mp4", fps=args.fps)


if __name__ == "__main__":
    main()
