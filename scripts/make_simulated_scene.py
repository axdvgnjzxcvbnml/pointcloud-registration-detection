#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_simulated_scene.py — 生成 5 帧模拟 SUN3D 场景（任务一：纯 CPU 跑通全流程）

功能
----
    在不依赖真实数据集的情况下，生成一个尽量接近 SUN3D 真实格式的模拟场景：
      - image/frame-000000.color.jpg      RGB 图（z-buffer 光栅化）
      - depth/frame-000000.depth.png      16bit uint16 深度图（单位：0.1mm，/10000 得米）
      - extrinsics/frame-000000.txt       3x4 相机位姿矩阵（cam2world）
      - intrinsics/frame-000000.txt       3x3 相机内参 K
      - clouds/frame-000000.npz           相机坐标点云（app/load_scene 等用）
    场景几何：地板平面 + 3 个球体 + 1 个立方体（保证 FPFH 特征丰富、RANSAC 可配准）。
    相机轨迹：绕场景中心小角度旋转（5 帧，帧号 0/5/10/15/20，帧间约 3°）。

设计要点（对齐真实 SUN3D）
--------------------------
    1. 深度图单位与 depth_scale=10000 一致：depth_px = 深度(米) * 10000，uint16。
    2. 位姿为 3x4 行优先（np.loadtxt 可直接读），与 compute_pose_gt.py 约定一致。
    3. 帧名含 'frame-NNNNNN'，匹配 sample_frame_pairs.py 的 FRAME_RE。
    4. 相机位姿为 cam2world：世界点 P_w 与相机点 P_c 满足 P_c = inv(Pose) @ P_w。

用法
----
    python scripts/make_simulated_scene.py \
        --scene_dir data/SUN3D/sim_scene_001 \
        --num_frames 5 --frame_step 5 --seed 42
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 640x480 内参（与 preprocess/defaults.py 的 DEFAULT_K 一致，真实数据需替换）
DEFAULT_K = np.array([[528.0, 0.0, 319.5],
                      [0.0, 528.0, 239.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
DEPTH_SCALE = 10000.0   # 深度图单位换算（SUN RGB-D 口径）
DEPTH_TRUNC = 8.0       # 最大有效深度（米）


# ---------------------------------------------------------------
# 场景几何
# ---------------------------------------------------------------
def build_world_points(seed=42):
    """构建世界坐标系下的场景点云（含颜色）。

    Returns:
        pts (N,3) float64, colors (N,3) float64 0~1
    """
    rng = np.random.default_rng(seed)
    parts, cols = [], []

    # 1) 地板平面 z=0，网格 0.04m（约 100x100 = 1 万点）
    xs = np.arange(-2.0, 2.0, 0.04)
    ys = np.arange(-2.0, 2.0, 0.04)
    gx, gy = np.meshgrid(xs, ys)
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros_like(gx.ravel())], axis=-1)
    parts.append(floor)
    cols.append(np.tile([0.72, 0.70, 0.66], (len(floor), 1)))

    # 2) 球体（曲率变化大，FPFH 特征丰富）
    sphere_cfg = [
        ((0.00, 0.00, 0.50), 0.38, [0.20, 0.45, 0.90], 2400),
        ((0.95, 0.60, 0.40), 0.32, [0.90, 0.40, 0.20], 1800),
        ((-0.90, -0.55, 0.32), 0.26, [0.20, 0.80, 0.40], 1400),
    ]
    for center, r, color, n in sphere_cfg:
        phi = rng.uniform(0.0, np.pi, n)
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        sp = np.stack([np.sin(phi) * np.cos(theta),
                       np.sin(phi) * np.sin(theta),
                       np.cos(phi)], axis=-1)
        parts.append(np.asarray(center, dtype=np.float64) + r * sp)
        cols.append(np.tile(color, (n, 1)))

    # 3) 立方体（平面 + 锐利边，增加特征区分度）
    c, s = np.array([-0.35, 0.75, 0.30]), 0.55
    half = s / 2.0
    step = 0.05
    cube_pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            grid = np.arange(-half, half + 1e-9, step)
            g1, g2 = np.meshgrid(grid, grid)
            block = np.zeros((len(g1.ravel()), 3))
            block[:, axis] = sign * half
            block[:, (axis + 1) % 3] = g1.ravel()
            block[:, (axis + 2) % 3] = g2.ravel()
            cube_pts.append(block)
    cube = np.concatenate(cube_pts, axis=0) + c
    parts.append(cube)
    cols.append(np.tile([0.95, 0.75, 0.10], (len(cube), 1)))

    pts = np.concatenate(parts, axis=0).astype(np.float64)
    colors = np.concatenate(cols, axis=0).astype(np.float64)
    logger.info("世界点云: %d 点", len(pts))
    return pts, colors


# ---------------------------------------------------------------
# 相机位姿
# ---------------------------------------------------------------
def look_at_pose(cam_pos, look_at, up=(0.0, 0.0, 1.0)):
    """由相机位置与注视点构造 3x4 cam2world 位姿。

    Returns:
        (3, 4) 矩阵：世界点 P_w = R @ P_c + t（P_c 为相机坐标点）。
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up_cam = np.cross(forward, right)

    R = np.stack([right, up_cam, forward], axis=1)  # 列为相机 x/y/z 轴
    pose = np.hstack([R, cam_pos.reshape(3, 1)])
    return pose


def build_trajectory(num_frames=5, frame_step=5, seed=42):
    """生成沿圆弧小幅旋转的相机轨迹（5 帧，帧号 0/5/10/15/20）。

    Returns:
        frame_ids: list[int]
        poses: dict {frame_id: (3,4) cam2world}
    """
    rng = np.random.default_rng(seed)
    center = np.array([0.0, 0.0, 0.7])
    radius, height = 2.3, 1.5
    # 方位角：-6° ~ +6°，帧间 3°（保证帧间重叠大、RANSAC 易成功）
    angles = np.linspace(-6.0, 6.0, num_frames)
    frame_ids = [i * frame_step for i in range(num_frames)]

    poses = {}
    for fid, deg in zip(frame_ids, angles):
        theta = np.deg2rad(deg)
        cam_pos = np.array([center[0] + radius * np.sin(theta),
                            center[1] + radius * np.cos(theta),
                            height])
        look_at = np.array([center[0], center[1], center[2] - 0.1])
        # 加微小抖动，避免完全对称（更接近真实数据）
        cam_pos += rng.normal(0.0, 0.02, 3)
        poses[fid] = look_at_pose(cam_pos, look_at)
    return frame_ids, poses


# ---------------------------------------------------------------
# 深度 / RGB 渲染（z-buffer）
# ---------------------------------------------------------------
def render_frame(world_pts, world_colors, pose, K, h=480, w=640):
    """把世界点云投影到一帧，输出 16bit 深度图与 RGB 图。

    Returns:
        depth_uint16 (h,w) uint16, rgb (h,w,3) uint8
    """
    R, t = pose[:, :3], pose[:, 3]
    Pc = (world_pts - t) @ R                # 相机坐标 (N,3)
    z = Pc[:, 2]

    valid = (z > 0.05) & (z < DEPTH_TRUNC)
    u = K[0, 0] * Pc[:, 0] / np.where(valid, z, 1.0) + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / np.where(valid, z, 1.0) + K[1, 2]
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

    ui = np.clip(np.floor(u[valid]).astype(np.int32), 0, w - 1)
    vi = np.clip(np.floor(v[valid]).astype(np.int32), 0, h - 1)
    z_valid = z[valid]

    depth_img = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(depth_img, (vi, ui), z_valid)      # z-buffer 取最近
    mask = np.isfinite(depth_img)
    depth_uint16 = np.clip(np.where(mask, depth_img, 0.0) * DEPTH_SCALE,
                           0, 65535).astype(np.uint16)

    rgb_img = np.full((h, w, 3), 245, dtype=np.uint8)
    # 最近点的颜色：按 z 排序后逐个写入，后写覆盖=更近（近的排在后面）
    order = np.argsort(z_valid)                      # 近 -> 远
    rgb_img[vi[order], ui[order]] = (world_colors[valid][order] * 255.0).astype(np.uint8)
    return depth_uint16, rgb_img


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def generate_scene(scene_dir, num_frames=5, frame_step=5, seed=42,
                   img_size=(640, 480)):
    """生成模拟场景目录与帧文件。"""
    scene_dir = Path(scene_dir)
    for sub in ("image", "depth", "extrinsics", "intrinsics", "clouds"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)

    w, h = img_size
    K = DEFAULT_K.copy()
    world_pts, world_colors = build_world_points(seed)
    frame_ids, poses = build_trajectory(num_frames, frame_step, seed)

    manifest = {"scene": scene_dir.name, "num_frames": num_frames,
                "frame_step": frame_step, "seed": seed,
                "K": K.tolist(), "frames": []}

    for fid in frame_ids:
        pose = poses[fid]
        depth_uint16, rgb = render_frame(world_pts, world_colors, pose, K, h, w)

        # SUN3D 风格帧名
        tag = f"frame-{fid:06d}"
        rgb_path = scene_dir / "image" / f"{tag}.color.jpg"
        depth_path = scene_dir / "depth" / f"{tag}.depth.png"
        ext_path = scene_dir / "extrinsics" / f"{tag}.txt"
        int_path = scene_dir / "intrinsics" / f"{tag}.txt"
        cloud_path = scene_dir / "clouds" / f"{tag}.npz"

        Image.fromarray(rgb).save(str(rgb_path), quality=95)
        Image.fromarray(depth_uint16, mode="I;16").save(str(depth_path))
        np.savetxt(str(ext_path), pose, fmt="%.9f")
        np.savetxt(str(int_path), K, fmt="%.6f")

        # 相机坐标点云（含颜色）：等价于 depth_to_pointcloud 输出（供 app 使用）
        R, t = pose[:, :3], pose[:, 3]
        Pc = (world_pts - t) @ R
        z = Pc[:, 2]
        valid = (z > 0.05) & (z < DEPTH_TRUNC)
        u = K[0, 0] * Pc[:, 0] / np.where(valid, z, 1.0) + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / np.where(valid, z, 1.0) + K[1, 2]
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        xyz = Pc[valid].astype(np.float32)
        col = (world_colors[valid] * 255.0).astype(np.uint8)
        np.savez_compressed(cloud_path, point_cloud=xyz, colors=col,
                            K=K, pose=pose)

        manifest["frames"].append({
            "frame_id": fid,
            "n_points": int(valid.sum()),
            "rgb_path": str(rgb_path), "depth_path": str(depth_path),
            "ext_path": str(ext_path), "int_path": str(int_path),
            "cloud_path": str(cloud_path),
        })
        logger.info("帧 %s: 可见点 %d", tag, int(valid.sum()))

    with open(scene_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 帧对统计（与 sample_frame_pairs 的间隔语义一致）
    intervals = [5, 10, 30]
    n_pairs = sum(1 for itv in intervals
                  for a in frame_ids if (a + itv) in set(frame_ids))
    logger.info("场景 %s 生成完成：%d 帧，候选帧对 %d 对（间隔 %s）",
                scene_dir, len(frame_ids), n_pairs, intervals)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="生成 5 帧模拟 SUN3D 场景")
    parser.add_argument("--scene_dir", default="data/SUN3D/sim_scene_001",
                        help="场景输出目录")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--frame_step", type=int, default=5,
                        help="帧号步长（对齐间隔 5/10/30 采样）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate_scene(args.scene_dir, args.num_frames, args.frame_step, args.seed)


if __name__ == "__main__":
    main()
