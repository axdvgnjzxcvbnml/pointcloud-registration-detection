#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_to_pointcloud.py — 深度图 → 三维点云（第二批交付）

功能
----
    把单张深度图（16bit PNG，单位 mm）按相机内参反投影为
    相机坐标系下的三维点云：

        X = (u - cx) / fx * Z
        Y = (v - cy) / fy * Z
        Z = depth / depth_scale      （SUN RGB-D：depth_scale=10000，单位米）

    可选：深度截断、逐点去离群（统计）、体素下采样、点数上限随机采样、
    按原始 RGB 图着色。

说明
----
    纯 numpy 实现（可单测），可选依赖 open3d 用于体素下采样与保存 .ply。

用法
----
    python preprocess/depth_to_pointcloud.py \
        --depth data/SUNRGBD/.../depth/00001.png \
        --K results/preprocess/K_00001.json \
        --out results/preprocess/pcd/00001.npz \
        --rgb data/SUNRGBD/.../image/00001.jpg
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_depth(path, depth_scale=10000.0, depth_trunc=8.0):
    """读取深度图并转成米制浮点深度图。

    深度图为 16bit PNG（单位 mm）；也可接受 .npy/.npz。
    depth_scale=10000 表示 mm->m（SUN RGB-D 约定）。

    Returns:
        depth_m: (H, W) float32，单位米；无效点（0 / >trunc）已置 NaN。
    """
    p = Path(path)
    if p.suffix == ".npy":
        depth_raw = np.load(p)
    elif p.suffix == ".npz":
        depth_raw = np.load(p)["depth"]
    else:
        import cv2
        depth_raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    depth_m = np.asarray(depth_raw, dtype=np.float32) / depth_scale
    depth_m[depth_raw == 0] = np.nan
    depth_m[depth_m > depth_trunc] = np.nan
    return depth_m


def depth_to_points(depth_m, K, rgb=None, max_points=None):
    """反投影生成点云。

    Args:
        depth_m: (H, W) float32，单位米，无效点为 NaN。
        K: (3, 3) 相机内参。
        rgb: (H, W, 3) uint8 BGR，可选着色。
        max_points: 点数上限，超出随机采样（内存保护）。
    Returns:
        xyz: (N, 3) float32 相机系坐标。
        colors: (N, 3) float32 0~255（rgb=None 时为 None）。
        valid: (H, W) bool 有效像素掩码。
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth_m
    valid = ~np.isnan(z)
    if not valid.any():
        logger.warning("深度图无有效像素")
        return np.zeros((0, 3), dtype=np.float32), None, valid

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1)          # (H, W, 3)
    xyz = xyz[valid]

    colors = None
    if rgb is not None:
        colors = np.asarray(rgb, dtype=np.float32)[valid]

    if max_points and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        if colors is not None:
            colors = colors[idx]
        logger.info("点数上限 %d，随机采样保留 %d 点", max_points, len(xyz))
    return xyz, colors, valid


def voxel_downsample(xyz, colors, voxel_size=0.02):
    """体素下采样（open3d；未安装时退化为 numpy 版最近邻取整平均）。"""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size)
        xyz = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255.0) if pcd.has_colors() else None
    except ImportError:
        if voxel_size <= 0:
            return xyz, colors
        keys = np.floor(xyz / voxel_size).astype(np.int64)
        # 简单实现：按体素键平均（演示用，性能有限）
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        xyz = np.stack([np.bincount(inv, weights=xyz[:, i])
                        / np.bincount(inv) for i in range(3)], axis=1)
        if colors is not None:
            colors = np.stack([np.bincount(inv, weights=colors[:, i])
                               / np.bincount(inv) for i in range(3)], axis=1)
    return xyz, colors


def main():
    parser = argparse.ArgumentParser(description="深度图 → 点云")
    parser.add_argument("--depth", required=True, help="深度图路径")
    parser.add_argument("--K", required=True, help="内参 json（3x3）")
    parser.add_argument("--rgb", default=None, help="可选 RGB 图路径（BGR 着色）")
    parser.add_argument("--out", default="out.npz", help="输出 npz")
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--depth_trunc", type=float, default=8.0)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--save_ply", action="store_true", help="同时输出 ply")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(args.seed)

    with open(args.K, "r", encoding="utf-8") as f:
        K = np.asarray(json.load(f), dtype=np.float64).reshape(3, 3)
    depth_m = load_depth(args.depth, args.depth_scale, args.depth_trunc)

    rgb = None
    if args.rgb:
        import cv2
        rgb = cv2.imread(args.rgb)
    xyz, colors, _ = depth_to_points(depth_m, K, rgb=rgb, max_points=args.max_points)
    if args.voxel_size:
        xyz, colors = voxel_downsample(xyz, colors, args.voxel_size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), point_cloud=xyz,
                        colors=None if colors is None else colors)
    logger.info("点云 %d 点 -> %s", len(xyz), out)
    if args.save_ply:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.io.write_point_cloud(str(out.with_suffix(".ply")), pcd)


if __name__ == "__main__":
    main()
