#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_to_pointcloud.py — 深度图 → 三维点云（第二批交付）

功能
----
    用相机内参把深度图反投影为三维点云，可选叠加 RGB 颜色，
    输出 numpy 数组（.npy/.npz）或 .ply 文件（Open3D 写入）。

反投影公式（针孔相机模型）
--------------------------
    z = depth[u, v] / depth_scale            # 深度原始单位转米
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

输入
----
    --depth 支持三种来源：
      - .npz：内部含 depth 与 K（parse_sunrgbd.py 的输出）
      - .mat：SUN RGB-D 官方深度（变量 'depth'），需另给 --K
      - .png：16bit 深度图，需另给 --K

用法
----
    python preprocess/depth_to_pointcloud.py \
        --depth results/preprocess/frames/scene_000_frame-000000.npz \
        --out_path results/preprocess/pcd/scene_000.ply \
        --out_format ply --depth_scale 10000.0 --depth_trunc 8.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 占位内参集中维护在 preprocess/defaults.py（真实数据请用元数据替换）
try:
    from defaults import DEFAULT_K
except ImportError:  # 作为 preprocess 包被导入时
    from preprocess.defaults import DEFAULT_K


def depth_to_pointcloud(depth, K, depth_scale=10000.0, depth_trunc=8.0):
    """深度图反投影为点云。

    Args:
        depth: (H, W) float32，原始单位（mm，除以 depth_scale 得米）。
        K: (3, 3) 相机内参。
        depth_scale: 深度单位换算比例（SUN RGB-D 为 10000）。
        depth_trunc: 最大有效深度（米），超出截断（过滤异常值）。
    Returns:
        (N, 3) float32 点云，米制坐标，按行优先排列。
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 像素网格（u: 列索引, v: 行索引）
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth / depth_scale
    valid = (z > 0) & (z < depth_trunc)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)          # (H, W, 3)
    return pts[valid].astype(np.float32)


def colorize_pointcloud(points, rgb, depth, K,
                        depth_scale=10000.0, depth_trunc=8.0):
    """为点云附上对应像素的 RGB 颜色，返回 (N, 6) [x,y,z,r,g,b]。

    rgb 归一化到 [0, 1]。越界像素颜色置 0。
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = np.round(fx * x / z + cx).astype(np.int32)
    v = np.round(fy * y / z + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    colors = np.zeros((len(points), 3), dtype=np.float32)
    if rgb is not None:
        colors[in_bounds] = rgb[v[in_bounds], u[in_bounds]].astype(np.float32) / 255.0
    return np.concatenate([points, colors], axis=1)


def save_pointcloud(path, points, out_format="ply"):
    """保存点云，支持 ply / npy / npz 三种格式。

    ply 使用 Open3D 写入；points 可为 (N,3) 或 (N,6)（含颜色）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if out_format == "npy":
        np.save(str(path.with_suffix(".npy")), points)
    elif out_format == "npz":
        # 同时写 points 与 point_cloud 两个键：points 向后兼容，
        # point_cloud 供 app/（load_scene/compare_registration/export_demo）读取。
        np.savez_compressed(str(path.with_suffix(".npz")),
                            points=points, point_cloud=points)
    elif out_format == "ply":
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        if points.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6].astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd)
    else:
        raise ValueError(f"未知输出格式: {out_format}（支持 ply/npy/npz）")


def load_depth_input(depth_path, K_path=None):
    """加载深度图与内参，返回 (depth, K)。

    - .npz：读 'depth' 与 'K'（K 缺失时返回 None）
    - .mat：读变量 'depth'，K 需由 --K 提供
    - 图像：读 16bit 深度，K 需由 --K 提供
    """
    p = Path(depth_path)
    if p.suffix.lower() == ".npz":
        data = np.load(str(p))
        depth = np.asarray(data["depth"], dtype=np.float32)
        K = np.asarray(data["K"], dtype=np.float64) if "K" in data else None
        return depth, K
    if p.suffix.lower() == ".mat":
        from scipy.io import loadmat
        depth = loadmat(str(p))["depth"].astype(np.float32)
        return depth, None
    # 图像（16bit png 等）
    from PIL import Image
    with Image.open(p) as img:
        depth = np.asarray(img, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth, None


def main():
    parser = argparse.ArgumentParser(description="深度图转三维点云")
    parser.add_argument("--depth", required=True,
                        help="深度图路径（.npz/.mat/.png）")
    parser.add_argument("--rgb", default=None,
                        help="RGB 图路径（可选，叠加颜色）")
    parser.add_argument("--K", default=None,
                        help="内参文件路径（3x3 txt，.npz 输入时可不给）")
    parser.add_argument("--out_path", default="results/preprocess/pcd/frame",
                        help="输出文件路径（不含扩展名时按 --out_format 补全）")
    parser.add_argument("--out_format", choices=["ply", "npy", "npz"],
                        default="ply", help="输出格式")
    parser.add_argument("--depth_scale", type=float, default=10000.0,
                        help="深度单位换算比例（SUN RGB-D: 10000）")
    parser.add_argument("--depth_trunc", type=float, default=8.0,
                        help="最大有效深度（米）")
    parser.add_argument("--max_points", type=int, default=None,
                        help="点数上限，超出随机采样（内存保护）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    depth, K = load_depth_input(args.depth, args.K)
    if K is None:
        if args.K:
            K = np.loadtxt(args.K).reshape(3, 3).astype(np.float64)
        else:
            logger.warning("未提供内参，使用默认 K（占位）")
            K = DEFAULT_K.copy()

    points = depth_to_pointcloud(depth, K,
                                 depth_scale=args.depth_scale,
                                 depth_trunc=args.depth_trunc)
    if args.rgb:
        from PIL import Image
        with Image.open(args.rgb) as img:
            rgb = np.asarray(img.convert("RGB"))
        points = colorize_pointcloud(points, rgb, depth, K,
                                     depth_scale=args.depth_scale,
                                     depth_trunc=args.depth_trunc)
    if args.max_points and len(points) > args.max_points:
        idx = np.random.default_rng(0).choice(len(points),
                                              size=args.max_points, replace=False)
        points = points[idx]

    out_path = Path(args.out_path)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(f".{args.out_format}")
    save_pointcloud(out_path, points, out_format=args.out_format)
    logger.info("点云: %d 点 -> %s", len(points), out_path)


if __name__ == "__main__":
    main()
