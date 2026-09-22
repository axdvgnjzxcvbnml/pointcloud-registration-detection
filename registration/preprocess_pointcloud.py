#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_pointcloud.py — 点云预处理（第三批交付）

功能
----
    配准前的统一预处理流水线（纯 CPU）：
      1. 统计离群点去除（StatisticalOutlierRemoval）
      2. 体素下采样（voxel_down_sample）
      3. 法向量估计（开放接口：优先 open3d；提供 PCA 手工实现）

    PCA 法线估计
    -----------
    对每个点取 k 近邻，拟合局部平面：
      - 法线 = 最小特征值对应特征向量
      - 朝向统一指向视点（相机位置，默认原点）
    复杂度 O(N·k)，适合中等规模点云；大规模点云建议换 open3d。

用法
----
    python registration/preprocess_pointcloud.py \
        --input a.ply --output a_clean.ply --voxel 0.02
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def pca_normals(points, k=20, view_point=(0.0, 0.0, 0.0)):
    """PCA 法线估计（纯 numpy，无 open3d 依赖）。

    Args:
        points: (N, 3) 点云。
        k: 近邻数。
        view_point: 视点（法线朝向该点）。
    Returns:
        normals: (N, 3) 单位法线。
    """
    from scipy.spatial import KDTree

    tree = KDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / max(len(neighbors) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]                       # 最小特征值对应的特征向量
        if n @ (np.asarray(view_point) - points[i]) < 0:
            n = -n
        normals[i] = n / max(np.linalg.norm(n), 1e-12)
    return normals


def preprocess_pointcloud(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0,
                          radius_normal=0.1, use_open3d_normals=True):
    """点云预处理主函数（open3d PointCloud 接口）。

    Args:
        pcd: open3d.geometry.PointCloud。
        voxel_size: 体素下采样尺寸（米）；None/0 跳过。
        nb_neighbors: 统计去噪近邻数。
        std_ratio: 统计去噪标准差倍数。
        radius_normal: 法线估计搜索半径（open3d 用）。
        use_open3d_normals: True=open3d 估计法线；False=PCA 手工实现。
    Returns:
        处理后的 open3d PointCloud。
    """
    import open3d as o3d

    work = o3d.geometry.PointCloud(pcd)
    # 1) 统计离群点去除
    work, _ = work.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                              std_ratio=std_ratio)
    # 2) 体素下采样
    if voxel_size and voxel_size > 0:
        work = work.voxel_down_sample(voxel_size)
    # 3) 法向量
    if use_open3d_normals:
        work.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=30))
        work.orient_normals_towards_camera_location()
    else:
        normals = pca_normals(np.asarray(work.points), k=min(20, len(work.points)))
        work.normals = o3d.utility.Vector3dVector(normals)
    return work


def preprocess_from_files(input_path, output_path, voxel_size=0.02,
                          nb_neighbors=20, std_ratio=2.0, radius_normal=0.1):
    """文件级入口：读点云 → 预处理 → 保存。"""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(input_path))
    out = preprocess_pointcloud(pcd, voxel_size, nb_neighbors, std_ratio,
                                radius_normal)
    o3d.io.write_point_cloud(str(output_path), out)
    logger.info("%d 点 -> %d 点 -> %s", len(pcd.points), len(out.points), output_path)


def main():
    parser = argparse.ArgumentParser(description="点云预处理（去噪+下采样+法线）")
    parser.add_argument("--input", required=True, help="输入点云 .ply/.npz")
    parser.add_argument("--output", required=True, help="输出点云 .ply")
    parser.add_argument("--voxel", type=float, default=0.02, help="体素尺寸（米）")
    parser.add_argument("--nb_neighbors", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--radius_normal", type=float, default=0.1)
    parser.add_argument("--pca_normals", action="store_true",
                        help="用 PCA 手工法线（默认 open3d）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import numpy as np

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".npz":
        data = np.load(input_path)
        points = data["point_cloud"] if "point_cloud" in data.files else data["xyz"]
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tmp = input_path.with_suffix("_tmp.ply")
        o3d.io.write_point_cloud(str(tmp), pcd)
        preprocess_from_files(tmp, args.output, args.voxel,
                              args.nb_neighbors, args.std_ratio,
                              args.radius_normal)
        tmp.unlink()
    else:
        preprocess_from_files(args.input, args.output, args.voxel,
                              args.nb_neighbors, args.std_ratio,
                              args.radius_normal)


if __name__ == "__main__":
    main()
