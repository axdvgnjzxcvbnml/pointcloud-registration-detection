#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_pointcloud.py — 点云预处理（第三批交付，纯 CPU）

功能
----
    基于 Open3D 实现点云标准预处理流水线（顺序固定）：
      1. 统计离群点去除（KNN k=20, 标准差倍数 std=2.0）
      2. 体素下采样（voxel_size=0.02m）
      3. 法向量估计（搜索半径 normal_radius=0.1m）

说明
----
    本模块只依赖 numpy / open3d，不依赖 GPU。
    参数默认值与 configs/default.yaml 的 registration 区块一致。

用法
----
    python registration/preprocess_pointcloud.py \
        --input results/preprocess/pcd/scene_a.ply \
        --output results/registration/scene_a_clean.ply \
        --voxel_size 0.02 --nb_neighbors 20 --std_ratio 2.0 --normal_radius 0.1
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """统计离群点去除（Statistical Outlier Removal）。

    Args:
        pcd: open3d.geometry.PointCloud。
        nb_neighbors: 每个点的近邻数 k（默认 20）。
        std_ratio: 距离均值超出 std_ratio×标准差则视为离群点（默认 2.0）。
    Returns:
        去噪后的 open3d PointCloud。
    """
    import open3d as o3d

    cleaned, _ = o3d.geometry.PointCloud.remove_statistical_outlier(
        pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned


def voxel_downsample(pcd, voxel_size=0.02):
    """体素下采样。

    Args:
        pcd: open3d.geometry.PointCloud。
        voxel_size: 体素边长（米），默认 0.02m。
    Returns:
        下采样后的 open3d PointCloud。
    """
    import open3d as o3d

    if voxel_size <= 0:
        raise ValueError(f"voxel_size 必须 > 0，当前 {voxel_size}")
    return o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)


def estimate_normals(pcd, normal_radius=0.1, max_nn=30):
    """法向量估计（在 pcd 上原地计算并返回）。

    Args:
        pcd: open3d.geometry.PointCloud。
        normal_radius: 法向量估计的邻域搜索半径（米），默认 0.1m。
        max_nn: 邻域最大点数上限。
    Returns:
        同一点云（含法向量）。
    """
    import open3d as o3d

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=max_nn))
    # 统一法向朝向（指向视点/原点），避免 ICP 点到平面度量符号抖动
    pcd.orient_normals_towards_camera_location(
        camera_location=pcd.get_center())
    return pcd


def preprocess_pointcloud(pcd, voxel_size=0.02, nb_neighbors=20,
                          std_ratio=2.0, normal_radius=0.1):
    """标准预处理流水线：去离群 → 体素降采样 → 法向量估计。

    Args:
        pcd: 原始 open3d.geometry.PointCloud。
        其余参数含义见各子函数。
    Returns:
        处理后的 open3d PointCloud（含法向量）。
    """
    logger.info("输入点数: %d", len(pcd.points))
    pcd = statistical_outlier_removal(pcd, nb_neighbors, std_ratio)
    logger.info("去离群后: %d 点", len(pcd.points))
    pcd = voxel_downsample(pcd, voxel_size)
    logger.info("体素降采样(v=%.3fm)后: %d 点", voxel_size, len(pcd.points))
    pcd = estimate_normals(pcd, normal_radius)
    return pcd


def main():
    parser = argparse.ArgumentParser(description="点云预处理（去噪/降采样/法向量）")
    parser.add_argument("--input", required=True, help="输入点云路径（.ply/.pcd/.xyz）")
    parser.add_argument("--output", default="results/registration/cleaned.ply",
                        help="输出点云路径（.ply）")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="体素下采样尺寸（米）")
    parser.add_argument("--nb_neighbors", type=int, default=20,
                        help="统计去噪近邻数 k")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                        help="统计去噪标准差倍数")
    parser.add_argument("--normal_radius", type=float, default=0.1,
                        help="法向量估计半径（米）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import open3d as o3d
    from pathlib import Path

    src = o3d.io.read_point_cloud(args.input)
    if src.is_empty():
        raise RuntimeError(f"点云为空或读取失败: {args.input}")

    pcd = preprocess_pointcloud(
        src, voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio,
        normal_radius=args.normal_radius)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    logger.info("预处理完成: %d 点 -> %s", len(pcd.points), out_path)


if __name__ == "__main__":
    main()
