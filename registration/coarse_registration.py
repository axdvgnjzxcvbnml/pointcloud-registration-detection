#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_registration.py — FPFH + RANSAC 粗配准（第三批交付）

功能
----
    基于 FPFH 特征匹配的全局粗配准：
      - 计算 FPFH 特征（0.25m 半径）
      - RANSAC（100k 迭代，距离阈值 0.03m = 1.5 × voxel）
      - 返回 4×4 初始变换 T_src2dst 与 fitness

说明
----
    纯 CPU（open3d）；输入点云应已预处理（含法向量）。

用法
----
    python registration/coarse_registration.py \
        --src a.ply --dst b.ply \
        --out results/reg/T_coarse.npy
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_fpfh(pcd, radius=0.25, max_nn=100):
    """计算 FPFH 特征（需先有法向量）。"""
    import open3d as o3d

    if not pcd.has_normals():
        raise ValueError("点云无法向量，请先执行 preprocess_pointcloud")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return fpfh


def coarse_registration(src, dst, voxel_size=0.02,
                        fpfh_radius=0.25, ransac_max_iteration=100000,
                        ransac_distance_threshold=None, ransac_confidence=0.999,
                        ransac_max_validation=1000):
    """FPFH + RANSAC 粗配准。

    Args:
        src: 源点云（open3d，已预处理）。
        dst: 目标点云（open3d，已预处理）。
        voxel_size: 体素尺寸（用于默认距离阈值 = 1.5×voxel）。
        ransac_distance_threshold: RANSAC 距离阈值，默认 1.5×voxel。
    Returns:
        (T_src2dst (4,4), result_info dict)。
    """
    import open3d as o3d

    if ransac_distance_threshold is None:
        ransac_distance_threshold = 1.5 * voxel_size

    src_fpfh = compute_fpfh(src, fpfh_radius)
    dst_fpfh = compute_fpfh(dst, fpfh_radius)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_fpfh, dst_fpfh, True,
        ransac_distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [ransac_max_validation],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            ransac_max_iteration, ransac_confidence))
    T = result.transformation
    info = {"fitness": result.fitness, "rmse": result.inlier_rmse,
            "threshold": ransac_distance_threshold}
    logger.info("粗配准：fitness=%.4f rmse=%.4f", info["fitness"], info["rmse"])
    return T, info


def main():
    parser = argparse.ArgumentParser(description="FPFH+RANSAC 粗配准")
    parser.add_argument("--src", required=True, help="源点云 .ply")
    parser.add_argument("--dst", required=True, help="目标点云 .ply")
    parser.add_argument("--out", default="results/reg/T_coarse.npy",
                        help="输出 4x4 变换")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--fpfh_radius", type=float, default=0.25)
    parser.add_argument("--ransac_max_iteration", type=int, default=100000)
    parser.add_argument("--ransac_distance_threshold", type=float, default=None,
                        help="默认 1.5×voxel_size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import open3d as o3d

    src = o3d.io.read_point_cloud(args.src)
    dst = o3d.io.read_point_cloud(args.dst)
    T, info = coarse_registration(src, dst, args.voxel_size, args.fpfh_radius,
                                  args.ransac_max_iteration,
                                  args.ransac_distance_threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, T)
    logger.info("变换 -> %s", out)
    logger.info("信息 -> %s", info)


if __name__ == "__main__":
    main()
