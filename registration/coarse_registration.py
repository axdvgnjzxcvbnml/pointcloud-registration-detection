#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_registration.py — FPFH + RANSAC 粗配准（第三批交付，纯 CPU）

功能
----
    对源/目标点云执行粗配准：
      1. 复用 preprocess_pointcloud 做去噪/降采样/法向量；
      2. 计算 FPFH 特征描述子（搜索半径 fpfh_radius=0.25m）；
      3. RANSAC 配准（最大迭代 100000 次，
         距离阈值 distance_threshold = 1.5 × voxel_size = 0.03m）；
    输出把源点云变换到目标点云坐标系的 4×4 矩阵。

输出
----
    {output}.txt    4×4 变换矩阵（行优先，可直接 np.loadtxt）
    {output}.json   配准信息（fitness / inlier_rmse / 参数快照）

用法
----
    python registration/coarse_registration.py \
        --source results/preprocess/pcd/a.ply \
        --target results/preprocess/pcd/b.ply \
        --output results/registration/pair_ab_coarse.txt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 与 configs/default.yaml 对齐的默认值
DEFAULT_VOXEL_SIZE = 0.02
DEFAULT_FPFH_RADIUS = 0.25
DEFAULT_MAX_ITERATION = 100000
DEFAULT_CONFIDENCE = 0.999


def preprocess_pointcloud(pcd, voxel_size, normal_radius=0.1):
    """轻量包装：只做体素降采样 + 法向量估计（粗配准阶段）。

    说明：RANSAC 用 FPFH 特征匹配，去离群已由上游处理；
    若需要完整流水线，可改调 preprocess_pointcloud.py 的版本。
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    return pcd


def compute_fpfh(pcd, fpfh_radius=0.25, max_nn=100):
    """计算 FPFH 特征描述子。

    Args:
        pcd: 含法向量的 open3d.geometry.PointCloud。
        fpfh_radius: FPFH 特征搜索半径（米），默认 0.25m。
        max_nn: 邻域最大点数。
    Returns:
        open3d.pipelines.registration.Feature（33 维 FPFH）。
    """
    import open3d as o3d

    if not pcd.has_normals():
        raise RuntimeError("点云缺少法向量，请先 estimate_normals")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius,
                                                  max_nn=max_nn))
    return fpfh


def coarse_registration(source, target, voxel_size=DEFAULT_VOXEL_SIZE,
                        fpfh_radius=DEFAULT_FPFH_RADIUS,
                        max_iteration=DEFAULT_MAX_ITERATION,
                        distance_threshold=None,
                        confidence=DEFAULT_CONFIDENCE):
    """FPFH + RANSAC 粗配准。

    Args:
        source: 源点云（open3d PointCloud，未预处理也可，内部会降采样）。
        target: 目标点云。
        voxel_size: 体素降采样尺寸（米）。
        fpfh_radius: FPFH 搜索半径（米）。
        max_iteration: RANSAC 最大迭代次数。
        distance_threshold: 对应点距离阈值（米），默认 1.5 × voxel_size。
        confidence: RANSAC 置信度。
    Returns:
        (T_4x4, info_dict)：粗变换矩阵与配准信息。
    """
    import open3d as o3d

    if distance_threshold is None:
        distance_threshold = 1.5 * voxel_size
    logger.info("粗配准参数: voxel=%.3f, fpfh_r=%.2f, iter=%d, thr=%.4f",
                voxel_size, fpfh_radius, max_iteration, distance_threshold)

    src = preprocess_pointcloud(source, voxel_size)
    tgt = preprocess_pointcloud(target, voxel_size)

    src_fpfh = compute_fpfh(src, fpfh_radius)
    tgt_fpfh = compute_fpfh(tgt, fpfh_radius)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_iteration, confidence=confidence))

    info = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "n_correspondences": int(np.asarray(result.correspondence_set).shape[0])
        if result.correspondence_set is not None else 0,
        "params": {
            "voxel_size": voxel_size,
            "fpfh_radius": fpfh_radius,
            "max_iteration": max_iteration,
            "distance_threshold": distance_threshold,
            "confidence": confidence,
        },
    }
    logger.info("RANSAC 结果: fitness=%.4f, inlier_rmse=%.4f",
                info["fitness"], info["inlier_rmse"])
    return result.transformation, info


def save_result(transform, info, out_path):
    """保存 4×4 矩阵（.txt）与配准信息（.json）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out_path), transform, fmt="%.9f")
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info("变换矩阵 -> %s，信息 -> %s", out_path, json_path)


def main():
    parser = argparse.ArgumentParser(description="FPFH + RANSAC 粗配准")
    parser.add_argument("--source", required=True, help="源点云路径")
    parser.add_argument("--target", required=True, help="目标点云路径")
    parser.add_argument("--output", default="results/registration/coarse.txt",
                        help="输出变换矩阵路径（.txt）")
    parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE,
                        help="体素降采样尺寸（米）")
    parser.add_argument("--fpfh_radius", type=float, default=DEFAULT_FPFH_RADIUS,
                        help="FPFH 搜索半径（米）")
    parser.add_argument("--max_iteration", type=int, default=DEFAULT_MAX_ITERATION,
                        help="RANSAC 最大迭代次数")
    parser.add_argument("--distance_threshold", type=float, default=None,
                        help="对应点距离阈值（米），默认 1.5×voxel_size")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help="RANSAC 置信度")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import open3d as o3d

    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)
    if source.is_empty() or target.is_empty():
        raise RuntimeError("源/目标点云为空，请检查路径")

    T, info = coarse_registration(
        source, target,
        voxel_size=args.voxel_size,
        fpfh_radius=args.fpfh_radius,
        max_iteration=args.max_iteration,
        distance_threshold=args.distance_threshold,
        confidence=args.confidence)
    save_result(T, info, args.output)


if __name__ == "__main__":
    main()
