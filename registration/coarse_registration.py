#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""粗配准：FPFH 特征 + RANSAC（CPU）。

流程：
1. 输入两帧原始点云（open3d.PointCloud）；
2. 体素下采样 + 法向量估计（用于 FPFH，不在本模块做离群点去除，
   由 preprocess_pointcloud.py 负责）；
3. 计算 FPFH 特征；
4. RANSAC 特征匹配求 4×4 粗变换；
5. 返回 (T_coarse, info)。

默认参数与 configs/default.yaml 的 registration 节一致。
"""
import logging
import time

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def _preprocess_for_fpfh(pcd, voxel_size, normal_radius=0.1):
    """下采样 + 法向量估计（FPFH 需要）。"""
    p = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    return p


def _compute_fpfh(pcd, fpfh_radius, max_nn=100):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=max_nn))


def coarse_registration(
    source,
    target,
    voxel_size=0.03,
    fpfh_radius=0.40,
    max_iteration=500000,
    mutual_filter=False,
    distance_threshold=None,
    ransac_n=3,
    confidence=0.999,
    normal_angle_deg=None,
):
    """FPFH + RANSAC 粗配准。

    Parameters
    ----------
    source, target : open3d.geometry.PointCloud
        原始（未预处理）点云。
    voxel_size : float
        体素下采样大小（米），默认 0.03（真实数据最优）。
    fpfh_radius : float
        FPFH 搜索半径（米），默认 0.40（真实数据最优）。
    max_iteration : int
        RANSAC 最大迭代，默认 500000。
    mutual_filter : bool
        是否启用对应点互滤波（True 可提升精度但降低内点数，默认 False，
        真实数据扫描结论：False 成功率更高）。
    distance_threshold : float | None
        RANSAC 距离阈值；None 时取 1.5 × voxel_size。
    normal_angle_deg : float | None
        若给定，则增加法向量一致性 checker（夹角 < 该角度，单位度）。
        实验结论：设为 30 时 iv10 成功率 29%→43%，但 iv30 33%→17%，
        适合以间隔 ≤10 为主的应用；追求整体鲁棒性保持 None。

    Returns
    -------
    (T, info) : (np.ndarray(4,4), dict)
        info 含 fitness / inlier_rmse / n_correspondences / time_s。
    """
    if voxel_size <= 0 or fpfh_radius <= 0:
        raise ValueError("voxel_size 与 fpfh_radius 必须为正")
    t0 = time.perf_counter()

    src = _preprocess_for_fpfh(source, voxel_size)
    tgt = _preprocess_for_fpfh(target, voxel_size)
    if src.is_empty() or tgt.is_empty():
        raise ValueError("下采样后点云为空，请检查体素大小/输入点云")

    src_f = _compute_fpfh(src, fpfh_radius)
    tgt_f = _compute_fpfh(tgt, fpfh_radius)

    thr = distance_threshold if distance_threshold is not None else 1.5 * voxel_size

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr),
    ]
    if normal_angle_deg is not None:
        checkers.append(
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                np.deg2rad(normal_angle_deg)))

    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f,
        mutual_filter=mutual_filter,
        max_correspondence_distance=thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_iteration, confidence=confidence),
    )

    info = {
        "fitness": float(res.fitness),
        "inlier_rmse": float(res.inlier_rmse),
        "n_correspondences": int(len(res.correspondence_set))
        if res.correspondence_set is not None else 0,
        "time_s": round(time.perf_counter() - t0, 3),
        "voxel_size": voxel_size,
        "fpfh_radius": fpfh_radius,
        "max_iteration": max_iteration,
        "mutual_filter": mutual_filter,
    }
    return res.transformation, info


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if len(sys.argv) != 3:
        print("用法: python registration/coarse_registration.py <source.ply> <target.ply>")
        sys.exit(1)
    src = o3d.io.read_point_cloud(sys.argv[1])
    tgt = o3d.io.read_point_cloud(sys.argv[2])
    T, info = coarse_registration(src, tgt)
    print("粗变换矩阵:\n", T)
    print("info:", info)
