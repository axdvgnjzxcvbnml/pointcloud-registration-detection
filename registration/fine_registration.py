#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_registration.py — 改进 ICP 精配准（第三批交付）

功能
----
    在粗配准初值基础上做精配准，三项改进：
      1. 点到平面（PointToPlane），提升收敛性与精度
      2. 自适应距离阈值：threshold = max(icp_threshold_min,
                                       factor × 上一次 RMSE)
      3. 早停：fitness 相对变化 < icp_early_stop_fitness

说明
----
    纯 CPU（open3d ICP）。输入点云应已预处理（含法向量）。
    --global 可选：先跑全局配准（coarse_registration）再精配准。

用法
----
    python registration/fine_registration.py \
        --src a.ply --dst b.ply \
        --init results/reg/T_coarse.npy \
        --out results/reg/T_fine.npy
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def fine_registration(src, dst, init_T=None, voxel_size=0.02,
                      max_iteration=50, adaptive=True, threshold_min=0.02,
                      threshold_factor=0.5, early_stop_fitness=1e-6):
    """改进 ICP 精配准。

    Args:
        src: 源点云（已预处理，含法向量）。
        dst: 目标点云（已预处理，含法向量）。
        init_T: 初始 4×4 变换；None 时用单位阵。
        adaptive: 自适应阈值开关。
    Returns:
        (T_src2dst (4,4), info dict)。
    """
    import open3d as o3d

    if init_T is None:
        init_T = np.eye(4)

    threshold = 1.5 * voxel_size if not adaptive else max(threshold_min,
                                                          1.5 * voxel_size)
    best = None
    for it in range(max_iteration):
        result = o3d.pipelines.registration.registration_icp(
            src, dst, threshold, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=1, relative_fitness=0.0, relative_rmse=0.0))
        init_T = result.transformation
        if best is None or result.fitness > best.fitness:
            best = result
        if early_stop_fitness > 0 and abs(best.fitness - result.fitness) <= early_stop_fitness:
            break
        if adaptive:
            threshold = max(threshold_min, threshold_factor * result.inlier_rmse)
    info = {"fitness": best.fitness, "rmse": best.inlier_rmse,
            "threshold_final": threshold, "iterations": it + 1}
    logger.info("精配准：fitness=%.4f rmse=%.4f（%d 次内迭代）",
                best.fitness, best.inlier_rmse, it + 1)
    return best.transformation, info


def main():
    parser = argparse.ArgumentParser(description="改进 ICP 精配准")
    parser.add_argument("--src", required=True, help="源点云 .ply")
    parser.add_argument("--dst", required=True, help="目标点云 .ply")
    parser.add_argument("--init", default=None,
                        help="初始变换 .npy（缺省=单位阵）")
    parser.add_argument("--out", default="results/reg/T_fine.npy")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--max_iteration", type=int, default=50)
    parser.add_argument("--no_adaptive", action="store_true",
                        help="关闭自适应阈值")
    parser.add_argument("--threshold_min", type=float, default=0.02)
    parser.add_argument("--threshold_factor", type=float, default=0.5)
    parser.add_argument("--global", dest="use_global", action="store_true",
                        help="先跑全局粗配准再精配准")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import open3d as o3d

    src = o3d.io.read_point_cloud(args.src)
    dst = o3d.io.read_point_cloud(args.dst)
    init_T = np.load(args.init) if args.init else None
    if args.use_global:
        from coarse_registration import coarse_registration
        init_T, _ = coarse_registration(src, dst, voxel_size=args.voxel_size)
    T, info = fine_registration(src, dst, init_T, args.voxel_size,
                                args.max_iteration, not args.no_adaptive,
                                args.threshold_min, args.threshold_factor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, T)
    logger.info("变换 -> %s", out)
    logger.info("信息 -> %s", info)


if __name__ == "__main__":
    main()
