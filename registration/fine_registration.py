#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_registration.py — 改进 ICP 精配准（第三批交付，纯 CPU）

功能
----
    在粗配准结果基础上做精配准，核心改进点：
      1. 距离度量：点到平面（point-to-plane，利用目标点云法向量）；
      2. 自适应距离阈值：每次迭代阈值 = max(icp_threshold_min,
                                             icp_threshold_factor × 上次 RMSE)；
      3. 早停：相对 fitness 变化 < icp_early_stop_fitness 时停止，
         最大迭代 icp_max_iteration=50 次。

实现说明
--------
    通过 Open3D 的 registration_icp 逐次迭代推进：
    每次只执行 1 个内部迭代步（ICPConvergenceCriteria(max_iteration=1)），
    由外层循环控制阈值与早停，保证自适应策略可观测、可复现。

输出
----
    {output}.txt      最终 4×4 变换矩阵
    {output}_hist.json  逐迭代历史（阈值/fitness/RMSE），便于分析收敛

用法
----
    python registration/fine_registration.py \
        --source results/preprocess/pcd/a.ply \
        --target results/preprocess/pcd/b.ply \
        --init results/registration/pair_ab_coarse.txt \
        --output results/registration/pair_ab_fine.txt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 与 configs/default.yaml 对齐的默认值
DEFAULT_MAX_ITERATION = 50
DEFAULT_THRESHOLD_MIN = 0.02       # 自适应阈值下限（米）
DEFAULT_THRESHOLD_FACTOR = 0.5     # 阈值 = factor × 上次 RMSE
DEFAULT_EARLY_STOP_FITNESS = 1e-6  # 相对 fitness 变化早停阈值
DEFAULT_INITIAL_THRESHOLD = 0.1    # 首次迭代阈值（米）


def _one_step_icp(source, target, threshold, init_T):
    """执行一步点到平面 ICP，返回 (新变换, fitness, inlier_rmse)。"""
    import open3d as o3d

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.0,   # 关闭内部收敛判断，只走 1 步
        relative_rmse=0.0,
        max_iteration=1)
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria)
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def improved_icp(source, target, init_transformation=None,
                 max_iteration=DEFAULT_MAX_ITERATION,
                 threshold_min=DEFAULT_THRESHOLD_MIN,
                 threshold_factor=DEFAULT_THRESHOLD_FACTOR,
                 early_stop_fitness=DEFAULT_EARLY_STOP_FITNESS,
                 initial_threshold=DEFAULT_INITIAL_THRESHOLD):
    """改进 ICP 精配准（点到平面 + 自适应阈值 + 早停）。

    Args:
        source: 源点云（open3d PointCloud，含法向量更佳）。
        target: 目标点云（需含法向量，点到平面度量用）。
        init_transformation: 初始 4×4 变换（如粗配准结果），默认单位阵。
        max_iteration: 最大迭代次数（默认 50）。
        threshold_min: 自适应阈值下限（米，默认 0.02）。
        threshold_factor: 阈值更新系数（默认 0.5）。
        early_stop_fitness: 相对 fitness 变化早停阈值（默认 1e-6）。
        initial_threshold: 首次迭代阈值（米，默认 0.1）。
    Returns:
        (T_4x4, history)：最终变换与逐迭代历史列表。
    """
    if init_transformation is None:
        init_transformation = np.eye(4, dtype=np.float64)
    if not target.has_normals():
        import open3d as o3d
        logger.warning("目标点云缺少法向量，将补估（半径 0.1m）")
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                              max_nn=30))

    T = np.asarray(init_transformation, dtype=np.float64)
    threshold = float(initial_threshold)
    prev_fitness = -1.0
    history = []

    for it in range(max_iteration):
        T, fitness, rmse = _one_step_icp(source, target, threshold, T)
        history.append({
            "iter": it,
            "threshold": round(threshold, 6),
            "fitness": fitness,
            "inlier_rmse": rmse,
        })

        # 早停 1：无有效对应点（阈值过小或已发散）
        if fitness <= 0.0:
            logger.warning("第 %d 次迭代 fitness=0，提前终止", it)
            break
        # 早停 2：相对 fitness 变化小于阈值（收敛）
        if prev_fitness > 0:
            rel_change = abs(fitness - prev_fitness) / max(prev_fitness, 1e-12)
            if rel_change < early_stop_fitness:
                logger.info("第 %d 次迭代收敛：相对 fitness 变化 %.2e < %.2e",
                            it, rel_change, early_stop_fitness)
                break
        # 早停 3：RMSE 已到下限阈值（几乎完美对齐）
        if rmse <= threshold_min:
            logger.info("第 %d 次迭代 RMSE=%.4f ≤ 阈值下限 %.3f，停止",
                        it, rmse, threshold_min)
            break

        prev_fitness = fitness
        # 自适应距离阈值：max(threshold_min, factor × 上次 RMSE)
        threshold = max(threshold_min, threshold_factor * rmse)
        logger.debug("iter=%d thr=%.4f fitness=%.4f rmse=%.4f",
                     it, threshold, fitness, rmse)

    logger.info("ICP 完成：%d 次迭代，最终 fitness=%.4f, rmse=%.4f",
                len(history), history[-1]["fitness"], history[-1]["inlier_rmse"])
    return T, history


def save_result(transform, history, out_path):
    """保存变换矩阵（.txt）与迭代历史（{stem}_hist.json）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out_path), transform, fmt="%.9f")
    hist_path = out_path.with_name(out_path.stem + "_hist.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("变换矩阵 -> %s，迭代历史 -> %s", out_path, hist_path)


def main():
    parser = argparse.ArgumentParser(description="改进 ICP 精配准（点到平面+自适应阈值+早停）")
    parser.add_argument("--source", required=True, help="源点云路径")
    parser.add_argument("--target", required=True, help="目标点云路径")
    parser.add_argument("--init", default=None,
                        help="初始变换矩阵（.txt，粗配准输出），默认单位阵")
    parser.add_argument("--output", default="results/registration/fine.txt",
                        help="输出变换矩阵路径（.txt）")
    parser.add_argument("--max_iteration", type=int, default=DEFAULT_MAX_ITERATION)
    parser.add_argument("--threshold_min", type=float, default=DEFAULT_THRESHOLD_MIN)
    parser.add_argument("--threshold_factor", type=float, default=DEFAULT_THRESHOLD_FACTOR)
    parser.add_argument("--early_stop_fitness", type=float,
                        default=DEFAULT_EARLY_STOP_FITNESS)
    parser.add_argument("--initial_threshold", type=float,
                        default=DEFAULT_INITIAL_THRESHOLD)
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

    init_T = None
    if args.init:
        init_T = np.loadtxt(args.init).reshape(4, 4)
        logger.info("加载初始变换: %s", args.init)

    T, history = improved_icp(
        source, target, init_transformation=init_T,
        max_iteration=args.max_iteration,
        threshold_min=args.threshold_min,
        threshold_factor=args.threshold_factor,
        early_stop_fitness=args.early_stop_fitness,
        initial_threshold=args.initial_threshold)
    save_result(T, history, args.output)


if __name__ == "__main__":
    main()
