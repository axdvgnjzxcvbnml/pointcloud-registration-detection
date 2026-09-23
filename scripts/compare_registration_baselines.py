#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_registration_baselines.py — 需求文档 4.2 对比基线的补跑（纯 CPU）

背景
----
    需求文档 4.2 明确「对比基线为传统点到点 ICP 和 FGR」。
    早期任务四跑过 FGR/法向检查/多尺度，但均未按「需求基线参数」做
    「改进ICP vs 传统ICP(无粗配准) vs FGR→改进ICP」的三组严格对比。
    本脚本在需求基线参数（voxel=0.02 / FPFH=0.25 / RANSAC=100k）下补跑：

      组 A  FPFH+RANSAC → 改进ICP（点到平面）    —— 直接复用 req_baseline，不重跑
      组 B  传统点到点ICP（单位阵初始化，无粗配准）—— 本脚本
      组 C  FGR → 改进ICP                        —— 本脚本

    组 B / 组 C 各跑 --repeats 次重复（Open3D RANSAC/FGR 无固定随机种子，
    组 B 纯 ICP 为确定性算法，重复结果应一致）。

成功判据（全项目统一口径，见 docs/experiment_log.md §0.1）：
    旋转角误差 < 5° 且 平移误差 < 0.05m（不使用 inlier_rmse）。

用法
----
    python scripts/compare_registration_baselines.py \
        --pairs results/preprocess/pairs/pairs_real20.json \
        --pcd_dir results/preprocess/pcd_real \
        --pose_gt_dir results/preprocess/pose_gt_real20 \
        --voxel 0.02 --fpfh 0.25 --ransac 100000 --ransac_th 0.03 \
        --repeats 3 --out_dir results/registration/sweep_real/baselines_compare
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from registration.preprocess_pointcloud import preprocess_pointcloud  # noqa: E402
from registration.fine_registration import improved_icp  # noqa: E402
from registration.evaluate_registration import (  # noqa: E402
    rotation_error, translation_error, correspondence_rmse,
    load_pcd_points, pcd_path_for,
)


# ---------------------------------------------------------------
# 评测（与 evaluate_registration.py 口径一致）
# ---------------------------------------------------------------
def evaluate(T_est, T_gt, src_pts, tgt_pts, max_dist=0.05):
    """返回指标字典；T_est 语义 source→target，与 T_gt(=inv(T_AB)) 同向。"""
    rmse, ratio = correspondence_rmse(src_pts, tgt_pts, T_est, max_dist=max_dist)
    m = {
        "rmse": rmse,
        "inlier_ratio": ratio,
        "rot_deg": rotation_error(T_est, T_gt),
        "trans_m": translation_error(T_est, T_gt),
    }
    m["success"] = bool(m["rot_deg"] < 5.0 and m["trans_m"] < 0.05)
    return m


def run_pair_baselines(pair, pcd_dir, pose_gt_dir, params, rng_seed):
    """对单对跑组 B（纯点到点 ICP）与组 C（FGR → 改进ICP）。"""
    import open3d as o3d

    scene = pair["scene"]
    src_idx, tgt_idx = pair["frame_a"], pair["frame_b"]
    pair_id = f"{scene}_{int(src_idx):06d}_{int(tgt_idx):06d}"

    src_pcd = load_pcd_points(pcd_path_for(pcd_dir, scene, src_idx))
    tgt_pcd = load_pcd_points(pcd_path_for(pcd_dir, scene, tgt_idx))

    # 统一预处理（与 evaluate_registration.run_methods 一致）
    src = preprocess_pointcloud(src_pcd, voxel_size=params["voxel"])
    tgt = preprocess_pointcloud(tgt_pcd, voxel_size=params["voxel"])
    src_pts = np.asarray(src.points, dtype=np.float64)
    tgt_pts = np.asarray(tgt.points, dtype=np.float64)

    T_gt = np.loadtxt(pose_gt_dir / f"{pair_id}.txt")
    T_gt_align = np.linalg.inv(T_gt)  # 方向约定：估计 A→B vs 真值 B→A，取逆对齐

    I4 = np.eye(4)
    results = {}

    # --- 组 B：传统点到点 ICP（单位阵初始化，无粗配准） ---
    t0 = time.time()
    p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, params["ransac_th"], I4,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=params["icp_max_iteration"]))
    results["point2point_raw"] = evaluate(
        p2p.transformation, T_gt_align, src_pts, tgt_pts)
    results["point2point_raw"]["time_s"] = time.time() - t0

    # --- 组 C：FGR → 改进ICP ---
    t0 = time.time()
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(
            radius=params["fpfh"], max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(
            radius=params["fpfh"], max_nn=100))
    fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=params["ransac_th"]))
    T_fine, _ = improved_icp(
        src, tgt, init_transformation=fgr.transformation,
        max_iteration=params["icp_max_iteration"],
        threshold_min=params["icp_threshold_min"],
        threshold_factor=params["icp_threshold_factor"],
        early_stop_fitness=params["icp_early_stop_fitness"])
    results["fgr_improved"] = evaluate(
        T_fine, T_gt_align, src_pts, tgt_pts)
    results["fgr_improved"]["time_s"] = time.time() - t0

    return pair_id, pair["interval"], results


# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
def summarize(rows, methods, n_pairs):
    """rows: list of dict(pair_id, interval, method, repeat, success, rot, trans, rmse, time)"""
    summary = {}
    for method in methods:
        ms = [r for r in rows if r["method"] == method]
        n_reps = max(r["repeat"] for r in ms)
        rates = []
        for rep in range(1, n_reps + 1):
            rr = [r for r in ms if r["repeat"] == rep]
            rates.append(sum(r["success"] for r in rr))
        rates = sorted(rates)
        med = rates[len(rates) // 2]
        q = np.percentile(rates, [25, 50, 75])
        # 中位重复的逐对明细
        rep_med = [r for r in ms if r["repeat"] == rates.index(med) + 1] \
            if rates.index(med) + 1 in [r["repeat"] for r in ms] else \
            [r for r in ms if r["repeat"] == 1]
        ivs = {}
        for iv in sorted({r["interval"] for r in rep_med}):
            rr = [r for r in rep_med if r["interval"] == iv]
            ivs[f"iv{iv}"] = f"{sum(r['success'] for r in rr)}/{len(rr)}"
        summary[method] = {
            "n_pairs": n_pairs,
            "rates_all": rates,
            "success_rate_med": med / n_pairs,
            "success_rate_min": rates[0] / n_pairs,
            "success_rate_max": rates[-1] / n_pairs,
            "q1": float(q[0]) / n_pairs,
            "q3": float(q[2]) / n_pairs,
            "mean_rot_deg": float(np.mean([r["rot"] for r in ms])),
            "mean_trans_m": float(np.mean([r["trans"] for r in ms])),
            "mean_rmse": float(np.mean([r["rmse"] for r in ms])),
            "mean_time_s": float(np.mean([r["time"] for r in ms])),
            "ivs": ivs,
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="需求 4.2 对比基线补跑（组B 纯点到点ICP / 组C FGR→改进ICP）")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pcd_dir", required=True)
    ap.add_argument("--pose_gt_dir", required=True)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--fpfh", type=float, default=0.25)
    ap.add_argument("--ransac", type=int, default=100000)
    ap.add_argument("--ransac_th", type=float, default=0.03)
    ap.add_argument("--icp_max_iteration", type=int, default=50)
    ap.add_argument("--icp_threshold_min", type=float, default=0.02)
    ap.add_argument("--icp_threshold_factor", type=float, default=0.5)
    ap.add_argument("--icp_early_stop_fitness", type=float, default=1e-6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.load(open(args.pairs, "r", encoding="utf-8"))
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs", pairs.get("frames", []))
    logger.info("使用 %d 个帧对，需求基线参数 voxel=%.2f fpfh=%.2f ransac=%d，重复 %d 次",
                len(pairs), args.voxel, args.fpfh, args.ransac, args.repeats)

    pcd_dir = Path(args.pcd_dir)
    pose_gt_dir = Path(args.pose_gt_dir)
    params = vars(args)

    rows = []
    for rep in range(1, args.repeats + 1):
        logger.info("===== 重复 %d/%d =====", rep, args.repeats)
        for pair in pairs:
            pair_id, iv, res = run_pair_baselines(
                pair, pcd_dir, pose_gt_dir, params, rep)
            for method in ("point2point_raw", "fgr_improved"):
                m = res[method]
                rows.append({
                    "pair_id": pair_id, "interval": iv, "method": method,
                    "repeat": rep, "success": m["success"],
                    "rot": m["rot_deg"], "trans": m["trans_m"],
                    "rmse": m["rmse"], "time": m["time_s"],
                })
            logger.info(
                "  %s(iv=%d) p2p_raw: rot=%.3f° trans=%.4fm %s | "
                "fgr_imp: rot=%.3f° trans=%.4fm %s",
                pair_id, iv,
                res["point2point_raw"]["rot_deg"], res["point2point_raw"]["trans_m"],
                "OK" if res["point2point_raw"]["success"] else "X",
                res["fgr_improved"]["rot_deg"], res["fgr_improved"]["trans_m"],
                "OK" if res["fgr_improved"]["success"] else "X")

    # CSV（逐对 × 重复）
    csv_path = out_dir / "per_pair_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "interval", "method", "repeat",
                                          "success", "rot_deg", "trans_m", "rmse", "time_s"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "pair_id": r["pair_id"], "interval": r["interval"],
                "method": r["method"], "repeat": r["repeat"],
                "success": "1" if r["success"] else "0", "rot_deg": round(r["rot"], 6),
                "trans_m": round(r["trans"], 6), "rmse": round(r["rmse"], 6),
                "time_s": round(r["time"], 3)})

    summary = summarize(rows, ["point2point_raw", "fgr_improved"], len(pairs))
    json_path = out_dir / "sweep_summary.json"
    json.dump(summary, open(json_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\n=== 对比基线汇总（需求基线参数 voxel=%.2f/fpfh=%.2f/ransac=%d，%d 对 × %d 次）==="
          % (args.voxel, args.fpfh, args.ransac, len(pairs), args.repeats))
    print(f"{'方法':<22}{'med':>6}{'min':>6}{'max':>6}  {'rot°':>8}{'trans_m':>9}{'rmse':>8}{'time_s':>7}  ivs")
    for method in ("point2point_raw", "fgr_improved"):
        s = summary[method]
        print(f"{method:<22}{s['success_rate_med']*100:>5.0f}%{s['success_rate_min']*100:>5.0f}%"
              f"{s['success_rate_max']*100:>5.0f}%  {s['mean_rot_deg']:>8.2f}{s['mean_trans_m']:>9.3f}"
              f"{s['mean_rmse']:>8.4f}{s['mean_time_s']:>7.2f}  "
              f"iv5={s['ivs'].get('iv5','?')} iv10={s['ivs'].get('iv10','?')} iv30={s['ivs'].get('iv30','?')}")
    logger.info("CSV -> %s", csv_path)
    logger.info("汇总 -> %s", json_path)


if __name__ == "__main__":
    main()
