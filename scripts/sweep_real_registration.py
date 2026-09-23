#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_real_registration.py — 真实 SUN3D 数据配准单参数扫描（CPU）

用途
----
    在真实数据 20 对帧对（间隔 5/10/30）上，对粗配准关键超参做单变量扫描：
      - FPFH 搜索半径（--param fpfh_radius）
      - RANSAC 最大迭代（--param ransac_iteration）
      - 体素下采样大小（--param voxel_size）
      - 对应点互滤波开关（--param mutual_filter）
    固定其余参数为 configs/default.yaml 的基线值；每组跑粗配准 + 改进 ICP，
    记录：成功率（旋转<5° 且平移<0.05m）、平均 RMSE、平均旋转误差、平均平移误差、
    平均内点数（粗配准 correspondence 数）、平均耗时，并按间隔 5/10/30 分组输出。

用法
----
    # FPFH 半径扫描（任务一）
    python scripts/sweep_real_registration.py \
        --pairs results/preprocess/pairs/pairs_real20.json \
        --pcd_dir results/preprocess/pcd_real \
        --pose_gt_dir results/preprocess/pose_gt_real20 \
        --param fpfh_radius \
        --values 0.10,0.15,0.20,0.25,0.30,0.35,0.40 \
        --out_dir results/registration/sweep_real/fpfh

    # RANSAC 迭代 + mutual_filter 扫描（任务二）
    python scripts/sweep_real_registration.py --pairs ... --pose_gt_dir ... \
        --param ransac_iteration --values 100000,200000,300000,500000,1000000 \
        --mutual_filter 1 --out_dir results/registration/sweep_real/ransac_mf1
    # （关闭互滤波对比：--mutual_filter 0）

    # 体素扫描（任务三）
    python scripts/sweep_real_registration.py --pairs ... --pose_gt_dir ... \
        --param voxel_size --values 0.01,0.02,0.03,0.05,0.08 \
        --out_dir results/registration/sweep_real/voxel

输出
----
    {out_dir}/sweep_results.csv     每组一行（含按间隔成功率）
    {out_dir}/sweep_summary.json    结构化汇总
    stdout 打印按成功率降序的对比表

说明
----
    - 纯 CPU，依赖 numpy / open3d；
    - 真值方向约定与 evaluate_registration.py 一致：T_AB = inv(Pose_A)@Pose_B
      为 B→A，配准估计 A→B，比较前取逆；
    - 每对失败不影响整组（try/except 包裹）。
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# 基线参数（与 configs/default.yaml 对齐）
BASE_FPFH_RADIUS = 0.25
BASE_RANSAC_ITER = 100000
BASE_VOXEL_SIZE = 0.02
BASE_MUTUAL_FILTER = True

ROT_THRESH_DEG = 5.0
TRANS_THRESH_M = 0.05


def load_pair_clouds(pcd_dir, scene, frame_a, frame_b):
    """按约定文件名加载帧点云（.ply）。"""
    import open3d as o3d

    def _load(fid):
        p = Path(pcd_dir) / f"{scene}_{fid:06d}.ply"
        if not p.is_file():
            raise FileNotFoundError(f"点云不存在: {p}")
        return o3d.io.read_point_cloud(str(p))

    return _load(frame_a), _load(frame_b)


def pose_error(T_pred, T_gt):
    """旋转角误差（度）与平移误差（米）。"""
    R_err = T_pred[:3, :3].T @ T_gt[:3, :3]
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.rad2deg(np.arccos(cos_a)))
    trans_m = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))
    return rot_deg, trans_m


def run_one(source, target, T_gt, voxel, fpfh_r, ransac_it, mutual_filter):
    """跑一组参数的粗+精配准，返回结果 dict。"""
    from registration.coarse_registration import coarse_registration
    from registration.fine_registration import improved_icp

    t0 = time.perf_counter()
    T_coarse, info = coarse_registration(
        source, target, voxel_size=voxel,
        fpfh_radius=fpfh_r, max_iteration=ransac_it,
        mutual_filter=mutual_filter)
    T_fine, history = improved_icp(source, target, init_transformation=T_coarse)
    elapsed = time.perf_counter() - t0

    rot_deg, trans_m = pose_error(T_fine, T_gt)
    success = (rot_deg < ROT_THRESH_DEG) and (trans_m < TRANS_THRESH_M)
    return {
        "rot_deg": rot_deg, "trans_m": trans_m,
        "success": success, "time_s": round(elapsed, 3),
        "rmse": history[-1]["inlier_rmse"] if history else None,
        "n_corr": info.get("n_correspondences", 0),
        "coarse_fitness": info.get("fitness", 0.0),
    }


def run_once(pairs, pcd_dir, pose_gt_dir, voxel, fpfh_r, ransac_it, mf):
    """对全部帧对跑一次给定参数，返回逐对结果与汇总。"""
    n_succ = 0
    rot_list, trans_list, rmse_list, time_list, corr_list = [], [], [], [], []
    per_interval = {5: [0, 0], 10: [0, 0], 30: [0, 0]}  # [成功, 总数]
    per_pair = []  # 逐对明细（供失败分析）
    for item in pairs:
        scene = item["scene"]
        try:
            source, target = load_pair_clouds(
                pcd_dir, scene, item["frame_a"], item["frame_b"])
            gt_path = Path(pose_gt_dir) / f"{item['pair_id']}.txt"
            if not gt_path.is_file():
                logger.warning("缺少真值 %s，跳过", gt_path)
                continue
            T_gt = np.loadtxt(str(gt_path)).reshape(4, 4)
            T_gt = np.linalg.inv(T_gt)  # 方向约定，见 docstring
            res = run_one(source, target, T_gt, voxel, fpfh_r,
                          ransac_it, mf)
        except Exception as e:
            logger.warning("帧对 %s 失败：%s", item["pair_id"], e)
            continue

        iv = item["interval"]
        n_succ += int(res["success"])
        rot_list.append(res["rot_deg"])
        trans_list.append(res["trans_m"])
        if res["rmse"] is not None:
            rmse_list.append(res["rmse"])
        time_list.append(res["time_s"])
        corr_list.append(res["n_corr"])
        per_interval[iv][1] += 1
        per_interval[iv][0] += int(res["success"])
        per_pair.append({**item, **res})
        logger.info("  %s(iv=%d) rot=%.3f° trans=%.4fm rmse=%.4f corr=%d t=%.2fs %s",
                    item["pair_id"], iv, res["rot_deg"], res["trans_m"],
                    res["rmse"] or -1, res["n_corr"], res["time_s"],
                    "OK" if res["success"] else "X")
    n_run = max(len(rot_list), 1)
    return {
        "n_succ": n_succ, "n_run": len(rot_list),
        "rot_list": rot_list, "trans_list": trans_list,
        "rmse_list": rmse_list, "time_list": time_list,
        "corr_list": corr_list, "per_interval": per_interval,
        "per_pair": per_pair,
    }


def main():
    parser = argparse.ArgumentParser(description="真实数据配准单参数扫描（CPU）")
    parser.add_argument("--pairs", required=True, help="帧对清单 json")
    parser.add_argument("--pcd_dir", required=True, help="帧点云目录（.ply）")
    parser.add_argument("--pose_gt_dir", required=True, help="6DOF 真值目录")
    parser.add_argument("--param", required=True,
                        choices=["fpfh_radius", "ransac_iteration",
                                 "voxel_size", "mutual_filter"],
                        help="扫描的参数")
    parser.add_argument("--values", required=True,
                        help="逗号分隔的取值列表（mutual_filter 用 0/1）")
    parser.add_argument("--mutual_filter", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=None, help="固定互滤波开关（不随 --param 扫描时生效）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每个取值重复次数（默认 1；>1 时成功率取中位数，"
                             "并记录 min/max 范围）")
    parser.add_argument("--fixed_fpfh", type=float, default=None,
                        help="固定 FPFH 半径（覆盖 BASE，用于非 fpfh_radius 扫描）")
    parser.add_argument("--fixed_ransac", type=int, default=None,
                        help="固定 RANSAC 迭代（覆盖 BASE，用于非 ransac_iteration 扫描）")
    parser.add_argument("--fixed_voxel", type=float, default=None,
                        help="固定体素大小（覆盖 BASE，用于非 voxel_size 扫描）")
    parser.add_argument("--out_dir", default="results/registration/sweep_real",
                        help="结果输出目录")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    raw_values = [v.strip() for v in args.values.split(",")]
    if args.param == "mutual_filter":
        values = [v.lower() in ("1", "true", "yes") for v in raw_values]
    elif args.param == "ransac_iteration":
        values = [int(v) for v in raw_values]
    else:
        values = [float(v) for v in raw_values]

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    logger.info("使用 %d 个帧对，扫描参数 %s：%s，重复 %d 次",
                len(pairs), args.param, values, args.repeats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_pair_all = []  # 所有组的逐对明细（失败分析用）
    for val in values:
        # 构造本次组合的参数（fixed_* 优先覆盖 BASE）
        voxel = args.fixed_voxel if args.fixed_voxel is not None else BASE_VOXEL_SIZE
        fpfh_r = args.fixed_fpfh if args.fixed_fpfh is not None else BASE_FPFH_RADIUS
        ransac_it = args.fixed_ransac if args.fixed_ransac is not None else BASE_RANSAC_ITER
        mf = BASE_MUTUAL_FILTER if args.mutual_filter is None else args.mutual_filter
        if args.param == "fpfh_radius":
            fpfh_r = val
        elif args.param == "ransac_iteration":
            ransac_it = val
        elif args.param == "voxel_size":
            voxel = val
        elif args.param == "mutual_filter":
            mf = val

        runs = []
        for rep in range(args.repeats):
            logger.info("[%s=%s，重复 %d/%d] voxel=%.2f fpfh=%.2f ransac=%d mf=%s",
                        args.param, val, rep + 1, args.repeats,
                        voxel, fpfh_r, ransac_it, mf)
            runs.append(run_once(pairs, args.pcd_dir, args.pose_gt_dir,
                                 voxel, fpfh_r, ransac_it, mf))

        # 汇总：成功率取中位数，其余指标取多次均值
        rates = [r["n_succ"] / max(r["n_run"], 1) for r in runs]
        med_rate = float(np.median(rates))
        min_rate, max_rate = float(np.min(rates)), float(np.max(rates))
        n_succ_med = int(round(med_rate * runs[0]["n_run"]))
        # 选取中位成功率对应的一次作为间隔分组/逐对明细代表
        idx_med = int(np.argsort(rates)[len(rates) // 2]) if args.repeats > 1 else 0
        rep_med = runs[idx_med]

        def _mean(key):
            vals = [r[key] for r in runs if r[key]]
            return float(np.mean(vals)) if vals else None

        row = {
            "param": args.param,
            "value": str(val),
            "n_pairs_run": runs[0]["n_run"],
            "n_success_med": n_succ_med,
            "success_rate_med": round(med_rate, 4),
            "success_rate_min": round(min_rate, 4),
            "success_rate_max": round(max_rate, 4),
            "rates_all": [round(r, 4) for r in rates],
            "mean_rot_deg": round(_mean("rot_list"), 4) if _mean("rot_list") else None,
            "mean_trans_m": round(_mean("trans_list"), 5) if _mean("trans_list") else None,
            "mean_rmse": round(_mean("rmse_list"), 5) if _mean("rmse_list") else None,
            "mean_n_corr": round(_mean("corr_list"), 1) if _mean("corr_list") else None,
            "mean_time_s": round(_mean("time_list"), 3) if _mean("time_list") else None,
            "succ_iv5": f"{rep_med['per_interval'][5][0]}/{rep_med['per_interval'][5][1]}",
            "succ_iv10": f"{rep_med['per_interval'][10][0]}/{rep_med['per_interval'][10][1]}",
            "succ_iv30": f"{rep_med['per_interval'][30][0]}/{rep_med['per_interval'][30][1]}",
        }
        all_rows.append(row)
        for p in rep_med["per_pair"]:
            per_pair_all.append({**p, "param": args.param, "value": str(val)})
        logger.info("  => 成功率中位 %.0f%%（范围 %.0f%%-%.0f%%），iv5=%s iv10=%s iv30=%s，耗时 %.2fs",
                    100 * med_rate, 100 * min_rate, 100 * max_rate,
                    row["succ_iv5"], row["succ_iv10"], row["succ_iv30"],
                    row["mean_time_s"] or -1)

    # 排序：中位成功率降序 → 耗时升序
    all_rows.sort(key=lambda r: (-r["success_rate_med"],
                                 r["mean_time_s"] if r["mean_time_s"] is not None else 1e9))
    best = all_rows[0] if all_rows else None

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [k for k in all_rows[0].keys() if k != "rates_all"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    logger.info("CSV -> %s", csv_path)

    # 逐对明细（失败分析）
    if per_pair_all:
        pair_csv = out_dir / "per_pair_results.csv"
        with open(pair_csv, "w", newline="", encoding="utf-8") as f:
            keys = ["pair_id", "interval", "param", "value",
                    "success", "rot_deg", "trans_m", "rmse", "n_corr", "time_s"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_pair_all)
        logger.info("逐对明细 -> %s", pair_csv)

    summary = {"param": args.param, "values": values,
               "repeats": args.repeats, "n_pairs": len(pairs),
               "best": best, "rows": all_rows}
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== 真实数据单参数扫描汇总（{args.param}，中位成功率降序） ===")
    print(f"{'rank':<4}{'value':<10}{'med%':<6}{'min%':<6}{'max%':<6}"
          f"{'rot°':<9}{'trans_m':<10}{'rmse':<9}{'mean_s':<8}"
          f"{'iv5':<6}{'iv10':<6}{'iv30':<6}")
    for rank, r in enumerate(all_rows, 1):
        print(f"{rank:<4}{r['value']:<10}{100*r['success_rate_med']:<6.0f}"
              f"{100*r['success_rate_min']:<6.0f}{100*r['success_rate_max']:<6.0f}"
              f"{(r['mean_rot_deg'] if r['mean_rot_deg'] is not None else 0):<9}"
              f"{(r['mean_trans_m'] if r['mean_trans_m'] is not None else 0):<10}"
              f"{(r['mean_rmse'] if r['mean_rmse'] is not None else 0):<9}"
              f"{(r['mean_time_s'] if r['mean_time_s'] is not None else 0):<8}"
              f"{r['succ_iv5']:<6}{r['succ_iv10']:<6}{r['succ_iv30']}")
    if best:
        print(f"\n最优取值：{args.param}={best['value']}，中位成功率 "
              f"{100*best['success_rate_med']:.0f}%（范围 "
              f"{100*best['success_rate_min']:.0f}-{100*best['success_rate_max']:.0f}%）")
    logger.info("汇总 -> %s", out_dir / "sweep_summary.json")


if __name__ == "__main__":
    main()
