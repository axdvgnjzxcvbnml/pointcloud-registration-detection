#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_registration.py — CPU 侧配准超参粗调（V100 上线前预计算）

功能
----
    在模拟/真实帧对数据上，对三个配准超参做网格粗调：
      - RANSAC 迭代次数：100k / 300k / 500k
      - FPFH 半径：0.15 / 0.25 / 0.35（米）
      - 体素下采样：0.02 / 0.05 / 0.08（米）
    每组组合在全部帧对上跑「粗配准（FPFH+RANSAC）+ 精配准（改进 ICP）」,
    记录：成功率（旋转 <5° 且平移 <0.05m）、平均旋转误差、平均平移误差、
    平均耗时。输出 CSV 供人工挑选最优组合（默认写入
    results/registration/sweep/sweep_results.csv）。

用法
----
    python scripts/sweep_registration.py \
        --pairs results/preprocess/pairs/pairs.json \
        --pcd_dir results/preprocess/pcd \
        --pose_gt_dir results/preprocess/pose_gt \
        --out_dir results/registration/sweep \
        --max_pairs 7          # 可选：最多用几对（加速）

    # 只扫 RANSAC 迭代次数（快速模式，3 组合 × 3 对）
    python scripts/sweep_registration.py --pairs ... --quick

说明
----
    - 纯 CPU，只依赖 numpy / open3d；
    - 精配准参数固定用 configs/default.yaml 的 improved_icp 默认值，
      只扫粗配准的三个超参；
    - 帧对真值来自 compute_pose_gt.py 输出的 4×4 T_AB（B→A）。
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 保证以 `python scripts/xxx.py` 方式运行时能 import 项目包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

RANSAC_ITERS = [100000, 300000, 500000]
FPFH_RADII = [0.15, 0.25, 0.35]
VOXEL_SIZES = [0.02, 0.05, 0.08]

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


def run_one(source, target, T_gt, voxel, fpfh_r, ransac_it):
    """跑一组 (voxel, fpfh_r, ransac_it) 的粗+精配准，返回结果 dict。"""
    from registration.coarse_registration import coarse_registration
    from registration.fine_registration import improved_icp

    t0 = time.perf_counter()
    T_coarse, _ = coarse_registration(
        source, target, voxel_size=voxel,
        fpfh_radius=fpfh_r, max_iteration=ransac_it)
    T_fine, history = improved_icp(source, target, init_transformation=T_coarse)
    elapsed = time.perf_counter() - t0

    rot_deg, trans_m = pose_error(T_fine, T_gt)
    success = (rot_deg < ROT_THRESH_DEG) and (trans_m < TRANS_THRESH_M)
    n_icp_iters = len(history)
    return {
        "voxel_size": voxel, "fpfh_radius": fpfh_r,
        "ransac_max_iteration": ransac_it,
        "rot_deg": rot_deg, "trans_m": trans_m,
        "success": success, "time_s": round(elapsed, 3),
        "icp_iters": n_icp_iters,
    }


def main():
    parser = argparse.ArgumentParser(description="配准超参网格粗调（CPU）")
    parser.add_argument("--pairs", default="results/preprocess/pairs/pairs.json",
                        help="帧对清单 json（sample_frame_pairs.py 输出）")
    parser.add_argument("--pcd_dir", default="results/preprocess/pcd",
                        help="帧点云目录（.ply）")
    parser.add_argument("--pose_gt_dir", default="results/preprocess/pose_gt",
                        help="6DOF 真值目录（compute_pose_gt.py 输出）")
    parser.add_argument("--out_dir", default="results/registration/sweep",
                        help="结果输出目录（CSV + summary.json）")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="最多使用的帧对数（0 = 全部）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：只扫 RANSAC 迭代次数（3 组 × 前 3 对）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)
    pairs = pairs_data["pairs"]
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    logger.info("使用 %d 个帧对", len(pairs))

    # 超参网格
    grid = [(r, fr, v) for r in RANSAC_ITERS for fr in FPFH_RADII for v in VOXEL_SIZES]
    if args.quick:
        grid = [(r, 0.25, 0.02) for r in RANSAC_ITERS]  # 只扫 RANSAC

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for idx, (ransac_it, fpfh_r, voxel) in enumerate(grid, 1):
        n_succ = 0
        rot_list, trans_list, time_list, icp_list = [], [], [], []
        logger.info("[%d/%d] voxel=%.2f fpfh=%.2f ransac=%d",
                    idx, len(grid), voxel, fpfh_r, ransac_it)
        for item in pairs:
            scene = item["scene"]
            try:
                source, target = load_pair_clouds(
                    args.pcd_dir, scene, item["frame_a"], item["frame_b"])
                gt_path = Path(args.pose_gt_dir) / f"{item['pair_id']}.txt"
                if not gt_path.is_file():
                    logger.warning("缺少真值 %s，跳过该对", gt_path)
                    continue
                T_gt = np.loadtxt(str(gt_path)).reshape(4, 4)
                # 约定（与 evaluate_registration.py 一致）：compute_pose_gt 输出的
                # T_AB = inv(Pose_A)@Pose_B 是「B 帧坐标 -> A 帧坐标」；
                # 配准估计的是 source->target（A->B），故取逆再比较。
                T_gt = np.linalg.inv(T_gt)
                res = run_one(source, target, T_gt, voxel, fpfh_r, ransac_it)
            except Exception as e:  # 单对失败不影响整组
                logger.warning("帧对 %s 失败：%s", item["pair_id"], e)
                continue

            n_succ += int(res["success"])
            rot_list.append(res["rot_deg"])
            trans_list.append(res["trans_m"])
            time_list.append(res["time_s"])
            icp_list.append(res["icp_iters"])
            logger.info("  %s rot=%.3f° trans=%.4fm t=%.2fs %s",
                        item["pair_id"], res["rot_deg"], res["trans_m"],
                        res["time_s"], "OK" if res["success"] else "X")

        n_run = max(len(rot_list), 1)
        row = {
            "ransac_max_iteration": ransac_it,
            "fpfh_radius": fpfh_r,
            "voxel_size": voxel,
            "n_pairs_run": len(rot_list),
            "n_success": n_succ,
            "success_rate": round(n_succ / n_run, 4),
            "mean_rot_deg": round(float(np.mean(rot_list)), 4) if rot_list else None,
            "mean_trans_m": round(float(np.mean(trans_list)), 4) if trans_list else None,
            "median_time_s": round(float(np.median(time_list)), 3) if time_list else None,
            "mean_time_s": round(float(np.mean(time_list)), 3) if time_list else None,
            "mean_icp_iters": round(float(np.mean(icp_list)), 2) if icp_list else None,
        }
        all_rows.append(row)
        logger.info("  => 成功率 %.0f%%（%d/%d），中位耗时 %.2fs",
                    100 * row["success_rate"], n_succ, len(rot_list),
                    row["median_time_s"] or -1)

    # 排序：成功率降序 → 中位耗时升序
    all_rows.sort(key=lambda r: (-r["success_rate"],
                                 r["median_time_s"] if r["median_time_s"] is not None else 1e9))
    best = all_rows[0] if all_rows else None

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("CSV -> %s", csv_path)

    summary = {
        "grid_ransac": RANSAC_ITERS,
        "grid_fpfh": FPFH_RADII,
        "grid_voxel": VOXEL_SIZES,
        "n_pairs": len(pairs),
        "best": best,
        "rows": all_rows,
    }
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 超参扫描汇总（按成功率降序 / 耗时升序） ===")
    print(f"{'rank':<4}{'ransac':<8}{'fpfh':<7}{'voxel':<7}"
          f"{'succ%':<7}{'rot°':<8}{'trans_m':<9}{'med_s':<7}{'pairs'}")
    for rank, r in enumerate(all_rows, 1):
        print(f"{rank:<4}{r['ransac_max_iteration']:<8}{r['fpfh_radius']:<7}"
              f"{r['voxel_size']:<7}{100*r['success_rate']:<7.0f}"
              f"{(r['mean_rot_deg'] if r['mean_rot_deg'] is not None else 0):<8}"
              f"{(r['mean_trans_m'] if r['mean_trans_m'] is not None else 0):<9}"
              f"{(r['median_time_s'] if r['median_time_s'] is not None else 0):<7}"
              f"{r['n_pairs_run']}")
    if best:
        print(f"\n推荐最优组合：ransac_max_iteration={best['ransac_max_iteration']}, "
              f"fpfh_radius={best['fpfh_radius']}, voxel_size={best['voxel_size']}, "
              f"成功率 {100*best['success_rate']:.0f}%")
    logger.info("汇总 -> %s", out_dir / "sweep_summary.json")


if __name__ == "__main__":
    main()
