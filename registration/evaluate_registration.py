#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_registration.py — 配准批量评测（第三批交付）

功能
----
    对帧对清单逐对执行“粗配准 + 精配准”，并按 GT 位姿评测：
      - 旋转误差（度）
      - 平移误差（米）
      - 对齐后 RMSE
      - 成功率（各阈值组合）
    输出 eval.json 与终端汇总。

说明
----
    - GT 位姿由 compute_pose_gt.py 生成（{src}_to_{dst}.npy）
    - RMSE 口径：T_fine 与 GT 误差（两变换差的旋转/平移范数）

用法
----
    python registration/evaluate_registration.py \
        --pairs results/preprocess/pairs.json \
        --pose-gt-dir results/preprocess/pose_gt \
        --pcd-dir results/preprocess/pcd \
        --out results/reg/eval.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def rotation_error_deg(R_pred, R_gt):
    """两旋转矩阵的夹角误差（度）。"""
    R = R_pred @ R_gt.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def translation_error_m(t_pred, t_gt):
    """两平移向量欧氏距离（米）。"""
    return float(np.linalg.norm(t_pred - t_gt))


def evaluate_pair(src_pcd, dst_pcd, T_gt, voxel_size=0.02):
    """跑完整配准流水线并返回误差指标。"""
    from preprocess_pointcloud import preprocess_pointcloud
    from coarse_registration import coarse_registration
    from fine_registration import fine_registration

    s = preprocess_pointcloud(src_pcd)
    d = preprocess_pointcloud(dst_pcd)
    T_coarse, c_info = coarse_registration(s, d, voxel_size=voxel_size)
    T_fine, f_info = fine_registration(s, d, T_coarse, voxel_size=voxel_size)

    rot_err = rotation_error_deg(T_fine[:3, :3], T_gt[:3, :3])
    trans_err = translation_error_m(T_fine[:3, 3], T_gt[:3, 3])
    return {
        "rotation_error_deg": rot_err,
        "translation_error_m": trans_err,
        "coarse_fitness": c_info["fitness"],
        "fine_fitness": f_info["fitness"],
        "fine_rmse": f_info["rmse"],
    }


def summarize(results, thresholds=None):
    """汇总成功率等统计。"""
    if thresholds is None:
        thresholds = [{"rot": 5.0, "trans": 0.05},
                      {"rot": 10.0, "trans": 0.10}]
    n = len(results)
    summary = {"count": n}
    for i, th in enumerate(thresholds):
        ok = sum(1 for r in results
                 if r["rotation_error_deg"] <= th["rot"]
                 and r["translation_error_m"] <= th["trans"])
        summary[f"success_rate_th{i}"] = (ok / n) if n else 0.0
    summary["mean_rot_err"] = float(np.mean([r["rotation_error_deg"]
                                             for r in results])) if n else None
    summary["mean_trans_err"] = float(np.mean([r["translation_error_m"]
                                               for r in results])) if n else None
    return summary


def main():
    parser = argparse.ArgumentParser(description="配准批量评测")
    parser.add_argument("--pairs", required=True, help="帧对清单 pairs.json")
    parser.add_argument("--pose-gt-dir", default="results/preprocess/pose_gt")
    parser.add_argument("--pcd-dir", default="results/preprocess/pcd")
    parser.add_argument("--out", default="results/reg/eval.json")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测前 N 对（调试用）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import open3d as o3d

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[: args.limit]

    results = []
    for p in pairs:
        src = p["src"].replace("/", "_")
        dst = p["dst"].replace("/", "_")
        src_pcd = o3d.io.read_point_cloud(str(Path(args.pcd_dir) / f"{src}.ply"))
        dst_pcd = o3d.io.read_point_cloud(str(Path(args.pcd_dir) / f"{dst}.ply"))
        gt_path = Path(args.pose_gt_dir) / f"{src}_to_{dst}.npy"
        if not gt_path.is_file():
            logger.warning("缺 GT: %s，跳过", gt_path)
            continue
        T_gt = np.load(gt_path)
        try:
            r = evaluate_pair(src_pcd, dst_pcd, T_gt, args.voxel_size)
        except Exception as e:
            logger.warning("%s->%s 失败: %s", p["src"], p["dst"], e)
            continue
        r.update({"src": p["src"], "dst": p["dst"]})
        results.append(r)

    summary = summarize(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    logger.info("评测 %d 对，成功率(5°/5cm)=%.3f, (10°/10cm)=%.3f",
                summary["count"],
                summary.get("success_rate_th0"),
                summary.get("success_rate_th1"))


if __name__ == "__main__":
    main()
