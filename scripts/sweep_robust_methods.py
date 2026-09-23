#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务四：鲁棒粗配准方法对比扫描（FGR / 法向量一致性检查 / 多尺度）。

三种方法（都接同一个改进 ICP 精配准，公平对比）：
1. fgr          ：Open3D Fast Global Registration，调 maximum_correspondence_distance
                  （=1.5×voxel）与 iteration_number（=64），对低重叠/噪声更鲁棒；
2. normal_check ：FPFH+RANSAC 的 checkers 增加
                  CorrespondenceCheckerBasedOnNormal(30°)，要求对应点法向量夹角一致；
3. multiscale   ：两级流水线——0.08m 体素粗配准（RANSAC 500k）→ 原始点云改进 ICP 精配准。

对比基准：任务一/二/三最优组合（FPFH=0.40 / RANSAC=500k / mf=False / voxel=0.03），
结果见 results/registration/sweep_real/voxel/voxel003/（中位 50%，iv30=33%）。

输出（与 sweep_real_registration.py 同构，可复用 analyze_failures.py）：
- {out_dir}/{method}/sweep_results.csv / sweep_summary.json / per_pair_results.csv
- 每方法 3 次重复取中位成功率 + min/max + 按间隔分组

用法：
    python scripts/sweep_robust_methods.py \
        --method fgr --pairs ... --pcd_dir ... --pose_gt_dir ... --repeats 3 \
        --out_dir results/registration/sweep_real/robust
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

# 与任务一/二/三最优组合对齐
V = 0.03          # 主体素（米）
FPFH_R = 0.40     # FPFH 半径（米）
RANSAC_IT = 500000
MUTUAL_FILTER = False
NORMAL_ANGLE_DEG = 30.0

ROT_THRESH_DEG = 5.0
TRANS_THRESH_M = 0.05


def load_pair_clouds(pcd_dir, scene, frame_a, frame_b):
    import open3d as o3d

    def _load(fid):
        p = Path(pcd_dir) / f"{scene}_{fid:06d}.ply"
        if not p.is_file():
            raise FileNotFoundError(f"点云不存在: {p}")
        return o3d.io.read_point_cloud(str(p))

    return _load(frame_a), _load(frame_b)


def pose_error(T_pred, T_gt):
    """旋转角误差（度）与平移误差（米），方向约定同 sweep_real_registration.py。"""
    R_err = T_pred[:3, :3].T @ T_gt[:3, :3]
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.rad2deg(np.arccos(cos_a)))
    trans_m = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))
    return rot_deg, trans_m


def _preproc(pcd, voxel):
    """粗配准用轻量预处理：体素降采样 + 法向量（与 coarse_registration 一致）。"""
    import open3d as o3d
    p = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return p


def _fpfh(pcd):
    import open3d as o3d
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_R, max_nn=100))


def coarse_fgr(source, target):
    """方法 1：FGR 粗配准。返回 (T, n_corr, fitness)。"""
    import open3d as o3d
    src, tgt = _preproc(source, V), _preproc(target, V)
    sf, tf = _fpfh(src), _fpfh(tgt)
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=1.5 * V,
        iteration_number=64)
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src, tgt, sf, tf, opt)
    n_corr = int(len(res.correspondence_set)) if res.correspondence_set is not None else 0
    # FGR 无 fitness 字段，自行计算
    src_t = o3d.geometry.PointCloud(src).transform(res.transformation)
    d = np.asarray(src_t.compute_point_cloud_distance(tgt))
    fitness = float(np.mean(d < 1.5 * V))
    return res.transformation, n_corr, fitness


def coarse_normal_check(source, target):
    """方法 2：RANSAC + 法向量一致性 checker（夹角 < 30°）。"""
    import open3d as o3d
    src, tgt = _preproc(source, V), _preproc(target, V)
    sf, tf = _fpfh(src), _fpfh(tgt)
    thr = 1.5 * V
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf,
        mutual_filter=MUTUAL_FILTER,
        max_correspondence_distance=thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                np.deg2rad(NORMAL_ANGLE_DEG)),  # 单位：弧度
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=RANSAC_IT, confidence=0.999))
    n_corr = int(len(res.correspondence_set)) if res.correspondence_set is not None else 0
    return res.transformation, n_corr, float(res.fitness)


def coarse_multiscale(source, target):
    """方法 3：0.08m 粗配准（RANSAC）→ 返回 T_coarse（精配准在 run_one 统一做）。"""
    import open3d as o3d
    v8 = 0.08
    src, tgt = _preproc(source, v8), _preproc(target, v8)
    sf, tf = _fpfh(src), _fpfh(tgt)
    thr = 1.5 * v8
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf,
        mutual_filter=MUTUAL_FILTER,
        max_correspondence_distance=thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=RANSAC_IT, confidence=0.999))
    n_corr = int(len(res.correspondence_set)) if res.correspondence_set is not None else 0
    return res.transformation, n_corr, float(res.fitness)


METHODS = {
    "fgr": coarse_fgr,
    "normal_check": coarse_normal_check,
    "multiscale": coarse_multiscale,
}


def run_once(method, source, target, T_gt):
    """粗配准（按方法）→ 改进 ICP 精配准（原始点云）→ 误差。"""
    from registration.fine_registration import improved_icp

    t0 = time.perf_counter()
    T_coarse, n_corr, fitness = METHODS[method](source, target)
    T_fine, history = improved_icp(source, target, init_transformation=T_coarse)
    elapsed = time.perf_counter() - t0

    rot_deg, trans_m = pose_error(T_fine, T_gt)
    success = (rot_deg < ROT_THRESH_DEG) and (trans_m < TRANS_THRESH_M)
    return {
        "rot_deg": rot_deg, "trans_m": trans_m,
        "success": success, "time_s": round(elapsed, 3),
        "rmse": history[-1]["inlier_rmse"] if history else None,
        "n_corr": n_corr, "coarse_fitness": fitness,
    }


def run_once_all(pairs, pcd_dir, pose_gt_dir, method):
    """对全部帧对跑一次，返回逐对结果与汇总（结构同 sweep 脚本）。"""
    n_succ = 0
    rot_list, trans_list, rmse_list, time_list, corr_list = [], [], [], [], []
    per_interval = {5: [0, 0], 10: [0, 0], 30: [0, 0]}
    per_pair = []
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
            res = run_once(method, source, target, T_gt)
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
    parser = argparse.ArgumentParser(description="任务四：鲁棒粗配准方法对比")
    parser.add_argument("--method", required=True,
                        choices=["fgr", "normal_check", "multiscale"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--pcd_dir", required=True)
    parser.add_argument("--pose_gt_dir", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out_dir", default="results/registration/sweep_real/robust")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]
    logger.info("方法 %s：%d 个帧对，重复 %d 次", args.method, len(pairs), args.repeats)

    out_dir = Path(args.out_dir) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    rates, per_pair_all = [], []
    for rep in range(1, args.repeats + 1):
        logger.info("[%s 重复 %d/%d] 开始", args.method, rep, args.repeats)
        res = run_once_all(pairs, args.pcd_dir, args.pose_gt_dir, args.method)
        rate = 100.0 * res["n_succ"] / res["n_run"]
        rates.append(rate)
        for p in res["per_pair"]:
            per_pair_all.append({**p, "method": args.method})
        logger.info("[%s 重复 %d/%d] 成功率 %.1f%% (%d/%d)",
                    args.method, rep, args.repeats, rate,
                    res["n_succ"], res["n_run"])

    # 汇总（中位成功率 + min/max + 按间隔）
    med = float(np.median(rates))
    mn, mx = min(rates), max(rates)
    iv_sum = {5: [0, 0], 10: [0, 0], 30: [0, 0]}
    # 用中位那次重复的逐对明细做按间隔统计
    med_idx = sorted(range(len(rates)), key=lambda i: rates[i])[len(rates) // 2]
    rep_rows = per_pair_all[med_idx * len(pairs):(med_idx + 1) * len(pairs)] \
        if len(per_pair_all) == len(pairs) * args.repeats else per_pair_all
    for p in rep_rows:
        iv = p["interval"]
        iv_sum[iv][1] += 1
        iv_sum[iv][0] += int(p["success"])

    summary = {
        "method": args.method,
        "repeats": args.repeats,
        "rates": rates,
        "median_success_pct": med,
        "min_pct": mn, "max_pct": mx,
        "per_interval": {str(k): v for k, v in iv_sum.items()},
    }
    with open(out_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "repeat", "success_pct"])
        for i, r in enumerate(rates, 1):
            w.writerow([args.method, i, r])

    with open(out_dir / "per_pair_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "interval", "method", "success",
                    "rot_deg", "trans_m", "rmse", "n_corr", "time_s"])
        for p in per_pair_all:
            w.writerow([p["pair_id"], p["interval"], args.method,
                        p["success"], round(p["rot_deg"], 4),
                        round(p["trans_m"], 4),
                        round(p["rmse"], 4) if p["rmse"] is not None else "",
                        p["n_corr"], p["time_s"]])

    print("\n=== %s（%d 次重复）===" % (args.method, args.repeats))
    print("单次成功率: %s" % ["%.1f%%" % r for r in rates])
    print("中位 %.1f%%（min %.1f%% / max %.1f%%）" % (med, mn, mx))
    for iv, (ok, total) in sorted(iv_sum.items()):
        if total:
            print("  间隔 %d: %d/%d = %.0f%%" % (iv, ok, total, 100.0 * ok / total))
    print("summary -> %s" % (out_dir / "sweep_summary.json"))


if __name__ == "__main__":
    main()
