#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""失败案例分析：按间隔统计失败率、分类失败原因、统计成功案例精度。

数据来源：
- 帧对清单：results/preprocess/pairs/pairs_real20.json（20 对，含 interval）
- 最优组合逐对明细：results/registration/sweep_real/voxel/voxel003/per_pair_results.csv
  （voxel=0.03 / FPFH=0.40 / RANSAC=500k / mutual_filter=False，最后一次 repeat）
- 基线逐对明细：results/registration/eval_real/detail.csv（60 对 × 4 方法，取 improved_icp
  并筛出 20 对评测子集）
- 真值：results/preprocess/pose_gt_real20/*.txt（3x4 相机位姿）

失败原因分类规则（对每个失败对，按优先级取主因）：
  E. 次成功     ：旋转 < 5° 或 平移 < 0.05m（一个指标达标、另一个差一点）
  D. 初始位姿太远：真值旋转 > 30° 或 平移 > 1.0m
  A. 重叠不足   ：真值对齐后重叠度 < 20%
  C. 纹理重复   ：n_corr 低（< 5000）且重叠度/位姿正常 → FPFH 区分度不足
  B. 深度噪声   ：RMSE > 0.05m 或点云点数异常少
  其他/原因不明：无法归入以上类

输出：
- results/figures/failure_rate_by_interval.png（各间隔失败率柱状图，基线 vs 最优）
- results/figures/failure_reason_pie.png（失败原因分布饼图）
- results/figures/failure_rot_trans_scatter.png（失败对 rot-trans 散点图）
- 统计结果打印 + 结果 JSON
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

# 成功判定阈值（与 evaluate_registration.py 一致）
ROT_THRESH_DEG = 5.0
TRANS_THRESH_M = 0.05


def load_pair_list(pairs_json):
    with open(pairs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = {}
    for p in data["pairs"]:
        pairs[p["pair_id"]] = {"interval": p.get("interval", 0)}
    return pairs


def load_gt_pose(txt_path):
    """读变换矩阵：支持 3x4（[R|t]）或 4x4（齐次），统一返回 4x4。"""
    arr = np.loadtxt(txt_path)
    if arr.shape == (3, 4):
        return np.vstack([arr, [0.0, 0.0, 0.0, 1.0]])
    if arr.shape == (4, 4):
        return arr
    raise ValueError("位姿文件应为 3x4 或 4x4 矩阵: %s (shape=%s)" % (txt_path, arr.shape))


def compute_init_pose_diff(pose_a, pose_b):
    """真值 T_AB = inv(Pose_A) @ Pose_B（B→A）。返回旋转角(度)、平移距离(m)。"""
    T_ab = np.linalg.inv(pose_a) @ pose_b
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]
    # 旋转角 = acos((trace-1)/2)
    cos_ang = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(cos_ang))
    trans_m = float(np.linalg.norm(t))
    return rot_deg, trans_m, T_ab


def load_downsampled(pcd_path, voxel=0.05):
    """加载点云并下采样，返回 (N,3) ndarray。"""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        return np.zeros((0, 3))
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)


def compute_overlap(pts_a, pts_b, T_ab, thr=0.05):
    """用真值 T_ab（B→A）把 B 变换到 A 坐标系，统计 A 中每点最近邻 < thr 的比例。"""
    from scipy.spatial import cKDTree
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    pts_b_in_a = (T_ab[:3, :3] @ pts_b.T).T + T_ab[:3, 3]
    tree = cKDTree(pts_a)
    d, _ = tree.query(pts_b_in_a, k=1, workers=-1)
    return float(np.mean(d < thr))


def classify_failure(item, init_rot, init_trans, overlap, n_points_ok):
    """返回 (主因类别, 说明)。优先级：次成功 > 平移严重错配 > 初始位姿 > 重叠 > 纹理 > 噪声。"""
    rot, trans = item["rot_deg"], item["trans_m"]
    # E. 次成功：旋转 < 5° 但平移略超阈（0.05~0.15m）→ 精度边缘，粗配准基本成功
    if rot < ROT_THRESH_DEG and TRANS_THRESH_M <= trans < 0.15:
        return "E.次成功(精度边缘)", "旋转%.1f°达标，平移%.3fm略超阈（0.05m）" % (rot, trans)
    # F. 旋转对平移错配：旋转 < 5° 但平移严重超阈（≥0.15m）→ 低重叠/对称结构致平移漂移
    if rot < ROT_THRESH_DEG and trans >= 0.15:
        return "F.平移严重错配", "旋转%.1f°达标但平移%.3fm（重叠%.0f%%/初始%.1f°）" % (
            rot, trans, overlap * 100, init_rot)
    # D. 初始位姿太远
    if init_rot > 30.0 or init_trans > 1.0:
        return "D.初始位姿太远", "真值旋转%.1f°平移%.2fm" % (init_rot, init_trans)
    # A. 重叠不足
    if overlap < 0.20:
        return "A.重叠不足", "重叠度%.0f%%" % (overlap * 100)
    # C. 纹理重复（特征区分度不足）
    if item.get("n_corr", 0) < 5000:
        return "C.纹理重复", "内点数%d（FPFH 区分度不足）" % item.get("n_corr", 0)
    # B. 深度噪声
    if item.get("rmse", 0) > 0.05 or not n_points_ok:
        return "B.深度噪声", "RMSE=%.3fm" % item.get("rmse", 0)
    return "Z.原因不明", "无"


def main():
    parser = argparse.ArgumentParser(description="失败案例分析")
    parser.add_argument("--pairs", default="results/preprocess/pairs/pairs_real20.json")
    parser.add_argument("--pcd_dir", default="results/preprocess/pcd_real")
    parser.add_argument("--pose_gt_dir", default="results/preprocess/pose_gt_real20")
    parser.add_argument("--best_csv",
                        default="results/registration/sweep_real/voxel/voxel003/per_pair_results.csv",
                        help="最优组合逐对明细")
    parser.add_argument("--baseline_csv", default="results/registration/eval_real/detail.csv")
    parser.add_argument("--fig_dir", default="results/figures")
    parser.add_argument("--out_json", default="results/figures/failure_analysis.json")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir = Path(args.pcd_dir)

    pairs = load_pair_list(args.pairs)
    n_pairs = len(pairs)
    print("评测帧对总数：%d" % n_pairs)

    # ---- 读最优组合逐对明细（最后一次 repeat 的 20 对） ----
    best = {}
    with open(args.best_csv) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            best[d["pair_id"]] = {
                "success": d["success"].strip().lower() in ("1", "true", "yes"),
                "rot_deg": float(d["rot_deg"]),
                "trans_m": float(d["trans_m"]),
                "rmse": float(d["rmse"]),
                "n_corr": int(float(d["n_corr"])),
                "time_s": float(d["time_s"]),
            }

    # ---- 读基线明细（improved_icp，筛 20 对） ----
    baseline = {}
    with open(args.baseline_csv) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            if d["method"] != "improved_icp":
                continue
            if d["pair_id"] not in pairs:
                continue
            baseline[d["pair_id"]] = {
                "success": d["success"] == "1",
                "rot_deg": float(d["rot_deg"]),
                "trans_m": float(d["trans_m"]),
                "rmse": float(d["rmse"]),
            }

    # ---- 加载点云与真值，计算初始位姿差异与重叠度 ----
    # 真值文件：pose_gt_dir/{pair_id}.txt = 4x4 T_AB（B→A），直接读
    cache_pcd = {}
    info = {}
    for pid in pairs:
        parts = pid.split("_")
        fa, fb = parts[-2], parts[-1]
        pcd_a_path = pcd_dir / ("mit_studyroom_%s.ply" % fa)
        pcd_b_path = pcd_dir / ("mit_studyroom_%s.ply" % fb)
        gt_path = Path(args.pose_gt_dir) / ("%s.txt" % pid)
        T_ab = load_gt_pose(gt_path)
        init_rot, init_trans, _ = compute_init_pose_diff_from_T(T_ab)

        pts_a = cache_pcd.get(pcd_a_path)
        if pts_a is None:
            pts_a = load_downsampled(pcd_a_path)
            cache_pcd[pcd_a_path] = pts_a
        pts_b = cache_pcd.get(pcd_b_path)
        if pts_b is None:
            pts_b = load_downsampled(pcd_b_path)
            cache_pcd[pcd_b_path] = pts_b

        overlap = compute_overlap(pts_a, pts_b, T_ab)
        info[pid] = {
            "interval": pairs[pid]["interval"],
            "init_rot_deg": init_rot,
            "init_trans_m": init_trans,
            "overlap": overlap,
            "n_pts_a": len(pts_a),
            "n_pts_b": len(pts_b),
        }

    # ---- 统计 ----
    rows = []
    for pid in pairs:
        b = best.get(pid)
        base = baseline.get(pid)
        row = {"pair_id": pid, "interval": info[pid]["interval"],
               "init_rot_deg": info[pid]["init_rot_deg"],
               "init_trans_m": info[pid]["init_trans_m"],
               "overlap": info[pid]["overlap"],
               "best_success": b["success"] if b else None,
               "best_rot": b["rot_deg"] if b else None,
               "best_trans": b["trans_m"] if b else None,
               "best_rmse": b["rmse"] if b else None,
               "best_n_corr": b["n_corr"] if b else None,
               "base_success": base["success"] if base else None}
        if b and not b["success"]:
            cat, why = classify_failure(b, info[pid]["init_rot_deg"],
                                        info[pid]["init_trans_m"],
                                        info[pid]["overlap"],
                                        info[pid]["n_pts_a"] > 5000)
            row["fail_cat"] = cat
            row["fail_why"] = why
        rows.append(row)

    # 各间隔成功率（最优 vs 基线）
    print("\n=== 各间隔成功率（最优组合 vs 基线 improved_icp，20 对） ===")
    print("%-6s %-10s %-10s %-6s %-6s" % ("间隔", "最优", "基线", "帧对数", "最优失败对"))
    interval_stats = {}
    for iv in sorted(set(info[p]["interval"] for p in pairs)):
        sel = [r for r in rows if r["interval"] == iv]
        n = len(sel)
        ok_b = sum(1 for r in sel if r["best_success"])
        ok_base = sum(1 for r in sel if r["base_success"])
        failed = [r["pair_id"] for r in sel if not r["best_success"]]
        interval_stats[iv] = {"n": n, "best_ok": ok_b, "base_ok": ok_base,
                              "failed": failed}
        print("%-6d %-10s %-10s %-6d %s" % (
            iv, "%d/%d=%.0f%%" % (ok_b, n, 100.0 * ok_b / n),
            "%d/%d=%.0f%%" % (ok_base, n, 100.0 * ok_base / n), n,
            ",".join(failed) if failed else "-"))

    # 失败原因分布
    fails = [r for r in rows if r.get("fail_cat")]
    print("\n=== 失败原因分布（%d 个失败对） ===" % len(fails))
    from collections import Counter
    cat_counter = Counter(r["fail_cat"] for r in fails)
    for cat, cnt in cat_counter.most_common():
        print("  %s: %d" % (cat, cnt))
    for r in fails:
        print("  %s iv=%d %s" % (r["pair_id"], r["interval"], r["fail_why"]))

    # 成功案例精度
    oks = [r for r in rows if r["best_success"]]
    print("\n=== 成功案例精度（最优组合） ===")
    if oks:
        print("成功 %d 对：旋转误差 中位 %.2f°（min %.2f / max %.2f），"
              "平移 中位 %.3fm（min %.3f / max %.3f）" % (
                  len(oks),
                  float(np.median([r["best_rot"] for r in oks])),
                  float(np.min([r["best_rot"] for r in oks])),
                  float(np.max([r["best_rot"] for r in oks])),
                  float(np.median([r["best_trans"] for r in oks])),
                  float(np.min([r["best_trans"] for r in oks])),
                  float(np.max([r["best_trans"] for r in oks]))))

    # ---- 图表 ----
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    ivs = sorted(interval_stats)
    best_rate = [100.0 * interval_stats[iv]["best_ok"] / interval_stats[iv]["n"] for iv in ivs]
    base_rate = [100.0 * interval_stats[iv]["base_ok"] / interval_stats[iv]["n"] for iv in ivs]

    # 图1：各间隔失败率柱状图（用失败率展示，含基线与最优对比）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ivs))
    w = 0.35
    ax.bar(x - w / 2, [100 - v for v in best_rate], w, label="最优组合 (voxel0.03/FPFH0.40/500k/mfFalse)")
    ax.bar(x + w / 2, [100 - v for v in base_rate], w, label="基线 (voxel0.02/FPFH0.25/100k)")
    ax.set_xticks(x)
    ax.set_xticklabels(["间隔 %d" % iv for iv in ivs])
    ax.set_ylabel("失败率 (%)")
    ax.set_title("各间隔失败率：最优组合 vs 基线（20 对真实 SUN3D）")
    for i, (bv, br) in enumerate(zip([100 - v for v in best_rate], [100 - v for v in base_rate])):
        ax.text(i - w / 2, bv + 1, "%.0f%%" % bv, ha="center", fontsize=9)
        ax.text(i + w / 2, br + 1, "%.0f%%" % br, ha="center", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_rate_by_interval.png", dpi=150)
    plt.close(fig)
    print("\n图1 -> %s" % (fig_dir / "failure_rate_by_interval.png"))

    # 图2：失败原因饼图
    if fails:
        labels = list(cat_counter.keys())
        sizes = list(cat_counter.values())
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90,
               colors=["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"])
        ax.set_title("失败原因分布（%d 个失败对，最优组合）" % len(fails))
        fig.tight_layout()
        fig.savefig(fig_dir / "failure_reason_pie.png", dpi=150)
        plt.close(fig)
        print("图2 -> %s" % (fig_dir / "failure_reason_pie.png"))

    # 图3：失败对 rot-trans 散点图（含阈值线）
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if r["best_rot"] is None:
            continue
        if r["best_success"]:
            ax.scatter(r["best_rot"], r["best_trans"], c="#2ecc71", marker="o", label="成功" if r["pair_id"] == rows[0]["pair_id"] else "")
        else:
            ax.scatter(r["best_rot"], r["best_trans"], c="#e74c3c", marker="x", label="失败" if r["pair_id"] == rows[0]["pair_id"] else "")
    ax.axvline(ROT_THRESH_DEG, color="gray", ls="--", lw=1)
    ax.axhline(TRANS_THRESH_M, color="gray", ls="--", lw=1)
    ax.set_xlabel("旋转误差 (度)")
    ax.set_ylabel("平移误差 (m)")
    ax.set_title("20 对帧对配准结果分布（绿=成功 红=失败）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_rot_trans_scatter.png", dpi=150)
    plt.close(fig)
    print("图3 -> %s" % (fig_dir / "failure_rot_trans_scatter.png"))

    # 结果 JSON
    out = {
        "n_pairs": n_pairs,
        "interval_stats": interval_stats,
        "failure_cats": dict(cat_counter),
        "rows": rows,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果 JSON -> %s" % args.out_json)


def compute_init_pose_diff_from_T(T_ab):
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]
    cos_ang = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(cos_ang))
    return rot_deg, float(np.linalg.norm(t)), T_ab


if __name__ == "__main__":
    main()
