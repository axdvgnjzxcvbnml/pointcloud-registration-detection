#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重叠率量化分析（CPU 纯计算，2026-09-23 补做任务三）。

数据来源：results/figures/failure_analysis.json（由 analyze_failures.py 生成，
overlap 字段 = 用真值 T_AB 把 B 变换到 A 坐标系后，近邻 < 0.05m 的点比例，
在 0.05m 体素下采样点云上计算；逐对 interval / best_success / best_trans /
best_rot 一并记录）。

输出：
- results/figures/overlap_vs_success.png
    左子图：重叠率 vs 平移误差散点（绿=成功，红=失败，含 0.05m 阈值线）；
    右子图：按重叠率分箱的成功率柱状图（标注每箱帧对数）。
- results/figures/overlap_analysis.json   逐对重叠率 + 按间隔平均重叠率 + 分箱统计
- results/figures/overlap_by_interval.csv 按间隔平均重叠率表

用法：
    python scripts/overlap_analysis.py \
        --json results/figures/failure_analysis.json \
        --fig_dir results/figures
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TRANS_THRESH_M = 0.05  # 与 evaluate_registration.py 一致


def main():
    parser = argparse.ArgumentParser(description="重叠率量化分析")
    parser.add_argument("--json", default="results/figures/failure_analysis.json")
    parser.add_argument("--fig_dir", default="results/figures")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    n = len(rows)
    print("评测帧对总数：%d" % n)

    # ---- 按间隔统计平均重叠率 ----
    by_interval = defaultdict(list)
    for r in rows:
        by_interval[r["interval"]].append(r["overlap"])

    print("\n=== 按间隔的平均重叠率 ===")
    print("%-8s %-10s %-10s %-10s" % ("间隔", "帧对数", "平均重叠率", "范围"))
    iv_stats = {}
    for iv in sorted(by_interval):
        vals = by_interval[iv]
        mean = float(np.mean(vals))
        iv_stats[str(iv)] = {
            "n": len(vals),
            "mean_overlap": round(mean, 4),
            "min_overlap": round(float(np.min(vals)), 4),
            "max_overlap": round(float(np.max(vals)), 4),
        }
        print("%-8d %-10d %-10.2f %-10s" % (
            iv, len(vals), mean,
            "%.2f-%.2f" % (np.min(vals), np.max(vals))))

    # ---- 按重叠率分箱统计成功率 ----
    bins = [(0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]
    bin_stats = []
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["overlap"] < hi]
        if not sel:
            continue
        ok = sum(1 for r in sel if r["best_success"])
        bin_stats.append({
            "lo": lo, "hi": hi if hi <= 1.0 else 1.0,
            "n": len(sel), "ok": ok,
            "success_pct": round(100.0 * ok / len(sel), 1),
        })
        print("\n重叠率 [%.2f, %.2f)：%d 对，成功 %d，成功率 %.1f%%"
              % (lo, min(hi, 1.0), len(sel), ok, 100.0 * ok / len(sel)))

    # ---- 相关性：重叠率 vs 平移误差（成功子集） ----
    succ = [r for r in rows if r["best_success"]]
    fail = [r for r in rows if not r["best_success"]]
    if succ:
        corr = float(np.corrcoef([r["overlap"] for r in succ],
                                 [r["best_trans"] for r in succ])[0, 1])
        print("\n成功子集内 重叠率-平移误差 相关系数：%.3f" % corr)
    else:
        corr = None

    # ---- 画图：左=重叠率vs平移误差散点，右=分箱成功率 ----
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter([r["overlap"] for r in succ], [r["best_trans"] for r in succ],
               c="#2ecc71", marker="o", s=45, label="成功 (%d)" % len(succ))
    ax.scatter([r["overlap"] for r in fail], [r["best_trans"] for r in fail],
               c="#e74c3c", marker="x", s=55, label="失败 (%d)" % len(fail))
    ax.axhline(TRANS_THRESH_M, color="gray", ls="--", lw=1)
    ax.text(0.03, 0.052, "平移阈值 0.05m", fontsize=8, color="gray")
    ax.set_xlabel("重叠率（真值 T_AB 对齐后近邻比例）")
    ax.set_ylabel("平移误差 (m)")
    ax.set_title("重叠率 vs 平移误差（成功/失败）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["%.0f-%.0f%%" % (b["lo"] * 100, b["hi"] * 100) for b in bin_stats]
    rates = [b["success_pct"] for b in bin_stats]
    counts = [b["n"] for b in bin_stats]
    bars = ax.bar(labels, rates, color="#3498db", alpha=0.85)
    for bar, cnt, rate in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2,
                "%d对\n%.0f%%" % (cnt, rate), ha="center", fontsize=8)
    ax.set_ylabel("成功率 (%)")
    ax.set_xlabel("重叠率分箱")
    ax.set_title("成功率 vs 重叠率分箱（20 对真实 SUN3D）")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("重叠率量化：FPFH+RANSAC 成功率随重叠率下降（最优组合，20 对）",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = Path(args.fig_dir) / "overlap_vs_success.png"
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("\n图 -> %s" % out_png)

    # ---- CSV / JSON ----
    csv_path = Path(args.fig_dir) / "overlap_by_interval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interval", "n_pairs", "mean_overlap", "min_overlap",
                    "max_overlap"])
        for iv in sorted(iv_stats, key=int):
            s = iv_stats[iv]
            w.writerow([iv, s["n"], s["mean_overlap"], s["min_overlap"],
                        s["max_overlap"]])
    print("按间隔重叠率表 -> %s" % csv_path)

    out = {
        "n_pairs": n,
        "by_interval": iv_stats,
        "bins": bin_stats,
        "corr_overlap_trans_success": corr,
        "rows": [{
            "pair_id": r["pair_id"], "interval": r["interval"],
            "overlap": r["overlap"], "best_success": r["best_success"],
            "best_trans": r["best_trans"], "best_rot": r["best_rot"],
        } for r in rows],
    }
    out_json = Path(args.fig_dir) / "overlap_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("结果 JSON -> %s" % out_json)


if __name__ == "__main__":
    main()
