#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值敏感性分析：平移/旋转阈值对成功率的影响（理解方法行为，非调参）。

数据来源：最优组合（voxel=0.03 / FPFH=0.40 / RANSAC=500k / mf=False）最后一次
repeat 的 20 对逐对明细（results/registration/sweep_real/voxel/voxel003/
per_pair_results.csv）。同一批 rot/trans 结果在不同阈值下重新判定 success，
观察成功率随阈值的变化曲线。

说明：这是敏感性分析，不是调参——真实数据上的成功率必须在固定阈值口径
（旋转 5° 且平移 0.05m）下报告；本分析只用于理解方法行为与阈值边缘案例占比。

输出：
- results/figures/threshold_sensitivity.png（平移阈值曲线 + 旋转阈值曲线，双子图）
- 打印敏感性表格
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 固定口径（evaluate_registration.py 一致）
BASE_ROT = 5.0
BASE_TRANS = 0.05


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            rows.append({
                "pair_id": d["pair_id"],
                "interval": int(d["interval"]),
                "rot_deg": float(d["rot_deg"]),
                "trans_m": float(d["trans_m"]),
            })
    return rows


def success_rate(rows, rot_thr, trans_thr):
    ok = sum(1 for r in rows if r["rot_deg"] < rot_thr and r["trans_m"] < trans_thr)
    return 100.0 * ok / len(rows)


def rate_by_interval(rows, rot_thr, trans_thr):
    out = {}
    for iv in sorted(set(r["interval"] for r in rows)):
        sel = [r for r in rows if r["interval"] == iv]
        ok = sum(1 for r in sel if r["rot_deg"] < rot_thr and r["trans_m"] < trans_thr)
        out[iv] = 100.0 * ok / len(sel)
    return out


def main():
    parser = argparse.ArgumentParser(description="阈值敏感性分析")
    parser.add_argument("--csv",
                        default="results/registration/sweep_real/voxel/voxel003/per_pair_results.csv")
    parser.add_argument("--fig_dir", default="results/figures")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    n = len(rows)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 平移阈值扫描（旋转固定 5°）
    trans_thrs = [0.03, 0.05, 0.08, 0.10, 0.15]
    trans_rates = [success_rate(rows, BASE_ROT, t) for t in trans_thrs]
    trans_iv = {t: rate_by_interval(rows, BASE_ROT, t) for t in trans_thrs}

    # 旋转阈值扫描（平移固定 0.05m）
    rot_thrs = [0.5, 1.0, 2.0, 5.0, 10.0]
    rot_rates = [success_rate(rows, r, BASE_TRANS) for r in rot_thrs]
    rot_iv = {r: rate_by_interval(rows, r, BASE_TRANS) for r in rot_thrs}

    print("=== 平移阈值扫描（旋转固定 5°） ===")
    print("%-8s %-8s %-8s %-8s %-8s" % ("阈值", "总成功率", "iv5", "iv10", "iv30"))
    for t, rate in zip(trans_thrs, trans_rates):
        iv = trans_iv[t]
        print("%-8.2f %-8.1f%% %-8.1f%% %-8.1f%% %-8.1f%%" % (t, rate, iv[5], iv[10], iv[30]))

    print("\n=== 旋转阈值扫描（平移固定 0.05m） ===")
    print("%-8s %-8s %-8s %-8s %-8s" % ("阈值", "总成功率", "iv5", "iv10", "iv30"))
    for r, rate in zip(rot_thrs, rot_rates):
        iv = rot_iv[r]
        print("%-8.1f %-8.1f%% %-8.1f%% %-8.1f%% %-8.1f%%" % (r, rate, iv[5], iv[10], iv[30]))

    # 阈值说明（写进图标题）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(trans_thrs, trans_rates, "o-", color="#e74c3c", lw=2)
    for iv in (5, 10, 30):
        ax.plot(trans_thrs, [trans_iv[t][iv] for t in trans_thrs], "o--", lw=1.2,
                label="间隔 %d" % iv)
    ax.axvline(0.05, color="gray", ls="--", lw=1)
    ax.text(0.052, 5, "0.05m (固定口径)", fontsize=8, color="gray")
    ax.set_xlabel("平移阈值 (m)（旋转固定 5°）")
    ax.set_ylabel("成功率 (%)")
    ax.set_title("平移阈值敏感性")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1]
    ax.plot(rot_thrs, rot_rates, "o-", color="#3498db", lw=2)
    for iv in (5, 10, 30):
        ax.plot(rot_thrs, [rot_iv[r][iv] for r in rot_thrs], "o--", lw=1.2,
                label="间隔 %d" % iv)
    ax.axvline(5.0, color="gray", ls="--", lw=1)
    ax.text(5.2, 5, "5° (固定口径)", fontsize=8, color="gray")
    ax.set_xlabel("旋转阈值 (度)（平移固定 0.05m）")
    ax.set_ylabel("成功率 (%)")
    ax.set_title("旋转阈值敏感性")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle("阈值敏感性分析（20 对真实 SUN3D，最优组合单次明细；敏感性≠调参）",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = fig_dir / "threshold_sensitivity.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("\n图 -> %s" % out_png)


if __name__ == "__main__":
    main()
