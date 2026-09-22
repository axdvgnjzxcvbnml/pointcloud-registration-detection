#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_frame_pairs.py — SUN3D 帧对采样（第二批交付）

功能
----
    在 SUN3D 场景序列中按指定帧间隔枚举帧对候选：
      - 间隔：5 / 10 / 30 帧
      - 目标总数：400（300 ~ 500）
      - 可选：按重叠率过滤（默认关闭，避免需要先跑配准）
    输出 pairs.json 供 compute_pose_gt 与配准评测使用。

用法
----
    python preprocess/sample_frame_pairs.py \
        --scene-dir data/SUN3D/scene_001 \
        --intervals 5 10 30 --target-num 400 \
        --out results/preprocess/pairs.json
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def enumerate_candidates(frame_names, intervals=(5, 10, 30)):
    """按间隔枚举帧对候选。

    Args:
        frame_names: 按时间序排列的帧名列表（已排序）。
        intervals: 帧间隔集合。
    Returns:
        list[{"src", "dst", "interval"}]，dst = 序号 + interval。
    """
    n = len(frame_names)
    pairs = []
    for i in range(n):
        for d in intervals:
            j = i + d
            if j < n:
                pairs.append({"src": frame_names[i], "dst": frame_names[j],
                              "interval": int(d)})
    return pairs


def filter_by_overlap(candidates, overlap_fn, threshold=0.3):
    """按重叠率过滤帧对（可选；overlap_fn 返回 0~1）。

    默认不启用——需要先跑配准或读取位姿才能算重叠率，
    故骨架只保留接口，由用户在服务器上实现。
    """
    out = []
    for c in candidates:
        o = overlap_fn(c["src"], c["dst"])
        if o >= threshold:
            c["overlap"] = o
            out.append(c)
    return out


def balance_to_target(pairs, target_num=400):
    """按间隔均衡采样到目标数量（不足时全部保留）。"""
    import numpy as np

    if len(pairs) <= target_num:
        return pairs
    by_interval = {}
    for p in pairs:
        by_interval.setdefault(p["interval"], []).append(p)
    picked = []
    rng = np.random.default_rng(2024)
    intervals = sorted(by_interval)
    per = target_num // len(intervals)
    for iv in intervals:
        arr = by_interval[iv]
        picked.extend(rng.choice(arr, size=min(per, len(arr)), replace=False).tolist())
    # 补足剩余额度
    rest = [p for p in pairs if p not in picked]
    need = target_num - len(picked)
    if need > 0 and rest:
        picked.extend(rng.choice(rest, size=min(need, len(rest)),
                                 replace=False).tolist())
    picked.sort(key=lambda p: (p["src"], p["dst"]))
    return picked


def main():
    parser = argparse.ArgumentParser(description="SUN3D 帧对采样")
    parser.add_argument("--scene-dir", required=True,
                        help="SUN3D 场景目录（含 frames_index 或 *.npz）")
    parser.add_argument("--intervals", nargs="+", type=int, default=[5, 10, 30],
                        help="帧间隔（默认 5 10 30）")
    parser.add_argument("--target-num", type=int, default=400,
                        help="目标帧对数（默认 400）")
    parser.add_argument("--out", default="results/preprocess/pairs.json")
    parser.add_argument("--frame-index", default=None,
                        help="可选：frames_index.json（优先级高于目录扫描）")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import numpy as np
    np.random.seed(args.seed)

    # 帧名列表：优先读 frames_index.json
    if args.frame_index:
        with open(args.frame_index, "r", encoding="utf-8") as f:
            frames = json.load(f)
        frame_names = [fr["name"] for fr in frames]
    else:
        frame_names = sorted(p.stem for p in Path(args.scene_dir).glob("*.npz"))
        if not frame_names:
            frame_names = sorted(p.stem for p in Path(args.scene_dir).glob("*.ply"))
    if not frame_names:
        raise FileNotFoundError(f"场景目录无帧文件: {args.scene_dir}")

    candidates = enumerate_candidates(frame_names, tuple(args.intervals))
    picked = balance_to_target(candidates, args.target_num)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
    logger.info("候选 %d → 采样 %d 帧对 -> %s", len(candidates), len(picked), out)


if __name__ == "__main__":
    main()
