#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_frame_pairs.py — SUN3D 序列帧对采样（第二批交付）

功能
----
    按场景名分组，将每个场景的帧按帧号升序排列，
    以间隔 5 / 10 / 30 帧抽取“相邻帧对”（i, i+interval），
    全局平衡采样到目标数量（默认 400 对，区间 300~500），
    输出配对清单 json，供 compute_pose_gt.py / registration 模块使用。

帧命名约定（SUN3D 常见形式，可配置正则）
----------------------------------------
    frame-000000.color.jpg / frame-000000.depth.pgm
    本脚本只按文件名中的帧号排序，不解析图像内容。

用法
----
    python preprocess/sample_frame_pairs.py \
        --sun3d_dir data/SUN3D \
        --out_path results/preprocess/pairs/pairs.json \
        --intervals 5,10,30 --target_num 400 --seed 2024
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# SUN3D 帧名中的帧号正则（可按实际命名修改，如 'frame-000000'）
FRAME_RE = re.compile(r"frame-(\d+)")


def discover_scene_frames(scene_root):
    """扫描单个场景目录，返回按帧号升序的 [(frame_idx, Path), ...]。

    以帧名中最先出现的 'frame-NNNNNN' 为帧号；找不到任何帧则返回空列表。
    """
    entries = []
    for p in sorted(scene_root.rglob("*")):
        if not p.is_file():
            continue
        m = FRAME_RE.search(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0])
    return entries


def build_candidate_pairs(frame_indices, intervals):
    """生成候选帧对：对每个间隔取 (i, i+interval) 的帧对。

    间隔由帧号差值定义；要求两端帧都存在（自动跳过缺帧/越界）。

    Args:
        frame_indices: discover_scene_frames 的返回值。
        intervals: 帧间隔列表，如 [5, 10, 30]。
    Returns:
        list[dict]，每个元素:
            {"interval": int, "frame_a": int, "frame_b": int}
    """
    idx_list = sorted({f[0] for f in frame_indices})  # 帧号去重（image/depth/extrinsics 同名文件）
    idx_set = set(idx_list)

    candidates = []
    for interval in intervals:
        for idx in idx_list:
            nxt = idx + interval
            if nxt in idx_set:
                candidates.append({
                    "interval": interval,
                    "frame_a": idx,
                    "frame_b": nxt,
                })
    return candidates


def sample_pairs(candidates, target_num, seed=2024, per_interval_ratio=None):
    """从候选帧对中平衡采样到 target_num 对。

    策略：
      1. 按间隔分组，每组分配 target_num × ratio 的数量；
      2. 组内随机无放回采样；某组候选不足则取全部；
      3. 总量不足时，用剩余候选补足。
    """
    rng = np.random.default_rng(seed)
    intervals = sorted({c["interval"] for c in candidates})
    if per_interval_ratio is None:
        per_interval_ratio = [1.0 / len(intervals)] * len(intervals)

    chosen = []
    for interval, ratio in zip(intervals, per_interval_ratio):
        group = [c for c in candidates if c["interval"] == interval]
        n = int(round(target_num * ratio))
        picked = rng.choice(len(group), size=min(n, len(group)), replace=False)
        chosen.extend(group[int(i)] for i in picked)

    # 补足：若整体不足目标数量，用其余候选补足
    if len(chosen) < target_num:
        used_ids = {id(c) for c in chosen}
        rest = [c for c in candidates if id(c) not in used_ids]
        need = target_num - len(chosen)
        picked = rng.choice(len(rest), size=min(need, len(rest)), replace=False)
        chosen.extend(rest[int(i)] for i in picked)

    return chosen


def filter_by_overlap(pairs, pcd_dir, min_overlap=0.3):
    """按点云重叠率过滤帧对（可选）。

    说明：计算重叠率需要先反投影点云（depth_to_pointcloud.py），
    开销较大，默认跳过。如需启用：
      1. 先生成所有帧点云到 pcd_dir；
      2. 对每对帧用体素下采样 + 最近邻统计重叠点占比。
    此处为占位实现（直接返回原列表），请按实际需求补全。
    """
    logger.warning("filter_by_overlap 为占位实现（TODO），返回全部帧对")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="SUN3D 帧对采样")
    parser.add_argument("--sun3d_dir", default="data/SUN3D",
                        help="SUN3D 数据根目录（含各场景子目录）")
    parser.add_argument("--out_path", default="results/preprocess/pairs/pairs.json",
                        help="配对清单输出路径")
    parser.add_argument("--intervals", default="5,10,30",
                        help="逗号分隔的帧间隔，如 5,10,30")
    parser.add_argument("--target_num", type=int, default=400,
                        help="目标帧对数量（建议 300~500）")
    parser.add_argument("--scene_list", default=None,
                        help="逗号分隔的场景名子集，None=全部")
    parser.add_argument("--overlap_pcd_dir", default=None,
                        help="点云目录（启用重叠率过滤时提供，占位）")
    parser.add_argument("--overlap_threshold", type=float, default=0.3,
                        help="重叠率下限（0~1）")
    parser.add_argument("--seed", type=int, default=2024,
                        help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]
    sun3d_dir = Path(args.sun3d_dir)
    scene_names = [s.name for s in sorted(sun3d_dir.iterdir())
                   if s.is_dir()] if sun3d_dir.is_dir() else []
    if args.scene_list:
        scene_names = [s for s in scene_names if s in args.scene_list.split(",")]
    logger.info("场景数: %d, 间隔: %s, 目标对数: %d",
                len(scene_names), intervals, args.target_num)

    # 各场景分别构造候选帧对
    all_candidates = []
    for scene in scene_names:
        frame_indices = discover_scene_frames(sun3d_dir / scene)
        cands = build_candidate_pairs(frame_indices, intervals)
        for c in cands:
            c["scene"] = scene
        logger.debug("场景 %s: 帧 %d, 候选对 %d", scene, len(frame_indices), len(cands))
        all_candidates.extend(cands)

    if not all_candidates:
        raise RuntimeError("未找到任何候选帧对，请检查 --sun3d_dir 与帧命名正则 FRAME_RE")

    chosen = sample_pairs(all_candidates, args.target_num, seed=args.seed)

    if args.overlap_pcd_dir:
        chosen = filter_by_overlap(chosen, args.overlap_pcd_dir,
                                   min_overlap=args.overlap_threshold)

    # 输出清单（含相对路径，便于后续脚本定位文件）
    pairs_out = []
    for c in chosen:
        pairs_out.append({
            "scene": c["scene"],
            "frame_a": c["frame_a"],
            "frame_b": c["frame_b"],
            "interval": c["interval"],
            "pair_id": f"{c['scene']}_{c['frame_a']:06d}_{c['frame_b']:06d}",
        })

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"target_num": args.target_num,
                   "num_pairs": len(pairs_out),
                   "intervals": intervals,
                   "seed": args.seed,
                   "pairs": pairs_out},
                  f, ensure_ascii=False, indent=2)

    n_by_interval = {}
    for p in pairs_out:
        n_by_interval[p["interval"]] = n_by_interval.get(p["interval"], 0) + 1
    logger.info("完成：%d 对 -> %s（按间隔: %s）",
                len(pairs_out), out_path, n_by_interval)


if __name__ == "__main__":
    main()
