#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 data/gt/gt.json 真值，输出与 VoteNet 预测格式对齐的张量。

对齐约定（与 detection/evaluate_detection.compute_mAP 一致）：
    gt 列表项格式  {"boxes": (K,7) float32, "labels": (K,) int32}
    帧顺序与 detections 列表一一对应（由调用方按相同排序传入 frame_ids）。

用法：
    from load_gt import load_gt, load_gt_indexed

    gts = load_gt("data/gt/gt.json")           # list[dict]，顺序 = frames 顺序
    idx = load_gt_indexed("data/gt/gt.json")   # {(scene, frame): dict} 便于按帧对齐
"""
import json
from pathlib import Path

import numpy as np


def load_gt(path):
    """读取 gt.json，返回按文件内顺序排列的真值列表。

    每项 {"boxes": (K,7) float32 [cx,cy,cz,l,w,h,heading],
          "labels": (K,) int32, "scene": str, "frame": str}
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("真值文件不存在: %s（先跑 preprocess/extract_gt.py）" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for fr in data.get("frames", []):
        boxes = np.asarray(fr["boxes"], dtype=np.float32).reshape(-1, 7)
        labels = np.asarray(fr["labels"], dtype=np.int32).reshape(-1)
        if len(boxes) != len(labels):
            raise ValueError("帧 %s/%s boxes 与 labels 数量不一致" % (fr["scene"], fr["frame"]))
        out.append({"scene": fr["scene"], "frame": fr["frame"],
                    "boxes": boxes, "labels": labels})
    return out


def load_gt_indexed(path):
    """返回 {(scene, frame): gt_item} 的索引，便于把预测按帧对齐到真值。"""
    return {(g["scene"], g["frame"]): g for g in load_gt(path)}


def align_gts_by_frames(gts_indexed, frame_ids):
    """按 frame_ids 顺序重排真值。

    frame_ids: [(scene, frame), ...]（与预测顺序一致）。
    缺失帧返回空真值（boxes 空、labels 空），保证与预测列表等长。
    """
    aligned = []
    for sid in frame_ids:
        g = gts_indexed.get(sid)
        if g is None:
            aligned.append({"boxes": np.zeros((0, 7), dtype=np.float32),
                            "labels": np.zeros((0,), dtype=np.int32)})
        else:
            aligned.append(g)
    return aligned


if __name__ == "__main__":
    import sys
    gt_path = sys.argv[1] if len(sys.argv) > 1 else "data/gt/gt.json"
    gts = load_gt(gt_path)
    print("加载 %d 帧真值" % len(gts))
    for g in gts[:3]:
        print("  %s/%s: %d 框, 类别 %s" % (
            g["scene"], g["frame"], len(g["boxes"]), g["labels"].tolist()))
