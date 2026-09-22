#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 SUN RGB-D 官方标注提取 10 类 3D 边界框真值 -> data/gt/gt.json。

SUN RGB-D 官方标注格式（SUNRGBDtoolbox 解析，与 VoteNet 预处理一致）：
    data/SUNRGBD/label/{scene}/{frame}.txt，每行：
    class_name cx cy cz l w h heading_angle
    （坐标系为相机系，单位米，朝向角弧度）

输出 data/gt/gt.json：
    {
      "classes": ["bathtub", ..., "toilet"],   # 10 类固定顺序
      "frames": [
        {"scene": "scene_001", "frame": "000000",
         "class_names": ["chair"],             # 该帧出现的类别名
         "boxes": [[cx,cy,cz,l,w,h,heading], ...],  # (K,7) float32
         "labels": [2, ...]}                   # (K,) int32（对齐 classes）
      ]
    }

用法：
    python preprocess/extract_gt.py \
        --label_dir data/SUNRGBD/label \
        --out_path data/gt/gt.json \
        [--scenes scene_001 scene_002]   # 可选，只提取指定场景

说明：
    - 只保留 10 类中的类别，其余行跳过并计数。
    - 标注目录缺失时输出空 gt.json（并提示），不影响其他流程。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# 让 `python3 preprocess/extract_gt.py` 与 `python3 -m preprocess.extract_gt` 都能导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_detection_data import CLASS_NAMES, CLASS_TO_ID, load_bboxes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("extract_gt")


def extract_gt(label_dir, scenes=None):
    """扫描标注目录，返回 gt.json 的 frames 列表。

    label_dir: data/SUNRGBD/label，子目录结构 {scene}/{frame}.txt。
    scenes: 指定场景名列表（None = 全部）。
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        log.warning("标注目录不存在: %s（SUN3D 序列帧无 3D 框标注时属正常）", label_dir)
        return []

    scene_dirs = sorted(p for p in label_dir.iterdir() if p.is_dir())
    if scenes:
        scene_dirs = [p for p in scene_dirs if p.name in set(scenes)]

    frames, skipped = [], 0
    for scene_dir in scene_dirs:
        scene = scene_dir.name
        for label_file in sorted(scene_dir.glob("*.txt")):
            bboxes, class_ids = load_bboxes(label_file)  # 与 VoteNet 预处理同源
            if len(bboxes) == 0:
                skipped += 1
                continue
            frames.append({
                "scene": scene,
                "frame": label_file.stem,
                "class_names": [CLASS_NAMES[i] for i in class_ids],
                "boxes": bboxes.tolist(),     # (K,7) [cx,cy,cz,l,w,h,heading]
                "labels": class_ids.tolist(), # (K,) int
            })
    if skipped:
        log.info("跳过 %d 个无有效框的标注文件", skipped)
    log.info("提取 %d 帧真值（%d 个场景）", len(frames), len(scene_dirs))
    return frames


def main():
    parser = argparse.ArgumentParser(description="提取 SUN RGB-D 3D 边界框真值 -> data/gt/gt.json")
    parser.add_argument("--label_dir", default="data/SUNRGBD/label",
                        help="SUN RGB-D 官方标注目录")
    parser.add_argument("--out_path", default="data/gt/gt.json",
                        help="输出 gt.json 路径")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="只提取指定场景（默认全部）")
    args = parser.parse_args()

    frames = extract_gt(args.label_dir, args.scenes)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASS_NAMES, "frames": frames}, f, ensure_ascii=False, indent=1)
    log.info("已写入 %s（%d 帧）", out, len(frames))


if __name__ == "__main__":
    main()
