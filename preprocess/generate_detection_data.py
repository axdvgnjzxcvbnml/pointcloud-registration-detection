#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_detection_data.py — 生成 VoteNet 训练数据（第二批交付）

功能
----
    从预处理产物（点云 + 标注）生成 VoteNet 风格训练样本：
      - 每帧点云：采样至 points_per_frame（默认 50000）
      - GT 框：bboxes (M,7) [cx,cy,cz,l,w,h,heading] + class_ids
      - 投票标签：每个点生成到所属物体中心的投票向量
        vote_label (N,3) + vote_mask (N,)
    同时输出 train/val 划分 split.json（按帧名哈希，比例 8:2）。

说明
----
    标注来源与坐标系约定以 SUN RGB-D 官方工具箱为准
    （3D 框通常为相机系，与点云一致；TODO 按实际工具箱核对）。

用法
----
    python preprocess/generate_detection_data.py \
        --data_root results/preprocess \
        --out_dir results/preprocess/detection
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NUM_CLASSES = 10


def _stable_split(name, ratio=0.8):
    """按帧名哈希稳定划分 train/val（与顺序无关）。"""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return "train" if int(h[:4], 16) / 0xFFFF < ratio else "val"


def compute_vote_labels(points, bboxes, vote_radius=0.1):
    """为每个点计算投票标签。

    规则：
      - 若点落在某 GT 框内（含 1cm 余量），投票到该框中心；
      - 否则 vote_mask=0，vote_label 置 0。
    与 VoteNet 数据生成器（sunrgbd_data.py）思路一致。

    Returns:
        vote_label: (N,3) float32
        vote_mask: (N,) float32
    """
    N = len(points)
    vote_label = np.zeros((N, 3), dtype=np.float32)
    vote_mask = np.zeros(N, dtype=np.float32)
    margin = 0.01
    for box in bboxes:
        cx, cy, cz, l, w, h, heading = box
        c, s = np.cos(heading), np.sin(heading)
        d = points - np.array([cx, cy, cz])
        local_x = d[:, 0] * c + d[:, 1] * s
        local_y = -d[:, 0] * s + d[:, 1] * c
        local_z = d[:, 2]
        inside = ((np.abs(local_x) <= l / 2 + margin)
                  & (np.abs(local_y) <= w / 2 + margin)
                  & (np.abs(local_z) <= h / 2 + margin))
        if not inside.any():
            continue
        vote_label[inside] = np.array([cx, cy, cz]) - points[inside]
        vote_mask[inside] = 1.0
    # 可选：超出 vote_radius 的投票裁剪（保持 VoteNet 约定）
    over = (np.linalg.norm(vote_label, axis=1) > vote_radius) & (vote_mask > 0)
    vote_label[over] *= vote_radius / np.maximum(
        np.linalg.norm(vote_label, axis=1)[over], 1e-9)[:, None]
    return vote_label, vote_mask


def load_annotations(frame_meta):
    """从帧元数据加载标注框。骨架返回空列表（TODO 实现）。

    说明：SUN RGB-D 的 3D 标注（mat）解析依赖 SUNRGBDtoolbox，
    键名（gt_corners / class_id 等）需按官方工具箱核对后实现。
    """
    # TODO: 解析 label 文件/对应 mat，返回 {"bboxes": (M,7), "class_ids": (M,)}
    return {"bboxes": np.zeros((0, 7), dtype=np.float32),
            "class_ids": np.zeros((0,), dtype=np.int64)}


def generate(data_root, out_dir, points_per_frame=50000, vote_radius=0.1,
             num_classes=10, val_ratio=0.2):
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 parse_sunrgbd 生成的帧索引
    with open(data_root / "frames_index.json", "r", encoding="utf-8") as f:
        frames = json.load(f)

    pcd_dir = data_root / "pcd"
    split = {"train": [], "val": []}
    for fr in frames:
        name = fr["name"]
        pcd_file = pcd_dir / f"{name.replace('/', '_')}.npz"
        if not pcd_file.is_file():
            continue
        points = np.load(pcd_file)["point_cloud"]
        if len(points) > points_per_frame:
            idx = np.random.default_rng(abs(hash(name)) % (2**32)).choice(
                len(points), points_per_frame, replace=False)
            points = points[idx]

        ann = load_annotations(fr)
        vote_label, vote_mask = compute_vote_labels(points, ann["bboxes"],
                                                    vote_radius)
        np.savez_compressed(
            str(out_dir / f"{name.replace('/', '_')}.npz"),
            point_cloud=points.astype(np.float32),
            bboxes=ann["bboxes"].astype(np.float32),
            class_ids=ann["class_ids"].astype(np.int64),
            vote_label=vote_label,
            vote_mask=vote_mask,
        )
        split[_stable_split(name, 1 - val_ratio)].append({"name": name})

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    logger.info("生成 %d 样本（train %d / val %d）-> %s",
                len(split["train"]) + len(split["val"]),
                len(split["train"]), len(split["val"]), out_dir)


def main():
    parser = argparse.ArgumentParser(description="生成 VoteNet 训练数据")
    parser.add_argument("--data_root", default="results/preprocess",
                        help="预处理根目录（含 frames_index.json 与 pcd/）")
    parser.add_argument("--out_dir", default="results/preprocess/detection")
    parser.add_argument("--points_per_frame", type=int, default=50000)
    parser.add_argument("--vote_radius", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate(args.data_root, args.out_dir, args.points_per_frame,
             args.vote_radius, args.num_classes, args.val_ratio)


if __name__ == "__main__":
    main()
