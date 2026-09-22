#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 VoteNet 训练数据（第二批交付）。

从每帧点云采样固定点数（默认 50000），结合 SUN RGB-D 官方标注
（label/{scene}/{frame}.txt，与 VoteNet 预处理同格式），输出：

    .npz（每帧一个）：
      - point_cloud : (N, 3) 采样后的点云
      - point_clouds: 同 point_cloud（与 VoteNet 官方 dataloader 键名一致）
      - bboxes     : (M, 7) 3D 边界框 [cx, cy, cz, l, w, h, heading]
      - class_ids  : (M,) 类别 id（见 CLASS_NAMES）
      - vote_label : (N, 3) 每个点的投票向量（指向所属框中心）
      - vote_mask  : (N,) 是否参与投票监督（1=有效，0=背景）

SUN RGB-D 3D 标注格式（label/*.txt，每行）：
    class_name cx cy cz l w h heading_angle
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_detection_data")

CLASS_NAMES = [
    "bathtub", "bed", "bookshelf", "chair", "desk", "dresser",
    "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_pcd(path):
    """读取点云，返回 (N, 3) float32。支持 .ply / .npz / .npy。"""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        pts = data["point_cloud"] if "point_cloud" in data.files else data["points"]
        pts = np.asarray(pts, dtype=np.float32)[:, :3]
    else:  # .npy
        pts = np.load(path).astype(np.float32)[:, :3]
    if len(pts) == 0:
        raise ValueError(f"点云为空: {path}")
    return pts


def sample_points(points, num_points, seed=2024):
    """随机采样（无放回，不足则重复），返回 (num_points, 3)。"""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return points[idx]


def load_bboxes(label_path):
    """读取 SUN RGB-D 标注，返回 (bboxes, class_ids)。

    label_path: label/*.txt 路径。
    Returns:
        bboxes: (M, 7) float32 [cx, cy, cz, l, w, h, heading]
        class_ids: (M,) int32
    """
    bboxes, class_ids = [], []
    label_path = Path(label_path)
    if not label_path.is_file():
        logger.warning("标注文件不存在: %s", label_path)
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            cls_name = parts[0]
            if cls_name not in CLASS_TO_ID:
                logger.warning("跳过未知类别: %s（%s）", cls_name, label_path.name)
                continue
            bboxes.append([float(v) for v in parts[1:8]])
            class_ids.append(CLASS_TO_ID[cls_name])
    if not bboxes:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))
    return (np.asarray(bboxes, dtype=np.float32),
            np.asarray(class_ids, dtype=np.int32))


def compute_vote_targets(points, bboxes, vote_radius=0.1):
    """为每个点计算投票目标（VoteNet 的 seed -> vote 监督）。

    对每个点：找到最近的框中心，若距离 <= vote_radius 则视为有效
    （vote_mask=1），vote_label = 框中心 - 点坐标；否则 vote=0、mask=0。
    """
    n = len(points)
    vote_label = np.zeros((n, 3), dtype=np.float32)
    vote_mask = np.zeros(n, dtype=np.uint8)
    if len(bboxes) == 0:
        return vote_label, vote_mask

    centers = bboxes[:, :3]
    # 距离矩阵 (N, M)：内存敏感，M 通常 < 20
    diff = points[:, None, :] - centers[None, :, :]
    dist2 = np.einsum("nmk,nmk->nm", diff, diff)
    nearest = np.argmin(dist2, axis=1)
    near_dist = np.sqrt(dist2[np.arange(n), nearest])

    valid = near_dist <= vote_radius
    vote_mask[valid] = 1
    vote_label[valid] = centers[nearest[valid]] - points[valid]
    return vote_label, vote_mask


def generate_detection_data(pcd_dir, label_dir, out_dir, num_points=50000,
                            num_classes=10, vote_radius=0.1, seed=2024,
                            val_scenes=None, val_ratio=0.2):
    """主流程：为每帧生成 VoteNet 训练样本。

    Args:
        pcd_dir: 点云目录（.ply/.npy/.npz，文件名与帧一一对应）。
        label_dir: 标注目录（label/{scene}/{frame}.txt）。
        out_dir: 输出目录（每帧一个 .npz + split.json）。
        num_points: 每帧采样点数。
        num_classes: 类别数（与 CLASS_NAMES 一致）。
        vote_radius: 投票有效半径（米）。
        val_scenes: 指定作为验证集的场景名列表；None 时按 val_ratio 随机划分。
        val_ratio: 验证集比例（当 val_scenes 为空时生效）。
    """
    pcd_dir = Path(pcd_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd_files = sorted(pcd_dir.glob("*.ply")) + \
                sorted(pcd_dir.glob("*.npy")) + \
                sorted(pcd_dir.glob("*.npz"))
    logger.info("点云文件数: %d", len(pcd_files))
    if not pcd_files:
        raise FileNotFoundError(f"点云目录为空: {pcd_dir}")

    samples = []
    for pcd_path in pcd_files:
        points = load_pcd(pcd_path)
        # 标注路径：label/{scene}/{frame}.txt，与点云文件名同 stem
        label_path = label_dir / f"{pcd_path.stem}.txt"
        if not label_path.is_file():
            logger.warning("缺标注 %s，仍生成背景样本（bbox 为空）", label_path)

        bboxes, class_ids = load_bboxes(label_path)
        pts = sample_points(points, num_points, seed=seed)
        vote_label, vote_mask = compute_vote_targets(pts, bboxes,
                                                     vote_radius=vote_radius)

        out_npz = out_dir / f"{pcd_path.stem}.npz"
        # point_cloud 与 point_clouds 双键：前者本仓库内部约定，
        # 后者与 VoteNet 官方 sunrgbd_dataset.py 的 data['point_clouds'] 一致，
        # 保证生成的数据能被官方 dataloader 直接读取。
        np.savez_compressed(
            out_npz,
            point_cloud=pts,
            point_clouds=pts,
            bboxes=bboxes,
            class_ids=class_ids,
            vote_label=vote_label,
            vote_mask=vote_mask,
        )
        samples.append({"name": pcd_path.stem, "path": str(out_npz)})

    # train/val 划分
    rng = np.random.default_rng(seed)
    if val_scenes:
        # 按场景划分（推荐，避免同一场景帧跨集合）
        train, val = [], []
        for s in samples:
            # 约定：文件名以 {scene}_ 开头（见 parse_sunrgbd.py）
            scene = s["name"].split("_")[0]
            (val if scene in set(val_scenes) else train).append(s)
    else:
        idx = rng.permutation(len(samples))
        n_val = int(len(samples) * val_ratio)
        train = [samples[i] for i in idx[n_val:]]
        val = [samples[i] for i in idx[:n_val]]

    split = {"train": train, "val": val}
    (out_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("完成：%d 帧 -> %s（train %d / val %d）",
                len(samples), out_dir, len(train), len(val))


def main():
    parser = argparse.ArgumentParser(description="生成 VoteNet 训练数据")
    parser.add_argument("--pcd_dir", default="results/preprocess/pcd",
                        help="点云目录（.ply/.npy/.npz）")
    parser.add_argument("--label_dir", default="data/SUNRGBD/label",
                        help="SUN RGB-D 标注目录（label/{scene}/{frame}.txt）")
    parser.add_argument("--out_dir", default="results/preprocess/detection",
                        help="输出目录（每帧一个 .npz + split.json）")
    parser.add_argument("--num_points", type=int, default=50000,
                        help="每帧采样点数")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="类别数（应与 CLASS_NAMES 长度一致）")
    parser.add_argument("--vote_radius", type=float, default=0.1,
                        help="投票有效半径（米）")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--val_scenes", nargs="*", default=None,
                        help="指定验证集场景名（默认随机 20%%）")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    generate_detection_data(
        args.pcd_dir, args.label_dir, args.out_dir,
        num_points=args.num_points, num_classes=args.num_classes,
        vote_radius=args.vote_radius, seed=args.seed,
        val_scenes=args.val_scenes, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
