#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pose_gt.py — 计算帧对 6DOF 配准真值（第二批交付）

功能
----
    利用 SUN3D 的 extrinsics/*.txt（3x4 相机位姿矩阵）计算帧对间的
    相对变换：

        T_AB = Pose_A^{-1} @ Pose_B

    其中 A 为参考帧，T_AB 把帧 B 的坐标系变换到帧 A 的坐标系，
    即：P_B 中的点在 T_AB 作用下得到 P_A 中的坐标。
    该 4x4 矩阵即拼接评测用的 6DOF 真值。

    同时输出旋转角（度）与平移距离（米），便于快速核验真值合理性。

约定
----
    SUN3D 的 extrinsics/*.txt 为 3x4 矩阵（行优先 12 个浮点数）。
    无论位姿是 camera-to-world 还是 world-to-camera，
    T_AB = inv(Pose_A) @ Pose_B 在“两帧位姿同坐标系约定”下均成立；
    --pose_convention 仅用于记录/校验，不改变公式。

用法
----
    python preprocess/compute_pose_gt.py \
        --pairs results/preprocess/pairs/pairs.json \
        --sun3d_dir data/SUN3D \
        --out_dir results/preprocess/pose_gt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_pose_txt(path):
    """读取 3x4 位姿矩阵（12 个浮点数，行优先），返回 (3, 4) float64。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"位姿文件不存在: {path}")
    mat = np.loadtxt(str(path)).reshape(3, 4).astype(np.float64)
    return mat


def to_homogeneous(pose34):
    """3x4 -> 4x4 齐次变换矩阵（最后一行 [0,0,0,1]）。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = pose34
    return T


def compute_relative_pose(pose_a, pose_b):
    """计算相对变换 T_AB = inv(Pose_A) @ Pose_B，返回 (4, 4)。"""
    Ta, Tb = to_homogeneous(pose_a), to_homogeneous(pose_b)
    return np.linalg.inv(Ta) @ Tb


def decompose_transform(T):
    """分解相对变换，返回 (旋转角(度), 平移距离(米))。"""
    rot = T[:3, :3]
    cos_ang = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.rad2deg(np.arccos(cos_ang)))
    trans_m = float(np.linalg.norm(T[:3, 3]))
    return rot_deg, trans_m


def find_extrinsic(scene_dir, frame_idx):
    """在场景目录下查找帧对应的 extrinsics 文件。

    候选命名（SUN3D 常见形式，按需修改）：
      - extrinsics/frame-{idx:06d}.txt
      - extrinsics/{idx:06d}.txt
      - pose/frame-{idx:06d}.txt
    返回路径或 None。
    """
    scene_dir = Path(scene_dir)
    candidates = [
        scene_dir / "extrinsics" / f"frame-{frame_idx:06d}.txt",
        scene_dir / "extrinsics" / f"{frame_idx:06d}.txt",
        scene_dir / "pose" / f"frame-{frame_idx:06d}.txt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def compute_pose_gt(pairs, sun3d_dir, out_dir, pose_convention="cam2world"):
    """主流程：为每个帧对计算 T_AB 并落盘。

    Args:
        pairs: sample_frame_pairs.py 输出的配对清单 dict。
        sun3d_dir: SUN3D 数据根目录。
        out_dir: 输出目录（每对一个 .txt + 汇总 pose_gt.json）。
        pose_convention: 记录用，cam2world / world2cam。
    Returns:
        汇总列表。
    """
    sun3d_dir = Path(sun3d_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in pairs["pairs"]:
        scene = item["scene"]
        ext_a = find_extrinsic(sun3d_dir / scene, item["frame_a"])
        ext_b = find_extrinsic(sun3d_dir / scene, item["frame_b"])
        if ext_a is None or ext_b is None:
            logger.warning("帧对 %s 缺少位姿文件，跳过", item["pair_id"])
            continue

        pose_a = read_pose_txt(ext_a)
        pose_b = read_pose_txt(ext_b)
        T_ab = compute_relative_pose(pose_a, pose_b)
        rot_deg, trans_m = decompose_transform(T_ab)

        out_txt = out_dir / f"{item['pair_id']}.txt"
        np.savetxt(str(out_txt), T_ab, fmt="%.9f")

        summary.append({
            "pair_id": item["pair_id"],
            "scene": scene,
            "frame_a": item["frame_a"],
            "frame_b": item["frame_b"],
            "interval": item["interval"],
            "pose_a_path": str(ext_a),
            "pose_b_path": str(ext_b),
            "T_AB_path": str(out_txt),
            "rot_deg": round(rot_deg, 4),
            "trans_m": round(trans_m, 4),
            "pose_convention": pose_convention,
        })
        logger.debug("帧对 %s: rot=%.2f°, trans=%.3fm",
                     item["pair_id"], rot_deg, trans_m)

    summary_path = out_dir / "pose_gt.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"num_gt": len(summary), "items": summary},
                  f, ensure_ascii=False, indent=2)
    logger.info("完成：%d 对真值 -> %s（汇总 %s）",
                len(summary), out_dir, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="计算帧对 6DOF 配准真值")
    parser.add_argument("--pairs", default="results/preprocess/pairs/pairs.json",
                        help="帧对清单（sample_frame_pairs.py 输出）")
    parser.add_argument("--sun3d_dir", default="data/SUN3D",
                        help="SUN3D 数据根目录")
    parser.add_argument("--out_dir", default="results/preprocess/pose_gt",
                        help="真值输出目录")
    parser.add_argument("--pose_convention", choices=["cam2world", "world2cam"],
                        default="cam2world",
                        help="位姿坐标系约定（仅记录，不改变公式）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    compute_pose_gt(pairs, args.sun3d_dir, args.out_dir,
                    pose_convention=args.pose_convention)


if __name__ == "__main__":
    main()
