#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pose_gt.py — 计算帧间相对位姿真值（第二批交付）

功能
----
    由两帧相机外参（world→camera）计算相对位姿 T_src2dst：

        T_src2dst = T_dst_world @ inv(T_src_world)

    使得：p_dst = T_src2dst @ p_src（把 src 系点变换到 dst 系）。

输入
----
    pairs.json：{src, dst, interval}
    外参目录：每个帧一个 3×4 / 4×4 文本矩阵
    （文件名 = 帧名.txt，对应 extrinsics/ 目录）

输出
----
    pose_gt/{src}_to_{dst}.npy / .txt：4×4 矩阵

说明
----
    若部分帧无外参，脚本会跳过并在日志中记录缺失清单；
    默认阈值检查：旋转角与平移量超范围视为异常（可 --skip-threshold 关闭）。

用法
----
    python preprocess/compute_pose_gt.py \
        --pairs results/preprocess/pairs.json \
        --extrinsics-dir data/SUN3D \
        --out-dir results/preprocess/pose_gt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_extrinsic(path):
    """读取外参矩阵（3×4 或 4×4 文本），补全为 4×4。"""
    m = np.loadtxt(path).reshape(-1, 4)
    if m.shape[0] == 3:
        m = np.vstack([m, [0, 0, 0, 1]])
    return m


def extrinsic_to_world_to_camera(mat):
    """把外参规范化为 world→camera 的 4×4 矩阵。"""
    return np.asarray(mat, dtype=np.float64).reshape(4, 4)


def compute_relative_pose(T_src_world, T_dst_world):
    """T_src2dst = T_dst_world @ inv(T_src_world)。"""
    T_src_world = np.asarray(T_src_world, dtype=np.float64)
    T_dst_world = np.asarray(T_dst_world, dtype=np.float64)
    return T_dst_world @ np.linalg.inv(T_src_world)


def decompose_rotation_translation(T):
    """从 4×4 变换提取旋转角（度）与平移距离（米）。"""
    R = T[:3, :3]
    t = T[:3, 3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    return angle, float(np.linalg.norm(t))


def main():
    parser = argparse.ArgumentParser(description="计算帧间相对位姿真值")
    parser.add_argument("--pairs", required=True,
                        help="帧对清单 pairs.json")
    parser.add_argument("--extrinsics-dir", required=True,
                        help="外参目录（{帧名}.txt，3x4/4x4）")
    parser.add_argument("--out-dir", default="results/preprocess/pose_gt")
    parser.add_argument("--max_rotation_deg", type=float, default=180.0,
                        help="旋转角异常阈值（度）")
    parser.add_argument("--max_translation", type=float, default=10.0,
                        help="平移异常阈值（米）")
    parser.add_argument("--skip-threshold", action="store_true",
                        help="跳过阈值检查")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    ext_dir = Path(args.extrinsics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    skipped = []
    for p in pairs:
        src, dst = p["src"], p["dst"]
        src_file = ext_dir / f"{src}.txt"
        dst_file = ext_dir / f"{dst}.txt"
        if not src_file.is_file() or not dst_file.is_file():
            skipped.append(f"{src}->{dst} (缺外参)")
            n_skip += 1
            continue
        T_src_world = extrinsic_to_world_to_camera(load_extrinsic(src_file))
        T_dst_world = extrinsic_to_world_to_camera(load_extrinsic(dst_file))
        T = compute_relative_pose(T_src_world, T_dst_world)

        if not args.skip_threshold:
            angle, dist = decompose_rotation_translation(T)
            if angle > args.max_rotation_deg or dist > args.max_translation:
                skipped.append(f"{src}->{dst} (角度{angle:.1f}°/平移{dist:.2f}m 超阈值)")
                n_skip += 1
                continue

        stem = f"{src}_to_{dst}".replace("/", "__")
        np.save(out_dir / f"{stem}.npy", T)
        np.savetxt(out_dir / f"{stem}.txt", T, fmt="%.8f")
        n_ok += 1

    logger.info("完成：有效 %d，跳过 %d -> %s", n_ok, n_skip, out_dir)
    if skipped:
        with open(out_dir / "skipped.json", "w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
