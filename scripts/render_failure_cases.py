#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""失败案例截图：对指定帧对渲染「真值对齐」与「估计对齐」的左右对比图。

用法：
  python scripts/render_failure_cases.py \
      --pair_ids mit_studyroom_000030_000035,mit_studyroom_000150_000180 \
      --pcd_dir results/preprocess/pcd_real \
      --pose_gt_dir results/preprocess/pose_gt_real20 \
      --out_dir results/failure_cases

渲染逻辑（兼容 Open3D 0.17 与 0.19+，参考 visualize_registration.py）：
- 左图：B 用真值 T_gt（B→A）变换到 A 坐标系 → 理想对齐
- 右图：B 用估计 T_est（粗配准+改进 ICP 复现，最优参数）变换 → 实际配准结果
- 颜色：A 蓝色，B 橙色
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("render_failure")

SRC_COLOR = [0.2, 0.4, 0.9]   # A（source）蓝
TGT_COLOR = [0.95, 0.55, 0.1]  # B（target）橙


def load_gt_pose(txt_path):
    arr = np.loadtxt(txt_path)
    if arr.shape == (3, 4):
        return np.vstack([arr, [0.0, 0.0, 0.0, 1.0]])
    if arr.shape == (4, 4):
        return arr
    raise ValueError("位姿文件应为 3x4 或 4x4: %s" % txt_path)


def load_pcd(path, voxel=None):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise FileNotFoundError("点云为空: %s" % path)
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def setup_camera_compat(renderer, fov, center, eye, up):
    """兼容 Open3D 0.17（scene.setup_camera）与 0.19+（renderer.setup_camera）。"""
    if hasattr(renderer, "setup_camera"):
        renderer.setup_camera(fov, center, eye, up)
    else:
        renderer.scene.setup_camera(fov, center, eye, up)


def render_pair(pcd_a, pcd_b_aligned, title_tag, out_path, width=1280, height=640):
    import open3d as o3d
    from open3d.visualization import rendering
    from PIL import Image

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    renderer.scene.add_geometry("a", pcd_a, mat)
    renderer.scene.add_geometry("b", pcd_b_aligned, mat)

    pts = np.concatenate([np.asarray(pcd_a.points), np.asarray(pcd_b_aligned.points)])
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) or 1.0
    eye = center + np.array([radius * 1.2, radius * 0.7, radius * 1.0])
    setup_camera_compat(renderer, 60.0, center, eye, np.array([0.0, 0.0, 1.0]))

    img = renderer.render_to_image()
    path = out_path.parent / ("%s_%s.png" % (out_path.stem, title_tag))
    o3d.io.write_image(str(path), img)
    logger.info("渲染 -> %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="失败案例截图")
    parser.add_argument("--pair_ids", required=True,
                        help="逗号分隔的 pair_id 列表")
    parser.add_argument("--pcd_dir", default="results/preprocess/pcd_real")
    parser.add_argument("--pose_gt_dir", default="results/preprocess/pose_gt_real20")
    parser.add_argument("--out_dir", default="results/failure_cases")
    parser.add_argument("--voxel", type=float, default=0.03,
                        help="渲染下采样体素（默认 0.03，与最优组合一致）")
    parser.add_argument("--fpfh_radius", type=float, default=0.40)
    parser.add_argument("--ransac_iter", type=int, default=500000)
    parser.add_argument("--mutual_filter", action="store_true")
    args = parser.parse_args()

    from registration.coarse_registration import coarse_registration
    from registration.fine_registration import improved_icp
    from registration.preprocess_pointcloud import preprocess_pointcloud

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir = Path(args.pcd_dir)

    pair_ids = [p.strip() for p in args.pair_ids.split(",") if p.strip()]
    for pid in pair_ids:
        parts = pid.split("_")
        fa, fb = parts[-2], parts[-1]
        pcd_a_path = pcd_dir / ("mit_studyroom_%s.ply" % fa)
        pcd_b_path = pcd_dir / ("mit_studyroom_%s.ply" % fb)
        T_gt = load_gt_pose(Path(args.pose_gt_dir) / ("%s.txt" % pid))

        logger.info("处理 %s（%s <-> %s）", pid, fa, fb)
        src = load_pcd(pcd_a_path, args.voxel)
        tgt = load_pcd(pcd_b_path, args.voxel)
        src.paint_uniform_color(SRC_COLOR)
        tgt.paint_uniform_color(TGT_COLOR)

        # 真值对齐（B 变换到 A 坐标系）
        tgt_gt = load_pcd(pcd_b_path, args.voxel)
        tgt_gt.paint_uniform_color(TGT_COLOR)
        tgt_gt.transform(T_gt)

        # 估计对齐：粗配准 + 改进 ICP（最优参数复现）
        src_proc = preprocess_pointcloud(load_pcd(pcd_a_path), voxel_size=args.voxel)
        tgt_proc = preprocess_pointcloud(load_pcd(pcd_b_path), voxel_size=args.voxel)
        T_init, _ = coarse_registration(
            src_proc, tgt_proc, voxel_size=args.voxel,
            fpfh_radius=args.fpfh_radius, max_iteration=args.ransac_iter,
            mutual_filter=args.mutual_filter)
        T_est, _ = improved_icp(src_proc, tgt_proc, T_init)

        tgt_est = load_pcd(pcd_b_path, args.voxel)
        tgt_est.paint_uniform_color(TGT_COLOR)
        tgt_est.transform(T_est)

        # 渲染两张：真值对齐（理想） vs 估计对齐（实际）
        render_pair(src, tgt_gt, "gt", out_dir / pid)
        render_pair(src, tgt_est, "est", out_dir / pid)

        # 计算估计与真值的误差，记录
        err = np.linalg.inv(T_est) @ T_gt
        R = err[:3, :3]
        cos_ang = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = np.degrees(np.arccos(cos_ang))
        trans_m = float(np.linalg.norm(err[:3, 3]))
        logger.info("  %s: 旋转误差 %.2f° 平移误差 %.3fm", pid, rot_deg, trans_m)


if __name__ == "__main__":
    main()
