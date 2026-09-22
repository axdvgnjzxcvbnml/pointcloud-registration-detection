#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_sunrgbd.py — 解析 SUN RGB-D 文件结构（第二批交付）

功能
----
    扫描 SUN RGB-D 数据目录，对每帧读取：
        - RGB 图（image/*.jpg|png）
        - 深度图（depth/*.mat 或 16bit png，单位 mm）
        - 相机内参 K（3x3）
        - 相机外参 Rtilt（3x4，部分子集提供）
    输出统一格式（.npz + manifest.json），供后续模块使用。

解析逻辑参考
------------
    VoteNet 官方（facebookresearch/votenet）的 sunrgbd_data.py：
      通过 SUN RGB-D 官方工具箱元数据读取帧列表，深度图用
      scipy.io.loadmat 读取（变量名 'depth'）。

输入目录约定（以 data/SUNRGBD 为根）
-----------------------------------
    image/      RGB 图
    depth/      深度图（.mat 或 .png，16bit，单位 mm）
    label/      标注（3D bbox 等，本脚本仅登记路径）
    extrinsics/ 相机外参（3x4 .txt，仅部分子集提供）

输出
----
    {out_dir}/manifest.json                    全量帧清单
    {out_dir}/frames/{scene}_{frame_key}.npz   rgb / depth / K / Rtilt

用法
----
    python preprocess/parse_sunrgbd.py --data_root data/SUNRGBD \
        --out_dir results/preprocess --scenes SUNRGBD1,SUNRGBD2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from PIL import Image

logger = logging.getLogger(__name__)

# 占位内参集中维护在 preprocess/defaults.py（避免多处定义不一致）；
# 真实数据必须从 SUNRGBDtoolbox 元数据（SUNRGBDMeta）读取。
try:
    from defaults import DEFAULT_K
except ImportError:  # 作为 preprocess 包被导入时
    from preprocess.defaults import DEFAULT_K


def load_rgb(path):
    """读取 RGB 图，返回 (H, W, 3) uint8 数组。"""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    return rgb


def load_depth(path):
    """读取深度图，返回 (H, W) float32（保持原始单位，不缩放）。

    支持两种格式：
      - .mat：loadmat 读取变量 'depth'（SUN RGB-D 官方格式，单位 mm）
      - .png：16bit 单通道，单位 mm
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        data = loadmat(str(path))
        if "depth" not in data:
            raise KeyError(
                f"{path} 中未找到变量 'depth'，实际变量: {list(data.keys())}")
        depth = data["depth"].astype(np.float32)
    else:
        with Image.open(path) as img:
            depth = np.asarray(img, dtype=np.float32)
        if depth.ndim == 3:  # 保险：某些 png 带 alpha/灰度三通道
            depth = depth[..., 0]
    return depth


def load_intrinsic(scene_dir, frame_key):
    """读取相机内参 K（3x3）。

    查找顺序：
      1. {scene_dir}/intrinsics/{frame_key}.txt（若数据集提供）
      2. {scene_dir}/intrin.txt 或 intrinsics.txt
      3. 回退 DEFAULT_K（占位，并给出警告）
    """
    scene_dir = Path(scene_dir)
    candidates = [
        scene_dir / "intrinsics" / f"{frame_key}.txt",
        scene_dir / "intrin.txt",
        scene_dir / "intrinsics.txt",
    ]
    for cand in candidates:
        if cand.is_file():
            return np.loadtxt(str(cand)).reshape(3, 3).astype(np.float64)
    logger.warning("未找到内参文件（%s），使用默认 K（占位）", scene_dir)
    return DEFAULT_K.copy()


def load_extrinsic(path):
    """读取相机外参（3x4 .txt），返回 (3, 4) float64 或 None。

    SUN RGB-D 的 extrinsics 为 3x4 矩阵（旋转 + 平移）。
    本脚本按“原始 3x4 直读、不追加齐次行”处理；
    坐标系的含义在 compute_pose_gt.py 中按 --pose_convention 解释。
    """
    path = Path(path)
    if not path.is_file():
        logger.warning("外参文件不存在: %s（部分子集不提供）", path)
        return None
    return np.loadtxt(str(path)).reshape(3, 4).astype(np.float64)


def discover_frames(data_root):
    """扫描 image/ 目录，返回帧清单（list[dict]）。

    每条记录包含：
        scene, frame_key, rgb_path, depth_path, label_path, ext_path
    """
    data_root = Path(data_root)
    img_root = data_root / "image"
    if not img_root.is_dir():
        raise FileNotFoundError(f"未找到图像目录: {img_root}")

    frames = []
    for scene_dir in sorted(img_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for img_path in sorted(scene_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            frame_key = img_path.stem
            scene = scene_dir.name
            depth_path = data_root / "depth" / scene / f"{frame_key}.mat"
            if not depth_path.is_file():
                depth_path = data_root / "depth" / scene / f"{frame_key}.png"
            label_path = data_root / "label" / scene / f"{frame_key}.txt"
            ext_path = data_root / "extrinsics" / scene / f"{frame_key}.txt"
            frames.append({
                "scene": scene,
                "frame_key": frame_key,
                "rgb_path": str(img_path),
                "depth_path": str(depth_path) if depth_path.is_file() else None,
                "label_path": str(label_path) if label_path.is_file() else None,
                "ext_path": str(ext_path) if ext_path.is_file() else None,
            })
    return frames


def parse_sunrgbd(data_root, out_dir, scenes=None, max_frames=None):
    """主流程：解析全部（或指定场景子集）帧并输出统一格式。

    Args:
        data_root: SUN RGB-D 数据根目录。
        out_dir:   输出目录（manifest.json + frames/）。
        scenes:    场景名列表，None 表示全部场景。
        max_frames: 最多解析帧数（调试用）。
    Returns:
        manifest 列表。
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(data_root)
    if scenes:
        scene_set = set(scenes)
        frames = [f for f in frames if f["scene"] in scene_set]
    if max_frames:
        frames = frames[:max_frames]
    logger.info("待解析帧数: %d", len(frames))

    manifest = []
    skipped = []   # 缺深度等被跳过的帧，单独落盘以便审计（不污染主 manifest）
    for i, item in enumerate(frames):
        rgb = load_rgb(item["rgb_path"])
        if not item["depth_path"]:
            logger.warning("帧 %s 缺少深度图，跳过", item["frame_key"])
            skipped.append({"scene": item["scene"],
                            "frame_key": item["frame_key"],
                            "reason": "missing_depth"})
            continue
        depth = load_depth(item["depth_path"])
        K = load_intrinsic(data_root / "image" / item["scene"], item["frame_key"])
        Rtilt = load_extrinsic(item["ext_path"]) if item["ext_path"] else None

        out_npz = frames_dir / f"{item['scene']}_{item['frame_key']}.npz"
        np.savez_compressed(
            out_npz,
            rgb=rgb,
            depth=depth,
            K=K,
            Rtilt=Rtilt if Rtilt is not None else np.zeros((3, 4), dtype=np.float64),
        )

        manifest.append({
            "scene": item["scene"],
            "frame_key": item["frame_key"],
            "npz_path": str(out_npz),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "has_depth": True,
            "has_extrinsic": Rtilt is not None,
        })
        if (i + 1) % 200 == 0:
            logger.info("已解析 %d/%d 帧", i + 1, len(frames))

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    # 跳过帧审计清单（无跳过时写空列表）
    skipped_path = out_dir / "skipped_frames.json"
    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)
    logger.info("完成：共 %d 帧，跳过 %d 帧（见 %s），manifest -> %s",
                len(manifest), len(skipped), skipped_path, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="解析 SUN RGB-D 文件结构")
    parser.add_argument("--data_root", default="data/SUNRGBD",
                        help="SUN RGB-D 数据根目录")
    parser.add_argument("--out_dir", default="results/preprocess",
                        help="统一格式输出目录")
    parser.add_argument("--scenes", default=None,
                        help="逗号分隔的场景子集，如 SUNRGBD1,SUNRGBD2")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="最多解析帧数（调试用）")
    parser.add_argument("--verbose", action="store_true",
                        help="输出调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    scenes = args.scenes.split(",") if args.scenes else None
    parse_sunrgbd(args.data_root, args.out_dir,
                  scenes=scenes, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
