#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_sunrgbd.py — 解析 SUN RGB-D 数据集（第二批交付）

功能
----
    遍历 SUN RGB-D / SUN3D 目录，为每一帧生成元数据与 RGB+K 中间产物：
      - image/depth/label/extrinsics 路径索引（相对路径）
      - 相机内参 K（3×3）
      - 帧 RGB 图（可选中转 .npz 或 .jpg）
    供后续 depth_to_pointcloud / generate_detection_data 使用。

SUN RGB-D 官方结构（参考 SUNRGBDtoolbox）：
    SUNRGBD/
      ├── kv1/ kv2/ kv3/ ...        # 分区
      │   └── 3DOffice/ ...         # 场景
      │       ├── image/            # RGB（jpg）
      │       ├── depth/            # 深度（16bit png，单位 mm）
      │       ├── label/            # 2D/3D 标注
      │       └── extrinsics/       # 相机外参（部分子集）
      └── SUNRGBDtoolbox/           # 官方 MATLAB 工具箱
            └── Metadata/           # all2all.mat（内参/外参映射）

说明
----
    本脚本为骨架实现，目录结构与 Metadata 解析逻辑以实际数据集为准，
    与 SUNRGBDtoolbox 不一致处按官方工具箱调整（TODO 标注）。

用法
----
    python preprocess/parse_sunrgbd.py --data_root data/SUNRGBD \
        --out_dir results/preprocess \
        --toolbox_dir data/SUNRGBDtoolbox
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def find_frame_files(scene_dir):
    """在单个场景目录下索引 image/depth/label/extrinsics 文件。

    返回: list[dict]（每帧）：
        {"name", "image", "depth", "label", "extrinsics"}
    """
    img_dir = scene_dir / "image"
    dep_dir = scene_dir / "depth"
    lab_dir = scene_dir / "label"
    ext_dir = scene_dir / "extrinsics"

    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    if not imgs:
        imgs = sorted(img_dir.glob("*.png")) if img_dir.is_dir() else []

    frames = []
    for img in imgs:
        stem = img.stem
        depth = dep_dir / f"{stem}.png"
        label = lab_dir / f"{stem}.txt"
        ext = ext_dir / f"{stem}.txt"
        frames.append({
            "name": f"{scene_dir.name}/{stem}",
            "image": str(img.relative_to(scene_dir)),
            "depth": str(depth.relative_to(scene_dir)) if depth.is_file() else None,
            "label": str(label.relative_to(scene_dir)) if label.is_file() else None,
            "extrinsics": str(ext.relative_to(scene_dir)) if ext.is_file() else None,
        })
    return frames


def read_intrinsics(scene_dir, toolbox_dir=None):
    """读取相机内参 K。

    优先：场景目录下的 intrinsics.txt / K.txt；
    回退：SUNRGBDtoolbox Metadata（all2all.mat，MATLAB 格式，需 scipy.io）。
    TODO：按实际工具箱 Metadata 键名解析（depth_KL / K 等）。
    """
    for cand in ("intrinsics.txt", "K.txt", "camera_params.txt"):
        p = scene_dir / cand
        if p.is_file():
            return np.loadtxt(p).reshape(3, 3)
    if toolbox_dir is not None:
        meta = Path(toolbox_dir) / "Metadata" / "all2all.mat"
        if meta.is_file():
            try:
                from scipy.io import loadmat
                data = loadmat(str(meta))
                # TODO: 键名以官方工具箱为准
                for key in ("depth_KL", "K", "RGB_KL"):
                    if key in data:
                        return np.asarray(data[key]).reshape(3, 3)
            except Exception as e:
                logger.warning("读取 Metadata 失败: %s", e)
    raise FileNotFoundError(f"未找到内参文件: {scene_dir}")


def save_frame_rgb(frame, scene_dir, out_dir):
    """把帧 RGB 图转为 640×640 归一化 npz（供训练/可视化，可选）。"""
    import cv2

    img_path = scene_dir / frame["image"]
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("读取图片失败: %s", img_path)
        return None
    # letterbox 到 640×640（与 YOLOv8 预处理一致，见 detection/yolov8_feature）
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(640 / h, 640 / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    x0, y0 = (640 - nw) // 2, (640 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = canvas[..., ::-1]                      # BGR -> RGB
    np.savez_compressed(str(out_dir / f"{frame['name'].replace('/', '_')}.npz"),
                        rgb=rgb.astype(np.float32) / 255.0)


def parse_dataset(data_root, out_dir, toolbox_dir=None, save_rgb=False):
    """遍历整个数据集生成元数据索引。"""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    frames_dir = out_dir / "frames"
    if save_rgb:
        frames_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    for scene in scene_dirs:
        for f in find_frame_files(scene):
            f["K"] = read_intrinsics(scene, toolbox_dir).tolist()
            if save_rgb:
                save_frame_rgb(f, scene, frames_dir)
            frames.append(f)

    with open(out_dir / "frames_index.json", "w", encoding="utf-8") as fp:
        json.dump(frames, fp, ensure_ascii=False, indent=2)
    logger.info("索引 %d 帧 -> %s", len(frames), out_dir / "frames_index.json")
    return frames


def main():
    parser = argparse.ArgumentParser(description="解析 SUN RGB-D 数据集")
    parser.add_argument("--data_root", default="data/SUNRGBD",
                        help="SUN RGB-D 原始数据根目录")
    parser.add_argument("--toolbox_dir", default="data/SUNRGBDtoolbox",
                        help="官方工具箱目录（内参 Metadata）")
    parser.add_argument("--out_dir", default="results/preprocess",
                        help="输出目录")
    parser.add_argument("--save_rgb", action="store_true",
                        help="同时输出 640×640 归一化 RGB npz（供训练）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parse_dataset(args.data_root, args.out_dir,
                  toolbox_dir=args.toolbox_dir, save_rgb=args.save_rgb)


if __name__ == "__main__":
    main()
