#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projection.py — 3D 种子点 → 2D 像素投影 + 双线性采样（第四批交付）

功能
----
    1. 通过相机内参 K 将 3D 种子点投影到 2D 像素坐标；
    2. 可选经外参变换（世界系 → 相机系）；
    3. 可选应用 letterbox 参数（对齐 YOLOv8 预处理）；
    4. 在图像特征图上双线性采样，得到每个种子点的图像特征。

坐标链路
--------
    3D 点 (相机系) --K--> 像素 (u,v) --letterbox--> 640 图坐标
    --缩放--> 特征图坐标 (80×80) --双线性采样--> 特征向量 (C,)

说明
----
    - project_points 为纯 numpy 实现（可单测、可离线可视化）；
    - ProjectionModule 为 torch 版本（训练/推理用，grid_sample 实现）。
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def project_points(points, K, extrinsic=None, letterbox=None,
                   feature_size=(80, 80), image_size=640):
    """把 3D 点投影到特征图坐标（纯 numpy）。

    Args:
        points: (N, 3) 点云，米制。相机系；若为世界系需给 extrinsic。
        K: (3, 3) 相机内参。
        extrinsic: (3, 4) 或 (4, 4) 相机外参（世界系->相机系），可选。
        letterbox: (scale, pad_x, pad_y) 三元组，YOLOv8 预处理参数，可选。
        feature_size: (fw, fh) 特征图宽高（默认 80×80）。
        image_size: 网络输入边长（默认 640）。
    Returns:
        uv: (N, 2) 特征图坐标 [u_f, v_f]
        valid: (N,) bool，z>0 且在特征图范围内
    """
    pts = np.asarray(points, dtype=np.float64)
    if extrinsic is not None:
        E = np.asarray(extrinsic, dtype=np.float64)
        R, t = E[:3, :3], E[:3, 3]
        pts = (R @ pts.T).T + t

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = pts[:, 2]
    u = fx * pts[:, 0] / np.maximum(z, 1e-9) + cx
    v = fy * pts[:, 1] / np.maximum(z, 1e-9) + cy

    if letterbox is not None:
        scale, pad_x, pad_y = letterbox
        u = u * scale + pad_x
        v = v * scale + pad_y

    fw, fh = feature_size
    u_f = u * (fw / image_size)
    v_f = v * (fh / image_size)

    valid = ((z > 0)
             & (u_f >= 0) & (u_f < fw - 1e-6)
             & (v_f >= 0) & (v_f < fh - 1e-6))
    uv = np.stack([u_f, v_f], axis=1)
    return uv, valid


def bilinear_sample_numpy(feature_map, uv, valid):
    """特征图双线性采样（纯 numpy，用于验证/离线）。

    Args:
        feature_map: (fw, fh, C) 特征图（宽, 高, 通道）。
        uv: (N, 2) [u_f, v_f]。
        valid: (N,) bool。
    Returns:
        (N, C) float32；无效点返回 0。
    """
    fw, fh, C = feature_map.shape
    u, v = uv[:, 0], uv[:, 1]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u0 = np.clip(u0, 0, fw - 1)
    v0 = np.clip(v0, 0, fh - 1)
    u1 = np.clip(u0 + 1, 0, fw - 1)
    v1 = np.clip(v0 + 1, 0, fh - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]

    feat = feature_map.astype(np.float32)
    out = (feat[u0, v0] * (1 - du) * (1 - dv)
           + feat[u1, v0] * du * (1 - dv)
           + feat[u0, v1] * (1 - du) * dv
           + feat[u1, v1] * du * dv)
    out[~valid] = 0.0
    return out


class ProjectionModule:
    """torch 版本投影 + 双线性采样（训练/推理用）。"""

    def __init__(self, image_size=640):
        import torch.nn as nn
        self.image_size = image_size

    def forward(self, img_feat, pc_xyz, K, extrinsic=None, letterbox=None):
        """在图像特征图上采样种子点特征。

        Args:
            img_feat: (B, C, fh, fw) 图像特征（YOLOv8 P3 升维后）。
            pc_xyz: (B, N, 3) 种子点坐标。
            K: (B, 3, 3) 相机内参。
            extrinsic: (B, 3, 4) 可选外参。
            letterbox: (scale, pad_x, pad_y) 或 None。
        Returns:
            sampled: (B, N, C) float32，无效点置 0。
            valid: (B, N) bool。
        """
        import torch
        import torch.nn.functional as F

        B, C, fh, fw = img_feat.shape
        N = pc_xyz.shape[1]

        # 逐样本投影（batch 内 K 可不同）
        uv_list, valid_list = [], []
        for b in range(B):
            Kb = K[b].detach().cpu().numpy() if torch.is_tensor(K) else K[b]
            ext = None
            if extrinsic is not None:
                ext = (extrinsic[b].detach().cpu().numpy()
                       if torch.is_tensor(extrinsic) else extrinsic[b])
            uv, valid = project_points(
                pc_xyz[b].detach().cpu().numpy(), Kb, extrinsic=ext,
                letterbox=letterbox, feature_size=(fw, fh),
                image_size=self.image_size)
            uv_list.append(torch.from_numpy(uv).float().to(img_feat.device))
            valid_list.append(torch.from_numpy(valid).to(img_feat.device))
        uv = torch.stack(uv_list, dim=0)       # (B, N, 2)
        valid = torch.stack(valid_list, dim=0)  # (B, N)

        # grid: (B, N, 1, 2)，坐标归一化到 [-1, 1]
        u_n = uv[..., 0] / max(fw - 1, 1) * 2.0 - 1.0
        v_n = uv[..., 1] / max(fh - 1, 1) * 2.0 - 1.0
        grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(2)  # (B, N, 1, 2)

        sampled = F.grid_sample(img_feat, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(3).transpose(1, 2)         # (B, N, C)
        sampled = sampled * valid.unsqueeze(-1).float()
        return sampled, valid

    def __call__(self, img_feat, pc_xyz, K, extrinsic=None, letterbox=None):
        return self.forward(img_feat, pc_xyz, K, extrinsic, letterbox)


def main():
    parser = argparse.ArgumentParser(description="3D→2D 投影与采样（调试/离线验证）")
    parser.add_argument("--points", required=True, help="点云文件（.npy/.npz/.ply）")
    parser.add_argument("--K", required=True, help="内参（3x3 txt）")
    parser.add_argument("--extrinsic", default=None, help="外参（3x4 txt，可选）")
    parser.add_argument("--feature", required=True, help="特征图（.npz，feature 字段 (C,80,80)）")
    parser.add_argument("--out", default="results/detection/projected.npz",
                        help="采样特征输出")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.points.endswith(".ply"):
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(args.points).points)
    elif args.points.endswith(".npz"):
        points = np.load(args.points)["points"]
    else:
        points = np.load(args.points)
    K = np.loadtxt(args.K).reshape(3, 3)
    ext = np.loadtxt(args.extrinsic).reshape(3, 4) if args.extrinsic else None

    feat = np.load(args.feature)["feature"]        # (C, 80, 80)
    C, fw, fh = feat.shape
    uv, valid = project_points(points, K, extrinsic=ext,
                               feature_size=(fw, fh), image_size=640)
    sampled = bilinear_sample_numpy(feat.transpose(1, 2, 0), uv, valid)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), uv=uv, valid=valid, sampled=sampled)
    logger.info("有效点 %d/%d，采样特征 -> %s", int(valid.sum()), len(points), out_path)


if __name__ == "__main__":
    main()
