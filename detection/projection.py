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
        feature_map: (fh, fw, C) 特征图（高, 宽, 通道，HWC 惯例；
                     由 torch 的 (C,fh,fw) 经 transpose(1,2,0) 得到）。
        uv: (N, 2) [u_f, v_f]（u 对应宽 W，v 对应高 H）。
        valid: (N,) bool。
    Returns:
        (N, C) float32；无效点返回 0。
    """
    fh, fw, C = feature_map.shape
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
    # HWC：第一索引为 v（行/高），第二索引为 u（列/宽）
    out = (feat[v0, u0] * (1 - du) * (1 - dv)
           + feat[v0, u1] * du * (1 - dv)
           + feat[v1, u0] * (1 - du) * dv
           + feat[v1, u1] * du * dv)
    out[~valid] = 0.0
    return out


class ProjectionModule:
    """torch 版本投影 + 双线性采样（训练/推理用）。

    实现为**整批向量化的纯 torch 运算**：投影、letterbox、归一化全部在
    GPU 上一次完成，不在 batch 维做 Python 循环、不做 CPU-GPU 往返
    （旧实现逐样本 detach().cpu().numpy() 会在训练时造成同步瓶颈）。
    """

    def __init__(self, image_size=640):
        self.image_size = image_size

    def forward(self, img_feat, pc_xyz, K, extrinsic=None, letterbox=None):
        """在图像特征图上采样种子点特征。

        Args:
            img_feat: (B, C, fh, fw) 图像特征（YOLOv8 P3 升维后）。
            pc_xyz: (B, N, 3) 种子点坐标（相机系；世界系需给 extrinsic）。
            K: (B, 3, 3) 相机内参（tensor 或 numpy 均可）。
            extrinsic: (B, 3, 4) / (B, 4, 4) 可选外参（世界系->相机系）。
            letterbox: None / 三元组 (scale, pad_x, pad_y)（全 batch 相同）
                       / (B, 3)（逐样本不同，数据集应传此形式）。
        Returns:
            sampled: (B, N, C) float32，无效点置 0。
            valid: (B, N) bool。
        """
        import torch
        import torch.nn.functional as F

        B, C, fh, fw = img_feat.shape
        device, dtype = img_feat.device, img_feat.dtype
        N = pc_xyz.shape[1]

        def _as_batch_tensor(x):
            if not torch.is_tensor(x):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return x.to(device=device, dtype=dtype)

        pts = _as_batch_tensor(pc_xyz)                    # (B, N, 3)
        Kb = _as_batch_tensor(K)                         # (B, 3, 3)

        # 可选外参：世界系 -> 相机系（整批矩阵乘）
        if extrinsic is not None:
            E = _as_batch_tensor(extrinsic)[:, :3, :]    # (B, 3, 4)
            R, t = E[:, :, :3], E[:, :, 3]               # (B,3,3),(B,3)
            pts = torch.einsum("bij,bnj->bni", R, pts) + t.unsqueeze(1)

        fx = Kb[:, 0, 0].view(B, 1)
        fy = Kb[:, 1, 1].view(B, 1)
        cx = Kb[:, 0, 2].view(B, 1)
        cy = Kb[:, 1, 2].view(B, 1)

        z = pts[..., 2]                                  # (B, N)
        safe_z = z.clamp_min(1e-9)
        u = fx * pts[..., 0] / safe_z + cx               # 原图像素坐标
        v = fy * pts[..., 1] / safe_z + cy

        # letterbox：u_new = u*scale + pad（支持逐样本参数）
        if letterbox is not None:
            lb = letterbox
            if not torch.is_tensor(lb):
                lb = torch.as_tensor(np.asarray(lb), dtype=torch.float32)
            lb = lb.to(device=device, dtype=dtype)
            if lb.dim() == 1:
                lb = lb.unsqueeze(0).expand(B, 3)
            scale, pad_x, pad_y = lb[:, 0:1], lb[:, 1:2], lb[:, 2:3]
            u = u * scale + pad_x
            v = v * scale + pad_y

        # 原图 -> 特征图坐标
        u_f = u * (fw / self.image_size)
        v_f = v * (fh / self.image_size)

        valid = ((z > 0)
                 & (u_f >= 0) & (u_f < fw - 1e-6)
                 & (v_f >= 0) & (v_f < fh - 1e-6))

        # 归一化到 [-1, 1]：u_f 以“像素角点”为坐标（0..fw-1），
        # 与 numpy 参考实现的 floor 双线性插值口径一致，故用
        # align_corners=True（像素 i 中心 ↔ 归一化坐标 2i/(W-1)-1）。
        u_n = u_f / max(fw - 1, 1) * 2.0 - 1.0
        v_n = v_f / max(fh - 1, 1) * 2.0 - 1.0
        grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(2)   # (B,N,1,2)

        sampled = F.grid_sample(img_feat, grid, mode="bilinear",
                                padding_mode="zeros",
                                align_corners=True)
        sampled = sampled.squeeze(3).transpose(1, 2)          # (B, N, C)
        sampled = sampled * valid.unsqueeze(-1).to(dtype)
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
