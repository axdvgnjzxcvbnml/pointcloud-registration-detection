#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_head.py — 点云×图像融合头 + 检测头（第四批交付）

功能
----
    两种融合方式（由 fusion_mode 切换）：
      - concat（主融合）：点云特征 ⊕ 图像特征 -> Linear 投影
      - attention（消融对比）：Cross-Attention
        （query=点云特征, key/value=图像特征，多头）
    融合后接 VoteNet 风格检测头：
      - objectness：每个种子点一个目标性分数
      - center：投票向量（每点 vote_num 个中心偏移）
      - size：框尺寸
      - heading：朝向（sin/cos 编码，简化；可替换官方 bin 方案）
      - class_scores：类别分数

参数量
------
    目标：融合头 + 检测头合计约 2M。
    实际值以 evaluate_detection.py 的 count_parameters 为准，
    可通过 hidden_dim / vote_num 调节（见配置文件 detection.*）。

用法
----
    from fusion_head import FusionHead, DetectionHead
    head = FusionHead(fusion_mode="concat", pc_dim=128, img_dim=128,
                      hidden_dim=256, num_heads=8)
    det  = DetectionHead(in_dim=256, num_classes=10, vote_num=10)
"""

import argparse
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    """两层 MLP：Linear(hidden) + ReLU + [Dropout] + Linear(out)。"""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class FusionHead(nn.Module):
    """点云特征与图像特征的融合头。

    Args:
        fusion_mode: 'concat' 或 'attention'。
        pc_dim: 点云分支特征维度（VoteNet 种子点，默认 128）。
        img_dim: 图像分支特征维度（YOLOv8 P3 升维后，默认 128）。
        hidden_dim: 融合输出维度（默认 256）。
        num_heads: Cross-Attention 头数（attention 模式）。
        dropout: 融合头 Dropout。
    """

    def __init__(self, fusion_mode="concat", pc_dim=128, img_dim=128,
                 hidden_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        if fusion_mode not in ("concat", "attention"):
            raise ValueError(f"fusion_mode 仅支持 concat/attention，当前 {fusion_mode}")
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            self.projector = nn.Sequential(
                nn.Linear(pc_dim + img_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
        else:  # attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=pc_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(pc_dim)
            self.projector = nn.Sequential(
                nn.Linear(pc_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

    def forward(self, pc_feat, img_feat):
        """融合点云与图像特征。

        Args:
            pc_feat: (B, N, pc_dim) 种子点特征。
            img_feat: (B, N, img_dim) 投影采样得到的图像特征（无效点已置 0）。
        Returns:
            (B, N, hidden_dim) 融合特征。
        """
        if self.fusion_mode == "concat":
            fused = torch.cat([pc_feat, img_feat], dim=-1)
        else:
            attn_out, _ = self.cross_attn(pc_feat, img_feat, img_feat)
            fused = self.norm(pc_feat + attn_out)   # 残差 + LayerNorm
        return self.projector(fused)


class DetectionHead(nn.Module):
    """VoteNet 风格检测头（作用于每个种子点的融合特征）。

    Args:
        in_dim: 融合特征维度（默认 256）。
        num_classes: 类别数（SUN RGB-D 为 10）。
        vote_num: 每个种子点生成的投票数（默认 10）。
        hidden_dim: 内部 MLP 隐藏维度。
    """

    def __init__(self, in_dim=256, num_classes=10, vote_num=10,
                 hidden_dim=256, dropout=0.0):
        super().__init__()
        self.vote_num = vote_num
        self.num_classes = num_classes

        # 每个分支：两层 MLP，输出形状含 vote_num 维度
        self.objectness = build_mlp(in_dim, hidden_dim, 1, dropout)
        self.center = build_mlp(in_dim, hidden_dim, 3 * vote_num, dropout)
        self.size = build_mlp(in_dim, hidden_dim, 3 * vote_num, dropout)
        self.heading = build_mlp(in_dim, hidden_dim, 2 * vote_num, dropout)
        self.class_scores = build_mlp(in_dim, hidden_dim, num_classes, dropout)

    def forward(self, fused):
        """返回检测头输出字典（各字段均 (B, N, ...)）。"""
        return {
            "objectness": self.objectness(fused),                     # (B,N,1)
            "center": self.center(fused).view(*fused.shape[:2], self.vote_num, 3),
            "size": self.size(fused).view(*fused.shape[:2], self.vote_num, 3),
            "heading": self.heading(fused).view(*fused.shape[:2], self.vote_num, 2),
            "class_scores": self.class_scores(fused),                 # (B,N,10)
        }


def count_parameters(module):
    """统计可训练参数量（单位：百万）。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


def main():
    parser = argparse.ArgumentParser(description="融合头/检测头结构自检")
    parser.add_argument("--fusion_mode", choices=["concat", "attention"],
                        default="concat")
    parser.add_argument("--pc_dim", type=int, default=128)
    parser.add_argument("--img_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--vote_num", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    head = FusionHead(args.fusion_mode, args.pc_dim, args.img_dim,
                      args.hidden_dim, args.num_heads)
    det = DetectionHead(args.hidden_dim, args.num_classes, args.vote_num)

    B, N = 2, 256
    pc = torch.randn(B, N, args.pc_dim)
    img = torch.randn(B, N, args.img_dim)
    fused = head(pc, img)
    out = det(fused)
    for k, v in out.items():
        logger.info("%-15s %s", k, tuple(v.shape))

    total = count_parameters(head) + count_parameters(det)
    logger.info("融合头 %.3fM + 检测头 %.3fM = 合计 %.3fM（目标约 2M，可调 hidden_dim）",
                count_parameters(head), count_parameters(det), total)


if __name__ == "__main__":
    main()
