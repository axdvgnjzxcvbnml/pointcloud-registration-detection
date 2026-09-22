#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lightweight_head.py — 轻量化融合头/检测头（第四批交付）

功能
----
    把原始 Conv1d 替换为深度可分离 1D 卷积（DSConv1d）：
      - 深度卷积（depthwise，groups=in_channels）+ 逐点卷积（1×1）；
      - 通道数从 256 减半至 128；
      - 添加 Dropout(0.3)。

    提供与 fusion_head.py 相同接口的：
      - LightweightFusionHead（fusion_mode=concat / attention）
      - LightweightDetectionHead
    用于消融实验 5（融合+轻量化）与 finetune_lightweight.py。

说明
----
    DSConv1d 默认 kernel_size=1（在通道维度上做逐点/深度可分离变换）。
    若需在种子维度上卷积（kernel>1），需先保证种子点顺序稳定
    （例如按坐标排序），谨慎使用。
"""

import argparse
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DSConv1d(nn.Module):
    """深度可分离 1D 卷积：depthwise Conv1d + pointwise Conv1d。"""

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dropout=0.3):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """x: (B, C, L) -> (B, out_channels, L)"""
        return self.dropout(self.pointwise(self.depthwise(x)))


class _LightweightProjector(nn.Module):
    """DSConv1d 投影块：DSConv -> ReLU -> DSConv（通道 256->128）。"""

    def __init__(self, in_dim, hidden_dim=128, dropout=0.3, conv_kernel=1):
        super().__init__()
        self.net = nn.Sequential(
            DSConv1d(in_dim, hidden_dim, kernel_size=conv_kernel, dropout=dropout),
            nn.ReLU(inplace=True),
            DSConv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, dropout=dropout),
        )

    def forward(self, x):
        """x: (B, N, C) -> (B, N, hidden_dim)"""
        x = x.transpose(1, 2)                 # (B, C, N) 种子维度为长度维
        x = self.net(x)
        return x.transpose(1, 2)


class LightweightFusionHead(nn.Module):
    """轻量化融合头（通道 256 -> 128，DSConv1d，Dropout 0.3）。

    接口与 fusion_head.FusionHead 一致：
        forward(pc_feat (B,N,pc_dim), img_feat (B,N,img_dim)) -> (B,N,hidden_dim)
    """

    def __init__(self, fusion_mode="concat", pc_dim=128, img_dim=128,
                 hidden_dim=128, num_heads=8, dropout=0.3, conv_kernel=1):
        super().__init__()
        if fusion_mode not in ("concat", "attention"):
            raise ValueError(f"fusion_mode 仅支持 concat/attention，当前 {fusion_mode}")
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            in_dim = pc_dim + img_dim          # 256
            self.projector = _LightweightProjector(in_dim, hidden_dim,
                                                   dropout, conv_kernel)
        else:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=pc_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(pc_dim)
            self.projector = _LightweightProjector(pc_dim, hidden_dim,
                                                   dropout, conv_kernel)

    def forward(self, pc_feat, img_feat):
        if self.fusion_mode == "concat":
            fused = torch.cat([pc_feat, img_feat], dim=-1)
        else:
            attn_out, _ = self.cross_attn(pc_feat, img_feat, img_feat)
            fused = self.norm(pc_feat + attn_out)
        return self.projector(fused)


class LightweightDetectionHead(nn.Module):
    """轻量化检测头（隐藏维度 128，Dropout 0.3）。

    接口与 fusion_head.DetectionHead 一致。
    """

    def __init__(self, in_dim=128, num_classes=10, vote_num=10, dropout=0.3):
        super().__init__()
        self.vote_num = vote_num
        self.num_classes = num_classes

        def _mlp(out):
            return nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(128, out))

        self.objectness = _mlp(1)
        self.center = _mlp(3 * vote_num)
        self.size = _mlp(3 * vote_num)
        self.heading = _mlp(2 * vote_num)
        self.class_scores = _mlp(num_classes)

    def forward(self, fused):
        return {
            "objectness": self.objectness(fused),
            "center": self.center(fused).view(*fused.shape[:2], self.vote_num, 3),
            "size": self.size(fused).view(*fused.shape[:2], self.vote_num, 3),
            "heading": self.heading(fused).view(*fused.shape[:2], self.vote_num, 2),
            "class_scores": self.class_scores(fused),
        }


def build_lightweight_heads(fusion_mode="concat", pc_dim=128, img_dim=128,
                            hidden_dim=128, num_heads=8, dropout=0.3,
                            num_classes=10, vote_num=10):
    """便捷构建轻量化双头，返回 (fusion_head, det_head)。"""
    fusion_head = LightweightFusionHead(
        fusion_mode, pc_dim, img_dim, hidden_dim, num_heads, dropout)
    det_head = LightweightDetectionHead(hidden_dim, num_classes, vote_num, dropout)
    return fusion_head, det_head


def count_parameters(module):
    """统计可训练参数量（单位：百万）。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


def main():
    parser = argparse.ArgumentParser(description="轻量化头结构自检")
    parser.add_argument("--fusion_mode", choices=["concat", "attention"],
                        default="concat")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--vote_num", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    head, det = build_lightweight_heads(args.fusion_mode,
                                        hidden_dim=args.hidden_dim,
                                        num_classes=args.num_classes,
                                        vote_num=args.vote_num)
    B, N = 2, 256
    pc = torch.randn(B, N, 128)
    img = torch.randn(B, N, 128)
    fused = head(pc, img)
    out = det(fused)
    for k, v in out.items():
        logger.info("%-15s %s", k, tuple(v.shape))

    total = count_parameters(head) + count_parameters(det)
    logger.info("轻量化头合计 %.3fM（融合头 %.3fM + 检测头 %.3fM）",
                total, count_parameters(head), count_parameters(det))


if __name__ == "__main__":
    main()
