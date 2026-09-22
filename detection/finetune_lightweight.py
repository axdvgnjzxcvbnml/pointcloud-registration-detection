#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_lightweight.py — 轻量化模型微调（第四批交付）

流程
----
    1. 构建轻量化模型（DSConv1d 头，通道 128）；
    2. 加载融合模型权重（按形状匹配迁移，头部从零训练）；
    3. 冻结双主干，仅微调轻量化头 + 检测头 20 epoch。

用法
----
    python detection/finetune_lightweight.py \
        --config configs/ablation/04_fusion_lightweight.yaml \
        --resume results/detection/fusion/fusion_epoch060.pth \
        --out_dir results/detection/lightweight
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finetune_lightweight")


def build_lightweight_model(cfg, votenet_ckpt, yolov8_ckpt,
                            freeze_backbone=True):
    """构建带轻量化头的融合模型（复用 train_fusion 的装配）。"""
    from train_fusion import FusionModel, load_config
    from lightweight_head import LightweightFusionHead, LightweightDetectionHead

    det_cfg = cfg["detection"]
    fc = det_cfg["fusion"]
    lw = cfg.get("lightweight", {})

    model = FusionModel(cfg, votenet_ckpt, yolov8_ckpt,
                        freeze_backbone=freeze_backbone)
    # 替换为轻量化头（通道 256 -> 128，DSConv1d，Dropout 0.3）
    model.fusion_head = LightweightFusionHead(
        fusion_mode=fc["method"],
        pc_dim=det_cfg["votenet_feature_dim"],
        img_dim=det_cfg["projected_channels"],
        hidden_dim=lw.get("hidden_dim", 128),
        num_heads=fc["attn_num_heads"],
        dropout=lw.get("dropout", 0.3))
    model.det_head = LightweightDetectionHead(
        in_dim=lw.get("hidden_dim", 128),
        num_classes=det_cfg["detect_head"]["num_classes"],
        vote_num=det_cfg["detect_head"]["vote_num"],
        dropout=lw.get("dropout", 0.3))
    return model


def _filter_by_shape(module, state):
    """只保留键存在且参数形状一致的项。

    轻量化微调时头部通道从 256 -> 128，融合头/检测头的权重形状
    与教师权重不一致（load_state_dict 的 strict=False 仍会对形状
    不匹配抛 RuntimeError），因此先按 shape 过滤，只迁移可复用的
    主干/投影层权重，头部按初始化从零训练。
    """
    model_state = module.state_dict()
    return {k: v for k, v in (state or {}).items()
            if k in model_state and model_state[k].shape == v.shape}


def load_pretrained(model, ckpt_path, device="cpu"):
    """加载融合模型权重（按形状匹配迁移，跳过重建的轻量化头部键）。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.votebackbone.load_state_dict(
        _filter_by_shape(model.votebackbone, state.get("votebackbone", {})),
        strict=False)
    model.yolo_feat.proj_conv.load_state_dict(
        _filter_by_shape(model.yolo_feat.proj_conv, state.get("yolo_proj_conv", {})),
        strict=False)
    fh_state = state.get("fusion_head", {})
    dh_state = state.get("det_head", {})
    fh_fit = _filter_by_shape(model.fusion_head, fh_state)
    dh_fit = _filter_by_shape(model.det_head, dh_state)
    missing_h, _ = model.fusion_head.load_state_dict(fh_fit, strict=False)
    missing_d, _ = model.det_head.load_state_dict(dh_fit, strict=False)
    logger.info(
        "加载预训练：主干缺失 %d；融合头迁移 %d/%d 键、检测头迁移 %d/%d 键（形状不符的键从零训练）",
        len(missing), len(fh_fit), len(fh_state), len(dh_fit), len(dh_state))
    return model
