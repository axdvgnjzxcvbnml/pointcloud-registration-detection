#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_lightweight.py — 轻量化模型微调（第四批交付）

功能
----
    基于融合模型权重（train_fusion.py 输出）微调轻量化头，
    共 20 个 epoch（configs/default.yaml: detection.train.finetune_epochs）。

    流程：
      1. 构建轻量化模型（DSConv1d 头，通道 128）；
      2. 加载融合模型 checkpoint（strict=False：主干/投影参数复用，
         头部因结构不同自动丢弃，重新初始化）；
      3. 冻结双主干，只训练轻量化头 + 检测头；
      4. 较小学习率微调 20 epoch。

用法
----
    python detection/finetune_lightweight.py \
        --config configs/default.yaml \
        --resume results/detection/fusion/fusion_epoch060.pth \
        --det_data_dir results/preprocess/detection \
        --out_dir results/detection/lightweight
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


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


def load_pretrained(model, ckpt_path, device="cpu"):
    """加载融合模型权重（strict=False，丢弃不匹配的头部键）。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.votebackbone.load_state_dict(
        state.get("votebackbone", {}), strict=False)
    model.yolo_feat.proj_conv.load_state_dict(
        state.get("yolo_proj_conv", {}), strict=False)
    # 融合头/检测头：轻量化结构不同，只加载形状匹配的键
    missing_h, _ = model.fusion_head.load_state_dict(
        state.get("fusion_head", {}), strict=False)
    missing_d, _ = model.det_head.load_state_dict(
        state.get("det_head", {}), strict=False)
    logger.info("加载预训练：主干缺失 %d，融合头缺失 %d，检测头缺失 %d（头部预期重建）",
                len(missing), len(missing_h), len(missing_d))
    return model


def finetune_lightweight(cfg, resume_ckpt, det_data_dir, frames_dir,
                         out_dir, votenet_ckpt, yolov8_ckpt, device="cuda"):
    from train_fusion import (SunRGBDDataset, compute_loss,
                              load_config, _collect_state)
    from torch.utils.data import DataLoader

    det_cfg = cfg["detection"]
    train_cfg = det_cfg["train"]
    finetune_epochs = train_cfg.get("finetune_epochs", 20)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(det_data_dir) / "split.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    torch.manual_seed(cfg.get("seed", 2024))

    train_ds = SunRGBDDataset(det_data_dir, frames_dir,
                              sample_names=[s["name"] for s in split["train"]])
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=train_cfg["num_workers"])

    model = build_lightweight_model(cfg, votenet_ckpt, yolov8_ckpt,
                                    freeze_backbone=True)
    model = load_pretrained(model, resume_ckpt, device="cpu").to(device)

    # 微调用较小学习率（默认 lr/10）
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=train_cfg["lr"] / 10.0,
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs)

    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        writer = None

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch, device=device)
            loss_dict = compute_loss(pred, batch,
                                     det_cfg["detect_head"]["vote_num"],
                                     det_cfg["detect_head"]["num_classes"])
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss_dict["total"]) * len(batch["point_cloud"])
            n += len(batch["point_cloud"])
        scheduler.step()
        logger.info("微调 Epoch %3d/%d | %.1fs | loss=%.4f | lr=%.6f",
                    epoch, finetune_epochs, time.time() - t0,
                    total_loss / max(n, 1), optimizer.param_groups[0]["lr"])
        if writer is not None:
            writer.add_scalar("finetune/loss", total_loss / max(n, 1), epoch)

        if epoch == finetune_epochs or epoch % 10 == 0:
            torch.save({"epoch": epoch, "cfg": cfg,
                        "model": _collect_state(model)},
                       out_dir / f"lightweight_epoch{epoch:03d}.pth")

    if writer is not None:
        writer.close()
    logger.info("轻量化微调完成，权重目录: %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="轻量化头微调（20 epoch）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", required=True,
                        help="融合模型权重（train_fusion.py 输出）")
    parser.add_argument("--det_data_dir", default="results/preprocess/detection")
    parser.add_argument("--frames_dir", default="results/preprocess/frames")
    parser.add_argument("--out_dir", default="results/detection/lightweight")
    parser.add_argument("--votenet_ckpt", default="weights/votenet_sunrgbd.pth")
    parser.add_argument("--yolov8_ckpt", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finetune_epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from train_fusion import load_config
    cfg = load_config(args.config)
    if args.finetune_epochs:
        cfg["detection"]["train"]["finetune_epochs"] = args.finetune_epochs

    finetune_lightweight(cfg, args.resume, args.det_data_dir, args.frames_dir,
                         args.out_dir, args.votenet_ckpt, args.yolov8_ckpt,
                         device=args.device)


if __name__ == "__main__":
    main()
