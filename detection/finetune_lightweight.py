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
    # 首次微调：--resume 传入融合模型（教师）权重
    python detection/finetune_lightweight.py \
        --config configs/default.yaml \
        --resume results/detection/fusion/fusion_epoch060.pth \
        --det_data_dir results/preprocess/detection \
        --out_dir results/detection/lightweight

    # 断点续训：--resume_train 传入轻量化训练自身的 checkpoint
    python detection/finetune_lightweight.py \
        --config configs/default.yaml \
        --resume results/detection/fusion/fusion_epoch060.pth \
        --resume_train results/detection/lightweight/latest.pth \
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


def _resume_train_state(model, ckpt_path, optimizer, scheduler, start_key="epoch"):
    """从轻量化 checkpoint 恢复训练状态（模型/优化器/调度器/epoch）。

    返回 (start_epoch, best_metric)。模型权重按形状过滤加载，兼容
    train_fusion 产出的 checkpoint（头部形状不同时只迁移可复用层）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    from train_fusion import _load_state
    _load_state(model, state)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            logger.warning("调度器状态恢复失败，按新周期继续: %s", e)
    start_epoch = int(ckpt.get(start_key, 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("inf")))
    logger.info("从断点恢复: %s（已训至 epoch %d，best_metric=%.6f）",
                ckpt_path, start_epoch - 1, best_metric)
    return start_epoch, best_metric


def finetune_lightweight(cfg, resume_ckpt, det_data_dir, frames_dir,
                         out_dir, votenet_ckpt, yolov8_ckpt, device="cuda",
                         resume_train=None):
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

    use_image_branch = cfg.get("ablation", {}).get("use_image_branch", True)
    train_ds = SunRGBDDataset(det_data_dir, frames_dir,
                              sample_names=[s["name"] for s in split["train"]],
                              use_image_branch=use_image_branch)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=train_cfg["num_workers"])
    val_ds = SunRGBDDataset(det_data_dir, frames_dir,
                            sample_names=[s["name"] for s in split["val"]],
                            use_image_branch=use_image_branch)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=train_cfg["num_workers"])

    model = build_lightweight_model(cfg, votenet_ckpt, yolov8_ckpt,
                                    freeze_backbone=True)
    model = load_pretrained(model, resume_ckpt, device="cpu").to(device)

    # 微调用较小学习率（默认 lr/10）
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=train_cfg["lr"] / 10.0,
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs)

    # ---- 断点续训（--resume_train：轻量化训练自身的 checkpoint） ----
    start_epoch, best_metric = 1, float("inf")
    if resume_train:
        if not Path(resume_train).is_file():
            raise FileNotFoundError(f"断点文件不存在: {resume_train}")
        start_epoch, best_metric = _resume_train_state(
            model, resume_train, optimizer, scheduler)
        if start_epoch > finetune_epochs:
            logger.warning("断点 epoch %d 已超过目标 %d，直接结束",
                           start_epoch - 1, finetune_epochs)

    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        writer = None

    for epoch in range(start_epoch, finetune_epochs + 1):
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
        epoch_loss = total_loss / max(n, 1)
        logger.info("微调 Epoch %3d/%d | %.1fs | loss=%.4f | lr=%.6f",
                    epoch, finetune_epochs, time.time() - t0,
                    epoch_loss, optimizer.param_groups[0]["lr"])
        if writer is not None:
            writer.add_scalar("finetune/loss", epoch_loss, epoch)

        # 验证集 loss（best 指标；mAP 由 evaluate_detection.py 评测）
        val_loss, nv = 0.0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch, device=device)
                ld = compute_loss(pred, batch,
                                  det_cfg["detect_head"]["vote_num"],
                                  det_cfg["detect_head"]["num_classes"])
                val_loss += float(ld["total"]) * len(batch["point_cloud"])
                nv += len(batch["point_cloud"])
        model.train()
        val_loss = val_loss / max(nv, 1)
        logger.info("  val loss=%.4f（best=%.4f）", val_loss, best_metric)
        if writer is not None:
            writer.add_scalar("finetune/val_loss", val_loss, epoch)
        if val_loss < best_metric:
            best_metric = val_loss
            torch.save({"epoch": epoch, "cfg": cfg,
                        "model": _collect_state(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_metric": best_metric},
                       out_dir / "lightweight_best.pth")
            logger.info("  best 更新 -> %s", out_dir / "lightweight_best.pth")

        # 每 epoch 覆盖 latest.pth（断点续训用），周期/末尾保留命名权重
        torch.save({"epoch": epoch, "cfg": cfg,
                    "model": _collect_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric},
                   out_dir / "latest.pth")
        if epoch == finetune_epochs or epoch % 10 == 0:
            torch.save({"epoch": epoch, "cfg": cfg,
                        "model": _collect_state(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_metric": best_metric},
                       out_dir / f"lightweight_epoch{epoch:03d}.pth")

    if writer is not None:
        writer.close()
    logger.info("轻量化微调完成，权重目录: %s（断点续训：--resume_train %s）",
                out_dir, out_dir / "latest.pth")


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
    parser.add_argument("--resume_train", default=None,
                        help="轻量化训练断点（finetune_lightweight 输出的 "
                             "latest.pth / lightweight_epochNNN.pth），"
                             "从保存的 epoch 继续微调；区别于 --resume（教师权重）")
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
                         device=args.device, resume_train=args.resume_train)


if __name__ == "__main__":
    main()
