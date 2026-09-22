#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fusion.py — 融合模型训练（第四批交付）

功能
----
    冻结 VoteNet 主干与 YOLOv8n 主干，只训练融合头 + 检测头，
    默认 60 个 epoch（configs/default.yaml: detection.train.epochs）。

    模型结构：
        点云分支 : VoteNet 主干（Pointnet2Backbone）→ 种子点特征 (N,128)
        图像分支 : YOLOv8n P3（80×80×64）→ 1×1 Conv(128) → 投影采样 (N,128)
        融合头   : Concat+1×1 / Cross-Attention（fusion_head.py）
        检测头   : objectness / center / size / heading / class（fusion_head.py）

依赖
----
    - external/votenet（含编译好的 PointNet2 算子）
    - yolov8n.pt（首次运行自动下载）
    数据：results/preprocess/detection/*.npz（generate_detection_data.py 输出）

用法
----
    python detection/train_fusion.py --config configs/default.yaml \
        --det_data_dir results/preprocess/detection \
        --frames_dir results/preprocess/frames \
        --out_dir results/detection/fusion
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------
def _deep_merge(base, override):
    """递归合并两个配置字典：override 覆盖 base，子字典递归合并。

    用于消融配置（configs/ablation/*.yaml）以 base 继承 default.yaml。
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    """加载 YAML 配置。

    支持 base 继承：若配置含 `base: <相对路径>`，先加载 base 再递归覆盖
    （configs/ablation/*.yaml 均以 ../default.yaml 为 base）。
    """
    import yaml

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "base" in cfg:
        base_path = path.parent / cfg.pop("base")
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


# ---------------------------------------------------------------
# VoteNet 主干封装
# ---------------------------------------------------------------
class VoteNetBackbone(nn.Module):
    """VoteNet 点云主干（Pointnet2Backbone）封装，暴露种子点特征。

    说明：官方 Pointnet2Backbone.forward(xyz) 返回 [xyz, features, ...]，
    features 为 (B, C, N)，此处统一转成 (B, N, C)。
    TODO：以 external/votenet/models/backbone_module.py 实际签名为准。
    """

    def __init__(self, ckpt_path=None, input_feature_dim=0, freeze=True):
        super().__init__()
        try:
            from external.votenet.models.backbone_module import Pointnet2Backbone
        except ImportError as e:
            raise ImportError(
                "无法导入 external/votenet，请先克隆并编译 PointNet2 算子"
                "（见 docs/setup.md，TORCH_CUDA_ARCH_LIST=7.0）") from e

        self.backbone = Pointnet2Backbone(input_feature_dim=input_feature_dim)
        if ckpt_path:
            self._load_weights(ckpt_path)
        if freeze:
            self.freeze()

    def _load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = (ckpt.get("state_dict", ckpt.get("model", ckpt))
                 if isinstance(ckpt, dict) else ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, _ = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning("主干缺失权重 %d 个（首 5: %s）", len(missing), missing[:5])

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, xyz):
        """xyz: (B, N, 3) -> (seed_xyz (B,N,3), seed_features (B,N,C))"""
        out = self.backbone(xyz)
        seed_xyz, seed_features = out[0], out[1]
        if seed_features.dim() == 3:
            seed_features = seed_features.transpose(1, 2)
        return seed_xyz, seed_features


def build_votenet_model(num_class=10):
    """构建完整 VoteNet 模型（votenet_baseline.py 使用）。

    官方 VoteNet 构造参数：num_class / num_heading_bin / num_size_cluster /
    mean_size_arr。mean_size_arr 请从 external/votenet 的 SUN RGB-D
    数据处理代码中加载（TODO 核对）。
    """
    try:
        from external.votenet.models.votenet import VoteNet
    except ImportError as e:
        raise ImportError("无法导入 external/votenet（先克隆并编译）") from e
    model = VoteNet(num_class=num_class, num_heading_bin=12,
                    num_size_cluster=10, mean_size_arr=None)
    return model


# ---------------------------------------------------------------
# 数据集（检测数据 npz + 帧 rgb/K npz）
# ---------------------------------------------------------------
class SunRGBDDataset(Dataset):
    """VoteNet 风格检测数据集。

    每个样本（det_data_dir/{name}.npz）：
        point_cloud (N,3) / bboxes (M,7) / class_ids (M,)
        vote_label (N,3) / vote_mask (N,)
    帧图像与内参（frames_dir/{name}.npz）：rgb (H,W,3) / K (3,3)
    """

    def __init__(self, det_data_dir, frames_dir=None, sample_names=None,
                 use_image_branch=False):
        det_dir = Path(det_data_dir)
        self.files = sorted(det_dir.glob("*.npz"))
        if sample_names is not None:
            name_set = set(sample_names)
            self.files = [f for f in self.files if f.stem in name_set]
        if not self.files:
            raise FileNotFoundError(f"检测数据目录为空: {det_dir}")
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.use_image_branch = use_image_branch
        self._missing_warned = 0
        if use_image_branch and self.frames_dir is None:
            logger.warning(
                "图像分支已开启（use_image_branch=True）但未提供 frames_dir，"
                "所有样本将使用零图像/单位内参，融合训练无意义，请检查数据路径！")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        item = {
            "point_cloud": torch.from_numpy(data["point_cloud"]).float(),
            "bboxes": torch.from_numpy(data["bboxes"]).float(),
            "class_ids": torch.from_numpy(data["class_ids"]).long(),
            "vote_label": torch.from_numpy(data["vote_label"]).float(),
            "vote_mask": torch.from_numpy(data["vote_mask"]).float(),
        }
        # 图像与内参（用于图像分支）；缺省时用零张量占位
        rgb = torch.zeros((3, 640, 640), dtype=torch.float32)
        K = torch.eye(3, dtype=torch.float32)
        letterbox = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        if self.frames_dir is not None:
            frame_npz = self.frames_dir / f"{path.stem}.npz"
            if frame_npz.is_file():
                fr = np.load(frame_npz)
                if "rgb" in fr:
                    rgb, lb = letterbox_rgb(fr["rgb"], 640)
                    letterbox = torch.tensor(lb, dtype=torch.float32)
                if "K" in fr:
                    K = torch.from_numpy(fr["K"]).float()
            elif self.use_image_branch and self._missing_warned < 3:
                logger.warning("帧文件缺失（图像分支将用零张量占位）: %s", frame_npz)
                self._missing_warned += 1
        item["rgb"] = rgb
        item["K"] = K
        item["letterbox"] = letterbox   # (scale, pad_x, pad_y)，供投影对齐
        return item


def letterbox_rgb(img, size=640):
    """等比缩放 + 灰边填充到 size×size。

    返回:
        tensor: (3, size, size) 归一化 RGB 张量；
        letterbox: (scale, pad_x, pad_y) 三元组，投影时用于把
                   “原图像素坐标”对齐到 letterbox 后的网络输入坐标。
    与 yolov8_feature.letterbox_image 保持同一套几何参数。
    """
    from yolov8_feature import letterbox_image
    import torch

    canvas, ratio, pad_x, pad_y = letterbox_image(
        np.ascontiguousarray(img), target=size)
    rgb = canvas[..., ::-1].astype(np.float32) / 255.0   # BGR -> RGB
    return torch.from_numpy(rgb).permute(2, 0, 1), (float(ratio),
                                                    float(pad_x),
                                                    float(pad_y))


# ---------------------------------------------------------------
# 融合模型装配
# ---------------------------------------------------------------
class FusionModel:
    """点云分支 + 图像分支 + 投影 + 融合头 + 检测头（非 nn.Module 装配）。"""

    def __init__(self, cfg, votenet_ckpt, yolov8_ckpt, freeze_backbone=True):
        from yolov8_feature import YOLOv8FeatureExtractor
        from fusion_head import FusionHead, DetectionHead
        from projection import ProjectionModule

        det_cfg = cfg["detection"]
        fc = det_cfg["fusion"]

        # 消融开关（configs/ablation/*.yaml 覆盖）：见 default.yaml 的 ablation 区块
        self.ablation = cfg.get("ablation", {})
        self.img_dim = det_cfg["projected_channels"]

        self.votebackbone = VoteNetBackbone(
            ckpt_path=votenet_ckpt,
            input_feature_dim=cfg["detection"].get("input_feature_dim", 0),
            freeze=freeze_backbone)
        self.yolo_feat = YOLOv8FeatureExtractor(
            ckpt=yolov8_ckpt,
            proj_channels=det_cfg["projected_channels"],
            freeze=freeze_backbone)
        self.projection = ProjectionModule(image_size=det_cfg["image_size"])
        self.fusion_head = FusionHead(
            fusion_mode=fc["method"],
            pc_dim=det_cfg["votenet_feature_dim"],
            img_dim=det_cfg["projected_channels"],
            hidden_dim=fc["head_hidden_dim"],
            num_heads=fc["attn_num_heads"],
            dropout=fc["head_dropout"])
        self.det_head = DetectionHead(
            in_dim=fc["head_hidden_dim"],
            num_classes=det_cfg["detect_head"]["num_classes"],
            vote_num=det_cfg["detect_head"]["vote_num"])

    def to(self, device):
        # 注意：YOLOv8FeatureExtractor 内含 backbone（DetectionModel）
        # 与 proj_conv 两部分，二者都必须迁移，否则前向时
        # “主干在 CPU、输入/proj_conv 在 GPU”会报 device mismatch。
        for m in (self.votebackbone,
                  self.yolo_feat.backbone, self.yolo_feat.proj_conv,
                  self.fusion_head, self.det_head):
            m.to(device)
        self.device = device
        return self

    def train(self):
        self.fusion_head.train()
        self.det_head.train()
        self.yolo_feat.proj_conv.train()

    def eval(self):
        self.fusion_head.eval()
        self.det_head.eval()
        self.yolo_feat.proj_conv.eval()

    def trainable_parameters(self):
        params = list(self.fusion_head.parameters())
        params += list(self.det_head.parameters())
        params += self.yolo_feat.trainable_params()
        return params

    def forward(self, batch, device="cuda"):
        # 消融开关：use_point_branch / use_image_branch（configs/ablation/）
        use_point = self.ablation.get("use_point_branch", True)
        use_image = self.ablation.get("use_image_branch", True)

        xyz = batch["point_cloud"].to(device)
        B, N, _ = xyz.shape

        if use_point:
            # 点云分支：VoteNet 主干提取种子点特征
            seed_xyz, seed_feat = self.votebackbone(xyz)   # (B,N,3),(B,N,128)
        else:
            # 单 RGB（伪 3D）：输入点云即种子，特征由图像分支提供
            seed_xyz, seed_feat = xyz, None

        if use_image:
            rgb = batch["rgb"].to(device)
            K = batch["K"].to(device)
            img_feat = self.yolo_feat.forward(rgb)         # (B,128,80,80)
            # letterbox 参数 (B,3)：把原图像素坐标对齐到网络输入；
            # 数据集中正方形输入时为 (1,0,0)，非正方形（如 640x480）必须传
            letterbox = batch.get("letterbox")
            if letterbox is not None:
                letterbox = letterbox.to(device)
            proj_feat, _ = self.projection.forward(img_feat, seed_xyz, K,
                                                   letterbox=letterbox)  # (B,N,128)
        else:
            # 单点云分支：图像特征置零（融合层等效单点云投影）
            proj_feat = torch.zeros(B, N, self.img_dim, device=device)

        if seed_feat is None:
            # 伪 3D：图像特征即点特征
            seed_feat = proj_feat

        fused = self.fusion_head(seed_feat, proj_feat)     # (B,N,256)
        return self.det_head(fused)

    def __call__(self, batch, device="cuda"):
        return self.forward(batch, device=device)


# ---------------------------------------------------------------
# 损失函数
# ---------------------------------------------------------------
def compute_loss(pred, batch, vote_num=10, num_classes=10):
    """融合检测头损失（当前：objectness + 投票中心；其余为占位）。

    说明：VoteNet 官方对尺寸/朝向/类别在 proposal 聚合后监督
    （按聚合结果指派 GT 框）。本骨架先实现 seed 级可监督的两项
    （目标性 + 投票中心），其余分支以占位损失 0 保持训练管线可跑通。
    TODO：按 external/votenet 的 compute_loss 思路补全
    “种子点/提议 → GT 框”指派后的 size/heading/class 监督。
    """
    device = pred["objectness"].device

    gt_obj = batch["vote_mask"].to(device)                    # (B,N)
    gt_vote = batch["vote_label"].to(device)                  # (B,N,3)

    # 1) 目标性
    loss_obj = F.binary_cross_entropy_with_logits(
        pred["objectness"].squeeze(-1), gt_obj)

    # 2) 投票中心（简化为每个种子点的第 1 个投票，按 vote_mask 加权）
    center_best = pred["center"][:, :, 0, :]                  # (B,N,3)
    loss_center = (F.smooth_l1_loss(center_best, gt_vote, reduction="none")
                   * gt_obj.unsqueeze(-1)).mean()

    # 3) 其余分支占位（TODO 接入真值指派后启用）
    loss_extra = torch.tensor(0.0, device=device)

    return {"total": loss_obj + loss_center + loss_extra,
            "obj": loss_obj, "center": loss_center, "extra": loss_extra}


# ---------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------
def train_fusion(cfg, det_data_dir, frames_dir, out_dir,
                 votenet_ckpt, yolov8_ckpt, device="cuda"):
    det_cfg = cfg["detection"]
    train_cfg = det_cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(det_data_dir) / "split.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    torch.manual_seed(cfg.get("seed", 2024))
    np.random.seed(cfg.get("seed", 2024))

    use_image_branch = cfg.get("ablation", {}).get("use_image_branch", True)
    train_ds = SunRGBDDataset(det_data_dir, frames_dir,
                              sample_names=[s["name"] for s in split["train"]],
                              use_image_branch=use_image_branch)
    val_ds = SunRGBDDataset(det_data_dir, frames_dir,
                            sample_names=[s["name"] for s in split["val"]],
                            use_image_branch=use_image_branch)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=train_cfg["num_workers"])
    logger.info("train %d / val %d", len(train_ds), len(val_ds))

    model = FusionModel(cfg, votenet_ckpt, yolov8_ckpt,
                        freeze_backbone=train_cfg["freeze_backbone"]).to(device)

    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5)

    # TensorBoard 日志（可选，tensorboardX）
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        writer = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        losses = {"total": 0.0}
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(batch, device=device)
            loss_dict = compute_loss(pred, batch, det_cfg["detect_head"]["vote_num"],
                                     det_cfg["detect_head"]["num_classes"])
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
            optimizer.step()
            for k, v in loss_dict.items():
                losses[k] = losses.get(k, 0.0) + float(v) * len(batch["point_cloud"])
        n = len(train_ds)
        losses = {k: v / max(n, 1) for k, v in losses.items()}

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %3d/%d | %.1fs | loss=%.4f (obj=%.4f ctr=%.4f) | lr=%.6f",
                    epoch, train_cfg["epochs"], time.time() - t0,
                    losses["total"], losses["obj"], losses["center"], lr)
        if writer is not None:
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v, epoch)

        # 周期性保存权重
        if epoch % train_cfg["checkpoint_interval"] == 0 or epoch == train_cfg["epochs"]:
            ckpt = {
                "epoch": epoch,
                "cfg": cfg,
                "model": _collect_state(model),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, out_dir / f"fusion_epoch{epoch:03d}.pth")
            logger.info("已保存权重: %s", out_dir / f"fusion_epoch{epoch:03d}.pth")

    if writer is not None:
        writer.close()
    logger.info("训练完成，权重目录: %s", out_dir)


def _collect_state(model):
    """收集模型全部参数（含 YOLO 主干与 proj_conv）。"""
    state = {
        "votebackbone": model.votebackbone.state_dict(),
        "yolo_proj_conv": model.yolo_feat.proj_conv.state_dict(),
        "fusion_head": model.fusion_head.state_dict(),
        "det_head": model.det_head.state_dict(),
    }
    return state


def main():
    parser = argparse.ArgumentParser(description="融合模型训练（冻结双主干）")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--det_data_dir", default="results/preprocess/detection",
                        help="检测数据目录")
    parser.add_argument("--frames_dir", default="results/preprocess/frames",
                        help="帧 rgb/K npz 目录（parse_sunrgbd.py 输出）")
    parser.add_argument("--out_dir", default="results/detection/fusion",
                        help="权重与日志输出目录")
    parser.add_argument("--votenet_ckpt", default="weights/votenet_sunrgbd.pth")
    parser.add_argument("--yolov8_ckpt", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置的轮数")
    parser.add_argument("--fusion_method", choices=["concat", "attention"],
                        default=None, help="覆盖配置的融合方式")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.epochs:
        cfg["detection"]["train"]["epochs"] = args.epochs
    if args.fusion_method:
        cfg["detection"]["fusion"]["method"] = args.fusion_method

    train_fusion(cfg, args.det_data_dir, args.frames_dir, args.out_dir,
                 args.votenet_ckpt, args.yolov8_ckpt, device=args.device)


if __name__ == "__main__":
    main()
