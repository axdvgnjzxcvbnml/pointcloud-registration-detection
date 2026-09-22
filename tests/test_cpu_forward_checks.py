#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only 模型自检（任务六）。

在纯 CPU 环境验证 detection/ 各模块可前向、可训练：
  1) YOLOv8n P3 层特征：输入 640x640 -> 升维后 (B,128,80,80)
  2) FusionModel 前向（Concat / Attention 两种融合）：输出键与 shape
  3) 一次反向 + 优化器步进（验证梯度回路）
  4) train_fusion() 真实训练循环：模拟 mini 数据集跑 1 epoch
  5) finetune_lightweight() 真实微调循环：基于 4) 权重跑 1 epoch

VoteNet 主干依赖 PointNet2 CUDA 算子（external/votenet/pointnet2），
CPU 上无法实例化 —— 自检用 MockVoteNetBackbone（MLP）替换，仅验证
"融合头 + 投影 + 检测头 + 训练循环" 的装配与形状正确性；
真实 VoteNet 主干在 V100 上由 v100_step1 编译后生效。
"""
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detection"))

import train_fusion as tf_mod                     # noqa: E402
from train_fusion import FusionModel, compute_loss, load_config  # noqa: E402
from yolov8_feature import YOLOv8FeatureExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
PASS = []


def check(name, cond):
    assert cond, "FAIL: " + name
    PASS.append(name)


class MockVoteNetBackbone(torch.nn.Module):
    """CPU 占位主干：xyz -> (seed_xyz, seed_feat)，形状与 VoteNet 一致。"""

    def __init__(self, ckpt_path=None, input_feature_dim=0, freeze=True):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.ReLU())

    def forward(self, xyz):
        return xyz, self.mlp(xyz)


def make_mini_dataset(root):
    """生成 4 帧模拟数据（det npz + frames npz + split.json）。"""
    det_dir = root / "det"
    fr_dir = root / "frames"
    det_dir.mkdir(parents=True, exist_ok=True)
    fr_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for i in range(4):
        name = "sim_%04d" % i
        names.append(name)
        pts = rng.uniform(-1, 1, (5000, 3)).astype(np.float32)
        np.savez_compressed(
            det_dir / (name + ".npz"),
            point_cloud=pts, point_clouds=pts,
            bboxes=np.zeros((0, 7), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            vote_label=np.zeros((5000, 3), dtype=np.float32),
            vote_mask=np.zeros((5000,), dtype=np.uint8),
        )
        rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        K = np.array([[570.34, 0, 320.0],
                      [0, 570.34, 240.0],
                      [0, 0, 1.0]], dtype=np.float32)
        np.savez_compressed(fr_dir / (name + ".npz"), rgb=rgb, K=K)
    split = {"train": [{"name": n} for n in names[:3]],
             "val": [{"name": n} for n in names[3:]]}
    (det_dir / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return det_dir, fr_dir


def step1_yolo_p3():
    print("== 1) YOLOv8n P3 特征（CPU） ==")
    ext = YOLOv8FeatureExtractor(ckpt=str(ROOT / "weights/yolov8n.pt"),
                                 proj_channels=128, freeze=True)
    rgb = torch.rand(1, 3, 640, 640)
    feat = ext.forward(rgb)
    check("P3 升维特征 shape == (1,128,80,80)", tuple(feat.shape) == (1, 128, 80, 80))
    print("  shape:", tuple(feat.shape), "OK")
    return ext


def step2_fusion_forward():
    print("== 2) FusionModel 前向（concat / attention） ==")
    torch.manual_seed(0)
    for method in ("concat", "attention"):
        cfg = load_config(str(ROOT / "configs/ablation/02_fusion_concat.yaml"))
        cfg["detection"]["fusion"]["method"] = method
        model = FusionModel(cfg, votenet_ckpt=None,
                            yolov8_ckpt=str(ROOT / "weights/yolov8n.pt"),
                            freeze_backbone=True).to("cpu")
        batch = {
            "point_cloud": torch.randn(2, 1024, 3),
            "rgb": torch.rand(2, 3, 640, 640),
            "K": torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
            "letterbox": torch.tensor([[1.0, 0.0, 0.0]] * 2),
            "vote_mask": torch.randint(0, 2, (2, 1024)).float(),
            "vote_label": torch.randn(2, 1024, 3),
        }
        pred = model(batch, device="cpu")
        exp = {"objectness": (2, 1024, 1), "center": (2, 1024, 10, 3),
               "size": (2, 1024, 10, 3), "heading": (2, 1024, 10, 2),
               "class_scores": (2, 1024, 10)}
        for k, shp in exp.items():
            check("%s %s shape %s" % (method, k, shp),
                  tuple(pred[k].shape) == shp)
        loss = compute_loss(pred, batch)
        loss["total"].backward()
        # 占位损失只监督 objectness+center（见 train_fusion.compute_loss TODO），
        # 故仅断言：融合头全部参数 + 检测头受监督分支有梯度。
        fh_grad = sum(p.grad is not None for p in model.fusion_head.parameters())
        dh_grad = sum(p.grad is not None for p in model.det_head.parameters())
        check("%s 融合头梯度全覆盖 (%d/%d)" % (method, fh_grad,
              len(list(model.fusion_head.parameters()))),
              fh_grad == len(list(model.fusion_head.parameters())))
        check("%s 检测头受监督分支有梯度 (%d)" % (method, dh_grad), dh_grad >= 8)
        print("  %s: 输出键/shape/反向 OK" % method)


def step3_train_one_epoch(base_dir):
    print("== 3) train_fusion() 1 epoch（模拟 mini 数据，CPU） ==")
    det_dir, fr_dir = make_mini_dataset(base_dir / "ds3")
    cfg = load_config(str(ROOT / "configs/ablation/02_fusion_concat.yaml"))
    cfg["detection"]["train"].update(epochs=1, batch_size=2,
                                     num_workers=0, checkpoint_interval=1)
    out_dir = base_dir / "out_fusion"
    tf_mod.train_fusion(cfg, str(det_dir), str(fr_dir), str(out_dir),
                        votenet_ckpt=None,
                        yolov8_ckpt=str(ROOT / "weights/yolov8n.pt"),
                        device="cpu")
    ckpt = out_dir / "fusion_epoch001.pth"
    check("训练循环产出 fusion_epoch001.pth", ckpt.is_file())
    print("  权重:", ckpt)
    return str(ckpt)


def step4_finetune_one_epoch(resume):
    print("== 4) finetune_lightweight() 1 epoch ==")
    import finetune_lightweight as flw
    base = Path(resume).parent.parent
    det_dir, fr_dir = make_mini_dataset(base / "ds4")
    cfg = load_config(str(ROOT / "configs/ablation/04_fusion_lightweight.yaml"))
    cfg["detection"]["train"].update(finetune_epochs=1, batch_size=2,
                                     num_workers=0, checkpoint_interval=1)
    out_dir = base / "out_lw"
    flw.finetune_lightweight(cfg, resume, str(det_dir), str(fr_dir),
                             str(out_dir), votenet_ckpt=None,
                             yolov8_ckpt=str(ROOT / "weights/yolov8n.pt"),
                             device="cpu")
    ckpt = out_dir / "lightweight_epoch001.pth"
    check("微调循环产出 lightweight_epoch001.pth", ckpt.is_file())
    print("  权重:", ckpt)


def main():
    # 全局替换 VoteNet 主干为 CPU mock（真实主干需 CUDA pointnet2，见文件头说明）
    tf_mod.VoteNetBackbone = MockVoteNetBackbone
    step1_yolo_p3()
    step2_fusion_forward()
    with tempfile.TemporaryDirectory() as tmp:
        resume = step3_train_one_epoch(Path(tmp))
        step4_finetune_one_epoch(resume)
    print("")
    print("全部 %d 项通过" % len(PASS))
    print("PASS")


if __name__ == "__main__":
    main()
