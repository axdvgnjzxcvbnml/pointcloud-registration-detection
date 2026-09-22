#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fusion_device.py —— FusionModel 设备迁移一致性测试

验证修复：FusionModel.to(device) 必须把全部子模块（含 YOLOv8 主干
yolo_feat.backbone，而不仅是 proj_conv）迁移到目标设备，否则前向时
“主干在 CPU、输入/proj_conv 在 GPU”会报 device mismatch。

说明：
    - 沙箱/CI 无 external/votenet（PointNet2 需 CUDA 编译），故用
      StubBackbone 打桩 VoteNetBackbone；YOLOv8 部分用真实权重
      （weights/yolov8n.pt 或自动下载），以真实验证设备迁移。
    - CUDA 不可用时自动跳过 GPU 用例（沙箱纯 CPU 也能验证 to('cpu')
      与一次完整前向无 device mismatch）。

运行：
    python tests/test_fusion_device.py
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detection"))

from train_fusion import FusionModel, load_config  # noqa: E402


class StubBackbone(nn.Module):
    """VoteNet 主干打桩：xyz (B,N,3) -> (seed_xyz, seed_features (B,N,128))。"""

    def __init__(self, ckpt_path=None, input_feature_dim=0, freeze=True):
        super().__init__()
        self.proj = nn.Linear(3, 128)

    def forward(self, xyz):
        return xyz, self.proj(xyz)


def _build_model():
    """构造融合模型（VoteNet 主干打桩，YOLO 用真实权重）。"""
    import train_fusion
    # 关键：在构造 FusionModel 前打桩
    train_fusion.VoteNetBackbone = StubBackbone

    cfg = load_config(str(ROOT / "configs" / "ablation" / "02_fusion_concat.yaml"))
    yolo_ckpt = str(ROOT / "weights" / "yolov8n.pt")
    if not Path(yolo_ckpt).is_file():
        yolo_ckpt = "yolov8n.pt"   # 兜底：ultralytics 自动下载
    model = FusionModel(cfg,
                        votenet_ckpt=None,
                        yolov8_ckpt=yolo_ckpt,
                        freeze_backbone=True)
    return model


def _module_devices(model):
    """收集 FusionModel 全部含参数子模块的设备集合。

    FusionModel 是普通组合类（非 nn.Module），需显式枚举各子模块，
    这也正好精确覆盖 to() 中应迁移的全部对象（含 YOLO 主干）。
    """
    modules = [
        model.votebackbone,
        model.yolo_feat.backbone,
        model.yolo_feat.proj_conv,
        model.fusion_head,
        model.det_head,
    ]
    return {p.device.type for m in modules for p in m.parameters()}


class TestFusionModelDevice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()

    def test_01_to_cpu_all_modules_consistent(self):
        """to('cpu') 后所有子模块（含 YOLO 主干）参数都在 cpu。"""
        self.model.to("cpu")
        devices = _module_devices(self.model)
        self.assertEqual(devices, {"cpu"}, f"设备不一致: {devices}")

    def test_02_forward_cpu_no_device_mismatch(self):
        """CPU 上完整前向一次，验证无 device mismatch（覆盖 to() 修复）。"""
        self.model.to("cpu")
        self.model.eval()
        B, N = 1, 64
        batch = {
            "point_cloud": torch.randn(B, N, 3),
            "rgb": torch.rand(B, 3, 640, 640),
            "K": torch.eye(3).unsqueeze(0).repeat(B, 1, 1),
            "letterbox": torch.tensor([[1.0, 0.0, 0.0]]).repeat(B, 1),
            "bboxes": torch.zeros(B, 1, 7),
            "class_ids": torch.zeros(B, 1, dtype=torch.long),
            "vote_label": torch.zeros(B, N, 3),
            "vote_mask": torch.zeros(B, N),
        }
        with torch.no_grad():
            out = self.model(batch, device="cpu")
        self.assertEqual(out["objectness"].shape, (B, N, 1))
        self.assertEqual(out["class_scores"].shape, (B, N, 10))

    @unittest.skipUnless(torch.cuda.is_available(), "无 CUDA，跳过 GPU 设备一致性用例")
    def test_03_to_cuda_all_modules_consistent(self):
        """to('cuda') 后所有子模块（含 YOLO 主干）参数都在 cuda。"""
        self.model.to("cuda")
        devices = _module_devices(self.model)
        self.assertEqual(devices, {"cuda"}, f"设备不一致: {devices}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
