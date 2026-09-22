#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolov8_feature.py — YOLOv8n P3 层语义特征提取（第四批交付）

功能
----
    加载 YOLOv8n COCO 预训练权重，冻结主干与 Neck，
    输入 resize 到 640×640，从 P3 层（stride=8）提取
    80×80×64 语义特征图，经 1×1 Conv 升维到 128 通道。

说明
----
    - P3 层索引：YOLOv8n 的 model.model 为 nn.Sequential，
      P3 输出位于索引 15（C2f，80×80，64 通道）。
      若版本不同，请用 --p3_layer_index 调整（可用 evaluate_detection
      的逐层形状打印确认）。
    - letterbox 的 scale/pad 会随返回值给出，供 projection.py
      做 3D→2D 对齐时使用。

用法
----
    python detection/yolov8_feature.py \
        --ckpt yolov8n.pt --image assets/demo.jpg --out results/detection/feat.npz
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def get_letterbox_params(h, w, target=640):
    """计算 letterbox 缩放与填充参数（纯 numpy，可单测）。

    Returns:
        (scale, pad_x, pad_y, new_h, new_w)
        原始坐标 (u, v) -> 目标图坐标: u' = u*scale + pad_x, v' = v*scale + pad_y
    """
    scale = min(target / h, target / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_x = (target - new_w) / 2.0
    pad_y = (target - new_h) / 2.0
    return scale, pad_x, pad_y, new_h, new_w


def letterbox_image(img_bgr, target=640):
    """对 BGR 图做 letterbox（等比缩放 + 灰边填充），返回 (canvas, scale, pad_x, pad_y)。

    img_bgr: (H, W, 3) uint8 BGR。
    """
    import cv2

    h, w = img_bgr.shape[:2]
    scale, pad_x, pad_y, new_h, new_w = get_letterbox_params(h, w, target)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, pad_x, pad_y


def preprocess_image(img_bgr, target=640):
    """预处理为模型输入张量。

    Returns:
        tensor: (1, 3, 640, 640) float32，RGB、[0,1] 归一化
        (scale, pad_x, pad_y): letterbox 参数（供投影对齐）
    """
    import torch

    canvas, scale, pad_x, pad_y = letterbox_image(img_bgr, target)
    rgb = canvas[..., ::-1].astype(np.float32) / 255.0   # BGR -> RGB
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor, (scale, pad_x, pad_y)


class YOLOv8FeatureExtractor:
    """YOLOv8n P3 特征提取器（冻结主干+Neck，只训练 1×1 升维卷积）。"""

    def __init__(self, ckpt="yolov8n.pt", p3_layer_index=15,
                 proj_channels=128, freeze=True, image_size=640):
        import torch.nn as nn
        from ultralytics import YOLO

        self.image_size = image_size
        self.p3_layer_index = p3_layer_index

        # ultralytics 的 YOLO(ckpt).model 即 nn.Sequential 网络体
        self.backbone = YOLO(ckpt).model
        if not isinstance(self.backbone, nn.Sequential):
            raise TypeError("YOLO(ckpt).model 不是 nn.Sequential，请检查 ultralytics 版本")

        # P3 层通道数（YOLOv8n 为 64；若自定义结构请调整）
        self.p3_channels = 64
        self.proj_conv = nn.Conv2d(self.p3_channels, proj_channels, 1)

        if freeze:
            self.freeze_backbone()

        self.eval_mode = False
        self._verify_layer()

    def _verify_layer(self):
        """校验 P3 层索引是否越界，并打印层信息。"""
        n = len(self.backbone)
        if not (0 <= self.p3_layer_index < n):
            raise IndexError(f"p3_layer_index={self.p3_layer_index} 越界（共 {n} 层）")
        logger.info("P3 层: model.model[%d] = %s",
                    self.p3_layer_index, type(self.backbone[self.p3_layer_index]).__name__)

    def freeze_backbone(self):
        """冻结主干与 Neck（含 P3 之前全部参数）。"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def trainable_params(self):
        """返回可训练参数（仅 proj_conv）。"""
        return list(self.proj_conv.parameters())

    def forward(self, img):
        """前向提取 P3 特征。

        Args:
            img: (B, 3, 640, 640) float32，RGB、[0,1]（preprocess_image 输出）。
        Returns:
            (B, proj_channels, 80, 80) 升维后的 P3 特征图。
        """
        import torch

        if not torch.is_tensor(img):
            raise TypeError("img 应为 torch.Tensor，请先调用 preprocess_image")

        self.backbone.eval()
        x = img
        # 只跑到 P3 层（跳过 Detect 头，省显存、避免干扰）
        for layer in self.backbone[:self.p3_layer_index + 1]:
            with torch.no_grad():
                x = layer(x)
        return self.proj_conv(x)

    def __call__(self, img):
        return self.forward(img)


def extract_from_image(ckpt, image_path, out_path, p3_layer_index=15,
                       proj_channels=128, target=640):
    """单图特征提取入口（调试用）。"""
    import cv2

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读取图片失败: {image_path}")
    tensor, letterbox = preprocess_image(img_bgr, target=target)

    extractor = YOLOv8FeatureExtractor(ckpt=ckpt, p3_layer_index=p3_layer_index,
                                       proj_channels=proj_channels, freeze=True)
    feat = extractor.forward(tensor)   # (1, 128, 80, 80)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), feature=feat.detach().cpu().numpy(),
                        letterbox=letterbox)
    logger.info("P3 特征 -> %s，形状 %s", out_path, tuple(feat.shape))


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n P3 特征提取")
    parser.add_argument("--ckpt", default="yolov8n.pt", help="YOLOv8n 权重")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--out", default="results/detection/p3_feat.npz",
                        help="特征输出路径")
    parser.add_argument("--p3_layer_index", type=int, default=15,
                        help="P3 层在 model.model 中的索引")
    parser.add_argument("--proj_channels", type=int, default=128,
                        help="1×1 升维通道数")
    parser.add_argument("--target", type=int, default=640,
                        help="输入尺寸（640×640）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    extract_from_image(args.ckpt, args.image, args.out,
                       p3_layer_index=args.p3_layer_index,
                       proj_channels=args.proj_channels, target=args.target)


if __name__ == "__main__":
    main()
