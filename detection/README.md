# detection/ —— 三维目标检测模块

VoteNet 点云分支与 YOLOv8n 图像分支的特征级融合检测，支持消融与轻量化。**需要 GPU（V100）。**

## 文件

| 脚本 | 用途 |
| --- | --- |
| `votenet_baseline.py` | 加载 VoteNet 官方预训练权重推理，输出 mAP@0.25 / mAP@0.5 |
| `yolov8_feature.py` | YOLOv8n（冻结主干+Neck），640 输入，P3 层（stride=8）80×80×64 → 1×1 Conv 升 128 |
| `projection.py` | 3D 种子点按内参投影到 2D，双线性采样图像特征（numpy 版 + torch 版） |
| `fusion_head.py` | Concat+1×1 Conv（主融合）与 Cross-Attention（消融）融合头 + VoteNet 风格检测头（约 2M 参数） |
| `train_fusion.py` | 冻结双主干，只训融合头+检测头，60 epoch；含 `load_config`（支持 base 继承）与消融开关 |
| `lightweight_head.py` | DSConv1d 深度可分离卷积，通道 256→128，Dropout(0.3) |
| `finetune_lightweight.py` | 基于融合权重微调轻量化模型，20 epoch（lr/10） |
| `evaluate_detection.py` | mAP@0.25 / mAP@0.5（3D IoU，10 类+均值，VOC 101 点插值）、参数量、FLOPs、推理速度 |

## 消融配置

五组实验配置位于 `configs/ablation/`，通过 `--config` 指定，见 `configs/README.md`。

## 示例

```bash
python detection/train_fusion.py --config configs/ablation/02_fusion_concat.yaml
python detection/evaluate_detection.py --config configs/ablation/02_fusion_concat.yaml \
    --ckpt results/ablation/02_fusion_concat/fusion_epoch060.pth
```

> VoteNet 官方 API / checkpoint 键、YOLO P3 层索引等待按实际环境核对的 TODO 已在代码注释标注。
