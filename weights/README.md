# weights/ —— 模型权重目录

本目录存放模型权重。**内容不入库**（见 `.gitignore`），仅保留本 README。

## 约定

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `votenet_sunrgbd.pth` | VoteNet 官方预训练权重（README 提供下载链接） | 点云分支初始化 / 基线评测 |
| `yolov8n.pt` | ultralytics 自动下载（放在工作目录也可） | 图像分支初始化 |
| `fusion_epoch060.pth` | 训练产出（train_fusion.py） | 主融合模型 |
| `lightweight_epoch020.pth` | 微调产出（finetune_lightweight.py） | 轻量化模型 |

## 说明

- 权重文件较大（数百 MB），不入库；如需备份请自行处理；
- 下载 VoteNet 权重前先确认许可证与来源；
- 训练/微调权重会自动保存到 `results/ablation/<实验ID>/` 而非本目录（见 `configs/default.yaml`）。
