# weights/ —— 模型权重

存放预训练权重与训练产出 checkpoint，**不提交到本仓库**（已在 `.gitignore` 忽略，本 README 除外）。

## 预期文件

| 文件 | 来源 |
| --- | --- |
| `yolov8n.pt` | YOLOv8n COCO 预训练权重（ultralytics 首次运行自动下载，或手动放入） |
| `votenet_sunrgbd.pth` | VoteNet 官方 SUN RGB-D 预训练权重（按 VoteNet 官方 README 链接下载） |
| `fusion_epoch060.pth` 等 | `detection/train_fusion.py` / `finetune_lightweight.py` 的训练产出（默认写到 `results/`，可按配置改到此处） |

> 权重文件较大，请勿拷贝入库；实验记录中只写权重路径与来源。
