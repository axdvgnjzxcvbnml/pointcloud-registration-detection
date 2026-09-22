# configs/ 配置说明

本目录存放所有实验配置，采用 YAML 格式，由各模块脚本通过 `--config` 参数加载
（加载器：`detection/train_fusion.py` 的 `load_config`，支持 `base` 继承合并）。

## 文件约定

| 文件 | 用途 | 输出批次 |
| --- | --- | --- |
| `default.yaml` | 全局默认配置模板：数据路径 / 预处理 / 配准 / 检测训练 / 融合开关 / 轻量化开关 / 消融开关 | 第一批 |
| `ablation/*.yaml` | 消融实验配置（5 组），以 `base: ../default.yaml` 继承并覆盖差异项 | 第五批 |

## 使用方式

```bash
python detection/train_fusion.py --config configs/default.yaml
python detection/evaluate_detection.py --config configs/ablation/03_fusion_attention.yaml \
    --ckpt results/ablation/03_fusion_attention/fusion_epoch060.pth
```

## 消融实验总览（configs/ablation/）

| 编号 | 实验 | 模型开关 | 融合方式 | 轻量化 | 对比目的 |
| --- | --- | --- | --- | --- | --- |
| 00 | 单点云分支（VoteNet 基线） | 点云✅ 图像❌ | concat（图支置零） | 否 | 双分支 vs 单点云 |
| 01 | 单 RGB（YOLOv8+伪 3D） | 点云❌ 图像✅ | concat（伪 3D） | 否 | 双分支 vs 单 RGB |
| 02 | 融合-Concat（主融合） | 点云✅ 图像✅ | concat | 否 | 主方案 |
| 03 | 融合-Attention（消融） | 点云✅ 图像✅ | attention | 否 | Concat vs Attention |
| 04 | 融合+轻量化 | 点云✅ 图像✅ | concat | DSConv1d, 128 通道 | 精度 vs 效率 |

> 训练脚本读取 `ablation.use_point_branch` / `use_image_branch`（`FusionModel.forward`）
> 与 `detection.fusion.method` / `lightweight.enabled`；实验 05 需先跑 02 得到
> 教师权重，再 `finetune_lightweight.py --resume ...` 微调。

## 修改规范

- 除特殊标注外，路径均相对项目根目录。
- 训练实验请复制 `default.yaml` 为 `ablation/` 下的新配置后再修改，不要直接改动 `default.yaml`。
- 融合方式通过 `detection.fusion.method` 切换：`concat`（主融合）/ `attention`（消融对比）。
- 轻量化通过 `lightweight.enabled` 开关，启用时通道数减半并使用 DSConv1d。
