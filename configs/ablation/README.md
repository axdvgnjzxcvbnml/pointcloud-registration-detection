# 消融实验配置说明

本项目共 5 组消融实验，全部继承 `configs/default.yaml`（`base:` 字段），仅覆盖差异字段。
统一评测口径：3D IoU，mAP@0.25 / mAP@0.5，10 类（bathtub, bed, bookshelf, chair, desk,
dresser, night_stand, sofa, table, toilet）及均值，并记录参数量 / FLOPs / 推理速度。

## 五组配置一览

| 配置 | 模式标识 | 点云分支 | 图像分支 | 融合方式 | 轻量化 | 训练入口 |
| --- | --- | --- | --- | --- | --- | --- |
| `00_votenet_baseline.yaml` | `votenet_baseline` | ✅ | ❌ | 无（纯 VoteNet） | ❌ | 评测预训练权重（不训练） |
| `01_rgb_pseudo3d.yaml` | `rgb_pseudo3d` | ❌ | ✅ | 伪 3D（图像特征即点特征） | ❌ | `train_fusion.py` |
| `02_fusion_concat.yaml` | `fusion_concat` | ✅ | ✅ | Concat + 1×1 Conv（主融合） | ❌ | `train_fusion.py` |
| `03_fusion_attention.yaml` | `fusion_attention` | ✅ | ✅ | Cross-Attention | ❌ | `train_fusion.py` |
| `04_fusion_lightweight.yaml` | `fusion_lightweight` | ✅ | ✅ | Concat | ✅ DSConv1d 256→128 | `finetune_lightweight.py`（依赖 02 权重） |

## 对比关系（回答什么问题）

```
00 VoteNet 基线 ──────┬──────────────────────────► 全项目参照系
                      │
01 单 RGB（伪 3D） ────┼──► 图像语义分支单独能力（应有明显落差）
                      │
02 融合-Concat（主） ──┼──► 双分支互补，主结果（应 ≥ 00 与 01）
                      │
03 融合-Attention ────┼──► 跨模态注意力的边际收益（与 02 对比，±1~2 点内为相当）
                      │
04 融合+轻量化 ────────┘──► 精度-效率权衡（与 02 对比：mAP 略降、参数/FLOPs 减半）
```

- **02 vs 00 vs 01**：验证“图像 + 点云融合优于任一单分支”。
- **03 vs 02**：验证融合方式选择（简单拼接是否足够）。
- **04 vs 02**：验证轻量化头在精度损失可控下带来的效率收益。

## 运行

单组（以主融合为例）：

```bash
python detection/train_fusion.py --config configs/ablation/02_fusion_concat.yaml
python detection/evaluate_detection.py \
    --config configs/ablation/02_fusion_concat.yaml \
    --ckpt results/ablation/02_fusion_concat/fusion_epoch060.pth
```

全量（按 00→04 顺序，含汇总表）：

```bash
bash scripts/v100_step7_ablation.sh
```

汇总表输出到 `results/ablation/ablation_summary.csv`。

## 预期结论（占位，跑完替换）

| 组 | mAP@0.25 | mAP@0.5 | 参数量 | FLOPs | 备注 |
| --- | --- | --- | --- | --- | --- |
| 00 基线 | TBD | TBD | TBD | TBD | VoteNet 公开水平参照 |
| 01 伪 3D | TBD | TBD | TBD | TBD | 应最低（缺几何） |
| 02 Concat | TBD | TBD | TBD | TBD | 主结果 |
| 03 Attention | TBD | TBD | TBD | TBD | 与 02 相当 |
| 04 轻量化 | TBD | TBD | TBD | TBD | 参数/FLOPs 减半，mAP 微降 |
