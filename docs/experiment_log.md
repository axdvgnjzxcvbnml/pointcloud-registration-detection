# docs/experiment_log.md —— 实验记录模板（第七批交付）

> 本文件为**模板**：所有 `TBD` 均为占位，需在目标服务器（单卡 V100）
> 运行实验后回填；回填规则见文末。

## 1. 记录规范

- 每次实验独立一段，编号 `EXP-YYYYMMDD-xx`；
- 数值必须来自实际运行输出，禁止估算或转载；
- 环境、配置、命令、日志路径一并记录，保证可复现；
- 结果回填后同步更新 `README.md` 的“实验结果”表。

## 2. 实验配置速查

| 编号 | 实验 | 配置 | 运行命令 |
| --- | --- | --- | --- |
| EXP-01 | 数据预处理 | `configs/default.yaml` | `python preprocess/parse_sunrgbd.py --data_root data/SUNRGBD` |
| EXP-02 | 点云拼接（粗+精） | `configs/default.yaml` | `python registration/coarse_registration.py --src a.ply --dst b.ply` |
| EXP-03 | 配准评测 | `configs/default.yaml` | `python registration/evaluate_registration.py --pairs results/pairs.json` |
| EXP-04 | VoteNet 基线 | `configs/ablation/00_votenet_baseline.yaml` | `python detection/train_fusion.py --config configs/ablation/00_votenet_baseline.yaml` |
| EXP-05 | 融合-Concat（主融合） | `configs/ablation/02_fusion_concat.yaml` | `python detection/train_fusion.py --config configs/ablation/02_fusion_concat.yaml` |
| EXP-06 | 融合-Attention（消融） | `configs/ablation/03_fusion_attention.yaml` | `python detection/train_fusion.py --config configs/ablation/03_fusion_attention.yaml` |
| EXP-07 | 融合+轻量化 | `configs/ablation/04_fusion_lightweight.yaml` | `python detection/finetune_lightweight.py --config configs/ablation/04_fusion_lightweight.yaml` |
| EXP-08 | 检测评测 | `configs/ablation/02_fusion_concat.yaml` | `python detection/evaluate_detection.py --config configs/ablation/02_fusion_concat.yaml` |

## 3. 结果记录模板

### EXP-YYYYMMDD-xx：<实验名称>

**配置**：`configs/<文件>.yaml`（或命令行参数）  
**环境**：<GPU / CUDA / Python / 依赖版本>  
**数据**：<数据子集 / 帧数 / 采样参数>  
**命令**：

```bash
<完整命令>
```

**结果**：

| 指标 | 数值 | 备注 |
| --- | --- | --- |
| 训练时间 | TBD | |
| 收敛损失 | TBD | |
| mAP@0.25 | TBD | 10 类均值 |
| mAP@0.5 | TBD | 10 类均值 |
| 参数量 | TBD | M |
| FLOPs | TBD | G |
| 推理速度 | TBD | ms/帧 |

**对比/结论**：<与基线或前次实验对比，分析差异原因>  
**日志/产物**：`results/...`（权重、report.json、日志路径）

## 4. 回填规则

1. 数值来源仅限本机实际运行输出（日志 / report.json / 官方复现）；
2. 回填时在 README“实验结果”表中同步更新，并注明运行日期与机器；
3. 参数不一致（如 batch 下调、epoch 提前中断）必须在备注列说明；
4. 同一实验复跑后以最新一次为准，旧记录保留在文档历史中并标注“已更新”。
