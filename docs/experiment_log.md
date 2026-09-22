# 实验记录（Experiment Log）

> 用途：逐条记录每次实验的配置、结果与结论，保证**可复现、可追溯**。
> 约定：所有数值为**实际运行结果**，未跑前填 `TBD`；配置一律引用 `configs/` 下的文件 + 命令行覆盖项，不粘贴整份 YAML。

---

## 0. 复现信息模板（每条实验必填）

```text
实验编号   ：EXP-{序号}
日期       ：YYYY-MM-DD
执行人     ：
运行环境   ：V100 / torch 1.13.1+cu117 / open3d 0.17.0 / Python 3.8
Git 提交   ：（如使用版本管理，记录 commit hash）
配置文件   ：configs/...yaml（含 base 继承链）
命令行覆盖 ：--batch-size 4 --num-points 30000（无则写"无"）
数据范围   ：训练/验证帧对或场景划分
随机种子   ：（configs/default.yaml 的 seed）
```

---

## 1. 实验总览表

| 实验编号 | 日期 | 配置 | 阶段 | mAP@0.25 | mAP@0.5 | 参数量 | 推理速度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | TBD | ablation/00_votenet_baseline | 检测 | TBD | TBD | TBD | TBD | 单点云基线 |
| EXP-002 | TBD | ablation/01_rgb_pseudo3d | 检测 | TBD | TBD | TBD | TBD | 单 RGB 伪 3D |
| EXP-003 | TBD | ablation/02_fusion_concat | 检测 | TBD | TBD | TBD | TBD | 主融合 |
| EXP-004 | TBD | ablation/03_fusion_attention | 检测 | TBD | TBD | TBD | TBD | Attention 消融 |
| EXP-005 | TBD | ablation/04_fusion_lightweight | 检测 | TBD | TBD | TBD | TBD | 轻量化 |
| EXP-0xx | TBD | 配准评测 | 配准 | — | — | — | — | RMSE/旋转/平移/成功率见配准小节 |

**配准评测指标**（`registration/evaluate_registration.py` 输出）：RMSE、旋转角误差（°）、平移误差（m）、配准成功率。建议与检测实验分开建表：

| 实验编号 | 日期 | 帧对集合 | RMSE | 旋转误差° | 平移误差 m | 成功率 | 方法 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REG-001 | 2026-09-22 | 7 对（模拟场景 sim_scene_001，间隔 5/10） | **0.0077** | **0.010** | **0.0004** | **100%** | 本方案（FPFH+RANSAC+改进ICP） | 模拟数据 CPU 验证（Open3D 0.19）；粗配准阶段 RMSE=0.0147 / 旋转 0.408° / 平移 0.0184m。真实数据待 V100 回填 |
| REG-002 | 2026-09-22 | 同左 | 0.0077 | 0.006 | 0.0002 | 100% | 基线：点到点 ICP | 模拟数据；与改进 ICP 精度相当（此场景接近理想对齐） |
| REG-003 | 2026-09-22 | 同左 | 0.0136 | 0.639 | 0.0206 | 100% | 基线：FGR | 模拟数据；无精配准阶段，误差略高于本方案 |

---

## 2. 单条实验详细记录模板

```markdown
## EXP-{序号}：{一句话标题}

### 2.1 目标
{要验证的假设/对比关系，如："Concat 融合相对单点云基线的 mAP 增益"}

### 2.2 配置
- 配置文件：configs/ablation/xx.yaml（base: ../default.yaml）
- 模型开关：use_point_branch=?, use_image_branch=?, pseudo3d=?, method=?
- 训练：epochs=?, batch_size=?, lr=?, freeze_backbone=?
- 数据：points_per_frame=?, 训练帧数=?, 验证帧数=?

### 2.3 复现命令
```bash
# 训练
python detection/train_fusion.py --config configs/ablation/xx.yaml
# 评测
python detection/evaluate_detection.py --config configs/ablation/xx.yaml \
    --ckpt results/ablation/xx/fusion_epoch060.pth
```

### 2.4 结果
| 指标 | 数值 |
| --- | --- |
| mAP@0.25 | TBD |
| mAP@0.5 | TBD |
| 10 类 AP@0.25 | bathtub=TBD, bed=TBD, ...（逐类贴表格或链接） |
| 参数量 | TBD（evaluate_detection 输出） |
| FLOPs | TBD |
| 推理速度 | TBD ms/帧 |

### 2.5 结论与备注
{结论一句话 + 与对比实验的差值 + 异常现象、复现注意事项}
```

---

## 3. 记录规范（防呆）

1. **数值只填实跑结果**：未跑 = `TBD`；估算值必须标"（估算）"并给口径。
2. **配置只写引用不写拷贝**：改过配置就把 `git diff` 或改动的行贴进备注。
3. **一次实验一条记录**：同一配置换 seed/数据范围算新实验（EXP-xxx-a/b）。
4. **失败实验也记录**：报错信息摘要 + 解决过程（引用 `docs/troubleshooting.md` 对应小节），避免重复踩坑。
5. **产物留档**：训练日志（tensorboardX events / 终端重定向）、checkpoint 路径、评测 JSON、可视化图（`results/` 下按实验分目录）。
6. **结论可追溯**：每个数字给出它来自哪个命令/哪个输出文件。

---

## 4. 实验进度清单（勾选）

- [ ] EXP-001 单点云基线（VoteNet）训练 + 评测
- [ ] EXP-002 单 RGB 伪 3D 训练 + 评测
- [ ] EXP-003 融合-Concat（主融合）训练 + 评测
- [ ] EXP-004 融合-Attention 训练 + 评测
- [ ] EXP-005 融合+轻量化（教师蒸馏微调）训练 + 评测
- [ ] REG-001/002/003 配准方案与两个基线对比
- [ ] 全量消融汇总表 + 结论章节（写入 README 结果占位表）
