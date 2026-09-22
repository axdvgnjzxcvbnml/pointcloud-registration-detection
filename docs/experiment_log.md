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
| REG-001 | 2026-09-22 | 7 对（模拟场景 sim_scene_001，间隔 5/10） | **0.0077** | **0.010** | **0.0004** | **100%** | 本方案（FPFH+RANSAC+改进ICP） | 模拟数据 CPU 验证（Open3D 0.19）；粗配准阶段 RMSE=0.0147 / 旋转 0.408° / 平移 0.0184m。 |
| REG-002 | 2026-09-22 | 同左 | 0.0077 | 0.006 | 0.0002 | 100% | 基线：点到点 ICP | 模拟数据；与改进 ICP 精度相当（此场景接近理想对齐） |
| REG-003 | 2026-09-22 | 同左 | 0.0136 | 0.639 | 0.0206 | 100% | 基线：FGR | 模拟数据；无精配准阶段，误差略高于本方案 |
| REG-004 | 2026-09-22 | 60 对（真实 SUN3D mit_studyroom，间隔 5/10/30 各 20） | **0.0214**（成功子集） | **0.80**（成功子集） | **0.0334**（成功子集） | **27%**（16/60） | 本方案（FPFH+RANSAC+改进ICP） | 真实数据 CPU 验证；粗配准成功 13/60（22%）。详见下方「真实 vs 模拟对比」 |
| REG-005 | 2026-09-22 | 同左 | 0.0206（成功子集） | 0.61 | 0.0291 | 30%（18/60） | 基线：点到点 ICP | 成功子集精度与本方案相当；总体成功率略高（初值已由同一粗配准给出） |
| REG-006 | 2026-09-22 | 同左 | 0.0242（成功子集） | 1.07 | 0.0284 | 27%（16/60） | 基线：FGR | 无精配准阶段；对真实数据与改进 ICP 成功率相同 |
| REG-007 | 2026-09-22 | 同左 | 0.0244（成功子集） | 1.29 | 0.0289 | 22%（13/60） | 粗配准（无精配准） | 真实数据粗配准单独成功率；改进 ICP 在 3 对失败对上救回 |

### 1.1 真实 vs 模拟数据配准对比（2026-09-22，CPU 验证）

**结论先行**：真实数据（SUN3D mit_studyroom 49 帧 / 60 对）上，**成功子集精度与模拟数据相当**（改进 ICP：RMSE 0.0214m / 旋转 0.80° / 平移 3.3cm，模拟为 0.0077m / 0.010° / 0.4mm），但**总体成功率从模拟的 100% 大幅下降到 27%**，失败集中在粗配准阶段。

**按间隔分解（改进 ICP 成功率）**：

| 间隔 | 5 帧 | 10 帧 | 30 帧 |
| --- | --- | --- | --- |
| 成功率 | 55%（11/20） | 20%（4/20） | 5%（1/20） |

**原因分析**：

1. **帧间距（视点基线）是首要因素**：间隔 5 帧成功率 55%，间隔 30 帧仅 5%。视点变化越大，FPFH 特征匹配歧义越大（室内重复结构、纹理缺失墙面），RANSAC 100k 次难以命中正确对应。
2. **失败呈双峰**：44 个失败对中，约一半“接近但未达阈值”（旋转中位 2.3°、平移中位 0.12m，旋转 <5° 但平移 >0.05m 判失败），另一半是 RANSAC 错配（旋转最高 179°、平移最高 6.8m）→ ICP 从错误初值无法收敛。
3. **距离阈值偏紧**：粗配准 `ransac_distance_threshold=0.03m`（1.5×voxel）对真实传感器噪声偏严，模拟数据无噪声所以 100% 命中。
4. **判定阈值 5°/0.05m 严格**：部分 rot<5° 但 trans 略超 0.05m 的对被计入失败（可视为“次成功”）。

**建议（上 V100 / 正式实验前）**：

- 评测按间隔分列报告，或对 10/30 间隔单独评估（大基线对更接近真实拼接场景，也更有挑战）；
- 粗配准增强：迭代提到 200k+、`ransac_distance_threshold` 放宽到 2~3×voxel、加入法向一致性 checker、或 FGR 先粗对齐再 RANSAC 兜底；
- 多假设策略：RANSAC top-k 假设各跑一次 ICP，取 fitness 最优者（对“接近但未达阈值”的对收益最大）；
- 改进 ICP 自适应阈值起点可由粗配准 `inlier_rmse` 初始化（当前固定从 0.02 起步）。

### 1.2 配准超参扫描（2026-09-22，模拟数据 7 对 × 27 组合，纯 CPU）

**命令**：`python scripts/sweep_registration.py --pairs results/preprocess/pairs/pairs.json --pcd_dir results/preprocess/pcd --pose_gt_dir results/preprocess/pose_gt --out_dir results/registration/sweep`（网格：RANSAC {100k,300k,500k} × FPFH {0.15,0.25,0.35} × voxel {0.02,0.05,0.08}，共 27 组；成功判定：旋转 <5° 且平移 <0.05m）。

**结论先行**：模拟数据较简单，**27 组组合成功率均为 100%**；差异主要体现在精度与耗时。完整明细见 `results/registration/sweep/sweep_results.csv`（未入库）。

**按体素档位汇总**：

| 体素 | FPFH 半径 | RANSAC | 成功率 | 旋转误差°（均值区间） | 平移误差 m（均值区间） | 中位耗时 s（区间） |
| --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 0.15~0.35 | 100k~500k | 100% | 0.011 ~ 0.019 | 0.0004 ~ 0.0007 | 0.63 ~ 0.76 |
| 0.05 | 0.15~0.35 | 100k~500k | 100% | 0.015 ~ 0.059 | 0.0006 ~ 0.0026 | 0.38 ~ 0.60 |
| 0.08 | 0.15~0.35 | 100k~500k | 100% | 0.072 ~ 0.206 | 0.0039 ~ 0.0093 | 0.20 ~ 0.42 |

**关键观察**：
1. **精度优先 → voxel=0.02**：所有 0.02 组合旋转误差 ≤0.019°、平移 ≤0.0007m，显著优于 0.08（最高 0.206°/0.0093m，精度差约 4 倍）；
2. **速度优先 → voxel=0.08**：中位耗时 0.20~0.42s（约 0.02 档的 1/3），代价是精度下降；
3. **RANSAC 迭代在易数据上无差异**：100k↔500k 耗时几乎持平（置信度 0.999 提前终止），说明对高内点率场景 100k 足够；但真实数据（§1.1 REG-004）失败集中在低内点率/歧义帧对，仍建议 V100 上提到 200k+；
4. FPFH 半径三档差异小，0.25 为稳健中间值。

**写入 configs/default.yaml 的取值**（精度优先，兼顾真实数据稳健性）：`voxel_size=0.02`、`fpfh_radius=0.25`、`ransac_max_iteration=300000`（§1.1 建议 200k+；若 V100 上耗时敏感可回落 100000，见 default.yaml 注释）。



---

<!-- ABLATION-REPORT:START -->
## 1.3 消融实验对比（自动生成，`scripts/ablation_report.py` 维护，勿手改）

| 实验 | mAP@0.25 | mAP@0.5 | 参数量(M) | FLOPs(G) | 推理速度(ms/帧) | 结果文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 00 基线（单点云 VoteNet） | TBD | TBD | TBD | TBD | TBD | （未跑） |
| 01 单 RGB（伪 3D） | TBD | TBD | TBD | TBD | TBD | （未跑） |
| 02 融合-Concat（主） | TBD | TBD | TBD | TBD | TBD | （未跑） |
| 03 融合-Attention | TBD | TBD | TBD | TBD | TBD | （未跑） |
| 04 融合+轻量化 | TBD | TBD | TBD | TBD | TBD | （未跑） |
<!-- ABLATION-REPORT:END -->

### 1.4 V100 上线前准备完成（2026-09-22）

- **状态**：V100 上线前准备完成，CI 全绿，本地与远端一致。
- **内容**：
  1. 12 个文件（V100 准备轮全部产物）已按两条 commit 推送到 `axdvgnjzxcvbnml/pointcloud-registration-detection`（main），内容与本地逐字节校验一致；
  2. CI 三个 job（syntax-check / env-check / smoke，均不依赖 GPU/数据集）通过；
  3. 数据完整性检查（`scripts/check_data.sh`）、配准超参扫描（`scripts/sweep_registration.py`，结论见 `configs/default.yaml`）、断点续训（`train_fusion.py --resume` / `finetune_lightweight.py --resume_train`）、消融一键报告（`scripts/ablation_report.py`）、排坑手册（`docs/v100_pitfalls.md`）均已在 CPU 侧完成并验证。
- **待办（仅剩 GPU 侧）**：V100 上按 `docs/v100_checklist.md` 顺序执行，跑完五组消融后用 `scripts/ablation_report.py` 生成对比表回填上方 §1.3。

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
