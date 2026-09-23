# 基于深度相机的物体点云拼接与目标检测

![CI](https://img.shields.io/github/actions/workflow/status/axdvgnjzxcvbnml/pointcloud-registration-detection/ci.yml?branch=main&label=CI&logo=github)
![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

面向室内场景（SUN RGB-D 数据集）的「RGB-D 点云拼接 + 三维目标检测」工程骨架与文档模板。

## 0. V100 上手指南

> 目标：V100 开机后**只跑 GPU 训练和推理**，环境、数据、调试全部已在 CPU 侧完成。
> 在 V100 上按顺序执行以下 8 个一键脚本（每个脚本自带前置检查、逐步日志、收尾提示）：

| 步骤 | 命令 | 预期耗时 | 预期输出 | 失败看 |
| --- | --- | --- | --- | --- |
| 1 环境 | `bash scripts/v100_step1_env.sh` | 5~15 分钟 | PointNet2 编译 OK、`weights/votenet_sunrgbd.pth` | `docs/troubleshooting.md` 1/2 节 |
| 2 预处理 | `bash scripts/v100_step2_preprocess.sh` | 10~30 分钟 | `results/preprocess/{pcd,pairs,pose_gt,detection}` | 3 节（数据路径/深度单位） |
| 3 配准 | `bash scripts/v100_step3_registration.sh` | 30~60 分钟 | `results/registration/eval/summary.json`（RMSE/旋转/平移/成功率） | 3.4 节 |
| 4 基线 | `bash scripts/v100_step4_votenet_baseline.sh` | 5~10 分钟 | `results/detection/baseline/`（mAP@0.25/0.5） | 1.2 节（kernel image） |
| 5 融合训练 | `bash scripts/v100_step5_fusion_train.sh` | 4~8 小时 | `results/detection/fusion/fusion_epoch060.pth` + mAP | 4 节（OOM） |
| 6 轻量化 | `bash scripts/v100_step6_lightweight.sh` | 1~2 小时 | `results/detection/lightweight/lightweight_epoch020.pth` + 参数量/FLOPs | 4 节 |
| 7 消融 | `bash scripts/v100_step7_ablation.sh` | 18~36 小时 | `results/ablation/ablation_summary.csv`（五组对比表） | 4 节 |
| 8 可视化 | `bash scripts/v100_step8_visualize.sh` | ~10 分钟 | `results/vis/`、`results/demo/*.gif|mp4` | 6 节（libGL/ffmpeg） |

详细的分步命令与检查项见 `docs/v100_checklist.md`；消融配置说明见 `configs/ablation/README.md`。

> **配准参数已锁定**（`configs/default.yaml` → `registration` 节）：voxel 0.03 / FPFH 0.40 /
> RANSAC 500k / mutual_filter=False——这是真实 SUN3D 数据（20 对 × 3 次重复）扫描出的
> 最优组合（中位成功率 50%），V100 上**直接使用，无需再调参**。配准评测命令：
> `python registration/evaluate_registration.py --config configs/default.yaml --pairs results/preprocess/pairs/pairs_real.json --pose_gt_dir results/preprocess/pose_gt`（真实数据对应帧对/真值路径以 `docs/v100_checklist.md` Step 3 为准）。

> **脚本调用方式**：仓库内所有 `.sh` 脚本（`scripts/*.sh`）一律用 `bash xxx.sh` 调用，**不要用 `./xxx.sh`**（GitHub 上文件无执行位，且 `bash` 调用与执行位无关、更稳定）。

### 0.1 CPU 侧已完成清单（上 V100 前不用重复做）

- [x] **真实 SUN3D 数据全流程验证**：MIT studyroom 49 帧（3DMatch 镜像）→ 深度转点云（`depth_scale=1000`）→ 60 帧对 → 6DOF 真值 → 配准全链路 → 可视化全部跑通；真实 vs 模拟配准指标对比见 `docs/experiment_log.md`
- [x] **VoteNet 训练数据管线**：真实点云 49 帧 × 50000 点 mini 集生成，与 VoteNet 官方 dataloader 格式对齐（双键 `point_cloud`/`point_clouds`），`tests/test_votenet_dataloader_cpu.py` 49/49 PASS
- [x] **模型自检**：`tests/test_cpu_forward_checks.py` 17 项 PASS（YOLO P3 特征、FusionModel 双融合方式前向、train_fusion 1 epoch、finetune_lightweight 1 epoch）；`tests/test_map_cpu.py` 18 项 PASS（IoU/mAP/load_gt）
- [x] **发现并修复的上线级 bug**：轻量化 `load_pretrained` 形状不匹配（`strict=False` 不跳过 size mismatch）→ 已按 shape 过滤（见 `docs/troubleshooting.md` 5.5）
- [x] **8 个 V100 一键脚本**：参数已调好、前置检查/日志/收尾齐全（`scripts/v100_step1~8.sh`）
- [x] **消融配置**：五组配置注释「跑完应得到什么结果」+ 对比关系 README
- [x] **真值管线**：`preprocess/extract_gt.py`（官方标注 → `data/gt/gt.json`）+ `detection/load_gt.py` + mAP 单测
- [x] **CI**：push 自动跑语法检查 + 无数据集/无 GPU 冒烟 + 环境检查（`.github/workflows/ci.yml`），三个 job 全绿
- [x] **数据完整性检查**：`scripts/check_data.sh`（场景三件套/文件数量一致/空文件与损坏文件检测），真实 studyroom 数据实测通过
- [x] **配准超参扫描**：`scripts/sweep_registration.py` 27 组合（RANSAC 100k/300k/500k × FPFH 0.15/0.25/0.35 × voxel 0.02/0.05/0.08）模拟数据全部 100% 成功，最优组合已写入 `configs/default.yaml`
- [x] **断点续训**：`detection/train_fusion.py --resume` 与 `detection/finetune_lightweight.py --resume_train`（checkpoint 含 epoch/模型/优化器/调度器/best 指标），单测通过
- [x] **消融一键报告**：`scripts/ablation_report.py` 读取五组配置结果生成 Markdown 对比表（写入 `docs/experiment_log.md` §1.3）+ 柱状图（`results/figures/ablation.png`），未跑实验自动留 TBD
- [x] **V100 排坑手册**：`docs/v100_pitfalls.md`（10 个坑：PointNet2 编译/CUDA 版本/软链失效/OOM/断点续训格式/消融字段/YOLOv8n 版本/Open3D 渲染/权重加载/mAP 口径，每坑含症状·原因·方案·文档指针）


## 1. 项目简介

本工程实现一条从原始 RGB-D 数据到拼接点云、再到三维目标检测的完整流水线：

- **数据预处理**（`preprocess/`）：解析 SUN RGB-D / SUN3D 序列，深度图反投影为三维点云，抽取帧对并计算 6DOF 配准真值，生成 VoteNet 训练数据；
- **点云拼接**（`registration/`）：FPFH + RANSAC 粗配准 → 改进 ICP（点到平面 + 自适应阈值 + 早停）精配准，纯 CPU / Python 实现；
- **三维目标检测**（`detection/`）：VoteNet 点云分支 + YOLOv8n RGB 分支特征级融合（Concat / Cross-Attention），支持轻量化头（DSConv1d）；
- **可视化集成**（`app/`）：Open3D 场景展示、拼接前后对比、检测结果叠加与演示导出。

> **说明**：本仓库为代码骨架与文档模板，所有实验数值（mAP、RMSE、参数量、耗时等）均为占位（TBD），
> 需在目标服务器（单卡 V100）上运行后回填。仓库不包含任何需要实际运行才能得到的结果。

## 2. 环境要求

### 硬件

- NVIDIA V100（16GB）单卡
- 建议内存 ≥ 32GB（点云预处理以 CPU 为主）

### 软件

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 20.04 / 22.04 | 服务器环境 |
| Python | 3.8 | 与 PyTorch 1.13.1 官方 wheel 匹配 |
| CUDA | 11.7 | 驱动版本需 ≥ 515 |
| PyTorch | 1.13.1（cu117） | 安装命令见下 |
| torchvision | 0.14.1 | 与 PyTorch 1.13.1 配套 |
| Open3D | 0.17.0 | 点云 IO / FPFH / RANSAC / ICP（CPU） |
| ultralytics | 8.2.x | YOLOv8n |
| VoteNet | facebookresearch/votenet | 需编译 PointNet2 CUDA 算子 |

详细安装步骤见 `docs/setup.md`（第七批输出）。

## 3. 数据准备

1. 下载 SUN RGB-D 数据集：
   - **SUNRGBD**：原始 RGB-D 帧（image / depth / label / extrinsics）；
   - **SUNRGBDtoolbox**：官方 MATLAB 工具箱（标注元数据、读码逻辑）；
   - **SUN3D 序列**：用于帧间拼接实验，含 `extrinsics/*.txt` 相机位姿。
2. 建立软链接（实体数据保留在服务器自己的目录，不拷贝进项目）：

   ```bash
   mkdir -p data
   ln -s /your/data/path/SUNRGBD        data/SUNRGBD
   ln -s /your/data/path/SUNRGBDtoolbox data/SUNRGBDtoolbox
   ln -s /your/data/path/SUN3D          data/SUN3D
   ```

   > 软链接必须使用绝对路径；`data/README.md` 中有更详细说明。

3. 数据组织约定（以 `data/SUNRGBD` 为例）：

   ```
   data/SUNRGBD/
   ├── depth/          # 深度图（16bit PNG，单位 mm）
   ├── image/          # RGB 图
   ├── label/          # 2D / 3D 标注
   ├── extrinsics/     # 相机外参 3x4 矩阵（部分子集提供）
   └── ...
   ```

   > 详细解析逻辑见 `preprocess/parse_sunrgbd.py`（第二批输出）。

## 4. 快速开始

### 4.1 克隆仓库

```bash
git clone <你的仓库地址> pointcloud-registration-detection
cd pointcloud-registration-detection
```

> 提交代码时建议启用仓库自带的提交信息模板：
> `git config commit.template .gitmessage`（前缀：feat / fix / docs / refactor / exp）。

### 4.2 安装依赖

```bash
conda create -n pcrd python=3.8 -y && conda activate pcrd

# 先装 PyTorch（必须 cu117 wheel，不能用默认源）
pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cu117

# 再装其余依赖
pip install -r requirements.txt
```

VoteNet / PointNet2 CUDA 算子的编译步骤（含 `TORCH_CUDA_ARCH_LIST="7.0"`）见 `docs/setup.md`。

### 4.3 准备数据

```bash
ln -s /your/data/path/SUNRGBD        data/SUNRGBD
ln -s /your/data/path/SUNRGBDtoolbox data/SUNRGBDtoolbox
ln -s /your/data/path/SUN3D          data/SUN3D
```

### 4.4 冒烟测试

```bash
# 完整模式：检查 Python / PyTorch / Open3D / 数据路径 / 深度转点云 / 粗+精配准
bash scripts/run_smoke.sh

# CI 模式：只跑不依赖数据集与 GPU 的检查项（合成数据）
bash scripts/run_smoke.sh --ci
```

### 4.5 模拟数据快速体验（无数据集也能跑通全流程）

不依赖 SUN RGB-D 真实数据，用 5 帧模拟 SUN3D 场景（`scripts/make_simulated_scene.py` 生成；深度图为 uint16、位姿为 3×4 矩阵，格式贴近真实数据）即可在纯 CPU 环境跑通全部拼接链路：

```bash
# 1) 生成模拟场景（5 帧，帧号 0/5/10/15/20，含 image/depth/extrinsics/intrinsics/clouds）
python scripts/make_simulated_scene.py --scene_dir data/SUN3D/sim_scene_001

# 2) 深度图 → 点云（5 帧，输出 results/preprocess/pcd/sim_scene_001_*.ply）
for f in 000000 000005 000010 000015 000020; do
  python preprocess/depth_to_pointcloud.py \
      --depth data/SUN3D/sim_scene_001/depth/frame-$f.depth.png \
      --K data/SUN3D/sim_scene_001/intrinsics/frame-$f.txt \
      --out_path results/preprocess/pcd/sim_scene_001_$f --out_format ply
done

# 3) 帧对采样 + 6DOF 真值
python preprocess/sample_frame_pairs.py --sun3d_dir data/SUN3D \
    --scene_list sim_scene_001 --intervals 5,10,30 --target_num 20 \
    --out_path results/preprocess/pairs/pairs.json
python preprocess/compute_pose_gt.py --pairs results/preprocess/pairs/pairs.json \
    --sun3d_dir data/SUN3D --out_dir results/preprocess/pose_gt

# 4) 配准链路：预处理 → 粗配准 → 精配准（以帧 0→5 为例）
python registration/preprocess_pointcloud.py \
    --input results/preprocess/pcd/sim_scene_001_000000.ply \
    --output results/registration/sim_scene_001_000000_clean.ply
python registration/coarse_registration.py \
    --source results/preprocess/pcd/sim_scene_001_000000.ply \
    --target results/preprocess/pcd/sim_scene_001_000005.ply \
    --output results/registration/sim_000000_000005_coarse.txt
python registration/fine_registration.py \
    --source results/preprocess/pcd/sim_scene_001_000000.ply \
    --target results/preprocess/pcd/sim_scene_001_000005.ply \
    --init results/registration/sim_000000_000005_coarse.txt \
    --output results/registration/sim_000000_000005_fine.txt

# 5) 评测（7 对 × 4 方法）与可视化
python registration/evaluate_registration.py \
    --pairs results/preprocess/pairs/pairs.json \
    --pcd_dir results/preprocess/pcd \
    --pose_gt_dir results/preprocess/pose_gt \
    --out_dir results/registration/eval
python registration/visualize_registration.py \
    --source results/preprocess/pcd/sim_scene_001_000000.ply \
    --target results/preprocess/pcd/sim_scene_001_000005.ply \
    --transform results/registration/sim_000000_000005_fine.txt \
    --out_dir results/registration/viz --offscreen

# 6) 可视化集成：场景加载 / 拼接前后对比 / 导出 GIF
python app/load_scene.py --scene-dir data/SUN3D/sim_scene_001/clouds --out-dir results/vis
python app/compare_registration.py \
    --source results/preprocess/pcd/sim_scene_001_000000.ply \
    --target results/preprocess/pcd/sim_scene_001_000005.ply \
    --transform results/registration/sim_000000_000005_fine.txt --out-dir results/vis
python app/export_demo.py --scene-dir data/SUN3D/sim_scene_001/clouds \
    --out-dir results/demo --n-frames 18 --fps 10 --gif --width 640 --height 360
```

> 模拟数据评测结果（7 对，Open3D 0.19 / 纯 CPU）：本方案（FPFH+RANSAC+改进 ICP）成功率 100%、RMSE 0.0077m、旋转误差 0.010°、平移误差 0.0004m；详见 `docs/experiment_log.md` 配准表 REG-001~003。

### 4.6 训练与评测

```bash
# 数据预处理
python preprocess/parse_sunrgbd.py --data_root data/SUNRGBD --out_dir results/preprocess
python preprocess/generate_detection_data.py --data_root results/preprocess \
    --out_dir results/detection_data

# 点云拼接（纯 CPU）
python registration/coarse_registration.py --src a.ply --dst b.ply
python registration/fine_registration.py --src a.ply --dst b.ply --init T_coarse.npy
python registration/evaluate_registration.py --pairs results/pairs.json

# 三维目标检测：训练（主融合 60 epoch）与评测
python detection/train_fusion.py --config configs/ablation/02_fusion_concat.yaml
python detection/evaluate_detection.py --config configs/ablation/02_fusion_concat.yaml \
    --ckpt results/ablation/02_fusion_concat/fusion_epoch060.pth

# 轻量化：基于融合权重微调 20 epoch
python detection/finetune_lightweight.py \
    --config configs/ablation/04_fusion_lightweight.yaml \
    --resume results/ablation/02_fusion_concat/fusion_epoch060.pth
```

### 4.7 真实 SUN3D 数据快速体验（CPU 已验证跑通）

> 已用真实 SUN3D 序列（MIT studyroom 49 帧，3DMatch 官方镜像，`data/SUN3D/mit_studyroom/`）跑通全部脚本，
> 命令如下（深度图单位 = 毫米，故 `--depth_scale 1000`）：

```bash
# 1) 深度图 → 点云（每帧 .ply + .npz，约 26 万点）
python preprocess/depth_to_pointcloud.py --depth data/SUN3D/mit_studyroom/depth/frame-000000.depth.png \
    --K data/SUN3D/mit_studyroom/intrinsics/frame-000000.txt \
    --depth_scale 1000 --out_path results/preprocess/pcd_real/mit_studyroom_000000 --out_format ply

# 2) 帧对采样（间隔 5/10/30 各 20 对）+ 6DOF 真值
python preprocess/sample_frame_pairs.py --sun3d_dir data/SUN3D --scene_list mit_studyroom \
    --target_num 60 --out_path results/preprocess/pairs/pairs_real.json
python preprocess/compute_pose_gt.py --pairs results/preprocess/pairs/pairs_real.json \
    --sun3d_dir data/SUN3D --out_dir results/preprocess/pose_gt_real

# 3) 配准全链路评测（60 对 × 4 方法，纯 CPU 约 25 分钟）
python registration/evaluate_registration.py --pairs results/preprocess/pairs/pairs_real.json \
    --pcd_dir results/preprocess/pcd_real --pose_gt_dir results/preprocess/pose_gt_real \
    --out_dir results/registration/eval_real

# 4) 可视化（场景总览 / 拼接前后对比 / GIF / 检测框叠加）
python app/load_scene.py --scene-dir data/SUN3D/mit_studyroom/clouds --frame-limit 5 --out-dir results/vis
python app/compare_registration.py --source results/preprocess/pcd_real/mit_studyroom_000060.ply \
    --target results/preprocess/pcd_real/mit_studyroom_000065.ply \
    --transform results/preprocess/pose_gt_real/mit_studyroom_000060_000065.txt --out-dir results/vis
python app/export_demo.py --scene-dir data/SUN3D/mit_studyroom/clouds --n-frames 25 --fps 8 --gif --out-dir results/demo
python app/overlay_detection.py --point-cloud results/preprocess/pcd_real/mit_studyroom_000000.ply \
    --detections results/detection/sim_dets.json --out-dir results/vis
```

真实 vs 模拟配准指标对比与原因分析见 `docs/experiment_log.md` 1.1 节。

> push 代码时 GitHub Actions 会自动执行 Python 语法检查与 `--ci` 冒烟测试（不跑 GPU / 数据集任务），配置见 `.github/workflows/ci.yml`。

## 5. 目录结构

```text
项目/
├── data/                        # 数据软链接目录（不存放实体文件）
│   ├── SUNRGBD/                 # → 软链接到 SUN RGB-D 原始数据
│   ├── SUNRGBDtoolbox/          # → 软链接到官方工具箱
│   └── SUN3D/                   # → 软链接到 SUN3D 序列（拼接实验）
├── external/                    # 第三方源码（只读引用）
│   └── votenet/                 # facebookresearch/votenet 克隆
├── preprocess/                  # 数据预处理模块（第二批）
│   ├── parse_sunrgbd.py         # 解析文件结构：RGB/深度/内参/外参
│   ├── depth_to_pointcloud.py   # 深度图 → 三维点云
│   ├── sample_frame_pairs.py    # SUN3D 帧对采样（间隔 5/10/30）
│   ├── compute_pose_gt.py       # 相对位姿 T_AB 真值（6DOF）
│   └── generate_detection_data.py # VoteNet 训练数据生成
├── registration/                # 点云拼接模块（第三批，纯 CPU）
│   ├── preprocess_pointcloud.py # 去噪/体素降采样/法向量
│   ├── coarse_registration.py   # FPFH + RANSAC 粗配准
│   ├── fine_registration.py     # 改进 ICP 精配准
│   ├── evaluate_registration.py # RMSE/旋转/平移/成功率评测
│   └── visualize_registration.py # 拼接前后对比
├── detection/                   # 三维目标检测模块（第四批）
│   ├── votenet_baseline.py      # VoteNet 基线推理
│   ├── yolov8_feature.py        # YOLOv8n P3 层特征提取
│   ├── projection.py            # 3D 种子点 → 2D 投影 + 双线性采样
│   ├── fusion_head.py           # Concat / Cross-Attention 融合头
│   ├── train_fusion.py          # 融合模型训练（60 epoch）
│   ├── lightweight_head.py      # DSConv1d 轻量化头
│   ├── finetune_lightweight.py  # 轻量化微调（20 epoch）
│   └── evaluate_detection.py    # mAP / 参数量 / FLOPs / 速度
├── app/                         # 集成与可视化（第六批）
│   ├── load_scene.py            # 场景多帧加载与展示
│   ├── compare_registration.py  # 拼接前后左右视图对比
│   ├── overlay_detection.py     # 3D 检测框 + 类别标注叠加
│   └── export_demo.py           # 导出 GIF / 视频
├── configs/                     # 配置文件
│   ├── default.yaml             # 主配置模板（数据路径/超参/融合开关/轻量化开关）
│   └── ablation/                # 消融实验配置（第五批）
├── weights/                     # 模型权重（官方预训练 / 训练产出）
├── results/                     # 实验结果与中间产物（预处理/配准/检测）
├── docs/                        # 文档（第七批）
│   ├── setup.md                 # 环境搭建
│   ├── troubleshooting.md       # 排错思路
│   └── experiment_log.md        # 实验记录模板
├── pipeline_overview.html       # 流水线总览图（本文件可视化）
├── scripts/                     # 工具脚本
│   └── run_smoke.sh             # 一键冒烟测试（--ci / SMOKE_CI=1 跳过数据与 GPU 检查）
├── .github/workflows/ci.yml     # GitHub Actions：语法检查 + CI 模式冒烟
├── .gitmessage                  # Git 提交信息模板（feat/fix/docs/refactor/exp）
├── requirements.txt             # 依赖清单
└── README.md
```

各批次交付对应关系：第二批 → `preprocess/`，第三批 → `registration/`，第四批 → `detection/`，第五批 → `configs/ablation/`，第六批 → `app/`，第七批 → `docs/`。

## 6. 实验结果

### 6.1 点云拼接（配准）——真实 SUN3D 数据结论（2026-09-23 锁定）

> 评测：MIT studyroom 49 帧 / 20 对（间隔 5/10/30 分层抽样），每组合 3 次重复取中位，
> 固定口径旋转 5° + 平移 0.05m。完整过程见 `docs/experiment_log.md` §1.5–§1.10。

| 方法 | 中位成功率 | min/max | iv5 | iv10 | iv30 |
| --- | --- | --- | --- | --- | --- |
| 基线（默认参数） | 27%（60 对历史值）/ 25%（20 对） | — | 57% | 14% | 0% |
| **最优组合**（voxel 0.03 / FPFH 0.40 / RANSAC 500k / mf=False） | **50%** | 35%–50% | **86%** | 29% | 33% |
| FGR | 35% | 35%–40% | 43% | 29% | 33% |
| 法向量一致性检查（30°） | 50% | 50%×3 | 86% | **43%** | 17% |
| 多尺度（0.08→0.03m） | 45% | 25%–50% | 57% | 57% | 17% |

**结论**：

- **传统方法最终成功率：50%（iv5 86% / iv10 29% / iv30 33%）**，相对基线 27% 提升
  23 个百分点，参数已锁定到 `configs/default.yaml`（`registration` 节）；
- **瓶颈分析**：低重叠场景（间隔 30）是结构性难题——失败对平移误差 0.31–1.23m，
  阈值敏感性分析证实 iv30 对任何阈值都不变；参数微调（任务一/二/三）与鲁棒方法
  （任务四：FGR / 法向检查 / 多尺度）均无法突破；
- **失败模式**：E 类「精度边缘」6 对（旋转 <2°、平移 0.05–0.12m 略超阈）+ F 类
  「平移严重错配」4 对（全 iv30）；成功子集精度旋转中位 0.54° / 平移 0.029m，
  精配准（改进 ICP）无问题；
- **未来工作**：低重叠场景需深度学习方法做初始对齐（Predator / CoFiNet /
  GeoTransformer），详见 `docs/registration_failure_analysis.md` §7。

### 6.2 检测（占位，待 V100 运行回填）

| 模块 | 配置 | 指标 | 数值 | 备注 |
| --- | --- | --- | --- | --- |
| 点云拼接 | FPFH+RANSAC+ICP（最优组合） | 成功率 | **50%**（真实数据已锁定） | iv5 86% / iv10 29% / iv30 33% |
| 点云拼接 | 点到点 ICP / FGR 基线 | 成功率 | 25% / 35% | 对比基线（20 对） |
| 检测 | VoteNet 基线（单点云） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合-Concat（主融合） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合-Attention（消融） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合+轻量化 | mAP / 参数量 / FLOPs | TBD | 效率对比 |

> 记录规范见 `docs/experiment_log.md`。

## 7. 相关参考

- VoteNet：<https://github.com/facebookresearch/votenet>
- YOLOv8：<https://github.com/ultralytics/ultralytics>
- SUN RGB-D：<https://rgbd.cs.princeton.edu/>

## 8. License 与声明

- 本仓库代码为工程骨架，仅供学习研究使用；
- 引用数据集与开源项目请遵守各自许可证；
- 本项目不提供任何需要实际运行才能得到的实验数值。
