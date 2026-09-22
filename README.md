# 基于深度相机的物体点云拼接与目标检测

面向室内场景（SUN RGB-D 数据集）的「RGB-D 点云拼接 + 三维目标检测」工程骨架与文档模板。

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

### 4.5 训练与评测

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

## 6. 实验结果（占位，待运行回填）

| 模块 | 配置 | 指标 | 数值 | 备注 |
| --- | --- | --- | --- | --- |
| 点云拼接 | FPFH+RANSAC+ICP | RMSE (m) | TBD | 待运行 |
| 点云拼接 | 点到点 ICP / FGR 基线 | 成功率 | TBD | 对比基线 |
| 检测 | VoteNet 基线（单点云） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合-Concat（主融合） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合-Attention（消融） | mAP@0.25 / mAP@0.5 | TBD | 10 类 |
| 检测 | 融合+轻量化 | mAP / 参数量 / FLOPs | TBD | 效率对比 |

> 记录规范见 `docs/experiment_log.md`（第七批输出）。

## 7. 相关参考

- VoteNet：<https://github.com/facebookresearch/votenet>
- YOLOv8：<https://github.com/ultralytics/ultralytics>
- SUN RGB-D：<https://rgbd.cs.princeton.edu/>

## 8. License 与声明

- 本仓库代码为工程骨架，仅供学习研究使用；
- 引用数据集与开源项目请遵守各自许可证；
- 本项目不提供任何需要实际运行才能得到的实验数值。
