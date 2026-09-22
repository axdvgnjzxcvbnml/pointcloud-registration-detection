# docs/setup.md —— 环境搭建（第七批交付）

目标环境：**单卡 NVIDIA V100（sm_70）**，Ubuntu 20.04 / 22.04。

> 核心约束：Python 3.8 + CUDA 11.7 + PyTorch 1.13.1（cu117 wheel）
> + Open3D 0.17.0 + ultralytics 8.2.x。
> 配准模块为纯 CPU；检测模块需要 GPU。

## 0. 环境速查表

| 组件 | 版本 | 安装方式 |
| --- | --- | --- |
| Python | 3.8 | conda |
| CUDA Toolkit | 11.7 | 系统已装（驱动 ≥ 515） |
| PyTorch | 1.13.1+cu117 | pip（官方 cu117 index） |
| torchvision | 0.14.1 | pip（同上） |
| Open3D | 0.17.0 | pip |
| ultralytics | 8.2.x | pip |
| VoteNet | master | 源码编译（需编译 PointNet2 算子） |
| numpy | 1.24.4 | pip |
| scipy | 1.9~1.10 | pip |
| PyYAML | ≥6.0 | pip |
| Pillow | ≥10.0 | pip |

## 1. 创建 conda 环境

```bash
conda create -n pcrd python=3.8 -y
conda activate pcrd
python -m pip install --upgrade pip
```

## 2. 安装 PyTorch（必须先装，用官方 cu117 wheel）

```bash
pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cu117
```

> 不要使用默认 PyPI 的 torch（CPU 版）或更高版本 wheel，
> 否则与 CUDA 11.7 / Open3D 0.17.0 的 ABI 可能不兼容。

## 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

其中包含：

```text
numpy==1.24.4
scipy>=1.9,<1.11
open3d==0.17.0
pillow>=10.0
PyYAML>=6.0
opencv-python>=4.8
ultralytics==8.2.*
# 可选（评测/训练增强）：
# thop
# tensorboardX
# imageio
# imageio-ffmpeg
```

## 4. 克隆并编译 VoteNet / PointNet2

```bash
mkdir -p external && cd external
# 只读引用（不 fork，不修改官方代码）
git clone https://github.com/facebookresearch/votenet.git
cd votenet
# 编译 PointNet2 CUDA 算子（V100 = sm_70，必须显式指定）
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
cd ../..
```

> 若编译报错：
> 1. 确认 `nvcc -V` 为 CUDA 11.7；
> 2. 确认 PyTorch 的 CUDA 版本与 nvcc 一致（`python -c "import torch;print(torch.version.cuda)"`）；
> 3. 依赖 `gcc/g++` 与系统头文件，缺什么补什么；
> 4. 详见 docs/troubleshooting.md。

## 5. 下载模型权重

```bash
# YOLOv8n（首次调用自动下载到工作目录）
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# VoteNet 官方预训练权重（README 中链接下载，放 weights/ 目录）
# 若为 .tar（含 'state_dict' 键），votenet_baseline.py 已兼容加载
wget <votenet下载链接> -O weights/votenet_sunrgbd.pth
```

## 6. 数据准备（软链接）

```bash
ln -s /your/data/path/SUNRGBD        data/SUNRGBD
ln -s /your/data/path/SUNRGBDtoolbox data/SUNRGBDtoolbox
ln -s /your/data/path/SUN3D          data/SUN3D
```

## 7. 冒烟测试

```bash
bash scripts/run_smoke.sh          # 完整模式
bash scripts/run_smoke.sh --ci     # CI 模式（不依赖 GPU / 数据）
```

## 8. 常见坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| import torch 报 CUDA 错误 | wheel 与驱动不匹配 | 确认 cu117 wheel + 驱动 ≥ 515 |
| Open3D import 失败 | 缺 libGL | `apt install libgl1 libglib2.0-0` |
| votenet 编译失败 | 架构没写 / 版本不匹配 | `TORCH_CUDA_ARCH_LIST=7.0`，核对 nvcc |
| YOLOv8 输入尺寸不符 | 没做 letterbox | 统一 640×640 + letterbox |

更多排错见 `docs/troubleshooting.md`。
