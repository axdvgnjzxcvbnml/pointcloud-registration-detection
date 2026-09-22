# 环境搭建（V100 单卡服务器）

目标环境：**Python 3.8 + PyTorch 1.13.1 + CUDA 11.7 + Open3D 0.17.0**
适用：NVIDIA V100（sm_70，16GB）单卡服务器。

---

## 0. 安装前检查

```bash
# 1) 系统与驱动
uname -a                     # x86_64 Linux
nvidia-smi                   # 应显示 V100；右上角 Driver/CUDA Version 建议 ≥ 11.7
                             # CUDA 11.7 最低驱动：Linux 515.43（驱动过旧需先升级）

# 2) Python（无 3.8 时用 conda 建环境，见下）
python3 --version

# 3) 磁盘空间（数据集 + 权重 + 中间产物，建议预留 ≥ 100GB）
df -h /home
```

## 1. 创建 Conda 环境（Python 3.8）

```bash
conda create -n pcrd python=3.8 -y      # pcrd = pointcloud-registration-detection
conda activate pcrd
python --version                         # 确认 3.8.x
```

## 2. 安装 PyTorch 1.13.1（CUDA 11.7 wheel）

**必须使用 cu117 官方 wheel**，不能用 `pip install torch` 的默认源（会装 CPU 版）:

```bash
pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cu117
```

验证 CUDA 可用：

```bash
python -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
"
# 期望输出：available True，device NVIDIA V100...
```

## 3. 安装项目依赖

```bash
cd <项目根目录>
pip install -r requirements.txt
```

> `requirements.txt` 中 numpy 锁定 1.24.4（兼容 torch1.13 / open3d 0.17）；
> ultralytics 8.2.x；含 ninja（PointNet2 编译加速）。

验证 Open3D：

```bash
python -c "import open3d; print(open3d.__version__)"   # 0.17.0
```

> 若 headless 服务器报 `libGL.so.1: cannot open shared object file`：
> `sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0`

## 4. 克隆 VoteNet 并编译 PointNet2 CUDA 算子（重点）

```bash
# 1) 克隆到 external/
git clone https://github.com/facebookresearch/votenet.git external/votenet
cd external/votenet

# 2) （若仓库无该文件）补空 __init__.py，保证可被 import
touch pointnet2/__init__.py 2>/dev/null || true

# 3) 安装 VoteNet 自身依赖（easydict 等，项目 requirements 已含，可跳过重复项）
pip install -r requirements.txt

# 4) 编译 PointNet2 —— 关键：V100 是 Volta 架构，sm_70！
cd pointnet2
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
```

**重点说明（V100 必读）：**

| 要点 | 值 | 原因 |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST` | **`"7.0"`** | V100 是 Volta（sm_70）。写成 7.5（Turing）或 8.0（Ampere）会导致运行时 `no kernel image is available for execution on the device` |
| 多卡混插 | `"7.0;7.5"` | 若机器混有 V100 与 RTX 卡才需要多架构 |
| 编译器 | gcc ≤ 10 | PyTorch 1.13 对过新 gcc 兼容性差；报错时 `conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9` 后用 `gcc_linux-64` 编译 |
| 清缓存重编 | `rm -rf build _ext*` | 改过 ARCH 列表后必须清掉旧产物再编译 |

验证编译结果（应有 `_ext` 相关产物，且能 import）：

```bash
cd external/votenet/pointnet2
ls _ext/ 2>/dev/null || find . -name "*.so"      # 应有编译出的 .so
cd <项目根目录>
python -c "
import sys; sys.path.insert(0, 'external/votenet')
from pointnet2 import pointnet2_utils
print('pointnet2 OK:', pointnet2_utils.__file__)
"
```

> 编译失败排查见 `docs/troubleshooting.md` 第 1 节。

## 5. 数据准备（SUN RGB-D / SUN3D）

```bash
# 目录结构（data/ 是软链接目录，见 data/README.md）
mkdir -p data
ln -s /绝对路径/SUNRGBD           data/SUNRGBD
ln -s /绝对路径/SUNRGBDtoolbox    data/SUNRGBDtoolbox
ln -s /绝对路径/SUN3D             data/SUN3D

# 验证软链接有效
ls -l data/
ls data/SUNRGBD/ | head
```

官方数据来源：
- SUN RGB-D：http://rgbd.cs.princeton.edu/（SUNRGBD 图像/深度 + 标注）
- SUNRGBDtoolbox：官方工具箱（标注解析、相机内参元数据）
- SUN3D：https://sun3d.cs.princeton.edu/（多帧序列，帧对拼接用；
  每场景含 `extrinsics/*.txt` 3×4 相机位姿）

数据约定（与 `preprocess/` 各脚本对齐，详见各脚本 docstring）：
- 深度图 scale=10000（mm），截断 8m；
- 相机内参优先读 SUNRGBDtoolbox 元数据；占位默认内参在代码中标注 TODO；
- SUN3D 帧对按场景名分组，间隔 5/10/30 帧，目标 300–500 对。

## 6. 权重下载

```bash
mkdir -p weights
# 1) YOLOv8n（COCO 预训练）—— 自动下载或手动放置
#    ultralytics 首次调用自动下载；离线时手动放置到 weights/yolov8n.pt
# 2) VoteNet 预训练权重（SUN RGB-D）
#    从 VoteNet 官方 README 提供的链接下载，放置 weights/votenet_sunrgbd.pth
```

## 7. 冒烟测试（整条流水线）

```bash
# 预处理：单帧点云
python preprocess/depth_to_pointcloud.py --depth data/SUNRGBD/.../depth.png \
    --out results/smoke

# 配准（纯 CPU）：帧对粗+精配准
python registration/fine_registration.py --source ... --target ... --out results/smoke

# 检测：融合模型参数打印（--dry-run，不训练）
python detection/train_fusion.py --config configs/default.yaml --dry-run

# 可视化：离屏渲染（headless 服务器）
python app/load_scene.py --scene-dir data/SUN3D/scene_001 --headless \
    --out-dir results/smoke
```

每步能正常产出文件即环境 OK；报错对照 `docs/troubleshooting.md`。

## 8. 常见坑速查

| 现象 | 一句解法 |
| --- | --- |
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 torch，按第 2 节重装 cu117 wheel |
| PointNet2 运行时无 kernel | 重编时 `TORCH_CUDA_ARCH_LIST="7.0"` 并清 build 缓存 |
| open3d 导入缺 libGL | `apt-get install -y libgl1 libglib2.0-0` |
| 数据集路径 FileNotFound | 软链接失效或绝对路径已变，重新 `ln -s` |
| CUDA OOM | 见 troubleshooting 第 4 节（减 batch / 减点数 / 梯度累积） |
