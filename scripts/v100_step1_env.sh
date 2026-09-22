#!/usr/bin/env bash
# ============================================================
# v100_step1_env.sh —— V100 环境准备
# ------------------------------------------------------------
# 1) 检查 nvidia-smi / nvcc / python / torch 版本
# 2) 编译 VoteNet PointNet2 CUDA 算子（TORCH_CUDA_ARCH_LIST="7.0"）
# 3) 下载 VoteNet 官方预训练权重
#
# 用法：bash scripts/v100_step1_env.sh
# 预期耗时：编译 5~15 分钟（首次），权重下载视网络而定
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

step "1/4 硬件与驱动"
nvidia-smi || fail "nvidia-smi 不可用：请安装 NVIDIA 驱动（CUDA 11.7 需驱动 >= 515.43）"
ok "GPU 已识别"
nvcc --version | tail -2 || fail "nvcc 不可用：请安装 CUDA Toolkit 11.7 并加入 PATH"
python3 --version
python3 - <<'PY' || fail "PyTorch/CUDA 检查失败"
import torch
print("  torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available()=False，检查 cu117 wheel（见 docs/setup.md 2 节）"
assert torch.cuda.get_device_capability(0)[0] == 7, "GPU 架构不是 sm_70（V100 应为 7.0）"
PY
ok "Python + PyTorch 1.13.1 + CUDA 11.7 就绪"

step "2/4 VoteNet 仓库检查"
if [ ! -d external/votenet ]; then
  fail "缺少 external/votenet：请先 git clone https://github.com/facebookresearch/votenet external/votenet"
fi
ok "external/votenet 存在"

step "3/4 编译 PointNet2 CUDA 算子（sm_70）"
PN2_DIR="external/votenet/pointnet2"
if [ ! -d "${PN2_DIR}" ]; then
  fail "缺少 external/votenet/pointnet2（VoteNet 目录结构异常）"
fi
cd "${PN2_DIR}"
rm -rf build _ext lib/*.so
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
cd "${ROOT_DIR}"
ok "PointNet2 编译完成（TORCH_CUDA_ARCH_LIST=7.0）"
python3 -c "import sys; sys.path.insert(0, 'external/votenet/pointnet2'); import pointnet2_utils; print('  import pointnet2_utils OK')" || fail "pointnet2_utils 导入失败（见 docs/troubleshooting.md 1 节）"

step "4/4 VoteNet 预训练权重"
CKPT="weights/votenet_sunrgbd.pth"
if [ -f "${CKPT}" ]; then
  ok "权重已存在：${CKPT}（跳过下载）"
else
  mkdir -p weights
  # 官方权重发布在 VoteNet README 的发布链接；先试官方托管域名，失败则给出手动指引
  for url in \
      "https://dl.fbaipublicfiles.com/votenet/sunrgbd_votenet.tar" \
      "https://dl.fbaipublicfiles.com/votenet/votenet_sunrgbd_10class.tar"; do
    if curl -fL --max-time 600 -o weights/votenet_download.tar "${url}"; then
      tar -xf weights/votenet_download.tar -C weights || true
      rm -f weights/votenet_download.tar
      # 若解出的是 checkpoint.tar 而非 .pth，改名
      [ -f weights/checkpoint.tar ] && mv weights/checkpoint.tar "${CKPT}"
      break
    fi
  done
  if [ ! -f "${CKPT}" ]; then
    echo "  [WARN] 自动下载未成功。请手动下载 VoteNet SUN RGB-D 预训练权重"
    echo "         （https://github.com/facebookresearch/votenet README 中的 release 链接）"
    echo "         保存为 weights/votenet_sunrgbd.pth 后重新运行本脚本。"
  else
    ok "权重就绪：${CKPT}"
  fi
fi

echo ""
echo "=============================================="
echo " 第 1 步完成 ✅"
echo " 下一步：bash scripts/v100_step2_preprocess.sh"
echo "=============================================="
