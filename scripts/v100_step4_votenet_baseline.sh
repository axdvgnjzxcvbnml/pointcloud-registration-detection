#!/usr/bin/env bash
# ============================================================
# v100_step4_votenet_baseline.sh —— VoteNet 基线评测
# ------------------------------------------------------------
# 加载 VoteNet 官方预训练权重，在 SUN RGB-D 验证集上推理，
# 输出 mAP@0.25 与 mAP@0.5（3D IoU，10 类及均值）。
#
# 用法：bash scripts/v100_step4_votenet_baseline.sh
# 前置：v100_step1（权重）+ v100_step2（results/preprocess/detection）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

step "0/2 前置检查"
[ -f weights/votenet_sunrgbd.pth ] || fail "缺少 weights/votenet_sunrgbd.pth：先跑 v100_step1_env.sh"
[ -f results/preprocess/detection/split.json ] || fail "缺少训练数据：先跑 v100_step2_preprocess.sh"
python3 -c "import torch; assert torch.cuda.is_available()" || fail "CUDA 不可用"
ok "前置就绪"

step "1/2 VoteNet 推理 + mAP 评测"
python3 detection/votenet_baseline.py \
    --ckpt weights/votenet_sunrgbd.pth \
    --det_data_dir results/preprocess/detection \
    --split results/preprocess/detection/split.json \
    --num_class 10 --batch_size 8 --device cuda \
    --out_dir results/detection/baseline 2>&1 | tail -8
ok "评测完成 -> results/detection/baseline/"

step "2/2 结果汇总"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/detection/baseline/map.json")
if p.is_file():
    m = json.load(open(p))
    print("  mAP@0.25 = %.4f   mAP@0.5 = %.4f" % (m.get("mAP@0.25", -1), m.get("mAP@0.5", -1)))
else:
    print("  map.json 未生成；请查看上方日志（该文件为训练后由 evaluate_detection 生成）")
PY

echo ""
echo "=============================================="
echo " 第 4 步完成 ✅"
echo " 下一步：bash scripts/v100_step5_fusion_train.sh"
echo "=============================================="
