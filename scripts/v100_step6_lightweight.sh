#!/usr/bin/env bash
# ============================================================
# v100_step6_lightweight.sh —— 轻量化微调 + 评测
# ------------------------------------------------------------
# 基于融合模型权重（v100_step5 产物），将融合头替换为轻量化头
# （DSConv1d，通道 256->128，Dropout0.3）微调 20 epoch，
# 输出参数量 / FLOPs / mAP，与主融合对比轻量化收益。
#
# 用法：bash scripts/v100_step6_lightweight.sh
# 前置：v100_step5（results/detection/fusion/fusion_epoch060.pth）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

FUSION_CKPT="results/detection/fusion/fusion_epoch060.pth"

step "0/3 前置检查"
[ -f "${FUSION_CKPT}" ] || fail "缺少 ${FUSION_CKPT}：先跑 v100_step5_fusion_train.sh"
python3 -c "import torch; assert torch.cuda.is_available()" || fail "CUDA 不可用"
ok "前置就绪"

step "1/3 轻量化头微调（20 epoch）"
python3 detection/finetune_lightweight.py \
    --config configs/ablation/04_fusion_lightweight.yaml \
    --resume "${FUSION_CKPT}" \
    --det_data_dir results/preprocess/detection \
    --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth \
    --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/detection/lightweight \
    --device cuda --finetune_epochs 20 2>&1 | tail -6
ok "微调完成 -> results/detection/lightweight/"

step "2/3 评测（参数量/FLOPs/mAP）"
python3 detection/evaluate_detection.py \
    --config configs/ablation/04_fusion_lightweight.yaml \
    --ckpt results/detection/lightweight/lightweight_epoch020.pth \
    --det_data_dir results/preprocess/detection \
    --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth \
    --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/detection/lightweight/eval \
    --device cuda --iou_thresholds "0.25,0.5" 2>&1 | tail -6
ok "评测完成 -> results/detection/lightweight/eval/"

step "3/3 汇总"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/detection/lightweight/eval/map.json")
if p.is_file():
    m = json.load(open(p))
    print("  轻量化  mAP@0.25=%.4f  mAP@0.5=%.4f" % (m.get("mAP@0.25", -1), m.get("mAP@0.5", -1)))
    for k in ("params", "flops", "latency_ms"):
        if k in m:
            print("  %s=%s" % (k, m[k]))
PY

echo ""
echo "=============================================="
echo " 第 6 步完成 ✅"
echo " 下一步：bash scripts/v100_step7_ablation.sh"
echo "=============================================="
