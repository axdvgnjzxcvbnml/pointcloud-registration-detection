#!/usr/bin/env bash
# ============================================================
# v100_step5_fusion_train.sh —— 多模态融合头训练（主融合 Concat）
# ------------------------------------------------------------
# 冻结 VoteNet + YOLOv8n 主干，只训练融合头 + 检测头（~2M 参数），
# 60 epoch，输出权重与 mAP。
#
# 用法：bash scripts/v100_step5_fusion_train.sh
# 前置：v100_step1（两个主干权重）+ v100_step2（detection 数据 + frames）
# 预期耗时：60 epoch × 单卡 V100 ≈ 4~8 小时
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

step "0/3 前置检查"
[ -f weights/votenet_sunrgbd.pth ] || fail "缺少 VoteNet 权重：先跑 v100_step1_env.sh"
[ -f weights/yolov8n.pt ] || { echo "  [INFO] 自动下载 yolov8n.pt"; python3 -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
print('  yolov8n.pt 已就位')"; }
[ -f results/preprocess/detection/split.json ] || fail "缺少检测数据：先跑 v100_step2_preprocess.sh"
python3 -c "import torch; assert torch.cuda.is_available()" || fail "CUDA 不可用"
ok "前置就绪"

step "1/3 训练融合头（60 epoch，冻结双主干）"
python3 detection/train_fusion.py \
    --config configs/ablation/02_fusion_concat.yaml \
    --det_data_dir results/preprocess/detection \
    --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth \
    --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/detection/fusion \
    --device cuda --epochs 60 2>&1 | tail -8
ok "训练完成 -> results/detection/fusion/"

step "2/3 评测（mAP@0.25 / 0.5 + 参数量/FLOPs/推理速度）"
python3 detection/evaluate_detection.py \
    --config configs/ablation/02_fusion_concat.yaml \
    --ckpt results/detection/fusion/fusion_epoch060.pth \
    --det_data_dir results/preprocess/detection \
    --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth \
    --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/detection/fusion/eval \
    --device cuda --iou_thresholds "0.25,0.5" 2>&1 | tail -6
ok "评测完成 -> results/detection/fusion/eval/"

step "3/3 汇总"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/detection/fusion/eval/map.json")
if p.is_file():
    m = json.load(open(p))
    print("  融合-Concat  mAP@0.25=%.4f  mAP@0.5=%.4f" % (m.get("mAP@0.25", -1), m.get("mAP@0.5", -1)))
    for k in ("params", "flops", "latency_ms"):
        if k in m:
            print("  %s=%s" % (k, m[k]))
PY

echo ""
echo "=============================================="
echo " 第 5 步完成 ✅"
echo " 下一步：bash scripts/v100_step6_lightweight.sh"
echo "=============================================="
