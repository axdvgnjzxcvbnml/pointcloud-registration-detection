#!/usr/bin/env bash
# ============================================================
# v100_step7_ablation.sh —— 五组消融实验全流程
# ------------------------------------------------------------
# 按 configs/ablation/ 五组配置依次执行，输出完整消融实验表：
#   00 votenet_baseline   ：VoteNet 预训练基线评测（不训练）
#   01 rgb_pseudo3d       ：单 RGB（伪 3D）训练 + 评测
#   02 fusion_concat      ：融合-Concat 训练 + 评测（主融合）
#   03 fusion_attention   ：融合-Attention 训练 + 评测
#   04 fusion_lightweight ：轻量化微调 + 评测（依赖 02 权重）
#
# 结果汇总：results/ablation/ablation_summary.csv
#
# 用法：bash scripts/v100_step7_ablation.sh
# 前置：v100_step1/2/4/5（权重与数据；02 权重供 04 复用）
# 预期耗时：4 组 × 60 epoch + 20 epoch ≈ 18~36 小时（可按需 --epochs 缩短）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

EPOCHS="${EPOCHS:-60}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
SUMMARY="results/ablation/ablation_summary.csv"

step "0/3 前置检查"
[ -f weights/votenet_sunrgbd.pth ] || fail "缺少 VoteNet 权重：先跑 v100_step1_env.sh"
[ -f results/preprocess/detection/split.json ] || fail "缺少检测数据：先跑 v100_step2_preprocess.sh"
python3 -c "import torch; assert torch.cuda.is_available()" || fail "CUDA 不可用"
mkdir -p results/ablation
[ -f "${SUMMARY}" ] || echo "mode,mAP@0.25,mAP@0.5,params,flops,latency_ms" > "${SUMMARY}"
ok "前置就绪（汇总表：${SUMMARY}）"

run_eval() { # $1=config $2=ckpt $3=outdir
  python3 detection/evaluate_detection.py \
      --config "$1" --ckpt "$2" \
      --det_data_dir results/preprocess/detection \
      --frames_dir results/preprocess/frames \
      --votenet_ckpt weights/votenet_sunrgbd.pth \
      --yolov8_ckpt weights/yolov8n.pt \
      --out_dir "$3" --device cuda --iou_thresholds "0.25,0.5" >/dev/null 2>&1 || true
  python3 - <<'PY' "$1" "$3" "$SUMMARY"
import csv, json, sys
from pathlib import Path
cfg, out, summary = sys.argv[1], sys.argv[2], sys.argv[3]
mode = Path(cfg).stem.replace("0", "", 1)
p = Path(out) / "map.json"
if not p.is_file():
    print("  [WARN] %s 无 map.json，跳过汇总" % mode); sys.exit(0)
m = json.load(open(p))
row = [mode, m.get("mAP@0.25", ""), m.get("mAP@0.5", ""),
       m.get("params", ""), m.get("flops", ""), m.get("latency_ms", "")]
with open(summary, "a", newline="") as f:
    csv.writer(f).writerow(row)
print("  已记录 %s 到 %s" % (mode, summary))
PY
}

step "1/5 消融 00：VoteNet 基线（评测预训练权重，不训练）"
run_eval configs/ablation/00_votenet_baseline.yaml \
    weights/votenet_sunrgbd.pth results/ablation/00_votenet_baseline/eval

step "2/5 消融 01：单 RGB（伪 3D）"
python3 detection/train_fusion.py --config configs/ablation/01_rgb_pseudo3d.yaml \
    --det_data_dir results/preprocess/detection --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/ablation/01_rgb_pseudo3d --device cuda --epochs "${EPOCHS}" 2>&1 | tail -2
run_eval configs/ablation/01_rgb_pseudo3d.yaml \
    results/ablation/01_rgb_pseudo3d/fusion_epoch$(printf "%03d" "${EPOCHS}").pth \
    results/ablation/01_rgb_pseudo3d/eval

step "3/5 消融 02：融合-Concat（主融合）"
python3 detection/train_fusion.py --config configs/ablation/02_fusion_concat.yaml \
    --det_data_dir results/preprocess/detection --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/ablation/02_fusion_concat --device cuda --epochs "${EPOCHS}" 2>&1 | tail -2
run_eval configs/ablation/02_fusion_concat.yaml \
    results/ablation/02_fusion_concat/fusion_epoch$(printf "%03d" "${EPOCHS}").pth \
    results/ablation/02_fusion_concat/eval

step "4/5 消融 03：融合-Attention"
python3 detection/train_fusion.py --config configs/ablation/03_fusion_attention.yaml \
    --det_data_dir results/preprocess/detection --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/ablation/03_fusion_attention --device cuda --epochs "${EPOCHS}" 2>&1 | tail -2
run_eval configs/ablation/03_fusion_attention.yaml \
    results/ablation/03_fusion_attention/fusion_epoch$(printf "%03d" "${EPOCHS}").pth \
    results/ablation/03_fusion_attention/eval

step "5/5 消融 04：融合 + 轻量化（基于 02 权重微调）"
python3 detection/finetune_lightweight.py --config configs/ablation/04_fusion_lightweight.yaml \
    --resume results/ablation/02_fusion_concat/fusion_epoch$(printf "%03d" "${EPOCHS}").pth \
    --det_data_dir results/preprocess/detection --frames_dir results/preprocess/frames \
    --votenet_ckpt weights/votenet_sunrgbd.pth --yolov8_ckpt weights/yolov8n.pt \
    --out_dir results/ablation/04_fusion_lightweight --device cuda \
    --finetune_epochs "${FINETUNE_EPOCHS}" 2>&1 | tail -2
run_eval configs/ablation/04_fusion_lightweight.yaml \
    results/ablation/04_fusion_lightweight/lightweight_epoch$(printf "%03d" "${FINETUNE_EPOCHS}").pth \
    results/ablation/04_fusion_lightweight/eval

echo ""
echo "===== 消融实验汇总表：${SUMMARY} ====="
column -s, -t "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo ""
echo "=============================================="
echo " 第 7 步完成 ✅"
echo " 下一步：bash scripts/v100_step8_visualize.sh"
echo "=============================================="
