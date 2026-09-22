#!/usr/bin/env bash
# ============================================================
# v100_step8_visualize.sh —— 可视化与演示导出
# ------------------------------------------------------------
# 跑 app/ 全部脚本，导出拼接对比图、GIF 与视频：
#   1) load_scene           场景多帧点云总览
#   2) compare_registration 拼接前后左右对比
#   3) export_demo          环绕演示 GIF / MP4
#   4) overlay_detection    3D 检测框叠加
#
# 用法：bash scripts/v100_step8_visualize.sh
# 前置：v100_step2（pcd/clouds）+ 检测结果（可选，覆盖检测用）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

mkdir -p results/vis results/demo

step "0/3 前置检查"
SCENE_DIR=""
for cand in data/SUN3D/mit_studyroom/clouds data/SUN3D/sim_scene_001/clouds; do
  [ -d "${cand}" ] && { SCENE_DIR="${cand}"; break; }
done
if [ -z "${SCENE_DIR}" ]; then
  # 退路：用 results/preprocess/pcd 里的 ply/npz 生成一个总览
  [ -d results/preprocess/pcd ] || fail "缺少点云数据：先跑 v100_step2_preprocess.sh"
  SCENE_DIR="results/preprocess/pcd"
fi
echo "  [INFO] 场景点云目录：${SCENE_DIR}"
ok "前置就绪"

step "1/4 app/load_scene —— 场景多帧总览"
python3 app/load_scene.py --scene-dir "${SCENE_DIR}" \
    --frame-limit 20 --voxel-size 0.02 \
    --out-dir results/vis --width 1280 --height 720 2>&1 | tail -2
ok "results/vis/clouds_scene.png"

step "2/4 app/compare_registration —— 拼接前后对比"
PAIR_JSON="results/preprocess/pairs/pairs.json"
if [ -f "${PAIR_JSON}" ]; then
  python3 - <<'PY'
import json, subprocess
from pathlib import Path
pairs = json.load(open("results/preprocess/pairs/pairs.json"))
pairs = pairs if isinstance(pairs, list) else pairs["pairs"]
if pairs:
    pr = pairs[0]
    gt = Path("results/preprocess/pose_gt") / (pr["pair_id"] + ".txt")
    src = Path("results/preprocess/pcd") / f"{pr['scene']}_{pr['frame_a']:06d}.ply"
    tgt = Path("results/preprocess/pcd") / f"{pr['scene']}_{pr['frame_b']:06d}.ply"
    if src.exists() and tgt.exists():
        cmd = ["python3", "app/compare_registration.py",
               "--source", str(src), "--target", str(tgt),
               "--out-dir", "results/vis", "--width", "1280", "--height", "720"]
        if gt.exists():
            cmd += ["--transform", str(gt)]
        else:
            cmd += ["--run-pipeline"]
        subprocess.run(cmd, check=True, capture_output=True)
        print("  已生成对比图：", pr["pair_id"])
PY
else
  warn_no_pairs=1
  echo "  [WARN] 无帧对清单，跳过对比图（先跑 v100_step2/3）"
fi
ok "results/vis/*_compare.png"

step "3/4 app/export_demo —— GIF + 视频"
python3 app/export_demo.py --scene-dir "${SCENE_DIR}" \
    --n-frames 60 --fps 15 --width 1280 --height 720 \
    --gif --video --out-dir results/demo 2>&1 | tail -2
ok "results/demo/clouds_demo.gif + clouds_demo.mp4"

step "4/4 app/overlay_detection —— 3D 检测框叠加"
DET_JSON=""
for cand in results/detection/fusion/eval/detections.json \
           results/detection/sim_dets.json; do
  [ -f "${cand}" ] && { DET_JSON="${cand}"; break; }
done
if [ -n "${DET_JSON}" ]; then
  PCD=$(ls results/preprocess/pcd/*.ply 2>/dev/null | head -1 || true)
  if [ -n "${PCD}" ]; then
    python3 app/overlay_detection.py --point-cloud "${PCD}" \
        --detections "${DET_JSON}" --out-dir results/vis \
        --width 1280 --height 720 2>&1 | tail -2
    ok "results/vis/*_detection.png"
  else
    echo "  [WARN] 无点云文件，跳过 overlay_detection"
  fi
else
  echo "  [WARN] 无检测结果 JSON，跳过 overlay_detection（训练后可重跑）"
fi

echo ""
echo "=============================================="
echo " 第 8 步完成 ✅ 全部 V100 步骤已执行完毕"
echo " 产物：results/vis/、results/demo/、results/registration/viz/"
echo "=============================================="
