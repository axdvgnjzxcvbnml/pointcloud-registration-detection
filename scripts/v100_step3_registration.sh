#!/usr/bin/env bash
# ============================================================
# v100_step3_registration.sh —— 配准评测（纯 CPU）
# ------------------------------------------------------------
# 对 results/preprocess/pairs 的全部帧对跑
#   预处理（去离群+降采样+法向）-> 粗配准（FPFH+RANSAC）-> 精配准（改进 ICP）
# 输出 RMSE / 旋转角误差 / 平移误差 / 成功率，与基线（点到点 ICP / FGR）对比。
#
# 用法：bash scripts/v100_step3_registration.sh
# 预期耗时：400 对约 30~60 分钟（纯 CPU，每对 ~5-10s）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

step "0/3 前置检查"
for f in results/preprocess/pairs/pairs.json \
         results/preprocess/pose_gt/pose_gt.json; do
  [ -f "$f" ] || fail "缺少 $f：先跑 scripts/v100_step2_preprocess.sh"
done
ok "帧对与真值就绪"

step "1/3 配准评测（改进 ICP + 点到点 ICP + FGR + 粗配准 四方法）"
python3 registration/evaluate_registration.py \
    --pairs results/preprocess/pairs/pairs.json \
    --pcd_dir results/preprocess/pcd \
    --pose_gt_dir results/preprocess/pose_gt \
    --out_dir results/registration/eval \
    --voxel_size 0.02 --fpfh_radius 0.25 \
    --ransac_max_iteration 100000 --ransac_distance_threshold 0.03 \
    --icp_max_iteration 50 --icp_threshold_min 0.02 --icp_threshold_factor 0.5 \
    --icp_early_stop_fitness 1e-6 \
    --rmse_th 0.05 --rot_th 5.0 --trans_th 0.05 2>&1 | tail -6
ok "评测完成 -> results/registration/eval/"

step "2/3 汇总输出"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/registration/eval/summary.json")
if p.is_file():
    s = json.load(open(p))
    print("  %-14s %8s %10s %10s %8s" % ("方法", "RMSE(m)", "旋转(°)", "平移(m)", "成功率"))
    for k, v in s.items():
        if isinstance(v, dict):
            print("  %-14s %8.4f %10.3f %10.4f %7.1f%%" % (
                k, v.get("rmse_mean", 0), v.get("rot_err_mean", 0),
                v.get("trans_err_mean", 0), 100 * v.get("success_rate", 0)))
else:
    print("  summary.json 未生成，请查看上方日志")
PY

step "3/3 可视化抽检（前 3 对）"
mkdir -p results/registration/viz
python3 - <<'PY'
import json, subprocess
from pathlib import Path
pairs = json.load(open("results/preprocess/pairs/pairs.json"))
pairs = pairs if isinstance(pairs, list) else pairs["pairs"]
for pr in pairs[:3]:
    gt = Path("results/preprocess/pose_gt") / (pr["pair_id"] + ".txt")
    src = Path("results/preprocess/pcd") / f"{pr['scene']}_{pr['frame_a']:06d}.ply"
    tgt = Path("results/preprocess/pcd") / f"{pr['scene']}_{pr['frame_b']:06d}.ply"
    if gt.exists() and src.exists() and tgt.exists():
        subprocess.run(["python3", "registration/visualize_registration.py",
                        "--source", str(src), "--target", str(tgt),
                        "--transform", str(gt),
                        "--out_dir", "results/registration/viz",
                        "--offscreen", "--window_size", "1280,480"],
                       check=True, capture_output=True)
        print("  已渲染", pr["pair_id"])
PY
ok "可视化 -> results/registration/viz/"

echo ""
echo "=============================================="
echo " 第 3 步完成 ✅"
echo " 下一步：bash scripts/v100_step4_votenet_baseline.sh"
echo "=============================================="
