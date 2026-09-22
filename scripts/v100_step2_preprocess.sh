#!/usr/bin/env bash
# ============================================================
# v100_step2_preprocess.sh —— 完整 SUN RGB-D / SUN3D 预处理
# ------------------------------------------------------------
# 1) parse_sunrgbd：解析 SUN RGB-D（RGB/深度/内参/外参 -> 统一 manifest）
# 2) depth_to_pointcloud：SUN3D 每帧深度图 -> 点云（.ply + .npz）
# 3) sample_frame_pairs：帧对采样（间隔 5/10/30，目标 ~400 对）
# 4) compute_pose_gt：帧对 6DOF 真值 T_AB
# 5) generate_detection_data：VoteNet 训练数据（每帧 50000 点）
#
# 用法：bash scripts/v100_step2_preprocess.sh
# 前置：data/SUNRGBD 与 data/SUN3D 软链接就绪（data/README.md）
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

step() { echo ""; echo "===== $1 ====="; }
ok()   { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }

step "0/6 前置检查"
[ -e data/SUNRGBD ]  || fail "缺少 data/SUNRGBD 软链接（见 data/README.md）"
[ -e data/SUN3D ]    || fail "缺少 data/SUN3D 软链接（见 data/README.md）"
python3 -c "import open3d, numpy; print('  open3d', open3d.__version__, '/ numpy', numpy.__version__)"
ok "数据与依赖就绪"

# 深度单位 sanity check：SUN RGB-D 原生深度为 16bit 毫米值。
# 深度转点云脚本的 --depth_scale 需与数据单位一致（本仓库约定 10000.0）。
step "1/6 解析 SUN RGB-D（统一格式 manifest）"
FIRST_DEPTH=$(find -L data/SUNRGBD -type f \( -iname '*depth*.png' -o -iname '*depth*.jpg' \) 2>/dev/null | head -1 || true)
if [ -n "${FIRST_DEPTH}" ]; then
  python3 - <<'PY' "${FIRST_DEPTH}"
import sys, numpy as np
try:
    from PIL import Image
    d = np.array(Image.open(sys.argv[1]))
except Exception:
    import cv2
    d = cv2.imread(sys.argv[1], -1)
print("  深度样例 %s: shape=%s dtype=%s min=%d max=%d" % (
    sys.argv[1], d.shape, d.dtype, d.min(), d.max()))
if d.max() > 20000:
    print("  -> 单位约 0.1mm，用 --depth_scale 10000.0（默认）")
else:
    print("  -> 单位约 1mm，用 --depth_scale 1000.0")
PY
fi
python3 preprocess/parse_sunrgbd.py --data_root data/SUNRGBD \
    --out_dir results/preprocess 2>&1 | tail -3
ok "SUN RGB-D 解析完成"

step "2/6 SUN3D 深度图 -> 点云（全部场景）"
python3 - <<'PY'
import subprocess
from pathlib import Path
scene_root = Path("data/SUN3D")
out_pcd = Path("results/preprocess/pcd")
out_pcd.mkdir(parents=True, exist_ok=True)
total = 0
for scene in sorted(p for p in scene_root.iterdir() if p.is_dir() and (p / "depth").is_dir()):
    for depth in sorted((scene / "depth").glob("*.png")):
        stem = depth.stem  # frame-000000
        idx = stem.replace("frame-", "")
        k = scene / "intrinsics" / ("frame-%s.txt" % idx)
        k_arg = ["--K", str(k)] if k.is_file() else []
        subprocess.run(["python3", "preprocess/depth_to_pointcloud.py",
                        "--depth", str(depth), *k_arg,
                        "--out_path", str(out_pcd / f"{scene.name}_{idx}"),
                        "--out_format", "ply"], check=True, capture_output=True)
        total += 1
print("  生成 %d 帧点云 -> results/preprocess/pcd" % total)
PY
ok "SUN3D 点云生成完成"

step "3/6 帧对采样（间隔 5/10/30，目标 400 对）"
python3 preprocess/sample_frame_pairs.py --sun3d_dir data/SUN3D \
    --intervals "5,10,30" --target_num 400 \
    --out_path results/preprocess/pairs/pairs.json 2>&1 | tail -2
ok "帧对清单 -> results/preprocess/pairs/pairs.json"

step "4/6 6DOF 真值 T_AB = inv(Pose_A) @ Pose_B"
python3 preprocess/compute_pose_gt.py --pairs results/preprocess/pairs/pairs.json \
    --sun3d_dir data/SUN3D --out_dir results/preprocess/pose_gt 2>&1 | tail -2
ok "真值 -> results/preprocess/pose_gt/pose_gt.json"

step "5/6 VoteNet 训练数据（每帧 50000 点）"
python3 preprocess/generate_detection_data.py \
    --pcd_dir results/preprocess/pcd \
    --label_dir data/SUNRGBD/label \
    --out_dir results/preprocess/detection \
    --num_points 50000 --num_classes 10 --vote_radius 0.1 --seed 2024 2>&1 | tail -2
ok "训练数据 -> results/preprocess/detection（含 split.json）"

step "6/6 VoteNet 格式自检（CPU，无需 GPU）"
python3 tests/test_votenet_dataloader_cpu.py 2>&1 | tail -3 || true

echo ""
echo "=============================================="
echo " 第 2 步完成 ✅（产物：pcd / pairs / pose_gt / detection）"
echo " 下一步：bash scripts/v100_step3_registration.sh"
echo "=============================================="
