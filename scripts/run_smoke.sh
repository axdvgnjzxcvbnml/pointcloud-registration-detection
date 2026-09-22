#!/usr/bin/env bash
# ============================================================
# scripts/run_smoke.sh —— 一键冒烟测试
# ------------------------------------------------------------
# 用法：
#   bash scripts/run_smoke.sh          # 完整模式
#   bash scripts/run_smoke.sh --ci     # CI 模式（不依赖 GPU / 数据集）
#   SMOKE_CI=1 bash scripts/run_smoke.sh  # 等价于 --ci（GitHub Actions 用）
#
# 完整模式检查项：
#   1. Python / PyTorch / CUDA 可用
#   2. Open3D 可用（CPU）
#   3. 数据软链接存在（data/SUNRGBD 等）
#   4. 深度图 -> 点云（合成深度图）
#   5. 粗配准 + 精配准（合成点云对，纯 CPU）
#
# CI 模式（GitHub Actions）只跑：
#   Open3D 可用 + 合成深度图转点云 + 合成配准，
#   不检查 GPU / PyTorch / 数据集路径。
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CI_MODE=0
if [[ "${1:-}" == "--ci" || "${SMOKE_CI:-0}" == "1" ]]; then
  CI_MODE=1
fi

pass=0
fail=0

ok()   { echo "[OK]   $1"; pass=$((pass+1)); }
fail_() { echo "[FAIL] $1"; fail=$((fail+1)); }

run_py() {
  local name="$1"; shift
  if python "$@"; then ok "$name"; else fail_ "$name"; fi
}

# ---------- 1. 基础环境 ----------
if [[ $CI_MODE -eq 0 ]]; then
  run_py "Python 版本" python -c "import sys; assert sys.version_info[:2]==(3,8), sys.version" || true
  run_py "PyTorch 可用" python -c "import torch; print('torch', torch.__version__)"
  run_py "CUDA 可用" python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'"
fi
run_py "Open3D 可用" python -c "import open3d as o3d; print('open3d', o3d.__version__)"

# ---------- 2. 数据路径（完整模式） ----------
if [[ $CI_MODE -eq 0 ]]; then
  for d in data/SUNRGBD data/SUNRGBDtoolbox data/SUN3D; do
    if [[ -d "$d" ]]; then ok "数据软链接 $d"; else fail_ "数据软链接 $d（缺失）"; fi
  done
fi

# ---------- 3. 合成深度图 -> 点云 ----------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
run_py "深度转点云(合成)" python -c "
import numpy as np, json
H, W = 64, 64
# 合成深度图：近处斜面
v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
depth = (2000 + (u + v) * 2).astype(np.uint16)   # mm
np.save('$TMP/d.npy', depth)
K = [[W, 0, W/2], [0, W, H/2], [0, 0, 1]]
json.dump(K, open('$TMP/K.json', 'w'))
import sys; sys.path.insert(0, 'preprocess')
from depth_to_pointcloud import depth_to_points, load_depth
from PIL import Image
Image.fromarray(depth).save('$TMP/d.png')
dm = load_depth('$TMP/d.png')
xyz, _, _ = depth_to_points(dm, np.array(K))
assert len(xyz) > 1000, '点云过少'
print('点云点数', len(xyz))
"

# ---------- 4. 合成配准（粗 + 精） ----------
run_py "合成配准流水线" python -c "
import numpy as np, open3d as o3d, sys
sys.path.insert(0, 'registration')
from preprocess_pointcloud import preprocess_pointcloud
from coarse_registration import coarse_registration
from fine_registration import fine_registration

rng = np.random.default_rng(0)
# 合成：平面上两个相交球体点集（保证有可配准结构）
pts = rng.uniform(-1, 1, (3000, 3))
pts = pts[np.linalg.norm(pts, axis=1) <= 1.0]
src = o3d.geometry.PointCloud()
src.points = o3d.utility.Vector3dVector(pts)
dst = o3d.geometry.PointCloud()
dst.points = o3d.utility.Vector3dVector(pts)
T_gt = np.eye(4); T_gt[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.1, -0.2, 0.3))
T_gt[:3, 3] = [0.05, -0.03, 0.02]
dst.transform(T_gt)

s = preprocess_pointcloud(src, voxel_size=0.05)
d = preprocess_pointcloud(dst, voxel_size=0.05)
T_c, _ = coarse_registration(s, d, voxel_size=0.05)
T_f, info = fine_registration(s, d, T_c, voxel_size=0.05)
print('粗 fitness', info.get('fitness', 0))
assert np.all(np.isfinite(T_f))
"

# ---------- 汇总 ----------
echo
echo "================ 冒烟测试汇总 ================"
echo "通过: $pass    失败: $fail"
if [[ $fail -gt 0 ]]; then
  echo "存在失败项，请检查上方 [FAIL] 日志。"
  exit 1
else
  echo "全部通过。"
fi
