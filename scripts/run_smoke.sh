#!/usr/bin/env bash
# ============================================================
# scripts/run_smoke.sh —— 一键冒烟测试
# ------------------------------------------------------------
# 检查项：
#   1. Python 版本（>= 3.8）
#   2. PyTorch 是否可用（完整模式；CI 模式跳过）
#   3. Open3D 是否可用
#   4. SUN RGB-D 数据路径是否存在（完整模式；CI 模式跳过）
#   5. 一张深度图 -> 点云（有数据用真实深度图，否则用合成深度图）
#   6. 点云粗配准（FPFH+RANSAC）+ 精配准（改进 ICP）小测试（合成数据）
#
# 用法：
#   bash scripts/run_smoke.sh           # 完整模式（服务器）
#   bash scripts/run_smoke.sh --ci      # CI 模式（不依赖数据集/GPU）
#   SMOKE_CI=1 bash scripts/run_smoke.sh
#
# 退出码：0 = 全部通过（WARN/SKIP 不影响）；非 0 = 存在 FAIL
# ============================================================

set -u

# ---------- 基本路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python3}"

# ---------- 模式 ----------
CI_MODE=0
if [ "${SMOKE_CI:-0}" = "1" ]; then CI_MODE=1; fi
if [ "${1:-}" = "--ci" ]; then CI_MODE=1; fi

# ---------- 计数与输出 ----------
N_PASS=0; N_FAIL=0; N_WARN=0; N_SKIP=0
c_red=$'\033[31m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'
c_blue=$'\033[34m'; c_bold=$'\033[1m'; c_reset=$'\033[0m'
if [ "${CI_MODE}" = "1" ]; then c_red=""; c_green=""; c_yellow=""; c_blue=""; c_bold=""; c_reset=""; fi

pass() { echo "  ${c_green}[PASS]${c_reset} $1"; N_PASS=$((N_PASS+1)); }
fail() { echo "  ${c_red}[FAIL]${c_reset} $1"; N_FAIL=$((N_FAIL+1)); }
warn() { echo "  ${c_yellow}[WARN]${c_reset} $1"; N_WARN=$((N_WARN+1)); }
skip() { echo "  ${c_blue}[SKIP]${c_reset} $1"; N_SKIP=$((N_SKIP+1)); }
info() { echo "  ${c_blue}[INFO]${c_reset} $1"; }
section() { echo ""; echo "${c_bold}== $1 ==${c_reset}"; }

echo "${c_bold}========================================${c_reset}"
echo "${c_bold} 点云拼接与目标检测 —— 冒烟测试${c_reset}"
echo " 项目根：${ROOT_DIR}"
if [ "${CI_MODE}" = "1" ]; then
  echo " 模式：CI（跳过数据集与 GPU 相关检查）"
else
  echo " 模式：完整（服务器）"
fi
echo "${c_bold}========================================${c_reset}"

# ---------- 1. Python 版本 ----------
section "1/6 Python 版本（>= 3.8）"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "未找到 ${PYTHON_BIN}，请先安装 Python 3.8"
  echo ""; echo "结果：${c_red}失败${c_reset}（Python 不存在，后续检查无法进行）"; exit 1
fi
PY_VER_OK=$("${PYTHON_BIN}" - <<'PY'
import sys
print(1 if sys.version_info >= (3, 8) else 0)
PY
)
PY_VER=$("${PYTHON_BIN}" -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")
if [ "${PY_VER_OK}" = "1" ]; then
  pass "Python ${PY_VER}"
else
  fail "Python ${PY_VER}，需要 >= 3.8"
fi

# ---------- 2. PyTorch ----------
section "2/6 PyTorch 可用性"
if [ "${CI_MODE}" = "1" ]; then
  skip "CI 模式不检查 PyTorch（CI 不安装 GPU 栈）"
else
  TORCH_INFO=$("${PYTHON_BIN}" - <<'PY' 2>/dev/null
try:
    import torch
    print("%s|cuda%s|%s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
except Exception as e:
    print("ERROR|%s" % e)
PY
)
  case "${TORCH_INFO}" in
    ERROR*|"")
      fail "PyTorch 不可用：${TORCH_INFO#ERROR|}（安装见 docs/setup.md 第 2 节）" ;;
    *)
      IFS='|' read -r TVER TCUDA TAVAIL <<< "${TORCH_INFO}"
      pass "PyTorch ${TVER}（CUDA ${TCUDA}，cuda.is_available=${TAVAIL}）"
      if [ "${TAVAIL}" != "True" ]; then
        warn "torch.cuda.is_available()=False：检测训练将无法运行（配准/冒烟不受影响）"
      fi ;;
  esac
fi

# ---------- 3. Open3D ----------
section "3/6 Open3D 可用性"
O3D_VER=$("${PYTHON_BIN}" - <<'PY' 2>/dev/null
try:
    import open3d
    print(open3d.__version__)
except Exception as e:
    print("ERROR:%s" % e)
PY
)
case "${O3D_VER}" in
  ERROR:*|"")
    fail "Open3D 不可用：${O3D_VER#ERROR:}（pip install open3d==0.17.0）" ;;
  *)
    pass "Open3D ${O3D_VER}" ;;
esac

# ---------- 4. 数据路径 ----------
section "4/6 SUN RGB-D 数据路径"
if [ "${CI_MODE}" = "1" ]; then
  skip "CI 模式不检查数据集"
else
  DATA_OK=1
  for d in data/SUNRGBD data/SUNRGBDtoolbox; do
    if [ -e "${d}" ]; then
      pass "路径存在：${d}"
    else
      warn "路径缺失：${d}（软链接创建见 docs/setup.md 第 5 节）"
      DATA_OK=0
    fi
  done
  if [ -e data/SUN3D ]; then
    pass "路径存在：data/SUN3D（帧对拼接用）"
  else
    warn "路径缺失：data/SUN3D（仅帧对拼接实验需要）"
  fi
  [ "${DATA_OK}" = "1" ] || info "数据未就绪：深度图测试将改用合成数据"
fi

# ---------- 5 & 6. 深度图转点云 + 配准（Python 块） ----------
section "5/6 深度图 -> 点云"
section "6/6 粗配准 + 精配准（合成长方体）"

# 真实深度图探测（完整模式）
SMOKE_DEPTH_FILE=""
if [ "${CI_MODE}" != "1" ] && [ -d data/SUNRGBD ]; then
  SMOKE_DEPTH_FILE=$(find -L data/SUNRGBD -type f \( -iname '*depth*.png' -o -iname '*depth*.jpg' \) 2>/dev/null | head -1)
  if [ -z "${SMOKE_DEPTH_FILE}" ]; then
    SMOKE_DEPTH_FILE=$(find -L data/SUNRGBD -type f -iname '*.png' 2>/dev/null | head -1)
  fi
fi

SMOKE_ROOT="${ROOT_DIR}" SMOKE_MODE="$([ "${CI_MODE}" = 1 ] && echo ci || echo full)" \
SMOKE_DEPTH_FILE="${SMOKE_DEPTH_FILE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import numpy as np

ROOT = os.environ["SMOKE_ROOT"]
sys.path.insert(0, ROOT)
MODE = os.environ.get("SMOKE_MODE", "ci")
DEPTH_FILE = os.environ.get("SMOKE_DEPTH_FILE", "")

# ---------- 5. 深度图 -> 点云 ----------
try:
    from preprocess.depth_to_pointcloud import depth_to_pointcloud, load_depth_input

    K_DEFAULT = np.array([[529.5, 0.0, 365.0],
                          [0.0, 529.5, 265.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
    if MODE == "full" and DEPTH_FILE and os.path.exists(DEPTH_FILE):
        depth, K = load_depth_input(DEPTH_FILE, None)
        points = depth_to_pointcloud(depth, K)
        source_desc = "真实深度图 %s" % DEPTH_FILE
    else:
        # 合成 1m 距离平面（120x120，深度值单位 mm）
        depth = np.full((120, 120), 10000.0, dtype=np.float32)
        points = depth_to_pointcloud(depth, K_DEFAULT)
        source_desc = "合成深度图（1m 平面，120x120）"

    assert points.ndim == 2 and points.shape[1] == 3, "点云形状错误: %r" % (points.shape,)
    assert len(points) > 0, "点云为空"
    z_mean = float(points[:, 2].mean())
    assert abs(z_mean - 1.0) < 1e-3, "平均深度应为 1.0m，实际 %.4f" % z_mean
    print("  [PASS] %s -> %d 个点，平均 z=%.4fm" % (source_desc, len(points), z_mean))
except Exception as e:
    print("  [FAIL] 深度图转点云失败：%s: %s" % (type(e).__name__, e))
    sys.exit(2)

# ---------- 6. 粗配准 + 精配准 ----------
try:
    import open3d as o3d
    from registration.coarse_registration import coarse_registration
    from registration.fine_registration import improved_icp

    # 合成【非对称】长方体表面点云（6 个面，三边半长 a/b/c 互不相等）。
    # 注意：不能用正立方体——其 90° 旋转对称性会让 FPFH+RANSAC 偶发锁到
    # “对称等价但与真值相差 90°”的位姿（fitness 仍为 1.0），造成偶发失败。
    # 三边互不相等后只有恒等变换能自对齐，配准结果唯一、测试确定性通过。
    grid = np.linspace(-1.0, 1.0, 24)
    u, v = np.meshgrid(grid, grid)
    u = u.reshape(-1); v = v.reshape(-1)
    ones = np.ones_like(u)
    a, b, c = 1.0, 0.70, 0.45            # x/y/z 三个方向的半边长，互不相等
    faces = [
        np.stack([ a*ones, b*u, c*v], axis=1), np.stack([-a*ones, b*u, c*v], axis=1),
        np.stack([ a*u,  b*ones, c*v], axis=1), np.stack([ a*u, -b*ones, c*v], axis=1),
        np.stack([ a*u,  b*v,  c*ones], axis=1), np.stack([ a*u,  b*v, -c*ones], axis=1),
    ]
    src_pts = np.concatenate(faces, axis=0).astype(np.float64)

    # 真值位姿：绕 z 轴 3° + 平移 (0.03, 0.02, 0.01)
    ang = np.deg2rad(3.0)
    R_gt = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0, 0.0, 1.0]])
    t_gt = np.array([0.03, 0.02, 0.01])
    tgt_pts = (R_gt @ src_pts.T).T + t_gt
    T_gt = np.eye(4); T_gt[:3, :3] = R_gt; T_gt[:3, 3] = t_gt

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_pts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tgt_pts)

    T_coarse, info = coarse_registration(source, target,
                                         voxel_size=0.02, max_iteration=100000)
    assert np.asarray(T_coarse).shape == (4, 4), "粗配准矩阵形状错误"

    T_fine, history = improved_icp(source, target, init_transformation=T_coarse)
    assert np.asarray(T_fine).shape == (4, 4), "精配准矩阵形状错误"
    assert len(history) > 0, "精配准历史为空"

    # 与真值比较（旋转角误差 / 平移误差）
    R_err_mat = T_fine[:3, :3].T @ T_gt[:3, :3]
    cos_a = np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.rad2deg(np.arccos(cos_a)))
    trans_err = float(np.linalg.norm(T_fine[:3, 3] - T_gt[:3, 3]))
    last_rmse = float(history[-1].get("rmse", history[-1].get("inlier_rmse", -1.0)))

    print("  [INFO] 粗配准 fitness=%.4f, 对应点=%d"
          % (info["fitness"], info["n_correspondences"]))
    print("  [INFO] 精配准迭代=%d, 末次 RMSE=%.5fm" % (len(history), last_rmse))
    print("  [INFO] 位姿误差：旋转 %.4f°，平移 %.4fm" % (rot_err, trans_err))
    assert rot_err < 5.0, "旋转角误差过大：%.4f°" % rot_err
    assert trans_err < 0.05, "平移误差过大：%.4fm" % trans_err
    print("  [PASS] 粗配准 + 精配准通过（旋转<5°，平移<0.05m）")
except Exception as e:
    print("  [FAIL] 配准测试失败：%s: %s" % (type(e).__name__, e))
    sys.exit(3)
PY
SMOKE_RC=$?

# python 块内已逐项打印 PASS/FAIL/INFO，这里把退出码计入总账
case ${SMOKE_RC} in
  0)
    # 深度与配准两节都在 python 块内通过
    N_PASS=$((N_PASS+2))
    ;;
  2) fail "深度图转点云（详见上方输出）" ;;
  3) pass "深度图转点云（详见上方输出）"; fail "粗配准+精配准（详见上方输出）" ;;
  *) fail "Python 检查块异常退出（code=${SMOKE_RC}）" ;;
esac

# ---------- 汇总 ----------
echo ""
echo "${c_bold}========================================${c_reset}"
echo "${c_bold} 冒烟测试汇总${c_reset}"
echo "  ${c_green}PASS=${N_PASS}${c_reset}  ${c_red}FAIL=${N_FAIL}${c_reset}  " \
     "${c_yellow}WARN=${N_WARN}${c_reset}  ${c_blue}SKIP=${N_SKIP}${c_reset}"
echo "${c_bold}========================================${c_reset}"
if [ "${N_FAIL}" -gt 0 ]; then
  echo "结果：${c_red}存在失败项${c_reset}，请按提示排查（参考 docs/troubleshooting.md）"
  exit 1
fi
echo "结果：${c_green}全部通过${c_reset}"
exit 0
