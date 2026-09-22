#!/usr/bin/env bash
# ============================================================
# scripts/check_data.sh —— SUN RGB-D 数据完整性检查（V100 上线前）
# ------------------------------------------------------------
# 检查项：
#   1. data/SUNRGBD 是否存在且可访问（软链有效）
#   2. 场景根（xtion/sun3ddata/，兼容 kv1/kv2/xtion 直接布局）下是否有场景序列
#   3. 每个场景是否包含 image/、depth/、extrinsics/ 三个目录
#   4. 每个目录下的文件数量是否一致（image == depth 必须相等；
#      extrinsics 允许少于帧数，但必须存在且非空）
#   5. 是否有空文件 / 损坏文件（PNG magic 校验 + 抽样解码）
#
# 用法：
#   bash scripts/check_data.sh                       # 检查 data/SUNRGBD
#   bash scripts/check_data.sh /abs/path/SUNRGBD     # 检查指定路径
#
# 退出码：0 = 全部通过；1 = 存在失败项
# 说明：脚本用 find -L 跟随软链；场景目录不要求固定的 x 层深度，
#       只要某目录同时包含 image/ 与 depth/ 即视为一个场景。
# ============================================================

set -u

SUNRGBD_DIR="${1:-data/SUNRGBD}"

# ---------- 计数与输出 ----------
N_PASS=0; N_FAIL=0; N_WARN=0
c_red=$'\033[31m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_blue=$'\033[34m'
c_bold=$'\033[1m'; c_reset=$'\033[0m'
pass() { echo "  ${c_green}[PASS]${c_reset} $1"; N_PASS=$((N_PASS+1)); }
fail() { echo "  ${c_red}[FAIL]${c_reset} $1"; N_FAIL=$((N_FAIL+1)); }
warn() { echo "  ${c_yellow}[WARN]${c_reset} $1"; N_WARN=$((N_WARN+1)); }
info() { echo "  ${c_blue}[INFO]${c_reset} $1"; }
section() { echo ""; echo "${c_bold}== $1 ==${c_reset}"; }

echo "${c_bold}========================================${c_reset}"
echo "${c_bold} SUN RGB-D 数据完整性检查${c_reset}"
echo " 目标路径：${SUNRGBD_DIR}"
echo "${c_bold}========================================${c_reset}"

# ---------- 1. 根路径 ----------
section "1/5 根路径可访问性"
if [ ! -e "${SUNRGBD_DIR}" ]; then
  fail "路径不存在：${SUNRGBD_DIR}"
  echo ""
  echo "修复建议："
  echo "  mkdir -p data"
  echo "  ln -s /真实数据路径/SUNRGBD        data/SUNRGBD"
  echo "  ln -s /真实数据路径/SUNRGBDtoolbox data/SUNRGBDtoolbox"
  echo "  ln -s /真实数据路径/SUN3D          data/SUN3D"
  echo "（软链必须用绝对路径；详见 docs/setup.md 第 5 节）"
  echo ""
  echo "结果：${c_red}失败（根路径不存在，后续检查跳过）${c_reset}"
  exit 1
fi
if [ -L "${SUNRGBD_DIR}" ] && [ ! -d "${SUNRGBD_DIR}" ]; then
  fail "软链已断：${SUNRGBD_DIR} -> $(readlink "${SUNRGBD_DIR}")"
  echo "修复建议：重新 ln -s 指向真实数据路径（docs/setup.md 第 5 节）"
  exit 1
fi
if [ ! -d "${SUNRGBD_DIR}" ]; then
  fail "不是目录：${SUNRGBD_DIR}"
  exit 1
fi
pass "根路径可访问：${SUNRGBD_DIR}"

# ---------- 2. 场景定位 ----------
section "2/5 场景序列定位"
# 收集"同时含 image/ 与 depth/ 的目录"作为场景目录（find -L 跟随软链）
SCENES="$(find -L "${SUNRGBD_DIR}" -type d -name image 2>/dev/null \
  | while read -r imgdir; do
      parent="$(dirname "${imgdir}")"
      [ -d "${parent}/depth" ] && echo "${parent}"
    done | sort -u)"
if [ -z "${SCENES}" ]; then
  fail "未找到任何场景（需要目录同时包含 image/ 与 depth/）"
  echo ""
  echo "期望布局（以 xtion/sun3ddata 为例）："
  echo "  ${SUNRGBD_DIR}/xtion/sun3ddata/<scene>/"
  echo "      ├── image/        # RGB 图（jpg/png）"
  echo "      ├── depth/        # 深度图（16bit png）"
  echo "      └── extrinsics/   # 3x4 相机位姿 txt"
  echo "修复建议：确认软链指向的是 SUNRGBD 数据根（含 kv1/kv2/xtion 子目录），"
  echo "         或检查数据是否解压完整（官方 zip 解压后应有上述布局）。"
  echo ""
  echo "结果：${c_red}失败（无场景，后续检查跳过）${c_reset}"
  exit 1
fi
N_SCENES=$(echo "${SCENES}" | grep -c .)
pass "找到 ${N_SCENES} 个场景"
echo "${SCENES}" | head -5 | sed 's/^/    - /'

# ---------- 3. 场景三件套目录 ----------
section "3/5 场景目录结构（image / depth / extrinsics）"
MISSING_ANY=0
while read -r scene; do
  [ -z "${scene}" ] && continue
  name="$(basename "${scene}")"
  for sub in image depth extrinsics; do
    if [ -d "${scene}/${sub}" ]; then
      pass "${name}/${sub}/ 存在"
    else
      fail "${name}/${sub}/ 缺失（路径：${scene}/${sub}）"
      MISSING_ANY=1
    fi
  done
done <<< "${SCENES}"

# ---------- 4. 文件数量一致性 ----------
section "4/5 文件数量一致性（image vs depth，extrinsics 至少存在）"
COUNT_BAD=0
while read -r scene; do
  [ -z "${scene}" ] && continue
  name="$(basename "${scene}")"
  n_img=$(find -L "${scene}/image" -type f 2>/dev/null | wc -l)
  n_dep=$(find -L "${scene}/depth" -type f 2>/dev/null | wc -l)
  n_ext=$(find -L "${scene}/extrinsics" -type f 2>/dev/null | wc -l)
  if [ "${n_img}" -eq 0 ] && [ "${n_dep}" -eq 0 ]; then
    fail "${name}：image/depth 均为 0 个文件"
    COUNT_BAD=1
    continue
  fi
  if [ "${n_img}" -ne "${n_dep}" ]; then
    fail "${name}：image=${n_img} ≠ depth=${n_dep}"
    COUNT_BAD=1
  else
    pass "${name}：image == depth == ${n_img}"
  fi
  if [ "${n_ext}" -eq 0 ]; then
    fail "${name}：extrinsics 为空（0 个位姿文件，配准真值将无法生成）"
    COUNT_BAD=1
  elif [ "${n_ext}" -lt "${n_img}" ]; then
    warn "${name}：extrinsics=${n_ext} < image=${n_img}（部分帧无位姿，compute_pose_gt 会跳过缺位姿的帧对）"
  fi
done <<< "${SCENES}"

# ---------- 5. 空文件 / 损坏文件 ----------
section "5/5 空文件与损坏文件"
BROKEN_ANY=0

check_empty_and_magic() {  # $1=目录 $2=类型名
  local dir="$1" label="$2"
  # 空文件（0 字节）
  local empty
  empty="$(find -L "${dir}" -type f -size 0 2>/dev/null | head -5)"
  if [ -n "${empty}" ]; then
    fail "${label}：存在空文件（0 字节）"
    echo "${empty}" | head -3 | sed 's/^/        - /'
    BROKEN_ANY=1
  fi
  # PNG magic 校验（SUN RGB-D 深度图为 PNG；RGB 也可能是 jpg/png）
  local bad_png
  bad_png="$(find -L "${dir}" -type f \( -iname '*.png' \) 2>/dev/null \
    | while read -r f; do
        magic="$(head -c 8 "${f}" 2>/dev/null | od -An -tx1 | tr -d ' \n')"
        [ "${magic}" != "89504e470d0a1a0a" ] && echo "${f}"
      done | head -5)"
  if [ -n "${bad_png}" ]; then
    fail "${label}：存在非 PNG 魔数的 .png 文件（可能损坏或格式错误）"
    echo "${bad_png}" | head -3 | sed 's/^/        - /'
    BROKEN_ANY=1
  fi
}

while read -r scene; do
  [ -z "${scene}" ] && continue
  name="$(basename "${scene}")"
  [ -d "${scene}/image" ]      && check_empty_and_magic "${scene}/image"      "${name}/image"
  [ -d "${scene}/depth" ]      && check_empty_and_magic "${scene}/depth"      "${name}/depth"
  [ -d "${scene}/extrinsics" ] && check_empty_and_magic "${scene}/extrinsics" "${name}/extrinsics"
done <<< "${SCENES}"

# 抽样解码校验（可选：python3 + PIL 可用时，每目录抽 3 个文件解码）
if command -v python3 >/dev/null 2>&1 && python3 -c "import PIL" 2>/dev/null; then
  info "抽样解码校验（PIL 可用，每目录 3 个文件）"
  while read -r scene; do
    [ -z "${scene}" ] && continue
    name="$(basename "${scene}")"
    for sub in image depth; do
      [ -d "${scene}/${sub}" ] || continue
      SAMPLE="$(find -L "${scene}/${sub}" -type f 2>/dev/null | head -3)"
      [ -z "${SAMPLE}" ] && continue
      BAD="$(echo "${SAMPLE}" | python3 -c "
import sys
try:
    from PIL import Image
except Exception:
    sys.exit(0)
bad = []
for line in sys.stdin:
    p = line.strip()
    if not p:
        continue
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception:
        bad.append(p)
if bad:
    print('\n'.join(bad[:3]))
" 2>/dev/null)"
      if [ -n "${BAD}" ]; then
        fail "${name}/${sub}：抽样解码失败（可能损坏）"
        echo "${BAD}" | head -3 | sed 's/^/        - /'
        BROKEN_ANY=1
      fi
    done
  done <<< "${SCENES}"
else
  info "python3/PIL 不可用，跳过抽样解码校验（仅做魔数校验）"
fi

# ---------- 汇总 ----------
echo ""
echo "${c_bold}========================================${c_reset}"
echo "${c_bold} 检查汇总${c_reset}"
echo "  ${c_green}PASS=${N_PASS}${c_reset}  ${c_red}FAIL=${N_FAIL}${c_reset}  ${c_yellow}WARN=${N_WARN}${c_reset}  （场景数 ${N_SCENES}）"
echo "${c_bold}========================================${c_reset}"
if [ "${N_FAIL}" -gt 0 ]; then
  echo "结果：${c_red}存在失败项${c_reset}，请按上方提示修复后重跑；"
  echo "      数据准备细节见 docs/setup.md 第 5 节，报错排查见 docs/troubleshooting.md 第 3 节。"
  exit 1
fi
echo "结果：${c_green}全部通过${c_reset}，可以继续预处理（bash scripts/v100_step2_preprocess.sh）。"
exit 0
