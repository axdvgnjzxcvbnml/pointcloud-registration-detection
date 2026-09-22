#!/usr/bin/env bash
# ============================================================
# scripts/check_env.sh —— 一键检查环境是否满足 requirements.txt
#
# 用法：
#   bash scripts/check_env.sh           # 全量检查（含 torch / ultralytics）
#   bash scripts/check_env.sh --ci      # CI 模式：跳过 torch/torchvision/
#                                       # ultralytics/tensorboardX（CI 不装重依赖）
#
# 输出：每项 PASS / FAIL / SKIP，最后汇总；任一 FAIL 则退出码 1。
# ============================================================
set -uo pipefail

CI_MODE=0
[[ "${1:-}" == "--ci" ]] && CI_MODE=1

echo "============================================================"
echo " 环境检查（$(date '+%Y-%m-%d %H:%M:%S')）"
echo " 模式：$([ $CI_MODE -eq 1 ] && echo 'CI（跳过重依赖）' || echo '全量')"
echo "============================================================"

CI_MODE=$CI_MODE python3 - <<'PY'
import importlib.metadata as md
import os
import re
import sys

ci = os.environ.get("CI_MODE", "0") == "1"

# (名称, 期望约束, CI 模式是否检查)
# 约束语法：'==' 精确；'>=x,<y' 区间；'>=x' 下限；'8.2.x' 系列；None 只查存在性
CHECKS = [
    ("python",        ">=3.8",            True),
    ("numpy",         "==1.24.4",         True),
    ("scipy",         ">=1.9,<1.11",      True),
    ("open3d",        "==0.17.0",         True),
    ("PyYAML",        ">=6.0",            True),
    ("Pillow",        ">=10.0",           True),
    ("opencv-python-headless", ">=4.8",   True),
    ("plyfile",       ">=0.7.4",          True),
    ("matplotlib",    ">=3.6,<3.8",       True),
    ("tqdm",          ">=4.66",           True),
    ("imageio",       ">=2.31",           True),
    ("torch",         "==1.13.1",         False),
    ("torchvision",   "==0.14.1",         False),
    ("ultralytics",   "8.2.x",            False),
    ("tensorboardX",  ">=2.6",            False),
]

def parse_spec(spec):
    """返回 (op, ver)；'8.2.x' 视为系列：>=8.2.0,<8.3.0"""
    if spec == "8.2.x":
        return ("series", "8.2.0")
    m = re.match(r"^(==|>=|<=|>|<)?\s*(.+)$", spec.strip())
    return (m.group(1) or ">=", m.group(2))

def ver_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)

def ok_version(installed, op, want):
    iv, wv = ver_tuple(installed), ver_tuple(want)
    if op == "==":
        return iv[:2] == wv[:2] and iv[2] == wv[2]
    if op == "series":
        return (wv[0], wv[1], 0) <= iv < (wv[0], wv[1] + 1, 0)
    if op == ">=":
        return iv >= wv
    if op == ">":
        return iv > wv
    if op == "<":
        return iv < wv
    if op == "<=":
        return iv <= wv
    return True

def python_ok():
    v = sys.version_info
    return (v.major, v.minor) >= (3, 8), "%d.%d.%d" % (v.major, v.minor, v.micro)

results = []
for name, spec, check_in_ci in CHECKS:
    if ci and not check_in_ci:
        results.append((name, spec, "SKIP", "CI 模式跳过（重依赖）"))
        continue
    if name == "python":
        ok, cur = python_ok()
        results.append((name, spec, "PASS" if ok else "FAIL",
                        cur + ("（满足 >=3.8）" if ok else "（不满足，需 >=3.8）")))
        continue
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        results.append((name, spec, "FAIL", "未安装"))
        continue
    op, want = parse_spec(spec)
    ok = ok_version(installed, op, want)
    detail = f"已装 {installed}，要求 {spec}"
    if spec == "8.2.x" and not ok:
        detail += "（需 8.2.x 系列）"
    results.append((name, spec, "PASS" if ok else "FAIL", detail))

# 输出
n_pass = n_fail = n_skip = 0
for name, spec, status, detail in results:
    if status == "PASS":
        n_pass += 1
        flag = "  [PASS]"
    elif status == "SKIP":
        n_skip += 1
        flag = "  [SKIP]"
    else:
        n_fail += 1
        flag = "  [FAIL]"
    print(f"{flag} {name:<12} {detail}")

print("------------------------------------------------------------")
print(f" 汇总：PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
print("============================================================")
sys.exit(1 if n_fail else 0)
PY

exit $?
