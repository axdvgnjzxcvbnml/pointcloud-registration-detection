#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_report.py — 一键生成五组消融实验对比报告（V100 上线前预置）

功能
----
    读取 configs/ablation/ 五组配置的结果（evaluate_detection.py 输出的
    report.json / map.json），生成：
      1. Markdown 对比表：mAP@0.25 / mAP@0.5 / 参数量(M) / FLOPs(G) /
         推理速度(ms/帧)；未跑的实验留 TBD 占位；
      2. 自动写入 docs/experiment_log.md 的「消融实验对比」一节
         （<!-- ABLATION-REPORT:START/END --> 标记之间，整节替换）；
      3. matplotlib 柱状图 results/figures/ablation.png（mAP@0.25/0.5
         分组柱状；matplotlib 缺失或全部 TBD 时跳过并提示）。

结果目录约定（与 v100_step7_ablation.sh 一致）：
    results/ablation/<mode>/eval/map.json   （首选）
    results/ablation/<mode>/eval/report.json
    results/ablation/<mode>/map.json
    results/ablation/<mode>/report.json

用法
----
    python scripts/ablation_report.py                 # 默认更新 docs/experiment_log.md
    python scripts/ablation_report.py --log docs/experiment_log.md --no-figure
"""

import argparse
import json
import sys
from pathlib import Path

# 项目根（保证从任意 cwd 运行都能定位结果目录）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 五组消融实验：目录名 -> 表格展示名
EXPERIMENTS = [
    ("00_votenet_baseline", "00 基线（单点云 VoteNet）"),
    ("01_rgb_pseudo3d", "01 单 RGB（伪 3D）"),
    ("02_fusion_concat", "02 融合-Concat（主）"),
    ("03_fusion_attention", "03 融合-Attention"),
    ("04_fusion_lightweight", "04 融合+轻量化"),
]

REPORT_NAMES = ["map.json", "report.json"]


def find_report(mode: str) -> Path | None:
    """在约定目录中查找评测报告 json。"""
    base = ROOT / "results" / "ablation" / mode
    for sub in ("eval", ""):
        for name in REPORT_NAMES:
            p = base / sub / name
            if p.is_file():
                return p
    return None


def load_metrics(mode: str):
    """读取一组实验的指标；无结果返回 None（表格留 TBD）。"""
    report = find_report(mode)
    if report is None:
        return None
    try:
        m = json.loads(report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] {mode} 报告解析失败（{e}），按 TBD 处理: {report}")
        return None
    return {
        "map25": m.get("mAP@0.25"),
        "map50": m.get("mAP@0.5"),
        "params_m": m.get("params_m"),
        "flops_g": m.get("flops_g"),
        "speed_ms": m.get("speed_ms_mean"),
    }


def fmt(v, unit=""):
    """数字 -> 字符串；None 或非数字 -> TBD。"""
    if v is None:
        return "TBD"
    try:
        return f"{float(v):.3f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def build_markdown_table(rows):
    """rows: list[(mode, 展示名, metrics dict | None)] -> Markdown 表格文本。"""
    lines = [
        "| 实验 | mAP@0.25 | mAP@0.5 | 参数量(M) | FLOPs(G) | 推理速度(ms/帧) | 结果文件 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, name, m in rows:
        if m is None:
            lines.append(f"| {name} | TBD | TBD | TBD | TBD | TBD | （未跑） |")
            continue
        src = find_report(mode)
        lines.append(
            f"| {name} | {fmt(m['map25'])} | {fmt(m['map50'])} | "
            f"{fmt(m['params_m'])} | {fmt(m['flops_g'])} | {fmt(m['speed_ms'])} | "
            f"`{src.relative_to(ROOT) if src else 'TBD'}` |")
    return "\n".join(lines)


def update_experiment_log(log_path, table):
    """把表格写入 experiment_log.md 的消融对比节（标记间整节替换，幂等）。

    标题与内容都在标记区内：重复运行不会累积标题。
    标记不存在时新建「1.3 消融实验对比」节，插在「2. 单条实验详细记录」之前。
    """
    log = Path(log_path)
    text = log.read_text(encoding="utf-8")
    start_marker = "<!-- ABLATION-REPORT:START -->"
    end_marker = "<!-- ABLATION-REPORT:END -->"
    block = (
        f"{start_marker}\n"
        "## 1.3 消融实验对比（自动生成，`scripts/ablation_report.py` 维护，勿手改）\n\n"
        f"{table}\n"
        f"{end_marker}\n"
    )
    if start_marker in text and end_marker in text:
        new_text = text[:text.index(start_marker)] + block + \
            text[text.index(end_marker) + len(end_marker):]
    else:
        anchor = "## 2. 单条实验详细记录模板"
        if anchor in text:
            new_text = text.replace(anchor, "\n" + block + "\n" + anchor, 1)
        else:
            new_text = text.rstrip() + "\n\n---\n\n" + block
    log.write_text(new_text, encoding="utf-8")
    print(f"已更新 {log}（消融实验对比节）")


def make_figure(rows, out_png):
    """生成 mAP@0.25/0.5 分组柱状图；无 matplotlib 或全 TBD 时跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib 未安装，跳过柱状图（pip install matplotlib）")
        return
    names, v25, v50 = [], [], []
    for _mode, name, m in rows:
        if m is None or m["map25"] is None or m["map50"] is None:
            continue
        names.append(name.split(" ", 1)[0])
        v25.append(float(m["map25"]))
        v50.append(float(m["map50"]))
    if not names:
        print("  [WARN] 无可用 mAP 数据，跳过柱状图")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    import numpy as np
    x = np.arange(len(names))
    w = 0.38
    b1 = ax.bar(x - w / 2, v25, w, label="mAP@0.25")
    b2 = ax.bar(x + w / 2, v50, w, label="mAP@0.5")
    ax.set_xticks(x, names)
    ax.set_ylabel("mAP")
    ax.set_title("Ablation Study (3D IoU)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2,
                                              b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"已生成柱状图: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="消融实验对比报告生成")
    parser.add_argument("--log", default=str(ROOT / "docs" / "experiment_log.md"),
                        help="experiment_log.md 路径")
    parser.add_argument("--no-update-log", action="store_true",
                        help="不写回 experiment_log.md（只打印表格/出图）")
    parser.add_argument("--no-figure", action="store_true",
                        help="跳过柱状图")
    args = parser.parse_args()

    rows = []
    for mode, name in EXPERIMENTS:
        m = load_metrics(mode)
        rows.append((mode, name, m))
        status = "OK" if m else "TBD"
        print(f"  [{status}] {name}: " + (json.dumps(m) if m else "未跑"))

    table = build_markdown_table(rows)
    print("\n=== 消融实验对比表 ===")
    print(table)

    if not args.no_update_log:
        update_experiment_log(args.log, table)
    if not args.no_figure:
        make_figure(rows, ROOT / "results" / "figures" / "ablation.png")

    # 机器可读输出（CI/脚本可解析）
    (ROOT / "results" / "ablation" / "ablation_report.md").parent.mkdir(
        parents=True, exist_ok=True)
    (ROOT / "results" / "ablation" / "ablation_report.md").write_text(
        table + "\n", encoding="utf-8")
    print("\nMarkdown 表 -> results/ablation/ablation_report.md")


if __name__ == "__main__":
    main()
