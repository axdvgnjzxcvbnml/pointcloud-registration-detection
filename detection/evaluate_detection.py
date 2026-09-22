#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_detection.py — 检测评测（第四批交付）

功能
----
    对训练好的融合/轻量化模型在验证集上评测：
      - mAP@0.25 与 mAP@0.5（3D IoU 口径，10 类及均值）
      - 每类 AP 明细
      - 参数量 / FLOPs（可选 thop）/ 推理速度（ms/帧）
    输出 report.json + 终端表格。

说明
----
    box3d_iou / compute_ap / compute_mAP 为纯 numpy 实现（可单测），
    torch 相关函数在目标环境（V100）运行。

用法
----
    python detection/evaluate_detection.py \
        --config configs/default.yaml \
        --ckpt results/detection/fusion/fusion_epoch060.pth \
        --det_data_dir results/preprocess/detection \
        --out_dir results/detection/eval
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 3D IoU（带朝向的矩形棱柱，纯 numpy）
# ---------------------------------------------------------------
def _poly_corners(cx, cy, l, w, heading):
    """xy 平面旋转矩形四角（heading 绕 z 轴）。"""
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    half = np.array([[-l / 2, -w / 2], [l / 2, -w / 2],
                     [l / 2, w / 2], [-l / 2, w / 2]], dtype=np.float64)
    return (R @ half.T).T + np.array([cx, cy])


def _seg_inter(p1, p2, p3, p4):
    """线段 p1p2 与 p3p4 交点（参数化求解）。"""
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return (p1 + p2) / 2.0
    t = ((p3 - p1)[0] * d2[1] - (p3 - p1)[1] * d2[0]) / denom
    return p1 + t * d1


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman 多边形裁剪（subject 被 clip 裁剪）。"""
    out = subject
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        edge = b - a

        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-9

        inp, out = out, []
        if len(inp) == 0:
            break
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(_seg_inter(s, e, a, b))
                out.append(e)
            elif inside(s):
                out.append(_seg_inter(s, e, a, b))
            s = e
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box3d_iou(box_a, box_b):
    """两个 3D 框的 IoU（带朝向，xy 旋转多边形 × z 高度重叠）。

    box: [cx, cy, cz, l, w, h, heading]
    """
    cx1, cy1, cz1, l1, w1, h1, hd1 = box_a
    cx2, cy2, cz2, l2, w2, h2, hd2 = box_b

    inter_poly = _clip_polygon(_poly_corners(cx1, cy1, l1, w1, hd1),
                               _poly_corners(cx2, cy2, l2, w2, hd2))
    inter2d = _poly_area(inter_poly)
    z_overlap = max(0.0, min(cz1 + h1 / 2, cz2 + h2 / 2)
                    - max(cz1 - h1 / 2, cz2 - h2 / 2))
    inter = inter2d * z_overlap
    vol1, vol2 = l1 * w1 * h1, l2 * w2 * h2
    union = vol1 + vol2 - inter
    return float(inter / max(union, 1e-12))


# ---------------------------------------------------------------
# AP / mAP（VOC 风格，101 点插值）
# ---------------------------------------------------------------
def compute_ap(det_scores, det_tp, npos):
    """计算单类 AP（101 点插值）。

    Args:
        det_scores: 该类的检测分数（按任意顺序）。
        det_tp: 与 det_scores 一一对应的 TP 标记（1/0）。
        npos: 该类真值框总数。
    """
    det_scores = np.asarray(det_scores, dtype=np.float64)
    det_tp = np.asarray(det_tp, dtype=np.int32)
    if len(det_scores) == 0:
        return 0.0

    order = np.argsort(-det_scores)
    tp = np.cumsum(det_tp[order])
    fp = np.cumsum(1 - det_tp[order])
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, 1e-12)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        m = prec[rec >= t]
        ap += (m.max() if m.size else 0.0) / 101.0
    return float(ap)


def compute_mAP(detections, gts, iou_th=0.25, num_classes=10):
    """计算 mAP（贪心匹配：每个真值框最多匹配一个检测）。

    Args:
        detections: list[dict]，每帧 {"boxes" (M,7), "scores" (M,), "labels" (M,)}
        gts: list[dict]，每帧 {"boxes" (K,7), "labels" (K,)}
        iou_th: 3D IoU 阈值（0.25 或 0.5）。
        num_classes: 类别数。
    Returns:
        (aps list[float] 每类, mAP float)
    """
    aps = []
    for cls in range(num_classes):
        det_scores, det_tp, npos = [], [], 0
        for det, gt in zip(detections, gts):
            d = det
            m = d["labels"] == cls
            g = gt
            gm = g["labels"] == cls
            npos += int(gm.sum())

            if m.sum() == 0:
                continue
            boxes, scores = d["boxes"][m], d["scores"][m]
            gt_boxes = g["boxes"][gm]

            matched = set()
            order = np.argsort(-scores)
            for i in order:
                iou_max, j_best = 0.0, -1
                for j in range(len(gt_boxes)):
                    if j in matched:
                        continue
                    iou = box3d_iou(boxes[i], gt_boxes[j])
                    if iou > iou_max:
                        iou_max, j_best = iou, j
                if iou_max >= iou_th:
                    matched.add(j_best)
                    det_tp.append(1)
                else:
                    det_tp.append(0)
                det_scores.append(float(scores[i]))

        aps.append(compute_ap(det_scores, det_tp, npos))

    return aps, float(np.mean(aps))


# ---------------------------------------------------------------
# 模型效率指标（torch，需在目标环境运行）
# ---------------------------------------------------------------
def count_parameters(model):
    """可训练参数量。"""
    params = model.trainable_parameters() if hasattr(model, "trainable_parameters") \
        else list(model.parameters())
    return sum(p.numel() for p in params)


def estimate_flops(model, sample, device="cuda"):
    """用 thop 估算 FLOPs（未安装 thop 时返回 None 并告警）。"""
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(sample,), verbose=False)
        return int(flops)
    except ImportError:
        logger.warning("未安装 thop，跳过 FLOPs 统计（pip install thop）")
        return None


def measure_speed(model, loader, warmup=10, iters=50, device="cuda"):
    """推理速度（ms/帧，含 warmup）。返回 (mean_ms, std_ms)。"""
    import torch

    model.eval()
    # 预热（CUDA kernel 初始化）
    for _ in range(warmup):
        batch = next(iter(loader))
        with torch.no_grad():
            _ = model(batch, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for i, batch in enumerate(loader):
        if i >= iters:
            break
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    if not times:
        return 0.0, 0.0
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------
# 输出
# ---------------------------------------------------------------
def save_eval_report(out_dir, report):
    """保存评测报告 json 并打印终端表格。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== 检测评测报告 ==========")
    if "AP@0.25" in report:
        print(f"{'类别':<14}{'AP@0.25':>10}{'AP@0.5':>10}")
        for cls, (a1, a2) in enumerate(zip(report["AP@0.25"], report["AP@0.5"])):
            print(f"{cls:<14}{a1:>10.3f}{a2:>10.3f}")
        print(f"{'mAP':<14}{report['mAP@0.25']:>10.3f}{report['mAP@0.5']:>10.3f}")
    if "params_m" in report:
        print(f"参数量: {report['params_m']:.3f} M | FLOPs: {report.get('flops_g', 'N/A')} G"
              f" | 推理: {report.get('speed_ms_mean', 'N/A')} ms/帧")
    print("===================================")
    logger.info("报告 -> %s", out_dir / "report.json")


def main():
    parser = argparse.ArgumentParser(description="检测评测（mAP/参数量/FLOPs/速度）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True, help="模型权重（fusion/lightweight）")
    parser.add_argument("--det_data_dir", default="results/preprocess/detection")
    parser.add_argument("--frames_dir", default="results/preprocess/frames")
    parser.add_argument("--out_dir", default="results/detection/eval")
    parser.add_argument("--votenet_ckpt", default="weights/votenet_sunrgbd.pth")
    parser.add_argument("--yolov8_ckpt", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iou_thresholds", default="0.25,0.5",
                        help="逗号分隔的 IoU 阈值")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--test_iters", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import torch
    from torch.utils.data import DataLoader
    from train_fusion import FusionModel, SunRGBDDataset, load_config
    from finetune_lightweight import build_lightweight_model

    cfg = load_config(args.config)

    # 判断是否为轻量化权重（含 lightweight 键或结构匹配）
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    is_lightweight = any("dsconv" in k or "depthwise" in k
                         for k in state.get("fusion_head", {})) \
        if isinstance(state, dict) else False

    if is_lightweight:
        model = build_lightweight_model(cfg, args.votenet_ckpt, args.yolov8_ckpt,
                                        freeze_backbone=True)
    else:
        model = FusionModel(cfg, args.votenet_ckpt, args.yolov8_ckpt,
                            freeze_backbone=True)
    # 加载权重（结构一致时全量加载；轻量化头部键自动忽略）
    model.votebackbone.load_state_dict(state.get("votebackbone", {}), strict=False)
    model.yolo_feat.proj_conv.load_state_dict(state.get("yolo_proj_conv", {}),
                                              strict=False)
    model.fusion_head.load_state_dict(state.get("fusion_head", {}), strict=False)
    model.det_head.load_state_dict(state.get("det_head", {}), strict=False)
    model = model.to(args.device)
    model.eval()

    with open(Path(args.det_data_dir) / "split.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    val_names = [s["name"] for s in split["val"]]
    dataset = SunRGBDDataset(args.det_data_dir, args.frames_dir, val_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4)

    # 推理得到检测结果
    detections, gts = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch, device=args.device)
            # 检测框生成：当前为占位实现（每个种子点第 1 个投票，未做 NMS）。
            # TODO: 接入投票聚合 + 3D NMS（见 _placeholder_pred_to_detections 注释）。
            det_batch = _placeholder_pred_to_detections(pred, batch)
            detections.extend(det_batch)
            for i in range(len(batch["point_cloud"])):
                gts.append({
                    "boxes": batch["bboxes"][i].numpy(),
                    "labels": batch["class_ids"][i].numpy(),
                })

    # mAP
    report = {}
    for th_str in args.iou_thresholds.split(","):
        th = float(th_str)
        aps, mAP = compute_mAP(detections, gts, iou_th=th,
                               num_classes=cfg["detection"]["detect_head"]["num_classes"])
        report[f"AP@{th}"] = [round(a, 4) for a in aps]
        report[f"mAP@{th}"] = round(mAP, 4)

    # 效率指标
    report["params_m"] = round(count_parameters(model) / 1e6, 3)
    sample = next(iter(loader))
    flops = estimate_flops(model, sample, device=args.device)
    report["flops_g"] = round(flops / 1e9, 3) if flops else None
    mean_ms, std_ms = measure_speed(model, loader, warmup=args.warmup,
                                    iters=args.test_iters, device=args.device)
    report["speed_ms_mean"] = round(mean_ms, 2)
    report["speed_ms_std"] = round(std_ms, 2)
    report["config"] = args.config

    save_eval_report(args.out_dir, report)


def _placeholder_pred_to_detections(pred, batch):
    """占位实现：把每个种子点第 1 个投票转成检测（未做 NMS）。

    TODO：替换为 generate_proposals.py 的投票聚合 + 3D NMS，
    当前仅保证评测管线可跑通。
    """
    import torch
    B, N = pred["objectness"].shape[:2]
    outs = []
    center = pred["center"][:, :, 0, :]                 # (B,N,3)
    size = pred["size"][:, :, 0, :].sigmoid() * 2.0     # (B,N,3) 归一化到 [0,2]
    heading_sin = pred["heading"][:, :, 0, 0]
    heading_cos = pred["heading"][:, :, 0, 1]
    heading = torch.atan2(heading_sin, heading_cos)     # (B,N)
    scores = pred["objectness"].squeeze(-1).sigmoid()   # (B,N)
    labels = pred["class_scores"].argmax(-1)            # (B,N)
    base = batch["point_cloud"]
    for b in range(B):
        centers = base[b] + center[b]
        boxes = torch.cat([centers, size[b],
                           heading[b].unsqueeze(-1)], dim=-1)  # (N,7)
        outs.append({
            "boxes": boxes.detach().cpu().numpy(),
            "scores": scores[b].detach().cpu().numpy(),
            "labels": labels[b].detach().cpu().numpy(),
        })
    return outs


if __name__ == "__main__":
    main()
