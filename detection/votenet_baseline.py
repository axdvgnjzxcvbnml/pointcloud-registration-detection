#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
votenet_baseline.py — VoteNet 官方基线推理（第四批交付）

功能
----
    加载 VoteNet 官方预训练权重（SUN RGB-D 10 类），在验证集上推理，
    输出 mAP@0.25 与 mAP@0.5（3D IoU 口径，10 类及均值）。

依赖
----
    - external/votenet（facebookresearch/votenet 克隆，需编译 PointNet2 算子）
    - 官方权重：weights/votenet_sunrgbd.pth
      （VoteNet README 提供下载；.tar 格式内部含 'state_dict' 也可加载）

说明
----
    本脚本不产生任何“需要运行才能得到”的数值；
    运行后请将 mAP 回填 README 实验结果表。

用法
----
    python detection/votenet_baseline.py \
        --ckpt weights/votenet_sunrgbd.pth \
        --det_data_dir results/preprocess/detection \
        --split results/preprocess/detection/split.json \
        --out_dir results/detection/baseline
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_votenet_model(ckpt_path, num_class=10, device="cuda"):
    """加载 VoteNet 官方预训练模型。

    兼容三种 checkpoint 形态：
      - 裸 state_dict（key 可能带 'module.' 前缀）
      - 字典（含 'state_dict' / 'model' 键，VoteNet 官方 .tar）
    """
    import torch
    from train_fusion import build_votenet_model  # 与训练共用 VoteNet 构建

    model = build_votenet_model(num_class=num_class)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # 去掉 'module.' 前缀（DataParallel 保存）
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("缺失权重 %d 个: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("多余权重 %d 个（可能因版本差异）", len(unexpected))

    model.to(device).eval()
    logger.info("VoteNet 加载完成: %s", ckpt_path)
    return model


def run_inference(model, data_loader, device="cuda"):
    """在验证集上推理，返回逐帧检测结果。

    每帧输出：
        boxes (M,7) [cx,cy,cz,l,w,h,heading]
        scores (M,)
        labels (M,)
    VoteNet 官方 forward 返回 data_dict，含 'batch_boxes'/'batch_scores'/
    'batch_objectness' 等键，具体键名以 external/votenet 版本为准（TODO 核对）。
    """
    import torch

    detections = []
    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            data_dict = model(batch)
            boxes = data_dict.get("batch_boxes", data_dict.get("boxes"))
            scores = data_dict.get("batch_scores", data_dict.get("scores"))
            labels = data_dict.get("batch_labels", data_dict.get("labels"))
            if boxes is None or scores is None or labels is None:
                raise KeyError(
                    "模型输出缺少 batch_boxes/batch_scores/batch_labels 字段，"
                    "请核对 external/votenet 输出键名（data_dict 现有键: "
                    f"{list(data_dict.keys())}）")
            for b, s, l in zip(boxes, scores, labels):
                detections.append({
                    "boxes": b.cpu().numpy(),
                    "scores": s.cpu().numpy(),
                    "labels": l.cpu().numpy().astype(int),
                })
    return detections


def main():
    parser = argparse.ArgumentParser(description="VoteNet 官方基线推理与评测")
    parser.add_argument("--ckpt", default="weights/votenet_sunrgbd.pth",
                        help="VoteNet 官方预训练权重路径")
    parser.add_argument("--det_data_dir", default="results/preprocess/detection",
                        help="检测数据目录（generate_detection_data.py 输出）")
    parser.add_argument("--split", default="results/preprocess/detection/split.json",
                        help="train/val 划分清单")
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default="results/detection/baseline",
                        help="评测输出目录")
    parser.add_argument("--device", default="cuda",
                        help="cpu / cuda:0")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import torch
    from torch.utils.data import DataLoader

    # 数据加载：复用 train_fusion.py 的 SunRGBDDataset（同为检测数据格式）
    from train_fusion import SunRGBDDataset

    with open(args.split, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_names = [s["name"] for s in split["val"]]

    dataset = SunRGBDDataset(det_data_dir=args.det_data_dir,
                             sample_names=val_names)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    model = load_votenet_model(args.ckpt, num_class=args.num_class,
                               device=args.device)
    detections = run_inference(model, loader, device=args.device)

    # 评测：复用 evaluate_detection.py 的 mAP 计算（3D IoU 口径）
    from evaluate_detection import compute_mAP, save_eval_report

    # ================================================================
    # TODO（真值接入，V100 上跑真实数据时完成；当前为骨架占位）
    # ----------------------------------------------------------------
    # 1) 真值来源：
    #    generate_detection_data.py 为每帧导出一个 npz（位于
    #    results/preprocess/detection/<frame_name>.npz），其中：
    #      - "bboxes"    : (K, 7)，K 个真值框，每行
    #                      [cx, cy, cz, length, width, height, heading]
    #      - "class_ids" : (K,)，类别 id（0~9，类别表见
    #                      evaluate_detection.SUNRGBD_CLASSES）
    #    验证集帧名来自 split.json 的 split["val"]（元素含 "name"）。
    #
    # 2) 真值加载（每帧一个 dict，顺序与 run_inference 输出一致）：
    #      gts = []
    #      for name in val_names:
    #          d = np.load(os.path.join(args.det_data_dir, name + ".npz"))
    #          gts.append({"boxes": d["bboxes"], "labels": d["class_ids"]})
    #
    # 3) 预测对齐：
    #    run_inference 已按 DataLoader 顺序（shuffle=False）逐帧输出
    #    {"boxes"(M,7), "scores"(M,), "labels"(M,)}，与 gts 逐帧一一对应；
    #    注意预测框为 (cx,cy,cz,l,w,h,heading)，与真值 7 字段口径相同，
    #    可直接送入 compute_mAP（内部 boxes_to_corners_3d 统一转 8 角点）。
    #
    # 4) 计算与落盘：
    #      aps_025, mAP_025 = compute_mAP(detections, gts, 0.25, args.num_class)
    #      aps_05,  mAP_05  = compute_mAP(detections, gts, 0.5,  args.num_class)
    #      save_eval_report(args.out_dir, {
    #          "mAP@0.25": mAP_025, "mAP@0.5": mAP_05,
    #          "AP@0.25": aps_025, "AP@0.5": aps_05})
    # ================================================================
    raise NotImplementedError(
        "真值加载尚未接入：请按上方 TODO 从检测数据 npz 读取 "
        "bboxes/class_ids 构建 gts，再调用 compute_mAP；"
        "参考 evaluate_detection.py 的 main()。")


if __name__ == "__main__":
    main()
