#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only 单测：用模拟预测 + 模拟真值验证 3D IoU / AP / mAP 计算逻辑。

依赖仅 numpy（detection/evaluate_detection 顶层导入即 numpy）。
不访问数据集、不调用 GPU、不依赖 torch。

验证点：
  1) box3d_iou：完全重叠 = 1.0，完全不重叠 = 0.0，部分重叠数值正确。
  2) compute_mAP@0.25：全部检测命中真值 -> mAP = 1.0。
  3) compute_mAP 带 FP：误检（无对应真值）-> 该类 AP 下降，mAP < 1。
  4) compute_mAP@0.5：跨类别误检（label 错）-> 该检测不算 TP。
  5) load_gt 读写回路：临时 gt.json -> load_gt -> 形状/内容一致。
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection.evaluate_detection import box3d_iou, compute_ap, compute_mAP  # noqa: E402
from detection.load_gt import load_gt, load_gt_indexed, align_gts_by_frames  # noqa: E402

PASS = []


def check(name, cond):
    assert cond, "FAIL: " + name
    PASS.append(name)


def make_box(cx, cy, cz, l=1.0, w=1.0, h=1.0, heading=0.0):
    return np.array([cx, cy, cz, l, w, h, heading], dtype=np.float32)


def test_iou():
    a = make_box(0, 0, 0)
    b = make_box(0, 0, 0)
    check("IoU 完全重叠 == 1.0", abs(box3d_iou(a, b) - 1.0) < 1e-6)
    c = make_box(100, 100, 100)
    check("IoU 完全不重叠 == 0.0", box3d_iou(a, c) == 0.0)
    # 半重叠：x 方向重叠 0.5（中心平移 0.5，l=1 -> 交 0.5），y/z 完全重叠
    d = make_box(0.5, 0, 0, l=1.0, w=1.0, h=1.0)
    iou = box3d_iou(a, d)
    inter = 0.5 * 1.0 * 1.0          # 0.5m^3
    union = 1.0 + 1.0 - inter        # 1.5m^3
    check("IoU 半重叠数值 %.4f == %.4f" % (iou, inter / union),
          abs(iou - inter / union) < 1e-4)


def test_map_all_hit():
    # 两帧：每帧 1 个真值，检测完全命中 -> 全部 TP -> mAP = 1.0
    gt_box = make_box(1, 1, 1)
    gts = [
        {"boxes": gt_box[None], "labels": np.array([0], dtype=np.int32)},
        {"boxes": (make_box(5, 5, 5))[None], "labels": np.array([1], dtype=np.int32)},
    ]
    dets = [
        {"boxes": gt_box[None], "scores": np.array([0.9]), "labels": np.array([0], dtype=np.int32)},
        {"boxes": (make_box(5.01, 5, 5))[None], "scores": np.array([0.8]),
         "labels": np.array([1], dtype=np.int32)},
    ]
    aps25, map25 = compute_mAP(dets, gts, iou_th=0.25, num_classes=2)
    aps50, map50 = compute_mAP(dets, gts, iou_th=0.5, num_classes=2)
    check("mAP@0.25 全命中 == 1.0", abs(map25 - 1.0) < 1e-9)
    check("mAP@0.5 全命中 == 1.0", abs(map50 - 1.0) < 1e-9)
    check("2 类 AP 列表长度 == 2", len(aps25) == 2)
    # 官方口径：mAP 对全部 10 类取平均，无真值/无检测的类 AP=0
    _, map10 = compute_mAP(dets, gts, iou_th=0.25, num_classes=10)
    check("10 类平均（2 类有值）== 0.2", abs(map10 - 0.2) < 1e-9)


def test_map_with_fp():
    # class0 有 2 个真值；检测 = 2 TP + 1 FP（高分 TP 后跟远距误检）
    # VOC 101 点插值下：recall 上限 1.0 处的 precision 被 FP 拉低 -> AP < 1.0
    gts = [{"boxes": np.stack([make_box(0, 0, 0), make_box(3, 3, 3)]),
            "labels": np.array([0, 0], dtype=np.int32)}]
    dets = [{"boxes": np.stack([make_box(0, 0, 0), make_box(50, 50, 50), make_box(3, 3, 3)]),
             "scores": np.array([0.9, 0.7, 0.5]),
             "labels": np.array([0, 0, 0], dtype=np.int32)}]
    aps, m = compute_mAP(dets, gts, iou_th=0.25, num_classes=2)
    check("含 FP 时 mAP < 1.0", m < 1.0)
    check("含 FP 时 class0 AP < 1.0", aps[0] < 1.0)
    check("含 FP 时 class0 AP > 0.5（高分 TP 保底）", aps[0] > 0.5)


def test_map_wrong_label():
    # 检测框位置完全正确但类别标错 -> 按类隔离匹配，不是任何类的 TP
    gts = [{"boxes": (make_box(0, 0, 0))[None], "labels": np.array([0], dtype=np.int32)}]
    dets = [{"boxes": (make_box(0, 0, 0))[None], "scores": np.array([0.9]),
             "labels": np.array([1], dtype=np.int32)}]
    aps, m = compute_mAP(dets, gts, iou_th=0.5, num_classes=2)
    check("类别错 -> class0 AP == 0", aps[0] == 0.0)
    check("类别错 -> mAP < 0.05（class1 无真值，AP 趋近 0）", m < 0.05)


def test_load_gt_roundtrip():
    frames = [
        {"scene": "s1", "frame": "000000",
         "class_names": ["chair"],
         "boxes": [[0.1, 0.2, 0.3, 1.0, 0.5, 0.8, 0.0]],
         "labels": [2]},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"classes": ["bathtub", "bed", "bookshelf", "chair", "desk",
                               "dresser", "night_stand", "sofa", "table", "toilet"],
                   "frames": frames}, f)
        tmp = f.name
    gts = load_gt(tmp)
    check("load_gt 返回 1 帧", len(gts) == 1)
    check("boxes 形状 (1,7)", gts[0]["boxes"].shape == (1, 7))
    check("labels == [2]", gts[0]["labels"].tolist() == [2])
    idx = load_gt_indexed(tmp)
    check("索引键 (s1,000000)", ("s1", "000000") in idx)
    aligned = align_gts_by_frames(idx, [("s1", "000000"), ("s2", "000001")])
    check("对齐后 2 项", len(aligned) == 2)
    check("缺失帧为空真值", aligned[1]["boxes"].shape == (0, 7))
    Path(tmp).unlink(missing_ok=True)


def main():
    test_iou()
    test_map_all_hit()
    test_map_with_fp()
    test_map_wrong_label()
    test_load_gt_roundtrip()
    print("全部 %d 项通过：%s" % (len(PASS), ", ".join(PASS)))
    print("PASS")


if __name__ == "__main__":
    main()
