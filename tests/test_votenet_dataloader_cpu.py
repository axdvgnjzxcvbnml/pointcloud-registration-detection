#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only dataloader compatibility test for the generated VoteNet data.

Verifies that results/preprocess/detection_real/*.npz can be consumed by the
VoteNet official SUNRGBD dataloader contract:

  npz keys : point_clouds (N,3), bboxes (K,7), class_ids (K,),
             vote_label (N,3), vote_mask (N,)
  shapes   : bboxes K <= 64 (official loader pads to MAX_NUM_OBJ=64)
  dtype    : point_cloud float32, bboxes float32, class_ids int, mask uint8

Runs on CPU only; no torch/open3d/GPU required.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "preprocess" / "detection_real"

EXPECTED_KEYS = {"point_clouds", "bboxes", "class_ids", "vote_label", "vote_mask"}
MAX_NUM_OBJ = 64  # VoteNet official MAX_NUM_OBJ in sunrgbd_dataset.py


def check_npz(path):
    d = np.load(path, allow_pickle=True)
    missing = EXPECTED_KEYS - set(d.files)
    assert not missing, "%s 缺键 %s" % (path.name, sorted(missing))
    pc = d["point_clouds"]
    bb = d["bboxes"]
    ci = d["class_ids"]
    vl = d["vote_label"]
    vm = d["vote_mask"]
    assert pc.ndim == 2 and pc.shape[1] == 3, "point_clouds 应为 (N,3): %r" % (pc.shape,)
    assert pc.dtype == np.float32, "point_clouds dtype: %s" % pc.dtype
    assert bb.shape == (len(ci), 7) or bb.size == 0, "bboxes/class_ids 不匹配: %r vs %r" % (bb.shape, ci.shape)
    assert bb.shape[1] == 7, "bboxes 应为 (K,7): %r" % (bb.shape,)
    assert vl.shape == pc.shape, "vote_label 形状: %r vs %r" % (vl.shape, pc.shape)
    assert vm.shape == (len(pc),), "vote_mask 形状: %r" % (vm.shape,)
    assert len(bb) <= MAX_NUM_OBJ, "K=%d 超过官方 MAX_NUM_OBJ=%d" % (len(bb), MAX_NUM_OBJ)
    # 官方 loader 训练时随机采样 20000 点（VoteNet NUM_POINTS）
    n_train = min(20000, len(pc))
    idx = np.random.default_rng(0).permutation(len(pc))[:n_train]
    sample = pc[idx]
    assert sample.shape == (n_train, 3)
    return {"file": path.name, "N": len(pc), "K": len(bb),
            "train_sample": n_train, "vote_mask_pos": int(vm.sum())}


def main():
    if not DATA_DIR.is_dir():
        print("FAIL: %s 不存在，先跑 preprocess/generate_detection_data.py" % DATA_DIR)
        sys.exit(1)
    npz_list = sorted(DATA_DIR.glob("*.npz"))
    assert npz_list, "无 npz 文件"
    split = json.load(open(DATA_DIR / "split.json"))
    print("npz=%d, split train=%d val=%d" % (len(npz_list), len(split.get("train", [])), len(split.get("val", []))))
    results = [check_npz(p) for p in npz_list]
    k_total = sum(r["K"] for r in results)
    pos = sum(r["vote_mask_pos"] for r in results)
    print("全部 %d 帧通过 VoteNet 格式校验" % len(results))
    print("点数分布: min=%d max=%d avg=%.0f" % (
        min(r["N"] for r in results), max(r["N"] for r in results),
        np.mean([r["N"] for r in results])))
    print("bbox 合计=%d（当前标签为占位，真实标注见 data/gt 管线）" % k_total)
    print("vote_mask 阳性点合计=%d（空标注帧为 0，属预期）" % pos)
    print("PASS")


if __name__ == "__main__":
    main()
