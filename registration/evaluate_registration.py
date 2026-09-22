#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_registration.py — 配准结果评测（第三批交付，纯 CPU）

功能
----
    对每个帧对执行完整配准流水线并评测：
      方法：改进 ICP（点到平面+自适应阈值）  [主方法]
      基线：点到点 ICP（传统） / FGR（Fast Global Registration）
      指标：
        - RMSE           对齐后最近邻点距的均方根（米）
        - 旋转角误差      估计与真值旋转矩阵的夹角（度）
        - 平移误差        估计与真值平移向量差值的模（米）
        - 配准成功率      三个指标均在阈值内视为成功
    输出逐对明细 + 汇总统计（json / csv / 终端表格）。

输入
----
    --pairs      帧对清单（preprocess/sample_frame_pairs.py 输出）
    --pcd_dir    每帧点云目录（ply，命名 {scene}_{frame:06d}.ply）
    --pose_gt_dir 6DOF 真值目录（preprocess/compute_pose_gt.py 输出）

用法
----
    python registration/evaluate_registration.py \
        --pairs results/preprocess/pairs/pairs.json \
        --pcd_dir results/preprocess/pcd \
        --pose_gt_dir results/preprocess/pose_gt \
        --out_dir results/registration/eval
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 评测阈值（与 configs/default.yaml 的 registration.eval_* 一致）
DEFAULT_RMSE_TH = 0.05
DEFAULT_ROT_TH = 5.0
DEFAULT_TRANS_TH = 0.05


# ---------------------------------------------------------------
# 指标计算（纯 numpy，不依赖 open3d，便于单测）
# ---------------------------------------------------------------
def rotation_error(T_est, T_gt):
    """旋转角误差（度）：R_est 与 R_gt 的夹角。"""
    R_est, R_gt = T_est[:3, :3], T_gt[:3, :3]
    cos_ang = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_ang)))


def translation_error(T_est, T_gt):
    """平移误差（米）：||t_est - t_gt||。"""
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def correspondence_rmse(src_pts, tgt_pts, T, max_dist=0.05):
    """对齐后最近邻点距的 RMSE（米）。

    对每个变换后的源点找最近目标点，取距离 < max_dist 的内点
    计算均方根误差。无内点时返回 (inf, 0.0)。
    Returns:
        (rmse, inlier_ratio)
    """
    from scipy.spatial import cKDTree

    src_aligned = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
    tree = cKDTree(tgt_pts)
    dist, _ = tree.query(src_aligned, k=1, workers=-1)
    inliers = dist[dist < max_dist]
    if len(inliers) == 0:
        return float("inf"), 0.0
    rmse = float(np.sqrt(np.mean(inliers ** 2)))
    return rmse, float(len(inliers) / max(len(dist), 1))


def is_success(metrics, rmse_th=DEFAULT_RMSE_TH, rot_th=DEFAULT_ROT_TH,
               trans_th=DEFAULT_TRANS_TH):
    """是否判定为配准成功（三个指标均需达标）。"""
    return (metrics["rmse"] < rmse_th
            and metrics["rot_deg"] < rot_th
            and metrics["trans_m"] < trans_th)


# ---------------------------------------------------------------
# 配准流水线
# ---------------------------------------------------------------
def load_pcd_points(pcd_path):
    """读取点云为 numpy (N,3)（调用 Open3D）。"""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise RuntimeError(f"点云为空: {pcd_path}")
    return pcd


def pcd_path_for(pcd_dir, scene, frame_idx):
    """按命名约定定位帧点云路径（尝试多种命名）。"""
    pcd_dir = Path(pcd_dir)
    candidates = [
        pcd_dir / f"{scene}_{frame_idx:06d}.ply",
        pcd_dir / f"{scene}_frame-{frame_idx:06d}.ply",
        pcd_dir / f"{scene}_{frame_idx:06d}.npy",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    # 兜底：按文件名模糊匹配（含 scene 且含帧号）
    for p in sorted(pcd_dir.glob(f"{scene}*.ply")):
        if str(frame_idx) in p.stem:
            return p
    raise FileNotFoundError(f"未找到帧点云: scene={scene}, frame={frame_idx} in {pcd_dir}")


def run_methods(src_pcd, tgt_pcd, T_gt, params):
    """对一对点云运行所有方法，返回各方法指标字典。"""
    import open3d as o3d

    from coarse_registration import coarse_registration
    from fine_registration import improved_icp

    voxel = params["voxel_size"]
    # 预处理（去噪 + 降采样 + 法向量，统一各方法输入）
    from preprocess_pointcloud import preprocess_pointcloud
    src = preprocess_pointcloud(src_pcd, voxel_size=voxel)
    tgt = preprocess_pointcloud(tgt_pcd, voxel_size=voxel)
    src_pts = np.asarray(src.points, dtype=np.float64)
    tgt_pts = np.asarray(tgt.points, dtype=np.float64)

    # 方向约定（重要）：
    #   compute_pose_gt.py 输出 T_AB = inv(Pose_A) @ Pose_B，语义为“B 帧坐标 -> A 帧坐标”；
    #   而本评测中 src=A 帧点云、tgt=B 帧点云，配准估计 T_est 的语义是“source -> target”
    #   （即 A -> B）。两者互为逆，比较前需取逆，否则旋转/平移误差会失真。
    T_gt_align = np.linalg.inv(T_gt)

    results = {}

    # --- 主方法：改进 ICP（粗配准初始化） ---
    T_coarse, info_coarse = coarse_registration(
        src, tgt, voxel_size=voxel,
        fpfh_radius=params["fpfh_radius"],
        max_iteration=params["ransac_max_iteration"],
        distance_threshold=params["ransac_distance_threshold"])
    results["coarse"] = _to_metrics(
        T_coarse, T_gt_align, src_pts, tgt_pts, params["rmse_th"],
        params["rot_th"], params["trans_th"],
        extra={"fitness": info_coarse["fitness"]})

    T_fine, _ = improved_icp(
        src, tgt, init_transformation=T_coarse,
        max_iteration=params["icp_max_iteration"],
        threshold_min=params["icp_threshold_min"],
        threshold_factor=params["icp_threshold_factor"],
        early_stop_fitness=params["icp_early_stop_fitness"])
    results["improved_icp"] = _to_metrics(
        T_fine, T_gt_align, src_pts, tgt_pts, params["rmse_th"],
        params["rot_th"], params["trans_th"])

    # --- 基线 1：传统点到点 ICP（同粗配准初始化） ---
    p2p = o3d.pipelines.registration.registration_icp(
        src, tgt, params["ransac_distance_threshold"], T_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=params["icp_max_iteration"]))
    results["point2point_icp"] = _to_metrics(
        p2p.transformation, T_gt_align, src_pts, tgt_pts, params["rmse_th"],
        params["rot_th"], params["trans_th"])

    # --- 基线 2：FGR（Fast Global Registration，单位阵初始化） ---
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=params["fpfh_radius"], max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=params["fpfh_radius"], max_nn=100))
    fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=params["ransac_distance_threshold"]))
    results["fgr"] = _to_metrics(
        fgr.transformation, T_gt_align, src_pts, tgt_pts, params["rmse_th"],
        params["rot_th"], params["trans_th"])

    return results


def _to_metrics(T_est, T_gt, src_pts, tgt_pts, rmse_th,
                rot_th=DEFAULT_ROT_TH, trans_th=DEFAULT_TRANS_TH, extra=None):
    """把估计变换整理为指标字典（成功判定使用传入阈值）。"""
    rmse, ratio = correspondence_rmse(src_pts, tgt_pts, T_est, max_dist=rmse_th)
    m = {
        "rmse": rmse,
        "inlier_ratio": ratio,
        "rot_deg": rotation_error(T_est, T_gt),
        "trans_m": translation_error(T_est, T_gt),
    }
    if extra:
        m.update(extra)
    m["success"] = bool(is_success(m, rmse_th=rmse_th, rot_th=rot_th,
                                   trans_th=trans_th))
    return m


# ---------------------------------------------------------------
# 汇总与输出
# ---------------------------------------------------------------
def summarize(detail_rows):
    """把逐对明细汇总为每方法的均值/中位数/成功率。"""
    methods = sorted({r["method"] for r in detail_rows})
    summary = {}
    for method in methods:
        rows = [r for r in detail_rows if r["method"] == method]
        summary[method] = {
            "num_pairs": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "rmse_median": float(np.median([r["rmse"] for r in rows])),
            "rot_deg_mean": float(np.mean([r["rot_deg"] for r in rows])),
            "trans_m_mean": float(np.mean([r["trans_m"] for r in rows])),
        }
    return summary


def print_table(detail_rows, summary):
    """终端输出汇总表（无需第三方库）。"""
    print("\n========== 配准评测汇总 ==========")
    print(f"{'方法':<16}{'成功率':>10}{'RMSE均值':>12}{'旋转误差':>12}{'平移误差':>12}")
    for method, s in summary.items():
        print(f"{method:<16}{s['success_rate']*100:>9.1f}%"
              f"{s['rmse_mean']:>12.4f}{s['rot_deg_mean']:>12.3f}"
              f"{s['trans_m_mean']:>12.4f}")
    print("==================================")


def save_outputs(detail_rows, summary, out_dir):
    """保存明细 csv 与汇总 json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "method", "rmse", "inlier_ratio",
                         "rot_deg", "trans_m", "success"])
        for r in detail_rows:
            writer.writerow([r["pair_id"], r["method"], r["rmse"],
                             r["inlier_ratio"], r["rot_deg"], r["trans_m"],
                             int(r["success"])])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"detail_count": len(detail_rows), "summary": summary},
                  f, ensure_ascii=False, indent=2)
    logger.info("明细 -> %s/detail.csv，汇总 -> %s/summary.json", out_dir, out_dir)


def evaluate_registration(pairs, pcd_dir, pose_gt_dir, out_dir, params):
    """主流程：遍历帧对评测全部方法。"""
    detail_rows = []
    for item in pairs["pairs"]:
        pair_id = item["pair_id"]
        gt_path = Path(pose_gt_dir) / f"{pair_id}.txt"
        if not gt_path.is_file():
            logger.warning("缺真值 %s，跳过", gt_path)
            continue
        T_gt = np.loadtxt(gt_path).reshape(4, 4)

        try:
            src_path = pcd_path_for(pcd_dir, item["scene"], item["frame_a"])
            tgt_path = pcd_path_for(pcd_dir, item["scene"], item["frame_b"])
        except FileNotFoundError as e:
            logger.warning("跳过 %s：%s", pair_id, e)
            continue

        src_pcd = load_pcd_points(src_path)
        tgt_pcd = load_pcd_points(tgt_path)
        methods = run_methods(src_pcd, tgt_pcd, T_gt, params)
        for method, metrics in methods.items():
            detail_rows.append({"pair_id": pair_id, "method": method,
                                **{k: v for k, v in metrics.items()}})
        logger.info("完成 %s：coarse rmse=%.4f / improved rmse=%.4f",
                    pair_id, methods["coarse"]["rmse"], methods["improved_icp"]["rmse"])

    summary = summarize(detail_rows)
    save_outputs(detail_rows, summary, out_dir)
    print_table(detail_rows, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="配准评测（RMSE/旋转/平移/成功率，含基线条目）")
    parser.add_argument("--pairs", default="results/preprocess/pairs/pairs.json",
                        help="帧对清单 json")
    parser.add_argument("--pcd_dir", default="results/preprocess/pcd",
                        help="帧点云目录")
    parser.add_argument("--pose_gt_dir", default="results/preprocess/pose_gt",
                        help="6DOF 真值目录")
    parser.add_argument("--out_dir", default="results/registration/eval",
                        help="评测输出目录")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--fpfh_radius", type=float, default=0.25)
    parser.add_argument("--ransac_max_iteration", type=int, default=100000)
    parser.add_argument("--ransac_distance_threshold", type=float, default=0.03)
    parser.add_argument("--icp_max_iteration", type=int, default=50)
    parser.add_argument("--icp_threshold_min", type=float, default=0.02)
    parser.add_argument("--icp_threshold_factor", type=float, default=0.5)
    parser.add_argument("--icp_early_stop_fitness", type=float, default=1e-6)
    parser.add_argument("--rmse_th", type=float, default=DEFAULT_RMSE_TH,
                        help="RMSE 成功阈值（米）")
    parser.add_argument("--rot_th", type=float, default=DEFAULT_ROT_TH,
                        help="旋转误差成功阈值（度）")
    parser.add_argument("--trans_th", type=float, default=DEFAULT_TRANS_TH,
                        help="平移误差成功阈值（米）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    params = {
        "voxel_size": args.voxel_size,
        "fpfh_radius": args.fpfh_radius,
        "ransac_max_iteration": args.ransac_max_iteration,
        "ransac_distance_threshold": args.ransac_distance_threshold,
        "icp_max_iteration": args.icp_max_iteration,
        "icp_threshold_min": args.icp_threshold_min,
        "icp_threshold_factor": args.icp_threshold_factor,
        "icp_early_stop_fitness": args.icp_early_stop_fitness,
        "rmse_th": args.rmse_th,
        "rot_th": args.rot_th,
        "trans_th": args.trans_th,
    }
    evaluate_registration(pairs, args.pcd_dir, args.pose_gt_dir,
                          args.out_dir, params)


if __name__ == "__main__":
    main()
