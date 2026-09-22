#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_registration.py — 拼接结果可视化（第三批交付）

功能
----
    加载源/目标点云与变换矩阵，渲染配准前后对比：
      左 = 配准前（源红 / 目标灰）
      右 = 配准后（源经变换转绿 / 目标灰）
    使用 OffscreenRenderer 离屏渲染 PNG（服务器无显示器可用）。

用法
----
    python registration/visualize_registration.py \
        --src a.npz --dst b.npz \
        --transform results/reg/T_fine.npy \
        --out-dir results/vis
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COLOR_SRC_RAW = (0.85, 0.25, 0.20)
COLOR_TGT = (0.62, 0.62, 0.62)
COLOR_ALIGNED = (0.20, 0.75, 0.25)


def load_pointcloud(path):
    """加载 .npz（point_cloud/xyz + colors/rgb）或 .ply 为 open3d PointCloud。"""
    import open3d as o3d

    p = Path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(p)
        key = "point_cloud" if "point_cloud" in data.files else "xyz"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(data[key], dtype=np.float64))
        for ckey in ("colors", "rgb", "color"):
            if ckey in data.files:
                pcd.colors = o3d.utility.Vector3dVector(
                    np.clip(np.asarray(data[ckey], dtype=np.float32) / 255.0, 0, 1))
                break
        return pcd
    return o3d.io.read_point_cloud(str(path))


def render_compare(src_pcd, dst_pcd, T, out_path, width=1280, height=720,
                   fov=60.0, point_size=2.0):
    """左右拼接对比图渲染。"""
    import open3d as o3d
    from open3d.visualization import rendering
    from PIL import Image

    src_raw = o3d.geometry.PointCloud(src_pcd)
    src_raw.paint_uniform_color(COLOR_SRC_RAW)
    src_alg = o3d.geometry.PointCloud(src_pcd)
    src_alg.transform(T)
    src_alg.paint_uniform_color(COLOR_ALIGNED)
    tgt = o3d.geometry.PointCloud(dst_pcd)
    tgt.paint_uniform_color(COLOR_TGT)

    pts = np.concatenate([np.asarray(g.points) for g in (src_alg, tgt)], axis=0)
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) or 1.0
    eye = center + np.array([0.0, -1.5 * radius, 0.8 * radius])

    tmp_dir = Path("__vis_tmp")
    tmp_dir.mkdir(exist_ok=True)
    l_png, r_png = tmp_dir / "l.png", tmp_dir / "r.png"
    for png, geos in ((l_png, [src_raw, tgt]), (r_png, [src_alg, tgt])):
        renderer = rendering.OffscreenRenderer(width, height, headless=True)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = point_size
        for i, g in enumerate(geos):
            renderer.scene.add_geometry(f"g_{i}", g, mat)
        renderer.scene.setup_camera(fov, center, eye, [0.0, 0.0, 1.0])
        o3d.io.write_image(str(png), renderer.render_to_image())

    l, r = Image.open(l_png), Image.open(r_png)
    canvas = Image.new("RGB", (l.width + r.width, l.height), (255, 255, 255))
    canvas.paste(l, (0, 0))
    canvas.paste(r, (l.width, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    for p in (l_png, r_png):
        p.unlink()
    tmp_dir.rmdir()
    logger.info("对比图 -> %s（左=配准前 右=配准后）", out_path)


def main():
    parser = argparse.ArgumentParser(description="配准前后对比可视化")
    parser.add_argument("--src", required=True, help="源点云 .npz/.ply")
    parser.add_argument("--dst", required=True, help="目标点云 .npz/.ply")
    parser.add_argument("--transform", required=True, help="4x4 变换 .npy")
    parser.add_argument("--out-dir", default="results/vis")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    src = load_pointcloud(args.src)
    dst = load_pointcloud(args.dst)
    T = np.load(args.transform)
    out = Path(args.out_dir) / f"{Path(args.src).stem}_vs_{Path(args.dst).stem}.png"
    render_compare(src, dst, T, out, args.width, args.height)


if __name__ == "__main__":
    main()
