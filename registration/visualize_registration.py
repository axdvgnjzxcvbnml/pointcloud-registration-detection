#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_registration.py — 拼接前后左右视图对比（第三批交付，纯 CPU）

功能
----
    用 Open3D 展示点云拼接效果，左右两视图并排：
      - 左视图：拼接前（源点云红色 + 目标点云灰色，原始位姿）
      - 右视图：拼接后（源点云经 4×4 变换对齐后绿色 + 目标点云灰色）
    支持两种模式：
      - GUI 模式（有显示环境）：双窗口并排
      - --offscreen 离屏模式（无显示环境 / 服务器）：输出 PNG
        （before.png / after.png / comparison.png 左右拼合图）

说明
----
    本模块属于 registration/，按约束只依赖 numpy / open3d / scipy，
    因此保留一份独立的渲染兼容层（不 import app/render_compat.py）。

用法
----
    python registration/visualize_registration.py \
        --source results/preprocess/pcd/a.ply \
        --target results/preprocess/pcd/b.ply \
        --transform results/registration/pair_ab_fine.txt \
        --out_dir results/registration/viz --offscreen
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SRC_COLOR_BEFORE = [0.9, 0.2, 0.2]   # 拼接前：源点云 红
SRC_COLOR_AFTER = [0.2, 0.8, 0.3]    # 拼接后：源点云 绿
TGT_COLOR = [0.6, 0.6, 0.65]         # 目标点云 灰


def load_point_cloud(path, color=None):
    """加载点云并设置颜色。"""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"点云为空: {path}")
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def _copy(pcd):
    """复制点云（变换显示用，不修改原对象）。"""
    import open3d as o3d

    c = o3d.geometry.PointCloud(pcd)
    return c


def visualize_gui(source_path, target_path, T, window_size=(960, 640)):
    """GUI 双窗口并排展示（左=拼接前，右=拼接后）。"""
    import open3d as o3d

    src = load_point_cloud(source_path)
    tgt = load_point_cloud(target_path, TGT_COLOR)

    src_before = _copy(src)
    src_before.paint_uniform_color(SRC_COLOR_BEFORE)
    src_after = _copy(src)
    src_after.paint_uniform_color(SRC_COLOR_AFTER)
    src_after.transform(T)

    windows = [
        ("拼接前（左视图）", [src_before, tgt], 0),
        ("拼接后（右视图）", [src_after, tgt], window_size[0] + 40),
    ]
    for name, geoms, x in windows:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=window_size[0],
                          height=window_size[1], left=x, top=40)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        vis.run()  # 阻塞直到窗口关闭
        vis.destroy_window()
    logger.info("GUI 展示完成")


def setup_camera_compat(renderer, fov, center, eye, up):
    """兼容 Open3D 0.17（Open3DScene.setup_camera）与 0.19+（OffscreenRenderer.setup_camera）。

    0.17：renderer.scene.setup_camera(fov, center, eye, up)，可接受 1D/2D 数组；
    0.19+：renderer.setup_camera(...)，且 center/eye/up 必须为 (3,1) float32。
    """
    if hasattr(renderer, "setup_camera"):
        renderer.setup_camera(fov,
                              np.asarray(center, dtype=np.float32).reshape(3, 1),
                              np.asarray(eye, dtype=np.float32).reshape(3, 1),
                              np.asarray(up, dtype=np.float32).reshape(3, 1))
    else:
        renderer.scene.setup_camera(fov, center, eye, up)


def render_offscreen(source_path, target_path, T, out_dir,
                     width=1280, height=640):
    """离屏渲染左右视图为 PNG（无显示环境可用）。"""
    import open3d as o3d
    from open3d.visualization import rendering

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = load_point_cloud(source_path)
    tgt = load_point_cloud(target_path, TGT_COLOR)
    src_before = _copy(src)
    src_before.paint_uniform_color(SRC_COLOR_BEFORE)
    src_after = _copy(src)
    src_after.paint_uniform_color(SRC_COLOR_AFTER)
    src_after.transform(T)

    def _render(geoms, name):
        # 注意：OffscreenRenderer 在 0.17/0.19 均无 headless 参数，直接 (w,h) 构造；
        # 无 GPU 环境依赖 EGL/OSMesa（详见 docs/troubleshooting.md）。
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        for i, g in enumerate(geoms):
            renderer.scene.add_geometry(f"pcd_{i}", g, mat)
        # 视角：以全部点云包围盒中心为焦点
        pts = np.concatenate([np.asarray(g.points) for g in geoms])
        center = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - center, axis=1).max()) or 1.0
        eye = center + np.array([radius * 1.2, radius * 0.7, radius * 1.0])
        setup_camera_compat(renderer, 60.0, center, eye, np.array([0.0, 0.0, 1.0]))
        img = renderer.render_to_image()
        path = out_dir / name
        o3d.io.write_image(str(path), img)
        logger.info("渲染 -> %s", path)
        return path

    p_before = _render([src_before, tgt], "before.png")
    p_after = _render([src_after, tgt], "after.png")

    # 左右拼合成一张对比图 comparison.png（内存拼接，不落临时文件）
    from PIL import Image
    imgs = [Image.open(p_before).convert("RGB"),
            Image.open(p_after).convert("RGB")]
    canvas = Image.new("RGB", (imgs[0].width + imgs[1].width,
                               max(imgs[0].height, imgs[1].height)), "white")
    canvas.paste(imgs[0], (0, 0))
    canvas.paste(imgs[1], (imgs[0].width, 0))
    comp_path = out_dir / "comparison.png"
    canvas.save(str(comp_path))
    logger.info("左右拼合 -> %s", comp_path)


def main():
    parser = argparse.ArgumentParser(description="拼接前后左右视图对比")
    parser.add_argument("--source", required=True, help="源点云路径")
    parser.add_argument("--target", required=True, help="目标点云路径")
    parser.add_argument("--transform", default=None,
                        help="4×4 变换矩阵（.txt，精/粗配准输出）；缺省则只展示拼接前")
    parser.add_argument("--out_dir", default="results/registration/viz",
                        help="离屏渲染输出目录")
    parser.add_argument("--offscreen", action="store_true",
                        help="离屏渲染（无显示环境），输出 PNG 而非 GUI")
    parser.add_argument("--window_size", default="960,640",
                        help="GUI 窗口尺寸（宽,高）")
    parser.add_argument("--verbose", action="store_true", help="调试日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    T = np.eye(4, dtype=np.float64)
    if args.transform:
        T = np.loadtxt(args.transform).reshape(4, 4)
        logger.info("加载变换矩阵: %s", args.transform)

    w, h = (int(x) for x in args.window_size.split(","))
    if args.offscreen:
        render_offscreen(args.source, args.target, T, args.out_dir,
                         width=w, height=h)
    else:
        try:
            visualize_gui(args.source, args.target, T, window_size=(w, h))
        except Exception as e:  # 无显示环境时 GUI 会抛错
            logger.error("GUI 展示失败（可能无显示环境）：%s", e)
            logger.info("建议改用 --offscreen 离屏渲染")
            raise


if __name__ == "__main__":
    main()
