# ============================================================
# app/render_compat.py —— Open3D 离屏渲染兼容层
# ------------------------------------------------------------
# 统一处理 Open3D 0.17 与 0.19+ 的相机 API 差异：
#   - 0.17：OffscreenRenderer(width, height)，相机设置在
#           renderer.scene.setup_camera(fov, center, eye, up)
#   - 0.19+：OffscreenRenderer(width, height)，相机设置在
#           renderer.setup_camera(fov, center(3,1), eye(3,1), up(3,1))
# 另：0.17/0.19 的 OffscreenRenderer 均无 headless 参数
#     （构造器本身就是离屏渲染器），传 headless=True 会 TypeError。
# 无 GPU 环境：0.17 需 EGL（libEGL），0.19 自动启用 EGL headless
# + Mesa 软件 OpenGL；0.20 强制 Vulkan 后端，无 GPU 会崩溃，勿用。
# ============================================================
import numpy as np


def make_offscreen_renderer(width, height):
    """构造离屏渲染器（不传 headless 参数，0.17/0.19 通用）。"""
    from open3d.visualization import rendering
    return rendering.OffscreenRenderer(width, height)


def setup_camera(renderer, fov, center, eye, up):
    """兼容 0.17（Open3DScene.setup_camera）与 0.19+（OffscreenRenderer.setup_camera）。

    Args:
        renderer: OffscreenRenderer 实例。
        fov: 垂直视场角（度）。
        center / eye / up: 长度 3 的 array-like（世界坐标）。
    """
    if hasattr(renderer, "setup_camera"):
        # 0.19+：center/eye/up 要求 (3, 1) float32
        renderer.setup_camera(
            fov,
            np.asarray(center, dtype=np.float32).reshape(3, 1),
            np.asarray(eye, dtype=np.float32).reshape(3, 1),
            np.asarray(up, dtype=np.float32).reshape(3, 1))
    else:
        # 0.17：挂在 scene 上，接受长度 3 的向量
        renderer.scene.setup_camera(fov, np.asarray(center, dtype=np.float64),
                                    np.asarray(eye, dtype=np.float64),
                                    np.asarray(up, dtype=np.float64))


def orbit_eye(center, radius, elevation_deg, azimuth_deg):
    """由中心/半径/俯仰角/方位角计算相机位置 eye（与各 app 脚本原逻辑一致）。"""
    elev = np.deg2rad(elevation_deg)
    azim = np.deg2rad(azimuth_deg)
    return np.array([
        center[0] + radius * np.cos(elev) * np.sin(azim),
        center[1] + radius * np.cos(elev) * np.cos(azim),
        center[2] + radius * np.sin(elev),
    ])


def scene_center_radius(geometries):
    """合并多个点云几何，返回 (center, radius)；空几何抛 ValueError。"""
    pts = np.concatenate([np.asarray(g.points) for g in geometries], axis=0)
    if len(pts) == 0:
        raise ValueError("几何中没有任何点，无法计算相机视角")
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    if radius <= 1e-9:
        radius = 1.0
    return center, radius
