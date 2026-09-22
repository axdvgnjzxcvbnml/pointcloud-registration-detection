# app/ —— 系统集成与可视化

基于 Open3D 的场景展示、配准对比、检测叠加与演示导出。所有脚本同时支持交互模式（`--interactive`，有显示器）与离屏模式（headless 服务器，默认渲染 PNG / GIF / MP4）。**纯 CPU。**

## 文件

| 脚本 | 用途 |
| --- | --- |
| `load_scene.py` | 加载 SUN3D 场景多帧点云并展示（支持体素下采样、离屏渲染 PNG） |
| `compare_registration.py` | 左右视图对比配准前后：左=未对齐（红/灰），右=对齐后（绿/灰），同一相机；可读矩阵或现场跑配准流水线 |
| `overlay_detection.py` | 点云叠加 3D 检测框（Open3D LineSet，12 边）+ 框心小球 + “类别 分数”文本（视锥投影 + PIL） |
| `export_demo.py` | 相机环绕动画逐帧离屏渲染，导出 GIF（Pillow）与 MP4（imageio-ffmpeg），可选叠加检测框 |

## 输入数据约定

- 点云：`.npz`（`point_cloud`/`xyz` + `colors`/`rgb`）或 `.ply`；
- 检测结果：JSON，字段 `boxes`（`[cx,cy,cz,sx,sy,sz,heading]` 列表）、`labels`、`scores`。

## 示例

```bash
python app/load_scene.py --scene-dir data/SUN3D/scene_001 --out-dir results/vis
python app/compare_registration.py --source a.npz --target b.npz --transform T.npy --out-dir results/vis
python app/overlay_detection.py --point-cloud frame.npz --detections pred.json
python app/export_demo.py --scene-dir data/SUN3D/scene_001 --gif --video
```
