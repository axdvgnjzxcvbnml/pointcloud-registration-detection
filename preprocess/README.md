# preprocess/ —— 数据预处理模块

把 SUN RGB-D / SUN3D 原始数据转换为后续配准与检测可用的统一格式。**纯 CPU**。

## 文件

| 脚本 | 用途 |
| --- | --- |
| `parse_sunrgbd.py` | 解析 SUN RGB-D 文件结构：RGB 图、深度图、相机内参 / 外参，参考 VoteNet 官方解析逻辑输出统一格式 |
| `depth_to_pointcloud.py` | 深度图 + 相机内参反投影为三维点云（numpy 计算，可导出 .ply） |
| `sample_frame_pairs.py` | SUN3D 序列按场景分组，抽取间隔 5 / 10 / 30 帧的相邻帧对（目标 300–500 对） |
| `compute_pose_gt.py` | 由 SUN3D `extrinsics/*.txt` 的 3×4 位姿计算相对变换 T_AB = Pose_A⁻¹ · Pose_B（6DOF 真值） |
| `generate_detection_data.py` | 生成 VoteNet 训练数据：每帧采样 50000 点，含 3D 边界框与投票向量 |

## 约定

- 深度图单位 mm，`depth_scale=10000`，有效深度截断 8m；
- 所有脚本均有 `argparse` 入口与 `--out_dir` 可配置输出；
- 参数默认值与 `configs/default.yaml` 的 `preprocess` 段一致。

## 示例

```bash
python preprocess/depth_to_pointcloud.py --in_dir <深度图目录> --out_dir results/preprocess/pcd
python preprocess/sample_frame_pairs.py --data_root data/SUN3D --out results/pairs.json
```
