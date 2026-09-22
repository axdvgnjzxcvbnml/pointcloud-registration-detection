# preprocess/ —— 数据预处理模块（第二批交付）

从原始 SUN RGB-D / SUN3D 数据生成点云、帧对与 VoteNet 训练数据。**纯 CPU。**

## 文件

| 脚本 | 用途 |
| --- | --- |
| `parse_sunrgbd.py` | 解析文件结构：遍历 image/depth/label/extrinsics，读取内参（K）与标注，输出帧 RGB+K 元数据 json/npz |
| `depth_to_pointcloud.py` | 深度图（16bit 毫米）反投影为相机系点云，按 depth_trunc/max_points 截断与采样 |
| `sample_frame_pairs.py` | SUN3D 序列帧对采样：按间隔 5/10/30 枚举候选，可选按重叠率过滤，输出 pairs.json |
| `compute_pose_gt.py` | 由两帧外参计算相对位姿 T_AB（world→A 与 world→B 求逆相乘），输出 npy/txt |
| `generate_detection_data.py` | 生成 VoteNet 训练样本：采样点云、GT 框、投票标签（vote_label/vote_mask） |

## 数据流

```text
SUNRGBD 原始帧 ──parse_sunrgbd──▶ 帧 RGB+K npz
                                     │
深度图 ──depth_to_pointcloud──▶ 每帧点云 (npz/ply)
SUN3D 序列 ──sample_frame_pairs──▶ pairs.json ──compute_pose_gt──▶ T_AB 真值
帧+标注 ──generate_detection_data──▶ VoteNet 训练 npz + split.json
```

## 用法示例

```bash
python preprocess/parse_sunrgbd.py --data_root data/SUNRGBD --out_dir results/preprocess
python preprocess/depth_to_pointcloud.py --depth depth/00001.png --K K.json --out 00001.npz
python preprocess/sample_frame_pairs.py --scene-dir data/SUN3D/scene_001 \
    --intervals 5 10 30 --target-num 400 --out results/preprocess/pairs.json
python preprocess/compute_pose_gt.py --pairs results/preprocess/pairs.json \
    --extrinsics-dir data/SUN3D --out-dir results/preprocess/pose_gt
python preprocess/generate_detection_data.py --data_root results/preprocess \
    --out_dir results/preprocess/detection
```

## 输出格式约定

- 帧元数据：`{name, image, depth, label, K, extrinsics}`；
- 点云：`.npz`，字段 `point_cloud`/`xyz`（必选）、`colors`/`rgb`（可选）；
- 帧对：`[{src, dst, interval}]`；
- 位姿真值：4×4 矩阵 `T_src2dst`（.npy / .txt）；
- 检测数据：`{point_cloud, bboxes (M,7), class_ids (M,), vote_label (N,3), vote_mask (N,)}`。
