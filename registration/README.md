# registration/ —— 点云拼接模块

帧间点云粗配准 + 精配准 + 评测 + 可视化。**纯 Python / CPU，只依赖 numpy、open3d、scipy，不需要 GPU。**

## 文件

| 脚本 | 用途 |
| --- | --- |
| `preprocess_pointcloud.py` | 统计离群点去除（k=20, std=2.0）、体素下采样（0.02m）、法向量估计（半径 0.1m） |
| `coarse_registration.py` | FPFH 特征（半径 0.25m）+ RANSAC（100000 次，距离阈值 1.5×voxel）输出 4×4 粗变换 |
| `fine_registration.py` | 改进 ICP：点到平面、自适应阈值（max(0.02, 0.5×上次 RMSE)）、早停（Δfitness<1e-6，≤50 次） |
| `evaluate_registration.py` | 评测 RMSE、旋转角误差（°）、平移误差（m）、成功率；基线：点到点 ICP、FGR |
| `visualize_registration.py` | Open3D 左右视图展示拼接前后对比 |

## 示例

```bash
python registration/coarse_registration.py --source results/preprocess/pcd/a.ply --target results/preprocess/pcd/b.ply --output results/registration/coarse.txt
python registration/fine_registration.py --source results/preprocess/pcd/a.ply --target results/preprocess/pcd/b.ply --init results/registration/coarse.txt --output results/registration/fine.txt
python registration/evaluate_registration.py --pairs results/preprocess/pairs/pairs.json
```

参数默认值与 `configs/default.yaml` 的 `registration` 段一致。
