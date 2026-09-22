# registration/ —— 点云拼接模块（第三批交付）

FPFH + RANSAC 粗配准 → 改进 ICP（点到平面 + 自适应阈值 + 早停）精配准。**纯 CPU / Python。**

## 文件

| 脚本 | 用途 |
| --- | --- |
| `preprocess_pointcloud.py` | 统计去噪 + 体素下采样 + 法向量估计（含 PCA 手工实现，不依赖 open3d 法线） |
| `coarse_registration.py` | FPFH 特征 + RANSAC 粗配准（`registration_ransac_based_on_feature_matching`），返回 T 与 fitness |
| `fine_registration.py` | 改进 ICP：点到平面 + 自适应距离阈值 + fitness 早停；可选调用全局配准作为初值 |
| `evaluate_registration.py` | 批量评测帧对：RMSE / 旋转误差 / 平移误差 / 成功率（`registration_icp` 与 ICP 结果对比） |
| `visualize_registration.py` | 拼接前后对比（左右视图，离屏渲染 PNG，服务器可用） |

## 方法

```text
粗配准：FPFH (0.25m) + RANSAC (100k 迭代, 0.03m 阈值)
精配准：点到平面 ICP
    - 最大迭代 50
    - 自适应阈值：threshold = max(0.02, 0.5 × 上次 RMSE)
    - 早停：fitness 相对变化 < 1e-6
```

## 用法示例

```bash
python registration/coarse_registration.py --src a.ply --dst b.ply \
    --out results/reg/T_coarse.npy
python registration/fine_registration.py --src a.ply --dst b.ply \
    --init results/reg/T_coarse.npy --out results/reg/T_fine.npy
python registration/evaluate_registration.py --pairs results/pairs.json \
    --out results/reg/eval.json
python registration/visualize_registration.py --src a.npz --dst b.npz \
    --transform T_fine.npy --out-dir results/vis
```
