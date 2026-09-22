# docs/troubleshooting.md —— 常见问题排错（第七批交付）

> 按“现象 → 原因 → 解决”组织，覆盖编译、显存、数据、配准、检测五大类。
> 服务器排错请优先查阅本文件，再查官方 Issue。

## 1. 环境与编译

### Q1. `import torch` 报 CUDA driver 错误

**现象**：`CUDA error: no kernel image is available for execution on the device` 或 driver 版本过旧。

**原因**：PyTorch wheel 与驱动 / CUDA 不匹配（常见：默认源装了 CPU 版或 cu118+）。

**解决**：

```bash
# 卸载重装官方 cu117 wheel
pip uninstall -y torch torchvision
pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cu117
nvidia-smi   # 确认驱动 >= 515
```

### Q2. VoteNet / PointNet2 编译失败

**现象**：`setup.py build_ext --inplace` 报错。

**原因**：未指定 `TORCH_CUDA_ARCH_LIST`；或 nvcc 版本与 PyTorch 的 CUDA 版本不一致。

**解决**：

```bash
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
nvcc -V                                  # CUDA 11.7
python -c "import torch;print(torch.version.cuda)"   # 与上一致
```

### Q3. Open3D import 报 libGL 错误

**现象**：`ImportError: libGL.so.1: cannot open shared object file`。

**原因**：headless 环境缺 OpenGL 库。

**解决**：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

### Q4. ultralytics 版本不一致导致 API 变化

**现象**：`YOLO(ckpt).model` 不是 `nn.Sequential`，或 P3 层索引不对。

**解决**：固定 `ultralytics==8.2.*`；P3 层索引用 `evaluate_detection` 的逐层打印确认（默认 15）。

## 2. 显存 / 性能

### Q5. OOM（out of memory）

**现象**：训练中途 `CUDA out of memory`。

**解决**：

```bash
# 按顺序尝试：
batch_size 8 -> 4 -> 2
num_workers 4 -> 2
增加 max_points_per_frame 上限 / 体素下采样
关闭 TensorBoard 日志（省显存）
```

### Q6. 训练很慢

**解决**：

```bash
# 确认数据加载不是瓶颈
--num_workers 4 --prefetch_factor 2
# 确认 GPU 利用率（nvidia-smi 监控）
# 配准模块为纯 CPU，检测训练才用 GPU
```

## 3. 数据

### Q7. 数据路径错误 / 找不到文件

**现象**：`FileNotFoundError`。

**解决**：检查 `data/` 软链接（绝对路径），并核对 `configs/default.yaml` 的 `data.data_root` 等路径。

### Q8. 深度图范围异常（过亮/过暗）

**现象**：点云出现大量离群点或黑洞。

**解决**：核对 `depth_scale=10000`（SUN RGB-D 为毫米）与 `depth_trunc=8.0`；必要时单独调参并可视化。

### Q9. 帧对采样结果为空

**现象**：`sample_frame_pairs.py` 输出 0 个帧对。

**解决**：检查 `pair_intervals` 是否超过序列长度；确认 `data/SUN3D/scene_xxx` 存在 extrinsics 与帧文件；降低 `pair_overlap_threshold`。

## 4. 配准

### Q10. 粗配准失败（RANSAC 无收敛）

**现象**：fitness 过低 / 变换明显错误。

**解决**：调大 `ransac_max_iteration`（至 200000）；增大 `ransac_distance_threshold`（至 0.05）；确认预处理一致（voxel、法向量）。

### Q11. 精配准发散 / 误差大

**现象**：ICP 后 RMSE 不降反升。

**解决**：核对初始变换方向（`T_src->dst`）；调低 `icp_threshold_factor`；确认 `icp_threshold_min` 不要过小；可视化检查（`visualize_registration.py`）。

## 5. 检测

### Q12. 图像特征投影错位

**现象**：3D 框与图像内容明显错位。

**解决**：核对内参 K 与 `letterbox` 参数（`yolov8_feature.get_letterbox_params`）；确认投影用同一 resize/letterbox 链路。

### Q13. mAP 异常低 / 全零

**现象**：mAP@0.25 全 0。

**解决**：检查评测真值加载与 IoU 口径（`evaluate_detection.compute_mAP`）；确认检测框未做 NMS 前的原始输出是否合理；检查类别索引对齐。

### Q14. 训练 loss 不下降

**现象**：loss 震荡或恒值。

**解决**：确认学习率（cosine 调度）；确认冻结主干后只有融合头/检测头在更新；核对数据归一化；尝试 lr 调低一个量级。

## 6. 通用

### Q15. 日志输出编码问题

**现象**：终端打印中文乱码。

**解决**：`export PYTHONIOENCODING=utf-8`；终端用 UTF-8 locale。
