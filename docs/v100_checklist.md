# V100 上线检查清单（docs/v100_checklist.md）

目标服务器：单卡 NVIDIA V100（16GB）/ CUDA 11.7 / Python 3.8。
按顺序执行，每项给出**命令 → 预期输出 → 失败时查看**。全部通过后再跑真实实验。

---

## 1. 环境检查

```bash
nvidia-smi                    # 驱动 / 显存 / 卡状态
nvcc --version                # CUDA 编译器版本（应 11.7）
python --version              # 应 3.8.x
conda list | grep torch       # torch 应为 1.13.1 +cu117
```

> **脚本调用方式（重要）**：本仓库所有 `.sh` 脚本（`scripts/*.sh`）一律用 `bash xxx.sh` 调用，**不要用 `./xxx.sh`**——GitHub 上的脚本文件未设置可执行位（仓库经 API 推送，模式固定为 100644），用 `bash` 调用即可，与执行位无关。

- **预期**：`nvidia-smi` 显示 V100（16GB）、驱动版本 ≥ 515；`nvcc` 输出 `release 11.7`；`python` 3.8.x；torch 行含 `1.13.1` 与 `+cu117`。
- **失败**：驱动/CUDA 版本不符 → `docs/setup.md` 第 0 节（安装前检查）；torch 版本不对 → `docs/setup.md` 第 2 节（PyTorch 安装）；`torch.cuda.is_available()` 为 False → `docs/troubleshooting.md` §2.1。

## 2. 编译 PointNet2 CUDA 算子（重点：V100 = sm_70）

```bash
cd external/votenet/pointnet2
TORCH_CUDA_ARCH_LIST="7.0" python setup.py install
```

- **预期**：`Compiling ... build/lib.../pointnet2_cuda*.so` 编译成功，无报错；`python -c "import pointnet2_utils"` 可导入。
- **失败**：`THC/THC.h: No such file` → `docs/troubleshooting.md` §1.1；编译成功但运行时 `no kernel image` → §1.2（检查是否漏了 `TORCH_CUDA_ARCH_LIST="7.0"`）；gcc 报错 → §1.3；ninja/nvcc 找不到 → §1.4。

## 3. 数据集软链

```bash
# 真实数据路径示例（按实际位置调整）
ln -s /mnt/data/SUNRGBD data/SUNRGBD
ln -s /mnt/data/SUNRGBDtoolbox data/SUNRGBDtoolbox
ls -la data/                 # 确认两个软链存在
ls data/SUNRGBD | head       # 应看到 SUNRGBD/ 与 SUN3D/ 等子目录
```

- **预期**：`data/SUNRGBD` 与 `data/SUNRGBDtoolbox` 均为软链且可访问；`SUN3D/` 序列目录存在。
- **失败**：路径不存在 → `docs/troubleshooting.md` §3.1（FileNotFoundError）；目录结构不完整 → `docs/setup.md` 第 5 节（数据准备）。

### 3.5 数据完整性检查（推荐，软链建立后必跑）

```bash
bash scripts/check_data.sh                    # 检查 data/SUNRGBD（默认）
bash scripts/check_data.sh /mnt/data/SUNRGBD  # 指定路径
```

- **检查内容**：① 根路径可访问；② 场景定位（目录同时含 `image/` 与 `depth/`，兼容 `xtion/sun3ddata/<scene>/` 及 kv1/kv2/xtion 直接布局）；③ 每场景含 `image/`、`depth/`、`extrinsics/` 三目录；④ `image` 与 `depth` 文件数一致（`extrinsics` 允许少于帧数但必须非空）；⑤ 空文件 / PNG 魔数 / 抽样解码校验。
- **预期**：汇总 `PASS=…  FAIL=0`，退出码 0，提示可继续 `bash scripts/v100_step2_preprocess.sh`。
- **失败**：脚本会输出具体缺失路径与修复建议（软链失效 → 重新 `ln -s`；无场景 → 检查 zip 是否解压完整 / 软链是否指向数据根）；仍不解 → `docs/troubleshooting.md` 第 3 节。

## 4. 冒烟测试

```bash
bash scripts/run_smoke.sh
```

- **预期**：输出各项 `[OK]`，最后 `冒烟测试通过`（exit 0）。
- **失败**：Open3D 版本/导入问题 → `docs/troubleshooting.md` §2 相关小节；数据路径检查失败 → §3.1。注意：脚本含数据/GPU 检查分支，CI 模式用 `SMOKE_CI=1 bash scripts/run_smoke.sh`（只跑不依赖数据集与 GPU 的项）。

## 5. 预处理

```bash
# 解析 SUN RGB-D 文件结构，输出统一格式（RGB/深度/内参/外参）
python preprocess/parse_sunrgbd.py \
    --data_root data/SUNRGBD \
    --out_dir results/preprocess/parsed \
    --scenes 客厅_001 卧室_002          # 按需指定场景；不传则全量

# 深度图 → 点云（示例单帧）
python preprocess/depth_to_pointcloud.py \
    --depth data/SUNRGBD/SUN3D/xxx/frame-000000.depth.png \
    --K data/SUNRGBD/SUN3D/xxx/intrinsics/frame-000000.txt \
    --out_path results/preprocess/pcd/xxx_000000 --out_format ply

# 帧对采样 + 6DOF 配准真值
python preprocess/sample_frame_pairs.py \
    --sun3d_dir data/SUNRGBD/SUN3D --intervals 5,10,30 --target_num 400 \
    --out_path results/preprocess/pairs/pairs.json
python preprocess/compute_pose_gt.py \
    --pairs results/preprocess/pairs/pairs.json \
    --sun3d_dir data/SUNRGBD/SUN3D --out_dir results/preprocess/pose_gt
```

- **预期**：parsed 目录生成统一格式；点云 PLY 非空；pairs.json 含 300–500 对；pose_gt.json 每对含 4×4 变换与 rot/trans 统计。
- **失败**：深度全 0 / 点云空 → `docs/troubleshooting.md` §3.2；帧对数量为 0 → §3.4；内参/标注坐标系对不上（检测框偏）→ §3.3。

## 6. 配准评测

```bash
python registration/evaluate_registration.py \
    --pairs results/preprocess/pairs/pairs.json \
    --pcd_dir results/preprocess/pcd \
    --pose_gt_dir results/preprocess/pose_gt \
    --out_dir results/registration/eval \
    --config configs/default.yaml
```

- **参数已锁定**：使用 `configs/default.yaml` → `registration` 节的最优组合
  （voxel 0.03 / FPFH 0.40 / RANSAC 500k / mutual_filter=False），**无需再调参**——
  来自真实 SUN3D 数据 20 对 × 3 次重复扫描（`docs/experiment_log.md` §1.5–§1.8）。
- **预期**：输出 4 方法（coarse / fgr / improved_icp / point2point_icp）的成功率、RMSE、旋转角误差、平移误差；写入 `results/registration/eval/summary.json` 与 `detail.csv`。数值回填 `docs/experiment_log.md` 配准表 REG-001~003。
- **预期结果（本地实测，V100 应接近）**：最优组合中位成功率约 **50%**（iv5 86% / iv10 29% / iv30 33%）；**iv5 最高、iv30 最低**（低重叠结构性难题，见 `docs/registration_failure_analysis.md`）。
- **V100 结果差异排查**：若 V100 上结果与本地差异 >10 个百分点，优先检查——
  ① 数据加载（帧对/点云路径、`depth_scale` 单位）；② 真值方向约定（T_AB 为 B→A，
  评测端已取逆）；③ Open3D 版本差异（本地 0.19，V100 建议 0.17.0+）。
- **失败**：所有方法成功率 0 → 检查真值方向约定（`compute_pose_gt` 的 T_AB 为 B→A，评测端已取逆）与帧对文件；单方法异常 → 查看该方法对应模块日志（`registration/coarse_registration.py` / `fine_registration.py`）。

## 7. VoteNet 基线

```bash
python detection/votenet_baseline.py \
    --ckpt weights/votenet_checkpoint.tar \
    --det_data_dir results/detection_data \
    --split test --num_class 10 --batch_size 8 \
    --out_dir results/detection/baseline --device cuda
```

- **预期**：输出 mAP@0.25 与 mAP@0.5（3D IoU，10 类及均值），写入 `results/detection/baseline/`。
- **失败**：PointNet2 算子加载失败（编译或架构）→ `docs/troubleshooting.md` §1；OOM → §4；数据格式错 → §3.2/§3.3。权重下载/放置 → `docs/setup.md` 第 6 节。

## 8. 融合训练（冻结双主干，只训融合头 + 检测头）

```bash
python detection/train_fusion.py \
    --config configs/ablation/02_fusion_concat.yaml \
    --device cuda --epochs 60 \
    --votenet_ckpt weights/votenet_checkpoint.tar \
    --yolov8_ckpt weights/yolov8n.pt
```

- **预期**：60 个 epoch 训练完成，每 epoch 打印 loss；checkpoint 存到 `results/ablation/02_fusion_concat/`（含 `latest.pth`、`fusion_best.pth`、每 10 epoch 的 `fusion_epochNNN.pth`）；TensorBoard 日志（`tensorboard --logdir results/ablation/02_fusion_concat` 可看）。
- **失败**：OOM → `docs/troubleshooting.md` §4（batch_size 减半/清缓存/减小点采样数）；`grid_sample` 形状错 → 检查 `projection.py` 的 letterbox/特征图尺寸对齐。

**断点续训（训练中断/超时后继续）**：

```bash
python detection/train_fusion.py \
    --config configs/ablation/02_fusion_concat.yaml \
    --resume results/ablation/02_fusion_concat/latest.pth \
    --device cuda \
    --votenet_ckpt weights/votenet_checkpoint.tar \
    --yolov8_ckpt weights/yolov8n.pt
```

- 断点文件含 `epoch / model / optimizer / scheduler / best_metric`，脚本自动从保存的 epoch 继续，无需手动改 `--epochs`；
- `latest.pth` 每 epoch 覆盖保存（磁盘占用最小），`fusion_best.pth` 为验证集 loss 最优，`fusion_epochNNN.pth` 为周期快照；
- 恢复后调度器（CosineAnnealing）从断点处继续，学习率不重置。

## 9. 轻量化微调（基于融合模型权重）

```bash
python detection/finetune_lightweight.py \
    --config configs/ablation/04_fusion_lightweight.yaml \
    --resume results/ablation/02_fusion_concat/fusion_epoch060.pth \
    --device cuda --finetune_epochs 20 \
    --votenet_ckpt weights/votenet_checkpoint.tar \
    --yolov8_ckpt weights/yolov8n.pt
```

- **预期**：20 epoch 微调完成，权重存 `results/ablation/04_fusion_lightweight/`（`latest.pth` / `lightweight_best.pth` / 每 10 epoch 的 `lightweight_epochNNN.pth`）；记录参数量与 mAP。
- **失败**：加载融合权重维度不匹配（轻量化头通道 256→128）→ 确认 `resume` 指向的是**融合模型**而非轻量化模型；OOM → §4。

**断点续训（注意与教师权重区分）**：

```bash
# --resume 始终传「融合模型教师权重」；--resume_train 传轻量化训练自身断点
python detection/finetune_lightweight.py \
    --config configs/ablation/04_fusion_lightweight.yaml \
    --resume results/ablation/02_fusion_concat/fusion_epoch060.pth \
    --resume_train results/ablation/04_fusion_lightweight/latest.pth \
    --device cuda \
    --votenet_ckpt weights/votenet_checkpoint.tar \
    --yolov8_ckpt weights/yolov8n.pt
```

## 10. 消融实验（五组依次跑）

```bash
for cfg in configs/ablation/00_votenet_baseline.yaml \
           configs/ablation/01_rgb_pseudo3d.yaml \
           configs/ablation/02_fusion_concat.yaml \
           configs/ablation/03_fusion_attention.yaml \
           configs/ablation/04_fusion_lightweight.yaml; do
  python detection/train_fusion.py --config "$cfg" --device cuda
done
```

- **预期**：5 组各自产出 `results/ablation/XX_*/`（权重 + 日志）；每组跑 `evaluate_detection.py` 得 mAP、参数量、FLOPs、推理速度。
- **失败**：单组失败不影响其他组（脚本各自独立）；结果统一回填 `docs/experiment_log.md` 总览表与消融表（含 5 组对比：单点云 / 单RGB / 融合-Concat / 融合-Attention / 融合+轻量化）。

---

## 附：上线顺序总览

```
1 环境检查 → 2 编译 PointNet2 → 3 数据软链 → 4 冒烟测试
→ 5 预处理 → 6 配准评测 → 7 VoteNet 基线
→ 8 融合训练 → 9 轻量化微调 → 10 消融实验
```

任一环节失败：先看 `docs/troubleshooting.md` 对应小节；环境类问题看 `docs/setup.md`；实验数值回填 `docs/experiment_log.md`。
