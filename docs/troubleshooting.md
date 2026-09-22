# 排错手册（Troubleshooting）

按"症状 → 可能原因 → 排查步骤 → 解决"组织。每节开头标注出现阶段。

---

## 1. PointNet2 CUDA 算子编译失败

**出现阶段**：`external/votenet/pointnet2` 执行 `python setup.py build_ext --inplace` 时。

### 1.1 症状：`fatal error: THC/THC.h: No such file or directory`

- **原因**：PyTorch 1.13 已移除 `THC/THC.h` 等旧头文件，VoteNet 仓库的 pointnet2 与 torch 1.13 不兼容（VoteNet 官方基于较老 torch）。
- **排查**：
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```
- **解决**：优先使用本仓库 `detection/` 的 `VoteNetBackbone` 封装路径（已把官方 `pointnet2_utils` 作为可替换依赖）；若仍需官方算子，参考 VoteNet 社区 patch 修改 `pointnet2/lib/src/*.cu` 中 `THC/THC.h` → `ATen/cuda/CUDAContext.h`（并去掉 `THCState` 用法），或改用官方提供的兼容分支。**不要**用降级 torch 的办法换兼容性。

### 1.2 症状：`no kernel image is available for execution on the device`（编译成功但运行时爆）

- **原因**：`TORCH_CUDA_ARCH_LIST` 未设或设成了非 V100 架构（如 7.5/8.0/8.6），编译产物里没有 sm_70 的 kernel。
- **排查**：`nvidia-smi` 确认显卡型号（V100 = Volta = sm_70）。
- **解决**：
  ```bash
  cd external/votenet/pointnet2
  rm -rf build _ext*           # 必须清旧产物！
  TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
  ```

### 1.3 症状：gcc 编译报错（`error: #error unsupported GNU version!` 等）

- **原因**：gcc 版本过新（PyTorch 1.13 官方二进制基于 gcc 9）。
- **解决**：
  ```bash
  conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9
  # 在 pointnet2 目录下，用 conda 编译器重新编译
  CC=x86_64-conda-linux-gnu-cc CXX=x86_64-conda-linux-gnu-c++ \
      TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
  ```

### 1.4 症状：`ninja: build stopped: subcommand failed` / nvcc 找不到

- **原因**：ninja 未装或 nvcc 不在 PATH；或 CUDA 环境变量缺失。
- **解决**：
  ```bash
  pip install ninja
  export PATH=/usr/local/cuda-11.7/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
  nvcc --version                # 确认 11.7
  ```

---

## 2. CUDA / PyTorch 版本不匹配

**出现阶段**：环境搭建与首次运行任何 torch 代码。

### 2.1 症状：`torch.cuda.is_available()` 返回 False

- **原因**：
  a) 装了 CPU 版 torch（默认 pip 源）；b) 驱动过旧（CUDA 11.7 需驱动 ≥ 515.43）。
- **排查**：
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
  nvidia-smi
  ```
- **解决**：
  ```bash
  pip uninstall -y torch torchvision
  pip install torch==1.13.1 torchvision==0.14.1 \
      --index-url https://download.pytorch.org/whl/cu117
  ```

### 2.2 症状：`AssertionError: Torch not compiled with CUDA enabled` / `CUDA error: invalid device function`

- **原因**：cu118 wheel 与 CUDA 11.7 运行时混用、或 arch 列表错误（同 1.2）。
- **解决**：统一用 cu117 wheel；编译类算子一律 `TORCH_CUDA_ARCH_LIST="7.0"`。

### 2.3 症状：`CUDA error: out of memory` 出现在 import / 前向第一帧

- 若一上来就 OOM，先看第 4 节（OOM 排查），优先减 `--num-points` 与 `batch_size`。

---

## 3. SUN RGB-D 数据路径 / 格式问题

**出现阶段**：`preprocess/` 各脚本。

### 3.1 症状：`FileNotFoundError: data/SUNRGBD/...`

- **原因**：软链接失效（源路径被移动/卸载）、或根本没建链接。
- **排查**：
  ```bash
  ls -l data/                    # 看 l 开头软链接是否指向存在路径
  ls data/SUNRGBD/ | head        # 链接目标可读？
  ```
- **解决**：重新建软链接（用绝对路径）：
  ```bash
  ln -s /绝对路径/SUNRGBD data/SUNRGBD
  ln -s /绝对路径/SUNRGBDtoolbox data/SUNRGBDtoolbox
  ln -s /绝对路径/SUN3D data/SUN3D
  ```

### 3.2 症状：解析出的深度全是 0 / 点云空

- **原因**：深度图 scale 不对（SUN RGB-D 深度单位 mm，scale=10000），或数据是 16bit PNG 被按 8bit 读。
- **排查**：`python preprocess/parse_sunrgbd.py --depth ... --out debug.npz` 后检查深度统计。
- **解决**：确认用 `cv2.imread(..., -1)` / PIL 16bit 读取；`scale=10000`、截断 `8m`（与 `configs/default.yaml` 的 `preprocess` 段一致）。

### 3.3 症状：相机内参与标注坐标系对不上（检测框偏）

- **原因**：SUN RGB-D 不同子集（SUNRGBD2DImages / NYUv2 / SUN3D）内参不同，占位默认内参 `DEFAULT_K` 只是骨架默认值。
- **解决**：按代码中 TODO，用 `SUNRGBDtoolbox` 元数据（`SUNRGBD2DImages/meta` 等）读取每帧真实内参与外参，替换占位默认值。

### 3.4 症状：SUN3D 帧对数量为 0

- **原因**：`sample_frame_pairs.py` 候选帧判定依赖连续帧号；场景帧号不连续时按"帧号+间隔"取候选会漏。
- **解决**：确认按场景名分组后帧号单调递增；脚本已按"下一帧是否在集合中"取候选，若你的目录命名不同，调整 `build_candidate_pairs` 的帧号提取逻辑（脚本内注释已说明）。

---

## 4. GPU 显存 OOM（CUDA out of memory）

**出现阶段**：检测训练/推理（V100 16GB）。

### 4.1 症状与直接解法（按优先级）

| 手段 | 做法 | 预期收益 |
| --- | --- | --- |
| 减 batch | `--batch-size 8 → 4/2` | 线性 |
| 减采样点数 | `configs` 中 `preprocess.points_per_frame: 50000 → 30000/20000` | 大（VoteNet 点数即显存大头） |
| 梯度累积 | 训练脚本 `--grad-accum`（如需可加） | 等效大 batch 不增显存 |
| 关图像分支 | 消融 `01/00` 配置 | 省 YOLO 特征与投影 |
| 降低特征图开销 | YOLO P3 升维 128 → 64（改 config） | 中等 |
| 手动清缓存 | `torch.cuda.empty_cache()`（脚本已按需调用） | 缓解碎片 |

### 4.2 定位占用

```bash
nvidia-smi                                   # 实时显存
python -c "
import torch
m = torch.load('results/.../checkpoint.pth', map_location='cpu')
print(sum(v.numel() for v in m['model_state'].values()) * 4 / 1e9, 'GB 权重')
"
```

### 4.3 缓解分配碎片（PyTorch 1.13）

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 4.4 说明

- 本仓库为**单卡 V100 设计**：融合模型冻结双主干、只训 ~2M 参数的融合头+检测头，显存占用远小于端到端训练；若仍 OOM，优先按 4.1 减 `points_per_frame`。
- 不推荐直接对 VoteNet 主干开 AMP/FP16（官方未支持，易出 NaN）；如确需可先小 batch 试跑验证。

---

## 5. 已验证项（CPU 沙箱 / CI，2026-09-22）

> 以下结论来自纯 CPU 云沙箱（Python 3.12.11 / Open3D 0.19.0 / torch 2.14.0+cpu / ultralytics 8.2.103）实测。
> 目标环境为 Python 3.8 + PyTorch 1.13.1 + CUDA 11.7 + Open3D 0.17.0，接口层面已兼容，最终以 V100 上 `docs/v100_checklist.md` 的验证为准。

### 5.1 YOLOv8n P3 层（ultralytics 8.2.x）

- **已验证**：加载 `yolov8n.pt`（8.2.103），`model.model` 是 `DetectionModel`（不可迭代），**其 `.model` 属性才是 `nn.Sequential`**；`yolov8_feature.py` 已改为取 `YOLO(ckpt).model.model`。
- **P3 索引**：`model.model[15]` 为 `C2f`，输出 `(B, 64, 80, 80)`（stride=8，80×80×64）——与代码默认 `p3_layer_index=15` 一致。注意 `model.model[3]`（Conv）也是 80×80×64，那是主干浅层特征，**不是** neck 融合后的 P3，请勿改索引到 3。
- **forward 跳连**：YOLOv8 neck 含多输入 `Concat`（`f=[-1, k]`），不能逐层 `x = layer(x)` 直推（会在 `layer[11]` 抛 `torch.cat` TypeError）。`yolov8_feature.py` 已按官方 `_predict_once` 维护历史输出 `y` 并处理 `m.f` 跳连。
- **升维验证**：1×1 Conv 后输出 `(B, 128, 80, 80)`，`proj_conv` 参数 8320（= 64×128 + 128）。

### 5.2 Open3D 0.17 / 0.19 API 兼容性

- **已验证（0.19 实跑）**：`registration/` 全部 Open3D 调用跑通 —— `voxel_down_sample` / `estimate_normals` / `orient_normals_towards_camera_location` / `remove_statistical_outlier` / `compute_fpfh_feature` / `registration_ransac_based_on_feature_matching`（含 `CorrespondenceCheckerBasedOnEdgeLength/Distance`、`RANSACConvergenceCriteria`）/ `registration_icp`（点到平面/点到点，`ICPConvergenceCriteria`）/ `registration_fgr_based_on_feature_matching`（`FastGlobalRegistrationOption`）/ `read/write_point_cloud` / `write_image`。这些 API 在 0.17 均为标准形态、参数名一致。
- **0.17 ↔ 0.19 唯一差异**：`OffscreenRenderer.setup_camera` —— 0.17 挂在 `renderer.scene` 上，0.19 挂在 `renderer` 上（且 center/eye/up 需 `(3,1) float32`）。4 个可视化脚本已统一使用 `setup_camera_compat()` 兼容函数（app 侧在 `app/render_compat.py`，registration 侧保留独立副本）。
- **`OffscreenRenderer` 没有 `headless` 参数**（0.17 / 0.19 签名均为 `(width, height, resource_path='')`）；传 `headless=True` 会抛 TypeError，所有 app 脚本已去除。
- **matplotlib 兼容**：matplotlib ≥ 3.9 移除了 `cm.get_cmap`，类别配色已改为优先 `matplotlib.colormaps['tab20']`、回退 `cm.get_cmap`，0.17 目标环境自带的 3.6/3.7 与沙箱 3.11 均可运行。
- **无 GPU 离屏渲染**：0.17 需要 EGL（`libEGL`）；0.19 在无 GPU 环境自动启用 `EGL headless mode` + Mesa 软件 OpenGL（沙箱实测渲染成功）。0.20 在无 GPU 环境强制 Vulkan 后端会崩溃（`vkCreateInstance: ErrorIncompatibleDriver`），**不要**在服务器上升级到 0.20。

### 5.3 PyTorch 1.13.1 兼容性（静态审查 + 沙箱实测）

- `detection/` 使用的 torch API 均为 1.13.1 稳定接口：`F.grid_sample`（投影双线性采样，`align_corners=True` 与 numpy floor 插值严格对齐，padding_mode="zeros"）、`nn.MultiheadAttention(batch_first=True)`、`nn.GELU`、`nn.Conv1d(groups=...)`、`torch.optim.AdamW`、`CosineAnnealingLR`、`F.smooth_l1_loss`、`F.binary_cross_entropy_with_logits`、`clip_grad_norm_`、`torch.einsum`（批量外参）。
- 沙箱（torch 2.14+cpu）实测：融合头（concat/attention）、轻量化头结构自检输出形状正确；`ProjectionModule` 整批向量化重写后，与 numpy 参考实现在 B=4/N=800 上 valid mask 完全一致、特征最大误差 2.88e-05，且在 `warnings.simplefilter('error', UserWarning)` 下无任何 UserWarning。
- **注意**：沙箱无 Python 3.8 / torch 1.13.1 wheel，1.13.1 的精确行为需在 V100 上最终确认；如遇 1.13 特有告警，请记录到实验日志备注。

### 5.4 冒烟测试与环境检查

- `bash scripts/run_smoke.sh --ci` 在沙箱全通过（PASS=4 / FAIL=0）：Python 版本、Open3D 可用、合成深度图转点云（z=1.0000m 断言通过）、合成立方体粗配准 + 精配准（旋转/平移误差 < 阈值）。完整模式 PASS=8 / FAIL=0（CUDA 不可用仅 WARN）。
- `scripts/check_env.sh --ci` 可正确报告核心依赖版本（沙箱版本与目标环境不同时会如实报 FAIL，属预期；CI 会安装目标版本）。

### 5.5 CPU 侧模型自检（2026-09-22）

> 目标：把「训练循环能不能跑」这类调试全部在 CPU 侧完成，V100 开机后只跑真数据。
> 入口：`python3 tests/test_cpu_forward_checks.py`（17 项 PASS）与 `python3 tests/test_map_cpu.py`（18 项 PASS）。

- **VoteNet 主干是 CUDA-only**：`Pointnet2Backbone` 依赖 `external/votenet/pointnet2` 的 CUDA 算子，CPU 沙箱无法实例化。自检用 `MockVoteNetBackbone`（MLP，输出形状 `(B,N,3)+(B,N,128)` 与真实主干一致）替换，验证「投影 + 融合头 + 检测头 + 训练循环」的装配正确性；真实主干在 V100 上由 `v100_step1` 编译后生效。
- **YOLOv8n P3 特征（CPU 实跑）**：输入随机 `(1,3,640,640)` → `model.model[15]`（C2f）→ 1×1 Conv 升维 → 输出 `(1,128,80,80)`，与 `yolov8_feature.py` 默认索引一致（见 5.1）。
- **FusionModel 前向（concat / attention，CPU 实跑）**：两种融合方式的输出键与 shape 全部符合预期（`objectness (B,N,1)`、`center/size (B,N,10,3)`、`heading (B,N,10,2)`、`class_scores (B,N,10)`）；反向传播中融合头梯度全覆盖、检测头受监督分支（objectness/center）有梯度。
  - **预期行为说明**：`compute_loss` 目前只监督 objectness + center（其余分支为占位损失 0，见 `train_fusion.compute_loss` 的 TODO），因此 size/heading/class 分支无梯度属正常，不是 bug。
- **train_fusion() 真实训练循环（CPU 实跑 1 epoch）**：4 帧模拟 mini 数据（5000 点/帧 + rgb/K）跑通完整训练循环（dataloader → forward → loss → backward → clip_grad → optimizer.step → scheduler → 存权重），产出 `fusion_epoch001.pth`。
- **finetune_lightweight() 真实微调循环（CPU 实跑 1 epoch）**：基于上一步融合权重微调 1 epoch 跑通，产出 `lightweight_epoch001.pth`。
  - **发现并修复的 bug**：`load_pretrained` 原实现 `strict=False` 仍会对轻量化头（通道 256→128）形状不匹配抛 `RuntimeError: size mismatch ...`。已改为 `_filter_by_shape` 先按键名 + shape 过滤，只迁移可复用的主干/投影权重，头部从零训练；日志按「迁移 X/Y 键」如实报告。
- **mAP 评测逻辑（CPU 单测 18 项）**：`tests/test_map_cpu.py` 用模拟预测 + 模拟真值验证 `box3d_iou`（完全重叠 1.0 / 无重叠 0.0 / 半重叠精确值）、`compute_mAP`（全命中 = 1.0；含 FP 时 AP 下降；类别标错不算 TP）、`load_gt` 读写回路（形状/索引/缺失帧对齐）。
  - **口径说明**：mAP 按 VoteNet 官方对全部 10 类取平均，无真值/无检测的类 AP=0（单测中以 10 类平均 = 0.2 验证该口径）。
- **真实数据格式验证**：`tests/test_votenet_dataloader_cpu.py` 校验 49 帧真实点云生成的 VoteNet 数据（50000 点/帧、双键 `point_cloud`/`point_clouds`、bbox/vote 键齐全）49/49 PASS；当前 SUN3D studyroom 无 3D 框标注，bbox=0 属预期，真实标注由 `preprocess/extract_gt.py` 在 V100 数据上生成。

---

## 6. 其他常见问题

| 症状 | 原因 | 解决 |
| --- | --- | --- |
| `ImportError: libGL.so.1`（headless） | 缺 OpenGL 运行库 | `sudo apt-get install -y libgl1 libglib2.0-0` |
| `visualizer window could not be created` | 无显示器跑交互模式 | 改用各 app 脚本的离屏模式（默认）；需要弹窗时加 `--interactive` |
| `imageio` / `ffmpeg` 报错（export_demo） | 未装视频依赖 | `pip install imageio imageio-ffmpeg` |
| YOLOv8n 权重下载卡住/失败 | 网络/代理访问 ultralytics 资产 | 手动下载 `yolov8n.pt` 放 `weights/`；代码支持 `--weights` 指定 |
| `P3 层索引报错`（yolov8_feature） | ultralytics 版本模型结构不同 | 逐层打印各层 shape，改 `--p3_layer_index` |
| VoteNet checkpoint 加载键名不匹配 | 官方权重是裸 state_dict / 带 module. / .tar | `votenet_baseline.py` 已兼容多种形态；仍失败则打印 `state_dict` 键对比（脚本内 TODO） |
| `results/`、`weights/` 写入权限 | 目录属主问题 | `mkdir -p results weights && chmod u+w` |

---

## 7. 通用排查原则

1. **先看日志与退出码**：本项目脚本均打印阶段日志（`[INFO]`），定位到具体函数再查。
2. **最小复现**：用 `--dry-run` / 单帧 / 单 batch 跑通再放大。
3. **改环境后清缓存**：torch 算子类编译产物（`build/`、`_ext*`、`~/.cache/torch_extensions`）改动后必须清理重编。
4. **版本三件套确认**：
   ```bash
   python -c "import torch, open3d; print(torch.__version__, torch.version.cuda)"
   python -c "import open3d; print(open3d.__version__)"
   nvidia-smi
   ```
5. 仍无法定位时，把完整报错 + 运行命令 + 环境三件套输出贴到实验记录（`docs/experiment_log.md` 的备注栏）再求助。
