# V100 上线排坑手册（Top 10）

> 面向对象：首次把项目部署到 V100 单卡服务器（Volta/sm_70、CUDA 11.7、
> Python 3.8、PyTorch 1.13.1）的接手人。按 **踩坑概率从高到低** 排列。
> 每个坑 = 症状 / 原因 / 解决方案 / 相关文档章节。
> 通用原则：按 `docs/v100_checklist.md` 的顺序执行；先跑
> `bash scripts/check_env.sh`，再跑 `bash scripts/run_smoke.sh`，红一个先修一个。

---

## 坑 1：PointNet2 CUDA 算子编译失败

- **症状**：`pip install -e .` 或 `python setup.py install` 报
  `fatal error: THC/THC.h: No such file or directory`、`ninja: build stopped`、
  `error: #error unsupported GNU version`，或编译成功但 import 报 `undefined symbol`。
- **原因**：① PyTorch 1.13 已移除 THC/THC.h，官方 VoteNet 代码旧；② gcc 版本过新
  （CUDA 11.7 对 gcc ≤ 11）；③ nvcc 不在 PATH；④ 忘了为 Volta 指定架构。
- **解决方案**：
  ```bash
  # V100 = sm_70，必须显式指定，否则运行时才炸（见坑 2）
  cd external/votenet/pointnet2
  TORCH_CUDA_ARCH_LIST="7.0" python setup.py install
  # gcc 过新时：conda install -c conda-forge gcc=10 -y（或 gxx_linux-64=10）
  # THC/THC.h：按 docs/setup.md §4 的补丁说明处理（官方仓库 README 有对应 patch）
  ```
- **相关章节**：`docs/setup.md` §4；`docs/troubleshooting.md` §1.1~1.4。

## 坑 2：CUDA 版本不匹配 / torch 未编译 CUDA

- **症状**：`torch.cuda.is_available()` 返回 `False`；或
  `AssertionError: Torch not compiled with CUDA enabled`；或
  `CUDA error: no kernel image is available for execution on the device`。
- **原因**：装了 CPU 版 torch；或 torch 的 CUDA 版本与驱动/算子架构不符
  （典型：算子编了 sm_86，V100 是 sm_70）；或 conda 环境串了。
- **解决方案**：
  ```bash
  nvidia-smi        # 确认驱动 ≥ 450（CUDA 11.7 要求）
  nvcc --version    # 确认 CUDA 工具链 11.x
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  # 期望：1.13.1 / 11.7 / True
  # 不对就重装：pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  # 编译过的算子：确认 build 时 TORCH_CUDA_ARCH_LIST="7.0"（见坑 1）
  ```
- **相关章节**：`docs/setup.md` §0~§2；`docs/troubleshooting.md` §2.1~2.2。

## 坑 3：数据路径软链失效（最常见的数据坑）

- **症状**：`FileNotFoundError: data/SUNRGBD/...`、预处理产物为空、`check_data.sh`
  报"未找到任何场景"。
- **原因**：软链是相对路径/绝对路径写错、目标目录未挂载、`data/` 被 .gitignore
  但重建时忘了 `ln -s`、或者解压不完整（缺 `xtion/sun3ddata/<scene>/image|depth|extrinsics`）。
- **解决方案**：
  ```bash
  ls -la data/SUNRGBD            # 看软链是否指向真实路径
  bash scripts/check_data.sh     # 一键完整性检查，直接输出缺失路径与修复建议
  # 修复：ln -s /mnt/data/SUNRGBD data/SUNRGBD （用绝对路径，别用相对路径）
  ```
- **相关章节**：`docs/v100_checklist.md` Step 3 + 3.5；`docs/troubleshooting.md` §3.1。

## 坑 4：GPU 显存 OOM（CUDA out of memory）

- **症状**：训练中途 `CUDA error: out of memory`，或 import/前向第一帧即 OOM。
- **原因**：V100 16GB 单卡跑 60 epoch 融合训练，batch/点数/分辨率组合超显存；
  或进程残留占用；或 PyTorch 缓存碎片。
- **解决方案**（按优先级）：
  ```bash
  nvidia-smi                                   # 先看是不是别的进程占着（kill 掉）
  # 1) batch_size 8 -> 4 -> 2（configs/default.yaml: detection.train.batch_size）
  # 2) 减小每帧点数：points_per_frame 50000 -> 30000
  # 3) 换小输入：image_size 640 -> 480（需同步 projection/yolov8_feature）
  # 4) 清缓存：torch.cuda.empty_cache()（train_fusion.py 每 epoch 已兜底）
  # 5) 训练中断用断点续训恢复，不用从头跑（见坑 5）
  ```
- **相关章节**：`docs/troubleshooting.md` §4.1~4.4；`docs/v100_checklist.md` Step 8 失败项。

## 坑 5：断点续训格式错误 / 参数用错

- **症状**：`--resume` 加载后 loss 异常、维度不匹配 RuntimeError、或
  "已训至 epoch 0"（等于没恢复）；轻量化微调时误把教师权重当断点。
- **原因**：① 断点文件缺 `optimizer/scheduler/epoch` 字段（旧格式，见下）；
  ② **train_fusion 的 `--resume` 是训练断点，而 finetune_lightweight 的
  `--resume` 是教师权重**，两者语义不同容易混；
  ③ 用轻量化 checkpoint 恢复融合模型（头部通道 256→128 形状不符）。
- **解决方案**：
  ```bash
  # 融合训练断点续训（含 epoch/optimizer/scheduler/best_metric）：
  python detection/train_fusion.py --config ... --resume results/.../latest.pth
  # 轻量化微调：--resume 传教师（融合）权重，--resume_train 传轻量化自身断点
  python detection/finetune_lightweight.py --config ... \
      --resume results/.../02_.../fusion_epoch060.pth \
      --resume_train results/.../04_.../latest.pth
  # 旧格式（缺 scheduler/best_metric）也可恢复：代码按 .get() 兼容，loss 从头计
  ```
- **相关章节**：`docs/v100_checklist.md` Step 8/9 断点续训小节；`docs/experiment_log.md` 记录规范。

## 坑 6：消融配置字段缺失 / 汇总表落空

- **症状**：`v100_step7_ablation.sh` 汇总 CSV 里 mAP/参数/FLOPs 全空；或
  `ablation_report.py` 全部 TBD（明明跑过评测）；或某组实验用了别人的配置。
- **原因**：① 评测输出键名不一致（早期脚本读 `params/flops/latency_ms`，
  evaluate_detection 实际写 `params_m/flops_g/speed_ms_mean`——已修）；
  ② `ablation_report.py` 找的是 `results/ablation/<mode>/eval/map.json`，
  评测时 `--out_dir` 写错位置；③ 配置没继承 base 导致字段缺失。
- **解决方案**：
  ```bash
  python scripts/ablation_report.py   # 一键重生成对比表 + 柱状图，缺的显示 TBD
  ls results/ablation/<mode>/eval/    # 确认 map.json 存在且非空
  python -c "import json; m=json.load(open('results/ablation/02_fusion_concat/eval/map.json')); print(m['mAP@0.25'], m['params_m'])"
  ```
- **相关章节**：`configs/ablation/README.md`（五组配置含义）；`docs/experiment_log.md` §1.3。

## 坑 7：YOLOv8n 版本差异导致特征图索引/通道错

- **症状**：`yolov8_feature.py` 取 P3 层输出 shape 不是 `(B,64,80,80)`、
  投影采样报尺寸不匹配、或 ultralytics API 变动报 `AttributeError`。
- **原因**：ultralytics 版本漂移（8.0 与 8.2 的模型索引/输出顺序不同），
  或手动下载的 `yolov8n.pt` 与 requirements 版本不一致。
- **解决方案**：
  ```bash
  pip install "ultralytics==8.2.*"
  python -c "
  from ultralytics import YOLO
  import torch
  m = YOLO('yolov8n.pt'); f = m.model.model
  x = torch.zeros(1,3,640,640)
  print([l.shape for l in m.predict(source=x, verbose=False)[0].boxes] or 'ok')"
  # 本项目已验证：P3 层索引与通道数见 docs/troubleshooting.md §5.1，
  # 版本对不上就以该节实测 shape 为准修正 detection/yolov8_feature.py
  ```
- **相关章节**：`docs/troubleshooting.md` §5.1；`docs/v100_checklist.md` Step 4（依赖安装）。

## 坑 8：Open3D 无 GPU 渲染 / 渲染 API 版本差异

- **症状**：`app/*.py` 在无显示器/SSH 下报 `GLFW`/`offscreen` 错误；或
  `registration/visualize_registration.py` 在 Open3D 0.17 与 0.19 间
  `OffscreenRenderer` 参数/方法名不兼容。
- **原因**：Open3D 渲染走 CPU（无 GPU 光栅化），依赖 GLFW 上下文；headless 服务器
  需要 `offscreen` 模式；0.17→0.19 的 `setup_camera`/`render_to_image` API 有变动。
- **解决方案**：
  ```bash
  # 项目内已做 0.17/0.19 兼容层（visualize_registration.py / overlay_detection.py）：
  # 统一 try/except 切换 OffscreenRenderer 构造与 setup_camera 调用
  # 服务器无显示器时：export LIBGL_ALWAYS_SOFTWARE=1 或改用 headless 渲染（先试默认即可）
  # 若仍报 GLFW：sudo apt-get install libgl1 libglib2.0-0 libglfw3 后重试
  ```
- **相关章节**：`docs/troubleshooting.md` §5.2（已验证 0.17/0.19 兼容性）；
  `app/visualize_registration.py` 兼容层代码。

## 坑 9：VoteNet 权重加载失败

- **症状**：`votenet_baseline.py` / `train_fusion.py` 报
  `KeyError`/`size mismatch`/`missing keys`，或 mAP 全 0（权重没真加载上）。
- **原因**：① 权重没下载（`weights/votenet_sunrgbd.pth` 缺失）；
  ② 官方 checkpoint 的键名带 `module.` 前缀或结构不同（本项目按
  `strict=False` + 去 `module.` 前缀加载，缺失会告警）；
  ③ 文件名对不上（`votenet_sunrgbd.pth` vs `votenet_checkpoint.tar`）。
- **解决方案**：
  ```bash
  ls -la weights/                          # 确认 votenet_sunrgbd.pth 存在且非空（~40MB+）
  # 下载：按 docs/setup.md §6 的链接放 weights/votenet_sunrgbd.pth
  python detection/votenet_baseline.py --config configs/default.yaml --device cpu \
      --votenet_ckpt weights/votenet_sunrgbd.pth   # CPU 侧先验证能加载并前向
  # 日志里若出现 "主干缺失权重 N 个" 且 N 很大 → 权重结构不匹配，检查文件版本
  ```
- **相关章节**：`docs/setup.md` §6；`docs/v100_checklist.md` Step 4/8；
  `docs/troubleshooting.md` §3.2（数据与标注对齐）。

## 坑 10：mAP 计算口径不一致（对比实验结论失真）

- **症状**：同一模型两个脚本算出的 mAP 差很多；消融表里 02 比 00 还低
  但没做任何改动；README 与 experiment_log 数值对不上。
- **原因**：3D IoU 口径（轴对齐框 vs 朝向框）、匹配策略（贪心 vs 全局最优）、
  AP 插值（101 点 vs 11 点）、类别集合（10 类 vs 官方 37 类）不一致。
- **解决方案**：
  - 全项目统一走 `detection/evaluate_detection.py` 的 `compute_mAP`
    （VOC 101 点插值 + 贪心匹配 + 3D IoU，纯 numpy 可单测）；
  - 记录口径：`configs/default.yaml: evaluate.iou_thresholds=[0.25,0.5]`、
    `classes` 10 类列表，实验日志里注明 IoU 阈值与类别集合；
  - 对比实验必须同阈值、同数据集划分、同权重文件命名规则；
  - 结论以 `ablation_report.py` 输出表为准，禁止手抄数值。
- **相关章节**：`docs/experiment_log.md` §0 复现信息模板 + 记录规范；
  `configs/ablation/README.md`（统一口径说明）。

---

## 速查表

| # | 坑 | 一行解决方案 | 先看文档 |
| --- | --- | --- | --- |
| 1 | PointNet2 编译失败 | `TORCH_CUDA_ARCH_LIST="7.0" python setup.py install` | setup §4 |
| 2 | CUDA 版本不匹配 | 重装 `torch==1.13.1+cu117`，驱动 ≥450 | troubleshooting §2 |
| 3 | 数据软链失效 | `bash scripts/check_data.sh`，重新 `ln -s`（绝对路径） | checklist Step 3.5 |
| 4 | OOM | batch 减半 → 点数减半 → 清缓存 → 断点续训 | troubleshooting §4 |
| 5 | 断点格式/参数混用 | 融合用 `--resume`；轻量化 `--resume`(教师)+`--resume_train`(断点) | checklist Step 8/9 |
| 6 | 消融汇总落空 | `python scripts/ablation_report.py`；检查 eval/map.json 键名 | ablation README |
| 7 | YOLOv8n 版本差异 | 锁 `ultralytics==8.2.*`，以 §5.1 实测 shape 为准 | troubleshooting §5.1 |
| 8 | Open3D 渲染失败 | 用项目兼容层；`LIBGL_ALWAYS_SOFTWARE=1`；装 libglfw3 | troubleshooting §5.2 |
| 9 | VoteNet 权重加载失败 | 确认 `weights/votenet_sunrgbd.pth` 存在，CPU 前向先验证 | setup §6 |
| 10 | mAP 口径不一致 | 一律走 evaluate_detection.compute_mAP，同阈值同类别 | experiment_log §0 |
