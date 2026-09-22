# results/ —— 实验结果与中间产物

存放预处理输出、配准矩阵与评测、检测 checkpoint / 评测 JSON、可视化图片 / GIF / 视频，**不提交到本仓库**（已在 `.gitignore` 忽略，本 README 除外）。

## 建议组织

```text
results/
├── preprocess/        # 解析后的帧数据、点云 .ply/.npz
├── pairs.json         # 帧对清单（间隔 5/10/30）
├── registration/      # 粗/精配准矩阵、评测结果
├── ablation/          # 五组消融实验，每组一个子目录
│   ├── 00_votenet_baseline/
│   ├── 01_rgb_pseudo3d/
│   ├── 02_fusion_concat/
│   ├── 03_fusion_attention/
│   └── 04_fusion_lightweight/
├── vis/               # 可视化 PNG
└── demo/              # GIF / MP4 演示
```

> 结果数值请回填到 `docs/experiment_log.md` 与根 README 的实验结果表，产物本身不入库。
