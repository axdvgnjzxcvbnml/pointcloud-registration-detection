# results/ —— 实验结果与中间产物

本目录保存所有预处理、配准、检测的实验产物与结果。**内容不入库**（见 `.gitignore`），仅保留本 README。

## 目录约定

```
results/
├── preprocess/         # 预处理产物（parse_sunrgbd / depth_to_pointcloud / ...）
│   ├── frames_index.json
│   ├── frames/         # 帧 RGB+K npz
│   ├── pcd/            # 每帧点云
│   ├── pairs.json      # 帧对清单
│   ├── pose_gt/        # 6DOF 位姿真值
│   └── detection/      # VoteNet 训练数据 + split.json
├── reg/                # 配准结果（T 矩阵、eval.json）
├── detection/          # 检测训练权重 / 评测 report.json
├── ablation/           # 各消融实验（00~04）权重与报告
└── vis/                # 可视化产物（对比图 / 叠加图 / GIF / MP4）
```

## 规范

- 每个实验的产物放 `results/ablation/<实验ID>/` 下，与 `configs/ablation/<实验ID>.yaml` 一一对应；
- 数值结果统一输出 `report.json` / `eval.json`，并回填 README 结果表；
- 大文件（权重 .pth、点云 .ply、GIF/MP4）不入库，需要归档时另行处理。
