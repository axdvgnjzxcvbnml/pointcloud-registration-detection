# data/ —— 数据软链接目录

本目录不存放实体数据，通过软链接指向服务器上的 SUN RGB-D 数据集。

## 软链接命令

```bash
# 实体数据放在服务器自己的目录（例如 /data/sunrgbd），只链接不拷贝
ln -s /your/data/path/SUNRGBD        data/SUNRGBD
ln -s /your/data/path/SUNRGBDtoolbox data/SUNRGBDtoolbox
ln -s /your/data/path/SUN3D          data/SUN3D
```

## 预期结构（链接完成后）

```
data/
├── SUNRGBD/        # 原始 RGB-D 帧：image/ depth/ label/ extrinsics/
├── SUNRGBDtoolbox/ # 官方 MATLAB 工具箱（标注元数据、读码逻辑）
└── SUN3D/          # SUN3D 序列（拼接实验），含 extrinsics/*.txt 相机位姿
```

## 注意

- 软链接路径必须是**绝对路径**，否则跨目录调用脚本会失效。
- 若原始数据目录结构与上述约定不同，请优先调整
  `preprocess/parse_sunrgbd.py` 中的路径拼接逻辑（第二批输出），
  而不是移动实体数据。
