# external/ —— 第三方源码（只读引用）

存放以 Git 子模块 / 克隆方式引入的第三方仓库，**不提交到本仓库**（已在 `.gitignore` 忽略，本 README 除外）。

## 目录

```text
external/
└── votenet/        # facebookresearch/votenet（需手动克隆并编译 PointNet2）
```

## 准备方式

```bash
git clone https://github.com/facebookresearch/votenet.git external/votenet
cd external/votenet/pointnet2
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace   # V100 = sm_70
```

详细步骤与编译排错见 `docs/setup.md` 第 4 节、`docs/troubleshooting.md` 第 1 节。
