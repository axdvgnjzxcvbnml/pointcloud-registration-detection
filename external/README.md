# external/ —— 第三方源码（只读引用）

本目录用于存放以源码方式引用的第三方项目，**不修改、不 fork、不提交**其内部代码。

## 当前引用

| 项目 | 来源 | 用途 | 说明 |
| --- | --- | --- | --- |
| VoteNet | <https://github.com/facebookresearch/votenet> | 点云主干与检测头（PointNet2） | 克隆后编译 CUDA 算子（见 docs/setup.md） |

## 安装

```bash
cd external
git clone https://github.com/facebookresearch/votenet.git
cd votenet
TORCH_CUDA_ARCH_LIST="7.0" python setup.py build_ext --inplace
cd ../..
```

## 约定

- `external/*` 已被 `.gitignore` 忽略（内容不入库），仅保留本 README；
- 引用的提交版本以 `git -C external/votenet log -1` 为准，实验记录时注明；
- 如需对第三方代码做修改，请 fork 到自己的仓库并单独管理，不要在 external/ 内直接改。
