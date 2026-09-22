# docs/ —— 文档

环境搭建、排错与实验记录模板。

| 文档 | 用途 |
| --- | --- |
| `setup.md` | V100 服务器环境搭建：Python 3.8、PyTorch 1.13.1+cu117、Open3D 0.17.0、VoteNet 克隆、PointNet2 CUDA 编译（重点 `TORCH_CUDA_ARCH_LIST="7.0"`）、数据软链、冒烟测试 |
| `troubleshooting.md` | 常见报错与排查：PointNet2 编译失败、CUDA 版本不匹配、SUN RGB-D 路径/格式问题、OOM 等 |
| `experiment_log.md` | 实验记录模板：日期、配置、mAP、参数量、备注，含消融总览表与配准评测表（数值均为 TBD 占位） |
