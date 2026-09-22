# docs/ —— 项目文档

| 文件 | 内容 | 输出批次 |
| --- | --- | --- |
| `setup.md` | 环境搭建：V100 / Python 3.8 / PyTorch 1.13.1 (cu117) / Open3D 0.17.0 / ultralytics 8.2.x / VoteNet 编译 | 第七批 |
| `troubleshooting.md` | 常见问题排错：编译、显存、数据、配准、检测 | 第七批 |
| `experiment_log.md` | 实验记录模板：记录规范、模板表、回填规则 | 第七批 |

## 约定

- 所有版本号以 README 环境表为准（PyTorch 1.13.1+cu117、Open3D 0.17.0、ultralytics 8.2.x、Python 3.8）；
- 实验数值一律回填到 `experiment_log.md` 与 README 结果表，不入库任何需要运行才能得到的数值；
- 服务器排错先查 `troubleshooting.md`，再查官方 Issue。
