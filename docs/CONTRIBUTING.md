# 贡献指南（CONTRIBUTING）

欢迎为本项目贡献代码、文档或实验记录。请先阅读本文，再提 issue 或 PR。

## 1. 如何提 Issue

- 先搜索已有 issue，避免重复。
- 标题用一句话概括问题（如 `fix: 深度图读取在 16-bit PNG 下溢出`）。
- 正文包含：
  - 环境：Python 版本、PyTorch / Open3D / ultralytics 版本、是否 GPU、CUDA 版本；
  - 复现步骤：命令、输入文件路径、报错堆栈（贴关键 10 行即可）；
  - 期望行为与实际行为；
  - 如果是性能/精度问题，附上 `results/` 下对应实验产物或数值。
- 标签建议：`bug` / `enhancement` / `question` / `documentation` / `experiment`。

## 2. 分支与命名

- 主分支：`main`（受保护，禁止直接 push）。
- 新功能/修复一律从 `main` 拉分支，命名规则：
  - `feat/<简短描述>` —— 新功能、新模块
  - `fix/<简短描述>` —— 缺陷修复
  - `docs/<简短描述>` —— 文档
  - `refactor/<简短描述>` —— 重构
  - `exp/<简短描述>` —— 实验（消融、调参、数据集改动）
- 示例：`feat/registration-viewer`、`fix/pose-gt-inverse`、`exp/ablation-attention`。

## 3. Commit Message 格式

遵循仓库根目录 `.gitmessage` 的五类前缀（大写前缀 + 冒号 + 空格 + 小写开头描述）：

```
feat: 新增 FPFH 特征可视化脚本
fix: 修正 compute_pose_gt 的相对变换方向
docs: 补充 V100 环境搭建说明
refactor: 抽取公共点云预处理函数
exp: 记录 attention 融合消融实验配置
```

- 一个 commit 只做一件事；不混提交（如把文档改动混进 feat）。
- 需要补充说明时，正文列出"为什么改、怎么验证、影响范围"。
- 提交前运行 `bash scripts/check_env.sh --ci` 与本模块自检（如 `python detection/fusion_head.py`）。

## 4. PR 流程

1. 从 `main` 拉分支并提交改动。
2. 本地验证：
   - `python -m compileall -q preprocess registration detection app` 语法检查；
   - `bash scripts/run_smoke.sh --ci` 冒烟测试通过；
   - 涉及渲染时，确保 `--offscreen` 离屏模式可用（CI 无显示环境）。
3. 推送分支并创建 PR：
   - 标题复用 commit 前缀（如 `fix: ...`）；
   - 描述：改动内容、验证方式、`results/` 产物路径、是否影响已有实验。
4. CI 要求：`syntax-check`、`env-check`、`smoke` 三个 job 全部通过。
5. 至少 1 人 review 通过后由维护者 squash merge 到 `main`；合并后删除分支。

## 5. 数据与实验注意事项

- `data/`、`weights/`、`results/`、`external/` 被 `.gitignore` 忽略，**不要**把数据集、权重和大体积实验结果提交进仓库；实验结果请写入 `docs/experiment_log.md`。
- 实验类改动必须同时更新对应 `configs/` 配置与 `docs/experiment_log.md` 记录（日期、配置、指标、备注）。
- 不要提交任何 API key / token / 服务器凭据（仓库 CI 已开启 secret scanning）。
