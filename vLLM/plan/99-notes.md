# 99 · 核实记录、假设与未决问题

## 核实记录（2026-11-17，对照 `vllm-project/vllm` 的 `main`，v0.34.0 于 2026-11-16 发布）

| 内容 | 结果 |
|---|---|
| 贡献规范 | `docs/contributing/README.md`：DCO（`Signed-off-by`）、pre-commit、PR 标题标签、PR 模板（Overview / Claims / Validation / Details）、AI 辅助贡献规定（参与、有实质意义、披露、`Co-authored-by`）、review 节奏（2–3 天更新、7 天后可 ping）、无写权限贡献者最多 6 个 open PR、> 500 行需 RFC、CI 要求零落后、`/ci run` 权限 |
| macOS 支持 | CPU：实验性支持，必须从源码构建，FP32 / FP16，Xcode 15.4+ / Apple Clang ≥ 15，CI 在 Apple Silicon 上做冒烟测试。GPU：通过 vllm-metal 插件 |
| 测试能否在 CPU 上跑 | `tests/v1/core/test_prefix_caching.py`（7010 行）和 `test_scheduler.py`（9060 行）中没有 CUDA 依赖；`test_kv_cache_utils.py` 有 1 处提到 cuda。官方文档同时提醒"并非所有单元测试都能在 CPU 上通过" |
| 源码路径与名字 | `vllm/v1/core/` 下各文件及其中的类和方法（见 [source-map.md](source-map.md)）；`CommonAttentionMetadata` 的字段；`get_computed_blocks` 中"整段命中时设为 prompt_length − 1"的注释；`free_blocks` 的参数顺序约定 |
| 环境变量 | `VLLM_ENABLE_V1_MULTIPROCESSING`（默认 1）、`VLLM_CPU_KVCACHE_SPACE`（整数 GB，未设时 4）、`VLLM_LOGGING_LEVEL` |
| CLI | `vllm/entrypoints/cli/benchmark/` 下有 `latency`、`serve`、`throughput` 等 |
| 论文 | 8 个 arXiv 编号的标题逐一核对 |
| 博客 | 3 篇博客地址可访问，标题已核对 |
| 相关仓库 | vllm-metal（活跃）、nano-vllm（`scheduler.py` 把 prefill 与 decode 分成两种 step，分块 prefill 只对第一个请求开放）、LMCache |
| 模型 | 3 个小模型在 Hugging Face 上存在 |

## 关键假设

1. **入场时机**：开始本计划时 Plan 1 至少已通过 Gate 3。
2. **每天 2–3 小时**。Plan 1 的实际进度明显慢于计划；如果这里也一样，按失败协议整体后移，不压缩基础阶段。
3. **贡献方向限定在 Mac 能验证的范围**（调度器、KV cache 管理、前缀缓存、CPU 路径、测试、文档；可选 vllm-metal）。不依赖租用 GPU。
4. **vLLM 会继续快速变化**。到你开始时，部分文件可能已经改名或拆分。源码地图标了核实日期，找不到时按 source-map 顶部的方法定位，并记在下面。

## 未决问题（开始后用数据回答）

- 本地 CPU 构建上 `tests/v1/core/` 有多少测试失败、为什么？→ Day 31 的 `notes/test_baseline.md`
- 你的 macOS 版本能否安装 vllm-metal 源码开发要求的 Xcode / SDK 26.2+？→ 只有选 vllm-metal 做 PR #2 时才需要
- PR 实际需要多久得到 review？→ Phase IV 记录在 `contribution_log.md`，作为下一阶段的规划依据

## 源码变化记录

（开始执行后，每发现一处路径或名字与 source-map 不一致，就在这里记一行：日期、旧名字、新名字、怎么找到的。）
