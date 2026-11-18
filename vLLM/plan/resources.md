# Resources · 资料（2026-11-17 全部核实可访问）

规则回顾：资料只占 20–25% 的时间；Phase I 的论文和参考实现都在你自己写完之后读。

## 论文（arXiv 编号与标题已核对）

| 何时读 | 论文 | 读哪部分 |
|---|---|---|
| Day 1 | **Orca**（Yu et al., OSDI 2022）：[usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu) | §1–3：iteration-level scheduling、selective batching |
| Day 1, 5 | **PagedAttention**：[arXiv 2309.06180](https://arxiv.org/abs/2309.06180) *Efficient Memory Management for Large Language Model Serving with PagedAttention* | Day 1 读 §1–3（问题），Day 5 读 §4（设计） |
| Day 15 | **Sarathi**：[arXiv 2308.16369](https://arxiv.org/abs/2308.16369) *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* | 分块 prefill 的思想 |
| 选读 | **Sarathi-Serve**：[arXiv 2403.02310](https://arxiv.org/abs/2403.02310) *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* | stall-free batching |
| Day 22 前后 | **SGLang**：[arXiv 2312.07104](https://arxiv.org/abs/2312.07104) *SGLang: Efficient Execution of Structured Language Model Programs* | RadixAttention 一节（前缀缓存的另一种设计：基数树 vs vLLM 的哈希） |
| Day 65 前后 | **DistServe**：[arXiv 2401.09670](https://arxiv.org/abs/2401.09670) · **Mooncake**：[arXiv 2407.00079](https://arxiv.org/abs/2407.00079) | prefill / decode 分离、以 KV cache 为中心的架构。只为理解 KV connector 存在的意义 |
| 不在范围内 | FlashAttention [arXiv 2205.14135](https://arxiv.org/abs/2205.14135)、投机解码 [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) | 知道它们解决什么问题即可 |

## vLLM 博客

- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release)（Day 15、36）：统一调度、V1 的进程结构
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)（Day 40）：从引擎到调度、分页注意力、前缀缓存、分块 prefill 的完整导览
- [vLLM Metal 公告（v0.28.0）](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)（Day 33）

（旧地址 `blog.vllm.ai/...` 会自动跳转到 `vllm.ai/blog/...`。）

## vLLM 仓库与文档

- 仓库：[github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- 贡献指南：[docs.vllm.ai/en/latest/contributing](https://docs.vllm.ai/en/latest/contributing/)（仓库内 `docs/contributing/README.md`）
- 安装文档（macOS CPU）：仓库内 `docs/getting_started/installation/cpu.apple.inc.md`；Apple GPU：`gpu.apple.inc.md`
- 设计文档：仓库内 `docs/design/`（清单见 [source-map.md](source-map.md)）
- Good first issues：[链接](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) · Onboarding tasks：[链接](https://github.com/orgs/vllm-project/projects/6)
- Slack：[slack.vllm.ai](https://slack.vllm.ai)（`#pr-reviews` 找 reviewer；vllm-metal 在 `#hw-metal`）

## 相关仓库

| 仓库 | 用途 | 注意 |
|---|---|---|
| [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) | 在 Apple Silicon GPU 上运行 vLLM 的社区硬件插件（MLX 后端）；第二个可以提 PR 的地方 | 用户安装：Homebrew；源码开发需要完整 Xcode 和 macOS SDK 26.2+（见它的 `docs/CONTRIBUTING.md`），要求 macOS 15+ |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 约 1200 行的 vLLM 简化实现：`nanovllm/engine/` 下的 `block_manager.py`（约 120 行）、`scheduler.py`（约 90 行）、`model_runner.py` | 依赖 CUDA，在 Mac 上只能读不能跑；**你自己写完对应模块之后**才读 |
| [LMCache/LMCache](https://github.com/LMCache/LMCache) | KV cache 层，通过 KV connector 接入 vLLM | Phase III Day 65 只看它和 vLLM 的接口位置 |

## 小模型（Hugging Face，已核实存在）

- `HuggingFaceTB/SmolLM2-135M-Instruct`：Llama 架构，CPU 上很快，Plan 1 Week 10 见过它的基座版
- `Qwen/Qwen2.5-0.5B-Instruct`
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`：给 vllm-metal 用

## 工具

- [uv](https://docs.astral.sh/uv/)：建 Python 3.12 环境
- [pre-commit](https://pre-commit.com/)：vLLM 的代码检查
- [GitHub CLI `gh`](https://cli.github.com/)：`gh pr checkout`、`gh pr view`、`gh pr checks`
