# Capstone 产品需求 · MiniKVLab

**这份文件只有需求，没有架构。** 几个文件？哪些类？cache 接口长什么样？benchmark 怎么组织？测试怎么分？——全部由你在 Day 78 的 `SPEC.md` 里决定。

## 一句话

一个可安装的 Python 包，实现一个小型 Llama 风格语言模型的自回归推理，支持可插拔的 KV cache 策略，并能用一条命令证明"哪种 cache 在什么条件下快多少、省多少内存、是否改变输出"。

## REQUIRED（毕业标准，缺一不可）

### 模型
- 可配置的 decoder-only Transformer：Embedding → N × Block → RMSNorm → LM head。
- Block：RMSNorm → 带 RoPE 的 Grouped-Query Attention → residual → RMSNorm → SwiGLU MLP → residual。
- `n_head` 与 `n_kv_head` 可独立配置（`n_kv_head == n_head` 与 `== 1` 都要能工作）。
- 能在 tiny shakespeare 上训练到 loss 明显下降（训练脚本可以很薄，不是重点）；checkpoint 内嵌 config。

### 推理
- 三个层次的 API：单次 `forward`；`prefill`（一次进多 token）与 `decode`（一次进 1 token）；`generate`（端到端）。
- 采样：greedy、temperature、top-k。固定 seed 可复现。
- 支持分块 prefill（cache 非空时再喂多 token）。

### Cache
- 统一接口（由你定义），至少含更新、读取、重置、当前长度四种能力。
- 三种实现：`DynamicCache`（按需增长）、`StaticCache`（预分配、原地写入、容量受限）、`SlidingWindowCache`（固定窗口）。
- 无 cache 的路径（每步全重算）作为 reference 保留。

### 正确性（必须有测试证明）
1. no-cache logits ≈ DynamicCache logits（同位置，`assert_close`）
2. DynamicCache ≈ StaticCache
3. greedy：NoCache tokens == Cached tokens（完全一致）
4. SlidingWindow 与"同窗口 reference attention"一致（**不与 full cache 比**——语义不同）
5. 分块 prefill == 一次 prefill
6. reset 无泄漏；Static 超容量报错；因果性；seed 确定性；CPU 与 MPS 一致（容差）

### Benchmark
- 一条 CLI 命令；`batch_size=1`；context ∈ {32, 64, 128, 256, 512, 1024}；`new_tokens=64`；方法 NoCache / Dynamic / Static / Sliding-256。
- 每组：5 warmup + 20 测量，报告中位数；MPS 上 `torch.mps.synchronize()`。
- 指标：prefill latency、decode latency / token（TPOT）、tokens/s、KV bytes（估算 + 实测）。
- 结果记录 device / dtype / model config / context；输出 CSV + README 表格 + 一段分析（数据是否符合理论：KV 线性、cached TPOT 平坦、Static vs Dynamic）。

### 工程
- `pyproject.toml` + `src/` layout，`pip install -e .` 可用；`pytest` 全绿，有快速路径（`-m "not slow"` < 10 s）。
- README 必须回答 7 个问题：什么是 KV cache？为什么自回归解码需要它？prefill 与 decode 有什么不同？Dynamic 与 Static 有什么区别？cache tensor 的 shape 是什么？理论 KV 内存怎么算？benchmark 是否验证了理论预期？——外加安装 / 测试 / benchmark 三条命令与架构图。
- `bugs.md` ≥ 5 条；`ai_log.md`；`git log` 可审计。

## Optional（REQUIRED 全部完成后才允许；Day 96 未开始则取消）

- `QuantizedCache`：K/V int8 + per-token scale；测试反量化误差与 greedy 一致率。
- `EvictionCache`（研究衔接）：预算 `K = S + R + H`，sink + recent + 累计 attention 分数 top-H；与同预算 SlidingWindow 比 greedy 一致率 / PPL / KV bytes / TPOT。注意：需要 eager attention 路径取分数；驱逐后保留原始绝对位置。
- HF adapter：让 SmolLM2-135M 使用你的 cache 类。
- `torch.compile`。

## Out of scope（明确不做）

server / API · continuous batching · paged attention · CUDA / Triton / Metal kernel · 分布式 · 大模型训练 · 权重格式转换。
