# Phase V · 独立 Capstone：MiniKVLab（Day 78–100）

**目标**：证明"面对一个空目录，只查文档，我能把 LLM 推理 + KV cache 项目从需求做成可安装、有测试、有 benchmark、有文档的软件"。
**产物**：独立仓库 `minikvlab/`（在本目录下新建，或单独 `git init`，二选一但要说清楚）；`final_exam/`。
**Gate 5**：Day 98 审计 + Day 99–100 闭卷考。

## Capstone 规则（Day 78 起生效，到 Day 100）

1. **信息隔离**：不打开 Week 1–11 自己的代码；不打开 nanoGPT / gpt-fast / HF / minGPT 源码；不看教程。只允许 PyTorch / Python / pytest / packaging 官方文档。
2. **AI 政策切换为 Capstone 模式**（rules.md R4）：算法 / 接口 / 架构 / correctness bug ❌；API 语义 ✅；环境 / 打包 / traceback 含义 ✅ 但需先独立排查 ≥60 分钟；每次记 `ai_log.md`；从你自己的 SPEC 生成黑盒测试 ✅（实现前）。
3. **Scope freeze**：REQUIRED 未全部完成前，不碰任何 Optional。Optional 到 Day 96 未开始则自动取消。
4. **Debug 流程**：观察 → 假设 → 最小复现 → breakpoint/assert → 修改 → 测试。每个真 bug 进 `bugs.md`（毕业要求 ≥5 条）。
5. **git**：每天 ≥1 commit；`git log` 必须能看出 spec → tests → implementation → fix/refactor 的顺序。
6. 周五阅读与周日 Retrieval 停止；周日改为 Sprint 复盘（对照 SPEC 里程碑重排下周）。

## 范围（来自 capstone-spec.md，这里只列层级）

| 层级 | 内容 |
|---|---|
| **REQUIRED**（毕业标准） | Llama 风格 tiny 模型（RMSNorm / RoPE / GQA / SwiGLU）· prefill / decode / generate（greedy / temperature / top-k）· `DynamicCache` / `StaticCache` / `SlidingWindowCache` 统一接口 · 数值等价测试 · benchmark CLI · README 回答 7 个问题 |
| **Optional** | `QuantizedCache`（int8 per-token）· `EvictionCache`（研究衔接，见下）· HF adapter（让 SmolLM2 用你的 cache）· `torch.compile` |
| **Out of scope** | server · continuous batching · paged attention · CUDA / Triton kernel · 分布式 · 大模型训练 |

---

## Week 12 · Sprint 1：SPEC → 冻结测试 → 核心链路

### Day 78（周一）· 写 SPEC.md（今天不写代码）

读 [capstone-spec.md](capstone-spec.md)（那里只有产品需求）。然后写**你自己的** `SPEC.md`：
- 模块划分与每个文件的职责
- 公共接口：函数 / 类签名、每个 tensor 参数的 shape 契约（`[B, H_kv, T, D]` 这种）
- cache 接口与三种实现的行为差异（用表）
- 测试计划：每个算法假设对应哪个测试、reference 是什么
- benchmark 计划：指标、方法、输出格式
- 里程碑：Day 84 / 91 / 96 各要有什么
- 风险清单（你预计最容易出 bug 的三处）
commit：`docs: capstone spec`。

### Day 79（周二）· 冻结测试 + 基础层

- 从 SPEC 生成黑盒测试（AI 只看 SPEC）：cache 接口契约、模型 forward 契约、生成契约。检查、冻结、commit `test: freeze capstone acceptance tests`。
- 包骨架（`pyproject.toml`、`src/minikvlab/`）、`config.py`、`RMSNorm` / `SwiGLU` / RoPE + 各自单元测试。

### Day 80（周三）· Attention + 模型

GQA + RoPE + cache 钩子的 attention；`TransformerLM.forward`（无 cache）；shape 测试；初始 loss ≈ ln(V)；overfit-one-batch。

### Day 81（周四）· prefill / decode + DynamicCache

`generation.py` 的 `prefill / decode / generate`；`DynamicCache`；**等价测试全绿**（同位置 logits `assert_close`，greedy 完全一致）。

### Day 82（周五）· StaticCache + 分块 prefill

`StaticCache`（预分配、in-place、容量错误、`data_ptr` 不变）；分块 prefill 等价测试；通用 mask。

### Day 83（周六）· SlidingWindowCache + 采样 + CLI

`SlidingWindowCache` 与窗口 reference 测试；greedy / temperature / top-k；`python -m minikvlab.cli generate ...` 端到端跑通（随机权重也行）。

### Day 84（周日）· Sprint 1 复盘

对照 SPEC 里程碑：REQUIRED 完成了哪些？等价测试是否全绿？bugs.md 有几条？重排 Week 13。写 `logs/weekly_review.md`。

---

## Week 13 · Sprint 2：训练权重 → benchmark → 测试加固 → README

### Day 85（周一）· 得到非随机权重
在 tiny shakespeare 上训练几分钟（MPS），保存 checkpoint（内嵌 config）。这是 benchmark 与 PPL 有意义的前提。训练脚本尽量薄——它不是本项目重点。

### Day 86（周二）· benchmark 模块 + CLI
warmup 5 / 测 20 取中位数 / `torch.mps.synchronize()` / CSV；context∈{32,64,128,256,512,1024}，new_tokens 64；方法 NoCache / Dynamic / Static / Sliding-256；输出 prefill_ms、decode_ms_per_token、tokens_per_s、kv_bytes；记录 device / dtype / config。

### Day 87（周三）· 测试加固
invariant / regression / reset 无泄漏 / 容量 / CPU-MPS 一致；`conftest.py`；`pytest -m "not slow"` < 10 s。

### Day 88（周四）· 预留 debug 日
bug 一定会有。若今天没有 bug 可修：检查 stretch 解锁条件（REQUIRED 全绿 + benchmark 能跑 + README 核心段已写），提前进入 Day 92 的打磨。

### Day 89（周五）· README
必须回答 7 个问题：什么是 KV cache？为什么自回归解码需要它？prefill 与 decode 有何不同？Dynamic 与 Static 有何区别？cache tensor 的 shape？理论 KV 内存怎么算？benchmark 是否验证了理论预期？——外加安装 / 运行测试 / 运行 benchmark 三条命令，和 ASCII 架构图。

### Day 90（周六）· 完整 benchmark 跑一遍 + 分析
`benchmarks/results.csv` + README 里的表 + 一段分析：KV bytes 是否线性？cached TPOT 是否平坦？Static 是否比 Dynamic 稳？哪里和理论不符、为什么？

### Day 91（周日）· Sprint 2 复盘 + stretch go/no-go
只有 REQUIRED **全部**完成才允许 stretch。否则 Week 14 全部用于打磨与修补。

---

## Week 14 · Sprint 3：打磨 → （stretch）→ fresh-clone → release

### Day 92（周一）· 代码质量
typing 过一遍（`def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor`）；`ruff check --fix && ruff format`；删 dead code / magic number / debug print / notebook 残留；公共 API docstring。

### Day 93–94（周二、周三）· Stretch（若解锁）或继续打磨

**EvictionCache（研究衔接，推荐）**——固定预算 `K = S + R + H`：`S` 个 sink token + 最近 `R` 个 + 历史中按**累计 attention 分数**保留 top-`H`。累计分数 `s_i ← s_i + mean_heads(A[t, i])`。这是受 StreamingLLM / H2O 思路启发的**教学版**驱逐策略，README 里不要写成"复现 H2O"。
- 两个已知的坑：(1) `F.scaled_dot_product_attention` 拿不到 attention 概率——用你自己的 eager attention 路径取 `[B,H,1,T]`，**不要为它改主工程的 attention 架构**；(2) 驱逐后**必须保留幸存 token 的原始绝对位置**，不能重新编号成 0..K-1，否则 RoPE 语义变了。
- 对比实验（Day 94）：`FullCache` vs `SlidingWindow(K)` vs `EvictionCache(K)`（同预算），指标：validation NLL/PPL、**greedy 一致率** `#{t: argmax p_compressed == argmax p_full} / N`、KV bytes、TPOT。结论只需回答："同预算下，score-based 策略是否比纯滑窗更少破坏原模型输出？"

**QuantizedCache（替代选项）**：K/V 存 int8 + per-token scale，`get` 时反量化；测试：反量化误差上界、greedy 一致率。

### Day 95（周四）· Fresh-clone 测试
`git clone` 到一个新目录，**只按 README** 建环境、安装、`pytest`、跑 benchmark。任何卡住的地方都是 README 的 bug——修掉。

### Day 96（周五）· Release
最终 README / `CHANGELOG.md`；`git tag v0.1.0`。Optional 未开始的自动取消。

### Day 97（周六）· ⏸ Buffer

### Day 98（周日）· 🚪 Gate 5 ①：项目审计
项目冻结（之后不再改）。逐项过 [gates.md → Gate 5](gates.md#gate-5) 的 1–4、6–10 项。`git tag gate-5-audit`。

### Day 99（周一）· 🚪 Gate 5 ②：毕业考 Part 1（3 小时）
新空目录 `final_exam/day99/`，只查文档：`RMSNorm / RoPE / GQA CausalSelfAttention / SwiGLU MLP / Block / TransformerLM` + shape 测试 + 初始 loss 测试。

### Day 100（周二）· 🚪 Gate 5 ③：毕业考 Part 2（3 小时）
`final_exam/day100/`（可以 import Day 99 的模型）：`DynamicKVCache / cached forward / prefill / decode / greedy generation / cache-vs-no-cache 等价测试 PASS`。必考到此为止；benchmark 是可选加分，不计 Pass/Fail。

然后写 `logs/final_reflection.md`：Day 1 与 Day 100 的空白文件延迟；100 天里最有价值的 3 个 bug；下一个你想独立做的项目是什么、第一步是什么。
