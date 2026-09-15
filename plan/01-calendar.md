# 01 · 100 天日历（Day 1 = 周一）

每天一行：**做什么** → 当天产物。详细任务、资料、验收见对应 phase 文件。
标记：🔨 Build · 📐 Spec Challenge · 📖 限定范围读源码 · 🧪 测试/benchmark/refactor · 🔁 Retrieval Drill · 🚪 Gate · ⏸ Buffer

## Phase I · Python + PyTorch 最小工程能力（[phase1.md](phase1.md)）

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 1 | 一 | 🔨 Developer Workbench：环境、torch/MPS 验证、git init、VS Code pytest + 调试器、断点看 shape 演练 | `day01_workbench/`，首个 commit |
| 2 | 二 | 🔨 class / dataclass / typing → `CharTokenizer` + 最小 pytest | `week01_text_pipeline/tokenizer.py`, tests |
| 3 | 三 | 🔨 iterator / generator / pathlib → `TextDataset` + `batch_iterator` | `dataset.py`, tests（x/y shift） |
| 4 | 四 | 🔨 模块与包、argparse、pytest fixture → 组装成可 `python -m` 运行的包 | 带 CLI 的 `text_pipeline/` |
| 5 | 五 | 🔨 Basic BPE ①：bytes、pair 统计、merge、train | `bpe.py` 前半 + tests |
| 6 | 六 | 🔨 Basic BPE ②：encode / decode / `BasicTokenizer` 类 + tests；**写完后**📖 minbpe `basic.py` | `bpe.py` 完成，`notes/minbpe.md` |
| 7 | 日 | 🔁 60 min：空白重写 `CharTokenizer` + `batch_iterator`（≤100 LOC）+ 3 tests | `scratch/w1_retrieval/` |
| 8 | 一 | 🔨 Tensor 基础（shape/view/reshape/transpose/contiguous/broadcast）+ Tensor Puzzles 1–7 | `week02_bigram/tensor_drills.py` |
| 9 | 二 | 🔨 autograd / nn.Module / Embedding / cross_entropy → `BigramLM` + Tensor Puzzles 8–14 | `model.py`, tests |
| 10 | 三 | 🔨 AdamW 训练循环、train/eval、no_grad、device → `train.py`（tiny shakespeare）+ Tensor Puzzles 15–21 | 训练 loss 下降曲线 |
| 11 | 四 | 🔨 `generate()`（multinomial / greedy）、save / load checkpoint | `generate.py`, 存取 tests |
| 12 | 五 | 📖 makemore bigram 对照 → 🧪 整理 `mini_lm/` 结构、ruff、补测试；复读 Gate 1 清单 | 干净的 `mini_lm/` |
| 13 | 六 | 🚪 **Gate 1**（3 h）：空目录只查文档写 Bigram 项目 + 5 个必测 | `gate1/`，tag `gate-1` |
| 14 | 日 | ⏸ Buffer：Pass 则休息/复盘；Recoverable 则定向修 + 复测；Structural 则启动 +2 天协议 | |

## Phase II · 从零写 Decoder-only Transformer（[phase2.md](phase2.md)）

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 15 | 一 | 🔨 attention 版本 A：显式循环（先均值聚合，再 q/k/v）+ 因果性测试 | `week03_attention/attn_loop.py` |
| 16 | 二 | 🔨 版本 B：矩阵形式（scale / tril mask / masked_fill / softmax）与 A 数值对比 | `attn_matrix.py` |
| 17 | 三 | 🔨 版本 C：多头 `CausalSelfAttention(nn.Module)`，parametrize 测试，与 SDPA 对比 | `attention.py`, tests |
| 18 | 四 | 🔨 纸上推导 B=2,T=8,C=32,H=4 每步 shape 再验证；dropout；返回 attention 权重的选项 | `notes/shapes.md` |
| 19 | 五 | 📖 SDPA 文档 + minGPT `CausalSelfAttention`，写 3 条设计差异 | `notes/mingpt_attn.md` |
| 20 | 六 | 🧪 性质测试（因果、softmax 行和、batch 独立）；loop vs matrix vs SDPA 微 benchmark（首次用 `torch.mps.synchronize()`） | `bench_attn.py` |
| 21 | 日 | 🔁 60 min：空白重写多头 `CausalSelfAttention` + 因果测试 | `scratch/w3_retrieval/` |
| 22 | 一 | 🔨 手写 LayerNorm（与 nn.LayerNorm 对比）+ MLP（Linear-GELU-Linear） | `week04_gpt/layers.py` |
| 23 | 二 | 🔨 `Block`（pre-LN + residual）+ `GPT`（tok emb + pos emb + N×Block + LN + LM head）+ shape/初始 loss 测试 | `model.py` |
| 24 | 三 | 🔨 在 tiny shakespeare 上训练 TinyGPT（MPS），loss 4.2 → <2.0；naive 生成 | 训练曲线、样本文本 |
| 25 | 四 | 🔨 weight tying、init、参数量统计、val loss、`eval()`/`no_grad`；写自己的 `ARCHITECTURE.md` | `ARCHITECTURE.md` |
| 26 | 五 | 📖 **现在才允许**读 nanoGPT `model.py`，写 5 条差异及原因 | `notes/nanogpt_diff.md` |
| 27 | 六 | 🧪 parametrize 模型测试、模型级因果测试、CPU 2 步训练 smoke test、ruff | 全绿测试套件 |
| 28 | 日 | 🔁 60 min：空白重写 `Block` + `GPT.forward` | `scratch/w4_retrieval/` |
| 29 | 一 | 🔨 `@dataclass GPTConfig`、seed、device 选择、argparse `train.py`、loss 记 CSV | `week05_tinygpt/` |
| 30 | 二 | 🔨 checkpoint 内嵌 config；`generate.py` CLI；temperature + top-k 采样 | 可用的 CLI |
| 31 | 三 | 🔨 采样测试（T→0≈greedy、top_k=1==greedy、seed 可复现）；整理项目结构 | tests |
| 32 | 四 | 📐 **Spec #1 `top_p_sample`**：读 spec → AI 生成黑盒测试 → 冻结 → 实现 | `sampling.py` + frozen tests |
| 33 | 五 | 🚪 **Gate 2 ①**：给自己的 tokenizer/data/train harness，`model.py` 从空白写（3 h 内） | `gate2/` |
| 34 | 六 | 🚪 **Gate 2 ②**：60 min 指定模块闭卷重构 + 不运行代码写出全部 shape + 清单自评 | tag `gate-2` |
| 35 | 日 | ⏸ Buffer | |

## Phase III · Prefill / Decode / KV cache（[phase3.md](phase3.md)）

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 36 | 一 | 🔨 测 naive 生成每 token 耗时随 T 增长；**先写一页**"forward 要改什么才能支持 cache"再动手 | `notes/cache_plan.md`，耗时曲线 |
| 37 | 二 | 🔨 `forward(idx, past_kv=None, use_cache=False)`：attention 拼接 past K/V、位置偏移 past_len、decode 步 mask；cache = `list[tuple[K,V]]` | `model.py` 支持 cache |
| 38 | 三 | 🔨 `generate_cached`（prefill 一次 + 逐 token decode）；**等价测试**：同位置 logits `assert_close`、greedy 序列完全一致 | 等价测试全绿 |
| 39 | 四 | 📐 **Spec #2 `AutoregressiveDecoder`** 状态机（prefill / step / reset，非法顺序抛异常） | `decoder.py` + frozen tests |
| 40 | 五 | 📖 gpt-fast `generate.py`：`prefill / decode_one_token / decode_n_tokens / generate`，看 `input_pos` 怎么传 | `notes/gptfast_generate.md` |
| 41 | 六 | 🧪 naive vs cached benchmark（CPU & MPS，T∈{64,128,256,512}，TPOT）；KV 内存 = Σ numel×element_size | `bench_generate.py`, 结果表 |
| 42 | 日 | 🔁 60 min：空白重写"带 past K/V 拼接 + 位置偏移"的 attention forward | `scratch/w6_retrieval/` |
| 43 | 一 | 🔨 设计 `KVCache` 接口（update / get / reset / seq_len）+ 任何 cache 都要过的契约测试；list[tuple] 重构为 `DynamicCache` | `cache/base.py`, `dynamic.py`, `test_cache.py` |
| 44 | 二 | 🔨 模型改为接受 cache 对象；旧路径全绿；契约测试 parametrize 到 cache 类 | 重构完成 |
| 45 | 三 | 🔨 分块 prefill：q 长 T、kv 长 past+T 时的 mask 偏移；测试"两块 prefill == 一块 prefill" | 通用 mask + tests |
| 46 | 四 | 📐 **Spec #3 `StaticCache`**：只给接口 + 黑盒测试（不许 cat、reset 不重分配、超容量报错、data_ptr 不变） | `cache/static.py` |
| 47 | 五 | 📖 HF `cache_utils.py` 的 `CacheLayerMixin / DynamicLayer / StaticLayer / DynamicCache / StaticCache`（导航式 ≤1 h）→ 写 `design_comparison.md` | `notes/design_comparison.md` |
| 48 | 六 | 🧪 `SlidingWindowCache`（与"同窗口 reference attention"比，不与 full cache 比）；Dynamic vs Static TPOT | `cache/sliding.py`, 结果 |
| 49 | 日 | 🔁 60 min：给定预分配 cache[B,H,max,D]、pos、new_k，不用 cat 完成更新 + seq_len 记账 | `scratch/w7_retrieval/` |
| 50 | 一 | 🔨 前 1 h：RMSNorm + SwiGLU（各几行）；之后 RoPE：`precompute_freqs` / `apply_rope(x, positions)` + 范数不变、相对位置性质测试 | `rope.py`, `layers.py` |
| 51 | 二 | 🔨 RoPE 替换学习式位置嵌入（positions = arange(past_len, past_len+T)）；短训验证 loss 下降；**cache 等价测试必须重新全绿** | Llama 风格模型变体 |
| 52 | 三 | 🔨 **Readiness checkpoint**：RoPE + Dynamic + Static + 分块 prefill 等价测试全绿？绿→准备 GQA；红→周四继续修，GQA Spec 顺延到 Day 60 | 检查记录 |
| 53 | 四 | 📐 **Spec #4 `GroupedQueryAttention`**：只给数学规格（MHA/MQA 特例、KV 字节随 n_kv_heads 下降） | `attention.py` GQA |
| 54 | 五 | 🚪 **Gate 3 ①**：prefill/decode/Dynamic/Static/position/等价 | `gate3/` |
| 55 | 六 | 🚪 **Gate 3 ②**：清单剩余项（off-by-one 测试、内存解释、最小 latency 对比）+ 自评 | tag `gate-3` |
| 56 | 日 | ⏸ Buffer | |

## Phase IV · 测试加固 / 真实模型 / 打包 / benchmark（[phase4.md](phase4.md)）

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 57 | 一 | 📖 30 min HF `modeling_llama.py` 的 `apply_rotary_pos_emb` / `repeat_kv` 对照自己的；🔨 测试分类（unit / invariant / equivalence / regression），列出没有测试的算法假设 | `notes/test_audit.md` |
| 58 | 二 | 🔨 补 invariant 测试：模型级因果+cache、分块 prefill、reset 不泄漏、sliding vs 窗口 reference、static 容量、seed 确定性、CPU vs MPS 一致（容差） | `tests/` 扩充 |
| 59 | 三 | 🔨 失败测试先行：从 bugs.md 挑一个 bug，写在旧 commit 上失败、现在通过的回归测试（`git checkout <old>` 验证）；学 `-x --lf`、`pytest.raises`、`approx`、`tmp_path` | 回归测试 |
| 60 | 四 | 📐 **Spec #5 Bug Hunt**：`week09_bughunt/faulty_kv.py`（4 个隐藏 bug）：先写测试抓 bug → 定位 → 修 → 对答案（若 GQA 顺延则今天做 GQA，Bug Hunt 移到周六） | `week09_bughunt/tests/`, 修复 diff |
| 61 | 五 | 📖 pytest 文档 fixtures / parametrize / marks + minbpe `tests/` 作为行为驱动测试范例，记 3 个可采用的模式 | `notes/pytest_patterns.md` |
| 62 | 六 | 🧪 `conftest.py` fixtures（tiny 模型、seed）、`slow` / `mps` markers，`pytest -m "not slow"` < 10 s | 快速测试路径 |
| 63 | 日 | 🔁 60 min：空白写"cache/no-cache 等价 + 分块 prefill"测试模块（≤100 LOC） | `scratch/w9_retrieval/` |
| 64 | 一 | 🔨 安装 transformers，加载 SmolLM2-135M，greedy 生成；打印 config（30 层 / 9 头 / 3 KV 头 / head_dim 64）和每层 `past_key_values` K/V shape、seq_len | `week10_hf/inspect_cache.py` |
| 65 | 二 | 🔨 用 HF 模型手写 prefill + 逐 token decode 循环，greedy 结果 == `model.generate(do_sample=False)`；试 `cache_implementation="static"` | `manual_decode.py` + test |
| 66 | 三 | 🔨 30 min 自己的 Basic BPE vs SmolLM2 tokenizer 对比；KV 内存实测 T∈{128,512,2048} 验证线性增长、dtype 影响 | `notes/tokenizer_compare.md`, 内存表 |
| 67 | 四 | 📐 **Spec #6 `estimate_kv_bytes`**：与真实 tensor numel×element_size 吻合（公式不给） | `kv_estimator.py` + frozen tests |
| 68 | 五 | 📖 导航式读 HF `LlamaAttention.forward` → `past_key_values.update` 与 `generation/utils.py` `_sample` 骨架（≤45 min），看 `cache_position` 怎么流 | `notes/hf_cache_path.md` |
| 69 | 六 | 🧪 SmolLM2 decode benchmark（MPS vs CPU，TPOT vs context，synchronize，记录 device/dtype） | `bench_smollm.py`, 结果表 |
| 70 | 日 | 🔁 60 min：空白写对 HF 模型的手动 prefill+decode 循环（≤60 LOC），greedy 与 generate 一致 | `scratch/w10_retrieval/` |
| 71 | 一 | 🔨 `pyproject.toml` + `src/mini_kv/` layout，`pip install -e .`，`python -m mini_kv.cli`，ruff 配置，README 骨架 | 可安装的包 |
| 72 | 二 | 🔨 benchmark 框架：`BenchmarkConfig` / `BenchmarkResult`、warmup 5 + 测 20 取中位数、synchronize、CSV | `benchmark.py` |
| 73 | 三 | 🔨 内存指标（估算 vs 实测、`torch.mps.current_allocated_memory()` 差值）；结果表进 README（记录 device/dtype/config）；cached TPOT 是否平坦、naive 是否增长 | README 结果表 |
| 74 | 四 | 📐 **Spec #7 benchmark CLI contract**：`python -m mini_kv.benchmark --cache dynamic --context 512 --new-tokens 64`；新增 cache 不改核心逻辑（用 dummy cache 验证） | CLI + frozen tests |
| 75 | 五 | 🚪 **Gate 4 ①**：包安装、三类测试、真实模型 cache 检查、估算器吻合 | `gate4/` 记录 |
| 76 | 六 | 🚪 **Gate 4 ②**：benchmark 项、统一接口项、失败测试流程项、README、Spec #7 冻结测试 | tag `gate-4` |
| 77 | 日 | ⏸ Buffer | |

## Phase V · 独立 Capstone：MiniKVLab（[phase5.md](phase5.md)）

从 Day 78 起：新建独立仓库 `minikvlab/`，**不打开 Week 1–11 的代码**，不打开任何参考实现，AI 政策切换为 Capstone 模式。

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 78 | 一 | 读 [capstone-spec.md](capstone-spec.md)（只有需求）→ 写自己的 `SPEC.md`：模块、接口、张量契约、测试计划、benchmark 计划、里程碑 | `SPEC.md`，commit |
| 79 | 二 | 从 SPEC 生成并冻结验收测试（cache 接口 / 模型契约）；包骨架 + config + RMSNorm / SwiGLU / RoPE + 单元测试 | frozen tests, 基础层 |
| 80 | 三 | Attention（GQA + RoPE + cache 钩子）；模型 forward（无 cache）；overfit-one-batch 测试 | `model.py` |
| 81 | 四 | prefill / decode + DynamicCache + 等价测试全绿 | `generation.py`, `cache/dynamic.py` |
| 82 | 五 | StaticCache + 契约测试；分块 prefill | `cache/static.py` |
| 83 | 六 | SlidingWindowCache + 窗口 reference 测试；greedy / temperature / top-k；首个端到端 generate CLI | `cache/sliding.py`, `cli.py` |
| 84 | 日 | Sprint 1 复盘：对照 SPEC 里程碑；bugs.md；重排 Week 13 | `logs/weekly_review.md` |
| 85 | 一 | 在 tiny shakespeare 上训练几分钟得到非随机权重（benchmark / PPL 需要）；checkpoint 格式 | `checkpoints/` |
| 86 | 二 | benchmark 模块（warmup / 中位数 / synchronize / CSV）+ CLI | `benchmark.py` |
| 87 | 三 | 测试加固：invariant / regression / reset / 容量 / CPU-MPS 一致；`pytest -m "not slow"` 快速路径 | 全绿套件 |
| 88 | 四 | 预留 debug 日（bug 一定会有）；顺便检查 stretch 解锁条件 | bugs.md |
| 89 | 五 | README 草稿：回答 7 个问题；ASCII 架构图；结果表位置 | `README.md` |
| 90 | 六 | 完整 benchmark 跑一遍；分析数据是否符合理论（KV 线性、cached TPOT 平坦） | `benchmarks/results.csv`, 分析段 |
| 91 | 日 | Sprint 2 复盘；决定 stretch go / no-go（仅当 REQUIRED 全部完成） | 复盘记录 |
| 92 | 一 | typing 过一遍、ruff、删 dead code / magic number / debug print；公共 API docstring | 干净代码 |
| 93 | 二 | Stretch（若解锁）：EvictionCache 或 QuantizedCache；否则继续打磨 | `cache/eviction.py` |
| 94 | 三 | Stretch 对比实验：Full vs Sliding(K) vs Eviction(K) 的 greedy 一致率 + PPL；或打磨 | 对比表 |
| 95 | 四 | **Fresh-clone 测试**：克隆到新目录，只按 README 安装、跑测试、跑 benchmark，修掉一切缺口 | 修复 commit |
| 96 | 五 | 最终 README / CHANGELOG；tag `v0.1.0` | release |
| 97 | 六 | ⏸ Buffer | |
| 98 | 日 | 🚪 **Gate 5 ① 审计**：项目冻结，逐项过清单（1–4, 6–10），tag `gate-5-audit` | 审计记录 |
| 99 | 一 | 🚪 **Gate 5 ② 毕业考 Part 1**（3 h）：新空目录只查文档：RMSNorm / RoPE / GQA Attention / MLP / Block / TransformerLM + shape 测试 | `final_exam/day99/` |
| 100 | 二 | 🚪 **Gate 5 ③ 毕业考 Part 2**（3 h）：DynamicKVCache / cached forward / prefill / decode / greedy / **等价测试 PASS**（benchmark 为可选加分）；写最终复盘：Day 1 vs Day 100 的空白文件延迟 | `final_exam/day100/`, `logs/final_reflection.md` |
