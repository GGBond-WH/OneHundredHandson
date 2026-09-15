# Phase IV · 测试加固 / 真实 HF 模型 / 打包 / benchmark（Day 57–77）

**目标**：从"会写算法代码"升级为"会维护一个小型项目"：会**设计**测试（而不只是会写）、能对接真实 LLM 并验证理论、能打包与 benchmark。
**产物**：`week09_tests/`（加固后的测试套件 + Bug Hunt）、`week10_hf/`（SmolLM2-135M 观察与手动 decode）、`mini_kv/`（`pyproject.toml` + `src/` layout + benchmark CLI 的可安装包）。
**Gate 4**：Day 75–76。

---

## Week 9 · 测试加固周（Testing Hardening，不是"首次学测试"）

从"会写测试"到"会设计测试"：算法声称什么性质 → 什么可观察量能证明它 → 怎样构造让错误实现必挂的反例。

### Day 57（周一）· 读 llama 源码 30 min + 测试审计

【读】(30 min) [HF `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) 的 `rotate_half / apply_rotary_pos_emb / repeat_kv`，对照你的 `rope.py` 与 GQA（`repeat_kv` 用 `expand + reshape`——为什么不用 `repeat_interleave`？）。

【写】`notes/test_audit.md`：把现有测试分为 unit / invariant / equivalence / regression 四类；列出**每个算法假设**（因果、位置、cache 顺序、窗口语义、容量、reset、确定性、设备一致）并标出哪个还没有对应测试。

### Day 58（周二）· 补 invariant 测试

【写】`tests/test_invariants.py`：模型级因果 + cache；分块 prefill；reset 后 prefill 与新建 cache 一致（无泄漏）；sliding vs 窗口 reference；static 超容量 `pytest.raises`；同 seed 确定性；CPU vs MPS 在 `atol=1e-4` 内一致（MPS 用 float32）。

### Day 59（周三）· 失败测试先行 + 回归

【写】从 `bugs.md` 挑一个真实 bug：`git log` 找到修复前的 commit，`git checkout <sha>`（看完 `git checkout -`  回来）验证你写的回归测试在旧代码上**失败**、在新代码上通过。
【读】pytest：`-x --lf --ff`、`pytest.raises`、`pytest.approx`、`tmp_path` fixture。

### Day 60（周四）· 📐 Spec #5：Bug Hunt

打开 `week09_bughunt/README.md`。`faulty_kv.py` 是一份**外部提供**的、埋了 4 个 bug 的 KV cache 实现（你不知道 bug 在哪）。规则：
1. **不改代码**，先写测试；测试必须让 bug 暴露（至少 4 个失败的测试）。
2. 再定位、再修，每个 bug 记 bugs.md。
3. 最后解封 `answers.b64` 对答案（`base64 -d answers.b64`）。
若 GQA Spec 从 Day 53 顺延到今天：今天做 GQA，Bug Hunt 移到周六。

### Day 61（周五）· 📖 pytest 文档 + minbpe 测试

【读】[pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)、[parametrize](https://docs.pytest.org/en/stable/how-to/parametrize.html)、[markers](https://docs.pytest.org/en/stable/how-to/mark.html)；[minbpe `tests/test_tokenizer.py`](https://github.com/karpathy/minbpe/blob/master/tests/test_tokenizer.py)——看别人怎么围绕**行为契约**写测试（不是围绕实现）。
【记】`notes/pytest_patterns.md`：3 个要采用的模式。

### Day 62（周六）· 🧪 测试基础设施

`conftest.py`：`tiny_config / tiny_model / seeded` fixtures；markers `slow`、`mps`（`pytest.mark.skipif(not torch.backends.mps.is_available())`）；`pytest -m "not slow"` < 10 s；`pyproject` 里配置 markers。可选：`pytest-cov` 看看哪些分支没覆盖。

### Day 63（周日）· 🔁 Retrieval #9（60 min）

空白写一个测试模块（≤100 LOC）：给定 `model(idx, cache=None)` 接口，写 cache/no-cache 等价测试 + 分块 prefill 测试 + reset 无泄漏测试。

---

## Week 10 · 真实 HF 模型：SmolLM2-135M（只推理，不训练）

为什么是它：135M 参数、Llama 架构、`num_attention_heads=9, num_key_value_heads=3`（GQA 3:1）、30 层、head_dim 64——正是你刚写完的东西的工业版本，M4 Pro 上很轻松。

### Day 64（周一）· 加载与观察 cache

【做】`uv pip install transformers`（或 pip）。
【写】`week10_hf/inspect_cache.py`
```python
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32).eval()
out = model(**tok("The cat sat", return_tensors="pt"), use_cache=True)
# 打印 model.config 中 num_hidden_layers / num_attention_heads / num_key_value_heads / hidden_size
# 遍历 out.past_key_values：每层 K/V shape（期望 [1, 3, T, 64]）、get_seq_length()
```
【记】和你的 cache 布局对比：一样是 `[B, H_kv, T, D]` 吗？seq_len 怎么取？

### Day 65（周二）· 手动 prefill + decode == generate

【写】`manual_decode.py`：`model(input_ids, use_cache=True)` → 循环喂 `next_token[:, None]` + `past_key_values` + `cache_position`/`position_ids`（查 HF 文档决定用哪个）→ greedy。
【测】与 `model.generate(input_ids, do_sample=False, max_new_tokens=20)` 的 token 序列**完全一致**。再试 `model.generate(..., cache_implementation="static")`。

### Day 66（周三）· tokenizer 对比 + KV 内存实测

【写】(30 min) 用 tiny shakespeare 训练你的 `BasicTokenizer(vocab_size=1024)`，与 SmolLM2 tokenizer 在同一段文本上比较：token 数、`bytes→token` 的差异、special tokens、为什么不同。记 `notes/tokenizer_compare.md`。
【写】KV 内存：对 T∈{128,512,2048} 各跑一次 prefill，`Σ numel×element_size` 汇总；验证随 T 线性；float32 vs bfloat16 各测一次。做成表。

### Day 67（周四）· 📐 Spec #6：`estimate_kv_bytes`

见 [specs.md → Spec #6](specs.md#spec-6)。公式不给；验收是与 Day 66 的真实 tensor 数据吻合。

### Day 68（周五）· 📖 HF 的 cache 路径（导航式，≤45 min）

`LlamaAttention.forward` 里 `past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)` 的调用；`generation/utils.py` 的 `_sample` 循环骨架（只看：每步输入怎么准备、`cache_position` 怎么递增）。**不通读**。
【记】`notes/hf_cache_path.md`：用 10 行以内写清一个 token 从 `generate` 到 `cache.update` 的路径。

### Day 69（周六）· 🧪 SmolLM2 benchmark

`bench_smollm.py`：MPS vs CPU，context∈{128,512,1024}，new_tokens 64，TPOT 与 tokens/s；warmup + `synchronize`；记录 device / dtype / model / context。观察：MPS 上 TPOT 相对 CPU 快多少？context 增大时 TPOT 变化大吗（cached 应接近平坦）？

### Day 70（周日）· 🔁 Retrieval #10（60 min）

空白写对 HF 模型的手动 prefill + decode 循环（≤60 LOC），greedy 与 `generate` 一致。

---

## Week 11 · 打包 + benchmark 框架

### Day 71（周一）· pyproject + src layout

【读】[PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) 的 `pyproject.toml` 与 `src/` layout 部分。
【写】`mini_kv/`：
```
mini_kv/
├── pyproject.toml      # [project] name/version/dependencies; [tool.ruff]; [tool.pytest.ini_options]
├── README.md
├── src/mini_kv/{__init__,config,model,attention,rope,layers,generation,benchmark,cli}.py
├── src/mini_kv/cache/{base,dynamic,static,sliding}.py
└── tests/
```
`uv pip install -e .`（或 `pip install -e .`）；测试从安装的包 import（不改 `sys.path`）；`python -m mini_kv.cli generate --ckpt ... --prompt ...`。

### Day 72（周二）· benchmark 框架

【写】`benchmark.py`：`@dataclass BenchmarkConfig(cache: str, context: int, new_tokens: int, device, dtype, warmup=5, runs=20)`；`@dataclass BenchmarkResult(prefill_ms, decode_ms_per_token, tokens_per_s, kv_bytes, ...)`；`timeit(fn)` 内部 warmup + synchronize + 取中位数；结果写 CSV。方法：NoCache / Dynamic / Static / Sliding-256。

### Day 73（周三）· 内存指标 + 结果表

估算（Spec #6 的函数）vs 实测 `Σ numel×element_size`；MPS 上 `torch.mps.current_allocated_memory()` 前后差值作为辅助指标（注意它不等于进程真实占用）。README 结果表必须写明 device / dtype / config / context。检查：cached TPOT 随 context 基本平坦？naive 明显增长？Static 比 Dynamic 更稳？

### Day 74（周四）· 📐 Spec #7：benchmark CLI contract

见 [specs.md → Spec #7](specs.md#spec-7)。验收含"新增一种 cache 不改 benchmark 核心逻辑"（用一个 dummy cache 类注册后跑通验证）。

### Day 75–76（周五、周六）· 🚪 Gate 4

见 [gates.md → Gate 4](gates.md#gate-4)。`git tag gate-4`。

### Day 77（周日）· ⏸ Buffer

Gate 4 通过后，读一遍 [capstone-spec.md](capstone-spec.md) 与 [phase5.md](phase5.md) 的规则，明天开始独立项目。**今天不要开始设计。**
