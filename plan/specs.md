# Specs · 每周四 Spec Challenge（W5–W11）

**这条轨道训练的是：需求 → 抽象 → 接口 → 算法 → 测试 → 实现。** 与周日的 Retrieval（复现自己写过的东西）完全不同。

三条硬规则：**没有参考实现、没有教程链接、没有 starter 代码。** 只有需求、签名、示例、验收测试。随周数增加，给你的信息越来越少：

| 周 | Spec | 给你的信息 |
|---|---|---|
| W5 | #1 `top_p_sample` | 函数签名 + 行为 + 测试 |
| W6 | #2 `AutoregressiveDecoder` | 类接口 + 状态行为 |
| W7 | #3 `StaticCache` | 抽象接口 + 黑盒测试 |
| W8 | #4 `GroupedQueryAttention` | 数学规格 |
| W9 | #5 Bug Hunt | 一份有 bug 的实现，只要求你定义 correctness |
| W10 | #6 `estimate_kv_bytes` | 输入输出需求 + 真实验证对象 |
| W11 | #7 benchmark CLI | 用户需求 + CLI contract |
| W12+ | Capstone | 只有产品需求 |

标准流程（每次相同）：
1. 读 spec，**不搜、不看任何实现**。
2. 让 AI **只根据 spec** 生成黑盒测试（不给它你的代码 / 思路）。检查测试是否符合 spec → `git commit -m "test: freeze specN tests"`。
3. 实现，直到冻结测试全绿。不许为通过而改测试；可以加测试。
4. 记录：空白文件延迟、卡点、是"不会 API"还是"不会拆问题"。

下面每个 spec 都附"最少验收测试"——AI 生成的测试至少要覆盖这些；你也可以自己先把它们写出来再让 AI 补充。

---

<a id="spec-1"></a>
## Spec #1（Day 32）· `top_p_sample`

```python
def top_p_sample(logits: torch.Tensor, p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: [B, V] → 返回 [B] 的 token id（long）。"""
```
行为：对每一行，按概率从高到低排序，保留**累计概率首次达到或超过 p 的最小 token 集合**；其余 token 概率置零；重新归一化；从剩余分布采样。`p` 不在 `(0, 1]` 内抛 `ValueError`。

最少验收测试：`p=1.0` 等价于原分布采样（统计上：大样本频率接近 softmax）；`p` 极小时只保留最高概率 token（输出恒等于 argmax）；batch 维保持、每行独立；筛选后概率和为 1（暴露一个 `top_p_filter(logits, p) -> probs` 辅助函数便于测试）；非法 `p` 抛错；固定 generator 可复现。

---

<a id="spec-2"></a>
## Spec #2（Day 39）· `AutoregressiveDecoder`

```python
class AutoregressiveDecoder:
    def __init__(self, model, max_new_tokens_limit: int | None = None): ...
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:   # [B,T] → 下一 token 的 logits [B,V]
    def step(self, token: torch.Tensor) -> torch.Tensor:          # [B] 或 [B,1] → logits [B,V]
    def reset(self) -> None: ...
    @property
    def seq_len(self) -> int: ...
```
状态行为：初始为 `EMPTY`；`prefill` 后进入 `READY`；`step` 只在 `READY` 下合法且每次只接受一个 token；`reset` 回到 `EMPTY` 并清空全部历史；非法调用顺序（未 prefill 就 step、重复 prefill 不 reset）抛 `RuntimeError`。cache 的实现不限定（可用你 Week 6 的 list[tuple]）。

最少验收测试：合法顺序 prefill→step×N 的 greedy 序列 == naive 全重算的 greedy 序列；未 prefill 就 step 抛错；prefill 两次不 reset 抛错；reset 后再次 prefill 与新建对象结果一致；`seq_len` 在 prefill 后 == T、每 step +1；`step` 传入 `[B,2]` 抛错。

---

<a id="spec-3"></a>
## Spec #3（Day 46）· `StaticCache`

给定抽象接口（与你 Week 7 周一设计的一致）：
```python
class KVCache(Protocol):
    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reset(self) -> None: ...
    @property
    def seq_len(self) -> int: ...
```
实现 `StaticCache(n_layers, batch, n_kv_heads, max_seq_len, head_dim, dtype, device)`：
- 构造时一次性预分配全部 K/V 存储；
- `update` 把 `k, v: [B, H, T, D]` 写入位置 `[seq_len, seq_len+T)`，**不允许 `torch.cat`**，返回当前有效的 K/V 视图（长度 `seq_len+T`）；
- 超过 `max_seq_len` 抛 `ValueError`（或自定义异常），且不破坏已有内容；
- `reset` 只把 `seq_len` 归零，**不重新分配 tensor**；
- 内部表示由你决定（要求写进 docstring）。

最少验收测试：通过 Week 7 的全部契约测试；`update` 后底层存储 `data_ptr()` 不变；超容量抛错且此前内容不变；`reset` 后 `data_ptr()` 不变；用 `DynamicCache` 与 `StaticCache` 分别做 cached decode，logits `assert_close`；每层 seq_len 一致。

完成后才允许读 HF `StaticCache`，并写 `design_comparison.md`。

---

<a id="spec-4"></a>
## Spec #4（Day 53，或顺延到 Day 60）· `GroupedQueryAttention`

只给数学规格。设 `n_head = 8, n_kv_head = 2, head_dim = 64`：
```
Q: [B, 8, T, 64]    K: [B, 2, T, 64]    V: [B, 2, T, 64]
query head h 使用 kv head  h // (n_head // n_kv_head)      （每 4 个 q head 共享 1 个 kv head）
out_h = softmax(Q_h K_{g(h)}^T / sqrt(64) + causal_mask) V_{g(h)}
```
实现 `GroupedQueryAttention(n_embd, n_head, n_kv_head, ...)`，含 q/k/v 投影（k、v 投影输出维度为 `n_kv_head * head_dim`）、RoPE 接口、cache 接口（cache 里存的是 `[B, n_kv_head, T, D]`）。**不给 `repeat_kv` 的实现提示**——由你决定 head 映射怎么做（提示只有一句：想想 `expand` 与 `repeat` 的区别）。

最少验收测试：`n_kv_head == n_head` 时与普通 MHA 数值一致（MHA 特例）；`n_kv_head == 1` 时所有 q head 用同一 K/V（MQA 特例，用手算 reference 验证）；shape；`n_head % n_kv_head != 0` 抛错；cache 里的 K/V 字节数随 `n_kv_head` 减小而按比例减小；与 `F.scaled_dot_product_attention(enable_gqa=True)`（若你的 torch 版本支持）一致。

---

<a id="spec-5"></a>
## Spec #5（Day 60）· Bug Hunt

见 [`week09_bughunt/README.md`](../week09_bughunt/README.md)。给你一份**外部提供**的、埋了 4 个 bug 的 KV cache 实现。你不知道 bug 在哪。要求：不改代码 → 先写能让 bug 暴露的测试（≥4 个失败）→ 定位 → 修 → 对封存答案。

这一周不给实现任务，反过来训练："优秀工程师不是会把正确代码写出来，而是能定义什么叫正确。"

---

<a id="spec-6"></a>
## Spec #6（Day 67）· `estimate_kv_bytes`

```python
@dataclass
class KVEstimate:
    total_bytes: int
    mib: float
    bytes_per_token: int

def estimate_kv_bytes(n_layers: int, n_kv_heads: int, head_dim: int, batch_size: int, seq_len: int, dtype: torch.dtype) -> KVEstimate: ...
```
**不给公式。** 唯一的验收标准：对 SmolLM2-135M（从 `model.config` 读参数）与你自己的 TinyGPT，估算值必须与真实 `past_key_values` 各层 tensor `numel() * element_size()` 之和**完全相等**（float32 与 bfloat16 各测）。额外要求：能接受 `config` 对象作为便捷入口（`from_hf_config(config)`）。

最少验收测试：与真实 tensor 逐 dtype 相等；线性性（`seq_len` 翻倍 → 字节翻倍）；`bytes_per_token * seq_len * batch == total_bytes`；`n_kv_heads` 减半 → 减半。

---

<a id="spec-7"></a>
## Spec #7（Day 74）· benchmark CLI contract

用户需求：一条命令比较任意 cache 实现。
```
python -m mini_kv.benchmark --cache dynamic --context 512 --new-tokens 64 [--device mps] [--dtype float32] [--runs 20] [--warmup 5] [--csv out.csv]
```
输出（stdout 一行 JSON 或对齐表格，同时可写 CSV）至少含：`cache, context, new_tokens, device, dtype, prefill_ms, decode_ms_per_token, tokens_per_s, kv_bytes, model_config`。`--cache` 接受 `none | dynamic | static | sliding:<window>`。计时必须 warmup + `torch.mps.synchronize()`（MPS 时）+ 中位数。

**架构验收**：新增一种 cache（例如一个 `DummyCache`）**不允许修改 benchmark 的核心逻辑**——只允许在一处注册（registry / factory）。测试方式：黑盒测试里定义一个 `DummyCache` 注册进去后 `--cache dummy` 能跑通。

最少验收测试：CLI 返回码 0 且输出可解析；四种 cache 各跑一次；`kv_bytes` 与 Spec #6 估算一致；`--runs 1 --warmup 0` 也能跑（快速测试路径）；dummy cache 注册测试；非法 `--cache` 报错并列出可用项。
