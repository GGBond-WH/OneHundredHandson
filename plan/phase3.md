# Phase III · Prefill / Decode / KV cache / RoPE / GQA（Day 36–56）

**目标**：你的 KV cache 理论在这里兑现——把 `model(input_ids)` 拆成 prefill + decode，亲手实现 cache，并用**数值不变量**（而不是"生成文本看起来差不多"）证明它对。
**产物**：`week06_kv/`（带 cache 的 TinyGPT）、`week07_cache/`（Dynamic / Static / Sliding 三种 cache + 契约测试）、`week08_llama/`（RoPE / RMSNorm / SwiGLU / GQA 的 Llama 风格变体）。
**Gate 3**：Day 54–55。**Week 8 是全程最重的一周，见 Day 52 的 readiness checkpoint。**

核心数据流（以后闭卷要能画出来）：
```
input_ids → embedding → layer 0 { q ; k → cache ; v → cache } → layer 1 … → logits → sample → next token
prefill: T 个 token 一次进                 decode: 每次只进 1 个 token，q 是 [B,H,1,D]，K/V 从 cache 取 [B,H,past+1,D]
```

---

## Week 6 · Prefill / Decode（先用最朴素的 cache）

### Day 36（周一）· 先测痛点，再写计划

【写】`week06_kv/bench_naive.py`：用 Week 5 的 TinyGPT，naive 生成 T=64→512，记录**每个新 token 的耗时**随 T 的增长（应近似线性增长，因为每步重算全序列）。

【写】`notes/cache_plan.md`（**动手前必须写**，1 页）：forward 要改哪几处才能支持 cache？
- attention 的 K/V 从哪来（本步计算的 + cache 里的）
- position embedding 的下标从哪开始
- decode 步 q 长 1、kv 长 past+1 时 mask 还需要吗
- cache 的数据结构（本周只允许 `list[tuple[K, V]]`，每层一个 tuple，K/V 是 `[B,H,T,D]`）
- 返回值：`(logits, new_kv)`

### Day 37（周二）· forward 支持 cache

【写】`model.py`
```python
def forward(self, idx, targets=None, past_kv: list[tuple[Tensor,Tensor]] | None = None, use_cache: bool = False):
    past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
    pos = torch.arange(past_len, past_len + T, device=idx.device)   # ← 最常见的 bug 就在这里
    ...
# attention 里：
#   k = cat([past_k, k], dim=2); v = cat([past_v, v], dim=2)   # past_k: [B,H,past,D], k: [B,H,T,D]
#   mask: q 长 T、kv 长 past+T → 允许 q_i 看 kv_j 当且仅当 j <= past + i
```
先在 CPU float32 上做。

【测】`use_cache=False` 路径与旧行为完全一致（回归）；`use_cache=True` 且 `past_kv=None` 时返回的每层 K/V shape 为 `[B,H,T,D]`。

### Day 38（周三）· generate_cached + 等价测试（本周核心）

【写】`generate_cached(model, idx, max_new_tokens)`：prefill 一次拿到 logits + kv → 循环：只喂最后一个 token + `past_kv` → 更新 kv。

【测】`tests/test_cache_equivalence.py`（以后每次改 attention 都跑）
```python
torch.manual_seed(0); model.eval()
full_logits = model(tokens)[0]                        # [B,T,V]
# prefill 前 t0 个，再逐 token decode，收集每步 logits
torch.testing.assert_close(cached_logits, full_logits[:, t0-1:], rtol=1e-4, atol=1e-5)
assert greedy_tokens_naive == greedy_tokens_cached   # 完全一致
```
不过就 debug：优先怀疑 position / mask / cat 的 dim / head 拆分。每个真 bug 记入 bugs.md（R6）。

### Day 39（周四）· 📐 Spec #2：`AutoregressiveDecoder`

见 [specs.md → Spec #2](specs.md#spec-2)。只给接口和状态行为，训练 stateful API 设计。

### Day 40（周五）· 📖 gpt-fast generate.py

【读】(≤45 min) [gpt-fast `generate.py`](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py) 的 `prefill / decode_one_token / decode_n_tokens / generate`。**只**看：`input_pos` 怎么传、prefill 与 decode 的调用差异。不读 compile / tensor parallel / speculative。
【记】`notes/gptfast_generate.md`：它的 decode 为什么只需要 `input_pos` 而不需要拼接？（预告 StaticCache。）

### Day 41（周六）· 🧪 benchmark + 内存

【写】`bench_generate.py`：naive vs cached，CPU 与 MPS，T∈{64,128,256,512}，new_tokens=64，warmup 3 + 测 10 取中位数，`synchronize`；指标：prefill 时间、TPOT（每 token 时间）、tokens/s。
KV 内存：`sum(k.numel()*k.element_size() + v.numel()*v.element_size() for k,v in kv)`，与你手算的 `2 · L · B · H · T · D · bytes` 对比。
【记】结果表进 README（记录 device / dtype / config）。

### Day 42（周日）· 🔁 Retrieval #6（60 min）

空白写"带 past K/V 拼接 + 位置偏移 + 通用 mask"的 attention forward（≤100 LOC）+ 等价测试骨架。

---

## Week 7 · Cache 抽象：Dynamic / Static / Sliding

### Day 43（周一）· 设计接口 + 契约测试

【写】`week07_cache/cache/base.py`——**你自己设计**，不看 HF：
```python
class KVCache(Protocol):            # 或 ABC
    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]: ...  # 返回本层用于 attention 的完整 K/V
    def get(self, layer: int) -> tuple[Tensor, Tensor]: ...
    def reset(self) -> None: ...
    @property
    def seq_len(self) -> int: ...
```
`cache/dynamic.py`：把 list[tuple] 重构成 `DynamicCache`（内部仍是 `torch.cat`）。

【测】`tests/test_cache_contract.py`：用 fixture 参数化到"所有 cache 类"，任何实现都必须过：update 后 seq_len 增长；get 返回 shape；reset 后 seq_len==0；两次 update 的 K 顺序正确（第一次的在前）。

### Day 44（周二）· 模型接入 cache 对象

【写】`forward(idx, cache: KVCache | None = None)`；attention 里 `k, v = cache.update(layer_idx, k, v)`。删除 list[tuple] 路径（或保留为 `DynamicCache` 的薄封装）。
【测】Week 6 的等价测试在 `DynamicCache` 下全绿；契约测试全绿。

### Day 45（周三）· 分块 prefill 与通用 mask

【写】当 cache 已有 `past` 个 token、再喂 `T` 个 token 时（chunked prefill），mask 必须是 `j <= past + i`。实现 `build_causal_mask(past_len, T, device)`。
【测】"分两块 prefill 再 decode" == "一块 prefill 再 decode"（logits 与 greedy 序列）。这个测试会抓住大量 off-by-one。

### Day 46（周四）· 📐 Spec #3：`StaticCache`

见 [specs.md → Spec #3](specs.md#spec-3)。不许 `cat`、预分配 `max_seq_len`、`reset` 不重分配、超容量报错、`data_ptr()` 不变。完成前不许读 HF。

### Day 47（周五）· 📖 HF cache_utils（导航式）

【读】(≤60 min) [transformers `cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)：只找 `CacheLayerMixin / DynamicLayer / StaticLayer / DynamicCache / StaticCache`，看 `update / get_seq_length / reset` 的签名与内部 tensor 布局 `[B, H, T, D]`。文件很长，不通读。
【记】`notes/design_comparison.md`（模板 templates/）：我的接口哪里不同？HF 为什么这样设计？哪些差异是工程需求造成的（多模型、多设备、compile）？

### Day 48（周六）· 🧪 SlidingWindowCache + benchmark

【写】`cache/sliding.py`：只保留最近 `window` 个 token 的 K/V。
【测】**reference 是"在全序列上做只看最近 window 个 token 的 attention"**，不是 full cache——两者语义本来不同。写 `windowed_reference_attention` 作为 oracle，与 sliding cache 的 decode logits `assert_close`。
【写】Dynamic vs Static 的 TPOT 对比（Static 应更稳定、无重复分配）。

### Day 49（周日）· 🔁 Retrieval #7（60 min）

只给一句话：给定预分配 `cache[B,H,max_len,D]`、位置 `pos: [T]`、`new_k: [B,H,T,D]`，**不用 cat** 完成更新并维护 `seq_len`；写测试证明 `data_ptr()` 不变。（≤60 LOC，应远用不完 60 分钟——测的是能否立刻想到正确的张量操作。）

---

## Week 8 · RoPE / RMSNorm / SwiGLU / GQA（最重的一周）

负荷规则：RMSNorm 与 SwiGLU 各只有几行，周一 1 小时内做完；其余全部给 RoPE 及其与 cache 的对接；Gate 周取消周五阅读；Day 52 有 readiness checkpoint。

### Day 50（周一）· RMSNorm + SwiGLU（1 h）→ RoPE

【写】`week08_llama/layers.py`：`RMSNorm`（`x * rsqrt(mean(x²)+eps) * weight`，与 `nn.RMSNorm` 对比）；`SwiGLU`（`w2(silu(w1(x)) * w3(x))`）。

【读】RoPE 只需理解：把每个 head 的 D 维两两配对成 D/2 个复数（或 `rotate_half`），按位置 m 旋转角度 `m·θ_i`，`θ_i = base^(-2i/D)`。点积 `q_m·k_n` 只依赖 `m−n`。

【写】`rope.py`
```python
def precompute_freqs(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:  # cos, sin: [max_len, D/2] 或 [max_len, D]
def apply_rope(x: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:                  # x: [B,H,T,D], positions: [T]
```
【测】旋转不改范数；**相对位置性质**：随机 q,k，`dot(rope(q,m), rope(k,n))` 只依赖 `m−n`（比较 (m,n)=(5,2) 与 (13,10)）；`positions=0` 时恒等。

### Day 51（周二）· RoPE 接入 + cache 等价重新全绿

【写】Llama 风格 `Block`：`x = x + attn(rmsnorm(x)); x = x + swiglu(rmsnorm(x))`；去掉学习式 pos_emb，attention 里对 q,k 做 `apply_rope(·, positions)`，**`positions = arange(past_len, past_len+T)`**。短训 500 步确认 loss 下降。
【测】Week 6/7 全部等价测试（Dynamic / Static / 分块 prefill）在 RoPE 模型上重新跑——**这是本周最容易出 bug 的地方**：decode 时 position 从 0 重启会让 greedy 悄悄变化。

### Day 52（周三）· Readiness checkpoint

- ✅ RoPE + Dynamic + Static + 分块 prefill 等价测试全绿 → 今天补 sliding 与 RoPE 的组合测试、整理代码，明天正常做 GQA Spec。
- ❌ 未全绿 → 今天与明天继续修（仍受 20 分钟规则约束），**GQA Spec 顺延到 Day 60**；Gate 3 时间不后移，且 Gate 3 不把 GQA 当 critical。

### Day 53（周四）· 📐 Spec #4：`GroupedQueryAttention`

见 [specs.md → Spec #4](specs.md#spec-4)。只给数学规格：`n_head=8, n_kv_head=2, head_dim=64`，每 4 个 q head 共享 1 个 kv head。不给 `repeat_kv` 提示。

### Day 54–55（周五、周六）· 🚪 Gate 3

见 [gates.md → Gate 3](gates.md#gate-3)。核心：prefill/decode、Dynamic/Static、position 正确性、cache/no-cache 数值等价。`git tag gate-3`。

### Day 56（周日）· ⏸ Buffer
