# Phase II · 从零写 Decoder-only Transformer（Day 15–35）

**目标**：不看任何实现，从空白写出完整 GPT 的 forward + 训练 + 生成，并能不运行代码写出每一步的 tensor shape。
**主参考**：Karpathy《Let's build GPT: from scratch, in code, spelled out》——但规则是**看一段概念 → 关视频 → 自己写**，绝不跟着打。
**产物**：`week03_attention/`、`week04_gpt/`、`week05_tinygpt/`（可训练、可采样、有 CLI 的 TinyGPT）。
**Gate 2**：Day 33–34。**Spec Track 从 Week 5 开始。**

---

## Week 3 · Attention（三个版本）

### Day 15（周一）· 版本 A：显式循环

【读】视频从开头到"the mathematical trick in self-attention"结束（约 40 min），关掉。

【写】`week03_attention/attn_loop.py`
1. 先做**因果均值聚合**：`out[b,t] = mean(x[b, :t+1])`，用 `for b: for t:` 双循环。
2. 再引入 q/k/v：单头，`x: [B,T,C]` → `q,k,v = Wq(x), Wk(x), Wv(x)`，对每个 t 只对 `k[:, :t+1]` 做点积 → softmax → 加权 `v`。全部显式循环，不用 mask。

【测】`tests/test_attn.py`
- 输出 shape `[B,T,C]`
- **因果性**：改 `x[:, t+1:]` 的值，`out[:, :t+1]` 不变（用 `torch.equal` 或 `assert_close`）
- 均值版本与 `torch.cumsum(x,1)/arange` 一致

### Day 16（周二）· 版本 B：矩阵形式

【读】视频"the mathematical trick"到单头 attention 实现结束（约 30 min），关掉。

【写】`attn_matrix.py`
```python
# q,k,v: [B,T,D]
# wei = q @ k.transpose(-2,-1) * D**-0.5      -> [B,T,T]
# wei = wei.masked_fill(tril[:T,:T] == 0, float("-inf"))
# wei = softmax(wei, dim=-1)                  -> 每行和为 1
# out = wei @ v                               -> [B,T,D]
```
学会 `torch.tril`、`masked_fill`、`register_buffer`（mask 不是参数，但要随模型 `.to(device)`）。

【测】版本 B 与版本 A 在相同权重下 `assert_close`（先把 A 改成能注入相同 Wq/Wk/Wv）；`wei` 每行和为 1；上三角全 0。

【记】为什么除以 `sqrt(D)`？用一个实验回答：不除时 `wei` 的方差随 D 怎么变、softmax 变得多尖。

### Day 17（周三）· 版本 C：多头 CausalSelfAttention

【读】视频多头部分（约 20 min），关掉。

【写】`attention.py`
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> qkv: [B,T,3C] -> q,k,v: [B,T,C] -> view [B,T,H,D] -> transpose [B,H,T,D]
        # wei: [B,H,T,T] -> out: [B,H,T,D] -> transpose+contiguous+view [B,T,C] -> proj
```
一个融合的 `nn.Linear(C, 3C)` + 输出投影 `nn.Linear(C, C)`。

【测】
- `@pytest.mark.parametrize("B,T,C,H", [(1,4,32,4),(2,8,64,8),(4,16,32,2)])` shape 测试
- 与 `F.scaled_dot_product_attention(q,k,v,is_causal=True)` 在相同 q/k/v 下 `assert_close`（这是你的 reference implementation）
- 因果性测试升级到多头

### Day 18（周四）· shape 推导 + 细节

Spec Track 下周才开始，今天仍是 Build。

【写】`notes/shapes.md`：**不运行代码**，手写 B=2,T=8,C=32,H=4 下 `x → qkv → q → q(view) → q(transpose) → wei → softmax → out → out(transpose) → out(view) → proj` 每一步的 shape 和含义。然后运行验证。错一处就说明还没真懂。

【写】
- attention dropout 与 residual dropout 的位置
- `forward(..., return_weights: bool = False)`：可选返回 `[B,H,T,T]` 的权重（Capstone 的 eviction stretch 会用到）
- 融合 qkv 与三个独立 Linear 在数学上等价：写测试证明

### Day 19（周五）· 📖 SDPA 文档 + minGPT

【读】(≤45 min) [`F.scaled_dot_product_attention` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)（`is_causal`、`attn_mask`、`scale` 的语义，尤其是 mask 为 bool 与 float 时的区别）；[minGPT `model.py`](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) 的 `CausalSelfAttention`。

【记】`notes/mingpt_attn.md`：3 条设计差异（例如：mask 怎么存、dropout 位置、head 拆分顺序）。

### Day 20（周六）· 🧪 性质测试 + 微 benchmark

【测】补性质测试：batch 元素互相独立（改 batch 1 不影响 batch 0）；softmax 行和；mask 上三角；`eval()` 下确定性。

【写】`bench_attn.py`：loop vs matrix vs SDPA，T∈{64,256,1024}，CPU 与 MPS。**首次使用**计时规范：warmup → `torch.mps.synchronize()` → 计时 → 运行 → `synchronize()` → 计时。记录 device/dtype。

### Day 21（周日）· 🔁 Retrieval #3（60 min）

`scratch/w3_retrieval/`：空白写多头 `CausalSelfAttention`（≤100 LOC）+ 因果测试 + 与 SDPA 对比测试。对比、记日志。

---

## Week 4 · Transformer Block → GPT

### Day 22（周一）· LayerNorm + MLP

【读】视频 LayerNorm / residual / feed-forward 段落（约 25 min），关掉。

【写】`week04_gpt/layers.py`
- 手写 `LayerNorm`（按公式：减均值、除 sqrt(var+eps)、乘 gamma 加 beta），与 `nn.LayerNorm` `assert_close`
- `MLP`：`Linear(C, 4C) → GELU → Linear(4C, C) → Dropout`

【测】LN 输出每个位置均值≈0 方差≈1；与 `nn.LayerNorm` 一致；MLP shape。

### Day 23（周二）· Block + GPT

【写】`model.py`
```python
class Block(nn.Module):      # x = x + attn(ln1(x)); x = x + mlp(ln2(x))   (pre-LN)
class GPT(nn.Module):
    # tok_emb [V,C] + pos_emb [block_size,C] -> N×Block -> ln_f -> lm_head [C,V]
    def forward(self, idx, targets=None) -> (logits [B,T,V], loss | None)
```
【测】shape；初始 loss ≈ ln(V)；模型级因果性（改未来 token 不影响过去位置 logits）。

### Day 24（周三）· 训练 TinyGPT

【写】复用 Week 2 的训练循环。config：`n_layer=4, n_head=4, n_embd=128, block_size=64, batch=32, lr=3e-4`，MPS 上跑 ~2000 步（几分钟）。loss 从 ~4.2 降到 <2.0 即达标（R9：不调参）。
naive 生成：每步把全序列（截到 block_size）喂进去，取最后 logits。生成 300 字符看看像不像英文。

【测】overfit-one-batch；CPU 上 5 步 smoke。

### Day 25（周四）· 细节 + 写自己的架构说明

【写】
- weight tying（`lm_head.weight = tok_emb.weight`）；`_init_weights`（normal std=0.02）；`count_params()`
- val loss；`model.eval()` + `no_grad` 的生成
- `ARCHITECTURE.md`：用你自己的话 + ASCII 图写出数据流与每层 shape。Gate 2 第 7 项会对照这份文件。

### Day 26（周五）· 📖 nanoGPT model.py（现在才允许）

【读】(≤60 min) [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py)：`GPTConfig / CausalSelfAttention / MLP / Block / GPT.forward / generate`。**不读**训练脚本与 `from_pretrained`。

【记】`notes/nanogpt_diff.md`：5 条差异及原因（例如 bias 选项、flash 路径、`crop_block_size`、init 的 `c_proj` 缩放）。哪一条你想采纳？为什么？

### Day 27（周六）· 🧪

parametrize 模型 shape 测试；save/load；`pytest` < 30 s；`ruff`；commit。

### Day 28（周日）· 🔁 Retrieval #4（60 min）

空白写 `Block` + `GPT.forward`（≤100 LOC）+ 初始 loss 测试。

---

## Week 5 · 项目化 + 第一次 Spec Challenge + Gate 2

### Day 29（周一）· config / seed / device / CLI

【写】`week05_tinygpt/tinygpt/`
- `@dataclass class GPTConfig: vocab_size, n_layer, n_head, n_embd, block_size, dropout`
- `set_seed(seed)`（`torch.manual_seed`；MPS 也受它控制）；`get_device(pref: str) -> torch.device`
- `train.py` argparse：`--n-layer --n-head --n-embd --steps --lr --device --out`；loss 写 CSV

### Day 30（周二）· checkpoint + 采样 CLI

【写】
- checkpoint 内嵌 config（`asdict(config)`），load 时重建
- `generate.py --ckpt --prompt --max-new-tokens --temperature --top-k --seed`
- `temperature`：`logits / T`；`top_k`：`torch.topk` 取阈值，其余 `-inf`

### Day 31（周三）· 采样测试 + 结构整理

【测】temperature→很小 ≈ greedy；`top_k=1` == greedy；同 seed 两次生成一致；`top_k > V` 不报错。
【测】项目结构：`tinygpt/{config,model,attention,layers,sampling,train,generate}.py` + `tests/`。

### Day 32（周四）· 📐 Spec Challenge #1：`top_p_sample`

流程（以后每周四相同）：
1. 读 [specs.md → Spec #1](specs.md#spec-1)。不搜、不看任何实现。
2. 让 AI **只根据 spec** 生成黑盒测试（R4）。检查 → 冻结 commit。
3. 实现，直到冻结测试全绿。记录空白文件延迟。

### Day 33–34（周五、周六）· 🚪 Gate 2

见 [gates.md → Gate 2](gates.md#gate-2)。设置：`gate2/` 目录，允许复制**自己的** `tokenizer.py / data.py / train.py / tests 骨架`进去（基础设施不重考），`model.py` 必须空白开始：`GPTConfig / CausalSelfAttention / MLP / Block / GPT / generate`。3 小时。
Day 34：60 min 指定模块闭卷重构（抽 `CausalSelfAttention` 或 `Block`）+ 不运行代码写出全部 shape + 清单自评 + `git tag gate-2`。

### Day 35（周日）· ⏸ Buffer
