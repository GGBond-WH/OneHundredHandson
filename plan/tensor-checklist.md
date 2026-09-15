# 张量操作闭卷清单

KV cache 代码的难点 80% 是张量操作。**Week 6 开始前，A 级必须做到"不查文档就知道该用哪个"**（参数细节可以查）。周日 Retrieval Drill 会抽查。

## A 级：必须闭卷知道"该用什么"

### Shape / layout
`shape / size / ndim` · `view / reshape / flatten` · `unsqueeze / squeeze` · `transpose / permute` · `contiguous / is_contiguous / stride`
必须能解释：`view` 与 `reshape` 的区别；为什么 `transpose` 后可能 non-contiguous、为什么 `view` 会报错。

### Broadcasting
`None` 索引 / `unsqueeze` · `expand`（不拷贝）· `repeat`（拷贝）· `repeat_interleave`
必须能回答：GQA 里为什么 `unsqueeze + expand + reshape` 可以把 KV head 映射到 query head 而几乎不占额外内存？

### 序列 / cache 操作
`cat / stack / split / chunk / narrow / select`
必须能解释：`torch.cat([past_k, new_k], dim=2)` 与 `cache[:, :, pos:pos+T].copy_(new_k)` 分别对应 Dynamic / Static 的思路，代价差在哪。

### Indexing
basic slicing · 整数索引 · boolean mask · advanced indexing · `arange` · `index_select` · `gather` · `scatter / scatter_add` · `take_along_dim`（和排序后的 index 配合）

### StaticCache 必需
`copy_` · `index_copy_` · in-place 写入 vs 重新分配（`data_ptr()` 可验证）

### Mask / positions
`arange` · `triu / tril` · `masked_fill` · `where`
必须闭卷能写：`positions = torch.arange(past_len, past_len + T, device=...)`；`mask[i, j] = j <= past_len + i`。

### 采样 / 研究方向马上会用
`topk / sort / argsort / cumsum / softmax / multinomial`（top-k、top-p、token eviction、attention 分数排序的共同基础）

### Memory / debug
`numel / element_size / dtype / device / data_ptr`

## B 级：知道存在、会查

`einsum` · `baddbmm` · `roll` · `unfold` · `register_buffer` 的持久化选项 · `torch.testing.assert_close` 的 rtol/atol · `torch.Generator` · `torch.mps.synchronize / current_allocated_memory / driver_allocated_memory` · `PYTORCH_ENABLE_MPS_FALLBACK`（只用于排查，不用于 benchmark）

## 自测题（Week 5 周日抽 3 题）

1. `x: [B,T,C]`，`H` 个头 → `[B,H,T,D]`，一行。
2. `k: [B,Hkv,T,D]` 复制到 `[B,Hq,T,D]`（`Hq = Hkv * g`），不额外拷贝内存，写出 shape 变化。
3. 在预分配 `cache[B,H,max,D]` 的位置 `pos: [T]` 写入 `new: [B,H,T,D]`，不用 cat。
4. 生成 q 长 `T`、kv 长 `past+T` 的因果 mask（bool）。
5. `probs: [B,V]` 做 top-p：排序、累计、掩码、恢复原顺序（`scatter` 或 `gather`），写出每步 shape。
6. 用 `data_ptr()` 证明 `expand` 不拷贝而 `repeat` 拷贝。
