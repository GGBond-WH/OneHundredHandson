# Phase I · 原理 + 自己写 mini 推理服务引擎（Day 1–28）

**目标**：在读 vLLM 之前，先用纯 PyTorch 把它最核心的几件事**自己做一遍**：分页 KV 存储、块分配、扁平化的混合 batch、连续批处理调度、抢占、分块 prefill、前缀缓存。全部在 CPU / MPS 上验证正确性，不追求速度。
**为什么先写**：vLLM 的 `vllm/v1/core/` 只有十来个文件，但每个文件都在处理大量边界情况。如果你自己写过一个只有核心逻辑的版本，Phase III 读它时就是在"认出"你已经做过的设计，而不是从零学一个陌生系统。Plan 1 里你先写 GPT 再读 nanoGPT，是同一个道理。
**产物**：`vLLM/minivllm/`。
**Gate A**：Day 26–27，清单见 [gates.md](gates.md#gate-a)。

## 准备：minivllm 的起点

```
vLLM/minivllm/
├── minivllm/
│   ├── model.py          # 从 Plan 1 复制你自己的 Llama 风格模型（Week 8 或 MiniKVLab），attention 之后要改
│   ├── reference.py      # 保留 Plan 1 的 DynamicCache 推理路径，作为所有等价测试的参照答案
│   └── ...               # 每天新增的模块
└── tests/
```

复制**你自己的** Plan 1 代码是允许的（规则只禁止复制别人的代码）。所有正确性测试都以 Plan 1 那条已经验证过的 DynamicCache 路径为参照：**同一个模型、同一个 prompt，greedy 输出必须完全一致，logits 必须 `assert_close`**。模型用小配置的随机权重即可，正确性测试不需要训练过的权重。

---

## Week 1 · 为什么推理服务需要分页

### Day 1（周一）· 先量出问题

【读】(≤60 min) Orca（OSDI 2022）的第 1–3 节：iteration-level scheduling，也就是"每生成一个 token 就重新决定这一步跑哪些请求"；PagedAttention 论文（arXiv 2309.06180）第 1–3 节：KV cache 的三种浪费，即为最大长度预留、内部碎片、外部碎片。读完关掉。

【写】`notes/waste.md`：用 Plan 1 的 `estimate_kv_bytes`，算一个具体场景。8 个请求，prompt 长度和输出长度各不相同（自己编，比如 20–500），用 Plan 1 的 StaticCache 方式（每个请求按 `max_len=1024` 预留）：

- 预留了多少字节？实际用了多少？浪费比例是多少？
- 如果改成每块 16 个 token、按需分配，最多浪费多少？（每个请求最后一块没填满的部分）

【写】建 `minivllm/` 骨架，复制模型和 reference 路径，写一个测试确认复制过来的 reference 路径能跑。commit。

### Day 2（周二）· BlockAllocator

【写】`minivllm/allocator.py`：管理 `num_blocks` 个物理块。
- 分配一个块 → 返回块 id，引用计数设为 1；
- 引用计数加一 / 减一，减到 0 时块回到空闲池；
- 查询空闲块数量；
- 没有空闲块时分配、对引用计数为 0 的块再减一：怎么处理由你决定，写进 docstring。

为什么要引用计数：Week 4 的前缀缓存会让多个请求共享同一个块。今天先把机制做好。

【测】分配到耗尽；释放后可再分配；引用计数为 2 的块减一次不回到空闲池；"空闲 + 已分配 = num_blocks"这条守恒式在任何操作序列后都成立（写一个随机操作序列的测试）。

### Day 3（周三）· 分页 KV 存储

【写】`minivllm/kv_store.py`：每层两个大张量，形状 `[num_blocks, block_size, H_kv, D]`（或者把层也放进第一维，你选）。

关键视角：把一层的存储看成 `num_blocks × block_size` 个**槽位**（slot）排成一行，每个槽位放一个 token 的 K（或 V）。用 `view` 得到 `[num_blocks * block_size, H_kv, D]`，就可以按槽位编号批量写入（回忆 Plan 1 的 StaticCache：`index_copy_`）。

【测】往一组槽位写入随机 K/V，再按同样的槽位读回，完全相等；写入不改动其他槽位；`data_ptr()` 不变（没有重新分配）。

### Day 4（周四）· 📐 Spec #1：`blocks_needed` + `compute_slot_mapping`

见 [specs.md → Spec #1](specs.md#spec-1)。流程和 Plan 1 相同：读 spec → AI 只看 spec 生成黑盒测试 → 检查、冻结、commit → 实现。

### Day 5（周五）· 📖 PagedAttention 论文第 4 节（写完之后读）

【读】(≤60 min) 第 4 节：PagedAttention 算法、KV cache manager、block table、copy-on-write。
【记】`notes/paper_vs_mine.md`：论文的 block table 和你的 slot mapping 是什么关系？论文里的 copy-on-write 解决什么场景（提示：parallel sampling、beam search）？你的 allocator 支持它还缺什么？

### Day 6（周六）· 🧪 单请求分页注意力 + 第一个等价测试

【写】一个请求的 attention 从分页存储里取 K/V：按它的 block table 取出这些块 → 拼成连续的 `[seq_len, H_kv, D]` → 只保留前 `seq_len` 个 → 换成 `[H_kv, seq_len, D]` → 和 Plan 1 一样做 attention（GQA、RoPE 都沿用你的实现）。

这一版是"先 gather 再算"，会多一次拷贝。vLLM 的 CUDA kernel 直接按 block table 读，不拷贝。但正确性上两者等价，而你现在要的就是正确性。

【测】**本阶段最重要的测试**：一个请求做 prefill，再逐 token decode 20 步，每一步的 logits 和 Plan 1 的 DynamicCache 路径 `assert_close`，greedy 序列完全一致。用 `block_size=4` 这种小值，确保跨块边界的情况被测到。

### Day 7（周日）· 🔁 Retrieval（60 min）

空白文件写 BlockAllocator + `compute_slot_mapping`（≤100 LOC）+ 3 个测试。不打开本周代码。

---

## Week 2 · 一步里同时处理多个请求

### Day 8（周一）· Request 对象

【写】`minivllm/request.py`：一个请求至少有：id、prompt token、已生成 token、**已计算的 token 数**（`num_computed_tokens`）、block table、状态（等待 / 运行 / 完成）、`max_tokens`。

`num_computed_tokens` 是整个设计的核心：它表示这个请求的前多少个 token 的 K/V 已经在缓存里了。prompt 有 100 个 token、`num_computed_tokens=0`，说明还没做 prefill；生成了 3 个 token、`num_computed_tokens=102`，说明最后一个新 token 还没算。

【写】循环版多请求：对每个请求单独调用 Day 6 的逻辑。先让它对，作为下面扁平化版本的参照。

### Day 9（周二）· 扁平化 token

关键洞察：**模型里除了 attention，每一层都是逐 token 的**：embedding、RMSNorm、MLP、LM head 都只看单个 token 自己。所以可以把这一步所有请求的新 token 首尾拼成**一条** `[N, C]`，一起过这些层，不需要 padding。只有 attention 需要知道"哪些 token 属于哪个请求、每个请求之前已经有多少 token"。RoPE 也没问题，它只需要每个 token 自己的绝对位置。

【写】`minivllm/model_runner.py` 初版：输入是扁平的 token、每个 token 的位置、每个 token 的槽位，以及每个请求在这条扁平序列里从哪开始（`query_start_loc`）、总长度多少（`seq_lens`）。attention 按请求切开 q，从分页存储取各自的 K/V。

这些名字是**故意**和 vLLM 一致的。vLLM 的 `CommonAttentionMetadata` 里就有 `query_start_loc`、`seq_lens`、`slot_mapping`（2026-11-17 核实）。Phase II 你在断点里看到它们时，会认出来。

### Day 10（周三）· 一步里混合 prefill 与 decode

【写】同一步里：请求 A 做 10 个 token 的 prefill，请求 B 和 C 各 decode 1 个 token。扁平序列长度是 12。

想清楚因果 mask：对请求 A，第 i 个新 token 能看到它之前缓存的全部 token 加上新 token 里的第 0…i 个。这和 Plan 1 Day 45 的分块 prefill mask 是同一个公式。

【测】混合 batch 里每个请求的 logits，和"这个请求单独用 reference 路径跑"`assert_close`。再加一个随机测试：随机生成 5 个请求、随机长度、随机地一部分 prefill 一部分 decode。

### Day 11（周四）· 📐 Spec #2：`build_step_inputs`

见 [specs.md → Spec #2](specs.md#spec-2)。把"一组被调度的请求"变成 Day 9 那几个扁平张量，这一步在 vLLM 里属于 model runner 的输入准备。

### Day 12（周五）· 📖 nano-vllm 的 model_runner.py（写完之后读）

【读】(≤45 min) [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 `nanovllm/engine/model_runner.py` 里的 `prepare_prefill` 和 `prepare_decode`（2026-11-17 核实存在）。nano-vllm 依赖 CUDA 和 flash-attn，在你的 Mac 上跑不了，但可以读。
【记】`notes/nanovllm_runner.md`：它为 prefill 和 decode 分别准备输入，你的 `build_step_inputs` 只有一个。这两种设计各有什么代价？

### Day 13（周六）· 🧪 静态批处理 vs 逐步混合

【写】`bench_batching.py`：16 个请求，输出长度差别很大（10 到 200）。
- 静态批处理：一批请求一起开始，必须等最长的那个结束，才开始下一批。
- 你的逐步方式：每一步都可以加入新请求。

比较总耗时、平均每个请求的完成时间、峰值 KV 占用（块数 × 每块字节）。记录 device / dtype / 配置（Plan 1 R8）。不需要很快，看相对差别。

### Day 14（周日）· 🔁 Retrieval（60 min）

空白写 `build_step_inputs`（≤100 LOC）+ 2 个测试。

---

## Week 3 · 调度：连续批处理、预算、抢占、分块 prefill

### Day 15（周一）· 统一的调度视角

【读】(≤45 min) Sarathi（arXiv 2308.16369）的分块 prefill 思想；vLLM V1 博客（见 resources）里讲调度器的一节。

关键洞察（来自 V1 博客）：**V1 的调度器不区分 prefill 和 decode**。每个请求只有两个数：总 token 数和已计算的 token 数。调度器每一步给每个请求分配一些"新 token 数"，让后者追赶前者。
- prefill = 一次追很多个；
- decode = 一次追 1 个；
- 分块 prefill = 一次只追一部分，下一步继续追。

所以调度决策就是一个"请求 → 本步新 token 数"的映射，不需要两种模式。

【写】`minivllm/scheduler.py` 骨架：等待队列、运行队列；每步的 token 预算 `max_num_batched_tokens`、请求数上限 `max_num_seqs`；先给运行中的请求分配，再从等待队列补充。

### Day 16（周二）· 引擎循环

【写】`minivllm/engine.py`：`add_request()`；`step()` 依次做：调度 → 为需要的请求分配块 → `build_step_inputs` → 前向 → 对**已经算到最后一个 token** 的请求采样 → 更新状态 → 完成的请求释放全部块。

【测】端到端：8 个请求、不同长度、在不同的步加入，每个请求的 greedy 输出和"单独用 reference 路径跑"完全一致。结束后空闲块数 == `num_blocks`（无泄漏）。

### Day 17（周三）· 抢占

【写】块不够分配时：选一个运行中的请求（你决定选哪个，vLLM 的默认策略是什么留到 Phase III 去验证），释放它的全部块，把 `num_computed_tokens` 归零，放回等待队列**最前面**。下次轮到它时，从头重新计算（recompute）。

【测】把 `num_blocks` 设得很小，迫使抢占发生（断言抢占确实发生过，比如计数）。所有请求的输出仍然和单独运行一致，最后无泄漏。

### Day 18（周四）· 📐 Spec #3：调度器不变量

见 [specs.md → Spec #3](specs.md#spec-3)。这次不是实现新东西，而是把你的调度器适配到给定接口，然后用**随机负载的性质测试**检验：预算从不超、块守恒、所有请求最终完成、抢占后能恢复。

### Day 19（周五）· 📖 nano-vllm 的 scheduler.py（写完之后读）

【读】(≤45 min) `nanovllm/engine/scheduler.py`，只有约 90 行。
【记】`notes/nanovllm_scheduler.md`：它的 `schedule()` 返回"这一步是 prefill 还是 decode"，同一步里两者不混合；分块 prefill 只对第一个请求允许。和你的统一调度比，各有什么好处和代价？哪一种更容易让新请求的首 token 延迟（TTFT）变低？

### Day 20（周六）· 🧪 分块 prefill

【写】一个 300 token 的 prompt 在预算 64 下跨 5 步完成。

边界情况：**prompt 还没算完的请求，这一步的输出不能当作新 token**。它最后一个位置的 logits 预测的是"prompt 里的下一个 token"，不是生成内容。

【测】分块与不分块的 greedy 输出完全一致；分块过程中这个请求的输出列表保持为空，直到 prompt 全部算完。再测一个场景：长 prompt 分块期间，其他请求的 decode 每一步都在继续。这就是分块 prefill 要解决的问题：长 prompt 不再阻塞别人。

### Day 21（周日）· 🔁 Retrieval（60 min）

空白写带预算和抢占的 `schedule()`（≤100 LOC）。

---

## Week 4 · 前缀缓存 + Gate A

### Day 22（周一）· 前缀缓存 ①：块哈希

设计（这是 vLLM 的做法，概念层面）：
- 只给**满块**算哈希；
- 哈希 = H(父块的哈希, 本块的 token)。把父块哈希放进去，是为了让"本块 token 相同、但前文不同"的两个块得到不同的哈希，因为它们的 K/V 不同；
- 维护"哈希 → 物理块"的表；新请求到来时，沿着 prompt 的满块逐个查表，查到就复用（引用计数 +1），`num_computed_tokens` 直接跳过这些块。

【写】`minivllm/prefix_cache.py`。哈希函数先用 Python 自带的 `hash(tuple(...))`，它对整数元组在不同运行之间是稳定的。Phase III 读 vLLM 时，看它在哈希函数上做了什么选择、为什么。

【测】两个请求共享前 3 个满块的 prompt：第二个请求复用 3 个块，只计算剩余部分；输出和不开缓存时一致。

### Day 23（周二）· 前缀缓存 ②：淘汰

【写】
- 请求结束后，它的块引用计数归零，但**块里的内容和哈希都还在**：这些块仍然留在"哈希 → 块"表里，同时进入空闲队列。下一个请求命中它时，从空闲队列里摘出来、引用计数 +1。
- 分配新块时从空闲队列头部取（最久没用的）。如果取到的块还带着哈希，就把它从哈希表里删掉，这就是淘汰。
- **整段 prompt 都命中缓存时怎么办？** 想一想：模型要输出下一个 token，至少得算一个位置的 logits。把你的决定写进 docstring 并测试。Phase III 你会在 vLLM 里找到它的答案。

【测】淘汰顺序符合 LRU；被淘汰的块再请求时不命中；空闲但有哈希的块被命中时正确地从空闲队列移除。

### Day 24（周三）· 🧪 验证 + 读 nano-vllm 的块管理

【测】一组共享系统提示词的请求（比如 20 个请求共用 200 token 的前缀）：开 / 关前缀缓存时输出完全一致；记录命中的块数、省下的计算 token 数。
【读】(≤30 min，写完之后) `nanovllm/engine/block_manager.py`（约 120 行）：`compute_hash`、`allocate`、`may_append`。
【记】`notes/nanovllm_blocks.md`：它的哈希链、引用计数、淘汰和你的有什么不同？

### Day 25（周四）· 📐 Spec #4：`FreeBlockQueue`

见 [specs.md → Spec #4](specs.md#spec-4)。Day 23 里"命中一个空闲块就把它从空闲队列中间摘出来"，如果用 Python list 是 O(n)。这个 spec 要求 O(1)。vLLM 的 `FreeKVCacheBlockQueue` 就是为这件事存在的（Phase III 对照）。

### Day 26–27（周五、周六）· 🚪 Gate A

见 [gates.md → Gate A](gates.md#gate-a)。Day 27 包括 60 分钟闭卷重写 allocator + 哈希链，以及写 `notes/minivllm_vs_nanovllm.md`。通过后 `git tag vllm-gate-a`（加 `vllm-` 前缀，和 Plan 1 的 `gate-1…5` 区分）。

### Day 28（周日）· ⏸ Buffer
