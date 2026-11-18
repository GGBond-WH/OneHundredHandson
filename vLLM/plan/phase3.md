# Phase III · 源码深读：KV cache 路径（Day 50–70）

**目标**：把 `vllm/v1/core/` 的 KV cache 管理和调度读到"能闭卷讲清楚、能写测试证明"的程度；看清混合 KV、KV connector、KV offloading 这几块的地图；整理出至少 5 个可以在 Mac 上验证的 issue 候选。
**读法**：先预测再验证。每个文件都从**测试**和**断点**进入，不从第一行读到最后一行。
**产物**：`logs/probe_log.md` 累计 ≥ 15 条；`notes/design_comparison.md`；`notes/bug_archaeology.md`；`logs/issue_candidates.md`。
**Gate C**：Day 68–69。

---

## Week 8 · KV cache 管理器的三层结构

vLLM 的 KV cache 管理大致分三层（文件都在 `vllm/v1/core/`）：

```
KVCacheManager          kv_cache_manager.py   面向调度器：这个请求命中了多少？给它分配槽位；释放
  └ KVCacheCoordinator  kv_cache_coordinator.py   面向多个 KV cache 组（普通层、滑动窗口层……）
      └ SingleTypeKVCacheManager  single_type_kv_cache_manager.py   每种 attention 类型一个
          └ BlockPool   block_pool.py   物理块、哈希表、空闲队列（LRU）
```

### Day 50（周一）· 📖 kv_cache_utils.py：块、空闲队列、哈希

【读】`KVCacheBlock`（引用计数、块哈希）、`FreeKVCacheBlockQueue`、`hash_block_tokens`、`get_request_block_hasher`。
【对照】`FreeKVCacheBlockQueue` 对照你的 Spec #4：链接字段放在哪里？`remove` 怎么做到 O(1)？`hash_block_tokens` 对照你 Day 22 的哈希链。
【问】`generate_block_hash_extra_keys`：除了 token，哈希里还放了什么？为什么两个 token 完全相同的块有时**不能**共享？（提示：想想 LoRA、多模态输入，以及用户指定的 cache salt。）vLLM 在哈希函数上提供了什么选择、为什么不直接用 Python 的 `hash`？

### Day 51（周二）· 🔍 block_pool.py

【读】`BlockPool` 的 `get_cached_block`、`cache_full_blocks`、`get_new_blocks`、`touch`、`free_blocks`。
【问】
- `get_new_blocks` 从空闲队列拿到一个还带着哈希的块时，做了什么？对照你 Day 23 的淘汰逻辑。
- `free_blocks` 的 docstring 说参数要"按淘汰优先级排序，第一个最先被淘汰"。那么调用方传进来的顺序是什么？为什么一个请求的块要按这个顺序放回？（顺着调用链往上找。）

### Day 52（周三）· 🔍 kv_cache_manager.py

【读】`KVCacheManager.get_computed_blocks` 和 `allocate_slots`。用 `tests/v1/core/test_prefix_caching.py` 里的一个测试打断点进入，不要从文件开头读。
【找】Phase I Day 23 你自己做过的那个决定：整段 prompt 都命中缓存时怎么办？在 `get_computed_blocks` 附近找到 vLLM 的答案和那行注释。你的决定和它一样吗？
【记】`allocate_slots` 的输入（新 token 数、已命中的块、lookahead）和输出，以及它在什么情况下返回"分配失败"，调度器收到失败后会做什么。

### Day 53（周四）· 🔍 Probe #7–9

- 整段 prompt 命中：预测被计算的 token 数，用测试验证；
- 两个 token 完全相同、但 cache salt（或 LoRA）不同的请求，预测是否共享块；
- 请求结束 → 块进入空闲队列 → 新请求命中：预测命中后这些块的引用计数和在空闲队列里的位置。

### Day 54（周五）· 📰 PR 阅读 ×2

`git log --oneline -- vllm/v1/core/kv_cache_manager.py vllm/v1/core/block_pool.py | head -30`，挑 2 个。重点看：改动多大？加了什么测试？review 意见集中在哪？

### Day 55（周六）· 🧪 设计对比

【写】`notes/design_comparison.md`，至少 8 行：

| 维度 | minivllm（你写的） | vLLM | 差异的原因 |
|---|---|---|---|
| 块哈希 | | | |
| 空闲队列 | | | |
| 整段命中 | | | |
| 抢占策略 | | | |
| 预算与分块 prefill | | | |
| 块的释放顺序 | | | |
| 指标与事件（`kv_cache_metrics.py`、`take_events`） | 没有 | | 谁需要这些？ |
| 多种 attention 类型 | 没有 | | |

最后写两段：vLLM 有而你没有的东西里，哪一个最让你意外？你的设计里有没有一处你仍然认为更简单、更好？

### Day 56（周日）· 🔁 Retrieval（60 min）

闭卷写出 `allocate_slots` 的流程伪代码，以及它调用的 BlockPool 方法。写完对照源码修正。

---

## Week 9 · 调度器

### Day 57（周一）· 📖 schedule() 的主路径

`vllm/v1/core/sched/scheduler.py` 有 3888 行，`schedule()` 一个函数就有约 1500 行。**不要线性读。**

【做】在 `tests/v1/core/test_scheduler.py` 里找一个只有几个请求的简单测试，在 `schedule()` 入口打断点，单步跟着走一遍。只跟主路径：先处理运行中的请求，再处理等待中的请求；token 预算怎么扣；`allocate_slots` 失败时怎么抢占。遇到投机解码、结构化输出、多模态编码器、KV connector 的分支，一律跳过。
【记】主路径的流程图（15 行以内），每个节点标上行号。

### Day 58（周二）· 🔍 update_from_output

【读】`Scheduler.update_from_output`：采样结果怎么写回请求？停止条件在哪里判断？请求完成后它的块是怎么释放的？
【对照】你 Spec #3 的 `update()`。特别是"prompt 还没算完的请求不追加 token"这一条，vLLM 在哪里处理？

### Day 59（周三）· 🔍 Bug 考古

`tests/v1/core/` 里有一个文件名就叫 `test_priority_preemption_bug.py`，说明它是为某个真实 bug 写的回归测试。

【做】
```bash
git log --follow --oneline -- tests/v1/core/test_priority_preemption_bug.py
```
找到引入它的 commit 和对应的 PR，读 PR 描述和讨论。
【写】`notes/bug_archaeology.md`：bug 的症状是什么、根因在哪个函数、怎么修的、测试怎么证明修好了、review 里讨论了什么。这就是你 Phase IV 要写的那种 PR，现在先看一个真实的例子。

### Day 60（周四）· 🔍 Probe #10–12

- token 预算的边界：预算恰好等于（或比它少 1）某个请求剩余的 token 数时，预测这一步给它分多少；
- 优先级调度下的抢占：预测谁被抢占；
- 一个长 prompt 分块 prefill 期间，同一步里还有几个 decode 请求：预测每一步的调度结果。

### Day 61（周五）· 📰 PR 阅读 ×2

`git log --oneline -- vllm/v1/core/sched/ | head -30`，挑 2 个调度相关的 PR。

### Day 62（周六）· 🧪 异步调度

【读】`vllm/v1/core/sched/async_scheduler.py` 和 `tests/v1/core/test_async_scheduler.py`。
【问】它想把什么和什么重叠起来？这样做之后，调度器在决定下一步时，有哪些信息还不知道？它怎么处理这种"不知道"？
【记】一页笔记即可。这一块不要求读透。

### Day 63（周日）· 🔁 Retrieval（60 min）

闭卷默写 `schedule()` 主路径的流程图，对照 Day 57 的笔记修正。

---

## Week 10 · 地图的边缘 + issue 调研 + Gate C

### Day 64（周一）· 📖 混合 KV cache

【读】`docs/design/hybrid_kv_cache_manager.md`；`kv_cache_coordinator.py` 里 `UnitaryKVCacheCoordinator` 和 `HybridKVCacheCoordinator` 的区别；`single_type_kv_cache_manager.py` 里滑动窗口层的管理。
【问】滑动窗口层里，窗口外的块会怎样？对照你 Plan 1 的 `SlidingWindowCache`：那时你丢掉的是张量里的数据，这里丢掉的是什么？一个模型同时有全注意力层和滑动窗口层时，前缀缓存的"命中"怎么定义？

### Day 65（周二）· 📖 KV connector 与 offloading（只画地图）

【读】`vllm/distributed/kv_transfer/kv_connector/v1/base.py` 的 `KVConnectorBase_V1`：它分 scheduler 侧和 worker 侧两种角色，各自要实现哪些方法？再看一眼 `vllm/v1/kv_offload/` 的目录结构（`cpu/`、`tiering/`）。
【写】`notes/kv_transfer_map.md`：一张图说明 connector 在调度器和 worker 之间的位置。这是你 KV cache 研究方向在 vLLM 里最可能落地的地方，但这 100 天只画地图，不深入。

### Day 66（周三）· 📰 issue 调研

【做】在 vLLM 的 issue 里搜索（方法见 [contributing-playbook.md](contributing-playbook.md#find)）：KV cache、prefix caching、scheduler、preemption、CPU backend、`good first issue`。也翻一翻你 probe_log 里记下的"没被测试覆盖的行为"。
【写】`logs/issue_candidates.md`，**至少 5 个候选**，每个都判断：
- 能不能在你的 Mac（CPU 构建或 vllm-metal）上复现、验证？
- 预计改动多大？
- 有没有人在做（看评论、关联的 PR）？
- 是不是"凑数"类改动？

### Day 67（周四）· 🔍 Probe #13–15

- 滑动窗口层：预测请求进行到某个长度时，它持有的块数；
- 混合 KV 组下的前缀缓存命中；
- 一个从你 issue 候选表里来的问题，先用 Probe 验证这个问题真实存在。这一条可能直接变成 PR #1 的失败测试。

### Day 68–69（周五、周六）· 🚪 Gate C

见 [gates.md → Gate C](gates.md#gate-c)。通过后 `git tag vllm-gate-c`。

### Day 70（周日）· ⏸ Buffer

读一遍 [contributing-playbook.md](contributing-playbook.md)，明天开始选第一个 PR。
