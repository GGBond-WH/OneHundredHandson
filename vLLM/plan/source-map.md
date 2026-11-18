# vLLM 源码地图 · KV cache 路径

> 2026-11-17 对照 `vllm-project/vllm` 的 `main`（v0.34.0 于前一天发布）核实。vLLM 迭代很快，文件会改名或拆分。找不到某个名字时，用 `git log --follow -- <旧路径>` 和 `grep -rn "<名字>" vllm/` 定位，**并把变化记进 [99-notes.md](99-notes.md)**。

<a id="debug"></a>
## 调试开关

| 环境变量 | 作用 |
|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 让 EngineCore 在主进程里运行（默认是 1，跑在子进程里）。**打断点必须设它** |
| `VLLM_LOGGING_LEVEL=DEBUG` | 更详细的日志 |
| `VLLM_CPU_KVCACHE_SPACE=<GB>` | CPU 后端的 KV cache 空间，整数，单位 GB，不设时为 4 |

## 一个请求经过的地方（Phase II Day 37 用断点逐站核实）

```
LLM.generate                         vllm/entrypoints/llm.py
  └ LLMEngine                        vllm/v1/engine/llm_engine.py
      ├ InputProcessor               vllm/v1/engine/input_processor.py      prompt → 请求
      ├ EngineCoreClient             vllm/v1/engine/core_client.py          多进程时负责跨进程通信
      │   └ EngineCore.step          vllm/v1/engine/core.py
      │       ├ Scheduler.schedule   vllm/v1/core/sched/scheduler.py        → SchedulerOutput（vllm/v1/core/sched/output.py）
      │       ├ 执行模型               worker：vllm/v1/worker/cpu_worker.py（你的 Mac）/ gpu_worker.py
      │       │                       model runner：vllm/v1/worker/cpu/model_runner.py / gpu/model_runner.py
      │       └ Scheduler.update_from_output
      └ OutputProcessor              vllm/v1/engine/output_processor.py     detokenize、组装输出
```

注意：GPU 侧的 model runner 已经是 "Model Runner V2"（`vllm/v1/worker/gpu/`，设计文档 `docs/design/model_runner_v2.md`）。你在 Mac 上实际运行的是 `vllm/v1/worker/cpu/`，它有自己的 `model_runner.py` 和 `block_table.py`。

## KV cache 管理（`vllm/v1/core/`）· Phase III 的主战场

| 文件 | 关键名字 | 读的时候要回答的问题 |
|---|---|---|
| `kv_cache_utils.py` | `KVCacheBlock`、`FreeKVCacheBlockQueue`（`popleft` / `remove` / `append`）、`hash_block_tokens`、`get_request_block_hasher`、`generate_block_hash_extra_keys` | 空闲队列怎么做到 O(1) 删除？块哈希里除了 token 还有什么？为什么？ |
| `block_pool.py` | `BlockPool`：`get_cached_block`、`cache_full_blocks`、`get_new_blocks`、`touch`、`free_blocks`、`evict_blocks`、`reset_prefix_cache`；`BlockHashToBlockMap` | 拿到一个带哈希的空闲块时怎么淘汰？`free_blocks` 的参数顺序代表什么？ |
| `kv_cache_manager.py` | `KVCacheManager`：`get_computed_blocks`、`allocate_slots`、`free`、`cache_blocks`；`KVCacheBlocks` | 整段 prompt 命中时怎么处理（找那条注释）？`allocate_slots` 什么时候失败？ |
| `kv_cache_coordinator.py` | `KVCacheCoordinator`、`UnitaryKVCacheCoordinator`、`HybridKVCacheCoordinator`、`find_longest_cache_hit` | 只有一种 attention 和有多种 attention 时，命中的计算有什么不同？ |
| `single_type_kv_cache_manager.py` | 每种 attention 类型一个管理器（全注意力、滑动窗口等） | 滑动窗口层怎么释放窗口外的块？ |
| `kv_cache_config_builder.py`、`kv_cache_planning.py` | 决定块数和 KV cache 布局 | 可用内存 → 块数的链条在哪？ |
| `kv_cache_metrics.py` | KV cache 指标 | 谁需要这些指标？ |

## 调度（`vllm/v1/core/sched/`）

| 文件 | 关键名字 | 备注 |
|---|---|---|
| `scheduler.py` | `Scheduler`：`schedule`、`update_from_output`、`add_request`、`finish_requests` | 3888 行，`schedule()` 约 1500 行。**从测试打断点进入，只跟主路径** |
| `output.py` | `SchedulerOutput` | 调度器交给 worker 的全部信息 |
| `request_queue.py` | 等待队列（FCFS / 优先级） | |
| `async_scheduler.py` | 异步调度 | Phase III Day 62 |
| `interface.py` | 调度器接口 | |

## Attention 元数据

`vllm/v1/attention/backend.py` 的 `CommonAttentionMetadata`：`query_start_loc`、`seq_lens`、`slot_mapping`、`block_table_tensor`、`num_actual_tokens`。名字和你 Phase I 的 `StepInputs` 一致。具体的 attention 实现在 `vllm/v1/attention/backends/`。

## KV 传输与卸载（Phase III Day 65 只画地图）

| 位置 | 内容 |
|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | `KVConnectorBase_V1`：scheduler 侧和 worker 侧两种角色 |
| `vllm/distributed/kv_transfer/kv_connector/v1/` | 各种 connector：`lmcache_connector.py`、`offloading_connector.py`、`simple_cpu_offload_connector.py`、`nixl/`、`mooncake/`、`example_connector.py` …… |
| `vllm/v1/kv_offload/` | KV 卸载：`cpu/`、`tiering/`、`factory.py` |

## 测试（`tests/v1/core/`）

不依赖 GPU（已核实：`test_prefix_caching.py` 和 `test_scheduler.py` 里没有 CUDA 依赖），是你在 Mac 上理解和验证 KV cache 行为的主要工具。

| 文件 | 用途 |
|---|---|
| `utils.py` | 造请求、造调度器的辅助函数。**写 Probe 测试前先读它** |
| `test_kv_cache_utils.py` | 块、空闲队列、哈希 |
| `test_prefix_caching.py` | 前缀缓存（约 7000 行，按测试名挑着读） |
| `test_scheduler.py` | 调度器（约 9000 行，同上） |
| `test_single_type_kv_cache_manager.py` | 每种 attention 的管理器 |
| `test_priority_preemption_bug.py` | 一个真实 bug 的回归测试（Phase III Day 59 的考古对象） |
| `test_async_scheduler.py` | 异步调度 |
| `prefix_cache/` | 前缀缓存的更多测试 |

## 设计文档（仓库内 `docs/design/`）

`arch_overview.md` · `paged_attention.md` · `prefix_caching.md` · `hybrid_kv_cache_manager.md` · `model_runner_v2.md` · `attention_backends.md` · `plugin_system.md`（vllm-metal 就是通过它接入的）· `metrics.md`

## 找"最近谁在改这里"

vLLM 用 squash 合并，`main` 上一个 commit 就是一个 PR，标题末尾带 PR 号：

```bash
git log --since="3 months ago" --oneline -- vllm/v1/core/            # 最近的 PR
git log --since="3 months ago" --format="%an" -- vllm/v1/core/ | sort | uniq -c | sort -rn | head   # 最活跃的作者
git log --follow --oneline -- <文件>                                 # 一个文件的完整历史
```
