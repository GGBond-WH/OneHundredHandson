# 01 · 100 天日历（Day 1 = 周一）

每天一行：**做什么** → 当天产物。详细任务见对应的 phase 文件。
标记：🔨 Build · 📐 Spec · 🔍 Probe / 追踪源码 · 📖 读 · 📰 PR 阅读 / issue 调研 · 🧪 测试 / 实验 · 🤝 贡献 · 🔁 Retrieval · 🚪 Gate · ⏸ Buffer


## Phase I · 原理 + 自己写 mini 推理服务引擎（[phase1.md](phase1.md)）

Phase I 不读 vLLM 源码。所有参考实现（nano-vllm）都在你自己写完之后才读。

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 1 | 一 | 📖 Orca §1–3 + PagedAttention §1–3；用 Plan 1 的 estimate_kv_bytes 量出"按最大长度预留"浪费了多少；建 minivllm 骨架 | `notes/waste.md`、`minivllm/` 骨架 |
| 2 | 二 | 🔨 BlockAllocator：空闲池、分配 / 释放、引用计数、错误处理；块守恒的随机测试 | `allocator.py` + tests |
| 3 | 三 | 🔨 分页 KV 存储 [num_blocks, block_size, H_kv, D]；按槽位写入、读回 | `kv_store.py` + tests |
| 4 | 四 | 📐 Spec #1 `blocks_needed` + `compute_slot_mapping` | `paging.py` + 冻结测试 |
| 5 | 五 | 📖 PagedAttention §4（写完之后读）：block table、copy-on-write | `notes/paper_vs_mine.md` |
| 6 | 六 | 🧪 单请求分页注意力（gather 块 → attention）；与 Plan 1 DynamicCache 逐步等价 | 第一个等价测试 |
| 7 | 日 | 🔁 60 min 闭卷：BlockAllocator + slot mapping | `scratch/w1/` |
| 8 | 一 | 🔨 Request 对象（num_computed_tokens、block_table、状态）；多请求循环版 | `request.py` |
| 9 | 二 | 🔨 扁平化 token：所有请求拼成 [N, C]，只有 attention 按请求切分 | `model_runner.py` 初版 |
| 10 | 三 | 🔨 一步里混合 prefill 与 decode；与每个请求单独运行等价（含随机测试） | 混合 batch 等价测试 |
| 11 | 四 | 📐 Spec #2 `build_step_inputs` | `inputs.py` + 冻结测试 |
| 12 | 五 | 📖 nano-vllm `model_runner.py` 的 prepare_prefill / prepare_decode（写完之后读） | `notes/nanovllm_runner.md` |
| 13 | 六 | 🧪 静态批处理 vs 逐步混合 batch：完成时间与峰值 KV 占用 | `bench_batching.py` 结果表 |
| 14 | 日 | 🔁 60 min 闭卷：build_step_inputs | `scratch/w2/` |
| 15 | 一 | 📖 Sarathi 分块 prefill + V1 博客的统一调度；🔨 调度器骨架：队列、token 预算、max_num_seqs | `scheduler.py` 骨架 |
| 16 | 二 | 🔨 引擎循环 add_request / step；N 个请求输出 == 各自单独运行；无泄漏 | `engine.py` + 端到端测试 |
| 17 | 三 | 🔨 块不够时抢占（recompute）；强制抢占下输出不变 | 抢占测试 |
| 18 | 四 | 📐 Spec #3 调度器不变量（随机负载性质测试） | 冻结测试全绿 |
| 19 | 五 | 📖 nano-vllm `scheduler.py`（写完之后读）：prefill / decode 分离 vs 统一调度 | `notes/nanovllm_scheduler.md` |
| 20 | 六 | 🧪 分块 prefill：跨多步、prompt 未算完不产出 token、与不分块等价 | 分块测试 |
| 21 | 日 | 🔁 60 min 闭卷：带预算和抢占的 schedule() | `scratch/w3/` |
| 22 | 一 | 🔨 前缀缓存 ①：满块哈希 H(父哈希, 块内 token)；命中表；跳过已缓存的块 | `prefix_cache.py` |
| 23 | 二 | 🔨 前缀缓存 ②：空闲但有哈希的块、LRU 淘汰、整段 prompt 命中的处理 | 淘汰测试 |
| 24 | 三 | 🧪 共享前缀开 / 关缓存输出一致 + 命中统计；📖 nano-vllm `block_manager.py` | 对比表、`notes/nanovllm_blocks.md` |
| 25 | 四 | 📐 Spec #4 `FreeBlockQueue`（O(1) 从中间摘除） | `free_queue.py` + 冻结测试 |
| 26 | 五 | 🚪 **Gate A ①**：等价性、块守恒、抢占、前缀缓存现场演示 | `gate_a/` |
| 27 | 六 | 🚪 **Gate A ②**：60 min 闭卷重写 allocator + 哈希链；minivllm vs nano-vllm 五条差异；清单自评 | tag `vllm-gate-a` |
| 28 | 日 | ⏸ Buffer |  |

## Phase II · vLLM 使用者视角 + 开发环境 + 架构通读（[phase2.md](phase2.md)）

从这里起，周四是 🔍 Probe（先预测再验证），周五是 📰 PR 阅读。

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 29 | 一 | 🔨 fork + 部分克隆到 ~/code/vllm + upstream remote；uv 建 Python 3.12 环境；macOS 源码构建 CPU 版 | `notes/setup.md`（含 commit） |
| 30 | 二 | 🔨 当用户：离线 LLM.generate + `vllm serve` + OpenAI 客户端；VLLM_CPU_KVCACHE_SPACE | `notes/first_run.md` |
| 31 | 三 | 🧪 本地跑 tests/v1/core 三个文件，如实记录并分类失败；安装 pre-commit | `notes/test_baseline.md` |
| 32 | 四 | 🔨 GitHub 演练（只在自己的 fork 里）：分支、commit -s、fork 内 PR、冲突、rebase、--force-with-lease | `notes/github_workflow.md` |
| 33 | 五 | 📖 vllm-metal 架构；brew 安装稳定版，与 CPU 版对比生成速度 | 速度对比表 |
| 34 | 六 | 🧪 引擎参数实验：前缀缓存开 / 关、max_num_batched_tokens、block_size（先写预期） | `notes/engine_args.md` |
| 35 | 日 | 🔁 闭卷写出 PR 工作流命令 + 源码开发安装命令，再对照修正 |  |
| 36 | 一 | 📖 V1 博客 + `docs/design/arch_overview.md`；先画预测的请求生命周期 | `notes/lifecycle_predicted.md` |
| 37 | 二 | 🔍 断点追踪（VLLM_ENABLE_V1_MULTIPROCESSING=0）：LLM.generate → EngineCore.step → schedule → CPU model runner → update_from_output | `notes/lifecycle.md` |
| 38 | 三 | 🔍 SchedulerOutput 字段 ↔ minivllm 的 StepInputs；找到 CommonAttentionMetadata 的构造处 | 映射表 |
| 39 | 四 | 🔍 Probe #1–2：读 tests/v1/core/utils.py；前缀命中时的 num_computed_tokens | `logs/probe_log.md` |
| 40 | 五 | 📰 《Inside vLLM: Anatomy…》+ 第一次 PR 阅读（git log -- vllm/v1/core/） | `logs/pr_reading_log.md` |
| 41 | 六 | 🧪 读 vllm/v1/worker/cpu/model_runner.py + block_table.py：你 Mac 上实际运行的路径 | `notes/cpu_runner.md` |
| 42 | 日 | 🔁 闭卷默写生命周期 file:function 链，对照修正 |  |
| 43 | 一 | 📖 设计文档 paged_attention.md + prefix_caching.md；用 Plan 1 估算器预测块数，对照启动日志 | 预测 vs 实际 |
| 44 | 二 | 🔍 从启动日志反向找到决定块数的代码（kv_cache_config_builder / planning / utils） | 一段话说明 |
| 45 | 三 | 🔍 引擎参数在 scheduler.py / kv_cache_manager.py 里生效的位置 ↔ minivllm | 对照表 |
| 46 | 四 | 🔍 Probe #3–6：淘汰顺序、命中后 touch、抢占对象、整段 prompt 命中 | probe_log |
| 47 | 五 | 🚪 **Gate B ①** | `gate_b/` |
| 48 | 六 | 🚪 **Gate B ②** | tag `vllm-gate-b` |
| 49 | 日 | ⏸ Buffer |  |

## Phase III · 源码深读：KV cache 路径（[phase3.md](phase3.md)）

每个文件都从测试和断点进入，不从第一行读到最后一行。

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 50 | 一 | 📖 kv_cache_utils.py：KVCacheBlock、FreeKVCacheBlockQueue（对照 Spec #4）、hash_block_tokens、extra keys | 笔记 |
| 51 | 二 | 🔍 block_pool.py：get_cached_block / cache_full_blocks / get_new_blocks / touch / free_blocks | 笔记 |
| 52 | 三 | 🔍 kv_cache_manager.py：get_computed_blocks（整段命中规则）+ allocate_slots，从测试打断点进入 | 笔记 |
| 53 | 四 | 🔍 Probe #7–9：整段命中、cache salt / LoRA 不共享、释放后再命中 | probe_log |
| 54 | 五 | 📰 PR 阅读 ×2（kv_cache_manager / block_pool） | pr_reading_log |
| 55 | 六 | 🧪 minivllm vs vLLM 设计对比（≥ 8 行，每行写原因） | `notes/design_comparison.md` |
| 56 | 日 | 🔁 闭卷：allocate_slots 流程伪代码 |  |
| 57 | 一 | 📖 schedule() 主路径：从 test_scheduler.py 打断点单步跟，只跟主路径 | 带行号的流程图 |
| 58 | 二 | 🔍 update_from_output：采样回写、停止条件、释放 | 笔记 |
| 59 | 三 | 🔍 Bug 考古：test_priority_preemption_bug.py → 引入它的 PR → 讨论 | `notes/bug_archaeology.md` |
| 60 | 四 | 🔍 Probe #10–12：预算边界、优先级抢占、分块 prefill 与 decode 混合 | probe_log |
| 61 | 五 | 📰 PR 阅读 ×2（vllm/v1/core/sched/） | pr_reading_log |
| 62 | 六 | 🧪 异步调度：async_scheduler.py + test_async_scheduler.py | 一页笔记 |
| 63 | 日 | 🔁 闭卷默写 schedule() 主路径 |  |
| 64 | 一 | 📖 hybrid_kv_cache_manager.md；Unitary vs Hybrid coordinator；滑动窗口层的块释放 | 笔记 |
| 65 | 二 | 📖 KV connector（base.py 两种角色）+ vllm/v1/kv_offload/：只画地图 | `notes/kv_transfer_map.md` |
| 66 | 三 | 📰 issue 调研：KV cache / 前缀缓存 / 调度 / CPU；判断可验证性、规模、是否有人在做 | `logs/issue_candidates.md`（≥ 5） |
| 67 | 四 | 🔍 Probe #13–15：滑动窗口持有块数、混合 KV 命中、一个候选 issue 的复现 | probe_log |
| 68 | 五 | 🚪 **Gate C ①**：闭卷写接纳路径 + 对照源码 | `gate_c/` |
| 69 | 六 | 🚪 **Gate C ②**：其余各项核对 | tag `vllm-gate-c` |
| 70 | 日 | ⏸ Buffer：读一遍 contributing-playbook.md |  |

## Phase IV · 贡献：issue → PR → review（[phase4.md](phase4.md)）

PR #1 必须在 Day 75 左右提交，给 review 留出至少三周。等 review 时推进另一条线，不催。

| Day | 周 | 任务 | 产物 |
|---|---|---|---|
| 71 | 一 | 🤝 选定 PR #1（选题标准）；在 issue 下认领，或先开 issue | 认领评论 |
| 72 | 二 | 🤝 从 upstream/main 拉分支；先写失败的测试，commit -s | 失败测试 |
| 73 | 三 | 🤝 修复；相关测试文件全跑、对照基线；pre-commit 全绿 | 修复 commit |
| 74 | 四 | 🤝 过提交前清单；按模板写 Overview / Claims / Validation / Details；起标题 | PR 描述 |
| 75 | 五 | 🤝 rebase 到最新 main，提交 PR #1（勾选 Allow edits by maintainers）；读 2 个同领域 PR | PR #1 链接 |
| 76 | 六 | 🤝 review 别人的一个 PR：拉到本地跑测试，留一条有内容的评论 | review 评论 |
| 77 | 日 | 🔁 闭卷用 5 句英文讲清 PR #1；周复盘 |  |
| 78 | 一 | 🤝 处理 PR #1 的 review（没有就按等待协议，不催） | 回复 |
| 79 | 二 | 🤝 PR #2 选题：bug / 未覆盖分支（pytest --cov）/ 用户可见但没写文档的行为 / vllm-metal | 选题记录 |
| 80 | 三 | 🤝 涉及设计选择的话先在 issue 下写方案，等维护者表态 | 方案评论 |
| 81 | 四 | 🤝 PR #2：失败测试 | 失败测试 |
| 82 | 五 | 📰 PR 阅读 ×2（和 PR #2 同一个文件的历史 PR） | pr_reading_log |
| 83 | 六 | 🤝 PR #2：实现 |  |
| 84 | 日 | 🔁 周复盘 + 回复 review |  |
| 85 | 一 | 🤝 PR #2：实现完成，相关测试全绿 |  |
| 86 | 二 | 🤝 PR #2：Mac 上端到端验证；性能改动给前后数据并注明只测了 CPU | 验证记录 |
| 87 | 三 | 🤝 用户可见的改动：同一 PR 更新文档 |  |
| 88 | 四 | 🤝 PR #2：描述 + 清单 + rebase |  |
| 89 | 五 | 🤝 提交 PR #2 | PR #2 链接 |
| 90 | 六 | 🤝 两个 PR 的 review 往返 |  |
| 91 | 日 | 🔁 周复盘 |  |
| 92 | 一 | 🤝 review 往返；在 Mac 上复现一个别人报告的 bug，给出最小复现 | issue 评论 |
| 93 | 二 | 🤝 review 别人的第二个 PR | review 评论 |
| 94 | 三 | 🤝 review 往返；需要时 rebase |  |
| 95 | 四 | 🤝 贡献记录：所有链接、状态、收获；想长期参与的子系统 | `logs/contribution_log.md` |
| 96 | 五 | 🤝 review 往返 / 收尾 |  |
| 97 | 六 | ⏸ Buffer |  |
| 98 | 日 | 🚪 **Gate D ① 审计**：逐项过毕业清单 |  |
| 99 | 一 | 🚪 **Gate D ② 毕业考 Part 1**（3 h）：闭卷推演前缀缓存 + 块分配场景，再用测试验证 | `final_exam/day99/` |
| 100 | 二 | 🚪 **Gate D ③ 毕业考 Part 2**（3 h）：从 bug 描述写出复现测试；最终复盘 | `final_exam/day100/`、`logs/final_reflection.md` |
