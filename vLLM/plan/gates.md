# Gates · 入场条件与四个阶段门

和 Plan 1 一样：每项只答 yes / no，★ 为 critical，不打主观分。过程中违反 AI 规则或信息隔离规则，对应的独立完成项直接记 No。

| 结果 | 条件 | 处理 |
|---|---|---|
| **Pass** | ★ 全过，且总通过率 ≥ 80% | 进入下一阶段，打 tag |
| **Recoverable Fail** | ★ 全过但通过率 60–79%，或只有 1 个 ★ 没过 | 用 buffer day 只修失败的那一项，再做 60–90 分钟定向复测 |
| **Structural Fail** | 通过率 < 60%，或 ≥ 2 个 ★ 没过 | 最多追加 2 个学习日针对性训练再重考；后续砍 optional，不砍基础 |

中断协议（漏 ≤ 2 天用 buffer、3–4 天砍 optional、> 4 天整体后移）与 Plan 1 相同。

---

<a id="entry"></a>
## 入场 Gate（Day 0）

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | Plan 1 的 Gate 3 已通过（`git tag` 里有 `gate-3`）：能独立写出 prefill / decode、Dynamic / Static cache，并用数值等价测试证明 | |
| 2 ★ | 能独立写 pytest 的 fixture 和 parametrize，能用断点调试，熟悉本地 git（Plan 1 Week 1–9 的能力） | |
| 3 | Plan 1 的 Gate 4 已通过（HF 模型、`pyproject.toml`、benchmark） | |
| 4 | 有 GitHub 账号，会在网页上浏览 issue 和 PR | |
| 5 | 读英文技术文档没有障碍（散文可以借助翻译，代码和 API 名直接读英文） | |
| 6 | 装好了 Xcode Command Line Tools，磁盘有几十 GB 空闲（vLLM 源码、构建产物、模型） | |

★ 没过就不要开始：继续 Plan 1。这不是形式要求。Phase I 的每一个等价测试都以你 Plan 1 的 DynamicCache 为参照答案，没有它这个阶段无从开始。

---

<a id="gate-a"></a>
## Gate A（Day 26–27）· 自己写的 mini 推理服务引擎

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | 单请求：分页 KV 路径 prefill + 20 步 decode，每步 logits 与 Plan 1 DynamicCache `assert_close`，greedy 序列完全一致（`block_size` 取小值，跨越块边界） | |
| 2 ★ | 多请求：不同长度、不同时间加入的请求在扁平化混合 batch 里运行，每个的 greedy 输出与单独运行完全一致 | |
| 3 ★ | 块守恒：任何时刻"空闲 + 被持有 = num_blocks"（开前缀缓存时按引用计数计）；全部请求完成后无泄漏 | |
| 4 ★ | 抢占：`num_blocks` 很小、抢占确实发生时，所有输出仍与单独运行一致 | |
| 5 | 分块 prefill：与不分块的输出完全一致；prompt 没算完的请求不产出 token | |
| 6 | 前缀缓存：共享前缀的请求复用了块（有计数证据），输出与关闭缓存时一致 | |
| 7 | 淘汰顺序符合 LRU 的测试通过；"整段 prompt 命中"有明确处理并有测试 | |
| 8 | Spec #1–4 的冻结测试全部通过 | |
| 9 | 60 分钟闭卷重写 BlockAllocator + 块哈希链，测试通过 | |
| 10 | `notes/minivllm_vs_nanovllm.md` 至少 5 条差异，每条说明原因（必须包括"统一调度 vs prefill / decode 分离"） | |

Pass：★ 4/4 且 ≥ 8/10。通过后 `git tag vllm-gate-a`。

---

<a id="gate-b"></a>
## Gate B（Day 47–48）· 开发环境 + 使用 + 架构

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | vLLM 在 macOS 上从源码构建成功，`import vllm` 能打印版本，构建所用的 commit 记录在 `notes/setup.md` | |
| 2 ★ | 离线 `LLM.generate` 和 `vllm serve` + OpenAI 客户端都能用小模型在 CPU 上跑通 | |
| 3 ★ | `tests/v1/core/` 的 `test_kv_cache_utils.py`、`test_prefix_caching.py`、`test_scheduler.py` 在本地跑过，通过 / 失败 / 跳过数如实记录，每个失败都已分类（或标"未分类"） | |
| 4 ★ | `notes/lifecycle.md`：从 `LLM.generate` 到输出的 `file:function` 链，每一站都用断点核实过 | |
| 5 | `SchedulerOutput` 的关键字段已对应到 minivllm 的调度结果和 `StepInputs` | |
| 6 | GitHub 演练有证据：你 fork 里的演练 PR、带 `Signed-off-by` 的 commit、一次 rebase + `--force-with-lease` | |
| 7 | pre-commit 已安装，对一次改动运行结果为全绿 | |
| 8 | 用 Plan 1 的 `estimate_kv_bytes` 预测的块数与 vLLM 启动日志一致（或者能解释差在哪里） | |
| 9 | 引擎参数实验表：每组都有"预期 / 实际 / 解释"，记录了环境 | |
| 10 | `probe_log.md` ≥ 6 条（每条的预测都写在运行之前），`pr_reading_log.md` ≥ 4 条 | |

Pass：★ 4/4 且 ≥ 8/10。通过后 `git tag vllm-gate-b`。

---

<a id="gate-c"></a>
## Gate C（Day 68–69）· 源码深读

Day 68 先做第 1 项（闭卷），再逐项核对其余各项。

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | 闭卷写出一个新请求的接纳路径：`Scheduler.schedule` → `KVCacheManager.get_computed_blocks` → `allocate_slots` → BlockPool 的哪些方法；再对照源码，函数和顺序都没有错误（写错的地方改正并记录） | |
| 2 ★ | `probe_log.md` ≥ 15 条，每条的预测都写在运行之前 | |
| 3 ★ | `notes/design_comparison.md` ≥ 8 行，每行都有"差异原因" | |
| 4 ★ | `issue_candidates.md` ≥ 5 个候选，每个都判断了能否在 Mac 上验证、规模、是否有人在做 | |
| 5 | `notes/bug_archaeology.md`：一个回归测试追溯到它的 PR，写清症状、根因、修复、测试 | |
| 6 | 能指出"整段 prompt 命中时至少计算一个 token"在 vLLM 里的代码位置，并解释原因 | |
| 7 | 能解释 `free_blocks` 的参数顺序代表什么、调用方传的是什么顺序、为什么 | |
| 8 | 能解释滑动窗口层里窗口外的块会怎样，以及这和 Plan 1 的 `SlidingWindowCache` 的区别 | |
| 9 | `pr_reading_log.md` ≥ 12 条 | |
| 10 | Phase III 的每个周日 Retrieval 都完成了，并对照源码修正过 | |

Pass：★ 4/4 且 ≥ 8/10。通过后 `git tag vllm-gate-c`。

---

<a id="gate-d"></a>
## Gate D（Day 98–100）· 毕业

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | PR #1 已提交到 `vllm-project/vllm`，满足全部形式要求：标题带方括号标签、每个 commit 有 `Signed-off-by`、pre-commit 全绿、包含测试、按模板写了 Overview / Claims / Validation / Details | |
| 2 ★ | review 往返：每一条 review 意见都用"修改"或"解释"回应了；如果 7 天内没有 review，按指南礼貌地 ping 过或在 `#pr-reviews` 询问过（有记录） | |
| 3 ★ | PR 的代码和测试由你自己写；`ai_log.md` 审计通过；AI 使用的披露符合 vLLM 规定 | |
| 4 ★ | 第二份贡献：PR #2（vLLM 或 vllm-metal）达到同样标准，**或者** ≥ 3 条有实质内容的 issue 复现 / 分析评论（附最小复现） | |
| 5 ★ | Day 99–100 两场闭卷毕业考：推演结果被测试证实；bug 复现测试能稳定复现 | |
| 6 | 至少一个 PR 收到了维护者的正式 review（approve 或 request changes） | |
| 7 | review 过至少 2 个别人的 PR，每次都基于你自己跑过的结果 | |
| 8 | 失败测试先行：两个 PR 的 git 历史里都能看到"先测试、后修复" | |
| 9 | `probe_log.md` ≥ 15 条、`pr_reading_log.md` ≥ 16 条 | |
| 10 | `logs/contribution_log.md` 完整，并写了接下来想长期参与的子系统 | |

Pass：★ 5/5 且 ≥ 8/10。

**加分项（不计入判定）**：至少一个 PR 被合并。

为什么"合并"不是毕业条件：合并取决于维护者有没有时间、这个改动在当前路线图上的优先级、和你同时在改同一处代码的其他 PR。这些都不在你的控制范围内。毕业衡量的是你能不能把 PR 做到可合并的质量、能不能和维护者有效协作，这两件事完全由你决定。
