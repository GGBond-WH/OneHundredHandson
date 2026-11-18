# vLLM 百日方案 · 从读懂到提 PR

> 目标：第 100 天，你已经向 `vllm-project/vllm` 提交过**符合项目全部规范**的 PR（标题标签、DCO 签名、pre-commit 全绿、带测试、按模板写清证据），并且和维护者完成过至少一轮 review 往返。
> 更重要的是：拿到一个 KV cache / 调度相关的 issue，你知道去哪几个文件找、怎么先写一个会失败的测试复现它、怎么改、怎么证明改对了。

这份方案是 [Plan 1（One Hundred Days of KV Cache）](../README.md) 的续篇，沿用它已经被验证有效的机制：每日一行的日历、Build / Spec / Retrieval 轨道、yes/no 客观 Gate、失败协议、分阶段的 AI 规则。所有引用的源码路径、类名、测试文件、文档和论文链接都在 2026-11-17 对照 vLLM `main`（v0.34.0 刚发布）核实过，记录见 [plan/99-notes.md](plan/99-notes.md)。

## 先看入场条件

**这份方案不能和 Plan 1 并行。** 它的 Phase I 直接建立在你 Plan 1 Phase III 写的 KV cache 代码上，Phase III 读 vLLM 源码时默认你已经亲手写过 prefill / decode / Dynamic / Static cache。

- **硬条件**：Plan 1 的 Gate 3 已通过（`git tag gate-3` 存在）。
- **推荐**：Plan 1 全部完成（Gate 5）。Plan 1 Week 10–11 的 HF 模型、`pyproject.toml`、benchmark 经验会让这里的 Phase II 轻松很多。

完整入场清单见 [plan/gates.md → 入场 Gate](plan/gates.md#entry)。

## 硬件：没有 CUDA 怎么给 vLLM 做贡献

| 工具 | 用途 | 依据 |
|---|---|---|
| **vLLM macOS CPU 版**（从源码构建） | 主力开发环境：跑模型、打断点追踪、跑单元测试 | 官方文档："vLLM has experimental support for macOS with Apple Silicon"，必须从源码构建，支持 FP32/FP16 |
| **`tests/v1/core/` 单元测试** | 你的主战场。KV cache 管理器、前缀缓存、调度器的测试不依赖 GPU | 已核实：`test_prefix_caching.py`（7010 行）、`test_scheduler.py`（9060 行）中没有任何 CUDA 依赖 |
| **vllm-metal**（`vllm-project` 官方组织下的社区硬件插件） | 在你 Mac 的 GPU 上跑 vLLM；也是第二个可以提 PR 的地方 | 上游 vLLM 提供调度器和分页块管理器，vllm-metal 负责 Metal 上的 paged attention |
| 按小时租 GPU | **可选，不计入任何 Gate**。只在你想验证某个 CUDA 路径时用，预算由你决定 | — |

所以这份方案的贡献方向是有意选择的：**调度器、KV cache 管理器、前缀缓存、CPU 路径、测试与文档**。这些地方的正确性在你的 Mac 上就能完整验证，而且正好是你研究的 KV cache 方向。

## 五条轨道

| 轨道 | 训练什么 | 什么时候 |
|---|---|---|
| **Build** | 自己先写一个 mini 推理服务引擎（分页 KV、连续批处理、抢占、分块 prefill、前缀缓存），再读 vLLM 就是"认出来"而不是"从头学" | Phase I |
| **Spec** | 只给规格 + 冻结的黑盒测试，自己设计实现（和 Plan 1 相同） | Phase I 每周四 |
| **Probe**（先预测再验证） | 读一段 vLLM 源码 → **先写下你预测的行为** → 写一个测试去验证 → 记录对错。读大型代码库最有效的方法 | Phase II–III 每周四 |
| **PR 阅读** | 每周读 2 个已合并的 KV/调度相关 PR：问题、改动、测试、review 意见。培养"什么样的 PR 能被合并"的判断 | Week 6 起每周五 |
| **Retrieval** | 每周日 60 分钟闭卷：Phase I 重写模块，Phase II–III 默写调用链 | 每周日（Gate 周除外） |

Phase IV（贡献阶段）里，周四和周五变成 PR 工作日。

## 日历骨架（Day 1 = 周一）

| 阶段 | 学习日 | Gate | Buffer | 产物 |
|---|---|---|---|---|
| I  原理 + 自己写 mini 推理服务引擎 | Day 1–25 | Day 26–27 | Day 28 | `minivllm/`：分页 KV、扁平化 batch、调度器、抢占、分块 prefill、前缀缓存 |
| II vLLM 使用者视角 + 开发环境 + 架构通读 | Day 29–46 | Day 47–48 | Day 49 | 源码构建的 vLLM、GitHub 工作流、请求生命周期图、前 6 个 Probe |
| III 源码深读：KV cache 路径 | Day 50–67 | Day 68–69 | Day 70 | 15+ 个 Probe、设计对比、issue 候选清单 |
| IV 贡献：issue → PR → review | Day 71–96 | Day 98–100 | Day 97 | PR #1、PR #2、review 记录、毕业考 |

## 每周节奏

| 阶段 | 周一 | 周二 | 周三 | 周四 | 周五 | 周六 | 周日 |
|---|---|---|---|---|---|---|---|
| I | 新概念 / 论文 | 首次实现 | 完善 + debug | 📐 Spec | 读参考实现（写完之后） | 测试 / benchmark | 🔁 Retrieval |
| II–III | 文档 / 设计 | 追踪源码（断点） | 映射到 minivllm | 🔍 Probe | 📰 PR 阅读 | 实验 | 🔁 Retrieval |
| IV | 选题 / review 回复 | 写失败测试 | 修复 | 验证 + 写 PR 说明 | 提交 / review 别人 | review 回复 | 🔁 复盘 |

每天 2–3 小时。时间分配和 Plan 1 一样：阅读不超过 20–25%，其余是写代码、写测试、debug。

## 文件导航

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| [plan/00-rules.md](plan/00-rules.md) | 硬规则：AI 政策（含 vLLM 官方的 AI 辅助贡献规定）、GitHub 工作流、DCO、PR 礼仪 | **Day 1 先读** |
| [plan/01-calendar.md](plan/01-calendar.md) | 100 天，每天一行 | 每天早上 |
| [plan/phase1.md](plan/phase1.md) … [phase4.md](plan/phase4.md) | 每天的详细任务 | 进入阶段时 |
| [plan/gates.md](plan/gates.md) | 入场 Gate + Gate A–D 的 yes/no 清单 + 失败协议 | 阶段末 |
| [plan/specs.md](plan/specs.md) | Phase I 的 4 个 Spec Challenge | Phase I 周四 |
| [plan/source-map.md](plan/source-map.md) | vLLM KV cache 路径的源码地图（文件、类、函数、测试、调试开关） | Phase II 起 |
| [plan/contributing-playbook.md](plan/contributing-playbook.md) | 找 issue、认领、写 PR、过 CI、回 review 的全流程，附英文模板 | Phase IV（Phase II 先读一遍） |
| [plan/resources.md](plan/resources.md) | 论文、博客、文档、仓库（全部已核实） | 需要时 |
| [plan/99-notes.md](plan/99-notes.md) | 核实记录、假设、未决问题 | 有疑问时 |
| [templates/](templates/) | Probe 记录、PR 阅读记录、issue 候选表、PR 草稿 | Day 1 复制到 `logs/` |

## 目录约定

- 本目录 `vLLM/` 放计划、笔记、日志和 Phase I 的 `minivllm/` 代码。
- **vLLM 源码克隆到这个仓库外面**（比如 `~/code/vllm`），不要克隆进 `One Hundred/`。在一个 git 仓库里嵌套另一个仓库会让两边的 `git status` 都乱掉。

## 明确不在这 100 天里

CUDA / Triton kernel 开发 · FlashAttention / FlashInfer 内部实现 · CUDA graph 与 `torch.compile` · 分布式（TP / PP / EP / DP）· MoE · 量化 · 多模态 · 推测解码内部实现。这些在 vLLM 里都是重要的子系统，但它们要么需要 GPU 才能验证，要么离 KV cache 主线太远。100 天后你可以按兴趣继续，方法是一样的。
