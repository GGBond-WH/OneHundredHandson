# One Hundred — LLM 动手能力百日方案

> 目标：第 100 天，面对一个**空目录**，只查官方文档，不看教程、不复制粘贴、不让 AI 写代码，
> 独立写出一个围绕 LLM 推理 / KV cache 的完整项目（代码结构、测试、benchmark、README）。

这套方案由 Claude 与 GPT 经五轮讨论后达成共识（讨论记录见 [plan/99-discussion.md](plan/99-discussion.md)）。
学习者画像：懂 KV cache 理论，PyTorch 与动手能力弱；硬件 Apple M4 Pro / 48GB / MPS，无 CUDA；每天 2–3 小时。

## 一句话原则

**看得懂 ≠ 写得出。** 这 100 天里，"读"只占 20%，其余全部是自己从空白文件写、跑、错、debug、测。

## 文件导航

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| [plan/00-rules.md](plan/00-rules.md) | 硬规则、AI 使用政策、20 分钟规则、git 制度、进度指标 | **Day 1 先读，之后每周复读** |
| [plan/01-calendar.md](plan/01-calendar.md) | 100 天日历：每天一行 | 每天早上 |
| [plan/phase1.md](plan/phase1.md) | Phase I（Day 1–14）Python/PyTorch 最小工程能力 | 进入该阶段时 |
| [plan/phase2.md](plan/phase2.md) | Phase II（Day 15–35）从零写 GPT | |
| [plan/phase3.md](plan/phase3.md) | Phase III（Day 36–56）Prefill / Decode / KV cache / RoPE / GQA | |
| [plan/phase4.md](plan/phase4.md) | Phase IV（Day 57–77）测试加固 / 真实 HF 模型 / 打包 / benchmark | |
| [plan/phase5.md](plan/phase5.md) | Phase V（Day 78–100）独立 Capstone：MiniKVLab + 毕业考 | |
| [plan/gates.md](plan/gates.md) | 5 个 Gate 的 yes/no 清单 + 失败协议 | 每个阶段末 |
| [plan/specs.md](plan/specs.md) | W5–W11 每周四 Spec Challenge 的正式规格 | 每周四 |
| [plan/capstone-spec.md](plan/capstone-spec.md) | Capstone 的产品需求（只有需求，没有架构） | Day 78 |
| [plan/tensor-checklist.md](plan/tensor-checklist.md) | 张量操作闭卷清单（Week 6 前必须会） | Week 2 起 |
| [plan/resources.md](plan/resources.md) | 全部资料链接（已核实可访问） | 需要时 |
| [templates/](templates/) | daily_log / bugs / ai_log / design_comparison / weekly_review 模板 | Day 1 复制到 `logs/` |
| [week09_bughunt/](week09_bughunt/) | Week 9 的"埋 bug 实现"（答案已封存，Day 60 前不要打开） | Day 60 |

## 三条轨道

| 轨道 | 回答的问题 | 频率 |
|---|---|---|
| **Build** | 我懂不懂？——学新东西并实现 | 周一至周三 |
| **Spec** | 面对没见过实现的需求，我能不能自己设计？——只给规格+黑盒测试 | Week 5 起每周四 |
| **Retrieval** | 不靠资料我还写不写得出来？——60 分钟闭卷重写一个模块 | 每周日（Gate 周除外） |

## 日历骨架（Day 1 = 周一）

| 阶段 | 学习日 | Gate | Buffer | 产物 |
|---|---|---|---|---|
| I  Python + PyTorch 最小工程能力 | Day 1–12 | Day 13 | Day 14 | `mini_lm/` Bigram 语言模型 + Basic BPE |
| II 从零写 Decoder-only Transformer | Day 15–32 | Day 33–34 | Day 35 | `tinygpt/` 可训练可采样的 GPT |
| III Prefill / Decode / KV cache | Day 36–53 | Day 54–55 | Day 56 | Dynamic / Static / Sliding cache + 等价测试 + RoPE/GQA |
| IV 工程化 + 真实模型 + benchmark | Day 57–74 | Day 75–76 | Day 77 | 可安装的 `mini_kv` 包 + benchmark CLI |
| V  独立 Capstone | Day 78–96 | Day 98 审计 + Day 99–100 闭卷考 | Day 97 | `minikvlab/` |

## 每周节奏

| 周一 | 周二 | 周三 | 周四 | 周五 | 周六 | 周日 |
|---|---|---|---|---|---|---|
| 新概念 + 最小实验 | 核心模块首次实现 | 完善 + debug | **Spec Challenge**（W5 起） | 限定范围读源码（Gate 周跳过） | 测试 + benchmark + refactor | **60 min Retrieval Drill**（Gate 周由 Gate 取代） |

每日时间（按 150 分钟）：阅读 20–30 / 独立编码 75–90 / debug + 测试 30–40 / 日志 10–15。多出的时间只给编码，不给视频。

## 怎么开始

1. 读完 [plan/00-rules.md](plan/00-rules.md)。
2. 打开 [plan/phase1.md](plan/phase1.md) 的 Day 1，照做（Day 1 的第一件事就是在**这个目录**里 `git init`——你的代码和这份方案放在同一个仓库里）。
3. 把 `templates/` 里的模板复制到 `logs/`，每天写 `daily_log.md`。
4. 每天结束前：至少一个有意义的 commit。

## 100 天后你应该是什么样

别人说"实现一个 sliding-window KV cache"，你不会先去搜 GitHub，而是开始想：接口要什么？cache tensor 的 shape？什么时候 update？position 怎么办？mask 怎么办？reference 是什么？怎么证明正确？怎么 benchmark？——然后建 `cache.py / attention.py / generation.py / test_cache.py / benchmark.py`，开始写。不知道的 API，查文档。
