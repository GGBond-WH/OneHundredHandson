# Gates · 五个阶段门的 yes/no 清单与失败协议

原则：**不打主观百分比**。每项只答 yes/no；★ 为 critical。过程中违反"旧代码 / 实现型 AI"规则，对应的独立实现项直接记 No。

## 统一判定

| 结果 | 条件 | 处理 |
|---|---|---|
| **Pass** | 全部 ★ 通过，且总通过率 ≥ 80%（Gate 5：★ 5/5 且 ≥ 9/10） | 进入下一阶段；buffer day 休息 / 轻复盘 / 偿还欠账；打 tag |
| **Recoverable Fail** | 全部 ★ 通过但总通过率 60–79%；或仅 1 个 ★ 未通过 | **不重学整个阶段**。buffer day 只修失败维度（例如只练 causal mask + 测试），然后 60–90 min 定向复测 |
| **Structural Fail** | 通过率 < 60%；或 ≥ 2 个 ★ 未通过 | 不进主线。最多追加 2 个学习日：失败模块最小化 → 针对训练 → 重新 Gate。后续砍 optional（top-p、quantized cache、额外源码阅读、benchmark 扩展），**不砍基础** |

## 中断协议（生病 / 加班）

buffer day 首先是日历缓冲，其次才是强化日。
- 本阶段漏 ≤ 2 天：用 buffer 补。
- 漏 3–4 天：删本阶段 optional 项，不靠每天多学 5 小时追。
- 漏 > 4 天：整个计划向后平移。**真正不能破坏的是能力依赖关系，不是日历上的第 100 天。** attention 还写不出来，绝不因为"今天已经 Day 36"进 KV cache。

## 过程证据（每个 Gate 都查）

- 违规查旧实现次数 = 0
- 违规让 AI 写实现次数 = 0
- 为迁就实现而修改冻结测试次数 = 0
- `bugs.md` / `ai_log.md` 有审计痕迹

（"卡住超过 20 分钟次数"**不**作为指标：它会奖励回避难题，且难以客观界定。）

---

<a id="gate-1"></a>
## Gate 1（Day 13）· Python / PyTorch 最小独立工程能力

设置：空目录 `gate1/`，3 小时，只查官方文档，完成 Bigram LM 小项目（data / model / train / generate / tests）。

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | 不打开此前的 Bigram 实现、不复制教程代码，建立完整项目骨架 | |
| 2 ★ | `pytest` 全绿，≥ 5 个 mandatory tests（roundtrip / x-y shift / shape / one-batch loss 下降 / checkpoint roundtrip） | |
| 3 ★ | tokenizer `decode(encode(x)) == x` 对测试文本成立 | |
| 4 ★ | dataset 的 x/y 确实满足 next-token shift（不是 shape 对了语义错） | |
| 5 | `model(x)` 输出 shape 正确 `[B,T,V]` | |
| 6 | 固定小 batch 训练后 loss 明显低于初始 | |
| 7 | save → 新进程 load → 同输入 logits 一致 | |
| 8 | `generate()` 能从 prompt 产生指定数量的新 token | |
| 9 | 至少一次自己通过 traceback / debugger / assert 解决 bug 并记入 bugs.md | |
| 10 | 3 小时内完成，未用 AI 生成实现 | |

Pass：★ 4/4 且 ≥ 8/10。

<a id="gate-2"></a>
## Gate 2（Day 33–34）· 独立实现 Decoder-only Transformer

设置：`gate2/`，可复制自己的 tokenizer / data / train harness / tests 骨架；`model.py` 从空白开始，3 小时；Day 34 做第 9、10 项。

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | 独立实现 `CausalSelfAttention`，未打开过去实现 | |
| 2 ★ | 因果测试通过：修改未来 token 不影响之前位置的输出 | |
| 3 ★ | 独立实现 `MLP → Block → GPT`，forward 全链路可运行 | |
| 4 ★ | 模型能在小数据上过拟合一个 batch | |
| 5 | ≥ 3 组 `(B,T,C,H)` 的 attention shape 测试全过 | |
| 6 | 手写 attention 与 `F.scaled_dot_product_attention` 数值接近 | |
| 7 | residual / LayerNorm 位置符合自己写的 `ARCHITECTURE.md` | |
| 8 | greedy / temperature / top-k 中至少 greedy 与 top-k 正常 | |
| 9 | **不看代码**写出 q, k, v, scores, softmax, out 每步 tensor shape（B=2,T=8,C=32,H=4） | |
| 10 | 60 分钟内完成一个指定核心模块的闭卷重构 | |

Pass：★ 4/4 且 ≥ 8/10。

<a id="gate-3"></a>
## Gate 3（Day 54–55）· Prefill / Decode / KV cache

设置：`gate3/`，用自己的 Phase III 代码作答（这是能力审计而非从零重写）；但第 7、8 项要在一个新文件里现场写。若 GQA Spec 顺延，GQA 不计入本 Gate。

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | full recomputation 与 cached decoding 的 greedy token 序列完全一致 | |
| 2 ★ | 相同位置的 cache / no-cache logits 通过 `torch.testing.assert_close` | |
| 3 ★ | 每层 cache 的 K/V shape、seq_len 符合预先写出的契约 | |
| 4 ★ | `DynamicCache` 与 `StaticCache` 通过同一套 cache 接口黑盒测试 | |
| 5 | `StaticCache` decode 过程中不依赖 `torch.cat` 扩容（`data_ptr` 不变） | |
| 6 | `reset()` 后能再次 prefill / decode，旧状态不泄漏 | |
| 7 | 正确处理 past_len / position，cached decode 的位置不会从 0 重启（RoPE 模型上验证） | |
| 8 | 至少一个测试能故意抓到 off-by-one / cache-length 类错误（现场演示：改坏一处，测试变红） | |
| 9 | 能用实际 tensor 数据解释 KV 内存为什么随序列长度线性增长 | |
| 10 | NoCache / Dynamic / Static 有最小 latency 对比，计时方法无明显错误（warmup、synchronize） | |

Pass：★ 4/4 且 ≥ 8/10。"生成文本看起来差不多"不算通过。

<a id="gate-4"></a>
## Gate 4（Day 75–76）· 工程化 + 真实模型 + benchmark

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | `pytest` 全绿，且含 unit + invariant/equivalence + regression 三类测试 | |
| 2 ★ | 包可通过 `pyproject.toml` 安装 / 导入，不靠改 `sys.path` | |
| 3 ★ | 用真实小型 HF decoder 模型（SmolLM2-135M）成功检查真实 `past_key_values` 的 shape 与 seq_len | |
| 4 ★ | 理论 KV byte 估算器与实际 `numel()*element_size()` 在预期范围内一致 | |
| 5 | benchmark 有 warmup，并对 MPS 正确 `synchronize` | |
| 6 | benchmark 至少报告 prefill latency、decode latency/token、tokens/s、KV bytes，并记录 device/dtype/config/context | |
| 7 | Dynamic / Static / Sliding 使用统一接口；新增 cache 不需要大改 generator | |
| 8 | 至少完成一次"先写失败测试 → 定位 bug → 修复 → 回归测试保留"的完整流程（有 commit 证据） | |
| 9 | README 说明运行方式、设计、测试方法，而不只是安装命令 | |
| 10 | Week 11 CLI Spec Challenge 的冻结黑盒测试全部通过 | |

Pass：★ 4/4 且 ≥ 8/10。

<a id="gate-5"></a>
## Gate 5（Day 98 审计 + Day 99–100 闭卷考）· 毕业

| # | 项 | Y/N |
|---|---|---|
| 1 ★ | 在没有参考实现的情况下，根据自己的 SPEC.md 完成目录、模块边界与接口设计 | |
| 2 ★ | 完整测试套件全绿；NoCache / Dynamic / Static 的 correctness 均有数值或 token 等价证明 | |
| 3 ★ | benchmark 可通过一条 CLI 命令复现，输出结构化结果 | |
| 4 ★ | README 让另一个人仅按文档即可安装、跑测试、跑 benchmark（Day 95 fresh-clone 验证过） | |
| 5 ★ | Day 99–100 在新空目录仅查文档重新实现指定 Transformer + KV 核心链路，等价测试 PASS | |
| 6 | Capstone 至少有 Dynamic、Static、Sliding 三种 cache policy | |
| 7 | `git log` 能看到 spec → tests → implementation → fix/refactor 的过程，而非一次性倒入代码 | |
| 8 | `bugs.md` ≥ 5 个真实 bug，含根因与回归策略 | |
| 9 | `ai_log.md` 中不存在"AI 生成实现 / 设计接口 / review 架构"的违规记录 | |
| 10 | 不运行代码能画出 prefill / decode 数据流并解释各 tensor shape 与内存 scaling | |

Pass：★ 5/5 且 ≥ 9/10。
