# 99 · Claude × GPT 讨论记录（2026-09-14，五轮）

方案不是单方面写的：Claude 先独立分析，再让 GPT 在不知道 Claude 判断的情况下独立设计，然后逐点碰撞四轮，最后做逐条共识确认。下面是结果，关键原话保留，便于你自己判断谁更有理。

## 双方共识（经 GPT 逐条确认）

**主线与阶段**：Python 工程能力 → PyTorch 张量与模块 → 手写 Decoder-only Transformer → 手写自回归推理 → 手写 KV cache → 工程化测试 / benchmark → 独立 Capstone。五阶段 + 5 个 Gate + 5 个 buffer day 全部落在 100 天之内。GPT 的定位："你现在最大的瓶颈其实不是 KV cache 理论，而是知道算法是什么，但无法把'概念 → 数据结构 → 函数 → 类 → 项目'独立展开。"

**三条轨道**（这是讨论中最重要的结构性修改）：Build（学新东西并实现）/ Retrieval（每周日 60 min 闭卷重写一个 ≤100 LOC 模块）/ Spec（W5 起每周四只给规格 + 黑盒测试，脚手架逐周撤）。起因是 Claude 指出"闭卷重写练的是复现自己写过的东西，capstone 要的是从 spec 设计"，GPT 回应："学习者非常擅长'默写 GPT'，但拿到一个稍微变化的新需求仍然不会开工……你的第四条不是'小修正'，而是原方案里一个确实缺失的能力维度。"

**pytest 从 Week 1 引入**（修正了 GPT 初版的自相矛盾：Day 14 要 8 个 pytest，但 Day 57 才"正式补 pytest"）。Week 9 改为"测试加固周：从会写测试到会设计测试"。

**Gate 用 yes/no 清单，不打主观分**。GPT："一个人在自学时让自己判断'我的结构设计值 21/25 分'几乎没有意义。" 判定：Pass = ★ 全过且 ≥80%；Recoverable Fail 只修失败维度；Structural Fail 最多追加 2 天并砍 optional。GPT 反驳了 Claude 提出的"卡住次数 ≤2"指标——"它会奖励'不碰困难问题'的行为"——Claude 采纳。

**AI 政策分类型而非一刀切**。Claude 提出 capstone 完全禁 AI 会让人在 MPS 算子 / 打包问题上卡两天学不到东西；GPT 接受并定下边界：算法 / 接口 / 架构 / correctness bug ❌；API 语义 / 环境 / 打包 / traceback 含义 ✅ 且需先独立排查 60 分钟；每次记 `ai_log.md`。原则："AI explains the failure; learner decides the fix." 另外明确允许"从冻结的 spec 生成黑盒测试"（AI 只看 spec 不看实现），并规定 AI review 只能在自测 + commit 之后、不给 patch。

**Day 14 考试从 90 分钟改为 3 小时、5 个必测**。GPT 自己估算 Bigram 项目 205–315 LOC，承认 90 分钟"主要测的是打字速度 + 短期记忆"。Gate 2 改为给基础设施、只重考模型核心。

**Week 8 负荷**：RMSNorm / SwiGLU 周一 1 小时内；RoPE 与 cache 对接占周一下半到周三；周三 readiness checkpoint，不过则 GQA Spec 顺延到 Day 60，Gate 3 不后移、GQA 不计 critical。

**Day 99–100 毕业考**：Day 99 模型构造；Day 100 到"cache/no-cache 等价测试 PASS"为止，benchmark 降为可选加分（"再要求 benchmark 并不会显著提高这个能力测量的效度"）。

**补上的缺口**（GPT 初版没有、Claude 提出、GPT 全部接受）：git 从 Day 1（`init/status/add/commit/diff/log/tag/restore`）；Day 1 环境 + 调试器演练；Week 2 Tensor Puzzles + 两档张量闭卷清单；周五源码阅读逐周具体化（W6 读 gpt-fast `generate.py` 而非 HF `generation/utils.py`——"教育信噪比偏低"；W9 读 minbpe 的 tests）；Basic BPE 放 Week 1 Day 5–6（"始终在 LLM 主线上，不是拿学生管理系统练 Python"）；`EvictionCache` 作为研究 stretch（预算 K=S+R+H；两个坑：SDPA 拿不到 attention 概率、驱逐后必须保留原始绝对位置）；数据集固定 tiny shakespeare。

**W9 的埋 bug 实现必须外部提供**（GPT 指出学习者自己埋的 bug 没有训练价值）——已由 Claude 写好并封存答案，见 `week09_bughunt/`。

**硬件**：M4 Pro 48GB 对这套课程"非常够"；CPU correctness → MPS correctness → MPS performance；MPS 无 float64；benchmark 禁止 fallback；CUDA / Triton / 分布式明确不做。

## 仍有分歧

没有实质性分歧留下。两处属于"需要执行数据才能判断"的估计：
- **每日 2–3 小时是否够**：所有 LOC 与时长估计（Gate 1 的 3 小时、Week 8 的负荷）都是 GPT 与 Claude 的经验估计，未经真人验证。解决办法：前两周的 daily_log 里"空白文件延迟"和"任务是否当天完成"两项数据，Day 14 复盘时决定是否整体放慢。
- **Capstone 用 Llama 风格（RoPE/GQA/SwiGLU）而非 GPT-2 风格**：GPT 主张它让 KV cache 尺寸（由 n_kv_heads 决定）与 position bookkeeping 真正变得有意义；代价是 Week 8 变重。Claude 接受但标注：若 Week 8 的 readiness checkpoint 连续两天不过，可把 capstone 的 GQA 降为 optional。

## 待核实

- GPT 给的 SmolLM2-135M 参数（30 层 / 9 头 / 3 KV 头 / hidden 576）——**已核实**（HF config.json）。
- HF `cache_utils.py` 的类名 `CacheLayerMixin / DynamicLayer / StaticLayer / DynamicCache / StaticCache`——**已核实**；GPT 说的"1359 行"不对（当前 2154 行），无关紧要。
- gpt-fast 已从 `pytorch-labs` 迁移到 `meta-pytorch`，`generate.py` 有 `prefill / decode_one_token / decode_n_tokens / generate`，`model.py` 有 `class KVCache` 与 `setup_caches`——**已核实**。
- `torch.mps.synchronize / current_allocated_memory / driver_allocated_memory` 存在、MPS 不支持 float64——**已在本机 torch 2.9.1 核实**。
- GPT 提到 SmolLM2-135M 权重文件"约 272 MB"——未核实（bf16 × 135M ≈ 270 MB，量级合理）。
- GPT 的 LOC / 时长估计——未核实（见"仍有分歧"）。

## 下一步（给学习者）

1. Day 1 按 `phase1.md` 建实验室，记下空白文件延迟基线。
2. Day 14 复盘时用两周数据决定是否放慢（这是唯一需要真实数据才能定的参数）。
3. Gate 3 后回看 `99-discussion.md` 的"Llama 风格 vs GPT-2 风格"分歧，决定 capstone 的 GQA 是 required 还是 optional。
