# 00 · 硬规则（Day 1 先读，每周一复读）

这些规则不是建议。整套方案的有效性建立在它们之上——你的问题是"看得懂写不出"，而下面每一条都是针对这个问题的。

## R1 复制粘贴 = 0

100 天内不复制任何教程 / 仓库 / StackOverflow / AI 输出的代码。哪怕是
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
也自己敲。代码结构记忆和肌肉记忆是动手能力的物理基础。

## R2 Read → Close → Write

永远不要"左边教程、右边编辑器、照着写"。正确流程：读 10–20 分钟 → **关掉资料** → 自己实现 → 卡住 → 再查（只查 API 文档）→ 最后再和参考实现比较。

Karpathy 的视频尤其如此：看一段概念 → 关视频 → 自己写 → 跑 → 错 → debug → 最后对比。

## R3 20 分钟规则（卡住时）

卡住的前 20 分钟禁止问 AI、禁止搜答案。必须按顺序做：

1. 读完整 traceback（最后一行 → 往上找自己代码的第一帧）
2. `breakpoint()` 或 IDE 断点 → 在断点处看 `x.shape / x.dtype / x.device / x.is_contiguous()`
3. 加 `assert`，把假设写成代码
4. 构造最小复现（B=1, T=2 这种）
5. 查官方文档

20 分钟后仍无解，可以问 AI："**解释**这个错误为什么发生"。不能问"给我正确代码"。

## R4 AI 使用政策

### Day 1–77

| AI 可以 | AI 不可以 |
|---|---|
| 解释概念、解释 traceback、解释 API 文档语义 | 实现任何模块 / 补完函数 / 生成整个类 |
| 在你**已经写完、跑过测试、commit 之后** review 你的代码：指出风险和问题 | 给出替换实现或 patch（哪怕是"修正版函数"） |
| 给测试思路 | 设计接口、建议目录结构、重构项目 |
| **从 spec 生成黑盒测试**（见下） | 看你的实现后帮你找 bug 并给修改代码 |

### Day 78–100（Capstone）

| 问题类型 | AI |
|---|---|
| 算法设计 / 接口设计 / 项目架构 | ❌ |
| correctness bug（如 cached logits ≠ full logits） | ❌ ——这正是训练目标，自己查 mask / position / RoPE / cache append / head shape / seq dim |
| "测试怎么才能让实现通过" | ❌ |
| API 文档语义（如 `index_copy_` 的 dim 与 index 的关系） | ✅ |
| MPS / Python 环境问题、packaging / tooling、traceback 本身的含义 | ✅ **但需先独立排查 ≥ 60 分钟**，且 60 分钟内必须做过 R3 的五步 |
| 从冻结的 spec 生成黑盒测试 | ✅ 实现前 |

每次 Capstone 期间使用 AI，都记入 `logs/ai_log.md`（模板见 templates/）。Gate 5 直接审计这个文件。

原则：**AI 解释失败，你决定修法。**

### 从 spec 生成黑盒测试的规则（Spec Track 与 Capstone 都适用）

1. 只给 AI：spec、函数/类签名、允许的依赖、edge-case 要求。**绝不给**自己的实现、以前的代码、自己的算法思路。
2. 拿到测试后，你先检查它是否符合 spec（不符合就在 `test_issue.md` 写明哪条测试违反了哪条 spec、为什么，然后重新生成那条——不许直接手改成对自己有利的版本）。
3. 确认后 `git commit -m "test: freeze <name> spec tests"`。此后测试**冻结**。
4. 开始写实现。实现失败不允许为了通过而改测试。可以新增测试，不能弱化原测试。

## R5 git 制度

Day 1 就 `git init`。只需要这些命令：`init / status / add / commit / diff / log / tag / restore`。不学分支协作。

- 每个有效学习日 ≥ 1 个有意义的 commit（不是为了 KPI 的空 commit）
- 每周四 Spec：冻结测试单独一次 commit
- 每个 Gate 通过后打 tag：`gate-1` … `gate-5`
- `.gitignore` 至少排除：`.venv/ __pycache__/ .pytest_cache/ .ruff_cache/ *.pyc checkpoints/ data/*.txt`

## R6 每个 bug 必须形成知识

`logs/bugs.md`，每条：症状 / 我的假设 / 实验 / 根因 / 修复 / 防止复发的回归测试。例：

```
Bug: cached generation 结果与 no-cache 不同
假设: causal mask 错
实验: T=2 最小复现，打印两条路径的 attention 权重
根因: decode 时 RoPE position 从 0 重新开始
修复: position_ids = arange(past_len, past_len + T)
回归测试: test_cached_logits_match_full_logits
```

## R7 shape 注释

Week 2 起，每个张量操作前先在注释里写 shape：
```python
# x: [B, T]
# emb: [B, T, C]
# logits: [B, T, V]
```
不允许"让它自己 broadcast，反正能跑"。

## R8 CPU correctness → MPS correctness → MPS performance

数学正确性先在 CPU 上（float32，必要时 float64——MPS 不支持 float64）验证，再上 MPS。不要在 MPS 上 debug 数学问题。benchmark 时**禁止** `PYTORCH_ENABLE_MPS_FALLBACK=1` 偷偷回退 CPU。任何 benchmark 结果必须记录 device / dtype / model config / context length。

## R9 训练目标不是模型效果

Tiny Shakespeare 上 loss 明显下降、模型能生成，就够了。不要花三天调 learning rate。整个 Week 2–8 不换数据集、不换 tokenizer、不换 objective——出问题时不需要怀疑数据。

## R10 进度指标（写在每日日志里）

记录这些，**不要**记"今天学了 2.5 小时 / 看完 40 页"：

- 空白文件延迟：从打开空文件到写下第一个 class/def 用了几分钟
- 今天独立写出的函数 / 类数量；第一次写就对的比例
- 违规查旧代码次数（目标 0）、AI 违规次数（目标 0）
- 今天解决的 bug 数（记入 bugs.md）
- 新增的**有效** invariant / regression 测试数；被测试抓到的真实 bug 数
- 每周日：Retrieval Drill 完成度

## R11 失败协议（详见 gates.md）

- Gate 判 Pass / Recoverable Fail / Structural Fail，按 yes/no 清单，不打主观分。
- 漏 ≤ 2 天：用 buffer 补。漏 3–4 天：砍本阶段 optional。漏 > 4 天：整个计划向后平移。
- **绝不跳过能力依赖**：attention 还写不出来，不进 KV cache，不管日历上是第几天。

## R12 Spec / Retrieval 期间的信息隔离

- Spec Challenge：没有参考实现、没有教程链接、没有 starter 代码；只有需求、签名、示例、验收测试。
- Retrieval Drill：不打开本周代码；允许查纯 API 文档，不允许查任何实现。
- Capstone（Day 78 起）：**不打开 Week 1–11 的自己的代码**，不打开 nanoGPT / gpt-fast / HF 源码。
