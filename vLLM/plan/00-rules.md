# 00 · 硬规则（Day 1 先读，每周一复读）

Plan 1 的规则（复制粘贴 = 0、Read → Close → Write、20 分钟规则、bugs.md、shape 注释、CPU 先于加速器）全部继续有效。下面只写**新增或改变**的部分。

## R1 三种读法，按阶段切换

| 阶段 | 读法 | 为什么 |
|---|---|---|
| Phase I | **Read → Close → Write**：论文和参考实现（nano-vllm）都在你自己写完之后才读 | 和 Plan 1 一样：先自己设计，再看别人怎么做 |
| Phase II–III | **先预测，再验证**（Probe）：读一段 vLLM 源码 → 合上 → 写下你预测它在某个输入下的行为 → 写测试验证 → 记录对错 | vLLM 有几十万行，线性阅读一定读不完也记不住。预测会逼你真正理解，验证会立刻纠正你 |
| Phase IV | **带着问题读**：只读和你要修的问题有关的代码，但要读到能解释根因为止 | 贡献阶段的阅读是为修改服务的 |

Probe 的预测**必须在运行任何东西之前写下来**（记在 `logs/probe_log.md`），哪怕只有一句话。事后补写的预测不算。

## R2 读源码的工具顺序

1. 先找测试：`tests/v1/core/` 里几乎每个核心行为都有测试，测试就是最好的"用法说明"。
2. 再打断点：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 让 EngineCore 在同一个进程里运行，断点才能停下来（见 [source-map.md](source-map.md#debug)）。
3. 最后才线性阅读。`scheduler.py` 有 3888 行，`schedule()` 一个函数就有约 1500 行，从头读到尾是最低效的方式。

## R3 AI 使用政策

### Phase I（写 minivllm）

和 Plan 1 Day 1–77 完全相同：AI 可以解释概念、traceback、API 文档；可以在你写完并 commit 之后 review；可以**只根据 spec** 生成黑盒测试。AI 不写实现、不设计接口、不重构。

### Phase II–III（读 vLLM）

| 可以 | 不可以 |
|---|---|
| 解释一个 Python / PyTorch 语法或 API | 在你读之前让 AI 总结整个文件（这等于让别人替你读） |
| 在你**已经写下自己的理解之后**，让 AI 指出哪里不对 | 让 AI 替你写 Probe 的预测 |
| 解释一段英文文档或一条 review 意见的意思 | 让 AI 写 Probe 测试 |

### Phase IV（给 vLLM 提 PR）

这一阶段除了本计划的规则，还**必须遵守 vLLM 官方的 AI 辅助贡献规定**（`docs/contributing/README.md` → "AI Assisted Contributions"，2026-11-17 核实）：

- **Be involved**：不提交"纯 agent"PR。提交者要审阅每一行改动、端到端验证行为、亲自跑相关测试。
- **Ensure significance**：不提交一次性的"凑数"PR，比如只改一个错别字、孤立的风格清理。
- AI 对代码提供了实质帮助时：必须在 PR 描述里说明，并在 commit 里加 `Co-authored-by:` 之类的署名行；PR 模板里有一个"AI assistance was used"的勾选框。

本计划在此之上更严格：**PR 的代码和测试由你自己写**，所以正常情况下不会触发上面的披露要求。允许的 AI 用途只有两个：

1. 解释（同 Phase II–III）；
2. **润色你自己写好的英文**（PR 描述、issue 评论、review 回复）。内容、结论和数据必须是你自己的、亲自验证过的。

如果你违反了本计划、让 AI 写了代码，那就按 vLLM 规定如实披露，同时记进 `logs/ai_log.md`。永远不要隐瞒。

**不要用 AI 写 review 评论。** 你对别人 PR 的每一条评论都要基于你自己跑过、读过的东西。

## R4 GitHub 工作流（Plan 1 没教过的部分）

Plan 1 只用到本地 git。这里必须会：

```
upstream (vllm-project/vllm)  ──fork──>  origin (你的账号/vllm)  ──clone──>  本地 ~/code/vllm
```

- 永远不要往 upstream 推任何东西（你也没有权限）。所有工作在 `origin` 的**分支**上完成，每个 PR 一个分支，从最新的 `upstream/main` 拉出来。
- 同步上游：`git fetch upstream` 然后 `git rebase upstream/main`。改写过历史后推送用 `git push --force-with-lease`，**不要用 `--force`**。
- **DCO 签名**：vLLM 要求每个 commit 都有 `Signed-off-by:` 行，用 `git commit -s` 自动加上。
  - 隐私提醒：`Signed-off-by` 会把你的名字和邮箱**永久公开**在 vLLM 的提交历史里。如果不想公开真实邮箱，先在 GitHub 设置里启用 noreply 邮箱，再用 `git config user.email` 设置成它（只对 vLLM 仓库设置也可以）。
- 提交 PR 时勾选 **Allow edits by maintainers**。vLLM 的 CI 机器人有时需要把 `main` 合并进你的分支。
- 练习在你自己的 fork 里进行（Day 32）：在 fork 内部开 PR、制造冲突、rebase、force-with-lease。不要拿 upstream 练手。

## R5 PR 礼仪（来自 vLLM 贡献指南 + 本计划的自我约束）

- **先认领再动手**：在 issue 下留言说明你打算怎么做，确认没有别人在做（看评论和关联的 PR）。
- **不要凑数**：单个错别字、孤立的格式调整不要单独提 PR（vLLM 明确不鼓励）。好的第一个 PR 是：带回归测试的小 bugfix、给一个确实没被测到的行为补测试、给用户可见但没写进文档的行为补文档。
- **同时最多 2 个未合并的 PR**（本计划的约束；vLLM 对没有写权限的贡献者的上限是 6 个）。
- **回复 review 不超过 2 天**；每条意见都要用"修改"或"解释"回应。
- **review 没动静要耐心**：vLLM 的约定是指派 reviewer 后每 2–3 天有状态更新，**超过 7 天**没人看才礼貌地 ping。
- 改动超过 500 行（不含 kernel / 数据 / 配置 / 测试）需要先开 RFC issue。这 100 天里你不应该碰到这种规模。

## R6 日志

在 `vLLM/logs/` 下建自己的日志（`cp -r ../templates logs` 复制 Plan 1 的模板，再 `cp templates/* logs/` 复制本计划新增的模板）：

| 文件 | 用途 |
|---|---|
| `daily_log.md`、`bugs.md`、`ai_log.md` | 和 Plan 1 相同 |
| `probe_log.md` | 每个 Probe：读了什么 → 预测 → 测试 → 结果 → 更新后的理解 |
| `pr_reading_log.md` | 每个读过的 PR：问题、改动、测试、review 意见、可学的地方 |
| `issue_candidates.md` | Phase III 起维护的候选 issue 表 |

## R7 不要声称你没验证过的东西

你只有 CPU 和 Metal。在 PR 描述、issue 评论里，**只写你实际跑过的结果**，并注明环境（macOS、CPU 构建、vLLM commit）。不确定 CUDA 上是否成立，就直接写 "Validated on macOS CPU build; not tested on CUDA."。维护者很看重这种诚实，CI 会补上 GPU 那部分的测试。

## R8 进度指标

每日日志继续记 Plan 1 的那几项（空白文件延迟、违规次数、bug 数），新增：

- 今天写了几个 Probe，预测对了几个
- 今天读过的 vLLM 文件和函数（写 `file:function`，不写"读了调度器"）
- PR 状态：开了几个、等待中几个、回复了几条 review

## R9 失败协议

和 Plan 1 完全相同（[../../plan/gates.md](../../plan/gates.md)）：Gate 用 yes/no 清单判 Pass / Recoverable Fail / Structural Fail；漏 ≤ 2 天用 buffer 补，3–4 天砍 optional，> 4 天整体后移；**绝不跳过能力依赖**。

Phase IV 有一个特殊情况：**等 review 不算失败**。维护者什么时候看你的 PR 不由你控制，所以毕业条件里没有"必须被合并"（见 [gates.md](gates.md#gate-d)）。PR 在等待时，推进另一条线：下一个 PR、复现别人的 issue、review 别人的 PR。
