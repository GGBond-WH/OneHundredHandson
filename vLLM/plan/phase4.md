# Phase IV · 贡献：issue → PR → review（Day 71–100）

**目标**：向 `vllm-project/vllm` 提交至少一个符合全部规范的 PR，并和维护者完成 review 往返；再完成第二份贡献；review 别人的 PR；通过闭卷毕业考。
**规范来源**：vLLM 的 `docs/contributing/README.md`（2026-11-17 核实），细节整理在 [contributing-playbook.md](contributing-playbook.md)。
**Gate D（毕业）**：Day 98–100。

## 这一阶段的三条现实

1. **review 的节奏不由你控制。** vLLM 的约定是指派 reviewer 后每 2–3 天有状态更新，超过 7 天没人看才 ping。所以 PR #1 必须在 Day 75 左右提交，留出至少三周的 review 时间。
2. **等待时不要闲着，也不要催。** PR 在等待时，推进另一条线：下一个 PR、复现别人的 issue、review 别人的 PR。
3. **毕业不要求"已合并"。** 合并取决于维护者的时间和项目优先级。毕业要求的是：PR 本身达到可合并的质量，而且你完成了 review 往返（见 [gates.md](gates.md#gate-d)）。

---

## Week 11 · PR #1

### Day 71（周一）· 选题 + 认领

【做】从 `logs/issue_candidates.md` 里按 [playbook 的选题标准](contributing-playbook.md#choose) 选定 PR #1：能在 Mac 上验证、改动 200 行以内、对错有明确标准、不是凑数、没人在做。
【做】在 issue 下留言认领，说明你的计划（英文模板见 playbook）。如果你的 PR 是"补测试"或"修一个你自己 Probe 发现的问题"，而且还没有 issue，**先开一个 issue** 描述问题，再提 PR。

### Day 72（周二）· 先写失败的测试

【做】
```bash
cd ~/code/vllm && git fetch upstream
git switch -c fix/<简短描述> upstream/main
```
先写一个**在当前代码上失败**的测试来复现问题，跑一遍确认它确实失败，`git commit -s`。这和 Plan 1 Day 59 的"失败测试先行"完全一样。如果是纯补测试的 PR，测试应该通过，那就换个方式证明它有价值：故意把被测代码改坏一处，测试能变红。

### Day 73（周三）· 修复

【做】改代码，直到新测试变绿；再把相关的整个测试文件跑一遍，和 `notes/test_baseline.md` 对照，确认没有新增失败。`pre-commit run` 全绿后 `git commit -s`。

### Day 74（周四）· 验证 + PR 描述

【做】过一遍 [playbook 的提交前清单](contributing-playbook.md#checklist)。
【写】按 vLLM 的 PR 模板写描述（`templates/pr_draft.md`）：**Overview**（1–2 行，链接 issue）、**Claims**（做到了什么）、**Validation**（证据：跑了哪些命令、结果是什么、在什么环境）、**Details**（根因、设计取舍、局限）。按 [00-rules.md R7](00-rules.md)，只写你实际跑过的东西。
【写】标题：方括号标签 + 一句话，比如 `[Bugfix][Core] ...` 或 `[Test][Core] ...`。

### Day 75（周五）· 提交 PR #1

【做】`git fetch upstream && git rebase upstream/main`（vLLM 跑 CI 前要求分支和 main 零落后），`git push -u origin <分支>`，然后在 GitHub 上开 PR，目标是 `vllm-project/vllm:main`，**勾选 Allow edits by maintainers**。PR 模板里的 AI 勾选框如实填。
【读】读 2 个同领域、已合并的 PR，对照你的描述写得够不够清楚。

### Day 76（周六）· review 别人的一个 PR

【做】在 KV cache / 调度领域找一个 open PR（最好有人请求测试）：拉到本地（`gh pr checkout <号>`），在 Mac CPU 构建上跑它的测试，读懂改动，留一条**有内容的**评论：你跑了什么、结果如何、你看到的一个问题或疑问。不写"LGTM"之类的空评论。
这是真实的贡献，也是最好的学习：你会看到 reviewer 眼中的 PR。

### Day 77（周日）· 🔁 复盘

闭卷用 5 句英文讲清 PR #1：问题是什么、根因在哪、怎么修的、怎么证明修好了、还有什么局限。这就是 review 对话里你要能回答的东西。写周复盘。

---

## Week 12 · review 往返 + PR #2 选题

### Day 78（周一）· 处理 review

【做】有 review 意见就逐条回应：每条要么修改（回复里写"在哪个 commit 改了"），要么解释为什么不改。不清楚就问。
没有 review 也正常：PR 没有指派 reviewer 的话，按 playbook 的方法看看这个领域谁常 review（`git log` 看最近的作者和 reviewer），或者去 vLLM Slack 的 `#pr-reviews` 频道问。**不要提前催。**

### Day 79（周二）· PR #2 选题

PR #2 要比 PR #1 更实质，方向（任选其一）：
- KV cache 管理器 / 前缀缓存 / 调度器里一个能复现的 bug；
- 一段确实没被测试覆盖的重要分支（用 `pytest --cov=vllm/v1/core` 找）；
- 一个用户可见、但文档里没写清楚的行为（vLLM 明确要求用户可见的改动必须写进文档）；
- vllm-metal 上的一个问题（它在你的 GPU 上运行，前提是你按它的 CONTRIBUTING 装好了完整 Xcode 和 SDK）。

【做】`git log --oneline -- <要改的文件>` 看这个文件最近的改动和作者，了解谁可能 review 你。

### Day 80（周三）· 先讨论设计（如果需要）

如果 PR #2 涉及设计选择（不止一种合理的修法），先在 issue 下写 3–5 句方案和备选方案，问维护者倾向哪种，等回复再写代码。不涉及设计选择就跳过这一步，直接进入明天的失败测试。

### Day 81（周四）· PR #2：失败测试

同 Day 72。

### Day 82（周五）· 📰 PR 阅读 ×2

读和 PR #2 改动同一个文件的两个历史 PR：`git log --oneline -- <文件>`。看看那里的 reviewer 关心什么。

### Day 83（周六）· PR #2：实现

### Day 84（周日）· 🔁 复盘 + 回复 review

---

## Week 13 · PR #2 完成并提交

### Day 85（周一）· 实现完成，相关测试全绿

### Day 86（周二）· 端到端验证

在 Mac CPU 构建上端到端跑一遍真实模型。如果是性能改动，按 Plan 1 的 benchmark 规范（warmup、多次取中位数、记录环境）给出改动前后的数据，并在描述里写明只在 CPU 上测过。

### Day 87（周三）· 文档

如果改动是用户可见的：同一个 PR 里更新文档。vLLM 规定 CLI 参数、配置项、环境变量的文档写在各自的 docstring、`help=` 或环境变量注释里，其他内容写在 `docs/` 下对应的页面。

### Day 88（周四）· 描述 + 自查 + rebase

同 Day 74–75 的流程。

### Day 89（周五）· 提交 PR #2

### Day 90（周六）· 两个 PR 的 review 往返

### Day 91（周日）· 🔁 复盘

---

## Week 14 · 收尾

### Day 92（周一）· review 往返 + 复现一个别人的 bug

在 KV cache / 调度领域找一个别人报告、还没有最小复现的 bug，在你的 Mac 上复现，在 issue 下给出最小复现脚本和环境信息（或者说明你在 CPU 上复现不了，并附上你试过的步骤）。

### Day 93（周二）· review 别人的第二个 PR

同 Day 76。

### Day 94（周三）· review 往返；需要时 rebase

### Day 95（周四）· 贡献记录

【写】`logs/contribution_log.md`：每个 PR、issue 评论、review 的链接和状态，每一项学到了什么；最后写你想长期参与的 1–2 个子系统，以及下一个想做的贡献。

### Day 96（周五）· review 往返 / 收尾

### Day 97（周六）· ⏸ Buffer

### Day 98（周日）· 🚪 Gate D ①：审计

逐项过 [gates.md → Gate D](gates.md#gate-d) 的清单。

### Day 99（周一）· 🚪 Gate D ②：毕业考 Part 1（3 小时）

闭卷。给定一个场景，在纸上推演 vLLM 的块分配过程，再写测试验证。场景示例：`block_size=4`、开启前缀缓存、总共 10 个块，请求 A（prompt 10 个 token）生成 3 个 token 后结束，请求 B 的 prompt 与 A 的前 8 个 token 相同。推演：A 结束后哪些块进入空闲队列、以什么顺序；B 命中几个块；B 需要几个新块，从哪里来。写出涉及的 vLLM 函数名，然后用 `tests/v1/core/` 风格的测试验证你的推演。

### Day 100（周二）· 🚪 Gate D ③：毕业考 Part 2（3 小时）

只查文档。给一段 bug 描述（可以从 vLLM 已关闭的 bug issue 里选一个你没读过的，只看描述，不看修复），在 vLLM 测试风格下从零写出复现它的测试。然后写 `logs/final_reflection.md`：Day 1 和 Day 100 相比，你面对一个陌生 issue 的第一步有什么不同。
