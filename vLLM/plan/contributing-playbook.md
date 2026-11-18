# 贡献手册 · 从 issue 到合并

规范来源：vLLM `docs/contributing/README.md`（2026-11-17 核实）。下面标注"vLLM 规定"的是项目的硬性要求，标注"本计划"的是这份方案额外加的约束。

<a id="find"></a>
## 1. 找 issue

**官方入口**（贡献指南的 Job Board）：
- [Good first issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)（2026-11-17 时有 14 个 open）
- [Selected onboarding tasks](https://github.com/orgs/vllm-project/projects/6)

**按你的方向搜**（在 GitHub issue 搜索框里输入）：
```
is:issue is:open "prefix caching"
is:issue is:open "kv cache" in:title
is:issue is:open preemption
is:issue is:open scheduler label:bug
is:issue is:open "cpu" in:title
```

**从你自己的 Probe 里找**：Phase II–III 里你发现"这个行为没有被任何测试覆盖"或者"这个行为和文档说的不一样"，本身就是选题来源。先开 issue 说清楚，再提 PR。

**从 vllm-metal 找**：它的 open issue 也是候选（2026-11-17 时没有标 good first issue 的，需要自己判断难度）。

<a id="choose"></a>
## 2. 选题标准（本计划）

| 条件 | 为什么 |
|---|---|
| 能在你的 Mac（CPU 构建或 vllm-metal）上**复现和验证** | 你只能为自己验证过的东西负责（[00-rules.md R7](00-rules.md)） |
| 第一个 PR 改动 ≤ 200 行（不含测试） | 小 PR 更快得到 review；vLLM 对 > 500 行的架构改动要求先写 RFC |
| 对错有**明确标准**：一个会失败的测试，或者一个可测量的数字 | 没有明确标准的 PR 很难被说服 |
| **不是凑数**：不是单个错别字、不是孤立的格式调整 | vLLM 规定明确不鼓励这类 PR |
| 没人在做：issue 下没有人认领，也没有关联的 open PR | 避免重复劳动 |

好的第一个 PR 通常是这三类之一：**带回归测试的小 bugfix**、**给一个确实没被测到的重要行为补测试**、**给一个用户可见但文档没写清楚的行为补文档**。

## 3. 认领

在 issue 下用英文留言，说明你打算怎么做。模板：

> Hi, I'd like to work on this. I can reproduce it on a macOS CPU build (commit `abc1234`): <一两句复现结果>. My plan is to <一句话方案> and add a regression test in `tests/v1/core/...`. Does that sound reasonable?

如果 issue 比较老、之前有人认领但没有动静，礼貌地问一句：

> Is anyone still working on this? If not, I'd be happy to pick it up.

## 4. 开发流程

```bash
cd ~/code/vllm
git fetch upstream
git switch -c fix/<简短描述> upstream/main     # 每个 PR 一个分支，从最新的 upstream/main 拉出来
# 1) 先写一个会失败的测试，确认它失败          git commit -s
# 2) 修复，相关测试文件全部跑一遍              git commit -s
# 3) pre-commit run                            （全绿）
git fetch upstream && git rebase upstream/main   # 提交前和 main 对齐
git push -u origin fix/<简短描述>
```

- **vLLM 规定**：每个 commit 都要有 `Signed-off-by:`（DCO），用 `git commit -s`。
- 提交 PR 后再改：继续 `git commit -s`，需要同步 main 时 `git rebase upstream/main` 然后 `git push --force-with-lease`。

<a id="checklist"></a>
## 5. 提交前清单

**标题（vLLM 规定）**：以一个或多个方括号标签开头，空格，再写简洁描述。
- 类型标签：`[Bugfix]` `[Feature]` `[Perf]` `[Refactor]` `[Test]` `[Doc]` `[CI/Build]` `[Misc]`
- 范围标签：`[Core]` `[Frontend]` `[Kernel]` `[KV Connector]` `[KV Offload]` `[CPU]` 等
- 可以叠加，标签之间不加空格：`[Bugfix][Core] Fix ...`、`[Test][Core] Add tests for ...`

**内容**：
- [ ] 每个 commit 都有 `Signed-off-by:`（`git log` 检查）
- [ ] `pre-commit run` 全绿
- [ ] 有测试；先失败后通过的证据在 git 历史里看得到（本计划）
- [ ] 相关测试文件在本地全部跑过，结果和 `notes/test_baseline.md` 对照，没有新增失败
- [ ] 用户可见的改动（CLI 参数、配置项、环境变量、行为变化）同一个 PR 里更新了文档（vLLM 规定）
- [ ] 代码风格遵循 Google Python style guide（vLLM 规定）
- [ ] 描述按模板写了 Overview / Claims / Validation / Details（见 `templates/pr_draft.md`）
- [ ] Validation 里只写实际跑过的东西，注明环境：macOS、CPU 构建、vLLM commit；没测 CUDA 就写没测（本计划 R7）
- [ ] PR 模板里的 AI 勾选框如实填写（vLLM 规定）
- [ ] 分支已 rebase 到最新 `upstream/main`
- [ ] 开 PR 时勾选 **Allow edits by maintainers**

## 6. 提交之后：CI

- vLLM 的 GPU CI 资源有限，**新 commit 不会自动触发完整 CI**。在 PR 被批准或打上 `ready` 标签之前，`/ci run` 只有有写权限的 reviewer 和受信任的贡献者能用。你不需要也无法自己触发，等 reviewer 处理。
- PR 被批准或有了 `ready` 标签后，作者可以用 `/ci run`、`/ci retry`、`/ci cancel`。
- 跑 CI 前，你的分支必须和目标分支**零落后**。落后时机器人会尝试把 main 合并进你的分支（所以要勾选 Allow edits by maintainers）。冲突时机器人不会处理，你要自己 rebase。
- pre-commit 和 DCO 检查对所有 PR 都会运行，这两项必须你自己保证通过。

## 7. review

**节奏（vLLM 的约定）**：
- PR 会被指派给一位 reviewer，之后每 2–3 天有状态更新；
- 需要你修改时，reviewer 会打上 `action-required` 标签。改完后 ping reviewer 重新看；
- 超过 7 天没有 review，可以礼貌地 ping reviewer 或 vLLM 团队；也可以去 [Slack](https://slack.vllm.ai) 的 `#pr-reviews` 频道问。

**回应原则**：每条意见都用"修改"或"解释"回应，不懂就问，有分歧可以讨论，必要时由 committer 决定。本计划要求 2 天内回应。

**英文模板**：

| 场景 | 模板 |
|---|---|
| 按意见改了 | Thanks! Addressed in `<commit>`. |
| 意见指出了一个你没测到的情况 | Good catch — I added a test for this case in `<commit>`. |
| 你选择不改，说明原因 | I kept `<X>` because `<原因>`. Happy to change it if you'd prefer `<Y>`. |
| 没看懂意见 | Could you clarify what you mean by `<…>`? Do you mean `<你的理解>`? |
| 改完请再看 | I've addressed all the comments — could you take another look when you have time? |
| 7 天没动静 | Gentle ping @<reviewer> — this is ready for review whenever you have a moment. Thanks! |

## 8. 数量与规模的边界

- **vLLM 规定**：没有写权限的贡献者最多同时有 6 个 open PR。
- **本计划**：同时最多 2 个。
- **vLLM 规定**：主要的架构改动超过 500 行（不含 kernel / 数据 / 配置 / 测试）需要先开 RFC issue 讨论设计，否则会被打上 `rfc-required` 标签。这 100 天里你不应该碰到这种规模。

## 9. review 别人的 PR

这是真实的贡献，也是最快的学习方式。

1. 挑你熟悉的领域（KV cache / 调度），最好是作者说"只在 CUDA 上测过"、或者请人帮忙测试的 PR；
2. `gh pr checkout <PR 号>`，在你的 Mac CPU 构建上跑它的测试和相关测试文件；
3. 读懂改动，评论写你**实际跑过的结果**和**具体的问题**：

> I ran `pytest tests/v1/core/test_prefix_caching.py` on a macOS CPU build with this PR (commit `abc1234`): all N tests pass. One question: in `allocate_slots`, what happens when `<边界情况>`? I couldn't find a test covering it.

不写"LGTM"之类没有信息量的评论，也不用 AI 写 review。
