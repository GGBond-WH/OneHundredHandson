# PR 草稿

先在这里写好，再贴到 GitHub。结构与 vLLM 的 `.github/PULL_REQUEST_TEMPLATE.md` 一致（2026-11-17 核实）。

## 标题

`[类型标签][范围标签] 一句话描述`，例如 `[Bugfix][Core] Fix ...`、`[Test][Core] Add tests for ...`

## Overview

<!-- 1–2 行概括改动。链接相关 issue 和 PR：Fixes #12345 -->

## Claims

<!-- 几条具体的成果：修了什么 bug、补了哪些测试、支持了什么 -->
-

## Validation

<!-- 证明上面每一条成立的证据和复现步骤。只写实际跑过的东西。 -->
Environment: macOS <版本>, Apple M4 Pro, vLLM CPU build from source at commit `<sha>`, Python 3.12.

```bash
pytest -q tests/v1/core/<file>.py::<test>    # before this PR: FAILED; after: PASSED
pytest -q tests/v1/core/<file>.py            # N passed (same as baseline)
pre-commit run --files <changed files>       # all passed
```

Not tested on CUDA.   <!-- 没测就照实写 -->

## Details

<!-- 动机、根因、设计取舍、局限。简洁；长的材料折叠起来 -->

---

## 提交前自查（不贴到 GitHub）

- [ ] 每个 commit 都有 `Signed-off-by`
- [ ] pre-commit 全绿
- [ ] git 历史里能看到"先失败测试、后修复"
- [ ] 用户可见的改动更新了文档
- [ ] AI 勾选框如实填写
- [ ] 已 rebase 到最新 `upstream/main`
- [ ] 开 PR 时勾选 Allow edits by maintainers
