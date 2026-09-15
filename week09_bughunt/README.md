# Week 9 · Bug Hunt（Day 60 · Spec Challenge #5）

`faulty_kv.py` 是一份**外部提供**的、埋了 **4 个 bug** 的实现：一个带 RoPE 的 2 层 tiny LM + `DynamicCache` + `SlidingWindowCache` + greedy `generate`。你不知道 bug 在哪。它在"正常用法"下不会崩溃——所有 bug 都是**静默的**。

## 契约（模块声称自己做到的事——这就是你的 spec）

见 `faulty_kv.py` 顶部 docstring 的 5 条。概括：
1. `forward(idx, cache=None, window=None)`；有 cache 时 `idx` 只含新 token；位置是绝对的；`window=W` 表示查询位置 t 恰好看 `max(0, t-W+1)..t` 这 `min(t+1, W)` 个 key（含自己）。
2. prefill + 逐 token decode == 一次全量 forward（logits 容差内相等，greedy 完全一致）。
3. 分块 prefill == 一次 prefill。
4. `reset()` 后与新建的 cache 行为一致。
5. `SlidingWindowCache(W)` 复现第 1 条的窗口语义，且每层存储 ≤ W 个位置；`seq_len` 统计 reset 以来处理过的**全部** token 数。

## 规则

1. **先不改代码**。在 `tests/` 下写测试，把契约的每一条变成可执行的断言。目标：至少 4 个测试在 `faulty_kv.py` 上**失败**。
2. 然后定位、修复（在 `faulty_kv.py` 的副本 `fixed_kv.py` 里改，保留原件），每修一个 bug 记一条 `bugs.md`（症状 / 假设 / 实验 / 根因 / 修复 / 回归测试）。
3. 全部测试绿之后，才解封答案：
   ```bash
   base64 -d < answers.tar.gz.b64 | tar xz
   ```
   里面有 `ANSWERS.md`、正确实现 `reference_kv.py`、一套参考测试 `test_reference.py`。对照：你漏了哪个？你的测试为什么没抓到？

## 两个测试设计提示（不涉及 bug 在哪）

- 多层模型里，窗口 W 的**影响范围**不等于 W——先想清楚感受野，再写"改动远处 token 不应影响输出"这类测试，否则会在正确实现上误报。
- 模块自己的两条路径（有 cache / 无 cache）互相比较，不一定是可靠的 oracle。想想 reference 应该从哪来。

## 运行

```bash
cd week09_bughunt && python -m pytest -q tests/
```
（`faulty_kv.py` 只依赖 torch；CPU float32 即可。）

预计用时：写测试 60–90 min，定位与修复 60–90 min。超过 20 分钟卡住时按 rules R3 走；不可以让 AI 看 `faulty_kv.py` 找 bug。
