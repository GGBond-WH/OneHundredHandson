# bugs.md

复制到 `logs/bugs.md`。每个**真正解决过**的 bug 一条。Gate 5 要求 ≥ 5 条。

## Bug #1 · YYYY-MM-DD · <一句话标题>

- **症状**：
- **我的假设**：
- **实验**（最小复现 / 打印了什么 / 断点看了什么）：
- **根因**：
- **修复**（commit sha）：
- **回归测试**（test 名称）：
- **以后怎么防**：

<!-- 示例
## Bug #0 · 示例 · cached generation 与 no-cache 结果不同
- 症状：greedy 序列前 5 个 token 一致，之后分叉
- 我的假设：causal mask 错
- 实验：T=2 最小复现；打印两条路径的 attention 权重；发现 decode 步的 q 用了 position 0
- 根因：decode 时 RoPE position 从 0 重启
- 修复：position_ids = arange(past_len, past_len + T)
- 回归测试：test_cached_logits_match_full_logits
- 以后怎么防：任何带 position 的代码，第一条测试就是 past_len > 0 的情况
-->
