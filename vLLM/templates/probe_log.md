# probe_log.md · 先预测，再验证

复制到 `logs/probe_log.md`。**"我的预测"必须在运行任何东西之前写下来**，事后补写的不算（规则 R1）。

## Probe #N · YYYY-MM-DD · <一句话问题>

- **读了什么**：`file:function`（写到函数级）
- **场景**：输入是什么（`block_size`、请求的 token、是否开前缀缓存……）
- **我的预测**（运行前）：
- **验证用的测试**：`tests/v1/core/test_probe_xxx.py::test_...`（在本地 probe 分支上）
- **实际结果**：
- **预测对了吗？** 对 / 错。错的话，错在哪一步理解？
- **更新后的理解**（一两句）：
- **副产品**：这个行为有没有被现有测试覆盖？没有的话，记进 `issue_candidates.md`
