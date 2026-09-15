# design_comparison.md（Day 47：读完 HF cache_utils 之后）

## 我的接口
```python
# 贴你的 KVCache Protocol / ABC
```

## HF 的接口（只列签名）
```python
# DynamicCache.update(...) / get_seq_length() / reset() / StaticCache(...)
```

## 逐项对比

| 维度 | 我的设计 | HF 的设计 | 差异原因（算法需求 / 工程需求 / 历史包袱？） |
|---|---|---|---|
| update 的返回值 | | | |
| seq_len 从哪来 | | | |
| 每层存储的组织方式 | | | |
| 预分配 / 增长策略 | | | |
| reset 的语义 | | | |
| device / dtype 处理 | | | |

## 我要采纳的一条，以及我坚持不采纳的一条（各说明理由）
