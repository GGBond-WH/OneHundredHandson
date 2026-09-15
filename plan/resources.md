# Resources · 全部资料（2026-09-14 已核实可访问）

规则回顾：资料只占 20% 时间；Read → Close → Write；视频不跟着打。

## Python
- Python 教程 [6. Modules](https://docs.python.org/3/tutorial/modules.html) · [9. Classes](https://docs.python.org/3/tutorial/classes.html)（含 9.8 Iterators / 9.9 Generators）
- [dataclasses](https://docs.python.org/3/library/dataclasses.html) · [typing](https://docs.python.org/3/library/typing.html) · [pathlib](https://docs.python.org/3/library/pathlib.html) · [argparse HOWTO](https://docs.python.org/3/howto/argparse.html)
- [uv](https://docs.astral.sh/uv/)（环境；配置超 30 min 就换 venv）· [ruff](https://docs.astral.sh/ruff/)
- [PyPA packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)（W11）

## pytest
- [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html) · [parametrize](https://docs.pytest.org/en/stable/how-to/parametrize.html) · [markers](https://docs.pytest.org/en/stable/how-to/mark.html)

## PyTorch
- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：只读 Tensors / Datasets & DataLoaders / Build Model / Autograd / Optimization / Save & Load（FashionMNIST 示例跳过，只学 API）
- [`F.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)（W3 的 reference implementation）
- [torch.mps](https://docs.pytorch.org/docs/stable/mps.html)：`synchronize / current_allocated_memory / driver_allocated_memory`；MPS 不支持 float64
- [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html)
- [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles)（W2，21 题，不看 walkthrough）

## Karpathy（主线视频与仓库）
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)：
  - makemore Part 1（bigram 段落，W2）· [makemore 仓库](https://github.com/karpathy/makemore)
  - 《Let's build GPT: from scratch, in code, spelled out》（W3–W4，分段看、关掉、自己写）
  - 《Let's build the GPT Tokenizer》（W1 只看前 30 min 概念）· [minbpe](https://github.com/karpathy/minbpe)（`basic.py` 写完后读；`tests/` W9 读）
- [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py)（**W4 周五才允许读**）· [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)（W3 attention 对照）

## 推理 / KV cache 源码（限定范围阅读）
- [gpt-fast](https://github.com/meta-pytorch/gpt-fast)（原 pytorch-labs，已迁移）：`generate.py` 的 `prefill / decode_one_token / decode_n_tokens / generate`（W6）；`model.py` 的 `class KVCache` / `setup_caches` / attention 里的 cache update（W11）。**不读** compile / tensor parallel / speculative / int8 fast path。
- HF transformers [`cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)：只找 `CacheLayerMixin / DynamicLayer / StaticLayer / DynamicCache / StaticCache`（W7）；[KV cache 文档](https://huggingface.co/docs/transformers/kv_cache)
- HF [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)：`rotate_half / apply_rotary_pos_emb / repeat_kv / LlamaAttention.forward`（W9 周一、W10 周五）

## 模型与数据
- [tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)（W2–W8 唯一数据集）
- [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)：30 层、hidden 576、9 attention heads、3 KV heads、head_dim 64、bf16 权重约 270 MB（W10）

## 研究衔接（仅 Capstone stretch，且学习者已熟悉）
- StreamingLLM（attention sink）、H2O（heavy-hitter oracle）——作为 `EvictionCache` 的灵感来源，不要求复现。

## 明确不学（这 100 天）
CUDA / Triton / FlashAttention kernel · TensorRT-LLM · NCCL / 多卡并行 · FSDP / DeepSpeed · vLLM 部署 · bitsandbytes · 大模型训练。它们回答"怎样把大型 LLM 系统跑得更快"，你现在要先回答"我能不能自己把 LLM 系统写出来"。
