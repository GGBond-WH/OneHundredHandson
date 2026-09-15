# Phase I · Python + PyTorch 最小工程能力（Day 1–14）

**目标**：从"会看代码"变成"能自己组织 100–300 行的程序并用测试证明它对"。
**不学**：高级装饰器、元类、asyncio、Web、数据科学库。装饰器只需看懂 `@dataclass` / `@torch.no_grad()` / `@pytest.fixture` 的用法。
**产物**：`week01_text_pipeline/`（tokenizer + dataset + Basic BPE）、`week02_bigram/mini_lm/`（Bigram 语言模型完整 pipeline）。
**Gate 1**：Day 13。清单见 [gates.md](gates.md#gate-1)。

每日格式：【读】20–30 min ·【写】75–90 min ·【测】30–40 min ·【记】10–15 min（日志 + commit）。

---

## Week 1 · Python for LLM（不是 Python 入门）

### Day 1（周一）· Developer Workbench

今天不写 LLM 代码。今天建实验室。

【做】(30–40 min) 环境
- 在本目录（`One Hundred/`）执行 `git init`，写 `.gitignore`（见 rules R5），首个 commit：`chore: init 100-day repo`。
- 建环境（二选一，配置超过 30 min 就换 venv）：
  - `uv venv && source .venv/bin/activate && uv pip install torch pytest ruff`
  - 或 `python3 -m venv .venv && source .venv/bin/activate && pip install torch pytest ruff`
- 写 `day01_workbench/check_env.py`：打印 `torch.__version__`、`torch.backends.mps.is_available()`，在 CPU 和 MPS 上各做一次 `torch.randn(3,3) @ torch.randn(3,3)`。

【做】(30 min) VS Code
- 选中 `.venv` 解释器；打开 Testing 面板让它发现 pytest；写 `day01_workbench/test_smoke.py`（一个 `def test_torch_add(): assert (torch.tensor([1])+1).item() == 2`），从面板点 Run Test 和 Debug Test 各一次。

【做】(45–60 min) **Debugger Drill**（今天最重要的部分）
- 写 `day01_workbench/debug_drill.py`：
  ```python
  import torch
  x = torch.randn(2, 4, 8)
  y = x.transpose(1, 2)
  z = y.reshape(2, -1)
  w = y.view(2, -1)   # 这行会报错——这就是今天要观察的东西
  ```
- 在每行打断点（或用 `breakpoint()`），在调试控制台看：`x.shape, x.dtype, x.device, x.stride()`；`y.shape, y.is_contiguous(), y.stride()`；`z.shape`。
- 弄清：为什么 `view` 报错而 `reshape` 不报？（提示：contiguous）改成 `y.contiguous().view(2, -1)` 再看。
- 练熟：step over / step into / continue / watch 表达式 / call stack。
- 把 `x` 放到 MPS 再跑一遍，看 `device`。

【记】日志第一条：记下"Day 1 空白文件延迟"（今天从打开空文件到写下第一行代码用了几分钟）作为基线。

✅ 今日验收：`pytest` 从命令行和面板都能跑；能在断点处说出任意 tensor 的 shape/dtype/device/是否 contiguous；`git log` 有 ≥1 个 commit。

### Day 2（周二）· class / dataclass / typing → CharTokenizer

【读】Python 教程 [9. Classes](https://docs.python.org/3/tutorial/classes.html)（9.1–9.5 即可）；[dataclasses](https://docs.python.org/3/library/dataclasses.html) 开头示例；[typing](https://docs.python.org/3/library/typing.html) 只看基本注解 `list[int]`、`str | None`、`-> torch.Tensor`。读完关掉。

【写】`week01_text_pipeline/text_pipeline/tokenizer.py`
```python
class CharTokenizer:
    def __init__(self, text: str) -> None: ...   # 从文本建 vocab（排序去重的字符）
    @property
    def vocab_size(self) -> int: ...
    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```
决定：遇到 vocab 外的字符怎么办？（抛 `KeyError`？映射到 `<unk>`？）——自己选，写进 docstring，用测试固定。

【测】最小 pytest（今天引入）：`tests/test_tokenizer.py`
- `decode(encode(s)) == s`
- `vocab_size` 等于不同字符数
- 未知字符的行为符合你的 docstring
- 学会：`pytest -x`（第一个失败就停）、`pytest -k roundtrip`、`pytest -q`

✅ 验收：3 个测试绿；类里没有任何"先写了再说、不知道为什么能跑"的行。

### Day 3（周三）· iterator / generator / pathlib → TextDataset + batch_iterator

【读】Python 教程 [9.8 Iterators, 9.9 Generators](https://docs.python.org/3/tutorial/classes.html#iterators)；[pathlib](https://docs.python.org/3/library/pathlib.html) 基本用法（`Path(...).read_text()`）。

【写】`text_pipeline/dataset.py`
```python
class TextDataset:
    """把 token 序列切成 (x, y) 对：x = ids[i:i+block], y = ids[i+1:i+block+1]"""
    def __init__(self, ids: list[int], block_size: int) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[list[int], list[int]]: ...

def batch_iterator(dataset: TextDataset, batch_size: int, shuffle: bool = True, seed: int = 0):
    """generator：每次 yield 一个 batch 的 (xs, ys)，用 yield 实现"""
```
先用纯 Python list，不用 torch（明天再转 tensor）。

【测】`tests/test_dataset.py`
- **x/y 真的错位一位**（不是 shape 对了语义错）：构造 `ids=[0,1,2,3,4]`, block=2，断言 `ds[0] == ([0,1],[1,2])`
- `len(ds)` 正确（边界：ids 长度刚好 = block+1 时 len==1）
- batch_iterator 在 seed 固定时两次遍历顺序一致
- 用 `pytest.fixture` 提供一个小 dataset（今天学 fixture）

✅ 验收：能口头解释 generator 和 list 的区别（惰性、状态）；测试绿。

### Day 4（周四）· 模块、包、argparse → 可运行的包

【读】Python 教程 [6. Modules](https://docs.python.org/3/tutorial/modules.html)（6.1、6.4 packages）；[argparse tutorial](https://docs.python.org/3/howto/argparse.html) 前半。

【写】
- 把 `text_pipeline/` 变成包：`__init__.py`、`__main__.py`；`python -m text_pipeline --file data/input.txt --block-size 8 --batch-size 4` 打印第一个 batch 的 x/y 和解码后的文本。
- 下载 tiny shakespeare 到 `data/input.txt`（链接见 resources.md），`data/*.txt` 进 `.gitignore`。
- 写 `text_pipeline/io.py`：`load_text(path: Path) -> str`。

【测】CLI 的 smoke test（`subprocess.run([sys.executable, "-m", "text_pipeline", ...])` 返回码 0）。

✅ 验收：`python -m text_pipeline ...` 能跑；`from text_pipeline.tokenizer import CharTokenizer` 在任何目录都能 import（理解 `python -m` 与直接跑脚本的路径区别）。

### Day 5（周五）· Basic BPE ①

这是本周的纯 Python 项目，也是对"我到底会不会自己写普通 Python"的检验。**只做 Basic BPE**：不做 regex 预切分、不做 special tokens、不做 GPT-4 复现、不做优化。目标 80–150 LOC。

【读】只读概念：BPE 就是"反复把最频繁的相邻 byte pair 合并成新 token"。可以看 Karpathy《Let's build the GPT Tokenizer》前 30 分钟的概念部分，**然后关掉**。

【写】`text_pipeline/bpe.py`
```python
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]: ...   # 相邻 pair 计数
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]: ...
class BasicTokenizer:
    def __init__(self) -> None: self.merges: dict[tuple[int,int], int] = {}; self.vocab: dict[int, bytes] = {}
    def train(self, text: str, vocab_size: int) -> None: ...   # 从 256 个 byte 开始合并到 vocab_size
```

【测】`get_stats([1,2,2,1,2]) == {(1,2):2, (2,2):1, (2,1):1}`；`merge([1,2,2,1,2], (1,2), 9) == [9,2,9]`；train 后 `len(merges) == vocab_size - 256`。

### Day 6（周六）· Basic BPE ② + 读 minbpe

【写】`encode(text: str) -> list[int]`（按 merges 的顺序反复合并）、`decode(ids) -> str`（bytes 拼接后 `decode("utf-8", errors="replace")`）。

【测】任意 unicode 文本 roundtrip（含中文、emoji）；训练确定性（同文本两次 train 得到同 merges）；`vocab_size` 被尊重。

【读】**写完并全绿之后**，读 [minbpe/basic.py](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py)（约 70 行）。写 `notes/minbpe.md`：它和你的实现有哪 3 处不同？哪一处是你没想到的？

【测】周六例行：`ruff check . && ruff format .`；整理 `week01_text_pipeline/` 结构；commit。

### Day 7（周日）· Retrieval Drill #1（60 分钟）

新建 `scratch/w1_retrieval/`，**不打开本周任何代码**，只查 Python 文档：
- 从空白写 `CharTokenizer` + `batch_iterator`（合计 ≤100 LOC）+ 3 个测试。
- 60 分钟到即停，无论完没完成。
- 然后和 `week01_text_pipeline/` 对比：哪里卡住了？卡住的原因是"不记得 API"还是"不知道结构该怎么组织"？记入日志。

【记】周复盘（模板 templates/weekly_review.md）：本周空白文件延迟的变化；违规次数；下周要改的一件事。

---

## Week 2 · PyTorch 最小闭环 → Bigram 语言模型

数据固定：tiny shakespeare，char-level（用 Week 1 的 CharTokenizer）。Week 2–8 不换。

### Day 8（周一）· Tensor 基础 + Tensor Puzzles 1–7

【读】PyTorch [Learn the Basics → Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。教程里的 FashionMNIST 一律跳过，只学 API。

【写】`week02_bigram/tensor_drills.py`：每题先写 shape 注释再写代码（R7）
- 创建：`zeros/ones/randn/arange`，指定 `dtype` 与 `device`
- `view / reshape / transpose / permute / contiguous`：复现 Day 1 的 drill，并解释 `stride`
- broadcasting：`[B,T,1] + [1,1,C]` 结果 shape 先猜再跑；`[T,T]` mask 加到 `[B,H,T,T]`
- `unsqueeze / squeeze / expand / repeat`：expand 不拷贝、repeat 拷贝——用 `data_ptr()` 或 `storage` 验证

【做】(30 min) [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) 1–7。不看 walkthrough。做不出来的记下题号，明天继续。

【记】把 [tensor-checklist.md](tensor-checklist.md) A 级清单读一遍，标出今天已经会的。

### Day 9（周二）· autograd / nn.Module / Embedding → BigramLM

【读】Learn the Basics → [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)、[Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。只看 `nn.Module` 子类怎么写、`forward`、`parameters()`、`requires_grad`、`backward()`。

【写】`week02_bigram/mini_lm/model.py`
```python
class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        # self.table = nn.Embedding(vocab_size, vocab_size)  # 当前 token → 下一个 token 的 logits
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: [B, T] -> logits: [B, T, V]; loss 用 F.cross_entropy（注意它要 [N, V] 和 [N]）
```
把 Week 1 的 dataset 改成返回 `torch.Tensor`（`torch.tensor(ids, dtype=torch.long)`），batch 用 `torch.stack`。

【测】`logits.shape == (B, T, V)`；初始化时 `loss ≈ ln(V)`（容差 0.5——这是个很有用的 sanity check，以后每个模型都写）；`targets=None` 时 loss 为 None。

【做】(30 min) Tensor Puzzles 8–14。

### Day 10（周三）· 训练循环

【读】Learn the Basics → [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。看 `zero_grad / backward / step`、`model.train()/eval()`、`torch.no_grad()`。

【写】`mini_lm/train.py`
- 数据：encode 全文 → 90/10 切 train/val → `get_batch(split)` 随机采样 `[B,T]`（用 `torch.randint`）
- `AdamW(lr=1e-3)`；循环 N 步；每 100 步用 `torch.no_grad()` 估计 train/val loss
- `device` 参数：`"cpu"` / `"mps"`；所有 tensor `.to(device)`
- 先在 CPU 上跑通，再切 MPS（R8）

【测】**overfit one batch**：固定一个 batch 训练 200 步，loss 必须降到远低于初始（如 < 初始的 60%）。这条测试以后每个模型都要有。

【做】(30 min) Tensor Puzzles 15–21。统计：独立完成几题？目标 ≥15/21；其余能解释参考解法。

### Day 11（周四）· generate + checkpoint

【写】
- `mini_lm/generate.py`：`generate(model, idx, max_new_tokens, greedy=False)`：每步取最后一个位置的 logits → softmax → `torch.multinomial`（或 argmax）→ `torch.cat` 到序列。用 `@torch.no_grad()`。
- checkpoint：`torch.save({"model": model.state_dict(), "vocab_size": V}, path)`；`load` 后重建模型。

【测】
- checkpoint roundtrip：save → 新建模型 load → 同输入 logits `torch.equal`
- `generate` 输出长度 == 输入长度 + max_new_tokens
- greedy 在固定模型下确定性；采样在固定 seed 下确定性（`torch.manual_seed`）

### Day 12（周五）· 读 makemore + 整理

【读】(30 min) [makemore](https://github.com/karpathy/makemore) 中 bigram 相关部分（或视频 makemore part 1 的 bigram 神经网络段落）。对照：他怎么组织 batch？loss 怎么算？写 3 条差异。

【测】整理 `mini_lm/` 为最终结构：`data.py / model.py / train.py / generate.py / tests/`；`ruff`；确保 `pytest` 全绿且 < 30 s（训练类测试用很少的步数）。

【记】复读 [gates.md → Gate 1](gates.md#gate-1) 清单。明天考试用的目录名：`gate1/`。

### Day 13（周六）· 🚪 Gate 1（3 小时）

规则：空目录 `gate1/`，只允许查 Python / PyTorch / pytest **官方文档**；不打开 `mini_lm/`；不问 AI 任何实现问题；计时 3 小时。

要求：`data.py / model.py / train.py / generate.py / tests/`，功能：读 txt → char vocab → encode/decode → (x,y) batch → `BigramLM` → CE loss → AdamW 训练循环 → greedy/sample 生成 → save/load → **5 个必测**：tokenizer roundtrip / dataset x-y shift / 输出 shape / one-batch loss 下降 / checkpoint roundtrip。

结束后：对照清单逐项 yes/no → 判级 → `git tag gate-1`（Pass 时）。把结果写进日志。

### Day 14（周日）· ⏸ Buffer

- **Pass**：休息，或做 minbpe 的 RegexTokenizer 当作可选加餐；不开新内容。
- **Recoverable Fail**：只修失败的维度（例如只练 checkpoint + 测试），然后 60–90 min 定向复测。
- **Structural Fail**：启动失败协议（gates.md）：最多追加 2 天定向训练再重考；后续砍 optional，不砍基础。
