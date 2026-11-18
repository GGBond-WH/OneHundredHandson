# Specs · Phase I 每周四的 Spec Challenge

流程和 Plan 1 完全相同：

1. 读 spec，不搜、不看任何实现（包括 vLLM 和 nano-vllm）。
2. 让 AI **只根据 spec** 生成黑盒测试（不给它你的代码和思路）。检查测试是否符合 spec → `git commit -m "test: freeze specN tests"`。
3. 实现到冻结测试全绿。不许为了通过而改测试；可以加测试。
4. 记录空白文件延迟和卡点。

每个 spec 都附"最少验收测试"，AI 生成的测试至少要覆盖这些。

---

<a id="spec-1"></a>
## Spec #1（Day 4）· `blocks_needed` + `compute_slot_mapping`

```python
def blocks_needed(num_tokens: int, block_size: int) -> int:
    """存放 num_tokens 个 token 至少需要几个块。"""

def compute_slot_mapping(
    block_table: list[int], start_pos: int, num_tokens: int, block_size: int
) -> torch.Tensor:
    """返回 [num_tokens] 的 long 张量：绝对位置为 start_pos, start_pos+1, ... 的这些 token
    各自应该写入哪个槽位。"""
```

行为：
- 槽位编号的含义：把整个分页存储看成 `num_blocks × block_size` 个槽位排成一行时，这个 token 所在的下标。逻辑上第 `j` 个块对应物理块 `block_table[j]`。**公式不给**，从这句话推。
- `block_table` 不够长（装不下 `start_pos + num_tokens` 个 token）时抛 `ValueError`。
- `num_tokens == 0` 返回空张量；任何参数为负抛 `ValueError`。

最少验收测试：
- `block_size=4, block_table=[7, 2], start_pos=0, num_tokens=6` → `[28, 29, 30, 31, 8, 9]`
- `start_pos=3, num_tokens=2`（同上表）→ `[31, 8]`，跨块边界
- `block_table=[7]`、`num_tokens=5` 抛 `ValueError`
- `blocks_needed(0, 4) == 0`，`blocks_needed(1, 4) == 1`，`blocks_needed(4, 4) == 1`，`blocks_needed(5, 4) == 2`
- 返回 dtype 是 `torch.long`

---

<a id="spec-2"></a>
## Spec #2（Day 11）· `build_step_inputs`

```python
@dataclass
class StepInputs:
    input_ids: torch.Tensor        # [N]   本步所有请求的新 token，首尾相接
    positions: torch.Tensor        # [N]   每个新 token 的绝对位置
    slot_mapping: torch.Tensor     # [N]   每个新 token 写入的槽位
    query_start_loc: torch.Tensor  # [R+1] 第 r 个请求的新 token 在扁平序列里占 [query_start_loc[r], query_start_loc[r+1])
    seq_lens: torch.Tensor         # [R]   本步结束后每个请求在缓存里的总长度
    logits_indices: torch.Tensor   # [R]   每个请求最后一个新 token 在扁平序列里的下标

def build_step_inputs(scheduled: list[tuple[Request, int]], block_size: int) -> StepInputs:
    """scheduled 中每一项是 (请求, 本步新 token 数)，顺序即输出顺序。"""
```

`Request` 至少提供：全部 token（prompt + 已生成）、`num_computed_tokens`、`block_table`。本步要算的是 `token[c : c+n]`，其中 `c = num_computed_tokens`。假设调用前块已经分配好了。

行为：
- 所有张量 dtype 为 `torch.long`。
- `n <= 0`，或 `c + n` 超过请求的 token 总数，抛 `ValueError`。
- `scheduled` 为空抛 `ValueError`。
- 槽位必须和 Spec #1 的结果一致。

最少验收测试：
- 请求 A：`c=0`、`n=3`（prefill）；请求 B：`c=5`、`n=1`（decode，总共 6 个 token）。期望：`N=4`，`positions=[0,1,2,5]`，`query_start_loc=[0,3,4]`，`seq_lens=[3,6]`，`logits_indices=[2,3]`
- `input_ids` 等于两个请求对应切片的拼接
- `slot_mapping` 与分别调用 Spec #1 的结果拼接后相等
- `query_start_loc[-1] == N`
- 非法 `n` 和空输入抛错；输出顺序与输入顺序一致

---

<a id="spec-3"></a>
## Spec #3（Day 18）· 调度器不变量（性质测试）

把你 Day 15–17 写的调度器适配到下面的接口（可以加内部方法，公开接口不变）：

```python
class Scheduler:
    def __init__(self, num_blocks: int, block_size: int,
                 max_num_seqs: int, max_num_batched_tokens: int) -> None: ...
    def add_request(self, req: Request) -> None: ...
    def schedule(self) -> list[tuple[Request, int]]:
        """决定本步每个请求算几个新 token，并保证这些请求的块已经分配好。"""
    def update(self, scheduled: list[tuple[Request, int]], sampled: list[int]) -> list[Request]:
        """sampled[i] 是 scheduled[i] 最后一个位置的采样结果。返回本步完成的请求。"""
    @property
    def num_free_blocks(self) -> int: ...
```

测试不使用真实模型：用一个假的"采样器"，比如 `sampled = (请求 id × 31 + 位置) % 50`，再规定 token 0 是 EOS。

必须始终成立的不变量：

1. 每步 `Σ n ≤ max_num_batched_tokens`；被调度的请求数 `≤ max_num_seqs`；同一请求一步内最多出现一次。
2. 对每个被调度的请求，`1 ≤ n` 且 `c + n ≤ 请求的 token 总数`。
3. `schedule()` 返回后，每个被调度请求的块数 `≥ blocks_needed(c + n)`。
4. 块守恒：空闲块数 + 所有存活请求持有的块数 == `num_blocks`（本 spec 不开前缀缓存，所以不存在共享块）。
5. 本步结束时还没算完 prompt 的请求，`update` 不能给它追加 token。
6. 被抢占的请求：块全部归还，`num_computed_tokens` 归零，之后能恢复并最终完成。
7. 活性：只要 `num_blocks` 至少够装下最大的单个请求，随机到达的请求最终全部完成，而且步数有上界（测试里设一个宽松的上界，比如所有请求 token 总数的 10 倍）。
8. 全部完成后，空闲块数 == `num_blocks`。

最少验收测试：用固定 seed 生成 50 组随机负载（请求数、到达时间、prompt 长度、`max_tokens`、`num_blocks`、预算各不相同），每一步检查 1–5，结束时检查 6–8；再加 3 个手工构造的边界场景：预算恰好等于一个 prompt 的长度、`num_blocks` 恰好够最大请求、`max_num_seqs = 1`。

---

<a id="spec-4"></a>
## Spec #4（Day 25）· `FreeBlockQueue`

```python
class Block:
    block_id: int
    ref_cnt: int
    block_hash: int | None
    # 队列需要的链接字段由你决定

class FreeBlockQueue:
    def __init__(self, blocks: list[Block]) -> None:
        """初始顺序即队列顺序（头部最先被取走）。"""
    def popleft(self) -> Block: ...
    def remove(self, block: Block) -> None:
        """把一个块从队列任意位置摘掉，必须是 O(1)。"""
    def append(self, block: Block) -> None: ...
    def __len__(self) -> int: ...
    def to_list(self) -> list[Block]:
        """从头到尾列出（测试用）。"""
```

行为：
- `remove` 和 `append` 都是 O(1)，不允许遍历队列。
- 空队列 `popleft` 抛错；`remove` 一个不在队列里的块抛错；`append` 一个已经在队列里的块抛错。
- 摘除和追加后，`to_list()` 的顺序正确，`len` 正确。

最少验收测试：头、中、尾三种位置的 `remove`；`remove` 后再 `append` 到尾部；连续 `popleft` 直到空；三种非法操作抛错；**规模测试**：10 万个块按随机顺序全部 `remove` 必须在 2 秒内完成（list 的 O(n) 删除需要几分钟，这个上限对 O(1) 实现非常宽松）。

完成之后才允许读 vLLM 的 `FreeKVCacheBlockQueue`（`vllm/v1/core/kv_cache_utils.py`），在 Phase III Day 50 对照。
