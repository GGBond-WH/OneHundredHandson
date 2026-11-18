# Phase II · vLLM 使用者视角 + 开发环境 + 架构通读（Day 29–49）

**目标**：在 Mac 上从源码构建 vLLM，能当用户用它，能打断点追踪一个请求从进入到输出的完整路径，掌握 fork / branch / rebase / DCO 这套 GitHub 工作流。
**产物**：可调试的 vLLM 源码环境；`notes/lifecycle.md`（经断点核实的调用链）；`logs/probe_log.md` 前 6 条；`logs/pr_reading_log.md` 前 4 条。
**Gate B**：Day 47–48。

从这一阶段起，周四是 🔍 **Probe**（先预测再验证，见 [00-rules.md R1](00-rules.md)），周五是 📰 **PR 阅读**。源码路径都列在 [source-map.md](source-map.md)。

---

## Week 5 · 开发环境与 GitHub 工作流

### Day 29（周一）· fork + 构建

【做】
1. 在 GitHub 上 fork `vllm-project/vllm` 到你的账号。
2. 克隆**你的 fork** 到仓库外面。vLLM 历史很长，用部分克隆只下载需要的文件内容，`git log` 照样完整：
   ```bash
   git clone --filter=blob:none https://github.com/<你的用户名>/vllm.git ~/code/vllm
   cd ~/code/vllm
   git remote add upstream https://github.com/vllm-project/vllm.git
   git remote -v
   ```
3. 建 Python 3.12 环境。你现在的系统 Python 是 3.13；vLLM 支持 3.11–3.14，但官方 CI 用 3.12，贡献指南建议开发时也用 3.12：
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   ```
4. 按官方 macOS 文档从源码构建 CPU 版（需要 Xcode Command Line Tools，Apple Clang ≥ 15）：
   ```bash
   uv pip install -r requirements/cpu.txt
   uv pip install -e .
   ```
   macOS 上会自动设 `VLLM_TARGET_DEVICE=cpu`。构建会编译 C++ CPU 扩展，需要一段时间。报错先对照文档里的 Troubleshooting（找不到 `<map>` 这类头文件，就重装 Command Line Tools）。

【记】`notes/setup.md`：构建用了多久、遇到什么报错、怎么解决的，以及 `git rev-parse HEAD`（你构建的 vLLM commit）。以后所有实验和 PR 描述都要注明这个 commit。

✅ `python -c "import vllm; print(vllm.__version__)"` 能打印版本号。

### Day 30（周二）· 当一次用户

【写】`notes/first_run.md` 里的两个小脚本：
- **离线推理**：`from vllm import LLM, SamplingParams`，用一个小模型（比如 `HuggingFaceTB/SmolLM2-135M-Instruct`，Plan 1 Week 10 你见过它的基座版；或 `Qwen/Qwen2.5-0.5B-Instruct`），对 4 个 prompt 做 greedy 生成。
- **在线服务**：`vllm serve <模型>` 启动 OpenAI 兼容服务，用 `curl` 或 `openai` 客户端各发一次请求。

CPU 版的 KV cache 空间由环境变量 `VLLM_CPU_KVCACHE_SPACE` 控制，单位 GB，默认 4。试着设成 1 再启动，观察启动日志里报告的 KV cache 容量怎么变。

### Day 31（周三）· 测试基线 + pre-commit

【做】
```bash
uv pip install pytest pytest-asyncio
pytest -q tests/v1/core/test_kv_cache_utils.py
pytest -q tests/v1/core/test_prefix_caching.py
pytest -q tests/v1/core/test_scheduler.py
```
官方文档提醒过："not all unit tests pass when run on CPU platforms"。所以如实记录每个文件的通过 / 失败 / 跳过数量，对每个失败做分类：是 CPU 平台限制、环境问题，还是看起来像真 bug？分不清就写"未分类"，不要猜。这份 `notes/test_baseline.md` 以后会帮你判断"这个失败是我改出来的，还是本来就有"。

【做】`uv pip install "pre-commit>=4.5.1" && pre-commit install`。随便改一个文件的注释，`git add` 后跑 `pre-commit run`，看它检查了什么；然后 `git restore` 撤销。

### Day 32（周四）· GitHub 演练（只在你自己的 fork 里）

Plan 1 只教了本地 git。今天把 PR 工作流完整走一遍，**全部发生在你自己的 fork 里，不碰 upstream**：

1. `git fetch upstream && git switch -c drill/first upstream/main`
2. 改一个文件，`git commit -s`（`-s` 加 DCO 签名，`git log -1` 确认有 `Signed-off-by:` 行）。先读 [00-rules.md R4](00-rules.md) 里关于邮箱隐私的提醒，再决定用哪个邮箱。
3. `git push -u origin drill/first`，然后在 GitHub 上开一个 PR，**目标选你自己 fork 的 main**（不是 vllm-project）。
4. 制造冲突：在 fork 的 main 上改同一行，然后在分支上 `git rebase origin/main`，解决冲突，`git push --force-with-lease`。
5. 学会 `gh` 命令行：`gh pr view`、`gh pr checks`、`gh pr list`。

【记】把这套命令写进 `notes/github_workflow.md`，以后每个 PR 照着走。

### Day 33（周五）· 📖 vllm-metal

【读】(≤45 min) [vllm-metal](https://github.com/vllm-project/vllm-metal) README 的 Architecture 一节。它的分工是：上游 vLLM 提供 API server、调度器、分页块管理器，vllm-metal 负责 Metal 上 paged attention 这条路径。也就是说，你 Phase III 要读的调度器和 KV cache 管理器，在 vllm-metal 里同样在工作。
【做】需要 Homebrew：`brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal && brew install vllm-project/vllm-metal/vllm-metal`。它要求 macOS 15+，你的机器满足。用 `mlx-community/Qwen2.5-0.5B-Instruct-4bit` 跑一次，和 Day 30 的 CPU 版比较生成速度。注意两边的模型精度不同（4-bit 对比 FP32/FP16），只能看数量级。

### Day 34（周六）· 🧪 引擎参数实验

【写】`notes/engine_args.md`。用离线 `LLM` 写一个小脚本（或者看 `vllm bench --help` 后用 `vllm bench throughput` / `latency`），在 CPU 上做三组对比，每组只改一个参数：

| 参数 | 实验 | 你预期会看到什么（先写） |
|---|---|---|
| `enable_prefix_caching` | 20 个共享 200 token 系统提示词的请求，开 vs 关 | |
| `max_num_batched_tokens` | 长 prompt 与短请求混合，调小 vs 调大 | |
| `block_size` | 两个值（CPU 后端支持哪些块大小，以报错或文档为准） | |

先写预期，再跑，再解释差异。记录 device / dtype / 模型 / vLLM commit。这些参数对应你 minivllm 里的同名概念，正是 Phase I 自己写过的东西。

### Day 35（周日）· 🔁 Retrieval（60 min）

闭卷写出：fork → 同步 upstream → 建分支 → 签名提交 → rebase → 推送 → 开 PR 的全部命令；以及 vLLM macOS 源码开发安装的命令。写完对照 `notes/` 修正。

---

## Week 6 · 一个请求的完整生命周期

### Day 36（周一）· 先预测

【读】vLLM V1 博客（resources）+ 仓库里的 `docs/design/arch_overview.md`。
【写】`notes/lifecycle_predicted.md`：**先不看代码**，画出你认为一次 `LLM.generate()` 会经过的组件和顺序，从 Python API 到模型前向再回到输出。写不出函数名就写组件名。明天用断点检验。

### Day 37（周二）· 🔍 断点追踪

vLLM 默认把 EngineCore 放在**另一个进程**里运行，你在主进程打的断点停不下来。调试时用：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=DEBUG python your_script.py
```

【做】从 `LLM.generate` 出发，用 VS Code 断点逐层进入，至少确认这些站点（文件都在 [source-map.md](source-map.md)）：`vllm/v1/engine/llm_engine.py` → `vllm/v1/engine/core.py` 的 `EngineCore.step` → `vllm/v1/core/sched/scheduler.py` 的 `Scheduler.schedule` → CPU worker 和 model runner（`vllm/v1/worker/cpu_worker.py`、`vllm/v1/worker/cpu/model_runner.py`）→ `Scheduler.update_from_output` → `vllm/v1/engine/output_processor.py`。
【记】`notes/lifecycle.md`：经断点核实的 `file:function` 链，每一站一句话说明它做什么。和昨天的预测对照：哪些组件你没想到？

### Day 38（周三）· 🔍 SchedulerOutput 对应到 StepInputs

【做】在 `Scheduler.schedule` 返回的地方停下，展开 `SchedulerOutput`：新请求和已在运行的请求分别放在哪里？每个请求本步的新 token 数放在哪个字段？
【写】映射表：vLLM 的字段 ↔ 你 minivllm 的 `StepInputs` / 调度结果。再往下追到 model runner，找到 `query_start_loc`、`seq_lens`、`slot_mapping` 被构造的地方（它们在 `vllm/v1/attention/backend.py` 的 `CommonAttentionMetadata` 里）。

### Day 39（周四）· 🔍 Probe #1–2

【读】`tests/v1/core/utils.py`：vLLM 的测试怎么造请求、造调度器。这是你以后写测试的模板。
【做】Probe（格式见 `templates/probe_log.md`）：
1. **预测**（写进 probe_log）：前缀缓存开启、`block_size=16`，第一个请求 prompt 有 50 个 token，第二个请求 prompt 是"前 48 个 token 相同 + 不同的结尾"。第二个请求被调度时，有多少 token 被认为已经计算过？
2. 在你的 vLLM 分支 `probe/week6` 上，照 `tests/v1/core/` 的写法写一个测试验证。
3. 记录结果。预测错了就写错在哪。

Probe #2 自选一个关于调度的问题，同样流程。Probe 测试留在本地 probe 分支，**不提交 PR**。但如果你发现某个行为确实没被任何测试覆盖，把它记进 `logs/issue_candidates.md`，它可能成为 Phase IV 的 PR 选题。

### Day 40（周五）· 📰 Anatomy 博客 + 第一次 PR 阅读

【读】《Inside vLLM: Anatomy of a High-Throughput LLM Inference System》（resources），对照你 Day 37 的调用链。
【做】第一次 PR 阅读。vLLM 合并 PR 用 squash，所以 `main` 上一个 commit 就是一个 PR，标题末尾带 PR 号：
```bash
git log --since="2 months ago" --oneline -- vllm/v1/core/ | head -20
```
挑 1 个改动不大的，在 GitHub 打开对应 PR，按 `templates/pr_reading_log.md` 记录。

### Day 41（周六）· 🧪 你 Mac 上实际运行的那条路径

【读】`vllm/v1/worker/cpu/model_runner.py` 和 `vllm/v1/worker/cpu/block_table.py`。你在 Mac 上跑 vLLM 时，模型前向就走这里。
【记】`notes/cpu_runner.md`：slot mapping 在哪里算出来？block table 在 worker 侧怎么存？和你 Spec #1、Spec #2 的实现对照。

### Day 42（周日）· 🔁 Retrieval（60 min）

闭卷默写请求生命周期的 `file:function` 链，然后对照 `notes/lifecycle.md` 修正。

---

## Week 7 · KV cache 的容量从哪来 + Gate B

### Day 43（周一）· 设计文档 + 容量预测

【读】`docs/design/paged_attention.md`、`docs/design/prefix_caching.md`。前缀缓存那篇对照你 Week 4 的设计逐条看。
【做】用 Plan 1 的 `estimate_kv_bytes`，预测 SmolLM2-135M 在 `VLLM_CPU_KVCACHE_SPACE=1` 时能放多少个块（`block_size` 用你实验时的值），再和启动日志报告的容量对照。对不上就找原因：dtype？块大小？还有别的开销？

### Day 44（周二）· 🔍 块数是怎么定出来的

【做】从启动日志里那一行容量信息出发，用断点或 `grep` 反向找到决定 KV cache 块数的代码（提示：`vllm/v1/core/kv_cache_config_builder.py`、`kv_cache_planning.py`、`kv_cache_utils.py`）。
【记】一段话说清：可用内存 → 每块字节数 → 块数，这条链在代码里的位置。

### Day 45（周三）· 🔍 参数在哪里生效

【做】Day 34 那几个引擎参数（`max_num_seqs`、`max_num_batched_tokens`、`enable_prefix_caching`、`block_size`），用 `grep -n` 找到它们在 `scheduler.py` 和 `kv_cache_manager.py` 里被读取的位置。
【写】对照表：参数 → vLLM 里起作用的位置 → minivllm 里对应的代码。

### Day 46（周四）· 🔍 Probe #3–6

四个 Probe，每个都先写预测：
- 空闲块的淘汰顺序（请求 A、B 先后结束，新请求分配块时先淘汰谁的块）
- 一个空闲但带哈希的块被新请求命中后，它在空闲队列里的位置会怎样
- 块不够时，调度器抢占哪个请求
- 整段 prompt 都命中缓存时，有多少 token 会被计算（你 Phase I Day 23 自己做过一个决定，vLLM 的决定是什么？）

### Day 47–48（周五、周六）· 🚪 Gate B

见 [gates.md → Gate B](gates.md#gate-b)。通过后 `git tag vllm-gate-b`（打在 One Hundred 仓库，不是 vLLM 仓库）。

### Day 49（周日）· ⏸ Buffer
