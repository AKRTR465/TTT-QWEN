# TTT-SVCBench-Qwen

面向 SVCBench 长视频问答的 Qwen3-VL-8B State-TTT 实现。架构规范 `state_ttt_qwen3vl8b_slot_memory_delta_v1`。

## 架构

- **Fast Adapter** 位于 Qwen Main Visual Merger 与 video `masked_scatter` 之间；DeepStack（8/16/24）保持原始路径不变。
- **per-video 在线状态**是单块零初始化的 768×768 delta-rule memory `M`。写入对来自已提交 slot state 的
  (probe-attention key, projected value, gated eta)；`P_C` 与 memory 接口（key probe、value projection、
  eta gate、read gate、forget gate）同属 `associative` 参数组。
- **更新顺序固定**：当前 chunk 读取 `M_{t-1}`，闭式 delta-rule 写出的 `M_t` 从下一 chunk 生效。
- **状态路**：Spatial Slot Encoder、Temporal Encoder、O1/O2/E1/E2 heads、Structured State Bank、
  Identity Bank、Retriever、Deterministic Reader。State Bank 同时维护写后 aggregate/Confirmed 状态与
  append-only retrieval history；Query 从写前 history 重投影 768D source，Reader 直接读取写后状态。
- **Reader 负责精确计数与证据，Qwen 负责自然语言答案** —— Answer 路径消费 Reader 发出的数字 token id。
- **A5 没有 Support 内层损失**：写入是闭式并行 delta rule（Ση ≤ 1 收缩界），K=8 截断的 meta 梯度沿线性
  递推回传到 memory 接口与 token keys，截断点 detach 保值。
- **loss 平衡**只有 `ema_answer_ref` 一种，两级串联：先用 loss EMA 对齐 Answer 尺度，再按
  `q_target/q_operator/q_time` 激活面的梯度 RMS EMA 平衡四项 official-weak loss。辅助组上限为 Answer 的
  40%。Task/Operator/Retrieval/Time 始终占固定四槽，缺失监督不更新对应 EMA 也不重分配预算。
- **Outer AdamW** 将状态参数拆为 `state_shared` / `state_task` / `state_router_time` / `state_retrieval`
  四个独立裁剪组，每组 cap 0.05。注意所有 h200 配置 `max_grad_norm: 0.0`，HuggingFace 自带裁剪已关闭，
  `OuterGradientController` 是系统中唯一的梯度裁剪。
- **O2-Unique 计数监督**为软去重目标：sg(写前 confirmed 基数) + 当前 chunk 可微软新颖数，`o2.identity`
  直接获得任务梯度；O2-Gain 保持池化计数回归。

## 训练

生产配置在 `configs/h200/`。三个阶段，顺序执行：

```bash
# M1 · A2：Qwen、状态模块、W0 全量解冻；Associative 投影冻结；memory 写入不可达
bash scripts/h200/train_fullprefix256.sh a2

# M2 · A5 Warmup：从完整 A2 checkpoint 初始化，冻结 Qwen/W0/RMSNorm/P_in/P_out，
#                  只训 P_C + memory 接口 + 四个 state 组共 256 step，产出原子 handoff bundle
bash scripts/h200/train_a5_fast_state_warmup.sh \
  /absolute/path/a2/final-checkpoint /absolute/path/v4_manifest.json

# M3 · A5 Main：重载完整 A2 checkpoint 并叠加 handoff bundle，恢复部分解冻，4 epoch，只存 final checkpoint
bash scripts/h200/train_a5_associative_lttt_finalonly.sh \
  /absolute/path/a2/final-checkpoint \
  /absolute/path/warmup_run/a5_warmup_bundle \
  /absolute/path/v4_manifest.json
```

固定语义：

- Support 保持 8/16 帧动态块；每个 Query 独立读取 `[0, query_time]` 因果前缀，2 FPS、最多 256 帧。
  `state_query_visual_mode: recent_chunk`（16 帧）与 `answer_query_visual_mode: causal_prefix`（256 帧）
  是两个不同的旋钮，前者是唯一合法的 state 取值。
- A5 多 Query 逐个 forward/backward 并释放各自激活；所有 Query 使用同一段末 memory state，Bank/FSM 是
  只读权威状态；每个 Query 的 FP32 memory cotangent 按联合范数裁剪到
  `a5.query_meta_gradient.max_norm`（10.0）后在 segment 内求和。
- W0 保持 checkpoint/model dtype；瞬态 `M` 与 fast/memory 核心固定 FP32。
- 4 rank sampler 保持任务/segment parity，padding 样本 loss 权重为零（所有 rank 执行相同次数的 backward
  集合通信，这是分布式正确性要求，不是记账）。
- checkpoint 保存模型、optimizer、scheduler、RNG，排除 `M`、Bank、cache 与 FSM runtime。
- 非有限 loss/梯度导致整步跳过而非部分应用。
- 所有 h200 配置 `report_to: none`，因此 `scripts/h200/launch_4gpu.sh` 的 `tee train.log` 是承载
  per-step loss 的唯一通道。

## 推理

`ttt-svcbench-infer` 是正式 JSON 入口，要求 `--run`、`--checkpoint`、`--model-root`、`--device`、
`--dtype`、`--output`。默认 Query 视觉模式为完整因果前缀 256 帧。

运行时保证：

- 禁止 answer、count、occurrence_times、counting_type、counting_subtype 进入 Support/Query 模型输入
  （`data.assert_runtime_payload_safe` 是全仓唯一保留的强校验：此处静默失败等于训练目标泄漏、指标作废）；
- `query_time` 之后的帧不进入状态更新或回答；
- 新 `M_t` 只影响下一 chunk；
- 每个视频 reset/release，异常路径同样 release；
- generate 不修改 memory、Bank、FSM 或 temporal state。

## 环境

```powershell
uv sync --frozen
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m ruff check src tests
```

Python 3.12、PyTorch 2.9、Transformers 4.57.1。`configs/h200/requirements-h200.lock.txt` 与
`scripts/h200/launch_4gpu.sh` 的版本墙共同防止 Qwen3-VL 在 torch 2.9.x 上的 Conv3D 回归，两者都不要删。
模型、数据、checkpoint、环境目录与密钥不得提交。

`scripts/preprocess_cache.py prewarm` 预热的磁盘缓存（约 200GB）是训练吞吐的前提；三个主线配置都以
`readonly` 模式读取它。缓存 miss 不会报错，但会静默使 run 慢一到两个数量级。

## 已知局限与不可静默破坏的三条

1. **`memory_eta_gate_output.weight` 必须零初始化。** 它使 Ση = `active_slots × eta_gate_init`
   = 32 × 0.02 = **0.64**，与 seed 和 slot 尺度无关。恢复默认 Xavier 初始化会把 Ση 静默钉在 **1.0**，
   正是 delta-rule 的湮灭点（retention = −β = −0.01）。不会有任何报错。
2. **token centering 只作用于 `memory_keys`，绝不作用于 W0 路径。** 对 W0 输入 centering 会破坏
   `M = 0` 时逐位等于静态 forward 的不变量，A2 能力就不再在每个 episode 起点被保留。
   `_centered_over_valid_slots` / `_centered_over_valid_tokens` 中的 `clamp_max` 是纠正（防止落在均值上的
   slot 把舍入噪声放大到满量程），不是诊断残留。
3. **`associative_contract_version = 4`。** revision 3 的 warmup bundle 与 revision 4 的 key 语义不兼容；
   混用会产生两套 key 空间叠加，且没有任何报错。bundle 的 provenance 校验已移除，请以文件名区分。

slot 塌缩是**结构性**的，不是训练缺陷：未训练 encoder 上实测 slot pairwise cosine 0.999780，跨 seed 稳定。
写入容量经两级 centering 从 1.001 提升到约 32（满量程 32）。

**验证边界**：本机 tiny/CPU 测试只证明接口、梯度、因果性与状态隔离。真实 Qwen3-VL-8B、BF16、4 卡显存、
吞吐、收敛与效果必须由独立 H200 运行记录证明。主线削减后的第一次 H200 run 必须与削减前做同 seed 对照。
