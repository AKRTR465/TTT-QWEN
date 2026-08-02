# TTT-SVCBench-Qwen

面向 SVCBench 长视频问题回答的 Qwen3-VL-8B State-TTT 实现。仓库只维护两条正式主线：

- A2 全量状态模型训练，再初始化 A5 Meta-TTT；
- 按视频隔离、按 chunk 因果更新的在线推理。

当前架构规范为 `state_ttt_qwen3vl8b_slot_memory_delta_v1`，正式配置 schema 为 13；历史阶段 gate 与 synthetic 报告不再随源码分发。

## 架构摘要

- Fast Adapter 位于 Qwen Main Visual Merger 与 video `masked_scatter` 之间；DeepStack 保持原始路径。
- A5 的 token key 由 Query 和写入前 Bank 的有效语义记录共同构造；per-video 在线状态是
  单块零初始化 768x768 delta-rule memory `M`，写入对来自已提交 slot state 的
  (probe-attention key, projected value, gated eta)，`P_C` 与 memory 接口
  （key probe、value projection、eta gate、read gate、forget gate）同属 `associative` 组。
- 更新顺序固定为“当前 chunk 读取 M_{t-1}，闭式 delta-rule 写入的 M_t 从下一 chunk 生效”。
- 状态路包含 Spatial Slot Encoder、Temporal Encoder、O1/O2/E1/E2 heads、Structured State Bank、Identity Bank、Retriever 和 Deterministic Reader。
- State Bank 同时维护写后 aggregate/Confirmed 状态和 append-only retrieval history；Query 从写前 history 重投影 768D source，Reader 直接读取写后状态。
- Reader 负责精确计数及证据，Qwen 负责自然语言答案。
- A5 没有 Support 内层损失：写入是闭式并行 delta rule（Ση ≤ 1 收缩界），K=8 截断的
  meta 梯度沿线性递推回传到 memory 接口与 token keys，截断点 detach 保值。
- A2/A5 正式训练唯一使用 `ema_answer_ref`：loss EMA 对齐 Answer 尺度，再按
  `q_target/q_operator/q_time` 激活面的梯度 RMS EMA 平衡四项 official-weak loss；辅助组仍限制为
  Answer 的至多 40%（`official_weak_balance.group_weight`）。
- O2-Unique 计数监督为软去重目标：sg(写前 confirmed 基数) + 当前 chunk 可微软新颖数，
  `o2.identity` 直接获得任务梯度；O2-Gain 保持池化计数回归。

完整设计见 [ARCHITECTURE.md](./ARCHITECTURE.md)，固定决策见 [DECISIONS.md](./DECISIONS.md)。

## 环境

```powershell
uv sync --frozen
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m ruff check src tests
```

要求 Python 3.12、PyTorch 2.9、Transformers 4.57.1。模型、数据、checkpoint、环境目录和密钥不得提交到仓库。

## 正式训练

生产配置位于 `configs/h200/`，详细说明见 [docs/production-a2-a5.md](./docs/production-a2-a5.md)。

```bash
bash scripts/h200/train_fullprefix256.sh a2
bash scripts/h200/train_a5_fast_state_warmup.sh \
  /absolute/path/a2/final-checkpoint /absolute/path/v4_manifest.json
# 单个完整 8 卡 worker 可用时：
bash scripts/h200/train_a5_fast_state_warmup_8gpu.sh \
  /absolute/path/a2/final-checkpoint /absolute/path/v4_manifest.json
bash scripts/h200/train_a5_associative_lttt_finalonly.sh \
  /absolute/path/a2/final-checkpoint \
  /absolute/path/warmup_run/a5_warmup_bundle \
  /absolute/path/v4_manifest.json
```

固定训练语义：

- A2：Qwen、状态模块和 W0 全量解冻，Associative 投影冻结，memory 写入不可达；
- A5 Warmup：从完整 A2 checkpoint 初始化，Qwen、W0、RMSNorm/P_in/P_out 全冻结且不进入
  optimizer；只训练 P_C、memory 接口和四个 state 组共 256 step，并保存不含 Qwen、optimizer
  与瞬态状态的原子 handoff bundle；
- A5 Main：重新加载完整 A2 checkpoint并严格叠加 handoff bundle，恢复原部分解冻策略，训练
  4 epoch 且只保存 final checkpoint；
- A5 Support 写入是纯归档：slot (key, value) 对以学习到的 eta 强度写入 `M`，无
  auxiliary loss 加入 Outer objective，Answer/State Query loss 通过 deferred VJP 学习
  写入几何与强度；
- Support 保持 8/16 帧动态块；每个 Query 独立读取 `[0, query_time]` 因果前缀，2 FPS、最多
  256 帧，动态视觉 Token 数不变；
- A5 多 Query 逐个 forward/backward，释放各自激活；所有 Query 使用同一段末 memory
  state，Bank/FSM 仍是只读权威状态；每个 Query 的 FP32 memory cotangent 按联合范数裁剪
  到 `a5.query_meta_gradient.max_norm`（当前 10.0）后在 segment 内求和，直接
  Query→Qwen/State 梯度不参与该逐 Query 裁剪；
- W0 保持 checkpoint/model dtype，瞬态 `M` 和 fast/memory 核心固定为 FP32；写入路径不
  产生独立 Outer backward，关联中间量不跨调用保存。
- 4/8 rank sampler 保持任务/segment parity，padding 样本 loss 权重为零；
- checkpoint 保存模型、optimizer、scheduler、RNG，但排除 `M`、Bank、cache 和 FSM runtime。

`ema_answer_ref` 是唯一 official-weak loss-balance 算法，不再提供 mode 或 experimental
开关。loss 与 gradient EMA 均采用一步滞后并随同阶段 checkpoint 恢复；A2 初始化 A5 时
清零。Task、Operator、Retrieval、Time 始终占固定四槽，缺失监督不更新对应 EMA，也不
重分配预算。

Outer AdamW 将状态参数拆为 `state_shared`、`state_task`、`state_router_time` 和
`state_retrieval` 四个独立裁剪组；四组 cap 均为 0.05，RSS 更新预算等于旧单组 0.1，
因此 Task 尖峰不会再同步缩小 SemanticProjector/Retrieval 梯度。

## 在线推理

`ttt-svcbench-infer` 是正式 JSON 入口，要求 `--run`、`--checkpoint`、`--model-root`、`--device`、`--dtype` 与 `--output`。默认 Query 视觉模式为完整因果前缀 256 帧；可用
`--query-visual-mode recent_chunk --query-max-frames 16` 运行兼容消融。Qwen generation、在线
updater、严格 checkpoint 和 per-video runtime 生命周期均由同一 bundle 组装。

运行时必须保证：

- 禁止答案、count、occurrence_times、counting_type 和 counting_subtype 进入 Support/Query 模型输入；
- query_time 之后的帧不得进入状态更新或回答；
- 新 M_t 只影响下一 chunk；
- 每个视频 reset/release，异常路径同样 release；
- generate 不修改 memory、Bank、FSM 或 temporal state。

## 验证边界

本机 tiny/CPU 测试只证明接口、梯度、因果性和状态隔离。真实 Qwen3-VL-8B、BF16、4/8 卡显存、吞吐、收敛和效果必须由独立 H200 运行记录证明。
