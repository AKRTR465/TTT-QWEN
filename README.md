# TTT-SVCBench-Qwen

面向 SVCBench 长视频问题回答的 Qwen3-VL-8B State-TTT 实现。仓库只维护两条正式主线：

- A2 全量状态模型训练，再初始化 A5 Meta-TTT；
- 按视频隔离、按 chunk 因果更新的在线推理。

当前规范版本为 `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。配置、运行时和测试均以 v5 为准；历史阶段 gate 与 synthetic 报告不再随源码分发。

## 架构摘要

- Fast Adapter 位于 Qwen Main Visual Merger 与 video `masked_scatter` 之间；DeepStack 保持原始路径。
- 在线状态仅更新两块 768x768 fast matrix，更新顺序固定为“当前 chunk 使用 Wt，更新后的 Wt+1 从下一 chunk 生效”。
- 状态路包含 Spatial Slot Encoder、Temporal Encoder、O1/O2/E1/E2 heads、Structured State Bank、Identity Bank、Retriever 和 Deterministic Reader。
- Reader 负责精确计数及证据，Qwen 负责自然语言答案。
- A5 使用 `L_pred + 0.5 L_id + 0.5 L_event`，K=8 截断二阶梯度并重锚 W0。
- A2/A5 正式训练默认使用 `instant_equal`：按当前全局 sum/count 对齐四项
  official-weak loss，并将辅助组限制为 Answer 的至多 30%；`legacy_sum` 仅保留为兼容消融。

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
bash scripts/h200/train_fullprefix256.sh a5 /absolute/path/a2/checkpoints/final-checkpoint
```

固定训练语义：

- A2：Qwen、状态模块和 W0 全量解冻，Predictor 冻结，禁用 Inner SGD；
- A5：从完整 A2 checkpoint 初始化，Predictor 启用，Support 不设人工上限，K=8 截断二阶；
- Support 保持 8/16 帧动态块；每个 Query 独立读取 `[0, query_time]` 因果前缀，2 FPS、最多
  256 帧，动态视觉 Token 数不变；
- A5 多 Query 逐个 forward/backward，释放各自激活；所有 Query 共用同一 `W_after` 和只读
  Bank/FSM snapshot；
- 四卡 sampler 保持任务/segment parity，padding 样本 loss 权重为零；
- checkpoint 保存模型、optimizer、scheduler、RNG，但排除 Wt、Bank、cache 和 FSM runtime；
- `formal_evaluation_enabled=false`，直至独立校准和正式评估完成。

`loss.official_weak_balance.mode` 默认为 `instant_equal`。它无 EMA、无额外 checkpoint
状态；每个 A2 micro-step 或 A5 episode 使用一次固定长度 collective，Task、Operator、
Retrieval、Time 各占固定四分之一插槽，缺失监督不重分配预算。`legacy_sum` 仍可用于消融。

## 在线推理

`ttt-svcbench-infer` 是正式 JSON 入口，要求 `--run`、`--checkpoint`、`--model-root`、`--device`、`--dtype` 与 `--output`。默认 Query 视觉模式为完整因果前缀 256 帧；可用
`--query-visual-mode recent_chunk --query-max-frames 16` 运行兼容消融。Qwen generation、在线
updater、严格 checkpoint 和 per-video runtime 生命周期均由同一 bundle 组装。

运行时必须保证：

- 禁止答案、count、occurrence_times、counting_type 和 counting_subtype 进入 Support/Query 模型输入；
- query_time 之后的帧不得进入状态更新或回答；
- 新 Wt 只影响下一 chunk；
- 每个视频 reset/release，异常路径同样 release；
- generate 不修改 Fast、Bank、FSM 或 temporal state。

## 验证边界

本机 tiny/CPU 测试只证明接口、梯度、因果性和状态隔离。真实 Qwen3-VL-8B、BF16、四卡显存、吞吐、收敛和效果必须由独立 H200 运行记录证明。
