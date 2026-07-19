# State-TTT v5 固定决策

## 当前主线

仓库只维护：

1. 正式 A2 全量状态训练；
2. A2 checkpoint 初始化的 A5 K=8 Meta-TTT；
3. per-video 在线 State-TTT 推理。

历史阶段 gate、standalone trainer、synthetic ablation harness 和 evidence bundle 不属于运行时，也不进入 wheel。

## 模型

- Qwen3-VL-8B Main Visual Merger 后插入 4096→768→4096 Fast Adapter。
- DeepStack 保持原始路径，不接入 Fast Adapter。
- 在线只更新两块 768x768 Wt；W0 是 checkpointed meta parameter。
- Spatial Encoder 固定为两阶段 Slot Attention；Temporal Encoder 固定为 6 层因果 Transformer。
- Query Encoder 固定为 4 层、512 维；State Resampler 固定输出 16 个 4096 维 token。
- Deterministic Reader 是精确计数唯一真值源，LLM 不覆盖 Reader 算术。

## 训练

- 正式流程直接 A2→A5。
- A2 全量解冻 Qwen、状态路径和 W0，冻结 Predictor，禁止 Inner SGD。
- A5 固定 `pred + 0.5 identity + 0.5 event`，K=8 截断二阶并重锚 W0。
- A3/A4 不作为生产训练阶段。
- graph anchor 只服务真实多卡动态分支，单卡不启用。
- Outer checkpoint 完整保存模型/optimizer/scheduler/RNG，排除所有临时 runtime state。

## 因果与泄漏

- Support/Query runtime 禁止答案、count、occurrence_times、counting_type 和 counting_subtype。
- query_time 后帧必须在模型边界前裁剪。
- 当前 chunk 使用旧 Wt，功能性 SGD 的结果只供下一 chunk 使用。
- hard state、Bank 与 overlap snapshot 必须 detach；不同视频不得共享 storage 或 owner。
- clean test 不得进入训练或校准。

## 推理

- 每个视频独立 reset、observe/update、answer、release。
- generation 期间 Fast、Bank、FSM、cache 不变。
- retry 只允许在相同因果状态上执行。
- 默认 audit 为 `boundary`；只有 `full` 计算 Tensor 内容 hash。
- 首版 generation 固定 greedy、`num_beams=1`、`do_sample=false`、`max_new_tokens=16`。
- 推理 checkpoint 只接受严格匹配的 safetensors。

## 验证声明

- `formal_evaluation_enabled=false`，直至阈值在训练折或独立校准集冻结。
- tiny/CPU 测试是工程证据，不是 8B 收敛、性能或科学收益证据。
- 真实 8B/H200 结果必须记录模型 revision/hash、BF16、峰值显存、时延、吞吐和失败恢复。
