# State-TTT retrieval-history 固定决策

## 当前主线

仓库只维护：

1. 正式 A2 全量状态训练；
2. A2 checkpoint 初始化的 A5 K=8 Meta-TTT；
3. per-video 在线 State-TTT 推理。

历史阶段 gate、standalone trainer、synthetic ablation harness 和 evidence bundle 不属于运行时，也不进入 wheel。

## 模型

- Qwen3-VL-8B Main Visual Merger 后插入 4096→768→4096 Fast Adapter。
- DeepStack 保持原始路径，不接入 Fast Adapter。
- 在线只写入一块零初始化的 768x768 delta-rule memory M；W0₁/W0₂ 是 checkpointed
  静态核参数，memory 接口（key probe、value projection、eta gate、read gate α、
  forget gate β）是 checkpointed Outer 参数，随 `associative` 组训练。
- 写入是闭式并行 delta rule（Ση ≤ 1 保证收缩），没有 inner 优化器、inner 学习率或
  inner 梯度裁剪；schema-12 的 functional SGD、`L_assoc` cosine 目标与 1e-4 频率固定
  contract 已整体删除（2026-07-30，slot-memory refactor）。
- Spatial Encoder 固定为两阶段 Slot Attention；Temporal Encoder 固定为 6 层因果 Transformer。
- Query Encoder 固定为 4 层、512 维；State Resampler 固定输出 16 个 4096 维 token。
- Deterministic Reader 是精确计数唯一真值源，LLM 不覆盖 Reader 算术。
- O2 relevance 头固定为乘性 query 交互 `σ(⟨identity, W·q_target⟩)`（schema-14 新增，
  Linear 512→256）；分数随 Identity Bank 生命周期以 `prototype_ema` 同衰减传递并
  落入 Confirmed 列存。Reader 闸门冻结为 audit_only + 无阈值；切 enforce 需要已标定
  阈值并构成显式契约变更，且该变更必须同时解决三笔已知欠账：
  （a）训练目标的 confirmed 基数随门控对齐——现为未门控全量，历史 confirmed 记录的
  relevance 拿不到梯度，enforce 下训练路与 Reader 计数会在基数项重新分叉；
  （b）Reader 的 `arithmetic` 标识与 `matched_first_seen_count` 如实反映门控，
  否则审计误归因；
  （c）门控排空全部记录时"OK + count 0"的枯竭语义显式敲定（错误的门会把
  "不知道"变成"0"）。

## 训练

- 正式流程为 A2→256-step Memory/State Warmup→A5 Main。
- A2 全量解冻 Qwen、状态路径和 W0，冻结 `P_C`，memory 写入不可达。
- A5 Support 写入直接归档本 chunk 的 slot (key, value) 对，无 inner loss；K=8 截断
  meta 图（detach 保值，无 W0 重锚）；Query Answer/State loss 是唯一 Outer 目标。
- Query memory cotangent 的裁剪上限由 `a5.query_meta_gradient.max_norm` 配置（当前
  10.0），不再是冻结常量；counterfactual 参照为 `episode_zero`（精确 M=0）与
  `segment_start`，每 rank 可审计多条 Query。
- NoWrite 对照改名 `no_write`（旧名 `static_w0` 报错拒绝）。
- O2-Unique 官方弱监督计数为软去重目标：sg(写前 confirmed 基数) + 当前 chunk 的
  relevance 门控可微软新颖数（log 域累积，τ 对齐 match_threshold=0.8、温度 0.1 为
  固定目标形状）；O2-Gain 保持池化计数回归；dedup 上下文缺失整体回退池化路径。
  基数快照必须取自 query chunk 硬提交之前。
- Warmup 完全冻结 Qwen、W0 与 RMSNorm/P_in/P_out；只训练 P_C、memory 接口和四个 state
  组，并仅保存带来源 hash 的非 Qwen handoff bundle；A5 Main 重新加载 A2 后叠加 bundle，
  恢复部分 Qwen 解冻，4 epoch 只保存 final checkpoint。
- A1/A3/A4 与 full-graph Meta-TTT 已从生产实现和配置中删除。
- graph anchor 只服务真实多卡动态分支，单卡不启用。
- Outer checkpoint 完整保存模型/optimizer/scheduler/RNG，排除所有临时 runtime state。

## 因果与泄漏

- Support/Query runtime 禁止答案、count、occurrence_times、counting_type 和 counting_subtype。
- query_time 后帧必须在模型边界前裁剪。
- 当前 chunk 使用 M_{t-1}，delta-rule 写入的结果只供下一 chunk 使用。
- hard state、Bank 与 overlap snapshot 必须 detach；不同视频不得共享 storage 或 owner。
- clean test 不得进入训练或校准。

## 推理

- 每个视频独立 reset、observe/update、answer、release。
- generation 期间 memory、Bank、FSM、cache 不变。
- retry 只允许在相同因果状态上执行。
- 默认 audit 为 `boundary`；只有 `full` 计算 Tensor 内容 hash。
- 首版 generation 固定 greedy、`num_beams=1`、`do_sample=false`、`max_new_tokens=16`。
- 推理 checkpoint 只接受严格匹配的 safetensors。

## 验证声明

- 阈值在训练折或独立校准集冻结前，不作正式评估声明；该策略不属于运行配置字段。
- tiny/CPU 测试是工程证据，不是 8B 收敛、性能或科学收益证据。
- 真实 8B/H200 结果必须记录模型 revision/hash、BF16、峰值显存、时延、吞吐和失败恢复。
