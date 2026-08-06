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

## Slot-memory 几何与门的已知缺陷（2026-08-06）

4 卡 A5 warmup 32 步 smoke 报出 `key_pairwise_cosine_mean = 0.999987`（S=32）。对单位
范数行，该均值本身即强制 `Σ‖k_i − m‖² = (S−1)(1−ρ̄) = 4.03e−4`，故每个 slot 都落在
质心 0.0201 之内，写入的参与比有效秩为 1.0000252，而设计容量是 32 对 (k,v)。同时
`Ση = 1` 精确成立，沿写入方向的保留系数为 `−β = −0.01`：每次写入抹掉该方向先前内容
的 99%，`eta_chunk_budget = 1.0` 恰落在 delta 规则的湮灭点上。

- **根因是凸池化被用了两次**：`fast_ttt.py` 的 token key 池化与 `state_encoder.py` 的
  slot 池化权重都和为 1，故任何 token-无关分量以系数 1 穿过，`memory_key_probe` 对其
  杠杆恒为零。已在 CPU 上用未训练编码器实测：256 个近乎正交的输入 token（两两余弦
  8e−6）产出 32 个两两余弦 0.999780 的 slot，跨种子稳定 —— 该坍缩是架构性的，不是
  训练或数据造成的。中心化后余弦 −0.0257（理论地板 −0.0323）说明残差信息尚存。
- **唯一有效杠杆是 `slot_codes` 的尺度**，不是任何 bias：把编码器全部 bias 归零后
  余弦仍为 0.999748；`slot_codes × 20` 则降至 0.957335（`1−cos` 改善 190 倍）。
  `_initial_slots` 的三项里有两项（`shared_slot_seed`、`query_condition`）是 slot-无关的。
- **G1 既未按规格实现、按规格实现也是空的**：文档定义为 episode 内末 support 减首
  support，代码发的是 episode 级扁平均值，首/末 support 的量在代码库中不存在；若按
  规格实现，首 support 的 `M=0` 使 `cos ≡ 0`，于是"末−首 ≥ 0.15"对任何会写入的系统
  自动通过。其绝对腿在完全坍缩时取最大值，因此**部分地是它本应认证的多样性的反向
  指标**。本轮只修报数伪影（见下），不设新阈值。
- **G2 结构上分不清常量读出与选择性读出**：`readout_share` 是行范数之比，一个纯常量
  偏置也能落在 [0.01, 0.3] 带内。需要补一条 across-token 方差条款。
- **G3 是唯一能检测该失效的门，而它正在失败**：正例率 31/96（精确二项检验
  p = 6.7e−4），descent cosine 均值 −0.000754，95% CI [−0.0058, +0.0043]，以 11.6 倍
  排除阈值 0.05。注意 `episode_zero` 反事实比较 `M=0` 与 `M=M_T`，**不依赖参数训练**，
  故该结果不能用"欠训练"解释。通过 G1+G2 而失败 G3 正是本坍缩的预期签名，
  `docs/production-a2-a5.md` §9 的决策规则在此会导向错误方向。
- **`memory/readout_target_cosine` 被结构性稀释**：它是所有已接受写入上的扁平均值，
  而每个视频的首次写入落在零记忆上，其召回为零向量，masked cosine 精确贡献 0.0 且
  仍计入分母。首末两步可精确重建：`(4×0.733053 + 7×0.970620)/11 = 0.884232`、
  `(8×0.860320 + 2×0.987775)/10 = 0.885811`。剔除该结构性零后真实稳态为
  **0.973 → 0.984**。现已并行发出 `a5/memory/readout_target_cosine_recall_only`，
  按写入前记忆代次筛选而非按 `值 == 0.0` 筛选，旧指标保留以维持可比性。
- **`slots_written` 无信息量**：`slot_valid_mask` 恒为全 True，overflow 记账在生产路径
  恒为 0，故它恒等于 `32 × 写入次数`。
- **`state_encoder.py` refinement 循环内重新注入 `query_condition` 在数学上是惰性的**：
  该项沿 slot 轴恒定，而其后的 softmax 沿 slot 轴取，故被精确抵消；query 条件化真正
  起作用的位置是 `_initial_slots`。但移除它并非逐位中性（FP32 下扰动 ~1.7e−6，因为
  softmax 前的量级改变），故**刻意保留并加注释**，不做清理。
- **η 预算护栏已补上**：`active_slots × eta_gate_init ≤ eta_chunk_budget` 现在在
  config 装载期强制（跨 `spatial_encoder` 与 `fast_memory` 两节，故无法放在
  `FastMemoryConfig` 上——这正是漂移未被发现的原因）。`eta_gate_init` 由 0.05 降为
  0.02（32 × 0.02 = 0.64 < 1.0）。`eta_chunk_budget` 保持 1.0：它被冻结 contract 钉住，
  改动需要 4 处联动并有触发 schema 升版、进而作废全部既有 warmup bundle 的风险。
  该改动改变训练动力学，改动前后的 bundle 不可比。
- **但仅降 `eta_gate_init` 修不了 G4**：H200 重归一率仍是 100%。该护栏的算术依赖
  "gate 数据项为零"，而 gate 读原始 slot 状态（范数 ≈9）、silu 非奇、slot 近乎共线，
  于是 `W_out · silu(W_h · s)` 是**全部 slot 共享的一个 DC 偏置**：不跨 slot 平均掉、
  每 seed 只抽一次、跨 chunk 近乎恒定、随 slot 范数**线性增长**（本地实测 |c| 跨度
  0.31 → 2.38 对应范数 8.7 → 70）。用随机近正交 slot 测不出来（per-slot 抽样相互抵消），
  这正是本地全绿而 H200 每个 chunk 都重归一的原因。旧初始化下 `Ση` 实测在
  0.049–0.617 摆动（意图 0.64，**13 倍跨度**，即写入可能比设计弱 13 倍），H200 抽到
  相反方向而超预算，把 `Ση` 精确钉在 1.0——`delta` 规则的湮灭点，保留系数 `−β = −0.01`。
  **修法**：`memory_eta_gate_output.weight` 改为**零初始化**，logit 恒等于 bias，
  `Ση = active_slots × eta_gate_init = 0.64` 在任意 seed、任意 slot 尺度下精确成立；
  `W_out` 仍照常训练（`∂logit/∂W_out = hidden ≠ 0`），仅钉初始化，`memory_eta_gate_hidden`
  只在第 0 步无梯度、第 1 步起恢复。副作用：同时把递推抬离湮灭点（保留系数
  `1 − β − Ση = 0.35`）。回归测试按 seed × slot 尺度扫共线 slot 并断言 `Ση` 精确等于
  乘积（`rel=1e-6`），原先的 `rel=0.2` 正是掩盖该缺陷的松弛。
  **曾考虑并否决**：`eta_max_per_slot` 0.25 → 0.0625（只缩天花板，`Ση` 仍是 seed 抽签）；
  → 1/32（给出结构性保证，但让 gate 永远无法把写入强度集中到单个 slot，代价大于收益）。
- **`memory_eta_gate_output.bias` 会被 checkpoint 静默覆盖**：它由 `eta_gate_init` 在
  构造期导出且是 state-dict 键。已核实 A2 checkpoint（1193 个张量）不含任何
  `memory_*` / `eta` / `p_context` / `alpha` / `beta_raw` / `slot_codes` 键，只含
  `shared_slot_seed`，故从 A2 初始化时配置生效；但 warmup bundle 的 allowlist 扫入
  每一个非 Qwen 持久参数，因此 bundle **必然**携带它，A5 resume 或 bundle 初始化会
  恢复旧值。`reset_associative_projections` 只归零 `p_context`，不重导出该 bias。
- **写入批次现在拒绝零范数的有效 slot**：上界检查对退化无话可说，而一个有效但为零的
  行在下游全程被接受（delta 为零、update 为零、仍报 `did_write` 且 `write_norm = 0.0`、
  cosine 审计返回 0.0 而非 NaN），属于静默的容量损失。减去 slot 均值正是产生它的一种
  途径，故改为 fail-closed。
- **记忆 key 空间改为 token 中心化（contract revision 3→4）**：slot 侧中心化把实现容量
  从 1.001 提到 4.213 后，key PR 饱和于 token key 自身的共享均值天花板（两两余弦
  0.46–0.50 ⇒ PR ≈ 4.2），而 `key_centered_pairwise` 探针显示去掉跨 slot 共享分量后
  同一批 key 落在 −0.027 —— **剩余的容量赤字整个就是那个均值**。现在 `forward` 里把
  视觉分量在 chunk 内有效 token 上中心化后再加回 `p_context` 广播，得到
  `memory_keys`，**只**喂读出 bmm 与 `intermediates.keys`；喂 W0 静态核的 `projected`
  保持不变（就地中心化会改变零记忆必须逐位复现的 A2 静态函数）。softmax 沿 token 轴
  平移不变 ⇒ token 选择可证明不变；c 被原样加回 ⇒ 唯一的跨 chunk key 变化源与唯一的
  Bank→写 key 梯度通道保留；M=0 时读出精确为零 ⇒ A2 逐位不变量保持。经三方对抗审计
  （两个独立仿真收敛于 recall_only 机械预测 0.27–0.50、key PR ~25–30）。**预登记的
  预期位移，不是回退**：`readout_share` 会下降（读出失去共享均值共模项 —— G2 需要
  重校准）；G1 的 episode 内上升腿会读负（漂移下健康召回的正常形态）；
  `token_key_pairwise` 语义变为对中心化 key 的度量，且 A5 中 P_in 冻结 ⇒ 它此后的
  任何上行都是 ‖c‖ 增长（再投毒监视器）。write-only 变体被否决：存储 key 中心化而读
  查询不中心化会让每次检索携带共享均值偏置项，还留下一条"意外的各向异性均值通道"让
  模型绕过有意的 c 通道读全局信号。`ASSOCIATIVE_CONTRACT_VERSION` 与
  `_WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION` 同步 3→4（新增测试钉住两者相等），
  revision 3 的 warmup bundle 以清晰的 provenance mismatch 拒绝加载；contract 家族
  字符串 `bank_conditioned_slot_memory_v3` 是四处联动的 config 字面量且其命名的机制
  未变，刻意不动，config schema 不升版。
- **recall_only（M1）在当前目标下不可训练，这是结构事实**：它是 `torch.no_grad` 下的
  审计量，没有任何损失项奖励召回保真度；answer CE 经 α(K Mᵀ) 到 M 的间接压力被
  Reader 文本插入进一步饿死（Reader 对时 CE≈0 ⇒ dL/dM≈0），state/operator、time、
  retrieval 三头无到 M 的路径。因此它的训练平坦不构成"几何无效"的证据；当前 0.18 =
  支撑调度的内容重叠（40/8/4，相邻 chunk ≥4s）× 几何乘数。若未来要让它可训练，唯一
  杠杆是显式辅助召回损失，且必须带 PR 地板 kill criteria 防 value-collapse 捷径
  （把 value 写塌成同一向量即可刷高召回）。
