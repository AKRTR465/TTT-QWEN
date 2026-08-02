# Qwen3-VL-8B Slot-Memory Delta-Rule State-TTT 架构

> 规范版本：state_ttt_qwen3vl8b_slot_memory_delta_v1
> 配置 schema：13（schema 12 及更早的 A5 checkpoint/bundle 不自动迁移）
> 修订日期：2026-07-30
> 状态：A2/A5 TRAINING MAINLINE IMPLEMENTED；ONLINE INFERENCE WIRED

## 1. 固定目标

在不改造 Qwen3-VL-8B DeepStack 的前提下，为长视频流增加可在线写入的视觉 slot memory 和确定性结构化状态。系统只保留正式 A2→A5 训练与在线推理，不保留阶段 gate、standalone trainer 或 synthetic ablation runtime。

核心不变量：

- base model：`Qwen/Qwen3-VL-8B-Instruct`；
- Fast Adapter：4096→768→4096 静态核（W0₁/W0₂ 为普通 Outer 参数）；per-video 在线状态是
  单块零初始化 delta-rule memory `M ∈ ℝ^{768×768}`（589,824 个瞬态 FP32 值）；另有
  `P_C:512→768` Associative context 投影和 memory 接口参数（`memory_key_probe`、
  `memory_value_projection`、eta gate MLP、`memory_alpha`、`memory_beta_raw`，约 1.23M）；
- 插入点：Main Visual Merger 输出之后、video `masked_scatter` 之前；
- DeepStack indexes：8、16、24，保持 Qwen 原路径；
- 更新顺序：Query/写入前 Bank context → observe with M_{t-1} → hard-state commit →
  slot-memory parallel delta-rule write → next chunk uses M_t；
- 写入规则为闭式并行 delta rule：`M_t = (1-β)·M_{t-1} + Σᵢ ηᵢ(vᵢ - M_{t-1}kᵢ)kᵢᵀ`，
  keys 单位化、Ση ≤ 1（chunk budget），因此每个 chunk 的 BPTT 雅可比因子算子范数 ≤ 1-β，
  K-step 截断图天然收缩；不存在 inner 优化器、inner 学习率或 inner 梯度裁剪；
- η/α/β 由 Outer loop 学习：η 是 per-slot sigmoid gate（上限 `eta_max_per_slot`，chunk
  预算 1.0，超预算重归一并审计），α 是 per-channel 读取门，β 是标量遗忘门（上限
  `forget_beta_max`）；
- hard state 不参与反向传播，Reader 算术不进入 optimizer。

## 2. 数据流

```text
video chunk
  -> Query Encoder + write-before Bank semantic context
  -> Qwen ViT + Main Merger
  -> Fast Adapter: K = P_in(RMSNorm(X)) + P_C(LayerNorm(Query + b))
       core = f_W0(K) + alpha ⊙ (K Mᵀ_{t-1})          # FP32 memory read
  -> Spatial Slot Encoder
  -> Temporal Causal Encoder
  -> O1/O2/E1/E2
  -> State Bank + Identity Bank hard write
  -> slot write payload (k_i, v_i, eta_i) from committed soft state
  -> parallel delta-rule write -> M_t

question + query_time
  -> Query Encoder + operator/time routing
  -> pre-write Retrieval History -> Semantic Projector -> Retriever
  -> Retriever -> 16-token State Resampler
  -> post-write aggregate/Confirmed Bank -> Deterministic Reader
  -> Qwen answer prefill/generation
```

每个 `TrajectoryRuntimeState` 是单视频唯一状态源，持有 memory state、slot/cache、
E1/E2、State/Identity Bank 和 Reader audit；不持有 inner 优化器状态、关联 context 或其他关联临时中间量。

## 3. 状态模型

### 3.1 Fast Adapter 与 slot memory

输入输出维度为 4096，bottleneck 为 768。对每个 Main Merger token：

```text
b_{t-1} = attention_pool(Query, present & valid Bank semantics)
K_t = P_in(RMSNorm(X_t)) + P_C(LayerNorm(Query + b_{t-1}))
core = f_W0(K_t) + alpha ⊙ (K_t Mᵀ_{t-1})
```

每个 Support chunk hard commit 之后，adapter 从已提交的 soft state 派生至多 32 条
(key, value, eta) 写入对：

```text
k_i = normalize( Σ_t softmax_t(⟨W_k·sg(s_i), K_t⟩/√768) · K_t )    # probe attention over live token keys
v_i = normalize( W_v · sg(s_i) )
eta_i = eta_max · σ(gate([sg(s_i); sg(c_i)]))，Σeta > 1 时重归一（审计标记）
M_t = (1-β)·M_{t-1} + Σᵢ eta_i (v_i - M_{t-1} k_i) k_iᵀ
```

slot state `s_i` 与 confidence `c_i` 在 probe 输入和 value 两条路径上都全程 detach：写入是
纯归档，不得把 encoder 表征拉向 memory；encoder 梯度只经由读取路径和 Query loss 回传。
token keys `K_t` 保持活梯度，因此 Outer loop 通过 P_in/P_C 学习 memory 的 key 几何。
空 Bank 的 `b` 固定为零；硬 payload、count、phase、timestamp 不进入任何写入路径。
无有效 slot 或非有限 payload 的 chunk 跳过写入（`no_valid_slot` / `nonfinite_key_value`），
跳过是 fail-closed 的且计入 skip 计数。

W0 与 memory 接口参数属于 checkpoint 和 Outer optimizer；`M` 是 per-video FP32 master
临时状态，每个视频起点严格为零，不注册为 parameter/buffer，不进入 checkpoint。零初始化
是结构性约束：memory 无法被 Outer loop 挪用为跨视频静态容量，`M=0` 前向与纯静态前向
bitwise 相同（A2 行为在每个 episode 起点被精确保留）。fast 核、memory 读写均固定为
FP32，残差输出边界再转回模型 dtype。

### 3.2 Spatial 与 Temporal

Spatial Encoder 使用 2-stage Slot Attention，32 active slots、64 最大容量。Temporal Encoder 为 6 层因果 Transformer，hidden 768、12 heads、64 tubelet cache；overlap replay 只允许已见位置。

### 3.3 Observation heads

- O1：瞬时计数；
- O2：身份向量与去重证据；
- E1：事件概率；
- E2：事件与阶段状态。

hard path 在提交前 detach；Identity Bank 只依据模型输出和因果 overlap 更新。

### 3.4 Query、Retriever 与 Reader

Query Encoder 为 4 层、输出 512 维，并产生 operator prototype 路由与时间窗口。Semantic Retriever 只读取当前 Query 写入前的 append-only retrieval history，并在 Query graph 中用现有 SemanticProjector 将 detached 768D source 重投影为 512D key；因此 retrieval loss 可同时更新 q_target 与 Projector，但不会回传到历史 Support encoder。Reader 不经过 semantic threshold 或 retrieval history，直接读取当前 Query 写入后的 aggregate/Confirmed Bank，并作为唯一精确计数所有者输出状态、record IDs、算术结果和审计字段。

## 4. 训练主线

### A2

- 全量解冻 Qwen、状态模块与 W0；
- schema-13 是当前唯一正式训练契约；schema-12 及更早的 A5 checkpoint/bundle 明确拒绝，
  A5 只能从显式允许的 A2 权重初始化，并全新初始化 memory 接口、optimizer、scheduler、
  RNG 与 runtime state；
- `P_C` 冻结、memory 写入不可达（A2 前向不绑定 memory state，等价于 `M=0`）；
- Query outer loss 正式使用 `ema_answer_ref`：先用一步滞后的 loss EMA 对齐 Answer，
  再用 `q_target/q_operator/q_time` 激活梯度 RMS EMA 平衡 Task、Operator、Retrieval、Time；
  四槽固定且辅助组限制为 Answer 的至多 40%（`official_weak_balance.group_weight`）；
- O2-Unique 行的官方弱监督计数使用软去重目标（A2/A5 共用同一 builder）：预测 =
  写前 Identity Bank confirmed 基数（detach）+ 当前 chunk 的可微软新颖数——identity 与
  confirmed 原型及同 chunk 更早槽的余弦经 logsigmoid 在 log 域累积，阈值对齐
  `match_threshold=0.8`、温度 0.1 为固定目标形状，不进配置。`o2.identity` 由此获得
  任务梯度；O2-Gain 保持池化计数头回归；dedup 上下文缺失时整体回退池化路径。
  基数快照必须取自 query chunk 硬提交之前，否则当前槽会与自身匹配、软新颖数退化为零；
- loss/gradient EMA 随同阶段 resume 恢复，A2 初始化 A5 时重置；不提供其他 loss-balance
  模式；
- 状态参数按 shared、task、router-time、retrieval 四组独立裁剪，四组 RSS 预算保持与旧
  state 单组相同；
- 多卡动态分支使用零值 graph anchor 保持梯度集合一致，单卡不构造 anchor。

### A5

- 先独立执行 256-step Memory/State Warmup：重新加载完整 A2 checkpoint，Qwen、W0 和
  RMSNorm/P_in/P_out bitwise 冻结且不进入 optimizer；只训练 P_C、memory 接口与全部状态
  模块，并只使用现有 Query Outer objective；
- Warmup 成功后仅原子保存小型 handoff bundle。Main 再加载原 A2 checkpoint，严格校验并叠加
  bundle，重置 loss-balancer EMA，创建全新 optimizer/scheduler，恢复部分 Qwen 解冻；
- Support 写入不含任何 inner loss：memory 直接归档本 chunk 的 (key, value) 对，K=8 截断的
  meta 梯度沿收缩线性递推回传到 `W_k/W_v/gate/β`、token keys（P_in/P_C）与 M_{t-1}；
- Bank 语义影响当前 key，Fast Adapter 输出随后影响 soft object selection 和唯一一次
  hard Bank/FSM write；
- Support 不设人工数值上限；
- 每 8 个 Support 截断 meta 图（`truncate_memory_state`：detach 后保值成为新 leaf，
  无 W0 直通重锚——memory 零初始化后没有需要保留的 W0 血统）；
- 每个 segment 只对 Query Answer/State Outer loss 执行 backward，deferred VJP 将 Query 梯度
  传回 `M`、memory 接口、`P_C` 和慢模块；episode 末由 Outer optimizer 单次 step；
- 每个 Query 的 memory cotangent 在 unscale 后按联合范数独立裁剪到
  `a5.query_meta_gradient.max_norm`（可配置，当前 10.0），同一 segment 内将裁剪结果求和；
  Query 对 Qwen、State 和其他 Outer 参数的直接梯度不参与此裁剪；
- memory 接口参数并入既有 `associative` optimizer 组（Outer LR 当前 `5e-5`），组预算与
  Qwen/W0 严格对齐；eta gate 本身就是合法的可学习写入强度控制器；
- 该组是混合来源的，因此组级梯度范数无法证明写入机制在被训练：`p_context` 与
  `memory_alpha` 同时从读取路径拿梯度，而 `W_k/W_v/eta gate/β` 只能经 Query deferred VJP
  拿梯度。`OuterGradientController` 因此对该组挂两个 `GradientProbe`——`memory_write`
  （9 个只写张量）与 `memory_read`（`memory_alpha`），各自报裁剪前的跨 rank 范数及其
  与 `w0` 组的比值（`outer_grad/probe/*`）。判据是比值而非绝对值：schema-12 的失效是
  写入梯度比 W0 低四个数量级而非恰为零。probe 集合必须精确划分 `associative` 组，
  新增参数漏挂 probe 会在测试期失败；


- `no_write` 保留为 NoWrite 对照（memory 恒为零、memory 接口参数冻结；旧名 `static_w0`
  已删除并在入口报错指明新名）；counterfactual 仅作为 Meta-TTT 的无梯度因果诊断
  （参照 `episode_zero` 即精确 `M=0` 与 `segment_start`，每 rank 可审计多条 Query），
  不参与优化。

## 5. 在线推理主线

生命周期固定为：

```text
load checkpoint
  -> reset video (M = 0)
  -> causal observe
  -> online memory write
  -> prepare answer
  -> prefill/generate
  -> release
```

约束：

- query_time 之后帧在进入模型前裁剪；
- updater 在 no-grad 下执行 `prepare_write` + 闭式写入并发布新的 leaf `M`；纯未来 chunk
  不触发状态观察或写入；
- updater 只允许修改当前视频的 memory state；Associative context 是本次调用的
  短生命周期临时对象，不跨请求、重试或异常路径残留；
- 写入后的 M_t 不得回溯影响当前 chunk；
- generation 不重跑视频状态路径、不修改 Bank/FSM/memory；
- 正常、异常和中断均 release。

审计级别：

- `off`：不持久化状态快照；
- `boundary`：记录 owner、版本、对象/存储身份和 Tensor version，不复制内容到 CPU；
- `full`：仅在 reset、update、generate、release 边界增加内容 SHA-256（含 `M`）。

## 6. Checkpoint 与分布式

正式 checkpoint 必须完整匹配 schema-13 模型 key，支持单文件和 sharded safetensors，并包含
`memory_contract_version`。Warmup bundle 只含 allowlist 中的非 Qwen persistent tensor
（memory 接口参数随 adapter 注册自动进入），并绑定 A2/config/data/code hash；禁止保存
`M`、Bank、cache、FSM 和 Associative 临时 context。

唯一历史权重兼容是私有 `a2_to_a5_memory_v1` profile：只允许旧 A2 缺少新增的 `p_context`、
memory 接口参数（`memory_key_probe`、`memory_value_projection`、eta gate、`memory_alpha`、
`memory_beta_raw`）与 `memory_contract_version`，包含已删除的 `predictor`、`p_value` 与旧
`associative_contract_version` buffer；其余 missing/unexpected key 一律拒绝。加载后立即重置
Associative 状态，且该 profile 不得用于 same-stage resume。旧 A5 checkpoint 仍必须按
schema-13/current contract 严格恢复，不推断、不迁移。

A2/A5 sampler 必须保持配置的 4/8 rank 任务或 segment parity；每 rank 每 episode 的 backward 数固定为
`query_count + segment_count`，写入本身是本地张量运算、不含 collective。非有限
loss/gradient 必须 warning/skip，不能产生部分参数更新。Warmup 的 Qwen bitwise 审计只
覆盖 parameter 与 persistent buffer；被排除的 non-persistent buffer 名单随审计 JSON 一并
持久化。ZeRO、BF16、显存和性能是否可接受只由真实 H200 记录决定。

schema-13 的冻结常量分两层强制：`ProjectConfig._FROZEN_CONTRACT` 在 `load_config()` 处拒绝
其覆盖的路径漂移；observation head、State Bank、Spatial/Temporal Encoder 与 fast memory 的
字段由各模块 `_validate_*_config`/构造器在 build 时拒绝。后者是唯一能拦住
`model_copy(update=...)` 绕过 pydantic validator 的路径，因此这些字段只在模块构建期报错——
多卡启动时意味着在 distributed init 之后。schema-13 不含 `paths` 配置块，四个环境变量名直接从
`os.environ` 读取。

## 7. 验证边界

代码测试验证 shape、dtype、因果性、泄漏、梯度、state_dict、checkpoint 和 lifecycle。未执行真实 8B/H200 时，不得从 tiny/CPU 测试推导训练收敛、吞吐或科学收益。
