# Method 概览：Qwen3-VL-8B Slot-Memory Delta-Rule State-TTT

> 用途：讲解材料，帮助快速理解本仓库的方法设计。规范性定义以
> [README](../README.md)、[ARCHITECTURE](../ARCHITECTURE.md)、[DECISIONS](../DECISIONS.md)
> 为准（规范版本 `state_ttt_qwen3vl8b_slot_memory_delta_v1`，schema 14）。
> 本文对应 2026-08 的主线实现。

## 1. 问题与动机

任务是 **SVCBench 长视频计数问答**：给定一段长视频和一个带时间点的问题
（"到 query_time 为止，X 出现/发生了多少次"），模型要给出精确计数和自然语言答案。

直接把长视频塞进 VLM 上下文有两个根本困难：

1. **上下文容量**：2 FPS 下长视频远超可承受的视觉 token 数；
2. **计数需要跨时间的状态**，而不是一次性注意力——"数到第几个"本质上是
   一个随时间演化、需要去重和阶段判断的**在线状态**，LLM 的算术又不可靠。

我们的回答是 **State-TTT（Test-Time Training 的状态化变体）**：

- 视频按 chunk **流式**进入模型，每个 chunk 处理完即丢弃视觉 token；
- 跨 chunk 的信息由两类状态承载：
  - 一块 **可微的在线记忆 `M`**（delta-rule 联想记忆，测试时在线写入）；
  - 一组 **确定性的结构化状态**（State Bank / Identity Bank / FSM），
    由确定性 Reader 负责精确计数；
- "**怎么写记忆**"不是手工规则，而是由外层（meta）训练学出来的——
  这就是 TTT 的含义：推理时权重不变，但记忆 `M` 在每个视频内被在线更新，
  更新算子的参数由训练习得。

一句话概括方法：

> 在 Qwen3-VL-8B 的视觉通路中插入一个带 per-video delta-rule 记忆的
> Fast Adapter 和一条结构化状态路径；先用 A2 训练"静态"完整系统，
> 再用 A5 Meta-TTT（K=8 截断）学习在线写入几何与强度；
> 推理时按 chunk 严格因果地在线更新记忆，精确计数由确定性 Reader 输出，
> 自然语言答案由 Qwen 生成。

## 2. 总体数据流

```text
─── Support 路（观察视频，维护状态）────────────────────────────
video chunk（8/16 帧动态块）
  -> Query Encoder + 写入前 Bank 语义上下文 b
  -> Qwen ViT + Main Visual Merger            （Qwen 原生，DeepStack 不动）
  -> Fast Adapter：K = P_in(RMSNorm(X)) + P_C(LayerNorm(Query + b))
       core = f_W0(K) + α ⊙ (K Mᵀ_{t-1})       # 读上一步记忆，FP32
  -> Spatial Slot Encoder（2-stage Slot Attention，32 slots）
  -> Temporal Causal Encoder（6 层因果 Transformer）
  -> O1/O2/E1/E2 观测头（瞬时计数/身份去重/事件概率/事件阶段）
  -> State Bank + Identity Bank 硬提交（detach，不进反传）
  -> 从已提交 slot state 派生 (k, v, η) 写入对
  -> 闭式并行 delta-rule 写入 -> M_t           # 只影响下一个 chunk

─── Query 路（回答问题）───────────────────────────────────────
question + query_time
  -> Query Encoder（4 层，512 维）+ operator/时间窗路由
  -> 写前 Retrieval History -> Semantic Projector -> Retriever
  -> 16-token State Resampler -> 送入 Qwen
  -> 写后 aggregate/Confirmed Bank -> Deterministic Reader（精确计数唯一真值）
  -> Qwen prefill/generate（自然语言答案）
```

分工原则：**Reader 管数、Qwen 管说**。确定性 Reader 是精确计数的唯一所有者，
LLM 不覆盖 Reader 的算术。

## 3. 架构组件

### 3.1 插入点：不改造 Qwen

- 基座 `Qwen/Qwen3-VL-8B-Instruct`。Fast Adapter 插在 **Main Visual Merger
  输出之后、video `masked_scatter` 之前**——即视觉特征进入 LLM 序列前的
  最后一个纯视觉位置。
- DeepStack（layer 8/16/24 注入）保持 Qwen 原始路径，完全不接 adapter。
  这样对基座的侵入只有一个残差式的插入点，A2 行为可被精确保留（见 3.3 零初始化）。

### 3.2 Fast Adapter 与 slot memory（方法核心）

Adapter 是 4096→768→4096 的瓶颈结构。静态核 `W0₁/W0₂` 是普通可训练参数；
在线部分是一块 **零初始化的 delta-rule 记忆 `M ∈ ℝ^{768×768}`**
（约 59 万个瞬态 FP32 值，per-video，不进 checkpoint）。

**读取**（对每个 Main Merger token）：

```text
b_{t-1} = attention_pool(Query, 写入前 Bank 的有效语义记录)
K_t     = P_in(RMSNorm(X_t)) + P_C(LayerNorm(Query + b_{t-1}))
core    = f_W0(K_t) + α ⊙ (K_t Mᵀ_{t-1})     # α 为 per-channel 读取门
```

token key `K_t` 由"当前视觉 + 当前问题 + 已有语义状态"共同构造——
记忆的检索几何是任务感知的。

**写入**（每个 Support chunk 硬提交之后，从已提交 soft state 派生至多 32 条写入对）：

```text
k_i = normalize( Σ_t softmax_t(⟨W_k·sg(s_i), K_t⟩/√768) · K_t )   # 对活 token keys 的 probe attention
v_i = normalize( W_v · sg(s_i) )
η_i = η_max · σ(gate([sg(s_i); sg(c_i)]))，Ση > 1 时重归一（并审计）
M_t = (1-β)·M_{t-1} + Σᵢ η_i (v_i - M_{t-1} k_i) k_iᵀ             # 闭式并行 delta rule
```

关键性质：

- **没有 inner 优化器**。写入是闭式并行 delta rule，不存在 inner loss、
  inner 学习率或 inner 梯度裁剪。keys 单位化且 Ση ≤ 1（chunk 预算），
  每个 chunk 的 BPTT 雅可比因子算子范数 ≤ 1-β，**K 步截断图天然收缩**，
  meta 梯度不会爆。
- **写入是纯归档**。slot state `s_i` 与置信度 `c_i` 在 probe 和 value 两条
  路径上全程 detach（`sg`）：写入不许把 encoder 表征拉向记忆；encoder 只从
  读取路径和 Query loss 拿梯度。而 token keys `K_t` 保持活梯度，所以外层
  训练通过 `P_in/P_C` 学到的是**记忆的 key 几何**。
- **门控全部由外层学习**：η（per-slot 写入强度）、α（per-channel 读取门）、
  β（标量遗忘门）都是 Outer 参数，属于 `associative` 优化器组
  （`P_C` + memory 接口约 1.23M 参数）。
- **零初始化是结构性约束**：`M=0` 时前向与纯静态前向 bitwise 相同，
  记忆无法被训练挪用为跨视频的静态容量；每个视频从零开始，天然隔离。
- 数值上，fast 核与记忆读写固定 FP32，残差输出边界再转回模型 dtype；
  无有效 slot 或非有限 payload 的 chunk fail-closed 跳过写入并计数。

### 3.3 结构化状态路径

| 组件 | 配置 | 职责 |
|---|---|---|
| Spatial Slot Encoder | 2-stage Slot Attention，32 active / 64 max slots | 把 chunk 特征分解为对象 slot |
| Temporal Encoder | 6 层因果 Transformer，hidden 768，64 tubelet cache | 跨帧因果建模，overlap 只回放已见位置 |
| O1 / O2 / E1 / E2 | 观测头 | 瞬时计数 / 身份向量、去重证据与 relevance / 事件概率 / 事件与阶段状态 |
| State Bank + Identity Bank | 硬提交，detach | 结构化事实：聚合状态、Confirmed 记录、append-only retrieval history |
| Retriever + Semantic Projector | 只读**写前** history | Query 的语义检索（768D detached source 重投影为 512D key） |
| Deterministic Reader | 读**写后** aggregate/Confirmed Bank | 精确计数、record IDs、算术与审计字段的唯一所有者 |

硬状态（Bank/FSM）在提交前 detach，不参与反向传播；Reader 的算术不进
optimizer。软路径（可微）与硬路径（确定性）的分离贯穿全部设计。

O2 头额外输出 query 条件化的 relevance 分数 `r_i = σ(⟨identity_i, W·q_target⟩)`
（schema-14 新增），回答"该对象是否问题所指的类别"；分数随 Identity Bank 生命周期
EMA 传递并落入 Confirmed 记录。Reader 侧对应一个 O2 relevance 闸门，当前冻结为
audit_only——只在审计里记录"若按阈值过滤会剩几条"，不改变计数；切 enforce
需要已标定阈值，属未来显式契约变更。

## 4. 训练方法：A2 → A5 Warmup → A5 Main

### 4.1 A2：先把"静态"系统训到位

- 全量解冻 Qwen（ViT、Merger、36 层 Decoder）、全部状态模块和 W0；
- `P_C` 冻结、记忆写入不可达（前向不绑定 memory state，**等价于 M=0**）——
  即先训练一个"没有在线记忆"的完整系统；
- 目标为 `L_state + L_answer`；loss balance 用唯一算法 `ema_answer_ref`：
  loss EMA 先对齐 Answer 尺度，再按 `q_target/q_operator/q_time` 激活面的
  梯度 RMS EMA 平衡 Task/Operator/Retrieval/Time 四项弱监督，
  辅助组合计不超过 Answer 的 40%（`official_weak_balance.group_weight`）。
- O2-Unique 的官方弱监督计数训练在与 Reader 同构的软路径上（A2/A5 共用同一
  builder）。动机：此前该监督压在池化计数头（训练路），而推理时 Reader 数的是
  Identity Bank confirmed 记录数（推理路），两路分叉使 `o2.identity` 拿不到
  任务梯度、身份几何没有分离压力。软去重目标为：预测 = sg(query chunk 硬提交
  **之前**的 confirmed 基数) + 当前 chunk 的 relevance 门控可微软新颖数——
  identity 与 confirmed 原型及同 chunk 更早槽的余弦经 logsigmoid 在 log 域累积
  （τ 对齐 `match_threshold=0.8`，温度 0.1 为固定目标形状）。`o2.identity` 与
  relevance 头由此获得任务梯度；O2-Gain 保持池化计数回归。

### 4.2 A5 Warmup：256 步只学"记忆机制"

从完整 A2 checkpoint 初始化后，先跑一个独立的 256-step Memory/State Warmup：

- Qwen、W0、RMSNorm/P_in/P_out **bitwise 冻结**且不进 optimizer；
- 只训练 `P_C`、memory 接口（W_k/W_v/η gate/α/β）和四个 state 组；
- 成功后只原子保存一个小型 **handoff bundle**（非 Qwen 张量 + 来源 hash），
  按 G1–G5 释放门判定（episode 内召回、读取显著性、`episode_zero` 反事实、
  管路健康、基础设施，见 [production-a2-a5](./production-a2-a5.md)）。

动机：直接进 Main 时记忆是全新参数，梯度信号弱且和 Qwen 微调纠缠；
Warmup 让写入机制先在冻结底座上"活起来"，再进主训练。

### 4.3 A5 Main：K=8 截断的 Meta-TTT

Main 重新加载 A2 checkpoint、严格叠加 bundle、恢复部分 Qwen 解冻，训 4 epoch：

- 每个 episode 是"1..8 个 Support chunk + 若干条官方 Query 标签"的
  因果 segment 序列；Support 逐 chunk 观察并闭式写入 `M`，
  **写入本身没有任何 loss**；
- **唯一的外层目标是 Query 的 Answer/State loss**。每个 segment 对 Query loss
  执行 backward，deferred VJP（vector-Jacobian product）把 Query 梯度沿
  收缩的线性递推回传 K=8 步，到达 memory 接口、token keys（P_in/P_C）
  与 M_{t-1}；每 8 个 Support 截断 meta 图（detach 保值成为新 leaf）；
- 每条 Query 的 FP32 memory cotangent 按联合范数裁剪
  （`a5.query_meta_gradient.max_norm`，当前 10.0）后在 segment 内求和；
  episode 末外层 AdamW 单次 step；
- 直觉："这次写入让后面的问题答得更好吗"——写入的几何（W_k/W_v）与
  强度（η/β）完全由未来 Query 的表现反向塑造，这正是 meta-learning 的
  learning-to-memorize。

**监控写入是否真的在学**（schema-12 的教训）：`associative` 组是混合来源的
——α、P_C 从读取路径拿梯度，而 W_k/W_v/η gate/β 只能经 deferred VJP 拿梯度，
所以组级梯度范数说明不了写入机制是否被训练。系统对该组挂两个
`GradientProbe`（`memory_write` 9 个只写张量 / `memory_read`），报各自与
W0 组的**范数比值**：schema-12 的失效模式正是写入梯度比 W0 低四个数量级
而非恰为零。

对照与诊断：`no_write`（记忆恒零、接口冻结）作为 NoWrite 对照；
`episode_zero`（精确 M=0）与 `segment_start` 作为无梯度反事实参照。

### 4.4 为什么是闭式 delta rule 而不是 inner SGD

早期方案（schema-12）用 episode 内 functional SGD 最小化一个 inner TTT loss
来更新 fast weights。真实运行诊断发现该机制被**结构性架空**：写入路径拿到的
meta 梯度比 W0 低四个数量级，外层辅助 loss 还被目标尺度漂移污染。
schema-13 的重构把 inner 优化整体删除，换成上面的闭式 delta rule——
更新算子线性、收缩、可并行，meta 梯度路径短且范数可控，
"学什么写什么"完全交给外层 Query 监督。

## 5. 在线推理

每个视频一个独立生命周期：

```text
load checkpoint -> reset video (M=0) -> causal observe -> online memory write
  -> prepare answer -> prefill/generate -> release
```

严格因果与隔离约束（训练与推理一致执行）：

- 答案、count、occurrence_times、counting_type/subtype **禁止**进入
  Support/Query 模型输入（防标签泄漏）；
- `query_time` 之后的帧在进模型前裁剪，不参与状态更新与回答；
- 当前 chunk 读 `M_{t-1}`，写出的 `M_t` 只从下一 chunk 生效（无回溯）;
- generate 阶段只读：不重跑状态路径，不改 memory/Bank/FSM/temporal state;
- 每个视频 reset/release，异常路径同样 release；跨视频不共享任何 storage。

Query 默认读取 `[0, query_time]` 完整因果前缀（2 FPS、最多 256 帧）；
正式入口为 `ttt-svcbench-infer`。

## 6. 讲解时的常见问题

**Q：这和普通 TTT（如 TTT-Linear/Titans 一类）的区别？**
写入对象不是隐藏层 fast weights 而是一块显式联想记忆 `M`，更新是闭式
delta rule 而非 inner 梯度步；且系统是"可微记忆 + 确定性状态机"双轨——
精确计数不指望可微部分，由 Reader 保证。

**Q：为什么记忆每个视频清零，不做跨视频记忆？**
任务是 per-video 计数问答，跨视频记忆只会带来泄漏面；零初始化同时是
训练时的结构性约束（`M=0` 前向 bitwise 等于静态前向），保证 A2 能力
在每个 episode 起点被精确保留。

**Q：训练开销大吗？**
Support 写入是本地张量运算、无 collective、无独立 backward；每 rank 每
episode 的 backward 数固定为 `query_count + segment_count`。多 Query 逐个
forward/backward 释放激活。真实吞吐/显存/收敛结论只由 H200 运行记录证明
（tiny/CPU 测试只证接口与因果性，这是仓库的固定验证边界）。

**Q：模型规模的变化？**
基座 8B 不变。新增：Fast Adapter（4096→768→4096 静态核）、
memory 接口约 1.23M 参数、状态路径各模块；瞬态 `M` 为 768×768 FP32，
不进 checkpoint。
