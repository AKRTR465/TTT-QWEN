# SVCBench State-TTT 模型分析

> 讲解用文档。规范性定义以 [README](../README.md)、[ARCHITECTURE](../ARCHITECTURE.md)、
> [DECISIONS](../DECISIONS.md) 为准（规范版本 `state_ttt_qwen3vl8b_slot_memory_delta_v1`，schema 13）。

## 1. 相比 Qwen3-VL-8B 原始模型，多了哪几个大模块

基座 `Qwen/Qwen3-VL-8B-Instruct` 本身**结构不变**（ViT 27 层、Merger、36 层 LLM、
DeepStack 8/16/24 注入路径全部保持原样）。新增部分挂在一个插入点上：
**Main Visual Merger 输出之后、video `masked_scatter` 之前**。

一共 10 个新增模块，按所在通路分三组：

| # | 模块 | 通路 | 有无参数 | 一句话作用 |
|---|---|---|---|---|
| 1 | **Fast Adapter + delta-rule 记忆 `M`** | 视觉 | 有（W0 1.18M + P_in/P_out 6.3M + P_C 0.39M + 记忆接口 1.23M） | 唯一的在线可写记忆；跨 chunk 携带视觉信息 |
| 2 | **Spatial Slot Encoder** | Support | 有 | 把 chunk 特征拆成 32 个对象 slot |
| 3 | **Temporal Causal Encoder** | Support | 有（48.4M） | 跨帧因果建模，维护 64 tubelet KV cache |
| 4 | **Observation Heads O1/O2/E1/E2** | Support | 有（2.63M / 2.50M / 9.72M / 7.29M） | 从 slot 读出计数、身份、事件、阶段四类观测 |
| 5 | **State Bank（含 Semantic Projector）** | Support | 有（1.32M） | 存结构化事实与 append-only 检索历史 |
| 6 | **Identity Bank** | Support | 无（确定性） | 对象身份去重，决定"这是不是新的一个" |
| 7 | **Query Encoder（+ Operator Router + Time Resolver）** | Query | 有 | 编码问题，路由算子类型与时间窗 |
| 8 | **Semantic Retriever** | Query | 无（零参数打分器） | 从写前检索历史里选出相关记录 |
| 9 | **State Resampler** | Query | 有（14.72M） | 把检索结果压成 16 个 state token 喂给 Qwen |
| 10 | **Deterministic Reader** | Query | 无（确定性） | 精确计数的唯一所有者，算术不进反传 |

另有一处非模块改动：**Input Composer** 向 tokenizer 追加 5 个 special token
（`<|state_start|>` / `<|state_pad|>` / `<|state_end|>` / `<|number_start|>` / `<|number_end|>`，
词表 151,669 → 151,674），负责把 state token 与 Reader 输出的数字 ID 拼进 Qwen prefill 序列。

## 2. 逐模块简述

### 2.1 视觉通路

**Fast Adapter + delta-rule 记忆 `M`** 是方法核心，详见第 4 节。

### 2.2 结构化状态路径（Support 路：看视频、维护状态）

- **Spatial Slot Encoder** — 2-stage Slot Attention，32 active / 64 max slots，hidden 768。
  把一个 chunk 的视觉特征分解成对象级 slot 表示。
- **Temporal Causal Encoder** — 6 层因果 Transformer，hidden 768、12 heads、64 tubelet
  layerwise KV cache。跨帧建模，overlap replay 只允许回放已见位置。
- **Observation Heads** — O1 瞬时计数 / O2 身份向量与去重证据 / E1 事件概率 / E2 事件与阶段
  状态。四个头把连续 slot 表示转成可提交的观测量。
- **State Bank / Identity Bank** — 结构化事实存储与身份去重，详见第 5 节。

硬状态（Bank / FSM）在提交前 detach，不参与反向传播。

### 2.3 Query 路（回答问题）

Query Encoder（含算子路由与时间窗解析）与 Semantic Retriever 见第 6 节；
State Resampler 与 Deterministic Reader 见第 7 节。分工原则是
**Reader 管数、Qwen 管说**：LLM 不覆盖 Reader 的算术。

## 3. Qwen 本体没有动的部分

- ViT、Main Merger、36 层 Decoder 的结构与权重路径；
- DeepStack（layer 8/16/24 注入）完全走原生路径，不接 adapter；
- 原生 video `masked_scatter` 占位符逻辑保持不变，本项目只额外 scatter state 占位符。

对基座的侵入只有一个残差式插入点，加 5 个新增 special token 的 embedding 行。

## 4. Fast Adapter 与 per-video delta-rule 记忆

方法核心。它是系统里**唯一在测试时会被写入的东西**——权重全程不变，变的是每个视频
自己的一块记忆 `M`；"怎么写"由外层训练学得。

参数与状态严格分家：

| | 内容 | 归属 |
|---|---|---|
| 静态参数 | RMSNorm、`P_in` 4096→768、`P_C` 512→768、`W0₁`/`W0₂` 768×768（无 bias、xavier）、`P_out` 768→4096 | checkpoint + Outer optimizer |
| 记忆接口 | `W_k`、`W_v`（768×768+bias）、η gate MLP（769→64→1）、`α`（768）、`β`（标量），约 1.23M | checkpoint + Outer optimizer（`associative` 组） |
| 在线状态 | `M ∈ ℝ^{768×768}` FP32，589,824 个瞬态值 | **不注册为 parameter/buffer，不进 checkpoint，每个视频归零** |

### 4.1 读路径

```text
b    = attention_pool(q_target, 写入前 Bank 的 present∧valid 语义记录)   # 无参数
K    = P_in(RMSNorm(X)) + P_C(LayerNorm(q_target + b))                   # token key [B,N,768]
core = W0₂ · SiLU(W0₁ K)  +  α ⊙ (K Mᵀ)                                  # FP32，显式关闭 autocast
out  = X + 0.1 · P_out(core)                                             # residual_scale=0.1
```

- token key `K` 由**当前视觉 + 当前问题 + 已有语义状态**三者共同构造——记忆的检索几何
  是任务感知的，不是纯视觉的；
- `b` 是对写入前 Bank 语义记录的无参数注意力池化（`softmax(qᵀe/√512)`），空 Bank 时为零；
- `α` 是 per-channel 读取门（初值 0.1）；无效 token 的残差置零；
- **`M = 0` 时 readout 恒为零，前向与纯静态前向 bitwise 相同**——A2 的能力在每个 episode
  起点被精确保留；
- 前向审计记录 `readout_share_norm`（记忆读出占残差的比例）、`memory_norm`、输入/残差范数。

### 4.2 写路径（每个 Support chunk 硬提交后一次）

```text
k_i = normalize( Σ_t softmax_t(⟨W_k·sg(s_i), K_t⟩/√768) · K_t )   # 对本 chunk 活 token key 的 probe attention
v_i = normalize( W_v · sg(s_i) )
η_i = 0.25 · σ(gate([sg(s_i); sg(c_i)])) · mask_i                 # Ση > 1 则整行重归一并标记审计
β   = 0.1 · σ(β_raw)
M_t = (1−β)·M_{t−1} + Σ_i η_i (v_i − M_{t−1}k_i) k_iᵀ             # 闭式并行 delta rule
```

三个设计点：

1. **写入是纯归档**。slot state `s_i` 与置信度 `c_i` 在 probe、value、gate 三条路上
   **全部 stop-gradient**。写入不许把 encoder 表征拉向记忆；encoder 只从读取路径和
   Query loss 拿梯度。
2. **唯一带梯度的写入输入是 token key `K_t`**。所以外层学到的是**记忆的 key 几何**
   （经 `P_in` / `P_C`），以及写入强度（`W_k`/`W_v`/η gate/β）。
3. **收缩性是结构保证的**。keys 单位化、`Ση ≤ 1`（chunk 预算）、`β > 0`，于是每个 chunk
   的 BPTT 雅可比算子范数 ≤ `1−β`，K=8 截断图天然收缩，meta 梯度不会爆。
   **没有 inner 优化器、inner loss、inner 学习率、inner 梯度裁剪**——整条 schema-12 的
   inner-SGD 机制被删除，换成这条线性闭式更新。

### 4.3 数值与失败处理

- 核运算与记忆读写固定 FP32（`torch.autocast(enabled=False)` 显式关闭），只在残差输出
  边界转回模型 dtype；
- 两种 fail-closed 跳过：`no_valid_slot`（本 chunk 无有效 slot）、`nonfinite_key_value`
  （payload 或写后 `M` 非有限）。跳过时不写、`skip_count+1`、`M` 原值进入下一代；
- 每次写入产出审计：写前/写后 recall 余弦、`write_norm`、`memory_norm`、`eta_sum`、
  是否触发 η 重归一。写前写后余弦之差就是"这次写入到底记住了没有"的直接证据。

### 4.4 两种梯度模式

| 模式 | 场景 | 行为 |
|---|---|---|
| `online_leaf` | 在线推理 / A2 | `no_grad` 写入，新 `M` 是 leaf |
| `meta_linear_recurrence` | A5 Meta-TTT | `enable_grad`，`M` 带图；每 8 个 Support 调 `truncate_memory_state` 截断（detach 保值成新 leaf，审计要求 drift 严格为 0） |

零初始化是结构性约束而不只是初始化选择：记忆无法被外层挪用成跨视频的静态容量，
Query 时刻记忆里的任何东西**必然是本视频写进去的**。

## 5. Bank：结构化状态与内嵌 FSM

定位：`M` 负责**可微地带信息**，Bank 负责**确定性地存事实**。精确计数只走 Bank，
不指望可微部分。整条硬路径是 **概率 → 事实 → 数** 三段：

```text
观测头 O1/O2/E1/E2        soft，只出概率（禁止阈值判断与整数累加）
      ↓ detach
State Bank 硬提交          ← FSM 在这一层内部，不是独立模块
   ├ E1/E2  阈值 + 状态跃迁 → 就地更新 payload        = FSM
   ├ O1     recompute_from_full_slot_state 聚合
   └ O2     交 Identity Bank 决策，回写 candidate / confirmed
      ↓（Query 时）
Deterministic Reader       读 payload → 精确整数（第 7 节）
```

两个 Bank 的分工：

| | State Bank | Identity Bank |
|---|---|---|
| 管什么 | 四个头的 typed 记录 + 检索历史 | O2 对象身份的生命周期 |
| 回答什么 | "当前状态是什么、历史上发生过什么" | "这个对象是不是新的" |
| 参数 | 只有 Semantic Projector（1.32M） | 无 |

分界线是硬的：State Bank 对 O2 只做 generic CRUD，身份匹配/晋升/淘汰全部归 Identity
Bank（配置写死 `o2_p9_policy: generic_crud_only_p10_owns_lifecycle`）。

### 5.1 State Bank 技术栈

| 技术点 | 做法 |
|---|---|
| 状态管理 | 函数式不可变：每次写返回新的 runtime state + `version+1`，全程 `@torch.no_grad`，从不 in-place；snapshot / restore 免费 |
| 记录组织 | Typed record —— O1/E1/E2 各一条聚合记录（functional replace 更新），O2 每对象一条 |
| 写入语义 | 幂等 + 因果闸门：`position_id` 不前进则只写审计不改状态；`record_id` 轨迹内单调、永不复用 |
| 检索面 | 512 维 L2 单位化 detached embedding；时间点与时间区间二选一 |
| 检索历史 | `[4, 512, 768]` 按头分槽张量环，存**投影前**的 768D source，Query 时用当前 Projector 重投影 → retrieval 梯度能更新 Projector，但不回传历史 Support encoder |
| 唯一可学习件 | Semantic Projector：head-type embedding + LayerNorm + 768→1024 SiLU→512 + L2 归一，1.32M 参数 |

### 5.2 内嵌的两台计数 FSM

FSM 没有独立的类，也没有独立存储：转移逻辑是 `StructuredStateBank.update_e1` /
`update_e2`，状态量就是记录 payload 的字段（`E1Payload.active/armed/cooldown_until`、
`E2Payload.phase`）。上游观测头只出概率，**所有阈值判断与整数累加都发生在这一层**。

**E1 事件 FSM**（`eventness_hysteresis_completion_transition`），状态量 `(active, armed, cooldown)`：

| 当前 | 条件 | 转移 |
|---|---|---|
| armed | `eventness ≥ 0.7` | → active（cooldown 内只记 hit） |
| active | `completion ≥ 0.7` 且 `transition ≥ 0.7` | **计数 +1**，→ inactive 且 disarmed |
| active | `eventness ≤ 0.3` | 放弃候选，→ armed |
| disarmed | `eventness ≤ 0.3` | → armed（唯一重新武装路径，防同一事件反复计数） |

**E2 区间 FSM**（`phase_gated_single_transition_per_position`）：
`inactive → active → end_candidate → completed → inactive`。每步跃迁要求
**概率阈值与 argmax 阶段证据同时同意**，只满足其一记 `conflict` 且不动状态；
只有 `end_candidate → completed` 才 `completed_count += 1` 并 append 区间。

共同守卫：position 严格连续 +1、timestamp 严格递增，重复 position 只记 duplicate，
**每 position 至多一次跃迁**。E1/E2 历史各存 512 条，超出即 evict 并置位
（Reader 遇到需要那段历史的窗口会判无效）。O1 不是 FSM，走
`recompute_from_full_slot_state`。

### 5.3 Identity Bank（第三台状态机）

| 技术点 | 做法 |
|---|---|
| 生命周期 | 两级：Candidate（易失）→ Confirmed（权威） |
| 匹配 | 256 维单位原型内积，阈值 0.8；near-tie（margin 1e-6）判冲突、不猜；同 chunk 内强制一对一 |
| 晋升 | 连续两个 committed position 都可靠才确认（断档即重置连击），压掉偶发闪现 |
| 老化 | 未匹配候选 TTL 8 递减 + 低置信剪枝，淘汰时同步 invalidate State Bank 侧记录 |
| 原型更新 | EMA 0.9 |
| 存储 | CPU FP32 分块存储（初始/增量 256）精确检索、不用 ANN；GPU 256 条 bf16 LRU hot cache，仅作加速、非真值 |
| 检索可见性 | 候选不可检索，只有 confirmed 进入 Query 检索面 |

### 5.4 共同约束

- 写入前全程 detach，硬状态不参与反向传播；
- Bank 运行时状态不进 checkpoint（只有 Semantic Projector 进），per-video reset / release；
- 读取分两个时刻：Retriever 读**写入前**快照（因果、防自我泄漏），Reader 读**写入后**状态。

## 6. Query 路：从问题到检索

一句问题要变成三样东西——**查什么（`q_target`）、怎么算（算子）、算哪一段（时间窗）**，
然后据此检索。

```text
question + query_time
  -> Query Embedding Encoder ── q_target ─→ Retriever / Resampler / Adapter 的 key 上下文
                            ├─ q_operator ─→ Operator Router ─→ 硬算子
                            └─ q_time ─────→ Time Resolver  ─→ TimeWindow
  -> Retriever（读写入前历史）-> 选中记录 -> 第 7 节
```

输入契约先卡死泄漏面：`query_tokens.py` 只 tokenize**完整问题**，
答案、count、occurrence_times、counting_type 一律禁止进入模型输入。

### 6.1 Query Embedding Encoder

4 层 Pre-LN **双向** Transformer（hidden 768、12 heads、FFN 3072、正弦位置编码），
输入是 Qwen 的 4096 维问题 token embedding。池化用 learned attention
（`tanh` 投影 + 打分 + softmax，**权重固定 FP32** ——BF16 下归一化残差会触发结构不变量
并让 DDP rank 失同步）。

池化向量后接**三个独立的头**，各自 L2 归一到 512 维：

| 输出 | 去向 | 梯度来源 |
|---|---|---|
| `q_target` | Retriever 打分、Resampler query 注入、Adapter 的 key 上下文 | retrieval loss + answer loss |
| `q_operator` | Operator Router | 算子分类 loss |
| `q_time` | Time Window Resolver（配合 `token_states`） | 时间 loss（mode CE + pointer） |

三头共享主干、只在最后一层分叉；分开设是为了让三种语义各有独立监督出口
（也对应 outer loss 按这三个激活面分别算梯度 RMS EMA 做平衡）。

### 6.2 Operator Router

9 个可学习原型（8 个算子 + `unsupported`）+ 一个可学习温度：
`logits = normalize(q_operator) @ normalize(prototypes)ᵀ / T`，softmax 取 argmax。
**不做关键词匹配**（文件头明令禁止 keyword task routing）。

| 算子 | 头 | 默认时间模式 |
|---|---|---|
| `o1-snap` / `o1-delta` | O1 | now / recent |
| `o2-unique` / `o2-gain` | O2 | history / recent |
| `e1-action` / `e1-transit` | E1 | history |
| `e2-periodic` / `e2-episode` | E2 | history |

置信门 fail-closed：阈值当前标注 `CALIBRATION_REQUIRED`（值为 `None`），
一旦开门而阈值未标定，**全部落到 `unsupported`** 而不是放行。

### 6.3 Time Window Resolver

两条腿：神经的 mode 分类（4 类 MLP，吃 `q_time`）+ 两个 pointer head 指向问题里的数字
token（吃逐 token 的 `token_states`）；确定性的窗口构造：

| 模式 | 窗口 |
|---|---|
| `now` | `[—, query_time]`，不接受数字 |
| `history` | `[0, query_time]`，不接受数字 |
| `recent` | 需要恰好 1 个正时长，`[query_time − d, query_time]` |
| `explicit_range` | 需要恰好 2 个端点，且 `start ≤ end ≤ query_time` |

**神经部分只做选择，不做算术**：窗口端点来自问题文本解析出的数值，再逐条校验——
解析失败、显式时间值与文本不一致、数量不对、区间反向、越过 `query_time`，
任一项直接判 `invalid` / `unsupported`，绝不猜窗口（文件头：`no guessed time windows`）。
没有显式时间就退回算子默认模式，并在 `reason` 标 `operator_default`。

### 6.4 Semantic Retriever

零参数精确打分器。只读**当前 Query 写入前**的检索历史快照，用当前 Semantic Projector
把 768D source 重投影成 512D key，与 `q_target` 算余弦。

过滤顺序固定：`invalid → retrieval_ineligible → future → outside_window → below_similarity`
（阈值 0.35，闭区间比较）；选择顺序 `score_desc → record_id_asc`。
**`top_k = None`，不做截断**——Reader 的算术要求"看到全部相关记录"，
截断会静默改变计数。

注意算子与时间窗在这里只用于过滤（选头、卡因果窗），它们在上游已被 detach 成
Python 枚举与 float，**不带梯度**；`q_target` 是唯一以连续形式参与打分的向量，
retrieval loss 也只沿它回传。

## 7. Reader

定位：**精确计数的唯一所有者**，零参数。文件头的禁止项写死了它不做什么——不做 Top-K
截断、不做神经计数回归、不替换真值、不检索、不改 Bank、不生成自然语言。

两个入口：`read(retrieval)` 读 Retriever 结果；`read_bank(...)` 绕过语义检索与余弦/时间
预过滤，直接读写入后的 aggregate / Confirmed Bank（主线走这条）。

### 7.1 算术表：8 个算子 → 确定性表达式

| 算子 | 数据来源 | 表达式 |
|---|---|---|
| `o1-snap` | O1 聚合 | `current_visible_count` |
| `o1-delta` | O1 聚合 | `current_visible_count − baseline_count`（baseline 未初始化 → `invalid`） |
| `o2-unique` | Confirmed | `first_seen ≤ query_time` 的确认身份数 |
| `o2-gain` | Confirmed | `first_seen` 落在闭窗内的确认身份数 |
| `e1-action` / `e1-transit` | E1 聚合 | history 模式取 FSM 累计 `event_count`；否则数闭窗内完成时刻 |
| `e2-periodic` / `e2-episode` | E2 聚合 | 落在闭窗内的 completed interval 结束时刻数 |

全是整数计数与加减，**不做任何概率阈值判断**——阈值都在 Bank 的 FSM 里定完了，
Reader 只剩闭区间时间比较与整数运算。

### 7.2 输出契约与校验

- **四态 fail-closed**：`ok`（必须有整数，只有 `o1-delta` 允许负）/ `empty`（count=0 且无
  record）/ `unsupported`、`invalid`（禁止携带 count）；number token 与 `exact_count`
  同时在或同时不在。
- **校验闸门**（任一不过即 `invalid`）：算子与 head 不匹配、记录无效、`record_id` 重复、
  聚合算子拿到多条记录、O2 候选或未来身份漏到 Reader、检索 OK 但时间窗非 OK、
  E1 窗口需要已被 evict 的历史——**宁可报无效，不给近似值**。
- **数字 token**：输出 token ID 而非文本，canonical ASCII 十进制 + encode/decode/re-encode
  三重校验，tokenizer 用 manifest SHA-256 钉死；`audit_number_tokens()` 可在交付前复检。
- **全量审计**：每条结果附算子、检索状态、`n_state`/`n_retrieved`、`bank_version`、时间窗、
  每个操作数与 `arithmetic` 名，可逐条复算。

耦合是单向的：Bank/FSM 的状态决定 Reader 的结果（历史被 evict 会让相关窗口判 `invalid`，
未完成区间只作审计操作数不计数）；反向 Reader 不改 Bank，算术不进 optimizer。

### 7.3 与 State Resampler 的分界

| | State Resampler | Deterministic Reader |
|---|---|---|
| 性质 | 可微，进 optimizer（14.72M） | 确定性，零参数 |
| 结构 | 3 层 Perceiver/Q-Former，16 个可学习 query 叠加 `q_target`，512 → 4096 | 整数算术 + 校验 + tokenizer 审计 |
| 职责 | 让 LLM **感知**状态 | 给出**数** |

## 8. Loss 设计

### 8.1 顶层只有两项

```python
outer = answer_after + state_after
```

**记忆写入路径没有任何直接 loss**——`W_k/W_v/η/β` 的唯一梯度来源是 Query 的 deferred VJP。

### 8.2 Answer loss

shift-by-one token CE(`-100` 忽略),FP32 归约。另产出四个**只作指标、不进 loss** 的量:
teacher-forced token 准确率、数字 token 准确率、整行 exact match、
**Reader 精确计数准确率**(int64 相等比较——硬计数不可导,不进梯度)。

### 8.3 State loss:七个来源,权重冻死为 1

```python
state = task + operator + retrieval + time
task  = o1 + o2 + e1 + e2
```

| 项 | 形式 |
|---|---|
| O1 | 逐 slot dense BCE |
| O2 | identity cosine + score BCE |
| E1 | eventness / completion / transition 三路 dense BCE |
| E2 | event BCE + **phase CE**(soft-FSM 代理,不是硬 FSM 输入) |
| operator | 9 类 CE |
| retrieval | 相似度 logits 上的 BCE |
| time | mode CE + start/end span CE |

两条硬约束:**每行只能监督一个观测头**(row_indices 不相交);四个权重在构造器里强制
`== 1.0`,相对权重完全交给平衡器,不留静态旋钮。

### 8.4 `LossTerm`:区分"损失为 0"和"没有标签"

每项不是标量,而是 `value + per_row + row_valid_mask + valid_counts + mask_counts +
skip_reasons`,并强制"有效行不带理由、无效行必须带理由"。10 个 `LossSkipReason` 逐行
记录缺失原因。

这不是洁癖:分布式下某 rank 某行缺标签,若按 0 参与均值会拉低全局尺度,进而污染下面的
EMA 缩放。这套 counts 保证**分母只数真有标签的行**。

### 8.5 平衡器 `ema_answer_ref`:两级缩放 + 一道封顶

四个弱监督项各乘一个 scale:

```python
loss_scale = EMA(answer) / EMA(term)                        # clamp [0.001, 20]
grad_scale = geomean(EMA(grad_rms) over active) / EMA(term_grad_rms)   # clamp [0.1, 10]
scale      = clamp(loss_scale × grad_scale, [0.001, 20])

aux   = Σ(scale × term_mean) / 4
guard = min(1, max(EMA(answer), 0.1) / aux)
state = group_weight × guard × aux          # group_weight = 0.4
```

- 第一级把各项拉到 Answer 的**损失量级**,第二级拉到彼此**梯度量级**的几何均值;
- 梯度量在**激活面**上测(`q_target` / `q_operator` / `q_time`),不是参数梯度;
  `task` 与 `retrieval` 共用 `q_target`;
- `guard` 保证辅助组 ≤ **40%** 的 Answer EMA,主目标永不被盖过。

所有均值都是**全局归约后**再算(一次 all-reduce 打包 12 元统计向量),否则各 rank 会用
不同的 scale 导致梯度不一致。EMA 状态进 checkpoint(schema 7),同阶段 resume 恢复,
A2 初始化 A5 时重置。
