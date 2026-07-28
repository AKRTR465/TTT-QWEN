# Qwen3-VL-8B Bank-conditioned Associative State-TTT 架构

> 规范版本：state_ttt_qwen3vl8b_bank_associative_v1
> 配置 schema：10（旧 schema 9 及更早配置不自动升级）
> 修订日期：2026-07-28
> 状态：A2/A5 TRAINING MAINLINE IMPLEMENTED；ONLINE INFERENCE WIRED

## 1. 固定目标

在不改造 Qwen3-VL-8B DeepStack 的前提下，为长视频流增加可在线更新的视觉 fast state 和确定性结构化状态。系统只保留正式 A2→A5 训练与在线推理，不保留阶段 gate、standalone trainer 或 synthetic ablation runtime。

核心不变量：

- base model：`Qwen/Qwen3-VL-8B-Instruct`；
- Fast Adapter：4096→768→4096，两块在线矩阵共 1,179,648 参数；另有 `P_C:512→768`
  和 `P_V:4096→768` 两个 Associative 投影；
- 插入点：Main Visual Merger 输出之后、video `masked_scatter` 之前；
- DeepStack indexes：8、16、24，保持 Qwen 原路径；
- 更新顺序：Query/写入前 Bank context → observe with Wt → hard-state commit →
  `L_assoc` functional update → next chunk uses Wt+1；
- hard state 不参与反向传播，Reader 算术不进入 optimizer。

## 2. 数据流

```text
video chunk
  -> Query Encoder + write-before Bank semantic context
  -> Qwen ViT + Main Merger
  -> Fast Adapter(Wt, K_t)
  -> Spatial Slot Encoder
  -> Temporal Causal Encoder
  -> O1/O2/E1/E2
  -> State Bank + Identity Bank hard write
  -> masked FP32 visual association loss L_assoc
  -> functional SGD -> Wt+1

question + query_time
  -> Query Encoder + operator/time routing
  -> pre-write Retrieval History -> Semantic Projector -> Retriever
  -> Retriever -> 16-token State Resampler
  -> post-write aggregate/Confirmed Bank -> Deterministic Reader
  -> Qwen answer prefill/generation
```

每个 `TrajectoryRuntimeState` 是单视频唯一状态源，持有 fast weights、optimizer state、slot/cache、
E1/E2、State/Identity Bank 和 Reader audit；不持有关联 context 或其他关联临时中间量。

## 3. 状态模型

### 3.1 Fast Adapter

输入输出维度为 4096，bottleneck 为 768。对每个 Main Merger token：

```text
b_{t-1} = attention_pool(Query, present & valid Bank semantics)
K_t = P_in(RMSNorm(X_t)) + P_C(LayerNorm(Query + b_{t-1}))
V_t = P_V(RMSNorm(stopgrad(X_t)))
L_assoc = masked_fp32_mse(f_Wt(K_t), V_t)
```

空 Bank 的 `b` 固定为零；硬 payload、count、phase、timestamp 不进入池化。`P_C` 零初始化，
`P_V` 在 A2→A5 初始化时复制 `P_in`，使新 A5 初始前向保持 W0 数值兼容。W0 属于 checkpoint
和 Outer optimizer；Wt 是 per-video 临时状态，不注册为 parameter/buffer，不进入 checkpoint。
在线更新必须有限、版本单调且 storage 隔离。

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
- schema-10 是当前唯一正式训练契约；旧 schema checkpoint 不自动迁移，A5 只能从显式允许的旧 A2
  权重初始化，并重新创建 Associative 投影、optimizer、scheduler、RNG 与 runtime state；
- `P_C/P_V` 冻结、Inner SGD 不可达；
- Query outer loss 正式使用 `ema_answer_ref`：先用一步滞后的 loss EMA 对齐 Answer，
  再用 `q_target/q_operator/q_time` 激活梯度 RMS EMA 平衡 Task、Operator、Retrieval、Time；
  四槽固定且辅助组限制为 Answer 的至多 30%；
- loss/gradient EMA 随同阶段 resume 恢复，A2 初始化 A5 时重置；不提供其他 loss-balance
  模式；
- 状态参数按 shared、task、router-time、retrieval 四组独立裁剪，四组 RSS 预算保持与旧
  state 单组相同；
- 多卡动态分支使用零值 graph anchor 保持梯度集合一致，单卡不构造 anchor。

### A5

- 从完整 A2 safetensors checkpoint 初始化；
- Support 只使用 Bank-conditioned visual association loss：Bank 语义影响当前 key，Fast
  Adapter 输出随后影响 soft object selection 和唯一一次 hard Bank/FSM write；
- `L_assoc` 是 masked FP32 MSE，只对原始视觉输入停止梯度，不使用 Answer/State 标签，也不
  读取硬 payload；它只用于 functional SGD，不以 auxiliary 权重加入 Outer loss；
- Support 不设人工数值上限；
- 每 8 个 Support 截断二阶图并重锚 W0；
- 每个 segment 只对 Query Answer/State Outer loss 执行 backward，deferred VJP 将 Query 梯度
  传回 `W_t`、W0、`P_C/P_V` 和慢模块；episode 末由 Outer optimizer 单次 step；
- Inner SGD 使用配置中的 fast update LR（当前 `1e-4`）；Associative projection 的 Outer LR
  当前为 `5e-5`，不注册可学习步长控制器；Associative 组更新预算与 Qwen/W0 严格对齐；
- `static_w0` 保留为 NoUpdate 对照；counterfactual 仅作为 Meta-TTT 的无梯度因果诊断，
  不参与优化。

## 5. 在线推理主线

生命周期固定为：

```text
load checkpoint
  -> reset video
  -> causal observe
  -> online TTT update
  -> prepare answer
  -> prefill/generate
  -> release
```

约束：

- query_time 之后帧在进入模型前裁剪；
- updater 固定使用配置中的 Inner SGD LR；纯未来 chunk 不触发状态观察或更新；
- updater 只允许修改当前视频的 fast/optimizer state；Associative context 是本次调用的
  短生命周期临时对象，不跨请求、重试或异常路径残留；
- 更新后的 Wt 不得回溯影响当前 chunk；
- generation 不重跑视频状态路径、不修改 Bank/FSM/Fast；
- 正常、异常和中断均 release。

审计级别：

- `off`：不持久化状态快照；
- `boundary`：记录 owner、版本、对象/存储身份和 Tensor version，不复制内容到 CPU；
- `full`：仅在 reset、update、generate、release 边界增加内容 SHA-256。

## 6. Checkpoint 与分布式

正式 checkpoint 必须完整匹配 schema-10 模型 key，支持单文件和 sharded safetensors，并包含
`associative_contract_version`。禁止保存或加载 Wt、optimizer runtime、Bank、cache、FSM 和
Associative 临时 context；旧 A5 checkpoint 缺少该版本 buffer 时直接拒绝。

A2/A5 sampler 必须保持四卡任务或 segment parity。非有限 loss/gradient 必须 warning/skip，不能产生部分参数更新。ZeRO、BF16、显存和性能是否可接受只由真实 H200 记录决定。

## 7. 验证边界

代码测试验证 shape、dtype、因果性、泄漏、梯度、state_dict、checkpoint 和 lifecycle。未执行真实 8B/H200 时，不得从 tiny/CPU 测试推导训练收敛、吞吐或科学收益。
