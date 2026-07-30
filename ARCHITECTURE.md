# Qwen3-VL-8B State-Write Associative State-TTT 架构

> 规范版本：state_ttt_qwen3vl8b_state_write_associative_v3
> 配置 schema：12（schema 11 A5 不自动迁移）
> 修订日期：2026-07-30
> 状态：A2/A5 TRAINING MAINLINE IMPLEMENTED；ONLINE INFERENCE WIRED

## 1. 固定目标

在不改造 Qwen3-VL-8B DeepStack 的前提下，为长视频流增加可在线更新的视觉 fast state 和确定性结构化状态。系统只保留正式 A2→A5 训练与在线推理，不保留阶段 gate、standalone trainer 或 synthetic ablation runtime。

核心不变量：

- base model：`Qwen/Qwen3-VL-8B-Instruct`；
- Fast Adapter：4096→768→4096，两块在线矩阵共 1,179,648 参数；另有
  `P_C:512→768` Associative context 投影；
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
  -> active-head soft-write source
  -> normalized FP32 state-write loss L_assoc
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
p_t = normalize(masked_mean(f_Wt(K_t)))
t_t = normalize(stopgrad(active_head_soft_write_source))
L_assoc = mean_valid(1 - cosine(p_t, t_t))
```

空 Bank 的 `b` 固定为零；硬 payload、count、phase、timestamp 不进入池化。O1/E1/E2 直接使用
当前 active head source，O2 对 present source 做 masked mean；`UNSUPPORTED` 或空 target 跳过
inner update。target 全程 detach，不读取官方标签。W0 属于 checkpoint 和 Outer optimizer；
Wt 是 per-video FP32 master 临时状态，不注册为 parameter/buffer，不进入 checkpoint。fast MLP、
functional SGD 和 associative loss 均固定为 FP32，残差输出边界再转回模型 dtype。

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
- schema-12 是当前唯一正式训练契约；schema-11 A5 checkpoint 明确拒绝，A5 只能从显式允许的
  A2 权重初始化，并重新创建 state-write Associative 状态、optimizer、scheduler、RNG 与 runtime state；
- `P_C` 冻结、Inner SGD 不可达；
- Query outer loss 正式使用 `ema_answer_ref`：先用一步滞后的 loss EMA 对齐 Answer，
  再用 `q_target/q_operator/q_time` 激活梯度 RMS EMA 平衡 Task、Operator、Retrieval、Time；
  四槽固定且辅助组限制为 Answer 的至多 30%；
- loss/gradient EMA 随同阶段 resume 恢复，A2 初始化 A5 时重置；不提供其他 loss-balance
  模式；
- 状态参数按 shared、task、router-time、retrieval 四组独立裁剪，四组 RSS 预算保持与旧
  state 单组相同；
- 多卡动态分支使用零值 graph anchor 保持梯度集合一致，单卡不构造 anchor。

### A5

- 先独立执行 128-step Fast/State Warmup：重新加载完整 A2 checkpoint，Qwen bitwise 冻结且
  不进入 optimizer，Fast persistent 参数与全部状态模块训练；只使用现有 Query Outer objective；
- Warmup 成功后仅原子保存小型 handoff bundle。Main 再加载原 A2 checkpoint，严格校验并叠加
  bundle，重置 loss-balancer EMA，创建全新 optimizer/scheduler，恢复部分 Qwen 解冻；
- Support inner objective 预测当前 active head 的 soft-write source；Bank 语义影响当前 key，
  Fast Adapter 输出随后影响 soft object selection 和唯一一次 hard Bank/FSM write；
- `L_assoc` 是 normalized FP32 cosine，不使用官方标签或硬 payload；它只用于 functional SGD，
  不以 auxiliary 权重加入 Outer loss；
- Support 不设人工数值上限；
- 每 8 个 Support 截断二阶图并重锚 W0；
- 每个 segment 只对 Query Answer/State Outer loss 执行 backward，deferred VJP 将 Query 梯度
  传回 `W_t`、W0、`P_C` 和慢模块；episode 末由 Outer optimizer 单次 step；
- 每个 Query 的全部 fast matrix cotangent 在 unscale 后按联合范数独立裁剪到 1.0，同一
  segment 内将裁剪结果求和；Query 对 Qwen、State 和其他 Outer 参数的直接梯度不参与此裁剪；
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

正式 checkpoint 必须完整匹配 schema-12 模型 key，支持单文件和 sharded safetensors，并包含
`associative_contract_version`。Warmup bundle 只含 allowlist 中的非 Qwen persistent tensor，
并绑定 A2/config/data/code hash；禁止保存 Wt、optimizer runtime、Bank、cache、FSM 和
Associative 临时 context。

唯一历史权重兼容是私有 `legacy_a2_to_a5` profile：只允许旧 A2 缺少新增的 `p_context`、
`associative_contract_version`，以及包含已删除的 `predictor`、`p_value`；其余 missing/unexpected
key 一律拒绝。加载后立即重置 Associative 状态，且该 profile 不得用于 same-stage resume。
旧 A5 checkpoint 仍必须按 schema-12/current contract 严格恢复，不推断、不迁移。

A2/A5 sampler 必须保持四卡任务或 segment parity。非有限 loss/gradient 必须 warning/skip，不能产生部分参数更新。ZeRO、BF16、显存和性能是否可接受只由真实 H200 记录决定。

schema-12 的冻结常量分两层强制：`ProjectConfig._FROZEN_CONTRACT` 在 `load_config()` 处拒绝
其覆盖的路径漂移；observation head、State Bank、Spatial/Temporal Encoder 与 Inner SGD 的字段
由各模块 `_validate_*_config` 在 build 时拒绝。后者是唯一能拦住 `model_copy(update=...)`
绕过 pydantic validator 的路径，因此这些字段只在模块构建期报错——四卡上意味着在
distributed init 之后。schema-12 不含 `paths` 配置块，四个环境变量名直接从 `os.environ` 读取。

## 7. 验证边界

代码测试验证 shape、dtype、因果性、泄漏、梯度、state_dict、checkpoint 和 lifecycle。未执行真实 8B/H200 时，不得从 tiny/CPU 测试推导训练收敛、吞吐或科学收益。
