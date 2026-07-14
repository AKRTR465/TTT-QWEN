# State-TTT-Qwen3VL-8B 项目实施计划

> 规范版本：state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval  
> 修订日期：2026-07-14
> 状态：PARTIALLY IMPLEMENTED / P0-P14 ENGINEERING-VERIFIED
> 说明：本文描述完整目标实现；当前 P0–P14 已通过工程门禁，P15–P22 尚未完整实现或运行。

## 0. 计划目标

本项目以 Qwen3-VL-8B-Instruct 为基座，为 SVCBench 长视频计数任务增加一条可在线适应、
可显式审计的状态路径：

1. Qwen3-VL ViT 提取视频表示；
2. Fast TTT Adapter 在单个视频内部用无标签损失做一步 SGD；
3. 空间对象路和时间事件路把视觉 token 解码为对象、身份和事件观测；
4. Structured State Bank 维护对象身份、事件、时间戳和精确整数状态；
5. Query Embedding Encoder 将问题解耦为 target、operator 和 time；
6. embedding 负责检索和路由，Deterministic State Reader 负责精确算术；
7. 视觉 token、State Token 和精确数字 token 一起送入 Qwen LLM，由 LLM 负责表达答案。

核心原则是：

> 神经网络负责理解“查什么”，显式状态负责保存“发生了什么”，Reader 负责计算“答案是多少”。

### 0.1 第一版固定边界

- 基座固定为 Qwen/Qwen3-VL-8B-Instruct；
- 主插入点位于 Visual Merger 主输出之后、video masked_scatter 之前；
- DeepStack 保持原始 Qwen3-VL 路径，不参与在线更新；
- Fast Adapter 使用 4096 → 768 → 768 → 4096 残差结构；
- 测试时只更新两个 768 × 768 fast weight，共 1,179,648 个在线参数；
- 空间对象和时间事件主干统一使用 768 维状态表示；
- 默认活动对象槽增至 32，State Token 增至 16；
- Inner optimizer 固定为无 momentum、无 weight decay 的单步 SGD；
- 当前 chunk 完成观测和状态更新后再执行 SGD，更新从下一 chunk 生效；
- 四个 Observation Head 不直接输出最终累计答案；
- Query 不使用关键词规则硬分词，改用 embedding 路由和检索；
- 最终整数由 Deterministic State Reader 计算；
- 每个新视频重置 fast weights、SGD 状态、时序缓存和 State Bank；
- 测试时禁止把答案、count、occurrence_times、counting_type 或 counting_subtype 输入模型。

### 0.2 第一版明确不做

- Surprise Gate、学习型更新 Gate；
- Inner AdamW、Muon、momentum SGD 或每 chunk 多步更新；
- 让 LLM 或连续 embedding 直接回归最终累计整数；
- 固定 Top-K 状态检索；
- O1 无标签一致性损失；
- DeepStack 改造；
- ANN 向量数据库；
- 多种辅助正则项堆叠；
- 使用 query_time 之后的帧；
- 在 generate 自回归循环中重复执行 TTT 更新。

## 1. 贯穿全文的演示输入

本节数字只用于核对张量，不是 SVCBench 的固定数据规格。

### 1.1 Demo video

假设视频经过采样和 resize 后为：

\[
X\in\mathbb R^{1\times16\times3\times224\times224}.
\]

| 符号 | 演示值 | 含义 |
| :--- | :--- | :--- |
| \(B\) | 1 | batch size |
| \(F\) | 16 | 采样帧数 |
| \(C\) | 3 | RGB 通道 |
| \(H,W\) | 224, 224 | resize 后分辨率 |
| fps | 2 | 每秒采样帧数 |
| 时长 | 8 秒 | \(16/2\) |
| temporal patch | 2 | 每个 tubelet 覆盖 2 帧 |
| spatial patch | 16 | 每个 patch 为 \(16\times16\) |
| spatial merge | 2 | Visual Merger 合并 \(2\times2\) patch token |

Video Processor 先得到：

\[
T_g=16/2=8,\qquad
H_g=224/16=14,\qquad
W_g=224/16=14.
\]

因此：

\[
\text{video\_grid\_thw}=[8,14,14].
\]

tubelet 数量为：

\[
N_{\mathrm{patch}}=8\times14\times14=1568.
\]

每个 tubelet 展平维度为：

\[
D_{\mathrm{patch}}
=2\times3\times16\times16
=1536.
\]

所以：

\[
\text{pixel\_values\_videos}
\in\mathbb R^{1\times1568\times1536}.
\]

pixel_values_videos 已经不是原始 \([B,F,C,H,W]\) 像素张量。每一行代表一个归一化并
展平的时空 tubelet。

### 1.2 Demo query

假设问题为：

> 当前画面有几架无人机？

为了图示，假设 tokenizer 后：

\[
L_q=7.
\]

问题 token 表示为：

\[
Q_h\in\mathbb R^{1\times7\times4096}.
\]

\(L_q=7\) 只是演示值；真实长度由 tokenizer 和 chat template 决定。

### 1.3 演示长度与网络容量无关

| 数字 | 来源 | 是否动态 |
| :--- | :--- | :--- |
| \(T=8\) | 16 帧按 2 帧一个 tubelet 得到的时间长度 | 随视频输入变化 |
| temporal heads = 12 | Causal Transformer 的注意力头数 | 配置超参数 |
| state token count = 16 | State Token Cross-Attention 的 learned query 数 | 配置超参数 |

时间长度、注意力头数和 State Token 数彼此独立。

## 2. 总体数据流

~~~mermaid
flowchart TB
    VIDEO["Demo video<br/>[1,16,3,224,224]"] --> PROC["Video Processor<br/>grid_thw=[8,14,14]<br/>pixels=[1,1568,1536]"]
    PROC --> VIT["Qwen3-VL ViT<br/>27 layers, dim=1152"]
    VIT --> MERGER["Main Visual Merger<br/>1568 → 392 tokens<br/>[1,392,4096]"]
    VIT -.->|"ViT indexes 8,16,24"| DEEP["原始 DeepStack<br/>3 × [1,392,4096]"]

    MERGER --> FAST["Fast TTT Adapter<br/>4096 → 768 → 768 → 4096<br/>online params≈1.18M"]
    FAST --> Z["Adapted video tokens<br/>Z_t=[1,392,4096]"]

    Z --> SPACE["空间对象路<br/>2-stage Recurrent Slot Attention<br/>A_t=[1,32,768]"]
    Z --> TIME["时间事件路<br/>空间池化 + 6-layer Causal Transformer<br/>H_t=[1,8,768]"]

    SPACE --> O1["O1 当前数量 Decoder<br/>[1,32,6]"]
    SPACE --> O2["O2 身份 Decoder<br/>identity=[1,32,256]"]
    TIME --> E1["E1 点事件 Head<br/>[1,8,3]"]
    TIME --> E2["E2 区间事件 Head<br/>event=[1,8,4]<br/>phase=[1,8,4]"]

    O1 --> BANK["Structured State Bank<br/>typed records + exact state<br/>semantic=[1,N_s,512]"]
    O2 --> BANK
    E1 --> BANK
    E2 --> BANK

    QUESTION["Demo query<br/>Q_h=[1,L_q,4096]"] --> QENC["Query Embedding Encoder<br/>4-layer Bi-Transformer, dim=768"]
    QENC --> QT["q_target [1,512]"]
    QENC --> QO["q_operator [1,512]"]
    QENC --> QTIME["q_time [1,512]"]
    QT -.->|"条件化对象槽"| SPACE
    QT -.->|"条件化事件表示"| TIME

    QT --> RETRIEVER["Embedding State Retriever<br/>threshold retrieval, no Top-K"]
    BANK --> RETRIEVER
    QO --> ROUTER["9 learned operator prototypes<br/>8 legal + unsupported"]
    QTIME --> TRES["Time Window Resolver<br/>显式 start/end/query_time"]

    RETRIEVER --> READER["Deterministic State Reader"]
    ROUTER --> READER
    TRES --> READER
    READER --> NUMBER["Exact integer<br/>number tokens"]

    RETRIEVER --> STOK["16 learned State Queries<br/>3-layer Perceiver Resampler<br/>[1,16,4096]"]
    QT --> STOK

    QUESTION --> COMPOSER["LLM Input Composer"]
    Z --> COMPOSER
    STOK --> COMPOSER
    NUMBER --> COMPOSER
    COMPOSER --> LLM["Qwen LLM Decoder<br/>36 layers, dim=4096"]
    DEEP -.->|"保持原始中层注入"| LLM
    LLM --> ANSWER["自然语言答案"]

    TIME --> TTT["L_TTT<br/>L_pred + 0.5 L_id + 0.5 L_event"]
    O2 --> TTT
    E1 --> TTT
    E2 --> TTT
    TTT --> SGD["一步 SGD<br/>只更新 fast weights"]
    SGD -.->|"下一 chunk 生效"| FAST
~~~

## 3. Qwen3-VL 基座接口

### 3.1 已核对的基础配置

| 组件 | 当前计划值 |
| :--- | :--- |
| Vision Transformer depth | 27 |
| Vision hidden size | 1152 |
| Vision heads | 16 |
| patch size | 16 |
| temporal patch size | 2 |
| spatial merge size | 2 |
| Visual Merger output | 4096 |
| DeepStack visual indexes | 8、16、24 |
| LLM layers | 36 |
| LLM hidden size | 4096 |

真实运行以所加载 checkpoint 的 config 为准，启动时必须断言这些关键值。

### 3.2 ViT 与 Visual Merger

3D PatchEmbed 将每个 1536 维 tubelet 投影为 1152 维视觉 token：

\[
[1568,1536]\rightarrow[1568,1152].
\]

27 个 ViT block 保持 token 数：

\[
[1568,1152]\rightarrow[1568,1152].
\]

Main Visual Merger 对每个时间片的 \(2\times2\) 空间 token 做分组：

\[
4\times1152=4608
\rightarrow\operatorname{MLP}
\rightarrow4096.
\]

因此：

\[
1568/4=392,
\qquad
V_t\in\mathbb R^{1\times392\times4096}.
\]

逻辑网格由：

\[
[8,14,14]\rightarrow[8,7,7].
\]

Visual Merger 只压缩空间维，时间长度仍为 8。

### 3.3 State-TTT 插入点

目标调用链为：

\[
\text{Main Visual Merger}
\rightarrow
\text{Fast TTT Adapter}
\rightarrow
\text{adapted video embeddings}
\rightarrow
\text{video masked scatter}.
\]

不得重新使用已经过时的 pooler_output 接口描述。

### 3.4 DeepStack 保持原路径

DeepStack 从配置指定的 ViT block index 8、16、24 取特征，各自通过独立 merger 映射到：

\[
[N_v,4096].
\]

Transformers 当前实现把这三个特征按顺序注入 Qwen decoder 的前三个层级处理路径。这里的
8、16、24 是视觉层索引，不是 LLM 注入层索引。

第一版约束：

- DeepStack 不经过 Fast Adapter；
- DeepStack 不进入 State Bank；
- DeepStack 不计入 LLM 输入 token 长度；
- DeepStack 的 shape、mask 和注入顺序必须与原 Qwen3-VL 完全一致；
- 在线 SGD 不更新 DeepStack 相关参数。

## 4. Fast TTT Adapter

### 4.1 张量结构

输入：

\[
V_t\in\mathbb R^{B\times N_v\times4096}.
\]

降维：

\[
U_t=P_{\mathrm{in}}\operatorname{RMSNorm}(V_t),
\qquad
P_{\mathrm{in}}\in\mathbb R^{4096\times768}.
\]

RMSNorm 固定使用 `eps=1e-6`。输入、输出慢投影都是带 bias 的标准 Linear；只有下述两个
fast matrix 明确不带 bias。meta-learned 初值
\(W_0^{(1)},W_0^{(2)}\) 使用 Xavier uniform 初始化，之后由 Outer Training 学习并随 checkpoint
保存；初始化方法不是测试时更新规则。

Fast MLP：

\[
F_t(U_t)
=W_t^{(2)}\operatorname{SiLU}\left(W_t^{(1)}U_t\right),
\]

\[
W_t^{(1)},W_t^{(2)}\in\mathbb R^{768\times768}.
\]

残差输出：

\[
Z_t
=V_t
+\alpha P_{\mathrm{out}}F_t(U_t),
\qquad
\alpha=0.1.
\]

\[
Z_t\in\mathbb R^{B\times N_v\times4096}.
\]

Demo 中：

\[
Z_t\in\mathbb R^{1\times392\times4096}.
\]

变长 batch 使用 `valid_mask[B,N_v]` 标记真实 Main Merger token。无效 padding 位置的 Adapter
残差必须严格置零，因此这些位置满足 \(Z_t=V_t\)。在线 batch 的每一行必须绑定一个独立的
per-video fast state；不同 batch 行、同一行的两块 \(W_t\) 以及各自的 \(W_0\) snapshot 均不得
共享 storage，也不得把一个视频的 fast version 或更新计数广播到另一行。

### 4.2 Fast 与 Slow 参数

测试时只允许更新：

- \(W_t^{(1)}\)；
- \(W_t^{(2)}\)。

两个 fast matrix 不使用 bias，在线参数量严格为：

\[
2\times768^2
=1,179,648
\approx1.18\text{M}.
\]

带 bias 的输入/输出慢投影和 RMSNorm 精确包含 6,300,416 个参数，因此整个 Adapter 精确包含
7,480,064 个 checkpointed 参数；其中只有 1,179,648 个临时 \(W_t\) 元素进入测试时 SGD。

`state_dict` 只保存 RMSNorm、\(P_{\mathrm{in}}\)、\(P_{\mathrm{out}}\) 和 meta-learned
\(W_0^{(1)},W_0^{(2)}\)。每个视频从当前 \(W_0\) 创建 storage-independent snapshot，再克隆出
临时 \(W_t\)；临时 \(W_t\)、active binding 和 forward audit 均不注册为 Parameter/Buffer，不能
进入 checkpoint。`collect_fast_parameters()` 固定按
\((W_t^{(1)},W_t^{(2)})\) 返回，禁止混入 \(W_0\)、慢投影或 RMSNorm。

以下参数在线冻结：

- Qwen ViT、Visual Merger 和 DeepStack；
- \(P_{\mathrm{in}}\)、\(P_{\mathrm{out}}\)、RMSNorm；
- 空间对象编码器和时间事件编码器；
- O1、O2、E1、E2；
- Query Encoder、Retriever、Reader；
- Qwen LLM。

梯度边界分为两种且同一 batch 不得混用：

- online/inference state：\(W_t\) 是 detached、可求梯度的 leaf tensor；全部 checkpointed
  Adapter 参数临时冻结且不得携带旧梯度。慢参数的数值仍参与前向，但只允许梯度到达输入和
  两块 \(W_t\)，P14 才负责消费这些梯度执行一步 SGD；
- differentiable/meta state：\(W_t\) 从 \(W_0\) 可微克隆，慢参数不 detach，使后续 outer loss
  能回传到 \(W_0\)、RMSNorm 和两个慢投影。P14 固定记录为
  `meta_full_second_order`：inner `autograd.grad(create_graph=True)`，候选 fast weights 不 detach；
  online state 则记录为 `online_leaf`。该模式用于后续 Meta-TTT 训练，不代表在线允许更新慢参数；
  一阶近似只能作为 P16/P21 的显式配置与消融，不能静默替换。

online 路径的 detach 只切断 checkpointed 参数梯度，不能把整个 Adapter 包在 `no_grad()` 中；
因此冻结参数的数值仍位于到 fast weights 的 autograd 计算路径上。

直接向 Fast Adapter 传入与 batch 行一一对应的 `fast_state` sequence，是 P14 和单元测试使用的
functional 边界：online state 下 slow/\(W_0\) 在该计算图中 detach，只有输入与每行两块
\(W_t\) 获得梯度；它本身不是 P18 的正式在线生命周期管理器。

P18 正式推理通过受管的 `use_fast_state()` 生命周期兼容 P3 既有
`adapter(embeddings, valid_mask, metadata)` 签名。该 module-local binding 负责拒绝 stale module
gradient，在线期间冻结 checkpointed 参数的 `requires_grad`，并在 exception-safe、非重入的
context 退出时恢复原标志。P18 仍必须另外证明 video ID 与 batch row/order 对齐、并发调用被正确
串行化或隔离，以及每次受管调用的 `last_audit.used_runtime_state=True`；这些系统级性质不能由 P5
模块测试代替，也不能把 bridge 当作全局或并发 fast-state store。

### 4.3 单步 SGD 与生效顺序

对 chunk \(t\)：

~~~text
1. 使用 W_t 前向得到 Z_t
2. 生成软状态并更新 hard State Bank
3. 计算有效的 L_TTT
4. 从强类型 TTTLossOutput 按当前 video row 取 loss/valid/reason
5. 对 W_t^(1)、W_t^(2) 做一步 SGD
6. 得到 W_(t+1)
7. W_(t+1) 从下一 chunk 开始生效
~~~

单视频 functional update 禁止接收跨视频 batch scalar。每个 row 独立决定更新或跳过，optimizer
attempt counter 必须始终等于 accepted update 与 skip 的和。

优化器固定为：

| 参数 | 值 |
| :--- | :--- |
| optimizer | SGD |
| learning rate | \(1\times10^{-4}\) |
| momentum | 0 |
| weight decay | 0 |
| steps per chunk | 1 |
| gradient norm clip | 1.0 |

下列情况跳过更新：

- 当前 chunk 没有有效 TTT 项；
- 有效时间位置不足；
- loss 非有限；
- gradient 非有限；
- gradient norm 裁剪后仍不可用。

每个新视频从 meta-learned \(W_0\) 重新初始化，严禁跨视频共享 fast state。

## 5. 共享状态编码器

### 5.1 空间对象路

P6 的显式输入边界为：

- adapted merger tokens `Z_t: [B,N_max,4096]`；
- `visual_valid_mask: bool[B,N_max]`；
- `merged_grid_thw: int[B,3]` 和与之对齐的 token count/offset metadata；
- `tubelet_valid_mask: bool[B,T_max]`；
- `q_target: [B,512]`；
- 可选的逐样本 previous slot runtime；
- 仅用于容量审计的 `required_slot_counts: int[B]`。

每个 batch row 必须先按自己的
`N_i=T_i H_i W_i=prod(merged_grid_thw[i])` 切出有效 token，再恢复为
`[T_i,H_i,W_i,4096]`。batch 内允许 `T_i/H_i/W_i` 不同；实现可以在内部按时间和空间补齐，
但必须同时生成对应的 bool mask，不能把 Demo 的 49 个空间位置写死。Demo 的 adapted video token
恢复为：

\[
Z_t:
[1,392,4096]
\rightarrow
[1,8,7,7,4096].
\]

逐样本展平空间后，先将每个 merger token 投影到 768 维：

\[
[B,T_i,H_iW_i,4096]
\xrightarrow{\operatorname{LayerNorm}_{\epsilon=10^{-5}}+\operatorname{Linear}_{4096\to768}}
[B,T_i,H_iW_i,768].
\]

LayerNorm 和 Linear 都是 checkpointed Outer Training 参数，Linear 带 bias。无效 token 和无效
tubelet 不得进入 attention、occupancy confidence 或 recurrent update。

空间对象路使用两阶段 Query-conditioned Recurrent Slot Attention。首次有效 tubelet 的槽初始化为：

\[
S_0=s_{\mathrm{shared}}+P_q(q_{\mathrm{target}})
+\frac{\operatorname{SinusoidalCode}(k,d)}{\sqrt{768}}.
\]

其中 `s_shared: [1,1,768]` 是所有槽共享的唯一可学习 seed，`P_q` 是全模块唯一的带 bias
`512→768` 投影。固定 sinusoidal slot code 按 slot index 和 feature dimension 确定，作为
`persistent=False` buffer 注册，没有可学习缩放、不计参数、不进入 `state_dict`。后续有效 tubelet
使用上一有效 tubelet 的槽作为 recurrent 初始化；`P_q(q_target)` 仍在每次 refinement 中条件化
attention query。

对每个 Stage、每次 refinement，令当前槽为 `S`、当前时间片有效空间 token 为 `X`：

\[
\begin{aligned}
Q &= W_Q\left(\operatorname{LN}_{\mathrm{slot}}(S)+P_q(q_{\mathrm{target}})\right),\\
K &= W_K\operatorname{LN}_{\mathrm{input}}(X),\\
V &= W_V\operatorname{LN}_{\mathrm{input}}(X),\\
L &= QK^\top/\sqrt{64}.
\end{aligned}
\]

Q/K/V/O 都是带 bias 的完整 `768→768` 投影。这里不能直接使用标准 MHA forward 的
token-axis softmax；归一化固定为经典 Slot Attention：

1. 对 slot 轴执行 softmax，使每个有效 token 在有效槽之间竞争；
2. 将无效 token 和无效槽的 assignment 严格置零；
3. 对每个槽沿有效 token 轴再次归一化，分母 epsilon 固定为 `1e-8`；
4. 聚合 V、拼接 12 个 head，并经过 O projection。

随后执行：

\[
S' = \operatorname{GRUCell}(U,S),
\qquad
S'' = S' + W_2\operatorname{SiLU}
\left(W_1\operatorname{LN}_{\mathrm{ffn}}(S')\right),
\]

其中 GRU hidden size=768，FFN 为 `768→3072→768`，三个 LayerNorm 的 eps 都是 `1e-5`。
每个时间片的递归关系为：

\[
A_{t,\tau}
=\operatorname{SlotStage}_2
\left(
X_{t,\tau},q_{\mathrm{target}},
\operatorname{SlotStage}_1
(X_{t,\tau},q_{\mathrm{target}},A_{t,\tau-1})
\right).
\]

每个 Slot Stage 固定为：

- slot dim = 768；
- multi-head slot attention = 12 heads，head dim = 64；
- 每个时间片 3 次共享参数的 slot refinement；
- GRUCell hidden size = 768；
- slot FFN = 768 → 3072 → 768；
- 三个 Pre-LayerNorm、SiLU、FFN residual；
- q_target 通过上述同一个 512 → 768 投影条件化共享槽初始化和 attention query；
- 两个 Stage 的 Q/K/V/O、三个 LayerNorm、GRUCell 和 FFN 参数及 storage 均不共享；
- stage 内的 3 次 refinement 重复调用同一个 Stage 对象，不复制参数；
- Stage 2 复用同一个 `X`，并以 Stage 1 输出作为初始化。

输出：

\[
A_t\in\mathbb R^{B\times K_a\times768},
\qquad K_a=32.
\]

`A_t` 取当前 chunk 最后一个有效 tubelet 的 Stage 2 输出；无效 tubelet 原样 carry previous
state。有 previous state 但当前 row 无有效 tubelet 时保留原状态并写 skip audit；既无 previous
state 又无有效 tubelet 时必须 fail closed，不能静默生成伪观测。

Demo：

\[
A_t\in\mathbb R^{1\times32\times768}.
\]

\(K_a=32\) 的含义是“当前 chunk 最多并行处理 32 个活动对象槽”，不是：

- 整段视频最多只有 32 个身份；
- State Bank 只能保存 32 条记录；
- O2 Confirmed 身份库的容量。

实现约束：

- 槽初始化使用共享、query-conditioned seed 和固定非持久 code，不能使用独立的
  \([32,768]\) 永久可学习身份参数；
- batch 内使用 `slot_valid_mask`，无效槽保持原值，occupancy confidence 严格为 0；
- occupancy confidence 不增加参数：对每个 head、每个槽求 token 再归一化前的有效 assignment
  mass，除以有效 token 数，再对 12 个 head 求均值；结果位于 `[0,1]`，使用最后一个有效
  tubelet 的值；
- `required_slot_counts` 是调用方提供的结构容量需求，不是 P6 从视频预测的对象数。每个 forward
  只审计一次 `excess=max(required_slot_counts-K_a,0)`；仍计算已有 `K_a` 个槽，累计 excess 并另记
  overflow event，不替换、不扩容，且 required 值不得改变槽数值；
- 真实对象语义、新对象判断和长期记录生命周期属于 P8/P9，P6 不得把 capacity audit 表述为
  semantic overflow 检测；
- 默认 max_active_slots = 64；
- 后续消融比较 16、32、48、64，但正式基线固定 32，forward 不动态创建参数。

slot runtime 按 video 和 batch row 隔离，至少包含槽、valid mask、confidence、累计
overflow 及其审计计数。forward 不原地修改 previous runtime，每一行 next runtime 使用新的、
彼此不共享的 storage；runtime 不注册为参数、不进入 `state_dict`。`detach_runtime=True` 只 detach
交给下一 chunk 的 runtime，当前 `A_t` 仍保留到 adapted embeddings 和 fast weights 的 autograd
路径；`detach_runtime=False` 保留跨 chunk 图，供 Outer Training 使用。

P6 到此为止：可以提供无状态的 grid 恢复 helper，但不实现 P7 的时间空间池化、因果 Transformer
或 cache；不实现 P8/P9 的对象语义与 hard state，也不承担 P13 的模型编排或 P18 的完整受管
推理生命周期。

### 5.2 时间事件路

P7 的显式输入边界为：

- adapted merger tokens `Z_t: [B,N_max,4096]`、`visual_valid_mask` 和 merged-grid metadata；
- `tubelet_valid_mask: bool[B,T_max]`、tubelet timestamps 和显式全局
  `position_ids: int[B,T_max]`；
- 合法 `query_time: [B]` 与 `q_target: [B,512]`；
- 每行 `video_id`、`trajectory_id` 和稳定的 query signature；
- 可选的逐样本 previous temporal cache，以及默认开启的 `detach_cache`。

timestamps 用于 query-time 因果审计，global position id 用于位置编码、滑窗 mask 和 overlap
对齐；二者不得互相冒充。输入 timestamp/query_time 使用独立于模型 FP16/BF16 的 FP32 或
FP64，cache 内 timestamp 统一保存为 FP64。有效 timestamp 必须有限、非负、严格递增且不超过
本行 `query_time`，cache 中也执行同一检查。检测到未来时间、owner 不匹配或不可解释的倒退时
fail closed，不静默过滤。

每个时间 tubelet 对应 \(7\times7=49\) 个 merger token：

\[
[B,392,4096]
\rightarrow
[B,8,49,4096].
\]

先执行 `LayerNorm(4096,eps=1e-5)` 和带 bias 的 `Linear(4096→768)`，再使用
q_target 条件化的多头空间注意力池化。`q_target` 先经带 bias 的 `512→768` 投影；空间
attention 的 Q/K/V/O 都是带 bias 的 `768→768` 投影，12 heads、head dim 64。无效空间 token
不参与 Key/Value；整条无效 tubelet 的输出严格为零，不能让 all-masked softmax 产生 NaN：

\[
[B,8,49,4096]
\rightarrow
[B,8,768].
\]

随后按显式 global position id 加入无参数 absolute sinusoidal tubelet position encoding。位置编码
没有 learned table、缩放参数或 modulo/clamp，chunk 边界不能把 position id 重置为 0。再使用六层
参数互不共享的 Pre-LayerNorm 严格因果 Transformer：

| 参数 | 值 |
| :--- | :--- |
| hidden size | 768 |
| layers | 6 |
| attention heads | 12 |
| head dimension | 64 |
| intermediate size | 3072 |
| LayerNorm epsilon | 1e-5 |
| FFN activation | GELU |
| dropout | 0.1 |
| Q/K/V/O bias | true |
| mask | causal，包含当前位置 |
| causal window | 含当前位置共 64 个 tubelet |
| temporal cache | 六层逐层 K/V，最近 64 个有效 tubelet |

每层使用两个 Pre-LayerNorm、带 bias 的 Q/K/V/O 和 `768→3072→768` GELU FFN；六层之后
不增加额外 final LayerNorm。对绝对 query/key 位置 \(p_q,p_k\)，full forward 和 cache forward
必须使用同一个窗口：

\[
p_q-63 \le p_k \le p_q.
\]

因此“strict causal”在本项目中表示禁止未来、允许 self，不表示严格下三角。padding 既不作为
Key/Value，也不进入 cache 或 loss target。只在 chunk 结束时裁 cache、却允许 chunk 后部看到
`64 + chunk_prefix` 个位置是不合规的；滑窗条件必须在 attention mask 本身执行。

cache 保存每一层的 K/V，而不是把上一 chunk 的最终 `hidden` 重新送入六层。主 attention cache
始终只含最近 64 个位置；最终 hidden 可以与 timestamps、valid mask 一起保留用于输出兼容和审计，
但不能替代 layer-wise K/V。每个 cache row 还保存 absolute position ids、`total_seen`、`video_id`、
`trajectory_id` 和 query signature；owner 必须与当前 batch row 完全一致，batch 内禁止交换或共享
可变 runtime。

P2 的固定 overlap 是 4 个 tubelet。为了在主 cache 已满后仍能按当前 adapted embeddings 重算
overlap，runtime 另外保存紧邻主 cache 之前、最多 3 个位置的 replay-only per-layer K/V margin
（`overlap-1`）。该 margin 不进入 `H_t`、不计入主 `cache_length`，也不会扩大 attention mask；每个
query 仍只能选择上式定义的 64 个位置。它只补足 overlap 首位置重算时已经从主 cache 淘汰、但仍在
其合法 64-window 内的三个 key/value。

P2 的相邻 chunk 含重叠 tubelet。重叠位置按 global position id 执行 replay/replace：结合上述
replay margin，先移除被当前 chunk 重放的同位置旧 K/V，再以当前 adapted token 重算并替换，不能
把重叠前缀作为新位置重复追加。无法由主 cache 加 replay margin 解释的历史倒退必须 reset 或 fail
closed。追加后主 cache 只保留按全局位置排序的最近 64 个有效 tubelet；新 video、trajectory 或
query signature 都需要空 cache。

`detach_cache=True` 是运行时默认值：只 detach/clone 交给下一 chunk 的 K/V 和 hidden，当前
`H_t` 仍保留到 adapted embeddings、q_target 和 fast weights 的梯度；Outer Training 若显式使用
`False`，必须同时承担有界 autograd graph。full/chunk 数值等价验收在 `eval()` 下进行，因为训练态
dropout 0.1 不承诺逐元素相同。

输出：

\[
H_t\in\mathbb R^{B\times T\times768}.
\]

Demo 中 \(T=8\)：

\[
H_t\in\mathbb R^{1\times8\times768}.
\]

每个时间位置覆盖 2 个采样帧；文档中的“帧级状态”严格说是 tubelet 级状态。T 来自每行
merged grid，不得与 12 个 attention heads 或 16 个 State Token 混淆。全无效 row 安全返回零
输出并保持/生成合法空 cache；`T=1`、变长 T 和尾 padding 都遵循同一 mask/cache 契约。

## 6. 四个 Observation Head

四个模块是任务解码器，不是 LLM Decoder，也不直接生成最终累计数字。v5 提高解码器容量，
但仍显著小于 Qwen 基座。

### 6.1 输出契约

| Head | 输入 | 输出 | 含义 |
| :--- | :--- | :--- | :--- |
| O1 当前数量 | \(A_t,q_{\mathrm{target}}\) | \([B,K_a,6]\) | 对象存在、目标、可见、进入、离开、置信度 |
| O2 身份 | \(A_t\) | identity \([B,K_a,256]\)；score \([B,K_a,2]\) | 身份向量、novelty、match confidence |
| E1 点事件 | 已由 P7 用 \(q_{\mathrm{target}}\) 条件化的 \(H_t\) | \([B,T,3]\) | eventness、completion、transition |
| E2 区间事件 | 已由 P7 用 \(q_{\mathrm{target}}\) 条件化的 \(H_t\) | event \([B,T,4]\)；phase \([B,T,4]\) | start、active、end、complete 和阶段分布 |

O1/O2 都读取完整的 768 维对象槽：

\[
\mathbb R^{768}\rightarrow\mathbb R^6,
\qquad
\mathbb R^{768}\rightarrow\mathbb R^{256}.
\]

这两个输出是并行投影，不是把 768 维切成 \(6+256\)。

四个 Head 的主分类输出统一保留 raw logits，并额外返回仅用于诊断的 probability、与序列轴同形
的 valid mask、timestamp 和 global position id。O1/O2 metadata 为 `[B,K_a]`，E1/E2 为
`[B,T]`；无效位置的 logit、probability 和 identity 全部为零，timestamp/global position 使用
`-1`。O1、E1、E2 event
和 O2 score 使用 sigmoid debug probability，E2 phase 使用 softmax。P8 不应用任何 bootstrap
阈值，也不修改 hard state。

所有 Observation Head 的 LayerNorm 固定 `eps=1e-5`、affine=True；所有 Linear、Conv1d 和 GRU
使用 bias，Head 内不使用 dropout（GRU 显式 `dropout=0`）。E1/E2 不再直接接收第二份 q_target；
流式 runtime 只保存 query signature 做 owner 校验。在线策略冻结四个 Head 的参数，但 forward 不得
包裹 `torch.no_grad()`，也不得 detach 输入，确保梯度仍能从 soft observation 回到 Fast Adapter。

### 6.2 O1 当前数量

O1 使用逐槽共享的三层 MLP，并以 q_target 生成 FiLM scale/shift：

~~~text
q_target [512] → Linear 512→1536 → FiLM(scale, shift)
A_t [B,32,768]
→ LayerNorm(x) * (1 + scale) + shift
→ Linear 768→1024 → SiLU
→ Linear 1024→1024 → SiLU
→ Linear 1024→6
~~~

输出六个 raw logits：object、target、visible、enter、exit、confidence。FiLM 的 `1+scale` 是固定
identity-preserving 形式，不得改为直接 `scale*x+shift`。该解码器精确为 2,632,710 参数。

逐槽软计数：

\[
\hat c_t
=\sum_i
p_i^{\mathrm{object}}
p_i^{\mathrm{target}}
p_i^{\mathrm{visible}}.
\]

hard 状态保存：

- current_visible_count；
- baseline_count；
- 每槽 enter/exit/visible 状态；
- 更新时间和置信度。

### 6.3 O2 身份

O2 使用共享 trunk 和两个输出分支：

~~~text
A_t [B,32,768]
→ LayerNorm
→ Linear 768→1024 → SiLU
→ Linear 1024→1024 → SiLU
├→ Linear 1024→256 → L2Norm    identity
└→ Linear 1024→2               novelty / match confidence
~~~

identity 的 L2 norm 在 FP32 中以 `eps=1e-8` 计算；有效位置若出现零范数，确定性回退为第一个
坐标为 1 的 unit-basis 向量，再转换回模型 dtype。无效位置仍严格为零。该解码器精确为
2,103,042 参数。

身份向量：

\[
e_{t,i}\in\mathbb R^{256},
\qquad
\lVert e_{t,i}\rVert_2=1.
\]

256 维 identity embedding 用于判断“是不是同一实体”，不能替代用于语义查询的 512 维
semantic_embedding。

身份生命周期：

~~~text
unmatched observation
→ Candidate
→ 连续可靠观测达到阈值
→ Confirmed
→ unique_count 只在首次晋升时 +1
~~~

### 6.4 E1 点事件

E1 面向持续时间很短、应计一次的事件，输出：

- eventness；
- completion；
- transition。

E1 使用五层 gated causal TCN：

~~~text
H_t [B,T,768]
→ LayerNorm + Linear 768→512
→ 5个gated residual TCN blocks
   kernel=3, dilations=[1,2,4,8,16], channels=512
→ Linear 512→3
~~~

每个 block 使用 filter/gate dilated Conv1d、1×1 residual projection、LayerNorm 和 SiLU。
每层只在左侧 padding `(kernel-1)*dilation`，并固定为：

\[
g=\operatorname{SiLU}(\operatorname{Conv}_{filter}(x))
\odot\sigma(\operatorname{Conv}_{gate}(x)),\qquad
y=\operatorname{LayerNorm}(x+\operatorname{Conv}_{1\times1}(g)).
\]

五层的总 receptive field 为 63 个 tubelet。流式路径保存无参数 `projected_history`：最近 66 个
`Linear 768→512` 后的状态，其中 62 个为最深 TCN 所需左上下文，4 个为 P2 overlap。新 chunk
先删除 cache 尾部四个同 global position 的旧 projected state，再用当前 adapted 路径重算并替换；
owner 固定为 `(video_id, trajectory_id, query_signature)`，owner、position 或 timestamp 不匹配时
fail closed。该解码器精确为 9,584,643 参数，不使用 BatchNorm。

hard 状态机使用双阈值、cooldown 和 Temporal NMS，防止一个持续多帧的事件重复计数。

### 6.5 E2 区间事件

E2 面向具有开始、持续、结束过程的事件，状态迁移为：

E2 使用两层 GRU 捕获持续阶段，再由双分支解码：

~~~text
H_t [B,T,768]
→ LayerNorm
→ 2-layer unidirectional GRU, hidden=768, batch_first=True, bias=True, dropout=0
├→ Linear 768→4    start / active / end / complete
└→ Linear 768→4    phase logits
~~~

GRU hidden state 按 batch row 隔离并在 reset 时清空。流式状态按
`(video_id, trajectory_id, query_signature)` 隔离，并保存 5 个 hidden checkpoint：overlap 之前的
anchor 加四个 overlap 位置后的状态。新 chunk 恢复 anchor，以当前四个 overlap hidden 重新运行
GRU并替换旧 checkpoint，禁止把 overlap 重复追加。event raw logits 顺序固定
start/active/end/complete；phase raw logits 顺序固定
inactive/active/end_candidate/completed。该解码器精确为 7,094,792 参数。

~~~text
INACTIVE
→ ACTIVE
→ END_CANDIDATE
→ COMPLETED
~~~

只有确认完整结束后：

\[
\text{completed\_count}
\leftarrow
\text{completed\_count}+1.
\]

### 6.6 v5 参数预算

以下预算不含 Qwen3-VL 基座，按标准 Linear、MHA、FFN、GRU 和本文给定层数估算；除
Query learned-attention 的最终标量 scorer 明确不带 bias 外，其余 Linear 按带 bias 计算：

| 模块 | 约参数量 |
| :--- | ---: |
| Fast TTT Adapter（含慢投影） | 7.48M（精确 7,480,064） |
| 其中在线 fast matrices | 1.18M（精确 1,179,648） |
| 空间对象编码器 | 24.81536M（精确 24,815,360） |
| 时间事件编码器 | 48.438272M（精确 48,438,272） |
| Query Embedding Encoder | 36.03M |
| O1 Decoder | 2.632710M（精确 2,632,710） |
| O2 Decoder | 2.103042M（精确 2,103,042） |
| E1 Decoder | 9.584643M（精确 9,584,643） |
| E2 Decoder | 7.094792M（精确 7,094,792） |
| Semantic Projector | 1.316864M（精确 1,316,864） |
| TTT Temporal Predictor | 2.36M |
| 16-token State Resampler | 14.72M |
| Operator Router、Time Resolver、empty record | 约 0.14M |
| **新增模块合计** | **当前分项和 156.715683M（156,715,683）** |

空间对象编码器的精确审计式为：输入 LayerNorm 8,192 + 输入 Linear 3,146,496 + 单一
q projection 393,984 + shared seed 768 + 两个独立 Stage。每个 Stage 为 Q/K/V/O 2,362,368 +
GRUCell 3,543,552 + FFN 4,722,432 + 三个 LayerNorm 4,608，共 10,632,960；因此总计
24,815,360。固定 sinusoidal code、occupancy confidence、mask、runtime 和 capacity audit 都不增加
模型参数。

时间事件编码器的精确审计式为：输入 LayerNorm 8,192 + 输入 Linear 3,146,496 + q_target
projection 393,984 + 空间 pooling MHA 2,362,368 + 六个独立 Transformer layer。每层为
Q/K/V/O 2,362,368 + GELU FFN 4,722,432 + 两个 LayerNorm 3,072，共 7,087,872；因此总计
48,438,272。absolute sinusoidal position、causal/sliding mask、逐层 KV cache、owner metadata、
overlap replay margin/replay-replace 和 detach runtime 都是零参数。

四个 Observation Head 的精确审计式为：O1 的 affine LayerNorm 1,536 + FiLM Linear 787,968 +
三层 MLP 1,843,206 = 2,632,710；O2 的 affine LayerNorm 1,536 + shared trunk 1,837,056 +
identity/score 分支 264,450 = 2,103,042；E1 的输入 LayerNorm/Linear 395,264 + 五个各
1,837,568 的 gated causal block + 输出 Linear 1,539 = 9,584,643；E2 的输入 LayerNorm 1,536 +
两个各 3,543,552 的标准 GRU layer + 两个输出 Linear 6,152 = 7,094,792。debug probability、mask、
timestamp、projected history、rollback checkpoint 和 owner metadata 都是零参数。

Semantic Projector 的精确审计式为：四个 768 维 head-type embedding 3,072 + affine LayerNorm
1,536 + `Linear(768→1024,bias=True)` 787,456 + `Linear(1024→512,bias=True)` 524,800 =
1,316,864。FP32 L2 normalize、dynamic view、record、FSM、audit 和 runtime snapshot 都是零参数。

新增模块约占 8B 基座的 2%。测试时真正变化的参数只有两个 768 × 768 fast matrix，即
1,179,648 个参数；其余新增参数只在 Outer Training 中学习，在线推理时冻结。

## 7. Structured State Bank

### 7.1 作用

State Bank 是当前视频的运行时结构化内存，不是模型参数，也不是外部数据库。

它按以下 key 隔离：

\[
(\text{video id},\text{trajectory id},\text{head type}).
\]

其中代码字段 `trajectory_id` 即 question trajectory id。每个新视频或问题轨迹结束后释放对应
状态；不同 batch row 的 Bank、record、payload、audit 和 snapshot 不共享可变 storage。

### 7.2 统一记录

所有可查询记录至少包含：

~~~text
record_id
head_type
semantic_embedding [512]
timestamp / time_range
valid
confidence
type-specific payload
~~~

`timestamp` 与 `time_range` 严格二选一。`record_id` 在整个 trajectory 内跨 head 单调唯一，记录
失效或释放后也不得复用。append、update、invalidate、snapshot、query 和 release 都是 functional
操作：返回独立新状态，不原地修改输入 Bank。

统一语义视图写作：

\[
E_{\mathrm{state}}
\in\mathbb R^{B\times N_{s,\max}\times512}.
\]

其中：

- 每行 \(N_s\) 是当前 owner/head 分区中、P11 valid/time/similarity 过滤前的实际 stored record 数，
  包括 `valid=false` 记录；
- batch tensor 只 pad 到本 batch 的 \(N_{s,\max}\)，同时返回 `n_state[B]`、present mask、
  record-valid mask 和对齐 record IDs；padding embedding 严格为零；
- 空 Bank 合法返回 `[B,0,512]`，而不是伪造一条 empty record；
- 512 是固定语义检索维度；
- \(N_s\) 不是活动槽数量、身份容量或时间长度。

### 7.3 类型化 payload

| 类型 | 主要字段 |
| :--- | :--- |
| O1 | current count、baseline count、活动槽状态 |
| O2 Candidate | 256 维身份原型、观测次数、TTL、置信度 |
| O2 Confirmed | identity_id、256 维原型、first_seen、last_seen、observation_count |
| E1 | event_kind(Action/Transit)、event_count、recent_event_times、cooldown |
| E2 | event_kind(Periodic/Episode)、completed_count、phase、已完成区间、recent_event_times |

O1、E1、E2 每个 owner/head 分区各维护一条稳定 `record_id` 的聚合记录，更新使用 functional
replace；因此持续更新不会追加重复 summary。O2 在 P9 只使用 generic CRUD 接口，identity matching、
Candidate→Confirmed、prototype EMA、容量增长和 `unique_count` 全部属于 P10。

E1/E2 aggregate 额外冻结由 effective hard operator 派生的 event kind provenance：
`E1-Action/Transit→Action/Transit`，`E2-Periodic/Episode→Periodic/Episode`。kind 不是监督标签，
首次写入后同一 aggregate 不得换型；Reader 收到 kind 与 operator 不一致的记录必须返回 invalid，
不能把错型记录跳过后伪装成计数 0。

每条对象或事件记录额外保存归一化 512 维 semantic_embedding，表示“这条记录是什么”。

semantic embedding 由共享高容量投影器生成：

~~~text
object slot / event state [768]
+ learned head-type embedding [768]
→ LayerNorm(eps=1e-5)
→ Linear 768→1024 → SiLU
→ Linear 1024→512
→ FP32 L2Norm(eps=1e-8, zero-norm→first unit basis)
~~~

head-type table 顺序固定为 O1/O2/E1/E2，形状 `[4,768]`；共享 trunk 不按 head 复制。LayerNorm
使用 affine 参数，两个 Linear 带 bias，dropout=0。Semantic Projector 精确为 1,316,864 参数。
语义检索维度仍保持 512，避免 State Bank 的完整检索成本随状态主干宽度同步增长。

### 7.4 O2 动态容量

- 活动槽默认 32，是 GPU 计算工作集，最大配置 64；
- Confirmed 初始分配 256 个位置，按 256 分块增长，无语义硬上限；
- Candidate 初始分配 64 个位置，可增长但受 TTL 和 512 安全上限约束；
- Confirmed 完整记录默认存于 CPU FP32 分块张量；
- GPU Hot Cache 默认 256，只负责加速；
- E1/E2 最近事件时间戳容量由 256 提高到 512；
- Hot Cache 换出不得改变 unique_count；
- 不同 batch 样本拥有独立 Bank；
- 连续出现超过 256 个身份时必须扩容，不能覆盖旧身份。

P10 的工程 bootstrap 匹配规则固定如下，所有边界均采用 `>=`：

- 256 维 identity 在 CPU FP32 中做全量归一化余弦搜索，Candidate 与 Confirmed 的
  `match_threshold=0.80`；
- novelty、match confidence、reliability 和 Candidate 最低置信度均为 `0.50`；
- `match_confidence>=0.50` 且 `novelty<0.50` 才是 match intent，`novelty>=0.50` 且
  `match_confidence<0.50` 才是 new intent；双高或双低属于矛盾/低信息证据，必须 fail closed、保留
  旧状态并审计；
- 每个 observation 只允许一个全局 top-1，且同一 committed position 的一个身份最多被更新一次。
  同一 observation 的 top-2 cosine 差不超过 `1e-6` 时视为 near tie，禁止用 ID 顺序猜测身份；多个
  observation 争用同一身份时按 cosine、match confidence 降序，再按 slot index、identity ID 升序
  选唯一胜者，其余只审计；
- new intent 仍必须先搜索完整 Confirmed store。若已有 Confirmed cosine 达到 `0.80`，则记
  novelty/similarity 冲突且禁止创建重复 Candidate；match intent 找不到唯一 Candidate/Confirmed 时也
  不猜测新身份。

Candidate 第一次可靠观测把 TTL 设为 8。每个 owner 的同一 global position 重放必须幂等，不增加
观测次数、不推进 promotion、也不减少 TTL；只有新的 committed position 才先执行匹配，再对未匹配
Candidate 将 TTL 减 1。可靠匹配把 TTL 重置为 8；减到 0 的 Candidate 在该 position 末尾失效并
删除。promotion 需要两个不同且连续的 committed position 都产生可靠匹配，中间任何未匹配 position
都会清零连续可靠 streak。prototype 更新固定为

\[
p_{new}=\operatorname{L2Norm}_{FP32}(0.9p_{old}+0.1e_{obs}),
\]

归一化 `eps=1e-8`，零范数仍回退第一个 unit basis；旧/新 prototype、阈值、cosine、position 和决策
都必须可审计。Candidate confidence `>=0.50` 才保留，低于 `0.50` 才属于 low-confidence prune。

Candidate 容量按 64 增长到 512。满 512 的新建请求先清理 TTL 已过期项，再清理
confidence `<0.50` 的项；低置信度候选按 `(confidence asc, last_position_id asc, candidate_id asc)`
确定性排序。仍无空位时拒绝新 Candidate 并增加 `candidate_overflow`，禁止覆盖已有项；每次满容量
admission attempt 都保留 overflow 审计。

Identity Bank owner 固定为 `(video_id, trajectory_id)`，head 隐含为 O2；不同 owner 的 ID、CPU
chunks、Candidate、Hot Cache、审计和 snapshot 不共享可变 storage。Candidate 至少保存
`candidate_id`、FP32 normalized prototype、总观测数、连续可靠 streak、TTL、confidence、
`first_seen/last_seen`、对应 position 以及 `semantic_record_id`；Confirmed 至少保存单调且不复用的
`identity_id`、prototype、`first_seen/last_seen`、observation_count、prototype version 和
`semantic_record_id`。Confirmed 的 `first_seen` 继承 Candidate 第一次可靠观测，而不是 promotion
时刻；对应 O2 StateRecord 的 `timestamp` 固定为 `first_seen`，重复观测只更新 payload 中的
`last_seen`、计数、prototype、semantic 和 confidence。

Identity Bank 是 identity matching、容量和 `unique_count` 的唯一真值，Structured State Bank 是
512 维 semantic record 的唯一真值。两者必须由一次 functional transaction 同时返回新状态；任一
校验、扩容或 record 写入失败都不得留下半提交。Candidate 创建/更新链接一条 O2 Candidate record；
过期或 prune 时显式 invalidate。promotion 先 invalidate Candidate record，再 append 新的 Confirmed
record，旧 record ID 作为 tombstone 保留且不复用；Confirmed 重复观测 functional replace 同一
Confirmed record。Candidate record 不具备语义检索资格，只有 valid Confirmed record 才具备资格；
真正的查询和过滤已由 P11 实现并通过工程验收。

Confirmed CPU FP32 分块 store 始终是匹配和计数真值，搜索不得因 Hot Cache hit 提前返回。第一版
`exact_search=true`、`ann_enabled=false`：每次决策都以完整 CPU store 的 FP32 top-1 为准。Hot
Cache 默认开启，容量 256，保存 CUDA BF16 加速副本；CPU-only 单元测试允许显式设备替代，但不能
改变决策。cache miss 在 CPU full exact 之后换入；命中只 touch/prefetch。prototype version 不一致
视为 miss；CPU prototype 更新时同步刷新已缓存副本。换出使用
`(last_accessed_position asc, identity_id asc)` 的确定性 LRU，不删除 CPU record、不修改
`unique_count`，cache 开/关必须返回相同 identity 和 hard count。

identity duplicate rate 与 missed-new-identity rate 只在带标签的离线 evaluator 中计算，标签禁止
进入 Bank/runtime。设 (M) 为能映射到真值实体的 Confirmed ID 数，(G_m) 为其中覆盖的不同真值
实体数，(G) 为轨迹内真值实体总数：

\[
\text{duplicate-rate}=\frac{M-G_m}{M},\qquad
\text{missed-new-rate}=\frac{G-G_m}{G}.
\]

分母为 0 时返回 `not_applicable`，不得静默写 0。Bank 只输出 observation→Candidate/Confirmed 的
decision audit，评估器据此计算上述指标。`0.80/0.50/1e-6`、两次 promotion 和 EMA 都是 P10
工程 bootstrap，状态必须为 `bootstrap_calibration_required`，P21 再用训练折或独立校准集复校。

### 7.5 梯度边界

- Semantic Projector 是独立 `nn.Module`：参数进入模型 `state_dict` 和 Outer optimizer，不进入
  Inner SGD；在线冻结参数，但 forward 不包 `torch.no_grad()`、不 detach 输入；
- projector 先在正常 autograd 中产生 soft semantic embedding，hard writer 再统一位于
  `torch.no_grad()` 中对写入 tensor 执行 detach+clone；
- Bank/FSM/runtime 不注册为 `nn.Parameter` 或 buffer，不进入模型 `state_dict`、Outer optimizer
  或 Inner SGD；显式 runtime snapshot 与模型 checkpoint 分离；
- TTT/State Loss 使用 detach 前的 soft 分支，保证梯度能到达 Projector、Observation Head 和
  Fast Adapter；hard record、payload、audit 与输入不得共享 storage。

### 7.6 O1/E1/E2 hard-state 冻结规则

O1 的 object、target、visible、enter、exit、confidence 六个 bootstrap 阈值均为 0.5，边界采用
`>=`。baseline 只能由调用方在 trajectory 内显式 set once；P9 不猜测 baseline 时间。每个新
global position 从完整逐槽 hard state 重算 `current_visible_count`，禁止仅靠 enter/exit 做累计
增减。同 position 重复输入幂等；证据漂移、低置信度、enter/exit 冲突、invalid slot 和空间 overflow
只审计并保留已提交状态。

E1 使用 eventness 0.7/0.3 hysteresis：IDLE 在 eventness `>=0.7` 时进入 ACTIVE；只有 ACTIVE 且
completion、transition 都 `>=0.7` 时才确认一次事件。cooldown 与 Temporal NMS 共用
`min_gap_seconds=0.5`，eventness `<=0.3` 才 re-arm。recent event times 最多 512；淘汰最旧项必须
记录 audit，且不得减少累计 `event_count`。

E2 每个 global position 最多迁移一次：INACTIVE→ACTIVE 需要 start `>=0.6` 且 phase argmax 为
ACTIVE；ACTIVE→END_CANDIDATE 需要 end `>=0.6` 且 phase argmax 为 END_CANDIDATE；
END_CANDIDATE→COMPLETED 需要 complete `>=0.7` 且 phase argmax 为 COMPLETED。active event
evidence 只用于诊断和 phase 一致性，不新增未冻结 active threshold。COMPLETED 至少保持一个位置；
后续只有 phase argmax 为 INACTIVE 且所有 event probability 都 `<=0.5` 时才能 re-arm。只有完整
三步迁移才增加一次 `completed_count` 并保存完整区间。

P8 overlap 中已提交的 global position 在 hard path 只做幂等检查，不 replay/replace、不得重复计数；
证据漂移进入 audit。P8 的 E2 GRU runtime 仍归 P8，P9 只拥有 hard FSM；P18 在 reset/release 时
统一清理两条 runtime。

## 8. Query Embedding Encoder

### 8.1 输入与池化

输入：

\[
Q_h\in\mathbb R^{B\times L_q\times4096}.
\]

\(Q_h\) 必须只来自问题 token，不得含答案或测试标签。v5 固定使用 Qwen token embedding，
不额外执行一次完整 36 层回答解码。

先投影到 768 维，再使用四层双向 Transformer Encoder：

~~~text
Q_h [B,L_q,4096]
→ Linear 4096→768
→ 无参数 sinusoidal position encoding
→ 4-layer bidirectional Transformer Encoder
   hidden=768, heads=12, FFN=3072, Pre-LN, GELU, dropout=0.1
→ X_q [B,L_q,768]
~~~

问题在查询前已经完整可见，因此这里使用双向 attention，只屏蔽 padding，不使用 causal mask。
sinusoidal position encoding 用来保留词序，不引入额外可训练参数；Transformer Encoder 和三个
embedding head 后均不再添加额外 final LayerNorm。

随后执行 learned-attention pooling：

\[
\alpha
=\operatorname{softmax}
\left(
w^\top\tanh(WX_q)+M
\right),
\]

\[
h_q
=\sum_{\ell=1}^{L_q}\alpha_\ell X_{q,\ell}
\in\mathbb R^{B\times768}.
\]

其中 \(W\) 对应带 bias 的 `768→768` 投影，最终 \(w^\top\) scorer 不带 bias；padding token 的
权重必须严格为 0。

### 8.2 三个独立 embedding

\[
q_{\mathrm{target}}
=\operatorname{norm}
\left(
\operatorname{MLP}_{768\rightarrow1024\rightarrow512,\,GELU}^{target}(h_q)
\right),
\]

\[
q_{\mathrm{operator}}
=\operatorname{norm}
\left(
\operatorname{MLP}_{768\rightarrow1024\rightarrow512,\,GELU}^{operator}(h_q)
\right),
\]

\[
q_{\mathrm{time}}
=\operatorname{norm}
\left(
\operatorname{MLP}_{768\rightarrow1024\rightarrow512,\,GELU}^{time}(h_q)
\right).
\]

三者形状均为：

\[
[B,512].
\]

职责严格分离：

- q_target：问题在问哪个对象或事件；
- q_operator：使用哪种计数操作；
- q_time：使用 now、history、recent window 或显式区间中的哪种时间语义。

完整 Query Encoder 约 36.03M 参数。

### 8.3 Operator prototypes

固定 9 个可训练 prototype：

~~~text
o1-snap
o1-delta
o2-unique
o2-gain
e1-action
e1-transit
e2-periodic
e2-episode
unsupported
~~~

路由 logits：

\[
l_k
=\frac{
\operatorname{norm}(q_{\mathrm{operator}})^\top
\operatorname{norm}(p_k)
}{\tau}.
\]

温度使用正的可训练标量：保存 `log_tau`，以 `exp(log_tau)` 取得 τ，并把数值限制在
`[1e-4,1e4]`；初值固定为 1.0。训练时保留 9 类 raw logits、probability、raw argmax 供监督
分类；测试时使用 hard argmax。最大置信度低于校准阈值时必须返回 unsupported，不能强行分到
最相近的合法操作。P21 校准前 `confidence_threshold=null`：训练路径仍保留 raw 结果，但
eval/inference 路径的 effective operator 一律为 unsupported。

### 8.4 Time Window Resolver

q_time 不能直接作为精确时间窗口。Reader 必须接收显式结构：

~~~text
TimeWindow:
  mode: now | history | recent | explicit_range
  query_time: float
  start_time: float | null
  end_time: float
  valid: bool
~~~

解析与完整性规则：

- 合法 query_points.time 作为 query_time；
- `explicit_time_values` 只能由 canonical question 抽取，并按出现顺序换算为秒逐值核对；它只做
  question-derived 完整性检查，不参与窗口构造，也不是标签；
- 无显式数字时直接使用 operator 默认语义；有显式数字时，全文受限 grammar 必须得到唯一候选，
  两个 pointer 再在所有非 padding token 上分别 hard argmax，并完整覆盖该候选首尾 numeric
  component；错序、只覆盖数字/单位、落在候选外或多候选均判 invalid；
- baseline grammar 只接受英文 `last/past/previous ... seconds|minutes`、中文
  `最近/过去/近 ... 秒|分钟`，以及 `from|between ... to|and ...`、`从 ... 到|至 ...` 区间；
  recent 支持 `and/+` 或 `和/+` 组合单位；
- `2 minutes and 3 seconds` 的完整性 tuple 是 `(120,3)`，窗口 duration 是 123 秒；共享单位区间
  `from 2 to 8 seconds` 的完整性 tuple 和窗口端点均为 `(2,8)`；
- q_time 负责时间语义分类；
- 默认时间语义固定为：O1-Snap→now；O1-Delta/O2-Gain→recent，但没有显式正 duration 时
  invalid；O2-Unique、E1-Action、E1-Transit、E2-Periodic、E2-Episode→history；
- now 为 `(start=null,end=query_time)`，history 为 `[0,query_time]`，recent 为
  `[query_time-duration,query_time]`，explicit_range 使用显式 `[start,end]`；
- 负数、零 recent、非法单位、反向区间、未来端点、早于视频起点、metadata mismatch 或不完整
  pointer span 均返回 `status=invalid`；未校准/低置信度或 mode 不一致返回
  `status=unsupported`。两者都会把 effective operator 强制降为 unsupported，不得猜测或 clamp；
- count 和 occurrence_times 不能参与解析。

Time Window Resolver 固定使用：

~~~text
q_time [512]
→ MLP 512→256→4
→ now / history / recent / explicit_range

X_q [B,L_q,768]
→ 两个Linear 768→1 pointer head
→ 全局非 padding numeric span start / end
→ 唯一候选 grammar 与完整边界一致性
→ 确定性数值与单位解析
~~~

默认 API 在 `train()` 下保留未校准 raw 路径，在 `eval()` 下自动启用 confidence gate；调用方也可
显式传入 inference 开关。这里的受限规则只解析时间数值与单位，禁止把关键词规则用作 operator
路由。

## 9. Embedding State Retrieval

### 9.1 去哪里查询

q_target 查询当前视频、当前问题轨迹的 Structured State Bank，而不是：

- 原始视频像素；
- LLM KV cache；
- O2 的 256 维身份原型；
- 外部向量数据库。

hard operator 先确定 head type，q_target 再在该类型记录中选择语义匹配项。

映射固定为：

| hard operator | 合法 head type |
| :--- | :--- |
| O1-Snap / O1-Delta | O1 |
| O2-Unique / O2-Gain | O2 |
| E1-Action / E1-Transit | E1 |
| E2-Periodic / E2-Episode | E2 |
| unsupported | 无分区，不发起 Bank 查询 |

Retriever 必须使用 Query Encoder 的 effective hard operator，不能退回 raw argmax。查询 owner 与
Bank 的 `(video_id,trajectory_id)` 不一致是结构契约错误，返回 `invalid`，不能伪装成空 Bank。

### 9.2 相似度

\[
q_{\mathrm{target}}\in\mathbb R^{B\times512},
\qquad
E_{\mathrm{state}}\in\mathbb R^{B\times N_s\times512}.
\]

q_target 与记录 embedding 都在 FP32 中以 `eps=1e-8` 重新做 L2 normalize，再计算：

\[
s_i
=\cos(q_{\mathrm{target}},E_{\mathrm{state},i})
=q_{\mathrm{target}}^\top E_{\mathrm{state},i}.
\]

分数：

\[
S\in\mathbb R^{B\times N_s}.
\]

阈值比较固定为 FP32 `score >= 0.35`，等于边界时命中；`record.confidence` 不增加第二道
P11 阈值。有限但范数不大于 `1e-8` 的 q_target 表示检索查询不可靠，返回 `unsupported`，禁止
回退到任意单位基向量。shape、dtype 或非有限输入属于直接输入契约错误。

### 9.3 硬过滤

先按 query owner 和 hard operator 对应 head type 建立候选分区。每行 \(N_s\) 是该分区中
padding 前的全部 stored records 数，包含 `valid=false` 记录和 O2 Candidate；wrong-head 记录不进入
\(N_s\)。随后按下列固定顺序做互斥过滤，一条记录只记入最先命中的原因：

~~~text
invalid
retrieval_ineligible
future
outside_window
below_similarity
~~~

其中 `retrieval_ineligible` 至少排除 O2 Candidate，只有 valid O2 Confirmed 可做身份语义检索。
owner mismatch 在分区前令整行 `invalid`，不计作 empty；wrong-head 只是没有进入合法分区。

时间过滤必须按 record kind 解释：

- O1/E1/E2 是 functional replace 的 aggregate record。其 `StateRecord.timestamp` 表示最新状态的
  因果可用时间，不是 payload 内对象/事件的语义时间。P11 只要求该可用时间不晚于
  `query_time`，不得因 aggregate timestamp 未与 recent/explicit window 相交而丢弃整条记录；
- O1/E1/E2 payload 内的 baseline、event times 和 completed intervals 由 P12 Reader 按 resolved
  TimeWindow 做精确解释；P11 不读取这些字段做算术；
- O2 Confirmed 的 record timestamp 固定为 `first_seen`，以及未来真正的 atomic point/range record，
  在 P11 先满足不晚于 `query_time`，再按闭区间规则与 requested TimeWindow 相交；端点相等算命中；
- future record 一律先于 window/similarity 排除。当前 causal runtime 若传入 owner 不匹配、released
  state 或自相矛盾的 operator/head/time 元数据，必须 fail closed。

第一版默认：

~~~text
record_similarity_threshold: 0.35
threshold_comparison: greater_than_or_equal
similarity_dtype: float32
normalization_eps: 1.0e-8
top_k: null
ann_enabled: false
~~~

阈值只能在训练折或独立校准集确定。固定 Top-K 会遗漏多个合法对象并导致静默少计，因此
第一版返回所有达到或超过阈值的有效记录。`selected_mask` 与原候选列对齐；对全部命中项仅做
`score desc, record_id asc` 的确定性排序，不能截断。并列分数不是冲突，所有并列命中都必须保留。

定义：

- \(N_s\)：过滤前候选记录数；
- \(N_{\mathrm{ret}}\)：最终选中记录数；
- \(0\le N_{\mathrm{ret}}\le N_s\)。

低置信度查询和真正的空集合必须区分：

- 查询与路由可靠、Bank 覆盖有效、无匹配记录：可以解释为计数 0；
- 查询或时间解析不可靠：返回 unsupported，不能把“不知道”当成 0。

状态传播固定为：`TimeResolutionStatus.INVALID -> Retriever invalid`；
`TimeResolutionStatus.UNSUPPORTED`、低置信 effective operator 或零范数 q_target ->
`Retriever unsupported`；可靠查询在 empty Bank、无语义匹配、全部 future、全部 invalid 或全部
retrieval-ineligible 时返回 `empty`，同时保留互斥过滤计数和具体 empty reason。合法多命中返回
`ok`。Retriever 输出必须包含未压缩的 typed selected records、selected IDs、对应分数、候选对齐
mask、\(N_s\)、\(N_{ret}\) 和逐原因审计，供 P12 使用；P11 不计算 exact count、number token，
不做 State Resampler，也不修改 Bank。

Retriever runtime 严禁接收 ground truth。precision/recall 由离线 evaluator 将 selected IDs 与标注
relevant record 集合合并计算；分母为零时返回 `not_applicable`。空检索率定义为
`EMPTY / (OK + EMPTY)`，`UNSUPPORTED` 与 `INVALID` 另行报告，不能混入 empty 分母。

## 10. State Token 与 Deterministic Reader

### 10.1 16 个 State Token 怎么生成

16 个 token 不是从检索结果中选出的 Top-16 记录，而是 16 个可训练 State Query：

\[
Q_{\mathrm{state}}\in\mathbb R^{16\times512}.
\]

对 batch 广播并用 q_target 条件化：

\[
Q
=Q_{\mathrm{state}}[None,:,:]
+q_{\mathrm{target}}[:,None,:]
\in\mathbb R^{B\times16\times512}.
\]

检索到的记录作为 Key 和 Value：

\[
K,V\in\mathbb R^{B\times N_{\mathrm{ret}}\times512}.
\]

使用三层 Perceiver/Q-Former 风格 Resampler。每层依次执行：

~~~text
16个Query之间的self-attention，8 heads
→ 对N_ret条记录的cross-attention，8 heads
→ FFN 512→2048→512
→ Pre-LayerNorm + residual
~~~

Cross-Attention 权重：

\[
A
=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}+M
\right)
\in\mathbb R^{B\times16\times N_{\mathrm{ret}}},
\]

\[
H_{\mathrm{state}}=AV
\in\mathbb R^{B\times16\times512}.
\]

最后投影到 LLM hidden size：

\[
R_t
=P_{\mathrm{state}}H_{\mathrm{state}}
\in\mathbb R^{B\times16\times4096}.
\]

因此，无论命中 3、30 还是 300 条记录，输出始终是 16 个 State Token。可靠 `EMPTY` 使用显式
trainable `empty_record_embedding` 作为内部唯一 K/V，禁止对全 mask 直接 softmax 或产生 NaN；外部
审计仍保持真实 zero-width record 轴、record mask 全空且 selected mass=0。`UNSUPPORTED/INVALID`
同样避免全 mask 计算，但最终 hidden/state token 必须严格归零，`state_token_valid_mask=false`，不得
把“不知道”注入 LLM 或向 q_target/empty K/V 传播梯度。非空 `OK` 行只将全部
`selected_record_ids` 按 P11 canonical 顺序打包，不做 Top-K；即使 `N_s>N_ret` 且 selected 列不连续，
未选候选也不得进入 K/V。最终层 8-head 平均后的 cross-attention 权重为
`[B,16,max_N_ret]`，padding 权重严格为 0，selected mass=1。

每层固定三个 affine LayerNorm(`eps=1e-5`)、带 bias 的 self/cross Q/K/V/O、GELU
`512→2048→512` FFN；attention logits 与 masked softmax 使用 FP32，dropout=0。加上
`Q_state[16,512]`、empty embedding 和带 bias `512→4096` 投影后，State Resampler 精确为
14,722,048 参数（约 14.72M）。

### 10.2 State Token 的职责

State Token 给 LLM 提供：

- 当前问题相关对象与事件的语义摘要；
- 状态置信度和时间背景；
- 便于自然语言解释的软信息。

它不承担精确计数。精确整数仍由 Reader 从未压缩的类型化记录计算。

### 10.3 Deterministic State Reader

Reader 输入：

~~~text
hard operator
resolved TimeWindow
retrieved typed records
~~~

上述三项由同一个 `RetrieverOutput` 绑定保存；若调用方仍显式传入 operator/window，必须逐行与
Retriever provenance 完全一致，否则在算术前 fail closed。禁止用 O1-Snap 的检索结果改算
O1-Delta，或在检索后放宽时间窗口。Retriever 还必须保存 candidate typed-record snapshot；selected
record 的完整 payload、semantic tensor、ID 与 owner 必须逐字段匹配该 snapshot，Reader 每次读取前
重新验证，不能只凭相同 record ID 接受被替换的计数 payload。

Reader 使用固定算术：

| Operator | 精确读出 |
| :--- | :--- |
| O1-Snap | current_visible_count |
| O1-Delta | signed `current_visible_count - baseline_count`；第一版固定 baseline，不伪造任意历史 delta |
| O2-Unique | query_time 前 Confirmed 身份数 |
| O2-Gain | 闭区间内 first_seen 的唯一 Confirmed identity 数 |
| E1-Action / Transit | kind 匹配的完成事件数；history 用 cumulative count，其他窗口按 completion time 闭区间 |
| E2-Periodic / Episode | kind 匹配且 completion end 落在闭区间的完整区间数 |
| unsupported | 不生成伪造整数 |

输出结构至少包含：

~~~text
status: ok | empty | unsupported | invalid
exact_count: int | null
number_token_ids: [L_num]
selected_record_ids
operator
time_window
audit_fields
~~~

整数序列化必须使用 Reader 计算结果，不能在训练时偷换为 ground-truth count。

Reader 精确传播 Retriever status：`OK→OK`，可靠 `EMPTY→EMPTY + exact_count=0`，
`UNSUPPORTED/INVALID` 不产生整数或 number tokens。OK 算术可以得到 0；O1-Delta 可以得到负数，
不得 clamp。E1 retained history 最多 512 个 completion time；history 查询可使用 cumulative
`event_count`，但 bounded window 若可能覆盖已驱逐时间则返回 invalid，禁止猜测。

每个成功结果除 selected record IDs 外，还必须记录可复算的标量 operands：O1 current/baseline 与
`baseline_policy=fixed_baseline_v1`，O2 Confirmed/distinct/first-seen 匹配数，E1 cumulative/retained/
eviction/matched 数，以及 E2 completed/matched interval 数。Composer 接入前必须把同一个
`RetrieverOutput` 与 `ReaderResult` 交回 Reader 做确定性重算并要求整项相等；只让 number IDs 与被
篡改的 count 自洽不构成来源审计。

number text 固定为 Python/ASCII canonical signed decimal `str(exact_count)`，使用 pinned Qwen
tokenizer 且 `add_special_tokens=false`；IDs 必须 decode 回同一 canonical text 和整数，并重新编码为
同一 IDs。运行时同时固定 tokenizer class、vocab size 和四个 tokenizer-only 文件的 SHA256 manifest，
不得下载或加载 8B 权重来完成该审计。Reader API 不接收 answer、count、occurrence_times 或
counting subtype 标签，结果不可变，State Token 的任何修改也不能改变 exact_count。

## 11. LLM 输入拼接与 DeepStack

### 11.1 逻辑输入

送入 Qwen LLM 的信息包括：

- 原始 system/user/question token；
- adapted video token \(Z_t\)；
- 16 个 State Token \(R_t\)；
- Reader 生成的精确数字 token；
- 必要的边界和类型 special token。

简化后的 payload 长度为：

\[
L_{\mathrm{payload}}
=L_q+N_v+K_s+L_{\mathrm{num}},
\]

其中：

\[
N_v=392,\qquad K_s=16.
\]

Demo 中：

\[
L_{\mathrm{payload}}
=7+392+16+L_{\mathrm{num}}
=415+L_{\mathrm{num}}.
\]

真实 \(L_{\mathrm{total}}\) 还必须包含 chat template、system prompt、视觉边界和状态边界 special token。

P13 固定注册且按顺序保存 5 个 token：`<|state_start|>`、`<|state_pad|>`、
`<|state_end|>`、`<|number_start|>`、`<|number_end|>`。pinned tokenizer 对应 ID 为
151669–151673；Qwen text embedding 已有 151936 行，因此绝不能把模型 resize 缩小到 151674。
首次新增时，input embedding 与非 tied lm_head 分别使用 vision-start/video-pad/vision-end 三行
FP32 均值再 cast；checkpoint reload 不得重新初始化。OK/EMPTY 注入启用的 State/number 段，
UNSUPPORTED/INVALID 两段均省略；batch 使用左 padding，每行保留完整 token 位置审计。

### 11.2 实现约束

- video embeddings 继续使用 Qwen 原生 video placeholder mask 和 masked_scatter；
- State Token 使用固定数量的 state placeholder，再执行独立 masked_scatter；
- exact number 使用 Reader 序列化得到的真实 tokenizer token id；
- state 和 number payload 必须位于 assistant answer 之前；
- position id、attention mask 和 cache position 必须覆盖新增位置；
- State Token 不得被误标为 visual position；
- DeepStack 只在原 visual positions 注入，不作用到 State Token 或 number token；
- generate 的 prefill 只构建一次状态输入，自回归阶段不得重复更新 Bank 或 fast weights。

生产 prefill 必须保留完整 `input_ids` 给原生 Qwen：Composer 用同一 IDs、attention mask 和 video grid
预先审计 `[3,B,L] position_ids` 与 `[B,1] rope_deltas`，但不以 `inputs_embeds` 替代原生输入。
State embedding 仅在首个 token embedding forward 独立 scatter 一次；video 继续由 HF 原生 scatter。
Fast 后 Main splits 与未改动的三组 DeepStack tensor 由一次性 prepared provider 交回同一原生 forward，
不得重跑 ViT/Fast。当前 provider 只允许 greedy 单序列；beam expansion 未实现时必须 fail closed。

### 11.3 LLM 职责

LLM 负责：

- 根据问题组织答案；
- 使用 State Token 解释对象、身份和事件背景；
- 读取精确数字 token；
- 输出自然语言。

LLM 不负责：

- 从 392 个视频 token 重新累计长视频计数；
- 从 16 个 State Token 猜测精确整数；
- 覆盖 Reader 已确定的数字。

## 12. 无标签 TTT Loss

顶层固定为：

\[
\boxed{
L_{\mathrm{TTT}}
=L_{\mathrm{pred}}
+0.5L_{\mathrm{id}}
+0.5L_{\mathrm{event}}
}
\]

### 12.1 时序预测 \(L_{\mathrm{pred}}\)

v5 使用当前 chunk 内的 next-tubelet prediction，避免为了跨 chunk loss 长期保留上一 chunk
的 autograd graph：

\[
L_{\mathrm{pred}}
=\operatorname{MSE}
\left(
P(H_{t,:-1}),
\operatorname{sg}(H_{t,1:})
\right).
\]

Predictor 固定为：

~~~text
LayerNorm 768
→ Linear 768→1536
→ SiLU
→ Linear 1536→768
~~~

它预测下一个 tubelet 的时序状态，不预测像素或最终计数。Predictor 约 2.36M 参数；有效时间
位置少于 2 时该项为无效，不执行更新。

### 12.2 身份一致性 \(L_{\mathrm{id}}\)

相邻 chunk 交接时只持久化不入 Bank/checkpoint 的 detached `SoftOverlapSnapshot`。对可靠匹配的
同一对象，当前 chunk 的 prediction 是唯一带梯度的 source，上一 chunk snapshot 是固定 target：

\[
L_{\mathrm{id}}
=1-\cos
\left(
e_{t,j}^{\mathrm{current}},
\operatorname{sg}(e_{t-1,i}^{\mathrm{snapshot}})
\right).
\]

这一路径禁止保留上一 chunk 的 fast autograd graph，也禁止把 detached current prediction 与
previous prediction 比较；否则当前 functional update 无法获得有效梯度。position 必须相等，
timestamp 必须落在冻结容差内，且 source/target index 均不得重复。pair status 显式区分 matched、
mismatch、duplicate、low-confidence 与 invalid-source/padding。

若没有有效匹配：

\[
L_{\mathrm{id}}=0.
\]

匹配 mask 是有效性检查，不是学习型 Gate。

### 12.3 事件一致性 \(L_{\mathrm{event}}\)

\[
L_{\mathrm{event}}
=L_{\mathrm{E1-overlap}}
+L_{\mathrm{E2-overlap}}.
\]

E1 在相同重叠时间位置比较当前 prediction 与 detached previous snapshot target：

- eventness；
- completion；
- transition。

E2 使用同一 current-source/previous-target 方向比较：

- start；
- active；
- end；
- complete；
- phase distribution。

第一实现建议：

- 二值软输出使用 masked MSE；
- phase distribution 使用 detached previous target 的 KL；
- 所有项按有效时间位置求均值；
- hard FSM、整数计数器和 Bank 记录不参与反向传播。

P14 只定义强类型 source/target loss 与 snapshot 合同；跨 chunk snapshot 建立、pair matching 和
生命周期编排由 P17 实现。

### 12.4 为什么没有 O1 无标签 loss

当前可见对象数量会因进入、离开、遮挡和时间偏移合法变化。简单一致性容易把 O1 推向恒定
输出，因此第一版：

- O1 不进入 \(L_{\mathrm{TTT}}\)；
- O1 保留有标签 State Loss；
- O1 间接受益于 Fast Adapter 被其他无标签目标更新后的视觉特征。

所有无标签项先在每个视频 row 内组成：

\[
L_{\mathrm{TTT}}[b]
=L_{\mathrm{pred}}[b]+0.5L_{\mathrm{id}}[b]+0.5L_{\mathrm{event}}[b].
\]

无效项按零保留原权重，不对剩余项重新归一。唯一 batch scalar 是对至少一个分项有效的 row
求均值；禁止先对 pred/id/event 的不同有效 row 各自求均值再相加。E1/E2 同理先在 row 内相加，
再对 event-valid row 求均值。

## 13. 有标签训练目标

### 13.1 State Loss

每个样本只计算对应任务 Head 的监督：

\[
L_{\mathrm{task}}
\in
\{L_{\mathrm{O1}},L_{\mathrm{O2}},L_{\mathrm{E1}},L_{\mathrm{E2}}\}.
\]

Query 相关监督统一放入 State Loss：

\[
L_{\mathrm{state}}
=L_{\mathrm{task}}
+\lambda_{\mathrm{op}}L_{\mathrm{operator}}
+\lambda_{\mathrm{ret}}L_{\mathrm{retrieval}}
+\lambda_{\mathrm{time}}L_{\mathrm{time}}.
\]

其中：

- \(L_{\mathrm{operator}}\)：9 类 prototype 路由；
- \(L_{\mathrm{retrieval}}\)：记录级正负样本检索；
- \(L_{\mathrm{time}}\)：时间语义分类和合法数值窗口监督。

P14 只消费显式 dense target：O1 六字段使用 P15 target builder 提供的 pre-matched slot/mask，
不得从最终 count 反推或伪造 assignment；E2 使用四个 event BCE 与 phase CE，phase CE 是 hard
FSM 不入图时的 soft proxy。缺失标签的分量保持 invalid，不参与 reduction。

### 13.2 Answer Loss

\[
L_{\mathrm{answer}}
=\operatorname{CE}
(\text{generated answer},\text{target answer}).
\]

训练时必须分别记录：

- number token accuracy；
- 完整自然语言 answer accuracy；
- Reader exact count accuracy。

### 13.3 Meta-TTT Outer Loss

执行 Support chunk 的 inner update 后，在后续 Query 上计算：

\[
L_{\mathrm{outer}}
=L_{\mathrm{answer}}^{after}
+L_{\mathrm{state}}^{after}.
\]

最终：

\[
L_{\mathrm{total}}
=L_{\mathrm{outer}}
+0.1\operatorname{mean}(L_{\mathrm{TTT}}).
\]

本次 `compute_losses` 新计算的 support TTT 必须自动进入上述有效-row mean；调用方传入的 tuple
只表示额外/更早 support，空 tuple 不能导致当前 auxiliary loss 静默丢失。

真正训练“更新后是否更好”的是 after-update Query Loss；TTT auxiliary loss 只保证无标签目标本身
可学习。

## 14. 训练阶段

### Stage 0：基线与数据

- 建立原始 Qwen3-VL-8B 零样本基线；
- 完成 SVCBench 读取、视频采样、query_time 因果截断；
- 按 video_path 做 GroupKFold；
- 建立 demo tensor shape 单元测试；
- 固定评估协议和泄漏检查。

### Stage A：显式状态 Warm-up

关闭 Inner SGD，训练：

- Query Embedding Encoder；
- operator prototypes 和 Time Window Resolver；
- State Retriever 与 State Token Projector；
- 空间对象路和时间事件路；
- O1、O2、E1、E2；
- State Reader；
- 必要的 Qwen 参数或 LoRA。

目标：在没有 TTT 的情况下，状态模块与 Reader 已能正确工作。

### Stage B：单步 Meta-TTT

每条 episode：

~~~text
1 个 Support chunk
→ 计算无标签 L_TTT
→ 只对 fast weights 做一步 SGD
→ 后续 Query
→ Answer Loss + State Loss
~~~

先只开启 \(L_{\mathrm{pred}}\)，确认梯度范围、reset 和更新方向。

### Stage C：加入身份和事件一致性

依次加入：

1. \(L_{\mathrm{id}}\)；
2. \(L_{\mathrm{event}}\)；
3. 4 到 8 个连续 Support chunk；
4. 多个后续 query point。

每个增量必须独立做消融。

### Stage D：完整 8B 集成

- 接入真实 Qwen3-VL-8B checkpoint；
- 验证 Main Merger 插入点和 DeepStack 不变；
- 完成 FlashAttention、DeepSpeed 和多 GPU；
- 记录峰值显存、每 chunk 延迟和每视频更新时间；
- 在 clean 官方评测集上锁定最终结果。

## 15. 测试时协议

每个视频开始：

~~~text
reset fast weights to W0
reset SGD state
reset temporal cache
reset slot state
reset Identity Bank
reset event FSM
reset Reader audit state
~~~

每个 chunk：

~~~text
1. 严格裁剪到 query_time 以前
2. Qwen ViT + Main Merger
3. Fast Adapter 使用当前 W_t
4. 空间对象路和时间事件路
5. 四个 Head 产生 soft observation
6. no_grad 更新 State Bank / FSM
7. 若存在有效无标签目标，计算 L_TTT
8. 一步 SGD 得到 W_(t+1)
~~~

回答 query：

~~~text
1. Query Encoder 生成 q_target/q_operator/q_time
2. operator prototypes 生成 hard operator
3. Time Window Resolver 生成显式窗口
4. q_target 检索当前 Bank
5. Reader 计算 exact integer
6. Cross-Attention 生成 16 个 State Token
7. Composer 组装 question/video/state/number
8. Qwen LLM 生成自然语言答案
~~~

更新顺序必须保证当前 query 不受 query_time 之后信息影响。

## 16. 数据与防泄漏约束

测试时允许输入：

- video frames；
- question；
- 合法 query_time；
- 由问题文本显式给出的时间数值；
- 当前视频因果历史产生的模型状态。

测试时禁止输入：

- ground-truth answer；
- count；
- occurrence_times；
- counting_type；
- counting_subtype；
- query_time 之后的帧；
- 由完整视频离线计算的身份或事件记录。

训练时使用官方数据开发必须满足：

- 同一视频的全部问题与 query point 位于同一折；
- 阈值只在训练折或独立校准集确定；
- 官方 clean 测试视频不能进入外部预训练或阈值调参；
- 所有报告记录模型 revision、Git commit、数据划分和随机种子。

## 17. 推荐实现模块

模块按以下职责拆分；P3–P14 对应模块已通过工程门禁，P15 及后续模块仍按本计划施工：

~~~text
src/ttt_svcbench_qwen/
├── model.py
├── qwen_adapter.py
├── fast_ttt.py
├── state_encoder.py
├── observation_heads.py
├── state_bank.py
├── identity_bank.py
├── query_encoder.py
├── state_retriever.py
├── state_reader.py
├── input_composer.py
├── losses.py
├── functional_sgd.py
├── trainer.py
├── inference.py
└── config.py
~~~

职责边界：

- fast_ttt.py：Fast Adapter 与 fast parameter collection；
- losses.py：typed per-row TTT/State/Answer/Outer loss、mask/reduction 与指标；
- functional_sgd.py：typed row→一步 SGD、有限性/clip/reset、gradient mode 和 gradient/delta 审计；
- state_bank.py：hard state、事件记录和审计字段；
- identity_bank.py：Candidate/Confirmed/Hot Cache；
- query_encoder.py：target/operator/time；
- state_retriever.py：全记录阈值检索；
- state_reader.py：确定性算术和数字序列化；
- qwen_adapter.py：预计算 adapted Main/原 DeepStack 的一次性 provider 与 prefill-only State scatter；
- input_composer.py：新增 placeholder、mask、mRoPE/cache 审计和 token 拼接；
- model.py：只负责 DI、observe/answer/decode 生命周期和统一输出，不重复实现子模块逻辑。

## 18. 必须通过的验收测试

### 18.1 Demo 张量

1. \([1,16,3,224,224]\) 得到 grid_thw=[8,14,14]；
2. pixel_values_videos=[1,1568,1536]；
3. PatchEmbed 后为 \([1568,1152]\)；
4. Main Merger 后为 \([1,392,4096]\)；
5. 空间槽为 \([1,32,768]\)；
6. 时间状态为 \([1,8,768]\)；
7. O1/O2/E1/E2 输出符合本计划；
8. O2 identity 为 \([1,32,256]\)；
9. State Token 为 \([1,16,4096]\)；
10. payload 长度符合 \(L_q+N_v+K_s+L_{\mathrm{num}}\)。

### 18.2 更新边界

1. Inner SGD 后只有两个 fast matrix 变化；
2. 两个 fast matrix 均为 768 × 768，合计 1,179,648 个在线参数；
3. momentum 和 weight decay 为 0；
4. 每个 chunk 最多一步；
5. 当前 chunk 更新只影响下一 chunk；
6. 新视频恢复 \(W_0\)；
7. 非有限 loss/gradient 时跳过更新；
8. Shared Encoder 参数不变，但梯度能穿过它到 Fast Adapter；
9. hard Bank/FSM 不在 autograd 图中；
10. generate 自回归阶段不会重复更新。

### 18.3 状态与检索

1. O1/O2 共享 32 个活动槽但长期身份库可超过 32；
2. 超过 256 个 Confirmed 身份时扩容且旧记录不丢失；
3. Candidate 只在首次晋升时使 unique_count +1；
4. CPU store 与 GPU Hot Cache 换入换出不改变结果；
5. 不同 batch 和不同视频状态隔离；
6. \(N_s\) 动态变化不改变模型参数形状；
7. q_target 只查询当前合法 Bank 分区；
8. 默认无 Top-K，所有超过阈值记录可见；
9. 低置信度 unsupported 与真实空集合可区分；
10. 空检索时 State Token 有限且无 NaN；
11. 16 个 State Token 不是 Top-16 记录。

### 18.4 Reader 与输入

1. 每类 hard operator 的算术有独立单元测试；
2. exact count 与 number token 可双向审计；
3. number token 来自 Reader，不来自 ground truth；
4. DeepStack shape、mask 和注入顺序与原模型一致；
5. State Token 不进入 visual mask；
6. number token 不进入 visual mask；
7. 新增 token 的 attention mask、mRoPE position、rope delta 和 cache position 正确；
8. prefill 后至少两个 decode step 不重复 State scatter、Bank 写入或 fast 更新；
9. query_time 之后的帧不会进入 Bank、TTT 或答案。

## 19. 最小消融

| 编号 | 配置 | 目的 |
| :--- | :--- | :--- |
| A0 | 原始 Qwen3-VL-8B | 零样本基线 |
| A1 | 普通 SFT/LoRA，无状态模块 | 普通微调收益 |
| A2 | 四 Head + Bank + Reader，关闭 TTT | 显式状态收益 |
| A3 | A2 + \(L_{\mathrm{pred}}\) + SGD | 时序 TTT 收益 |
| A4 | A3 + \(L_{\mathrm{id}}\) | 身份一致性收益 |
| A5 | A4 + \(L_{\mathrm{event}}\) | 完整方案 |
| A6 | A5 去掉 exact Reader | 验证确定性计数必要性 |
| Q0 | 规则 Parser，仅诊断 | 固定问法上限 |
| Q1 | prototype operator，无语义检索 | 路由收益 |
| Q2 | Q1 + 全记录阈值检索 | q_target 检索收益 |
| Q3 | Q2 + 16 State Token | 语义摘要收益 |

最关键比较：

\[
\text{A5 vs A2}
\]

用于判断 TTT 是否真实增益；

\[
\text{A5 vs A6}
\]

用于判断精确 Reader 是否必要。

## 20. 评估与审计

至少报告：

- exact count accuracy；
- count MAE；
- answer accuracy；
- O1/O2/E1/E2 分任务结果；
- early/middle/late query 结果；
- operator 9 类准确率和 unsupported 召回率；
- operator Expected Calibration Error；
- State Retriever precision、recall、空检索率；
- identity duplicate rate 和 missed-new-identity rate；
- event duplicate/miss rate；
- active slot overflow 和 candidate overflow；
- 每视频 fast update 次数；
- 单 chunk forward/backward 延迟；
- 峰值 GPU/CPU 内存；
- 开启/关闭 TTT 的差值；
- 按视频长度、对象密度和遮挡程度分桶的结果。

## 21. 仍需实验决定

- Outer Training 使用全量微调、分阶段解冻还是 LLM LoRA；
- 高容量主干相对原 512 维主干的净收益；
- 活动槽 32 相对 16、48、64 的容量与开销权衡；
- State Token 16 相对 8、32 的摘要能力与上下文开销；
- 是否用训练折证据替换或增强 P4 已冻结的“双 pointer + 唯一候选受限 grammar” baseline；
- operator 和 record similarity 的最终校准阈值；
- E1/E2 overlap loss 使用 MSE、BCE 还是其他一致性距离；
- Confirmed 数量达到多大时启用 ANN 候选召回；
- Fast learning rate 在 \(3\times10^{-5}\)、\(10^{-4}\)、\(3\times10^{-4}\) 中的选择；
- 是否需要在后续版本让 DeepStack 也经过可适应路径。

这些选择必须通过训练折或独立校准集决定，不能根据官方测试结果反向调参。

## 22. 一句话定义

本项目是在 Qwen3-VL-8B 的 Main Visual Merger 与 video masked_scatter 之间加入一个只更新
两个 768 × 768 fast matrix 的单步 SGD 适配器；适配后的视觉 token 经 32 槽、768 维空间
对象编码器和 6 层、768 维因果时序编码器产生 O1/O2/E1/E2 观测，hard State Bank 保存
身份、事件、时间戳和整数状态，问题通过 4 层 Query Encoder 生成的
target/operator/time embedding 检索记录并选择确定性操作，Reader 计算精确数字，最后将
video、16 个 State Token 和 number 信息交给保持原 DeepStack 路径的 Qwen LLM 表达答案。
