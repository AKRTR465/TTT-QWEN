# State-TTT-Qwen3VL-8B 项目实施计划

> 规范版本：state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval  
> 修订日期：2026-07-13  
> 状态：DOCUMENT-ONLY / UNVERIFIED  
> 说明：本文描述目标实现。当前源码、运行 YAML 和测试尚未完整实现本计划。

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
  能回传到 \(W_0\)、RMSNorm 和两个慢投影。该模式用于后续 Meta-TTT 训练，不代表在线允许更新
  慢参数。

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
4. 对 W_t^(1)、W_t^(2) 做一步 SGD
5. 得到 W_(t+1)
6. W_(t+1) 从下一 chunk 开始生效
~~~

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

Demo 的 adapted video token 可恢复为：

\[
Z_t:
[1,392,4096]
\rightarrow
[1,8,7,7,4096].
\]

先将每个 merger token 投影到 768 维：

\[
[B,8,49,4096]
\xrightarrow{\operatorname{LayerNorm}+\operatorname{Linear}}
[B,8,49,768].
\]

空间对象路使用两阶段 Query-conditioned Recurrent Slot Attention。每个时间片依次读取 49 个
空间 token，并用上一时间片的槽作为当前初始化：

\[
A_{t,\tau}
=\operatorname{SlotStage}_2
\left(
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
- Pre-LayerNorm、SiLU、residual；
- q_target 通过 512 → 768 投影条件化共享槽初始化和 attention query；
- 两个 Stage 参数不共享，stage 内的 3 次 refinement 共享参数。

输出：

\[
A_t\in\mathbb R^{B\times K_a\times768},
\qquad K_a=32.
\]

Demo：

\[
A_t\in\mathbb R^{1\times32\times768}.
\]

\(K_a=32\) 的含义是“当前 chunk 最多并行处理 32 个活动对象槽”，不是：

- 整段视频最多只有 32 个身份；
- State Bank 只能保存 32 条记录；
- O2 Confirmed 身份库的容量。

实现约束：

- 槽初始化使用共享、query-conditioned 参数，不能使用独立的 \([32,768]\) 永久身份参数；
- batch 内使用 slot_valid_mask；
- 槽不足时记录 active_slot_overflow_count；
- 禁止静默覆盖高置信度活动对象；
- 默认 max_active_slots = 64；
- 后续消融比较 16、32、48、64，但正式基线固定 32。

### 5.2 时间事件路

每个时间 tubelet 对应 \(7\times7=49\) 个 merger token：

\[
[B,392,4096]
\rightarrow
[B,8,49,4096].
\]

先执行 LayerNorm + 4096 → 768 投影，再使用 q_target 条件化的多头空间注意力池化：

\[
[B,8,49,4096]
\rightarrow
[B,8,768].
\]

加入 tubelet 时间位置编码后，使用六层 Pre-Norm 严格因果 Transformer：

| 参数 | 值 |
| :--- | :--- |
| hidden size | 768 |
| layers | 6 |
| attention heads | 12 |
| head dimension | 64 |
| intermediate size | 3072 |
| dropout | 0.1 |
| mask | strict causal |
| temporal cache | 最近 64 个 tubelet |

输出：

\[
H_t\in\mathbb R^{B\times T\times768}.
\]

Demo 中 \(T=8\)：

\[
H_t\in\mathbb R^{1\times8\times768}.
\]

每个时间位置覆盖 2 个采样帧；文档中的“帧级状态”严格说是 tubelet 级状态。跨 chunk
temporal cache 按视频隔离，并在新视频开始时清空。

## 6. 四个 Observation Head

四个模块是任务解码器，不是 LLM Decoder，也不直接生成最终累计数字。v5 提高解码器容量，
但仍显著小于 Qwen 基座。

### 6.1 输出契约

| Head | 输入 | 输出 | 含义 |
| :--- | :--- | :--- | :--- |
| O1 当前数量 | \(A_t,q_{\mathrm{target}}\) | \([B,K_a,6]\) | 对象存在、目标、可见、进入、离开、置信度 |
| O2 身份 | \(A_t\) | identity \([B,K_a,256]\)；score \([B,K_a,2]\) | 身份向量、novelty、match confidence |
| E1 点事件 | \(H_t,q_{\mathrm{target}}\) | \([B,T,3]\) | eventness、completion、transition |
| E2 区间事件 | \(H_t,q_{\mathrm{target}}\) | event \([B,T,4]\)；phase \([B,T,4]\) | start、active、end、complete 和阶段分布 |

O1/O2 都读取完整的 768 维对象槽：

\[
\mathbb R^{768}\rightarrow\mathbb R^6,
\qquad
\mathbb R^{768}\rightarrow\mathbb R^{256}.
\]

这两个输出是并行投影，不是把 768 维切成 \(6+256\)。

### 6.2 O1 当前数量

O1 使用逐槽共享的三层 MLP，并以 q_target 生成 FiLM scale/shift：

~~~text
q_target [512] → Linear 512→1536 → FiLM(scale, shift)
A_t [B,32,768]
→ LayerNorm + FiLM
→ Linear 768→1024 → SiLU
→ Linear 1024→1024 → SiLU
→ Linear 1024→6
~~~

输出六个 logits：object、target、visible、enter、exit、confidence。该解码器约 2.63M 参数。

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

该解码器约 2.10M 参数。

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
该解码器约 9.58M 参数，不使用 BatchNorm。

hard 状态机使用双阈值、cooldown 和 Temporal NMS，防止一个持续多帧的事件重复计数。

### 6.5 E2 区间事件

E2 面向具有开始、持续、结束过程的事件，状态迁移为：

E2 使用两层 GRU 捕获持续阶段，再由双分支解码：

~~~text
H_t [B,T,768]
→ LayerNorm
→ 2-layer GRU, hidden=768
├→ Linear 768→4    start / active / end / complete
└→ Linear 768→4    phase logits
~~~

GRU hidden state 按视频隔离并在 reset 时清空。该解码器约 7.09M 参数。

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
| 空间对象编码器 | 24.88M |
| 时间事件编码器 | 48.49M |
| Query Embedding Encoder | 36.03M |
| O1 Decoder | 2.63M |
| O2 Decoder | 2.10M |
| E1 Decoder | 9.58M |
| E2 Decoder | 7.09M |
| Semantic Projector | 1.32M |
| TTT Temporal Predictor | 2.36M |
| 16-token State Resampler | 14.72M |
| Operator Router、Time Resolver、empty record | 约 0.14M |
| **新增模块合计** | **约 156.83M** |

新增模块约占 8B 基座的 2%。测试时真正变化的参数只有两个 768 × 768 fast matrix，即
1,179,648 个参数；其余新增参数只在 Outer Training 中学习，在线推理时冻结。

## 7. Structured State Bank

### 7.1 作用

State Bank 是当前视频的运行时结构化内存，不是模型参数，也不是外部数据库。

它按以下 key 隔离：

\[
(\text{video id},\text{question trajectory id},\text{head type}).
\]

每个新视频或问题轨迹结束后释放对应状态。

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

统一语义视图写作：

\[
E_{\mathrm{state}}
\in\mathbb R^{B\times N_s\times512}.
\]

其中：

- \(N_s\) 是当前轨迹中候选可查询记录数，随视频动态变化；
- 512 是固定语义检索维度；
- \(N_s\) 不是活动槽数量、身份容量或时间长度。

### 7.3 类型化 payload

| 类型 | 主要字段 |
| :--- | :--- |
| O1 | current count、baseline count、活动槽状态 |
| O2 Candidate | 256 维身份原型、观测次数、TTL、置信度 |
| O2 Confirmed | identity_id、256 维原型、first_seen、last_seen、observation_count |
| E1 | event_count、recent_event_times、cooldown |
| E2 | completed_count、phase、已完成区间、recent_event_times |

每条对象或事件记录额外保存归一化 512 维 semantic_embedding，表示“这条记录是什么”。

semantic embedding 由共享高容量投影器生成：

~~~text
object slot / event state [768]
+ learned head-type embedding [768]
→ LayerNorm
→ Linear 768→1024 → SiLU
→ Linear 1024→512
→ L2Norm
~~~

Semantic Projector 约 1.32M 参数。语义检索维度仍保持 512，避免 State Bank 的完整检索成本
随状态主干宽度同步增长。

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

### 7.5 梯度边界

- hard 状态更新统一位于 torch.no_grad；
- 写入 Bank 前对 soft embedding 执行 detach；
- Bank 不注册为 nn.Parameter；
- Bank 不进入 model.state_dict；
- Bank 不进入 Outer optimizer 或 Inner SGD；
- TTT loss 使用 detach 前的 soft 输出，保证梯度能到达 Fast Adapter。

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

### 9.2 相似度

\[
q_{\mathrm{target}}\in\mathbb R^{B\times512},
\qquad
E_{\mathrm{state}}\in\mathbb R^{B\times N_s\times512}.
\]

向量归一化后：

\[
s_i
=\cos(q_{\mathrm{target}},E_{\mathrm{state},i})
=q_{\mathrm{target}}^\top E_{\mathrm{state},i}.
\]

分数：

\[
S\in\mathbb R^{B\times N_s}.
\]

### 9.3 硬过滤

一条记录只有同时满足以下条件才进入 Reader：

~~~text
same video_id
same question_trajectory_id
head_type matches hard operator
record.valid = true
record time <= query_time
record intersects requested time window when required
semantic similarity >= calibrated threshold
~~~

第一版默认：

~~~text
record_similarity_threshold: 0.35
top_k: null
ann_enabled: false
~~~

阈值只能在训练折或独立校准集确定。固定 Top-K 会遗漏多个合法对象并导致静默少计，因此
第一版返回所有超过阈值的有效记录。

定义：

- \(N_s\)：过滤前候选记录数；
- \(N_{\mathrm{ret}}\)：最终选中记录数；
- \(0\le N_{\mathrm{ret}}\le N_s\)。

低置信度查询和真正的空集合必须区分：

- 查询与路由可靠、Bank 覆盖有效、无匹配记录：可以解释为计数 0；
- 查询或时间解析不可靠：返回 unsupported，不能把“不知道”当成 0。

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

因此，无论命中 3、30 还是 300 条记录，输出始终是 16 个 State Token。没有匹配记录时使用
显式 empty_record_embedding，禁止产生 NaN。完整 State Token Resampler 约 14.72M 参数。

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

Reader 使用固定算术：

| Operator | 精确读出 |
| :--- | :--- |
| O1-Snap | current_visible_count |
| O1-Delta | current_visible_count - baseline_count |
| O2-Unique | query_time 前 Confirmed 身份数 |
| O2-Gain | 时间窗口内 first_seen 身份数 |
| E1-Action / Transit | query_time 前符合类型的完成事件数 |
| E2-Periodic / Episode | query_time 前符合类型的完整区间数 |
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

### 11.2 实现约束

- video embeddings 继续使用 Qwen 原生 video placeholder mask 和 masked_scatter；
- State Token 使用固定数量的 state placeholder，再执行独立 masked_scatter；
- exact number 使用 Reader 序列化得到的真实 tokenizer token id；
- state 和 number payload 必须位于 assistant answer 之前；
- position id、attention mask 和 cache position 必须覆盖新增位置；
- State Token 不得被误标为 visual position；
- DeepStack 只在原 visual positions 注入，不作用到 State Token 或 number token；
- generate 的 prefill 只构建一次状态输入，自回归阶段不得重复更新 Bank 或 fast weights。

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

对相邻重叠 chunk 中可靠匹配的同一对象：

\[
L_{\mathrm{id}}
=1-\cos
\left(
e_{t-1,i},
\operatorname{sg}(e_{t,j})
\right).
\]

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

E1 在相同重叠时间位置比较：

- eventness；
- completion；
- transition。

E2 比较：

- start；
- active；
- end；
- complete；
- phase distribution。

第一实现建议：

- 二值软输出使用 masked MSE；
- phase distribution 使用 stop-gradient target 的 KL；
- 所有项按有效时间位置求均值；
- hard FSM、整数计数器和 Bank 记录不参与反向传播。

### 12.4 为什么没有 O1 无标签 loss

当前可见对象数量会因进入、离开、遮挡和时间偏移合法变化。简单一致性容易把 O1 推向恒定
输出，因此第一版：

- O1 不进入 \(L_{\mathrm{TTT}}\)；
- O1 保留有标签 State Loss；
- O1 间接受益于 Fast Adapter 被其他无标签目标更新后的视觉特征。

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

当前只保留实施计划；正式编码建议按以下职责拆分：

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
- functional_sgd.py：一步 SGD、有限性检查和 reset；
- state_bank.py：hard state、事件记录和审计字段；
- identity_bank.py：Candidate/Confirmed/Hot Cache；
- query_encoder.py：target/operator/time；
- state_retriever.py：全记录阈值检索；
- state_reader.py：确定性算术和数字序列化；
- input_composer.py：新增 placeholder、mask、position 和 token 拼接；
- model.py：只负责组合模块，不重复实现子模块逻辑。

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
6. 新增 token 的 attention mask、position id 和 cache position 正确；
7. query_time 之后的帧不会进入 Bank、TTT 或答案。

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
