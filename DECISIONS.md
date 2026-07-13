# 实施决策（v5 高容量版，P0–P6 已通过）

本文件记录已经冻结的 v5 边界。P0 已冻结规格和仓库基线；P1 已把运行 YAML、强类型配置、
运行时类型和推荐模块骨架迁移到 v5；P2 已通过数据、因果预处理和合成 A0 工程门禁；P3 已实现
Qwen video boundary、Main Merger 插入点和 DeepStack 保护；P4 已实现 Query Encoder、Operator
Router 和 Time Window Resolver；P5 已通过 Fast Adapter 的本地合成张量工程门禁。P6 已通过
空间对象编码器的本地合成张量工程门禁。
时间状态、训练和推理入口仍按后续 Part 实现。详细论证见
[ARCHITECTURE.md](./ARCHITECTURE.md)。当前规范版本为
`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。

## P1 已验证边界

1. `configs/model_state_ttt_8b.yaml` 是唯一活跃 v5 运行配置；旧 v3 的 512 bottleneck、16 slots、
   8 State Token 契约已从活跃配置和测试移除。
2. `config.py` 使用 immutable/extra-forbid Pydantic schema；固定维度、容量、9 个 operator、
   DeepStack 索引、单步 SGD 和参数预算在启动前校验。P6 将空间路精确预算重算为
   24,815,360；P7 施工契约又将时间路冻结为 48,438,272，并把当前新增分项和重算为
   156.703632M。这些后续值不能倒推为 P1 当时已经验证的实现事实。
3. Retriever 的 0.35 只是 bootstrap；Time Resolver、operator、O1/O2/E1/E2 FSM 和匹配阈值
   均保留未校准状态。任一状态未校准时，配置拒绝正式评估。
4. P1 类型覆盖 VideoBatch、Query/TimeWindow、空间/时间输出与 cache、四类 soft output、
   typed records、Retriever、ReaderResult 以及完整 per-video runtime ownership。
5. P3 `qwen_adapter.py`、P4 `query_encoder.py` 和 P5 `fast_ttt.py` 已通过各自工程门禁。其余推荐
   模块当前只提供职责边界、类型和显式
   未实现入口；模块可导入不等于算法已实现。

## P2 合成退出决策

2026-07-14，用户明确批准：为节省本机空间，P2 缺失的视频、非 clean 训练集和 8B 权重可由
合成 fixture/predictor 完成工程链路验收，以便在全部代码门禁通过后开始 P3。

- 合成 fold 只验证分组、seed、manifest 和 video_id 零交集；不用于阈值或训练结论；
- 合成 A0 只验证禁用 State-TTT、指标聚合、prompt/generation 参数、失败案例和报告序列化；
- 合成 A0 的模型 ID 必须以 `synthetic/` 开头，指标不得称为原始 Qwen3-VL-8B 结果；
- 原始 Qwen3-VL-8B + 官方视频 A0 仍是 P19/P21/P22 的发布前必需证据；
- P2 工程门禁通过不改变上述科学验证要求。

## P3 已验证边界

1. wrapper 临时拦截内层 `Qwen3VLModel.get_video_features()`；插入点严格位于 Main Merger
   和 video `masked_scatter` 之间，不 hook image 共用的 `visual.merger`。
2. Main 输出保留原生 per-video split，并额外暴露 padding view、valid mask、原/合并 grid、
   token count 和 prefix offset；变长 batch 不把 padding 回传给 Qwen。
3. DeepStack 三组 packed tensor 保持原对象、顺序、dtype、device 和 mask，并按原实现进入
   decoder 0/1/2；ViT 8/16/24 不解释为 LLM 层号。
4. disabled 模式逐张量保持 Main、DeepStack 和 logits bitwise 等价；enabled 模式只变换 video
   Main，image-only、text-only 和 mixed image/video 路径受隔离测试保护。
5. 本地 loader 先用 `Qwen3VLConfig.from_pretrained(..., local_files_only=True)` 做轻量 fail-fast
   预检，通过后才允许加载权重；权重 loader 同样强制 local-only。
6. Qwen 参数默认冻结并保持 eval，但不切断 Adapter 梯度；hook 带互斥、禁止重入、异常恢复和
   stale capture 清理，且 inner owner 不被重复注册到 `state_dict`。
7. P3 证据只来自 Transformers 4.57.1 官方模块的 meta shape 和 tiny 随机权重端到端测试；没有
   下载视频或 8B 权重，不得表述为真实 8B 集成结果。P5 已验证本地 Fast Adapter 参数、
   runtime state 和 device/dtype 边界；P19 仍负责真实 8B hook、DeepStack 和分布式复验。

## P4 已验证边界

1. Query 只调用 pinned Qwen 的 input embedding table，不运行 36 层回答 decoder；生产入口必须从
   trusted canonical question 经 `tokenize_questions()` 构造。`source_fields` 是调用方声明，不能
   识别问题正文中伪装的答案文字，也不能替代 P2/P20/P22 的数据审计。
2. 主干固定为 `4096→768`、无参数 sinusoidal position encoding、4 层 Pre-LN 双向 Transformer
   （12 heads、FFN 3072、GELU、dropout 0.1）和 learned-attention pooling。attention 只屏蔽
   padding，显式 `is_causal=False`；最终 pool scorer 无 bias，主干/池化/三头共 36,026,112 参数。
3. target/operator/time 使用三个互不共享参数的 `768→1024→512` GELU head，分别 L2 normalize；
   没有额外 final LayerNorm，单独修改任一 head 不改变另外两路。
4. Router 使用 9 个归一化 512 维 prototype 和正的可训练温度；`log_tau` 初值 0 即 `tau=1.0`，
   Router 共 4,609 参数。校准前 threshold 为 null：训练保留 raw logits/argmax，eval/inference 的
   effective operator 一律 unsupported。
5. Time Resolver 为 `512→256→4` GELU mode MLP 加两个 `768→1` pointer，共 133,894 参数。
   全文受限 grammar 必须给出唯一中英文 recent/range 候选；全局非 padding pointer 必须按序完整
   覆盖首尾 numeric component，随后才允许确定性解析。多候选、局部 span、非法单位、反向或未来
   窗口均 fail closed。
6. `explicit_time_values` 是 canonical question 中各 numeric component 按出现顺序换算成秒的 tuple，
   只做逐值完整性核对，不从 count/occurrence_times 构造，也不直接决定窗口。compound recent 的
   component tuple 可为 `(120,3)`，Resolver 再确定性聚合为 123 秒。
7. 默认时间语义固定为 O1-Snap→now，O1-Delta/O2-Gain→recent（必须显式给出正 duration），
   O2-Unique/E1/E2→history。语法/窗口错误标记 invalid；未校准、低置信度或 mode 不一致标记
   unsupported；两者都强制 effective operator=unsupported，不 clamp、不交换端点、不猜秒数。
8. `QueryEncoder.forward()` 默认在 train 模式保留 raw 路径、eval 模式启用 gate；监督类型与 forward
   分离，pointer target 使用 inclusive start/end，且只允许成对非负索引或成对 `-100` ignore。
9. P4 仅验证工程结构、参数、失败策略和本地 pinned tokenizer offsets；没有训练 Router/Resolver，
   没有下载视频或 8B 权重，不能据此声明 operator/time 语义准确率。最终阈值仍由 P21 校准。

## P5 已验证工程边界

1. Fast Adapter 固定为 RMSNorm(`eps=1e-6`) + 带 bias 的 `4096→768` 慢投影、两块无 bias
   `768×768` fast matrix、SiLU、带 bias 的 `768→4096` 慢投影和比例 0.1 的残差；两块
   meta-learned `W0` 使用 Xavier uniform 初始化。
2. checkpointed 参数精确为 7,480,064：慢参数 6,300,416，meta-fast `W0` 1,179,648。
   online 参数集合只允许按稳定顺序返回每个视频的 `(W_t^(1),W_t^(2))`，精确为 1,179,648。
3. `state_dict` 保存 RMSNorm、两个慢投影及 `W0`，不保存 per-video `W_t`、active binding 或
   forward audit。reset 始终从当前 checkpointed `W0` 创建独立 snapshot 和独立 `W_t`，并把
   fast version、update count、skip count 清零。
4. 在线 batch 必须每行绑定一个 fast state；不同视频行的 `W_t` storage 相互隔离，padding 行为
   由 valid mask 控制且无效位置残差为零。不同 batch 行可以有不同 fast version，但同一 batch
   不能混合 online 与 differentiable state。
5. 显式传入 per-row `fast_state` 是 P14/测试的 functional 边界：online state 使用 detached leaf
   `W_t`，slow/`W0` 在该图中 detach，前向只保留输入和 `W_t` 的梯度。differentiable state 从
   `W0` 可微克隆并保留慢参数梯度，供后续 Meta-TTT outer training 使用；这不扩大在线可更新
   参数集合。
6. P18 正式在线推理使用 module-local `use_fast_state()` 受管生命周期兼容 P3 调用：拒绝 stale
   module grad，进入时冻结 checkpointed 参数的 `requires_grad`，退出时恢复，并保证
   exception-safe、非重入。video ID/batch order 对齐、并发串行化或实例隔离，以及受管调用必须
   产生 `last_audit.used_runtime_state=True` 的系统级证明仍属于 P18，P5 不提前认领。
7. forward audit 按 batch 行记录 fast version、update count、valid token count、两块 `W_t` norm、
   输入 norm 和实际缩放后残差 norm；这些值全部 detach，只用于审计，不参与 loss。
8. P5 只冻结 Adapter、state/reset、参数分组和数值边界；实际 loss、gradient clip 和一步 SGD 仍由
   P14 实现，正式受管推理协议仍由 P18 实现。P5 本地验收不得被表述为真实 8B、真实视频、
   在线收益、并发安全或 Meta-TTT 训练结果。

## P6 已验证工程边界

1. P6 显式接收 adapted embeddings、visual/tubelet bool masks、merged-grid metadata、
   `q_target`、逐样本 previous runtime 和 `required_slot_counts`；batch 内按每行 grid 恢复，允许
   异构 T/H/W，不能把 Demo 的 49 个空间位置写死。
2. 输入投影固定为 `LayerNorm(4096,eps=1e-5)+Linear(4096→768,bias=True)`。首次有效 tubelet
   使用 shared seed + 全模块唯一的带 bias `512→768` q projection +
   `sinusoidal(slot_index,dim)/sqrt(768)` 初始化；fixed code 是 `persistent=False` buffer，没有
   可学习缩放、不进入 `state_dict`。
3. 两个 Stage 各自拥有完整带 bias Q/K/V/O、三个 LayerNorm、GRUCell 和 SiLU FFN，参数和
   storage 不共享；每个 Stage 内三次 refinement 复用同一对象。Stage 2 复用同一个 X，以 Stage 1
   输出为初始化。
4. attention 先对 slot 轴 softmax 形成 token-to-slot 竞争，再沿有效 token 轴以 `eps=1e-8`
   归一。标准 MHA forward 的 token-axis softmax 不符合该算法；实现需要保留完整 Q/K/V/O 参数但
   自定义归一化和 mask。
5. occupancy confidence 是无参审计量：使用 token 再归一化前的有效 assignment mass / 有效 token
   数，并对 12 heads 求均值；无效槽为 0。`A_t` 和 confidence 都取最后一个有效 tubelet。
6. slot runtime 按 video/batch row 隔离，forward functional 地返回每行独立新 storage，
   不原地修改 previous state、不注册为参数、不进入 `state_dict`。`detach_runtime=True` 只 detach
   下一 chunk runtime，当前输出仍保留到 adapted embeddings/fast weights 的梯度；`False` 保图。
7. `required_slot_counts` 只执行容量审计：每个 forward 累计一次
   `max(required_slot_counts-32,0)` 并记录 event，仍计算原 32 槽，不替换、不扩容，required 值
   不得改变槽数值。它不是从视频预测的真实对象数。
8. 固定拓扑精确为 24,815,360 参数；fixed code、confidence、mask、runtime 和 capacity audit
   都是零参数。P6 验收时仍沿用时间路旧估算；P7 冻结后当前新增模块分项和改为
   156.703632M。
9. P6 只拥有空间路和可复用的无状态 grid helper；P7 拥有时间池化/因果 Transformer/cache，
   P8/P9 拥有对象语义与 hard state，P13/P18 拥有模型编排和受管推理生命周期。
10. P6 证据只来自合成张量工程检查；不得据此声称真实视频、真实 8B、语义 overflow 准确率、
    在线收益或端到端推理协议已通过。

## P7 已验收冻结边界

1. 时间路显式接收 tubelet timestamps、global position ids、合法 query_time、q_target，以及逐行
   video/trajectory/query signature owner；不能只依赖上游已经做过因果裁剪。
2. 空间池化固定为 `LayerNorm(4096,eps=1e-5)+Linear(4096→768,bias=True)`，再以带 bias
   `512→768` q_target projection 和带 bias Q/K/V/O 的 12-head MHA 生成每个 tubelet 的 768 维
   pooled state；无效空间 token 和全无效 tubelet 必须安全屏蔽。
3. tubelet 位置编码固定为无参数 absolute sinusoidal，并显式使用 global position id。chunk-local
   位置从 0 重启、learned 64-position table、modulo 或 clamp 都不符合该契约。
4. 时间主干固定为六个参数不共享的 Pre-LN layer：hidden 768、12×64 attention、
   `768→3072→768` GELU FFN、dropout 0.1、LayerNorm eps `1e-5`、Q/K/V/O 带 bias，末尾没有额外
   final LayerNorm。
5. causal attention 包含 self，并把当前位置计入总共 64 个位置的窗口：
   `q_position-63 <= key_position <= q_position`。full/no-cache 与 chunk/cache 两条路径必须使用同一
   mask；只在 chunk 结束后截 cache 不足以证明等价。
6. temporal cache 的主状态保存最近 64 个位置的六层 K/V、最终 hidden、FP64 timestamp、valid、
   absolute position、total_seen 和 owner metadata。最终 hidden 只用于输出兼容/审计，禁止作为
   下一 chunk 的六层输入；输入 timestamp/query_time 可为 FP32/FP64，不与模型 dtype 绑定。
7. P2 overlap 固定为 4 个 tubelet。runtime 额外保存紧邻主 cache 之前最多 3 个位置的 replay-only
   per-layer K/V margin；它不计入主 cache length 且不扩大 64-token mask，只用于在主 cache 满后仍
   能用当前 adapted token 重算 overlap。global position replay/replace 删除同位置旧 K/V 后替换，
   不得重复追加。
8. cache ownership 固定为 `(video_id, trajectory_id, query_signature)`；新 owner 必须 reset，batch
   row 交换、owner 模糊匹配或共享可变 runtime 都必须拒绝。
9. 默认 `detach_cache=True`，只 detach/clone 下一 chunk runtime，当前输出保留梯度；训练态
   dropout 不承诺 full/chunk 逐元素等价，等价门禁必须在 `eval()` 下执行。
10. 固定拓扑精确为 48,438,272 参数；position、mask、cache、overlap 和 owner audit 都是零参数。
    当前新增模块分项和为 156,703,632。

## 已固定

1. 基座使用Qwen3-VL-8B-Instruct，不使用4B版本。
2. 主视觉维度为4096；第一版在Visual Merger主输出之后、video `masked_scatter`
   之前插入State-TTT模块。
3. 原始DeepStack路径保持不变。
4. Fast Adapter使用4096→768→768→4096残差结构，固定残差比例为0.1。
5. 测试时只允许更新Fast Adapter中的两个768×768 fast矩阵，共1,179,648个在线参数，约1.18M；
   包含冻结慢投影的完整Adapter约7.48M。
6. 空间对象编码器使用两阶段Query-conditioned Recurrent Slot Attention：hidden size为768，
    12个attention heads，FFN为768→3072→768，每阶段执行3次共享参数refinement；默认32个
    活动槽、最大64个，精确24,815,360参数。两个Stage参数不共享，单一q projection、shared seed、
    fixed non-persistent slot code和经典slot竞争归一化按上述P6候选边界执行。
7. 时间事件编码器使用q_target条件化空间池化和6层Pre-Norm严格因果Transformer：hidden
   size为768，12个attention heads，GELU FFN为768→3072→768；无参absolute sinusoidal使用显式
   global position id，含self的滑窗总长64，逐层KV主cache配合3-position replay margin执行
   4-tubelet overlap replay/replace，并按video/trajectory/query signature隔离。完整模块精确
   48,438,272参数。
8. 四个任务模块是高容量Observation Decoder，不直接生成最终累计数字：
   - O1使用FiLM条件化的768→1024→1024→6 MLP，约2.63M参数；
   - O2使用768→1024→1024共享trunk，输出256维identity和2维score，约2.10M参数；
   - E1使用5层、512通道gated causal TCN，约9.58M参数；
   - E2使用2层、hidden size 768的GRU和两个4维输出分支，约7.09M参数。
9. Query Encoder先将问题token从4096投影到768，再经过4层双向Transformer（12 heads、
   FFN 3072）和learned-attention pooling；三个独立的768→1024→512输出头分别生成target、
   operator和time embedding，完整模块约36.03M参数，不使用关键词规则机械划分任务。
10. operator embedding与9个learned prototypes做归一化余弦路由；8个合法操作之外保留
    unsupported，低置信度不得强制分配到合法计数类型。
11. target embedding用于检索State Bank中的语义记录；默认不设top-k，防止因截断候选而静默
    少计。相似度阈值和unsupported阈值只能在训练折或外部校准集确定。
12. time embedding只表示时间语义；精确窗口必须由Time Window Resolver结合合法query_time
    和问题中显式数值解析为start/end，解析失败时返回unsupported。
13. 16个learned State Query通过3层Perceiver Resampler汇总全部命中记录并生成固定16个4096维
    State Token；它们不是Top-16记录，只提供语义摘要。
14. 显式状态机维护对象槽、带语义embedding的身份库、事件日志、阶段和整数计数。
15. State Reader根据硬operator、显式时间窗口和检索结果确定性计算数字；embedding只决定
    “查什么”，不负责猜测最终整数；LLM负责读取和表达。
16. 在线更新只作用于fast weights；共享状态编码器参与梯度路径但参数冻结，硬状态更新不反向传播。
17. 默认因果顺序为：当前chunk观测和状态更新完成后再执行TTT更新，更新从下一chunk生效。
18. 每个新视频重置fast weights、SGD状态、时序缓存和全部State Bank。
19. 测试时不得把答案、count、occurrence_times、counting_type或counting_subtype输入查询编码器、
    检索器或更新loss。
20. 第一版不使用学习型Gate或Surprise Gate，也不改造DeepStack。

## 状态容量

1. O1/O2共享默认32个活动对象槽，最大配置为64；它们是当前chunk的固定GPU计算工作集，
   不是整段视频的身份上限。
2. O2 Confirmed身份库初始容量为256，之后按256个位置分块增长，不设置语义硬上限。
3. O2 Candidate库初始容量为64，可动态增长，但受TTL、置信度清理和512安全上限约束。
4. Confirmed完整记录默认保存在CPU FP32分块张量中；容量为256的GPU Hot Cache只负责加速，
   不能决定计数。
5. E1/E2最近事件时间戳容量固定为512。
6. Candidate只有第一次晋升为Confirmed时才使`unique_count`加1；Confirmed记录不得因扩容或缓存换出而被覆盖。
7. 不同视频和不同batch样本的Identity Bank彼此隔离，并在轨迹结束后释放。

## TTT训练原则

1. Stage A关闭Inner SGD，先训练Query Embedding Encoder、operator prototypes、Time Window
   Resolver、State Retriever、State Token Projector、共享状态编码器、四个Observation Head、
   State Reader和必要的Qwen参数。Operator cross entropy、record retrieval loss和time-window
   loss统一计入State Loss，不新增顶层loss组。
2. Stage B进行单步Meta-TTT：support只使用无标签inner loss，后续query使用有标签outer loss。
3. Stage C逐步扩展到4到8个连续support chunks、每个有效chunk一步SGD和多个后续query points。
4. Inner loop必须与测试阶段使用相同的优化器、更新步数、reset、因果更新顺序和有效性检查。
5. 离散FSM使用硬状态做rollout，同时使用软状态代理提供训练梯度。
6. 无标签Inner loss固定为：

   \[
   L_{\mathrm{TTT}}
   =L_{\mathrm{pred}}+0.5L_{\mathrm{id}}+0.5L_{\mathrm{event}}.
   \]

7. `L_pred`固定为当前chunk内的next-tubelet prediction：
   `MSE(P(H_t[:,:-1]), stop_gradient(H_t[:,1:]))`；有效时间位置不足2时该项无效，不跨chunk
   保留autograd graph。
8. O1不进入无标签TTT loss，但必须保留对应的有标签State Loss。
9. 顶层Outer loss固定为：

   \[
   L_{\mathrm{total}}
   =L_{\mathrm{answer}}^{after}
   +L_{\mathrm{state}}^{after}
   +0.1\operatorname{mean}(L_{\mathrm{TTT}}).
   \]

10. 真正训练TTT更新方向的是更新后query的Answer Loss和State Loss；TTT Auxiliary Loss只维持无标签目标可学习。
11. 第一版不加入Gate Loss、harmful-update loss、improvement margin、fast-weight drift、update-norm或KL retention等额外正则项。

## 在线优化器

Fast TTT inner loop固定使用SGD：

- learning rate = 1.0e-4；
- momentum = 0；
- weight decay = 0；
- 每个有效chunk最多更新一步；
- gradient norm clip = 1.0；
- 每个新视频从meta-learned `W0`重新初始化fast weights；
- 无有效TTT项、有效帧不足、loss非有限或梯度非有限时跳过更新。

学习率只允许在训练折或外部校准集搜索`3e-5`、`1e-4`和`3e-4`。第一版不比较其他在线优化器，
优化重点放在TTT目标有效性、梯度范围、更新幅度和状态任务对齐上。

## 本机与服务器职责

- 本机：模块实现、配置、FSM、loss、functional SGD、reset及小张量单元测试。
- 服务器：Qwen3-VL-8B真实加载、视频训练、FlashAttention、DeepSpeed、多GPU与正式评估。
- 代码通过Git同步；数据、模型、checkpoint和日志不进入Git。
- 本机和服务器分别维护平台相关环境，实验必须记录uv.lock、Git commit、模型revision和数据划分。

## 尚待实验决定

- Outer训练采用全量微调、分阶段解冻还是LLM LoRA；
- 是否在后续版本改造DeepStack；
- O1/O2/E1/E2状态机阈值；
- operator unsupported阈值和State Bank记录相似度阈值；
- embedding检索使用纯阈值、阈值加精确回退还是后续ANN候选召回；
- O2精确搜索何时需要ANN加速；
- 外部训练数据与官方clean评测协议。
