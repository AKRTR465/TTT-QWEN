# 实施决策（v5 高容量版，P1 配置/类型/骨架阶段）

本文件记录已经冻结的 v5 边界。P0 已冻结规格和仓库基线；P1 已把运行 YAML、强类型配置、
运行时类型和推荐模块骨架迁移到 v5。所有真实模型、状态、训练和推理入口仍未实现，并明确
抛出 `NotImplementedError`。详细论证见
[ARCHITECTURE.md](./ARCHITECTURE.md)。当前规范版本为
`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。

## P1 已验证边界

1. `configs/model_state_ttt_8b.yaml` 是唯一活跃 v5 运行配置；旧 v3 的 512 bottleneck、16 slots、
   8 State Token 契约已从活跃配置和测试移除。
2. `config.py` 使用 immutable/extra-forbid Pydantic schema；固定维度、容量、9 个 operator、
   DeepStack 索引、单步 SGD 和约 156.83M 参数预算在启动前校验。
3. Retriever 的 0.35 只是 bootstrap；Time Resolver、operator、O1/O2/E1/E2 FSM 和匹配阈值
   均保留未校准状态。任一状态未校准时，配置拒绝正式评估。
4. P1 类型覆盖 VideoBatch、Query/TimeWindow、空间/时间输出与 cache、四类 soft output、
   typed records、Retriever、ReaderResult 以及完整 per-video runtime ownership。
5. 推荐模块当前只提供职责边界、类型和显式未实现入口；模块可导入不等于算法已实现。

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
   活动槽、最大64个，约24.88M参数。
7. 时间事件编码器使用q_target条件化空间池化和6层Pre-Norm严格因果Transformer：hidden
   size为768，12个attention heads，FFN为768→3072→768，缓存最近64个tubelet，约48.49M参数。
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
