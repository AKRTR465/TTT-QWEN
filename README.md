# ttt-svcbench-qwen

面向 SVCBench 的 Qwen3-VL-8B State-TTT 研究工程。

完整架构、训练协议和消融方案见 [ARCHITECTURE.md](./ARCHITECTURE.md)。当前对齐版本为
`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。
当前实施边界、验证结果与服务器迁移顺序见
[IMPLEMENTATION_PROGRESS.md](./IMPLEMENTATION_PROGRESS.md)。

正式 A2 → A5 四卡入口、数据 manifest、LLaMA-Factory bridge、checkpoint 与续训说明见
[docs/production-a2-a5.md](./docs/production-a2-a5.md)。

> 当前生产代码已固定为直接 A2 → A5：A2 全量解冻 Qwen 与状态路径；A5 支持无限数值在线
> Support、`K=8` 截断二阶梯度、重锚 `W0`、逐段 backward 与 episode 末单次 Outer AdamW。
> 数据 manifest、official weak sidecar、防泄漏、四卡 segment bucket/零权重 padding、ZeRO-2、
> 内置真实视频 runtime、LLaMA-Factory Trainer bridge、完整 checkpoint 与同阶段续训入口已实现
> 并通过本机定向测试。
> P0–P18 文档是历史 synthetic/tiny 门禁记录；正式长训尚未在本次 H200 预检中启动。
> H200 入口沿用 `play/projects/qwen3vl_dist_train` 的薄 wrapper + 公共 launcher
> 风格：A2 使用 `scripts/h200/launch_qwen3vl8b_ttt_a2_full4.sh`，A5 使用
> `scripts/h200/launch_qwen3vl8b_ttt_a5_k8_full4.sh`，两者都会自动进入 tmux。

## 当前固定条件

- 基座：Qwen/Qwen3-VL-8B-Instruct；
- Transformers：4.57.1；
- Python：3.12；
- 本机PyTorch：2.9.0，CUDA 12.8 wheel；
- 主插入点：Visual Merger主输出之后、video `masked_scatter`之前；
- 第一版不修改DeepStack；
- Fast TTT Adapter为4096→768→768→4096；在线仅更新两个768×768 fast矩阵，共1,179,648
  个参数，约1.18M；
- Fast Adapter使用`eps=1e-6`的RMSNorm、带bias的慢投影和Xavier-uniform `W0`；checkpoint
  保存`W0`而不保存per-video `W_t`，batched online forward要求每行状态storage相互隔离；
- Inner loop固定使用无momentum、无weight decay的单步SGD，不使用Surprise Gate；online 记录为
  `online_leaf`，Meta 路固定为显式 `meta_full_second_order`，不得静默 detach；
- 空间对象路使用两个参数不共享的768维Recurrent Slot Stage，默认32个活动槽；单一q投影和
  shared seed结合固定非持久sinusoidal slot code，attention先做slot轴竞争再按token归一，精确
  24,815,360参数；时间事件路使用6层、768维Pre-LN GELU因果Transformer，absolute sinusoidal
  使用显式global position id，Q/K/V/O带bias，LayerNorm eps为`1e-5`；
- P6的`required_slot_counts`只做preserve-existing/reject-excess容量审计，不表示已从视频识别
  真实对象；P13 已提供组合入口，P18 runtime 骨架已补齐跨视频 reset/release 与受管生命周期边界；
- O1/O2/E1/E2分别使用FiLM MLP、256维identity MLP、5层gated causal TCN和2层GRU；仅O1
  直接读取q_target，E1/E2读取P7已query-conditioned的H_t；O1固定`1+scale` FiLM，O2在FP32
  做L2归一化并对有效零范数回退到unit basis；
- E1使用63-tubelet感受野和无参66-position projected-history（62上下文+4 overlap）；E2使用
  单向batch-first GRU及5个rollback checkpoint重算4-position overlap；二者runtime都按
  video/trajectory/query signature隔离；
- 四个Head输出raw logits及debug probability/mask/timestamp/global position，invalid位置清零；
  在线只冻结Head参数，不使用`torch.no_grad()`或detach输入，hard state mutation仍禁止；
- 时间路使用含self且含当前位置总长64的同一full/chunk滑窗；cache保存六层逐层K/V并按
  video/trajectory/query signature隔离，overlap按global position replay/replace，默认detach下一
  chunk cache；主cache严格64，另有不扩大mask的3-position replay margin用于重算固定4-tubelet
  overlap；时间元数据保持FP32/FP64并在cache中统一为FP64；时间路精确48,438,272参数；
- P8精确参数为O1 2,632,710、O2 2,103,042、E1 9,584,643、E2 7,094,792；P9 Semantic
  Projector固定共享`768→1024→512` trunk和四个768维head embedding，精确1,316,864参数；
- Projector进入模型state_dict和Outer optimizer；hard Bank/FSM/runtime不注册参数/buffer、不进入
  state_dict或optimizer，写入统一no-grad+detach+clone，snapshot与模型checkpoint分离；
- O1六个bootstrap阈值为0.5且baseline显式set once；E1使用0.7/0.3 hysteresis及0.5秒
  cooldown/NMS；E2使用phase-gated三步FSM和0.5低事件re-arm；三者各维护单一aggregate record；
- 当前新增模块分项合计156.715683M（156,715,683），但在线变化的仍只有约1.18M fast参数；
- 无标签TTT loss仅由当前chunk内next-tubelet prediction、O2身份一致性和E1/E2事件一致性组成；
  overlap 一律为 current prediction→detached previous snapshot，先逐视频组成完整 loss，再只对
  update-valid rows 求均值；单视频 SGD 禁止消费跨视频 batch scalar；
- 问题不再通过关键词规则机械划分；Qwen input embeddings先经4096→768投影、无参sinusoidal
  position encoding和4层双向Transformer，再由三个768→1024→512 GELU输出头形成
  target/operator/time embedding；
- 计数操作由9个learned prototypes和初值1.0的可训练正温度进行语义路由；未校准时
  eval/inference显式落到unsupported；
- time embedding必须结合合法query_time、全局pointer和唯一候选受限grammar解析为确定性时间
  窗口；失败时不猜测、不clamp；
- State Bank记录按 hard operator 分区后通过 FP32 归一化 cosine 全量阈值检索，默认阈值 0.35、
  不设 top-k、ANN 关闭；最终整数仍由确定性 Reader 计算；
- 16个learned State Query经3层Perceiver Resampler汇总全部命中记录，生成16个4096维
  State Token；它们不是Top-16记录；
- Composer 固定注册 `<|state_start|>/<|state_pad|>/<|state_end|>/<|number_start|>/`
  `<|number_end|>`，本地 tokenizer ID 为 151669–151673；模型已有 151936 行，注册绝不缩表，
  新 input/lm_head 行由既有视觉边界三行 FP32 均值确定性初始化；
- batch 使用左 padding；video/state/number mask 两两互斥。Composer 用完整模板 IDs 预审
  Qwen mRoPE/`rope_deltas`，运行时保留 `input_ids`，State 仅在 prefill 独立 scatter 一次，video 与
  DeepStack 仍走 HF 原生路径；
- O2 Confirmed身份库从256开始按块动态增长；Candidate从64开始并设512安全上限；
- 每个新视频重置fast weights、SGD状态、时序缓存和State Bank；
- 测试时禁止使用答案、count、occurrence_times、counting_type和counting_subtype。
- 正式 A2 全量解冻 Qwen、状态模块与 `W0`，冻结 Predictor，禁用 Inner SGD；A3/A4 只保留作消融。
- 正式 A5 使用完整 `L_pred+0.5L_id+0.5L_event`，Support 不设上限，`K=8` 截断二阶梯度，
  每段重锚 `W0`，训练时不运行 static-W0 counterfactual。
- 四卡路径使用 BF16、TF32、SDPA、non-reentrant gradient checkpointing 与 ZeRO-2；真实 H200
  验收结果必须来自独立 run 目录，不能用本机 synthetic/tiny 结果替代。

## 本机安装

~~~powershell
uv sync --frozen
uv run python -m ttt_svcbench_qwen.config --config configs/model_state_ttt_8b.yaml
uv run pytest
~~~

uv会在根目录创建 .venv，并依据 uv.lock 安装依赖。

配置加载使用 Pydantic 强校验并拒绝未知键、旧 v3 固定值和非法组合。当前所有 FSM、匹配、
operator 及检索阈值仍带 `calibration_required` 或 `bootstrap_calibration_required` 状态，因此
`formal_evaluation_enabled` 必须保持 false，直至 P21 使用训练折或独立校准集完成冻结。

## 已验证实现与计划设计

| 范围 | 状态 |
| :--- | :--- |
| v5 YAML、完整解析、固定维度/容量/优化器校验 | P1 已实现并有契约测试 |
| Video/Query/Encoder/Observation/Record/Retriever/Reader/runtime 类型 | P1 已实现并有 shape/dtype/边界测试 |
| 推荐模块导入与职责边界 | P1 已实现；P3–P17 对应模块已通过各自工程门禁，P18 仅通过 runtime 骨架门禁，P19 及后续入口仍按阶段 fail closed |
| 数据 schema、防泄漏、因果切分、processor/query token、A0 runner | P2 工程门禁已通过；fold/A0 为明确标注的合成替代 |
| Qwen video boundary、Main Merger 插入点、DeepStack 保护 | P3 已实现；tiny/meta 工程契约已验证，真实 8B 留至 P19 |
| Query Encoder、9-prototype Router、Time Window Resolver | P4 已实现；本地结构/参数/offset/fail-closed 契约已验证，模型尚未训练、阈值尚未校准 |
| Fast Adapter、per-video fast state、参数边界 | P5 状态边界与 P14 typed row→functional SGD/gradient audit 已通过合成门禁；P18 只验证受管绑定和注入式更新边界，生产 updater 与真实 8B 留至 P19+ |
| P6 空间对象编码器 | 已通过本地合成张量工程门禁；真实视频/8B、语义对象 overflow 与端到端 runtime 仍留后续阶段 |
| P7 时间事件编码器 | 已通过本地合成张量工程门禁；逐层 KV、overlap replay margin、因果滑窗和 runtime 隔离均已验证 |
| P8 四类 Observation Decoder | 已通过本地合成张量工程门禁；输出/metadata、因果流式 replay、runtime 隔离、精确参数和 online freeze 均已验证 |
| P9 Semantic Projector、State Bank 与事件 FSM | 已通过小型合成张量工程门禁；统一 records、动态 padded view、O1/E1/E2 hard state、隔离、审计、snapshot 和梯度/持久化边界均已验证 |
| P10 Identity Bank | 已通过小型合成 identity 工程门禁；Candidate→Confirmed、CPU exact store、动态容量、非权威 Hot Cache 与离线指标边界均已验证 |
| P11 Embedding State Retriever | 已通过小型合成 Bank 工程门禁；row-wise owner/head 分区、FP32 cosine、因果/窗口/valid filters、无 Top-K 全量返回及结构化审计均已验证；0.35 阈值留 P21 校准 |
| P12 State Resampler 与 Deterministic Reader | 已通过小型合成 typed-record 工程门禁；16×4096 固定输出、FP32 masked attention、状态隔离、candidate/selected snapshot 完整性、8 operator 精确算术、record operands、signed number token 和本地 tokenizer manifest 均已验证 |
| P13 Input Composer 与模型编排 | 已通过 synthetic/tiny 工程门禁；固定 token 注册/初始化、变长 payload/左 padding、三类 mask、原生 mRoPE/rope delta/cache、预计算 adapted Main+原 DeepStack、Reader 重验、observe/answer/decode 生命周期均已验证 |
| P14 Loss 与 functional SGD | 已通过合成门禁；逐 row loss/valid/skip、full-second-order meta、finite/clip/reset、模块 gradient/delta 表均已验证 |
| P15 Stage A 显式状态 Warm-up | 已通过 synthetic/tiny A2 工程门禁；Qwen/Inner SGD 冻结，typed provenance target、四类 hard rollout、Reader/Qwen prefill、State+Answer Outer step、指标、checkpoint 与 fail-closed exit gate 已验证 |
| P16 Stage B 单步 Meta-TTT | 已通过 synthetic/tiny CPU A3 工程门禁；单 Support、`L_pred` inner step、next-only functional SGD、后续 Query outer loss、full-second-order `W0` gradient、reset/skip/leakage 审计均已验证 |
| P17 Stage C 多 Support Meta-TTT | 已通过 synthetic/tiny CPU A4/A5 工程闭环；identity/event objective 隔离、O2/E1/E2 overlap matcher、1/4/8 Support、多 Query 与 bounded CPU graph 回收已验证；独立 missed-new runner 指标和 GPU 显存证据待补 |
| P18 测试时协议与推理入口 | 已通过 synthetic/tiny CPU runtime 骨架门禁；reset/checksum/release、因果裁剪、真实 Fast Adapter state 绑定、Stage A runtime bridge、update stage 注入/next-only 边界、单次 prefill、immutable decode 与失败释放已验证；生产 `L_TTT`→functional SGD updater 和真实 8B `GenerationDriver` 未运行 |
| 真实 Qwen3-VL-8B、分布式、消融、校准、clean 评估 | P19–P22 计划设计，尚未运行；P16–P17 的 CPU 工程证据与 P18 runtime 骨架证据不得用于真实训练、性能或科学增益结论 |

## 环境变量

~~~powershell
Copy-Item .env.example .env
~~~

修改 .env 中的模型、数据和输出路径。源码中不得硬编码Windows或Linux绝对路径。

## Linux服务器

~~~bash
uv python install 3.12
uv sync --frozen
uv run pytest
~~~

FlashAttention、DeepSpeed和bitsandbytes不进入Windows基础锁文件。确认服务器CUDA、
编译器和PyTorch版本后再安装：

~~~bash
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation
uv pip install deepspeed bitsandbytes
~~~

服务器环境与CUDA 12.8不兼容时，应建立服务器专用lock或调整PyTorch index，不能静默改动
现有 uv.lock 后继续使用相同实验名称。

## 开发原则

1. 本机完成模块、FSM、loss、optimizer reset和小张量单元测试；
2. 服务器完成8B模型集成、视频训练和多GPU实验；
3. 代码通过Git同步，不使用scp覆盖工作目录；
4. 数据、基座权重、checkpoint和日志不进入Git；
5. 每个实验记录Git commit、uv.lock hash、模型revision、数据划分和完整命令。
