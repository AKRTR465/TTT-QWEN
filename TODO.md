# State-TTT-Qwen3VL-8B v5 施工 TODO

> 对齐源：[ARCHITECTURE.md](./ARCHITECTURE.md)  
> 规范版本：`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`  
> 生成日期：2026-07-13  
> 文档状态：施工分解 / P0–P6 已通过，P7 允许开始
> 总原则：本文件只描述施工顺序和验收门禁；任何勾选都必须有代码、测试、日志或实验记录作为证据。

## 0. 使用方法

### 0.1 勾选规则

- `[ ]`：未开始；
- `[~]`：实施中，但尚未通过本阶段全部验收；
- `[x]`：已经完成，并能定位到对应代码、测试和验证记录；
- `[!]`：阻塞，必须在条目下记录阻塞原因、责任人、解除条件和日期；
- `[?]`：实验决策，不能凭直觉勾选，必须关联训练折或独立校准集结果。

### 0.2 阶段门禁

- [ ] 严格按 P0 → P22 推进；依赖阶段未验收时，不得把后续阶段标记为完成。
- [ ] 每个 P 开始前保存可复现基线：Git commit、配置快照、`uv.lock` hash、模型 revision、数据划分和随机种子。
- [ ] 每个 P 只引入本阶段职责；不得把 Retriever、Reader、FSM、训练循环等逻辑偷塞进 `model.py`。
- [ ] 每个 P 完成后先跑该阶段定向测试，再跑当前全部 `pytest`、`ruff` 和 `mypy`。
- [ ] shape、mask、dtype、device、梯度、状态隔离和 query-time 因果性必须分别测试，不能只凭最终 loss 正常判断实现正确。
- [ ] 文档、配置、实现和契约测试必须同步；若冲突，以 `ARCHITECTURE.md` 当前规范为准并先停止施工。
- [ ] 官方 clean 测试集只用于最终锁定评估，不得用于阈值、学习率、FSM 或模型结构选择。
- [ ] 任一阶段出现数据泄漏、跨视频状态污染、DeepStack 路径变化或非 fast 参数在线变化，立即回退到上一通过门禁的版本。

### 0.3 “阶段完成”的统一定义

一个阶段只有同时满足以下条件才可标记为 `[x]`：

- [ ] 本阶段所有固定契约均有实现；
- [ ] 本阶段所有 TODO 已完成，或明确移入“实验待定”且不影响固定基线；
- [ ] 正向、边界和失败路径测试齐全；
- [ ] 所有新增张量都有 shape、dtype、device 和 mask 断言；
- [ ] 所有运行时状态都有创建、更新、序列化/审计、reset 和释放路径；
- [ ] 没有答案字段或未来帧泄漏；
- [ ] 没有未解释的 NaN、Inf、silent overwrite、silent truncation 或 silent fallback；
- [ ] 代码质量检查通过；
- [ ] 产物和验证命令记录在阶段日志中。

---

## 1. 全局不可变契约

以下条目贯穿全部 P；任何实现不得偏离。

### 1.1 职责边界

- [ ] Qwen3-VL ViT 和 Main Visual Merger 负责生成视觉表示。
- [ ] Fast TTT Adapter 负责单视频内部的少量在线适应。
- [ ] 空间对象路和时间事件路负责产生对象、身份和事件的连续观测。
- [ ] Structured State Bank 负责保存“发生了什么”：类型化记录、时间戳、身份、FSM 和精确整数状态。
- [ ] Query Encoder 只决定 target、operator 和 time 语义。
- [ ] embedding 只负责“查什么”和“路由到哪里”。
- [ ] Deterministic State Reader 使用未压缩 hard records 做精确算术。
- [ ] 16 个 State Token 只向 LLM 提供语义摘要和解释背景，不承担精确计数。
- [ ] Qwen LLM 只读取 video/state/number payload 并组织自然语言答案，不重新累计长视频计数。

### 1.2 基座与插入点

- [ ] 基座固定为 `Qwen/Qwen3-VL-8B-Instruct`。
- [ ] 启动时断言 Vision depth=27、Vision hidden=1152、Vision heads=16。
- [ ] 启动时断言 patch size=16、temporal patch size=2、spatial merge size=2。
- [ ] 启动时断言 Visual Merger output=4096。
- [ ] 启动时断言 DeepStack visual indexes=[8,16,24]。
- [ ] 启动时断言 LLM layers=36、LLM hidden=4096。
- [ ] State-TTT 只插在 Main Visual Merger 主输出之后、video `masked_scatter` 之前。
- [ ] 禁止重新引入过时的 `pooler_output` 接口假设。
- [ ] DeepStack 继续从 ViT block 8、16、24 取特征，并保持原 merger、shape、mask 和注入顺序。
- [ ] DeepStack 不经过 Fast Adapter、不进入 State Bank、不计入新增输入 token 长度、不参与在线 SGD。

### 1.3 维度与参数总账

| 模块 | 固定结构或形状 | 预算/说明 |
| :--- | :--- | ---: |
| Demo 原始视频 | `[1,16,3,224,224]` | 仅用于验收，不是数据集固定长度 |
| `video_grid_thw` | `[8,14,14]` | temporal patch=2，spatial patch=16 |
| `pixel_values_videos` | `[1,1568,1536]` | 1568 个展平 tubelet |
| ViT 输出 | `[1568,1152]` | 27 层保持 token 数 |
| Main Merger 输出 | `[1,392,4096]` | 空间 `2×2` merge，时间长度仍为 8 |
| Fast Adapter | `4096→768→768→4096` | 完整约 7.48M |
| 在线 fast matrices | 两个无 bias `768×768` | 1,179,648，约 1.18M |
| 空间对象编码器 | 2-stage slot attention，dim=768，32 slots | 精确 24,815,360，约 24.81536M |
| 时间事件编码器 | 6-layer causal Transformer，dim=768 | 约 48.49M |
| Query Encoder | `4096→768` + 4-layer Bi-Transformer + 3 heads | 约 36.03M |
| O1 | FiLM + `768→1024→1024→6` | 约 2.63M |
| O2 | `768→1024→1024`，identity=256，score=2 | 约 2.10M |
| E1 | 5-layer gated causal TCN，channels=512 | 约 9.58M |
| E2 | 2-layer GRU，hidden=768，双 4 维 head | 约 7.09M |
| Semantic Projector | `768→1024→512` | 约 1.32M |
| TTT Predictor | `768→1536→768` | 约 2.36M |
| State Resampler | 16 queries，3 layers，`512→4096` | 约 14.72M |
| Router/Resolver/empty record | 9 prototypes 等 | 约 0.14M |
| 新增模块总计 | 不含 8B 基座 | 当前分项和 156.75536M，约 156.76M |

- [ ] 参数审计同时报告：新增模块约占 8B 基座的 2%，但在线实际变化的只有约 1.18M fast 参数。

### 1.4 在线更新硬边界

- [ ] 测试时只允许 `W_t^(1)` 和 `W_t^(2)` 两个 fast matrix 变化。
- [ ] `P_in`、`P_out`、RMSNorm、ViT、Merger、DeepStack、状态编码器、四个 Decoder、Query、Retriever、Reader 和 LLM 在线冻结。
- [ ] 冻结模块仍保留到 fast weights 的 autograd 路径。
- [ ] Inner optimizer 固定为 SGD，默认 learning rate=`1e-4`、momentum=0、weight decay=0。
- [ ] 每个有效 chunk 最多一步，gradient norm clip=1.0。
- [ ] 顺序固定为“当前 `W_t` 观测与写状态 → 计算 `L_TTT` → 一步 SGD → `W_(t+1)` 从下一 chunk 生效”。
- [ ] 每个新视频从 meta-learned `W0` 重置，严禁跨视频共享 fast state。
- [ ] 无有效 TTT 项、时间位置不足、loss 非有限、gradient 非有限或裁剪后仍不可用时跳过更新并写审计记录。

### 1.5 第一版禁止项

- [ ] 不实现 Surprise Gate 或学习型更新 Gate。
- [ ] 不使用 Inner AdamW、Muon、momentum SGD 或每 chunk 多步更新。
- [ ] 不让 LLM 或连续 embedding 直接回归最终累计整数。
- [ ] 不做固定 Top-K 状态检索。
- [ ] 不给 O1 添加无标签一致性 loss。
- [ ] 不改造 DeepStack。
- [ ] 不启用 ANN 向量数据库。
- [ ] 不堆叠 harmful-update、margin、drift、update-norm、KL retention 等额外正则。
- [ ] 不使用 query_time 之后的帧。
- [ ] 不在 generate 自回归循环中重复执行 Bank 更新或 TTT 更新。

### 1.6 数据与防泄漏红线

- [ ] 测试时只允许 video frames、question、合法 query_time、问题中显式时间数值和当前视频因果历史状态。
- [ ] ground-truth answer、count、occurrence_times、counting_type、counting_subtype 禁止进入 Query、Retriever、Reader、Bank、TTT loss 或生成输入。
- [ ] query_time 之后的帧不得进入 Bank、TTT 或答案路径。
- [ ] 不得使用完整视频离线生成的身份或事件记录。
- [ ] 同一 video 的所有问题和 query point 必须在同一数据折。
- [ ] 阈值只能由训练折或独立校准集确定。

---

## 2. P0–P22 总览

| 阶段 | 主题 | 直接依赖 | 主要退出产物 |
| :--- | :--- | :--- | :--- |
| P0 | 规格冻结与仓库基线 | 无 | 规格锁、基线日志、覆盖矩阵 |
| P1 | v5 配置、类型契约与模块骨架 | P0 | 配置 schema、模块空壳、契约测试 |
| P2 | 数据、视频预处理、因果切分与 A0 基线 | P1 | Dataset/Collator、Demo tensor、零样本基线 |
| P3 | Qwen3-VL 接口、Merger 插入点与 DeepStack 保护 | P1–P2 | `qwen_adapter.py`、原模型等价测试 |
| P4 | Query Encoder、Operator Router、Time Resolver | P1–P2 | 三个 query embedding、hard operator、TimeWindow |
| P5 | Fast TTT Adapter 与 fast 参数收集 | P1、P3 | `fast_ttt.py`、1.18M 参数边界 |
| P6 | 空间对象编码器 | P4–P5 | `A_t[B,32,768]` |
| P7 | 时间事件编码器 | P4–P5 | `H_t[B,T,768]` 和 causal cache |
| P8 | O1/O2/E1/E2 Observation Decoder | P6–P7 | 四类 soft observation |
| P9 | Semantic Projector、类型化 State Bank 与事件 FSM | P8 | hard records、事件审计、梯度隔离 |
| P10 | Identity Bank 动态容量与 Hot Cache | P8–P9 | Candidate/Confirmed 生命周期 |
| P11 | Embedding State Retriever | P4、P9–P10 | 无 Top-K 的阈值检索 |
| P12 | 16-token Resampler 与 Deterministic Reader | P4、P11 | State Token、exact count、number token |
| P13 | Input Composer 与完整模型编排 | P3、P5–P12 | Qwen prefill payload、`model.py` |
| P14 | TTT/State/Answer Loss 与 functional SGD | P5–P13 | Loss、一步可微更新、梯度审计 |
| P15 | Stage A 显式状态 Warm-up | P13–P14 | 无 TTT 的可用状态系统 |
| P16 | Stage B 单步 Meta-TTT | P15 | 单 Support + `L_pred` 元训练 |
| P17 | Stage C 身份/事件一致性与多 Support | P16 | 完整 `L_TTT` 元训练 |
| P18 | 测试时协议与推理入口 | P13–P17 | `inference.py`、逐视频 reset |
| P19 | Stage D 真实 8B、FlashAttention、DeepSpeed、多 GPU | P18 | 可扩展训练/推理作业 |
| P20 | 全量验收与回归契约 | P0–P19 | shape/update/state/reader/leakage 测试 |
| P21 | 消融、校准与未决实验 | P20 | A0–A6、Q0–Q3、决策记录 |
| P22 | 最终评估、审计与发布门禁 | P21 | clean 结果、完整审计包 |

---

## P0. 规格冻结与仓库基线

### 目标与依赖

把 `ARCHITECTURE.md` 变成可执行、可追踪的唯一规格，并确认当前仓库仍处于 DOCUMENT-ONLY 起点。无前置依赖。

### 实施前注意事项

- P0 开始时仓库只有环境检查和旧 v3 YAML/契约测试，不代表 v5 已实现。
- P1 迁移前 `configs/model_state_ttt_8b.yaml` 包含 512 bottleneck、16 slots、8 State Token 等旧值；该基线不得用于 v5 实验。
- `__pycache__` 或历史字节码不能作为源码存在的证据。
- 本阶段不写模型逻辑，只冻结边界和验证起点。

### 实施过程 TODO

- [x] 记录 `ARCHITECTURE.md` 规范名、修订日期和文件 hash。
- [x] 记录 Python 3.12、PyTorch 2.9.0、Transformers 4.57.1 和 CUDA 环境。
- [x] 记录本机/服务器各自职责和平台差异。
- [x] 运行并保存基线命令：`uv sync --frozen`、`uv run pytest -q`、`uv run ruff check .`、`uv run mypy src`。
- [x] 记录当前模型、数据和输出路径只来自环境变量/路径配置，禁止硬编码 Windows 或 Linux 绝对路径。
- [x] 为 0–22 章建立需求 ID，例如 `ARCH-04-FAST-001`。
- [x] 把本 TODO 末尾追踪矩阵纳入评审，确保每个源章节至少有一个实施阶段和一个验收位置。
- [x] 固定第一版禁止项清单，并在代码评审模板中加入检查框。
- [x] 定义实验命名规则，必须包含规范版本、数据折、seed、模型 revision 和 TTT 开关。
- [x] 定义每阶段产物目录：配置快照、日志、checkpoint、指标、审计 JSON、失败样例。
- [x] 明确本机负责模块/FSM/loss/小张量测试，服务器负责 8B/视频/FlashAttention/DeepSpeed/多 GPU。
- [x] 建立“计划设计”和“已验证实现”两列状态，禁止文档先行后被误报为已完成。

### 实施后验收项

- [x] 能从任一源章节反查到对应 P 和验收测试。
- [x] 基线测试、Ruff、mypy 均有原始日志。
- [x] v5 固定项、禁止项和实验待定项三者已分开。
- [x] 没有把旧 v3 配置值写成 v5 运行事实。
- [x] 没有修改模型行为。

### 交付物与退出条件

- [x] 交付规格锁、环境快照、仓库基线报告和需求追踪表。
- [x] P0 评审通过后才允许修改配置或创建模型模块。

---

## P1. v5 配置、类型契约与模块骨架

### 目标与依赖

建立一个能拒绝错误组合的 v5 配置层，以及 `ARCHITECTURE.md` 建议的模块职责骨架。依赖 P0。

### 实施前注意事项

- 配置迁移必须原子完成：YAML、解析校验、README/DECISIONS 和契约测试必须同一阶段同步。
- 旧 `test_v3_architecture_config.py` 不能继续验证 512/16/8 等过期值。
- 尚未实验确定的阈值应标记为 bootstrap/default/calibration-required，不能伪装成最终结论。
- `model.py` 只做编排，不复制子模块逻辑。

### 实施过程 TODO

#### P1.1 文件骨架

- [x] 创建 `src/ttt_svcbench_qwen/model.py`。
- [x] 创建 `src/ttt_svcbench_qwen/qwen_adapter.py`。
- [x] 创建 `src/ttt_svcbench_qwen/fast_ttt.py`。
- [x] 创建 `src/ttt_svcbench_qwen/state_encoder.py`。
- [x] 创建 `src/ttt_svcbench_qwen/observation_heads.py`。
- [x] 创建 `src/ttt_svcbench_qwen/state_bank.py`。
- [x] 创建 `src/ttt_svcbench_qwen/identity_bank.py`。
- [x] 创建 `src/ttt_svcbench_qwen/query_encoder.py`。
- [x] 创建 `src/ttt_svcbench_qwen/state_retriever.py`。
- [x] 创建 `src/ttt_svcbench_qwen/state_reader.py`。
- [x] 创建 `src/ttt_svcbench_qwen/input_composer.py`。
- [x] 创建 `src/ttt_svcbench_qwen/losses.py`。
- [x] 创建 `src/ttt_svcbench_qwen/functional_sgd.py`。
- [x] 创建 `src/ttt_svcbench_qwen/trainer.py`。
- [x] 创建 `src/ttt_svcbench_qwen/inference.py`。
- [x] 保留 `config.py` 为配置加载、验证和环境摘要入口。
- [x] 每个文件写清唯一职责、输入输出类型和禁止依赖。

#### P1.2 配置 schema

- [x] 增加 `spec_version` 并固定为 v5 规范名。
- [x] 配置并校验基座 checkpoint、Transformers 版本和 27/1152/16/36/4096 等关键值。
- [x] 配置 Fast Adapter：input=4096、bottleneck=768、residual=0.1、fast_bias=false。
- [x] 配置 Inner SGD：SGD、`1e-4`、momentum=0、weight_decay=0、steps=1、clip=1.0、reset_per_video=true。
- [x] 配置空间路：dim=768、stages=2、heads=12、head_dim=64、refinements=3、FFN=3072、slots=32、max=64。
- [x] 配置时间路：dim=768、layers=6、heads=12、head_dim=64、FFN=3072、dropout=0.1、strict causal、cache=64。
- [x] 配置 O1/O2/E1/E2 的全部层宽、输出维度、TCN kernel/dilation 和 GRU 层数。
- [x] 配置 State Bank：semantic_dim=512、Confirmed 256/增长 256/无硬上限、Candidate 64/上限 512、Hot Cache 256、event history 512。
- [x] 配置 Query Encoder：input=4096、hidden=768、layers=4、heads=12、FFN=3072、dropout=0.1、output=512。
- [x] 配置 9 个 operator prototype 和 unsupported。
- [x] 配置 Retriever bootstrap：`record_similarity_threshold=0.35`、`top_k=null`、
      `ann_enabled=false`，并标记阈值需校准。
- [x] 配置 State Resampler：queries=16、layers=3、heads=8、FFN=2048、hidden=512、output=4096。
- [x] 配置 Predictor `768→1536→768`。
- [x] 配置 loss 权重：pred=1.0、id=0.5、event=0.5、O1-unlabeled=0、auxiliary-outer=0.1。
- [x] 为 Time Resolver、FSM 和匹配阈值添加“未校准”状态，未校准时禁止正式评估。

#### P1.3 运行时类型

- [x] 定义视频 batch：像素、`video_grid_thw`、时间戳、query_time、valid mask、video_id、trajectory_id。
- [x] 定义 Query 输出：`q_target`、`q_operator`、`q_time`、operator logits/confidence、padding mask。
- [x] 定义 `TimeWindow`：mode=now|history|recent|explicit_range，query_time=float，
      start_time=float|null，end_time=float，valid=bool。
- [x] 定义空间/时间 encoder 输出和缓存结构。
- [x] 定义 O1/O2/E1/E2 soft output，字段名与输出契约逐一对应。
- [x] 定义统一 State Record 和各类型 payload。
- [x] 定义 Retriever 输出：selected ids、scores、mask、status、`N_s`、`N_ret`。
- [x] 定义 ReaderResult：status、exact_count、number_token_ids、selected_record_ids、operator、time_window、audit_fields。
- [x] 定义 per-video runtime state，覆盖 fast weights、optimizer state、slot state、temporal cache、Bank、FSM 和 Reader audit。

#### P1.4 配置与参数契约测试

- [x] 将旧 v3 配置测试迁移为 v5，删除所有过期断言。
- [x] 断言两个 fast matrix 均为 `768×768`，总数 1,179,648。
- [x] 断言新增模块预算与 `ARCHITECTURE.md` 允许的舍入误差一致。
- [x] 断言 top_k 为 null、ANN 关闭、O1 unlabeled loss 权重为 0。
- [x] 断言 DeepStack online freeze 和原索引不变。
- [x] 断言非法组合在启动前失败，例如 hidden 不能被 heads 整除、State Token 不为 16、momentum 非 0。

### 实施后验收项

- [x] 一个 v5 YAML 能通过强校验并打印完整解析配置。
- [x] 一个故意混入 v3 值的 YAML 会明确失败，而不是静默使用默认值。
- [x] 所有推荐模块均可导入，但尚未实现的路径显式报 `NotImplementedError`，不能返回假结果。
- [x] 配置测试准确覆盖 768 bottleneck、32 slots、16 State Token 和新容量。
- [x] 全仓不存在仍声称“当前 v5 使用 512 fast bottleneck”的活跃配置或测试。

### 交付物与退出条件

- [x] 交付 v5 YAML、强类型配置、模块骨架和 v5 契约测试。
- [x] P1 通过后，后续阶段只能从配置读取固定维度，禁止散落 magic numbers。

---

## P2. 数据、视频预处理、因果切分与 A0 基线

### 目标与依赖

建立 SVCBench 数据读取、Qwen 视频预处理、query-time 因果切分、分组划分和原始 8B 零样本基线。依赖 P1。

> 2026-07-14 用户批准的临时退出口径：受本机空间限制，缺失的视频、非 clean 训练集和 8B
> 权重以合成 fixture/predictor 完成 P2 工程链路验收并解锁 P3。下列 A0 `[x]` 只表示数据、
> 指标、报告和禁用 State-TTT 的工程接口可重复；原始 Qwen3-VL-8B 科学基线仍必须在
> P19/P21/P22 使用真实权重和视频完成，合成指标不得用于论文比较或增益结论。

### 实施前注意事项

- Demo 的 16 帧、224×224、2 fps 只用于 shape 验收，不是数据集长度硬编码。
- `pixel_values_videos` 已是展平并归一化的 tubelet，不得当作 `[B,F,C,H,W]` 再处理。
- 时间长度 T、12 个 attention heads 和 16 个 State Token 相互独立。
- 所有视频裁剪必须在模型看到帧之前执行，不能先处理完整视频再 mask。

### 实施过程 TODO

#### P2.1 SVCBench schema 与防泄漏

- [x] 解析 video_path、question、query point/time、训练标签和评估字段。
- [x] 为运行时输入建立 allowlist，只传 video/question/query_time/显式时间数值。
- [x] 为 answer、count、occurrence_times、counting_type、counting_subtype 建立 denylist。
- [x] 在 Dataset、Collator、Trainer、Inference 四层分别加入泄漏断言。
- [x] 按 video_path 执行 GroupKFold，确保同视频全部问题和 query point 同折。
- [x] 保存合成 fold manifest 并检查 video_id 交集为空。
- [x] 禁止 clean 官方测试视频参与预训练、校准或阈值选择。

#### P2.2 因果视频采样

- [x] 在 query_time 处做严格右截断，验证边界帧是否允许由时间戳定义一致决定。
- [x] 生成 chunk、chunk_start/end、tubelet timestamp、overlap mapping 和 valid mask。
- [x] 处理最后一个不足完整 chunk 的 padding，以 valid mask 建立禁止其进入 loss、Bank 或 Reader 的消费契约；后续消费者在 P9/P12/P14 复验。
- [x] 保留从采样帧到 tubelet、从 tubelet 到秒的可审计映射。
- [x] 为相邻重叠 chunk 生成身份/事件一致性所需对齐索引，但不读取未来帧。

#### P2.3 Qwen Video Processor Demo

- [x] 构造 `X=[1,16,3,224,224]`、fps=2 的固定 fixture。
- [x] 验证 temporal grid：`16/2=8`。
- [x] 验证 spatial grid：`224/16=14`、`224/16=14`。
- [x] 验证 `video_grid_thw=[8,14,14]`。
- [x] 验证 patch/tubelet 数：`8×14×14=1568`。
- [x] 验证每 tubelet 展平维：`2×3×16×16=1536`。
- [x] 验证 `pixel_values_videos=[1,1568,1536]`。
- [x] 用非 16 帧、非 224 分辨率样例证明 shape 来自 grid，而不是硬编码。

#### P2.4 Query token 数据

- [x] 只截取完整问题 token 对应的 embedding 输入范围，超长时拒绝而不静默截断。
- [x] 排除 system answer、assistant target、标签 token 和 padding。
- [x] 生成 query padding mask。
- [x] 用 Demo 问题“当前画面有几架无人机？”验证演示 `L_q=7` 的接口，但不固定真实长度。

#### P2.5 A0 原始模型基线

- [x] 以合成 predictor 跑通 A0 工程链路并完全关闭状态模块和 TTT；真实 8B 实测移交 P19/P21。
- [x] 在合成报告中记录 exact count accuracy、count MAE、answer accuracy、延迟和显存字段。
- [x] 保存合成 dry-run 的 prompt template、generation 参数和失败案例。
- [x] 将合成 A0 作为接口等价锚点；科学增益对照仍要求 P21 的真实 A0。

### 实施后验收项

- [x] Demo 十个核心 shape 中 P2 数据侧项目全部通过。
- [x] 任意样本都能证明最大可见时间不超过 query_time。
- [x] Dataset/Collator 输出中不存在 denylist 字段。
- [x] GroupKFold 无 video 泄漏。
- [x] 变长视频、尾 chunk、空/短视频和多 query point 均有测试。
- [x] 合成 A0 可重复运行并生成完整工程指标字段；真实 A0 保留在 P19/P21。

### 交付物与退出条件

- [x] 交付 Dataset、Collator、causal chunker、Demo fixtures、合成 fold manifest 和合成 A0 报告。
- [x] 数据防泄漏测试通过，并保留失败时禁止进入任何模型训练的强制 guard。

---

## P3. Qwen3-VL 接口、Merger 插入点与 DeepStack 保护

### 目标与依赖

在 Transformers 4.57.1 的真实调用链中取得 Main Merger 主输出，并在 video `masked_scatter` 前提供唯一 State-TTT 插入点，同时保持 DeepStack 完全原样。依赖 P1–P2。

### 实施前注意事项

- 必须以实际 checkpoint config 和当前 Transformers 源码为准，不引用旧版 `pooler_output`。
- ViT index 8、16、24 是视觉层索引，不等于 LLM 注入层索引。
- Main Merger 只把空间 `14×14` 压到 `7×7`，时间 8 不变。
- DeepStack 等价性是阻断门禁，不是性能优化项。

### 实施过程 TODO

- [x] 在 `qwen_adapter.py` 封装 Qwen3-VL 加载与配置断言。
- [x] 跟踪 `get_video_features()` / visual forward / Main Merger / `inputs_embeds.masked_scatter` 的真实调用链。
- [x] 捕获 Main Merger 输出 `V_t[B,N_v,4096]`，保留 batch 到 token 的映射。
- [x] 用 Demo 验证 PatchEmbed `[1568,1536]→[1568,1152]`。
- [x] 验证 27 个 ViT block 保持 `[1568,1152]`。
- [x] 验证 Main Merger 的分组维 `4×1152=4608` 并输出 4096。
- [x] 验证 token 数 `1568/4=392`，逻辑网格 `[8,14,14]→[8,7,7]`。
- [x] 暴露 `V_t`、merged grid metadata 和原 DeepStack features，接口不复制 Qwen 内部实现。
- [x] 把 Adapter hook 放在 Main Merger 输出之后、video placeholder scatter 之前。
- [x] 保持 image 路径与非视频输入行为不变。
- [x] 保持 DeepStack 三组 `[N_v,4096]` 的顺序、dtype、device、mask 和注入位置不变。
- [x] 明确验证三组 DeepStack 特征按顺序进入 Qwen decoder 的前三个对应处理层级；不得把
      ViT indexes 8/16/24 误当成 LLM layer indexes。
- [x] 加入 adapter-disabled 模式，用于与原 Qwen 输出逐张量比较。
- [x] 加入启动断言；checkpoint config 不匹配时 fail fast。

### 实施后验收项

- [x] Demo Main Merger 输出严格为 `[1,392,4096]`。
- [x] Adapter disabled 时，video embeddings、DeepStack tensors 和模型输出与原模型在容差内一致。
- [x] 插入点测试证明发生在 Main Merger 后和 `masked_scatter` 前。
- [x] DeepStack 不经过 Adapter、Bank 或新增 token mask。
- [x] image-only、text-only 和 video 输入没有被 wrapper 破坏。
- [x] 代码与测试中无 `pooler_output` 依赖。

### 交付物与退出条件

- [x] 交付 `qwen_adapter.py`、配置断言和原模型等价测试。
- [x] DeepStack 等价测试失败时禁止实施 P5/P13。

---

## P4. Query Encoder、Operator Router 与 Time Window Resolver

### 目标与依赖

把完整问题表示编码为互相分工的 `q_target`、`q_operator`、`q_time`，并输出 hard operator 与显式 TimeWindow。依赖 P1–P2。

### 实施前注意事项

- Query 只看问题 token，不能看答案或测试标签。
- 使用 Qwen token embedding，不额外执行完整 36 层回答 decoder。
- 问题在查询时完整可见，因此 Query Transformer 是双向注意力，只屏蔽 padding。
- q_time 不能直接充当精确窗口；operator 低置信度不能强行归类。

### 实施过程 TODO

#### P4.1 Query 主干

- [x] 实现 `Q_h[B,L_q,4096]→Linear 4096→768`。
- [x] 加入无参数 sinusoidal position encoding，保证词序可见且不改变参数预算。
- [x] 实现 4 层 Pre-LN 双向 Transformer Encoder。
- [x] 固定 hidden=768、heads=12、head_dim=64、FFN=3072、GELU、dropout=0.1，且无额外 final LayerNorm。
- [x] attention mask 只屏蔽 padding，禁止 causal mask。
- [x] 实现 learned-attention pooling：`softmax(w^T tanh(WX_q)+M)`，最终 scorer 无 bias。
- [x] 断言 padding 权重严格为 0。
- [x] 内部得到 `h_q[B,768]`。

#### P4.2 三个 embedding head

- [x] 分别实现三个独立 `768→1024→512` GELU MLP，且无额外 final LayerNorm。
- [x] 对 `q_target`、`q_operator`、`q_time` 分别 L2 normalize。
- [x] 禁止三个 head 共享最后一层参数。
- [x] 验证三者均为 `[B,512]`。
- [x] 将职责写入接口：target=对象/事件语义，operator=计数操作，time=时间语义。

#### P4.3 Operator Router

- [x] 创建 9 个可训练 512 维 prototype：O1-Snap、O1-Delta、O2-Unique、O2-Gain、E1-Action、E1-Transit、E2-Periodic、E2-Episode、unsupported。
- [x] 实现归一化余弦 logits 和正的可训练温度 `tau`，初值为 1.0。
- [x] 训练时输出 9 类监督 logits。
- [x] 测试时 hard argmax，并返回 confidence。
- [x] 最大置信度低于校准阈值时强制 unsupported；阈值为 null 时 eval/inference 一律 unsupported。
- [x] 建立 operator→head_type 的确定映射。
- [x] 不实现关键词规则作为正式路由；规则 Parser 只允许在 Q0 诊断消融中存在。

#### P4.4 Time Window Resolver

- [x] 实现 `q_time[512]→MLP 512→256→4`，分类 now/history/recent/explicit_range。
- [x] 在 `X_q[B,L_q,768]` 上实现两个 `Linear 768→1` pointer head，预测 numeric span start/end。
- [x] 用全局非 padding pointer 边界与唯一候选 grammar 受约束解析中英文 seconds/minutes、recent 和 range。
- [x] 使用合法 `query_points.time` 作为 query_time。
- [x] 无显式窗口时使用固定映射：O1-Snap→now，O1-Delta/O2-Gain→recent（须显式正 duration），O2-Unique/E1/E2→history。
- [x] 生成 `TimeWindow(mode,query_time,start_time,end_time,valid)`。
- [x] 语法/窗口错误返回 invalid，未校准/低置信度返回 unsupported；二者都强制 effective operator=unsupported，不猜窗口。
- [x] 保证 end_time 不超过 query_time。
- [x] 明确禁止 count 和 occurrence_times 参与解析。

### 实施后验收项

- [x] 变长问题和 padding batch 的三个输出 shape 正确且有限。
- [x] 双向 mask 测试证明问题 token 可互相注意、padding 不可见。
- [x] API denylist、canonical-question/offset 边界及调用方 provenance 声明违规会被拒绝；不声称能识别问题正文中伪装的答案文字。
- [x] 9 类 operator 均有单元测试；低置信度返回 unsupported。
- [x] TimeWindow 覆盖 now/history/recent/explicit_range、无数值、非法单位、反向区间和未来区间。
- [x] 完整 Query Encoder 参数量约 36.03M。

### 交付物与退出条件

- [x] 交付 `query_encoder.py`、operator router、Time Resolver 和监督接口。
- [x] Router/Resolver 的最终阈值保留为 P21 校准项。

---

## P5. Fast TTT Adapter 与 fast 参数收集

### 目标与依赖

实现 `4096→768→768→4096` 残差 Adapter，并建立严格的 1.18M 在线参数边界。依赖 P1、P3。

### 实施前注意事项

- 两个 fast matrix 无 bias；任何额外 bias 进入 inner optimizer 都违反参数契约。
- 慢投影与 RMSNorm 在 Outer Training 可学习，但在线必须冻结。
- 本阶段只实现 Adapter 和参数收集；一步 SGD 的完整损失编排在 P14。
- 初始输出和数值尺度需要稳定，但不能增加未规定的 Gate。

### 实施过程 TODO

- [x] 在 `fast_ttt.py` 实现输入 `V_t[B,N_v,4096]`。
- [x] 实现 RMSNorm 和 `P_in:4096→768`，得到 `U_t`。
- [x] 实现无 bias `W_t^(1):768×768`。
- [x] 实现 SiLU。
- [x] 实现无 bias `W_t^(2):768×768`。
- [x] 实现 `P_out:768→4096`。
- [x] 实现 `Z_t=V_t+0.1*P_out(W2*SiLU(W1*U_t))`。
- [x] 输出保持 `[B,N_v,4096]`、dtype 和 device。
- [x] 保存 meta-learned `W0`，支持 per-video 克隆和 reset。
- [x] 实现 `collect_fast_parameters()`，只返回两个矩阵且顺序稳定。
- [x] 实现 slow/fast 参数分组和在线冻结断言。
- [x] 为 state_dict/checkpoint 明确保存 `W0` 而非某个视频临时 `W_t`。
- [x] 统计完整 Adapter 参数约 7.48M，慢部分约 6.30M。
- [x] 增加 forward hooks/审计字段，记录 fast 参数 norm、输出残差 norm 和更新版本号。

### 实施后验收项

- [x] Demo 输入 `[1,392,4096]` 输出同 shape。
- [x] 两个 fast matrix shape 均为 `[768,768]`，无 bias。
- [x] 在线参数精确等于 `2×768²=1,179,648`。
- [x] fast 参数集合中没有 `P_in`、`P_out`、RMSNorm 或其他模块。
- [x] reset 后 fast weights 与 `W0` 逐元素一致。
- [x] 两个不同 video runtime 的 fast weights 不共享 storage。
- [x] forward/backward 后 fast weights 能获得有限梯度。
- [x] Adapter 参数预算在允许舍入误差内匹配 7.48M。

### 交付物与退出条件

- [x] 交付 `fast_ttt.py`、fast state/reset API、参数边界测试和数值审计。
- [x] 在线参数计数不精确等于 1,179,648 时禁止进入 P14。

## P6. 空间对象编码器

### 目标与依赖

把 adapted merger tokens 解码为 query-conditioned、跨 tubelet 连续的 32 个对象槽。依赖 P4–P5。

### 实施前注意事项

- 32 是当前 chunk 的 GPU 活动工作集，不是整段视频身份上限、Bank 容量或最大计数。
- 槽初始化固定为 shared seed + 单一 `q_target:512→768` 投影 +
  `sinusoidal(slot_index,dim)/sqrt(768)`；固定 code 必须是 `persistent=False` buffer，禁止把
  `[32,768]` 永久可学习向量当作身份。
- 两个 Slot Stage 的 Q/K/V/O、三个 LayerNorm、GRUCell 和 FFN 参数/storage 不共享；同一
  Stage 内 3 次 refinement 复用同一对象。
- P6 的 overflow 是显式 `required_slot_counts` 的结构容量审计，不是从视频识别真实对象数；
  语义对象、新对象和长期生命周期分别留给 P8/P9。
- P6 当前实现候选只允许使用合成张量做工程门禁，不得声称真实视频、真实 8B、语义 overflow
  准确率或在线收益。

### 实施过程 TODO

#### P6.1 网格恢复与投影

- [x] forward 显式接收 `Z_t[B,N_max,4096]`、`visual_valid_mask[B,N_max]`、merged grid
  metadata、`tubelet_valid_mask[B,T_max]`、`q_target[B,512]` 和逐样本 runtime。
- [x] 使用 merged grid metadata 将每行前 `T_i×H_i×W_i` 个有效 token 恢复为
  `[T_i,H_i,W_i,4096]`，校验 token count/offset/mask 一致。
- [x] Demo 验证 `[1,392,4096]→[1,8,7,7,4096]`。
- [x] batch 内支持异构 `T_i/H_i/W_i`；展平空间为 `[B,T_max,S_max,4096]` 时同步构造 bool
  mask，禁止假定所有输入固定 49。
- [x] 实现 per-token `LayerNorm(4096,eps=1e-5)` + 带 bias `Linear 4096→768`。
- [x] 传播 tubelet/spatial valid mask；无效位置不得进入 attention、confidence 或 recurrent
  update。

#### P6.2 Query-conditioned Recurrent Slot Attention

- [x] 实现一个共享可学习 seed、一个全模块唯一的带 bias `q_target:512→768` 投影，以及按
  `sinusoidal/sqrt(768)` 确定的非持久 fixed slot code。
- [x] 首个有效 tubelet 用 shared/query/fixed 初始化；后续有效 tubelet 用上一有效 tubelet 槽作为
  recurrent 初始化，无效 tubelet 原样 carry。
- [x] 实现 Stage 1：12-head、head_dim=64 的完整带 bias Q/K/V/O 投影。
- [x] 自定义经典 Slot Attention 归一化：logits 先在 slot 轴 softmax，再对每槽有效 token 权重
  以 `eps=1e-8` 归一；不得直接采用标准 MHA forward 的 token-axis softmax。
- [x] 实现 Stage 1 GRUCell，hidden=768。
- [x] 实现 Stage 1 FFN `768→3072→768`。
- [x] 在 Stage 1 使用三个 `eps=1e-5` Pre-LayerNorm、SiLU 和 FFN residual。
- [x] Stage 1 每个时间片执行 3 次共享参数 refinement。
- [x] 独立实现同结构 Stage 2，确认参数对象与 Stage 1 不共享。
- [x] Stage 2 复用同一个空间输入 `X`，以 Stage 1 输出作初始化。
- [x] 将同一个 q projection 同时用于共享槽初始化和每次 attention query 条件化。
- [x] 在所有 refinement 中正确使用 spatial mask 和 slot_valid_mask。

#### P6.3 活动槽生命周期

- [x] 正式基线初始化 32 个活动槽。
- [x] 配置允许最大 64，不在 forward 中创建不可追踪的新参数。
- [x] 维护 per-video slot recurrent runtime，并在新视频 reset；runtime 不注册为参数、
  不进入 `state_dict`。
- [x] batch 内每个样本维护独立 slot state、slot_valid_mask、occupancy confidence、overflow
  counter/audit；previous runtime 不被原地修改，next runtime 每行使用独立新 storage。
- [x] `detach_runtime=True` 只 detach 下一 chunk runtime，当前 `A_t` 保留到 adapted embeddings/
  fast weights 的梯度；`False` 保留跨 chunk 图。
- [x] occupancy confidence 使用无参 assignment mass 公式：token 再归一化前的有效 mass / 有效
  token 数，再对 heads 求均值；范围 `[0,1]`，无效槽为 0。
- [x] 显式接收非负整数 `required_slot_counts[B]`；每个 forward 只累计一次
  `max(required_slot_counts-32,0)` 并记录 overflow event。
- [x] overflow 时仍计算原 32 槽，不替换、不扩容；required 值只影响 audit，不能改变 slot 数值。
- [x] 输出最后一个有效 tubelet 的 `A_t[B,K_a,768]` 及 confidence/validity/runtime/audit；无有效
  tubelet 且无 previous runtime 时 fail closed。

### 实施后验收项

- [x] Demo 输出严格为 `A_t[1,32,768]`。
- [x] 12 heads × 64 head_dim = 768。
- [x] Stage 内 refinement 参数共享、Stage 间参数不共享的对象身份测试通过。
- [x] q_target 改变时槽条件化输出可观察变化；padding q 不污染输出。
- [x] 异构 grid、padding token、无效 tubelet 和 slot mask 测试通过。
- [x] 不同 batch/video 的 recurrent slot state 和 storage 隔离。
- [x] reset 后首时间片行为可重复。
- [x] `detach_runtime` 两种模式的梯度测试证明当前输出不截断 fast 路径。
- [x] required=31/32/>32 时 overflow 精确记录且 slot 数值不变，无 silent overwrite。
- [x] 参数量精确为 24,815,360；fixed slot code、confidence、runtime 和 audit 均为零参数。
- [x] 16/32/48/64 只作为 P21 消融配置，正式基线仍为 32。

### 交付物与退出条件

- [x] 交付 `state_encoder.py` 中空间路、slot runtime state、overflow audit 和单元测试。
- [x] recurrent state 隔离或 overflow 保护失败时禁止接入 O1/O2。
- [x] P6 只可提供无状态 grid helper；不得提前实现 P7 temporal pooling/Transformer/cache、P8/P9
  对象语义/hard state、P13 模型编排或 P18 受管推理生命周期。

---

## P7. 时间事件编码器

### 目标与依赖

从 adapted merger tokens 生成严格因果、query-conditioned 的 tubelet 级时间状态。依赖 P4–P5。

### 实施前注意事项

- 文档中的“帧级状态”实际是 tubelet 级；Demo 每个位置覆盖 2 个采样帧。
- temporal cache 最多保留最近 64 个 tubelet，必须按视频隔离并 reset。
- P2 固定 4-tubelet overlap；主 cache 长度仍为 64，但允许额外保存紧邻主 cache 之前最多 3 个
  replay-only per-layer K/V，仅用于按当前 adapted token 重算 overlap，不能扩大 64-token mask。
- timestamp/query_time 必须使用独立于模型 FP16/BF16 的 FP32/FP64；cache timestamp 统一为 FP64。
- strict causal mask 必须在带 cache 和无 cache 两条路径一致。
- 时间长度 T 来自输入 grid，不能和 12 个 heads 或 16 个 State Token 混淆。

### 实施过程 TODO

#### P7.1 空间池化

- [x] 恢复 `[B,T,H_m,W_m,4096]` 并展平为 `[B,T,N_spatial,4096]`。
- [x] Demo 验证 `[B,392,4096]→[B,8,49,4096]`。
- [x] 实现 LayerNorm + `Linear 4096→768`。
- [x] 使用 q_target 条件化的多头空间 attention pooling。
- [x] 对无效空间 token 使用 mask。
- [x] 输出池化状态 `[B,T,768]`。

#### P7.2 六层严格因果 Transformer

- [x] 加入 tubelet 时间位置编码，明确相对/绝对位置与 cache offset 的一致规则。
- [x] 实现 6 层 Pre-LN Transformer。
- [x] 固定 hidden=768、heads=12、head_dim=64、intermediate=3072、dropout=0.1。
- [x] 实现 strict causal attention mask。
- [x] 处理尾 chunk padding，padding 不作为 Key/Value 或 loss target。
- [x] 输出 `H_t[B,T,768]`。

#### P7.3 Temporal cache

- [x] 为每个 video/batch sample 保存最近 64 个有效 tubelet 状态或 KV。
- [x] cache 追加时保存时间戳和有效性。
- [x] 超过 64 时按时间顺序淘汰最旧位置。
- [x] 新视频和 trajectory reset 时清空。
- [x] 禁止 batch 样本交换 cache。
- [x] 禁止 query_time 之后的 cache 内容进入当前 forward。

### 实施后验收项

- [x] Demo 输出严格为 `H_t[1,8,768]`。
- [x] 改变未来 tubelet 不会影响过去位置输出。
- [x] chunked+cache 与等价完整因果 forward 在容差内一致。
- [x] cache 长度不超过 64，reset 和 batch 隔离测试通过。
- [x] T<2、变长 T、尾 padding 和空有效位置均安全处理。
- [x] 参数量精确为 48,438,272（48.438272M）。

### 交付物与退出条件

- [x] 交付 `state_encoder.py` 中时间路、cache API、causal/padding 测试。
- [x] 任何未来信息可影响历史输出时立即阻断后续施工。

---

## P8. O1/O2/E1/E2 Observation Decoder

### 目标与依赖

把空间槽和时间状态转换为四类连续观测；Decoder 不直接产生最终累计答案。依赖 P6–P7。

### 实施前注意事项

- O1/O2 都读取完整 768 维对象槽；绝不是把 768 拆成 6+256。
- identity embedding=256 只用于实体匹配，semantic embedding=512 用于问题检索，两者不可混用。
- E1 面向短促点事件；E2 面向有开始、持续、结束的区间事件。
- hard FSM 与整数更新在 P9/P10 执行，Decoder 只输出 soft evidence。

### 实施过程 TODO

#### P8.1 O1 当前数量 Decoder

- [x] 实现 `q_target[512]→Linear 512→1536`，拆为 768 维 FiLM scale 和 768 维 shift。
- [x] 对 `A_t[B,32,768]` 执行 LayerNorm + FiLM。
- [x] 实现共享逐槽 MLP `768→1024→1024→6`，激活 SiLU。
- [x] 六个 logits 命名为 object、target、visible、enter、exit、confidence。
- [x] 实现软计数 `sum_i p_object*p_target*p_visible`，只作为训练/诊断软量。
- [x] 输出与 slot_valid_mask 对齐。
- [x] 定义写入 hard state 所需 current_visible_count、baseline_count、enter/exit/visible、timestamp、confidence 字段。

#### P8.2 O2 身份 Decoder

- [x] 实现 LayerNorm + 共享 trunk `768→1024→1024`，激活 SiLU。
- [x] identity 分支 `1024→256` 并 L2 normalize。
- [x] score 分支 `1024→2`，字段为 novelty 和 match confidence。
- [x] 输出 identity `[B,K_a,256]`、score `[B,K_a,2]`。
- [x] 为零向量归一化提供数值保护。
- [x] 接口明确 O2 不直接修改 unique_count。

#### P8.3 E1 点事件 Decoder

- [x] 输入 `H_t[B,T,768]`，执行 LayerNorm + `Linear 768→512`。
- [x] 实现 5 个 gated residual causal TCN block。
- [x] 固定 kernel=3、dilations=[1,2,4,8,16]、channels=512。
- [x] 每个 block 包含 filter/gate dilated Conv1d、1×1 residual projection、LayerNorm 和 SiLU。
- [x] 禁止使用 BatchNorm。
- [x] 输出 `Linear 512→3`：eventness、completion、transition。
- [x] 确保所有 convolution 严格 causal。

#### P8.4 E2 区间事件 Decoder

- [x] 对 `H_t[B,T,768]` 执行 LayerNorm。
- [x] 实现 2-layer GRU，hidden=768。
- [x] event 分支 `768→4`：start、active、end、complete。
- [x] phase 分支 `768→4`：阶段分布。
- [x] GRU hidden state 按 video/batch 隔离并支持 reset。
- [x] 输出 event `[B,T,4]` 和 phase `[B,T,4]`。

#### P8.5 参数与通用接口

- [x] 为四个 Decoder 统一返回 soft tensor、valid mask、timestamp 和必要 debug logits。
- [x] O1/O2 使用对象槽 mask，E1/E2 使用 tubelet mask。
- [x] 统计 O1≈2.63M、O2≈2.10M、E1≈9.58M、E2≈7.09M。
- [x] 在线推理时冻结四个 Decoder，但保留到 Fast Adapter 的梯度路径。

### 实施后验收项

- [x] O1 `[B,32,6]`、O2 identity `[B,32,256]`、O2 score `[B,32,2]`。
- [x] E1 `[B,T,3]`、E2 event/phase 均为 `[B,T,4]`。
- [x] O2 identity 每个有效向量 L2 norm 约为 1。
- [x] E1/E2 的未来输入不影响过去输出。
- [x] E2 hidden state reset 与样本隔离通过。
- [x] O1/O2 并行读取同一 768 维槽的测试通过。
- [x] 参数预算和 online freeze 契约通过。

### 交付物与退出条件

- [x] 交付 `observation_heads.py`、四类输出 dataclass 和形状/因果/参数测试。
- [x] 四个输出契约未稳定前禁止实现 hard count。

---

## P9. Semantic Projector、类型化 State Bank 与事件 FSM

### 目标与依赖

把 soft observation 转为隔离、可审计、不可反向传播的 hard state，并为每条记录生成统一 512 维语义检索视图。依赖 P8。

### 实施前注意事项

- Bank 是当前视频的运行时内存，不是模型参数，也不是外部数据库。
- hard write 必须位于 `torch.no_grad()` 且写入前 detach；TTT loss 必须使用 detach 前 soft branch。
- `N_s` 是动态候选记录数，不是活动槽数、身份容量或时间长度。
- hard FSM 阈值尚需校准；先实现可配置、可审计状态迁移，不把 bootstrap 阈值当最终值。

### 实施过程 TODO

#### P9.1 统一记录和隔离键

- [x] 以 `(video_id, trajectory_id, head_type)` 隔离所有 Bank 分区；`trajectory_id` 即 question trajectory。
- [x] 定义统一字段：record_id、head_type、semantic_embedding[512]、timestamp/time_range、valid、confidence、type-specific payload。
- [x] record_id 在轨迹内唯一且不可复用。
- [x] 支持按分区追加、更新、失效、快照、查询和释放。
- [x] 新视频或 trajectory 结束时释放对应状态。
- [x] 不同 batch 样本的 Bank 对象和 storage 不共享。

#### P9.2 Semantic Projector

- [x] 接收 object slot/event state `[768]`。
- [x] 加入 learned head-type embedding `[768]`。
- [x] 实现 LayerNorm + `Linear 768→1024→512` + SiLU。
- [x] 对 semantic embedding L2 normalize。
- [x] 对 O1/O2/E1/E2 使用共享投影器和不同 head-type embedding。
- [x] 语义检索维保持 512，不随 768 状态主干扩大。
- [x] 精确参数量为 1,316,864。

#### P9.3 O1 hard state

- [x] 根据 object/target/visible/enter/exit/confidence 和 slot mask 更新活动状态。
- [x] 维护 current_visible_count。
- [x] 在轨迹规定位置建立 baseline_count，定义何时初始化和何时 reset。
- [x] 保存每槽 enter/exit/visible、更新时间和置信度。
- [x] 防止同一槽在单一时间位置重复增减。
- [x] 记录低置信度、invalid slot、槽溢出、证据漂移和状态冲突。

#### P9.4 E1 hard FSM

- [x] 实现双阈值 on/off 状态迁移。
- [x] 实现 cooldown。
- [x] 实现 Temporal NMS。
- [x] 仅在确认完成证据时增加 event_count。
- [x] 保存 recent_event_times，容量 512。
- [x] 防止一个持续多帧脉冲重复计数。
- [x] 记录 duplicate suppression、miss candidate 和 cooldown 命中。

#### P9.5 E2 hard FSM

- [x] 实现 INACTIVE→ACTIVE→END_CANDIDATE→COMPLETED。
- [x] 根据 start/active/end/complete 和 phase logits 更新状态。
- [x] 只有确认完整结束后 completed_count +1。
- [x] 保存已完成时间区间、当前 phase 和 recent_event_times。
- [x] recent_event_times 容量 512，淘汰不得改变累计 completed_count。
- [x] P9 reset/release 清空 Bank 与 hard FSM runtime；P8 E2 GRU runtime 由 P18 受管生命周期协调清理。

#### P9.6 梯度与持久化边界

- [x] 所有 hard update 包裹 `torch.no_grad()`。
- [x] 写入 semantic/identity/event tensor 前 detach+clone。
- [x] Bank/FSM/runtime 不注册为 `nn.Parameter` 或 buffer。
- [x] Bank/FSM/runtime 不进入 `model.state_dict()`；Semantic Projector 进入模型 `state_dict()`。
- [x] Bank/FSM/runtime 不进入 Outer optimizer 或 Inner SGD；Projector 进入 Outer optimizer、排除在 Inner SGD 之外。
- [x] 提供显式 runtime snapshot 仅用于审计/恢复，和模型 checkpoint 分离。

### 实施后验收项

- [x] 动态记录可生成 padded `E_state[B,N_max,512]`、`n_state[B]`、present/valid masks 和 record IDs；`N_max` 变化不改变任何模型参数 shape。
- [x] O1、E1、E2 payload 及 O2 Candidate/Confirmed generic payload/CRUD 字段完整；O2 生命周期留 P10。
- [x] hard Bank/FSM 张量无 `grad_fn`，而 soft outputs 仍可向 Fast Adapter 反传。
- [x] event history 超过 512 时审计正确且累计计数不变。
- [x] video/trajectory/head_type/batch 隔离测试通过。
- [x] reset/release 后无残留记录。
- [x] Semantic Projector 输出归一化、有限且精确参数量为 1,316,864。

### 交付物与退出条件

- [x] 交付 `state_bank.py`、Semantic Projector、O1/E1/E2 FSM、审计结构和梯度隔离测试。
- [x] 任意 hard state 进入 autograd 或模型 state_dict 时阻断后续训练。

---

## P10. Identity Bank 动态容量与 GPU Hot Cache

### 目标与依赖

实现 O2 Candidate→Confirmed 身份生命周期、无语义硬上限的 CPU store，以及不改变计数语义的 GPU Hot Cache。依赖 P8–P9。

### 实施前注意事项

- 32 个活动槽只是当前计算工作集；长期 Confirmed 身份可远超 32。
- Confirmed 扩容、缓存换出和重复观测都不能再次增加 unique_count。
- 256 维 identity 用于实体匹配；512 维 semantic embedding 用于 q_target 检索。
- 第一版 exact search 为真，ANN 关闭；ANN 触发规模留到 P21。

### 实施过程 TODO

#### P10.1 Candidate store

- [ ] 初始容量 64。
- [ ] 支持增长，但强制安全上限 512。
- [ ] 保存 256 维 normalized prototype、观测次数、TTL、置信度、最近观测时间和关联 record_id。
- [ ] 未匹配 observation 创建或更新 Candidate。
- [ ] 依据连续可靠观测阈值晋升 Confirmed。
- [ ] TTL 过期和置信度清理有确定顺序与审计字段。
- [ ] 达到 512 时执行显式 prune/overflow 处理并增加 `candidate_overflow`，禁止 silent overwrite。

#### P10.2 Confirmed CPU store

- [ ] 初始分配 256 个位置。
- [ ] 每次按 256 分块增长。
- [ ] 不设置语义硬上限。
- [ ] 使用 CPU FP32 分块张量保存完整记录。
- [ ] 字段包含 identity_id、256 维 prototype、first_seen、last_seen、observation_count、semantic record link。
- [ ] 连续出现超过 256 个身份时扩容并保留全部旧记录。
- [ ] 轨迹释放时释放 CPU storage。

#### P10.3 匹配与 prototype 更新

- [ ] 对有效 O2 identity 做 exact matching。
- [ ] 使用 match confidence/novelty 和可配置阈值区分更新、Candidate、新身份。
- [ ] prototype 更新规则可配置并记录旧值/新值。
- [ ] Candidate 只有首次晋升为 Confirmed 时 `unique_count+1`。
- [ ] 已 Confirmed 实体再次出现只更新 last_seen/observation_count/prototype。
- [ ] 防止同一 observation 匹配多个 Confirmed；冲突必须审计。

#### P10.4 GPU Hot Cache

- [ ] 默认容量 256。
- [ ] 只缓存 Confirmed 的加速副本，不成为真值来源。
- [ ] 定义换入/换出策略和 CPU→GPU dtype/device 转换。
- [ ] cache miss 必须回退完整 CPU store exact search。
- [ ] Hot Cache 换出不删除 CPU record、不改变 unique_count。
- [ ] 缓存开启/关闭时 Reader 结果和匹配结果一致。

### 实施后验收项

- [ ] 第 257 个 Confirmed 身份触发扩容且前 256 个逐条可读。
- [ ] 同一身份重复 100 次只增加一次 unique_count。
- [ ] Candidate 过期、低置信度清理和上限溢出均有测试。
- [ ] CPU store 与 Hot Cache 开/关、换入/换出结果一致。
- [ ] 不同 video/batch 的 identity_id、store 和 cache 隔离。
- [ ] identity duplicate rate 和 missed-new-identity rate 可计算。
- [ ] ANN 未被启用。

### 交付物与退出条件

- [ ] 交付 `identity_bank.py`、容量/匹配/cache API 和 >256 身份压力测试。
- [ ] 任意扩容或 cache 操作改变 exact count 时阻断 Retriever/Reader。

---

## P11. Embedding State Retriever

### 目标与依赖

让 hard operator 先限定记录类型，再由 q_target 在当前合法 Bank 分区进行全记录阈值检索。依赖 P4、P9–P10。

### 实施前注意事项

- 查询目标是当前 video + 当前 trajectory 的 Structured State Bank。
- 不查询原始像素、LLM KV cache、O2 256 维 identity prototype 或外部向量库。
- 第一版不做 Top-K；所有超过阈值的合法记录都必须可见。
- 固定 Top-K 会截断合法对象/事件并造成静默少计，因此不能用“更方便 batching”作为启用理由。
- “可靠查询但没有记录”可返回 0；“查询或时间解析不可靠”必须 unsupported。

### 实施过程 TODO

#### P11.1 候选分区

- [ ] hard operator 映射到合法 head_type。
- [ ] 按 video_id、trajectory_id、head_type 取得候选 records。
- [ ] 统计过滤前 `N_s`。
- [ ] 生成 ragged/padded `E_state[B,N_s,512]` 和 record mask。
- [ ] q_target 和 E_state 都执行 L2 normalize。

#### P11.2 相似度与硬过滤

- [ ] 计算余弦分数 `S[B,N_s]`。
- [ ] 强制 same video_id。
- [ ] 强制 same question_trajectory_id。
- [ ] 强制 head_type 匹配 hard operator。
- [ ] 强制 `record.valid=true`。
- [ ] 强制 record time ≤ query_time。
- [ ] 需要窗口时强制 record 与 requested TimeWindow 相交。
- [ ] 强制 semantic similarity ≥ calibrated threshold。
- [ ] `record_similarity_threshold` bootstrap 值使用 0.35，但标记为待校准。
- [ ] `top_k=null`，不得按数量截断。
- [ ] `ann_enabled=false`。
- [ ] 返回全部命中记录、score、record_id、`N_s` 和 `N_ret`。

#### P11.3 状态判定与审计

- [ ] 路由/时间/检索置信度不可靠时返回 unsupported。
- [ ] Bank 覆盖有效且可靠查询无匹配时返回 empty，允许 Reader 解释为 0。
- [ ] 区分 empty Bank、无语义匹配、全部超时、全部 invalid 和 unsupported。
- [ ] 记录每个过滤原因的计数。
- [ ] 保留 selected_record_ids 和分数供 Reader/评估审计。
- [ ] 统计 Retriever precision、recall、空检索率。

### 实施后验收项

- [ ] `0≤N_ret≤N_s` 对所有 batch 成立。
- [ ] 3、30、300 条命中均不被 Top-K 截断。
- [ ] 跨视频、跨 trajectory、错误 head、未来时间和 invalid 记录全部被过滤。
- [ ] threshold 边界值有确定行为。
- [ ] empty 与 unsupported 可由结构化 status 区分。
- [ ] O2 identity prototype 不会被误当 semantic embedding 查询。
- [ ] 无外部 ANN/向量数据库依赖。

### 交付物与退出条件

- [ ] 交付 `state_retriever.py`、过滤审计、ragged batch 支持和检索测试。
- [ ] 出现 silent truncation 或未来记录命中时禁止接入 Reader。

---

## P12. 16-token State Resampler 与 Deterministic Reader

### 目标与依赖

把可变数量检索记录压缩成固定 16 个语义 token，同时从未压缩 hard records 计算精确整数。依赖 P4、P11。

### 实施前注意事项

- 16 个 token 是 learned State Query 的输出，不是 Top-16 records。
- Resampler 可以压缩语义，但 Reader 必须读取完整命中记录。
- empty retrieval 必须使用显式 empty embedding，不能让 attention 对空维 softmax 产生 NaN。
- number token 必须由 Reader 的 exact_count 序列化，严禁训练时偷换 ground truth。

### 实施过程 TODO

#### P12.1 State Resampler

- [ ] 创建 `Q_state[16,512]` learned queries。
- [ ] batch 广播并加 `q_target[:,None,:]`，得到 `[B,16,512]`。
- [ ] 将检索 semantic records 作为 K/V `[B,N_ret,512]`。
- [ ] 实现 3 层 Perceiver/Q-Former 风格 Resampler。
- [ ] 每层先执行 16 queries 间 8-head self-attention。
- [ ] 每层再执行 queries 对 records 的 8-head cross-attention。
- [ ] 每层实现 FFN `512→2048→512`。
- [ ] 每个子层使用 Pre-LayerNorm + residual。
- [ ] 正确使用 record mask，attention 权重 shape 为 `[B,16,N_ret]`。
- [ ] 核对 cross-attention 计算为 `softmax(QK^T/sqrt(d)+M)V`，mask 后不得把无效 record
      分配注意力质量。
- [ ] 输出 `H_state[B,16,512]`。
- [ ] 实现 `P_state:512→4096`，输出 `R_t[B,16,4096]`。
- [ ] N_ret=0 时注入显式 `empty_record_embedding`。
- [ ] 记录 cross-attention selected mass 供解释审计。
- [ ] 验证 State Token 编码当前问题相关对象/事件语义、状态置信度、时间背景和自然语言解释所需软信息。
- [ ] 参数量约 14.72M。

#### P12.2 Deterministic Reader

- [ ] 输入只接受 hard operator、resolved TimeWindow、retrieved typed records。
- [ ] O1-Snap 返回 current_visible_count。
- [ ] O1-Delta 返回 current_visible_count - baseline_count。
- [ ] O2-Unique 返回 query_time 前 Confirmed 身份数。
- [ ] O2-Gain 返回时间窗口内 first_seen 身份数。
- [ ] E1-Action/E1-Transit 返回 query_time 前符合类型的完成事件数。
- [ ] E2-Periodic/E2-Episode 返回 query_time 前符合类型的完整区间数。
- [ ] unsupported 不生成伪造整数。
- [ ] empty 可靠查询返回 exact_count=0。
- [ ] invalid 时间/状态返回 exact_count=null。
- [ ] 输出 status=ok|empty|unsupported|invalid。
- [ ] 输出 exact_count、selected_record_ids、operator、time_window 和 audit_fields。
- [ ] 使用真实 tokenizer 把 exact_count 序列化为 `number_token_ids[L_num]`。
- [ ] 支持 number token ids→文本→整数的双向审计。
- [ ] 防止 LLM 或调用方覆盖 Reader exact_count。

### 实施后验收项

- [ ] 命中 0、3、30、300 条时 State Token 始终 `[B,16,4096]`。
- [ ] 空检索输出有限、无 NaN，且 empty embedding 可训练。
- [ ] 16 tokens 与任何 16 条 record 不存在位置一一对应假设。
- [ ] 每个 hard operator 均有独立算术 fixture 和边界测试。
- [ ] Reader exact_count 与手工类型化记录计算完全一致。
- [ ] number token 来自 Reader，替换 ground-truth count 的负向测试会失败。
- [ ] State Token 修改不会改变 Reader exact_count。
- [ ] Reader 参数中不存在学习型计数器。

### 交付物与退出条件

- [ ] 交付 State Resampler、`state_reader.py`、数字序列化和 operator 算术测试。
- [ ] exact count 与 record audit 无法双向核对时禁止进入 LLM Composer。

---

## P13. Input Composer 与完整模型编排

### 目标与依赖

把 question、adapted video、16 个 State Token 和 Reader number token 正确注入 Qwen prefill，并保持 DeepStack 原生注入语义。依赖 P3、P5–P12。

### 实施前注意事项

- video、state、number 三种 payload 的 mask/position/type 必须分开。
- State Token 不得标成 visual position；DeepStack 只能作用于原 visual positions。
- state 和 number payload 必须位于 assistant answer 之前。
- generate prefill 只构建一次；decode 循环不能重复 Bank/TTT 更新。

### 实施过程 TODO

#### P13.1 Special token 与模板

- [ ] 定义 state begin/end、state placeholder、number begin/end 等必要 special token。
- [ ] 将 16 个固定 state placeholder 加入 tokenizer/chat template。
- [ ] 定义可变长度 number token 放置规则。
- [ ] 保存 tokenizer revision 和新增 token id 映射。
- [ ] 确保新增 token embedding 初始化和 checkpoint 保存可复现。

#### P13.2 Payload 拼接

- [ ] 保留原始 system/user/question token。
- [ ] video embeddings 继续走 Qwen 原生 video placeholder mask + `masked_scatter`。
- [ ] 对 16 个 state placeholder 执行独立 `masked_scatter`。
- [ ] number 位置使用 Reader 返回的真实 tokenizer token id。
- [ ] 组装顺序保证 question/video/state/number 在 assistant answer 之前。
- [ ] 计算简化长度 `L_payload=L_q+N_v+K_s+L_num`。
- [ ] Demo 验证 `7+392+16+L_num=415+L_num`。
- [ ] 真实长度额外计入 chat template、system、视觉边界和状态边界 token。

#### P13.3 Mask、position 与 cache

- [ ] attention mask 覆盖所有新增位置。
- [ ] position_ids 覆盖所有新增位置且与 Qwen RoPE/多模态位置约定兼容。
- [ ] cache_position 覆盖 prefill 后的正确偏移。
- [ ] State Token 不进入 visual mask。
- [ ] number token 不进入 visual mask。
- [ ] DeepStack 只注入原 visual positions，不作用于 state/number。
- [ ] padding batch 中每种 payload 的 mask 正确。

#### P13.4 模型编排

- [ ] `model.py` 只按顺序调用 qwen adapter、Query、Fast、双 Encoder、四 Decoder、Bank、Retriever、Reader、Resampler、Composer。
- [ ] 明确“视频观测/状态更新”和“回答 query”两个入口。
- [ ] 允许关闭 Fast、Bank、Reader、State Token，支持 A0–A6/Q0–Q3。
- [ ] 统一返回 answer logits、ReaderResult、state audit、TTT soft intermediates。
- [ ] 保留 Qwen 原 generate API 所需字段。
- [ ] prefill 构建完成后设置一次性标记，decode step 拒绝再次写 Bank 或更新 fast weights。

#### P13.5 LLM 职责约束

- [ ] prompt 明确要求读取提供的 exact number。
- [ ] LLM 负责根据问题组织答案、使用 State Token 解释对象/身份/事件背景、读取 number token
      并输出自然语言。
- [ ] 训练标签与 Reader number 一致时才训练最终表达。
- [ ] 记录 LLM 是否输出了与 Reader 不一致的数字。
- [ ] 不让 LLM 从 392 个 video tokens 重新累计长视频计数。
- [ ] 不让 LLM 从 16 个 State Token 猜测精确整数。
- [ ] 不允许 LLM 覆盖 Reader 已确定的数字。

### 实施后验收项

- [ ] Demo payload 长度公式正确，真实模板长度可逐 token 审计。
- [ ] video/state/number placeholder 数与 payload 数严格一致；不一致时 fail fast。
- [ ] 新增 attention mask、position id、cache position 全部通过单元测试。
- [ ] State Token/number token 不属于 visual positions。
- [ ] DeepStack shape、mask、注入顺序与原模型一致。
- [ ] generate prefill 后多个 decode step 不改变 Bank、fast weights 或 runtime state。
- [ ] LLM 输出数字与 Reader exact_count 的一致率可单独统计。

### 交付物与退出条件

- [ ] 交付 `input_composer.py`、`model.py`、tokenizer 变更、prefill/decode 集成测试。
- [ ] 任意 placeholder 错位、DeepStack 污染或 decode 重复更新时阻断训练。

## P14. TTT/State/Answer Loss 与 functional SGD

### 目标与依赖

实现无标签 `L_TTT`、有标签 State/Answer loss、Meta-TTT Outer loss，以及只更新两个 fast matrix 的一步 SGD。依赖 P5–P13。

### 实施前注意事项

- hard FSM、整数计数器和 Bank records 不参与反向传播。
- TTT loss 使用写 Bank 前、detach 前的 soft outputs。
- `L_pred` 只做当前 chunk 内 next-tubelet prediction，不跨 chunk 长期保留 autograd graph。
- identity match mask 是有效性检查，不是学习型 Gate。
- 当前可见对象数会因进入、离开、遮挡和时间偏移合法变化；简单 O1 一致性会把输出推向常数。
- O1 不进入无标签 loss，但必须保留有标签 State Loss，并可间接受益于其他 TTT 项更新后的视觉特征。

### 实施过程 TODO

#### P14.1 时序预测 `L_pred`

- [ ] 实现 Predictor：LayerNorm 768 → Linear 768→1536 → SiLU → Linear 1536→768。
- [ ] 对当前 chunk 使用 `P(H_t[:,:-1])` 预测 `stop_gradient(H_t[:,1:])`。
- [ ] 使用 MSE，只对连续且有效的 tubelet pair 求均值。
- [ ] Predictor 不预测像素、不预测最终计数。
- [ ] 有效时间位置少于 2 时返回 invalid term，而不是伪造 0 参与平均。
- [ ] 参数量约 2.36M。

#### P14.2 身份一致性 `L_id`

- [ ] 只选择相邻重叠 chunk 中可靠匹配的同一对象。
- [ ] 严格实现 `1-cos(e_(t-1,i), stop_gradient(e_(t,j)))`，stop-gradient 固定作用在当前
      chunk 的匹配 identity target 上。
- [ ] identity 向量必须是 O2 的 normalized 256 维向量。
- [ ] 没有有效匹配时该项值为 0，同时 valid=false，避免错误影响全局平均。
- [ ] 对 mismatch、重复匹配和低置信度匹配写审计计数。
- [ ] mask 仅决定有效样本，不控制是否学习更新。

#### P14.3 事件一致性 `L_event`

- [ ] 定义 `L_event=L_E1-overlap+L_E2-overlap`。
- [ ] E1 对重叠位置比较 eventness、completion、transition。
- [ ] E2 比较 start、active、end、complete 和 phase distribution。
- [ ] 第一实现对二值 soft outputs 使用 masked MSE。
- [ ] 第一实现对 phase distribution 使用 stop-gradient target KL。
- [ ] 所有项只在有效、时间对齐的重叠位置求均值。
- [ ] hard FSM、event_count、completed_count 和记录列表不进入 loss 图。
- [ ] 将 MSE/BCE/其他距离保留为 P21 消融，不提前宣称最终最优。

#### P14.4 顶层无标签 loss

- [ ] 实现 `L_TTT=L_pred+0.5*L_id+0.5*L_event`。
- [ ] O1 unlabeled 权重固定为 0，并断言没有 O1 项偷偷加入。
- [ ] 按项返回 value、valid count、mask count 和 skip reason。
- [ ] 无任何有效项时整次 inner update 跳过。
- [ ] 检查每项和总 loss 有限。

#### P14.5 有标签 State Loss

- [ ] 每个样本只计算对应任务的 `L_O1`、`L_O2`、`L_E1` 或 `L_E2`。
- [ ] 定义 O1 六字段监督和 slot matching/mask。
- [ ] 定义 O2 identity/match/novelty 监督。
- [ ] 定义 E1 eventness/completion/transition 监督。
- [ ] 定义 E2 event/phase/FSM 辅助监督。
- [ ] 加入 9 类 `L_operator`。
- [ ] 加入 record-level 正负样本 `L_retrieval`。
- [ ] 加入时间语义和合法数值窗口 `L_time`。
- [ ] 实现 `L_state=L_task+lambda_op*L_operator+lambda_ret*L_retrieval+lambda_time*L_time`。
- [ ] 不为不相关 Head 构造伪标签或 loss。

#### P14.6 Answer 与 Outer Loss

- [ ] 实现 teacher-forced answer CE。
- [ ] 单独记录 number token accuracy。
- [ ] 单独记录完整自然语言 answer accuracy。
- [ ] 单独记录 Reader exact count accuracy。
- [ ] Support inner update 后在后续 Query 计算 `L_outer=L_answer_after+L_state_after`。
- [ ] 最终实现 `L_total=L_outer+0.1*mean(L_TTT)`。
- [ ] 明确 after-update Query Loss 才训练“更新后是否更好”；auxiliary 只保持无标签目标可学习。

#### P14.7 Functional SGD

- [ ] 在 `functional_sgd.py` 实现只接收两个 fast matrix 的一步 SGD。
- [ ] 默认 lr=`1e-4`、momentum=0、weight_decay=0、steps=1。
- [ ] 更新前检查 loss finite。
- [ ] 用 `autograd.grad` 或等价方式只求 fast gradients。
- [ ] 检查每个 gradient finite。
- [ ] 计算并记录 pre-clip norm。
- [ ] 执行 global norm clip=1.0。
- [ ] 裁剪后再次检查 finite/usable。
- [ ] 无效时保持 `W_t` 不变并记录 skip reason。
- [ ] 有效时生成 `W_(t+1)`，不在当前 chunk 重跑观测。
- [ ] 为 Meta-TTT 保留所需可微更新路径；一阶/二阶近似若需选择，单独记录且不得静默 detach。
- [ ] reset 时清空任何 SGD runtime state。

#### P14.8 梯度审计

- [ ] 验证 Shared Encoder/Decoder 参数在线不变但其运算允许梯度传到 Fast Adapter。
- [ ] 验证 hard Bank/FSM 没有梯度。
- [ ] 验证 Query/Reader/LLM 不在 inner optimizer。
- [ ] 输出每模块 gradient presence、norm 和 parameter delta 审计表。
- [ ] 对非 finite loss/gradient 构造故障注入测试。

### 实施后验收项

- [ ] Predictor shape、参数量和 T<2 invalid 行为正确。
- [ ] 三个无标签项的 stop-gradient 方向和 valid mask 测试通过。
- [ ] `L_TTT` 权重严格为 1/0.5/0.5，O1=0。
- [ ] State Loss 只监督对应任务 Head。
- [ ] Inner SGD 后只有两个 fast matrix 发生变化。
- [ ] 当前 chunk 输出不因本 chunk 末更新而改变，下一 chunk 使用新参数。
- [ ] 非有限和无有效项均安全跳过。
- [ ] reset 后恢复 `W0`。
- [ ] 梯度能穿过冻结状态网络到达 fast weights。

### 交付物与退出条件

- [ ] 交付 `losses.py`、`functional_sgd.py`、loss dataclass、梯度/skip 审计和故障测试。
- [ ] 任何非 fast 参数在 inner step 变化时禁止开始 Meta-TTT。

---

## P15. Stage A：显式状态 Warm-up

### 目标与依赖

关闭 Inner SGD，先让 Query、状态编码、四 Decoder、Bank、Retriever、Reader 和 LLM 表达路径在无 TTT 情况下正确工作。依赖 P13–P14。

### 实施前注意事项

- Stage A 的目标是验证显式状态系统，不是证明在线适应收益。
- Outer Training 采用全量、分阶段解冻或 LLM LoRA 尚未决定；必须先在训练折上选择并记录。
- hard state rollout 与 soft training proxy 必须同时运行，但 hard 路径不反传。
- Reader exact count 指标必须独立于 LLM answer 指标。

### 实施过程 TODO

- [ ] 在 `trainer.py` 增加 Stage A 模式并强制 Inner SGD 关闭。
- [ ] 训练 Query Embedding Encoder。
- [ ] 训练 operator prototypes。
- [ ] 训练 Time Window Resolver。
- [ ] 训练 State Retriever 的语义表示与 record-level supervision。
- [ ] 训练 Semantic Projector 和 State Resampler/Projector。
- [ ] 训练空间对象路和时间事件路。
- [ ] 训练 O1/O2/E1/E2。
- [ ] 运行 hard State Bank、Identity Bank 和 FSM rollout。
- [ ] 将 Deterministic Reader 纳入 Stage A 端到端路径；其固定算术若无可学习参数则不加入
      optimizer，只通过 State/Answer loss 验证上游状态质量。
- [ ] 训练必要的 Qwen 参数或选定 LoRA；记录冻结清单。
- [ ] 使用 `L_state+L_answer`，Inner `L_TTT` 不执行更新。
- [ ] 为每类任务构建平衡采样和有效标签 mask。
- [ ] 监控 O1 soft/hard count、O2 duplicate/missed-new、E1/E2 duplicate/miss。
- [ ] 监控 operator 9 类、unsupported、retrieval、time-window 和 Reader exact count。
- [ ] 保存最优 checkpoint 时同时保存 tokenizer/config/spec version。
- [ ] 在 validation 上检查 Reader 与 LLM 数字不一致案例。

### 实施后验收项

- [ ] 关闭 TTT 时四类任务均能端到端生成 hard state 和 ReaderResult。
- [ ] Query Router、Time Resolver、Retriever、Reader 的独立指标可计算。
- [ ] Reader exact count 不依赖 ground-truth count 注入。
- [ ] Bank reset、扩容、cache 和 FSM 在训练 batch 中稳定。
- [ ] loss 无 NaN/Inf，梯度和显存可控。
- [ ] A2（四 Head+Bank+Reader、TTT off）可复现实验完成。
- [ ] Stage A 指标、checkpoint 和失败样例齐全。

### 交付物与退出条件

- [ ] 交付 Stage A 配置、checkpoint、A2 报告和冻结策略记录。
- [ ] 无 TTT 时 Reader 尚不能稳定工作，禁止用 Meta-TTT 掩盖基础错误。

---

## P16. Stage B：单步 Meta-TTT

### 目标与依赖

使用 1 个 Support chunk、仅 `L_pred` 和一步 functional SGD，验证元训练、reset 和 after-update Query 监督。依赖 P15。

### 实施前注意事项

- 本阶段先关闭 `L_id` 和 `L_event`，降低变量数量。
- Support 只能使用无标签信息；Query 才使用 Answer/State 标签。
- Support 更新必须和测试时 optimizer、步数、reset、因果顺序完全一致。
- 不能让当前 Support chunk 在更新后重算并污染其 hard state。

### 实施过程 TODO

- [ ] 构造 episode：1 Support chunk + 至少 1 个后续 Query point。
- [ ] Support 使用当前 `W_t` 前向并写 hard state。
- [ ] 计算当前 chunk 内 `L_pred`。
- [ ] 只对两个 fast matrix 做一步 SGD。
- [ ] 得到 `W_(t+1)` 并仅用于后续 Query/chunk。
- [ ] 在 Query 上计算 after-update `L_answer+L_state`。
- [ ] 加入 `0.1*L_pred` auxiliary。
- [ ] 通过 outer gradient 学习 `W0` 和允许的 Outer 参数。
- [ ] 每个 episode 开始 reset fast/SGD/cache/slot/Bank/FSM/audit。
- [ ] 记录 before-update 与 after-update Query 指标。
- [ ] 记录 update norm、gradient norm、skip reason、每视频 update 次数。
- [ ] 对 first-order/second-order 实现做梯度正确性小张量检查。
- [ ] 用固定 seed 做重复性测试。
- [ ] 构造无有效时间位置 Support，确认跳过而不破坏 episode。

### 实施后验收项

- [ ] 单 Support 流程严格按 observe→state→loss→SGD→next 生效。
- [ ] Support 标签没有进入 inner loss。
- [ ] after-update Query loss 能反传到 meta-learned `W0`。
- [ ] 只有两个 fast matrix 在 episode 内变化。
- [ ] 不同 episode/video 之间无 fast 或 state 污染。
- [ ] before/after 指标、update norm 和 skip 率均可审计。
- [ ] A3（A2+`L_pred`+SGD）可独立运行。

### 交付物与退出条件

- [ ] 交付 Stage B episode runner、meta-gradient 测试、A3 配置和报告。
- [ ] reset、更新方向或 outer gradient 任一不正确时禁止加入一致性 loss。

---

## P17. Stage C：身份/事件一致性与多 Support

### 目标与依赖

按可审计增量依次加入 `L_id`、`L_event`、4–8 个连续 Support chunks 和多个后续 Query points。依赖 P16。

### 实施前注意事项

- 每个增量都必须有独立消融，不能一次性全部打开。
- chunk overlap 对齐必须来自因果采样元数据。
- 当前 chunk `L_pred` 不保留跨 chunk autograd graph。
- 多 Support 下 hard state 连续、fast weights 连续，但每个新视频仍从 `W0` 开始。

### 实施过程 TODO

#### P17.1 加入身份一致性

- [ ] 在 A3 上只开启 `L_id`，形成 A4。
- [ ] 验证可靠匹配 mask、无匹配 invalid 和 stop-gradient。
- [ ] 监控 identity duplicate/missed-new、梯度 norm 和 update skip。
- [ ] 比较 A4 vs A3，记录收益、退化和置信区间。

#### P17.2 加入事件一致性

- [ ] 在 A4 上开启 `L_event`，形成 A5。
- [ ] 分开记录 E1 overlap 和 E2 overlap。
- [ ] 验证 event/phase mask、MSE/KL 和 FSM detach。
- [ ] 比较 A5 vs A4，记录分任务收益。

#### P17.3 多 Support

- [ ] 逐步从 1 增加到 4 个连续 Support chunks。
- [ ] 稳定后扩展到最多 8 个连续 Support chunks。
- [ ] 每个有效 chunk 最多一步 SGD。
- [ ] 每步更新只影响后续 chunk。
- [ ] 维护 fast version、temporal cache、slot state、Bank/FSM 的一致时间轴。
- [ ] 限制 autograd 生命周期，确认显存不会随整段视频无界增长。
- [ ] 对 invalid chunk 跳过更新但继续正确推进时间和 hard state。

#### P17.4 多 Query point

- [ ] 一个 episode 支持多个后续 query point。
- [ ] 每个 query 只读取其 query_time 之前的 state。
- [ ] 同视频 query point 共享因果历史时，明确 trajectory 隔离/复用策略并保持规范键。
- [ ] 分别计算每个 query 的 after-update Answer/State loss。
- [ ] 防止较晚 query 的标签或状态回流到较早 query。

### 实施后验收项

- [ ] A4、A5 能分别复现，配置差异只包含目标增量。
- [ ] 1/4/8 Support 的更新时间线可逐步审计。
- [ ] 多 Query 不发生未来信息或标签泄漏。
- [ ] 显存不会因跨 chunk graph 无界增长。
- [ ] `L_TTT=L_pred+0.5L_id+0.5L_event` 精确生效。
- [ ] 每个增量都有 before/after、分任务和失败案例。

### 交付物与退出条件

- [ ] 交付 Stage C runner、A4/A5 配置、multi-support/query 测试和消融报告。
- [ ] 只有 A5 完成因果与状态审计后才允许构建正式推理路径。

---

## P18. 测试时协议与推理入口

### 目标与依赖

实现逐视频 reset、逐 chunk 因果观测/更新和单次 prefill 回答流程。依赖 P13–P17。

### 实施前注意事项

- 测试时只允许合法输入；所有标签字段必须在 API 边界被拒绝。
- 当前 query 不能受 query_time 之后信息影响。
- generate 的 decode loop 不得调用 observe/update。
- skip update 是合法状态，必须记录原因。

### 实施过程 TODO

#### P18.1 每视频 reset

- [ ] reset fast weights 到 `W0`。
- [ ] reset SGD state。
- [ ] reset temporal cache。
- [ ] reset recurrent slot state。
- [ ] reset Identity Candidate/Confirmed/Hot Cache。
- [ ] reset O1/E1/E2 FSM 和 event histories。
- [ ] reset Reader audit state。
- [ ] reset GRU hidden、fast version 和 update counters。
- [ ] 为 reset 前后生成 state checksum，证明无跨视频残留。

#### P18.2 每 chunk 因果流程

- [ ] 第 1 步：严格裁剪到 query_time 以前。
- [ ] 第 2 步：Qwen ViT + Main Merger。
- [ ] 第 3 步：Fast Adapter 使用当前 `W_t`。
- [ ] 第 4 步：空间对象路和时间事件路。
- [ ] 第 5 步：四 Decoder 产生 soft observation。
- [ ] 第 6 步：`no_grad` 更新 State Bank/FSM。
- [ ] 第 7 步：若有有效无标签目标，计算 `L_TTT`。
- [ ] 第 8 步：一步 SGD 得到 `W_(t+1)`。
- [ ] 记录 chunk id、时间范围、fast version、更新是否执行和 skip reason。

#### P18.3 回答 query

- [ ] Query Encoder 生成三个 embedding。
- [ ] operator prototypes 生成 hard operator/unsupported。
- [ ] Time Resolver 生成显式 TimeWindow。
- [ ] q_target 检索当前合法 Bank。
- [ ] Reader 计算 exact integer 和 number tokens。
- [ ] Resampler 生成 16 个 State Token。
- [ ] Composer 组装 question/video/state/number。
- [ ] Qwen LLM 生成自然语言答案。
- [ ] 保存 ReaderResult、selected records、State attention 和最终文本。

#### P18.4 Generate 生命周期

- [ ] prefill 前只执行一次 query read/composition。
- [ ] decode step 只更新 LLM KV cache。
- [ ] decode step 禁止修改 Bank、FSM、fast weights、slot state 或 temporal cache。
- [ ] 重复调用 generate 时明确是新 query 还是同 query 重试，并保持 state 语义一致。
- [ ] abort/exception 后安全释放当前 runtime，不污染下一视频。

### 实施后验收项

- [ ] 两个连续视频的首 chunk 均从相同 `W0` 和空 Bank 开始。
- [ ] 当前 chunk 更新只影响下一 chunk。
- [ ] query_time 后帧的扰动不改变 ReaderResult 或答案输入。
- [ ] generate 多 token 期间 fast/Banks checksum 不变。
- [ ] invalid/unsupported/empty/ok 四种 Reader 状态均能生成合规响应。
- [ ] 推理日志能完整重放每次计数来源。

### 交付物与退出条件

- [ ] 交付 `inference.py`、per-video runtime manager、CLI/入口和端到端因果测试。
- [ ] reset 或 future-leakage 测试失败时禁止服务器正式评估。

---

## P19. Stage D：真实 8B、FlashAttention、DeepSpeed 与多 GPU

### 目标与依赖

把已通过小张量和单卡契约的系统接入真实 Qwen3-VL-8B checkpoint，并完成服务器级训练/推理。依赖 P18。

### 实施前注意事项

- Windows 基础锁文件不直接加入 FlashAttention、DeepSpeed、bitsandbytes；服务器按 CUDA/PyTorch 建专用环境或 lock。
- 真实 checkpoint config 必须重新断言，不能只依赖文档数字。
- CPU Identity Bank、GPU Hot Cache 和多 GPU batch shard 必须保持样本归属。
- 性能优化不得改变 causal mask、DeepStack、Reader 或 online update 边界。

### 实施过程 TODO

- [ ] 锁定 Qwen3-VL-8B checkpoint revision 和权重 hash。
- [ ] 在服务器验证 CUDA、驱动、编译器、PyTorch、Transformers 兼容性。
- [ ] 安装并验证 FlashAttention。
- [ ] 配置 DeepSpeed，明确 ZeRO stage、参数分片和 optimizer ownership。
- [ ] 配置多 GPU 数据并行/模型并行策略。
- [ ] 确保两个 fast matrix 的 per-sample/per-video state 不被错误全局同步。
- [ ] 确保 hard Bank/FSM 不被 DeepSpeed 当作模型参数分片。
- [ ] 确保 CPU Confirmed store 与负责该样本的 rank 绑定。
- [ ] 接入真实 Main Merger hook 并重新跑 DeepStack 等价测试。
- [ ] 验证 mixed precision 下 semantic/identity normalize、Reader hard count 和 functional SGD 的数值稳定性。
- [ ] 记录峰值 GPU 内存和 CPU 内存。
- [ ] 记录单 chunk forward/backward 延迟。
- [ ] 记录每视频 fast update 总时间和次数。
- [ ] 验证 checkpoint save/load 后 `W0`、Outer 参数、tokenizer 和配置完整，运行时 Bank 不误入模型 checkpoint。
- [ ] 建立中断恢复策略，明确视频中途是否允许恢复 runtime snapshot。

### 实施后验收项

- [ ] 真实 8B 单卡或可行最小配置完成一个端到端 episode。
- [ ] 多 GPU 与单卡在固定小样本上的 Reader exact count 和状态审计一致。
- [ ] DeepStack shape/mask/order 与原模型一致。
- [ ] online parameter delta 仍只覆盖 1,179,648 fast 参数。
- [ ] 无跨 rank/video 状态污染。
- [ ] 峰值显存、CPU 内存、chunk latency 和更新时间完整记录。
- [ ] 服务器环境可由 lock/安装记录重建。

### 交付物与退出条件

- [ ] 交付服务器环境说明、DeepSpeed 配置、启动命令、性能基线和 8B 集成报告。
- [ ] P20 全量验收前不得在 official clean test 上锁定结果。

---

## P20. 全量验收与回归契约

### 目标与依赖

把 `ARCHITECTURE.md` 第 18 章及跨阶段负向契约固化为自动测试。依赖 P0–P19。

### 实施前注意事项

- 本阶段不是第一次写测试；每个 P 已有定向测试。这里负责全链路复核和防漂移。
- 验收必须覆盖数值、状态、梯度和因果边界，不能只检查 shape。
- Demo shape 是固定 fixture；另外必须保留变长和极端输入测试。

### 实施过程 TODO

#### P20.1 Demo tensor 验收

- [ ] `[1,16,3,224,224]` 得到 `grid_thw=[8,14,14]`。
- [ ] `pixel_values_videos=[1,1568,1536]`。
- [ ] PatchEmbed 后 `[1568,1152]`。
- [ ] Main Merger 后 `[1,392,4096]`。
- [ ] adapted video token `[1,392,4096]`。
- [ ] 空间槽 `[1,32,768]`。
- [ ] 时间状态 `[1,8,768]`。
- [ ] O1 输出 `[1,32,6]`。
- [ ] O2 identity `[1,32,256]`、score `[1,32,2]`。
- [ ] E1 输出 `[1,8,3]`。
- [ ] E2 event/phase 均为 `[1,8,4]`。
- [ ] semantic state view 为 `[1,N_s,512]`。
- [ ] 三个 query embedding 均为 `[1,512]`。
- [ ] State Token 为 `[1,16,4096]`。
- [ ] payload 长度符合 `L_q+N_v+K_s+L_num`。

#### P20.2 Online update 验收

- [ ] Inner SGD 后只有两个 fast matrix 变化。
- [ ] 两个矩阵均为 `768×768`，合计 1,179,648 参数。
- [ ] momentum=0、weight_decay=0。
- [ ] 每个 chunk 最多一步。
- [ ] 当前 chunk 更新只影响下一 chunk。
- [ ] 新视频恢复 `W0`。
- [ ] 非 finite loss/gradient 时跳过。
- [ ] 无有效 TTT 项和 T<2 时跳过。
- [ ] Shared Encoder 参数不变但梯度可穿过到 Fast Adapter。
- [ ] hard Bank/FSM 不在 autograd 图中。
- [ ] generate decode 不重复更新。
- [ ] DeepStack 和 Qwen frozen online 参数 delta 为 0。

#### P20.3 状态与检索验收

- [ ] O1/O2 共享 32 个活动槽但长期身份库可超过 32。
- [ ] 超过 256 个 Confirmed 时扩容且旧记录不丢失。
- [ ] Candidate 只有首次晋升使 unique_count +1。
- [ ] Candidate 初始 64、最大 512 的容量行为正确。
- [ ] CPU store 与 GPU Hot Cache 换入换出不改变结果。
- [ ] 不同 batch、video、trajectory 和 head 状态隔离。
- [ ] `N_s` 动态变化不改变模型参数 shape。
- [ ] q_target 只查询当前合法 Bank 分区。
- [ ] 默认无 Top-K，所有超过阈值记录可见。
- [ ] ann=false。
- [ ] 低置信度 unsupported 与真实空集合可区分。
- [ ] 空检索 State Token 有限且无 NaN。
- [ ] 16 个 State Token 不是 Top-16 records。
- [ ] future、invalid、wrong-head records 均被过滤。

#### P20.4 Reader 与输入验收

- [ ] 每类 hard operator 的算术有独立单元测试。
- [ ] exact count 与 number token 可双向审计。
- [ ] number token 来自 Reader，不来自 ground truth。
- [ ] DeepStack shape、mask 和注入顺序与原模型一致。
- [ ] State Token 不进入 visual mask。
- [ ] number token 不进入 visual mask。
- [ ] 新增 token attention mask、position id、cache position 正确。
- [ ] query_time 之后的帧不进入 Bank、TTT 或答案。
- [ ] LLM 不可覆盖 Reader 审计结果。

#### P20.5 Reset、故障与禁止项

- [ ] fast/SGD/cache/slots/Identity/FSM/Reader audit 全 reset。
- [ ] loss NaN、gradient Inf、empty Bank、满 Candidate、slot overflow、CPU cache miss 均有故障测试。
- [ ] Surprise Gate、Inner AdamW/Muon、fixed Top-K、ANN、O1 unlabeled loss 均不存在于正式基线。
- [ ] decode loop 状态 checksum 不变。
- [ ] denylist 字段注入会在边界被拒绝。
- [ ] clean test 数据不会进入 calibration loader。

#### P20.6 工程质量

- [ ] `uv run pytest -q` 全部通过。
- [ ] `uv run ruff check .` 全部通过。
- [ ] `uv run mypy src` 全部通过。
- [ ] 关键模块有覆盖率报告，状态机和 Reader 分支不能缺失。
- [ ] 固定 seed 重跑关键小样本结果一致。
- [ ] 配置、README、DECISIONS、ARCHITECTURE 和测试无版本漂移。

### 实施后验收项

- [ ] P20.1–P20.6 无未解释失败。
- [ ] 所有失败路径都有显式 status/exception/audit，不存在 silent fallback。
- [ ] 全量测试报告关联到唯一 commit、lock、config 和模型 revision。
- [ ] 需求追踪矩阵每一行都有至少一个自动测试或明确的实验验收。

### 交付物与退出条件

- [ ] 交付全量测试套件、覆盖率、回归报告和漂移检查。
- [ ] P20 未全绿时禁止开展官方 clean 评估。

---

## P21. 消融、校准与未决实验

### 目标与依赖

在训练折或独立校准集上完成最小消融、阈值选择和架构未决项，冻结正式评估配置。依赖 P20。

### 实施前注意事项

- 每次实验只改变一个目标因素。
- 不得根据 official clean test 反向选择结构、阈值或学习率。
- 规则 Parser 只用于 Q0 诊断，不能回到正式 embedding 路由。
- ANN 和 DeepStack 改造属于后续选择，默认基线仍关闭。

### 实施过程 TODO

#### P21.1 主方案消融

| ID | 待运行配置 | 验证目的 |
| :--- | :--- | :--- |
| [ ] A0 | 原始 Qwen3-VL-8B | 零样本基线 |
| [ ] A1 | 普通 SFT/LoRA，无状态模块 | 普通微调收益 |
| [ ] A2 | 四 Decoder + Bank + Reader，TTT off | 显式状态收益 |
| [ ] A3 | A2 + `L_pred` + SGD | 时序 TTT 收益 |
| [ ] A4 | A3 + `L_id` | 身份一致性收益 |
| [ ] A5 | A4 + `L_event` | 完整方案 |
| [ ] A6 | A5 去掉 exact Reader | 确定性计数必要性 |

- [ ] 重点比较 A5 vs A2，判断 TTT 是否真实增益。
- [ ] 重点比较 A5 vs A6，判断 exact Reader 是否必要。
- [ ] 每组使用相同 fold、seed、训练预算和评估协议。

#### P21.2 Query/检索消融

| ID | 待运行配置 | 验证目的 |
| :--- | :--- | :--- |
| [ ] Q0 | 规则 Parser，仅诊断 | 固定问法上限 |
| [ ] Q1 | prototype operator，无语义检索 | 路由收益 |
| [ ] Q2 | Q1 + 全记录阈值检索 | q_target 检索收益 |
| [ ] Q3 | Q2 + 16 State Token | 语义摘要收益 |

#### P21.3 固定待决实验

- [?] 比较 Outer Training：全量微调、分阶段解冻、LLM LoRA。
- [?] 比较 768 高容量主干与原 512 主干的净收益、显存和延迟。
- [?] 比较活动槽 16、32、48、64；正式候选基线为 32。
- [?] 比较 State Token 8、16、32；正式候选基线为 16。
- [?] 比较是否替换或增强 P4 已冻结的“双 pointer + 唯一候选受限 grammar” baseline。
- [?] 校准 9 类 operator confidence/unsupported threshold。
- [?] 校准各 head_type 的 record similarity threshold；bootstrap=0.35。
- [?] 校准 O1/O2/E1/E2 FSM、match、promotion、cooldown/NMS 阈值。
- [?] 比较 E1/E2 overlap 的 masked MSE、BCE 或其他一致性距离。
- [?] 确定 Confirmed 多大规模才值得 ANN 候选召回；第一版 ANN=false。
- [?] 比较 Fast LR `3e-5`、`1e-4`、`3e-4`。
- [?] 评估未来版本是否让 DeepStack 经过可适应路径；第一版保持不变。

#### P21.4 校准协议

- [ ] 为每个阈值定义目标指标、约束和 tie-breaker。
- [ ] 只使用训练折或独立 calibration split。
- [ ] 保存完整搜索空间，而不只保存最优点。
- [ ] 报告均值、方差/置信区间和 seed 敏感性。
- [ ] 分析 threshold 对 empty/unsupported、少计/多计的影响。
- [ ] 在正式评估前冻结全部阈值和配置 hash。
- [ ] 把选定值同步回 YAML、DECISIONS、README 和契约测试。

### 实施后验收项

- [ ] A0–A6、Q0–Q3 均有可复现实验记录或明确的资源阻塞说明。
- [ ] A5 vs A2、A5 vs A6 的统计结论可审计。
- [ ] 所有正式阈值都有 train/calibration provenance。
- [ ] 未决项均变成“已选”“保留默认”或“推迟到后续版本”，没有模糊状态。
- [ ] official clean test 从未参与选择。

### 交付物与退出条件

- [ ] 交付消融表、校准曲线、决策记录和 frozen final config。
- [ ] final config 未冻结前禁止 P22 正式测试。

---

## P22. 最终评估、审计与发布门禁

### 目标与依赖

在 frozen config 上运行最终 clean 评估，输出完整精度、状态质量、性能和可复现审计包。依赖 P21。

### 实施前注意事项

- P22 只能读取 P21 冻结的模型、阈值和配置。
- 运行中发现问题可以修复后重新走 P20–P21，但不能在 clean 结果上直接调参。
- 必须同时报告最终答案、Reader、状态模块和 TTT 的分层指标。

### 实施过程 TODO

#### P22.1 核心任务指标

- [ ] exact count accuracy。
- [ ] count MAE。
- [ ] answer accuracy。
- [ ] O1/O2/E1/E2 分任务结果。
- [ ] early/middle/late query 分段结果。
- [ ] 按视频长度分桶结果。
- [ ] 按对象密度分桶结果。
- [ ] 按遮挡程度分桶结果。

#### P22.2 Query、检索与状态指标

- [ ] operator 9 类 accuracy。
- [ ] unsupported recall。
- [ ] operator Expected Calibration Error。
- [ ] State Retriever precision、recall、空检索率。
- [ ] identity duplicate rate。
- [ ] missed-new-identity rate。
- [ ] event duplicate rate 和 miss rate。
- [ ] active slot overflow。
- [ ] candidate overflow。
- [ ] Reader exact count accuracy 与 LLM number agreement。

#### P22.3 TTT 与系统指标

- [ ] 每视频 fast update 次数。
- [ ] update skip rate 及原因分布。
- [ ] update/gradient norm 分布。
- [ ] 单 chunk forward latency。
- [ ] 单 chunk backward/update latency。
- [ ] 峰值 GPU 内存。
- [ ] 峰值 CPU 内存。
- [ ] TTT on/off 的总体与分桶差值。
- [ ] 长视频中 fast/state 漂移趋势。

#### P22.4 审计包

- [ ] 保存模型 revision、Git commit、`uv.lock` hash、最终 YAML、数据 split 和全部 seed。
- [ ] 保存 tokenizer/special token 映射。
- [ ] 保存每个 query 的 operator、TimeWindow、selected_record_ids、ReaderResult。
- [ ] 保存每个视频 reset、chunk timeline、fast version、update/skip 审计。
- [ ] 保存 identity 扩容/cache、event FSM、overflow 和异常记录。
- [ ] 保存硬件、CUDA、PyTorch、Transformers、FlashAttention 和 DeepSpeed 版本。
- [ ] 保存原始指标 JSON、聚合脚本和最终报告。
- [ ] 对随机样本进行从答案→number token→Reader→records→video time 的人工回溯。

#### P22.5 发布门禁

- [ ] P20 全量测试在最终 commit 上重新通过。
- [ ] frozen config 与实际运行 config hash 一致。
- [ ] 无 clean-test tuning。
- [ ] 所有数据泄漏检查通过。
- [ ] 所有状态隔离和 reset 检查通过。
- [ ] README/DECISIONS/ARCHITECTURE/TODO 与最终实现状态同步。
- [ ] 明确标注已实现、未实现、实验支持和未来工作。
- [ ] 用一句话核对最终系统：Main Merger 后插入只更新两个 `768×768` fast matrix 的单步 SGD Adapter；32 槽空间路和 6 层时间路产生四类观测；hard Bank + embedding 查询 + Reader 给出 exact count；video + 16 State Token + number 交给保持原 DeepStack 的 Qwen LLM 表达。

### 实施后验收项

- [ ] 所有规定指标均有数值，不以“未记录”跳过不利结果。
- [ ] 任一最终答案可回溯到 Reader 和类型化记录。
- [ ] 性能、显存、CPU store 和 update 开销均有报告。
- [ ] 最终结论只基于 frozen clean run。
- [ ] 审计包可由另一环境复现核心结果。

### 交付物与退出条件

- [ ] 交付最终 checkpoint/config、clean 评估报告、消融报告、审计包和复现说明。
- [ ] P22 完成即代表 v5 第一版施工结束；任何新结构进入下一规范版本，不回写污染本结果。

---

## 附录 A. 推荐模块职责与禁止越界

| 文件 | 必须负责 | 禁止负责 | 主要阶段 |
| :--- | :--- | :--- | :--- |
| `config.py` | v5 schema、校验、环境/路径、配置快照 | 模型 forward、训练逻辑 | P1 |
| `qwen_adapter.py` | Qwen 加载、真实视觉调用链、Merger hook、DeepStack 透传 | 重写 Qwen block、Bank | P3 |
| `fast_ttt.py` | Adapter、`W0/W_t`、fast 参数收集 | optimizer step、状态机 | P5 |
| `state_encoder.py` | 空间 slot 路、时间 causal 路、slot/cache runtime | hard count、Reader | P6–P7 |
| `observation_heads.py` | O1/O2/E1/E2 soft Decoder | 直接累计最终答案 | P8 |
| `state_bank.py` | 统一 records、O1/E1/E2 hard state、审计、reset/release | 学习型计数、outer 参数 | P9 |
| `identity_bank.py` | Candidate/Confirmed/Hot Cache、匹配、扩容 | q_target 语义检索 | P10 |
| `query_encoder.py` | Query Transformer、三 embedding、operator、Time Resolver | 读取答案/标签 | P4 |
| `state_retriever.py` | hard filters、余弦阈值检索、检索 status/audit | Reader 算术、Top-K | P11 |
| `state_reader.py` | hard operator 算术、exact count、number 序列化 | 神经回归、ground truth 替换 | P12 |
| `input_composer.py` | placeholder、masked scatter、mask/position/cache | 状态更新、TTT step | P13 |
| `losses.py` | TTT/State/Answer/Outer loss 和 valid mask | 参数原地更新 | P14 |
| `functional_sgd.py` | 一步 SGD、finite、clip、reset/skip | Bank/FSM、完整 trainer | P14 |
| `trainer.py` | Stage A/B/C/D episode 和优化器编排 | 子模块内部算法复制 | P15–P19 |
| `inference.py` | per-video reset、chunk 流程、query 回答、generate 生命周期 | 训练标签读取 | P18 |
| `model.py` | 组合与统一输出 | 重复实现上述任何子模块 | P13 |

### A.1 测试目录建议

- [ ] `tests/test_v5_config_contract.py`：所有固定维度、容量、loss 和 optimizer 契约。
- [ ] `tests/test_video_preprocessing.py`：Demo grid/pixels、变长、query-time cut。
- [ ] `tests/test_qwen_adapter.py`：真实插入点和 DeepStack 等价。
- [ ] `tests/test_fast_ttt.py`：shape、参数数、reset、freeze。
- [ ] `tests/test_state_encoder.py`：slot recurrence、causality、cache。
- [ ] `tests/test_observation_heads.py`：四 Decoder 输出和参数预算。
- [x] `tests/test_state_bank.py`：hard state、FSM、isolation、detach。
- [ ] `tests/test_identity_bank.py`：Candidate/Confirmed、>256 扩容、Hot Cache。
- [ ] `tests/test_query_encoder.py`：padding、9 prototypes、TimeWindow。
- [ ] `tests/test_state_retriever.py`：filters、no Top-K、empty/unsupported。
- [ ] `tests/test_state_reader.py`：八个合法 operator、number audit。
- [ ] `tests/test_input_composer.py`：placeholder、mask、position、DeepStack。
- [ ] `tests/test_losses.py`：三个 TTT 项、State/Answer/Outer loss。
- [ ] `tests/test_functional_sgd.py`：只更新 fast、finite/clip/skip。
- [ ] `tests/test_inference_protocol.py`：reset、next-chunk 生效、generate 单次 prefill。
- [ ] `tests/test_leakage_guards.py`：denylist、future frames、fold isolation。
- [ ] `tests/test_end_to_end_demo.py`：ARCH 第 18 章全链路。

---

## 附录 B. 接口、状态和 reset 总账

### B.1 核心张量接口

- [ ] `V_t: [B,N_v,4096]`：Main Merger 主输出。
- [ ] `Z_t: [B,N_v,4096]`：Fast Adapter 后视觉 token。
- [ ] `Q_h: [B,L_q,4096]`：仅问题 token 表示。
- [ ] `q_target/q_operator/q_time: [B,512]`。
- [ ] `A_t: [B,32,768]`：空间活动槽。
- [ ] `H_t: [B,T,768]`：tubelet 级时间状态。
- [ ] `O1: [B,32,6]`。
- [ ] `O2.identity: [B,32,256]`，`O2.score: [B,32,2]`。
- [ ] `E1: [B,T,3]`。
- [ ] `E2.event/phase: [B,T,4]`。
- [ ] `E_state: [B,N_s,512]`。
- [ ] Retriever score `S: [B,N_s]`。
- [ ] Resampler attention `A: [B,16,N_ret]`。
- [ ] `H_state: [B,16,512]`。
- [ ] `R_t: [B,16,4096]`。

### B.2 每视频运行时状态

- [ ] fast：`W_t^(1)`、`W_t^(2)`、fast_version、update_count、skip_count。
- [ ] optimizer：本次 SGD 所需临时状态；momentum 始终不存在或为 0。
- [ ] spatial：32 slots、slot_valid_mask、overflow counter。
- [ ] temporal：最多 64 tubelets 的 cache、时间戳、valid mask。
- [ ] O1：current_visible_count、baseline_count、每槽状态。
- [ ] O2：Candidate、Confirmed CPU store、GPU Hot Cache、unique_count。
- [ ] E1：event_count、recent_event_times、cooldown/NMS state。
- [ ] E2：completed_count、phase、GRU hidden、已完成区间、recent_event_times。
- [ ] Reader：最后 operator、TimeWindow、selected ids、status、exact count、audit counters。

### B.3 Reset 验收矩阵

| 状态 | 新视频 reset | trajectory 结束释放 | batch 隔离 | 不进 model state_dict |
| :--- | :---: | :---: | :---: | :---: |
| Fast `W_t`→`W0` | [ ] | [ ] | [ ] | 临时 `W_t` [ ] |
| SGD runtime | [ ] | [ ] | [ ] | [ ] |
| Slot recurrent state | [ ] | [ ] | [ ] | [ ] |
| Temporal cache | [ ] | [ ] | [ ] | [ ] |
| E2 GRU hidden | [ ] | [ ] | [ ] | [ ] |
| O1 hard state | [ ] | [ ] | [ ] | [ ] |
| Candidate/Confirmed/Hot Cache | [ ] | [ ] | [ ] | [ ] |
| E1/E2 FSM/history | [ ] | [ ] | [ ] | [ ] |
| Reader audit | [ ] | [ ] | [ ] | [ ] |

---

## 附录 C. 端到端执行序列核对

### C.1 观测与在线更新

~~~text
输入因果 chunk
→ Qwen3-VL ViT
→ Main Visual Merger
→ Fast Adapter(W_t)
→ 空间/时间编码
→ O1/O2/E1/E2 soft observation
→ no_grad hard Bank/FSM update
→ detach 前 soft branch 计算有效 L_TTT
→ 只对两个 fast matrix 做一步 SGD
→ W_(t+1) 仅供下一 chunk
~~~

- [ ] 每个箭头都有明确接口、shape、mask 和所有权。
- [ ] hard write 和 soft loss 从同一 observation 分叉，detach 点清楚。
- [ ] 无效 loss 不更新，但 hard 因果状态按协议处理。

### C.2 回答 query

~~~text
问题 token
→ 4-layer Query Encoder
→ q_target / q_operator / q_time
→ hard operator + TimeWindow
→ 当前合法 Bank 全记录阈值检索
→ Deterministic Reader exact count
→ 16-query State Resampler
→ question + video + state + number prefill
→ 保持原 DeepStack 的 Qwen LLM
→ 自然语言答案
~~~

- [ ] hard operator 先限定 head_type，再做 q_target 语义检索。
- [ ] Reader 使用完整 records，Resampler 使用同一检索结果的软摘要。
- [ ] number 与 State Token 在 assistant answer 前注入。
- [ ] decode 期间不重复以上状态路径。

---

## 附录 D. ARCHITECTURE.md 需求追踪矩阵

| 源章节 | 需求主题 | 实施阶段 | 主要验收 |
| :--- | :--- | :--- | :--- |
| `0` | 目标与职责分离 | P0、P13、P22 | 全局契约、最终一句话核对 |
| `0.1` | 第一版固定边界 | P0–P5、P18、P20 | 配置/更新/输入契约 |
| `0.2` | 第一版明确不做 | P0、P20 | 禁止项负向扫描 |
| `1.1` | Demo video/grid/pixels | P2 | P20.1 |
| `1.2` | Demo query/Q_h | P2、P4 | Query shape 与泄漏测试 |
| `1.3` | 动态 T 与容量独立 | P1、P2、P7、P12 | 变长输入测试 |
| `2` | 总体数据流 | P3–P18 | 附录 C 端到端测试 |
| `3.1` | Qwen 基础配置 | P1、P3 | checkpoint 启动断言 |
| `3.2` | PatchEmbed/Main Merger | P2–P3 | Demo shape |
| `3.3` | State-TTT 插入点 | P3 | hook 顺序测试 |
| `3.4` | DeepStack 原路径 | P3、P13、P20 | 原模型等价测试 |
| `4.1` | Fast 张量结构 | P5 | Adapter shape/公式 |
| `4.2` | Fast/Slow 参数边界 | P5、P14 | 参数计数与 delta audit |
| `4.3` | 单步 SGD/生效顺序 | P14、P16、P18 | next-chunk 与 skip 测试 |
| `5.1` | 空间对象路 | P6 | slot/recurrent/overflow 测试 |
| `5.2` | 时间事件路 | P7 | causal/cache 测试 |
| `6.1` | 四 Decoder 输出契约 | P8 | P20.1 |
| `6.2` | O1 | P8–P9 | soft count/hard state |
| `6.3` | O2 | P8、P10 | identity 生命周期 |
| `6.4` | E1 | P8–P9 | causal TCN/FSM |
| `6.5` | E2 | P8–P9 | GRU/区间 FSM |
| `6.6` | v5 参数预算 | P1、P5–P12 | 参数预算测试 |
| `7.1` | Bank 作用与隔离 | P9 | reset/isolation |
| `7.2` | 统一记录/N_s | P9、P11 | 动态记录测试 |
| `7.3` | typed payload/semantic | P9–P10 | 字段与 projector |
| `7.4` | O2 动态容量 | P10 | >256/cache 压力测试 |
| `7.5` | 梯度边界 | P9、P14 | detach/grad audit |
| `7.6` | O1/E1/E2 hard-state FSM | P9 | `tests/test_state_bank.py` |
| `8.1` | Query 输入/池化 | P4 | padding/Bi-Attention |
| `8.2` | 三个 embedding | P4 | shape/独立 head |
| `8.3` | 9 prototypes | P4、P21 | 路由与校准 |
| `8.4` | Time Resolver | P4、P21 | TimeWindow/数值 span |
| `9.1` | 查询位置 | P11 | 禁止源测试 |
| `9.2` | 归一化余弦 | P11 | score shape/数值 |
| `9.3` | hard filters/no Top-K | P11、P20 | 过滤、empty/unsupported |
| `10.1` | 16 State Token | P12 | 0/3/30/300 records |
| `10.2` | State Token 职责 | P12–P13 | 不改变 exact count |
| `10.3` | Deterministic Reader | P12 | operator 算术/number audit |
| `11.1` | LLM 逻辑输入/长度 | P13 | payload 长度 |
| `11.2` | scatter/mask/position | P13、P20 | Composer 集成测试 |
| `11.3` | LLM 职责 | P13、P22 | Reader/LLM 一致性 |
| `12.1` | `L_pred` | P14、P16 | T<2/stop-grad |
| `12.2` | `L_id` | P14、P17 | overlap/match mask |
| `12.3` | `L_event` | P14、P17、P21 | MSE/KL/消融 |
| `12.4` | 无 O1 unlabeled | P14、P20 | loss 权重断言 |
| `13.1` | State Loss | P14–P15 | task-specific supervision |
| `13.2` | Answer Loss/三指标 | P14–P15 | metric 分离 |
| `13.3` | Meta-TTT Outer | P14、P16–P17 | after-update gradient |
| `14 Stage 0` | 数据与基线 | P2 | A0/fold/leakage |
| `14 Stage A` | 状态 Warm-up | P15 | A2 |
| `14 Stage B` | 单步 Meta-TTT | P16 | A3 |
| `14 Stage C` | 一致性/多 Support | P17 | A4/A5 |
| `14 Stage D` | 8B/分布式 | P19 | 服务器集成 |
| `15` | 测试时协议 | P18 | reset/chunk/query/generate |
| `16` | 数据与防泄漏 | P2、P18、P20、P22 | denylist/future/fold |
| `17` | 推荐模块 | P1、附录 A | 职责评审 |
| `18.1` | Demo 张量验收 | P20.1 | 自动测试 |
| `18.2` | 更新边界 | P20.2 | 自动测试 |
| `18.3` | 状态与检索 | P20.3 | 自动测试 |
| `18.4` | Reader 与输入 | P20.4 | 自动测试 |
| `19` | 最小消融 | P21.1–P21.2 | A0–A6/Q0–Q3 |
| `20` | 评估与审计 | P22 | 指标/审计包 |
| `21` | 实验待定项 | P21.3–P21.4 | frozen decision record |
| `22` | 一句话定义 | P22.5 | 发布核对 |

### D.1 覆盖审计

- [ ] 上表每个源章节都有实施阶段。
- [ ] 上表每个固定需求都有自动验收或可审计运行验收。
- [ ] 任何 `ARCHITECTURE.md` 新增章节必须先更新本矩阵再施工。
- [ ] 任何阶段删项必须说明对应源需求如何迁移，禁止无声丢失。

---

## 附录 E. 每阶段实施记录模板

复制以下模板到每次 P 的施工记录：

~~~text
阶段：
日期：
负责人：
Git branch/commit：
spec_version：
uv.lock hash：
model revision：
config path/hash：
dataset fold：
seed：

实施前基线：
- pytest：
- ruff：
- mypy：
- 关键指标：

本次完成 TODO：
- 

实际变更文件：
- 

固定契约核对：
- shape：
- mask：
- dtype/device：
- gradient/update：
- reset/isolation：
- leakage：

验证命令与结果：
- 

失败/跳过项：
- 

性能变化：
- GPU memory：
- CPU memory：
- latency：

风险与回退点：
- 

是否通过阶段门禁：yes/no
下一阶段允许开始：yes/no
~~~

---

## 附录 F. 最终施工完成总检查

- [ ] P0–P22 全部通过各自门禁。
- [ ] v5 固定契约无未解释偏差。
- [ ] 所有实验选择有训练折/校准集证据。
- [ ] 所有 hard count 可从 Reader 回溯到 records。
- [ ] 所有 fast update 可回溯到有效 `L_TTT` 和两个参数 delta。
- [ ] 所有视频开始都能证明完整 reset。
- [ ] 所有 query 都能证明没有未来帧和标签泄漏。
- [ ] DeepStack 与原 Qwen3-VL 路径一致。
- [ ] official clean test 只运行在 frozen final config。
- [ ] 代码、测试、配置、文档、checkpoint 和审计包版本一致。
