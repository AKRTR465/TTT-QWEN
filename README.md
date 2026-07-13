# ttt-svcbench-qwen

面向 SVCBench 的 Qwen3-VL-8B State-TTT 研究工程。

完整架构、训练协议和消融方案见 [ARCHITECTURE.md](./ARCHITECTURE.md)。当前对齐版本为
`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。

> 当前施工状态：P0–P6 已通过；P2 按用户批准的低空间口径，以合成 fold/A0 完成工程门禁，
> P3 用官方 HF meta 模块和 tiny 随机权重模型完成 Qwen 接口与 DeepStack 工程验收。真实 8B
> A0/集成仍保留在 P19/P21/P22；P4 已完成 Query Encoder、Operator Router 与 Time Window
> Resolver 的工程门禁，但尚未训练或校准。P5 Fast Adapter 已通过纯合成张量工程门禁，
> P6 空间对象编码器已通过纯合成张量工程门禁，P7 允许开始；其余空壳
> 被调用时会明确抛出 `NotImplementedError`。

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
- Inner loop固定使用无momentum、无weight decay的单步SGD，不使用Surprise Gate；
- 空间对象路使用两个参数不共享的768维Recurrent Slot Stage，默认32个活动槽；单一q投影和
  shared seed结合固定非持久sinusoidal slot code，attention先做slot轴竞争再按token归一，精确
  24,815,360参数；时间事件路使用6层、768维Pre-LN GELU因果Transformer，absolute sinusoidal
  使用显式global position id，Q/K/V/O带bias，LayerNorm eps为`1e-5`；
- P6的`required_slot_counts`只做preserve-existing/reject-excess容量审计，不表示已从视频识别
  真实对象；语义判断和hard state留给P8/P9，模型编排和受管推理生命周期留给P13/P18；
- O1/O2/E1/E2分别使用FiLM MLP、256维identity MLP、5层gated causal TCN和2层GRU；
- 时间路使用含self且含当前位置总长64的同一full/chunk滑窗；cache保存六层逐层K/V并按
  video/trajectory/query signature隔离，overlap按global position replay/replace，默认detach下一
  chunk cache；主cache严格64，另有不扩大mask的3-position replay margin用于重算固定4-tubelet
  overlap；时间元数据保持FP32/FP64并在cache中统一为FP64；时间路精确48,438,272参数；
- 当前新增模块分项合计156.703632M（156,703,632），但在线变化的仍只有约1.18M fast参数；
- 无标签TTT loss仅由当前chunk内next-tubelet prediction、O2身份一致性和E1/E2事件一致性组成；
- 问题不再通过关键词规则机械划分；Qwen input embeddings先经4096→768投影、无参sinusoidal
  position encoding和4层双向Transformer，再由三个768→1024→512 GELU输出头形成
  target/operator/time embedding；
- 计数操作由9个learned prototypes和初值1.0的可训练正温度进行语义路由；未校准时
  eval/inference显式落到unsupported；
- time embedding必须结合合法query_time、全局pointer和唯一候选受限grammar解析为确定性时间
  窗口；失败时不猜测、不clamp；
- State Bank记录通过归一化embedding检索，默认不设top-k；最终整数仍由确定性Reader计算；
- 16个learned State Query经3层Perceiver Resampler汇总全部命中记录，生成16个4096维
  State Token；它们不是Top-16记录；
- O2 Confirmed身份库从256开始按块动态增长；Candidate从64开始并设512安全上限；
- 每个新视频重置fast weights、SGD状态、时序缓存和State Bank；
- 测试时禁止使用答案、count、occurrence_times、counting_type和counting_subtype。

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
| 推荐模块导入与职责边界 | P1 已实现；P3 `qwen_adapter.py`、P4 `query_encoder.py`、P5 `fast_ttt.py` 已通过各自工程门禁，其余后续入口显式 `NotImplementedError` |
| 数据 schema、防泄漏、因果切分、processor/query token、A0 runner | P2 工程门禁已通过；fold/A0 为明确标注的合成替代 |
| Qwen video boundary、Main Merger 插入点、DeepStack 保护 | P3 已实现；tiny/meta 工程契约已验证，真实 8B 留至 P19 |
| Query Encoder、9-prototype Router、Time Window Resolver | P4 已实现；本地结构/参数/offset/fail-closed 契约已验证，模型尚未训练、阈值尚未校准 |
| Fast Adapter、per-video fast state、参数边界 | P5 已通过本地合成张量门禁；显式 functional SGD 编排留至 P14，受管在线生命周期留至 P18，真实 8B 留至 P19 |
| P6 空间对象编码器 | 已通过本地合成张量工程门禁；真实视频/8B、语义对象 overflow 与端到端 runtime 仍留后续阶段 |
| P7 时间事件编码器 | 已通过本地合成张量工程门禁；逐层 KV、overlap replay margin、因果滑窗和 runtime 隔离均已验证 |
| P8–P19 Observation、Bank、Reader、loss、训练、推理 | 计划设计，尚未实现；P8 允许开始 |
| 真实 8B、消融、校准、clean 评估 | P19–P22 计划设计，尚未运行 |

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
