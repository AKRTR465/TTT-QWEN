# ttt-svcbench-qwen

面向 SVCBench 的 Qwen3-VL-8B State-TTT 研究工程。

完整架构、训练协议和消融方案见 [ARCHITECTURE.md](./ARCHITECTURE.md)。当前对齐版本为
`state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`。

> 当前施工状态：P0 已完成规格/基线冻结；P1 已迁移 v5 运行 YAML、强类型配置、运行时类型契约
> 和模块骨架。高容量网络、状态算法、训练与推理尚未实现；对应空壳被调用时会明确抛出
> `NotImplementedError`，不能把可导入模块误报为可运行模型。

## 当前固定条件

- 基座：Qwen/Qwen3-VL-8B-Instruct；
- Transformers：4.57.1；
- Python：3.12；
- 本机PyTorch：2.9.0，CUDA 12.8 wheel；
- 主插入点：Visual Merger主输出之后、video `masked_scatter`之前；
- 第一版不修改DeepStack；
- Fast TTT Adapter为4096→768→768→4096；在线仅更新两个768×768 fast矩阵，共1,179,648
  个参数，约1.18M；
- Inner loop固定使用无momentum、无weight decay的单步SGD，不使用Surprise Gate；
- 空间对象路使用两阶段、768维Recurrent Slot Attention，默认32个活动槽；时间事件路使用
  6层、768维因果Transformer；
- O1/O2/E1/E2分别使用FiLM MLP、256维identity MLP、5层gated causal TCN和2层GRU；
- 新增模块合计约156.83M，但在线变化的仍只有约1.18M fast参数；
- 无标签TTT loss仅由当前chunk内next-tubelet prediction、O2身份一致性和E1/E2事件一致性组成；
- 问题不再通过关键词规则机械划分；Qwen问题hidden states先经4096→768投影和4层双向
  Transformer，再由三个768→1024→512输出头形成target/operator/time embedding；
- 计数操作由9个learned prototypes进行语义路由，低置信度显式落到unsupported；
- time embedding必须结合合法query_time和问题中的显式数值解析为确定性时间窗口；
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
| 推荐模块导入与职责边界 | P1 已实现；实际入口显式 `NotImplementedError` |
| 数据预处理、Qwen hook、Adapter、状态、Reader、loss、训练、推理 | P2–P19 计划设计，尚未实现 |
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
