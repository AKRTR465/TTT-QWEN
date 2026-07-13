# P0 ARCHITECTURE 需求追踪表

每个 ID 是稳定审计键。`计划设计` 来自当前规范；`已验证实现` 在对应 Part 通过门禁前一律为
“未实现”。验收位置可以是自动测试或明确的实验/审计产物。

| 需求 ID | 源章节 | 需求主题 | 实施阶段 | 主要验收位置 | 计划设计 | 已验证实现 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARCH-00-GOAL-001 | `0` | 目标与职责分离 | P0、P13、P22 | `tests/test_end_to_end_demo.py`、P22 最终一句话审计 | 已冻结 | 未实现 |
| ARCH-00-BOUNDARY-001 | `0.1` | 第一版固定边界 | P0–P5、P18、P20 | `tests/test_v5_config_contract.py`、`tests/test_inference_protocol.py` | 已冻结 | 未实现 |
| ARCH-00-NOGO-001 | `0.2` | 第一版明确不做 | P0、P20 | `tests/test_leakage_guards.py`、P20 禁止项扫描 | 已冻结 | 未实现 |
| ARCH-01-VIDEO-001 | `1.1` | Demo video/grid/pixels | P2 | `tests/test_video_preprocessing.py` | 已定义 | P2 已验证 |
| ARCH-01-QUERY-001 | `1.2` | Demo query/Q_h | P2、P4 | `tests/test_query_tokens.py`、`tests/test_query_encoder.py` | 已定义 | P2 token 范围已验证；P4 encoder 待实现 |
| ARCH-01-DYNAMIC-001 | `1.3` | 动态 T 与容量独立 | P1、P2、P7、P12 | 变长输入测试、`tests/test_state_encoder.py` | 已定义 | P2 变长预处理已验证；后续模块待实现 |
| ARCH-02-FLOW-001 | `2` | 总体数据流 | P3–P18 | `tests/test_end_to_end_demo.py` | 已定义 | P3 Qwen video boundary 已验证；P4–P18 待实现 |
| ARCH-03-BASE-001 | `3.1` | Qwen 基础配置 | P1、P3 | `tests/test_v5_config_contract.py`、`tests/test_qwen_adapter.py` | 已定义 | P3 config-only local preflight、checkpoint/runtime 断言已验证；真实 8B 留至 P19 |
| ARCH-03-MERGER-001 | `3.2` | PatchEmbed/Main Merger | P2–P3 | `tests/test_video_preprocessing.py`、`tests/test_qwen_adapter.py` | 已定义 | P2 processor 输入及 P3 官方 meta shape、packed mapping 已验证 |
| ARCH-03-INSERT-001 | `3.3` | State-TTT 插入点 | P3 | `tests/test_qwen_adapter.py` hook 顺序测试 | 已定义 | P3 已验证 Main Merger→Adapter→video masked scatter |
| ARCH-03-DEEPSTACK-001 | `3.4` | DeepStack 原路径 | P3、P13、P20 | `tests/test_qwen_adapter.py` 原模型等价测试 | 已定义 | P3 tiny HF 对象/顺序/mask/decoder 0–2 已验证；P13/P20/P19 继续回归 |
| ARCH-04-FAST-001 | `4.1` | Fast 张量结构 | P5 | `tests/test_fast_ttt.py` | 已定义 | 未实现 |
| ARCH-04-BOUNDARY-001 | `4.2` | Fast/Slow 参数边界 | P5、P14 | `tests/test_fast_ttt.py`、`tests/test_functional_sgd.py` | 已定义 | 未实现 |
| ARCH-04-SGD-001 | `4.3` | 单步 SGD/下一 chunk 生效 | P14、P16、P18 | `tests/test_functional_sgd.py`、`tests/test_inference_protocol.py` | 已定义 | 未实现 |
| ARCH-05-SPATIAL-001 | `5.1` | 空间对象路 | P6 | `tests/test_state_encoder.py` slot/recurrent/overflow | 已定义 | 未实现 |
| ARCH-05-TEMPORAL-001 | `5.2` | 时间事件路 | P7 | `tests/test_state_encoder.py` causal/cache | 已定义 | 未实现 |
| ARCH-06-OUTPUT-001 | `6.1` | 四 Decoder 输出契约 | P8 | `tests/test_observation_heads.py` | 已定义 | 未实现 |
| ARCH-06-O1-001 | `6.2` | O1 当前数量 | P8–P9 | `tests/test_observation_heads.py`、`tests/test_state_bank.py` | 已定义 | 未实现 |
| ARCH-06-O2-001 | `6.3` | O2 身份 | P8、P10 | `tests/test_observation_heads.py`、`tests/test_identity_bank.py` | 已定义 | 未实现 |
| ARCH-06-E1-001 | `6.4` | E1 点事件 | P8–P9 | `tests/test_observation_heads.py`、`tests/test_state_bank.py` | 已定义 | 未实现 |
| ARCH-06-E2-001 | `6.5` | E2 区间事件 | P8–P9 | `tests/test_observation_heads.py`、`tests/test_state_bank.py` | 已定义 | 未实现 |
| ARCH-06-BUDGET-001 | `6.6` | v5 参数预算 | P1、P5–P12 | `tests/test_v5_config_contract.py` 及逐模块参数预算测试 | 已定义 | 未实现 |
| ARCH-07-ISOLATION-001 | `7.1` | Bank 作用与隔离 | P9 | `tests/test_state_bank.py` reset/isolation | 已定义 | 未实现 |
| ARCH-07-RECORD-001 | `7.2` | 统一记录/N_s | P9、P11 | `tests/test_state_bank.py`、`tests/test_state_retriever.py` | 已定义 | 未实现 |
| ARCH-07-PAYLOAD-001 | `7.3` | typed payload/semantic | P9–P10 | `tests/test_state_bank.py` 字段/Projector 测试 | 已定义 | 未实现 |
| ARCH-07-CAPACITY-001 | `7.4` | O2 动态容量 | P10 | `tests/test_identity_bank.py` >256/cache 压力测试 | 已定义 | 未实现 |
| ARCH-07-GRAD-001 | `7.5` | 梯度边界 | P9、P14 | `tests/test_state_bank.py`、`tests/test_functional_sgd.py` | 已定义 | 未实现 |
| ARCH-08-POOL-001 | `8.1` | Query 输入与池化 | P4 | `tests/test_query_encoder.py` padding/Bi-Attention | 已定义 | 未实现 |
| ARCH-08-EMBED-001 | `8.2` | 三个独立 embedding | P4 | `tests/test_query_encoder.py` shape/独立 head | 已定义 | 未实现 |
| ARCH-08-OPERATOR-001 | `8.3` | 9 prototypes | P4、P21 | `tests/test_query_encoder.py`、P21 Q0–Q3 | 已定义 | 未实现 |
| ARCH-08-TIME-001 | `8.4` | Time Window Resolver | P4、P21 | `tests/test_query_encoder.py` TimeWindow/数值 span | 已定义 | 未实现 |
| ARCH-09-SOURCE-001 | `9.1` | 检索查询位置 | P11 | `tests/test_state_retriever.py` 禁止源测试 | 已定义 | 未实现 |
| ARCH-09-COSINE-001 | `9.2` | 归一化余弦 | P11 | `tests/test_state_retriever.py` score shape/数值 | 已定义 | 未实现 |
| ARCH-09-FILTER-001 | `9.3` | hard filters/no Top-K | P11、P20 | `tests/test_state_retriever.py` empty/unsupported | 已定义 | 未实现 |
| ARCH-10-TOKEN-001 | `10.1` | 16 State Token | P12 | `tests/test_state_reader.py` 0/3/30/300 records | 已定义 | 未实现 |
| ARCH-10-SUMMARY-001 | `10.2` | State Token 职责 | P12–P13 | `tests/test_state_reader.py` exact count 不变量 | 已定义 | 未实现 |
| ARCH-10-READER-001 | `10.3` | Deterministic Reader | P12 | `tests/test_state_reader.py` operator/number audit | 已定义 | 未实现 |
| ARCH-11-PAYLOAD-001 | `11.1` | LLM 逻辑输入与长度 | P13 | `tests/test_input_composer.py` payload 长度 | 已定义 | 未实现 |
| ARCH-11-SCATTER-001 | `11.2` | scatter/mask/position | P13、P20 | `tests/test_input_composer.py` | 已定义 | 未实现 |
| ARCH-11-LLM-001 | `11.3` | LLM 职责 | P13、P22 | `tests/test_input_composer.py`、P22 Reader/LLM 一致性 | 已定义 | 未实现 |
| ARCH-12-PRED-001 | `12.1` | L_pred | P14、P16 | `tests/test_losses.py` T<2/stop-gradient | 已定义 | 未实现 |
| ARCH-12-ID-001 | `12.2` | L_id | P14、P17 | `tests/test_losses.py` overlap/match mask | 已定义 | 未实现 |
| ARCH-12-EVENT-001 | `12.3` | L_event | P14、P17、P21 | `tests/test_losses.py`、P21 一致性距离消融 | 已定义 | 未实现 |
| ARCH-12-O1-001 | `12.4` | 无 O1 unlabeled loss | P14、P20 | `tests/test_losses.py` 权重/负向契约 | 已定义 | 未实现 |
| ARCH-13-STATE-001 | `13.1` | State Loss | P14–P15 | `tests/test_losses.py` task-specific supervision | 已定义 | 未实现 |
| ARCH-13-ANSWER-001 | `13.2` | Answer Loss/三指标 | P14–P15 | `tests/test_losses.py` 指标分离 | 已定义 | 未实现 |
| ARCH-13-OUTER-001 | `13.3` | Meta-TTT Outer Loss | P14、P16–P17 | `tests/test_losses.py` after-update gradient | 已定义 | 未实现 |
| ARCH-14-STAGE0-001 | `14 Stage 0` | 数据与 A0 基线 | P2 | P2 A0/fold/leakage 审计 | 已定义 | P2 合成工程门禁已验证；真实 8B A0 延至 P19/P21/P22 |
| ARCH-14-STAGEA-001 | `14 Stage A` | 显式状态 Warm-up | P15 | P15 A2 训练记录 | 已定义 | 未运行 |
| ARCH-14-STAGEB-001 | `14 Stage B` | 单步 Meta-TTT | P16 | P16 A3 after-update 记录 | 已定义 | 未运行 |
| ARCH-14-STAGEC-001 | `14 Stage C` | 身份/事件一致性 | P17 | P17 A4/A5 训练记录 | 已定义 | 未运行 |
| ARCH-14-STAGED-001 | `14 Stage D` | 真实 8B 与分布式 | P19 | P19 服务器集成报告 | 已定义 | 未运行 |
| ARCH-15-INFER-001 | `15` | 测试时协议 | P18 | `tests/test_inference_protocol.py` | 已定义 | 未实现 |
| ARCH-16-LEAK-001 | `16` | 数据与防泄漏 | P2、P18、P20、P22 | `tests/test_svcbench_data.py`、P22 clean 审计 | 已定义 | P2 四层 payload guard 已验证；后续协议待复验 |
| ARCH-17-MODULE-001 | `17` | 推荐模块与职责 | P1、附录 A | `tests/test_v5_config_contract.py`、P1 职责审计 | 已定义 | P1 职责已验证；P3 `qwen_adapter.py` 已实现，其余模块待实现 |
| ARCH-18-DEMO-001 | `18.1` | Demo 张量验收 | P20.1 | `tests/test_end_to_end_demo.py` | 已定义 | 未实现 |
| ARCH-18-UPDATE-001 | `18.2` | 更新边界验收 | P20.2 | `tests/test_functional_sgd.py` | 已定义 | 未实现 |
| ARCH-18-STATE-001 | `18.3` | 状态与检索验收 | P20.3 | State/Identity/Retriever 回归套件 | 已定义 | 未实现 |
| ARCH-18-INPUT-001 | `18.4` | Reader 与输入验收 | P20.4 | Reader/Composer/DeepStack 回归套件 | 已定义 | 未实现 |
| ARCH-19-ABLATE-001 | `19` | 最小消融 | P21.1–P21.2 | P21 A0–A6、Q0–Q3 实验表 | 已定义 | 未运行 |
| ARCH-20-AUDIT-001 | `20` | 评估与审计 | P22 | P22 指标 JSON 与审计包 | 已定义 | 未运行 |
| ARCH-21-DECIDE-001 | `21` | 仍需实验决定 | P21.3–P21.4 | P21 frozen decision record | 已定义 | 未运行 |
| ARCH-22-SUMMARY-001 | `22` | 一句话定义 | P22.5 | P22 发布报告核对 | 已定义 | 未运行 |

## 覆盖规则

1. 上表源章节集合必须与 `TODO.md` 附录 D 一致，并覆盖顶层 0–22。
2. 新增或删除 `ARCHITECTURE.md` 章节时必须在施工前更新此表和 TODO 附录 D。
3. 自动测试尚未创建时，表中的路径表示后续阶段的强制交付位置，不表示当前已有实现。
4. 实验项必须指向带 fold、seed、模型 revision、配置 hash 的审计产物，不能以口头结论验收。
