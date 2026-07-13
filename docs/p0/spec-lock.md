# P0 规格锁

## 身份

| 字段 | 锁定值 |
| :--- | :--- |
| 规范文件 | `ARCHITECTURE.md` |
| SPEC_VERSION | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| 修订日期 | `2026-07-13` |
| 文档状态 | `DOCUMENT-ONLY / UNVERIFIED` |
| ARCHITECTURE_SHA256 | `0690c9cf5d8301b644abd87deb01a0a02b3126e30eaa3d18e69b5bb105c57adc` |
| 基线 Git commit | `7f0185f8136faf88cc59e5ba2ec7309c36f8d013` |
| UV_LOCK_SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 基座模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| 模型 revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

`ARCHITECTURE.md` 是 v5 目标实现的唯一规格。若文档、配置、实现或测试与上述规范发生冲突，
当前阶段立即停止；先更新规格锁和需求追踪表，再继续施工。规格 hash 变化不得沿用旧实验 ID。

## v5 固定项

1. 基座为 Qwen3-VL-8B-Instruct；插入点在 Main Visual Merger 主输出之后、video
   `masked_scatter` 之前，DeepStack 保持原路径。
2. Fast Adapter 为 `4096→768→768→4096`；测试时只更新两个无 bias 的 `768×768`
   fast matrix，共 1,179,648 个在线参数。
3. Inner optimizer 是 learning rate `1e-4`、momentum `0`、weight decay `0`、每个有效 chunk
   最多一步、clip `1.0` 的 SGD；更新从下一 chunk 生效，每视频从 `W0` reset。
4. 空间路为 2-stage、dim 768、32 slots；时间路为 6-layer causal Transformer、dim 768。
5. O1/O2/E1/E2 只产生观测；hard Bank 保存事实，embedding 负责路由和检索，Deterministic
   Reader 使用完整 hard records 做精确算术。
6. Query Encoder 产生 target/operator/time 三个 512 维 embedding；operator 为 8 个合法类型加
   unsupported；State Retriever 使用归一化余弦阈值且不做固定 Top-K。
7. 16 个 State Token 只提供语义摘要；精确 number payload 由 Reader 给出，LLM 只负责表达。
8. query_time 之后的帧和答案/计数标签字段不进入 Bank、TTT、Retriever、Reader 或生成输入。

## 第一版禁止项

1. Surprise Gate 或学习型更新 Gate。
2. Inner AdamW、Muon、momentum SGD 或每 chunk 多步更新。
3. 让 LLM 或连续 embedding 直接回归最终累计整数。
4. 固定 Top-K 状态检索。
5. O1 无标签一致性 loss。
6. DeepStack 改造。
7. ANN 向量数据库。
8. harmful-update、margin、drift、update-norm、KL retention 等额外正则堆叠。
9. query_time 之后的帧，或在 generate decode 循环中重复 Bank/TTT 更新。

上述禁止项同时固定在 `.github/pull_request_template.md`，每次评审必须逐项确认。

## 实验待定项

以下内容不是 v5 已验证事实，只有训练折或独立校准集证据才能冻结：

- Outer Training 使用全量微调、分阶段解冻或 LLM LoRA；
- 768 与旧 512 主干的净收益，活动槽 16/32/48/64，State Token 8/16/32；
- Time Window Resolver 的 numeric span decoder；
- operator unsupported、record similarity、O1/O2/E1/E2 FSM/match/cooldown/NMS 阈值；
- E1/E2 overlap 一致性距离；
- ANN 候选召回的启用规模；
- Fast LR `3e-5`、`1e-4`、`3e-4` 的比较；
- 后续版本是否改造 DeepStack。

## 旧 v3 运行事实隔离

`configs/model_state_ttt_8b.yaml` 和 `tests/test_v3_architecture_config.py` 是基线 commit 中保留的
旧 v3 运行契约：bottleneck 512、16 slots、8 State Token 等值只描述施工起点。它们不得以 v5
运行配置、实现覆盖或实验结果的名义引用；原子迁移属于 P1。
