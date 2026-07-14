# P0 规格锁

## 身份

| 字段 | 锁定值 |
| :--- | :--- |
| 规范文件 | `ARCHITECTURE.md` |
| SPEC_VERSION | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| 修订日期 | `2026-07-14` |
| 文档状态 | `DOCUMENT-ONLY / UNVERIFIED` |
| ARCHITECTURE_SHA256 | `efd613bc0f73aba8f66c18c2e03692c88762320288b110d897ac8e2a8fb7442a` |
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
4. 空间路为两个参数不共享的 Slot Stage、dim 768、32 slots：单一 q projection、shared seed、
   fixed non-persistent sinusoidal code、slot-axis competition 后 token normalization，精确
   24,815,360 参数；时间路为 6-layer Pre-LN GELU causal Transformer、dim 768，以无参 absolute
   sinusoidal、显式 global position、含 self 的 64-position 窗口和逐层 KV cache 实现；固定
   4-tubelet overlap 使用不扩大 mask 的 3-position replay margin 重算，精确 48,438,272 参数。
   P6 的显式 required-slot overflow 只做容量审计，真实对象语义留给 P8/P9。
5. P8 四个 Head 的 LayerNorm `eps=1e-5`、无 dropout、标准层均带 bias；仅 O1 直接读取
   q_target 并使用 `1+scale` FiLM，E1/E2 读取 P7 已 query-conditioned 的 H_t。O2 有效 identity
   使用 FP32 L2 和 unit-basis 零范数回退。E1 使用 RF63 与无参 66-position projected-history；
   E2 使用单向 batch-first GRU 与 5 个 rollback checkpoint；二者 replay 4-position overlap 并按
   video/trajectory/query signature 隔离。精确参数依次为 2,632,710、2,103,042、9,584,643、
   7,094,792，P8 四头合计 21,415,187。
6. 四个 Head 只产生 raw-logit soft observation 和 debug probability/mask/timestamp/global
   position；invalid 清零，在线只冻结 Head 参数而不使用 `torch.no_grad()` 或 detach 输入。
   hard Bank 保存事实，embedding 负责路由和检索，Deterministic Reader 使用完整 hard records
   做精确算术。
7. P9 Semantic Projector 固定四个 768 维 head embedding 和共享 `768→1024→512` SiLU trunk，
   精确 1,316,864 参数；Projector 进入模型 state_dict/Outer optimizer，Bank/FSM/runtime 全部
   零参数、零模型持久化并通过独立 snapshot 恢复。O1 六阈值为 0.5 且 baseline 显式 set once；
   E1 使用 0.7/0.3 hysteresis、0.7 completion/transition 和 0.5 秒 cooldown/NMS；E2 使用
   phase-gated 三步 FSM，并在 INACTIVE phase 与全部 event probability 不高于 0.5 时 re-arm。
   当前新增模块分项和为 156,715,683。
8. Query Encoder 产生 target/operator/time 三个 512 维 embedding；operator 为 8 个合法类型加
   unsupported；State Retriever 使用归一化余弦阈值且不做固定 Top-K。
9. 16 个 State Token 只提供语义摘要；精确 number payload 由 Reader 给出，LLM 只负责表达。
10. query_time 之后的帧和答案/计数标签字段不进入 Bank、TTT、Retriever、Reader 或生成输入。

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
- 是否用训练折证据替换或增强 P4 的“双 pointer + 唯一候选受限 grammar” baseline；
- operator unsupported、record similarity、O1/O2/E1/E2 FSM/match/cooldown/NMS 阈值；
- E1/E2 overlap 一致性距离；
- ANN 候选召回的启用规模；
- Fast LR `3e-5`、`1e-4`、`3e-4` 的比较；
- 后续版本是否改造 DeepStack。

## 旧 v3 运行事实隔离

`configs/model_state_ttt_8b.yaml` 和 `tests/test_v3_architecture_config.py` 是基线 commit 中保留的
旧 v3 运行契约：bottleneck 512、16 slots、8 State Token 等值只描述施工起点。它们不得以 v5
运行配置、实现覆盖或实验结果的名义引用；原子迁移属于 P1。
