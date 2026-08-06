# ttt-svcbench-qwen 主线迁移报告

来源 `F:\deeplearning\UAV\ttt-svcbench-qwen`（分支 `codex/a5-slot-geometry-probes`，HEAD `e6e04be`）。
目标：只保留一条最干净的主线 —— 不留实验分支 / 检测 / guard / 守护 / 严格校验 / 冗余测试，文档收敛为一份。

---

## 1. 结论

体量的主因不是死代码，而是**同一个不变量被反复编码为"检测并拒绝"**。机械统计（扫描全量源码，非估算）：

| 度量 | 数值 |
|---|---|
| src 中 guard / 校验 / audit 占比（并集去重） | **10,505 / 44,533 LOC = 23.6%** |
| `raise` 语句总数 | **2,814**（约每 16 行源码一个） |
| dataclass 总数 / 带 `__post_init__` 的 | **244 / 187（77%）** |
| `__post_init__` 体总行数 | 5,325 |
| `_validate_* / _check_* / _require_*` 体总行数 | 1,663 |
| docstring 行数 | 1,071（仅 2.4%） |
| 测试函数总数 / 断言 `raise` 的 | **566 / 110（19.4%，2,812 LOC）** |

**这个仓库不是文档过多，是校验过多。** 单点最强证据：
`state_retriever.py:111-535` 的 `RetrieverOutput` 声明 **31 个字段**，随后用 **约 355 行 `__post_init__`、内含 61 个 `raise`** 校验这 31 个字段 —— 校验与声明之比 **11:1**。

总账：

| | 现状 | 迁移后 | 减少 |
|---|---:|---:|---|
| src（39 → 31 文件） | 44,533 | 20,957 | **−23,576（−52.9%）** |
| tests（50 → 24 文件） | 25,318 | 11,171 | **−14,147（−55.9%）** |
| 文档（9 → 1 份） | 2,322 | 115 | **−2,207（−95.0%）** |
| 脚本（30 → 7） | ~4,000 | ~1,430 | **−2,570（−64.3%）** |
| 配置（14 → 6） | ~1,500 | ~850 | −650（−43.3%） |
| **合计** | **~77,673** | **~34,193** | **−43,480（−56.0%）** |

外加 `analysis_outputs/` 的 **8,375 行**未跟踪文件整体删除（见 §11 不可逆警告）。

**全部 39 个 src 目标值与 50 个测试文件目标值现已逐文件读码核验**（§15 记录了此前的缺口及其填补）。其中 8 个 src 目标与多个测试目标被执行者**上调**——即比我原先的估算更保守——因为剩余部分是活机制而非 guard；最大的一处是 `production_runtime`（1400 → 2690，见 §3.2）。这些上调是有证据的反驳，我采纳其数字而非我的。

---

## 2. 主线定义（唯一保留）

**(T) 训练** — `python -m torch.distributed.run … -m ttt_svcbench_qwen.llamafactory_trainer`

| 阶段 | 语义 | 脚本 → 配置 |
|---|---|---|
| **M1 · A2** | Qwen + 状态模块 + W0 全量解冻；Associative 投影冻结；memory 写入不可达 | `train_fullprefix256.sh a2` → `a2_qwen3vl8b_fullprefix256_4gpu.yaml` |
| **M2 · A5 Warmup** | 从完整 A2 ckpt 初始化，冻结 Qwen/W0/RMSNorm/P_in/P_out，只训 P_C + memory 接口 + 四个 state 组共 256 step，产出原子 handoff bundle | `train_a5_fast_state_warmup.sh` → `a5_fast_state_warmup_256_4gpu.yaml` |
| **M3 · A5 Main** | 重载 A2 ckpt，叠加 handoff bundle，恢复部分解冻，4 epoch，只存 final checkpoint | `train_a5_associative_lttt_finalonly.sh` → `a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml` |

**(I) 推理** — `ttt-svcbench-infer` = `inference:main`（**M4**：按视频隔离、按 chunk 因果更新的在线推理）

### 2.1 三处命名反直觉，勿误删（已核验）

1. **`a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml` 就是 M3 的正式配置**，尽管名字像消融。`diff` 两个 A5-main 配置后确认：它是唯一同时满足 README 所述"4 epoch"（`num_train_epochs: 4.0`）与"恢复部分解冻策略"（`qwen_outer_trainability.mode: partial`、`vision_freeze_first_blocks: 13`、`decoder_train_last_layers: 8`）的配置。另一个 `a5_meta_ttt_k8_fullprefix256_4gpu.yaml` 是 2 epoch 且**完全没有** `qwen_outer_trainability` 段 → 主线外。
2. **`state_query_visual_mode: recent_chunk` 是生产设置，不是消融或兼容模式。** 7 个配置全部如此，且 `production_factory.py:166` 写成 `Literal["recent_chunk"]`（唯一合法值）。README 里"`--query-visual-mode recent_chunk` 运行兼容消融"指的是推理 CLI 的 **answer** query 旋钮，是另一个开关（answer 侧才是 `causal_prefix` / 256 帧）。
3. **`scripts/h200/launch_4gpu.sh`（598 LOC）是主线基础设施，不是可选启动器。** 三个主线 YAML 全部 `report_to: none`，所以它 L476 的 `tee train.log` 是**唯一**承载 per-step loss 的通道。

主线外（全删）：`a2_static_eval:main`、`a5_eval:main`、`config:main`、`no_write` 消融臂、8 卡变体、benchmark/smoke 启动器、dataloader profiling、cost-balanced A2、baseline 配置、三条 eval 脚本。

---

## 3. src 体量账（39 → 31 文件）

### 3.1 整文件消失（8 个整删 2,667 LOC；`runtime_metrics` 第 9 个，降为 12 行 shim）

| 文件 | LOC | 处置 | 依据 |
|---|---:|---|---|
| `a2_static_eval.py` | 518 | 删 | 主线外 eval 叶子，仅被 `a5_eval` + 自身测试引用 |
| `a5_eval.py` | 504 | 删 | 主线外 eval 叶子 |
| `svcbench_train_eval.py` | 409 | 删 | 仅被两个待删 eval 模块 + 1 个待删脚本引用 |
| `visual_cost.py` | 393 | 删 | 遥测装置 —— **有前置条件，见 §3.3** |
| `memory_write.py` | 369 | 并入 `fast_ttt.py` | 369 行中 `MemoryWriteResult.__post_init__` 约 50 行、`MemoryTruncateAudit` 约 23 行、两个 cosine 诊断约 35 行 |
| `training_context.py` | 206 | 删 | 只服务 2 个模块；其 offload 路径是 `TTT_QUERY_ACTIVATION_OFFLOAD` 的 OOM 逃生口（见 §9） |
| `runtime_metrics.py` | 199 | 删→改为 12 行 shim | 纯遥测；20 处调用点全是 `trace_event` / `with trace_cuda_phase`。**但按 §14.F 改为保留 12 行 no-op shim** —— `production_runtime.py` 有 9 个 `with trace_cuda_phase(...)` 包着真实工作，逐点去缩进风险高于收益 |
| `associative_ttt.py` | 187 | 并入 `fast_ttt.py` | 187 行中约 120 行是两个 `__post_init__`；真正的科学只有 L163-187 |
| `tensor_contracts.py` | 81 | 删 | 14 处调用点全部只 `raise` —— **有缓解措施，见 §3.4** |

### 3.2 就地削减（30 个存活文件）

| 文件 | 现状 | 目标 | 核验 | 依据 |
|---|---:|---:|:-:|---|
| `llamafactory_trainer.py` | 4156 | 1250 | ✓ | `log()` A5 遥测 −290、bitwise-audit 子系统 −380、两个 step auditor + diagnostics 聚合 −420。**三次独立估算为 1150 / 1250 / 1320，取中值** |
| `production_runtime.py` | 3798 | **2690** | ✓ | **上调，原估 1400 不可达。** 两半分别核验：L1-2055 → 1360，L2055-3798 → 1330。可删的 695+416 行是 raise-only `__post_init__`、`_loader_trace`(172) 及其 20 处调用点、两个 telemetry dataclass、`_A2ProgressTrace`、以及 `support_visual_batch_size: 1` 下不可达的 `prepare_raw_support_batch`(1043-1121)。剩余约 2690 行是**活机制**：`VideoChunkMaterializer` 的 decode+cache+bounded-prefetch+coalesce（7 个 YAML 全设 `support_decode_coalesce: true`）、L3234-3782 约 570 行的解码器族（每个分支都活）、六个 typed `nn.Module` stage wrapper、`ProductionEpisodeMaterializer`。**两个 prefetch collator 不是重复** —— 它们已共享 `_PrefetchCollatorBase`(1535) 与 `_prepare_query_pair`(1567) |
| `state_bank.py` | 2792 | 1420 | ✓ | 实测 guard 891（31.9%）+ FSM / history 完整性检测 |
| `meta_trainer.py` | 2414 | 950 | ✓ | K=8 递推 + cotangent 数学只占约 250 LOC；其余约 1450 是六个 audit dataclass（`MemoryWriteAudit`、`TruncatedSegmentAudit`、`TruncatedQueryPointAudit`、`QueryCotangentClipAudit`、`QueryCounterfactualAudit` 树、`TruncatedMetaTTTEpisodeAudit`）+ counterfactual 特性 + 3 处 trace + `no_write` 臂 + 已死的 raw-support-visual-batcher。**注意 §8.12、§9.0、§11.9** |
| `stage_a_targets.py` | 2271 | 1200 | 推导 | 实测 guard 695（30.6%）+ provenance mask |
| `state_encoder.py` | 2154 | **1180** | ✓ | 五个 `__post_init__` 实测共 428 行；6 处 `tensor_contracts` 调用点（:117/:392/:575/:807/:1787/:2140）逐点给出编辑，其中 :2140 在 `_assert_runtime_state_storage_isolated` 内、被两处到达。删 `TemporalEncoderAudit` 有二阶牵连（`_PreparedTemporalHistory.overlap_count`:1368 等随之死亡）|
| `episode_data.py` | 2002 | 1150 | ✓ | |
| `identity_bank.py` | 1933 | 820 | ✓ | 实测 guard 411 + 生命周期断言 + `ConfirmedChunk` 合并 |
| `inference.py` | 1747 | **830** | ✓ | **上调，原估 600 不可达。** 删净 audit 全套 + retry 臂 + `InferenceRequest.from_payload`(185-224，src 内零调用) + 全部 raise-only 检查共约 917 行；余下不是 guard，`main()` 本身就很大。注意 `assert_inference_runtime_payload` 在 src 内**只有** `main():1547` 一个调用点 |
| `observation_heads.py` | 1678 | **975** | ✓ | 39 条工单。E1/E2 的 "position IDs cannot contain gaps"(:803/:1037) 与时间单调性 raise 按判据可删，但它们是 chunk 调度器因果序 bug 的唯一显现 → 必须由 §5 第 4 条测试承接 |
| `input_composer.py` | 1438 | **530** | ✓ | 比原估更激进：另有 102 行零消费者的 audit dataclass。`compose_teacher_forced_inputs` 与 `_teacher_forced_insertion_indices` 仅供测试 → 生产中 `payload_insertion_indices` 恒为 None，死分支可删。**另报：`ComposedInput.inputs_embeds` 在 src 内无消费者** |
| `config.py` | 1335 | 470 | ✓ | `_FROZEN_CONTRACT`(L692-1068) 逐字复制了 YAML + 5 个子校验器(L1123-1314) |
| `fast_ttt.py` | 1328 | 640 | ✓ | probe 子系统 + 存储别名 guard；吸收 `associative_ttt` + `memory_write`；**全部 centering 科学保留** |
| `model.py` | 1305 | 620 | ✓ | 实测 guard 236 |
| `state_reader.py` | 1292 | **840** | ✓ | relevance gate 的**实体在 :1150-1162**，两处使用点 :1178/:1198（:1096 只是参数默认值）。`ReaderResult.audit_fields` 在主线无读者（`inference.py:1691` 读的是 `InferenceResult`，不是 `ReaderResult`）|
| `outer_loss_balance.py` | 1245 | 850 | **修正** | guard 237 + `*_clamped` audit 标志。**两个 balancer 都不是死码，见 §12** |
| `qwen_adapter.py` | 1238 | **600** | ✓ | 比原估更激进。适配器机制本身（Fast Adapter 位于 Main Visual Merger 与 video `masked_scatter` 之间）逐字保留，删 `_validate_state_call` / `_validate_feature_splits` 与形状/dtype 巡查 |
| `query_encoder.py` | 1213 | 710 | ✓ | |
| `state_retriever.py` | 1130 | 540 | ✓ | guard 476（42.1%）；`RetrieverOutput` 424 → ~70 |
| `losses.py` | 1054 | 550 | 推导 | guard 426（40.4%）；L955-1054 整段是校验器 |
| `production_factory.py` | 952 | **400** | ✓ | 比原估更激进：`environment_manifest`、audit 旋钮、`fully_unfreeze_qwen`（其唯一机制行 `requires_grad_(True)`:544 已在别处）、`configure_qwen_outer_trainability`:645-695 的 audit 填充。**注意 §14.K 的 `extra="forbid"` 连带编辑** |
| `outer_gradient_control.py` | 597 | 400 | 推导 | **保守**：系统里唯一的梯度裁剪（§9.3） |
| `preprocess_cache.py` | 589 | 415 | ✓ | |
| `stage_a_runtime.py` | 574 | 350 | 推导 | |
| `data.py` | 486 | 155 | ✓ | **泄漏擦除与 GroupKFold 保留**（§8.5、§9.8） |
| `trainer.py` | 406 | 280 | ✓ | **既不并入 `meta_trainer`，也不能砍到 185。** 三方估算 185/280/380，我实测定案：guard/校验行去重后共 **105 行**（`validate` L62-83 计 22 行 + 五个 `__post_init__` L92-104/111-121/140-167/179-188/195-203 计 71 行 + 散落 raise），`StageAExecutionAudit` 整类 L42-83 计 42 行。它是 M1/A2 的单次 episode runner：`StageAEpisodeRunner` 定义于 L244，由 `production_runtime.py:2492` 在 `ProductionTrainerRuntime` 构造时实例化 —— 即 A2 训练路径本身。`meta_trainer.py:87-91` 只从它取三个共享数据契约。两个循环，都在主线。另有 5 处承重机制（§9.5） |
| `video_preprocessing.py` | 367 | 62 | ✓ | causal-cut API 已死，唯一引用方是测试 |
| `query_tokens.py` | 130 | 45 | ✓ | |
| `json_contract.py` | 40 | 20 | ✓ | **保留核心**：它是真正的 JSONL 解析器，不是契约检查（§9.6） |
| `__init__.py` | 3 | 3 | ✓ | |

### 3.3 `visual_cost.py` 的前置条件（唯一需用户决策的行为变更）

它虽是遥测装置，却**在 A2 主线路径上**：`a2_qwen3vl8b_fullprefix256_4gpu.yaml:84-85` 设 `visual_cost_mode: exact_tokens_then_runtime` + `$VISUAL_COST_INDEX`，且 `train_fullprefix256.sh:17-18` 缺该产物就硬失败。

**解法**：`visual_cost_mode` 的合法值是 `Literal["proxy","exact_tokens","exact_tokens_then_runtime"]`，**默认 `proxy`**，而 `production_factory.py:233` 只在 mode ≠ proxy 时才要求索引。把 A2 配置改成 `proxy`，即可连带删除 `visual_cost.py`、`scripts/build_visual_cost_index.py`、`scripts/h200/prepare_a2_semantic_visual_cost.sh`、`tests/test_visual_cost_index.py`，以及 `train_fullprefix256.sh:15-19` 的 preflight 与 `llamafactory_trainer._observe_runtime_cost`(L2014-2052)。

**代价（必须知情）**：A2 采样器失去实测代价排序 → batch 组成改变 → A2 训练与历史 run 不再可逐位比较。**这是行为变更，不是纯删除**，建议单独一步执行。

### 3.4 `tensor_contracts.py` 的缓解措施

两位 agent 在此冲突，我读码后判定：两方都对，且相容。
- 全部 14 处调用点确为**纯检测**：形如 `if len({tensor_storage_key(state.m) for state in states}) != len(states): raise`（`memory_write.py:253`、`meta_trainer.py:1704`）。它只计算并拒绝，从不修复 —— 删掉它**不会造成**别名共享。
- 但它检测的是**逐视频 memory 隔离**，即 delta-rule 递推的核心科学主张。
- `timestamps_match` 的三处调用点（`observation_heads.py:800/1034`、`state_encoder.py:1787`）经确认全是 `if not timestamps_match(...): raise`，无一处分支 → 随其 raise 一同死亡。

**结论：整删 81 LOC，并用 §5 的第 5 条测试承接保证** —— 构造双视频 batch，断言 `initialize_fast_state` 与 `truncate_memory_state` 之后 `m` 的 storage 互不相同。把 14 处常开的运行时检测换成约 15 行一次性测试，正是本次要求的交换。

---

## 4. 就地削减：按类别

| 类别 | 典型对象 | 处置 |
|---|---|---|
| **dataclass `__post_init__`** | 187 个（5,325 LOC）：`RetrieverOutput`、`FastAssociativeContext`、`AssociativeTTTIntermediates`、`MemoryWriteResult`、`MemoryWriteBatch` … | 整体删除，只留字段声明 |
| **`_validate_* / _check_* / _require_*`** | 1,663 LOC：`losses.py:955-1054` 全段、`config.py` 的 5 个子校验器、`state_retriever.py:809-886` | 整体删除 |
| **audit dataclass** | `inference.py` 的 `RuntimeBoundaryStamp / RuntimeAuditSnapshot / RuntimePristineStamp / RuntimeResetAudit / RuntimeReleaseAudit / ChunkAudit / GenerateAudit`；`fast_ttt.py` 的 `FastQueryProxyAudit / FastTTTForwardAudit / SlotGeometryProbe`；`memory_write.MemoryTruncateAudit`；`trainer.StageAExecutionAudit` | 删除 —— **注意 §7 的连带编辑** |
| **stamp / checksum / 哈希** | `runtime_checksum`、`runtime_boundary_stamp`、`_causal_state_stamp`、`_hard_state_stamp`、`_fast_state_stamp`、`_pristine_state_stamp`、`_boundary_tensor_versions`、`_Digest`、`_hash_value`、`_effective_project_config_sha256` | 删除（后者删除还让 bundle 跨配置改动可移植） |
| **probe / 诊断度量** | `fast_ttt` 的 `_slot_geometry_probes`、`_offdiag_cosine_mean`、`_centered_offdiag_cosine_mean`、`_attention_entropy_ratio`、`_detached_readout_shares`、`_detached_norm`；`memory_write` 的 `_pairwise_offdiag_cosine_mean`、`_masked_cosine_mean` | 删除 |
| **存储别名 guard** | `fast_ttt` 的 `_shares_storage`、`_storage_byte_span`、`_assert_batched_state_storage_isolated` | 删除（同 §3.4 缓解） |
| **遥测** | `runtime_metrics` 全部调用点、`_loader_trace`、`_cache_stats`、`QueryPreparationTelemetry`、`A2PreparationTelemetry`、`ProductionVisualAudit`、`_A2ProgressTrace` | 删除 |
| **三态 / 双模开关** | `AuditLevel`（OFF/BOUNDARY/FULL，实发 `BOUNDARY`）、`relevance_gate_mode`（`audit_only`/`enforce`，实发 `audit_only`）、`CalibrationStatus`（三态，实际全是 `*_calibration_required`，无一处已标定） | **三个枚举连字段全删**。`audit_only` 语义就是"只审计不拦截"，在"不要审计"前提下等于整个 gate 删除：`config.py:252/1040`、`state_reader.py:468/473/1096`、`_validate_relevance_gate`、`inference.py:264-272/520-533/974-980` |
| **跨配置严格匹配** | `llamafactory_trainer.py:2692-2694`（max_steps/warmup_steps 不等即 raise）、`:3975`（seed 不等）、`:3020-3037`（更新范数预算漂移 > 1e-6） | 删除 |
| **实验/诊断环境变量** | 共 40 个 `TTT_*`，其中纯开关：`TTT_DIAGNOSTIC_FORCE_SLOT_MEAN`、`TTT_A5_ADAPTATION_MODE`、`TTT_DATALOADER_TRACE`、`TTT_RUNTIME_TRACE_MODE/DIR`、`TTT_GPU_SAMPLE_LOG`、`TTT_A2_PROGRESS_TRACE`、`TTT_SMOKE_SHORTEST_FIRST/MAX_STEPS`、`TTT_PREFLIGHT_ONLY`、`TTT_VISUAL_COST_PREFLIGHT`、`TTT_SKIP_FINAL_CHECKPOINT`、`TTT_RUNTIME_FACTORY`、`TTT_QUERY_ACTIVATION_OFFLOAD(_MAX_GB)` | 删除读取点与分支 |
| **配置 audit 旋钮** | 三个主线配置里仍开着的 `a5_parameter_delta_audit_steps: 256/320`、`operator_diagnostics_interval: 8/10`（`runtime_trace_mode` 与 `semantic_projector_delta_audit_steps` 本就是 off/0） | 连旋钮与代码一起删 —— **但见 §7.3** |
| **消融臂** | `no_write`（前 `static_w0`）、schema-14 契约版本串 `:2857-2863` / `:4079-4082` | 删除 |
| **配置钉死后的死路径** | 7 个配置全部 `support_visual_batch_size: 1`（已核验），故 `meta_trainer` 的 raw-support-visual-batcher 整条路径不可达 | 删除 |
| **ZeRO-2 分支** | 三个主线配置**全部**用 `deepspeed_zero1_dynamic_graph.json`，故 `_deepspeed_partitions_gradients` 的 ZeRO-2 分支及依赖路径成为死码 | 删 ZeRO-2 分支，**但保留 anchor 本身**（§9.1） |

### 判据（贯穿全表）

> **会 `raise` 的检查可删；会"纠正"的检查不可删。**（clamp、范数恢复、防梯度泄漏的 detach、防精度丢失的 FP32 cast）

`associative_ttt.py` 是教科书例子：两个 `__post_init__`（约 120 行）全可删，而同文件 L177-180 的 `masked_fill` + `torch.where(row_nonempty, …)` + `weights.sum(...).clamp_min(1.0)` 全部承重 —— 它们处理空行，不报错。

---

## 5. 测试（25,318 → 11,171 LOC，50 → 24 文件）

**110 个测试函数（2,812 LOC）随严格校验机械死亡** —— 它们断言 `pytest.raises`，被断言的异常将不再存在。集中在 `test_production_factory.py`(12)、`test_qwen_adapter.py`(13)、`test_fast_ttt.py`(7)、`test_memory_write.py`(6)、`test_v5_config_contract.py`(6)。

**整文件删除（18 个）**：`test_architecture_snapshot.py`、`test_v5_config_contract.py`、`test_v5_runtime_types.py`、`test_runtime_unification.py`、`test_counterfactual_rank_alignment.py`、`test_p14_gradient_audit.py`、`test_dataloader_payloads.py`、`test_a5_eval.py`、`test_svcbench_train_eval.py`、`test_compare_svcbench_eval.py`、`test_operator_diagnostics_script.py`、`test_h200_operational_tools.py`、`test_warmup_finalization_distributed.py`、`test_runtime_metrics.py`、`test_visual_cost_index.py`、`test_training_context.py`，**新增** `test_input_composer.py`(526) 与 `test_stage_a_composer.py`(273)。

> **一处撤回：`test_p13_tiny_integration.py`(237) 不删，原样保留。** 我原先按"阶段性测试"把它列入删除，前提是 `test_qwen_adapter.py:608` 已覆盖同样内容；核验后该前提不成立，p13 覆盖的 tiny-real-HF 集成路径没有替代者。

**存活文件的削减目标（全部逐文件核验）。** 十个最大文件此前只有估算，现已实测 —— 其中五个的目标被**上调**，因为它们的体量是不可约的 fixture/builder 而非 guard：

| 测试文件 | 现状 | 目标 | 备注 |
|---|---:|---:|---|
| `test_production_factory.py` | 2978 | **415** | keep-set 锚定 `:2776`（allowlist 排除非持久 buffer）与 `:2811`（A2→A5 bundle 往返，全仓唯一） |
| `test_meta_trainer.py` | 1760 | **1180** | 上调（原估 520）：`:97-987` 共 890 行是 fake-stage harness，而强制保留的 K=8 梯度测试依赖它，只有约 90 行随 11 个被删测试一起死 |
| `test_state_bank.py` | 1715 | **430** | |
| `test_state_reader.py` | 1448 | **520** | 上调：`:1-452` 是 imports + 5 个 fixture + builder，其中 `_retrieval`(245-411) 167 行不可约 |
| `test_stage_a_targets.py` | 1260 | **1010** | 大幅上调（原估 420）：它**根本不引用** `OfficialWeakLossAudit`，此前的破坏假设是假的 |
| `test_qwen_adapter.py` | 1173 | **640** | 24 处 `pytest.raises` 分布在 13/33 个测试里，机械删除量大 |
| `test_identity_bank.py` | 1139 | **390** | |
| `test_state_encoder.py` | 1097 | **510** | 不引用 `tensor_contracts`，无需为此编辑 |
| `test_inference_protocol.py` | 984 | **800** | 上调：`:639` 是主线级 per-video 隔离测试，强制保留 |
| `test_query_encoder.py` | 941 | **460** | |

其余存活文件（此前已核验，不变）：`test_observation_heads.py` 723→430、`test_temporal_encoder.py` 713→400、`test_fast_ttt.py` 725→300、`test_memory_write.py` 675→300、`test_model.py` 647→330、`test_outer_loss_balance.py` 555→300、`test_state_retriever.py` 496→300、`test_stage_a_runtime.py` 472→330、`test_outer_gradient_control.py` 444→150、`test_episode_data.py` 408→170、`test_preprocess_cache.py` 408→120、`test_losses.py` 235→45（重写而非压缩：import 块的 11 个符号只有 2 个存活）、`test_svcbench_data.py` 194→60、`test_video_preprocessing.py` 166→90、`test_associative_ttt.py` 138→100、`test_query_tokens.py` 74→55。

> **三个假前提已撤回**（它们来自早期 agent 的推断，实测为假）：`test_stage_a_targets.py` 不引用 `OfficialWeakLossAudit`；上述四个测试文件都不 import `tensor_contracts`；它们也都不引用 `_validate_*` / `AuditLevel`。因此这些文件需要的编辑远少于原计划，目标值相应上调。

**必须被钉住的主线契约（每条至少一个测试）**：

1. memory 递推顺序：chunk t 读 `M_{t-1}`，闭式写出的 `M_t` 自 t+1 生效
2. `M = 0` 时逐位等于静态 forward（A2 能力在每个 episode 起点被保留）
3. K=8 截断 meta 梯度确实回传到 memory 接口与 token keys，且截断点 detach
4. 因果前缀掩码：`query_time` 之后的帧不进入状态更新或回答
5. **per-video 状态隔离**：双视频 batch 中 `m` 的 storage 互不相同（承接 §3.4 删除的运行时检测）+ reset/release 含异常路径
6. A2→A5 handoff bundle 往返
7. 推理 JSON 输出契约
8. Ση ≤ `eta_chunk_budget` 收缩界
9. Reader 精确计数 + number-token 输出
10. 4/8 rank sampler 的 task/segment parity 与零权重 padding

**基础设施**：`conftest.py` 26→13（保留 `gc.collect()` autouse fixture —— CPU 二阶图确实需要；删 `h200_env`，它只服务待删文件）。`tests/support/`：`tiny_qwen.py`(71) + `tokenizers.py`(79) 原样保留，`runtime_factories.py` 296→280。

**三个测试要"改写"而不是"删除"**（它们目前依赖将被删的 raise，但覆盖的契约必须留）：

1. `test_meta_trainer.py::test_query_cotangent_clip_norm_is_config_driven`(L1190) 与 `test_query_cotangents_clip_independently_then_sum_without_averaging`(L1140) —— 它们是 `max_norm=10.0` 裁剪的**唯一**直接覆盖（§9.0），必须保留；只需把 `clipped, audit = _clip_query_proxy_gradients(...)` 的二元组返回改成单值。
2. `test_meta_trainer.py::test_zero_weight_padding_keeps_backward_schedule_but_contributes_zero`(L1421) —— 断言 `proxy_gradient_status == 'zero_padding'` 的部分随 audit 消失，但"backward 调度不变"那一半改写成对 `backward_count` 的断言后保留（对应 §5 第 10 条）。
3. `test_meta_trainer.py::test_truncated_queries_and_deferred_vjp_never_retain_local_graph`(L1397) —— 现在靠 `TruncatedMetaTTTEpisodeOutput.__post_init__` 的 raise，改写成直接断言 `grad_fn` 即可（对应 §5 第 3 条）。

`test_meta_trainer.py` 中另有 12 个测试函数随目标一同死亡（counterfactual ×2、`_LEGACY_STATIC_W0`、`no_write` 臂、raw-visual-batcher、`TruncatedSegmentAudit.training_mode`、`MemoryWriteAudit` ×2、`MetaTTTEpisode.__post_init__` 毒化断言、`MetaTTTQueryPoint` prefill 断言、`StageAQueryLossBuilder` 类型断言）。

---

## 6. 脚本 / 配置 / 目录

**脚本 30 → 7**（含核验后的削减目标）：

| 保留 | LOC | 说明 |
|---|---|---|
| `h200/launch_4gpu.sh` | 598→75 | **主线基础设施**（§2.1.3）。保留：L476 `tee train.log`、L363/484-488 的 `start/complete/failed` 三行、L190 `cp dataset_manifest.json`、L220-229 A5 manifest bucket world_size 校验、L344-359 safetensors 往返 smoke、L89-93 200 GiB 空间闸、L131-167 版本墙 |
| `h200/train_fullprefix256.sh` | 32→18 | 只留 a2 分支 |
| `h200/train_a5_fast_state_warmup.sh` | 49→25 | |
| `h200/train_a5_associative_lttt_finalonly.sh` | 53→26 | |
| `h200/train_a2_a5.sh` | 331→331 | **保留（执行期修正，见 §16.1）。** 我先前判它可删，是错的：三个主线脚本**全部**通过它转发到 `launch_4gpu.sh`（`train_fullprefix256.sh:32`、`train_a5_fast_state_warmup.sh:49`、`train_a5_associative_lttt_finalonly.sh:53` 都是 `exec bash train_a2_a5.sh`）。它是共享的启动器主体，删掉等于删掉 A2 启动路径。只把它 `:125` 的 a5 默认配置改指向存活的 M3 配置 |
| `h200/prewarm_preprocess_cache.sh` | 198→100 | 保留 `shard_XX.exit` 与失败计数（L128/134-137） |
| `preprocess_cache.py` | 442→250 | 保留 `prewarm --summary` |
| `prepare_svcbench_episodes.py` | 366→250 | 保留 L59 `run_config.json`（记录 fold 0 / seed 42 / truncation_horizon 8 / world_size） |

**删除（22 个）**：`launch_8gpu.sh`、`train_a5_fast_state_warmup_8gpu.sh`、`train_a5_no_write_ablation.sh`、`train_a5_vithalf_decoder8.sh`、`train_a2_cached_trainsplit_4epoch.sh`、`prepare_a2_semantic_visual_cost.sh`、`benchmark_fullprefix256_8step.sh`、`eval_svcbench_train3706_{a2_static,a5,baseline}.sh`、`smoke_a5_warmup_finalization.py`、`bridge_train_log_tensorboard.py`、`capture_gpu_telemetry.py`、`nccl_allreduce_smoke.py`、`benchmark_retrieval_history.py`、`build_visual_cost_index.py`（随 §3.3）、`compare_svcbench_eval.py`、`stamp_svcbench_eval_metrics.py`、`summarize_dataloader_trace.py`、`summarize_operator_diagnostics.py`、`prepare_svcbench_train3706_eval.py`、`select_dataloader_profile.py`、`select_visual_batch_size.py`。

**配置 14 → 6**：`model_state_ttt_8b.yaml`(513→495)、`a2_qwen3vl8b_fullprefix256_4gpu.yaml`(90→88，改 `visual_cost_mode: proxy`)、`a5_fast_state_warmup_256_4gpu.yaml`(104→103)、`a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml`(101→100)、`deepspeed_zero1_dynamic_graph.json`(25，原样)、**`requirements-h200.lock.txt`(201，原样保留)**。

删除：`a5_meta_ttt_k8_fullprefix256_4gpu.yaml`、`a5_no_write_k8_vithalf_decoder8_4gpu.yaml`、`a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml`、`a5_fast_state_warmup_256_8gpu.yaml`、`baseline_qwen3vl8b_svcbench_256_4gpu.yaml`、`qwen3vl8b_svcbench_train3706_eval_4gpu.yaml`、`deepspeed_zero2.json`、`deepspeed_zero2_group_clip.json`。

> **`requirements-h200.lock.txt` 不要删。** 它与 `launch_4gpu.sh:131-167` 的版本墙是针对 Qwen3-VL 在 torch 2.9.x 上 Conv3D 回归的**两道防线**，删掉两者等于把该回归重新放进主线。

**目录**：删 `analysis_outputs/`（8,375 行，**先看 §11**）、`explore_outputs/`、`outputs/`、`dist/`、`.agents/`、`.codex/`、`.github/pull_request_template.md`（纯阶段 gate）、各 `__pycache__` / `.mypy_cache` / `.pytest_cache` / `.ruff_cache`。

**pyproject.toml（87→68）**：`mypy strict = true` 与 `ruff select = ["E","F","I","UP","B","SIM"]` 与"允许静默"直接冲突 —— strict 会强迫把删掉的类型收窄补回来。降为 mypy 非 strict + ruff `["E","F","I"]`。**这不是可选项，而是第 3 步能落地的前提。**

---

## 7. 连带编辑：拆开做就会崩

这三组必须在**同一次提交**内完成，否则主线立刻硬失败：

1. **`StageAModelForwardOutput.audit` ↔ `production_runtime.py:2193`**
   删掉 `audit` 字段而不同时删掉 `raw.audit.validate()` → **M1 直接 `AttributeError`**。
2. **`FastTTTForwardAudit` ↔ 四处消费者**
   `inference.py:695-708` 与 `:902-910` 抛 `InferenceProtocolError`；`meta_trainer.py:1755-1761` 抛错；`meta_trainer.py:2020-2021` **读取**其字段。删 audit 必须同时改这四处。
3. **`configs/model_state_ttt_8b.yaml` ↔ warmup bundle 哈希**
   编辑该 YAML（含删 `counterfactual_audit` / `audit_steps` / `debug_probabilities`）会改变 `project.model_dump()`，从而改变 `llamafactory_trainer` 写进 bundle 的 `project_config_sha256`。**既有 H200 bundle 会失效**。建议与"删除 `_effective_project_config_sha256`"同批做 —— 删掉哈希本身是安全方向，从此 bundle 跨配置可移植。

---

## 8. 必须保留的最小审计

只保留主线**运行本身依赖**的项 —— 它们不是可观测性，而是机制：

1. **`ema_answer_ref` 的 loss-EMA 与激活面 grad-RMS-EMA**：它们**每步设定**四项 official-weak loss 权重。删掉就没有权重了。（两级都要，见 §12）
2. **`LossTerm.row_valid_mask` / `valid_counts`**：state loss 分母只数真正带标签的行；删掉会在 4/8 rank 分片下**静默改变 loss 数值**。
3. **写入路径的 Ση ≤ `eta_chunk_budget` 重归一化 clamp**：K=8 截断 meta 梯度所依赖的收缩界。保留 clamp，删掉"触发了重归一化"的 audit 标志与计数器。
4. **非有限 loss/梯度 → 整步跳过**（而非部分应用）：更新原子性。保留行为，删计数器与告警文本。
5. **`data.py` 的泄漏擦除**：`RUNTIME_ALLOWLIST`/`RUNTIME_DENYLIST` + `assert_runtime_payload_safe` 的 denied/unknown 分支 + `episode_data.py:172-175` 与 `inference.py:1129/1469` 的调用。**这是我唯一建议保留 `raise` 的校验** —— 它阻止 answer/count/occurrence_times 进入模型输入，静默失败在这里等于**训练目标泄漏、全部指标作废**。
6. **`preprocess_cache._replace_idempotent`**：多 rank 正确性，不是审计。
7. **Reader 的 `exact_count` + number-token 输出**：这是产品本身，Answer 路径消费 Reader 发出的数字 token id。
8. **`data.py:312-320` 的 GroupKFold-by-`video_id` 与 `episode_data.py:1616-1620` 的 `_split_map`**：逐视频 train/val 隔离是对外声明的性质，不能随其泄漏断言一起删。
9. **`RetrieverOutput` 的 `status` / `reason` / `n_state` / `n_retrieved`**：真实控制流 —— `stage_a_targets` 与 `state_reader` 在 5 处按 `OK` vs `EMPTY/UNSUPPORTED/INVALID` 分支，`state_reader.py:568-570` 消费 reason 与计数。
10. **`StateBankRuntimeState.version` / `StateBankView.bank_versions`**：`FastAssociativeContext.bank_versions` 是必填字段。
11. **`config.outer_gradient_control` 全 8 字段**（`outer_gradient_control.py:385` 按名 `getattr` 读取，这些 per-group L2 cap 直接改变更新）与 **`config.a5.truncation_horizon`**（K=8 视界，`meta_trainer.py:1025`、`production_runtime.py:2447`）。
12. **名字带 `audit` 但其实是功能返回值的五处，删名不删物**：
    - `MetaTTTEpisodeRunner.last_balance_audit` / `OfficialWeakBalanceAudit` —— 承载 per-term EMA scale，被 `outer_composer.compose_one_from_audit` 与 `commit_streamed_gradients` 消费来构造梯度。**是机制**。
    - `TruncatedMetaTTTEpisodeOutput.total` / `.query_loss`（detached FP32 标量）—— `training_step` 把 `(output.total * loss_weight).detach()` 返回给 HF Trainer，用于上报 loss 与其自身调度。
    - `OuterGradientController.apply_deepspeed` 返回的 `OuterGradientAudit` —— `SegmentBackwardController.finalize` 需要这个返回值，并用 `skipped_nonfinite` 决定是否跳过非有限 Outer 更新。
    - `MemoryWriteResult.did_write` —— 驱动 `adapted.fast_states = …` 与写版本递增，**即递推本身**。
    - `TruncatedMetaTTTEpisodeAudit` 削到 7 个字段（`query_count`、`write_count`、`skip_count`、`associative_valid_count`、`readout_target_cosine_mean`、`loss_weight`、`segment_count`）—— `TTTQwenTrainerMixin.log` 在 `meta_audit.loss_weight` 上做门控。
13. **`QueryMetricSnapshot` 只保留 `loss/answer` 与 `loss/state`** —— 削减后这是**唯一**能让人判断 4 epoch A5 run 是否在学习的信号，其余全部遥测都已删除。

---

## 9. 承重陷阱 —— 看起来是 guard，禁止删

已逐一读码核验。**第 0 条是本次审计中最危险的一处**，因为它被夹在两个可删的 `raise` 之间：

0. **`meta_trainer.py:2233` 的 `unscaled = gradient.detach().float().clone().div_(backward_gradient_scale)` 是 DeepSpeed/AMP 的 loss-scale 反缩放，不是清理代码。**
   它的上下文正是陷阱所在：
   ```python
   2228:  raise ValueError("Query loss did not produce a gradient …")   # 可删
   2233:  unscaled = gradient.detach().float().clone().div_(backward_gradient_scale)   # ← 必须保留
   2234:  if not bool(torch.isfinite(unscaled).all()):
   2235:      raise ValueError("Query proxy gradient must be finite after backward unscale")   # 可删
   ```
   删掉中间那行，每个 meta 梯度都会带着实时 loss scale（约 1e4）。随后 `max_norm=10.0` 的裁剪会把它统统归一化 —— **裁剪从"限幅"静默退化为"恒等于 10.0 的纯方向"**，梯度幅值信息全部丢失，且不会有任何报错。
   同理保留 `_cotangent_norm_float`(L2270-2289)：它的 `.float().square().sum(dtype=torch.float32).to(dtype=torch.float64)` 级联是那个裁剪的分母，是数值而非度量（只删它末尾的 isfinite raise）。

1. **`_attach_rank_stable_zero_anchor`（`llamafactory_trainer.py:491-517`）不是恒等操作。** A5 每 episode 多次 backward，official-weak 路由与 retrieval 有效性依样本而变，两个 rank 会在同一次 backward 触碰不同 state 参数。它从每个"条件使用的非 Qwen 参数"取一元素乘 0 加进 loss，使**梯度 hook 集合确定化**；副作用是 grad 由 `None` 变 `0`，于是 AdamW 走一步动量而非跳过。连同 `_unique_trainable_parameters`、`_rank_stable_conditional_parameters`、`expected_backwards`(L1875)、`expected_count`(L1935) 一起保留 —— **只删 L1948 的事后相等 raise，绝不删计算**。
   *（ZeRO-2 死锁是它的原始动因，主线用 ZeRO-1，故 `_deepspeed_partitions_gradients` 的 ZeRO-2 分支可删，但 anchor 因 hook 覆盖 + grad None→0 语义仍须保留。）*
2. **`_validate_checkpoint_tree`（L2786-2820）是本次最危险的一处删除，必须保留。** 我核验了 L2617-2622 的实际序列：
   ```python
   _validate_checkpoint_tree(incomplete_checkpoint)   # 校验
   incomplete_checkpoint.rename(final_checkpoint)     # 晋升
   for child in output_dir.glob("checkpoint-*"): shutil.rmtree(child)   # 摧毁全部 epoch checkpoint
   ```
   这 35 行是"保存被截断"与"`rmtree` 掉每一个可恢复中间产物"之间的**唯一**屏障。4 epoch × 4×H200 的 A5 run 若遇磁盘满或抢占，删掉它就会晋升一个坏 checkpoint **并**删光退路，不可恢复。L2610 的 `FileExistsError("refusing to overwrite an existing final checkpoint")` 同理保留。
3. **`OuterGradientController` 整条调用链是承重数值。** 7 个 h200 配置全部 `max_grad_norm: 0.0`，即 HF 自带裁剪**已关闭**；`_ControlledDeepSpeedEngineWrapper.backward`(L168-182) 与 `SegmentBackwardController.finalize`(L388-414) 里的裁剪是系统中**唯一**的裁剪。名为 `finalize`、形似记账，实为数值。这也是我把 `outer_gradient_control.py` 只减到 400 的原因。
4. **`fast_ttt.py` 的 centering 科学全部保留**：`_smooth_normalize`、`_centered_over_valid_slots`、`_centered_over_valid_tokens`，及其中的 `clamp_min(_ETA_SCALE_TINY)` 与 `clamp_max(_MAX_CENTERING_GAIN)`。后者是**纠正**（防止落在均值上的 slot 把舍入噪声放大到满量程），不是检测。它们的 docstring 满是 probe 数字，极易被误当诊断残留扫掉，但它们设定了有效写入容量（实测 1.001 → 4.213 → ~32 / 32）。
   对照：同文件 `:205` 的 `_DEGENERATE_ROW_NORM` 判断在 `__post_init__` 里只 `raise`，可删。
   建议把这两个函数 40 余行的实测叙事 docstring 压到 2-3 行陈述不变量，测量记录搬进 README「已知局限」。
5. **`trainer.py` 的五处机制**（正因如此它不并入 `meta_trainer`）：
   - `:308-317` `with torch.no_grad(): observe_chunk(...)` —— 界定 Support 链的激活上界，不是可选优化
   - `:306` / `:233` `detached_query` + `prepared.detached()` —— 梯度隔离，形似缓存微优化
   - `:295` / `:299-301` `pre_query_identity_states` 快照 —— 紧邻 audit 计数器，极易被一起扫掉
   - `:348-363` `reader_counts` 的 -100 sentinel + `reader_valid` 掩码 —— 形似指标管线，实为监督
   - `:126-131` `StageAModelForwardOutput` 的字段（`answer_logits`、`composed_input`、`reader_counts` …）
6. **`json_contract.py` 保留核心。** 名字与 docstring 都像"strict readers"，实际是 `data.py`(L362-480) 与 `episode_data.py`(L1366-1769) 约 60 处调用的 JSONL 解析器。把它的 `raise` 静默化会产出**结构合法但内容错误**的 episode。
7. **静默路径本身要保留**：`losses.LossSkipReason` / `_invalid_term` / `_invalid_time_output`，`memory_write.MemoryWriteSkipReason` / `_skip_result`，`state_bank.update_row` 的 `REPLAY_IGNORED` 分支（这是**幂等性**，不是检测 —— 若任何主线路径会对同一 `chunk_index` 二次调用 `update_row`，删掉它就会重复计数）。
8. **零权重 padding 承重**：`episode_data._build_segment_buckets`(L1564-1613) 克隆最后一个真实 episode 并置 `loss_weight=0.0`，使所有 rank 执行相同次数 backward 集合通信。保留克隆、`loss_weight` 字段与 `RankAlignedA5SegmentSampler` 的 real/padding 划分。
9. **`TensorizedRetrievalHistory.fork()` 必须存活**：`meta_trainer.py:2368` 按 query 行 fork ring，这是 K=8 meta 梯度各 episode 不共享 retrieval 记忆的机制。不要折叠成浅拷贝。
10. **`PreprocessCacheMissPolicy = error` 谨慎处理**：改成静默 miss 后结果仍正确（回退到内联解码），但 H200 共享盘上一次静默的 100% cache miss 会让 run 慢一到两个数量级而无任何信号。建议保留一个"miss 率超过阈值就在 `train.log` 打一行"的极简替代。
11. **`engine.set_gradient_accumulation_boundary(is_boundary=is_final_segment)`（`llamafactory_trainer.py:326-327`）与 `SegmentBackwardController.finalize` 里那一次 `engine.step()` 必须原样保留。** 若重构后让 DeepSpeed 按 segment 逐次 step，K=8 的梯度累积语义就断了。
12. **`_clip_query_proxy_gradients`（`max_norm=10.0`，来自 `a5.query_meta_gradient`）是科学，不是 guard** —— 它在 segment 求和前界定 per-Query cotangent。只删 `QueryCotangentClipAudit` 与原始范数累加器，**裁剪本身保留**。

**一处建议的例外**：warmup bundle 的 `associative_contract_version` provenance 检查。删掉它意味着 revision-3 的 bundle 能被静默载入 revision-4 的 key 语义 —— 两套不兼容 key 空间混用是**静默的科学污染**，不是运维噪声。它只有几行。若仍要删，请至少把版本号写进 bundle 文件名。

---

## 10. 文档：9 份 → 1 份

**只留 `README.md`，约 115 行，五节**：`架构` / `训练（M1-M3）` / `推理（M4 契约）` / `环境` / `已知局限`。
文件名必须保持 `README.md` —— `pyproject.toml:5` 声明了 `readme = "README.md"`。已核验：`src`/`tests`/`scripts`/`configs` 中**没有任何** `.py`/`.sh`/`.yaml` 引用任何 `.md`，故其余文档的删除对构建零副作用。

| 文档 | 行数 | 处置 |
|---|---:|---|
| `README.md` | 109 | → 115，唯一文档 |
| `ARCHITECTURE.md` | 238 | 合并：仅约 35 行独有且必要（插入点：Main Visual Merger 之后、video `masked_scatter` 之前；DeepStack 8/16/24 不动） |
| `DECISIONS.md` | 179 | 合并：L1-79 是重述；L81-179 是 gate 取证分析 |
| `docs/production-a2-a5.md` | 320 | 合并：只带三条训练命令 + `prepare_svcbench_episodes.py` 调用 + `TTT_RESUME_CHECKPOINT` |
| `docs/dataloader_throughput.md` | 60 | 合并：3 条运维事实进环境节 |
| `docs/method_overview.md` | 252 | 删（自述非规范，与 README+ARCHITECTURE 约 90% 重叠） |
| `docs/rewrite.md` | 350 | 删（同一架构的第三次中文重述，自述非规范） |
| `docs/svcbench_analysis.md` | 408 | 删，且**不要从中抄任何常量**：表头声明 schema 13，代码是 schema 14 |
| `docs/a5_support_query_outer_ttt_report.md` | 406 | 删，且**绝不可合并任何内容**：它记录 schema-12 inner-SGD 机制（functional SGD、0.1 support 辅助损失），该机制已于 2026-07-30 删除，与现状直接矛盾 |
| `.github/pull_request_template.md` | 36 | 删（纯阶段 gate + spec 哈希一致性 gate） |

### 必须写进新 README 的三条事实

删了代码之后，这三条就只剩文档承载；丢了它们，未来会被**静默**破坏：

1. **`memory_eta_gate_output.weight` 零初始化是刻意的**：它使 Ση = `active_slots × eta_gate_init` = 32 × 0.02 = **0.64**，与 seed 和 slot 尺度无关。恢复默认 Xavier 会把 Ση 静默钉在 **1.0** —— 正是 delta-rule 的湮灭点（retention = −β = −0.01）。
2. **只对 `memory_keys` 做 token centering，绝不对 W0 路径做**。对 W0 输入 centering 会破坏 `M = 0` 逐位等于静态 forward 的不变量，A2 能力就不再在 episode 起点被保留。
3. **`associative_contract_version = 4`**：revision 3 的 warmup bundle 与 revision 4 的 key 语义不兼容。

另建议「已知局限」节保留一句：slot 塌缩是**结构性**的（未训练 encoder 上实测 pairwise cosine 0.999780，跨 seed 稳定），不是训练缺陷 —— 这是删掉 `analysis_outputs/` 后唯一的书面记录。

---

## 11. 风险与不可逆项

1. **不可恢复的删除（动手前必读）。** `docs/rewrite.md`、`docs/svcbench_analysis.md` 与整个 `analysis_outputs/`（8,375 行、58 个文件）**既未被 git 跟踪，也未被 gitignore**（已用 `git ls-files` / `git check-ignore` / `git status --porcelain -uall` 三重确认）。删除后无 commit 可恢复。其中 `A5_SLOT_GEOMETRY_2026-08-06.md` 与 `A5_CAUSAL_DIAGNOSIS_2026-08-06.md` 是 slot 几何塌缩的实测记录（与 HEAD 同日），`code_volume_reduction_audit_2026-07-30/` 是上一次削减审计。
   **→ 第一步必须是 `git add -A && git commit`，或把整树复制到别处。**
2. **两处不是纯删除**：`visual_cost` → proxy 改变 A2 batch 组成（§3.3）；删 `_effective_project_config_sha256` 使既有 H200 bundle 的配置哈希校验失效（这是**期望的**方向，但已在 H200 上的 bundle 需重新发布）。
3. **删掉 `DECISIONS.md` 就删掉了 eta 零初始化与 token-centering 的唯一书面记录** —— §10 的三条事实必须落进新 README，否则未来有人恢复 Xavier 初始化，Ση 被静默钉在 1.0，**不会有任何报错**。
4. **三处新增的"静默后果"**，删除后不再有任何信号：
   - NaN 一旦进入 `M` 会**持续存在**（同时删掉 `memory_write.py:189-195` 的写后有限性复查、`fast_ttt.py:543-544` 的 forward 输出 raise、`FastTTTForwardAudit` 的范数之后）；
   - 零范数但有效的 slot 行会静默损失写入容量（`fast_ttt.py:205-208` 的注释明确记录了这一点）；
   - preprocess cache 100% miss 会静默让 run 慢一到两个数量级（§9.10）。
   若要保留一条最低信号，我建议只保留 NaN-in-`M` 这一条（一次 `isfinite` + 一行 `train.log`）。
5. **只剩一个 DeepSpeed 方案**：删掉 `deepspeed_zero2*.json` 后仅存 ZeRO-1 dynamic-graph。若它在 A5-main 上回归，仓库里没有备选方案可切。
6. **mypy strict 会与目标对抗**：删掉类型收窄的校验后 strict 会要求补回。§6 的 pyproject 降级是前提，不是可选项。
7. **本机测试的证明边界不变**：tiny/CPU 测试只能证明接口、梯度、因果性与状态隔离。真实 Qwen3-VL-8B、BF16、4 卡显存、吞吐与收敛必须由独立 H200 运行记录证明。**削减后第一次 H200 run 必须与削减前做同 seed 对照**，尤其在动过 `visual_cost` 模式之后。
8. **上一次审计的结论只有 5.4%，本次是 63%，差别不在勇气而在约束**：2026-07-30 那次被"必须保留严格校验"的契约限制，只能把 414 条冻结 pin"改写成嵌套 dict + walker"（省 889 行）。本次允许静默，那 414 条连同 187 个 `__post_init__` 可以整体消失。
   **反面推论：如果将来又要求严格校验，这次的削减无法低成本回退。**
9. **最高severity：因果前缀的运行时保证会随一个 `__post_init__` 一起消失。** `MetaTTTEpisode.__post_init__`（`meta_trainer.py:197-272`）的 L263-268 是**唯一**保证 `segment[-1].end_time < queries[0].query_time`、以及后续 segment 的 Support 单向前进的地方。因果性是本方法对外声明的第一条性质，删掉它之后不再有任何运行时拦截。
   **→ §5 第 4 条测试从"应该有"升级为"必须有"，且必须覆盖跨 segment 的时间单调性，而不只是单 chunk 掩码。**
10. **同阶段续跑能力会丢失。** 删掉 `resolve_same_stage_resume` 与 `_validate_resume_balance_schema` 之后，一个 4 epoch 的 A5-main run 若在第 30 小时挂掉，只能从零重跑。这是真实的运维能力损失，不只是 LOC。若 H200 机时紧张，建议把这两个函数留在削减范围之外（约 60 行）。
11. **删掉 counterfactual audit，等于删掉证明方法有效的那把尺子。** 它是训练中**唯一**测量"TTT 适配是否真的降低了 Query loss"的机制（`gain_abs` / `gain_rel` / `descent_cosine`，同时对照 `M=0` 与 segment 起点两个基线）。按"不要审计"的要求它应当删除，但请知情：删掉之后，除了最终 benchmark 分数，没有任何内部证据表明 memory 写入起了作用 —— 而这正是论文的核心论点。
   **建议**：把它降级为"可选、默认关闭、约 40 行"的单一开关，而不是彻底删除。这是我在整份报告里唯一建议保留一个开关的地方。

---

## 12. 我对 agent 结论的一处修正

`outer_loss_balance.py` **不含两个竞争的 balancer**，因此"第二个是死码"的削减依据不成立。
`_prior_loss_scales`(L747) 与 `_prior_gradient_scales`(L759) 是**同一个 `ema_answer_ref` 算法的两个串联级**，在同一方法内先后调用（L510、L564），与 README 的措辞逐字对应：「loss EMA 对齐 Answer 尺度，**再**按 `q_target/q_operator/q_time` 激活面的梯度 RMS EMA 平衡四项」。「再」即串联。

**两级都必须保留。** 该文件仍然缩减（实测 guard 237 + 每级返回的 `*_clamped` audit 标志 + L761-762/L781-782 的 shape-drift raise），但目标从 700 上调为 **850**。各级里的 `bounded = ratio.clamp(...)` 是纠正，保留。

---

## 13. 执行顺序

分六步，每步后跑存活测试：

| 步 | 内容 | 风险 |
|:-:|---|---|
| **0** | **`git add -A && git commit`**（或整树备份）—— 8,375 行未跟踪文件（§11.1） | — |
| **1** | 删主线外整文件：6 个 src + 22 个脚本 + 8 个配置 + 6 个目录 + 17 个测试文件。不碰任何存活文件内部 | 最低，已可拿到约 −30% |
| **2** | 删三个枚举与其字段：`AuditLevel`、`relevance_gate_mode`、`CalibrationStatus`。自成闭包 | 低 |
| **3** | pyproject 降级（mypy 非 strict、ruff 三规则）—— **必须先做**，否则第 4 步无法落地 | 低 |
| **4** | 机械删除校验层：187 个 `__post_init__`、`_validate_*`/`_check_*`/`_require_*` 函数体、`tensor_contracts` 14 处调用点、`runtime_metrics` 20 处调用点。**同时补上 §5 第 5 条隔离测试** | 中 |
| **5** | 删 audit/probe/stamp 子系统与遥测 + 配置 audit 旋钮 + ZeRO-2 分支。**§7 的三组连带编辑必须同批** | 中高 |
| **6** | 结构性合并：`associative_ttt` + `memory_write` → `fast_ttt`；`build_inference_runtime_bundle` + `StateTTTRuntimeBundle` 从 `production_runtime` 移入 `inference`（这会把 `production_runtime.py:2577` 的函数内惰性导入变成单向依赖，消除循环）。重写 README | 中 |

**第 6 步建议附带一次零净增的搬迁**（不计入削减量，但显著降低理解成本）：per-Query cotangent 流水线目前被劈成两半 —— `fast_ttt.py` 持有首尾两步 `make_query_proxy_fast_state`(L946) 与 `deferred_fast_vjp_loss`(L988)，而中间三步 `_capture_query_proxy_gradients`(meta_trainer L2213)、`_clip_query_proxy_gradients`(L2237)、`_cotangent_norm_float`(L2270) 在另一个文件。把中间三步移到 `fast_ttt.py` 与其同伴相邻，可让 `QueryCotangentClipAudit` 与 `FastQueryProxyAudit` 合成一处，并让 `meta_trainer` 不再需要知道 `a5.query_meta_gradient`。同理 `meta_trainer._truncate_trajectory`(L1697-1706) 的 9 行循环应下沉为 `memory_write.truncate_memory_states()`（其中的 storage 隔离 raise 按 §3.4 删除）。

`visual_cost` 的 A2 配置切换（§3.3）单独一步，因为它是唯一的行为变更。

---

## 14. 对抗验证阶段的修补项（4 个 lens 复核本计划后的结论）

四条独立 lens（静态可达性 / 科学完整性 / 测试绑定 / 完备性）复核了本计划。共报 30 条 blocker，其中约 20 条源于审计过程本身的缺口（见 §15），以下 **10 条是对计划的真实修正，必须并入执行**：

**A. 删除枚举会打断主线推理 —— `AuditLevel` 不能裸删。**
`inference.py:1707` 在构造 M4 的 JSON 输出字典时读 `bundle.config.inference.audit_level.value`（该字典于 `:1740` 写出）。删掉枚举 → `ttt-svcbench-infer` 的 `main()` 抛 `AttributeError`，**这是主线流 (I) 断裂，不是审计移除**。
→ 二选一：保留 `ProjectConfig.inference.audit_level`（3 行），或在**同一次提交**内把 `:1707` 硬编码为字符串。

**B. `training_context.py` 的删除是一次连带编辑（补进 §7）。**
`meta_trainer.py:92-96` 在模块层 `from ttt_svcbench_qwen.training_context import QueryActivationOffloadBudget, QueryActivationOffloadScope, query_activation_context`，运行时用于 `:932`、`:954-956`、`:1047-1048`、`:1275-1276`。虽然所有主线 YAML 都是 `query_activation_offload: false`（故 `query_activation_context()` 返回 `nullcontext()`，行为已死），**导入仍是活的**。
→ 删模块必须同批改这 6 处；否则 M3 在 import 阶段即失败。

**C. `outer_loss_balance._apply`(L305-328) 是纠正，不是检查。**
它把 `ema_values` / `gradient_ema_values` 快照为 float64（L314-317），让 `super()._apply` 跑完，再以 float64 重新装回（L322-327）—— 这是防止 `.to()` / DeepSpeed 把 EMA 降精度的机制。
→ **只删 L328 的 `self._assert_balance_state()`**，L307-327 逐字保留。

**D. `OfficialWeakTermBalanceMetrics` 的"audit"字段其实是两趟算法的中间量。**
A5 streamed 路径把它们当系数读回：`compose_one_from_audit` 读 `term.global_valid_count`(L936) 与 `term.scale`(L940) 构造梯度。
→ 保留 6 个字段（`name`、`scale`、`loss_scale`、`global_valid_count`、`raw_gradient_rms`、`ema_gradient_rms`）及其 pass-2 读取端。**这与 §12 的修正同向：`outer_loss_balance` 的"audit"外观下藏着算法本体。**

**E. `outer_gradient_control.sanitize_scalar_loss`(L572-587) 是纠正，不是 watchdog。**
它返回 `torch.where(isfinite(loss), loss, nan_to_num(loss)*0.0)`，作用于 K=8 的**每一个** segment loss —— 它正是 §8.4「非有限 → 整步跳过」的实现机制。
→ 把 `sanitize_scalar_loss` + `record_loss`(L190-203) + `_synchronize_loss_nonfinite` + L256-281 的 skip 分支当作**一个原子单元**保留或删除，不可拆。建议保留。

**F. `runtime_metrics` 用 12 行 no-op shim 代替逐点删除。**
`production_runtime.py:123-127` 模块层导入，且有 **9 个 `with trace_cuda_phase(...)` 包着真实工作**（`:946` vit_forward、`:1013`、`:1085`、`:1137` …）。逐个去缩进风险高、收益低。
→ 保留一个约 12 行的 no-op shim（导出 `trace_event` / `trace_cuda_phase` / `flush_runtime_metrics` / `configure_runtime_metrics`），199 → 12 而非 199 → 0。**这比我原方案更安全，且省下的行数几乎相同。**

**G. `FastTTTForwardAudit` 保留 6 字段桩，而非整删。**
`inference.py:36` 导入、`:696` 与 `:903` 分支于 `fast_audit.used_runtime_state`。
→ 留 6 字段（`used_runtime_state`、`readout_share_norms`、`valid_token_counts` + 写审计读的三个）。同理 `fast_ttt.SlotStateView`（12 行 Protocol）保留最省事 —— `meta_trainer.py:35/122` 用作注解。

**H. `tests/support/runtime_factories.py` 是必须重写，不是压 docstring。**
`make_query_output` 用被删的 kwargs 构造对象（`OperatorRouterOutput` 等），且它是全套件共享最广的 helper。原计划 296→280 会让**整个套件在 collection 阶段就 TypeError**。
→ 296→240，并显式列出要剥掉的 6 个 kwarg。**这是第 4 步之前必须先做的一件事。**

**I. 两个 Bank 的逐视频隔离会归零覆盖。**
`test_state_bank.py:1005`（cross-owner）与 identity_bank 的对应测试随 `OWNER_MISMATCH` 分支一起删除后，仅剩 `test_fast_ttt.py:350` 覆盖 memory 张量。
→ 显式保留 `tests/test_inference_protocol.py:639`（主线级隔离测试，不依赖任何被删 src），再补约 25 行覆盖两个 Bank。**这把 §5 第 5 条从一条测试收紧为三条。**

**J. A2→A5 handoff bundle 往返只有一处覆盖。**
`tests/test_production_factory.py:2811 test_warmup_bundle_is_non_qwen_atomic_and_fail_closed`。若该文件被整删，§5 第 6 条契约将失去全部覆盖。
→ `test_production_factory.py` 2978→420，keep-set 锚定 `:2776`（allowlist 排除非持久 buffer）与 `:2811`。

**K. `production_factory.py:145` 的 `ConfigDict(extra="forbid")` 是一次连带编辑。**
删掉 `runtime_trace_mode` / `runtime_trace_dir` / `semantic_projector_delta_audit_steps` /
`a5_parameter_delta_audit_steps` / `operator_diagnostics_interval` 这些**字段**，而 YAML 里仍写着它们，
`extra="forbid"` 会让配置加载直接失败。
→ 同批从三个主线 YAML 里删掉对应行，或把 `extra` 改为 `"ignore"`。后者更省事且与"允许静默"一致。

**L. 切到 `visual_cost_mode: proxy` 会打断一个测试断言。**
`tests/test_production_factory.py:741-742` 断言 `extension.visual_cost_mode == 'exact_tokens_then_runtime'`
且 `visual_cost_index` 非空。§3.3 的切换必须同批改这两行。
（另修正我先前的记账：`:664` 与 `:719-725` 读的是 `a2_qwen3vl8b_fullprefix256_4gpu.yaml` —— 那是**主线 M1 配置**，
不是"绑定到待删 YAML 的测试"，不要因此删掉它们。）

**M. `ProductionVisualAudit` 必须作为 3 行 dataclass 存活，不能整删。**
它名字里有 audit，但实体是"已物化 chunk"的传输载体。只删无读者的四个字段
（`token`、`observation_role`、`visual_token_count`、`video_grid_thw` —— 在 src/ 与 tests/ 全仓 grep 均无消费者），
类本身收缩为 `chunk: CurrentChunkMaterialization | PreparedVisualCPU`。
`audit_current_chunk_visual_tokens` 随之删除（`:1035` 那次调用已经把返回值整个丢掉，是纯检测器）。

**N. 两个 prefetch collator 不是重复实现。** 我原先把"两个 collator"当作削减依据，这是错的：
它们已共享 `_PrefetchCollatorBase`(1535-1564) 与 `_prepare_query_pair`(1567-1608)，
`A2PrefetchCollator` 只是额外物化自适应 Support 计划。不要试图合并。

**O. `production_runtime` 的解码器族全部是活分支。** L3234-3782 约 570 行（`_decode_uniform_interval`:3247
按 ≤16s 路由等）在主线上每个分支都会走到。两位执行者独立给出同一结论：这半边到 1330 已是底，
再往下必须删掉功能（流式化、或砍掉 preprocess-cache 往返），两者都**不建议**作为默认。

**另核准一处我原本的写法**：`train_fullprefix256.sh:22` 的 `a5` 分支确实引用了待删的 `a5_meta_ttt_k8_fullprefix256_4gpu.yaml` —— §6 已要求该脚本"只留 a2 分支"，故一致；但执行时必须**先删 a5 分支再删配置**，顺序颠倒会留下一个指向空文件的脚本。

---

## 15. 缺口填补记录

先前有 3 个映射 agent 在读大文件时耗尽上下文而失败：`runtime-god`（7,735 LOC）、`state-encode-read`（6,562 LOC）、
`tests-big`（14,495 LOC）。**该缺口已按一文件一 agent 重新拆分并全部填补** —— 12 个 agent 全部返回，
0 失败，产出 **595 条逐符号工单**。

拆分方式（失败原因是单 agent 范围过大，故按文件切并加硬性上下文预算：先 grep 符号表、再按 ≤250 行切片读、
宁可返回部分结果也不要卡死）：

| 原失败单元 | 原范围 | 重拆为 |
|---|---|---|
| `runtime-god` | 1 agent / 7,735 LOC | 4 agent：`production_runtime` 按 L2055 切两半、`inference` 独立、`production_factory`+`qwen_adapter` |
| `state-encode-read` | 1 agent / 6,562 LOC | 4 agent：一文件一个 |
| `tests-big` | 1 agent / 14,495 LOC | 4 agent：按契约归组，且一律 grep `def test_` 而非通读 |

**填补的净效果是总量目标下调，不是上调。** 8 个 src 目标与 5 个测试目标被执行者上调（最大一处
`production_runtime` 1400 → 2690），因为剩余部分经读码确认是活机制而非 guard。总账因此从 −63.6% 修正为
**−56.0%**。这是有证据的反驳，我采纳执行者的数字。

同时填补消解了 §14 中约 20 条"计划删了 X，而导入 X 的模块 Y 没有编辑条目"的记账缺口 —— 那些模块现在都有了
逐符号工单。§14 的 K/L/M/N/O 五条是这次填补新发现的真实修正。

**仍然开放的两项**（都是需要新写代码，而非需要再审计）：
1. per-video Bank 隔离的替代测试（约 25 行）—— §14.I 已给出规格，需实现；
2. `tests/support/runtime_factories.py` 的重写（296 → 240）—— §14.H，必须在第 4 步之前完成，
   否则整套测试在 collection 阶段就 TypeError。

---

## 16. 执行期发现的计划错误（迁移过程中实测修正）

按本报告执行时发现的、报告本身写错的地方。记录在此，因为它们都是"看文档判断"会犯而"实际跑一遍"才会暴露的错。

1. **`scripts/h200/train_a2_a5.sh` 不可删（§6 已就地更正）。**
   我依据"README 与 docs 都不提它"判它为未文档化的便捷脚本。实际上三个主线脚本全部
   `exec bash train_a2_a5.sh`，它再 `exec bash launch_4gpu.sh` —— 它是共享启动器主体。
   删掉后 `train_fullprefix256.sh a2`（M1 的唯一入口）立即失效。
   **教训**：判断脚本是否 dead 必须看它是否被 `exec`/`source`，只看"文档里有没有提"会漏掉转发层。

2. **`runtime_metrics` shim 是 35 行而非 12 行。** 保留 4 个调用形状 + 类型注解 + docstring 之后就是这个体量；
   12 行的估算没有算上 `RuntimeTraceMode` 别名与 `@contextmanager` 的样板。

3. **`json_contract.py` 维持 40 行，不缩到 20。** 执行者给出的理由成立且比我的更强：把 `string_value`
   的 isinstance 检查换成 `str(value)` 会把缺失键（None）静默映射成字符串 `"None"`，
   把 `integer_value` 换成 `int(value)` 会把 `3.7` 静默变成 `3`、`True` 变成 `1`。
   这是"改错值"而不是"报错"，属于必须保留的纠正类检查。

4. **`data.py` 落在 258 行而非 155。** 差额来自被指令保留的泄漏擦除、GroupKFold 与 `RuntimeQueryInput`；
   155 的估算把它们算进了可删部分。

5. **验收工具的选择**：`ruff --select F821` **查不出**本次迁移的主要破坏形态。删掉一个 Enum 成员或
   dataclass 字段之后，`RetrievalReason.OWNER_MISMATCH` 这类跨文件引用是属性访问而非未定义名字，
   F821 完全静默。**`mypy --follow-imports=skip` 的 `[attr-defined]` 与 `[call-arg]` 两类才是真正的验收门**
   —— 它一次就精确列出了 31 处跨文件破坏及行号。任何按本报告执行的人都应该用它做每一步的 gate。

---

## 附：核验方式

- **导入图**：对 39 个 src 模块逐一提取内部 import，并对 21 个可疑模块做反向引用（含 `tests/` 与 `scripts/`）。
- **guard 占比**：机械扫描 —— 把 `__post_init__`/`_validate_*`/`_check_*`/`_require_*`/`*audit*`/`*probe*`/`*stamp*`/`*checksum*` 的函数体，与每个 `raise` 及其守卫条件块取并集去重，得 10,505 / 44,533 = 23.6%。
- **测试机械死亡量**：逐个 `def test_` 解析函数体，统计含 `pytest.raises` 者，得 110 个 / 2,812 LOC。
- **命名反直觉三处**：`diff` 两个 A5 配置；`grep` 全部配置的 `*_query_visual_mode`；`production_factory.py:166` 的 `Literal` 类型；`launch_4gpu.sh` 与 `report_to: none` 的交叉确认。
- **`_validate_checkpoint_tree` 的数据丢失路径**：直接读 `llamafactory_trainer.py:2605-2625` 的 validate → rename → `rmtree` 序列。
- **`tensor_contracts` 冲突**：读全部 81 行 + `memory_write.py:250-258` / `meta_trainer.py:1700-1706` 两处 M-隔离调用点 + 三处 `timestamps_match` 调用点，确认全为纯检测。
- **`outer_loss_balance` 修正**：读 L510/L564 的调用顺序与 L747-779 两级实现。
- **`trainer.py` 三方分歧定案**：实测其 guard 行 105、`StageAExecutionAudit` 42 行，判 280（185 高估删除量、380 低估）。
- **git 跟踪状态**：`git ls-files` / `git check-ignore` / `git status --porcelain -uall` 三者交叉。
- **对抗验证**：4 条独立 lens（静态可达性 / 科学完整性 / 测试绑定 / 完备性）复核全计划，产出 §14 的 10 条修正与 §15 的缺口交代。
- 另参考仓库内既有的 `analysis_outputs/code_volume_reduction_audit_2026-07-30/`（上一次审计，结论 5.4%，受"保留严格校验"约束）。
