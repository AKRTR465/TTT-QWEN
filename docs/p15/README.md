# P15 Stage A 显式状态 Warm-up

GATE_STATUS: `passed`

P16_ALLOWED: `true`

P16_STARTED: `false`

ASSET_POLICY: `Synthetic/tiny cases only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `A2 typed supervision, hard/soft state rollout, Retriever/Reader/Composer/Qwen prefill, State+Answer Outer step, metrics, compact checkpoint, and fail-closed P16 exit gate`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `319fd80bbca28c4c771a815ef033f12368879b6a`（P14 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| baseline ARCHITECTURE SHA256 | `d397cab2943fac9a032b718800d37c69c60e1b82152d395a8aa869cc3f454597` |
| final ARCHITECTURE SHA256 | `31f3d98eea2d9d6b4dbeca687d025c717de22009319df883e37a511eb7d40352` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 运行变体 | A2；`static_w0_no_inner_sgd`；`frozen_synthetic_engineering_gate` |
| 数据与模型资产 | 未下载视频、数据集或 8B 权重；只使用进程内 synthetic/tiny case |

## 已验证实现

| 范围 | 已验证合同 |
| :--- | :--- |
| Stage A 变体 | A1 只计算 Answer Loss；A2 精确计算 State+Answer Loss。两者都禁止 TTT loss、Predictor、functional SGD 和任何 Inner SGD counter |
| Typed target | 三值 `official_explicit`/`synthetic_explicit`/`missing` provenance；O1 pre-matched 六字段、O2/E1/E2、operator/time/retrieval 逐 row 对齐；missing 不变成零标签，不从最终 count 反推 dense assignment |
| Answer 监督 | teacher-forced labels/number mask 按 Composer provenance 映射；Reader number 只是 context，不得变成 answer supervision；A1 plain-Qwen 路径支持空 Reader |
| Hard/soft 状态 | `StageABankWriter` 调用真实 P9/P10 O1/O2/E1/E2 Bank/Identity/FSM API；hard records detach/clone，Semantic Projector soft branch 保留梯度 |
| Episode | owner 必须从 reset runtime 开始，因果运行 observe chunks，再执行 Retriever→Reader→Resampler→Composer→一次 teacher-forced Qwen prefill；Stage A 不运行 decode |
| Outer optimizer | Qwen 全冻结且 allowlist 为空；static `W0` 与显式状态模块按 allowlist 进入 AdamW `3e-4`；FP32 global clip=1，非有限/无梯度整步 skip |
| 采样与指标 | O1/O2/E1/E2 按 seed 确定性过采样；O1 soft/hard、O2/E1/E2 duplicate/miss、9-class operator、time、retrieval、Reader exact count 与 Reader/LLM disagreement 分开报告；零分母为 `null` |
| Checkpoint | 原子保存 allowlisted `trainable.safetensors`、optimizer/RNG `training_state.pt` 和 UTF-8 manifest；记录 config/spec/Git/tokenizer/data provenance 与 hash，拒绝 full model 和 runtime state |
| P16 exit gate | 校验 A2 synthetic marker、完整指标、Reader 稳定/零 LLM 数字分歧、零 TTT activity、四类 hard rollout、finite/reset/cache/FSM/checkpoint、已处理失败样例及全部产物 hash；任一缺失均 fail closed |

## 冻结与产物策略

P15 的 Qwen 全冻结是为低空间工程门禁选定的显式策略，不替代 P21 对全量微调、
分阶段解冻或 LoRA 的正式比较。checkpoint 位于忽略的 `outputs/p15/` 目录；Git 中只保留
`artifacts/` 下的小型 UTF-8 配置、指标、审计、失败样例、冻结策略和指向 checkpoint 的 hash
manifest。

## 证据边界

本阶段证明 Stage A 的工程拓扑、标签隔离、状态生命周期、Reader 算术路径、Outer 参数边界、
指标与产物门禁可执行，不证明真实视频语义、训练收敛、8B 质量、显式状态收益或 TTT
科学增益。真实资产留 P19，消融/校准留 P21，clean 评估与发布审计留 P22。

## 验收结果

验收日期：`2026-07-14`。P15 定向套件（Composer/targets/runtime/trainer/metrics/artifacts、
tiny 端到端、model/config 边界）为 `194 passed`；P3–P15 相邻回归为 `542 passed`；最终全量为
`574 passed`。`ruff check .`、19 个变更 Python 文件 format check、`mypy src`（25 个源码文件）、
配置 CLI、199 个 Git 范围文本文件严格 UTF-8、Architecture/artifact hash、P16 exit gate 和
`git diff --check` 均通过。小型 checkpoint 共 31,462 bytes，保存在忽略目录 `outputs/p15/`。

P16 未开始，也不在本阶段验收范围；本次提交在 P15 验收后停止。

## 证据索引

- `evidence/commands/p15-targeted-pytest.log`
- `evidence/commands/p15-adjacent-pytest.log`
- `evidence/commands/p15-full-checks.log`
- `evidence/commands/p15-utf8-audit.log`
- `artifacts/config-snapshot.yaml`
- `artifacts/metrics.json`
- `artifacts/audit.json`
- `artifacts/failure-examples.json`
- `artifacts/freeze-strategy.md`
- `artifacts/checkpoint-manifest.json`
- `artifacts/bundle-manifest.json`
