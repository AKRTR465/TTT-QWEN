# P14 TTT/State/Answer Loss 与 Functional SGD

GATE_STATUS: `passed`

P15_ALLOWED: `true`

ASSET_POLICY: `Synthetic tensors and tiny in-memory module chains only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `Typed per-row losses, current-to-snapshot detach direction, State/Answer/Outer objectives, row-isolated one-step SGD, full-second-order meta gradients, skip accounting, and module gradient/delta audit`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `e11aba3cbf10f97845e2a02e4c5d05536ad321ae`（P13 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| baseline ARCHITECTURE SHA256 | `b1ef88726da124835b43e1e64f2ee3d430a28d4670db785a223ff51224e07329` |
| final ARCHITECTURE SHA256 | `d397cab2943fac9a032b718800d37c69c60e1b82152d395a8aa869cc3f454597` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | 未下载视频、数据集或 8B 权重；全部为进程内合成张量/模块 |

## 已实现合同

| 范围 | 合同 |
| :--- | :--- |
| Predictor | `LayerNorm(768)→Linear(1536)→SiLU→Linear(768)`，精确 2,363,136 参数；仅连续 valid tubelet pair，target detach |
| Overlap | O2/E1/E2 统一 current prediction→detached previous snapshot；position/timestamp/唯一 pair/status 强类型审计 |
| TTT reduction | 每 row 固定 `pred + 0.5 id + 0.5 event`，无效项为零且不重归一；唯一 scalar 为 union-valid rows mean；O1 权重严格为零 |
| State | 仅对应任务 Head；O1 pre-matched 六字段、O2 identity/score、E1、E2 event/soft-phase proxy，以及 operator/retrieval/time 显式标签 |
| Answer/Outer | causal shift、`ignore_index=-100`、number/answer/Reader 三类指标；当前与额外 support TTT 均进入 `outer + 0.1 mean(TTT)` |
| Functional SGD | typed TTT row→单 FastWeightsState；SGD `1e-4`、无 momentum/decay、联合 FP32 clip=1；成功/skip/attempt counter 事务化 |
| 梯度模式 | online=`online_leaf`；meta=`meta_full_second_order`，`create_graph=True` 且不 detach，outer gradient 可回到 `W0` |
| 故障保护 | 无有效项、时间不足、非有限 loss/gradient、零梯度、clip 后非法、dtype 不可表示更新均显式审计并安全跳过 |
| 模块审计 | 输出每组 parameter count、gradient presence/norm、delta norm 与 expected/allowed；只允许下一代 fast matrices 产生 inner delta |

## 合成口径与证据边界

所有样例均在测试进程内构造。门禁只证明 loss 数学、detach、mask、梯度、更新事务和类型边界，
不证明真实视频语义、8B 质量、训练收敛或 TTT 科学增益。O1 slot assignment/标签构造属于 P15；
跨 chunk snapshot/matching 生命周期属于 P17；正式跨视频推理 reset 与并发隔离属于 P18。

## 验收结果

验收日期：`2026-07-14`。P14 定向 `188 passed`；P3–P14 相邻回归 `497 passed`；最终全量
`537 passed`。`ruff check .`、9 个变更 Python 文件 format check、`mypy src`（21 个源码文件）、
配置 CLI、172 个文本文件严格 UTF-8、Architecture/uv.lock hash 和 `git diff --check` 均通过。

## 证据索引

- `evidence/commands/p14-targeted-pytest.log`
- `evidence/commands/p14-adjacent-pytest.log`
- `evidence/commands/p14-full-checks.log`
- `evidence/commands/p14-utf8-audit.log`
