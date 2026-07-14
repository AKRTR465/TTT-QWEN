# P9 Semantic Projector、类型化 State Bank 与事件 FSM

GATE_STATUS: `passed`

P10_ALLOWED: `true`

ASSET_POLICY: `Meta/small synthetic tensors only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `CPU/meta synthetic records and FSM evidence, offline regression`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `bb3747ee404c4cd7ab2808d55dc9767197bb9ba3`（P8 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `efd613bc0f73aba8f66c18c2e03692c88762320288b110d897ac8e2a8fb7442a` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | `not-applicable`；未下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| Semantic Projector | 四个 768 维 head embedding；共享 `LN→768→1024→512` SiLU trunk；FP32 L2 normalize 与 unit-basis 零向量回退；支持 1D/batch、BF16 输入和梯度 |
| Record/CRUD | `timestamp`/`time_range` 严格二选一；单调 ID、terminal invalid、functional append/update/invalidate/query/snapshot/restore/clear/release；tensor detach+clone 与 storage 隔离 |
| Dynamic view | batch 内只 pad 到 `N_max`；返回 `n_state`、present/record-valid 双 mask、时间和 owner metadata；空 Bank 为 `[B,0,512]` 且 FP32 dtype 稳定 |
| O1 | 六阈值 0.5；baseline 显式 set once；从完整逐槽状态重算；invalid/低置信度/冲突保留已提交状态；exit、overflow 与 overlap evidence drift 可审计 |
| E1 | 0.7/0.3 hysteresis、0.7 completion/transition、0.5 秒 cooldown/NMS；overlap 幂等、non-prefix mask、padding no-op 与 512 history 淘汰 |
| E2 | phase-gated start/end/complete 三步 FSM；精确 0.6/0.7/0.5 边界；COMPLETED 后受控 re-arm；完整区间永久保留、recent history 512 |
| 边界 | Projector 进入模型 `state_dict` 和 Outer optimizer；Bank/FSM/runtime 为零参数，不进入模型 checkpoint 或 optimizer；O2 仅 generic CRUD，生命周期留 P10 |

## 精确参数审计

| 项 | 参数量 |
| :--- | ---: |
| head-type embedding `[4,768]` | 3,072 |
| affine LayerNorm 768 | 1,536 |
| Linear 768→1024 | 787,456 |
| Linear 1024→512 | 524,800 |
| **Semantic Projector 合计** | **1,316,864** |
| **当前新增模块分项总计** | **156,715,683** |
| Bank/FSM/runtime | 0 |

模型 `state_dict()` 仅新增 Projector 的 7 个参数键；record、payload、audit、view 和 snapshot 均不注册
Parameter/buffer。在线可变参数仍仅为 P5 的 1,179,648 个 fast weights。

## 证据边界

本阶段只证明本地结构、数值、状态迁移、隔离、梯度和持久化工程契约，不证明 Projector 或
Observation Head 已学习真实对象/事件语义。没有使用真实视频、正式 SVCBench fold 或
Qwen3-VL-8B 权重。O2 Candidate→Confirmed/容量/Hot Cache 属于 P10；跨 P8 GRU、P9 hard FSM 与
Fast state 的统一 reset/release 编排属于 P18；真实 8B 集成仍属于 P19。

## 验收结果

验收时间：`2026-07-14T00:48:22.4127670Z`。P9 定向验收为 17 passed；P8/P9 相邻定向验收为
107 passed；全量 `pytest` 为 309 passed。`ruff check .`、P9 变更文件 `ruff format --check`、
`mypy src`、配置 CLI、严格 UTF-8、Architecture/uv.lock hash 和 `git diff --check` 均通过。

## 证据索引

- `evidence/commands/p9-baseline.log`
- `evidence/commands/p9-targeted-pytest.log`
- `evidence/commands/p9-fsm-boundary-audit.log`
- `evidence/commands/p9-parameter-audit.log`
- `evidence/commands/p9-full-checks.log`
- `evidence/commands/p9-utf8-audit.log`
