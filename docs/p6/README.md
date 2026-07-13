# P6 空间对象编码器

GATE_STATUS: `passed`

P7_ALLOWED: `true`

ASSET_POLICY: `synthetic tensors only; no video or 8B-weight download`

VALIDATION_SCOPE: `CPU/meta synthetic grids, production dimensions, offline regression`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `700578e219f5a774a9e5647e3f3fb93b8a9afedc`（P5 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| dataset/fold | `not-applicable`（P6 只使用合成张量） |
| 新增资产 | `none`；没有下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已实现且有本地证据的部分

| 范围 | 已验证实现 |
| :--- | :--- |
| Grid 恢复 | 显式 adapted embeddings、visual/tubelet mask 和 merger metadata；支持异构 T/H/W；分别审计 geometry、tubelet 与 effective spatial mask |
| 输入与初始化 | `LayerNorm(4096,eps=1e-5)+Linear(4096→768)`；single q projection；shared seed + fixed non-persistent sinusoidal slot code |
| 两阶段 Slot Attention | 每 Stage 独立 Q/K/V/O、三个 Pre-LN、GRUCell、SiLU FFN；Stage 内三次 refinement 共享参数；经典 slot-axis competition 后 token normalization |
| Recurrent runtime | 每视频显式 previous/next state；无效 tubelet carry；reset 可复现；batch/video/order/storage 隔离；detach 与 differentiable 两种交接 |
| Confidence | 无参数 attention occupancy，范围 `[0,1]`；invalid slot 严格为零 |
| Capacity audit | `required_slot_counts` 只累计 excess/event；preserve-existing/reject-excess，不扩容、不替换且不改变 slot 数值 |
| 数值保护 | padding/NaN 隔离；FP16 masked-slot normalization 在 FP32 中进行，forward/backward 梯度保持有限 |

## 精确参数审计

| 组件 | 参数量 |
| :--- | ---: |
| 输入 LayerNorm + Linear | 3,154,688 |
| 单一 q projection | 393,984 |
| shared slot seed | 768 |
| Stage 1 | 10,632,960 |
| Stage 2 | 10,632,960 |
| **P6 合计** | **24,815,360** |

固定 slot code、confidence、mask、runtime 和 capacity audit 都是零参数。当前新增模块分项和同步为
156.75536M（约 156.76M）。

## 证据边界

P6 的 overflow 只证明调用方显式容量请求的工程保护，不表示编码器已经识别真实对象数。对象语义、
新对象判断和 hard state 属于 P8/P9；P7 负责时间路；P13/P18 负责模型编排和完整受管 runtime；
P19 负责真实 8B/device-map/分布式复验。

本阶段没有训练模型，没有读取真实视频或标签，没有下载 8B 权重，也没有证明对象发现准确率、
吞吐、显存、在线收益或端到端推理效果。

## 验收结果

最终验收时间：`2026-07-13T21:52:09.0682286Z`。强制离线环境下，P3–P6 联合定向验收为
204 passed；状态回写后的最终全量 `pytest` 为 239 passed；`ruff check .`、`mypy src`、严格
UTF-8 和 `git diff --check` 均通过。完整复跑记录保存在 `p6-full-checks.log`。Architecture
SHA256 为
`a7ea6aa94e5726ec848e48f7e14550c269b7c1df01b7174c0dd396d04836661b`。

## 证据索引

- `evidence/commands/p6-baseline.log`
- `evidence/commands/p6-parameter-audit.log`
- `evidence/commands/p6-grid-demo.log`
- `evidence/commands/p6-runtime-overflow-audit.log`
- `evidence/commands/p6-low-precision-audit.log`
- `evidence/commands/p6-targeted-pytest.log`
- `evidence/commands/p6-full-checks.log`
- `evidence/commands/p6-utf8-audit.log`
