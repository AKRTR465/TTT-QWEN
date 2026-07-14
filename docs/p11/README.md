# P11 Embedding State Retriever

GATE_STATUS: `passed`

P12_ALLOWED: `true`

ASSET_POLICY: `Small deterministic synthetic StateRecord tensors only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `FP32 exact threshold retrieval over detached typed State Bank snapshots`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `9d59a25552dee8431d7f053677e4dddeff9c16ed`（P10 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `0db99432134fbb576069298172c7747b265d1cb5d0ec0d215e1e2abd773b0993` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | `not-applicable`；未下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| 候选分区 | effective hard operator 映射到逐行 head；owner/head 分区后统计包含 invalid 与 O2 Candidate 的 `N_s`；ragged batch 生成 padded FP32 view |
| 相似度 | q_target 与 semantic embedding 在 FP32、`eps=1e-8` 下重新 L2 normalize；阈值先量化到 FP32，`score >= 0.35` 边界确定命中 |
| 硬过滤 | `invalid → retrieval_ineligible → future → outside_window → below_similarity` 互斥归因；aggregate 与 O2 atomic 使用 kind-aware 时间规则 |
| 全量返回 | 3/30/300 条命中全部保留，仅按 `score desc, record_id asc` 排序；`top_k=null`、`ann_enabled=false` |
| 输出合同 | 返回未压缩 typed records、IDs、scores、candidate/selected masks、`N_s/N_ret`、Bank version 与结构化 status/reason/audit |
| 状态分流 | owner mismatch/invalid time → invalid；unsupported operator/time/退化 query → unsupported；可靠无命中 → empty；混合过滤 → no_match |
| 因果防线 | State Bank 拒绝 aggregate record timestamp 与 payload 最新时间不一致，以及 payload slot/event/interval 含未来时间的记录 |
| 离线指标 | runtime 无标签；离线 evaluator 计算 micro precision/recall，零分母返回 `None`，empty rate 排除 unsupported/invalid 并单独报告其 rate |
| 边界 | Retriever 零参数、不修改 Bank；scores 保留 q_target 梯度，hard records 与输出 snapshot 保持 detached/clone |

## 小型合成口径

全部验收只在测试进程内构造 512 维单位 semantic vectors、256 维 identity prototypes 和 typed
payload。最大 Retriever case 为 300 条记录，未创建或下载真实视频、SVCBench 数据、checkpoint
或 Qwen3-VL-8B 权重。

## 证据边界

本阶段只证明检索工程合同、因果过滤、状态分流和离线指标实现。`0.35` 仍是
`bootstrap_calibration_required`，必须在 P21 使用训练折或独立校准集复校；合成检索结果不得用于
论文性能、召回率或科学增益结论。P12 只能从 P11 返回的未压缩 typed records 做确定性算术，不能
从 embedding 或 LLM 猜测 exact integer。

## 验收结果

验收时间：`2026-07-14T03:12:39.6354609Z`。P11 Retriever 定向为 21 passed；P11 与 State Bank
因果边界联合定向为 43 passed；P4/P9/P10/P11 相邻回归为 229 passed；最终全量 `pytest` 为
381 passed。`ruff check .`、9 个变更 Python 文件 `ruff format --check`、`mypy`、配置 CLI、严格
UTF-8、Architecture/uv.lock hash 和 `git diff --check` 均通过。

## 证据索引

- `evidence/commands/p11-targeted-pytest.log`
- `evidence/commands/p11-adjacent-pytest.log`
- `evidence/commands/p11-full-checks.log`
- `evidence/commands/p11-utf8-audit.log`
