# P10 Identity Bank 动态容量与 Hot Cache

GATE_STATUS: `passed`

P11_ALLOWED: `true`

ASSET_POLICY: `Small deterministic synthetic identity tensors only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `CPU FP32 exact identity lifecycle plus explicit CPU Hot Cache test backend`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `78e7fd930595173aa31e1f4d30a57edf6099c2f0`（P9 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `65cbbdd69f34aa14b504830f343e7bd03fc3656e8bbbb4ee5fd368b7ad108caf` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | `not-applicable`；未下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| Candidate | 逻辑容量 64 起、按 64 增长至 512；CPU FP32 单位 prototype；TTL=8；same-position replay 幂等；expiry→low-confidence→reject 顺序与显式 audit |
| Confirmed | 初始 256、按 256 CPU chunk 增长、无硬上限；第 257 个身份保留前 256 个全部字段；`unique_count` 从 authoritative store 派生 |
| Matching | 完整 CPU store cosine exact scan；阈值 0.80；top-2 差不超过 `1e-6` fail closed；多人争用时确定性一对一；ANN 明确关闭 |
| Lifecycle | 两个连续可靠 committed position 才晋升；Candidate record terminal invalid 后追加 Confirmed record；`first_seen` 与 record timestamp 保持不变 |
| Prototype | FP32 `normalize(0.9*old+0.1*observation)`；零向量回退首个 unit basis；旧/新 checksum 和版本可审计 |
| Hot Cache | 容量 256、CUDA BF16 LRU 副本；无 CUDA 时显式禁用；测试可显式注入 CPU backend；cache hit 也不短路 CPU full exact scan |
| 边界 | video/trajectory/release 隔离；snapshot/restore/clear/release 深拷贝；hard write detach+clone；runtime 不进入模型 state_dict 或 optimizer |
| 指标 | duplicate excess 与 missed-new rate 由独立 trajectory-end evaluator 使用 GT 计算；标签不进入在线 Bank/runtime |

## 小型合成压力口径

容量与缓存验收只使用固定随机种子的 256 维单位向量。513 个 identity prototype 约占
`513×256×4 ≈ 0.50 MiB`，semantic fixture 也只在测试进程内临时生成；未创建或下载真实视频、
SVCBench 数据、checkpoint 或 Qwen3-VL-8B 权重。

## 证据边界

本阶段只证明 Identity Bank 的工程状态迁移、容量、exact semantics、缓存非权威性、隔离和持久化
边界。0.80/0.50/`1e-6` 均为 `bootstrap_calibration_required`，必须在 P21 用训练折或独立校准集
复校；合成 duplicate/missed-new 结果不得用于论文性能或科学增益结论。P10 只提供
Reader-facing authoritative count/view，一直到 P12/P20 才复验真实 ReaderResult。

## 验收结果

验收时间：`2026-07-14T01:42:47.6223741Z`。P10 定向验收为 15 passed；P9/P10 相邻定向验收为
109 passed；最终全量 `pytest` 为 344 passed。`ruff check .`、10 个变更 Python 文件
`ruff format --check`、`mypy`、配置 CLI、151 个仓库文件严格 UTF-8、Architecture/uv.lock hash 和
`git diff --check` 均通过。

## 证据索引

- `evidence/commands/p10-baseline.log`
- `evidence/commands/p10-targeted-pytest.log`
- `evidence/commands/p10-capacity-cache-audit.log`
- `evidence/commands/p10-full-checks.log`
- `evidence/commands/p10-utf8-audit.log`
