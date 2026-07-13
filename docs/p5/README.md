# P5 Fast TTT Adapter 与 fast 参数边界

GATE_STATUS: `passed`

P6_ALLOWED: `true`

ASSET_POLICY: `synthetic tensors only; no video or 8B-weight download`

VALIDATION_SCOPE: `CPU/meta synthetic tensors, local P3 boundary, offline full regression`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `412a9d95f6ce85248eb27ecc2bc6c1f1278a0489`（P4 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| dataset/fold | `not-applicable`（P5 只使用合成张量） |
| 新增资产 | `none`；没有下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已实现且有本地证据的部分

| 范围 | 已验证实现 |
| :--- | :--- |
| 张量结构 | RMSNorm(`eps=1e-6`) → 带 bias `4096→768` → 两块无 bias `768×768` fast matrix → SiLU → 带 bias `768→4096` → 0.1 residual |
| 参数边界 | checkpointed `W0` 与 slow 参数完整注册；per-video `W_t` 为外部临时 Tensor；fast 参数固定顺序且精确为 1,179,648 |
| runtime/reset | 每个视频从当前 `W0` 独立克隆；reset 清零 version/update/skip；不同 batch row 的 `W_t` storage 不共享 |
| 梯度 | online functional 路径只给输入与 `W_t` 梯度；differentiable 路径保留 `W0` 和 slow meta-gradient；forward/backward 不原地修改状态 |
| P3 边界 | `use_fast_state()` 受管绑定期间冻结注册参数、拒绝 stale grad、禁止重入并异常安全恢复；Main Merger 输出被适配而 DeepStack 保持原对象 |
| mask/audit | padding 残差严格为零；逐 batch row 记录 version、有效 token、`W_t` norm、输入 norm 与真实缩放后残差 norm |

## 参数审计

| 参数组 | 精确参数量 |
| :--- | ---: |
| RMSNorm 与两层带 bias 慢投影 | 6,300,416 |
| checkpointed meta-fast `W0` | 1,179,648 |
| **checkpointed Adapter 合计** | **7,480,064** |
| 每视频 transient online-fast `W_t` | 1,179,648 |

`W_t` 不额外计入 module 参数或 checkpoint；其数值只是 `W0` 的 per-video 临时副本。

## 证据边界

P5 没有实现或执行一步 SGD、gradient clip、TTT loss、跨 chunk 生效顺序或正式推理协议：这些分别
属于 P14 和 P18。P18 仍须验证 video ID 与 batch row/order 对齐、并发隔离及每次受管调用确实使用
runtime state。P19 仍负责真实 8B/device-map/分布式复验。

本阶段没有训练模型，没有读取真实视频或标签，没有下载 8B 权重，也没有证明吞吐、显存、在线
收益或 Meta-TTT 科学效果。全量离线回归仅复用仓库中已有的 pinned tokenizer snapshot。

## 验收结果

最终验收时间：`2026-07-13T20:55:49.3144986Z`。强制离线环境下，P5 定向验收为 106 passed；
状态回写后的最终全量 `pytest` 为 191 passed；`ruff check .`、`mypy src`、严格 UTF-8 和
`git diff --check` 均通过。完整复跑记录保存在 `p5-full-checks.log`，Architecture SHA256 已同步为
`83e968fadc69961aebbd615e1d2146aa21ab25982fbcb5104d919c90934fa7df`。

## 证据索引

- `evidence/commands/p5-baseline.log`
- `evidence/commands/p5-parameter-audit.log`
- `evidence/commands/p5-runtime-state-audit.log`
- `evidence/commands/p5-qwen-boundary-audit.log`
- `evidence/commands/p5-targeted-pytest.log`
- `evidence/commands/p5-full-checks.log`
- `evidence/commands/p5-utf8-audit.log`
