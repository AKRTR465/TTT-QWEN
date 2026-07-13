# P8 四类 Observation Decoder

GATE_STATUS: `passed`

P9_ALLOWED: `true`

ASSET_POLICY: `Meta/small synthetic tensors only; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `CPU/meta synthetic tensors, production dimensions, offline regression`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `95cbd1248f1dfbd5f9ba49a97d8931ff8a95ee07`（P7 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| 数据与模型资产 | `not-applicable`；未下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| O1 | `LN(A)*(1+scale)+shift` FiLM；逐槽 `768→1024→1024→6`；raw logits、sigmoid debug probability 与 soft count |
| O2 | 共享 `768→1024→1024` trunk；256 维 identity 在 FP32 L2 normalize；有效零向量确定性回退到首个单位基；2 维 raw score logits |
| E1 | 五层严格左填充 gated causal TCN，dilation `1/2/4/8/16`、RF=63；66-position projected history 支持固定 4-position replay/replace |
| E2 | 两层单向 768 维 GRU、dropout=0；最近 5 个 checkpoint 支持固定 4-position rollback/recompute |
| 输出 | O1/O2 与槽轴、E1/E2 与 tubelet 轴对齐；统一携带 mask、timestamp、global position；invalid tensor 为零且 metadata 为 `-1` |
| Runtime | E1/E2 按 video/trajectory/query signature 隔离；owner、query、position、timestamp、gap 和超范围 rewind 均 fail closed |
| 梯度 | 在线冻结只关闭 Head 参数梯度并进入 eval；未使用 `no_grad` 或 detach 输入，soft output 仍可回传到 slots、H_t 和 q_target |
| 边界 | 未实现 hard count、Bank/FSM、identity lifecycle、loss 或 P13/P18 编排；这些职责仍属于后续阶段 |

## 精确参数审计

| Head | 参数量 |
| :--- | ---: |
| O1 | 2,632,710 |
| O2 | 2,103,042 |
| E1 | 9,584,643 |
| E2 | 7,094,792 |
| **P8 合计** | **21,415,187** |

当前新增模块分项总计同步为 156,718,819（156.718819M）；在线可变参数仍仅为 P5 的
1,179,648 个 fast weights。所有 runtime cache、checkpoint、mask、probability 和 metadata 均为零参数。

## 证据边界

本阶段只证明本地工程契约、因果性、流式等价、隔离和梯度边界，不证明四个 Head 已学到真实对象或
事件语义。没有使用真实视频、正式 SVCBench fold 或 Qwen3-VL-8B 权重，也没有运行 hard State Bank、
身份生命周期、Reader、在线 TTT 更新或端到端评估。P9 可在这些稳定 soft observation 上实现
Semantic Projector、类型化 State Bank 与事件 FSM。

## 验收结果

最终验收时间：`2026-07-13T23:40:56.5140318Z`。P5–P8 相邻定向验收为 123 passed；全量
`pytest` 为 285 passed；`ruff check .`、`mypy src`、配置 CLI、严格 UTF-8、Architecture hash
和 `git diff --check` 均通过。Architecture SHA256 为
`edf71c762d742d79fbe9fe8e607c8db2fb2e5df4921e2b9bd32c9d94643fea2b`。

## 证据索引

- `evidence/commands/p8-baseline.log`
- `evidence/commands/p8-parameter-audit.log`
- `evidence/commands/p8-streaming-audit.log`
- `evidence/commands/p8-targeted-pytest.log`
- `evidence/commands/p8-full-checks.log`
- `evidence/commands/p8-utf8-audit.log`
