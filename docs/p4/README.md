# P4 Query Encoder、Operator Router 与 Time Window Resolver

GATE_STATUS: `passed`

P5_ALLOWED: `true`

ASSET_POLICY: `pinned tokenizer-only; no video or 8B-weight download`

VALIDATION_SCOPE: `synthetic tensors, question-only fixtures, meta parameter audit, local pinned tokenizer offsets`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `549d94706123c5b1871e8bbb76165bf9e4aa7196`（P3 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| dataset/fold | `not-applicable`（P4 不读取视频或标签） |
| 资产 | 只复用 P2 的 11,491,943-byte pinned tokenizer snapshot |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已实现且有本地证据的部分

| 范围 | 已验证实现 |
| :--- | :--- |
| Query 输入 | 只调用 Qwen input embedding table；完整 decoder 不运行；runtime denylist 与 canonical-question/offset 边界保持 |
| Query 主干 | `4096→768`、无参数 sinusoidal position、4 层 Pre-LN 双向 Transformer、GELU、padding-only mask、learned-attention pooling |
| 三路 embedding | 三个独立 `768→1024→512` GELU head；target/operator/time 分别 L2 normalize；无额外 final LayerNorm |
| Router | 9 个归一化 prototype、正的可训练温度 `tau=1.0`、监督 raw logits、hard operator/head mapping、未校准 inference fail closed |
| Resolver | `512→256→4` mode MLP、两个全局 non-padding pointer、唯一候选中英文 seconds/minutes recent/range grammar、显式 TimeWindow |
| 时间完整性 | question-derived component tuple 逐值核对；pointer 必须按序完整覆盖 numeric expression；反向/未来/非法/歧义/局部 span 均失败 |
| 默认语义 | O1-Snap→now；O1-Delta/O2-Gain→recent 且须显式正 duration；O2-Unique/E1/E2→history |
| API gate | train 默认保留 raw 路径；eval 默认启用 calibration gate；invalid/unsupported 均强制 effective operator unsupported |

## 参数审计

| 模块 | 精确参数量 |
| :--- | ---: |
| Query 主干、池化、三个 head | 36,026,112 |
| Operator Router | 4,609 |
| Time Window Resolver | 133,894 |
| **P4 合计** | **36,164,615** |

无参数 sinusoidal position encoding 不改变原 36.03M Query 预算；pool scorer 无 bias，Transformer
和三个 embedding head 均无额外 final LayerNorm。

## 防泄漏与证据边界

`QuestionTokenBatch.source_fields` 是调用方 provenance 声明，不是对自然语言语义的安全证明。
生产路径必须从 trusted canonical question 经 `tokenize_questions()` 构造；当前边界能拒绝 denylist
字段、question 不一致、越界/非单调 offset 和错误声明，但不能识别问题正文中伪装的答案文字。

本阶段没有训练 Query/Router/Resolver，没有读取视频、答案、count 或 occurrence_times，也没有下载
8B 权重。测试只证明工程结构、参数、梯度、mask、解析和 fail-closed 契约，不能支持 operator/time
准确率、吞吐或显存结论。Router/Resolver threshold 仍为 null，最终值必须在 P21 的训练折或独立
校准集冻结。

## 验收结果

验收时间：`2026-07-13T20:20:43.6921761Z`。强制离线环境下，P4 定向验收为 106 passed；全量
`pytest` 为 164 passed；`ruff check .`、`mypy src`、严格 UTF-8 和 `git diff --check` 均通过。
首次全量运行只因同步后的 ARCHITECTURE SHA256 尚未回写 P0 spec-lock 而失败；更新 hash 后完整
重跑为全绿，失败和修复均保留在 full-checks 日志中。

## 证据索引

- `evidence/commands/p4-baseline.log`
- `evidence/commands/p4-parameter-audit.log`
- `evidence/commands/p4-tokenizer-offset-audit.log`
- `evidence/commands/p4-targeted-pytest.log`
- `evidence/commands/p4-full-checks.log`
- `evidence/commands/p4-utf8-audit.log`
