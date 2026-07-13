# P7 时间事件编码器

GATE_STATUS: `passed`

P8_ALLOWED: `true`

ASSET_POLICY: `synthetic tensors only; no video or 8B-weight download`

VALIDATION_SCOPE: `CPU/meta/available local FP16 synthetic tensors, production dimensions, offline regression`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `850a2f56d0cecf88eff9e952dd667c5965fdad6b`（P6 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| 数据 | `not-applicable`（P7 只使用小型合成张量） |
| 新增资产 | `none`；没有下载视频、数据集或 8B 权重 |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已实现且有本地证据的部分

| 范围 | 已验证实现 |
| :--- | :--- |
| Tubelet pooling | 复用异构 merger-grid 恢复；`LayerNorm(4096)+Linear(4096→768)`；单一 q_target projection；12×64 完整 Q/K/V/O 空间 attention；无效空间/tubelet 严格屏蔽 |
| 时间主干 | 无参数 FP64 计算的 absolute sinusoidal global position；六个独立 Pre-LN GELU Transformer layer；12×64 attention、FFN 3072、dropout 0.1、无 final norm |
| 因果规则 | full/cache 共用 `q-63 <= k <= q`，禁止未来并允许 self；padding 不进入 K/V/cache；未来扰动不改变过去 |
| Runtime cache | 六层独立 K/V、最终 hidden、global position、canonical FP64 timestamps、owner/query signature、total_seen；主 cache 最多 64 |
| Overlap | 固定 4-tubelet replay/replace；额外 3-position replay-only K/V margin 只补足重算上下文，不扩大 64-token mask；满 cache 后仍与 full 对齐 |
| 隔离与梯度 | video/trajectory/query owner、batch/order/storage/reset fail-closed；函数式 next cache；detach true/false 与当前输出梯度边界均验证 |
| 数值保护 | FP32/FP64 时间元数据与模型 dtype 解耦；invalid NaN/Inf poison 隔离；all-invalid 安全；本地 FP16 forward/backward 有限 |

## 精确参数审计

| 组件 | 参数量 |
| :--- | ---: |
| 输入 LayerNorm + Linear | 3,154,688 |
| 单一 q_target projection | 393,984 |
| 空间 pooling Q/K/V/O | 2,362,368 |
| 单个 Transformer layer | 7,087,872 |
| 六个 Transformer layer | 42,527,232 |
| **P7 合计** | **48,438,272** |

absolute position、mask、main/replay cache、owner metadata、audit 和 detach runtime 都是零参数。
当前新增模块分项和同步为 156,703,632（156.703632M）。

## 证据边界

本阶段只证明时间编码器的本地工程契约，不表示已训练出事件语义。没有读取真实视频或标签，
没有实例化/下载真实 Qwen3-VL-8B，没有运行 Observation Head、Bank、Reader、在线 TTT loss、
端到端推理、吞吐/显存或正式 SVCBench 评估。P8 才拥有 O1/O2/E1/E2 soft decoder；P9 以后
拥有 hard state、检索和确定性读取。

timestamp overlap identity 容差按 FP32 相对精度定义，适用于当前视频内相对秒数；若未来改用
Unix 绝对时间或多日轨迹，需要增加时间戳来源精度元数据并重新冻结该契约。

## 验收结果

最终验收时间：`2026-07-13T22:57:43.7360378Z`。强制离线环境下，P3–P7 联合定向验收为
236 passed；状态与证据回写后的最终全量 `pytest` 为 271 passed；`ruff check .`、`mypy src`、
严格 UTF-8、配置 CLI 和 `git diff --check` 均通过。完整记录保存在 `p7-full-checks.log`。

Architecture SHA256：
`99d261401c5f8b403fee2732aca36ac43910d5603f2c6f60365fa0e0f3b6578a`。

## 证据索引

- `evidence/commands/p7-baseline.log`
- `evidence/commands/p7-parameter-audit.log`
- `evidence/commands/p7-causal-cache-audit.log`
- `evidence/commands/p7-low-precision-audit.log`
- `evidence/commands/p7-targeted-pytest.log`
- `evidence/commands/p7-full-checks.log`
- `evidence/commands/p7-utf8-audit.log`
