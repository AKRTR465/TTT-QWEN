# P12 16-token State Resampler 与 Deterministic Reader

GATE_STATUS: `passed`

P13_ALLOWED: `true`

ASSET_POLICY: `Small deterministic synthetic typed records plus the existing 11.5 MB tokenizer-only snapshot; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `State-token resampling, typed-record integer arithmetic, provenance, status, and number-token integrity`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `b54d1258cba8852aa16b6ca9c808684e9c236255`（P11 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| baseline ARCHITECTURE SHA256 | `0db99432134fbb576069298172c7747b265d1cb5d0ec0d215e1e2abd773b0993` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | 未下载视频、数据集或 8B 权重；只读取既有 tokenizer-only snapshot |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| Resampler 拓扑 | 16 learned queries、3 层 Pre-LN、8-head self/cross attention、GELU `512→2048→512`、`512→4096` 输出；精确 14,722,048 参数 |
| K/V 与 mask | 只打包全部 selected records；`N_s>N_ret` 非连续 selected 不混入候选；0/3/30/300 records 均输出 `[B,16,4096]` |
| 数值 | QK logits、缩放、masked softmax 与最终 attention audit 为 FP32；padding 权重严格 0，非空 selected mass=1 |
| 状态隔离 | OK/EMPTY 产生有效 State Token；EMPTY 使用 trainable sentinel；UNSUPPORTED/INVALID hidden/token 归零、valid mask=false 且不向 query/empty KV 传播梯度 |
| Reader provenance | RetrieverOutput 固定 operator、完整 TimeResolution 与 candidate typed snapshot；selected payload/semantic/ID/owner 替换或 tensor mutation 会在消费前拒绝 |
| 精确算术 | O1 signed fixed baseline、O2 unique/closed first_seen、E1 cumulative/retained completion、E2 completion-end 共 8 operator 与错型/截断边界均已验证 |
| 双向审计 | 成功结果携带 records→operands→exact_count 标量链；`audit_results` 用同一 RetrieverOutput 重算完整结果，拒绝同步替换 count/number IDs |
| Tokenizer | Qwen2TokenizerFast、vocab 151,643、revision 对应四文件 manifest SHA256 `ccd18347...f44c3`；canonical signed text 执行 encode→decode→re-encode 同 IDs |

## 小型合成口径

全部 Resampler/Reader case 在测试进程内构造 512 维 semantic vectors、256 维 identity prototypes、
O1/E1/E2 aggregate 与 O2 Candidate/Confirmed records。最大 case 为 300 条记录。唯一外部只读资产是
现有四文件 tokenizer snapshot，共 11,491,943 bytes；未触发 Hugging Face 下载。

## 证据边界

本阶段证明工程拓扑、数值、状态、因果窗口、来源与整数完整性，不证明随机初始化 State Token 已有
自然语言语义，也不报告训练、SVCBench 指标或科学增益。Resampler 语义质量留 P15/P21，真实 8B
集成留 P19。P13 Composer 只能消费通过 `audit_results` 的 ReaderResult。

## 验收结果

验收日期：`2026-07-14`。P12 定向为 41 passed；P12/P11/config 联合定向为 194 passed；
P4/P9/P10/P11/P12 相邻回归为 285 passed；最终全量 `pytest` 为 442 passed。`ruff check .`、
13 个变更 Python 文件 `ruff format --check`、`mypy src`、配置 CLI、严格 UTF-8、Architecture/uv.lock
hash 和 `git diff --check` 均通过。

## 证据索引

- `evidence/commands/p12-targeted-pytest.log`
- `evidence/commands/p12-adjacent-pytest.log`
- `evidence/commands/p12-full-checks.log`
- `evidence/commands/p12-utf8-audit.log`
