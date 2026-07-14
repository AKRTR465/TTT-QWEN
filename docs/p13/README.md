# P13 Input Composer 与完整模型编排

GATE_STATUS: `passed`

P14_ALLOWED: `true`

ASSET_POLICY: `Synthetic tensors, a tiny randomly initialized HF Qwen, and the existing 11.5 MB tokenizer-only snapshot; no video, dataset, or 8B-weight download`

VALIDATION_SCOPE: `Special-token persistence, payload placement, native Qwen video/DeepStack/mRoPE, Reader provenance, composition order, and one-prefill lifecycle`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `9c628a7b293190104b32e4b10ca9793bc208e55d`（P12 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| baseline ARCHITECTURE SHA256 | `ae8f738cff76c1ca49b19a515521b89f0d392eb2d5832dff02dcf1478f302c18` |
| final ARCHITECTURE SHA256 | `b1ef88726da124835b43e1e64f2ee3d430a28d4670db785a223ff51224e07329` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| 数据与模型资产 | 未下载视频、数据集或 8B 权重；只读取既有 tokenizer-only snapshot |
| 软件 | Python 3.12.13；PyTorch 2.9.0+cu128；Transformers 4.57.1 |

## 已验证实现

| 范围 | 已验证契约 |
| :--- | :--- |
| Token 注册 | 5 个固定 token 按序注册，pinned ID 为 151669–151673；tokenizer 151669→151674；151936-row 模型表绝不缩小 |
| 可复现 embedding | 首次新增时 input/lm_head 使用 vision-start/video-pad/vision-end 三行 FP32 均值再 cast；tied 只写一次；reload 不重置 |
| Payload | 保留原 chat/template/video IDs，在最后 user end 前插入 State 与 exact-number instruction/Reader IDs；OK/EMPTY 注入，UNSUPPORTED/INVALID 省略 |
| Batch/mask | 变长 batch 左 padding；video/state/number mask 两两互斥且为 attention 子集；逐行位置、真实长度和 Reader IDs 可审计 |
| mRoPE/cache | Composer 以完整 IDs/grid/mask 调原生 `get_rope_index`；`[3,B,L]` position、`[B,1]` rope delta、`[L]` cache 与 tiny HF 原生 prefill 完全一致 |
| Qwen continuation | `PreparedVideoFeatures` 复用 Adapter 后 Main 与原 DeepStack，不重跑 ViT/Fast；State embedding 仅 prefill scatter 一次，video 仍原生 scatter |
| DeepStack/decode | 三组 DeepStack 对象、顺序与 visual-only mask 不变；prefill 后 2 个以上 decode step 不再 scatter State 或触发状态路径 |
| Reader/Resampler | 同一 Retriever snapshot 必须执行 `read→audit_results→audit_number_tokens` 后才允许 Resampler/Composer；篡改与 provenance 漂移阻断 |
| Model 编排 | `StateTTTModel(nn.Module)` 提供 observe/answer/decode，按 identity 唯一注册组件；runtime/lifecycle 不入 state_dict；prefill 失败或重复调用 fail closed |
| 消融与指标 | Fast 独立开关；Bank off 强制 Reader/State off；支持 state-only/number-only payload；Reader/LLM number agreement 与训练 target agreement 独立计算 |

## 小型合成口径

所有 Composer、编排和失败样例均在测试进程构造。tiny HF Qwen 为 hidden size 8、3 层视觉和
3 层文本的随机模型，仅验证真实 Transformers 4.57.1 调用语义。唯一外部只读资产是既有
Qwen tokenizer 四文件 snapshot，共 11,491,943 bytes。没有下载或加载 8B 权重。

## 证据边界

本阶段证明工程拓扑、token/embedding 持久化、payload、mRoPE、DeepStack、Reader provenance 和
一次性生命周期，不证明自然语言质量、SVCBench 精度、训练收敛或 TTT 科学增益。Loss/functional
SGD 属于 P14，完整测试时 reset 属于 P18，真实 8B 属于 P19。

## 验收结果

验收日期：`2026-07-14`。P13 定向 `69 passed`；P3–P13 相邻回归 `432 passed`；最终全量
`478 passed`。`ruff check .`、11 个变更 Python 文件 format check、`mypy src`、配置 CLI、
167 文件严格 UTF-8、Architecture/uv.lock hash 和 `git diff --check` 均通过。

## 证据索引

- `evidence/commands/p13-targeted-pytest.log`
- `evidence/commands/p13-adjacent-pytest.log`
- `evidence/commands/p13-full-checks.log`
- `evidence/commands/p13-tokenizer-audit.log`
- `evidence/commands/p13-utf8-audit.log`
