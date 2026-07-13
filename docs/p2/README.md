# P2 数据、视频预处理、因果切分与 A0 工程验收

GATE_STATUS: `passed`

P3_ALLOWED: `true`

ASSET_POLICY: `annotation-only / tokenizer-only; no video or 8B-weight download`

PROTOCOL_DEVIATION: `user-approved synthetic fold/A0 engineering substitute`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `f8d568c331ec4e5fbc73a05396036d54957cb3b8`（P1 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `0690c9cf5d8301b644abd87deb01a0a02b3126e30eaa3d18e69b5bb105c57adc` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| P1 最终配置 SHA256 | `70a7497751a9ceb33a2fb4edf8de8b75aea036745571c226026714cbfe6d24da` |
| 当前 P2 配置 SHA256 | `f05e8430532e8f7eb32421091fee2802d663fdd19c92e649641a5795dd8bdb9a` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| fold/seed | 合成非 clean fixture；`GroupKFold(n_splits=3, seed=42)`；manifest 已提交 |

## 已实现且有本地证据的部分

| 范围 | 当前证据 | 边界 |
| :--- | :--- | :--- |
| SVCBench schema | grouped/flat JSONL 均可严格 UTF-8 解析；`occurrence_times` 同时支持点列表和 start/end 对象 | 官方 clean 数据只允许评估，不生成训练 fold |
| 防泄漏 | Dataset、Collator、Trainer、Inference 四层执行 runtime allowlist/denylist | evaluator/trainer 的监督读取入口按用途隔离 |
| 分组划分 | 合成非 clean fixture 按 `source_dataset/video_path` 做 GroupKFold；提交可重建 manifest 并验证视频集合无交集 | 只作为工程门禁，不用于训练或阈值结论 |
| 因果视频 | query-time 右闭截断、重叠 chunk、尾部 padding、tubelet 时间/来源/valid mask 和 overlap 对齐 | 额外用测试期 1,128-byte MP4 验证真实 PyAV 解码路径；文件自动删除 |
| Qwen processor | Transformers 4.57.1 的真实 `Qwen3VLVideoProcessor` 跑通固定和变长 tensor | 使用配置中固定的 checkpoint pixel budget，不加载 8B 权重 |
| Query token | exact-revision 本地 tokenizer 对 Demo 中文问题得到 7 token，并验证动态 padding mask | tokenizer-only cache；不把 `L_q=7` 硬编码 |
| A0 工具链 | local-files-only predictor、严格整数解析、指标聚合、JSON 报告、失败案例和资产审计 | 提交合成 dry-run 报告；明确不是原始 Qwen3-VL-8B 指标 |

## 官方轻量标注审计

只使用已经存在的 `F:\datasets\SVCBench` Git checkout；本阶段没有新增下载。

| 项目 | 已验证值 |
| :--- | :--- |
| Git revision | `a0f0ec08bbc962111dc9761a828267bd032a0f8d` |
| checkout 内全部普通文件 | 6,714,173 bytes（含 `.git`） |
| `data/vcbench_data.jsonl` | 421,830 bytes；1,000 grouped rows；展开为 4,576 query points；406 videos |
| grouped SHA256 | `942b4b6cc2c0d8d339c8ab4605edff03ee5e81784310f24b282faab88211f9ee` |
| `data/vcbench_eval.jsonl` | 1,293,297 bytes；4,576 flat rows；406 videos |
| flat SHA256 | `c2a47e24e5d53a9976f285779affb0da48a77235793de0803bd9a058e361d097` |
| occurrence schema | 621 grouped rows 为 point list；379 rows 为 `{start,end}` 对象 |

官方 README 指定 `vcbench_eval.jsonl` 为评估输入。两个 JSONL 的行身份、时间和标签并非可逐行
互换，因此实现分别解析并保留各自 SHA256，不把 grouped 文件静默替代为 A0 评估文件。

## 小资产占用

- `F:\huggingface_cache` 当前共 11,491,943 bytes，只含 pinned revision 的 tokenizer 四个文件；
- `F:\datasets\SVCBench\data/videos` 不存在；
- `QWEN_MODEL_ROOT` 未设置，8B 权重不存在；
- `SVCBENCH_ROOT` 未设置，但已有标注 checkout 可通过显式审计路径读取；
- 后续任何额外资产下载必须先报告预计大小并取得用户确认。

## 用户批准的合成退出口径

2026-07-14，用户明确要求用合成假数据补齐 P2 缺失部分并尽快进入 P3。该决定只改变 P2
工程施工门禁：`synthetic-fold-manifest.json` 和 `synthetic-a0-report.json` 可作为 P2 交付物。
它不改变科学口径：真实 Qwen3-VL-8B + 官方视频 A0 仍须在 P19/P21/P22 完成；合成指标不得
进入论文比较、增益结论或阈值选择。

## 验收结果

在强制离线环境下，P2 定向验收为 39 passed；`uv sync --frozen` 检查 74 个包；全量
`pytest` 为 70 passed；`ruff check .`、`mypy src` 和严格 UTF-8 审计均通过。P2 已按批准的
合成工程口径通过，`TODO.md` 已勾选，P3 允许开始。

## 证据索引

- `evidence/commands/p2-official-schema-audit.log`
- `evidence/commands/p2-video-processor-demo.log`
- `evidence/commands/p2-tokenizer-demo.log`
- `evidence/commands/p2-a0-asset-audit.log`
- `evidence/commands/p2-synthetic-exit.log`
- `evidence/commands/p2-targeted-checks.log`
- `evidence/commands/p2-full-checks.log`
- `evidence/commands/p2-utf8-audit.log`
- `artifacts/synthetic-fold-manifest.json`
- `artifacts/synthetic-a0-report.json`
