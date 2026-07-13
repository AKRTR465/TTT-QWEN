# P0 仓库基线报告

GATE_STATUS: `passed`

## 可复现起点

| 字段 | 值 |
| :--- | :--- |
| 分支 | `main` |
| 基线 commit | `7f0185f8136faf88cc59e5ba2ec7309c36f8d013` |
| commit 时间 | `2026-07-14T00:27:41+08:00` |
| commit 主题 | `chore: capture document-only repository baseline` |
| 规范/hash | 见 `spec-lock.md` |
| 环境 | 见 `environment-snapshot.md` |

基线 commit 是仓库第一个完整可复现快照。P0 命令执行前，tracked 工作树无修改；命令只新增
`docs/p0/evidence/commands/` 原始日志。

## DOCUMENT-ONLY 起点核对

基线源码只有：

- `src/ttt_svcbench_qwen/__init__.py`；
- `src/ttt_svcbench_qwen/config.py`；
- `src/ttt_svcbench_qwen/py.typed`。

不存在 v5 的模型、Adapter、状态编码器、Bank、Reader、loss、训练或推理实现。现有
`configs/model_state_ttt_8b.yaml` 与 `tests/test_v3_architecture_config.py` 明确属于旧 v3 起点；
`__pycache__` 不计作源码证据。

## 基线命令

| 命令 | 退出状态 | 结果 | 原始日志 |
| :--- | :---: | :--- | :--- |
| `uv sync --frozen` | 0 | Checked 74 packages | `evidence/commands/uv-sync-frozen.log` |
| `uv run pytest -q` | 0 | 5 passed in 8.65s | `evidence/commands/pytest-q.log` |
| `uv run ruff check .` | 0 | All checks passed | `evidence/commands/ruff-check.log` |
| `uv run mypy src` | 0 | no issues in 2 source files | `evidence/commands/mypy-src.log` |

环境、模型 revision 与 hash 的原始记录分别见 `environment-summary.log`、
`model-revision.log` 和 `baseline-hashes.log`。

## 路径和行为审计

- `configs/paths.example.yaml` 只声明 `QWEN_MODEL_ROOT`、`SVCBENCH_ROOT`、`HF_HOME`、
  `OUTPUT_ROOT` 四个环境变量名；
- 排除 `.env.example` 的路径示例与证据日志后，`src/` 和运行配置没有平台绝对路径；
- 相对基线 commit，`src/`、`configs/`、`ARCHITECTURE.md`、`README.md`、`DECISIONS.md`、
  `pyproject.toml` 和 `uv.lock` 均无差异，因此 P0 没有修改模型或运行行为；
- P0 新增内容限定为文档、评审模板和基线文档契约测试。

## P0 交付物与门禁映射

| P0 要求 | 证据 |
| :--- | :--- |
| 规格名、日期、hash | `spec-lock.md`、`baseline-hashes.log` |
| Python/PyTorch/Transformers/CUDA | `environment-snapshot.md`、`environment-summary.log` |
| 本机/服务器职责与差异 | `environment-snapshot.md` |
| 四项原始基线日志 | `evidence/commands/` |
| 路径来源与禁止硬编码 | `environment-snapshot.md`、P0 契约测试 |
| 0–22 需求 ID 和反向追踪 | `requirements-traceability.md` |
| TODO 附录 D 纳入评审 | 追踪表覆盖测试、`.github/pull_request_template.md` |
| 固定/禁止/实验待定分离 | `spec-lock.md` |
| 实验命名和阶段产物 | `execution-policy.md` |
| 计划设计/已验证实现双栏 | `execution-policy.md`、追踪表 |
| 没有把旧 v3 值写成 v5 事实 | `spec-lock.md` 的旧 v3 隔离、本文起点核对 |
| 没有修改模型行为 | 基线差异审计、全量门禁 |

## 阶段验收

验收时间：`2026-07-13T16:50:05.4197640Z`。

| 顺序 | 命令 | 结果 | 原始日志 |
| :---: | :--- | :--- | :--- |
| 1 | `uv run pytest -q tests/test_p0_baseline_contract.py` | 8 passed | `evidence/commands/p0-targeted-pytest.log` |
| 2 | `uv run pytest -q` | 13 passed | `evidence/commands/p0-full-pytest.log` |
| 3 | `uv run ruff check .` | All checks passed | `evidence/commands/p0-ruff-check.log` |
| 4 | `uv run mypy src` | no issues in 2 source files | `evidence/commands/p0-mypy-src.log` |

四项门禁均为 exit code 0，P0 已通过并在 `TODO.md` 标记完成；P1 现在允许开始。
