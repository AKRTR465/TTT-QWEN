# P1 v5 配置、类型契约与模块骨架

GATE_STATUS: `passed`

## 实施前基线

| 字段 | 值 |
| :--- | :--- |
| Git branch/commit | `main` / `573a01d`（P0 已验收提交） |
| spec_version | `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval` |
| ARCHITECTURE SHA256 | `0690c9cf5d8301b644abd87deb01a0a02b3126e30eaa3d18e69b5bb105c57adc` |
| uv.lock SHA256 | `c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2` |
| P1 前配置 SHA256 | `cdc0c23c873346176458c09534166f01d1c2a9208e05aae7ff50782de593b43c` |
| model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| dataset fold | `not-applicable`（P1 不读取数据） |
| seed | `not-applicable`（P1 不执行随机模型计算） |

## 本阶段实现

- 将旧 v3 YAML 原子迁移为 v5 全量配置；
- 在 `config.py` 定义 immutable、unknown-key-forbid 的 Pydantic schema 和跨组件强校验；
- 对 27/1152/16、36/4096、768 fast/state 主干、32 slots、16 State Token、动态 Bank 容量、
  9 operator、no Top-K/ANN、单步 SGD、loss 和参数预算建立启动前契约；
- 所有 bootstrap/FSM/match/time/operator 阈值保留未校准状态，并阻止正式评估；
- 建立 15 个职责单一的模块骨架；
- 建立 Video、Query、TimeWindow、Encoder/cache、四 Decoder、typed record、Retriever、
  ReaderResult 和 per-video runtime 类型；
- 将旧 v3 配置测试迁移为 v5 配置、类型和空壳测试。

## 固定契约核对

| 维度 | 证据 |
| :--- | :--- |
| shape | 配置跨组件校验、`tests/test_v5_runtime_types.py` |
| mask | Video/slot/temporal/query/retrieval 类型的 bool mask 断言 |
| dtype/device | Tensor 类型断言；fast/DeepStack 要求相同 dtype/device |
| gradient/update | 两个 `768×768` fast matrix、1,179,648 参数与 SGD 配置断言 |
| reset/isolation | per-video runtime 覆盖 fast/optimizer/slot/cache/Bank/FSM/Reader audit |
| leakage | TimeWindow 禁止越过 query_time；模块职责禁止答案/未来记录 |

## 明确未实现

所有 builder/compute/run 入口仍显式抛出 `NotImplementedError`，并在错误中标出负责实现的
P3–P19。P1 没有返回占位 tensor、伪造结果或模型能力；P2 及以后必须按 Part 顺序替换对应入口。

## 验收记录

验收时间：`2026-07-13T17:11:38.0136856Z`。最终 v5 配置 SHA256：
`70a7497751a9ceb33a2fb4edf8de8b75aea036745571c226026714cbfe6d24da`。

| 顺序 | 命令 | 结果 | 原始日志 |
| :---: | :--- | :--- | :--- |
| 1 | P1 三个定向测试文件 | 38 passed | `evidence/commands/p1-targeted-pytest.log` |
| 2 | 配置 CLI | exit 0，完整输出 264 行 | `evidence/commands/p1-config-print.log` |
| 3 | `uv sync --frozen` | Checked 74 packages | `evidence/commands/p1-uv-sync-frozen.log` |
| 4 | `uv run pytest -q` | 47 passed | `evidence/commands/p1-full-pytest.log` |
| 5 | `uv run ruff check .` | All checks passed | `evidence/commands/p1-ruff-check.log` |
| 6 | `uv run mypy src` | no issues in 17 source files | `evidence/commands/p1-mypy-src.log` |
| 7 | UTF-8 strict decode audit | 54 text files | `evidence/commands/p1-utf8-audit.log` |
| 8 | 勾选后最终全量回归 | 全部绿色 | `evidence/commands/p1-final-recheck.log` |

全部 exit code 为 0，P1 已通过并在 `TODO.md` 标记完成；P2 现在允许开始。
