# P18 测试时协议与受管推理 Runtime

GATE_STATUS: `passed`

GATE_SCOPE: `runtime_skeleton`

P19_FORMAL_EVALUATION_ALLOWED: `false`

ASSET_POLICY: `Synthetic/tiny CPU engineering evidence only; no video, dataset, or 8B weights`

## 已验证闭环

- per-video manager 原子 reset fast/SGD/cache/slot/State/Identity/FSM/Reader audit 并记录 SHA256；
- 每 chunk 强制 query-time 因果裁剪，observe/hard write 后才调用注入式 updater，并校验更新
  next-only；
- `use_fast_state()` 受管绑定必须被真实 Adapter 消费，版本、update count 和 owner row 对齐；
- `StageARuntimeBridge` 已使用真实 `StageABankWriter` 完成 hard record 回填；
- Query 只执行一次 read/compose/prefill，decode 只推进 LLM KV，runtime checksum 不变；
- invalid/unsupported/empty/ok、retry/new query、future suffix、nested denylist、异常/abort release 已覆盖；
- `python -m ttt_svcbench_qwen.inference --describe-protocol` 与项目 CLI 入口可用。

## 证据边界

本门禁证明 CPU synthetic/tiny runtime 骨架、updater/driver 注入协议和失败边界。生产
`TTTUpdateStage` 尚未把真实 `L_TTT` 与 `functional_sgd_steps_from_ttt` 接入 manager，
`GenerationDriver` 也尚未在真实 8B 上运行；因此不证明真实 generation、BF16、FlashAttention、
多 GPU 或性能，也尚不允许据此开始正式服务器评估。

## 验收

P18 定向/相邻套件 `37 passed`；阶段产物 fail-closed 套件 `8 passed`；Ruff 与 Mypy 通过。
核心证据位于 `tests/test_inference_protocol.py` 与 `tests/test_stage_gate_artifacts.py`。
