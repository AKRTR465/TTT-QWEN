# v5 optimization baseline

This baseline was captured before the six-stage repository optimization.

- Baseline commit: `c7126a7b6ccbdc071f93c94554d6612a2cf498e9`.
- Architecture spec: `state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval`.
- Production training entry: `python -m ttt_svcbench_qwen.llamafactory_trainer CONFIG.yaml`.
- Inference entry: `ttt-svcbench-infer --run REQUEST.json --checkpoint CHECKPOINT
  --model-root MODEL_ROOT --device cuda:0 --dtype bfloat16 --output RESULT.json`.
- Registered checkpoint state must exclude transient `W_t`, Bank/FSM runtime, temporal caches,
  and other per-video state. Existing checkpoint-boundary tests define the exact policy.
- The architecture snapshot test fixes the Fast Adapter, spatial/temporal encoders, four
  Observation Heads, Query/Reader path, DeepStack indexes, and loss weights during refactoring.

Baseline verification commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m ttt_svcbench_qwen.config
git diff --check
```
