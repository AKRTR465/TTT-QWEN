from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    module_name = "test_script_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_gpu_telemetry_parser_rejects_malformed_rows() -> None:
    module = _load_script("scripts/h200/capture_gpu_telemetry.py")
    assert module.parse_nvidia_smi("0, 91, 12345, 612.5\n1, 87, 12000, 598.0\n") == [
        (0, 91.0, 12345, 612.5),
        (1, 87.0, 12000, 598.0),
    ]

    try:
        module.parse_nvidia_smi("0, 91, 12345")
    except ValueError as error:
        assert "unexpected nvidia-smi" in str(error)
    else:
        raise AssertionError("malformed telemetry must fail")


def test_train_log_bridge_extracts_only_finite_training_scalars() -> None:
    module = _load_script("scripts/h200/bridge_train_log_tensorboard.py")
    row = module.parse_row(
        "prefix {'loss': 0.5, 'learning_rate': 1e-5, 'loss/task': 0.2, "
        "'outer_grad/qwen': 3.0, 'ignored': 8, 'flag': True, 'bad': nan}"
    )
    assert row is None

    row = module.parse_row(
        "prefix {'loss': 0.5, 'learning_rate': 1e-5, 'loss/task': 0.2, "
        "'outer_grad/qwen': 3.0, 'ignored': 8, 'flag': True}"
    )
    assert row is not None
    assert dict(module.scalar_items(row)) == {
        "loss": 0.5,
        "learning_rate": 1e-5,
        "loss/task": 0.2,
        "outer_grad/qwen": 3.0,
    }


def test_retrieval_history_benchmark_uses_current_tensor_ring() -> None:
    module = _load_script("scripts/benchmark_retrieval_history.py")
    batch = module._batch(3, dtype=torch.float32, device=torch.device("cpu"))
    rows = module._row_batches(batch)
    assert len(rows) == 6
    assert sum(row.sources.shape[0] for row in rows) == batch.sources.shape[0]

    rowwise = module._write_seconds(
        2,
        3,
        1,
        device=torch.device("cpu"),
        vectorized=False,
    )
    batched = module._write_seconds(
        2,
        3,
        1,
        device=torch.device("cpu"),
        vectorized=True,
    )
    assert math.isfinite(rowwise) and rowwise > 0.0
    assert math.isfinite(batched) and batched > 0.0
