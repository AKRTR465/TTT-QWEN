"""Test-process cleanup for the CPU-heavy second-order contract suites."""

import gc
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def release_test_graphs() -> Iterator[None]:
    yield
    gc.collect()


@pytest.fixture
def h200_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key, value in {
        "OUTPUT_DIR": "/tmp/output",
        "MODEL": "/tmp/qwen3vl8b",
        "DATASET_DIR": "/tmp/svcbench",
        "DATASET_NAME": "svcbench_qwen3vl_sft",
        "VISUAL_COST_INDEX": "/tmp/visual_cost_index.json",
    }.items():
        monkeypatch.setenv(key, value)
    yield
