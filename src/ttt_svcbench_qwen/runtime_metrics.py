"""No-op runtime-tracing shim.

The buffered JSONL writer, CUDA event-pair machinery, and trace configuration were removed.
This shim keeps the four call shapes so wrapped production work needs no risky unindenting.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

RuntimeTraceMode = Literal["off", "cuda"]


def configure_runtime_metrics(mode: RuntimeTraceMode, root: str | Path | None) -> None:
    """Accept and discard the trace configuration."""


def trace_event(event: str, **fields: object) -> None:
    """Discard a CPU trace event."""


@contextmanager
def trace_cuda_phase(event: str, **fields: object) -> Iterator[None]:
    """Run the wrapped block untimed."""

    yield


def flush_runtime_metrics(*, resolve_cuda: bool = True) -> None:
    """Nothing is buffered, so there is nothing to flush."""
