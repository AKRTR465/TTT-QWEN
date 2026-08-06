"""No-op runtime-tracing shim.

The mainline runs with ``runtime_trace_mode: "off"`` in every h200 config, so tracing never
produced output on a production run. The buffered JSONL writer and CUDA event-pair machinery
were removed; this shim keeps the four call shapes so the ~31 ``trace_event`` /
``with trace_cuda_phase(...)`` sites need no edit and no unindenting.
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
