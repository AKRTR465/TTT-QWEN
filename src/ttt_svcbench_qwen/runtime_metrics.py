"""Low-overhead process-local runtime tracing for H200 training.

CPU events are buffered as JSON objects. CUDA phases retain event pairs and resolve them only
when the run flushes, so tracing never inserts a per-phase synchronization into the hot path.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch

RuntimeTraceMode = Literal["off", "cuda"]


class RuntimeMetricsWriter:
    """One buffered trace stream per rank/worker process."""

    def __init__(self, mode: RuntimeTraceMode, root: Path | None) -> None:
        self.mode = mode
        self.root = root
        self._pid = os.getpid()
        self._records: list[dict[str, object]] = []
        self._cuda_records: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event, dict[str, object]]
        ] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and self.root is not None

    def emit(self, event: str, **fields: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._records.append(self._payload(event, fields))
            if len(self._records) >= 64:
                self._flush_records()

    @contextmanager
    def cuda_phase(self, event: str, **fields: object) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            started = time.perf_counter()
            try:
                yield
            finally:
                if self.enabled:
                    self.emit(event, seconds=time.perf_counter() - started, clock="cpu")
            return
        started_event = torch.cuda.Event(enable_timing=True)
        ended_event = torch.cuda.Event(enable_timing=True)
        started_event.record()
        try:
            yield
        finally:
            ended_event.record()
            with self._lock:
                self._cuda_records.append((event, started_event, ended_event, dict(fields)))

    def flush(self, *, resolve_cuda: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if resolve_cuda and self._cuda_records:
                torch.cuda.synchronize()
                for event, started, ended, fields in self._cuda_records:
                    self._records.append(
                        self._payload(
                            event,
                            {
                                **fields,
                                "seconds": float(started.elapsed_time(ended)) / 1000.0,
                                "clock": "cuda_event",
                            },
                        )
                    )
                self._cuda_records.clear()
            self._flush_records()

    def _payload(self, event: str, fields: dict[str, object]) -> dict[str, object]:
        return {
            "monotonic_seconds": time.monotonic(),
            "pid": self._pid,
            "rank": int(os.environ.get("RANK", "0")),
            "worker": _worker_id(),
            "event": event,
            **fields,
        }

    def _flush_records(self) -> None:
        if not self._records or self.root is None:
            return
        rank = os.environ.get("RANK", "0")
        directory = self.root / f"rank_{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"runtime_{self._pid}_{_worker_id()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.writelines(
                json.dumps(_materialize_trace_tree(row), ensure_ascii=False, sort_keys=True) + "\n"
                for row in self._records
            )
        self._records.clear()


_WRITER: RuntimeMetricsWriter | None = None
_CONFIG_LOCK = threading.Lock()


def configure_runtime_metrics(mode: RuntimeTraceMode, root: str | Path | None) -> None:
    """Configure this process and future DataLoader workers from strict runtime config."""

    if mode not in ("off", "cuda"):
        raise ValueError("runtime trace mode must be off or cuda")
    if mode == "cuda" and root is None:
        raise ValueError("cuda runtime tracing requires a trace directory")
    os.environ["TTT_RUNTIME_TRACE_MODE"] = mode
    if root is None:
        os.environ.pop("TTT_RUNTIME_TRACE_DIR", None)
    else:
        os.environ["TTT_RUNTIME_TRACE_DIR"] = str(Path(root).expanduser().resolve())
    global _WRITER
    with _CONFIG_LOCK:
        if _WRITER is not None:
            _WRITER.flush(resolve_cuda=True)
        _WRITER = RuntimeMetricsWriter(
            mode,
            None if root is None else Path(root).expanduser().resolve(),
        )


def trace_event(event: str, **fields: object) -> None:
    _writer().emit(event, **fields)


@contextmanager
def trace_cuda_phase(event: str, **fields: object) -> Iterator[None]:
    with _writer().cuda_phase(event, **fields):
        yield


def flush_runtime_metrics(*, resolve_cuda: bool = True) -> None:
    _writer().flush(resolve_cuda=resolve_cuda)


def _writer() -> RuntimeMetricsWriter:
    global _WRITER
    pid = os.getpid()
    if _WRITER is None or _WRITER._pid != pid:
        mode_raw = os.environ.get("TTT_RUNTIME_TRACE_MODE", "off")
        mode: RuntimeTraceMode = "cuda" if mode_raw == "cuda" else "off"
        root_raw = os.environ.get("TTT_RUNTIME_TRACE_DIR")
        _WRITER = RuntimeMetricsWriter(mode, None if root_raw is None else Path(root_raw))
    return _WRITER


def _worker_id() -> str:
    try:
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
    except RuntimeError:
        worker = None
    return "main" if worker is None else str(worker.id)


def _materialize_trace_tree(value: object) -> object:
    """Move buffered scalar tensors to host only when the trace is flushed."""

    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.requires_grad:
            raise ValueError("runtime trace tensors must be detached scalars")
        return value.detach().cpu().item()
    if isinstance(value, Mapping):
        return {str(key): _materialize_trace_tree(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_materialize_trace_tree(item) for item in value]
    return value


atexit.register(flush_runtime_metrics)


__all__ = [
    "RuntimeMetricsWriter",
    "RuntimeTraceMode",
    "configure_runtime_metrics",
    "flush_runtime_metrics",
    "trace_cuda_phase",
    "trace_event",
]
