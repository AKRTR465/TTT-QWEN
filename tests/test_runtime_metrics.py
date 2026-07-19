from __future__ import annotations

import json
from pathlib import Path

from ttt_svcbench_qwen.runtime_metrics import RuntimeMetricsWriter


def test_runtime_metrics_off_is_a_true_noop(tmp_path: Path) -> None:
    writer = RuntimeMetricsWriter("off", tmp_path)

    writer.emit("ignored", seconds=1.0)
    writer.flush(resolve_cuda=True)

    assert not tuple(tmp_path.rglob("*.jsonl"))


def test_runtime_metrics_buffers_process_local_jsonl(tmp_path: Path) -> None:
    writer = RuntimeMetricsWriter("cuda", tmp_path)
    writer.emit("video_decode", seconds=0.25, record_id="r0")

    assert not tuple(tmp_path.rglob("*.jsonl"))
    writer.flush()

    paths = tuple(tmp_path.rglob("runtime_*.jsonl"))
    assert len(paths) == 1
    row = json.loads(paths[0].read_text(encoding="utf-8"))
    assert row["event"] == "video_decode"
    assert row["seconds"] == 0.25
    assert row["record_id"] == "r0"
