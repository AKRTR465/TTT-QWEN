from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_svcbench_qwen.episode_data import load_visual_cost_index
from ttt_svcbench_qwen.visual_cost import (
    EpochBoundaryCostEMA,
    make_visual_cost_fingerprint,
)


def _fingerprint() -> dict[str, object]:
    return make_visual_cost_fingerprint(
        manifest_sha256="a" * 64,
        model_revision="qwen@main",
        transformers_version="4.57.1",
        processor="Qwen3VLProcessor",
        minimum_pixels=65_536,
        maximum_pixels=131_072,
        dtype="bfloat16",
        visual_batch_size=4,
        cache_mode="readonly",
        loss_mode="instant_equal",
        loss_group_weight=0.3,
        loss_scale_min=0.1,
        loss_scale_max=10.0,
        loss_epsilon=1.0e-8,
        gpu_model="NVIDIA H200",
    )


def _record() -> dict[str, object]:
    return {
        "record_id": "q1",
        "support_count": 2,
        "segment_lengths": [],
        "query_count": 1,
        "visual_tokens": [32, 48, 16],
        "total_visual_tokens": 96,
        "maximum_visual_tokens": 48,
        "decode_seconds": 0.2,
        "processor_seconds": 0.1,
        "vit_seconds": 0.5,
        "query_seconds": 0.1,
        "loss_collective_seconds": 0.03,
        "predicted_total_seconds": 0.93,
    }


def test_visual_cost_schema2_loads_strict_records_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "visual_cost_index.json"
    fingerprint = _fingerprint()
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fingerprint": fingerprint,
                "records": [_record()],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_visual_cost_index(path, expected_fingerprint=fingerprint)

    assert loaded["q1"].sort_key == (0.93, 96, 48)
    changed = dict(fingerprint, visual_batch_size=2)
    with pytest.raises(ValueError, match="visual_batch_size"):
        load_visual_cost_index(path, expected_fingerprint=changed)


def test_visual_cost_schema2_rejects_bad_or_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "visual_cost_index.json"
    bad = _record()
    bad["total_visual_tokens"] = 95
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fingerprint": _fingerprint(),
                "records": [bad],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="total tokens"):
        load_visual_cost_index(path)


def test_runtime_cost_ema_only_changes_at_epoch_boundary() -> None:
    ema = EpochBoundaryCostEMA({"record": 1.0})

    ema.observe("record", 3.0)
    assert ema.value("record", 0.0) == 1.0
    ema.advance_epoch(1)
    assert ema.value("record", 0.0) == pytest.approx(1.4)
    ema.observe("record", 5.0)
    ema.advance_epoch(1)
    assert ema.value("record", 0.0) == pytest.approx(1.4)
    ema.advance_epoch(2)
    assert ema.value("record", 0.0) == pytest.approx(2.12)
