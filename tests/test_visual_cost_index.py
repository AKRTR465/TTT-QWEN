from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.build_visual_cost_index import _load_runtime_measurements

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
        loss_mode="ema_answer_ref",
        loss_group_weight=0.3,
        loss_scale_min=0.001,
        loss_scale_max=20.0,
        loss_epsilon=1.0e-8,
        gpu_model="NVIDIA H200",
        query_decode_strategy="grouped_seek",
        query_decode_max_groups=16,
        state_query_visual_mode="recent_chunk",
        state_query_max_frames=16,
        answer_query_visual_mode="causal_prefix",
        answer_query_max_frames=256,
        query_sample_fps=2.0,
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
        "query_frame_count": 16,
        "query_visual_tokens": 16,
        "source_codec": "h264",
        "source_width": 640,
        "source_height": 360,
        "keyframe_interval_seconds": 2.0,
        "support_cache_bytes": 4096,
        "decode_seconds": 0.2,
        "processor_seconds": 0.1,
        "preparation_seconds": 0.3,
        "training_seconds": 0.63,
        "vit_seconds": 0.5,
        "query_seconds": 0.1,
        "loss_collective_seconds": 0.03,
        "predicted_total_seconds": 0.93,
        "measurement_source": "runtime_trace",
    }


def test_visual_cost_schema4_loads_measured_records_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "visual_cost_index.json"
    fingerprint = _fingerprint()
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "fingerprint": fingerprint,
                "records": [_record()],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_visual_cost_index(
        path,
        expected_fingerprint=fingerprint,
        require_runtime_measurements=True,
    )

    assert loaded["q1"].sort_key == (0.93, loaded["q1"].history_write_units, 96, 48)
    changed = dict(fingerprint, visual_batch_size=2)
    with pytest.raises(ValueError, match="visual_batch_size"):
        load_visual_cost_index(path, expected_fingerprint=changed)


def test_visual_cost_schema4_rejects_bad_or_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "visual_cost_index.json"
    bad = _record()
    bad["total_visual_tokens"] = 95
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "fingerprint": _fingerprint(),
                "records": [bad],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="total tokens"):
        load_visual_cost_index(path)


def test_visual_cost_schema4_rejects_legacy_schema_and_estimates_in_runtime_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "visual_cost_index.json"
    row = _record()
    row["measurement_source"] = "estimated"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "fingerprint": _fingerprint(),
                "records": [row],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lacks runtime measurement"):
        load_visual_cost_index(path, require_runtime_measurements=True)

    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 3
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be 4"):
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


def test_schema4_runtime_trace_extracts_end_to_end_record_cost(tmp_path: Path) -> None:
    trace = tmp_path / "runtime_rank0.jsonl"
    rows = (
        {
            "event": "a2_collate_done",
            "query_id": "q1",
            "query_frame_count": 256,
            "query_visual_token_count": 2048,
            "state_query_frame_count": 16,
            "state_query_visual_token_count": 128,
            "query_decode_seconds": 4.0,
            "query_processor_seconds": 2.0,
            "seconds": 9.0,
        },
        {
            "event": "support_cache_read",
            "record_id": "q1",
            "chunk_id": "q1:a2:0",
            "cache_bytes": 100,
        },
        {
            "event": "support_cache_read",
            "record_id": "q1",
            "chunk_id": "q1:a2:1",
            "cache_bytes": 200,
        },
        {
            "event": "runtime_cost_observation",
            "record_id": "q1",
            "training_seconds": 3.0,
        },
    )
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    measured = _load_runtime_measurements(trace)["q1"]

    assert measured.query_frame_count == 272
    assert measured.query_visual_tokens == 2176
    assert measured.decode_seconds == 4.0
    assert measured.processor_seconds == 2.0
    assert measured.preparation_seconds == 9.0
    assert measured.training_seconds == 3.0
    assert measured.support_cache_bytes == 300
