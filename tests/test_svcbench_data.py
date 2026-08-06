from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_svcbench_qwen.data import (
    RUNTIME_ALLOWLIST,
    RUNTIME_DENYLIST,
    DatasetSource,
    LoadedAnnotations,
    RuntimeQueryInput,
    assert_runtime_payload_safe,
    create_group_kfold_manifest,
    extract_explicit_time_values,
    load_annotations,
)
from ttt_svcbench_qwen.inference import assert_inference_runtime_payload

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "svcbench"
SOURCE = DatasetSource(name="buaaplay/SVCBench", revision="4c9bd87")


def load_fixture(name: str) -> LoadedAnnotations:
    return load_annotations(FIXTURES / name, source=SOURCE)


def test_grouped_and_flat_schemas_parse_to_query_point_records() -> None:
    grouped = load_fixture("grouped.jsonl")
    flat = load_fixture("flat.jsonl")

    assert len(grouped.records) == 3
    assert grouped.records[0].identity.query_id == "0000:0"
    assert grouped.records[1].identity.video_id == grouped.records[0].identity.video_id
    assert grouped.records[1].labels.occurrence_times.points == (1.0, 2.0)
    assert len(flat.records) == 2
    assert flat.records[1].identity.query_id == "0001"
    assert flat.records[1].labels.count == 4


def test_annotation_projection_and_inference_keep_labels_out_of_runtime_payloads() -> None:
    record = load_fixture("grouped.jsonl").records[0]
    runtime = RuntimeQueryInput(
        video_id=record.identity.video_id,
        trajectory_id=record.identity.trajectory_id,
        query_id=record.identity.query_id,
        query_index=record.identity.query_index,
        video=FIXTURES / "videos" / record.source_dataset / record.relative_video_path,
        question=record.question,
        query_time=record.query_time,
        explicit_time_values=extract_explicit_time_values(record.question),
    )
    payload = runtime.as_payload()

    assert set(payload) == RUNTIME_ALLOWLIST
    assert not (set(payload) & RUNTIME_DENYLIST)
    assert record.labels.count == 2
    assert_inference_runtime_payload(payload)

    for denied_field in sorted(RUNTIME_DENYLIST):
        poisoned = {**payload, denied_field: "forbidden"}
        with pytest.raises(ValueError, match="denied fields"):
            assert_runtime_payload_safe(poisoned, layer="JSON")
        with pytest.raises(ValueError, match="denied fields"):
            assert_inference_runtime_payload(poisoned)

    with pytest.raises(ValueError, match="non-allowlisted fields"):
        assert_runtime_payload_safe({**payload, "surprise": 1}, layer="JSON")


def test_explicit_time_parser_accepts_only_question_visible_values() -> None:
    assert extract_explicit_time_values("What happened in the last 5 seconds?") == (5.0,)
    assert extract_explicit_time_values("Count events in the last 2 minutes and 3 seconds") == (
        120.0,
        3.0,
    )
    assert extract_explicit_time_values("过去 10 秒内发生了几次？") == (10.0,)
    assert extract_explicit_time_values("from 2 to 8 seconds") == (2.0, 8.0)
    assert extract_explicit_time_values("从 2 到 8 秒") == (2.0, 8.0)
    assert extract_explicit_time_values("从2到8秒") == (2.0, 8.0)
    assert extract_explicit_time_values("How many are visible now?") == ()


def test_group_kfold_keeps_every_video_in_exactly_one_validation_split(tmp_path: Path) -> None:
    row = {
        "source_dataset": "synthetic",
        "question": "How many objects are visible?",
        "counting_type": "O1",
        "counting_subtype": "O1-Snap",
        "query_points": {"time": [1.0, 2.0], "count": [1, 2]},
    }
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(
            json.dumps({**row, "id": f"q-{i}", "video_path": f"video-{i}.mp4"}) + "\n"
            for i in range(6)
        ),
        encoding="utf-8",
    )
    annotations = load_annotations(path, source=DatasetSource("synthetic", "fixture-v1"))

    manifest = create_group_kfold_manifest(annotations, n_splits=3, seed=42)
    validation: list[str] = []
    for fold in manifest.folds:
        assert not (set(fold.train_video_ids) & set(fold.validation_video_ids))
        validation.extend(fold.validation_video_ids)

    assert len(manifest.folds) == 3
    assert sorted(validation) == sorted({r.identity.video_id for r in annotations.records})
