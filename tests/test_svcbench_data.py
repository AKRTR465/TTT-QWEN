from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_svcbench_qwen.data import (
    RUNTIME_ALLOWLIST,
    RUNTIME_DENYLIST,
    DatasetPurpose,
    DatasetSource,
    LoadedAnnotations,
    RuntimeQueryInput,
    assert_runtime_payload_safe,
    create_group_kfold_manifest,
    extract_explicit_time_values,
    load_annotations,
    write_fold_manifest,
)
from ttt_svcbench_qwen.inference import assert_inference_runtime_payload

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "svcbench"
OFFICIAL_SOURCE = DatasetSource(
    name="buaaplay/SVCBench",
    revision="4c9bd87ef3b0f269ca8b4503c081f5f38bc1fc9a",
    official_clean=True,
)


def load_fixture(name: str, *, official_clean: bool = True) -> LoadedAnnotations:
    source = (
        OFFICIAL_SOURCE
        if official_clean
        else DatasetSource(name="synthetic-training", revision="fixture-v1", official_clean=False)
    )
    purpose = (
        DatasetPurpose.OFFICIAL_CLEAN_EVALUATION if official_clean else DatasetPurpose.TRAINING
    )
    return load_annotations(FIXTURES / name, source=source, purpose=purpose)


def test_grouped_and_flat_official_schemas_parse_to_query_point_records() -> None:
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
    annotations = load_fixture("grouped.jsonl")
    record = annotations.records[0]
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
    assert runtime.query_id == record.identity.query_id
    assert_inference_runtime_payload(payload)
    assert annotations.records[0].labels.count == 2

    for denied_field in sorted(RUNTIME_DENYLIST):
        poisoned = {**payload, denied_field: "forbidden"}
        with pytest.raises(ValueError, match="denied fields"):
            assert_runtime_payload_safe(poisoned, layer="JSON")
        with pytest.raises(ValueError, match="denied fields"):
            assert_inference_runtime_payload(poisoned)


def test_official_clean_annotations_cannot_be_repurposed_for_selection() -> None:
    for purpose in (DatasetPurpose.TRAINING, DatasetPurpose.CALIBRATION):
        with pytest.raises(ValueError, match="official clean"):
            load_annotations(FIXTURES / "grouped.jsonl", source=OFFICIAL_SOURCE, purpose=purpose)

    annotations = load_fixture("grouped.jsonl")
    with pytest.raises(PermissionError, match="official clean"):
        create_group_kfold_manifest(annotations, n_splits=2, seed=42)


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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_runtime_time_fields_reject_non_finite_and_negative_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RuntimeQueryInput(
            "video-0",
            "trajectory-0",
            "query-0",
            0,
            Path("synthetic.mp4"),
            "question",
            bad_value,
            (),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        RuntimeQueryInput(
            "video-0",
            "trajectory-0",
            "query-0",
            0,
            Path("synthetic.mp4"),
            "question",
            1.0,
            (bad_value,),
        )

def test_group_kfold_keeps_all_query_points_from_each_video_together(tmp_path: Path) -> None:
    rows = []
    for video_index in range(6):
        rows.append(
            {
                "id": f"question-{video_index}",
                "source_dataset": "synthetic",
                "video_path": f"video-{video_index}.mp4",
                "question": "How many objects are visible?",
                "counting_type": "O1",
                "counting_subtype": "O1-Snap",
                "occurrence_times": [1.0, 2.0],
                "query_points": {"time": [1.0, 2.0], "count": [1, 2]},
            }
        )
    annotation_path = tmp_path / "train.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    annotations = load_annotations(
        annotation_path,
        source=DatasetSource("synthetic", "fixture-v1", False),
        purpose=DatasetPurpose.TRAINING,
    )
    manifest = create_group_kfold_manifest(annotations, n_splits=3, seed=42)
    manifest_path = tmp_path / "fold-manifest.json"
    write_fold_manifest(manifest, manifest_path)

    assert len(manifest.folds) == 3
    validation_queries = []
    for fold in manifest.folds:
        assert not (set(fold.train_video_ids) & set(fold.validation_video_ids))
        validation_queries.extend(fold.validation_query_ids)
        for record in annotations.records:
            in_validation = record.identity.video_id in fold.validation_video_ids
            assert (record.identity.query_id in fold.validation_query_ids) is in_validation
    assert len(validation_queries) == len(set(validation_queries)) == len(annotations.records)
    manifest_path.read_bytes().decode("utf-8", errors="strict")


def test_grouped_schema_rejects_misaligned_multi_query_points(tmp_path: Path) -> None:
    bad = {
        "id": "bad",
        "source_dataset": "synthetic",
        "video_path": "bad.mp4",
        "question": "How many?",
        "counting_type": "O1",
        "counting_subtype": "O1-Snap",
        "occurrence_times": [],
        "query_points": {"time": [1.0, 2.0], "count": [1]},
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must align"):
        load_annotations(
            path,
            source=DatasetSource("synthetic", "fixture-v1", False),
            purpose=DatasetPurpose.TRAINING,
        )
