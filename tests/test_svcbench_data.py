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
    SVCBenchCollator,
    SVCBenchDataset,
    assert_runtime_payload_safe,
    create_group_kfold_manifest,
    extract_explicit_time_values,
    load_annotations,
    write_fold_manifest,
)
from ttt_svcbench_qwen.inference import assert_inference_runtime_payload
from ttt_svcbench_qwen.trainer import assert_trainer_runtime_payload

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


def test_dataset_collator_trainer_and_inference_keep_labels_out_of_runtime_payloads() -> None:
    dataset = SVCBenchDataset(load_fixture("grouped.jsonl"), FIXTURES / "videos")
    sample = dataset[0]
    batch = SVCBenchCollator()([dataset[0], dataset[1]])
    payloads = (sample.model_input.as_payload(), batch.as_model_payload())

    assert set(payloads[0]) == RUNTIME_ALLOWLIST
    assert set(payloads[1]) == RUNTIME_ALLOWLIST
    assert not (set(payloads[0]) & RUNTIME_DENYLIST)
    assert_trainer_runtime_payload(payloads[1])
    assert_inference_runtime_payload(payloads[1])
    assert dataset.supervision_for(0, consumer="evaluator").count == 2
    with pytest.raises(PermissionError, match="training dataset"):
        dataset.supervision_for(0, consumer="trainer")

    for denied_field in sorted(RUNTIME_DENYLIST):
        poisoned = {**payloads[0], denied_field: "forbidden"}
        for layer in ("Dataset", "Collator"):
            with pytest.raises(ValueError, match="denied fields"):
                assert_runtime_payload_safe(poisoned, layer=layer)
        with pytest.raises(ValueError, match="denied fields"):
            assert_trainer_runtime_payload(poisoned)
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
    assert extract_explicit_time_values("How many are visible now?") == ()


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
