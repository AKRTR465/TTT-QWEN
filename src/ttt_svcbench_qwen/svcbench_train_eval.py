"""Build the exact 3,706-row SVCBench train-set view used for convergence evaluation.

The Qwen3-VL SFT conversion has its own global ``q_id`` namespace while the production
manifest uses annotation-local Query IDs.  This module resolves the two identities with the
same rule used by ``prepare_svcbench_episodes.py`` and writes an isolated LLaMA-Factory
dataset without changing row order, prompts, answers, or video paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ttt_svcbench_qwen.data import (
    DatasetPurpose,
    DatasetSource,
    LoadedAnnotations,
    SVCBenchRecord,
    load_annotations,
)
from ttt_svcbench_qwen.episode_data import (
    EpisodeSplit,
    load_production_episode_manifest,
)

EXPECTED_SFT_ROW_COUNT = 4_576
EXPECTED_TRAIN_ROW_COUNT = 3_706
DEFAULT_SOURCE_DATASET_NAME = "svcbench_qwen3vl_sft"
DEFAULT_EVAL_DATASET_NAME = "svcbench_train3706"


@dataclass(frozen=True, slots=True)
class TrainEvalSelection:
    selection_index: int
    sft_index: int
    q_id: str
    manifest_query_id: str
    label: str
    manifest_label: str
    counting_subtype: str

    def __post_init__(self) -> None:
        if self.selection_index < 0 or self.sft_index < 0:
            raise ValueError("evaluation selection indices must be non-negative")
        if (
            not self.q_id
            or not self.manifest_query_id
            or not self.label
            or not self.manifest_label
        ):
            raise ValueError("evaluation selection identity and label must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedTrainEvalDataset:
    output_dir: Path
    sft_data: Path
    dataset_info: Path
    selection: Path
    summary: Path
    dataset_name: str
    row_count: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sft_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    raw = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("SVCBench SFT data must be one JSON array of objects")
    return cast(list[dict[str, Any]], raw)


def read_selection(path: str | Path) -> tuple[TrainEvalSelection, ...]:
    rows: list[TrainEvalSelection] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = cast(object, json.loads(line))
            if not isinstance(raw, dict):
                raise ValueError(f"selection row {line_number} must be an object")
            rows.append(
                TrainEvalSelection(
                    selection_index=int(raw["selection_index"]),
                    sft_index=int(raw["sft_index"]),
                    q_id=str(raw["q_id"]),
                    manifest_query_id=str(raw["manifest_query_id"]),
                    label=str(raw["label"]),
                    manifest_label=str(raw.get("manifest_label", raw["label"])),
                    counting_subtype=str(raw["counting_subtype"]),
                )
            )
    if tuple(row.selection_index for row in rows) != tuple(range(len(rows))):
        raise ValueError("evaluation selection rows must be contiguous and ordered")
    if len({row.q_id for row in rows}) != len(rows):
        raise ValueError("evaluation selection contains duplicate baseline q_id values")
    if len({row.manifest_query_id for row in rows}) != len(rows):
        raise ValueError("evaluation selection contains duplicate manifest Query IDs")
    return tuple(rows)


def resolve_sft_annotation_rows(
    annotations: LoadedAnnotations,
    sft_rows: Sequence[Mapping[str, object]],
) -> tuple[SVCBenchRecord, ...]:
    """Resolve every converted SFT row to one raw annotation Query."""

    annotations_by_video: dict[tuple[str, str], list[SVCBenchRecord]] = defaultdict(list)
    for record in annotations.records:
        annotations_by_video[(record.source_dataset, record.relative_video_path)].append(record)

    resolved: list[SVCBenchRecord] = []
    seen_query_ids: set[str] = set()
    for index, item in enumerate(sft_rows):
        source = _required_string(item, "source_dataset", index)
        source_path = _required_string(item, "source_video_path", index)
        question = _required_string(item, "question", index)
        query_time = item.get("query_time")
        if (
            isinstance(query_time, bool)
            or not isinstance(query_time, (int, float))
            or not math.isfinite(float(query_time))
            or float(query_time) < 0.0
        ):
            raise ValueError(f"SFT row {index} has an invalid query_time")
        query_index = item.get("query_index")
        if query_index is None:
            subtype = _required_string(item, "counting_subtype", index)
            candidates = [
                record
                for record in annotations_by_video[(source, source_path)]
                if record.labels.counting_subtype.casefold() == subtype.casefold()
            ]
        else:
            if (
                isinstance(query_index, bool)
                or not isinstance(query_index, int)
                or query_index < 0
            ):
                raise ValueError(f"SFT row {index} has an invalid query_index")
            candidates = [
                record
                for record in annotations_by_video[(source, source_path)]
                if record.identity.query_index == query_index and record.question == question
            ]
        if not candidates:
            raise ValueError(
                f"SFT row {index} has no annotation candidate for "
                f"{source}/{source_path} query_index={query_index!r}"
            )
        nearest_distance = min(
            abs(record.query_time - float(query_time)) for record in candidates
        )
        nearest = tuple(
            record
            for record in candidates
            if abs(abs(record.query_time - float(query_time)) - nearest_distance) < 1.0e-9
        )
        if len(nearest) != 1 or nearest_distance > 1.001:
            raise ValueError(
                f"SFT row {index} cannot be resolved uniquely by query_time "
                f"(distance={nearest_distance}, matches={len(nearest)})"
            )
        record = nearest[0]
        if record.identity.query_id in seen_query_ids:
            raise ValueError(
                f"SFT conversion maps multiple rows to Query {record.identity.query_id}"
            )
        seen_query_ids.add(record.identity.query_id)
        resolved.append(record)
    return tuple(resolved)


def build_train_selection(
    *,
    sft_rows: Sequence[Mapping[str, object]],
    annotations: LoadedAnnotations,
    manifest_path: str | Path,
    expected_sft_rows: int = EXPECTED_SFT_ROW_COUNT,
    expected_train_rows: int = EXPECTED_TRAIN_ROW_COUNT,
) -> tuple[tuple[TrainEvalSelection, ...], tuple[dict[str, Any], ...]]:
    manifest = load_production_episode_manifest(manifest_path)
    if manifest.annotation_sha256 != annotations.annotation_sha256:
        raise ValueError("production manifest and raw annotation SHA256 values differ")
    if len(sft_rows) != expected_sft_rows:
        raise ValueError(
            f"expected {expected_sft_rows} baseline SFT rows, found {len(sft_rows)}"
        )
    resolved = resolve_sft_annotation_rows(annotations, sft_rows)
    train_by_id = {
        row.query.runtime.query_id: row
        for row in manifest.a2_queries
        if row.split is EpisodeSplit.TRAIN
    }
    if len(train_by_id) != expected_train_rows:
        raise ValueError(
            f"expected {expected_train_rows} manifest train rows, found {len(train_by_id)}"
        )

    selected_rows: list[dict[str, Any]] = []
    selection: list[TrainEvalSelection] = []
    for sft_index, (item, annotation) in enumerate(zip(sft_rows, resolved, strict=True)):
        manifest_row = train_by_id.get(annotation.identity.query_id)
        if manifest_row is None:
            continue
        q_id = _required_string(item, "q_id", sft_index)
        label = _assistant_label(item, sft_index)
        manifest_label = str(manifest_row.query.weak.count)
        videos = item.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], str)
        ):
            raise ValueError(f"SFT row {sft_index} must contain exactly one video")
        video = PurePosixPath(videos[0])
        if video.is_absolute() or ".." in video.parts:
            raise ValueError(f"SFT row {sft_index} contains an unsafe video path")
        selected_rows.append(dict(item))
        selection.append(
            TrainEvalSelection(
                selection_index=len(selection),
                sft_index=sft_index,
                q_id=q_id,
                manifest_query_id=annotation.identity.query_id,
                # The scorer and the Qwen3-VL baseline both use the converted SFT
                # answer as ground truth.  Preserve it even when the production
                # manifest differs at a clip boundary; keep the manifest value
                # alongside it so the discrepancy remains explicit and auditable.
                label=label.strip(),
                manifest_label=manifest_label,
                counting_subtype=manifest_row.query.weak.counting_subtype,
            )
        )

    missing = tuple(sorted(set(train_by_id) - {row.manifest_query_id for row in selection}))
    if len(selection) != expected_train_rows or missing:
        raise ValueError(
            f"train selection is incomplete: selected={len(selection)} missing={len(missing)}"
        )
    if len({row.q_id for row in selection}) != len(selection):
        raise ValueError("selected baseline q_id values are not unique")
    return tuple(selection), tuple(selected_rows)


def prepare_train_eval_dataset(
    *,
    sft_data: str | Path,
    raw_annotations: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    source_dataset_name: str = DEFAULT_SOURCE_DATASET_NAME,
    output_dataset_name: str = DEFAULT_EVAL_DATASET_NAME,
    expected_sft_sha256: str | None = None,
) -> PreparedTrainEvalDataset:
    source_sft = Path(sft_data).resolve()
    raw_path = Path(raw_annotations).resolve()
    manifest_source = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation dataset: {destination}")
    for source in (source_sft, raw_path, manifest_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    source_hash = sha256_file(source_sft)
    if expected_sft_sha256 is not None and source_hash != expected_sft_sha256:
        raise ValueError(
            f"baseline SFT SHA256 drift: expected {expected_sft_sha256}, found {source_hash}"
        )

    source_info_path = source_sft.parent / "dataset_info.json"
    if not source_info_path.is_file():
        raise FileNotFoundError(source_info_path)
    info_raw = cast(object, json.loads(source_info_path.read_text(encoding="utf-8")))
    if not isinstance(info_raw, dict) or not isinstance(info_raw.get(source_dataset_name), dict):
        raise ValueError(f"dataset_info.json has no {source_dataset_name!r} entry")
    dataset_entry = dict(cast(dict[str, object], info_raw[source_dataset_name]))

    manifest = load_production_episode_manifest(manifest_source)
    annotations = load_annotations(
        raw_path,
        source=DatasetSource(manifest.dataset_name, manifest.dataset_revision, False),
        purpose=DatasetPurpose.TRAINING,
    )
    sft_rows = read_sft_rows(source_sft)
    selection, selected_rows = build_train_selection(
        sft_rows=sft_rows,
        annotations=annotations,
        manifest_path=manifest_source,
    )

    videos_source = source_sft.parent / "videos"
    if not videos_source.is_dir():
        raise FileNotFoundError(videos_source)
    missing_videos = tuple(
        str(row["videos"][0])
        for row in selected_rows
        if not (source_sft.parent / str(row["videos"][0])).is_file()
    )
    if missing_videos:
        raise FileNotFoundError(
            f"{len(missing_videos)} selected baseline clips are missing; first={missing_videos[0]}"
        )

    destination.mkdir(parents=True)
    output_sft = destination / f"{output_dataset_name}.json"
    output_info = destination / "dataset_info.json"
    output_selection = destination / "selection.jsonl"
    output_summary = destination / "selection_summary.json"
    _write_json(output_sft, selected_rows)
    dataset_entry["file_name"] = output_sft.name
    _write_json(output_info, {output_dataset_name: dataset_entry})
    os.symlink(videos_source, destination / "videos", target_is_directory=True)
    _write_jsonl(output_selection, (asdict(row) for row in selection))
    _write_json(
        output_summary,
        {
            "schema_version": "svcbench_train_eval_v1",
            "evaluation_scope": "train_set",
            "dataset_name": output_dataset_name,
            "row_count": len(selection),
            "source_sft_row_count": len(sft_rows),
            "source_sft_sha256": source_hash,
            "raw_annotation_sha256": annotations.annotation_sha256,
            "manifest_path": str(manifest_source),
            "manifest_sha256": sha256_file(manifest_source),
            "split": EpisodeSplit.TRAIN.value,
            "validation_rows_excluded": sum(
                row.split is EpisodeSplit.VALIDATION for row in manifest.a2_queries
            ),
            "failed_manifest_queries_excluded": len(manifest.failures),
            "sft_manifest_label_mismatch_count": sum(
                row.label != row.manifest_label for row in selection
            ),
            "sft_manifest_label_mismatches": [
                {
                    "selection_index": row.selection_index,
                    "sft_index": row.sft_index,
                    "q_id": row.q_id,
                    "manifest_query_id": row.manifest_query_id,
                    "sft_label": row.label,
                    "manifest_label": row.manifest_label,
                }
                for row in selection
                if row.label != row.manifest_label
            ],
            "ordering": "original_sft_order",
        },
    )
    return PreparedTrainEvalDataset(
        output_dir=destination,
        sft_data=output_sft,
        dataset_info=output_info,
        selection=output_selection,
        summary=output_summary,
        dataset_name=output_dataset_name,
        row_count=len(selection),
    )


def _required_string(item: Mapping[str, object], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SFT row {index} requires non-empty {key}")
    return value


def _assistant_label(item: Mapping[str, object], index: int) -> str:
    messages = item.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
            ):
                return cast(str, message["content"])
    answer = item.get("answer")
    if isinstance(answer, (str, int)) and not isinstance(answer, bool):
        return str(answer)
    raise ValueError(f"SFT row {index} has no assistant label")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
