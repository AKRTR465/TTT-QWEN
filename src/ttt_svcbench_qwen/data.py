"""Load SVCBench annotations and enforce runtime data-leakage boundaries.

Inputs: UTF-8 official-style JSONL annotations, a video root, and an explicit dataset purpose.
Outputs: label-free runtime samples/batches, separate supervision, and group-safe fold manifests.
Forbidden: labels in model payloads, clean-test selection, cross-video folds, or absolute data
paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import torch
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]
from torch import Tensor

RUNTIME_ALLOWLIST = frozenset({"video", "question", "query_time", "explicit_time_values"})
RUNTIME_DENYLIST = frozenset(
    {"answer", "count", "occurrence_times", "counting_type", "counting_subtype"}
)

_EXPLICIT_TIME_PATTERN = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|秒|分钟)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_SHARED_UNIT_RANGE_PATTERNS = (
    re.compile(
        r"\b(?:from|between)\s+(?P<start>\d+(?:\.\d+)?)\s+(?:to|and)\s+"
        r"(?P<end>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m)"
        r"(?![A-Za-z])",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"从\s*(?P<start>\d+(?:\.\d+)?)\s*(?:到|至)\s*"
        r"(?P<end>\d+(?:\.\d+)?)\s*(?P<unit>秒|分钟)",
        flags=re.IGNORECASE,
    ),
)


class DatasetPurpose(StrEnum):
    TRAINING = "training"
    CALIBRATION = "calibration"
    OFFICIAL_CLEAN_EVALUATION = "official_clean_evaluation"


class AnnotationFormat(StrEnum):
    GROUPED = "grouped"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    revision: str
    official_clean: bool

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("dataset source name and revision must be non-empty")


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    query_id: str
    query_index: int
    video_id: str
    trajectory_id: str

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id or not self.trajectory_id:
            raise ValueError("sample identity fields must be non-empty")
        if self.query_index < 0:
            raise ValueError("query_index must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeModelInput:
    video: Path
    question: str
    query_time: float
    explicit_time_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("question must be non-empty")
        if not math.isfinite(self.query_time) or self.query_time < 0.0:
            raise ValueError("query_time must be finite and non-negative")
        if any(not math.isfinite(value) or value < 0.0 for value in self.explicit_time_values):
            raise ValueError("explicit time values must be finite and non-negative")
        assert_runtime_payload_safe(self.as_payload(), layer="Dataset")

    def as_payload(self) -> dict[str, object]:
        return {
            "video": self.video,
            "question": self.question,
            "query_time": self.query_time,
            "explicit_time_values": self.explicit_time_values,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSample:
    identity: SampleIdentity
    model_input: RuntimeModelInput


@dataclass(frozen=True, slots=True)
class OccurrenceAnnotations:
    points: tuple[float, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in (*self.points, *self.starts, *self.ends)):
            raise ValueError("occurrence_times must be non-negative")
        if len(self.starts) != len(self.ends):
            raise ValueError("occurrence_times start/end arrays must align")
        if any(end < start for start, end in zip(self.starts, self.ends, strict=True)):
            raise ValueError("occurrence_times intervals must have end >= start")


@dataclass(frozen=True, slots=True)
class SupervisionLabels:
    answer: str | None
    count: int
    occurrence_times: OccurrenceAnnotations
    counting_type: str
    counting_subtype: str

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")
        if not self.counting_type or not self.counting_subtype:
            raise ValueError("counting type and subtype must be non-empty")


@dataclass(frozen=True, slots=True)
class SVCBenchRecord:
    identity: SampleIdentity
    source_dataset: str
    relative_video_path: str
    question: str
    query_time: float
    labels: SupervisionLabels

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.relative_video_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("video_path must be a safe dataset-relative path")
        expected_video_id = canonical_video_id(self.source_dataset, self.relative_video_path)
        if self.identity.video_id != expected_video_id:
            raise ValueError("record video_id must match source_dataset/video_path")
        if not math.isfinite(self.query_time) or self.query_time < 0.0 or not self.question:
            raise ValueError("record query_time/question is invalid")

    def runtime_sample(self, video_root: Path) -> RuntimeSample:
        video_path = video_root / self.source_dataset / Path(self.relative_video_path)
        return RuntimeSample(
            identity=self.identity,
            model_input=RuntimeModelInput(
                video=video_path,
                question=self.question,
                query_time=self.query_time,
                explicit_time_values=extract_explicit_time_values(self.question),
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedAnnotations:
    records: tuple[SVCBenchRecord, ...]
    source: DatasetSource
    purpose: DatasetPurpose
    annotation_format: AnnotationFormat
    annotation_sha256: str
    annotation_path: Path

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("annotation file must contain at least one query point")
        if (
            self.source.official_clean
            and self.purpose is not DatasetPurpose.OFFICIAL_CLEAN_EVALUATION
        ):
            raise ValueError(
                "official clean annotations cannot be used for training or calibration"
            )


class SVCBenchDataset:
    """Map-style dataset whose public samples contain no supervision fields."""

    def __init__(self, annotations: LoadedAnnotations, video_root: str | Path) -> None:
        self._annotations = annotations
        self._video_root = Path(video_root)

    @property
    def annotations(self) -> LoadedAnnotations:
        return self._annotations

    def __len__(self) -> int:
        return len(self._annotations.records)

    def __getitem__(self, index: int) -> RuntimeSample:
        sample = self._annotations.records[index].runtime_sample(self._video_root)
        assert_runtime_payload_safe(sample.model_input.as_payload(), layer="Dataset")
        return sample

    def supervision_for(
        self, index: int, *, consumer: Literal["trainer", "evaluator"]
    ) -> SupervisionLabels:
        if consumer == "trainer" and self._annotations.purpose is not DatasetPurpose.TRAINING:
            raise PermissionError("trainer supervision is available only for a training dataset")
        if (
            consumer == "evaluator"
            and self._annotations.purpose is not DatasetPurpose.OFFICIAL_CLEAN_EVALUATION
        ):
            raise PermissionError(
                "evaluator supervision requires the frozen official evaluation purpose"
            )
        return self._annotations.records[index].labels


@dataclass(frozen=True, slots=True)
class RuntimeBatch:
    video: tuple[Path, ...]
    question: tuple[str, ...]
    query_time: Tensor
    explicit_time_values: tuple[tuple[float, ...], ...]
    identities: tuple[SampleIdentity, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.video)
        if not batch_size or len(self.question) != batch_size or len(self.identities) != batch_size:
            raise ValueError("RuntimeBatch fields must have one entry per batch item")
        if len(self.explicit_time_values) != batch_size:
            raise ValueError("explicit_time_values must have one entry per batch item")
        if any(
            not math.isfinite(value) or value < 0.0
            for values in self.explicit_time_values
            for value in values
        ):
            raise ValueError("explicit_time_values must be finite and non-negative")
        if self.query_time.shape != (batch_size,) or not torch.is_floating_point(self.query_time):
            raise ValueError("query_time must be floating [B]")
        if not bool(torch.isfinite(self.query_time).all()) or bool(torch.any(self.query_time < 0)):
            raise ValueError("query_time must be finite and non-negative")
        assert_runtime_payload_safe(self.as_model_payload(), layer="Collator")

    def as_model_payload(self) -> dict[str, object]:
        return {
            "video": self.video,
            "question": self.question,
            "query_time": self.query_time,
            "explicit_time_values": self.explicit_time_values,
        }


class SVCBenchCollator:
    def __call__(self, samples: Sequence[RuntimeSample]) -> RuntimeBatch:
        if not samples:
            raise ValueError("cannot collate an empty sample sequence")
        batch = RuntimeBatch(
            video=tuple(sample.model_input.video for sample in samples),
            question=tuple(sample.model_input.question for sample in samples),
            query_time=torch.tensor(
                [sample.model_input.query_time for sample in samples], dtype=torch.float32
            ),
            explicit_time_values=tuple(
                sample.model_input.explicit_time_values for sample in samples
            ),
            identities=tuple(sample.identity for sample in samples),
        )
        assert_runtime_payload_safe(batch.as_model_payload(), layer="Collator")
        return batch


@dataclass(frozen=True, slots=True)
class FoldSplit:
    fold_index: int
    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]
    train_video_ids: tuple[str, ...]
    validation_video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if set(self.train_video_ids) & set(self.validation_video_ids):
            raise ValueError("GroupKFold contains cross-split video leakage")


@dataclass(frozen=True, slots=True)
class FoldManifest:
    dataset_name: str
    dataset_revision: str
    annotation_sha256: str
    seed: int
    group_key: str
    folds: tuple[FoldSplit, ...]


def assert_runtime_payload_safe(payload: Mapping[str, object], *, layer: str) -> None:
    keys = frozenset(payload)
    denied = keys & RUNTIME_DENYLIST
    unknown = keys - RUNTIME_ALLOWLIST
    if denied:
        raise ValueError(f"{layer} runtime payload contains denied fields: {sorted(denied)}")
    if unknown:
        raise ValueError(
            f"{layer} runtime payload contains non-allowlisted fields: {sorted(unknown)}"
        )
    missing = RUNTIME_ALLOWLIST - keys
    if missing:
        raise ValueError(f"{layer} runtime payload is missing required fields: {sorted(missing)}")


def extract_explicit_time_values(question: str) -> tuple[float, ...]:
    positioned_values: dict[int, float] = {}
    for match in _EXPLICIT_TIME_PATTERN.finditer(question):
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"minute", "minutes", "min", "mins", "m", "分钟"}:
            value *= 60.0
        positioned_values[match.start(1)] = value
    for pattern in _SHARED_UNIT_RANGE_PATTERNS:
        for match in pattern.finditer(question):
            unit = match.group("unit").lower()
            scale = (
                60.0
                if unit
                in {
                    "minute",
                    "minutes",
                    "min",
                    "mins",
                    "m",
                    "分钟",
                }
                else 1.0
            )
            positioned_values[match.start("start")] = float(match.group("start")) * scale
            positioned_values[match.start("end")] = float(match.group("end")) * scale
    return tuple(value for _, value in sorted(positioned_values.items()))


def canonical_video_id(source_dataset: str, relative_video_path: str) -> str:
    if not source_dataset:
        raise ValueError("source_dataset must be non-empty")
    return f"{source_dataset}/{PurePosixPath(relative_video_path).as_posix()}"


def load_annotations(
    annotation_path: str | Path,
    *,
    source: DatasetSource,
    purpose: DatasetPurpose,
) -> LoadedAnnotations:
    path = Path(annotation_path)
    content = path.read_bytes()
    text = content.decode("utf-8", errors="strict")
    rows = tuple(_parse_json_object(line, line_number) for line_number, line in _jsonl_lines(text))
    if not rows:
        raise ValueError(f"annotation file is empty: {path}")
    annotation_format = (
        AnnotationFormat.GROUPED if "query_points" in rows[0] else AnnotationFormat.FLAT
    )
    records: list[SVCBenchRecord] = []
    for row_index, row in enumerate(rows):
        if ("query_points" in row) != (annotation_format is AnnotationFormat.GROUPED):
            raise ValueError("annotation file mixes grouped and flat schemas")
        if annotation_format is AnnotationFormat.GROUPED:
            records.extend(_parse_grouped_row(row, row_index))
        else:
            records.append(_parse_flat_row(row, row_index))
    return LoadedAnnotations(
        records=tuple(records),
        source=source,
        purpose=purpose,
        annotation_format=annotation_format,
        annotation_sha256=hashlib.sha256(content).hexdigest(),
        annotation_path=path,
    )


def create_group_kfold_manifest(
    annotations: LoadedAnnotations,
    *,
    n_splits: int,
    seed: int,
) -> FoldManifest:
    if annotations.source.official_clean:
        raise PermissionError("official clean data cannot be split for training or calibration")
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two folds")
    groups = [record.identity.video_id for record in annotations.records]
    if len(set(groups)) < n_splits:
        raise ValueError("GroupKFold n_splits exceeds the number of unique videos")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = list(range(len(annotations.records)))
    folds: list[FoldSplit] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(indices, groups=groups)
    ):
        train_records = tuple(annotations.records[int(index)] for index in train_indices)
        validation_records = tuple(annotations.records[int(index)] for index in validation_indices)
        split = FoldSplit(
            fold_index=fold_index,
            train_query_ids=tuple(record.identity.query_id for record in train_records),
            validation_query_ids=tuple(record.identity.query_id for record in validation_records),
            train_video_ids=tuple(sorted({record.identity.video_id for record in train_records})),
            validation_video_ids=tuple(
                sorted({record.identity.video_id for record in validation_records})
            ),
        )
        folds.append(split)
    _assert_manifest_coverage(folds, annotations.records)
    return FoldManifest(
        dataset_name=annotations.source.name,
        dataset_revision=annotations.source.revision,
        annotation_sha256=annotations.annotation_sha256,
        seed=seed,
        group_key="source_dataset/video_path",
        folds=tuple(folds),
    )


def write_fold_manifest(manifest: FoldManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _jsonl_lines(text: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    )


def _parse_json_object(line: str, line_number: int) -> dict[str, object]:
    raw = cast(object, json.loads(line))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"JSONL line {line_number} must contain an object with string keys")
    return cast(dict[str, object], raw)


def _parse_flat_row(row: Mapping[str, object], row_index: int) -> SVCBenchRecord:
    trajectory_id = _string(row, "id")
    source_dataset = _string(row, "source_dataset")
    relative_video_path = _string(row, "video_path")
    query_index = _integer(row, "query_index")
    query_id = _string(row, "q_id")
    return SVCBenchRecord(
        identity=SampleIdentity(
            query_id=query_id,
            query_index=query_index,
            video_id=canonical_video_id(source_dataset, relative_video_path),
            trajectory_id=trajectory_id,
        ),
        source_dataset=source_dataset,
        relative_video_path=relative_video_path,
        question=_string(row, "question"),
        query_time=_number(row, "query_time"),
        labels=SupervisionLabels(
            answer=_optional_string(row, "answer"),
            count=_integer(row, "count"),
            occurrence_times=OccurrenceAnnotations((), (), ()),
            counting_type=_string(row, "counting_type"),
            counting_subtype=_string(row, "counting_subtype"),
        ),
    )


def _parse_grouped_row(row: Mapping[str, object], row_index: int) -> tuple[SVCBenchRecord, ...]:
    trajectory_id = _string(row, "id")
    source_dataset = _string(row, "source_dataset")
    relative_video_path = _string(row, "video_path")
    query_points = _mapping(row, "query_points")
    times = _number_sequence(query_points, "time")
    counts = _integer_sequence(query_points, "count")
    if len(times) != len(counts) or not times:
        raise ValueError(f"grouped row {row_index} query point times/counts must align")
    occurrence_times = _occurrence_annotations(row)
    video_id = canonical_video_id(source_dataset, relative_video_path)
    return tuple(
        SVCBenchRecord(
            identity=SampleIdentity(
                query_id=f"{trajectory_id}:{query_index}",
                query_index=query_index,
                video_id=video_id,
                trajectory_id=trajectory_id,
            ),
            source_dataset=source_dataset,
            relative_video_path=relative_video_path,
            question=_string(row, "question"),
            query_time=query_time,
            labels=SupervisionLabels(
                answer=_optional_string(row, "answer"),
                count=counts[query_index],
                occurrence_times=occurrence_times,
                counting_type=_string(row, "counting_type"),
                counting_subtype=_string(row, "counting_subtype"),
            ),
        )
        for query_index, query_time in enumerate(times)
    )


def _assert_manifest_coverage(
    folds: Sequence[FoldSplit], records: Sequence[SVCBenchRecord]
) -> None:
    expected_queries = {record.identity.query_id for record in records}
    validation_queries = [query for fold in folds for query in fold.validation_query_ids]
    if len(validation_queries) != len(set(validation_queries)):
        raise ValueError("a query appears in validation for more than one fold")
    if set(validation_queries) != expected_queries:
        raise ValueError("GroupKFold validation folds do not cover every query exactly once")


def _string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _mapping(row: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = row.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an object with string keys")
    return cast(dict[str, object], value)


def _number_sequence(row: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return tuple(_coerce_number(item, key) for item in value)


def _occurrence_annotations(row: Mapping[str, object]) -> OccurrenceAnnotations:
    value = row.get("occurrence_times")
    if isinstance(value, list):
        return OccurrenceAnnotations(
            points=tuple(_coerce_number(item, "occurrence_times") for item in value),
            starts=(),
            ends=(),
        )
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        mapping = cast(dict[str, object], value)
        return OccurrenceAnnotations(
            points=(),
            starts=_number_sequence(mapping, "start"),
            ends=_number_sequence(mapping, "end"),
        )
    raise ValueError("occurrence_times must be a point list or a start/end object")


def _integer_sequence(row: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return tuple(_coerce_integer(item, key) for item in value)


def _coerce_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} entries must be numeric")
    return float(value)


def _coerce_integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} entries must be integers")
    return value
