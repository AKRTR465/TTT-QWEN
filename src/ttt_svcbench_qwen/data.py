"""Load SVCBench annotations and keep runtime payloads label-free."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

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
_MINUTE_UNITS = frozenset({"minute", "minutes", "min", "mins", "m", "分钟"})


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    revision: str


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    query_id: str
    query_index: int
    video_id: str
    trajectory_id: str


@dataclass(frozen=True, slots=True)
class RuntimeQueryInput:
    video_id: str
    trajectory_id: str
    query_id: str
    query_index: int
    video: Path
    question: str
    query_time: float
    explicit_time_values: tuple[float, ...]
    episode_nonce: int = 0

    def as_payload(self) -> dict[str, object]:
        return {
            "video": self.video,
            "question": self.question,
            "query_time": self.query_time,
            "explicit_time_values": self.explicit_time_values,
        }


@dataclass(frozen=True, slots=True)
class OccurrenceAnnotations:
    points: tuple[float, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SupervisionLabels:
    answer: str | None
    count: int
    occurrence_times: OccurrenceAnnotations
    counting_type: str
    counting_subtype: str


@dataclass(frozen=True, slots=True)
class SVCBenchRecord:
    identity: SampleIdentity
    source_dataset: str
    relative_video_path: str
    question: str
    query_time: float
    labels: SupervisionLabels


@dataclass(frozen=True, slots=True)
class LoadedAnnotations:
    records: tuple[SVCBenchRecord, ...]
    source: DatasetSource
    annotation_sha256: str
    annotation_path: Path


@dataclass(frozen=True, slots=True)
class FoldSplit:
    fold_index: int
    train_video_ids: tuple[str, ...]
    validation_video_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FoldManifest:
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


def extract_explicit_time_values(question: str) -> tuple[float, ...]:
    positioned_values: dict[int, float] = {}
    for match in _EXPLICIT_TIME_PATTERN.finditer(question):
        value = float(match.group(1))
        if match.group(2).lower() in _MINUTE_UNITS:
            value *= 60.0
        positioned_values[match.start(1)] = value
    for pattern in _SHARED_UNIT_RANGE_PATTERNS:
        for match in pattern.finditer(question):
            scale = 60.0 if match.group("unit").lower() in _MINUTE_UNITS else 1.0
            positioned_values[match.start("start")] = float(match.group("start")) * scale
            positioned_values[match.start("end")] = float(match.group("end")) * scale
    return tuple(value for _, value in sorted(positioned_values.items()))


def canonical_video_id(source_dataset: str, relative_video_path: str) -> str:
    return f"{source_dataset}/{PurePosixPath(relative_video_path).as_posix()}"


def load_annotations(
    annotation_path: str | Path,
    *,
    source: DatasetSource,
) -> LoadedAnnotations:
    path = Path(annotation_path)
    content = path.read_bytes()
    text = content.decode("utf-8", errors="strict")
    records = tuple(
        record
        for line in text.splitlines()
        if line.strip()
        for record in _parse_rows(cast(dict[str, object], json.loads(line)))
    )
    return LoadedAnnotations(
        records=records,
        source=source,
        annotation_sha256=hashlib.sha256(content).hexdigest(),
        annotation_path=path,
    )


def create_group_kfold_manifest(
    annotations: LoadedAnnotations,
    *,
    n_splits: int,
    seed: int,
) -> FoldManifest:
    groups = [record.identity.video_id for record in annotations.records]
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = list(range(len(annotations.records)))
    folds: list[FoldSplit] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(indices, groups=groups)
    ):
        folds.append(
            FoldSplit(
                fold_index=fold_index,
                train_video_ids=_video_ids(annotations.records, train_indices),
                validation_video_ids=_video_ids(annotations.records, validation_indices),
            )
        )
    return FoldManifest(folds=tuple(folds))


def _video_ids(records: Sequence[SVCBenchRecord], selected: Sequence[int]) -> tuple[str, ...]:
    return tuple(sorted({records[int(index)].identity.video_id for index in selected}))


def _parse_rows(row: Mapping[str, object]) -> tuple[SVCBenchRecord, ...]:
    trajectory_id = str(row["id"])
    source_dataset = str(row["source_dataset"])
    relative_video_path = str(row["video_path"])
    if "query_points" in row:
        query_points = cast(Mapping[str, object], row["query_points"])
        times = tuple(float(value) for value in cast(Sequence[float], query_points["time"]))
        counts = tuple(int(value) for value in cast(Sequence[int], query_points["count"]))
        indexes = tuple(range(len(times)))
        query_ids = tuple(f"{trajectory_id}:{index}" for index in indexes)
        occurrence_times = _occurrence_annotations(row)
    else:
        times = (float(cast(float, row["query_time"])),)
        counts = (int(cast(int, row["count"])),)
        indexes = (int(cast(int, row["query_index"])),)
        query_ids = (str(row["q_id"]),)
        occurrence_times = OccurrenceAnnotations((), (), ())
    video_id = canonical_video_id(source_dataset, relative_video_path)
    return tuple(
        SVCBenchRecord(
            identity=SampleIdentity(
                query_id=query_id,
                query_index=query_index,
                video_id=video_id,
                trajectory_id=trajectory_id,
            ),
            source_dataset=source_dataset,
            relative_video_path=relative_video_path,
            question=str(row["question"]),
            query_time=query_time,
            labels=SupervisionLabels(
                answer=cast("str | None", row.get("answer")),
                count=count,
                occurrence_times=occurrence_times,
                counting_type=str(row["counting_type"]),
                counting_subtype=str(row["counting_subtype"]),
            ),
        )
        for query_id, query_index, query_time, count in zip(query_ids, indexes, times, counts)
    )


def _occurrence_annotations(row: Mapping[str, object]) -> OccurrenceAnnotations:
    value = row.get("occurrence_times")
    if isinstance(value, list):
        return OccurrenceAnnotations(
            points=tuple(float(item) for item in cast(Sequence[float], value)),
            starts=(),
            ends=(),
        )
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return OccurrenceAnnotations(
            points=(),
            starts=tuple(float(item) for item in cast(Sequence[float], mapping.get("start", ()))),
            ends=tuple(float(item) for item in cast(Sequence[float], mapping.get("end", ()))),
        )
    return OccurrenceAnnotations((), (), ())
