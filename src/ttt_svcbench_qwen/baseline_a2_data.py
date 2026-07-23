"""Strict A2 dataset bridge for the authoritative Qwen3-VL SVCBench SFT clips.

The SFT JSON remains the only source of runtime video, prompt, answer, and row order.  Official
SVCBench annotations are joined as a loss-only sidecar and converted to the clip-local timeline;
they never enter the runtime prompt, Retriever forward inputs, Bank, or FSM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import av
from torch.utils.data import Dataset, RandomSampler, Sampler

from ttt_svcbench_qwen.data import (
    DatasetPurpose,
    DatasetSource,
    RuntimeQueryInput,
    SVCBenchRecord,
    canonical_video_id,
    extract_explicit_time_values,
    load_annotations,
)
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    AnswerSupervisionSidecar,
    EpisodeSplit,
    ManifestStage,
    ProductionQueryRecord,
    WeakQuerySidecar,
    official_operator,
    official_time_mode,
)
from ttt_svcbench_qwen.query_encoder import Operator

BASELINE_A2_DATASET_NAME = "svcbench_qwen3vl_sft"
BASELINE_A2_FILE_NAME = "svcbench_qwen3vl_sft.json"
BASELINE_A2_EXPECTED_SHA256 = (
    "aae450f9d82ea067a28c294d2ab8c8dcde99be58c225651546fc62bde5a3d7eb"
)
BASELINE_A2_EXPECTED_ROWS = 4_576
_BASELINE_SOURCE_NAME = "svcbench_qwen3vl_sft"
_VIDEO_PREFIX = "<video>\n"


@dataclass(frozen=True, slots=True)
class BaselineA2DatasetAudit:
    dataset_path: str
    dataset_sha256: str
    row_count: int
    joined_count: int
    unique_id_count: int
    unique_q_id_count: int
    video_count: int
    question_normalized_rows: int
    time_disambiguated_rows: int
    query_time_drift_rows: int
    official_count_mismatch_rows: int
    visible_occurrence_points: int
    masked_occurrence_points: int
    visible_occurrence_intervals: int
    clipped_occurrence_intervals: int
    masked_occurrence_intervals: int

    def __post_init__(self) -> None:
        if self.row_count <= 0 or self.joined_count != self.row_count:
            raise ValueError("baseline A2 sidecar join must cover every SFT row")
        if self.unique_id_count != self.row_count or self.unique_q_id_count != self.row_count:
            raise ValueError("baseline A2 id/q_id values must be unique")
        if self.video_count != self.row_count:
            raise ValueError("baseline A2 requires one existing short video per SFT row")


class BaselineA2ClipDataset(Dataset[A2QueryRecord]):  # type: ignore[misc]
    """Immutable 4,576-row A2 view preserving the baseline JSON order exactly."""

    def __init__(
        self,
        records: tuple[A2QueryRecord, ...],
        *,
        audit: BaselineA2DatasetAudit,
    ) -> None:
        if not records:
            raise ValueError("baseline A2 dataset cannot be empty")
        self.records = records
        self.audit = audit
        self.stage = ManifestStage.A2
        self.split = EpisodeSplit.TRAIN
        self.index_by_id = {
            record.query.runtime.query_id: index for index, record in enumerate(records)
        }
        if len(self.index_by_id) != len(records):
            raise ValueError("baseline A2 runtime query IDs must be unique")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> A2QueryRecord:
        return self.records[index]


@dataclass(frozen=True, slots=True)
class _LocalOccurrenceAudit:
    visible_points: int
    masked_points: int
    visible_intervals: int
    clipped_intervals: int
    masked_intervals: int


def load_baseline_a2_clip_dataset(
    dataset_dir: str | Path,
    *,
    dataset_name: str | Sequence[str],
    weak_sidecar_path: str | Path,
    expected_sha256: str = BASELINE_A2_EXPECTED_SHA256,
    expected_rows: int = BASELINE_A2_EXPECTED_ROWS,
    duration_resolver: Callable[[Path], float] | None = None,
) -> BaselineA2ClipDataset:
    """Load and fully audit the baseline SFT rows plus official-weak sidecar.

    Production callers use the pinned hash/count defaults.  Tests may inject a fixture hash,
    count, and duration resolver without weakening the production entry point.
    """

    if isinstance(dataset_name, str):
        normalized_dataset_name = dataset_name
    else:
        dataset_names = tuple(dataset_name)
        if len(dataset_names) != 1 or not isinstance(dataset_names[0], str):
            raise ValueError(
                "baseline A2 requires exactly one LLaMA-Factory dataset, "
                f"got {dataset_names!r}"
            )
        normalized_dataset_name = dataset_names[0]
    if normalized_dataset_name != BASELINE_A2_DATASET_NAME:
        raise ValueError(
            "baseline A2 requires "
            f"dataset={BASELINE_A2_DATASET_NAME!r}, got {normalized_dataset_name!r}"
        )
    root = Path(dataset_dir).expanduser().resolve()
    dataset_path = root / BASELINE_A2_FILE_NAME
    _assert_dataset_info(root, normalized_dataset_name)
    content = dataset_path.read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "baseline A2 SFT JSON SHA256 drift: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    raw_rows = cast(object, json.loads(content.decode("utf-8", errors="strict")))
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_rows:
        raise ValueError(
            f"baseline A2 requires exactly {expected_rows} SFT rows, "
            f"found {len(raw_rows) if isinstance(raw_rows, list) else 'non-list'}"
        )

    annotations = load_annotations(
        weak_sidecar_path,
        source=DatasetSource(
            name="svcbench_official_weak",
            revision="baseline-clips-v1",
            official_clean=False,
        ),
        purpose=DatasetPurpose.TRAINING,
    )
    sidecar_by_key: dict[tuple[str, str, str, int], list[SVCBenchRecord]] = defaultdict(list)
    for record in annotations.records:
        key = _sidecar_key(
            record.source_dataset,
            record.relative_video_path,
            record.question,
            record.identity.query_index,
        )
        sidecar_by_key[key].append(record)

    probe_duration = duration_resolver or _probe_video_duration
    ids: set[str] = set()
    q_ids: set[str] = set()
    used_sidecar_ids: set[str] = set()
    records: list[A2QueryRecord] = []
    point_visible = point_masked = 0
    interval_visible = interval_clipped = interval_masked = 0
    question_normalized_rows = 0
    time_disambiguated_rows = 0
    query_time_drift_rows = 0
    official_count_mismatch_rows = 0
    for row_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError(f"baseline A2 row {row_index} must be one string-keyed object")
        row = cast(dict[str, object], raw)
        identity, q_id = _baseline_identity(row, row_index)
        if identity in ids or q_id in q_ids:
            raise ValueError(f"baseline A2 duplicate id/q_id at row {row_index}")
        ids.add(identity)
        q_ids.add(q_id)
        source_dataset = _required_string(row, "source_dataset")
        source_video_path = _required_string(row, "source_video_path")
        question = _required_string(row, "question")
        query_index = _normalized_query_index(row.get("query_index"))
        original_query_time = _required_number(row, "query_time")
        messages, user_content, assistant_content = _baseline_messages(row)
        del messages  # validation retains the exact content in the typed A2 record below.
        declared_answer = _required_string(row, "answer")
        if declared_answer != assistant_content:
            raise ValueError(f"baseline A2 row {identity!r} answer/messages disagree")
        try:
            integer_answer = int(declared_answer)
        except ValueError as error:
            raise ValueError(f"baseline A2 row {identity!r} answer is not an integer") from error
        if str(integer_answer) != declared_answer.strip():
            raise ValueError(f"baseline A2 row {identity!r} answer is not a canonical integer")
        baseline_operator = _baseline_operator(_required_string(row, "counting_subtype"))
        key = _sidecar_key(source_dataset, source_video_path, question, query_index)
        candidates = sidecar_by_key.get(key, ())
        if not candidates:
            normalized_question = _official_question_variant(question)
            if normalized_question is not None:
                normalized_key = _sidecar_key(
                    source_dataset,
                    source_video_path,
                    normalized_question,
                    query_index,
                )
                candidates = sidecar_by_key.get(normalized_key, ())
                if candidates:
                    question_normalized_rows += 1
        if not candidates:
            raise ValueError(f"baseline A2 row {identity!r} has no official-weak sidecar match")
        operator_matches = tuple(
            candidate
            for candidate in candidates
            if official_operator(
                candidate.labels.counting_type,
                candidate.labels.counting_subtype,
            )
            is baseline_operator
        )
        if len(operator_matches) > 1:
            count_matches = tuple(
                candidate
                for candidate in operator_matches
                if candidate.labels.count == integer_answer
            )
            if count_matches:
                operator_matches = count_matches
        if len(operator_matches) > 1:
            minimum_distance = min(
                abs(candidate.query_time - original_query_time)
                for candidate in operator_matches
            )
            operator_matches = tuple(
                candidate
                for candidate in operator_matches
                if math.isclose(
                    abs(candidate.query_time - original_query_time),
                    minimum_distance,
                    abs_tol=1e-9,
                )
            )
        if len(operator_matches) != 1:
            raise ValueError(
                f"baseline A2 row {identity!r} has {len(operator_matches)} unique matches "
                f"within {len(candidates)} official-weak sidecar candidates"
            )
        if len(candidates) > 1:
            time_disambiguated_rows += 1
        weak_source = operator_matches[0]
        if weak_source.identity.query_id in used_sidecar_ids:
            raise ValueError(
                f"official-weak row {weak_source.identity.query_id!r} matched more than once"
            )
        used_sidecar_ids.add(weak_source.identity.query_id)

        sidecar_operator = official_operator(
            weak_source.labels.counting_type,
            weak_source.labels.counting_subtype,
        )
        if baseline_operator is not sidecar_operator:
            raise ValueError(f"baseline A2 row {identity!r} subtype sidecar disagrees")
        if not math.isclose(original_query_time, weak_source.query_time, abs_tol=0.05):
            query_time_drift_rows += 1
        if integer_answer != weak_source.labels.count:
            official_count_mismatch_rows += 1

        video_relative = _single_video(row)
        expected_video = f"videos/{q_id}.mp4"
        if video_relative != expected_video:
            raise ValueError(
                f"baseline A2 row {identity!r} video must be {expected_video!r}, "
                f"found {video_relative!r}"
            )
        video_path = (root / video_relative).resolve()
        if not video_path.is_relative_to(root) or not video_path.is_file():
            raise FileNotFoundError(f"baseline A2 video is missing: {video_path}")
        duration = float(probe_duration(video_path))
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"baseline A2 video duration is invalid: {video_path}")
        points, intervals, occurrence_audit = _clip_local_occurrences(
            weak_source,
            original_query_time=original_query_time,
            clip_duration=duration,
        )
        point_visible += occurrence_audit.visible_points
        point_masked += occurrence_audit.masked_points
        interval_visible += occurrence_audit.visible_intervals
        interval_clipped += occurrence_audit.clipped_intervals
        interval_masked += occurrence_audit.masked_intervals

        query_id = identity
        video_id = canonical_video_id(_BASELINE_SOURCE_NAME, video_relative)
        runtime = RuntimeQueryInput(
            video_id=video_id,
            trajectory_id=identity,
            query_id=query_id,
            query_index=query_index,
            video=Path(video_relative),
            question=question,
            query_time=duration,
            explicit_time_values=extract_explicit_time_values(question),
        )
        weak = WeakQuerySidecar(
            query_id=query_id,
            query_index=query_index,
            query_time=duration,
            # The SFT JSON is the A2 fact source. Two pinned rows disagree with the grouped
            # annotation after its early-time clip clamp; retain the exact baseline target and
            # expose the discrepancy in BaselineA2DatasetAudit instead of dropping either row.
            count=integer_answer,
            counting_type=weak_source.labels.counting_type,
            counting_subtype=weak_source.labels.counting_subtype,
            operator=sidecar_operator.value,
            time_mode=official_time_mode(weak_source, sidecar_operator).value,
            occurrence_points=points,
            occurrence_intervals=intervals,
        )
        records.append(
            A2QueryRecord(
                source_dataset=_BASELINE_SOURCE_NAME,
                relative_video_path=video_relative,
                video_id=video_id,
                trajectory_id=identity,
                split=EpisodeSplit.TRAIN,
                task_class=weak_source.labels.counting_type.upper(),
                query=ProductionQueryRecord(
                    runtime=runtime,
                    answer=AnswerSupervisionSidecar(
                        query_id=query_id,
                        answer=assistant_content,
                        provenance="official_explicit",
                    ),
                    weak=weak,
                ),
                sampling_weight=1.0,
                answer_user_content=user_content,
            )
        )

    if len(used_sidecar_ids) != len(records):
        raise RuntimeError("baseline A2 sidecar join did not remain one-to-one")
    audit = BaselineA2DatasetAudit(
        dataset_path=str(dataset_path),
        dataset_sha256=actual_sha256,
        row_count=len(records),
        joined_count=len(used_sidecar_ids),
        unique_id_count=len(ids),
        unique_q_id_count=len(q_ids),
        video_count=len(records),
        question_normalized_rows=question_normalized_rows,
        time_disambiguated_rows=time_disambiguated_rows,
        query_time_drift_rows=query_time_drift_rows,
        official_count_mismatch_rows=official_count_mismatch_rows,
        visible_occurrence_points=point_visible,
        masked_occurrence_points=point_masked,
        visible_occurrence_intervals=interval_visible,
        clipped_occurrence_intervals=interval_clipped,
        masked_occurrence_intervals=interval_masked,
    )
    return BaselineA2ClipDataset(tuple(records), audit=audit)


def build_baseline_a2_train_sampler(
    dataset: object,
    rank: int,
    world_size: int,
) -> Sampler[int]:
    """Match the standard Trainer sampler; Accelerate performs distributed batch sharding."""

    if not isinstance(dataset, BaselineA2ClipDataset):
        raise TypeError("baseline A2 sampler requires BaselineA2ClipDataset")
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError("baseline A2 distributed rank/world size is invalid")
    return RandomSampler(dataset)


def _assert_dataset_info(root: Path, dataset_name: str) -> None:
    path = root / "dataset_info.json"
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or not isinstance(raw.get(dataset_name), dict):
        raise ValueError(f"dataset_info.json is missing {dataset_name!r}")
    entry = cast(dict[str, object], raw[dataset_name])
    if entry.get("file_name") != BASELINE_A2_FILE_NAME:
        raise ValueError("dataset_info.json baseline file_name drifted")
    columns = entry.get("columns")
    if not isinstance(columns, dict) or columns.get("messages") != "messages" or columns.get(
        "videos"
    ) != "videos":
        raise ValueError("dataset_info.json baseline ShareGPT/video columns drifted")


def _baseline_identity(row: Mapping[str, object], row_index: int) -> tuple[str, str]:
    identity = _required_string(row, "id")
    q_id = _required_string(row, "q_id")
    expected_q_id = f"{row_index:04d}"
    if q_id != expected_q_id or identity != f"svcbench-{q_id}":
        raise ValueError(
            f"baseline A2 id/q_id order drift at row {row_index}: {identity!r}, {q_id!r}"
        )
    return identity, q_id


def _baseline_messages(
    row: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], str, str]:
    raw = row.get("messages")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(item, dict) for item in raw)
    ):
        raise ValueError("baseline A2 messages must contain exactly one user and one assistant")
    user = cast(dict[str, object], raw[0])
    assistant = cast(dict[str, object], raw[1])
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("baseline A2 messages roles must be user then assistant")
    user_content = _required_string(user, "content")
    assistant_content = _required_string(assistant, "content")
    if not user_content.startswith(_VIDEO_PREFIX) or user_content.count("<video>") != 1:
        raise ValueError("baseline A2 user message must contain exactly one leading video marker")
    if not user_content.removeprefix(_VIDEO_PREFIX).strip():
        raise ValueError("baseline A2 user message text cannot be empty")
    return (user, assistant), user_content, assistant_content


def _single_video(row: Mapping[str, object]) -> str:
    videos = row.get("videos")
    if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str):
        raise ValueError("baseline A2 requires exactly one video path per SFT row")
    relative = PurePosixPath(videos[0])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("baseline A2 video path must be dataset-relative")
    return relative.as_posix()


def _sidecar_key(
    source_dataset: str,
    source_video_path: str,
    question: str,
    query_index: int,
) -> tuple[str, str, str, int]:
    return source_dataset, PurePosixPath(source_video_path).as_posix(), question, query_index


def _official_question_variant(question: str) -> str | None:
    """Undo the one deterministic O1-Delta wording change in the baseline export."""

    prefix = "How many new "
    if not question.startswith(prefix):
        return None
    return "How many " + question.removeprefix(prefix)


def _normalized_query_index(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("baseline A2 query_index must be a non-negative integer or null")
    return value


def _baseline_operator(value: str) -> Operator:
    normalized = value.strip().casefold().replace("_", "-")
    mapping = {
        "o1-snapshot": Operator.O1_SNAP,
        "o1-delta": Operator.O1_DELTA,
        "o2-unique": Operator.O2_UNIQUE,
        "o2-gain": Operator.O2_GAIN,
        "e1-act": Operator.E1_ACTION,
        "e1-trans": Operator.E1_TRANSIT,
        "e2-cyclic": Operator.E2_PERIODIC,
        "e2-episodic": Operator.E2_EPISODE,
    }
    operator = mapping.get(normalized)
    if operator is None:
        raise ValueError(f"unsupported baseline counting_subtype: {value!r}")
    return operator


def _clip_local_occurrences(
    record: SVCBenchRecord,
    *,
    original_query_time: float,
    clip_duration: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...], _LocalOccurrenceAudit]:
    clip_start = original_query_time - clip_duration
    clip_end = original_query_time
    tolerance = 1e-6
    points: list[float] = []
    masked_points = 0
    occurrence = record.labels.occurrence_times
    for point in occurrence.points:
        local = point - clip_start
        if -tolerance <= local <= clip_duration + tolerance:
            points.append(min(clip_duration, max(0.0, local)))
        else:
            masked_points += 1

    intervals: list[tuple[float, float]] = []
    clipped_intervals = 0
    masked_intervals = 0
    for start, end in zip(occurrence.starts, occurrence.ends, strict=True):
        if end < clip_start - tolerance or start > clip_end + tolerance:
            masked_intervals += 1
            continue
        clipped_start = max(start, clip_start)
        clipped_end = min(end, clip_end)
        if clipped_end < clipped_start:
            masked_intervals += 1
            continue
        if not math.isclose(clipped_start, start, abs_tol=tolerance) or not math.isclose(
            clipped_end, end, abs_tol=tolerance
        ):
            clipped_intervals += 1
        intervals.append(
            (
                min(clip_duration, max(0.0, clipped_start - clip_start)),
                min(clip_duration, max(0.0, clipped_end - clip_start)),
            )
        )
    return (
        tuple(points),
        tuple(intervals),
        _LocalOccurrenceAudit(
            visible_points=len(points),
            masked_points=masked_points,
            visible_intervals=len(intervals),
            clipped_intervals=clipped_intervals,
            masked_intervals=masked_intervals,
        ),
    )


def _probe_video_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
            else:
                duration = 0.0
    except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError) as error:
        raise ValueError(f"cannot probe baseline A2 video duration: {path}") from error
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"baseline A2 video has no positive duration: {path}")
    return duration


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"baseline A2 {key} must be a non-empty string")
    return value


def _required_number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"baseline A2 {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"baseline A2 {key} must be finite and non-negative")
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the strict baseline-clips A2 dataset")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=BASELINE_A2_DATASET_NAME)
    parser.add_argument("--weak-sidecar", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    dataset = load_baseline_a2_clip_dataset(
        args.dataset_dir,
        dataset_name=args.dataset,
        weak_sidecar_path=args.weak_sidecar,
    )
    payload = json.dumps(asdict(dataset.audit), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
