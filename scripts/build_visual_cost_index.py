"""Build a strict schema-4 A2/A5 visual and runtime cost index.

Pass ``--runtime-trace --require-measured-runtime`` for the formal A2 profile. The output contains
only media metadata, counts and timings; decoded Query frames or processor tensors are never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import transformers

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    adaptive_support_schedule,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.visual_cost import (
    VISUAL_COST_SCHEMA_VERSION,
    VisualCostRecord,
    make_visual_cost_fingerprint,
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    query_frame_count: int
    query_visual_tokens: int
    decode_seconds: float
    processor_seconds: float
    preparation_seconds: float
    training_seconds: float
    support_cache_bytes: int


@dataclass(frozen=True, slots=True)
class MediaProbe:
    codec: str = "unknown"
    width: int = 0
    height: int = 0
    keyframe_interval_seconds: float | None = None


def _load_runtime_measurements(path: Path) -> dict[str, RuntimeMeasurement]:
    paths = (path,) if path.is_file() else tuple(sorted(path.rglob("runtime_*.jsonl")))
    if not paths:
        raise ValueError("runtime trace contains no JSONL files")
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    support_sizes: dict[str, dict[str, int]] = defaultdict(dict)
    for source in paths:
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            event = row.get("event")
            if event == "a2_collate_done" and isinstance(row.get("query_id"), str):
                record_id = str(row["query_id"])
                for answer_field, state_field, destination in (
                    ("query_frame_count", "state_query_frame_count", "query_frame_count"),
                    (
                        "query_visual_token_count",
                        "state_query_visual_token_count",
                        "query_visual_token_count",
                    ),
                ):
                    answer_value = row.get(answer_field)
                    state_value = row.get(state_field, 0.0)
                    if all(
                        isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in (answer_value, state_value)
                    ):
                        values[record_id][destination].append(
                            float(answer_value) + float(state_value)
                        )
                _append_numeric(values[record_id], row, "query_decode_seconds")
                _append_numeric(values[record_id], row, "query_processor_seconds")
                _append_numeric(
                    values[record_id], row, "seconds", destination="preparation_seconds"
                )
            elif event == "runtime_cost_observation" and isinstance(
                row.get("record_id"), str
            ):
                record_id = str(row["record_id"])
                _append_numeric(values[record_id], row, "training_seconds")
            elif (
                event == "support_cache_read"
                and isinstance(row.get("record_id"), str)
                and isinstance(row.get("chunk_id"), str)
                and isinstance(row.get("cache_bytes"), (int, float))
            ):
                record_id = str(row["record_id"])
                chunk_id = str(row["chunk_id"])
                support_sizes[record_id][chunk_id] = max(
                    support_sizes[record_id].get(chunk_id, 0), int(row["cache_bytes"])
                )
    result: dict[str, RuntimeMeasurement] = {}
    required = (
        "query_frame_count",
        "query_visual_token_count",
        "query_decode_seconds",
        "query_processor_seconds",
        "preparation_seconds",
        "training_seconds",
    )
    for record_id, fields in values.items():
        if any(not fields.get(name) for name in required):
            continue
        result[record_id] = RuntimeMeasurement(
            query_frame_count=max(1, round(statistics.fmean(fields["query_frame_count"]))),
            query_visual_tokens=max(
                1, round(statistics.fmean(fields["query_visual_token_count"]))
            ),
            decode_seconds=statistics.fmean(fields["query_decode_seconds"]),
            processor_seconds=statistics.fmean(fields["query_processor_seconds"]),
            preparation_seconds=statistics.fmean(fields["preparation_seconds"]),
            training_seconds=statistics.fmean(fields["training_seconds"]),
            support_cache_bytes=sum(support_sizes.get(record_id, {}).values()),
        )
    return result


def _append_numeric(
    fields: dict[str, list[float]],
    row: dict[object, object],
    field: str,
    *,
    destination: str | None = None,
) -> None:
    key = field if destination is None else destination
    value = row.get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0:
        fields[key].append(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("a2", "a5"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-config", type=Path, default=None)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--minimum-pixels", required=True, type=int)
    parser.add_argument("--maximum-pixels", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float32"))
    parser.add_argument("--visual-batch-size", required=True, type=int)
    parser.add_argument(
        "--cache-mode",
        required=True,
        choices=("disabled", "read_write", "readonly"),
    )
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument(
        "--state-query-visual-mode",
        choices=("recent_chunk",),
        default="recent_chunk",
    )
    parser.add_argument("--state-query-max-frames", type=int, default=16)
    parser.add_argument(
        "--answer-query-visual-mode",
        choices=("causal_prefix",),
        default="causal_prefix",
    )
    parser.add_argument("--answer-query-max-frames", type=int, default=256)
    parser.add_argument("--query-sample-fps", type=float, default=2.0)
    parser.add_argument(
        "--query-decode-strategy",
        choices=("legacy_seek", "grouped_seek"),
        default="grouped_seek",
    )
    parser.add_argument("--query-decode-max-groups", type=int, default=16)
    parser.add_argument("--runtime-trace", type=Path, default=None)
    parser.add_argument("--require-measured-runtime", action="store_true")
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--decode-seconds-per-chunk", type=float, default=0.0)
    parser.add_argument("--processor-seconds-per-chunk", type=float, default=0.0)
    parser.add_argument("--vit-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--query-seconds-per-query", type=float, default=0.0)
    parser.add_argument("--loss-collective-seconds", type=float, default=0.0)
    args = parser.parse_args()

    project = load_config(args.project_config)
    balance = project.loss.official_weak_balance
    fingerprint = make_visual_cost_fingerprint(
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        model_revision=args.model_revision,
        transformers_version=transformers.__version__,
        processor=args.processor,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
        dtype=args.dtype,
        visual_batch_size=args.visual_batch_size,
        cache_mode=args.cache_mode,
        loss_mode=balance.mode.value,
        loss_group_weight=balance.group_weight,
        loss_scale_min=balance.scale_min,
        loss_scale_max=balance.scale_max,
        loss_epsilon=balance.epsilon,
        gpu_model=args.gpu_model,
        query_decode_strategy=args.query_decode_strategy,
        query_decode_max_groups=args.query_decode_max_groups,
        state_query_visual_mode=args.state_query_visual_mode,
        state_query_max_frames=args.state_query_max_frames,
        answer_query_visual_mode=args.answer_query_visual_mode,
        answer_query_max_frames=args.answer_query_max_frames,
        query_sample_fps=args.query_sample_fps,
    )
    train, _validation = load_production_manifest_views(
        args.manifest,
        stage=ManifestStage(args.stage),
    )
    records = tuple(train.records)
    measurements = (
        {} if args.runtime_trace is None else _load_runtime_measurements(args.runtime_trace)
    )
    if args.require_measured_runtime:
        missing = {
            _record_id(record) for record in records if _record_id(record) not in measurements
        }
        if missing:
            raise ValueError(
                "runtime trace does not cover every manifest record: "
                + ", ".join(sorted(missing)[:8])
            )
    video_root = args.video_root
    if video_root is None and os.environ.get("SVCBENCH_VIDEO_ROOT"):
        video_root = Path(os.environ["SVCBENCH_VIDEO_ROOT"])
    rows = [
        _cost_record(
            record,
            decode_per_chunk=args.decode_seconds_per_chunk,
            processor_per_chunk=args.processor_seconds_per_chunk,
            vit_per_token=args.vit_seconds_per_token,
            query_per_query=args.query_seconds_per_query,
            loss_collective=args.loss_collective_seconds,
            query_sample_fps=args.query_sample_fps,
            state_query_visual_mode=args.state_query_visual_mode,
            state_query_max_frames=args.state_query_max_frames,
            answer_query_visual_mode=args.answer_query_visual_mode,
            answer_query_max_frames=args.answer_query_max_frames,
            measurement=measurements.get(_record_id(record)),
            media=_probe_media(record, video_root),
        )
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": VISUAL_COST_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "records": [asdict(row) for row in rows],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} strict visual-cost rows to {args.output}")
    return 0


def _record_id(record: object) -> str:
    if isinstance(record, A2QueryRecord):
        return record.query.runtime.query_id
    if isinstance(record, A5EpisodeRecord):
        return record.episode_id
    raise TypeError("visual cost builder received an unknown manifest record")


def _probe_media(record: object, video_root: Path | None) -> MediaProbe:
    if video_root is None or not isinstance(record, (A2QueryRecord, A5EpisodeRecord)):
        return MediaProbe()
    root = video_root.resolve()
    candidates = (
        (root / record.relative_video_path).resolve(),
        (root / record.source_dataset / record.relative_video_path).resolve(),
    )
    path = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_relative_to(root) and candidate.is_file()
        ),
        None,
    )
    if path is None:
        return MediaProbe()
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            codec = str(getattr(stream.codec_context, "name", None) or "unknown")
            keyframes: list[float] = []
            if stream.time_base is not None:
                for packet in container.demux(stream):
                    if packet.is_keyframe and packet.pts is not None:
                        keyframes.append(float(packet.pts * stream.time_base))
                        if len(keyframes) >= 33:
                            break
            intervals = [
                right - left
                for left, right in zip(keyframes, keyframes[1:], strict=False)
                if right > left
            ]
            return MediaProbe(
                codec=codec,
                width=max(0, int(stream.width or 0)),
                height=max(0, int(stream.height or 0)),
                keyframe_interval_seconds=(
                    None if not intervals else float(statistics.median(intervals))
                ),
            )
    except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError):
        return MediaProbe()


def _cost_record(
    record: object,
    *,
    decode_per_chunk: float,
    processor_per_chunk: float,
    vit_per_token: float,
    query_per_query: float,
    loss_collective: float,
    query_sample_fps: float,
    state_query_visual_mode: str = "recent_chunk",
    state_query_max_frames: int = 16,
    answer_query_visual_mode: str = "causal_prefix",
    answer_query_max_frames: int = 256,
    measurement: RuntimeMeasurement | None = None,
    media: MediaProbe | None = None,
) -> VisualCostRecord:
    coefficients = (
        decode_per_chunk,
        processor_per_chunk,
        vit_per_token,
        query_per_query,
        loss_collective,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
        raise ValueError("cost coefficients must be finite and non-negative")
    if not math.isfinite(query_sample_fps) or query_sample_fps <= 0.0:
        raise ValueError("visual cost Query FPS must be positive")

    if (state_query_visual_mode, state_query_max_frames) != ("recent_chunk", 16):
        raise ValueError("visual cost State Query must be recent_chunk/16")
    if (answer_query_visual_mode, answer_query_max_frames) != ("causal_prefix", 256):
        raise ValueError("visual cost Answer Query must be causal_prefix/256")

    def query_intervals(query_time: float) -> tuple[tuple[float, float, int, float], ...]:
        roles = (
            (state_query_visual_mode, state_query_max_frames),
            (answer_query_visual_mode, answer_query_max_frames),
        )
        result: list[tuple[float, float, int, float]] = []
        for mode, maximum in roles:
            start = 0.0 if mode == "causal_prefix" else max(0.0, query_time - 8.0)
            result.append((start, query_time, maximum, query_sample_fps))
        return tuple(result)

    if isinstance(record, A2QueryRecord):
        _, supports = adaptive_support_schedule(record.query.runtime.query_time)
        intervals = tuple(
            (chunk.start_time, chunk.end_time, chunk.maximum_frames) for chunk in supports
        )
        query_intervals_for_record = query_intervals(record.query.runtime.query_time)
        interval_fps = tuple(2.0 for _ in intervals) + tuple(
            interval[3] for interval in query_intervals_for_record
        )
        intervals = intervals + tuple(interval[:3] for interval in query_intervals_for_record)
        record_id = record.query.runtime.query_id
        support_count = len(supports)
        segment_lengths: tuple[int, ...] = ()
        query_count = 1
    elif isinstance(record, A5EpisodeRecord):
        support_intervals = (
            (record.prewarm.start_time, record.prewarm.end_time, record.prewarm.maximum_frames),
            *(
                (chunk.start_time, chunk.end_time, chunk.maximum_frames)
                for chunk in record.supports
            ),
        )
        query_intervals_for_record = tuple(
            interval
            for query in record.queries
            for interval in query_intervals(query.runtime.query_time)
        )
        intervals = support_intervals + tuple(
            interval[:3] for interval in query_intervals_for_record
        )
        interval_fps = tuple(2.0 for _ in support_intervals) + tuple(
            interval[3] for interval in query_intervals_for_record
        )
        record_id = record.episode_id
        support_count = record.support_count
        segment_lengths = _segment_lengths(record)
        query_count = record.query_count
    else:
        raise TypeError("visual cost builder received an unknown manifest record")
    visual_tokens = tuple(
        _frame_budget(*interval, sample_fps)
        for interval, sample_fps in zip(intervals, interval_fps, strict=True)
    )
    query_role_count = query_count * 2
    query_frame_count = sum(visual_tokens[-query_role_count:])
    query_visual_tokens = query_frame_count
    if measurement is not None:
        query_frame_count = measurement.query_frame_count
        query_visual_tokens = measurement.query_visual_tokens
        visual_tokens = (*visual_tokens[:-query_role_count], query_visual_tokens)
    chunk_count = len(visual_tokens)
    total_tokens = sum(visual_tokens)
    if measurement is None:
        decode_seconds = decode_per_chunk * chunk_count
        processor_seconds = processor_per_chunk * chunk_count
        preparation_seconds = decode_seconds + processor_seconds
        training_seconds = (
            vit_per_token * total_tokens + query_per_query * query_count + loss_collective
        )
        vit_seconds = vit_per_token * total_tokens
        query_seconds = query_per_query * query_count
        support_cache_bytes = 0
        measurement_source = "estimated"
    else:
        decode_seconds = measurement.decode_seconds
        processor_seconds = measurement.processor_seconds
        preparation_seconds = measurement.preparation_seconds
        training_seconds = measurement.training_seconds
        vit_seconds = 0.0
        query_seconds = 0.0
        loss_collective = 0.0
        support_cache_bytes = measurement.support_cache_bytes
        measurement_source = "runtime_trace"
    predicted = preparation_seconds + training_seconds
    media = MediaProbe() if media is None else media
    return VisualCostRecord(
        record_id=record_id,
        support_count=support_count,
        segment_lengths=segment_lengths,
        query_count=query_count,
        visual_tokens=visual_tokens,
        total_visual_tokens=total_tokens,
        maximum_visual_tokens=max(visual_tokens),
        query_frame_count=query_frame_count,
        query_visual_tokens=query_visual_tokens,
        source_codec=media.codec,
        source_width=media.width,
        source_height=media.height,
        keyframe_interval_seconds=media.keyframe_interval_seconds,
        support_cache_bytes=support_cache_bytes,
        decode_seconds=decode_seconds,
        processor_seconds=processor_seconds,
        preparation_seconds=preparation_seconds,
        training_seconds=training_seconds,
        vit_seconds=vit_seconds,
        query_seconds=query_seconds,
        loss_collective_seconds=loss_collective,
        predicted_total_seconds=predicted,
        measurement_source=measurement_source,
    )


def _frame_budget(start: float, end: float, maximum: int, sample_fps: float = 2.0) -> int:
    desired = min(maximum, max(2, int(math.floor((end - start) * sample_fps))))
    return max(2, desired - desired % 2)


def _segment_lengths(record: A5EpisodeRecord) -> tuple[int, ...]:
    remaining = record.support_count
    values: list[int] = []
    while remaining:
        length = min(record.truncation_horizon, remaining)
        values.append(length)
        remaining -= length
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
