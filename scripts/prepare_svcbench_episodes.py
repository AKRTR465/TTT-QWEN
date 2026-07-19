#!/usr/bin/env python3
"""Prepare versioned A2/A5 SVCBench manifests for one production run."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

import av

from ttt_svcbench_qwen.data import (
    DatasetPurpose,
    DatasetSource,
    LoadedAnnotations,
    SVCBenchRecord,
    load_annotations,
)
from ttt_svcbench_qwen.episode_data import (
    build_production_episode_manifest,
    write_production_episode_manifest,
)


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    run_id = args.run_id or datetime.now().strftime("%m%d_%H%M%S_prepare_svcbench_k8")
    output_dir = args.output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    run_config = {
        "run_id": run_id,
        "annotation": str(args.annotation),
        "video_duration_manifest": (
            None if args.video_durations is None else str(args.video_durations)
        ),
        "converted_dataset": (
            None if args.converted_dataset is None else str(args.converted_dataset)
        ),
        "video_root": None if args.video_root is None else str(args.video_root),
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "fold_index": 0,
        "seed": 42,
        "truncation_horizon": 8,
        "world_size": 4,
        "video_duration_tolerance_seconds": 1.0,
    }
    _write_json(output_dir / "run_config.json", run_config)
    (output_dir / "experiment.log").write_text(
        f"start run_id={run_id}\nstage=load_annotations\n",
        encoding="utf-8",
    )

    annotations = load_annotations(
        args.annotation,
        source=DatasetSource(args.dataset_name, args.dataset_revision, False),
        purpose=DatasetPurpose.TRAINING,
    )
    runtime_video_paths = (
        None
        if args.converted_dataset is None
        else _load_converted_video_paths(annotations, args.converted_dataset, args.video_root)
    )
    if args.video_durations is not None:
        durations = _load_durations(args.video_durations)
    else:
        if runtime_video_paths is None or args.video_root is None:
            raise ValueError(
                "omit --video-durations only when --converted-dataset and --video-root are set"
            )
        durations = _derive_video_durations(
            annotations,
            runtime_video_paths,
            args.video_root,
        )
    _write_json(output_dir / "video_durations.json", durations)
    manifest = build_production_episode_manifest(
        annotations,
        video_durations=durations,
        runtime_video_paths=runtime_video_paths,
        fold_index=0,
        seed=42,
        n_splits=5,
        truncation_horizon=8,
        world_size=4,
    )
    write_production_episode_manifest(
        manifest,
        manifest_path=output_dir / "dataset_manifest.json",
        failed_path=output_dir / "failed.jsonl",
    )
    real_episodes = tuple(episode for episode in manifest.episodes if episode.loss_weight == 1.0)
    (output_dir / "succeeded.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "split": episode.split,
                    "support_count": episode.support_count,
                    "query_count": episode.query_count,
                    "tbptt_segment_count": episode.tbptt_segment_count,
                },
                ensure_ascii=False,
            )
            + "\n"
            for episode in real_episodes
        ),
        encoding="utf-8",
    )
    elapsed = time.monotonic() - started
    summary = {
        "run_id": run_id,
        "a2_query_count": len(manifest.a2_query_ids),
        "a5_episode_count": len(real_episodes),
        "a5_query_count": sum(episode.query_count for episode in real_episodes),
        "padding_episode_count": len(manifest.episodes) - len(real_episodes),
        "failed_query_count": len(manifest.failures),
        "task_query_counts": dict(manifest.task_query_counts),
        "elapsed_seconds": elapsed,
        "status": "completed",
    }
    _write_json(output_dir / "run_summary.json", summary)
    with (output_dir / "experiment.log").open("a", encoding="utf-8") as handle:
        handle.write("stage=build_manifest\n")
        handle.write(
            "complete "
            f"a2_queries={summary['a2_query_count']} "
            f"a5_episodes={summary['a5_episode_count']} "
            f"failed={summary['failed_query_count']} "
            f"elapsed_seconds={elapsed:.3f}\n"
        )
    print(output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--video-durations", type=Path)
    parser.add_argument("--converted-dataset", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    return parser.parse_args()


def _load_converted_video_paths(
    annotations: LoadedAnnotations,
    path: Path,
    video_root: Path | None,
) -> dict[str, str]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, list):
        raise ValueError("converted SVCBench dataset must contain one JSON list")
    annotations_by_video: dict[tuple[str, str], list[SVCBenchRecord]] = defaultdict(list)
    for record in annotations.records:
        annotations_by_video[(record.source_dataset, record.relative_video_path)].append(record)
    mapping: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("converted SVCBench rows must be objects")
        source = _required_string(item, "source_dataset")
        source_path = _required_string(item, "source_video_path")
        question = _required_string(item, "question")
        query_index = item.get("query_index")
        query_time = item.get("query_time")
        videos = item.get("videos")
        if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str):
            raise ValueError("converted SVCBench rows require exactly one video path")
        relative = PurePosixPath(videos[0])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("converted SVCBench video paths must be safe and relative")
        converted_video = path.parent / relative
        if not converted_video.is_file():
            raise FileNotFoundError(f"converted SVCBench clip does not exist: {converted_video}")
        if (
            isinstance(query_time, bool)
            or not isinstance(query_time, (int, float))
            or not math.isfinite(float(query_time))
            or float(query_time) < 0.0
        ):
            raise ValueError("converted SVCBench rows require a non-negative query_time")

        if query_index is None:
            subtype = _required_string(item, "counting_subtype")
            candidates = [
                record
                for record in annotations_by_video[(source, source_path)]
                if record.labels.counting_subtype.casefold() == subtype.casefold()
            ]
        else:
            if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
                raise ValueError("converted SVCBench query_index must be non-negative or null")
            candidates = [
                record
                for record in annotations_by_video[(source, source_path)]
                if record.identity.query_index == query_index and record.question == question
            ]
        if not candidates:
            raise ValueError(
                "converted SVCBench row has no annotation candidate: "
                f"source={source!r} video={source_path!r} query_index={query_index!r} "
                f"query_time={query_time!r} question={question!r}"
            )
        nearest_distance = min(abs(record.query_time - float(query_time)) for record in candidates)
        nearest = tuple(
            record
            for record in candidates
            if abs(abs(record.query_time - float(query_time)) - nearest_distance) < 1e-9
        )
        if len(nearest) != 1 or nearest_distance > 1.001:
            raise ValueError(
                "converted SVCBench row could not be resolved uniquely by nearest query_time: "
                f"source={source!r} video={source_path!r} query_index={query_index!r} "
                f"query_time={query_time!r} distance={nearest_distance!r} "
                f"candidates={len(nearest)}"
            )
        query_id = nearest[0].identity.query_id
        if query_id in mapping:
            raise ValueError(f"converted SVCBench contains duplicate Query mapping: {query_id}")
        runtime_relative = PurePosixPath(source) / PurePosixPath(source_path)
        if runtime_relative.is_absolute() or ".." in runtime_relative.parts:
            raise ValueError("original SVCBench video paths must be safe and relative")
        relative_path = runtime_relative.as_posix()
        if video_root is not None and not (video_root / relative_path).is_file():
            raise FileNotFoundError(
                f"converted SVCBench video does not exist: {video_root / relative_path}"
            )
        mapping[query_id] = relative_path

    if len(mapping) != len(raw) or len(mapping) != len(annotations.records):
        raise ValueError("converted SVCBench rows and annotation Queries do not match exactly")
    return mapping


def _derive_video_durations(
    annotations: LoadedAnnotations,
    runtime_video_paths: dict[str, str],
    video_root: Path,
) -> dict[str, float]:
    grouped: dict[str, list[SVCBenchRecord]] = defaultdict(list)
    for record in annotations.records:
        grouped[record.identity.video_id].append(record)
    durations: dict[str, float] = {}
    for video_id, records in grouped.items():
        latest = max(records, key=lambda row: (row.query_time, row.identity.query_index))
        path = video_root / runtime_video_paths[latest.identity.query_id]
        durations[video_id] = _video_duration(path)
    return durations


def _video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        else:
            last = None
            for frame in container.decode(stream):
                if frame.time is not None:
                    last = float(frame.time)
                elif frame.pts is not None and frame.time_base is not None:
                    last = float(frame.pts * frame.time_base)
            duration = 0.0 if last is None else last
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"could not determine a positive video duration: {path}")
    return duration


def _required_string(row: dict[object, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"converted SVCBench row requires non-empty {key}")
    return value


def _load_durations(path: Path) -> dict[str, float]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if isinstance(raw, dict):
        if not all(isinstance(key, str) for key in raw):
            raise ValueError("duration manifest object keys must be strings")
        result = {key: _duration(value) for key, value in raw.items()}
    elif isinstance(raw, list):
        result = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("duration manifest list entries must be objects")
            key = item.get("video_id") or item.get("video_path")
            if not isinstance(key, str) or not key:
                raise ValueError("duration rows require video_id or video_path")
            result[key] = _duration(item.get("duration_seconds"))
    else:
        raise ValueError("duration manifest must be an object or list")
    if not result:
        raise ValueError("duration manifest cannot be empty")
    return result


def _duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("video duration must be numeric")
    duration = float(value)
    if duration <= 0.0:
        raise ValueError("video duration must be positive")
    return duration


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
