from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from ttt_svcbench_qwen.data import DatasetPurpose, DatasetSource, load_annotations
from ttt_svcbench_qwen.episode_data import (
    BalancedA2DistributedSampler,
    EpisodeSplit,
    ManifestStage,
    ProductionManifestDataset,
    RankAlignedA5SegmentSampler,
    adaptive_support_schedule,
    build_production_episode_manifest,
    greedy_nonoverlap_query_groups,
    load_production_episode_manifest,
    load_production_manifest_views,
    write_production_episode_manifest,
)


def test_adaptive_support_schedule_covers_history_and_keeps_four_second_overlap() -> None:
    prewarm, supports = adaptive_support_schedule(200.0)

    assert prewarm.start_time == 0.0
    assert prewarm.end_time < supports[0].end_time
    assert supports[0].start_time == 0.0
    assert supports[-1].end_time < 200.0
    assert len(supports) > 8
    assert all(
        chunk.maximum_frames == (16 if chunk.end_time - chunk.start_time <= 8.0 + 1.0e-6 else 8)
        for chunk in supports
    )
    assert {chunk.maximum_frames for chunk in supports} == {8, 16}
    assert all(left.end_time - right.start_time >= 4.0 for left, right in pairwise(supports))
    assert all(left.end_time < right.end_time for left, right in pairwise(supports))


def test_greedy_query_groups_are_maximal_bounded_and_nonoverlapping(tmp_path: Path) -> None:
    annotations = _annotations(
        tmp_path,
        rows=(
            _row("trajectory-a", "video-a.mp4", [10.0, 30.0, 70.0, 150.0, 160.0, 240.0]),
            _row("trajectory-b", "video-b.mp4", [10.0, 20.0]),
            _row("trajectory-c", "video-c.mp4", [10.0, 20.0]),
            _row("trajectory-d", "video-d.mp4", [10.0, 20.0]),
            _row("trajectory-e", "video-e.mp4", [10.0, 20.0]),
        ),
    )
    records = tuple(
        record for record in annotations.records if record.identity.trajectory_id == "trajectory-a"
    )
    groups = greedy_nonoverlap_query_groups(records)

    assert [[item.query_time for item in group] for group in groups] == [
        [10.0, 30.0, 70.0],
        [150.0, 160.0],
    ]
    flattened = [item.identity.query_id for group in groups for item in group]
    assert len(flattened) == len(set(flattened))


def test_production_manifest_has_fold0_buckets_padding_and_explicit_failures(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _row(f"trajectory-{index}", f"video-{index}.mp4", [50.0, 70.0]) for index in range(5)
    )
    annotations = _annotations(tmp_path, rows=rows)
    durations = {f"Demo/video-{index}.mp4": 100.0 for index in range(5)}
    manifest = build_production_episode_manifest(
        annotations,
        video_durations=durations,
    )

    real = tuple(episode for episode in manifest.episodes if episode.loss_weight == 1.0)
    padding = tuple(episode for episode in manifest.episodes if episode.loss_weight == 0.0)
    assert len(manifest.a2_query_ids) == 10
    assert len(real) == 5
    assert sum(episode.query_count for episode in real) == 10
    assert len(padding) == 3
    assert all(episode.padding_source_episode_id for episode in padding)
    assert all(len(bucket.episode_ids) % 4 == 0 for bucket in manifest.buckets)
    train_videos = {episode.video_id for episode in real if episode.split is EpisodeSplit.TRAIN}
    validation_videos = {
        episode.video_id for episode in real if episode.split is EpisodeSplit.VALIDATION
    }
    assert len(train_videos) == 4
    assert len(validation_videos) == 1
    assert train_videos.isdisjoint(validation_videos)
    assert all(episode.sampling_weight == 0.2 for episode in real)
    assert all(query.sampling_weight == 0.1 for query in manifest.a2_queries)
    runtime = manifest.a2_queries[0].query.runtime.as_payload()
    assert set(runtime) == {"video", "question", "query_time", "explicit_time_values"}
    assert "count" not in runtime and "answer" not in runtime

    failed_annotations = _annotations(
        tmp_path,
        rows=(*rows, _row("tomato-oob", "tomato.mp4", [20.0, 120.0], source="TOMATO")),
        name="failed.jsonl",
    )
    failed_durations = {**durations, "TOMATO/tomato.mp4": 100.0}
    failed_manifest = build_production_episode_manifest(
        failed_annotations,
        video_durations=failed_durations,
    )
    assert len(failed_manifest.failures) == 1
    assert failed_manifest.failures[0].source_dataset == "TOMATO"
    assert failed_manifest.failures[0].reason == "query_time_exceeds_video_duration"

    output = tmp_path / "output"
    write_production_episode_manifest(
        failed_manifest,
        manifest_path=output / "dataset_manifest.json",
        failed_path=output / "failed.jsonl",
    )
    stored = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    failures = (output / "failed.jsonl").read_text(encoding="utf-8").splitlines()
    assert stored["schema_version"] == "svcbench_a2_a5_v2"
    assert set(stored["a2_queries"][0]["query"]) == {"runtime", "answer", "weak"}
    assert len(failures) == 1
    assert load_production_episode_manifest(output / "dataset_manifest.json") == failed_manifest
    train_view, validation_view = load_production_manifest_views(
        output / "dataset_manifest.json",
        stage=ManifestStage.A5,
    )
    assert train_view.manifest is validation_view.manifest
    assert train_view.split is EpisodeSplit.TRAIN
    assert validation_view.split is EpisodeSplit.VALIDATION
    assert all(record.split is EpisodeSplit.TRAIN for record in train_view.records)
    assert all(record.split is EpisodeSplit.VALIDATION for record in validation_view.records)

    stored["a2_queries"][0]["query"]["runtime"]["count"] = 99
    leaked = output / "leaked_manifest.json"
    leaked.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime Query keys drifted"):
        load_production_episode_manifest(leaked)


def test_remote_query_video_mapping_uses_each_a2_clip_and_latest_a5_clip(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _row(f"trajectory-{index}", f"source-{index}.mp4", [10.0, 20.0]) for index in range(5)
    )
    annotations = _annotations(tmp_path, rows=rows, name="mapped.jsonl")
    durations = {f"Demo/source-{index}.mp4": 30.0 for index in range(5)}
    runtime_paths = {
        record.identity.query_id: f"videos/{flat_index:04d}.mp4"
        for flat_index, record in enumerate(annotations.records)
    }

    manifest = build_production_episode_manifest(
        annotations,
        video_durations=durations,
        runtime_video_paths=runtime_paths,
    )

    assert tuple(record.relative_video_path for record in manifest.a2_queries) == tuple(
        runtime_paths[record.identity.query_id] for record in annotations.records
    )
    real_episodes = tuple(row for row in manifest.episodes if row.loss_weight == 1.0)
    for episode in real_episodes:
        final_query_id = episode.queries[-1].runtime.query_id
        assert episode.relative_video_path == runtime_paths[final_query_id]

    incomplete = dict(runtime_paths)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="cover every annotation Query"):
        build_production_episode_manifest(
            annotations,
            video_durations=durations,
            runtime_video_paths=incomplete,
        )


def test_distributed_manifest_samplers_balance_a2_and_align_a5_segments(tmp_path: Path) -> None:
    task_rows = []
    task_specs = (
        ("O1", "O1-Snap"),
        ("O2", "O2-Unique"),
        ("E1", "E1-Action"),
        ("E2", "E2-Periodic"),
    )
    for index in range(20):
        counting_type, subtype = task_specs[index % len(task_specs)]
        task_rows.append(
            _row(
                f"task-trajectory-{index}",
                f"task-video-{index}.mp4",
                [50.0, 70.0],
                counting_type=counting_type,
                counting_subtype=subtype,
            )
        )
    annotations = _annotations(tmp_path, rows=tuple(task_rows), name="tasks.jsonl")
    durations = {f"Demo/task-video-{index}.mp4": 100.0 for index in range(20)}
    manifest = build_production_episode_manifest(annotations, video_durations=durations)

    a2_dataset = ProductionManifestDataset(
        manifest,
        stage=ManifestStage.A2,
        split=EpisodeSplit.TRAIN,
    )
    a2_samplers = [
        BalancedA2DistributedSampler(a2_dataset, rank=rank, world_size=4) for rank in range(4)
    ]
    global_a2_indices = [list(sampler) for sampler in a2_samplers]
    assert all(indices == global_a2_indices[0] for indices in global_a2_indices[1:])
    global_a2 = [a2_dataset[index] for index in global_a2_indices[0]]
    counts = Counter(record.task_class for record in global_a2)
    assert len(set(counts.values())) == 1
    local_a2_indices = [global_a2_indices[0][rank::4] for rank in range(4)]
    for step in range(len(local_a2_indices[0])):
        rows = [a2_dataset[local_a2_indices[rank][step]] for rank in range(4)]
        assert len({row.task_class for row in rows}) == 1
        assert (
            len({len(adaptive_support_schedule(row.query.runtime.query_time)[1]) for row in rows})
            == 1
        )

    visual_value = {
        record.query.runtime.query_id: index for index, record in enumerate(a2_dataset.records)
    }
    visual_sampler = BalancedA2DistributedSampler(
        a2_dataset,
        rank=0,
        world_size=4,
        visual_length_fn=lambda record: visual_value[record.query.runtime.query_id],
    )
    visual_indices = list(visual_sampler)
    for start in range(0, len(visual_indices), 4):
        batch = visual_indices[start : start + 4]
        first = a2_dataset[batch[0]]
        support_count = len(adaptive_support_schedule(first.query.runtime.query_time)[1])
        bucket = [
            index
            for index, record in enumerate(a2_dataset.records)
            if record.task_class == first.task_class
            and len(adaptive_support_schedule(record.query.runtime.query_time)[1]) == support_count
        ]
        ordered = sorted(
            bucket,
            key=lambda index: visual_value[a2_dataset[index].query.runtime.query_id],
        )
        positions = {index: position for position, index in enumerate(ordered)}
        assert (
            max(positions[index] for index in batch) - min(positions[index] for index in batch) < 4
        )
    visual_sampler.set_epoch(1)
    assert list(visual_sampler) != visual_indices

    a5_dataset = ProductionManifestDataset(
        manifest,
        stage=ManifestStage.A5,
        split=EpisodeSplit.TRAIN,
    )
    global_a5_indices = [
        list(RankAlignedA5SegmentSampler(a5_dataset, rank=rank, world_size=4)) for rank in range(4)
    ]
    assert all(indices == global_a5_indices[0] for indices in global_a5_indices[1:])
    local_indices = [global_a5_indices[0][rank::4] for rank in range(4)]
    assert len({len(values) for values in local_indices}) == 1
    for step in range(len(local_indices[0])):
        rows = [a5_dataset[local_indices[rank][step]] for rank in range(4)]
        segment_counts = {row.tbptt_segment_count for row in rows}
        shapes = {
            (
                tuple(
                    min(row.truncation_horizon, row.support_count - start)
                    for start in range(0, row.support_count, row.truncation_horizon)
                ),
                row.query_count,
            )
            for row in rows
        }
        assert len(segment_counts) == 1
        assert len(shapes) == 1


def _annotations(
    tmp_path: Path,
    *,
    rows: tuple[dict[str, object], ...],
    name: str = "annotations.jsonl",
):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return load_annotations(
        path,
        source=DatasetSource("fixture", "revision-1", False),
        purpose=DatasetPurpose.TRAINING,
    )


def _row(
    trajectory_id: str,
    video_path: str,
    times: list[float],
    *,
    source: str = "Demo",
    counting_type: str = "O1",
    counting_subtype: str = "O1-Snap",
) -> dict[str, object]:
    return {
        "id": trajectory_id,
        "source_dataset": source,
        "video_path": video_path,
        "question": "How many objects are visible now?",
        "counting_type": counting_type,
        "counting_subtype": counting_subtype,
        "occurrence_times": [1.0],
        "query_points": {"time": times, "count": list(range(1, len(times) + 1))},
    }
