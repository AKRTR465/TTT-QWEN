from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

from ttt_svcbench_qwen.data import DatasetSource, load_annotations
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
        rows=(_row("trajectory-a", "video-a.mp4", [10.0, 30.0, 70.0, 150.0, 160.0, 240.0]),),
    )
    groups = greedy_nonoverlap_query_groups(annotations.records)

    assert [[item.query_time for item in group] for group in groups] == [
        [10.0, 30.0, 70.0],
        [150.0, 160.0],
    ]
    flattened = [item.identity.query_id for group in groups for item in group]
    assert len(flattened) == len(set(flattened))


def test_production_manifest_splits_by_video_pads_buckets_and_round_trips(tmp_path: Path) -> None:
    rows = tuple(_row(f"trajectory-{i}", f"video-{i}.mp4", [50.0, 70.0]) for i in range(5))
    annotations = _annotations(tmp_path, rows=rows)
    durations = {f"Demo/video-{i}.mp4": 100.0 for i in range(5)}
    runtime_paths = {
        record.identity.query_id: f"videos/{index:04d}.mp4"
        for index, record in enumerate(annotations.records)
    }
    manifest = build_production_episode_manifest(
        annotations, video_durations=durations, runtime_video_paths=runtime_paths
    )

    real = tuple(episode for episode in manifest.episodes if episode.loss_weight == 1.0)
    padding = tuple(episode for episode in manifest.episodes if episode.loss_weight == 0.0)
    assert len(manifest.a2_query_ids) == 10
    assert (len(real), len(padding)) == (5, 3)
    assert sum(episode.meta_query_count for episode in real) == 10
    # Zero-weight padding keeps every bucket rank-divisible: same backward collectives per rank.
    assert all(len(bucket.episode_ids) % 4 == 0 for bucket in manifest.buckets)
    # GroupKFold by video_id: no video may appear on both sides of the split.
    train_videos = {episode.video_id for episode in real if episode.split is EpisodeSplit.TRAIN}
    val_videos = {episode.video_id for episode in real if episode.split is EpisodeSplit.VALIDATION}
    assert (len(train_videos), len(val_videos)) == (4, 1)
    assert train_videos.isdisjoint(val_videos)
    # Runtime payload stays label-free, and each query keeps its own remote clip.
    payload = manifest.a2_queries[0].query.runtime.as_payload()
    assert set(payload) == {"video", "question", "query_time", "explicit_time_values"}
    assert tuple(record.relative_video_path for record in manifest.a2_queries) == tuple(
        runtime_paths[record.identity.query_id] for record in annotations.records
    )
    assert all(
        episode.relative_video_path == runtime_paths[episode.queries[-1].runtime.query_id]
        for episode in real
    )

    output = tmp_path / "output" / "dataset_manifest.json"
    write_production_episode_manifest(manifest, manifest_path=output)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert set(stored["a2_queries"][0]["query"]) == {"runtime", "answer", "weak"}
    assert load_production_episode_manifest(output) == manifest

    train_view, validation_view = load_production_manifest_views(output, stage=ManifestStage.A5)
    assert train_view.manifest is validation_view.manifest
    assert all(record.split is EpisodeSplit.TRAIN for record in train_view.records)
    assert all(record.split is EpisodeSplit.VALIDATION for record in validation_view.records)


def test_a5_segments_stay_causal_and_bound_truncation_at_eight_chunks(tmp_path: Path) -> None:
    rows = (
        _row("double", "double.mp4", [50.0, 52.0, 70.0]),
        _row("collapsed", "collapsed.mp4", [50.0, 52.0]),
        *tuple(_row(f"regular-{i}", f"regular-{i}.mp4", [50.0, 70.0]) for i in range(3)),
    )
    annotations = _annotations(tmp_path, rows=rows, name="aligned.jsonl")
    durations = {"Demo/double.mp4": 100.0, "Demo/collapsed.mp4": 100.0} | {
        f"Demo/regular-{i}.mp4": 100.0 for i in range(3)
    }
    manifest = build_production_episode_manifest(annotations, video_durations=durations)
    real = {ep.trajectory_id: ep for ep in manifest.episodes if ep.loss_weight == 1.0}

    double = real["double"]
    assert double.segment_lengths == (8, 3)
    assert double.segment_query_counts == (2, 1)
    assert real["collapsed"].segment_lengths == (8,)
    assert real["collapsed"].segment_query_counts == (2,)
    # K=8 truncation horizon bounds every segment.
    assert all(1 <= length <= 8 for episode in real.values() for length in episode.segment_lengths)
    # Causal-prefix masking: no Support chunk may end after the Query it supervises.
    assert all(
        chunk.end_time < segment.queries[0].runtime.query_time
        for segment in double.supervised_segments
        for chunk in segment.supports
    )
    assert sum(episode.meta_query_count for episode in real.values()) == 11


def test_distributed_samplers_balance_a2_tasks_and_keep_rank_parity(tmp_path: Path) -> None:
    specs = (("O1", "O1-Snap"), ("O2", "O2-Unique"), ("E1", "E1-Action"), ("E2", "E2-Periodic"))
    rows = tuple(
        _row(
            f"task-trajectory-{i}",
            f"task-video-{i}.mp4",
            [50.0, 70.0],
            counting_type=specs[i % 4][0],
            counting_subtype=specs[i % 4][1],
        )
        for i in range(20)
    )
    annotations = _annotations(tmp_path, rows=rows, name="tasks.jsonl")
    durations = {f"Demo/task-video-{i}.mp4": 100.0 for i in range(20)}
    manifest = build_production_episode_manifest(annotations, video_durations=durations)

    dataset = ProductionManifestDataset(manifest, stage=ManifestStage.A2, split=EpisodeSplit.TRAIN)
    streams = [
        list(BalancedA2DistributedSampler(dataset, rank=rank, world_size=4)) for rank in range(4)
    ]
    assert all(indices == streams[0] for indices in streams[1:])
    assert len(set(Counter(dataset[index].task_class for index in streams[0]).values())) == 1
    local = [streams[0][rank::4] for rank in range(4)]
    for step in range(len(local[0])):
        rows_at_step = [dataset[local[rank][step]] for rank in range(4)]
        assert len({row.task_class for row in rows_at_step}) == 1
        schedules = {
            len(adaptive_support_schedule(row.query.runtime.query_time)[1]) for row in rows_at_step
        }
        assert len(schedules) == 1

    _assert_a5_rank_parity(manifest, world_size=4)
    manifest_8 = build_production_episode_manifest(
        annotations, video_durations=durations, world_size=8
    )
    assert all(bucket.world_size == 8 for bucket in manifest_8.buckets)
    _assert_a5_rank_parity(manifest_8, world_size=8)


def _assert_a5_rank_parity(manifest, world_size: int) -> None:
    """Zero-weight padding parity: identical step count and shape on every rank."""

    dataset = ProductionManifestDataset(manifest, stage=ManifestStage.A5, split=EpisodeSplit.TRAIN)
    streams = [
        list(RankAlignedA5SegmentSampler(dataset, rank=rank, world_size=world_size))
        for rank in range(world_size)
    ]
    assert all(indices == streams[0] for indices in streams[1:])
    local = [streams[0][rank::world_size] for rank in range(world_size)]
    assert len({len(values) for values in local}) == 1
    for step in range(len(local[0])):
        rows = [dataset[local[rank][step]] for rank in range(world_size)]
        assert len({row.tbptt_segment_count for row in rows}) == 1
        assert len({(row.segment_lengths, row.meta_query_count) for row in rows}) == 1


def _annotations(tmp_path: Path, *, rows: tuple[dict[str, object], ...], name: str = "ann.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return load_annotations(path, source=DatasetSource("fixture", "revision-1"))


def _row(
    trajectory_id: str,
    video_path: str,
    times: list[float],
    *,
    counting_type: str = "O1",
    counting_subtype: str = "O1-Snap",
) -> dict[str, object]:
    return {
        "id": trajectory_id,
        "source_dataset": "Demo",
        "video_path": video_path,
        "question": "How many objects are visible now?",
        "counting_type": counting_type,
        "counting_subtype": counting_subtype,
        "occurrence_times": [1.0],
        "query_points": {"time": times, "count": list(range(1, len(times) + 1))},
    }
