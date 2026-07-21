from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from scripts.preprocess_cache import (
    _fingerprinted_specs,
    _iter_specs,
    _load_training_config,
)

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    AnswerSupervisionSidecar,
    EpisodeSplit,
    ProductionQueryRecord,
    WeakQuerySidecar,
)
from ttt_svcbench_qwen.preprocess_cache import (
    CachedChunk,
    PreprocessCache,
    PreprocessCacheMissError,
    build_fingerprint,
)


def _fingerprint(video: Path, *, end: float = 1.0):
    return build_fingerprint(
        source_dataset="svcbench",
        relative_video_path="clip.mp4",
        video_path=video,
        start_time=0.0,
        end_time=end,
        maximum_frames=4,
        sample_fps=2.0,
        minimum_pixels=256,
        maximum_pixels=4096,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        transformers_version="4.57.1",
    )


def _chunk() -> CachedChunk:
    return CachedChunk(
        frames=torch.zeros((4, 3, 8, 8), dtype=torch.uint8),
        frame_timestamps=torch.arange(4, dtype=torch.float64),
        pixel_values_videos=torch.zeros((8, 1536), dtype=torch.float32),
        video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
        tubelet_timestamps=torch.tensor([[1.0, 3.0]], dtype=torch.float64),
        tubelet_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        tubelet_position_ids=torch.tensor([[0, 1]], dtype=torch.int64),
    )


def test_cache_roundtrip_and_stable_key(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    first = _fingerprint(video)
    second = _fingerprint(video)
    assert first.digest == second.digest
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0, namespace="model-a")
    cache.put(first, _chunk())
    path = cache._path_for(first)
    assert path is not None
    assert cache.payload_size(first) == path.stat().st_size
    loaded = cache.get(second)
    assert loaded is not None
    assert torch.equal(loaded.frames, _chunk().frames)
    assert torch.equal(loaded.pixel_values_videos, _chunk().pixel_values_videos)
    assert list((tmp_path / "cache" / "model-a").rglob("*.json"))


def test_query_role_and_sampling_policy_cannot_reuse_support_cache_key(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    support = _fingerprint(video)
    values = {
        "source_dataset": "svcbench",
        "relative_video_path": "clip.mp4",
        "video_path": video,
        "start_time": 0.0,
        "end_time": 1.0,
        "maximum_frames": 4,
        "sample_fps": 2.0,
        "minimum_pixels": 256,
        "maximum_pixels": 4096,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "transformers_version": "4.57.1",
        "frame_sampling": "llamafactory_uniform_cap",
    }
    state_query = build_fingerprint(**values, observation_role="state_query")
    answer_query = build_fingerprint(**values, observation_role="answer_query")

    assert len({support.digest, state_query.digest, answer_query.digest}) == 3
    with pytest.raises(ValueError, match="observation role"):
        build_fingerprint(**values, observation_role="query")


def test_cache_invalidates_media_and_metadata(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fingerprint = _fingerprint(video)
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0)
    cache.put(fingerprint, _chunk())
    video.write_bytes(b"changed-video")
    assert cache.get(_fingerprint(video)) is None

    # Restore the original key and corrupt only the JSON sidecar.  The embedded tensor metadata
    # remains intact, but a mismatched sidecar must conservatively force a miss.
    video.write_bytes(b"video")
    restored = _fingerprint(video)
    cache.put(restored, _chunk())
    path = cache._path_for(restored)
    assert path is not None
    path.with_suffix(".json").write_text(json.dumps({"fingerprint": "wrong"}), encoding="utf-8")
    assert cache.get(restored) is None


def test_cache_prune_removes_oldest_entries(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0, max_bytes=10**9)
    cache.put(_fingerprint(video), _chunk())
    assert cache.disk_size_bytes() > 0
    cache.max_bytes = 1
    assert cache.prune() >= 1
    assert cache.disk_size_bytes() == 0


def test_readonly_cache_never_writes_or_updates_atime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fingerprint = _fingerprint(video)
    root = tmp_path / "cache"
    PreprocessCache(root, memory_entries=0).put(fingerprint, _chunk())
    cache = PreprocessCache(root, mode="readonly", memory_entries=0)

    monkeypatch.setattr(os, "utime", lambda *_args, **_kwargs: pytest.fail("atime write"))
    assert cache.get(fingerprint) is not None
    with pytest.raises(PermissionError, match="readonly"):
        cache.put(fingerprint, _chunk())
    with pytest.raises(PermissionError, match="read_write"):
        cache.prune()


def test_strict_readonly_cache_raises_on_miss(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    root = tmp_path / "cache"
    root.mkdir()
    cache = PreprocessCache(root, mode="readonly", miss_policy="error", memory_entries=0)

    with pytest.raises(PreprocessCacheMissError, match="entry_missing"):
        cache.get(_fingerprint(video))


def test_put_does_not_scan_or_prune_cache_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0)
    monkeypatch.setattr(
        cache,
        "disk_size_bytes",
        lambda: pytest.fail("hot-path capacity scan"),
    )
    monkeypatch.setattr(cache, "prune", lambda: pytest.fail("hot-path prune"))

    cache.put(_fingerprint(video), _chunk())


def test_prewarm_enumerates_distinct_state_and_answer_query_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("SVCBENCH_VIDEO_ROOT", str(tmp_path))
    runtime = RuntimeQueryInput(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_id="query-a",
        query_index=0,
        video=video,
        question="How many?",
        query_time=4.0,
        explicit_time_values=(),
    )
    query = ProductionQueryRecord(
        runtime=runtime,
        answer=AnswerSupervisionSidecar("query-a", "1", "official_explicit"),
        weak=WeakQuerySidecar(
            query_id="query-a",
            query_index=0,
            query_time=4.0,
            count=1,
            counting_type="O1",
            counting_subtype="O1-Snap",
            operator="o1-snap",
            time_mode="now",
            occurrence_points=(),
            occurrence_intervals=(),
        ),
    )
    record = A2QueryRecord(
        source_dataset="svcbench",
        relative_video_path="clip.mp4",
        video_id="video-a",
        trajectory_id="trajectory-a",
        split=EpisodeSplit.TRAIN,
        task_class="O1",
        query=query,
        sampling_weight=1.0,
    )
    root = Path(__file__).parents[1]
    ttt_config = _load_training_config(
        root / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml"
    )

    candidates = tuple(_iter_specs((record,), ttt_config))
    query_specs = [spec for spec, _source in candidates if hasattr(spec, "query_role")]
    assert [(spec.query_role, spec.maximum_frames) for spec in query_specs] == [
        ("state_query", 16),
        ("answer_query", 256),
    ]
    fingerprinted = _fingerprinted_specs(
        candidates,
        config=load_config(),
        minimum_pixels=256,
        maximum_pixels=131_072,
    )
    query_fingerprints = {
        fingerprint.observation_role: fingerprint.digest
        for _spec, _source, fingerprint in fingerprinted
        if fingerprint.observation_role != "support"
    }
    assert set(query_fingerprints) == {"state_query", "answer_query"}
    assert len(set(query_fingerprints.values())) == 2
