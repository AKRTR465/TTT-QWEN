from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from scripts.preprocess_cache import (
    _fingerprinted_specs,
    _iter_specs,
    _load_training_config,
)

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    A5QueryRole,
    A5SupervisedSegmentRecord,
    AdaptiveChunkSpec,
    AnswerSupervisionSidecar,
    ChunkRole,
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


def test_float16_disk_cache_restores_float32_runtime_pixels(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fingerprint = _fingerprint(video)
    cache = PreprocessCache(
        tmp_path / "cache",
        memory_entries=0,
        namespace="fp16",
        storage_dtype="float16",
    )
    source = _chunk()
    source.pixel_values_videos[0, 0] = 0.123456
    cache.put(fingerprint, source)
    path = cache._path_for(fingerprint)
    assert path is not None
    stored = load_file(str(path), device="cpu")
    assert stored["pixel_values_videos"].dtype == torch.float16

    loaded = cache.get(fingerprint)
    assert loaded is not None
    assert loaded.pixel_values_videos.dtype == torch.float32
    assert loaded.pixel_values_videos[0, 0].item() == pytest.approx(0.123456, abs=1.0e-4)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["storage_dtype"] == "float16"


def test_cache_rejects_storage_dtype_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fingerprint = _fingerprint(video)
    assert fingerprint.cache_schema_version == 1
    root = tmp_path / "cache"
    PreprocessCache(root, memory_entries=0, storage_dtype="float16").put(
        fingerprint,
        _chunk(),
    )
    mismatched = PreprocessCache(root, memory_entries=0, storage_dtype="float32")
    assert mismatched.get(fingerprint) is None


def test_legacy_float32_sidecar_remains_readable(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    fingerprint = _fingerprint(video)
    root = tmp_path / "cache"
    cache = PreprocessCache(root, memory_entries=0, storage_dtype="float32")
    cache.put(fingerprint, _chunk())
    path = cache._path_for(fingerprint)
    assert path is not None
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("storage_dtype")
    sidecar.pop("cache_schema_version")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    loaded = cache.get(fingerprint)
    assert loaded is not None
    assert torch.equal(loaded.pixel_values_videos, _chunk().pixel_values_videos)
    assert PreprocessCache(
        root,
        memory_entries=0,
        storage_dtype="float16",
    ).get(fingerprint) is None


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
    ttt_config = _load_training_config(root / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml")

    candidates = tuple(_iter_specs((record,), ttt_config))
    query_specs = [spec for spec, _source in candidates if hasattr(spec, "query_role")]
    assert [(spec.query_role, spec.maximum_frames) for spec in query_specs] == [
        ("state_query", 16),
        ("answer_query", 256),
    ]

    state_only = tuple(_iter_specs((record,), ttt_config, roles=frozenset(("state_query",))))
    assert len(state_only) == 1
    state_spec, _source = state_only[0]
    assert state_spec.query_role == "state_query"
    assert state_spec.maximum_frames == 16
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


def test_a5_prewarm_binds_each_support_segment_to_its_meta_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("SVCBENCH_VIDEO_ROOT", str(tmp_path))

    def query(query_id: str, query_index: int, query_time: float) -> ProductionQueryRecord:
        runtime = RuntimeQueryInput(
            video_id="video-a",
            trajectory_id="trajectory-a",
            query_id=query_id,
            query_index=query_index,
            video=video,
            question="How many?",
            query_time=query_time,
            explicit_time_values=(),
        )
        return ProductionQueryRecord(
            runtime=runtime,
            answer=AnswerSupervisionSidecar(query_id, "1", "official_explicit"),
            weak=WeakQuerySidecar(
                query_id=query_id,
                query_index=query_index,
                query_time=query_time,
                count=1,
                counting_type="O1",
                counting_subtype="O1-Snap",
                operator="o1-snap",
                time_mode="now",
                occurrence_points=(),
                occurrence_intervals=(),
            ),
        )

    first = query("query-a", 0, 4.0)
    final = query("query-b", 1, 10.0)
    first_supports = (
        AdaptiveChunkSpec(ChunkRole.SUPPORT, 0.5, 2.0, 8),
        AdaptiveChunkSpec(ChunkRole.SUPPORT, 2.0, 3.0, 8),
    )
    final_supports = (
        AdaptiveChunkSpec(ChunkRole.SUPPORT, 4.5, 6.0, 8),
        AdaptiveChunkSpec(ChunkRole.SUPPORT, 6.0, 8.0, 8),
    )
    record = A5EpisodeRecord(
        episode_id="episode-a",
        source_dataset="svcbench",
        relative_video_path="clip.mp4",
        video_id="video-a",
        trajectory_id="trajectory-a",
        split=EpisodeSplit.TRAIN,
        task_class="O1",
        operator="o1-snap",
        prewarm=AdaptiveChunkSpec(ChunkRole.PREWARM, 0.0, 0.5, 8),
        supervised_segments=(
            A5SupervisedSegmentRecord(A5QueryRole.INTERMEDIATE, first_supports, first),
            A5SupervisedSegmentRecord(A5QueryRole.FINAL, final_supports, final),
        ),
        diagnostic_queries=(),
        support_count=4,
        meta_query_count=2,
        diagnostic_query_count=0,
        truncation_horizon=8,
        tbptt_segment_count=2,
        sampling_weight=1.0,
    )
    root = Path(__file__).parents[1]
    ttt_config = _load_training_config(
        root / "configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"
    )

    candidates = tuple(
        _iter_specs((record,), ttt_config, roles=frozenset(("support",)))
    )
    support_specs = [spec for spec, _source in candidates]
    assert [spec.query_time for spec in support_specs] == [4.0, 4.0, 4.0, 10.0, 10.0]
    assert [spec.end_time for spec in support_specs] == [0.5, 2.0, 3.0, 6.0, 8.0]
