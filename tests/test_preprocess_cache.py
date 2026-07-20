from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

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
    query = build_fingerprint(
        source_dataset="svcbench",
        relative_video_path="clip.mp4",
        video_path=video,
        start_time=0.0,
        end_time=1.0,
        maximum_frames=4,
        sample_fps=2.0,
        minimum_pixels=256,
        maximum_pixels=4096,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        transformers_version="4.57.1",
        observation_role="query",
        frame_sampling="llamafactory_uniform_cap",
    )

    assert support.digest != query.digest


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
