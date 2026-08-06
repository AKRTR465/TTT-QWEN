from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from ttt_svcbench_qwen.preprocess_cache import (
    CachedChunk,
    PreprocessCache,
    _replace_idempotent,
    build_fingerprint,
)


def _fingerprint(video: Path, *, end: float = 1.0, **overrides: str):
    return build_fingerprint(
        **overrides,
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


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    return video


def test_cache_roundtrip_and_stable_key(tmp_path: Path) -> None:
    video = _video(tmp_path)
    first = _fingerprint(video)
    second = _fingerprint(video)
    assert first.digest == second.digest
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0, namespace="model-a")
    cache.put(first, _chunk())
    path = cache._path_for(first)
    assert path is not None
    assert path.is_file()
    loaded = cache.get(second)
    assert loaded is not None
    assert torch.equal(loaded.frames, _chunk().frames)
    assert torch.equal(loaded.pixel_values_videos, _chunk().pixel_values_videos)
    assert list((tmp_path / "cache" / "model-a").rglob("*.safetensors"))


def test_float16_disk_cache_restores_float32_runtime_pixels(tmp_path: Path) -> None:
    fingerprint = _fingerprint(_video(tmp_path))
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
    assert load_file(str(path), device="cpu")["pixel_values_videos"].dtype == torch.float16

    loaded = cache.get(fingerprint)
    assert loaded is not None
    assert loaded.pixel_values_videos.dtype == torch.float32
    assert loaded.pixel_values_videos[0, 0].item() == pytest.approx(0.123456, abs=1.0e-4)


def test_query_roles_cannot_reuse_the_support_cache_key(tmp_path: Path) -> None:
    video = _video(tmp_path)
    query_digests = {
        _fingerprint(
            video,
            observation_role=role,
            frame_sampling="llamafactory_uniform_cap",
        ).digest
        for role in ("state_query", "answer_query")
    }
    assert len({_fingerprint(video).digest, *query_digests}) == 3


def test_cache_invalidates_media_and_embedded_fingerprint(tmp_path: Path) -> None:
    video = _video(tmp_path)
    fingerprint = _fingerprint(video)
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0)
    cache.put(fingerprint, _chunk())
    video.write_bytes(b"changed-video")
    assert cache.get(_fingerprint(video)) is None

    # Relocating a payload under a foreign digest must still miss: the entry carries its own
    # ``__fingerprint_json`` and get() compares it against the requested fingerprint.
    video.write_bytes(b"video")
    restored = _fingerprint(video)
    cache.put(restored, _chunk())
    source = cache._path_for(restored)
    other = cache._path_for(_fingerprint(video, end=2.0))
    assert source is not None and other is not None
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(source.read_bytes())
    assert cache.get(_fingerprint(video, end=2.0)) is None


def test_repeated_put_publishes_one_readable_entry(tmp_path: Path) -> None:
    # Several ranks prewarm the same fingerprint concurrently.  Entries are immutable by
    # fingerprint, so a duplicate publish must leave exactly one readable payload.
    fingerprint = _fingerprint(_video(tmp_path))
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0)
    cache.put(fingerprint, _chunk())
    cache.put(fingerprint, _chunk())
    path = cache._path_for(fingerprint)
    assert path is not None
    assert len(list((tmp_path / "cache").rglob("*.safetensors"))) == 1

    temporary = path.with_name("rank-two.tmp")
    temporary.write_bytes(path.read_bytes())
    _replace_idempotent(temporary, path)
    assert not temporary.exists()
    assert cache.get(fingerprint) is not None


def test_cache_prune_removes_oldest_entries(tmp_path: Path) -> None:
    video = _video(tmp_path)
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0, max_bytes=10**9)
    cache.put(_fingerprint(video), _chunk())
    assert cache.disk_size_bytes() > 0
    cache.max_bytes = 1
    assert cache.prune() >= 1
    assert cache.disk_size_bytes() == 0


def test_readonly_cache_reads_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = _fingerprint(_video(tmp_path))
    root = tmp_path / "cache"
    PreprocessCache(root, memory_entries=0).put(fingerprint, _chunk())
    cache = PreprocessCache(root, mode="readonly", memory_entries=0)

    monkeypatch.setattr(os, "utime", lambda *_args, **_kwargs: pytest.fail("atime write"))
    assert cache.get(fingerprint) is not None
    before = sorted(path.stat().st_mtime_ns for path in root.rglob("*.safetensors"))
    cache.put(fingerprint, _chunk())
    assert cache.prune() == 0
    assert sorted(path.stat().st_mtime_ns for path in root.rglob("*.safetensors")) == before


def test_put_does_not_scan_or_prune_cache_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = PreprocessCache(tmp_path / "cache", memory_entries=0)
    monkeypatch.setattr(cache, "disk_size_bytes", lambda: pytest.fail("hot-path scan"))
    monkeypatch.setattr(cache, "prune", lambda: pytest.fail("hot-path prune"))
    cache.put(_fingerprint(_video(tmp_path)), _chunk())
