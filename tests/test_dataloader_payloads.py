from __future__ import annotations

from collections import deque
from pathlib import Path

import av
import pytest
import torch

from ttt_svcbench_qwen.production_runtime import (
    CurrentChunkMaterialization,
    PreparedVisualCPU,
    QueryObservationSpec,
    SupportChunkSpec,
    VideoChunkMaterializer,
    _compact_materialized_chunk,
    _decode_coalesced_intervals,
    _decode_uniform_interval,
)


def _specs(tmp_path: Path, count: int = 4) -> tuple[SupportChunkSpec, ...]:
    path = tmp_path / "clip.mp4"
    path.touch()
    return tuple(
        SupportChunkSpec(
            chunk_id=f"chunk-{index}",
            video_path=path,
            start_time=float(index),
            end_time=float(index + 1),
            maximum_frames=2,
            query_time=float(count + 2),
        )
        for index in range(count)
    )


def test_compact_worker_payload_drops_raw_rgb_frames(tmp_path: Path) -> None:
    spec = _specs(tmp_path, 1)[0]
    materialized = CurrentChunkMaterialization(
        spec=spec,
        frames=torch.zeros((2, 3, 64, 64), dtype=torch.uint8),
        frame_timestamps=torch.tensor([0.0, 0.5], dtype=torch.float64),
        tubelet_timestamps=torch.tensor([[0.5]], dtype=torch.float64),
        tubelet_valid_mask=torch.ones((1, 1), dtype=torch.bool),
        tubelet_position_ids=torch.zeros((1, 1), dtype=torch.int64),
        pixel_values_videos=torch.zeros((16, 1536), dtype=torch.float32),
        video_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int64),
    )

    compact = _compact_materialized_chunk(materialized)

    assert isinstance(compact, PreparedVisualCPU)
    assert not hasattr(compact, "frames")
    assert compact.frame_count == 2
    assert compact.patch_count == 16
    assert torch.equal(compact.pixel_values_videos, materialized.pixel_values_videos)
    assert torch.equal(compact.video_grid_thw, materialized.video_grid_thw)


def test_bounded_prefetch_preserves_order_and_depth(tmp_path: Path) -> None:
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.prefetch_depth = 2
    materializer.decode_coalesce = False
    materializer.preprocess_cache = None
    materializer._executor = None
    materializer._pending_queue = deque()
    materializer._remaining_specs = deque()
    calls: list[str] = []

    def fake_materialize(spec: SupportChunkSpec) -> SupportChunkSpec:
        calls.append(spec.chunk_id)
        return spec

    materializer._materialize = fake_materialize  # type: ignore[method-assign]
    specs = _specs(tmp_path)
    materializer.begin_prefetch(specs)
    assert len(materializer._pending_queue) == 2
    assert [materializer(spec).chunk_id for spec in specs] == [spec.chunk_id for spec in specs]
    assert calls == [spec.chunk_id for spec in specs]
    materializer.end_prefetch()
    materializer._executor.shutdown(wait=True)


def test_prefetch_rejects_out_of_order_consumption(tmp_path: Path) -> None:
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.prefetch_depth = 2
    materializer.decode_coalesce = False
    materializer.preprocess_cache = None
    materializer._executor = None
    materializer._pending_queue = deque()
    materializer._remaining_specs = deque()
    materializer._materialize = lambda spec: spec  # type: ignore[method-assign]
    specs = _specs(tmp_path, 2)
    materializer.begin_prefetch(specs)
    with pytest.raises(RuntimeError, match="out of prefetch order"):
        materializer(specs[1])
    materializer.end_prefetch()
    materializer._executor.shutdown(wait=True)


def test_disabled_query_cache_does_not_build_a_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "clip.mp4"
    path.touch()
    spec = QueryObservationSpec(
        "state-query",
        path,
        0.0,
        4.0,
        16,
        4.0,
        query_role="state_query",
    )
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.config = object()  # Query FPS is owned by the spec, not the project config.
    materializer._cache_for = lambda _spec: None  # type: ignore[method-assign]
    materializer._fingerprint = lambda _spec: pytest.fail(  # type: ignore[method-assign]
        "disabled Query cache must not construct a fingerprint"
    )
    sentinel = object()

    def materialize_decoded(
        _spec: object,
        _frames: torch.Tensor,
        _timestamps: torch.Tensor,
        fingerprint: object,
        *,
        cache: object,
    ) -> object:
        assert fingerprint is None
        assert cache is None
        return sentinel

    materializer._materialize_decoded = materialize_decoded  # type: ignore[method-assign]
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._decode_uniform_interval",
        lambda _spec, _fps: (
            torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
            torch.tensor((0.0, 1.0), dtype=torch.float64),
        ),
    )

    assert materializer._materialize(spec) is sentinel


def test_disabled_support_cache_bypasses_shared_cache(tmp_path: Path) -> None:
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.cache_support_visuals = False
    materializer.preprocess_cache = object()

    assert materializer._cache_for(_specs(tmp_path, 1)[0]) is None


def test_support_group_preserves_requested_order(tmp_path: Path) -> None:
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    specs = _specs(tmp_path, 4)
    sentinels = {spec.chunk_id: object() for spec in specs}
    materializer._materialize_group = lambda values: {  # type: ignore[method-assign]
        value.chunk_id: sentinels[value.chunk_id] for value in values
    }

    assert materializer.materialize_support_group(specs) == tuple(
        sentinels[spec.chunk_id] for spec in specs
    )


def test_coalesced_decode_matches_individual_decode(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for index in range(40):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), index, dtype=torch.uint8).numpy(), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    specs = tuple(
        SupportChunkSpec(f"c{index}", path, float(index), float(index + 2), 8, 20.0)
        for index in range(3)
    )
    coalesced = _decode_coalesced_intervals(specs, 2.0)
    for spec in specs:
        individual = _decode_uniform_interval(spec, 2.0)
        actual = coalesced[spec.chunk_id]
        assert torch.equal(individual[0], actual[0])
        assert torch.equal(individual[1], actual[1])
