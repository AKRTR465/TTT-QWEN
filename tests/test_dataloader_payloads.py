from __future__ import annotations

from collections import deque
from pathlib import Path

import av
import pytest
import torch

from ttt_svcbench_qwen.production_runtime import (
    CurrentChunkSpec,
    VideoChunkMaterializer,
    _decode_coalesced_intervals,
    _decode_uniform_interval,
)


def _specs(tmp_path: Path, count: int = 4) -> tuple[CurrentChunkSpec, ...]:
    path = tmp_path / "clip.mp4"
    path.touch()
    return tuple(
        CurrentChunkSpec(
            chunk_id=f"chunk-{index}",
            video_path=path,
            start_time=float(index),
            end_time=float(index + 1),
            maximum_frames=2,
            query_time=float(count + 2),
        )
        for index in range(count)
    )


def test_bounded_prefetch_preserves_order_and_depth(tmp_path: Path) -> None:
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.prefetch_depth = 2
    materializer.decode_coalesce = False
    materializer.preprocess_cache = None
    materializer._executor = None
    materializer._pending_queue = deque()
    materializer._remaining_specs = deque()
    calls: list[str] = []

    def fake_materialize(spec: CurrentChunkSpec) -> CurrentChunkSpec:
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
        CurrentChunkSpec(
            f"c{index}", path, float(index), float(index + 2), 8, 20.0
        )
        for index in range(3)
    )
    coalesced = _decode_coalesced_intervals(specs, 2.0)
    for spec in specs:
        individual = _decode_uniform_interval(spec, 2.0)
        actual = coalesced[spec.chunk_id]
        assert torch.equal(individual[0], actual[0])
        assert torch.equal(individual[1], actual[1])
