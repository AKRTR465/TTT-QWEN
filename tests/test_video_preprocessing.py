from __future__ import annotations

from collections import deque
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.production_runtime import SupportChunkSpec, VideoChunkMaterializer
from ttt_svcbench_qwen.video_preprocessing import (
    QwenVideoPreprocessor,
    build_demo_video,
    causal_right_cut,
    chunk_causal_cut,
    decode_video_causally,
)


def _write_tiny_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 4)
        for index in range(8):
            array = np.full((32, 32, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_qwen_processor_demo_contract() -> None:
    processor = QwenVideoPreprocessor(load_config())
    demo = build_demo_video()
    processed = processor.process(demo)

    assert demo.shape == (1, 16, 3, 224, 224)
    assert processed.video_grid_thw.tolist() == [[8, 14, 14]]
    assert processed.pixel_values_videos.shape == (1, 1568, 1536)
    assert processed.flatten_for_qwen().shape == (1568, 1536)


def test_real_qwen_processor_derives_variable_shape_from_grid() -> None:
    processor = QwenVideoPreprocessor(load_config())
    processed = processor.process(torch.zeros(7, 3, 160, 288, dtype=torch.uint8))

    assert processed.video_grid_thw.tolist() == [[4, 10, 18]]
    assert processed.pixel_values_videos.shape == (1, 720, 1536)
    assert 720 == 4 * 10 * 18


def test_right_closed_cut_includes_boundary_and_excludes_every_future_frame() -> None:
    frames = torch.arange(6 * 3 * 2 * 2).reshape(6, 3, 2, 2)
    timestamps = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cut = causal_right_cut(frames, timestamps, query_time=1.5)

    assert cut.frames.shape[0] == 4
    assert cut.timestamps.tolist() == [0.0, 0.5, 1.0, 1.5]
    assert cut.max_visible_time == 1.5
    assert torch.all(cut.timestamps <= cut.query_time)


def test_real_tiny_video_is_sampled_and_cut_before_future_frames(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    _write_tiny_video(path)

    decoded = decode_video_causally(path, query_time=1.0, sample_fps=2.0)

    assert path.stat().st_size < 2_000
    assert decoded.frames.shape == (3, 3, 32, 32)
    assert decoded.timestamps.tolist() == [0.0, 0.5, 1.0]
    assert decoded.source_fps == 4.0
    assert torch.all(decoded.timestamps <= 1.0)


def test_support_materializer_prefetches_in_order_with_bounded_queue(tmp_path: Path) -> None:
    path = tmp_path / "placeholder.mp4"
    path.touch()
    specs = tuple(
        SupportChunkSpec(
            chunk_id=f"chunk-{index}",
            video_path=path,
            start_time=float(index),
            end_time=float(index + 1),
            maximum_frames=2,
            query_time=3.0,
        )
        for index in range(3)
    )
    materializer = VideoChunkMaterializer.__new__(VideoChunkMaterializer)
    materializer.prefetch_depth = 2
    materializer.decode_coalesce = False
    materializer._executor = None
    materializer._pending_queue = deque()
    materializer._remaining_specs = deque()
    calls: list[str] = []

    def fake_materialize(spec: SupportChunkSpec) -> SupportChunkSpec:
        calls.append(spec.chunk_id)
        return spec

    materializer._materialize = fake_materialize  # type: ignore[method-assign]
    try:
        materializer.begin_prefetch(specs)
        assert [entry[0] for entry in materializer._pending_queue] == list(specs[:2])
        assert materializer(specs[0]) == specs[0]
        assert [entry[0] for entry in materializer._pending_queue] == list(specs[1:3])
        assert materializer(specs[1]) == specs[1]
        assert [entry[0] for entry in materializer._pending_queue] == [specs[2]]
        assert materializer(specs[2]) == specs[2]
        assert not materializer._pending_queue
        assert calls == [spec.chunk_id for spec in specs]
    finally:
        materializer.end_prefetch()
        if materializer._executor is not None:
            materializer._executor.shutdown(wait=True)


def test_overlapping_chunks_audit_tail_padding_tubelets_and_alignment() -> None:
    config = load_config().video_preprocessing
    frames = torch.zeros(20, 3, 4, 4, dtype=torch.uint8)
    timestamps = torch.arange(20, dtype=torch.float64) / 2.0
    cut = causal_right_cut(frames, timestamps, query_time=9.0)
    result = chunk_causal_cut(cut, config)

    assert len(result.chunks) == 2
    first, tail = result.chunks
    assert first.frame_valid_mask.sum().item() == 16
    assert tail.frame_valid_mask.sum().item() == 11
    assert tail.original_frame_indices.tolist()[-5:] == [-1, -1, -1, -1, -1]
    assert tail.tubelet_frame_counts.tolist() == [2, 2, 2, 2, 2, 1, 0, 0]
    assert tail.tubelet_valid_mask.tolist() == [True, True, True, True, True, False, False, False]
    assert tail.overlap_with_previous == ((4, 0), (5, 1), (6, 2), (7, 3))
    assert result.max_visible_time == 9.0
    assert all(
        chunk.chunk_end_time is None or chunk.chunk_end_time <= result.query_time
        for chunk in result.chunks
    )


def test_empty_short_and_multiple_query_points_remain_auditable() -> None:
    config = load_config().video_preprocessing
    frames = torch.zeros(3, 3, 4, 4, dtype=torch.uint8)
    timestamps = torch.tensor([0.0, 0.5, 1.0])

    empty = chunk_causal_cut(causal_right_cut(frames, timestamps, -0.0), config)
    assert empty.chunks[0].frame_valid_mask.sum().item() == 1
    assert not empty.chunks[0].tubelet_valid_mask.any()

    before_first = chunk_causal_cut(
        causal_right_cut(frames, timestamps + 0.5, query_time=0.0), config
    )
    assert before_first.max_visible_time is None
    assert before_first.chunks[0].frame_valid_mask.sum().item() == 0

    for query_time, expected_count in ((0.0, 1), (0.5, 2), (1.0, 3)):
        result = chunk_causal_cut(causal_right_cut(frames, timestamps, query_time), config)
        assert result.chunks[0].frame_valid_mask.sum().item() == expected_count
        assert result.max_visible_time is not None and result.max_visible_time <= query_time
