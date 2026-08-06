from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.production_runtime import SupportChunkSpec, VideoChunkMaterializer
from ttt_svcbench_qwen.video_preprocessing import QwenVideoPreprocessor


def test_real_qwen_processor_demo_contract() -> None:
    processor = QwenVideoPreprocessor(load_config())
    demo = torch.zeros((1, 16, 3, 224, 224), dtype=torch.uint8)
    processed = processor.process(demo)

    assert demo.shape == (1, 16, 3, 224, 224)
    assert processed.video_grid_thw.tolist() == [[8, 14, 14]]
    assert processed.pixel_values_videos.shape == (1, 1568, 1536)
    assert processed.flatten_for_qwen().shape == (1568, 1536)


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
