"""Apply the pinned Qwen3-VL video transform.

Inputs: decoded frames and the preprocessing config.
Outputs: project-normalized Qwen tubelet tensors plus grid metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import av
from torch import Tensor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from ttt_svcbench_qwen.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class QwenProcessedVideo:
    pixel_values_videos: Tensor
    video_grid_thw: Tensor

    def flatten_for_qwen(self) -> Tensor:
        """Return the raw 2-D layout expected by the upstream Qwen vision tower."""

        return self.pixel_values_videos.squeeze(0)


class QwenVideoPreprocessor:
    """Pinned wrapper around the checkpoint's real Transformers video processor."""

    def __init__(self, config: ProjectConfig) -> None:
        video = config.video_preprocessing
        self._config = video
        self._processor = Qwen3VLVideoProcessor(
            size={
                "shortest_edge": video.processor_shortest_edge,
                "longest_edge": video.processor_longest_edge,
            },
            patch_size=video.patch_size,
            temporal_patch_size=video.temporal_patch_size,
            merge_size=video.spatial_merge_size,
        )

    def process(self, frames: Tensor) -> QwenProcessedVideo:
        if frames.ndim == 5:
            frames = frames[0]
        output = self._processor(
            videos=[frames],
            do_sample_frames=False,
            return_tensors="pt",
        )
        raw_pixels = cast(Tensor, output["pixel_values_videos"])
        grid = cast(Tensor, output["video_grid_thw"])
        return QwenProcessedVideo(pixel_values_videos=raw_pixels.unsqueeze(0), video_grid_thw=grid)


def av_frame_timestamp(frame: av.VideoFrame) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None or frame.time_base is None:
        raise ValueError("decoded frame has no auditable timestamp")
    return float(frame.pts * frame.time_base)
