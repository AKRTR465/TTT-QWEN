"""Define the Qwen visual integration boundary and video batch contract.

Inputs: causal video processor tensors and the frozen Qwen checkpoint configuration.
Outputs: Main Visual Merger embeddings plus untouched DeepStack features.
Forbidden: State Bank, query routing, online SGD, or rewriting Qwen blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class VideoBatch:
    """One causal video batch before the Qwen vision tower."""

    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    timestamps: Tensor
    query_time: Tensor
    valid_mask: Tensor
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        pixels = self.pixel_values_videos
        if pixels.ndim != 3 or pixels.shape[-1] != 1536 or not torch.is_floating_point(pixels):
            raise ValueError("pixel_values_videos must be floating [B, N_patch, 1536]")
        batch_size = pixels.shape[0]
        if self.video_grid_thw.shape != (batch_size, 3):
            raise ValueError("video_grid_thw must be [B, 3]")
        if self.video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("video_grid_thw must use an integer dtype")
        if self.timestamps.ndim != 2 or self.timestamps.shape[0] != batch_size:
            raise ValueError("timestamps must be [B, T]")
        if not torch.is_floating_point(self.timestamps):
            raise TypeError("timestamps must use a floating dtype")
        if self.query_time.shape != (batch_size,) or not torch.is_floating_point(self.query_time):
            raise ValueError("query_time must be floating [B]")
        if (
            self.valid_mask.shape != self.timestamps.shape
            or self.valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be bool [B, T]")
        if len(self.video_ids) != batch_size or len(self.trajectory_ids) != batch_size:
            raise ValueError("video_ids and trajectory_ids must contain one value per batch item")
        if not all(self.video_ids) or not all(self.trajectory_ids):
            raise ValueError("video_id and trajectory_id must be non-empty")


@dataclass(frozen=True, slots=True)
class QwenVisualOutput:
    """Visual features at the only supported v5 integration surface."""

    main_visual_embeddings: Tensor
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    video_grid_thw: Tensor

    def __post_init__(self) -> None:
        main = self.main_visual_embeddings
        if main.ndim != 3 or main.shape[-1] != 4096 or not torch.is_floating_point(main):
            raise ValueError("main_visual_embeddings must be floating [B, N_v, 4096]")
        for feature in self.deepstack_features:
            if (
                feature.shape != main.shape
                or feature.dtype != main.dtype
                or feature.device != main.device
            ):
                raise ValueError("each DeepStack feature must match Main Merger shape/dtype/device")


def build_qwen_adapter(_config: ProjectConfig | None = None) -> NoReturn:
    """P3 owns the real Qwen loader and Merger hook."""

    raise NotImplementedError("Qwen adapter implementation is deferred to P3")
