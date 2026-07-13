"""Define spatial-slot and temporal-causal encoder output/cache contracts.

Inputs: adapted visual tokens, q_target, masks, timestamps, and prior per-video state.
Outputs: A_t [B, 32, 768], H_t [B, T, 768], and a bounded causal cache.
Forbidden: hard counting, Reader arithmetic, Bank mutation, or online optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class SpatialEncoderOutput:
    slots: Tensor
    slot_valid_mask: Tensor
    active_slot_overflow_count: Tensor

    def __post_init__(self) -> None:
        if self.slots.ndim != 3 or self.slots.shape[-1] != 768:
            raise ValueError("slots must be [B, K_a, 768]")
        if not torch.is_floating_point(self.slots):
            raise TypeError("slots must use a floating dtype")
        expected_mask = self.slots.shape[:2]
        if (
            self.slot_valid_mask.shape != expected_mask
            or self.slot_valid_mask.dtype != torch.bool
        ):
            raise ValueError("slot_valid_mask must be bool [B, K_a]")
        if self.active_slot_overflow_count.shape != (self.slots.shape[0],):
            raise ValueError("active_slot_overflow_count must be [B]")
        if self.active_slot_overflow_count.dtype not in (torch.int32, torch.int64):
            raise TypeError("active_slot_overflow_count must use an integer dtype")


@dataclass(frozen=True, slots=True)
class TemporalCache:
    hidden: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3 or self.hidden.shape[-1] != 768:
            raise ValueError("temporal cache hidden must be [B, T_cache, 768]")
        if self.hidden.shape[1] > 64:
            raise ValueError("temporal cache cannot exceed 64 tubelets")
        shape = self.hidden.shape[:2]
        if self.timestamps.shape != shape or not torch.is_floating_point(self.timestamps):
            raise ValueError("temporal cache timestamps must be floating [B, T_cache]")
        if self.valid_mask.shape != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("temporal cache valid_mask must be bool [B, T_cache]")
        if len(self.video_ids) != self.hidden.shape[0] or not all(self.video_ids):
            raise ValueError("temporal cache requires one non-empty video_id per batch item")


@dataclass(frozen=True, slots=True)
class TemporalEncoderOutput:
    hidden: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    cache: TemporalCache

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3 or self.hidden.shape[-1] != 768:
            raise ValueError("temporal hidden must be [B, T, 768]")
        shape = self.hidden.shape[:2]
        if self.timestamps.shape != shape or not torch.is_floating_point(self.timestamps):
            raise ValueError("temporal timestamps must be floating [B, T]")
        if self.valid_mask.shape != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("temporal valid_mask must be bool [B, T]")


def build_state_encoders(_config: ProjectConfig | None = None) -> NoReturn:
    """P6 and P7 own the spatial and temporal implementations."""

    raise NotImplementedError("State encoder implementation is deferred to P6-P7")
