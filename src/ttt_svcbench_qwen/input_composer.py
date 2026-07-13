"""Define the one-time Qwen prefill payload contract.

Inputs: question/chat tokens, adapted video embeddings, 16 State Tokens, and Reader number IDs.
Outputs: embeddings, attention/position/cache metadata, and exact placement masks.
Forbidden: State/TTT updates, decode-loop recomposition, Reader arithmetic, or DeepStack remapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ComposedInput:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    cache_position: Tensor
    video_position_mask: Tensor
    state_position_mask: Tensor
    number_token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        embeds = self.inputs_embeds
        if embeds.ndim != 3 or embeds.shape[-1] != 4096 or not torch.is_floating_point(embeds):
            raise ValueError("inputs_embeds must be floating [B, L, 4096]")
        batch_size, sequence_length = embeds.shape[:2]
        if self.attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must be [B, L]")
        if self.position_ids.shape[-2:] != (batch_size, sequence_length):
            raise ValueError("position_ids must end in [B, L]")
        for mask, name in (
            (self.video_position_mask, "video_position_mask"),
            (self.state_position_mask, "state_position_mask"),
        ):
            if mask.shape != (batch_size, sequence_length) or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be bool [B, L]")
        if torch.any(self.video_position_mask & self.state_position_mask):
            raise ValueError("State Token positions cannot be marked as visual positions")
        if len(self.number_token_ids) != batch_size:
            raise ValueError("number_token_ids must contain one sequence per batch item")


def compose_inputs(*_args: object, **_kwargs: object) -> NoReturn:
    """P13 owns placeholder scatter and all mask/position/cache updates."""

    raise NotImplementedError("Input composition is deferred to P13")
