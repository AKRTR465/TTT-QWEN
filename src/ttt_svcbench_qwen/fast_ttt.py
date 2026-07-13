"""Define Fast Adapter weight and optimizer runtime contracts.

Inputs: Main Merger embeddings and per-video W0/W_t fast matrices.
Outputs: adapted embeddings plus auditable fast-version metadata.
Forbidden: optimizer steps, State Bank mutation, query routing, or non-fast online updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class FastWeightsState:
    w0_1: Tensor
    w0_2: Tensor
    w_t_1: Tensor
    w_t_2: Tensor
    fast_version: int
    update_count: int
    skip_count: int

    def __post_init__(self) -> None:
        matrices = (self.w0_1, self.w0_2, self.w_t_1, self.w_t_2)
        for matrix in matrices:
            if matrix.shape != (768, 768) or not torch.is_floating_point(matrix):
                raise ValueError("all fast matrices must be floating [768, 768]")
            if matrix.dtype != self.w0_1.dtype or matrix.device != self.w0_1.device:
                raise ValueError("all fast matrices must share dtype and device")
        if min(self.fast_version, self.update_count, self.skip_count) < 0:
            raise ValueError("fast runtime counters must be non-negative")
        if self.fast_version != self.update_count:
            raise ValueError("fast_version must equal the number of accepted updates")


@dataclass(frozen=True, slots=True)
class OptimizerRuntimeState:
    optimizer_name: str
    learning_rate: float
    momentum: float
    weight_decay: float
    steps_per_chunk: int
    grad_clip_norm: float
    attempted_update_count: int
    last_skip_reason: str | None

    def __post_init__(self) -> None:
        fixed = (
            self.optimizer_name == "sgd"
            and self.learning_rate == 1.0e-4
            and self.momentum == 0.0
            and self.weight_decay == 0.0
            and self.steps_per_chunk == 1
            and self.grad_clip_norm == 1.0
        )
        if not fixed:
            raise ValueError("optimizer runtime must match the frozen single-step SGD contract")
        if self.attempted_update_count < 0:
            raise ValueError("attempted_update_count must be non-negative")


def build_fast_ttt_adapter(_config: ProjectConfig | None = None) -> NoReturn:
    """P5 owns the Adapter layers and fast parameter collection."""

    raise NotImplementedError("Fast TTT Adapter implementation is deferred to P5")
