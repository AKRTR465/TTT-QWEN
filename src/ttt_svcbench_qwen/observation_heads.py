"""Define the four soft observation decoder outputs.

Inputs: spatial slots or causal temporal states plus q_target where specified.
Outputs: O1 [B,K,6], O2 identity/score, E1 [B,T,3], and E2 event/phase.
Forbidden: final integer accumulation, hard FSM mutation, retrieval, or answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


def _require_float_shape(tensor: Tensor, last_dim: int, name: str) -> None:
    if tensor.ndim != 3 or tensor.shape[-1] != last_dim or not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be floating [B, N, {last_dim}]")


@dataclass(frozen=True, slots=True)
class O1SoftOutput:
    logits: Tensor

    def __post_init__(self) -> None:
        _require_float_shape(self.logits, 6, "O1 logits")


@dataclass(frozen=True, slots=True)
class O2SoftOutput:
    identity: Tensor
    score: Tensor

    def __post_init__(self) -> None:
        _require_float_shape(self.identity, 256, "O2 identity")
        _require_float_shape(self.score, 2, "O2 score")
        if self.identity.shape[:2] != self.score.shape[:2]:
            raise ValueError("O2 identity and score must share batch and slot dimensions")


@dataclass(frozen=True, slots=True)
class E1SoftOutput:
    logits: Tensor

    def __post_init__(self) -> None:
        _require_float_shape(self.logits, 3, "E1 logits")


@dataclass(frozen=True, slots=True)
class E2SoftOutput:
    event_logits: Tensor
    phase_logits: Tensor

    def __post_init__(self) -> None:
        _require_float_shape(self.event_logits, 4, "E2 event logits")
        _require_float_shape(self.phase_logits, 4, "E2 phase logits")
        if self.event_logits.shape != self.phase_logits.shape:
            raise ValueError("E2 event and phase logits must have identical shapes")


@dataclass(frozen=True, slots=True)
class ObservationOutputs:
    o1: O1SoftOutput
    o2: O2SoftOutput
    e1: E1SoftOutput
    e2: E2SoftOutput


def build_observation_heads(_config: ProjectConfig | None = None) -> NoReturn:
    """P8 owns all four decoder implementations."""

    raise NotImplementedError("Observation head implementation is deferred to P8")
