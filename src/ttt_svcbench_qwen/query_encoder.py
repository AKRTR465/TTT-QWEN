"""Define question encoding, operator routing, and deterministic time-window types.

Inputs: question-only Qwen token embeddings, padding mask, and legal query time.
Outputs: target/operator/time embeddings, hard operator metadata, and TimeWindow.
Forbidden: answers, count labels, State Bank mutation, Reader arithmetic, or keyword routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


class Operator(StrEnum):
    O1_SNAP = "o1-snap"
    O1_DELTA = "o1-delta"
    O2_UNIQUE = "o2-unique"
    O2_GAIN = "o2-gain"
    E1_ACTION = "e1-action"
    E1_TRANSIT = "e1-transit"
    E2_PERIODIC = "e2-periodic"
    E2_EPISODE = "e2-episode"
    UNSUPPORTED = "unsupported"


class TimeWindowMode(StrEnum):
    NOW = "now"
    HISTORY = "history"
    RECENT = "recent"
    EXPLICIT_RANGE = "explicit_range"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    mode: TimeWindowMode
    query_time: float
    start_time: float | None
    end_time: float
    valid: bool

    def __post_init__(self) -> None:
        if self.query_time < 0.0 or self.end_time < 0.0:
            raise ValueError("TimeWindow times must be non-negative")
        if self.end_time > self.query_time:
            raise ValueError("TimeWindow cannot extend beyond query_time")
        if self.start_time is not None and not 0.0 <= self.start_time <= self.end_time:
            raise ValueError("TimeWindow start_time must be within [0, end_time]")
        if self.valid and self.mode is TimeWindowMode.EXPLICIT_RANGE and self.start_time is None:
            raise ValueError("a valid explicit range requires start_time")


@dataclass(frozen=True, slots=True)
class QueryEncoderOutput:
    q_target: Tensor
    q_operator: Tensor
    q_time: Tensor
    operator_logits: Tensor
    operator_confidence: Tensor
    padding_mask: Tensor

    def __post_init__(self) -> None:
        embeddings = (self.q_target, self.q_operator, self.q_time)
        batch_size = self.q_target.shape[0] if self.q_target.ndim == 2 else -1
        for embedding in embeddings:
            if embedding.shape != (batch_size, 512) or not torch.is_floating_point(embedding):
                raise ValueError("query embeddings must be floating [B, 512]")
            if embedding.device != self.q_target.device or embedding.dtype != self.q_target.dtype:
                raise ValueError("query embeddings must share dtype and device")
        if self.operator_logits.shape != (batch_size, len(Operator)):
            raise ValueError("operator_logits must be [B, 9]")
        if self.operator_confidence.shape != (batch_size,):
            raise ValueError("operator_confidence must be [B]")
        if self.padding_mask.ndim != 2 or self.padding_mask.shape[0] != batch_size:
            raise ValueError("padding_mask must be [B, L_q]")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must use bool dtype")


def build_query_encoder(_config: ProjectConfig | None = None) -> NoReturn:
    """P4 owns the query network, prototypes, and time resolver."""

    raise NotImplementedError("Query encoder implementation is deferred to P4")
