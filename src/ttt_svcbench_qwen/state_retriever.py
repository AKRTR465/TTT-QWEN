"""Define threshold-retrieval outputs over the current typed State Bank.

Inputs: q_target, hard operator/head type, TimeWindow, and legal current-trajectory records.
Outputs: all threshold-passing record IDs, scores, masks, status, N_s, and N_ret.
Forbidden: fixed Top-K, ANN, Reader arithmetic, labels, future records, or Bank mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


class RetrievalStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RetrieverOutput:
    selected_record_ids: tuple[tuple[str, ...], ...]
    scores: Tensor
    selected_mask: Tensor
    status: tuple[RetrievalStatus, ...]
    n_state: Tensor
    n_retrieved: Tensor

    def __post_init__(self) -> None:
        if self.scores.ndim != 2 or not torch.is_floating_point(self.scores):
            raise ValueError("Retriever scores must be floating [B, N_s]")
        batch_size = self.scores.shape[0]
        if (
            self.selected_mask.shape != self.scores.shape
            or self.selected_mask.dtype != torch.bool
        ):
            raise ValueError("selected_mask must be bool [B, N_s]")
        if len(self.selected_record_ids) != batch_size or len(self.status) != batch_size:
            raise ValueError("Retriever metadata must contain one entry per batch item")
        for counts, name in ((self.n_state, "n_state"), (self.n_retrieved, "n_retrieved")):
            if counts.shape != (batch_size,) or counts.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be integer [B]")


def build_state_retriever(_config: ProjectConfig | None = None) -> NoReturn:
    """P11 owns normalized cosine scoring and legal hard filters."""

    raise NotImplementedError("State Retriever implementation is deferred to P11")
