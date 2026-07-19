"""Deterministic H200 visual-batch selection from measured trial summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VisualBatchTrial:
    batch_size: int
    oom: bool
    free_memory_gib: float
    visual_p50_seconds: float
    discrete_results_match: bool
    loss_balance_match: bool

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("visual batch trial size must be positive")
        if not math.isfinite(self.free_memory_gib) or self.free_memory_gib < 0.0:
            raise ValueError("visual batch trial free memory must be non-negative")
        if not math.isfinite(self.visual_p50_seconds) or self.visual_p50_seconds <= 0.0:
            raise ValueError("visual batch trial p50 must be positive")

    @property
    def safe(self) -> bool:
        return (
            not self.oom
            and self.free_memory_gib >= 12.0
            and self.discrete_results_match
            and self.loss_balance_match
        )


def select_visual_batch_size(
    stage: Literal["a2", "a5"],
    trials: Sequence[VisualBatchTrial],
) -> int:
    allowed = (1, 2, 4, 8) if stage == "a2" else (1, 2, 4)
    by_size = {trial.batch_size: trial for trial in trials}
    if len(by_size) != len(trials) or any(size not in allowed for size in by_size):
        raise ValueError("visual batch trials contain duplicate or unsupported sizes")
    baseline = by_size.get(1)
    if baseline is None or not baseline.safe:
        raise ValueError("visual batch tuning requires one safe batch-size-1 baseline")
    selected = 1
    previous = baseline
    for size in allowed[1:]:
        trial = by_size.get(size)
        if trial is None:
            break
        improvement = (
            previous.visual_p50_seconds - trial.visual_p50_seconds
        ) / previous.visual_p50_seconds
        if not trial.safe or improvement < 0.05:
            break
        selected = size
        previous = trial
    return selected


def parse_visual_batch_trials(value: object) -> tuple[VisualBatchTrial, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("visual batch trial input must be a list")
    fields = set(VisualBatchTrial.__dataclass_fields__)
    rows: list[VisualBatchTrial] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("visual batch trial fields are invalid")
        rows.append(VisualBatchTrial(**item))
    return tuple(rows)


__all__ = [
    "VisualBatchTrial",
    "parse_visual_batch_trials",
    "select_visual_batch_size",
]
