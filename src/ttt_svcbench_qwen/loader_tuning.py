"""Deterministic selection between the bounded A2 DataLoader profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoaderTrial:
    workers: int
    prefetch_factor: int
    ga_group_wait_p95_seconds: float
    support_read_p95_seconds: float
    host_memory_gib: float
    host_memory_budget_gib: float

    def __post_init__(self) -> None:
        if self.workers <= 0 or self.prefetch_factor <= 0:
            raise ValueError("DataLoader worker/prefetch counts must be positive")
        values = (
            self.ga_group_wait_p95_seconds,
            self.support_read_p95_seconds,
            self.host_memory_gib,
            self.host_memory_budget_gib,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("DataLoader trial measurements must be finite and non-negative")
        if self.ga_group_wait_p95_seconds <= 0.0 or self.host_memory_budget_gib <= 0.0:
            raise ValueError("DataLoader wait and memory budget must be positive")

    @property
    def profile(self) -> tuple[int, int]:
        return self.workers, self.prefetch_factor


def select_loader_profile(trials: Sequence[LoaderTrial]) -> tuple[int, int]:
    """Choose 4x1 only when it passes every predeclared GPFS/host-memory gate."""

    by_profile = {trial.profile: trial for trial in trials}
    if len(by_profile) != len(trials) or any(
        profile not in {(2, 2), (4, 1)} for profile in by_profile
    ):
        raise ValueError("loader trials contain duplicate or unsupported profiles")
    baseline = by_profile.get((2, 2))
    if baseline is None or baseline.host_memory_gib > baseline.host_memory_budget_gib:
        raise ValueError("loader tuning requires one memory-safe 2x2 baseline")
    candidate = by_profile.get((4, 1))
    if candidate is None:
        return baseline.profile
    wait_improved = candidate.ga_group_wait_p95_seconds <= 0.80 * baseline.ga_group_wait_p95_seconds
    support_read_safe = candidate.support_read_p95_seconds <= max(
        1.0e-9, 1.10 * baseline.support_read_p95_seconds
    )
    memory_safe = candidate.host_memory_gib <= candidate.host_memory_budget_gib
    return (
        candidate.profile
        if wait_improved and support_read_safe and memory_safe
        else baseline.profile
    )


def parse_loader_trials(value: object) -> tuple[LoaderTrial, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("loader trial input must be a list")
    fields = set(LoaderTrial.__dataclass_fields__)
    rows: list[LoaderTrial] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("loader trial fields are invalid")
        rows.append(LoaderTrial(**item))
    return tuple(rows)


__all__ = ["LoaderTrial", "parse_loader_trials", "select_loader_profile"]
