"""Define Candidate, Confirmed, and GPU Hot Cache identity runtime types.

Inputs: detached 256-D identity observations and causal timestamps for one trajectory.
Outputs: auditable Candidate/Confirmed records and a non-authoritative Hot Cache view.
Forbidden: q_target semantic retrieval, silent overwrite, model parameters, or final answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig


def _validate_identity(prototype: Tensor) -> None:
    if prototype.shape != (256,) or not torch.is_floating_point(prototype):
        raise ValueError("identity prototype must be floating [256]")


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    candidate_id: str
    identity_prototype: Tensor
    observation_count: int
    ttl_remaining: int
    confidence: float

    def __post_init__(self) -> None:
        _validate_identity(self.identity_prototype)
        if not self.candidate_id or self.observation_count < 1 or self.ttl_remaining < 0:
            raise ValueError("candidate identity metadata is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ConfirmedIdentity:
    identity_id: str
    identity_prototype: Tensor
    first_seen: float
    last_seen: float
    observation_count: int

    def __post_init__(self) -> None:
        _validate_identity(self.identity_prototype)
        if not self.identity_id or self.observation_count < 1:
            raise ValueError("confirmed identity metadata is invalid")
        if not 0.0 <= self.first_seen <= self.last_seen:
            raise ValueError("confirmed identity timestamps are invalid")


@dataclass(frozen=True, slots=True)
class HotCacheEntry:
    identity_id: str
    identity_prototype: Tensor
    last_accessed: float

    def __post_init__(self) -> None:
        _validate_identity(self.identity_prototype)
        if not self.identity_id or self.last_accessed < 0.0:
            raise ValueError("hot cache metadata is invalid")


@dataclass(frozen=True, slots=True)
class IdentityBankRuntimeState:
    candidates: tuple[CandidateIdentity, ...]
    confirmed: tuple[ConfirmedIdentity, ...]
    hot_cache: tuple[HotCacheEntry, ...]
    unique_count: int
    candidate_overflow_count: int

    def __post_init__(self) -> None:
        if self.unique_count < 0 or self.candidate_overflow_count < 0:
            raise ValueError("identity counters must be non-negative")
        if self.unique_count != len(self.confirmed):
            raise ValueError("unique_count must equal the number of Confirmed identities")
        if len(self.hot_cache) > 256:
            raise ValueError("GPU Hot Cache cannot exceed 256 entries")


def build_identity_bank(_config: ProjectConfig | None = None) -> NoReturn:
    """P10 owns dynamic storage, promotion, matching, and cache eviction."""

    raise NotImplementedError("Identity Bank implementation is deferred to P10")
