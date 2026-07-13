"""Define detached typed records and per-trajectory hard State Bank state.

Inputs: detached O1/O2/E1/E2 observations, timestamps, confidence, and semantic embeddings.
Outputs: isolated typed records, exact hard counters/FSM payloads, and audit entries.
Forbidden: gradients, nn.Parameter registration, model state_dict, retrieval, or Reader arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity


class HeadType(StrEnum):
    O1 = "o1"
    O2 = "o2"
    E1 = "e1"
    E2 = "e2"


class E2Phase(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    END_CANDIDATE = "end_candidate"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class O1Payload:
    current_visible_count: int
    baseline_count: int
    active_slot_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.current_visible_count < 0 or self.baseline_count < 0:
            raise ValueError("O1 counts must be non-negative")


@dataclass(frozen=True, slots=True)
class E1Payload:
    event_count: int
    recent_event_times: tuple[float, ...]
    cooldown_until: float

    def __post_init__(self) -> None:
        if self.event_count < 0 or self.cooldown_until < 0.0:
            raise ValueError("E1 state is invalid")
        if len(self.recent_event_times) > 512 or any(
            time < 0.0 for time in self.recent_event_times
        ):
            raise ValueError("E1 recent_event_times are invalid")


@dataclass(frozen=True, slots=True)
class E2Payload:
    completed_count: int
    phase: E2Phase
    completed_intervals: tuple[tuple[float, float], ...]
    recent_event_times: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.completed_count < 0 or self.completed_count != len(self.completed_intervals):
            raise ValueError("E2 completed_count must match completed intervals")
        if any(start < 0.0 or end < start for start, end in self.completed_intervals):
            raise ValueError("E2 completed intervals are invalid")
        if len(self.recent_event_times) > 512:
            raise ValueError("E2 recent event history cannot exceed 512")


type StatePayload = O1Payload | CandidateIdentity | ConfirmedIdentity | E1Payload | E2Payload


@dataclass(frozen=True, slots=True)
class StateRecord:
    record_id: str
    video_id: str
    trajectory_id: str
    head_type: HeadType
    semantic_embedding: Tensor
    timestamp: float | None
    time_range: tuple[float, float] | None
    valid: bool
    confidence: float
    payload: StatePayload

    def __post_init__(self) -> None:
        if not self.record_id or not self.video_id or not self.trajectory_id:
            raise ValueError("StateRecord isolation identifiers must be non-empty")
        embedding = self.semantic_embedding
        if embedding.shape != (512,) or not torch.is_floating_point(embedding):
            raise ValueError("semantic_embedding must be floating [512]")
        if self.timestamp is None and self.time_range is None:
            raise ValueError("StateRecord requires timestamp or time_range")
        if self.timestamp is not None and self.timestamp < 0.0:
            raise ValueError("StateRecord timestamp must be non-negative")
        if self.time_range is not None:
            start, end = self.time_range
            if start < 0.0 or end < start:
                raise ValueError("StateRecord time_range is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("StateRecord confidence must be within [0, 1]")
        expected_head = {
            O1Payload: HeadType.O1,
            CandidateIdentity: HeadType.O2,
            ConfirmedIdentity: HeadType.O2,
            E1Payload: HeadType.E1,
            E2Payload: HeadType.E2,
        }[type(self.payload)]
        if self.head_type is not expected_head:
            raise ValueError("StateRecord head_type does not match its typed payload")


@dataclass(frozen=True, slots=True)
class StateBankAuditEntry:
    action: str
    record_id: str | None
    timestamp: float
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class StateBankRuntimeState:
    video_id: str
    trajectory_id: str
    records: tuple[StateRecord, ...]
    audit_log: tuple[StateBankAuditEntry, ...]

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("State Bank runtime isolation identifiers must be non-empty")
        if any(
            record.video_id != self.video_id or record.trajectory_id != self.trajectory_id
            for record in self.records
        ):
            raise ValueError("State Bank records cannot cross video or trajectory boundaries")


def build_state_bank(_config: ProjectConfig | None = None) -> NoReturn:
    """P9 owns semantic projection, hard writes, FSMs, reset, and release."""

    raise NotImplementedError("Structured State Bank implementation is deferred to P9")
