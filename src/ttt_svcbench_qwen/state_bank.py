"""Implement the learned semantic view and detached typed State Bank runtime.

Inputs: semantic source states plus detached O1/O2/E1/E2 evidence and owner metadata.
Outputs: normalized semantic embeddings, functional typed records, hard FSM state, and audit.
Forbidden: identity matching, retrieval, Reader arithmetic, gradients in runtime, or in-place state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import (
    ProjectConfig,
    SemanticProjectorConfig,
    StateBankConfig,
)
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.observation_heads import E1SoftOutput, E2SoftOutput, O1SoftOutput
from ttt_svcbench_qwen.tensor_contracts import tensor_storage_key

if TYPE_CHECKING:
    from ttt_svcbench_qwen.query_encoder import Operator


class HeadType(StrEnum):
    O1 = "o1"
    O2 = "o2"
    E1 = "e1"
    E2 = "e2"


class StateRecordKind(StrEnum):
    """Distinguish lifecycle subtypes that share one coarse observation head."""

    O1_AGGREGATE = "o1_aggregate"
    O2_CANDIDATE = "o2_candidate"
    O2_CONFIRMED = "o2_confirmed"
    E1_AGGREGATE = "e1_aggregate"
    E2_AGGREGATE = "e2_aggregate"


class E2Phase(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    END_CANDIDATE = "end_candidate"
    COMPLETED = "completed"


class E1EventKind(StrEnum):
    ACTION = "action"
    TRANSIT = "transit"


class E2EventKind(StrEnum):
    PERIODIC = "periodic"
    EPISODE = "episode"


@dataclass(frozen=True, slots=True)
class O1SlotState:
    slot_id: int
    is_object: bool
    is_target: bool
    visible: bool
    enter: bool
    exit: bool
    last_timestamp: float
    last_position_id: int
    confidence: float

    def __post_init__(self) -> None:
        if type(self.slot_id) is not int or self.slot_id < 0:
            raise ValueError("O1 slot_id must be a non-negative integer")
        flags = (self.is_object, self.is_target, self.visible, self.enter, self.exit)
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("O1 slot evidence flags must be bool")
        if (
            not math.isfinite(self.last_timestamp)
            or self.last_timestamp < 0.0
            or type(self.last_position_id) is not int
            or self.last_position_id < 0
        ):
            raise ValueError("O1 slot metadata must be finite and non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("O1 slot confidence must stay within [0, 1]")


@dataclass(frozen=True, slots=True)
class O1Payload:
    current_visible_count: int
    baseline_count: int
    active_slot_ids: tuple[int, ...]
    slot_states: tuple[O1SlotState, ...] = ()
    baseline_initialized: bool = True
    baseline_position_id: int | None = None
    last_timestamp: float = -1.0
    last_position_id: int = -1
    update_count: int = 0
    last_spatial_overflow_count: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.current_visible_count) is not int
            or self.current_visible_count < 0
            or type(self.baseline_count) is not int
            or self.baseline_count < 0
        ):
            raise ValueError("O1 counts must be non-negative integers")
        if type(self.baseline_initialized) is not bool:
            raise TypeError("O1 baseline_initialized must be bool")
        if (
            type(self.update_count) is not int
            or self.update_count < 0
            or type(self.last_spatial_overflow_count) is not int
            or self.last_spatial_overflow_count < 0
        ):
            raise ValueError("O1 update/overflow counts must be non-negative integers")
        if tuple(sorted(set(self.active_slot_ids))) != self.active_slot_ids:
            raise ValueError("O1 active_slot_ids must be unique and sorted")
        if self.current_visible_count != len(self.active_slot_ids):
            raise ValueError("O1 current count must match active_slot_ids")
        slot_ids = tuple(slot.slot_id for slot in self.slot_states)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("O1 slot states cannot duplicate slot IDs")
        if self.slot_states:
            visible_ids = tuple(slot.slot_id for slot in self.slot_states if slot.visible)
            if visible_ids != self.active_slot_ids:
                raise ValueError("O1 active_slot_ids must match visible slot states")
        if self.baseline_initialized:
            if self.baseline_position_id is not None and self.baseline_position_id < 0:
                raise ValueError("O1 baseline position must be non-negative")
        elif self.baseline_count != 0 or self.baseline_position_id is not None:
            raise ValueError("an uninitialized O1 baseline cannot carry baseline state")
        _validate_last_metadata(self.last_timestamp, self.last_position_id, "O1")


@dataclass(frozen=True, slots=True)
class E1Payload:
    event_kind: E1EventKind
    event_count: int
    recent_event_times: tuple[float, ...]
    cooldown_until: float
    active: bool = False
    armed: bool = True
    candidate_start: float | None = None
    last_timestamp: float = -1.0
    last_position_id: int = -1
    duplicate_suppression_count: int = 0
    cooldown_hit_count: int = 0
    nms_suppression_count: int = 0
    miss_candidate_count: int = 0
    history_eviction_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, E1EventKind):
            raise TypeError("E1 event_kind must be an E1EventKind")
        counters = (
            self.event_count,
            self.duplicate_suppression_count,
            self.cooldown_hit_count,
            self.nms_suppression_count,
            self.miss_candidate_count,
            self.history_eviction_count,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("E1 counters must be non-negative integers")
        if type(self.active) is not bool or type(self.armed) is not bool:
            raise TypeError("E1 active/armed flags must be bool")
        if not math.isfinite(self.cooldown_until) or self.cooldown_until < 0.0:
            raise ValueError("E1 cooldown must be finite and non-negative")
        if len(self.recent_event_times) > 512:
            raise ValueError("E1 recent event history cannot exceed 512")
        _validate_strict_times(self.recent_event_times, "E1 recent event")
        if self.event_count < len(self.recent_event_times):
            raise ValueError("E1 event_count cannot be smaller than retained history")
        if self.active != (self.candidate_start is not None):
            raise ValueError("E1 active state and candidate_start must agree")
        if self.candidate_start is not None and (
            not math.isfinite(self.candidate_start) or self.candidate_start < 0.0
        ):
            raise ValueError("E1 candidate_start must be finite and non-negative")
        _validate_last_metadata(self.last_timestamp, self.last_position_id, "E1")

    @property
    def history_truncated(self) -> bool:
        return self.history_eviction_count > 0


@dataclass(frozen=True, slots=True)
class E2Payload:
    event_kind: E2EventKind
    completed_count: int
    phase: E2Phase
    completed_intervals: tuple[tuple[float, float], ...]
    recent_event_times: tuple[float, ...]
    current_start: float | None = None
    last_timestamp: float = -1.0
    last_position_id: int = -1
    duplicate_suppression_count: int = 0
    conflict_count: int = 0
    rearm_suppression_count: int = 0
    history_eviction_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, E2EventKind):
            raise TypeError("E2 event_kind must be an E2EventKind")
        counters = (
            self.completed_count,
            self.duplicate_suppression_count,
            self.conflict_count,
            self.rearm_suppression_count,
            self.history_eviction_count,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("E2 counters must be non-negative integers")
        if not isinstance(self.phase, E2Phase):
            raise TypeError("E2 phase must be an E2Phase")
        if self.completed_count != len(self.completed_intervals):
            raise ValueError("E2 completed_count must match complete interval history")
        previous_end = -1.0
        for start, end in self.completed_intervals:
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or end < start
                or start < previous_end
            ):
                raise ValueError("E2 completed intervals must be finite and ordered")
            previous_end = end
        if len(self.recent_event_times) > 512:
            raise ValueError("E2 recent event history cannot exceed 512")
        _validate_strict_times(self.recent_event_times, "E2 recent event")
        if self.completed_count < len(self.recent_event_times):
            raise ValueError("E2 completed_count cannot be smaller than retained history")
        if self.phase in (E2Phase.ACTIVE, E2Phase.END_CANDIDATE):
            if self.current_start is None:
                raise ValueError("an active E2 interval requires current_start")
        elif self.current_start is not None:
            raise ValueError("inactive/completed E2 state cannot keep current_start")
        if self.current_start is not None and (
            not math.isfinite(self.current_start) or self.current_start < 0.0
        ):
            raise ValueError("E2 current_start must be finite and non-negative")
        _validate_last_metadata(self.last_timestamp, self.last_position_id, "E2")

    @property
    def history_truncated(self) -> bool:
        return self.history_eviction_count > 0


type StatePayload = O1Payload | CandidateIdentity | ConfirmedIdentity | E1Payload | E2Payload
type AuditValue = str | int | float | bool | None


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
        if not isinstance(self.head_type, HeadType) or type(self.valid) is not bool:
            raise TypeError("StateRecord head_type/valid fields have invalid types")
        embedding = self.semantic_embedding
        if embedding.shape != (512,) or not torch.is_floating_point(embedding):
            raise ValueError("semantic_embedding must be floating [512]")
        if embedding.requires_grad or embedding.grad_fn is not None:
            raise ValueError("StateRecord semantic_embedding must be detached")
        if embedding.device.type != "meta":
            if not bool(torch.isfinite(embedding).all()):
                raise ValueError("semantic_embedding must be finite")
            norm = torch.linalg.vector_norm(embedding.float())
            norm_tolerance = max(
                5.0e-4,
                2.0 * float(torch.finfo(embedding.dtype).eps),
            )
            if not torch.allclose(
                norm,
                torch.ones_like(norm),
                atol=norm_tolerance,
                rtol=0.0,
            ):
                raise ValueError("semantic_embedding must have unit L2 norm")
        if (self.timestamp is None) == (self.time_range is None):
            raise ValueError("StateRecord requires exactly one of timestamp or time_range")
        if self.timestamp is not None and (
            not math.isfinite(self.timestamp) or self.timestamp < 0.0
        ):
            raise ValueError("StateRecord timestamp must be finite and non-negative")
        if self.time_range is not None:
            start, end = self.time_range
            if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
                raise ValueError("StateRecord time_range is invalid")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("StateRecord confidence must be finite and within [0, 1]")
        expected_head = {
            O1Payload: HeadType.O1,
            CandidateIdentity: HeadType.O2,
            ConfirmedIdentity: HeadType.O2,
            E1Payload: HeadType.E1,
            E2Payload: HeadType.E2,
        }.get(type(self.payload))
        if expected_head is None or self.head_type is not expected_head:
            raise ValueError("StateRecord head_type does not match its typed payload")
        _validate_payload_tensors_detached(self.payload)
        _validate_record_payload_time(self)


@dataclass(frozen=True, slots=True)
class RetrievalHistoryRecord:
    """Detached pre-projector source retained only as transient retrieval memory."""

    record_id: str
    video_id: str
    trajectory_id: str
    head_type: HeadType
    operator: Operator
    semantic_source: Tensor
    timestamp: float | None
    time_range: tuple[float, float] | None
    valid: bool
    retrieval_eligible: bool
    lifecycle_id: str | None = None

    def __post_init__(self) -> None:
        from ttt_svcbench_qwen.query_encoder import OPERATOR_TO_HEAD_TYPE, Operator

        if not self.record_id or not self.video_id or not self.trajectory_id:
            raise ValueError("retrieval history isolation identifiers must be non-empty")
        if not isinstance(self.head_type, HeadType) or not isinstance(self.operator, Operator):
            raise TypeError("retrieval history head/operator metadata is invalid")
        if OPERATOR_TO_HEAD_TYPE[self.operator] is not self.head_type:
            raise ValueError("retrieval history operator must match its head")
        if type(self.valid) is not bool or type(self.retrieval_eligible) is not bool:
            raise TypeError("retrieval history validity flags must be bool")
        if self.retrieval_eligible and not self.valid:
            raise ValueError("invalid retrieval history cannot remain eligible")
        source = self.semantic_source
        if source.shape != (768,) or not torch.is_floating_point(source):
            raise ValueError("retrieval history semantic_source must be floating [768]")
        if source.requires_grad or source.grad_fn is not None:
            raise ValueError("retrieval history semantic_source must be detached")
        if source.device.type != "meta" and not bool(torch.isfinite(source).all()):
            raise ValueError("retrieval history semantic_source must be finite")
        if (self.timestamp is None) == (self.time_range is None):
            raise ValueError("retrieval history requires exactly one timestamp representation")
        if self.timestamp is not None and (
            not math.isfinite(self.timestamp) or self.timestamp < 0.0
        ):
            raise ValueError("retrieval history timestamp must be finite and non-negative")
        if self.time_range is not None:
            start, end = self.time_range
            if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
                raise ValueError("retrieval history time_range is invalid")
        if self.lifecycle_id is not None and not self.lifecycle_id:
            raise ValueError("retrieval history lifecycle_id cannot be empty")


RETRIEVAL_HEAD_ORDER: tuple[HeadType, ...] = (
    HeadType.O1,
    HeadType.O2,
    HeadType.E1,
    HeadType.E2,
)


@dataclass(frozen=True, slots=True)
class RetrievalHistoryAppendBatch:
    """One vectorized, label-free write produced by a single observation chunk."""

    sources: Tensor
    head_codes: Tensor
    operator_codes: Tensor
    timestamps: Tensor
    time_ranges: Tensor
    valid_mask: Tensor
    eligible_mask: Tensor
    lifecycle_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        count = self.sources.shape[0] if self.sources.ndim == 2 else -1
        if self.sources.shape != (count, 768) or not torch.is_floating_point(self.sources):
            raise ValueError("retrieval append sources must be floating [M, 768]")
        vectors = (
            self.head_codes,
            self.operator_codes,
            self.timestamps,
            self.valid_mask,
            self.eligible_mask,
        )
        if any(value.shape != (count,) for value in vectors):
            raise ValueError("retrieval append metadata must align to M")
        if self.head_codes.dtype != torch.int64 or self.operator_codes.dtype != torch.int64:
            raise TypeError("retrieval append head/operator codes must be int64")
        if self.timestamps.dtype != torch.float64:
            raise TypeError("retrieval append timestamps must be float64")
        if self.time_ranges.shape != (count, 2) or self.time_ranges.dtype != torch.float64:
            raise ValueError("retrieval append time_ranges must be float64 [M, 2]")
        if self.valid_mask.dtype != torch.bool or self.eligible_mask.dtype != torch.bool:
            raise TypeError("retrieval append validity masks must be bool")
        tensors = (
            self.head_codes,
            self.operator_codes,
            self.timestamps,
            self.time_ranges,
            self.valid_mask,
            self.eligible_mask,
        )
        if any(value.device != self.sources.device for value in tensors):
            raise ValueError("retrieval append tensors must share one device")
        if self.lifecycle_ids and len(self.lifecycle_ids) != count:
            raise ValueError("retrieval append lifecycle metadata must align to M")
        if any(value is not None and not value for value in self.lifecycle_ids):
            raise ValueError("retrieval append lifecycle IDs cannot be empty")
        if self.sources.device.type != "meta":
            if not bool(torch.isfinite(self.sources).all()):
                raise ValueError("retrieval append sources must be finite")
            if bool(
                torch.any((self.head_codes < 0) | (self.head_codes >= len(RETRIEVAL_HEAD_ORDER)))
            ):
                raise ValueError("retrieval append head codes are out of range")
            if bool(torch.any(self.eligible_mask & ~self.valid_mask)):
                raise ValueError("invalid retrieval rows cannot be eligible")
            point = self.timestamps >= 0.0
            ranged = self.time_ranges[:, 0] >= 0.0
            if not bool(torch.all(point ^ ranged)):
                raise ValueError("retrieval append rows require exactly one time representation")
            if bool(torch.any(ranged & (self.time_ranges[:, 1] < self.time_ranges[:, 0]))):
                raise ValueError("retrieval append time ranges are invalid")


class TensorizedRetrievalHistory:
    """Episode-local mutable tensor ring; never registered in model/checkpoint state."""

    def __init__(
        self,
        video_id: str,
        trajectory_id: str,
        *,
        capacity_per_head: int,
        source_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not video_id or not trajectory_id:
            raise ValueError("tensor retrieval history requires non-empty owners")
        if capacity_per_head <= 0 or source_dim != 768:
            raise ValueError("tensor retrieval history capacity/source dim is invalid")
        self.video_id = video_id
        self.trajectory_id = trajectory_id
        self.capacity_per_head = capacity_per_head
        self.source_dim = source_dim
        self.sources = torch.zeros((4, capacity_per_head, source_dim), dtype=dtype, device=device)
        self.sequence_ids = torch.full((4, capacity_per_head), -1, dtype=torch.int64, device=device)
        self.operator_codes = torch.full_like(self.sequence_ids, -1)
        self.timestamps = torch.full(
            (4, capacity_per_head), -1.0, dtype=torch.float64, device=device
        )
        self.time_ranges = torch.full(
            (4, capacity_per_head, 2), -1.0, dtype=torch.float64, device=device
        )
        self.valid_mask = torch.zeros((4, capacity_per_head), dtype=torch.bool, device=device)
        self.eligible_mask = torch.zeros_like(self.valid_mask)
        self.sizes = [0, 0, 0, 0]
        self.write_ptrs = [0, 0, 0, 0]
        self.lifecycle_ids: list[list[str | None]] = [
            [None] * capacity_per_head for _ in RETRIEVAL_HEAD_ORDER
        ]
        self.next_sequence = 0
        self.version = 0
        self.released = False

    @property
    def count(self) -> int:
        return sum(self.sizes)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def append_many(self, batch: RetrievalHistoryAppendBatch) -> None:
        if self.released:
            raise RuntimeError("released tensor retrieval history cannot be written")
        if batch.sources.dtype != self.sources.dtype or batch.sources.device != self.sources.device:
            raise ValueError("retrieval append batch must match ring dtype/device")
        count = batch.sources.shape[0]
        if count == 0:
            return
        sequence_ids = torch.arange(
            self.next_sequence,
            self.next_sequence + count,
            dtype=torch.int64,
            device=self.sources.device,
        )
        for head_code in range(len(RETRIEVAL_HEAD_ORDER)):
            source_indices = torch.nonzero(batch.head_codes == head_code, as_tuple=False).flatten()
            head_count = source_indices.numel()
            if head_count == 0:
                continue
            if head_count > self.capacity_per_head:
                source_indices = source_indices[-self.capacity_per_head :]
                head_count = self.capacity_per_head
            destinations = (
                torch.arange(head_count, dtype=torch.int64, device=self.sources.device)
                + self.write_ptrs[head_code]
            ) % self.capacity_per_head
            self.sources[head_code].index_copy_(
                0, destinations, batch.sources.index_select(0, source_indices).detach()
            )
            self.sequence_ids[head_code].index_copy_(
                0, destinations, sequence_ids.index_select(0, source_indices)
            )
            self.operator_codes[head_code].index_copy_(
                0, destinations, batch.operator_codes.index_select(0, source_indices)
            )
            self.timestamps[head_code].index_copy_(
                0, destinations, batch.timestamps.index_select(0, source_indices)
            )
            self.time_ranges[head_code].index_copy_(
                0, destinations, batch.time_ranges.index_select(0, source_indices)
            )
            self.valid_mask[head_code].index_copy_(
                0, destinations, batch.valid_mask.index_select(0, source_indices)
            )
            self.eligible_mask[head_code].index_copy_(
                0, destinations, batch.eligible_mask.index_select(0, source_indices)
            )
            if batch.lifecycle_ids and any(value is not None for value in batch.lifecycle_ids):
                # Lifecycle metadata is CPU-only and absent from the production hot path.
                source_cpu = source_indices.detach().cpu().tolist()
                destination_cpu = destinations.detach().cpu().tolist()
                for source, destination in zip(source_cpu, destination_cpu, strict=True):
                    self.lifecycle_ids[head_code][destination] = batch.lifecycle_ids[source]
            else:
                start = self.write_ptrs[head_code]
                first = min(head_count, self.capacity_per_head - start)
                self.lifecycle_ids[head_code][start : start + first] = [None] * first
                remainder = head_count - first
                if remainder:
                    self.lifecycle_ids[head_code][:remainder] = [None] * remainder
            self.write_ptrs[head_code] = (
                self.write_ptrs[head_code] + head_count
            ) % self.capacity_per_head
            self.sizes[head_code] = min(self.capacity_per_head, self.sizes[head_code] + head_count)
        self.next_sequence += count
        self.version += 1

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def fork(self) -> TensorizedRetrievalHistory:
        clone = TensorizedRetrievalHistory(
            self.video_id,
            self.trajectory_id,
            capacity_per_head=self.capacity_per_head,
            source_dim=self.source_dim,
            dtype=self.sources.dtype,
            device=self.sources.device,
        )
        for name in (
            "sources",
            "sequence_ids",
            "operator_codes",
            "timestamps",
            "time_ranges",
            "valid_mask",
            "eligible_mask",
        ):
            setattr(clone, name, getattr(self, name).clone())
        clone.sizes = list(self.sizes)
        clone.write_ptrs = list(self.write_ptrs)
        clone.lifecycle_ids = [list(row) for row in self.lifecycle_ids]
        clone.next_sequence = self.next_sequence
        clone.version = self.version
        clone.released = self.released
        return clone

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def release(self) -> None:
        self.sources = self.sources.new_empty((0, 0, self.source_dim))
        self.sequence_ids = self.sequence_ids.new_empty((0, 0))
        self.operator_codes = self.operator_codes.new_empty((0, 0))
        self.timestamps = self.timestamps.new_empty((0, 0))
        self.time_ranges = self.time_ranges.new_empty((0, 0, 2))
        self.valid_mask = self.valid_mask.new_empty((0, 0))
        self.eligible_mask = self.eligible_mask.new_empty((0, 0))
        self.sizes = [0, 0, 0, 0]
        self.write_ptrs = [0, 0, 0, 0]
        self.lifecycle_ids = [[], [], [], []]
        self.released = True
        self.version += 1


@dataclass(frozen=True, slots=True)
class StateBankAuditEntry:
    action: str
    record_id: str | None
    timestamp: float
    details: tuple[tuple[str, AuditValue], ...]

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("State Bank audit action must be non-empty")
        if self.record_id is not None and not self.record_id:
            raise ValueError("State Bank audit record_id cannot be empty")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("State Bank audit timestamp must be finite and non-negative")
        keys = tuple(key for key, _ in self.details)
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("State Bank audit detail keys must be unique and non-empty")
        if any(isinstance(value, Tensor) for _, value in self.details):
            raise TypeError("State Bank audit details cannot retain tensors")


@dataclass(frozen=True, slots=True)
class StateBankRuntimeState:
    video_id: str
    trajectory_id: str
    records: tuple[StateRecord, ...]
    audit_log: tuple[StateBankAuditEntry, ...]
    retrieval_history: tuple[RetrievalHistoryRecord, ...] = ()
    issued_record_ids: tuple[str, ...] = ()
    next_record_sequence: int = 0
    next_retrieval_sequence: int = 0
    released: bool = False
    version: int = 0

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("State Bank runtime isolation identifiers must be non-empty")
        if type(self.released) is not bool:
            raise TypeError("State Bank released flag must be bool")
        if (
            type(self.next_record_sequence) is not int
            or self.next_record_sequence < 0
            or type(self.next_retrieval_sequence) is not int
            or self.next_retrieval_sequence < 0
            or type(self.version) is not int
            or self.version < 0
        ):
            raise ValueError("State Bank runtime sequence/version must be non-negative integers")
        if self.released and (
            self.records or self.retrieval_history or self.audit_log or self.issued_record_ids
        ):
            raise ValueError("released State Bank runtime cannot retain trajectory state")
        if any(
            record.video_id != self.video_id or record.trajectory_id != self.trajectory_id
            for record in self.records
        ):
            raise ValueError("State Bank records cannot cross video or trajectory boundaries")
        record_ids = tuple(record.record_id for record in self.records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("State Bank record IDs must be unique within a trajectory")
        history_ids = tuple(record.record_id for record in self.retrieval_history)
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("retrieval history record IDs must be unique within a trajectory")
        if set(record_ids).intersection(history_ids):
            raise ValueError("aggregate and retrieval history record IDs cannot overlap")
        if any(
            record.video_id != self.video_id or record.trajectory_id != self.trajectory_id
            for record in self.retrieval_history
        ):
            raise ValueError("retrieval history cannot cross video or trajectory boundaries")
        issued = self.issued_record_ids
        if not issued and record_ids:
            object.__setattr__(self, "issued_record_ids", record_ids)
            issued = record_ids
        if len(set(issued)) != len(issued) or any(
            record_id not in issued for record_id in record_ids
        ):
            raise ValueError("State Bank record ID tombstones are inconsistent")
        if self.audit_log and any(
            right.timestamp < left.timestamp
            for left, right in zip(self.audit_log, self.audit_log[1:], strict=False)
        ):
            raise ValueError("State Bank audit timestamps must be monotonic")
        if self.records:
            reference = self.records[0].semantic_embedding
            if any(
                record.semantic_embedding.dtype != reference.dtype
                or record.semantic_embedding.device != reference.device
                for record in self.records[1:]
            ):
                raise ValueError("State Bank semantic embeddings must share dtype/device")
        tensor_groups = tuple(_record_tensors(record) for record in self.records)
        _assert_tensor_groups_isolated(tensor_groups, "State Bank records")
        _assert_tensor_groups_isolated(
            tuple((record.semantic_source,) for record in self.retrieval_history),
            "retrieval history records",
        )


@dataclass(frozen=True, slots=True)
class StateBankView:
    embeddings: Tensor
    present_mask: Tensor
    record_valid_mask: Tensor
    timestamps: Tensor
    time_ranges: Tensor
    n_state: Tensor
    owner_record_counts: Tensor
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    bank_versions: tuple[int, ...]
    record_ids: tuple[tuple[str | None, ...], ...]
    head_types: tuple[tuple[HeadType | None, ...], ...]
    record_kinds: tuple[tuple[StateRecordKind | None, ...], ...]
    retrieval_eligible_mask: Tensor
    cloned_records: tuple[tuple[StateRecord | None, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.embeddings.ndim != 3
            or self.embeddings.shape[-1] != 512
            or not torch.is_floating_point(self.embeddings)
        ):
            raise ValueError("StateBankView embeddings must be floating [B, N, 512]")
        shape = self.embeddings.shape[:2]
        if (
            self.present_mask.shape != shape
            or self.record_valid_mask.shape != shape
            or self.present_mask.dtype != torch.bool
            or self.record_valid_mask.dtype != torch.bool
            or self.present_mask.device != self.embeddings.device
            or self.record_valid_mask.device != self.embeddings.device
        ):
            raise ValueError("StateBankView masks must be bool [B, N]")
        if (
            self.timestamps.shape != shape
            or self.timestamps.dtype != torch.float64
            or self.timestamps.device != self.embeddings.device
            or self.time_ranges.shape != (*shape, 2)
            or self.time_ranges.dtype != torch.float64
            or self.time_ranges.device != self.embeddings.device
        ):
            raise ValueError("StateBankView time metadata has invalid shape/dtype/device")
        if (
            self.n_state.shape != (shape[0],)
            or self.n_state.dtype != torch.int64
            or self.n_state.device != self.embeddings.device
        ):
            raise ValueError("StateBankView n_state must be int64 [B]")
        if (
            self.owner_record_counts.shape != (shape[0],)
            or self.owner_record_counts.dtype != torch.int64
            or self.owner_record_counts.device != self.embeddings.device
        ):
            raise ValueError("StateBankView owner_record_counts must be int64 [B]")
        if (
            len(self.video_ids) != shape[0]
            or len(self.trajectory_ids) != shape[0]
            or len(self.bank_versions) != shape[0]
            or len(self.record_ids) != shape[0]
            or len(self.head_types) != shape[0]
            or len(self.record_kinds) != shape[0]
            or len(self.cloned_records) != shape[0]
        ):
            raise ValueError("StateBankView owner metadata must match batch size")
        if any(not value for value in self.video_ids + self.trajectory_ids):
            raise ValueError("StateBankView owner identifiers must be non-empty")
        if len(set(zip(self.video_ids, self.trajectory_ids, strict=True))) != shape[0]:
            raise ValueError("StateBankView owners must be unique")
        if any(type(version) is not int or version < 0 for version in self.bank_versions):
            raise ValueError("StateBankView bank versions must be non-negative integers")
        if any(
            len(row) != shape[1]
            for row in (self.record_ids + self.head_types + self.record_kinds + self.cloned_records)
        ):
            raise ValueError("StateBankView owner metadata must match padded record width")
        if (
            self.retrieval_eligible_mask.shape != shape
            or self.retrieval_eligible_mask.dtype != torch.bool
            or self.retrieval_eligible_mask.device != self.embeddings.device
        ):
            raise ValueError("StateBankView retrieval eligibility must be bool [B, N]")
        _validate_state_bank_view_records(self)
        if self.embeddings.device.type != "meta":
            if bool(torch.any(self.record_valid_mask & ~self.present_mask)):
                raise ValueError("StateBankView valid records must also be present")
            if not bool(torch.isfinite(self.embeddings).all()):
                raise ValueError("StateBankView embeddings must be finite")
            if bool(torch.any(self.embeddings[~self.present_mask] != 0.0)):
                raise ValueError("StateBankView padding embeddings must be zero")
            expected_retrieval = self.present_mask & self.record_valid_mask
            for row, kinds in enumerate(self.record_kinds):
                for column, kind in enumerate(kinds):
                    if kind in (None, StateRecordKind.O2_CANDIDATE):
                        expected_retrieval[row, column] = False
            if not torch.equal(self.retrieval_eligible_mask, expected_retrieval):
                raise ValueError(
                    "retrieval eligibility must exclude invalid and O2 Candidate records"
                )
            if not torch.equal(self.n_state, self.present_mask.sum(dim=1)):
                raise ValueError("StateBankView n_state must count stored records")
            if bool(torch.any(self.owner_record_counts < self.n_state)):
                raise ValueError("owner_record_counts cannot be smaller than the head partition")
            if bool(torch.any(self.timestamps[~self.present_mask] != -1.0)) or bool(
                torch.any(self.time_ranges[~self.present_mask] != -1.0)
            ):
                raise ValueError("StateBankView padding metadata must use -1")


@dataclass(frozen=True, slots=True)
class RetrievalHistoryView:
    sources: Tensor
    present_mask: Tensor
    record_valid_mask: Tensor
    retrieval_eligible_mask: Tensor
    timestamps: Tensor
    time_ranges: Tensor
    n_state: Tensor
    owner_record_counts: Tensor
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    bank_versions: tuple[int, ...]
    record_ids: tuple[tuple[str | None, ...], ...]
    head_types: tuple[tuple[HeadType | None, ...], ...]
    record_kinds: tuple[tuple[StateRecordKind | None, ...], ...]
    cloned_records: tuple[tuple[RetrievalHistoryRecord | None, ...], ...]
    lifecycle_ids: tuple[tuple[str | None, ...], ...] = ()
    sequence_ids: Tensor | None = None
    head_codes: Tensor | None = None
    operator_codes: Tensor | None = None
    ring_guards: tuple[tuple[TensorizedRetrievalHistory, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.sources.ndim != 3
            or self.sources.shape[-1] != 768
            or not torch.is_floating_point(self.sources)
        ):
            raise ValueError("RetrievalHistoryView sources must be floating [B, N, 768]")
        shape = self.sources.shape[:2]
        if self.sequence_ids is None or self.head_codes is None or self.operator_codes is None:
            sequence_ids = torch.full(shape, -1, dtype=torch.int64, device=self.sources.device)
            head_codes = torch.full_like(sequence_ids, -1)
            operator_codes = torch.full_like(sequence_ids, -1)
            from ttt_svcbench_qwen.query_encoder import OPERATORS

            for row, records in enumerate(self.cloned_records):
                for column, record in enumerate(records):
                    if record is None:
                        continue
                    try:
                        sequence_ids[row, column] = int(record.record_id.rsplit("-", 1)[-1])
                    except ValueError:
                        sequence_ids[row, column] = column
                    head_codes[row, column] = RETRIEVAL_HEAD_ORDER.index(record.head_type)
                    operator_codes[row, column] = OPERATORS.index(record.operator)
            object.__setattr__(self, "sequence_ids", sequence_ids)
            object.__setattr__(self, "head_codes", head_codes)
            object.__setattr__(self, "operator_codes", operator_codes)
        assert self.sequence_ids is not None
        assert self.head_codes is not None
        assert self.operator_codes is not None
        masks = (self.present_mask, self.record_valid_mask, self.retrieval_eligible_mask)
        if any(mask.shape != shape or mask.dtype != torch.bool for mask in masks):
            raise ValueError("RetrievalHistoryView masks must be bool [B, N]")
        if any(mask.device != self.sources.device for mask in masks):
            raise ValueError("RetrievalHistoryView tensors must share one device")
        if (
            self.timestamps.shape != shape
            or self.timestamps.dtype != torch.float64
            or self.time_ranges.shape != (*shape, 2)
            or self.time_ranges.dtype != torch.float64
            or self.timestamps.device != self.sources.device
            or self.time_ranges.device != self.sources.device
        ):
            raise ValueError("RetrievalHistoryView time metadata is invalid")
        integer_metadata = (self.sequence_ids, self.head_codes, self.operator_codes)
        if any(value.shape != shape or value.dtype != torch.int64 for value in integer_metadata):
            raise ValueError("RetrievalHistoryView tensor metadata must be int64 [B, N]")
        if any(value.device != self.sources.device for value in integer_metadata):
            raise ValueError("RetrievalHistoryView tensor metadata must share the source device")
        batch_size, width = shape
        for counts, name in (
            (self.n_state, "n_state"),
            (self.owner_record_counts, "owner_record_counts"),
        ):
            if counts.shape != (batch_size,) or counts.dtype != torch.int64:
                raise ValueError(f"RetrievalHistoryView {name} must be int64 [B]")
        metadata = (
            self.video_ids,
            self.trajectory_ids,
            self.bank_versions,
            self.record_ids,
            self.head_types,
            self.record_kinds,
            self.cloned_records,
            self.lifecycle_ids or tuple(() for _ in range(batch_size)),
        )
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("RetrievalHistoryView metadata must align to batch size")
        if any(
            len(row) != width
            for row in self.record_ids + self.head_types + self.record_kinds + self.cloned_records
        ):
            raise ValueError("RetrievalHistoryView metadata must align to padded width")
        if self.lifecycle_ids and any(len(row) != width for row in self.lifecycle_ids):
            raise ValueError("RetrievalHistoryView lifecycle metadata must align to padded width")
        if self.sources.device.type != "meta":
            if not bool(torch.isfinite(self.sources).all()):
                raise ValueError("RetrievalHistoryView sources must be finite")
            if bool(torch.any(self.sources[~self.present_mask] != 0.0)):
                raise ValueError("RetrievalHistoryView padding sources must be zero")
            if bool(torch.any(self.record_valid_mask & ~self.present_mask)) or bool(
                torch.any(self.retrieval_eligible_mask & ~self.record_valid_mask)
            ):
                raise ValueError("RetrievalHistoryView masks are inconsistent")
            if not torch.equal(self.n_state, self.present_mask.sum(dim=1)):
                raise ValueError("RetrievalHistoryView n_state must count present records")
            if bool(torch.any(self.sequence_ids[self.present_mask] < 0)) or bool(
                torch.any(self.sequence_ids[~self.present_mask] != -1)
            ):
                raise ValueError("RetrievalHistoryView sequence IDs are inconsistent")

    def assert_snapshot_current(self) -> None:
        if self.ring_guards and len(self.ring_guards) != len(self.video_ids):
            raise ValueError("retrieval snapshot ring guards must align to batch rows")
        for history, version in self.ring_guards:
            if history.released or history.version != version:
                raise RuntimeError("retrieval history changed after its read-only snapshot")


@torch.no_grad()  # type: ignore[untyped-decorator]
def tensorized_retrieval_view(
    histories: Sequence[TensorizedRetrievalHistory],
    *,
    guard_current_version: bool = True,
) -> RetrievalHistoryView:
    """Gather each four-head ring once and restore global sequence order."""

    normalized = tuple(histories)
    if not normalized or any(
        not isinstance(item, TensorizedRetrievalHistory) for item in normalized
    ):
        raise ValueError("tensor retrieval view requires at least one ring")
    if any(item.released for item in normalized):
        raise RuntimeError("released tensor retrieval histories cannot be viewed")
    owners = tuple((item.video_id, item.trajectory_id) for item in normalized)
    if len(set(owners)) != len(owners):
        raise ValueError("tensor retrieval view owners must be unique")
    reference = normalized[0].sources
    if any(
        item.sources.dtype != reference.dtype or item.sources.device != reference.device
        for item in normalized
    ):
        raise ValueError("tensor retrieval rings must share dtype/device")

    gathered: list[dict[str, object]] = []
    for history in normalized:
        source_rows: list[Tensor] = []
        sequence_rows: list[Tensor] = []
        operator_rows: list[Tensor] = []
        timestamp_rows: list[Tensor] = []
        range_rows: list[Tensor] = []
        valid_rows: list[Tensor] = []
        eligible_rows: list[Tensor] = []
        head_rows: list[Tensor] = []
        lifecycle_rows: list[str | None] = []
        for head_code, size in enumerate(history.sizes):
            if size == 0:
                continue
            if size < history.capacity_per_head:
                physical = torch.arange(size, dtype=torch.int64, device=reference.device)
            else:
                physical = (
                    torch.arange(size, dtype=torch.int64, device=reference.device)
                    + history.write_ptrs[head_code]
                ) % history.capacity_per_head
            source_rows.append(history.sources[head_code].index_select(0, physical))
            sequence_rows.append(history.sequence_ids[head_code].index_select(0, physical))
            operator_rows.append(history.operator_codes[head_code].index_select(0, physical))
            timestamp_rows.append(history.timestamps[head_code].index_select(0, physical))
            range_rows.append(history.time_ranges[head_code].index_select(0, physical))
            valid_rows.append(history.valid_mask[head_code].index_select(0, physical))
            eligible_rows.append(history.eligible_mask[head_code].index_select(0, physical))
            head_rows.append(torch.full_like(physical, head_code))
            if any(value is not None for value in history.lifecycle_ids[head_code]):
                lifecycle_rows.extend(
                    history.lifecycle_ids[head_code][index]
                    for index in physical.detach().cpu().tolist()
                )
            else:
                lifecycle_rows.extend((None,) * size)
        if source_rows:
            sequence = torch.cat(sequence_rows)
            order = torch.argsort(sequence, stable=True)
            lifecycle_order = (
                order.detach().cpu().tolist()
                if any(value is not None for value in lifecycle_rows)
                else []
            )
            gathered.append(
                {
                    "sources": torch.cat(source_rows).index_select(0, order),
                    "sequence": sequence.index_select(0, order),
                    "operator": torch.cat(operator_rows).index_select(0, order),
                    "timestamp": torch.cat(timestamp_rows).index_select(0, order),
                    "ranges": torch.cat(range_rows).index_select(0, order),
                    "valid": torch.cat(valid_rows).index_select(0, order),
                    "eligible": torch.cat(eligible_rows).index_select(0, order),
                    "head": torch.cat(head_rows).index_select(0, order),
                    "lifecycle": tuple(lifecycle_rows[index] for index in lifecycle_order)
                    if lifecycle_order
                    else (None,) * len(lifecycle_rows),
                }
            )
        else:
            gathered.append(
                {
                    "sources": reference.new_empty((0, history.source_dim)),
                    "sequence": history.sequence_ids.new_empty((0,)),
                    "operator": history.operator_codes.new_empty((0,)),
                    "timestamp": history.timestamps.new_empty((0,)),
                    "ranges": history.time_ranges.new_empty((0, 2)),
                    "valid": history.valid_mask.new_empty((0,)),
                    "eligible": history.eligible_mask.new_empty((0,)),
                    "head": history.sequence_ids.new_empty((0,)),
                    "lifecycle": (),
                }
            )

    batch_size = len(normalized)
    width = max(history.count for history in normalized)
    sources = reference.new_zeros((batch_size, width, reference.shape[-1]))
    present = torch.zeros((batch_size, width), dtype=torch.bool, device=reference.device)
    valid = torch.zeros_like(present)
    eligible = torch.zeros_like(present)
    timestamps = torch.full((batch_size, width), -1.0, dtype=torch.float64, device=reference.device)
    ranges = torch.full((batch_size, width, 2), -1.0, dtype=torch.float64, device=reference.device)
    sequences = torch.full((batch_size, width), -1, dtype=torch.int64, device=reference.device)
    heads = torch.full_like(sequences, -1)
    operators = torch.full_like(sequences, -1)
    n_state = torch.zeros(batch_size, dtype=torch.int64, device=reference.device)
    lifecycle_ids: list[tuple[str | None, ...]] = []
    for row, values in enumerate(gathered):
        count = normalized[row].count
        if count:
            sources[row, :count] = cast(Tensor, values["sources"])
            sequences[row, :count] = cast(Tensor, values["sequence"])
            operators[row, :count] = cast(Tensor, values["operator"])
            timestamps[row, :count] = cast(Tensor, values["timestamp"])
            ranges[row, :count] = cast(Tensor, values["ranges"])
            valid[row, :count] = cast(Tensor, values["valid"])
            eligible[row, :count] = cast(Tensor, values["eligible"])
            heads[row, :count] = cast(Tensor, values["head"])
            present[row, :count] = True
            n_state[row] = count
        lifecycle = cast(tuple[str | None, ...], values["lifecycle"])
        lifecycle_ids.append(lifecycle + (None,) * (width - count))
    # Tensor-ring snapshots deliberately keep the full candidate axis tensor-only.
    # Python records/IDs/head enums are created lazily for selected audit rows only.
    empty_metadata = tuple((None,) * width for _ in normalized)
    return RetrievalHistoryView(
        sources=sources,
        present_mask=present,
        record_valid_mask=valid,
        retrieval_eligible_mask=eligible,
        timestamps=timestamps,
        time_ranges=ranges,
        sequence_ids=sequences,
        head_codes=heads,
        operator_codes=operators,
        n_state=n_state,
        owner_record_counts=n_state.clone(),
        video_ids=tuple(item.video_id for item in normalized),
        trajectory_ids=tuple(item.trajectory_id for item in normalized),
        bank_versions=tuple(item.version for item in normalized),
        record_ids=empty_metadata,
        head_types=empty_metadata,
        record_kinds=empty_metadata,
        cloned_records=empty_metadata,
        lifecycle_ids=tuple(lifecycle_ids),
        ring_guards=(
            tuple((item, item.version) for item in normalized)
            if guard_current_version
            else ()
        ),
    )


class SemanticProjector(nn.Module):  # type: ignore[misc]
    HEAD_TYPE_ORDER: ClassVar[tuple[HeadType, ...]] = (
        HeadType.O1,
        HeadType.O2,
        HeadType.E1,
        HeadType.E2,
    )

    def __init__(self, config: SemanticProjectorConfig) -> None:
        super().__init__()
        _validate_semantic_projector_config(config)
        self.config = config
        self.head_type_embeddings = nn.Embedding(config.head_type_count, config.input_dim)
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.hidden_projection = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim, bias=True)

    def forward(
        self,
        source_states: Tensor,
        head_types: HeadType | Sequence[HeadType],
    ) -> Tensor:
        if (
            source_states.ndim < 1
            or source_states.shape[-1] != self.config.input_dim
            or not torch.is_floating_point(source_states)
        ):
            raise ValueError("semantic source states must be floating [..., 768]")
        parameter = next(self.parameters())
        if source_states.dtype != parameter.dtype or source_states.device != parameter.device:
            raise ValueError("Semantic Projector and source states must share dtype/device")
        if source_states.device.type != "meta" and not bool(torch.isfinite(source_states).all()):
            raise ValueError("semantic source states must be finite")
        flattened = source_states.reshape(-1, self.config.input_dim)
        normalized_heads = _normalize_head_types(head_types, flattened.shape[0])
        indices = torch.tensor(
            [self.HEAD_TYPE_ORDER.index(head_type) for head_type in normalized_heads],
            dtype=torch.int64,
            device=source_states.device,
        )
        conditioned = flattened + self.head_type_embeddings(indices)
        hidden = F.silu(self.hidden_projection(self.input_norm(conditioned)))
        raw = self.output_projection(hidden)
        normalized = _normalize_semantic(raw, self.config.normalization_eps)
        return normalized.reshape(*source_states.shape[:-1], self.config.output_dim)

    def forward_codes(self, source_states: Tensor, head_codes: Tensor) -> Tensor:
        """Project tensor-ring candidates without materializing Python head enums."""

        if (
            source_states.ndim < 1
            or source_states.shape[-1] != self.config.input_dim
            or not torch.is_floating_point(source_states)
        ):
            raise ValueError("semantic source states must be floating [..., 768]")
        if head_codes.shape != source_states.shape[:-1] or head_codes.dtype != torch.int64:
            raise ValueError("head_codes must be int64 and align to semantic source rows")
        parameter = next(self.parameters())
        if (
            source_states.dtype != parameter.dtype
            or source_states.device != parameter.device
            or head_codes.device != source_states.device
        ):
            raise ValueError("Semantic Projector sources/codes must share parameter device")
        if source_states.device.type != "meta":
            if not bool(torch.isfinite(source_states).all()):
                raise ValueError("semantic source states must be finite")
            if bool(torch.any((head_codes < 0) | (head_codes >= len(self.HEAD_TYPE_ORDER)))):
                raise ValueError("semantic head codes are outside the four-head range")
        flattened = source_states.reshape(-1, self.config.input_dim)
        indices = head_codes.reshape(-1)
        conditioned = flattened + self.head_type_embeddings(indices)
        hidden = F.silu(self.hidden_projection(self.input_norm(conditioned)))
        raw = self.output_projection(hidden)
        normalized = _normalize_semantic(raw, self.config.normalization_eps)
        return normalized.reshape(*source_states.shape[:-1], self.config.output_dim)

    def set_online_frozen(self, frozen: bool = True) -> SemanticProjector:
        if type(frozen) is not bool:
            raise TypeError("online frozen flag must be bool")
        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.eval()
        return self


class StructuredStateBank(nn.Module):  # type: ignore[misc]
    """Model-owned projector plus parameter-free functional hard-state operators."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        _validate_state_bank_config(config.state_bank)
        self.config = config.state_bank
        self.o1_config = config.observation_heads.o1
        self.e1_config = config.observation_heads.e1
        self.e2_config = config.observation_heads.e2
        self.semantic_projector = SemanticProjector(self.config.semantic_projector)

    def project(
        self,
        source_states: Tensor,
        head_types: HeadType | Sequence[HeadType],
    ) -> Tensor:
        """Compute trainable soft semantics before entering any hard no-grad write."""

        return self.semantic_projector(source_states, head_types)

    def project_codes(self, source_states: Tensor, head_codes: Tensor) -> Tensor:
        return self.semantic_projector.forward_codes(source_states, head_codes)

    def reset(self, video_id: str, trajectory_id: str) -> StateBankRuntimeState:
        if not video_id or not trajectory_id:
            raise ValueError("State Bank reset requires non-empty owner identifiers")
        return StateBankRuntimeState(video_id, trajectory_id, (), ())

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def clear(self, state: StateBankRuntimeState) -> StateBankRuntimeState:
        """Clear one live trajectory while preserving its never-reuse ID tombstones."""

        _require_live_state(state)
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=(),
            audit_log=(),
            retrieval_history=(),
            issued_record_ids=state.issued_record_ids,
            next_record_sequence=state.next_record_sequence,
            next_retrieval_sequence=state.next_retrieval_sequence,
            released=False,
            version=state.version + 1,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def release(self, state: StateBankRuntimeState) -> StateBankRuntimeState:
        _require_live_state(state)
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=(),
            audit_log=(),
            retrieval_history=(),
            issued_record_ids=(),
            next_record_sequence=0,
            next_retrieval_sequence=0,
            released=True,
            version=state.version + 1,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def snapshot(self, state: StateBankRuntimeState) -> StateBankRuntimeState:
        _require_live_state(state)
        return _clone_runtime_state(state)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def restore(self, snapshot: StateBankRuntimeState) -> StateBankRuntimeState:
        _require_live_state(snapshot)
        return _clone_runtime_state(snapshot)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def append_record(
        self,
        state: StateBankRuntimeState,
        *,
        head_type: HeadType,
        semantic_embedding: Tensor,
        timestamp: float | None,
        time_range: tuple[float, float] | None,
        valid: bool,
        confidence: float,
        payload: StatePayload,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        if not isinstance(head_type, HeadType):
            raise TypeError("append_record requires a HeadType")
        if head_type is not HeadType.O2 and any(
            record.head_type is head_type for record in state.records
        ):
            raise ValueError(f"{head_type.value} partition already has its aggregate record")
        issued = state.issued_record_ids
        record_id, next_sequence = _next_available_record_id(state)
        record = StateRecord(
            record_id=record_id,
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            head_type=head_type,
            semantic_embedding=_hard_semantic(semantic_embedding, self.config.semantic_projector),
            timestamp=timestamp,
            time_range=time_range,
            valid=valid,
            confidence=confidence,
            payload=_clone_payload(payload),
        )
        audit_time = _canonical_audit_time(state, _record_audit_time(record))
        audit = StateBankAuditEntry(
            action="append",
            record_id=record.record_id,
            timestamp=audit_time,
            details=(("head_type", head_type.value),),
        )
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=tuple(_clone_record(item) for item in state.records) + (_clone_record(record),),
            audit_log=state.audit_log + (audit,),
            retrieval_history=tuple(
                _clone_retrieval_record(item) for item in state.retrieval_history
            ),
            issued_record_ids=issued + (record.record_id,),
            next_record_sequence=next_sequence,
            next_retrieval_sequence=state.next_retrieval_sequence,
            released=False,
            version=state.version + 1,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def append_retrieval_history(
        self,
        state: StateBankRuntimeState,
        *,
        head_type: HeadType,
        operator: Operator,
        semantic_source: Tensor,
        timestamp: float | None,
        time_range: tuple[float, float] | None,
        valid: bool = True,
        retrieval_eligible: bool = True,
        lifecycle_id: str | None = None,
    ) -> StateBankRuntimeState:
        """Append one immutable source record without changing aggregate topology."""

        _require_live_state(state)
        record_id = f"retrieval-{state.next_retrieval_sequence:08d}"
        record = RetrievalHistoryRecord(
            record_id=record_id,
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            head_type=head_type,
            operator=operator,
            semantic_source=semantic_source.detach().clone(),
            timestamp=timestamp,
            time_range=time_range,
            valid=valid,
            retrieval_eligible=retrieval_eligible,
            lifecycle_id=lifecycle_id,
        )
        history = [_clone_retrieval_record(item) for item in state.retrieval_history]
        same_head = [index for index, item in enumerate(history) if item.head_type is head_type]
        capacity = self.config.retrieval_history_capacity_per_head
        if len(same_head) >= capacity:
            del history[same_head[0]]
        history.append(_clone_retrieval_record(record))
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=tuple(_clone_record(item) for item in state.records),
            audit_log=tuple(state.audit_log),
            retrieval_history=tuple(history),
            issued_record_ids=tuple(state.issued_record_ids),
            next_record_sequence=state.next_record_sequence,
            next_retrieval_sequence=state.next_retrieval_sequence + 1,
            released=False,
            version=state.version,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_record(
        self,
        state: StateBankRuntimeState,
        record: StateRecord,
        *,
        action: str = "update",
        details: tuple[tuple[str, AuditValue], ...] = (),
        audit_timestamp: float | None = None,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        index = _find_record_index(state, record.record_id)
        previous = state.records[index]
        if (
            record.video_id != state.video_id
            or record.trajectory_id != state.trajectory_id
            or record.head_type is not previous.head_type
            or type(record.payload) is not type(previous.payload)
        ):
            raise ValueError("State Bank replacement cannot change owner/head/payload type")
        if (
            isinstance(previous.payload, E1Payload)
            and isinstance(record.payload, E1Payload)
            and (record.payload.event_kind is not previous.payload.event_kind)
        ):
            raise ValueError("State Bank replacement cannot change E1 event kind")
        if (
            isinstance(previous.payload, E2Payload)
            and isinstance(record.payload, E2Payload)
            and (record.payload.event_kind is not previous.payload.event_kind)
        ):
            raise ValueError("State Bank replacement cannot change E2 event kind")
        if not previous.valid:
            raise ValueError("invalidated State Bank records are terminal")
        if record.valid != previous.valid:
            detail_map = dict(details)
            valid_invalidation = (
                not record.valid
                and action == "invalidate"
                and audit_timestamp is not None
                and isinstance(detail_map.get("reason"), str)
                and bool(detail_map["reason"])
            )
            if not valid_invalidation:
                raise ValueError("record validity can only change through explicit invalidation")
        records = [_clone_record(item) for item in state.records]
        records[index] = _clone_record(record)
        audit = StateBankAuditEntry(
            action=action,
            record_id=record.record_id,
            timestamp=_canonical_audit_time(
                state,
                _record_audit_time(record) if audit_timestamp is None else audit_timestamp,
            ),
            details=details,
        )
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=tuple(records),
            audit_log=state.audit_log + (audit,),
            retrieval_history=tuple(
                _clone_retrieval_record(item) for item in state.retrieval_history
            ),
            issued_record_ids=state.issued_record_ids,
            next_record_sequence=state.next_record_sequence,
            next_retrieval_sequence=state.next_retrieval_sequence,
            released=False,
            version=state.version + 1,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def invalidate_record(
        self,
        state: StateBankRuntimeState,
        record_id: str,
        *,
        audit_timestamp: float,
        reason: str,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        if not reason or not math.isfinite(audit_timestamp) or audit_timestamp < 0.0:
            raise ValueError("record invalidation requires a legal timestamp and reason")
        previous = state.records[_find_record_index(state, record_id)]
        if not previous.valid:
            return _append_runtime_audit(
                state,
                action="invalidate_duplicate",
                record_id=record_id,
                timestamp=audit_timestamp,
                details=(("reason", reason),),
            )
        lifecycle_id = (
            previous.payload.identity_id
            if isinstance(previous.payload, ConfirmedIdentity)
            else None
        )
        disabled_history_count = (
            sum(
                record.lifecycle_id == lifecycle_id and record.retrieval_eligible
                for record in state.retrieval_history
            )
            if lifecycle_id is not None
            else 0
        )
        replacement = replace(previous, valid=False)
        updated = cast(
            StateBankRuntimeState,
            self.update_record(
                state,
                replacement,
                action="invalidate",
                details=(
                    ("reason", reason),
                    ("audit_timestamp", audit_timestamp),
                    ("retrieval_history_disabled", disabled_history_count),
                ),
                audit_timestamp=audit_timestamp,
            ),
        )
        if lifecycle_id is None or disabled_history_count == 0:
            return updated
        return replace(
            updated,
            retrieval_history=tuple(
                _clone_retrieval_record(
                    replace(record, retrieval_eligible=False)
                    if record.lifecycle_id == lifecycle_id
                    else record
                )
                for record in updated.retrieval_history
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def append_o2_candidate(
        self,
        state: StateBankRuntimeState,
        *,
        semantic_embedding: Tensor,
        candidate: CandidateIdentity,
        confidence: float,
    ) -> tuple[StateBankRuntimeState, StateRecord]:
        """Append one linked Candidate record without exposing ID allocation to P10."""

        _require_live_state(state)
        if type(candidate) is not CandidateIdentity:
            raise TypeError("append_o2_candidate requires a CandidateIdentity payload")
        expected_record_id, _ = _next_available_record_id(state)
        linked_payload = cast(
            CandidateIdentity,
            _with_semantic_record_link(candidate, expected_record_id),
        )
        next_state = self.append_record(
            state,
            head_type=HeadType.O2,
            semantic_embedding=semantic_embedding,
            timestamp=candidate.first_seen,
            time_range=None,
            valid=True,
            confidence=confidence,
            payload=linked_payload,
        )
        record = next_state.records[_find_record_index(next_state, expected_record_id)]
        return next_state, _clone_record(record)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_o2_candidate(
        self,
        state: StateBankRuntimeState,
        *,
        semantic_embedding: Tensor,
        confidence: float,
        candidate: CandidateIdentity,
        audit_timestamp: float,
        details: tuple[tuple[str, AuditValue], ...] = (),
    ) -> StateBankRuntimeState:
        """Functionally update one Candidate while preserving its first-seen record time."""

        record_id = _require_semantic_record_link(candidate)
        prior = _require_o2_payload(state, record_id, CandidateIdentity)
        linked_payload = cast(CandidateIdentity, _with_semantic_record_link(candidate, record_id))
        replacement = StateRecord(
            record_id=prior.record_id,
            video_id=prior.video_id,
            trajectory_id=prior.trajectory_id,
            head_type=HeadType.O2,
            semantic_embedding=_hard_semantic(
                semantic_embedding,
                self.config.semantic_projector,
            ),
            timestamp=prior.timestamp,
            time_range=None,
            valid=True,
            confidence=confidence,
            payload=linked_payload,
        )
        return cast(
            StateBankRuntimeState,
            self.update_record(
                state,
                replacement,
                action="o2_candidate_update",
                details=details,
                audit_timestamp=audit_timestamp,
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def invalidate_o2_candidate(
        self,
        state: StateBankRuntimeState,
        record_id: str,
        *,
        audit_timestamp: float,
        reason: str,
    ) -> StateBankRuntimeState:
        """Invalidate exactly one Candidate link; invalid records remain terminal tombstones."""

        _require_o2_payload(state, record_id, CandidateIdentity)
        return cast(
            StateBankRuntimeState,
            self.invalidate_record(
                state,
                record_id,
                audit_timestamp=audit_timestamp,
                reason=reason,
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def promote_o2_candidate(
        self,
        state: StateBankRuntimeState,
        candidate_record_id: str,
        *,
        semantic_embedding: Tensor,
        confirmed: ConfirmedIdentity,
        confidence: float,
        audit_timestamp: float,
        reason: str = "candidate_promoted",
    ) -> tuple[StateBankRuntimeState, StateRecord]:
        """Atomically invalidate a Candidate and append a new linked Confirmed record."""

        candidate_record = _require_o2_payload(state, candidate_record_id, CandidateIdentity)
        if not candidate_record.valid:
            raise ValueError("invalidated O2 Candidate records cannot be promoted")
        if type(confirmed) is not ConfirmedIdentity:
            raise TypeError("promotion requires a ConfirmedIdentity payload")
        if audit_timestamp < confirmed.first_seen:
            raise ValueError("promotion timestamp cannot precede first_seen")
        invalidated = self.invalidate_o2_candidate(
            state,
            candidate_record_id,
            audit_timestamp=audit_timestamp,
            reason=reason,
        )
        confirmed_record_id, _ = _next_available_record_id(invalidated)
        linked_payload = cast(
            ConfirmedIdentity,
            _with_semantic_record_link(confirmed, confirmed_record_id),
        )
        promoted = self.append_record(
            invalidated,
            head_type=HeadType.O2,
            semantic_embedding=semantic_embedding,
            timestamp=confirmed.first_seen,
            time_range=None,
            valid=True,
            confidence=confidence,
            payload=linked_payload,
        )
        promoted = _append_runtime_audit(
            promoted,
            action="o2_candidate_promoted",
            record_id=confirmed_record_id,
            timestamp=audit_timestamp,
            details=(("candidate_record_id", candidate_record_id),),
        )
        record = promoted.records[_find_record_index(promoted, confirmed_record_id)]
        return promoted, _clone_record(record)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_o2_confirmed(
        self,
        state: StateBankRuntimeState,
        *,
        semantic_embedding: Tensor,
        confidence: float,
        confirmed: ConfirmedIdentity,
        audit_timestamp: float,
        details: tuple[tuple[str, AuditValue], ...] = (),
    ) -> StateBankRuntimeState:
        """Update Confirmed evidence without changing its first-seen retrieval timestamp."""

        record_id = _require_semantic_record_link(confirmed)
        prior = _require_o2_payload(state, record_id, ConfirmedIdentity)
        prior_payload = cast(ConfirmedIdentity, prior.payload)
        if (
            prior.timestamp != prior_payload.first_seen
            or confirmed.first_seen != prior_payload.first_seen
        ):
            raise ValueError("Confirmed updates cannot change first_seen")
        linked_payload = cast(ConfirmedIdentity, _with_semantic_record_link(confirmed, record_id))
        replacement = StateRecord(
            record_id=prior.record_id,
            video_id=prior.video_id,
            trajectory_id=prior.trajectory_id,
            head_type=HeadType.O2,
            semantic_embedding=_hard_semantic(
                semantic_embedding,
                self.config.semantic_projector,
            ),
            timestamp=prior.timestamp,
            time_range=None,
            valid=True,
            confidence=confidence,
            payload=linked_payload,
        )
        return cast(
            StateBankRuntimeState,
            self.update_record(
                state,
                replacement,
                action="o2_confirmed_update",
                details=details,
                audit_timestamp=audit_timestamp,
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def records_for(
        self,
        state: StateBankRuntimeState,
        head_type: HeadType | None = None,
        *,
        include_invalid: bool = True,
    ) -> tuple[StateRecord, ...]:
        _require_live_state(state)
        if head_type is not None and not isinstance(head_type, HeadType):
            raise TypeError("records_for head_type must be a HeadType or None")
        return tuple(
            _clone_record(record)
            for record in state.records
            if (head_type is None or record.head_type is head_type)
            and (include_invalid or record.valid)
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def view(
        self,
        states: Sequence[StateBankRuntimeState],
        head_type: HeadType | Sequence[HeadType | None] | None = None,
    ) -> StateBankView:
        normalized = tuple(states)
        if not normalized or any(
            not isinstance(state, StateBankRuntimeState) for state in normalized
        ):
            raise ValueError("State Bank view requires at least one runtime state")
        for state in normalized:
            _require_live_state(state)
        row_head_types = _normalize_view_head_filter(head_type, len(normalized))
        owners = tuple((state.video_id, state.trajectory_id) for state in normalized)
        if len(set(owners)) != len(owners):
            raise ValueError("State Bank batch owners must be unique")
        source_records = tuple(record for state in normalized for record in state.records)
        _assert_tensor_groups_isolated(
            tuple(_record_tensors(record) for record in source_records),
            "batched State Bank records",
        )
        rows = tuple(
            tuple(
                _clone_record(record)
                for record in state.records
                if row_head_types is None or record.head_type is row_head_types[row]
            )
            if row_head_types is None or row_head_types[row] is not None
            else ()
            for row, state in enumerate(normalized)
        )
        all_records = tuple(record for records in rows for record in records)
        if all_records:
            _assert_tensor_groups_isolated(
                tuple(_record_tensors(record) for record in all_records),
                "batched State Bank records",
            )
            reference = all_records[0].semantic_embedding
            if any(
                record.semantic_embedding.dtype != reference.dtype
                or record.semantic_embedding.device != reference.device
                for record in all_records[1:]
            ):
                raise ValueError("batched State Bank semantics must share dtype/device")
        else:
            parameter = next(self.semantic_projector.parameters())
            reference = torch.empty((), dtype=torch.float32, device=parameter.device)
        batch_size = len(normalized)
        max_records = max(len(records) for records in rows)
        embeddings = reference.new_zeros((batch_size, max_records, self.config.semantic_dim))
        present_mask = torch.zeros(
            (batch_size, max_records), dtype=torch.bool, device=reference.device
        )
        valid_mask = torch.zeros_like(present_mask)
        retrieval_eligible_mask = torch.zeros_like(present_mask)
        timestamps = torch.full(
            (batch_size, max_records), -1.0, dtype=torch.float64, device=reference.device
        )
        time_ranges = torch.full(
            (batch_size, max_records, 2), -1.0, dtype=torch.float64, device=reference.device
        )
        n_state = torch.zeros(batch_size, dtype=torch.int64, device=reference.device)
        owner_record_counts = torch.tensor(
            tuple(len(state.records) for state in normalized),
            dtype=torch.int64,
            device=reference.device,
        )
        record_ids: list[tuple[str | None, ...]] = []
        head_types: list[tuple[HeadType | None, ...]] = []
        record_kinds: list[tuple[StateRecordKind | None, ...]] = []
        cloned_records: list[tuple[StateRecord | None, ...]] = []
        for row, records in enumerate(rows):
            count = len(records)
            n_state[row] = count
            ids: list[str | None] = [None] * max_records
            heads: list[HeadType | None] = [None] * max_records
            kinds: list[StateRecordKind | None] = [None] * max_records
            record_copies: list[StateRecord | None] = [None] * max_records
            for column, record in enumerate(records):
                kind = _record_kind(record)
                embeddings[row, column] = record.semantic_embedding
                present_mask[row, column] = True
                valid_mask[row, column] = record.valid
                retrieval_eligible_mask[row, column] = (
                    record.valid and kind is not StateRecordKind.O2_CANDIDATE
                )
                ids[column] = record.record_id
                heads[column] = record.head_type
                kinds[column] = kind
                record_copies[column] = record
                if record.timestamp is not None:
                    timestamps[row, column] = record.timestamp
                else:
                    assert record.time_range is not None
                    time_ranges[row, column] = torch.tensor(
                        record.time_range, dtype=torch.float64, device=reference.device
                    )
            record_ids.append(tuple(ids))
            head_types.append(tuple(heads))
            record_kinds.append(tuple(kinds))
            cloned_records.append(tuple(record_copies))
        return StateBankView(
            embeddings=embeddings,
            present_mask=present_mask,
            record_valid_mask=valid_mask,
            timestamps=timestamps,
            time_ranges=time_ranges,
            n_state=n_state,
            owner_record_counts=owner_record_counts,
            video_ids=tuple(state.video_id for state in normalized),
            trajectory_ids=tuple(state.trajectory_id for state in normalized),
            bank_versions=tuple(state.version for state in normalized),
            record_ids=tuple(record_ids),
            head_types=tuple(head_types),
            record_kinds=tuple(record_kinds),
            retrieval_eligible_mask=retrieval_eligible_mask,
            cloned_records=tuple(cloned_records),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def retrieval_view(
        self,
        states: Sequence[StateBankRuntimeState],
        head_type: HeadType | Sequence[HeadType | None] | None = None,
    ) -> RetrievalHistoryView:
        """Return detached pre-projector sources for Query-time reprojection."""

        normalized = tuple(states)
        if not normalized or any(
            not isinstance(state, StateBankRuntimeState) for state in normalized
        ):
            raise ValueError("retrieval history view requires at least one runtime state")
        for state in normalized:
            _require_live_state(state)
        row_head_types = _normalize_view_head_filter(head_type, len(normalized))
        owners = tuple((state.video_id, state.trajectory_id) for state in normalized)
        if len(set(owners)) != len(owners):
            raise ValueError("retrieval history view owners must be unique")
        from ttt_svcbench_qwen.query_encoder import OPERATORS

        rows = tuple(
            tuple(
                _clone_retrieval_record(record)
                for record in state.retrieval_history
                if row_head_types is None or record.head_type is row_head_types[row]
            )
            if row_head_types is None or row_head_types[row] is not None
            else ()
            for row, state in enumerate(normalized)
        )
        all_records = tuple(record for records in rows for record in records)
        if all_records:
            reference = all_records[0].semantic_source
            if any(
                record.semantic_source.dtype != reference.dtype
                or record.semantic_source.device != reference.device
                for record in all_records[1:]
            ):
                raise ValueError("retrieval history sources must share dtype/device")
        else:
            parameter = next(self.semantic_projector.parameters())
            reference = torch.empty((), dtype=parameter.dtype, device=parameter.device)
        batch_size = len(normalized)
        width = max(len(records) for records in rows)
        sources = reference.new_zeros((batch_size, width, self.config.retrieval_history_source_dim))
        present = torch.zeros((batch_size, width), dtype=torch.bool, device=reference.device)
        valid = torch.zeros_like(present)
        eligible = torch.zeros_like(present)
        timestamps = torch.full(
            (batch_size, width), -1.0, dtype=torch.float64, device=reference.device
        )
        time_ranges = torch.full(
            (batch_size, width, 2), -1.0, dtype=torch.float64, device=reference.device
        )
        n_state = torch.zeros(batch_size, dtype=torch.int64, device=reference.device)
        owner_counts = torch.tensor(
            tuple(len(state.retrieval_history) for state in normalized),
            dtype=torch.int64,
            device=reference.device,
        )
        record_ids: list[tuple[str | None, ...]] = []
        head_types: list[tuple[HeadType | None, ...]] = []
        record_kinds: list[tuple[StateRecordKind | None, ...]] = []
        cloned_records: list[tuple[RetrievalHistoryRecord | None, ...]] = []
        lifecycle_ids: list[tuple[str | None, ...]] = []
        sequence_ids = torch.full(
            (batch_size, width), -1, dtype=torch.int64, device=reference.device
        )
        head_codes = torch.full_like(sequence_ids, -1)
        operator_codes = torch.full_like(sequence_ids, -1)
        for row, records in enumerate(rows):
            n_state[row] = len(records)
            ids: list[str | None] = [None] * width
            heads: list[HeadType | None] = [None] * width
            kinds: list[StateRecordKind | None] = [None] * width
            copies: list[RetrievalHistoryRecord | None] = [None] * width
            lifecycles: list[str | None] = [None] * width
            for column, record in enumerate(records):
                sources[row, column] = record.semantic_source
                present[row, column] = True
                valid[row, column] = record.valid
                eligible[row, column] = record.retrieval_eligible
                ids[column] = record.record_id
                heads[column] = record.head_type
                kinds[column] = _history_record_kind(record)
                copies[column] = record
                lifecycles[column] = record.lifecycle_id
                sequence_ids[row, column] = int(record.record_id.rsplit("-", 1)[-1])
                head_codes[row, column] = RETRIEVAL_HEAD_ORDER.index(record.head_type)
                operator_codes[row, column] = OPERATORS.index(record.operator)
                if record.timestamp is not None:
                    timestamps[row, column] = record.timestamp
                else:
                    assert record.time_range is not None
                    time_ranges[row, column] = torch.tensor(
                        record.time_range, dtype=torch.float64, device=reference.device
                    )
            record_ids.append(tuple(ids))
            head_types.append(tuple(heads))
            record_kinds.append(tuple(kinds))
            cloned_records.append(tuple(copies))
            lifecycle_ids.append(tuple(lifecycles))
        return RetrievalHistoryView(
            sources=sources,
            present_mask=present,
            record_valid_mask=valid,
            retrieval_eligible_mask=eligible,
            timestamps=timestamps,
            time_ranges=time_ranges,
            sequence_ids=sequence_ids,
            head_codes=head_codes,
            operator_codes=operator_codes,
            n_state=n_state,
            owner_record_counts=owner_counts,
            video_ids=tuple(state.video_id for state in normalized),
            trajectory_ids=tuple(state.trajectory_id for state in normalized),
            bank_versions=tuple(state.version for state in normalized),
            record_ids=tuple(record_ids),
            head_types=tuple(head_types),
            record_kinds=tuple(record_kinds),
            cloned_records=tuple(cloned_records),
            lifecycle_ids=tuple(lifecycle_ids),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_o1(
        self,
        state: StateBankRuntimeState,
        observation: O1SoftOutput,
        semantic_embedding: Tensor,
        *,
        observation_timestamp: float,
        observation_position_id: int,
        row: int = 0,
        set_baseline: bool = False,
        slot_overflow_count: int = 0,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        _validate_row(row, observation.logits.shape[0], "O1")
        if (
            not math.isfinite(observation_timestamp)
            or observation_timestamp < 0.0
            or type(observation_position_id) is not int
            or observation_position_id < 0
            or type(set_baseline) is not bool
            or type(slot_overflow_count) is not int
            or slot_overflow_count < 0
        ):
            raise ValueError("O1 row metadata/baseline/overflow arguments are invalid")
        mask = observation.valid_mask[row]
        if bool(mask.any()):
            if not bool(torch.all(observation.position_ids[row, mask] == observation_position_id)):
                raise ValueError("O1 row position does not match valid slot metadata")
            valid_times = observation.timestamps[row, mask].double()
            expected = torch.full_like(valid_times, observation_timestamp)
            if not torch.allclose(valid_times, expected, atol=1.0e-6, rtol=1.0e-6):
                raise ValueError("O1 row timestamp does not match valid slot metadata")
        prior_record = _find_aggregate_record(state, HeadType.O1)
        prior_payload = (
            prior_record.payload
            if prior_record is not None
            else O1Payload(0, 0, (), baseline_initialized=False)
        )
        assert isinstance(prior_payload, O1Payload)
        if slot_overflow_count < prior_payload.last_spatial_overflow_count:
            raise ValueError("O1 cumulative spatial overflow count cannot decrease")
        overflow_delta = slot_overflow_count - prior_payload.last_spatial_overflow_count
        probabilities = observation.probabilities[row]
        incoming_slots, low_confidence_count, conflict_count, invalid_slot_count = (
            self._decode_o1_row(
                probabilities,
                mask,
                timestamp=observation_timestamp,
                position_id=observation_position_id,
            )
        )
        if observation_position_id <= prior_payload.last_position_id:
            if set_baseline and (
                not prior_payload.baseline_initialized
                or prior_payload.baseline_position_id != observation_position_id
            ):
                raise ValueError("O1 baseline cannot be initialized from replayed evidence")
            assert prior_record is not None
            comparable = observation_position_id == prior_payload.last_position_id
            prior_slots = {slot.slot_id: slot for slot in prior_payload.slot_states}
            drift_count = (
                sum(
                    prior_slots.get(slot_id) is None
                    or not _same_o1_evidence(prior_slots[slot_id], incoming)
                    for slot_id, incoming in incoming_slots.items()
                )
                if comparable
                else 0
            )
            timestamp_drift = comparable and not _float_close(
                observation_timestamp, prior_payload.last_timestamp
            )
            incoming_semantic = _hard_semantic(semantic_embedding, self.config.semantic_projector)
            semantic_drift = comparable and not torch.allclose(
                incoming_semantic,
                prior_record.semantic_embedding,
                atol=1.0e-6,
                rtol=1.0e-5,
            )
            updated_payload = replace(
                prior_payload,
                last_spatial_overflow_count=slot_overflow_count,
            )
            return cast(
                StateBankRuntimeState,
                self.update_record(
                    state,
                    replace(prior_record, payload=updated_payload),
                    action="o1_duplicate_position",
                    audit_timestamp=observation_timestamp,
                    details=(
                        ("position_id", observation_position_id),
                        ("evidence_comparable", comparable),
                        ("timestamp_drift", timestamp_drift),
                        ("slot_evidence_drift_count", drift_count),
                        ("semantic_drift", semantic_drift),
                        ("low_confidence_slots", low_confidence_count),
                        ("enter_exit_conflicts", conflict_count),
                        ("invalid_slots", invalid_slot_count),
                        ("slot_overflow_delta", overflow_delta),
                    ),
                ),
            )
        prior_slots = {slot.slot_id: slot for slot in prior_payload.slot_states}
        slot_states: list[O1SlotState] = []
        active_slot_ids: list[int] = []
        invalid_slot_count += len(set(prior_slots).difference(range(mask.shape[0])))
        reliable_slot_count = 0
        for slot_id in sorted(set(prior_slots) | set(incoming_slots)):
            incoming = incoming_slots.get(slot_id)
            confident = (
                incoming is not None and incoming.confidence >= self.o1_config.confidence_threshold
            )
            conflict = incoming is not None and incoming.enter and incoming.exit
            if incoming is None or not confident or conflict:
                committed = prior_slots.get(slot_id)
                if committed is None:
                    continue
            else:
                committed = incoming
                reliable_slot_count += 1
            slot_states.append(committed)
            if committed.visible:
                active_slot_ids.append(slot_id)
        if set_baseline and prior_payload.baseline_initialized:
            raise ValueError("O1 baseline can only be initialized once per trajectory")
        current_count = len(active_slot_ids)
        baseline_initialized = prior_payload.baseline_initialized or set_baseline
        baseline_count = current_count if set_baseline else prior_payload.baseline_count
        baseline_position = (
            observation_position_id if set_baseline else prior_payload.baseline_position_id
        )
        payload = O1Payload(
            current_visible_count=current_count,
            baseline_count=baseline_count,
            active_slot_ids=tuple(active_slot_ids),
            slot_states=tuple(slot_states),
            baseline_initialized=baseline_initialized,
            baseline_position_id=baseline_position,
            last_timestamp=observation_timestamp,
            last_position_id=observation_position_id,
            update_count=prior_payload.update_count + 1,
            last_spatial_overflow_count=slot_overflow_count,
        )
        confidence = (
            float(probabilities[mask, 5].float().mean().item()) if bool(mask.any()) else 0.0
        )
        semantic_to_store = semantic_embedding
        if prior_record is not None and reliable_slot_count == 0:
            semantic_to_store = prior_record.semantic_embedding
            confidence = prior_record.confidence
        details: tuple[tuple[str, AuditValue], ...] = (
            ("position_id", observation_position_id),
            ("current_visible_count", current_count),
            ("baseline_initialized", baseline_initialized),
            ("low_confidence_slots", low_confidence_count),
            ("enter_exit_conflicts", conflict_count),
            ("invalid_slots", invalid_slot_count),
            ("slot_overflow_delta", overflow_delta),
        )
        return self._upsert_aggregate(
            state,
            head_type=HeadType.O1,
            semantic_embedding=semantic_to_store,
            timestamp=observation_timestamp,
            confidence=confidence,
            payload=payload,
            action="o1_update",
            details=details,
        )

    def _decode_o1_row(
        self,
        probabilities: Tensor,
        mask: Tensor,
        *,
        timestamp: float,
        position_id: int,
    ) -> tuple[dict[int, O1SlotState], int, int, int]:
        slots: dict[int, O1SlotState] = {}
        low_confidence_count = conflict_count = invalid_slot_count = 0
        for slot_id in range(mask.shape[0]):
            if not bool(mask[slot_id].item()):
                invalid_slot_count += 1
                continue
            values = probabilities[slot_id].float()
            confidence = float(values[5].item())
            is_object = bool(values[0] >= self.o1_config.object_threshold)
            is_target = bool(values[1] >= self.o1_config.target_threshold)
            visible_evidence = bool(values[2] >= self.o1_config.visible_threshold)
            enter = bool(values[3] >= self.o1_config.enter_threshold)
            exit_evidence = bool(values[4] >= self.o1_config.exit_threshold)
            confident = confidence >= self.o1_config.confidence_threshold
            conflict = enter and exit_evidence
            low_confidence_count += int(not confident)
            conflict_count += int(conflict)
            slots[slot_id] = O1SlotState(
                slot_id=slot_id,
                is_object=is_object,
                is_target=is_target,
                visible=(
                    is_object
                    and is_target
                    and visible_evidence
                    and confident
                    and not conflict
                    and not exit_evidence
                ),
                enter=enter,
                exit=exit_evidence,
                last_timestamp=timestamp,
                last_position_id=position_id,
                confidence=confidence,
            )
        return slots, low_confidence_count, conflict_count, invalid_slot_count

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_e1(
        self,
        state: StateBankRuntimeState,
        observation: E1SoftOutput,
        semantic_embeddings: Tensor,
        *,
        event_kind: E1EventKind,
        row: int = 0,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        if not isinstance(event_kind, E1EventKind):
            raise TypeError("event_kind must be an E1EventKind")
        _validate_row(row, observation.logits.shape[0], "E1")
        semantics = _select_semantics(semantic_embeddings, observation.logits.shape[:2], row)
        prior_record = _find_aggregate_record(state, HeadType.E1)
        prior = (
            prior_record.payload if prior_record is not None else E1Payload(event_kind, 0, (), 0.0)
        )
        assert isinstance(prior, E1Payload)
        if prior.event_kind is not event_kind:
            raise ValueError("E1 event kind cannot change within a trajectory")
        event_count = prior.event_count
        recent = list(prior.recent_event_times)
        cooldown_until = prior.cooldown_until
        active = prior.active
        armed = prior.armed
        candidate_start = prior.candidate_start
        last_timestamp = prior.last_timestamp
        last_position = prior.last_position_id
        duplicate_count = prior.duplicate_suppression_count
        cooldown_hits = prior.cooldown_hit_count
        nms_hits = prior.nms_suppression_count
        misses = prior.miss_candidate_count
        evictions = prior.history_eviction_count
        delta_duplicates = delta_cooldown = delta_nms = delta_misses = delta_events = 0
        last_new_index: int | None = None
        valid_indices: list[int] = (
            torch.nonzero(observation.valid_mask[row], as_tuple=False).flatten().tolist()
        )
        if not valid_indices:
            return state
        for index in valid_indices:
            position = int(observation.position_ids[row, index].item())
            timestamp = float(observation.timestamps[row, index].item())
            if position <= last_position:
                if position == last_position and not _float_close(timestamp, last_timestamp):
                    raise ValueError("E1 duplicate position timestamp drift")
                duplicate_count += 1
                delta_duplicates += 1
                continue
            if last_position >= 0 and position != last_position + 1:
                raise ValueError("E1 hard FSM positions cannot contain gaps")
            if last_timestamp >= 0.0 and timestamp <= last_timestamp:
                raise ValueError("E1 hard FSM timestamps must increase strictly")
            values = observation.probabilities[row, index].float()
            eventness, completion, transition = values.unbind()
            if active:
                if bool(completion >= self.e1_config.completion_threshold) and bool(
                    transition >= self.e1_config.transition_threshold
                ):
                    if timestamp < cooldown_until:
                        cooldown_hits += 1
                        delta_cooldown += 1
                    elif recent and timestamp - recent[-1] < self.e1_config.min_gap_seconds:
                        nms_hits += 1
                        delta_nms += 1
                    else:
                        event_count += 1
                        delta_events += 1
                        recent.append(timestamp)
                        cooldown_until = timestamp + self.e1_config.min_gap_seconds
                    active = False
                    armed = False
                    candidate_start = None
                elif bool(eventness <= self.e1_config.tau_off):
                    active = False
                    armed = True
                    candidate_start = None
                    misses += 1
                    delta_misses += 1
            elif not armed:
                if bool(eventness <= self.e1_config.tau_off):
                    armed = True
                elif bool(eventness >= self.e1_config.tau_on):
                    duplicate_count += 1
                    delta_duplicates += 1
            elif bool(eventness >= self.e1_config.tau_on):
                if timestamp < cooldown_until:
                    cooldown_hits += 1
                    delta_cooldown += 1
                else:
                    active = True
                    armed = False
                    candidate_start = timestamp
            last_timestamp = timestamp
            last_position = position
            last_new_index = index
        if len(recent) > self.config.event_history_capacity:
            removed = len(recent) - self.config.event_history_capacity
            recent = recent[removed:]
            evictions += removed
        if last_new_index is None:
            assert prior_record is not None
            replacement = replace(
                prior_record,
                payload=replace(
                    prior,
                    duplicate_suppression_count=duplicate_count,
                ),
            )
            return cast(
                StateBankRuntimeState,
                self.update_record(
                    state,
                    replacement,
                    action="e1_overlap_ignored",
                    details=(
                        ("event_kind", event_kind.value),
                        ("duplicate_positions", delta_duplicates),
                    ),
                    audit_timestamp=max(last_timestamp, 0.0),
                ),
            )
        payload = E1Payload(
            event_kind=event_kind,
            event_count=event_count,
            recent_event_times=tuple(recent),
            cooldown_until=cooldown_until,
            active=active,
            armed=armed,
            candidate_start=candidate_start,
            last_timestamp=last_timestamp,
            last_position_id=last_position,
            duplicate_suppression_count=duplicate_count,
            cooldown_hit_count=cooldown_hits,
            nms_suppression_count=nms_hits,
            miss_candidate_count=misses,
            history_eviction_count=evictions,
        )
        confidence = float(observation.probabilities[row, last_new_index].float().max().item())
        return self._upsert_aggregate(
            state,
            head_type=HeadType.E1,
            semantic_embedding=semantics[last_new_index],
            timestamp=last_timestamp,
            confidence=confidence,
            payload=payload,
            action="e1_fsm_update",
            details=(
                ("event_kind", event_kind.value),
                ("position_id", last_position),
                ("events_added", delta_events),
                ("duplicate_positions", delta_duplicates),
                ("cooldown_hits", delta_cooldown),
                ("nms_suppressions", delta_nms),
                ("missed_candidates", delta_misses),
                ("history_evictions", evictions - prior.history_eviction_count),
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update_e2(
        self,
        state: StateBankRuntimeState,
        observation: E2SoftOutput,
        semantic_embeddings: Tensor,
        *,
        event_kind: E2EventKind,
        row: int = 0,
    ) -> StateBankRuntimeState:
        _require_live_state(state)
        if not isinstance(event_kind, E2EventKind):
            raise TypeError("event_kind must be an E2EventKind")
        _validate_row(row, observation.event_logits.shape[0], "E2")
        semantics = _select_semantics(semantic_embeddings, observation.event_logits.shape[:2], row)
        prior_record = _find_aggregate_record(state, HeadType.E2)
        prior = (
            prior_record.payload
            if prior_record is not None
            else E2Payload(event_kind, 0, E2Phase.INACTIVE, (), ())
        )
        assert isinstance(prior, E2Payload)
        if prior.event_kind is not event_kind:
            raise ValueError("E2 event kind cannot change within a trajectory")
        completed_count = prior.completed_count
        intervals = list(prior.completed_intervals)
        recent = list(prior.recent_event_times)
        phase = prior.phase
        current_start = prior.current_start
        last_timestamp = prior.last_timestamp
        last_position = prior.last_position_id
        duplicates = prior.duplicate_suppression_count
        conflicts = prior.conflict_count
        rearm_suppressions = prior.rearm_suppression_count
        evictions = prior.history_eviction_count
        delta_duplicates = delta_conflicts = delta_rearm = delta_completed = 0
        last_new_index: int | None = None
        valid_indices: list[int] = (
            torch.nonzero(observation.valid_mask[row], as_tuple=False).flatten().tolist()
        )
        if not valid_indices:
            return state
        phase_values = tuple(E2Phase)
        for index in valid_indices:
            position = int(observation.position_ids[row, index].item())
            timestamp = float(observation.timestamps[row, index].item())
            if position <= last_position:
                if position == last_position and not _float_close(timestamp, last_timestamp):
                    raise ValueError("E2 duplicate position timestamp drift")
                duplicates += 1
                delta_duplicates += 1
                continue
            if last_position >= 0 and position != last_position + 1:
                raise ValueError("E2 hard FSM positions cannot contain gaps")
            if last_timestamp >= 0.0 and timestamp <= last_timestamp:
                raise ValueError("E2 hard FSM timestamps must increase strictly")
            events = observation.event_probabilities[row, index].float()
            phase_index = int(observation.phase_probabilities[row, index].argmax().item())
            evidence_phase = phase_values[phase_index]
            start, _active_evidence, end, complete = events.unbind()
            if phase is E2Phase.INACTIVE:
                if (
                    bool(start >= self.e2_config.start_threshold)
                    and evidence_phase is E2Phase.ACTIVE
                ):
                    phase = E2Phase.ACTIVE
                    current_start = timestamp
                elif (
                    bool(start >= self.e2_config.start_threshold)
                    or bool(end >= self.e2_config.end_threshold)
                    or bool(complete >= self.e2_config.complete_threshold)
                    or evidence_phase is not E2Phase.INACTIVE
                ):
                    conflicts += 1
                    delta_conflicts += 1
            elif phase is E2Phase.ACTIVE:
                if (
                    bool(end >= self.e2_config.end_threshold)
                    and evidence_phase is E2Phase.END_CANDIDATE
                ):
                    phase = E2Phase.END_CANDIDATE
                elif (
                    bool(end >= self.e2_config.end_threshold)
                    or bool(complete >= self.e2_config.complete_threshold)
                    or evidence_phase is not E2Phase.ACTIVE
                ):
                    conflicts += 1
                    delta_conflicts += 1
            elif phase is E2Phase.END_CANDIDATE:
                if (
                    bool(complete >= self.e2_config.complete_threshold)
                    and evidence_phase is E2Phase.COMPLETED
                ):
                    assert current_start is not None
                    intervals.append((current_start, timestamp))
                    recent.append(timestamp)
                    completed_count += 1
                    delta_completed += 1
                    current_start = None
                    phase = E2Phase.COMPLETED
                elif evidence_phase is not E2Phase.END_CANDIDATE:
                    conflicts += 1
                    delta_conflicts += 1
            else:
                low_event_evidence = bool(
                    events.max() <= self.e2_config.rearm_max_event_probability
                )
                if evidence_phase is E2Phase.INACTIVE and low_event_evidence:
                    phase = E2Phase.INACTIVE
                else:
                    rearm_suppressions += 1
                    delta_rearm += 1
            last_timestamp = timestamp
            last_position = position
            last_new_index = index
        if len(recent) > self.config.event_history_capacity:
            removed = len(recent) - self.config.event_history_capacity
            recent = recent[removed:]
            evictions += removed
        if last_new_index is None:
            assert prior_record is not None
            replacement = replace(
                prior_record,
                payload=replace(
                    prior,
                    duplicate_suppression_count=duplicates,
                ),
            )
            return cast(
                StateBankRuntimeState,
                self.update_record(
                    state,
                    replacement,
                    action="e2_overlap_ignored",
                    details=(
                        ("event_kind", event_kind.value),
                        ("duplicate_positions", delta_duplicates),
                    ),
                    audit_timestamp=max(last_timestamp, 0.0),
                ),
            )
        payload = E2Payload(
            event_kind=event_kind,
            completed_count=completed_count,
            phase=phase,
            completed_intervals=tuple(intervals),
            recent_event_times=tuple(recent),
            current_start=current_start,
            last_timestamp=last_timestamp,
            last_position_id=last_position,
            duplicate_suppression_count=duplicates,
            conflict_count=conflicts,
            rearm_suppression_count=rearm_suppressions,
            history_eviction_count=evictions,
        )
        confidence = float(
            observation.event_probabilities[row, last_new_index].float().max().item()
        )
        return self._upsert_aggregate(
            state,
            head_type=HeadType.E2,
            semantic_embedding=semantics[last_new_index],
            timestamp=last_timestamp,
            confidence=confidence,
            payload=payload,
            action="e2_fsm_update",
            details=(
                ("event_kind", event_kind.value),
                ("position_id", last_position),
                ("completed_added", delta_completed),
                ("duplicate_positions", delta_duplicates),
                ("conflicts", delta_conflicts),
                ("rearm_suppressions", delta_rearm),
                ("history_evictions", evictions - prior.history_eviction_count),
                ("phase", phase.value),
            ),
        )

    def _upsert_aggregate(
        self,
        state: StateBankRuntimeState,
        *,
        head_type: HeadType,
        semantic_embedding: Tensor,
        timestamp: float,
        confidence: float,
        payload: O1Payload | E1Payload | E2Payload,
        action: str,
        details: tuple[tuple[str, AuditValue], ...],
    ) -> StateBankRuntimeState:
        prior = _find_aggregate_record(state, head_type)
        if prior is None:
            appended = self.append_record(
                state,
                head_type=head_type,
                semantic_embedding=semantic_embedding,
                timestamp=timestamp,
                time_range=None,
                valid=True,
                confidence=confidence,
                payload=payload,
            )
            record = _find_aggregate_record(appended, head_type)
            assert record is not None
            audit = StateBankAuditEntry(
                action,
                record.record_id,
                _canonical_audit_time(appended, timestamp),
                details,
            )
            return cast(
                StateBankRuntimeState,
                replace(
                    appended,
                    audit_log=appended.audit_log + (audit,),
                    version=appended.version + 1,
                ),
            )
        replacement = StateRecord(
            record_id=prior.record_id,
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            head_type=head_type,
            semantic_embedding=_hard_semantic(semantic_embedding, self.config.semantic_projector),
            timestamp=timestamp,
            time_range=None,
            valid=True,
            confidence=confidence,
            payload=_clone_payload(payload),
        )
        return cast(
            StateBankRuntimeState,
            self.update_record(state, replacement, action=action, details=details),
        )


def build_state_bank(config: ProjectConfig | None = None) -> StructuredStateBank:
    if config is None:
        raise ValueError("build_state_bank requires a validated ProjectConfig")
    return StructuredStateBank(config)


def clone_state_record(record: StateRecord) -> StateRecord:
    """Return a storage-isolated typed record for downstream snapshot consumers."""

    if not isinstance(record, StateRecord):
        raise TypeError("clone_state_record requires a StateRecord")
    return _clone_record(record)


def clone_retrieval_history_record(record: RetrievalHistoryRecord) -> RetrievalHistoryRecord:
    if not isinstance(record, RetrievalHistoryRecord):
        raise TypeError("clone_retrieval_history_record requires RetrievalHistoryRecord")
    return _clone_retrieval_record(record)


def _validate_last_metadata(timestamp: float, position_id: int, name: str) -> None:
    fresh = timestamp == -1.0 and position_id == -1
    committed = math.isfinite(timestamp) and timestamp >= 0.0 and position_id >= 0
    if type(position_id) is not int or not (fresh or committed):
        raise ValueError(f"{name} last timestamp/position metadata is invalid")


def _validate_strict_times(times: tuple[float, ...], name: str) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in times):
        raise ValueError(f"{name} times must be finite and non-negative")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError(f"{name} times must increase strictly")


def _validate_record_payload_time(record: StateRecord) -> None:
    payload = record.payload
    if isinstance(payload, (O1Payload, E1Payload, E2Payload)):
        if record.timestamp is None:
            raise ValueError("aggregate StateRecord payloads require a point timestamp")
        timestamp = record.timestamp
        if payload.last_timestamp != -1.0 and not _float_close(payload.last_timestamp, timestamp):
            raise ValueError("aggregate payload last_timestamp must match the record timestamp")
        if isinstance(payload, O1Payload):
            _require_times_not_after(
                tuple(slot.last_timestamp for slot in payload.slot_states),
                timestamp,
                "O1 slot",
            )
        elif isinstance(payload, E1Payload):
            _require_times_not_after(payload.recent_event_times, timestamp, "E1 event")
            if payload.candidate_start is not None:
                _require_times_not_after((payload.candidate_start,), timestamp, "E1 candidate")
        else:
            interval_times = tuple(
                value for interval in payload.completed_intervals for value in interval
            )
            _require_times_not_after(interval_times, timestamp, "E2 interval")
            _require_times_not_after(payload.recent_event_times, timestamp, "E2 event")
            if payload.current_start is not None:
                _require_times_not_after((payload.current_start,), timestamp, "E2 candidate")
        return
    if isinstance(payload, (CandidateIdentity, ConfirmedIdentity)):
        if record.timestamp is not None:
            if not _float_close(record.timestamp, payload.first_seen):
                raise ValueError("O2 point record timestamp must match payload first_seen")
            return
        assert record.time_range is not None
        start, end = record.time_range
        if not _float_close(start, payload.first_seen) or not _float_close(end, payload.last_seen):
            raise ValueError("O2 range record boundaries must match payload first_seen/last_seen")
        return
    raise TypeError("StateRecord carries an unsupported payload type")


def _require_times_not_after(times: tuple[float, ...], endpoint: float, name: str) -> None:
    if any(value > endpoint and not _float_close(value, endpoint) for value in times):
        raise ValueError(f"{name} time cannot be later than the record timestamp")


def _validate_payload_tensors_detached(payload: StatePayload) -> None:
    tensors = _payload_tensors(payload)
    if any(tensor.requires_grad or tensor.grad_fn is not None for tensor in tensors):
        raise ValueError("State Bank payload tensors must be detached")
    for tensor in tensors:
        if tensor.device.type != "meta" and not bool(torch.isfinite(tensor).all()):
            raise ValueError("State Bank payload tensors must be finite")


def _payload_tensors(payload: StatePayload) -> tuple[Tensor, ...]:
    if isinstance(payload, (CandidateIdentity, ConfirmedIdentity)):
        return (payload.identity_prototype,)
    return ()


def _record_kind(record: StateRecord) -> StateRecordKind:
    payload = record.payload
    if isinstance(payload, O1Payload):
        return StateRecordKind.O1_AGGREGATE
    if isinstance(payload, CandidateIdentity):
        return StateRecordKind.O2_CANDIDATE
    if isinstance(payload, ConfirmedIdentity):
        return StateRecordKind.O2_CONFIRMED
    if isinstance(payload, E1Payload):
        return StateRecordKind.E1_AGGREGATE
    if isinstance(payload, E2Payload):
        return StateRecordKind.E2_AGGREGATE
    raise TypeError("StateRecord carries an unsupported payload type")


def _history_record_kind(record: RetrievalHistoryRecord) -> StateRecordKind:
    return {
        HeadType.O1: StateRecordKind.O1_AGGREGATE,
        HeadType.O2: StateRecordKind.O2_CONFIRMED,
        HeadType.E1: StateRecordKind.E1_AGGREGATE,
        HeadType.E2: StateRecordKind.E2_AGGREGATE,
    }[record.head_type]


def _record_tensors(record: StateRecord) -> tuple[Tensor, ...]:
    return (record.semantic_embedding, *_payload_tensors(record.payload))


def _validate_state_bank_view_records(view: StateBankView) -> None:
    groups: list[tuple[Tensor, ...]] = []
    meta = view.embeddings.device.type == "meta"
    for row, records in enumerate(view.cloned_records):
        present_ids: list[str] = []
        for column, record in enumerate(records):
            record_id = view.record_ids[row][column]
            head_type = view.head_types[row][column]
            record_kind = view.record_kinds[row][column]
            if record is None:
                if record_id is not None or head_type is not None or record_kind is not None:
                    raise ValueError("StateBankView padding record metadata must be None")
                if not meta and bool(view.present_mask[row, column]):
                    raise ValueError("StateBankView present entries require cloned records")
                continue
            if record_id is None or head_type is None or record_kind is None:
                raise ValueError("StateBankView cloned records require complete metadata")
            if (
                record.video_id != view.video_ids[row]
                or record.trajectory_id != view.trajectory_ids[row]
            ):
                raise ValueError("StateBankView cloned record owner metadata is inconsistent")
            if record.record_id != record_id:
                raise ValueError("StateBankView cloned record ID metadata is inconsistent")
            if record.head_type is not head_type:
                raise ValueError("StateBankView cloned record head metadata is inconsistent")
            if _record_kind(record) is not record_kind:
                raise ValueError("StateBankView cloned record kind metadata is inconsistent")
            if (
                record.semantic_embedding.dtype != view.embeddings.dtype
                or record.semantic_embedding.device != view.embeddings.device
            ):
                raise ValueError(
                    "StateBankView cloned record semantics must match view dtype/device"
                )
            present_ids.append(record.record_id)
            groups.append(_record_tensors(record))
            if meta:
                continue
            if not bool(view.present_mask[row, column]):
                raise ValueError("StateBankView cloned records must be marked present")
            if bool(view.record_valid_mask[row, column]) is not record.valid:
                raise ValueError("StateBankView cloned record validity metadata is inconsistent")
            expected_eligible = record.valid and record_kind is not StateRecordKind.O2_CANDIDATE
            if bool(view.retrieval_eligible_mask[row, column]) is not expected_eligible:
                raise ValueError("StateBankView cloned record retrieval metadata is inconsistent")
            if not torch.equal(view.embeddings[row, column], record.semantic_embedding):
                raise ValueError("StateBankView cloned record semantic metadata is inconsistent")
            stored_timestamp = float(view.timestamps[row, column].item())
            stored_range = view.time_ranges[row, column]
            if record.timestamp is not None:
                if not _float_close(stored_timestamp, record.timestamp) or bool(
                    torch.any(stored_range != -1.0)
                ):
                    raise ValueError(
                        "StateBankView cloned record timestamp metadata is inconsistent"
                    )
            else:
                assert record.time_range is not None
                expected_range = torch.tensor(
                    record.time_range,
                    dtype=torch.float64,
                    device=view.embeddings.device,
                )
                if stored_timestamp != -1.0 or not torch.equal(stored_range, expected_range):
                    raise ValueError(
                        "StateBankView cloned record time-range metadata is inconsistent"
                    )
        if len(set(present_ids)) != len(present_ids):
            raise ValueError("StateBankView cloned record IDs must be unique within each owner")
    _assert_tensor_groups_isolated(tuple(groups), "StateBankView cloned records")
    if view.embeddings.numel() > 0:
        embedding_storage = tensor_storage_key(view.embeddings)
        if any(
            tensor.numel() > 0 and tensor_storage_key(tensor) == embedding_storage
            for group in groups
            for tensor in group
        ):
            raise ValueError("StateBankView tensors and cloned records must not share storage")


def _assert_tensor_groups_isolated(groups: Sequence[tuple[Tensor, ...]], name: str) -> None:
    seen: set[tuple[str, int | None, int]] = set()
    for group in groups:
        group_keys: set[tuple[str, int | None, int]] = set()
        for tensor in group:
            if tensor.numel() == 0:
                continue
            group_keys.add(tensor_storage_key(tensor))
        if seen.intersection(group_keys):
            raise ValueError(f"{name} must not share mutable tensor storage")
        seen.update(group_keys)


def _normalize_head_types(
    head_types: HeadType | Sequence[HeadType], count: int
) -> tuple[HeadType, ...]:
    if isinstance(head_types, HeadType):
        return (head_types,) * count
    normalized = tuple(head_types)
    if len(normalized) != count or any(not isinstance(value, HeadType) for value in normalized):
        raise ValueError("head_types must provide one valid HeadType per source state")
    return normalized


def _normalize_view_head_filter(
    head_type: HeadType | Sequence[HeadType | None] | None,
    count: int,
) -> tuple[HeadType | None, ...] | None:
    if head_type is None:
        return None
    if isinstance(head_type, HeadType):
        return (head_type,) * count
    if isinstance(head_type, (str, bytes)) or not isinstance(head_type, Sequence):
        raise TypeError("State Bank view head_type must be a HeadType, sequence, or None")
    normalized = tuple(head_type)
    if len(normalized) != count:
        raise ValueError("row-wise State Bank head filters must match the batch size")
    if any(value is not None and not isinstance(value, HeadType) for value in normalized):
        raise TypeError("row-wise State Bank head filters must contain HeadType or None")
    return cast(tuple[HeadType | None, ...], normalized)


def _normalize_semantic(raw: Tensor, eps: float) -> Tensor:
    raw_fp32 = raw.float()
    norms = torch.linalg.vector_norm(raw_fp32, dim=-1, keepdim=True)
    fallback = torch.zeros_like(raw_fp32)
    fallback[..., 0] = 1.0
    safe = torch.where(norms > eps, raw_fp32, fallback)
    return F.normalize(safe, dim=-1, eps=eps)


def _hard_semantic(embedding: Tensor, config: SemanticProjectorConfig) -> Tensor:
    if embedding.shape != (config.output_dim,) or not torch.is_floating_point(embedding):
        raise ValueError("hard semantic embedding must be floating [512]")
    if embedding.device.type != "meta" and not bool(torch.isfinite(embedding).all()):
        raise ValueError("hard semantic embedding must be finite")
    normalized = _normalize_semantic(embedding.detach().unsqueeze(0), config.normalization_eps)[0]
    return normalized.clone()


def _clone_payload(payload: StatePayload) -> StatePayload:
    if isinstance(payload, CandidateIdentity):
        return replace(
            payload,
            identity_prototype=payload.identity_prototype.detach().clone(),
        )
    if isinstance(payload, ConfirmedIdentity):
        return replace(
            payload,
            identity_prototype=payload.identity_prototype.detach().clone(),
        )
    return payload


def _clone_record(record: StateRecord) -> StateRecord:
    return StateRecord(
        record_id=record.record_id,
        video_id=record.video_id,
        trajectory_id=record.trajectory_id,
        head_type=record.head_type,
        semantic_embedding=record.semantic_embedding.detach().clone(),
        timestamp=record.timestamp,
        time_range=record.time_range,
        valid=record.valid,
        confidence=record.confidence,
        payload=_clone_payload(record.payload),
    )


def _clone_runtime_state(state: StateBankRuntimeState) -> StateBankRuntimeState:
    return StateBankRuntimeState(
        video_id=state.video_id,
        trajectory_id=state.trajectory_id,
        records=tuple(_clone_record(record) for record in state.records),
        audit_log=tuple(state.audit_log),
        retrieval_history=tuple(
            _clone_retrieval_record(record) for record in state.retrieval_history
        ),
        issued_record_ids=tuple(state.issued_record_ids),
        next_record_sequence=state.next_record_sequence,
        next_retrieval_sequence=state.next_retrieval_sequence,
        released=state.released,
        version=state.version,
    )


def _clone_retrieval_record(record: RetrievalHistoryRecord) -> RetrievalHistoryRecord:
    return RetrievalHistoryRecord(
        record_id=record.record_id,
        video_id=record.video_id,
        trajectory_id=record.trajectory_id,
        head_type=record.head_type,
        operator=record.operator,
        semantic_source=record.semantic_source.detach().clone(),
        timestamp=record.timestamp,
        time_range=record.time_range,
        valid=record.valid,
        retrieval_eligible=record.retrieval_eligible,
        lifecycle_id=record.lifecycle_id,
    )


def _require_live_state(state: StateBankRuntimeState) -> None:
    if not isinstance(state, StateBankRuntimeState):
        raise TypeError("State Bank operation requires StateBankRuntimeState")
    if state.released:
        raise ValueError("released State Bank runtime cannot be used")


def _find_record_index(state: StateBankRuntimeState, record_id: str) -> int:
    matches = [index for index, record in enumerate(state.records) if record.record_id == record_id]
    if len(matches) != 1:
        raise ValueError("State Bank record ID does not identify exactly one record")
    return matches[0]


def _next_available_record_id(state: StateBankRuntimeState) -> tuple[str, int]:
    issued = set(state.issued_record_ids)
    next_sequence = state.next_record_sequence
    while True:
        record_id = f"record-{next_sequence:08d}"
        next_sequence += 1
        if record_id not in issued:
            return record_id, next_sequence


def _with_semantic_record_link(
    payload: CandidateIdentity | ConfirmedIdentity,
    record_id: str,
) -> CandidateIdentity | ConfirmedIdentity:
    if not record_id:
        raise ValueError("O2 semantic record link must be non-empty")
    existing = getattr(payload, "semantic_record_id", None)
    if existing not in (None, record_id):
        raise ValueError("O2 payload semantic record link cannot be reassigned")
    return replace(payload, semantic_record_id=record_id)


def _require_semantic_record_link(payload: CandidateIdentity | ConfirmedIdentity) -> str:
    record_id = getattr(payload, "semantic_record_id", None)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("O2 update payload requires a semantic record link")
    return record_id


def _require_o2_payload(
    state: StateBankRuntimeState,
    record_id: str,
    payload_type: type[CandidateIdentity] | type[ConfirmedIdentity],
) -> StateRecord:
    _require_live_state(state)
    record = state.records[_find_record_index(state, record_id)]
    if record.head_type is not HeadType.O2 or type(record.payload) is not payload_type:
        raise ValueError(f"O2 record must carry {payload_type.__name__}")
    return record


def _find_aggregate_record(state: StateBankRuntimeState, head_type: HeadType) -> StateRecord | None:
    matches = [record for record in state.records if record.head_type is head_type]
    if head_type is HeadType.O2:
        raise ValueError("O2 identity records are not an aggregate P9 hard state")
    if len(matches) > 1:
        raise ValueError(f"{head_type.value} partition must contain one aggregate record")
    return matches[0] if matches else None


def _record_audit_time(record: StateRecord) -> float:
    if record.timestamp is not None:
        return record.timestamp
    assert record.time_range is not None
    return record.time_range[1]


def _append_runtime_audit(
    state: StateBankRuntimeState,
    *,
    action: str,
    record_id: str | None,
    timestamp: float,
    details: tuple[tuple[str, AuditValue], ...],
) -> StateBankRuntimeState:
    audit = StateBankAuditEntry(action, record_id, _canonical_audit_time(state, timestamp), details)
    return StateBankRuntimeState(
        video_id=state.video_id,
        trajectory_id=state.trajectory_id,
        records=tuple(_clone_record(record) for record in state.records),
        audit_log=state.audit_log + (audit,),
        retrieval_history=tuple(
            _clone_retrieval_record(record) for record in state.retrieval_history
        ),
        issued_record_ids=state.issued_record_ids,
        next_record_sequence=state.next_record_sequence,
        next_retrieval_sequence=state.next_retrieval_sequence,
        released=False,
        version=state.version + 1,
    )


def _validate_row(row: int, batch_size: int, name: str) -> None:
    if type(row) is not int or not 0 <= row < batch_size:
        raise ValueError(f"{name} row index is out of range")


def _select_semantics(semantics: Tensor, shape: torch.Size, row: int) -> Tensor:
    if semantics.ndim == 3:
        if semantics.shape[:2] != shape or semantics.shape[2] != 512:
            raise ValueError("semantic embeddings must align as [B, T, 512]")
        selected = semantics[row]
    elif semantics.ndim == 2:
        if semantics.shape != (shape[1], 512):
            raise ValueError("singleton semantic embeddings must be [T, 512]")
        selected = semantics
    else:
        raise ValueError("semantic embeddings must be [B, T, 512] or [T, 512]")
    if not torch.is_floating_point(selected):
        raise ValueError("semantic embeddings must use a floating dtype")
    if selected.device.type != "meta" and not bool(torch.isfinite(selected).all()):
        raise ValueError("semantic embeddings must be finite")
    return selected


def _float_close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= 4.0 * float(torch.finfo(torch.float32).eps) * scale


def _same_o1_evidence(left: O1SlotState, right: O1SlotState) -> bool:
    return (
        left.slot_id == right.slot_id
        and left.is_object == right.is_object
        and left.is_target == right.is_target
        and left.visible == right.visible
        and left.enter == right.enter
        and left.exit == right.exit
        and _float_close(left.confidence, right.confidence)
    )


def _canonical_audit_time(state: StateBankRuntimeState, timestamp: float) -> float:
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("State Bank audit timestamp must be finite and non-negative")
    return max(timestamp, state.audit_log[-1].timestamp if state.audit_log else 0.0)


def _validate_semantic_projector_config(config: SemanticProjectorConfig) -> None:
    expected: dict[str, object] = {
        "input_dim": 768,
        "hidden_dim": 1024,
        "output_dim": 512,
        "head_type_count": 4,
        "head_types": ("o1", "o2", "e1", "e2"),
        "layer_norm_eps": 1.0e-5,
        "activation": "silu",
        "dropout": 0.0,
        "linear_bias": True,
        "normalization_dtype": "float32",
        "normalization_eps": 1.0e-8,
        "zero_norm_fallback": "first_unit_basis",
        "parameter_count": 1_316_864,
        "included_in_model_state_dict": True,
        "included_in_outer_optimizer": True,
        "included_in_inner_optimizer": False,
        "online_frozen": True,
        "online_forward_no_grad": False,
        "detach_inputs": False,
    }
    _validate_config_fields(config, expected, "Semantic Projector")


def _validate_state_bank_config(config: StateBankConfig) -> None:
    expected: dict[str, object] = {
        "semantic_dim": 512,
        "identity_dim": 256,
        "event_history_capacity": 512,
        "retrieval_history_capacity_per_head": 512,
        "retrieval_history_source_dim": 768,
        "isolation_keys": ("video_id", "trajectory_id", "head_type"),
        "hard_updates_no_grad": True,
        "detach_before_write": True,
        "runtime_in_model_state_dict": False,
        "runtime_registered_parameters": False,
        "runtime_registered_buffers": False,
        "runtime_in_outer_optimizer": False,
        "runtime_in_inner_optimizer": False,
        "snapshot_separate_from_model_checkpoint": True,
        "aggregate_update_mode": "functional_replace",
        "record_time_metadata_policy": "exactly_one",
        "record_id_policy": "trajectory_monotonic_never_reuse",
        "aggregate_record_heads": ("o1", "e1", "e2"),
        "committed_position_policy": "idempotent_ignore_and_audit",
        "o2_p9_policy": "generic_crud_only_p10_owns_lifecycle",
        "dynamic_view_padding": "batch_max",
        "n_state_definition": "owner_head_present_records_before_filters",
    }
    _validate_config_fields(config, expected, "State Bank")


def _validate_config_fields(config: object, expected: dict[str, object], name: str) -> None:
    for field, required in expected.items():
        if getattr(config, field) != required:
            raise ValueError(f"P9 requires {name} {field}={required!r}")
