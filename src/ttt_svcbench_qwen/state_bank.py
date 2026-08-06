"""Implement the learned semantic view and detached typed State Bank runtime.

Inputs: semantic source states plus detached O1/O2/E1/E2 evidence and owner metadata.
Outputs: normalized semantic embeddings, functional typed records, hard FSM state, and audit.
Forbidden: identity matching, retrieval, Reader arithmetic, gradients in runtime, or in-place state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, SemanticProjectorConfig
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.observation_heads import E1SoftOutput, E2SoftOutput, O1SoftOutput

if TYPE_CHECKING:
    from ttt_svcbench_qwen.query_encoder import Operator


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
        self.next_sequence = 0
        self.version = 0
        self.released = False

    @property
    def count(self) -> int:
        return sum(self.sizes)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def append_many(self, batch: RetrievalHistoryAppendBatch) -> None:
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
        self.released = True
        self.version += 1


@dataclass(frozen=True, slots=True)
class StateBankAuditEntry:
    action: str
    record_id: str | None
    timestamp: float
    details: tuple[tuple[str, AuditValue], ...]

@dataclass(frozen=True, slots=True)
class StateBankRuntimeState:
    video_id: str
    trajectory_id: str
    records: tuple[StateRecord, ...]
    audit_log: tuple[StateBankAuditEntry, ...]
    issued_record_ids: tuple[str, ...] = ()
    next_record_sequence: int = 0
    released: bool = False
    version: int = 0

    def __post_init__(self) -> None:
        record_ids = tuple(record.record_id for record in self.records)
        if not self.issued_record_ids and record_ids:
            object.__setattr__(self, "issued_record_ids", record_ids)


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
    retrieval_eligible_mask: Tensor
    cloned_records: tuple[tuple[StateRecord | None, ...], ...]


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
    cloned_records: tuple[tuple[RetrievalHistoryRecord | None, ...], ...]
    sequence_ids: Tensor | None = None
    head_codes: Tensor | None = None
    operator_codes: Tensor | None = None

    def require_tensor_metadata(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return the materialized integer metadata gathered by the ring snapshot."""

        return (
            cast(Tensor, self.sequence_ids),
            cast(Tensor, self.head_codes),
            cast(Tensor, self.operator_codes),
        )


@torch.no_grad()  # type: ignore[untyped-decorator]
def tensorized_retrieval_view(
    histories: Sequence[TensorizedRetrievalHistory],
    *,
    guard_current_version: bool = True,
) -> RetrievalHistoryView:
    """Gather each four-head ring once and restore global sequence order."""

    normalized = tuple(histories)
    reference = normalized[0].sources

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
        if source_rows:
            sequence = torch.cat(sequence_rows)
            order = torch.argsort(sequence, stable=True)
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
        cloned_records=empty_metadata,
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

        flattened = source_states.reshape(-1, self.config.input_dim)
        indices = head_codes.reshape(-1)
        conditioned = flattened + self.head_type_embeddings(indices)
        hidden = F.silu(self.hidden_projection(self.input_norm(conditioned)))
        raw = self.output_projection(hidden)
        normalized = _normalize_semantic(raw, self.config.normalization_eps)
        return normalized.reshape(*source_states.shape[:-1], self.config.output_dim)

    def set_online_frozen(self, frozen: bool = True) -> SemanticProjector:
        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.eval()
        return self


class StructuredStateBank(nn.Module):  # type: ignore[misc]
    """Model-owned projector plus parameter-free functional hard-state operators."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
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
        return StateBankRuntimeState(video_id, trajectory_id, (), ())

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def release(self, state: StateBankRuntimeState) -> StateBankRuntimeState:
        return StateBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            records=(),
            audit_log=(),
            issued_record_ids=(),
            next_record_sequence=0,
            released=True,
            version=state.version + 1,
        )

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
            issued_record_ids=issued + (record.record_id,),
            next_record_sequence=next_sequence,
            released=False,
            version=state.version + 1,
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
        index = _find_record_index(state, record.record_id)
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
            issued_record_ids=state.issued_record_ids,
            next_record_sequence=state.next_record_sequence,
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
        previous = state.records[_find_record_index(state, record_id)]
        if not previous.valid:
            return _append_runtime_audit(
                state,
                action="invalidate_duplicate",
                record_id=record_id,
                timestamp=audit_timestamp,
                details=(("reason", reason),),
            )
        replacement = replace(previous, valid=False)
        return cast(
            StateBankRuntimeState,
            self.update_record(
                state,
                replacement,
                action="invalidate",
                details=(
                    ("reason", reason),
                    ("audit_timestamp", audit_timestamp),
                ),
                audit_timestamp=audit_timestamp,
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
    def view(
        self,
        states: Sequence[StateBankRuntimeState],
        head_type: HeadType | Sequence[HeadType | None] | None = None,
    ) -> StateBankView:
        normalized = tuple(states)
        row_head_types = _normalize_view_head_filter(head_type, len(normalized))
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
            reference = all_records[0].semantic_embedding
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
        cloned_records: list[tuple[StateRecord | None, ...]] = []
        for row, records in enumerate(rows):
            count = len(records)
            n_state[row] = count
            ids: list[str | None] = [None] * max_records
            heads: list[HeadType | None] = [None] * max_records
            record_copies: list[StateRecord | None] = [None] * max_records
            for column, record in enumerate(records):
                embeddings[row, column] = record.semantic_embedding
                present_mask[row, column] = True
                valid_mask[row, column] = record.valid
                retrieval_eligible_mask[row, column] = record.valid and not isinstance(
                    record.payload, CandidateIdentity
                )
                ids[column] = record.record_id
                heads[column] = record.head_type
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
            retrieval_eligible_mask=retrieval_eligible_mask,
            cloned_records=tuple(cloned_records),
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
        mask = observation.valid_mask[row]
        prior_record = _find_aggregate_record(state, HeadType.O1)
        prior_payload = (
            prior_record.payload
            if prior_record is not None
            else O1Payload(0, 0, (), baseline_initialized=False)
        )
        assert isinstance(prior_payload, O1Payload)
        probabilities = observation.probabilities[row]
        incoming_slots = self._decode_o1_row(
            probabilities,
            mask,
            timestamp=observation_timestamp,
            position_id=observation_position_id,
        )
        if observation_position_id <= prior_payload.last_position_id:
            return state
        prior_slots = {slot.slot_id: slot for slot in prior_payload.slot_states}
        slot_states: list[O1SlotState] = []
        active_slot_ids: list[int] = []
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
    ) -> dict[int, O1SlotState]:
        slots: dict[int, O1SlotState] = {}
        for slot_id in range(mask.shape[0]):
            if not bool(mask[slot_id].item()):
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
        return slots

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
        semantics = _select_semantics(semantic_embeddings, observation.logits.shape[:2], row)
        prior_record = _find_aggregate_record(state, HeadType.E1)
        prior = (
            prior_record.payload if prior_record is not None else E1Payload(event_kind, 0, (), 0.0)
        )
        assert isinstance(prior, E1Payload)
        event_count = prior.event_count
        recent = list(prior.recent_event_times)
        cooldown_until = prior.cooldown_until
        active = prior.active
        armed = prior.armed
        candidate_start = prior.candidate_start
        last_timestamp = prior.last_timestamp
        last_position = prior.last_position_id
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
                continue
            values = observation.probabilities[row, index].float()
            eventness, completion, transition = values.unbind()
            if active:
                if bool(completion >= self.e1_config.completion_threshold) and bool(
                    transition >= self.e1_config.transition_threshold
                ):
                    if timestamp >= cooldown_until and not (
                        recent and timestamp - recent[-1] < self.e1_config.min_gap_seconds
                    ):
                        event_count += 1
                        recent.append(timestamp)
                        cooldown_until = timestamp + self.e1_config.min_gap_seconds
                    active = False
                    armed = False
                    candidate_start = None
                elif bool(eventness <= self.e1_config.tau_off):
                    active = False
                    armed = True
                    candidate_start = None
            elif not armed:
                if bool(eventness <= self.e1_config.tau_off):
                    armed = True
            elif bool(eventness >= self.e1_config.tau_on):
                if timestamp >= cooldown_until:
                    active = True
                    armed = False
                    candidate_start = timestamp
            last_timestamp = timestamp
            last_position = position
            last_new_index = index
        if len(recent) > self.config.event_history_capacity:
            recent = recent[len(recent) - self.config.event_history_capacity :]
        if last_new_index is None:
            return state
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
            details=(),
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
        semantics = _select_semantics(semantic_embeddings, observation.event_logits.shape[:2], row)
        prior_record = _find_aggregate_record(state, HeadType.E2)
        prior = (
            prior_record.payload
            if prior_record is not None
            else E2Payload(event_kind, 0, E2Phase.INACTIVE, (), ())
        )
        assert isinstance(prior, E2Payload)
        completed_count = prior.completed_count
        intervals = list(prior.completed_intervals)
        recent = list(prior.recent_event_times)
        phase = prior.phase
        current_start = prior.current_start
        last_timestamp = prior.last_timestamp
        last_position = prior.last_position_id
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
                continue
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
            elif phase is E2Phase.ACTIVE:
                if (
                    bool(end >= self.e2_config.end_threshold)
                    and evidence_phase is E2Phase.END_CANDIDATE
                ):
                    phase = E2Phase.END_CANDIDATE
            elif phase is E2Phase.END_CANDIDATE:
                if (
                    bool(complete >= self.e2_config.complete_threshold)
                    and evidence_phase is E2Phase.COMPLETED
                ):
                    assert current_start is not None
                    intervals.append((current_start, timestamp))
                    recent.append(timestamp)
                    completed_count += 1
                    current_start = None
                    phase = E2Phase.COMPLETED
            else:
                low_event_evidence = bool(
                    events.max() <= self.e2_config.rearm_max_event_probability
                )
                if evidence_phase is E2Phase.INACTIVE and low_event_evidence:
                    phase = E2Phase.INACTIVE
            last_timestamp = timestamp
            last_position = position
            last_new_index = index
        if len(recent) > self.config.event_history_capacity:
            recent = recent[len(recent) - self.config.event_history_capacity :]
        if last_new_index is None:
            return state
        payload = E2Payload(
            event_kind=event_kind,
            completed_count=completed_count,
            phase=phase,
            completed_intervals=tuple(intervals),
            recent_event_times=tuple(recent),
            current_start=current_start,
            last_timestamp=last_timestamp,
            last_position_id=last_position,
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
            details=(),
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
            return self.append_record(
                state,
                head_type=head_type,
                semantic_embedding=semantic_embedding,
                timestamp=timestamp,
                time_range=None,
                valid=True,
                confidence=confidence,
                payload=payload,
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
    return StructuredStateBank(cast(ProjectConfig, config))


def clone_state_record(record: StateRecord) -> StateRecord:
    """Return a storage-isolated typed record for downstream snapshot consumers."""

    return _clone_record(record)


def clone_retrieval_history_record(record: RetrievalHistoryRecord) -> RetrievalHistoryRecord:
    return _clone_retrieval_record(record)



def _normalize_head_types(
    head_types: HeadType | Sequence[HeadType], count: int
) -> tuple[HeadType, ...]:
    if isinstance(head_types, HeadType):
        return (head_types,) * count
    return tuple(head_types)


def _normalize_view_head_filter(
    head_type: HeadType | Sequence[HeadType | None] | None,
    count: int,
) -> tuple[HeadType | None, ...] | None:
    if head_type is None:
        return None
    if isinstance(head_type, HeadType):
        return (head_type,) * count
    return cast(tuple[HeadType | None, ...], tuple(head_type))


def _normalize_semantic(raw: Tensor, eps: float) -> Tensor:
    raw_fp32 = raw.float()
    norms = torch.linalg.vector_norm(raw_fp32, dim=-1, keepdim=True)
    fallback = torch.zeros_like(raw_fp32)
    fallback[..., 0] = 1.0
    safe = torch.where(norms > eps, raw_fp32, fallback)
    return F.normalize(safe, dim=-1, eps=eps)


def _hard_semantic(embedding: Tensor, config: SemanticProjectorConfig) -> Tensor:
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
    )


def _find_record_index(state: StateBankRuntimeState, record_id: str) -> int:
    return next(
        index for index, record in enumerate(state.records) if record.record_id == record_id
    )


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
    return replace(payload, semantic_record_id=record_id)


def _require_semantic_record_link(payload: CandidateIdentity | ConfirmedIdentity) -> str:
    return cast(str, getattr(payload, "semantic_record_id", None))


def _require_o2_payload(
    state: StateBankRuntimeState,
    record_id: str,
    payload_type: type[CandidateIdentity] | type[ConfirmedIdentity],
) -> StateRecord:
    return state.records[_find_record_index(state, record_id)]


def _find_aggregate_record(state: StateBankRuntimeState, head_type: HeadType) -> StateRecord | None:
    return next((record for record in state.records if record.head_type is head_type), None)


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
        issued_record_ids=state.issued_record_ids,
        next_record_sequence=state.next_record_sequence,
        released=False,
        version=state.version + 1,
    )


def _select_semantics(semantics: Tensor, shape: torch.Size, row: int) -> Tensor:
    if semantics.ndim == 3:
        return semantics[row]
    return semantics


def _canonical_audit_time(state: StateBankRuntimeState, timestamp: float) -> float:
    return max(timestamp, state.audit_log[-1].timestamp if state.audit_log else 0.0)
