"""Implement the detached O2 Candidate/Confirmed identity lifecycle.

Inputs: O2 soft observations, semantic embeddings, causal owner metadata, and chunk index.
Outputs: functional CPU-FP32 identity state, linked O2 records, exact decisions, and audit.
Forbidden: q_target retrieval, labels in runtime, ANN, silent overwrite, or cache truth.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig

if TYPE_CHECKING:
    from ttt_svcbench_qwen.observation_heads import O2SoftOutput
    from ttt_svcbench_qwen.state_bank import (
        StateBankRuntimeState,
        StructuredStateBank,
    )


IDENTITY_DIM = 256
SEMANTIC_DIM = 512
_UNIT_ATOL = 5.0e-4
_UNIT_RTOL = 5.0e-4

type AuditValue = str | int | float | bool | None


class IdentityDecisionStatus(StrEnum):
    INVALID = "invalid"
    REPLAY_IGNORED = "replay_ignored"
    SIGNAL_CONFLICT = "signal_conflict"
    MATCH_CONFLICT = "match_conflict"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_UPDATED = "candidate_updated"
    PROMOTED = "promoted"
    CONFIRMED_UPDATED = "confirmed_updated"
    OVERFLOW_REJECTED = "overflow_rejected"


class ExactMatchStatus(StrEnum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    candidate_id: str
    identity_prototype: Tensor
    observation_count: int
    ttl_remaining: int
    confidence: float
    first_seen: float = 0.0
    last_seen: float = 0.0
    first_seen_position_id: int = 0
    last_seen_position_id: int = 0
    last_reliable_chunk_index: int = 0
    reliable_streak: int = 1
    semantic_record_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identity_tensor(self.identity_prototype, "candidate identity prototype")
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if (
            type(self.observation_count) is not int
            or self.observation_count < 1
            or type(self.ttl_remaining) is not int
            or self.ttl_remaining < 0
            or type(self.reliable_streak) is not int
            or self.reliable_streak < 0
        ):
            raise ValueError("candidate counts/TTL/streak are invalid")
        _validate_probability(self.confidence, "candidate confidence")
        _validate_seen_metadata(
            self.first_seen,
            self.last_seen,
            self.first_seen_position_id,
            self.last_seen_position_id,
            "candidate",
        )
        if type(self.last_reliable_chunk_index) is not int or self.last_reliable_chunk_index < 0:
            raise ValueError("candidate reliable chunk index is invalid")
        if self.semantic_record_id is not None and not self.semantic_record_id:
            raise ValueError("candidate semantic_record_id cannot be empty")


@dataclass(frozen=True, slots=True)
class ConfirmedIdentity:
    identity_id: str
    identity_prototype: Tensor
    first_seen: float
    last_seen: float
    observation_count: int
    semantic_record_id: str | None = None
    prototype_version: int = 0
    first_seen_position_id: int = 0
    last_seen_position_id: int = 0

    def __post_init__(self) -> None:
        _validate_identity_tensor(self.identity_prototype, "confirmed identity prototype")
        if (
            not self.identity_id
            or type(self.observation_count) is not int
            or self.observation_count < 1
        ):
            raise ValueError("confirmed identity metadata is invalid")
        _validate_seen_metadata(
            self.first_seen,
            self.last_seen,
            self.first_seen_position_id,
            self.last_seen_position_id,
            "confirmed",
        )
        if type(self.prototype_version) is not int or self.prototype_version < 0:
            raise ValueError("confirmed prototype_version must be non-negative")
        if self.semantic_record_id is not None and not self.semantic_record_id:
            raise ValueError("confirmed semantic_record_id cannot be empty")


@dataclass(frozen=True, slots=True)
class ConfirmedChunk:
    """One authoritative fixed-capacity CPU FP32 allocation."""

    prototypes: Tensor
    occupied: Tensor
    identity_ids: tuple[str | None, ...]
    first_seen: Tensor
    last_seen: Tensor
    observation_counts: Tensor
    first_seen_position_ids: Tensor
    last_seen_position_ids: Tensor
    semantic_record_ids: tuple[str | None, ...]
    prototype_versions: Tensor

    def __post_init__(self) -> None:
        capacity = len(self.identity_ids)
        if capacity <= 0 or len(self.semantic_record_ids) != capacity:
            raise ValueError("Confirmed chunk metadata must have a positive aligned capacity")
        expected_vectors = (capacity, IDENTITY_DIM)
        if (
            self.prototypes.shape != expected_vectors
            or self.prototypes.dtype != torch.float32
            or self.prototypes.device.type != "cpu"
            or self.prototypes.requires_grad
            or self.prototypes.grad_fn is not None
        ):
            raise ValueError("Confirmed prototypes must be detached CPU FP32 [capacity, 256]")
        expected = (capacity,)
        tensors = (
            self.occupied,
            self.first_seen,
            self.last_seen,
            self.observation_counts,
            self.first_seen_position_ids,
            self.last_seen_position_ids,
            self.prototype_versions,
        )
        if any(tensor.shape != expected or tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("Confirmed chunk fields must be aligned CPU vectors")
        if self.occupied.dtype != torch.bool:
            raise ValueError("Confirmed occupied mask must be bool")
        if self.first_seen.dtype != torch.float64 or self.last_seen.dtype != torch.float64:
            raise ValueError("Confirmed timestamps must be CPU float64")
        integer_tensors = (
            self.observation_counts,
            self.first_seen_position_ids,
            self.last_seen_position_ids,
            self.prototype_versions,
        )
        if any(tensor.dtype != torch.int64 for tensor in integer_tensors):
            raise ValueError("Confirmed integer metadata must be int64")
        occupied_indices = torch.nonzero(self.occupied, as_tuple=False).flatten().tolist()
        for index in range(capacity):
            occupied = index in occupied_indices
            identity_id = self.identity_ids[index]
            record_id = self.semantic_record_ids[index]
            if occupied != (identity_id is not None and record_id is not None):
                raise ValueError("Confirmed occupancy, identity ID, and record link must agree")
            if not occupied:
                continue
            if not identity_id or not record_id:
                raise ValueError("Confirmed live IDs cannot be empty")
            prototype = self.prototypes[index]
            _validate_unit_identity(prototype, "Confirmed authoritative prototype")
            first = float(self.first_seen[index].item())
            last = float(self.last_seen[index].item())
            first_position = int(self.first_seen_position_ids[index].item())
            last_position = int(self.last_seen_position_ids[index].item())
            _validate_seen_metadata(first, last, first_position, last_position, "confirmed slot")
            if int(self.observation_counts[index].item()) < 1:
                raise ValueError("Confirmed observation count must be positive")
            if int(self.prototype_versions[index].item()) < 0:
                raise ValueError("Confirmed prototype version must be non-negative")

    @property
    def capacity(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def size(self) -> int:
        return int(self.occupied.sum().item())


@dataclass(frozen=True, slots=True)
class HotCacheEntry:
    identity_id: str
    identity_prototype: Tensor
    last_accessed_position_id: int
    prototype_version: int = 0

    def __post_init__(self) -> None:
        _validate_identity_tensor(self.identity_prototype, "Hot Cache identity prototype")
        if (
            not self.identity_id
            or type(self.last_accessed_position_id) is not int
            or self.last_accessed_position_id < 0
            or type(self.prototype_version) is not int
            or self.prototype_version < 0
        ):
            raise ValueError("Hot Cache metadata is invalid")
        if self.identity_prototype.requires_grad or self.identity_prototype.grad_fn is not None:
            raise ValueError("Hot Cache prototypes must be detached")

@dataclass(frozen=True, slots=True)
class IdentityBankAuditEntry:
    action: str
    timestamp: float
    position_id: int
    details: tuple[tuple[str, AuditValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Identity Bank audit action must be non-empty")
        if (
            not math.isfinite(self.timestamp)
            or self.timestamp < 0.0
            or type(self.position_id) is not int
            or self.position_id < 0
        ):
            raise ValueError("Identity Bank audit time/position is invalid")
        keys = tuple(key for key, _ in self.details)
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Identity Bank audit detail keys must be unique and non-empty")
        if any(isinstance(value, Tensor) for _, value in self.details):
            raise TypeError("Identity Bank audit cannot retain tensors")


@dataclass(frozen=True, slots=True)
class IdentityBankRuntimeState:
    video_id: str = "unowned-video"
    trajectory_id: str = "unowned-trajectory"
    candidates: tuple[CandidateIdentity, ...] = ()
    confirmed_chunks: tuple[ConfirmedChunk, ...] = ()
    hot_cache: tuple[HotCacheEntry, ...] = ()
    candidate_capacity: int = 64
    hot_cache_capacity: int = 256
    next_candidate_sequence: int = 0
    next_identity_sequence: int = 0
    issued_candidate_ids: tuple[str, ...] = ()
    issued_identity_ids: tuple[str, ...] = ()
    candidate_overflow_count: int = 0
    candidate_expired_count: int = 0
    candidate_low_confidence_pruned_count: int = 0
    match_conflict_count: int = 0
    signal_conflict_count: int = 0
    last_chunk_index: int = -1
    last_committed_position_id: int = -1
    hot_cache_requested: bool = True
    hot_cache_enabled: bool = False
    hot_cache_device: str | None = None
    hot_cache_dtype: str = "bfloat16"
    hot_cache_disabled_reason: str | None = "not_initialized"
    audit_log: tuple[IdentityBankAuditEntry, ...] = ()
    released: bool = False
    version: int = 0

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("Identity Bank owner identifiers must be non-empty")
        bool_fields = (self.hot_cache_requested, self.hot_cache_enabled, self.released)
        if any(type(value) is not bool for value in bool_fields):
            raise TypeError("Identity Bank runtime flags must be bool")
        counters = (
            self.candidate_capacity,
            self.hot_cache_capacity,
            self.next_candidate_sequence,
            self.next_identity_sequence,
            self.candidate_overflow_count,
            self.candidate_expired_count,
            self.candidate_low_confidence_pruned_count,
            self.match_conflict_count,
            self.signal_conflict_count,
            self.version,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("Identity Bank capacities/sequences/counters must be non-negative")
        if type(self.last_chunk_index) is not int or self.last_chunk_index < -1:
            raise ValueError("Identity Bank last_chunk_index is invalid")
        if type(self.last_committed_position_id) is not int or self.last_committed_position_id < -1:
            raise ValueError("Identity Bank last_committed_position_id is invalid")
        if self.released:
            if (
                self.candidates
                or self.confirmed_chunks
                or self.hot_cache
                or self.candidate_capacity
                or self.hot_cache_capacity
                or self.audit_log
            ):
                raise ValueError("released Identity Bank cannot retain trajectory storage")
            return
        if self.candidate_capacity < 64 or self.candidate_capacity > 512:
            raise ValueError("Candidate logical capacity must stay within [64, 512]")
        if self.candidate_capacity % 64 or len(self.candidates) > self.candidate_capacity:
            raise ValueError("Candidate capacity must grow in aligned chunks")
        if self.hot_cache_capacity != 256 or len(self.hot_cache) > self.hot_cache_capacity:
            raise ValueError("Hot Cache capacity must stay exactly 256")
        if not self.confirmed_chunks:
            raise ValueError("live Identity Bank must retain its initial Confirmed allocation")
        if any(chunk.capacity != 256 for chunk in self.confirmed_chunks):
            raise ValueError("Confirmed store must grow in 256-slot chunks")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        identity_ids = tuple(identity.identity_id for identity in self.confirmed)
        if len(set(candidate_ids)) != len(candidate_ids) or len(set(identity_ids)) != len(
            identity_ids
        ):
            raise ValueError("Identity IDs must be unique within their stores")
        if any(candidate.semantic_record_id is None for candidate in self.candidates):
            raise ValueError("live Candidate entries require semantic record links")
        if any(
            candidate.candidate_id not in self.issued_candidate_ids for candidate in self.candidates
        ):
            raise ValueError("Candidate ID tombstones are inconsistent")
        if any(identity_id not in self.issued_identity_ids for identity_id in identity_ids):
            raise ValueError("Confirmed identity ID tombstones are inconsistent")
        if len(set(self.issued_candidate_ids)) != len(self.issued_candidate_ids) or len(
            set(self.issued_identity_ids)
        ) != len(self.issued_identity_ids):
            raise ValueError("issued identity IDs cannot be reused")
        for candidate in self.candidates:
            _validate_authoritative_identity(candidate.identity_prototype, "Candidate prototype")
        if self.hot_cache_enabled:
            if self.hot_cache_device is None or self.hot_cache_disabled_reason is not None:
                raise ValueError("enabled Hot Cache requires a device and no disabled reason")
            for entry in self.hot_cache:
                if str(entry.identity_prototype.device) != self.hot_cache_device:
                    raise ValueError("Hot Cache entries must use the configured device")
                if _dtype_name(entry.identity_prototype.dtype) != self.hot_cache_dtype:
                    raise ValueError("Hot Cache entries must use the configured dtype")
        elif self.hot_cache or self.hot_cache_device is not None:
            raise ValueError("disabled Hot Cache cannot retain device storage")
        cached_ids = tuple(entry.identity_id for entry in self.hot_cache)
        if len(set(cached_ids)) != len(cached_ids) or any(
            identity_id not in identity_ids for identity_id in cached_ids
        ):
            raise ValueError("Hot Cache must be a unique subset of Confirmed IDs")
        if self.audit_log and any(
            right.position_id < left.position_id
            for left, right in zip(self.audit_log, self.audit_log[1:], strict=False)
        ):
            raise ValueError("Identity Bank audit positions must be monotonic")
        _assert_runtime_storage_isolated(self)

    @property
    def confirmed_capacity(self) -> int:
        return sum(chunk.capacity for chunk in self.confirmed_chunks)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def confirmed_count(self) -> int:
        return self.unique_count

    @property
    def unique_count(self) -> int:
        return sum(chunk.size for chunk in self.confirmed_chunks)

    @property
    def candidate_overflow(self) -> int:
        return self.candidate_overflow_count

    @property
    def audit(self) -> tuple[IdentityBankAuditEntry, ...]:
        return self.audit_log

    @property
    def confirmed(self) -> tuple[ConfirmedIdentity, ...]:
        records: list[ConfirmedIdentity] = []
        for chunk in self.confirmed_chunks:
            for index in torch.nonzero(chunk.occupied, as_tuple=False).flatten().tolist():
                identity_id = chunk.identity_ids[index]
                record_id = chunk.semantic_record_ids[index]
                assert identity_id is not None and record_id is not None
                records.append(
                    ConfirmedIdentity(
                        identity_id=identity_id,
                        identity_prototype=chunk.prototypes[index].detach().clone(),
                        first_seen=float(chunk.first_seen[index].item()),
                        last_seen=float(chunk.last_seen[index].item()),
                        observation_count=int(chunk.observation_counts[index].item()),
                        semantic_record_id=record_id,
                        prototype_version=int(chunk.prototype_versions[index].item()),
                        first_seen_position_id=int(chunk.first_seen_position_ids[index].item()),
                        last_seen_position_id=int(chunk.last_seen_position_ids[index].item()),
                    )
                )
        return tuple(records)


@dataclass(frozen=True, slots=True)
class IdentityObservationDecision:
    slot_index: int
    position_id: int
    timestamp: float
    status: IdentityDecisionStatus
    candidate_id: str | None = None
    identity_id: str | None = None
    similarity: float | None = None
    novelty: float | None = None
    match_confidence: float | None = None
    scanned_confirmed_count: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("identity decision slot_index must be non-negative")
        if type(self.position_id) is not int or self.position_id < -1:
            raise ValueError("identity decision position_id is invalid")
        if not math.isfinite(self.timestamp) or self.timestamp < -1.0:
            raise ValueError("identity decision timestamp is invalid")
        if not isinstance(self.status, IdentityDecisionStatus):
            raise TypeError("identity decision status is invalid")
        for value, name in (
            (self.novelty, "novelty"),
            (self.match_confidence, "match confidence"),
        ):
            if value is not None:
                _validate_probability(value, name)
        if self.similarity is not None and (
            not math.isfinite(self.similarity) or not -1.0001 <= self.similarity <= 1.0001
        ):
            raise ValueError("identity decision similarity is invalid")
        if type(self.scanned_confirmed_count) is not int or self.scanned_confirmed_count < 0:
            raise ValueError("identity decision scanned count must be non-negative")


@dataclass(frozen=True, slots=True)
class IdentityUpdateResult:
    identity_state: IdentityBankRuntimeState
    state_bank_state: StateBankRuntimeState
    decisions: tuple[IdentityObservationDecision, ...]

    @property
    def assignments(self) -> tuple[IdentityObservationDecision, ...]:
        """Compatibility alias for downstream orchestration terminology."""

        return self.decisions


@dataclass(frozen=True, slots=True)
class ExactMatchDecision:
    query_index: int
    status: ExactMatchStatus
    identity_id: str | None
    score: float | None
    ambiguous_identity_ids: tuple[str, ...]
    scanned_confirmed_count: int
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class ExactMatchResult:
    state: IdentityBankRuntimeState
    matches: tuple[ExactMatchDecision, ...]
    search_mode: str = "exact"
    ann_enabled: bool = False

    def __post_init__(self) -> None:
        if self.search_mode != "exact" or self.ann_enabled:
            raise ValueError("P10 only permits full exact identity search")


@dataclass(frozen=True, slots=True)
class IdentityRuntimeMetrics:
    candidate_count: int
    confirmed_count: int
    candidate_overflow_count: int
    candidate_expired_count: int
    candidate_low_confidence_pruned_count: int
    match_conflict_count: int
    signal_conflict_count: int


@dataclass(frozen=True, slots=True)
class IdentityQualityMetrics:
    duplicate_excess_count: int
    duplicate_denominator: int
    duplicate_rate: float
    missed_new_count: int
    ground_truth_identity_denominator: int
    missed_new_identity_rate: float


@dataclass(frozen=True, slots=True)
class _Match:
    status: ExactMatchStatus
    entry_id: str | None
    score: float | None
    ambiguous_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Observation:
    slot_index: int
    timestamp: float
    position_id: int
    identity: Tensor
    semantic: Tensor
    novelty: float
    match_confidence: float
    confidence: float


class IdentityBank:
    """Parameter-free functional O2 identity operator."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.o2_config = config.observation_heads.o2
        self.candidate_config = config.state_bank.candidate_store
        self.confirmed_config = config.state_bank.confirmed_store
        self._validate_config()

    def reset(
        self,
        video_id: str,
        trajectory_id: str,
        *,
        hot_device: str | torch.device | None = None,
        hot_cache_enabled: bool | None = None,
    ) -> IdentityBankRuntimeState:
        if not video_id or not trajectory_id:
            raise ValueError("Identity Bank reset requires non-empty owner identifiers")
        requested = (
            self.confirmed_config.hot_cache_enabled
            if hot_cache_enabled is None
            else hot_cache_enabled
        )
        if type(requested) is not bool:
            raise TypeError("hot_cache_enabled must be bool")
        enabled, device, reason = self._resolve_hot_cache(requested, hot_device)
        return IdentityBankRuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
            confirmed_chunks=(_empty_confirmed_chunk(self.confirmed_config.initial_capacity),),
            candidate_capacity=self.candidate_config.initial_capacity,
            hot_cache_capacity=self.confirmed_config.gpu_hot_capacity,
            hot_cache_requested=requested,
            hot_cache_enabled=enabled,
            hot_cache_device=device,
            hot_cache_dtype=self.confirmed_config.hot_cache_dtype,
            hot_cache_disabled_reason=reason,
        )

    def snapshot(self, state: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
        _require_live_state(state)
        return _clone_runtime_state(state)

    def restore(self, snapshot: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
        _require_live_state(snapshot)
        return _clone_runtime_state(snapshot)

    def clear(self, state: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
        _require_live_state(state)
        return IdentityBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            confirmed_chunks=(_empty_confirmed_chunk(self.confirmed_config.initial_capacity),),
            candidate_capacity=self.candidate_config.initial_capacity,
            hot_cache_capacity=self.confirmed_config.gpu_hot_capacity,
            next_candidate_sequence=state.next_candidate_sequence,
            next_identity_sequence=state.next_identity_sequence,
            issued_candidate_ids=state.issued_candidate_ids,
            issued_identity_ids=state.issued_identity_ids,
            hot_cache_requested=state.hot_cache_requested,
            hot_cache_enabled=state.hot_cache_enabled,
            hot_cache_device=state.hot_cache_device,
            hot_cache_dtype=state.hot_cache_dtype,
            hot_cache_disabled_reason=state.hot_cache_disabled_reason,
            version=state.version + 1,
        )

    def release(self, state: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
        _require_live_state(state)
        return IdentityBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            candidate_capacity=0,
            hot_cache_capacity=0,
            hot_cache_requested=state.hot_cache_requested,
            hot_cache_enabled=False,
            hot_cache_device=None,
            hot_cache_dtype=state.hot_cache_dtype,
            hot_cache_disabled_reason="released",
            released=True,
            version=state.version + 1,
        )

    def confirmed_by_id(
        self, state: IdentityBankRuntimeState, identity_id: str
    ) -> ConfirmedIdentity:
        _require_live_state(state)
        if not identity_id:
            raise ValueError("identity_id must be non-empty")
        matches = tuple(
            identity for identity in state.confirmed if identity.identity_id == identity_id
        )
        if len(matches) != 1:
            raise KeyError(f"Confirmed identity not found: {identity_id}")
        return _clone_confirmed(matches[0])

    def metrics(self, state: IdentityBankRuntimeState) -> IdentityRuntimeMetrics:
        _require_live_state(state)
        return IdentityRuntimeMetrics(
            candidate_count=len(state.candidates),
            confirmed_count=state.unique_count,
            candidate_overflow_count=state.candidate_overflow_count,
            candidate_expired_count=state.candidate_expired_count,
            candidate_low_confidence_pruned_count=state.candidate_low_confidence_pruned_count,
            match_conflict_count=state.match_conflict_count,
            signal_conflict_count=state.signal_conflict_count,
        )

    def exact_match(
        self,
        state: IdentityBankRuntimeState,
        queries: Tensor,
        *,
        access_position_id: int,
        use_hot_cache: bool = True,
    ) -> ExactMatchResult:
        """Scan the complete authoritative CPU store for every query.

        The cache is checked only for audit and warmed after the CPU decision. A cache hit never
        permits early acceptance and therefore cannot hide a better cold CPU match.
        """

        _require_live_state(state)
        if type(access_position_id) is not int or access_position_id < 0:
            raise ValueError("access_position_id must be non-negative")
        normalized = queries.unsqueeze(0) if queries.ndim == 1 else queries
        if normalized.ndim != 2 or normalized.shape[1] != IDENTITY_DIM:
            raise ValueError("identity queries must be [N, 256] or [256]")
        if not torch.is_floating_point(normalized):
            raise ValueError("identity queries must be floating")
        next_state = state
        matches: list[ExactMatchDecision] = []
        cached_ids = {entry.identity_id for entry in state.hot_cache} if use_hot_cache else set()
        for query_index, query in enumerate(normalized):
            hard_query = _hard_identity(query)
            match = self._match_confirmed(state, hard_query)
            cache_hit = match.entry_id in cached_ids if match.entry_id is not None else False
            if match.status is ExactMatchStatus.MATCHED and match.entry_id is not None:
                next_state = self._touch_hot_cache(
                    next_state,
                    match.entry_id,
                    access_position_id,
                    enabled=use_hot_cache,
                )
            matches.append(
                ExactMatchDecision(
                    query_index=query_index,
                    status=match.status,
                    identity_id=match.entry_id,
                    score=match.score,
                    ambiguous_identity_ids=match.ambiguous_ids,
                    scanned_confirmed_count=state.unique_count,
                    cache_hit=cache_hit,
                )
            )
        return ExactMatchResult(state=next_state, matches=tuple(matches))

    def update_row(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observation: O2SoftOutput,
        semantic_embeddings: Tensor,
        *,
        row: int,
        chunk_index: int,
    ) -> IdentityUpdateResult:
        """Commit one owner row exactly once for a monotonically increasing chunk index."""

        from ttt_svcbench_qwen.state_bank import StateBankRuntimeState, StructuredStateBank

        _require_live_state(identity_state)
        if not isinstance(state_bank, StructuredStateBank) or not isinstance(
            state_state, StateBankRuntimeState
        ):
            raise TypeError("update_row requires StructuredStateBank and StateBankRuntimeState")
        _validate_cross_bank_owner(identity_state, state_state)
        if type(row) is not int or row < 0 or row >= observation.identity.shape[0]:
            raise ValueError("O2 row is out of range")
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if chunk_index < identity_state.last_chunk_index:
            raise ValueError("Identity Bank rejects out-of-order chunks")
        width = int(observation.identity.shape[1])
        if semantic_embeddings.shape != (width, SEMANTIC_DIM) or not torch.is_floating_point(
            semantic_embeddings
        ):
            raise ValueError("O2 semantic_embeddings must be floating [K, 512]")
        observations = self._extract_observations(observation, semantic_embeddings, row)
        committed_positions = {item.position_id for item in observations}
        if len(committed_positions) > 1:
            raise ValueError("one O2 owner row must describe one committed position")
        committed_position = next(iter(committed_positions), None)
        if (
            committed_position is not None
            and committed_position < identity_state.last_committed_position_id
        ):
            raise ValueError("Identity Bank rejects out-of-order committed positions")
        same_position_replay = (
            committed_position is not None
            and committed_position == identity_state.last_committed_position_id
        )
        if chunk_index == identity_state.last_chunk_index or same_position_replay:
            replay_reason = (
                "same_committed_position" if same_position_replay else "same_committed_chunk"
            )
            replay_decisions = tuple(
                IdentityObservationDecision(
                    slot_index=item.slot_index,
                    position_id=item.position_id,
                    timestamp=item.timestamp,
                    status=IdentityDecisionStatus.REPLAY_IGNORED,
                    novelty=item.novelty,
                    match_confidence=item.match_confidence,
                    scanned_confirmed_count=identity_state.unique_count,
                    reason=replay_reason,
                )
                for item in observations
            )
            return IdentityUpdateResult(identity_state, state_state, replay_decisions)

        next_identity = identity_state
        next_state_bank = state_state
        decisions: dict[int, IdentityObservationDecision] = {}
        eligible: list[_Observation] = []
        for item in observations:
            signal = self._signal_kind(item)
            if signal is None:
                next_identity = replace(
                    next_identity,
                    signal_conflict_count=next_identity.signal_conflict_count + 1,
                    audit_log=next_identity.audit_log
                    + (
                        _audit(
                            "signal_conflict",
                            item,
                            (
                                ("novelty", item.novelty),
                                ("match_confidence", item.match_confidence),
                            ),
                        ),
                    ),
                    version=next_identity.version + 1,
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.SIGNAL_CONFLICT,
                    next_identity.unique_count,
                    reason="novelty_and_match_confidence_are_both_high_or_both_low",
                )
            else:
                eligible.append(item)

        next_identity, next_state_bank, confirmed_decisions, remaining = (
            self._assign_and_update_confirmed(
                next_identity,
                state_bank,
                next_state_bank,
                eligible,
            )
        )
        decisions.update(confirmed_decisions)
        next_identity, next_state_bank, candidate_decisions, remaining = (
            self._assign_and_update_candidates(
                next_identity,
                state_bank,
                next_state_bank,
                remaining,
                chunk_index,
            )
        )
        decisions.update(candidate_decisions)

        matched_candidate_ids = {
            decision.candidate_id
            for decision in candidate_decisions.values()
            if decision.status
            in {IdentityDecisionStatus.CANDIDATE_UPDATED, IdentityDecisionStatus.PROMOTED}
            and decision.candidate_id is not None
        }
        next_identity, next_state_bank = self._age_and_prune_candidates(
            next_identity,
            state_bank,
            next_state_bank,
            matched_candidate_ids,
            chunk_index,
            observations,
        )
        for item in sorted(remaining, key=lambda value: (value.position_id, value.slot_index)):
            next_identity, next_state_bank, decision = self._create_or_reject_candidate(
                next_identity,
                state_bank,
                next_state_bank,
                item,
                chunk_index,
            )
            decisions[item.slot_index] = decision
        next_identity = replace(
            next_identity,
            last_chunk_index=chunk_index,
            last_committed_position_id=(
                committed_position
                if committed_position is not None
                else identity_state.last_committed_position_id
            ),
            version=next_identity.version + 1,
        )
        ordered_decisions = tuple(decisions[index] for index in sorted(decisions))
        return IdentityUpdateResult(next_identity, next_state_bank, ordered_decisions)

    def _extract_observations(
        self, observation: O2SoftOutput, semantic_embeddings: Tensor, row: int
    ) -> tuple[_Observation, ...]:
        items: list[_Observation] = []
        for slot_index in range(observation.identity.shape[1]):
            if not bool(observation.valid_mask[row, slot_index]):
                continue
            timestamp = float(observation.timestamps[row, slot_index].item())
            position_id = int(observation.position_ids[row, slot_index].item())
            if not math.isfinite(timestamp) or timestamp < 0.0 or position_id < 0:
                raise ValueError("valid O2 observations require legal timestamp/position metadata")
            novelty = float(observation.score_probabilities[row, slot_index, 0].float().item())
            match_confidence = float(
                observation.score_probabilities[row, slot_index, 1].float().item()
            )
            confidence = max(novelty, match_confidence)
            items.append(
                _Observation(
                    slot_index=slot_index,
                    timestamp=timestamp,
                    position_id=position_id,
                    identity=_hard_identity(observation.identity[row, slot_index]),
                    semantic=semantic_embeddings[slot_index],
                    novelty=novelty,
                    match_confidence=match_confidence,
                    confidence=confidence,
                )
            )
        return tuple(sorted(items, key=lambda item: (item.position_id, item.slot_index)))

    def _signal_kind(self, item: _Observation) -> str | None:
        novelty_high = item.novelty >= self.o2_config.novelty_threshold
        match_high = item.match_confidence >= self.o2_config.match_confidence_threshold
        if novelty_high == match_high:
            return None
        return "new" if novelty_high else "match"

    def _assign_and_update_confirmed(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observations: Sequence[_Observation],
    ) -> tuple[
        IdentityBankRuntimeState,
        StateBankRuntimeState,
        dict[int, IdentityObservationDecision],
        tuple[_Observation, ...],
    ]:
        matches = {
            item.slot_index: self._match_confirmed(identity_state, item.identity)
            for item in observations
        }
        decisions: dict[int, IdentityObservationDecision] = {}
        remaining: list[_Observation] = []
        claims: dict[str, list[tuple[_Observation, _Match]]] = {}
        next_identity = identity_state
        next_state_bank = state_state
        for item in observations:
            match = matches[item.slot_index]
            if match.status is ExactMatchStatus.AMBIGUOUS:
                next_identity = self._record_match_conflict(
                    next_identity, item, "confirmed_near_tie", match.ambiguous_ids
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    similarity=match.score,
                    reason="confirmed_near_tie",
                )
            elif match.status is ExactMatchStatus.MATCHED and match.entry_id is not None:
                claims.setdefault(match.entry_id, []).append((item, match))
            else:
                remaining.append(item)
        winners: list[tuple[_Observation, _Match]] = []
        for identity_id, group in claims.items():
            ordered = sorted(
                group,
                key=lambda pair: (
                    -cast(float, pair[1].score),
                    -pair[0].match_confidence,
                    pair[0].slot_index,
                    identity_id,
                ),
            )
            winners.append(ordered[0])
            for loser, loser_match in ordered[1:]:
                next_identity = self._record_match_conflict(
                    next_identity, loser, "one_to_one_confirmed_claim", (identity_id,)
                )
                decisions[loser.slot_index] = self._decision(
                    loser,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    identity_id=identity_id,
                    similarity=loser_match.score,
                    reason="one_to_one_confirmed_claim",
                )
        for item, match in sorted(
            winners, key=lambda pair: (pair[0].position_id, pair[0].slot_index)
        ):
            assert match.entry_id is not None
            previous = self.confirmed_by_id(next_identity, match.entry_id)
            updated = replace(
                previous,
                identity_prototype=_prototype_ema(
                    previous.identity_prototype,
                    item.identity,
                    self.o2_config.prototype_ema,
                ),
                last_seen=item.timestamp,
                last_seen_position_id=item.position_id,
                observation_count=previous.observation_count + 1,
                prototype_version=previous.prototype_version + 1,
            )
            next_state_bank = state_bank.update_o2_confirmed(
                next_state_bank,
                confirmed=updated,
                semantic_embedding=item.semantic,
                confidence=item.confidence,
                audit_timestamp=item.timestamp,
            )
            next_identity = self._replace_confirmed(next_identity, updated, item)
            next_identity = self._touch_hot_cache(
                next_identity, updated.identity_id, item.position_id, enabled=True
            )
            decisions[item.slot_index] = self._decision(
                item,
                IdentityDecisionStatus.CONFIRMED_UPDATED,
                identity_state.unique_count,
                identity_id=updated.identity_id,
                similarity=match.score,
            )
        return next_identity, next_state_bank, decisions, tuple(remaining)

    def _assign_and_update_candidates(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observations: Sequence[_Observation],
        chunk_index: int,
    ) -> tuple[
        IdentityBankRuntimeState,
        StateBankRuntimeState,
        dict[int, IdentityObservationDecision],
        tuple[_Observation, ...],
    ]:
        matches = {
            item.slot_index: self._match_candidates(identity_state, item.identity)
            for item in observations
        }
        decisions: dict[int, IdentityObservationDecision] = {}
        remaining: list[_Observation] = []
        claims: dict[str, list[tuple[_Observation, _Match]]] = {}
        next_identity = identity_state
        next_state_bank = state_state
        for item in observations:
            match = matches[item.slot_index]
            if match.status is ExactMatchStatus.AMBIGUOUS:
                next_identity = self._record_match_conflict(
                    next_identity, item, "candidate_near_tie", match.ambiguous_ids
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    similarity=match.score,
                    reason="candidate_near_tie",
                )
            elif match.status is ExactMatchStatus.MATCHED and match.entry_id is not None:
                claims.setdefault(match.entry_id, []).append((item, match))
            else:
                remaining.append(item)
        winners: list[tuple[_Observation, _Match]] = []
        for candidate_id, group in claims.items():
            ordered = sorted(
                group,
                key=lambda pair: (
                    -cast(float, pair[1].score),
                    -pair[0].match_confidence,
                    pair[0].slot_index,
                    candidate_id,
                ),
            )
            winners.append(ordered[0])
            for loser, loser_match in ordered[1:]:
                next_identity = self._record_match_conflict(
                    next_identity, loser, "one_to_one_candidate_claim", (candidate_id,)
                )
                decisions[loser.slot_index] = self._decision(
                    loser,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    candidate_id=candidate_id,
                    similarity=loser_match.score,
                    reason="one_to_one_candidate_claim",
                )
        for item, match in sorted(
            winners, key=lambda pair: (pair[0].position_id, pair[0].slot_index)
        ):
            assert match.entry_id is not None
            previous = _candidate_by_id(next_identity, match.entry_id)
            reliable = item.confidence >= self.o2_config.reliability_threshold
            if reliable and chunk_index == previous.last_reliable_chunk_index + 1:
                reliable_streak = previous.reliable_streak + 1
            elif reliable and chunk_index > previous.last_reliable_chunk_index:
                reliable_streak = 1
            else:
                reliable_streak = previous.reliable_streak
            updated = replace(
                previous,
                identity_prototype=_prototype_ema(
                    previous.identity_prototype,
                    item.identity,
                    self.o2_config.prototype_ema,
                ),
                observation_count=previous.observation_count + 1,
                ttl_remaining=self.candidate_config.ttl_chunks,
                confidence=self.o2_config.prototype_ema * previous.confidence
                + (1.0 - self.o2_config.prototype_ema) * item.confidence,
                last_seen=item.timestamp,
                last_seen_position_id=item.position_id,
                last_reliable_chunk_index=chunk_index
                if reliable
                else previous.last_reliable_chunk_index,
                reliable_streak=reliable_streak,
            )
            if reliable_streak >= self.o2_config.confirmation_observations:
                next_identity, next_state_bank, confirmed = self._promote_candidate(
                    next_identity,
                    state_bank,
                    next_state_bank,
                    updated,
                    item,
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.PROMOTED,
                    identity_state.unique_count,
                    candidate_id=updated.candidate_id,
                    identity_id=confirmed.identity_id,
                    similarity=match.score,
                )
            else:
                next_state_bank = state_bank.update_o2_candidate(
                    next_state_bank,
                    candidate=updated,
                    semantic_embedding=item.semantic,
                    confidence=updated.confidence,
                    audit_timestamp=item.timestamp,
                )
                next_identity = _replace_candidate(next_identity, updated, item)
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.CANDIDATE_UPDATED,
                    next_identity.unique_count,
                    candidate_id=updated.candidate_id,
                    similarity=match.score,
                )
        return next_identity, next_state_bank, decisions, tuple(remaining)

    def _create_or_reject_candidate(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        item: _Observation,
        chunk_index: int,
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState, IdentityObservationDecision]:
        dynamic_match = self._match_candidates(identity_state, item.identity)
        if dynamic_match.status is not ExactMatchStatus.UNMATCHED:
            identity_state = self._record_match_conflict(
                identity_state,
                item,
                "same_chunk_candidate_claim",
                dynamic_match.ambiguous_ids
                or ((dynamic_match.entry_id,) if dynamic_match.entry_id is not None else ()),
            )
            return (
                identity_state,
                state_state,
                self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    candidate_id=dynamic_match.entry_id,
                    similarity=dynamic_match.score,
                    reason="same_chunk_candidate_claim",
                ),
            )
        next_identity = identity_state
        next_state_bank = state_state
        if len(next_identity.candidates) >= next_identity.candidate_capacity:
            if next_identity.candidate_capacity < self.candidate_config.hard_limit:
                next_identity = replace(
                    next_identity,
                    candidate_capacity=min(
                        next_identity.candidate_capacity + self.candidate_config.growth_chunk,
                        self.candidate_config.hard_limit,
                    ),
                    audit_log=next_identity.audit_log
                    + (
                        _audit(
                            "candidate_capacity_grow",
                            item,
                            (
                                (
                                    "new_capacity",
                                    min(
                                        next_identity.candidate_capacity
                                        + self.candidate_config.growth_chunk,
                                        self.candidate_config.hard_limit,
                                    ),
                                ),
                            ),
                        ),
                    ),
                    version=next_identity.version + 1,
                )
            else:
                next_identity = replace(
                    next_identity,
                    candidate_overflow_count=next_identity.candidate_overflow_count + 1,
                    audit_log=next_identity.audit_log
                    + (
                        _audit(
                            "candidate_overflow_rejected",
                            item,
                            (("hard_limit", self.candidate_config.hard_limit),),
                        ),
                    ),
                    version=next_identity.version + 1,
                )
                return (
                    next_identity,
                    next_state_bank,
                    self._decision(
                        item,
                        IdentityDecisionStatus.OVERFLOW_REJECTED,
                        next_identity.unique_count,
                        reason="candidate_hard_limit_reached",
                    ),
                )
        candidate_id = f"candidate-{next_identity.next_candidate_sequence:08d}"
        draft = CandidateIdentity(
            candidate_id=candidate_id,
            identity_prototype=item.identity,
            observation_count=1,
            ttl_remaining=self.candidate_config.ttl_chunks,
            confidence=item.confidence,
            first_seen=item.timestamp,
            last_seen=item.timestamp,
            first_seen_position_id=item.position_id,
            last_seen_position_id=item.position_id,
            last_reliable_chunk_index=chunk_index,
            reliable_streak=1 if item.confidence >= self.o2_config.reliability_threshold else 0,
            semantic_record_id=None,
        )
        next_state_bank, record = state_bank.append_o2_candidate(
            next_state_bank,
            semantic_embedding=item.semantic,
            candidate=draft,
            confidence=item.confidence,
        )
        linked = cast(CandidateIdentity, record.payload)
        next_identity = replace(
            next_identity,
            candidates=next_identity.candidates + (_clone_candidate(linked),),
            next_candidate_sequence=next_identity.next_candidate_sequence + 1,
            issued_candidate_ids=next_identity.issued_candidate_ids + (candidate_id,),
            audit_log=next_identity.audit_log
            + (
                _audit(
                    "candidate_created",
                    item,
                    (("candidate_id", candidate_id), ("record_id", linked.semantic_record_id)),
                ),
            ),
            version=next_identity.version + 1,
        )
        return (
            next_identity,
            next_state_bank,
            self._decision(
                item,
                IdentityDecisionStatus.CANDIDATE_CREATED,
                next_identity.unique_count,
                candidate_id=candidate_id,
            ),
        )

    def _promote_candidate(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        candidate: CandidateIdentity,
        item: _Observation,
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState, ConfirmedIdentity]:
        identity_id = f"identity-{identity_state.next_identity_sequence:08d}"
        draft = ConfirmedIdentity(
            identity_id=identity_id,
            identity_prototype=candidate.identity_prototype,
            first_seen=candidate.first_seen,
            last_seen=candidate.last_seen,
            observation_count=candidate.observation_count,
            semantic_record_id=None,
            prototype_version=0,
            first_seen_position_id=candidate.first_seen_position_id,
            last_seen_position_id=candidate.last_seen_position_id,
        )
        assert candidate.semantic_record_id is not None
        next_state_bank, record = state_bank.promote_o2_candidate(
            state_state,
            candidate.semantic_record_id,
            semantic_embedding=item.semantic,
            confirmed=draft,
            confidence=candidate.confidence,
            audit_timestamp=item.timestamp,
        )
        linked = cast(ConfirmedIdentity, record.payload)
        remaining = tuple(
            _clone_candidate(value)
            for value in identity_state.candidates
            if value.candidate_id != candidate.candidate_id
        )
        next_identity = replace(
            identity_state,
            candidates=remaining,
            confirmed_chunks=_append_confirmed(
                identity_state.confirmed_chunks,
                linked,
                self.confirmed_config.growth_chunk,
            ),
            next_identity_sequence=identity_state.next_identity_sequence + 1,
            issued_identity_ids=identity_state.issued_identity_ids + (identity_id,),
            audit_log=identity_state.audit_log
            + (
                _audit(
                    "candidate_promoted",
                    item,
                    (
                        ("candidate_id", candidate.candidate_id),
                        ("identity_id", identity_id),
                        ("record_id", linked.semantic_record_id),
                    ),
                ),
            ),
            version=identity_state.version + 1,
        )
        next_identity = self._touch_hot_cache(
            next_identity, identity_id, item.position_id, enabled=True
        )
        return next_identity, next_state_bank, linked

    def _age_and_prune_candidates(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        refreshed_candidate_ids: set[str],
        chunk_index: int,
        observations: Sequence[_Observation],
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState]:
        last_audit = identity_state.audit_log[-1] if identity_state.audit_log else None
        timestamp = max(
            max((item.timestamp for item in observations), default=float(chunk_index)),
            last_audit.timestamp if last_audit is not None else 0.0,
        )
        position_id = max(
            max((item.position_id for item in observations), default=chunk_index),
            last_audit.position_id if last_audit is not None else 0,
        )
        kept: list[CandidateIdentity] = []
        expired: list[CandidateIdentity] = []
        low_confidence: list[CandidateIdentity] = []
        for candidate in identity_state.candidates:
            if candidate.candidate_id in refreshed_candidate_ids:
                aged = candidate
            else:
                aged = replace(candidate, ttl_remaining=max(candidate.ttl_remaining - 1, 0))
            if aged.ttl_remaining == 0:
                expired.append(aged)
            elif aged.confidence < self.candidate_config.low_confidence_threshold:
                low_confidence.append(aged)
            else:
                kept.append(aged)
        low_confidence.sort(
            key=lambda candidate: (
                candidate.confidence,
                candidate.last_seen_position_id,
                candidate.candidate_id,
            )
        )
        next_state_bank = state_state
        for candidate, reason in (
            *((candidate, "ttl_expired") for candidate in expired),
            *((candidate, "low_confidence") for candidate in low_confidence),
        ):
            assert candidate.semantic_record_id is not None
            next_state_bank = state_bank.invalidate_o2_candidate(
                next_state_bank,
                candidate.semantic_record_id,
                audit_timestamp=timestamp,
                reason=reason,
            )
        if not expired and not low_confidence and tuple(kept) == identity_state.candidates:
            return identity_state, next_state_bank
        audit_log = identity_state.audit_log
        if expired:
            audit_log += (
                IdentityBankAuditEntry(
                    "candidate_ttl_prune",
                    timestamp,
                    position_id,
                    (("count", len(expired)),),
                ),
            )
        if low_confidence:
            audit_log += (
                IdentityBankAuditEntry(
                    "candidate_low_confidence_prune",
                    timestamp,
                    position_id,
                    (("count", len(low_confidence)),),
                ),
            )
        return (
            replace(
                identity_state,
                candidates=tuple(_clone_candidate(candidate) for candidate in kept),
                candidate_expired_count=identity_state.candidate_expired_count + len(expired),
                candidate_low_confidence_pruned_count=(
                    identity_state.candidate_low_confidence_pruned_count + len(low_confidence)
                ),
                audit_log=audit_log,
                version=identity_state.version + 1,
            ),
            next_state_bank,
        )

    def _match_confirmed(self, state: IdentityBankRuntimeState, query: Tensor) -> _Match:
        identities = state.confirmed
        if not identities:
            return _Match(ExactMatchStatus.UNMATCHED, None, None)
        prototypes = torch.stack([identity.identity_prototype for identity in identities])
        scores = prototypes @ query
        return _select_match(
            tuple(identity.identity_id for identity in identities),
            scores,
            self.o2_config.match_threshold,
            self.o2_config.match_ambiguity_margin,
        )

    def _match_candidates(self, state: IdentityBankRuntimeState, query: Tensor) -> _Match:
        if not state.candidates:
            return _Match(ExactMatchStatus.UNMATCHED, None, None)
        prototypes = torch.stack([candidate.identity_prototype for candidate in state.candidates])
        scores = prototypes @ query
        return _select_match(
            tuple(candidate.candidate_id for candidate in state.candidates),
            scores,
            self.candidate_config.match_threshold,
            self.o2_config.match_ambiguity_margin,
        )

    def _replace_confirmed(
        self,
        state: IdentityBankRuntimeState,
        confirmed: ConfirmedIdentity,
        item: _Observation,
    ) -> IdentityBankRuntimeState:
        old = self.confirmed_by_id(state, confirmed.identity_id)
        chunks = _update_confirmed(state.confirmed_chunks, confirmed)
        return replace(
            state,
            confirmed_chunks=chunks,
            audit_log=state.audit_log
            + (
                _audit(
                    "confirmed_updated",
                    item,
                    (
                        ("identity_id", confirmed.identity_id),
                        ("old_prototype_checksum", _tensor_checksum(old.identity_prototype)),
                        ("new_prototype_checksum", _tensor_checksum(confirmed.identity_prototype)),
                        ("prototype_version", confirmed.prototype_version),
                    ),
                ),
            ),
            version=state.version + 1,
        )

    def _touch_hot_cache(
        self,
        state: IdentityBankRuntimeState,
        identity_id: str,
        position_id: int,
        *,
        enabled: bool,
    ) -> IdentityBankRuntimeState:
        if not enabled or not state.hot_cache_enabled:
            return state
        confirmed = self.confirmed_by_id(state, identity_id)
        assert state.hot_cache_device is not None
        dtype = _parse_dtype(state.hot_cache_dtype)
        prototype = confirmed.identity_prototype.to(
            device=torch.device(state.hot_cache_device), dtype=dtype, copy=True
        ).detach()
        replacement = HotCacheEntry(
            identity_id=identity_id,
            identity_prototype=prototype,
            last_accessed_position_id=position_id,
            prototype_version=confirmed.prototype_version,
        )
        entries = [
            _clone_hot_cache(entry) for entry in state.hot_cache if entry.identity_id != identity_id
        ]
        entries.append(replacement)
        evicted: HotCacheEntry | None = None
        if len(entries) > state.hot_cache_capacity:
            evicted = min(
                entries,
                key=lambda entry: (entry.last_accessed_position_id, entry.identity_id),
            )
            entries = [entry for entry in entries if entry.identity_id != evicted.identity_id]
        timestamp = confirmed.last_seen
        details: tuple[tuple[str, AuditValue], ...] = (
            ("identity_id", identity_id),
            ("prototype_version", confirmed.prototype_version),
            ("evicted_identity_id", evicted.identity_id if evicted is not None else None),
        )
        return replace(
            state,
            hot_cache=tuple(entries),
            audit_log=state.audit_log
            + (IdentityBankAuditEntry("hot_cache_touch", timestamp, position_id, details),),
            version=state.version + 1,
        )

    def _record_match_conflict(
        self,
        state: IdentityBankRuntimeState,
        item: _Observation,
        reason: str,
        conflicting_ids: tuple[str, ...],
    ) -> IdentityBankRuntimeState:
        return replace(
            state,
            match_conflict_count=state.match_conflict_count + 1,
            audit_log=state.audit_log
            + (
                _audit(
                    "match_conflict",
                    item,
                    (("reason", reason), ("conflicting_ids", ",".join(conflicting_ids))),
                ),
            ),
            version=state.version + 1,
        )

    def _decision(
        self,
        item: _Observation,
        status: IdentityDecisionStatus,
        scanned: int,
        *,
        candidate_id: str | None = None,
        identity_id: str | None = None,
        similarity: float | None = None,
        reason: str | None = None,
    ) -> IdentityObservationDecision:
        return IdentityObservationDecision(
            slot_index=item.slot_index,
            position_id=item.position_id,
            timestamp=item.timestamp,
            status=status,
            candidate_id=candidate_id,
            identity_id=identity_id,
            similarity=similarity,
            novelty=item.novelty,
            match_confidence=item.match_confidence,
            scanned_confirmed_count=scanned,
            reason=reason,
        )

    def _resolve_hot_cache(
        self,
        requested: bool,
        hot_device: str | torch.device | None,
    ) -> tuple[bool, str | None, str | None]:
        if not requested:
            return False, None, "disabled_by_caller"
        explicit = torch.device(hot_device) if hot_device is not None else None
        configured = torch.device(self.confirmed_config.hot_cache_device)
        device = explicit or configured
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return False, None, "cuda_unavailable"
            device = torch.device(
                "cuda",
                torch.cuda.current_device() if device.index is None else device.index,
            )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Hot Cache device must be explicit CPU test backend or CUDA")
        return True, str(device), None

    def _validate_config(self) -> None:
        checks: tuple[tuple[str, object, object], ...] = (
            ("O2 identity_dim", self.o2_config.identity_dim, IDENTITY_DIM),
            ("O2 prototype_ema", self.o2_config.prototype_ema, 0.9),
            ("O2 confirmation_observations", self.o2_config.confirmation_observations, 2),
            ("O2 match_threshold", self.o2_config.match_threshold, 0.8),
            ("Candidate initial_capacity", self.candidate_config.initial_capacity, 64),
            ("Candidate growth_chunk", self.candidate_config.growth_chunk, 64),
            ("Candidate hard_limit", self.candidate_config.hard_limit, 512),
            ("Candidate ttl_chunks", self.candidate_config.ttl_chunks, 8),
            ("Confirmed initial_capacity", self.confirmed_config.initial_capacity, 256),
            ("Confirmed growth_chunk", self.confirmed_config.growth_chunk, 256),
            ("Confirmed hard_limit", self.confirmed_config.hard_limit, None),
            ("Confirmed storage_device", self.confirmed_config.storage_device, "cpu"),
            ("Confirmed storage_dtype", self.confirmed_config.storage_dtype, "float32"),
            ("Confirmed gpu_hot_capacity", self.confirmed_config.gpu_hot_capacity, 256),
            ("Confirmed exact_search", self.confirmed_config.exact_search, True),
            ("Confirmed ann_enabled", self.confirmed_config.ann_enabled, False),
        )
        for name, actual, expected in checks:
            if actual != expected:
                raise ValueError(f"{name} must equal {expected!r}; got {actual!r}")


def build_identity_bank(config: ProjectConfig | None = None) -> IdentityBank:
    if config is None:
        raise ValueError("build_identity_bank requires a validated ProjectConfig")
    return IdentityBank(config)


def evaluate_identity_quality(
    ground_truth_entity_ids: Sequence[str],
    predicted_identity_ids: Sequence[str | None],
) -> IdentityQualityMetrics:
    """Compute trajectory-end quality outside runtime; labels never enter IdentityBank."""

    if len(ground_truth_entity_ids) != len(predicted_identity_ids):
        raise ValueError("ground truth and prediction sequences must have equal length")
    if any(not entity_id for entity_id in ground_truth_entity_ids):
        raise ValueError("ground-truth entity IDs must be non-empty")
    grouped: dict[str, set[str]] = {}
    for ground_truth, predicted in zip(
        ground_truth_entity_ids, predicted_identity_ids, strict=True
    ):
        grouped.setdefault(ground_truth, set())
        if predicted is not None:
            if not predicted:
                raise ValueError("predicted identity IDs cannot be empty strings")
            grouped[ground_truth].add(predicted)
    duplicate_excess = sum(max(len(predictions) - 1, 0) for predictions in grouped.values())
    mapped_confirmed = sum(len(predictions) for predictions in grouped.values())
    duplicate_denominator = max(mapped_confirmed, 1)
    missed = sum(not predictions for predictions in grouped.values())
    ground_truth_denominator = max(len(grouped), 1)
    return IdentityQualityMetrics(
        duplicate_excess_count=duplicate_excess,
        duplicate_denominator=duplicate_denominator,
        duplicate_rate=duplicate_excess / duplicate_denominator,
        missed_new_count=missed,
        ground_truth_identity_denominator=ground_truth_denominator,
        missed_new_identity_rate=missed / ground_truth_denominator,
    )


def _empty_confirmed_chunk(capacity: int) -> ConfirmedChunk:
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Confirmed capacity must be positive")
    return ConfirmedChunk(
        prototypes=torch.zeros(capacity, IDENTITY_DIM, dtype=torch.float32),
        occupied=torch.zeros(capacity, dtype=torch.bool),
        identity_ids=(None,) * capacity,
        first_seen=torch.full((capacity,), -1.0, dtype=torch.float64),
        last_seen=torch.full((capacity,), -1.0, dtype=torch.float64),
        observation_counts=torch.zeros(capacity, dtype=torch.int64),
        first_seen_position_ids=torch.full((capacity,), -1, dtype=torch.int64),
        last_seen_position_ids=torch.full((capacity,), -1, dtype=torch.int64),
        semantic_record_ids=(None,) * capacity,
        prototype_versions=torch.zeros(capacity, dtype=torch.int64),
    )


def _append_confirmed(
    chunks: tuple[ConfirmedChunk, ...],
    confirmed: ConfirmedIdentity,
    growth_chunk: int,
) -> tuple[ConfirmedChunk, ...]:
    if confirmed.semantic_record_id is None:
        raise ValueError("Confirmed store requires a semantic record link")
    cloned = [_clone_chunk(chunk) for chunk in chunks]
    target_index = next(
        (index for index, chunk in enumerate(cloned) if chunk.size < chunk.capacity), None
    )
    if target_index is None:
        cloned.append(_empty_confirmed_chunk(growth_chunk))
        target_index = len(cloned) - 1
    target = cloned[target_index]
    slot = int(torch.nonzero(~target.occupied, as_tuple=False)[0].item())
    prototypes = target.prototypes.detach().clone()
    occupied = target.occupied.detach().clone()
    first_seen = target.first_seen.detach().clone()
    last_seen = target.last_seen.detach().clone()
    counts = target.observation_counts.detach().clone()
    first_positions = target.first_seen_position_ids.detach().clone()
    last_positions = target.last_seen_position_ids.detach().clone()
    versions = target.prototype_versions.detach().clone()
    identity_ids = list(target.identity_ids)
    record_ids = list(target.semantic_record_ids)
    prototypes[slot] = _hard_identity(confirmed.identity_prototype)
    occupied[slot] = True
    first_seen[slot] = confirmed.first_seen
    last_seen[slot] = confirmed.last_seen
    counts[slot] = confirmed.observation_count
    first_positions[slot] = confirmed.first_seen_position_id
    last_positions[slot] = confirmed.last_seen_position_id
    versions[slot] = confirmed.prototype_version
    identity_ids[slot] = confirmed.identity_id
    record_ids[slot] = confirmed.semantic_record_id
    cloned[target_index] = ConfirmedChunk(
        prototypes,
        occupied,
        tuple(identity_ids),
        first_seen,
        last_seen,
        counts,
        first_positions,
        last_positions,
        tuple(record_ids),
        versions,
    )
    return tuple(cloned)


def _update_confirmed(
    chunks: tuple[ConfirmedChunk, ...], confirmed: ConfirmedIdentity
) -> tuple[ConfirmedChunk, ...]:
    if confirmed.semantic_record_id is None:
        raise ValueError("Confirmed update requires a semantic record link")
    cloned = [_clone_chunk(chunk) for chunk in chunks]
    locations = [
        (chunk_index, slot)
        for chunk_index, chunk in enumerate(cloned)
        for slot, identity_id in enumerate(chunk.identity_ids)
        if identity_id == confirmed.identity_id
    ]
    if len(locations) != 1:
        raise KeyError(f"Confirmed identity not found exactly once: {confirmed.identity_id}")
    chunk_index, slot = locations[0]
    chunk = cloned[chunk_index]
    prototypes = chunk.prototypes.detach().clone()
    last_seen = chunk.last_seen.detach().clone()
    counts = chunk.observation_counts.detach().clone()
    last_positions = chunk.last_seen_position_ids.detach().clone()
    versions = chunk.prototype_versions.detach().clone()
    record_ids = list(chunk.semantic_record_ids)
    prototypes[slot] = _hard_identity(confirmed.identity_prototype)
    last_seen[slot] = confirmed.last_seen
    counts[slot] = confirmed.observation_count
    last_positions[slot] = confirmed.last_seen_position_id
    versions[slot] = confirmed.prototype_version
    record_ids[slot] = confirmed.semantic_record_id
    cloned[chunk_index] = ConfirmedChunk(
        prototypes,
        chunk.occupied.detach().clone(),
        tuple(chunk.identity_ids),
        chunk.first_seen.detach().clone(),
        last_seen,
        counts,
        chunk.first_seen_position_ids.detach().clone(),
        last_positions,
        tuple(record_ids),
        versions,
    )
    return tuple(cloned)


def _select_match(
    entry_ids: tuple[str, ...],
    scores: Tensor,
    threshold: float,
    ambiguity_margin: float,
) -> _Match:
    if scores.ndim != 1 or scores.shape[0] != len(entry_ids):
        raise ValueError("match scores and IDs must align")
    ordered = sorted(
        (
            (float(score.item()), entry_id)
            for score, entry_id in zip(scores, entry_ids, strict=True)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_id = ordered[0]
    if best_score < threshold:
        return _Match(ExactMatchStatus.UNMATCHED, None, best_score)
    if len(ordered) > 1:
        second_score, second_id = ordered[1]
        if second_score >= threshold and best_score - second_score <= ambiguity_margin:
            return _Match(
                ExactMatchStatus.AMBIGUOUS,
                None,
                best_score,
                tuple(sorted((best_id, second_id))),
            )
    return _Match(ExactMatchStatus.MATCHED, best_id, best_score)


def _prototype_ema(old: Tensor, observation: Tensor, decay: float) -> Tensor:
    return _normalize_identity(decay * old.float() + (1.0 - decay) * observation.float())


def _hard_identity(identity: Tensor) -> Tensor:
    _validate_identity_tensor(identity, "identity observation")
    return _normalize_identity(identity.detach().to(device="cpu", dtype=torch.float32, copy=True))


def _normalize_identity(identity: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(identity.float())
    if not bool(torch.isfinite(norm)):
        raise ValueError("identity prototype norm must be finite")
    if float(norm.item()) <= 1.0e-8:
        output = torch.zeros(IDENTITY_DIM, dtype=torch.float32, device="cpu")
        output[0] = 1.0
        return output
    return F.normalize(identity.float(), dim=0, eps=1.0e-8).to(device="cpu", copy=True).detach()


def _clone_candidate(candidate: CandidateIdentity) -> CandidateIdentity:
    return replace(candidate, identity_prototype=candidate.identity_prototype.detach().clone())


def _clone_confirmed(confirmed: ConfirmedIdentity) -> ConfirmedIdentity:
    return replace(confirmed, identity_prototype=confirmed.identity_prototype.detach().clone())


def _clone_chunk(chunk: ConfirmedChunk) -> ConfirmedChunk:
    return ConfirmedChunk(
        prototypes=chunk.prototypes.detach().clone(),
        occupied=chunk.occupied.detach().clone(),
        identity_ids=tuple(chunk.identity_ids),
        first_seen=chunk.first_seen.detach().clone(),
        last_seen=chunk.last_seen.detach().clone(),
        observation_counts=chunk.observation_counts.detach().clone(),
        first_seen_position_ids=chunk.first_seen_position_ids.detach().clone(),
        last_seen_position_ids=chunk.last_seen_position_ids.detach().clone(),
        semantic_record_ids=tuple(chunk.semantic_record_ids),
        prototype_versions=chunk.prototype_versions.detach().clone(),
    )


def _clone_hot_cache(entry: HotCacheEntry) -> HotCacheEntry:
    return replace(entry, identity_prototype=entry.identity_prototype.detach().clone())


def _clone_runtime_state(state: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
    return replace(
        state,
        candidates=tuple(_clone_candidate(candidate) for candidate in state.candidates),
        confirmed_chunks=tuple(_clone_chunk(chunk) for chunk in state.confirmed_chunks),
        hot_cache=tuple(_clone_hot_cache(entry) for entry in state.hot_cache),
        audit_log=tuple(state.audit_log),
    )


def _replace_candidate(
    state: IdentityBankRuntimeState, candidate: CandidateIdentity, item: _Observation
) -> IdentityBankRuntimeState:
    matches = [value for value in state.candidates if value.candidate_id == candidate.candidate_id]
    if len(matches) != 1:
        raise KeyError(f"Candidate not found exactly once: {candidate.candidate_id}")
    old = matches[0]
    candidates = tuple(
        _clone_candidate(candidate if value.candidate_id == candidate.candidate_id else value)
        for value in state.candidates
    )
    return replace(
        state,
        candidates=candidates,
        audit_log=state.audit_log
        + (
            _audit(
                "candidate_updated",
                item,
                (
                    ("candidate_id", candidate.candidate_id),
                    ("old_prototype_checksum", _tensor_checksum(old.identity_prototype)),
                    ("new_prototype_checksum", _tensor_checksum(candidate.identity_prototype)),
                    ("reliable_streak", candidate.reliable_streak),
                ),
            ),
        ),
        version=state.version + 1,
    )


def _candidate_by_id(state: IdentityBankRuntimeState, candidate_id: str) -> CandidateIdentity:
    matches = tuple(
        candidate for candidate in state.candidates if candidate.candidate_id == candidate_id
    )
    if len(matches) != 1:
        raise KeyError(f"Candidate not found: {candidate_id}")
    return _clone_candidate(matches[0])


def _audit(
    action: str,
    item: _Observation,
    details: tuple[tuple[str, AuditValue], ...] = (),
) -> IdentityBankAuditEntry:
    return IdentityBankAuditEntry(action, item.timestamp, item.position_id, details)


def _tensor_checksum(tensor: Tensor) -> str:
    raw = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _validate_identity_tensor(tensor: Tensor, name: str) -> None:
    if tensor.shape != (IDENTITY_DIM,) or not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be floating [256]")
    if tensor.device.type != "meta" and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _validate_authoritative_identity(tensor: Tensor, name: str) -> None:
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or tensor.grad_fn is not None
    ):
        raise ValueError(f"{name} must be detached CPU FP32")
    _validate_unit_identity(tensor, name)


def _validate_unit_identity(tensor: Tensor, name: str) -> None:
    norm = torch.linalg.vector_norm(tensor.float())
    if not torch.allclose(norm, torch.ones_like(norm), atol=_UNIT_ATOL, rtol=_UNIT_RTOL):
        raise ValueError(f"{name} must have unit L2 norm")


def _validate_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must stay within [0, 1]")


def _validate_seen_metadata(
    first_seen: float,
    last_seen: float,
    first_position: int,
    last_position: int,
    name: str,
) -> None:
    if (
        not math.isfinite(first_seen)
        or not math.isfinite(last_seen)
        or first_seen < 0.0
        or last_seen < first_seen
        or type(first_position) is not int
        or type(last_position) is not int
        or first_position < 0
        or last_position < first_position
    ):
        raise ValueError(f"{name} first/last seen metadata is invalid")


def _validate_cross_bank_owner(
    identity_state: IdentityBankRuntimeState, state_state: StateBankRuntimeState
) -> None:
    if (
        identity_state.video_id != state_state.video_id
        or identity_state.trajectory_id != state_state.trajectory_id
    ):
        raise ValueError("Identity Bank and State Bank owner identifiers must agree")
    if identity_state.released != state_state.released:
        raise ValueError("Identity Bank and State Bank release state must agree")


def _require_live_state(state: IdentityBankRuntimeState) -> None:
    if not isinstance(state, IdentityBankRuntimeState):
        raise TypeError("Identity Bank operation requires IdentityBankRuntimeState")
    if state.released:
        raise ValueError("released Identity Bank runtime cannot be used")


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"unsupported Hot Cache dtype: {name}")
    return mapping[name]


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
    if dtype not in mapping:
        raise ValueError(f"unsupported Hot Cache dtype: {dtype}")
    return mapping[dtype]


def _storage_key(tensor: Tensor) -> tuple[str, int | None]:
    if tensor.device.type == "meta":
        return (str(tensor.device), None)
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _assert_runtime_storage_isolated(state: IdentityBankRuntimeState) -> None:
    tensors: list[Tensor] = []
    tensors.extend(candidate.identity_prototype for candidate in state.candidates)
    for chunk in state.confirmed_chunks:
        tensors.extend(
            (
                chunk.prototypes,
                chunk.occupied,
                chunk.first_seen,
                chunk.last_seen,
                chunk.observation_counts,
                chunk.first_seen_position_ids,
                chunk.last_seen_position_ids,
                chunk.prototype_versions,
            )
        )
    tensors.extend(entry.identity_prototype for entry in state.hot_cache)
    keys = [_storage_key(tensor) for tensor in tensors if tensor.device.type != "meta"]
    if len(keys) != len(set(keys)):
        raise ValueError("Identity Bank runtime tensors must not share storage")
