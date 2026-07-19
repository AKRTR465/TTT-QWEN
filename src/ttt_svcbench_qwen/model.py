"""Compose the P13 State-TTT stages without owning their algorithms.

Inputs: injected P3/P5-P12 components, immutable stage requests, and one explicit
per-owner prefill lifecycle.
Outputs: observation intermediates, one audited Qwen prefill, and decode outputs.
Forbidden: local Adapter/SGD, FSM/Bank mutation, Retriever, Reader, Resampler,
Composer, or Qwen masking implementations.

The deliberately small protocols in this module are orchestration seams.  Thin
adapters may translate them to the existing component signatures, while the
authoritative component implementations continue to own every numerical or hard
state rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import RLock
from typing import Protocol

from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import FastWeightsState, OptimizerRuntimeState
from ttt_svcbench_qwen.identity_bank import IdentityBankRuntimeState
from ttt_svcbench_qwen.input_composer import ComposedInput
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E2RuntimeState,
    ObservationOutputs,
)
from ttt_svcbench_qwen.query_encoder import QueryEncoderOutput
from ttt_svcbench_qwen.state_bank import StateBankRuntimeState, StructuredStateBank
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
    TemporalEncoderOutput,
)
from ttt_svcbench_qwen.state_reader import ReaderResult, StateResamplerOutput
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


class LifecycleError(RuntimeError):
    """Raised when an owner attempts an illegal observe/prefill/decode transition."""


class LifecyclePhase(StrEnum):
    READY = "ready"
    PREFILLED = "prefilled"
    DECODING = "decoding"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeOwner:
    """Canonical batch ownership used by every P13 entrypoint."""

    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.video_ids or len(self.video_ids) != len(self.trajectory_ids):
            raise ValueError("runtime owner IDs must contain one aligned non-empty batch")
        pairs = tuple(zip(self.video_ids, self.trajectory_ids, strict=True))
        if any(not video_id or not trajectory_id for video_id, trajectory_id in pairs):
            raise ValueError("runtime owner IDs must be non-empty")
        if len(set(pairs)) != len(pairs):
            raise ValueError("runtime owner rows must be unique")


class OnlineOverlapMemory(Protocol):
    """Typed detached overlap state shared by Meta-TTT and online inference."""

    owner: RuntimeOwner
    end_time: float
    identity: Tensor
    identity_valid_mask: Tensor
    identity_position_ids: Tensor
    identity_timestamps: Tensor
    e1_probabilities: Tensor
    e2_event_probabilities: Tensor
    e2_phase_probabilities: Tensor
    event_valid_mask: Tensor
    event_position_ids: Tensor
    event_timestamps: Tensor

    @property
    def tensors(self) -> tuple[Tensor, ...]: ...


@dataclass(frozen=True, slots=True)
class TrajectoryRuntimeState:
    """One authoritative trajectory across training and online inference."""

    owner: RuntimeOwner
    next_chunk_index: int
    slot_state: SpatialSlotRuntimeState | None
    temporal_cache: TemporalCache | None
    e1_state: E1RuntimeState | None
    e2_state: E2RuntimeState | None
    state_bank: StateBankRuntimeState
    identity_bank: IdentityBankRuntimeState
    fast_weights: FastWeightsState | None = None
    optimizer: OptimizerRuntimeState | None = None
    reader_audit: tuple[ReaderResult, ...] = ()
    online_overlap_memory: OnlineOverlapMemory | None = None
    released: bool = False

    def __post_init__(self) -> None:
        if len(self.owner.video_ids) != 1:
            raise ValueError("trajectory runtime owner must contain exactly one row")
        if type(self.next_chunk_index) is not int or self.next_chunk_index < 0:
            raise ValueError("trajectory next_chunk_index must be non-negative")
        if (self.fast_weights is None) != (self.optimizer is None):
            raise ValueError("fast weights and optimizer state must be both present or both absent")
        video_id = self.video_id
        trajectory_id = self.trajectory_id
        for name, state in (("State Bank", self.state_bank), ("Identity Bank", self.identity_bank)):
            if (state.video_id, state.trajectory_id) != (video_id, trajectory_id):
                raise ValueError(f"{name} ownership does not match trajectory runtime")
            if state.released != self.released:
                raise ValueError(f"{name} release state does not match trajectory runtime")
        if self.slot_state is not None and self.slot_state.video_id != video_id:
            raise ValueError("slot-state ownership does not match trajectory runtime")
        for name, operator_state in (("E1", self.e1_state), ("E2", self.e2_state)):
            if operator_state is not None and (
                operator_state.video_id,
                operator_state.trajectory_id,
            ) != (
                video_id,
                trajectory_id,
            ):
                raise ValueError(f"{name} ownership does not match trajectory runtime")
        if self.temporal_cache is not None:
            owners = tuple(
                zip(
                    self.temporal_cache.video_ids,
                    self.temporal_cache.trajectory_ids,
                    strict=True,
                )
            )
            if (video_id, trajectory_id) not in owners:
                raise ValueError("temporal-cache ownership does not include trajectory runtime")
            row_index = owners.index((video_id, trajectory_id))
            for name, operator_state in (("E1", self.e1_state), ("E2", self.e2_state)):
                if operator_state is None:
                    continue
                if (
                    operator_state.query_signature.dtype != self.temporal_cache.hidden.dtype
                    or operator_state.query_signature.device != self.temporal_cache.hidden.device
                    or not operator_state.query_signature.equal(
                        self.temporal_cache.query_signatures[row_index]
                    )
                ):
                    raise ValueError(f"{name} state query signature does not match temporal cache")
                if operator_state.total_seen != int(
                    self.temporal_cache.total_seen[row_index].item()
                ):
                    raise ValueError(f"{name} state position does not match temporal cache")

    @property
    def video_id(self) -> str:
        return self.owner.video_ids[0]

    @property
    def trajectory_id(self) -> str:
        return self.owner.trajectory_ids[0]


@dataclass(frozen=True, slots=True)
class BatchRuntimeState:
    """The sole batch representation: an aligned tuple of trajectory rows."""

    rows: tuple[TrajectoryRuntimeState, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("batch runtime requires at least one trajectory row")
        owners = tuple((row.video_id, row.trajectory_id) for row in self.rows)
        if len(set(owners)) != len(owners):
            raise ValueError("batch runtime trajectory owners must be unique")
        if len({row.next_chunk_index for row in self.rows}) != 1:
            raise ValueError("batch runtime rows must share one next_chunk_index")
        caches = tuple(row.temporal_cache for row in self.rows if row.temporal_cache is not None)
        if caches and any(cache is not caches[0] for cache in caches[1:]):
            raise ValueError("batch runtime rows must share the authoritative temporal cache")

    @property
    def owner(self) -> RuntimeOwner:
        return RuntimeOwner(
            tuple(row.video_id for row in self.rows),
            tuple(row.trajectory_id for row in self.rows),
        )

    @property
    def next_chunk_index(self) -> int:
        return self.rows[0].next_chunk_index

    @property
    def slot_states(self) -> tuple[SpatialSlotRuntimeState | None, ...]:
        return tuple(row.slot_state for row in self.rows)

    @property
    def temporal_cache(self) -> TemporalCache | None:
        return self.rows[0].temporal_cache

    @property
    def e1_states(self) -> tuple[E1RuntimeState | None, ...]:
        return tuple(row.e1_state for row in self.rows)

    @property
    def e2_states(self) -> tuple[E2RuntimeState | None, ...]:
        return tuple(row.e2_state for row in self.rows)

    @property
    def state_bank_states(self) -> tuple[StateBankRuntimeState, ...]:
        return tuple(row.state_bank for row in self.rows)

    @property
    def identity_bank_states(self) -> tuple[IdentityBankRuntimeState, ...]:
        return tuple(row.identity_bank for row in self.rows)

    @property
    def bank_states(self) -> tuple[StateBankRuntimeState, ...]:
        return self.state_bank_states

    @property
    def fast_states(self) -> tuple[FastWeightsState, ...]:
        if any(row.fast_weights is None for row in self.rows):
            raise ValueError("batch runtime has no fast state")
        return tuple(row.fast_weights for row in self.rows if row.fast_weights is not None)

    @property
    def optimizer_states(self) -> tuple[OptimizerRuntimeState, ...]:
        if any(row.optimizer is None for row in self.rows):
            raise ValueError("batch runtime has no optimizer state")
        return tuple(row.optimizer for row in self.rows if row.optimizer is not None)

    def with_fast_states(
        self,
        fast_states: Sequence[FastWeightsState],
        optimizer_states: Sequence[OptimizerRuntimeState] | None = None,
    ) -> BatchRuntimeState:
        fast = tuple(fast_states)
        optimizers = self.optimizer_states if optimizer_states is None else tuple(optimizer_states)
        if len(fast) != len(self.rows) or len(optimizers) != len(self.rows):
            raise ValueError("fast/optimizer states must align to batch runtime rows")
        return BatchRuntimeState(
            tuple(
                replace(row, fast_weights=fast_state, optimizer=optimizer_state)
                for row, fast_state, optimizer_state in zip(
                    self.rows,
                    fast,
                    optimizers,
                    strict=True,
                )
            )
        )

    def validate_for(self, owner: RuntimeOwner) -> None:
        if self.owner != owner:
            raise ValueError("runtime rows do not align to the requested owner")


@dataclass(frozen=True, slots=True)
class LifecycleAudit:
    owner: RuntimeOwner
    phase: LifecyclePhase
    observation_count: int
    prefill_count: int
    decode_count: int
    active_operation: str | None


@dataclass(slots=True)
class PrefillLifecycle:
    """Mutable, per-owner capability that can authorize exactly one prefill.

    This object is external runtime state.  It is intentionally not an ``nn.Module``
    parameter/buffer and must never be placed in a model checkpoint.
    """

    owner: RuntimeOwner
    phase: LifecyclePhase = LifecyclePhase.READY
    observation_count: int = 0
    prefill_count: int = 0
    decode_count: int = 0
    _active_operation: str | None = field(default=None, init=False, repr=False)
    _runtime_state: object | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def audit(self) -> LifecycleAudit:
        with self._lock:
            return LifecycleAudit(
                owner=self.owner,
                phase=self.phase,
                observation_count=self.observation_count,
                prefill_count=self.prefill_count,
                decode_count=self.decode_count,
                active_operation=self._active_operation,
            )

    def runtime_state(self) -> object | None:
        with self._lock:
            return self._runtime_state

    def _validate_observation_ready(self, owner: RuntimeOwner) -> None:
        """Fail before expensive soft work without claiming the observe capability."""

        with self._lock:
            if owner != self.owner:
                raise LifecycleError("request owner does not match the prefill lifecycle")
            if self.phase is LifecyclePhase.FAILED:
                raise LifecycleError("failed lifecycle must be reset before reuse")
            if self._active_operation is not None:
                raise LifecycleError("prefill lifecycle operations are not re-entrant")
            if self.phase is not LifecyclePhase.READY or self.prefill_count:
                raise LifecycleError("observation is forbidden after prefill")

    def _begin(self, operation: str, owner: RuntimeOwner) -> None:
        with self._lock:
            if owner != self.owner:
                raise LifecycleError("request owner does not match the prefill lifecycle")
            if self.phase is LifecyclePhase.FAILED:
                raise LifecycleError("failed lifecycle must be reset before reuse")
            if self._active_operation is not None:
                raise LifecycleError("prefill lifecycle operations are not re-entrant")
            if operation == "observe":
                if self.phase is not LifecyclePhase.READY or self.prefill_count:
                    raise LifecycleError("observation is forbidden after prefill")
            elif operation == "prefill":
                if self.phase is not LifecyclePhase.READY or self.prefill_count:
                    raise LifecycleError("Qwen prefill may be built exactly once")
            elif operation == "decode":
                if self.phase not in (LifecyclePhase.PREFILLED, LifecyclePhase.DECODING):
                    raise LifecycleError("decode requires one successful prefill")
            else:  # pragma: no cover - private caller invariant
                raise ValueError(f"unknown lifecycle operation: {operation}")
            self._active_operation = operation

    def _succeed(self, operation: str, runtime_state: object | None = None) -> None:
        with self._lock:
            if self._active_operation != operation:
                raise LifecycleError("lifecycle completion does not match the active operation")
            if operation == "observe":
                self.observation_count += 1
                self._runtime_state = runtime_state
            elif operation == "prefill":
                self.prefill_count += 1
                self.phase = LifecyclePhase.PREFILLED
                self._runtime_state = runtime_state
            else:
                self.decode_count += 1
                self.phase = LifecyclePhase.DECODING
            self._active_operation = None

    def _fail(self, operation: str) -> None:
        with self._lock:
            if self._active_operation == operation:
                self._active_operation = None
            self.phase = LifecyclePhase.FAILED


@dataclass(frozen=True, slots=True)
class ModelFeatureFlags:
    fast_enabled: bool = True
    bank_enabled: bool = True
    reader_enabled: bool = True
    state_tokens_enabled: bool = True

    def __post_init__(self) -> None:
        values = (
            self.fast_enabled,
            self.bank_enabled,
            self.reader_enabled,
            self.state_tokens_enabled,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("model feature flags must be bool")
        if self.reader_enabled and not self.bank_enabled:
            raise ValueError("Reader requires the Structured State Bank")
        if self.state_tokens_enabled and not self.bank_enabled:
            raise ValueError("State Tokens require the Structured State Bank")


@dataclass(frozen=True, slots=True)
class ObservationChunkRequest:
    owner: RuntimeOwner
    video_input: object
    query_input: RuntimeQueryInput
    runtime_state: BatchRuntimeState
    bank_states: tuple[StateBankRuntimeState, ...]
    inference: bool = True

    def __post_init__(self) -> None:
        if type(self.inference) is not bool:
            raise TypeError("observation inference flag must be bool")
        if self.bank_states and len(self.bank_states) != len(self.owner.video_ids):
            raise ValueError("bank_states must align to the owner batch")


@dataclass(frozen=True, slots=True)
class VisualStageOutput:
    """Adapter-owned visual payload and its single-use Qwen continuation capability."""

    value: object
    prepared_video_features: object
    audit: object | None = None


@dataclass(frozen=True, slots=True)
class BankWriteOutput:
    runtime_state: BatchRuntimeState
    bank_states: tuple[StateBankRuntimeState, ...]
    audit: object
    soft_write: object | None = None


@dataclass(frozen=True, slots=True)
class SoftIntermediates:
    adapted_visual: object
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput | None
    temporal: TemporalEncoderOutput | None
    observations: ObservationOutputs | None
    state_write: object | None = None


@dataclass(slots=True)
class ObservationCommitGuard:
    """Single-use capability preventing checkpoint recompute from repeating a hard write."""

    owner: RuntimeOwner
    committed: bool = False
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def claim(self, owner: RuntimeOwner) -> None:
        with self._lock:
            if owner != self.owner:
                raise LifecycleError("soft observation commit owner changed")
            if self.committed:
                raise LifecycleError("soft observation hard state was already committed")
            self.committed = True


@dataclass(frozen=True, slots=True)
class SoftObservationChunkOutput:
    """Checkpoint-safe soft path with no Bank/FSM mutation."""

    owner: RuntimeOwner
    request_identity: int
    visual: VisualStageOutput
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput | None
    temporal: TemporalEncoderOutput | None
    observations: ObservationOutputs | None
    commit_guard: ObservationCommitGuard


@dataclass(frozen=True, slots=True)
class ObservationChunkOutput:
    owner: RuntimeOwner
    visual: VisualStageOutput
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput | None
    temporal: TemporalEncoderOutput | None
    observations: ObservationOutputs | None
    runtime_state: BatchRuntimeState
    bank_states: tuple[StateBankRuntimeState, ...]
    state_audit: object | None
    soft_intermediates: SoftIntermediates
    lifecycle: LifecycleAudit


@dataclass(frozen=True, slots=True)
class AnswerQueryRequest:
    owner: RuntimeOwner
    observation: ObservationChunkOutput
    base_input_ids: Tensor
    base_attention_mask: Tensor
    pixel_values_videos: Tensor | None
    video_grid_thw: Tensor | None
    tokenizer: object
    embedding_owner: object
    rope_indexer: object
    qwen_kwargs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.observation.owner != self.owner:
            raise ValueError("answer request and observation owners must match")
        names = tuple(name for name, _ in self.qwen_kwargs)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("qwen_kwargs names must be unique and non-empty")
        reserved = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "rope_deltas",
            "pixel_values_videos",
            "video_grid_thw",
            "prepared_video_features",
            "state_embedding_payload",
        }
        overlap = reserved.intersection(names)
        if overlap:
            raise ValueError(f"qwen_kwargs cannot override P13-owned fields: {sorted(overlap)}")


@dataclass(frozen=True, slots=True)
class QwenPrefillRequest:
    """Fields consumed by the P3 adapter for one native-HF prefill.

    ``composer_position_ids_audit`` and ``composer_rope_deltas_audit`` are evidence
    only.  Production Qwen receives IDs/masks/pixels and computes/caches its own
    multimodal positions.  In particular, this request never asks Qwen to consume
    Composer ``inputs_embeds``.
    """

    input_ids: Tensor
    attention_mask: Tensor
    pixel_values_videos: Tensor | None
    video_grid_thw: Tensor | None
    prepared_video_features: object
    state_position_mask: Tensor | None
    state_tokens: Tensor | None
    composer_position_ids_audit: Tensor
    composer_rope_deltas_audit: Tensor
    qwen_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class StateAudit:
    observation: object | None
    retrieval: object | None
    reader: tuple[ReaderResult, ...]
    resampler: StateResamplerOutput | None


@dataclass(frozen=True, slots=True)
class NumberAgreementMetrics:
    """Reader-owned integer agreement, computed independently of answer quality."""

    comparable_rows: int
    matched_rows: int
    mismatched_rows: int
    missing_rows: int

    def __post_init__(self) -> None:
        values = (
            self.comparable_rows,
            self.matched_rows,
            self.mismatched_rows,
            self.missing_rows,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("number-agreement counts must be non-negative integers")
        if self.matched_rows + self.mismatched_rows + self.missing_rows != self.comparable_rows:
            raise ValueError("number-agreement row counts must add up")

    @property
    def accuracy(self) -> float | None:
        return None if self.comparable_rows == 0 else self.matched_rows / self.comparable_rows


@dataclass(frozen=True, slots=True)
class StateTTTModelOutput:
    answer_logits: Tensor
    qwen_output: QwenPrefillOutput
    visual: VisualStageOutput
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput | None
    temporal: TemporalEncoderOutput | None
    observations: ObservationOutputs | None
    retrieval: RetrieverOutput | None
    reader: tuple[ReaderResult, ...]
    resampler: StateResamplerOutput | None
    composed: ComposedInput
    prefill_request: QwenPrefillRequest
    runtime_state: BatchRuntimeState
    state_audit: StateAudit
    soft_intermediates: SoftIntermediates
    lifecycle: LifecycleAudit


@dataclass(frozen=True, slots=True)
class DecodeStepRequest:
    owner: RuntimeOwner
    model_inputs: object


@dataclass(frozen=True, slots=True)
class DecodeStepOutput:
    qwen_output: object
    runtime_state: object
    lifecycle: LifecycleAudit


class VisualStage(Protocol):
    def __call__(self, request: ObservationChunkRequest) -> VisualStageOutput: ...


class QueryStage(Protocol):
    def __call__(
        self, query_input: RuntimeQueryInput, *, inference: bool
    ) -> QueryEncoderOutput: ...


class FastStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput: ...


class SpatialStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> SpatialEncoderOutput: ...


class TemporalStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput: ...


class ObservationStage(Protocol):
    def __call__(
        self,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> ObservationOutputs: ...


class BankWriter(Protocol):
    def __call__(
        self,
        observations: ObservationOutputs,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput: ...


class RetrieverStage(Protocol):
    def retrieve_query(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput: ...


class ReaderStage(Protocol):
    def read(self, retrieval: RetrieverOutput) -> Sequence[ReaderResult]: ...

    def audit_results(
        self,
        retrieval: RetrieverOutput,
        results: Sequence[ReaderResult],
    ) -> Sequence[ReaderResult]: ...

    def audit_number_tokens(self, result: ReaderResult) -> int | None: ...


class ResamplerStage(Protocol):
    def __call__(self, q_target: Tensor, retrieval: RetrieverOutput) -> StateResamplerOutput: ...


class ComposerStage(Protocol):
    def __call__(
        self,
        *,
        base_input_ids: Tensor,
        base_attention_mask: Tensor,
        state_tokens: Tensor | None,
        state_token_valid_mask: Tensor | None,
        reader_results: Sequence[ReaderResult],
        tokenizer: object,
        embedding_owner: object,
        rope_indexer: object,
        video_grid_thw: Tensor | None,
        include_state: bool,
        include_number: bool,
    ) -> ComposedInput: ...


class QwenPrefillStage(Protocol):
    def __call__(self, request: QwenPrefillRequest) -> QwenPrefillOutput: ...


class QwenDecodeStage(Protocol):
    def __call__(self, model_inputs: object) -> object: ...


class QwenPrefillOutput(Protocol):
    logits: Tensor


@dataclass(frozen=True, slots=True)
class ModelComponents:
    visual_stage: VisualStage
    query_encoder: QueryStage
    composer: ComposerStage
    qwen_prefill: QwenPrefillStage
    qwen_decode: QwenDecodeStage
    fast_adapter: FastStage | None = None
    spatial_encoder: SpatialStage | None = None
    temporal_encoder: TemporalStage | None = None
    observation_heads: ObservationStage | None = None
    state_bank: StructuredStateBank | None = None
    bank_writer: BankWriter | None = None
    retriever: RetrieverStage | None = None
    reader: ReaderStage | None = None
    resampler: ResamplerStage | None = None

    def validate(self, flags: ModelFeatureFlags) -> None:
        always = {
            "visual_stage": self.visual_stage,
            "query_encoder": self.query_encoder,
            "composer": self.composer,
            "qwen_prefill": self.qwen_prefill,
            "qwen_decode": self.qwen_decode,
        }
        missing = [name for name, value in always.items() if not callable(value)]
        if flags.fast_enabled and self.fast_adapter is None:
            missing.append("fast_adapter")
        if flags.bank_enabled:
            bank_dependencies = {
                "spatial_encoder": self.spatial_encoder,
                "temporal_encoder": self.temporal_encoder,
                "observation_heads": self.observation_heads,
                "bank_writer": self.bank_writer,
                "state_bank": self.state_bank,
            }
            missing.extend(
                name
                for name, value in bank_dependencies.items()
                if value is None
            )
        if (flags.reader_enabled or flags.state_tokens_enabled) and self.retriever is None:
            missing.append("retriever")
        if flags.reader_enabled and self.reader is None:
            missing.append("reader")
        if flags.state_tokens_enabled and self.resampler is None:
            missing.append("resampler")
        if missing:
            raise ValueError(
                "enabled model features have missing dependencies: "
                + ", ".join(dict.fromkeys(missing))
            )

    def require_fast_adapter(self) -> FastStage:
        assert self.fast_adapter is not None
        return self.fast_adapter

    def require_spatial_encoder(self) -> SpatialStage:
        assert self.spatial_encoder is not None
        return self.spatial_encoder

    def require_temporal_encoder(self) -> TemporalStage:
        assert self.temporal_encoder is not None
        return self.temporal_encoder

    def require_observation_heads(self) -> ObservationStage:
        assert self.observation_heads is not None
        return self.observation_heads

    def require_bank_writer(self) -> BankWriter:
        assert self.bank_writer is not None
        return self.bank_writer

    def require_retriever(self) -> RetrieverStage:
        assert self.retriever is not None
        return self.retriever

    def require_reader(self) -> ReaderStage:
        assert self.reader is not None
        return self.reader

    def require_resampler(self) -> ResamplerStage:
        assert self.resampler is not None
        return self.resampler

    def require_state_bank(self) -> StructuredStateBank:
        assert self.state_bank is not None
        return self.state_bank


class StateTTTModel(nn.Module):  # type: ignore[misc]
    """Dependency-injected P13 orchestrator with no numerical implementation."""

    def __init__(
        self,
        config: ProjectConfig,
        components: ModelComponents,
        feature_flags: ModelFeatureFlags,
    ) -> None:
        super().__init__()
        if not isinstance(config, ProjectConfig):
            raise TypeError("StateTTTModel requires a validated ProjectConfig")
        components.validate(feature_flags)
        self.config = config
        self.components = components
        self.feature_flags = feature_flags
        self.component_modules = nn.ModuleDict()
        seen_modules: set[int] = set()
        for name, value in _component_items(components):
            module = _component_module(value)
            if module is not None and id(module) not in seen_modules:
                self.component_modules[name] = module
                seen_modules.add(id(module))

    def observe_chunk(
        self,
        request: ObservationChunkRequest,
        lifecycle: PrefillLifecycle,
    ) -> ObservationChunkOutput:
        """Compose the checkpoint-safe soft path with exactly one hard commit."""

        lifecycle._validate_observation_ready(request.owner)
        soft = self.observe_chunk_soft(request)
        return self.commit_observation(request, soft, lifecycle)

    def observe_chunk_soft(
        self,
        request: ObservationChunkRequest,
    ) -> SoftObservationChunkOutput:
        """Run only differentiable observation stages; safe to recompute for checkpointing."""

        visual = self.components.visual_stage(request)
        if not isinstance(visual, VisualStageOutput):
            raise TypeError("visual_stage must return VisualStageOutput")
        query = self.components.query_encoder(request.query_input, inference=request.inference)
        adapted = visual
        if self.feature_flags.fast_enabled:
            fast_adapter = self.components.require_fast_adapter()
            adapted = fast_adapter(visual, query, request)
            if not isinstance(adapted, VisualStageOutput):
                raise TypeError("fast_adapter must return VisualStageOutput")

        spatial: SpatialEncoderOutput | None = None
        temporal: TemporalEncoderOutput | None = None
        observations: ObservationOutputs | None = None
        if self.feature_flags.bank_enabled:
            spatial_encoder = self.components.require_spatial_encoder()
            temporal_encoder = self.components.require_temporal_encoder()
            heads = self.components.require_observation_heads()
            spatial = spatial_encoder(adapted, query, request)
            temporal = temporal_encoder(adapted, query, request)
            observations = heads(spatial, temporal, query, request)
        return SoftObservationChunkOutput(
            owner=request.owner,
            request_identity=id(request),
            visual=adapted,
            query=query,
            spatial=spatial,
            temporal=temporal,
            observations=observations,
            commit_guard=ObservationCommitGuard(request.owner),
        )

    def commit_observation(
        self,
        request: ObservationChunkRequest,
        soft: SoftObservationChunkOutput,
        lifecycle: PrefillLifecycle,
    ) -> ObservationChunkOutput:
        """Consume one soft result and execute the sole hard Bank/FSM write."""

        if not isinstance(soft, SoftObservationChunkOutput):
            raise TypeError("hard observation commit requires SoftObservationChunkOutput")
        if soft.owner != request.owner or soft.request_identity != id(request):
            raise LifecycleError("soft observation must commit with its exact originating request")
        lifecycle._begin("observe", request.owner)
        try:
            soft.commit_guard.claim(request.owner)
            runtime_state = request.runtime_state
            bank_states = request.bank_states
            bank_audit: object | None = None
            soft_write: object | None = None
            if self.feature_flags.bank_enabled:
                writer = self.components.require_bank_writer()
                assert soft.observations is not None
                assert soft.spatial is not None
                assert soft.temporal is not None
                write = writer(
                    soft.observations,
                    soft.spatial,
                    soft.temporal,
                    soft.query,
                    request,
                )
                if not isinstance(write, BankWriteOutput):
                    raise TypeError("bank_writer must return BankWriteOutput")
                if len(write.bank_states) != len(request.owner.video_ids):
                    raise ValueError("Bank writer output must align to the owner batch")
                runtime_state = write.runtime_state
                bank_states = write.bank_states
                bank_audit = write.audit
                soft_write = write.soft_write

            lifecycle._succeed("observe", runtime_state)
            return ObservationChunkOutput(
                owner=request.owner,
                visual=soft.visual,
                query=soft.query,
                spatial=soft.spatial,
                temporal=soft.temporal,
                observations=soft.observations,
                runtime_state=runtime_state,
                bank_states=bank_states,
                state_audit=bank_audit,
                soft_intermediates=SoftIntermediates(
                    adapted_visual=soft.visual.value,
                    query=soft.query,
                    spatial=soft.spatial,
                    temporal=soft.temporal,
                    observations=soft.observations,
                    state_write=soft_write,
                ),
                lifecycle=lifecycle.audit(),
            )
        except Exception:
            lifecycle._fail("observe")
            raise

    def answer_query(
        self,
        request: AnswerQueryRequest,
        lifecycle: PrefillLifecycle,
    ) -> StateTTTModelOutput:
        """Audit one Bank snapshot, compose once, and execute one Qwen prefill."""

        lifecycle._begin("prefill", request.owner)
        try:
            observation = request.observation
            retrieval: RetrieverOutput | None = None
            reader_results: tuple[ReaderResult, ...] = ()
            resampler_output: StateResamplerOutput | None = None
            if self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled:
                retriever = self.components.require_retriever()
                retrieval = retriever.retrieve_query(
                    self.components.require_state_bank(),
                    observation.bank_states,
                    observation.query,
                    video_ids=request.owner.video_ids,
                    trajectory_ids=request.owner.trajectory_ids,
                )

            if self.feature_flags.reader_enabled:
                reader = self.components.require_reader()
                assert retrieval is not None
                computed = tuple(reader.read(retrieval))
                audited = tuple(reader.audit_results(retrieval, computed))
                if audited != computed:
                    raise ValueError("Reader audit must return the unchanged authoritative results")
                for result in audited:
                    reader.audit_number_tokens(result)
                reader_results = audited

            if self.feature_flags.state_tokens_enabled:
                q_target = observation.query.q_target
                resampler = self.components.require_resampler()
                assert retrieval is not None
                resampler_output = resampler(q_target, retrieval)
                _validate_answer_provenance(retrieval, reader_results, resampler_output)

            state_tokens = None if resampler_output is None else resampler_output.state_tokens
            state_token_valid_mask = (
                None if resampler_output is None else resampler_output.state_token_valid_mask
            )
            composed = self.components.composer(
                base_input_ids=request.base_input_ids,
                base_attention_mask=request.base_attention_mask,
                state_tokens=state_tokens,
                state_token_valid_mask=state_token_valid_mask,
                reader_results=reader_results,
                tokenizer=request.tokenizer,
                embedding_owner=request.embedding_owner,
                rope_indexer=request.rope_indexer,
                video_grid_thw=request.video_grid_thw,
                include_state=self.feature_flags.state_tokens_enabled,
                include_number=self.feature_flags.reader_enabled,
            )
            prefill_request = QwenPrefillRequest(
                input_ids=composed.input_ids,
                attention_mask=composed.attention_mask,
                pixel_values_videos=request.pixel_values_videos,
                video_grid_thw=request.video_grid_thw,
                prepared_video_features=observation.visual.prepared_video_features,
                state_position_mask=composed.state_position_mask,
                state_tokens=state_tokens,
                composer_position_ids_audit=composed.position_ids,
                composer_rope_deltas_audit=composed.rope_deltas,
                qwen_kwargs=request.qwen_kwargs,
            )
            qwen_output = self.components.qwen_prefill(prefill_request)
            answer_logits = qwen_output.logits
            lifecycle._succeed("prefill", observation.runtime_state)
            return StateTTTModelOutput(
                answer_logits=answer_logits,
                qwen_output=qwen_output,
                visual=observation.visual,
                query=observation.query,
                spatial=observation.spatial,
                temporal=observation.temporal,
                observations=observation.observations,
                retrieval=retrieval,
                reader=reader_results,
                resampler=resampler_output,
                composed=composed,
                prefill_request=prefill_request,
                runtime_state=observation.runtime_state,
                state_audit=StateAudit(
                    observation=observation.state_audit,
                    retrieval=None if retrieval is None else retrieval.audit,
                    reader=reader_results,
                    resampler=resampler_output,
                ),
                soft_intermediates=observation.soft_intermediates,
                lifecycle=lifecycle.audit(),
            )
        except Exception:
            lifecycle._fail("prefill")
            raise

    def decode_step(
        self,
        request: DecodeStepRequest,
        lifecycle: PrefillLifecycle,
    ) -> DecodeStepOutput:
        """Run Qwen decode only; no state-writing dependency is reachable here."""

        lifecycle._begin("decode", request.owner)
        runtime_before = lifecycle.runtime_state()
        try:
            output = self.components.qwen_decode(request.model_inputs)
            if lifecycle.runtime_state() is not runtime_before:
                raise LifecycleError("decode cannot replace the authoritative runtime state")
            lifecycle._succeed("decode")
            return DecodeStepOutput(
                qwen_output=output,
                runtime_state=runtime_before,
                lifecycle=lifecycle.audit(),
            )
        except Exception:
            lifecycle._fail("decode")
            raise


def build_model(
    config: ProjectConfig | None = None,
    *,
    components: ModelComponents | None = None,
    feature_flags: ModelFeatureFlags | None = None,
) -> StateTTTModel:
    """Build the P13 composition container from explicit dependencies."""

    if config is None:
        raise ValueError("build_model requires a validated ProjectConfig")
    if components is None:
        raise ValueError("build_model requires explicit ModelComponents")
    return StateTTTModel(config, components, feature_flags or ModelFeatureFlags())


def evaluate_number_agreement(
    reader_results: Sequence[ReaderResult],
    predicted_numbers: Sequence[int | None],
) -> NumberAgreementMetrics:
    """Compare externally parsed answer integers without changing Reader results."""

    results = tuple(reader_results)
    predictions = tuple(predicted_numbers)
    if len(results) != len(predictions):
        raise ValueError("Reader results and predicted numbers must have equal batch size")
    matched = mismatched = missing = comparable = 0
    for result, predicted in zip(results, predictions, strict=True):
        exact = result.exact_count
        if exact is None:
            if predicted is not None and type(predicted) is not int:
                raise TypeError("predicted numbers must contain int or None")
            continue
        if type(exact) is not int:
            raise TypeError("Reader exact_count must be int or None")
        comparable += 1
        if predicted is None:
            missing += 1
        elif type(predicted) is not int:
            raise TypeError("predicted numbers must contain int or None")
        elif predicted == exact:
            matched += 1
        else:
            mismatched += 1
    return NumberAgreementMetrics(comparable, matched, mismatched, missing)


def assert_training_number_agreement(
    reader_results: Sequence[ReaderResult],
    target_numbers: Sequence[int | None],
) -> None:
    """Block final-expression supervision whose integer target disagrees with Reader."""

    metrics = evaluate_number_agreement(reader_results, target_numbers)
    if metrics.mismatched_rows or metrics.missing_rows:
        raise ValueError("answer supervision number must equal the authoritative Reader number")


def _validate_answer_provenance(
    retrieval: RetrieverOutput,
    reader_results: tuple[ReaderResult, ...],
    resampler: StateResamplerOutput,
) -> None:
    """Check IDs/status provenance only; arithmetic remains wholly Reader-owned."""

    retrieval_ids = retrieval.selected_record_ids
    resampler_ids = resampler.selected_record_ids
    if retrieval_ids != resampler_ids:
        raise ValueError("Resampler must consume the same Retriever selected-record snapshot")
    if reader_results:
        reader_ids = tuple(result.selected_record_ids for result in reader_results)
        if reader_ids != retrieval_ids:
            raise ValueError("Reader results must preserve Retriever selected-record IDs")
    if retrieval.status != resampler.retrieval_status:
        raise ValueError("Resampler must preserve Retriever row statuses")


def _component_items(components: ModelComponents) -> tuple[tuple[str, object | None], ...]:
    return (
        ("visual_stage", components.visual_stage),
        ("query_encoder", components.query_encoder),
        ("composer", components.composer),
        ("qwen_prefill", components.qwen_prefill),
        ("qwen_decode", components.qwen_decode),
        ("fast_adapter", components.fast_adapter),
        ("spatial_encoder", components.spatial_encoder),
        ("temporal_encoder", components.temporal_encoder),
        ("observation_heads", components.observation_heads),
        ("state_bank", components.state_bank),
        ("bank_writer", components.bank_writer),
        ("retriever", components.retriever),
        ("reader", components.reader),
        ("resampler", components.resampler),
    )


def _component_module(value: object | None) -> nn.Module | None:
    if isinstance(value, nn.Module):
        return value
    owner = getattr(value, "__self__", None)
    return owner if isinstance(owner, nn.Module) else None
