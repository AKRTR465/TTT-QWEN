"""Compose the P13 State-TTT stages without owning their algorithms.

Inputs: injected P3/P5-P12 components, immutable stage requests, and one explicit
per-owner prefill lifecycle.
Outputs: observation intermediates, one audited training prefill, or one generated answer.
Forbidden: local Adapter/SGD, FSM/Bank mutation, Retriever, Reader, Resampler,
Composer, or Qwen masking implementations.

The deliberately small protocols in this module are orchestration seams.  Thin
adapters may translate them to the existing component signatures, while the
authoritative component implementations continue to own every numerical or hard
state rule.
"""

from __future__ import annotations

import hashlib
import math
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
from ttt_svcbench_qwen.query_encoder import (
    QueryEncoderOutput,
    detach_query_encoder_output,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.state_bank import (
    RetrievalHistoryView,
    StateBankRuntimeState,
    StructuredStateBank,
    TensorizedRetrievalHistory,
    tensorized_retrieval_view,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
    TemporalEncoderOutput,
)
from ttt_svcbench_qwen.state_reader import ReaderResult, StateResamplerOutput
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.tensor_contracts import validate_finite_tensor_tree


class LifecycleError(RuntimeError):
    """Raised when an owner attempts an illegal observe/prefill/generate transition."""


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


@dataclass(frozen=True, slots=True)
class OnlineOverlapSnapshot:
    """The sole detached adjacent-chunk memory used by training and inference."""

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

    def __post_init__(self) -> None:
        if not math.isfinite(self.end_time) or self.end_time < 0.0:
            raise ValueError("overlap snapshot end_time must be finite and non-negative")
        tensors = self.tensors
        if any(value.requires_grad or value.grad_fn is not None for value in tensors):
            raise ValueError("overlap snapshots must be detached from autograd")
        materialized = tuple(value for value in tensors if value.device.type != "meta")
        storage = tuple(
            (str(value.device), value.untyped_storage().data_ptr()) for value in materialized
        )
        if len(set(storage)) != len(storage):
            raise ValueError("overlap snapshot tensors must use isolated storage")

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.identity,
            self.identity_valid_mask,
            self.identity_position_ids,
            self.identity_timestamps,
            self.e1_probabilities,
            self.e2_event_probabilities,
            self.e2_phase_probabilities,
            self.event_valid_mask,
            self.event_position_ids,
            self.event_timestamps,
        )

    @classmethod
    def capture(
        cls,
        output: ObservationChunkOutput,
        *,
        end_time: float,
    ) -> OnlineOverlapSnapshot:
        observations = output.observations
        if not isinstance(observations, ObservationOutputs):
            raise TypeError("overlap snapshots require typed ObservationOutputs")
        return cls(
            owner=output.owner,
            end_time=end_time,
            identity=observations.o2.identity.detach().clone(),
            identity_valid_mask=observations.o2.valid_mask.detach().clone(),
            identity_position_ids=observations.o2.position_ids.detach().clone(),
            identity_timestamps=observations.o2.timestamps.detach().clone(),
            e1_probabilities=observations.e1.probabilities.detach().clone(),
            e2_event_probabilities=observations.e2.event_probabilities.detach().clone(),
            e2_phase_probabilities=observations.e2.phase_probabilities.detach().clone(),
            event_valid_mask=observations.e1.valid_mask.detach().clone(),
            event_position_ids=observations.e1.position_ids.detach().clone(),
            event_timestamps=observations.e1.timestamps.detach().clone(),
        )


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
    retrieval_history: TensorizedRetrievalHistory | None = None
    fast_weights: FastWeightsState | None = None
    optimizer: OptimizerRuntimeState | None = None
    reader_audit: tuple[ReaderResult, ...] = ()
    online_overlap_memory: OnlineOverlapSnapshot | None = None
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
        if self.retrieval_history is not None:
            history = self.retrieval_history
            if (history.video_id, history.trajectory_id) != (video_id, trajectory_id):
                raise ValueError("retrieval ring ownership does not match trajectory runtime")
            if history.released != self.released:
                raise ValueError("retrieval ring release state does not match trajectory runtime")
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
    def retrieval_histories(self) -> tuple[TensorizedRetrievalHistory, ...]:
        histories = tuple(row.retrieval_history for row in self.rows)
        if any(value is None for value in histories):
            raise ValueError("batch runtime does not own tensorized retrieval histories")
        return tuple(value for value in histories if value is not None)

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

    def _validate_prefill_ready(self, owner: RuntimeOwner) -> None:
        """Reject repeated preparation before Reader/retrieval work is executed."""

        with self._lock:
            if owner != self.owner:
                raise LifecycleError("request owner does not match the prefill lifecycle")
            if self.phase is LifecyclePhase.FAILED:
                raise LifecycleError("failed lifecycle must be reset before reuse")
            if self._active_operation is not None:
                raise LifecycleError("prefill lifecycle operations are not re-entrant")
            if self.phase is not LifecyclePhase.READY or self.prefill_count:
                raise LifecycleError("Qwen prefill may be built exactly once")

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
class PreparedQueryOutput:
    """One explicitly scoped Query graph; never stored beyond its caller-owned episode."""

    key: tuple[int, str, str]
    value: QueryEncoderOutput

    @classmethod
    def bind(cls, query: RuntimeQueryInput, value: QueryEncoderOutput) -> PreparedQueryOutput:
        return cls(query_reuse_key(query), value)

    def validate_for(self, query: RuntimeQueryInput) -> None:
        if self.key != query_reuse_key(query):
            raise ValueError("prepared Query key does not match the observation request")

    def detached(self) -> PreparedQueryOutput:
        return PreparedQueryOutput(self.key, detach_query_encoder_output(self.value))


def query_reuse_key(query: RuntimeQueryInput) -> tuple[int, str, str]:
    return (query.episode_nonce, query.query_id, query.question)


def query_dropout_seed(query: RuntimeQueryInput) -> int:
    # Preserve the production dropout contract that predates Query graph reuse.
    encoded = f"{query.episode_nonce}:{query.query_id}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True, slots=True)
class ObservationChunkRequest:
    owner: RuntimeOwner
    video_input: object
    query_input: RuntimeQueryInput
    runtime_state: BatchRuntimeState
    bank_states: tuple[StateBankRuntimeState, ...]
    inference: bool = True
    retrieval_snapshot_required: bool = True
    retrieval_history_write_enabled: bool = True
    prepared_query: PreparedQueryOutput | None = None

    def __post_init__(self) -> None:
        if type(self.inference) is not bool:
            raise TypeError("observation inference flag must be bool")
        if type(self.retrieval_snapshot_required) is not bool:
            raise TypeError("retrieval_snapshot_required must be bool")
        if type(self.retrieval_history_write_enabled) is not bool:
            raise TypeError("retrieval_history_write_enabled must be bool")
        if self.bank_states and len(self.bank_states) != len(self.owner.video_ids):
            raise ValueError("bank_states must align to the owner batch")
        if self.prepared_query is not None:
            self.prepared_query.validate_for(self.query_input)


@dataclass(frozen=True, slots=True)
class VisualStageOutput:
    """Adapter-owned visual payload consumed only by the State observation path."""

    value: object
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
    retrieval_history: RetrievalHistoryView | None
    state_audit: object | None
    soft_intermediates: SoftIntermediates
    lifecycle: LifecycleAudit


@dataclass(frozen=True, slots=True)
class AnswerQueryRequest:
    owner: RuntimeOwner
    observation: ObservationChunkOutput
    base_input_ids: Tensor
    base_attention_mask: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    tokenizer: object
    embedding_owner: object
    rope_indexer: object
    qwen_kwargs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.observation.owner != self.owner:
            raise ValueError("answer request and observation owners must match")
        if self.pixel_values_videos.ndim != 2 or not self.pixel_values_videos.is_floating_point():
            raise ValueError("Answer Query pixels must be packed floating [sum(N_patch), D]")
        if self.video_grid_thw.ndim != 2 or self.video_grid_thw.shape[1] != 3:
            raise ValueError("Answer Query video_grid_thw must be [B, 3]")
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
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    state_position_mask: Tensor | None
    state_tokens: Tensor | None
    composer_position_ids_audit: Tensor
    composer_rope_deltas_audit: Tensor
    qwen_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class QwenGenerateRequest:
    prefill: QwenPrefillRequest
    max_new_tokens: int = 16

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class QwenGenerateOutput:
    answer_text: str
    token_ids: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str):
            raise TypeError("Qwen generation answer must be text")
        if self.token_ids.ndim != 2 or self.token_ids.shape[0] != 1:
            raise ValueError("Qwen generated token IDs must be [1, T]")


@dataclass(frozen=True, slots=True)
class StateAudit:
    observation: object | None
    retrieval: object | None
    reader: tuple[ReaderResult, ...]
    resampler: StateResamplerOutput | None


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
class PreparedAnswer:
    request: AnswerQueryRequest
    retrieval: RetrieverOutput | None
    reader: tuple[ReaderResult, ...]
    resampler: StateResamplerOutput | None
    composed: ComposedInput
    qwen_request: QwenPrefillRequest
    state_audit: StateAudit


@dataclass(frozen=True, slots=True)
class StateTTTGenerationOutput:
    answer_text: str
    generated_token_ids: Tensor
    reader: tuple[ReaderResult, ...]
    resampler: StateResamplerOutput | None
    runtime_state: BatchRuntimeState
    state_audit: StateAudit
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
    def __call__(
        self,
        state_bank: StructuredStateBank,
        history: RetrievalHistoryView,
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput: ...


class ReaderStage(Protocol):
    def read(self, retrieval: RetrieverOutput) -> Sequence[ReaderResult]: ...

    def read_bank(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> Sequence[ReaderResult]: ...

    def audit_results(
        self,
        retrieval: RetrieverOutput,
        results: Sequence[ReaderResult],
    ) -> Sequence[ReaderResult]: ...

    def audit_bank_results(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        results: Sequence[ReaderResult],
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
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


class QwenGenerateStage(Protocol):
    def __call__(self, request: QwenGenerateRequest) -> QwenGenerateOutput: ...


class QwenPrefillOutput(Protocol):
    logits: Tensor


@dataclass(frozen=True, slots=True)
class ModelComponents:
    visual_stage: VisualStage
    query_encoder: QueryStage
    composer: ComposerStage
    qwen_prefill: QwenPrefillStage
    qwen_generate: QwenGenerateStage
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
            "qwen_generate": self.qwen_generate,
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
            missing.extend(name for name, value in bank_dependencies.items() if value is None)
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
        query = (
            request.prepared_query.value
            if request.prepared_query is not None
            else self.components.query_encoder(request.query_input, inference=request.inference)
        )
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
            validate_finite_tensor_tree(soft, "soft observation commit")
            soft.commit_guard.claim(request.owner)
            runtime_state = request.runtime_state
            bank_states = request.bank_states
            retrieval_history: RetrievalHistoryView | None = None
            bank_audit: object | None = None
            soft_write: object | None = None
            if self.feature_flags.bank_enabled:
                if request.retrieval_snapshot_required and (
                    self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled
                ):
                    # Keep the write-before snapshot label-free and expose every head. Runtime
                    # selection is still constrained by the predicted hard operator inside the
                    # Retriever; official labels may only build a target-head MIL bag later.
                    if not isinstance(request.runtime_state, BatchRuntimeState):
                        raise TypeError("tensor retrieval backend requires BatchRuntimeState")
                    with trace_cuda_phase("retrieval_history_snapshot"):
                        retrieval_history = tensorized_retrieval_view(
                            request.runtime_state.retrieval_histories,
                            # A normal online Support observation intentionally commits a
                            # hard write after taking its pre-write view. The gathered view
                            # owns tensor copies, so it remains immutable. Query snapshots
                            # never write and retain the strict version guard.
                            guard_current_version=(
                                not request.retrieval_history_write_enabled
                            ),
                        )
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
                retrieval_history=retrieval_history,
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

    def prepare_answer(
        self,
        request: AnswerQueryRequest,
        lifecycle: PrefillLifecycle,
    ) -> PreparedAnswer:
        """Run Reader, retrieval, resampling and composition exactly once."""

        lifecycle._validate_prefill_ready(request.owner)
        observation = request.observation
        retrieval: RetrieverOutput | None = None
        reader_results: tuple[ReaderResult, ...] = ()
        resampler_output: StateResamplerOutput | None = None
        if self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled:
            retriever = self.components.require_retriever()
            history = observation.retrieval_history
            if not isinstance(history, RetrievalHistoryView):
                raise TypeError("answer preparation requires a write-before retrieval history")
            retrieval = retriever(
                self.components.require_state_bank(),
                history,
                observation.query,
                video_ids=request.owner.video_ids,
                trajectory_ids=request.owner.trajectory_ids,
            )
        if self.feature_flags.reader_enabled:
            reader = self.components.require_reader()
            state_bank = self.components.require_state_bank()
            computed = tuple(
                reader.read_bank(
                    state_bank,
                    observation.bank_states,
                    observation.query,
                    video_ids=request.owner.video_ids,
                    trajectory_ids=request.owner.trajectory_ids,
                )
            )
            reader_results = tuple(
                reader.audit_bank_results(
                    state_bank,
                    observation.bank_states,
                    observation.query,
                    computed,
                    video_ids=request.owner.video_ids,
                    trajectory_ids=request.owner.trajectory_ids,
                )
            )
            if reader_results != computed:
                raise ValueError("Reader audit must return the unchanged authoritative results")
            for result in reader_results:
                reader.audit_number_tokens(result)
        if self.feature_flags.state_tokens_enabled:
            assert retrieval is not None
            resampler_output = self.components.require_resampler()(
                observation.query.q_target, retrieval
            )
            _validate_answer_provenance(retrieval, resampler_output)
        state_tokens = None if resampler_output is None else resampler_output.state_tokens
        state_valid = None if resampler_output is None else resampler_output.state_token_valid_mask
        composed = self.components.composer(
            base_input_ids=request.base_input_ids,
            base_attention_mask=request.base_attention_mask,
            state_tokens=state_tokens,
            state_token_valid_mask=state_valid,
            reader_results=reader_results,
            tokenizer=request.tokenizer,
            embedding_owner=request.embedding_owner,
            rope_indexer=request.rope_indexer,
            video_grid_thw=request.video_grid_thw,
            include_state=self.feature_flags.state_tokens_enabled,
            include_number=self.feature_flags.reader_enabled,
        )
        qwen_request = QwenPrefillRequest(
            input_ids=composed.input_ids,
            attention_mask=composed.attention_mask,
            pixel_values_videos=request.pixel_values_videos,
            video_grid_thw=request.video_grid_thw,
            state_position_mask=composed.state_position_mask,
            state_tokens=state_tokens,
            composer_position_ids_audit=composed.position_ids,
            composer_rope_deltas_audit=composed.rope_deltas,
            qwen_kwargs=request.qwen_kwargs,
        )
        return PreparedAnswer(
            request=request,
            retrieval=retrieval,
            reader=reader_results,
            resampler=resampler_output,
            composed=composed,
            qwen_request=qwen_request,
            state_audit=StateAudit(
                observation=observation.state_audit,
                retrieval=None if retrieval is None else retrieval.audit,
                reader=reader_results,
                resampler=resampler_output,
            ),
        )

    def prefill_answer(
        self,
        prepared: PreparedAnswer,
        lifecycle: PrefillLifecycle,
    ) -> StateTTTModelOutput:
        """Execute the sole teacher-forced Qwen prefill used by A2/A5 training."""

        request = prepared.request
        lifecycle._begin("prefill", request.owner)
        try:
            qwen_output = self.components.qwen_prefill(prepared.qwen_request)
            lifecycle._succeed("prefill", request.observation.runtime_state)
            observation = request.observation
            return StateTTTModelOutput(
                answer_logits=qwen_output.logits,
                qwen_output=qwen_output,
                visual=observation.visual,
                query=observation.query,
                spatial=observation.spatial,
                temporal=observation.temporal,
                observations=observation.observations,
                retrieval=prepared.retrieval,
                reader=prepared.reader,
                resampler=prepared.resampler,
                composed=prepared.composed,
                prefill_request=prepared.qwen_request,
                runtime_state=observation.runtime_state,
                state_audit=prepared.state_audit,
                soft_intermediates=observation.soft_intermediates,
                lifecycle=lifecycle.audit(),
            )
        except Exception:
            lifecycle._fail("prefill")
            raise

    def generate_answer(
        self,
        prepared: PreparedAnswer,
        lifecycle: PrefillLifecycle,
        *,
        max_new_tokens: int = 16,
    ) -> StateTTTGenerationOutput:
        """Execute one greedy HF generate call; its first pass is the sole Qwen prefill."""

        request = prepared.request
        lifecycle._begin("prefill", request.owner)
        try:
            generated = self.components.qwen_generate(
                QwenGenerateRequest(prepared.qwen_request, max_new_tokens=max_new_tokens)
            )
            lifecycle._succeed("prefill", request.observation.runtime_state)
            return StateTTTGenerationOutput(
                answer_text=generated.answer_text,
                generated_token_ids=generated.token_ids,
                reader=prepared.reader,
                resampler=prepared.resampler,
                runtime_state=request.observation.runtime_state,
                state_audit=prepared.state_audit,
                lifecycle=lifecycle.audit(),
            )
        except Exception:
            lifecycle._fail("prefill")
            raise


def assert_training_number_agreement(
    reader_results: Sequence[ReaderResult],
    target_numbers: Sequence[int | None],
) -> None:
    """Block final-expression supervision whose integer target disagrees with Reader."""

    results = tuple(reader_results)
    targets = tuple(target_numbers)
    if len(results) != len(targets):
        raise ValueError("Reader results and target numbers must have equal batch size")
    for result, target in zip(results, targets, strict=True):
        exact = result.exact_count
        if exact is not None and type(exact) is not int:
            raise TypeError("Reader exact_count must be int or None")
        if target is not None and type(target) is not int:
            raise TypeError("target numbers must contain int or None")
        if exact is not None and target != exact:
            raise ValueError("answer supervision number must equal the authoritative Reader number")


def _validate_answer_provenance(
    retrieval: RetrieverOutput,
    resampler: StateResamplerOutput,
) -> None:
    """Check semantic Retriever/Resampler provenance; Reader owns aggregate state."""

    retrieval_ids = retrieval.selected_record_ids
    resampler_ids = resampler.selected_record_ids
    if retrieval_ids != resampler_ids:
        raise ValueError("Resampler must consume the same Retriever selected-record snapshot")
    if retrieval.status != resampler.retrieval_status:
        raise ValueError("Resampler must preserve Retriever row statuses")


def _component_items(components: ModelComponents) -> tuple[tuple[str, object | None], ...]:
    return (
        ("visual_stage", components.visual_stage),
        ("query_encoder", components.query_encoder),
        ("composer", components.composer),
        ("qwen_prefill", components.qwen_prefill),
        ("qwen_generate", components.qwen_generate),
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
