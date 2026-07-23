"""Run the causal, per-video online State-TTT inference protocol.

Inputs: one label-free runtime payload, timestamped chunks, injected P13 model
components, and one explicit TTT update/generation driver.
Outputs: one Reader-backed answer plus reset/chunk/update/generate/release audits.
Forbidden: training labels, cross-video state, future frames, in-place runtime
mutation, or repeated observe/update work during autoregressive decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum, StrEnum
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import AuditLevel, InnerSGDConfig, ProjectConfig
from ttt_svcbench_qwen.data import (
    RUNTIME_DENYLIST,
    RuntimeQueryInput,
    assert_runtime_payload_safe,
)
from ttt_svcbench_qwen.fast_ttt import (
    FastTTTAdapter,
    FastTTTForwardAudit,
    FastWeightsState,
    OptimizerRuntimeState,
)
from ttt_svcbench_qwen.functional_sgd import (
    functional_sgd_steps_from_ttt,
    reset_optimizer_state,
)
from ttt_svcbench_qwen.identity_bank import IdentityBank
from ttt_svcbench_qwen.losses import TemporalPredictor, compute_ttt_loss
from ttt_svcbench_qwen.meta_trainer import CausalOverlapTTTInputBuilder
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BatchRuntimeState,
    LifecyclePhase,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    RuntimeOwner,
    StateTTTModel,
    TrajectoryRuntimeState,
)
from ttt_svcbench_qwen.state_bank import StructuredStateBank, TensorizedRetrievalHistory
from ttt_svcbench_qwen.state_encoder import TemporalCache
from ttt_svcbench_qwen.state_reader import ReaderResult

type AuditValue = str | int | float | bool | None


class InferenceProtocolError(RuntimeError):
    """Raised when causal ordering or runtime ownership is violated."""


class QueryAttemptKind(StrEnum):
    NEW = "new"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class QueryAttempt:
    query_id: str
    kind: QueryAttemptKind = QueryAttemptKind.NEW
    retry_of: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query attempt requires a non-empty query_id")
        if not isinstance(self.kind, QueryAttemptKind):
            raise TypeError("query attempt kind must be QueryAttemptKind")
        if self.kind is QueryAttemptKind.NEW and self.retry_of is not None:
            raise ValueError("a new query cannot carry retry_of")
        if self.kind is QueryAttemptKind.RETRY and (
            not self.retry_of or self.retry_of == self.query_id
        ):
            raise ValueError("a retry requires a distinct non-empty retry_of query_id")


@dataclass(frozen=True, slots=True)
class CausalChunk:
    """Timestamp-aligned frame payload that can be cropped before model handoff."""

    chunk_id: str
    frames: tuple[object, ...]
    timestamps: tuple[float, ...]
    position_ids: tuple[int, ...]
    model_input: object | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("inference chunk_id must be non-empty")
        if not self.frames or len(self.frames) != len(self.timestamps):
            raise ValueError("inference chunk requires aligned non-empty frames/timestamps")
        if len(self.position_ids) != len(self.timestamps):
            raise ValueError("inference chunk position_ids must align to timestamps")
        if any(not math.isfinite(value) or value < 0.0 for value in self.timestamps):
            raise ValueError("inference chunk timestamps must be finite and non-negative")
        if any(type(value) is not int or value < 0 for value in self.position_ids):
            raise ValueError("inference chunk position_ids must be non-negative integers")
        if any(
            right <= left for left, right in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("inference chunk timestamps must increase strictly")
        if any(
            right != left + 1
            for left, right in zip(self.position_ids, self.position_ids[1:], strict=False)
        ):
            raise ValueError("inference chunk position_ids must be contiguous")

    @property
    def start_time(self) -> float:
        return self.timestamps[0]

    @property
    def end_time(self) -> float:
        return self.timestamps[-1]

    def causal_prefix(self, query_time: float) -> CausalChunk | None:
        """Return only frames at or before query_time; never trust upstream cropping."""

        if not math.isfinite(query_time) or query_time < 0.0:
            raise ValueError("query_time must be finite and non-negative")
        keep = sum(value <= query_time for value in self.timestamps)
        if keep == 0:
            return None
        return CausalChunk(
            chunk_id=self.chunk_id,
            frames=self.frames[:keep],
            timestamps=self.timestamps[:keep],
            position_ids=self.position_ids[:keep],
            model_input=self.model_input,
        )


@dataclass(frozen=True, slots=True)
class AnswerInputs:
    base_input_ids: Tensor
    base_attention_mask: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    tokenizer: object
    embedding_owner: object
    rope_indexer: object
    qwen_kwargs: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    query_input: RuntimeQueryInput
    query_signature: Tensor
    chunks: tuple[CausalChunk, ...]
    answer_inputs: AnswerInputs
    attempt: QueryAttempt
    query_observation: CausalChunk | None = None
    max_new_tokens: int = 16

    def __post_init__(self) -> None:
        if self.query_input.query_id != self.attempt.query_id:
            raise ValueError("inference request Query identity must match the attempt")
        if (
            self.query_signature.shape != (512,)
            or not torch.is_floating_point(self.query_signature)
            or not bool(torch.isfinite(self.query_signature).all())
        ):
            raise ValueError("query_signature must be finite floating [512]")
        if not self.chunks:
            raise ValueError("inference request requires at least one chunk")
        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("inference request chunk IDs must be unique")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

    @classmethod
    def from_payload(
        cls,
        *,
        video_id: str,
        trajectory_id: str,
        payload: Mapping[str, object],
        query_signature: Tensor,
        chunks: tuple[CausalChunk, ...],
        answer_inputs: AnswerInputs,
        attempt: QueryAttempt,
        query_observation: CausalChunk | None = None,
        max_new_tokens: int = 16,
    ) -> InferenceRequest:
        """Validate one JSON boundary and immediately convert it to a typed Query."""

        assert_inference_runtime_payload(payload)
        explicit = payload["explicit_time_values"]
        if not isinstance(explicit, (tuple, list)):
            raise TypeError("explicit_time_values must be a sequence")
        query = RuntimeQueryInput(
            video_id=video_id,
            trajectory_id=trajectory_id,
            query_id=attempt.query_id,
            query_index=0,
            video=Path(str(payload["video"])),
            question=cast(str, payload["question"]),
            query_time=_payload_query_time(payload),
            explicit_time_values=tuple(float(value) for value in explicit),
        )
        return cls(
            query_input=query,
            query_signature=query_signature,
            chunks=chunks,
            answer_inputs=answer_inputs,
            attempt=attempt,
            query_observation=query_observation,
            max_new_tokens=max_new_tokens,
        )

    @property
    def video_id(self) -> str:
        return self.query_input.video_id

    @property
    def trajectory_id(self) -> str:
        return self.query_input.trajectory_id

    @property
    def query_time(self) -> float:
        return self.query_input.query_time


@dataclass(frozen=True, slots=True)
class RuntimeBoundaryStamp:
    """CPU-copy-free lifecycle identity for the authoritative runtime."""

    video_id: str
    trajectory_id: str
    next_chunk_index: int
    released: bool
    fast_version: int
    update_count: int
    skip_count: int
    state_bank_version: int
    identity_bank_version: int
    component_ids: tuple[int, ...]
    tensor_versions: tuple[tuple[str, str, str, tuple[int, ...], int | None, int], ...]


@dataclass(frozen=True, slots=True)
class RuntimeAuditSnapshot:
    """Persisted audit work selected by ``AuditLevel``."""

    level: AuditLevel
    boundary: RuntimeBoundaryStamp | None
    content_sha256: str | None

    def __post_init__(self) -> None:
        if self.level is AuditLevel.OFF and (
            self.boundary is not None or self.content_sha256 is not None
        ):
            raise ValueError("off audit snapshots cannot retain runtime state")
        if self.level is not AuditLevel.OFF and self.boundary is None:
            raise ValueError("enabled audit snapshots require a boundary stamp")
        if self.content_sha256 is not None and len(self.content_sha256) != 64:
            raise ValueError("runtime content hashes must be SHA-256 values")
        if self.level is not AuditLevel.FULL and self.content_sha256 is not None:
            raise ValueError("only full audit snapshots may retain a content hash")


@dataclass(frozen=True, slots=True)
class RuntimePristineStamp:
    """Owner-independent proof that reset produced an empty runtime."""

    fast_shape: tuple[int, ...]
    fast_dtype: str
    fast_version: int
    update_count: int
    skip_count: int
    optimizer_attempted_updates: int
    temporal_width: int
    state_record_count: int
    identity_candidate_count: int
    identity_confirmed_count: int
    reader_audit_count: int
    released: bool


@dataclass(frozen=True, slots=True)
class RuntimeResetAudit:
    video_id: str
    trajectory_id: str
    previous_runtime: RuntimeAuditSnapshot | None
    previous_release: RuntimeAuditSnapshot | None
    reset: RuntimeAuditSnapshot
    pristine: RuntimePristineStamp


@dataclass(frozen=True, slots=True)
class RuntimeReleaseAudit:
    video_id: str
    trajectory_id: str
    before: RuntimeAuditSnapshot
    released: RuntimeAuditSnapshot
    state_bank_version: int
    identity_bank_version: int
    runtime_state: TrajectoryRuntimeState

    def __post_init__(self) -> None:
        if not self.runtime_state.released:
            raise ValueError("release audit must carry a released runtime state")
        if (self.runtime_state.video_id, self.runtime_state.trajectory_id) != (
            self.video_id,
            self.trajectory_id,
        ):
            raise ValueError("release audit runtime owner is inconsistent")


@dataclass(frozen=True, slots=True)
class TTTUpdateOutcome:
    runtime_state: TrajectoryRuntimeState
    did_update: bool
    skip_reason: str | None
    valid_term_count: int
    loss_value: float | None = None

    def __post_init__(self) -> None:
        if type(self.did_update) is not bool:
            raise TypeError("TTT update did_update must be bool")
        if type(self.valid_term_count) is not int or self.valid_term_count < 0:
            raise ValueError("TTT update valid_term_count must be non-negative")
        if self.did_update:
            if self.skip_reason is not None or self.valid_term_count == 0:
                raise ValueError("successful TTT update requires valid terms and no skip reason")
        elif not self.skip_reason:
            raise ValueError("skipped TTT update requires a skip reason")
        if self.loss_value is not None and not math.isfinite(self.loss_value):
            raise ValueError("TTT update loss audit must be finite")


class TTTUpdateStage(Protocol):
    def __call__(
        self,
        observation: ObservationChunkOutput,
        runtime_state: TrajectoryRuntimeState,
        *,
        current_end_time: float,
    ) -> TTTUpdateOutcome: ...


class OnlineTTTUpdater:
    """Apply label-free adjacent-chunk State-TTT and publish W_(t+1)."""

    def __init__(self, config: ProjectConfig, predictor: TemporalPredictor) -> None:
        if not isinstance(config, ProjectConfig):
            raise TypeError("online updater requires validated ProjectConfig")
        if not isinstance(predictor, TemporalPredictor):
            raise TypeError("online updater requires TemporalPredictor")
        self.config = config
        self.predictor = predictor
        self.input_builder = CausalOverlapTTTInputBuilder(config)

    def __call__(
        self,
        observation: ObservationChunkOutput,
        runtime_state: TrajectoryRuntimeState,
        *,
        current_end_time: float,
    ) -> TTTUpdateOutcome:
        if observation.owner != runtime_state.owner:
            raise InferenceProtocolError("online update owner does not match observation")
        previous = runtime_state.online_overlap_memory
        built = self.input_builder(
            observation,
            previous=previous,
            current_end_time=current_end_time,
            enabled_terms=("pred", "identity", "event"),
        )
        output = compute_ttt_loss(self.predictor, built.inputs)
        result = functional_sgd_steps_from_ttt(
            ttt_output=output,
            fast_states=(_require_fast_state(runtime_state),),
            optimizer_config=self.config.fast_ttt.optimizer,
            optimizer_states=(_require_optimizer_state(runtime_state),),
        )[0]
        updated = replace(
            runtime_state,
            fast_weights=result.fast_state,
            optimizer=result.optimizer_state,
            online_overlap_memory=built.snapshot,
        )
        return TTTUpdateOutcome(
            runtime_state=updated,
            did_update=result.did_update,
            skip_reason=None if result.skip_reason is None else result.skip_reason.value,
            valid_term_count=result.valid_term_count,
            loss_value=float(output.total.detach().item()),
        )


@dataclass(frozen=True, slots=True)
class ChunkAudit:
    chunk_id: str
    original_start_time: float
    original_end_time: float
    causal_start_time: float | None
    causal_end_time: float | None
    original_frame_count: int
    causal_frame_count: int
    future_frame_count: int
    fast_version_used: int
    next_fast_version: int
    update_attempted: bool
    did_update: bool
    skip_reason: str | None
    valid_term_count: int
    state_before: RuntimeAuditSnapshot
    state_after_observe: RuntimeAuditSnapshot
    state_after_update: RuntimeAuditSnapshot

    def __post_init__(self) -> None:
        counts = (
            self.original_frame_count,
            self.causal_frame_count,
            self.future_frame_count,
            self.fast_version_used,
            self.next_fast_version,
            self.valid_term_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("chunk audit counts/versions must be non-negative integers")
        if self.causal_frame_count + self.future_frame_count != self.original_frame_count:
            raise ValueError("chunk audit frame counts must add up")
        if self.did_update and not self.update_attempted:
            raise ValueError("chunk update cannot succeed without an attempt")
        if self.did_update and self.next_fast_version != self.fast_version_used + 1:
            raise ValueError("accepted chunk update must advance fast version exactly once")
        if not self.did_update and self.next_fast_version != self.fast_version_used:
            raise ValueError("skipped chunk update cannot change fast version")


@dataclass(frozen=True, slots=True)
class ChunkExecution:
    observation: ObservationChunkOutput | None
    runtime_state: TrajectoryRuntimeState
    audit: ChunkAudit


@dataclass(frozen=True, slots=True)
class GenerateAudit:
    query_id: str
    query_kind: QueryAttemptKind
    retry_of: str | None
    prefill_count: int
    decode_count: int
    state_before: RuntimeAuditSnapshot
    state_after: RuntimeAuditSnapshot

    def __post_init__(self) -> None:
        if self.prefill_count != 1:
            raise ValueError("one inference query must execute exactly one prefill")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    answer_text: str
    reader_result: ReaderResult
    runtime_state: TrajectoryRuntimeState
    selected_record_ids: tuple[str, ...]
    state_attention: Tensor | None
    reset_audit: RuntimeResetAudit
    chunk_audit: tuple[ChunkAudit, ...]
    generate_audit: GenerateAudit
    release_audit: RuntimeReleaseAudit | None
    audit_fields: tuple[tuple[str, AuditValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str):
            raise TypeError("inference answer_text must be text")
        if self.selected_record_ids != self.reader_result.selected_record_ids:
            raise ValueError("inference selected records must preserve Reader provenance")
        keys = tuple(key for key, _ in self.audit_fields)
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("inference audit keys must be unique and non-empty")
        if self.state_attention is not None and (
            self.state_attention.requires_grad or self.state_attention.grad_fn is not None
        ):
            raise ValueError("inference State attention audit must be detached")
        if self.release_audit is not None and not self.runtime_state.released:
            raise ValueError("released inference result must expose a released runtime")


class PerVideoRuntimeManager:
    """Own exactly one functional per-video runtime and all lifecycle transitions."""

    def __init__(
        self,
        *,
        fast_adapter: FastTTTAdapter,
        state_bank: StructuredStateBank,
        identity_bank: IdentityBank,
        optimizer_config: InnerSGDConfig,
        audit_level: AuditLevel = AuditLevel.BOUNDARY,
        hot_cache_enabled: bool = False,
        hot_device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(fast_adapter, FastTTTAdapter):
            raise TypeError("runtime manager requires FastTTTAdapter")
        if not isinstance(state_bank, StructuredStateBank):
            raise TypeError("runtime manager requires StructuredStateBank")
        if not isinstance(identity_bank, IdentityBank):
            raise TypeError("runtime manager requires IdentityBank")
        if not isinstance(optimizer_config, InnerSGDConfig):
            raise TypeError("runtime manager requires validated InnerSGDConfig")
        if type(hot_cache_enabled) is not bool:
            raise TypeError("hot_cache_enabled must be bool")
        if not isinstance(audit_level, AuditLevel):
            raise TypeError("audit_level must be AuditLevel")
        self.fast_adapter = fast_adapter
        self.state_bank = state_bank
        self.identity_bank = identity_bank
        self.optimizer_config = optimizer_config
        self.audit_level = audit_level
        self.hot_cache_enabled = hot_cache_enabled
        self.hot_device = hot_device
        self._runtime: TrajectoryRuntimeState | None = None
        self._lifecycle: PrefillLifecycle | None = None
        self._reset_audit: RuntimeResetAudit | None = None
        self._chunk_audits: list[ChunkAudit] = []
        self._query_snapshots: dict[str, tuple[object, ...]] = {}
        self._lock = RLock()

    @property
    def active_runtime(self) -> TrajectoryRuntimeState | None:
        with self._lock:
            return self._runtime

    @property
    def reset_audit(self) -> RuntimeResetAudit:
        with self._lock:
            if self._reset_audit is None:
                raise InferenceProtocolError("runtime has not been reset")
            return self._reset_audit

    @property
    def chunk_audits(self) -> tuple[ChunkAudit, ...]:
        with self._lock:
            return tuple(self._chunk_audits)

    def reset(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> RuntimeResetAudit:
        """Reset all fast/optimizer/cache/Bank/FSM/Reader state for one video."""

        if not video_id or not trajectory_id:
            raise ValueError("runtime reset owner identifiers must be non-empty")
        if (
            query_signature.shape != (512,)
            or not torch.is_floating_point(query_signature)
            or not bool(torch.isfinite(query_signature).all())
        ):
            raise ValueError("runtime reset query_signature must be finite floating [512]")
        with self._lock:
            previous_snapshot = None
            previous_release_snapshot = None
            if self._runtime is not None:
                previous_snapshot = self._snapshot(self._runtime, content=True)
                prior_release = self._release_locked()
                previous_release_snapshot = prior_release.released

            owner = RuntimeOwner((video_id,), (trajectory_id,))
            dtype = self.fast_adapter.w0_1.dtype
            device = self.fast_adapter.w0_1.device
            signature = query_signature.detach().to(device=device, dtype=dtype).clone()
            fast = self.fast_adapter.reset_fast_state(differentiable=False)
            runtime = TrajectoryRuntimeState(
                owner=owner,
                next_chunk_index=0,
                fast_weights=fast,
                optimizer=reset_optimizer_state(self.optimizer_config),
                slot_state=None,
                temporal_cache=_empty_temporal_cache(owner, signature),
                e1_state=None,
                e2_state=None,
                state_bank=self.state_bank.reset(video_id, trajectory_id),
                identity_bank=self.identity_bank.reset(
                    video_id,
                    trajectory_id,
                    hot_device=self.hot_device,
                    hot_cache_enabled=self.hot_cache_enabled,
                ),
                retrieval_history=TensorizedRetrievalHistory(
                    video_id,
                    trajectory_id,
                    capacity_per_head=self.state_bank.config.retrieval_history_capacity_per_head,
                    source_dim=self.state_bank.config.retrieval_history_source_dim,
                    dtype=next(self.state_bank.semantic_projector.parameters()).dtype,
                    device=next(self.state_bank.semantic_projector.parameters()).device,
                ),
                reader_audit=(),
                released=False,
            )
            audit = RuntimeResetAudit(
                video_id=video_id,
                trajectory_id=trajectory_id,
                previous_runtime=previous_snapshot,
                previous_release=previous_release_snapshot,
                reset=self._snapshot(runtime, content=True),
                pristine=_pristine_state_stamp(runtime),
            )
            self._runtime = runtime
            self._lifecycle = PrefillLifecycle(owner)
            self._reset_audit = audit
            self._chunk_audits = []
            self._query_snapshots = {}
            return audit

    def observe_chunk(
        self,
        *,
        model: StateTTTModel,
        chunk: CausalChunk,
        query_input: RuntimeQueryInput,
        query_time: float,
        updater: TTTUpdateStage,
    ) -> ChunkExecution:
        """Observe with W_t, write hard state, then create W_(t+1) for the next chunk."""

        with self._lock:
            runtime = self._require_live_runtime()
            fast = _require_fast_state(runtime)
            lifecycle = self._require_ready_lifecycle()
            before_snapshot = self._snapshot(runtime)
            before_fast_stamp = _fast_state_stamp(fast)
            causal = chunk.causal_prefix(query_time)
            if causal is None:
                audit = ChunkAudit(
                    chunk_id=chunk.chunk_id,
                    original_start_time=chunk.start_time,
                    original_end_time=chunk.end_time,
                    causal_start_time=None,
                    causal_end_time=None,
                    original_frame_count=len(chunk.frames),
                    causal_frame_count=0,
                    future_frame_count=len(chunk.frames),
                    fast_version_used=fast.fast_version,
                    next_fast_version=fast.fast_version,
                    update_attempted=False,
                    did_update=False,
                    skip_reason="no_causal_frames",
                    valid_term_count=0,
                    state_before=before_snapshot,
                    state_after_observe=before_snapshot,
                    state_after_update=before_snapshot,
                )
                self._chunk_audits.append(audit)
                return ChunkExecution(None, runtime, audit)

            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            model_runtime = BatchRuntimeState((runtime,))
            model_bank_states = model_runtime.bank_states
            if not model.feature_flags.fast_enabled:
                raise InferenceProtocolError("online inference requires the managed Fast Adapter")
            self.fast_adapter.last_audit = None
            with self.fast_adapter.use_fast_state(fast):
                observation = model.observe_chunk(
                    ObservationChunkRequest(
                        owner=owner,
                        video_input=causal if causal.model_input is None else causal.model_input,
                        query_input=query_input,
                        runtime_state=model_runtime,
                        bank_states=model_bank_states,
                        inference=True,
                    ),
                    lifecycle,
                )
            fast_audit = self.fast_adapter.last_audit
            if not isinstance(fast_audit, FastTTTForwardAudit) or not fast_audit.used_runtime_state:
                raise InferenceProtocolError(
                    "observe did not consume the manager-bound FastWeightsState"
                )
            expected_version = (fast.fast_version,)
            expected_updates = (fast.update_count,)
            if (
                fast_audit.fast_versions != expected_version
                or fast_audit.update_counts != expected_updates
                or len(fast_audit.valid_token_counts) != 1
            ):
                raise InferenceProtocolError(
                    "Fast Adapter audit owner/version does not match the per-video runtime"
                )
            observed_batch = _require_batch_runtime(observation.runtime_state, owner)
            observed = observed_batch.rows[0]
            observation = replace(
                observation,
                runtime_state=observed_batch,
                bank_states=(observed.state_bank,),
            )
            observed_fast = _require_fast_state(observed)
            if _fast_state_stamp(observed_fast) != before_fast_stamp:
                raise InferenceProtocolError(
                    "observe/hard-write mutated fast weights before the TTT update boundary"
                )
            if (
                observed.optimizer != runtime.optimizer
                or observed.reader_audit != runtime.reader_audit
            ):
                raise InferenceProtocolError(
                    "observe/hard-write cannot change optimizer or Reader audit state"
                )
            after_observe_snapshot = self._snapshot(observed)
            hard_stamp = _hard_state_stamp(observed)
            outcome = updater(observation, observed, current_end_time=causal.end_time)
            updated = _require_trajectory_runtime(outcome.runtime_state, owner)
            if _hard_state_stamp(updated) != hard_stamp:
                raise InferenceProtocolError(
                    "TTT updater may only change fast/optimizer/overlap runtime state"
                )
            _validate_update_transition(observed, outcome)
            after_update_snapshot = self._snapshot(updated, content=True)
            next_observation = replace(
                observation,
                runtime_state=BatchRuntimeState((updated,)),
                bank_states=(updated.state_bank,),
            )
            audit = ChunkAudit(
                chunk_id=chunk.chunk_id,
                original_start_time=chunk.start_time,
                original_end_time=chunk.end_time,
                causal_start_time=causal.start_time,
                causal_end_time=causal.end_time,
                original_frame_count=len(chunk.frames),
                causal_frame_count=len(causal.frames),
                future_frame_count=len(chunk.frames) - len(causal.frames),
                fast_version_used=observed_fast.fast_version,
                next_fast_version=_require_fast_state(updated).fast_version,
                update_attempted=True,
                did_update=outcome.did_update,
                skip_reason=outcome.skip_reason,
                valid_term_count=outcome.valid_term_count,
                state_before=before_snapshot,
                state_after_observe=after_observe_snapshot,
                state_after_update=after_update_snapshot,
            )
            self._runtime = updated
            self._chunk_audits.append(audit)
            return ChunkExecution(next_observation, updated, audit)

    def answer_query(
        self,
        *,
        model: StateTTTModel,
        observation: ObservationChunkOutput,
        answer_inputs: AnswerInputs,
        attempt: QueryAttempt,
        max_new_tokens: int = 16,
    ) -> InferenceResult:
        """Prepare once, generate once and prove the complete answer leaves state immutable."""

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        with self._lock:
            runtime = self._require_live_runtime()
            fast = _require_fast_state(runtime)
            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            if observation.owner != owner:
                raise InferenceProtocolError("answer observation owner does not match runtime")
            observation = replace(
                observation,
                runtime_state=BatchRuntimeState((runtime,)),
                bank_states=(runtime.state_bank,),
            )
            causal_stamp = _causal_state_stamp(runtime)
            self._register_query_attempt(attempt, causal_stamp)
            lifecycle = self._query_lifecycle(owner)
            before_guard = _runtime_guard_stamp(runtime)
            before_snapshot = self._snapshot(runtime, content=True)
            prepared = model.prepare_answer(
                AnswerQueryRequest(
                    owner=owner,
                    observation=observation,
                    base_input_ids=answer_inputs.base_input_ids,
                    base_attention_mask=answer_inputs.base_attention_mask,
                    pixel_values_videos=answer_inputs.pixel_values_videos,
                    video_grid_thw=answer_inputs.video_grid_thw,
                    tokenizer=answer_inputs.tokenizer,
                    embedding_owner=answer_inputs.embedding_owner,
                    rope_indexer=answer_inputs.rope_indexer,
                    qwen_kwargs=answer_inputs.qwen_kwargs,
                ),
                lifecycle,
            )
            generated = model.generate_answer(
                prepared,
                lifecycle,
                max_new_tokens=max_new_tokens,
            )
            if len(generated.reader) != 1 or not isinstance(generated.reader[0], ReaderResult):
                raise InferenceProtocolError("online inference requires one ReaderResult")
            reader_result = generated.reader[0]
            after_guard = _runtime_guard_stamp(runtime)
            if after_guard != before_guard:
                raise InferenceProtocolError("answer prefill/generation mutated runtime state")
            after_snapshot = self._snapshot(runtime, content=True)
            state_attention = _state_attention(generated.resampler)
            self._runtime = replace(
                runtime,
                reader_audit=runtime.reader_audit + (reader_result,),
            )
            lifecycle_audit = lifecycle.audit()
            generate_audit = GenerateAudit(
                query_id=attempt.query_id,
                query_kind=attempt.kind,
                retry_of=attempt.retry_of,
                prefill_count=lifecycle_audit.prefill_count,
                decode_count=lifecycle_audit.decode_count,
                state_before=before_snapshot,
                state_after=after_snapshot,
            )
            result_runtime = self._runtime
            return InferenceResult(
                answer_text=generated.answer_text,
                reader_result=reader_result,
                runtime_state=result_runtime,
                selected_record_ids=reader_result.selected_record_ids,
                state_attention=state_attention,
                reset_audit=self.reset_audit,
                chunk_audit=self.chunk_audits,
                generate_audit=generate_audit,
                release_audit=None,
                audit_fields=(
                    ("video_id", runtime.video_id),
                    ("trajectory_id", runtime.trajectory_id),
                    ("query_id", attempt.query_id),
                    ("query_kind", attempt.kind.value),
                    ("reader_status", reader_result.status.value),
                    ("selected_record_count", len(reader_result.selected_record_ids)),
                    ("prefill_count", lifecycle_audit.prefill_count),
                    ("decode_count", lifecycle_audit.decode_count),
                    ("final_fast_version", fast.fast_version),
                    ("final_update_count", fast.update_count),
                    ("final_skip_count", fast.skip_count),
                ),
            )

    def observe_query_readonly(
        self,
        *,
        model: StateTTTModel,
        chunk: CausalChunk,
        query_input: RuntimeQueryInput,
        query_time: float,
    ) -> ObservationChunkOutput:
        """Observe one Query feature set with current W_t without committing Bank/FSM state."""

        with self._lock:
            runtime = self._require_live_runtime()
            fast = _require_fast_state(runtime)
            causal = chunk.causal_prefix(query_time)
            if causal is None:
                raise InferenceProtocolError("Query observation contains no causal frame")
            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            before_guard = _runtime_guard_stamp(runtime)
            lifecycle = PrefillLifecycle(owner)
            self.fast_adapter.last_audit = None
            with self.fast_adapter.use_fast_state(fast), torch.no_grad():
                observation = model.observe_chunk(
                    ObservationChunkRequest(
                        owner=owner,
                        video_input=causal if causal.model_input is None else causal.model_input,
                        query_input=query_input,
                        runtime_state=BatchRuntimeState((runtime,)),
                        bank_states=(runtime.state_bank,),
                        inference=True,
                        retrieval_history_write_enabled=False,
                    ),
                    lifecycle,
                )
            fast_audit = self.fast_adapter.last_audit
            if not isinstance(fast_audit, FastTTTForwardAudit) or not fast_audit.used_runtime_state:
                raise InferenceProtocolError(
                    "Query observation did not consume the manager-bound FastWeightsState"
                )
            if fast_audit.fast_versions != (fast.fast_version,) or fast_audit.update_counts != (
                fast.update_count,
            ):
                raise InferenceProtocolError("Query observation used the wrong fast version")
            if _runtime_guard_stamp(runtime) != before_guard:
                raise InferenceProtocolError("read-only Query observation mutated runtime state")
            return replace(
                observation,
                runtime_state=BatchRuntimeState((runtime,)),
                bank_states=(runtime.state_bank,),
            )

    def release(self) -> RuntimeReleaseAudit | None:
        """Release all trajectory storage; safe and idempotent for exception cleanup."""

        with self._lock:
            if self._runtime is None:
                return None
            return self._release_locked()

    def _release_locked(self) -> RuntimeReleaseAudit:
        runtime = self._require_live_runtime()
        fast = _require_fast_state(runtime)
        temporal_cache = _require_temporal_cache(runtime)
        before_snapshot = self._snapshot(runtime, content=True)
        released_bank = self.state_bank.release(runtime.state_bank)
        released_identity = self.identity_bank.release(runtime.identity_bank)
        if runtime.retrieval_history is not None:
            runtime.retrieval_history.release()
        owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
        released = TrajectoryRuntimeState(
            owner=owner,
            next_chunk_index=0,
            fast_weights=_released_fast_state(fast.w0_1.dtype),
            optimizer=reset_optimizer_state(self.optimizer_config),
            slot_state=None,
            temporal_cache=_empty_temporal_cache(
                owner,
                torch.empty((512,), dtype=temporal_cache.hidden.dtype, device="meta"),
            ),
            e1_state=None,
            e2_state=None,
            state_bank=released_bank,
            identity_bank=released_identity,
            retrieval_history=runtime.retrieval_history,
            reader_audit=(),
            released=True,
        )
        audit = RuntimeReleaseAudit(
            video_id=runtime.video_id,
            trajectory_id=runtime.trajectory_id,
            before=before_snapshot,
            released=self._snapshot(released, content=True),
            state_bank_version=released_bank.version,
            identity_bank_version=released_identity.version,
            runtime_state=released,
        )
        self._runtime = None
        self._lifecycle = None
        return audit

    def _snapshot(
        self,
        state: TrajectoryRuntimeState,
        *,
        content: bool = False,
    ) -> RuntimeAuditSnapshot:
        if self.audit_level is AuditLevel.OFF:
            return RuntimeAuditSnapshot(AuditLevel.OFF, None, None)
        return RuntimeAuditSnapshot(
            level=self.audit_level,
            boundary=runtime_boundary_stamp(state),
            content_sha256=(
                runtime_checksum(state) if content and self.audit_level is AuditLevel.FULL else None
            ),
        )

    def _register_query_attempt(
        self,
        attempt: QueryAttempt,
        stamp: tuple[object, ...],
    ) -> None:
        if attempt.query_id in self._query_snapshots:
            raise InferenceProtocolError("query_id has already been used in this runtime")
        if attempt.kind is QueryAttemptKind.RETRY:
            expected = self._query_snapshots.get(cast(str, attempt.retry_of))
            if expected is None:
                raise InferenceProtocolError("retry_of does not name a completed query")
            if expected != stamp:
                raise InferenceProtocolError("retry requires the unchanged causal runtime snapshot")
        self._query_snapshots[attempt.query_id] = stamp

    def _query_lifecycle(self, owner: RuntimeOwner) -> PrefillLifecycle:
        lifecycle = self._lifecycle
        if lifecycle is None:
            raise InferenceProtocolError("runtime lifecycle is unavailable")
        if lifecycle.audit().phase is not LifecyclePhase.READY:
            lifecycle = PrefillLifecycle(owner)
            self._lifecycle = lifecycle
        return lifecycle

    def _require_live_runtime(self) -> TrajectoryRuntimeState:
        runtime = self._runtime
        if runtime is None or runtime.released:
            raise InferenceProtocolError("a live per-video runtime is required")
        return runtime

    def _require_ready_lifecycle(self) -> PrefillLifecycle:
        lifecycle = self._lifecycle
        if lifecycle is None or lifecycle.audit().phase is not LifecyclePhase.READY:
            raise InferenceProtocolError("observation requires a fresh prefill lifecycle")
        return lifecycle


def run_inference(
    *,
    manager: PerVideoRuntimeManager,
    model: StateTTTModel,
    request: InferenceRequest,
    updater: TTTUpdateStage,
) -> InferenceResult:
    """Execute reset -> causal chunks -> one greedy generate -> unconditional release."""

    if not isinstance(manager, PerVideoRuntimeManager):
        raise TypeError("run_inference requires PerVideoRuntimeManager")
    if not isinstance(model, StateTTTModel):
        raise TypeError("run_inference requires StateTTTModel")
    if not isinstance(request, InferenceRequest):
        raise TypeError("run_inference requires InferenceRequest")
    manager.reset(request.video_id, request.trajectory_id, request.query_signature)
    latest_observation: ObservationChunkOutput | None = None
    try:
        for chunk in request.chunks:
            execution = manager.observe_chunk(
                model=model,
                chunk=chunk,
                query_input=request.query_input,
                query_time=request.query_time,
                updater=updater,
            )
            if execution.observation is not None:
                latest_observation = execution.observation
        if request.query_observation is not None:
            latest_observation = manager.observe_query_readonly(
                model=model,
                chunk=request.query_observation,
                query_input=request.query_input,
                query_time=request.query_time,
            )
        if latest_observation is None:
            raise InferenceProtocolError("no causal frame was available before query_time")
        result = manager.answer_query(
            model=model,
            observation=latest_observation,
            answer_inputs=request.answer_inputs,
            attempt=request.attempt,
            max_new_tokens=request.max_new_tokens,
        )
        release_audit = manager.release()
        if release_audit is None:  # pragma: no cover - protected by the live result path
            raise InferenceProtocolError("runtime disappeared before successful release")
        return replace(
            result,
            runtime_state=release_audit.runtime_state,
            release_audit=release_audit,
            audit_fields=result.audit_fields
            + (
                ("released", True),
                ("release_content_sha256", release_audit.released.content_sha256),
            ),
        )
    except BaseException:
        manager.release()
        raise


def runtime_checksum(state: TrajectoryRuntimeState) -> str:
    """Hash tensor values for explicit full-audit boundaries only."""

    if not isinstance(state, TrajectoryRuntimeState):
        raise TypeError("runtime_checksum requires TrajectoryRuntimeState")
    digest = hashlib.sha256()
    _hash_value(digest, state)
    return digest.hexdigest()


def runtime_boundary_stamp(state: TrajectoryRuntimeState) -> RuntimeBoundaryStamp:
    """Describe runtime identity and Tensor versions without reading Tensor contents."""

    if not isinstance(state, TrajectoryRuntimeState):
        raise TypeError("runtime_boundary_stamp requires TrajectoryRuntimeState")
    fast = _require_fast_state(state)
    components = (
        state.fast_weights,
        state.optimizer,
        state.slot_state,
        state.temporal_cache,
        state.e1_state,
        state.e2_state,
        state.state_bank,
        state.identity_bank,
        state.retrieval_history,
        state.online_overlap_memory,
    )
    return RuntimeBoundaryStamp(
        video_id=state.video_id,
        trajectory_id=state.trajectory_id,
        next_chunk_index=state.next_chunk_index,
        released=state.released,
        fast_version=fast.fast_version,
        update_count=fast.update_count,
        skip_count=fast.skip_count,
        state_bank_version=state.state_bank.version,
        identity_bank_version=state.identity_bank.version,
        component_ids=tuple(0 if value is None else id(value) for value in components),
        tensor_versions=_boundary_tensor_versions(state),
    )


def assert_inference_runtime_payload(payload: Mapping[str, object]) -> None:
    """Apply the P2 allowlist and recursively reject nested supervision fields."""

    assert_runtime_payload_safe(payload, layer="Inference")
    denied_paths = tuple(_nested_denied_paths(payload))
    if denied_paths:
        raise ValueError(
            "Inference runtime payload contains nested denied fields: "
            + ", ".join(sorted(denied_paths))
        )
    video = payload["video"]
    question = payload["question"]
    explicit = payload["explicit_time_values"]
    if video is None:
        raise ValueError("Inference runtime video cannot be None")
    if isinstance(question, str):
        if not question.strip():
            raise ValueError("Inference runtime question must be non-empty")
        _payload_query_time(payload)
        if not isinstance(explicit, (tuple, list)) or any(
            not _is_finite_nonnegative_number(value) for value in explicit
        ):
            raise ValueError("explicit_time_values must contain finite non-negative numbers")
        return
    if (
        not isinstance(question, (tuple, list))
        or not question
        or any(not isinstance(value, str) or not value.strip() for value in question)
    ):
        raise ValueError("Inference runtime batch questions must be non-empty strings")
    batch_size = len(question)
    query_times = payload["query_time"]
    if isinstance(query_times, Tensor):
        valid_times = (
            query_times.ndim == 1
            and query_times.numel() == batch_size
            and bool(torch.isfinite(query_times).all())
            and bool(torch.all(query_times >= 0))
        )
    else:
        valid_times = isinstance(query_times, (tuple, list)) and len(query_times) == batch_size
        valid_times = valid_times and all(
            _is_finite_nonnegative_number(value) for value in cast(Sequence[object], query_times)
        )
    if not valid_times:
        raise ValueError("Inference runtime batch query_time values are invalid")
    if (
        not isinstance(explicit, (tuple, list))
        or len(explicit) != batch_size
        or any(
            not isinstance(row, (tuple, list))
            or any(not _is_finite_nonnegative_number(value) for value in row)
            for row in explicit
        )
    ):
        raise ValueError("batched explicit_time_values must align and be finite")


def _empty_temporal_cache(owner: RuntimeOwner, query_signature: Tensor) -> TemporalCache:
    device = query_signature.device
    dtype = query_signature.dtype
    hidden = torch.empty((1, 0, 768), dtype=dtype, device=device)
    kv = tuple(torch.empty((1, 12, 0, 64), dtype=dtype, device=device) for _ in range(6))
    replay_kv = tuple(torch.empty((1, 12, 0, 64), dtype=dtype, device=device) for _ in range(6))
    return TemporalCache(
        hidden=hidden,
        layer_keys=kv,
        layer_values=tuple(value.clone() for value in kv),
        replay_layer_keys=replay_kv,
        replay_layer_values=tuple(value.clone() for value in replay_kv),
        timestamps=torch.empty((1, 0), dtype=torch.float64, device=device),
        replay_timestamps=torch.empty((1, 0), dtype=torch.float64, device=device),
        position_ids=torch.empty((1, 0), dtype=torch.int64, device=device),
        replay_position_ids=torch.empty((1, 0), dtype=torch.int64, device=device),
        valid_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
        replay_valid_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
        query_signatures=query_signature.detach().reshape(1, 512).clone(),
        total_seen=torch.zeros((1,), dtype=torch.int64, device=device),
    )


def _released_fast_state(dtype: torch.dtype) -> FastWeightsState:
    def matrix(*, requires_grad: bool) -> Tensor:
        return torch.empty((768, 768), dtype=dtype, device="meta", requires_grad=requires_grad)

    return FastWeightsState(
        w0_1=matrix(requires_grad=False),
        w0_2=matrix(requires_grad=False),
        w_t_1=matrix(requires_grad=True),
        w_t_2=matrix(requires_grad=True),
        fast_version=0,
        update_count=0,
        skip_count=0,
        differentiable=False,
    )


def _require_trajectory_runtime(value: object, owner: RuntimeOwner) -> TrajectoryRuntimeState:
    if not isinstance(value, TrajectoryRuntimeState):
        raise TypeError("online update stages must return TrajectoryRuntimeState")
    if value.released:
        raise InferenceProtocolError("online stages cannot return a released runtime")
    if (value.video_id, value.trajectory_id) != (owner.video_ids[0], owner.trajectory_ids[0]):
        raise InferenceProtocolError("online stage returned runtime for a different owner")
    return value


def _require_batch_runtime(value: object, owner: RuntimeOwner) -> BatchRuntimeState:
    if not isinstance(value, BatchRuntimeState):
        raise TypeError("online model stages must return BatchRuntimeState")
    value.validate_for(owner)
    if len(value.rows) != 1:
        raise InferenceProtocolError("online runtime must contain exactly one trajectory")
    _require_trajectory_runtime(value.rows[0], owner)
    return value


def _require_fast_state(state: TrajectoryRuntimeState) -> FastWeightsState:
    if state.fast_weights is None:
        raise InferenceProtocolError("online trajectory is missing fast weights")
    return state.fast_weights


def _require_optimizer_state(state: TrajectoryRuntimeState) -> OptimizerRuntimeState:
    if state.optimizer is None:
        raise InferenceProtocolError("online trajectory is missing optimizer state")
    return state.optimizer


def _require_temporal_cache(state: TrajectoryRuntimeState) -> TemporalCache:
    if state.temporal_cache is None:
        raise InferenceProtocolError("online trajectory is missing temporal cache")
    return state.temporal_cache


def _validate_update_transition(before: TrajectoryRuntimeState, outcome: TTTUpdateOutcome) -> None:
    after = outcome.runtime_state
    before_fast = _require_fast_state(before)
    after_fast = _require_fast_state(after)
    before_optimizer = _require_optimizer_state(before)
    after_optimizer = _require_optimizer_state(after)
    if _boundary_tensor_versions((before_fast.w0_1, before_fast.w0_2)) != (
        _boundary_tensor_versions((after_fast.w0_1, after_fast.w0_2))
    ):
        raise InferenceProtocolError("TTT updater cannot change the meta-learned W0 snapshot")
    optimizer_contract_before = (
        before_optimizer.optimizer_name,
        before_optimizer.learning_rate,
        before_optimizer.momentum,
        before_optimizer.weight_decay,
        before_optimizer.steps_per_chunk,
        before_optimizer.grad_clip_norm,
    )
    optimizer_contract_after = (
        after_optimizer.optimizer_name,
        after_optimizer.learning_rate,
        after_optimizer.momentum,
        after_optimizer.weight_decay,
        after_optimizer.steps_per_chunk,
        after_optimizer.grad_clip_norm,
    )
    if optimizer_contract_after != optimizer_contract_before:
        raise InferenceProtocolError("TTT updater cannot change the optimizer contract")
    if after_optimizer.attempted_update_count != before_optimizer.attempted_update_count + 1:
        raise InferenceProtocolError("one chunk must make exactly one optimizer update attempt")
    if outcome.did_update:
        expected = (
            before_fast.fast_version + 1,
            before_fast.update_count + 1,
            before_fast.skip_count,
        )
        actual = (after_fast.fast_version, after_fast.update_count, after_fast.skip_count)
        if actual != expected or after_optimizer.last_skip_reason is not None:
            raise InferenceProtocolError("accepted TTT update has inconsistent counters")
        after_tensors = (after_fast.w_t_1, after_fast.w_t_2)
        if any(not bool(torch.isfinite(value).all()) for value in after_tensors):
            raise InferenceProtocolError("accepted TTT update produced non-finite W_t")
        if _boundary_tensor_versions((before_fast.w_t_1, before_fast.w_t_2)) == (
            _boundary_tensor_versions(after_tensors)
        ):
            raise InferenceProtocolError("accepted TTT update must publish new W_t tensors")
    else:
        expected = (
            before_fast.fast_version,
            before_fast.update_count,
            before_fast.skip_count + 1,
        )
        actual = (after_fast.fast_version, after_fast.update_count, after_fast.skip_count)
        if actual != expected or after_optimizer.last_skip_reason != outcome.skip_reason:
            raise InferenceProtocolError("skipped TTT update has inconsistent counters/reason")
        before_tensors = (before_fast.w_t_1, before_fast.w_t_2)
        after_tensors = (after_fast.w_t_1, after_fast.w_t_2)
        if any(
            before.shape != after.shape
            or before.dtype != after.dtype
            or before.device != after.device
            or not torch.equal(before.detach(), after.detach())
            for before, after in zip(before_tensors, after_tensors, strict=True)
        ):
            raise InferenceProtocolError("skipped TTT update cannot change W_t values")


def _state_attention(resampler: object | None) -> Tensor | None:
    value = None if resampler is None else getattr(resampler, "cross_attention_weights", None)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError("State Resampler attention audit must be a Tensor")
    return value.detach().clone()


def _runtime_guard_stamp(state: TrajectoryRuntimeState) -> tuple[object, ...]:
    stamp = runtime_boundary_stamp(state)
    return (
        stamp.video_id,
        stamp.trajectory_id,
        stamp.next_chunk_index,
        stamp.released,
        stamp.fast_version,
        stamp.update_count,
        stamp.skip_count,
        stamp.state_bank_version,
        stamp.identity_bank_version,
        stamp.component_ids,
        stamp.tensor_versions,
    )


def _causal_state_stamp(state: TrajectoryRuntimeState) -> tuple[object, ...]:
    return _runtime_guard_stamp(replace(state, reader_audit=()))


def _hard_state_stamp(state: TrajectoryRuntimeState) -> tuple[object, ...]:
    values = (
        state.video_id,
        state.trajectory_id,
        state.next_chunk_index,
        state.slot_state,
        state.temporal_cache,
        state.e1_state,
        state.e2_state,
        state.state_bank,
        state.identity_bank,
        state.retrieval_history,
        state.reader_audit,
        state.released,
    )
    return (
        tuple(id(value) for value in values[3:10]),
        state.state_bank.version,
        state.identity_bank.version,
        None if state.retrieval_history is None else state.retrieval_history.version,
        _boundary_tensor_versions(values),
        values[:3],
        values[10:],
    )


def _fast_state_stamp(state: FastWeightsState) -> tuple[object, ...]:
    return (
        state.fast_version,
        state.update_count,
        state.skip_count,
        state.differentiable,
        _boundary_tensor_versions(state),
    )


def _pristine_state_stamp(state: TrajectoryRuntimeState) -> RuntimePristineStamp:
    """Build an owner-independent, content-free reset fingerprint."""

    fast = _require_fast_state(state)
    optimizer = _require_optimizer_state(state)
    temporal_cache = _require_temporal_cache(state)
    return RuntimePristineStamp(
        fast_shape=tuple(fast.w_t_1.shape),
        fast_dtype=str(fast.w_t_1.dtype),
        fast_version=fast.fast_version,
        update_count=fast.update_count,
        skip_count=fast.skip_count,
        optimizer_attempted_updates=optimizer.attempted_update_count,
        temporal_width=temporal_cache.hidden.shape[1],
        state_record_count=len(state.state_bank.records),
        identity_candidate_count=len(state.identity_bank.candidates),
        identity_confirmed_count=len(state.identity_bank.confirmed),
        reader_audit_count=len(state.reader_audit),
        released=state.released,
    )


def _boundary_tensor_versions(
    value: object,
) -> tuple[tuple[str, str, str, tuple[int, ...], int | None, int], ...]:
    found: list[tuple[str, str, str, tuple[int, ...], int | None, int]] = []
    seen: set[int] = set()

    def visit(item: object, path: str) -> None:
        if isinstance(item, Tensor):
            pointer = None if item.device.type == "meta" else item.untyped_storage().data_ptr()
            found.append(
                (path, str(item.dtype), str(item.device), tuple(item.shape), pointer, item._version)
            )
            return
        if item is None or isinstance(item, (str, int, float, bool, Path, Enum)):
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(item[key], f"{path}.{key!r}")
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}")

    visit(value, "runtime")
    return tuple(found)


class _Digest(Protocol):
    def update(self, value: bytes) -> object: ...


def _hash_value(digest: _Digest, value: object) -> None:
    digest.update(type(value).__qualname__.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, Tensor):
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.device).encode("ascii"))
        digest.update(str(value.requires_grad).encode("ascii"))
        if value.device.type != "meta":
            raw = value.detach().to(device="cpu").contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        return
    if isinstance(value, TensorizedRetrievalHistory):
        for item in (
            value.video_id,
            value.trajectory_id,
            value.capacity_per_head,
            value.source_dim,
            value.sources,
            value.sequence_ids,
            value.operator_codes,
            value.timestamps,
            value.time_ranges,
            value.valid_mask,
            value.eligible_mask,
            value.sizes,
            value.write_ptrs,
            value.next_sequence,
            value.version,
            value.released,
        ):
            _hash_value(digest, item)
        return
    if value is None or isinstance(value, (str, int, float, bool, Path)):
        digest.update(repr(value).encode("utf-8"))
        return
    if isinstance(value, Enum):
        digest.update(str(value.value).encode("utf-8"))
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _hash_value(digest, item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            digest.update(field.name.encode("utf-8"))
            _hash_value(digest, getattr(value, field.name))
        return
    raise TypeError(f"runtime checksum does not support {type(value).__qualname__}")


def _nested_denied_paths(value: object, prefix: str = "payload") -> Sequence[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if prefix != "payload" and key_text in RUNTIME_DENYLIST:
                found.append(path)
            found.extend(_nested_denied_paths(child, path))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found.extend(_nested_denied_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


def _payload_query_time(payload: Mapping[str, object]) -> float:
    value = payload["query_time"]
    if not _is_finite_nonnegative_number(value):
        raise ValueError("Inference query_time must be a finite non-negative number")
    return float(cast(float | int, value))


def _is_finite_nonnegative_number(value: object) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(cast(float | int, value)))
        and float(cast(float | int, value)) >= 0.0
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online State-TTT video inference")
    parser.add_argument("--run", type=Path, required=True, metavar="REQUEST_JSON")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-sample-fps", type=float, default=2.0)
    parser.add_argument("--video-max-pixels", type=int, default=131_072)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one strict request, run production inference, and write one fixed JSON result."""

    from ttt_svcbench_qwen.production_runtime import (
        QueryObservationSpec,
        SupportChunkSpec,
        _expand_qwen_video_placeholders,
        _tokenize_text_only,
        _user_message,
        build_inference_runtime_bundle,
    )

    args = _build_parser().parse_args(argv)
    run_path = cast(Path, args.run)
    raw: object = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("request JSON must contain one object")
    payload = {str(key): value for key, value in raw.items()}
    required = {
        "video_id",
        "trajectory_id",
        "query_id",
        "video",
        "question",
        "query_time",
        "explicit_time_values",
    }
    if set(payload) != required:
        raise ValueError(
            f"request JSON fields must match exactly; missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    identity = {name: payload[name] for name in ("video_id", "trajectory_id", "query_id")}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("video_id, trajectory_id and query_id must be non-empty strings")
    runtime_payload = {
        name: payload[name] for name in ("video", "question", "query_time", "explicit_time_values")
    }
    assert_inference_runtime_payload(runtime_payload)
    video_path = Path(cast(str, payload["video"])).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"request video does not exist: {video_path}")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cast(str, args.dtype)]
    bundle = build_inference_runtime_bundle(
        model_root=cast(Path, args.model_root),
        checkpoint=cast(Path, args.checkpoint),
        device=cast(str, args.device),
        dtype=dtype,
        maximum_pixels=cast(int, args.video_max_pixels),
    )
    query = RuntimeQueryInput(
        video_id=cast(str, payload["video_id"]),
        trajectory_id=cast(str, payload["trajectory_id"]),
        query_id=cast(str, payload["query_id"]),
        query_index=0,
        video=video_path,
        question=cast(str, payload["question"]),
        query_time=_payload_query_time(runtime_payload),
        explicit_time_values=tuple(
            float(value) for value in cast(Sequence[float], payload["explicit_time_values"])
        ),
    )
    encoded = bundle.state_model.components.query_encoder(query, inference=True)
    query_signature = encoded.q_target[0].detach().clone()
    chunks: list[CausalChunk] = []
    specs: list[SupportChunkSpec] = []
    end = min(8.0, query.query_time)
    index = 0
    while end > 0.0:
        start = max(0.0, end - 8.0)
        spec = SupportChunkSpec(
            chunk_id=f"chunk-{index:06d}",
            video_path=video_path,
            start_time=start,
            end_time=end,
            maximum_frames=16,
            query_time=query.query_time,
            reset_soft_state=index == 0,
        )
        specs.append(spec)
        tubelet_count = spec.maximum_frames // 2
        times = tuple(
            start + (offset + 1) * (end - start) / tubelet_count for offset in range(tubelet_count)
        )
        positions = tuple(range(index * tubelet_count, (index + 1) * tubelet_count))
        chunks.append(
            CausalChunk(
                chunk_id=spec.chunk_id,
                frames=tuple(range(len(times))),
                timestamps=times,
                position_ids=positions,
                model_input=spec,
            )
        )
        index += 1
        if end >= query.query_time:
            break
        end = min(query.query_time, end + 8.0)
    if not specs:
        raise ValueError("query_time must be greater than zero for video inference")
    state_query_spec = QueryObservationSpec(
        chunk_id=f"state-query-{query.query_id}",
        video_path=video_path,
        start_time=max(0.0, query.query_time - 8.0),
        end_time=query.query_time,
        maximum_frames=16,
        query_time=query.query_time,
        sampling_fps=cast(float, args.query_sample_fps),
        query_role="state_query",
    )
    answer_query_spec = QueryObservationSpec(
        chunk_id=f"answer-query-{query.query_id}",
        video_path=video_path,
        start_time=0.0,
        end_time=query.query_time,
        maximum_frames=256,
        query_time=query.query_time,
        sampling_fps=cast(float, args.query_sample_fps),
        query_role="answer_query",
    )
    state_materialized = bundle.video_materializer(state_query_spec)
    answer_materialized = bundle.video_materializer(answer_query_spec)
    latest_times = tuple(
        float(value) for value in state_materialized.tubelet_timestamps[0].tolist()
    )
    latest_positions = tuple(
        int(value) for value in state_materialized.tubelet_position_ids[0].tolist()
    )
    query_observation = CausalChunk(
        chunk_id=state_query_spec.chunk_id,
        frames=tuple(range(len(latest_times))),
        timestamps=latest_times,
        position_ids=latest_positions,
        model_input=state_materialized,
    )
    processor = bundle.processor
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise TypeError("Qwen processor must provide apply_chat_template()")
    prompt = apply_template(
        [_user_message(query.question)], tokenize=False, add_generation_prompt=True
    )
    prompt = _expand_qwen_video_placeholders(
        processor,
        prompt,
        answer_materialized.video_grid_thw,
        answer_materialized.frames.shape[0],
    )
    input_ids, attention_mask = _tokenize_text_only(bundle.tokenizer, prompt)
    request = InferenceRequest(
        query_input=query,
        query_signature=query_signature,
        chunks=tuple(chunks),
        answer_inputs=AnswerInputs(
            base_input_ids=input_ids,
            base_attention_mask=attention_mask,
            pixel_values_videos=answer_materialized.pixel_values_videos,
            video_grid_thw=answer_materialized.video_grid_thw,
            tokenizer=bundle.tokenizer,
            embedding_owner=bundle.qwen_adapter.qwen_model,
            rope_indexer=bundle.qwen_adapter.qwen_model,
        ),
        attempt=QueryAttempt(query.query_id),
        query_observation=query_observation,
        max_new_tokens=16,
    )
    result = run_inference(
        manager=cast(PerVideoRuntimeManager, bundle.manager),
        model=bundle.state_model,
        request=request,
        updater=cast(TTTUpdateStage, bundle.updater),
    )
    audits = dict(result.audit_fields)
    output = {
        "video_id": query.video_id,
        "trajectory_id": query.trajectory_id,
        "query_id": query.query_id,
        "answer": result.answer_text,
        "reader": {
            "status": result.reader_result.status.value,
            "selected_record_ids": list(result.selected_record_ids),
        },
        "fast_version": audits["final_fast_version"],
        "update_count": audits["final_update_count"],
        "skip_count": audits["final_skip_count"],
        "audit": {
            "level": bundle.config.inference.audit_level.value,
            "prefill_count": result.generate_audit.prefill_count,
            "decode_count": result.generate_audit.decode_count,
            "chunk_count": len(result.chunk_audit),
            "state_query_visual_mode": "recent_chunk",
            "answer_query_visual_mode": "causal_prefix",
            "prepared_video_feature_count": 0,
            "history_feature_set_count": 0,
            "state_query_frame_count": int(state_materialized.frames.shape[0]),
            "answer_query_frame_count": int(answer_materialized.frames.shape[0]),
            "state_query_visual_token_count": int(state_materialized.pixel_values_videos.shape[0]),
            "answer_query_visual_token_count": int(
                answer_materialized.pixel_values_videos.shape[0]
            ),
            "state_query_video_grid_thw": [
                int(value) for value in state_materialized.video_grid_thw[0].tolist()
            ],
            "answer_query_video_grid_thw": [
                int(value) for value in answer_materialized.video_grid_thw[0].tolist()
            ],
            "state_query_timestamp_range": [
                float(state_materialized.frame_timestamps[0].item()),
                float(state_materialized.frame_timestamps[-1].item()),
            ],
            "answer_query_timestamp_range": [
                float(answer_materialized.frame_timestamps[0].item()),
                float(answer_materialized.frame_timestamps[-1].item()),
            ],
            "released": result.release_audit is not None,
            "runtime_unchanged_during_generate": (
                result.generate_audit.state_before == result.generate_audit.state_after
            ),
        },
    }
    output_path = cast(Path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
