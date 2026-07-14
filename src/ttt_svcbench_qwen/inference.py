"""Run the causal, per-video P18 inference protocol.

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

from ttt_svcbench_qwen.config import InnerSGDConfig
from ttt_svcbench_qwen.data import RUNTIME_DENYLIST, assert_runtime_payload_safe
from ttt_svcbench_qwen.fast_ttt import (
    FastTTTAdapter,
    FastTTTForwardAudit,
    FastWeightsState,
    OptimizerRuntimeState,
)
from ttt_svcbench_qwen.functional_sgd import reset_optimizer_state
from ttt_svcbench_qwen.identity_bank import IdentityBank, IdentityBankRuntimeState
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    DecodeStepOutput,
    DecodeStepRequest,
    LifecyclePhase,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
)
from ttt_svcbench_qwen.observation_heads import E1RuntimeState, E2RuntimeState
from ttt_svcbench_qwen.stage_a_runtime import StageABatchRuntime
from ttt_svcbench_qwen.state_bank import StateBankRuntimeState, StructuredStateBank
from ttt_svcbench_qwen.state_encoder import SpatialSlotRuntimeState, TemporalCache
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
        )


@dataclass(frozen=True, slots=True)
class AnswerInputs:
    base_input_ids: object
    base_attention_mask: object
    pixel_values_videos: object
    video_grid_thw: object
    tokenizer: object
    embedding_owner: object
    rope_indexer: object
    qwen_kwargs: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    video_id: str
    trajectory_id: str
    payload: Mapping[str, object]
    query_signature: Tensor
    chunks: tuple[CausalChunk, ...]
    answer_inputs: AnswerInputs
    attempt: QueryAttempt
    max_decode_steps: int = 128

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("inference request owner identifiers must be non-empty")
        assert_inference_runtime_payload(self.payload)
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
        if type(self.max_decode_steps) is not int or self.max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be a positive integer")

    @property
    def query_time(self) -> float:
        return _payload_query_time(self.payload)

    @property
    def query_input(self) -> Mapping[str, object]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class PerVideoRuntimeState:
    video_id: str
    trajectory_id: str
    fast_weights: FastWeightsState
    optimizer: OptimizerRuntimeState
    slot_state: SpatialSlotRuntimeState | None
    temporal_cache: TemporalCache
    e1_state: E1RuntimeState | None
    e2_state: E2RuntimeState | None
    state_bank: StateBankRuntimeState
    identity_bank: IdentityBankRuntimeState
    reader_audit: tuple[ReaderResult, ...]
    released: bool

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("per-video runtime identifiers must be non-empty")
        if type(self.released) is not bool:
            raise TypeError("per-video runtime released flag must be bool")
        if self.state_bank.video_id != self.video_id:
            raise ValueError("State Bank video_id does not match runtime ownership")
        if self.state_bank.trajectory_id != self.trajectory_id:
            raise ValueError("State Bank trajectory_id does not match runtime ownership")
        if self.state_bank.released != self.released:
            raise ValueError("State Bank and per-video release state must agree")
        if self.identity_bank.video_id != self.video_id:
            raise ValueError("Identity Bank video_id does not match runtime ownership")
        if self.identity_bank.trajectory_id != self.trajectory_id:
            raise ValueError("Identity Bank trajectory_id does not match runtime ownership")
        if self.identity_bank.released != self.released:
            raise ValueError("Identity Bank and per-video release state must agree")
        if self.slot_state is not None and self.slot_state.video_id != self.video_id:
            raise ValueError("spatial slot state video_id does not match runtime ownership")
        if self.temporal_cache.hidden.shape[0] != 1:
            raise ValueError("per-video temporal cache must have batch size 1")
        if self.temporal_cache.video_ids != (self.video_id,):
            raise ValueError("temporal cache video_ids do not match runtime ownership")
        if self.temporal_cache.trajectory_ids != (self.trajectory_id,):
            raise ValueError("temporal cache trajectory_ids do not match runtime ownership")
        for name, state in (("E1", self.e1_state), ("E2", self.e2_state)):
            if state is None:
                continue
            if state.video_id != self.video_id or state.trajectory_id != self.trajectory_id:
                raise ValueError(f"{name} state does not match runtime ownership")
            if (
                state.query_signature.dtype != self.temporal_cache.hidden.dtype
                or state.query_signature.device != self.temporal_cache.hidden.device
                or not state.query_signature.equal(self.temporal_cache.query_signatures[0])
            ):
                raise ValueError(f"{name} state query signature does not match temporal cache")
            if state.total_seen != int(self.temporal_cache.total_seen[0].item()):
                raise ValueError(f"{name} state position does not match temporal cache")


@dataclass(frozen=True, slots=True)
class RuntimeResetAudit:
    video_id: str
    trajectory_id: str
    previous_runtime_checksum: str | None
    previous_release_checksum: str | None
    reset_runtime_checksum: str
    pristine_state_checksum: str
    w0_checksum: str
    current_fast_checksum: str

    def __post_init__(self) -> None:
        checksums = (
            self.reset_runtime_checksum,
            self.pristine_state_checksum,
            self.w0_checksum,
            self.current_fast_checksum,
        )
        if any(len(value) != 64 for value in checksums):
            raise ValueError("runtime reset audit requires SHA-256 checksums")
        if self.w0_checksum != self.current_fast_checksum:
            raise ValueError("fresh runtime W_t must equal W0")


@dataclass(frozen=True, slots=True)
class RuntimeReleaseAudit:
    video_id: str
    trajectory_id: str
    before_checksum: str
    released_checksum: str
    state_bank_version: int
    identity_bank_version: int
    runtime_state: PerVideoRuntimeState

    def __post_init__(self) -> None:
        if not self.runtime_state.released:
            raise ValueError("release audit must carry a released runtime state")
        if (self.runtime_state.video_id, self.runtime_state.trajectory_id) != (
            self.video_id,
            self.trajectory_id,
        ):
            raise ValueError("release audit runtime owner is inconsistent")
        if runtime_checksum(self.runtime_state) != self.released_checksum:
            raise ValueError("release audit checksum does not match released runtime")


@dataclass(frozen=True, slots=True)
class TTTUpdateOutcome:
    runtime_state: PerVideoRuntimeState
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
        runtime_state: PerVideoRuntimeState,
    ) -> TTTUpdateOutcome: ...


class GenerationDriver(Protocol):
    """Own token selection/KV inputs while P18 owns state immutability."""

    def begin(self, prefill: StateTTTModelOutput) -> object | None: ...

    def advance(self, step_index: int, decode: DecodeStepOutput) -> object | None: ...

    def finish(
        self,
        prefill: StateTTTModelOutput,
        decode_steps: Sequence[DecodeStepOutput],
    ) -> str: ...


class InferenceRuntimeBridge(Protocol):
    """Translate the canonical P18 state to one model component runtime contract."""

    def to_model_runtime(
        self,
        runtime: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> object: ...

    def bank_states(self, runtime: PerVideoRuntimeState) -> tuple[object, ...]: ...

    def from_model_runtime(
        self,
        value: object,
        previous: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> PerVideoRuntimeState: ...


class PerVideoRuntimeBridge:
    """Identity bridge for P18-native model components and test doubles."""

    def to_model_runtime(
        self,
        runtime: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> PerVideoRuntimeState:
        _require_per_video_runtime(runtime, owner)
        return runtime

    @staticmethod
    def bank_states(runtime: PerVideoRuntimeState) -> tuple[object, ...]:
        return (runtime.state_bank,)

    def from_model_runtime(
        self,
        value: object,
        _previous: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> PerVideoRuntimeState:
        return _require_per_video_runtime(value, owner)


class StageARuntimeBridge:
    """Bridge the real P15 hard writer to P18's fast/optimizer-owned state."""

    def to_model_runtime(
        self,
        runtime: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> StageABatchRuntime:
        _require_per_video_runtime(runtime, owner)
        return StageABatchRuntime(
            owner=owner,
            next_chunk_index=runtime.optimizer.attempted_update_count,
            slot_states=(runtime.slot_state,),
            temporal_cache=runtime.temporal_cache,
            e1_states=(runtime.e1_state,),
            e2_states=(runtime.e2_state,),
            state_bank_states=(runtime.state_bank,),
            identity_bank_states=(runtime.identity_bank,),
        )

    @staticmethod
    def bank_states(runtime: PerVideoRuntimeState) -> tuple[object, ...]:
        return (runtime.state_bank,)

    def from_model_runtime(
        self,
        value: object,
        previous: PerVideoRuntimeState,
        owner: RuntimeOwner,
    ) -> PerVideoRuntimeState:
        if not isinstance(value, StageABatchRuntime):
            raise TypeError("Stage A runtime bridge requires StageABatchRuntime model output")
        if value.owner != owner or len(value.state_bank_states) != 1:
            raise InferenceProtocolError("Stage A runtime output owner/batch does not match P18")
        expected_chunk = previous.optimizer.attempted_update_count + 1
        if value.next_chunk_index != expected_chunk:
            raise InferenceProtocolError("Stage A runtime chunk index does not match P18 attempts")
        if value.temporal_cache is None:
            raise InferenceProtocolError("Stage A runtime must return the temporal cache")
        return PerVideoRuntimeState(
            video_id=previous.video_id,
            trajectory_id=previous.trajectory_id,
            fast_weights=previous.fast_weights,
            optimizer=previous.optimizer,
            slot_state=value.slot_states[0],
            temporal_cache=value.temporal_cache,
            e1_state=value.e1_states[0],
            e2_state=value.e2_states[0],
            state_bank=value.state_bank_states[0],
            identity_bank=value.identity_bank_states[0],
            reader_audit=previous.reader_audit,
            released=False,
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
    state_checksum_before: str
    state_checksum_after_observe: str
    state_checksum_after_update: str

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
    runtime_state: PerVideoRuntimeState
    audit: ChunkAudit


@dataclass(frozen=True, slots=True)
class GenerateAudit:
    query_id: str
    query_kind: QueryAttemptKind
    retry_of: str | None
    prefill_count: int
    decode_count: int
    state_checksum_before_prefill: str
    state_checksum_after_prefill: str
    decode_state_checksums: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.prefill_count != 1:
            raise ValueError("one inference query must execute exactly one prefill")
        if self.decode_count != len(self.decode_state_checksums):
            raise ValueError("decode audit count/checksums must align")
        if self.state_checksum_before_prefill != self.state_checksum_after_prefill:
            raise ValueError("query read/composition/prefill cannot mutate runtime state")
        if any(value != self.state_checksum_after_prefill for value in self.decode_state_checksums):
            raise ValueError("autoregressive decode cannot mutate runtime state")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    answer_text: str
    reader_result: ReaderResult
    runtime_state: PerVideoRuntimeState
    selected_record_ids: tuple[str, ...]
    state_attention: Tensor | None
    reset_audit: RuntimeResetAudit
    chunk_audit: tuple[ChunkAudit, ...]
    generate_audit: GenerateAudit
    release_audit: RuntimeReleaseAudit | None
    audit_fields: tuple[tuple[str, AuditValue], ...]

    def __post_init__(self) -> None:
        if not self.answer_text:
            raise ValueError("inference answer_text must be non-empty")
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
        hot_cache_enabled: bool = False,
        hot_device: str | torch.device | None = None,
        runtime_bridge: InferenceRuntimeBridge | None = None,
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
        self.fast_adapter = fast_adapter
        self.state_bank = state_bank
        self.identity_bank = identity_bank
        self.optimizer_config = optimizer_config
        self.hot_cache_enabled = hot_cache_enabled
        self.hot_device = hot_device
        self.runtime_bridge = runtime_bridge or PerVideoRuntimeBridge()
        self._runtime: PerVideoRuntimeState | None = None
        self._lifecycle: PrefillLifecycle | None = None
        self._reset_audit: RuntimeResetAudit | None = None
        self._chunk_audits: list[ChunkAudit] = []
        self._query_snapshots: dict[str, str] = {}
        self._lock = RLock()

    @property
    def active_runtime(self) -> PerVideoRuntimeState | None:
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
            previous_checksum = None
            previous_release_checksum = None
            if self._runtime is not None:
                previous_checksum = runtime_checksum(self._runtime)
                prior_release = self._release_locked()
                previous_release_checksum = prior_release.released_checksum

            owner = RuntimeOwner((video_id,), (trajectory_id,))
            dtype = self.fast_adapter.w0_1.dtype
            device = self.fast_adapter.w0_1.device
            signature = query_signature.detach().to(device=device, dtype=dtype).clone()
            fast = self.fast_adapter.reset_fast_state(differentiable=False)
            runtime = PerVideoRuntimeState(
                video_id=video_id,
                trajectory_id=trajectory_id,
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
                reader_audit=(),
                released=False,
            )
            w0_checksum = _fast_pair_checksum(fast.w0_1, fast.w0_2)
            current_checksum = _fast_pair_checksum(fast.w_t_1, fast.w_t_2)
            audit = RuntimeResetAudit(
                video_id=video_id,
                trajectory_id=trajectory_id,
                previous_runtime_checksum=previous_checksum,
                previous_release_checksum=previous_release_checksum,
                reset_runtime_checksum=runtime_checksum(runtime),
                pristine_state_checksum=_pristine_state_checksum(runtime),
                w0_checksum=w0_checksum,
                current_fast_checksum=current_checksum,
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
        query_input: object,
        query_time: float,
        updater: TTTUpdateStage,
    ) -> ChunkExecution:
        """Observe with W_t, write hard state, then create W_(t+1) for the next chunk."""

        with self._lock:
            runtime = self._require_live_runtime()
            lifecycle = self._require_ready_lifecycle()
            before_checksum = runtime_checksum(runtime)
            before_fast_checksum = _fast_state_checksum(runtime.fast_weights)
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
                    fast_version_used=runtime.fast_weights.fast_version,
                    next_fast_version=runtime.fast_weights.fast_version,
                    update_attempted=False,
                    did_update=False,
                    skip_reason="no_causal_frames",
                    valid_term_count=0,
                    state_checksum_before=before_checksum,
                    state_checksum_after_observe=before_checksum,
                    state_checksum_after_update=before_checksum,
                )
                self._chunk_audits.append(audit)
                return ChunkExecution(None, runtime, audit)

            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            model_runtime = self.runtime_bridge.to_model_runtime(runtime, owner)
            model_bank_states = self.runtime_bridge.bank_states(runtime)
            if len(model_bank_states) != 1 or model_bank_states[0] is not runtime.state_bank:
                raise InferenceProtocolError(
                    "runtime bridge must preserve the authoritative State Bank object"
                )
            if not model.feature_flags.fast_enabled:
                raise InferenceProtocolError("P18 requires the managed Fast Adapter path")
            self.fast_adapter.last_audit = None
            with self.fast_adapter.use_fast_state(runtime.fast_weights):
                observation = model.observe_chunk(
                    ObservationChunkRequest(
                        owner=owner,
                        video_input=causal,
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
                    "P18 observe did not consume the manager-bound FastWeightsState"
                )
            expected_version = (runtime.fast_weights.fast_version,)
            expected_updates = (runtime.fast_weights.update_count,)
            if (
                fast_audit.fast_versions != expected_version
                or fast_audit.update_counts != expected_updates
                or len(fast_audit.valid_token_counts) != 1
            ):
                raise InferenceProtocolError(
                    "Fast Adapter audit owner/version does not match the per-video runtime"
                )
            observed = self.runtime_bridge.from_model_runtime(
                observation.runtime_state,
                runtime,
                owner,
            )
            observation = replace(
                observation,
                runtime_state=observed,
                bank_states=(observed.state_bank,),
            )
            if _fast_state_checksum(observed.fast_weights) != before_fast_checksum:
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
            after_observe_checksum = runtime_checksum(observed)
            hard_checksum = _hard_state_checksum(observed)
            outcome = updater(observation, observed)
            updated = _require_per_video_runtime(outcome.runtime_state, owner)
            if _hard_state_checksum(updated) != hard_checksum:
                raise InferenceProtocolError(
                    "TTT updater may only change fast/optimizer runtime state"
                )
            _validate_update_transition(observed, outcome)
            after_update_checksum = runtime_checksum(updated)
            next_observation = replace(
                observation,
                runtime_state=updated,
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
                fast_version_used=observed.fast_weights.fast_version,
                next_fast_version=updated.fast_weights.fast_version,
                update_attempted=True,
                did_update=outcome.did_update,
                skip_reason=outcome.skip_reason,
                valid_term_count=outcome.valid_term_count,
                state_checksum_before=before_checksum,
                state_checksum_after_observe=after_observe_checksum,
                state_checksum_after_update=after_update_checksum,
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
        generation_driver: GenerationDriver,
        max_decode_steps: int,
    ) -> InferenceResult:
        """Read/compose/prefill once and prove every decode step leaves state immutable."""

        if type(max_decode_steps) is not int or max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be a positive integer")
        with self._lock:
            runtime = self._require_live_runtime()
            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            if observation.owner != owner:
                raise InferenceProtocolError("answer observation owner does not match runtime")
            observation = replace(
                observation,
                runtime_state=runtime,
                bank_states=(runtime.state_bank,),
            )
            causal_checksum = _causal_state_checksum(runtime)
            self._register_query_attempt(attempt, causal_checksum)
            lifecycle = self._query_lifecycle(owner)
            before_prefill = runtime_checksum(runtime)
            prefill = model.answer_query(
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
            after_prefill = runtime_checksum(runtime)
            if after_prefill != before_prefill:
                raise InferenceProtocolError("query read/composition/prefill mutated runtime state")
            if len(prefill.reader) != 1 or not isinstance(prefill.reader[0], ReaderResult):
                raise InferenceProtocolError("P18 requires one authoritative ReaderResult")
            reader_result = prefill.reader[0]

            decode_steps: list[DecodeStepOutput] = []
            decode_checksums: list[str] = []
            next_inputs = generation_driver.begin(prefill)
            if runtime_checksum(runtime) != after_prefill:
                raise InferenceProtocolError("generation begin mutated runtime state")
            while next_inputs is not None:
                if len(decode_steps) >= max_decode_steps:
                    raise InferenceProtocolError("generation exceeded max_decode_steps")
                before_decode = runtime_checksum(runtime)
                decoded = model.decode_step(
                    DecodeStepRequest(owner=owner, model_inputs=next_inputs),
                    lifecycle,
                )
                after_decode = runtime_checksum(runtime)
                if after_decode != before_decode:
                    raise InferenceProtocolError(
                        "decode step mutated Bank/FSM/fast/cache runtime state"
                    )
                decode_steps.append(decoded)
                next_inputs = generation_driver.advance(len(decode_steps) - 1, decoded)
                after_driver = runtime_checksum(runtime)
                if after_driver != after_prefill:
                    raise InferenceProtocolError("decode/generation driver mutated runtime state")
                decode_checksums.append(after_driver)

            answer_text = generation_driver.finish(prefill, tuple(decode_steps))
            if runtime_checksum(runtime) != after_prefill:
                raise InferenceProtocolError("generation finish mutated runtime state")
            state_attention = _state_attention(prefill.resampler)
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
                state_checksum_before_prefill=before_prefill,
                state_checksum_after_prefill=after_prefill,
                decode_state_checksums=tuple(decode_checksums),
            )
            result_runtime = self._runtime
            return InferenceResult(
                answer_text=answer_text,
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
                    ("final_fast_version", runtime.fast_weights.fast_version),
                    ("final_update_count", runtime.fast_weights.update_count),
                    ("final_skip_count", runtime.fast_weights.skip_count),
                ),
            )

    def release(self) -> RuntimeReleaseAudit | None:
        """Release all trajectory storage; safe and idempotent for exception cleanup."""

        with self._lock:
            if self._runtime is None:
                return None
            return self._release_locked()

    def _release_locked(self) -> RuntimeReleaseAudit:
        runtime = self._require_live_runtime()
        before_checksum = runtime_checksum(runtime)
        released_bank = self.state_bank.release(runtime.state_bank)
        released_identity = self.identity_bank.release(runtime.identity_bank)
        owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
        released = PerVideoRuntimeState(
            video_id=runtime.video_id,
            trajectory_id=runtime.trajectory_id,
            fast_weights=_released_fast_state(runtime.fast_weights.w0_1.dtype),
            optimizer=reset_optimizer_state(self.optimizer_config),
            slot_state=None,
            temporal_cache=_empty_temporal_cache(
                owner,
                torch.empty((512,), dtype=runtime.temporal_cache.hidden.dtype, device="meta"),
            ),
            e1_state=None,
            e2_state=None,
            state_bank=released_bank,
            identity_bank=released_identity,
            reader_audit=(),
            released=True,
        )
        audit = RuntimeReleaseAudit(
            video_id=runtime.video_id,
            trajectory_id=runtime.trajectory_id,
            before_checksum=before_checksum,
            released_checksum=runtime_checksum(released),
            state_bank_version=released_bank.version,
            identity_bank_version=released_identity.version,
            runtime_state=released,
        )
        self._runtime = None
        self._lifecycle = None
        return audit

    def _register_query_attempt(self, attempt: QueryAttempt, checksum: str) -> None:
        if attempt.query_id in self._query_snapshots:
            raise InferenceProtocolError("query_id has already been used in this runtime")
        if attempt.kind is QueryAttemptKind.RETRY:
            expected = self._query_snapshots.get(cast(str, attempt.retry_of))
            if expected is None:
                raise InferenceProtocolError("retry_of does not name a completed query")
            if expected != checksum:
                raise InferenceProtocolError("retry requires the unchanged causal runtime snapshot")
        self._query_snapshots[attempt.query_id] = checksum

    def _query_lifecycle(self, owner: RuntimeOwner) -> PrefillLifecycle:
        lifecycle = self._lifecycle
        if lifecycle is None:
            raise InferenceProtocolError("runtime lifecycle is unavailable")
        if lifecycle.audit().phase is not LifecyclePhase.READY:
            lifecycle = PrefillLifecycle(owner)
            self._lifecycle = lifecycle
        return lifecycle

    def _require_live_runtime(self) -> PerVideoRuntimeState:
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
    generation_driver: GenerationDriver,
) -> InferenceResult:
    """Execute reset -> causal chunks -> one prefill/decode -> unconditional release."""

    if not isinstance(manager, PerVideoRuntimeManager):
        raise TypeError("run_inference requires PerVideoRuntimeManager")
    if not isinstance(model, StateTTTModel):
        raise TypeError("run_inference requires StateTTTModel")
    if not isinstance(request, InferenceRequest):
        raise TypeError("run_inference requires InferenceRequest")
    assert_inference_runtime_payload(request.payload)
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
        if latest_observation is None:
            raise InferenceProtocolError("no causal frame was available before query_time")
        result = manager.answer_query(
            model=model,
            observation=latest_observation,
            answer_inputs=request.answer_inputs,
            attempt=request.attempt,
            generation_driver=generation_driver,
            max_decode_steps=request.max_decode_steps,
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
                ("release_checksum", release_audit.released_checksum),
            ),
        )
    except Exception:
        manager.release()
        raise


def runtime_checksum(state: PerVideoRuntimeState) -> str:
    """Hash tensor values plus all typed hard/fast runtime metadata."""

    if not isinstance(state, PerVideoRuntimeState):
        raise TypeError("runtime_checksum requires PerVideoRuntimeState")
    digest = hashlib.sha256()
    _hash_value(digest, state)
    return digest.hexdigest()


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


def _require_per_video_runtime(value: object, owner: RuntimeOwner) -> PerVideoRuntimeState:
    if not isinstance(value, PerVideoRuntimeState):
        raise TypeError("P18 model/update stages must return PerVideoRuntimeState")
    if value.released:
        raise InferenceProtocolError("P18 stages cannot return a released runtime")
    if (value.video_id, value.trajectory_id) != (owner.video_ids[0], owner.trajectory_ids[0]):
        raise InferenceProtocolError("P18 stage returned runtime for a different owner")
    return value


def _validate_update_transition(before: PerVideoRuntimeState, outcome: TTTUpdateOutcome) -> None:
    after = outcome.runtime_state
    before_fast = before.fast_weights
    after_fast = after.fast_weights
    if _fast_pair_checksum(before_fast.w0_1, before_fast.w0_2) != _fast_pair_checksum(
        after_fast.w0_1,
        after_fast.w0_2,
    ):
        raise InferenceProtocolError("TTT updater cannot change the meta-learned W0 snapshot")
    optimizer_contract_before = (
        before.optimizer.optimizer_name,
        before.optimizer.learning_rate,
        before.optimizer.momentum,
        before.optimizer.weight_decay,
        before.optimizer.steps_per_chunk,
        before.optimizer.grad_clip_norm,
    )
    optimizer_contract_after = (
        after.optimizer.optimizer_name,
        after.optimizer.learning_rate,
        after.optimizer.momentum,
        after.optimizer.weight_decay,
        after.optimizer.steps_per_chunk,
        after.optimizer.grad_clip_norm,
    )
    if optimizer_contract_after != optimizer_contract_before:
        raise InferenceProtocolError("TTT updater cannot change the optimizer contract")
    if after.optimizer.attempted_update_count != before.optimizer.attempted_update_count + 1:
        raise InferenceProtocolError("one chunk must make exactly one optimizer update attempt")
    if outcome.did_update:
        expected = (
            before_fast.fast_version + 1,
            before_fast.update_count + 1,
            before_fast.skip_count,
        )
        actual = (after_fast.fast_version, after_fast.update_count, after_fast.skip_count)
        if actual != expected or after.optimizer.last_skip_reason is not None:
            raise InferenceProtocolError("accepted TTT update has inconsistent counters")
        if _fast_pair_checksum(before_fast.w_t_1, before_fast.w_t_2) == _fast_pair_checksum(
            after_fast.w_t_1,
            after_fast.w_t_2,
        ):
            raise InferenceProtocolError("accepted TTT update must change W_t values")
    else:
        expected = (
            before_fast.fast_version,
            before_fast.update_count,
            before_fast.skip_count + 1,
        )
        actual = (after_fast.fast_version, after_fast.update_count, after_fast.skip_count)
        if actual != expected or after.optimizer.last_skip_reason != outcome.skip_reason:
            raise InferenceProtocolError("skipped TTT update has inconsistent counters/reason")
        if _fast_pair_checksum(before_fast.w_t_1, before_fast.w_t_2) != _fast_pair_checksum(
            after_fast.w_t_1,
            after_fast.w_t_2,
        ):
            raise InferenceProtocolError("skipped TTT update cannot change W_t values")


def _state_attention(resampler: object | None) -> Tensor | None:
    value = None if resampler is None else getattr(resampler, "cross_attention_weights", None)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError("State Resampler attention audit must be a Tensor")
    return value.detach().clone()


def _causal_state_checksum(state: PerVideoRuntimeState) -> str:
    return runtime_checksum(replace(state, reader_audit=()))


def _hard_state_checksum(state: PerVideoRuntimeState) -> str:
    digest = hashlib.sha256()
    for value in (
        state.video_id,
        state.trajectory_id,
        state.slot_state,
        state.temporal_cache,
        state.e1_state,
        state.e2_state,
        state.state_bank,
        state.identity_bank,
        state.reader_audit,
        state.released,
    ):
        _hash_value(digest, value)
    return digest.hexdigest()


def _fast_state_checksum(state: FastWeightsState) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, state)
    return digest.hexdigest()


def _fast_pair_checksum(first: Tensor, second: Tensor) -> str:
    digest = hashlib.sha256()
    _hash_tensor_content(digest, first)
    _hash_tensor_content(digest, second)
    return digest.hexdigest()


def _pristine_state_checksum(state: PerVideoRuntimeState) -> str:
    """Owner/query-independent reset fingerprint used across consecutive videos."""

    digest = hashlib.sha256()
    values: tuple[object, ...] = (
        _fast_pair_checksum(state.fast_weights.w_t_1, state.fast_weights.w_t_2),
        state.fast_weights.fast_version,
        state.fast_weights.update_count,
        state.fast_weights.skip_count,
        state.optimizer,
        state.slot_state is None,
        state.temporal_cache.hidden.shape[1],
        int(state.temporal_cache.total_seen[0].item()),
        state.e1_state is None,
        state.e2_state is None,
        len(state.state_bank.records),
        len(state.state_bank.audit_log),
        state.state_bank.version,
        len(state.identity_bank.candidates),
        len(state.identity_bank.confirmed),
        len(state.identity_bank.hot_cache),
        state.identity_bank.version,
        len(state.reader_audit),
        state.released,
    )
    _hash_value(digest, values)
    return digest.hexdigest()


class _Digest(Protocol):
    def update(self, value: bytes) -> object: ...


def _hash_tensor_content(digest: _Digest, value: Tensor) -> None:
    """Hash numerical tensor content while ignoring leaf/gradient metadata."""

    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.device).encode("ascii"))
    if value.device.type != "meta":
        raw = value.detach().to(device="cpu").contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())


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
    parser = argparse.ArgumentParser(description="P18 causal inference protocol utilities")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--describe-protocol", action="store_true")
    action.add_argument("--validate-payload", type=Path, metavar="JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CPU-safe CLI for protocol discovery and runtime-payload validation."""

    args = _build_parser().parse_args(argv)
    if args.describe_protocol:
        print(
            json.dumps(
                {
                    "protocol": "P18",
                    "order": [
                        "reset",
                        "causal_observe",
                        "ttt_update",
                        "prefill",
                        "decode",
                        "release",
                    ],
                    "decode_mutates": ["llm_kv_cache"],
                    "runtime_payload_fields": [
                        "video",
                        "question",
                        "query_time",
                        "explicit_time_values",
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    path = cast(Path, args.validate_payload)
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("payload JSON must contain one object")
    payload = {str(key): value for key, value in raw.items()}
    assert_inference_runtime_payload(payload)
    print(
        json.dumps(
            {
                "safe": True,
                "query_time": _payload_query_time(payload),
                "question_length": len(cast(str, payload["question"])),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
