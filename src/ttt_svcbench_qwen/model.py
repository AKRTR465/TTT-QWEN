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
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Protocol, cast

from torch import nn

from ttt_svcbench_qwen.config import ProjectConfig


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
    query_input: object
    runtime_state: object
    bank_states: tuple[object, ...]
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
    runtime_state: object
    bank_states: tuple[object, ...]
    audit: object
    soft_write: object | None = None


@dataclass(frozen=True, slots=True)
class SoftIntermediates:
    adapted_visual: object
    query: object
    spatial: object | None
    temporal: object | None
    observations: object | None
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
    query: object
    spatial: object | None
    temporal: object | None
    observations: object | None
    commit_guard: ObservationCommitGuard


@dataclass(frozen=True, slots=True)
class ObservationChunkOutput:
    owner: RuntimeOwner
    visual: VisualStageOutput
    query: object
    spatial: object | None
    temporal: object | None
    observations: object | None
    runtime_state: object
    bank_states: tuple[object, ...]
    state_audit: object | None
    soft_intermediates: SoftIntermediates
    lifecycle: LifecycleAudit


@dataclass(frozen=True, slots=True)
class AnswerQueryRequest:
    owner: RuntimeOwner
    observation: ObservationChunkOutput
    base_input_ids: object
    base_attention_mask: object
    pixel_values_videos: object
    video_grid_thw: object
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

    input_ids: object
    attention_mask: object
    pixel_values_videos: object
    video_grid_thw: object
    prepared_video_features: object
    state_position_mask: object | None
    state_tokens: object | None
    composer_position_ids_audit: object
    composer_rope_deltas_audit: object
    qwen_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class StateAudit:
    observation: object | None
    retrieval: object | None
    reader: tuple[object, ...]
    resampler: object | None


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
    answer_logits: object
    qwen_output: object
    visual: VisualStageOutput
    query: object
    spatial: object | None
    temporal: object | None
    observations: object | None
    retrieval: object | None
    reader: tuple[object, ...]
    resampler: object | None
    composed: object
    prefill_request: QwenPrefillRequest
    runtime_state: object
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
    def __call__(self, query_input: object, *, inference: bool) -> object: ...


class FastStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput: ...


class SpatialStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> object: ...


class TemporalStage(Protocol):
    def __call__(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> object: ...


class ObservationStage(Protocol):
    def __call__(
        self,
        spatial: object,
        temporal: object,
        query: object,
        request: ObservationChunkRequest,
    ) -> object: ...


class BankWriter(Protocol):
    def __call__(
        self,
        observations: object,
        spatial: object,
        temporal: object,
        query: object,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput: ...


class RetrieverStage(Protocol):
    def retrieve_query(
        self,
        state_bank: object,
        states: Sequence[object],
        query: object,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> object: ...


class ReaderStage(Protocol):
    def read(self, retrieval: object) -> Sequence[object]: ...

    def audit_results(
        self,
        retrieval: object,
        results: Sequence[object],
    ) -> Sequence[object]: ...

    def audit_number_tokens(self, result: object) -> int | None: ...


class ExactCountResult(Protocol):
    exact_count: int | None


class ResamplerStage(Protocol):
    def __call__(self, q_target: object, retrieval: object) -> object: ...


class ComposerStage(Protocol):
    def __call__(
        self,
        *,
        base_input_ids: object,
        base_attention_mask: object,
        state_tokens: object | None,
        state_token_valid_mask: object | None,
        reader_results: Sequence[object],
        tokenizer: object,
        embedding_owner: object,
        rope_indexer: object,
        video_grid_thw: object,
        include_state: bool,
        include_number: bool,
    ) -> object: ...


class QwenPrefillStage(Protocol):
    def __call__(self, request: QwenPrefillRequest) -> object: ...


class QwenDecodeStage(Protocol):
    def __call__(self, model_inputs: object) -> object: ...


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
    state_bank: object | None = None
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
        if flags.fast_enabled and not callable(self.fast_adapter):
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
                if value is None or (name != "state_bank" and not callable(value))
            )
        if (flags.reader_enabled or flags.state_tokens_enabled) and (
            self.retriever is None or not callable(getattr(self.retriever, "retrieve_query", None))
        ):
            missing.append("retriever")
        if flags.reader_enabled:
            reader_methods = ("read", "audit_results", "audit_number_tokens")
            if self.reader is None or any(
                not callable(getattr(self.reader, name, None)) for name in reader_methods
            ):
                missing.append("reader")
        if flags.state_tokens_enabled and not callable(self.resampler):
            missing.append("resampler")
        if missing:
            raise ValueError(
                "enabled model features have missing dependencies: "
                + ", ".join(dict.fromkeys(missing))
            )


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
            fast_adapter = cast(FastStage, self.components.fast_adapter)
            adapted = fast_adapter(visual, query, request)
            if not isinstance(adapted, VisualStageOutput):
                raise TypeError("fast_adapter must return VisualStageOutput")

        spatial: object | None = None
        temporal: object | None = None
        observations: object | None = None
        if self.feature_flags.bank_enabled:
            spatial_encoder = cast(SpatialStage, self.components.spatial_encoder)
            temporal_encoder = cast(TemporalStage, self.components.temporal_encoder)
            heads = cast(ObservationStage, self.components.observation_heads)
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
                writer = cast(BankWriter, self.components.bank_writer)
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
            retrieval: object | None = None
            reader_results: tuple[object, ...] = ()
            resampler_output: object | None = None
            if self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled:
                retriever = cast(RetrieverStage, self.components.retriever)
                retrieval = retriever.retrieve_query(
                    self.components.state_bank,
                    observation.bank_states,
                    observation.query,
                    video_ids=request.owner.video_ids,
                    trajectory_ids=request.owner.trajectory_ids,
                )

            if self.feature_flags.reader_enabled:
                reader = cast(ReaderStage, self.components.reader)
                computed = tuple(reader.read(retrieval))
                audited = tuple(reader.audit_results(retrieval, computed))
                if audited != computed:
                    raise ValueError("Reader audit must return the unchanged authoritative results")
                for result in audited:
                    reader.audit_number_tokens(result)
                reader_results = audited

            if self.feature_flags.state_tokens_enabled:
                q_target = _required_attribute(observation.query, "q_target", "query output")
                resampler = cast(ResamplerStage, self.components.resampler)
                resampler_output = resampler(q_target, retrieval)
                _validate_answer_provenance(retrieval, reader_results, resampler_output)

            state_tokens = _optional_attribute(resampler_output, "state_tokens")
            state_token_valid_mask = _optional_attribute(
                resampler_output,
                "state_token_valid_mask",
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
                input_ids=_required_attribute(composed, "input_ids", "Composer output"),
                attention_mask=_required_attribute(
                    composed,
                    "attention_mask",
                    "Composer output",
                ),
                pixel_values_videos=request.pixel_values_videos,
                video_grid_thw=request.video_grid_thw,
                prepared_video_features=observation.visual.prepared_video_features,
                state_position_mask=_optional_attribute(composed, "state_position_mask"),
                state_tokens=state_tokens,
                composer_position_ids_audit=_required_attribute(
                    composed,
                    "position_ids",
                    "Composer output",
                ),
                composer_rope_deltas_audit=_required_attribute(
                    composed,
                    "rope_deltas",
                    "Composer output",
                ),
                qwen_kwargs=request.qwen_kwargs,
            )
            qwen_output = self.components.qwen_prefill(prefill_request)
            answer_logits = _required_attribute(qwen_output, "logits", "Qwen prefill output")
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
                    retrieval=_optional_attribute(retrieval, "audit"),
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
    reader_results: Sequence[object],
    predicted_numbers: Sequence[int | None],
) -> NumberAgreementMetrics:
    """Compare externally parsed answer integers without changing Reader results."""

    results = tuple(reader_results)
    predictions = tuple(predicted_numbers)
    if len(results) != len(predictions):
        raise ValueError("Reader results and predicted numbers must have equal batch size")
    matched = mismatched = missing = comparable = 0
    for result, predicted in zip(results, predictions, strict=True):
        if not hasattr(result, "exact_count"):
            raise TypeError("Reader result must expose exact_count")
        exact = cast(ExactCountResult, result).exact_count
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
    reader_results: Sequence[object],
    target_numbers: Sequence[int | None],
) -> None:
    """Block final-expression supervision whose integer target disagrees with Reader."""

    metrics = evaluate_number_agreement(reader_results, target_numbers)
    if metrics.mismatched_rows or metrics.missing_rows:
        raise ValueError("answer supervision number must equal the authoritative Reader number")


def _required_attribute(value: object, name: str, label: str) -> object:
    result = getattr(value, name, None)
    if result is None:
        raise TypeError(f"{label} must expose {name}")
    return cast(object, result)


def _optional_attribute(value: object | None, name: str) -> object | None:
    return None if value is None else cast(object | None, getattr(value, name, None))


def _validate_answer_provenance(
    retrieval: object | None,
    reader_results: tuple[object, ...],
    resampler: object,
) -> None:
    """Check IDs/status provenance only; arithmetic remains wholly Reader-owned."""

    retrieval_ids = _optional_attribute(retrieval, "selected_record_ids")
    resampler_ids = _optional_attribute(resampler, "selected_record_ids")
    if retrieval_ids is not None and resampler_ids is not None and retrieval_ids != resampler_ids:
        raise ValueError("Resampler must consume the same Retriever selected-record snapshot")
    if retrieval_ids is not None and reader_results:
        reader_ids = tuple(
            _required_attribute(result, "selected_record_ids", "Reader result")
            for result in reader_results
        )
        if reader_ids != retrieval_ids:
            raise ValueError("Reader results must preserve Retriever selected-record IDs")
    retrieval_status = _optional_attribute(retrieval, "status")
    resampler_status = _optional_attribute(resampler, "retrieval_status")
    if (
        retrieval_status is not None
        and resampler_status is not None
        and retrieval_status != resampler_status
    ):
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
