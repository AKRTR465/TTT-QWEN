"""Compose the P13 State-TTT stages without owning their algorithms.

Inputs: injected P3/P5-P12 components, immutable stage requests, and one explicit
per-owner prefill lifecycle.
Outputs: observation intermediates, one training prefill, or one generated answer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, TypeAlias, cast

from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    AssociativeTTTIntermediates,
    FastAssociativeContext,
    FastMemoryState,
    StateWriteSourceView,
    build_fast_associative_context,
)
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


class LifecycleError(RuntimeError):
    """Raised when an owner attempts an illegal observe/prefill/generate transition."""


class LifecyclePhase(StrEnum):
    READY = "ready"
    PREFILLED = "prefilled"
    DECODING = "decoding"


@dataclass(frozen=True, slots=True)
class RuntimeOwner:
    """Canonical batch ownership used by every P13 entrypoint."""

    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]


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
    fast_weights: FastMemoryState | None = None
    reader_audit: tuple[ReaderResult, ...] = ()
    released: bool = False

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
        return tuple(value for value in histories if value is not None)

    @property
    def bank_states(self) -> tuple[StateBankRuntimeState, ...]:
        return self.state_bank_states

    @property
    def fast_states(self) -> tuple[FastMemoryState, ...]:
        return tuple(row.fast_weights for row in self.rows if row.fast_weights is not None)

    def with_fast_states(
        self,
        fast_states: Sequence[FastMemoryState],
    ) -> BatchRuntimeState:
        fast = tuple(fast_states)
        return BatchRuntimeState(
            tuple(
                replace(row, fast_weights=fast_state)
                for row, fast_state in zip(self.rows, fast, strict=True)
            )
        )


@dataclass(frozen=True, slots=True)
class LifecycleAudit:
    owner: RuntimeOwner
    phase: LifecyclePhase
    observation_count: int
    prefill_count: int


@dataclass(slots=True)
class PrefillLifecycle:
    """Per-owner runtime state.

    This object is external runtime state.  It is intentionally not an ``nn.Module``
    parameter/buffer and must never be placed in a model checkpoint.
    """

    owner: RuntimeOwner
    phase: LifecyclePhase = LifecyclePhase.READY
    observation_count: int = 0
    prefill_count: int = 0

    def audit(self) -> LifecycleAudit:
        return LifecycleAudit(
            owner=self.owner,
            phase=self.phase,
            observation_count=self.observation_count,
            prefill_count=self.prefill_count,
        )

    def _observed(self) -> None:
        self.observation_count += 1

    def _prefilled(self) -> None:
        self.prefill_count += 1
        self.phase = LifecyclePhase.PREFILLED


@dataclass(frozen=True, slots=True)
class ModelFeatureFlags:
    fast_enabled: bool = True
    bank_enabled: bool = True
    reader_enabled: bool = True
    state_tokens_enabled: bool = True


@dataclass(frozen=True, slots=True)
class PreparedQueryOutput:
    """One explicitly scoped Query graph; never stored beyond its caller-owned episode."""

    key: tuple[int, str, str]
    value: QueryEncoderOutput

    @classmethod
    def bind(cls, query: RuntimeQueryInput, value: QueryEncoderOutput) -> PreparedQueryOutput:
        return cls(query_reuse_key(query), value)

    def detached(self) -> PreparedQueryOutput:
        return PreparedQueryOutput(self.key, detach_query_encoder_output(self.value))


def query_reuse_key(query: RuntimeQueryInput) -> tuple[int, str, str]:
    return (query.episode_nonce, query.query_id, query.question)


def query_dropout_seed(query: RuntimeQueryInput) -> int:
    # The temporal cache is owned by one question signature.  A5 may expose several
    # causal Query points for that same question, each with a different query_id/time.
    # They must therefore receive the same dropout mask inside one episode; otherwise
    # question-only q_target drifts even though the semantic Query is unchanged.
    encoded = f"{query.episode_nonce}:{query.question}".encode()
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
    soft_write: StateWriteSourceView | None = None


@dataclass(frozen=True, slots=True)
class SoftIntermediates:
    adapted_visual: object
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput | None
    temporal: TemporalEncoderOutput | None
    observations: ObservationOutputs | None
    state_write: StateWriteSourceView | None = None
    fast_associative: AssociativeTTTIntermediates | None = None


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


@dataclass(frozen=True, slots=True)
class QwenPrefillRequest:
    """Fields consumed by the P3 adapter for one native-HF prefill.

    Production Qwen receives IDs/masks/pixels and computes/caches its own multimodal
    positions.  This request never asks Qwen to consume Composer ``inputs_embeds``.
    """

    input_ids: Tensor
    attention_mask: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    state_position_mask: Tensor | None
    state_tokens: Tensor | None
    qwen_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class QwenGenerateRequest:
    prefill: QwenPrefillRequest
    max_new_tokens: int = 16


@dataclass(frozen=True, slots=True)
class QwenGenerateOutput:
    answer_text: str
    token_ids: Tensor


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


class QwenPrefillOutput(Protocol):
    logits: Tensor


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


# Orchestration seams.  The authoritative component implementations own every
# numerical or hard state rule; these aliases only name the call signatures.
VisualStage: TypeAlias = Callable[..., VisualStageOutput]
QueryStage: TypeAlias = Callable[..., QueryEncoderOutput]
FastStage: TypeAlias = Callable[..., VisualStageOutput]
SpatialStage: TypeAlias = Callable[..., SpatialEncoderOutput]
TemporalStage: TypeAlias = Callable[..., TemporalEncoderOutput]
ObservationStage: TypeAlias = Callable[..., ObservationOutputs]
BankWriter: TypeAlias = Callable[..., BankWriteOutput]
RetrieverStage: TypeAlias = Callable[..., RetrieverOutput]
ResamplerStage: TypeAlias = Callable[..., StateResamplerOutput]
ComposerStage: TypeAlias = Callable[..., ComposedInput]
QwenPrefillStage: TypeAlias = Callable[..., QwenPrefillOutput]
QwenGenerateStage: TypeAlias = Callable[..., QwenGenerateOutput]


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


class StateTTTModel(nn.Module):  # type: ignore[misc]
    """Dependency-injected P13 orchestrator with no numerical implementation."""

    def __init__(
        self,
        config: ProjectConfig,
        components: ModelComponents,
        feature_flags: ModelFeatureFlags,
    ) -> None:
        super().__init__()
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
        """Run the soft observation stages and execute the sole hard Bank write."""

        query = (
            request.prepared_query.value
            if request.prepared_query is not None
            else self.components.query_encoder(request.query_input, inference=request.inference)
        )
        fast_adapter = self.components.fast_adapter
        associative_context = self._build_associative_context(query, request.bank_states)
        with _bind_associative_context(fast_adapter, associative_context):
            visual = self.components.visual_stage(request)
            adapted = visual
            if self.feature_flags.fast_enabled and fast_adapter is not None:
                adapted = fast_adapter(visual, query, request)
        fast_associative = self._consume_associative_intermediates(fast_adapter)

        spatial: SpatialEncoderOutput | None = None
        temporal: TemporalEncoderOutput | None = None
        observations: ObservationOutputs | None = None
        if self.feature_flags.bank_enabled:
            spatial = self.components.spatial_encoder(adapted, query, request)
            temporal = self.components.temporal_encoder(adapted, query, request)
            observations = self.components.observation_heads(spatial, temporal, query, request)

        runtime_state = request.runtime_state
        bank_states = request.bank_states
        retrieval_history: RetrievalHistoryView | None = None
        bank_audit: object | None = None
        soft_write: StateWriteSourceView | None = None
        if self.feature_flags.bank_enabled:
            if request.retrieval_snapshot_required and (
                self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled
            ):
                # Keep the write-before snapshot label-free and expose every head. Runtime
                # selection is still constrained by the predicted hard operator inside the
                # Retriever; official labels may only build a target-head MIL bag later.
                with trace_cuda_phase("retrieval_history_snapshot"):
                    retrieval_history = tensorized_retrieval_view(
                        request.runtime_state.retrieval_histories
                    )
            write = self.components.bank_writer(
                observations,
                spatial,
                temporal,
                query,
                request,
            )
            runtime_state = write.runtime_state
            bank_states = write.bank_states
            bank_audit = write.audit
            soft_write = write.soft_write

        lifecycle._observed()
        return ObservationChunkOutput(
            owner=request.owner,
            visual=adapted,
            query=query,
            spatial=spatial,
            temporal=temporal,
            observations=observations,
            runtime_state=runtime_state,
            bank_states=bank_states,
            retrieval_history=retrieval_history,
            state_audit=bank_audit,
            soft_intermediates=SoftIntermediates(
                adapted_visual=adapted.value,
                query=query,
                spatial=spatial,
                temporal=temporal,
                observations=observations,
                state_write=soft_write,
                fast_associative=fast_associative,
            ),
        )

    def prepare_answer(
        self,
        request: AnswerQueryRequest,
        lifecycle: PrefillLifecycle,
    ) -> PreparedAnswer:
        """Run Reader, retrieval, resampling and composition exactly once."""

        observation = request.observation
        retrieval: RetrieverOutput | None = None
        reader_results: tuple[ReaderResult, ...] = ()
        resampler_output: StateResamplerOutput | None = None
        if self.feature_flags.reader_enabled or self.feature_flags.state_tokens_enabled:
            retrieval = self.components.retriever(
                self.components.state_bank,
                observation.retrieval_history,
                observation.query,
                video_ids=request.owner.video_ids,
                trajectory_ids=request.owner.trajectory_ids,
            )
        if self.feature_flags.reader_enabled:
            reader_results = tuple(
                self.components.reader.read_bank(
                    self.components.state_bank,
                    observation.bank_states,
                    observation.query,
                    video_ids=request.owner.video_ids,
                    trajectory_ids=request.owner.trajectory_ids,
                )
            )
        if self.feature_flags.state_tokens_enabled:
            resampler_output = self.components.resampler(observation.query.q_target, retrieval)
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
        observation = request.observation
        context = self._build_associative_context(observation.query, observation.bank_states)
        fast_adapter = self.components.fast_adapter
        with _bind_associative_context(fast_adapter, context):
            qwen_output = self.components.qwen_prefill(prepared.qwen_request)
        self._consume_associative_intermediates(fast_adapter)
        lifecycle._prefilled()
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
        )

    def generate_answer(
        self,
        prepared: PreparedAnswer,
        lifecycle: PrefillLifecycle,
        *,
        max_new_tokens: int = 16,
    ) -> StateTTTGenerationOutput:
        """Execute one greedy HF generate call; its first pass is the sole Qwen prefill."""

        request = prepared.request
        observation = request.observation
        context = self._build_associative_context(observation.query, observation.bank_states)
        fast_adapter = self.components.fast_adapter
        with _bind_associative_context(fast_adapter, context):
            generated = self.components.qwen_generate(
                QwenGenerateRequest(prepared.qwen_request, max_new_tokens=max_new_tokens)
            )
        self._consume_associative_intermediates(fast_adapter)
        lifecycle._prefilled()
        return StateTTTGenerationOutput(
            answer_text=generated.answer_text,
            generated_token_ids=generated.token_ids,
            reader=prepared.reader,
            resampler=prepared.resampler,
            runtime_state=observation.runtime_state,
            state_audit=prepared.state_audit,
        )

    def _build_associative_context(
        self,
        query: QueryEncoderOutput,
        bank_states: Sequence[StateBankRuntimeState],
    ) -> FastAssociativeContext | None:
        if not self.feature_flags.fast_enabled or not self.feature_flags.bank_enabled:
            return None
        state_bank = self.components.state_bank
        if not isinstance(state_bank, StructuredStateBank):
            return None
        return build_fast_associative_context(query.q_target, state_bank.view(bank_states))

    @staticmethod
    def _consume_associative_intermediates(
        fast_adapter: FastStage | None,
    ) -> AssociativeTTTIntermediates | None:
        consumer = getattr(fast_adapter, "consume_associative_intermediates", None)
        if consumer is None:
            return None
        return cast(AssociativeTTTIntermediates, consumer())


def _bind_associative_context(
    fast_adapter: FastStage | None,
    context: FastAssociativeContext | None,
) -> AbstractContextManager[object]:
    """Bind one associative read context around a stage call, when supported."""

    if fast_adapter is None or context is None:
        return nullcontext()
    binder = getattr(fast_adapter, "use_associative_context", None)
    if binder is None:
        return nullcontext()
    return cast("AbstractContextManager[object]", binder(context))


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
