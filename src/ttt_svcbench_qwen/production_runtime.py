"""Concrete H200 runtime for production A2 and A5 training.

This is the numerical/materialization counterpart of the LLaMA-Factory bridge.  Support video
chunks stay as lightweight interval specifications until their step executes.  Exactly one
current chunk is then decoded, processed and handed to Qwen; no historical visual-feature list is
representable at this boundary.
"""

from __future__ import annotations

import hashlib
import math
import os
import weakref
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import av
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    AdaptiveChunkSpec,
    ProductionQueryRecord,
    adaptive_support_schedule,
)
from ttt_svcbench_qwen.fast_ttt import (
    AssociativeTTTIntermediates,
    FastAssociativeContext,
    FastTTTAdapter,
    build_fast_ttt_adapter,
)
from ttt_svcbench_qwen.identity_bank import IdentityBank, build_identity_bank
from ttt_svcbench_qwen.input_composer import (
    ComposedInput,
    compose_inputs,
    map_teacher_forced_targets,
    register_input_composer_tokens_with_audit,
)
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    compute_answer_loss,
)
from ttt_svcbench_qwen.meta_trainer import (
    MetaCausalChunk,
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    MetaTTTQueryPoint,
)
from ttt_svcbench_qwen.model import (
    BatchRuntimeState,
    ComposerStage,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    QwenGenerateOutput,
    QwenGenerateRequest,
    QwenPrefillOutput,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    VisualStageOutput,
    query_dropout_seed,
)
from ttt_svcbench_qwen.observation_heads import (
    ObservationHeads,
    ObservationOutputs,
    build_observation_heads,
)
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakGradientAnchors,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.preprocess_cache import (
    CachedChunk,
    PreprocessCache,
    PreprocessCacheMode,
    PreprocessCacheStorageDtype,
    PreprocessFingerprint,
    build_fingerprint,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    ProductionTTTConfig,
    load_outer_checkpoint,
)
from ttt_svcbench_qwen.query_encoder import (
    Operator,
    QueryEncoder,
    QueryEncoderInput,
    QueryEncoderOutput,
    TimeWindowMode,
    build_query_encoder,
    embed_question_tokens,
)
from ttt_svcbench_qwen.query_tokens import QuestionTokenBatch, tokenize_questions
from ttt_svcbench_qwen.qwen_adapter import (
    MergedVideoMetadata,
    PreparedVideoFeatures,
    PreparedVisualChunk,
    Qwen3VLAdapter,
    QwenVisualOutput,
    RawVisualChunk,
    StateEmbeddingPayload,
)
from ttt_svcbench_qwen.runtime_metrics import (
    configure_runtime_metrics,
    trace_cuda_phase,
)
from ttt_svcbench_qwen.stage_a_runtime import StageABankWriter
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    OfficialWeakLossAudit,
    OfficialWeakSupervision,
    OfficialWeakTargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import (
    StateBankRuntimeState,
    StructuredStateBank,
    build_state_bank,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialObjectEncoder,
    TemporalCache,
    TemporalEncoderOutput,
    TemporalEventEncoder,
    build_spatial_encoder,
    build_temporal_encoder,
)
from ttt_svcbench_qwen.state_reader import (
    DeterministicStateReader,
    ReaderResult,
    build_state_reader,
    build_state_resampler,
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput, build_state_retriever
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageAEpisodeInputs,
    StageAEpisodeRunner,
    StageASupervisionBatch,
    StageATrainingBatch,
)
from ttt_svcbench_qwen.video_preprocessing import QwenVideoPreprocessor, av_frame_timestamp

_ANSWER_INSTRUCTION = (
    "The video chunk ends at the question time. Answer the question using the structured "
    "state and output only the answer, with no explanation.\nQuestion: {question}"
)


@dataclass(frozen=True, slots=True)
class SupportChunkSpec:
    """A lightweight, label-free description of exactly one bounded Support chunk."""

    chunk_id: str
    video_path: Path
    start_time: float
    end_time: float
    maximum_frames: int
    query_time: float
    reset_soft_state: bool = False

    @property
    def observation_role(self) -> str:
        return "support"

    @property
    def sample_fps(self) -> float | None:
        return None

    @property
    def frame_sampling(self) -> str:
        return "uniform"


@dataclass(frozen=True, slots=True)
class QueryObservationSpec:
    """One causal Query observation; it is one feature set, never a history container."""

    chunk_id: str
    video_path: Path
    start_time: float
    end_time: float
    maximum_frames: int
    query_time: float
    reset_soft_state: bool = False
    sampling_fps: float = 2.0
    sampling_policy: str = "llamafactory_uniform_cap"
    decode_max_groups: int = 16
    query_role: Literal["query", "state_query", "answer_query"] = "query"

    @property
    def observation_role(self) -> str:
        return self.query_role

    @property
    def sample_fps(self) -> float:
        return self.sampling_fps

    @property
    def frame_sampling(self) -> str:
        return self.sampling_policy


ObservationSpec = SupportChunkSpec | QueryObservationSpec


@dataclass(frozen=True, slots=True)
class CurrentChunkMaterialization:
    spec: ObservationSpec
    frames: Tensor
    frame_timestamps: Tensor
    tubelet_timestamps: Tensor
    tubelet_valid_mask: Tensor
    tubelet_position_ids: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor


@dataclass(frozen=True, slots=True)
class PreparedVisualCPU:
    """Frame-free visual payload transferred from a DataLoader worker.

    Decoded RGB frames stay inside the worker long enough to run the Qwen processor and causal
    audit. The trainer process receives only tensors that are consumed by ViT/State code plus the
    timestamps needed to preserve the causal contract.
    """

    spec: ObservationSpec
    frame_timestamps: Tensor
    tubelet_timestamps: Tensor
    tubelet_valid_mask: Tensor
    tubelet_position_ids: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor

    @property
    def frame_count(self) -> int:
        return int(self.frame_timestamps.shape[0])

    @property
    def patch_count(self) -> int:
        return int(self.pixel_values_videos.shape[0])

    def pin_memory(self) -> PreparedVisualCPU:
        return replace(
            self,
            frame_timestamps=self.frame_timestamps.pin_memory(),
            tubelet_timestamps=self.tubelet_timestamps.pin_memory(),
            tubelet_valid_mask=self.tubelet_valid_mask.pin_memory(),
            tubelet_position_ids=self.tubelet_position_ids.pin_memory(),
            pixel_values_videos=self.pixel_values_videos.pin_memory(),
            video_grid_thw=self.video_grid_thw.pin_memory(),
        )


def _bind_runtime_query(
    query: RuntimeQueryInput,
    video_path: Path,
    episode_nonce: int,
) -> RuntimeQueryInput:
    return replace(query, video=video_path, episode_nonce=episode_nonce)


@dataclass(frozen=True, slots=True)
class PreparedAnswerCPU:
    """CPU-only Query tensors safe to build in a DataLoader worker."""

    spec: QueryObservationSpec
    base_input_ids: Tensor
    base_attention_mask: Tensor
    target_labels: AnswerTargetLabels
    materialized_query: PreparedVisualCPU

    def pin_memory(self) -> PreparedAnswerCPU:
        """Pin the compact tensors that cross the CPU-to-GPU boundary."""

        chunk = self.materialized_query.pin_memory()
        labels = replace(
            self.target_labels,
            base_labels=self.target_labels.base_labels.pin_memory(),
            base_number_token_mask=self.target_labels.base_number_token_mask.pin_memory(),
            target_counts=self.target_labels.target_counts.pin_memory(),
        )
        return replace(
            self,
            base_input_ids=self.base_input_ids.pin_memory(),
            base_attention_mask=self.base_attention_mask.pin_memory(),
            target_labels=labels,
            materialized_query=chunk,
        )


@dataclass(frozen=True, slots=True)
class PreparedA2Record:
    """One manifest row plus independent CPU-prepared State and Answer Queries."""

    record: A2QueryRecord
    answer: PreparedAnswerCPU
    state_query: PreparedVisualCPU
    supports: tuple[PreparedVisualCPU, ...] = ()

    def pin_memory(self) -> PreparedA2Record:
        return replace(
            self,
            answer=self.answer.pin_memory(),
            supports=tuple(chunk.pin_memory() for chunk in self.supports),
            state_query=self.state_query.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class PreparedA5Record:
    """One A5 episode plus CPU-prepared Query answer inputs."""

    record: A5EpisodeRecord
    query_answers: tuple[PreparedAnswerCPU, ...]
    state_queries: tuple[PreparedVisualCPU, ...] = ()

    def pin_memory(self) -> PreparedA5Record:
        return replace(
            self,
            query_answers=tuple(answer.pin_memory() for answer in self.query_answers),
            state_queries=tuple(query.pin_memory() for query in self.state_queries),
        )


@dataclass(frozen=True, slots=True)
class ProductionVisualAudit:
    chunk: CurrentChunkMaterialization | PreparedVisualCPU


def _identity_chunk(value: object) -> CurrentChunkMaterialization:
    return cast(CurrentChunkMaterialization, value)


class VideoChunkMaterializer:
    """Decode one interval with bounded CPU prefetch and reusable preprocessing cache."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None = None,
        cache_support_visuals: bool = True,
        cache_query_visuals: bool = True,
        cache_query_roles: frozenset[str] | None = None,
        prefetch_depth: int = 2,
        decode_coalesce: bool = True,
    ) -> None:
        self.config = config
        self.minimum_pixels = minimum_pixels
        self.maximum_pixels = maximum_pixels
        self.processor = QwenVideoPreprocessor(config)
        self.preprocess_cache = preprocess_cache
        self.cache_support_visuals = cache_support_visuals
        self.cache_query_visuals = cache_query_visuals
        if cache_query_roles is None:
            cache_query_roles = (
                frozenset(("state_query", "answer_query")) if cache_query_visuals else frozenset()
            )
        self.cache_query_roles = cache_query_roles
        self.prefetch_depth = prefetch_depth
        self.decode_coalesce = decode_coalesce
        self._source_dataset = "runtime"
        self._executor: ThreadPoolExecutor | None = None
        self._pending_queue: deque[
            tuple[
                SupportChunkSpec,
                Future[Any],
                Callable[[object], CurrentChunkMaterialization],
            ]
        ] = deque()
        self._remaining_specs: deque[SupportChunkSpec] = deque()

    def __call__(self, spec: ObservationSpec) -> CurrentChunkMaterialization:
        if self._pending_queue:
            _expected, future, resolver = self._pending_queue[0]
            result = resolver(future.result())
            self._pending_queue.popleft()
            self._schedule_next()
            return result
        return self._materialize(spec)

    def begin_prefetch(
        self, specs: Sequence[SupportChunkSpec], *, source_dataset: str | None = None
    ) -> None:
        """Start a bounded sequence while preserving strict consumption order."""

        self.end_prefetch()
        if os.environ.get("TTT_A2_SUPPORT_PREFETCH", "1") == "0" or not specs:
            return
        if source_dataset is not None:
            self._source_dataset = source_dataset
        self._remaining_specs.extend(specs)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.prefetch_depth,
                thread_name_prefix="ttt-support-prefetch",
            )
        self._schedule_next()

    def set_source_dataset(self, source_dataset: str) -> None:
        self._source_dataset = source_dataset

    def end_prefetch(self) -> None:
        """Drop pending bookkeeping without retaining a historical chunk list."""

        for entry in self._pending_queue:
            future = entry[1]
            if not future.done():
                future.cancel()
        self._pending_queue.clear()
        self._remaining_specs.clear()

    def _schedule_next(self) -> None:
        executor = cast(ThreadPoolExecutor, self._executor)
        while len(self._pending_queue) < self.prefetch_depth and self._remaining_specs:
            if self.decode_coalesce:
                group = [self._remaining_specs.popleft()]
                group_capacity = self.prefetch_depth - len(self._pending_queue)
                while (
                    len(group) < group_capacity
                    and self._remaining_specs
                    and self._remaining_specs[0].video_path == group[0].video_path
                ):
                    group.append(self._remaining_specs.popleft())
                group_future = executor.submit(self._materialize_group, tuple(group))
                for spec in group:
                    chunk_id = spec.chunk_id

                    def resolve_group(
                        value: object, chunk_id: str = chunk_id
                    ) -> CurrentChunkMaterialization:
                        return cast(dict[str, CurrentChunkMaterialization], value)[chunk_id]

                    self._pending_queue.append(
                        (
                            spec,
                            group_future,
                            resolve_group,
                        )
                    )
            else:
                spec = self._remaining_specs.popleft()
                future = executor.submit(self._materialize, spec)
                self._pending_queue.append((spec, future, _identity_chunk))

    def _materialize(self, spec: ObservationSpec) -> CurrentChunkMaterialization:
        cache = self._cache_for(spec)
        fingerprint = self._fingerprint(spec) if cache is not None else None
        if cache is not None:
            cached = cache.get(cast(PreprocessFingerprint, fingerprint))
            if cached is not None:
                return self._from_cached(spec, cached)
        frames, timestamps = _decode_uniform_interval(spec, _sample_fps_for(spec, self.config))
        return self._materialize_decoded(
            spec,
            frames,
            timestamps,
            fingerprint,
            cache=cache,
        )

    def _materialize_group(
        self, specs: tuple[SupportChunkSpec, ...]
    ) -> dict[str, CurrentChunkMaterialization]:
        """Decode one same-video Support group with a single PyAV container."""

        if not specs:
            return {}
        results: dict[str, CurrentChunkMaterialization] = {}
        misses: list[
            tuple[SupportChunkSpec, PreprocessFingerprint | None, PreprocessCache | None]
        ] = []
        for spec in specs:
            cache = self._cache_for(spec)
            fingerprint = self._fingerprint(spec) if cache is not None else None
            cached = (
                cache.get(fingerprint) if cache is not None and fingerprint is not None else None
            )
            if cached is not None:
                results[spec.chunk_id] = self._from_cached(spec, cached)
            else:
                misses.append((spec, fingerprint, cache))
        if not misses:
            return results
        try:
            decoded = _decode_coalesced_intervals(
                tuple(spec for spec, _, _ in misses),
                self.config.video_preprocessing.sample_fps,
            )
        except Exception:
            # Coalescing is an optimization only.  Preserve the proven decoder for unusual VFR
            # or non-seekable containers instead of turning a cache miss into a training failure.
            decoded = {
                spec.chunk_id: _decode_uniform_interval(
                    spec, self.config.video_preprocessing.sample_fps
                )
                for spec, _, _ in misses
            }
        for spec, fingerprint, cache in misses:
            frames, timestamps = decoded[spec.chunk_id]
            results[spec.chunk_id] = self._materialize_decoded(
                spec,
                frames,
                timestamps,
                fingerprint,
                cache=cache,
            )
        return results

    def materialize_support_group(
        self, specs: Sequence[SupportChunkSpec]
    ) -> tuple[CurrentChunkMaterialization, ...]:
        """Materialize one sample's Support windows with one video-container pass."""

        values = tuple(specs)
        if not values:
            return ()
        materialized = self._materialize_group(values)
        return tuple(materialized[spec.chunk_id] for spec in values)

    def _materialize_decoded(
        self,
        spec: ObservationSpec,
        frames: Tensor,
        timestamps: Tensor,
        fingerprint: PreprocessFingerprint | None,
        *,
        cache: PreprocessCache | None = None,
    ) -> CurrentChunkMaterialization:
        frames = _resize_to_pixel_budget(
            frames,
            minimum_pixels=self.minimum_pixels,
            maximum_pixels=self.maximum_pixels,
        )
        processed = self.processor.process(frames)
        tubelet_times = timestamps.reshape(-1, 2).amax(dim=1).unsqueeze(0)
        positions = _strict_tubelet_positions(
            tubelet_times[0],
            sample_fps=_sample_fps_for(spec, self.config),
        ).unsqueeze(0)
        materialized = CurrentChunkMaterialization(
            spec=spec,
            frames=frames,
            frame_timestamps=timestamps,
            tubelet_timestamps=tubelet_times,
            tubelet_valid_mask=torch.ones_like(tubelet_times, dtype=torch.bool),
            tubelet_position_ids=positions,
            pixel_values_videos=processed.flatten_for_qwen(),
            video_grid_thw=processed.video_grid_thw,
        )
        target_cache = self._cache_for(spec) if cache is None else cache
        if target_cache is not None and target_cache.writable:
            target_cache.put(
                cast(PreprocessFingerprint, fingerprint),
                _cached_from_materialized(materialized),
            )
        return materialized

    def _cache_for(self, spec: ObservationSpec) -> PreprocessCache | None:
        if isinstance(spec, SupportChunkSpec) and not self.cache_support_visuals:
            return None
        if (
            isinstance(spec, QueryObservationSpec)
            and spec.observation_role not in self.cache_query_roles
        ):
            return None
        return self.preprocess_cache

    def _fingerprint(self, spec: ObservationSpec) -> PreprocessFingerprint:
        return _build_preprocess_fingerprint(
            spec,
            config=self.config,
            minimum_pixels=self.minimum_pixels,
            maximum_pixels=self.maximum_pixels,
            source_dataset=self._source_dataset,
        )

    @staticmethod
    def _from_cached(spec: ObservationSpec, cached: CachedChunk) -> CurrentChunkMaterialization:
        return _materialized_from_cached(spec, cached)


class ProductionQwenRuntime(nn.Module):  # type: ignore[misc]
    """Single registered owner for visual extraction, training prefill and generation."""

    def __init__(
        self,
        qwen: Qwen3VLAdapter,
        materializer: VideoChunkMaterializer,
        tokenizer: object,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.materializer = materializer
        self.tokenizer = tokenizer

    def forward(self, request: object) -> object:
        if isinstance(request, ObservationChunkRequest):
            return self._visual(request)
        if isinstance(request, QwenPrefillRequest):
            return self._prefill(request)
        if isinstance(request, QwenGenerateRequest):
            return self._generate(request)
        raise TypeError("production Qwen runtime received an unknown request type")

    def _visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        raw = request.video_input
        if isinstance(raw, RawVisualChunk):
            prepared_chunk = replace(
                self.qwen.prepare_raw_visual_chunk(raw),
                source=raw.source,
            )
            return self._visual(replace(request, video_input=prepared_chunk))
        if isinstance(raw, PreparedVisualChunk):
            return VisualStageOutput(
                value=raw.value,
                audit=ProductionVisualAudit(
                    cast("CurrentChunkMaterialization | PreparedVisualCPU", raw.source)
                ),
            )
        chunk: CurrentChunkMaterialization | PreparedVisualCPU
        if isinstance(raw, (SupportChunkSpec, QueryObservationSpec)):
            chunk = self.materializer(raw)
        else:
            chunk = cast("CurrentChunkMaterialization | PreparedVisualCPU", raw)
        device = _module_device(self.qwen)
        pixels = chunk.pixel_values_videos.to(device=device, non_blocking=True)
        grid = chunk.video_grid_thw.to(device=device, non_blocking=True)
        with trace_cuda_phase("vit_forward", stage="visual"):
            self.qwen.get_video_features(pixels, grid)
        captured = cast(QwenVisualOutput, self.qwen.last_visual_output)
        prepared_features = cast(
            PreparedVideoFeatures, self.qwen.last_prepared_video_features
        )
        adapted_padded = pad_sequence(prepared_features.main_features, batch_first=True)
        adapted = QwenVisualOutput(
            main_visual_embeddings=adapted_padded,
            deepstack_features=prepared_features.deepstack_features,
            visual_valid_mask=captured.visual_valid_mask,
            metadata=prepared_features.metadata,
        )
        materialized = replace(
            chunk,
            pixel_values_videos=pixels,
            video_grid_thw=grid,
            tubelet_timestamps=chunk.tubelet_timestamps.to(device),
            tubelet_valid_mask=chunk.tubelet_valid_mask.to(device),
            tubelet_position_ids=chunk.tubelet_position_ids.to(device),
        )
        return VisualStageOutput(
            value=adapted,
            audit=ProductionVisualAudit(materialized),
        )

    def _prefill(self, request: QwenPrefillRequest) -> QwenPrefillOutput:
        device = _module_device(self.qwen)
        pixels = request.pixel_values_videos.to(device=device, non_blocking=True)
        grid = request.video_grid_thw.to(device=device, non_blocking=True)
        input_ids = cast(Tensor, request.input_ids).to(device=device)
        attention_mask = cast(Tensor, request.attention_mask).to(device=device)
        state_payload = _state_embedding_payload(request, input_ids)
        kwargs = dict(request.qwen_kwargs)
        kwargs.setdefault("use_cache", False)
        with trace_cuda_phase("llm_prefill", stage="prefill"):
            output = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixels,
                video_grid_thw=grid,
                state_embedding_payload=state_payload,
                **kwargs,
            )
        return cast(QwenPrefillOutput, output)

    def _generate(self, request: QwenGenerateRequest) -> QwenGenerateOutput:
        prefill = request.prefill
        device = _module_device(self.qwen)
        input_ids = prefill.input_ids.to(device=device)
        attention_mask = prefill.attention_mask.to(device=device)
        pixels = prefill.pixel_values_videos.to(device=device, non_blocking=True)
        grid = prefill.video_grid_thw.to(device=device, non_blocking=True)
        kwargs = dict(prefill.qwen_kwargs)
        kwargs.pop("labels", None)
        kwargs.update(
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=request.max_new_tokens,
        )
        generated = self.qwen.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixels,
            video_grid_thw=grid,
            state_embedding_payload=_state_embedding_payload(prefill, input_ids),
            **kwargs,
        )
        sequences = getattr(generated, "sequences", generated)
        new_tokens = sequences[:, input_ids.shape[1] :]
        decode = cast(Any, self.tokenizer).batch_decode
        texts = decode(new_tokens, skip_special_tokens=True)
        return QwenGenerateOutput(texts[0].strip(), new_tokens.detach())


class ProductionQueryRuntime(nn.Module):  # type: ignore[misc]
    def __init__(
        self, query_encoder: QueryEncoder, tokenizer: object, qwen_model: nn.Module
    ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.tokenizer = tokenizer
        self._qwen_ref = weakref.ref(qwen_model)
        self._token_cache: dict[str, QuestionTokenBatch] = {}

    def forward(self, value: RuntimeQueryInput, *, inference: bool) -> QueryEncoderOutput:
        question = value.question
        tokens = self._token_cache.get(question)
        if tokens is None:
            tokens = tokenize_questions(cast(Any, self.tokenizer), (question,))
            self._token_cache[question] = tokens
        qwen = self._qwen_ref()
        embeddings = embed_question_tokens(cast(Any, qwen), tokens, self.query_encoder_config)
        inputs = QueryEncoderInput.from_runtime_queries(embeddings, tokens, (value,))
        dropout_seed = query_dropout_seed(value)
        cuda_devices: list[int] = []
        if embeddings.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device()
                if embeddings.device.index is None
                else embeddings.device.index
            )
        # Recompute a fresh graph for every chunk (required by segmented A5 backward), while
        # keeping dropout masks identical for the same Query inside one episode so cached state
        # signatures remain exact.  The fork prevents these local masks from perturbing global RNG.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(dropout_seed)
            with trace_cuda_phase("query_encoder", query_id=value.query_id):
                output = self.query_encoder(inputs, inference=inference)
        if not isinstance(output, QueryEncoderOutput):
            raise TypeError("production Query encoder returned an invalid output")
        return output

    @property
    def query_encoder_config(self) -> ProjectConfig:
        return cast(ProjectConfig, getattr(self, "_project_config", None))

    def bind_project_config(self, config: ProjectConfig) -> None:
        object.__setattr__(self, "_project_config", config)


class FastVisualPassThrough:
    """Bind context to the Fast Adapter already executed inside Qwen vision."""

    def __init__(self, adapter: FastTTTAdapter) -> None:
        self.adapter = adapter

    def use_associative_context(self, context: FastAssociativeContext) -> object:
        return self.adapter.use_associative_context(context)

    def consume_associative_intermediates(self) -> AssociativeTTTIntermediates | None:
        return self.adapter.consume_associative_intermediates()

    def __call__(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        del query, request
        return visual


class ProductionSpatialRuntime(nn.Module):  # type: ignore[misc]
    def __init__(self, encoder: SpatialObjectEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> SpatialEncoderOutput:
        value, chunk, typed_query, runtime = _stage_inputs(visual, query, request)
        prior = (
            (None,) * len(request.owner.video_ids)
            if chunk.spec.reset_soft_state
            else runtime.slot_states
        )
        with trace_cuda_phase("state_modules", component="spatial"):
            output = self.encoder(
                value.main_visual_embeddings,
                value.visual_valid_mask,
                value.metadata,
                chunk.tubelet_valid_mask,
                typed_query.q_target,
                request.owner.video_ids,
                prior_states=prior,
                detach_runtime_state=True,
            )
        return cast(SpatialEncoderOutput, output)


class ProductionTemporalRuntime(nn.Module):  # type: ignore[misc]
    def __init__(self, encoder: TemporalEventEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput:
        value, chunk, typed_query, runtime = _stage_inputs(visual, query, request)
        cache = None if chunk.spec.reset_soft_state else runtime.temporal_cache
        temporal_value, temporal_mask, temporal_times, temporal_positions = _causal_temporal_tail(
            value, chunk, cache
        )
        query_time = torch.tensor(
            [chunk.spec.query_time],
            dtype=temporal_times.dtype,
            device=temporal_value.main_visual_embeddings.device,
        )
        with trace_cuda_phase("state_modules", component="temporal"):
            output = self.encoder(
                temporal_value.main_visual_embeddings,
                temporal_value.visual_valid_mask,
                temporal_value.metadata,
                temporal_mask,
                temporal_times,
                temporal_positions,
                query_time,
                typed_query.q_target,
                request.owner.video_ids,
                request.owner.trajectory_ids,
                cache=cache,
                detach_cache=True,
            )
        return cast(TemporalEncoderOutput, output)


def _causal_temporal_tail(
    value: QwenVisualOutput,
    chunk: CurrentChunkMaterialization | PreparedVisualCPU,
    cache: TemporalCache | None,
) -> tuple[QwenVisualOutput, Tensor, Tensor, Tensor]:
    """Keep only current tubelets newer than the bounded temporal cache.

    Adaptive long-history chunks are sampled uniformly, so their physical timestamps can be
    sparse and adjacent chunks can overlap without sharing the same sampling grid.  Temporal
    sequence IDs therefore describe the contiguous *processed* tubelet stream, while physical
    timestamps remain exact.  Overlap tubelets at or before the latest cached timestamp are
    omitted from the temporal path; the full current chunk still reaches Qwen, Spatial/O2 and the
    hard overlap machinery, and no historical visual feature is concatenated here.
    """

    times = chunk.tubelet_timestamps
    tubelet_count = times.shape[1]
    drop = 0
    next_position = 0
    if cache is not None:
        valid_count = int(cache.valid_mask[0].sum().item())
        if valid_count:
            latest = cache.timestamps[0, valid_count - 1].to(dtype=times.dtype)
            drop = int(torch.searchsorted(times[0], latest + 1.0e-9, right=True).item())
            next_position = int(cache.total_seen[0].item())

    if drop >= tubelet_count:
        invalid_mask = torch.zeros_like(chunk.tubelet_valid_mask, dtype=torch.bool)
        invalid_times = torch.full_like(times, -1.0)
        invalid_positions = torch.full_like(chunk.tubelet_position_ids, -1)
        return value, invalid_mask, invalid_times, invalid_positions

    remaining = tubelet_count - drop
    temporal_value = value if drop == 0 else _slice_temporal_visual_tail(value, drop, remaining)
    temporal_times = times[:, drop:]
    temporal_mask = torch.ones_like(temporal_times, dtype=torch.bool)
    temporal_positions = torch.arange(
        next_position,
        next_position + remaining,
        dtype=torch.int64,
        device=temporal_times.device,
    ).unsqueeze(0)
    return temporal_value, temporal_mask, temporal_times, temporal_positions


def _slice_temporal_visual_tail(
    value: QwenVisualOutput,
    drop: int,
    remaining: int,
) -> QwenVisualOutput:
    metadata = value.metadata
    merged_height = int(metadata.merged_grid_thw[0, 1].item())
    merged_width = int(metadata.merged_grid_thw[0, 2].item())
    tokens_per_tubelet = merged_height * merged_width
    start_token = drop * tokens_per_tubelet
    token_count = remaining * tokens_per_tubelet
    stop_token = start_token + token_count
    raw_grid = metadata.video_grid_thw.clone()
    merged_grid = metadata.merged_grid_thw.clone()
    raw_grid[0, 0] = remaining
    merged_grid[0, 0] = remaining
    tail_metadata = MergedVideoMetadata(
        video_grid_thw=raw_grid,
        merged_grid_thw=merged_grid,
        spatial_merge_size=metadata.spatial_merge_size,
        token_counts=(token_count,),
        token_offsets=(0, token_count),
    )
    main = value.main_visual_embeddings[:, start_token:stop_token]
    valid = torch.ones(main.shape[:2], dtype=torch.bool, device=main.device)
    deepstack = tuple(feature[start_token:stop_token] for feature in value.deepstack_features)
    return QwenVisualOutput(
        main_visual_embeddings=main,
        deepstack_features=cast(tuple[Tensor, Tensor, Tensor], deepstack),
        visual_valid_mask=valid,
        metadata=tail_metadata,
    )


class ProductionObservationRuntime(nn.Module):  # type: ignore[misc]
    def __init__(self, heads: ObservationHeads) -> None:
        super().__init__()
        self.heads = heads

    def forward(
        self,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> ObservationOutputs:
        runtime = _stage_runtime(request)
        raw_chunk = _current_chunk_input(request.video_input)
        reset = (
            raw_chunk.reset_soft_state
            if isinstance(raw_chunk, (SupportChunkSpec, QueryObservationSpec))
            else raw_chunk.spec.reset_soft_state
        )
        empty = (None,) * len(request.owner.video_ids)
        with trace_cuda_phase("state_modules", component="observation_heads"):
            output = self.heads(
                spatial,
                temporal,
                query.q_target,
                request.owner.video_ids,
                request.owner.trajectory_ids,
                e1_prior_states=empty if reset else runtime.e1_states,
                e2_prior_states=empty if reset else runtime.e2_states,
                detach_runtime_state=True,
            )
        return cast(ObservationOutputs, output)


class ProductionReaderRuntime:
    """Expose the concrete Reader through the shared typed model boundary."""

    def __init__(self, reader: DeterministicStateReader) -> None:
        self.reader = reader

    def read(self, retrieval: RetrieverOutput) -> Sequence[ReaderResult]:
        return self.reader.read(retrieval)

    def read_bank(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> Sequence[ReaderResult]:
        return self.reader.read_bank(
            state_bank,
            states,
            query,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )

    def audit_number_tokens(self, result: ReaderResult) -> int | None:
        return self.reader.audit_number_tokens(result)


class ProductionOuterModel(nn.Module):  # type: ignore[misc]
    """Checkpoint/optimizer owner; numerical entrypoints live in the injected runners."""

    def __init__(
        self,
        state_model: StateTTTModel,
        qwen_model: nn.Module,
        official_weak_balancer: OfficialWeakOuterLossComposer | None = None,
    ) -> None:
        super().__init__()
        self.state_model = state_model
        if official_weak_balancer is not None:
            self.official_weak_balancer = official_weak_balancer
        # Qwen is already registered below ``state_model``.  Keep only a weak reference here so
        # Hugging Face lifecycle methods can be forwarded without duplicating checkpoint keys.
        self._qwen_model_ref = weakref.ref(qwen_model)

    @property
    def supports_gradient_checkpointing(self) -> bool:
        return bool(getattr(self._qwen_model(), "supports_gradient_checkpointing", False))

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        enable = cast(Any, getattr(self._qwen_model(), "gradient_checkpointing_enable", None))
        # ``CustomSeq2SeqTrainer`` calls this owner again after LLaMA-Factory has
        # already configured Qwen.  Passing ``None`` would silently restore the
        # LLaMA-Factory wrapper's default re-entrant checkpointing, which lets
        # ZeRO-2 see a Decoder partition twice during checkpoint recomputation.
        # Production A2/A5 requires non-reentrant checkpointing end-to-end.
        kwargs = dict(gradient_checkpointing_kwargs or {})
        kwargs["use_reentrant"] = False
        enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self) -> None:
        disable = cast(Any, getattr(self._qwen_model(), "gradient_checkpointing_disable", None))
        disable()

    def _qwen_model(self) -> nn.Module:
        return cast(nn.Module, self._qwen_model_ref())

    def forward(self, **_inputs: object) -> Tensor:
        raise RuntimeError("ProductionOuterModel must be driven by the typed A2/A5 trainer hook")


class _PrefetchCollatorBase:
    def __init__(
        self,
        *,
        processor: object,
        tokenizer: object,
        config: ProjectConfig,
        ttt_config: ProductionTTTConfig,
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.ttt_config = ttt_config
        self.minimum_pixels = minimum_pixels
        self.maximum_pixels = maximum_pixels
        self.preprocess_cache = preprocess_cache
        self.video = VideoChunkMaterializer(
            config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            preprocess_cache=preprocess_cache,
            cache_query_roles=ttt_config.cached_query_roles,
            prefetch_depth=1,
            decode_coalesce=False,
        )


def _prepare_query_pair(
    collator: _PrefetchCollatorBase,
    query: ProductionQueryRecord,
    *,
    chunk_prefix: str,
    video_path: Path,
    reset_soft_state: bool,
    source_dataset: str,
) -> tuple[PreparedVisualCPU, PreparedAnswerCPU]:
    state_spec = _query_chunk_spec(
        f"{chunk_prefix}:state_query",
        video_path,
        query.runtime.query_time,
        reset_soft_state=reset_soft_state,
        config=collator.ttt_config,
        role="state_query",
    )
    answer_spec = _query_chunk_spec(
        f"{chunk_prefix}:answer_query",
        video_path,
        query.runtime.query_time,
        reset_soft_state=reset_soft_state,
        config=collator.ttt_config,
        role="answer_query",
    )
    state_query = _compact_materialized_chunk(collator.video(state_spec))
    answer = _prepare_answer_cpu(
        query,
        answer_spec,
        processor=collator.processor,
        tokenizer=collator.tokenizer,
        config=collator.config,
        minimum_pixels=collator.minimum_pixels,
        maximum_pixels=collator.maximum_pixels,
        preprocess_cache=(
            collator.preprocess_cache
            if collator.ttt_config.query_cache_enabled("answer_query")
            else None
        ),
        source_dataset=source_dataset,
    )
    return state_query, answer


class A2PrefetchCollator(_PrefetchCollatorBase):
    """Materialize the next Query in a persistent DataLoader worker.

    The collator deliberately owns only CPU processor/tokenizer objects.  It never receives the
    Qwen model, Bank/FSM writer, or any CUDA state, so hard runtime state remains rank-local and
    is committed exactly once by :class:`ProductionEpisodeMaterializer`.
    """

    def __init__(
        self,
        *,
        processor: object,
        tokenizer: object,
        config: ProjectConfig,
        ttt_config: ProductionTTTConfig,
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None = None,
        prepared_episode_max_bytes: int = 2_147_483_648,
    ) -> None:
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            config=config,
            ttt_config=ttt_config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            preprocess_cache=preprocess_cache,
        )
        self.prepared_episode_max_bytes = prepared_episode_max_bytes

    def __call__(self, records: Sequence[object]) -> dict[str, object]:
        record = cast(A2QueryRecord, records[0])
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        self.video.set_source_dataset(record.source_dataset)
        state_query, answer = _prepare_query_pair(
            self,
            record.query,
            chunk_prefix=record.query.runtime.query_id,
            video_path=video_path,
            reset_soft_state=False,
            source_dataset=record.source_dataset,
        )
        support_specs = _a2_support_chunk_specs(record, video_path)
        supports = tuple(
            _compact_materialized_chunk(value)
            for value in self.video.materialize_support_group(support_specs)
        )
        return {
            "prepared_a2": PreparedA2Record(
                record=record,
                answer=answer,
                supports=supports,
                state_query=state_query,
            )
        }


class A5PrefetchCollator(_PrefetchCollatorBase):
    """Prepare only CPU Query tensors; runtime/State-TTT objects stay in the trainer process."""

    def __init__(
        self,
        *,
        processor: object,
        tokenizer: object,
        config: ProjectConfig,
        ttt_config: ProductionTTTConfig,
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None = None,
    ) -> None:
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            config=config,
            ttt_config=ttt_config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            preprocess_cache=preprocess_cache,
        )

    def __call__(self, records: Sequence[object]) -> dict[str, object]:
        record = cast(A5EpisodeRecord, records[0])
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        answers: list[PreparedAnswerCPU] = []
        state_queries: list[PreparedVisualCPU] = []
        self.video.set_source_dataset(record.source_dataset)
        primary = record.queries[0].runtime
        for index, query in enumerate(record.queries):
            state_query, answer = _prepare_query_pair(
                self,
                query,
                chunk_prefix=f"{record.episode_id}:q{index}",
                video_path=video_path,
                reset_soft_state=index > 0 and query.runtime.question != primary.question,
                source_dataset=record.source_dataset,
            )
            state_queries.append(state_query)
            answers.append(answer)
        return {"prepared_a5": PreparedA5Record(record, tuple(answers), tuple(state_queries))}


class ProductionEpisodeMaterializer:
    def __init__(
        self,
        backbone: LlamaFactoryBackboneBundle,
        writer: StageABankWriter,
        video: VideoChunkMaterializer,
    ) -> None:
        self.backbone = backbone
        self.config = backbone.project_config
        self.ttt_config = backbone.ttt_config
        self.writer = writer
        self.video = video
        self.tokenizer = backbone.tokenizer
        self.processor = backbone.processor
        self._episode_nonce = 0

    def a2(self, source: PreparedA2Record) -> StageATrainingBatch:
        record = source.record
        self.video.set_source_dataset(record.source_dataset)
        episode_nonce = self._allocate_episode_nonce()
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        owner = RuntimeOwner((record.video_id,), (record.trajectory_id,))
        runtime = self.writer.reset(owner)
        _, supports = adaptive_support_schedule(record.query.runtime.query_time)
        prepared_answer = source.answer
        answer, labels, _ = self._bind_answer(prepared_answer)
        materialized_state = source.state_query
        requests_list: list[ObservationChunkRequest] = []
        for index, chunk in enumerate(supports):
            request = self._request(
                owner,
                runtime,
                video_path,
                record.query.runtime,
                chunk,
                index,
                episode_nonce,
            )
            if source.supports:
                prepared_support = source.supports[index]
                request = replace(request, video_input=prepared_support)
            requests_list.append(request)
        requests = tuple(requests_list) + (
            ObservationChunkRequest(
                owner=owner,
                video_input=materialized_state,
                query_input=_bind_runtime_query(record.query.runtime, video_path, episode_nonce),
                runtime_state=runtime,
                bank_states=runtime.state_bank_states,
                inference=False,
                retrieval_history_write_enabled=False,
            ),
        )
        supervision = StageASupervisionBatch(
            answer=labels,
            state=None,
            official_weak=(_official_weak(record.query),),
        )
        runtime_query = _bind_runtime_query(record.query.runtime, video_path, episode_nonce)
        return StageATrainingBatch(
            runtime_queries=(runtime_query,),
            model_inputs=StageAEpisodeInputs(owner, requests, answer),
            supervision=supervision,
        )

    def a5(self, source: PreparedA5Record) -> MetaTTTEpisode:
        prepared = source
        record = source.record
        self.video.set_source_dataset(record.source_dataset)
        episode_nonce = self._allocate_episode_nonce()
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        owner = RuntimeOwner((record.video_id,), (record.trajectory_id,))
        runtime = self.writer.reset(owner)
        primary = record.queries[0].runtime
        prewarm = self._meta_chunk(
            owner,
            runtime,
            video_path,
            primary,
            record.prewarm,
            "s0",
            episode_nonce,
        )
        supports_list: list[MetaCausalChunk] = []
        for segment in record.supervised_segments:
            runtime_query = segment.queries[0].runtime
            for chunk in segment.supports:
                supports_list.append(
                    self._meta_chunk(
                        owner,
                        runtime,
                        video_path,
                        runtime_query,
                        chunk,
                        f"s{len(supports_list) + 1}",
                        episode_nonce,
                    )
                )
        supports = tuple(supports_list)
        queries: list[MetaTTTQueryPoint] = []
        for index, query in enumerate(record.queries):
            state_spec = _query_chunk_spec(
                f"{record.episode_id}:q{index}:state_query",
                video_path,
                query.runtime.query_time,
                reset_soft_state=index > 0 and query.runtime.question != primary.question,
                config=self.ttt_config,
                role="state_query",
            )
            prepared_answer = prepared.query_answers[index]
            answer, labels, _ = self._bind_answer(prepared_answer)
            materialized = (
                prepared.state_queries[index]
                if prepared.state_queries
                else prepared_answer.materialized_query
            )
            request = ObservationChunkRequest(
                owner=owner,
                video_input=materialized,
                query_input=_bind_runtime_query(query.runtime, video_path, episode_nonce),
                runtime_state=runtime,
                bank_states=runtime.state_bank_states,
                inference=False,
                retrieval_history_write_enabled=False,
            )
            queries.append(
                MetaTTTQueryPoint(
                    chunk=MetaCausalChunk(
                        request,
                        state_spec.start_time,
                        state_spec.end_time,
                        _bind_runtime_query(query.runtime, video_path, episode_nonce),
                    ),
                    query_time=query.runtime.query_time,
                    answer=answer,
                    supervision=StageASupervisionBatch(
                        answer=labels,
                        state=None,
                        official_weak=(_official_weak(query),),
                    ),
                    task_name=record.task_class,
                    case_id=query.runtime.query_id,
                )
            )
        return MetaTTTEpisode(
            owner=owner,
            prewarm_chunk=prewarm,
            support_chunks=supports,
            query_points=tuple(queries),
            seed=self.config.a5.seed,
            segment_lengths=record.segment_lengths,
            segment_query_counts=record.segment_query_counts,
            query_roles=tuple(
                role.value for segment in record.supervised_segments for role in segment.query_roles
            ),
            query_weights=tuple(
                weight for segment in record.supervised_segments for weight in segment.query_weights
            ),
        )

    def _meta_chunk(
        self,
        owner: RuntimeOwner,
        runtime: BatchRuntimeState,
        video_path: Path,
        query: RuntimeQueryInput,
        chunk: AdaptiveChunkSpec,
        suffix: str,
        episode_nonce: int,
    ) -> MetaCausalChunk:
        spec = SupportChunkSpec(
            chunk_id=f"{query.query_id}:{suffix}",
            video_path=video_path,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            maximum_frames=chunk.maximum_frames,
            query_time=query.query_time,
        )
        request = ObservationChunkRequest(
            owner=owner,
            video_input=spec,
            query_input=_bind_runtime_query(query, video_path, episode_nonce),
            runtime_state=runtime,
            bank_states=runtime.state_bank_states,
            inference=False,
            retrieval_snapshot_required=False,
        )
        return MetaCausalChunk(
            request,
            spec.start_time,
            spec.end_time,
            _bind_runtime_query(query, video_path, episode_nonce),
        )

    def _request(
        self,
        owner: RuntimeOwner,
        runtime: BatchRuntimeState,
        video_path: Path,
        query: RuntimeQueryInput,
        chunk: AdaptiveChunkSpec,
        index: int,
        episode_nonce: int,
    ) -> ObservationChunkRequest:
        return ObservationChunkRequest(
            owner=owner,
            video_input=SupportChunkSpec(
                chunk_id=f"{query.query_id}:a2:{index}",
                video_path=video_path,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                maximum_frames=chunk.maximum_frames,
                query_time=query.query_time,
            ),
            query_input=_bind_runtime_query(query, video_path, episode_nonce),
            runtime_state=runtime,
            bank_states=runtime.state_bank_states,
            inference=False,
            retrieval_snapshot_required=False,
        )

    def _allocate_episode_nonce(self) -> int:
        nonce = self._episode_nonce
        self._episode_nonce += 1
        return nonce

    def _bind_answer(
        self,
        prepared: PreparedAnswerCPU,
    ) -> tuple[StageAEpisodeAnswerInputs, AnswerTargetLabels, PreparedVisualCPU]:
        owner = cast(Any, self.backbone.model)
        rope_owner = getattr(owner, "model", owner)
        return (
            StageAEpisodeAnswerInputs(
                base_input_ids=prepared.base_input_ids,
                base_attention_mask=prepared.base_attention_mask,
                pixel_values_videos=prepared.materialized_query.pixel_values_videos,
                video_grid_thw=prepared.materialized_query.video_grid_thw,
                tokenizer=self.tokenizer,
                embedding_owner=owner,
                rope_indexer=rope_owner,
                qwen_kwargs=(("use_cache", False),),
            ),
            prepared.target_labels,
            prepared.materialized_query,
        )


class ProductionA2LossStep:
    def __init__(
        self,
        runner: StageAEpisodeRunner,
        materializer: ProductionEpisodeMaterializer,
        graph_anchor_parameters: Sequence[nn.Parameter],
        config: ProjectConfig,
        outer_composer: OfficialWeakOuterLossComposer,
    ):
        self.runner = runner
        self.materializer = materializer
        self.weak_builder = OfficialWeakTargetBuilder()
        self.graph_anchor_parameters = tuple(
            parameter for parameter in graph_anchor_parameters if parameter.requires_grad
        )
        self.outer_composer = outer_composer
        self.last_balance_audit: OfficialWeakBalanceAudit | None = None
        self.last_weak_audit: OfficialWeakLossAudit | None = None

    def __call__(self, _model: nn.Module, inputs: Mapping[str, object]) -> Tensor:
        prefetched = cast(PreparedA2Record, inputs["prepared_a2"])
        source = prefetched
        record = prefetched.record
        batch = self.materializer.a2(source)
        model_inputs = cast(StageAEpisodeInputs, batch.model_inputs)
        support_specs = tuple(
            request.video_input
            for request in model_inputs.observation_requests
            if isinstance(request.video_input, SupportChunkSpec)
        )
        self.materializer.video.begin_prefetch(support_specs, source_dataset=record.source_dataset)
        try:
            raw = self.runner(batch, training=True)
        finally:
            self.materializer.video.end_prefetch()
        mapped = map_teacher_forced_targets(
            composed_input=cast(ComposedInput, raw.composed_input),
            source_input_ids=raw.source_input_ids,
            source_attention_mask=raw.source_attention_mask,
            source_labels=batch.supervision.answer.base_labels,
            source_number_token_mask=batch.supervision.answer.base_number_token_mask,
        )
        answer = compute_answer_loss(
            AnswerLossInput(
                logits=raw.answer_logits,
                labels=mapped.labels,
                number_token_mask=mapped.number_token_mask,
            )
        )
        weak = self.weak_builder(
            raw.observations,
            raw.query,
            raw.retrieval,
            batch.supervision.official_weak,
            dedup=raw.o2_dedup,
        )
        self.last_weak_audit = weak.audit
        balanced = self.outer_composer.compose(
            (answer,),
            (weak,),
            gradient_anchors=(
                OfficialWeakGradientAnchors(
                    q_target=raw.query.q_target,
                    q_operator=raw.query.q_operator,
                    q_time=raw.query.q_time,
                ),
            ),
        )
        total = balanced.mean_total
        self.last_balance_audit = balanced.audit
        # Official task masks and hard routing intentionally produce a sample-dependent graph.
        # Anchor every trainable Outer parameter with an exact zero dependency so every rank has
        # the same non-None gradient set.  Numerical loss and real gradients are unchanged.  A2
        # uses the dynamic-graph ZeRO-1 profile: it performs the reduction after backward in model
        # order.  ZeRO-2 is forbidden for this graph because its per-parameter hooks can construct
        # different bucket boundaries on ranks receiving different operator classes.
        if self.graph_anchor_parameters:
            graph_anchor = (
                torch.stack(
                    [
                        parameter.reshape(-1)[0].to(device=total.device, dtype=total.dtype)
                        for parameter in self.graph_anchor_parameters
                    ]
                ).sum()
                * 0.0
            )
            total = total + graph_anchor
        return total


class ProductionA5EpisodeAdapter:
    def __init__(self, materializer: ProductionEpisodeMaterializer) -> None:
        self.materializer = materializer

    def __call__(self, inputs: Mapping[str, object]) -> tuple[MetaTTTEpisode, float]:
        prepared = cast(PreparedA5Record, inputs["prepared_a5"])
        episode = self.materializer.a5(prepared)
        self._begin_prefetch(episode)
        return episode, prepared.record.loss_weight

    def _begin_prefetch(self, episode: MetaTTTEpisode) -> None:
        video = self.materializer.video
        specs = tuple(
            cast(SupportChunkSpec, chunk.request.video_input)
            for chunk in (
                *((episode.prewarm_chunk,) if episode.prewarm_chunk is not None else ()),
                *episode.support_chunks,
            )
        )
        video.begin_prefetch(specs)

    def end_prefetch(self) -> None:
        self.materializer.video.end_prefetch()


def _build_runtime_preprocess_cache(
    backbone: LlamaFactoryBackboneBundle,
    config: ProductionTTTConfig,
) -> PreprocessCache | None:
    mode = PreprocessCacheMode(config.preprocess_cache_mode)
    if mode is PreprocessCacheMode.DISABLED:
        return None
    root = os.environ.get(config.preprocess_cache_root_env, "")
    max_gb = config.preprocess_cache_max_gb
    model_id = str(getattr(backbone.model_args, "model_name_or_path", "unknown-model"))
    revision = str(getattr(backbone.model_args, "revision", "unknown-revision"))
    processor_name = (
        type(backbone.processor).__qualname__ if backbone.processor is not None else "none"
    )
    raw_cache_dtype = str(config.preprocess_cache_dtype)
    cache_dtype = cast(PreprocessCacheStorageDtype, raw_cache_dtype)
    namespace_seed = "|".join(
        (
            model_id,
            revision,
            transformers.__version__,
            processor_name,
            str(backbone.project_config.video_preprocessing.processor_shortest_edge),
            str(backbone.project_config.video_preprocessing.processor_longest_edge),
            cache_dtype,
        )
    )
    namespace = hashlib.sha256(namespace_seed.encode("utf-8")).hexdigest()[:20]
    return PreprocessCache(
        root,
        max_bytes=int(max_gb * 1024**3),
        memory_entries=2,
        mode=mode,
        namespace=namespace,
        storage_dtype=cache_dtype,
    )


@dataclass(frozen=True, slots=True)
class _StateStack:
    fast: FastTTTAdapter
    qwen_adapter: Qwen3VLAdapter
    qwen_runtime: ProductionQwenRuntime
    state_bank: StructuredStateBank
    identity_bank: IdentityBank
    writer: StageABankWriter
    state_model: StateTTTModel


def _assemble_state_stack(
    config: ProjectConfig,
    qwen_model: nn.Module,
    tokenizer: object,
    materializer: VideoChunkMaterializer,
) -> _StateStack:
    fast = build_fast_ttt_adapter(config)
    qwen = Qwen3VLAdapter(
        qwen_model,
        config,
        adapter=fast,
        adapter_enabled=True,
        freeze_base=False,
    )
    qwen_runtime = ProductionQwenRuntime(qwen, materializer, tokenizer)
    query_runtime = ProductionQueryRuntime(build_query_encoder(config), tokenizer, qwen_model)
    query_runtime.bind_project_config(config)
    state_bank: StructuredStateBank = build_state_bank(config)
    identity_bank = build_identity_bank(config)
    writer = StageABankWriter(state_bank, identity_bank)
    reader = ProductionReaderRuntime(build_state_reader(config, cast(Any, tokenizer)))
    register_input_composer_tokens_with_audit(cast(Any, tokenizer), qwen_model)
    state_model = StateTTTModel(
        config,
        ModelComponents(
            visual_stage=qwen_runtime,
            query_encoder=query_runtime,
            composer=cast(ComposerStage, compose_inputs),
            qwen_prefill=qwen_runtime,
            qwen_generate=qwen_runtime,
            fast_adapter=FastVisualPassThrough(fast),
            spatial_encoder=ProductionSpatialRuntime(build_spatial_encoder(config)),
            temporal_encoder=ProductionTemporalRuntime(build_temporal_encoder(config)),
            observation_heads=ProductionObservationRuntime(build_observation_heads(config)),
            state_bank=state_bank,
            bank_writer=writer,
            retriever=build_state_retriever(config),
            reader=reader,
            resampler=build_state_resampler(config),
        ),
        ModelFeatureFlags(),
    )
    return _StateStack(
        fast=fast,
        qwen_adapter=qwen,
        qwen_runtime=qwen_runtime,
        state_bank=state_bank,
        identity_bank=identity_bank,
        writer=writer,
        state_model=state_model,
    )


def build_runtime(
    backbone: LlamaFactoryBackboneBundle,
    config: ProductionTTTConfig,
) -> object:
    """Built-in ``TTT_RUNTIME_FACTORY`` used by the H200 launch scripts."""

    from ttt_svcbench_qwen.llamafactory_trainer import (
        ProductionStage,
        ProductionTrainerRuntime,
    )

    stage = ProductionStage(config.stage)
    configure_runtime_metrics(config.runtime_trace_mode, config.runtime_trace_dir)
    project = backbone.project_config
    minimum_pixels, maximum_pixels = _video_pixel_bounds(backbone)
    preprocess_cache = _build_runtime_preprocess_cache(backbone, config)
    support_prefetch_depth = config.support_prefetch_depth
    if stage is ProductionStage.A5 and config.support_materialization == "segment_double_buffer":
        support_prefetch_depth = (1 + config.segment_prefetch_depth) * project.a5.truncation_horizon
    support_decode_coalesce = config.support_decode_coalesce
    chunk_materializer = VideoChunkMaterializer(
        project,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        preprocess_cache=preprocess_cache,
        cache_query_roles=config.cached_query_roles,
        prefetch_depth=support_prefetch_depth,
        decode_coalesce=support_decode_coalesce,
    )
    stack = _assemble_state_stack(project, backbone.model, backbone.tokenizer, chunk_materializer)
    fast = stack.fast
    writer = stack.writer
    state_model = stack.state_model
    associative_trainable = (
        stage is ProductionStage.A5 and config.a5_adaptation_mode == "meta_ttt"
    )
    for parameter in fast.collect_associative_parameters():
        parameter.requires_grad_(associative_trainable)
    official_weak_balancer = OfficialWeakOuterLossComposer(project.loss.official_weak_balance)
    outer = ProductionOuterModel(
        state_model,
        backbone.model,
        official_weak_balancer,
    )
    materializer = ProductionEpisodeMaterializer(backbone, writer, chunk_materializer)
    if stage is ProductionStage.A2:
        collator: object = A2PrefetchCollator(
            processor=backbone.processor,
            tokenizer=backbone.tokenizer,
            config=project,
            ttt_config=config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            preprocess_cache=preprocess_cache,
            prepared_episode_max_bytes=config.prepared_episode_max_bytes,
        )
        graph_anchor_parameters = tuple(
            parameter for parameter in outer.parameters() if parameter.requires_grad
        )
        runner = StageAEpisodeRunner(
            model=state_model,
            metric_builder=lambda _output, _supervision: ((), ()),
            query_encoder_reuse=config.query_encoder_reuse,
            query_activation_offload=config.query_activation_offload,
        )
        return ProductionTrainerRuntime(
            stage=stage,
            model=outer,
            train_dataset=(),
            eval_dataset=None,
            data_collator=cast(Any, collator),
            stage_a_loss_step=cast(
                Any,
                ProductionA2LossStep(
                    runner,
                    materializer,
                    graph_anchor_parameters,
                    project,
                    official_weak_balancer,
                ),
            ),
        )
    collator = A5PrefetchCollator(
        processor=backbone.processor,
        tokenizer=backbone.tokenizer,
        config=project,
        ttt_config=config,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        preprocess_cache=preprocess_cache,
    )
    meta_runner = MetaTTTEpisodeRunner(
        config=project,
        model=state_model,
        fast_controller=fast,
        runtime_resetter=lambda owner: _reset_meta_runtime(writer, owner),
        query_encoder_reuse=config.query_encoder_reuse,
        query_activation_offload=config.query_activation_offload,
        outer_composer=official_weak_balancer,
        adaptation_mode=config.a5_adaptation_mode,
    )
    return ProductionTrainerRuntime(
        stage=stage,
        a5_adaptation_mode=config.a5_adaptation_mode,
        model=outer,
        train_dataset=(),
        eval_dataset=None,
        data_collator=collator,
        meta_runner=meta_runner,
        episode_adapter=ProductionA5EpisodeAdapter(materializer),
    )


@dataclass(frozen=True, slots=True)
class StateTTTRuntimeBundle:
    """Complete production ownership graph for one online inference process."""

    config: ProjectConfig
    qwen_adapter: Qwen3VLAdapter
    state_model: StateTTTModel
    outer_model: ProductionOuterModel
    manager: object
    updater: object
    processor: object
    tokenizer: object
    video_materializer: VideoChunkMaterializer


def build_inference_runtime_bundle(
    *,
    model_root: str | Path,
    checkpoint: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    config_path: str | Path = "configs/model_state_ttt_8b.yaml",
    minimum_pixels: int = 16 * 16,
    maximum_pixels: int = 131_072,
) -> StateTTTRuntimeBundle:
    """Load local Qwen assets and assemble the sole online State-TTT runtime."""

    from ttt_svcbench_qwen.inference import OnlineTTTUpdater, PerVideoRuntimeManager

    root = Path(model_root).resolve()
    config = load_config(config_path)
    processor = cast(
        Any, transformers.AutoProcessor.from_pretrained(root, local_files_only=True)
    )
    tokenizer = processor.tokenizer
    model_type = cast(Any, transformers).Qwen3VLForConditionalGeneration
    qwen_model = cast(
        nn.Module,
        model_type.from_pretrained(
            root,
            dtype=dtype,
            local_files_only=True,
        ),
    )
    qwen_model.to(device=torch.device(device))
    qwen_model.eval()
    materializer = VideoChunkMaterializer(
        config,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        cache_query_visuals=False,
    )
    stack = _assemble_state_stack(config, qwen_model, tokenizer, materializer)
    state_model = stack.state_model
    outer = ProductionOuterModel(
        state_model,
        qwen_model,
        OfficialWeakOuterLossComposer(config.loss.official_weak_balance),
    )
    load_outer_checkpoint(outer, checkpoint)
    # The Qwen backbone was moved above before the state stack was constructed.
    # Move the complete checkpoint owner so Query/State/TTT modules
    # share the requested inference device and precision.
    outer.to(device=torch.device(device), dtype=dtype)
    outer.eval()
    manager = PerVideoRuntimeManager(
        fast_adapter=stack.fast,
        state_bank=stack.state_bank,
        identity_bank=stack.identity_bank,
    )
    return StateTTTRuntimeBundle(
        config=config,
        qwen_adapter=stack.qwen_adapter,
        state_model=state_model,
        outer_model=outer,
        manager=manager,
        updater=OnlineTTTUpdater(config, stack.fast),
        processor=processor,
        tokenizer=tokenizer,
        video_materializer=materializer,
    )


def _reset_meta_runtime(writer: StageABankWriter, owner: RuntimeOwner) -> BatchRuntimeState:
    return writer.reset(owner)


def _stage_inputs(
    visual: VisualStageOutput,
    query: QueryEncoderOutput,
    request: ObservationChunkRequest,
) -> tuple[
    QwenVisualOutput,
    CurrentChunkMaterialization | PreparedVisualCPU,
    QueryEncoderOutput,
    BatchRuntimeState,
]:
    return (
        cast(QwenVisualOutput, visual.value),
        cast(ProductionVisualAudit, visual.audit).chunk,
        query,
        _stage_runtime(request),
    )


def _current_chunk_input(
    value: object,
) -> ObservationSpec | CurrentChunkMaterialization | PreparedVisualCPU:
    if isinstance(
        value,
        (
            SupportChunkSpec,
            QueryObservationSpec,
            CurrentChunkMaterialization,
            PreparedVisualCPU,
        ),
    ):
        return value
    if isinstance(value, PreparedVisualChunk) and isinstance(
        value.source,
        CurrentChunkMaterialization,
    ):
        return value.source
    if isinstance(value, RawVisualChunk) and isinstance(
        value.source,
        CurrentChunkMaterialization,
    ):
        return value.source
    raise TypeError("production runtime requires one current chunk input")


def _stage_runtime(request: ObservationChunkRequest) -> BatchRuntimeState:
    return request.runtime_state


def _state_embedding_payload(
    request: QwenPrefillRequest,
    input_ids: Tensor,
) -> StateEmbeddingPayload | None:
    if request.state_position_mask is None or request.state_tokens is None:
        return None
    mask = request.state_position_mask.to(device=input_ids.device, dtype=torch.bool)
    tokens = request.state_tokens.to(device=input_ids.device)
    rows: list[Tensor] = []
    for row in range(mask.shape[0]):
        count = int(mask[row].sum().item())
        if count:
            rows.append(tokens[row, :count])
    if not rows:
        return None
    return StateEmbeddingPayload(input_ids, mask, torch.cat(rows, dim=0))


def _official_weak(query: ProductionQueryRecord) -> OfficialWeakSupervision:
    weak = query.weak
    return OfficialWeakSupervision(
        query_id=weak.query_id,
        operator=Operator(weak.operator),
        time_mode=TimeWindowMode(weak.time_mode),
        count=weak.count,
        query_time=weak.query_time,
        occurrence_points=weak.occurrence_points,
        occurrence_intervals=weak.occurrence_intervals,
    )


def _a2_support_chunk_specs(
    record: A2QueryRecord,
    video_path: Path,
) -> tuple[SupportChunkSpec, ...]:
    _, supports = adaptive_support_schedule(record.query.runtime.query_time)
    return tuple(
        SupportChunkSpec(
            chunk_id=f"{record.query.runtime.query_id}:a2:{index}",
            video_path=video_path,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            maximum_frames=chunk.maximum_frames,
            query_time=record.query.runtime.query_time,
        )
        for index, chunk in enumerate(supports)
    )


def _query_chunk_spec(
    chunk_id: str,
    video_path: Path,
    query_time: float,
    *,
    reset_soft_state: bool,
    config: ProductionTTTConfig,
    role: Literal["state_query", "answer_query"],
) -> QueryObservationSpec:
    mode: str
    maximum_frames: int
    if role == "state_query":
        mode = config.state_query_visual_mode
        maximum_frames = config.state_query_max_frames
    else:
        mode = config.answer_query_visual_mode
        maximum_frames = config.answer_query_max_frames
    end = query_time
    start = 0.0 if mode == "causal_prefix" else max(0.0, end - 8.0)
    return QueryObservationSpec(
        chunk_id=chunk_id,
        video_path=video_path,
        start_time=start,
        end_time=end,
        maximum_frames=maximum_frames,
        query_time=query_time,
        reset_soft_state=reset_soft_state,
        sampling_fps=config.query_sample_fps,
        sampling_policy=config.query_frame_sampling,
        decode_max_groups=config.query_decode_max_groups,
        query_role=role,
    )


def _prepare_answer_cpu(
    query: ProductionQueryRecord,
    spec: QueryObservationSpec,
    *,
    processor: object,
    tokenizer: object,
    config: ProjectConfig,
    minimum_pixels: int,
    maximum_pixels: int,
    preprocess_cache: PreprocessCache | None = None,
    source_dataset: str = "runtime",
) -> PreparedAnswerCPU:
    """Decode, preprocess and tokenize one Query using CPU-only objects.

    The visual processor is called once.  Prompt and full-answer IDs are then produced by the
    tokenizer after expanding the exact Qwen video placeholders from the shared grid/metadata.
    This keeps the token contract while removing the second resize/normalization/patchify pass.
    """

    answer_text = query.answer.answer if query.answer.answer is not None else str(query.weak.count)
    typed_processor = cast(Any, processor)
    fingerprint = (
        _build_preprocess_fingerprint(
            spec,
            config=config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            source_dataset=source_dataset,
        )
        if preprocess_cache is not None
        else None
    )
    cached = (
        preprocess_cache.get(fingerprint)
        if preprocess_cache is not None and fingerprint is not None
        else None
    )
    if cached is not None:
        materialized = _materialized_from_cached(spec, cached)
        frames = materialized.frames
    else:
        frames, timestamps = _decode_uniform_interval(spec, spec.sampling_fps)
        frames = _resize_to_pixel_budget(
            frames,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
        )
        pixels, grid = _process_video_once(typed_processor, frames)
        materialized = _build_materialized_query(spec, frames, timestamps, pixels, grid, config)
        if preprocess_cache is not None and preprocess_cache.writable:
            preprocess_cache.put(
                cast(PreprocessFingerprint, fingerprint),
                _cached_from_materialized(materialized),
            )

    prompt_messages = [_user_message(query.runtime.question)]
    full_messages = [*prompt_messages, {"role": "assistant", "content": answer_text}]
    prompt_text = typed_processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = typed_processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_expanded = _expand_qwen_video_placeholders(
        typed_processor, prompt_text, materialized.video_grid_thw, frames.shape[0]
    )
    full_expanded = _expand_qwen_video_placeholders(
        typed_processor, full_text, materialized.video_grid_thw, frames.shape[0]
    )
    prompt_ids, _ = _tokenize_text_only(tokenizer, prompt_expanded)
    full_ids, full_mask = _tokenize_text_only(tokenizer, full_expanded)
    full_ids = full_ids.to(torch.int64)
    prompt_ids = prompt_ids.to(torch.int64)
    prompt_length = int(prompt_ids.shape[1])
    labels = torch.full_like(full_ids, -100)
    labels[:, prompt_length:] = full_ids[:, prompt_length:]
    number_mask = torch.zeros_like(labels, dtype=torch.bool)
    count_ids = _token_ids(tokenizer, str(query.weak.count))
    _mark_last_subsequence(number_mask[0], full_ids[0], count_ids, lower=prompt_length)
    provenance = (
        TargetProvenance.OFFICIAL_EXPLICIT
        if query.answer.answer is not None
        else TargetProvenance.OFFICIAL_WEAK
    )
    target_labels = AnswerTargetLabels(
        base_labels=labels,
        base_number_token_mask=number_mask,
        target_counts=torch.tensor([query.weak.count], dtype=torch.int64),
        answer_provenance=(provenance,),
        count_provenance=(TargetProvenance.OFFICIAL_WEAK,),
    )
    prepared_visual = _compact_materialized_chunk(materialized)
    return PreparedAnswerCPU(
        spec=spec,
        base_input_ids=full_ids,
        base_attention_mask=full_mask,
        target_labels=target_labels,
        materialized_query=prepared_visual,
    )


def _build_preprocess_fingerprint(
    spec: ObservationSpec,
    *,
    config: ProjectConfig,
    minimum_pixels: int,
    maximum_pixels: int,
    source_dataset: str = "runtime",
) -> PreprocessFingerprint:
    """Build the shared A2/A5 visual key from immutable media/config inputs only."""

    try:
        root = Path(os.environ["SVCBENCH_VIDEO_ROOT"]).resolve()
        relative = spec.video_path.resolve().relative_to(root).as_posix()
    except (KeyError, OSError, ValueError):
        relative = spec.video_path.as_posix()
    return build_fingerprint(
        source_dataset=source_dataset,
        relative_video_path=relative,
        video_path=spec.video_path,
        start_time=spec.start_time,
        end_time=spec.end_time,
        maximum_frames=spec.maximum_frames,
        sample_fps=_sample_fps_for(spec, config),
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        patch_size=config.video_preprocessing.patch_size,
        temporal_patch_size=config.video_preprocessing.temporal_patch_size,
        spatial_merge_size=config.video_preprocessing.spatial_merge_size,
        transformers_version=transformers.__version__,
        observation_role=spec.observation_role,
        frame_sampling=spec.frame_sampling,
    )


def _cached_from_materialized(materialized: CurrentChunkMaterialization) -> CachedChunk:
    return CachedChunk(
        frames=materialized.frames,
        frame_timestamps=materialized.frame_timestamps,
        pixel_values_videos=materialized.pixel_values_videos,
        video_grid_thw=materialized.video_grid_thw,
        tubelet_timestamps=materialized.tubelet_timestamps,
        tubelet_valid_mask=materialized.tubelet_valid_mask,
        tubelet_position_ids=materialized.tubelet_position_ids,
    )


def _compact_materialized_chunk(materialized: CurrentChunkMaterialization) -> PreparedVisualCPU:
    return PreparedVisualCPU(
        spec=materialized.spec,
        frame_timestamps=materialized.frame_timestamps,
        tubelet_timestamps=materialized.tubelet_timestamps,
        tubelet_valid_mask=materialized.tubelet_valid_mask,
        tubelet_position_ids=materialized.tubelet_position_ids,
        pixel_values_videos=materialized.pixel_values_videos,
        video_grid_thw=materialized.video_grid_thw,
    )


def _materialized_from_cached(
    spec: ObservationSpec, cached: CachedChunk
) -> CurrentChunkMaterialization:
    return CurrentChunkMaterialization(
        spec=spec,
        frames=cached.frames,
        frame_timestamps=cached.frame_timestamps,
        tubelet_timestamps=cached.tubelet_timestamps,
        tubelet_valid_mask=cached.tubelet_valid_mask,
        tubelet_position_ids=cached.tubelet_position_ids,
        pixel_values_videos=cached.pixel_values_videos,
        video_grid_thw=cached.video_grid_thw,
    )


def _build_materialized_query(
    spec: QueryObservationSpec,
    frames: Tensor,
    timestamps: Tensor,
    pixels: Tensor,
    grid: Tensor,
    config: ProjectConfig,
) -> CurrentChunkMaterialization:
    tubelet_times = timestamps.reshape(-1, 2).amax(dim=1)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels.squeeze(0)
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    return CurrentChunkMaterialization(
        spec=spec,
        frames=frames,
        frame_timestamps=timestamps,
        tubelet_timestamps=tubelet_times.unsqueeze(0),
        tubelet_valid_mask=torch.ones((1, frames.shape[0] // 2), dtype=torch.bool),
        tubelet_position_ids=_strict_tubelet_positions(
            tubelet_times, sample_fps=_sample_fps_for(spec, config)
        ).unsqueeze(0),
        pixel_values_videos=pixels,
        video_grid_thw=grid.to(torch.int64),
    )


def _process_video_once(processor: Any, frames: Tensor) -> tuple[Tensor, Tensor]:
    video_processor = processor.video_processor
    raw = video_processor(
        videos=[frames],
        do_sample_frames=False,
        return_tensors="pt",
        return_metadata=True,
    )
    pixels = _processor_tensor(raw, "pixel_values_videos")
    grid = _processor_tensor(raw, "video_grid_thw").to(torch.int64)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels.squeeze(0)
    return pixels.contiguous(), grid.contiguous()


def _tokenize_text_only(tokenizer: object, text: str) -> tuple[Tensor, Tensor]:
    raw = cast(Any, tokenizer)(
        [text],
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    return _processor_tensor(raw, "input_ids").to(torch.int64), _processor_tensor(
        raw, "attention_mask"
    )


def _expand_qwen_video_placeholders(
    processor: Any,
    text: str,
    grid: Tensor,
    frame_count: int,
) -> str:
    """Mirror Qwen3VLProcessor's timestamped placeholder expansion without patchifying twice."""

    video_token = str(processor.video_token)
    vision_start = str(processor.vision_start_token)
    vision_end = str(processor.vision_end_token)
    video_processor = processor.video_processor
    merge_size = int(video_processor.merge_size)
    grid_row = grid[0].tolist()
    temporal = int(grid_row[0])
    frame_seqlen = int(grid_row[1] * grid_row[2]) // (merge_size**2)
    indices = list(range(max(frame_count, temporal * 2)))
    if len(indices) % merge_size:
        indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))
    timestamps = list(processor._calculate_timestamps(indices, 24, merge_size))
    timestamps = (timestamps + [timestamps[-1]] * temporal)[:temporal]
    placeholder = "".join(
        f"<{float(t):.1f} seconds>{vision_start}" + ("<|placeholder|>" * frame_seqlen) + vision_end
        for t in timestamps
    )
    wrapped = f"{vision_start}{video_token}{vision_end}"
    text = text.replace(wrapped, placeholder, 1)
    return text.replace("<|placeholder|>", video_token)


def _resolve_video_path(source_dataset: str, relative_path: str) -> Path:
    root = os.environ.get("SVCBENCH_VIDEO_ROOT", "")
    root_path = Path(root).resolve()
    direct = (root_path / relative_path).resolve()
    nested = (root_path / source_dataset / relative_path).resolve()
    for candidate in (direct, nested):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "manifest video was not found as either a converted or source-layout path: "
        f"{direct}, {nested}"
    )


def _user_message(question: str) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": _ANSWER_INSTRUCTION.format(question=question)},
        ],
    }


def _processor_tensor(raw: object, key: str) -> Tensor:
    return cast(Tensor, cast(Mapping[str, Any], raw)[key])


def _token_ids(tokenizer: object, text: str) -> tuple[int, ...]:
    values = cast(Any, tokenizer).encode(text, add_special_tokens=False)
    return tuple(int(value) for value in values)


def _mark_last_subsequence(
    mask: Tensor, values: Tensor, target: tuple[int, ...], *, lower: int
) -> None:
    if not target:
        return
    sequence = tuple(int(value) for value in values.tolist())
    found = -1
    for start in range(lower, len(sequence) - len(target) + 1):
        if sequence[start : start + len(target)] == target:
            found = start
    if found >= 0:
        mask[found : found + len(target)] = True


def _video_pixel_bounds(backbone: LlamaFactoryBackboneBundle) -> tuple[int, int]:
    """Read LLaMA-Factory's processor controls from ModelArguments, not DataArguments."""

    owner = backbone.model_args
    maximum = getattr(owner, "video_max_pixels", None)
    minimum = getattr(owner, "video_min_pixels", None)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        maximum = 262_144
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        minimum = 16 * 16
    return minimum, maximum


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _sample_fps_for(spec: ObservationSpec, config: ProjectConfig) -> float:
    return (
        spec.sampling_fps
        if isinstance(spec, QueryObservationSpec)
        else config.video_preprocessing.sample_fps
    )


def _decode_uniform_interval(
    spec: ObservationSpec,
    sample_fps: float,
) -> tuple[Tensor, Tensor]:
    target_times = (
        _llamafactory_query_target_times(spec, sample_fps)
        if isinstance(spec, QueryObservationSpec)
        else _uniform_target_times(spec, sample_fps)
    )
    # Re-seeking for every target is efficient for a 2,048-second geometric interval but very
    # expensive for the overlapping 8-second recent windows (up to 16 keyframe seeks instead of
    # one short scan).  Stream short intervals once; retain target frames only, so residency is
    # still bounded by the current chunk's dynamic frame cap.
    if spec.end_time - spec.start_time <= 16.0 + 1.0e-6:
        frames, timestamps = _decode_targets_streaming(spec, target_times)
    elif isinstance(spec, QueryObservationSpec):
        try:
            frames, timestamps = _decode_query_targets_grouped(
                spec,
                target_times,
                max_groups=spec.decode_max_groups,
            )
        except _TargetSeekUnavailable:
            frames, timestamps = _decode_targets_streaming(spec, target_times)
    else:
        try:
            frames, timestamps = _decode_targets_with_seek(spec, target_times)
        except _TargetSeekUnavailable:
            # A small minority of containers do not expose a seekable timestamp index.  The
            # streaming decoder still converts/retains only target frames; host memory therefore
            # remains bounded by ``maximum_frames``.
            frames, timestamps = _decode_targets_streaming(spec, target_times)
    frames, timestamps = _finalize_decoded_frames(frames, timestamps, spec)
    return torch.stack(frames).contiguous(), torch.tensor(timestamps, dtype=torch.float64)


def _decode_coalesced_intervals(
    specs: tuple[SupportChunkSpec, ...], sample_fps: float
) -> dict[str, tuple[Tensor, Tensor]]:
    """Decode multiple overlapping intervals from one PyAV demux pass.

    Only target frames are converted and retained.  The implementation intentionally mirrors the
    streaming decoder's right-closed boundary and sparse/VFR handling; it is an optimization of
    container setup/demux, not a different sampling policy.
    """

    if not specs:
        return {}
    path = specs[0].video_path
    target_map = {spec.chunk_id: _uniform_target_times(spec, sample_fps) for spec in specs}
    frame_map: dict[str, list[Tensor]] = {spec.chunk_id: [] for spec in specs}
    time_map: dict[str, list[float]] = {spec.chunk_id: [] for spec in specs}
    next_target = {spec.chunk_id: 0 for spec in specs}
    last_candidate: dict[str, tuple[Any, float] | None] = {spec.chunk_id: None for spec in specs}
    min_start = min(spec.start_time for spec in specs)
    max_end = max(spec.end_time for spec in specs)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is not None and min_start > 0.0:
            try:
                offset = int(max(0.0, min_start - 1.0) / float(stream.time_base))
                container.seek(offset, stream=stream, backward=True, any_frame=False)
            except (OSError, ValueError, av.error.FFmpegError):
                pass
        for frame in container.decode(stream):
            timestamp = av_frame_timestamp(frame)
            if timestamp < min_start - 1.0e-9:
                continue
            if timestamp > max_end + 1.0e-9:
                break
            converted: Tensor | None = None
            for spec in specs:
                key = spec.chunk_id
                if timestamp < spec.start_time - 1.0e-9 or timestamp > spec.end_time + 1.0e-9:
                    continue
                previous = last_candidate[key]
                if previous is not None and timestamp <= previous[1] + 1.0e-9:
                    continue
                last_candidate[key] = (frame, timestamp)
                target_index = next_target[key]
                targets = target_map[key]
                if target_index >= len(targets) or timestamp + 1.0e-9 < targets[target_index]:
                    continue
                if converted is None:
                    converted = _rgb_frame_tensor(frame)
                frame_map[key].append(converted)
                time_map[key].append(timestamp)
                while target_index < len(targets) and timestamp + 1.0e-9 >= targets[target_index]:
                    target_index += 1
                next_target[key] = target_index
    for spec in specs:
        key = spec.chunk_id
        candidate = last_candidate[key]
        if (
            next_target[key] < len(target_map[key])
            and candidate is not None
            and (not time_map[key] or candidate[1] > time_map[key][-1] + 1.0e-9)
        ):
            frame_map[key].append(_rgb_frame_tensor(candidate[0]))
            time_map[key].append(candidate[1])
        results = _finalize_decoded_frames(frame_map[key], time_map[key], spec)
        frame_map[key] = list(results[0])
        time_map[key] = list(results[1])
    return {
        spec.chunk_id: (
            torch.stack(frame_map[spec.chunk_id]).contiguous(),
            torch.tensor(time_map[spec.chunk_id], dtype=torch.float64),
        )
        for spec in specs
    }


def _uniform_target_times(spec: ObservationSpec, sample_fps: float) -> list[float]:
    desired = min(
        spec.maximum_frames,
        max(2, int(math.floor((spec.end_time - spec.start_time) * sample_fps))),
    )
    desired -= desired % 2
    return cast(
        list[float],
        torch.linspace(spec.start_time, spec.end_time, desired, dtype=torch.float64).tolist(),
    )


def _llamafactory_uniform_frame_indices(
    *,
    total_frames: int,
    duration: float,
    video_fps: float,
    video_maxlen: int,
) -> tuple[int, ...]:
    """Mirror LLaMA-Factory commit 523f801 ``_get_video_sample_indices`` exactly."""

    if total_frames == 0:
        return tuple(range(video_maxlen))
    sample_frames = max(1, math.floor(duration * video_fps))
    sample_frames = min(total_frames, video_maxlen, sample_frames)
    return tuple(
        int(value) for value in torch.linspace(0, total_frames - 1, sample_frames).tolist()
    )


def _llamafactory_query_target_times(
    spec: QueryObservationSpec,
    sample_fps: float,
) -> list[float]:
    """Map LLaMA-Factory's uniform frame indices onto one causal source-video interval."""

    try:
        with av.open(str(spec.video_path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            source_fps = float(rate) if rate is not None else 0.0
            total_source_frames = int(stream.frames or 0)
    except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError):
        source_fps = 0.0
        total_source_frames = 0
    if source_fps <= 0.0 or not math.isfinite(source_fps) or total_source_frames <= 0:
        return _uniform_target_times(spec, sample_fps)

    first_index = max(0, int(math.ceil(spec.start_time * source_fps - 1.0e-9)))
    stop_index = min(
        total_source_frames,
        max(first_index + 1, int(math.floor(spec.end_time * source_fps + 1.0e-9))),
    )
    available = stop_index - first_index
    if available <= 0:
        return [spec.start_time]
    relative = _llamafactory_uniform_frame_indices(
        total_frames=available,
        duration=spec.end_time - spec.start_time,
        video_fps=sample_fps,
        video_maxlen=spec.maximum_frames,
    )
    # Qwen temporal patches require pairs. Preserve the full temporal span by recomputing an
    # even uniform grid; only a truly single-frame causal prefix is duplicated after decoding.
    if len(relative) > 1 and len(relative) % 2:
        relative = tuple(
            int(value) for value in torch.linspace(0, available - 1, len(relative) - 1).tolist()
        )
    return [min(spec.query_time, (first_index + index) / source_fps) for index in relative]


def _finalize_decoded_frames(
    frames: list[Tensor], timestamps: list[float], spec: ObservationSpec
) -> tuple[list[Tensor], list[float]]:
    padded_single_frame = len(frames) == 1
    if padded_single_frame:
        frames.append(frames[0].clone())
        timestamps.append(timestamps[0])
    elif len(frames) % 2:
        frames.pop()
        timestamps.pop()
    return frames, timestamps


class _TargetSeekUnavailable(RuntimeError):
    """Signal that a media container cannot service timestamp-targeted decoding."""


def _decode_query_targets_grouped(
    spec: QueryObservationSpec,
    target_times: Sequence[float],
    *,
    max_groups: int,
) -> tuple[list[Tensor], list[float]]:
    """Decode a Query with at most ``max_groups`` backward keyframe seeks.

    Groups contain contiguous target timestamps. Within each group a single forward scan applies
    the same nearest-frame rule as the legacy per-target decoder: compare the closest frame before
    and after the target, prefer the earlier frame on a tie, and never reuse a timestamp selected
    by an earlier target. Only selected frames are converted to RGB.
    """

    groups = _balanced_target_groups(target_times, max_groups=max_groups)
    if not groups:
        return [], []
    frames: list[Tensor] = []
    timestamps: list[float] = []
    try:
        with av.open(str(spec.video_path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise _TargetSeekUnavailable("video stream exposes no seek time base")
            time_base = float(stream.time_base)
            if not math.isfinite(time_base) or time_base <= 0.0:
                raise _TargetSeekUnavailable("video stream exposes an invalid seek time base")
            minimum_timestamp: float | None = None
            for group in groups:
                offset = int(max(0.0, group[0] - 1.0e-6) / time_base)
                container.seek(offset, stream=stream, backward=True, any_frame=False)
                selected = _decode_nearest_target_group(
                    container,
                    stream,
                    targets=group,
                    start_time=spec.start_time,
                    end_time=spec.end_time,
                    minimum_timestamp=minimum_timestamp,
                )
                for frame, timestamp in selected:
                    frames.append(_rgb_frame_tensor(frame))
                    timestamps.append(timestamp)
                    minimum_timestamp = timestamp
    except _TargetSeekUnavailable:
        raise
    except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError) as error:
        raise _TargetSeekUnavailable(str(error)) from error
    return frames, timestamps


def _balanced_target_groups(
    target_times: Sequence[float], *, max_groups: int
) -> tuple[tuple[float, ...], ...]:
    targets = tuple(float(value) for value in target_times)
    if not targets:
        return ()
    group_count = min(max_groups, len(targets))
    base_size, extra = divmod(len(targets), group_count)
    groups: list[tuple[float, ...]] = []
    start = 0
    for group_index in range(group_count):
        size = base_size + int(group_index < extra)
        groups.append(targets[start : start + size])
        start += size
    return tuple(groups)


def _decode_nearest_target_group(
    container: Any,
    stream: Any,
    *,
    targets: Sequence[float],
    start_time: float,
    end_time: float,
    minimum_timestamp: float | None,
) -> list[tuple[Any, float]]:
    selected: list[tuple[Any, float]] = []
    target_index = 0
    before: tuple[Any, float] | None = None
    decoded = container.decode(stream)
    try:
        for frame in decoded:
            timestamp = av_frame_timestamp(frame)
            if timestamp < start_time - 1.0e-9:
                continue
            if timestamp > end_time + 1.0e-9:
                break
            if minimum_timestamp is not None and timestamp <= minimum_timestamp + 1.0e-9:
                continue
            current = (frame, timestamp)
            while target_index < len(targets):
                target = targets[target_index]
                if timestamp < target - 1.0e-9:
                    before = current
                    break
                candidates = (current,) if before is None else (before, current)
                candidate = min(
                    candidates,
                    key=lambda item: (abs(item[1] - target), item[1]),
                )
                selected.append(candidate)
                minimum_timestamp = candidate[1]
                target_index += 1
                if timestamp > candidate[1] + 1.0e-9:
                    before = current
                    continue
                before = None
                break
            if target_index >= len(targets):
                break
    finally:
        close = getattr(decoded, "close", None)
        if callable(close):
            close()
    if (
        target_index < len(targets)
        and before is not None
        and (minimum_timestamp is None or before[1] > minimum_timestamp + 1.0e-9)
    ):
        selected.append(before)
    return selected


def _decode_targets_with_seek(
    spec: ObservationSpec,
    target_times: Sequence[float],
) -> tuple[list[Tensor], list[float]]:
    """Decode nearest unique frames around target timestamps with bounded residency.

    Each target performs a backward keyframe seek and converts at most one frame to RGB.  Long
    geometric Support intervals therefore do not accumulate every intervening decoded frame.
    """

    frames: list[Tensor] = []
    timestamps: list[float] = []
    with av.open(str(spec.video_path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise _TargetSeekUnavailable("video stream exposes no seek time base")
        time_base = float(stream.time_base)
        if not math.isfinite(time_base) or time_base <= 0.0:
            raise _TargetSeekUnavailable("video stream exposes an invalid seek time base")
        for target in target_times:
            try:
                candidate = _decode_nearest_seek_target(
                    container,
                    stream,
                    target=target,
                    start_time=spec.start_time,
                    end_time=spec.end_time,
                    minimum_timestamp=timestamps[-1] if timestamps else None,
                    time_base=time_base,
                )
            except (OSError, ValueError, av.error.FFmpegError) as error:
                raise _TargetSeekUnavailable(str(error)) from error
            if candidate is None:
                continue
            frame, timestamp = candidate
            frames.append(_rgb_frame_tensor(frame))
            timestamps.append(timestamp)
    return frames, timestamps


def _decode_nearest_seek_target(
    container: Any,
    stream: Any,
    *,
    target: float,
    start_time: float,
    end_time: float,
    minimum_timestamp: float | None,
    time_base: float,
) -> tuple[Any, float] | None:
    offset = int(max(0.0, target - 1.0e-6) / time_base)
    container.seek(offset, stream=stream, backward=True, any_frame=False)
    before: tuple[Any, float] | None = None
    decoded = container.decode(stream)
    try:
        for frame in decoded:
            timestamp = av_frame_timestamp(frame)
            if timestamp < start_time - 1.0e-9:
                continue
            if timestamp > end_time + 1.0e-9:
                break
            if minimum_timestamp is not None and timestamp <= minimum_timestamp + 1.0e-9:
                continue
            if timestamp < target - 1.0e-9:
                before = (frame, timestamp)
                continue
            after = (frame, timestamp)
            if before is None:
                return after
            # Prefer the earlier frame on an exact tie.  Both choices remain inside the
            # right-closed causal interval.
            return min((before, after), key=lambda item: (abs(item[1] - target), item[1]))
    finally:
        close = getattr(decoded, "close", None)
        if callable(close):
            close()
    return before


def _decode_targets_streaming(
    spec: ObservationSpec,
    target_times: Sequence[float],
) -> tuple[list[Tensor], list[float]]:
    """Memory-bounded streaming decoder for media without reliable random access."""

    frames: list[Tensor] = []
    timestamps: list[float] = []
    target_index = 0
    last_candidate: tuple[Any, float] | None = None
    with av.open(str(spec.video_path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is not None and spec.start_time > 0.0:
            try:
                offset = int(max(0.0, spec.start_time - 1.0) / float(stream.time_base))
                container.seek(offset, stream=stream, backward=True, any_frame=False)
            except (OSError, ValueError, av.error.FFmpegError):
                # Decoding from the beginning is slower but remains bounded in memory.
                pass
        for frame in container.decode(stream):
            timestamp = av_frame_timestamp(frame)
            if timestamp < spec.start_time - 1.0e-9:
                continue
            if timestamp > spec.end_time + 1.0e-9:
                break
            if last_candidate is not None and timestamp <= last_candidate[1] + 1.0e-9:
                continue
            last_candidate = (frame, timestamp)
            if target_index >= len(target_times) or timestamp + 1.0e-9 < target_times[target_index]:
                continue
            frames.append(_rgb_frame_tensor(frame))
            timestamps.append(timestamp)
            while (
                target_index < len(target_times)
                and timestamp + 1.0e-9 >= target_times[target_index]
            ):
                target_index += 1
        if (
            target_index < len(target_times)
            and last_candidate is not None
            and (not timestamps or last_candidate[1] > timestamps[-1] + 1.0e-9)
        ):
            frames.append(_rgb_frame_tensor(last_candidate[0]))
            timestamps.append(last_candidate[1])
    return frames[: len(target_times)], timestamps[: len(target_times)]


def _rgb_frame_tensor(frame: Any) -> Tensor:
    return torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()


def _resize_to_pixel_budget(
    frames: Tensor,
    *,
    minimum_pixels: int,
    maximum_pixels: int,
) -> Tensor:
    height, width = frames.shape[-2:]
    area = height * width
    if minimum_pixels <= area <= maximum_pixels:
        return frames
    upscale = area < minimum_pixels
    target_pixels = minimum_pixels if upscale else maximum_pixels
    scale = math.sqrt(target_pixels / area)
    if upscale:
        target_h = max(32, math.ceil(height * scale / 32.0) * 32)
        target_w = max(32, math.ceil(width * scale / 32.0) * 32)
    else:
        target_h = max(32, int(height * scale) // 32 * 32)
        target_w = max(32, int(width * scale) // 32 * 32)
    resized = (
        F.interpolate(
            frames.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        .round()
        .clamp_(0, 255)
    )
    return resized.to(dtype=torch.uint8)


def _strict_tubelet_positions(times: Tensor, *, sample_fps: float) -> Tensor:
    raw = torch.floor(times * sample_fps / 2.0).to(torch.int64)
    values: list[int] = []
    previous = -1
    for value in raw.tolist():
        current = max(int(value), previous + 1)
        values.append(current)
        previous = current
    return torch.tensor(values, dtype=torch.int64)


__all__ = [
    "A2PrefetchCollator",
    "A5PrefetchCollator",
    "CurrentChunkMaterialization",
    "QueryObservationSpec",
    "SupportChunkSpec",
    "PreparedA2Record",
    "PreparedA5Record",
    "PreparedAnswerCPU",
    "PreparedVisualCPU",
    "ProductionVisualAudit",
    "VideoChunkMaterializer",
    "build_runtime",
]
