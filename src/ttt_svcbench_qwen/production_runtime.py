"""Concrete H200 runtime for production A2 and A5 training.

This is the numerical/materialization counterpart of the LLaMA-Factory bridge.  Support video
chunks stay as lightweight interval specifications until their step executes.  Exactly one
current chunk is then decoded, processed and handed to Qwen; no historical visual-feature list is
representable at this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import weakref
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import av
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ttt_svcbench_qwen.config import MetaTTTVariant, ProjectConfig, StageAVariant, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    AdaptiveChunkSpec,
    ProductionQueryRecord,
    adaptive_support_schedule,
)
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter
from ttt_svcbench_qwen.identity_bank import build_identity_bank
from ttt_svcbench_qwen.input_composer import (
    ComposedInput,
    compose_inputs,
    map_teacher_forced_targets,
    register_input_composer_tokens_with_audit,
)
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    ReaderCountMetricInput,
    TemporalPredictor,
    build_temporal_predictor,
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
)
from ttt_svcbench_qwen.observation_heads import (
    ObservationHeads,
    ObservationOutputs,
    build_observation_heads,
)
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.preprocess_cache import (
    CachedChunk,
    PreprocessCache,
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
    CurrentChunkVisualTokenAudit,
    MergedVideoMetadata,
    PreparedVideoFeatures,
    Qwen3VLAdapter,
    QwenVisualOutput,
    StateEmbeddingPayload,
    audit_current_chunk_visual_tokens,
)
from ttt_svcbench_qwen.runtime_metrics import (
    configure_runtime_metrics,
    trace_cuda_phase,
    trace_event,
)
from ttt_svcbench_qwen.stage_a_runtime import StageABankWriter
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    OfficialWeakSupervision,
    OfficialWeakTargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import StructuredStateBank, build_state_bank
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
from ttt_svcbench_qwen.video_preprocessing import QwenVideoPreprocessor

_ANSWER_INSTRUCTION = (
    "The video chunk ends at the question time. Answer the question using the structured "
    "state and output only the answer, with no explanation.\nQuestion: {question}"
)


def _loader_trace(event: str, **fields: object) -> None:
    """Emit one buffered rank/worker-local timing event."""

    trace_event(event, **fields)


def _cache_stats(owner: object) -> dict[str, int | bool]:
    cache = getattr(owner, "preprocess_cache", None)
    if isinstance(cache, PreprocessCache):
        return cache.stats()
    return {}


@dataclass(frozen=True, slots=True)
class CurrentChunkSpec:
    """A lightweight, label-free description of exactly one current chunk."""

    chunk_id: str
    video_path: Path
    start_time: float
    end_time: float
    maximum_frames: int
    query_time: float
    reset_soft_state: bool = False
    history_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.video_path.is_file():
            raise FileNotFoundError(f"current chunk video does not exist: {self.video_path}")
        if (
            not math.isfinite(self.start_time)
            or not math.isfinite(self.end_time)
            or self.start_time < 0.0
            or self.end_time <= self.start_time
            or self.end_time > self.query_time + 1.0e-6
        ):
            raise ValueError("current chunk must satisfy 0 <= start < end <= query_time")
        if self.maximum_frames < 2 or self.maximum_frames > 16:
            raise ValueError("production current chunks permit 2..16 frames")
        if self.history_chunk_ids:
            raise ValueError("a current chunk specification cannot carry historical chunks")


@dataclass(frozen=True, slots=True)
class CurrentChunkMaterialization:
    spec: CurrentChunkSpec
    frames: Tensor
    frame_timestamps: Tensor
    tubelet_timestamps: Tensor
    tubelet_valid_mask: Tensor
    tubelet_position_ids: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[1] != 3:
            raise ValueError("materialized current frames must be [F, 3, H, W]")
        frame_count = self.frames.shape[0]
        if frame_count < 2 or frame_count > self.spec.maximum_frames or frame_count % 2:
            raise ValueError("materialized current frame count must be even within [2, max]")
        if self.frame_timestamps.shape != (frame_count,):
            raise ValueError("current frame timestamps must align to frames")
        tubelets = frame_count // 2
        if (
            self.tubelet_timestamps.shape != (1, tubelets)
            or self.tubelet_valid_mask.shape != (1, tubelets)
            or self.tubelet_position_ids.shape != (1, tubelets)
        ):
            raise ValueError("current tubelet metadata must be [1, F/2]")
        if self.tubelet_valid_mask.dtype != torch.bool:
            raise TypeError("current tubelet validity must use bool dtype")
        if self.pixel_values_videos.ndim != 2 or not torch.is_floating_point(
            self.pixel_values_videos
        ):
            raise ValueError("current Qwen pixels must be packed floating [N_patch, D]")
        if self.video_grid_thw.shape != (1, 3):
            raise ValueError("current Qwen video grid must be [1, 3]")
        if int(self.video_grid_thw[0, 0].item()) != tubelets:
            raise ValueError("Qwen temporal grid must equal the current chunk tubelet count")


def _bind_runtime_query(
    query: RuntimeQueryInput,
    video_path: Path,
    episode_nonce: int,
) -> RuntimeQueryInput:
    return replace(query, video=video_path, episode_nonce=episode_nonce)


@dataclass(frozen=True, slots=True)
class PreparedAnswerCPU:
    """CPU-only Query tensors safe to build in a DataLoader worker."""

    spec: CurrentChunkSpec
    base_input_ids: Tensor
    base_attention_mask: Tensor
    target_labels: AnswerTargetLabels
    materialized_query: CurrentChunkMaterialization

    def __post_init__(self) -> None:
        if self.materialized_query.spec != self.spec:
            raise ValueError("prepared answer/current chunk specifications must match")
        if (
            self.base_input_ids.ndim != 2
            or self.base_attention_mask.shape != self.base_input_ids.shape
        ):
            raise ValueError("prepared answer IDs/mask must be aligned rank-2 tensors")
        if self.target_labels.base_labels.shape != self.base_input_ids.shape:
            raise ValueError("prepared answer labels must align to input IDs")

    def pin_memory(self) -> PreparedAnswerCPU:
        """Pin only tensors that cross the CPU-to-GPU boundary.

        Raw uint8 frames remain pageable: they exist only for the causal audit and are never
        copied to the accelerator.  Keeping them unpinned bounds the per-rank pinned-memory pool.
        """

        started = time.perf_counter()
        chunk = replace(
            self.materialized_query,
            frame_timestamps=self.materialized_query.frame_timestamps.pin_memory(),
            tubelet_timestamps=self.materialized_query.tubelet_timestamps.pin_memory(),
            tubelet_valid_mask=self.materialized_query.tubelet_valid_mask.pin_memory(),
            tubelet_position_ids=self.materialized_query.tubelet_position_ids.pin_memory(),
            pixel_values_videos=self.materialized_query.pixel_values_videos.pin_memory(),
            video_grid_thw=self.materialized_query.video_grid_thw.pin_memory(),
        )
        labels = replace(
            self.target_labels,
            base_labels=self.target_labels.base_labels.pin_memory(),
            base_number_token_mask=self.target_labels.base_number_token_mask.pin_memory(),
            target_counts=self.target_labels.target_counts.pin_memory(),
        )
        result = replace(
            self,
            base_input_ids=self.base_input_ids.pin_memory(),
            base_attention_mask=self.base_attention_mask.pin_memory(),
            target_labels=labels,
            materialized_query=chunk,
        )
        _loader_trace(
            "pin_memory",
            tensor_bytes=_prepared_answer_bytes(result),
            seconds=time.perf_counter() - started,
        )
        return result


@dataclass(frozen=True, slots=True)
class PreparedA2Record:
    """One manifest row plus its prefetched, label-separated Query tensors."""

    record: A2QueryRecord
    answer: PreparedAnswerCPU

    def __post_init__(self) -> None:
        expected = f"{self.record.query.runtime.query_id}:query"
        if self.answer.spec.chunk_id != expected:
            raise ValueError("prepared A2 answer does not belong to its manifest Query")

    def pin_memory(self) -> PreparedA2Record:
        return replace(self, answer=self.answer.pin_memory())


@dataclass(frozen=True, slots=True)
class PreparedA5Record:
    """One A5 episode plus CPU-prepared Query answer inputs."""

    record: A5EpisodeRecord
    query_answers: tuple[PreparedAnswerCPU, ...]

    def __post_init__(self) -> None:
        if len(self.query_answers) != len(self.record.queries):
            raise ValueError("prepared A5 answers must align to every Query point")
        for index, prepared in enumerate(self.query_answers):
            expected_prefix = f"{self.record.episode_id}:q{index}"
            if prepared.spec.chunk_id != expected_prefix:
                raise ValueError("prepared A5 answer does not belong to its Query point")

    def pin_memory(self) -> PreparedA5Record:
        return replace(
            self,
            query_answers=tuple(answer.pin_memory() for answer in self.query_answers),
        )


@dataclass(frozen=True, slots=True)
class ProductionVisualAudit:
    chunk: CurrentChunkMaterialization
    token: CurrentChunkVisualTokenAudit
    current_chunk_only: bool = True

    def __post_init__(self) -> None:
        if not self.current_chunk_only or self.chunk.spec.history_chunk_ids:
            raise ValueError("production visual input must contain only the current chunk")


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
        prefetch_depth: int = 2,
        decode_coalesce: bool = True,
    ) -> None:
        if minimum_pixels <= 0 or maximum_pixels <= 0 or minimum_pixels > maximum_pixels:
            raise ValueError("video pixel bounds must satisfy 0 < minimum <= maximum")
        if type(prefetch_depth) is not int or prefetch_depth <= 0:
            raise ValueError("support prefetch_depth must be a positive integer")
        self.config = config
        self.minimum_pixels = minimum_pixels
        self.maximum_pixels = maximum_pixels
        self.processor = QwenVideoPreprocessor(config)
        self.preprocess_cache = preprocess_cache
        self.prefetch_depth = prefetch_depth
        self.decode_coalesce = decode_coalesce
        self._source_dataset = "runtime"
        self._executor: ThreadPoolExecutor | None = None
        self._pending_queue: deque[
            tuple[
                CurrentChunkSpec,
                Future[Any],
                Callable[[object], CurrentChunkMaterialization],
            ]
        ] = deque()
        self._remaining_specs: deque[CurrentChunkSpec] = deque()

    def __call__(self, spec: CurrentChunkSpec) -> CurrentChunkMaterialization:
        if self._pending_queue:
            expected, future, resolver = self._pending_queue[0]
            if expected != spec:
                raise RuntimeError("support chunks were consumed out of prefetch order")
            started = time.perf_counter()
            result = resolver(future.result())
            self._pending_queue.popleft()
            self._schedule_next()
            _loader_trace(
                "support_materialize_ready",
                chunk_id=spec.chunk_id,
                seconds=time.perf_counter() - started,
                cache_stats=_cache_stats(self),
            )
            return result
        return self._materialize(spec)

    def begin_prefetch(
        self, specs: Sequence[CurrentChunkSpec], *, source_dataset: str | None = None
    ) -> None:
        """Start a bounded sequence while preserving strict consumption order."""

        self.end_prefetch()
        if os.environ.get("TTT_A2_SUPPORT_PREFETCH", "1") == "0" or not specs:
            return
        if len({spec.chunk_id for spec in specs}) != len(specs):
            raise ValueError("support prefetch requires unique chunk IDs")
        if source_dataset is not None:
            if not source_dataset:
                raise ValueError("support prefetch source_dataset must be non-empty")
            self._source_dataset = source_dataset
        self._remaining_specs.extend(specs)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.prefetch_depth,
                thread_name_prefix="ttt-support-prefetch",
            )
        self._schedule_next()

    def set_source_dataset(self, source_dataset: str) -> None:
        if not source_dataset:
            raise ValueError("source_dataset must be non-empty")
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
        if self._executor is None:
            raise RuntimeError("support prefetch executor was not initialized")
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
                group_future = self._executor.submit(self._materialize_group, tuple(group))
                for spec in group:
                    chunk_id = spec.chunk_id

                    def resolve_group(
                        value: object, chunk_id: str = chunk_id
                    ) -> CurrentChunkMaterialization:
                        return cast(
                            dict[str, CurrentChunkMaterialization], value
                        )[chunk_id]

                    self._pending_queue.append(
                        (
                            spec,
                            group_future,
                            resolve_group,
                        )
                    )
            else:
                spec = self._remaining_specs.popleft()
                future = self._executor.submit(self._materialize, spec)
                self._pending_queue.append((spec, future, _identity_chunk))

    def _materialize(self, spec: CurrentChunkSpec) -> CurrentChunkMaterialization:
        started = time.perf_counter()
        fingerprint = self._fingerprint(spec)
        if self.preprocess_cache is not None:
            cached = self.preprocess_cache.get(fingerprint)
            if cached is not None:
                _loader_trace(
                    "support_cache_hit",
                    chunk_id=spec.chunk_id,
                    seconds=time.perf_counter() - started,
                )
                return self._from_cached(spec, cached)
        _loader_trace("support_cache_miss", chunk_id=spec.chunk_id)
        frames, timestamps = _decode_uniform_interval(
            spec, self.config.video_preprocessing.sample_fps
        )
        materialized = self._materialize_decoded(spec, frames, timestamps, fingerprint)
        _loader_trace(
            "support_materialize_done",
            chunk_id=spec.chunk_id,
            seconds=time.perf_counter() - started,
        )
        return materialized

    def _materialize_group(
        self, specs: tuple[CurrentChunkSpec, ...]
    ) -> dict[str, CurrentChunkMaterialization]:
        """Decode one same-video Support group with a single PyAV container."""

        if not specs:
            return {}
        group_started = time.perf_counter()
        if len({spec.video_path for spec in specs}) != 1:
            raise ValueError("coalesced Support group must contain one video path")
        results: dict[str, CurrentChunkMaterialization] = {}
        misses: list[tuple[CurrentChunkSpec, PreprocessFingerprint]] = []
        for spec in specs:
            fingerprint = self._fingerprint(spec)
            cached = self.preprocess_cache.get(fingerprint) if self.preprocess_cache else None
            if cached is not None:
                results[spec.chunk_id] = self._from_cached(spec, cached)
                _loader_trace("support_cache_hit", chunk_id=spec.chunk_id)
            else:
                misses.append((spec, fingerprint))
        if not misses:
            _loader_trace(
                "support_decode_coalesced",
                chunk_count=len(specs),
                cache_only=True,
                seconds=time.perf_counter() - group_started,
            )
            return results
        try:
            decoded = _decode_coalesced_intervals(
                tuple(spec for spec, _ in misses),
                self.config.video_preprocessing.sample_fps,
            )
        except Exception:
            # Coalescing is an optimization only.  Preserve the proven decoder for unusual VFR
            # or non-seekable containers instead of turning a cache miss into a training failure.
            decoded = {
                spec.chunk_id: _decode_uniform_interval(
                    spec, self.config.video_preprocessing.sample_fps
                )
                for spec, _ in misses
            }
        for spec, fingerprint in misses:
            frames, timestamps = decoded[spec.chunk_id]
            results[spec.chunk_id] = self._materialize_decoded(
                spec, frames, timestamps, fingerprint
            )
        _loader_trace(
            "support_decode_coalesced",
            chunk_count=len(specs),
            seconds=time.perf_counter() - group_started,
        )
        return results

    def _materialize_decoded(
        self,
        spec: CurrentChunkSpec,
        frames: Tensor,
        timestamps: Tensor,
        fingerprint: PreprocessFingerprint,
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
            sample_fps=self.config.video_preprocessing.sample_fps,
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
        if self.preprocess_cache is not None:
            self.preprocess_cache.put(fingerprint, _cached_from_materialized(materialized))
        return materialized

    def _fingerprint(self, spec: CurrentChunkSpec) -> PreprocessFingerprint:
        return _build_preprocess_fingerprint(
            spec,
            config=self.config,
            minimum_pixels=self.minimum_pixels,
            maximum_pixels=self.maximum_pixels,
            source_dataset=self._source_dataset,
        )

    @staticmethod
    def _from_cached(spec: CurrentChunkSpec, cached: CachedChunk) -> CurrentChunkMaterialization:
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
        if isinstance(raw, CurrentChunkSpec):
            chunk = self.materializer(raw)
        elif isinstance(raw, CurrentChunkMaterialization):
            chunk = raw
        else:
            raise TypeError("visual runtime accepts one CurrentChunkSpec/materialization only")
        device = _module_device(self.qwen)
        h2d_started = time.perf_counter()
        pixels = chunk.pixel_values_videos.to(device=device, non_blocking=True)
        grid = chunk.video_grid_thw.to(device=device, non_blocking=True)
        _loader_trace("pin_memory/H2D", seconds=time.perf_counter() - h2d_started)
        with trace_cuda_phase("vit_forward", stage="visual"):
            self.qwen.get_video_features(pixels, grid)
        captured = self.qwen.last_visual_output
        prepared = self.qwen.last_prepared_video_features
        if not isinstance(captured, QwenVisualOutput) or not isinstance(
            prepared, PreparedVideoFeatures
        ):
            raise RuntimeError("Qwen visual boundary did not return prepared current features")
        token_audit = audit_current_chunk_visual_tokens(prepared, pixels, grid)
        adapted_padded = pad_sequence(prepared.main_features, batch_first=True)
        adapted = QwenVisualOutput(
            main_visual_embeddings=adapted_padded,
            deepstack_features=prepared.deepstack_features,
            visual_valid_mask=captured.visual_valid_mask,
            metadata=prepared.metadata,
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
            prepared_video_features=prepared,
            audit=ProductionVisualAudit(materialized, token_audit),
        )

    def _prefill(self, request: QwenPrefillRequest) -> QwenPrefillOutput:
        if not isinstance(request.prepared_video_features, PreparedVideoFeatures):
            raise TypeError("production prefill requires one current PreparedVideoFeatures")
        if not isinstance(request.pixel_values_videos, Tensor) or not isinstance(
            request.video_grid_thw, Tensor
        ):
            raise TypeError("production prefill pixels/grid must be tensors")
        device = _module_device(self.qwen)
        h2d_started = time.perf_counter()
        pixels = request.pixel_values_videos.to(device=device, non_blocking=True)
        grid = request.video_grid_thw.to(device=device, non_blocking=True)
        _loader_trace("pin_memory/H2D", seconds=time.perf_counter() - h2d_started)
        audit_current_chunk_visual_tokens(request.prepared_video_features, pixels, grid)
        if not isinstance(request.input_ids, Tensor) or not isinstance(
            request.attention_mask, Tensor
        ):
            raise TypeError("production prefill IDs/mask must be tensors")
        input_ids = request.input_ids.to(device=device)
        attention_mask = request.attention_mask.to(device=device)
        state_payload = _state_embedding_payload(request, input_ids)
        kwargs = dict(request.qwen_kwargs)
        kwargs.setdefault("use_cache", False)
        with trace_cuda_phase("llm_prefill", stage="prefill"):
            output = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixels,
                video_grid_thw=grid,
                prepared_video_features=request.prepared_video_features,
                state_embedding_payload=state_payload,
                **kwargs,
            )
        return cast(QwenPrefillOutput, output)

    def _generate(self, request: QwenGenerateRequest) -> QwenGenerateOutput:
        prefill = request.prefill
        if not isinstance(prefill.prepared_video_features, PreparedVideoFeatures):
            raise TypeError("production generation requires prepared video features")
        if not isinstance(prefill.pixel_values_videos, Tensor) or not isinstance(
            prefill.video_grid_thw, Tensor
        ):
            raise TypeError("production generation pixels/grid must be tensors")
        device = _module_device(self.qwen)
        input_ids = prefill.input_ids.to(device=device)
        attention_mask = prefill.attention_mask.to(device=device)
        pixels = prefill.pixel_values_videos.to(device=device, non_blocking=True)
        grid = prefill.video_grid_thw.to(device=device, non_blocking=True)
        audit_current_chunk_visual_tokens(prefill.prepared_video_features, pixels, grid)
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
            prepared_video_features=prefill.prepared_video_features,
            state_embedding_payload=_state_embedding_payload(prefill, input_ids),
            **kwargs,
        )
        sequences = getattr(generated, "sequences", generated)
        if not isinstance(sequences, Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
            raise TypeError("Qwen generate() must return one [1, T] token tensor")
        new_tokens = sequences[:, input_ids.shape[1] :]
        decode = getattr(self.tokenizer, "batch_decode", None)
        if not callable(decode):
            raise TypeError("production tokenizer must provide batch_decode()")
        texts = decode(new_tokens, skip_special_tokens=True)
        if not isinstance(texts, Sequence) or len(texts) != 1 or not isinstance(texts[0], str):
            raise TypeError("tokenizer batch_decode() must return one string")
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
        if qwen is None:
            raise RuntimeError("Qwen embedding owner was released")
        embeddings = embed_question_tokens(cast(Any, qwen), tokens, self.query_encoder_config)
        inputs = QueryEncoderInput.from_runtime_queries(embeddings, tokens, (value,))
        seed_bytes = hashlib.sha256(
            f"{value.episode_nonce}:{value.query_id}".encode()
        ).digest()[:8]
        dropout_seed = int.from_bytes(seed_bytes, "little") % (2**63 - 1)
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
            output = self.query_encoder(inputs, inference=inference)
        if not isinstance(output, QueryEncoderOutput):
            raise TypeError("production Query encoder returned an invalid output")
        return output

    @property
    def query_encoder_config(self) -> ProjectConfig:
        config = getattr(self, "_project_config", None)
        if not isinstance(config, ProjectConfig):
            raise RuntimeError("production Query runtime lost its ProjectConfig")
        return config

    def bind_project_config(self, config: ProjectConfig) -> None:
        object.__setattr__(self, "_project_config", config)


class FastVisualPassThrough:
    def __call__(
        self,
        visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        del query, request
        if not isinstance(visual.audit, ProductionVisualAudit):
            raise TypeError("Fast pass-through requires an audited current visual chunk")
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
        if not isinstance(output, SpatialEncoderOutput):
            raise TypeError("production Spatial encoder returned an invalid output")
        return output


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
        if not isinstance(output, TemporalEncoderOutput):
            raise TypeError("production Temporal encoder returned an invalid output")
        return output


def _causal_temporal_tail(
    value: QwenVisualOutput,
    chunk: CurrentChunkMaterialization,
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

    if len(value.metadata.token_counts) != 1 or chunk.tubelet_timestamps.shape[0] != 1:
        raise ValueError("production temporal tail currently requires one episode per rank")
    times = chunk.tubelet_timestamps
    tubelet_count = times.shape[1]
    drop = 0
    next_position = 0
    if cache is not None:
        if cache.batch_size != 1:
            raise ValueError("production temporal cache must contain one episode per rank")
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
        raw_chunk = request.video_input
        if isinstance(raw_chunk, CurrentChunkSpec):
            reset = raw_chunk.reset_soft_state
        elif isinstance(raw_chunk, CurrentChunkMaterialization):
            reset = raw_chunk.spec.reset_soft_state
        else:
            raise TypeError("production Observation runtime requires one current chunk")
        empty = (None,) * len(request.owner.video_ids)
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
        if not isinstance(output, ObservationOutputs):
            raise TypeError("production Observation heads returned an invalid output")
        return output


class ProductionReaderRuntime:
    """Expose the concrete Reader through the shared typed model boundary."""

    def __init__(self, reader: DeterministicStateReader) -> None:
        self.reader = reader

    def read(self, retrieval: RetrieverOutput) -> Sequence[ReaderResult]:
        return self.reader.read(retrieval)

    def audit_results(
        self,
        retrieval: RetrieverOutput,
        results: Sequence[ReaderResult],
    ) -> Sequence[ReaderResult]:
        return self.reader.audit_results(retrieval, results)

    def audit_number_tokens(self, result: ReaderResult) -> int | None:
        return self.reader.audit_number_tokens(result)


def _compose_production_inputs(
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
) -> ComposedInput:
    return compose_inputs(
        base_input_ids=base_input_ids,
        base_attention_mask=base_attention_mask,
        state_tokens=state_tokens,
        state_token_valid_mask=state_token_valid_mask,
        reader_results=reader_results,
        tokenizer=cast(Any, tokenizer),
        embedding_owner=cast(Any, embedding_owner),
        rope_indexer=cast(Any, rope_indexer),
        video_grid_thw=video_grid_thw,
        include_state=include_state,
        include_number=include_number,
    )


class ProductionOuterModel(nn.Module):  # type: ignore[misc]
    """Checkpoint/optimizer owner; numerical entrypoints live in the injected runners."""

    def __init__(
        self,
        state_model: StateTTTModel,
        predictor: TemporalPredictor,
        qwen_model: nn.Module,
    ) -> None:
        super().__init__()
        self.state_model = state_model
        self.predictor = predictor
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
        enable = getattr(self._qwen_model(), "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TypeError("loaded Qwen model does not support gradient checkpointing")
        # ``CustomSeq2SeqTrainer`` calls this owner again after LLaMA-Factory has
        # already configured Qwen.  Passing ``None`` would silently restore the
        # LLaMA-Factory wrapper's default re-entrant checkpointing, which lets
        # ZeRO-2 see a Decoder partition twice during checkpoint recomputation.
        # Production A2/A5 requires non-reentrant checkpointing end-to-end.
        kwargs = dict(gradient_checkpointing_kwargs or {})
        kwargs["use_reentrant"] = False
        enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self) -> None:
        disable = getattr(self._qwen_model(), "gradient_checkpointing_disable", None)
        if not callable(disable):
            raise TypeError("loaded Qwen model does not support gradient checkpointing")
        disable()

    def _qwen_model(self) -> nn.Module:
        model = self._qwen_model_ref()
        if model is None:
            raise RuntimeError("loaded Qwen model was released before the outer runtime")
        return model

    def forward(self, **_inputs: object) -> Tensor:
        raise RuntimeError("ProductionOuterModel must be driven by the typed A2/A5 trainer hook")


class A2PrefetchCollator:
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
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None = None,
    ) -> None:
        _require_latest_qwen_processor(processor, context="A2 prefetch")
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.minimum_pixels = minimum_pixels
        self.maximum_pixels = maximum_pixels
        self.preprocess_cache = preprocess_cache

    def __call__(self, records: Sequence[object]) -> dict[str, object]:
        if len(records) != 1 or not isinstance(records[0], A2QueryRecord):
            raise ValueError("A2 H200 prefetch requires exactly one A2 record per rank")
        started = time.perf_counter()
        record = records[0]
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        spec = _query_chunk_spec(
            f"{record.query.runtime.query_id}:query",
            video_path,
            record.query.runtime.query_time,
            reset_soft_state=False,
        )
        answer = _prepare_answer_cpu(
            record.query,
            spec,
            processor=self.processor,
            tokenizer=self.tokenizer,
            config=self.config,
            minimum_pixels=self.minimum_pixels,
            maximum_pixels=self.maximum_pixels,
            preprocess_cache=self.preprocess_cache,
            source_dataset=record.source_dataset,
        )
        _loader_trace(
            "a2_collate_done",
            query_id=record.query.runtime.query_id,
            seconds=time.perf_counter() - started,
            cache_stats=(self.preprocess_cache.stats() if self.preprocess_cache else {}),
        )
        return {"prepared_a2": PreparedA2Record(record, answer)}


class A5PrefetchCollator:
    """Prepare only CPU Query tensors; runtime/State-TTT objects stay in the trainer process."""

    def __init__(
        self,
        *,
        processor: object,
        tokenizer: object,
        config: ProjectConfig,
        minimum_pixels: int,
        maximum_pixels: int,
        preprocess_cache: PreprocessCache | None = None,
    ) -> None:
        _require_latest_qwen_processor(processor, context="A5 prefetch")
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.minimum_pixels = minimum_pixels
        self.maximum_pixels = maximum_pixels
        self.preprocess_cache = preprocess_cache

    def __call__(self, records: Sequence[object]) -> dict[str, object]:
        if len(records) != 1 or not isinstance(records[0], A5EpisodeRecord):
            raise ValueError("A5 prefetch requires exactly one A5 episode per rank")
        started = time.perf_counter()
        record = records[0]
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        answers: list[PreparedAnswerCPU] = []
        primary = record.queries[0].runtime
        for index, query in enumerate(record.queries):
            spec = _query_chunk_spec(
                f"{record.episode_id}:q{index}",
                video_path,
                query.runtime.query_time,
                reset_soft_state=index > 0 and query.runtime.question != primary.question,
            )
            answers.append(
                _prepare_answer_cpu(
                    query,
                    spec,
                    processor=self.processor,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    minimum_pixels=self.minimum_pixels,
                    maximum_pixels=self.maximum_pixels,
                    preprocess_cache=self.preprocess_cache,
                    source_dataset=record.source_dataset,
                )
            )
        _loader_trace(
            "a5_collate_done",
            episode_id=record.episode_id,
            query_count=len(answers),
            seconds=time.perf_counter() - started,
            cache_stats=(self.preprocess_cache.stats() if self.preprocess_cache else {}),
        )
        return {"prepared_a5": PreparedA5Record(record, tuple(answers))}


class ProductionEpisodeMaterializer:
    def __init__(
        self,
        backbone: LlamaFactoryBackboneBundle,
        writer: StageABankWriter,
        video: VideoChunkMaterializer,
    ) -> None:
        self.backbone = backbone
        self.config = backbone.project_config
        self.writer = writer
        self.video = video
        self.tokenizer = backbone.tokenizer
        self.processor = backbone.processor
        self._episode_nonce = 0
        _require_latest_qwen_processor(self.processor, context="production Qwen training")

    def a2(self, source: PreparedA2Record) -> StageATrainingBatch:
        record = source.record
        self.video.set_source_dataset(record.source_dataset)
        episode_nonce = self._allocate_episode_nonce()
        video_path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        owner = RuntimeOwner((record.video_id,), (record.trajectory_id,))
        runtime = self.writer.reset(owner)
        _, supports = adaptive_support_schedule(record.query.runtime.query_time)
        query_chunk = _query_chunk_spec(
            f"{record.query.runtime.query_id}:query",
            video_path,
            record.query.runtime.query_time,
            reset_soft_state=False,
        )
        prepared_answer = source.answer
        if prepared_answer.spec != query_chunk:
            raise ValueError("prefetched A2 Query chunk drifted before runtime assembly")
        answer, labels, materialized_query = self._bind_answer(prepared_answer)
        requests = tuple(
            self._request(
                owner,
                runtime,
                video_path,
                record.query.runtime,
                chunk,
                index,
                episode_nonce,
            )
            for index, chunk in enumerate(supports)
        ) + (
            ObservationChunkRequest(
                owner=owner,
                video_input=materialized_query,
                query_input=_bind_runtime_query(record.query.runtime, video_path, episode_nonce),
                runtime_state=runtime,
                bank_states=runtime.state_bank_states,
                inference=False,
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
        video_path = self._video_path(record.source_dataset, record.relative_video_path)
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
        supports = tuple(
            self._meta_chunk(
                owner,
                runtime,
                video_path,
                primary,
                chunk,
                f"s{index + 1}",
                episode_nonce,
            )
            for index, chunk in enumerate(record.supports)
        )
        queries: list[MetaTTTQueryPoint] = []
        for index, query in enumerate(record.queries):
            spec = _query_chunk_spec(
                f"{record.episode_id}:q{index}",
                video_path,
                query.runtime.query_time,
                reset_soft_state=index > 0 and query.runtime.question != primary.question,
            )
            prepared_answer = prepared.query_answers[index]
            if prepared_answer.spec != spec:
                raise ValueError("prefetched A5 Query chunk drifted before runtime assembly")
            answer, labels, materialized = self._bind_answer(prepared_answer)
            request = ObservationChunkRequest(
                owner=owner,
                video_input=materialized,
                query_input=_bind_runtime_query(query.runtime, video_path, episode_nonce),
                runtime_state=runtime,
                bank_states=runtime.state_bank_states,
                inference=False,
            )
            queries.append(
                MetaTTTQueryPoint(
                    chunk=MetaCausalChunk(
                        request,
                        spec.start_time,
                        spec.end_time,
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
            seed=self.config.stage_c.seed,
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
        spec = CurrentChunkSpec(
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
            video_input=CurrentChunkSpec(
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
        )

    def _allocate_episode_nonce(self) -> int:
        nonce = self._episode_nonce
        self._episode_nonce += 1
        return nonce

    def _bind_answer(
        self,
        prepared: PreparedAnswerCPU,
    ) -> tuple[StageAEpisodeAnswerInputs, AnswerTargetLabels, CurrentChunkMaterialization]:
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

    @staticmethod
    def _video_path(source_dataset: str, relative_path: str) -> Path:
        return _resolve_video_path(source_dataset, relative_path)


class _A2ProgressTrace:
    """Rank-local, opt-in lifecycle trace for distributed A2 deadlock diagnosis."""

    def __init__(self, path: Path | None, *, rank: int) -> None:
        self.path = path
        self.rank = rank
        self.call_index = 0

    @classmethod
    def from_environment(cls) -> _A2ProgressTrace:
        rank = int(os.environ.get("RANK", "0"))
        if os.environ.get("TTT_A2_PROGRESS_TRACE") != "1":
            return cls(None, rank=rank)
        run_root = os.environ.get("RUN_ROOT")
        if not run_root:
            raise ValueError("TTT_A2_PROGRESS_TRACE=1 requires RUN_ROOT")
        directory = Path(run_root) / "samples" / f"rank_{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory / "a2_progress.jsonl", rank=rank)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def begin(self, record: A2QueryRecord) -> int:
        self.call_index += 1
        self.emit(
            self.call_index,
            "forward_begin",
            query_id=record.query.runtime.query_id,
            task_class=record.task_class,
            query_time=record.query.runtime.query_time,
            video_id=record.video_id,
        )
        return self.call_index

    def emit(self, call_index: int, event: str, **fields: object) -> None:
        if self.path is None:
            return
        payload = {
            "monotonic_seconds": time.monotonic(),
            "rank": self.rank,
            "pid": os.getpid(),
            "call_index": call_index,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class ProductionA2LossStep:
    def __init__(
        self,
        runner: StageAEpisodeRunner,
        materializer: ProductionEpisodeMaterializer,
        graph_anchor_parameters: Sequence[nn.Parameter],
        config: ProjectConfig,
    ):
        self.runner = runner
        self.materializer = materializer
        self.weak_builder = OfficialWeakTargetBuilder()
        self.progress = _A2ProgressTrace.from_environment()
        self._active_progress_call: int | None = None
        self.graph_anchor_parameters = tuple(
            parameter for parameter in graph_anchor_parameters if parameter.requires_grad
        )
        self.outer_composer = OfficialWeakOuterLossComposer(config.loss.official_weak_balance)
        self.last_balance_audit: OfficialWeakBalanceAudit | None = None

    def __call__(self, _model: nn.Module, inputs: Mapping[str, object]) -> Tensor:
        prefetched = inputs.get("prepared_a2")
        if not isinstance(prefetched, PreparedA2Record):
            raise TypeError("A2 Trainer batch must contain PreparedA2Record")
        _loader_trace(
            "dataloader_wait",
            prepared=True,
        )
        source = prefetched
        record = prefetched.record
        call_index = self.progress.begin(record)
        self._active_progress_call = call_index
        batch = self.materializer.a2(source)
        if not isinstance(batch.model_inputs, StageAEpisodeInputs):
            raise TypeError("A2 materializer must return StageAEpisodeInputs")
        model_inputs = batch.model_inputs
        self.progress.emit(
            call_index,
            "materialized",
            support_count=len(model_inputs.observation_requests) - 1,
            dataloader_prefetched=True,
        )
        support_specs = tuple(
            request.video_input
            for request in model_inputs.observation_requests
            if isinstance(request.video_input, CurrentChunkSpec)
        )
        self.materializer.video.begin_prefetch(
            support_specs, source_dataset=record.source_dataset
        )
        try:
            raw = self.runner(batch, training=True)
        finally:
            self.materializer.video.end_prefetch()
        self.progress.emit(call_index, "forward_complete")
        raw.audit.validate_for(StageAVariant.A2)
        if (
            not isinstance(raw.composed_input, ComposedInput)
            or raw.observations is None
            or raw.query is None
            or raw.retrieval is None
        ):
            raise TypeError("A2 production forward did not return typed State/Qwen outputs")
        mapped = map_teacher_forced_targets(
            composed_input=raw.composed_input,
            source_input_ids=raw.source_input_ids,
            source_attention_mask=raw.source_attention_mask,
            source_labels=batch.supervision.answer.base_labels,
            source_number_token_mask=batch.supervision.answer.base_number_token_mask,
        )
        device = raw.answer_logits.device
        count_valid = raw.reader_count_valid_mask
        answer = compute_answer_loss(
            AnswerLossInput(
                logits=raw.answer_logits,
                labels=mapped.labels,
                number_token_mask=mapped.number_token_mask,
                reader_counts=ReaderCountMetricInput(
                    predicted_counts=raw.reader_counts,
                    target_counts=batch.supervision.answer.target_counts.to(device),
                    valid_mask=count_valid,
                ),
            )
        )
        weak = self.weak_builder(
            raw.observations,
            raw.query,
            raw.retrieval,
            batch.supervision.official_weak,
        )
        balanced = self.outer_composer.compose((answer,), (weak,))
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
        if total.ndim != 0 or not total.requires_grad or not bool(torch.isfinite(total).item()):
            raise ValueError("A2 production loss must be one finite differentiable scalar")
        self.progress.emit(call_index, "loss_ready")

        def trace_backward_start(gradient: Tensor) -> Tensor:
            self.progress.emit(call_index, "backward_begin")
            return gradient

        if self.progress.enabled:
            total.register_hook(trace_backward_start)
        return total

    def mark_backward_returned(self) -> None:
        call_index = self._active_progress_call
        if call_index is None:
            raise RuntimeError("A2 backward returned without an active loss call")
        self.progress.emit(call_index, "backward_return")
        self._active_progress_call = None


class ProductionA5EpisodeAdapter:
    def __init__(self, materializer: ProductionEpisodeMaterializer) -> None:
        self.materializer = materializer

    def __call__(self, inputs: Mapping[str, object]) -> tuple[MetaTTTEpisode, float]:
        prepared = inputs.get("prepared_a5")
        if not isinstance(prepared, PreparedA5Record):
            raise TypeError("A5 Trainer batch must contain PreparedA5Record")
        _loader_trace("dataloader_wait", prepared=True, stage="a5")
        episode = self.materializer.a5(prepared)
        self._begin_prefetch(episode)
        return episode, prepared.record.loss_weight

    def _begin_prefetch(self, episode: MetaTTTEpisode) -> None:
        video = self.materializer.video
        specs = tuple(
            cast(CurrentChunkSpec, chunk.request.video_input)
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
    if not config.preprocess_cache_enabled:
        return None
    env_name = config.preprocess_cache_root_env
    root = os.environ.get(env_name)
    if not root:
        # Cache is optional; the direct production decode path remains available when it is unset.
        _loader_trace("cache_disabled", reason=f"missing_env:{env_name}")
        return None
    max_gb = config.preprocess_cache_max_gb
    model_id = str(getattr(backbone.model_args, "model_name_or_path", "unknown-model"))
    revision = str(getattr(backbone.model_args, "revision", "unknown-revision"))
    processor_name = (
        type(backbone.processor).__qualname__ if backbone.processor is not None else "none"
    )
    namespace_seed = "|".join(
        (
            model_id,
            revision,
            transformers.__version__,
            processor_name,
            str(backbone.project_config.video_preprocessing.processor_shortest_edge),
            str(backbone.project_config.video_preprocessing.processor_longest_edge),
        )
    )
    namespace = hashlib.sha256(namespace_seed.encode("utf-8")).hexdigest()[:20]
    return PreprocessCache(
        root,
        max_bytes=int(max_gb * 1024**3),
        memory_entries=2,
        enabled=True,
        namespace=namespace,
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
    support_decode_coalesce = config.support_decode_coalesce
    fast = build_fast_ttt_adapter(project)
    qwen = Qwen3VLAdapter(
        backbone.model,
        project,
        adapter=fast,
        adapter_enabled=True,
        freeze_base=False,
    )
    chunk_materializer = VideoChunkMaterializer(
        project,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        preprocess_cache=preprocess_cache,
        prefetch_depth=support_prefetch_depth,
        decode_coalesce=support_decode_coalesce,
    )
    qwen_runtime = ProductionQwenRuntime(qwen, chunk_materializer, backbone.tokenizer)
    query_runtime = ProductionQueryRuntime(
        build_query_encoder(project), backbone.tokenizer, backbone.model
    )
    query_runtime.bind_project_config(project)
    spatial = ProductionSpatialRuntime(build_spatial_encoder(project))
    temporal = ProductionTemporalRuntime(build_temporal_encoder(project))
    observations = ProductionObservationRuntime(build_observation_heads(project))
    state_bank: StructuredStateBank = build_state_bank(project)
    identity_bank = build_identity_bank(project)
    writer = StageABankWriter(state_bank, identity_bank)
    reader = ProductionReaderRuntime(build_state_reader(project, cast(Any, backbone.tokenizer)))
    register_input_composer_tokens_with_audit(cast(Any, backbone.tokenizer), backbone.model)
    state_model = StateTTTModel(
        project,
        ModelComponents(
            visual_stage=qwen_runtime,
            query_encoder=query_runtime,
            composer=_compose_production_inputs,
            qwen_prefill=qwen_runtime,
            qwen_generate=qwen_runtime,
            fast_adapter=FastVisualPassThrough(),
            spatial_encoder=spatial,
            temporal_encoder=temporal,
            observation_heads=observations,
            state_bank=state_bank,
            bank_writer=writer,
            retriever=build_state_retriever(project),
            reader=reader,
            resampler=build_state_resampler(project),
        ),
        ModelFeatureFlags(),
    )
    predictor = build_temporal_predictor(project.predictor)
    outer = ProductionOuterModel(state_model, predictor, backbone.model)
    materializer = ProductionEpisodeMaterializer(backbone, writer, chunk_materializer)
    if stage is ProductionStage.A2:
        collator: object = A2PrefetchCollator(
            processor=backbone.processor,
            tokenizer=backbone.tokenizer,
            config=project,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            preprocess_cache=preprocess_cache,
        )
        predictor.requires_grad_(False)
        world_size = int(getattr(backbone.training_args, "world_size", 1))
        graph_anchor_parameters = (
            tuple(parameter for parameter in outer.parameters() if parameter.requires_grad)
            if world_size > 1
            else ()
        )
        runner = StageAEpisodeRunner(
            model=state_model,
            variant=StageAVariant.A2,
            metric_builder=lambda _output, _supervision: ((), ()),
        )
        return ProductionTrainerRuntime(
            stage=stage,
            model=outer,
            train_dataset=(),
            eval_dataset=None,
            data_collator=cast(Any, collator),
            stage_a_loss_step=cast(
                Any,
                ProductionA2LossStep(runner, materializer, graph_anchor_parameters, project),
            ),
        )
    collator = A5PrefetchCollator(
        processor=backbone.processor,
        tokenizer=backbone.tokenizer,
        config=project,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        preprocess_cache=preprocess_cache,
    )
    predictor.requires_grad_(True)
    meta_runner = MetaTTTEpisodeRunner(
        config=project,
        model=state_model,
        fast_controller=fast,
        predictor=predictor,
        runtime_resetter=lambda owner: _reset_meta_runtime(writer, owner),
        variant=MetaTTTVariant.A5,
    )
    return ProductionTrainerRuntime(
        stage=stage,
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
) -> StateTTTRuntimeBundle:
    """Load local Qwen assets and assemble the sole online State-TTT runtime."""

    from ttt_svcbench_qwen.inference import OnlineTTTUpdater, PerVideoRuntimeManager

    root = Path(model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen model root does not exist: {root}")
    config = load_config(config_path)
    processor = transformers.AutoProcessor.from_pretrained(root, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    model_type = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_type is None:
        raise RuntimeError("installed transformers has no Qwen3VLForConditionalGeneration")
    qwen_model = model_type.from_pretrained(
        root,
        dtype=dtype,
        local_files_only=True,
    )
    if not isinstance(qwen_model, nn.Module):
        raise TypeError("Qwen loader returned a non-module")
    qwen_model.to(device=torch.device(device))
    qwen_model.eval()
    fast = build_fast_ttt_adapter(config)
    qwen = Qwen3VLAdapter(
        qwen_model,
        config,
        adapter=fast,
        adapter_enabled=True,
        freeze_base=False,
    )
    materializer = VideoChunkMaterializer(
        config,
        minimum_pixels=16 * 16,
        maximum_pixels=262_144,
    )
    qwen_runtime = ProductionQwenRuntime(qwen, materializer, tokenizer)
    query_runtime = ProductionQueryRuntime(build_query_encoder(config), tokenizer, qwen_model)
    query_runtime.bind_project_config(config)
    state_bank = build_state_bank(config)
    identity_bank = build_identity_bank(config)
    writer = StageABankWriter(state_bank, identity_bank)
    register_input_composer_tokens_with_audit(cast(Any, tokenizer), qwen_model)
    state_model = StateTTTModel(
        config,
        ModelComponents(
            visual_stage=qwen_runtime,
            query_encoder=query_runtime,
            composer=_compose_production_inputs,
            qwen_prefill=qwen_runtime,
            qwen_generate=qwen_runtime,
            fast_adapter=FastVisualPassThrough(),
            spatial_encoder=ProductionSpatialRuntime(build_spatial_encoder(config)),
            temporal_encoder=ProductionTemporalRuntime(build_temporal_encoder(config)),
            observation_heads=ProductionObservationRuntime(build_observation_heads(config)),
            state_bank=state_bank,
            bank_writer=writer,
            retriever=build_state_retriever(config),
            reader=ProductionReaderRuntime(
                build_state_reader(config, cast(Any, tokenizer))
            ),
            resampler=build_state_resampler(config),
        ),
        ModelFeatureFlags(),
    )
    predictor = build_temporal_predictor(config.predictor)
    outer = ProductionOuterModel(state_model, predictor, qwen_model)
    load_outer_checkpoint(outer, checkpoint)
    outer.eval()
    manager = PerVideoRuntimeManager(
        fast_adapter=fast,
        state_bank=state_bank,
        identity_bank=identity_bank,
        optimizer_config=config.fast_ttt.optimizer,
        audit_level=config.inference.audit_level,
    )
    return StateTTTRuntimeBundle(
        config=config,
        qwen_adapter=qwen,
        state_model=state_model,
        outer_model=outer,
        manager=manager,
        updater=OnlineTTTUpdater(config, predictor),
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
) -> tuple[QwenVisualOutput, CurrentChunkMaterialization, QueryEncoderOutput, BatchRuntimeState]:
    if (
        not isinstance(visual.value, QwenVisualOutput)
        or not isinstance(visual.audit, ProductionVisualAudit)
    ):
        raise TypeError("production state stages require a typed current visual output")
    return visual.value, visual.audit.chunk, query, _stage_runtime(request)


def _stage_runtime(request: ObservationChunkRequest) -> BatchRuntimeState:
    return request.runtime_state


def _state_embedding_payload(
    request: QwenPrefillRequest,
    input_ids: Tensor,
) -> StateEmbeddingPayload | None:
    if request.state_position_mask is None or request.state_tokens is None:
        return None
    if not isinstance(request.state_position_mask, Tensor) or not isinstance(
        request.state_tokens, Tensor
    ):
        raise TypeError("production State mask/tokens must be tensors")
    mask = request.state_position_mask.to(device=input_ids.device, dtype=torch.bool)
    tokens = request.state_tokens.to(device=input_ids.device)
    if tokens.ndim != 3 or mask.ndim != 2 or tokens.shape[0] != mask.shape[0]:
        raise ValueError("production State tokens/mask batch shapes do not align")
    rows: list[Tensor] = []
    for row in range(mask.shape[0]):
        count = int(mask[row].sum().item())
        if count not in (0, tokens.shape[1]):
            raise ValueError("each valid State row must consume all 16 current State tokens")
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


def _query_chunk_spec(
    chunk_id: str,
    video_path: Path,
    query_time: float,
    *,
    reset_soft_state: bool,
) -> CurrentChunkSpec:
    end = query_time
    start = max(0.0, end - 8.0)
    if end <= start:
        raise ValueError("Query point is too early to materialize a current video chunk")
    return CurrentChunkSpec(
        chunk_id=chunk_id,
        video_path=video_path,
        start_time=start,
        end_time=end,
        maximum_frames=16,
        query_time=query_time,
        reset_soft_state=reset_soft_state,
    )


def _prepare_answer_cpu(
    query: ProductionQueryRecord,
    spec: CurrentChunkSpec,
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

    started = time.perf_counter()
    _loader_trace("query_prepare", query_id=query.runtime.query_id)
    answer_text = query.answer.answer if query.answer.answer is not None else str(query.weak.count)
    typed_processor = _require_latest_qwen_processor(processor, context="Query preprocessing")
    fingerprint = _build_preprocess_fingerprint(
        spec,
        config=config,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        source_dataset=source_dataset,
    )
    cached = preprocess_cache.get(fingerprint) if preprocess_cache is not None else None
    if cached is not None:
        materialized = _materialized_from_cached(spec, cached)
        frames = materialized.frames
        _loader_trace("query_cache_hit", query_id=query.runtime.query_id)
    else:
        _loader_trace("query_cache_miss", query_id=query.runtime.query_id)
        frames, timestamps = _decode_uniform_interval(spec, config.video_preprocessing.sample_fps)
        frames = _resize_to_pixel_budget(
            frames,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
        )
        pixels, grid = _process_video_once(typed_processor, frames)
        materialized = _build_materialized_query(spec, frames, timestamps, pixels, grid, config)
        if preprocess_cache is not None:
            preprocess_cache.put(fingerprint, _cached_from_materialized(materialized))

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
    prompt_grid = materialized.video_grid_thw
    if not torch.equal(materialized.video_grid_thw, prompt_grid):
        raise ValueError("prompt/full processor calls produced different current video grids")
    full_ids = full_ids.to(torch.int64)
    prompt_ids = prompt_ids.to(torch.int64)
    prompt_length = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= prompt_length or not torch.equal(
        full_ids[:, :prompt_length], prompt_ids
    ):
        raise ValueError("Qwen chat template full sequence does not preserve its prompt prefix")
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
    _loader_trace(
        "query_prepare_done",
        query_id=query.runtime.query_id,
        seconds=time.perf_counter() - started,
        cache_stats=(preprocess_cache.stats() if preprocess_cache else {}),
    )
    return PreparedAnswerCPU(
        spec=spec,
        base_input_ids=full_ids,
        base_attention_mask=full_mask,
        target_labels=target_labels,
        materialized_query=materialized,
    )


def _build_preprocess_fingerprint(
    spec: CurrentChunkSpec,
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
        sample_fps=config.video_preprocessing.sample_fps,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        patch_size=config.video_preprocessing.patch_size,
        temporal_patch_size=config.video_preprocessing.temporal_patch_size,
        spatial_merge_size=config.video_preprocessing.spatial_merge_size,
        transformers_version=transformers.__version__,
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


def _prepared_answer_bytes(answer: PreparedAnswerCPU) -> int:
    tensors = (
        answer.base_input_ids,
        answer.base_attention_mask,
        answer.target_labels.base_labels,
        answer.target_labels.base_number_token_mask,
        answer.target_labels.target_counts,
        answer.materialized_query.frame_timestamps,
        answer.materialized_query.tubelet_timestamps,
        answer.materialized_query.tubelet_valid_mask,
        answer.materialized_query.tubelet_position_ids,
        answer.materialized_query.pixel_values_videos,
        answer.materialized_query.video_grid_thw,
    )
    return sum(int(value.numel() * value.element_size()) for value in tensors)


def _materialized_from_cached(
    spec: CurrentChunkSpec, cached: CachedChunk
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
    spec: CurrentChunkSpec,
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
            tubelet_times, sample_fps=config.video_preprocessing.sample_fps
        ).unsqueeze(0),
        pixel_values_videos=pixels,
        video_grid_thw=grid.to(torch.int64),
    )


def _require_latest_qwen_processor(processor: object, *, context: str) -> Any:
    typed = cast(Any, processor)
    if not callable(getattr(typed, "apply_chat_template", None)):
        raise TypeError(f"{context} requires Qwen3-VL apply_chat_template")
    if not callable(getattr(typed, "video_processor", None)):
        raise TypeError(f"{context} requires the Qwen3-VL video_processor")
    if not callable(getattr(typed, "tokenizer", None)):
        raise TypeError(f"{context} requires the Qwen3-VL tokenizer")
    return typed


def _process_video_once(processor: Any, frames: Tensor) -> tuple[Tensor, Tensor]:
    video_processor = processor.video_processor
    started = time.perf_counter()
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
    if pixels.ndim != 2 or not torch.is_floating_point(pixels):
        raise ValueError("direct Qwen video processor returned invalid pixel tensor")
    _loader_trace(
        "processor",
        seconds=time.perf_counter() - started,
        pixel_tokens=int(pixels.shape[0]),
    )
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
    if not timestamps:
        raise ValueError("Qwen video placeholder expansion received no frame timestamps")
    timestamps = (timestamps + [timestamps[-1]] * temporal)[:temporal]
    placeholder = "".join(
        f"<{float(t):.1f} seconds>{vision_start}"
        + ("<|placeholder|>" * frame_seqlen)
        + vision_end
        for t in timestamps
    )
    wrapped = f"{vision_start}{video_token}{vision_end}"
    if wrapped not in text:
        raise ValueError("Qwen chat template did not emit the expected video placeholder")
    text = text.replace(wrapped, placeholder, 1)
    return text.replace("<|placeholder|>", video_token)


def _resolve_video_path(source_dataset: str, relative_path: str) -> Path:
    root = os.environ.get("SVCBENCH_VIDEO_ROOT")
    if not root:
        raise ValueError("SVCBENCH_VIDEO_ROOT is required for production materialization")
    root_path = Path(root).resolve()
    direct = (root_path / relative_path).resolve()
    nested = (root_path / source_dataset / relative_path).resolve()
    for candidate in (direct, nested):
        if not candidate.is_relative_to(root_path):
            raise ValueError("production video path escaped SVCBENCH_VIDEO_ROOT")
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
    if not isinstance(raw, Mapping) or not isinstance(raw.get(key), Tensor):
        raise TypeError(f"Qwen processor output is missing Tensor {key}")
    return cast(Tensor, raw[key])


def _token_ids(tokenizer: object, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("production tokenizer must expose encode()")
    values = encode(text, add_special_tokens=False)
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
    if minimum > maximum:
        raise ValueError("LLaMA-Factory video_min_pixels cannot exceed video_max_pixels")
    return minimum, maximum


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("production module exposes no parameters/device")
    return parameter.device


def _decode_uniform_interval(
    spec: CurrentChunkSpec,
    sample_fps: float,
) -> tuple[Tensor, Tensor]:
    desired = min(
        spec.maximum_frames,
        max(2, int(math.floor((spec.end_time - spec.start_time) * sample_fps)) + 1),
    )
    desired -= desired % 2
    if desired < 2:
        raise ValueError(f"current chunk {spec.chunk_id} has no complete temporal tubelet")
    target_times = torch.linspace(
        spec.start_time,
        spec.end_time,
        desired,
        dtype=torch.float64,
    ).tolist()
    # Re-seeking for every target is efficient for a 2,048-second geometric interval but very
    # expensive for the overlapping 8-second recent windows (up to 16 keyframe seeks instead of
    # one short scan).  Stream short intervals once; retain target frames only, so residency is
    # still bounded by the current chunk's dynamic frame cap.
    if spec.end_time - spec.start_time <= 16.0 + 1.0e-6:
        frames, timestamps = _decode_targets_streaming(spec, target_times)
    else:
        try:
            frames, timestamps = _decode_targets_with_seek(spec, target_times)
        except _TargetSeekUnavailable:
            # A small minority of containers do not expose a seekable timestamp index.  The
            # streaming decoder still converts/retains only target frames; host memory therefore
            # remains bounded by ``maximum_frames``.
            frames, timestamps = _decode_targets_streaming(spec, target_times)
    if not frames:
        raise ValueError(f"current chunk {spec.chunk_id} contains no causal decoded frame")
    padded_single_frame = len(frames) == 1
    if padded_single_frame:
        # Qwen's temporal patch size is two.  Very early SVCBench Queries (0.1s)
        # can causally expose only the first decoded frame, so repeat that frame
        # inside the current chunk instead of reading a future frame.
        frames.append(frames[0].clone())
        timestamps.append(timestamps[0])
    elif len(frames) % 2:
        # Sparse/VFR sources can expose fewer unique frames than the nominal 2 FPS budget.
        # Drop one endpoint rather than duplicating an observation inside a normal interval.
        frames.pop()
        timestamps.pop()
    if len(frames) < 2:
        raise ValueError(f"current chunk {spec.chunk_id} has no complete temporal tubelet")
    selected_frames = torch.stack(frames).contiguous()
    selected_times = torch.tensor(timestamps, dtype=torch.float64)
    if not padded_single_frame and bool(torch.any(selected_times[1:] <= selected_times[:-1])):
        raise RuntimeError("sampled current-chunk timestamps are not strictly increasing")
    return selected_frames, selected_times


def _decode_coalesced_intervals(
    specs: tuple[CurrentChunkSpec, ...], sample_fps: float
) -> dict[str, tuple[Tensor, Tensor]]:
    """Decode multiple overlapping intervals from one PyAV demux pass.

    Only target frames are converted and retained.  The implementation intentionally mirrors the
    streaming decoder's right-closed boundary and sparse/VFR handling; it is an optimization of
    container setup/demux, not a different sampling policy.
    """

    if not specs:
        return {}
    path = specs[0].video_path
    if any(spec.video_path != path for spec in specs):
        raise ValueError("coalesced intervals must refer to one video")
    target_map = {spec.chunk_id: _uniform_target_times(spec, sample_fps) for spec in specs}
    frame_map: dict[str, list[Tensor]] = {spec.chunk_id: [] for spec in specs}
    time_map: dict[str, list[float]] = {spec.chunk_id: [] for spec in specs}
    next_target = {spec.chunk_id: 0 for spec in specs}
    last_candidate: dict[str, tuple[Any, float] | None] = {
        spec.chunk_id: None for spec in specs
    }
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
            timestamp = _av_timestamp(frame)
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
                while (
                    target_index < len(targets)
                    and timestamp + 1.0e-9 >= targets[target_index]
                ):
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


def _uniform_target_times(spec: CurrentChunkSpec, sample_fps: float) -> list[float]:
    desired = min(
        spec.maximum_frames,
        max(2, int(math.floor((spec.end_time - spec.start_time) * sample_fps)) + 1),
    )
    desired -= desired % 2
    if desired < 2:
        raise ValueError(f"current chunk {spec.chunk_id} has no complete temporal tubelet")
    return cast(
        list[float],
        torch.linspace(spec.start_time, spec.end_time, desired, dtype=torch.float64).tolist(),
    )


def _finalize_decoded_frames(
    frames: list[Tensor], timestamps: list[float], spec: CurrentChunkSpec
) -> tuple[list[Tensor], list[float]]:
    if not frames:
        raise ValueError(f"current chunk {spec.chunk_id} contains no causal decoded frame")
    padded_single_frame = len(frames) == 1
    if padded_single_frame:
        frames.append(frames[0].clone())
        timestamps.append(timestamps[0])
    elif len(frames) % 2:
        frames.pop()
        timestamps.pop()
    if len(frames) < 2:
        raise ValueError(f"current chunk {spec.chunk_id} has no complete temporal tubelet")
    if not padded_single_frame and any(
        right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise RuntimeError("sampled current-chunk timestamps are not strictly increasing")
    return frames, timestamps


class _TargetSeekUnavailable(RuntimeError):
    """Signal that a media container cannot service timestamp-targeted decoding."""


def _decode_targets_with_seek(
    spec: CurrentChunkSpec,
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
            timestamp = _av_timestamp(frame)
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
    spec: CurrentChunkSpec,
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
            timestamp = _av_timestamp(frame)
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


def _av_timestamp(frame: av.VideoFrame) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None or frame.time_base is None:
        raise ValueError("decoded video frame has no auditable timestamp")
    return float(frame.pts * frame.time_base)


def _resize_to_pixel_budget(
    frames: Tensor,
    *,
    minimum_pixels: int,
    maximum_pixels: int,
) -> Tensor:
    if minimum_pixels <= 0 or maximum_pixels <= 0 or minimum_pixels > maximum_pixels:
        raise ValueError("video pixel bounds must satisfy 0 < minimum <= maximum")
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
    "CurrentChunkSpec",
    "PreparedA2Record",
    "PreparedA5Record",
    "PreparedAnswerCPU",
    "ProductionVisualAudit",
    "VideoChunkMaterializer",
    "build_runtime",
]
