"""Run the causal, per-video online State-TTT inference protocol.

Inputs: one label-free runtime payload, timestamped chunks, injected P13 model
components, and one explicit TTT update/generation driver.
Outputs: one Reader-backed answer plus reset/chunk/update/generate/release audits.
Forbidden: training labels, cross-video state, future frames, in-place runtime
mutation, or repeated observe/update work during autoregressive decode.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import (
    RUNTIME_DENYLIST,
    RuntimeQueryInput,
    assert_runtime_payload_safe,
)
from ttt_svcbench_qwen.fast_ttt import (
    MEMORY_DIM,
    FastMemoryState,
    FastTTTAdapter,
    apply_memory_writes,
)
from ttt_svcbench_qwen.identity_bank import IdentityBank
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BatchRuntimeState,
    LifecyclePhase,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    PreparedQueryOutput,
    RuntimeOwner,
    StateTTTModel,
    TrajectoryRuntimeState,
)
from ttt_svcbench_qwen.state_bank import StructuredStateBank, TensorizedRetrievalHistory
from ttt_svcbench_qwen.state_encoder import TemporalCache
from ttt_svcbench_qwen.state_reader import ReaderResult

type AuditValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class QueryAttempt:
    query_id: str


@dataclass(frozen=True, slots=True)
class CausalChunk:
    """Timestamp-aligned frame payload that can be cropped before model handoff."""

    chunk_id: str
    frames: tuple[object, ...]
    timestamps: tuple[float, ...]
    position_ids: tuple[int, ...]
    model_input: object | None = None

    @property
    def start_time(self) -> float:
        return self.timestamps[0]

    @property
    def end_time(self) -> float:
        return self.timestamps[-1]

    def causal_prefix(self, query_time: float) -> CausalChunk | None:
        """Return only frames at or before query_time; never trust upstream cropping."""

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
    prepared_query: PreparedQueryOutput | None = None
    query_observation: CausalChunk | None = None
    max_new_tokens: int = 16

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
class TTTUpdateOutcome:
    runtime_state: TrajectoryRuntimeState
    did_update: bool
    skip_reason: str | None
    valid_token_count: int


class TTTUpdateStage(Protocol):
    def __call__(
        self,
        observation: ObservationChunkOutput,
        runtime_state: TrajectoryRuntimeState,
        *,
        current_end_time: float,
    ) -> TTTUpdateOutcome: ...


class OnlineTTTUpdater:
    """Apply the label-free slot-memory write and publish M_(t+1)."""

    def __init__(
        self,
        config: ProjectConfig,
        fast_adapter: FastTTTAdapter,
    ) -> None:
        self.config = config
        self.fast_adapter = fast_adapter

    def __call__(
        self,
        observation: ObservationChunkOutput,
        runtime_state: TrajectoryRuntimeState,
        *,
        current_end_time: float,
    ) -> TTTUpdateOutcome:
        del current_end_time
        intermediates = observation.soft_intermediates.fast_associative
        spatial = observation.soft_intermediates.spatial
        fast_state = runtime_state.fast_weights
        with torch.no_grad():
            batch = self.fast_adapter.prepare_write(intermediates, spatial)
            result = apply_memory_writes(fast_states=(fast_state,), batch=batch)[0]
        updated = replace(runtime_state, fast_weights=result.fast_state)
        valid_token_count = int(intermediates.valid_mask[0].sum().item())
        return TTTUpdateOutcome(
            runtime_state=updated,
            did_update=result.did_write,
            skip_reason=None if result.skip_reason is None else result.skip_reason.value,
            valid_token_count=valid_token_count if result.did_write else 0,
        )


@dataclass(frozen=True, slots=True)
class ChunkExecution:
    observation: ObservationChunkOutput | None
    runtime_state: TrajectoryRuntimeState


@dataclass(frozen=True, slots=True)
class InferenceResult:
    answer_text: str
    reader_result: ReaderResult
    runtime_state: TrajectoryRuntimeState
    selected_record_ids: tuple[str, ...]
    chunk_count: int
    released: bool
    audit_fields: tuple[tuple[str, AuditValue], ...]


class PerVideoRuntimeManager:
    """Own exactly one functional per-video runtime and all lifecycle transitions."""

    def __init__(
        self,
        *,
        fast_adapter: FastTTTAdapter,
        state_bank: StructuredStateBank,
        identity_bank: IdentityBank,
        hot_cache_enabled: bool = False,
        hot_device: str | torch.device | None = None,
    ) -> None:
        self.fast_adapter = fast_adapter
        self.state_bank = state_bank
        self.identity_bank = identity_bank
        self.hot_cache_enabled = hot_cache_enabled
        self.hot_device = hot_device
        self._runtime: TrajectoryRuntimeState | None = None
        self._lifecycle: PrefillLifecycle | None = None
        self._chunk_count = 0
        self._lock = RLock()

    @property
    def active_runtime(self) -> TrajectoryRuntimeState | None:
        with self._lock:
            return self._runtime

    def reset(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> None:
        """Reset all memory/cache/Bank/FSM/Reader state for one video."""

        with self._lock:
            if self._runtime is not None:
                self._release_locked()

            owner = RuntimeOwner((video_id,), (trajectory_id,))
            dtype = self.fast_adapter.w0_1.dtype
            device = self.fast_adapter.w0_1.device
            signature = query_signature.detach().to(device=device, dtype=dtype).clone()
            fast = self.fast_adapter.reset_fast_state(differentiable=False)
            runtime = TrajectoryRuntimeState(
                owner=owner,
                next_chunk_index=0,
                fast_weights=fast,
                slot_state=None,
                temporal_cache=_empty_temporal_cache(owner, signature),
                e1_state=None,
                e2_state=None,
                state_bank=self.state_bank.reset(video_id, trajectory_id),
                identity_bank=self.identity_bank.reset(
                    video_id,
                    trajectory_id,
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
            self._runtime = runtime
            self._lifecycle = PrefillLifecycle(owner)
            self._chunk_count = 0

    def observe_chunk(
        self,
        *,
        model: StateTTTModel,
        chunk: CausalChunk,
        query_input: RuntimeQueryInput,
        query_time: float,
        updater: TTTUpdateStage,
        prepared_query: PreparedQueryOutput | None = None,
    ) -> ChunkExecution:
        """Observe with M_t, write hard state, then create M_(t+1) for the next chunk."""

        with self._lock:
            runtime = self._require_live_runtime()
            fast = cast(FastMemoryState, runtime.fast_weights)
            lifecycle = cast(PrefillLifecycle, self._lifecycle)
            causal = chunk.causal_prefix(query_time)
            if causal is None:
                return ChunkExecution(None, runtime)

            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            model_runtime = BatchRuntimeState((runtime,))
            model_bank_states = model_runtime.bank_states
            with self.fast_adapter.use_fast_state(fast):
                observation = model.observe_chunk(
                    ObservationChunkRequest(
                        owner=owner,
                        video_input=causal if causal.model_input is None else causal.model_input,
                        query_input=query_input,
                        runtime_state=model_runtime,
                        bank_states=model_bank_states,
                        prepared_query=prepared_query,
                        inference=True,
                    ),
                    lifecycle,
                )
            observed_batch = cast(BatchRuntimeState, observation.runtime_state)
            observed = observed_batch.rows[0]
            observation = replace(
                observation,
                runtime_state=observed_batch,
                bank_states=(observed.state_bank,),
            )
            outcome = updater(
                observation,
                observed,
                current_end_time=causal.end_time,
            )
            updated = outcome.runtime_state
            next_observation = replace(
                observation,
                runtime_state=BatchRuntimeState((updated,)),
                bank_states=(updated.state_bank,),
            )
            self._runtime = updated
            self._chunk_count += 1
            return ChunkExecution(next_observation, updated)

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

        with self._lock:
            runtime = self._require_live_runtime()
            fast = cast(FastMemoryState, runtime.fast_weights)
            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            observation = replace(
                observation,
                runtime_state=BatchRuntimeState((runtime,)),
                bank_states=(runtime.state_bank,),
            )
            lifecycle = self._query_lifecycle(owner)
            with self.fast_adapter.use_fast_state(fast), torch.no_grad():
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
            reader_result = generated.reader[0]
            self._runtime = replace(
                runtime,
                reader_audit=runtime.reader_audit + (reader_result,),
            )
            lifecycle_audit = lifecycle.audit()
            result_runtime = self._runtime
            return InferenceResult(
                answer_text=generated.answer_text,
                reader_result=reader_result,
                runtime_state=result_runtime,
                selected_record_ids=reader_result.selected_record_ids,
                chunk_count=self._chunk_count,
                released=False,
                audit_fields=(
                    ("video_id", runtime.video_id),
                    ("trajectory_id", runtime.trajectory_id),
                    ("query_id", attempt.query_id),
                    ("reader_status", reader_result.status.value),
                    ("selected_record_count", len(reader_result.selected_record_ids)),
                    ("prefill_count", lifecycle_audit.prefill_count),
                    ("final_write_version", fast.write_version),
                    ("final_write_count", fast.write_count),
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
        prepared_query: PreparedQueryOutput | None = None,
    ) -> ObservationChunkOutput:
        """Observe one Query feature set with current W_t without committing Bank/FSM state."""

        with self._lock:
            runtime = self._require_live_runtime()
            fast = cast(FastMemoryState, runtime.fast_weights)
            causal = chunk.causal_prefix(query_time)
            if causal is None:
                raise RuntimeError("Query observation contains no causal frame")
            owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
            lifecycle = PrefillLifecycle(owner)
            with self.fast_adapter.use_fast_state(fast), torch.no_grad():
                observation = model.observe_chunk(
                    ObservationChunkRequest(
                        owner=owner,
                        video_input=causal if causal.model_input is None else causal.model_input,
                        query_input=query_input,
                        runtime_state=BatchRuntimeState((runtime,)),
                        bank_states=(runtime.state_bank,),
                        prepared_query=prepared_query,
                        inference=True,
                        retrieval_history_write_enabled=False,
                    ),
                    lifecycle,
                )
            return replace(
                observation,
                runtime_state=BatchRuntimeState((runtime,)),
                bank_states=(runtime.state_bank,),
            )

    def release(self) -> TrajectoryRuntimeState | None:
        """Release all trajectory storage; safe and idempotent for exception cleanup."""

        with self._lock:
            if self._runtime is None:
                return None
            return self._release_locked()

    def _release_locked(self) -> TrajectoryRuntimeState:
        runtime = self._require_live_runtime()
        temporal_cache = cast(TemporalCache, runtime.temporal_cache)
        released_bank = self.state_bank.release(runtime.state_bank)
        released_identity = self.identity_bank.release(runtime.identity_bank)
        if runtime.retrieval_history is not None:
            runtime.retrieval_history.release()
        owner = RuntimeOwner((runtime.video_id,), (runtime.trajectory_id,))
        released = TrajectoryRuntimeState(
            owner=owner,
            next_chunk_index=0,
            fast_weights=_released_fast_state(),
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
        self._runtime = None
        self._lifecycle = None
        return released

    def _query_lifecycle(self, owner: RuntimeOwner) -> PrefillLifecycle:
        lifecycle = self._lifecycle
        if lifecycle is None:
            raise RuntimeError("runtime lifecycle is unavailable")
        if lifecycle.audit().phase is not LifecyclePhase.READY:
            lifecycle = PrefillLifecycle(owner)
            self._lifecycle = lifecycle
        return lifecycle

    def _require_live_runtime(self) -> TrajectoryRuntimeState:
        runtime = self._runtime
        if runtime is None or runtime.released:
            raise RuntimeError("a live per-video runtime is required")
        return runtime


def run_inference(
    *,
    manager: PerVideoRuntimeManager,
    model: StateTTTModel,
    request: InferenceRequest,
    updater: TTTUpdateStage,
) -> InferenceResult:
    """Execute reset -> causal chunks -> one greedy generate -> unconditional release."""

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
                prepared_query=request.prepared_query,
            )
            if execution.observation is not None:
                latest_observation = execution.observation
        if request.query_observation is not None:
            latest_observation = manager.observe_query_readonly(
                model=model,
                chunk=request.query_observation,
                query_input=request.query_input,
                query_time=request.query_time,
                prepared_query=request.prepared_query,
            )
        if latest_observation is None:
            raise RuntimeError("no causal frame was available before query_time")
        result = manager.answer_query(
            model=model,
            observation=latest_observation,
            answer_inputs=request.answer_inputs,
            attempt=request.attempt,
            max_new_tokens=request.max_new_tokens,
        )
        released_state = cast(TrajectoryRuntimeState, manager.release())
        return replace(
            result,
            runtime_state=released_state,
            released=True,
            audit_fields=result.audit_fields + (("released", True),),
        )
    except BaseException:
        manager.release()
        raise


def assert_inference_runtime_payload(payload: Mapping[str, object]) -> None:
    """Apply the P2 allowlist and recursively reject nested supervision fields."""

    assert_runtime_payload_safe(payload, layer="Inference")
    denied_paths = tuple(_nested_denied_paths(payload))
    if denied_paths:
        raise ValueError(
            "Inference runtime payload contains nested denied fields: "
            + ", ".join(sorted(denied_paths))
        )


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


def _released_fast_state() -> FastMemoryState:
    return FastMemoryState(
        m=torch.empty(
            (MEMORY_DIM, MEMORY_DIM),
            dtype=torch.float32,
            device="meta",
            requires_grad=True,
        ),
        write_version=0,
        write_count=0,
        skip_count=0,
        differentiable=False,
    )


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
    return float(cast(float | int, payload["query_time"]))


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
    runtime_payload = {
        name: payload[name] for name in ("video", "question", "query_time", "explicit_time_values")
    }
    assert_inference_runtime_payload(runtime_payload)
    video_path = Path(cast(str, payload["video"])).resolve()
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
    apply_template = processor.apply_chat_template
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
            rope_indexer=getattr(
                bundle.qwen_adapter.qwen_model,
                "model",
                bundle.qwen_adapter.qwen_model,
            ),
        ),
        attempt=QueryAttempt(query.query_id),
        prepared_query=PreparedQueryOutput.bind(query, encoded),
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
        "write_version": audits["final_write_version"],
        "write_count": audits["final_write_count"],
        "skip_count": audits["final_skip_count"],
        "audit": {
            "level": bundle.config.inference.audit_level.value,
            "prefill_count": audits["prefill_count"],
            "decode_count": audits["decode_count"],
            "chunk_count": result.chunk_count,
            "state_query_visual_mode": "recent_chunk",
            "answer_query_visual_mode": "causal_prefix",
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
            "released": result.released,
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
