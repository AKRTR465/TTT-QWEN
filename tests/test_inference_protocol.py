from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.fast_ttt import FastTTTAdapter, FastWeightsState, build_fast_ttt_adapter
from ttt_svcbench_qwen.identity_bank import IdentityBank, build_identity_bank
from ttt_svcbench_qwen.inference import (
    AnswerInputs,
    CausalChunk,
    GenerationDriver,
    InferenceProtocolError,
    InferenceRequest,
    PerVideoRuntimeManager,
    QueryAttempt,
    QueryAttemptKind,
    TTTUpdateOutcome,
    assert_inference_runtime_payload,
    run_inference,
)
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    BatchRuntimeState,
    DecodeStepOutput,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkOutput,
    ObservationChunkRequest,
    QwenPrefillRequest,
    StateTTTModel,
    StateTTTModelOutput,
    TrajectoryRuntimeState,
    VisualStageOutput,
    build_model,
)
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
    StreamReplayAudit,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    OperatorRouterOutput,
    QueryEmbeddingOutput,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeResolverLogits,
    TimeResolverOutput,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.stage_a_runtime import StageABankWriter
from ttt_svcbench_qwen.state_bank import HeadType, StructuredStateBank, build_state_bank
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
    TemporalEncoderOutput,
)
from ttt_svcbench_qwen.state_reader import ReaderResult, ReaderStatus


class _Dependencies(SimpleNamespace):
    config: ProjectConfig
    fast_adapter: FastTTTAdapter
    state_bank: StructuredStateBank
    identity_bank: IdentityBank


@pytest.fixture(scope="module")
def dependencies() -> _Dependencies:
    config = load_config()
    return _Dependencies(
        config=config,
        fast_adapter=build_fast_ttt_adapter(config),
        state_bank=build_state_bank(config),
        identity_bank=build_identity_bank(config),
    )


def _manager(dependencies: _Dependencies) -> PerVideoRuntimeManager:
    return PerVideoRuntimeManager(
        fast_adapter=dependencies.fast_adapter,
        state_bank=dependencies.state_bank,
        identity_bank=dependencies.identity_bank,
        optimizer_config=dependencies.config.fast_ttt.optimizer,
        hot_cache_enabled=False,
    )


def _reader_result(status: ReaderStatus) -> ReaderResult:
    count_bearing = status in (ReaderStatus.OK, ReaderStatus.EMPTY)
    selected_ids = ("record-0",) if status is ReaderStatus.OK else ()
    exact_count = 2 if status is ReaderStatus.OK else 0 if status is ReaderStatus.EMPTY else None
    operator = Operator.O1_SNAP if status is not ReaderStatus.UNSUPPORTED else Operator.UNSUPPORTED
    valid_window = count_bearing
    audit: list[tuple[str, str | int | float | bool | None]] = [
        ("source", "retrieved_typed_records"),
        ("operator", operator.value),
        ("retrieval_status", status.value),
        ("retrieval_reason", f"synthetic_{status.value}"),
        ("n_state", len(selected_ids)),
        ("n_retrieved", len(selected_ids)),
        ("input_record_count", len(selected_ids)),
        ("bank_version", 1),
        ("time_resolution_status", "ok" if valid_window else status.value),
        ("window_start", None if operator is Operator.UNSUPPORTED else 0.0),
        ("window_end", 2.0),
        ("reader_reason", f"synthetic_{status.value}"),
    ]
    if count_bearing:
        audit.extend(
            (
                ("arithmetic", "synthetic_o1_snap"),
                ("contributing_count", len(selected_ids)),
                ("computed_exact_count", cast(int, exact_count)),
                ("number_text", str(exact_count)),
            )
        )
    if status is ReaderStatus.OK:
        audit.extend(
            (
                ("operand_current_visible_count", 2),
                ("operand_baseline_count", 0),
                ("operand_baseline_initialized", True),
                ("operand_baseline_position_id", 0),
            )
        )
    return ReaderResult(
        status=status,
        exact_count=exact_count,
        number_token_ids=() if exact_count is None else (48 + exact_count,),
        selected_record_ids=selected_ids,
        operator=operator,
        time_window=TimeWindow(
            TimeWindowMode.HISTORY,
            2.0,
            0.0,
            2.0,
            valid_window,
        ),
        audit_fields=tuple(audit),
    )


class _FakeSuite:
    def __init__(self, status: ReaderStatus = ReaderStatus.OK) -> None:
        self.status = status
        self.fast_adapter: FastTTTAdapter | None = None
        self.fast_mode = "consume"
        self.fast_versions: list[int] = []
        self.seen_frames: list[tuple[object, ...]] = []
        self.prefill_calls = 0
        self.decode_calls = 0

    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        batch = cast(BatchRuntimeState, request.runtime_state)
        runtime = batch.rows[0]
        assert runtime.fast_weights is not None
        chunk = cast(CausalChunk, request.video_input)
        self.fast_versions.append(runtime.fast_weights.fast_version)
        self.seen_frames.append(chunk.frames)
        value = (chunk.frames, runtime.fast_weights.fast_version)
        return VisualStageOutput(value=value, prepared_video_features=value)

    @staticmethod
    def query(_query_input: object, *, inference: bool) -> object:
        assert inference
        return SimpleNamespace(q_target=torch.zeros((1, 512)))

    def fast(
        self,
        visual: VisualStageOutput,
        _query: object,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        if self.fast_adapter is None:
            raise RuntimeError("test suite Fast Adapter was not installed")
        if self.fast_mode == "skip":
            return visual
        if self.fast_mode == "reenter":
            runtime = cast(BatchRuntimeState, request.runtime_state).rows[0]
            assert runtime.fast_weights is not None
            with self.fast_adapter.use_fast_state(runtime.fast_weights):
                pass
            return visual
        dtype = self.fast_adapter.w0_1.dtype
        device = self.fast_adapter.w0_1.device
        self.fast_adapter(torch.zeros((1, 1, 4096), dtype=dtype, device=device))
        return visual

    @staticmethod
    def spatial(
        _visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> object:
        return "spatial"

    @staticmethod
    def temporal(
        _visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> object:
        return "temporal"

    @staticmethod
    def heads(
        _spatial: object,
        _temporal: object,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> object:
        return "observations"

    @staticmethod
    def writer(
        _observations: object,
        _spatial: object,
        _temporal: object,
        _query: object,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput:
        batch = cast(BatchRuntimeState, request.runtime_state)
        runtime = batch.rows[0]
        bank = replace(runtime.state_bank, version=runtime.state_bank.version + 1)
        next_runtime = replace(runtime, state_bank=bank)
        return BankWriteOutput(
            runtime_state=BatchRuntimeState((next_runtime,)),
            bank_states=(bank,),
            audit=("bank_version", bank.version),
        )

    def retrieve_query(self, *_args: object, **_kwargs: object) -> object:
        result = _reader_result(self.status)
        return SimpleNamespace(
            selected_record_ids=(result.selected_record_ids,),
            status=(self.status.value,),
            audit=("retrieval", self.status.value),
        )

    def read(self, _retrieval: object) -> tuple[ReaderResult, ...]:
        return (_reader_result(self.status),)

    @staticmethod
    def audit_results(
        _retrieval: object,
        results: tuple[ReaderResult, ...],
    ) -> tuple[ReaderResult, ...]:
        return results

    @staticmethod
    def audit_number_tokens(result: ReaderResult) -> int | None:
        return result.exact_count

    def resample(self, _q_target: object, _retrieval: object) -> object:
        result = _reader_result(self.status)
        return SimpleNamespace(
            state_tokens=torch.zeros((1, 16, 4096)),
            state_token_valid_mask=torch.ones((1, 16), dtype=torch.bool),
            selected_record_ids=(result.selected_record_ids,),
            retrieval_status=(self.status.value,),
            cross_attention_weights=torch.ones((1, 16, max(1, len(result.selected_record_ids)))),
        )

    @staticmethod
    def compose(**_kwargs: object) -> object:
        return SimpleNamespace(
            input_ids=torch.ones((1, 4), dtype=torch.int64),
            attention_mask=torch.ones((1, 4), dtype=torch.bool),
            position_ids=torch.arange(4).reshape(1, 4),
            rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
            state_position_mask=torch.ones((1, 4), dtype=torch.bool),
        )

    def prefill(self, request: QwenPrefillRequest) -> object:
        self.prefill_calls += 1
        return SimpleNamespace(
            logits=torch.ones((1, 1, 8)),
            signature=repr(request.prepared_video_features),
        )

    def decode(self, model_inputs: object) -> object:
        self.decode_calls += 1
        return SimpleNamespace(token=model_inputs)


def _typed_query() -> QueryEncoderOutput:
    q_target = torch.zeros((1, 512))
    embeddings = QueryEmbeddingOutput(
        token_states=torch.zeros((1, 1, 768)),
        pooling_weights=torch.ones((1, 1)),
        q_target=q_target,
        q_operator=q_target.clone(),
        q_time=q_target.clone(),
        padding_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    operator = Operator.O1_SNAP
    raw_index = tuple(Operator).index(operator)
    logits = torch.full((1, len(tuple(Operator))), -5.0)
    logits[0, raw_index] = 5.0
    route = OperatorRouterOutput(
        logits=logits,
        confidence=torch.ones(1),
        raw_indices=torch.tensor((raw_index,), dtype=torch.int64),
        hard_operators=(operator,),
        head_types=(OPERATOR_TO_HEAD_TYPE[operator],),
        confidence_gate_applied=False,
    )
    time_logits = TimeResolverLogits(
        mode_logits=torch.zeros((1, 4)),
        mode_confidence=torch.ones(1),
        mode_indices=torch.ones(1, dtype=torch.int64),
        span_start_logits=torch.zeros((1, 1)),
        span_end_logits=torch.zeros((1, 1)),
        padding_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    resolution = TimeResolution(
        window=TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True),
        status=TimeResolutionStatus.OK,
        reason="synthetic_explicit",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
    )
    return QueryEncoderOutput(
        embeddings=embeddings,
        route=route,
        time=TimeResolverOutput(time_logits, (resolution,)),
        hard_operators=(operator,),
        head_types=(HeadType.O1,),
    )


def _typed_cache(hidden: Tensor, query: Tensor) -> TemporalCache:
    width = hidden.shape[1]
    kv = tuple(torch.zeros((1, 12, width, 64)) for _ in range(6))
    replay = tuple(torch.zeros((1, 12, 0, 64)) for _ in range(6))
    return TemporalCache(
        hidden=hidden.detach().clone(),
        layer_keys=kv,
        layer_values=tuple(value.clone() for value in kv),
        replay_layer_keys=replay,
        replay_layer_values=tuple(value.clone() for value in replay),
        timestamps=torch.arange(width, dtype=torch.float64).reshape(1, width),
        replay_timestamps=torch.empty((1, 0), dtype=torch.float64),
        position_ids=torch.arange(width, dtype=torch.int64).reshape(1, width),
        replay_position_ids=torch.empty((1, 0), dtype=torch.int64),
        valid_mask=torch.ones((1, width), dtype=torch.bool),
        replay_valid_mask=torch.empty((1, 0), dtype=torch.bool),
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
        query_signatures=query.detach().clone(),
        total_seen=torch.tensor((width,), dtype=torch.int64),
    )


def _typed_spatial() -> SpatialEncoderOutput:
    slots = torch.randn((1, 1, 768))
    state = SpatialSlotRuntimeState(
        video_id="video-a",
        slots=slots[0].detach().clone(),
        slot_valid_mask=torch.ones(1, dtype=torch.bool),
        slot_confidence=torch.ones(1),
        active_slot_overflow_count=0,
        overflow_event_count=0,
        processed_tubelets=2,
    )
    return SpatialEncoderOutput(
        slots=slots,
        slot_valid_mask=torch.ones((1, 1), dtype=torch.bool),
        active_slot_overflow_count=torch.zeros(1, dtype=torch.int64),
        slot_confidence=torch.ones((1, 1)),
        next_states=(state,),
    )


def _typed_temporal(query: QueryEncoderOutput) -> TemporalEncoderOutput:
    hidden = torch.randn((1, 2, 768))
    return TemporalEncoderOutput(
        hidden=hidden,
        timestamps=torch.tensor(((0.0, 1.0),), dtype=torch.float64),
        position_ids=torch.tensor(((0, 1),), dtype=torch.int64),
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        cache=_typed_cache(hidden, query.q_target),
    )


def _typed_observations(
    spatial: SpatialEncoderOutput,
    temporal: TemporalEncoderOutput,
    query: QueryEncoderOutput,
) -> ObservationOutputs:
    slot_times = torch.ones((1, 1), dtype=torch.float64)
    slot_positions = torch.ones((1, 1), dtype=torch.int64)
    o1_logits = torch.full((1, 1, 6), 5.0)
    o1_probabilities = torch.sigmoid(o1_logits)
    o1 = O1SoftOutput(
        logits=o1_logits,
        probabilities=o1_probabilities,
        soft_count=o1_probabilities[..., :3].prod(dim=-1).sum(dim=1),
        valid_mask=spatial.slot_valid_mask.clone(),
        timestamps=slot_times,
        position_ids=slot_positions,
    )
    identities = torch.nn.functional.normalize(torch.randn((1, 1, 256)), dim=-1)
    score_logits = torch.tensor([[[5.0, -5.0]]])
    o2 = O2SoftOutput(
        identity=identities,
        score_logits=score_logits,
        score_probabilities=torch.sigmoid(score_logits),
        valid_mask=spatial.slot_valid_mask.clone(),
        timestamps=slot_times.clone(),
        position_ids=slot_positions.clone(),
    )
    e1_state = E1RuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_signature=query.q_target[0].detach().clone(),
        projected_history=torch.zeros((2, 512)),
        timestamps=temporal.timestamps[0].clone(),
        position_ids=temporal.position_ids[0].clone(),
        total_seen=2,
    )
    e2_state = E2RuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_signature=query.q_target[0].detach().clone(),
        hidden=torch.zeros((2, 768)),
        checkpoint_hidden=torch.zeros((2, 2, 768)),
        timestamps=temporal.timestamps[0].clone(),
        position_ids=temporal.position_ids[0].clone(),
        total_seen=2,
    )
    e1_logits = torch.full((1, 2, 3), 5.0)
    e1 = E1SoftOutput(
        logits=e1_logits,
        probabilities=torch.sigmoid(e1_logits),
        valid_mask=temporal.valid_mask.clone(),
        timestamps=temporal.timestamps.clone(),
        position_ids=temporal.position_ids.clone(),
        next_states=(e1_state,),
        audit=StreamReplayAudit("e1", (2,), (0,), (2,)),
    )
    event_logits = torch.full((1, 2, 4), 5.0)
    phase_logits = torch.zeros((1, 2, 4))
    e2 = E2SoftOutput(
        event_logits=event_logits,
        phase_logits=phase_logits,
        event_probabilities=torch.sigmoid(event_logits),
        phase_probabilities=torch.softmax(phase_logits, dim=-1),
        valid_mask=temporal.valid_mask.clone(),
        timestamps=temporal.timestamps.clone(),
        position_ids=temporal.position_ids.clone(),
        next_states=(e2_state,),
        audit=StreamReplayAudit("e2", (2,), (0,), (2,)),
    )
    return ObservationOutputs(o1=o1, o2=o2, e1=e1, e2=e2)


class _TypedStageSuite(_FakeSuite):
    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        chunk = cast(CausalChunk, request.video_input)
        self.seen_frames.append(chunk.frames)
        value = (chunk.frames, "stage-a-runtime")
        return VisualStageOutput(value=value, prepared_video_features=value)

    @staticmethod
    def query(_query_input: object, *, inference: bool) -> QueryEncoderOutput:
        assert inference
        return _typed_query()

    @staticmethod
    def spatial(
        _visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> SpatialEncoderOutput:
        return _typed_spatial()

    @staticmethod
    def temporal(
        _visual: VisualStageOutput,
        query: object,
        _request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput:
        return _typed_temporal(cast(QueryEncoderOutput, query))

    @staticmethod
    def heads(
        spatial: object,
        temporal: object,
        query: object,
        _request: ObservationChunkRequest,
    ) -> ObservationOutputs:
        return _typed_observations(
            cast(SpatialEncoderOutput, spatial),
            cast(TemporalEncoderOutput, temporal),
            cast(QueryEncoderOutput, query),
        )


def _model(dependencies: _Dependencies, suite: _FakeSuite) -> StateTTTModel:
    suite.fast_adapter = dependencies.fast_adapter
    return build_model(
        dependencies.config,
        components=ModelComponents(
            visual_stage=suite.visual,
            query_encoder=suite.query,
            composer=suite.compose,
            qwen_prefill=suite.prefill,
            qwen_decode=suite.decode,
            fast_adapter=suite.fast,
            spatial_encoder=suite.spatial,
            temporal_encoder=suite.temporal,
            observation_heads=suite.heads,
            state_bank=dependencies.state_bank,
            bank_writer=suite.writer,
            retriever=suite,
            reader=suite,
            resampler=suite.resample,
        ),
        feature_flags=ModelFeatureFlags(),
    )


def _stage_a_model(dependencies: _Dependencies, suite: _TypedStageSuite) -> StateTTTModel:
    suite.fast_adapter = dependencies.fast_adapter
    return build_model(
        dependencies.config,
        components=ModelComponents(
            visual_stage=suite.visual,
            query_encoder=suite.query,
            composer=suite.compose,
            qwen_prefill=suite.prefill,
            qwen_decode=suite.decode,
            fast_adapter=suite.fast,
            spatial_encoder=suite.spatial,
            temporal_encoder=suite.temporal,
            observation_heads=suite.heads,
            state_bank=dependencies.state_bank,
            bank_writer=StageABankWriter(dependencies.state_bank, dependencies.identity_bank),
            retriever=suite,
            reader=suite,
            resampler=suite.resample,
        ),
        feature_flags=ModelFeatureFlags(),
    )


class _Updater:
    def __init__(self, skip_calls: set[int] | None = None) -> None:
        self.calls = 0
        self.skip_calls = skip_calls or set()

    def __call__(
        self,
        _observation: ObservationChunkOutput,
        runtime: TrajectoryRuntimeState,
    ) -> TTTUpdateOutcome:
        call = self.calls
        self.calls += 1
        fast = runtime.fast_weights
        optimizer = runtime.optimizer
        assert fast is not None and optimizer is not None
        if call in self.skip_calls:
            reason = "no_valid_term"
            return TTTUpdateOutcome(
                runtime_state=replace(
                    runtime,
                    fast_weights=replace(fast, skip_count=fast.skip_count + 1),
                    optimizer=replace(
                        optimizer,
                        attempted_update_count=optimizer.attempted_update_count + 1,
                        last_skip_reason=reason,
                    ),
                ),
                did_update=False,
                skip_reason=reason,
                valid_term_count=0,
            )
        with torch.no_grad():
            next_w1 = (fast.w_t_1 - 1.0e-4).detach().clone().requires_grad_(True)
            next_w2 = (fast.w_t_2 + 1.0e-4).detach().clone().requires_grad_(True)
        next_fast = FastWeightsState(
            w0_1=fast.w0_1,
            w0_2=fast.w0_2,
            w_t_1=next_w1,
            w_t_2=next_w2,
            fast_version=fast.fast_version + 1,
            update_count=fast.update_count + 1,
            skip_count=fast.skip_count,
        )
        return TTTUpdateOutcome(
            runtime_state=replace(
                runtime,
                fast_weights=next_fast,
                optimizer=replace(
                    optimizer,
                    attempted_update_count=optimizer.attempted_update_count + 1,
                    last_skip_reason=None,
                ),
            ),
            did_update=True,
            skip_reason=None,
            valid_term_count=1,
            loss_value=0.25,
        )


class _Driver(GenerationDriver):
    def __init__(
        self,
        steps: int,
        *,
        mutate_after_first: PerVideoRuntimeManager | None = None,
    ) -> None:
        self.steps = steps
        self.mutate_after_first = mutate_after_first

    def begin(self, _prefill: StateTTTModelOutput) -> object | None:
        return None if self.steps == 0 else {"step": 0}

    def advance(self, step_index: int, _decode: DecodeStepOutput) -> object | None:
        if step_index == 0 and self.mutate_after_first is not None:
            state = self.mutate_after_first.active_runtime
            assert state is not None
            with torch.no_grad():
                state.fast_weights.w_t_1.add_(0.5)
        next_step = step_index + 1
        return None if next_step >= self.steps else {"step": next_step}

    @staticmethod
    def finish(
        prefill: StateTTTModelOutput,
        _decode_steps: tuple[DecodeStepOutput, ...],
    ) -> str:
        return f"answer:{prefill.qwen_output.signature}"


def _answer_inputs() -> AnswerInputs:
    return AnswerInputs(
        base_input_ids=torch.ones((1, 2), dtype=torch.int64),
        base_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        pixel_values_videos="pixels",
        video_grid_thw="grid",
        tokenizer="tokenizer",
        embedding_owner="embedding",
        rope_indexer="rope",
    )


def _request(*, future_frame: object = "future") -> InferenceRequest:
    return InferenceRequest.from_payload(
        video_id="video-a",
        trajectory_id="trajectory-a",
        payload={
            "video": "video-a.mp4",
            "question": "How many?",
            "query_time": 2.0,
            "explicit_time_values": (),
        },
        query_signature=torch.zeros(512),
        chunks=(
            CausalChunk("chunk-0", ("a", "b"), (0.0, 1.0), (0, 1)),
            CausalChunk("chunk-1", ("c", future_frame), (2.0, 4.0), (2, 3)),
        ),
        answer_inputs=_answer_inputs(),
        attempt=QueryAttempt("query-a"),
        max_decode_steps=8,
    )


def test_reset_isolates_consecutive_videos_and_matches_pristine_checksum(
    dependencies: _Dependencies,
) -> None:
    manager = _manager(dependencies)
    first = manager.reset("video-a", "trajectory-a", torch.zeros(512))
    first_state = manager.active_runtime
    assert first_state is not None
    first_pointer = first_state.fast_weights.w_t_1.untyped_storage().data_ptr()

    second = manager.reset("video-b", "trajectory-b", torch.ones(512))
    second_state = manager.active_runtime
    assert second_state is not None

    assert first.pristine_state_checksum == second.pristine_state_checksum
    assert second.previous_runtime_checksum is not None
    assert second.previous_release_checksum is not None
    assert second.w0_checksum == second.current_fast_checksum
    assert second_state.fast_weights.fast_version == 0
    assert second_state.optimizer.attempted_update_count == 0
    assert second_state.temporal_cache.hidden.shape[1] == 0
    assert second_state.slot_state is None
    assert second_state.e1_state is None and second_state.e2_state is None
    assert second_state.state_bank.records == ()
    assert second_state.identity_bank.candidates == ()
    assert second_state.reader_audit == ()
    assert second_state.fast_weights.w_t_1.untyped_storage().data_ptr() != first_pointer
    manager.release()


def test_causal_chunks_next_only_updates_and_decode_immutability(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    result = run_inference(
        manager=manager,
        model=_model(dependencies, suite),
        request=_request(),
        updater=_Updater(skip_calls={1}),
        generation_driver=_Driver(steps=3),
    )

    assert suite.fast_versions == [0, 1]
    assert suite.seen_frames == [("a", "b"), ("c",)]
    assert tuple(audit.fast_version_used for audit in result.chunk_audit) == (0, 1)
    assert tuple(audit.next_fast_version for audit in result.chunk_audit) == (1, 1)
    assert result.chunk_audit[1].future_frame_count == 1
    assert result.chunk_audit[1].skip_reason == "no_valid_term"
    assert result.generate_audit.prefill_count == 1
    assert result.generate_audit.decode_count == 3
    assert len(set(result.generate_audit.decode_state_checksums)) == 1
    assert result.reader_result.status is ReaderStatus.OK
    assert result.selected_record_ids == ("record-0",)
    assert result.state_attention is not None
    assert result.runtime_state.released
    assert result.release_audit is not None
    assert manager.active_runtime is None


def test_future_frame_perturbation_does_not_change_answer_or_model_input(
    dependencies: _Dependencies,
) -> None:
    outputs: list[tuple[str, list[tuple[object, ...]]]] = []
    for future in ("future-a", "future-b-perturbed"):
        suite = _FakeSuite()
        result = run_inference(
            manager=_manager(dependencies),
            model=_model(dependencies, suite),
            request=_request(future_frame=future),
            updater=_Updater(skip_calls={1}),
            generation_driver=_Driver(steps=1),
        )
        outputs.append((result.answer_text, suite.seen_frames))

    assert outputs[0] == outputs[1]


def test_reader_statuses_and_explicit_retry_use_one_prefill_each(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    execution = manager.observe_chunk(
        model=_model(dependencies, suite),
        chunk=CausalChunk("chunk", ("a",), (0.0,), (0,)),
        query_input=_request().query_input,
        query_time=2.0,
        updater=_Updater(),
    )
    assert execution.observation is not None
    statuses = (
        ReaderStatus.OK,
        ReaderStatus.EMPTY,
        ReaderStatus.UNSUPPORTED,
        ReaderStatus.INVALID,
    )
    results = []
    for index, status in enumerate(statuses):
        suite.status = status
        attempt = (
            QueryAttempt("query-retry", QueryAttemptKind.RETRY, retry_of="query-ok")
            if status is ReaderStatus.EMPTY
            else QueryAttempt(f"query-{status.value}")
        )
        result = manager.answer_query(
            model=_model(dependencies, suite),
            observation=execution.observation,
            answer_inputs=_answer_inputs(),
            attempt=attempt,
            generation_driver=_Driver(steps=1),
            max_decode_steps=4,
        )
        results.append(result)
        assert result.generate_audit.prefill_count == 1, index
        assert result.answer_text

    assert tuple(result.reader_result.status for result in results) == statuses
    assert results[1].generate_audit.query_kind is QueryAttemptKind.RETRY
    assert results[1].generate_audit.retry_of == "query-ok"
    manager.release()


def test_decode_mutation_fails_closed_and_exception_releases_runtime(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    with pytest.raises(InferenceProtocolError, match="decode/generation driver mutated"):
        run_inference(
            manager=manager,
            model=_model(dependencies, suite),
            request=_request(),
            updater=_Updater(skip_calls={1}),
            generation_driver=_Driver(steps=2, mutate_after_first=manager),
        )
    assert manager.active_runtime is None


def test_fast_binding_is_fail_closed_and_not_reentrant(dependencies: _Dependencies) -> None:
    for mode, error, message in (
        ("skip", InferenceProtocolError, "manager-bound FastWeightsState"),
        ("reenter", RuntimeError, "not re-entrant"),
    ):
        suite = _FakeSuite()
        suite.fast_mode = mode
        manager = _manager(dependencies)
        manager.reset("video-a", "trajectory-a", torch.zeros(512))
        with pytest.raises(error, match=message):
            manager.observe_chunk(
                model=_model(dependencies, suite),
                chunk=CausalChunk("chunk", ("a",), (0.0,), (0,)),
                query_input=_request().query_input,
                query_time=2.0,
                updater=_Updater(),
            )
        manager.release()
        assert manager.active_runtime is None


def test_unified_runtime_commits_real_hard_state(dependencies: _Dependencies) -> None:
    suite = _TypedStageSuite()
    manager = PerVideoRuntimeManager(
        fast_adapter=dependencies.fast_adapter,
        state_bank=dependencies.state_bank,
        identity_bank=dependencies.identity_bank,
        optimizer_config=dependencies.config.fast_ttt.optimizer,
        hot_cache_enabled=False,
    )
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    execution = manager.observe_chunk(
        model=_stage_a_model(dependencies, suite),
        chunk=CausalChunk("chunk", ("a", "b"), (0.0, 1.0), (0, 1)),
        query_input=_request().query_input,
        query_time=2.0,
        updater=_Updater(),
    )

    runtime = execution.runtime_state
    assert isinstance(execution.observation, ObservationChunkOutput)
    assert isinstance(execution.observation.runtime_state, BatchRuntimeState)
    assert execution.observation.runtime_state.rows[0] is runtime
    assert runtime.slot_state is not None
    assert runtime.temporal_cache is not None
    assert runtime.temporal_cache.hidden.shape == (1, 2, 768)
    assert runtime.e1_state is not None and runtime.e1_state.total_seen == 2
    assert runtime.e2_state is not None and runtime.e2_state.total_seen == 2
    assert len(runtime.state_bank.records) == 1
    assert runtime.state_bank.records[0].head_type is HeadType.O1
    assert runtime.identity_bank.video_id == "video-a"
    assert runtime.optimizer is not None and runtime.optimizer.attempted_update_count == 1
    manager.release()


def test_inference_payload_recursively_rejects_labels() -> None:
    safe = {
        "video": "video.mp4",
        "question": "How many?",
        "query_time": 2.0,
        "explicit_time_values": (),
    }
    assert_inference_runtime_payload(safe)
    with pytest.raises(ValueError, match="denied fields"):
        assert_inference_runtime_payload({**safe, "answer": "2"})
    with pytest.raises(ValueError, match="nested denied fields"):
        assert_inference_runtime_payload({**safe, "video": {"frames": (), "count": 2}})
