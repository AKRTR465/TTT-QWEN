from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from tests.support import make_test_model as build_model
from tests.support.runtime_factories import (
    make_e1_state,
    make_e2_state,
    make_spatial_output,
    make_temporal_cache,
)
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    AssociativeTTTIntermediates,
    FastAssociativeContext,
    FastTTTAdapter,
    build_fast_ttt_adapter,
)
from ttt_svcbench_qwen.identity_bank import IdentityBank, build_identity_bank
from ttt_svcbench_qwen.inference import (
    AnswerInputs,
    CausalChunk,
    InferenceRequest,
    OnlineTTTUpdater,
    PerVideoRuntimeManager,
    QueryAttempt,
    TTTUpdateOutcome,
    _inference_output,
    assert_inference_runtime_payload,
    run_inference,
)
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    BatchRuntimeState,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkOutput,
    ObservationChunkRequest,
    QwenGenerateOutput,
    QwenGenerateRequest,
    QwenPrefillRequest,
    StateTTTModel,
    TrajectoryRuntimeState,
    VisualStageOutput,
)
from ttt_svcbench_qwen.observation_heads import (
    E1SoftOutput,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
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
        hot_cache_enabled=False,
    )


def _reader_result(status: ReaderStatus) -> ReaderResult:
    """Reader result carrying the exact_count / number-token payload for each status."""

    count_bearing = status in (ReaderStatus.OK, ReaderStatus.EMPTY)
    selected_ids = ("record-0",) if status is ReaderStatus.OK else ()
    exact_count = 2 if status is ReaderStatus.OK else 0 if status is ReaderStatus.EMPTY else None
    operator = Operator.O1_SNAP if status is not ReaderStatus.UNSUPPORTED else Operator.UNSUPPORTED
    return ReaderResult(
        status=status,
        exact_count=exact_count,
        number_token_ids=() if exact_count is None else (48 + exact_count,),
        selected_record_ids=selected_ids,
        operator=operator,
        time_window=TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, count_bearing),
        audit_fields=(
            ("retrieval_status", status.value),
            ("n_state", len(selected_ids)),
            ("number_text", str(exact_count)),
        ),
    )


class _FakeSuite:
    def __init__(self, status: ReaderStatus = ReaderStatus.OK) -> None:
        self.status = status
        self.fast_adapter: FastTTTAdapter | None = None
        self.fast_versions: list[int] = []
        self.answer_fast_versions: list[tuple[int, ...]] = []
        self.seen_frames: list[tuple[object, ...]] = []
        self.prefill_calls = 0
        self.generate_calls = 0
        self.raise_in_generate = False

    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        batch = cast(BatchRuntimeState, request.runtime_state)
        runtime = batch.rows[0]
        assert runtime.fast_weights is not None
        chunk = cast(CausalChunk, request.video_input)
        self.fast_versions.append(runtime.fast_weights.write_version)
        self.seen_frames.append(chunk.frames)
        value = (chunk.frames, runtime.fast_weights.write_version)
        return VisualStageOutput(value=value)

    @staticmethod
    def query(_query_input: object, *, inference: bool) -> object:
        assert inference
        return SimpleNamespace(
            q_target=torch.zeros((1, 512)),
            hard_operators=(Operator.O1_SNAP,),
        )

    def fast(
        self,
        visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        if self.fast_adapter is None:
            raise RuntimeError("test suite Fast Adapter was not installed")
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

    def temporal(
        self,
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

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        result = _reader_result(self.status)
        return SimpleNamespace(
            selected_record_ids=(result.selected_record_ids,),
            status=(self.status.value,),
            audit=("retrieval", self.status.value),
        )

    def read_bank(
        self,
        _state_bank: object,
        _states: object,
        _query: object,
        *,
        video_ids: object,
        trajectory_ids: object,
    ) -> tuple[ReaderResult, ...]:
        assert tuple(cast(Sequence[str], video_ids)) == ("video-a",)
        assert tuple(cast(Sequence[str], trajectory_ids)) == ("trajectory-a",)
        return (_reader_result(self.status),)

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
            signature=(
                request.pixel_values_videos.shape[0],
                tuple(int(value) for value in request.video_grid_thw[0].tolist()),
            ),
        )

    def generate(self, request: QwenGenerateRequest) -> QwenGenerateOutput:
        self.generate_calls += 1
        if self.raise_in_generate:
            raise RuntimeError("synthetic generation failure")
        if self.fast_adapter is None or self.fast_adapter._active_fast_states is None:
            raise RuntimeError("tiny Answer generation requires one managed fast-state binding")
        active = self.fast_adapter._active_fast_states
        self.answer_fast_versions.append(tuple(state.write_version for state in active))
        dtype = self.fast_adapter.w0_1.dtype
        device = self.fast_adapter.w0_1.device
        self.fast_adapter(torch.zeros((len(active), 1, 4096), dtype=dtype, device=device))
        signature = (
            request.prefill.pixel_values_videos.shape[0],
            tuple(int(value) for value in request.prefill.video_grid_thw[0].tolist()),
        )
        return QwenGenerateOutput(
            f"answer:{signature!r}",
            torch.tensor([[1]], dtype=torch.int64),
        )


def _typed_query() -> QueryEncoderOutput:
    """Build a confidently routed O1 query whose window is a valid 0..2s history span.

    Built locally rather than through ``tests.support.runtime_factories`` because that
    factory still passes the deleted ``TimeResolution``/``QueryEmbeddingOutput`` fields.
    """

    operators = (Operator.O1_SNAP,)
    q_target = torch.zeros((1, 512))
    raw_indices = torch.tensor([tuple(Operator).index(operators[0])], dtype=torch.int64)
    logits = torch.full((1, len(tuple(Operator))), -5.0)
    logits[0, raw_indices[0]] = 5.0
    route = OperatorRouterOutput(
        logits=logits,
        confidence=torch.ones(1),
        raw_indices=raw_indices,
        hard_operators=operators,
        head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in operators),
    )
    time_logits = TimeResolverLogits(
        mode_logits=torch.zeros((1, 4)),
        mode_confidence=torch.ones(1),
        mode_indices=torch.ones(1, dtype=torch.int64),
        span_start_logits=torch.zeros((1, 1)),
        span_end_logits=torch.zeros((1, 1)),
        padding_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    resolutions = (
        TimeResolution(
            window=TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True),
            status=TimeResolutionStatus.OK,
        ),
    )
    return QueryEncoderOutput(
        embeddings=QueryEmbeddingOutput(
            token_states=torch.zeros((1, 1, 768)),
            q_target=q_target,
            q_operator=q_target.clone(),
            q_time=q_target.clone(),
            padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        ),
        route=route,
        time=TimeResolverOutput(time_logits, resolutions),
        hard_operators=operators,
        head_types=route.head_types,
    )


def _typed_cache(hidden: Tensor, query: Tensor) -> TemporalCache:
    return make_temporal_cache(
        hidden=hidden,
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
        query_signatures=query,
    )


def _typed_spatial() -> SpatialEncoderOutput:
    return make_spatial_output(
        torch.randn((1, 1, 768)), video_ids=("video-a",), processed_tubelets=2
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
    e1_state = make_e1_state(
        query_signature=query.q_target[0].detach().clone(),
        total_seen=2,
        timestamps=temporal.timestamps[0].clone(),
        position_ids=temporal.position_ids[0].clone(),
    )
    e2_state = make_e2_state(
        query_signature=query.q_target[0].detach().clone(),
        total_seen=2,
        timestamps=temporal.timestamps[0].clone(),
        position_ids=temporal.position_ids[0].clone(),
    )
    e1_logits = torch.full((1, 2, 3), 5.0)
    e1 = E1SoftOutput(
        logits=e1_logits,
        probabilities=torch.sigmoid(e1_logits),
        valid_mask=temporal.valid_mask.clone(),
        timestamps=temporal.timestamps.clone(),
        position_ids=temporal.position_ids.clone(),
        next_states=(e1_state,),
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
    )
    return ObservationOutputs(o1=o1, o2=o2, e1=e1, e2=e2)


class _TypedStageSuite(_FakeSuite):
    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        chunk = cast(CausalChunk, request.video_input)
        self.seen_frames.append(chunk.frames)
        value = (chunk.frames, "stage-a-runtime")
        return VisualStageOutput(value=value)

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

    def temporal(
        self,
        _visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput:
        output = _typed_temporal(cast(QueryEncoderOutput, query))
        runtime = cast(BatchRuntimeState, request.runtime_state).rows[0]
        assert runtime.fast_weights is not None
        # Record which M_t the chunk observed and make the output depend on it, so the
        # recurrence order (chunk t reads M_(t-1)) is observable from the stage itself.
        self.fast_versions.append(runtime.fast_weights.write_version)
        fast_dependency = 1000.0 * runtime.fast_weights.m[0, 0]
        return replace(output, hidden=output.hidden + fast_dependency)

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


class _FakeFastStage:
    """Expose the real adapter's associative protocol around the synthetic stage."""

    def __init__(self, suite: _FakeSuite) -> None:
        self.suite = suite

    def _adapter(self) -> FastTTTAdapter:
        adapter = self.suite.fast_adapter
        if adapter is None:
            raise RuntimeError("test suite Fast Adapter was not installed")
        return adapter

    def use_associative_context(
        self,
        context: FastAssociativeContext,
    ) -> AbstractContextManager[FastTTTAdapter]:
        return self._adapter().use_associative_context(context)

    def consume_associative_intermediates(self) -> AssociativeTTTIntermediates | None:
        return self._adapter().consume_associative_intermediates()

    def __call__(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        return self.suite.fast(visual, query, request)


def _model(dependencies: _Dependencies, suite: _FakeSuite) -> StateTTTModel:
    suite.fast_adapter = dependencies.fast_adapter
    return build_model(
        dependencies.config,
        components=ModelComponents(
            visual_stage=suite.visual,
            query_encoder=suite.query,
            composer=suite.compose,
            qwen_prefill=suite.prefill,
            qwen_generate=suite.generate,
            fast_adapter=_FakeFastStage(suite),
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
            qwen_generate=suite.generate,
            fast_adapter=_FakeFastStage(suite),
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
        *,
        current_end_time: float,
    ) -> TTTUpdateOutcome:
        assert current_end_time >= 0.0
        call = self.calls
        self.calls += 1
        fast = runtime.fast_weights
        assert fast is not None
        if call in self.skip_calls:
            return TTTUpdateOutcome(
                runtime_state=replace(
                    runtime,
                    fast_weights=replace(fast, skip_count=fast.skip_count + 1),
                ),
                did_update=False,
                skip_reason="no_valid_slot",
                valid_token_count=0,
            )
        with torch.no_grad():
            next_memory = (fast.m + 1.0e-4).detach().clone().requires_grad_(True)
        next_fast = replace(
            fast,
            m=next_memory,
            write_version=fast.write_version + 1,
            write_count=fast.write_count + 1,
        )
        return TTTUpdateOutcome(
            runtime_state=replace(runtime, fast_weights=next_fast),
            did_update=True,
            skip_reason=None,
            valid_token_count=1,
        )


def _answer_inputs() -> AnswerInputs:
    return AnswerInputs(
        base_input_ids=torch.ones((1, 2), dtype=torch.int64),
        base_attention_mask=torch.ones((1, 2), dtype=torch.bool),
        pixel_values_videos=torch.ones((8, 4)),
        video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
        tokenizer="tokenizer",
        embedding_owner="embedding",
        rope_indexer="rope",
    )


def _query_input() -> RuntimeQueryInput:
    return RuntimeQueryInput(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_id="query-a",
        query_index=0,
        video=Path("video-a.mp4"),
        question="How many?",
        query_time=2.0,
        explicit_time_values=(),
    )


def _request(*, future_frame: object = "future") -> InferenceRequest:
    return InferenceRequest(
        query_input=_query_input(),
        query_signature=torch.zeros(512),
        chunks=(
            CausalChunk("chunk-0", ("a", "b"), (0.0, 1.0), (0, 1)),
            CausalChunk("chunk-1", ("c", future_frame), (2.0, 4.0), (2, 3)),
        ),
        answer_inputs=_answer_inputs(),
        attempt=QueryAttempt("query-a"),
        max_new_tokens=16,
    )


def test_reset_isolates_consecutive_videos(dependencies: _Dependencies) -> None:
    """Per-video isolation: a second reset shares no storage and inherits no state.

    This is the mainline replacement for the removed ``tensor_contracts`` runtime
    storage-isolation checks, so the ``data_ptr`` comparison is load bearing.
    """

    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    first_state = manager.active_runtime
    assert first_state is not None and first_state.fast_weights is not None
    first_pointer = first_state.fast_weights.m.untyped_storage().data_ptr()
    first_cache = cast(TemporalCache, first_state.temporal_cache)
    first_signature_pointer = first_cache.query_signatures.untyped_storage().data_ptr()

    manager.reset("video-b", "trajectory-b", torch.ones(512))
    second_state = manager.active_runtime
    assert second_state is not None and second_state.fast_weights is not None
    second_cache = cast(TemporalCache, second_state.temporal_cache)

    assert second_state.owner.video_ids == ("video-b",)
    assert second_state.owner.trajectory_ids == ("trajectory-b",)
    assert second_state.fast_weights.write_version == 0
    assert second_state.fast_weights.write_count == 0
    assert second_state.fast_weights.skip_count == 0
    assert torch.count_nonzero(second_state.fast_weights.m.detach()) == 0
    assert second_cache.hidden.shape[1] == 0
    assert second_state.slot_state is None
    assert second_state.e1_state is None and second_state.e2_state is None
    assert second_state.state_bank.records == ()
    assert second_state.identity_bank.candidates == ()
    assert second_state.identity_bank.confirmed == ()
    assert second_state.identity_bank.video_id == "video-b"
    assert second_state.reader_audit == ()
    assert not second_state.released
    # Storage isolation: neither the memory nor the query signature is shared or reused.
    assert second_state.fast_weights.m.untyped_storage().data_ptr() != first_pointer
    assert second_cache.query_signatures.untyped_storage().data_ptr() != first_signature_pointer
    assert torch.count_nonzero(second_cache.query_signatures) == 512
    manager.release()
    assert manager.active_runtime is None


def test_release_returns_a_released_state_and_is_idempotent(
    dependencies: _Dependencies,
) -> None:
    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    released = manager.release()

    assert released is not None
    assert released.released
    assert released.state_bank.released
    assert released.identity_bank.released
    assert released.slot_state is None
    assert released.e1_state is None and released.e2_state is None
    assert released.reader_audit == ()
    assert manager.active_runtime is None
    assert manager.release() is None
    with pytest.raises(RuntimeError, match="live per-video runtime"):
        manager.answer_query(
            model=_model(dependencies, _FakeSuite()),
            observation=cast(ObservationChunkOutput, None),
            answer_inputs=_answer_inputs(),
            attempt=QueryAttempt("query-a"),
        )


def test_causal_chunks_read_previous_memory_and_publish_next_only_updates(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    result = run_inference(
        manager=manager,
        model=_model(dependencies, suite),
        request=_request(),
        updater=_Updater(skip_calls={1}),
    )
    audits = dict(result.audit_fields)

    # Chunk t observes M_(t-1): chunk-0 sees version 0, chunk-1 sees the version 1
    # published by chunk-0's write and never its own.
    assert suite.fast_versions == [0, 1]
    # The frame at t=4.0 is beyond query_time=2.0 and is cropped before handoff.
    assert suite.seen_frames == [("a", "b"), ("c",)]
    assert suite.prefill_calls == 0
    assert suite.generate_calls == 1
    assert result.answer_text == "answer:(8, (2, 2, 2))"
    assert result.reader_result.status is ReaderStatus.OK
    assert result.selected_record_ids == ("record-0",)
    assert result.chunk_count == 2
    assert result.released
    assert result.runtime_state.released
    assert audits["final_write_version"] == 1
    assert audits["final_write_count"] == 1
    assert audits["final_skip_count"] == 1
    assert audits["prefill_count"] == 1
    assert manager.active_runtime is None


def test_future_only_chunks_are_skipped_without_update_or_state(
    dependencies: _Dependencies,
) -> None:
    chunks = (
        CausalChunk("full", ("a", "b"), (0.0, 1.0), (0, 1)),
        CausalChunk("partial", ("c", "future"), (2.0, 4.0), (2, 3)),
        CausalChunk("future", ("d", "e"), (3.0, 5.0), (4, 5)),
    )
    updater = _Updater()
    suite = _FakeSuite()
    manager = _manager(dependencies)
    result = run_inference(
        manager=manager,
        model=_model(dependencies, suite),
        request=replace(_request(), chunks=chunks),
        updater=updater,
    )

    # The wholly-future chunk never reaches the model and never drives an update.
    assert updater.calls == 2
    assert suite.seen_frames == [("a", "b"), ("c",)]
    assert result.chunk_count == 2
    assert result.released
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
        )
        outputs.append((result.answer_text, suite.seen_frames))

    assert outputs[0] == outputs[1]


def test_query_observation_is_read_only_and_uses_the_current_fast_state(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    request = replace(
        _request(),
        query_observation=CausalChunk(
            "query-full-prefix",
            ("q0", "q1", "q2"),
            (0.0, 1.0, 2.0),
            (0, 1, 2),
        ),
    )
    result = run_inference(
        manager=manager,
        model=_model(dependencies, suite),
        request=request,
        updater=_Updater(skip_calls={1}),
    )

    assert suite.seen_frames == [("a", "b"), ("c",), ("q0", "q1", "q2")]
    # The Query observation reuses M_1 and never publishes M_2.
    assert suite.fast_versions == [0, 1, 1]
    assert result.chunk_count == 2
    assert dict(result.audit_fields)["final_write_version"] == 1


def test_observe_query_readonly_commits_no_bank_or_runtime_state(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    model = _model(dependencies, suite)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    manager.observe_chunk(
        model=model,
        chunk=CausalChunk("chunk-0", ("a", "b"), (0.0, 1.0), (0, 1)),
        query_input=_query_input(),
        query_time=2.0,
        updater=_Updater(),
    )
    before = manager.active_runtime
    assert before is not None

    observation = manager.observe_query_readonly(
        model=model,
        chunk=CausalChunk("query", ("q0", "q1"), (0.0, 1.0), (0, 1)),
        query_input=_query_input(),
        query_time=2.0,
    )

    assert manager.active_runtime is before
    assert observation.runtime_state.rows[0] is before
    assert observation.bank_states == (before.state_bank,)
    manager.release()


def test_generate_does_not_mutate_memory_bank_fsm_or_temporal_state(
    dependencies: _Dependencies,
) -> None:
    suite = _TypedStageSuite()
    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    execution = manager.observe_chunk(
        model=_stage_a_model(dependencies, suite),
        chunk=CausalChunk("chunk-0", ("a", "b"), (0.0, 1.0), (0, 1)),
        query_input=_query_input(),
        query_time=2.0,
        updater=_Updater(),
    )
    assert execution.observation is not None
    before = manager.active_runtime
    assert before is not None and before.fast_weights is not None
    memory_before = before.fast_weights.m.detach().clone()

    result = manager.answer_query(
        model=_stage_a_model(dependencies, suite),
        observation=execution.observation,
        answer_inputs=_answer_inputs(),
        attempt=QueryAttempt("query-a"),
        max_new_tokens=16,
    )

    after = manager.active_runtime
    assert after is not None and after.fast_weights is not None
    assert torch.equal(after.fast_weights.m.detach(), memory_before)
    assert after.fast_weights.write_version == before.fast_weights.write_version
    assert after.fast_weights.write_count == before.fast_weights.write_count
    assert after.fast_weights.skip_count == before.fast_weights.skip_count
    assert after.state_bank is before.state_bank
    assert after.identity_bank is before.identity_bank
    assert after.temporal_cache is before.temporal_cache
    assert after.slot_state is before.slot_state
    assert after.e1_state is before.e1_state and after.e2_state is before.e2_state
    assert after.next_chunk_index == before.next_chunk_index
    # The only permitted mutation is appending the Reader result of this query.
    assert after.reader_audit == (result.reader_result,)
    assert dependencies.fast_adapter._active_fast_states is None
    manager.release()


def test_reader_statuses_each_use_one_prefill_and_the_current_fast_state(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    execution = manager.observe_chunk(
        model=_model(dependencies, suite),
        chunk=CausalChunk("chunk", ("a",), (0.0,), (0,)),
        query_input=_query_input(),
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
        result = manager.answer_query(
            model=_model(dependencies, suite),
            observation=execution.observation,
            answer_inputs=_answer_inputs(),
            attempt=QueryAttempt(f"query-{status.value}"),
            max_new_tokens=16,
        )
        results.append(result)
        assert dict(result.audit_fields)["prefill_count"] == 1, index
        assert dict(result.audit_fields)["query_id"] == f"query-{status.value}"
        assert result.answer_text

    assert tuple(result.reader_result.status for result in results) == statuses
    # Reader exact_count and the number-token payload survive per status.
    assert tuple(result.reader_result.exact_count for result in results) == (2, 0, None, None)
    assert results[0].reader_result.number_token_ids == (50,)
    assert results[2].reader_result.number_token_ids == ()
    runtime = manager.active_runtime
    assert runtime is not None and runtime.fast_weights is not None
    assert suite.answer_fast_versions == [
        (runtime.fast_weights.write_version,),
    ] * len(statuses)
    assert suite.fast_adapter is not None
    assert suite.fast_adapter._active_fast_states is None
    manager.release()


def test_generation_exception_releases_the_runtime_and_unbinds_fast_state(
    dependencies: _Dependencies,
) -> None:
    suite = _FakeSuite()
    suite.raise_in_generate = True
    manager = _manager(dependencies)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        run_inference(
            manager=manager,
            model=_model(dependencies, suite),
            request=_request(),
            updater=_Updater(skip_calls={1}),
        )

    assert manager.active_runtime is None
    assert suite.fast_adapter is not None
    assert suite.fast_adapter._active_fast_states is None
    # The manager is reusable after the exception path released the runtime.
    manager.reset("video-b", "trajectory-b", torch.zeros(512))
    assert manager.active_runtime is not None
    manager.release()


def test_no_causal_frame_before_query_time_fails_and_releases(
    dependencies: _Dependencies,
) -> None:
    manager = _manager(dependencies)
    request = replace(
        _request(),
        chunks=(CausalChunk("future", ("a", "b"), (5.0, 6.0), (0, 1)),),
    )
    with pytest.raises(RuntimeError, match="no causal frame"):
        run_inference(
            manager=manager,
            model=_model(dependencies, _FakeSuite()),
            request=request,
            updater=_Updater(),
        )
    assert manager.active_runtime is None


def test_unified_runtime_commits_real_hard_state(dependencies: _Dependencies) -> None:
    suite = _TypedStageSuite()
    manager = _manager(dependencies)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))
    execution = manager.observe_chunk(
        model=_stage_a_model(dependencies, suite),
        chunk=CausalChunk("chunk", ("a", "b"), (0.0, 1.0), (0, 1)),
        query_input=_query_input(),
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
    manager.release()


def test_online_updater_publishes_next_only_fast_state(
    dependencies: _Dependencies,
) -> None:
    torch.manual_seed(15)
    suite = _TypedStageSuite()
    manager = _manager(dependencies)
    model = _stage_a_model(dependencies, suite)
    updater = OnlineTTTUpdater(dependencies.config, dependencies.fast_adapter)
    manager.reset("video-a", "trajectory-a", torch.zeros(512))

    first = manager.observe_chunk(
        model=model,
        chunk=CausalChunk("chunk-0", ("a", "b"), (0.0, 1.0), (0, 1)),
        query_input=_query_input(),
        query_time=3.0,
        updater=updater,
    )
    first_fast = first.runtime_state.fast_weights
    assert first_fast is not None
    assert first_fast.write_version == 1
    assert first_fast.write_count == 1
    assert torch.count_nonzero(first_fast.m.detach()) > 0

    second = manager.observe_chunk(
        model=model,
        chunk=CausalChunk("chunk-1", ("c", "d"), (2.0, 3.0), (2, 3)),
        query_input=_query_input(),
        query_time=3.0,
        updater=updater,
    )
    second_fast = second.runtime_state.fast_weights
    assert second_fast is not None
    assert second_fast.write_version == 2
    assert second_fast.write_count == 2
    # Each chunk observed the memory published before it, never its own write.
    assert suite.fast_versions == [0, 1]
    manager.release()


def test_inference_json_output_contract(dependencies: _Dependencies) -> None:
    """Exercise the same fixed serializer used by the production CLI."""

    suite = _FakeSuite()
    result = run_inference(
        manager=_manager(dependencies),
        model=_model(dependencies, suite),
        request=_request(),
        updater=_Updater(skip_calls={1}),
    )
    audits = dict(result.audit_fields)
    query = _query_input()
    materialized = SimpleNamespace(
        frames=torch.zeros((2, 3, 4, 4)),
        frame_timestamps=torch.tensor((0.0, 2.0), dtype=torch.float64),
        pixel_values_videos=torch.zeros((8, 4)),
        video_grid_thw=torch.tensor(((2, 2, 2),), dtype=torch.int64),
    )
    output = _inference_output(
        query=query,
        result=result,
        audit_level="boundary",
        state_materialized=materialized,
        answer_materialized=materialized,
    )
    text = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    assert text.endswith("\n")
    assert json.loads(text) == output
    assert output["write_version"] == audits["final_write_version"]
    assert output["write_count"] == audits["final_write_count"]
    assert output["skip_count"] == audits["final_skip_count"]
    output_audit = cast(dict[str, object], output["audit"])
    assert output_audit["prefill_count"] == audits["prefill_count"]
    assert output_audit["state_query_frame_count"] == 2
    assert output_audit["answer_query_visual_token_count"] == 8
    assert audits["video_id"] == "video-a"
    assert audits["trajectory_id"] == "trajectory-a"
    assert audits["query_id"] == "query-a"
    assert audits["reader_status"] == ReaderStatus.OK.value
    assert audits["selected_record_count"] == 1
    assert audits["released"] is True


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
