from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E1Payload,
    E2EventKind,
    E2Payload,
    E2Phase,
    HeadType,
    O1Payload,
    StateBankRuntimeState,
    StatePayload,
    StateRecord,
    StructuredStateBank,
    build_state_bank,
)
from ttt_svcbench_qwen.state_retriever import (
    EmbeddingStateRetriever,
    RetrievalReason,
    RetrievalStatus,
    build_state_retriever,
    evaluate_retrieval_quality,
)

SEMANTIC_DIM = 512
IDENTITY_DIM = 256


@pytest.fixture
def components() -> tuple[StructuredStateBank, EmbeddingStateRetriever]:
    config = load_config()
    return build_state_bank(config), build_state_retriever(config)


def _unit_semantic(index: int = 0) -> Tensor:
    value = torch.zeros(SEMANTIC_DIM, dtype=torch.float32)
    value[index] = 1.0
    return value


def _semantic_with_cosine(cosine: float, *, secondary_index: int = 1) -> Tensor:
    value = torch.zeros(SEMANTIC_DIM, dtype=torch.float32)
    value[0] = cosine
    value[secondary_index] = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return F.normalize(value, dim=0)


def _unit_identity(index: int = 0) -> Tensor:
    value = torch.zeros(IDENTITY_DIM, dtype=torch.float32)
    value[index % IDENTITY_DIM] = 1.0
    return value


def _confirmed(
    sequence: int,
    *,
    first_seen: float,
    last_seen: float | None = None,
    prototype_index: int | None = None,
    record_id: str | None = None,
) -> ConfirmedIdentity:
    return ConfirmedIdentity(
        identity_id=f"identity-{sequence:08d}",
        identity_prototype=_unit_identity(sequence if prototype_index is None else prototype_index),
        first_seen=first_seen,
        last_seen=first_seen if last_seen is None else last_seen,
        observation_count=2,
        semantic_record_id=record_id,
    )


def _candidate(sequence: int, *, first_seen: float) -> CandidateIdentity:
    return CandidateIdentity(
        candidate_id=f"candidate-{sequence:08d}",
        identity_prototype=_unit_identity(sequence),
        observation_count=1,
        ttl_remaining=8,
        confidence=0.9,
        first_seen=first_seen,
        last_seen=first_seen,
    )


def _aggregate_payload(head_type: HeadType) -> StatePayload:
    if head_type is HeadType.O1:
        return O1Payload(0, 0, ())
    if head_type is HeadType.E1:
        return E1Payload(E1EventKind.ACTION, 0, (), 0.0)
    if head_type is HeadType.E2:
        return E2Payload(E2EventKind.PERIODIC, 0, E2Phase.INACTIVE, (), ())
    raise ValueError("O2 records require an explicit identity payload")


def _append_record(
    bank: StructuredStateBank,
    state: StateBankRuntimeState,
    *,
    head_type: HeadType,
    semantic: Tensor,
    timestamp: float | None = 1.0,
    time_range: tuple[float, float] | None = None,
    valid: bool = True,
    confidence: float = 0.9,
    payload: StatePayload | None = None,
) -> StateBankRuntimeState:
    sequence = state.next_record_sequence
    if payload is None:
        if head_type is HeadType.O2:
            first_seen = timestamp if timestamp is not None else time_range[0]  # type: ignore[index]
            last_seen = timestamp if timestamp is not None else time_range[1]  # type: ignore[index]
            payload = _confirmed(sequence, first_seen=first_seen, last_seen=last_seen)
        else:
            payload = _aggregate_payload(head_type)
    return bank.append_record(
        state,
        head_type=head_type,
        semantic_embedding=semantic,
        timestamp=timestamp,
        time_range=time_range,
        valid=valid,
        confidence=confidence,
        payload=payload,
    )


def _resolution(
    *,
    mode: TimeWindowMode = TimeWindowMode.HISTORY,
    query_time: float = 10.0,
    start_time: float | None = 0.0,
    end_time: float | None = None,
    status: TimeResolutionStatus = TimeResolutionStatus.OK,
) -> TimeResolution:
    resolved_end = query_time if end_time is None else end_time
    return TimeResolution(
        window=TimeWindow(
            mode=mode,
            query_time=query_time,
            start_time=start_time,
            end_time=resolved_end,
            valid=status is TimeResolutionStatus.OK,
        ),
        status=status,
        reason=f"synthetic-{status.value}",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
    )


def _resolution_for_operator(operator: Operator) -> TimeResolution:
    if operator is Operator.O1_SNAP:
        return _resolution(mode=TimeWindowMode.NOW, start_time=None)
    if operator in (Operator.O1_DELTA, Operator.O2_GAIN):
        return _resolution(mode=TimeWindowMode.RECENT, start_time=5.0)
    return _resolution()


def _retrieve(
    retriever: EmbeddingStateRetriever,
    bank: StructuredStateBank,
    states: tuple[StateBankRuntimeState, ...],
    q_target: Tensor,
    operators: tuple[Operator, ...],
    resolutions: tuple[TimeResolution, ...] | None = None,
    *,
    video_ids: tuple[str, ...] | None = None,
    trajectory_ids: tuple[str, ...] | None = None,
):
    if resolutions is None:
        resolutions = tuple(_resolution_for_operator(operator) for operator in operators)
    if video_ids is None:
        video_ids = tuple(state.video_id for state in states)
    if trajectory_ids is None:
        trajectory_ids = tuple(state.trajectory_id for state in states)
    return retriever.retrieve_states(
        bank,
        states,
        q_target,
        operators,
        resolutions,
        video_ids=video_ids,
        trajectory_ids=trajectory_ids,
    )


def _many_confirmed_state(
    count: int,
    *,
    video_id: str = "video-many",
    trajectory_id: str = "trajectory-many",
    cosine: float = 0.8,
) -> StateBankRuntimeState:
    records: list[StateRecord] = []
    for index in range(count):
        record_id = f"record-{index:08d}"
        records.append(
            StateRecord(
                record_id=record_id,
                video_id=video_id,
                trajectory_id=trajectory_id,
                head_type=HeadType.O2,
                semantic_embedding=_semantic_with_cosine(
                    cosine,
                    secondary_index=index + 1,
                ),
                timestamp=1.0,
                time_range=None,
                valid=True,
                confidence=0.9,
                payload=_confirmed(
                    index,
                    first_seen=1.0,
                    record_id=record_id,
                ),
            )
        )
    record_ids = tuple(record.record_id for record in records)
    return StateBankRuntimeState(
        video_id=video_id,
        trajectory_id=trajectory_id,
        records=tuple(records),
        audit_log=(),
        issued_record_ids=record_ids,
        next_record_sequence=count,
        version=count,
    )


def test_operator_partitions_head_before_similarity_for_all_routes(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-routes", "trajectory-routes")
    for head_type in (HeadType.O1, HeadType.O2, HeadType.E1, HeadType.E2):
        state = _append_record(
            bank,
            state,
            head_type=head_type,
            semantic=_unit_semantic(),
            timestamp=6.0,
        )

    expected_heads = {
        operator: head_type
        for operator, head_type in OPERATOR_TO_HEAD_TYPE.items()
        if head_type is not None
    }
    for operator, expected_head in expected_heads.items():
        output = _retrieve(
            retriever,
            bank,
            (state,),
            _unit_semantic().unsqueeze(0),
            (operator,),
        )
        assert output.status == (RetrievalStatus.OK,)
        assert output.n_state.tolist() == [1]
        assert output.n_retrieved.tolist() == [1]
        assert output.selected_records[0][0].head_type is expected_head
        assert output.audit[0].head_partition_excluded_count == 3

    unsupported = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.UNSUPPORTED,),
    )
    assert unsupported.status == (RetrievalStatus.UNSUPPORTED,)
    assert unsupported.reason == (RetrievalReason.UNSUPPORTED_OPERATOR,)
    assert unsupported.n_state.tolist() == [0]
    assert unsupported.n_retrieved.tolist() == [0]
    assert unsupported.audit[0].head_partition_excluded_count == 4


def test_query_history_reprojects_detached_sources_and_backpropagates_to_projector() -> None:
    torch.manual_seed(20260720)
    config = load_config()
    bank = build_state_bank(config)
    retriever = build_state_retriever(config)
    state = bank.reset("video-gradient", "trajectory-gradient")
    support_sources = (
        torch.randn(768, requires_grad=True),
        torch.randn(768, requires_grad=True),
    )
    for index, source in enumerate(support_sources):
        state = bank.append_retrieval_history(
            state,
            head_type=HeadType.O1,
            operator=Operator.O1_SNAP,
            semantic_source=source,
            timestamp=float(index + 1),
            time_range=None,
        )
    q_target = torch.randn((1, SEMANTIC_DIM), requires_grad=True)
    query = SimpleNamespace(
        q_target=q_target,
        hard_operators=(Operator.O1_SNAP,),
        time=SimpleNamespace(resolutions=(_resolution(query_time=3.0),)),
    )

    output = retriever.retrieve_query_history(
        bank,
        (state,),
        query,
        video_ids=("video-gradient",),
        trajectory_ids=("trajectory-gradient",),
    )
    candidate_mask = (
        output.present_mask
        & output.record_valid_mask
        & output.retrieval_eligible_mask
        & output.causal_mask
    )
    assert candidate_mask.tolist() == [[True, True]]
    assert output.state_embeddings.grad_fn is not None
    loss = torch.logsumexp(output.scores[0, candidate_mask[0]], dim=0) - output.scores[0, 0]
    assert float(loss.detach().item()) > 0.0
    loss.backward()

    assert q_target.grad is not None and float(q_target.grad.abs().sum().item()) > 0.0
    projector_grads = tuple(
        parameter.grad for parameter in bank.semantic_projector.parameters()
    )
    assert any(
        gradient is not None and float(gradient.abs().sum().item()) > 0.0
        for gradient in projector_grads
    )
    assert all(source.grad is None for source in support_sources)
    assert all(
        not record.semantic_source.requires_grad and record.semantic_source.grad_fn is None
        for record in state.retrieval_history
    )

def test_normalized_cosine_scores_every_record_in_partition(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-scores", "trajectory-scores")
    semantics = (_unit_semantic(), _semantic_with_cosine(0.6), _unit_semantic(1))
    for semantic in semantics:
        state = _append_record(
            bank,
            state,
            head_type=HeadType.O2,
            semantic=semantic,
        )

    query = (2.0 * _unit_semantic()).unsqueeze(0)
    output = _retrieve(retriever, bank, (state,), query, (Operator.O2_UNIQUE,))
    scaled = _retrieve(retriever, bank, (state,), query * 7.0, (Operator.O2_UNIQUE,))

    torch.testing.assert_close(output.scores[0], torch.tensor([1.0, 0.6, 0.0]))
    torch.testing.assert_close(output.scores, scaled.scores)
    assert output.scores.dtype is torch.float32
    assert output.n_state.tolist() == [3]
    assert output.n_retrieved.tolist() == [2]
    assert output.selected_record_ids == ((state.records[0].record_id, state.records[1].record_id),)
    assert output.selected_mask.tolist() == [[True, True, False]]

    bfloat_output = _retrieve(
        retriever,
        bank,
        (state,),
        query.to(torch.bfloat16),
        (Operator.O2_UNIQUE,),
    )
    assert bfloat_output.scores.dtype is torch.float32
    torch.testing.assert_close(bfloat_output.scores, output.scores, atol=2.0e-3, rtol=0.0)


@pytest.mark.parametrize("count", (3, 30, 300))
def test_no_topk_returns_every_hit(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
    count: int,
) -> None:
    bank, retriever = components
    state = _many_confirmed_state(count)
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    expected_ids = tuple(record.record_id for record in state.records)
    assert output.n_state.tolist() == [count]
    assert output.n_retrieved.tolist() == [count]
    assert output.selected_record_ids == (expected_ids,)
    assert output.selected_mask.all()
    assert output.audit[0].below_similarity_count == 0


def test_selected_ids_have_canonical_score_then_id_order(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-order", "trajectory-order")
    for cosine in (0.8, 0.9, 0.9):
        state = _append_record(
            bank,
            state,
            head_type=HeadType.O2,
            semantic=_semantic_with_cosine(cosine),
        )
    expected = (
        state.records[1].record_id,
        state.records[2].record_id,
        state.records[0].record_id,
    )

    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )
    reversed_state = replace(state, records=tuple(reversed(state.records)))
    reversed_output = _retrieve(
        retriever,
        bank,
        (reversed_state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    assert output.selected_record_ids == (expected,)
    assert reversed_output.selected_record_ids == (expected,)
    assert reversed_output.candidate_record_ids[0] == tuple(
        record.record_id for record in reversed_state.records
    )
    assert reversed_output.selected_mask.tolist() == [[True, True, True]]


def test_candidate_and_invalid_records_are_never_retrievable(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-eligibility", "trajectory-eligibility")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        payload=_candidate(0, first_seen=1.0),
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        valid=False,
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
    )
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    assert output.n_state.tolist() == [3]
    assert output.n_retrieved.tolist() == [1]
    assert output.selected_record_ids == ((state.records[2].record_id,),)
    assert output.scores.tolist() == [[1.0, 1.0, 1.0]]
    assert output.selected_mask.tolist() == [[False, False, True]]
    assert output.audit[0].retrieval_ineligible_count == 1
    assert output.audit[0].invalid_count == 1

    candidate_only = bank.reset("video-candidate", "trajectory-candidate")
    candidate_only = _append_record(
        bank,
        candidate_only,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        payload=_candidate(0, first_seen=1.0),
    )
    candidate_output = _retrieve(
        retriever,
        bank,
        (candidate_only,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )
    assert candidate_output.status == (RetrievalStatus.EMPTY,)
    assert candidate_output.reason == (RetrievalReason.ALL_RETRIEVAL_INELIGIBLE,)

    invalid_only = bank.reset("video-invalid", "trajectory-invalid")
    invalid_only = _append_record(
        bank,
        invalid_only,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        valid=False,
    )
    invalid_output = _retrieve(
        retriever,
        bank,
        (invalid_only,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )
    assert invalid_output.status == (RetrievalStatus.EMPTY,)
    assert invalid_output.reason == (RetrievalReason.ALL_INVALID,)


def test_retriever_uses_semantic_embedding_not_identity_prototype(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-semantic", "trajectory-semantic")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(1),
        payload=_confirmed(
            0,
            first_seen=1.0,
            prototype_index=0,
        ),
    )
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    assert output.status == (RetrievalStatus.EMPTY,)
    assert output.reason == (RetrievalReason.BELOW_SIMILARITY,)
    assert output.scores.item() == 0.0
    with pytest.raises(ValueError, match="q_target"):
        _retrieve(
            retriever,
            bank,
            (state,),
            _unit_identity().unsqueeze(0),
            (Operator.O2_UNIQUE,),
        )


@pytest.mark.parametrize(
    ("head_type", "operator"),
    (
        (HeadType.O1, Operator.O1_DELTA),
        (HeadType.E1, Operator.E1_ACTION),
        (HeadType.E2, Operator.E2_PERIODIC),
    ),
)
def test_aggregate_records_defer_window_arithmetic_to_reader(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
    head_type: HeadType,
    operator: Operator,
) -> None:
    bank, retriever = components
    suffix = head_type.value
    state = bank.reset(f"video-aggregate-{suffix}", f"trajectory-aggregate-{suffix}")
    state = _append_record(
        bank,
        state,
        head_type=head_type,
        semantic=_unit_semantic(),
        timestamp=1.0,
    )
    explicit = _resolution(
        mode=TimeWindowMode.EXPLICIT_RANGE,
        start_time=8.0,
        end_time=9.0,
    )
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (operator,),
        (explicit,),
    )
    assert output.status == (RetrievalStatus.OK,)
    assert output.audit[0].outside_window_count == 0

    future = bank.reset(f"video-future-{suffix}", f"trajectory-future-{suffix}")
    future = _append_record(
        bank,
        future,
        head_type=head_type,
        semantic=_unit_semantic(),
        timestamp=10.1,
    )
    future_output = _retrieve(
        retriever,
        bank,
        (future,),
        _unit_semantic().unsqueeze(0),
        (operator,),
        (explicit,),
    )
    assert future_output.status == (RetrievalStatus.EMPTY,)
    assert future_output.reason == (RetrievalReason.ALL_FUTURE,)
    assert future_output.audit[0].future_count == 1


def test_atomic_o2_records_use_closed_window_intersection(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    point_state = bank.reset("video-points", "trajectory-points")
    for timestamp in (4.999, 5.0, 10.0, 10.001):
        point_state = _append_record(
            bank,
            point_state,
            head_type=HeadType.O2,
            semantic=_unit_semantic(),
            timestamp=timestamp,
        )

    range_state = bank.reset("video-ranges", "trajectory-ranges")
    for time_range in ((0.0, 4.999), (4.0, 5.0), (10.0, 10.0), (9.0, 11.0)):
        range_state = _append_record(
            bank,
            range_state,
            head_type=HeadType.O2,
            semantic=_unit_semantic(),
            timestamp=None,
            time_range=time_range,
        )

    recent = _resolution(mode=TimeWindowMode.RECENT, start_time=5.0)
    output = _retrieve(
        retriever,
        bank,
        (point_state, range_state),
        torch.stack((_unit_semantic(), _unit_semantic())),
        (Operator.O2_GAIN, Operator.O2_GAIN),
        (recent, recent),
    )

    assert output.selected_record_ids == (
        (point_state.records[1].record_id, point_state.records[2].record_id),
        (range_state.records[1].record_id, range_state.records[2].record_id),
    )
    assert output.n_state.tolist() == [4, 4]
    assert output.n_retrieved.tolist() == [2, 2]
    assert [audit.outside_window_count for audit in output.audit] == [1, 1]
    assert [audit.future_count for audit in output.audit] == [1, 1]


def test_similarity_threshold_is_fp32_and_inclusive(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    threshold = torch.tensor(0.35, dtype=torch.float32)
    below = torch.nextafter(threshold, torch.tensor(float("-inf"), dtype=torch.float32))
    state = bank.reset("video-threshold", "trajectory-threshold")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_semantic_with_cosine(float(threshold)),
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_semantic_with_cosine(float(below)),
    )
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    assert output.scores[0, 0].item() == threshold.item()
    assert output.scores[0, 1].item() < threshold.item()
    assert output.selected_record_ids == ((state.records[0].record_id,),)
    assert output.selected_mask.tolist() == [[True, False]]


def test_empty_reasons_and_query_statuses_are_structured(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    query = _unit_semantic().unsqueeze(0)

    empty = bank.reset("video-empty", "trajectory-empty")
    empty_output = _retrieve(
        retriever,
        bank,
        (empty,),
        query,
        (Operator.O2_UNIQUE,),
    )
    assert empty_output.reason == (RetrievalReason.EMPTY_BANK,)

    wrong_head = bank.reset("video-wrong-head", "trajectory-wrong-head")
    wrong_head = _append_record(
        bank,
        wrong_head,
        head_type=HeadType.O1,
        semantic=_unit_semantic(),
    )
    wrong_head_output = _retrieve(
        retriever,
        bank,
        (wrong_head,),
        query,
        (Operator.O2_UNIQUE,),
    )
    assert wrong_head_output.reason == (RetrievalReason.EMPTY_HEAD_PARTITION,)
    assert wrong_head_output.audit[0].head_partition_excluded_count == 1

    below = bank.reset("video-below", "trajectory-below")
    below = _append_record(
        bank,
        below,
        head_type=HeadType.O2,
        semantic=_unit_semantic(1),
    )
    below_output = _retrieve(
        retriever,
        bank,
        (below,),
        query,
        (Operator.O2_UNIQUE,),
    )
    assert below_output.reason == (RetrievalReason.BELOW_SIMILARITY,)

    future = bank.reset("video-all-future", "trajectory-all-future")
    future = _append_record(
        bank,
        future,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        timestamp=11.0,
    )
    future_output = _retrieve(
        retriever,
        bank,
        (future,),
        query,
        (Operator.O2_UNIQUE,),
    )
    assert future_output.reason == (RetrievalReason.ALL_FUTURE,)

    outside = bank.reset("video-outside", "trajectory-outside")
    outside = _append_record(
        bank,
        outside,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        timestamp=4.0,
    )
    outside_output = _retrieve(
        retriever,
        bank,
        (outside,),
        query,
        (Operator.O2_GAIN,),
        (_resolution(mode=TimeWindowMode.RECENT, start_time=5.0),),
    )
    assert outside_output.reason == (RetrievalReason.ALL_OUTSIDE_WINDOW,)

    invalid_resolution = _resolution(status=TimeResolutionStatus.INVALID)
    invalid_output = _retrieve(
        retriever,
        bank,
        (below,),
        query,
        (Operator.O2_UNIQUE,),
        (invalid_resolution,),
    )
    assert invalid_output.status == (RetrievalStatus.INVALID,)
    assert invalid_output.reason == (RetrievalReason.INVALID_TIME,)
    assert invalid_output.audit[0].query_rejected_count == 1

    unsupported_resolution = _resolution(status=TimeResolutionStatus.UNSUPPORTED)
    unsupported_output = _retrieve(
        retriever,
        bank,
        (below,),
        query,
        (Operator.O2_UNIQUE,),
        (unsupported_resolution,),
    )
    assert unsupported_output.status == (RetrievalStatus.UNSUPPORTED,)
    assert unsupported_output.reason == (RetrievalReason.UNSUPPORTED_TIME,)

    zero_output = _retrieve(
        retriever,
        bank,
        (below,),
        torch.zeros(1, SEMANTIC_DIM),
        (Operator.O2_UNIQUE,),
    )
    assert zero_output.status == (RetrievalStatus.UNSUPPORTED,)
    assert zero_output.reason == (RetrievalReason.DEGENERATE_QUERY,)


def test_mixed_mutually_exclusive_filters_return_no_match(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-mixed", "trajectory-mixed")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        valid=False,
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        payload=_candidate(1, first_seen=1.0),
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        timestamp=11.0,
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
        timestamp=4.0,
    )
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(1),
        timestamp=6.0,
    )

    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_GAIN,),
        (_resolution(mode=TimeWindowMode.RECENT, start_time=5.0),),
    )

    assert output.status == (RetrievalStatus.EMPTY,)
    assert output.reason == (RetrievalReason.NO_MATCH,)
    audit = output.audit[0]
    assert (
        audit.invalid_count,
        audit.retrieval_ineligible_count,
        audit.future_count,
        audit.outside_window_count,
        audit.below_similarity_count,
        audit.selected_count,
    ) == (1, 1, 1, 1, 1, 0)


def test_ragged_batch_dynamic_ns_and_owner_isolation(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    empty = bank.reset("video-a", "trajectory-a")
    one = bank.reset("video-a", "trajectory-b")
    one = _append_record(
        bank,
        one,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
    )
    three = bank.reset("video-b", "trajectory-a")
    for index in (1, 2, 3):
        three = _append_record(
            bank,
            three,
            head_type=HeadType.O2,
            semantic=_unit_semantic(index),
        )

    states = (empty, one, three)
    output = _retrieve(
        retriever,
        bank,
        states,
        torch.stack((_unit_semantic(), _unit_semantic(), _unit_semantic())),
        (Operator.O2_UNIQUE,) * 3,
    )

    assert output.scores.shape == (3, 3)
    assert output.n_state.tolist() == [0, 1, 3]
    assert output.n_retrieved.tolist() == [0, 1, 0]
    assert output.present_mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, True],
    ]
    assert output.selected_mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [False, False, False],
    ]
    assert output.status == (
        RetrievalStatus.EMPTY,
        RetrievalStatus.OK,
        RetrievalStatus.EMPTY,
    )
    assert output.bank_versions == tuple(state.version for state in states)

    owner_mismatch = _retrieve(
        retriever,
        bank,
        (one,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
        video_ids=("video-wrong",),
    )
    assert owner_mismatch.status == (RetrievalStatus.INVALID,)
    assert owner_mismatch.reason == (RetrievalReason.OWNER_MISMATCH,)
    assert owner_mismatch.audit[0].owner_mismatch_count == 1

    with pytest.raises(ValueError, match="query owners must be unique"):
        _retrieve(
            retriever,
            bank,
            (one, three),
            torch.stack((_unit_semantic(), _unit_semantic())),
            (Operator.O2_UNIQUE, Operator.O2_UNIQUE),
            video_ids=("duplicate", "duplicate"),
            trajectory_ids=("owner", "owner"),
        )
    with pytest.raises(ValueError, match="owners must be unique"):
        _retrieve(
            retriever,
            bank,
            (one, one),
            torch.stack((_unit_semantic(), _unit_semantic())),
            (Operator.O2_UNIQUE, Operator.O2_UNIQUE),
        )


def test_retriever_output_contract_rejects_inconsistent_metadata(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = _many_confirmed_state(2, video_id="video-output", trajectory_id="trajectory-output")
    output = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
    )

    with pytest.raises(ValueError, match="n_retrieved"):
        replace(output, n_retrieved=torch.tensor([1], dtype=torch.int64))
    with pytest.raises(ValueError, match="n_state"):
        replace(output, n_state=torch.tensor([3], dtype=torch.int64))
    with pytest.raises(ValueError, match="selected metadata"):
        replace(output, selected_record_ids=((output.selected_record_ids[0][0],),))
    with pytest.raises(ValueError, match="unique and aligned"):
        duplicate = (output.selected_record_ids[0][0],) * 2
        replace(output, selected_record_ids=(duplicate,))
    with pytest.raises(ValueError, match="selected_scores"):
        replace(output, selected_scores=((0.1, 0.1),))
    with pytest.raises(ValueError, match="only Retriever OK"):
        replace(output, status=(RetrievalStatus.EMPTY,))
    with pytest.raises(ValueError, match="candidate IDs"):
        candidate_ids = ((None, output.candidate_record_ids[0][1]),)
        replace(output, candidate_record_ids=candidate_ids)
    with pytest.raises(ValueError, match="n_retrieved"):
        replace(output, selected_mask=torch.zeros_like(output.selected_mask))


def test_invalid_inputs_fail_closed_without_mutating_bank(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-errors", "trajectory-errors")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
    )
    original_semantic = state.records[0].semantic_embedding.clone()
    original_version = state.version
    resolution = (_resolution(),)
    owners = {
        "video_ids": (state.video_id,),
        "trajectory_ids": (state.trajectory_id,),
    }

    bad_queries = (
        torch.tensor(1.0),
        torch.zeros(SEMANTIC_DIM),
        torch.zeros(1, IDENTITY_DIM),
        torch.zeros(1, SEMANTIC_DIM, dtype=torch.int64),
        torch.full((1, SEMANTIC_DIM), float("nan")),
        torch.full((1, SEMANTIC_DIM), float("inf")),
    )
    for bad_query in bad_queries:
        with pytest.raises(ValueError):
            retriever.retrieve_states(
                bank,
                (state,),
                bad_query,
                (Operator.O2_UNIQUE,),
                resolution,
                **owners,
            )

    with pytest.raises(ValueError, match="batch size"):
        retriever.retrieve_states(
            bank,
            (state,),
            torch.zeros(2, SEMANTIC_DIM),
            (Operator.O2_UNIQUE, Operator.O2_UNIQUE),
            (_resolution(), _resolution()),
            **owners,
        )
    with pytest.raises(ValueError, match="hard_operators"):
        retriever.retrieve_states(
            bank,
            (state,),
            _unit_semantic().unsqueeze(0),
            ("o2-unique",),  # type: ignore[arg-type]
            resolution,
            **owners,
        )
    with pytest.raises(ValueError, match="time_resolutions"):
        retriever.retrieve_states(
            bank,
            (state,),
            _unit_semantic().unsqueeze(0),
            (Operator.O2_UNIQUE,),
            (),
            **owners,
        )
    with pytest.raises(ValueError, match="video_ids"):
        retriever.retrieve_states(
            bank,
            (state,),
            _unit_semantic().unsqueeze(0),
            (Operator.O2_UNIQUE,),
            resolution,
            video_ids=(),
            trajectory_ids=(state.trajectory_id,),
        )

    released = bank.release(state)
    with pytest.raises(ValueError, match="released"):
        _retrieve(
            retriever,
            bank,
            (released,),
            _unit_semantic().unsqueeze(0),
            (Operator.O2_UNIQUE,),
        )

    extreme = torch.full(
        (1, SEMANTIC_DIM),
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
    )
    assert bool(torch.isfinite(extreme).all())
    extreme_output = _retrieve(
        retriever,
        bank,
        (state,),
        extreme,
        (Operator.O2_UNIQUE,),
    )
    assert extreme_output.status == (RetrievalStatus.UNSUPPORTED,)
    assert extreme_output.reason == (RetrievalReason.DEGENERATE_QUERY,)
    assert bool(torch.isfinite(extreme_output.scores).all())
    assert torch.count_nonzero(extreme_output.scores) == 0

    invalid_time = _resolution(status=TimeResolutionStatus.INVALID)
    owner_first = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.O2_UNIQUE,),
        (invalid_time,),
        video_ids=("wrong-owner",),
    )
    assert owner_first.status == (RetrievalStatus.INVALID,)
    assert owner_first.reason == (RetrievalReason.OWNER_MISMATCH,)

    unsupported_operator = _retrieve(
        retriever,
        bank,
        (state,),
        _unit_semantic().unsqueeze(0),
        (Operator.UNSUPPORTED,),
        video_ids=("wrong-owner",),
    )
    assert unsupported_operator.status == (RetrievalStatus.INVALID,)
    assert unsupported_operator.reason == (RetrievalReason.OWNER_MISMATCH,)

    assert state.version == original_version
    torch.testing.assert_close(state.records[0].semantic_embedding, original_semantic)


def test_retriever_is_parameter_free_preserves_query_gradient_and_bank(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-grad", "trajectory-grad")
    state = _append_record(
        bank,
        state,
        head_type=HeadType.O2,
        semantic=_unit_semantic(),
    )
    original = state.records[0].semantic_embedding.clone()
    version = state.version
    audit = state.audit_log
    query = torch.zeros(1, SEMANTIC_DIM, requires_grad=True)
    with torch.no_grad():
        query[0, 0] = 1.0
        query[0, 1] = 0.5

    retriever.train()
    train_output = _retrieve(
        retriever,
        bank,
        (state,),
        query,
        (Operator.O2_UNIQUE,),
    )
    train_output.scores.sum().backward()

    assert sum(parameter.numel() for parameter in retriever.parameters()) == 0
    assert tuple(retriever.buffers()) == ()
    assert retriever.state_dict() == {}
    assert query.grad is not None
    assert bool(torch.isfinite(query.grad).all())
    assert float(torch.linalg.vector_norm(query.grad)) > 0.0
    assert not train_output.state_embeddings.requires_grad
    assert all(
        not record.semantic_embedding.requires_grad
        for row in train_output.selected_records
        for record in row
    )
    assert state.version == version
    assert state.audit_log == audit
    torch.testing.assert_close(state.records[0].semantic_embedding, original)

    retriever.eval()
    eval_output = _retrieve(
        retriever,
        bank,
        (state,),
        query.detach(),
        (Operator.O2_UNIQUE,),
    )
    torch.testing.assert_close(eval_output.scores, train_output.scores.detach())
    assert eval_output.selected_record_ids == train_output.selected_record_ids


def test_topk_ann_and_other_contract_drift_are_rejected() -> None:
    config = load_config()
    retriever = build_state_retriever(config)
    assert config.retriever.top_k is None
    assert config.retriever.ann_enabled is False
    assert tuple(retriever.parameters()) == ()

    updates: tuple[dict[str, object], ...] = (
        {"top_k": 1},
        {"ann_enabled": True},
        {"semantic_dim": 256},
        {"selection_order": ("record_id_asc",)},
        {"aggregate_time_policy": "filter_aggregate_by_window"},
    )
    for update in updates:
        bad_retriever = config.retriever.model_copy(update=update)
        bad_config: ProjectConfig = config.model_copy(update={"retriever": bad_retriever})
        with pytest.raises(ValueError, match="Retriever"):
            build_state_retriever(bad_config)


def test_offline_retrieval_quality_metrics_are_label_free_at_runtime() -> None:
    metrics = evaluate_retrieval_quality(
        (("a", "b"), (), (), ()),
        (("b", "c"), ("d",), ("ignored-u",), ("ignored-i",)),
        (
            RetrievalStatus.OK,
            RetrievalStatus.EMPTY,
            RetrievalStatus.UNSUPPORTED,
            RetrievalStatus.INVALID,
        ),
    )
    assert metrics.true_positive_count == 1
    assert metrics.selected_denominator == 2
    assert metrics.relevant_denominator == 5
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0 / 5.0)
    assert metrics.empty_retrieval_count == 1
    assert metrics.query_denominator == 2
    assert metrics.empty_retrieval_rate == pytest.approx(0.5)
    assert metrics.unsupported_count == 1
    assert metrics.invalid_count == 1
    assert metrics.total_query_count == 4
    assert metrics.unsupported_rate == pytest.approx(0.25)
    assert metrics.invalid_rate == pytest.approx(0.25)

    empty = evaluate_retrieval_quality((), (), ())
    assert empty.selected_denominator == 0
    assert empty.relevant_denominator == 0
    assert empty.query_denominator == 0
    assert empty.precision is None
    assert empty.recall is None
    assert empty.empty_retrieval_rate is None
    assert empty.unsupported_rate is None
    assert empty.invalid_rate is None

    all_empty = evaluate_retrieval_quality(
        ((),),
        ((),),
        (RetrievalStatus.EMPTY,),
    )
    assert all_empty.precision is None
    assert all_empty.recall is None
    assert all_empty.empty_retrieval_rate == 1.0

    with pytest.raises(ValueError, match="equal length"):
        evaluate_retrieval_quality((("a",),), (), (RetrievalStatus.OK,))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_retrieval_quality(
            (("a", "a"),),
            (("a",),),
            (RetrievalStatus.OK,),
        )
    with pytest.raises(ValueError, match="only retrieval status OK"):
        evaluate_retrieval_quality(
            (("a",),),
            (("a",),),
            (RetrievalStatus.EMPTY,),
        )
