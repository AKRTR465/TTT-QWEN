from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    RetrievalHistoryView,
    StateBankRuntimeState,
    StructuredStateBank,
    build_state_bank,
)
from ttt_svcbench_qwen.state_retriever import (
    EmbeddingStateRetriever,
    RetrievalReason,
    RetrievalStatus,
    build_state_retriever,
)

SEMANTIC_DIM = 512
SOURCE_DIM = 768


@pytest.fixture
def components() -> tuple[StructuredStateBank, EmbeddingStateRetriever]:
    config = load_config()
    return build_state_bank(config), build_state_retriever(config)


def _resolution(
    *,
    query_time: float = 10.0,
    start_time: float | None = 0.0,
    status: TimeResolutionStatus = TimeResolutionStatus.OK,
) -> TimeResolution:
    return TimeResolution(
        window=TimeWindow(
            mode=TimeWindowMode.HISTORY,
            query_time=query_time,
            start_time=start_time,
            end_time=query_time,
            valid=status is TimeResolutionStatus.OK,
        ),
        status=status,
        reason=f"synthetic-{status.value}",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
    )


def _append_history(
    bank: StructuredStateBank,
    state: StateBankRuntimeState,
    *,
    source: Tensor,
    timestamp: float,
    operator: Operator = Operator.O1_SNAP,
    valid: bool = True,
    eligible: bool = True,
) -> StateBankRuntimeState:
    head = OPERATOR_TO_HEAD_TYPE[operator]
    if head is None:
        raise ValueError("history helper requires a supported operator")
    return bank.append_retrieval_history(
        state,
        head_type=head,
        operator=operator,
        semantic_source=source,
        timestamp=timestamp,
        time_range=None,
        valid=valid,
        retrieval_eligible=eligible,
    )


def _view(
    bank: StructuredStateBank,
    states: tuple[StateBankRuntimeState, ...],
    operators: tuple[Operator, ...],
) -> RetrievalHistoryView:
    heads = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in operators)
    return bank.retrieval_view(states, heads)


def _query(
    q_target: Tensor,
    operators: tuple[Operator, ...],
    resolutions: tuple[TimeResolution, ...] | None = None,
) -> object:
    resolved = resolutions or tuple(_resolution() for _ in operators)
    return SimpleNamespace(
        q_target=q_target,
        hard_operators=operators,
        time=SimpleNamespace(resolutions=resolved),
    )


def _retrieve(
    retriever: EmbeddingStateRetriever,
    bank: StructuredStateBank,
    view: RetrievalHistoryView,
    query: object,
    *,
    video_ids: tuple[str, ...] | None = None,
    trajectory_ids: tuple[str, ...] | None = None,
):
    owners = view.video_ids if video_ids is None else video_ids
    trajectories = view.trajectory_ids if trajectory_ids is None else trajectory_ids
    return retriever(
        bank,
        view,
        query,  # type: ignore[arg-type]
        video_ids=owners,
        trajectory_ids=trajectories,
    )


def _matching_query(bank: StructuredStateBank, view: RetrievalHistoryView) -> Tensor:
    head = view.head_types[0][0]
    if head is None:
        raise ValueError("matching query requires one present history record")
    return bank.project(view.sources[0, 0].unsqueeze(0), (head,)).detach().clone()


def test_retriever_consumes_write_before_history_snapshot_only(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-a", "trajectory-a")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    before_write = _view(bank, (state,), (Operator.O1_SNAP,))
    query = _query(_matching_query(bank, before_write), (Operator.O1_SNAP,))

    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=2.0)
    stale_output = _retrieve(retriever, bank, before_write, query)
    after_write = _view(bank, (state,), (Operator.O1_SNAP,))
    current_output = _retrieve(retriever, bank, after_write, query)

    assert stale_output.n_state.tolist() == [1]
    assert stale_output.bank_versions == before_write.bank_versions
    assert current_output.n_state.tolist() == [2]
    assert before_write.record_ids[0][0] == stale_output.selected_record_ids[0][0]


def test_history_reprojection_preserves_query_and_projector_gradients(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    torch.manual_seed(20260720)
    bank, retriever = components
    support_source = torch.randn(SOURCE_DIM, requires_grad=True)
    state = bank.reset("video-gradient", "trajectory-gradient")
    state = _append_history(bank, state, source=support_source, timestamp=1.0)
    history = _view(bank, (state,), (Operator.O1_SNAP,))
    q_target = torch.randn((1, SEMANTIC_DIM), requires_grad=True)

    output = _retrieve(
        retriever,
        bank,
        history,
        _query(q_target, (Operator.O1_SNAP,)),
    )
    output.scores.sum().backward()

    assert history.sources.grad_fn is None and not history.sources.requires_grad
    assert support_source.grad is None
    assert q_target.grad is not None and float(q_target.grad.abs().sum()) > 0.0
    projector_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in bank.semantic_projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0.0
    assert output.state_embeddings.grad_fn is not None
    assert retriever.state_dict() == {}


def test_history_filters_future_invalid_and_ineligible_records(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-filter", "trajectory-filter")
    sources = tuple(torch.randn(SOURCE_DIM) for _ in range(4))
    state = _append_history(bank, state, source=sources[0], timestamp=1.0)
    state = _append_history(
        bank,
        state,
        source=sources[1],
        timestamp=2.0,
        valid=False,
        eligible=False,
    )
    state = _append_history(bank, state, source=sources[2], timestamp=2.5, eligible=False)
    state = _append_history(bank, state, source=sources[3], timestamp=8.0)
    history = _view(bank, (state,), (Operator.O1_SNAP,))
    query = _query(
        _matching_query(bank, history),
        (Operator.O1_SNAP,),
        (_resolution(query_time=3.0),),
    )

    output = _retrieve(retriever, bank, history, query)

    assert output.causal_mask.tolist() == [[True, True, True, False]]
    assert output.selected_record_ids == (("retrieval-00000000",),)
    assert output.status == (RetrievalStatus.OK,)
    assert output.audit[0].invalid_count == 1
    assert output.audit[0].retrieval_ineligible_count == 1
    assert output.audit[0].future_count == 1


def test_history_retriever_statuses_fail_closed(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-status", "trajectory-status")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    history = _view(bank, (state,), (Operator.O1_SNAP,))
    matching = _matching_query(bank, history)

    owner_mismatch = _retrieve(
        retriever,
        bank,
        history,
        _query(matching, (Operator.O1_SNAP,)),
        video_ids=("wrong-video",),
    )
    unsupported = _retrieve(
        retriever,
        bank,
        _view(bank, (state,), (Operator.UNSUPPORTED,)),
        _query(matching, (Operator.UNSUPPORTED,)),
    )
    invalid_time = _retrieve(
        retriever,
        bank,
        history,
        _query(
            matching,
            (Operator.O1_SNAP,),
            (_resolution(status=TimeResolutionStatus.INVALID),),
        ),
    )

    assert owner_mismatch.status == (RetrievalStatus.INVALID,)
    assert owner_mismatch.reason == (RetrievalReason.OWNER_MISMATCH,)
    assert unsupported.status == (RetrievalStatus.UNSUPPORTED,)
    assert unsupported.reason == (RetrievalReason.UNSUPPORTED_OPERATOR,)
    assert invalid_time.status == (RetrievalStatus.INVALID,)
    assert invalid_time.reason == (RetrievalReason.INVALID_TIME,)


def test_retriever_rejects_aggregate_or_malformed_inputs(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = bank.reset("video-errors", "trajectory-errors")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    history = _view(bank, (state,), (Operator.O1_SNAP,))
    query = _query(_matching_query(bank, history), (Operator.O1_SNAP,))

    with pytest.raises(TypeError, match="RetrievalHistoryView"):
        retriever(
            bank,
            bank.view((state,), (HeadType.O1,)),  # type: ignore[arg-type]
            query,  # type: ignore[arg-type]
            video_ids=history.video_ids,
            trajectory_ids=history.trajectory_ids,
        )
    with pytest.raises(ValueError, match="q_target"):
        _retrieve(
            retriever,
            bank,
            history,
            _query(torch.full((1, SEMANTIC_DIM), float("nan")), (Operator.O1_SNAP,)),
        )


def test_build_retriever_validates_frozen_config() -> None:
    config = load_config()
    retriever = build_state_retriever(config)
    assert tuple(retriever.parameters()) == ()
    assert tuple(retriever.buffers()) == ()
    raw = config.model_dump(mode="json")
    raw["retriever"]["top_k"] = 1
    with pytest.raises(ValueError, match="retriever.top_k"):
        ProjectConfig.model_validate(raw)
    with pytest.raises(ValueError, match="Retriever top_k"):
        build_state_retriever(
            config.model_copy(
                update={"retriever": config.retriever.model_copy(update={"top_k": 1})}
            )
        )
