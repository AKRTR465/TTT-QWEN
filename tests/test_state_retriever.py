from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    OPERATORS,
    Operator,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakSupervision,
    _official_weak_retrieval_loss,
)
from ttt_svcbench_qwen.state_bank import (
    RETRIEVAL_HEAD_ORDER,
    HeadType,
    RetrievalHistoryAppendBatch,
    RetrievalHistoryView,
    StateBankRuntimeState,
    StructuredStateBank,
    TensorizedRetrievalHistory,
    build_state_bank,
    tensorized_retrieval_view,
)
from ttt_svcbench_qwen.state_retriever import (
    EmbeddingStateRetriever,
    RetrievalReason,
    RetrievalStatus,
    _project_history_sources,
    build_state_retriever,
)

SEMANTIC_DIM = 512
SOURCE_DIM = 768


@dataclass
class _HistoryState:
    bank_state: StateBankRuntimeState
    ring: TensorizedRetrievalHistory


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
    state: _HistoryState,
    *,
    source: Tensor,
    timestamp: float,
    operator: Operator = Operator.O1_SNAP,
    valid: bool = True,
    eligible: bool = True,
) -> _HistoryState:
    head = OPERATOR_TO_HEAD_TYPE[operator]
    if head is None:
        raise ValueError("history helper requires a supported operator")
    del bank
    state.ring.append_many(
        RetrievalHistoryAppendBatch(
            sources=source.reshape(1, -1),
            head_codes=torch.tensor((RETRIEVAL_HEAD_ORDER.index(head),), dtype=torch.int64),
            operator_codes=torch.tensor((OPERATORS.index(operator),), dtype=torch.int64),
            timestamps=torch.tensor((timestamp,), dtype=torch.float64),
            time_ranges=torch.full((1, 2), -1.0, dtype=torch.float64),
            valid_mask=torch.tensor((valid,), dtype=torch.bool),
            eligible_mask=torch.tensor((eligible,), dtype=torch.bool),
        )
    )
    return state


def _new_history(bank: StructuredStateBank, video_id: str, trajectory_id: str) -> _HistoryState:
    parameter = next(bank.semantic_projector.parameters())
    return _HistoryState(
        bank_state=bank.reset(video_id, trajectory_id),
        ring=TensorizedRetrievalHistory(
            video_id,
            trajectory_id,
            capacity_per_head=bank.config.retrieval_history_capacity_per_head,
            source_dim=SOURCE_DIM,
            dtype=parameter.dtype,
            device=parameter.device,
        ),
    )


def _view(
    states: tuple[_HistoryState, ...],
) -> RetrievalHistoryView:
    return tensorized_retrieval_view(
        tuple(state.ring for state in states),
        guard_current_version=False,
    )


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
    _, head_codes, _ = view.require_tensor_metadata()
    head_code = int(head_codes[0, 0].item())
    if head_code < 0:
        raise ValueError("matching query requires one present history record")
    head = RETRIEVAL_HEAD_ORDER[head_code]
    return bank.project(view.sources[0, 0].unsqueeze(0), (head,)).detach().clone()


def test_retriever_consumes_write_before_history_snapshot_only(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = _new_history(bank, "video-a", "trajectory-a")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    before_write = _view((state,))
    query = _query(_matching_query(bank, before_write), (Operator.O1_SNAP,))

    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=2.0)
    stale_output = _retrieve(retriever, bank, before_write, query)
    after_write = _view((state,))
    current_output = _retrieve(retriever, bank, after_write, query)

    assert stale_output.n_state.tolist() == [1]
    assert stale_output.bank_versions == before_write.bank_versions
    assert current_output.n_state.tolist() == [2]
    assert stale_output.selected_record_ids == (("retrieval-00000000",),)


def test_all_head_scoring_preserves_predicted_head_runtime_selection(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = _new_history(bank, "video-all-head", "trajectory-all-head")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    state = _append_history(
        bank,
        state,
        source=torch.randn(SOURCE_DIM),
        timestamp=2.0,
        operator=Operator.O2_UNIQUE,
    )
    history = _view((state,))
    query = _query(_matching_query(bank, history), (Operator.O1_SNAP,))
    output = _retrieve(retriever, bank, history, query)

    assert output.n_state.tolist() == [2]
    assert output.present_mask.tolist() == [[True, True]]
    assert output.predicted_head_mask.tolist() == [[True, False]]
    assert output.audit[0].head_partition_excluded_count == 1
    assert all(record.head_type is HeadType.O1 for record in output.selected_records[0])
    chunked = _project_history_sources(bank, history, chunk_size=1)
    single = _project_history_sources(bank, history, chunk_size=256)
    assert torch.allclose(chunked, single, atol=1.0e-6, rtol=1.0e-6)


def test_history_reprojection_preserves_query_and_projector_gradients(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    torch.manual_seed(20260720)
    bank, retriever = components
    support_source = torch.randn(SOURCE_DIM, requires_grad=True)
    state = _new_history(bank, "video-gradient", "trajectory-gradient")
    state = _append_history(bank, state, source=support_source, timestamp=1.0)
    history = _view((state,))
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


def test_label_safe_wrong_route_bag_updates_semantic_projector(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    torch.manual_seed(20260722)
    bank, retriever = components
    state = _new_history(bank, "video-rescue", "trajectory-rescue")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=2.0)
    history = _view((state,))
    output = _retrieve(
        retriever,
        bank,
        history,
        _query(torch.randn(1, SEMANTIC_DIM), (Operator.O2_UNIQUE,)),
    )
    assert output.selected_record_ids == ((),)
    result = _official_weak_retrieval_loss(
        output,
        0,
        OfficialWeakSupervision(
            query_id="projector-rescue",
            operator=Operator.O1_SNAP,
            time_mode=TimeWindowMode.HISTORY,
            count=1,
            query_time=3.0,
            occurrence_points=(1.0,),
            occurrence_intervals=(),
        ),
    )
    assert result.status == "valid_bag"
    assert result.rescued_wrong_route
    assert result.loss is not None
    result.loss.backward()
    projector_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in bank.semantic_projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0.0


def test_tensor_ring_overflow_retrieval_loss_and_gradient() -> None:
    torch.manual_seed(20260723)
    config = load_config()
    bank = build_state_bank(config)
    object.__setattr__(bank.config, "retrieval_history_capacity_per_head", 2)
    retriever = build_state_retriever(config)
    history = _new_history(bank, "video-ring", "trajectory-ring")
    writes = (
        (Operator.O1_SNAP, 1.0),
        (Operator.O1_SNAP, 2.0),
        (Operator.O2_UNIQUE, 2.5),
        (Operator.O1_SNAP, 3.0),
    )
    for operator, timestamp in writes:
        source = torch.randn(SOURCE_DIM)
        _append_history(bank, history, source=source, timestamp=timestamp, operator=operator)

    ring_view = tensorized_retrieval_view((history.ring,))
    assert ring_view.n_state.tolist() == [3]
    assert ring_view.sequence_ids[0, :3].tolist() == [1, 2, 3]

    _, head_codes, _ = ring_view.require_tensor_metadata()
    head = RETRIEVAL_HEAD_ORDER[int(head_codes[0, 0].item())]
    q_target = bank.project(ring_view.sources[0, 0].unsqueeze(0), (head,)).detach()
    query = _query(q_target, (Operator.O1_SNAP,), (_resolution(query_time=4.0),))
    ring_output = _retrieve(retriever, bank, ring_view, query)
    assert ring_output.n_state.tolist() == [3]

    label = OfficialWeakSupervision(
        query_id="ring-equivalence",
        operator=Operator.O1_SNAP,
        time_mode=TimeWindowMode.HISTORY,
        count=1,
        query_time=4.0,
        occurrence_points=(2.0,),
        occurrence_intervals=(),
    )
    ring_loss = _official_weak_retrieval_loss(ring_output, 0, label)
    assert ring_loss.status == "valid_bag"
    assert ring_loss.loss is not None
    bank.zero_grad(set_to_none=True)
    ring_loss.loss.backward()
    ring_gradient = torch.cat(
        tuple(
            parameter.grad.detach().flatten()
            for parameter in bank.semantic_projector.parameters()
            if parameter.grad is not None
        )
    )
    assert torch.isfinite(ring_gradient).all() and float(ring_gradient.norm()) > 0.0


def test_tensor_ring_snapshot_is_immutable_and_release_is_terminal() -> None:
    ring = TensorizedRetrievalHistory(
        "video-snapshot",
        "trajectory-snapshot",
        capacity_per_head=2,
        source_dim=SOURCE_DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = RetrievalHistoryAppendBatch(
        sources=torch.ones((1, SOURCE_DIM)),
        head_codes=torch.zeros(1, dtype=torch.int64),
        operator_codes=torch.zeros(1, dtype=torch.int64),
        timestamps=torch.ones(1, dtype=torch.float64),
        time_ranges=torch.full((1, 2), -1.0, dtype=torch.float64),
        valid_mask=torch.ones(1, dtype=torch.bool),
        eligible_mask=torch.ones(1, dtype=torch.bool),
    )
    ring.append_many(batch)
    snapshot = tensorized_retrieval_view((ring,))
    ring.append_many(batch)
    assert snapshot.n_state.tolist() == [1]
    assert snapshot.sequence_ids.tolist() == [[0]]
    with pytest.raises(RuntimeError, match="changed"):
        snapshot.assert_snapshot_current()
    fork = ring.fork()
    fork.append_many(batch)
    assert fork.count == 2 and ring.count == 2
    ring.release()
    with pytest.raises(RuntimeError, match="released"):
        ring.append_many(batch)


def test_history_filters_future_invalid_and_ineligible_records(
    components: tuple[StructuredStateBank, EmbeddingStateRetriever],
) -> None:
    bank, retriever = components
    state = _new_history(bank, "video-filter", "trajectory-filter")
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
    history = _view((state,))
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
    state = _new_history(bank, "video-status", "trajectory-status")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    history = _view((state,))
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
        _view((state,)),
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
    state = _new_history(bank, "video-errors", "trajectory-errors")
    state = _append_history(bank, state, source=torch.randn(SOURCE_DIM), timestamp=1.0)
    history = _view((state,))
    query = _query(_matching_query(bank, history), (Operator.O1_SNAP,))

    with pytest.raises(TypeError, match="RetrievalHistoryView"):
        retriever(
            bank,
            bank.view((state.bank_state,), (HeadType.O1,)),  # type: ignore[arg-type]
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
