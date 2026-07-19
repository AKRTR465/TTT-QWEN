from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.losses import compute_state_loss
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
    OPERATORS,
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
from ttt_svcbench_qwen.stage_a_targets import (
    E1TargetLabels,
    E2TargetLabels,
    O1TargetLabels,
    O2TargetLabels,
    OfficialWeakSupervision,
    OfficialWeakTargetBuilder,
    QueryTargetLabels,
    RetrievalTargetLabels,
    StageATargetBatch,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import HeadType, O1Payload, StateRecord
from ttt_svcbench_qwen.state_retriever import (
    RetrievalFilterAudit,
    RetrievalReason,
    RetrievalStatus,
    RetrieverOutput,
)

OFFICIAL = TargetProvenance.OFFICIAL_EXPLICIT
SYNTHETIC = TargetProvenance.SYNTHETIC_EXPLICIT
MISSING = TargetProvenance.MISSING


def _leaf(shape: tuple[int, ...], *, fill: float = 0.0) -> Tensor:
    return torch.full(shape, fill, dtype=torch.float32).requires_grad_(True)


def _fresh_e1(row: int) -> E1RuntimeState:
    return E1RuntimeState(
        video_id=f"video-{row}",
        trajectory_id=f"trajectory-{row}",
        query_signature=torch.zeros(512),
        projected_history=torch.zeros(0, 512),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )


def _fresh_e2(row: int) -> E2RuntimeState:
    return E2RuntimeState(
        video_id=f"video-{row}",
        trajectory_id=f"trajectory-{row}",
        query_signature=torch.zeros(512),
        hidden=torch.zeros(2, 768),
        checkpoint_hidden=torch.zeros(0, 2, 768),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )


def _resolution(row: int) -> TimeResolution:
    query_time = float(row + 10)
    return TimeResolution(
        window=TimeWindow(
            mode=TimeWindowMode.HISTORY,
            query_time=query_time,
            start_time=0.0,
            end_time=query_time,
            valid=True,
        ),
        status=TimeResolutionStatus.OK,
        reason="synthetic-history",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
    )


def _record(row: int, column: int) -> StateRecord:
    semantic = torch.zeros(512)
    semantic[(row * 3 + column) % 512] = 1.0
    return StateRecord(
        record_id=f"record-{row}-{column}",
        video_id=f"video-{row}",
        trajectory_id=f"trajectory-{row}",
        head_type=HeadType.O1,
        semantic_embedding=semantic,
        timestamp=float(column + 1),
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=O1Payload(0, 0, ()),
    )


def _typed_predictions(
    batch_size: int,
    *,
    spatial_width: int = 2,
    temporal_width: int = 3,
    query_width: int = 5,
    last_spatial_invalid: bool = False,
) -> tuple[ObservationOutputs, QueryEncoderOutput, RetrieverOutput, dict[str, Tensor]]:
    spatial_mask = torch.ones(batch_size, spatial_width, dtype=torch.bool)
    if last_spatial_invalid:
        spatial_mask[:, -1] = False
    temporal_mask = torch.ones(batch_size, temporal_width, dtype=torch.bool)

    spatial_times = torch.arange(spatial_width, dtype=torch.float64).expand(batch_size, -1).clone()
    spatial_positions = (
        torch.arange(spatial_width, dtype=torch.int64).expand(batch_size, -1).clone()
    )
    spatial_times[~spatial_mask] = -1.0
    spatial_positions[~spatial_mask] = -1
    temporal_times = (
        torch.arange(temporal_width, dtype=torch.float64).expand(batch_size, -1).clone()
    )
    temporal_positions = (
        torch.arange(temporal_width, dtype=torch.int64).expand(batch_size, -1).clone()
    )

    o1_logits = _leaf((batch_size, spatial_width, 6))
    o1_logits.data[~spatial_mask] = 0.0
    o1_probabilities = torch.where(
        spatial_mask.unsqueeze(-1), torch.sigmoid(o1_logits), torch.zeros_like(o1_logits)
    )
    o1 = O1SoftOutput(
        logits=o1_logits,
        probabilities=o1_probabilities,
        soft_count=(
            o1_probabilities[..., 0] * o1_probabilities[..., 1] * o1_probabilities[..., 2]
        ).sum(dim=1),
        valid_mask=spatial_mask,
        timestamps=spatial_times,
        position_ids=spatial_positions,
    )

    o2_identity = torch.zeros(batch_size, spatial_width, 256)
    for row in range(batch_size):
        for column in range(spatial_width):
            if spatial_mask[row, column]:
                o2_identity[row, column, (row + column) % 256] = 1.0
    o2_identity.requires_grad_(True)
    o2_scores = _leaf((batch_size, spatial_width, 2))
    o2_scores.data[~spatial_mask] = 0.0
    o2 = O2SoftOutput(
        identity=o2_identity,
        score_logits=o2_scores,
        score_probabilities=torch.where(
            spatial_mask.unsqueeze(-1), torch.sigmoid(o2_scores), torch.zeros_like(o2_scores)
        ),
        valid_mask=spatial_mask,
        timestamps=spatial_times,
        position_ids=spatial_positions,
    )

    e1_logits = _leaf((batch_size, temporal_width, 3))
    e1 = E1SoftOutput(
        logits=e1_logits,
        probabilities=torch.sigmoid(e1_logits),
        valid_mask=temporal_mask,
        timestamps=temporal_times,
        position_ids=temporal_positions,
        next_states=tuple(_fresh_e1(row) for row in range(batch_size)),
        audit=StreamReplayAudit(
            "e1",
            (temporal_width,) * batch_size,
            (0,) * batch_size,
            (0,) * batch_size,
        ),
    )
    e2_events = _leaf((batch_size, temporal_width, 4))
    e2_phases = _leaf((batch_size, temporal_width, 4))
    e2 = E2SoftOutput(
        event_logits=e2_events,
        phase_logits=e2_phases,
        event_probabilities=torch.sigmoid(e2_events),
        phase_probabilities=torch.softmax(e2_phases, dim=-1),
        valid_mask=temporal_mask,
        timestamps=temporal_times,
        position_ids=temporal_positions,
        next_states=tuple(_fresh_e2(row) for row in range(batch_size)),
        audit=StreamReplayAudit(
            "e2",
            (temporal_width,) * batch_size,
            (0,) * batch_size,
            (0,) * batch_size,
        ),
    )
    observations = ObservationOutputs(o1=o1, o2=o2, e1=e1, e2=e2)

    token_states = torch.zeros(batch_size, query_width, 8)
    pooling = torch.full((batch_size, query_width), 1.0 / query_width)
    q_target = torch.zeros(batch_size, 512)
    q_operator = torch.zeros_like(q_target)
    q_time = torch.zeros_like(q_target)
    q_target[:, 0] = 1.0
    q_operator[:, 1] = 1.0
    q_time[:, 2] = 1.0
    padding_mask = torch.zeros(batch_size, query_width, dtype=torch.bool)
    embeddings = QueryEmbeddingOutput(
        token_states=token_states,
        pooling_weights=pooling,
        q_target=q_target,
        q_operator=q_operator,
        q_time=q_time,
        padding_mask=padding_mask,
    )
    operator_logits = _leaf((batch_size, 9))
    confidence, raw_indices = torch.softmax(operator_logits, dim=-1).max(dim=-1)
    hard_operators = (Operator.O1_SNAP,) * batch_size
    route = OperatorRouterOutput(
        logits=operator_logits,
        confidence=confidence,
        raw_indices=raw_indices,
        hard_operators=hard_operators,
        head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in hard_operators),
        confidence_gate_applied=False,
    )
    mode_logits = _leaf((batch_size, 4))
    span_start_logits = _leaf((batch_size, query_width))
    span_end_logits = _leaf((batch_size, query_width))
    mode_confidence, mode_indices = torch.softmax(mode_logits, dim=-1).max(dim=-1)
    time_logits = TimeResolverLogits(
        mode_logits=mode_logits,
        mode_confidence=mode_confidence,
        mode_indices=mode_indices,
        span_start_logits=span_start_logits,
        span_end_logits=span_end_logits,
        padding_mask=padding_mask,
    )
    resolutions = tuple(_resolution(row) for row in range(batch_size))
    query = QueryEncoderOutput(
        embeddings=embeddings,
        route=route,
        time=TimeResolverOutput(time_logits, resolutions),
        hard_operators=hard_operators,
        head_types=(HeadType.O1,) * batch_size,
    )

    candidate_records = tuple(
        tuple(_record(row, column) for column in range(2)) for row in range(batch_size)
    )
    state_embeddings = torch.stack(
        [torch.stack([record.semantic_embedding for record in row]) for row in candidate_records]
    )
    retrieval_scores = torch.tensor([[0.8, 0.2]] * batch_size, requires_grad=True)
    present_mask = torch.ones(batch_size, 2, dtype=torch.bool)
    selected_mask = torch.tensor([[True, False]] * batch_size)
    selected_records = tuple((row[0],) for row in candidate_records)
    selected_ids = tuple((row[0].record_id,) for row in candidate_records)
    candidate_ids = tuple(tuple(record.record_id for record in row) for row in candidate_records)
    selected_score = float(retrieval_scores[0, 0].detach().item())
    retrieval = RetrieverOutput(
        selected_record_ids=selected_ids,
        selected_scores=((selected_score,),) * batch_size,
        selected_records=selected_records,
        candidate_record_ids=candidate_ids,
        candidate_records=candidate_records,
        state_embeddings=state_embeddings,
        scores=retrieval_scores,
        present_mask=present_mask,
        selected_mask=selected_mask,
        status=(RetrievalStatus.OK,) * batch_size,
        reason=(RetrievalReason.MATCHED,) * batch_size,
        hard_operators=hard_operators,
        time_resolutions=resolutions,
        n_state=torch.full((batch_size,), 2, dtype=torch.int64),
        n_retrieved=torch.ones(batch_size, dtype=torch.int64),
        audit=tuple(
            RetrievalFilterAudit(
                n_state=2,
                head_partition_excluded_count=0,
                query_rejected_count=0,
                owner_mismatch_count=0,
                invalid_count=0,
                retrieval_ineligible_count=0,
                future_count=0,
                outside_window_count=0,
                below_similarity_count=1,
                selected_count=1,
            )
            for _ in range(batch_size)
        ),
        video_ids=tuple(f"video-{row}" for row in range(batch_size)),
        trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch_size)),
        bank_video_ids=tuple(f"video-{row}" for row in range(batch_size)),
        bank_trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch_size)),
        bank_versions=(0,) * batch_size,
    )
    leaves = {
        "o1": o1_logits,
        "o2_identity": o2_identity,
        "o2_score": o2_scores,
        "e1": e1_logits,
        "e2_event": e2_events,
        "e2_phase": e2_phases,
        "operator": operator_logits,
        "time_mode": mode_logits,
        "span_start": span_start_logits,
        "span_end": span_end_logits,
        "retrieval": retrieval_scores,
    }
    return observations, query, retrieval, leaves


def _unit_identity_targets(rows: int, slots: int, *, offset: int = 31) -> Tensor:
    targets = torch.zeros(rows, slots, 256)
    for row in range(rows):
        for column in range(slots):
            targets[row, column, (offset + row + column) % 256] = 1.0
    return targets


def _four_head_labels() -> StageATargetBatch:
    query = QueryTargetLabels(
        operator_targets=torch.tensor([0, 2, 4, 6]),
        time_mode_targets=torch.tensor([0, 1, 2, 3]),
        span_start_targets=torch.tensor([0, 0, 1, 1]),
        span_end_targets=torch.tensor([1, 1, 2, 2]),
        operator_provenance=(OFFICIAL, SYNTHETIC, OFFICIAL, SYNTHETIC),
        time_provenance=(OFFICIAL, SYNTHETIC, OFFICIAL, SYNTHETIC),
        span_provenance=(OFFICIAL, SYNTHETIC, OFFICIAL, SYNTHETIC),
    )
    return StageATargetBatch(
        o1=O1TargetLabels(
            row_indices=torch.tensor([0]),
            targets=torch.ones(1, 2, 6),
            slot_mask=torch.ones(1, 2, dtype=torch.bool),
            provenance=(OFFICIAL,),
        ),
        o2=O2TargetLabels(
            row_indices=torch.tensor([1]),
            identity_targets=_unit_identity_targets(1, 2),
            score_targets=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            slot_mask=torch.ones(1, 2, dtype=torch.bool),
            provenance=(SYNTHETIC,),
        ),
        e1=E1TargetLabels(
            row_indices=torch.tensor([2]),
            targets=torch.ones(1, 3, 3),
            time_mask=torch.ones(1, 3, dtype=torch.bool),
            provenance=(OFFICIAL,),
        ),
        e2=E2TargetLabels(
            row_indices=torch.tensor([3]),
            event_targets=torch.ones(1, 3, 4),
            phase_targets=torch.tensor([[0, 1, 2]]),
            time_mask=torch.ones(1, 3, dtype=torch.bool),
            provenance=(SYNTHETIC,),
        ),
        query=query,
        retrieval=RetrievalTargetLabels(
            relevant_record_ids=(
                ("record-0-1",),
                ("record-1-1",),
                ("record-2-1",),
                None,
            ),
            provenance=(OFFICIAL, SYNTHETIC, OFFICIAL, MISSING),
        ),
    )


def test_four_mutually_exclusive_heads_build_p14_input_and_keep_prediction_gradients() -> None:
    observations, query, retrieval, leaves = _typed_predictions(4)

    state_input = StageATargetBuilder().build(
        observations,
        query,
        retrieval,
        _four_head_labels(),
    )

    assert state_input.batch_size == 4
    assert state_input.o1 is not None and state_input.o1.row_indices.tolist() == [0]
    assert state_input.o2 is not None and state_input.o2.row_indices.tolist() == [1]
    assert state_input.e1 is not None and state_input.e1.row_indices.tolist() == [2]
    assert state_input.e2 is not None and state_input.e2.row_indices.tolist() == [3]
    assert state_input.retrieval is not None
    assert state_input.retrieval.label_mask.tolist() == [
        [True, True],
        [True, True],
        [True, True],
        [False, False],
    ]
    assert state_input.retrieval.targets.tolist() == [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]

    output = compute_state_loss(state_input)
    output.total.backward()

    for name, prediction in leaves.items():
        assert prediction.grad is not None, f"{name} prediction lost its gradient"
        assert bool(torch.isfinite(prediction.grad).all())
        assert float(prediction.grad.abs().sum()) > 0.0


def test_query_labels_cover_all_nine_operators_and_four_time_modes() -> None:
    observations, query, retrieval, _ = _typed_predictions(9)
    provenance = tuple(OFFICIAL if row % 2 == 0 else SYNTHETIC for row in range(9))
    labels = QueryTargetLabels(
        operator_targets=torch.arange(9),
        time_mode_targets=torch.arange(9) % 4,
        span_start_targets=torch.zeros(9, dtype=torch.int64),
        span_end_targets=torch.ones(9, dtype=torch.int64),
        operator_provenance=provenance,
        time_provenance=provenance,
        span_provenance=provenance,
    )

    state_input = StageATargetBuilder().build(
        observations,
        query,
        retrieval,
        StageATargetBatch(query=labels),
    )

    assert tuple(OPERATORS[index] for index in state_input.operator.targets.tolist()) == tuple(
        Operator
    )
    assert state_input.operator.targets[-1].item() == tuple(Operator).index(Operator.UNSUPPORTED)
    assert set(state_input.time.mode_targets.tolist()) == {0, 1, 2, 3}
    assert state_input.operator.valid_mask.all()
    assert state_input.time.mode_valid_mask.all()


def test_missing_provenance_never_constructs_a_loss_component() -> None:
    observations, query, retrieval, _ = _typed_predictions(2)
    query_labels = QueryTargetLabels(
        operator_targets=torch.full((2,), -100, dtype=torch.int64),
        time_mode_targets=torch.full((2,), -100, dtype=torch.int64),
        span_start_targets=torch.full((2,), -100, dtype=torch.int64),
        span_end_targets=torch.full((2,), -100, dtype=torch.int64),
        operator_provenance=(MISSING, MISSING),
        time_provenance=(MISSING, MISSING),
        span_provenance=(MISSING, MISSING),
    )
    missing_o1 = O1TargetLabels(
        row_indices=torch.tensor([0]),
        targets=torch.zeros(1, 2, 6),
        slot_mask=torch.zeros(1, 2, dtype=torch.bool),
        provenance=(MISSING,),
    )
    missing_retrieval = RetrievalTargetLabels(
        relevant_record_ids=(None, None),
        provenance=(MISSING, MISSING),
    )

    with pytest.raises(ValueError, match="no explicit supervised component"):
        StageATargetBuilder().build(
            observations,
            query,
            retrieval,
            StageATargetBatch(
                o1=missing_o1,
                query=query_labels,
                retrieval=missing_retrieval,
            ),
        )


def test_retrieval_relevant_ids_must_exist_on_present_candidate_axis() -> None:
    observations, query, retrieval, _ = _typed_predictions(2)
    labels = RetrievalTargetLabels(
        relevant_record_ids=(("record-0-1",), ("forged-record",)),
        provenance=(OFFICIAL, SYNTHETIC),
    )

    with pytest.raises(ValueError, match="absent from the present candidate axis"):
        StageATargetBuilder().build(
            observations,
            query,
            retrieval,
            StageATargetBatch(retrieval=labels),
        )


def test_head_masks_rows_and_operator_provenance_fail_closed() -> None:
    observations, query, retrieval, _ = _typed_predictions(4, last_spatial_invalid=True)
    labels = _four_head_labels()
    assert labels.o1 is not None
    bad_o1 = O1TargetLabels(
        row_indices=labels.o1.row_indices,
        targets=labels.o1.targets,
        slot_mask=labels.o1.slot_mask,
        provenance=labels.o1.provenance,
    )
    with pytest.raises(ValueError, match="invalid prediction positions"):
        StageATargetBuilder().build(
            observations,
            query,
            retrieval,
            StageATargetBatch(o1=bad_o1),
        )

    duplicate = StageATargetBatch(
        o1=O1TargetLabels(
            torch.tensor([0]),
            torch.ones(1, 2, 6),
            torch.ones(1, 2, dtype=torch.bool),
            (OFFICIAL,),
        ),
        e1=E1TargetLabels(
            torch.tensor([0]),
            torch.ones(1, 3, 3),
            torch.ones(1, 3, dtype=torch.bool),
            (OFFICIAL,),
        ),
    )
    clean_observations, clean_query, clean_retrieval, _ = _typed_predictions(4)
    with pytest.raises(ValueError, match="only one observation head"):
        StageATargetBuilder().build(
            clean_observations,
            clean_query,
            clean_retrieval,
            duplicate,
        )

    mismatched_query = QueryTargetLabels(
        operator_targets=torch.tensor([2, -100, -100, -100]),
        time_mode_targets=torch.full((4,), -100, dtype=torch.int64),
        span_start_targets=torch.full((4,), -100, dtype=torch.int64),
        span_end_targets=torch.full((4,), -100, dtype=torch.int64),
        operator_provenance=(OFFICIAL, MISSING, MISSING, MISSING),
        time_provenance=(MISSING,) * 4,
        span_provenance=(MISSING,) * 4,
    )
    with pytest.raises(ValueError, match="does not match.*head"):
        StageATargetBuilder().build(
            clean_observations,
            clean_query,
            clean_retrieval,
            StageATargetBatch(o1=duplicate.o1, query=mismatched_query),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: O1TargetLabels(
                torch.tensor([0]),
                torch.ones(1, 2, 6, requires_grad=True),
                torch.ones(1, 2, dtype=torch.bool),
                (OFFICIAL,),
            ),
            "detached pure labels",
        ),
        (
            lambda: E1TargetLabels(
                torch.tensor([0]),
                torch.tensor([[[1.0, float("nan"), 0.0]]]),
                torch.ones(1, 1, dtype=torch.bool),
                (OFFICIAL,),
            ),
            "finite",
        ),
        (
            lambda: E2TargetLabels(
                torch.tensor([0]),
                torch.ones(1, 1, 4),
                torch.tensor([[4]]),
                torch.ones(1, 1, dtype=torch.bool),
                (OFFICIAL,),
            ),
            "within",
        ),
        (
            lambda: O1TargetLabels(
                torch.tensor([0], device="meta"),
                torch.ones(1, 1, 6, device="meta"),
                torch.ones(1, 1, dtype=torch.bool, device="meta"),
                (OFFICIAL,),
            ),
            "materialized",
        ),
    ],
)
def test_label_shape_finite_detach_and_device_contracts_fail_closed(
    factory: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_closed_provenance_and_training_only_api_cannot_accept_answer_derived_fields() -> None:
    assert tuple(source.value for source in TargetProvenance) == (
        "official_explicit",
        "official_weak",
        "synthetic_explicit",
        "missing",
    )
    with pytest.raises(ValueError):
        TargetProvenance("pseudo_from_final_count")

    forbidden = {"answer", "count", "occurrence_times", "counting_type", "counting_subtype"}
    public_fields = {
        field.name
        for target_type in (
            O1TargetLabels,
            O2TargetLabels,
            E1TargetLabels,
            E2TargetLabels,
            QueryTargetLabels,
            RetrievalTargetLabels,
            StageATargetBatch,
        )
        for field in fields(target_type)
    }
    assert forbidden.isdisjoint(public_fields)
    assert forbidden.isdisjoint(inspect.signature(StageATargetBuilder.build).parameters)


def test_official_weak_post_forward_loss_uses_masks_bags_and_no_identity_pseudolabels() -> None:
    observations, query, retrieval, leaves = _typed_predictions(4)
    weak = (
        OfficialWeakSupervision(
            query_id="weak-o1",
            operator=Operator.O1_SNAP,
            time_mode=TimeWindowMode.NOW,
            count=1,
            query_time=10.0,
            occurrence_points=(1.0, 20.0),
            occurrence_intervals=(),
            numeric_token_span=(1, 2),
        ),
        OfficialWeakSupervision(
            query_id="weak-o2",
            operator=Operator.O2_UNIQUE,
            time_mode=TimeWindowMode.HISTORY,
            count=2,
            query_time=10.0,
            occurrence_points=(1.0,),
            occurrence_intervals=(),
        ),
        OfficialWeakSupervision(
            query_id="weak-e1",
            operator=Operator.E1_ACTION,
            time_mode=TimeWindowMode.HISTORY,
            count=1,
            query_time=10.0,
            occurrence_points=(1.0,),
            occurrence_intervals=(),
        ),
        OfficialWeakSupervision(
            query_id="weak-e2",
            operator=Operator.E2_EPISODE,
            time_mode=TimeWindowMode.HISTORY,
            count=1,
            query_time=10.0,
            occurrence_points=(),
            occurrence_intervals=((0.5, 1.5),),
        ),
    )

    output = OfficialWeakTargetBuilder().build(observations, query, retrieval, weak)

    assert output.task.valid_rows == 4
    assert output.operator.valid_rows == 4
    assert output.time.valid_rows == 4
    assert output.retrieval.valid_rows == 4
    assert output.audit.future_occurrences_ignored == 1
    assert output.audit.retrieval_bag_sizes == (1, 1, 1, 1)
    assert not output.audit.identity_target_fabricated
    assert not output.audit.unique_retrieval_id_fabricated
    assert torch.equal(
        output.total.detach(),
        (
            output.task.value + output.operator.value + output.retrieval.value + output.time.value
        ).detach(),
    )

    output.total.backward()
    for name in (
        "o1",
        "o2_score",
        "e1",
        "e2_event",
        "e2_phase",
        "operator",
        "time_mode",
        "span_start",
        "span_end",
        "retrieval",
    ):
        assert leaves[name].grad is not None, name
        assert float(leaves[name].grad.norm()) > 0.0, name
    assert leaves["o2_identity"].grad is None
