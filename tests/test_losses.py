from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import PredictorConfig, load_config
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    E1ConsistencyInput,
    E1StateTarget,
    E2ConsistencyInput,
    E2StateTarget,
    EventConsistencyInput,
    IdentityConsistencyAudit,
    IdentityConsistencyInput,
    IdentityPairStatus,
    LossSkipReason,
    LossTerm,
    O1StateTarget,
    O2StateTarget,
    OperatorLossInput,
    OuterLossInput,
    ReaderCountMetricInput,
    RetrievalLossInput,
    StateLossInput,
    TemporalPredictionInput,
    TimeLossInput,
    TrainingLossInput,
    TTTLossInput,
    TTTLossOutput,
    build_temporal_predictor,
    compute_answer_loss,
    compute_event_consistency_loss,
    compute_identity_consistency_loss,
    compute_losses,
    compute_outer_loss,
    compute_state_loss,
    compute_temporal_prediction_loss,
    compute_ttt_loss,
)


def _unit_rows(indices: list[int], *, requires_grad: bool = False) -> Tensor:
    result = torch.zeros(len(indices), 256)
    for row, index in enumerate(indices):
        result[row, abs(index)] = -1.0 if index < 0 else 1.0
    return result.requires_grad_(requires_grad)


def _identity_input(
    current: Tensor,
    previous: Tensor,
    statuses: list[IdentityPairStatus],
    *,
    current_indices: list[int] | None = None,
    previous_indices: list[int] | None = None,
    current_positions: list[int] | None = None,
    previous_positions: list[int] | None = None,
    current_timestamps: list[float] | None = None,
    previous_timestamps: list[float] | None = None,
) -> IdentityConsistencyInput:
    pair_count = len(statuses)
    current_indices = list(range(pair_count)) if current_indices is None else current_indices
    previous_indices = list(range(pair_count)) if previous_indices is None else previous_indices
    current_positions = list(range(pair_count)) if current_positions is None else current_positions
    previous_positions = (
        list(current_positions) if previous_positions is None else previous_positions
    )
    current_timestamps = (
        [float(value) for value in current_positions]
        if current_timestamps is None
        else current_timestamps
    )
    previous_timestamps = (
        list(current_timestamps) if previous_timestamps is None else previous_timestamps
    )
    return IdentityConsistencyInput(
        current_predictions=current.unsqueeze(0),
        previous_targets=previous.unsqueeze(0),
        current_valid_mask=torch.ones(1, current.shape[0], dtype=torch.bool),
        previous_valid_mask=torch.ones(1, previous.shape[0], dtype=torch.bool),
        current_indices=torch.tensor([current_indices]),
        previous_indices=torch.tensor([previous_indices]),
        statuses=torch.tensor([[int(status) for status in statuses]]),
        current_position_ids=torch.tensor([current_positions]),
        previous_position_ids=torch.tensor([previous_positions]),
        current_timestamps=torch.tensor([current_timestamps]),
        previous_timestamps=torch.tensor([previous_timestamps]),
    )


def _e1_input(
    current: Tensor,
    target: Tensor,
    *,
    positions: list[int] | None = None,
    previous_positions: list[int] | None = None,
    timestamps: list[float] | None = None,
    previous_timestamps: list[float] | None = None,
    pair_mask: Tensor | None = None,
    alignment_mask: Tensor | None = None,
) -> E1ConsistencyInput:
    count = current.shape[1]
    positions = list(range(count)) if positions is None else positions
    previous_positions = list(positions) if previous_positions is None else previous_positions
    timestamps = [float(value) for value in positions] if timestamps is None else timestamps
    previous_timestamps = list(timestamps) if previous_timestamps is None else previous_timestamps
    pair_mask = torch.ones(1, count, dtype=torch.bool) if pair_mask is None else pair_mask
    alignment_mask = pair_mask.clone() if alignment_mask is None else alignment_mask
    return E1ConsistencyInput(
        current_probabilities=current,
        previous_target_probabilities=target,
        pair_mask=pair_mask,
        alignment_mask=alignment_mask,
        current_position_ids=torch.tensor([positions]),
        previous_position_ids=torch.tensor([previous_positions]),
        current_timestamps=torch.tensor([timestamps]),
        previous_timestamps=torch.tensor([previous_timestamps]),
    )


def _e2_input(
    current_event: Tensor,
    target_event: Tensor,
    current_phase: Tensor,
    target_phase: Tensor,
    *,
    positions: list[int] | None = None,
) -> E2ConsistencyInput:
    count = current_event.shape[1]
    positions = list(range(count)) if positions is None else positions
    timestamps = [float(value) for value in positions]
    return E2ConsistencyInput(
        current_event_probabilities=current_event,
        previous_event_target_probabilities=target_event,
        current_phase_probabilities=current_phase,
        previous_phase_target_probabilities=target_phase,
        pair_mask=torch.ones(1, count, dtype=torch.bool),
        alignment_mask=torch.ones(1, count, dtype=torch.bool),
        current_position_ids=torch.tensor([positions]),
        previous_position_ids=torch.tensor([positions]),
        current_timestamps=torch.tensor([timestamps]),
        previous_timestamps=torch.tensor([timestamps]),
    )


def _loss_term(values: list[float], valid: list[bool]) -> LossTerm:
    per_row = torch.tensor(values, dtype=torch.float32)
    valid_mask = torch.tensor(valid, dtype=torch.bool)
    counts = valid_mask.to(torch.int64)
    value = per_row[valid_mask].mean() if any(valid) else per_row.sum() * 0.0
    return LossTerm(
        value=value,
        per_row=torch.where(valid_mask, per_row, torch.zeros_like(per_row)),
        row_valid_mask=valid_mask,
        valid_counts=counts,
        mask_counts=torch.ones_like(counts),
        skip_reasons=tuple(
            None if is_valid else LossSkipReason.NO_VALID_SUPPORT for is_valid in valid
        ),
    )


def _fake_ttt(values: list[float], valid: list[bool]) -> TTTLossOutput:
    term = _loss_term([0.0] * len(values), valid)
    zeros = torch.zeros(len(values), dtype=torch.int64)
    valid_mask = torch.tensor(valid, dtype=torch.bool)
    per_row = torch.where(
        valid_mask,
        torch.tensor(values, dtype=torch.float32),
        torch.zeros(len(values), dtype=torch.float32),
    )
    total = per_row[valid_mask].mean() if any(valid) else per_row.sum() * 0.0
    return TTTLossOutput(
        pred=term,
        identity=term,
        e1_event=term,
        e2_event=term,
        event=term,
        total=total,
        per_row_total=per_row,
        update_valid_mask=valid_mask,
        identity_audit=IdentityConsistencyAudit(
            matched_counts=zeros,
            mismatch_counts=zeros.clone(),
            duplicate_counts=zeros.clone(),
            low_confidence_counts=zeros.clone(),
            invalid_source_counts=zeros.clone(),
            padding_counts=zeros.clone(),
        ),
    )


def _minimal_state_output() -> tuple[StateLossInput, Tensor]:
    logits = torch.zeros(1, 1, 6, requires_grad=True)
    inputs = StateLossInput(
        batch_size=1,
        o1=O1StateTarget(
            row_indices=torch.tensor([0]),
            logits=logits,
            targets=torch.ones_like(logits),
            slot_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
    )
    return inputs, logits


def test_predictor_uses_frozen_config_and_exact_parameter_count() -> None:
    config = load_config().predictor
    predictor = build_temporal_predictor(config)

    assert sum(parameter.numel() for parameter in predictor.parameters()) == 2_363_136
    assert predictor.network[0].eps == pytest.approx(config.layer_norm_eps)
    assert predictor(torch.randn(2, 3, 768)).shape == (2, 3, 768)

    invalid = PredictorConfig(
        input_dim=768,
        hidden_dim=1536,
        output_dim=768,
        layer_norm_eps=1.0e-4,
        activation="silu",
        linear_bias=True,
        parameter_count=2_363_136,
    )
    with pytest.raises(ValueError, match="layer_norm_eps"):
        build_temporal_predictor(invalid)


def test_temporal_prediction_is_contiguous_fp32_and_target_stopped() -> None:
    predictor = build_temporal_predictor(load_config().predictor)
    hidden = torch.randn(2, 4, 768, requires_grad=True)
    hidden.data[1, 3].zero_()
    inputs = TemporalPredictionInput(
        hidden=hidden,
        valid_mask=torch.tensor([[True, True, True, True], [True, True, True, False]]),
        position_ids=torch.tensor([[0, 1, 3, 4], [10, 11, 12, -1]]),
    )

    output = compute_temporal_prediction_loss(predictor, inputs)

    predictions = predictor(hidden[:, :-1])
    target = hidden[:, 1:].detach()
    expected_items = (predictions.float() - target.float()).square().mean(dim=-1)
    assert output.valid_counts.tolist() == [2, 2]
    assert output.mask_counts.tolist() == [3, 2]
    assert torch.allclose(
        output.per_row,
        torch.stack((expected_items[0, [0, 2]].mean(), expected_items[1, :2].mean())),
    )
    assert output.value.dtype == torch.float32

    target_view = hidden[:, 1:]
    target_view.retain_grad()
    output.value.backward()
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[:, -1], torch.zeros_like(hidden.grad[:, -1]))
    assert target_view.grad is None


def test_temporal_t_less_than_two_is_invalid_not_a_valid_zero() -> None:
    predictor = build_temporal_predictor(load_config().predictor)
    hidden = torch.randn(2, 1, 768, requires_grad=True)
    output = compute_temporal_prediction_loss(
        predictor,
        TemporalPredictionInput(
            hidden=hidden,
            valid_mask=torch.ones(2, 1, dtype=torch.bool),
            position_ids=torch.tensor([[0], [10]]),
        ),
    )

    assert output.value.item() == 0.0
    assert output.row_valid_mask.tolist() == [False, False]
    assert output.skip_reasons == (
        LossSkipReason.INSUFFICIENT_TIME,
        LossSkipReason.INSUFFICIENT_TIME,
    )
    output.value.backward()
    assert hidden.grad is not None


def test_identity_current_to_previous_numeric_stopgrad_and_unit_norm() -> None:
    current = _unit_rows([0, 0, 0], requires_grad=True)
    previous = _unit_rows([0, 1, -0], requires_grad=True)
    previous.data[2, 0] = -1.0
    inputs = _identity_input(
        current,
        previous,
        [IdentityPairStatus.MATCHED] * 3,
    )

    output = compute_identity_consistency_loss(inputs)

    assert output.term.per_row.item() == pytest.approx(1.0)
    assert output.audit.matched_counts.tolist() == [3]
    output.term.value.backward()
    assert current.grad is not None
    assert previous.grad is None

    bad = current.detach().clone()
    bad[0].mul_(2.0)
    with pytest.raises(ValueError, match="unit L2"):
        _identity_input(bad, previous.detach(), [IdentityPairStatus.MATCHED] * 3)


def test_identity_status_audit_distinguishes_invalid_source_and_padding() -> None:
    current = _unit_rows([0])
    previous = _unit_rows([0])
    statuses = [
        IdentityPairStatus.MISMATCH,
        IdentityPairStatus.DUPLICATE,
        IdentityPairStatus.LOW_CONFIDENCE,
        IdentityPairStatus.INVALID_SOURCE,
        IdentityPairStatus.PADDING,
    ]
    inputs = _identity_input(
        current,
        previous,
        statuses,
        current_indices=[0, 0, 0, -1, -1],
        previous_indices=[0, 0, 0, -1, -1],
        current_positions=[0, 1, 2, -1, -1],
        previous_positions=[9, 8, 7, -1, -1],
        current_timestamps=[0.0, 1.0, 2.0, -1.0, -1.0],
        previous_timestamps=[9.0, 8.0, 7.0, -1.0, -1.0],
    )

    output = compute_identity_consistency_loss(inputs)

    assert not output.term.row_valid_mask.item()
    assert output.term.mask_counts.tolist() == [4]
    assert output.audit.mismatch_counts.tolist() == [1]
    assert output.audit.duplicate_counts.tolist() == [1]
    assert output.audit.low_confidence_counts.tolist() == [1]
    assert output.audit.invalid_source_counts.tolist() == [1]
    assert output.audit.padding_counts.tolist() == [1]

    with pytest.raises(ValueError, match="invalid-source"):
        _identity_input(
            current,
            previous,
            [IdentityPairStatus.INVALID_SOURCE],
            current_indices=[999],
            previous_indices=[-1],
            current_positions=[-1],
            previous_positions=[-1],
            current_timestamps=[-1.0],
            previous_timestamps=[-1.0],
        )


def test_identity_requires_one_to_one_and_aligned_time_metadata() -> None:
    current = _unit_rows([0, 1])
    previous = _unit_rows([0, 1])
    with pytest.raises(ValueError, match="one-to-one"):
        _identity_input(
            current,
            previous,
            [IdentityPairStatus.MATCHED, IdentityPairStatus.MATCHED],
            current_indices=[0, 0],
        )
    with pytest.raises(ValueError, match="timestamps"):
        _identity_input(
            current,
            previous,
            [IdentityPairStatus.MATCHED, IdentityPairStatus.MATCHED],
            previous_timestamps=[0.0, 1.01],
        )
    with pytest.raises(ValueError, match="positions"):
        _identity_input(
            current,
            previous,
            [IdentityPairStatus.MATCHED, IdentityPairStatus.MATCHED],
            previous_positions=[0, 9],
        )


def test_event_mse_phase_kl_and_all_targets_are_stopped() -> None:
    e1_current = torch.zeros(1, 1, 3, requires_grad=True)
    e1_target = torch.ones(1, 1, 3, requires_grad=True)
    e2_current = torch.zeros(1, 1, 4, requires_grad=True)
    e2_target = torch.zeros(1, 1, 4, requires_grad=True)
    phase_current = torch.tensor([[[0.5, 0.25, 0.125, 0.125]]], requires_grad=True)
    phase_target = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]], requires_grad=True)
    inputs = EventConsistencyInput(
        e1=_e1_input(e1_current, e1_target),
        e2=_e2_input(e2_current, e2_target, phase_current, phase_target),
    )

    output = compute_event_consistency_loss(inputs)

    expected_kl = (
        phase_target.detach() * (phase_target.detach().log() - phase_current.log())
    ).sum()
    assert output.e1.value.item() == pytest.approx(1.0)
    assert torch.allclose(output.e2.value, expected_kl)
    assert torch.allclose(output.total.value, 1.0 + expected_kl)
    output.total.value.backward()
    assert e1_current.grad is not None
    assert phase_current.grad is not None
    assert e1_target.grad is None
    assert e2_target.grad is None
    assert phase_target.grad is None


def test_event_alignment_mismatch_and_duplicate_positions_fail_closed() -> None:
    current = torch.zeros(1, 2, 3)
    target = torch.zeros_like(current)
    with pytest.raises(ValueError, match="positions"):
        _e1_input(current, target, previous_positions=[0, 9])
    with pytest.raises(ValueError, match="timestamps"):
        _e1_input(current, target, previous_timestamps=[0.0, 2.1])
    with pytest.raises(ValueError, match="unique"):
        _e1_input(current, target, positions=[5, 5])


def test_ttt_weights_are_exact_and_no_o1_term_exists() -> None:
    predictor = build_temporal_predictor(load_config().predictor)
    hidden = torch.randn(1, 2, 768, requires_grad=True)
    identity = _identity_input(
        _unit_rows([0], requires_grad=True),
        _unit_rows([1], requires_grad=True),
        [IdentityPairStatus.MATCHED],
    )
    zero_e1 = torch.zeros(1, 1, 3, requires_grad=True)
    zero_e2 = torch.zeros(1, 1, 4, requires_grad=True)
    uniform = torch.full((1, 1, 4), 0.25, requires_grad=True)
    inputs = TTTLossInput(
        temporal=TemporalPredictionInput(
            hidden=hidden,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            position_ids=torch.tensor([[0, 1]]),
        ),
        identity=identity,
        event=EventConsistencyInput(
            e1=_e1_input(zero_e1, torch.zeros_like(zero_e1)),
            e2=_e2_input(
                zero_e2,
                torch.zeros_like(zero_e2),
                uniform,
                torch.full_like(uniform, 0.25),
            ),
        ),
    )

    output = compute_ttt_loss(predictor, inputs)

    expected = output.pred.value + 0.5 * output.identity.value + 0.5 * output.event.value
    assert torch.allclose(output.total, expected)
    assert (output.pred_weight, output.identity_weight, output.event_weight) == (1.0, 0.5, 0.5)
    assert output.o1_unlabeled_weight == 0.0


def test_ttt_scalar_uses_union_valid_per_row_mean_for_mixed_batch() -> None:
    predictor = build_temporal_predictor(load_config().predictor)
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
    hidden = torch.zeros(4, 2, 768)
    hidden[0, 1] = 1.0
    temporal = TemporalPredictionInput(
        hidden=hidden,
        valid_mask=torch.tensor([[True, True], [True, False], [True, False], [True, False]]),
        position_ids=torch.tensor([[0, 1], [0, -1], [0, -1], [0, -1]]),
    )
    current_identities = torch.zeros(4, 1, 256)
    previous_identities = torch.zeros_like(current_identities)
    current_identities[:, 0, 0] = 1.0
    previous_identities[:, 0, 0] = 1.0
    previous_identities[1, 0, 0] = 0.0
    previous_identities[1, 0, 1] = 1.0
    identity = IdentityConsistencyInput(
        current_predictions=current_identities,
        previous_targets=previous_identities,
        current_valid_mask=torch.ones(4, 1, dtype=torch.bool),
        previous_valid_mask=torch.ones(4, 1, dtype=torch.bool),
        current_indices=torch.tensor([[-1], [0], [-1], [-1]]),
        previous_indices=torch.tensor([[-1], [0], [-1], [-1]]),
        statuses=torch.tensor(
            [
                [int(IdentityPairStatus.INVALID_SOURCE)],
                [int(IdentityPairStatus.MATCHED)],
                [int(IdentityPairStatus.INVALID_SOURCE)],
                [int(IdentityPairStatus.INVALID_SOURCE)],
            ]
        ),
        current_position_ids=torch.tensor([[-1], [5], [-1], [-1]]),
        previous_position_ids=torch.tensor([[-1], [5], [-1], [-1]]),
        current_timestamps=torch.tensor([[-1.0], [5.0], [-1.0], [-1.0]]),
        previous_timestamps=torch.tensor([[-1.0], [5.0], [-1.0], [-1.0]]),
    )
    e1_current = torch.zeros(4, 1, 3)
    e1_target = torch.zeros_like(e1_current)
    e1_target[2] = 1.0
    e1_mask = torch.tensor([[False], [False], [True], [False]])
    e2_current = torch.zeros(4, 1, 4)
    e2_target = torch.zeros_like(e2_current)
    e2_target[3] = 1.0
    e2_mask = torch.tensor([[False], [False], [False], [True]])
    positions = torch.tensor([[0], [1], [2], [3]])
    timestamps = positions.float()
    uniform = torch.full((4, 1, 4), 0.25)
    event = EventConsistencyInput(
        e1=E1ConsistencyInput(
            current_probabilities=e1_current,
            previous_target_probabilities=e1_target,
            pair_mask=e1_mask,
            alignment_mask=e1_mask,
            current_position_ids=positions,
            previous_position_ids=positions,
            current_timestamps=timestamps,
            previous_timestamps=timestamps,
        ),
        e2=E2ConsistencyInput(
            current_event_probabilities=e2_current,
            previous_event_target_probabilities=e2_target,
            current_phase_probabilities=uniform,
            previous_phase_target_probabilities=uniform.clone(),
            pair_mask=e2_mask,
            alignment_mask=e2_mask,
            current_position_ids=positions,
            previous_position_ids=positions,
            current_timestamps=timestamps,
            previous_timestamps=timestamps,
        ),
    )

    output = compute_ttt_loss(
        predictor,
        TTTLossInput(temporal=temporal, identity=identity, event=event),
    )

    assert output.pred.row_valid_mask.tolist() == [True, False, False, False]
    assert output.identity.row_valid_mask.tolist() == [False, True, False, False]
    assert output.e1_event.row_valid_mask.tolist() == [False, False, True, False]
    assert output.e2_event.row_valid_mask.tolist() == [False, False, False, True]
    assert output.event.value.item() == pytest.approx(1.0)
    assert output.per_row_total.tolist() == pytest.approx([1.0, 0.5, 0.5, 0.5])
    assert output.total.item() == pytest.approx(0.625)


def test_state_loss_supervises_exactly_one_dense_head_per_row() -> None:
    o1_logits = torch.zeros(1, 1, 6, requires_grad=True)
    o1_targets = torch.ones(1, 1, 6, requires_grad=True)
    o2_identity = _unit_rows([0]).reshape(1, 1, 256).requires_grad_()
    o2_identity_target = _unit_rows([0]).reshape(1, 1, 256).requires_grad_()
    o2_scores = torch.zeros(1, 1, 2, requires_grad=True)
    o2_score_targets = torch.ones(1, 1, 2, requires_grad=True)
    e1_logits = torch.zeros(1, 1, 3, requires_grad=True)
    e1_targets = torch.ones(1, 1, 3, requires_grad=True)
    e2_events = torch.zeros(1, 1, 4, requires_grad=True)
    e2_event_targets = torch.ones(1, 1, 4, requires_grad=True)
    e2_phases = torch.zeros(1, 1, 4, requires_grad=True)
    operator_logits = torch.zeros(4, 9, requires_grad=True)
    retrieval_logits = torch.zeros(4, 2, requires_grad=True)
    mode_logits = torch.zeros(4, 4, requires_grad=True)
    start_logits = torch.zeros(4, 3, requires_grad=True)
    end_logits = torch.zeros(4, 3, requires_grad=True)
    inputs = StateLossInput(
        batch_size=4,
        o1=O1StateTarget(
            row_indices=torch.tensor([0]),
            logits=o1_logits,
            targets=o1_targets,
            slot_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
        o2=O2StateTarget(
            row_indices=torch.tensor([1]),
            identity_predictions=o2_identity,
            identity_targets=o2_identity_target,
            score_logits=o2_scores,
            score_targets=o2_score_targets,
            slot_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
        e1=E1StateTarget(
            row_indices=torch.tensor([2]),
            logits=e1_logits,
            targets=e1_targets,
            time_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
        e2=E2StateTarget(
            row_indices=torch.tensor([3]),
            event_logits=e2_events,
            event_targets=e2_event_targets,
            phase_logits=e2_phases,
            phase_targets=torch.tensor([[2]]),
            time_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
        operator=OperatorLossInput(
            logits=operator_logits,
            targets=torch.tensor([0, 1, 2, 3]),
            valid_mask=torch.ones(4, dtype=torch.bool),
        ),
        retrieval=RetrievalLossInput(
            logits=retrieval_logits,
            targets=torch.tensor([[1.0, 0.0]] * 4),
            present_mask=torch.ones(4, 2, dtype=torch.bool),
            label_mask=torch.ones(4, 2, dtype=torch.bool),
        ),
        time=TimeLossInput(
            mode_logits=mode_logits,
            mode_targets=torch.tensor([0, 1, 2, 3]),
            mode_valid_mask=torch.ones(4, dtype=torch.bool),
            span_start_logits=start_logits,
            span_end_logits=end_logits,
            span_start_targets=torch.tensor([0, -100, 1, -100]),
            span_end_targets=torch.tensor([1, -100, 2, -100]),
            token_valid_mask=torch.ones(4, 3, dtype=torch.bool),
        ),
    )

    output = compute_state_loss(inputs)

    assert output.o1.row_valid_mask.tolist() == [True, False, False, False]
    assert output.o2.row_valid_mask.tolist() == [False, True, False, False]
    assert output.e1.row_valid_mask.tolist() == [False, False, True, False]
    assert output.e2.row_valid_mask.tolist() == [False, False, False, True]
    ln2 = math.log(2.0)
    expected_task = (ln2 + ln2 + ln2 + ln2 + math.log(4.0)) / 4.0
    expected_time = math.log(4.0) + 2.0 * math.log(3.0)
    expected_total = expected_task + math.log(9.0) + ln2 + expected_time
    assert output.task.value.item() == pytest.approx(expected_task)
    assert output.total.item() == pytest.approx(expected_total)
    assert (
        output.task_weight,
        output.operator_weight,
        output.retrieval_weight,
        output.time_weight,
    ) == (1.0, 1.0, 1.0, 1.0)

    output.total.backward()
    for prediction in (
        o1_logits,
        o2_identity,
        o2_scores,
        e1_logits,
        e2_events,
        e2_phases,
        operator_logits,
        retrieval_logits,
        mode_logits,
        start_logits,
        end_logits,
    ):
        assert prediction.grad is not None
    for target in (
        o1_targets,
        o2_identity_target,
        o2_score_targets,
        e1_targets,
        e2_event_targets,
    ):
        assert target.grad is None


def test_state_loss_rejects_cross_head_row_alias_and_masks_retrieval() -> None:
    o1 = O1StateTarget(
        row_indices=torch.tensor([0]),
        logits=torch.zeros(1, 1, 6),
        targets=torch.zeros(1, 1, 6),
        slot_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    e1 = E1StateTarget(
        row_indices=torch.tensor([0]),
        logits=torch.zeros(1, 1, 3),
        targets=torch.zeros(1, 1, 3),
        time_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="exactly one"):
        StateLossInput(batch_size=1, o1=o1, e1=e1)

    retrieval = RetrievalLossInput(
        logits=torch.tensor([[0.0, 100.0]]),
        targets=torch.tensor([[1.0, 1.0]]),
        present_mask=torch.tensor([[True, True]]),
        label_mask=torch.tensor([[True, False]]),
    )
    output = compute_state_loss(StateLossInput(batch_size=1, retrieval=retrieval))
    assert output.retrieval.value.item() == pytest.approx(math.log(2.0))


def test_answer_uses_causal_shift_and_reports_separate_metrics() -> None:
    logits = torch.zeros(2, 5, 6, dtype=torch.float16, requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        logits[0, 3, 0] = 10.0
        logits[1, 0, 1] = 10.0
        logits[1, 1, 2] = 10.0
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, 1, 2, -100, -100]])
    number_mask = torch.zeros(2, 5, dtype=torch.bool)
    number_mask[0, 3] = True
    inputs = AnswerLossInput(
        logits=logits,
        labels=labels,
        number_token_mask=number_mask,
        reader_counts=ReaderCountMetricInput(
            predicted_counts=torch.tensor([5, 2]),
            target_counts=torch.tensor([5, 3]),
            valid_mask=torch.ones(2, dtype=torch.bool),
        ),
    )

    output = compute_answer_loss(inputs)

    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    expected = F.cross_entropy(
        shift_logits.reshape(-1, 6), shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(2, 4)
    expected_rows = torch.stack((expected[0, 1:].mean(), expected[1, :2].mean()))
    assert torch.allclose(output.loss.per_row, expected_rows)
    assert output.loss.value.dtype == torch.float32
    assert output.teacher_forced_token_accuracy.value.item() == pytest.approx(5.0 / 6.0)
    assert output.number_token_accuracy.value.item() == pytest.approx(1.0)
    assert output.number_token_accuracy.row_valid_mask.tolist() == [True, False]
    assert output.answer_exact_match.value.item() == pytest.approx(0.5)
    assert output.reader_exact_count_accuracy.value.item() == pytest.approx(0.5)


def test_outer_auxiliary_mean_excludes_invalid_support_rows() -> None:
    state_input, _ = _minimal_state_output()
    state = compute_state_loss(state_input)
    answer = compute_answer_loss(
        AnswerLossInput(
            logits=torch.zeros(1, 2, 3, requires_grad=True),
            labels=torch.tensor([[-100, 1]]),
            number_token_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
    )
    support = (_fake_ttt([1.0, 0.0], [True, False]), _fake_ttt([3.0], [True]))

    output = compute_outer_loss(
        OuterLossInput(answer_after=answer, state_after=state, support_ttt=support)
    )

    assert output.auxiliary_ttt.value.item() == pytest.approx(2.0)
    assert output.auxiliary_ttt.valid_counts.sum().item() == 2
    assert torch.allclose(output.outer, answer.loss.value + state.total)
    assert torch.allclose(output.total, output.outer + 0.2)


def test_loss_inputs_defer_nonfinite_ownership_and_entrypoint_is_not_a_skeleton() -> None:
    inputs = AnswerLossInput(
        logits=torch.tensor([[[0.0, float("nan")], [0.0, 0.0]]]),
        labels=torch.tensor([[-100, 0]]),
        number_token_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert not torch.isfinite(compute_answer_loss(inputs).loss.value)
    with pytest.raises(ValueError, match="TrainingLossInput"):
        compute_losses()


def test_complete_entrypoint_requires_registered_predictor() -> None:
    predictor = build_temporal_predictor(load_config().predictor)
    hidden = torch.randn(1, 2, 768)
    identity = _identity_input(_unit_rows([0]), _unit_rows([0]), [IdentityPairStatus.MATCHED])
    zeros3 = torch.zeros(1, 1, 3)
    zeros4 = torch.zeros(1, 1, 4)
    uniform = torch.full((1, 1, 4), 0.25)
    ttt = TTTLossInput(
        temporal=TemporalPredictionInput(
            hidden=hidden,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            position_ids=torch.tensor([[0, 1]]),
        ),
        identity=identity,
        event=EventConsistencyInput(
            e1=_e1_input(zeros3, zeros3.clone()),
            e2=_e2_input(zeros4, zeros4.clone(), uniform, uniform.clone()),
        ),
    )
    state_input, _ = _minimal_state_output()
    answer = AnswerLossInput(
        logits=torch.zeros(1, 2, 3),
        labels=torch.tensor([[-100, 1]]),
        number_token_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    inputs = TrainingLossInput(
        ttt=ttt,
        state_after=state_input,
        answer_after=answer,
        support_ttt=(),
    )

    with pytest.raises(ValueError, match="Predictor"):
        compute_losses(inputs)
    output = compute_losses(inputs, predictor=predictor)
    assert output.total is output.outer.total
    assert output.outer.auxiliary_ttt.value.item() == pytest.approx(output.ttt.total.item())
    assert torch.allclose(
        output.total,
        output.answer.loss.value + output.state.total + 0.1 * output.ttt.total,
    )


def test_state_target_types_document_builder_owned_matching_and_soft_fsm_proxy() -> None:
    assert "pre-matched" in (O1StateTarget.__doc__ or "")
    assert "soft-FSM proxy" in (E2StateTarget.__doc__ or "")
