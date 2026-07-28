from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    E1StateTarget,
    E2StateTarget,
    O1StateTarget,
    O2StateTarget,
    OperatorLossInput,
    OuterLossInput,
    ReaderCountMetricInput,
    RetrievalLossInput,
    StateLossInput,
    TimeLossInput,
    compute_answer_loss,
    compute_outer_loss,
    compute_state_loss,
)


def _unit_rows(indices: list[int], *, requires_grad: bool = False) -> Tensor:
    result = torch.zeros(len(indices), 256)
    for row, index in enumerate(indices):
        result[row, abs(index)] = -1.0 if index < 0 else 1.0
    return result.requires_grad_(requires_grad)

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


def test_outer_contains_only_query_answer_and_state() -> None:
    state = compute_state_loss(
        StateLossInput(
            batch_size=1,
            o1=O1StateTarget(
                row_indices=torch.tensor([0]),
                logits=torch.zeros(1, 1, 6, requires_grad=True),
                targets=torch.zeros(1, 1, 6),
                slot_mask=torch.ones(1, 1, dtype=torch.bool),
            ),
        )
    )
    answer = compute_answer_loss(
        AnswerLossInput(
            logits=torch.zeros(1, 2, 3, requires_grad=True),
            labels=torch.tensor([[-100, 1]]),
            number_token_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
    )
    output = compute_outer_loss(OuterLossInput(answer_after=answer, state_after=state))
    assert torch.equal(output.outer, answer.loss.value + state.total)
    assert torch.equal(output.total, output.outer)
    assert not hasattr(output, "auxiliary_ttt")
