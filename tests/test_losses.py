from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    O1StateTarget,
    OperatorLossInput,
    ReaderCountMetricInput,
    StateLossInput,
    compute_answer_loss,
    compute_state_loss,
)


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

    output = compute_answer_loss(
        AnswerLossInput(
            logits=logits,
            labels=labels,
            number_token_mask=number_mask,
            reader_counts=ReaderCountMetricInput(
                predicted_counts=torch.tensor([5, 2]),
                target_counts=torch.tensor([5, 3]),
                valid_mask=torch.ones(2, dtype=torch.bool),
            ),
        )
    )

    expected = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, 6),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(2, 4)
    assert torch.allclose(
        output.loss.per_row, torch.stack((expected[0, 1:].mean(), expected[1, :2].mean()))
    )
    assert output.loss.value.dtype == torch.float32
    assert output.teacher_forced_token_accuracy.value.item() == pytest.approx(5.0 / 6.0)
    assert output.number_token_accuracy.value.item() == pytest.approx(1.0)
    assert output.number_token_accuracy.row_valid_mask.tolist() == [True, False]
    assert output.answer_exact_match.value.item() == pytest.approx(0.5)
    assert output.reader_exact_count_accuracy.value.item() == pytest.approx(0.5)


def test_state_loss_denominator_counts_only_label_carrying_rows() -> None:
    output = compute_state_loss(
        StateLossInput(
            batch_size=2,
            o1=O1StateTarget(
                row_indices=torch.tensor([0]),
                logits=torch.zeros(1, 1, 6),
                targets=torch.ones(1, 1, 6),
                slot_mask=torch.ones(1, 1, dtype=torch.bool),
            ),
            operator=OperatorLossInput(
                logits=torch.zeros(2, 9),
                targets=torch.tensor([0, 0]),
                valid_mask=torch.tensor([True, False]),
            ),
        )
    )

    assert output.o1.row_valid_mask.tolist() == [True, False]
    assert output.o1.valid_counts.tolist() == [1, 0]
    assert output.o1.value.item() == pytest.approx(math.log(2.0))
    assert output.operator.row_valid_mask.tolist() == [True, False]
    assert output.operator.valid_counts.tolist() == [1, 0]
    assert output.operator.value.item() == pytest.approx(math.log(9.0))
