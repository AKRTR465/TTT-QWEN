from __future__ import annotations

import pytest
import torch

from ttt_svcbench_qwen.stage_a_metrics import (
    DuplicateMissMetricInput,
    NumberAgreementMetricInput,
    O1CountMetricInput,
    OperatorMetricInput,
    RetrievalMetricInput,
    StageAMetricInput,
    TimeMetricInput,
    compute_stage_a_metrics,
)


def _inputs() -> StageAMetricInput:
    return StageAMetricInput(
        o1=O1CountMetricInput(
            soft_counts=torch.tensor([1.5, 3.0]),
            hard_counts=torch.tensor([2, 3]),
            target_counts=torch.tensor([2, 2]),
            valid_mask=torch.tensor([True, True]),
        ),
        o2=DuplicateMissMetricInput(1, 4, 2, 5),
        e1=DuplicateMissMetricInput(0, 0, 1, 2),
        e2=DuplicateMissMetricInput(2, 4, 0, 0),
        operator=OperatorMetricInput(
            predicted=torch.arange(9),
            targets=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7]),
            valid_mask=torch.ones(9, dtype=torch.bool),
        ),
        time=TimeMetricInput(
            predicted_modes=torch.tensor([0, 1, 2, 3]),
            target_modes=torch.tensor([0, 2, 2, 3]),
            mode_valid_mask=torch.ones(4, dtype=torch.bool),
            predicted_starts=torch.tensor([0, 1, 2, 3]),
            predicted_ends=torch.tensor([0, 2, 2, 4]),
            target_starts=torch.tensor([0, 1, 1, 3]),
            target_ends=torch.tensor([0, 2, 2, 4]),
            span_valid_mask=torch.tensor([True, True, True, False]),
        ),
        retrieval=RetrievalMetricInput(
            selected_record_ids=(("a", "b"), (), ("d",)),
            relevant_record_ids=(("a", "c"), None, ("d",)),
        ),
        numbers=NumberAgreementMetricInput(
            reader_counts=torch.tensor([2, 3, 4]),
            llm_counts=torch.tensor([2, 2, 4]),
            target_counts=torch.tensor([2, 3, 5]),
            reader_valid_mask=torch.ones(3, dtype=torch.bool),
            llm_valid_mask=torch.ones(3, dtype=torch.bool),
            target_valid_mask=torch.ones(3, dtype=torch.bool),
        ),
    )


def test_stage_a_metrics_use_hand_checked_denominators_and_confusion() -> None:
    report = compute_stage_a_metrics(_inputs())
    metrics = dict(report.metrics)
    assert metrics["o1/soft_count_mae"] == 0.75
    assert metrics["o1/hard_count_accuracy"] == 0.5
    assert metrics["o2/duplicate_rate"] == 0.25
    assert metrics["o2/missed_new_rate"] == 0.4
    assert metrics["e1/duplicate_rate"] is None
    assert metrics["e1/miss_rate"] == 0.5
    assert metrics["e2/duplicate_rate"] == 0.5
    assert metrics["e2/miss_rate"] is None
    assert metrics["operator/unsupported_rate"] == pytest.approx(1 / 9)
    assert metrics["time/mode_accuracy"] == 0.75
    assert metrics["time/span_exact"] == pytest.approx(2 / 3)
    assert metrics["retrieval/precision"] == pytest.approx(2 / 3)
    assert metrics["retrieval/recall"] == pytest.approx(2 / 3)
    assert metrics["reader/exact_count_accuracy"] == pytest.approx(2 / 3)
    assert metrics["reader/llm_number_disagreement_rate"] == pytest.approx(1 / 3)
    assert report.operator_confusion[7][8] == 1
    assert report.reader_llm_mismatch_rows == (1,)


def test_stage_a_metrics_report_na_instead_of_zero_for_empty_denominators() -> None:
    inputs = _inputs()
    empty = StageAMetricInput(
        o1=O1CountMetricInput(
            inputs.o1.soft_counts,
            inputs.o1.hard_counts,
            inputs.o1.target_counts,
            torch.zeros(2, dtype=torch.bool),
        ),
        o2=DuplicateMissMetricInput(0, 0, 0, 0),
        e1=DuplicateMissMetricInput(0, 0, 0, 0),
        e2=DuplicateMissMetricInput(0, 0, 0, 0),
        operator=OperatorMetricInput(
            inputs.operator.predicted,
            inputs.operator.targets,
            torch.zeros(9, dtype=torch.bool),
        ),
        time=TimeMetricInput(
            inputs.time.predicted_modes,
            inputs.time.target_modes,
            torch.zeros(4, dtype=torch.bool),
            inputs.time.predicted_starts,
            inputs.time.predicted_ends,
            inputs.time.target_starts,
            inputs.time.target_ends,
            torch.zeros(4, dtype=torch.bool),
        ),
        retrieval=RetrievalMetricInput(((),), (None,)),
        numbers=NumberAgreementMetricInput(
            inputs.numbers.reader_counts,
            inputs.numbers.llm_counts,
            inputs.numbers.target_counts,
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        ),
    )
    assert all(value is None for _, value in compute_stage_a_metrics(empty).metrics)
