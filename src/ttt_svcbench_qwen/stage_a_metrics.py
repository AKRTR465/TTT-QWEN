"""P15 offline metrics with explicit numerators, denominators, and N/A handling.

Inputs: detached explicit predictions, targets, masks, IDs, and audit counters.
Outputs: named metrics, confusion counts, denominators, and mismatch row indices.
Forbidden: treating an empty denominator as zero or feeding metric labels into model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class O1CountMetricInput:
    soft_counts: Tensor
    hard_counts: Tensor
    target_counts: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        _validate_count_triplet(
            self.soft_counts,
            self.hard_counts,
            self.target_counts,
            self.valid_mask,
            "O1",
        )


@dataclass(frozen=True, slots=True)
class DuplicateMissMetricInput:
    duplicate_count: int
    duplicate_denominator: int
    miss_count: int
    miss_denominator: int

    def __post_init__(self) -> None:
        values = (
            self.duplicate_count,
            self.duplicate_denominator,
            self.miss_count,
            self.miss_denominator,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("duplicate/miss counts must be non-negative integers")
        if self.duplicate_count > self.duplicate_denominator:
            raise ValueError("duplicate count cannot exceed its denominator")
        if self.miss_count > self.miss_denominator:
            raise ValueError("miss count cannot exceed its denominator")


@dataclass(frozen=True, slots=True)
class OperatorMetricInput:
    predicted: Tensor
    targets: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        _validate_index_metric(self.predicted, self.targets, self.valid_mask, 9, "operator")


@dataclass(frozen=True, slots=True)
class TimeMetricInput:
    predicted_modes: Tensor
    target_modes: Tensor
    mode_valid_mask: Tensor
    predicted_starts: Tensor
    predicted_ends: Tensor
    target_starts: Tensor
    target_ends: Tensor
    span_valid_mask: Tensor

    def __post_init__(self) -> None:
        _validate_index_metric(
            self.predicted_modes,
            self.target_modes,
            self.mode_valid_mask,
            4,
            "time mode",
        )
        tensors = (
            self.predicted_starts,
            self.predicted_ends,
            self.target_starts,
            self.target_ends,
        )
        batch_size = self.predicted_modes.shape[0]
        if any(tensor.shape != (batch_size,) or tensor.dtype != torch.int64 for tensor in tensors):
            raise ValueError("time span metric values must be int64 [B]")
        if self.span_valid_mask.shape != (batch_size,) or self.span_valid_mask.dtype != torch.bool:
            raise ValueError("time span valid mask must be bool [B]")
        _require_same_device((*tensors, self.span_valid_mask), "time span metrics")
        valid = self.span_valid_mask
        if bool(
            torch.any(
                valid
                & (
                    (self.predicted_starts < 0)
                    | (self.predicted_ends < self.predicted_starts)
                    | (self.target_starts < 0)
                    | (self.target_ends < self.target_starts)
                )
            )
        ):
            raise ValueError("valid time spans must satisfy 0 <= start <= end")


@dataclass(frozen=True, slots=True)
class RetrievalMetricInput:
    selected_record_ids: tuple[tuple[str, ...], ...]
    relevant_record_ids: tuple[tuple[str, ...] | None, ...]

    def __post_init__(self) -> None:
        if not self.selected_record_ids or len(self.selected_record_ids) != len(
            self.relevant_record_ids
        ):
            raise ValueError("retrieval metrics require aligned non-empty rows")
        for selected, relevant in zip(
            self.selected_record_ids,
            self.relevant_record_ids,
            strict=True,
        ):
            if any(not value for value in selected) or len(set(selected)) != len(selected):
                raise ValueError("selected retrieval IDs must be unique and non-empty")
            if relevant is not None and (
                any(not value for value in relevant) or len(set(relevant)) != len(relevant)
            ):
                raise ValueError("relevant retrieval IDs must be unique and non-empty")


@dataclass(frozen=True, slots=True)
class NumberAgreementMetricInput:
    reader_counts: Tensor
    llm_counts: Tensor
    target_counts: Tensor
    reader_valid_mask: Tensor
    llm_valid_mask: Tensor
    target_valid_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.reader_counts.shape[0] if self.reader_counts.ndim == 1 else -1
        for tensor, name in (
            (self.reader_counts, "Reader counts"),
            (self.llm_counts, "LLM counts"),
            (self.target_counts, "target counts"),
        ):
            if tensor.shape != (batch_size,) or tensor.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 [B]")
        for mask, name in (
            (self.reader_valid_mask, "Reader validity"),
            (self.llm_valid_mask, "LLM validity"),
            (self.target_valid_mask, "target validity"),
        ):
            if mask.shape != (batch_size,) or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be bool [B]")
        _require_same_device(
            (
                self.reader_counts,
                self.llm_counts,
                self.target_counts,
                self.reader_valid_mask,
                self.llm_valid_mask,
                self.target_valid_mask,
            ),
            "number agreement metrics",
        )


@dataclass(frozen=True, slots=True)
class StageAMetricInput:
    o1: O1CountMetricInput
    o2: DuplicateMissMetricInput
    e1: DuplicateMissMetricInput
    e2: DuplicateMissMetricInput
    operator: OperatorMetricInput
    time: TimeMetricInput
    retrieval: RetrievalMetricInput
    numbers: NumberAgreementMetricInput


@dataclass(frozen=True, slots=True)
class StageAMetricReport:
    metrics: tuple[tuple[str, float | None], ...]
    operator_confusion: tuple[tuple[int, ...], ...]
    retrieval_true_positive_count: int
    retrieval_selected_denominator: int
    retrieval_relevant_denominator: int
    reader_llm_mismatch_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("Stage A metric names must be unique")
        if len(self.operator_confusion) != 9 or any(
            len(row) != 9 for row in self.operator_confusion
        ):
            raise ValueError("operator confusion must be 9x9")


def compute_stage_a_metrics(inputs: StageAMetricInput) -> StageAMetricReport:
    o1_valid = inputs.o1.valid_mask
    o1_soft_mae = _masked_mean(
        (inputs.o1.soft_counts.float() - inputs.o1.target_counts.float()).abs(),
        o1_valid,
    )
    o1_hard_accuracy = _masked_mean(
        (inputs.o1.hard_counts == inputs.o1.target_counts).float(),
        o1_valid,
    )

    operator_valid = inputs.operator.valid_mask
    confusion = torch.zeros((9, 9), dtype=torch.int64)
    for target, prediction in zip(
        inputs.operator.targets[operator_valid].tolist(),
        inputs.operator.predicted[operator_valid].tolist(),
        strict=True,
    ):
        confusion[target, prediction] += 1
    per_class: list[float] = []
    for index in range(9):
        denominator = int(confusion[index].sum().item())
        if denominator:
            per_class.append(int(confusion[index, index].item()) / denominator)
    operator_macro = None if not per_class else sum(per_class) / len(per_class)
    unsupported = _masked_mean((inputs.operator.predicted == 8).float(), operator_valid)

    time_mode = _masked_mean(
        (inputs.time.predicted_modes == inputs.time.target_modes).float(),
        inputs.time.mode_valid_mask,
    )
    span_match = (
        (inputs.time.predicted_starts == inputs.time.target_starts)
        & (inputs.time.predicted_ends == inputs.time.target_ends)
    ).float()
    time_span = _masked_mean(span_match, inputs.time.span_valid_mask)

    true_positive = selected_denominator = relevant_denominator = 0
    for selected, relevant in zip(
        inputs.retrieval.selected_record_ids,
        inputs.retrieval.relevant_record_ids,
        strict=True,
    ):
        if relevant is None:
            continue
        selected_set = set(selected)
        relevant_set = set(relevant)
        true_positive += len(selected_set & relevant_set)
        selected_denominator += len(selected_set)
        relevant_denominator += len(relevant_set)
    precision = _safe_rate(true_positive, selected_denominator)
    recall = _safe_rate(true_positive, relevant_denominator)

    comparable = inputs.numbers.reader_valid_mask & inputs.numbers.llm_valid_mask
    mismatch = comparable & (inputs.numbers.reader_counts != inputs.numbers.llm_counts)
    reader_llm_disagreement = _masked_mean(mismatch.float(), comparable)
    reader_target_valid = inputs.numbers.reader_valid_mask & inputs.numbers.target_valid_mask
    reader_target_accuracy = _masked_mean(
        (inputs.numbers.reader_counts == inputs.numbers.target_counts).float(),
        reader_target_valid,
    )

    metrics = (
        ("o1/soft_count_mae", o1_soft_mae),
        ("o1/hard_count_accuracy", o1_hard_accuracy),
        (
            "o2/duplicate_rate",
            _safe_rate(inputs.o2.duplicate_count, inputs.o2.duplicate_denominator),
        ),
        ("o2/missed_new_rate", _safe_rate(inputs.o2.miss_count, inputs.o2.miss_denominator)),
        (
            "e1/duplicate_rate",
            _safe_rate(inputs.e1.duplicate_count, inputs.e1.duplicate_denominator),
        ),
        ("e1/miss_rate", _safe_rate(inputs.e1.miss_count, inputs.e1.miss_denominator)),
        (
            "e2/duplicate_rate",
            _safe_rate(inputs.e2.duplicate_count, inputs.e2.duplicate_denominator),
        ),
        ("e2/miss_rate", _safe_rate(inputs.e2.miss_count, inputs.e2.miss_denominator)),
        ("operator/macro_accuracy", operator_macro),
        ("operator/unsupported_rate", unsupported),
        ("retrieval/precision", precision),
        ("retrieval/recall", recall),
        ("time/mode_accuracy", time_mode),
        ("time/span_exact", time_span),
        ("reader/exact_count_accuracy", reader_target_accuracy),
        ("reader/llm_number_disagreement_rate", reader_llm_disagreement),
    )
    return StageAMetricReport(
        metrics=metrics,
        operator_confusion=tuple(tuple(int(value) for value in row) for row in confusion.tolist()),
        retrieval_true_positive_count=true_positive,
        retrieval_selected_denominator=selected_denominator,
        retrieval_relevant_denominator=relevant_denominator,
        reader_llm_mismatch_rows=tuple(torch.nonzero(mismatch).flatten().tolist()),
    )


def _validate_count_triplet(
    soft: Tensor,
    hard: Tensor,
    target: Tensor,
    valid: Tensor,
    name: str,
) -> None:
    batch_size = soft.shape[0] if soft.ndim == 1 else -1
    if soft.shape != (batch_size,) or not torch.is_floating_point(soft):
        raise ValueError(f"{name} soft counts must be floating [B]")
    if hard.shape != (batch_size,) or hard.dtype != torch.int64:
        raise ValueError(f"{name} hard counts must be int64 [B]")
    if target.shape != (batch_size,) or target.dtype != torch.int64:
        raise ValueError(f"{name} target counts must be int64 [B]")
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError(f"{name} valid mask must be bool [B]")
    _require_same_device((soft, hard, target, valid), f"{name} count metrics")
    if not bool(torch.isfinite(soft).all()) or bool(torch.any(valid & (target < 0))):
        raise ValueError(f"{name} valid counts must be finite and non-negative")


def _validate_index_metric(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    classes: int,
    name: str,
) -> None:
    batch_size = predicted.shape[0] if predicted.ndim == 1 else -1
    if predicted.shape != (batch_size,) or predicted.dtype != torch.int64:
        raise ValueError(f"{name} predictions must be int64 [B]")
    if target.shape != (batch_size,) or target.dtype != torch.int64:
        raise ValueError(f"{name} targets must be int64 [B]")
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError(f"{name} valid mask must be bool [B]")
    _require_same_device((predicted, target, valid), f"{name} metrics")
    if bool(torch.any(valid & ((predicted < 0) | (predicted >= classes)))) or bool(
        torch.any(valid & ((target < 0) | (target >= classes)))
    ):
        raise ValueError(f"valid {name} indices must be within [0, {classes})")


def _masked_mean(values: Tensor, mask: Tensor) -> float | None:
    if not bool(mask.any().item()):
        return None
    return float(values[mask].float().mean().item())


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _require_same_device(tensors: tuple[Tensor, ...], name: str) -> None:
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"{name} must share one device")
