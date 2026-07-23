"""Training-only explicit target assembly for Stage A.

Inputs: typed model predictions plus label-only dataclasses with explicit provenance.
Outputs: P14 ``StateLossInput`` values that preserve the prediction autograd graph.
Forbidden: deriving dense labels from an answer, final count, occurrence times, or runtime data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.losses import (
    E1StateTarget,
    E2StateTarget,
    O1StateTarget,
    O2StateTarget,
    OperatorLossInput,
    RetrievalLossInput,
    StateLossInput,
    TimeLossInput,
)
from ttt_svcbench_qwen.observation_heads import ObservationOutputs
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    OPERATORS,
    TIME_MODES,
    Operator,
    QueryEncoderOutput,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import RETRIEVAL_HEAD_ORDER, HeadType
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


class TargetProvenance(StrEnum):
    """The complete and intentionally closed set of Stage A label origins."""

    OFFICIAL_EXPLICIT = "official_explicit"
    OFFICIAL_WEAK = "official_weak"
    SYNTHETIC_EXPLICIT = "synthetic_explicit"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class O1TargetLabels:
    """Pre-matched O1 object/target/visible/enter/exit/confidence labels."""

    row_indices: Tensor
    targets: Tensor
    slot_mask: Tensor
    provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        _validate_dense_labels(
            self.row_indices,
            self.targets,
            self.slot_mask,
            self.provenance,
            width=6,
            name="O1",
        )


@dataclass(frozen=True, slots=True)
class O2TargetLabels:
    """Pre-matched O2 identity plus novelty/match-confidence labels."""

    row_indices: Tensor
    identity_targets: Tensor
    score_targets: Tensor
    slot_mask: Tensor
    provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        rows = _validate_row_indices(self.row_indices, self.provenance, "O2")
        if (
            self.identity_targets.ndim != 3
            or self.identity_targets.shape[0] != rows
            or self.identity_targets.shape[-1] != 256
            or not torch.is_floating_point(self.identity_targets)
        ):
            raise ValueError("O2 identity labels must be floating [R, N, 256]")
        shape = self.identity_targets.shape[:2]
        if self.score_targets.shape != (*shape, 2) or not torch.is_floating_point(
            self.score_targets
        ):
            raise ValueError("O2 novelty/match labels must be floating [R, N, 2]")
        _validate_mask(self.slot_mask, shape, "O2 slot_mask")
        _require_same_device(
            (self.row_indices, self.identity_targets, self.score_targets, self.slot_mask),
            "O2 labels",
        )
        _require_materialized_finite_label(self.identity_targets, "O2 identity labels")
        _require_probability_label(self.score_targets, "O2 novelty/match labels")
        _validate_provenance_mask(self.provenance, self.slot_mask, "O2")
        _require_masked_zero(self.identity_targets, self.slot_mask, "O2 identity labels")
        _require_masked_zero(self.score_targets, self.slot_mask, "O2 novelty/match labels")
        valid = self.identity_targets[self.slot_mask]
        if valid.numel():
            norms = torch.linalg.vector_norm(valid.float(), dim=-1)
            if not torch.allclose(
                norms,
                torch.ones_like(norms),
                atol=5.0e-4,
                rtol=5.0e-4,
            ):
                raise ValueError("valid O2 identity labels must be unit L2 normalized")


@dataclass(frozen=True, slots=True)
class E1TargetLabels:
    """Dense E1 eventness/completion/transition labels."""

    row_indices: Tensor
    targets: Tensor
    time_mask: Tensor
    provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        _validate_dense_labels(
            self.row_indices,
            self.targets,
            self.time_mask,
            self.provenance,
            width=3,
            name="E1",
        )


@dataclass(frozen=True, slots=True)
class E2TargetLabels:
    """Dense E2 event labels and categorical soft-FSM phase labels."""

    row_indices: Tensor
    event_targets: Tensor
    phase_targets: Tensor
    time_mask: Tensor
    provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        rows = _validate_row_indices(self.row_indices, self.provenance, "E2")
        if (
            self.event_targets.ndim != 3
            or self.event_targets.shape[0] != rows
            or self.event_targets.shape[-1] != 4
            or not torch.is_floating_point(self.event_targets)
        ):
            raise ValueError("E2 event labels must be floating [R, T, 4]")
        shape = self.event_targets.shape[:2]
        if self.phase_targets.shape != shape or self.phase_targets.dtype != torch.int64:
            raise ValueError("E2 phase labels must be int64 [R, T]")
        _validate_mask(self.time_mask, shape, "E2 time_mask")
        _require_same_device(
            (self.row_indices, self.event_targets, self.phase_targets, self.time_mask),
            "E2 labels",
        )
        _require_probability_label(self.event_targets, "E2 event labels")
        _require_materialized(self.phase_targets, "E2 phase labels")
        _validate_provenance_mask(self.provenance, self.time_mask, "E2")
        _require_masked_zero(self.event_targets, self.time_mask, "E2 event labels")
        valid = self.time_mask
        if bool(torch.any((self.phase_targets[valid] < 0) | (self.phase_targets[valid] >= 4))):
            raise ValueError("valid E2 phase labels must be within [0, 4)")
        if bool(torch.any(~valid & (self.phase_targets != -100))):
            raise ValueError("masked E2 phase labels must use -100")


@dataclass(frozen=True, slots=True)
class QueryTargetLabels:
    """Batch-aligned operator, time-mode, and inclusive numeric-span labels."""

    operator_targets: Tensor
    time_mode_targets: Tensor
    span_start_targets: Tensor
    span_end_targets: Tensor
    operator_provenance: tuple[TargetProvenance, ...]
    time_provenance: tuple[TargetProvenance, ...]
    span_provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        tensors = (
            self.operator_targets,
            self.time_mode_targets,
            self.span_start_targets,
            self.span_end_targets,
        )
        if any(tensor.ndim != 1 or tensor.dtype != torch.int64 for tensor in tensors):
            raise ValueError("Query labels must be int64 [B]")
        shapes = {tensor.shape for tensor in tensors}
        if len(shapes) != 1 or self.operator_targets.shape[0] <= 0:
            raise ValueError("Query labels must share one non-empty batch shape")
        _require_same_device(tensors, "Query labels")
        for tensor, name in zip(
            tensors,
            ("operator", "time mode", "span start", "span end"),
            strict=True,
        ):
            _require_materialized(tensor, f"Query {name} labels")
        batch_size = int(self.operator_targets.shape[0])
        _validate_index_labels(
            self.operator_targets,
            self.operator_provenance,
            batch_size,
            len(OPERATORS),
            "operator",
        )
        _validate_index_labels(
            self.time_mode_targets,
            self.time_provenance,
            batch_size,
            len(TIME_MODES),
            "time mode",
        )
        _validate_provenance(self.span_provenance, batch_size, "span")
        start_missing = self.span_start_targets == -100
        end_missing = self.span_end_targets == -100
        if not torch.equal(start_missing, end_missing):
            raise ValueError("Query span start/end labels must be missing together")
        for row, provenance in enumerate(self.span_provenance):
            missing = provenance is TargetProvenance.MISSING
            if missing != bool(start_missing[row].item()):
                raise ValueError("Query span provenance must exactly match the -100 sentinel")
            if not missing and (
                int(self.span_start_targets[row].item()) < 0
                or int(self.span_start_targets[row].item()) > int(self.span_end_targets[row].item())
            ):
                raise ValueError("explicit Query spans require 0 <= start <= end")

    @property
    def batch_size(self) -> int:
        return int(self.operator_targets.shape[0])


@dataclass(frozen=True, slots=True)
class RetrievalTargetLabels:
    """Relevant record IDs for every explicitly labelled Retriever row."""

    relevant_record_ids: tuple[tuple[str, ...] | None, ...]
    provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.relevant_record_ids)
        if batch_size <= 0:
            raise ValueError("Retrieval labels require a non-empty batch")
        _validate_provenance(self.provenance, batch_size, "retrieval")
        for row, (record_ids, provenance) in enumerate(
            zip(self.relevant_record_ids, self.provenance, strict=True)
        ):
            if provenance is TargetProvenance.MISSING:
                if record_ids is not None:
                    raise ValueError(
                        f"missing retrieval row {row} cannot carry relevant record IDs"
                    )
                continue
            if not isinstance(record_ids, tuple):
                raise TypeError("explicit retrieval labels must be tuples of record IDs")
            if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
                raise ValueError("relevant retrieval record IDs must be non-empty strings")
            if len(set(record_ids)) != len(record_ids):
                raise ValueError("relevant retrieval record IDs must be unique per row")

    @property
    def batch_size(self) -> int:
        return len(self.relevant_record_ids)


@dataclass(frozen=True, slots=True)
class AnswerTargetLabels:
    """Teacher-forced source labels plus an independent offline Reader-count target."""

    base_labels: Tensor
    base_number_token_mask: Tensor
    target_counts: Tensor
    answer_provenance: tuple[TargetProvenance, ...]
    count_provenance: tuple[TargetProvenance, ...]

    def __post_init__(self) -> None:
        if self.base_labels.ndim != 2 or self.base_labels.dtype != torch.int64:
            raise ValueError("Answer base labels must be int64 [B, L]")
        batch_size = int(self.base_labels.shape[0])
        if batch_size <= 0 or self.base_labels.shape[1] < 2:
            raise ValueError("Answer base labels require non-empty B and L>=2")
        if (
            self.base_number_token_mask.shape != self.base_labels.shape
            or self.base_number_token_mask.dtype != torch.bool
        ):
            raise ValueError("Answer number mask must be bool [B, L]")
        if self.target_counts.shape != (batch_size,) or self.target_counts.dtype != torch.int64:
            raise ValueError("Reader target counts must be int64 [B]")
        _require_same_device(
            (self.base_labels, self.base_number_token_mask, self.target_counts),
            "Answer labels",
        )
        for tensor, name in (
            (self.base_labels, "Answer base labels"),
            (self.base_number_token_mask, "Answer number mask"),
            (self.target_counts, "Reader target counts"),
        ):
            _require_materialized(tensor, name)
        _validate_provenance(self.answer_provenance, batch_size, "answer")
        _validate_provenance(self.count_provenance, batch_size, "Reader count")
        supervised = self.base_labels != -100
        if bool(torch.any(self.base_number_token_mask & ~supervised)):
            raise ValueError("Answer number mask must be a subset of supervised labels")
        for row, provenance in enumerate(self.answer_provenance):
            present = bool(supervised[row].any().item())
            if (provenance is TargetProvenance.MISSING) != (not present):
                raise ValueError("Answer provenance must exactly match supervised tokens")
        for row, provenance in enumerate(self.count_provenance):
            count = int(self.target_counts[row].item())
            if provenance is TargetProvenance.MISSING:
                if count != -100:
                    raise ValueError("missing Reader count targets must use -100")
            elif count < 0:
                raise ValueError("explicit Reader count targets must be non-negative")

    @property
    def batch_size(self) -> int:
        return int(self.base_labels.shape[0])


@dataclass(frozen=True, slots=True)
class StageATargetBatch:
    """Pure labels only; no prediction or runtime object is permitted here."""

    o1: O1TargetLabels | None = None
    o2: O2TargetLabels | None = None
    e1: E1TargetLabels | None = None
    e2: E2TargetLabels | None = None
    query: QueryTargetLabels | None = None
    retrieval: RetrievalTargetLabels | None = None

    def __post_init__(self) -> None:
        expected = (
            (self.o1, O1TargetLabels, "o1"),
            (self.o2, O2TargetLabels, "o2"),
            (self.e1, E1TargetLabels, "e1"),
            (self.e2, E2TargetLabels, "e2"),
            (self.query, QueryTargetLabels, "query"),
            (self.retrieval, RetrievalTargetLabels, "retrieval"),
        )
        for value, target_type, name in expected:
            if value is not None and not isinstance(value, target_type):
                raise TypeError(f"Stage A {name} labels have the wrong typed label class")


class StageATargetBuilder:
    """Join explicit Stage A labels to typed P13/P14 predictions, fail closed."""

    __slots__ = ()

    def __call__(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        labels: StageATargetBatch,
    ) -> StateLossInput:
        return self.build(observations, query, retrieval, labels)

    def build(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        labels: StageATargetBatch,
    ) -> StateLossInput:
        batch_size, device = _validate_builder_inputs(observations, query, retrieval, labels)

        o1 = self._build_o1(observations, labels.o1, batch_size, device)
        o2 = self._build_o2(observations, labels.o2, batch_size, device)
        e1 = self._build_e1(observations, labels.e1, batch_size, device)
        e2 = self._build_e2(observations, labels.e2, batch_size, device)
        operator, time = self._build_query(query, labels.query, batch_size, device)
        self._validate_task_operator_alignment((o1, o2, e1, e2), labels.query)
        retrieval_input = self._build_retrieval(retrieval, labels.retrieval, batch_size)

        components = (o1, o2, e1, e2, operator, retrieval_input, time)
        if all(component is None for component in components):
            raise ValueError("Stage A target batch contains no explicit supervised component")
        return StateLossInput(
            batch_size=batch_size,
            o1=o1,
            o2=o2,
            e1=e1,
            e2=e2,
            operator=operator,
            retrieval=retrieval_input,
            time=time,
        )

    @staticmethod
    def _build_o1(
        observations: ObservationOutputs,
        labels: O1TargetLabels | None,
        batch_size: int,
        device: torch.device,
    ) -> O1StateTarget | None:
        if labels is None:
            return None
        selected = _select_explicit_rows(labels.row_indices, labels.provenance, batch_size, device)
        if selected is None:
            return None
        label_positions, global_rows = selected
        mask = labels.slot_mask.index_select(0, label_positions)
        prediction_mask = observations.o1.valid_mask.index_select(0, global_rows)
        _require_label_mask_within_prediction(mask, prediction_mask, "O1")
        return O1StateTarget(
            row_indices=global_rows,
            logits=observations.o1.logits.index_select(0, global_rows),
            targets=labels.targets.index_select(0, label_positions),
            slot_mask=mask,
        )

    @staticmethod
    def _build_o2(
        observations: ObservationOutputs,
        labels: O2TargetLabels | None,
        batch_size: int,
        device: torch.device,
    ) -> O2StateTarget | None:
        if labels is None:
            return None
        selected = _select_explicit_rows(labels.row_indices, labels.provenance, batch_size, device)
        if selected is None:
            return None
        label_positions, global_rows = selected
        mask = labels.slot_mask.index_select(0, label_positions)
        prediction_mask = observations.o2.valid_mask.index_select(0, global_rows)
        _require_label_mask_within_prediction(mask, prediction_mask, "O2")
        return O2StateTarget(
            row_indices=global_rows,
            identity_predictions=observations.o2.identity.index_select(0, global_rows),
            identity_targets=labels.identity_targets.index_select(0, label_positions),
            score_logits=observations.o2.score_logits.index_select(0, global_rows),
            score_targets=labels.score_targets.index_select(0, label_positions),
            slot_mask=mask,
        )

    @staticmethod
    def _build_e1(
        observations: ObservationOutputs,
        labels: E1TargetLabels | None,
        batch_size: int,
        device: torch.device,
    ) -> E1StateTarget | None:
        if labels is None:
            return None
        selected = _select_explicit_rows(labels.row_indices, labels.provenance, batch_size, device)
        if selected is None:
            return None
        label_positions, global_rows = selected
        mask = labels.time_mask.index_select(0, label_positions)
        prediction_mask = observations.e1.valid_mask.index_select(0, global_rows)
        _require_label_mask_within_prediction(mask, prediction_mask, "E1")
        return E1StateTarget(
            row_indices=global_rows,
            logits=observations.e1.logits.index_select(0, global_rows),
            targets=labels.targets.index_select(0, label_positions),
            time_mask=mask,
        )

    @staticmethod
    def _build_e2(
        observations: ObservationOutputs,
        labels: E2TargetLabels | None,
        batch_size: int,
        device: torch.device,
    ) -> E2StateTarget | None:
        if labels is None:
            return None
        selected = _select_explicit_rows(labels.row_indices, labels.provenance, batch_size, device)
        if selected is None:
            return None
        label_positions, global_rows = selected
        mask = labels.time_mask.index_select(0, label_positions)
        prediction_mask = observations.e2.valid_mask.index_select(0, global_rows)
        _require_label_mask_within_prediction(mask, prediction_mask, "E2")
        return E2StateTarget(
            row_indices=global_rows,
            event_logits=observations.e2.event_logits.index_select(0, global_rows),
            event_targets=labels.event_targets.index_select(0, label_positions),
            phase_logits=observations.e2.phase_logits.index_select(0, global_rows),
            phase_targets=labels.phase_targets.index_select(0, label_positions),
            time_mask=mask,
        )

    @staticmethod
    def _build_query(
        query: QueryEncoderOutput,
        labels: QueryTargetLabels | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[OperatorLossInput | None, TimeLossInput | None]:
        if labels is None:
            return None, None
        if labels.batch_size != batch_size:
            raise ValueError("Query label batch size does not match predictions")
        if labels.operator_targets.device != device:
            raise ValueError("Query labels and predictions must share one device")

        operator_mask = _provenance_mask(labels.operator_provenance, device)
        operator = (
            OperatorLossInput(query.route.logits, labels.operator_targets, operator_mask)
            if bool(operator_mask.any().item())
            else None
        )
        mode_mask = _provenance_mask(labels.time_provenance, device)
        span_mask = _provenance_mask(labels.span_provenance, device)
        if not bool(mode_mask.any().item()) and not bool(span_mask.any().item()):
            return operator, None
        time = TimeLossInput(
            mode_logits=query.time.logits.mode_logits,
            mode_targets=labels.time_mode_targets,
            mode_valid_mask=mode_mask,
            span_start_logits=query.time.logits.span_start_logits,
            span_end_logits=query.time.logits.span_end_logits,
            span_start_targets=labels.span_start_targets,
            span_end_targets=labels.span_end_targets,
            token_valid_mask=~query.time.logits.padding_mask,
        )
        return operator, time

    @staticmethod
    def _build_retrieval(
        retrieval: RetrieverOutput,
        labels: RetrievalTargetLabels | None,
        batch_size: int,
    ) -> RetrievalLossInput | None:
        if labels is None:
            return None
        if labels.batch_size != batch_size:
            raise ValueError("Retrieval label batch size does not match predictions")
        explicit = tuple(
            provenance is not TargetProvenance.MISSING for provenance in labels.provenance
        )
        if not any(explicit):
            return None

        targets = torch.zeros_like(retrieval.scores)
        label_mask = torch.zeros_like(retrieval.present_mask)
        for row, is_explicit in enumerate(explicit):
            if not is_explicit:
                continue
            relevant = labels.relevant_record_ids[row]
            assert relevant is not None
            candidates = tuple(
                retrieval.candidate_record_id(row, column)
                for column in range(retrieval.scores.shape[1])
            )
            present_ids = tuple(
                record_id
                for column, record_id in enumerate(candidates)
                if bool(retrieval.present_mask[row, column].item())
            )
            if any(record_id is None for record_id in present_ids):
                raise ValueError("present Retriever candidates must have record IDs")
            present = tuple(record_id for record_id in present_ids if record_id is not None)
            missing_ids = set(relevant).difference(present)
            if missing_ids:
                raise ValueError(
                    "relevant retrieval IDs are absent from the present candidate axis: "
                    f"{sorted(missing_ids)}"
                )
            label_mask[row] = retrieval.present_mask[row]
            relevant_set = set(relevant)
            for column, record_id in enumerate(candidates):
                if record_id in relevant_set:
                    targets[row, column] = 1.0
        return RetrievalLossInput(
            logits=retrieval.scores,
            targets=targets,
            present_mask=retrieval.present_mask,
            label_mask=label_mask,
        )

    @staticmethod
    def _validate_task_operator_alignment(
        targets: tuple[
            O1StateTarget | None,
            O2StateTarget | None,
            E1StateTarget | None,
            E2StateTarget | None,
        ],
        query_labels: QueryTargetLabels | None,
    ) -> None:
        row_heads: dict[int, HeadType] = {}
        for target, head in zip(
            targets,
            (HeadType.O1, HeadType.O2, HeadType.E1, HeadType.E2),
            strict=True,
        ):
            if target is None:
                continue
            for row in target.row_indices.tolist():
                if row in row_heads:
                    raise ValueError("each Stage A row may label only one observation head")
                row_heads[row] = head
        if query_labels is None:
            return
        for row, head in row_heads.items():
            if query_labels.operator_provenance[row] is TargetProvenance.MISSING:
                continue
            operator_index = int(query_labels.operator_targets[row].item())
            expected_head = OPERATOR_TO_HEAD_TYPE[OPERATORS[operator_index]]
            if expected_head is not head:
                raise ValueError("explicit operator label does not match the row's head target")


def _validate_builder_inputs(
    observations: ObservationOutputs,
    query: QueryEncoderOutput,
    retrieval: RetrieverOutput,
    labels: StageATargetBatch,
) -> tuple[int, torch.device]:
    if not isinstance(observations, ObservationOutputs):
        raise TypeError("Stage A target builder requires ObservationOutputs")
    if not isinstance(query, QueryEncoderOutput):
        raise TypeError("Stage A target builder requires QueryEncoderOutput")
    if not isinstance(retrieval, RetrieverOutput):
        raise TypeError("Stage A target builder requires RetrieverOutput")
    if not isinstance(labels, StageATargetBatch):
        raise TypeError("Stage A target builder requires pure StageATargetBatch labels")

    if query.route.confidence_gate_applied:
        raise ValueError("Stage A targets require training-mode query predictions")
    batch_size = int(observations.o1.logits.shape[0])
    if (
        query.embeddings.q_target.shape[0] != batch_size
        or query.route.logits.shape[0] != batch_size
        or query.time.logits.mode_logits.shape[0] != batch_size
        or retrieval.scores.shape[0] != batch_size
    ):
        raise ValueError("Stage A predictions must share one batch size")
    if retrieval.hard_operators != query.hard_operators:
        raise ValueError("Retriever operator provenance does not match QueryEncoderOutput")
    if retrieval.time_resolutions != query.time.resolutions:
        raise ValueError("Retriever time provenance does not match QueryEncoderOutput")

    device = observations.o1.logits.device
    if device.type == "meta":
        raise ValueError("Stage A target assembly requires materialized predictions")
    prediction_tensors = (
        observations.o1.logits,
        observations.o2.identity,
        observations.o2.score_logits,
        observations.e1.logits,
        observations.e2.event_logits,
        observations.e2.phase_logits,
        query.route.logits,
        query.time.logits.mode_logits,
        query.time.logits.span_start_logits,
        query.time.logits.span_end_logits,
        retrieval.scores,
    )
    if any(tensor.device != device for tensor in prediction_tensors):
        raise ValueError("all Stage A predictions must share one device")
    return batch_size, device


def _validate_dense_labels(
    row_indices: Tensor,
    targets: Tensor,
    mask: Tensor,
    provenance: tuple[TargetProvenance, ...],
    *,
    width: int,
    name: str,
) -> None:
    rows = _validate_row_indices(row_indices, provenance, name)
    if (
        targets.ndim != 3
        or targets.shape[0] != rows
        or targets.shape[-1] != width
        or not torch.is_floating_point(targets)
    ):
        raise ValueError(f"{name} labels must be floating [R, N, {width}]")
    _validate_mask(mask, targets.shape[:2], f"{name} mask")
    _require_same_device((row_indices, targets, mask), f"{name} labels")
    _require_probability_label(targets, f"{name} labels")
    _validate_provenance_mask(provenance, mask, name)
    _require_masked_zero(targets, mask, f"{name} labels")


def _validate_row_indices(
    row_indices: Tensor,
    provenance: tuple[TargetProvenance, ...],
    name: str,
) -> int:
    if row_indices.ndim != 1 or row_indices.dtype != torch.int64 or row_indices.shape[0] <= 0:
        raise ValueError(f"{name} row_indices must be int64 [R>0]")
    _require_materialized(row_indices, f"{name} row_indices")
    rows = int(row_indices.shape[0])
    _validate_provenance(provenance, rows, name)
    values = row_indices.tolist()
    if any(row < 0 for row in values) or len(set(values)) != rows:
        raise ValueError(f"{name} row_indices must be unique and non-negative")
    return rows


def _validate_mask(mask: Tensor, shape: torch.Size, name: str) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be bool {tuple(shape)}")
    _require_materialized(mask, name)


def _validate_provenance(
    provenance: tuple[TargetProvenance, ...],
    rows: int,
    name: str,
) -> None:
    if not isinstance(provenance, tuple) or len(provenance) != rows:
        raise ValueError(f"{name} provenance must contain one entry per row")
    if any(not isinstance(value, TargetProvenance) for value in provenance):
        raise TypeError(f"{name} provenance contains an unsupported source")


def _validate_provenance_mask(
    provenance: tuple[TargetProvenance, ...],
    mask: Tensor,
    name: str,
) -> None:
    for row, source in enumerate(provenance):
        present = bool(mask[row].any().item())
        if source is TargetProvenance.MISSING and present:
            raise ValueError(f"missing {name} provenance cannot enable a label mask")
        if source is not TargetProvenance.MISSING and not present:
            raise ValueError(f"explicit {name} provenance requires at least one labelled item")


def _validate_index_labels(
    targets: Tensor,
    provenance: tuple[TargetProvenance, ...],
    batch_size: int,
    upper_bound: int,
    name: str,
) -> None:
    _validate_provenance(provenance, batch_size, name)
    for row, source in enumerate(provenance):
        value = int(targets[row].item())
        if source is TargetProvenance.MISSING:
            if value != -100:
                raise ValueError(f"missing {name} labels must use -100")
        elif not 0 <= value < upper_bound:
            raise ValueError(f"explicit {name} labels must be within [0, {upper_bound})")


def _select_explicit_rows(
    row_indices: Tensor,
    provenance: tuple[TargetProvenance, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    if row_indices.device != device:
        raise ValueError("head labels and predictions must share one device")
    positions = tuple(
        index for index, source in enumerate(provenance) if source is not TargetProvenance.MISSING
    )
    if not positions:
        return None
    label_positions = torch.tensor(positions, dtype=torch.int64, device=device)
    global_rows = row_indices.index_select(0, label_positions)
    if bool(torch.any(global_rows >= batch_size)):
        raise ValueError("Stage A head row index is outside the prediction batch")
    return label_positions, global_rows


def _provenance_mask(
    provenance: tuple[TargetProvenance, ...],
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        [source is not TargetProvenance.MISSING for source in provenance],
        dtype=torch.bool,
        device=device,
    )


def _require_label_mask_within_prediction(
    label_mask: Tensor,
    prediction_mask: Tensor,
    name: str,
) -> None:
    if label_mask.shape != prediction_mask.shape:
        raise ValueError(f"{name} label mask does not match selected prediction shape")
    if bool(torch.any(label_mask & ~prediction_mask)):
        raise ValueError(f"{name} labels cannot target invalid prediction positions")


def _require_same_device(tensors: tuple[Tensor, ...], name: str) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"{name} must share one device")


def _require_materialized(tensor: Tensor, name: str) -> None:
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must be materialized")


def _require_materialized_finite_label(tensor: Tensor, name: str) -> None:
    _require_materialized(tensor, name)
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ValueError(f"{name} must be detached pure labels")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _require_probability_label(tensor: Tensor, name: str) -> None:
    _require_materialized_finite_label(tensor, name)
    if bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError(f"{name} must stay within [0, 1]")


def _require_masked_zero(values: Tensor, mask: Tensor, name: str) -> None:
    if bool(torch.any(values[~mask] != 0.0)):
        raise ValueError(f"masked {name} must be zero")


@dataclass(frozen=True, slots=True)
class OfficialWeakSupervision:
    """One official training sidecar consumed strictly after model forward."""

    query_id: str
    operator: Operator
    time_mode: TimeWindowMode
    count: int
    query_time: float
    occurrence_points: tuple[float, ...]
    occurrence_intervals: tuple[tuple[float, float], ...]
    numeric_token_span: tuple[int, int] | None = None
    provenance: TargetProvenance = TargetProvenance.OFFICIAL_WEAK

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("official weak supervision requires a non-empty query_id")
        if self.operator is Operator.UNSUPPORTED:
            raise ValueError("official weak supervision requires one of eight operators")
        if self.count < 0 or not math.isfinite(self.query_time) or self.query_time < 0.0:
            raise ValueError("official weak count/query_time is invalid")
        if self.provenance is not TargetProvenance.OFFICIAL_WEAK:
            raise ValueError("official weak supervision requires official_weak provenance")
        if any(not math.isfinite(value) or value < 0.0 for value in self.occurrence_points):
            raise ValueError("official weak occurrence points must be finite and non-negative")
        for start, end in self.occurrence_intervals:
            if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
                raise ValueError(
                    "official weak occurrence intervals must satisfy 0 <= start <= end"
                )
        if self.numeric_token_span is not None:
            start, end = self.numeric_token_span
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError("numeric token span must use inclusive non-negative indices")


@dataclass(frozen=True, slots=True)
class OfficialWeakLossTerm:
    value: Tensor
    valid_rows: int

    def __post_init__(self) -> None:
        if self.value.ndim != 0 or self.value.dtype != torch.float32:
            raise ValueError("official weak losses must be FP32 scalars")
        if self.value.device.type != "meta" and not bool(torch.isfinite(self.value).item()):
            raise ValueError("official weak losses must be finite")
        if type(self.valid_rows) is not int or self.valid_rows < 0:
            raise ValueError("official weak valid_rows must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OperatorDiagnosticAudit:
    """Additive 8-target x 9-prediction Router diagnostics."""

    raw_confusion: tuple[int, ...]
    effective_confusion: tuple[int, ...]
    class_loss_sums: tuple[float, ...]
    class_support: tuple[int, ...]
    confidence_sum: float
    entropy_sum: float
    temperature_sum: float
    temperature_count: int

    @classmethod
    def empty(cls) -> OperatorDiagnosticAudit:
        return cls((0,) * 72, (0,) * 72, (0.0,) * 8, (0,) * 8, 0.0, 0.0, 0.0, 0)

    def __post_init__(self) -> None:
        if len(self.raw_confusion) != 72 or len(self.effective_confusion) != 72:
            raise ValueError("Operator confusion audits must be flattened 8x9 matrices")
        if len(self.class_loss_sums) != 8 or len(self.class_support) != 8:
            raise ValueError("Operator class audits must contain eight official classes")
        integer_values = (
            *self.raw_confusion,
            *self.effective_confusion,
            *self.class_support,
            self.temperature_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_values):
            raise ValueError("Operator diagnostic counts must be non-negative integers")
        floating_values = (
            *self.class_loss_sums,
            self.confidence_sum,
            self.entropy_sum,
            self.temperature_sum,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in floating_values):
            raise ValueError("Operator diagnostic sums must be finite and non-negative")
        row_count = sum(self.class_support)
        if sum(self.raw_confusion) != row_count or sum(self.effective_confusion) != row_count:
            raise ValueError("Operator confusion totals must equal official class support")


@dataclass(frozen=True, slots=True)
class TaskDiagnosticAudit:
    """Additive Task subloss, count-error, and dense-label diagnostics."""

    count_loss_sums: tuple[float, ...]
    count_abs_error_sums: tuple[float, ...]
    count_rows: tuple[int, ...]
    component_loss_sums: tuple[float, ...]
    component_rows: tuple[int, ...]
    o1_loss_sums: tuple[float, ...]
    o1_rows: tuple[int, ...]
    channel_positive_counts: tuple[int, ...]
    channel_negative_counts: tuple[int, ...]
    channel_masked_counts: tuple[int, ...]
    channel_true_positive_counts: tuple[int, ...]
    channel_false_positive_counts: tuple[int, ...]
    channel_false_negative_counts: tuple[int, ...]
    e1_representable_occurrences: int
    e1_unrepresentable_occurrences: int

    @classmethod
    def empty(cls) -> TaskDiagnosticAudit:
        return cls(
            (0.0,) * 4,
            (0.0,) * 4,
            (0,) * 4,
            (0.0,) * 3,
            (0,) * 3,
            (0.0,) * 2,
            (0,) * 2,
            (0,) * 7,
            (0,) * 7,
            (0,) * 7,
            (0,) * 7,
            (0,) * 7,
            (0,) * 7,
            0,
            0,
        )

    def __post_init__(self) -> None:
        expected_lengths = (
            (self.count_loss_sums, 4),
            (self.count_abs_error_sums, 4),
            (self.count_rows, 4),
            (self.component_loss_sums, 3),
            (self.component_rows, 3),
            (self.o1_loss_sums, 2),
            (self.o1_rows, 2),
            (self.channel_positive_counts, 7),
            (self.channel_negative_counts, 7),
            (self.channel_masked_counts, 7),
            (self.channel_true_positive_counts, 7),
            (self.channel_false_positive_counts, 7),
            (self.channel_false_negative_counts, 7),
        )
        if any(len(values) != width for values, width in expected_lengths):
            raise ValueError("Task diagnostic vector width drifted")
        float_values = (
            *self.count_loss_sums,
            *self.count_abs_error_sums,
            *self.component_loss_sums,
            *self.o1_loss_sums,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in float_values):
            raise ValueError("Task diagnostic loss/error sums must be finite and non-negative")
        count_values = (
            *self.count_rows,
            *self.component_rows,
            *self.o1_rows,
            *self.channel_positive_counts,
            *self.channel_negative_counts,
            *self.channel_masked_counts,
            *self.channel_true_positive_counts,
            *self.channel_false_positive_counts,
            *self.channel_false_negative_counts,
            self.e1_representable_occurrences,
            self.e1_unrepresentable_occurrences,
        )
        if any(type(value) is not int or value < 0 for value in count_values):
            raise ValueError("Task diagnostic counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class OfficialWeakLossAudit:
    labels_joined_after_forward: bool
    runtime_payload_reused_for_labels: bool
    identity_target_fabricated: bool
    unique_retrieval_id_fabricated: bool
    future_occurrences_ignored: int
    retrieval_bag_sizes: tuple[int, ...]
    retrieval_candidate_counts: tuple[int, ...] = ()
    retrieval_positive_counts: tuple[int, ...] = ()
    retrieval_negative_counts: tuple[int, ...] = ()
    retrieval_wrong_operator_rows: int = 0
    retrieval_target_head_candidate_rows: int = 0
    retrieval_no_target_head_candidate_rows: int = 0
    retrieval_no_candidate_rows: int = 0
    retrieval_no_positive_rows: int = 0
    retrieval_all_positive_rows: int = 0
    retrieval_valid_bag_rows: int = 0
    retrieval_rescued_from_wrong_route_rows: int = 0
    retrieval_legacy_valid_bag_rows: int = 0
    retrieval_invalid_excluded_count: int = 0
    retrieval_ineligible_excluded_count: int = 0
    retrieval_causal_excluded_count: int = 0
    retrieval_candidate_total: int | None = None
    retrieval_positive_total: int | None = None
    retrieval_negative_total: int | None = None
    annotation_count_mismatch: int = 0
    operator_diagnostics: OperatorDiagnosticAudit = field(
        default_factory=OperatorDiagnosticAudit.empty
    )
    task_diagnostics: TaskDiagnosticAudit = field(default_factory=TaskDiagnosticAudit.empty)

    def __post_init__(self) -> None:
        if not isinstance(self.operator_diagnostics, OperatorDiagnosticAudit):
            raise TypeError("official weak audit requires typed Operator diagnostics")
        if not isinstance(self.task_diagnostics, TaskDiagnosticAudit):
            raise TypeError("official weak audit requires typed Task diagnostics")
        if not self.labels_joined_after_forward:
            raise ValueError("official weak labels must be joined only after forward")
        if (
            self.runtime_payload_reused_for_labels
            or self.identity_target_fabricated
            or self.unique_retrieval_id_fabricated
        ):
            raise ValueError("official weak loss audit detected label leakage or pseudo labels")
        if self.future_occurrences_ignored < 0 or any(
            value < 0 for value in self.retrieval_bag_sizes
        ):
            raise ValueError("official weak audit counts must be non-negative")
        counts = (
            self.retrieval_wrong_operator_rows,
            self.retrieval_target_head_candidate_rows,
            self.retrieval_no_target_head_candidate_rows,
            self.retrieval_no_candidate_rows,
            self.retrieval_no_positive_rows,
            self.retrieval_all_positive_rows,
            self.retrieval_valid_bag_rows,
            self.retrieval_rescued_from_wrong_route_rows,
            self.retrieval_legacy_valid_bag_rows,
            self.retrieval_invalid_excluded_count,
            self.retrieval_ineligible_excluded_count,
            self.retrieval_causal_excluded_count,
            self.annotation_count_mismatch,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("official weak retrieval audit row counts must be non-negative")
        aligned = (
            self.retrieval_candidate_counts,
            self.retrieval_positive_counts,
            self.retrieval_negative_counts,
        )
        if any(any(value < 0 for value in values) for values in aligned):
            raise ValueError("official weak retrieval candidate counts must be non-negative")
        non_empty_lengths = {len(values) for values in aligned if values}
        if len(non_empty_lengths) > 1:
            raise ValueError("official weak retrieval count vectors must align")
        optional_totals = (
            self.retrieval_candidate_total,
            self.retrieval_positive_total,
            self.retrieval_negative_total,
        )
        if any(
            value is not None and (type(value) is not int or value < 0) for value in optional_totals
        ):
            raise ValueError("official weak retrieval global totals must be non-negative")

    def metrics(self) -> tuple[tuple[str, float], ...]:
        """Expose bag-validity counts to A2/A5 training logs."""

        return (
            ("retrieval/wrong_operator_rows", float(self.retrieval_wrong_operator_rows)),
            (
                "retrieval/target_head_candidate_rows",
                float(self.retrieval_target_head_candidate_rows),
            ),
            (
                "retrieval/no_target_head_candidate_rows",
                float(self.retrieval_no_target_head_candidate_rows),
            ),
            ("retrieval/no_candidate_rows", float(self.retrieval_no_candidate_rows)),
            ("retrieval/no_positive_rows", float(self.retrieval_no_positive_rows)),
            ("retrieval/all_positive_rows", float(self.retrieval_all_positive_rows)),
            ("retrieval/valid_bag_rows", float(self.retrieval_valid_bag_rows)),
            (
                "retrieval/rescued_from_wrong_route_rows",
                float(self.retrieval_rescued_from_wrong_route_rows),
            ),
            (
                "retrieval/legacy_valid_bag_rows",
                float(self.retrieval_legacy_valid_bag_rows),
            ),
            (
                "retrieval/invalid_excluded_count",
                float(self.retrieval_invalid_excluded_count),
            ),
            (
                "retrieval/ineligible_excluded_count",
                float(self.retrieval_ineligible_excluded_count),
            ),
            (
                "retrieval/causal_excluded_count",
                float(self.retrieval_causal_excluded_count),
            ),
            (
                "retrieval/candidate_count",
                float(
                    sum(self.retrieval_candidate_counts)
                    if self.retrieval_candidate_total is None
                    else self.retrieval_candidate_total
                ),
            ),
            (
                "retrieval/positive_count",
                float(
                    sum(self.retrieval_positive_counts)
                    if self.retrieval_positive_total is None
                    else self.retrieval_positive_total
                ),
            ),
            (
                "retrieval/negative_count",
                float(
                    sum(self.retrieval_negative_counts)
                    if self.retrieval_negative_total is None
                    else self.retrieval_negative_total
                ),
            ),
            ("task/annotation_count_mismatch", float(self.annotation_count_mismatch)),
        )


@dataclass(frozen=True, slots=True)
class OfficialWeakStateLossOutput:
    task: OfficialWeakLossTerm
    operator: OfficialWeakLossTerm
    retrieval: OfficialWeakLossTerm
    time: OfficialWeakLossTerm
    total: Tensor
    audit: OfficialWeakLossAudit

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or self.total.dtype != torch.float32:
            raise ValueError("official weak state total must be an FP32 scalar")
        expected = self.task.value + self.operator.value + self.retrieval.value + self.time.value
        if not torch.allclose(self.total.detach(), expected.detach(), atol=1.0e-7, rtol=1.0e-7):
            raise ValueError("official weak L_state must equal task+operator+retrieval+time")


class OfficialWeakTargetBuilder:
    """Build official weak losses from predictions, never runtime inputs or hard-state writes."""

    __slots__ = ()

    def __call__(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        supervision: Sequence[OfficialWeakSupervision],
    ) -> OfficialWeakStateLossOutput:
        return self.build(observations, query, retrieval, supervision)

    def build(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        supervision: Sequence[OfficialWeakSupervision],
    ) -> OfficialWeakStateLossOutput:
        batch_size, _ = _validate_builder_inputs(
            observations,
            query,
            retrieval,
            StageATargetBatch(),
        )
        labels = tuple(supervision)
        if len(labels) != batch_size or any(
            not isinstance(label, OfficialWeakSupervision) for label in labels
        ):
            raise ValueError("official weak supervision must align to the prediction batch")
        # Every soft head participates in every rank's differentiable graph even when the
        # official weak label masks a term. This preserves the exact numerical objective while
        # giving ZeRO-2 a stable, identical parameter-hook surface across mixed task classes.
        anchor = (
            observations.o1.logits.float().sum()
            + observations.o2.identity.float().sum()
            + observations.o2.score_logits.float().sum()
            + observations.o2.count_prediction.float().sum()
            + observations.e1.logits.float().sum()
            + observations.e1.count_prediction.float().sum()
            + observations.e2.event_logits.float().sum()
            + observations.e2.phase_logits.float().sum()
            + observations.e2.count_prediction.float().sum()
            + query.route.logits.float().sum()
            + query.time.logits.mode_logits.float().sum()
            + query.time.logits.span_start_logits.float().sum()
            + query.time.logits.span_end_logits.float().sum()
            + retrieval.state_embeddings.float().sum()
            + retrieval.scores.float().sum()
        ) * 0.0
        task_losses: list[Tensor] = []
        operator_losses: list[Tensor] = []
        retrieval_losses: list[Tensor] = []
        time_losses: list[Tensor] = []
        future_ignored = 0
        bag_sizes: list[int] = []
        candidate_counts: list[int] = []
        positive_counts: list[int] = []
        negative_counts: list[int] = []
        retrieval_status_counts = {
            "no_candidate": 0,
            "no_positive": 0,
            "all_positive": 0,
            "valid_bag": 0,
        }
        retrieval_wrong_operator_rows = 0
        retrieval_target_head_candidate_rows = 0
        retrieval_no_target_head_candidate_rows = 0
        retrieval_rescued_from_wrong_route_rows = 0
        retrieval_legacy_valid_bag_rows = 0
        retrieval_invalid_excluded_count = 0
        retrieval_ineligible_excluded_count = 0
        retrieval_causal_excluded_count = 0
        annotation_count_mismatch = 0
        raw_operator_confusion = [0] * 72
        effective_operator_confusion = [0] * 72
        operator_loss_sums = [0.0] * 8
        operator_support = [0] * 8
        operator_confidence_sum = 0.0
        operator_entropy_sum = 0.0
        operator_temperature_sum = 0.0
        operator_temperature_count = 0
        task_count_loss_sums = [0.0] * 4
        task_count_abs_error_sums = [0.0] * 4
        task_count_rows = [0] * 4
        task_component_loss_sums = [0.0] * 3
        task_component_rows = [0] * 3
        task_o1_loss_sums = [0.0] * 2
        task_o1_rows = [0] * 2
        task_channel_positive = [0] * 7
        task_channel_negative = [0] * 7
        task_channel_masked = [0] * 7
        task_channel_tp = [0] * 7
        task_channel_fp = [0] * 7
        task_channel_fn = [0] * 7
        e1_representable_occurrences = 0
        e1_unrepresentable_occurrences = 0

        for row, label in enumerate(labels):
            operator_index = OPERATORS.index(label.operator)
            operator_target = torch.tensor(
                [operator_index], dtype=torch.int64, device=query.route.logits.device
            )
            row_operator_loss = F.cross_entropy(
                query.route.logits[row : row + 1].float(), operator_target
            )
            operator_losses.append(row_operator_loss)
            raw_index = int(query.route.raw_indices[row].detach().item())
            effective_index = OPERATORS.index(query.hard_operators[row])
            raw_operator_confusion[operator_index * 9 + raw_index] += 1
            effective_operator_confusion[operator_index * 9 + effective_index] += 1
            operator_support[operator_index] += 1
            operator_loss_sums[operator_index] += float(row_operator_loss.detach().item())
            operator_confidence_sum += float(query.route.confidence[row].detach().item())
            probabilities = torch.softmax(query.route.logits[row].detach().float(), dim=-1)
            operator_entropy_sum += float(
                (-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()).item()
            )
            if query.route.temperature is not None:
                operator_temperature_sum += float(query.route.temperature.detach().item())
                operator_temperature_count += 1

            mode_index = TIME_MODES.index(label.time_mode)
            mode_target = torch.tensor(
                [mode_index], dtype=torch.int64, device=query.time.logits.mode_logits.device
            )
            row_time = F.cross_entropy(
                query.time.logits.mode_logits[row : row + 1].float(), mode_target
            )
            if label.numeric_token_span is not None:
                start, end = label.numeric_token_span
                valid_tokens = ~query.time.logits.padding_mask[row]
                if end >= valid_tokens.shape[0] or not bool(
                    valid_tokens[start] & valid_tokens[end]
                ):
                    raise ValueError(
                        "official weak numeric span targets must point to valid tokens"
                    )
                start_target = torch.tensor(
                    [start], dtype=torch.int64, device=query.time.logits.span_start_logits.device
                )
                end_target = torch.tensor(
                    [end], dtype=torch.int64, device=query.time.logits.span_end_logits.device
                )
                row_time = (
                    row_time
                    + F.cross_entropy(
                        query.time.logits.span_start_logits[row : row + 1].float(),
                        start_target,
                    )
                    + F.cross_entropy(
                        query.time.logits.span_end_logits[row : row + 1].float(),
                        end_target,
                    )
                )
            time_losses.append(row_time)

            task_result = _official_weak_task_result(observations, row, label)
            task_losses.append(task_result.loss)
            family = task_result.family_index
            task_count_loss_sums[family] += float(task_result.count_loss.detach().item())
            task_count_abs_error_sums[family] += task_result.count_abs_error
            task_count_rows[family] += 1
            if task_result.component_index is not None and task_result.component_loss is not None:
                component = task_result.component_index
                task_component_loss_sums[component] += float(
                    task_result.component_loss.detach().item()
                )
                task_component_rows[component] += 1
            if task_result.phase_loss is not None:
                task_component_loss_sums[2] += float(task_result.phase_loss.detach().item())
                task_component_rows[2] += 1
            if task_result.o1_subtype_index is not None:
                subtype = task_result.o1_subtype_index
                task_o1_loss_sums[subtype] += float(task_result.loss.detach().item())
                task_o1_rows[subtype] += 1
            for target, source in (
                (task_channel_positive, task_result.channel_positive_counts),
                (task_channel_negative, task_result.channel_negative_counts),
                (task_channel_masked, task_result.channel_masked_counts),
                (task_channel_tp, task_result.channel_true_positive_counts),
                (task_channel_fp, task_result.channel_false_positive_counts),
                (task_channel_fn, task_result.channel_false_negative_counts),
            ):
                for index, value in enumerate(source):
                    target[index] += value
            e1_representable_occurrences += task_result.e1_representable_occurrences
            e1_unrepresentable_occurrences += task_result.e1_unrepresentable_occurrences
            annotation_count_mismatch += int(_official_count_mismatch(label))
            retrieval_result = _official_weak_retrieval_loss(retrieval, row, label)
            retrieval_loss = retrieval_result.loss
            positives = retrieval_result.positive_count
            candidates = retrieval_result.candidate_count
            negatives = retrieval_result.negative_count
            bag_size = positives
            bag_sizes.append(bag_size)
            candidate_counts.append(candidates)
            positive_counts.append(positives)
            negative_counts.append(negatives)
            retrieval_status_counts[retrieval_result.status] += 1
            retrieval_wrong_operator_rows += int(retrieval_result.wrong_operator)
            retrieval_target_head_candidate_rows += int(
                retrieval_result.target_head_present_count > 0
            )
            retrieval_no_target_head_candidate_rows += int(
                retrieval_result.target_head_present_count == 0
            )
            retrieval_rescued_from_wrong_route_rows += int(retrieval_result.rescued_wrong_route)
            retrieval_legacy_valid_bag_rows += int(retrieval_result.legacy_valid_bag)
            retrieval_invalid_excluded_count += retrieval_result.invalid_excluded_count
            retrieval_ineligible_excluded_count += retrieval_result.ineligible_excluded_count
            retrieval_causal_excluded_count += retrieval_result.causal_excluded_count
            if retrieval_loss is not None:
                retrieval_losses.append(retrieval_loss)
            future_ignored += sum(point > label.query_time for point in label.occurrence_points)
            future_ignored += sum(
                start > label.query_time or end > label.query_time
                for start, end in label.occurrence_intervals
            )

        audit_device = query.route.logits.device
        operator_integer_values = _distributed_sum_integers(
            (
                *raw_operator_confusion,
                *effective_operator_confusion,
                *operator_support,
                operator_temperature_count,
            ),
            audit_device,
        )
        raw_operator_confusion = list(operator_integer_values[:72])
        effective_operator_confusion = list(operator_integer_values[72:144])
        operator_support = list(operator_integer_values[144:152])
        operator_temperature_count = operator_integer_values[152]
        operator_float_values = _distributed_sum_floats(
            (
                *operator_loss_sums,
                operator_confidence_sum,
                operator_entropy_sum,
                operator_temperature_sum,
            ),
            audit_device,
        )
        operator_loss_sums = list(operator_float_values[:8])
        (
            operator_confidence_sum,
            operator_entropy_sum,
            operator_temperature_sum,
        ) = operator_float_values[8:]

        task_integer_values = _distributed_sum_integers(
            (
                *task_count_rows,
                *task_component_rows,
                *task_o1_rows,
                *task_channel_positive,
                *task_channel_negative,
                *task_channel_masked,
                *task_channel_tp,
                *task_channel_fp,
                *task_channel_fn,
                e1_representable_occurrences,
                e1_unrepresentable_occurrences,
            ),
            audit_device,
        )
        offset = 0

        def take_ints(width: int) -> list[int]:
            nonlocal offset
            values = list(task_integer_values[offset : offset + width])
            offset += width
            return values

        task_count_rows = take_ints(4)
        task_component_rows = take_ints(3)
        task_o1_rows = take_ints(2)
        task_channel_positive = take_ints(7)
        task_channel_negative = take_ints(7)
        task_channel_masked = take_ints(7)
        task_channel_tp = take_ints(7)
        task_channel_fp = take_ints(7)
        task_channel_fn = take_ints(7)
        e1_representable_occurrences, e1_unrepresentable_occurrences = take_ints(2)

        task_float_values = _distributed_sum_floats(
            (
                *task_count_loss_sums,
                *task_count_abs_error_sums,
                *task_component_loss_sums,
                *task_o1_loss_sums,
            ),
            audit_device,
        )
        task_count_loss_sums = list(task_float_values[:4])
        task_count_abs_error_sums = list(task_float_values[4:8])
        task_component_loss_sums = list(task_float_values[8:11])
        task_o1_loss_sums = list(task_float_values[11:13])

        retrieval_global_values = _distributed_sum_integers(
            (
                retrieval_wrong_operator_rows,
                retrieval_target_head_candidate_rows,
                retrieval_no_target_head_candidate_rows,
                retrieval_status_counts["no_candidate"],
                retrieval_status_counts["no_positive"],
                retrieval_status_counts["all_positive"],
                retrieval_status_counts["valid_bag"],
                retrieval_rescued_from_wrong_route_rows,
                retrieval_legacy_valid_bag_rows,
                retrieval_invalid_excluded_count,
                retrieval_ineligible_excluded_count,
                retrieval_causal_excluded_count,
                sum(candidate_counts),
                sum(positive_counts),
                sum(negative_counts),
                annotation_count_mismatch,
            ),
            audit_device,
        )
        (
            retrieval_wrong_operator_rows,
            retrieval_target_head_candidate_rows,
            retrieval_no_target_head_candidate_rows,
            retrieval_status_counts["no_candidate"],
            retrieval_status_counts["no_positive"],
            retrieval_status_counts["all_positive"],
            retrieval_status_counts["valid_bag"],
            retrieval_rescued_from_wrong_route_rows,
            retrieval_legacy_valid_bag_rows,
            retrieval_invalid_excluded_count,
            retrieval_ineligible_excluded_count,
            retrieval_causal_excluded_count,
            retrieval_candidate_total,
            retrieval_positive_total,
            retrieval_negative_total,
            annotation_count_mismatch,
        ) = retrieval_global_values

        task = _official_weak_term(task_losses, anchor)
        operator = _official_weak_term(operator_losses, anchor)
        retrieval_term = _official_weak_term(retrieval_losses, anchor)
        time = _official_weak_term(time_losses, anchor)
        total = task.value + operator.value + retrieval_term.value + time.value
        return OfficialWeakStateLossOutput(
            task=task,
            operator=operator,
            retrieval=retrieval_term,
            time=time,
            total=total,
            audit=OfficialWeakLossAudit(
                labels_joined_after_forward=True,
                runtime_payload_reused_for_labels=False,
                identity_target_fabricated=False,
                unique_retrieval_id_fabricated=False,
                future_occurrences_ignored=future_ignored,
                retrieval_bag_sizes=tuple(bag_sizes),
                retrieval_candidate_counts=tuple(candidate_counts),
                retrieval_positive_counts=tuple(positive_counts),
                retrieval_negative_counts=tuple(negative_counts),
                retrieval_wrong_operator_rows=retrieval_wrong_operator_rows,
                retrieval_target_head_candidate_rows=retrieval_target_head_candidate_rows,
                retrieval_no_target_head_candidate_rows=(retrieval_no_target_head_candidate_rows),
                retrieval_no_candidate_rows=retrieval_status_counts["no_candidate"],
                retrieval_no_positive_rows=retrieval_status_counts["no_positive"],
                retrieval_all_positive_rows=retrieval_status_counts["all_positive"],
                retrieval_valid_bag_rows=retrieval_status_counts["valid_bag"],
                retrieval_rescued_from_wrong_route_rows=(retrieval_rescued_from_wrong_route_rows),
                retrieval_legacy_valid_bag_rows=retrieval_legacy_valid_bag_rows,
                retrieval_invalid_excluded_count=retrieval_invalid_excluded_count,
                retrieval_ineligible_excluded_count=retrieval_ineligible_excluded_count,
                retrieval_causal_excluded_count=retrieval_causal_excluded_count,
                retrieval_candidate_total=retrieval_candidate_total,
                retrieval_positive_total=retrieval_positive_total,
                retrieval_negative_total=retrieval_negative_total,
                annotation_count_mismatch=annotation_count_mismatch,
                operator_diagnostics=OperatorDiagnosticAudit(
                    raw_confusion=tuple(raw_operator_confusion),
                    effective_confusion=tuple(effective_operator_confusion),
                    class_loss_sums=tuple(operator_loss_sums),
                    class_support=tuple(operator_support),
                    confidence_sum=operator_confidence_sum,
                    entropy_sum=operator_entropy_sum,
                    temperature_sum=operator_temperature_sum,
                    temperature_count=operator_temperature_count,
                ),
                task_diagnostics=TaskDiagnosticAudit(
                    count_loss_sums=tuple(task_count_loss_sums),
                    count_abs_error_sums=tuple(task_count_abs_error_sums),
                    count_rows=tuple(task_count_rows),
                    component_loss_sums=tuple(task_component_loss_sums),
                    component_rows=tuple(task_component_rows),
                    o1_loss_sums=tuple(task_o1_loss_sums),
                    o1_rows=tuple(task_o1_rows),
                    channel_positive_counts=tuple(task_channel_positive),
                    channel_negative_counts=tuple(task_channel_negative),
                    channel_masked_counts=tuple(task_channel_masked),
                    channel_true_positive_counts=tuple(task_channel_tp),
                    channel_false_positive_counts=tuple(task_channel_fp),
                    channel_false_negative_counts=tuple(task_channel_fn),
                    e1_representable_occurrences=e1_representable_occurrences,
                    e1_unrepresentable_occurrences=e1_unrepresentable_occurrences,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _TaskLossResult:
    loss: Tensor
    family_index: int
    count_loss: Tensor
    count_abs_error: float
    component_index: int | None = None
    component_loss: Tensor | None = None
    phase_loss: Tensor | None = None
    o1_subtype_index: int | None = None
    channel_positive_counts: tuple[int, ...] = (0,) * 7
    channel_negative_counts: tuple[int, ...] = (0,) * 7
    channel_masked_counts: tuple[int, ...] = (0,) * 7
    channel_true_positive_counts: tuple[int, ...] = (0,) * 7
    channel_false_positive_counts: tuple[int, ...] = (0,) * 7
    channel_false_negative_counts: tuple[int, ...] = (0,) * 7
    e1_representable_occurrences: int = 0
    e1_unrepresentable_occurrences: int = 0


def _official_weak_task_loss(
    observations: ObservationOutputs,
    row: int,
    label: OfficialWeakSupervision,
) -> Tensor:
    """Backward-compatible scalar entry point; diagnostics use the typed result below."""

    return _official_weak_task_result(observations, row, label).loss


def _official_weak_task_result(
    observations: ObservationOutputs,
    row: int,
    label: OfficialWeakSupervision,
) -> _TaskLossResult:
    target_count = torch.tensor(
        float(label.count), dtype=torch.float32, device=observations.o1.logits.device
    )
    if label.operator in (Operator.O1_SNAP, Operator.O1_DELTA):
        prediction = observations.o1.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        return _TaskLossResult(
            loss=count_loss,
            family_index=0,
            count_loss=count_loss,
            count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
            o1_subtype_index=(0 if label.operator is Operator.O1_SNAP else 1),
        )
    if label.operator in (Operator.O2_UNIQUE, Operator.O2_GAIN):
        prediction = observations.o2.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        return _TaskLossResult(
            loss=count_loss,
            family_index=1,
            count_loss=count_loss,
            count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
        )
    if label.operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
        valid = observations.e1.valid_mask[row]
        prediction = observations.e1.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        if not bool(valid.any().item()):
            dense = observations.e1.logits[row].float().sum() * 0.0
            return _TaskLossResult(
                loss=(dense + count_loss) / 2.0,
                family_index=2,
                count_loss=count_loss,
                count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
                component_index=0,
                component_loss=dense,
            )
        logits = observations.e1.logits[row][valid].float()
        timestamps = observations.e1.timestamps[row][valid]
        targets, channel_mask, representable, unrepresentable = _build_e1_fsm_targets(
            timestamps,
            label.occurrence_points,
            label.query_time,
        )
        dense, stats = _balanced_dense_bce(logits, targets, channel_mask)
        return _TaskLossResult(
            loss=(dense + count_loss) / 2.0,
            family_index=2,
            count_loss=count_loss,
            count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
            component_index=0,
            component_loss=dense,
            channel_positive_counts=(*stats.positive_counts, 0, 0, 0, 0),
            channel_negative_counts=(*stats.negative_counts, 0, 0, 0, 0),
            channel_masked_counts=(*stats.masked_counts, 0, 0, 0, 0),
            channel_true_positive_counts=(*stats.true_positive_counts, 0, 0, 0, 0),
            channel_false_positive_counts=(*stats.false_positive_counts, 0, 0, 0, 0),
            channel_false_negative_counts=(*stats.false_negative_counts, 0, 0, 0, 0),
            e1_representable_occurrences=representable,
            e1_unrepresentable_occurrences=unrepresentable,
        )
    if label.operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
        valid = observations.e2.valid_mask[row]
        prediction = observations.e2.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        if not bool(valid.any().item()):
            dense_zero = observations.e2.event_logits[row].float().sum() * 0.0
            phase_zero = observations.e2.phase_logits[row].float().sum() * 0.0
            return _TaskLossResult(
                loss=(dense_zero + phase_zero + count_loss) / 3.0,
                family_index=3,
                count_loss=count_loss,
                count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
                component_index=1,
                component_loss=dense_zero,
                phase_loss=phase_zero,
            )
        event_logits = observations.e2.event_logits[row][valid].float()
        phase_logits = observations.e2.phase_logits[row][valid].float()
        timestamps = observations.e2.timestamps[row][valid]
        event_targets = torch.zeros_like(event_logits)
        phase_targets = torch.zeros(
            timestamps.shape[0], dtype=torch.int64, device=timestamps.device
        )
        for start, end in label.occurrence_intervals:
            if start > label.query_time:
                continue
            causal_end = min(end, label.query_time)
            tail_start, tail_end = _voronoi_timestamp_bounds(timestamps)
            if causal_end < tail_start or start > tail_end:
                continue
            active = (timestamps >= start) & (timestamps <= causal_end)
            event_targets[active, 1] = 1.0
            phase_targets[active] = 1
            start_index = _voronoi_timestamp_index(timestamps, start)
            if start_index is not None:
                event_targets[start_index, 0] = 1.0
                phase_targets[start_index] = 1
            if end <= label.query_time:
                end_index = _voronoi_timestamp_index(timestamps, end)
                if end_index is not None:
                    event_targets[end_index, 2:] = 1.0
                    phase_targets[end_index] = 3
                    completed = timestamps > end
                    phase_targets[completed] = 3
        dense, stats = _balanced_dense_bce(
            event_logits,
            event_targets,
            torch.ones_like(event_targets, dtype=torch.bool),
        )
        phase = F.cross_entropy(phase_logits, phase_targets)
        return _TaskLossResult(
            loss=(dense + phase + count_loss) / 3.0,
            family_index=3,
            count_loss=count_loss,
            count_abs_error=float((prediction.detach().float() - target_count).abs().item()),
            component_index=1,
            component_loss=dense,
            phase_loss=phase,
            channel_positive_counts=(0, 0, 0, *stats.positive_counts),
            channel_negative_counts=(0, 0, 0, *stats.negative_counts),
            channel_masked_counts=(0, 0, 0, *stats.masked_counts),
            channel_true_positive_counts=(0, 0, 0, *stats.true_positive_counts),
            channel_false_positive_counts=(0, 0, 0, *stats.false_positive_counts),
            channel_false_negative_counts=(0, 0, 0, *stats.false_negative_counts),
        )
    raise ValueError(f"unsupported official weak operator: {label.operator}")


@dataclass(frozen=True, slots=True)
class _DenseBCEStats:
    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    masked_counts: tuple[int, ...]
    true_positive_counts: tuple[int, ...]
    false_positive_counts: tuple[int, ...]
    false_negative_counts: tuple[int, ...]


def _balanced_dense_bce(
    logits: Tensor,
    targets: Tensor,
    supervision_mask: Tensor,
) -> tuple[Tensor, _DenseBCEStats]:
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("balanced dense BCE requires aligned [T, C] logits and targets")
    if supervision_mask.shape != logits.shape or supervision_mask.dtype != torch.bool:
        raise ValueError("balanced dense BCE supervision mask must be bool [T, C]")
    channel_losses: list[Tensor] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    masked_counts: list[int] = []
    true_positive_counts: list[int] = []
    false_positive_counts: list[int] = []
    false_negative_counts: list[int] = []
    for channel in range(logits.shape[1]):
        mask = supervision_mask[:, channel]
        channel_logits = logits[:, channel][mask]
        channel_targets = targets[:, channel][mask]
        positive = channel_targets >= 0.5
        negative = ~positive
        positive_count = int(positive.sum().item())
        negative_count = int(negative.sum().item())
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)
        masked_counts.append(int((~mask).sum().item()))
        if channel_logits.numel():
            losses = F.binary_cross_entropy_with_logits(
                channel_logits,
                channel_targets,
                reduction="none",
            )
            if positive_count and negative_count:
                channel_losses.append(0.5 * losses[positive].mean() + 0.5 * losses[negative].mean())
            else:
                channel_losses.append(losses.mean())
            predicted = channel_logits.detach() >= 0.0
            true_positive_counts.append(int((predicted & positive).sum().item()))
            false_positive_counts.append(int((predicted & negative).sum().item()))
            false_negative_counts.append(int((~predicted & positive).sum().item()))
        else:
            true_positive_counts.append(0)
            false_positive_counts.append(0)
            false_negative_counts.append(0)
    loss = torch.stack(channel_losses).mean() if channel_losses else logits.sum() * 0.0
    return loss, _DenseBCEStats(
        positive_counts=tuple(positive_counts),
        negative_counts=tuple(negative_counts),
        masked_counts=tuple(masked_counts),
        true_positive_counts=tuple(true_positive_counts),
        false_positive_counts=tuple(false_positive_counts),
        false_negative_counts=tuple(false_negative_counts),
    )


def _build_e1_fsm_targets(
    timestamps: Tensor,
    occurrence_points: Sequence[float],
    query_time: float,
) -> tuple[Tensor, Tensor, int, int]:
    """Build onset then completion+transition targets that the hard FSM can realize."""

    targets = torch.zeros((timestamps.shape[0], 3), dtype=torch.float32, device=timestamps.device)
    supervision_mask = torch.ones_like(targets, dtype=torch.bool)
    claimed_positions: set[int] = set()
    representable = 0
    unrepresentable = 0
    for point in sorted(point for point in occurrence_points if point <= query_time):
        completion_index = _voronoi_timestamp_index(timestamps, point)
        if completion_index is None:
            continue
        onset_index = completion_index - 1
        if onset_index < 0:
            supervision_mask[completion_index] = False
            unrepresentable += 1
            continue
        if onset_index in claimed_positions or completion_index in claimed_positions:
            supervision_mask[onset_index, 0] = False
            supervision_mask[completion_index, 1:] = False
            unrepresentable += 1
            continue
        targets[onset_index, 0] = 1.0
        targets[completion_index, 1:] = 1.0
        claimed_positions.update((onset_index, completion_index))
        representable += 1
    return targets, supervision_mask, representable, unrepresentable


@dataclass(frozen=True, slots=True)
class _RetrievalLossResult:
    loss: Tensor | None
    positive_count: int
    candidate_count: int
    negative_count: int
    status: str
    wrong_operator: bool
    rescued_wrong_route: bool
    legacy_valid_bag: bool
    target_head_present_count: int
    invalid_excluded_count: int
    ineligible_excluded_count: int
    causal_excluded_count: int


def _official_weak_retrieval_loss(
    retrieval: RetrieverOutput,
    row: int,
    label: OfficialWeakSupervision,
) -> _RetrievalLossResult:
    wrong_operator = retrieval.hard_operators[row] is not label.operator
    target_head = OPERATOR_TO_HEAD_TYPE[label.operator]
    if target_head is None:
        raise ValueError("official weak Retrieval requires a supported target head")
    assert retrieval.candidate_head_codes is not None
    assert retrieval.candidate_timestamps is not None
    assert retrieval.candidate_time_ranges is not None
    head_mask = (
        retrieval.candidate_head_codes[row] == RETRIEVAL_HEAD_ORDER.index(target_head)
    ) & retrieval.present_mask[row]
    record_end = torch.where(
        retrieval.candidate_timestamps[row] >= 0.0,
        retrieval.candidate_timestamps[row],
        retrieval.candidate_time_ranges[row, :, 1],
    )
    official_causal = retrieval.present_mask[row] & (record_end <= label.query_time)
    invalid_excluded = int((head_mask & ~retrieval.record_valid_mask[row]).sum().item())
    valid_head = head_mask & retrieval.record_valid_mask[row]
    ineligible_excluded = int((valid_head & ~retrieval.retrieval_eligible_mask[row]).sum().item())
    eligible_head = valid_head & retrieval.retrieval_eligible_mask[row]
    causal_excluded = int((eligible_head & ~official_causal).sum().item())
    candidate_mask = eligible_head & official_causal
    present_columns = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    candidate_count = int(present_columns.numel())
    legacy_valid = _legacy_retrieval_bag_is_valid(retrieval, row, label)
    common = {
        "legacy_valid_bag": legacy_valid,
        "target_head_present_count": int(head_mask.sum().item()),
        "invalid_excluded_count": invalid_excluded,
        "ineligible_excluded_count": ineligible_excluded,
        "causal_excluded_count": causal_excluded,
    }
    if not candidate_count:
        return _RetrievalLossResult(None, 0, 0, 0, "no_candidate", wrong_operator, False, **common)
    positive_mask = _retrieval_occurrence_mask(retrieval, row, label) & candidate_mask
    positive_columns = torch.nonzero(positive_mask, as_tuple=False).flatten()
    positive_count = int(positive_columns.numel())
    if not positive_count:
        return _RetrievalLossResult(
            None,
            0,
            candidate_count,
            candidate_count,
            "no_positive",
            wrong_operator,
            False,
            **common,
        )
    negative_count = candidate_count - positive_count
    if negative_count == 0:
        return _RetrievalLossResult(
            None,
            positive_count,
            candidate_count,
            0,
            "all_positive",
            wrong_operator,
            False,
            **common,
        )
    all_logits = retrieval.scores[row].index_select(0, present_columns).float()
    positive_logits = retrieval.scores[row].index_select(0, positive_columns).float()
    loss = torch.logsumexp(all_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
    return _RetrievalLossResult(
        loss,
        positive_count,
        candidate_count,
        negative_count,
        "valid_bag",
        wrong_operator,
        wrong_operator,
        **common,
    )


def _legacy_retrieval_bag_is_valid(
    retrieval: RetrieverOutput,
    row: int,
    label: OfficialWeakSupervision,
) -> bool:
    if retrieval.hard_operators[row] is not label.operator:
        return False
    mask = (
        retrieval.present_mask[row]
        & retrieval.predicted_head_mask[row]
        & retrieval.record_valid_mask[row]
        & retrieval.retrieval_eligible_mask[row]
        & retrieval.causal_mask[row]
    )
    positives = int((_retrieval_occurrence_mask(retrieval, row, label) & mask).sum().item())
    count = int(mask.sum().item())
    return bool(positives > 0 and positives < count)


def _retrieval_occurrence_mask(
    retrieval: RetrieverOutput,
    row: int,
    label: OfficialWeakSupervision,
) -> Tensor:
    assert retrieval.candidate_timestamps is not None
    assert retrieval.candidate_time_ranges is not None
    timestamps = retrieval.candidate_timestamps[row]
    ranges = retrieval.candidate_time_ranges[row]
    is_point = timestamps >= 0.0
    matched = torch.zeros_like(retrieval.present_mask[row])
    for point in label.occurrence_points:
        if point > label.query_time:
            continue
        matched |= is_point & ((timestamps - point).abs() <= 0.5)
        matched |= ~is_point & (ranges[:, 0] <= point) & (point <= ranges[:, 1])
    for start, end in label.occurrence_intervals:
        if start > label.query_time:
            continue
        causal_end = min(end, label.query_time)
        matched |= is_point & (timestamps >= start) & (timestamps <= causal_end)
        matched |= ~is_point & (ranges[:, 0] <= causal_end) & (start <= ranges[:, 1])
    return matched & retrieval.present_mask[row]


def _record_end_time(record: object) -> float:
    timestamp = getattr(record, "timestamp", None)
    if timestamp is not None:
        return float(timestamp)
    time_range = getattr(record, "time_range", None)
    if time_range is None:
        raise ValueError("Retrieval candidate requires timestamp or time_range")
    return float(time_range[1])


def _record_matches_causal_occurrence(
    record: object,
    label: OfficialWeakSupervision,
) -> bool:
    timestamp = getattr(record, "timestamp", None)
    time_range = getattr(record, "time_range", None)
    points = tuple(point for point in label.occurrence_points if point <= label.query_time)
    intervals = tuple(
        (start, min(end, label.query_time))
        for start, end in label.occurrence_intervals
        if start <= label.query_time
    )
    if timestamp is not None:
        value = float(timestamp)
        return any(abs(value - point) <= 0.5 for point in points) or any(
            start <= value <= end for start, end in intervals
        )
    if time_range is None:
        return False
    record_start, record_end = (float(value) for value in time_range)
    return any(record_start <= point <= record_end for point in points) or any(
        record_start <= end and start <= record_end for start, end in intervals
    )


def _robust_count_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(
        torch.log1p(prediction.float()),
        torch.log1p(target.float()),
        beta=0.25,
    )


def _voronoi_timestamp_index(timestamps: Tensor, target: float) -> int | None:
    if not timestamps.numel():
        return None
    values = timestamps.float()
    lower, upper = _voronoi_timestamp_bounds(values)
    if target < lower or target > upper:
        return None
    if values.numel() == 1:
        return 0
    midpoints = (values[:-1] + values[1:]) / 2.0
    return int(torch.searchsorted(midpoints, target, right=False).item())


def _voronoi_timestamp_bounds(timestamps: Tensor) -> tuple[float, float]:
    if not timestamps.numel():
        raise ValueError("Voronoi timestamp bounds require at least one timestamp")
    values = timestamps.float()
    if values.numel() == 1:
        value = float(values[0].item())
        return value, value
    lower = values[0] - (values[1] - values[0]) / 2.0
    upper = values[-1] + (values[-1] - values[-2]) / 2.0
    return max(0.0, float(lower.item())), float(upper.item())


def _official_count_mismatch(label: OfficialWeakSupervision) -> bool:
    if label.operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
        derived = sum(point <= label.query_time for point in label.occurrence_points)
        return derived != label.count
    if label.operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
        derived = sum(end <= label.query_time for _, end in label.occurrence_intervals)
        return derived != label.count
    return False


def _official_weak_term(losses: Sequence[Tensor], anchor: Tensor) -> OfficialWeakLossTerm:
    values = tuple(loss.float() for loss in losses)
    value = torch.stack(values).mean() if values else anchor
    return OfficialWeakLossTerm(value=value, valid_rows=len(values))


def _distributed_sum_integers(
    values: Sequence[int],
    device: torch.device,
) -> tuple[int, ...]:
    tensor = torch.tensor(tuple(values), dtype=torch.int64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tuple(int(value) for value in tensor.cpu().tolist())


def _distributed_sum_floats(
    values: Sequence[float],
    device: torch.device,
) -> tuple[float, ...]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tuple(float(value) for value in tensor.cpu().tolist())


__all__ = [
    "E1TargetLabels",
    "E2TargetLabels",
    "OfficialWeakLossAudit",
    "OfficialWeakLossTerm",
    "OfficialWeakStateLossOutput",
    "OfficialWeakSupervision",
    "OfficialWeakTargetBuilder",
    "O1TargetLabels",
    "O2TargetLabels",
    "OperatorDiagnosticAudit",
    "AnswerTargetLabels",
    "QueryTargetLabels",
    "RetrievalTargetLabels",
    "StageATargetBatch",
    "StageATargetBuilder",
    "TargetProvenance",
    "TaskDiagnosticAudit",
]
