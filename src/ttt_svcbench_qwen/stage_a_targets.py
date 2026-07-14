"""Training-only explicit target assembly for Stage A.

Inputs: typed model predictions plus label-only dataclasses with explicit provenance.
Outputs: P14 ``StateLossInput`` values that preserve the prediction autograd graph.
Forbidden: deriving dense labels from an answer, final count, occurrence times, or runtime data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

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
    QueryEncoderOutput,
)
from ttt_svcbench_qwen.state_bank import HeadType
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


class TargetProvenance(StrEnum):
    """The complete and intentionally closed set of Stage A label origins."""

    OFFICIAL_EXPLICIT = "official_explicit"
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
            candidates = retrieval.candidate_record_ids[row]
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


def build_stage_a_targets(
    observations: ObservationOutputs,
    query: QueryEncoderOutput,
    retrieval: RetrieverOutput,
    labels: StageATargetBatch,
) -> StateLossInput:
    """Functional convenience wrapper around :class:`StageATargetBuilder`."""

    return StageATargetBuilder().build(observations, query, retrieval, labels)


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

    # Frozen dataclasses still contain mutable tensors; revalidate at the consumption boundary.
    observations.o1.__post_init__()
    observations.o2.__post_init__()
    observations.e1.__post_init__()
    observations.e2.__post_init__()
    observations.__post_init__()
    query.embeddings.__post_init__()
    query.route.__post_init__()
    query.time.logits.__post_init__()
    query.time.__post_init__()
    query.__post_init__()
    retrieval.validate_integrity()

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


# Short aliases keep the public vocabulary discoverable without creating more provenance values.
Provenance = TargetProvenance
O1Labels = O1TargetLabels
O2Labels = O2TargetLabels
E1Labels = E1TargetLabels
E2Labels = E2TargetLabels
QueryLabels = QueryTargetLabels
RetrievalLabels = RetrievalTargetLabels
AnswerLabels = AnswerTargetLabels
StageATargetLabels = StageATargetBatch


__all__ = [
    "E1Labels",
    "E1TargetLabels",
    "E2Labels",
    "E2TargetLabels",
    "O1Labels",
    "O1TargetLabels",
    "O2Labels",
    "O2TargetLabels",
    "AnswerLabels",
    "AnswerTargetLabels",
    "Provenance",
    "QueryLabels",
    "QueryTargetLabels",
    "RetrievalLabels",
    "RetrievalTargetLabels",
    "StageATargetBatch",
    "StageATargetBuilder",
    "StageATargetLabels",
    "TargetProvenance",
    "build_stage_a_targets",
]
