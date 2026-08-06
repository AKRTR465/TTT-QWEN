"""Training-only explicit target assembly for Stage A.

Inputs: typed model predictions plus label-only dataclasses with explicit provenance.
Outputs: P14 ``StateLossInput`` values that preserve the prediction autograd graph.
Forbidden: deriving dense labels from an answer, final count, occurrence times, or runtime data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.identity_bank import IDENTITY_DIM, IdentityBankRuntimeState
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
from ttt_svcbench_qwen.observation_heads import O2SoftOutput, ObservationOutputs
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    OPERATORS,
    TIME_MODES,
    Operator,
    QueryEncoderOutput,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import RETRIEVAL_HEAD_ORDER
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


@dataclass(frozen=True, slots=True)
class O2TargetLabels:
    """Pre-matched O2 identity plus novelty/match-confidence labels."""

    row_indices: Tensor
    identity_targets: Tensor
    score_targets: Tensor
    slot_mask: Tensor
    provenance: tuple[TargetProvenance, ...]


@dataclass(frozen=True, slots=True)
class E1TargetLabels:
    """Dense E1 eventness/completion/transition labels."""

    row_indices: Tensor
    targets: Tensor
    time_mask: Tensor
    provenance: tuple[TargetProvenance, ...]


@dataclass(frozen=True, slots=True)
class E2TargetLabels:
    """Dense E2 event labels and categorical soft-FSM phase labels."""

    row_indices: Tensor
    event_targets: Tensor
    phase_targets: Tensor
    time_mask: Tensor
    provenance: tuple[TargetProvenance, ...]


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

    @property
    def batch_size(self) -> int:
        return int(self.operator_targets.shape[0])


@dataclass(frozen=True, slots=True)
class RetrievalTargetLabels:
    """Relevant record IDs for every explicitly labelled Retriever row."""

    relevant_record_ids: tuple[tuple[str, ...] | None, ...]
    provenance: tuple[TargetProvenance, ...]

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
        batch_size = int(observations.o1.logits.shape[0])
        device = observations.o1.logits.device

        o1 = self._build_o1(observations, labels.o1, batch_size, device)
        o2 = self._build_o2(observations, labels.o2, batch_size, device)
        e1 = self._build_e1(observations, labels.e1, batch_size, device)
        e2 = self._build_e2(observations, labels.e2, batch_size, device)
        operator, time = self._build_query(query, labels.query, batch_size, device)
        retrieval_input = self._build_retrieval(retrieval, labels.retrieval, batch_size)

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
        joined = _select_joined_rows(
            labels, observations.o1.valid_mask, "slot_mask", "O1", batch_size, device
        )
        if labels is None or joined is None:
            return None
        label_positions, global_rows, mask = joined
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
        joined = _select_joined_rows(
            labels, observations.o2.valid_mask, "slot_mask", "O2", batch_size, device
        )
        if labels is None or joined is None:
            return None
        label_positions, global_rows, mask = joined
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
        joined = _select_joined_rows(
            labels, observations.e1.valid_mask, "time_mask", "E1", batch_size, device
        )
        if labels is None or joined is None:
            return None
        label_positions, global_rows, mask = joined
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
        joined = _select_joined_rows(
            labels, observations.e2.valid_mask, "time_mask", "E2", batch_size, device
        )
        if labels is None or joined is None:
            return None
        label_positions, global_rows, mask = joined
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


def _select_explicit_rows(
    row_indices: Tensor,
    provenance: tuple[TargetProvenance, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    positions = tuple(
        index for index, source in enumerate(provenance) if source is not TargetProvenance.MISSING
    )
    if not positions:
        return None
    label_positions = torch.tensor(positions, dtype=torch.int64, device=device)
    global_rows = row_indices.index_select(0, label_positions)
    return label_positions, global_rows


def _select_joined_rows(
    labels: O1TargetLabels | O2TargetLabels | E1TargetLabels | E2TargetLabels | None,
    prediction_valid_mask: Tensor,
    mask_attr: str,
    name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if labels is None:
        return None
    selected = _select_explicit_rows(labels.row_indices, labels.provenance, batch_size, device)
    if selected is None:
        return None
    label_positions, global_rows = selected
    mask: Tensor = getattr(labels, mask_attr).index_select(0, label_positions)
    return label_positions, global_rows, mask


def _provenance_mask(
    provenance: tuple[TargetProvenance, ...],
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        [source is not TargetProvenance.MISSING for source in provenance],
        dtype=torch.bool,
        device=device,
    )


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


@dataclass(frozen=True, slots=True)
class O2DedupContext:
    """Detached pre-query Identity Bank view for the O2 soft-dedup objective.

    The prototypes and counts must come from the snapshot taken before the query
    chunk's own hard commit: a post-write view would let the current slots match
    themselves and degenerate the novelty target to zero.
    """

    prototypes: tuple[Tensor, ...]
    confirmed_counts: tuple[int, ...]

    @classmethod
    def from_identity_states(
        cls,
        states: Sequence[IdentityBankRuntimeState | None],
    ) -> O2DedupContext:
        prototypes: list[Tensor] = []
        counts: list[int] = []
        for state in states:
            if state is None or not state.confirmed:
                prototypes.append(torch.zeros((0, IDENTITY_DIM), dtype=torch.float32))
                counts.append(0)
                continue
            prototypes.append(
                torch.stack(
                    tuple(
                        identity.identity_prototype.detach().float()
                        for identity in state.confirmed
                    ),
                    dim=0,
                )
            )
            counts.append(int(state.unique_count))
        return cls(prototypes=tuple(prototypes), confirmed_counts=tuple(counts))


@dataclass(frozen=True, slots=True)
class OfficialWeakLossTerm:
    value: Tensor
    valid_rows: int


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
    retrieval_invalid_excluded_count: int = 0
    retrieval_ineligible_excluded_count: int = 0
    retrieval_causal_excluded_count: int = 0
    retrieval_candidate_total: int | None = None
    retrieval_positive_total: int | None = None
    retrieval_negative_total: int | None = None
    annotation_count_mismatch: int = 0
    o2_identity_cosine_sum: float = 0.0
    o2_identity_rows: int = 0
    o2_dedup_rows: int = 0
    o2_novelty_sum: float = 0.0
    o2_dedup_base_sum: float = 0.0

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
            ("task/o2_identity_cosine_sum", float(self.o2_identity_cosine_sum)),
            ("task/o2_identity_rows", float(self.o2_identity_rows)),
            ("task/o2_dedup_rows", float(self.o2_dedup_rows)),
            ("task/o2_novelty_sum", float(self.o2_novelty_sum)),
            ("task/o2_dedup_base_sum", float(self.o2_dedup_base_sum)),
        )


@dataclass(frozen=True, slots=True)
class OfficialWeakStateLossOutput:
    task: OfficialWeakLossTerm
    operator: OfficialWeakLossTerm
    retrieval: OfficialWeakLossTerm
    time: OfficialWeakLossTerm
    total: Tensor
    audit: OfficialWeakLossAudit


class OfficialWeakTargetBuilder:
    """Build official weak losses from predictions, never runtime inputs or hard-state writes."""

    __slots__ = ()

    def __call__(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        supervision: Sequence[OfficialWeakSupervision],
        *,
        dedup: O2DedupContext | None = None,
    ) -> OfficialWeakStateLossOutput:
        return self.build(observations, query, retrieval, supervision, dedup=dedup)

    def build(
        self,
        observations: ObservationOutputs,
        query: QueryEncoderOutput,
        retrieval: RetrieverOutput,
        supervision: Sequence[OfficialWeakSupervision],
        *,
        dedup: O2DedupContext | None = None,
    ) -> OfficialWeakStateLossOutput:
        batch_size = int(observations.o1.logits.shape[0])
        labels = tuple(supervision)
        if len(labels) != batch_size:
            raise ValueError("official weak supervision must align to the prediction batch")
        # Every soft head participates in every rank's differentiable graph even when the
        # official weak label masks a term. This preserves the exact numerical objective while
        # giving ZeRO-2 a stable, identical parameter-hook surface across mixed task classes.
        anchor = (
            observations.o1.logits.float().sum()
            + observations.o2.identity.float().sum()
            + observations.o2.score_logits.float().sum()
            + observations.o2.count_prediction.float().sum()
            + observations.o2.relevance.float().sum()
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
        retrieval_invalid_excluded_count = 0
        retrieval_ineligible_excluded_count = 0
        retrieval_causal_excluded_count = 0
        annotation_count_mismatch = 0
        o2_identity_cosine_sum = 0.0
        o2_identity_rows = 0
        o2_dedup_rows = 0
        o2_novelty_sum = 0.0
        o2_dedup_base_sum = 0.0

        for row, label in enumerate(labels):
            operator_index = OPERATORS.index(label.operator)
            operator_target = torch.tensor(
                [operator_index], dtype=torch.int64, device=query.route.logits.device
            )
            row_operator_loss = F.cross_entropy(
                query.route.logits[row : row + 1].float(), operator_target
            )
            operator_losses.append(row_operator_loss)

            mode_index = TIME_MODES.index(label.time_mode)
            mode_target = torch.tensor(
                [mode_index], dtype=torch.int64, device=query.time.logits.mode_logits.device
            )
            row_time = F.cross_entropy(
                query.time.logits.mode_logits[row : row + 1].float(), mode_target
            )
            if label.numeric_token_span is not None:
                start, end = label.numeric_token_span
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

            task_result = _official_weak_task_result(observations, row, label, dedup=dedup)
            task_losses.append(task_result.loss)
            if task_result.o2_row:
                o2_identity_cosine_sum += task_result.o2_identity_offdiag_cosine
                o2_identity_rows += 1
            if task_result.o2_dedup_row:
                o2_dedup_rows += 1
                o2_novelty_sum += task_result.o2_novelty_sum
                o2_dedup_base_sum += task_result.o2_dedup_base
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
        o2_integer_values = _distributed_sum_integers(
            (o2_identity_rows, o2_dedup_rows),
            audit_device,
        )
        o2_identity_rows, o2_dedup_rows = o2_integer_values
        o2_float_values = _distributed_sum_floats(
            (o2_identity_cosine_sum, o2_novelty_sum, o2_dedup_base_sum),
            audit_device,
        )
        o2_identity_cosine_sum, o2_novelty_sum, o2_dedup_base_sum = o2_float_values

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
                retrieval_invalid_excluded_count=retrieval_invalid_excluded_count,
                retrieval_ineligible_excluded_count=retrieval_ineligible_excluded_count,
                retrieval_causal_excluded_count=retrieval_causal_excluded_count,
                retrieval_candidate_total=retrieval_candidate_total,
                retrieval_positive_total=retrieval_positive_total,
                retrieval_negative_total=retrieval_negative_total,
                annotation_count_mismatch=annotation_count_mismatch,
                o2_identity_cosine_sum=o2_identity_cosine_sum,
                o2_identity_rows=o2_identity_rows,
                o2_dedup_rows=o2_dedup_rows,
                o2_novelty_sum=o2_novelty_sum,
                o2_dedup_base_sum=o2_dedup_base_sum,
            ),
        )


@dataclass(frozen=True, slots=True)
class _TaskLossResult:
    loss: Tensor
    o2_identity_offdiag_cosine: float = 0.0
    o2_row: bool = False
    o2_dedup_row: bool = False
    o2_novelty_sum: float = 0.0
    o2_dedup_base: float = 0.0


# Soft-dedup constants mirror the frozen Identity Bank match threshold; the temperature
# is a fixed objective shape, not a tunable contract surface.
_NOVELTY_MATCH_THRESHOLD = 0.8
_NOVELTY_TEMPERATURE = 0.1


def _o2_dedup_prediction(
    o2: O2SoftOutput,
    row: int,
    prototypes: Tensor,
    confirmed_count: int,
) -> tuple[Tensor, float, float]:
    """Differentiable soft-unique count: detached confirmed base + this chunk's novelty mass.

    novelty_i multiplies "not the same object" factors against the detached bank
    prototypes and against earlier in-chunk slots, accumulated in log space so a
    few hundred factors cannot underflow to NaN. Only the current chunk's identity
    vectors carry gradient; the base is a constant, keeping the target unbounded.
    """

    identity = o2.identity[row]
    valid = o2.valid_mask[row]
    anchor = identity.float().sum() * 0.0
    base = float(confirmed_count)
    if not bool(valid.any().item()):
        return anchor + base, 0.0, base
    selected = identity[valid].float()
    relevance = o2.relevance[row][valid].float()
    log_keep = torch.zeros(selected.shape[0], dtype=torch.float32, device=selected.device)
    if int(prototypes.shape[0]):
        bank_cosines = selected @ prototypes.to(
            device=selected.device, dtype=torch.float32
        ).transpose(0, 1)
        log_keep = log_keep + F.logsigmoid(
            -(bank_cosines - _NOVELTY_MATCH_THRESHOLD) / _NOVELTY_TEMPERATURE
        ).sum(dim=1)
    peer_cosines = selected @ selected.transpose(0, 1)
    peer_log = F.logsigmoid(-(peer_cosines - _NOVELTY_MATCH_THRESHOLD) / _NOVELTY_TEMPERATURE)
    earlier = torch.tril(torch.ones_like(peer_log, dtype=torch.bool), diagonal=-1)
    log_keep = log_keep + torch.where(earlier, peer_log, torch.zeros_like(peer_log)).sum(dim=1)
    # The relevance gate rides the same count label: over-counting pushes r down on
    # the least matching slots, under-counting pulls it up.
    novelty = relevance * torch.exp(log_keep)
    prediction = anchor + base + novelty.sum()
    return prediction, float(novelty.detach().sum().item()), base


def _official_weak_task_result(
    observations: ObservationOutputs,
    row: int,
    label: OfficialWeakSupervision,
    *,
    dedup: O2DedupContext | None = None,
) -> _TaskLossResult:
    target_count = torch.tensor(
        float(label.count), dtype=torch.float32, device=observations.o1.logits.device
    )
    if label.operator in (Operator.O1_SNAP, Operator.O1_DELTA):
        prediction = observations.o1.count_prediction[row]
        return _TaskLossResult(loss=_robust_count_loss(prediction, target_count))
    if label.operator in (Operator.O2_UNIQUE, Operator.O2_GAIN):
        with torch.no_grad():
            identity_cosine = _identity_offdiag_cosine(
                observations.o2.identity[row],
                observations.o2.valid_mask[row],
            )
        if label.operator is Operator.O2_UNIQUE and dedup is not None:
            prediction, novelty_sum, base = _o2_dedup_prediction(
                observations.o2,
                row,
                dedup.prototypes[row],
                dedup.confirmed_counts[row],
            )
            return _TaskLossResult(
                loss=_robust_count_loss(prediction, target_count),
                o2_identity_offdiag_cosine=identity_cosine,
                o2_row=True,
                o2_dedup_row=True,
                o2_novelty_sum=novelty_sum,
                o2_dedup_base=base,
            )
        # O2-Gain keeps the pooled fallback: its label counts a window increment the
        # cumulative confirmed base cannot represent, so full-base pressure would push
        # novelty toward zero (identity collapse) — the opposite of this objective.
        prediction = observations.o2.count_prediction[row]
        return _TaskLossResult(
            loss=_robust_count_loss(prediction, target_count),
            o2_identity_offdiag_cosine=identity_cosine,
            o2_row=True,
        )
    if label.operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
        valid = observations.e1.valid_mask[row]
        prediction = observations.e1.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        if not bool(valid.any().item()):
            dense = observations.e1.logits[row].float().sum() * 0.0
            return _TaskLossResult(loss=(dense + count_loss) / 2.0)
        logits = observations.e1.logits[row][valid].float()
        timestamps = observations.e1.timestamps[row][valid]
        targets, channel_mask = _build_e1_fsm_targets(
            timestamps,
            label.occurrence_points,
            label.query_time,
        )
        dense = _balanced_dense_bce(logits, targets, channel_mask)
        return _TaskLossResult(loss=(dense + count_loss) / 2.0)
    if label.operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
        valid = observations.e2.valid_mask[row]
        prediction = observations.e2.count_prediction[row]
        count_loss = _robust_count_loss(prediction, target_count)
        if not bool(valid.any().item()):
            dense_zero = observations.e2.event_logits[row].float().sum() * 0.0
            phase_zero = observations.e2.phase_logits[row].float().sum() * 0.0
            return _TaskLossResult(loss=(dense_zero + phase_zero + count_loss) / 3.0)
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
        dense = _balanced_dense_bce(
            event_logits,
            event_targets,
            torch.ones_like(event_targets, dtype=torch.bool),
        )
        phase = F.cross_entropy(phase_logits, phase_targets)
        return _TaskLossResult(loss=(dense + phase + count_loss) / 3.0)
    raise ValueError(f"unsupported official weak operator: {label.operator}")


def _balanced_dense_bce(
    logits: Tensor,
    targets: Tensor,
    supervision_mask: Tensor,
) -> Tensor:
    channel_losses: list[Tensor] = []
    for channel in range(logits.shape[1]):
        mask = supervision_mask[:, channel]
        channel_logits = logits[:, channel][mask]
        channel_targets = targets[:, channel][mask]
        positive = channel_targets >= 0.5
        negative = ~positive
        positive_count = int(positive.sum().item())
        negative_count = int(negative.sum().item())
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
    return torch.stack(channel_losses).mean() if channel_losses else logits.sum() * 0.0


def _build_e1_fsm_targets(
    timestamps: Tensor,
    occurrence_points: Sequence[float],
    query_time: float,
) -> tuple[Tensor, Tensor]:
    """Build onset then completion+transition targets that the hard FSM can realize."""

    targets = torch.zeros((timestamps.shape[0], 3), dtype=torch.float32, device=timestamps.device)
    supervision_mask = torch.ones_like(targets, dtype=torch.bool)
    claimed_positions: set[int] = set()
    for point in sorted(point for point in occurrence_points if point <= query_time):
        completion_index = _voronoi_timestamp_index(timestamps, point)
        if completion_index is None:
            continue
        onset_index = completion_index - 1
        if onset_index < 0:
            supervision_mask[completion_index] = False
            continue
        if onset_index in claimed_positions or completion_index in claimed_positions:
            supervision_mask[onset_index, 0] = False
            supervision_mask[completion_index, 1:] = False
            continue
        targets[onset_index, 0] = 1.0
        targets[completion_index, 1:] = 1.0
        claimed_positions.update((onset_index, completion_index))
    return targets, supervision_mask


@dataclass(frozen=True, slots=True)
class _RetrievalLossResult:
    loss: Tensor | None
    positive_count: int
    candidate_count: int
    negative_count: int
    status: str
    wrong_operator: bool
    rescued_wrong_route: bool
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
    _, head_codes, _, timestamps, time_ranges = retrieval.require_tensor_metadata()
    head_mask = (
        head_codes[row] == RETRIEVAL_HEAD_ORDER.index(target_head)
    ) & retrieval.present_mask[row]
    record_end = torch.where(
        timestamps[row] >= 0.0,
        timestamps[row],
        time_ranges[row, :, 1],
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
    target_head_present_count = int(head_mask.sum().item())
    if not candidate_count:
        return _RetrievalLossResult(
            loss=None,
            positive_count=0,
            candidate_count=0,
            negative_count=0,
            status="no_candidate",
            wrong_operator=wrong_operator,
            rescued_wrong_route=False,
            target_head_present_count=target_head_present_count,
            invalid_excluded_count=invalid_excluded,
            ineligible_excluded_count=ineligible_excluded,
            causal_excluded_count=causal_excluded,
        )
    positive_mask = _retrieval_occurrence_mask(retrieval, row, label) & candidate_mask
    positive_columns = torch.nonzero(positive_mask, as_tuple=False).flatten()
    positive_count = int(positive_columns.numel())
    if not positive_count:
        return _RetrievalLossResult(
            loss=None,
            positive_count=0,
            candidate_count=candidate_count,
            negative_count=candidate_count,
            status="no_positive",
            wrong_operator=wrong_operator,
            rescued_wrong_route=False,
            target_head_present_count=target_head_present_count,
            invalid_excluded_count=invalid_excluded,
            ineligible_excluded_count=ineligible_excluded,
            causal_excluded_count=causal_excluded,
        )
    negative_count = candidate_count - positive_count
    if negative_count == 0:
        return _RetrievalLossResult(
            loss=None,
            positive_count=positive_count,
            candidate_count=candidate_count,
            negative_count=0,
            status="all_positive",
            wrong_operator=wrong_operator,
            rescued_wrong_route=False,
            target_head_present_count=target_head_present_count,
            invalid_excluded_count=invalid_excluded,
            ineligible_excluded_count=ineligible_excluded,
            causal_excluded_count=causal_excluded,
        )
    all_logits = retrieval.scores[row].index_select(0, present_columns).float()
    positive_logits = retrieval.scores[row].index_select(0, positive_columns).float()
    loss = torch.logsumexp(all_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
    return _RetrievalLossResult(
        loss=loss,
        positive_count=positive_count,
        candidate_count=candidate_count,
        negative_count=negative_count,
        status="valid_bag",
        wrong_operator=wrong_operator,
        rescued_wrong_route=wrong_operator,
        target_head_present_count=target_head_present_count,
        invalid_excluded_count=invalid_excluded,
        ineligible_excluded_count=ineligible_excluded,
        causal_excluded_count=causal_excluded,
    )


def _retrieval_occurrence_mask(
    retrieval: RetrieverOutput,
    row: int,
    label: OfficialWeakSupervision,
) -> Tensor:
    _, _, _, candidate_timestamps, candidate_time_ranges = retrieval.require_tensor_metadata()
    timestamps = candidate_timestamps[row]
    ranges = candidate_time_ranges[row]
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


def _robust_count_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(
        torch.log1p(prediction.float()),
        torch.log1p(target.float()),
        beta=0.25,
    )


def _identity_offdiag_cosine(identity: Tensor, valid_mask: Tensor) -> float:
    """Mean pairwise cosine over one row's valid identity vectors; a pure probe scalar."""

    selected = identity[valid_mask].float()
    count = int(selected.shape[0])
    if count < 2:
        return 0.0
    gram = selected @ selected.transpose(0, 1)
    value = float((gram.sum() - gram.diagonal().sum()).item()) / float(count * (count - 1))
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


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
    # Preserve one identical all-head graph surface on every rank. The anchor is exactly
    # zero-valued, so adding it never changes the objective, but it prevents a locally valid
    # MIL/Task branch and a locally masked branch from presenting different ZeRO hook sets.
    value = torch.stack(values).mean() + anchor if values else anchor
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
    "AnswerTargetLabels",
    "QueryTargetLabels",
    "RetrievalTargetLabels",
    "StageATargetBatch",
    "StageATargetBuilder",
    "TargetProvenance",
]
