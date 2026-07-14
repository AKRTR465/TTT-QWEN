"""Implement complete threshold retrieval over one typed State Bank snapshot.

Inputs: q_target, hard operators, resolved time windows, owner IDs, and a row-wise Bank view.
Outputs: all passing typed records, aligned scores/masks, structured status, counts, and audit.
Forbidden: fixed Top-K, ANN, Reader arithmetic, labels, future records, or Bank mutation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import CalibrationStatus, ProjectConfig, RetrieverConfig
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
)
from ttt_svcbench_qwen.state_bank import (
    StateBankRuntimeState,
    StateBankView,
    StateRecord,
    StateRecordKind,
    StructuredStateBank,
    clone_state_record,
)


class RetrievalStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class RetrievalReason(StrEnum):
    MATCHED = "matched"
    EMPTY_BANK = "empty_bank"
    EMPTY_HEAD_PARTITION = "empty_head_partition"
    ALL_INVALID = "all_invalid"
    ALL_RETRIEVAL_INELIGIBLE = "all_retrieval_ineligible"
    ALL_FUTURE = "all_future"
    ALL_OUTSIDE_WINDOW = "all_outside_window"
    BELOW_SIMILARITY = "below_similarity"
    NO_MATCH = "no_match"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_TIME = "unsupported_time"
    DEGENERATE_QUERY = "degenerate_q_target"
    INVALID_TIME = "invalid_time"
    OWNER_MISMATCH = "owner_mismatch"


@dataclass(frozen=True, slots=True)
class RetrievalFilterAudit:
    n_state: int
    head_partition_excluded_count: int
    query_rejected_count: int
    owner_mismatch_count: int
    invalid_count: int
    retrieval_ineligible_count: int
    future_count: int
    outside_window_count: int
    below_similarity_count: int
    selected_count: int

    def __post_init__(self) -> None:
        values = (
            self.n_state,
            self.head_partition_excluded_count,
            self.query_rejected_count,
            self.owner_mismatch_count,
            self.invalid_count,
            self.retrieval_ineligible_count,
            self.future_count,
            self.outside_window_count,
            self.below_similarity_count,
            self.selected_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Retriever audit counts must be non-negative integers")
        exclusive = (
            self.query_rejected_count
            + self.owner_mismatch_count
            + self.invalid_count
            + self.retrieval_ineligible_count
            + self.future_count
            + self.outside_window_count
            + self.below_similarity_count
            + self.selected_count
        )
        if exclusive != self.n_state:
            raise ValueError("Retriever exclusive filter counts must sum to n_state")


@dataclass(frozen=True, slots=True)
class RetrieverOutput:
    selected_record_ids: tuple[tuple[str, ...], ...]
    selected_scores: tuple[tuple[float, ...], ...]
    selected_records: tuple[tuple[StateRecord, ...], ...]
    candidate_record_ids: tuple[tuple[str | None, ...], ...]
    candidate_records: tuple[tuple[StateRecord | None, ...], ...]
    state_embeddings: Tensor
    scores: Tensor
    present_mask: Tensor
    selected_mask: Tensor
    status: tuple[RetrievalStatus, ...]
    reason: tuple[RetrievalReason, ...]
    hard_operators: tuple[Operator, ...]
    time_resolutions: tuple[TimeResolution, ...]
    n_state: Tensor
    n_retrieved: Tensor
    audit: tuple[RetrievalFilterAudit, ...]
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    bank_video_ids: tuple[str, ...]
    bank_trajectory_ids: tuple[str, ...]
    bank_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.state_embeddings.ndim != 3
            or self.state_embeddings.shape[-1] != 512
            or not torch.is_floating_point(self.state_embeddings)
        ):
            raise ValueError("Retriever state_embeddings must be floating [B, N_s, 512]")
        if (
            self.scores.shape != self.state_embeddings.shape[:2]
            or self.scores.dtype != torch.float32
        ):
            raise ValueError("Retriever scores must be FP32 [B, N_s]")
        shape = self.scores.shape
        if (
            self.present_mask.shape != shape
            or self.selected_mask.shape != shape
            or self.present_mask.dtype != torch.bool
            or self.selected_mask.dtype != torch.bool
        ):
            raise ValueError("Retriever masks must be bool [B, N_s]")
        tensors = (self.state_embeddings, self.scores, self.present_mask, self.selected_mask)
        if any(tensor.device != self.scores.device for tensor in tensors):
            raise ValueError("Retriever aligned tensors must share one device")
        batch_size, width = shape
        metadata = (
            self.selected_record_ids,
            self.selected_scores,
            self.selected_records,
            self.candidate_record_ids,
            self.candidate_records,
            self.status,
            self.reason,
            self.hard_operators,
            self.time_resolutions,
            self.audit,
            self.video_ids,
            self.trajectory_ids,
            self.bank_video_ids,
            self.bank_trajectory_ids,
            self.bank_versions,
        )
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("Retriever metadata must contain one entry per batch item")
        if any(len(row) != width for row in self.candidate_record_ids) or any(
            len(row) != width for row in self.candidate_records
        ):
            raise ValueError("candidate record snapshots must align to the padded score width")
        for counts, name in ((self.n_state, "n_state"), (self.n_retrieved, "n_retrieved")):
            if (
                counts.shape != (batch_size,)
                or counts.dtype != torch.int64
                or counts.device != self.scores.device
            ):
                raise ValueError(f"{name} must be int64 [B] on the Retriever device")
        if not all(
            bool(torch.isfinite(tensor).all()) for tensor in (self.state_embeddings, self.scores)
        ):
            raise ValueError("Retriever embeddings/scores must be finite")
        if bool(torch.any(self.selected_mask & ~self.present_mask)):
            raise ValueError("Retriever cannot select padded records")
        if bool(torch.any(self.state_embeddings[~self.present_mask] != 0.0)) or bool(
            torch.any(self.scores[~self.present_mask] != 0.0)
        ):
            raise ValueError("Retriever padding embeddings/scores must be zero")
        if not torch.equal(self.n_state, self.present_mask.sum(dim=1)):
            raise ValueError("Retriever n_state must count present partition records")
        if not torch.equal(self.n_retrieved, self.selected_mask.sum(dim=1)):
            raise ValueError("Retriever n_retrieved must count selected records")
        if bool(torch.any(self.n_retrieved > self.n_state)):
            raise ValueError("Retriever requires 0 <= N_ret <= N_s")
        if any(
            not value
            for value in self.video_ids
            + self.trajectory_ids
            + self.bank_video_ids
            + self.bank_trajectory_ids
        ):
            raise ValueError("Retriever owner identifiers must be non-empty")
        if len(set(zip(self.video_ids, self.trajectory_ids, strict=True))) != batch_size:
            raise ValueError("Retriever batch owners must be unique")
        if any(type(version) is not int or version < 0 for version in self.bank_versions):
            raise ValueError("Retriever bank versions must be non-negative integers")
        if any(not isinstance(operator, Operator) for operator in self.hard_operators):
            raise TypeError("Retriever hard_operators must preserve one Operator per row")
        if any(not isinstance(resolution, TimeResolution) for resolution in self.time_resolutions):
            raise TypeError("Retriever time_resolutions must preserve one TimeResolution per row")
        for row in range(batch_size):
            self._validate_row(row)

    def validate_integrity(self) -> None:
        """Revalidate mutable tensor-backed snapshot contents before downstream consumption."""

        self.__post_init__()

    def _validate_row(self, row: int) -> None:
        n_state = int(self.n_state[row].item())
        n_retrieved = int(self.n_retrieved[row].item())
        ids = self.selected_record_ids[row]
        selected_scores = self.selected_scores[row]
        records = self.selected_records[row]
        operator = self.hard_operators[row]
        expected_head = OPERATOR_TO_HEAD_TYPE[operator]
        if (
            len(ids) != n_retrieved
            or len(selected_scores) != n_retrieved
            or len(records) != n_retrieved
        ):
            raise ValueError("Retriever selected metadata must have N_ret entries")
        if len(set(ids)) != len(ids) or tuple(record.record_id for record in records) != ids:
            raise ValueError("Retriever selected records and IDs must be unique and aligned")
        if any(
            record.video_id != self.video_ids[row]
            or record.trajectory_id != self.trajectory_ids[row]
            for record in records
        ):
            raise ValueError("Retriever selected records cannot cross owner boundaries")
        if expected_head is None and records:
            raise ValueError("unsupported Retriever operators cannot retain selected records")
        if expected_head is not None and any(
            record.head_type is not expected_head for record in records
        ):
            raise ValueError("Retriever selected records must match the preserved operator head")
        present_ids = self.candidate_record_ids[row]
        candidate_records = self.candidate_records[row]
        candidate_by_id: dict[str, StateRecord] = {}
        for column, record_id in enumerate(present_ids):
            if (record_id is not None) != bool(self.present_mask[row, column]):
                raise ValueError("Retriever candidate IDs must align to present_mask")
            candidate = candidate_records[column]
            if (candidate is not None) != (record_id is not None):
                raise ValueError("Retriever candidate typed snapshots must align to candidate IDs")
            if candidate is None:
                continue
            if (
                candidate.record_id != record_id
                or candidate.video_id != self.bank_video_ids[row]
                or candidate.trajectory_id != self.bank_trajectory_ids[row]
            ):
                raise ValueError("Retriever candidate typed snapshot metadata is inconsistent")
            if (
                candidate.semantic_embedding.dtype != self.state_embeddings.dtype
                or candidate.semantic_embedding.device != self.state_embeddings.device
                or not torch.equal(
                    candidate.semantic_embedding,
                    self.state_embeddings[row, column],
                )
            ):
                raise ValueError("Retriever candidate typed snapshot semantics are inconsistent")
            candidate_by_id[candidate.record_id] = candidate
        live_candidate_ids = tuple(record_id for record_id in present_ids if record_id is not None)
        if len(set(live_candidate_ids)) != len(live_candidate_ids):
            raise ValueError("Retriever candidate record IDs must be unique per row")
        selected_columns = torch.nonzero(self.selected_mask[row], as_tuple=False).flatten().tolist()
        mask_ids = tuple(present_ids[column] for column in selected_columns)
        if set(mask_ids) != set(ids):
            raise ValueError("Retriever selected IDs must exactly match selected_mask")
        if any(
            not _snapshot_values_equal(record, candidate_by_id[record.record_id])
            for record in records
        ):
            raise ValueError("Retriever selected typed records must match the candidate snapshot")
        score_by_id = {
            present_ids[column]: float(self.scores[row, column].detach().item())
            for column in selected_columns
        }
        expected_scores = tuple(score_by_id[record_id] for record_id in ids)
        if selected_scores != expected_scores:
            raise ValueError("Retriever selected_scores must align to scores and selected IDs")
        expected_order = tuple(
            sorted(ids, key=lambda record_id: (-score_by_id[record_id], record_id))
        )
        if ids != expected_order:
            raise ValueError("Retriever selected IDs must use score-desc/record-ID-asc order")
        status = self.status[row]
        reason = self.reason[row]
        if not isinstance(status, RetrievalStatus) or not isinstance(reason, RetrievalReason):
            raise TypeError("Retriever status/reason metadata is invalid")
        if (status is RetrievalStatus.OK) != (n_retrieved > 0):
            raise ValueError("only Retriever OK rows may contain selected records")
        if status is RetrievalStatus.OK and reason is not RetrievalReason.MATCHED:
            raise ValueError("Retriever OK rows require the matched reason")
        if status is not RetrievalStatus.OK and n_retrieved != 0:
            raise ValueError("non-OK Retriever rows cannot retain selections")
        allowed_reasons: dict[RetrievalStatus, set[RetrievalReason]] = {
            RetrievalStatus.OK: {RetrievalReason.MATCHED},
            RetrievalStatus.EMPTY: {
                RetrievalReason.EMPTY_BANK,
                RetrievalReason.EMPTY_HEAD_PARTITION,
                RetrievalReason.ALL_INVALID,
                RetrievalReason.ALL_RETRIEVAL_INELIGIBLE,
                RetrievalReason.ALL_FUTURE,
                RetrievalReason.ALL_OUTSIDE_WINDOW,
                RetrievalReason.BELOW_SIMILARITY,
                RetrievalReason.NO_MATCH,
            },
            RetrievalStatus.UNSUPPORTED: {
                RetrievalReason.UNSUPPORTED_OPERATOR,
                RetrievalReason.UNSUPPORTED_TIME,
                RetrievalReason.DEGENERATE_QUERY,
            },
            RetrievalStatus.INVALID: {
                RetrievalReason.INVALID_TIME,
                RetrievalReason.OWNER_MISMATCH,
            },
        }
        if reason not in allowed_reasons[status]:
            raise ValueError("Retriever reason is inconsistent with its structured status")
        query_owner = (self.video_ids[row], self.trajectory_ids[row])
        bank_owner = (self.bank_video_ids[row], self.bank_trajectory_ids[row])
        if (reason is RetrievalReason.OWNER_MISMATCH) == (query_owner == bank_owner):
            raise ValueError("Retriever owner-mismatch reason must match query/Bank provenance")
        if self.audit[row].n_state != n_state or self.audit[row].selected_count != n_retrieved:
            raise ValueError("Retriever audit counts must align to output counts")


@dataclass(frozen=True, slots=True)
class RetrievalQualityMetrics:
    true_positive_count: int
    selected_denominator: int
    relevant_denominator: int
    precision: float | None
    recall: float | None
    empty_retrieval_count: int
    query_denominator: int
    empty_retrieval_rate: float | None
    unsupported_count: int
    invalid_count: int
    total_query_count: int
    unsupported_rate: float | None
    invalid_rate: float | None


class EmbeddingStateRetriever(nn.Module):  # type: ignore[misc]
    """Zero-parameter exact scorer; soft scores retain q_target gradients."""

    def __init__(self, config: RetrieverConfig) -> None:
        super().__init__()
        _validate_retriever_config(config)
        self.config = config

    def retrieve_states(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        q_target: Tensor,
        hard_operators: Sequence[Operator],
        time_resolutions: Sequence[TimeResolution],
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput:
        """Create the row-wise owner/head snapshot, then retrieve from that exact version."""

        if q_target.ndim == 0:
            raise ValueError("q_target must be floating [B, 512]")
        operators = _normalize_operators(hard_operators, q_target.shape[0])
        heads = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in operators)
        view = state_bank.view(states, heads)
        return self.forward(
            q_target,
            operators,
            time_resolutions,
            view,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )

    def retrieve_query(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput:
        return self.retrieve_states(
            state_bank,
            states,
            query.q_target,
            query.hard_operators,
            query.time.resolutions,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )

    def forward(
        self,
        q_target: Tensor,
        hard_operators: Sequence[Operator],
        time_resolutions: Sequence[TimeResolution],
        state_view: StateBankView,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput:
        batch_size = _validate_retrieval_inputs(
            self.config,
            q_target,
            hard_operators,
            time_resolutions,
            state_view,
            video_ids,
            trajectory_ids,
        )
        operators = _normalize_operators(hard_operators, batch_size)
        resolutions = _normalize_resolutions(time_resolutions, batch_size)
        query_video_ids = _normalize_owner_ids(video_ids, batch_size, "video_ids")
        query_trajectory_ids = _normalize_owner_ids(trajectory_ids, batch_size, "trajectory_ids")
        query_fp32 = q_target.float()
        query_norms = torch.linalg.vector_norm(query_fp32, dim=-1)
        query_usable = (
            torch.isfinite(query_fp32).all(dim=-1)
            & torch.isfinite(query_norms)
            & (query_norms > self.config.normalization_eps)
        )
        safe_query = torch.where(
            query_usable.unsqueeze(-1),
            query_fp32,
            torch.zeros_like(query_fp32),
        )
        normalized_query = F.normalize(
            safe_query,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        normalized_state = F.normalize(
            state_view.embeddings.float(), dim=-1, eps=self.config.normalization_eps
        )
        scores = torch.einsum("bd,bnd->bn", normalized_query, normalized_state)
        scores = torch.where(state_view.present_mask, scores, torch.zeros_like(scores))
        selected_mask = torch.zeros_like(state_view.present_mask)
        statuses: list[RetrievalStatus] = []
        reasons: list[RetrievalReason] = []
        audits: list[RetrievalFilterAudit] = []
        selected_ids: list[tuple[str, ...]] = []
        selected_scores: list[tuple[float, ...]] = []
        selected_records: list[tuple[StateRecord, ...]] = []
        n_retrieved = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
        for row in range(batch_size):
            row_result = self._retrieve_row(
                row,
                scores,
                selected_mask,
                operators[row],
                resolutions[row],
                state_view,
                query_video_ids[row],
                query_trajectory_ids[row],
                float(query_norms[row].detach().item()),
            )
            status, reason, audit, ids, row_scores, records = row_result
            statuses.append(status)
            reasons.append(reason)
            audits.append(audit)
            selected_ids.append(ids)
            selected_scores.append(row_scores)
            selected_records.append(records)
            n_retrieved[row] = len(ids)

        return RetrieverOutput(
            selected_record_ids=tuple(selected_ids),
            selected_scores=tuple(selected_scores),
            selected_records=tuple(selected_records),
            candidate_record_ids=state_view.record_ids,
            candidate_records=tuple(
                tuple(clone_state_record(record) if record is not None else None for record in row)
                for row in state_view.cloned_records
            ),
            state_embeddings=state_view.embeddings.detach().clone(),
            scores=scores,
            present_mask=state_view.present_mask.detach().clone(),
            selected_mask=selected_mask,
            status=tuple(statuses),
            reason=tuple(reasons),
            hard_operators=operators,
            time_resolutions=resolutions,
            n_state=state_view.n_state.detach().clone(),
            n_retrieved=n_retrieved,
            audit=tuple(audits),
            video_ids=query_video_ids,
            trajectory_ids=query_trajectory_ids,
            bank_video_ids=state_view.video_ids,
            bank_trajectory_ids=state_view.trajectory_ids,
            bank_versions=state_view.bank_versions,
        )

    def _retrieve_row(
        self,
        row: int,
        scores: Tensor,
        selected_mask: Tensor,
        operator: Operator,
        resolution: TimeResolution,
        state_view: StateBankView,
        video_id: str,
        trajectory_id: str,
        query_norm: float,
    ) -> tuple[
        RetrievalStatus,
        RetrievalReason,
        RetrievalFilterAudit,
        tuple[str, ...],
        tuple[float, ...],
        tuple[StateRecord, ...],
    ]:
        n_state = int(state_view.n_state[row].item())
        owner_count = int(state_view.owner_record_counts[row].item())
        head_excluded = owner_count - n_state
        owner_matches = (
            state_view.video_ids[row] == video_id
            and state_view.trajectory_ids[row] == trajectory_id
        )
        if not owner_matches:
            return _rejected_row(
                RetrievalStatus.INVALID,
                RetrievalReason.OWNER_MISMATCH,
                n_state,
                head_excluded,
                owner_mismatch=True,
            )
        if resolution.status is TimeResolutionStatus.INVALID:
            return _rejected_row(
                RetrievalStatus.INVALID,
                RetrievalReason.INVALID_TIME,
                n_state,
                head_excluded,
                query_rejected=True,
            )
        if resolution.status is TimeResolutionStatus.UNSUPPORTED:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_TIME,
                n_state,
                head_excluded,
                query_rejected=True,
            )
        expected_head = OPERATOR_TO_HEAD_TYPE[operator]
        if operator is Operator.UNSUPPORTED or expected_head is None:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_OPERATOR,
                n_state,
                head_excluded,
                query_rejected=True,
            )
        if not math.isfinite(query_norm) or query_norm <= self.config.normalization_eps:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.DEGENERATE_QUERY,
                n_state,
                head_excluded,
                query_rejected=True,
            )

        invalid = ineligible = future = outside = below = 0
        similarity_threshold = scores.new_tensor(self.config.record_similarity_threshold)
        selected_columns: list[int] = []
        for column in range(n_state):
            record = state_view.cloned_records[row][column]
            kind = state_view.record_kinds[row][column]
            if record is None or kind is None:
                raise ValueError("present State Bank columns require cloned record metadata")
            if record.head_type is not expected_head:
                raise ValueError("State Bank row-wise partition does not match the hard operator")
            if not record.valid:
                invalid += 1
            elif not bool(state_view.retrieval_eligible_mask[row, column]):
                ineligible += 1
            elif _record_end(record) > resolution.window.query_time:
                future += 1
            elif _requires_atomic_window_filter(kind, resolution.window) and not _intersects_window(
                record, resolution.window
            ):
                outside += 1
            elif bool(scores[row, column].detach() < similarity_threshold):
                below += 1
            else:
                selected_columns.append(column)
                selected_mask[row, column] = True

        ordered_columns = sorted(
            selected_columns,
            key=lambda column: (
                -float(scores[row, column].detach().item()),
                _required_record_id(state_view, row, column),
            ),
        )
        ids = tuple(_required_record_id(state_view, row, column) for column in ordered_columns)
        row_scores = tuple(float(scores[row, column].detach().item()) for column in ordered_columns)
        records = tuple(
            clone_state_record(_required_record(state_view, row, column))
            for column in ordered_columns
        )
        audit = RetrievalFilterAudit(
            n_state=n_state,
            head_partition_excluded_count=head_excluded,
            query_rejected_count=0,
            owner_mismatch_count=0,
            invalid_count=invalid,
            retrieval_ineligible_count=ineligible,
            future_count=future,
            outside_window_count=outside,
            below_similarity_count=below,
            selected_count=len(ids),
        )
        if ids:
            return RetrievalStatus.OK, RetrievalReason.MATCHED, audit, ids, row_scores, records
        reason = _empty_reason(audit, owner_count)
        return RetrievalStatus.EMPTY, reason, audit, (), (), ()


def build_state_retriever(config: ProjectConfig | None = None) -> EmbeddingStateRetriever:
    if config is None:
        raise ValueError("build_state_retriever requires a validated ProjectConfig")
    return EmbeddingStateRetriever(config.retriever)


def evaluate_retrieval_quality(
    selected_record_ids: Sequence[Sequence[str]],
    relevant_record_ids: Sequence[Sequence[str]],
    statuses: Sequence[RetrievalStatus],
) -> RetrievalQualityMetrics:
    """Compute status-aware offline metrics; GT IDs never enter Retriever runtime."""

    selected_rows = tuple(tuple(row) for row in selected_record_ids)
    relevant_rows = tuple(tuple(row) for row in relevant_record_ids)
    normalized_statuses = tuple(statuses)
    if len({len(selected_rows), len(relevant_rows), len(normalized_statuses)}) != 1:
        raise ValueError("selected, relevant, and status retrieval rows must have equal length")
    if any(not isinstance(status, RetrievalStatus) for status in normalized_statuses):
        raise ValueError("retrieval quality statuses must contain RetrievalStatus values")
    true_positive = selected_total = relevant_total = empty = 0
    unsupported = invalid = 0
    for selected, relevant, status in zip(
        selected_rows,
        relevant_rows,
        normalized_statuses,
        strict=True,
    ):
        if any(not record_id for record_id in selected + relevant):
            raise ValueError("retrieval quality IDs must be non-empty strings")
        if len(set(selected)) != len(selected) or len(set(relevant)) != len(relevant):
            raise ValueError("retrieval quality rows cannot contain duplicate IDs")
        if (status is RetrievalStatus.OK) != bool(selected):
            raise ValueError("only retrieval status OK may contain selected record IDs")
        selected_set = set(selected)
        relevant_set = set(relevant)
        true_positive += len(selected_set & relevant_set)
        selected_total += len(selected_set)
        relevant_total += len(relevant_set)
        if status is RetrievalStatus.UNSUPPORTED:
            unsupported += 1
            continue
        if status is RetrievalStatus.INVALID:
            invalid += 1
            continue
        empty += int(status is RetrievalStatus.EMPTY)
    query_denominator = len(normalized_statuses) - unsupported - invalid
    total_query_count = len(normalized_statuses)
    return RetrievalQualityMetrics(
        true_positive_count=true_positive,
        selected_denominator=selected_total,
        relevant_denominator=relevant_total,
        precision=true_positive / selected_total if selected_total else None,
        recall=true_positive / relevant_total if relevant_total else None,
        empty_retrieval_count=empty,
        query_denominator=query_denominator,
        empty_retrieval_rate=empty / query_denominator if query_denominator else None,
        unsupported_count=unsupported,
        invalid_count=invalid,
        total_query_count=total_query_count,
        unsupported_rate=(unsupported / total_query_count if total_query_count else None),
        invalid_rate=invalid / total_query_count if total_query_count else None,
    )


def _validate_retriever_config(config: RetrieverConfig) -> None:
    expected_operator_heads = ("o1", "o1", "o2", "o2", "e1", "e1", "e2", "e2", None)
    checks: tuple[tuple[str, object, object], ...] = (
        ("semantic_dim", config.semantic_dim, 512),
        ("record_similarity_threshold", config.record_similarity_threshold, 0.35),
        (
            "threshold_status",
            config.threshold_status,
            CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
        ),
        ("similarity_dtype", config.similarity_dtype, "float32"),
        ("normalization_eps", config.normalization_eps, 1.0e-8),
        ("zero_query_policy", config.zero_query_policy, "unsupported"),
        ("threshold_comparison", config.threshold_comparison, "greater_than_or_equal"),
        ("record_confidence_threshold", config.record_confidence_threshold, None),
        ("operator_head_types", config.operator_head_types, expected_operator_heads),
        (
            "filter_order",
            config.filter_order,
            ("invalid", "retrieval_ineligible", "future", "outside_window", "below_similarity"),
        ),
        ("selection_order", config.selection_order, ("score_desc", "record_id_asc")),
        ("owner_mismatch_status", config.owner_mismatch_status, "invalid"),
        (
            "aggregate_time_policy",
            config.aggregate_time_policy,
            "causal_availability_only_window_in_reader",
        ),
        ("atomic_window_boundary", config.atomic_window_boundary, "closed"),
        (
            "metrics_policy",
            config.metrics_policy,
            "offline_ground_truth_runtime_label_free",
        ),
        ("top_k", config.top_k, None),
        ("ann_enabled", config.ann_enabled, False),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"Retriever {name} must equal {expected!r}; got {actual!r}")


def _validate_retrieval_inputs(
    config: RetrieverConfig,
    q_target: Tensor,
    hard_operators: Sequence[Operator],
    time_resolutions: Sequence[TimeResolution],
    state_view: StateBankView,
    video_ids: Sequence[str],
    trajectory_ids: Sequence[str],
) -> int:
    if (
        q_target.ndim != 2
        or q_target.shape[1] != config.semantic_dim
        or not torch.is_floating_point(q_target)
    ):
        raise ValueError("q_target must be floating [B, 512]")
    if not bool(torch.isfinite(q_target).all()):
        raise ValueError("q_target must be finite")
    batch_size = int(q_target.shape[0])
    if state_view.embeddings.shape[0] != batch_size:
        raise ValueError("q_target and State Bank view batch sizes must match")
    if state_view.embeddings.device != q_target.device:
        raise ValueError("q_target and State Bank view must share one device")
    _normalize_operators(hard_operators, batch_size)
    _normalize_resolutions(time_resolutions, batch_size)
    normalized_video_ids = _normalize_owner_ids(video_ids, batch_size, "video_ids")
    normalized_trajectory_ids = _normalize_owner_ids(trajectory_ids, batch_size, "trajectory_ids")
    if len(set(zip(normalized_video_ids, normalized_trajectory_ids, strict=True))) != batch_size:
        raise ValueError("Retriever query owners must be unique within a batch")
    return batch_size


def _normalize_operators(operators: Sequence[Operator], batch_size: int) -> tuple[Operator, ...]:
    normalized = tuple(operators)
    if len(normalized) != batch_size or any(not isinstance(item, Operator) for item in normalized):
        raise ValueError("hard_operators must contain one Operator per batch row")
    return normalized


def _normalize_resolutions(
    resolutions: Sequence[TimeResolution], batch_size: int
) -> tuple[TimeResolution, ...]:
    normalized = tuple(resolutions)
    if len(normalized) != batch_size or any(
        not isinstance(item, TimeResolution) for item in normalized
    ):
        raise ValueError("time_resolutions must contain one TimeResolution per batch row")
    return normalized


def _normalize_owner_ids(values: Sequence[str], batch_size: int, name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if len(normalized) != batch_size or any(
        not isinstance(value, str) or not value for value in normalized
    ):
        raise ValueError(f"{name} must contain one non-empty string per batch row")
    return normalized


def _rejected_row(
    status: RetrievalStatus,
    reason: RetrievalReason,
    n_state: int,
    head_excluded: int,
    *,
    query_rejected: bool = False,
    owner_mismatch: bool = False,
) -> tuple[
    RetrievalStatus,
    RetrievalReason,
    RetrievalFilterAudit,
    tuple[str, ...],
    tuple[float, ...],
    tuple[StateRecord, ...],
]:
    audit = RetrievalFilterAudit(
        n_state=n_state,
        head_partition_excluded_count=head_excluded,
        query_rejected_count=n_state if query_rejected else 0,
        owner_mismatch_count=n_state if owner_mismatch else 0,
        invalid_count=0,
        retrieval_ineligible_count=0,
        future_count=0,
        outside_window_count=0,
        below_similarity_count=0,
        selected_count=0,
    )
    return status, reason, audit, (), (), ()


def _record_end(record: StateRecord) -> float:
    if record.timestamp is not None:
        return float(record.timestamp)
    assert record.time_range is not None
    return float(record.time_range[1])


def _requires_atomic_window_filter(kind: StateRecordKind, window: TimeWindow) -> bool:
    return bool(kind is StateRecordKind.O2_CONFIRMED and window.start_time is not None)


def _intersects_window(record: StateRecord, window: TimeWindow) -> bool:
    if window.start_time is None:
        return True
    if record.timestamp is not None:
        return bool(window.start_time <= record.timestamp <= window.end_time)
    assert record.time_range is not None
    start, end = record.time_range
    return bool(start <= window.end_time and end >= window.start_time)


def _required_record_id(view: StateBankView, row: int, column: int) -> str:
    record_id = view.record_ids[row][column]
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("present State Bank records require record IDs")
    return record_id


def _required_record(view: StateBankView, row: int, column: int) -> StateRecord:
    record = view.cloned_records[row][column]
    if record is None:
        raise ValueError("present State Bank records require cloned records")
    return record


def _empty_reason(audit: RetrievalFilterAudit, owner_record_count: int) -> RetrievalReason:
    if owner_record_count == 0:
        return RetrievalReason.EMPTY_BANK
    if audit.n_state == 0:
        return RetrievalReason.EMPTY_HEAD_PARTITION
    if audit.invalid_count == audit.n_state:
        return RetrievalReason.ALL_INVALID
    if audit.retrieval_ineligible_count == audit.n_state:
        return RetrievalReason.ALL_RETRIEVAL_INELIGIBLE
    if audit.future_count == audit.n_state:
        return RetrievalReason.ALL_FUTURE
    if audit.outside_window_count == audit.n_state:
        return RetrievalReason.ALL_OUTSIDE_WINDOW
    if audit.below_similarity_count == audit.n_state:
        return RetrievalReason.BELOW_SIMILARITY
    return RetrievalReason.NO_MATCH


def _snapshot_values_equal(left: object, right: object) -> bool:
    """Compare frozen typed-record snapshots, including every tensor-backed payload field."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _snapshot_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _snapshot_values_equal(
                getattr(left, field.name),
                getattr(right, field.name),
            )
            for field in fields(left)
        )
    return bool(left == right)
