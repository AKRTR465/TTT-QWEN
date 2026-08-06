"""Implement complete threshold retrieval over one typed State Bank snapshot.

Inputs: q_target, hard operators, resolved time windows, owner IDs, and a row-wise Bank view.
Outputs: all passing typed records, aligned scores/masks, structured status, counts, and audit.
Forbidden: fixed Top-K, ANN, Reader arithmetic, labels, future records, or Bank mutation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, RetrieverConfig
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    OPERATORS,
    Operator,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.state_bank import (
    RETRIEVAL_HEAD_ORDER,
    HeadType,
    RetrievalHistoryRecord,
    RetrievalHistoryView,
    StateRecord,
    StructuredStateBank,
)

type RetrievalCandidate = StateRecord | RetrievalHistoryRecord


class RetrievalStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class RetrievalReason(StrEnum):
    MATCHED = "matched"
    EMPTY_BANK = "empty_bank"
    NO_MATCH = "no_match"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_TIME = "unsupported_time"
    DEGENERATE_QUERY = "degenerate_q_target"
    INVALID_TIME = "invalid_time"


@dataclass(frozen=True, slots=True)
class RetrievalFilterAudit:
    n_state: int
    selected_count: int


@dataclass(frozen=True, slots=True)
class RetrieverOutput:
    selected_record_ids: tuple[tuple[str, ...], ...]
    selected_scores: tuple[tuple[float, ...], ...]
    selected_records: tuple[tuple[RetrievalCandidate, ...], ...]
    candidate_record_ids: tuple[tuple[str | None, ...], ...]
    candidate_records: tuple[tuple[RetrievalCandidate | None, ...], ...]
    candidate_head_types: tuple[tuple[HeadType | None, ...], ...]
    state_embeddings: Tensor
    scores: Tensor
    present_mask: Tensor
    record_valid_mask: Tensor
    retrieval_eligible_mask: Tensor
    causal_mask: Tensor
    predicted_head_mask: Tensor
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
    candidate_sequence_ids: Tensor | None = None
    candidate_head_codes: Tensor | None = None
    candidate_operator_codes: Tensor | None = None
    candidate_timestamps: Tensor | None = None
    candidate_time_ranges: Tensor | None = None

    def require_tensor_metadata(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return the materialized candidate metadata tensors."""

        return (
            self.candidate_sequence_ids,  # type: ignore[return-value]
            self.candidate_head_codes,
            self.candidate_operator_codes,
            self.candidate_timestamps,
            self.candidate_time_ranges,
        )

    def candidate_record_id(self, row: int, column: int) -> str | None:
        value = self.candidate_record_ids[row][column]
        if value is not None:
            return value
        sequence_ids, _, _, _, _ = self.require_tensor_metadata()
        sequence_id = int(sequence_ids[row, column].item())
        return f"retrieval-{sequence_id:08d}" if sequence_id >= 0 else None

    def __post_init__(self) -> None:
        shape = self.scores.shape
        tensor_metadata = (
            self.candidate_sequence_ids,
            self.candidate_head_codes,
            self.candidate_operator_codes,
            self.candidate_timestamps,
            self.candidate_time_ranges,
        )
        if any(value is None for value in tensor_metadata):
            sequence_ids = torch.full(shape, -1, dtype=torch.int64, device=self.scores.device)
            head_codes = torch.full_like(sequence_ids, -1)
            operator_codes = torch.full_like(sequence_ids, -1)
            timestamps = torch.full(shape, -1.0, dtype=torch.float64, device=self.scores.device)
            time_ranges = torch.full(
                (*shape, 2), -1.0, dtype=torch.float64, device=self.scores.device
            )
            for row, records in enumerate(self.candidate_records):
                for column, record in enumerate(records):
                    if record is None:
                        continue
                    sequence_ids[row, column] = column
                    head_codes[row, column] = tuple(HeadType).index(record.head_type)
                    if isinstance(record, RetrievalHistoryRecord):
                        operator_codes[row, column] = OPERATORS.index(record.operator)
                    if record.timestamp is not None:
                        timestamps[row, column] = record.timestamp
                    else:
                        assert record.time_range is not None
                        time_ranges[row, column] = torch.tensor(
                            record.time_range, dtype=torch.float64, device=self.scores.device
                        )
            object.__setattr__(self, "candidate_sequence_ids", sequence_ids)
            object.__setattr__(self, "candidate_head_codes", head_codes)
            object.__setattr__(self, "candidate_operator_codes", operator_codes)
            object.__setattr__(self, "candidate_timestamps", timestamps)
            object.__setattr__(self, "candidate_time_ranges", time_ranges)



class EmbeddingStateRetriever(nn.Module):  # type: ignore[misc]
    """Zero-parameter exact scorer; soft scores retain q_target gradients."""

    def __init__(self, config: RetrieverConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        state_bank: StructuredStateBank,
        history: RetrievalHistoryView,
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> RetrieverOutput:
        """Reproject one write-before history snapshot in the grad-enabled Query path."""

        q_target = query.q_target
        hard_operators = query.hard_operators
        time_resolutions = query.time.resolutions
        with trace_cuda_phase("retrieval_project_and_score"):
            aligned_embeddings, scores, query_norms = _project_and_score_history(
                state_bank,
                history,
                q_target,
                chunk_size=self.config.score_chunk_size,
                normalization_eps=self.config.normalization_eps,
            )
        batch_size = int(q_target.shape[0])
        operators = tuple(hard_operators)
        resolutions = tuple(time_resolutions)
        query_video_ids = tuple(video_ids)
        query_trajectory_ids = tuple(trajectory_ids)
        sequence_ids, head_codes, operator_codes = history.require_tensor_metadata()
        selected_mask = torch.zeros_like(history.present_mask)
        predicted_head_mask = _predicted_head_mask(history, operators)
        statuses: list[RetrievalStatus] = []
        reasons: list[RetrievalReason] = []
        audits: list[RetrievalFilterAudit] = []
        selected_ids: list[tuple[str, ...]] = []
        selected_scores: list[tuple[float, ...]] = []
        selected_records: list[tuple[RetrievalCandidate, ...]] = []
        n_retrieved = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
        causal_mask = _causal_mask(history, resolutions, scores.device)
        for row in range(batch_size):
            row_result = self._retrieve_row(
                row,
                scores,
                selected_mask,
                operators[row],
                resolutions[row],
                history,
                causal_mask,
                predicted_head_mask,
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
            candidate_record_ids=history.record_ids,
            candidate_records=history.cloned_records,
            candidate_head_types=history.head_types,
            state_embeddings=aligned_embeddings,
            scores=scores,
            present_mask=history.present_mask.detach().clone(),
            record_valid_mask=history.record_valid_mask.detach().clone(),
            retrieval_eligible_mask=history.retrieval_eligible_mask.detach().clone(),
            causal_mask=causal_mask.detach().clone(),
            predicted_head_mask=predicted_head_mask.detach().clone(),
            selected_mask=selected_mask,
            status=tuple(statuses),
            reason=tuple(reasons),
            hard_operators=operators,
            time_resolutions=resolutions,
            n_state=history.n_state.detach().clone(),
            n_retrieved=n_retrieved,
            audit=tuple(audits),
            video_ids=query_video_ids,
            trajectory_ids=query_trajectory_ids,
            bank_video_ids=history.video_ids,
            bank_trajectory_ids=history.trajectory_ids,
            bank_versions=history.bank_versions,
            candidate_sequence_ids=sequence_ids.detach().clone(),
            candidate_head_codes=head_codes.detach().clone(),
            candidate_operator_codes=operator_codes.detach().clone(),
            candidate_timestamps=history.timestamps.detach().clone(),
            candidate_time_ranges=history.time_ranges.detach().clone(),
        )

    def _retrieve_row(
        self,
        row: int,
        scores: Tensor,
        selected_mask: Tensor,
        operator: Operator,
        resolution: TimeResolution,
        state_view: RetrievalHistoryView,
        causal_mask: Tensor,
        predicted_head_mask: Tensor,
        video_id: str,
        trajectory_id: str,
        query_norm: float,
    ) -> tuple[
        RetrievalStatus,
        RetrievalReason,
        RetrievalFilterAudit,
        tuple[str, ...],
        tuple[float, ...],
        tuple[RetrievalCandidate, ...],
    ]:
        n_state = int(state_view.n_state[row].item())
        owner_count = int(state_view.owner_record_counts[row].item())
        if resolution.status is TimeResolutionStatus.INVALID:
            return _rejected_row(
                RetrievalStatus.INVALID,
                RetrievalReason.INVALID_TIME,
                n_state,
            )
        if resolution.status is TimeResolutionStatus.UNSUPPORTED:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_TIME,
                n_state,
            )
        expected_head = OPERATOR_TO_HEAD_TYPE[operator]
        if operator is Operator.UNSUPPORTED or expected_head is None:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_OPERATOR,
                n_state,
            )
        if not math.isfinite(query_norm) or query_norm <= self.config.normalization_eps:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.DEGENERATE_QUERY,
                n_state,
            )

        sequence_ids, head_codes, _ = state_view.require_tensor_metadata()
        present = state_view.present_mask[row]
        predicted = predicted_head_mask[row]
        valid = state_view.record_valid_mask[row]
        eligible = state_view.retrieval_eligible_mask[row]
        causal = causal_mask[row]
        after_head = predicted
        after_valid = after_head & valid
        after_eligible = after_valid & eligible
        after_causal = after_eligible & causal
        outside_mask = torch.zeros_like(present)
        if resolution.window.start_time is not None:
            start = state_view.timestamps.new_tensor(resolution.window.start_time)
            end = state_view.timestamps.new_tensor(resolution.window.end_time)
            atomic = head_codes[row] == RETRIEVAL_HEAD_ORDER.index(HeadType.O2)
            timestamp_overlap = (state_view.timestamps[row] >= start) & (
                state_view.timestamps[row] <= end
            )
            range_overlap = (state_view.time_ranges[row, :, 0] <= end) & (
                state_view.time_ranges[row, :, 1] >= start
            )
            has_timestamp = state_view.timestamps[row] >= 0.0
            intersects = torch.where(has_timestamp, timestamp_overlap, range_overlap)
            outside_mask = after_causal & atomic & ~intersects
        after_window = after_causal & ~outside_mask
        similarity_threshold = scores.new_tensor(self.config.record_similarity_threshold)
        below_mask = after_window & (scores[row].detach() < similarity_threshold)
        chosen = after_window & ~below_mask
        selected_mask[row] = chosen
        ordered = torch.nonzero(chosen, as_tuple=False).flatten()
        if ordered.numel():
            id_order = torch.argsort(
                sequence_ids[row].index_select(0, ordered), stable=True
            )
            ordered = ordered.index_select(0, id_order)
            score_order = torch.argsort(
                scores[row].detach().index_select(0, ordered), descending=True, stable=True
            )
            ordered = ordered.index_select(0, score_order)
        ordered_columns = tuple(int(value) for value in ordered.detach().cpu().tolist())
        ids = tuple(_required_record_id(state_view, row, column) for column in ordered_columns)
        row_scores = tuple(float(scores[row, column].detach().item()) for column in ordered_columns)
        records = tuple(
            _materialize_history_record(state_view, row, column) for column in ordered_columns
        )
        audit = RetrievalFilterAudit(n_state=n_state, selected_count=len(ids))
        if ids:
            return RetrievalStatus.OK, RetrievalReason.MATCHED, audit, ids, row_scores, records
        reason = _empty_reason(audit, owner_count)
        return RetrievalStatus.EMPTY, reason, audit, (), (), ()


def build_state_retriever(config: ProjectConfig | None = None) -> EmbeddingStateRetriever:
    if config is None:
        raise ValueError("build_state_retriever requires a validated ProjectConfig")
    return EmbeddingStateRetriever(config.retriever)


def _rejected_row(
    status: RetrievalStatus,
    reason: RetrievalReason,
    n_state: int,
) -> tuple[
    RetrievalStatus,
    RetrievalReason,
    RetrievalFilterAudit,
    tuple[str, ...],
    tuple[float, ...],
    tuple[RetrievalCandidate, ...],
]:
    audit = RetrievalFilterAudit(n_state=n_state, selected_count=0)
    return status, reason, audit, (), (), ()


def _project_history_sources(
    state_bank: StructuredStateBank,
    view: RetrievalHistoryView,
    *,
    chunk_size: int,
) -> Tensor:
    """Recreate trainable keys without reconnecting detached Support encoders."""

    _, head_codes, _ = view.require_tensor_metadata()
    width = view.sources.shape[1]
    rows: list[Tensor] = []
    for row in range(view.sources.shape[0]):
        count = int(view.n_state[row].item())
        if count:
            projected_chunks: list[Tensor] = []
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                projected_chunks.append(
                    state_bank.project_codes(
                        view.sources[row, start:end],
                        head_codes[row, start:end],
                    )
                )
            projected = torch.cat(projected_chunks, dim=0)
            padding = projected.new_zeros((width - count, projected.shape[-1]))
            rows.append(torch.cat((projected, padding), dim=0))
        else:
            parameter = next(state_bank.semantic_projector.parameters())
            rows.append(parameter.new_zeros((width, state_bank.config.semantic_dim)))
    return torch.stack(rows)


def _project_and_score_history(
    state_bank: StructuredStateBank,
    view: RetrievalHistoryView,
    q_target: Tensor,
    *,
    chunk_size: int,
    normalization_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    aligned = _project_history_sources(state_bank, view, chunk_size=chunk_size)
    query_fp32 = q_target.float()
    query_norms = torch.linalg.vector_norm(query_fp32, dim=-1)
    query_usable = (
        torch.isfinite(query_fp32).all(dim=-1)
        & torch.isfinite(query_norms)
        & (query_norms > normalization_eps)
    )
    safe_query = torch.where(query_usable.unsqueeze(-1), query_fp32, torch.zeros_like(query_fp32))
    normalized_query = F.normalize(safe_query, dim=-1, eps=normalization_eps)
    normalized_state = F.normalize(aligned.float(), dim=-1, eps=normalization_eps)
    scores = torch.einsum("bd,bnd->bn", normalized_query, normalized_state)
    scores = torch.where(view.present_mask, scores, torch.zeros_like(scores))
    return aligned, scores, query_norms


def _predicted_head_mask(
    view: RetrievalHistoryView,
    operators: Sequence[Operator],
) -> Tensor:
    _, head_codes, _ = view.require_tensor_metadata()
    mask = torch.zeros_like(view.present_mask)
    for row, operator in enumerate(operators):
        expected = OPERATOR_TO_HEAD_TYPE[operator]
        if expected is None:
            continue
        mask[row] = head_codes[row] == RETRIEVAL_HEAD_ORDER.index(expected)
    return mask & view.present_mask


def _causal_mask(
    view: RetrievalHistoryView,
    resolutions: Sequence[TimeResolution],
    device: torch.device,
) -> Tensor:
    query_times = torch.tensor(
        tuple(value.window.query_time for value in resolutions),
        dtype=torch.float64,
        device=device,
    ).unsqueeze(1)
    record_end = torch.where(
        view.timestamps >= 0.0,
        view.timestamps,
        view.time_ranges[..., 1],
    )
    return view.present_mask & (record_end <= query_times)


def _required_record_id(view: RetrievalHistoryView, row: int, column: int) -> str:
    sequence_ids, _, _ = view.require_tensor_metadata()
    record_id = view.record_ids[row][column]
    if record_id is None:
        sequence_id = int(sequence_ids[row, column].item())
        record_id = f"retrieval-{sequence_id:08d}"
    return record_id


def _materialize_history_record(
    view: RetrievalHistoryView, row: int, column: int
) -> RetrievalHistoryRecord:
    from ttt_svcbench_qwen.query_encoder import OPERATORS

    _, head_codes, operator_codes = view.require_tensor_metadata()
    head = view.head_types[row][column]
    if head is None:
        head_code = int(head_codes[row, column].item())
        head = RETRIEVAL_HEAD_ORDER[head_code]
    record_id = _required_record_id(view, row, column)
    operator_code = int(operator_codes[row, column].item())
    timestamp_value = float(view.timestamps[row, column].item())
    if timestamp_value >= 0.0:
        timestamp: float | None = timestamp_value
        time_range: tuple[float, float] | None = None
    else:
        values = view.time_ranges[row, column].detach().cpu()
        timestamp = None
        time_range = (float(values[0]), float(values[1]))
    return RetrievalHistoryRecord(
        record_id=record_id,
        video_id=view.video_ids[row],
        trajectory_id=view.trajectory_ids[row],
        head_type=head,
        operator=OPERATORS[operator_code],
        semantic_source=view.sources[row, column].detach().clone(),
        timestamp=timestamp,
        time_range=time_range,
        valid=bool(view.record_valid_mask[row, column]),
        retrieval_eligible=bool(view.retrieval_eligible_mask[row, column]),
    )


def _empty_reason(audit: RetrievalFilterAudit, owner_record_count: int) -> RetrievalReason:
    if owner_record_count == 0 or audit.n_state == 0:
        return RetrievalReason.EMPTY_BANK
    return RetrievalReason.NO_MATCH
