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
    OPERATORS,
    Operator,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.state_bank import (
    RETRIEVAL_HEAD_ORDER,
    HeadType,
    RetrievalHistoryRecord,
    RetrievalHistoryView,
    StateRecord,
    StateRecordKind,
    StructuredStateBank,
    clone_retrieval_history_record,
    clone_state_record,
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
            self.head_partition_excluded_count
            + self.query_rejected_count
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
    candidate_lifecycle_ids: tuple[tuple[str | None, ...], ...] = ()

    def require_tensor_metadata(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return materialized candidate metadata or fail closed at the runtime boundary."""

        sequence_ids = self.candidate_sequence_ids
        head_codes = self.candidate_head_codes
        operator_codes = self.candidate_operator_codes
        timestamps = self.candidate_timestamps
        time_ranges = self.candidate_time_ranges
        if not isinstance(sequence_ids, Tensor):
            raise RuntimeError("RetrieverOutput candidate_sequence_ids are unavailable")
        if not isinstance(head_codes, Tensor):
            raise RuntimeError("RetrieverOutput candidate_head_codes are unavailable")
        if not isinstance(operator_codes, Tensor):
            raise RuntimeError("RetrieverOutput candidate_operator_codes are unavailable")
        if not isinstance(timestamps, Tensor):
            raise RuntimeError("RetrieverOutput candidate_timestamps are unavailable")
        if not isinstance(time_ranges, Tensor):
            raise RuntimeError("RetrieverOutput candidate_time_ranges are unavailable")
        return sequence_ids, head_codes, operator_codes, timestamps, time_ranges

    def candidate_record_id(self, row: int, column: int) -> str | None:
        value = self.candidate_record_ids[row][column]
        if value is not None:
            return value
        sequence_ids, _, _, _, _ = self.require_tensor_metadata()
        sequence_id = int(sequence_ids[row, column].item())
        return f"retrieval-{sequence_id:08d}" if sequence_id >= 0 else None

    def candidate_head_type(self, row: int, column: int) -> HeadType | None:
        value = self.candidate_head_types[row][column]
        if value is not None:
            return value
        _, head_codes, _, _, _ = self.require_tensor_metadata()
        code = int(head_codes[row, column].item())
        return RETRIEVAL_HEAD_ORDER[code] if 0 <= code < len(RETRIEVAL_HEAD_ORDER) else None

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
            or self.record_valid_mask.shape != shape
            or self.retrieval_eligible_mask.shape != shape
            or self.causal_mask.shape != shape
            or self.predicted_head_mask.shape != shape
            or self.selected_mask.shape != shape
            or self.present_mask.dtype != torch.bool
            or self.record_valid_mask.dtype != torch.bool
            or self.retrieval_eligible_mask.dtype != torch.bool
            or self.causal_mask.dtype != torch.bool
            or self.predicted_head_mask.dtype != torch.bool
            or self.selected_mask.dtype != torch.bool
        ):
            raise ValueError("Retriever masks must be bool [B, N_s]")
        tensors = (
            self.state_embeddings,
            self.scores,
            self.present_mask,
            self.record_valid_mask,
            self.retrieval_eligible_mask,
            self.causal_mask,
            self.predicted_head_mask,
            self.selected_mask,
        )
        if any(tensor.device != self.scores.device for tensor in tensors):
            raise ValueError("Retriever aligned tensors must share one device")
        batch_size, width = shape
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
        sequence_ids, head_codes, operator_codes, timestamps, time_ranges = (
            self.require_tensor_metadata()
        )
        integer_metadata = (sequence_ids, head_codes, operator_codes)
        if any(value.shape != shape or value.dtype != torch.int64 for value in integer_metadata):
            raise ValueError("Retriever candidate integer metadata must be int64 [B, N_s]")
        if (
            timestamps.shape != shape
            or timestamps.dtype != torch.float64
            or time_ranges.shape != (*shape, 2)
            or time_ranges.dtype != torch.float64
        ):
            raise ValueError("Retriever candidate time metadata is invalid")
        if any(
            value.device != self.scores.device
            for value in (*integer_metadata, timestamps, time_ranges)
        ):
            raise ValueError("Retriever candidate tensor metadata must share the score device")
        metadata = (
            self.selected_record_ids,
            self.selected_scores,
            self.selected_records,
            self.candidate_record_ids,
            self.candidate_records,
            self.candidate_head_types,
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
            self.candidate_lifecycle_ids or tuple(() for _ in range(batch_size)),
        )
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("Retriever metadata must contain one entry per batch item")
        if (
            any(len(row) != width for row in self.candidate_record_ids)
            or any(len(row) != width for row in self.candidate_records)
            or any(len(row) != width for row in self.candidate_head_types)
        ):
            raise ValueError("candidate record snapshots must align to the padded score width")
        if self.candidate_lifecycle_ids and any(
            len(row) != width for row in self.candidate_lifecycle_ids
        ):
            raise ValueError("candidate lifecycle metadata must align to padded score width")
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
        if bool(torch.any(self.predicted_head_mask & ~self.present_mask)):
            raise ValueError("Retriever predicted-head mask cannot include padding")
        if bool(torch.any(self.selected_mask & ~self.predicted_head_mask)):
            raise ValueError("Retriever selections must stay inside the predicted head")
        if bool(torch.any(self.record_valid_mask & ~self.present_mask)) or bool(
            torch.any(self.retrieval_eligible_mask & ~self.record_valid_mask)
        ):
            raise ValueError("Retriever candidate masks are inconsistent")
        if bool(torch.any(self.causal_mask & ~self.present_mask)):
            raise ValueError("Retriever causal mask cannot include padding")
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

    def _validate_row(self, row: int) -> None:
        sequence_ids, head_codes, _, _, _ = self.require_tensor_metadata()
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
        candidate_by_id: dict[str, RetrievalCandidate] = {}
        tensor_only = (
            all(value is None for value in present_ids)
            and all(value is None for value in self.candidate_head_types[row])
            and all(value is None for value in candidate_records)
        )
        if tensor_only:
            present = self.present_mask[row]
            sequences = sequence_ids[row]
            heads = head_codes[row]
            if bool(torch.any(sequences[present] < 0)) or bool(
                torch.any((heads[present] < 0) | (heads[present] >= len(RETRIEVAL_HEAD_ORDER)))
            ):
                raise ValueError("present tensor candidates require valid sequence/head codes")
            if bool(torch.any(sequences[~present] != -1)) or bool(
                torch.any(heads[~present] != -1)
            ):
                raise ValueError("padded tensor candidate metadata must use -1")
            live_sequences = sequences[present]
            if torch.unique(live_sequences).numel() != live_sequences.numel():
                raise ValueError("Retriever candidate sequence IDs must be unique per row")
            if expected_head is not None:
                expected_mask = present & (
                    heads == RETRIEVAL_HEAD_ORDER.index(expected_head)
                )
            else:
                expected_mask = torch.zeros_like(present)
            if not torch.equal(self.predicted_head_mask[row], expected_mask):
                raise ValueError("Retriever predicted-head mask disagrees with tensor metadata")
        else:
            for column, _record_id in enumerate(present_ids):
                is_present = bool(self.present_mask[row, column])
                resolved_id = self.candidate_record_id(row, column)
                resolved_head = self.candidate_head_type(row, column)
                if (resolved_id is not None) != is_present:
                    raise ValueError("Retriever candidate IDs must align to present_mask")
                candidate = candidate_records[column]
                if candidate is None:
                    if not is_present and resolved_head is not None:
                        raise ValueError("padded Retriever candidate head type must be None")
                    if is_present and resolved_head is None:
                        raise ValueError("present tensor candidate requires a valid head code")
                    continue
                if resolved_head is not candidate.head_type:
                    raise ValueError("Retriever candidate head metadata is inconsistent")
                if (
                    candidate.record_id != resolved_id
                    or candidate.video_id != self.bank_video_ids[row]
                    or candidate.trajectory_id != self.bank_trajectory_ids[row]
                ):
                    raise ValueError(
                        "Retriever candidate typed snapshot metadata is inconsistent"
                    )
                if isinstance(candidate, StateRecord) and (
                    candidate.semantic_embedding.dtype != self.state_embeddings.dtype
                    or candidate.semantic_embedding.device != self.state_embeddings.device
                    or not torch.equal(
                        candidate.semantic_embedding, self.state_embeddings[row, column]
                    )
                ):
                    raise ValueError("Retriever StateRecord snapshot semantics are inconsistent")
                if isinstance(candidate, RetrievalHistoryRecord) and (
                    candidate.semantic_source.device != self.state_embeddings.device
                ):
                    raise ValueError("Retriever history source and projected key devices differ")
                if bool(self.record_valid_mask[row, column]) is not candidate.valid:
                    raise ValueError("Retriever candidate validity metadata is inconsistent")
                if isinstance(candidate, RetrievalHistoryRecord) and (
                    bool(self.retrieval_eligible_mask[row, column])
                    is not candidate.retrieval_eligible
                ):
                    raise ValueError("Retriever candidate eligibility metadata is inconsistent")
                if bool(self.predicted_head_mask[row, column]) is not (
                    expected_head is not None and candidate.head_type is expected_head
                ):
                    raise ValueError(
                        "Retriever predicted-head mask disagrees with candidate metadata"
                    )
                candidate_by_id[candidate.record_id] = candidate
            live_candidate_ids = tuple(
                self.candidate_record_id(row, column)
                for column in range(self.scores.shape[1])
                if bool(self.present_mask[row, column])
            )
            if len(set(live_candidate_ids)) != len(live_candidate_ids):
                raise ValueError("Retriever candidate record IDs must be unique per row")
        selected_columns = torch.nonzero(self.selected_mask[row], as_tuple=False).flatten().tolist()
        mask_ids = tuple(self.candidate_record_id(row, column) for column in selected_columns)
        if set(mask_ids) != set(ids):
            raise ValueError("Retriever selected IDs must exactly match selected_mask")
        if any(
            record.record_id in candidate_by_id
            and not _snapshot_values_equal(record, candidate_by_id[record.record_id])
            for record in records
        ):
            raise ValueError("Retriever selected typed records must match the candidate snapshot")
        score_by_id = {
            self.candidate_record_id(row, column): float(
                self.scores[row, column].detach().item()
            )
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


class EmbeddingStateRetriever(nn.Module):  # type: ignore[misc]
    """Zero-parameter exact scorer; soft scores retain q_target gradients."""

    def __init__(self, config: RetrieverConfig) -> None:
        super().__init__()
        _validate_retriever_config(config)
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

        if not isinstance(state_bank, StructuredStateBank):
            raise TypeError("Retriever requires the StructuredStateBank projector owner")
        if not isinstance(history, RetrievalHistoryView):
            raise TypeError("Retriever requires a write-before RetrievalHistoryView")
        history.assert_snapshot_current()
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
        batch_size = _validate_retrieval_inputs(
            self.config,
            q_target,
            hard_operators,
            time_resolutions,
            history,
            video_ids,
            trajectory_ids,
            aligned_embeddings,
        )
        operators = _normalize_operators(hard_operators, batch_size)
        resolutions = _normalize_resolutions(time_resolutions, batch_size)
        query_video_ids = _normalize_owner_ids(video_ids, batch_size, "video_ids")
        query_trajectory_ids = _normalize_owner_ids(trajectory_ids, batch_size, "trajectory_ids")
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
            candidate_records=tuple(
                tuple(_clone_candidate(record) if record is not None else None for record in row)
                for row in history.cloned_records
            ),
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
            candidate_lifecycle_ids=history.lifecycle_ids,
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
        if owner_count != n_state:
            raise ValueError("all-head retrieval history must expose every owner record")
        owner_matches = (
            state_view.video_ids[row] == video_id
            and state_view.trajectory_ids[row] == trajectory_id
        )
        if not owner_matches:
            return _rejected_row(
                RetrievalStatus.INVALID,
                RetrievalReason.OWNER_MISMATCH,
                n_state,
                0,
                owner_mismatch=True,
            )
        if resolution.status is TimeResolutionStatus.INVALID:
            return _rejected_row(
                RetrievalStatus.INVALID,
                RetrievalReason.INVALID_TIME,
                n_state,
                0,
                query_rejected=True,
            )
        if resolution.status is TimeResolutionStatus.UNSUPPORTED:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_TIME,
                n_state,
                0,
                query_rejected=True,
            )
        expected_head = OPERATOR_TO_HEAD_TYPE[operator]
        if operator is Operator.UNSUPPORTED or expected_head is None:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.UNSUPPORTED_OPERATOR,
                n_state,
                0,
                query_rejected=True,
            )
        if not math.isfinite(query_norm) or query_norm <= self.config.normalization_eps:
            return _rejected_row(
                RetrievalStatus.UNSUPPORTED,
                RetrievalReason.DEGENERATE_QUERY,
                n_state,
                0,
                query_rejected=True,
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
        count_tensors = torch.stack(
            (
                (present & ~predicted).sum(),
                (after_head & ~valid).sum(),
                (after_valid & ~eligible).sum(),
                (after_eligible & ~causal).sum(),
                outside_mask.sum(),
                below_mask.sum(),
                chosen.sum(),
            )
        )
        head_excluded, invalid, ineligible, future, outside, below, _ = (
            int(value) for value in count_tensors.detach().cpu().tolist()
        )
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
        ("score_chunk_size", config.score_chunk_size, 256),
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
    state_view: RetrievalHistoryView,
    video_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    state_embeddings: Tensor,
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
    if state_embeddings.shape != (*state_view.present_mask.shape, config.semantic_dim):
        raise ValueError("projected retrieval embeddings must be [B, N, 512]")
    if state_embeddings.shape[0] != batch_size:
        raise ValueError("q_target and State Bank view batch sizes must match")
    if state_embeddings.device != q_target.device:
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
    tuple[RetrievalCandidate, ...],
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


def _record_end(record: RetrievalCandidate) -> float:
    if record.timestamp is not None:
        return float(record.timestamp)
    assert record.time_range is not None
    return float(record.time_range[1])


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


def _clone_candidate(record: RetrievalCandidate) -> RetrievalCandidate:
    if isinstance(record, StateRecord):
        return clone_state_record(record)
    return clone_retrieval_history_record(record)


def _requires_atomic_window_filter(kind: StateRecordKind, window: TimeWindow) -> bool:
    return bool(kind is StateRecordKind.O2_CONFIRMED and window.start_time is not None)


def _intersects_window(record: RetrievalCandidate, window: TimeWindow) -> bool:
    if window.start_time is None:
        return True
    if record.timestamp is not None:
        return bool(window.start_time <= record.timestamp <= window.end_time)
    assert record.time_range is not None
    start, end = record.time_range
    return bool(start <= window.end_time and end >= window.start_time)


def _required_record_id(view: RetrievalHistoryView, row: int, column: int) -> str:
    sequence_ids, _, _ = view.require_tensor_metadata()
    record_id = view.record_ids[row][column]
    if record_id is None:
        sequence_id = int(sequence_ids[row, column].item())
        if sequence_id >= 0:
            record_id = f"retrieval-{sequence_id:08d}"
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("present State Bank records require record IDs")
    return record_id


def _required_record(view: RetrievalHistoryView, row: int, column: int) -> RetrievalCandidate:
    record = view.cloned_records[row][column]
    if record is None:
        raise ValueError("present State Bank records require cloned records")
    return record


def _materialize_history_record(
    view: RetrievalHistoryView, row: int, column: int
) -> RetrievalHistoryRecord:
    from ttt_svcbench_qwen.query_encoder import OPERATORS

    _, head_codes, operator_codes = view.require_tensor_metadata()
    head = view.head_types[row][column]
    if head is None:
        head_code = int(head_codes[row, column].item())
        head = (
            RETRIEVAL_HEAD_ORDER[head_code]
            if 0 <= head_code < len(RETRIEVAL_HEAD_ORDER)
            else None
        )
    record_id = _required_record_id(view, row, column)
    if head is None:
        raise ValueError("selected retrieval columns require tensor metadata")
    operator_code = int(operator_codes[row, column].item())
    if not 0 <= operator_code < len(OPERATORS):
        raise ValueError("selected retrieval operator code is invalid")
    timestamp_value = float(view.timestamps[row, column].item())
    if timestamp_value >= 0.0:
        timestamp: float | None = timestamp_value
        time_range: tuple[float, float] | None = None
    else:
        values = view.time_ranges[row, column].detach().cpu()
        timestamp = None
        time_range = (float(values[0]), float(values[1]))
    lifecycle = view.lifecycle_ids[row][column] if view.lifecycle_ids else None
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
        lifecycle_id=lifecycle,
    )


def _intersects_window_metadata(
    view: RetrievalHistoryView,
    row: int,
    column: int,
    window: TimeWindow,
) -> bool:
    if window.start_time is None:
        return True
    timestamp = float(view.timestamps[row, column].item())
    if timestamp >= 0.0:
        return bool(window.start_time <= timestamp <= window.end_time)
    time_range = view.time_ranges[row, column]
    start = float(time_range[0].item())
    end = float(time_range[1].item())
    return bool(start <= window.end_time and end >= window.start_time)


def _empty_reason(audit: RetrievalFilterAudit, owner_record_count: int) -> RetrievalReason:
    if owner_record_count == 0:
        return RetrievalReason.EMPTY_BANK
    if audit.n_state == 0 or audit.head_partition_excluded_count == audit.n_state:
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
