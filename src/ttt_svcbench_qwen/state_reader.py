"""Resample retrieved state semantics and deterministically read exact integers.

Inputs: ``q_target``, the complete :class:`RetrieverOutput`, effective hard operators,
resolved time windows, and the pinned Qwen tokenizer used by the eventual composer.
Outputs: 16 learned State Tokens plus immutable, tokenizer-audited exact-count results.
Forbidden: Top-K truncation, neural count regression, ground-truth substitution, retrieval,
Bank mutation, or natural-language generation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, StateResamplerConfig
from ttt_svcbench_qwen.identity_bank import ConfirmedIdentity
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import (
    E1Payload,
    E2Payload,
    O1Payload,
    StateBankRuntimeState,
    StateRecord,
    StructuredStateBank,
    clone_state_record,
)
from ttt_svcbench_qwen.state_retriever import (
    RetrievalFilterAudit,
    RetrievalReason,
    RetrievalStatus,
    RetrieverOutput,
)


class ReaderStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


type AuditValue = str | int | float | bool | None


class NumberTokenizerProtocol(Protocol):
    """The minimal pinned-tokenizer surface needed for canonical integer audit."""

    name_or_path: str
    vocab_size: int

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class StateResamplerOutput:
    """Fixed-width State Tokens for the selected semantic records."""

    hidden_states: Tensor
    state_tokens: Tensor
    selected_record_ids: tuple[tuple[str, ...], ...]
    retrieval_status: tuple[RetrievalStatus, ...]
    state_token_valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class ReaderResult:
    status: ReaderStatus
    exact_count: int | None
    number_token_ids: tuple[int, ...]
    selected_record_ids: tuple[str, ...]
    operator: Operator
    time_window: TimeWindow
    audit_fields: tuple[tuple[str, AuditValue], ...]


class _StateResamplerLayer(nn.Module):  # type: ignore[misc]
    """One explicit three-sublayer Pre-LN Perceiver/Q-Former block."""

    def __init__(self, config: StateResamplerConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.hidden_dim = hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.self_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        self.self_q = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.self_k = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.self_v = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.self_out = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)

        self.cross_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        self.cross_q = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.cross_k = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.cross_v = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)
        self.cross_out = nn.Linear(hidden_dim, hidden_dim, bias=config.attention_bias)

        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        self.ffn_in = nn.Linear(hidden_dim, config.ffn_dim, bias=config.attention_bias)
        self.ffn_out = nn.Linear(config.ffn_dim, hidden_dim, bias=config.attention_bias)

    def forward(
        self,
        queries: Tensor,
        records: Tensor,
        record_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(queries)
        self_queries = self._split_heads(self.self_q(normalized))
        self_keys = self._split_heads(self.self_k(normalized))
        self_values = self._split_heads(self.self_v(normalized))
        self_logits = torch.matmul(self_queries.float(), self_keys.float().transpose(-1, -2))
        self_weights = torch.softmax(self_logits / math.sqrt(self.head_dim), dim=-1)
        self_context = torch.matmul(self_weights.to(self_values.dtype), self_values)
        queries = queries + self.self_out(self._merge_heads(self_context))

        normalized = self.cross_norm(queries)
        cross_queries = self._split_heads(self.cross_q(normalized))
        cross_keys = self._split_heads(self.cross_k(records))
        cross_values = self._split_heads(self.cross_v(records))
        cross_logits = torch.matmul(
            cross_queries.float(),
            cross_keys.float().transpose(-1, -2),
        ) / math.sqrt(self.head_dim)
        valid_pairs = record_mask[:, None, None, :]
        cross_logits = cross_logits.masked_fill(~valid_pairs, torch.finfo(torch.float32).min)
        cross_weights = torch.softmax(cross_logits, dim=-1)
        cross_weights = torch.where(valid_pairs, cross_weights, 0.0)
        denominator = cross_weights.sum(dim=-1, keepdim=True)
        cross_weights = cross_weights / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        cross_context = torch.matmul(cross_weights.to(cross_values.dtype), cross_values)
        queries = queries + self.cross_out(self._merge_heads(cross_context))

        feed_forward = self.ffn_in(self.ffn_norm(queries))
        queries = queries + self.ffn_out(F.gelu(feed_forward))
        return queries

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, item_count, _ = values.shape
        return values.reshape(
            batch_size,
            item_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        return values.transpose(1, 2).reshape(
            values.shape[0],
            values.shape[2],
            self.hidden_dim,
        )


class StateResampler(nn.Module):  # type: ignore[misc]
    """Compress every selected semantic record into 16 learned State Tokens."""

    def __init__(self, config: StateResamplerConfig) -> None:
        super().__init__()
        self.config = config
        self.q_state = nn.Parameter(torch.empty(config.num_queries, config.hidden_dim))
        self.empty_record_embedding = nn.Parameter(torch.empty(config.hidden_dim))
        self.layers = nn.ModuleList(_StateResamplerLayer(config) for _ in range(config.num_layers))
        self.p_state = nn.Linear(
            config.hidden_dim,
            config.output_dim,
            bias=config.output_projection_bias,
        )
        nn.init.normal_(self.q_state, std=config.hidden_dim**-0.5)
        nn.init.normal_(self.empty_record_embedding, std=config.hidden_dim**-0.5)

    def forward(self, q_target: Tensor, retrieval: RetrieverOutput) -> StateResamplerOutput:
        batch_size = int(q_target.shape[0])
        packed_records, internal_mask = self._pack_selected_records(retrieval)
        queries = self.q_state.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + q_target.to(dtype=self.q_state.dtype).unsqueeze(1)
        for layer in self.layers:
            queries = layer(queries, packed_records, internal_mask)
        state_tokens = self.p_state(queries).to(dtype=queries.dtype)
        valid_rows = torch.tensor(
            [status in (RetrievalStatus.OK, RetrievalStatus.EMPTY) for status in retrieval.status],
            dtype=torch.bool,
            device=queries.device,
        )
        valid_scale = valid_rows[:, None, None].to(dtype=queries.dtype)
        queries = queries * valid_scale
        state_tokens = state_tokens * valid_scale
        return StateResamplerOutput(
            hidden_states=queries,
            state_tokens=state_tokens,
            selected_record_ids=retrieval.selected_record_ids,
            retrieval_status=retrieval.status,
            state_token_valid_mask=valid_rows,
        )

    def _pack_selected_records(
        self,
        retrieval: RetrieverOutput,
    ) -> tuple[Tensor, Tensor]:
        row_counts = tuple(len(record_ids) for record_ids in retrieval.selected_record_ids)
        max_n_retrieved = max(row_counts, default=0)
        internal_width = max(max_n_retrieved, 1)
        rows: list[Tensor] = []
        internal_masks: list[Tensor] = []
        for row, selected_ids in enumerate(retrieval.selected_record_ids):
            if selected_ids:
                candidate_ids = retrieval.candidate_record_ids[row]
                if all(record_id is None for record_id in candidate_ids):
                    targets = torch.tensor(
                        tuple(_retrieval_sequence_from_id(record_id) for record_id in selected_ids),
                        dtype=torch.int64,
                        device=retrieval.state_embeddings.device,
                    )
                    assert retrieval.candidate_sequence_ids is not None
                    matches = (
                        targets.unsqueeze(1) == retrieval.candidate_sequence_ids[row].unsqueeze(0)
                    ) & retrieval.present_mask[row].unsqueeze(0)
                    columns = matches.to(torch.int64).argmax(dim=1)
                else:
                    id_to_column = {
                        record_id: column
                        for column, record_id in enumerate(candidate_ids)
                        if record_id is not None
                    }
                    column_values = [id_to_column[record_id] for record_id in selected_ids]
                    columns = torch.tensor(
                        column_values,
                        dtype=torch.int64,
                        device=retrieval.state_embeddings.device,
                    )
                selected = retrieval.state_embeddings[row].index_select(0, columns)
                selected = selected.to(dtype=self.q_state.dtype)
                padding = selected.new_zeros(
                    (internal_width - len(selected_ids), self.config.hidden_dim)
                )
                rows.append(torch.cat((selected, padding), dim=0))
                internal_masks.append(
                    torch.arange(internal_width, device=selected.device) < len(selected_ids)
                )
            else:
                padding = self.empty_record_embedding.new_zeros(
                    (internal_width - 1, self.config.hidden_dim)
                )
                rows.append(torch.cat((self.empty_record_embedding.unsqueeze(0), padding), dim=0))
                row_mask = torch.zeros(
                    internal_width,
                    dtype=torch.bool,
                    device=self.empty_record_embedding.device,
                )
                row_mask[0] = True
                internal_masks.append(row_mask)
        return torch.stack(rows), torch.stack(internal_masks)


def _retrieval_sequence_from_id(record_id: str) -> int:
    return int(record_id.removeprefix("retrieval-"))


class _ReaderStateError(ValueError):
    """A fail-closed typed-state condition that maps to ReaderStatus.INVALID."""


class DeterministicStateReader:
    """Read exact integers from the complete, uncompressed selected typed records."""

    def __init__(self, tokenizer: NumberTokenizerProtocol) -> None:
        self.tokenizer = tokenizer

    def read(
        self,
        retrieval: RetrieverOutput,
        hard_operators: Sequence[Operator] | None = None,
        time_resolutions: Sequence[TimeResolution] | None = None,
    ) -> tuple[ReaderResult, ...]:
        batch_size = len(retrieval.status)
        operators = retrieval.hard_operators if hard_operators is None else tuple(hard_operators)
        resolutions = (
            retrieval.time_resolutions if time_resolutions is None else tuple(time_resolutions)
        )
        return tuple(
            self._read_row(row, retrieval, operators[row], resolutions[row])
            for row in range(batch_size)
        )

    def read_bank(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> tuple[ReaderResult, ...]:
        """Read the post-write aggregate/Confirmed Bank without semantic retrieval."""

        snapshot = _reader_bank_snapshot(
            state_bank,
            states,
            query,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )
        return self.read(snapshot)

    def __call__(
        self,
        retrieval: RetrieverOutput,
        hard_operators: Sequence[Operator] | None = None,
        time_resolutions: Sequence[TimeResolution] | None = None,
    ) -> tuple[ReaderResult, ...]:
        return self.read(retrieval, hard_operators, time_resolutions)

    def audit_number_tokens(self, result: ReaderResult) -> int | None:
        """Return the authoritative exact_count carried by one immutable result."""

        return result.exact_count

    def _read_row(
        self,
        row: int,
        retrieval: RetrieverOutput,
        operator: Operator,
        resolution: TimeResolution,
    ) -> ReaderResult:
        status = retrieval.status[row]
        selected_ids = retrieval.selected_record_ids[row]
        records = retrieval.selected_records[row]
        common_audit: tuple[tuple[str, AuditValue], ...] = (
            ("operator", operator.value),
            ("retrieval_status", status.value),
            ("retrieval_reason", retrieval.reason[row].value),
            ("n_state", int(retrieval.n_state[row].item())),
            ("n_retrieved", int(retrieval.n_retrieved[row].item())),
        )
        if status is RetrievalStatus.UNSUPPORTED:
            return self._no_count_result(
                ReaderStatus.UNSUPPORTED,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "retriever_unsupported"),),
            )
        if status is RetrievalStatus.INVALID:
            return self._no_count_result(
                ReaderStatus.INVALID,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "retriever_invalid"),),
            )
        if status is RetrievalStatus.EMPTY:
            if (
                resolution.status is not TimeResolutionStatus.OK
                or not resolution.window.valid
                or operator is Operator.UNSUPPORTED
            ):
                return self._no_count_result(
                    ReaderStatus.INVALID,
                    operator,
                    resolution.window,
                    selected_ids,
                    common_audit + (("reader_reason", "inconsistent_empty_query_metadata"),),
                )
            return self._count_result(
                ReaderStatus.EMPTY,
                0,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "reliable_empty_retrieval"),),
            )
        if status is not RetrievalStatus.OK:
            raise ValueError("RetrieverOutput contains an unknown status")
        if resolution.status is not TimeResolutionStatus.OK or not resolution.window.valid:
            return self._no_count_result(
                ReaderStatus.INVALID,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "invalid_time_resolution_for_ok_retrieval"),),
            )
        if operator is Operator.UNSUPPORTED:
            return self._no_count_result(
                ReaderStatus.UNSUPPORTED,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "unsupported_operator"),),
            )
        if any(not isinstance(record, StateRecord) for record in records):
            return self._no_count_result(
                ReaderStatus.INVALID,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", "retrieval_history_reached_reader"),),
            )
        typed_records = cast(tuple[StateRecord, ...], records)
        try:
            exact_count = _read_exact_count(operator, resolution.window, typed_records)
        except _ReaderStateError as error:
            return self._no_count_result(
                ReaderStatus.INVALID,
                operator,
                resolution.window,
                selected_ids,
                common_audit + (("reader_reason", str(error)),),
            )
        return self._count_result(
            ReaderStatus.OK,
            exact_count,
            operator,
            resolution.window,
            selected_ids,
            common_audit + (("reader_reason", "exact_typed_payload_arithmetic"),),
        )

    def _count_result(
        self,
        status: ReaderStatus,
        exact_count: int,
        operator: Operator,
        window: TimeWindow,
        selected_ids: tuple[str, ...],
        audit: tuple[tuple[str, AuditValue], ...],
    ) -> ReaderResult:
        token_ids = serialize_number_token_ids(self.tokenizer, exact_count)
        return ReaderResult(
            status=status,
            exact_count=exact_count,
            number_token_ids=token_ids,
            selected_record_ids=selected_ids,
            operator=operator,
            time_window=window,
            audit_fields=audit + (("number_text", str(exact_count)),),
        )

    @staticmethod
    def _no_count_result(
        status: ReaderStatus,
        operator: Operator,
        window: TimeWindow,
        selected_ids: tuple[str, ...],
        audit: tuple[tuple[str, AuditValue], ...],
    ) -> ReaderResult:
        return ReaderResult(
            status=status,
            exact_count=None,
            number_token_ids=(),
            selected_record_ids=selected_ids,
            operator=operator,
            time_window=window,
            audit_fields=audit,
        )


def _reader_bank_snapshot(
    state_bank: StructuredStateBank,
    states: Sequence[StateBankRuntimeState],
    query: QueryEncoderOutput,
    *,
    video_ids: Sequence[str],
    trajectory_ids: Sequence[str],
) -> RetrieverOutput:
    """Build a typed Reader-only snapshot with no cosine or time prefiltering."""

    batch_size = len(query.hard_operators)
    normalized_states = tuple(states)
    normalized_video_ids = tuple(video_ids)
    normalized_trajectory_ids = tuple(trajectory_ids)
    heads = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in query.hard_operators)
    view = state_bank.view(normalized_states, None)
    scores = torch.zeros(
        view.present_mask.shape,
        dtype=torch.float32,
        device=view.embeddings.device,
    )
    selected_mask = torch.zeros_like(view.present_mask)
    predicted_head_mask = torch.zeros_like(view.present_mask)
    for row, expected_head in enumerate(heads):
        if expected_head is None:
            continue
        for column, head in enumerate(view.head_types[row]):
            if head is expected_head:
                predicted_head_mask[row, column] = True
    statuses: list[RetrievalStatus] = []
    reasons: list[RetrievalReason] = []
    audits: list[RetrievalFilterAudit] = []
    selected_ids: list[tuple[str, ...]] = []
    selected_scores: list[tuple[float, ...]] = []
    selected_records: list[tuple[StateRecord, ...]] = []
    n_retrieved = torch.zeros(batch_size, dtype=torch.int64, device=view.embeddings.device)

    for row, (operator, resolution) in enumerate(
        zip(query.hard_operators, query.time.resolutions, strict=True)
    ):
        n_state = int(view.n_state[row].item())
        owner_matches = (
            view.video_ids[row] == normalized_video_ids[row]
            and view.trajectory_ids[row] == normalized_trajectory_ids[row]
        )
        rejected_status: RetrievalStatus | None = None
        rejected_reason: RetrievalReason | None = None
        query_rejected = owner_mismatch = 0
        if not owner_matches:
            rejected_status = RetrievalStatus.INVALID
            rejected_reason = RetrievalReason.OWNER_MISMATCH
            owner_mismatch = n_state
        elif resolution.status is TimeResolutionStatus.INVALID:
            rejected_status = RetrievalStatus.INVALID
            rejected_reason = RetrievalReason.INVALID_TIME
            query_rejected = n_state
        elif resolution.status is TimeResolutionStatus.UNSUPPORTED:
            rejected_status = RetrievalStatus.UNSUPPORTED
            rejected_reason = RetrievalReason.UNSUPPORTED_TIME
            query_rejected = n_state
        elif operator is Operator.UNSUPPORTED or heads[row] is None:
            rejected_status = RetrievalStatus.UNSUPPORTED
            rejected_reason = RetrievalReason.UNSUPPORTED_OPERATOR
            query_rejected = n_state

        if rejected_status is not None and rejected_reason is not None:
            statuses.append(rejected_status)
            reasons.append(rejected_reason)
            audits.append(
                RetrievalFilterAudit(
                    n_state=n_state,
                    head_partition_excluded_count=0,
                    query_rejected_count=query_rejected,
                    owner_mismatch_count=owner_mismatch,
                    invalid_count=0,
                    retrieval_ineligible_count=0,
                    future_count=0,
                    outside_window_count=0,
                    below_similarity_count=0,
                    selected_count=0,
                )
            )
            selected_ids.append(())
            selected_scores.append(())
            selected_records.append(())
            continue

        head_excluded = invalid = ineligible = 0
        selected_columns: list[int] = []
        for column in range(n_state):
            record = view.cloned_records[row][column]
            if not bool(predicted_head_mask[row, column]):
                head_excluded += 1
            elif not record.valid:
                invalid += 1
            elif not bool(view.retrieval_eligible_mask[row, column]):
                ineligible += 1
            else:
                selected_columns.append(column)
                selected_mask[row, column] = True
        selected_columns.sort(key=lambda column: str(view.record_ids[row][column]))
        row_ids = tuple(str(view.record_ids[row][column]) for column in selected_columns)
        row_records = tuple(
            clone_state_record(view.cloned_records[row][column])
            for column in selected_columns
            if view.cloned_records[row][column] is not None
        )
        selected_count = len(selected_columns)
        n_retrieved[row] = selected_count
        selected_ids.append(row_ids)
        selected_scores.append((0.0,) * selected_count)
        selected_records.append(row_records)
        if selected_count:
            statuses.append(RetrievalStatus.OK)
            reasons.append(RetrievalReason.MATCHED)
        else:
            statuses.append(RetrievalStatus.EMPTY)
            if n_state == 0:
                reason = RetrievalReason.EMPTY_BANK
            elif head_excluded == n_state:
                reason = RetrievalReason.EMPTY_HEAD_PARTITION
            elif invalid == n_state - head_excluded:
                reason = RetrievalReason.ALL_INVALID
            elif invalid + ineligible == n_state - head_excluded and ineligible:
                reason = RetrievalReason.ALL_RETRIEVAL_INELIGIBLE
            else:
                reason = RetrievalReason.NO_MATCH
            reasons.append(reason)
        audits.append(
            RetrievalFilterAudit(
                n_state=n_state,
                head_partition_excluded_count=head_excluded,
                query_rejected_count=0,
                owner_mismatch_count=0,
                invalid_count=invalid,
                retrieval_ineligible_count=ineligible,
                future_count=0,
                outside_window_count=0,
                below_similarity_count=0,
                selected_count=selected_count,
            )
        )

    return RetrieverOutput(
        selected_record_ids=tuple(selected_ids),
        selected_scores=tuple(selected_scores),
        selected_records=tuple(selected_records),
        candidate_record_ids=view.record_ids,
        candidate_records=view.cloned_records,
        candidate_head_types=view.head_types,
        state_embeddings=view.embeddings,
        scores=scores,
        present_mask=view.present_mask,
        record_valid_mask=view.record_valid_mask,
        retrieval_eligible_mask=view.retrieval_eligible_mask,
        causal_mask=view.present_mask.clone(),
        predicted_head_mask=predicted_head_mask,
        selected_mask=selected_mask,
        status=tuple(statuses),
        reason=tuple(reasons),
        hard_operators=query.hard_operators,
        time_resolutions=query.time.resolutions,
        n_state=view.n_state,
        n_retrieved=n_retrieved,
        audit=tuple(audits),
        video_ids=normalized_video_ids,
        trajectory_ids=normalized_trajectory_ids,
        bank_video_ids=view.video_ids,
        bank_trajectory_ids=view.trajectory_ids,
        bank_versions=view.bank_versions,
    )


def serialize_number_token_ids(
    tokenizer: NumberTokenizerProtocol,
    exact_count: int,
) -> tuple[int, ...]:
    """Encode one signed integer into canonical number token IDs."""

    return tuple(tokenizer.encode(str(exact_count), add_special_tokens=False))


def build_state_resampler(config: ProjectConfig) -> StateResampler:
    return StateResampler(config.state_resampler)


def build_state_reader(
    config: ProjectConfig,
    tokenizer: NumberTokenizerProtocol,
) -> DeterministicStateReader:
    return DeterministicStateReader(tokenizer)


def _read_exact_count(
    operator: Operator,
    window: TimeWindow,
    records: tuple[StateRecord, ...],
) -> int:
    if not records:
        raise _ReaderStateError("ok_retrieval_without_records")

    if operator in (Operator.O1_SNAP, Operator.O1_DELTA):
        o1_payload = cast(O1Payload, records[0].payload)
        if operator is Operator.O1_SNAP:
            return o1_payload.current_visible_count
        return o1_payload.current_visible_count - o1_payload.baseline_count

    if operator in (Operator.O2_UNIQUE, Operator.O2_GAIN):
        confirmed = [
            record.payload for record in records if isinstance(record.payload, ConfirmedIdentity)
        ]
        return len(confirmed)

    if operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
        e1_payload = cast(E1Payload, records[0].payload)
        if window.mode is TimeWindowMode.HISTORY:
            return e1_payload.event_count
        start = window.end_time if window.start_time is None else window.start_time
        return sum(
            start <= event_time <= window.end_time for event_time in e1_payload.recent_event_times
        )

    if operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
        e2_payload = cast(E2Payload, records[0].payload)
        start = window.end_time if window.start_time is None else window.start_time
        return sum(
            start <= interval_end <= window.end_time
            for _, interval_end in e2_payload.completed_intervals
        )

    raise _ReaderStateError("unsupported_operator")
