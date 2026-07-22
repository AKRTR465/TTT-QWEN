"""Resample retrieved state semantics and deterministically read exact integers.

Inputs: ``q_target``, the complete :class:`RetrieverOutput`, effective hard operators,
resolved time windows, and the pinned Qwen tokenizer used by the eventual composer.
Outputs: 16 learned State Tokens plus immutable, tokenizer-audited exact-count results.
Forbidden: Top-K truncation, neural count regression, ground-truth substitution, retrieval,
Bank mutation, or natural-language generation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, StateReaderConfig, StateResamplerConfig
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_EVENT_KIND,
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E1Payload,
    E2EventKind,
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

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class StateResamplerOutput:
    """Fixed-width State Tokens and final-layer selected-record attention audit."""

    hidden_states: Tensor
    state_tokens: Tensor
    cross_attention_weights: Tensor
    record_mask: Tensor
    selected_record_ids: tuple[tuple[str, ...], ...]
    selected_attention_mass: Tensor
    retrieval_status: tuple[RetrievalStatus, ...]
    state_token_valid_mask: Tensor

    def __post_init__(self) -> None:
        hidden = self.hidden_states
        if hidden.ndim != 3 or hidden.shape[1:] != (16, 512):
            raise ValueError("hidden_states must be [B, 16, 512]")
        if not torch.is_floating_point(hidden):
            raise TypeError("hidden_states must be floating")
        batch_size = hidden.shape[0]
        if self.state_tokens.shape != (batch_size, 16, 4096):
            raise ValueError("state_tokens must be [B, 16, 4096]")
        if self.state_tokens.dtype != hidden.dtype or self.state_tokens.device != hidden.device:
            raise ValueError("State Resampler hidden/output tensors must share dtype/device")
        if self.record_mask.ndim != 2 or self.record_mask.shape[0] != batch_size:
            raise ValueError("record_mask must be bool [B, max_N_ret]")
        if self.record_mask.dtype is not torch.bool or self.record_mask.device != hidden.device:
            raise ValueError("record_mask must be bool on the State Resampler device")
        max_records = self.record_mask.shape[1]
        if self.cross_attention_weights.shape != (batch_size, 16, max_records):
            raise ValueError("cross_attention_weights must be [B, 16, max_N_ret]")
        if (
            self.cross_attention_weights.dtype is not torch.float32
            or self.cross_attention_weights.device != hidden.device
        ):
            raise ValueError("cross_attention_weights must be FP32 on the output device")
        if self.selected_attention_mass.shape != (batch_size, 16):
            raise ValueError("selected_attention_mass must be [B, 16]")
        if (
            self.selected_attention_mass.dtype is not torch.float32
            or self.selected_attention_mass.device != hidden.device
        ):
            raise ValueError("selected_attention_mass must be FP32 on the output device")
        if len(self.selected_record_ids) != batch_size:
            raise ValueError("selected_record_ids must contain one tuple per batch row")
        if len(self.retrieval_status) != batch_size or any(
            not isinstance(status, RetrievalStatus) for status in self.retrieval_status
        ):
            raise ValueError("retrieval_status must contain one RetrievalStatus per batch row")
        if (
            self.state_token_valid_mask.shape != (batch_size,)
            or self.state_token_valid_mask.dtype is not torch.bool
            or self.state_token_valid_mask.device != hidden.device
        ):
            raise ValueError("state_token_valid_mask must be bool [B] on the output device")
        expected_valid = torch.tensor(
            [
                status in (RetrievalStatus.OK, RetrievalStatus.EMPTY)
                for status in self.retrieval_status
            ],
            dtype=torch.bool,
            device=hidden.device,
        )
        if hidden.device.type != "meta" and not torch.equal(
            self.state_token_valid_mask, expected_valid
        ):
            raise ValueError("only OK/EMPTY retrieval rows may expose valid State Tokens")
        for row, record_ids in enumerate(self.selected_record_ids):
            if len(record_ids) != int(self.record_mask[row].sum().item()):
                raise ValueError("selected_record_ids must align to the packed record mask")
            if len(set(record_ids)) != len(record_ids) or any(not value for value in record_ids):
                raise ValueError("selected_record_ids must be unique and non-empty per row")
        tensors = (
            hidden,
            self.state_tokens,
            self.cross_attention_weights,
            self.selected_attention_mass,
        )
        if hidden.device.type != "meta" and not all(
            bool(torch.isfinite(tensor).all()) for tensor in tensors
        ):
            raise ValueError("State Resampler outputs must be finite")
        invalid_rows = ~self.state_token_valid_mask
        if (
            hidden.device.type != "meta"
            and bool(invalid_rows.any())
            and (
                bool(torch.any(hidden[invalid_rows] != 0.0))
                or bool(torch.any(self.state_tokens[invalid_rows] != 0.0))
            )
        ):
            raise ValueError("unsupported/invalid rows must expose zero State Tokens")
        if hidden.device.type != "meta" and (
            bool(torch.any(self.cross_attention_weights < 0.0))
            or bool(torch.any(self.cross_attention_weights > 1.0))
        ):
            raise ValueError("cross-attention weights must stay within [0, 1]")
        if max_records:
            invalid_weights = self.cross_attention_weights.masked_select(
                ~self.record_mask[:, None, :].expand_as(self.cross_attention_weights)
            )
            if invalid_weights.numel() and bool(torch.any(invalid_weights != 0.0)):
                raise ValueError("masked records must receive exactly zero attention")
        expected_mass = self.record_mask.any(dim=1).to(torch.float32)[:, None].expand(-1, 16)
        if hidden.device.type != "meta" and not torch.allclose(
            self.selected_attention_mass,
            expected_mass,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("selected attention mass must be one for hits and zero for empty rows")

@dataclass(frozen=True, slots=True)
class ReaderResult:
    status: ReaderStatus
    exact_count: int | None
    number_token_ids: tuple[int, ...]
    selected_record_ids: tuple[str, ...]
    operator: Operator
    time_window: TimeWindow
    audit_fields: tuple[tuple[str, AuditValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReaderStatus):
            raise TypeError("Reader status must be a ReaderStatus")
        if not isinstance(self.operator, Operator) or not isinstance(self.time_window, TimeWindow):
            raise TypeError("Reader operator/time_window metadata has invalid types")
        if len(set(self.selected_record_ids)) != len(self.selected_record_ids) or any(
            not record_id for record_id in self.selected_record_ids
        ):
            raise ValueError("Reader selected_record_ids must be unique and non-empty")
        if self.status is ReaderStatus.OK:
            if type(self.exact_count) is not int:
                raise ValueError("Reader status ok requires an integer exact_count")
            if self.operator is not Operator.O1_DELTA and self.exact_count < 0:
                raise ValueError("only O1-Delta may return a signed negative exact_count")
        elif self.status is ReaderStatus.EMPTY:
            if self.exact_count != 0 or self.selected_record_ids:
                raise ValueError("Reader empty requires exact_count=0 and no selected records")
        elif self.exact_count is not None:
            raise ValueError("unsupported/invalid Reader results cannot contain an exact_count")
        if self.status in (ReaderStatus.OK, ReaderStatus.EMPTY) and not self.time_window.valid:
            raise ValueError("count-bearing Reader results require a valid TimeWindow")
        if (
            self.status in (ReaderStatus.OK, ReaderStatus.EMPTY)
            and self.operator is Operator.UNSUPPORTED
        ):
            raise ValueError("unsupported operators cannot produce a Reader count")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.number_token_ids):
            raise ValueError("number_token_ids must be non-negative integers")
        if (self.exact_count is None) == bool(self.number_token_ids):
            raise ValueError("number tokens must be present exactly when exact_count is present")
        audit_keys = tuple(key for key, _ in self.audit_fields)
        if any(not key for key in audit_keys) or len(set(audit_keys)) != len(audit_keys):
            raise ValueError("Reader audit keys must be unique and non-empty")
        if any(not _is_audit_value(value) for _, value in self.audit_fields):
            raise TypeError("Reader audit values must be scalar immutable metadata")
        audit = dict(self.audit_fields)
        required = {
            "source",
            "operator",
            "retrieval_status",
            "retrieval_reason",
            "n_state",
            "n_retrieved",
            "input_record_count",
            "bank_version",
            "time_resolution_status",
            "window_start",
            "window_end",
            "reader_reason",
        }
        missing = required.difference(audit)
        if missing:
            raise ValueError(
                f"Reader audit is missing required provenance fields: {sorted(missing)}"
            )
        if audit["source"] != "retrieved_typed_records" or audit["operator"] != self.operator.value:
            raise ValueError("Reader audit source/operator provenance is inconsistent")
        if audit["n_retrieved"] != len(self.selected_record_ids) or audit[
            "input_record_count"
        ] != len(self.selected_record_ids):
            raise ValueError("Reader audit record counts must match selected_record_ids")
        if self.status in (ReaderStatus.OK, ReaderStatus.EMPTY):
            count_fields = {
                "arithmetic",
                "contributing_count",
                "computed_exact_count",
                "number_text",
            }
            if missing_count := count_fields.difference(audit):
                raise ValueError(
                    f"count-bearing Reader audit is missing fields: {sorted(missing_count)}"
                )
            if audit["computed_exact_count"] != self.exact_count or audit["number_text"] != str(
                self.exact_count
            ):
                raise ValueError("Reader audit operands do not reproduce exact_count/number text")
            contributing_count = audit["contributing_count"]
            if type(contributing_count) is not int or contributing_count < 0:
                raise ValueError("Reader contributing_count must be a non-negative integer")
        if self.status is ReaderStatus.OK:
            self._validate_exact_count_operands(audit)

    def _validate_exact_count_operands(self, audit: dict[str, AuditValue]) -> None:
        exact_count = cast(int, self.exact_count)
        if self.operator in (Operator.O1_SNAP, Operator.O1_DELTA):
            required = {
                "operand_current_visible_count",
                "operand_baseline_count",
                "operand_baseline_initialized",
                "operand_baseline_position_id",
            }
            _require_audit_keys(audit, required, self.operator)
            current = audit["operand_current_visible_count"]
            baseline = audit["operand_baseline_count"]
            if type(current) is not int or type(baseline) is not int:
                raise ValueError("O1 Reader operands must be integers")
            if self.operator is Operator.O1_SNAP:
                expected = current
            else:
                if audit.get("baseline_policy") != "fixed_baseline_v1":
                    raise ValueError("O1-Delta Reader audit must pin fixed_baseline_v1")
                expected = current - baseline
            if exact_count != expected:
                raise ValueError("O1 Reader operands do not reproduce exact_count")
            return
        if self.operator in (Operator.O2_UNIQUE, Operator.O2_GAIN):
            required = {
                "operand_confirmed_record_count",
                "operand_distinct_identity_count",
                "operand_first_seen_min",
                "operand_first_seen_max",
                "matched_first_seen_count",
            }
            _require_audit_keys(audit, required, self.operator)
            if exact_count != audit["matched_first_seen_count"]:
                raise ValueError("O2 Reader operands do not reproduce exact_count")
            return
        if self.operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
            required = {
                "operand_cumulative_event_count",
                "operand_retained_completion_count",
                "operand_history_eviction_count",
                "matched_completion_count",
            }
            _require_audit_keys(audit, required, self.operator)
            if exact_count != audit["matched_completion_count"]:
                raise ValueError("E1 Reader operands do not reproduce exact_count")
            return
        if self.operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
            required = {
                "operand_completed_interval_count",
                "operand_active_interval_present",
                "matched_completion_end_count",
            }
            _require_audit_keys(audit, required, self.operator)
            if exact_count != audit["matched_completion_end_count"]:
                raise ValueError("E2 Reader operands do not reproduce exact_count")
            return
        raise ValueError("unsupported operators cannot carry OK Reader audit operands")


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
    ) -> tuple[Tensor, Tensor]:
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
        if queries.device.type != "meta" and bool(torch.any(denominator <= 0.0)):
            raise RuntimeError("State Resampler cross-attention requires one valid KV per row")
        cross_weights = cross_weights / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        cross_context = torch.matmul(cross_weights.to(cross_values.dtype), cross_values)
        queries = queries + self.cross_out(self._merge_heads(cross_context))

        feed_forward = self.ffn_in(self.ffn_norm(queries))
        queries = queries + self.ffn_out(F.gelu(feed_forward))
        return queries, cross_weights.mean(dim=1)

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
        _validate_state_resampler_config(config)
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
        batch_size = self._validate_inputs(q_target, retrieval)
        packed_records, internal_mask, external_mask = self._pack_selected_records(retrieval)
        queries = self.q_state.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + q_target.to(dtype=self.q_state.dtype).unsqueeze(1)
        final_weights = torch.empty(
            (batch_size, self.config.num_queries, internal_mask.shape[1]),
            dtype=torch.float32,
            device=q_target.device,
        )
        for layer in self.layers:
            queries, final_weights = layer(queries, packed_records, internal_mask)

        max_n_retrieved = external_mask.shape[1]
        external_weights = final_weights[:, :, :max_n_retrieved]
        empty_rows = ~external_mask.any(dim=1)
        if max_n_retrieved:
            external_weights = torch.where(
                empty_rows[:, None, None],
                torch.zeros_like(external_weights),
                external_weights,
            )
            external_weights = torch.where(
                external_mask[:, None, :],
                external_weights,
                0.0,
            )
        selected_mass = external_weights.sum(dim=-1)
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
            cross_attention_weights=external_weights,
            record_mask=external_mask,
            selected_record_ids=retrieval.selected_record_ids,
            selected_attention_mass=selected_mass,
            retrieval_status=retrieval.status,
            state_token_valid_mask=valid_rows,
        )

    def _validate_inputs(self, q_target: Tensor, retrieval: RetrieverOutput) -> int:
        if (
            q_target.ndim != 2
            or q_target.shape[1] != self.config.hidden_dim
            or not torch.is_floating_point(q_target)
        ):
            raise ValueError("q_target must be floating [B, 512]")
        if not bool(torch.isfinite(q_target).all()):
            raise ValueError("q_target must be finite")
        batch_size = int(q_target.shape[0])
        if batch_size <= 0:
            raise ValueError("q_target batch must be non-empty")
        if retrieval.state_embeddings.shape[0] != batch_size:
            raise ValueError("q_target and RetrieverOutput batch sizes must match")
        if retrieval.state_embeddings.device != q_target.device:
            raise ValueError("q_target and RetrieverOutput must share one device")
        parameter = self.q_state
        if parameter.device != q_target.device:
            raise ValueError("State Resampler parameters and q_target must share one device")
        if len(retrieval.selected_record_ids) != batch_size:
            raise ValueError("RetrieverOutput selected IDs must align to q_target")
        return batch_size

    def _pack_selected_records(
        self,
        retrieval: RetrieverOutput,
    ) -> tuple[Tensor, Tensor, Tensor]:
        row_counts = tuple(len(record_ids) for record_ids in retrieval.selected_record_ids)
        max_n_retrieved = max(row_counts, default=0)
        internal_width = max(max_n_retrieved, 1)
        rows: list[Tensor] = []
        internal_masks: list[Tensor] = []
        external_masks: list[Tensor] = []
        for row, selected_ids in enumerate(retrieval.selected_record_ids):
            id_to_column = {
                record_id: column
                for column, record_id in enumerate(retrieval.candidate_record_ids[row])
                if record_id is not None
            }
            if selected_ids:
                try:
                    columns = [id_to_column[record_id] for record_id in selected_ids]
                except KeyError as error:
                    raise ValueError(
                        "selected record IDs are missing from the candidate axis"
                    ) from error
                indices = torch.tensor(
                    columns,
                    dtype=torch.int64,
                    device=retrieval.state_embeddings.device,
                )
                selected = retrieval.state_embeddings[row].index_select(0, indices)
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
            external_masks.append(
                torch.arange(max_n_retrieved, device=self.q_state.device) < len(selected_ids)
            )
        return (
            torch.stack(rows),
            torch.stack(internal_masks),
            torch.stack(external_masks),
        )


class _ReaderStateError(ValueError):
    """A fail-closed typed-state condition that maps to ReaderStatus.INVALID."""


@dataclass(frozen=True, slots=True)
class _ExactCountComputation:
    exact_count: int
    arithmetic: str
    contributing_count: int
    operands: tuple[tuple[str, AuditValue], ...]


class DeterministicStateReader:
    """Read exact integers from the complete, uncompressed selected typed records."""

    def __init__(self, tokenizer: NumberTokenizerProtocol) -> None:
        if tokenizer is None:
            raise ValueError("Deterministic State Reader requires the pinned tokenizer")
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
        if len(operators) != batch_size or any(
            not isinstance(operator, Operator) for operator in operators
        ):
            raise ValueError("hard_operators must contain one Operator per Retriever row")
        if len(resolutions) != batch_size or any(
            not isinstance(resolution, TimeResolution) for resolution in resolutions
        ):
            raise ValueError("time_resolutions must contain one TimeResolution per Retriever row")
        if operators != retrieval.hard_operators:
            raise ValueError("hard_operators do not match Retriever provenance")
        if resolutions != retrieval.time_resolutions:
            raise ValueError("time_resolutions do not match Retriever provenance")
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

    def audit_results(
        self,
        retrieval: RetrieverOutput,
        results: Sequence[ReaderResult],
    ) -> tuple[ReaderResult, ...]:
        """Recompute from the same Retriever snapshot and reject any caller rewrite."""

        normalized = tuple(results)
        expected = self.read(retrieval)
        if normalized != expected:
            raise ValueError(
                "Reader results do not match authoritative retrieved-record arithmetic"
            )
        return normalized

    def audit_bank_results(
        self,
        state_bank: StructuredStateBank,
        states: Sequence[StateBankRuntimeState],
        query: QueryEncoderOutput,
        results: Sequence[ReaderResult],
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> tuple[ReaderResult, ...]:
        """Recompute direct Bank arithmetic and reject any caller rewrite."""

        normalized = tuple(results)
        expected = self.read_bank(
            state_bank,
            states,
            query,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )
        if normalized != expected:
            raise ValueError("Reader results do not match authoritative Bank arithmetic")
        return normalized

    def audit_number_tokens(self, result: ReaderResult) -> int | None:
        """Re-decode one immutable result and reject any caller-substituted token IDs."""

        if not isinstance(result, ReaderResult):
            raise TypeError("number-token audit requires a ReaderResult")
        if result.exact_count is None:
            if result.number_token_ids:
                raise ValueError("a no-count Reader result cannot contain number tokens")
            return None
        decoded = decode_number_token_ids(self.tokenizer, result.number_token_ids)
        canonical_ids = serialize_number_token_ids(self.tokenizer, result.exact_count)
        if decoded != result.exact_count or result.number_token_ids != canonical_ids:
            raise ValueError("Reader number tokens do not match the authoritative exact_count")
        return decoded

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
            ("source", "retrieved_typed_records"),
            ("operator", operator.value),
            ("retrieval_status", status.value),
            ("retrieval_reason", retrieval.reason[row].value),
            ("n_state", int(retrieval.n_state[row].item())),
            ("n_retrieved", int(retrieval.n_retrieved[row].item())),
            ("input_record_count", len(records)),
            ("bank_version", retrieval.bank_versions[row]),
            ("time_resolution_status", resolution.status.value),
            ("window_start", resolution.window.start_time),
            ("window_end", resolution.window.end_time),
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
                common_audit
                + (
                    ("reader_reason", "reliable_empty_retrieval"),
                    ("arithmetic", "empty_set"),
                    ("contributing_count", 0),
                    ("computed_exact_count", 0),
                ),
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
            computation = _read_exact_count(
                operator,
                resolution.window,
                typed_records,
            )
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
            computation.exact_count,
            operator,
            resolution.window,
            selected_ids,
            common_audit
            + (
                ("reader_reason", "exact_typed_payload_arithmetic"),
                ("arithmetic", computation.arithmetic),
                ("contributing_count", computation.contributing_count),
            )
            + computation.operands
            + (("computed_exact_count", computation.exact_count),),
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
        decoded = decode_number_token_ids(self.tokenizer, token_ids)
        if decoded != exact_count:
            raise ValueError("number token IDs failed exact integer roundtrip")
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
    if (
        len(normalized_states) != batch_size
        or len(normalized_video_ids) != batch_size
        or len(normalized_trajectory_ids) != batch_size
    ):
        raise ValueError("Reader Bank inputs must align to the Query batch")
    heads = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in query.hard_operators)
    view = state_bank.view(normalized_states, heads)
    scores = torch.zeros(
        view.present_mask.shape,
        dtype=torch.float32,
        device=view.embeddings.device,
    )
    selected_mask = torch.zeros_like(view.present_mask)
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
        owner_count = int(view.owner_record_counts[row].item())
        head_excluded = owner_count - n_state
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
                    head_partition_excluded_count=head_excluded,
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

        invalid = ineligible = 0
        selected_columns: list[int] = []
        for column in range(n_state):
            record = view.cloned_records[row][column]
            if record is None:
                raise ValueError("present Reader Bank columns require typed records")
            if not record.valid:
                invalid += 1
            elif not bool(view.retrieval_eligible_mask[row, column]):
                ineligible += 1
            else:
                selected_columns.append(column)
                selected_mask[row, column] = True
        selected_columns.sort(
            key=lambda column: str(view.record_ids[row][column])
        )
        row_ids = tuple(
            str(view.record_ids[row][column]) for column in selected_columns
        )
        row_records = tuple(
            clone_state_record(view.cloned_records[row][column])
            for column in selected_columns
            if view.cloned_records[row][column] is not None
        )
        if len(row_records) != len(selected_columns):
            raise ValueError("Reader Bank record metadata lost alignment")
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
                reason = (
                    RetrievalReason.EMPTY_BANK
                    if owner_count == 0
                    else RetrievalReason.EMPTY_HEAD_PARTITION
                )
            elif invalid == n_state:
                reason = RetrievalReason.ALL_INVALID
            elif invalid + ineligible == n_state and ineligible:
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
        state_embeddings=view.embeddings,
        scores=scores,
        present_mask=view.present_mask,
        record_valid_mask=view.record_valid_mask,
        retrieval_eligible_mask=view.retrieval_eligible_mask,
        causal_mask=view.present_mask.clone(),
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


_CANONICAL_SIGNED_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


def serialize_number_token_ids(
    tokenizer: NumberTokenizerProtocol,
    exact_count: int,
) -> tuple[int, ...]:
    """Encode one signed integer without special tokens and prove a canonical roundtrip."""

    if type(exact_count) is not int:
        raise TypeError("exact_count must be an integer")
    canonical = str(exact_count)
    raw_ids = tokenizer.encode(canonical, add_special_tokens=False)
    token_ids = tuple(raw_ids)
    if not token_ids or any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer must return non-empty non-negative integer IDs")
    decoded = _decode_number_text(tokenizer, token_ids)
    if decoded != canonical:
        raise ValueError("number tokenizer must preserve the canonical signed integer text")
    reencoded = tuple(tokenizer.encode(decoded, add_special_tokens=False))
    if reencoded != token_ids:
        raise ValueError("number tokenizer must re-encode canonical text to identical token IDs")
    return token_ids


def decode_number_token_ids(
    tokenizer: NumberTokenizerProtocol,
    token_ids: Sequence[int],
) -> int:
    """Decode IDs, reject non-canonical text, and return the audited signed integer."""

    normalized = tuple(token_ids)
    if not normalized or any(type(token_id) is not int or token_id < 0 for token_id in normalized):
        raise ValueError("number token IDs must be non-empty non-negative integers")
    text = _decode_number_text(tokenizer, normalized)
    if _CANONICAL_SIGNED_INTEGER.fullmatch(text) is None:
        raise ValueError("decoded number text is not a canonical signed integer")
    value = int(text)
    if str(value) != text:
        raise ValueError("decoded number text is not canonical")
    return value


def build_state_resampler(config: ProjectConfig | None = None) -> StateResampler:
    if config is None:
        raise ValueError("build_state_resampler requires a validated ProjectConfig")
    return StateResampler(config.state_resampler)


def build_state_reader(
    config: ProjectConfig | None = None,
    tokenizer: NumberTokenizerProtocol | None = None,
) -> DeterministicStateReader:
    if config is None:
        raise ValueError("build_state_reader requires a validated ProjectConfig")
    if tokenizer is None:
        raise ValueError("build_state_reader requires the pinned tokenizer")
    _validate_state_reader_config(config.state_reader)
    _validate_pinned_tokenizer(config, tokenizer)
    return DeterministicStateReader(tokenizer)


def _validate_state_resampler_config(config: StateResamplerConfig) -> None:
    expected: tuple[tuple[str, object, object], ...] = (
        ("num_queries", config.num_queries, 16),
        ("num_layers", config.num_layers, 3),
        ("num_heads", config.num_heads, 8),
        ("head_dim", config.head_dim, 64),
        ("ffn_dim", config.ffn_dim, 2048),
        ("hidden_dim", config.hidden_dim, 512),
        ("output_dim", config.output_dim, 4096),
        ("layer_norm_eps", config.layer_norm_eps, 1.0e-5),
        ("activation", config.activation, "gelu"),
        ("dropout", config.dropout, 0.0),
        ("attention_bias", config.attention_bias, True),
        ("output_projection_bias", config.output_projection_bias, True),
        ("attention_softmax_dtype", config.attention_softmax_dtype, "float32"),
        ("empty_record_embedding", config.empty_record_embedding, True),
        (
            "empty_record_policy",
            config.empty_record_policy,
            "internal_trainable_kv_external_zero_width",
        ),
        (
            "attention_audit",
            config.attention_audit,
            "final_layer_mean_heads_selected_mass",
        ),
        ("parameter_count", config.parameter_count, 14_722_048),
    )
    for name, actual, frozen in expected:
        if actual != frozen:
            raise ValueError(f"State Resampler {name} must equal {frozen!r}; got {actual!r}")
    if config.num_heads * config.head_dim != config.hidden_dim:
        raise ValueError("State Resampler heads must exactly partition hidden_dim")


def _validate_state_reader_config(config: StateReaderConfig) -> None:
    expected: tuple[tuple[str, object, object], ...] = (
        ("signed_exact_count", config.signed_exact_count, True),
        ("empty_exact_count", config.empty_exact_count, 0),
        ("status_propagation", config.status_propagation, "retriever_exact"),
        ("o1_delta_policy", config.o1_delta_policy, "fixed_baseline_v1"),
        ("o2_identity_key", config.o2_identity_key, "identity_id"),
        ("point_window_boundary", config.point_window_boundary, "closed"),
        (
            "e1_history_policy",
            config.e1_history_policy,
            "cumulative_or_retained_completion_times",
        ),
        ("e1_truncated_window_status", config.e1_truncated_window_status, "invalid"),
        ("e2_window_anchor", config.e2_window_anchor, "completion_end"),
        ("event_kind_mismatch_status", config.event_kind_mismatch_status, "invalid"),
        ("number_text_format", config.number_text_format, "canonical_ascii_signed_decimal"),
        ("tokenizer_add_special_tokens", config.tokenizer_add_special_tokens, False),
        ("tokenizer_roundtrip_required", config.tokenizer_roundtrip_required, True),
        ("tokenizer_class", config.tokenizer_class, "Qwen2TokenizerFast"),
        ("tokenizer_vocab_size", config.tokenizer_vocab_size, 151_643),
        (
            "tokenizer_required_files",
            config.tokenizer_required_files,
            ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
        ),
        (
            "tokenizer_manifest_sha256",
            config.tokenizer_manifest_sha256,
            "ccd18347b6d6714d91d4c55b37ff05e473a0f8e84fbcba2bda1401a9572f44c3",
        ),
        ("ground_truth_input_forbidden", config.ground_truth_input_forbidden, True),
    )
    for name, actual, frozen in expected:
        if actual != frozen:
            raise ValueError(f"State Reader {name} must equal {frozen!r}; got {actual!r}")


def _validate_pinned_tokenizer(
    config: ProjectConfig,
    tokenizer: NumberTokenizerProtocol,
) -> None:
    reader_config = config.state_reader
    if type(tokenizer).__name__ != reader_config.tokenizer_class:
        raise ValueError("pinned tokenizer class does not match the frozen Reader contract")
    if tokenizer.vocab_size != reader_config.tokenizer_vocab_size:
        raise ValueError("pinned tokenizer vocabulary size does not match the frozen contract")
    source = getattr(tokenizer, "name_or_path", None)
    if not isinstance(source, str) or not source:
        raise ValueError("pinned tokenizer must expose its local snapshot path")
    snapshot = Path(source).resolve()
    if not snapshot.is_dir():
        raise ValueError("pinned tokenizer source must be an existing local snapshot directory")
    if re.fullmatch(r"[0-9a-f]{40}", snapshot.name) and snapshot.name != config.model.revision:
        raise ValueError("pinned tokenizer snapshot revision does not match model.revision")
    expected_model_cache_name = "models--" + config.model.base_model.replace("/", "--")
    cache_names = tuple(part for part in snapshot.parts if part.startswith("models--"))
    if cache_names and expected_model_cache_name not in cache_names:
        raise ValueError("pinned tokenizer snapshot model does not match model.base_model")
    actual_manifest = _tokenizer_manifest_sha256(
        str(snapshot),
        reader_config.tokenizer_required_files,
    )
    if actual_manifest != reader_config.tokenizer_manifest_sha256:
        raise ValueError("pinned tokenizer file manifest does not match the frozen SHA256")


def _tokenizer_manifest_sha256(snapshot_text: str, required_files: tuple[str, ...]) -> str:
    snapshot = Path(snapshot_text)
    digest = sha256()
    for filename in required_files:
        path = snapshot / filename
        if not path.is_file():
            raise ValueError(f"pinned tokenizer snapshot is missing {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_exact_count(
    operator: Operator,
    window: TimeWindow,
    records: tuple[StateRecord, ...],
) -> _ExactCountComputation:
    if not records:
        raise _ReaderStateError("ok_retrieval_without_records")
    expected_head = OPERATOR_TO_HEAD_TYPE[operator]
    if expected_head is None or any(record.head_type is not expected_head for record in records):
        raise _ReaderStateError("operator_head_mismatch")
    if any(not record.valid for record in records):
        raise _ReaderStateError("invalid_record_reached_reader")
    if len({record.record_id for record in records}) != len(records):
        raise _ReaderStateError("duplicate_record_id")

    if operator in (Operator.O1_SNAP, Operator.O1_DELTA):
        record = _require_single_record(records, O1Payload, "o1_aggregate")
        o1_payload = cast(O1Payload, record.payload)
        if operator is Operator.O1_SNAP:
            return _ExactCountComputation(
                exact_count=o1_payload.current_visible_count,
                arithmetic="o1_current_visible_count",
                contributing_count=1,
                operands=(
                    ("operand_current_visible_count", o1_payload.current_visible_count),
                    ("operand_baseline_count", o1_payload.baseline_count),
                    ("operand_baseline_initialized", o1_payload.baseline_initialized),
                    ("operand_baseline_position_id", o1_payload.baseline_position_id),
                ),
            )
        if not o1_payload.baseline_initialized:
            raise _ReaderStateError("o1_baseline_uninitialized")
        return _ExactCountComputation(
            exact_count=o1_payload.current_visible_count - o1_payload.baseline_count,
            arithmetic="o1_current_visible_count_minus_fixed_baseline_v1",
            contributing_count=1,
            operands=(
                ("baseline_policy", "fixed_baseline_v1"),
                ("operand_current_visible_count", o1_payload.current_visible_count),
                ("operand_baseline_count", o1_payload.baseline_count),
                ("operand_baseline_initialized", o1_payload.baseline_initialized),
                ("operand_baseline_position_id", o1_payload.baseline_position_id),
            ),
        )

    if operator in (Operator.O2_UNIQUE, Operator.O2_GAIN):
        confirmed: list[ConfirmedIdentity] = []
        for record in records:
            if isinstance(record.payload, CandidateIdentity):
                raise _ReaderStateError("o2_candidate_reached_reader")
            if not isinstance(record.payload, ConfirmedIdentity):
                raise _ReaderStateError("o2_wrong_payload")
            confirmed.append(record.payload)
        identity_ids = tuple(payload.identity_id for payload in confirmed)
        if len(set(identity_ids)) != len(identity_ids):
            raise _ReaderStateError("duplicate_confirmed_identity")
        if operator is Operator.O2_UNIQUE:
            if any(payload.first_seen > window.query_time for payload in confirmed):
                raise _ReaderStateError("o2_future_identity_reached_reader")
            count = len(confirmed)
            return _ExactCountComputation(
                exact_count=count,
                arithmetic="o2_confirmed_first_seen_at_or_before_query",
                contributing_count=count,
                operands=(
                    ("operand_confirmed_record_count", len(confirmed)),
                    ("operand_distinct_identity_count", len(set(identity_ids))),
                    ("operand_first_seen_min", min(item.first_seen for item in confirmed)),
                    ("operand_first_seen_max", max(item.first_seen for item in confirmed)),
                    ("matched_first_seen_count", count),
                ),
            )
        if window.start_time is None:
            raise _ReaderStateError("o2_gain_requires_bounded_window")
        if any(
            not window.start_time <= payload.first_seen <= window.end_time for payload in confirmed
        ):
            raise _ReaderStateError("o2_identity_outside_gain_window")
        count = len(confirmed)
        return _ExactCountComputation(
            exact_count=count,
            arithmetic="o2_confirmed_first_seen_in_closed_window",
            contributing_count=count,
            operands=(
                ("operand_confirmed_record_count", len(confirmed)),
                ("operand_distinct_identity_count", len(set(identity_ids))),
                ("operand_first_seen_min", min(item.first_seen for item in confirmed)),
                ("operand_first_seen_max", max(item.first_seen for item in confirmed)),
                ("matched_first_seen_count", count),
            ),
        )

    if operator in (Operator.E1_ACTION, Operator.E1_TRANSIT):
        record = _require_single_record(records, E1Payload, "e1_aggregate")
        e1_payload = cast(E1Payload, record.payload)
        expected_e1_kind = OPERATOR_TO_EVENT_KIND[operator]
        if not isinstance(expected_e1_kind, E1EventKind):
            raise _ReaderStateError("e1_operator_kind_missing")
        if e1_payload.event_kind is not expected_e1_kind:
            raise _ReaderStateError("e1_event_kind_mismatch")
        if window.mode is TimeWindowMode.HISTORY:
            return _ExactCountComputation(
                exact_count=e1_payload.event_count,
                arithmetic="e1_cumulative_completed_count",
                contributing_count=e1_payload.event_count,
                operands=(
                    ("operand_cumulative_event_count", e1_payload.event_count),
                    ("operand_retained_completion_count", len(e1_payload.recent_event_times)),
                    ("operand_history_eviction_count", e1_payload.history_eviction_count),
                    ("matched_completion_count", e1_payload.event_count),
                ),
            )
        start = window.end_time if window.start_time is None else window.start_time
        if e1_payload.history_truncated and (
            not e1_payload.recent_event_times or start < e1_payload.recent_event_times[0]
        ):
            raise _ReaderStateError("e1_window_history_truncated")
        count = sum(
            start <= event_time <= window.end_time for event_time in e1_payload.recent_event_times
        )
        return _ExactCountComputation(
            exact_count=count,
            arithmetic="e1_completion_time_in_closed_window",
            contributing_count=count,
            operands=(
                ("operand_cumulative_event_count", e1_payload.event_count),
                ("operand_retained_completion_count", len(e1_payload.recent_event_times)),
                ("operand_history_eviction_count", e1_payload.history_eviction_count),
                ("matched_completion_count", count),
            ),
        )

    if operator in (Operator.E2_PERIODIC, Operator.E2_EPISODE):
        record = _require_single_record(records, E2Payload, "e2_aggregate")
        e2_payload = cast(E2Payload, record.payload)
        expected_e2_kind = OPERATOR_TO_EVENT_KIND[operator]
        if not isinstance(expected_e2_kind, E2EventKind):
            raise _ReaderStateError("e2_operator_kind_missing")
        if e2_payload.event_kind is not expected_e2_kind:
            raise _ReaderStateError("e2_event_kind_mismatch")
        start = window.end_time if window.start_time is None else window.start_time
        count = sum(
            start <= interval_end <= window.end_time
            for _, interval_end in e2_payload.completed_intervals
        )
        return _ExactCountComputation(
            exact_count=count,
            arithmetic="e2_interval_end_in_closed_window",
            contributing_count=count,
            operands=(
                ("operand_completed_interval_count", len(e2_payload.completed_intervals)),
                ("operand_active_interval_present", e2_payload.current_start is not None),
                ("matched_completion_end_count", count),
            ),
        )

    raise _ReaderStateError("unsupported_operator")


def _require_single_record(
    records: tuple[StateRecord, ...],
    payload_type: type[O1Payload] | type[E1Payload] | type[E2Payload],
    name: str,
) -> StateRecord:
    if len(records) != 1:
        raise _ReaderStateError(f"{name}_requires_exactly_one_record")
    record = records[0]
    if not isinstance(record.payload, payload_type):
        raise _ReaderStateError(f"{name}_payload_mismatch")
    return record


def _decode_number_text(
    tokenizer: NumberTokenizerProtocol,
    token_ids: Sequence[int],
) -> str:
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise TypeError("number tokenizer decode must return text")
    return decoded


def _is_audit_value(value: object) -> bool:
    return (
        value is None
        or type(value) in (str, int, bool)
        or (type(value) is float and math.isfinite(value))
    )


def _require_audit_keys(
    audit: dict[str, AuditValue],
    required: set[str],
    operator: Operator,
) -> None:
    if missing := required.difference(audit):
        raise ValueError(f"{operator.value} Reader audit is missing operands: {sorted(missing)}")
