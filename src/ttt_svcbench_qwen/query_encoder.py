"""Encode question-only token embeddings and resolve operator/time metadata.

Inputs: complete-question Qwen token embeddings, P2 token provenance, and label-free runtime
metadata. Outputs: target/operator/time embeddings, a gated hard operator, and an explicit
TimeWindow. Forbidden: answer/label tokens, full Qwen decoder execution, keyword task routing,
State Bank access, Reader arithmetic, or guessed time windows.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import (
    OperatorRouterConfig,
    ProjectConfig,
    QueryEncoderConfig,
    TimeResolverConfig,
)
from ttt_svcbench_qwen.data import RuntimeQueryInput, extract_explicit_time_values
from ttt_svcbench_qwen.query_tokens import QuestionTokenBatch
from ttt_svcbench_qwen.state_bank import E1EventKind, E2EventKind, HeadType


class Operator(StrEnum):
    O1_SNAP = "o1-snap"
    O1_DELTA = "o1-delta"
    O2_UNIQUE = "o2-unique"
    O2_GAIN = "o2-gain"
    E1_ACTION = "e1-action"
    E1_TRANSIT = "e1-transit"
    E2_PERIODIC = "e2-periodic"
    E2_EPISODE = "e2-episode"
    UNSUPPORTED = "unsupported"


class TimeWindowMode(StrEnum):
    NOW = "now"
    HISTORY = "history"
    RECENT = "recent"
    EXPLICIT_RANGE = "explicit_range"


class TimeResolutionStatus(StrEnum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


OPERATORS = tuple(Operator)
TIME_MODES = tuple(TimeWindowMode)
UNSUPPORTED_OPERATOR_INDEX = OPERATORS.index(Operator.UNSUPPORTED)

OPERATOR_TO_HEAD_TYPE: Mapping[Operator, HeadType | None] = {
    Operator.O1_SNAP: HeadType.O1,
    Operator.O1_DELTA: HeadType.O1,
    Operator.O2_UNIQUE: HeadType.O2,
    Operator.O2_GAIN: HeadType.O2,
    Operator.E1_ACTION: HeadType.E1,
    Operator.E1_TRANSIT: HeadType.E1,
    Operator.E2_PERIODIC: HeadType.E2,
    Operator.E2_EPISODE: HeadType.E2,
    Operator.UNSUPPORTED: None,
}

OPERATOR_TO_EVENT_KIND: Mapping[Operator, E1EventKind | E2EventKind | None] = {
    Operator.O1_SNAP: None,
    Operator.O1_DELTA: None,
    Operator.O2_UNIQUE: None,
    Operator.O2_GAIN: None,
    Operator.E1_ACTION: E1EventKind.ACTION,
    Operator.E1_TRANSIT: E1EventKind.TRANSIT,
    Operator.E2_PERIODIC: E2EventKind.PERIODIC,
    Operator.E2_EPISODE: E2EventKind.EPISODE,
    Operator.UNSUPPORTED: None,
}

# These defaults follow the operator arithmetic in ARCHITECTURE section 10.3. RECENT has no
# default duration: O1-Delta/O2-Gain without an explicit positive duration remain unsupported.
OPERATOR_DEFAULT_TIME_MODE: Mapping[Operator, TimeWindowMode | None] = {
    Operator.O1_SNAP: TimeWindowMode.NOW,
    Operator.O1_DELTA: TimeWindowMode.RECENT,
    Operator.O2_UNIQUE: TimeWindowMode.HISTORY,
    Operator.O2_GAIN: TimeWindowMode.RECENT,
    Operator.E1_ACTION: TimeWindowMode.HISTORY,
    Operator.E1_TRANSIT: TimeWindowMode.HISTORY,
    Operator.E2_PERIODIC: TimeWindowMode.HISTORY,
    Operator.E2_EPISODE: TimeWindowMode.HISTORY,
    Operator.UNSUPPORTED: None,
}


@dataclass(frozen=True, slots=True)
class TimeWindow:
    mode: TimeWindowMode
    query_time: float
    start_time: float | None
    end_time: float
    valid: bool

    def __post_init__(self) -> None:
        values = (self.query_time, self.end_time)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("TimeWindow times must be finite")
        if self.start_time is not None and not math.isfinite(self.start_time):
            raise ValueError("TimeWindow start_time must be finite when present")
        if self.query_time < 0.0 or self.end_time < 0.0:
            raise ValueError("TimeWindow times must be non-negative")
        if self.end_time > self.query_time:
            raise ValueError("TimeWindow cannot extend beyond query_time")
        if self.start_time is not None and not 0.0 <= self.start_time <= self.end_time:
            raise ValueError("TimeWindow start_time must be within [0, end_time]")
        if not self.valid:
            return
        if self.mode is TimeWindowMode.NOW and (
            self.start_time is not None or self.end_time != self.query_time
        ):
            raise ValueError("a valid now window must end at query_time with no start_time")
        if self.mode is TimeWindowMode.HISTORY and (
            self.start_time != 0.0 or self.end_time != self.query_time
        ):
            raise ValueError("a valid history window must be [0, query_time]")
        if self.mode is TimeWindowMode.RECENT and (
            self.start_time is None
            or self.start_time >= self.end_time
            or self.end_time != self.query_time
        ):
            raise ValueError("a valid recent window requires 0 <= start < end == query_time")
        if self.mode is TimeWindowMode.EXPLICIT_RANGE and self.start_time is None:
            raise ValueError("a valid explicit range requires start_time")


@dataclass(frozen=True, slots=True)
class QueryEncoderInput:
    question_embeddings: Tensor
    question_tokens: QuestionTokenBatch
    query_time: Tensor
    explicit_time_values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        embeddings = self.question_embeddings
        if embeddings.ndim != 3 or not torch.is_floating_point(embeddings):
            raise ValueError("question_embeddings must be floating [B, L_q, D]")
        batch_size, width, _ = embeddings.shape
        if self.question_tokens.input_ids.shape != (batch_size, width):
            raise ValueError("question token shape must match question_embeddings [B, L_q]")
        if self.query_time.shape != (batch_size,) or not torch.is_floating_point(self.query_time):
            raise ValueError("query_time must be floating [B]")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("question_embeddings must be finite")
        if not bool(torch.isfinite(self.query_time).all()) or bool(torch.any(self.query_time < 0)):
            raise ValueError("query_time must be finite and non-negative")
        if len(self.explicit_time_values) != batch_size:
            raise ValueError("explicit_time_values must contain one tuple per batch item")
        if any(
            not math.isfinite(value) or value < 0.0
            for values in self.explicit_time_values
            for value in values
        ):
            raise ValueError("explicit_time_values must be finite and non-negative")

    @classmethod
    def from_runtime_queries(
        cls,
        question_embeddings: Tensor,
        question_tokens: QuestionTokenBatch,
        queries: Sequence[RuntimeQueryInput],
    ) -> Self:
        """Bind embeddings to already validated typed runtime Queries."""

        batch_size = question_embeddings.shape[0] if question_embeddings.ndim == 3 else -1
        rows = tuple(queries)
        if len(rows) != batch_size:
            raise ValueError("runtime Queries must align to question embeddings")
        questions = tuple(query.question for query in rows)
        if questions != question_tokens.questions:
            raise ValueError("runtime questions must exactly match canonical question tokens")
        query_time = torch.tensor(
            [query.query_time for query in rows],
            dtype=question_embeddings.dtype,
            device=question_embeddings.device,
        )
        return cls(
            question_embeddings=question_embeddings,
            question_tokens=question_tokens,
            query_time=query_time,
            explicit_time_values=tuple(query.explicit_time_values for query in rows),
        )

    @property
    def padding_mask(self) -> Tensor:
        return self.question_tokens.padding_mask.to(self.question_embeddings.device)


@dataclass(frozen=True, slots=True)
class QueryEmbeddingOutput:
    token_states: Tensor
    pooling_weights: Tensor
    q_target: Tensor
    q_operator: Tensor
    q_time: Tensor
    padding_mask: Tensor

    def __post_init__(self) -> None:
        states = self.token_states
        if states.ndim != 3 or not torch.is_floating_point(states):
            raise ValueError("token_states must be floating [B, L_q, H]")
        batch_size, width, _ = states.shape
        if self.pooling_weights.shape != (batch_size, width):
            raise ValueError("pooling_weights must be [B, L_q]")
        if not torch.is_floating_point(self.pooling_weights):
            raise ValueError("pooling_weights must be floating")
        if self.padding_mask.shape != (batch_size, width) or self.padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [B, L_q]")
        output_dim = self.q_target.shape[1] if self.q_target.ndim == 2 else -1
        for embedding in (self.q_target, self.q_operator, self.q_time):
            if embedding.shape != (batch_size, output_dim):
                raise ValueError("query embeddings must share [B, D_q] shape")
            if embedding.dtype != states.dtype or embedding.device != states.device:
                raise ValueError("query embeddings must share token state dtype/device")
        tensors = (
            states,
            self.pooling_weights,
            self.q_target,
            self.q_operator,
            self.q_time,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("query encoder outputs must be finite")
        if bool(torch.any(self.pooling_weights[self.padding_mask] != 0)):
            raise ValueError("padding pooling weights must be exactly zero")
        if not torch.allclose(
            self.pooling_weights.sum(dim=1),
            torch.ones(
                batch_size,
                dtype=self.pooling_weights.dtype,
                device=self.pooling_weights.device,
            ),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("pooling weights must sum to one")


@dataclass(frozen=True, slots=True)
class OperatorRouterOutput:
    logits: Tensor
    confidence: Tensor
    raw_indices: Tensor
    hard_operators: tuple[Operator, ...]
    head_types: tuple[HeadType | None, ...]
    confidence_gate_applied: bool

    def __post_init__(self) -> None:
        batch_size = self.logits.shape[0] if self.logits.ndim == 2 else -1
        if self.logits.shape != (batch_size, len(Operator)):
            raise ValueError("operator logits must be [B, 9]")
        if self.confidence.shape != (batch_size,) or not torch.is_floating_point(self.confidence):
            raise ValueError("operator confidence must be floating [B]")
        if self.raw_indices.shape != (batch_size,) or self.raw_indices.dtype != torch.int64:
            raise ValueError("raw operator indices must be int64 [B]")
        if len(self.hard_operators) != batch_size or len(self.head_types) != batch_size:
            raise ValueError("hard operator metadata must contain one entry per batch item")
        if not bool(torch.isfinite(self.logits).all()) or not bool(
            torch.isfinite(self.confidence).all()
        ):
            raise ValueError("operator outputs must be finite")
        expected_heads = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in self.hard_operators)
        if self.head_types != expected_heads:
            raise ValueError("operator head types must follow the deterministic mapping")


@dataclass(frozen=True, slots=True)
class TimeResolverLogits:
    mode_logits: Tensor
    mode_confidence: Tensor
    mode_indices: Tensor
    span_start_logits: Tensor
    span_end_logits: Tensor
    padding_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.mode_logits.shape[0] if self.mode_logits.ndim == 2 else -1
        if self.mode_logits.shape != (batch_size, len(TimeWindowMode)):
            raise ValueError("time mode logits must be [B, 4]")
        if self.mode_confidence.shape != (batch_size,):
            raise ValueError("time mode confidence must be [B]")
        if self.mode_indices.shape != (batch_size,) or self.mode_indices.dtype != torch.int64:
            raise ValueError("time mode indices must be int64 [B]")
        if self.span_start_logits.shape != self.span_end_logits.shape:
            raise ValueError("numeric span start/end logits must share [B, L_q] shape")
        if self.span_start_logits.shape[0] != batch_size:
            raise ValueError("numeric span logits batch size must match mode logits")
        if self.padding_mask.shape != self.span_start_logits.shape:
            raise ValueError("time resolver padding mask must match numeric span logits")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("time resolver padding_mask must use bool dtype")
        tensors = (
            self.mode_logits,
            self.mode_confidence,
            self.span_start_logits,
            self.span_end_logits,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("time resolver logits must be finite")


@dataclass(frozen=True, slots=True)
class TimeResolution:
    window: TimeWindow
    status: TimeResolutionStatus
    reason: str
    mode_confidence: float
    numeric_span: tuple[int, int] | None
    parsed_values_seconds: tuple[float, ...]
    used_operator_default: bool

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("time resolution reason must be non-empty")
        if not math.isfinite(self.mode_confidence) or not 0.0 <= self.mode_confidence <= 1.0:
            raise ValueError("time resolution confidence must be finite within [0, 1]")
        if self.numeric_span is not None:
            start, end = self.numeric_span
            if start < 0 or end <= start:
                raise ValueError("numeric_span must be a non-empty [start, end) token interval")
        if any(not math.isfinite(value) for value in self.parsed_values_seconds):
            raise ValueError("parsed time values must be finite")
        if (self.status is TimeResolutionStatus.OK) != self.window.valid:
            raise ValueError("only successful time resolutions may contain a valid TimeWindow")


@dataclass(frozen=True, slots=True)
class TimeResolverOutput:
    logits: TimeResolverLogits
    resolutions: tuple[TimeResolution, ...]

    def __post_init__(self) -> None:
        if len(self.resolutions) != self.logits.mode_logits.shape[0]:
            raise ValueError("time resolutions must contain one entry per batch item")


@dataclass(frozen=True, slots=True)
class QueryEncoderOutput:
    embeddings: QueryEmbeddingOutput
    route: OperatorRouterOutput
    time: TimeResolverOutput
    hard_operators: tuple[Operator, ...]
    head_types: tuple[HeadType | None, ...]

    def __post_init__(self) -> None:
        batch_size = self.embeddings.q_target.shape[0]
        if len(self.hard_operators) != batch_size or len(self.head_types) != batch_size:
            raise ValueError("effective operator metadata must contain one entry per batch item")
        expected = tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in self.hard_operators)
        if self.head_types != expected:
            raise ValueError("effective head types must match hard operators")
        for operator, resolution in zip(
            self.hard_operators,
            self.time.resolutions,
            strict=True,
        ):
            if (
                resolution.status is not TimeResolutionStatus.OK
                and operator is not Operator.UNSUPPORTED
            ):
                raise ValueError(
                    "invalid time resolutions must force the effective operator unsupported"
                )

    @property
    def q_target(self) -> Tensor:
        return self.embeddings.q_target

    @property
    def q_operator(self) -> Tensor:
        return self.embeddings.q_operator

    @property
    def q_time(self) -> Tensor:
        return self.embeddings.q_time

    @property
    def operator_logits(self) -> Tensor:
        return self.route.logits

    @property
    def padding_mask(self) -> Tensor:
        return self.embeddings.padding_mask


def detach_query_encoder_output(output: QueryEncoderOutput) -> QueryEncoderOutput:
    """Return a typed detached view without copying Query metadata or Tensor storage."""

    return replace(
        output,
        embeddings=replace(
            output.embeddings,
            token_states=output.embeddings.token_states.detach(),
            pooling_weights=output.embeddings.pooling_weights.detach(),
            q_target=output.embeddings.q_target.detach(),
            q_operator=output.embeddings.q_operator.detach(),
            q_time=output.embeddings.q_time.detach(),
            padding_mask=output.embeddings.padding_mask.detach(),
        ),
        route=replace(
            output.route,
            logits=output.route.logits.detach(),
            confidence=output.route.confidence.detach(),
            raw_indices=output.route.raw_indices.detach(),
        ),
        time=replace(
            output.time,
            logits=replace(
                output.time.logits,
                mode_logits=output.time.logits.mode_logits.detach(),
                mode_confidence=output.time.logits.mode_confidence.detach(),
                mode_indices=output.time.logits.mode_indices.detach(),
                span_start_logits=output.time.logits.span_start_logits.detach(),
                span_end_logits=output.time.logits.span_end_logits.detach(),
                padding_mask=output.time.logits.padding_mask.detach(),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class QueryEncoderSupervision:
    """Separate training-only targets; this type is deliberately absent from forward()."""

    operator_targets: Tensor
    time_mode_targets: Tensor
    span_start_targets: Tensor
    span_end_targets: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.operator_targets,
            self.time_mode_targets,
            self.span_start_targets,
            self.span_end_targets,
        )
        if any(tensor.ndim != 1 or tensor.dtype != torch.int64 for tensor in tensors):
            raise ValueError("query supervision targets must be int64 [B]")
        if len({tensor.shape for tensor in tensors}) != 1:
            raise ValueError("query supervision targets must share one batch shape")
        if bool(torch.any((self.operator_targets < 0) | (self.operator_targets >= len(Operator)))):
            raise ValueError("operator targets must be within [0, 9)")
        if bool(
            torch.any(
                (self.time_mode_targets < 0) | (self.time_mode_targets >= len(TimeWindowMode))
            )
        ):
            raise ValueError("time mode targets must be within [0, 4)")
        for target in (self.span_start_targets, self.span_end_targets):
            if bool(torch.any((target < 0) & (target != -100))):
                raise ValueError("span targets may use only non-negative indices or -100 ignore")
        start_ignored = self.span_start_targets == -100
        end_ignored = self.span_end_targets == -100
        if not torch.equal(start_ignored, end_ignored):
            raise ValueError("numeric span start/end targets must be ignored together")
        valid = ~start_ignored
        if bool(torch.any(self.span_start_targets[valid] > self.span_end_targets[valid])):
            raise ValueError("numeric span targets use inclusive indices with start <= end")


class QueryEmbeddingEncoder(nn.Module):  # type: ignore[misc]
    """4096-to-768 Pre-LN bidirectional encoder with three independent semantic heads."""

    def __init__(self, config: QueryEncoderConfig) -> None:
        super().__init__()
        if not config.bidirectional:
            raise ValueError("Query Transformer must be bidirectional")
        if config.position_encoding != "sinusoidal":
            raise ValueError("Query Transformer requires sinusoidal position encoding")
        if config.pooling != "learned_attention":
            raise ValueError("Query Encoder requires learned_attention pooling")
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.pool_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.pool_scorer = nn.Linear(config.hidden_dim, 1, bias=False)
        self.target_head = _embedding_head(config.hidden_dim, config.output_dim)
        self.operator_head = _embedding_head(config.hidden_dim, config.output_dim)
        self.time_head = _embedding_head(config.hidden_dim, config.output_dim)

    def forward(self, question_embeddings: Tensor, padding_mask: Tensor) -> QueryEmbeddingOutput:
        if question_embeddings.ndim != 3 or question_embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"question_embeddings must be [B, L_q, {self.input_dim}]")
        if padding_mask.shape != question_embeddings.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [B, L_q]")
        if padding_mask.device != question_embeddings.device:
            raise ValueError("padding_mask and question_embeddings must share a device")
        if bool(torch.any(padding_mask.all(dim=1))):
            raise ValueError("every question must contain at least one non-padding token")
        if not bool(torch.isfinite(question_embeddings).all()):
            raise ValueError("question_embeddings must be finite")
        clean_embeddings = question_embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self.input_projection(clean_embeddings)
        positions = _sinusoidal_position_encoding(
            hidden.shape[1],
            hidden.shape[2],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = (hidden + positions.unsqueeze(0)).masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )
        token_states = self.transformer(
            hidden,
            src_key_padding_mask=padding_mask,
            is_causal=False,
        )
        scores = self.pool_scorer(torch.tanh(self.pool_projection(token_states))).squeeze(-1)
        scores = scores.masked_fill(padding_mask, -torch.inf)
        # Keep learned-attention probabilities in FP32.  Under BF16 autocast,
        # normalizing in the model dtype can still leave the reduced sum at
        # 1 +/- 1 BF16 ULP (for example 1.0078125), which is large enough to
        # trip the structural invariant below and desynchronize DDP ranks.
        pooling_weights = torch.softmax(scores.float(), dim=1).masked_fill(
            padding_mask,
            0.0,
        )
        pooling_weights = pooling_weights / pooling_weights.sum(dim=1, keepdim=True)
        pooled = torch.sum(
            pooling_weights.unsqueeze(-1) * token_states.float(),
            dim=1,
        ).to(dtype=token_states.dtype)
        return QueryEmbeddingOutput(
            token_states=token_states,
            pooling_weights=pooling_weights,
            q_target=F.normalize(self.target_head(pooled), dim=-1),
            q_operator=F.normalize(self.operator_head(pooled), dim=-1),
            q_time=F.normalize(self.time_head(pooled), dim=-1),
            padding_mask=padding_mask,
        )


class OperatorRouter(nn.Module):  # type: ignore[misc]
    """Nine normalized trainable prototypes with a positive learned temperature."""

    def __init__(self, config: OperatorRouterConfig) -> None:
        super().__init__()
        names = tuple(operator.value for operator in Operator)
        if config.prototypes != names:
            raise ValueError("Operator Router requires the frozen nine prototype names")
        self.output_dim = config.output_dim
        self.confidence_threshold = config.confidence_threshold
        self.prototypes = nn.Parameter(torch.empty(len(Operator), config.output_dim))
        nn.init.normal_(self.prototypes, std=config.output_dim**-0.5)
        initial_log_temperature = math.log(config.temperature_initial)
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature),
            requires_grad=config.temperature_trainable,
        )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(min=1.0e-4, max=1.0e4)

    def forward(
        self,
        q_operator: Tensor,
        *,
        apply_confidence_gate: bool,
    ) -> OperatorRouterOutput:
        if q_operator.ndim != 2 or q_operator.shape[1] != self.output_dim:
            raise ValueError(f"q_operator must be [B, {self.output_dim}]")
        if not torch.is_floating_point(q_operator) or not bool(torch.isfinite(q_operator).all()):
            raise ValueError("q_operator must be finite and floating")
        logits = (
            F.normalize(q_operator, dim=-1) @ F.normalize(self.prototypes, dim=-1).transpose(0, 1)
        ) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        confidence, raw_indices = probabilities.max(dim=-1)
        effective_indices = raw_indices.clone()
        if apply_confidence_gate:
            if self.confidence_threshold is None:
                effective_indices.fill_(UNSUPPORTED_OPERATOR_INDEX)
            else:
                effective_indices = torch.where(
                    confidence < self.confidence_threshold,
                    torch.full_like(effective_indices, UNSUPPORTED_OPERATOR_INDEX),
                    effective_indices,
                )
        hard_operators = tuple(
            OPERATORS[index] for index in effective_indices.detach().cpu().tolist()
        )
        return OperatorRouterOutput(
            logits=logits,
            confidence=confidence,
            raw_indices=raw_indices,
            hard_operators=hard_operators,
            head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in hard_operators),
            confidence_gate_applied=apply_confidence_gate,
        )


class TimeWindowResolver(nn.Module):  # type: ignore[misc]
    """Predict time semantics/spans, then build conservative deterministic windows."""

    def __init__(self, config: TimeResolverConfig) -> None:
        super().__init__()
        if config.modes != tuple(mode.value for mode in TimeWindowMode):
            raise ValueError("Time Resolver requires the frozen four modes")
        if config.pointer_heads != 2:
            raise ValueError("Time Resolver requires exactly two pointer heads")
        self.input_dim = config.input_dim
        self.token_hidden_dim = config.token_hidden_dim
        self.confidence_threshold = config.confidence_threshold
        self.mode_classifier = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.mode_count),
        )
        self.span_start = nn.Linear(config.token_hidden_dim, 1)
        self.span_end = nn.Linear(config.token_hidden_dim, 1)

    def forward(
        self,
        q_time: Tensor,
        token_states: Tensor,
        padding_mask: Tensor,
    ) -> TimeResolverLogits:
        if q_time.ndim != 2 or q_time.shape[1] != self.input_dim:
            raise ValueError(f"q_time must be [B, {self.input_dim}]")
        if token_states.ndim != 3 or token_states.shape[-1] != self.token_hidden_dim:
            raise ValueError(f"token_states must be [B, L_q, {self.token_hidden_dim}]")
        if token_states.shape[0] != q_time.shape[0]:
            raise ValueError("q_time and token_states batch sizes must match")
        if padding_mask.shape != token_states.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [B, L_q]")
        if q_time.device != token_states.device or padding_mask.device != token_states.device:
            raise ValueError("time resolver tensors must share a device")
        if q_time.dtype != token_states.dtype:
            raise ValueError("q_time and token_states must share dtype")
        if not bool(torch.isfinite(q_time).all()) or not bool(torch.isfinite(token_states).all()):
            raise ValueError("time resolver inputs must be finite")
        mode_logits = self.mode_classifier(q_time)
        mode_probabilities = torch.softmax(mode_logits, dim=-1)
        mode_confidence, mode_indices = mode_probabilities.max(dim=-1)
        minimum = torch.finfo(token_states.dtype).min
        span_start_logits = (
            self.span_start(token_states)
            .squeeze(-1)
            .masked_fill(
                padding_mask,
                minimum,
            )
        )
        span_end_logits = (
            self.span_end(token_states)
            .squeeze(-1)
            .masked_fill(
                padding_mask,
                minimum,
            )
        )
        return TimeResolverLogits(
            mode_logits=mode_logits,
            mode_confidence=mode_confidence,
            mode_indices=mode_indices,
            span_start_logits=span_start_logits,
            span_end_logits=span_end_logits,
            padding_mask=padding_mask,
        )

    def resolve(
        self,
        logits: TimeResolverLogits,
        query_input: QueryEncoderInput,
        hard_operators: Sequence[Operator],
        *,
        apply_confidence_gate: bool,
    ) -> TimeResolverOutput:
        batch_size = logits.mode_logits.shape[0]
        if len(hard_operators) != batch_size:
            raise ValueError("hard_operators must contain one value per batch item")
        expected_padding = query_input.padding_mask
        if logits.padding_mask.shape != expected_padding.shape or not torch.equal(
            logits.padding_mask,
            expected_padding,
        ):
            raise ValueError("time resolver logits must use the query input padding mask")
        resolutions = tuple(
            self._resolve_one(
                row,
                logits,
                query_input,
                hard_operators[row],
                apply_confidence_gate=apply_confidence_gate,
            )
            for row in range(batch_size)
        )
        return TimeResolverOutput(logits=logits, resolutions=resolutions)

    def _resolve_one(
        self,
        row: int,
        logits: TimeResolverLogits,
        query_input: QueryEncoderInput,
        operator: Operator,
        *,
        apply_confidence_gate: bool,
    ) -> TimeResolution:
        query_time = float(query_input.query_time[row].detach().cpu().item())
        confidence = float(logits.mode_confidence[row].detach().cpu().item())
        if operator is Operator.UNSUPPORTED:
            return _failed_time_resolution(
                query_time,
                TimeWindowMode.HISTORY,
                confidence,
                TimeResolutionStatus.UNSUPPORTED,
                "unsupported_operator",
            )
        question = query_input.question_tokens.questions[row]
        candidate, parse_error = _parse_time_candidate(question)
        if parse_error is not None:
            return _failed_time_resolution(
                query_time,
                OPERATOR_DEFAULT_TIME_MODE[operator] or TimeWindowMode.HISTORY,
                confidence,
                TimeResolutionStatus.INVALID,
                parse_error,
            )
        expected_explicit_values = extract_explicit_time_values(question)
        supplied_explicit_values = query_input.explicit_time_values[row]
        if not _time_values_match(supplied_explicit_values, expected_explicit_values):
            return _failed_time_resolution(
                query_time,
                (
                    candidate.mode
                    if candidate is not None
                    else OPERATOR_DEFAULT_TIME_MODE[operator] or TimeWindowMode.HISTORY
                ),
                confidence,
                TimeResolutionStatus.INVALID,
                "explicit_time_values_mismatch",
                parsed_values=candidate.values_seconds if candidate is not None else (),
            )
        used_default = candidate is None
        desired_mode = (
            candidate.mode if candidate is not None else OPERATOR_DEFAULT_TIME_MODE[operator]
        )
        if desired_mode is None:
            return _failed_time_resolution(
                query_time,
                TimeWindowMode.HISTORY,
                confidence,
                TimeResolutionStatus.UNSUPPORTED,
                "unsupported_operator",
            )
        if apply_confidence_gate:
            if self.confidence_threshold is None or confidence < self.confidence_threshold:
                return _failed_time_resolution(
                    query_time,
                    desired_mode,
                    confidence,
                    TimeResolutionStatus.UNSUPPORTED,
                    "uncalibrated_or_low_time_confidence",
                )
            predicted_mode = TIME_MODES[int(logits.mode_indices[row].item())]
            if predicted_mode is not desired_mode:
                return _failed_time_resolution(
                    query_time,
                    desired_mode,
                    confidence,
                    TimeResolutionStatus.UNSUPPORTED,
                    "time_mode_expression_mismatch",
                )
        numeric_span: tuple[int, int] | None = None
        parsed_values: tuple[float, ...] = ()
        if candidate is not None:
            numeric_span, span_error = _select_numeric_token_span(
                row,
                candidate,
                logits,
                query_input.question_tokens,
            )
            parsed_values = candidate.values_seconds
            if span_error is not None:
                return _failed_time_resolution(
                    query_time,
                    desired_mode,
                    confidence,
                    TimeResolutionStatus.INVALID,
                    span_error,
                    parsed_values=parsed_values,
                )
        window, window_error = _build_time_window(desired_mode, query_time, parsed_values)
        if window_error is not None:
            return _failed_time_resolution(
                query_time,
                desired_mode,
                confidence,
                TimeResolutionStatus.INVALID,
                window_error,
                numeric_span=numeric_span,
                parsed_values=parsed_values,
                used_default=used_default,
            )
        if window is None:
            raise RuntimeError("valid time-window construction returned no window")
        return TimeResolution(
            window=window,
            status=TimeResolutionStatus.OK,
            reason="operator_default" if used_default else "explicit_question_time",
            mode_confidence=confidence,
            numeric_span=numeric_span,
            parsed_values_seconds=parsed_values,
            used_operator_default=used_default,
        )


class QueryEncoder(nn.Module):  # type: ignore[misc]
    """P4 composition of question embedding, operator routing, and time resolution."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.embedding_encoder = QueryEmbeddingEncoder(config.query_encoder)
        self.operator_router = OperatorRouter(config.operator_router)
        self.time_resolver = TimeWindowResolver(config.time_resolver)

    def forward(
        self,
        query_input: QueryEncoderInput,
        *,
        inference: bool | None = None,
    ) -> QueryEncoderOutput:
        apply_confidence_gate = not self.training if inference is None else inference
        padding_mask = query_input.padding_mask
        embeddings = self.embedding_encoder(
            query_input.question_embeddings,
            padding_mask,
        )
        route = self.operator_router(
            embeddings.q_operator,
            apply_confidence_gate=apply_confidence_gate,
        )
        time_logits = self.time_resolver(
            embeddings.q_time,
            embeddings.token_states,
            embeddings.padding_mask,
        )
        time_output = self.time_resolver.resolve(
            time_logits,
            query_input,
            route.hard_operators,
            apply_confidence_gate=apply_confidence_gate,
        )
        effective_operators = tuple(
            operator if resolution.status is TimeResolutionStatus.OK else Operator.UNSUPPORTED
            for operator, resolution in zip(
                route.hard_operators,
                time_output.resolutions,
                strict=True,
            )
        )
        return QueryEncoderOutput(
            embeddings=embeddings,
            route=route,
            time=time_output,
            hard_operators=effective_operators,
            head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in effective_operators),
        )


class InputEmbeddingOwner(Protocol):
    def get_input_embeddings(self) -> object: ...


def embed_question_tokens(
    qwen_model: InputEmbeddingOwner,
    question_tokens: QuestionTokenBatch,
    config: ProjectConfig,
) -> Tensor:
    """Run only Qwen's token embedding table, never the 36-layer answer decoder."""

    embedding_layer = qwen_model.get_input_embeddings()
    if not callable(embedding_layer):
        raise TypeError("Qwen get_input_embeddings() must return a callable embedding layer")
    parameters = getattr(embedding_layer, "parameters", None)
    device = question_tokens.input_ids.device
    if callable(parameters):
        first_parameter = next(iter(parameters()), None)
        if isinstance(first_parameter, Tensor):
            device = first_parameter.device
    embeddings = cast(Tensor, embedding_layer(question_tokens.input_ids.to(device)))
    expected_shape = (*question_tokens.input_ids.shape, config.query_encoder.input_dim)
    if embeddings.shape != expected_shape or not torch.is_floating_point(embeddings):
        raise ValueError(
            "Qwen question embeddings must be floating "
            f"{expected_shape}; got {tuple(embeddings.shape)}"
        )
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Qwen question embeddings must be finite")
    return embeddings


def build_query_encoder(config: ProjectConfig | None = None) -> QueryEncoder:
    """Build the P4 network from the fully validated v5 project config."""

    if config is None:
        raise ValueError("build_query_encoder requires a validated ProjectConfig")
    return QueryEncoder(config)


def query_embedding_parameter_count(module: QueryEmbeddingEncoder) -> int:
    """Count the P4 36.03M backbone/pooling/three-head parameters only."""

    return sum(parameter.numel() for parameter in module.parameters())


def operator_router_parameter_count(module: OperatorRouter) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def time_resolver_parameter_count(module: TimeWindowResolver) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _embedding_head(hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, 1024),
        nn.GELU(),
        nn.Linear(1024, output_dim),
    )


def _sinusoidal_position_encoding(
    length: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_dim)
    )
    encoding = torch.zeros(length, hidden_dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: hidden_dim // 2])
    return encoding.to(dtype=dtype)


@dataclass(frozen=True, slots=True)
class _TimeCandidate:
    mode: TimeWindowMode
    values_seconds: tuple[float, ...]
    char_span: tuple[int, int]
    start_component_span: tuple[int, int]
    end_component_span: tuple[int, int]


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = r"(?:minutes?|mins?|min|seconds?|secs?|sec|分钟|秒钟|秒|m|s)"
_COMPONENT = rf"{_NUMBER}\s*{_UNIT}(?![A-Za-z])"
_TIME_COMPONENT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_NEGATIVE_TIME_PATTERN = re.compile(
    rf"-\s*{_NUMBER}\s*{_UNIT}(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_UNSUPPORTED_TIME_UNIT_PATTERN = re.compile(
    rf"(?<![\w.]){_NUMBER}\s*(?:hours?|hrs?|days?|weeks?|months?|years?|"
    rf"milliseconds?|ms|frames?|小时|天|周|星期|个月|年|毫秒|帧)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
_RECENT_PATTERNS = (
    re.compile(
        rf"\b(?:last|past|previous)\s+(?P<body>{_COMPONENT}(?:\s*(?:and|\+)\s*{_COMPONENT})*)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:最近|过去|近)\s*(?P<body>{_COMPONENT}(?:\s*(?:和|\+)\s*{_COMPONENT})*)(?:内)?",
        flags=re.IGNORECASE,
    ),
)
_RANGE_PATTERNS = (
    re.compile(
        rf"\b(?:from|between)\s+(?P<start>{_NUMBER})\s*(?P<start_unit>{_UNIT})?\s+(?:to|and)\s+(?P<end>{_NUMBER})\s*(?P<end_unit>{_UNIT})(?![A-Za-z])",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"从\s*(?P<start>{_NUMBER})\s*(?P<start_unit>{_UNIT})?\s*(?:到|至)\s*(?P<end>{_NUMBER})\s*(?P<end_unit>{_UNIT})",
        flags=re.IGNORECASE,
    ),
)


def _parse_time_candidate(question: str) -> tuple[_TimeCandidate | None, str | None]:
    if _NEGATIVE_TIME_PATTERN.search(question):
        return None, "negative_time_value"
    if _UNSUPPORTED_TIME_UNIT_PATTERN.search(question):
        return None, "unsupported_time_unit"
    candidates: list[_TimeCandidate] = []
    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(question):
            start_unit = match.group("start_unit") or match.group("end_unit")
            end_unit = match.group("end_unit")
            start = _to_seconds(float(match.group("start")), start_unit)
            end = _to_seconds(float(match.group("end")), end_unit)
            candidates.append(
                _TimeCandidate(
                    mode=TimeWindowMode.EXPLICIT_RANGE,
                    values_seconds=(start, end),
                    char_span=match.span(),
                    start_component_span=(
                        match.start("start"),
                        match.end("start_unit")
                        if match.group("start_unit")
                        else match.end("start"),
                    ),
                    end_component_span=(match.start("end"), match.end("end_unit")),
                )
            )
    for pattern in _RECENT_PATTERNS:
        for match in pattern.finditer(question):
            components = tuple(_TIME_COMPONENT_PATTERN.finditer(match.group("body")))
            if not components:
                continue
            duration = sum(
                _to_seconds(float(component.group("value")), component.group("unit"))
                for component in components
            )
            body_start = match.start("body")
            first = components[0]
            last = components[-1]
            candidates.append(
                _TimeCandidate(
                    mode=TimeWindowMode.RECENT,
                    values_seconds=(duration,),
                    char_span=match.span(),
                    start_component_span=(
                        body_start + first.start(),
                        body_start + first.end(),
                    ),
                    end_component_span=(
                        body_start + last.start(),
                        body_start + last.end(),
                    ),
                )
            )
    unique = {
        (candidate.mode, candidate.char_span, candidate.values_seconds): candidate
        for candidate in candidates
    }
    candidates = list(unique.values())
    if len(candidates) > 1:
        return None, "ambiguous_time_expression"
    all_components = tuple(_TIME_COMPONENT_PATTERN.finditer(question))
    if not candidates:
        if all_components:
            return None, "unsupported_time_syntax"
        return None, None
    candidate = candidates[0]
    if any(
        not (
            candidate.char_span[0] <= component.start()
            and component.end() <= candidate.char_span[1]
        )
        for component in all_components
    ):
        return None, "ambiguous_time_expression"
    return candidate, None


def _to_seconds(value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"minute", "minutes", "min", "mins", "m", "分钟"}:
        return value * 60.0
    return value


def _select_numeric_token_span(
    row: int,
    candidate: _TimeCandidate,
    logits: TimeResolverLogits,
    tokens: QuestionTokenBatch,
) -> tuple[tuple[int, int] | None, str | None]:
    offsets = tokens.offset_mapping[row]
    valid_width = tokens.spans[row].end
    start_index = int(torch.argmax(logits.span_start_logits[row, :valid_width]).item())
    end_index = int(torch.argmax(logits.span_end_logits[row, :valid_width]).item())
    if start_index > end_index:
        return None, "pointer_order_invalid"
    if bool(tokens.padding_mask[row, start_index]) or bool(tokens.padding_mask[row, end_index]):
        return None, "pointer_selected_padding"
    start_candidates = _tokens_overlapping(offsets, valid_width, candidate.start_component_span)
    end_candidates = _tokens_overlapping(offsets, valid_width, candidate.end_component_span)
    if not start_candidates or not end_candidates:
        return None, "numeric_expression_has_no_token_alignment"
    if start_index not in start_candidates or end_index not in end_candidates:
        return None, "pointer_outside_numeric_expression"
    selected_char_start = int(offsets[start_index, 0].item())
    selected_char_end = int(offsets[end_index, 1].item())
    required_char_start = candidate.start_component_span[0]
    required_char_end = candidate.end_component_span[1]
    if selected_char_start > required_char_start or selected_char_end < required_char_end:
        return None, "pointer_does_not_cover_numeric_expression"
    return (start_index, end_index + 1), None


def _time_values_match(actual: tuple[float, ...], expected: tuple[float, ...]) -> bool:
    return actual == expected


def _tokens_overlapping(
    offsets: Tensor,
    valid_width: int,
    char_span: tuple[int, int],
) -> tuple[int, ...]:
    char_start, char_end = char_span
    indices = []
    for index in range(valid_width):
        token_start, token_end = (int(value) for value in offsets[index].tolist())
        if token_end > token_start and token_end > char_start and token_start < char_end:
            indices.append(index)
    return tuple(indices)


def _build_time_window(
    mode: TimeWindowMode,
    query_time: float,
    values: tuple[float, ...],
) -> tuple[TimeWindow | None, str | None]:
    if mode is TimeWindowMode.NOW:
        if values:
            return None, "now_mode_cannot_consume_numeric_window"
        return TimeWindow(mode, query_time, None, query_time, True), None
    if mode is TimeWindowMode.HISTORY:
        if values:
            return None, "history_mode_cannot_consume_numeric_window"
        return TimeWindow(mode, query_time, 0.0, query_time, True), None
    if mode is TimeWindowMode.RECENT:
        if len(values) != 1:
            return None, "recent_requires_one_duration"
        duration = values[0]
        if duration <= 0.0:
            return None, "recent_duration_must_be_positive"
        if duration > query_time:
            return None, "recent_window_starts_before_video"
        return TimeWindow(
            mode,
            query_time,
            query_time - duration,
            query_time,
            True,
        ), None
    if len(values) != 2:
        return None, "explicit_range_requires_two_endpoints"
    start, end = values
    if start > end:
        return None, "explicit_range_is_reversed"
    if end > query_time:
        return None, "explicit_range_ends_after_query_time"
    return TimeWindow(mode, query_time, start, end, True), None


def _failed_time_resolution(
    query_time: float,
    mode: TimeWindowMode,
    confidence: float,
    status: TimeResolutionStatus,
    reason: str,
    *,
    numeric_span: tuple[int, int] | None = None,
    parsed_values: tuple[float, ...] = (),
    used_default: bool = False,
) -> TimeResolution:
    return TimeResolution(
        window=TimeWindow(
            mode=mode,
            query_time=query_time,
            start_time=None,
            end_time=query_time,
            valid=False,
        ),
        status=status,
        reason=reason,
        mode_confidence=confidence,
        numeric_span=numeric_span,
        parsed_values_seconds=parsed_values,
        used_operator_default=used_default,
    )
