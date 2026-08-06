"""Encode question-only token embeddings and resolve operator/time metadata."""

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
from ttt_svcbench_qwen.data import RuntimeQueryInput
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


@dataclass(frozen=True, slots=True)
class QueryEncoderInput:
    question_embeddings: Tensor
    question_tokens: QuestionTokenBatch
    query_time: Tensor
    explicit_time_values: tuple[tuple[float, ...], ...]

    @classmethod
    def from_runtime_queries(
        cls,
        question_embeddings: Tensor,
        question_tokens: QuestionTokenBatch,
        queries: Sequence[RuntimeQueryInput],
    ) -> Self:
        """Bind embeddings to already validated typed runtime Queries."""

        rows = tuple(queries)
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
    q_target: Tensor
    q_operator: Tensor
    q_time: Tensor
    padding_mask: Tensor


@dataclass(frozen=True, slots=True)
class OperatorRouterOutput:
    logits: Tensor
    confidence: Tensor
    raw_indices: Tensor
    hard_operators: tuple[Operator, ...]
    head_types: tuple[HeadType | None, ...]
    temperature: Tensor | None = None


@dataclass(frozen=True, slots=True)
class TimeResolverLogits:
    mode_logits: Tensor
    mode_confidence: Tensor
    mode_indices: Tensor
    span_start_logits: Tensor
    span_end_logits: Tensor
    padding_mask: Tensor


@dataclass(frozen=True, slots=True)
class TimeResolution:
    window: TimeWindow
    status: TimeResolutionStatus


@dataclass(frozen=True, slots=True)
class TimeResolverOutput:
    logits: TimeResolverLogits
    resolutions: tuple[TimeResolution, ...]


@dataclass(frozen=True, slots=True)
class QueryEncoderOutput:
    embeddings: QueryEmbeddingOutput
    route: OperatorRouterOutput
    time: TimeResolverOutput
    hard_operators: tuple[Operator, ...]
    head_types: tuple[HeadType | None, ...]

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
            temperature=(
                output.route.temperature.detach() if output.route.temperature is not None else None
            ),
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


class QueryEmbeddingEncoder(nn.Module):  # type: ignore[misc]
    """4096-to-768 Pre-LN bidirectional encoder with three independent semantic heads."""

    def __init__(self, config: QueryEncoderConfig) -> None:
        super().__init__()
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
            q_target=F.normalize(self.target_head(pooled), dim=-1),
            q_operator=F.normalize(self.operator_head(pooled), dim=-1),
            q_time=F.normalize(self.time_head(pooled), dim=-1),
            padding_mask=padding_mask,
        )


class OperatorRouter(nn.Module):  # type: ignore[misc]
    """Nine normalized trainable prototypes with a positive learned temperature."""

    def __init__(self, config: OperatorRouterConfig) -> None:
        super().__init__()
        self.output_dim = config.output_dim
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

    def forward(self, q_operator: Tensor) -> OperatorRouterOutput:
        logits = (
            F.normalize(q_operator, dim=-1) @ F.normalize(self.prototypes, dim=-1).transpose(0, 1)
        ) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        confidence, raw_indices = probabilities.max(dim=-1)
        hard_operators = tuple(OPERATORS[index] for index in raw_indices.detach().cpu().tolist())
        return OperatorRouterOutput(
            logits=logits,
            confidence=confidence,
            raw_indices=raw_indices,
            hard_operators=hard_operators,
            head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in hard_operators),
            temperature=self.temperature,
        )


class TimeWindowResolver(nn.Module):  # type: ignore[misc]
    """Predict time semantics/spans, then build conservative deterministic windows."""

    def __init__(self, config: TimeResolverConfig) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.token_hidden_dim = config.token_hidden_dim
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
    ) -> TimeResolverOutput:
        batch_size = logits.mode_logits.shape[0]
        resolutions = tuple(
            self._resolve_one(row, query_input, hard_operators[row]) for row in range(batch_size)
        )
        return TimeResolverOutput(logits=logits, resolutions=resolutions)

    def _resolve_one(
        self,
        row: int,
        query_input: QueryEncoderInput,
        operator: Operator,
    ) -> TimeResolution:
        query_time = float(query_input.query_time[row].detach().cpu().item())
        if operator is Operator.UNSUPPORTED:
            return _failed_time_resolution(
                query_time,
                TimeWindowMode.HISTORY,
                TimeResolutionStatus.UNSUPPORTED,
            )
        question = query_input.question_tokens.questions[row]
        candidate = _parse_time_candidate(question)
        desired_mode = (
            candidate.mode if candidate is not None else OPERATOR_DEFAULT_TIME_MODE[operator]
        )
        if desired_mode is None:
            return _failed_time_resolution(
                query_time,
                TimeWindowMode.HISTORY,
                TimeResolutionStatus.UNSUPPORTED,
            )
        parsed_values = candidate.values_seconds if candidate is not None else ()
        window = _build_time_window(desired_mode, query_time, parsed_values)
        return TimeResolution(window=window, status=TimeResolutionStatus.OK)


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
        del inference
        padding_mask = query_input.padding_mask
        embeddings = self.embedding_encoder(
            query_input.question_embeddings,
            padding_mask,
        )
        route = self.operator_router(embeddings.q_operator)
        time_logits = self.time_resolver(
            embeddings.q_time,
            embeddings.token_states,
            embeddings.padding_mask,
        )
        time_output = self.time_resolver.resolve(
            time_logits,
            query_input,
            route.hard_operators,
        )
        return QueryEncoderOutput(
            embeddings=embeddings,
            route=route,
            time=time_output,
            hard_operators=route.hard_operators,
            head_types=route.head_types,
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
    parameters = getattr(embedding_layer, "parameters", None)
    device = question_tokens.input_ids.device
    if callable(parameters):
        first_parameter = next(iter(parameters()), None)
        if isinstance(first_parameter, Tensor):
            device = first_parameter.device
    return cast(Tensor, embedding_layer(question_tokens.input_ids.to(device)))  # type: ignore[operator]


def build_query_encoder(config: ProjectConfig) -> QueryEncoder:
    """Build the P4 network from the fully validated v5 project config."""

    return QueryEncoder(config)


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


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = r"(?:minutes?|mins?|min|seconds?|secs?|sec|分钟|秒钟|秒|m|s)"
_COMPONENT = rf"{_NUMBER}\s*{_UNIT}(?![A-Za-z])"
_TIME_COMPONENT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})(?![A-Za-z])",
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


def _parse_time_candidate(question: str) -> _TimeCandidate | None:
    """First-match-wins regex window extraction; no match falls back to the operator default."""

    for pattern in _RANGE_PATTERNS:
        match = pattern.search(question)
        if match is not None:
            start_unit = match.group("start_unit") or match.group("end_unit")
            start = _to_seconds(float(match.group("start")), start_unit)
            end = _to_seconds(float(match.group("end")), match.group("end_unit"))
            return _TimeCandidate(
                mode=TimeWindowMode.EXPLICIT_RANGE,
                values_seconds=(start, end),
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
            return _TimeCandidate(
                mode=TimeWindowMode.RECENT,
                values_seconds=(duration,),
            )
    return None


def _to_seconds(value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"minute", "minutes", "min", "mins", "m", "分钟"}:
        return value * 60.0
    return value


def _build_time_window(
    mode: TimeWindowMode,
    query_time: float,
    values: tuple[float, ...],
) -> TimeWindow:
    """Always return a window; endpoints are clamped into [0, query_time] and ordered."""

    upper = max(query_time, 0.0)
    if mode is TimeWindowMode.NOW:
        return TimeWindow(mode, upper, None, upper, True)
    if mode is TimeWindowMode.HISTORY:
        return TimeWindow(mode, upper, 0.0, upper, True)
    if mode is TimeWindowMode.RECENT:
        duration = values[0] if values else upper
        start = min(max(upper - duration, 0.0), upper)
        return TimeWindow(mode, upper, start, upper, True)
    if len(values) >= 2:
        start, end = sorted(values[:2])
    else:
        start, end = 0.0, upper
    start = min(max(start, 0.0), upper)
    end = min(max(end, start), upper)
    return TimeWindow(mode, upper, start, end, True)


def _failed_time_resolution(
    query_time: float,
    mode: TimeWindowMode,
    status: TimeResolutionStatus,
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
    )
