"""Tokenize only complete question text and expose padding-safe token spans.

Inputs: question strings and the tokenizer pinned by model ID/revision.
Outputs: integer input IDs, attention/padding masks, and per-question [start,end) spans.
Forbidden: system answers, assistant targets, labels, partial-question slicing, or fixed L_q.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor


class TokenizerProtocol(Protocol):
    def __call__(self, text: list[str], **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class QuestionTokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start != 0 or self.end <= self.start:
            raise ValueError(
                "question span must cover one complete non-empty question from index 0"
            )


@dataclass(frozen=True, slots=True)
class QuestionTokenBatch:
    questions: tuple[str, ...]
    input_ids: Tensor
    attention_mask: Tensor
    padding_mask: Tensor
    offset_mapping: Tensor
    spans: tuple[QuestionTokenSpan, ...]
    source_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("question input_ids must be integer [B, L_q]")
        batch_size, width = self.input_ids.shape
        if len(self.questions) != batch_size or not all(
            question.strip() for question in self.questions
        ):
            raise ValueError("questions must contain one complete non-empty string per batch item")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("question attention_mask must match input_ids")
        if self.padding_mask.shape != self.input_ids.shape or self.padding_mask.dtype != torch.bool:
            raise ValueError("question padding_mask must be bool [B, L_q]")
        if self.offset_mapping.shape != (batch_size, width, 2) or self.offset_mapping.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("question offset_mapping must be integer [B, L_q, 2]")
        if len(self.spans) != batch_size:
            raise ValueError("question spans must contain one entry per batch item")
        if self.source_fields != ("question",) * batch_size:
            raise ValueError("question tokens must have question-only provenance")
        expected_padding = self.attention_mask == 0
        if not torch.equal(self.padding_mask, expected_padding):
            raise ValueError("padding_mask must be exactly attention_mask == 0")
        for row, span in enumerate(self.spans):
            if span.end != int(self.attention_mask[row].sum().item()):
                raise ValueError("question span must exclude every padding token")
            if not bool(torch.all(self.attention_mask[row, : span.end] == 1)) or bool(
                torch.any(self.attention_mask[row, span.end :] != 0)
            ):
                raise ValueError("question tokens must use left-aligned right padding")
            offsets = self.offset_mapping[row]
            if bool(torch.any(offsets < 0)) or bool(torch.any(offsets[:, 1] < offsets[:, 0])):
                raise ValueError("question token offsets must be ordered and non-negative")
            valid_offsets = offsets[: span.end]
            if span.end > 1 and (
                bool(torch.any(valid_offsets[1:, 0] < valid_offsets[:-1, 0]))
                or bool(torch.any(valid_offsets[1:, 1] < valid_offsets[:-1, 1]))
            ):
                raise ValueError("question token offsets must be monotonic")
            if bool(torch.any(offsets[: span.end, 1] > len(self.questions[row]))):
                raise ValueError(
                    "question token offsets cannot extend beyond the canonical question"
                )
            if bool(torch.any(offsets[span.end :] != 0)):
                raise ValueError("padding token offsets must be [0, 0]")


def tokenize_questions(
    tokenizer: TokenizerProtocol,
    questions: tuple[str, ...],
    *,
    max_length: int | None = None,
) -> QuestionTokenBatch:
    if not questions or not all(question.strip() for question in questions):
        raise ValueError("questions must contain complete non-empty strings")
    kwargs: dict[str, object] = {
        "add_special_tokens": False,
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
        "truncation": False,
    }
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    raw = tokenizer(list(questions), **kwargs)
    if not isinstance(raw, Mapping):
        raise TypeError("tokenizer output must provide input_ids and attention_mask")
    values = cast(Mapping[str, object], raw)
    input_ids = cast(Tensor, values["input_ids"])
    attention_mask = cast(Tensor, values["attention_mask"])
    offset_mapping = cast(Tensor, values["offset_mapping"])
    if max_length is not None and bool(torch.any(attention_mask.sum(dim=1) > max_length)):
        raise ValueError("complete question exceeds max_length; truncation is forbidden")
    spans = tuple(
        QuestionTokenSpan(start=0, end=int(attention_mask[row].sum().item()))
        for row in range(attention_mask.shape[0])
    )
    return QuestionTokenBatch(
        questions=questions,
        input_ids=input_ids,
        attention_mask=attention_mask,
        padding_mask=attention_mask == 0,
        offset_mapping=offset_mapping,
        spans=spans,
        source_fields=("question",) * len(questions),
    )
