"""Tokenize only complete question text and expose padding-safe token spans.

Inputs: question strings and the tokenizer pinned by model ID/revision.
Outputs: integer input IDs, attention/padding masks, and per-question [start,end) spans.
Forbidden: system answers, assistant targets, labels, partial-question slicing, or fixed L_q.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor
from transformers import AutoTokenizer

from ttt_svcbench_qwen.config import ProjectConfig


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
    input_ids: Tensor
    attention_mask: Tensor
    padding_mask: Tensor
    spans: tuple[QuestionTokenSpan, ...]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("question input_ids must be integer [B, L_q]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("question attention_mask must match input_ids")
        if self.padding_mask.shape != self.input_ids.shape or self.padding_mask.dtype != torch.bool:
            raise ValueError("question padding_mask must be bool [B, L_q]")
        if len(self.spans) != self.input_ids.shape[0]:
            raise ValueError("question spans must contain one entry per batch item")
        expected_padding = self.attention_mask == 0
        if not torch.equal(self.padding_mask, expected_padding):
            raise ValueError("padding_mask must be exactly attention_mask == 0")
        for row, span in enumerate(self.spans):
            if span.end != int(self.attention_mask[row].sum().item()):
                raise ValueError("question span must exclude every padding token")


def load_pinned_tokenizer(
    config: ProjectConfig,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TokenizerProtocol:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        revision=config.model.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return cast(TokenizerProtocol, tokenizer)


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
    if max_length is not None and bool(torch.any(attention_mask.sum(dim=1) > max_length)):
        raise ValueError("complete question exceeds max_length; truncation is forbidden")
    spans = tuple(
        QuestionTokenSpan(start=0, end=int(attention_mask[row].sum().item()))
        for row in range(attention_mask.shape[0])
    )
    return QuestionTokenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        padding_mask=attention_mask == 0,
        spans=spans,
    )
