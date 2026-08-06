"""Tokenize only complete question text and expose padding-safe token tensors.

Inputs: question strings and the tokenizer pinned by model ID/revision.
Outputs: integer input IDs plus attention and padding masks.
Forbidden: system answers, assistant targets, labels, partial-question slicing, or fixed L_q.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from torch import Tensor


class TokenizerProtocol(Protocol):
    def __call__(self, text: list[str], **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class QuestionTokenBatch:
    questions: tuple[str, ...]
    input_ids: Tensor
    attention_mask: Tensor
    padding_mask: Tensor


def tokenize_questions(
    tokenizer: TokenizerProtocol,
    questions: tuple[str, ...],
    *,
    max_length: int | None = None,
) -> QuestionTokenBatch:
    kwargs: dict[str, object] = {
        "add_special_tokens": False,
        "padding": True,
        "return_tensors": "pt",
        "truncation": False,
    }
    values = cast(Mapping[str, object], tokenizer(list(questions), **kwargs))
    input_ids = cast(Tensor, values["input_ids"])
    attention_mask = cast(Tensor, values["attention_mask"])
    return QuestionTokenBatch(
        questions=questions,
        input_ids=input_ids,
        attention_mask=attention_mask,
        padding_mask=attention_mask == 0,
    )
