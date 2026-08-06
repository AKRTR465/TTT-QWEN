from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from ttt_svcbench_qwen.query_tokens import (
    QuestionTokenBatch,
    QuestionTokenSpan,
    tokenize_questions,
)


class FixtureTokenizer:
    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is True
        assert kwargs["return_tensors"] == "pt"
        assert kwargs["return_offsets_mapping"] is True
        lengths = [7 if question == "当前画面有几架无人机？" else 3 for question in text]
        width = max(lengths)
        input_ids = torch.zeros(len(text), width, dtype=torch.int64)
        attention_mask = torch.zeros_like(input_ids)
        offset_mapping = torch.zeros(len(text), width, 2, dtype=torch.int64)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
            offset_mapping[row, :length, 0] = torch.arange(length)
            offset_mapping[row, :length, 1] = torch.arange(1, length + 1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }


def test_complete_question_tokens_exclude_padding_and_keep_dynamic_length() -> None:
    batch = tokenize_questions(
        FixtureTokenizer(),
        ("当前画面有几架无人机？", "How many?"),
    )

    assert batch.input_ids.shape == (2, 7)
    assert tuple(span.end for span in batch.spans) == (7, 3)
    assert batch.questions == ("当前画面有几架无人机？", "How many?")
    assert batch.source_fields == ("question", "question")
    assert batch.padding_mask[0].sum().item() == 0
    assert batch.padding_mask[1].sum().item() == 4
    assert not batch.padding_mask[1, :3].any()
    assert batch.offset_mapping[1, 3:].eq(0).all()


def test_max_length_rejects_instead_of_slicing_a_question() -> None:
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_questions(
            FixtureTokenizer(),
            ("当前画面有几架无人机？",),
            max_length=6,
        )


def test_question_token_offsets_must_be_monotonic() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        QuestionTokenBatch(
            questions=("abc",),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.int64),
            padding_mask=torch.zeros(1, 3, dtype=torch.bool),
            offset_mapping=torch.tensor([[[0, 1], [2, 3], [1, 2]]]),
            spans=(QuestionTokenSpan(0, 3),),
            source_fields=("question",),
        )
