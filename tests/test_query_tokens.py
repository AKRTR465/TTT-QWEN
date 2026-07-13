from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from ttt_svcbench_qwen.query_tokens import tokenize_questions


class FixtureTokenizer:
    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        lengths = [7 if question == "当前画面有几架无人机？" else 3 for question in text]
        width = max(lengths)
        input_ids = torch.zeros(len(text), width, dtype=torch.int64)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_complete_question_tokens_exclude_padding_and_keep_dynamic_length() -> None:
    batch = tokenize_questions(
        FixtureTokenizer(),
        ("当前画面有几架无人机？", "How many?"),
    )

    assert batch.input_ids.shape == (2, 7)
    assert tuple(span.end for span in batch.spans) == (7, 3)
    assert batch.padding_mask[0].sum().item() == 0
    assert batch.padding_mask[1].sum().item() == 4
    assert not batch.padding_mask[1, :3].any()


def test_max_length_rejects_instead_of_slicing_a_question() -> None:
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_questions(
            FixtureTokenizer(),
            ("当前画面有几架无人机？",),
            max_length=6,
        )
