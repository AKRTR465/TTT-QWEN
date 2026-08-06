from __future__ import annotations

from collections.abc import Mapping

import torch

from ttt_svcbench_qwen.query_tokens import tokenize_questions


class FixtureTokenizer:
    def __call__(self, text: list[str], **kwargs: object) -> Mapping[str, torch.Tensor]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is True
        assert kwargs["return_tensors"] == "pt"
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
    assert batch.questions == ("当前画面有几架无人机？", "How many?")
    assert batch.input_ids[1].tolist() == [1, 2, 3, 0, 0, 0, 0]
    assert batch.padding_mask[0].sum().item() == 0
    assert batch.padding_mask[1].sum().item() == 4
    assert not batch.padding_mask[1, :3].any()
    assert torch.equal(batch.padding_mask, batch.attention_mask == 0)
