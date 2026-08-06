"""Shared dense-vocabulary stub tokenizer for the composer contract suites."""

from __future__ import annotations

from collections.abc import Sequence

from ttt_svcbench_qwen.input_composer import EXACT_NUMBER_INSTRUCTION


class StubTokenizer:
    def __init__(
        self,
        *,
        spare_tokens: tuple[str, str],
        preseeded_special_tokens: Sequence[str] = (),
    ) -> None:
        self.tokens = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            "<|video_pad|>": 2,
            "<|im_start|>": 3,
            "user": 4,
            "question": 5,
            "assistant": 6,
            "\n": 7,
            "12": 8,
            "0": 9,
            spare_tokens[0]: 10,
            spare_tokens[1]: 11,
            "<|vision_start|>": 12,
            "<|vision_end|>": 13,
            "instruction-a": 14,
            "instruction-b": 15,
        }
        self.pad_token_id = 0
        self.additional_special_tokens: list[str] = list(preseeded_special_tokens)
        self.registration_calls: list[tuple[tuple[str, ...], bool]] = []

    def __len__(self) -> int:
        return len(self.tokens)

    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, object],
        replace_additional_special_tokens: bool = True,
    ) -> int:
        raw = special_tokens_dict["additional_special_tokens"]
        assert isinstance(raw, list)
        values = tuple(str(value) for value in raw)
        self.registration_calls.append((values, replace_additional_special_tokens))
        if replace_additional_special_tokens:
            self.additional_special_tokens = []
        added = 0
        for token in values:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.tokens.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == EXACT_NUMBER_INSTRUCTION
        assert add_special_tokens is False
        return [self.tokens["instruction-a"], self.tokens["instruction-b"]]


def make_composer_tokenizer() -> StubTokenizer:
    return StubTokenizer(
        spare_tokens=("-3", "<|existing|>"),
        preseeded_special_tokens=("<|existing|>",),
    )


def make_stage_a_tokenizer() -> StubTokenizer:
    return StubTokenizer(spare_tokens=("answer-a", "answer-b"))
