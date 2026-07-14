from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.input_composer import (
    EXACT_NUMBER_INSTRUCTION,
    TeacherForcedComposedInput,
    compose_inputs,
    compose_teacher_forced_inputs,
    map_teacher_forced_targets,
)
from ttt_svcbench_qwen.state_reader import ReaderStatus


class TinyTokenizer:
    def __init__(self) -> None:
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
            "answer-a": 10,
            "answer-b": 11,
            "<|vision_start|>": 12,
            "<|vision_end|>": 13,
            "instruction-a": 14,
            "instruction-b": 15,
        }
        self.pad_token_id = 0
        self.additional_special_tokens: list[str] = []

    def __len__(self) -> int:
        return len(self.tokens)

    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, object],
        replace_additional_special_tokens: bool = True,
    ) -> int:
        raw = special_tokens_dict["additional_special_tokens"]
        assert isinstance(raw, list)
        if replace_additional_special_tokens:
            self.additional_special_tokens = []
        added = 0
        for raw_token in raw:
            token = str(raw_token)
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
        return [14, 15]


class TinyEmbeddingOwner(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def resize_token_embeddings(self, new_num_tokens: int, **_kwargs: object) -> nn.Embedding:
        old = self.embedding
        replacement = nn.Embedding(new_num_tokens, old.embedding_dim)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[: old.num_embeddings].copy_(old.weight)
        self.embedding = replacement
        return replacement


class TinyRopeIndexer:
    def get_rope_index(
        self,
        input_ids: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del image_grid_thw, video_grid_thw
        assert input_ids is not None and attention_mask is not None
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        position_ids = positions.unsqueeze(0).expand(3, -1, -1).clone()
        max_positions = positions.max(dim=-1).values
        rope_deltas = (max_positions + 1 - input_ids.shape[1]).unsqueeze(1)
        return position_ids, rope_deltas


@dataclass(frozen=True)
class TinyReaderResult:
    status: ReaderStatus
    exact_count: int | None
    number_token_ids: tuple[int, ...]


def _teacher_forced_fixture() -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    tuple[TinyReaderResult, ...],
]:
    first = [3, 4, 5, 1, 3, 6, 7, 10, 1, 3, 4, 2, 5, 1, 3, 6, 7, 11, 8, 1]
    second = [3, 4, 2, 5, 1, 3, 6, 7, 8, 1]
    input_ids = torch.tensor([first, [0] * 10 + second], dtype=torch.int64)
    attention_mask = torch.tensor([[1] * 20, [0] * 10 + [1] * 10], dtype=torch.int64)
    labels = torch.full_like(input_ids, -100)
    labels[0, [7, 8, 17, 18, 19]] = input_ids[0, [7, 8, 17, 18, 19]]
    labels[1, [18, 19]] = input_ids[1, [18, 19]]
    number_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    number_mask[0, 18] = True
    number_mask[1, 18] = True
    readers = (
        TinyReaderResult(ReaderStatus.OK, 12, (8,)),
        TinyReaderResult(ReaderStatus.OK, 120, (8, 9)),
    )
    return input_ids, attention_mask, labels, number_mask, readers


def _compose_teacher_fixture() -> TeacherForcedComposedInput:
    input_ids, attention_mask, labels, number_mask, readers = _teacher_forced_fixture()
    tokenizer = TinyTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer))
    return compose_teacher_forced_inputs(
        base_input_ids=input_ids,
        base_attention_mask=attention_mask,
        base_labels=labels,
        base_number_token_mask=number_mask,
        state_tokens=torch.randn(2, 16, 8),
        state_token_valid_mask=torch.ones(2, dtype=torch.bool),
        reader_results=readers,
        tokenizer=tokenizer,
        embedding_owner=owner,
        rope_indexer=TinyRopeIndexer(),
        video_grid_thw=torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int64),
    )


def test_teacher_forcing_maps_multiturn_targets_payloads_and_left_padding() -> None:
    output = _compose_teacher_fixture()
    composed = output.composed_input

    assert composed.row_audits[0].insertion_index == 13
    assert composed.row_audits[1].insertion_index == 4
    assert composed.row_audits[0].inserted_token_count == 23
    assert composed.row_audits[1].inserted_token_count == 24
    assert composed.row_audits[0].left_padding == 0
    assert composed.row_audits[1].left_padding == 9
    assert output.row_audits[0].source_supervised_positions == (7, 8, 17, 18, 19)
    assert output.row_audits[0].composed_supervised_positions == (7, 8, 40, 41, 42)
    assert output.row_audits[1].source_supervised_positions == (18, 19)
    assert output.row_audits[1].composed_supervised_positions == (41, 42)
    assert output.row_audits[0].composed_number_positions == (41,)
    assert output.row_audits[1].composed_number_positions == (41,)
    assert output.number_token_mask.sum(dim=1).tolist() == [1, 1]
    assert not torch.equal(output.number_token_mask, composed.number_position_mask)
    assert not bool(torch.any(output.number_token_mask & composed.number_position_mask))

    source_origin = torch.zeros_like(output.labels, dtype=torch.bool)
    for row, audit in enumerate(output.row_audits):
        source_origin[row, list(audit.composed_source_positions)] = True
    assert torch.all(output.labels[~source_origin] == -100)
    assert torch.all(output.labels[~composed.attention_mask.bool()] == -100)
    assert torch.all(output.labels[composed.state_position_mask] == -100)
    assert torch.all(output.labels[composed.number_position_mask] == -100)
    supervised = output.labels != -100
    assert torch.equal(output.labels[supervised], composed.input_ids[supervised])


def test_low_level_mapper_rejects_source_token_provenance_drift() -> None:
    output = _compose_teacher_fixture()
    input_ids, attention_mask, labels, number_mask, _ = _teacher_forced_fixture()
    input_ids[0, 2] = 4

    with pytest.raises(ValueError, match="do not preserve source provenance"):
        map_teacher_forced_targets(
            composed_input=output.composed_input,
            source_input_ids=input_ids,
            source_attention_mask=attention_mask,
            source_labels=labels,
            source_number_token_mask=number_mask,
        )


def test_teacher_forcing_rejects_malicious_source_labels_and_number_masks() -> None:
    input_ids, attention_mask, labels, number_mask, readers = _teacher_forced_fixture()
    tokenizer = TinyTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer))
    common = {
        "base_input_ids": input_ids,
        "base_attention_mask": attention_mask,
        "state_tokens": torch.randn(2, 16, 8),
        "state_token_valid_mask": torch.ones(2, dtype=torch.bool),
        "reader_results": readers,
        "tokenizer": tokenizer,
        "embedding_owner": owner,
        "rope_indexer": TinyRopeIndexer(),
        "video_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int64),
    }

    context_number_mask = number_mask.clone()
    context_number_mask[0, 0] = True
    with pytest.raises(ValueError, match="subset of supervised source labels"):
        compose_teacher_forced_inputs(
            **common,
            base_labels=labels,
            base_number_token_mask=context_number_mask,
        )

    padded_label = labels.clone()
    padded_label[1, 0] = input_ids[1, 0]
    with pytest.raises(ValueError, match="outside source attention"):
        compose_teacher_forced_inputs(
            **common,
            base_labels=padded_label,
            base_number_token_mask=number_mask,
        )

    mismatched_label = labels.clone()
    mismatched_label[0, 7] = 11
    with pytest.raises(ValueError, match="equal their source token IDs"):
        compose_teacher_forced_inputs(
            **common,
            base_labels=mismatched_label,
            base_number_token_mask=number_mask,
        )


def test_reader_number_context_cannot_be_forged_into_answer_supervision() -> None:
    output = _compose_teacher_fixture()
    labels = output.labels.clone()
    reader_mask = output.composed_input.number_position_mask.clone()
    labels[reader_mask] = output.composed_input.input_ids[reader_mask]

    with pytest.raises(ValueError, match="Reader number context"):
        TeacherForcedComposedInput(
            composed_input=output.composed_input,
            labels=labels,
            number_token_mask=reader_mask,
            row_audits=output.row_audits,
        )


def test_plain_qwen_a1_accepts_empty_reader_without_payload_or_im_end() -> None:
    input_ids = torch.tensor([[3, 6, 7, 10], [0, 3, 6, 11]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.int64)
    labels = torch.tensor([[-100, -100, -100, 10], [-100, -100, -100, 11]])
    number_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    tokenizer = TinyTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer))

    output = compose_teacher_forced_inputs(
        base_input_ids=input_ids,
        base_attention_mask=attention_mask,
        base_labels=labels,
        base_number_token_mask=number_mask,
        state_tokens=None,
        state_token_valid_mask=None,
        reader_results=(),
        tokenizer=tokenizer,
        embedding_owner=owner,
        rope_indexer=TinyRopeIndexer(),
        video_grid_thw=None,
        include_state=False,
        include_number=False,
    )

    assert [audit.reader_status for audit in output.composed_input.row_audits] == [
        "disabled",
        "disabled",
    ]
    assert all(audit.insertion_index is None for audit in output.composed_input.row_audits)
    assert output.composed_input.state_position_mask.sum().item() == 0
    assert output.composed_input.number_position_mask.sum().item() == 0
    assert output.labels.tolist() == [[-100, -100, -100, 10], [-100, -100, -100, 11]]


def test_reader_alignment_remains_strict_outside_empty_plain_qwen_mode() -> None:
    input_ids = torch.tensor([[3, 4, 5, 1], [3, 4, 5, 1]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    tokenizer = TinyTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer))
    common = {
        "base_input_ids": input_ids,
        "base_attention_mask": attention_mask,
        "state_tokens": None,
        "state_token_valid_mask": None,
        "tokenizer": tokenizer,
        "embedding_owner": owner,
        "rope_indexer": TinyRopeIndexer(),
        "video_grid_thw": None,
    }

    with pytest.raises(ValueError, match="requires one Reader row per item"):
        compose_inputs(
            **common,
            reader_results=(),
            include_state=False,
            include_number=True,
        )
    with pytest.raises(ValueError, match="one row per batch item"):
        compose_inputs(
            **common,
            reader_results=(TinyReaderResult(ReaderStatus.UNSUPPORTED, None, ()),),
            include_state=False,
            include_number=False,
        )
