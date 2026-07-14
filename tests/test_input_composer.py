from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.input_composer import (
    COMPOSER_SPECIAL_TOKENS,
    EXACT_NUMBER_INSTRUCTION,
    NUMBER_END_TOKEN,
    NUMBER_START_TOKEN,
    STATE_END_TOKEN,
    STATE_PAD_TOKEN,
    STATE_START_TOKEN,
    ComposedInput,
    compose_inputs,
    register_input_composer_tokens,
    register_input_composer_tokens_with_audit,
)
from ttt_svcbench_qwen.state_reader import ReaderStatus


class FakeTokenizer:
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
            "-3": 10,
            "<|existing|>": 11,
            "<|vision_start|>": 12,
            "<|vision_end|>": 13,
            "instruction-a": 14,
            "instruction-b": 15,
        }
        self.pad_token_id = 0
        self.additional_special_tokens = ["<|existing|>"]
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


class TinyEmbeddingOwner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        output_mode: str = "none",
    ) -> None:
        super().__init__()
        if output_mode not in {"none", "independent", "tied"}:
            raise ValueError("unknown fake output mode")
        self.output_mode = output_mode
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.embedding.weight.copy_(values.reshape(vocab_size, hidden_size) / 100.0)
        self.output = (
            nn.Linear(hidden_size, vocab_size, bias=False) if output_mode == "independent" else None
        )
        if self.output is not None:
            with torch.no_grad():
                values = torch.arange(vocab_size * hidden_size, dtype=torch.float32) + 1000
                self.output.weight.copy_(values.reshape(vocab_size, hidden_size) / 50.0)
        self.resize_calls: list[tuple[int, dict[str, object]]] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module | None:
        if self.output_mode == "tied":
            return self.embedding
        return self.output

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs: object) -> nn.Embedding:
        self.resize_calls.append((new_num_tokens, dict(kwargs)))
        old = self.embedding
        replacement = nn.Embedding(new_num_tokens, old.embedding_dim)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[: old.num_embeddings].copy_(old.weight)
        self.embedding = replacement
        if self.output_mode == "independent":
            assert self.output is not None
            old_output = self.output
            output = nn.Linear(old_output.in_features, new_num_tokens, bias=False)
            with torch.no_grad():
                output.weight.zero_()
                output.weight[: old_output.out_features].copy_(old_output.weight)
            self.output = output
        return replacement


class FakeRopeIndexer:
    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.calls: list[dict[str, Tensor | None]] = []

    def get_rope_index(
        self,
        input_ids: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert input_ids is not None
        assert attention_mask is not None
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": None
                if video_grid_thw is None
                else video_grid_thw.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
            }
        )
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        positions = positions.unsqueeze(0).expand(3, -1, -1).clone()
        if self.malformed:
            positions = positions[:2]
        max_positions = positions.max(dim=0).values.max(dim=-1).values
        deltas = (max_positions + 1 - input_ids.shape[1]).unsqueeze(1)
        return positions, deltas


@dataclass(frozen=True)
class FakeReaderResult:
    status: ReaderStatus | str
    exact_count: int | None
    number_token_ids: tuple[int, ...]


def _base_batch(batch_size: int = 2) -> tuple[Tensor, Tensor]:
    rows = (
        (3, 4, 2, 2, 5, 1, 6, 7),
        (3, 4, 2, 5, 1, 6, 7, 0),
        (3, 4, 2, 5, 1, 6, 7, 0),
        (3, 4, 2, 5, 1, 6, 7, 0),
    )[:batch_size]
    masks = (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 1, 1, 0),
    )[:batch_size]
    return torch.tensor(rows, dtype=torch.int64), torch.tensor(masks, dtype=torch.int64)


def _compose(
    *,
    statuses: tuple[ReaderStatus, ...] = (ReaderStatus.OK, ReaderStatus.UNSUPPORTED),
    malformed_rope: bool = False,
    state_tokens: Tensor | None = None,
    state_valid: Tensor | None = None,
    include_state: bool = True,
    include_number: bool = True,
) -> tuple[ComposedInput, FakeTokenizer, TinyEmbeddingOwner, FakeRopeIndexer, Tensor]:
    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(vocab_size=len(tokenizer), hidden_size=8)
    rope = FakeRopeIndexer(malformed=malformed_rope)
    input_ids, attention_mask = _base_batch(len(statuses))
    if state_tokens is None:
        state_tokens = torch.arange(
            len(statuses) * 16 * 8,
            dtype=torch.float32,
        ).reshape(len(statuses), 16, 8)
    if state_valid is None:
        state_valid = torch.tensor(
            [status in (ReaderStatus.OK, ReaderStatus.EMPTY) for status in statuses],
            dtype=torch.bool,
        )
    number_by_status = {
        ReaderStatus.OK: (12, (8,)),
        ReaderStatus.EMPTY: (0, (9,)),
        ReaderStatus.UNSUPPORTED: (None, ()),
        ReaderStatus.INVALID: (None, ()),
    }
    results = tuple(FakeReaderResult(status, *number_by_status[status]) for status in statuses)
    composed = compose_inputs(
        base_input_ids=input_ids,
        base_attention_mask=attention_mask,
        state_tokens=state_tokens,
        state_token_valid_mask=state_valid,
        reader_results=results,
        tokenizer=tokenizer,
        embedding_owner=owner,
        rope_indexer=rope,
        video_grid_thw=torch.tensor([[1, 2, 2]] * len(statuses), dtype=torch.int64),
        include_state=include_state,
        include_number=include_number,
    )
    return composed, tokenizer, owner, rope, state_tokens


def test_special_token_registration_initializes_independent_input_and_output_once() -> None:
    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(
        vocab_size=len(tokenizer),
        hidden_size=8,
        output_mode="independent",
    )
    input_before = owner.embedding.weight.detach().clone()
    assert owner.output is not None
    output_before = owner.output.weight.detach().clone()
    expected_input = input_before[[12, 2, 13]].float().mean(dim=0)
    expected_output = output_before[[12, 2, 13]].float().mean(dim=0)

    first = register_input_composer_tokens_with_audit(tokenizer, owner)

    assert COMPOSER_SPECIAL_TOKENS == (
        STATE_START_TOKEN,
        STATE_PAD_TOKEN,
        STATE_END_TOKEN,
        NUMBER_START_TOKEN,
        NUMBER_END_TOKEN,
    )
    assert first.token_ids.composer_ids == (16, 17, 18, 19, 20)
    assert owner.embedding.num_embeddings == 21
    assert owner.output is not None and owner.output.out_features == 21
    assert owner.resize_calls == [(21, {"mean_resizing": False})]
    assert torch.equal(owner.embedding.weight[:16], input_before)
    assert torch.equal(owner.output.weight[:16], output_before)
    for token_id in first.token_ids.composer_ids:
        assert torch.equal(owner.embedding.weight[token_id], expected_input)
        assert torch.equal(owner.output.weight[token_id], expected_output)
    assert first.audit.base_tokenizer_length == 16
    assert first.audit.new_tokenizer_length == 21
    assert first.audit.added_token_count == 5
    assert first.audit.initialized_input_ids == first.token_ids.composer_ids
    assert first.audit.initialized_output_ids == first.token_ids.composer_ids
    assert first.audit.output_embedding_tied is False
    assert tokenizer.additional_special_tokens[0] == "<|existing|>"
    assert tokenizer.additional_special_tokens[1:] == list(COMPOSER_SPECIAL_TOKENS)

    with torch.no_grad():
        rows = torch.tensor(first.token_ids.composer_ids)
        owner.embedding.weight.index_fill_(0, rows, 77.0)
        owner.output.weight.index_fill_(0, rows, 88.0)
    second = register_input_composer_tokens_with_audit(tokenizer, owner)
    assert second.token_ids == first.token_ids
    assert second.audit.added_token_count == 0
    assert second.audit.initialized_input_ids == ()
    assert second.audit.initialized_output_ids == ()
    assert torch.all(owner.embedding.weight[list(first.token_ids.composer_ids)] == 77.0)
    assert torch.all(owner.output.weight[list(first.token_ids.composer_ids)] == 88.0)
    assert owner.resize_calls == [(21, {"mean_resizing": False})]


def test_tied_output_is_initialized_once_and_registration_never_shrinks() -> None:
    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer), 8, output_mode="tied")
    before = owner.embedding.weight.detach().clone()
    expected = before[[12, 2, 13]].float().mean(dim=0)

    registration = register_input_composer_tokens_with_audit(tokenizer, owner)

    assert registration.audit.output_embedding_tied is True
    assert registration.audit.initialized_input_ids == registration.token_ids.composer_ids
    assert registration.audit.initialized_output_ids == ()
    assert owner.get_output_embeddings() is owner.get_input_embeddings()
    for token_id in registration.token_ids.composer_ids:
        assert torch.equal(owner.embedding.weight[token_id], expected)


def test_oversized_151936_embedding_is_not_resized_but_new_token_rows_are_initialized() -> None:
    oversized = TinyEmbeddingOwner(vocab_size=151_936, hidden_size=4)
    other_tokenizer = FakeTokenizer()
    base_rows = oversized.embedding.weight[: len(other_tokenizer)].detach().clone()
    last_row = oversized.embedding.weight[-1].detach().clone()
    expected = base_rows[[12, 2, 13]].float().mean(dim=0)

    registration = register_input_composer_tokens_with_audit(other_tokenizer, oversized)

    assert oversized.embedding.num_embeddings == 151_936
    assert oversized.resize_calls == []
    assert torch.equal(oversized.embedding.weight[:16], base_rows)
    assert torch.equal(oversized.embedding.weight[-1], last_row)
    for token_id in registration.token_ids.composer_ids:
        assert torch.equal(oversized.embedding.weight[token_id], expected)


def test_checkpoint_and_extended_tokenizer_reload_never_resets_trained_rows() -> None:
    tokenizer = FakeTokenizer()
    seed_owner = TinyEmbeddingOwner(len(tokenizer), 8, output_mode="independent")
    token_ids = register_input_composer_tokens(tokenizer, seed_owner)
    reloaded = TinyEmbeddingOwner(len(tokenizer), 8, output_mode="independent")
    assert reloaded.output is not None
    with torch.no_grad():
        rows = torch.tensor(token_ids.composer_ids)
        reloaded.embedding.weight.index_fill_(0, rows, 123.0)
        reloaded.output.weight.index_fill_(0, rows, 456.0)

    audit = register_input_composer_tokens_with_audit(tokenizer, reloaded).audit

    assert audit.added_token_count == 0
    assert audit.initialized_input_ids == audit.initialized_output_ids == ()
    assert reloaded.resize_calls == []
    assert torch.all(reloaded.embedding.weight[list(token_ids.composer_ids)] == 123.0)
    assert torch.all(reloaded.output.weight[list(token_ids.composer_ids)] == 456.0)


def test_compose_inserts_before_final_user_end_left_pads_and_scatter_state_only() -> None:
    composed, tokenizer, owner, rope, state_tokens = _compose()
    ids = composed.special_token_ids

    assert composed.inputs_embeds.shape == (2, 31, 8)
    assert composed.input_ids.shape == composed.attention_mask.shape == (2, 31)
    assert composed.position_ids.shape == (3, 2, 31)
    assert composed.rope_deltas.shape == (2, 1)
    assert torch.equal(composed.cache_position, torch.arange(31))
    assert composed.row_audits[0].left_padding == 0
    assert composed.row_audits[1].left_padding == 24
    assert composed.row_audits[0].insertion_index == 5
    assert composed.row_audits[1].insertion_index is None
    assert composed.row_audits[0].inserted_token_count == 23
    assert composed.row_audits[1].inserted_token_count == 0

    first_valid = composed.input_ids[0, composed.attention_mask[0].bool()].tolist()
    expected_payload = [
        ids.state_start,
        *([ids.state_pad] * 16),
        ids.state_end,
        14,
        15,
        ids.number_start,
        8,
        ids.number_end,
    ]
    assert first_valid == [3, 4, 2, 2, 5, *expected_payload, 1, 6, 7]
    second_valid = composed.input_ids[1, composed.attention_mask[1].bool()].tolist()
    assert second_valid == [3, 4, 2, 5, 1, 6, 7]

    assert composed.state_position_mask.sum(dim=1).tolist() == [16, 0]
    assert composed.number_position_mask.sum(dim=1).tolist() == [1, 0]
    assert composed.video_position_mask.sum(dim=1).tolist() == [2, 1]
    assert not bool(
        torch.any(composed.video_position_mask & composed.state_position_mask)
        or torch.any(composed.video_position_mask & composed.number_position_mask)
        or torch.any(composed.state_position_mask & composed.number_position_mask)
    )
    assert torch.equal(
        composed.inputs_embeds[0, composed.state_position_mask[0]],
        state_tokens[0],
    )
    assert torch.equal(
        composed.inputs_embeds[0, composed.video_position_mask[0]],
        owner.get_input_embeddings()(composed.input_ids[0, composed.video_position_mask[0]]),
    )
    assert composed.number_token_ids == ((8,), ())
    assert composed.row_audits[0].instruction_token_ids == (14, 15)
    assert len(composed.row_audits[0].instruction_positions) == 2
    assert composed.input_ids[0, composed.number_position_mask[0]].tolist() == [8]
    assert tokenizer.convert_tokens_to_ids(STATE_PAD_TOKEN) == ids.state_pad

    assert len(rope.calls) == 1
    assert torch.equal(rope.calls[0]["input_ids"], composed.input_ids)
    assert torch.equal(rope.calls[0]["attention_mask"], composed.attention_mask)
    assert rope.calls[0]["image_grid_thw"] is None
    assert torch.equal(
        rope.calls[0]["video_grid_thw"],
        torch.tensor([[1, 2, 2], [1, 2, 2]]),
    )


def test_empty_injects_zero_number_but_unknown_statuses_inject_nothing() -> None:
    statuses = (
        ReaderStatus.OK,
        ReaderStatus.EMPTY,
        ReaderStatus.UNSUPPORTED,
        ReaderStatus.INVALID,
    )
    composed, _, _, _, _ = _compose(statuses=statuses)

    assert composed.state_position_mask.sum(dim=1).tolist() == [16, 16, 0, 0]
    assert composed.number_token_ids == ((8,), (9,), (), ())
    assert [audit.reader_status for audit in composed.row_audits] == [
        "ok",
        "empty",
        "unsupported",
        "invalid",
    ]
    assert [audit.exact_count for audit in composed.row_audits] == [12, 0, None, None]
    for row in (2, 3):
        valid_ids = composed.input_ids[row, composed.attention_mask[row].bool()].tolist()
        assert valid_ids == [3, 4, 2, 5, 1, 6, 7]


def test_state_and_number_ablation_payloads_are_independent() -> None:
    state_only, _, _, _, _ = _compose(include_number=False)
    assert state_only.state_position_mask.sum(dim=1).tolist() == [16, 0]
    assert state_only.number_position_mask.sum().item() == 0
    assert state_only.number_token_ids == ((), ())
    assert state_only.row_audits[0].state_included is True
    assert state_only.row_audits[0].number_included is False
    assert state_only.row_audits[0].instruction_token_ids == ()

    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(len(tokenizer), 8)
    input_ids, attention_mask = _base_batch()
    number_only = compose_inputs(
        base_input_ids=input_ids,
        base_attention_mask=attention_mask,
        state_tokens=None,
        state_token_valid_mask=None,
        reader_results=(
            FakeReaderResult(ReaderStatus.OK, 12, (8,)),
            FakeReaderResult(ReaderStatus.UNSUPPORTED, None, ()),
        ),
        tokenizer=tokenizer,
        embedding_owner=owner,
        rope_indexer=FakeRopeIndexer(),
        video_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        include_state=False,
        include_number=True,
    )
    assert number_only.state_position_mask.sum().item() == 0
    assert number_only.number_token_ids == ((8,), ())
    assert number_only.row_audits[0].state_included is False
    assert number_only.row_audits[0].number_included is True
    assert number_only.row_audits[0].instruction_token_ids == (14, 15)


def test_state_scatter_preserves_gradient_only_for_injected_rows() -> None:
    state_tokens = torch.randn(2, 16, 8, requires_grad=True)
    composed, _, _, _, _ = _compose(state_tokens=state_tokens)

    composed.inputs_embeds[composed.state_position_mask].sum().backward()

    assert state_tokens.grad is not None
    assert torch.equal(state_tokens.grad[0], torch.ones_like(state_tokens.grad[0]))
    assert torch.equal(state_tokens.grad[1], torch.zeros_like(state_tokens.grad[1]))


@pytest.mark.parametrize(
    ("statuses", "state_valid", "message"),
    [
        (
            (ReaderStatus.OK, ReaderStatus.UNSUPPORTED),
            torch.tensor([False, False]),
            "require valid State Tokens",
        ),
        (
            (ReaderStatus.UNSUPPORTED, ReaderStatus.INVALID),
            torch.tensor([True, False]),
            "cannot inject State or number payload",
        ),
    ],
)
def test_status_and_state_validity_mismatch_fails_closed(
    statuses: tuple[ReaderStatus, ...],
    state_valid: Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _compose(statuses=statuses, state_valid=state_valid.bool())


def test_missing_user_end_recomposition_and_wrong_state_shape_fail_closed() -> None:
    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(vocab_size=len(tokenizer), hidden_size=8)
    rope = FakeRopeIndexer()
    input_ids, attention_mask = _base_batch()
    input_ids[0, 5] = 5
    results = (
        FakeReaderResult(ReaderStatus.OK, 12, (8,)),
        FakeReaderResult(ReaderStatus.UNSUPPORTED, None, ()),
    )
    state_tokens = torch.zeros(2, 16, 8)
    state_valid = torch.tensor([True, False])
    kwargs: dict[str, Any] = {
        "base_input_ids": input_ids,
        "base_attention_mask": attention_mask,
        "state_tokens": state_tokens,
        "state_token_valid_mask": state_valid,
        "reader_results": results,
        "tokenizer": tokenizer,
        "embedding_owner": owner,
        "rope_indexer": rope,
        "video_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    with pytest.raises(ValueError, match="final user"):
        compose_inputs(**kwargs)

    valid_ids, valid_mask = _base_batch()
    token_ids = register_input_composer_tokens(tokenizer, owner)
    valid_ids[0, 4] = token_ids.state_start
    kwargs.update(base_input_ids=valid_ids, base_attention_mask=valid_mask)
    with pytest.raises(ValueError, match="already contains Composer"):
        compose_inputs(**kwargs)

    kwargs.update(base_input_ids=_base_batch()[0], state_tokens=torch.zeros(2, 15, 8))
    with pytest.raises(ValueError, match=r"\[B, 16, H\]"):
        compose_inputs(**kwargs)


def test_hidden_size_and_native_mrope_shapes_are_strict() -> None:
    with pytest.raises(ValueError, match="hidden size"):
        _compose(state_tokens=torch.zeros(2, 16, 9))
    with pytest.raises(ValueError, match=r"\[3, B, L\]"):
        _compose(malformed_rope=True)


def test_binary_attention_and_original_reader_number_ids_are_strict() -> None:
    tokenizer = FakeTokenizer()
    owner = TinyEmbeddingOwner(vocab_size=len(tokenizer), hidden_size=8)
    input_ids, attention = _base_batch()
    attention[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        compose_inputs(
            base_input_ids=input_ids,
            base_attention_mask=attention,
            state_tokens=torch.zeros(2, 16, 8),
            state_token_valid_mask=torch.tensor([True, False]),
            reader_results=(
                FakeReaderResult(ReaderStatus.OK, 12, (8,)),
                FakeReaderResult(ReaderStatus.INVALID, None, ()),
            ),
            tokenizer=tokenizer,
            embedding_owner=owner,
            rope_indexer=FakeRopeIndexer(),
            video_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        )

    control_token_number = FakeTokenizer().convert_tokens_to_ids("<|video_pad|>")
    assert control_token_number is not None
    with pytest.raises(ValueError, match="control tokens"):
        compose_inputs(
            base_input_ids=_base_batch()[0],
            base_attention_mask=_base_batch()[1],
            state_tokens=torch.zeros(2, 16, 8),
            state_token_valid_mask=torch.tensor([True, False]),
            reader_results=(
                FakeReaderResult(ReaderStatus.OK, 12, (control_token_number,)),
                FakeReaderResult(ReaderStatus.INVALID, None, ()),
            ),
            tokenizer=FakeTokenizer(),
            embedding_owner=TinyEmbeddingOwner(12, 8),
            rope_indexer=FakeRopeIndexer(),
            video_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        )
