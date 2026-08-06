"""Compose one State-TTT Qwen prefill without touching runtime state.

Inputs are native Qwen chat-template IDs, 16 optional State Tokens, and the exact
number IDs already produced by the Deterministic Reader.  Video placeholders are
left untouched for Qwen's native video ``masked_scatter``/DeepStack path; only the
State placeholders are scattered here.

This module never updates the Bank, fast weights, Reader arithmetic, or decode
state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.state_reader import ReaderStatus

STATE_START_TOKEN = "<|state_start|>"
STATE_PAD_TOKEN = "<|state_pad|>"
STATE_END_TOKEN = "<|state_end|>"
NUMBER_START_TOKEN = "<|number_start|>"
NUMBER_END_TOKEN = "<|number_end|>"

COMPOSER_SPECIAL_TOKENS = (
    STATE_START_TOKEN,
    STATE_PAD_TOKEN,
    STATE_END_TOKEN,
    NUMBER_START_TOKEN,
    NUMBER_END_TOKEN,
)
STATE_TOKEN_COUNT = 16
IM_END_TOKEN = "<|im_end|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
EXACT_NUMBER_INSTRUCTION = "\nUse the exact number provided below; do not recount or override it.\n"
IGNORE_INDEX = -100

_REGISTRATION_LOCK = RLock()
_COUNT_BEARING_STATUSES = frozenset((ReaderStatus.OK.value, ReaderStatus.EMPTY.value))
_DISABLED_READER_STATUS = "disabled"


class ComposerTokenizer(Protocol):
    pad_token_id: int | None

    def __len__(self) -> int: ...

    def add_special_tokens(
        self,
        special_tokens_dict: Mapping[str, object],
        replace_additional_special_tokens: bool = True,
    ) -> int: ...

    def convert_tokens_to_ids(self, token: str) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


class EmbeddingOwner(Protocol):
    def get_input_embeddings(self) -> object: ...

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs: object) -> object: ...


class ReaderResultLike(Protocol):
    @property
    def status(self) -> ReaderStatus | str: ...

    @property
    def exact_count(self) -> int | None: ...

    @property
    def number_token_ids(self) -> tuple[int, ...]: ...


class RopeIndexer(Protocol):
    def get_rope_index(
        self,
        input_ids: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


RopeIndexCallable = Callable[..., tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class ComposerSpecialTokenIds:
    state_start: int
    state_pad: int
    state_end: int
    number_start: int
    number_end: int
    im_end: int
    vision_start: int
    video_pad: int
    vision_end: int
    pad: int

    @property
    def composer_ids(self) -> tuple[int, int, int, int, int]:
        return (
            self.state_start,
            self.state_pad,
            self.state_end,
            self.number_start,
            self.number_end,
        )

    @property
    def initialization_source_ids(self) -> tuple[int, int, int]:
        return (self.vision_start, self.video_pad, self.vision_end)


@dataclass(frozen=True, slots=True)
class CompositionRowAudit:
    """Source/composed token arithmetic for one row, consumed by target mapping."""

    source_token_count: int
    composed_token_count: int
    inserted_token_count: int
    insertion_index: int | None
    state_included: bool


@dataclass(frozen=True, slots=True)
class ComposedInput:
    """One left-padded prefill and every placement tensor needed by Qwen."""

    input_ids: Tensor
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    rope_deltas: Tensor
    state_position_mask: Tensor
    row_audits: tuple[CompositionRowAudit, ...]


@dataclass(frozen=True, slots=True)
class TeacherForcedComposedInput:
    """A composed Qwen input plus answer-only labels and their number-token subset."""

    composed_input: ComposedInput
    labels: Tensor
    number_token_mask: Tensor


def register_input_composer_tokens_with_audit(
    tokenizer: ComposerTokenizer,
    embedding_owner: EmbeddingOwner,
) -> ComposerSpecialTokenIds:
    """Register once, grow without shrinking, and deterministically initialize new rows.

    Only IDs added by this call are initialized.  An already-extended tokenizer therefore
    represents a checkpoint reload and leaves all learned input/output rows byte-for-byte alone.
    """

    with _REGISTRATION_LOCK:
        base_tokenizer_length = len(tokenizer)
        input_before = _embedding_layer(embedding_owner)
        input_size_before = _embedding_size(input_before)
        added_count = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(COMPOSER_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        new_tokenizer_length = len(tokenizer)
        special_ids = _resolve_special_token_ids(tokenizer)
        new_ids = tuple(
            token_id for token_id in special_ids.composer_ids if token_id >= base_tokenizer_length
        )
        if added_count == 0:
            new_ids = ()
        target_size = max(
            input_size_before,
            new_tokenizer_length,
            max(
                special_ids.composer_ids
                + (
                    special_ids.im_end,
                    special_ids.vision_start,
                    special_ids.video_pad,
                    special_ids.vision_end,
                    special_ids.pad,
                )
            )
            + 1,
        )
        if target_size > input_size_before:
            try:
                embedding_owner.resize_token_embeddings(target_size, mean_resizing=False)
            except TypeError:
                embedding_owner.resize_token_embeddings(target_size)
        input_after = _embedding_layer(embedding_owner)
        output_after = _optional_output_embedding_layer(embedding_owner)
        output_tied = output_after is not None and _embedding_weights_share_storage(
            input_after,
            output_after,
        )
        if new_ids:
            _initialize_embedding_rows(
                input_after,
                new_ids,
                special_ids.initialization_source_ids,
            )
            if output_after is not None and not output_tied:
                _initialize_embedding_rows(
                    output_after,
                    new_ids,
                    special_ids.initialization_source_ids,
                )
        return special_ids


def compose_inputs(
    *,
    base_input_ids: Tensor,
    base_attention_mask: Tensor,
    state_tokens: Tensor | None,
    state_token_valid_mask: Tensor | None,
    reader_results: Sequence[ReaderResultLike],
    tokenizer: ComposerTokenizer,
    embedding_owner: EmbeddingOwner,
    rope_indexer: RopeIndexer | RopeIndexCallable,
    video_grid_thw: Tensor | None,
    include_state: bool = True,
    include_number: bool = True,
) -> ComposedInput:
    """Insert state/number segments and build one native-Qwen prefill.

    ``base_input_ids`` must be the native processor/chat-template prefill ending in
    the assistant generation prefix.  For each count-bearing Reader row, the new
    payload is inserted immediately before the final user ``<|im_end|>``.  The
    returned IDs retain native video placeholders, while ``inputs_embeds`` has only
    the 16 State placeholders replaced.
    """

    special_ids = register_input_composer_tokens_with_audit(tokenizer, embedding_owner)
    batch_size = int(base_input_ids.shape[0])
    device = _embedding_device(_embedding_layer(embedding_owner), base_input_ids.device)
    source_ids = base_input_ids.to(device=device, dtype=torch.int64)
    source_mask = base_attention_mask.to(device=device).bool()
    state_values = None if state_tokens is None else state_tokens.to(device=device)
    state_valid = (
        None if state_token_valid_mask is None else state_token_valid_mask.to(device=device)
    )

    row_ids: list[list[int]] = []
    row_origins: list[list[str]] = []
    row_metadata: list[tuple[int, int | None, bool]] = []
    instruction_ids = _encode_exact_number_instruction(tokenizer) if include_number else ()
    for row in range(batch_size):
        valid_ids = [int(value) for value in source_ids[row, source_mask[row]].tolist()]
        if len(reader_results) != 0:
            status = _reader_status(reader_results[row])
            number_ids = tuple(reader_results[row].number_token_ids)
        else:
            status = _DISABLED_READER_STATUS
            number_ids = ()
        base_origins = ["base"] * len(valid_ids)
        state_included = status in _COUNT_BEARING_STATUSES and include_state
        number_included = status in _COUNT_BEARING_STATUSES and include_number
        inserted_number_ids = number_ids if number_included else ()
        inserted_instruction_ids = instruction_ids if number_included else ()
        if state_included or number_included:
            im_end_positions = [
                index for index, token_id in enumerate(valid_ids) if token_id == special_ids.im_end
            ]
            insertion_index = im_end_positions[-1]
            state_payload = (
                [
                    special_ids.state_start,
                    *([special_ids.state_pad] * STATE_TOKEN_COUNT),
                    special_ids.state_end,
                ]
                if state_included
                else []
            )
            state_origins = (
                ["boundary", *(["state"] * STATE_TOKEN_COUNT), "boundary"] if state_included else []
            )
            number_payload = (
                [
                    *inserted_instruction_ids,
                    special_ids.number_start,
                    *inserted_number_ids,
                    special_ids.number_end,
                ]
                if number_included
                else []
            )
            number_origins = (
                [
                    *(["instruction"] * len(inserted_instruction_ids)),
                    "boundary",
                    *(["number"] * len(inserted_number_ids)),
                    "boundary",
                ]
                if number_included
                else []
            )
            payload = state_payload + number_payload
            origins = state_origins + number_origins
            composed_ids = valid_ids[:insertion_index] + payload + valid_ids[insertion_index:]
            composed_origins = (
                base_origins[:insertion_index] + origins + base_origins[insertion_index:]
            )
        else:
            insertion_index = None
            composed_ids = valid_ids
            composed_origins = base_origins
        row_ids.append(composed_ids)
        row_origins.append(composed_origins)
        row_metadata.append((len(valid_ids), insertion_index, state_included))

    max_length = max(len(values) for values in row_ids)
    input_ids = torch.full(
        (batch_size, max_length),
        special_ids.pad,
        dtype=torch.int64,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.int64,
        device=device,
    )
    state_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    audits: list[CompositionRowAudit] = []
    for row, (ids, origins, metadata) in enumerate(
        zip(row_ids, row_origins, row_metadata, strict=True)
    ):
        width = len(ids)
        left_padding = max_length - width
        input_ids[row, left_padding:] = torch.tensor(ids, dtype=torch.int64, device=device)
        attention_mask[row, left_padding:] = 1
        for column, origin in enumerate(origins, start=left_padding):
            if origin == "state":
                state_mask[row, column] = True
        source_count, insertion_index, state_included = metadata
        audits.append(
            CompositionRowAudit(
                source_token_count=source_count,
                composed_token_count=width,
                inserted_token_count=width - source_count,
                insertion_index=insertion_index,
                state_included=state_included,
            )
        )

    embedding = _embedding_layer(embedding_owner)
    embedded = cast(Tensor, embedding(input_ids))
    inputs_embeds = embedded.clone()
    if include_state and state_values is not None and state_valid is not None:
        state_values = state_values.to(dtype=embedded.dtype)
        for row in range(batch_size):
            if audits[row].state_included and bool(state_valid[row].item()):
                positions = torch.nonzero(state_mask[row]).flatten()
                inputs_embeds[row, positions] = state_values[row]

    position_ids, rope_deltas = _call_get_rope_index(
        rope_indexer,
        input_ids=input_ids,
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device=device),
        attention_mask=attention_mask,
    )
    position_ids = position_ids.to(device=device)
    rope_deltas = rope_deltas.to(device=device)
    return ComposedInput(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        state_position_mask=state_mask,
        row_audits=tuple(audits),
    )


def map_teacher_forced_targets(
    *,
    composed_input: ComposedInput,
    source_input_ids: Tensor,
    source_attention_mask: Tensor,
    source_labels: Tensor,
    source_number_token_mask: Tensor,
) -> TeacherForcedComposedInput:
    """Map source-aligned answer labels through row-specific payload insertion/padding."""

    batch_size = int(composed_input.input_ids.shape[0])
    device = composed_input.input_ids.device
    source_attention = source_attention_mask.to(device=device).bool()
    source_targets = source_labels.to(device=device, dtype=torch.int64)
    source_numbers = source_number_token_mask.to(device=device)
    labels = torch.full_like(composed_input.input_ids, IGNORE_INDEX)
    number_token_mask = torch.zeros_like(composed_input.input_ids, dtype=torch.bool)

    for row in range(batch_size):
        composition_audit = composed_input.row_audits[row]
        source_valid_positions_tensor = torch.nonzero(source_attention[row]).flatten()
        source_count = int(source_valid_positions_tensor.numel())
        composed_valid_positions = torch.nonzero(
            composed_input.attention_mask[row].bool()
        ).flatten()

        insertion_index = composition_audit.insertion_index
        inserted_count = composition_audit.inserted_token_count
        if insertion_index is None:
            relative_source_positions = torch.arange(source_count, dtype=torch.int64, device=device)
        else:
            before = torch.arange(insertion_index, dtype=torch.int64, device=device)
            after = torch.arange(
                insertion_index + inserted_count,
                source_count + inserted_count,
                dtype=torch.int64,
                device=device,
            )
            relative_source_positions = torch.cat((before, after))
        composed_source_positions_tensor = composed_valid_positions.index_select(
            0, relative_source_positions
        )

        valid_targets = source_targets[row].index_select(0, source_valid_positions_tensor)
        valid_number_mask = source_numbers[row].index_select(0, source_valid_positions_tensor)
        labels[row].index_copy_(0, composed_source_positions_tensor, valid_targets)
        number_token_mask[row].index_copy_(0, composed_source_positions_tensor, valid_number_mask)

    return TeacherForcedComposedInput(
        composed_input=composed_input,
        labels=labels,
        number_token_mask=number_token_mask,
    )


def _resolve_special_token_ids(tokenizer: ComposerTokenizer) -> ComposerSpecialTokenIds:
    values = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in (
            *COMPOSER_SPECIAL_TOKENS,
            IM_END_TOKEN,
            VISION_START_TOKEN,
            VIDEO_PAD_TOKEN,
            VISION_END_TOKEN,
        )
    }
    pad_id = tokenizer.pad_token_id
    return ComposerSpecialTokenIds(
        state_start=cast(int, values[STATE_START_TOKEN]),
        state_pad=cast(int, values[STATE_PAD_TOKEN]),
        state_end=cast(int, values[STATE_END_TOKEN]),
        number_start=cast(int, values[NUMBER_START_TOKEN]),
        number_end=cast(int, values[NUMBER_END_TOKEN]),
        im_end=cast(int, values[IM_END_TOKEN]),
        vision_start=cast(int, values[VISION_START_TOKEN]),
        video_pad=cast(int, values[VIDEO_PAD_TOKEN]),
        vision_end=cast(int, values[VISION_END_TOKEN]),
        pad=cast(int, pad_id),
    )


def _embedding_layer(owner: EmbeddingOwner) -> nn.Module:
    return cast(nn.Module, owner.get_input_embeddings())


def _optional_output_embedding_layer(owner: EmbeddingOwner) -> nn.Module | None:
    getter = getattr(owner, "get_output_embeddings", None)
    if not callable(getter):
        return None
    embedding = getter()
    if embedding is None:
        return None
    return cast(nn.Module, embedding)


def _embedding_weight(embedding: nn.Module) -> Tensor:
    return cast(Tensor, embedding.weight)


def _embedding_size(embedding: nn.Module) -> int:
    return int(cast(nn.Embedding, embedding).num_embeddings)


def _embedding_weights_share_storage(left: nn.Module, right: nn.Module) -> bool:
    return _embedding_weight(left) is _embedding_weight(right)


def _initialize_embedding_rows(
    embedding: nn.Module,
    target_ids: tuple[int, ...],
    source_ids: tuple[int, int, int],
) -> None:
    weight = _embedding_weight(embedding)
    source_index = torch.tensor(source_ids, dtype=torch.int64, device=weight.device)
    target_index = torch.tensor(target_ids, dtype=torch.int64, device=weight.device)
    source_mean = weight.detach().index_select(0, source_index).float().mean(dim=0)
    initialized = source_mean.to(dtype=weight.dtype).expand(len(target_ids), -1)
    with torch.no_grad():
        weight.index_copy_(0, target_index, initialized)


def _embedding_device(embedding: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(embedding.parameters(), None)
    return parameter.device if parameter is not None else fallback


def _reader_status(result: ReaderResultLike) -> str:
    status = result.status
    return status.value if isinstance(status, ReaderStatus) else status


def _encode_exact_number_instruction(tokenizer: ComposerTokenizer) -> tuple[int, ...]:
    return tuple(tokenizer.encode(EXACT_NUMBER_INSTRUCTION, add_special_tokens=False))


def _call_get_rope_index(
    rope_indexer: RopeIndexer | RopeIndexCallable,
    *,
    input_ids: Tensor,
    video_grid_thw: Tensor | None,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    method = getattr(rope_indexer, "get_rope_index", None)
    callable_indexer = cast(RopeIndexCallable, method if callable(method) else rope_indexer)
    position_ids, rope_deltas = callable_indexer(
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    return position_ids, rope_deltas
