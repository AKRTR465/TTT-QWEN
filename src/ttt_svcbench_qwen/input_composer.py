"""Compose one audited State-TTT Qwen prefill without touching runtime state.

Inputs are native Qwen chat-template IDs, 16 optional State Tokens, and the exact
number IDs already produced by the Deterministic Reader.  Video placeholders are
left untouched for Qwen's native video ``masked_scatter``/DeepStack path; only the
State placeholders are scattered here.

This module never updates the Bank, fast weights, Reader arithmetic, or decode
state.  Calling code must perform the P12 Retriever/Reader provenance audit before
handing Reader results to this boundary.
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
TOKEN_INITIALIZATION_STRATEGY = "fp32_mean_of_vision_start_video_pad_vision_end_then_cast"
EXACT_NUMBER_INSTRUCTION = "\nUse the exact number provided below; do not recount or override it.\n"
IGNORE_INDEX = -100

_REGISTRATION_LOCK = RLock()
_COUNT_BEARING_STATUSES = frozenset((ReaderStatus.OK.value, ReaderStatus.EMPTY.value))
_NO_COUNT_STATUSES = frozenset((ReaderStatus.UNSUPPORTED.value, ReaderStatus.INVALID.value))
_DISABLED_READER_STATUS = "disabled"
_AUDIT_READER_STATUSES = (
    _COUNT_BEARING_STATUSES | _NO_COUNT_STATUSES | frozenset((_DISABLED_READER_STATUS,))
)


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

    def __post_init__(self) -> None:
        named = {
            "state_start": self.state_start,
            "state_pad": self.state_pad,
            "state_end": self.state_end,
            "number_start": self.number_start,
            "number_end": self.number_end,
            "im_end": self.im_end,
            "vision_start": self.vision_start,
            "video_pad": self.video_pad,
            "vision_end": self.vision_end,
            "pad": self.pad,
        }
        if any(type(value) is not int or value < 0 for value in named.values()):
            raise ValueError("Composer token IDs must be non-negative integers")
        composer_ids = self.composer_ids
        if len(set(composer_ids)) != len(composer_ids):
            raise ValueError("the five Composer special tokens must have unique IDs")
        native_control_ids = (
            self.im_end,
            self.vision_start,
            self.video_pad,
            self.vision_end,
        )
        if any(value in composer_ids for value in native_control_ids):
            raise ValueError("Composer tokens cannot alias native Qwen control tokens")
        if len(set(native_control_ids)) != len(native_control_ids):
            raise ValueError("native Qwen control tokens must have unique IDs")

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
class ComposerRegistrationAudit:
    base_tokenizer_length: int
    new_tokenizer_length: int
    added_token_count: int
    token_ids: tuple[tuple[str, int], ...]
    initialization_strategy: str
    initialized_input_ids: tuple[int, ...]
    initialized_output_ids: tuple[int, ...]
    input_embedding_size_before: int
    input_embedding_size_after: int
    output_embedding_size_after: int | None
    output_embedding_tied: bool

    def __post_init__(self) -> None:
        counts = (
            self.base_tokenizer_length,
            self.new_tokenizer_length,
            self.added_token_count,
            self.input_embedding_size_before,
            self.input_embedding_size_after,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("registration audit sizes/counts must be non-negative integers")
        if self.new_tokenizer_length < self.base_tokenizer_length:
            raise ValueError("Composer registration cannot shrink the tokenizer")
        if self.input_embedding_size_after < self.input_embedding_size_before:
            raise ValueError("Composer registration cannot shrink input embeddings")
        if self.initialization_strategy != TOKEN_INITIALIZATION_STRATEGY:
            raise ValueError("registration audit has an unknown initialization strategy")
        if len(self.token_ids) != len(COMPOSER_SPECIAL_TOKENS):
            raise ValueError("registration audit must contain all five Composer token IDs")
        if tuple(name for name, _ in self.token_ids) != COMPOSER_SPECIAL_TOKENS:
            raise ValueError("registration audit token order must be canonical")
        ids = tuple(token_id for _, token_id in self.token_ids)
        if len(set(ids)) != len(ids) or any(token_id < 0 for token_id in ids):
            raise ValueError("registration audit Composer IDs must be unique and non-negative")
        if any(token_id not in ids for token_id in self.initialized_input_ids):
            raise ValueError("input initialization audit contains a non-Composer ID")
        if any(token_id not in ids for token_id in self.initialized_output_ids):
            raise ValueError("output initialization audit contains a non-Composer ID")
        if self.added_token_count == 0 and (
            self.initialized_input_ids or self.initialized_output_ids
        ):
            raise ValueError("reloaded Composer tokens must not be reinitialized")
        if self.output_embedding_tied and self.initialized_output_ids:
            raise ValueError("tied output embeddings must not be written a second time")
        if self.output_embedding_size_after is not None and self.output_embedding_size_after <= 0:
            raise ValueError("output embedding size must be positive when present")


@dataclass(frozen=True, slots=True)
class ComposerTokenRegistration:
    token_ids: ComposerSpecialTokenIds
    audit: ComposerRegistrationAudit


@dataclass(frozen=True, slots=True)
class CompositionRowAudit:
    reader_status: str
    exact_count: int | None
    source_token_count: int
    composed_token_count: int
    inserted_token_count: int
    insertion_index: int | None
    left_padding: int
    video_positions: tuple[int, ...]
    state_positions: tuple[int, ...]
    number_positions: tuple[int, ...]
    number_token_ids: tuple[int, ...]
    instruction_positions: tuple[int, ...]
    instruction_token_ids: tuple[int, ...]
    state_included: bool
    number_included: bool

    def __post_init__(self) -> None:
        if self.reader_status not in _AUDIT_READER_STATUSES:
            raise ValueError("row audit contains an unknown Reader status")
        counts = (
            self.source_token_count,
            self.composed_token_count,
            self.inserted_token_count,
            self.left_padding,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("row audit token counts must be non-negative integers")
        if self.composed_token_count != self.source_token_count + self.inserted_token_count:
            raise ValueError("row audit source/inserted/composed lengths do not add up")
        if self.insertion_index is not None and self.insertion_index >= self.source_token_count:
            raise ValueError("row audit insertion index must point inside the source sequence")
        position_groups = (
            self.video_positions,
            self.state_positions,
            self.number_positions,
            self.instruction_positions,
        )
        if any(
            any(type(value) is not int or value < 0 for value in group) for group in position_groups
        ):
            raise ValueError("row audit positions must be non-negative integers")
        flattened = tuple(value for group in position_groups for value in group)
        if len(set(flattened)) != len(flattened):
            raise ValueError("video/state/number audit positions must be disjoint")
        if type(self.state_included) is not bool or type(self.number_included) is not bool:
            raise TypeError("row audit include flags must be bool")
        if len(self.instruction_positions) != len(self.instruction_token_ids):
            raise ValueError("instruction positions must align with instruction token IDs")
        if self.reader_status in _COUNT_BEARING_STATUSES:
            if (self.state_included or self.number_included) != (self.insertion_index is not None):
                raise ValueError("count-bearing insertion index must follow enabled payloads")
            if self.state_included:
                if len(self.state_positions) != STATE_TOKEN_COUNT:
                    raise ValueError("enabled State payload must audit exactly 16 positions")
            elif self.state_positions:
                raise ValueError("disabled State payload cannot expose State positions")
            if self.number_included:
                if type(self.exact_count) is not int:
                    raise ValueError("enabled number payload requires exact_count")
                if not self.number_token_ids or len(self.number_positions) != len(
                    self.number_token_ids
                ):
                    raise ValueError("enabled number payload must audit Reader number IDs")
                if not self.instruction_token_ids:
                    raise ValueError("enabled number payload requires the exact-number instruction")
            elif (
                self.number_token_ids
                or self.number_positions
                or self.instruction_token_ids
                or self.instruction_positions
            ):
                raise ValueError("disabled number payload cannot expose number/instruction tokens")
            expected_inserted = (STATE_TOKEN_COUNT + 2 if self.state_included else 0) + (
                len(self.instruction_token_ids) + 2 + len(self.number_token_ids)
                if self.number_included
                else 0
            )
            if self.inserted_token_count != expected_inserted:
                raise ValueError("count-bearing row inserted length is inconsistent")
        else:
            if (
                self.exact_count is not None
                or self.insertion_index is not None
                or self.inserted_token_count != 0
                or self.state_positions
                or self.number_positions
                or self.number_token_ids
                or self.instruction_positions
                or self.instruction_token_ids
                or self.state_included
                or self.number_included
            ):
                raise ValueError("non-count-bearing rows cannot audit injected payload")


@dataclass(frozen=True, slots=True)
class ComposedInput:
    """One left-padded prefill and every placement/provenance tensor needed by Qwen."""

    input_ids: Tensor
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    rope_deltas: Tensor
    cache_position: Tensor
    video_position_mask: Tensor
    state_position_mask: Tensor
    number_position_mask: Tensor
    number_token_ids: tuple[tuple[int, ...], ...]
    row_audits: tuple[CompositionRowAudit, ...]
    special_token_ids: ComposerSpecialTokenIds
    registration_audit: ComposerRegistrationAudit

    def __post_init__(self) -> None:
        embeds = self.inputs_embeds
        if embeds.ndim != 3 or embeds.shape[-1] <= 0 or not torch.is_floating_point(embeds):
            raise ValueError("inputs_embeds must be floating [B, L, H]")
        batch_size, sequence_length = embeds.shape[:2]
        if (
            self.input_ids.shape != (batch_size, sequence_length)
            or self.input_ids.dtype not in (torch.int32, torch.int64)
            or self.input_ids.device != embeds.device
        ):
            raise ValueError("input_ids must be integer [B, L] on the embedding device")
        if (
            self.attention_mask.shape != (batch_size, sequence_length)
            or self.attention_mask.dtype not in (torch.bool, torch.int32, torch.int64)
            or self.attention_mask.device != embeds.device
        ):
            raise ValueError("attention_mask must be bool/integer [B, L] on the embedding device")
        if self.position_ids.shape != (3, batch_size, sequence_length):
            raise ValueError("position_ids must be [3, B, L]")
        if (
            self.position_ids.dtype not in (torch.int32, torch.int64)
            or self.position_ids.device != embeds.device
        ):
            raise ValueError("position_ids must be integer on the embedding device")
        if self.rope_deltas.shape != (batch_size, 1):
            raise ValueError("rope_deltas must be [B, 1]")
        if (
            self.rope_deltas.dtype not in (torch.int32, torch.int64)
            or self.rope_deltas.device != embeds.device
        ):
            raise ValueError("rope_deltas must be integer on the embedding device")
        if (
            self.cache_position.shape != (sequence_length,)
            or self.cache_position.dtype != torch.int64
            or self.cache_position.device != embeds.device
        ):
            raise ValueError("cache_position must be int64 [L] on the embedding device")
        expected_cache = torch.arange(sequence_length, device=embeds.device, dtype=torch.int64)
        if embeds.device.type != "meta" and not torch.equal(self.cache_position, expected_cache):
            raise ValueError("prefill cache_position must be zero-based and contiguous")

        masks = (
            (self.video_position_mask, "video_position_mask"),
            (self.state_position_mask, "state_position_mask"),
            (self.number_position_mask, "number_position_mask"),
        )
        for mask, name in masks:
            if (
                mask.shape != (batch_size, sequence_length)
                or mask.dtype is not torch.bool
                or mask.device != embeds.device
            ):
                raise ValueError(f"{name} must be bool [B, L] on the embedding device")
            if embeds.device.type != "meta" and bool(torch.any(mask & ~self.attention_mask.bool())):
                raise ValueError(f"{name} cannot mark padding")
        if embeds.device.type != "meta" and bool(
            torch.any(self.video_position_mask & self.state_position_mask)
            or torch.any(self.video_position_mask & self.number_position_mask)
            or torch.any(self.state_position_mask & self.number_position_mask)
        ):
            raise ValueError("video, State, and number positions must be pairwise disjoint")
        if len(self.number_token_ids) != batch_size or len(self.row_audits) != batch_size:
            raise ValueError("number_token_ids/row_audits need one entry per batch row")
        if self.registration_audit.token_ids != tuple(
            zip(COMPOSER_SPECIAL_TOKENS, self.special_token_ids.composer_ids, strict=True)
        ):
            raise ValueError("registration audit and active Composer token IDs disagree")
        if embeds.device.type != "meta":
            for row, (number_ids, audit) in enumerate(
                zip(self.number_token_ids, self.row_audits, strict=True)
            ):
                actual_ids = tuple(
                    int(value)
                    for value in self.input_ids[row, self.number_position_mask[row]].tolist()
                )
                if actual_ids != number_ids or actual_ids != audit.number_token_ids:
                    raise ValueError("number position IDs must equal the original Reader IDs")
                if tuple(torch.nonzero(self.video_position_mask[row]).flatten().tolist()) != (
                    audit.video_positions
                ):
                    raise ValueError("video mask and row audit disagree")
                if tuple(torch.nonzero(self.state_position_mask[row]).flatten().tolist()) != (
                    audit.state_positions
                ):
                    raise ValueError("State mask and row audit disagree")
                if tuple(torch.nonzero(self.number_position_mask[row]).flatten().tolist()) != (
                    audit.number_positions
                ):
                    raise ValueError("number mask and row audit disagree")
                actual_instruction_ids = tuple(
                    int(self.input_ids[row, position].item())
                    for position in audit.instruction_positions
                )
                if actual_instruction_ids != audit.instruction_token_ids:
                    raise ValueError("instruction positions and row audit IDs disagree")
                valid_count = int(self.attention_mask[row].sum().item())
                if audit.left_padding != sequence_length - valid_count:
                    raise ValueError("row audit left padding does not match attention_mask")
                if valid_count != audit.composed_token_count:
                    raise ValueError("row audit composed length does not match attention_mask")


@dataclass(frozen=True, slots=True)
class TeacherForcedRowAudit:
    """Absolute source-to-composed positions for one teacher-forced row."""

    source_valid_positions: tuple[int, ...]
    composed_source_positions: tuple[int, ...]
    source_supervised_positions: tuple[int, ...]
    composed_supervised_positions: tuple[int, ...]
    source_number_positions: tuple[int, ...]
    composed_number_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (
            self.source_valid_positions,
            self.composed_source_positions,
            self.source_supervised_positions,
            self.composed_supervised_positions,
            self.source_number_positions,
            self.composed_number_positions,
        )
        if any(
            any(type(position) is not int or position < 0 for position in group) for group in groups
        ):
            raise ValueError("teacher-forced audit positions must be non-negative integers")
        if any(tuple(sorted(set(group))) != group for group in groups):
            raise ValueError("teacher-forced audit positions must be unique and increasing")
        if len(self.source_valid_positions) != len(self.composed_source_positions):
            raise ValueError("source and composed provenance positions must be one-to-one")
        if len(self.source_supervised_positions) != len(self.composed_supervised_positions):
            raise ValueError("source and composed supervised positions must be one-to-one")
        if len(self.source_number_positions) != len(self.composed_number_positions):
            raise ValueError("source and composed number positions must be one-to-one")
        if not set(self.source_supervised_positions).issubset(self.source_valid_positions):
            raise ValueError("source supervised positions must be valid source positions")
        if not set(self.composed_supervised_positions).issubset(self.composed_source_positions):
            raise ValueError("composed supervised positions must retain source provenance")
        if not set(self.source_number_positions).issubset(self.source_supervised_positions):
            raise ValueError("source number positions must be supervised")
        if not set(self.composed_number_positions).issubset(self.composed_supervised_positions):
            raise ValueError("composed number positions must be supervised")
        provenance = dict(
            zip(self.source_valid_positions, self.composed_source_positions, strict=True)
        )
        if tuple(provenance[position] for position in self.source_supervised_positions) != (
            self.composed_supervised_positions
        ):
            raise ValueError("supervised positions must preserve source-to-composed provenance")
        if tuple(provenance[position] for position in self.source_number_positions) != (
            self.composed_number_positions
        ):
            raise ValueError("number positions must preserve source-to-composed provenance")


@dataclass(frozen=True, slots=True)
class TeacherForcedComposedInput:
    """A composed Qwen input plus answer-only labels and their number-token subset."""

    composed_input: ComposedInput
    labels: Tensor
    number_token_mask: Tensor
    row_audits: tuple[TeacherForcedRowAudit, ...]

    def __post_init__(self) -> None:
        composed = self.composed_input
        shape = composed.input_ids.shape
        device = composed.input_ids.device
        if self.labels.shape != shape or self.labels.dtype != torch.int64:
            raise ValueError("teacher-forced labels must be int64 [B, L_composed]")
        if self.labels.device != device:
            raise ValueError("teacher-forced labels must share the composed-input device")
        if self.number_token_mask.shape != shape or self.number_token_mask.dtype is not torch.bool:
            raise ValueError("teacher-forced number_token_mask must be bool [B, L_composed]")
        if self.number_token_mask.device != device:
            raise ValueError(
                "teacher-forced number_token_mask must share the composed-input device"
            )
        if len(self.row_audits) != shape[0]:
            raise ValueError("teacher-forced row_audits need one entry per batch row")
        if device.type == "meta":
            return

        supervised = self.labels != IGNORE_INDEX
        attention = composed.attention_mask.bool()
        if bool(torch.any(supervised & ~attention)):
            raise ValueError("padding labels must use ignore index -100")
        if bool(torch.any(self.number_token_mask & ~supervised)):
            raise ValueError("number_token_mask must be a subset of supervised assistant labels")
        if bool(torch.any(self.number_token_mask & composed.number_position_mask)):
            raise ValueError("Reader number context cannot become an answer number label")
        if bool(torch.any(supervised & composed.video_position_mask)):
            raise ValueError("video context labels must use ignore index -100")
        if bool(torch.any(self.number_token_mask[:, 0])):
            raise ValueError("the first label cannot be predicted by the causal shift")
        if bool(torch.any(supervised & (self.labels != composed.input_ids))):
            raise ValueError("supervised labels must preserve composed token-ID provenance")

        source_origin_mask = torch.zeros_like(supervised)
        for row, audit in enumerate(self.row_audits):
            composition_audit = composed.row_audits[row]
            if len(audit.source_valid_positions) != composition_audit.source_token_count:
                raise ValueError("teacher-forced source audit disagrees with Composer source count")
            if any(position >= shape[1] for position in audit.composed_source_positions):
                raise ValueError("teacher-forced composed provenance position is out of range")
            valid_positions = torch.nonzero(attention[row]).flatten()
            insertion_index = composition_audit.insertion_index
            inserted_count = composition_audit.inserted_token_count
            if insertion_index is None:
                relative_source_positions = torch.arange(
                    composition_audit.source_token_count,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                before = torch.arange(insertion_index, dtype=torch.int64, device=device)
                after = torch.arange(
                    insertion_index + inserted_count,
                    composition_audit.composed_token_count,
                    dtype=torch.int64,
                    device=device,
                )
                relative_source_positions = torch.cat((before, after))
            expected_composed_sources = tuple(
                valid_positions.index_select(0, relative_source_positions).tolist()
            )
            if expected_composed_sources != audit.composed_source_positions:
                raise ValueError("teacher-forced row audit does not match payload insertion")
            source_origin_mask[row, list(audit.composed_source_positions)] = True
            actual_supervised = tuple(torch.nonzero(supervised[row]).flatten().tolist())
            if actual_supervised != audit.composed_supervised_positions:
                raise ValueError("teacher-forced supervised mask and row audit disagree")
            actual_numbers = tuple(torch.nonzero(self.number_token_mask[row]).flatten().tolist())
            if actual_numbers != audit.composed_number_positions:
                raise ValueError("teacher-forced number mask and row audit disagree")
        if bool(torch.any(supervised & ~source_origin_mask)):
            raise ValueError(
                "inserted payload, boundary, instruction, and context labels must be -100"
            )

    @property
    def composed(self) -> ComposedInput:
        """Compatibility alias for callers that name the wrapped input ``composed``."""

        return self.composed_input


def register_input_composer_tokens(
    tokenizer: ComposerTokenizer,
    embedding_owner: EmbeddingOwner,
) -> ComposerSpecialTokenIds:
    """Register/initialize Composer tokens and return their stable IDs."""

    return register_input_composer_tokens_with_audit(tokenizer, embedding_owner).token_ids


def register_input_composer_tokens_with_audit(
    tokenizer: ComposerTokenizer,
    embedding_owner: EmbeddingOwner,
) -> ComposerTokenRegistration:
    """Register once, grow without shrinking, and deterministically initialize new rows.

    Only IDs added by this call are initialized.  An already-extended tokenizer therefore
    represents a checkpoint reload and leaves all learned input/output rows byte-for-byte alone.
    """

    if tokenizer is None or embedding_owner is None:
        raise ValueError("tokenizer and embedding_owner are required")
    with _REGISTRATION_LOCK:
        base_tokenizer_length = len(tokenizer)
        input_before = _embedding_layer(embedding_owner)
        input_size_before = _embedding_size(input_before)
        added_count = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(COMPOSER_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        if type(added_count) is not int or added_count < 0:
            raise TypeError("tokenizer.add_special_tokens() must return a non-negative integer")
        new_tokenizer_length = len(tokenizer)
        if new_tokenizer_length < base_tokenizer_length:
            raise RuntimeError("Composer registration must never shrink the tokenizer")
        special_ids = _resolve_special_token_ids(tokenizer)
        new_ids = tuple(
            token_id for token_id in special_ids.composer_ids if token_id >= base_tokenizer_length
        )
        if added_count == 0:
            new_ids = ()
        elif len(new_ids) != added_count:
            raise RuntimeError(
                "tokenizer-reported additions do not match newly allocated Composer IDs"
            )
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
            resize = getattr(embedding_owner, "resize_token_embeddings", None)
            if not callable(resize):
                raise TypeError("embedding_owner must resize embeddings after tokenizer growth")
            try:
                resize(target_size, mean_resizing=False)
            except TypeError:
                resize(target_size)
        input_after = _embedding_layer(embedding_owner)
        input_size_after = _embedding_size(input_after)
        if input_size_after < input_size_before:
            raise RuntimeError("Composer token registration must never shrink embeddings")
        if input_size_after < target_size:
            raise RuntimeError("model embeddings do not cover every registered tokenizer ID")
        output_after = _optional_output_embedding_layer(embedding_owner)
        output_size_after = None if output_after is None else _embedding_size(output_after)
        if output_size_after is not None and output_size_after < target_size:
            raise RuntimeError("output embeddings do not cover every registered tokenizer ID")
        output_tied = output_after is not None and _embedding_weights_share_storage(
            input_after,
            output_after,
        )
        initialized_input_ids: tuple[int, ...] = ()
        initialized_output_ids: tuple[int, ...] = ()
        if new_ids:
            _initialize_embedding_rows(
                input_after,
                new_ids,
                special_ids.initialization_source_ids,
            )
            initialized_input_ids = new_ids
            if output_after is not None and not output_tied:
                _initialize_embedding_rows(
                    output_after,
                    new_ids,
                    special_ids.initialization_source_ids,
                )
                initialized_output_ids = new_ids
        audit = ComposerRegistrationAudit(
            base_tokenizer_length=base_tokenizer_length,
            new_tokenizer_length=new_tokenizer_length,
            added_token_count=added_count,
            token_ids=tuple(zip(COMPOSER_SPECIAL_TOKENS, special_ids.composer_ids, strict=True)),
            initialization_strategy=TOKEN_INITIALIZATION_STRATEGY,
            initialized_input_ids=initialized_input_ids,
            initialized_output_ids=initialized_output_ids,
            input_embedding_size_before=input_size_before,
            input_embedding_size_after=input_size_after,
            output_embedding_size_after=output_size_after,
            output_embedding_tied=output_tied,
        )
        return ComposerTokenRegistration(token_ids=special_ids, audit=audit)


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
    payload_insertion_indices: Sequence[int | None] | None = None,
) -> ComposedInput:
    """Insert audited state/number segments and build one native-Qwen prefill.

    ``base_input_ids`` must be the native processor/chat-template prefill ending in
    the assistant generation prefix.  For each count-bearing Reader row, the new
    payload is inserted immediately before the final user ``<|im_end|>``.  The
    returned IDs retain native video placeholders, while ``inputs_embeds`` has only
    the 16 State placeholders replaced.
    """

    if type(include_state) is not bool or type(include_number) is not bool:
        raise TypeError("include_state/include_number must be bool")
    registration = register_input_composer_tokens_with_audit(tokenizer, embedding_owner)
    special_ids = registration.token_ids
    batch_size = _validate_compose_inputs(
        base_input_ids,
        base_attention_mask,
        state_tokens,
        state_token_valid_mask,
        reader_results,
        include_state=include_state,
        include_number=include_number,
        payload_insertion_indices=payload_insertion_indices,
    )
    device = _embedding_device(_embedding_layer(embedding_owner), base_input_ids.device)
    source_ids = base_input_ids.to(device=device, dtype=torch.int64)
    source_mask = base_attention_mask.to(device=device).bool()
    state_values = None if state_tokens is None else state_tokens.to(device=device)
    state_valid = (
        None if state_token_valid_mask is None else state_token_valid_mask.to(device=device)
    )

    row_ids: list[list[int]] = []
    row_origins: list[list[str]] = []
    row_metadata: list[
        tuple[
            str,
            int | None,
            int,
            int | None,
            tuple[int, ...],
            tuple[int, ...],
            bool,
            bool,
        ]
    ] = []
    forbidden_number_ids = set(special_ids.composer_ids) | {
        special_ids.im_end,
        special_ids.vision_start,
        special_ids.video_pad,
        special_ids.vision_end,
        special_ids.pad,
    }
    instruction_ids = (
        _encode_exact_number_instruction(tokenizer, forbidden_number_ids) if include_number else ()
    )
    for row in range(batch_size):
        valid_ids = [int(value) for value in source_ids[row, source_mask[row]].tolist()]
        if not valid_ids:
            raise ValueError("each base prefill row must contain at least one valid token")
        if any(value in special_ids.composer_ids for value in valid_ids):
            raise ValueError("base prefill already contains Composer payload tokens")
        if len(reader_results) != 0:
            status = _reader_status(reader_results[row])
            exact_count = reader_results[row].exact_count
            number_ids = tuple(reader_results[row].number_token_ids)
            row_state_valid = (
                False if not include_state or state_valid is None else bool(state_valid[row].item())
            )
            _validate_reader_row(
                status,
                exact_count,
                number_ids,
                row_state_valid,
                forbidden_number_ids,
                include_state=include_state,
                include_number=include_number,
            )
        else:
            status = _DISABLED_READER_STATUS
            exact_count = None
            number_ids = ()
        base_origins = [
            "video" if token_id == special_ids.video_pad else "base" for token_id in valid_ids
        ]
        state_included = status in _COUNT_BEARING_STATUSES and include_state
        number_included = status in _COUNT_BEARING_STATUSES and include_number
        inserted_number_ids = number_ids if number_included else ()
        inserted_instruction_ids = instruction_ids if number_included else ()
        if state_included or number_included:
            requested_insertion = (
                None if payload_insertion_indices is None else payload_insertion_indices[row]
            )
            if requested_insertion is None:
                im_end_positions = [
                    index
                    for index, token_id in enumerate(valid_ids)
                    if token_id == special_ids.im_end
                ]
                if not im_end_positions:
                    raise ValueError("payload composition requires a final user <|im_end|>")
                insertion_index = im_end_positions[-1]
            else:
                insertion_index = requested_insertion
                if (
                    insertion_index >= len(valid_ids)
                    or valid_ids[insertion_index] != special_ids.im_end
                ):
                    raise ValueError(
                        "payload insertion index must point to a valid user <|im_end|>"
                    )
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
        row_metadata.append(
            (
                status,
                exact_count,
                len(valid_ids),
                insertion_index,
                inserted_number_ids,
                inserted_instruction_ids,
                state_included,
                number_included,
            )
        )

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
    video_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    state_mask = torch.zeros_like(video_mask)
    number_mask = torch.zeros_like(video_mask)
    audits: list[CompositionRowAudit] = []
    for row, (ids, origins, metadata) in enumerate(
        zip(row_ids, row_origins, row_metadata, strict=True)
    ):
        width = len(ids)
        left_padding = max_length - width
        input_ids[row, left_padding:] = torch.tensor(ids, dtype=torch.int64, device=device)
        attention_mask[row, left_padding:] = 1
        for column, origin in enumerate(origins, start=left_padding):
            if origin == "video":
                video_mask[row, column] = True
            elif origin == "state":
                state_mask[row, column] = True
            elif origin == "number":
                number_mask[row, column] = True
        (
            status,
            exact_count,
            source_count,
            insertion_index,
            number_ids,
            row_instruction_ids,
            state_included,
            number_included,
        ) = metadata
        instruction_positions = tuple(
            column
            for column, origin in enumerate(origins, start=left_padding)
            if origin == "instruction"
        )
        audits.append(
            CompositionRowAudit(
                reader_status=status,
                exact_count=exact_count,
                source_token_count=source_count,
                composed_token_count=width,
                inserted_token_count=width - source_count,
                insertion_index=insertion_index,
                left_padding=left_padding,
                video_positions=tuple(torch.nonzero(video_mask[row]).flatten().tolist()),
                state_positions=tuple(torch.nonzero(state_mask[row]).flatten().tolist()),
                number_positions=tuple(torch.nonzero(number_mask[row]).flatten().tolist()),
                number_token_ids=number_ids,
                instruction_positions=instruction_positions,
                instruction_token_ids=row_instruction_ids,
                state_included=state_included,
                number_included=number_included,
            )
        )

    embedding = _embedding_layer(embedding_owner)
    embedded = cast(Tensor, embedding(input_ids))
    if embedded.ndim != 3 or embedded.shape[:2] != input_ids.shape:
        raise ValueError("input embedding table must return floating [B, L, H]")
    if not torch.is_floating_point(embedded):
        raise TypeError("input embedding table must return floating values")
    inputs_embeds = embedded.clone()
    if include_state:
        if state_values is None or state_valid is None:
            raise RuntimeError("enabled State composition lost its validated State tensors")
        if state_values.shape[-1] != embedded.shape[-1]:
            raise ValueError("State Token hidden size must match the Qwen input embedding size")
        state_values = state_values.to(dtype=embedded.dtype)
        for row in range(batch_size):
            if audits[row].state_included:
                if not bool(state_valid[row].item()):
                    raise RuntimeError("enabled State payload lost its valid-mask provenance")
                positions = torch.nonzero(state_mask[row]).flatten()
                if positions.numel() != STATE_TOKEN_COUNT:
                    raise RuntimeError("count-bearing row lost one or more State placeholders")
                inputs_embeds[row, positions] = state_values[row]
            elif bool(state_mask[row].any()):
                raise RuntimeError("non-State rows cannot expose State placeholders")

    position_ids, rope_deltas = _call_get_rope_index(
        rope_indexer,
        input_ids=input_ids,
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.to(device=device),
        attention_mask=attention_mask,
    )
    position_ids = position_ids.to(device=device)
    rope_deltas = rope_deltas.to(device=device)
    cache_position = torch.arange(max_length, dtype=torch.int64, device=device)
    return ComposedInput(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        video_position_mask=video_mask,
        state_position_mask=state_mask,
        number_position_mask=number_mask,
        number_token_ids=tuple(audit.number_token_ids for audit in audits),
        row_audits=tuple(audits),
        special_token_ids=special_ids,
        registration_audit=registration.audit,
    )


def compose_teacher_forced_inputs(
    *,
    base_input_ids: Tensor,
    base_attention_mask: Tensor,
    base_labels: Tensor,
    base_number_token_mask: Tensor,
    state_tokens: Tensor | None,
    state_token_valid_mask: Tensor | None,
    reader_results: Sequence[ReaderResultLike],
    tokenizer: ComposerTokenizer,
    embedding_owner: EmbeddingOwner,
    rope_indexer: RopeIndexer | RopeIndexCallable,
    video_grid_thw: Tensor | None,
    include_state: bool = True,
    include_number: bool = True,
) -> TeacherForcedComposedInput:
    """Compose a teacher-forced chat and retain only source assistant supervision.

    The source labels use ``-100`` for every prompt/context position.  Payload placement is
    resolved from the last context ``<|im_end|>`` before the final supervised assistant target,
    so an assistant turn's own end token cannot move State/Reader context into the answer.  The
    returned number mask is mapped only from ``base_number_token_mask``; the Reader-number mask
    on :class:`ComposedInput` is deliberately never reused as answer supervision.
    """

    _validate_teacher_forced_source_tensors(
        source_input_ids=base_input_ids,
        source_attention_mask=base_attention_mask,
        source_labels=base_labels,
        source_number_token_mask=base_number_token_mask,
    )
    insertion_indices: tuple[int | None, ...] | None = None
    if include_state or include_number:
        raw_im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
        if type(raw_im_end_id) is not int or raw_im_end_id < 0:
            raise ValueError("tokenizer is missing the native Qwen <|im_end|> token")
        insertion_indices = _teacher_forced_insertion_indices(
            source_input_ids=base_input_ids,
            source_attention_mask=base_attention_mask,
            source_labels=base_labels,
            im_end_id=raw_im_end_id,
        )
        if len(reader_results) == base_input_ids.shape[0]:
            for row, result in enumerate(reader_results):
                if (
                    _reader_status(result) in _COUNT_BEARING_STATUSES
                    and insertion_indices[row] is None
                ):
                    raise ValueError(
                        "teacher-forced payload requires a context user <|im_end|> "
                        "before the assistant target"
                    )
    composed = compose_inputs(
        base_input_ids=base_input_ids,
        base_attention_mask=base_attention_mask,
        state_tokens=state_tokens,
        state_token_valid_mask=state_token_valid_mask,
        reader_results=reader_results,
        tokenizer=tokenizer,
        embedding_owner=embedding_owner,
        rope_indexer=rope_indexer,
        video_grid_thw=video_grid_thw,
        include_state=include_state,
        include_number=include_number,
        payload_insertion_indices=insertion_indices,
    )
    return map_teacher_forced_targets(
        composed_input=composed,
        source_input_ids=base_input_ids,
        source_attention_mask=base_attention_mask,
        source_labels=base_labels,
        source_number_token_mask=base_number_token_mask,
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

    batch_size = _validate_teacher_forced_source_tensors(
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        source_labels=source_labels,
        source_number_token_mask=source_number_token_mask,
        expected_batch_size=composed_input.input_ids.shape[0],
    )
    device = composed_input.input_ids.device
    source_ids = source_input_ids.to(device=device, dtype=torch.int64)
    source_attention = source_attention_mask.to(device=device).bool()
    source_targets = source_labels.to(device=device, dtype=torch.int64)
    source_numbers = source_number_token_mask.to(device=device)
    labels = torch.full_like(composed_input.input_ids, IGNORE_INDEX)
    number_token_mask = torch.zeros_like(composed_input.input_ids, dtype=torch.bool)
    row_audits: list[TeacherForcedRowAudit] = []

    for row in range(batch_size):
        composition_audit = composed_input.row_audits[row]
        source_valid_positions_tensor = torch.nonzero(source_attention[row]).flatten()
        source_valid_positions = tuple(source_valid_positions_tensor.tolist())
        source_count = len(source_valid_positions)
        if source_count != composition_audit.source_token_count:
            raise ValueError("source attention provenance disagrees with Composer source count")
        composed_valid_positions = torch.nonzero(
            composed_input.attention_mask[row].bool()
        ).flatten()
        if composed_valid_positions.numel() != composition_audit.composed_token_count:
            raise ValueError("composed attention provenance disagrees with row audit")

        insertion_index = composition_audit.insertion_index
        inserted_count = composition_audit.inserted_token_count
        if insertion_index is None:
            if inserted_count != 0:
                raise ValueError("Composer audit cannot insert tokens without an insertion index")
            relative_source_positions = torch.arange(source_count, dtype=torch.int64, device=device)
        else:
            if insertion_index > source_count:
                raise ValueError("Composer insertion index exceeds the source token count")
            before = torch.arange(insertion_index, dtype=torch.int64, device=device)
            after = torch.arange(
                insertion_index + inserted_count,
                source_count + inserted_count,
                dtype=torch.int64,
                device=device,
            )
            relative_source_positions = torch.cat((before, after))
        if relative_source_positions.numel() != source_count:
            raise RuntimeError("teacher-forced source mapping lost one or more tokens")
        composed_source_positions_tensor = composed_valid_positions.index_select(
            0, relative_source_positions
        )
        source_valid_ids = source_ids[row].index_select(0, source_valid_positions_tensor)
        composed_source_ids = composed_input.input_ids[row].index_select(
            0, composed_source_positions_tensor
        )
        if not torch.equal(source_valid_ids, composed_source_ids):
            raise ValueError("composed token IDs do not preserve source provenance")

        valid_targets = source_targets[row].index_select(0, source_valid_positions_tensor)
        valid_number_mask = source_numbers[row].index_select(0, source_valid_positions_tensor)
        labels[row].index_copy_(0, composed_source_positions_tensor, valid_targets)
        number_token_mask[row].index_copy_(0, composed_source_positions_tensor, valid_number_mask)
        supervised_relative = torch.nonzero(valid_targets != IGNORE_INDEX).flatten()
        number_relative = torch.nonzero(valid_number_mask).flatten()
        source_supervised = source_valid_positions_tensor.index_select(0, supervised_relative)
        composed_supervised = composed_source_positions_tensor.index_select(0, supervised_relative)
        source_number = source_valid_positions_tensor.index_select(0, number_relative)
        composed_number = composed_source_positions_tensor.index_select(0, number_relative)
        row_audits.append(
            TeacherForcedRowAudit(
                source_valid_positions=source_valid_positions,
                composed_source_positions=tuple(composed_source_positions_tensor.tolist()),
                source_supervised_positions=tuple(source_supervised.tolist()),
                composed_supervised_positions=tuple(composed_supervised.tolist()),
                source_number_positions=tuple(source_number.tolist()),
                composed_number_positions=tuple(composed_number.tolist()),
            )
        )

    return TeacherForcedComposedInput(
        composed_input=composed_input,
        labels=labels,
        number_token_mask=number_token_mask,
        row_audits=tuple(row_audits),
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
    missing = [token for token, value in values.items() if type(value) is not int or value < 0]
    if missing:
        raise ValueError(f"tokenizer is missing required special tokens: {missing}")
    pad_id = tokenizer.pad_token_id
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("tokenizer requires a non-negative pad_token_id")
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
        pad=pad_id,
    )


def _embedding_layer(owner: EmbeddingOwner) -> nn.Module:
    getter = getattr(owner, "get_input_embeddings", None)
    if not callable(getter):
        raise TypeError("embedding_owner must expose get_input_embeddings()")
    embedding = getter()
    if not isinstance(embedding, nn.Module) or not callable(embedding):
        raise TypeError("get_input_embeddings() must return a callable torch module")
    return embedding


def _optional_output_embedding_layer(owner: EmbeddingOwner) -> nn.Module | None:
    getter = getattr(owner, "get_output_embeddings", None)
    if not callable(getter):
        return None
    embedding = getter()
    if embedding is None:
        return None
    if not isinstance(embedding, nn.Module):
        raise TypeError("get_output_embeddings() must return a torch module or None")
    _embedding_weight(embedding)
    return embedding


def _embedding_weight(embedding: nn.Module) -> Tensor:
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise TypeError("embedding modules must expose a 2D weight tensor")
    return weight


def _embedding_size(embedding: nn.Module) -> int:
    num_embeddings = getattr(embedding, "num_embeddings", None)
    if type(num_embeddings) is int and num_embeddings > 0:
        return num_embeddings
    weight = getattr(embedding, "weight", None)
    if isinstance(weight, Tensor) and weight.ndim == 2 and weight.shape[0] > 0:
        return int(weight.shape[0])
    raise TypeError("input embedding module must expose num_embeddings or a 2D weight")


def _embedding_weights_share_storage(left: nn.Module, right: nn.Module) -> bool:
    left_weight = _embedding_weight(left)
    right_weight = _embedding_weight(right)
    if left_weight is right_weight:
        return True
    if left_weight.device.type == "meta" or right_weight.device.type == "meta":
        return False
    if left_weight.device != right_weight.device:
        return False
    return bool(
        left_weight.untyped_storage().data_ptr() == right_weight.untyped_storage().data_ptr()
    )


def _initialize_embedding_rows(
    embedding: nn.Module,
    target_ids: tuple[int, ...],
    source_ids: tuple[int, int, int],
) -> None:
    weight = _embedding_weight(embedding)
    if weight.device.type == "meta":
        raise ValueError("Composer token initialization requires materialized embeddings")
    if max((*target_ids, *source_ids)) >= weight.shape[0]:
        raise ValueError("Composer/source token ID exceeds the embedding table")
    if min((*target_ids, *source_ids)) < 0:
        raise ValueError("Composer/source token IDs must be non-negative")
    source_index = torch.tensor(source_ids, dtype=torch.int64, device=weight.device)
    target_index = torch.tensor(target_ids, dtype=torch.int64, device=weight.device)
    source_mean = weight.detach().index_select(0, source_index).float().mean(dim=0)
    initialized = source_mean.to(dtype=weight.dtype).expand(len(target_ids), -1)
    with torch.no_grad():
        weight.index_copy_(0, target_index, initialized)


def _embedding_device(embedding: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(embedding.parameters(), None)
    return parameter.device if parameter is not None else fallback


def _validate_compose_inputs(
    base_input_ids: Tensor,
    base_attention_mask: Tensor,
    state_tokens: Tensor | None,
    state_token_valid_mask: Tensor | None,
    reader_results: Sequence[ReaderResultLike],
    *,
    include_state: bool,
    include_number: bool,
    payload_insertion_indices: Sequence[int | None] | None,
) -> int:
    if (
        base_input_ids.ndim != 2
        or base_input_ids.shape[0] <= 0
        or base_input_ids.shape[1] <= 0
        or base_input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("base_input_ids must be non-empty integer [B, L_base]")
    if base_attention_mask.shape != base_input_ids.shape or base_attention_mask.dtype not in (
        torch.bool,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("base_attention_mask must be bool/integer [B, L_base]")
    if base_attention_mask.device.type != "meta":
        unique = set(int(value) for value in torch.unique(base_attention_mask).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("base_attention_mask must be binary")
    batch_size = base_input_ids.shape[0]
    if include_state:
        if state_tokens is None or (
            state_tokens.ndim != 3
            or state_tokens.shape[:2] != (batch_size, STATE_TOKEN_COUNT)
            or state_tokens.shape[-1] <= 0
            or not torch.is_floating_point(state_tokens)
        ):
            raise ValueError("state_tokens must be floating [B, 16, H] when enabled")
        if state_token_valid_mask is None or (
            state_token_valid_mask.shape != (batch_size,)
            or state_token_valid_mask.dtype is not torch.bool
        ):
            raise ValueError("state_token_valid_mask must be bool [B] when State is enabled")
        if state_tokens.device.type != "meta" and not bool(torch.isfinite(state_tokens).all()):
            raise ValueError("state_tokens must be finite")
    reader_count = len(reader_results)
    if reader_count == 0:
        if include_state or include_number:
            raise ValueError("enabled State/number composition requires one Reader row per item")
    elif reader_count != batch_size:
        raise ValueError("reader_results must contain one row per batch item")
    if payload_insertion_indices is not None:
        if len(payload_insertion_indices) != batch_size:
            raise ValueError("payload_insertion_indices need one entry per batch row")
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in payload_insertion_indices
        ):
            raise ValueError("payload insertion indices must be non-negative integers or None")
    return int(batch_size)


def _validate_teacher_forced_source_tensors(
    *,
    source_input_ids: Tensor,
    source_attention_mask: Tensor,
    source_labels: Tensor,
    source_number_token_mask: Tensor,
    expected_batch_size: int | None = None,
) -> int:
    if (
        source_input_ids.ndim != 2
        or source_input_ids.shape[0] <= 0
        or source_input_ids.shape[1] <= 0
        or source_input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("source_input_ids must be non-empty integer [B, L_source]")
    shape = source_input_ids.shape
    if source_attention_mask.shape != shape or source_attention_mask.dtype not in (
        torch.bool,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("source_attention_mask must be bool/integer [B, L_source]")
    if source_labels.shape != shape or source_labels.dtype != torch.int64:
        raise ValueError("source_labels must be int64 [B, L_source]")
    if source_number_token_mask.shape != shape or source_number_token_mask.dtype is not torch.bool:
        raise ValueError("source_number_token_mask must be bool [B, L_source]")
    devices = {
        source_input_ids.device,
        source_attention_mask.device,
        source_labels.device,
        source_number_token_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("teacher-forced source tensors must share one device")
    batch_size = int(shape[0])
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError("teacher-forced source batch size must match ComposedInput")
    if source_input_ids.device.type == "meta":
        return batch_size

    unique_attention = set(int(value) for value in torch.unique(source_attention_mask).tolist())
    if not unique_attention.issubset({0, 1}):
        raise ValueError("source_attention_mask must be binary")
    attention = source_attention_mask.bool()
    supervised = source_labels != IGNORE_INDEX
    if bool(torch.any(supervised & ~attention)):
        raise ValueError("masked source labels must be -100 outside source attention")
    if bool(torch.any(supervised & (source_labels < 0))):
        raise ValueError("supervised source labels must be non-negative token IDs")
    if bool(torch.any(supervised & (source_labels != source_input_ids))):
        raise ValueError("supervised source labels must equal their source token IDs")
    if bool(torch.any(source_number_token_mask & ~supervised)):
        raise ValueError("source number mask must be a subset of supervised source labels")
    return batch_size


def _teacher_forced_insertion_indices(
    *,
    source_input_ids: Tensor,
    source_attention_mask: Tensor,
    source_labels: Tensor,
    im_end_id: int,
) -> tuple[int | None, ...]:
    """Locate the last context user-end before the final supervised assistant token."""

    indices: list[int | None] = []
    attention = source_attention_mask.bool()
    for row in range(source_input_ids.shape[0]):
        valid_positions = torch.nonzero(attention[row]).flatten()
        valid_ids = source_input_ids[row].index_select(0, valid_positions)
        valid_labels = source_labels[row].index_select(0, valid_positions)
        supervised_positions = torch.nonzero(valid_labels != IGNORE_INDEX).flatten()
        upper_bound = (
            int(supervised_positions[-1].item())
            if supervised_positions.numel() > 0
            else int(valid_ids.numel())
        )
        candidates = torch.nonzero(
            (valid_ids == im_end_id)
            & (valid_labels == IGNORE_INDEX)
            & (torch.arange(valid_ids.numel(), device=valid_ids.device) < upper_bound)
        ).flatten()
        indices.append(int(candidates[-1].item()) if candidates.numel() > 0 else None)
    return tuple(indices)


def _reader_status(result: ReaderResultLike) -> str:
    status = result.status
    value = status.value if isinstance(status, ReaderStatus) else status
    if not isinstance(value, str) or value not in _COUNT_BEARING_STATUSES | _NO_COUNT_STATUSES:
        raise ValueError("Reader result has an unknown status")
    return value


def _validate_reader_row(
    status: str,
    exact_count: int | None,
    number_ids: tuple[int, ...],
    state_valid: bool,
    forbidden_number_ids: set[int],
    *,
    include_state: bool,
    include_number: bool,
) -> None:
    if any(type(value) is not int or value < 0 for value in number_ids):
        raise ValueError("Reader number IDs must be non-negative integers")
    if any(value in forbidden_number_ids for value in number_ids):
        raise ValueError("Reader number IDs cannot alias Composer/Qwen control tokens")
    if status in _COUNT_BEARING_STATUSES:
        if include_number and (type(exact_count) is not int or not number_ids):
            raise ValueError("enabled number payload requires exact_count and Reader IDs")
        if include_state and not state_valid:
            raise ValueError("OK/EMPTY Reader rows require valid State Tokens")
    elif exact_count is not None or number_ids or state_valid:
        raise ValueError("UNSUPPORTED/INVALID rows cannot inject State or number payload")


def _encode_exact_number_instruction(
    tokenizer: ComposerTokenizer,
    forbidden_ids: set[int],
) -> tuple[int, ...]:
    raw_ids = tokenizer.encode(EXACT_NUMBER_INSTRUCTION, add_special_tokens=False)
    instruction_ids = tuple(raw_ids)
    if not instruction_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in instruction_ids
    ):
        raise ValueError("exact-number instruction must encode to ordinary non-negative IDs")
    if any(token_id in forbidden_ids for token_id in instruction_ids):
        raise ValueError("exact-number instruction cannot contain Composer/Qwen control IDs")
    return instruction_ids


def _call_get_rope_index(
    rope_indexer: RopeIndexer | RopeIndexCallable,
    *,
    input_ids: Tensor,
    video_grid_thw: Tensor | None,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    method = getattr(rope_indexer, "get_rope_index", None)
    callable_indexer = method if callable(method) else rope_indexer
    if not callable(callable_indexer):
        raise TypeError("rope_indexer must be callable or expose get_rope_index()")
    result = callable_indexer(
        input_ids=input_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("get_rope_index must return (position_ids, rope_deltas)")
    position_ids, rope_deltas = result
    if not isinstance(position_ids, Tensor) or not isinstance(rope_deltas, Tensor):
        raise TypeError("get_rope_index outputs must be tensors")
    return position_ids, rope_deltas
