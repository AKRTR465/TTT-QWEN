from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

from tests.support.tiny_qwen import make_tiny_hf_model, make_tiny_project_config
from ttt_svcbench_qwen.input_composer import compose_inputs
from ttt_svcbench_qwen.qwen_adapter import Qwen3VLAdapter, StateEmbeddingPayload
from ttt_svcbench_qwen.state_reader import ReaderStatus


class SyntheticTokenizer:
    """Minimal fixed-vocabulary tokenizer with the native Qwen control IDs."""

    pad_token_id = 0

    def __init__(self) -> None:
        self._length = 32
        self._ids = {
            "<|im_end|>": 2,
            "<|vision_start|>": 26,
            "<|vision_end|>": 27,
            "<|video_pad|>": 29,
        }

    def __len__(self) -> int:
        return self._length

    def add_special_tokens(
        self,
        special_tokens_dict: Mapping[str, object],
        replace_additional_special_tokens: bool = True,
    ) -> int:
        assert replace_additional_special_tokens is False
        values = special_tokens_dict["additional_special_tokens"]
        assert isinstance(values, list)
        added = 0
        for value in values:
            assert isinstance(value, str)
            if value not in self._ids:
                self._ids[value] = self._length
                self._length += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._ids.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> tuple[int, int]:
        assert text
        assert add_special_tokens is False
        return (9, 10)


@dataclass(frozen=True, slots=True)
class SyntheticReaderResult:
    status: ReaderStatus
    exact_count: int
    number_token_ids: tuple[int, ...]


class CountingAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, embeddings: Tensor, valid_mask: Tensor, _metadata: object) -> Tensor:
        self.calls += 1
        return embeddings + valid_mask.unsqueeze(-1).to(embeddings.dtype)


class MaskedScatterCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        if func is torch.ops.aten.masked_scatter.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))  # type: ignore[operator]


def test_tiny_composer_to_native_qwen_prefill_and_decode_contract() -> None:
    qwen = make_tiny_hf_model(seed=13)
    owner = qwen.model
    tokenizer = SyntheticTokenizer()
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    pixels = torch.randn(4, 12)
    base_ids = torch.tensor([[7, 26, 29, 27, 5, 2, 6]], dtype=torch.int64)
    base_mask = torch.ones_like(base_ids)
    state_tokens = torch.arange(16 * 8, dtype=torch.float32).reshape(1, 16, 8) / 100.0
    reader = SyntheticReaderResult(ReaderStatus.OK, 12, (8,))

    composed = compose_inputs(
        base_input_ids=base_ids,
        base_attention_mask=base_mask,
        state_tokens=state_tokens,
        state_token_valid_mask=torch.tensor([True]),
        reader_results=(reader,),
        tokenizer=tokenizer,
        embedding_owner=qwen,
        rope_indexer=owner,
        video_grid_thw=grid,
        include_state=True,
        include_number=True,
    )
    video_position_mask = composed.input_ids == tokenizer.convert_tokens_to_ids("<|video_pad|>")
    number_position_mask = composed.input_ids == reader.number_token_ids[0]
    assert not torch.any(
        video_position_mask & composed.state_position_mask
        | video_position_mask & number_position_mask
        | composed.state_position_mask & number_position_mask
    )
    assert video_position_mask.sum().item() == 1
    assert composed.state_position_mask.sum().item() == 16
    assert number_position_mask.sum().item() == 1

    payload = StateEmbeddingPayload(
        composed.input_ids,
        composed.state_position_mask,
        composed.inputs_embeds[composed.state_position_mask],
    )
    adapter = CountingAdapter()
    wrapper = Qwen3VLAdapter(
        qwen,
        make_tiny_project_config(),
        adapter,
        adapter_enabled=True,
    )
    merger_outputs: list[Tensor] = []
    merger_handle = owner.visual.merger.register_forward_hook(
        lambda _module, _args, output: (merger_outputs.append(output), output)[1]
    )
    wrapper.get_video_features(pixels, grid)
    state_prepared = wrapper.last_prepared_video_features
    assert state_prepared is not None
    assert adapter.calls == 1
    assert len(merger_outputs) == 1

    language_calls: list[dict[str, Tensor | None]] = []

    def capture_language_call(
        _module: nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        record: dict[str, Tensor | None] = {}
        for name in ("inputs_embeds", "position_ids", "cache_position", "visual_pos_masks"):
            value = kwargs.get(name)
            record[name] = value.detach().clone() if isinstance(value, Tensor) else None
        language_calls.append(record)

    language_handle = owner.language_model.register_forward_pre_hook(
        capture_language_call,
        with_kwargs=True,
    )
    deepstack_masks: list[Tensor] = []
    deepstack_ids: list[int] = []
    original_deepstack = owner.language_model._deepstack_process

    def capture_deepstack(
        self: object,
        hidden_states: Tensor,
        visual_pos_masks: Tensor,
        visual_embeds: Tensor,
    ) -> Tensor:
        del self
        deepstack_masks.append(visual_pos_masks.detach().clone())
        deepstack_ids.append(id(visual_embeds))
        return original_deepstack(hidden_states, visual_pos_masks, visual_embeds)

    owner.language_model._deepstack_process = MethodType(  # type: ignore[method-assign]
        capture_deepstack,
        owner.language_model,
    )
    owner.rope_deltas = None
    scatter_counter = MaskedScatterCounter()
    try:
        with scatter_counter:
            generated = wrapper.generate(
                input_ids=composed.input_ids,
                attention_mask=composed.attention_mask,
                pixel_values_videos=pixels,
                video_grid_thw=grid,
                state_embedding_payload=payload,
                use_cache=True,
                max_new_tokens=3,
                do_sample=False,
            )
    finally:
        language_handle.remove()
        merger_handle.remove()
        owner.language_model._deepstack_process = original_deepstack  # type: ignore[method-assign]

    prompt_length = composed.input_ids.shape[1]
    assert generated.shape == (1, prompt_length + 3)
    answer_prepared = wrapper.last_prepared_video_features
    assert answer_prepared is not None
    assert answer_prepared is not state_prepared
    assert [call["inputs_embeds"].shape[1] for call in language_calls] == [  # type: ignore[union-attr]
        prompt_length,
        1,
        1,
    ]
    prefill = language_calls[0]
    assert torch.equal(prefill["position_ids"], composed.position_ids)
    assert torch.equal(owner.rope_deltas, composed.rope_deltas)
    assert torch.equal(prefill["visual_pos_masks"], video_position_mask)
    assert torch.equal(
        prefill["inputs_embeds"][composed.state_position_mask],  # type: ignore[index]
        payload.state_embeddings,
    )
    expected_number_embeddings = qwen.get_input_embeddings()(
        composed.input_ids[number_position_mask]
    )
    assert torch.equal(
        prefill["inputs_embeds"][number_position_mask],  # type: ignore[index]
        expected_number_embeddings,
    )
    assert all(torch.equal(mask, video_position_mask) for mask in deepstack_masks)
    assert deepstack_ids == [id(value) for value in answer_prepared.deepstack_features]
    assert adapter.calls == 2
    assert len(merger_outputs) == 2
    assert scatter_counter.calls == 2  # one State scatter plus native video scatter
