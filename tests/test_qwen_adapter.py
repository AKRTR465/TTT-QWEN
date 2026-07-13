from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionModel,
)

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.qwen_adapter import (
    MergedVideoMetadata,
    Qwen3VLAdapter,
    QwenVideoFeatureBoundary,
    QwenVisualOutput,
    VideoBatch,
    assert_qwen_checkpoint_config,
    build_qwen_adapter,
)

ROOT = Path(__file__).resolve().parents[1]


def make_tiny_project_config() -> ProjectConfig:
    base = load_config()
    vision = base.model.vision.model_copy(
        update={
            "depth": 3,
            "hidden_size": 8,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "output_size": 8,
            "deepstack_visual_indexes": (0, 1, 2),
        }
    )
    llm = base.model.llm.model_copy(update={"num_layers": 3, "hidden_size": 8})
    model = base.model.model_copy(update={"vision": vision, "llm": llm})
    return base.model_copy(update={"model": model})


def make_tiny_hf_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        vision_config={
            "depth": 3,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1, 2],
        },
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "use_cache": False,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
    )


def make_tiny_hf_model() -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(make_tiny_hf_config()).eval()


def video_inputs() -> dict[str, Tensor | bool]:
    return {
        "input_ids": torch.tensor([[26, 29, 27, 1]], dtype=torch.int64),
        "pixel_values_videos": torch.randn(4, 12),
        "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        "use_cache": False,
    }


def image_inputs() -> dict[str, Tensor | bool]:
    return {
        "input_ids": torch.tensor([[26, 28, 27, 1]], dtype=torch.int64),
        "pixel_values": torch.randn(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        "use_cache": False,
    }


def mixed_inputs() -> dict[str, Tensor | bool]:
    return {
        "input_ids": torch.tensor([[26, 28, 27, 26, 29, 27, 1]], dtype=torch.int64),
        "pixel_values": torch.randn(4, 12),
        "pixel_values_videos": torch.randn(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        "use_cache": False,
    }


def logits(model: nn.Module, inputs: dict[str, Tensor | bool]) -> Tensor:
    owner = getattr(model, "model", None)
    if owner is not None and hasattr(owner, "rope_deltas"):
        owner.rope_deltas = None
    with torch.no_grad():
        return model(**inputs).logits


class AddOneAdapter(nn.Module):
    def forward(
        self,
        embeddings: Tensor,
        valid_mask: Tensor,
        metadata: MergedVideoMetadata,
    ) -> Tensor:
        assert metadata.token_offsets[-1] == int(valid_mask.sum().item())
        return embeddings + valid_mask.unsqueeze(-1).to(embeddings.dtype)


class TrainableScaleAdapter(nn.Module):
    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward(
        self,
        embeddings: Tensor,
        valid_mask: Tensor,
        metadata: MergedVideoMetadata,
    ) -> Tensor:
        assert metadata.merged_grid_thw.shape[0] == embeddings.shape[0]
        if self.events is not None:
            self.events.append("adapter")
        return embeddings * self.scale * valid_mask.unsqueeze(-1).to(embeddings.dtype)


class BadShapeAdapter(nn.Module):
    def forward(
        self,
        embeddings: Tensor,
        valid_mask: Tensor,
        metadata: MergedVideoMetadata,
    ) -> Tensor:
        del valid_mask, metadata
        return embeddings[:, :-1]


class MaskedScatterRecorder(TorchDispatchMode):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        if func is torch.ops.aten.masked_scatter.default:
            self.events.append("masked_scatter")
        return func(*args, **(kwargs or {}))  # type: ignore[operator]


def test_official_qwen_modules_match_demo_shapes_on_meta_device() -> None:
    config = Qwen3VLVisionConfig(
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=4096,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
    )
    cu_seqlens = torch.tensor([0, 1568], dtype=torch.int32, device="cpu")
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(config)
        patches = visual.patch_embed(torch.empty(1568, 1536))
        merged = visual.merger(patches)
        position_embeddings = (torch.empty(1568, 72), torch.empty(1568, 72))
        block_output = patches
        block_shapes = []
        for block in visual.blocks:
            block_output = block(
                block_output,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            block_shapes.append(block_output.shape)

    assert patches.shape == (1568, 1152)
    assert block_output.shape == patches.shape
    assert len(visual.blocks) == 27
    assert block_shapes == [patches.shape] * 27
    assert visual.merger.hidden_size == 4 * 1152 == 4608
    assert visual.merger.linear_fc1.in_features == 4608
    assert visual.merger.linear_fc2.out_features == 4096
    assert merged.shape == (392, 4096)
    assert len(visual.deepstack_merger_list) == 3


def test_demo_boundary_exposes_padded_main_output_and_merged_grid() -> None:
    grid = torch.tensor([[8, 14, 14]], dtype=torch.int64)
    packed = torch.zeros(392, 4096, dtype=torch.float16)
    deepstack = [packed, packed, packed]
    boundary = QwenVideoFeatureBoundary(load_config())

    boundary.intercept_features((packed,), deepstack, grid)
    captured = boundary.last_output

    assert captured is not None
    assert captured.main_visual_embeddings.shape == (1, 392, 4096)
    assert captured.video_grid_thw.tolist() == [[8, 14, 14]]
    assert captured.merged_grid_thw.tolist() == [[8, 7, 7]]
    assert captured.token_offsets == (0, 392)


def test_variable_video_mapping_and_deepstack_objects_are_preserved() -> None:
    config = load_config()
    grid = torch.tensor([[2, 4, 4], [1, 2, 4]], dtype=torch.int64)
    main = (torch.zeros(8, 4096), torch.ones(2, 4096))
    deepstack = [torch.full((10, 4096), float(index)) for index in range(3)]
    boundary = QwenVideoFeatureBoundary(config)

    returned_main, returned_deepstack = boundary.intercept_features(main, deepstack, grid)
    captured = boundary.last_output

    assert returned_main is main
    assert returned_deepstack is deepstack
    assert captured is not None
    assert captured.main_visual_embeddings.shape == (2, 8, 4096)
    assert captured.visual_valid_mask.sum(dim=1).tolist() == [8, 2]
    assert captured.merged_grid_thw.tolist() == [[2, 2, 2], [1, 1, 2]]
    assert captured.token_offsets == (0, 8, 10)
    assert all(captured.deepstack_features[index] is deepstack[index] for index in range(3))
    assert captured.padded_deepstack_feature(0).shape == captured.main_visual_embeddings.shape


def test_visual_output_rejects_invalid_padding_width_and_deepstack_count() -> None:
    metadata = MergedVideoMetadata(
        video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
        merged_grid_thw=torch.tensor([[2, 1, 1]], dtype=torch.int64),
        spatial_merge_size=2,
        token_counts=(2,),
        token_offsets=(0, 2),
    )
    packed = torch.zeros(2, 4096)

    with pytest.raises(ValueError, match="padding width"):
        QwenVisualOutput(
            main_visual_embeddings=torch.zeros(1, 1, 4096),
            deepstack_features=(packed, packed, packed),
            visual_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            metadata=metadata,
        )
    with pytest.raises(ValueError, match="exactly three"):
        QwenVisualOutput(
            main_visual_embeddings=torch.zeros(1, 2, 4096),
            deepstack_features=(packed, packed),  # type: ignore[arg-type]
            visual_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            metadata=metadata,
        )


def test_boundary_preserves_standard_nn_module_apply_api() -> None:
    boundary = QwenVideoFeatureBoundary(load_config())
    visited: list[nn.Module] = []

    returned = boundary.apply(visited.append)

    assert returned is boundary
    assert visited == [boundary]


def test_enabled_boundary_adapts_only_main_and_rejects_shape_changes() -> None:
    config = load_config()
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    main = (torch.zeros(1, 4096),)
    deepstack = [torch.zeros(1, 4096) for _ in range(3)]
    boundary = QwenVideoFeatureBoundary(config, AddOneAdapter(), adapter_enabled=True)

    adapted, returned_deepstack = boundary.intercept_features(main, deepstack, grid)

    assert torch.equal(adapted[0], torch.ones_like(main[0]))
    assert returned_deepstack is deepstack
    assert all(returned_deepstack[index] is deepstack[index] for index in range(3))
    with pytest.raises(ValueError, match="preserve"):
        QwenVideoFeatureBoundary(
            config,
            BadShapeAdapter(),
            adapter_enabled=True,
        ).intercept_features(main, deepstack, grid)


def test_tiny_real_hf_disabled_wrapper_is_bitwise_equivalent_for_all_input_kinds() -> None:
    model = make_tiny_hf_model()
    wrapper = Qwen3VLAdapter(model, make_tiny_project_config())
    video = video_inputs()
    image = image_inputs()
    text = {"input_ids": torch.tensor([[1, 2, 3]]), "use_cache": False}

    pixel_values_videos = video["pixel_values_videos"]
    video_grid_thw = video["video_grid_thw"]
    assert isinstance(pixel_values_videos, Tensor)
    assert isinstance(video_grid_thw, Tensor)
    with torch.no_grad():
        expected_main, expected_deepstack = model.model.get_video_features(
            pixel_values_videos,
            video_grid_thw,
        )
    original_video = logits(model, video)
    wrapped_video = logits(wrapper, video)
    assert torch.equal(original_video, wrapped_video)
    captured = wrapper.last_visual_output
    assert captured is not None
    assert torch.equal(captured.packed_main_visual_embeddings(), torch.cat(expected_main))
    assert all(
        torch.equal(captured.deepstack_features[index], expected_deepstack[index])
        for index in range(3)
    )

    original_image = logits(model, image)
    wrapped_image = logits(wrapper, image)
    assert torch.equal(original_image, wrapped_image)
    assert wrapper.last_visual_output is None

    original_text = logits(model, text)
    wrapped_text = logits(wrapper, text)
    assert torch.equal(original_text, wrapped_text)
    assert wrapper.last_visual_output is None


def test_enabled_adapter_does_not_run_for_image_only_or_text_only() -> None:
    events: list[str] = []
    model = make_tiny_hf_model()
    wrapper = Qwen3VLAdapter(
        model,
        make_tiny_project_config(),
        TrainableScaleAdapter(events),
        adapter_enabled=True,
    )
    image = image_inputs()
    text = {"input_ids": torch.tensor([[1, 2, 3]]), "use_cache": False}

    assert torch.equal(logits(model, image), logits(wrapper, image))
    assert wrapper.last_visual_output is None
    assert torch.equal(logits(model, text), logits(wrapper, text))
    assert wrapper.last_visual_output is None
    assert events == []


def test_mixed_image_video_input_adapts_only_video_main_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_tiny_hf_model()
    owner = model.model
    inputs = mixed_inputs()
    pixel_values = inputs["pixel_values"]
    pixel_values_videos = inputs["pixel_values_videos"]
    image_grid_thw = inputs["image_grid_thw"]
    video_grid_thw = inputs["video_grid_thw"]
    assert isinstance(pixel_values, Tensor)
    assert isinstance(pixel_values_videos, Tensor)
    assert isinstance(image_grid_thw, Tensor)
    assert isinstance(video_grid_thw, Tensor)
    with torch.no_grad():
        expected_image = torch.cat(owner.get_image_features(pixel_values, image_grid_thw)[0])
        expected_video = torch.cat(
            owner.get_video_features(pixel_values_videos, video_grid_thw)[0]
        )

    captured: dict[str, Tensor] = {}
    original_get_placeholder_mask = owner.get_placeholder_mask

    def recording_get_placeholder_mask(
        self: object,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del self
        if image_features is not None:
            captured["image"] = image_features.detach().clone()
        if video_features is not None:
            captured["video"] = video_features.detach().clone()
        return original_get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )

    monkeypatch.setattr(
        owner,
        "get_placeholder_mask",
        MethodType(recording_get_placeholder_mask, owner),
    )
    wrapper = Qwen3VLAdapter(
        model,
        make_tiny_project_config(),
        AddOneAdapter(),
        adapter_enabled=True,
    )
    owner.rope_deltas = None

    wrapper(**inputs)

    assert torch.equal(captured["image"], expected_image)
    assert torch.equal(captured["video"], expected_video + 1)
    assert wrapper.last_visual_output is not None


def test_tiny_real_hf_order_and_deepstack_injection_are_unchanged() -> None:
    events: list[str] = []
    model = make_tiny_hf_model()
    owner = model.model
    captured_deepstack_ids: list[int] = []
    injected_deepstack_ids: list[int] = []
    injected_mask_ids: list[int] = []
    injected_masks: list[Tensor] = []
    original_get_video = owner.get_video_features
    original_deepstack_process = owner.language_model._deepstack_process

    def recording_get_video(
        self: object,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], list[Tensor]]:
        del self
        main, deepstack = original_get_video(pixel_values_videos, video_grid_thw)
        captured_deepstack_ids[:] = [id(feature) for feature in deepstack]
        return main, deepstack

    def recording_deepstack_process(
        self: object,
        hidden_states: Tensor,
        visual_pos_masks: Tensor,
        visual_embeds: Tensor,
    ) -> Tensor:
        del self
        injected_deepstack_ids.append(id(visual_embeds))
        injected_mask_ids.append(id(visual_pos_masks))
        injected_masks.append(visual_pos_masks.detach().clone())
        events.append(f"deepstack_{len(injected_deepstack_ids) - 1}")
        return original_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)

    owner.get_video_features = MethodType(recording_get_video, owner)
    owner.language_model._deepstack_process = MethodType(
        recording_deepstack_process,
        owner.language_model,
    )
    layer_handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, index=index: (
                events.append(f"layer_{index}"),
                output,
            )[1]
        )
        for index, layer in enumerate(owner.language_model.layers)
    ]
    merger_handle = owner.visual.merger.register_forward_hook(
        lambda _module, _inputs, output: (events.append("main_merger"), output)[1]
    )
    wrapper = Qwen3VLAdapter(
        model,
        make_tiny_project_config(),
        TrainableScaleAdapter(events),
        adapter_enabled=True,
    )
    owner.rope_deltas = None
    try:
        with MaskedScatterRecorder(events):
            wrapper(**video_inputs())
    finally:
        merger_handle.remove()
        for handle in layer_handles:
            handle.remove()

    assert events.index("main_merger") < events.index("adapter")
    assert events.index("adapter") < events.index("masked_scatter")
    decoder_events = [event for event in events if event.startswith(("layer_", "deepstack_"))]
    assert decoder_events == [
        "layer_0",
        "deepstack_0",
        "layer_1",
        "deepstack_1",
        "layer_2",
        "deepstack_2",
    ]
    assert injected_deepstack_ids == captured_deepstack_ids
    expected_mask = torch.tensor([[False, True, False, False]])
    assert all(torch.equal(mask, expected_mask) for mask in injected_masks)
    assert len(set(injected_mask_ids)) == 1


def test_frozen_qwen_stays_eval_while_gradient_reaches_video_adapter() -> None:
    model = make_tiny_hf_model()
    adapter = TrainableScaleAdapter()
    wrapper = Qwen3VLAdapter(
        model,
        make_tiny_project_config(),
        adapter,
        adapter_enabled=True,
    )
    wrapper.train()
    model.model.rope_deltas = None
    output = wrapper(**video_inputs())

    output.logits.square().sum().backward()

    assert model.training is False
    assert adapter.training is True
    assert adapter.scale.grad is not None
    assert torch.isfinite(adapter.scale.grad)
    assert adapter.scale.grad.abs().item() > 0.0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_inner_feature_owner_is_not_registered_twice() -> None:
    wrapper = Qwen3VLAdapter(make_tiny_hf_model(), make_tiny_project_config())

    assert set(wrapper._modules) == {"qwen_model", "video_boundary"}
    assert wrapper.feature_owner is wrapper.qwen_model.model
    assert all(not key.startswith("feature_owner.") for key in wrapper.state_dict())


def test_generate_adapts_video_only_during_prefill() -> None:
    events: list[str] = []
    model = make_tiny_hf_model()
    wrapper = Qwen3VLAdapter(
        model,
        make_tiny_project_config(),
        TrainableScaleAdapter(events),
        adapter_enabled=True,
    )
    inputs = video_inputs()
    inputs["use_cache"] = True

    generated = wrapper.generate(**inputs, max_new_tokens=2, do_sample=False)

    assert isinstance(generated, Tensor)
    assert generated.shape == (1, 6)
    assert events == ["adapter"]
    assert wrapper.last_visual_output is not None


def test_hook_and_capture_are_restored_after_upstream_failure() -> None:
    model = make_tiny_hf_model()
    owner = model.model
    wrapper = Qwen3VLAdapter(model, make_tiny_project_config())
    assert "get_video_features" not in vars(owner)
    invalid = video_inputs()
    invalid["input_ids"] = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)

    with pytest.raises(ValueError, match="Videos features and video tokens do not match"):
        wrapper(**invalid)

    assert "get_video_features" not in vars(owner)
    assert wrapper.last_visual_output is None
    owner.rope_deltas = None
    wrapper(**video_inputs())
    assert "get_video_features" not in vars(owner)
    assert wrapper.last_visual_output is not None


def test_direct_video_feature_call_is_rejected_while_forward_hook_is_active() -> None:
    model = make_tiny_hf_model()
    wrapper = Qwen3VLAdapter(model, make_tiny_project_config())
    inputs = video_inputs()
    pixel_values_videos = inputs["pixel_values_videos"]
    video_grid_thw = inputs["video_grid_thw"]
    assert isinstance(pixel_values_videos, Tensor)
    assert isinstance(video_grid_thw, Tensor)

    with (
        wrapper._patched_video_features(),
        pytest.raises(RuntimeError, match="while the forward hook is active"),
    ):
        wrapper.get_video_features(pixel_values_videos, video_grid_thw)


def test_direct_video_feature_failure_clears_stale_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_tiny_hf_model()
    owner = model.model
    wrapper = Qwen3VLAdapter(model, make_tiny_project_config())
    inputs = video_inputs()
    pixel_values_videos = inputs["pixel_values_videos"]
    video_grid_thw = inputs["video_grid_thw"]
    assert isinstance(pixel_values_videos, Tensor)
    assert isinstance(video_grid_thw, Tensor)
    wrapper.get_video_features(pixel_values_videos, video_grid_thw)
    assert wrapper.last_visual_output is not None

    def failing_get_video_features(
        self: object,
        pixels: Tensor,
        grid: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], list[Tensor]]:
        del self, pixels, grid
        raise RuntimeError("synthetic owner failure")

    monkeypatch.setattr(
        owner,
        "get_video_features",
        MethodType(failing_get_video_features, owner),
    )

    with pytest.raises(RuntimeError, match="synthetic owner failure"):
        wrapper.get_video_features(pixel_values_videos, video_grid_thw)
    assert wrapper.last_visual_output is None


def test_checkpoint_loader_guards_and_legacy_interface_scan(tmp_path: Path) -> None:
    project = load_config()
    assert_qwen_checkpoint_config(make_official_checkpoint_config(), project)
    bad = make_official_checkpoint_config()
    bad.vision_config.out_hidden_size = 3584
    with pytest.raises(ValueError, match="out_hidden_size"):
        assert_qwen_checkpoint_config(bad, project)
    with pytest.raises(ValueError, match="ProjectConfig"):
        build_qwen_adapter()
    with pytest.raises(FileNotFoundError, match="local Qwen"):
        build_qwen_adapter(project, tmp_path / "missing")

    legacy_name = "pooler_" + "output"
    code_files: Iterator[Path] = iter(
        [
            *(ROOT / "src" / "ttt_svcbench_qwen").glob("*.py"),
            *(ROOT / "tests").glob("*.py"),
        ]
    )
    assert all(legacy_name not in path.read_text(encoding="utf-8") for path in code_files)


@pytest.mark.parametrize(
    ("owner_name", "field_name", "invalid_value"),
    [
        ("vision_config", "depth", 26),
        ("vision_config", "hidden_size", 1024),
        ("vision_config", "num_heads", 12),
        ("vision_config", "in_channels", 1),
        ("vision_config", "patch_size", 14),
        ("vision_config", "temporal_patch_size", 1),
        ("vision_config", "spatial_merge_size", 4),
        ("vision_config", "out_hidden_size", 3584),
        ("vision_config", "deepstack_visual_indexes", [7, 15, 23]),
        ("text_config", "hidden_size", 3584),
        ("text_config", "num_hidden_layers", 32),
    ],
)
def test_every_pinned_checkpoint_field_fails_fast(
    owner_name: str,
    field_name: str,
    invalid_value: object,
) -> None:
    checkpoint = make_official_checkpoint_config()
    setattr(getattr(checkpoint, owner_name), field_name, invalid_value)

    with pytest.raises(ValueError, match=field_name):
        assert_qwen_checkpoint_config(checkpoint, load_config())


def test_transformers_version_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    version_owner = assert_qwen_checkpoint_config.__globals__["transformers"]
    monkeypatch.setattr(version_owner, "__version__", "0.0.0")

    with pytest.raises(ValueError, match="Transformers version mismatch"):
        assert_qwen_checkpoint_config(make_official_checkpoint_config(), load_config())


def test_loader_forces_local_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "tiny-local-checkpoint"
    model_root.mkdir()
    tiny_model = make_tiny_hf_model()
    captured: dict[str, object] = {}
    load_order: list[str] = []

    def fake_config_from_pretrained(
        checkpoint: str | Path,
        **kwargs: object,
    ) -> Qwen3VLConfig:
        load_order.append("config")
        captured["config_checkpoint"] = checkpoint
        captured["config_kwargs"] = kwargs
        return tiny_model.config

    def fake_from_pretrained(
        checkpoint: str | Path,
        **kwargs: object,
    ) -> Qwen3VLForConditionalGeneration:
        load_order.append("weights")
        captured["weight_checkpoint"] = checkpoint
        captured["weight_kwargs"] = kwargs
        return tiny_model

    monkeypatch.setattr(
        Qwen3VLConfig,
        "from_pretrained",
        staticmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    wrapper = build_qwen_adapter(make_tiny_project_config(), model_root)

    assert wrapper.qwen_model is tiny_model
    assert load_order == ["config", "weights"]
    assert Path(captured["config_checkpoint"]) == model_root
    assert captured["config_kwargs"] == {"local_files_only": True}
    assert Path(captured["weight_checkpoint"]) == model_root
    assert captured["weight_kwargs"] == {
        "local_files_only": True,
        "dtype": "auto",
        "device_map": "auto",
    }


def test_invalid_local_config_is_rejected_before_weight_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "invalid-local-checkpoint"
    model_root.mkdir()
    invalid_config = make_tiny_hf_config()
    invalid_config.vision_config.depth = 2
    weight_loader_called = False

    def fake_config_from_pretrained(
        checkpoint: str | Path,
        **kwargs: object,
    ) -> Qwen3VLConfig:
        assert Path(checkpoint) == model_root
        assert kwargs == {"local_files_only": True}
        return invalid_config

    def forbidden_weight_loader(
        checkpoint: str | Path,
        **kwargs: object,
    ) -> Qwen3VLForConditionalGeneration:
        del checkpoint, kwargs
        nonlocal weight_loader_called
        weight_loader_called = True
        raise AssertionError("weights must not load after config preflight failure")

    monkeypatch.setattr(
        Qwen3VLConfig,
        "from_pretrained",
        staticmethod(fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        Qwen3VLForConditionalGeneration,
        "from_pretrained",
        staticmethod(forbidden_weight_loader),
    )

    with pytest.raises(ValueError, match="vision.depth"):
        build_qwen_adapter(make_tiny_project_config(), model_root)
    assert weight_loader_called is False


def make_official_checkpoint_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        vision_config={
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4096,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [8, 16, 24],
        },
        text_config={
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 22016,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
    )


def test_video_batch_rejects_packed_patch_count_mismatch() -> None:
    with pytest.raises(ValueError, match="packed patch count"):
        VideoBatch(
            pixel_values_videos=torch.zeros(7, 1536),
            video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
            timestamps=torch.zeros(1, 2),
            query_time=torch.zeros(1),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            video_ids=("video",),
            trajectory_ids=("trajectory",),
        )


@pytest.mark.parametrize("invalid_grid_value", [0, -1])
def test_video_batch_rejects_non_positive_grid(invalid_grid_value: int) -> None:
    with pytest.raises(ValueError, match="entries must be positive"):
        VideoBatch(
            pixel_values_videos=torch.zeros(8, 1536),
            video_grid_thw=torch.tensor(
                [[invalid_grid_value, 2, 2]],
                dtype=torch.int64,
            ),
            timestamps=torch.zeros(1, 2),
            query_time=torch.zeros(1),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            video_ids=("video",),
            trajectory_ids=("trajectory",),
        )


def test_video_contracts_reject_empty_batches_and_scalar_grids() -> None:
    with pytest.raises(ValueError, match="at least one video"):
        VideoBatch(
            pixel_values_videos=torch.zeros(0, 1536),
            video_grid_thw=torch.empty(0, 3, dtype=torch.int64),
            timestamps=torch.empty(0, 2),
            query_time=torch.empty(0),
            valid_mask=torch.empty(0, 2, dtype=torch.bool),
            video_ids=(),
            trajectory_ids=(),
        )
    with pytest.raises(ValueError, match=r"\[B, 3\]"):
        MergedVideoMetadata(
            video_grid_thw=torch.tensor(1, dtype=torch.int64),
            merged_grid_thw=torch.tensor(1, dtype=torch.int64),
            spatial_merge_size=2,
            token_counts=(),
            token_offsets=(0,),
        )
