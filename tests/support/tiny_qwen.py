"""Shared tiny Qwen3-VL scaffolding for the CPU adapter and integration suites."""

from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from ttt_svcbench_qwen.config import ProjectConfig, load_config


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


def make_tiny_hf_model(*, seed: int) -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(make_tiny_hf_config()).eval()
