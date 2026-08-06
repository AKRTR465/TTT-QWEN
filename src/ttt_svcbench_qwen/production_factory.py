"""Read-only LLaMA-Factory integration for the independent TTT-QWEN project.

This module imports public loader/parser/Trainer symbols from an adjacent LLaMA-Factory checkout;
it never patches or writes that checkout.  Project-specific State-TTT assembly stays on this side
of the boundary.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import nn
from transformers.modeling_utils import load_sharded_checkpoint

from ttt_svcbench_qwen.config import ProjectConfig, load_config

DEFAULT_H200_PLAY_ROOT = Path(os.environ.get("TTT_H200_PLAY_ROOT", "play"))
DEFAULT_LLAMFACTORY_ROOT = DEFAULT_H200_PLAY_ROOT / "LLaMA-Factory"
DEFAULT_QWEN3_VL_8B_ROOT = DEFAULT_H200_PLAY_ROOT / "model/Qwen3-VL-8B-Instruct"


class QwenOuterTrainabilityConfig(BaseModel):  # type: ignore[misc]
    """Stage-local Qwen parameter policy applied after LLaMA-Factory model loading."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    mode: str = "full"
    vision_freeze_first_blocks: int = Field(default=0, ge=0)
    decoder_train_last_layers: int = Field(default=36, ge=0)
    train_vision_patch_embed: bool = True
    train_main_merger: bool = True
    train_deepstack_mergers: bool = True
    train_language_model_norm: bool = True
    train_input_embeddings: bool = True
    train_lm_head: bool = True


class ProductionTTTConfig(BaseModel):  # type: ignore[misc]
    """State-TTT extension for production LLaMA-Factory YAML files."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    stage: str
    a5_adaptation_mode: str = "meta_ttt"
    a5_phase: str = "main"
    warmup_bundle: str | None = None
    project_config: str = Field(min_length=1)
    dataset_manifest: str = Field(min_length=1)
    qwen_outer_trainability: QwenOuterTrainabilityConfig = Field(
        default_factory=QwenOuterTrainabilityConfig
    )
    initialize_from_a2_checkpoint: str | None = None
    support_prefetch_depth: int = Field(gt=0)
    support_decode_coalesce: bool
    support_materialization: str
    prepared_episode_max_bytes: int = Field(default=2_147_483_648, gt=0)
    support_visual_batch_size: int = Field(default=1, gt=0)
    query_encoder_reuse: bool = True
    query_frame_sampling: str = "llamafactory_uniform_cap"
    query_sample_fps: float = Field(default=2.0, gt=0.0)
    state_query_visual_mode: str
    state_query_max_frames: int
    answer_query_visual_mode: str
    answer_query_max_frames: int
    query_decode_max_groups: int = Field(default=16, ge=1, le=16)
    state_query_cache_mode: str
    answer_query_cache_mode: str
    query_activation_offload: bool = False
    preprocess_cache_mode: str
    preprocess_cache_root_env: str = Field(min_length=1)
    preprocess_cache_max_gb: float = Field(gt=0.0)
    preprocess_cache_dtype: str
    visual_cost_mode: str = "proxy"
    segment_prefetch_depth: int = 0

    def query_cache_enabled(self, role: Literal["state_query", "answer_query"]) -> bool:
        """Return the explicit role-specific cache policy."""

        mode = (
            self.state_query_cache_mode if role == "state_query" else self.answer_query_cache_mode
        )
        return mode == "inherit"

    @property
    def cached_query_roles(self) -> frozenset[str]:
        roles: tuple[Literal["state_query", "answer_query"], ...] = (
            "state_query",
            "answer_query",
        )
        return frozenset(role for role in roles if self.query_cache_enabled(role))


@dataclass(frozen=True, slots=True)
class LlamaFactorySymbols:
    get_train_args: Callable[..., tuple[Any, ...]]
    load_tokenizer: Callable[..., Mapping[str, object]]
    load_model: Callable[..., nn.Module]
    trainer_base: type


@dataclass(frozen=True, slots=True)
class LlamaFactoryBackboneBundle:
    model: nn.Module
    tokenizer: object
    processor: object | None
    model_args: object
    data_args: object
    training_args: object
    finetuning_args: object
    generating_args: object
    project_config: ProjectConfig
    ttt_config: ProductionTTTConfig
    symbols: LlamaFactorySymbols


@dataclass(frozen=True, slots=True)
class OuterCheckpointAudit:
    checkpoint: Path
    format: str
    tensor_count: int


def load_training_yaml(path: str | Path) -> tuple[dict[str, object], ProductionTTTConfig]:
    """Split native LLaMA-Factory keys from the namespaced ``ttt_qwen`` extension."""

    import yaml

    source = Path(path)
    text = os.path.expandvars(source.read_text(encoding="utf-8"))
    values = cast("dict[str, object]", yaml.safe_load(text))
    extension = cast("dict[str, object]", values.pop("ttt_qwen", None))
    project_config_override = os.environ.get("TTT_PROJECT_CONFIG")
    if project_config_override is not None:
        extension["project_config"] = project_config_override
    return values, ProductionTTTConfig.model_validate(extension)


def import_llamafactory(
    root: str | Path = DEFAULT_LLAMFACTORY_ROOT,
) -> LlamaFactorySymbols:
    """Import the exact public APIs verified on H200 commit 523f801."""

    checkout = Path(root).resolve()
    source = checkout / "src"
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    parser = importlib.import_module("llamafactory.hparams.parser")
    loader = importlib.import_module("llamafactory.model.loader")
    trainer = importlib.import_module("llamafactory.train.sft.trainer")
    return LlamaFactorySymbols(
        get_train_args=cast(Callable[..., tuple[Any, ...]], parser.get_train_args),
        load_tokenizer=cast(Callable[..., Mapping[str, object]], loader.load_tokenizer),
        load_model=cast(Callable[..., nn.Module], loader.load_model),
        trainer_base=cast(type, trainer.CustomSeq2SeqTrainer),
    )


def load_llamafactory_backbone(
    yaml_path: str | Path,
    *,
    llamafactory_root: str | Path = DEFAULT_LLAMFACTORY_ROOT,
    project_config_path: str | Path | None = None,
) -> LlamaFactoryBackboneBundle:
    """Parse LF arguments and load its tokenizer, processor, and trainable Qwen model."""

    native, ttt_config = load_training_yaml(yaml_path)
    symbols = import_llamafactory(llamafactory_root)
    parsed = symbols.get_train_args(native)
    model_args, data_args, training_args, finetuning_args, generating_args = parsed
    tokenizer_module = symbols.load_tokenizer(model_args)
    tokenizer = tokenizer_module.get("tokenizer")
    processor = tokenizer_module.get("processor")
    model = symbols.load_model(tokenizer, model_args, finetuning_args, True)
    configured_project_path = ttt_config.project_config
    if project_config_path is None:
        project_config_path = configured_project_path
    config = load_config(project_config_path)
    return LlamaFactoryBackboneBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        project_config=config,
        ttt_config=ttt_config,
        symbols=symbols,
    )


def configure_qwen_outer_trainability(
    model: nn.Module,
    config: ProjectConfig,
    policy: QwenOuterTrainabilityConfig,
) -> None:
    """Apply one exact full or partial Outer-AdamW policy to Qwen3-VL."""

    owner = getattr(model, "model", model)
    visual = _resolve_path(owner, "visual")
    language_model = _resolve_path(owner, "language_model")
    vision_blocks = _module_sequence(_resolve_path(visual, "blocks"), "visual.blocks")
    decoder_layers = _module_sequence(
        _resolve_path(language_model, "layers"),
        "language_model.layers",
    )

    if policy.mode == "full":
        model.requires_grad_(True)
    elif policy.mode == "frozen":
        model.requires_grad_(False)
    else:
        model.requires_grad_(False)
        trainable_modules: list[nn.Module] = []
        trainable_modules.extend(vision_blocks[policy.vision_freeze_first_blocks :])
        trainable_modules.extend(decoder_layers[-policy.decoder_train_last_layers :])
        if policy.train_vision_patch_embed:
            trainable_modules.append(cast(nn.Module, _resolve_path(visual, "patch_embed")))
        if policy.train_main_merger:
            trainable_modules.append(cast(nn.Module, _resolve_path(visual, "merger")))
        if policy.train_deepstack_mergers:
            trainable_modules.extend(
                _module_sequence(
                    _resolve_path(visual, "deepstack_merger_list"),
                    "visual.deepstack_merger_list",
                )
            )
        if policy.train_language_model_norm:
            trainable_modules.append(cast(nn.Module, _resolve_path(language_model, "norm")))
        input_embeddings = _optional_module(language_model, "embed_tokens")
        lm_head = _optional_module(model, "lm_head")
        if policy.train_input_embeddings and input_embeddings is not None:
            trainable_modules.append(input_embeddings)
        if policy.train_lm_head and lm_head is not None:
            trainable_modules.append(lm_head)
        for module in trainable_modules:
            module.requires_grad_(True)


def load_outer_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
) -> OuterCheckpointAudit:
    """Load an outer-only single or sharded safetensors checkpoint."""

    source = Path(checkpoint).resolve()
    if source.is_dir():
        single = source / "model.safetensors"
        index = source / "model.safetensors.index.json"
        source = single if single.is_file() else index
    state: dict[str, torch.Tensor] | None = None
    if source.suffix == ".safetensors":
        state = load_file(str(source), device="cpu")
        loaded = set(state)
        checkpoint_format = "safetensors"
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        weight_map = cast("dict[str, str]", raw["weight_map"])
        loaded = set(weight_map)
        checkpoint_format = "sharded_safetensors"
    if state is not None:
        model.load_state_dict(state, strict=True)
    else:
        loader = cast(Callable[..., Any], load_sharded_checkpoint)
        loader(model, str(source.parent), strict=True, prefer_safe=True)
    return OuterCheckpointAudit(
        checkpoint=source,
        format=checkpoint_format,
        tensor_count=len(loaded),
    )


def initialize_outer_model_from_a2(
    model: nn.Module,
    checkpoint: str | Path,
) -> OuterCheckpointAudit:
    """Load exact-current or narrowly profiled legacy A2 weights for A5 initialization."""

    root = Path(checkpoint).resolve()
    safe_index = root / "model.safetensors.index.json"
    torch_index = root / "pytorch_model.bin.index.json"
    safe_weights = root / "model.safetensors"
    torch_weights = root / "pytorch_model.bin"
    if safe_index.is_file() or torch_index.is_file():
        load_sharded_checkpoint(model, str(root), strict=False, prefer_safe=True)
        index_path = safe_index if safe_index.is_file() else torch_index
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = cast("dict[str, str]", index["weight_map"])
        loaded_keys = set(weight_map)
        checkpoint_format = "sharded_safetensors" if safe_index.is_file() else "sharded_torch"
    elif safe_weights.is_file():
        state = load_file(str(safe_weights), device="cpu")
        model.load_state_dict(state, strict=False)
        loaded_keys = set(state)
        checkpoint_format = "safetensors"
    else:
        raw = torch.load(torch_weights, map_location="cpu", weights_only=True)
        state = cast("dict[str, torch.Tensor]", raw)
        model.load_state_dict(state, strict=False)
        loaded_keys = set(state)
        checkpoint_format = "torch"
    return OuterCheckpointAudit(
        checkpoint=root,
        format=checkpoint_format,
        tensor_count=len(loaded_keys),
    )


def _resolve_path(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            raise ValueError(f"Qwen runtime is missing required path: {path}")
    return current


def _module_sequence(value: object, path: str) -> tuple[nn.Module, ...]:
    return tuple(cast("list[nn.Module]", value))


def _optional_module(value: object, name: str) -> nn.Module | None:
    return cast("nn.Module | None", getattr(value, name, None))


__all__ = [
    "DEFAULT_H200_PLAY_ROOT",
    "DEFAULT_LLAMFACTORY_ROOT",
    "DEFAULT_QWEN3_VL_8B_ROOT",
    "LlamaFactoryBackboneBundle",
    "LlamaFactorySymbols",
    "OuterCheckpointAudit",
    "ProductionTTTConfig",
    "QwenOuterTrainabilityConfig",
    "configure_qwen_outer_trainability",
    "import_llamafactory",
    "initialize_outer_model_from_a2",
    "load_llamafactory_backbone",
    "load_training_yaml",
]
