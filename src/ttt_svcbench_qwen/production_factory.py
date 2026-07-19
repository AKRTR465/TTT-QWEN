"""Read-only LLaMA-Factory integration for the independent TTT-QWEN project.

This module imports public loader/parser/Trainer symbols from an adjacent LLaMA-Factory checkout;
it never patches or writes that checkout.  Project-specific State-TTT assembly stays on this side
of the boundary.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from safetensors.torch import load_file
from torch import nn
from transformers.modeling_utils import load_sharded_checkpoint

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.qwen_adapter import assert_qwen_runtime_structure

DEFAULT_H200_PLAY_ROOT = Path(os.environ.get("TTT_H200_PLAY_ROOT", "play"))
DEFAULT_LLAMFACTORY_ROOT = DEFAULT_H200_PLAY_ROOT / "LLaMA-Factory"
DEFAULT_QWEN3_VL_8B_ROOT = DEFAULT_H200_PLAY_ROOT / "model/Qwen3-VL-8B-Instruct"
VERIFIED_LLAMFACTORY_COMMIT = "523f801"
_UNRESOLVED_ENV = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")
_FORBIDDEN_CHECKPOINT_TOKENS = (
    "transient_w_t",
    "state_bank_runtime",
    "identity_bank_runtime",
    "fsm_runtime",
    "temporal_cache",
    "visual_cache",
    "soft_overlap_snapshot",
)


@dataclass(frozen=True, slots=True)
class LlamaFactoryCheckoutAudit:
    root: Path
    commit: str
    dirty: bool
    imported_without_checkout_write: bool

    def __post_init__(self) -> None:
        if not self.root.is_dir() or not self.commit:
            raise ValueError("LLaMA-Factory checkout audit is incomplete")
        if not self.imported_without_checkout_write:
            raise ValueError("LLaMA-Factory integration may not mutate its checkout")


@dataclass(frozen=True, slots=True)
class LlamaFactorySymbols:
    get_train_args: Callable[..., tuple[Any, ...]]
    load_tokenizer: Callable[..., Mapping[str, object]]
    load_model: Callable[..., nn.Module]
    trainer_base: type
    checkout: LlamaFactoryCheckoutAudit


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
    ttt_config: Mapping[str, object]
    symbols: LlamaFactorySymbols


@dataclass(frozen=True, slots=True)
class FullUnfreezeAudit:
    total_parameters: int
    trainable_parameters: int
    vision_parameters: int
    merger_parameters: int
    deepstack_merger_parameters: int
    decoder_parameters: int
    decoder_layer_count: int
    all_qwen_parameters_trainable: bool

    def __post_init__(self) -> None:
        counts = (
            self.total_parameters,
            self.trainable_parameters,
            self.vision_parameters,
            self.merger_parameters,
            self.deepstack_merger_parameters,
            self.decoder_parameters,
            self.decoder_layer_count,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("full-unfreeze audit requires positive parameter/layer counts")
        if self.total_parameters != self.trainable_parameters:
            raise ValueError("production A2/A5 requires every Qwen parameter trainable")
        if self.decoder_layer_count != 36 or not self.all_qwen_parameters_trainable:
            raise ValueError("production Qwen audit requires all 36 Decoder layers trainable")


@dataclass(frozen=True, slots=True)
class OuterCheckpointAudit:
    checkpoint: Path
    format: str
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    forbidden_runtime_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.checkpoint.is_dir() or not self.format or self.tensor_count <= 0:
            raise ValueError("A2 checkpoint audit is incomplete")
        if self.missing_keys or self.unexpected_keys or self.forbidden_runtime_keys:
            raise ValueError("A2 checkpoint does not exactly match the outer model boundary")


class RuntimeFactory(Protocol):
    def __call__(
        self,
        backbone: LlamaFactoryBackboneBundle,
        config: Mapping[str, object],
    ) -> object: ...


def load_training_yaml(path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    """Split native LLaMA-Factory keys from the namespaced ``ttt_qwen`` extension."""

    import yaml

    source = Path(path)
    text = os.path.expandvars(source.read_text(encoding="utf-8"))
    unresolved = tuple(sorted(set(_UNRESOLVED_ENV.findall(text))))
    if unresolved:
        raise ValueError(f"training YAML contains unresolved environment variables: {unresolved}")
    raw = cast(object, yaml.safe_load(text))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("training YAML must contain one string-keyed object")
    values = cast(dict[str, object], raw)
    extension = values.pop("ttt_qwen", None)
    if not isinstance(extension, dict) or not all(isinstance(key, str) for key in extension):
        raise ValueError("training YAML requires a string-keyed ttt_qwen section")
    return values, cast(dict[str, object], extension)


def import_llamafactory(
    root: str | Path = DEFAULT_LLAMFACTORY_ROOT,
    *,
    expected_commit: str = VERIFIED_LLAMFACTORY_COMMIT,
) -> LlamaFactorySymbols:
    """Import the exact public APIs verified on H200 commit 523f801."""

    checkout = Path(root).resolve()
    source = checkout / "src"
    if not (source / "llamafactory").is_dir():
        raise FileNotFoundError(f"LLaMA-Factory Python package not found under {source}")
    commit = _git_output(checkout, "rev-parse", "--short", "HEAD")
    if expected_commit and commit != expected_commit:
        raise ValueError(f"LLaMA-Factory commit drift: expected {expected_commit}, found {commit}")
    dirty = bool(_git_output(checkout, "status", "--short"))
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
        checkout=LlamaFactoryCheckoutAudit(
            root=checkout,
            commit=commit,
            dirty=dirty,
            imported_without_checkout_write=True,
        ),
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
    if len(parsed) != 5:
        raise ValueError("LLaMA-Factory get_train_args must return five argument groups")
    model_args, data_args, training_args, finetuning_args, generating_args = parsed
    tokenizer_module = symbols.load_tokenizer(model_args)
    tokenizer = tokenizer_module.get("tokenizer")
    processor = tokenizer_module.get("processor")
    if tokenizer is None:
        raise ValueError("LLaMA-Factory tokenizer loader returned no tokenizer")
    model = symbols.load_model(tokenizer, model_args, finetuning_args, True)
    if not isinstance(model, nn.Module):
        raise TypeError("LLaMA-Factory model loader returned a non-module")
    configured_project_path = ttt_config.get("project_config")
    if project_config_path is None:
        if not isinstance(configured_project_path, str) or not configured_project_path:
            raise ValueError("ttt_qwen.project_config is required")
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


def fully_unfreeze_qwen(model: nn.Module, config: ProjectConfig) -> FullUnfreezeAudit:
    """Enable ViT, Main/DeepStack mergers, and all 36 Decoder layers for Outer AdamW."""

    owner = getattr(model, "model", model)
    assert_qwen_runtime_structure(owner, config)
    model.requires_grad_(True)
    named = tuple(model.named_parameters())
    if not named:
        raise ValueError("Qwen model exposes no parameters")
    decoder_layers = _resolve_path(owner, "language_model.layers")
    if not isinstance(decoder_layers, (list, tuple, nn.ModuleList)):
        raise TypeError("Qwen language_model.layers must be a module sequence")
    decoder_layer_count = len(decoder_layers)
    groups = {
        "vision": _parameter_count(_resolve_path(owner, "visual")),
        "merger": _parameter_count(_resolve_path(owner, "visual.merger")),
        "deepstack": _parameter_count(_resolve_path(owner, "visual.deepstack_merger_list")),
        "decoder": _parameter_count(_resolve_path(owner, "language_model.layers")),
    }
    return FullUnfreezeAudit(
        total_parameters=sum(parameter.numel() for _, parameter in named),
        trainable_parameters=sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        vision_parameters=groups["vision"],
        merger_parameters=groups["merger"],
        deepstack_merger_parameters=groups["deepstack"],
        decoder_parameters=groups["decoder"],
        decoder_layer_count=decoder_layer_count,
        all_qwen_parameters_trainable=all(parameter.requires_grad for _, parameter in named),
    )


def audit_outer_checkpoint_boundary(model: nn.Module) -> tuple[str, ...]:
    """Fail if transient/hard runtime state was accidentally registered on the model."""

    keys = tuple(model.state_dict())
    if not keys:
        raise ValueError("outer model exposes no checkpoint state")
    forbidden = tuple(
        name
        for name in keys
        if any(token in name.casefold() for token in _FORBIDDEN_CHECKPOINT_TOKENS)
    )
    if forbidden:
        raise ValueError(f"outer checkpoint contains transient/hard runtime keys: {forbidden}")
    return keys


def initialize_outer_model_from_a2(
    model: nn.Module,
    checkpoint: str | Path,
) -> OuterCheckpointAudit:
    """Load A2 weights only, leaving A5 optimizer/scheduler/RNG freshly initialized."""

    root = Path(checkpoint).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"A2 checkpoint directory does not exist: {root}")
    expected_keys = set(audit_outer_checkpoint_boundary(model))
    safe_index = root / "model.safetensors.index.json"
    torch_index = root / "pytorch_model.bin.index.json"
    safe_weights = root / "model.safetensors"
    torch_weights = root / "pytorch_model.bin"
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    if safe_index.is_file() or torch_index.is_file():
        result = load_sharded_checkpoint(model, str(root), strict=True, prefer_safe=True)
        missing = tuple(result.missing_keys)
        unexpected = tuple(result.unexpected_keys)
        index_path = safe_index if safe_index.is_file() else torch_index
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("A2 sharded checkpoint index has no weight_map")
        loaded_keys = set(weight_map)
        checkpoint_format = "sharded_safetensors" if safe_index.is_file() else "sharded_torch"
    elif safe_weights.is_file():
        state = load_file(str(safe_weights), device="cpu")
        result = model.load_state_dict(state, strict=True)
        missing = tuple(result.missing_keys)
        unexpected = tuple(result.unexpected_keys)
        loaded_keys = set(state)
        checkpoint_format = "safetensors"
    elif torch_weights.is_file():
        raw = torch.load(torch_weights, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("A2 torch checkpoint must contain a string-keyed state dict")
        state = cast(dict[str, torch.Tensor], raw)
        result = model.load_state_dict(state, strict=True)
        missing = tuple(result.missing_keys)
        unexpected = tuple(result.unexpected_keys)
        loaded_keys = set(state)
        checkpoint_format = "torch"
    else:
        raise FileNotFoundError(
            "A2 checkpoint has no model.safetensors[/index] or pytorch_model.bin[/index]"
        )
    forbidden = tuple(
        name
        for name in loaded_keys
        if any(token in name.casefold() for token in _FORBIDDEN_CHECKPOINT_TOKENS)
    )
    if loaded_keys != expected_keys:
        missing = tuple(sorted(expected_keys - loaded_keys))
        unexpected = tuple(sorted(loaded_keys - expected_keys))
    return OuterCheckpointAudit(
        checkpoint=root,
        format=checkpoint_format,
        tensor_count=len(loaded_keys),
        missing_keys=missing,
        unexpected_keys=unexpected,
        forbidden_runtime_keys=forbidden,
    )


def resolve_runtime_factory(specification: str) -> RuntimeFactory:
    """Resolve ``module:function`` without allowing an implicit remote checkout import."""

    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runtime_factory must use module:function syntax")
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise TypeError(f"runtime factory is not callable: {specification}")
    return cast(RuntimeFactory, value)


def environment_manifest(bundle: LlamaFactoryBackboneBundle) -> dict[str, object]:
    return {
        "llamafactory_root": str(bundle.symbols.checkout.root),
        "llamafactory_commit": bundle.symbols.checkout.commit,
        "llamafactory_dirty": bundle.symbols.checkout.dirty,
        "qwen_model_path": str(getattr(bundle.model_args, "model_name_or_path", "")),
        "project_spec_version": bundle.project_config.spec_version,
        "ttt_config": json.loads(json.dumps(dict(bundle.ttt_config))),
    }


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_path(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            raise ValueError(f"Qwen runtime is missing required path: {path}")
    return current


def _parameter_count(value: object) -> int:
    if isinstance(value, nn.Module):
        return sum(parameter.numel() for parameter in value.parameters())
    if isinstance(value, (list, tuple, nn.ModuleList)):
        return sum(_parameter_count(item) for item in value)
    raise TypeError("Qwen parameter audit path is not a module/module sequence")


__all__ = [
    "DEFAULT_H200_PLAY_ROOT",
    "DEFAULT_LLAMFACTORY_ROOT",
    "DEFAULT_QWEN3_VL_8B_ROOT",
    "FullUnfreezeAudit",
    "LlamaFactoryBackboneBundle",
    "LlamaFactoryCheckoutAudit",
    "LlamaFactorySymbols",
    "OuterCheckpointAudit",
    "audit_outer_checkpoint_boundary",
    "environment_manifest",
    "fully_unfreeze_qwen",
    "import_llamafactory",
    "initialize_outer_model_from_a2",
    "load_llamafactory_backbone",
    "load_training_yaml",
    "resolve_runtime_factory",
]
