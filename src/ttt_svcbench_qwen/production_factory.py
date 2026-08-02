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
from typing import Any, Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    "optimizer_runtime",
    "reader_audit",
)


@dataclass(frozen=True, slots=True)
class _LegacyA2ToA5Profile:
    """The sole tolerated historical checkpoint shape at the A2-to-A5 boundary.

    Profile ``a2_to_a5_memory_v1``: schema-12 A2 checkpoints predate the slot
    memory, so its interface tensors (probe, value projection, eta gate, read
    gate, forget gate, contract buffer) may be missing and are initialized
    fresh; pre-schema-14 checkpoints additionally predate the O2 relevance
    projection.  The retired ``associative_contract_version`` buffer they still
    carry is the sole tolerated unexpected non-module key.  Any other mismatch
    rejects the checkpoint.
    """

    missing_fragments: tuple[str, ...] = (
        ".p_context.",
        ".memory_key_probe.",
        ".memory_value_projection.",
        ".memory_eta_gate_hidden.",
        ".memory_eta_gate_output.",
        ".memory_alpha",
        ".memory_beta_raw",
        "memory_contract_version",
        ".relevance_projection.",
    )
    unexpected_modules: tuple[str, ...] = ("predictor.", "p_value.")
    unexpected_fragments: tuple[str, ...] = ("associative_contract_version",)

    def allows_missing(self, key: str) -> bool:
        return any(fragment in key for fragment in self.missing_fragments)

    def allows_unexpected(self, key: str) -> bool:
        if any(fragment in key for fragment in self.unexpected_fragments):
            return True
        return any(
            key.startswith(module) or f".{module}" in key
            for module in self.unexpected_modules
        )


_LEGACY_A2_TO_A5 = _LegacyA2ToA5Profile()


class QwenOuterTrainabilityConfig(BaseModel):  # type: ignore[misc]
    """Stage-local Qwen parameter policy applied after LLaMA-Factory model loading."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["full", "partial", "frozen"] = "full"
    vision_freeze_first_blocks: int = Field(default=0, ge=0)
    decoder_train_last_layers: int = Field(default=36, ge=0)
    train_vision_patch_embed: bool = True
    train_main_merger: bool = True
    train_deepstack_mergers: bool = True
    train_language_model_norm: bool = True
    train_input_embeddings: bool = True
    train_lm_head: bool = True

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_canonical_policy(self) -> Self:
        if self.mode == "full":
            expected = (0, 36, True, True, True, True, True, True)
            actual = (
                self.vision_freeze_first_blocks,
                self.decoder_train_last_layers,
                self.train_vision_patch_embed,
                self.train_main_merger,
                self.train_deepstack_mergers,
                self.train_language_model_norm,
                self.train_input_embeddings,
                self.train_lm_head,
            )
            if actual != expected:
                raise ValueError(
                    "full Qwen trainability must use the canonical all-trainable policy"
                )
        elif self.mode == "frozen":
            expected = (27, 0, False, False, False, False, False, False)
            actual = (
                self.vision_freeze_first_blocks,
                self.decoder_train_last_layers,
                self.train_vision_patch_embed,
                self.train_main_merger,
                self.train_deepstack_mergers,
                self.train_language_model_norm,
                self.train_input_embeddings,
                self.train_lm_head,
            )
            if actual != expected:
                raise ValueError(
                    "frozen Qwen trainability must use the canonical all-frozen policy"
                )
        return self


class ProductionTTTConfig(BaseModel):  # type: ignore[misc]
    """Strict State-TTT extension for production LLaMA-Factory YAML files."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    stage: Literal["a2", "a5"]
    a5_adaptation_mode: Literal["meta_ttt", "no_write"] = "meta_ttt"
    a5_phase: Literal["fast_state_warmup", "main"] = "main"
    warmup_bundle: str | None = Field(default=None, min_length=1)
    project_config: str = Field(min_length=1)
    dataset_manifest: str = Field(min_length=1)
    qwen_outer_trainability: QwenOuterTrainabilityConfig = Field(
        default_factory=QwenOuterTrainabilityConfig
    )
    visual_cost_index: str | None = Field(default=None, min_length=1)
    initialize_from_a2_checkpoint: str | None = Field(default=None, min_length=1)
    support_prefetch_depth: int = Field(gt=0)
    support_decode_coalesce: bool
    support_materialization: Literal["dataloader_episode", "segment_double_buffer"]
    prepared_episode_max_bytes: int = Field(default=2_147_483_648, gt=0)
    support_visual_batch_size: int = Field(default=1, gt=0)
    query_encoder_reuse: bool = True
    query_frame_sampling: Literal["llamafactory_uniform_cap"] = "llamafactory_uniform_cap"
    query_sample_fps: float = Field(default=2.0, gt=0.0)
    state_query_visual_mode: Literal["recent_chunk"]
    state_query_max_frames: Literal[16]
    answer_query_visual_mode: Literal["causal_prefix"]
    answer_query_max_frames: Literal[256]
    query_decode_max_groups: int = Field(default=16, ge=1, le=16)
    state_query_cache_mode: Literal["disabled", "inherit"]
    answer_query_cache_mode: Literal["disabled", "inherit"]
    query_activation_offload: bool = False
    preprocess_cache_mode: Literal["disabled", "read_write", "readonly"]
    preprocess_cache_miss_policy: Literal["decode", "error"]
    preprocess_cache_root_env: str = Field(min_length=1)
    preprocess_cache_max_gb: float = Field(gt=0.0)
    preprocess_cache_dtype: Literal["float32", "float16"]
    visual_cost_mode: Literal["proxy", "exact_tokens", "exact_tokens_then_runtime"] = "proxy"
    runtime_trace_mode: Literal["off", "cuda"] = "off"
    runtime_trace_dir: str | None = Field(default=None, min_length=1)
    segment_prefetch_depth: Literal[0, 1] = 0
    semantic_projector_delta_audit_steps: int = Field(default=0, ge=0)
    a5_parameter_delta_audit_steps: int = Field(default=0, ge=0)
    operator_diagnostics_interval: int = Field(default=10, gt=0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_stage_checkpoint(self) -> Self:
        if self.stage == "a2" and self.initialize_from_a2_checkpoint is not None:
            raise ValueError("A2 must not initialize from an A2 checkpoint")
        if self.stage == "a2" and self.a5_adaptation_mode != "meta_ttt":
            raise ValueError("a5_adaptation_mode applies only to A5")
        if self.stage == "a2" and (
            self.a5_phase != "main" or self.warmup_bundle is not None
        ):
            raise ValueError("A5 phase/bundle fields apply only to A5")
        if self.stage == "a5" and self.initialize_from_a2_checkpoint is None:
            raise ValueError("A5 requires initialize_from_a2_checkpoint")
        if self.stage == "a2" and self.qwen_outer_trainability.mode != "full":
            raise ValueError("A2 requires full Qwen outer trainability")
        if self.stage == "a5" and self.a5_adaptation_mode == "no_write":
            if self.a5_phase != "main" or self.warmup_bundle is not None:
                raise ValueError("no-write A5 cannot use the Memory/State warmup handoff")
            if self.qwen_outer_trainability.mode == "frozen":
                raise ValueError("no-write A5 requires a trainable Qwen policy")
        if self.stage == "a5" and self.a5_adaptation_mode == "meta_ttt":
            if self.a5_phase == "fast_state_warmup":
                if self.qwen_outer_trainability.mode != "frozen":
                    raise ValueError("Memory/State warmup requires fully frozen Qwen")
                if self.warmup_bundle is not None:
                    raise ValueError("Memory/State warmup cannot consume a warmup bundle")
            else:
                if self.qwen_outer_trainability.mode == "frozen":
                    raise ValueError("A5 main requires the configured partial Qwen policy")
                if self.warmup_bundle is None:
                    raise ValueError("A5 main Meta-TTT requires a warmup handoff bundle")
        required_materialization = {
            "a2": "dataloader_episode",
            "a5": "segment_double_buffer",
        }
        if self.support_materialization != required_materialization[self.stage]:
            raise ValueError("support_materialization is incompatible with the configured stage")
        if self.stage == "a2" and self.segment_prefetch_depth != 0:
            raise ValueError("A2 cannot enable segment prefetch")
        if (
            self.stage == "a5"
            and self.support_materialization == "segment_double_buffer"
            and self.segment_prefetch_depth != 1
        ):
            raise ValueError("A5 segment_double_buffer requires segment_prefetch_depth=1")
        if self.runtime_trace_mode == "cuda" and self.runtime_trace_dir is None:
            raise ValueError("cuda runtime tracing requires runtime_trace_dir")
        if self.visual_cost_mode != "proxy" and self.visual_cost_index is None:
            raise ValueError("strict visual cost modes require visual_cost_index")
        if (
            self.preprocess_cache_mode == "disabled"
            and self.preprocess_cache_miss_policy != "decode"
        ):
            raise ValueError("disabled preprocess cache requires miss_policy=decode")
        for role, mode, maximum in (
            ("State", self.state_query_visual_mode, self.state_query_max_frames),
            ("Answer", self.answer_query_visual_mode, self.answer_query_max_frames),
        ):
            if maximum % 2:
                raise ValueError(f"{role} query frame limit must be even for Qwen patching")
            if mode == "recent_chunk" and maximum > 16:
                raise ValueError(f"{role} recent_chunk Query permits at most 16 frames")
            if mode == "causal_prefix" and maximum != 256:
                raise ValueError(f"{role} causal_prefix Query requires exactly 256 frames")
        return self

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
    ttt_config: ProductionTTTConfig
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
class QwenTrainabilityAudit:
    mode: str
    total_parameters: int
    trainable_parameters: int
    vision_block_count: int
    frozen_vision_block_indices: tuple[int, ...]
    trainable_vision_block_indices: tuple[int, ...]
    decoder_layer_count: int
    frozen_decoder_layer_indices: tuple[int, ...]
    trainable_decoder_layer_indices: tuple[int, ...]
    vision_patch_embed_trainable: bool
    main_merger_trainable: bool
    deepstack_mergers_trainable: bool
    language_model_norm_trainable: bool
    input_embeddings_trainable: bool
    lm_head_trainable: bool
    all_qwen_parameters_trainable: bool

    def __post_init__(self) -> None:
        if self.mode not in {"full", "partial", "frozen"}:
            raise ValueError("Qwen trainability audit has an invalid mode")
        if self.total_parameters <= 0 or self.trainable_parameters < 0:
            raise ValueError("Qwen trainability audit has invalid parameter counts")
        if self.trainable_parameters > self.total_parameters:
            raise ValueError("Qwen trainability audit trainable count exceeds total count")
        if self.vision_block_count != 27 or self.decoder_layer_count != 36:
            raise ValueError("Qwen trainability audit requires the pinned 27/36 layer structure")
        vision_indices = self.frozen_vision_block_indices + self.trainable_vision_block_indices
        decoder_indices = self.frozen_decoder_layer_indices + self.trainable_decoder_layer_indices
        if tuple(sorted(vision_indices)) != tuple(range(self.vision_block_count)):
            raise ValueError("Qwen trainability audit does not partition every vision block")
        if tuple(sorted(decoder_indices)) != tuple(range(self.decoder_layer_count)):
            raise ValueError("Qwen trainability audit does not partition every Decoder layer")
        if self.mode == "full":
            if self.trainable_parameters != self.total_parameters:
                raise ValueError("full Qwen policy must train every parameter")
            if self.frozen_vision_block_indices or self.frozen_decoder_layer_indices:
                raise ValueError("full Qwen policy cannot freeze formal layers")
            if not self.all_qwen_parameters_trainable:
                raise ValueError("full Qwen policy audit reports frozen parameters")
        elif self.mode == "partial" and self.all_qwen_parameters_trainable:
            raise ValueError("partial Qwen policy must leave at least one parameter frozen")
        elif self.mode == "frozen":
            if self.trainable_parameters != 0 or self.all_qwen_parameters_trainable:
                raise ValueError("frozen Qwen policy cannot expose trainable parameters")
            if self.trainable_vision_block_indices or self.trainable_decoder_layer_indices:
                raise ValueError("frozen Qwen policy cannot expose trainable formal layers")
            trainability_flags = (
                self.vision_patch_embed_trainable,
                self.main_merger_trainable,
                self.deepstack_mergers_trainable,
                self.language_model_norm_trainable,
                self.input_embeddings_trainable,
                self.lm_head_trainable,
            )
            if any(trainability_flags):
                raise ValueError("frozen Qwen policy cannot expose trainable boundary modules")


@dataclass(frozen=True, slots=True)
class OuterCheckpointAudit:
    checkpoint: Path
    format: str
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    forbidden_runtime_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.checkpoint.exists() or not self.format or self.tensor_count <= 0:
            raise ValueError("A2 checkpoint audit is incomplete")
        if self.missing_keys or self.unexpected_keys or self.forbidden_runtime_keys:
            raise ValueError("outer checkpoint does not exactly match the model boundary")


def load_training_yaml(path: str | Path) -> tuple[dict[str, object], ProductionTTTConfig]:
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
    visual_batch_override = os.environ.get("TTT_SUPPORT_VISUAL_BATCH_SIZE")
    if visual_batch_override is not None:
        try:
            batch_size = int(visual_batch_override)
        except ValueError as error:
            raise ValueError("TTT_SUPPORT_VISUAL_BATCH_SIZE must be a positive integer") from error
        if batch_size <= 0:
            raise ValueError("TTT_SUPPORT_VISUAL_BATCH_SIZE must be a positive integer")
        extension["support_visual_batch_size"] = batch_size
    offload_override = os.environ.get("TTT_QUERY_ACTIVATION_OFFLOAD")
    if offload_override is not None:
        normalized = offload_override.strip().casefold()
        if normalized not in {"0", "1", "false", "true"}:
            raise ValueError("TTT_QUERY_ACTIVATION_OFFLOAD must be 0/1/false/true")
        extension["query_activation_offload"] = normalized in {"1", "true"}
    if os.environ.get("TTT_DATALOADER_TRACE") == "1":
        run_root = os.environ.get("RUN_ROOT")
        if not run_root:
            raise ValueError("TTT_DATALOADER_TRACE=1 requires RUN_ROOT")
        extension["runtime_trace_mode"] = "cuda"
        extension["runtime_trace_dir"] = str(Path(run_root) / "runtime_trace")
    if os.environ.get("TTT_VISUAL_COST_PREFLIGHT") == "1":
        if os.environ.get("TTT_SMOKE_MAX_STEPS") is None:
            raise ValueError("visual-cost preflight is allowed only for an explicit smoke run")
        extension["visual_cost_mode"] = "proxy"
        extension.pop("visual_cost_index", None)
    project_config_override = os.environ.get("TTT_PROJECT_CONFIG")
    if project_config_override is not None:
        if not project_config_override.strip():
            raise ValueError("TTT_PROJECT_CONFIG must be non-empty when set")
        extension["project_config"] = project_config_override
    return values, ProductionTTTConfig.model_validate(extension)


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


def configure_qwen_outer_trainability(
    model: nn.Module,
    config: ProjectConfig,
    policy: QwenOuterTrainabilityConfig,
) -> QwenTrainabilityAudit:
    """Apply one exact full or partial Outer-AdamW policy to Qwen3-VL."""

    owner = getattr(model, "model", model)
    assert_qwen_runtime_structure(owner, config)
    visual = _resolve_path(owner, "visual")
    language_model = _resolve_path(owner, "language_model")
    vision_blocks = _module_sequence(_resolve_path(visual, "blocks"), "visual.blocks")
    decoder_layers = _module_sequence(
        _resolve_path(language_model, "layers"),
        "language_model.layers",
    )
    if len(vision_blocks) != config.model.vision.depth:
        raise ValueError("Qwen vision block count drifted before trainability selection")
    if len(decoder_layers) != config.model.llm.num_layers:
        raise ValueError("Qwen Decoder layer count drifted before trainability selection")
    if policy.vision_freeze_first_blocks > len(vision_blocks):
        raise ValueError("vision_freeze_first_blocks exceeds the loaded ViT depth")
    if policy.decoder_train_last_layers > len(decoder_layers):
        raise ValueError("decoder_train_last_layers exceeds the loaded Decoder depth")

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
            trainable_modules.append(
                _module(_resolve_path(visual, "patch_embed"), "visual.patch_embed")
            )
        if policy.train_main_merger:
            trainable_modules.append(_module(_resolve_path(visual, "merger"), "visual.merger"))
        if policy.train_deepstack_mergers:
            trainable_modules.extend(
                _module_sequence(
                    _resolve_path(visual, "deepstack_merger_list"),
                    "visual.deepstack_merger_list",
                )
            )
        if policy.train_language_model_norm:
            trainable_modules.append(
                _module(_resolve_path(language_model, "norm"), "language_model.norm")
            )
        input_embeddings = _optional_module(language_model, "embed_tokens")
        lm_head = _optional_module(model, "lm_head")
        if policy.train_input_embeddings:
            if input_embeddings is None:
                raise ValueError("partial Qwen policy requires language_model.embed_tokens")
            trainable_modules.append(input_embeddings)
        if policy.train_lm_head:
            if lm_head is None:
                raise ValueError("partial Qwen policy requires lm_head")
            trainable_modules.append(lm_head)
        if input_embeddings is not None and lm_head is not None:
            tied = bool(
                {id(parameter) for parameter in input_embeddings.parameters()}
                & {id(parameter) for parameter in lm_head.parameters()}
            )
            if tied and policy.train_input_embeddings != policy.train_lm_head:
                raise ValueError(
                    "tied input embeddings/lm_head cannot use different trainability flags"
                )
        for module in trainable_modules:
            module.requires_grad_(True)

    named = tuple(model.named_parameters())
    if not named:
        raise ValueError("Qwen model exposes no parameters")
    frozen_vision = tuple(
        index for index, block in enumerate(vision_blocks) if not _all_parameters_trainable(block)
    )
    trainable_vision = tuple(
        index for index, block in enumerate(vision_blocks) if _all_parameters_trainable(block)
    )
    frozen_decoder = tuple(
        index for index, layer in enumerate(decoder_layers) if not _all_parameters_trainable(layer)
    )
    trainable_decoder = tuple(
        index for index, layer in enumerate(decoder_layers) if _all_parameters_trainable(layer)
    )
    input_embeddings = _optional_module(language_model, "embed_tokens")
    lm_head = _optional_module(model, "lm_head")
    return QwenTrainabilityAudit(
        mode=policy.mode,
        total_parameters=sum(parameter.numel() for _, parameter in named),
        trainable_parameters=sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        vision_block_count=len(vision_blocks),
        frozen_vision_block_indices=frozen_vision,
        trainable_vision_block_indices=trainable_vision,
        decoder_layer_count=len(decoder_layers),
        frozen_decoder_layer_indices=frozen_decoder,
        trainable_decoder_layer_indices=trainable_decoder,
        vision_patch_embed_trainable=_all_parameters_trainable(
            _module(_resolve_path(visual, "patch_embed"), "visual.patch_embed")
        ),
        main_merger_trainable=_all_parameters_trainable(
            _module(_resolve_path(visual, "merger"), "visual.merger")
        ),
        deepstack_mergers_trainable=all(
            _all_parameters_trainable(module)
            for module in _module_sequence(
                _resolve_path(visual, "deepstack_merger_list"),
                "visual.deepstack_merger_list",
            )
        ),
        language_model_norm_trainable=_all_parameters_trainable(
            _module(_resolve_path(language_model, "norm"), "language_model.norm")
        ),
        input_embeddings_trainable=(
            input_embeddings is not None and _all_parameters_trainable(input_embeddings)
        ),
        lm_head_trainable=lm_head is not None and _all_parameters_trainable(lm_head),
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


def load_outer_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
) -> OuterCheckpointAudit:
    """Strictly load an outer-only single or sharded safetensors checkpoint."""

    source = Path(checkpoint).resolve()
    if source.is_dir():
        single = source / "model.safetensors"
        index = source / "model.safetensors.index.json"
        if single.is_file() == index.is_file():
            raise ValueError(
                "checkpoint directory must contain exactly one of model.safetensors or its index"
            )
        source = single if single.is_file() else index
    if not source.is_file():
        raise FileNotFoundError(f"outer checkpoint does not exist: {source}")
    expected = set(audit_outer_checkpoint_boundary(model))
    state: dict[str, torch.Tensor] | None = None
    if source.suffix == ".safetensors":
        state = load_file(str(source), device="cpu")
        loaded = set(state)
        checkpoint_format = "safetensors"
    elif source.name == "model.safetensors.index.json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
        ):
            raise ValueError("sharded safetensors index requires a string weight_map")
        shard_names = set(cast(dict[str, str], weight_map).values())
        if not shard_names or any(not name.endswith(".safetensors") for name in shard_names):
            raise ValueError("sharded outer checkpoint may contain safetensors shards only")
        missing_shards = tuple(
            sorted(name for name in shard_names if not (source.parent / name).is_file())
        )
        if missing_shards:
            raise FileNotFoundError(f"checkpoint shards are missing: {missing_shards}")
        loaded = set(cast(dict[str, str], weight_map))
        checkpoint_format = "sharded_safetensors"
    else:
        raise ValueError(
            "outer checkpoint must be a .safetensors file or model.safetensors.index.json"
        )
    forbidden = tuple(
        sorted(
            name
            for name in loaded
            if any(token in name.casefold() for token in _FORBIDDEN_CHECKPOINT_TOKENS)
        )
    )
    missing = tuple(sorted(expected - loaded))
    unexpected = tuple(sorted(loaded - expected))
    if missing or unexpected or forbidden:
        return OuterCheckpointAudit(
            checkpoint=source,
            format=checkpoint_format,
            tensor_count=len(loaded),
            missing_keys=missing,
            unexpected_keys=unexpected,
            forbidden_runtime_keys=forbidden,
        )
    if state is not None:
        result = model.load_state_dict(state, strict=True)
    else:
        loader = cast(Callable[..., Any], load_sharded_checkpoint)
        result = loader(model, str(source.parent), strict=True, prefer_safe=True)
    result_missing = tuple(getattr(result, "missing_keys", ()))
    result_unexpected = tuple(getattr(result, "unexpected_keys", ()))
    return OuterCheckpointAudit(
        checkpoint=source,
        format=checkpoint_format,
        tensor_count=len(loaded),
        missing_keys=result_missing,
        unexpected_keys=result_unexpected,
        forbidden_runtime_keys=(),
    )


def initialize_outer_model_from_a2(
    model: nn.Module,
    checkpoint: str | Path,
) -> OuterCheckpointAudit:
    """Load exact-current or narrowly profiled legacy A2 weights for A5 initialization."""

    root = Path(checkpoint).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"A2 checkpoint directory does not exist: {root}")
    expected_all = set(audit_outer_checkpoint_boundary(model))
    safe_index = root / "model.safetensors.index.json"
    torch_index = root / "pytorch_model.bin.index.json"
    safe_weights = root / "model.safetensors"
    torch_weights = root / "pytorch_model.bin"
    if safe_index.is_file() or torch_index.is_file():
        load_sharded_checkpoint(model, str(root), strict=False, prefer_safe=True)
        index_path = safe_index if safe_index.is_file() else torch_index
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("A2 sharded checkpoint index has no weight_map")
        loaded_keys = set(weight_map)
        checkpoint_format = "sharded_safetensors" if safe_index.is_file() else "sharded_torch"
    elif safe_weights.is_file():
        state = load_file(str(safe_weights), device="cpu")
        model.load_state_dict(state, strict=False)
        loaded_keys = set(state)
        checkpoint_format = "safetensors"
    elif torch_weights.is_file():
        raw = torch.load(torch_weights, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError("A2 torch checkpoint must contain a string-keyed state dict")
        state = cast(dict[str, torch.Tensor], raw)
        model.load_state_dict(state, strict=False)
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
    missing = tuple(
        sorted(
            key
            for key in expected_all - loaded_keys
            if not _LEGACY_A2_TO_A5.allows_missing(key)
        )
    )
    unexpected = tuple(
        sorted(
            key
            for key in loaded_keys - expected_all
            if not _LEGACY_A2_TO_A5.allows_unexpected(key)
        )
    )
    return OuterCheckpointAudit(
        checkpoint=root,
        format=checkpoint_format,
        tensor_count=len(loaded_keys),
        missing_keys=missing,
        unexpected_keys=unexpected,
        forbidden_runtime_keys=forbidden,
    )


def environment_manifest(bundle: LlamaFactoryBackboneBundle) -> dict[str, object]:
    return {
        "llamafactory_root": str(bundle.symbols.checkout.root),
        "llamafactory_commit": bundle.symbols.checkout.commit,
        "llamafactory_dirty": bundle.symbols.checkout.dirty,
        "qwen_model_path": str(getattr(bundle.model_args, "model_name_or_path", "")),
        "project_spec_version": bundle.project_config.spec_version,
        "config_schema_version": bundle.project_config.config_schema_version,
        "associative_ttt_contract": bundle.project_config.associative_ttt.contract,
        "associative_ttt_contract_version": 3,
        "ttt_config": json.loads(bundle.ttt_config.model_dump_json()),
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


def _module(value: object, path: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise TypeError(f"Qwen trainability path must be a module: {path}")
    return value


def _module_sequence(value: object, path: str) -> tuple[nn.Module, ...]:
    if not isinstance(value, (list, tuple, nn.ModuleList)):
        raise TypeError(f"Qwen trainability path must be a module sequence: {path}")
    modules = tuple(value)
    if not modules or any(not isinstance(module, nn.Module) for module in modules):
        raise TypeError(f"Qwen trainability sequence is empty or invalid: {path}")
    return modules


def _optional_module(value: object, name: str) -> nn.Module | None:
    module = getattr(value, name, None)
    if module is None:
        return None
    return _module(module, name)


def _all_parameters_trainable(module: nn.Module) -> bool:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("Qwen trainability audit encountered a parameterless module")
    return all(parameter.requires_grad for parameter in parameters)


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
    "ProductionTTTConfig",
    "QwenOuterTrainabilityConfig",
    "QwenTrainabilityAudit",
    "audit_outer_checkpoint_boundary",
    "configure_qwen_outer_trainability",
    "environment_manifest",
    "fully_unfreeze_qwen",
    "import_llamafactory",
    "initialize_outer_model_from_a2",
    "load_llamafactory_backbone",
    "load_training_yaml",
]
