"""Custom LLaMA-Factory Trainer bridge for A2 and segmented A5 Meta-TTT.

The A5 ``training_step`` performs multiple segment backward calls but deliberately performs no
optimizer step.  Hugging Face/LLaMA-Factory's outer loop therefore clips and steps exactly once
for the complete episode.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast, overload

# ``python -m`` executes this file as ``__main__``.  The dynamically loaded production runtime
# imports the canonical package name, so register the running module under that name before the
# runtime factory is imported.  Otherwise Python creates a second copy of the dataclasses/enums
# and a valid ProductionTrainerRuntime fails the identity-based boundary audit.
if __name__ == "__main__":
    sys.modules.setdefault("ttt_svcbench_qwen.llamafactory_trainer", sys.modules[__name__])

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ttt_svcbench_qwen.config import OuterGradientControlMode, ProjectConfig
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    build_production_train_sampler,
    load_production_manifest_views,
    load_visual_cost_index,
)
from ttt_svcbench_qwen.fast_ttt import PROBE_FIELDS, FastTTTAdapter
from ttt_svcbench_qwen.meta_trainer import (
    CounterfactualAuditRequest,
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    TruncatedMetaTTTEpisodeOutput,
)
from ttt_svcbench_qwen.outer_gradient_control import (
    GradientProbe,
    OuterGradientAudit,
    OuterGradientController,
    sanitize_scalar_loss,
)
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    OuterCheckpointAudit,
    audit_outer_checkpoint_boundary,
    configure_qwen_outer_trainability,
    environment_manifest,
    fully_unfreeze_qwen,
    initialize_outer_model_from_a2,
    load_llamafactory_backbone,
)
from ttt_svcbench_qwen.query_encoder import OPERATORS
from ttt_svcbench_qwen.runtime_metrics import (
    flush_runtime_metrics,
    trace_cuda_phase,
    trace_event,
)
from ttt_svcbench_qwen.stage_a_targets import OfficialWeakLossAudit
from ttt_svcbench_qwen.visual_cost import (
    VisualCostRecord,
    make_visual_cost_fingerprint,
)


class ProductionStage(StrEnum):
    A2 = "a2"
    A5 = "a5"


class CheckpointPolicy(StrEnum):
    ATOMIC_FINAL_ONLY = "atomic_final_only"
    EPOCH_2_AND_EPOCH_4 = "epoch_2_and_epoch_4"


_WARMUP_BUNDLE_SCHEMA_VERSION = 2
_WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION = 4
_WARMUP_BUNDLE_EXCLUDED_TOKENS = (
    "official_weak_balancer",
    "transient_memory",
    "state_bank_runtime",
    "identity_bank_runtime",
    "fsm_runtime",
    "optimizer",
    "scheduler",
    "runtime",
    "cache",
)
_A5_ADAPTATION_MODES = ("meta_ttt", "no_write")
_A5_WARMUP_TRAINABLE_GROUPS = frozenset(
    {
        "state_shared",
        "state_task",
        "state_router_time",
        "state_retrieval",
        "associative",
    }
)


def _is_transient_memory_name(lowered: str) -> bool:
    """Match the per-video memory tensor `m` and any explicit transient marker."""

    return (
        "transient_memory" in lowered
        or lowered == "m"
        or lowered.endswith(".m")
        or lowered.endswith(("w_t_1", "w_t_2"))
    )


class StageALossStep(Protocol):
    def __call__(self, model: nn.Module, inputs: Mapping[str, object]) -> Tensor: ...


class EpisodeAdapter(Protocol):
    def __call__(self, inputs: Mapping[str, object]) -> tuple[MetaTTTEpisode, float]: ...


class TrainSamplerFactory(Protocol):
    def __call__(self, dataset: object, rank: int, world_size: int) -> object: ...


class _ControlledDeepSpeedEngineWrapper:
    """Pinned Accelerate wrapper with group clipping inserted before the real engine step."""

    def __init__(
        self,
        engine: object,
        gradient_controller: OuterGradientController,
        model: nn.Module | None = None,
        semantic_projector_delta_audit_steps: int = 0,
    ) -> None:
        required = ("set_gradient_accumulation_boundary", "backward", "step")
        if any(not callable(getattr(engine, name, None)) for name in required):
            raise TypeError("controlled DeepSpeed wrapper received an invalid engine")
        self.engine = engine
        self.gradient_controller = gradient_controller
        self.semantic_projector_auditor = (
            _SemanticProjectorStepAuditor(
                model, delta_audit_steps=semantic_projector_delta_audit_steps
            )
            if model is not None
            else None
        )

    def backward(self, loss: Tensor, sync_gradients: bool = True, **kwargs: object) -> None:
        engine = cast(Any, self.engine)
        engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
        engine.backward(loss, **kwargs)
        if sync_gradients:
            audit = self.gradient_controller.apply_deepspeed(engine.optimizer)
            snapshot = (
                self.semantic_projector_auditor.before_step(engine.optimizer, audit)
                if self.semantic_projector_auditor is not None
                else None
            )
            engine.step()
            if self.semantic_projector_auditor is not None:
                self.semantic_projector_auditor.after_step(snapshot, audit)

    def get_global_grad_norm(self) -> float:
        value = cast(Any, self.engine).get_global_grad_norm()
        return float(value.item()) if hasattr(value, "item") else float(value)


class SegmentBackwardController:
    """Accumulate segment gradients and make DeepSpeed step exactly once per episode.

    Accelerate's DeepSpeed backward wrapper also calls ``engine.step()``.  It therefore cannot
    be used for each TBPTT segment.  Direct ``engine.backward`` preserves all segment gradients;
    ``finalize`` executes the sole engine step only after the runner has audited unchanged Outer
    parameter versions.
    """

    def __init__(
        self,
        accelerator: object,
        model: nn.Module,
        *,
        expected_count: int,
        gradient_controller: OuterGradientController | None = None,
        semantic_projector_delta_audit_steps: int = 0,
        a5_parameter_delta_audit_steps: int = 0,
    ) -> None:
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("segment backward count must be a positive integer")
        self.accelerator = accelerator
        self.expected_count = expected_count
        self.backward_count = 0
        self.step_count = 0
        self.gradient_controller = gradient_controller
        self.semantic_projector_auditor = (
            _SemanticProjectorStepAuditor(
                model, delta_audit_steps=semantic_projector_delta_audit_steps
            )
            if isinstance(gradient_controller, OuterGradientController)
            and "state_retrieval" in gradient_controller.expected_groups
            else None
        )
        self.a5_parameter_delta_auditor = (
            _A5ParameterGroupStepAuditor(
                model,
                delta_audit_steps=a5_parameter_delta_audit_steps,
                group_names=tuple(
                    name
                    for name in _A5ParameterGroupStepAuditor._GROUP_NAMES
                    if name in gradient_controller.expected_groups
                ),
            )
            if isinstance(gradient_controller, OuterGradientController)
            and a5_parameter_delta_audit_steps > 0
            else None
        )
        self.is_deepspeed = (
            "deepspeed" in str(getattr(accelerator, "distributed_type", "")).casefold()
        )
        wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
        self.engine = getattr(wrapper, "engine", None) if self.is_deepspeed else None
        if self.is_deepspeed:
            if self.engine is None:
                self.engine = model
            required = (
                "set_gradient_accumulation_boundary",
                "zero_optimization_partition_gradients",
                "backward",
                "step",
            )
            if any(not callable(getattr(self.engine, name, None)) for name in required):
                raise TypeError(
                    "DeepSpeed segment controller requires boundary/partition/backward/step methods"
                )
        elif not callable(getattr(accelerator, "backward", None)):
            raise TypeError("segment controller requires accelerator.backward")
        if (
            self.is_deepspeed
            and isinstance(self.gradient_controller, OuterGradientController)
            and "qwen" in self.gradient_controller.expected_groups
        ):
            self._rank_stable_parameters = _rank_stable_conditional_parameters(
                model,
                expected_groups=self.gradient_controller.expected_groups,
            )
        else:
            self._rank_stable_parameters = _unique_trainable_parameters(model)
        if not self._rank_stable_parameters:
            raise ValueError("A5 segment controller requires rank-stable Outer parameters")
        self._rank_hook_order_audit_enabled = (
            self.is_deepspeed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        self._requires_exact_rank_hook_order = (
            self.is_deepspeed and _deepspeed_partitions_gradients(cast(Any, self.engine))
        )

    @property
    def proxy_gradient_scale(self) -> float:
        """Return the engine scale applied to non-parameter proxy leaf gradients.

        Production A5 is BF16 with episode-level GA=1, so the expected value is exactly one.
        Refuse an unmodelled DeepSpeed loss scale instead of silently squaring it in the
        deferred VJP.
        """

        if not self.is_deepspeed:
            return 1.0
        engine = cast(Any, self.engine)
        accumulation = getattr(engine, "gradient_accumulation_steps", None)
        if callable(accumulation) and int(accumulation()) != 1:
            raise ValueError("A5 deferred VJP requires DeepSpeed gradient accumulation of one")
        raw_scale = getattr(getattr(engine, "optimizer", None), "loss_scale", 1.0)
        if isinstance(raw_scale, Tensor):
            if raw_scale.numel() != 1:
                raise ValueError("DeepSpeed loss scale must be scalar")
            raw_scale = raw_scale.detach().float().cpu().item()
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale != 1.0:
            raise ValueError("A5 deferred VJP currently requires unscaled BF16 DeepSpeed backward")
        return scale

    def backward(self, loss: Tensor, retain_graph: bool = False) -> None:
        if self.backward_count >= self.expected_count:
            raise RuntimeError("segment runner emitted too many backward calls")
        with trace_cuda_phase(
            "backward",
            stage="a5_segment",
            segment_index=self.backward_count,
        ):
            if isinstance(self.gradient_controller, OuterGradientController):
                loss = sanitize_scalar_loss(
                    loss,
                    source=f"A5 backward {self.backward_count}",
                    controller=self.gradient_controller,
                )
            elif loss.ndim != 0 or not loss.requires_grad:
                raise ValueError("A5 segment loss must be one differentiable scalar Tensor")
            loss = _attach_rank_stable_zero_anchor(
                loss,
                self._rank_stable_parameters,
            )
            if self.is_deepspeed:
                engine = cast(Any, self.engine)
                is_final_segment = self.backward_count + 1 == self.expected_count
                engine.set_gradient_accumulation_boundary(is_boundary=is_final_segment)
                hook_order: list[int] | None = [] if self._rank_hook_order_audit_enabled else None
                handles = (
                    tuple(
                        parameter.register_post_accumulate_grad_hook(
                            lambda _parameter, index=index: hook_order.append(index)
                        )
                        for index, parameter in enumerate(self._rank_stable_parameters)
                    )
                    if hook_order is not None
                    else ()
                )
                try:
                    if retain_graph:
                        engine.backward(loss, retain_graph=True)
                    else:
                        engine.backward(loss)
                finally:
                    for handle in handles:
                        handle.remove()
                if hook_order is not None:
                    self._assert_rank_hook_order(hook_order, device=loss.device)
            else:
                cast(Any, self.accelerator).backward(loss, retain_graph=retain_graph)
        self.backward_count += 1

    def _assert_rank_hook_order(self, order: Sequence[int], *, device: torch.device) -> None:
        """Fail before the Outer step when ZeRO-2 observes rank-local hook order."""

        expected = len(self._rank_stable_parameters)
        if len(order) != expected or len(set(order)) != expected:
            raise RuntimeError(
                "A5 rank-stable anchor hook coverage drifted: "
                f"expected={expected}, observed={len(order)}"
            )
        world_size = torch.distributed.get_world_size()
        local_count = torch.tensor([len(order)], dtype=torch.int64, device=device)
        gathered_counts = [torch.empty_like(local_count) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_counts, local_count)
        counts = tuple(int(value.item()) for value in gathered_counts)
        if len(set(counts)) != 1:
            raise RuntimeError(f"A5 parameter hook count diverged across ranks: {counts}")
        local_order = torch.tensor(tuple(order), dtype=torch.int64, device=device)
        gathered_orders = [torch.empty_like(local_order) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_orders, local_order)
        reference = gathered_orders[0]
        for rank, candidate in enumerate(gathered_orders[1:], start=1):
            if self._requires_exact_rank_hook_order and not torch.equal(candidate, reference):
                raise RuntimeError(
                    "A5 parameter hook order diverged across ranks before Outer step: "
                    f"segment={self.backward_count}, rank=0/{rank}"
                )
            if not self._requires_exact_rank_hook_order and not torch.equal(
                candidate.sort().values,
                reference.sort().values,
            ):
                raise RuntimeError(
                    "A5 parameter hook coverage diverged across ranks before Outer step: "
                    f"segment={self.backward_count}, rank=0/{rank}"
                )

    def finalize(self) -> None:
        if self.backward_count != self.expected_count:
            raise RuntimeError("segment runner backward count did not match its bucket")
        if self.step_count:
            raise RuntimeError("segment backward controller was finalized more than once")
        if self.is_deepspeed:
            engine = cast(Any, self.engine)
            audit: OuterGradientAudit | None = None
            if self.gradient_controller is not None:
                audit = self.gradient_controller.apply_deepspeed(engine.optimizer)
            snapshot = (
                self.semantic_projector_auditor.before_step(engine.optimizer, audit)
                if self.semantic_projector_auditor is not None and audit is not None
                else None
            )
            a5_snapshot = (
                self.a5_parameter_delta_auditor.before_step(audit)
                if self.a5_parameter_delta_auditor is not None and audit is not None
                else None
            )
            engine.step()
            if self.semantic_projector_auditor is not None and audit is not None:
                self.semantic_projector_auditor.after_step(snapshot, audit)
            if self.a5_parameter_delta_auditor is not None and audit is not None:
                self.a5_parameter_delta_auditor.after_step(a5_snapshot, audit)
            self.step_count = 1

    @property
    def semantic_projector_metrics(self) -> dict[str, float]:
        metrics = (
            dict(self.semantic_projector_auditor.last_metrics)
            if self.semantic_projector_auditor is not None
            else {}
        )
        if self.a5_parameter_delta_auditor is not None:
            metrics.update(self.a5_parameter_delta_auditor.last_metrics)
        return metrics


def _unique_trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return every trainable Outer parameter once in deterministic model order."""

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if not parameter.requires_grad or parameter_id in seen:
            continue
        if parameter.numel() <= 0:
            raise ValueError("A5 trainable Outer parameters cannot be empty")
        seen.add(parameter_id)
        parameters.append(parameter)
    return tuple(parameters)


def _deepspeed_partitions_gradients(engine: object) -> bool:
    """Return whether DeepSpeed communicates gradient partitions from autograd hooks.

    ZeRO-2 appends gradients to collective buckets as parameter hooks fire, so every rank
    must observe the exact same hook order.  ZeRO-1 performs the reduction at the accumulation
    boundary by iterating optimizer parameters in a fixed order; for that profile the
    rank-stable anchor only needs to guarantee identical hook coverage.
    """

    result = cast(Any, engine).zero_optimization_partition_gradients()
    if type(result) is not bool:
        raise TypeError("DeepSpeed zero_optimization_partition_gradients() must return bool")
    return result


def _rank_stable_conditional_parameters(
    model: nn.Module,
    *,
    expected_groups: Sequence[str],
) -> tuple[nn.Parameter, ...]:
    """Return original conditionally used Parameters while excluding the shared Qwen."""

    if "qwen" not in expected_groups:
        raise ValueError("rank-stable A5 optimizer groups must include qwen")
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    qwen_ids: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        lowered = name.casefold()
        parameter_id = id(parameter)
        if lowered.startswith("qwen.") or ".visual_stage.qwen.qwen_model." in lowered:
            if parameter.requires_grad:
                qwen_ids.add(parameter_id)
            continue
        if not parameter.requires_grad or parameter_id in seen:
            continue
        if parameter.numel() <= 0:
            raise ValueError("A5 trainable Outer parameters cannot be empty")
        seen.add(parameter_id)
        parameters.append(parameter)
    if not qwen_ids:
        raise RuntimeError("rank-stable A5 anchor could not identify trainable Qwen parameters")
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if qwen_ids | seen != trainable_ids or qwen_ids & seen:
        raise RuntimeError("rank-stable A5 anchor did not partition trainable parameters")
    return tuple(parameters)


def _attach_rank_stable_zero_anchor(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
) -> Tensor:
    """Expose an identical ZeRO-2 gradient-hook surface on every segmented backward.

    A5 intentionally executes several backwards per episode.  Official-weak routing and
    retrieval validity are sample-dependent, so two ranks can otherwise touch different state
    parameters during the same backward call.  ZeRO-2 then launches different reduction
    sequences and deadlocks.  One differentiable zero scalar from every conditionally used
    non-Qwen parameter makes that hook set deterministic without changing the forward value or
    any gradient.  Qwen is excluded because every Support and Query backward already traverses
    its shared backbone; anchoring its 1.9B trainable parameters would add redundant reductions.
    The anchor is rebuilt for every call because a completed backward frees its graph.
    """

    if loss.ndim != 0 or not loss.requires_grad:
        raise ValueError("rank-stable anchor requires one differentiable scalar loss")
    anchor = loss.new_zeros(())
    for parameter in parameters:
        if not parameter.requires_grad:
            raise RuntimeError("rank-stable A5 parameter became frozen after controller setup")
        if parameter.device != loss.device:
            raise RuntimeError("rank-stable A5 parameter and loss must share one device")
        first = parameter if parameter.ndim == 0 else parameter[(0,) * parameter.ndim]
        anchor = anchor + first.to(dtype=loss.dtype) * 0.0
    return loss + anchor


class _SemanticProjectorStepAuditor:
    """Audit the exact retrieval optimizer group before clear and its real step delta."""

    _GROUP_NAME = "state_retrieval"

    def __init__(self, model: nn.Module, *, delta_audit_steps: int = 32) -> None:
        if type(delta_audit_steps) is not int or delta_audit_steps < 0:
            raise ValueError("SemanticProjector delta audit steps must be non-negative")
        parameters = tuple(
            parameter
            for name, parameter in model.named_parameters()
            if "semantic_projector" in name.casefold() and parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("formal model exposes no trainable SemanticProjector parameters")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("SemanticProjector parameter aliases are not supported")
        self.parameters = parameters
        self.parameter_ids = frozenset(id(parameter) for parameter in parameters)
        self.delta_audit_steps = delta_audit_steps
        self.last_metrics: dict[str, float] = {}
        self._optimizer_validated = False

    def before_step(
        self,
        optimizer: object,
        audit: OuterGradientAudit,
    ) -> tuple[Tensor, ...] | None:
        self._validate_optimizer_group(optimizer)
        group = audit.group(self._GROUP_NAME)
        self.last_metrics = {
            "grad/semantic_projector/pre_clip_norm": group.pre_clip_norm,
            "grad/semantic_projector/post_clip_norm": group.post_clip_norm,
            "grad/semantic_projector/clip_coefficient": group.clip_coefficient,
            "grad/semantic_projector/active_elements": float(group.active_elements),
            "grad/semantic_projector/nonfinite_elements": float(group.nonfinite_elements),
        }
        if audit.skipped_nonfinite or audit.successful_update_count > self.delta_audit_steps:
            return None
        return tuple(parameter.detach().float().clone() for parameter in self.parameters)

    def after_step(
        self,
        snapshot: tuple[Tensor, ...] | None,
        audit: OuterGradientAudit,
    ) -> None:
        if snapshot is None:
            return
        if len(snapshot) != len(self.parameters):
            raise RuntimeError("SemanticProjector parameter snapshot drifted")
        squared = torch.zeros((), dtype=torch.float64, device=self.parameters[0].device)
        for before, parameter in zip(snapshot, self.parameters, strict=True):
            if before.shape != parameter.shape or before.device != parameter.device:
                raise RuntimeError("SemanticProjector parameter topology changed across step")
            squared.add_((parameter.detach().float() - before).double().square().sum())
        local_delta = squared.sqrt().float()
        minimum = local_delta.clone()
        maximum = local_delta.clone()
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        min_value = float(minimum.item())
        max_value = float(maximum.item())
        tolerance = 1.0e-6 + 1.0e-4 * max_value
        if max_value - min_value > tolerance:
            raise RuntimeError(
                "SemanticProjector parameter delta diverged across ranks: "
                f"min={min_value:.9g}, max={max_value:.9g}"
            )
        self.last_metrics.update(
            {
                "grad/semantic_projector/parameter_delta_l2": float(local_delta.item()),
                "grad/semantic_projector/parameter_delta_nonzero": float(
                    bool(local_delta.item() > 0.0)
                ),
                "grad/semantic_projector/delta_audit_step": float(audit.successful_update_count),
            }
        )

    def _validate_optimizer_group(self, optimizer: object) -> None:
        if self._optimizer_validated:
            return
        base_optimizer = getattr(optimizer, "optimizer", optimizer)
        groups = getattr(base_optimizer, "param_groups", None)
        if not isinstance(groups, list):
            raise TypeError("SemanticProjector audit requires optimizer param_groups")
        matches = tuple(group for group in groups if group.get("group_name") == self._GROUP_NAME)
        if len(matches) != 1:
            raise RuntimeError("state_retrieval optimizer group is missing or duplicated")
        optimizer_module = type(optimizer).__module__.casefold()
        optimizer_name = type(optimizer).__name__.casefold()
        is_deepspeed = "deepspeed" in optimizer_module or "deepspeed" in optimizer_name
        if is_deepspeed:
            # The exact registered Parameter set was checked before DeepSpeed wrapped
            # AdamW.  ZeRO may replace it here with flat/partition tensors, so object
            # identity is no longer a meaningful ownership check at step time.
            self._optimizer_validated = True
            return
        actual = frozenset(id(parameter) for parameter in matches[0].get("params", ()))
        if actual != self.parameter_ids:
            raise RuntimeError(
                "state_retrieval optimizer group must equal the SemanticProjector parameter set"
            )
        self._optimizer_validated = True


class _A5ParameterGroupStepAuditor:
    """Measure real post-Adam deltas for the groups implicated in A5 TTT drift."""

    _GROUP_NAMES = ("associative", "fast_slow", "w0", "state_shared")
    _DEFAULT_GROUP_NAMES = ("associative", "w0", "state_shared")

    def __init__(
        self,
        model: nn.Module,
        *,
        delta_audit_steps: int,
        group_names: Sequence[str] = _DEFAULT_GROUP_NAMES,
    ) -> None:
        if type(delta_audit_steps) is not int or delta_audit_steps <= 0:
            raise ValueError("A5 parameter delta audit steps must be positive")
        selected = tuple(group_names)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(name not in self._GROUP_NAMES for name in selected)
        ):
            raise ValueError("A5 parameter delta audit groups are invalid")
        self.group_names = selected
        grouped: dict[str, list[nn.Parameter]] = {name: [] for name in self.group_names}
        fast_slow_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, FastTTTAdapter)
            for parameter in module.collect_slow_parameters()
        }
        seen: set[int] = set()
        for name, parameter in model.named_parameters(remove_duplicate=False):
            parameter_id = id(parameter)
            if not parameter.requires_grad or parameter_id in seen:
                continue
            group = "fast_slow" if parameter_id in fast_slow_ids else self._classify_parameter(name)
            if group in grouped:
                grouped[group].append(parameter)
            seen.add(parameter_id)
        empty = tuple(name for name, values in grouped.items() if not values)
        if empty:
            raise RuntimeError(f"A5 parameter delta audit groups are empty: {empty}")
        self.parameters = {name: tuple(values) for name, values in grouped.items()}
        self.delta_audit_steps = delta_audit_steps
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _classify_parameter(name: str) -> str:
        lowered = name.casefold()
        if (
            lowered.startswith(("qwen.", "module.qwen."))
            or ".visual_stage.qwen.qwen_model." in lowered
        ):
            return "qwen"
        return _state_group_for_name(lowered)

    def before_step(
        self,
        audit: OuterGradientAudit,
    ) -> dict[str, tuple[Tensor, ...]] | None:
        self.last_metrics = {}
        if audit.skipped_nonfinite or audit.successful_update_count > self.delta_audit_steps:
            return None
        return {
            name: tuple(parameter.detach().float().clone() for parameter in parameters)
            for name, parameters in self.parameters.items()
        }

    def after_step(
        self,
        snapshot: dict[str, tuple[Tensor, ...]] | None,
        audit: OuterGradientAudit,
    ) -> None:
        if snapshot is None:
            return
        local_l2: list[Tensor] = []
        group_metrics: dict[str, tuple[float, float, float, float, float, int]] = {}
        for name in self.group_names:
            parameters = self.parameters[name]
            before_values = snapshot.get(name)
            if before_values is None or len(before_values) != len(parameters):
                raise RuntimeError(f"A5 {name} parameter snapshot drifted")
            device = parameters[0].device
            delta_squared = torch.zeros((), dtype=torch.float64, device=device)
            before_squared = torch.zeros((), dtype=torch.float64, device=device)
            delta_max_abs = torch.zeros((), dtype=torch.float32, device=device)
            nonzero_count = torch.zeros((), dtype=torch.int64, device=device)
            element_count = 0
            for before, parameter in zip(before_values, parameters, strict=True):
                if before.shape != parameter.shape or before.device != parameter.device:
                    raise RuntimeError(f"A5 {name} parameter topology changed across step")
                delta = parameter.detach().float() - before
                delta_squared.add_(delta.square().sum(dtype=torch.float64))
                before_squared.add_(before.square().sum(dtype=torch.float64))
                delta_max_abs = torch.maximum(delta_max_abs, delta.abs().amax())
                nonzero_count.add_(torch.count_nonzero(delta))
                element_count += parameter.numel()
            delta_l2 = delta_squared.sqrt().float()
            before_l2 = before_squared.sqrt().float()
            delta_rms = delta_l2 / math.sqrt(element_count)
            relative_l2 = delta_l2 / before_l2.clamp_min(torch.finfo(torch.float32).tiny)
            nonzero_fraction = nonzero_count.float() / element_count
            local_l2.append(delta_l2)
            group_metrics[name] = (
                float(delta_l2.item()),
                float(delta_rms.item()),
                float(relative_l2.item()),
                float(delta_max_abs.item()),
                float(nonzero_fraction.item()),
                element_count,
            )
        local = torch.stack(local_l2)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, local)
            reference = gathered[0]
            tolerance = 1.0e-6 + 1.0e-4 * torch.maximum(
                reference.abs(),
                torch.stack(gathered[1:]).abs().amax(dim=0),
            )
            for rank, candidate in enumerate(gathered[1:], start=1):
                if bool(torch.any((candidate - reference).abs() > tolerance).item()):
                    raise RuntimeError(
                        "A5 parameter delta diverged across ranks: "
                        f"rank=0/{rank}, local={reference.tolist()}, "
                        f"candidate={candidate.tolist()}"
                    )
        for name, (
            delta_l2,
            delta_rms,
            relative_l2,
            delta_max_abs,
            nonzero_fraction,
            element_count,
        ) in group_metrics.items():
            prefix = f"a5/parameter_delta/{name}"
            self.last_metrics.update(
                {
                    f"{prefix}/l2": delta_l2,
                    f"{prefix}/rms": delta_rms,
                    f"{prefix}/relative_l2": relative_l2,
                    f"{prefix}/max_abs": delta_max_abs,
                    f"{prefix}/nonzero_fraction": nonzero_fraction,
                    f"{prefix}/element_count": float(element_count),
                    f"{prefix}/audit_step": float(audit.successful_update_count),
                }
            )


@dataclass(frozen=True, slots=True)
class OuterParameterAudit:
    stage: ProductionStage
    total_parameter_count: int
    trainable_parameter_count: int
    qwen_parameter_count: int
    qwen_trainable_count: int
    non_qwen_parameter_count: int
    non_qwen_trainable_count: int
    associative_parameter_count: int
    associative_trainable_count: int
    transient_parameter_names: tuple[str, ...]
    backbone_registered: bool
    a5_adaptation_mode: str = "meta_ttt"
    a5_phase: str = "main"

    def __post_init__(self) -> None:
        if self.a5_adaptation_mode not in _A5_ADAPTATION_MODES:
            raise ValueError("outer parameter audit has an invalid A5 adaptation mode")
        if self.a5_phase not in {"fast_state_warmup", "main"}:
            raise ValueError("outer parameter audit has an invalid A5 phase")
        if self.total_parameter_count <= 0 or self.trainable_parameter_count <= 0:
            raise ValueError("production outer model exposes no trainable parameters")
        if self.associative_parameter_count <= 0:
            raise ValueError("production outer model must register Associative parameters")
        if self.transient_parameter_names:
            raise ValueError("transient fast matrices entered registered outer parameters")
        if not self.backbone_registered:
            raise ValueError("runtime model did not register the loaded Qwen backbone")
        if self.qwen_parameter_count <= 0 or self.non_qwen_parameter_count <= 0:
            raise ValueError("production outer model must register Qwen and state parameters")
        if self.qwen_parameter_count + self.non_qwen_parameter_count != self.total_parameter_count:
            raise ValueError("Qwen/non-Qwen parameter audit does not cover the outer model")
        if (
            self.qwen_trainable_count + self.non_qwen_trainable_count
            != self.trainable_parameter_count
        ):
            raise ValueError("Qwen/non-Qwen trainable audit does not cover the outer model")
        if self.stage is ProductionStage.A2:
            if self.associative_trainable_count:
                raise ValueError("A2 Associative must remain frozen")
            expected = self.total_parameter_count - self.associative_parameter_count
            if self.trainable_parameter_count != expected:
                raise ValueError("A2 must train every registered non-Associative parameter")
        elif self.a5_adaptation_mode == "meta_ttt":
            if self.a5_phase == "fast_state_warmup":
                if self.qwen_trainable_count:
                    raise ValueError("Memory/State warmup must freeze every Qwen parameter")
            elif self.qwen_trainable_count <= 0:
                raise ValueError("A5 main must train configured Qwen parameters")
            if (
                self.a5_phase != "fast_state_warmup"
                and self.non_qwen_trainable_count != self.non_qwen_parameter_count
            ):
                raise ValueError("A5 must train every state, W0, and Associative parameter")
            if self.associative_trainable_count != self.associative_parameter_count:
                raise ValueError("A5 Associative must be fully trainable")
        else:
            if self.qwen_trainable_count <= 0:
                raise ValueError("no-write A5 must train configured Qwen parameters")
            if self.associative_trainable_count:
                raise ValueError("no-write A5 memory-interface parameters must remain frozen")
            expected_non_qwen = self.non_qwen_parameter_count - self.associative_parameter_count
            if self.non_qwen_trainable_count != expected_non_qwen:
                raise ValueError(
                    "no-write A5 must train every non-memory state and W0 parameter"
                )


@dataclass(frozen=True, slots=True)
class ProductionTrainerRuntime:
    """Dataset/materialization hooks assembled entirely inside TTT-QWEN."""

    stage: ProductionStage
    model: nn.Module
    train_dataset: object
    eval_dataset: object | None
    data_collator: Callable[..., object]
    a5_adaptation_mode: str = "meta_ttt"
    stage_a_loss_step: StageALossStep | None = None
    meta_runner: MetaTTTEpisodeRunner | None = None
    episode_adapter: EpisodeAdapter | None = None
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None
    gradient_controller: OuterGradientController | None = None
    train_sampler_factory: TrainSamplerFactory | None = None
    callbacks: tuple[object, ...] = ()
    semantic_projector_delta_audit_steps: int = 0
    a5_parameter_delta_audit_steps: int = 0
    operator_diagnostics_interval: int = 10
    warmup_qwen_bitwise_auditor: _WarmupQwenBitwiseAuditor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ProductionStage) or not isinstance(self.model, nn.Module):
            raise TypeError("production runtime stage/model is invalid")
        if not callable(self.data_collator):
            raise TypeError("production runtime requires a data collator")
        if self.a5_adaptation_mode not in _A5_ADAPTATION_MODES:
            raise ValueError("production runtime has an invalid A5 adaptation mode")
        if self.stage is ProductionStage.A2 and self.a5_adaptation_mode != "meta_ttt":
            raise ValueError("A2 cannot select an A5 adaptation mode")
        if (
            type(self.semantic_projector_delta_audit_steps) is not int
            or self.semantic_projector_delta_audit_steps < 0
        ):
            raise ValueError("SemanticProjector delta audit steps must be non-negative")
        if (
            type(self.a5_parameter_delta_audit_steps) is not int
            or self.a5_parameter_delta_audit_steps < 0
        ):
            raise ValueError("A5 parameter delta audit steps must be non-negative")
        if self.stage is ProductionStage.A2 and self.a5_parameter_delta_audit_steps:
            raise ValueError("A2 cannot enable the A5 parameter delta audit")
        if self.stage is ProductionStage.A2 and self.warmup_qwen_bitwise_auditor is not None:
            raise ValueError("A2 cannot enable the warmup Qwen bitwise audit")
        if self.warmup_qwen_bitwise_auditor is not None and not isinstance(
            self.warmup_qwen_bitwise_auditor,
            _WarmupQwenBitwiseAuditor,
        ):
            raise TypeError("warmup Qwen bitwise auditor is invalid")
        if (
            type(self.operator_diagnostics_interval) is not int
            or self.operator_diagnostics_interval <= 0
        ):
            raise ValueError("Operator diagnostics interval must be positive")
        if self.stage is ProductionStage.A2:
            if not callable(self.stage_a_loss_step):
                raise ValueError("A2 runtime requires a post-forward state+answer loss step")
            if self.meta_runner is not None or self.episode_adapter is not None:
                raise ValueError("A2 runtime cannot expose Meta-TTT hooks")
        else:
            if not isinstance(self.meta_runner, MetaTTTEpisodeRunner) or not callable(
                self.episode_adapter
            ):
                raise ValueError("A5 runtime requires a Meta runner and episode adapter")
            if self.stage_a_loss_step is not None:
                raise ValueError("A5 runtime cannot expose the A2 loss hook")


class _LazyGradientAccumulationGroup(Sequence[object]):
    """Pull one A2 microbatch only when the pinned Trainer loop is ready to execute it."""

    def __init__(self, iterator: Iterator[object], expected_count: int) -> None:
        if expected_count <= 0:
            raise ValueError("lazy GA group requires a positive batch count")
        self.iterator = iterator
        self.expected_count = expected_count
        self._cache: list[object] = []
        self._started = time.perf_counter()

    def __len__(self) -> int:
        return self.expected_count

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> list[object]: ...

    def __getitem__(self, index: int | slice) -> object | list[object]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.expected_count))]
        normalized = index + self.expected_count if index < 0 else index
        if normalized < 0 or normalized >= self.expected_count:
            raise IndexError(index)
        while len(self._cache) <= normalized:
            self._pull_next()
        return self._cache[normalized]

    def __iter__(self) -> Iterator[object]:
        for index in range(self.expected_count):
            yield self[index]

    def _pull_next(self) -> None:
        microbatch_index = len(self._cache)
        wait_started = time.perf_counter()
        try:
            batch = next(self.iterator)
        except StopIteration as error:
            raise RuntimeError(
                "A2 DataLoader ended before the declared gradient-accumulation group"
            ) from error
        self._cache.append(batch)
        trace_event(
            "a2_ga_microbatch_fetch",
            seconds=time.perf_counter() - wait_started,
            microbatch_index=microbatch_index,
            requested_batches=self.expected_count,
            lazy=True,
        )
        if len(self._cache) == self.expected_count:
            trace_event(
                "a2_ga_group_fetch",
                seconds=time.perf_counter() - self._started,
                requested_batches=self.expected_count,
                fetched_batches=len(self._cache),
                lazy=True,
            )


class _A2AuditAccumulator:
    """Aggregate detached A2 audits across every microbatch since the last Trainer log."""

    def __init__(self) -> None:
        self._balance: list[OfficialWeakBalanceAudit] = []
        self._weak: list[OfficialWeakLossAudit] = []

    def add(
        self,
        balance: OfficialWeakBalanceAudit,
        weak: OfficialWeakLossAudit,
    ) -> None:
        if not isinstance(balance, OfficialWeakBalanceAudit) or not isinstance(
            weak, OfficialWeakLossAudit
        ):
            raise TypeError("A2 audit accumulator requires typed balance and weak audits")
        self._balance.append(balance)
        self._weak.append(weak)

    def flush(self) -> dict[str, float]:
        if not self._balance:
            if self._weak:
                raise RuntimeError("A2 weak audits drifted from balance audits")
            return {}
        if len(self._balance) != len(self._weak):
            raise RuntimeError("A2 balance and weak audit counts drifted")
        balances = tuple(self._balance)
        weak = tuple(self._weak)
        self._balance.clear()
        self._weak.clear()

        metrics: dict[str, float] = {
            "loss/ga_microbatch_count": float(len(balances)),
        }
        answer = _weighted_audit_mean(
            tuple((audit.answer_global_mean, audit.answer_global_count) for audit in balances)
        )
        state = _audit_mean(tuple(audit.state_global_mean for audit in balances))
        if answer is not None:
            metrics["loss/answer"] = answer
        if state is not None:
            metrics["loss/state"] = state
        if answer is not None and state is not None:
            metrics["loss/outer_total"] = answer + state

        for term_index, name in enumerate(("task", "operator", "retrieval", "time")):
            terms = tuple(audit.terms[term_index] for audit in balances)
            count = sum(float(term.global_valid_count.item()) for term in terms)
            metrics[f"loss/global_valid_count/{name}"] = count
            metrics[f"grad_balance/global_valid_count/{name}"] = count
            raw = _weighted_audit_mean(
                tuple((term.raw_global_mean, term.global_valid_count) for term in terms)
            )
            _set_optional_metric(metrics, f"loss/raw/{name}", raw)
            for key, values in (
                (f"loss/scale/{name}", tuple(term.scale for term in terms)),
                (
                    f"loss/aligned/{name}",
                    tuple(term.aligned_global_mean for term in terms),
                ),
                (
                    f"loss/weighted/{name}",
                    tuple(term.weighted_global_mean for term in terms),
                ),
                (
                    f"grad_balance/raw_rms/{name}",
                    tuple(term.raw_gradient_rms for term in terms),
                ),
                (
                    f"grad_balance/ema_rms/{name}",
                    tuple(term.ema_gradient_rms for term in terms),
                ),
                (
                    f"grad_balance/loss_scale/{name}",
                    tuple(term.loss_scale for term in terms),
                ),
                (
                    f"grad_balance/grad_scale/{name}",
                    tuple(term.gradient_scale for term in terms),
                ),
                (f"grad_balance/final_scale/{name}", tuple(term.scale for term in terms)),
            ):
                _set_optional_metric(metrics, key, _audit_mean(values))
            active_terms = tuple(
                term for term in terms if float(term.global_valid_count.item()) > 0
            )
            clamp_rate = (
                sum(float(term.scale_clamped.item()) for term in active_terms)
                / float(len(active_terms))
                if active_terms
                else 0.0
            )
            metrics[f"loss/scale_clamped/{name}"] = clamp_rate
            metrics[f"grad_balance/scale_clamped/{name}"] = clamp_rate

        for key, values in (
            (
                "loss/aux_to_answer_ratio",
                tuple(audit.auxiliary_to_answer_ratio for audit in balances),
            ),
            ("loss/group_guard", tuple(audit.group_guard for audit in balances)),
            (
                "loss/group_guard_active",
                tuple(audit.group_guard_active for audit in balances),
            ),
            (
                "loss/group_guard_reference",
                tuple(audit.group_guard_reference for audit in balances),
            ),
            (
                "loss/group_guard_reference_floored",
                tuple(audit.group_guard_reference_floored for audit in balances),
            ),
            (
                "loss/state_to_reference_ratio",
                tuple(audit.state_to_reference_ratio for audit in balances),
            ),
            (
                "loss/state_to_current_answer_ratio",
                tuple(audit.state_to_current_answer_ratio for audit in balances),
            ),
        ):
            value = _audit_mean(values)
            if value is not None:
                metrics[key] = value

        last = balances[-1]
        for name, mean, updates in zip(
            ("answer", "task", "operator", "retrieval", "time"),
            last.ema_means,
            last.ema_update_counts,
            strict=True,
        ):
            value = _audit_scalar(mean)
            if value is not None:
                metrics[f"loss/ema/{name}"] = value
            metrics[f"loss/ema_updates/{name}"] = float(updates.item())
        for name, updates in zip(
            ("task", "operator", "retrieval", "time"),
            last.gradient_ema_update_counts,
            strict=True,
        ):
            metrics[f"grad_balance/ema_updates/{name}"] = float(updates.item())

        for audit in weak:
            for name, value in audit.metrics():
                metrics[name] = metrics.get(name, 0.0) + float(value)
        _aggregate_operator_diagnostics(metrics, weak)
        _aggregate_task_diagnostics(metrics, weak)
        return metrics


def _aggregate_operator_diagnostics(
    metrics: dict[str, float],
    audits: Sequence[OfficialWeakLossAudit],
) -> None:
    raw = [0] * 72
    effective = [0] * 72
    loss_sums = [0.0] * 8
    support = [0] * 8
    confidence_sum = entropy_sum = temperature_sum = 0.0
    temperature_count = 0
    for audit in audits:
        values = audit.operator_diagnostics
        raw = [left + right for left, right in zip(raw, values.raw_confusion, strict=True)]
        effective = [
            left + right for left, right in zip(effective, values.effective_confusion, strict=True)
        ]
        loss_sums = [
            left + right for left, right in zip(loss_sums, values.class_loss_sums, strict=True)
        ]
        support = [left + right for left, right in zip(support, values.class_support, strict=True)]
        confidence_sum += values.confidence_sum
        entropy_sum += values.entropy_sum
        temperature_sum += values.temperature_sum
        temperature_count += values.temperature_count
    total = sum(support)
    if total != sum(raw) or total != sum(effective):
        raise RuntimeError("aggregated Operator confusion totals drifted from support")
    recalls: list[float] = []
    raw_recalls: list[float] = []
    raw_correct = effective_correct = 0
    unsupported_index = len(OPERATORS) - 1
    for target_index, target in enumerate(OPERATORS[:-1]):
        target_name = target.value
        row_offset = target_index * len(OPERATORS)
        row_support = support[target_index]
        metrics[f"operator/support/{target_name}"] = float(row_support)
        if row_support:
            target_raw_correct = raw[row_offset + target_index]
            target_effective_correct = effective[row_offset + target_index]
            raw_recall = target_raw_correct / float(row_support)
            recall = target_effective_correct / float(row_support)
            metrics[f"operator/raw_recall/{target_name}"] = raw_recall
            metrics[f"operator/recall/{target_name}"] = recall
            metrics[f"operator/raw_loss/{target_name}"] = loss_sums[target_index] / float(
                row_support
            )
            raw_recalls.append(raw_recall)
            recalls.append(recall)
            raw_correct += target_raw_correct
            effective_correct += target_effective_correct
        for predicted_index, predicted in enumerate(OPERATORS):
            metrics[f"operator/raw_confusion/{target_name}/{predicted.value}"] = float(
                raw[row_offset + predicted_index]
            )
            metrics[f"operator/effective_confusion/{target_name}/{predicted.value}"] = float(
                effective[row_offset + predicted_index]
            )
    metrics["operator/observed_class_count"] = float(len(recalls))
    metrics["operator/observed_class_fraction"] = len(recalls) / 8.0
    if total:
        metrics["operator/raw_micro_accuracy"] = raw_correct / float(total)
        metrics["operator/micro_accuracy"] = effective_correct / float(total)
        metrics["operator/raw_macro_recall"] = sum(raw_recalls) / float(len(raw_recalls))
        metrics["operator/macro_recall"] = sum(recalls) / float(len(recalls))
        metrics["operator/raw_predicted_unsupported_rate"] = sum(
            raw[index * 9 + unsupported_index] for index in range(8)
        ) / float(total)
        metrics["operator/predicted_unsupported_rate"] = sum(
            effective[index * 9 + unsupported_index] for index in range(8)
        ) / float(total)
        metrics["operator/mean_confidence"] = confidence_sum / float(total)
        metrics["operator/mean_entropy"] = entropy_sum / float(total)
    if temperature_count:
        metrics["operator/temperature"] = temperature_sum / float(temperature_count)


def _aggregate_task_diagnostics(
    metrics: dict[str, float],
    audits: Sequence[OfficialWeakLossAudit],
) -> None:
    count_loss = [0.0] * 4
    count_error = [0.0] * 4
    count_rows = [0] * 4
    component_loss = [0.0] * 3
    component_rows = [0] * 3
    o1_loss = [0.0] * 2
    o1_rows = [0] * 2
    channel_stats = [[0] * 7 for _ in range(6)]
    representable = unrepresentable = 0
    for audit in audits:
        values = audit.task_diagnostics
        count_loss = [
            left + right for left, right in zip(count_loss, values.count_loss_sums, strict=True)
        ]
        count_error = [
            left + right
            for left, right in zip(count_error, values.count_abs_error_sums, strict=True)
        ]
        count_rows = [
            left + right for left, right in zip(count_rows, values.count_rows, strict=True)
        ]
        component_loss = [
            left + right
            for left, right in zip(component_loss, values.component_loss_sums, strict=True)
        ]
        component_rows = [
            left + right for left, right in zip(component_rows, values.component_rows, strict=True)
        ]
        o1_loss = [left + right for left, right in zip(o1_loss, values.o1_loss_sums, strict=True)]
        o1_rows = [left + right for left, right in zip(o1_rows, values.o1_rows, strict=True)]
        sources = (
            values.channel_positive_counts,
            values.channel_negative_counts,
            values.channel_masked_counts,
            values.channel_true_positive_counts,
            values.channel_false_positive_counts,
            values.channel_false_negative_counts,
        )
        for target, source in zip(channel_stats, sources, strict=True):
            for index, value in enumerate(source):
                target[index] += value
        representable += values.e1_representable_occurrences
        unrepresentable += values.e1_unrepresentable_occurrences
    for index, name in enumerate(("o1", "o2", "e1", "e2")):
        metrics[f"task/count_rows/{name}"] = float(count_rows[index])
        if count_rows[index]:
            metrics[f"task/count/{name}"] = count_loss[index] / float(count_rows[index])
            metrics[f"task/count_mae/{name}"] = count_error[index] / float(count_rows[index])
    for index, name in enumerate(("e1_dense", "e2_event", "e2_phase")):
        metrics[f"task/{name}_rows"] = float(component_rows[index])
        if component_rows[index]:
            metrics[f"task/{name}"] = component_loss[index] / float(component_rows[index])
    for index, name in enumerate(("snap", "delta")):
        metrics[f"task/o1_rows/{name}"] = float(o1_rows[index])
        if o1_rows[index]:
            metrics[f"task/o1/{name}"] = o1_loss[index] / float(o1_rows[index])
    metrics["task/e1_representable_occurrences"] = float(representable)
    metrics["task/e1_unrepresentable_occurrences"] = float(unrepresentable)
    channel_names = (
        "e1_eventness",
        "e1_completion",
        "e1_transition",
        "e2_start",
        "e2_active",
        "e2_end",
        "e2_complete",
    )
    positive, negative, masked, true_positive, false_positive, false_negative = channel_stats
    for index, name in enumerate(channel_names):
        metrics[f"task/dense_positive/{name}"] = float(positive[index])
        metrics[f"task/dense_negative/{name}"] = float(negative[index])
        metrics[f"task/dense_masked/{name}"] = float(masked[index])
        precision_denominator = true_positive[index] + false_positive[index]
        recall_denominator = true_positive[index] + false_negative[index]
        if precision_denominator:
            metrics[f"task/dense_precision/{name}"] = true_positive[index] / float(
                precision_denominator
            )
        if recall_denominator:
            metrics[f"task/dense_recall/{name}"] = true_positive[index] / float(recall_denominator)


def _audit_scalar(value: Tensor) -> float | None:
    result = float(value.item())
    return result if math.isfinite(result) else None


def _audit_mean(values: Sequence[Tensor]) -> float | None:
    finite = tuple(value for value in (_audit_scalar(item) for item in values) if value is not None)
    return sum(finite) / float(len(finite)) if finite else None


def _weighted_audit_mean(values: Sequence[tuple[Tensor, Tensor]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        scalar = _audit_scalar(value)
        count = float(weight.item())
        if scalar is None or not math.isfinite(count) or count <= 0.0:
            continue
        numerator += scalar * count
        denominator += count
    return numerator / denominator if denominator > 0.0 else None


def _set_optional_metric(metrics: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        metrics[name] = value


def _query_tail_distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise ValueError("Query-tail distribution requires finite non-negative values")

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def _top_fraction_share(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values, reverse=True)
    if not ordered or not 0.0 < fraction <= 1.0:
        raise ValueError("top-fraction share requires values and a fraction in (0, 1]")
    total = math.fsum(ordered)
    if total == 0.0:
        return 0.0
    count = max(1, math.ceil(len(ordered) * fraction))
    return math.fsum(ordered[:count]) / total


def _counterfactual_query_selector(optimizer_step: int) -> int:
    """Select one shared Query ordinal for every rank in an exact-shape bucket."""

    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise ValueError("counterfactual optimizer step must be a positive integer")
    return optimizer_step


def _counterfactual_all_ranks_eligible(loss_weights: Sequence[float]) -> bool:
    """Only audit a step when every rank owns a real, non-padding episode."""

    if not loss_weights or any(weight not in (0.0, 1.0) for weight in loss_weights):
        raise ValueError("counterfactual rank eligibility requires binary loss weights")
    return all(weight == 1.0 for weight in loss_weights)


class TTTQwenTrainerMixin:
    """Mixin dynamically combined with remote ``CustomSeq2SeqTrainer``."""

    def __init__(
        self,
        *args: object,
        ttt_runtime: ProductionTrainerRuntime,
        **kwargs: object,
    ) -> None:
        self.ttt_runtime = ttt_runtime
        self.last_meta_output: TruncatedMetaTTTEpisodeOutput | None = None
        self.last_semantic_projector_metrics: dict[str, float] = {}
        self._a2_audit_accumulator = _A2AuditAccumulator()
        self._last_a5_training_seconds: float | None = None
        self._last_optimizer_log_time: float | None = None
        self._last_timed_global_step = -1
        self._last_counterfactual_metrics: dict[str, float] = {}
        self._counterfactual_audit_pending = False
        self._query_tail_history: dict[str, list[tuple[float, float, bool]]] = {
            "all": [],
            "intermediate": [],
            "final": [],
        }
        self._last_query_tail_global_step = -1
        super().__init__(*args, **kwargs)

    def _install_a2_deepspeed_gradient_control(self) -> None:
        if "deepspeed" not in str(getattr(self.accelerator, "distributed_type", "")).casefold():  # type: ignore[attr-defined]
            return
        controller = self.ttt_runtime.gradient_controller
        if not isinstance(controller, OuterGradientController):
            raise RuntimeError("formal A2 requires an Outer gradient controller")
        wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)  # type: ignore[attr-defined]
        if isinstance(wrapper, _ControlledDeepSpeedEngineWrapper):
            if wrapper.gradient_controller is not controller:
                raise RuntimeError("A2 DeepSpeed wrapper changed gradient controller")
            return
        engine = getattr(wrapper, "engine", None)
        if engine is None:
            raise RuntimeError("A2 DeepSpeed engine is unavailable before backward")
        self.accelerator.deepspeed_engine_wrapped = _ControlledDeepSpeedEngineWrapper(  # type: ignore[attr-defined]
            engine,
            controller,
            self.ttt_runtime.model,
            self.ttt_runtime.semantic_projector_delta_audit_steps,
        )

    def create_optimizer(self, *args: object, **kwargs: object) -> torch.optim.Optimizer:
        factory = self.ttt_runtime.optimizer_factory
        if getattr(self, "optimizer", None) is None and factory is not None:
            self.optimizer = factory(self.model)  # type: ignore[attr-defined]
        return cast(torch.optim.Optimizer, super().create_optimizer(*args, **kwargs))  # type: ignore[misc]

    def _get_train_sampler(self, train_dataset: object | None = None) -> object:
        dataset = self.train_dataset if train_dataset is None else train_dataset  # type: ignore[attr-defined]
        factory = cast(TrainSamplerFactory, self.ttt_runtime.train_sampler_factory)
        sampler = factory(
            dataset,
            int(self.args.process_index),  # type: ignore[attr-defined]
            int(self.args.world_size),  # type: ignore[attr-defined]
        )
        self._ttt_train_sampler = sampler
        return sampler

    def get_batch_samples(
        self,
        epoch_iterator: Iterator[object],
        num_batches: int,
        device: torch.device,
    ) -> tuple[Sequence[object], Tensor | int | None]:
        """Return a lazy A2 GA group for the pinned Transformers 4.57.1 loop.

        A2 batches deliberately carry no conventional ``labels`` entry: the typed loss hook
        owns all supervision. Returning ``None`` for ``num_items_in_batch`` preserves upstream
        loss scaling. The outer loop observes the declared group length but each ``next()`` runs
        only immediately before its corresponding forward/backward. A5 stays on the upstream path.
        """

        if self.ttt_runtime.stage is not ProductionStage.A2:
            return cast(
                tuple[Sequence[object], Tensor | int | None],
                super().get_batch_samples(epoch_iterator, num_batches, device),  # type: ignore[misc]
            )
        if transformers.__version__ != "4.57.1":
            raise RuntimeError(
                "lazy A2 gradient accumulation is pinned to Transformers 4.57.1; "
                f"found {transformers.__version__}"
            )
        return _LazyGradientAccumulationGroup(epoch_iterator, num_batches), None

    def log(self, logs: dict[str, float], *args: object, **kwargs: object) -> None:
        enriched = dict(logs)
        if self.ttt_runtime.stage is ProductionStage.A2:
            enriched.update(self._a2_audit_accumulator.flush())
            global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
            if global_step % self.ttt_runtime.operator_diagnostics_interval:
                enriched = {
                    name: value
                    for name, value in enriched.items()
                    if not name.startswith("operator/raw_confusion/")
                    and not name.startswith("operator/effective_confusion/")
                }
        else:
            audit = getattr(self.ttt_runtime.meta_runner, "last_balance_audit", None)
            metrics = getattr(audit, "metrics", None)
            if callable(metrics):
                for name, value in metrics():
                    if value is not None:
                        enriched[name] = float(value)
        if self.ttt_runtime.stage is ProductionStage.A5 and self.last_meta_output is not None:
            now = time.perf_counter()
            global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
            if global_step != self._last_timed_global_step:
                if self._last_optimizer_log_time is not None:
                    enriched["a5/optimizer_step_seconds"] = now - self._last_optimizer_log_time
                self._last_optimizer_log_time = now
                self._last_timed_global_step = global_step
            if self._last_a5_training_seconds is not None:
                enriched["a5/training_step_seconds"] = self._last_a5_training_seconds
            enriched.update(self._last_counterfactual_metrics)
            retrieval_metrics: dict[str, float] = {}
            meta_output = self.last_meta_output
            meta_audit = meta_output.audit
            if meta_audit.loss_weight == 1.0:
                for query in meta_audit.queries:
                    for name, value in query.metrics.metrics:
                        if name.startswith("retrieval/") and value is not None:
                            retrieval_metrics[name] = retrieval_metrics.get(name, 0.0) + value
            balance = getattr(self.ttt_runtime.meta_runner, "last_balance_audit", None)
            if isinstance(balance, OfficialWeakBalanceAudit):
                retrieval_metrics["retrieval/valid_bag_rows"] = float(
                    balance.terms[2].global_valid_count.item()
                )
            enriched.update(retrieval_metrics)
            enriched.update(
                {
                    "a5/meta_query_count": float(meta_audit.query_count),
                    "a5/diagnostic_query_count": float(meta_audit.diagnostic_query_count),
                    "a5/zero_support_query_count": float(meta_audit.zero_support_query_count),
                    "a5/support_segments_without_query": float(
                        meta_audit.support_segments_without_query
                    ),
                    "a5/meta_ttt_segment_count": float(meta_audit.meta_ttt_segment_count),
                    "a5/outer_only_segment_count": float(meta_audit.outer_only_segment_count),
                    "a5/no_write_segment_count": float(meta_audit.no_write_segment_count),
                    "a5/ablation/ttt_enabled": float(meta_audit.ttt_enabled),
                    "a5/memory/write_valid": float(meta_audit.associative_valid_count > 0),
                    "a5/memory/write_count": float(meta_audit.write_count),
                    "a5/memory/write_attempt_count": float(meta_audit.write_attempt_count),
                    "a5/memory/skip_count": float(meta_audit.skip_count),
                    "a5/ablation/memory_interface_trainable": float(
                        self.ttt_runtime.a5_adaptation_mode == "meta_ttt"
                    ),
                    "a5/ablation/w0_outer_trainable": 1.0,
                    "a5/insufficient_inter_query_gap": float(
                        meta_audit.insufficient_inter_query_gap
                    ),
                    "a5/loss/meta_query_sum": float(meta_output.query_loss.item()),
                    "memory/readout_target_cosine": meta_audit.readout_target_cosine_mean,
                    "memory/pre_write_error_mean": (
                        1.0 - meta_audit.readout_target_cosine_mean
                    ),
                    "memory/post_write_cosine": meta_audit.post_write_cosine_mean,
                    "memory/readout_share": meta_audit.readout_share_mean,
                    "memory/memory_norm_max": meta_audit.memory_norm_max,
                    "memory/write_norm": meta_audit.write_norm_mean,
                    "memory/eta_sum": meta_audit.eta_sum_mean,
                    "memory/eta_renormalized_fraction": meta_audit.renormalized_fraction,
                    "memory/slots_written": float(meta_audit.slots_written_total),
                    "a5/memory/bank_record_count": float(meta_audit.bank_record_count),
                    "a5/memory/empty_bank_count": float(meta_audit.empty_bank_count),
                }
            )
            role_norms: dict[str, list[float]] = {}
            role_clipped_norms: dict[str, list[float]] = {}
            role_losses: dict[str, list[float]] = {}
            for query in meta_audit.queries:
                proxy_norm = math.sqrt(sum(value * value for value in query.proxy_gradient_norms))
                enriched[f"a5/query_weight/{query.query_role}"] = query.query_weight
                enriched[
                    f"a5/query_proxy_grad_norm/query_{query.query_index}_{query.query_role}"
                ] = proxy_norm
                enriched[
                    f"a5/query_proxy_grad_norm/raw/query_{query.query_index}_{query.query_role}"
                ] = query.proxy_gradient_joint_norm_raw
                enriched[
                    f"a5/query_proxy_grad_norm/clipped/query_{query.query_index}_{query.query_role}"
                ] = query.proxy_gradient_joint_norm_clipped
                enriched[
                    f"a5/query_proxy_grad_clip_scale/query_{query.query_index}_{query.query_role}"
                ] = query.proxy_gradient_clip_scale
                enriched[
                    f"a5/query_proxy_grad_clipped/query_{query.query_index}_{query.query_role}"
                ] = float(query.proxy_gradient_clipped)
                enriched[
                    f"a5/query_proxy_grad_nonzero/query_{query.query_index}_{query.query_role}"
                ] = float(query.proxy_gradient_status == "nonzero")
                status_key = f"a5/query_proxy_grad_status/{query.proxy_gradient_status}"
                enriched[status_key] = enriched.get(status_key, 0.0) + 1.0
                enriched[f"a5/query_outer_loss/query_{query.query_index}_{query.query_role}"] = (
                    query.weighted_outer_loss
                )
                role_norms.setdefault(query.query_role, []).append(proxy_norm)
                role_clipped_norms.setdefault(query.query_role, []).append(
                    query.proxy_gradient_joint_norm_clipped
                )
                role_losses.setdefault(query.query_role, []).append(query.weighted_outer_loss)
            if global_step != self._last_query_tail_global_step:
                for query in meta_audit.queries:
                    if query.proxy_gradient_status == "zero_padding":
                        continue
                    item = (
                        query.proxy_gradient_joint_norm_raw,
                        query.proxy_gradient_joint_norm_clipped,
                        query.proxy_gradient_clipped,
                    )
                    self._query_tail_history["all"].append(item)
                    self._query_tail_history[query.query_role].append(item)
                for history_values in self._query_tail_history.values():
                    del history_values[:-256]
                self._last_query_tail_global_step = global_step
            for role, role_values in role_norms.items():
                enriched[f"a5/query_proxy_grad_norm/{role}"] = sum(role_values) / len(role_values)
            for role, role_values in role_clipped_norms.items():
                enriched[f"a5/query_proxy_grad_norm/clipped/{role}"] = sum(role_values) / len(
                    role_values
                )
            for role, role_values in role_losses.items():
                enriched[f"a5/query_outer_loss/{role}"] = sum(role_values) / len(role_values)
            for role, history in self._query_tail_history.items():
                if not history:
                    continue
                raw_values = tuple(item[0] for item in history)
                clipped_values = tuple(item[1] for item in history)
                for label, distribution_values in (
                    ("raw", raw_values),
                    ("clipped", clipped_values),
                ):
                    for statistic, value in _query_tail_distribution(distribution_values).items():
                        enriched[f"a5/query_proxy_grad_norm/{label}/recent_{role}/{statistic}"] = (
                            value
                        )
                enriched[f"a5/query_proxy_grad_clip_rate/recent_{role}"] = sum(
                    float(item[2]) for item in history
                ) / len(history)
                enriched[f"a5/query_proxy_grad_count/recent_{role}"] = float(len(history))
            all_history = self._query_tail_history["all"]
            if all_history:
                for label, index in (("raw", 0), ("clipped", 1)):
                    share_values = tuple(item[index] for item in all_history)
                    enriched[f"a5/query_proxy_grad_norm/{label}/recent_top1_share"] = (
                        _top_fraction_share(share_values, 0.01)
                    )
                    enriched[f"a5/query_proxy_grad_norm/{label}/recent_top5_share"] = (
                        _top_fraction_share(share_values, 0.05)
                    )
            for segment in meta_audit.segments:
                enriched[f"a5/deferred_vjp_norm/segment_{segment.segment_index}"] = (
                    segment.deferred_vjp_norm
                )
                enriched[f"a5/write_version_at_query/segment_{segment.segment_index}"] = float(
                    max(segment.write_version_at_query)
                )
                enriched[f"a5/write_version_delta/segment_{segment.segment_index}"] = float(
                    max(segment.write_version_at_query)
                    - max(segment.write_version_before_segment)
                )
                enriched[f"a5/write_attempt_count/segment_{segment.segment_index}"] = float(
                    segment.write_attempt_count
                )
                enriched[f"a5/write_count/segment_{segment.segment_index}"] = float(
                    segment.write_count
                )
                enriched[f"a5/skip_count/segment_{segment.segment_index}"] = float(
                    segment.skip_count
                )
                enriched[f"a5/query_count/segment_{segment.segment_index}"] = float(
                    segment.query_count
                )
                enriched[f"a5/query_cotangent_sum_norm/raw/segment_{segment.segment_index}"] = (
                    segment.raw_query_cotangent_sum_norm
                )
                enriched[f"a5/query_cotangent_sum_norm/clipped/segment_{segment.segment_index}"] = (
                    segment.clipped_query_cotangent_sum_norm
                )
                enriched[f"a5/meta_ttt_active/segment_{segment.segment_index}"] = float(
                    segment.training_mode == "meta_ttt"
                )
                enriched[f"a5/outer_only/segment_{segment.segment_index}"] = float(
                    segment.training_mode == "outer_only"
                )
                enriched[f"a5/no_write/segment_{segment.segment_index}"] = float(
                    segment.training_mode == "no_write"
                )
                enriched[f"memory/pre_write_cosine/segment_{segment.segment_index}"] = (
                    segment.pre_write_cosine_mean
                )
                enriched[f"memory/eta_sum/segment_{segment.segment_index}"] = segment.eta_sum
                for reason, count in segment.skip_reason_counts:
                    enriched[f"a5/skip_reason/segment_{segment.segment_index}/{reason}"] = float(
                        count
                    )
            if meta_audit.writes:
                write_norms = tuple(
                    value for write in meta_audit.writes for value in write.write_norms
                )
                memory_norms = tuple(
                    value for write in meta_audit.writes for value in write.memory_norms
                )
                readout_shares = tuple(
                    value for write in meta_audit.writes for value in write.readout_shares
                )
                enriched["a5/memory/write_norm_mean"] = sum(write_norms) / len(write_norms)
                enriched["a5/memory/memory_norm_mean"] = sum(memory_norms) / len(memory_norms)
                enriched["a5/memory/readout_share_mean"] = sum(readout_shares) / len(
                    readout_shares
                )
                pairwise_fields = (
                    ("key_pairwise_cosine_mean", "key_pairwise_cosine_means"),
                    ("value_pairwise_cosine_mean", "value_pairwise_cosine_means"),
                    ("delta_pairwise_cosine_mean", "delta_pairwise_cosine_means"),
                )
                for metric_name, field_name in pairwise_fields:
                    written_values = tuple(
                        value
                        for write in meta_audit.writes
                        for did_write, value in zip(
                            write.did_write,
                            getattr(write, field_name),
                            strict=True,
                        )
                        if did_write
                    )
                    if written_values:
                        enriched[f"a5/memory/{metric_name}"] = sum(written_values) / len(
                            written_values
                        )
                    # The episode mean hides whether the per-video forward loop
                    # tightens the collapse as the video proceeds: `M` feeds the
                    # readout, which feeds the slots, which produce the next
                    # chunk's keys.  `meta_audit.writes` is ordered by Support
                    # index, so the first and last entries bracket that loop.
                    for label, write in (
                        ("first", meta_audit.writes[0]),
                        ("last", meta_audit.writes[-1]),
                    ):
                        bounded = tuple(
                            value
                            for did_write, value in zip(
                                write.did_write,
                                getattr(write, field_name),
                                strict=True,
                            )
                            if did_write
                        )
                        if bounded:
                            enriched[f"a5/memory/{metric_name}/{label}_write"] = sum(
                                bounded
                            ) / len(bounded)
                for probe_field in PROBE_FIELDS:
                    probe_values = tuple(
                        getattr(probe, probe_field)
                        for write in meta_audit.writes
                        if write.slot_geometry_probes
                        for did_write, probe in zip(
                            write.did_write,
                            write.slot_geometry_probes,
                            strict=True,
                        )
                        if did_write and probe is not None
                    )
                    if probe_values:
                        enriched[f"a5/geometry/{probe_field}"] = sum(probe_values) / len(
                            probe_values
                        )
                # `memory/readout_target_cosine` is diluted by construction: it is
                # a flat mean over every accepted write, and each video's FIRST
                # write lands on a zero memory, so its recall is the zero vector
                # and the masked cosine contributes an exact 0.0 while still
                # counting in the denominator.  That is not a recall measurement,
                # it is an arithmetic artifact, and it depresses the reported
                # value well below the steady state.  Emit the recall-only mean
                # alongside, selecting on the pre-write memory generation rather
                # than on `value == 0.0` so a legitimately orthogonal recall is
                # never silently dropped.
                recall_values = tuple(
                    value
                    for write in meta_audit.writes
                    for did_write, before, value in zip(
                        write.did_write,
                        write.write_versions_before,
                        write.pre_write_cosine_means,
                        strict=True,
                    )
                    if did_write and before > 0
                )
                if recall_values:
                    enriched["a5/memory/readout_target_cosine_recall_only"] = sum(
                        recall_values
                    ) / len(recall_values)
                skip_reasons = tuple(
                    reason
                    for write in meta_audit.writes
                    for did_write, reason in zip(
                        write.did_write,
                        write.skip_reasons,
                        strict=True,
                    )
                    if not did_write
                )
                enriched["a5/memory/skip_count/no_valid_slot"] = float(
                    sum(reason == "no_valid_slot" for reason in skip_reasons)
                )
                enriched["a5/memory/skip_count/nonfinite"] = float(
                    sum(reason == "nonfinite_key_value" for reason in skip_reasons)
                )
        enriched.update(self.last_semantic_projector_metrics)
        controller = self.ttt_runtime.gradient_controller
        if isinstance(controller, OuterGradientController) and controller.last_audit is not None:
            enriched.update(dict(controller.last_audit.metrics()))
        super().log(enriched, *args, **kwargs)  # type: ignore[misc]

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, object],
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        if self.ttt_runtime.stage is ProductionStage.A2:
            step = cast(StageALossStep, self.ttt_runtime.stage_a_loss_step)
            loss = step(model, inputs)
            controller = self.ttt_runtime.gradient_controller
            if not isinstance(controller, OuterGradientController):
                raise RuntimeError("formal A2 requires an Outer gradient controller")
            return sanitize_scalar_loss(
                loss,
                source="A2 state+answer",
                controller=controller,
            )
        return cast(Tensor, super().compute_loss(model, inputs, *args, **kwargs))  # type: ignore[misc]

    def training_step(
        self,
        model: nn.Module,
        inputs: Mapping[str, object],
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor:
        step_started = time.perf_counter()
        if self.ttt_runtime.stage is ProductionStage.A2:
            self._install_a2_deepspeed_gradient_control()
            result = cast(
                Tensor,
                super().training_step(  # type: ignore[misc]
                    model,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                ),
            )
            marker = getattr(self.ttt_runtime.stage_a_loss_step, "mark_backward_returned", None)
            if callable(marker):
                marker()
            balance_audit = getattr(
                self.ttt_runtime.stage_a_loss_step,
                "last_balance_audit",
                None,
            )
            weak_audit = getattr(
                self.ttt_runtime.stage_a_loss_step,
                "last_weak_audit",
                None,
            )
            if not isinstance(balance_audit, OfficialWeakBalanceAudit) or not isinstance(
                weak_audit, OfficialWeakLossAudit
            ):
                raise RuntimeError("formal A2 step did not publish typed loss audits")
            self._a2_audit_accumulator.add(balance_audit, weak_audit)
            wrapper = getattr(self.accelerator, "deepspeed_engine_wrapped", None)  # type: ignore[attr-defined]
            if not isinstance(wrapper, _ControlledDeepSpeedEngineWrapper):
                raise RuntimeError("formal A2 lost its controlled DeepSpeed wrapper")
            auditor = wrapper.semantic_projector_auditor
            self.last_semantic_projector_metrics = (
                dict(auditor.last_metrics) if auditor is not None else {}
            )
            self._observe_runtime_cost(inputs, time.perf_counter() - step_started)
            return result
        if int(self.args.gradient_accumulation_steps) != 1:  # type: ignore[attr-defined]
            raise ValueError("A5 uses one complete episode/rank and episode-level GA=1")
        warmup_qwen_auditor = self.ttt_runtime.warmup_qwen_bitwise_auditor
        if warmup_qwen_auditor is not None:
            trainer_state = getattr(self, "state", None)
            warmup_qwen_auditor.capture_post_prepare_baseline(
                global_step=int(getattr(trainer_state, "global_step", 0))
            )
        model.train()
        optimizer = getattr(self, "optimizer", None)
        optimizer_train = getattr(optimizer, "train", None)
        if callable(optimizer_train):
            optimizer_train()
        prepared = self._prepare_inputs(dict(inputs))  # type: ignore[attr-defined]
        adapter = cast(EpisodeAdapter, self.ttt_runtime.episode_adapter)
        episode, loss_weight = adapter(prepared)
        if loss_weight not in (0.0, 1.0):
            raise ValueError("A5 episode loss weight must be one or deterministic-padding zero")
        runner = cast(MetaTTTEpisodeRunner, self.ttt_runtime.meta_runner)
        counterfactual_config = runner.config.a5.counterfactual_audit
        trainer_state = getattr(self, "state", None)
        next_optimizer_step = int(getattr(trainer_state, "global_step", 0)) + 1
        audit_due_now = bool(
            counterfactual_config.enabled
            and next_optimizer_step % counterfactual_config.interval_steps == 0
        )
        self._counterfactual_audit_pending = self._counterfactual_audit_pending or audit_due_now
        segment_lengths = episode.segment_lengths
        expected_segments = len(segment_lengths)
        expected_backwards = len(episode.query_points) + expected_segments
        self._assert_rank_episode_parity(
            segment_lengths,
            episode.segment_query_counts,
        )
        counterfactual_request: CounterfactualAuditRequest | None = None
        if counterfactual_config.enabled and self._counterfactual_audit_pending:
            local_eligibility = torch.tensor(
                [loss_weight],
                dtype=torch.float32,
                device=self.args.device,  # type: ignore[attr-defined]
            )
            gathered_eligibility = self.accelerator.gather(local_eligibility)  # type: ignore[attr-defined]
            rank_loss_weights = tuple(
                float(value) for value in gathered_eligibility.detach().cpu().reshape(-1).tolist()
            )
            if _counterfactual_all_ranks_eligible(rank_loss_weights):
                # Every rank in the exact-shape bucket must audit the same Query
                # ordinal.  A rank-dependent ordinal, or auditing while one rank
                # is deterministic padding, can interleave local no-grad forwards
                # with different distributed backward collectives and deadlock.
                counterfactual_request = CounterfactualAuditRequest(
                    optimizer_step=next_optimizer_step,
                    query_selector=_counterfactual_query_selector(next_optimizer_step),
                    queries_per_rank=counterfactual_config.queries_per_rank,
                )
                self._counterfactual_audit_pending = False
            else:
                trace_event(
                    "a5_diagnostic_counterfactual_deferred",
                    optimizer_step=next_optimizer_step,
                    scheduled_now=audit_due_now,
                    reason="padding_rank",
                    rank_loss_weights=rank_loss_weights,
                )

        backward_controller = SegmentBackwardController(
            self.accelerator,  # type: ignore[attr-defined]
            model,
            expected_count=expected_backwards,
            gradient_controller=self.ttt_runtime.gradient_controller,
            semantic_projector_delta_audit_steps=(
                self.ttt_runtime.semantic_projector_delta_audit_steps
            ),
            a5_parameter_delta_audit_steps=(self.ttt_runtime.a5_parameter_delta_audit_steps),
        )

        def distributed_backward(loss: Tensor, retain_graph: bool) -> None:
            backward_controller.backward(loss * loss_weight, retain_graph=retain_graph)

        end_prefetch = getattr(adapter, "end_prefetch", None)
        try:
            output = runner.run_truncated(
                episode,
                backward=distributed_backward,
                backward_gradient_scale=backward_controller.proxy_gradient_scale,
                episode_loss_weight=loss_weight,
                counterfactual_audit=counterfactual_request,
            )
        finally:
            if callable(end_prefetch):
                end_prefetch()
        if output.audit.backward_count != expected_backwards:
            raise RuntimeError("A5 streamed backward collective count drifted from its bucket")
        backward_controller.finalize()
        self.last_semantic_projector_metrics = backward_controller.semantic_projector_metrics
        self.last_meta_output = output
        self._last_counterfactual_metrics = self._gather_counterfactual_metrics(output)
        local_training_seconds = time.perf_counter() - step_started
        timing = torch.tensor(
            [local_training_seconds],
            dtype=torch.float64,
            device=self.args.device,  # type: ignore[attr-defined]
        )
        gathered_timing = self.accelerator.gather(timing)  # type: ignore[attr-defined]
        self._last_a5_training_seconds = float(gathered_timing.max().item())
        self._observe_runtime_cost(inputs, local_training_seconds)
        return (output.total * loss_weight).detach().to(self.args.device)  # type: ignore[attr-defined]

    def _gather_counterfactual_metrics(
        self,
        output: TruncatedMetaTTTEpisodeOutput,
    ) -> dict[str, float]:
        audited = tuple(
            query.counterfactual
            for query in output.audit.queries
            if query.counterfactual is not None
        )
        local = torch.zeros((9,), dtype=torch.float64, device=self.args.device)  # type: ignore[attr-defined]
        for counterfactual in audited:
            references = {item.reference: item for item in counterfactual.references}
            if set(references) != {"episode_zero", "segment_start"}:
                raise RuntimeError("counterfactual audit did not publish both references")
            local[0] += 1.0
            for offset, name in ((1, "episode_zero"), (5, "segment_start")):
                item = references[name]
                local[offset : offset + 4] += torch.tensor(
                    (
                        item.gain_abs,
                        item.gain_rel,
                        float(item.gain_abs > 0.0),
                        item.descent_cosine,
                    ),
                    dtype=torch.float64,
                    device=local.device,
                )
        gathered = self.accelerator.gather(local)  # type: ignore[attr-defined]
        rows = gathered.reshape(int(self.args.world_size), 9)  # type: ignore[attr-defined]
        count = float(rows[:, 0].sum().item())
        if count == 0.0:
            return {}
        metrics = {"a5/cf/audited_query_count": count}
        for offset, name in ((1, "episode_zero"), (5, "segment_start")):
            metrics.update(
                {
                    f"a5/cf/query_gain_abs/{name}": float(rows[:, offset].sum().item()) / count,
                    f"a5/cf/query_gain_rel/{name}": float(rows[:, offset + 1].sum().item()) / count,
                    f"a5/cf/gain_positive_rate/{name}": float(rows[:, offset + 2].sum().item())
                    / count,
                    f"a5/cf/descent_cosine/{name}": float(rows[:, offset + 3].sum().item()) / count,
                }
            )
        return metrics

    def _observe_runtime_cost(
        self,
        inputs: Mapping[str, object],
        seconds: float,
    ) -> None:
        prepared = inputs.get(
            "prepared_a2" if self.ttt_runtime.stage is ProductionStage.A2 else "prepared_a5"
        )
        record = getattr(prepared, "record", None)
        preparation_seconds = 0.0
        record_id: str | None = None
        if isinstance(record, A2QueryRecord):
            record_id = record.query.runtime.query_id
            telemetry = getattr(prepared, "preparation", None)
            raw_seconds = getattr(telemetry, "collate_seconds", 0.0)
            if isinstance(raw_seconds, (int, float)):
                preparation_seconds = float(raw_seconds)
        elif isinstance(record, A5EpisodeRecord):
            record_id = record.episode_id
            answers = getattr(prepared, "query_answers", ())
            for answer in answers if isinstance(answers, tuple) else ():
                telemetry = getattr(answer, "preparation", None)
                raw_seconds = getattr(telemetry, "total_seconds", 0.0)
                if isinstance(raw_seconds, (int, float)):
                    preparation_seconds += float(raw_seconds)
        if record_id is None:
            return
        total_seconds = preparation_seconds + seconds
        trace_event(
            "runtime_cost_observation",
            record_id=record_id,
            preparation_seconds=preparation_seconds,
            training_seconds=seconds,
            seconds=total_seconds,
        )
        sampler = getattr(self, "_ttt_train_sampler", None)
        observe = getattr(sampler, "observe_runtime_cost", None)
        if callable(observe):
            observe(record_id, total_seconds)

    def _assert_rank_episode_parity(
        self,
        segment_lengths: tuple[int, ...],
        segment_query_counts: tuple[int, ...],
    ) -> None:
        device = self.args.device  # type: ignore[attr-defined]
        local = torch.tensor(
            tuple(
                value
                for pair in zip(
                    segment_lengths,
                    segment_query_counts,
                    strict=True,
                )
                for value in pair
            ),
            dtype=torch.int64,
            device=device,
        )
        gathered = self.accelerator.gather(local)  # type: ignore[attr-defined]
        world_size = int(self.args.world_size)  # type: ignore[attr-defined]
        signatures = tuple(
            tuple(int(value) for value in row)
            for row in gathered.detach().cpu().reshape(world_size, -1).tolist()
        )
        if len(set(signatures)) != 1:
            raise ValueError(
                f"A5 ranks received unequal segment lengths or Query counts: {signatures}"
            )


def build_trainer_class(base: type) -> type:
    """Create a concrete class without importing the remote checkout at module import time."""

    if not isinstance(base, type):
        raise TypeError("LLaMA-Factory Trainer base must be a class")
    return type("TTTQwenLlamaFactoryTrainer", (TTTQwenTrainerMixin, base), {})


def build_production_trainer(
    backbone: LlamaFactoryBackboneBundle,
    runtime: ProductionTrainerRuntime,
) -> object:
    if not callable(runtime.optimizer_factory) or not callable(runtime.train_sampler_factory):
        raise ValueError("production bridge must inject optimizer and rank-aware sampler factories")
    backbone_ids = {id(parameter) for parameter in backbone.model.parameters()}
    runtime_ids = {id(parameter) for parameter in runtime.model.parameters()}
    if not backbone_ids or not backbone_ids <= runtime_ids:
        raise ValueError("production runtime model must register the exact loaded Qwen backbone")
    trainer_class = build_trainer_class(backbone.symbols.trainer_base)
    return trainer_class(
        model=runtime.model,
        args=backbone.training_args,
        finetuning_args=backbone.finetuning_args,
        processor=backbone.processor,
        model_args=backbone.model_args,
        tokenizer=backbone.tokenizer,
        train_dataset=runtime.train_dataset,
        eval_dataset=runtime.eval_dataset,
        data_collator=runtime.data_collator,
        callbacks=list(runtime.callbacks),
        ttt_runtime=runtime,
    )


def _destroy_default_process_group() -> None:
    """Best-effort teardown for the process group created by Accelerate/DeepSpeed.

    Do not add a final barrier here: when one rank raises, a cleanup barrier would turn the
    original failure into another distributed hang.  ``destroy_process_group`` is local
    teardown and is safe to call from the entrypoint ``finally`` block on both success and
    failure paths.
    """

    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return
    try:
        distributed.destroy_process_group()
    except Exception as error:  # pragma: no cover - backend-specific defensive boundary
        warnings.warn(
            f"failed to destroy the default distributed process group: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def main(argv: list[str] | None = None) -> int:
    try:
        return _run_main(argv)
    finally:
        _destroy_default_process_group()


def _run_main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise ValueError("usage: python -m ttt_svcbench_qwen.llamafactory_trainer CONFIG.yaml")
    started = time.monotonic()
    backbone = load_llamafactory_backbone(arguments[0])
    configured_stage = ProductionStage(backbone.ttt_config.stage)
    requested_adaptation_mode = os.environ.get("TTT_A5_ADAPTATION_MODE")
    if (
        requested_adaptation_mode is not None
        and requested_adaptation_mode != backbone.ttt_config.a5_adaptation_mode
    ):
        raise ValueError("TTT_A5_ADAPTATION_MODE disagrees with ttt_qwen.a5_adaptation_mode")
    trainability_audit = configure_qwen_outer_trainability(
        backbone.model,
        backbone.project_config,
        backbone.ttt_config.qwen_outer_trainability,
    )
    full_unfreeze_audit = (
        fully_unfreeze_qwen(backbone.model, backbone.project_config)
        if trainability_audit.mode == "full"
        else None
    )
    if getattr(backbone.training_args, "resume_from_checkpoint", None) is not None:
        raise ValueError(
            "set TTT_RESUME_CHECKPOINT for same-stage resume; YAML resume is forbidden"
        )
    same_stage_resume = resolve_same_stage_resume(
        os.environ.get("TTT_RESUME_CHECKPOINT"),
        configured_stage,
        a5_adaptation_mode=backbone.ttt_config.a5_adaptation_mode,
    )
    if backbone.ttt_config.a5_phase == "fast_state_warmup" and same_stage_resume is not None:
        raise ValueError("Memory/State warmup is restart-only and cannot resume")
    if same_stage_resume is not None:
        _validate_resume_balance_schema(same_stage_resume)
    from ttt_svcbench_qwen.production_runtime import _video_pixel_bounds, build_runtime

    runtime_raw = build_runtime(backbone, backbone.ttt_config)
    if not isinstance(runtime_raw, ProductionTrainerRuntime):
        raise TypeError("built-in runtime must return ProductionTrainerRuntime")
    if runtime_raw.stage is not configured_stage:
        raise ValueError("runtime factory stage disagrees with ttt_qwen.stage")
    manifest_path = backbone.ttt_config.dataset_manifest
    train_dataset, eval_dataset = load_production_manifest_views(
        manifest_path,
        stage=ManifestStage(configured_stage.value),
    )
    runtime_raw = replace(
        runtime_raw,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    visual_cost_index: Mapping[str, VisualCostRecord] | None = None
    raw_cost_index = backbone.ttt_config.visual_cost_index
    if raw_cost_index is not None:
        minimum_pixels, maximum_pixels = _video_pixel_bounds(backbone)
        balance = backbone.project_config.loss.official_weak_balance
        model_name = str(getattr(backbone.model_args, "model_name_or_path", "unknown-model"))
        revision = str(getattr(backbone.model_args, "revision", "unknown-revision"))
        parameter = next(backbone.model.parameters())
        expected_fingerprint = make_visual_cost_fingerprint(
            manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            model_revision=f"{model_name}@{revision}",
            transformers_version=transformers.__version__,
            processor=(
                f"{type(backbone.processor).__module__}.{type(backbone.processor).__qualname__}"
            ),
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            dtype=str(parameter.dtype).removeprefix("torch."),
            visual_batch_size=backbone.ttt_config.support_visual_batch_size,
            cache_mode=backbone.ttt_config.preprocess_cache_mode,
            loss_mode="ema_answer_ref",
            loss_group_weight=balance.group_weight,
            loss_scale_min=balance.scale_min,
            loss_scale_max=balance.scale_max,
            loss_epsilon=balance.epsilon,
            gpu_model=(
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            ),
            query_decode_strategy="grouped_seek",
            query_decode_max_groups=backbone.ttt_config.query_decode_max_groups,
            state_query_visual_mode=backbone.ttt_config.state_query_visual_mode,
            state_query_max_frames=backbone.ttt_config.state_query_max_frames,
            answer_query_visual_mode=backbone.ttt_config.answer_query_visual_mode,
            answer_query_max_frames=backbone.ttt_config.answer_query_max_frames,
            query_sample_fps=backbone.ttt_config.query_sample_fps,
        )
        visual_cost_index = load_visual_cost_index(
            raw_cost_index,
            expected_fingerprint=expected_fingerprint,
            require_runtime_measurements=(
                backbone.ttt_config.visual_cost_mode == "exact_tokens_then_runtime"
            ),
        )
    checkpoint_audit: OuterCheckpointAudit | None = None
    warmup_bundle_audit: dict[str, object] | None = None
    warmup_trainability_audit: WarmupOuterTrainabilityAudit | None = None
    if configured_stage is ProductionStage.A5 and same_stage_resume is None:
        checkpoint = backbone.ttt_config.initialize_from_a2_checkpoint
        if checkpoint is None:
            raise RuntimeError("validated A5 config lost initialize_from_a2_checkpoint")
        _validate_resume_balance_schema(Path(checkpoint).expanduser().resolve())
        checkpoint_audit = initialize_outer_model_from_a2(runtime_raw.model, checkpoint)
        _reset_a2_to_a5_associative(runtime_raw.model)
        if backbone.ttt_config.a5_phase == "main":
            warmup_bundle_audit = _load_warmup_bundle(
                model=runtime_raw.model,
                qwen_model=backbone.model,
                backbone=backbone,
            )
        _reset_a2_to_a5_balance(runtime_raw.model)
    if (
        configured_stage is ProductionStage.A5
        and backbone.ttt_config.a5_phase == "fast_state_warmup"
    ):
        warmup_trainability_audit = _configure_fast_state_warmup_trainability(
            runtime_raw.model,
            backbone.model,
        )
    expected_gradient_groups = (
        (
            "qwen",
            "state_shared",
            "state_task",
            "state_router_time",
            "state_retrieval",
            "w0",
        )
        if configured_stage is ProductionStage.A2
        else (
            *(("qwen", "fast_slow") if backbone.ttt_config.a5_phase == "main" else ()),
            "state_shared",
            "state_task",
            "state_router_time",
            "state_retrieval",
            *(("w0",) if backbone.ttt_config.a5_phase == "main" else ()),
            *(("associative",) if backbone.ttt_config.a5_adaptation_mode == "meta_ttt" else ()),
        )
    )
    warmup_qwen_auditor = (
        _WarmupQwenBitwiseAuditor(backbone.model)
        if backbone.ttt_config.a5_phase == "fast_state_warmup"
        else None
    )
    runtime_raw = replace(
        runtime_raw,
        optimizer_factory=make_production_outer_optimizer_factory(
            backbone,
            configured_stage,
            a5_adaptation_mode=backbone.ttt_config.a5_adaptation_mode,
            a5_phase=backbone.ttt_config.a5_phase,
        ),
        gradient_controller=OuterGradientController(
            backbone.project_config.outer_gradient_control,
            expected_groups=expected_gradient_groups,
            probes=_memory_gradient_probes(runtime_raw.model, expected_gradient_groups),
        ),
        semantic_projector_delta_audit_steps=(
            backbone.ttt_config.semantic_projector_delta_audit_steps
        ),
        a5_parameter_delta_audit_steps=(backbone.ttt_config.a5_parameter_delta_audit_steps),
        operator_diagnostics_interval=backbone.ttt_config.operator_diagnostics_interval,
        warmup_qwen_bitwise_auditor=warmup_qwen_auditor,
        train_sampler_factory=(
            lambda dataset, rank, world_size: build_production_train_sampler(
                dataset,
                rank,
                world_size,
                visual_cost_index=visual_cost_index,
                query_sample_fps=backbone.ttt_config.query_sample_fps,
                state_query_visual_mode=backbone.ttt_config.state_query_visual_mode,
                state_query_max_frames=backbone.ttt_config.state_query_max_frames,
                answer_query_visual_mode=backbone.ttt_config.answer_query_visual_mode,
                answer_query_max_frames=backbone.ttt_config.answer_query_max_frames,
            )
        ),
    )
    parameter_audit = _audit_outer_parameters(
        backbone,
        runtime_raw,
        a5_phase=backbone.ttt_config.a5_phase,
    )
    audit_outer_checkpoint_boundary(runtime_raw.model)
    training_args = cast(Any, backbone.training_args)
    project = backbone.project_config
    if configured_stage is ProductionStage.A2:
        budget_lrs = (
            float(project.a2.optimizer.qwen_learning_rate),
            float(project.a2.optimizer.state_learning_rate),
            float(project.a2.optimizer.state_learning_rate),
            float(project.a2.optimizer.w0_learning_rate),
            float(project.a2.optimizer.state_learning_rate),
        )
    else:
        phase_optimizer = (
            project.a5.warmup
            if backbone.ttt_config.a5_phase == "fast_state_warmup"
            else project.a5.optimizer
        )
        budget_lrs = (
            (
                0.0
                if backbone.ttt_config.a5_phase == "fast_state_warmup"
                else float(training_args.learning_rate)
            ),
            float(phase_optimizer.fast_slow_learning_rate),
            float(phase_optimizer.state_learning_rate),
            float(phase_optimizer.w0_learning_rate),
            float(phase_optimizer.associative_learning_rate),
        )
    budget_audit = _outer_update_norm_budget_audit(
        project,
        configured_stage,
        qwen_lr=budget_lrs[0],
        fast_slow_lr=budget_lrs[1],
        state_lr=budget_lrs[2],
        w0_lr=budget_lrs[3],
        associative_lr=budget_lrs[4],
        a5_adaptation_mode=runtime_raw.a5_adaptation_mode,
        a5_phase=backbone.ttt_config.a5_phase,
    )
    raw_smoke_steps = os.environ.get("TTT_SMOKE_MAX_STEPS")
    smoke_max_steps: int | None = None
    if raw_smoke_steps is not None:
        try:
            smoke_max_steps = int(raw_smoke_steps)
        except ValueError as error:
            raise ValueError("TTT_SMOKE_MAX_STEPS must be a positive integer") from error
        if smoke_max_steps <= 0:
            raise ValueError("TTT_SMOKE_MAX_STEPS must be a positive integer")
        training_args.max_steps = smoke_max_steps
    raw_skip_final = os.environ.get("TTT_SKIP_FINAL_CHECKPOINT", "0")
    if raw_skip_final not in {"0", "1"}:
        raise ValueError("TTT_SKIP_FINAL_CHECKPOINT must be 0 or 1")
    skip_final_checkpoint = raw_skip_final == "1"
    if skip_final_checkpoint and smoke_max_steps is None:
        raise ValueError("final checkpoint may be skipped only for an explicit max-step smoke")
    checkpoint_policy = _checkpoint_policy_from_environment()
    if skip_final_checkpoint and checkpoint_policy is not CheckpointPolicy.ATOMIC_FINAL_ONLY:
        raise ValueError("a smoke run cannot retain epoch checkpoints")
    if checkpoint_policy is CheckpointPolicy.ATOMIC_FINAL_ONLY:
        _disable_smoke_checkpoints(training_args)
    if checkpoint_policy is CheckpointPolicy.EPOCH_2_AND_EPOCH_4:
        _validate_epoch_two_four_training_arguments(training_args)
    if backbone.ttt_config.a5_phase == "fast_state_warmup" and smoke_max_steps is None:
        _validate_fast_state_warmup_training_arguments(training_args, project)
    trainer = cast(Any, build_production_trainer(backbone, runtime_raw))
    output_dir = Path(str(training_args.output_dir))
    artifact_root = Path(os.environ.get("RUN_ROOT", str(output_dir)))
    sequence_audit: tuple[str, int] | None = None
    sequence_world_size: int | None = None
    qwen_source_sha256: str | None = None
    if backbone.ttt_config.a5_phase == "fast_state_warmup":
        # This pre-prepare digest is provenance only.  The actual frozen-Qwen baseline
        # is captured by the Trainer on every rank after DeepSpeed preparation and
        # before the first optimizer update.
        qwen_source_sha256 = _module_bitwise_sha256(backbone.model)
    if configured_stage is ProductionStage.A5:
        # The sampler synchronizes its runtime-cost EMA when advancing epochs. Every rank
        # must therefore execute this preflight in the same collective order; running it
        # only on world rank zero races DeepSpeed's process-group construction on peers.
        sequence_world_size = int(os.environ.get("WORLD_SIZE", "4"))
        sequence_audit = _a5_global_sample_sequence_sha256(
            train_dataset,
            runtime_raw.train_sampler_factory,
            epoch_count=float(training_args.num_train_epochs),
            world_size=sequence_world_size,
        )
    if trainer.is_world_process_zero():
        environment = environment_manifest(backbone)
        if configured_stage is ProductionStage.A5:
            if sequence_audit is None:
                raise RuntimeError("A5 sample-sequence audit was not computed")
            sequence_sha256, sequence_count = sequence_audit
            environment["a5_global_sample_sequence_sha256"] = sequence_sha256
            environment["a5_global_sample_sequence_count"] = sequence_count
            environment["a5_global_sample_sequence_world_size"] = sequence_world_size
        environment["qwen_trainability_audit"] = asdict(trainability_audit)
        environment["a5_phase"] = backbone.ttt_config.a5_phase
        environment["warmup_trainability_audit"] = (
            asdict(warmup_trainability_audit)
            if warmup_trainability_audit is not None
            else None
        )
        if full_unfreeze_audit is not None:
            environment["full_unfreeze_audit"] = asdict(full_unfreeze_audit)
        environment["outer_parameter_audit"] = asdict(parameter_audit)
        environment["memory_eta_max_per_slot"] = float(project.fast_memory.eta_max_per_slot)
        environment["memory_eta_chunk_budget"] = float(project.fast_memory.eta_chunk_budget)
        environment["memory_forget_beta_max"] = float(project.fast_memory.forget_beta_max)
        environment["outer_update_norm_budget_audit"] = budget_audit
        checkpoint_environment = None
        if checkpoint_audit is not None:
            checkpoint_environment = asdict(checkpoint_audit)
            checkpoint_environment["checkpoint"] = str(checkpoint_audit.checkpoint)
        environment["a2_initialization_audit"] = checkpoint_environment
        environment["warmup_bundle_audit"] = warmup_bundle_audit
        environment["warmup_qwen_source_sha256"] = qwen_source_sha256
        environment["warmup_qwen_bitwise_baseline_stage"] = (
            _WarmupQwenBitwiseAuditor.baseline_stage if warmup_qwen_auditor is not None else None
        )
        _write_json(artifact_root / "environment.json", environment)
    if configured_stage is ProductionStage.A5:
        trainer.accelerator.wait_for_everyone()
    try:
        result = trainer.train(
            resume_from_checkpoint=None if same_stage_resume is None else str(same_stage_resume)
        )
    finally:
        flush_runtime_metrics(resolve_cuda=True)
    if skip_final_checkpoint:
        qwen_bitwise_audit: WarmupQwenBitwiseAudit | None = None
        if warmup_qwen_auditor is not None:
            qwen_bitwise_audit = warmup_qwen_auditor.finalize(device=trainer.accelerator.device)
            _write_warmup_qwen_bitwise_audit(
                artifact_root=artifact_root,
                audit=qwen_bitwise_audit,
            )
        trainer.accelerator.wait_for_everyone()
        if qwen_bitwise_audit is not None:
            _assert_warmup_qwen_bitwise_unchanged(qwen_bitwise_audit)
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        if trainer.is_world_process_zero():
            _write_json(
                artifact_root / "run_summary.json",
                {
                    "status": "smoke_completed",
                    "stage": runtime_raw.stage.value,
                    "a5_phase": backbone.ttt_config.a5_phase,
                    "a5_adaptation_mode": runtime_raw.a5_adaptation_mode,
                    "outer_update_norm_budget_audit": budget_audit,
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics": result.metrics,
                    "checkpoint_policy": "none_for_smoke",
                    "qwen_bitwise_audit": (
                        None
                        if qwen_bitwise_audit is None
                        else {
                            "baseline_stage": qwen_bitwise_audit.baseline_stage,
                            "baseline_global_step": (qwen_bitwise_audit.baseline_global_step),
                            "baseline_sha256": qwen_bitwise_audit.baseline_sha256,
                            "final_sha256": qwen_bitwise_audit.final_sha256,
                            "tensor_count": qwen_bitwise_audit.tensor_count,
                            "changed_tensor_count": (qwen_bitwise_audit.changed_tensor_count),
                            "all_ranks_unchanged": (qwen_bitwise_audit.all_ranks_unchanged),
                            "path": str(artifact_root / "qwen_bitwise_audit.json"),
                        }
                    ),
                    "final_checkpoint": None,
                    "resume_state": None,
                    "resumed_from": None,
                },
            )
        return 0
    if backbone.ttt_config.a5_phase == "fast_state_warmup":
        trainer.accelerator.wait_for_everyone()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        if qwen_source_sha256 is None or warmup_qwen_auditor is None:
            raise RuntimeError("warmup lost its Qwen provenance or bitwise auditor")
        qwen_bitwise_audit, prepared_bundle = _prepare_distributed_warmup_handoff(
            model=runtime_raw.model,
            qwen_model=backbone.model,
            qwen_auditor=warmup_qwen_auditor,
            device=trainer.accelerator.device,
        )
        _write_warmup_qwen_bitwise_audit(
            artifact_root=artifact_root,
            audit=qwen_bitwise_audit,
        )
        # No rank may enter the publishing barrier while another rank is still copying
        # tensors from CUDA.  From this point onward rank zero performs CPU/filesystem work.
        trainer.accelerator.wait_for_everyone()
        _assert_warmup_qwen_bitwise_unchanged(qwen_bitwise_audit)
        if prepared_bundle is None:
            raise RuntimeError("unchanged warmup Qwen audit produced no handoff tensors")
        if trainer.is_world_process_zero():
            bundle_path, bundle_manifest = _publish_warmup_bundle(
                model=runtime_raw.model,
                qwen_model=backbone.model,
                backbone=backbone,
                artifact_root=artifact_root,
                global_step=int(trainer.state.global_step),
                qwen_sha256=qwen_source_sha256,
                prepared_bundle=prepared_bundle,
                qwen_warmup_audit=qwen_bitwise_audit,
            )
            _write_json(
                artifact_root / "run_summary.json",
                {
                    "status": "completed",
                    "stage": runtime_raw.stage.value,
                    "a5_phase": "fast_state_warmup",
                    "a5_adaptation_mode": runtime_raw.a5_adaptation_mode,
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics": result.metrics,
                    "checkpoint_policy": "warmup_bundle_only",
                    "warmup_bundle": str(bundle_path),
                    "warmup_bundle_manifest": bundle_manifest,
                    "qwen_bitwise_unchanged": True,
                    "qwen_bitwise_audit": {
                        "baseline_stage": qwen_bitwise_audit.baseline_stage,
                        "baseline_global_step": qwen_bitwise_audit.baseline_global_step,
                        "baseline_sha256": qwen_bitwise_audit.baseline_sha256,
                        "final_sha256": qwen_bitwise_audit.final_sha256,
                        "tensor_count": qwen_bitwise_audit.tensor_count,
                        "changed_tensor_count": qwen_bitwise_audit.changed_tensor_count,
                        "all_ranks_unchanged": (qwen_bitwise_audit.all_ranks_unchanged),
                        "path": str(artifact_root / "qwen_bitwise_audit.json"),
                    },
                    "final_checkpoint": None,
                    "resume_state": None,
                    "resumed_from": None,
                },
            )
        trainer.accelerator.wait_for_everyone()
        return 0
    if checkpoint_policy is CheckpointPolicy.EPOCH_2_AND_EPOCH_4:
        trainer.accelerator.wait_for_everyone()
        epoch_checkpoints: dict[int, Path] = {}
        if trainer.is_world_process_zero():
            epoch_checkpoints = _publish_epoch_two_four_checkpoints(output_dir)
        trainer.accelerator.wait_for_everyone()
        trainer.save_state()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        if trainer.is_world_process_zero():
            epoch_two_checkpoint = epoch_checkpoints[2]
            epoch_four_checkpoint = epoch_checkpoints[4]
            _write_json(
                artifact_root / "run_summary.json",
                {
                    "status": "completed",
                    "stage": runtime_raw.stage.value,
                    "a5_adaptation_mode": runtime_raw.a5_adaptation_mode,
                    "outer_update_norm_budget_audit": budget_audit,
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics": result.metrics,
                    "checkpoint_policy": checkpoint_policy.value,
                    "epoch_checkpoints": {
                        "2": str(epoch_two_checkpoint),
                        "4": str(epoch_four_checkpoint),
                    },
                    "final_checkpoint": str(epoch_four_checkpoint),
                    "resume_state": str(epoch_four_checkpoint),
                    "resumed_from": (None if same_stage_resume is None else str(same_stage_resume)),
                },
            )
        return 0
    final_checkpoint = output_dir / "final-checkpoint"
    incomplete_checkpoint = output_dir / ".final-checkpoint.incomplete"
    if trainer.is_world_process_zero() and (
        final_checkpoint.exists() or incomplete_checkpoint.exists()
    ):
        raise FileExistsError("refusing to overwrite an existing final checkpoint")
    trainer.accelerator.wait_for_everyone()
    audit_outer_checkpoint_boundary(runtime_raw.model)
    trainer.save_model(str(incomplete_checkpoint))
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.save_state(str(incomplete_checkpoint / "resume_state"))
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(incomplete_checkpoint / "trainer_state.json"))
        _validate_checkpoint_tree(incomplete_checkpoint)
        incomplete_checkpoint.rename(final_checkpoint)
        for child in output_dir.glob("checkpoint-*"):
            if child.is_dir():
                shutil.rmtree(child)
    trainer.accelerator.wait_for_everyone()
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    if trainer.is_world_process_zero():
        _write_json(
            artifact_root / "run_summary.json",
            {
                "status": "completed",
                "stage": runtime_raw.stage.value,
                "a5_adaptation_mode": runtime_raw.a5_adaptation_mode,
                "outer_update_norm_budget_audit": budget_audit,
                "global_step": int(trainer.state.global_step),
                "elapsed_seconds": time.monotonic() - started,
                "metrics": result.metrics,
                "checkpoint_policy": checkpoint_policy.value,
                "final_checkpoint": str(final_checkpoint),
                "resume_state": str(final_checkpoint / "resume_state"),
                "resumed_from": (None if same_stage_resume is None else str(same_stage_resume)),
            },
        )
    return 0


def _checkpoint_policy_from_environment() -> CheckpointPolicy:
    raw = os.environ.get("TTT_CHECKPOINT_POLICY", CheckpointPolicy.ATOMIC_FINAL_ONLY.value)
    try:
        return CheckpointPolicy(raw)
    except ValueError as error:
        choices = ", ".join(policy.value for policy in CheckpointPolicy)
        raise ValueError(f"TTT_CHECKPOINT_POLICY must be one of: {choices}") from error


def _disable_smoke_checkpoints(training_args: object) -> None:
    """Disable periodic saves when the atomic policy publishes only the final checkpoint."""

    arguments = cast(Any, training_args)
    strategy = arguments.save_strategy
    strategy_type = type(strategy)
    arguments.save_strategy = "no" if strategy_type is str else strategy_type("no")
    arguments.save_steps = 0


def _validate_epoch_two_four_training_arguments(training_args: object) -> None:
    arguments = cast(Any, training_args)
    epochs = float(arguments.num_train_epochs)
    strategy_raw = arguments.save_strategy
    strategy = getattr(strategy_raw, "value", str(strategy_raw))
    save_steps = float(arguments.save_steps)
    save_total_limit = int(arguments.save_total_limit)
    if not math.isclose(epochs, 4.0):
        raise ValueError("epoch_2_and_epoch_4 checkpoint policy requires num_train_epochs=4")
    if strategy != "steps" or not math.isclose(save_steps, 0.5):
        raise ValueError(
            "epoch_2_and_epoch_4 checkpoint policy requires save_strategy=steps and save_steps=0.5"
        )
    if save_total_limit < 2:
        raise ValueError("epoch_2_and_epoch_4 checkpoint policy requires save_total_limit>=2")


def _validate_fast_state_warmup_training_arguments(
    training_args: object,
    project: ProjectConfig,
) -> None:
    """Fail closed unless the independent warmup scheduler is exactly reproducible."""

    arguments = cast(Any, training_args)
    scheduler_raw = arguments.lr_scheduler_type
    scheduler = getattr(scheduler_raw, "value", str(scheduler_raw))
    if int(arguments.max_steps) != project.a5.warmup.max_steps:
        raise ValueError("Memory/State warmup requires exactly 256 optimizer steps")
    if int(arguments.warmup_steps) != project.a5.warmup.linear_warmup_steps:
        raise ValueError("Memory/State warmup requires exactly four linear warmup steps")
    if scheduler != "cosine":
        raise ValueError("Memory/State warmup requires the cosine scheduler")
    if int(arguments.gradient_accumulation_steps) != 1:
        raise ValueError("Memory/State warmup requires one optimizer step per episode batch")


def _standard_checkpoint_progress(checkpoint: Path) -> tuple[int, int, float]:
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.is_file():
        raise RuntimeError(f"standard checkpoint is missing trainer_state.json: {checkpoint}")
    raw = cast(object, json.loads(trainer_state.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise RuntimeError(f"standard checkpoint has invalid trainer_state.json: {checkpoint}")
    try:
        global_step = int(raw["global_step"])
        max_steps = int(raw["max_steps"])
        epoch = float(raw["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"standard checkpoint has invalid progress metadata: {checkpoint}"
        ) from error
    if global_step <= 0 or max_steps <= 0 or global_step > max_steps or not math.isfinite(epoch):
        raise RuntimeError(f"standard checkpoint has impossible progress metadata: {checkpoint}")
    return global_step, max_steps, epoch


def _validate_standard_resume_checkpoint(checkpoint: Path) -> None:
    model_candidates = (
        checkpoint / "model.safetensors",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "pytorch_model.bin",
        checkpoint / "pytorch_model.bin.index.json",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in model_candidates):
        raise RuntimeError(f"standard checkpoint has no model weights: {checkpoint}")
    if not (checkpoint / "scheduler.pt").is_file():
        raise RuntimeError(f"standard checkpoint is missing scheduler.pt: {checkpoint}")
    optimizer_state_present = (checkpoint / "optimizer.pt").is_file() or any(
        child.is_dir() and child.name.startswith("global_step") for child in checkpoint.iterdir()
    )
    if not optimizer_state_present:
        raise RuntimeError(f"standard checkpoint has no optimizer state: {checkpoint}")


def _publish_epoch_two_four_checkpoints(output_dir: Path) -> dict[int, Path]:
    """Publish exactly two resumable checkpoints at the 2/4-epoch boundaries."""

    candidates = tuple(sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir()))
    if len(candidates) != 2:
        raise RuntimeError(
            "epoch_2_and_epoch_4 checkpoint policy expected exactly two scheduled checkpoints, "
            f"found {len(candidates)}"
        )
    progress = {path: _standard_checkpoint_progress(path) for path in candidates}
    max_steps_values = {item[1] for item in progress.values()}
    if len(max_steps_values) != 1:
        raise RuntimeError("scheduled checkpoints disagree on max_steps")
    max_steps = next(iter(max_steps_values))
    target_steps = {2: math.ceil(max_steps * 0.5), 4: max_steps}
    selected: dict[int, Path] = {}
    for epoch_number, target_step in target_steps.items():
        matches = [path for path, item in progress.items() if item[0] == target_step]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one checkpoint at epoch {epoch_number} step {target_step}, "
                f"found {len(matches)}"
            )
        source = matches[0]
        observed_epoch = progress[source][2]
        if not math.isclose(observed_epoch, float(epoch_number), abs_tol=0.01):
            raise RuntimeError(
                f"checkpoint {source} reports epoch={observed_epoch}, expected {epoch_number}"
            )
        _validate_standard_resume_checkpoint(source)
        selected[epoch_number] = source

    destinations = {
        epoch_number: output_dir / f"epoch-{epoch_number}-checkpoint" for epoch_number in (2, 4)
    }
    for destination in destinations.values():
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    published: dict[int, Path] = {}
    for epoch_number, destination in destinations.items():
        source = selected[epoch_number]
        source.rename(destination)
        published[epoch_number] = destination
    return published


def _validate_checkpoint_tree(checkpoint: Path) -> None:
    """Validate model and resume artifacts before publishing and deleting the prior epoch."""

    if not checkpoint.is_dir():
        raise FileNotFoundError("incomplete checkpoint directory was not created")
    model_candidates = (
        checkpoint / "model.safetensors",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "pytorch_model.bin",
        checkpoint / "pytorch_model.bin.index.json",
    )
    present = tuple(path for path in model_candidates if path.is_file() and path.stat().st_size > 0)
    if len(present) != 1:
        raise RuntimeError("final checkpoint must contain exactly one model weight entrypoint")
    entrypoint = present[0]
    if entrypoint.name.endswith(".index.json"):
        raw = cast(object, json.loads(entrypoint.read_text(encoding="utf-8")))
        weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("final checkpoint shard index has no weight_map")
        shard_names = {value for value in weight_map.values() if isinstance(value, str)}
        if len(shard_names) != len(set(weight_map.values())):
            raise RuntimeError("final checkpoint shard index contains invalid shard names")
        if any(
            not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size <= 0
            for name in shard_names
        ):
            raise RuntimeError("final checkpoint shard index references a missing/empty shard")
    trainer_state = checkpoint / "trainer_state.json"
    resume_state = checkpoint / "resume_state"
    if not trainer_state.is_file() or trainer_state.stat().st_size <= 0:
        raise RuntimeError("final checkpoint is missing trainer_state.json")
    if not resume_state.is_dir() or not any(resume_state.iterdir()):
        raise RuntimeError("final checkpoint is missing complete Accelerate resume state")


def resolve_same_stage_resume(
    checkpoint: str | None,
    stage: ProductionStage,
    *,
    a5_adaptation_mode: str = "meta_ttt",
) -> Path | None:
    """Validate a standard Trainer checkpoint without conflating it with A2→A5 init."""

    if checkpoint is None or not checkpoint.strip():
        return None
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir() or not (root / "trainer_state.json").is_file():
        raise FileNotFoundError(
            "TTT_RESUME_CHECKPOINT must be a standard checkpoint directory containing "
            "trainer_state.json"
        )
    optimizer_state_present = (root / "optimizer.pt").is_file() or any(
        child.is_dir() and child.name.startswith("global_step") for child in root.iterdir()
    )
    if not (root / "scheduler.pt").is_file() or not optimizer_state_present:
        raise FileNotFoundError(
            "same-stage resume requires a standard Trainer/DeepSpeed optimizer and scheduler "
            "checkpoint; final-checkpoint/resume_state is archival, not a Trainer resume path"
        )
    run_config: Path | None = None
    for parent in (root, *root.parents[:4]):
        candidate = parent / "run_config.json"
        if candidate.is_file():
            run_config = candidate
            break
    if run_config is None:
        raise FileNotFoundError("same-stage resume requires an ancestor run_config.json")
    raw = cast(object, json.loads(run_config.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or raw.get("stage") != stage.value:
        raise ValueError("resume checkpoint stage does not match the configured production stage")
    if (
        raw.get("config_schema_version") != 14
        or raw.get("associative_ttt_contract") != "bank_conditioned_slot_memory_v3"
    ):
        raise ValueError(
            "same-stage resume requires the schema-14 slot-memory contract"
        )
    if stage is ProductionStage.A5:
        checkpoint_mode = raw.get("a5_adaptation_mode", "meta_ttt")
        if checkpoint_mode != a5_adaptation_mode:
            raise ValueError(
                "resume checkpoint A5 adaptation mode does not match the configured mode"
            )
    return root


def _validate_resume_balance_schema(checkpoint: Path) -> None:
    expected = {
        "official_weak_balancer.ema_values": (torch.float64, (5,)),
        "official_weak_balancer.ema_valid": (torch.bool, (5,)),
        "official_weak_balancer.ema_update_counts": (torch.int64, (5,)),
        "official_weak_balancer.gradient_ema_values": (torch.float64, (4,)),
        "official_weak_balancer.gradient_ema_valid": (torch.bool, (4,)),
        "official_weak_balancer.gradient_ema_update_counts": (torch.int64, (4,)),
        "official_weak_balancer.balance_schema_version": (torch.int64, ()),
    }
    single = checkpoint / "model.safetensors"
    index = checkpoint / "model.safetensors.index.json"
    if single.is_file():
        sources = {key: single for key in expected}
    elif index.is_file():
        raw = cast(object, json.loads(index.read_text(encoding="utf-8")))
        weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError("balance checkpoint index has no weight_map")
        sources = {}
        for key in expected:
            shard = weight_map.get(key)
            if not isinstance(shard, str):
                raise ValueError(f"balance checkpoint is missing required tensor: {key}")
            sources[key] = checkpoint / shard
    else:
        raise ValueError("formal balance checkpoint requires safetensors weights")
    tensors: dict[str, Tensor] = {}
    for source in set(sources.values()):
        if not source.is_file():
            raise FileNotFoundError(f"balance checkpoint shard is missing: {source}")
        keys = tuple(key for key, path in sources.items() if path == source)
        with safe_open(source, framework="pt", device="cpu") as reader:
            available = set(reader.keys())
            for key in keys:
                if key not in available:
                    raise ValueError(f"balance checkpoint is missing required tensor: {key}")
                tensors[key] = reader.get_tensor(key)
    for key, (dtype, shape) in expected.items():
        value = tensors[key]
        if value.dtype != dtype or tuple(value.shape) != shape:
            raise ValueError(
                f"balance checkpoint tensor {key} must be {dtype} {shape}; "
                f"found {value.dtype} {tuple(value.shape)}"
            )
    schema = tensors["official_weak_balancer.balance_schema_version"]
    if int(schema.item()) != 7:
        raise ValueError(
            "balance checkpoint has incompatible schema; formal training requires schema 7"
        )


def _audit_outer_parameters(
    backbone: LlamaFactoryBackboneBundle,
    runtime: ProductionTrainerRuntime,
    *,
    a5_phase: str = "main",
) -> OuterParameterAudit:
    named = tuple(runtime.model.named_parameters())
    associative = tuple(
        (name, parameter) for name, parameter in named if _is_associative_parameter_name(name)
    )
    transient = tuple(
        name for name, _ in named if _is_transient_memory_name(name.casefold())
    )
    backbone_ids = {id(parameter) for parameter in backbone.model.parameters()}
    runtime_ids = {id(parameter) for _, parameter in named}
    qwen = tuple((name, parameter) for name, parameter in named if id(parameter) in backbone_ids)
    non_qwen = tuple(
        (name, parameter) for name, parameter in named if id(parameter) not in backbone_ids
    )
    return OuterParameterAudit(
        stage=runtime.stage,
        a5_adaptation_mode=runtime.a5_adaptation_mode,
        total_parameter_count=sum(parameter.numel() for _, parameter in named),
        trainable_parameter_count=sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        qwen_parameter_count=sum(parameter.numel() for _, parameter in qwen),
        qwen_trainable_count=sum(
            parameter.numel() for _, parameter in qwen if parameter.requires_grad
        ),
        non_qwen_parameter_count=sum(parameter.numel() for _, parameter in non_qwen),
        non_qwen_trainable_count=sum(
            parameter.numel() for _, parameter in non_qwen if parameter.requires_grad
        ),
        associative_parameter_count=sum(parameter.numel() for _, parameter in associative),
        associative_trainable_count=sum(
            parameter.numel() for _, parameter in associative if parameter.requires_grad
        ),
        transient_parameter_names=transient,
        backbone_registered=bool(backbone_ids) and backbone_ids <= runtime_ids,
        a5_phase=a5_phase,
    )


def _outer_update_norm_budget_audit(
    project: ProjectConfig,
    stage: ProductionStage,
    *,
    qwen_lr: float,
    fast_slow_lr: float,
    state_lr: float,
    w0_lr: float,
    associative_lr: float,
    a5_adaptation_mode: str,
    a5_phase: str = "main",
) -> dict[str, object]:
    if a5_adaptation_mode not in _A5_ADAPTATION_MODES:
        raise ValueError("A5 adaptation mode must be meta_ttt or no_write")
    caps = project.outer_gradient_control.max_grad_norm
    if stage is ProductionStage.A5 and a5_phase == "fast_state_warmup":
        reference_budget = associative_lr * float(caps.associative)
    elif stage is ProductionStage.A5:
        reference_budget = fast_slow_lr * float(caps.fast_slow)
    else:
        reference_budget = qwen_lr * float(caps.qwen)
    if stage is ProductionStage.A5 and a5_phase == "fast_state_warmup":
        independent_budgets = (
            {"associative": associative_lr * float(caps.associative)}
            if a5_adaptation_mode == "meta_ttt"
            else {}
        )
    else:
        independent_budgets = {
            "w0": w0_lr * float(caps.w0),
            **(
                {"fast_slow": fast_slow_lr * float(caps.fast_slow)}
                if stage is ProductionStage.A5
                else {}
            ),
            **(
                {"associative": associative_lr * float(caps.associative)}
                if stage is ProductionStage.A5 and a5_adaptation_mode == "meta_ttt"
                else {}
            ),
        }
    mode = project.outer_gradient_control.mode
    if mode is not OuterGradientControlMode.PER_GROUP_L2_EQUAL_UPDATE_CAP:
        raise ValueError("production training requires the canonical equal-update budget policy")
    expected_budgets = {name: reference_budget for name in independent_budgets}
    if any(
        name not in expected_budgets
        or not math.isclose(value, expected_budgets[name], rel_tol=1.0e-6)
        for name, value in independent_budgets.items()
    ):
        raise ValueError("active optimizer update-norm budgets must remain aligned")
    state_names = (
        "state_shared",
        "state_task",
        "state_router_time",
        "state_retrieval",
    )
    state_rss_budget = math.sqrt(
        sum((state_lr * float(getattr(caps, name))) ** 2 for name in state_names)
    )
    if a5_phase != "fast_state_warmup" and not math.isclose(
        state_rss_budget, reference_budget, rel_tol=1.0e-6
    ):
        raise ValueError("state subgroup RSS update-norm budget drifted from the formal cap")
    qwen_budget = qwen_lr * float(caps.qwen)
    if (
        stage is ProductionStage.A5
        and a5_phase == "main"
        and not math.isclose(qwen_budget, reference_budget, rel_tol=1.0e-6)
    ):
        raise ValueError("A5 main Qwen update-norm budget drifted from Fast/State reference")
    return {
        "policy": mode.value,
        "memory_eta_max_per_slot": float(project.fast_memory.eta_max_per_slot),
        "memory_eta_chunk_budget": float(project.fast_memory.eta_chunk_budget),
        "memory_forget_beta_max": float(project.fast_memory.forget_beta_max),
        "reference": reference_budget,
        "independent": independent_budgets,
        "state_rss": state_rss_budget,
        "qwen": qwen_budget,
        "a5_phase": a5_phase,
    }


def make_production_outer_optimizer_factory(
    backbone: LlamaFactoryBackboneBundle,
    stage: ProductionStage,
    *,
    a5_adaptation_mode: str = "meta_ttt",
    a5_phase: str = "main",
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    if a5_adaptation_mode not in _A5_ADAPTATION_MODES:
        raise ValueError("A5 adaptation mode must be meta_ttt or no_write")
    if stage is ProductionStage.A2 and a5_adaptation_mode != "meta_ttt":
        raise ValueError("A2 cannot select an A5 adaptation mode")
    if a5_phase not in {"fast_state_warmup", "main"}:
        raise ValueError("A5 phase must be fast_state_warmup or main")
    if stage is ProductionStage.A2 and a5_phase != "main":
        raise ValueError("A2 cannot select an A5 phase")
    qwen_ids = {id(parameter) for parameter in backbone.model.parameters()}
    training_args = cast(Any, backbone.training_args)
    if stage is ProductionStage.A2:
        qwen_lr = backbone.project_config.a2.optimizer.qwen_learning_rate
        state_lr = backbone.project_config.a2.optimizer.state_learning_rate
        w0_lr = backbone.project_config.a2.optimizer.w0_learning_rate
        associative_lr = state_lr
        fast_slow_lr = state_lr
    else:
        qwen_lr = (
            0.0 if a5_phase == "fast_state_warmup" else float(training_args.learning_rate)
        )
        if a5_phase == "fast_state_warmup":
            optimizer = backbone.project_config.a5.warmup
        else:
            optimizer = backbone.project_config.a5.optimizer
        fast_slow_lr = optimizer.fast_slow_learning_rate
        state_lr = optimizer.state_learning_rate
        w0_lr = optimizer.w0_learning_rate
        associative_lr = optimizer.associative_learning_rate

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        groups: dict[str, list[nn.Parameter]] = {
            "qwen": [],
            "fast_slow": [],
            "state_shared": [],
            "state_task": [],
            "state_router_time": [],
            "state_retrieval": [],
            "w0": [],
            "associative": [],
        }
        adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
        if len(adapters) != 1:
            raise RuntimeError("Outer optimizer requires exactly one FastTTTAdapter")
        fast_slow_ids = {id(parameter) for parameter in adapters[0].collect_slow_parameters()}
        ownership: dict[int, str] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            lowered = name.casefold()
            if _is_transient_memory_name(lowered):
                raise ValueError("the transient per-video memory cannot enter the Outer optimizer")
            if parameter_id in qwen_ids:
                group = "qwen"
            elif stage is ProductionStage.A5 and parameter_id in fast_slow_ids:
                group = "fast_slow"
            else:
                group = _state_group_for_name(lowered)
            previous = ownership.get(parameter_id)
            if previous is not None:
                if previous != group:
                    raise ValueError(
                        f"aliased Outer parameter crossed optimizer groups: {previous}/{group}"
                    )
                continue
            ownership[parameter_id] = group
            groups[group].append(parameter)
        required = (
            (
                "qwen",
                "state_shared",
                "state_task",
                "state_router_time",
                "state_retrieval",
                "w0",
            )
            if stage is ProductionStage.A2
            else (
                (
                    "state_shared",
                    "state_task",
                    "state_router_time",
                    "state_retrieval",
                )
                if a5_phase == "fast_state_warmup"
                else (
                    "qwen",
                    "fast_slow",
                    "state_shared",
                    "state_task",
                    "state_router_time",
                    "state_retrieval",
                    "w0",
                )
            )
        )
        empty = tuple(name for name in required if not groups[name])
        if empty:
            raise ValueError(f"Outer AdamW requires non-empty formal groups: {empty}")
        if stage is ProductionStage.A2 and groups["associative"]:
            raise ValueError("A2 Outer AdamW cannot own Associative")
        if stage is ProductionStage.A5:
            if a5_phase == "fast_state_warmup":
                frozen_groups = tuple(
                    name for name in ("qwen", "fast_slow", "w0") if groups[name]
                )
                if frozen_groups:
                    raise ValueError(
                        "Memory/State warmup cannot own frozen optimizer groups: "
                        f"{frozen_groups}"
                    )
            if a5_adaptation_mode == "meta_ttt" and not groups["associative"]:
                raise ValueError("Meta-TTT A5 Outer AdamW must own Associative")
            if a5_adaptation_mode == "no_write" and groups["associative"]:
                raise ValueError("no-write A5 Outer AdamW cannot own the memory interface")
        semantic_projector_ids = {
            id(parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if "semantic_projector" in name.casefold() and parameter.requires_grad
        }
        retrieval_group_ids = {id(parameter) for parameter in groups["state_retrieval"]}
        if retrieval_group_ids != semantic_projector_ids:
            raise ValueError(
                "state_retrieval optimizer group must exactly equal SemanticProjector "
                "before DeepSpeed wrapping"
            )
        trainable_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if set(ownership) != trainable_ids or sum(map(len, groups.values())) != len(trainable_ids):
            raise ValueError("every trainable Outer parameter must belong to exactly one group")
        learning_rates = {
            "qwen": qwen_lr,
            "fast_slow": fast_slow_lr,
            "state_shared": state_lr,
            "state_task": state_lr,
            "state_router_time": state_lr,
            "state_retrieval": state_lr,
            "w0": w0_lr,
            "associative": associative_lr,
        }
        parameter_groups: list[dict[str, Any]] = [
            {
                "params": values,
                "lr": learning_rates[name],
                "group_name": name,
            }
            for name, values in groups.items()
            if values
        ]
        budget_audit = _outer_update_norm_budget_audit(
            backbone.project_config,
            stage,
            qwen_lr=qwen_lr,
            fast_slow_lr=float(fast_slow_lr),
            state_lr=state_lr,
            w0_lr=w0_lr,
            associative_lr=associative_lr,
            a5_adaptation_mode=a5_adaptation_mode,
            a5_phase=a5_phase,
        )
        trace_event("outer_optimizer_update_norm_budgets", **budget_audit)
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(float(training_args.adam_beta1), float(training_args.adam_beta2)),
            eps=float(training_args.adam_epsilon),
            weight_decay=float(training_args.weight_decay),
        )
        active_trace: list[Any] = []

        def optimizer_start(*_args: object, **_kwargs: object) -> None:
            context = trace_cuda_phase("optimizer", stage=stage.value)
            context.__enter__()
            active_trace.append(context)

        def optimizer_end(*_args: object, **_kwargs: object) -> None:
            if not active_trace:
                raise RuntimeError("optimizer trace hook order drifted")
            active_trace.pop().__exit__(None, None, None)

        optimizer.register_step_pre_hook(optimizer_start)
        optimizer.register_step_post_hook(optimizer_end)
        return optimizer

    return factory


_MEMORY_INTERFACE_TOKENS = (
    "p_context",
    "memory_key_probe",
    "memory_value_projection",
    "memory_eta_gate_hidden",
    "memory_eta_gate_output",
    "memory_alpha",
    "memory_beta_raw",
)


def _is_associative_parameter_name(name: str) -> bool:
    lowered = name.casefold()
    return any(
        lowered.startswith(f"{token}.") or f".{token}." in lowered or lowered.endswith(f".{token}")
        or lowered == token
        for token in _MEMORY_INTERFACE_TOKENS
    )


def _state_group_for_name(lowered: str) -> str:
    if _is_associative_parameter_name(lowered):
        return "associative"
    if lowered.endswith(("w0_1", "w0_2")) or "meta_fast" in lowered:
        return "w0"
    if "component_modules.observation_heads" in lowered:
        return "state_task"
    if "operator_router" in lowered or "time_resolver" in lowered:
        return "state_router_time"
    if "semantic_projector" in lowered or "component_modules.retriever" in lowered:
        return "state_retrieval"
    return "state_shared"


@dataclass(frozen=True, slots=True)
class WarmupOuterTrainabilityAudit:
    qwen_parameter_count: int
    qwen_trainable_count: int
    fast_slow_parameter_count: int
    fast_slow_trainable_count: int
    w0_parameter_count: int
    w0_trainable_count: int
    state_parameter_count: int
    state_trainable_count: int
    associative_parameter_count: int
    associative_trainable_count: int

    def __post_init__(self) -> None:
        if any(
            count <= 0
            for count in (
                self.qwen_parameter_count,
                self.fast_slow_parameter_count,
                self.w0_parameter_count,
                self.state_parameter_count,
                self.associative_parameter_count,
            )
        ):
            raise ValueError("Memory/State warmup trainability audit found an empty group")
        if any(
            count != 0
            for count in (
                self.qwen_trainable_count,
                self.fast_slow_trainable_count,
                self.w0_trainable_count,
            )
        ):
            raise ValueError("Memory/State warmup left a frozen group trainable")
        if self.state_trainable_count != self.state_parameter_count:
            raise ValueError("Memory/State warmup must train every state parameter")
        if self.associative_trainable_count != self.associative_parameter_count:
            raise ValueError("Memory/State warmup must train every Associative parameter")


def _configure_fast_state_warmup_trainability(
    model: nn.Module,
    qwen_model: nn.Module,
) -> WarmupOuterTrainabilityAudit:
    """Freeze Qwen/W0/slow projections and train only memory interface plus state modules."""

    qwen_ids = {id(parameter) for parameter in qwen_model.parameters()}
    adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
    if len(adapters) != 1:
        raise RuntimeError("Memory/State warmup requires exactly one FastTTTAdapter")
    fast_slow_ids = {id(parameter) for parameter in adapters[0].collect_slow_parameters()}
    grouped: dict[int, tuple[str, nn.Parameter]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_id = id(parameter)
        lowered = name.casefold()
        if parameter_id in qwen_ids:
            group = "qwen"
        elif parameter_id in fast_slow_ids:
            group = "fast_slow"
        else:
            group = _state_group_for_name(lowered)
        previous = grouped.get(parameter_id)
        if previous is not None:
            if previous[0] != group:
                raise ValueError(
                    "aliased warmup parameter crossed trainability groups: "
                    f"{previous[0]}/{group}"
                )
            continue
        grouped[parameter_id] = (group, parameter)
    if not qwen_ids <= set(grouped):
        raise RuntimeError("Memory/State warmup lost Qwen parameters from the outer model")
    allowed_groups = _A5_WARMUP_TRAINABLE_GROUPS
    for group, parameter in grouped.values():
        parameter.requires_grad_(group in allowed_groups)

    def parameter_count(groups: frozenset[str]) -> tuple[int, int]:
        values = tuple(
            parameter
            for group, parameter in grouped.values()
            if group in groups
        )
        return (
            sum(parameter.numel() for parameter in values),
            sum(parameter.numel() for parameter in values if parameter.requires_grad),
        )

    qwen_count, qwen_trainable = parameter_count(frozenset({"qwen"}))
    fast_slow_count, fast_slow_trainable = parameter_count(frozenset({"fast_slow"}))
    w0_count, w0_trainable = parameter_count(frozenset({"w0"}))
    state_count, state_trainable = parameter_count(
        frozenset(
            {
                "state_shared",
                "state_task",
                "state_router_time",
                "state_retrieval",
            }
        )
    )
    associative_count, associative_trainable = parameter_count(frozenset({"associative"}))
    return WarmupOuterTrainabilityAudit(
        qwen_parameter_count=qwen_count,
        qwen_trainable_count=qwen_trainable,
        fast_slow_parameter_count=fast_slow_count,
        fast_slow_trainable_count=fast_slow_trainable,
        w0_parameter_count=w0_count,
        w0_trainable_count=w0_trainable,
        state_parameter_count=state_count,
        state_trainable_count=state_trainable,
        associative_parameter_count=associative_count,
        associative_trainable_count=associative_trainable,
    )


def _reset_a2_to_a5_associative(model: nn.Module) -> None:
    adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
    if len(adapters) != 1:
        raise RuntimeError("A2→A5 initialization requires exactly one FastTTTAdapter")
    adapters[0].reset_associative_projections()


def _memory_gradient_probes(
    model: nn.Module,
    expected_groups: tuple[str, ...],
) -> tuple[GradientProbe, ...]:
    """Split the pooled ``associative`` group into read-path and write-path witnesses.

    The write-path parameters reach the loss only through the Query deferred VJP,
    so their gradient norm is the sole direct evidence that the delta-rule write
    is being trained.  Pooled with ``p_context`` and ``memory_alpha`` — both of
    which are also fed by the read path — that signal is unrecoverable from the
    group norm alone.  ``w0`` is the reference denominator because the failure
    mode this exists to catch is W0's static-adapter gradient dominating the
    write path by orders of magnitude rather than the write path being exactly
    zero.
    """

    if "associative" not in expected_groups:
        return ()
    adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
    if len(adapters) != 1:
        raise RuntimeError("memory gradient probes require exactly one FastTTTAdapter")
    adapter = adapters[0]
    reference = "w0" if "w0" in expected_groups else None
    return (
        GradientProbe(
            name="memory_write",
            group_name="associative",
            parameters=tuple(adapter.collect_memory_write_parameters()),
            reference_group=reference,
        ),
        GradientProbe(
            name="memory_read",
            group_name="associative",
            parameters=tuple(adapter.collect_memory_read_parameters()),
            reference_group=reference,
        ),
    )


def _reset_a2_to_a5_balance(model: nn.Module) -> None:
    balancer = getattr(model, "official_weak_balancer", None)
    if not isinstance(balancer, OfficialWeakOuterLossComposer):
        raise RuntimeError("A5 outer model lost the official-weak EMA reset boundary")
    balancer.reset_ema()


def _a5_global_sample_sequence_sha256(
    dataset: object,
    sampler_factory: TrainSamplerFactory | None,
    *,
    epoch_count: float,
    world_size: int = 4,
) -> tuple[str, int]:
    """Hash the exact active-world-size global A5 record sequence before training starts."""

    if sampler_factory is None:
        raise RuntimeError("A5 sample-sequence audit requires the production sampler")
    if not math.isfinite(epoch_count) or epoch_count <= 0.0 or not epoch_count.is_integer():
        raise ValueError("A5 sample-sequence audit requires an integer epoch count")
    if world_size not in {4, 8}:
        raise ValueError("A5 sample-sequence audit supports only four or eight ranks")
    sampler = sampler_factory(dataset, 0, world_size)
    set_epoch = getattr(sampler, "set_epoch", None)
    if not callable(set_epoch):
        raise TypeError("A5 sample-sequence audit requires an epoch-aware sampler")
    digest = hashlib.sha256()
    count = 0
    for epoch in range(int(epoch_count)):
        set_epoch(epoch)
        for index in cast(Iterable[int], sampler):
            record = cast(Any, dataset)[index]
            if not isinstance(record, A5EpisodeRecord):
                raise TypeError("A5 sample-sequence audit encountered a non-A5 record")
            digest.update(f"{epoch}\t{record.episode_id}\n".encode())
            count += 1
    if count <= 0:
        raise RuntimeError("A5 sample-sequence audit produced no records")
    return digest.hexdigest(), count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _a2_checkpoint_sha256(checkpoint: Path) -> str:
    """Hash the complete loadable A2 model payload in stable filename order."""

    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"A2 checkpoint directory does not exist: {root}")
    single_candidates = (root / "model.safetensors", root / "pytorch_model.bin")
    present_single = tuple(path for path in single_candidates if path.is_file())
    index_candidates = (
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    )
    present_index = tuple(path for path in index_candidates if path.is_file())
    if len(present_single) + len(present_index) != 1:
        raise RuntimeError("A2 checkpoint must expose exactly one model weight entrypoint")
    if present_single:
        files = present_single
    else:
        index_path = present_index[0]
        raw = cast(object, json.loads(index_path.read_text(encoding="utf-8")))
        weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("A2 checkpoint shard index has no weight_map")
        shard_names = tuple(
            sorted({value for value in weight_map.values() if isinstance(value, str)})
        )
        if not shard_names or len(shard_names) != len(set(weight_map.values())):
            raise RuntimeError("A2 checkpoint shard index contains invalid shard names")
        files = (index_path, *(root / name for name in shard_names))
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"A2 checkpoint weight file is missing or empty: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _effective_project_config_sha256(project: ProjectConfig) -> str:
    payload = json.dumps(
        project.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _current_git_commit(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not commit:
        raise RuntimeError("Git did not return a code commit")
    return commit, dirty


def _warmup_bundle_allowlist(
    model: nn.Module,
    qwen_model: nn.Module,
) -> tuple[str, ...]:
    """Return every persistent non-Qwen tensor trained/carried by warmup."""

    qwen_ids = {
        id(value)
        for value in (
            *qwen_model.parameters(),
            *qwen_model.buffers(),
        )
    }
    # A non-persistent buffer is rebuilt from config at construction and is never
    # trained -- ``slot_codes`` is the fixed sinusoidal code table, pinned by
    # ``_validate_spatial_config`` -- so it is not bundle state and the receiver
    # reconstructs it identically.  Such buffers used to enter the allowlist (their
    # names match none of the excluded tokens) and then fail the subset check
    # below, which broke publication *after* a full warmup had already trained.
    # Parameters are always in ``state_dict``, so that check keeps its guard value
    # against a genuinely unsaveable trained tensor.
    non_persistent = _non_persistent_buffer_names(model)

    def _carried(name: str, value: Tensor) -> bool:
        if id(value) in qwen_ids:
            return False
        lowered = name.casefold()
        return not any(token in lowered for token in _WARMUP_BUNDLE_EXCLUDED_TOKENS)

    names: set[str] = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if _carried(name, parameter)
    }
    names |= {
        name
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if name not in non_persistent and _carried(name, buffer)
    }
    if not names:
        raise RuntimeError("warmup bundle allowlist contains no non-Qwen tensors")
    if not names <= set(model.state_dict()):
        raise RuntimeError("warmup bundle allowlist contains non-persistent tensors")
    if any(_is_transient_memory_name(name.casefold()) for name in names):
        raise RuntimeError("the transient per-video memory entered the warmup bundle allowlist")
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class _TensorBitwiseDigest:
    """One parameter or buffer's metadata and content digest."""

    name: str
    tensor_kind: str
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _ModuleBitwiseSnapshot:
    """Streaming module digest plus per-tensor evidence, without tensor copies."""

    sha256: str
    tensors: tuple[_TensorBitwiseDigest, ...]
    excluded_non_persistent_buffers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TensorBitwiseChange:
    """A complete per-tensor before/after difference."""

    name: str
    change_type: str
    baseline_kind: str | None
    final_kind: str | None
    baseline_dtype: str | None
    final_dtype: str | None
    baseline_shape: tuple[int, ...] | None
    final_shape: tuple[int, ...] | None
    baseline_content_sha256: str | None
    final_content_sha256: str | None


@dataclass(frozen=True, slots=True)
class WarmupQwenBitwiseAudit:
    """Post-DeepSpeed, pre-update Qwen invariance result for one rank."""

    baseline_stage: str
    baseline_global_step: int
    baseline_sha256: str
    final_sha256: str
    tensor_count: int
    parameter_count: int
    buffer_count: int
    changed_tensor_count: int
    changed_parameter_count: int
    changed_buffer_count: int
    final_trainable_parameter_names: tuple[str, ...]
    local_unchanged: bool
    all_ranks_unchanged: bool
    changed_tensors: tuple[_TensorBitwiseChange, ...]
    excluded_non_persistent_buffers: tuple[str, ...] = ()


class _WarmupQwenBitwiseAuditor:
    """Capture the frozen Qwen baseline only after Trainer/DeepSpeed preparation."""

    baseline_stage = "post_deepspeed_prepare_pre_first_training_step"

    def __init__(self, qwen_model: nn.Module) -> None:
        if not isinstance(qwen_model, nn.Module):
            raise TypeError("warmup Qwen bitwise audit requires an nn.Module")
        self.qwen_model = qwen_model
        self._baseline: _ModuleBitwiseSnapshot | None = None
        self._baseline_global_step: int | None = None
        self._final_audit: WarmupQwenBitwiseAudit | None = None

    @property
    def baseline_sha256(self) -> str | None:
        return None if self._baseline is None else self._baseline.sha256

    def capture_post_prepare_baseline(self, *, global_step: int) -> None:
        """Hash once, after prepare and before the first optimizer update."""

        if self._baseline is not None:
            return
        if type(global_step) is not int or global_step != 0:
            raise RuntimeError("warmup Qwen baseline must be captured before optimizer step one")
        trainable = tuple(
            name
            for name, parameter in self.qwen_model.named_parameters()
            if parameter.requires_grad
        )
        if trainable:
            raise RuntimeError(f"warmup Qwen baseline found trainable parameters: {trainable[:8]}")
        self._baseline = _module_bitwise_snapshot(self.qwen_model)
        self._baseline_global_step = global_step

    def finalize(self, *, device: torch.device) -> WarmupQwenBitwiseAudit:
        """Compare the final Qwen state on every rank without early divergence."""

        if self._final_audit is not None:
            return self._final_audit
        baseline_ready = _all_ranks_true(self._baseline is not None, device=device)
        if not baseline_ready or self._baseline is None:
            raise RuntimeError("one or more ranks missed the post-prepare warmup Qwen baseline")
        if self._baseline_global_step is None:
            raise RuntimeError("warmup Qwen baseline lost its optimizer-step boundary")
        final = _module_bitwise_snapshot(self.qwen_model)
        changes = _module_bitwise_changes(self._baseline, final)
        final_trainable = tuple(
            name
            for name, parameter in self.qwen_model.named_parameters()
            if parameter.requires_grad
        )
        local_unchanged = (
            not changes and not final_trainable and final.sha256 == self._baseline.sha256
        )
        all_ranks_unchanged = _all_ranks_true(local_unchanged, device=device)
        baseline_tensors = self._baseline.tensors
        parameter_count = sum(value.tensor_kind == "parameter" for value in baseline_tensors)
        buffer_count = sum(value.tensor_kind == "buffer" for value in baseline_tensors)
        changed_parameter_count = sum(
            (change.baseline_kind or change.final_kind) == "parameter" for change in changes
        )
        changed_buffer_count = sum(
            (change.baseline_kind or change.final_kind) == "buffer" for change in changes
        )
        self._final_audit = WarmupQwenBitwiseAudit(
            baseline_stage=self.baseline_stage,
            baseline_global_step=self._baseline_global_step,
            baseline_sha256=self._baseline.sha256,
            final_sha256=final.sha256,
            tensor_count=len(baseline_tensors),
            parameter_count=parameter_count,
            buffer_count=buffer_count,
            changed_tensor_count=len(changes),
            changed_parameter_count=changed_parameter_count,
            changed_buffer_count=changed_buffer_count,
            final_trainable_parameter_names=final_trainable,
            local_unchanged=local_unchanged,
            all_ranks_unchanged=all_ranks_unchanged,
            changed_tensors=changes,
            excluded_non_persistent_buffers=final.excluded_non_persistent_buffers,
        )
        return self._final_audit


def _non_persistent_buffer_names(module: nn.Module) -> frozenset[str]:
    """Collect the fully-qualified names of every non-persistent buffer."""

    names: set[str] = set()
    for prefix, submodule in module.named_modules():
        for local_name in getattr(submodule, "_non_persistent_buffers_set", ()):
            names.add(f"{prefix}.{local_name}" if prefix else local_name)
    return frozenset(names)


def _module_bitwise_snapshot(module: nn.Module) -> _ModuleBitwiseSnapshot:
    """Hash every parameter and persistent buffer.

    Non-persistent buffers are runtime scratch (rotary caches and the like) that
    legitimately changes value across forwards; hashing them produced false
    "Qwen changed" verdicts.  The excluded names are recorded so the exclusion
    itself stays auditable.
    """

    non_persistent = _non_persistent_buffer_names(module)
    named_tensors = [
        (name, "parameter", tensor)
        for name, tensor in module.named_parameters(remove_duplicate=False)
    ]
    named_tensors.extend(
        (name, "buffer", tensor)
        for name, tensor in module.named_buffers(remove_duplicate=False)
        if name not in non_persistent
    )
    names = tuple(name for name, _, _ in named_tensors)
    if len(names) != len(set(names)):
        raise RuntimeError("module exposes a parameter/buffer name collision")
    digest = hashlib.sha256()
    tensors: list[_TensorBitwiseDigest] = []
    for name, tensor_kind, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        dtype = str(value.dtype)
        shape = tuple(value.shape)
        payload = value.view(torch.uint8).numpy().tobytes()
        content_sha256 = hashlib.sha256(payload).hexdigest()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\n")
        tensors.append(
            _TensorBitwiseDigest(
                name=name,
                tensor_kind=tensor_kind,
                dtype=dtype,
                shape=shape,
                content_sha256=content_sha256,
            )
        )
    return _ModuleBitwiseSnapshot(
        sha256=digest.hexdigest(),
        tensors=tuple(tensors),
        excluded_non_persistent_buffers=tuple(sorted(non_persistent)),
    )


def _module_bitwise_changes(
    baseline: _ModuleBitwiseSnapshot,
    final: _ModuleBitwiseSnapshot,
) -> tuple[_TensorBitwiseChange, ...]:
    before = {value.name: value for value in baseline.tensors}
    after = {value.name: value for value in final.tensors}
    changes: list[_TensorBitwiseChange] = []
    for name in sorted(before.keys() | after.keys()):
        old = before.get(name)
        new = after.get(name)
        if old == new:
            continue
        if old is None:
            change_type = "added"
        elif new is None:
            change_type = "removed"
        else:
            metadata_changed = (
                old.tensor_kind != new.tensor_kind
                or old.dtype != new.dtype
                or old.shape != new.shape
            )
            content_changed = old.content_sha256 != new.content_sha256
            if metadata_changed and content_changed:
                change_type = "metadata_and_content"
            elif metadata_changed:
                change_type = "metadata"
            else:
                change_type = "content"
        changes.append(
            _TensorBitwiseChange(
                name=name,
                change_type=change_type,
                baseline_kind=None if old is None else old.tensor_kind,
                final_kind=None if new is None else new.tensor_kind,
                baseline_dtype=None if old is None else old.dtype,
                final_dtype=None if new is None else new.dtype,
                baseline_shape=None if old is None else old.shape,
                final_shape=None if new is None else new.shape,
                baseline_content_sha256=(None if old is None else old.content_sha256),
                final_content_sha256=(None if new is None else new.content_sha256),
            )
        )
    return tuple(changes)


def _module_bitwise_sha256(module: nn.Module) -> str:
    """Keep the original persistent-state digest for checkpoint provenance."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"module state entry {name!r} is not a Tensor")
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _all_ranks_true(value: bool, *, device: torch.device) -> bool:
    """Return the distributed logical AND without letting one rank fail early."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    result = torch.tensor(int(value), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MIN)
    return bool(result.item())


def _prepare_warmup_bundle_tensors(
    model: nn.Module,
    qwen_model: nn.Module,
) -> tuple[tuple[str, ...], dict[str, Tensor]]:
    """Materialize the handoff state on CPU; distributed callers run this on every rank."""

    allowlist = _warmup_bundle_allowlist(model, qwen_model)
    state = model.state_dict()
    tensors = {name: state[name].detach().cpu().contiguous().clone() for name in allowlist}
    return allowlist, tensors


def _prepare_distributed_warmup_handoff(
    *,
    model: nn.Module,
    qwen_model: nn.Module,
    qwen_auditor: _WarmupQwenBitwiseAuditor,
    device: torch.device,
) -> tuple[
    WarmupQwenBitwiseAudit,
    tuple[tuple[str, ...], dict[str, Tensor]] | None,
]:
    """Finish all GPU-backed warmup audits before any rank enters a publish barrier."""

    if qwen_auditor.qwen_model is not qwen_model:
        raise ValueError("warmup Qwen auditor does not own the loaded Qwen module")
    audit = qwen_auditor.finalize(device=device)
    prepared_bundle = (
        _prepare_warmup_bundle_tensors(model, qwen_model) if audit.all_ranks_unchanged else None
    )
    return audit, prepared_bundle


def _write_warmup_qwen_bitwise_audit(
    *,
    artifact_root: Path,
    audit: WarmupQwenBitwiseAudit,
) -> tuple[Path, Path | None]:
    """Persist complete rank-local tensor evidence before a possible fail-closed exit."""

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    payload = {"rank": rank, **asdict(audit)}
    rank_path = artifact_root / f"qwen_bitwise_audit.rank{rank}.json"
    _write_json(rank_path, payload)
    canonical_path: Path | None = None
    if rank == 0:
        canonical_path = artifact_root / "qwen_bitwise_audit.json"
        _write_json(canonical_path, payload)
    return rank_path, canonical_path


def _assert_warmup_qwen_bitwise_unchanged(
    audit: WarmupQwenBitwiseAudit,
) -> None:
    if audit.all_ranks_unchanged:
        return
    changed_names = tuple(change.name for change in audit.changed_tensors[:8])
    raise RuntimeError(
        "Qwen parameters/buffers changed after the post-DeepSpeed warmup baseline: "
        f"local_changed={audit.changed_tensor_count}, "
        f"changed_names={changed_names}, "
        "see qwen_bitwise_audit.rank*.json"
    )


def _warmup_source_manifest(
    *,
    backbone: LlamaFactoryBackboneBundle,
    code_root: Path,
) -> dict[str, object]:
    checkpoint_raw = backbone.ttt_config.initialize_from_a2_checkpoint
    if checkpoint_raw is None:
        raise RuntimeError("warmup source manifest lost the A2 checkpoint")
    commit, dirty = _current_git_commit(code_root)
    if dirty:
        raise RuntimeError("warmup bundle requires a clean Git working tree")
    dataset_manifest = Path(backbone.ttt_config.dataset_manifest).expanduser().resolve()
    if not dataset_manifest.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {dataset_manifest}")
    training_args = cast(Any, backbone.training_args)
    seed = int(training_args.seed)
    data_seed = int(training_args.data_seed)
    if seed != backbone.project_config.a5.seed or data_seed != seed:
        raise ValueError("warmup handoff requires matching seed/data_seed 42")
    return {
        "a2_checkpoint_sha256": _a2_checkpoint_sha256(Path(checkpoint_raw)),
        "code_commit": commit,
        "code_dirty": False,
        "project_config_sha256": _effective_project_config_sha256(backbone.project_config),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest),
        "seed": seed,
        "data_seed": data_seed,
    }


def _publish_warmup_bundle(
    *,
    model: nn.Module,
    qwen_model: nn.Module,
    backbone: LlamaFactoryBackboneBundle,
    artifact_root: Path,
    global_step: int,
    qwen_sha256: str,
    prepared_bundle: tuple[tuple[str, ...], dict[str, Tensor]] | None = None,
    qwen_warmup_audit: WarmupQwenBitwiseAudit | None = None,
) -> tuple[Path, dict[str, object]]:
    """Atomically publish the non-Qwen handoff only after a successful 256-step run."""

    expected_steps = int(backbone.project_config.a5.warmup.max_steps)
    if global_step != expected_steps:
        raise RuntimeError(f"warmup bundle requires step {expected_steps}, found {global_step}")
    if qwen_warmup_audit is not None and not qwen_warmup_audit.all_ranks_unchanged:
        raise RuntimeError("cannot publish a bundle after Qwen bitwise drift")
    destination = artifact_root / "a5_warmup_bundle"
    incomplete = artifact_root / ".a5_warmup_bundle.incomplete"
    if destination.exists() or incomplete.exists():
        raise FileExistsError("refusing to overwrite a warmup handoff bundle")
    if prepared_bundle is None:
        allowlist, tensors = _prepare_warmup_bundle_tensors(model, qwen_model)
    else:
        allowlist, tensors = prepared_bundle
        if tuple(sorted(tensors)) != allowlist:
            raise ValueError("prepared warmup bundle keys do not match its allowlist")
        if any(value.device.type != "cpu" for value in tensors.values()):
            raise ValueError("prepared warmup bundle tensors must be on CPU")
    incomplete.mkdir(parents=False)
    weights = incomplete / "model.safetensors"
    save_file(tensors, str(weights))
    source = _warmup_source_manifest(backbone=backbone, code_root=Path.cwd())
    manifest: dict[str, object] = {
        "bundle_schema_version": _WARMUP_BUNDLE_SCHEMA_VERSION,
        "associative_contract": backbone.project_config.associative_ttt.contract,
        "associative_contract_version": _WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION,
        "config_schema_version": backbone.project_config.config_schema_version,
        **source,
        "optimizer_steps": global_step,
        "parameter_allowlist": list(allowlist),
        "tensor_count": len(tensors),
        "qwen_bitwise_sha256": qwen_sha256,
        "bundle_sha256": _sha256_file(weights),
    }
    if qwen_warmup_audit is not None:
        manifest.update(
            {
                "qwen_warmup_baseline_stage": qwen_warmup_audit.baseline_stage,
                "qwen_warmup_baseline_sha256": qwen_warmup_audit.baseline_sha256,
                "qwen_warmup_final_sha256": qwen_warmup_audit.final_sha256,
                "qwen_warmup_tensor_count": qwen_warmup_audit.tensor_count,
            }
        )
    _write_json(incomplete / "manifest.json", manifest)
    incomplete.rename(destination)
    return destination, manifest


def _load_warmup_bundle(
    *,
    model: nn.Module,
    qwen_model: nn.Module,
    backbone: LlamaFactoryBackboneBundle,
    allow_code_drift: bool = False,
) -> dict[str, object]:
    """Strictly overlay a verified non-Qwen handoff before optimizer creation.

    ``allow_code_drift`` exempts ``code_commit`` from the equality comparison -- and only that
    key.  Training must never set it: A5 main training consumes a bundle published from the same
    commit, and a silent code difference there would invalidate the handoff.  Read-only
    *evaluation* of a published bundle is the one legitimate case, because adding an evaluator is
    itself a commit, so the bundle's recorded commit can never equal the evaluating commit.  The
    observed and expected commits are returned either way, so the exemption stays auditable
    rather than invisible.  The clean-working-tree requirement is *not* relaxed.
    """

    raw_path = backbone.ttt_config.warmup_bundle
    if raw_path is None:
        raise RuntimeError("A5 main lost the required warmup handoff path")
    root = Path(raw_path).expanduser().resolve()
    weights = root / "model.safetensors"
    manifest_path = root / "manifest.json"
    if not weights.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("warmup bundle requires model.safetensors and manifest.json")
    manifest = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
    if not isinstance(manifest, dict):
        raise ValueError("warmup bundle manifest must be a JSON object")
    expected_source = _warmup_source_manifest(backbone=backbone, code_root=Path.cwd())
    expected_scalars: dict[str, object] = {
        "bundle_schema_version": _WARMUP_BUNDLE_SCHEMA_VERSION,
        "associative_contract": backbone.project_config.associative_ttt.contract,
        "associative_contract_version": _WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION,
        "config_schema_version": backbone.project_config.config_schema_version,
        "optimizer_steps": int(backbone.project_config.a5.warmup.max_steps),
        "qwen_bitwise_sha256": _module_bitwise_sha256(qwen_model),
        **expected_source,
        "bundle_sha256": _sha256_file(weights),
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_scalars.items()
        if manifest.get(key) != expected
        and not (allow_code_drift and key == "code_commit")
    }
    if mismatches:
        raise ValueError(f"warmup bundle provenance mismatch: {mismatches}")
    allowlist = _warmup_bundle_allowlist(model, qwen_model)
    if manifest.get("parameter_allowlist") != list(allowlist):
        raise ValueError("warmup bundle parameter allowlist does not match the current model")
    if manifest.get("tensor_count") != len(allowlist):
        raise ValueError("warmup bundle tensor count does not match its strict allowlist")
    tensors = load_file(str(weights), device="cpu")
    if tuple(sorted(tensors)) != allowlist:
        raise ValueError("warmup bundle tensor keys do not match its strict allowlist")
    current = model.state_dict()
    for name, value in tensors.items():
        expected = current[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"warmup tensor {name} must be {expected.dtype} {tuple(expected.shape)}, "
                f"found {value.dtype} {tuple(value.shape)}"
            )
    result = model.load_state_dict(tensors, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"warmup bundle produced unexpected keys: {result.unexpected_keys}")
    expected_missing = tuple(sorted(set(current) - set(allowlist)))
    if tuple(sorted(result.missing_keys)) != expected_missing:
        raise RuntimeError("warmup bundle missing-key boundary drifted")
    return {
        "path": str(root),
        "bundle_sha256": expected_scalars["bundle_sha256"],
        "tensor_count": len(tensors),
        "parameter_allowlist_sha256": hashlib.sha256(
            "\n".join(allowlist).encode("utf-8")
        ).hexdigest(),
        "source": expected_source,
        # Recorded unconditionally so a relaxed load is never indistinguishable from a strict one.
        "code_commit_exempted": bool(allow_code_drift),
        "bundle_code_commit": manifest.get("code_commit"),
        "loading_code_commit": expected_source.get("code_commit"),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EpisodeAdapter",
    "OuterParameterAudit",
    "ProductionStage",
    "ProductionTrainerRuntime",
    "SegmentBackwardController",
    "StageALossStep",
    "TrainSamplerFactory",
    "TTTQwenTrainerMixin",
    "build_production_trainer",
    "build_trainer_class",
    "make_production_outer_optimizer_factory",
    "main",
    "resolve_same_stage_resume",
]
