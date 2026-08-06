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
import sys
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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

from ttt_svcbench_qwen.episode_data import (
    ManifestStage,
    build_production_train_sampler,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.fast_ttt import FastTTTAdapter
from ttt_svcbench_qwen.meta_trainer import (
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    TruncatedMetaTTTEpisodeOutput,
)
from ttt_svcbench_qwen.outer_gradient_control import (
    OuterGradientController,
    sanitize_scalar_loss,
)
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    configure_qwen_outer_trainability,
    initialize_outer_model_from_a2,
    load_llamafactory_backbone,
)
from ttt_svcbench_qwen.runtime_metrics import (
    flush_runtime_metrics,
    trace_cuda_phase,
)
from ttt_svcbench_qwen.stage_a_targets import OfficialWeakLossAudit


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

    def backward(self, loss: Tensor, sync_gradients: bool = True, **kwargs: object) -> None:
        engine = cast(Any, self.engine)
        engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
        engine.backward(loss, **kwargs)
        if sync_gradients:
            self.gradient_controller.apply_deepspeed(engine.optimizer)
            engine.step()

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
                if retain_graph:
                    engine.backward(loss, retain_graph=True)
                else:
                    engine.backward(loss)
            else:
                cast(Any, self.accelerator).backward(loss, retain_graph=retain_graph)
        self.backward_count += 1

    def finalize(self) -> None:
        if self.is_deepspeed:
            engine = cast(Any, self.engine)
            if self.gradient_controller is not None:
                self.gradient_controller.apply_deepspeed(engine.optimizer)
            engine.step()
            self.step_count = 1

    @property
    def semantic_projector_metrics(self) -> dict[str, float]:
        return {}


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


class _LazyGradientAccumulationGroup(Sequence[object]):
    """Pull one A2 microbatch only when the pinned Trainer loop is ready to execute it."""

    def __init__(self, iterator: Iterator[object], expected_count: int) -> None:
        if expected_count <= 0:
            raise ValueError("lazy GA group requires a positive batch count")
        self.iterator = iterator
        self.expected_count = expected_count
        self._cache: list[object] = []

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
        try:
            batch = next(self.iterator)
        except StopIteration as error:
            raise RuntimeError(
                "A2 DataLoader ended before the declared gradient-accumulation group"
            ) from error
        self._cache.append(batch)


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
        for term_index, name in enumerate(("task", "operator", "retrieval", "time")):
            terms = tuple(audit.terms[term_index] for audit in balances)
            count = sum(float(term.global_valid_count.item()) for term in terms)
            metrics[f"loss/global_valid_count/{name}"] = count
            metrics[f"grad_balance/global_valid_count/{name}"] = count
            for key, values in (
                (f"loss/scale/{name}", tuple(term.scale for term in terms)),
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
                (f"grad_balance/final_scale/{name}", tuple(term.scale for term in terms)),
            ):
                _set_optional_metric(metrics, key, _audit_mean(values))

        guard = _audit_mean(tuple(audit.group_guard for audit in balances))
        if guard is not None:
            metrics["loss/group_guard"] = guard

        for audit in weak:
            for name, value in audit.metrics():
                metrics[name] = metrics.get(name, 0.0) + float(value)
        return metrics


def _audit_scalar(value: Tensor) -> float | None:
    result = float(value.item())
    return result if math.isfinite(result) else None


def _audit_mean(values: Sequence[Tensor]) -> float | None:
    finite = tuple(value for value in (_audit_scalar(item) for item in values) if value is not None)
    return sum(finite) / float(len(finite)) if finite else None


def _set_optional_metric(metrics: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        metrics[name] = value


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
            retrieval_metrics: dict[str, float] = {}
            meta_output = self.last_meta_output
            meta_audit = meta_output.audit
            if meta_audit.loss_weight == 1.0:
                balance = getattr(self.ttt_runtime.meta_runner, "last_balance_audit", None)
                if isinstance(balance, OfficialWeakBalanceAudit):
                    retrieval_metrics["retrieval/valid_bag_rows"] = float(
                        balance.terms[2].global_valid_count.item()
                    )
            enriched.update(retrieval_metrics)
            enriched.update(
                {
                    "a5/meta_query_count": float(meta_audit.query_count),
                    "a5/segment_count": float(meta_audit.segment_count),
                    "a5/memory/write_valid": float(meta_audit.associative_valid_count > 0),
                    "a5/memory/write_count": float(meta_audit.write_count),
                    "a5/memory/skip_count": float(meta_audit.skip_count),
                    "a5/loss/meta_query_sum": float(meta_output.query_loss.item()),
                    "memory/readout_target_cosine": meta_audit.readout_target_cosine_mean,
                    "memory/pre_write_error_mean": (
                        1.0 - meta_audit.readout_target_cosine_mean
                    ),
                }
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
            self.last_semantic_projector_metrics = {}
            return result
        if int(self.args.gradient_accumulation_steps) != 1:  # type: ignore[attr-defined]
            raise ValueError("A5 uses one complete episode/rank and episode-level GA=1")
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
        segment_lengths = episode.segment_lengths
        expected_segments = len(segment_lengths)
        expected_backwards = len(episode.query_points) + expected_segments

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
            )
        finally:
            if callable(end_prefetch):
                end_prefetch()
        backward_controller.finalize()
        self.last_semantic_projector_metrics = backward_controller.semantic_projector_metrics
        self.last_meta_output = output
        local_training_seconds = time.perf_counter() - step_started
        timing = torch.tensor(
            [local_training_seconds],
            dtype=torch.float64,
            device=self.args.device,  # type: ignore[attr-defined]
        )
        gathered_timing = self.accelerator.gather(timing)  # type: ignore[attr-defined]
        self._last_a5_training_seconds = float(gathered_timing.max().item())
        return (output.total * loss_weight).detach().to(self.args.device)  # type: ignore[attr-defined]


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
    configure_qwen_outer_trainability(
        backbone.model,
        backbone.project_config,
        backbone.ttt_config.qwen_outer_trainability,
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
    from ttt_svcbench_qwen.production_runtime import build_runtime

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
    warmup_bundle_audit: dict[str, object] | None = None
    if configured_stage is ProductionStage.A5 and same_stage_resume is None:
        checkpoint = backbone.ttt_config.initialize_from_a2_checkpoint
        if checkpoint is None:
            raise RuntimeError("validated A5 config lost initialize_from_a2_checkpoint")
        _validate_resume_balance_schema(Path(checkpoint).expanduser().resolve())
        initialize_outer_model_from_a2(runtime_raw.model, checkpoint)
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
        _configure_fast_state_warmup_trainability(
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
        ),
        semantic_projector_delta_audit_steps=(
            backbone.ttt_config.semantic_projector_delta_audit_steps
        ),
        a5_parameter_delta_audit_steps=(backbone.ttt_config.a5_parameter_delta_audit_steps),
        operator_diagnostics_interval=backbone.ttt_config.operator_diagnostics_interval,
        train_sampler_factory=(
            lambda dataset, rank, world_size: build_production_train_sampler(
                dataset,
                rank,
                world_size,
                query_sample_fps=backbone.ttt_config.query_sample_fps,
                state_query_visual_mode=backbone.ttt_config.state_query_visual_mode,
                state_query_max_frames=backbone.ttt_config.state_query_max_frames,
                answer_query_visual_mode=backbone.ttt_config.answer_query_visual_mode,
                answer_query_max_frames=backbone.ttt_config.answer_query_max_frames,
            )
        ),
    )
    training_args = cast(Any, backbone.training_args)
    project = backbone.project_config
    _disable_smoke_checkpoints(training_args)
    trainer = cast(Any, build_production_trainer(backbone, runtime_raw))
    output_dir = Path(str(training_args.output_dir))
    artifact_root = Path(os.environ.get("RUN_ROOT", str(output_dir)))
    if trainer.is_world_process_zero():
        _write_json(
            artifact_root / "environment.json",
            {
                "stage": runtime_raw.stage.value,
                "a5_phase": backbone.ttt_config.a5_phase,
                "memory_eta_max_per_slot": float(project.fast_memory.eta_max_per_slot),
                "memory_eta_chunk_budget": float(project.fast_memory.eta_chunk_budget),
                "memory_forget_beta_max": float(project.fast_memory.forget_beta_max),
                "warmup_bundle_audit": warmup_bundle_audit,
            },
        )
    if configured_stage is ProductionStage.A5:
        trainer.accelerator.wait_for_everyone()
    try:
        result = trainer.train(
            resume_from_checkpoint=None if same_stage_resume is None else str(same_stage_resume)
        )
    finally:
        flush_runtime_metrics(resolve_cuda=True)
    if backbone.ttt_config.a5_phase == "fast_state_warmup":
        trainer.accelerator.wait_for_everyone()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        # No rank may enter the publishing barrier while another rank is still copying
        # tensors from CUDA.  From this point onward rank zero performs CPU/filesystem work.
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            bundle_path, bundle_manifest = _publish_warmup_bundle(
                model=runtime_raw.model,
                qwen_model=backbone.model,
                backbone=backbone,
                artifact_root=artifact_root,
                global_step=int(trainer.state.global_step),
            )
            _write_json(
                artifact_root / "run_summary.json",
                {
                    "status": "completed",
                    "stage": runtime_raw.stage.value,
                    "a5_phase": "fast_state_warmup",
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics": result.metrics,
                    "warmup_bundle": str(bundle_path),
                    "warmup_bundle_manifest": bundle_manifest,
                    "final_checkpoint": None,
                },
            )
        trainer.accelerator.wait_for_everyone()
        return 0
    final_checkpoint = output_dir / "final-checkpoint"
    incomplete_checkpoint = output_dir / ".final-checkpoint.incomplete"
    if trainer.is_world_process_zero() and (
        final_checkpoint.exists() or incomplete_checkpoint.exists()
    ):
        raise FileExistsError("refusing to overwrite an existing final checkpoint")
    trainer.accelerator.wait_for_everyone()
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
                "global_step": int(trainer.state.global_step),
                "elapsed_seconds": time.monotonic() - started,
                "metrics": result.metrics,
                "final_checkpoint": str(final_checkpoint),
                "resume_state": str(final_checkpoint / "resume_state"),
                "resumed_from": (None if same_stage_resume is None else str(same_stage_resume)),
            },
        )
    return 0


def _disable_smoke_checkpoints(training_args: object) -> None:
    """Disable periodic saves when the atomic policy publishes only the final checkpoint."""

    arguments = cast(Any, training_args)
    strategy = arguments.save_strategy
    strategy_type = type(strategy)
    arguments.save_strategy = "no" if strategy_type is str else strategy_type("no")
    arguments.save_steps = 0


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


def make_production_outer_optimizer_factory(
    backbone: LlamaFactoryBackboneBundle,
    stage: ProductionStage,
    *,
    a5_adaptation_mode: str = "meta_ttt",
    a5_phase: str = "main",
) -> Callable[[nn.Module], torch.optim.Optimizer]:
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
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(float(training_args.adam_beta1), float(training_args.adam_beta2)),
            eps=float(training_args.adam_epsilon),
            weight_decay=float(training_args.weight_decay),
        )
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


def _configure_fast_state_warmup_trainability(
    model: nn.Module,
    qwen_model: nn.Module,
) -> None:
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
        if parameter_id in grouped:
            continue
        grouped[parameter_id] = (group, parameter)
    allowed_groups = _A5_WARMUP_TRAINABLE_GROUPS
    for group, parameter in grouped.values():
        parameter.requires_grad_(group in allowed_groups)


def _reset_a2_to_a5_associative(model: nn.Module) -> None:
    adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
    if len(adapters) != 1:
        raise RuntimeError("A2→A5 initialization requires exactly one FastTTTAdapter")
    adapters[0].reset_associative_projections()


def _reset_a2_to_a5_balance(model: nn.Module) -> None:
    balancer = getattr(model, "official_weak_balancer", None)
    if not isinstance(balancer, OfficialWeakOuterLossComposer):
        raise RuntimeError("A5 outer model lost the official-weak EMA reset boundary")
    balancer.reset_ema()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    return tuple(sorted(names))


def _non_persistent_buffer_names(module: nn.Module) -> frozenset[str]:
    """Collect the fully-qualified names of every non-persistent buffer."""

    names: set[str] = set()
    for prefix, submodule in module.named_modules():
        for local_name in getattr(submodule, "_non_persistent_buffers_set", ()):
            names.add(f"{prefix}.{local_name}" if prefix else local_name)
    return frozenset(names)


def _prepare_warmup_bundle_tensors(
    model: nn.Module,
    qwen_model: nn.Module,
) -> tuple[tuple[str, ...], dict[str, Tensor]]:
    """Materialize the handoff state on CPU; distributed callers run this on every rank."""

    allowlist = _warmup_bundle_allowlist(model, qwen_model)
    state = model.state_dict()
    tensors = {name: state[name].detach().cpu().contiguous().clone() for name in allowlist}
    return allowlist, tensors


def _publish_warmup_bundle(
    *,
    model: nn.Module,
    qwen_model: nn.Module,
    backbone: LlamaFactoryBackboneBundle,
    artifact_root: Path,
    global_step: int,
) -> tuple[Path, dict[str, object]]:
    """Atomically publish the non-Qwen handoff after the warmup run."""

    destination = artifact_root / "a5_warmup_bundle"
    incomplete = artifact_root / ".a5_warmup_bundle.incomplete"
    if destination.exists() or incomplete.exists():
        raise FileExistsError("refusing to overwrite a warmup handoff bundle")
    allowlist, tensors = _prepare_warmup_bundle_tensors(model, qwen_model)
    incomplete.mkdir(parents=False)
    weights = incomplete / "model.safetensors"
    save_file(tensors, str(weights))
    manifest: dict[str, object] = {
        "bundle_schema_version": _WARMUP_BUNDLE_SCHEMA_VERSION,
        "associative_contract": backbone.project_config.associative_ttt.contract,
        "associative_contract_version": _WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION,
        "optimizer_steps": global_step,
        "parameter_allowlist": list(allowlist),
        "tensor_count": len(tensors),
        "bundle_sha256": _sha256_file(weights),
    }
    _write_json(incomplete / "manifest.json", manifest)
    incomplete.rename(destination)
    return destination, manifest


def _load_warmup_bundle(
    *,
    model: nn.Module,
    qwen_model: nn.Module,
    backbone: LlamaFactoryBackboneBundle,
) -> dict[str, object]:
    """Overlay the published non-Qwen handoff before optimizer creation."""

    raw_path = backbone.ttt_config.warmup_bundle
    if raw_path is None:
        raise RuntimeError("A5 main lost the required warmup handoff path")
    root = Path(raw_path).expanduser().resolve()
    weights = root / "model.safetensors"
    manifest_path = root / "manifest.json"
    manifest = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
    allowlist = _warmup_bundle_allowlist(model, qwen_model)
    tensors = load_file(str(weights), device="cpu")
    result = model.load_state_dict(tensors, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"warmup bundle produced unexpected keys: {result.unexpected_keys}")
    return {
        "path": str(root),
        "bundle_sha256": _sha256_file(weights),
        "tensor_count": len(tensors),
        "parameter_allowlist_sha256": hashlib.sha256(
            "\n".join(allowlist).encode("utf-8")
        ).hexdigest(),
        "manifest_schema_version": (
            manifest.get("bundle_schema_version") if isinstance(manifest, dict) else None
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EpisodeAdapter",
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
