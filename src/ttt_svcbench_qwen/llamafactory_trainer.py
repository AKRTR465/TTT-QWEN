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
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

# ``python -m`` executes this file as ``__main__``.  The dynamically loaded production runtime
# imports the canonical package name, so register the running module under that name before the
# runtime factory is imported.  Otherwise Python creates a second copy of the dataclasses/enums
# and a valid ProductionTrainerRuntime fails the identity-based boundary audit.
if __name__ == "__main__":
    sys.modules.setdefault("ttt_svcbench_qwen.llamafactory_trainer", sys.modules[__name__])

import torch
import transformers
from torch import Tensor, nn

from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    build_production_train_sampler,
    load_production_manifest_views,
    load_visual_cost_index,
)
from ttt_svcbench_qwen.meta_trainer import (
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    TruncatedMetaTTTEpisodeOutput,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    OuterCheckpointAudit,
    audit_outer_checkpoint_boundary,
    environment_manifest,
    fully_unfreeze_qwen,
    initialize_outer_model_from_a2,
    load_llamafactory_backbone,
)
from ttt_svcbench_qwen.runtime_metrics import flush_runtime_metrics, trace_cuda_phase
from ttt_svcbench_qwen.visual_cost import (
    VisualCostRecord,
    make_visual_cost_fingerprint,
)


class ProductionStage(StrEnum):
    A2 = "a2"
    A5 = "a5"


class StageALossStep(Protocol):
    def __call__(self, model: nn.Module, inputs: Mapping[str, object]) -> Tensor: ...


class EpisodeAdapter(Protocol):
    def __call__(self, inputs: Mapping[str, object]) -> tuple[MetaTTTEpisode, float]: ...


class TrainSamplerFactory(Protocol):
    def __call__(self, dataset: object, rank: int, world_size: int) -> object: ...


class SegmentBackwardController:
    """Accumulate segment gradients and make DeepSpeed step exactly once per episode.

    Accelerate's DeepSpeed backward wrapper also calls ``engine.step()``.  It therefore cannot
    be used for each TBPTT segment.  Direct ``engine.backward`` preserves all segment gradients;
    ``finalize`` executes the sole engine step only after the runner has audited unchanged Outer
    parameter versions.
    """

    def __init__(self, accelerator: object, model: nn.Module, *, expected_count: int) -> None:
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("segment backward count must be a positive integer")
        self.accelerator = accelerator
        self.expected_count = expected_count
        self.backward_count = 0
        self.step_count = 0
        self.is_deepspeed = (
            "deepspeed" in str(getattr(accelerator, "distributed_type", "")).casefold()
        )
        wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
        self.engine = getattr(wrapper, "engine", None) if self.is_deepspeed else None
        if self.is_deepspeed:
            if self.engine is None:
                self.engine = model
            required = ("set_gradient_accumulation_boundary", "backward", "step")
            if any(not callable(getattr(self.engine, name, None)) for name in required):
                raise TypeError(
                    "DeepSpeed segment controller requires boundary/backward/step methods"
                )
        elif not callable(getattr(accelerator, "backward", None)):
            raise TypeError("segment controller requires accelerator.backward")

    def backward(self, loss: Tensor) -> None:
        if self.backward_count >= self.expected_count:
            raise RuntimeError("segment runner emitted too many backward calls")
        with trace_cuda_phase(
            "backward",
            stage="a5_segment",
            segment_index=self.backward_count,
        ):
            if self.is_deepspeed:
                engine = cast(Any, self.engine)
                is_final_segment = self.backward_count + 1 == self.expected_count
                engine.set_gradient_accumulation_boundary(is_boundary=is_final_segment)
                engine.backward(loss)
            else:
                cast(Any, self.accelerator).backward(loss)
        self.backward_count += 1

    def finalize(self) -> None:
        if self.backward_count != self.expected_count:
            raise RuntimeError("segment runner backward count did not match its bucket")
        if self.step_count:
            raise RuntimeError("segment backward controller was finalized more than once")
        if self.is_deepspeed:
            cast(Any, self.engine).step()
            self.step_count = 1


@dataclass(frozen=True, slots=True)
class OuterParameterAudit:
    stage: ProductionStage
    total_parameter_count: int
    trainable_parameter_count: int
    predictor_parameter_count: int
    predictor_trainable_count: int
    transient_parameter_names: tuple[str, ...]
    backbone_registered: bool

    def __post_init__(self) -> None:
        if self.total_parameter_count <= 0 or self.trainable_parameter_count <= 0:
            raise ValueError("production outer model exposes no trainable parameters")
        if self.predictor_parameter_count <= 0:
            raise ValueError("production outer model must register Predictor parameters")
        if self.transient_parameter_names:
            raise ValueError("transient fast matrices entered registered outer parameters")
        if not self.backbone_registered:
            raise ValueError("runtime model did not register the loaded Qwen backbone")
        if self.stage is ProductionStage.A2:
            if self.predictor_trainable_count:
                raise ValueError("A2 Predictor must remain frozen")
            expected = self.total_parameter_count - self.predictor_parameter_count
            if self.trainable_parameter_count != expected:
                raise ValueError("A2 must train every registered non-Predictor parameter")
        elif self.trainable_parameter_count != self.total_parameter_count:
            raise ValueError("A5 must train Qwen, state modules, W0, and Predictor")


@dataclass(frozen=True, slots=True)
class ProductionTrainerRuntime:
    """Dataset/materialization hooks assembled entirely inside TTT-QWEN."""

    stage: ProductionStage
    model: nn.Module
    train_dataset: object
    eval_dataset: object | None
    data_collator: Callable[..., object]
    stage_a_loss_step: StageALossStep | None = None
    meta_runner: MetaTTTEpisodeRunner | None = None
    episode_adapter: EpisodeAdapter | None = None
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] | None = None
    train_sampler_factory: TrainSamplerFactory | None = None
    callbacks: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ProductionStage) or not isinstance(self.model, nn.Module):
            raise TypeError("production runtime stage/model is invalid")
        if not callable(self.data_collator):
            raise TypeError("production runtime requires a data collator")
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
        super().__init__(*args, **kwargs)

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

    def log(self, logs: dict[str, float], *args: object, **kwargs: object) -> None:
        enriched = dict(logs)
        audit: object | None
        if self.ttt_runtime.stage is ProductionStage.A2:
            audit = getattr(self.ttt_runtime.stage_a_loss_step, "last_balance_audit", None)
        else:
            audit = getattr(self.ttt_runtime.meta_runner, "last_balance_audit", None)
        metrics = getattr(audit, "metrics", None)
        if callable(metrics):
            for name, value in metrics():
                if value is not None:
                    enriched[name] = float(value)
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
            _validate_scalar_loss(loss, "A2 state+answer")
            return loss
        return cast(Tensor, super().compute_loss(model, inputs, *args, **kwargs))  # type: ignore[misc]

    def training_step(
        self,
        model: nn.Module,
        inputs: Mapping[str, object],
        num_items_in_batch: Tensor | None = None,
    ) -> Tensor:
        step_started = time.perf_counter()
        if self.ttt_runtime.stage is ProductionStage.A2:
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
            self._observe_runtime_cost(inputs, time.perf_counter() - step_started)
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
        expected_segments = math.ceil(
            len(episode.support_chunks) / runner.config.stage_c.truncation_horizon
        )
        horizon = runner.config.stage_c.truncation_horizon
        segment_lengths = tuple(
            min(horizon, len(episode.support_chunks) - start)
            for start in range(0, len(episode.support_chunks), horizon)
        )
        self._assert_rank_episode_parity(segment_lengths, len(episode.query_points))

        backward_controller = SegmentBackwardController(
            self.accelerator,  # type: ignore[attr-defined]
            model,
            expected_count=expected_segments,
        )

        def distributed_backward(loss: Tensor) -> None:
            backward_controller.backward(loss * loss_weight)

        end_prefetch = getattr(adapter, "end_prefetch", None)
        try:
            output = runner.run_truncated(episode, backward=distributed_backward)
        finally:
            if callable(end_prefetch):
                end_prefetch()
        if output.audit.backward_count != expected_segments:
            raise RuntimeError("A5 backward collective count drifted from its segment bucket")
        backward_controller.finalize()
        self.last_meta_output = output
        self._observe_runtime_cost(inputs, time.perf_counter() - step_started)
        return (output.total * loss_weight).detach().to(self.args.device)  # type: ignore[attr-defined]

    def _observe_runtime_cost(
        self,
        inputs: Mapping[str, object],
        seconds: float,
    ) -> None:
        sampler = getattr(self, "_ttt_train_sampler", None)
        observe = getattr(sampler, "observe_runtime_cost", None)
        if not callable(observe):
            return
        prepared = inputs.get(
            "prepared_a2"
            if self.ttt_runtime.stage is ProductionStage.A2
            else "prepared_a5"
        )
        record = getattr(prepared, "record", None)
        if isinstance(record, A2QueryRecord):
            observe(record.query.runtime.query_id, seconds)
        elif isinstance(record, A5EpisodeRecord):
            observe(record.episode_id, seconds)

    def _assert_rank_episode_parity(
        self,
        segment_lengths: tuple[int, ...],
        query_count: int,
    ) -> None:
        device = self.args.device  # type: ignore[attr-defined]
        local = torch.tensor(
            (query_count, *segment_lengths),
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
                "A5 ranks received unequal segment lengths or Query counts: "
                f"{signatures}"
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


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise ValueError("usage: python -m ttt_svcbench_qwen.llamafactory_trainer CONFIG.yaml")
    started = time.monotonic()
    backbone = load_llamafactory_backbone(arguments[0])
    balance_mode = backbone.project_config.loss.official_weak_balance.mode.value
    if balance_mode != "instant_equal" and os.environ.get("TTT_LEGACY_SUM_ABLATION") != "1":
        raise ValueError(
            "formal A2/A5 requires instant_equal; set TTT_LEGACY_SUM_ABLATION=1 "
            "only for an explicit ablation run"
        )
    unfreeze_audit = fully_unfreeze_qwen(backbone.model, backbone.project_config)
    configured_stage = ProductionStage(backbone.ttt_config.stage)
    if getattr(backbone.training_args, "resume_from_checkpoint", None) is not None:
        raise ValueError(
            "set TTT_RESUME_CHECKPOINT for same-stage resume; YAML resume is forbidden"
        )
    same_stage_resume = resolve_same_stage_resume(
        os.environ.get("TTT_RESUME_CHECKPOINT"),
        configured_stage,
    )
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
            manifest_sha256=hashlib.sha256(
                Path(manifest_path).read_bytes()
            ).hexdigest(),
            model_revision=f"{model_name}@{revision}",
            transformers_version=transformers.__version__,
            processor=(
                f"{type(backbone.processor).__module__}."
                f"{type(backbone.processor).__qualname__}"
            ),
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            dtype=str(parameter.dtype).removeprefix("torch."),
            visual_batch_size=backbone.ttt_config.support_visual_batch_size,
            cache_mode=backbone.ttt_config.preprocess_cache_mode,
            loss_mode=balance.mode.value,
            loss_group_weight=balance.group_weight,
            loss_scale_min=balance.scale_min,
            loss_scale_max=balance.scale_max,
            loss_epsilon=balance.epsilon,
            gpu_model=(
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            ),
        )
        visual_cost_index = load_visual_cost_index(
            raw_cost_index,
            expected_fingerprint=expected_fingerprint,
        )
    checkpoint_audit: OuterCheckpointAudit | None = None
    if configured_stage is ProductionStage.A5 and same_stage_resume is None:
        checkpoint = backbone.ttt_config.initialize_from_a2_checkpoint
        if checkpoint is None:
            raise RuntimeError("validated A5 config lost initialize_from_a2_checkpoint")
        checkpoint_audit = initialize_outer_model_from_a2(runtime_raw.model, checkpoint)
    runtime_raw = replace(
        runtime_raw,
        optimizer_factory=make_production_outer_optimizer_factory(
            backbone,
            configured_stage,
        ),
        train_sampler_factory=(
            lambda dataset, rank, world_size: build_production_train_sampler(
                dataset,
                rank,
                world_size,
                visual_cost_index=visual_cost_index,
            )
        ),
    )
    parameter_audit = _audit_outer_parameters(backbone, runtime_raw)
    audit_outer_checkpoint_boundary(runtime_raw.model)
    training_args = cast(Any, backbone.training_args)
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
    trainer = cast(Any, build_production_trainer(backbone, runtime_raw))
    output_dir = Path(str(training_args.output_dir))
    artifact_root = Path(os.environ.get("RUN_ROOT", str(output_dir)))
    if trainer.is_world_process_zero():
        environment = environment_manifest(backbone)
        environment["full_unfreeze_audit"] = asdict(unfreeze_audit)
        environment["outer_parameter_audit"] = asdict(parameter_audit)
        checkpoint_environment = None
        if checkpoint_audit is not None:
            checkpoint_environment = asdict(checkpoint_audit)
            checkpoint_environment["checkpoint"] = str(checkpoint_audit.checkpoint)
        environment["a2_initialization_audit"] = checkpoint_environment
        _write_json(artifact_root / "environment.json", environment)
    try:
        result = trainer.train(
            resume_from_checkpoint=None if same_stage_resume is None else str(same_stage_resume)
        )
    finally:
        flush_runtime_metrics(resolve_cuda=True)
    if skip_final_checkpoint:
        trainer.accelerator.wait_for_everyone()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        if trainer.is_world_process_zero():
            _write_json(
                artifact_root / "run_summary.json",
                {
                    "status": "smoke_completed",
                    "stage": runtime_raw.stage.value,
                    "global_step": int(trainer.state.global_step),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics": result.metrics,
                    "checkpoint_policy": "none_for_smoke",
                    "final_checkpoint": None,
                    "resume_state": None,
                    "resumed_from": None,
                },
            )
        return 0
    final_checkpoint = output_dir / "final-checkpoint"
    audit_outer_checkpoint_boundary(runtime_raw.model)
    trainer.save_model(str(final_checkpoint))
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.save_state(str(final_checkpoint / "resume_state"))
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(final_checkpoint / "trainer_state.json"))
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
                "checkpoint_policy": "final_epoch",
                "final_checkpoint": str(final_checkpoint),
                "resume_state": str(final_checkpoint / "resume_state"),
                "resumed_from": (None if same_stage_resume is None else str(same_stage_resume)),
            },
        )
    return 0


def _validate_scalar_loss(loss: Tensor, name: str) -> None:
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ValueError(f"{name} loss must be one differentiable scalar Tensor")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ValueError(f"{name} loss must be finite")


def resolve_same_stage_resume(
    checkpoint: str | None,
    stage: ProductionStage,
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
    return root


def _audit_outer_parameters(
    backbone: LlamaFactoryBackboneBundle,
    runtime: ProductionTrainerRuntime,
) -> OuterParameterAudit:
    named = tuple(runtime.model.named_parameters())
    predictor = tuple(
        (name, parameter) for name, parameter in named if "predictor" in name.casefold()
    )
    transient = tuple(
        name
        for name, _ in named
        if "transient_w_t" in name.casefold() or name.casefold().endswith(("w_t_1", "w_t_2"))
    )
    backbone_ids = {id(parameter) for parameter in backbone.model.parameters()}
    runtime_ids = {id(parameter) for _, parameter in named}
    return OuterParameterAudit(
        stage=runtime.stage,
        total_parameter_count=sum(parameter.numel() for _, parameter in named),
        trainable_parameter_count=sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        predictor_parameter_count=sum(parameter.numel() for _, parameter in predictor),
        predictor_trainable_count=sum(
            parameter.numel() for _, parameter in predictor if parameter.requires_grad
        ),
        transient_parameter_names=transient,
        backbone_registered=bool(backbone_ids) and backbone_ids <= runtime_ids,
    )


def make_production_outer_optimizer_factory(
    backbone: LlamaFactoryBackboneBundle,
    stage: ProductionStage,
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    qwen_ids = {id(parameter) for parameter in backbone.model.parameters()}
    training_args = cast(Any, backbone.training_args)
    if stage is ProductionStage.A2:
        qwen_lr = backbone.project_config.stage_a.optimizer.qwen_learning_rate
        state_lr = backbone.project_config.stage_a.optimizer.state_learning_rate
        w0_lr = backbone.project_config.stage_a.optimizer.w0_learning_rate
        predictor_lr = state_lr
    else:
        qwen_lr = float(training_args.learning_rate)
        optimizer = backbone.project_config.stage_c.optimizer
        state_lr = optimizer.state_learning_rate
        w0_lr = optimizer.w0_learning_rate
        predictor_lr = optimizer.predictor_learning_rate

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        groups: dict[str, list[nn.Parameter]] = {
            "qwen": [],
            "state": [],
            "w0": [],
            "predictor": [],
        }
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise ValueError("outer optimizer encountered an aliased trainable parameter")
            seen.add(parameter_id)
            lowered = name.casefold()
            if "transient_w_t" in lowered or lowered.endswith(("w_t_1", "w_t_2")):
                raise ValueError("transient W_t cannot enter the Outer optimizer")
            if parameter_id in qwen_ids:
                group = "qwen"
            elif "predictor" in lowered:
                group = "predictor"
            elif lowered.endswith(("w0_1", "w0_2")) or "meta_fast" in lowered:
                group = "w0"
            else:
                group = "state"
            groups[group].append(parameter)
        if not groups["qwen"] or not groups["state"] or not groups["w0"]:
            raise ValueError("Outer AdamW requires non-empty Qwen/state/W0 groups")
        if stage is ProductionStage.A2 and groups["predictor"]:
            raise ValueError("A2 Outer AdamW cannot own Predictor")
        if stage is ProductionStage.A5 and not groups["predictor"]:
            raise ValueError("A5 Outer AdamW must own Predictor")
        learning_rates = {
            "qwen": qwen_lr,
            "state": state_lr,
            "w0": w0_lr,
            "predictor": predictor_lr,
        }
        parameter_groups = [
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
