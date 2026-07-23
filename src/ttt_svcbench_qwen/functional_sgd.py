"""Functional one-step SGD for the two explicit Fast-TTT runtime matrices.

Inputs: one scalar support loss, one :class:`FastWeightsState`, and the frozen
SGD/runtime contract.
Outputs: a new next-chunk fast generation plus finite gradient/update audits.
Forbidden: arbitrary parameter lists, in-place mutation, Bank/FSM logic,
momentum, weight decay, multi-step updates, or silent meta-gradient detaches.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import InnerSGDConfig
from ttt_svcbench_qwen.fast_ttt import FastWeightsState, OptimizerRuntimeState
from ttt_svcbench_qwen.losses import LossSkipReason, LossTerm, TTTLossOutput
from ttt_svcbench_qwen.tensor_contracts import tensor_storage_key


class UpdateSkipReason(StrEnum):
    NO_VALID_TERM = "no_valid_term"
    INSUFFICIENT_TIME = "insufficient_time"
    NONFINITE_LOSS = "nonfinite_loss"
    NONFINITE_GRADIENT = "nonfinite_gradient"
    ZERO_GRADIENT = "zero_gradient"
    UNREPRESENTABLE_UPDATE = "unrepresentable_update"
    INVALID_AFTER_CLIP = "invalid_after_clip"


class GradientMode(StrEnum):
    """Exact autograd contract used by one functional inner update."""

    ONLINE_LEAF = "online_leaf"
    META_FULL_SECOND_ORDER = "meta_full_second_order"


@dataclass(frozen=True, slots=True)
class GradientDeltaSnapshot:
    """Detached before-values and the expected boundary for one named group."""

    name: str
    parameter_count: int
    gradient_expected: bool
    update_allowed: bool
    before_values: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("gradient audit group name must be non-empty")
        if type(self.parameter_count) is not int or self.parameter_count < 0:
            raise ValueError("gradient audit parameter_count must be a non-negative integer")
        if type(self.gradient_expected) is not bool or type(self.update_allowed) is not bool:
            raise TypeError("gradient audit boundary flags must be bool")
        actual_count = sum(value.numel() for value in self.before_values)
        if self.parameter_count != actual_count:
            raise ValueError("gradient audit parameter_count does not match before-values")
        if self.gradient_expected and not self.before_values:
            raise ValueError("a zero-parameter group cannot expect gradients")
        if self.update_allowed and not self.before_values:
            raise ValueError("a zero-parameter group cannot allow updates")
        if any(value.requires_grad for value in self.before_values):
            raise ValueError("gradient audit snapshots must be detached")
        if any(
            value.device.type == "meta" or not torch.is_floating_point(value)
            for value in self.before_values
        ):
            raise ValueError("gradient audit snapshots require materialized floating tensors")
        if any(not bool(torch.isfinite(value).all()) for value in self.before_values):
            raise ValueError("gradient audit snapshots must be finite")


@dataclass(frozen=True, slots=True)
class GradientDeltaAudit:
    """Finite group-level gradient and parameter-delta evidence."""

    name: str
    parameter_count: int
    gradient_present: bool
    gradient_norm: float
    delta_norm: float
    gradient_expected: bool
    update_allowed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("gradient audit group name must be non-empty")
        if type(self.parameter_count) is not int or self.parameter_count < 0:
            raise ValueError("gradient audit parameter_count must be a non-negative integer")
        flags = (self.gradient_present, self.gradient_expected, self.update_allowed)
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("gradient audit flags must be bool")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.gradient_norm, self.delta_norm)
        ):
            raise ValueError("gradient and delta norms must be finite and non-negative")
        if self.gradient_expected and not self.gradient_present:
            raise ValueError(f"expected gradient is missing for group {self.name!r}")
        if self.gradient_expected and self.gradient_norm <= 0.0:
            raise ValueError(f"expected gradient has zero norm for group {self.name!r}")
        if not self.gradient_expected and self.gradient_present:
            raise ValueError(f"forbidden gradient appeared for group {self.name!r}")
        if not self.update_allowed and self.delta_norm != 0.0:
            raise ValueError(f"forbidden parameter delta appeared for group {self.name!r}")
        if self.update_allowed and self.delta_norm <= 0.0:
            raise ValueError(f"allowed update did not occur for group {self.name!r}")


@dataclass(frozen=True, slots=True)
class FunctionalSGDResult:
    fast_state: FastWeightsState
    optimizer_state: OptimizerRuntimeState
    did_update: bool
    valid_term_count: int
    gradient_norm: float | None
    clipped_gradient_norm: float | None
    per_matrix_gradient_norms: tuple[float, float] | None
    per_matrix_clipped_norms: tuple[float, float] | None
    update_norm: float
    skip_reason: UpdateSkipReason | None
    skip_detail: str | None
    gradient_mode: GradientMode

    def __post_init__(self) -> None:
        if type(self.did_update) is not bool:
            raise TypeError("functional SGD update flag must be bool")
        if not isinstance(self.gradient_mode, GradientMode):
            raise TypeError("functional SGD gradient_mode must be GradientMode")
        if type(self.valid_term_count) is not int or self.valid_term_count < 0:
            raise ValueError("valid_term_count must be a non-negative exact integer")
        if not math.isfinite(self.update_norm) or self.update_norm < 0.0:
            raise ValueError("update_norm must be finite and non-negative")
        scalars = (self.gradient_norm, self.clipped_gradient_norm)
        pairs = (self.per_matrix_gradient_norms, self.per_matrix_clipped_norms)
        for value in scalars:
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError("reported gradient norms must be finite and non-negative")
        for values in pairs:
            if values is not None and any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise ValueError("per-matrix norms must be finite and non-negative")
        if self.did_update:
            if self.skip_reason is not None or self.skip_detail is not None:
                raise ValueError("successful updates cannot carry skip metadata")
            if self.valid_term_count == 0:
                raise ValueError("successful updates require at least one valid term")
            if any(value is None for value in scalars + pairs) or self.update_norm <= 0.0:
                raise ValueError("successful updates require complete positive finite audits")
            if self.optimizer_state.last_skip_reason is not None:
                raise ValueError("successful optimizer state cannot retain a skip reason")
        else:
            if self.skip_reason is None or not self.skip_detail:
                raise ValueError("skipped updates require a reason and detail")
            if self.update_norm != 0.0:
                raise ValueError("skipped updates must report zero update norm")
            if self.optimizer_state.last_skip_reason != self.skip_reason.value:
                raise ValueError("optimizer and result skip reasons must agree")
        mode_is_differentiable = self.gradient_mode is GradientMode.META_FULL_SECOND_ORDER
        if mode_is_differentiable is not self.fast_state.differentiable:
            raise ValueError("result gradient mode must match FastWeightsState")
        expected_attempts = self.fast_state.update_count + self.fast_state.skip_count
        if self.optimizer_state.attempted_update_count != expected_attempts:
            raise ValueError("optimizer attempts must equal accepted updates plus skips")


def initialize_optimizer_state(config: InnerSGDConfig) -> OptimizerRuntimeState:
    """Create empty runtime counters from the frozen stateless-SGD config."""

    _validate_optimizer_config(config)
    return OptimizerRuntimeState(
        optimizer_name=config.name,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        steps_per_chunk=config.steps_per_chunk,
        grad_clip_norm=config.grad_clip_norm,
        attempted_update_count=0,
        last_skip_reason=None,
    )


def reset_optimizer_state(config: InnerSGDConfig) -> OptimizerRuntimeState:
    """Reset the momentum-free optimizer audit for a fresh video."""

    return initialize_optimizer_state(config)


def snapshot_gradient_delta_group(
    *,
    name: str,
    parameters: Iterable[Tensor],
    gradient_expected: bool,
    update_allowed: bool,
) -> GradientDeltaSnapshot:
    """Capture a detached generation before backward or an optimizer action."""

    values = tuple(parameters)
    if any(not isinstance(value, Tensor) for value in values):
        raise TypeError("gradient audit groups may contain only tensors")
    if len({id(value) for value in values}) != len(values):
        raise ValueError("gradient audit groups cannot contain duplicate tensors")
    before_values = tuple(value.detach().clone() for value in values)
    return GradientDeltaSnapshot(
        name=name,
        parameter_count=sum(value.numel() for value in values),
        gradient_expected=gradient_expected,
        update_allowed=update_allowed,
        before_values=before_values,
    )


def audit_gradient_delta_group(
    snapshot: GradientDeltaSnapshot,
    *,
    parameters: Iterable[Tensor],
    gradients: Iterable[Tensor | None] | None = None,
) -> GradientDeltaAudit:
    """Compare one current generation with its snapshot and fail closed."""

    if not isinstance(snapshot, GradientDeltaSnapshot):
        raise TypeError("gradient/delta audit requires GradientDeltaSnapshot")
    current = tuple(parameters)
    if any(not isinstance(value, Tensor) for value in current):
        raise TypeError("gradient audit groups may contain only tensors")
    if len(current) != len(snapshot.before_values):
        raise ValueError("gradient audit parameter tensor count changed after snapshot")
    if len({id(value) for value in current}) != len(current):
        raise ValueError("gradient audit groups cannot contain duplicate tensors")
    for before, after in zip(snapshot.before_values, current, strict=True):
        if (
            after.shape != before.shape
            or after.dtype != before.dtype
            or after.device != before.device
        ):
            raise ValueError("gradient audit parameter metadata changed after snapshot")
        if after.device.type == "meta" or not torch.is_floating_point(after):
            raise ValueError("gradient audit requires materialized floating tensors")
        if not bool(torch.isfinite(after.detach()).all()):
            raise ValueError("gradient audit current parameters must be finite")

    observed_gradients = (
        tuple(value.grad for value in current) if gradients is None else tuple(gradients)
    )
    if len(observed_gradients) != len(current):
        raise ValueError("gradient audit gradients must align with parameters")
    for parameter, gradient in zip(current, observed_gradients, strict=True):
        if gradient is None:
            continue
        if not isinstance(gradient, Tensor) or gradient.shape != parameter.shape:
            raise ValueError("gradient audit gradient shapes must match parameters")
        if gradient.device != parameter.device or not torch.is_floating_point(gradient):
            raise ValueError("gradient audit gradients must share parameter device and be floating")
        if not bool(torch.isfinite(gradient.detach()).all()):
            raise ValueError("gradient audit gradients must be finite")

    gradient_present = any(gradient is not None for gradient in observed_gradients)
    if snapshot.gradient_expected and any(gradient is None for gradient in observed_gradients):
        raise ValueError(f"expected gradient is missing for group {snapshot.name!r}")
    gradient_values = tuple(gradient for gradient in observed_gradients if gradient is not None)
    deltas = tuple(
        after.detach().float() - before.float()
        for before, after in zip(snapshot.before_values, current, strict=True)
    )
    gradient_norm = _sequence_norm_float(gradient_values)
    delta_norm = _sequence_norm_float(deltas)
    return GradientDeltaAudit(
        name=snapshot.name,
        parameter_count=snapshot.parameter_count,
        gradient_present=gradient_present,
        gradient_norm=gradient_norm,
        delta_norm=delta_norm,
        gradient_expected=snapshot.gradient_expected,
        update_allowed=snapshot.update_allowed,
    )


def functional_sgd_step_from_ttt_row(
    *,
    ttt_output: TTTLossOutput,
    row: int,
    fast_state: FastWeightsState,
    optimizer_config: InnerSGDConfig,
    optimizer_state: OptimizerRuntimeState,
    _retain_graph: bool = False,
) -> FunctionalSGDResult:
    """Update exactly one video's state from its authoritative TTT row."""

    batch_size = _validate_ttt_output_for_sgd(ttt_output)
    if type(_retain_graph) is not bool:
        raise TypeError("TTT row retain-graph control must be bool")
    if type(row) is not int or row < 0 or row >= batch_size:
        raise IndexError("TTT row must be an exact in-range batch index")
    valid_term_count = sum(
        int(term.valid_counts[row].item())
        for term in (ttt_output.pred, ttt_output.identity, ttt_output.event)
    )
    row_is_valid = bool(ttt_output.update_valid_mask[row].item())
    if row_is_valid != (valid_term_count > 0):
        raise ValueError("TTT row validity must agree with its valid support count")
    if row_is_valid:
        return functional_sgd_step(
            loss=ttt_output.per_row_total[row],
            fast_state=fast_state,
            optimizer_config=optimizer_config,
            optimizer_state=optimizer_state,
            valid_term_count=valid_term_count,
            _retain_graph=_retain_graph,
        )
    return functional_sgd_step(
        loss=None,
        fast_state=fast_state,
        optimizer_config=optimizer_config,
        optimizer_state=optimizer_state,
        valid_term_count=0,
        invalid_reason=_invalid_ttt_row_reason(ttt_output, row),
    )


def functional_sgd_steps_from_ttt(
    *,
    ttt_output: TTTLossOutput,
    fast_states: Sequence[FastWeightsState],
    optimizer_config: InnerSGDConfig,
    optimizer_states: Sequence[OptimizerRuntimeState],
) -> tuple[FunctionalSGDResult, ...]:
    """Apply independent row losses to a storage-isolated batch of video states."""

    batch_size = _validate_ttt_output_for_sgd(ttt_output)
    states = tuple(fast_states)
    runtimes = tuple(optimizer_states)
    if len(states) != batch_size or len(runtimes) != batch_size:
        raise ValueError("TTT batch, fast states, and optimizer states must have identical B")
    if not all(isinstance(state, FastWeightsState) for state in states):
        raise TypeError("TTT batch bridge requires only FastWeightsState values")
    if not all(isinstance(state, OptimizerRuntimeState) for state in runtimes):
        raise TypeError("TTT batch bridge requires only OptimizerRuntimeState values")
    fast_values = tuple(value for state in states for value in state.fast_parameters)
    if len({tensor_storage_key(value) for value in fast_values}) != len(fast_values):
        raise ValueError("batched fast states must use storage-isolated W_t tensors")
    valid_rows = tuple(
        row for row in range(batch_size) if bool(ttt_output.update_valid_mask[row].item())
    )
    last_valid_row = valid_rows[-1] if valid_rows else None
    results: list[FunctionalSGDResult] = []
    for row in range(batch_size):
        results.append(
            functional_sgd_step_from_ttt_row(
                ttt_output=ttt_output,
                row=row,
                fast_state=states[row],
                optimizer_config=optimizer_config,
                optimizer_state=runtimes[row],
                _retain_graph=row in valid_rows and row != last_valid_row,
            )
        )
    return tuple(results)


def functional_sgd_step(
    *,
    loss: Tensor | None,
    fast_state: FastWeightsState,
    optimizer_config: InnerSGDConfig,
    optimizer_state: OptimizerRuntimeState,
    valid_term_count: int,
    invalid_reason: UpdateSkipReason | None = None,
    _retain_graph: bool = False,
) -> FunctionalSGDResult:
    """Return one new fast generation without mutating the current chunk state.

    The API intentionally has no arbitrary ``parameters`` argument: the only
    differentiable inner parameters are ``fast_state.fast_parameters`` in their
    frozen ``(w_t_1, w_t_2)`` order.
    """

    if not isinstance(fast_state, FastWeightsState):
        raise TypeError("functional SGD requires one FastWeightsState")
    if type(_retain_graph) is not bool:
        raise TypeError("functional SGD retain-graph control must be bool")
    _validate_optimizer_config(optimizer_config)
    _validate_optimizer_runtime(optimizer_config, optimizer_state, fast_state)
    if type(valid_term_count) is not int or valid_term_count < 0:
        raise ValueError("valid_term_count must be a non-negative exact integer")

    if valid_term_count == 0:
        if loss is not None:
            raise ValueError("invalid TTT terms must not provide a fabricated scalar loss")
        if invalid_reason not in (
            UpdateSkipReason.NO_VALID_TERM,
            UpdateSkipReason.INSUFFICIENT_TIME,
        ):
            raise ValueError("invalid terms require no_valid_term or insufficient_time")
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=invalid_reason,
            detail=invalid_reason.value,
            valid_term_count=0,
        )
    if invalid_reason is not None:
        raise ValueError("valid TTT terms cannot carry a caller-supplied skip reason")
    if not isinstance(loss, Tensor):
        raise TypeError("valid TTT terms require a scalar Tensor loss")
    _validate_loss(loss, fast_state)
    if not bool(torch.isfinite(loss.detach()).item()):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.NONFINITE_LOSS,
            detail="loss_is_not_finite",
            valid_term_count=valid_term_count,
        )

    gradient_mode = _gradient_mode(fast_state, optimizer_config)
    parameters = fast_state.fast_parameters
    if not fast_state.differentiable and any(
        parameter.grad is not None for parameter in parameters
    ):
        raise ValueError("clear stale online fast gradients before functional SGD")
    gradients_raw = torch.autograd.grad(
        loss,
        parameters,
        create_graph=gradient_mode is GradientMode.META_FULL_SECOND_ORDER,
        retain_graph=(gradient_mode is GradientMode.META_FULL_SECOND_ORDER or _retain_graph),
        allow_unused=True,
    )
    if any(gradient is None for gradient in gradients_raw):
        raise ValueError("valid TTT loss must connect to both fast matrices")
    gradients = (cast_tensor(gradients_raw[0]), cast_tensor(gradients_raw[1]))
    if any(not bool(torch.isfinite(gradient.detach()).all()) for gradient in gradients):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.NONFINITE_GRADIENT,
            detail="one_or_more_fast_gradients_are_not_finite",
            valid_term_count=valid_term_count,
        )

    gradient_norm_tensor, per_matrix_norm_tensors = _global_norm(gradients)
    if not bool(torch.isfinite(gradient_norm_tensor.detach()).item()):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.NONFINITE_GRADIENT,
            detail="fp32_global_gradient_norm_is_not_finite",
            valid_term_count=valid_term_count,
        )
    gradient_norm = _audit_float(gradient_norm_tensor)
    per_matrix_norms = tuple(_audit_float(value) for value in per_matrix_norm_tensors)

    max_norm = gradient_norm_tensor.new_tensor(optimizer_config.grad_clip_norm)
    tiny = torch.finfo(gradient_norm_tensor.dtype).tiny
    clip_scale = torch.clamp(max_norm / gradient_norm_tensor.clamp_min(tiny), max=1.0)
    clipped = tuple(gradient * clip_scale.to(dtype=gradient.dtype) for gradient in gradients)
    if any(not bool(torch.isfinite(gradient.detach()).all()) for gradient in clipped):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.INVALID_AFTER_CLIP,
            detail="clipped_gradient_is_not_finite",
            valid_term_count=valid_term_count,
            gradient_norm=gradient_norm,
            per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
        )
    clipped_norm_tensor, clipped_per_matrix_tensors = _global_norm(clipped)
    if not bool(torch.isfinite(clipped_norm_tensor.detach()).item()):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.INVALID_AFTER_CLIP,
            detail="clipped_fp32_global_norm_is_not_finite",
            valid_term_count=valid_term_count,
            gradient_norm=gradient_norm,
            per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
        )
    clipped_norm = _audit_float(clipped_norm_tensor)
    clipped_per_matrix = tuple(_audit_float(value) for value in clipped_per_matrix_tensors)
    if clipped_norm <= 0.0:
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.ZERO_GRADIENT,
            detail="clipped_gradient_has_zero_global_norm",
            valid_term_count=valid_term_count,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
            per_matrix_clipped_norms=cast_norm_pair(clipped_per_matrix),
        )

    candidates = tuple(
        parameter - optimizer_config.learning_rate * gradient
        for parameter, gradient in zip(parameters, clipped, strict=True)
    )
    if any(not bool(torch.isfinite(candidate.detach()).all()) for candidate in candidates):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.INVALID_AFTER_CLIP,
            detail="candidate_fast_weight_is_not_finite",
            valid_term_count=valid_term_count,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
            per_matrix_clipped_norms=cast_norm_pair(clipped_per_matrix),
        )
    deltas = tuple(
        candidate - parameter for candidate, parameter in zip(candidates, parameters, strict=True)
    )
    update_norm_tensor, _ = _global_norm(deltas)
    update_norm = _audit_float(update_norm_tensor)
    if update_norm <= 0.0 or all(
        torch.equal(candidate.detach(), parameter.detach())
        for candidate, parameter in zip(candidates, parameters, strict=True)
    ):
        return _skip_result(
            fast_state,
            optimizer_state,
            reason=UpdateSkipReason.UNREPRESENTABLE_UPDATE,
            detail="update_not_representable_in_fast_dtype",
            valid_term_count=valid_term_count,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
            per_matrix_clipped_norms=cast_norm_pair(clipped_per_matrix),
        )

    next_parameters = _next_parameters(candidates, differentiable=fast_state.differentiable)
    next_state = FastWeightsState(
        w0_1=fast_state.w0_1,
        w0_2=fast_state.w0_2,
        w_t_1=next_parameters[0],
        w_t_2=next_parameters[1],
        fast_version=fast_state.fast_version + 1,
        update_count=fast_state.update_count + 1,
        skip_count=fast_state.skip_count,
        differentiable=fast_state.differentiable,
    )
    next_optimizer = _next_optimizer_state(optimizer_state, skip_reason=None)
    return FunctionalSGDResult(
        fast_state=next_state,
        optimizer_state=next_optimizer,
        did_update=True,
        valid_term_count=valid_term_count,
        gradient_norm=gradient_norm,
        clipped_gradient_norm=clipped_norm,
        per_matrix_gradient_norms=cast_norm_pair(per_matrix_norms),
        per_matrix_clipped_norms=cast_norm_pair(clipped_per_matrix),
        update_norm=update_norm,
        skip_reason=None,
        skip_detail=None,
        gradient_mode=gradient_mode,
    )


def _skip_result(
    fast_state: FastWeightsState,
    optimizer_state: OptimizerRuntimeState,
    *,
    reason: UpdateSkipReason,
    detail: str,
    valid_term_count: int,
    gradient_norm: float | None = None,
    clipped_gradient_norm: float | None = None,
    per_matrix_gradient_norms: tuple[float, float] | None = None,
    per_matrix_clipped_norms: tuple[float, float] | None = None,
) -> FunctionalSGDResult:
    next_parameters = _next_parameters(
        fast_state.fast_parameters,
        differentiable=fast_state.differentiable,
    )
    next_state = FastWeightsState(
        w0_1=fast_state.w0_1,
        w0_2=fast_state.w0_2,
        w_t_1=next_parameters[0],
        w_t_2=next_parameters[1],
        fast_version=fast_state.fast_version,
        update_count=fast_state.update_count,
        skip_count=fast_state.skip_count + 1,
        differentiable=fast_state.differentiable,
    )
    next_optimizer = _next_optimizer_state(optimizer_state, skip_reason=reason)
    return FunctionalSGDResult(
        fast_state=next_state,
        optimizer_state=next_optimizer,
        did_update=False,
        valid_term_count=valid_term_count,
        gradient_norm=gradient_norm,
        clipped_gradient_norm=clipped_gradient_norm,
        per_matrix_gradient_norms=per_matrix_gradient_norms,
        per_matrix_clipped_norms=per_matrix_clipped_norms,
        update_norm=0.0,
        skip_reason=reason,
        skip_detail=detail,
        gradient_mode=_gradient_mode_from_state(fast_state),
    )


def _next_parameters(
    values: tuple[Tensor, Tensor],
    *,
    differentiable: bool,
) -> tuple[Tensor, Tensor]:
    if differentiable:
        return (values[0].clone(), values[1].clone())
    return (
        values[0].detach().clone().requires_grad_(True),
        values[1].detach().clone().requires_grad_(True),
    )


def _global_norm(values: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    fp32_values = (values[0].float(), values[1].float())
    individual = tuple(torch.sqrt(value.square().sum(dtype=torch.float32)) for value in fp32_values)
    total = torch.sqrt(
        individual[0].square().to(torch.float32) + individual[1].square().to(torch.float32)
    )
    return total, cast_tensor_pair(individual)


def _sequence_norm_float(values: Sequence[Tensor]) -> float:
    if not values:
        return 0.0
    squared_sums = tuple(
        float(
            value.detach()
            .float()
            .square()
            .sum(dtype=torch.float32)
            .to(device="cpu", dtype=torch.float64)
            .item()
        )
        for value in values
    )
    norm = math.sqrt(math.fsum(squared_sums))
    if not math.isfinite(norm):
        raise ValueError("gradient audit norm accumulation must be finite")
    return norm


def _audit_float(value: Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float64).item())


def _next_optimizer_state(
    state: OptimizerRuntimeState,
    *,
    skip_reason: UpdateSkipReason | None,
) -> OptimizerRuntimeState:
    return OptimizerRuntimeState(
        optimizer_name=state.optimizer_name,
        learning_rate=state.learning_rate,
        momentum=state.momentum,
        weight_decay=state.weight_decay,
        steps_per_chunk=state.steps_per_chunk,
        grad_clip_norm=state.grad_clip_norm,
        attempted_update_count=state.attempted_update_count + 1,
        last_skip_reason=None if skip_reason is None else skip_reason.value,
    )


def _validate_ttt_output_for_sgd(ttt_output: TTTLossOutput) -> int:
    if not isinstance(ttt_output, TTTLossOutput):
        raise TypeError("TTT SGD bridge requires TTTLossOutput")
    batch_size = int(ttt_output.per_row_total.shape[0])
    if batch_size <= 0:
        raise ValueError("TTT SGD bridge requires a non-empty batch")
    terms: tuple[LossTerm, ...] = (
        ttt_output.pred,
        ttt_output.identity,
        ttt_output.e1_event,
        ttt_output.e2_event,
        ttt_output.event,
    )
    if any(term.per_row.shape != (batch_size,) for term in terms):
        raise ValueError("all TTT loss terms must share the output batch size")
    if any(term.per_row.device != ttt_output.per_row_total.device for term in terms):
        raise ValueError("all TTT loss rows must share one device")
    expected_event_rows = ttt_output.e1_event.per_row + ttt_output.e2_event.per_row
    expected_event_counts = ttt_output.e1_event.valid_counts + ttt_output.e2_event.valid_counts
    if not torch.allclose(
        ttt_output.event.per_row.detach(),
        expected_event_rows.detach(),
        atol=1.0e-6,
        rtol=1.0e-6,
    ) or not torch.equal(ttt_output.event.valid_counts, expected_event_counts):
        raise ValueError("TTT event rows/counts must equal E1 plus E2")
    expected_rows = (
        ttt_output.pred_weight * ttt_output.pred.per_row
        + ttt_output.identity_weight * ttt_output.identity.per_row
        + ttt_output.event_weight * ttt_output.event.per_row
    )
    if not torch.allclose(
        ttt_output.per_row_total.detach(),
        expected_rows.detach(),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise ValueError("TTT per-row total must equal its frozen weighted terms")
    expected_valid = (
        ttt_output.pred.row_valid_mask
        | ttt_output.identity.row_valid_mask
        | ttt_output.event.row_valid_mask
    )
    if not torch.equal(ttt_output.update_valid_mask, expected_valid):
        raise ValueError("TTT update-valid mask must be the union of weighted terms")
    return batch_size


def _invalid_ttt_row_reason(
    ttt_output: TTTLossOutput,
    row: int,
) -> UpdateSkipReason:
    pred_reason = ttt_output.pred.skip_reasons[row]
    if pred_reason is LossSkipReason.INSUFFICIENT_TIME:
        return UpdateSkipReason.INSUFFICIENT_TIME
    return UpdateSkipReason.NO_VALID_TERM


def _gradient_mode(
    fast_state: FastWeightsState,
    config: InnerSGDConfig,
) -> GradientMode:
    if config.meta_gradient_mode != "full_second_order":
        raise ValueError("inner SGD meta_gradient_mode must be 'full_second_order'")
    return _gradient_mode_from_state(fast_state)


def _gradient_mode_from_state(fast_state: FastWeightsState) -> GradientMode:
    return (
        GradientMode.META_FULL_SECOND_ORDER
        if fast_state.differentiable
        else GradientMode.ONLINE_LEAF
    )


def _validate_optimizer_config(config: InnerSGDConfig) -> None:
    if not isinstance(config, InnerSGDConfig):
        raise TypeError("functional SGD requires InnerSGDConfig")
    expected = {
        "name": "sgd",
        "learning_rate": 1.0e-4,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "steps_per_chunk": 1,
        "grad_clip_norm": 1.0,
        "reset_per_video": True,
        "meta_gradient_mode": "full_second_order",
    }
    for name, required in expected.items():
        if getattr(config, name) != required:
            raise ValueError(f"inner SGD {name} must be {required!r}")


def _validate_optimizer_runtime(
    config: InnerSGDConfig,
    optimizer: OptimizerRuntimeState,
    fast_state: FastWeightsState,
) -> None:
    if not isinstance(optimizer, OptimizerRuntimeState):
        raise TypeError("functional SGD requires OptimizerRuntimeState")
    actual = (
        optimizer.optimizer_name,
        optimizer.learning_rate,
        optimizer.momentum,
        optimizer.weight_decay,
        optimizer.steps_per_chunk,
        optimizer.grad_clip_norm,
    )
    expected = (
        config.name,
        config.learning_rate,
        config.momentum,
        config.weight_decay,
        config.steps_per_chunk,
        config.grad_clip_norm,
    )
    if actual != expected:
        raise ValueError("optimizer runtime does not match the frozen config")
    expected_attempts = fast_state.update_count + fast_state.skip_count
    if optimizer.attempted_update_count != expected_attempts:
        raise ValueError("optimizer attempts must equal accepted updates plus skips")
    if optimizer.last_skip_reason is not None and optimizer.last_skip_reason not in {
        reason.value for reason in UpdateSkipReason
    }:
        raise ValueError("optimizer runtime contains an unknown skip reason")


def _validate_loss(loss: Tensor, fast_state: FastWeightsState) -> None:
    if loss.ndim != 0 or not torch.is_floating_point(loss):
        raise ValueError("functional SGD loss must be a floating scalar")
    if loss.device != fast_state.w_t_1.device:
        raise ValueError("functional SGD loss and fast state must share one device")
    if not loss.requires_grad:
        raise ValueError("valid functional SGD loss must require gradients")


def cast_tensor(value: Tensor | None) -> Tensor:
    if value is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("missing fast gradient")
    return value


def cast_tensor_pair(values: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    if len(values) != 2:  # pragma: no cover - private fixed-arity invariant
        raise RuntimeError("functional SGD requires exactly two tensors")
    return (values[0], values[1])


def cast_norm_pair(values: tuple[float, ...]) -> tuple[float, float]:
    if len(values) != 2:  # pragma: no cover - private fixed-arity invariant
        raise RuntimeError("functional SGD requires exactly two norms")
    return (values[0], values[1])
