"""Bounded learned Inner-SGD step size used only by the explicit A5 ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import (
    STEP_CONTROLLER_FEATURE_CONTRACT,
    FastTTTStepControllerConfig,
)
from ttt_svcbench_qwen.losses import TTTLossOutput

STEP_CONTROLLER_FEATURE_CONTRACT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CausalSupportPosition:
    """Deployable causal position shared by Meta-TTT training and online inference."""

    support_index: int
    support_count: int
    truncation_horizon: int

    def __post_init__(self) -> None:
        values = (self.support_index, self.support_count, self.truncation_horizon)
        if any(type(value) is not int for value in values):
            raise TypeError("causal Support positions must be exact integers")
        if (
            self.support_index < 0
            or self.support_count <= 0
            or self.support_index >= self.support_count
            or self.truncation_horizon <= 0
        ):
            raise ValueError("causal Support position must lie inside a positive sequence")


@dataclass(frozen=True, slots=True)
class StepControllerAudit:
    """Detached per-Support step-size evidence."""

    values: tuple[float, ...]
    saturation_low_count: int
    saturation_high_count: int

    def __post_init__(self) -> None:
        if not self.values or any(
            not math.isfinite(value) or not 0.0 < value < 3.0e-4 for value in self.values
        ):
            raise ValueError("learned step sizes must be finite and strictly bounded")
        counts = (self.saturation_low_count, self.saturation_high_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("step-size saturation counts must be non-negative integers")
        if any(value > len(self.values) for value in counts):
            raise ValueError("step-size saturation count exceeds its batch")


class InnerStepController(nn.Module):  # type: ignore[misc]
    """Map seven detached causal features to ``0 < alpha_t < 3e-4``."""

    def __init__(self, config: FastTTTStepControllerConfig) -> None:
        super().__init__()
        if not isinstance(config, FastTTTStepControllerConfig):
            raise TypeError("InnerStepController requires validated controller config")
        self.input = nn.Linear(config.input_dim, config.hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(config.hidden_dim, 1)
        self.maximum_step_size = float(config.maximum_step_size)
        self.initial_step_size = float(config.initial_step_size)
        if config.feature_contract != STEP_CONTROLLER_FEATURE_CONTRACT:
            raise ValueError("InnerStepController feature contract is incompatible")
        self.register_buffer(
            "feature_contract_version",
            torch.tensor(STEP_CONTROLLER_FEATURE_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        nn.init.zeros_(self.output.weight)
        initial_probability = self.initial_step_size / self.maximum_step_size
        nn.init.constant_(
            self.output.bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError("InnerStepController features must be a Tensor")
        if features.ndim != 2 or features.shape[1] != self.input.in_features:
            raise ValueError("InnerStepController requires [B, 7] features")
        if features.requires_grad or features.grad_fn is not None:
            raise ValueError("InnerStepController features must be detached")
        if not torch.is_floating_point(features) or not bool(torch.isfinite(features).all()):
            raise ValueError("InnerStepController features must be finite floating values")
        logits = self.output(self.activation(self.input(features)))
        step_sizes = self.maximum_step_size * torch.sigmoid(logits.squeeze(-1))
        if step_sizes.ndim != 1 or step_sizes.shape[0] != features.shape[0]:
            raise RuntimeError("InnerStepController output shape drifted")
        if not bool(torch.isfinite(step_sizes.detach()).all()):
            raise ValueError("InnerStepController produced non-finite step sizes")
        if bool(
            torch.any(step_sizes.detach() <= 0.0)
            or torch.any(step_sizes.detach() >= self.maximum_step_size)
        ):
            raise ValueError("InnerStepController step sizes left their open bound")
        return step_sizes

    def audit(self, step_sizes: Tensor) -> StepControllerAudit:
        if step_sizes.ndim != 1:
            raise ValueError("step-size audit requires one scalar per batch row")
        detached = step_sizes.detach().float()
        values = tuple(float(value) for value in detached.cpu().tolist())
        low = int((detached <= 0.05 * self.maximum_step_size).sum().item())
        high = int((detached >= 0.95 * self.maximum_step_size).sum().item())
        return StepControllerAudit(
            values=values,
            saturation_low_count=low,
            saturation_high_count=high,
        )


def build_inner_step_controller(
    config: FastTTTStepControllerConfig,
) -> InnerStepController | None:
    """Return no module for the fixed ablation so its checkpoint topology stays unchanged."""

    if config.mode == "fixed":
        return None
    if config.mode != "learned":  # pragma: no cover - rejected by Pydantic
        raise ValueError("unknown InnerStepController mode")
    return InnerStepController(config)


def build_step_controller_features(
    *,
    ttt_output: TTTLossOutput,
    start_time: float,
    end_time: float,
    previous_end_time: float,
    position: CausalSupportPosition,
    controller: InnerStepController,
) -> Tensor:
    """Build the shared seven detached causal features for training and inference."""

    times = (start_time, end_time, previous_end_time)
    if any(not math.isfinite(value) or value < 0.0 for value in times):
        raise ValueError("step controller times must be finite and non-negative")
    if start_time > end_time or previous_end_time > end_time:
        raise ValueError("step controller received a future causal boundary")
    if not isinstance(position, CausalSupportPosition):
        raise TypeError("step controller requires a CausalSupportPosition")
    per_row = ttt_output.per_row_total.detach().float()
    batch_size = per_row.shape[0]
    device = per_row.device
    dtype = next(controller.parameters()).dtype

    def repeated(value: float) -> Tensor:
        return torch.full((batch_size,), value, dtype=torch.float32, device=device)

    scale = ttt_output.temporal_scale_audit
    pair_count = scale.pair_element_count.detach().float().clamp_min(1.0)
    target_rms = torch.sqrt(scale.target_sum_squares.detach().float() / pair_count)
    error_rms = torch.sqrt(scale.error_sum_squares.detach().float() / pair_count)
    valid_ratio = (
        (ttt_output.pred.valid_counts.detach() > 0).float()
        + (ttt_output.identity.valid_counts.detach() > 0).float()
        + (ttt_output.event.valid_counts.detach() > 0).float()
    ) / 3.0
    features = torch.stack(
        (
            repeated(
                ((position.support_index % position.truncation_horizon) + 1)
                / position.truncation_horizon
            ),
            repeated((position.support_index + 1) / position.support_count),
            torch.log1p(repeated(end_time - start_time)),
            torch.log1p(repeated(end_time - previous_end_time)),
            torch.log1p(per_row.clamp_min(0.0)),
            valid_ratio,
            torch.log1p((target_rms + error_rms).expand(batch_size)),
        ),
        dim=1,
    )
    if features.shape != (batch_size, 7):
        raise RuntimeError("step controller feature topology drifted")
    return features.detach().to(dtype=dtype)


__all__ = [
    "CausalSupportPosition",
    "InnerStepController",
    "STEP_CONTROLLER_FEATURE_CONTRACT_VERSION",
    "StepControllerAudit",
    "build_inner_step_controller",
    "build_step_controller_features",
]
