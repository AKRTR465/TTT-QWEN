"""Bounded learned Inner-SGD step size used only by the explicit A5 ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import FastTTTStepControllerConfig


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


__all__ = [
    "InnerStepController",
    "StepControllerAudit",
    "build_inner_step_controller",
]
