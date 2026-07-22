"""Per-optimizer-group Outer gradient clipping for A2/A5 DeepSpeed training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import OuterGradientControlConfig

_SUPPORTED_DEEPSPEED_VERSION = "0.18.8"


@dataclass(frozen=True, slots=True)
class GroupGradientAudit:
    name: str
    learning_rate: float
    max_norm: float
    pre_clip_norm: float
    post_clip_norm: float
    clip_coefficient: float
    rms: float
    max_abs: float
    active_elements: int
    nonfinite_elements: int

    @property
    def clipped(self) -> bool:
        return self.clip_coefficient < 1.0


@dataclass(frozen=True, slots=True)
class OuterGradientAudit:
    attempted_update_count: int
    successful_update_count: int
    skipped_update_count: int
    within_initial_audit_window: bool
    skipped_nonfinite: bool
    groups: tuple[GroupGradientAudit, ...]

    def metrics(self) -> tuple[tuple[str, float], ...]:
        values: list[tuple[str, float]] = [
            ("outer_grad/attempted_updates", float(self.attempted_update_count)),
            ("outer_grad/successful_updates", float(self.successful_update_count)),
            ("outer_grad/skipped_updates", float(self.skipped_update_count)),
            ("outer_grad/nonfinite_skip", float(self.skipped_nonfinite)),
            ("outer_grad/initial_audit_window", float(self.within_initial_audit_window)),
        ]
        for group in self.groups:
            prefix = f"outer_grad/{group.name}"
            values.extend(
                (
                    (f"{prefix}/pre_norm", group.pre_clip_norm),
                    (f"{prefix}/post_norm", group.post_clip_norm),
                    (f"{prefix}/clip_coefficient", group.clip_coefficient),
                    (f"{prefix}/clipped", float(group.clipped)),
                    (f"{prefix}/rms", group.rms),
                    (f"{prefix}/max_abs", group.max_abs),
                    (f"{prefix}/lr", group.learning_rate),
                    (f"{prefix}/lr_x_pre_norm", group.learning_rate * group.pre_clip_norm),
                    (f"{prefix}/lr_x_post_norm", group.learning_rate * group.post_clip_norm),
                    (f"{prefix}/active_elements", float(group.active_elements)),
                    (f"{prefix}/nonfinite_elements", float(group.nonfinite_elements)),
                )
            )
        return tuple(values)


class OuterGradientController:
    """Clip named optimizer groups without allowing one group to scale another."""

    def __init__(
        self, config: OuterGradientControlConfig, *, expected_groups: tuple[str, ...]
    ) -> None:
        if not expected_groups or len(set(expected_groups)) != len(expected_groups):
            raise ValueError("Outer gradient groups must be unique and non-empty")
        self.config = config
        self.expected_groups = expected_groups
        self.attempted_update_count = 0
        self.successful_update_count = 0
        self.skipped_update_count = 0
        self.last_audit: OuterGradientAudit | None = None

    def apply_deepspeed(self, optimizer: object) -> OuterGradientAudit:
        """Scale DeepSpeed ZeRO-1/2 partition gradients immediately before engine.step()."""

        try:
            installed = version("deepspeed")
        except PackageNotFoundError as error:  # pragma: no cover - production-only dependency
            raise RuntimeError(
                "DeepSpeed is required for production Outer gradient control"
            ) from error
        if installed != _SUPPORTED_DEEPSPEED_VERSION:
            raise RuntimeError(
                "Outer gradient control is pinned to DeepSpeed "
                f"{_SUPPORTED_DEEPSPEED_VERSION}; found {installed}"
            )
        zero = cast(Any, optimizer)
        required = (
            "optimizer",
            "averaged_gradients",
            "params_in_partition",
            "get_grad_norm_direct",
            "has_overflow",
            "loss_scale",
            "partition_gradients",
        )
        if any(not hasattr(zero, name) for name in required):
            raise TypeError("production Outer clipping requires a DeepSpeed ZeRO-1/2 optimizer")
        if float(getattr(zero, "clip_grad", 0.0)) != 0.0:
            raise ValueError("DeepSpeed global gradient clipping must be disabled")
        base_optimizer = zero.optimizer
        groups = self._validate_param_groups(base_optimizer.param_groups)
        averaged = zero.averaged_gradients
        if len(groups) != len(zero.params_in_partition):
            raise RuntimeError("DeepSpeed gradient partitions drifted from optimizer groups")
        if any(index not in averaged or averaged[index] is None for index in range(len(groups))):
            raise RuntimeError(
                "DeepSpeed gradient partitions are unavailable at the update boundary"
            )

        self.attempted_update_count += 1
        group_nonfinite = tuple(
            self._distributed_nonfinite_count(
                cast(list[Tensor], averaged[index]), self._process_group(zero, index)
            )
            for index in range(len(groups))
        )
        overflow = any(value > 0 for value in group_nonfinite)
        if overflow != bool(zero.has_overflow(partition_gradients=zero.partition_gradients)):
            raise RuntimeError("DeepSpeed overflow detection disagreed with group gradient audit")
        if overflow:
            self.skipped_update_count += 1
            nonfinite_audits = tuple(
                self._nonfinite_group_audit(group, count)
                for group, count in zip(groups, group_nonfinite, strict=True)
            )
            return self._record(nonfinite_audits, skipped_nonfinite=True)

        loss_scale = float(zero.loss_scale)
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise RuntimeError("DeepSpeed exposed an invalid loss scale")
        group_audits: list[GroupGradientAudit] = []
        for index, (name, group) in enumerate(groups):
            gradients = cast(list[Tensor], averaged[index])
            params = zero.params_in_partition[index]
            scaled_norm = zero.get_grad_norm_direct(gradients, params)
            pre_norm = float(scaled_norm.detach().float().item()) / loss_scale
            max_norm = self._max_norm(name)
            coefficient = self._clip_coefficient(pre_norm, max_norm)
            active_elements, max_abs = self._distributed_shape_and_max(
                gradients, self._process_group(zero, index), loss_scale
            )
            for gradient in gradients:
                gradient.mul_(coefficient)
            group_audits.append(
                GroupGradientAudit(
                    name=name,
                    learning_rate=float(group["lr"]),
                    max_norm=max_norm,
                    pre_clip_norm=pre_norm,
                    post_clip_norm=pre_norm * coefficient,
                    clip_coefficient=coefficient,
                    rms=(pre_norm / math.sqrt(active_elements) if active_elements else 0.0),
                    max_abs=max_abs,
                    active_elements=active_elements,
                    nonfinite_elements=0,
                )
            )
        self.successful_update_count += 1
        return self._record(tuple(group_audits), skipped_nonfinite=False)

    def _validate_param_groups(
        self, param_groups: list[dict[str, Any]]
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        groups: list[tuple[str, dict[str, Any]]] = []
        for group in param_groups:
            name = group.get("group_name")
            if not isinstance(name, str) or not name:
                raise ValueError("every Outer optimizer group requires group_name")
            groups.append((name, group))
        actual = tuple(name for name, _ in groups)
        if actual != self.expected_groups:
            raise ValueError(
                f"Outer optimizer groups must be {self.expected_groups}, found {actual}"
            )
        return tuple(groups)

    def _max_norm(self, name: str) -> float:
        return float(getattr(self.config.max_grad_norm, name))

    @staticmethod
    def _clip_coefficient(pre_norm: float, max_norm: float) -> float:
        if pre_norm <= max_norm * (1.0 + 1.0e-6):
            return 1.0
        return max_norm / max(pre_norm, float(torch.finfo(torch.float32).tiny))

    def _record(
        self, groups: tuple[GroupGradientAudit, ...], *, skipped_nonfinite: bool
    ) -> OuterGradientAudit:
        audit = OuterGradientAudit(
            attempted_update_count=self.attempted_update_count,
            successful_update_count=self.successful_update_count,
            skipped_update_count=self.skipped_update_count,
            within_initial_audit_window=(
                self.successful_update_count <= self.config.audit_steps
            ),
            skipped_nonfinite=skipped_nonfinite,
            groups=groups,
        )
        self.last_audit = audit
        return audit

    @staticmethod
    def _nonfinite_group_audit(
        group: tuple[str, dict[str, Any]], count: int
    ) -> GroupGradientAudit:
        name, values = group
        return GroupGradientAudit(
            name=name,
            learning_rate=float(values["lr"]),
            max_norm=0.0,
            pre_clip_norm=math.nan,
            post_clip_norm=math.nan,
            clip_coefficient=0.0,
            rms=math.nan,
            max_abs=math.nan,
            active_elements=0,
            nonfinite_elements=count,
        )

    @staticmethod
    def _process_group(zero: Any, index: int) -> object | None:
        groups = getattr(zero, "real_dp_process_group", None)
        return groups[index] if groups is not None else getattr(zero, "dp_process_group", None)

    @staticmethod
    def _distributed_nonfinite_count(gradients: list[Tensor], group: object | None) -> int:
        device = gradients[0].device if gradients else torch.device("cpu")
        count = torch.tensor(
            sum(int((~torch.isfinite(value.detach())).sum().item()) for value in gradients),
            dtype=torch.int64,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM, group=group)
        return int(count.item())

    @staticmethod
    def _distributed_shape_and_max(
        gradients: list[Tensor], group: object | None, loss_scale: float
    ) -> tuple[int, float]:
        device = gradients[0].device if gradients else torch.device("cpu")
        count = torch.tensor(
            sum(value.numel() for value in gradients), dtype=torch.int64, device=device
        )
        maximum = torch.tensor(
            max((float(value.detach().abs().max().item()) for value in gradients), default=0.0),
            dtype=torch.float32,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM, group=group)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        return int(count.item()), float(maximum.item()) / loss_scale


__all__ = [
    "GroupGradientAudit",
    "OuterGradientAudit",
    "OuterGradientController",
]
