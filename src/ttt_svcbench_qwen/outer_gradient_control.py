"""Per-optimizer-group Outer gradient clipping for A2/A5 DeepSpeed training."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import OuterGradientControlConfig


@dataclass(frozen=True, slots=True)
class GroupGradientAudit:
    name: str
    max_norm: float
    pre_clip_norm: float
    post_clip_norm: float
    clip_coefficient: float

    @property
    def clipped(self) -> bool:
        return self.clip_coefficient < 1.0


@dataclass(frozen=True, slots=True)
class OuterGradientAudit:
    attempted_update_count: int
    successful_update_count: int
    skipped_update_count: int
    skipped_nonfinite: bool
    skipped_nonfinite_loss: bool
    nonfinite_loss_sources: tuple[str, ...]
    groups: tuple[GroupGradientAudit, ...]

    def group(self, name: str) -> GroupGradientAudit:
        """Return one named optimizer-group audit or fail closed on topology drift."""

        matches = tuple(group for group in self.groups if group.name == name)
        if len(matches) != 1:
            raise RuntimeError(
                f"Outer gradient audit requires exactly one {name!r} group; found {len(matches)}"
            )
        return matches[0]

    def metrics(self) -> tuple[tuple[str, float], ...]:
        values: list[tuple[str, float]] = [
            ("outer_grad/attempted_updates", float(self.attempted_update_count)),
            ("outer_grad/successful_updates", float(self.successful_update_count)),
            ("outer_grad/skipped_updates", float(self.skipped_update_count)),
            ("outer_grad/nonfinite_skip", float(self.skipped_nonfinite)),
            ("outer_grad/nonfinite_loss_skip", float(self.skipped_nonfinite_loss)),
        ]
        for group in self.groups:
            prefix = f"outer_grad/{group.name}"
            values.extend(
                (
                    (f"{prefix}/pre_norm", group.pre_clip_norm),
                    (f"{prefix}/post_norm", group.post_clip_norm),
                    (f"{prefix}/clip_coefficient", group.clip_coefficient),
                    (f"{prefix}/clipped", float(group.clipped)),
                )
            )
        return tuple(values)


class OuterGradientController:
    """Clip named optimizer groups without allowing one group to scale another."""

    def __init__(
        self,
        config: OuterGradientControlConfig,
        *,
        expected_groups: tuple[str, ...],
    ) -> None:
        self.config = config
        self.expected_groups = expected_groups
        self.attempted_update_count = 0
        self.successful_update_count = 0
        self.skipped_update_count = 0
        self.last_audit: OuterGradientAudit | None = None
        self._loss_nonfinite: Tensor | None = None
        self._loss_nonfinite_sources: dict[str, Tensor] = {}

    def record_loss(self, loss: Tensor, source: str) -> None:
        """Accumulate one device-side nonfinite flag until the next real update boundary."""

        flag = ~torch.isfinite(loss.detach())
        if self._loss_nonfinite is None:
            self._loss_nonfinite = flag
        else:
            self._loss_nonfinite = self._loss_nonfinite.to(flag.device) | flag
        prior = self._loss_nonfinite_sources.get(source)
        self._loss_nonfinite_sources[source] = (
            flag if prior is None else prior.to(flag.device) | flag
        )

    def apply_deepspeed(self, optimizer: object) -> OuterGradientAudit:
        """Scale DeepSpeed ZeRO-1/2 partition gradients immediately before engine.step()."""

        zero = cast(Any, optimizer)
        if float(getattr(zero, "clip_grad", 0.0)) != 0.0:
            raise ValueError("DeepSpeed global gradient clipping must be disabled")
        base_optimizer = zero.optimizer
        groups = self._named_param_groups(base_optimizer.param_groups)
        averaged = zero.averaged_gradients

        self.attempted_update_count += 1
        loss_nonfinite, local_loss_nonfinite = self._synchronize_loss_nonfinite(zero, averaged)
        if loss_nonfinite:
            self._inject_loss_overflow(averaged)
        overflow = bool(zero.has_overflow(partition_gradients=zero.partition_gradients))
        if overflow:
            self.skipped_update_count += 1
            nonfinite_audits = tuple(self._nonfinite_group_audit(group) for group in groups)
            sources: tuple[str, ...] = ()
            if loss_nonfinite:
                sources = (
                    self._materialize_loss_sources() if local_loss_nonfinite else ("remote_rank",)
                )
                warnings.warn(
                    "nonfinite Outer loss detected; DeepSpeed overflow will skip the complete "
                    f"optimizer/scheduler update ({', '.join(sources)})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._record(
                nonfinite_audits,
                skipped_nonfinite=True,
                skipped_nonfinite_loss=loss_nonfinite,
                nonfinite_loss_sources=sources,
            )

        loss_scale = float(zero.loss_scale)
        group_audits: list[GroupGradientAudit] = []
        for index, (name, _group) in enumerate(groups):
            gradients = cast(list[Tensor], averaged[index])
            params = zero.params_in_partition[index]
            parameter_gradients = self._parameter_partition_gradients(gradients, params)
            scaled_norm = zero.get_grad_norm_direct(parameter_gradients, params)
            pre_norm = float(scaled_norm.detach().float().item()) / loss_scale
            max_norm = self._max_norm(name)
            coefficient = self._clip_coefficient(pre_norm, max_norm)
            for gradient in parameter_gradients:
                gradient.mul_(coefficient)
            group_audits.append(
                GroupGradientAudit(
                    name=name,
                    max_norm=max_norm,
                    pre_clip_norm=pre_norm,
                    post_clip_norm=pre_norm * coefficient,
                    clip_coefficient=coefficient,
                )
            )
        self.successful_update_count += 1
        return self._record(
            tuple(group_audits),
            skipped_nonfinite=False,
            skipped_nonfinite_loss=False,
            nonfinite_loss_sources=(),
        )

    @staticmethod
    def _named_param_groups(
        param_groups: list[dict[str, Any]],
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Pair each optimizer group with its own ``group_name``.

        Per-group caps are looked up by the group's own name, so ordering drift
        cannot mis-cap a group.
        """

        return tuple((cast(str, group.get("group_name")), group) for group in param_groups)

    @staticmethod
    def _parameter_partition_gradients(
        gradients: list[Tensor],
        params: Sequence[object],
    ) -> list[Tensor]:
        """Drop only DeepSpeed's trailing ZeRO partition-alignment gradient.

        DeepSpeed 0.18.8 ``get_flat_partition(..., return_tensor_list=True)``
        appends one zero tensor when the final data-parallel partition ends
        after its last owned parameter.  That tensor has no matching entry in
        ``params_in_partition``.
        """

        return gradients[: len(params)]

    def _max_norm(self, name: str) -> float:
        return float(getattr(self.config.max_grad_norm, name))

    @staticmethod
    def _clip_coefficient(pre_norm: float, max_norm: float) -> float:
        if pre_norm <= max_norm * (1.0 + 1.0e-6):
            return 1.0
        return max_norm / max(pre_norm, float(torch.finfo(torch.float32).tiny))

    def _record(
        self,
        groups: tuple[GroupGradientAudit, ...],
        *,
        skipped_nonfinite: bool,
        skipped_nonfinite_loss: bool,
        nonfinite_loss_sources: tuple[str, ...],
    ) -> OuterGradientAudit:
        audit = OuterGradientAudit(
            attempted_update_count=self.attempted_update_count,
            successful_update_count=self.successful_update_count,
            skipped_update_count=self.skipped_update_count,
            skipped_nonfinite=skipped_nonfinite,
            skipped_nonfinite_loss=skipped_nonfinite_loss,
            nonfinite_loss_sources=nonfinite_loss_sources,
            groups=groups,
        )
        self.last_audit = audit
        self._loss_nonfinite = None
        self._loss_nonfinite_sources.clear()
        return audit

    def _synchronize_loss_nonfinite(self, zero: Any, averaged: object) -> tuple[bool, bool]:
        device = self._collective_device(zero, averaged)
        flag = torch.zeros((), dtype=torch.int64, device=device)
        if self._loss_nonfinite is not None:
            flag.copy_(self._loss_nonfinite.to(device=device, dtype=torch.int64))
        local_nonfinite = bool(flag.item())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                group=self._process_group(zero, 0),
            )
        return bool(flag.item()), local_nonfinite

    def _materialize_loss_sources(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                source
                for source, flag in self._loss_nonfinite_sources.items()
                if bool(flag.item())
            )
        )

    @staticmethod
    def _collective_device(zero: Any, averaged: object) -> torch.device:
        for gradients in cast(dict[int, list[Tensor]], averaged).values():
            for gradient in gradients:
                return gradient.device
        for parameters in zero.params_in_partition:
            for parameter in parameters:
                if isinstance(parameter, Tensor):
                    return parameter.device
        raise RuntimeError("no ZeRO partition tensor is available for nonfinite synchronization")

    @staticmethod
    def _inject_loss_overflow(averaged: object) -> None:
        for gradients in cast(dict[int, list[Tensor]], averaged).values():
            for gradient in gradients:
                if gradient.numel() > 0:
                    gradient.reshape(-1)[0] = float("nan")
                    return
        raise RuntimeError(
            "nonfinite loss was synchronized but no ZeRO averaged gradient was available"
        )

    @staticmethod
    def _nonfinite_group_audit(group: tuple[str, dict[str, Any]]) -> GroupGradientAudit:
        name, _values = group
        return GroupGradientAudit(
            name=name,
            max_norm=0.0,
            pre_clip_norm=math.nan,
            post_clip_norm=math.nan,
            clip_coefficient=0.0,
        )

    @staticmethod
    def _process_group(zero: Any, index: int) -> object | None:
        groups = getattr(zero, "real_dp_process_group", None)
        return groups[index] if groups is not None else getattr(zero, "dp_process_group", None)


def sanitize_scalar_loss(
    loss: Tensor,
    *,
    source: str,
    controller: OuterGradientController,
) -> Tensor:
    """Keep the autograd/collective schedule while deferring skip ownership to DeepSpeed."""

    controller.record_loss(loss, source)
    return torch.where(
        torch.isfinite(loss.detach()),
        loss,
        torch.nan_to_num(loss) * 0.0,
    )


__all__ = [
    "GroupGradientAudit",
    "OuterGradientAudit",
    "OuterGradientController",
    "sanitize_scalar_loss",
]
