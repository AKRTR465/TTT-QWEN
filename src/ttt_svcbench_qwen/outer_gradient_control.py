"""Per-optimizer-group Outer gradient clipping for A2/A5 DeepSpeed training."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
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
class GradientProbe:
    """One named parameter subset measured separately inside its owning group.

    A whole-group norm cannot answer "is this specific mechanism receiving
    gradient", because one group pools parameters fed by different paths.  A
    probe isolates the subset whose gradient source is the mechanism itself.
    """

    name: str
    group_name: str
    parameters: tuple[Tensor, ...]
    reference_group: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.group_name:
            raise ValueError("gradient probes require a non-empty name and group_name")
        if not self.parameters:
            raise ValueError(f"gradient probe {self.name!r} requires at least one parameter")
        if len(self.member_ids) != len(self.parameters):
            raise ValueError(f"gradient probe {self.name!r} lists one parameter twice")

    @property
    def member_ids(self) -> frozenset[int]:
        return frozenset(id(parameter) for parameter in self.parameters)


@dataclass(frozen=True, slots=True)
class ProbeGradientAudit:
    """Pre-clip gradient norm for one probe, summed across data-parallel ranks."""

    name: str
    group_name: str
    reference_group: str | None
    norm: float
    parameter_count: int
    element_count: int


@dataclass(frozen=True, slots=True)
class OuterGradientAudit:
    attempted_update_count: int
    successful_update_count: int
    skipped_update_count: int
    within_initial_audit_window: bool
    skipped_nonfinite: bool
    skipped_nonfinite_loss: bool
    nonfinite_loss_sources: tuple[str, ...]
    groups: tuple[GroupGradientAudit, ...]
    probes: tuple[ProbeGradientAudit, ...] = ()

    def probe(self, name: str) -> ProbeGradientAudit:
        """Return one named probe audit or fail closed on probe-topology drift."""

        matches = tuple(probe for probe in self.probes if probe.name == name)
        if len(matches) != 1:
            raise RuntimeError(
                f"Outer gradient audit requires exactly one {name!r} probe; found {len(matches)}"
            )
        return matches[0]

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
        for probe in self.probes:
            prefix = f"outer_grad/probe/{probe.name}"
            values.extend(
                (
                    (f"{prefix}/norm", probe.norm),
                    (f"{prefix}/element_count", float(probe.element_count)),
                )
            )
            if probe.reference_group is None:
                continue
            reference = self.group(probe.reference_group).pre_clip_norm
            # A bare norm cannot distinguish "small" from "small relative to the
            # competing group"; the schema-12 neutralization was four orders below
            # W0 while still being nonzero.  nan marks an unusable denominator
            # rather than silently reporting a ratio against zero.
            values.append(
                (
                    f"{prefix}/norm_over_{probe.reference_group}",
                    probe.norm / reference if reference > 0.0 else math.nan,
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
        probes: Sequence[GradientProbe] = (),
    ) -> None:
        if not expected_groups or len(set(expected_groups)) != len(expected_groups):
            raise ValueError("Outer gradient groups must be unique and non-empty")
        self.probes = tuple(probes)
        names = tuple(probe.name for probe in self.probes)
        if len(set(names)) != len(names):
            raise ValueError("Outer gradient probe names must be unique")
        for probe in self.probes:
            for group_name in (probe.group_name, probe.reference_group):
                if group_name is not None and group_name not in expected_groups:
                    raise ValueError(
                        f"gradient probe {probe.name!r} references unknown group {group_name!r}"
                    )
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

        if loss.ndim != 0 or not loss.requires_grad:
            raise ValueError(f"{source} loss must be one differentiable scalar Tensor")
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
        loss_nonfinite, local_loss_nonfinite = self._synchronize_loss_nonfinite(zero, averaged)
        if loss_nonfinite:
            self._inject_loss_overflow(averaged)
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
                # Keep every probe metric key present across skipped steps so a
                # dashboard gap means "not measured", never "measured as zero".
                probes=tuple(self._nonfinite_probe_audit(probe) for probe in self.probes),
            )

        loss_scale = float(zero.loss_scale)
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise RuntimeError("DeepSpeed exposed an invalid loss scale")
        group_audits: list[GroupGradientAudit] = []
        probe_audits: list[ProbeGradientAudit] = []
        for index, (name, group) in enumerate(groups):
            gradients = cast(list[Tensor], averaged[index])
            params = zero.params_in_partition[index]
            parameter_gradients = self._parameter_partition_gradients(gradients, params)
            scaled_norm = zero.get_grad_norm_direct(parameter_gradients, params)
            pre_norm = float(scaled_norm.detach().float().item()) / loss_scale
            max_norm = self._max_norm(name)
            coefficient = self._clip_coefficient(pre_norm, max_norm)
            active_elements, max_abs = self._distributed_shape_and_max(
                parameter_gradients, self._process_group(zero, index), loss_scale
            )
            # Measure before clipping so probe and group norms share one scale.
            probe_audits.extend(
                self._probe_audits(
                    group_name=name,
                    gradients=parameter_gradients,
                    params=params,
                    process_group=self._process_group(zero, index),
                    loss_scale=loss_scale,
                )
            )
            for gradient in parameter_gradients:
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
        # Probe audits come out in group order, so compare membership rather than
        # sequence; each probe must be measured by exactly one owning group.
        measured = sorted(audit.name for audit in probe_audits)
        if measured != sorted(probe.name for probe in self.probes):
            raise RuntimeError(
                "Outer gradient probes did not resolve to exactly one owning optimizer group"
            )
        return self._record(
            tuple(group_audits),
            skipped_nonfinite=False,
            skipped_nonfinite_loss=False,
            nonfinite_loss_sources=(),
            probes=tuple(probe_audits),
        )

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

    @staticmethod
    def _parameter_partition_gradients(
        gradients: list[Tensor],
        params: Sequence[object],
    ) -> list[Tensor]:
        """Drop only DeepSpeed's trailing ZeRO partition-alignment gradient.

        DeepSpeed 0.18.8 ``get_flat_partition(..., return_tensor_list=True)``
        appends one zero tensor when the final data-parallel partition ends
        after its last owned parameter.  That tensor has no matching entry in
        ``params_in_partition``.  Treating the pinned padding as a parameter
        topology error makes the last rank leave the collective schedule while
        every other rank enters the probe all-reduce.
        """

        parameter_count = len(params)
        if len(gradients) < parameter_count:
            raise RuntimeError(
                "DeepSpeed partition exposed fewer gradients than owned parameters"
            )
        padding = gradients[parameter_count:]
        if len(padding) > 1 or any(bool(torch.count_nonzero(value).item()) for value in padding):
            raise RuntimeError(
                "DeepSpeed partition gradients contain unexpected nonzero or repeated padding"
            )
        return gradients[:parameter_count]

    def _max_norm(self, name: str) -> float:
        return float(getattr(self.config.max_grad_norm, name))

    @staticmethod
    def _clip_coefficient(pre_norm: float, max_norm: float) -> float:
        if pre_norm <= max_norm * (1.0 + 1.0e-6):
            return 1.0
        return max_norm / max(pre_norm, float(torch.finfo(torch.float32).tiny))

    def _probe_audits(
        self,
        *,
        group_name: str,
        gradients: list[Tensor],
        params: Sequence[object],
        process_group: object | None,
        loss_scale: float,
    ) -> tuple[ProbeGradientAudit, ...]:
        """Measure each probe owned by one group as a cross-rank pre-clip norm."""

        probes = tuple(probe for probe in self.probes if probe.group_name == group_name)
        if not probes:
            return ()
        if len(gradients) != len(params):
            raise RuntimeError(
                "DeepSpeed partition gradients and parameters must align to measure probes"
            )
        device = gradients[0].device if gradients else torch.device("cpu")
        # [probe, (sum of squares, element count)] — one collective per group keeps
        # the all-reduce schedule identical on ranks that hold no probe parameter.
        statistics = torch.zeros((len(probes), 2), dtype=torch.float64, device=device)
        for row, probe in enumerate(probes):
            members = probe.member_ids
            for gradient, parameter in zip(gradients, params, strict=True):
                if id(parameter) not in members:
                    continue
                statistics[row, 0] += gradient.detach().double().square().sum()
                statistics[row, 1] += gradient.numel()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                statistics, op=torch.distributed.ReduceOp.SUM, group=process_group
            )
        return tuple(
            ProbeGradientAudit(
                name=probe.name,
                group_name=group_name,
                reference_group=probe.reference_group,
                norm=float(statistics[row, 0].sqrt().item()) / loss_scale,
                parameter_count=len(probe.parameters),
                element_count=int(statistics[row, 1].item()),
            )
            for row, probe in enumerate(probes)
        )

    @staticmethod
    def _nonfinite_probe_audit(probe: GradientProbe) -> ProbeGradientAudit:
        return ProbeGradientAudit(
            name=probe.name,
            group_name=probe.group_name,
            reference_group=probe.reference_group,
            norm=math.nan,
            parameter_count=len(probe.parameters),
            element_count=0,
        )

    def _record(
        self,
        groups: tuple[GroupGradientAudit, ...],
        *,
        skipped_nonfinite: bool,
        skipped_nonfinite_loss: bool,
        nonfinite_loss_sources: tuple[str, ...],
        probes: tuple[ProbeGradientAudit, ...] = (),
    ) -> OuterGradientAudit:
        audit = OuterGradientAudit(
            attempted_update_count=self.attempted_update_count,
            successful_update_count=self.successful_update_count,
            skipped_update_count=self.skipped_update_count,
            within_initial_audit_window=(self.successful_update_count <= self.config.audit_steps),
            skipped_nonfinite=skipped_nonfinite,
            skipped_nonfinite_loss=skipped_nonfinite_loss,
            nonfinite_loss_sources=nonfinite_loss_sources,
            groups=groups,
            probes=probes,
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
        sources = tuple(
            sorted(
                source for source, flag in self._loss_nonfinite_sources.items() if bool(flag.item())
            )
        )
        if not sources:
            raise RuntimeError("local nonfinite loss flag had no owning source")
        return sources

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
    def _nonfinite_group_audit(group: tuple[str, dict[str, Any]], count: int) -> GroupGradientAudit:
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


def sanitize_scalar_loss(
    loss: Tensor,
    *,
    source: str,
    controller: OuterGradientController,
) -> Tensor:
    """Keep the autograd/collective schedule while deferring skip ownership to DeepSpeed."""

    if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ValueError(f"{source} loss must be one differentiable scalar Tensor")
    controller.record_loss(loss, source)
    return torch.where(
        torch.isfinite(loss.detach()),
        loss,
        torch.nan_to_num(loss) * 0.0,
    )


__all__ = [
    "GradientProbe",
    "GroupGradientAudit",
    "OuterGradientAudit",
    "OuterGradientController",
    "ProbeGradientAudit",
    "sanitize_scalar_loss",
]
