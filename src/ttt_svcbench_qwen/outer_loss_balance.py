"""Distributed Answer-dominant official-weak loss composition with checkpointed EMA state."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ttt_svcbench_qwen.config import OfficialWeakBalanceConfig
from ttt_svcbench_qwen.losses import (
    AnswerLossOutput,
    OuterLossOutput,
    TTTLossOutput,
    compose_outer_loss_terms,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase, trace_event
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
)

_TERM_NAMES = ("task", "operator", "retrieval", "time")
_TERM_SLOT_COUNT = len(_TERM_NAMES)
_STAT_TERM_COUNT = 1 + _TERM_SLOT_COUNT
_LOSS_STAT_VECTOR_LENGTH = 2 * _STAT_TERM_COUNT
_GRAD_STAT_VECTOR_LENGTH = 2 * _TERM_SLOT_COUNT
_STAT_VECTOR_LENGTH = _LOSS_STAT_VECTOR_LENGTH + _GRAD_STAT_VECTOR_LENGTH
_BALANCE_CHECKPOINT_SCHEMA = 7


@dataclass(frozen=True, slots=True)
class OfficialWeakGradientAnchors:
    """Activation reference planes used to measure the four weak-loss gradients."""

    q_target: Tensor
    q_operator: Tensor
    q_time: Tensor

    def __post_init__(self) -> None:
        values = (self.q_target, self.q_operator, self.q_time)
        if any(value.ndim != 2 or not torch.is_floating_point(value) for value in values):
            raise ValueError("official-weak gradient anchors must be floating [B, D]")
        if len({(value.shape, value.device) for value in values}) != 1:
            raise ValueError("official-weak gradient anchors must share shape and device")

    def for_term(self, name: str) -> Tensor:
        if name in {"task", "retrieval"}:
            return self.q_target
        if name == "operator":
            return self.q_operator
        if name == "time":
            return self.q_time
        raise ValueError(f"unknown official-weak gradient term: {name}")


@dataclass(frozen=True, slots=True)
class OfficialWeakTermBalanceMetrics:
    name: str
    raw_global_mean: Tensor
    scale: Tensor
    aligned_global_mean: Tensor
    weighted_global_mean: Tensor
    global_valid_count: Tensor
    scale_clamped: Tensor
    loss_scale: Tensor
    gradient_scale: Tensor
    raw_gradient_rms: Tensor
    ema_gradient_rms: Tensor

    def __post_init__(self) -> None:
        if self.name not in _TERM_NAMES:
            raise ValueError(f"unknown official-weak balance term: {self.name}")
        values = (
            self.raw_global_mean,
            self.scale,
            self.aligned_global_mean,
            self.weighted_global_mean,
            self.loss_scale,
            self.gradient_scale,
            self.raw_gradient_rms,
            self.ema_gradient_rms,
            self.global_valid_count,
            self.scale_clamped,
        )
        if any(value.ndim != 0 or value.requires_grad for value in values):
            raise ValueError("official-weak balance audit values must be detached scalars")
        if self.global_valid_count.dtype != torch.float64:
            raise TypeError("official-weak audit counts must use the packed float64 dtype")
        if self.scale_clamped.dtype != torch.bool:
            raise TypeError("official-weak audit clamp flags must be bool")


@dataclass(frozen=True, slots=True)
class OfficialWeakBalanceAudit:
    answer_global_mean: Tensor
    answer_global_count: Tensor
    state_global_mean: Tensor
    terms: tuple[OfficialWeakTermBalanceMetrics, ...]
    auxiliary_to_answer_ratio: Tensor
    group_guard: Tensor
    group_guard_active: Tensor
    group_guard_reference: Tensor
    group_guard_reference_floored: Tensor
    state_to_reference_ratio: Tensor
    state_to_current_answer_ratio: Tensor
    ema_means: tuple[Tensor, ...] = ()
    ema_update_counts: tuple[Tensor, ...] = ()
    gradient_ema_rms: tuple[Tensor, ...] = ()
    gradient_ema_update_counts: tuple[Tensor, ...] = ()

    def __post_init__(self) -> None:
        if tuple(term.name for term in self.terms) != _TERM_NAMES:
            raise ValueError("official-weak balance audit term order drifted")
        scalars = (
            self.answer_global_mean,
            self.answer_global_count,
            self.state_global_mean,
            self.auxiliary_to_answer_ratio,
            self.group_guard,
            self.group_guard_active,
            self.group_guard_reference,
            self.group_guard_reference_floored,
            self.state_to_reference_ratio,
            self.state_to_current_answer_ratio,
            *self.ema_means,
            *self.ema_update_counts,
            *self.gradient_ema_rms,
            *self.gradient_ema_update_counts,
        )
        if any(value.ndim != 0 or value.requires_grad for value in scalars):
            raise ValueError("official-weak audit values must be detached scalar tensors")
        if self.group_guard_active.dtype != torch.bool:
            raise TypeError("official-weak group guard audit flag must be bool")
        if self.group_guard_reference_floored.dtype != torch.bool:
            raise TypeError("official-weak Answer-reference floor audit flag must be bool")
        if self.ema_means and len(self.ema_means) != _STAT_TERM_COUNT:
            raise ValueError("official-weak EMA means must include Answer plus four terms")
        if self.ema_update_counts and (len(self.ema_update_counts) != _STAT_TERM_COUNT):
            raise ValueError("official-weak EMA update counts are invalid")
        if self.gradient_ema_rms and len(self.gradient_ema_rms) != _TERM_SLOT_COUNT:
            raise ValueError("official-weak gradient EMA must contain four slots")
        if self.gradient_ema_update_counts and (
            len(self.gradient_ema_update_counts) != _TERM_SLOT_COUNT
        ):
            raise ValueError("official-weak gradient EMA update counts are invalid")

    def metrics(self) -> tuple[tuple[str, float | None], ...]:
        values: list[tuple[str, float | None]] = [
            ("loss/answer", _audit_float(self.answer_global_mean)),
            ("loss/state", _audit_float(self.state_global_mean)),
            ("loss/outer_total", _audit_float(self.answer_global_mean + self.state_global_mean)),
        ]
        for term in self.terms:
            values.extend(
                (
                    (f"loss/raw/{term.name}", _audit_optional_float(term.raw_global_mean)),
                    (f"loss/scale/{term.name}", _audit_optional_float(term.scale)),
                    (f"loss/aligned/{term.name}", _audit_optional_float(term.aligned_global_mean)),
                    (
                        f"loss/weighted/{term.name}",
                        _audit_optional_float(term.weighted_global_mean),
                    ),
                    (f"loss/global_valid_count/{term.name}", _audit_float(term.global_valid_count)),
                    (f"loss/scale_clamped/{term.name}", _audit_float(term.scale_clamped)),
                    (
                        f"grad_balance/raw_rms/{term.name}",
                        _audit_optional_float(term.raw_gradient_rms),
                    ),
                    (
                        f"grad_balance/ema_rms/{term.name}",
                        _audit_optional_float(term.ema_gradient_rms),
                    ),
                    (
                        f"grad_balance/loss_scale/{term.name}",
                        _audit_optional_float(term.loss_scale),
                    ),
                    (
                        f"grad_balance/grad_scale/{term.name}",
                        _audit_optional_float(term.gradient_scale),
                    ),
                    (f"grad_balance/final_scale/{term.name}", _audit_optional_float(term.scale)),
                    (
                        f"grad_balance/scale_clamped/{term.name}",
                        _audit_float(term.scale_clamped),
                    ),
                    (
                        f"grad_balance/global_valid_count/{term.name}",
                        _audit_float(term.global_valid_count),
                    ),
                )
            )
        values.extend(
            (
                ("loss/aux_to_answer_ratio", _audit_float(self.auxiliary_to_answer_ratio)),
                ("loss/group_guard", _audit_float(self.group_guard)),
                ("loss/group_guard_active", _audit_float(self.group_guard_active)),
                ("loss/group_guard_reference", _audit_float(self.group_guard_reference)),
                (
                    "loss/group_guard_reference_floored",
                    _audit_float(self.group_guard_reference_floored),
                ),
                ("loss/state_to_reference_ratio", _audit_float(self.state_to_reference_ratio)),
                (
                    "loss/state_to_current_answer_ratio",
                    _audit_float(self.state_to_current_answer_ratio),
                ),
            )
        )
        if self.ema_means:
            for name, mean, updates in zip(
                ("answer", *_TERM_NAMES),
                self.ema_means,
                self.ema_update_counts,
                strict=True,
            ):
                values.append((f"loss/ema/{name}", _audit_optional_float(mean)))
                values.append((f"loss/ema_updates/{name}", _audit_float(updates)))
        if self.gradient_ema_rms:
            for name, _rms, updates in zip(
                _TERM_NAMES,
                self.gradient_ema_rms,
                self.gradient_ema_update_counts,
                strict=True,
            ):
                values.append((f"grad_balance/ema_updates/{name}", _audit_float(updates)))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class OfficialWeakBalancedBatch:
    objectives: tuple[OuterLossOutput, ...]
    mean_total: Tensor
    audit: OfficialWeakBalanceAudit | None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("official-weak composition requires at least one objective")
        if self.mean_total.ndim != 0:
            raise ValueError("official-weak batch total must be scalar")


ReduceSum = Callable[[Tensor], Tensor]


class OfficialWeakOuterLossComposer(nn.Module):  # type: ignore[misc]
    """Compose one A2 micro-step or all A5 Query points with one fixed collective."""

    def __init__(
        self,
        config: OfficialWeakBalanceConfig,
        *,
        reduce_sum: ReduceSum | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, OfficialWeakBalanceConfig):
            raise TypeError("official-weak composer requires validated balance config")
        if (reduce_sum is None) != (world_size is None):
            raise ValueError("custom reduction and world_size must be provided together")
        if world_size is not None and world_size <= 0:
            raise ValueError("official-weak world_size must be positive")
        self.config = config
        self._reduce_sum = reduce_sum
        self._world_size = world_size
        self.register_buffer(
            "ema_values",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "ema_valid",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "ema_update_counts",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "gradient_ema_values",
            torch.zeros(_TERM_SLOT_COUNT, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "gradient_ema_valid",
            torch.zeros(_TERM_SLOT_COUNT, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "gradient_ema_update_counts",
            torch.zeros(_TERM_SLOT_COUNT, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "balance_schema_version",
            torch.tensor(_BALANCE_CHECKPOINT_SCHEMA, dtype=torch.int64),
            persistent=True,
        )
        self._assert_balance_state()

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> OfficialWeakOuterLossComposer:
        """Move persistent EMA state without ever applying a lower floating dtype."""

        snapshots = {
            name: getattr(self, name).detach().clone().to(dtype=torch.float64)
            for name in ("ema_values", "gradient_ema_values")
        }
        result = cast(
            OfficialWeakOuterLossComposer,
            super()._apply(fn, recurse=recurse),
        )
        for name, snapshot in snapshots.items():
            transformed = getattr(self, name)
            self._buffers[name] = snapshot.to(
                device=transformed.device,
                dtype=torch.float64,
            )
        self._assert_balance_state()
        return result

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject quantized or stale balance state before PyTorch can silently cast it."""

        expected = {
            "ema_values": (torch.float64, (_STAT_TERM_COUNT,)),
            "ema_valid": (torch.bool, (_STAT_TERM_COUNT,)),
            "ema_update_counts": (torch.int64, (_STAT_TERM_COUNT,)),
            "gradient_ema_values": (torch.float64, (_TERM_SLOT_COUNT,)),
            "gradient_ema_valid": (torch.bool, (_TERM_SLOT_COUNT,)),
            "gradient_ema_update_counts": (torch.int64, (_TERM_SLOT_COUNT,)),
            "balance_schema_version": (torch.int64, ()),
        }
        validation_errors = 0
        for name, (dtype, shape) in expected.items():
            key = f"{prefix}{name}"
            source = state_dict.get(key)
            if source is None:
                error_msgs.append(f'Missing required balance-state key "{key}".')
                validation_errors += 1
                continue
            if source.dtype != dtype or tuple(source.shape) != shape:
                error_msgs.append(
                    f'Invalid balance-state tensor "{key}": expected {dtype} {shape}, '
                    f"found {source.dtype} {tuple(source.shape)}."
                )
                validation_errors += 1
        schema = state_dict.get(f"{prefix}balance_schema_version")
        if (
            schema is not None
            and schema.dtype == torch.int64
            and tuple(schema.shape) == ()
            and int(schema.item()) != _BALANCE_CHECKPOINT_SCHEMA
        ):
            error_msgs.append(
                "Incompatible official-weak balance checkpoint schema: expected "
                f"{_BALANCE_CHECKPOINT_SCHEMA}, found {int(schema.item())}."
            )
            validation_errors += 1
        if validation_errors:
            return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._assert_balance_state()

    def _assert_balance_state(self) -> None:
        expected = (
            (self.ema_values, torch.float64, (_STAT_TERM_COUNT,), "ema_values"),
            (self.ema_valid, torch.bool, (_STAT_TERM_COUNT,), "ema_valid"),
            (
                self.ema_update_counts,
                torch.int64,
                (_STAT_TERM_COUNT,),
                "ema_update_counts",
            ),
            (
                self.gradient_ema_values,
                torch.float64,
                (_TERM_SLOT_COUNT,),
                "gradient_ema_values",
            ),
            (
                self.gradient_ema_valid,
                torch.bool,
                (_TERM_SLOT_COUNT,),
                "gradient_ema_valid",
            ),
            (
                self.gradient_ema_update_counts,
                torch.int64,
                (_TERM_SLOT_COUNT,),
                "gradient_ema_update_counts",
            ),
            (
                self.balance_schema_version,
                torch.int64,
                (),
                "balance_schema_version",
            ),
        )
        devices = {value.device for value, _dtype, _shape, _name in expected}
        if len(devices) != 1:
            raise RuntimeError("official-weak balance buffers must share one device")
        for value, dtype, shape, name in expected:
            if value.dtype != dtype or tuple(value.shape) != shape:
                raise RuntimeError(
                    f"official-weak {name} must remain {dtype} with shape {shape}"
                )
        if self.ema_values.device.type == "meta":
            return
        if not bool(torch.isfinite(self.ema_values).all()) or not bool(
            torch.isfinite(self.gradient_ema_values).all()
        ):
            raise RuntimeError("official-weak EMA buffers must remain finite")
        if bool(torch.any(self.ema_update_counts < 0)) or bool(
            torch.any(self.gradient_ema_update_counts < 0)
        ):
            raise RuntimeError("official-weak EMA update counts must remain non-negative")
        if int(self.balance_schema_version.item()) != _BALANCE_CHECKPOINT_SCHEMA:
            raise RuntimeError("official-weak balance schema buffer is incompatible")

    def compose(
        self,
        answers: Sequence[AnswerLossOutput],
        states: Sequence[OfficialWeakStateLossOutput],
        *,
        support_ttt: Sequence[tuple[TTTLossOutput, ...]] | None = None,
        gradient_anchors: Sequence[OfficialWeakGradientAnchors] | None = None,
        measure_gradients: bool = True,
    ) -> OfficialWeakBalancedBatch:
        self._assert_balance_state()
        answer_items = tuple(answers)
        state_items = tuple(states)
        if not answer_items or len(answer_items) != len(state_items):
            raise ValueError("official-weak Answer and State batches must be non-empty and aligned")
        supports = tuple(() for _ in answer_items) if support_ttt is None else tuple(support_ttt)
        if len(supports) != len(answer_items):
            raise ValueError("official-weak support-TTT batches must align to Query objectives")
        anchors = () if gradient_anchors is None else tuple(gradient_anchors)
        if anchors and len(anchors) != len(answer_items):
            raise ValueError("official-weak gradient anchors must align to Query objectives")
        device = answer_items[0].loss.value.device
        if any(answer.loss.value.device != device for answer in answer_items) or any(
            state.total.device != device for state in state_items
        ):
            raise ValueError("official-weak composed losses must share one device")
        if measure_gradients and len(anchors) != len(answer_items):
            raise ValueError("ema_answer_ref requires one gradient-anchor set per Query")
        if measure_gradients and anchors and not torch.is_grad_enabled():
            raise RuntimeError("official-weak activation gradients require grad-enabled execution")

        pack_started = time.perf_counter()
        answer_sums = tuple(_answer_local_sum(answer) for answer in answer_items)
        answer_counts = tuple(_answer_valid_rows(answer) for answer in answer_items)
        term_items = tuple(
            tuple(getattr(state, name) for state in state_items) for name in _TERM_NAMES
        )
        term_sums = tuple(tuple(_weak_local_sum(term) for term in terms) for terms in term_items)
        term_counts = tuple(tuple(term.valid_rows for term in terms) for terms in term_items)
        local_sums = (
            _sum_tensors(answer_sums),
            *(_sum_tensors(sums) for sums in term_sums),
        )
        local_counts = (
            sum(answer_counts),
            *(sum(counts) for counts in term_counts),
        )
        prior_loss_scales, loss_scale_clamped = self._prior_loss_scales(device)
        if measure_gradients:
            local_gradient_squares, local_gradient_counts = _gradient_local_statistics(
                term_items,
                anchors,
                tuple(prior_loss_scales.unbind()),
            )
        else:
            local_gradient_squares = tuple(
                local_sums[0].detach().new_zeros((), dtype=torch.float64) for _ in _TERM_NAMES
            )
            local_gradient_counts = (0,) * _TERM_SLOT_COUNT
        stats = _pack_stats(
            local_sums,
            local_counts,
            local_gradient_squares,
            local_gradient_counts,
        )
        trace_event(
            "outer_loss_balance_pack",
            seconds=time.perf_counter() - pack_started,
            query_count=len(answer_items),
        )
        with trace_cuda_phase("outer_loss_balance_collective", payload_values=stats.numel()):
            reduced, world_size = self._global_sum(stats)
        finalize_started = time.perf_counter()
        global_sums, global_counts, global_gradient_squares, global_gradient_counts = _unpack_stats(
            reduced
        )
        _validate_reduced_statistics(
            global_sums,
            global_counts,
            global_gradient_squares,
            global_gradient_counts,
        )
        epsilon = float(self.config.epsilon)
        loss_valid = global_counts > 0.0
        gradient_valid = global_gradient_counts > 0.0
        loss_update_valid = loss_valid & torch.isfinite(global_sums)
        gradient_update_valid = gradient_valid & torch.isfinite(
            global_gradient_squares
        )
        nan = torch.full((), float("nan"), dtype=torch.float64, device=device)
        current_means = torch.where(
            loss_valid,
            global_sums.detach() / global_counts.clamp_min(1.0),
            nan,
        )
        current_gradient_rms = torch.where(
            gradient_valid,
            (global_gradient_squares.detach() / global_gradient_counts.clamp_min(1.0)).sqrt(),
            nan,
        )
        answer_mean = global_sums[0] / global_counts[0]
        prior_gradient_scales, gradient_scale_clamped = self._prior_gradient_scales(
            global_counts[1:], device
        )
        history_valid = self.ema_valid[0].to(device) & self.ema_valid[1:].to(device)
        history_valid &= self.gradient_ema_valid.to(device)
        unbounded_scales = prior_loss_scales * prior_gradient_scales
        bounded_scales = unbounded_scales.clamp(
            min=float(self.config.scale_min),
            max=float(self.config.scale_max),
        )
        scales = torch.where(history_valid, bounded_scales, torch.ones_like(bounded_scales))
        clamped = history_valid & (
            loss_scale_clamped | gradient_scale_clamped | ~torch.isclose(scales, unbounded_scales)
        )
        term_raw_means = global_sums[1:] / global_counts[1:].clamp_min(1.0)
        aligned_means = torch.where(
            loss_valid[1:],
            scales * term_raw_means,
            torch.zeros_like(term_raw_means),
        )
        auxiliary_mean = aligned_means.sum() / float(_TERM_SLOT_COUNT)
        prior_answer = torch.where(
            self.ema_valid[0].to(device),
            self.ema_values[0].to(device=device, dtype=torch.float64),
            answer_mean.detach().to(dtype=torch.float64),
        )
        answer_reference_floor = float(self.config.answer_reference_floor)
        group_guard_reference_floored = prior_answer < answer_reference_floor
        group_guard_reference = prior_answer.clamp_min(answer_reference_floor)
        group_guard = torch.where(
            auxiliary_mean > 0.0,
            torch.minimum(
                torch.ones_like(answer_mean),
                group_guard_reference / auxiliary_mean.clamp_min(epsilon),
            ),
            torch.ones_like(answer_mean),
        )
        query_count = len(answer_items)
        objectives: list[OuterLossOutput] = []
        for item_index, (_answer, support) in enumerate(zip(answer_items, supports, strict=True)):
            answer_contribution = (
                float(query_count * world_size)
                * answer_sums[item_index]
                / global_counts[0].to(dtype=answer_sums[item_index].dtype)
            )
            item_term_sums = torch.stack(
                tuple(term_sums[term_index][item_index] for term_index in range(_TERM_SLOT_COUNT))
            )
            aligned_contributions = (
                float(query_count * world_size)
                * scales.to(dtype=item_term_sums.dtype)
                * item_term_sums
                / global_counts[1:].clamp_min(1.0).to(dtype=item_term_sums.dtype)
                * loss_valid[1:].to(dtype=item_term_sums.dtype)
            )
            state_contribution = (
                float(self.config.group_weight)
                * group_guard.to(dtype=answer_contribution.dtype)
                * aligned_contributions.sum()
                / float(_TERM_SLOT_COUNT)
            )
            objectives.append(
                compose_outer_loss_terms(
                    answer_after=answer_contribution,
                    state_after=state_contribution,
                    support_ttt=support,
                )
            )

        if self.training:
            self._update_ema(current_means, loss_update_valid)
            if measure_gradients:
                self._update_gradient_ema(
                    current_gradient_rms, gradient_update_valid
                )
        gradient_ema_rms = self._gradient_ema_for_audit()
        term_gradient_ema_rms = torch.stack(gradient_ema_rms)
        weighted_factor = float(self.config.group_weight) * group_guard
        term_metrics = tuple(
            OfficialWeakTermBalanceMetrics(
                name=name,
                raw_global_mean=torch.where(active, raw_mean, nan).detach().clone(),
                scale=torch.where(active, scale, nan).detach().clone(),
                aligned_global_mean=torch.where(active, aligned, nan).detach().clone(),
                weighted_global_mean=torch.where(
                    active,
                    weighted_factor * aligned / float(_TERM_SLOT_COUNT),
                    nan,
                )
                .detach()
                .clone(),
                global_valid_count=global_count.detach().clone(),
                scale_clamped=(was_clamped & active).detach().clone(),
                loss_scale=torch.where(active, loss_scale, nan).detach().clone(),
                gradient_scale=torch.where(active, gradient_scale, nan).detach().clone(),
                raw_gradient_rms=torch.where(raw_gradient_valid, raw_gradient_rms, nan)
                .detach()
                .clone(),
                ema_gradient_rms=ema_gradient_rms.detach().clone(),
            )
            for (
                name,
                global_count,
                scale,
                loss_scale,
                gradient_scale,
                raw_mean,
                aligned,
                was_clamped,
                raw_gradient_rms,
                ema_gradient_rms,
                active,
                raw_gradient_valid,
            ) in zip(
                _TERM_NAMES,
                global_counts[1:].unbind(),
                scales.unbind(),
                prior_loss_scales.unbind(),
                prior_gradient_scales.unbind(),
                term_raw_means.unbind(),
                aligned_means.unbind(),
                clamped.unbind(),
                current_gradient_rms.unbind(),
                term_gradient_ema_rms.unbind(),
                loss_valid[1:].unbind(),
                gradient_valid.unbind(),
                strict=True,
            )
        )
        weighted_auxiliary = weighted_factor * auxiliary_mean
        current_answer_ratio = weighted_auxiliary / answer_mean.clamp_min(epsilon)
        reference_ratio = weighted_auxiliary / group_guard_reference.clamp_min(epsilon)
        audit = OfficialWeakBalanceAudit(
            answer_global_mean=answer_mean.detach().clone(),
            answer_global_count=global_counts[0].detach().clone(),
            state_global_mean=weighted_auxiliary.detach().clone(),
            terms=term_metrics,
            auxiliary_to_answer_ratio=current_answer_ratio.detach().clone(),
            group_guard=group_guard.detach().clone(),
            group_guard_active=(group_guard < 1.0).detach().clone(),
            group_guard_reference=group_guard_reference.detach().clone(),
            group_guard_reference_floored=group_guard_reference_floored.detach().clone(),
            state_to_reference_ratio=reference_ratio.detach().clone(),
            state_to_current_answer_ratio=current_answer_ratio.detach().clone(),
            ema_means=self._ema_means_for_audit(),
            ema_update_counts=tuple(value.detach().clone() for value in self.ema_update_counts),
            gradient_ema_rms=gradient_ema_rms,
            gradient_ema_update_counts=tuple(
                value.detach().clone() for value in self.gradient_ema_update_counts
            ),
        )
        trace_event(
            "outer_loss_balance_finalize",
            seconds=time.perf_counter() - finalize_started,
            global_answer_count=global_counts[0],
            global_task_count=global_counts[1],
            global_operator_count=global_counts[2],
            global_retrieval_count=global_counts[3],
            global_time_count=global_counts[4],
            group_guard_active=audit.group_guard_active,
            scale_clamped=tuple(term.scale_clamped for term in audit.terms),
        )
        objective_tuple = tuple(objectives)
        return OfficialWeakBalancedBatch(
            objectives=objective_tuple,
            mean_total=torch.stack(tuple(item.total for item in objective_tuple)).mean(),
            audit=audit,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _update_ema(self, current_means: Tensor, valid: Tensor) -> None:
        if current_means.shape != (_STAT_TERM_COUNT,) or valid.shape != (_STAT_TERM_COUNT,):
            raise ValueError("official-weak EMA update shape drifted")
        beta = float(self.config.ema_beta)
        values = current_means.to(device=self.ema_values.device, dtype=torch.float64)
        update = valid.to(device=self.ema_valid.device, dtype=torch.bool)
        prior_valid = self.ema_valid.clone()
        candidate = torch.where(
            prior_valid,
            self.ema_values * beta + values * (1.0 - beta),
            values,
        )
        self.ema_values.copy_(torch.where(update, candidate, self.ema_values))
        self.ema_valid.logical_or_(update)
        self.ema_update_counts.add_(update.to(dtype=torch.int64))

    def _prior_loss_scales(self, device: torch.device) -> tuple[Tensor, Tensor]:
        values = self.ema_values.detach().to(device=device, dtype=torch.float64)
        history_valid = self.ema_valid[0].to(device) & self.ema_valid[1:].to(device)
        epsilon = float(self.config.epsilon)
        ratio = values[0] / (values[1:] + epsilon)
        bounded = ratio.clamp(
            min=float(self.config.scale_min),
            max=float(self.config.scale_max),
        )
        scales = torch.where(history_valid, bounded, torch.ones_like(bounded))
        return scales, history_valid & ~torch.isclose(bounded, ratio)

    def _prior_gradient_scales(
        self,
        active_counts: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if active_counts.shape != (_TERM_SLOT_COUNT,):
            raise ValueError("official-weak gradient active-count shape drifted")
        epsilon = float(self.config.epsilon)
        historical = self.gradient_ema_values.detach().to(device=device, dtype=torch.float64)
        active = (active_counts > 0.0) & self.gradient_ema_valid.to(device)
        active_float = active.to(dtype=torch.float64)
        target = (
            ((historical + epsilon).log() * active_float).sum() / active_float.sum().clamp_min(1.0)
        ).exp()
        ratio = target / (historical + epsilon)
        bounded = ratio.clamp(
            min=float(self.config.grad_scale_min),
            max=float(self.config.grad_scale_max),
        )
        scales = torch.where(active, bounded, torch.ones_like(bounded))
        return scales, active & ~torch.isclose(bounded, ratio)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _update_gradient_ema(self, current_rms: Tensor, valid: Tensor) -> None:
        if current_rms.shape != (_TERM_SLOT_COUNT,) or valid.shape != (_TERM_SLOT_COUNT,):
            raise ValueError("official-weak gradient EMA update shape drifted")
        beta = float(self.config.grad_ema_beta)
        values = current_rms.to(device=self.gradient_ema_values.device, dtype=torch.float64)
        update = valid.to(device=self.gradient_ema_valid.device, dtype=torch.bool)
        prior_valid = self.gradient_ema_valid.clone()
        candidate = torch.where(
            prior_valid,
            self.gradient_ema_values * beta + values * (1.0 - beta),
            values,
        )
        self.gradient_ema_values.copy_(torch.where(update, candidate, self.gradient_ema_values))
        self.gradient_ema_valid.logical_or_(update)
        self.gradient_ema_update_counts.add_(update.to(dtype=torch.int64))

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reset_ema(self) -> None:
        """Reset A2 statistics at the A2-to-A5 stage boundary."""

        self.ema_values.zero_()
        self.ema_valid.zero_()
        self.ema_update_counts.zero_()
        self.gradient_ema_values.zero_()
        self.gradient_ema_valid.zero_()
        self.gradient_ema_update_counts.zero_()
        self.balance_schema_version.fill_(_BALANCE_CHECKPOINT_SCHEMA)
        self._assert_balance_state()

    def _ema_means_for_audit(self) -> tuple[Tensor, ...]:
        values = torch.where(
            self.ema_valid,
            self.ema_values,
            torch.full_like(self.ema_values, float("nan")),
        )
        return tuple(value.detach().clone() for value in values)

    def _gradient_ema_for_audit(self) -> tuple[Tensor, ...]:
        values = torch.where(
            self.gradient_ema_valid,
            self.gradient_ema_values,
            torch.full_like(self.gradient_ema_values, float("nan")),
        )
        return tuple(value.detach().clone() for value in values)

    def calibrate(
        self,
        answers: Sequence[AnswerLossOutput],
        states: Sequence[OfficialWeakStateLossOutput],
    ) -> OfficialWeakBalancedBatch:
        """Select streamed-A5 coefficients without differentiating no-grad calibration graphs."""

        return self.compose(
            answers,
            states,
            measure_gradients=False,
        )

    def measure_streamed_gradients(
        self,
        state: OfficialWeakStateLossOutput,
        anchors: OfficialWeakGradientAnchors,
        audit: OfficialWeakBalanceAudit,
    ) -> Tensor:
        """Measure one streamed Query locally; buffer/parameter gradients remain untouched."""

        device = state.total.device
        loss_scales = tuple(
            term.loss_scale.to(device=device, dtype=torch.float64) for term in audit.terms
        )
        squares, counts = _gradient_local_statistics(
            tuple((getattr(state, name),) for name in _TERM_NAMES),
            (anchors,),
            loss_scales,
        )
        return _pack_gradient_stats(squares, counts)

    def commit_streamed_gradients(
        self,
        local_statistics: Sequence[Tensor],
        audit: OfficialWeakBalanceAudit,
    ) -> OfficialWeakBalanceAudit:
        """Commit fixed gradient stats after all streamed Query graphs are measured."""

        values = tuple(local_statistics)
        if not values:
            raise ValueError("streamed gradient balancing requires Query statistics")
        packed = torch.stack(values).sum(dim=0)
        with trace_cuda_phase("outer_gradient_balance_collective", payload_values=packed.numel()):
            reduced = self._global_gradient_sum(packed)
        squares, counts = _unpack_gradient_stats(reduced)
        _validate_reduced_statistics(
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.float64, device=reduced.device),
            torch.ones(_STAT_TERM_COUNT, dtype=torch.float64, device=reduced.device),
            squares,
            counts,
        )
        valid = counts > 0.0
        update_valid = valid & torch.isfinite(squares)
        current_rms = torch.where(
            valid,
            (squares.detach() / counts.clamp_min(1.0)).sqrt(),
            torch.full_like(squares, float("nan")),
        )
        if self.training:
            self._update_gradient_ema(current_rms, update_valid)
        ema_rms = self._gradient_ema_for_audit()
        terms = tuple(
            replace(
                term,
                raw_gradient_rms=raw.detach().clone(),
                ema_gradient_rms=ema,
            )
            for term, raw, ema in zip(audit.terms, current_rms.unbind(), ema_rms, strict=True)
        )
        return replace(
            audit,
            terms=terms,
            gradient_ema_rms=ema_rms,
            gradient_ema_update_counts=tuple(
                value.detach().clone() for value in self.gradient_ema_update_counts
            ),
        )

    def compose_one_from_audit(
        self,
        answer: AnswerLossOutput,
        state: OfficialWeakStateLossOutput,
        *,
        query_count: int,
        audit: OfficialWeakBalanceAudit,
        support_ttt: tuple[TTTLossOutput, ...] = (),
    ) -> OuterLossOutput:
        """Apply detached batch/global balance coefficients to one streamed Query graph."""

        if query_count <= 0:
            raise ValueError("streamed Query balance requires a positive query_count")
        world_size = self._configured_world_size()
        answer_sum = _answer_local_sum(answer)
        answer_count = audit.answer_global_count.to(
            device=answer_sum.device, dtype=answer_sum.dtype
        )
        answer_contribution = float(query_count * world_size) * answer_sum / answer_count
        term_sums = torch.stack(
            tuple(_weak_local_sum(getattr(state, name)) for name in _TERM_NAMES)
        )
        counts = torch.stack(tuple(term.global_valid_count for term in audit.terms)).to(
            device=term_sums.device, dtype=term_sums.dtype
        )
        active = counts > 0.0
        scales = torch.stack(tuple(term.scale for term in audit.terms)).to(
            device=term_sums.device, dtype=term_sums.dtype
        )
        aligned = (
            float(query_count * world_size)
            * torch.where(active, scales, torch.ones_like(scales))
            * term_sums
            / counts.clamp_min(1.0)
            * active.to(dtype=term_sums.dtype)
        )
        state_contribution = (
            float(self.config.group_weight)
            * audit.group_guard.to(
                device=answer_contribution.device, dtype=answer_contribution.dtype
            )
            * aligned.sum()
            / float(_TERM_SLOT_COUNT)
        )
        return compose_outer_loss_terms(
            answer_after=answer_contribution,
            state_after=state_contribution,
            support_ttt=support_ttt,
        )

    def _global_sum(self, values: Tensor) -> tuple[Tensor, int]:
        if values.shape != (_STAT_VECTOR_LENGTH,) or values.dtype != torch.float64:
            raise ValueError("official-weak collective payload contract drifted")
        if self._reduce_sum is not None:
            assert self._world_size is not None
            reduced = self._reduce_sum(values.detach().clone())
            world_size = self._world_size
        elif dist.is_available() and dist.is_initialized():
            reduced = values.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        else:
            reduced = values.detach().clone()
            world_size = 1
        if (
            reduced.shape != values.shape
            or reduced.dtype != torch.float64
            or reduced.device != values.device
        ):
            raise ValueError("official-weak collective returned an invalid payload")
        return reduced, int(world_size)

    def _global_gradient_sum(self, values: Tensor) -> Tensor:
        if values.shape != (_GRAD_STAT_VECTOR_LENGTH,) or values.dtype != torch.float64:
            raise ValueError("official-weak gradient collective payload contract drifted")
        if self._reduce_sum is not None:
            reduced = self._reduce_sum(values.detach().clone())
        elif dist.is_available() and dist.is_initialized():
            reduced = values.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        else:
            reduced = values.detach().clone()
        if (
            reduced.shape != values.shape
            or reduced.dtype != torch.float64
            or reduced.device != values.device
        ):
            raise ValueError("official-weak gradient collective returned an invalid payload")
        return reduced

    def _configured_world_size(self) -> int:
        if self._world_size is not None:
            return int(self._world_size)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1


def _answer_valid_rows(answer: AnswerLossOutput) -> Tensor:
    return answer.loss.row_valid_mask.sum(dtype=torch.int64)


def _answer_local_sum(answer: AnswerLossOutput) -> Tensor:
    return answer.loss.value * _answer_valid_rows(answer).to(dtype=answer.loss.value.dtype)


def _weak_local_sum(term: OfficialWeakLossTerm) -> Tensor:
    return term.value * float(term.valid_rows)


def _sum_tensors(values: Sequence[Tensor]) -> Tensor:
    tensors = tuple(values)
    if not tensors:
        raise ValueError("cannot sum an empty loss sequence")
    return torch.stack(tensors).sum()


def _pack_stats(
    local_sums: Sequence[Tensor],
    local_counts: Sequence[Tensor | int],
    gradient_squares: Sequence[Tensor],
    gradient_counts: Sequence[int],
) -> Tensor:
    if len(local_sums) != _STAT_TERM_COUNT or len(local_counts) != _STAT_TERM_COUNT:
        raise ValueError("official-weak local statistics contract drifted")
    values: list[Tensor] = []
    for local_sum, local_count in zip(local_sums, local_counts, strict=True):
        count = (
            local_count.detach().to(device=local_sum.device, dtype=torch.float64)
            if isinstance(local_count, Tensor)
            else torch.tensor(float(local_count), dtype=torch.float64, device=local_sum.device)
        )
        values.extend(
            (
                local_sum.detach().to(dtype=torch.float64),
                count,
            )
        )
    values.extend(_pack_gradient_stats(gradient_squares, gradient_counts).unbind())
    packed = torch.stack(values)
    if packed.shape != (_STAT_VECTOR_LENGTH,):
        raise ValueError("official-weak packed statistics length drifted")
    return packed


def _unpack_stats(
    values: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if values.shape != (_STAT_VECTOR_LENGTH,):
        raise ValueError("official-weak reduced statistics length drifted")
    loss_values = values[:_LOSS_STAT_VECTOR_LENGTH]
    sums = loss_values[0::2]
    counts = loss_values[1::2]
    gradient_squares, gradient_counts = _unpack_gradient_stats(values[_LOSS_STAT_VECTOR_LENGTH:])
    return sums, counts, gradient_squares, gradient_counts


def _gradient_local_statistics(
    term_items: Sequence[Sequence[OfficialWeakLossTerm]],
    anchors: Sequence[OfficialWeakGradientAnchors],
    loss_scales: Sequence[Tensor],
) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    if (
        len(term_items) != _TERM_SLOT_COUNT
        or len(loss_scales) != _TERM_SLOT_COUNT
        or any(len(terms) != len(anchors) for terms in term_items)
    ):
        raise ValueError("official-weak activation-gradient inputs are not aligned")
    if not anchors:
        raise ValueError("official-weak activation-gradient measurement requires anchors")
    device = anchors[0].q_target.device
    squared_sums: list[Tensor] = []
    counts: list[int] = []
    for name, terms, loss_scale in zip(
        _TERM_NAMES,
        term_items,
        loss_scales,
        strict=True,
    ):
        squared = torch.zeros((), dtype=torch.float64, device=device)
        count = 0
        for term, anchor_set in zip(terms, anchors, strict=True):
            if term.valid_rows <= 0:
                continue
            anchor = anchor_set.for_term(name)
            if anchor.device != device or term.value.device != device:
                raise ValueError("official-weak gradient terms and anchors must share one device")
            count += anchor.numel()
            gradient: Tensor | None = None
            if anchor.requires_grad and term.value.requires_grad:
                scaled = loss_scale.to(device=device, dtype=term.value.dtype) * term.value
                gradient = torch.autograd.grad(
                    scaled,
                    anchor,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            if gradient is not None:
                squared = squared + gradient.detach().double().square().sum()
        squared_sums.append(squared)
        counts.append(count)
    return tuple(squared_sums), tuple(counts)


def _pack_gradient_stats(squared_sums: Sequence[Tensor], counts: Sequence[int]) -> Tensor:
    if len(squared_sums) != _TERM_SLOT_COUNT or len(counts) != _TERM_SLOT_COUNT:
        raise ValueError("official-weak gradient statistics contract drifted")
    values: list[Tensor] = []
    for squared, count in zip(squared_sums, counts, strict=True):
        if count < 0:
            raise ValueError("official-weak gradient counts must be non-negative")
        values.extend(
            (
                squared.detach().to(dtype=torch.float64),
                torch.tensor(float(count), dtype=torch.float64, device=squared.device),
            )
        )
    return torch.stack(values)


def _unpack_gradient_stats(values: Tensor) -> tuple[Tensor, Tensor]:
    if values.shape != (_GRAD_STAT_VECTOR_LENGTH,):
        raise ValueError("official-weak reduced gradient statistics length drifted")
    return values[0::2], values[1::2]


def _validate_reduced_statistics(
    sums: Tensor,
    counts: Tensor,
    gradient_squares: Tensor,
    gradient_counts: Tensor,
) -> None:
    """Validate one collective result at the explicit statistics boundary."""

    expected = (
        (sums, (_STAT_TERM_COUNT,)),
        (counts, (_STAT_TERM_COUNT,)),
        (gradient_squares, (_TERM_SLOT_COUNT,)),
        (gradient_counts, (_TERM_SLOT_COUNT,)),
    )
    if any(value.shape != shape or value.dtype != torch.float64 for value, shape in expected):
        raise ValueError("official-weak reduced statistics contract drifted")
    if len({value.device for value, _shape in expected}) != 1:
        raise ValueError("official-weak reduced statistics must share one device")
    if not bool(torch.isfinite(torch.cat((counts, gradient_counts))).all()):
        raise ValueError("official-weak reduced counts must be finite")
    if not bool(
        torch.all(counts >= 0.0)
        and torch.all(gradient_counts >= 0.0)
        and torch.allclose(counts, counts.round(), atol=1.0e-6, rtol=0.0)
        and torch.allclose(gradient_counts, gradient_counts.round(), atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("official-weak reduced counts must be non-negative integers")
    finite_gradient_squares = torch.isfinite(gradient_squares)
    if not bool(
        torch.where(
            finite_gradient_squares,
            gradient_squares.ge(0.0),
            torch.ones_like(finite_gradient_squares),
        ).all()
    ):
        raise ValueError("official-weak reduced gradient squares must be non-negative")
    if not bool(counts[0] > 0.0):
        raise ValueError("official-weak balancing requires a valid Answer row")


def _audit_float(value: Tensor) -> float:
    if value.ndim != 0 or value.requires_grad:
        raise ValueError("official-weak audit metric must be one detached scalar")
    return float(value.detach().cpu().item())


def _audit_optional_float(value: Tensor) -> float | None:
    materialized = _audit_float(value)
    return None if math.isnan(materialized) else materialized


__all__ = [
    "OfficialWeakBalanceAudit",
    "OfficialWeakBalancedBatch",
    "OfficialWeakGradientAnchors",
    "OfficialWeakOuterLossComposer",
    "OfficialWeakTermBalanceMetrics",
]
