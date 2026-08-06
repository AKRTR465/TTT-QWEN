"""Distributed Answer-dominant official-weak loss composition with checkpointed EMA state."""

from __future__ import annotations

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
    compose_outer_loss_terms,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
)

_TERM_NAMES = ("task", "operator", "retrieval", "time")
_TERM_SLOT_COUNT = len(_TERM_NAMES)
_STAT_TERM_COUNT = 1 + _TERM_SLOT_COUNT
_LOSS_STAT_VECTOR_LENGTH = 2 * _STAT_TERM_COUNT


@dataclass(frozen=True, slots=True)
class OfficialWeakGradientAnchors:
    """Activation reference planes used to measure the four weak-loss gradients."""

    q_target: Tensor
    q_operator: Tensor
    q_time: Tensor

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
    """Detached per-term coefficients replayed by the streamed A5 composition path."""

    name: str
    scale: Tensor
    loss_scale: Tensor
    global_valid_count: Tensor
    raw_gradient_rms: Tensor
    ema_gradient_rms: Tensor


@dataclass(frozen=True, slots=True)
class OfficialWeakBalanceAudit:
    answer_global_count: Tensor
    terms: tuple[OfficialWeakTermBalanceMetrics, ...]
    group_guard: Tensor


@dataclass(frozen=True, slots=True)
class OfficialWeakBalancedBatch:
    objectives: tuple[OuterLossOutput, ...]
    mean_total: Tensor
    audit: OfficialWeakBalanceAudit | None


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
        return result

    def compose(
        self,
        answers: Sequence[AnswerLossOutput],
        states: Sequence[OfficialWeakStateLossOutput],
        *,
        gradient_anchors: Sequence[OfficialWeakGradientAnchors] | None = None,
        measure_gradients: bool = True,
        statistical_weights: Sequence[float] | None = None,
    ) -> OfficialWeakBalancedBatch:
        answer_items = tuple(answers)
        state_items = tuple(states)
        anchors = () if gradient_anchors is None else tuple(gradient_anchors)
        weights = _normalize_statistical_weights(
            statistical_weights,
            count=len(answer_items),
        )
        device = answer_items[0].loss.value.device
        answer_sums = tuple(
            _answer_local_sum(answer) * weight
            for answer, weight in zip(answer_items, weights, strict=True)
        )
        answer_counts = tuple(
            _answer_valid_rows(answer) * int(weight)
            for answer, weight in zip(answer_items, weights, strict=True)
        )
        term_items = tuple(
            tuple(getattr(state, name) for state in state_items) for name in _TERM_NAMES
        )
        term_sums = tuple(
            tuple(
                _weak_local_sum(term) * weight
                for term, weight in zip(terms, weights, strict=True)
            )
            for terms in term_items
        )
        term_counts = tuple(
            tuple(
                term.valid_rows * int(weight)
                for term, weight in zip(terms, weights, strict=True)
            )
            for terms in term_items
        )
        local_sums = (
            _sum_tensors(answer_sums),
            *(_sum_tensors(sums) for sums in term_sums),
        )
        local_counts = (
            sum(answer_counts),
            *(sum(counts) for counts in term_counts),
        )
        prior_loss_scales = self._prior_loss_scales(device)
        if measure_gradients:
            local_gradient_squares, local_gradient_counts = _gradient_local_statistics(
                term_items,
                anchors,
                tuple(prior_loss_scales.unbind()),
                statistical_weights=weights,
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
        with trace_cuda_phase("outer_loss_balance_collective", payload_values=stats.numel()):
            reduced, world_size = self._global_sum(stats)
        global_sums, global_counts, global_gradient_squares, global_gradient_counts = _unpack_stats(
            reduced
        )
        epsilon = float(self.config.epsilon)
        loss_valid = global_counts > 0.0
        gradient_valid = global_gradient_counts > 0.0
        loss_update_valid = loss_valid & torch.isfinite(global_sums)
        gradient_update_valid = gradient_valid & torch.isfinite(global_gradient_squares)
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
        prior_gradient_scales = self._prior_gradient_scales(global_counts[1:], device)
        history_valid = self.ema_valid[0].to(device) & self.ema_valid[1:].to(device)
        history_valid &= self.gradient_ema_valid.to(device)
        unbounded_scales = prior_loss_scales * prior_gradient_scales
        bounded_scales = unbounded_scales.clamp(
            min=float(self.config.scale_min),
            max=float(self.config.scale_max),
        )
        scales = torch.where(history_valid, bounded_scales, torch.ones_like(bounded_scales))
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
        for item_index, _answer in enumerate(answer_items):
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
                )
            )

        if self.training:
            self._update_ema(current_means, loss_update_valid)
            if measure_gradients:
                self._update_gradient_ema(current_gradient_rms, gradient_update_valid)
        gradient_ema_rms = self._gradient_ema_for_audit()
        term_gradient_ema_rms = torch.stack(gradient_ema_rms)
        term_metrics = tuple(
            OfficialWeakTermBalanceMetrics(
                name=name,
                scale=torch.where(active, scale, nan).detach().clone(),
                loss_scale=torch.where(active, loss_scale, nan).detach().clone(),
                global_valid_count=global_count.detach().clone(),
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
                raw_gradient_rms,
                ema_gradient_rms,
                active,
                raw_gradient_valid,
            ) in zip(
                _TERM_NAMES,
                global_counts[1:].unbind(),
                scales.unbind(),
                prior_loss_scales.unbind(),
                current_gradient_rms.unbind(),
                term_gradient_ema_rms.unbind(),
                loss_valid[1:].unbind(),
                gradient_valid.unbind(),
                strict=True,
            )
        )
        audit = OfficialWeakBalanceAudit(
            answer_global_count=global_counts[0].detach().clone(),
            terms=term_metrics,
            group_guard=group_guard.detach().clone(),
        )
        objective_tuple = tuple(objectives)
        return OfficialWeakBalancedBatch(
            objectives=objective_tuple,
            mean_total=torch.stack(tuple(item.total for item in objective_tuple)).mean(),
            audit=audit,
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _update_ema(self, current_means: Tensor, valid: Tensor) -> None:
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

    def _prior_loss_scales(self, device: torch.device) -> Tensor:
        """Stage 1 of ema_answer_ref: align each weak-term loss EMA to the Answer scale."""

        values = self.ema_values.detach().to(device=device, dtype=torch.float64)
        history_valid = self.ema_valid[0].to(device) & self.ema_valid[1:].to(device)
        epsilon = float(self.config.epsilon)
        ratio = values[0] / (values[1:] + epsilon)
        bounded = ratio.clamp(
            min=float(self.config.scale_min),
            max=float(self.config.scale_max),
        )
        return torch.where(history_valid, bounded, torch.ones_like(bounded))

    def _prior_gradient_scales(
        self,
        active_counts: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Stage 2 of ema_answer_ref: balance the four terms by their gradient-RMS EMA."""

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
        return torch.where(active, bounded, torch.ones_like(bounded))

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def _update_gradient_ema(self, current_rms: Tensor, valid: Tensor) -> None:
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
        *,
        statistical_weights: Sequence[float] | None = None,
    ) -> OfficialWeakBalancedBatch:
        """Select streamed-A5 coefficients without differentiating no-grad calibration graphs."""

        return self.compose(
            answers,
            states,
            measure_gradients=False,
            statistical_weights=statistical_weights,
        )

    def measure_streamed_gradients(
        self,
        state: OfficialWeakStateLossOutput,
        anchors: OfficialWeakGradientAnchors,
        audit: OfficialWeakBalanceAudit,
        *,
        statistical_weight: float = 1.0,
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
            statistical_weights=_normalize_statistical_weights(
                (statistical_weight,),
                count=1,
            ),
        )
        return _pack_gradient_stats(squares, counts)

    def commit_streamed_gradients(
        self,
        local_statistics: Sequence[Tensor],
        audit: OfficialWeakBalanceAudit,
    ) -> OfficialWeakBalanceAudit:
        """Commit fixed gradient stats after all streamed Query graphs are measured."""

        values = tuple(local_statistics)
        packed = torch.stack(values).sum(dim=0)
        with trace_cuda_phase("outer_gradient_balance_collective", payload_values=packed.numel()):
            reduced = self._global_gradient_sum(packed)
        squares, counts = _unpack_gradient_stats(reduced)
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
        return replace(audit, terms=terms)

    def compose_one_from_audit(
        self,
        answer: AnswerLossOutput,
        state: OfficialWeakStateLossOutput,
        *,
        query_count: int,
        audit: OfficialWeakBalanceAudit,
    ) -> OuterLossOutput:
        """Apply detached batch/global balance coefficients to one streamed Query graph."""

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
        )

    def _global_sum(self, values: Tensor) -> tuple[Tensor, int]:
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
        return reduced, int(world_size)

    def _global_gradient_sum(self, values: Tensor) -> Tensor:
        if self._reduce_sum is not None:
            reduced = self._reduce_sum(values.detach().clone())
        elif dist.is_available() and dist.is_initialized():
            reduced = values.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        else:
            reduced = values.detach().clone()
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


def _normalize_statistical_weights(
    values: Sequence[float] | None,
    *,
    count: int,
) -> tuple[float, ...]:
    weights = (1.0,) * count if values is None else tuple(values)
    return tuple(float(value) for value in weights)


def _sum_tensors(values: Sequence[Tensor]) -> Tensor:
    return torch.stack(tuple(values)).sum()


def _pack_stats(
    local_sums: Sequence[Tensor],
    local_counts: Sequence[Tensor | int],
    gradient_squares: Sequence[Tensor],
    gradient_counts: Sequence[int],
) -> Tensor:
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
    return torch.stack(values)


def _unpack_stats(
    values: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    loss_values = values[:_LOSS_STAT_VECTOR_LENGTH]
    sums = loss_values[0::2]
    counts = loss_values[1::2]
    gradient_squares, gradient_counts = _unpack_gradient_stats(values[_LOSS_STAT_VECTOR_LENGTH:])
    return sums, counts, gradient_squares, gradient_counts


def _gradient_local_statistics(
    term_items: Sequence[Sequence[OfficialWeakLossTerm]],
    anchors: Sequence[OfficialWeakGradientAnchors],
    loss_scales: Sequence[Tensor],
    *,
    statistical_weights: Sequence[float] | None = None,
) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    weights = _normalize_statistical_weights(
        statistical_weights,
        count=len(anchors),
    )
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
        for term, anchor_set, weight in zip(terms, anchors, weights, strict=True):
            anchor = anchor_set.for_term(name)
            locally_valid = term.valid_rows > 0
            statistically_valid = locally_valid and weight == 1.0
            if statistically_valid:
                count += anchor.numel()
            gradient: Tensor | None = None
            if anchor.requires_grad:
                source = (
                    term.value
                    if term.value.requires_grad
                    else anchor.sum().to(dtype=term.value.dtype) * 0.0
                )
                scaled = loss_scale.to(device=device, dtype=source.dtype) * source
                gradient = torch.autograd.grad(
                    scaled,
                    anchor,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            if statistically_valid and gradient is not None:
                squared = squared + gradient.detach().double().square().sum()
        squared_sums.append(squared)
        counts.append(count)
    return tuple(squared_sums), tuple(counts)


def _pack_gradient_stats(squared_sums: Sequence[Tensor], counts: Sequence[int]) -> Tensor:
    values: list[Tensor] = []
    for squared, count in zip(squared_sums, counts, strict=True):
        values.extend(
            (
                squared.detach().to(dtype=torch.float64),
                torch.tensor(float(count), dtype=torch.float64, device=squared.device),
            )
        )
    return torch.stack(values)


def _unpack_gradient_stats(values: Tensor) -> tuple[Tensor, Tensor]:
    return values[0::2], values[1::2]


__all__ = [
    "OfficialWeakBalanceAudit",
    "OfficialWeakBalancedBatch",
    "OfficialWeakGradientAnchors",
    "OfficialWeakOuterLossComposer",
    "OfficialWeakTermBalanceMetrics",
]
