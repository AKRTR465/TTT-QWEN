"""Distributed Answer-dominant official-weak loss composition with checkpointed EMA state."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ttt_svcbench_qwen.config import OfficialWeakBalanceConfig, OfficialWeakBalanceMode
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
_STAT_VECTOR_LENGTH = 2 * _STAT_TERM_COUNT


@dataclass(frozen=True, slots=True)
class OfficialWeakTermBalanceMetrics:
    name: str
    raw_global_mean: float | None
    scale: float | None
    aligned_global_mean: float | None
    weighted_global_mean: float | None
    global_valid_count: int
    scale_clamped: bool

    def __post_init__(self) -> None:
        if self.name not in _TERM_NAMES:
            raise ValueError(f"unknown official-weak balance term: {self.name}")
        if self.global_valid_count < 0:
            raise ValueError("official-weak global valid counts must be non-negative")
        values = (
            self.raw_global_mean,
            self.scale,
            self.aligned_global_mean,
            self.weighted_global_mean,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("official-weak balance metrics must be finite or absent")


@dataclass(frozen=True, slots=True)
class OfficialWeakBalanceAudit:
    answer_global_mean: float
    answer_global_count: int
    state_global_mean: float
    terms: tuple[OfficialWeakTermBalanceMetrics, ...]
    auxiliary_to_answer_ratio: float
    group_guard: float
    group_guard_active: bool
    ema_means: tuple[float | None, ...] = ()
    ema_update_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.answer_global_mean) or self.answer_global_mean < 0.0:
            raise ValueError("official-weak Answer mean must be finite and non-negative")
        if self.answer_global_count <= 0:
            raise ValueError("official-weak Answer global count must be positive")
        if not math.isfinite(self.state_global_mean) or self.state_global_mean < 0.0:
            raise ValueError("official-weak State mean must be finite and non-negative")
        if tuple(term.name for term in self.terms) != _TERM_NAMES:
            raise ValueError("official-weak balance audit term order drifted")
        if not math.isfinite(self.auxiliary_to_answer_ratio) or not (
            0.0 <= self.auxiliary_to_answer_ratio <= 1.0
        ):
            raise ValueError("official-weak auxiliary/Answer ratio is invalid")
        if not math.isfinite(self.group_guard) or not 0.0 <= self.group_guard <= 1.0:
            raise ValueError("official-weak group guard must be within [0, 1]")
        if self.ema_means and len(self.ema_means) != _STAT_TERM_COUNT:
            raise ValueError("official-weak EMA means must include Answer plus four terms")
        if self.ema_update_counts and (
            len(self.ema_update_counts) != _STAT_TERM_COUNT
            or any(value < 0 for value in self.ema_update_counts)
        ):
            raise ValueError("official-weak EMA update counts are invalid")

    def metrics(self) -> tuple[tuple[str, float | None], ...]:
        values: list[tuple[str, float | None]] = [
            ("loss/answer", self.answer_global_mean),
            ("loss/state", self.state_global_mean),
            ("loss/outer_total", self.answer_global_mean + self.state_global_mean),
        ]
        for term in self.terms:
            values.extend(
                (
                    (f"loss/raw/{term.name}", term.raw_global_mean),
                    (f"loss/scale/{term.name}", term.scale),
                    (f"loss/aligned/{term.name}", term.aligned_global_mean),
                    (f"loss/weighted/{term.name}", term.weighted_global_mean),
                    (f"loss/global_valid_count/{term.name}", float(term.global_valid_count)),
                    (f"loss/scale_clamped/{term.name}", float(term.scale_clamped)),
                )
            )
        values.extend(
            (
                ("loss/aux_to_answer_ratio", self.auxiliary_to_answer_ratio),
                ("loss/group_guard_active", float(self.group_guard_active)),
            )
        )
        if self.ema_means:
            for name, mean, updates in zip(
                ("answer", *_TERM_NAMES),
                self.ema_means,
                self.ema_update_counts,
                strict=True,
            ):
                values.append((f"loss/ema/{name}", mean))
                values.append((f"loss/ema_updates/{name}", float(updates)))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class OfficialWeakBalancedBatch:
    objectives: tuple[OuterLossOutput, ...]
    mean_total: Tensor
    audit: OfficialWeakBalanceAudit | None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("official-weak composition requires at least one objective")
        expected = torch.stack(tuple(item.total for item in self.objectives)).mean()
        if not torch.allclose(
            self.mean_total.detach(), expected.detach(), atol=1.0e-7, rtol=1.0e-7
        ):
            raise ValueError("official-weak batch total must equal the mean point objective")


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
        persistent = config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF
        self.register_buffer(
            "ema_values",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.float64),
            persistent=persistent,
        )
        self.register_buffer(
            "ema_valid",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.bool),
            persistent=persistent,
        )
        self.register_buffer(
            "ema_update_counts",
            torch.zeros(_STAT_TERM_COUNT, dtype=torch.int64),
            persistent=persistent,
        )

    def compose(
        self,
        answers: Sequence[AnswerLossOutput],
        states: Sequence[OfficialWeakStateLossOutput],
        *,
        support_ttt: Sequence[tuple[TTTLossOutput, ...]] | None = None,
    ) -> OfficialWeakBalancedBatch:
        answer_items = tuple(answers)
        state_items = tuple(states)
        if not answer_items or len(answer_items) != len(state_items):
            raise ValueError("official-weak Answer and State batches must be non-empty and aligned")
        supports = tuple(() for _ in answer_items) if support_ttt is None else tuple(support_ttt)
        if len(supports) != len(answer_items):
            raise ValueError("official-weak support-TTT batches must align to Query objectives")
        device = answer_items[0].loss.value.device
        if any(answer.loss.value.device != device for answer in answer_items) or any(
            state.total.device != device for state in state_items
        ):
            raise ValueError("official-weak composed losses must share one device")
        if self.config.mode is OfficialWeakBalanceMode.LEGACY_SUM:
            legacy_objectives = tuple(
                compose_outer_loss_terms(
                    answer_after=answer.loss.value,
                    state_after=state.total,
                    support_ttt=support,
                )
                for answer, state, support in zip(answer_items, state_items, supports, strict=True)
            )
            return OfficialWeakBalancedBatch(
                objectives=legacy_objectives,
                mean_total=torch.stack(tuple(item.total for item in legacy_objectives)).mean(),
                audit=None,
            )

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
        stats = _pack_stats(local_sums, local_counts)
        trace_event(
            "outer_loss_balance_pack",
            seconds=time.perf_counter() - pack_started,
            query_count=len(answer_items),
        )
        with trace_cuda_phase("outer_loss_balance_collective", payload_values=stats.numel()):
            reduced, world_size = self._global_sum(stats)
        finalize_started = time.perf_counter()
        global_sums, global_counts = _unpack_stats(reduced)
        if global_counts[0] <= 0:
            raise ValueError("official-weak balancing requires a valid Answer row")

        epsilon = float(self.config.epsilon)
        current_means = tuple(
            (
                None
                if global_count <= 0
                else global_sum.detach() / float(global_count)
            )
            for global_sum, global_count in zip(global_sums, global_counts, strict=True)
        )
        if self.config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF and self.training:
            self._update_ema(current_means)
        if (
            self.config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF
            and not bool(self.ema_valid[0].item())
        ):
            raise ValueError("EMA Answer reference is uninitialized")
        answer_mean = global_sums[0] / float(global_counts[0])
        answer_reference = (
            self.ema_values[0].detach()
            if self.config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF
            else answer_mean.detach()
        )
        scales: list[Tensor | None] = []
        aligned_means: list[Tensor | None] = []
        clamped: list[bool] = []
        for global_sum, global_count in zip(global_sums[1:], global_counts[1:], strict=True):
            term_index = len(scales) + 1
            if global_count <= 0:
                historical_scale: Tensor | None = None
                if (
                    self.config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF
                    and bool(self.ema_valid[term_index].item())
                ):
                    ratio = answer_reference / (self.ema_values[term_index].detach() + epsilon)
                    historical_scale = ratio.clamp(
                        min=float(self.config.scale_min), max=float(self.config.scale_max)
                    )
                scales.append(historical_scale)
                aligned_means.append(None)
                clamped.append(False)
                continue
            raw_mean = global_sum / float(global_count)
            term_reference = (
                self.ema_values[term_index].detach()
                if self.config.mode is OfficialWeakBalanceMode.EMA_ANSWER_REF
                else raw_mean.detach()
            )
            ratio = answer_reference / (term_reference + epsilon)
            scale = ratio.clamp(min=float(self.config.scale_min), max=float(self.config.scale_max))
            scales.append(scale)
            aligned_means.append(scale * raw_mean)
            clamped.append(not bool(torch.isclose(scale, ratio).item()))

        auxiliary_mean = sum(
            (value for value in aligned_means if value is not None),
            start=answer_mean * 0.0,
        ) / float(_TERM_SLOT_COUNT)
        if bool((auxiliary_mean > 0.0).item()):
            group_guard = torch.minimum(
                torch.ones_like(answer_mean),
                answer_mean / auxiliary_mean.clamp_min(epsilon),
            )
        else:
            group_guard = torch.ones_like(answer_mean)
        query_count = len(answer_items)
        objectives: list[OuterLossOutput] = []
        for item_index, (_answer, support) in enumerate(zip(answer_items, supports, strict=True)):
            answer_contribution = (
                float(query_count * world_size) * answer_sums[item_index] / float(global_counts[0])
            )
            aligned_contributions: list[Tensor] = []
            for term_index, scale in enumerate(scales):
                local_sum = term_sums[term_index][item_index]
                global_count = global_counts[term_index + 1]
                contribution = local_sum * 0.0
                if scale is not None and global_count > 0:
                    contribution = (
                        float(query_count * world_size)
                        * scale.to(dtype=local_sum.dtype)
                        * local_sum
                        / float(global_count)
                    )
                aligned_contributions.append(contribution)
            state_contribution = (
                float(self.config.group_weight)
                * group_guard.to(dtype=answer_contribution.dtype)
                * torch.stack(tuple(aligned_contributions)).sum()
                / float(_TERM_SLOT_COUNT)
            )
            objectives.append(
                compose_outer_loss_terms(
                    answer_after=answer_contribution,
                    state_after=state_contribution,
                    support_ttt=support,
                )
            )

        weighted_factor = float(self.config.group_weight) * float(group_guard.item())
        term_metrics = tuple(
            OfficialWeakTermBalanceMetrics(
                name=name,
                raw_global_mean=(
                    None if global_count <= 0 else float((global_sum / float(global_count)).item())
                ),
                scale=None if scale is None else float(scale.item()),
                aligned_global_mean=None if aligned is None else float(aligned.item()),
                weighted_global_mean=(
                    None
                    if aligned is None
                    else weighted_factor * float(aligned.item()) / float(_TERM_SLOT_COUNT)
                ),
                global_valid_count=global_count,
                scale_clamped=was_clamped,
            )
            for name, global_sum, global_count, scale, aligned, was_clamped in zip(
                _TERM_NAMES,
                global_sums[1:],
                global_counts[1:],
                scales,
                aligned_means,
                clamped,
                strict=True,
            )
        )
        answer_mean_value = float(answer_mean.item())
        weighted_auxiliary = weighted_factor * float(auxiliary_mean.item())
        ratio_value = (
            0.0 if answer_mean_value <= epsilon else weighted_auxiliary / answer_mean_value
        )
        if ratio_value > float(self.config.group_weight) + 1.0e-6:
            raise ValueError("official-weak group guard failed to preserve Answer dominance")
        audit = OfficialWeakBalanceAudit(
            answer_global_mean=answer_mean_value,
            answer_global_count=global_counts[0],
            state_global_mean=weighted_auxiliary,
            terms=term_metrics,
            auxiliary_to_answer_ratio=min(float(self.config.group_weight), ratio_value),
            group_guard=float(group_guard.item()),
            group_guard_active=bool((group_guard < 1.0).item()),
            ema_means=self._ema_means_for_audit(),
            ema_update_counts=tuple(int(value.item()) for value in self.ema_update_counts),
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
    def _update_ema(self, current_means: Sequence[Tensor | None]) -> None:
        if len(current_means) != _STAT_TERM_COUNT:
            raise ValueError("official-weak EMA update shape drifted")
        beta = float(self.config.ema_beta)
        for index, current in enumerate(current_means):
            if current is None:
                continue
            value = current.to(device=self.ema_values.device, dtype=torch.float64)
            if not bool(self.ema_valid[index].item()):
                self.ema_values[index].copy_(value)
                self.ema_valid[index] = True
            else:
                self.ema_values[index].mul_(beta).add_(value, alpha=1.0 - beta)
            self.ema_update_counts[index] += 1

    def _ema_means_for_audit(self) -> tuple[float | None, ...]:
        if self.config.mode is not OfficialWeakBalanceMode.EMA_ANSWER_REF:
            return ()
        return tuple(
            float(value.item()) if bool(valid.item()) else None
            for value, valid in zip(self.ema_values, self.ema_valid, strict=True)
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
        answer_contribution = (
            float(query_count * world_size) * answer_sum / float(audit.answer_global_count)
        )
        aligned: list[Tensor] = []
        for name, metrics in zip(_TERM_NAMES, audit.terms, strict=True):
            term = getattr(state, name)
            contribution = term.value * 0.0
            if metrics.scale is not None and metrics.global_valid_count > 0:
                contribution = (
                    float(query_count * world_size)
                    * float(metrics.scale)
                    * _weak_local_sum(term)
                    / float(metrics.global_valid_count)
                )
            aligned.append(contribution)
        state_contribution = (
            float(self.config.group_weight)
            * float(audit.group_guard)
            * torch.stack(tuple(aligned)).sum()
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
            or not bool(torch.isfinite(reduced).all().item())
        ):
            raise ValueError("official-weak collective returned an invalid payload")
        return reduced, int(world_size)

    def _configured_world_size(self) -> int:
        if self._world_size is not None:
            return int(self._world_size)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1


def _answer_valid_rows(answer: AnswerLossOutput) -> int:
    return int(answer.loss.row_valid_mask.sum(dtype=torch.int64).item())


def _answer_local_sum(answer: AnswerLossOutput) -> Tensor:
    return answer.loss.value * float(_answer_valid_rows(answer))


def _weak_local_sum(term: OfficialWeakLossTerm) -> Tensor:
    return term.value * float(term.valid_rows)


def _sum_tensors(values: Sequence[Tensor]) -> Tensor:
    tensors = tuple(values)
    if not tensors:
        raise ValueError("cannot sum an empty loss sequence")
    return torch.stack(tensors).sum()


def _pack_stats(local_sums: Sequence[Tensor], local_counts: Sequence[int]) -> Tensor:
    if len(local_sums) != _STAT_TERM_COUNT or len(local_counts) != _STAT_TERM_COUNT:
        raise ValueError("official-weak local statistics contract drifted")
    values: list[Tensor] = []
    for local_sum, local_count in zip(local_sums, local_counts, strict=True):
        values.extend(
            (
                local_sum.detach().to(dtype=torch.float64),
                torch.tensor(float(local_count), dtype=torch.float64, device=local_sum.device),
            )
        )
    return torch.stack(values)


def _unpack_stats(values: Tensor) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    sums = tuple(values[index] for index in range(0, _STAT_VECTOR_LENGTH, 2))
    counts: list[int] = []
    for index in range(1, _STAT_VECTOR_LENGTH, 2):
        raw = float(values[index].item())
        count = int(round(raw))
        if count < 0 or not math.isclose(raw, float(count), abs_tol=1.0e-6):
            raise ValueError("official-weak reduced counts must be non-negative integers")
        counts.append(count)
    return sums, tuple(counts)


__all__ = [
    "OfficialWeakBalanceAudit",
    "OfficialWeakBalancedBatch",
    "OfficialWeakOuterLossComposer",
    "OfficialWeakTermBalanceMetrics",
]
