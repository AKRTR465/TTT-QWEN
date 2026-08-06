"""Causal Meta-TTT episode orchestration and slot-memory write audits.

Inputs: resettable model/runtime factories, causal Support/Query chunks, typed query labels,
and the frozen delta-rule memory-write contract.
Outputs: a Query-only after-update objective, per-video next-only memory generations,
before/after Query metrics, and bounded graph/lifecycle audits.
Forbidden: Support labels, batch-scalar inner updates, in-place memory mutation, inner
optimizer state, cross-video runtime reuse, observe-after-prefill, or carrying
differentiable runtime snapshots.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    AssociativeTTTIntermediates,
    FastMemoryState,
    FastTTTForwardAudit,
    MemoryWriteBatch,
    SlotStateView,
    apply_memory_writes,
    deferred_fast_vjp_loss,
    make_query_proxy_fast_state,
    truncate_memory_states,
)
from ttt_svcbench_qwen.input_composer import map_teacher_forced_targets
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    AnswerLossOutput,
    OuterLossInput,
    OuterLossOutput,
    ReaderCountMetricInput,
    StateLossInput,
    StateLossOutput,
    compute_answer_loss,
    compute_outer_loss,
    compute_state_loss,
)
from ttt_svcbench_qwen.model import (
    BatchRuntimeState,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    PreparedQueryOutput,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
    query_dropout_seed,
    query_reuse_key,
)
from ttt_svcbench_qwen.observation_heads import ObservationOutputs
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakGradientAnchors,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.query_encoder import QueryEncoderOutput
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.stage_a_targets import (
    O2DedupContext,
    OfficialWeakStateLossOutput,
    OfficialWeakTargetBuilder,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageASupervisionBatch,
    answer_query_request,
)


class FastStateController(Protocol):
    """Subset of :class:`FastTTTAdapter` needed by a managed meta episode."""

    last_audit: FastTTTForwardAudit | None

    def reset_fast_state(
        self,
        state: FastMemoryState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastMemoryState: ...

    def use_fast_state(
        self,
        state: FastMemoryState | Sequence[FastMemoryState],
    ) -> AbstractContextManager[object]: ...

    def prepare_write(
        self,
        intermediates: AssociativeTTTIntermediates,
        spatial: SlotStateView,
    ) -> MemoryWriteBatch: ...

    def collect_meta_fast_parameters(self) -> tuple[nn.Parameter, nn.Parameter]: ...

    def collect_associative_parameters(self) -> tuple[nn.Parameter, ...]: ...


class EpisodeRuntimeResetter(Protocol):
    def __call__(self, owner: RuntimeOwner) -> BatchRuntimeState: ...


@dataclass(frozen=True, slots=True)
class MetaCausalChunk:
    """One model observation plus independently audited label-free runtime payload."""

    request: ObservationChunkRequest
    start_time: float
    end_time: float
    query_input: RuntimeQueryInput


@dataclass(frozen=True, slots=True)
class MetaTTTQueryPoint:
    """A later causal observation and labels exposed only after model prefill."""

    chunk: MetaCausalChunk
    query_time: float
    answer: StageAEpisodeAnswerInputs
    supervision: StageASupervisionBatch
    task_name: str
    case_id: str


@dataclass(frozen=True, slots=True)
class MetaTTTEpisode:
    owner: RuntimeOwner
    support_chunks: tuple[MetaCausalChunk, ...]
    query_points: tuple[MetaTTTQueryPoint, ...]
    seed: int
    prewarm_chunk: MetaCausalChunk | None = None
    segment_lengths: tuple[int, ...] = ()
    segment_query_counts: tuple[int, ...] = ()
    query_roles: tuple[str, ...] = ()
    query_weights: tuple[float, ...] = ()
    diagnostic_query_count: int = 0
    insufficient_inter_query_gap: bool = False

    def __post_init__(self) -> None:
        if not self.segment_query_counts and len(self.segment_lengths) == len(self.query_points):
            object.__setattr__(
                self,
                "segment_query_counts",
                (1,) * len(self.segment_lengths),
            )


@dataclass(frozen=True, slots=True)
class MetaQueryLossInput:
    answer: AnswerLossInput
    state: StateLossInput | OfficialWeakStateLossOutput


class MetaQueryLossBuilder(Protocol):
    def __call__(
        self,
        output: StateTTTModelOutput,
        *,
        answer: StageAEpisodeAnswerInputs,
        supervision: StageASupervisionBatch,
        dedup: O2DedupContext | None = None,
    ) -> MetaQueryLossInput: ...


class StageAQueryLossBuilder:
    """Reuse P15's typed label join at the post-prefill Query boundary."""

    def __init__(self, target_builder: StageATargetBuilder | None = None) -> None:
        self.target_builder = target_builder or StageATargetBuilder()
        self.official_weak_builder = OfficialWeakTargetBuilder()

    def __call__(
        self,
        output: StateTTTModelOutput,
        *,
        answer: StageAEpisodeAnswerInputs,
        supervision: StageASupervisionBatch,
        dedup: O2DedupContext | None = None,
    ) -> MetaQueryLossInput:
        mapped = map_teacher_forced_targets(
            composed_input=output.composed,
            source_input_ids=answer.base_input_ids,
            source_attention_mask=answer.base_attention_mask,
            source_labels=supervision.answer.base_labels,
            source_number_token_mask=supervision.answer.base_number_token_mask,
        )
        device = output.answer_logits.device
        reader_counts = torch.full((len(output.reader),), -100, dtype=torch.int64, device=device)
        reader_valid = torch.zeros(len(output.reader), dtype=torch.bool, device=device)
        for row, result in enumerate(output.reader):
            exact_count = getattr(result, "exact_count", None)
            if type(exact_count) is int:
                reader_counts[row] = exact_count
                reader_valid[row] = True
        count_label_valid = torch.tensor(
            [
                provenance is not TargetProvenance.MISSING
                for provenance in supervision.answer.count_provenance
            ],
            dtype=torch.bool,
            device=device,
        )
        observations = cast(ObservationOutputs, output.observations)
        retrieval = cast(RetrieverOutput, output.retrieval)
        if supervision.official_weak:
            state: StateLossInput | OfficialWeakStateLossOutput = self.official_weak_builder(
                observations,
                output.query,
                retrieval,
                supervision.official_weak,
                dedup=dedup,
            )
        else:
            assert supervision.state is not None
            state = self.target_builder(
                observations,
                output.query,
                retrieval,
                supervision.state,
            )
        return MetaQueryLossInput(
            answer=AnswerLossInput(
                logits=output.answer_logits,
                labels=mapped.labels,
                number_token_mask=mapped.number_token_mask,
                reader_counts=ReaderCountMetricInput(
                    predicted_counts=reader_counts,
                    target_counts=supervision.answer.target_counts.to(device),
                    valid_mask=reader_valid & count_label_valid,
                ),
            ),
            state=state,
        )


@dataclass(frozen=True, slots=True)
class QueryMetricSnapshot:
    metrics: tuple[tuple[str, float | None], ...]

    def value(self, name: str) -> float | None:
        return dict(self.metrics)[name]


@dataclass(frozen=True, slots=True)
class MetaQueryObjective:
    answer: AnswerLossOutput
    state: StateLossOutput | OfficialWeakStateLossOutput
    outer: OuterLossOutput
    metrics: QueryMetricSnapshot
    gradient_anchors: OfficialWeakGradientAnchors


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeAudit:
    """Bounded-memory evidence for an otherwise unbounded numeric memory trajectory."""

    associative_valid_count: int
    loss_weight: float
    query_count: int
    segment_count: int
    write_count: int
    skip_count: int
    readout_target_cosine_mean: float


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeOutput:
    """Detached logging values returned after all segment backward calls have completed."""

    total: Tensor
    query_loss: Tensor
    final_fast_states: tuple[FastMemoryState, ...]
    final_runtime: BatchRuntimeState
    audit: TruncatedMetaTTTEpisodeAudit


@dataclass(slots=True)
class _Trajectory:
    runtime: BatchRuntimeState

    @property
    def fast_states(self) -> tuple[FastMemoryState, ...]:
        return self.runtime.fast_states

    @fast_states.setter
    def fast_states(self, values: tuple[FastMemoryState, ...]) -> None:
        self.runtime = self.runtime.with_fast_states(values)


class MetaTTTEpisodeRunner:
    """Run the production A5 trajectory with bounded truncated graph segments."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        model: StateTTTModel,
        fast_controller: FastStateController,
        runtime_resetter: EpisodeRuntimeResetter,
        query_loss_builder: MetaQueryLossBuilder | None = None,
        query_encoder_reuse: bool = False,
        query_activation_offload: bool = False,
        outer_composer: OfficialWeakOuterLossComposer | None = None,
        adaptation_mode: str = "meta_ttt",
    ) -> None:
        del adaptation_mode
        self.config = config
        self.model = model
        self.fast_controller = fast_controller
        self.runtime_resetter = runtime_resetter
        self.query_loss_builder = query_loss_builder or StageAQueryLossBuilder()
        self.query_encoder_reuse = query_encoder_reuse
        self.query_activation_offload = query_activation_offload
        self.outer_composer = outer_composer or OfficialWeakOuterLossComposer(
            config.loss.official_weak_balance
        )
        self.last_balance_audit: OfficialWeakBalanceAudit | None = None

    def run_truncated(
        self,
        episode: MetaTTTEpisode,
        *,
        backward: Callable[[Tensor, bool], None] | None = None,
        backward_gradient_scale: float = 1.0,
        episode_loss_weight: float = 1.0,
    ) -> TruncatedMetaTTTEpisodeOutput:
        """Run one Query-aligned deferred-VJP closure per bounded Support segment."""

        episode_weight = float(episode_loss_weight)
        backward_fn = backward or _plain_backward
        self.model.train()
        adapted = self._reset_trajectory(episode.owner, differentiable=True)
        support_offset = 0
        device = adapted.fast_states[0].m.device
        query_loss_detached = torch.zeros((), dtype=torch.float32, device=device)
        pre_write_cosine_sum = 0.0
        write_attempt_total = 0
        write_success_total = 0
        support_lifecycle = PrefillLifecycle(episode.owner)

        prewarm = cast(MetaCausalChunk, episode.prewarm_chunk)
        first_segment_query: PreparedQueryOutput | None = None
        prewarm_query: PreparedQueryOutput | None = None
        if self.query_encoder_reuse:
            first_support = episode.support_chunks[0]
            if query_reuse_key(prewarm.request.query_input) == query_reuse_key(
                first_support.request.query_input
            ):
                first_segment_query = self._prepare_query(
                    first_support,
                    adapted,
                    with_grad=True,
                )
                prewarm_query = (
                    first_segment_query.detached()
                    if isinstance(first_segment_query.value, QueryEncoderOutput)
                    else first_segment_query
                )
        prewarm_observation = self._observe(
            prewarm,
            adapted,
            support_lifecycle,
            seed=episode.seed,
            with_grad=False,
            prepared_query=prewarm_query,
        )
        adapted.runtime = _runtime_from_observation(prewarm_observation, episode.owner)
        del prewarm_observation

        query_offset = 0
        for segment_index, (segment_length, segment_query_count) in enumerate(
            zip(
                episode.segment_lengths,
                episode.segment_query_counts,
                strict=True,
            )
        ):
            queries = episode.query_points[query_offset : query_offset + segment_query_count]
            query_weights = episode.query_weights[query_offset : query_offset + segment_query_count]
            active_segment = episode.support_chunks[
                support_offset : support_offset + segment_length
            ]
            segment_query = first_segment_query if segment_index == 0 else None
            for segment_offset, chunk in enumerate(active_segment):
                support_index = support_offset + segment_offset
                if self.query_encoder_reuse and segment_query is None:
                    segment_query = self._prepare_query(
                        chunk,
                        adapted,
                        with_grad=True,
                    )
                observation = self._observe(
                    chunk,
                    adapted,
                    support_lifecycle,
                    seed=episode.seed + support_index + 1,
                    with_grad=True,
                    prepared_query=segment_query,
                )
                adapted.runtime = _runtime_from_observation(observation, episode.owner)
                intermediates = observation.soft_intermediates.fast_associative
                spatial = observation.soft_intermediates.spatial
                if intermediates is None or spatial is None:
                    del observation
                    continue
                write_batch = self.fast_controller.prepare_write(intermediates, spatial)
                results = apply_memory_writes(
                    fast_states=adapted.fast_states,
                    batch=write_batch,
                )
                adapted.fast_states = tuple(result.fast_state for result in results)
                write_attempt_total += len(results)
                for result in results:
                    if result.did_write:
                        write_success_total += 1
                        pre_write_cosine_sum += result.pre_write_cosine_mean
                del results, write_batch, intermediates, spatial, observation

            query_runtime_snapshot = adapted.runtime
            # Pre-query snapshot: the dedup base must not see the query chunk's own
            # commit, or the current slots would match themselves and zero the novelty.
            query_dedup = O2DedupContext.from_identity_states(
                query_runtime_snapshot.identity_bank_states
            )
            authoritative_fast_states = query_runtime_snapshot.fast_states
            accumulated_gradients: tuple[Tensor, ...] | None = None
            for bundle_offset, (query, query_weight) in enumerate(
                zip(queries, query_weights, strict=True)
            ):
                global_query_index = query_offset + bundle_offset
                balance_audit: OfficialWeakBalanceAudit | None = None
                if all(query.supervision.official_weak) or bool(
                    getattr(
                        self.query_loss_builder,
                        "streamed_balance_calibration",
                        False,
                    )
                ):
                    adapted.runtime = _fork_retrieval_runtime(query_runtime_snapshot)
                    calibration_lifecycle = PrefillLifecycle(episode.owner)
                    calibration_prepared_query = (
                        self._prepare_query(query.chunk, adapted, with_grad=False)
                        if self.query_encoder_reuse
                        else None
                    )
                    calibration_observation = self._observe(
                        query.chunk,
                        adapted,
                        calibration_lifecycle,
                        seed=episode.seed + 10_000 + global_query_index,
                        with_grad=False,
                        prepared_query=calibration_prepared_query,
                    )
                    calibration_output = self._answer(
                        query,
                        calibration_observation,
                        calibration_lifecycle,
                        fast_states=adapted.fast_states,
                        with_grad=False,
                    )
                    self._balance_query_objectives(
                        (self._query_objective(query, calibration_output, dedup=query_dedup),),
                        calibration=True,
                        statistical_weight=episode_weight,
                    )
                    balance_audit = self.last_balance_audit
                    adapted.runtime = query_runtime_snapshot
                    del (
                        calibration_output,
                        calibration_observation,
                        calibration_prepared_query,
                        calibration_lifecycle,
                    )

                proxy_states = tuple(
                    make_query_proxy_fast_state(state) for state in authoritative_fast_states
                )
                query_trajectory = _Trajectory(
                    _fork_retrieval_runtime(query_runtime_snapshot).with_fast_states(proxy_states)
                )
                query_lifecycle = PrefillLifecycle(episode.owner)
                prepared_query = (
                    self._prepare_query(query.chunk, query_trajectory, with_grad=True)
                    if self.query_encoder_reuse
                    else None
                )
                with contextlib.nullcontext():
                    observation = self._observe(
                        query.chunk,
                        query_trajectory,
                        query_lifecycle,
                        seed=episode.seed + 10_000 + global_query_index,
                        with_grad=True,
                        prepared_query=prepared_query,
                    )
                    output = self._answer(
                        query,
                        observation,
                        query_lifecycle,
                        fast_states=proxy_states,
                        with_grad=True,
                    )
                objective = self._query_objective(query, output, dedup=query_dedup)
                balanced_outer: OuterLossOutput | None = None
                gradient_statistics: Tensor | None = None
                if balance_audit is not None:
                    gradient_statistics = self.outer_composer.measure_streamed_gradients(
                        cast(OfficialWeakStateLossOutput, objective.state),
                        objective.gradient_anchors,
                        balance_audit,
                        statistical_weight=episode_weight,
                    )
                    balanced_outer = self.outer_composer.compose_one_from_audit(
                        objective.answer,
                        cast(OfficialWeakStateLossOutput, objective.state),
                        query_count=1,
                        audit=balance_audit,
                    )
                    objective = replace(objective, outer=balanced_outer)
                query_loss = episode_weight * query_weight * objective.outer.outer
                query_loss_detached = query_loss_detached + query_loss.detach().to(torch.float32)
                backward_fn(query_loss, False)
                captured_gradients = _capture_query_proxy_gradients(
                    proxy_states,
                    backward_gradient_scale=float(backward_gradient_scale),
                )
                clipped_gradients = _clip_query_proxy_gradients(
                    captured_gradients,
                    max_norm=self.config.a5.query_meta_gradient.max_norm,
                    epsilon=self.config.a5.query_meta_gradient.epsilon,
                )
                if accumulated_gradients is None:
                    accumulated_gradients = clipped_gradients
                else:
                    accumulated_gradients = tuple(
                        total + current
                        for total, current in zip(
                            accumulated_gradients,
                            clipped_gradients,
                            strict=True,
                        )
                    )
                if balance_audit is not None and gradient_statistics is not None:
                    balance_audit = self.outer_composer.commit_streamed_gradients(
                        (gradient_statistics,),
                        balance_audit,
                    )
                    self.last_balance_audit = balance_audit
                for state in proxy_states:
                    state.m.grad = None
                adapted.runtime = query_runtime_snapshot
                del (
                    captured_gradients,
                    clipped_gradients,
                    gradient_statistics,
                    query_loss,
                    objective,
                    output,
                    observation,
                    prepared_query,
                    query_lifecycle,
                    query_trajectory,
                    proxy_states,
                    balanced_outer,
                )

            if accumulated_gradients is not None:
                deferred_vjp = deferred_fast_vjp_loss(
                    authoritative_fast_states,
                    accumulated_gradients,
                )
                backward_fn(deferred_vjp, False)
                del deferred_vjp
            adapted.fast_states = truncate_memory_states(adapted.fast_states)
            support_offset += segment_length
            query_offset += segment_query_count
            del accumulated_gradients, segment_query

        detached_query = query_loss_detached.detach().clone()
        detached_total = detached_query.detach().clone()
        written = write_success_total
        episode_audit = TruncatedMetaTTTEpisodeAudit(
            associative_valid_count=written,
            loss_weight=episode_weight,
            query_count=len(episode.query_points),
            segment_count=len(episode.segment_lengths),
            write_count=written,
            skip_count=write_attempt_total - written,
            readout_target_cosine_mean=(pre_write_cosine_sum / written if written else 0.0),
        )
        return TruncatedMetaTTTEpisodeOutput(
            total=detached_total,
            query_loss=detached_query,
            final_fast_states=adapted.fast_states,
            final_runtime=adapted.runtime,
            audit=episode_audit,
        )

    def _reset_trajectory(
        self,
        owner: RuntimeOwner,
        *,
        differentiable: bool,
    ) -> _Trajectory:
        runtime = self.runtime_resetter(owner)
        fast_states = tuple(
            self.fast_controller.reset_fast_state(differentiable=differentiable)
            for _ in owner.video_ids
        )
        return _Trajectory(runtime.with_fast_states(fast_states))

    def _observe(
        self,
        chunk: MetaCausalChunk,
        trajectory: _Trajectory,
        lifecycle: PrefillLifecycle,
        *,
        seed: int,
        with_grad: bool,
        prepared_query: PreparedQueryOutput | None = None,
    ) -> ObservationChunkOutput:
        request = replace(
            chunk.request,
            runtime_state=trajectory.runtime,
            bank_states=trajectory.runtime.bank_states,
            prepared_query=prepared_query,
        )
        with (
            _seeded_rng(seed, trajectory.fast_states),
            torch.set_grad_enabled(with_grad),
            self.fast_controller.use_fast_state(trajectory.fast_states),
        ):
            output = self.model.observe_chunk(request, lifecycle)
        return output

    def _prepare_query(
        self,
        chunk: MetaCausalChunk,
        trajectory: _Trajectory,
        *,
        with_grad: bool,
    ) -> PreparedQueryOutput:
        query_input = chunk.request.query_input
        with (
            _seeded_rng(query_dropout_seed(query_input), trajectory.fast_states),
            torch.set_grad_enabled(with_grad),
        ):
            output = self.model.components.query_encoder(
                query_input,
                inference=chunk.request.inference,
            )
        cache = trajectory.runtime.temporal_cache
        video_input = chunk.request.video_input
        spec = getattr(video_input, "spec", video_input)
        reset_soft_state = bool(getattr(spec, "reset_soft_state", False))
        if isinstance(output, QueryEncoderOutput) and cache is not None and not reset_soft_state:
            output = replace(
                output,
                embeddings=replace(
                    output.embeddings,
                    q_target=_reanchor_query_signature(
                        output.q_target,
                        cache.query_signatures,
                    ),
                ),
            )
        return PreparedQueryOutput.bind(query_input, output)

    def _answer(
        self,
        query: MetaTTTQueryPoint,
        observation: ObservationChunkOutput,
        lifecycle: PrefillLifecycle,
        *,
        fast_states: Sequence[FastMemoryState],
        with_grad: bool,
    ) -> StateTTTModelOutput:
        request = answer_query_request(observation.owner, observation, query.answer)
        with (
            torch.set_grad_enabled(with_grad),
            self.fast_controller.use_fast_state(fast_states),
        ):
            return self.model.prefill_answer(
                self.model.prepare_answer(request, lifecycle),
                lifecycle,
            )

    def _query_objective(
        self,
        query: MetaTTTQueryPoint,
        output: StateTTTModelOutput,
        *,
        dedup: O2DedupContext | None = None,
    ) -> MetaQueryObjective:
        with trace_cuda_phase("outer_loss", stage="a5_query"):
            inputs = self.query_loss_builder(
                output,
                answer=query.answer,
                supervision=query.supervision,
                dedup=dedup,
            )
            answer = compute_answer_loss(inputs.answer)
            state = (
                inputs.state
                if isinstance(inputs.state, OfficialWeakStateLossOutput)
                else compute_state_loss(inputs.state)
            )
            outer = compute_outer_loss(
                OuterLossInput(
                    answer_after=answer,
                    state_after=cast(StateLossOutput, state),
                )
            )
        return MetaQueryObjective(
            answer,
            state,
            outer,
            _query_metrics(answer, state),
            OfficialWeakGradientAnchors(
                q_target=output.query.q_target,
                q_operator=getattr(output.query, "q_operator", output.query.q_target),
                q_time=getattr(output.query, "q_time", output.query.q_target),
            ),
        )

    def _balance_query_objectives(
        self,
        objectives: tuple[MetaQueryObjective, ...],
        *,
        calibration: bool = False,
        statistical_weight: float = 1.0,
    ) -> tuple[MetaQueryObjective, ...]:
        official = tuple(
            isinstance(objective.state, OfficialWeakStateLossOutput) for objective in objectives
        )
        if not all(official):
            self.last_balance_audit = None
            return objectives
        states = tuple(cast(OfficialWeakStateLossOutput, item.state) for item in objectives)
        balanced = (
            self.outer_composer.calibrate(
                tuple(item.answer for item in objectives),
                states,
                statistical_weights=(statistical_weight,) * len(objectives),
            )
            if calibration
            else self.outer_composer.compose(
                tuple(item.answer for item in objectives),
                states,
                gradient_anchors=tuple(item.gradient_anchors for item in objectives),
                statistical_weights=(statistical_weight,) * len(objectives),
            )
        )
        self.last_balance_audit = balanced.audit
        if balanced.audit is None:
            return objectives
        outputs: list[MetaQueryObjective] = []
        for objective, outer in zip(objectives, balanced.objectives, strict=True):
            outputs.append(
                replace(
                    objective,
                    outer=outer,
                )
            )
        return tuple(outputs)


def _query_metrics(
    answer: AnswerLossOutput,
    state: StateLossOutput | OfficialWeakStateLossOutput,
) -> QueryMetricSnapshot:
    return QueryMetricSnapshot(
        metrics=(
            ("loss/answer", _term_float(answer.loss)),
            ("loss/state", float(state.total.detach().item())),
        )
    )


def _term_float(term: object) -> float | None:
    value = getattr(term, "value", None)
    valid = getattr(term, "row_valid_mask", None)
    if not isinstance(value, Tensor) or not isinstance(valid, Tensor):
        return None
    return float(value.detach().item()) if bool(valid.any().item()) else None


def _runtime_from_observation(
    observation: ObservationChunkOutput,
    owner: RuntimeOwner,
) -> BatchRuntimeState:
    runtime = observation.runtime_state
    return runtime


def _plain_backward(loss: Tensor, retain_graph: bool = False) -> None:
    loss.backward(retain_graph=retain_graph)


def _capture_query_proxy_gradients(
    states: Sequence[FastMemoryState],
    *,
    backward_gradient_scale: float,
) -> tuple[Tensor, ...]:
    """Capture unscaled Query cotangents before releasing its isolated proxy state."""

    values = tuple(value for state in states for value in state.fast_parameters)
    gradients: list[Tensor] = []
    for value in values:
        gradient = value.grad
        if gradient is None:
            gradients.append(torch.zeros_like(value, dtype=torch.float32))
            continue
        gradients.append(gradient.detach().float().clone().div_(backward_gradient_scale))
    return tuple(gradients)


def _clip_query_proxy_gradients(
    gradients: Sequence[Tensor],
    *,
    max_norm: float,
    epsilon: float,
) -> tuple[Tensor, ...]:
    """Clip one Query's complete FP32 cotangent before segment-level summation."""

    values = tuple(gradients)
    raw_norm = _cotangent_norm_float(values)
    scale = min(1.0, max_norm / max(raw_norm, epsilon))
    return tuple(gradient * scale for gradient in values)


def _cotangent_norm_float(gradients: Sequence[Tensor]) -> float:
    values = tuple(gradients)
    squared_sums = tuple(
        float(
            gradient.detach()
            .float()
            .square()
            .sum(dtype=torch.float32)
            .to(device="cpu", dtype=torch.float64)
            .item()
        )
        for gradient in values
    )
    return math.sqrt(math.fsum(squared_sums))


class _SeededRNG:
    def __init__(self, seed: int, devices: tuple[int, ...]) -> None:
        self.seed = seed
        self.context = torch.random.fork_rng(devices=list(devices))

    def __enter__(self) -> None:
        self.context.__enter__()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.context.__exit__(exc_type, exc, traceback)


def _seeded_rng(seed: int, states: Sequence[FastMemoryState]) -> _SeededRNG:
    devices = tuple(
        sorted(
            {
                cast(int, state.m.device.index)
                for state in states
                if state.m.device.type == "cuda" and state.m.device.index is not None
            }
        )
    )
    return _SeededRNG(seed, devices)


def _reanchor_query_signature(current: Tensor, reference: Tensor) -> Tensor:
    """Keep one episode's Query signature bitwise stable without cutting its new graph."""

    # Forward value is exactly the authoritative cache signature.  The zero-valued residual
    # preserves an identity gradient to the freshly recomputed Query graph.
    return reference.detach() + (current - current.detach())


def _fork_retrieval_runtime(runtime: BatchRuntimeState) -> BatchRuntimeState:
    """Isolate mutable retrieval rings while retaining functional Bank/FSM state."""

    return BatchRuntimeState(
        tuple(
            replace(row, retrieval_history=history.fork())
            for row, history in zip(runtime.rows, runtime.retrieval_histories, strict=True)
        )
    )
