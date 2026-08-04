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

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import pairwise
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.associative_ttt import (
    ASSOCIATIVE_CONTRACT,
    AssociativeTTTIntermediates,
)
from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    FastMemoryState,
    FastTTTForwardAudit,
    MemoryWriteBatch,
    SlotStateView,
    deferred_fast_vjp_loss,
    make_query_proxy_fast_state,
)
from ttt_svcbench_qwen.input_composer import ComposedInput, map_teacher_forced_targets
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
from ttt_svcbench_qwen.memory_write import (
    MemoryTruncateAudit,
    MemoryWriteResult,
    apply_memory_writes,
    truncate_memory_state,
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
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase, trace_event
from ttt_svcbench_qwen.stage_a_targets import (
    O2DedupContext,
    OfficialWeakStateLossOutput,
    OfficialWeakTargetBuilder,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.tensor_contracts import tensor_storage_key
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageASupervisionBatch,
    answer_query_request,
)
from ttt_svcbench_qwen.training_context import (
    QueryActivationOffloadBudget,
    QueryActivationOffloadScope,
    query_activation_context,
)

_ADAPTATION_MODES = ("meta_ttt", "no_write")
_LEGACY_STATIC_W0 = "static_w0"


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

    def __post_init__(self) -> None:
        if not isinstance(self.request, ObservationChunkRequest):
            raise TypeError("Meta-TTT chunks require ObservationChunkRequest")
        if self.request.inference:
            raise ValueError("Meta-TTT training chunks must set inference=False")
        if (
            not math.isfinite(self.start_time)
            or not math.isfinite(self.end_time)
            or self.start_time < 0.0
            or self.end_time < self.start_time
        ):
            raise ValueError("Meta-TTT chunk times must be finite and ordered")


RawSupportVisualBatcher = Callable[
    [tuple[MetaCausalChunk, ...], int],
    tuple[MetaCausalChunk, ...],
]


@dataclass(frozen=True, slots=True)
class MetaTTTQueryPoint:
    """A later causal observation and labels exposed only after model prefill."""

    chunk: MetaCausalChunk
    query_time: float
    answer: StageAEpisodeAnswerInputs
    supervision: StageASupervisionBatch
    task_name: str
    case_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.query_time) or self.query_time < self.chunk.end_time:
            raise ValueError("query_time must be finite and include no future observation")
        if not self.task_name or not self.case_id:
            raise ValueError("Meta-TTT Query task_name/case_id must be non-empty")
        if self.answer.base_input_ids.shape[0] != self.supervision.answer.batch_size:
            raise ValueError("Meta-TTT Query Answer inputs and labels must share B")


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
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Meta-TTT episode seed must be a non-negative integer")
        support_count = len(self.support_chunks)
        if support_count < 1:
            raise ValueError("Meta-TTT episodes require at least one Support chunk")
        if not self.query_points:
            raise ValueError("Meta-TTT episodes require at least one later Query point")
        if (
            not self.segment_lengths
            or any(
                type(length) is not int or length < 1 or length > 8
                for length in self.segment_lengths
            )
            or sum(self.segment_lengths) != support_count
        ):
            raise ValueError("Meta-TTT requires bounded 1-8 Support segments")
        if not self.segment_query_counts and len(self.segment_lengths) == len(self.query_points):
            object.__setattr__(
                self,
                "segment_query_counts",
                (1,) * len(self.segment_lengths),
            )
        if (
            len(self.segment_query_counts) != len(self.segment_lengths)
            or any(type(count) is not int or count < 1 for count in self.segment_query_counts)
            or sum(self.segment_query_counts) != len(self.query_points)
        ):
            raise ValueError("Meta-TTT requires one non-empty Query bundle per Support segment")
        expected_roles = ("intermediate",) * (len(self.query_points) - 1) + ("final",)
        if self.query_roles != expected_roles:
            raise ValueError("Meta-TTT Query roles must be Final or Intermediate then Final")
        if self.query_weights != (1.0,) * len(self.query_points):
            raise ValueError("Meta-TTT Query weights are frozen to one")
        if type(self.diagnostic_query_count) is not int or self.diagnostic_query_count < 0:
            raise ValueError("Meta-TTT diagnostic Query count must be a non-negative integer")
        if type(self.insufficient_inter_query_gap) is not bool:
            raise TypeError("Meta-TTT insufficient-gap audit must be bool")
        prefix = () if self.prewarm_chunk is None else (self.prewarm_chunk,)
        chunks = (*prefix, *self.support_chunks, *(query.chunk for query in self.query_points))
        if any(chunk.request.owner != self.owner for chunk in chunks):
            raise ValueError("all Meta-TTT requests must share the episode owner")
        batch_size = len(self.owner.video_ids)
        if any(query.answer.base_input_ids.shape[0] != batch_size for query in self.query_points):
            raise ValueError("all Meta-TTT Query rows must align to the owner batch")
        support_ends = tuple(chunk.end_time for chunk in self.support_chunks)
        if any(right <= left for left, right in pairwise(support_ends)):
            raise ValueError("Support chunk end times must advance strictly")
        if self.prewarm_chunk is not None and self.prewarm_chunk.end_time >= support_ends[0]:
            raise ValueError("the no-update prewarm chunk must precede every Support chunk")
        query_ends = tuple(query.chunk.end_time for query in self.query_points)
        query_times = tuple(query.query_time for query in self.query_points)
        if any(right <= left for left, right in pairwise(query_ends)):
            raise ValueError("Query observation end times must advance strictly")
        if any(right <= left for left, right in pairwise(query_times)):
            raise ValueError("Query points must advance strictly in causal time")
        support_offset = 0
        query_offset = 0
        previous_query_time: float | None = None
        for length, query_count in zip(
            self.segment_lengths,
            self.segment_query_counts,
            strict=True,
        ):
            segment = self.support_chunks[support_offset : support_offset + length]
            queries = self.query_points[query_offset : query_offset + query_count]
            if segment[-1].end_time >= queries[0].query_time:
                raise ValueError("each Query bundle must follow every Support in its segment")
            if previous_query_time is not None and any(
                chunk.end_time <= previous_query_time for chunk in segment
            ):
                raise ValueError("later segment Supports must advance beyond the prior Meta Query")
            previous_query_time = queries[-1].query_time
            support_offset += length
            query_offset += query_count


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
        if not isinstance(output.answer_logits, Tensor) or not isinstance(
            output.composed, ComposedInput
        ):
            raise TypeError("Meta-TTT Query requires Tensor logits and ComposedInput")
        if not isinstance(output.observations, ObservationOutputs):
            raise TypeError("Meta-TTT State loss requires ObservationOutputs")
        if not isinstance(output.query, QueryEncoderOutput) or not isinstance(
            output.retrieval, RetrieverOutput
        ):
            raise TypeError("Meta-TTT State loss requires typed Query/Retrieval outputs")
        if supervision.state is None and not supervision.official_weak:
            raise ValueError("Meta-TTT Query points require explicit or official-weak State labels")
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
        if supervision.official_weak:
            state: StateLossInput | OfficialWeakStateLossOutput = self.official_weak_builder(
                output.observations,
                output.query,
                output.retrieval,
                supervision.official_weak,
                dedup=dedup,
            )
        else:
            assert supervision.state is not None
            state = self.target_builder(
                output.observations,
                output.query,
                output.retrieval,
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

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.metrics)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Query metrics must have unique non-empty names")
        if any(value is not None and not math.isfinite(value) for _, value in self.metrics):
            raise ValueError("Query metrics must be finite or N/A")

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
class MemoryWriteAudit:
    """One Support chunk's per-video slot-memory write evidence."""

    support_index: int
    start_time: float
    end_time: float
    write_versions_before: tuple[int, ...]
    write_versions_observed: tuple[int, ...]
    write_versions_after: tuple[int, ...]
    did_write: tuple[bool, ...]
    skip_reasons: tuple[str | None, ...]
    slots_written: tuple[int, ...]
    eta_sums: tuple[float, ...]
    eta_renormalized: tuple[bool, ...]
    pre_write_cosine_means: tuple[float, ...]
    post_write_cosine_means: tuple[float, ...]
    key_pairwise_cosine_means: tuple[float, ...]
    value_pairwise_cosine_means: tuple[float, ...]
    delta_pairwise_cosine_means: tuple[float, ...]
    write_norms: tuple[float, ...]
    memory_norms: tuple[float, ...]
    readout_shares: tuple[float, ...]
    valid_token_counts: tuple[int, ...]
    bank_record_counts: tuple[int, ...]
    runtime_detached: bool
    next_only_verified: bool

    def __post_init__(self) -> None:
        batch_size = len(self.write_versions_before)
        aligned = (
            self.write_versions_observed,
            self.write_versions_after,
            self.did_write,
            self.skip_reasons,
            self.slots_written,
            self.eta_sums,
            self.eta_renormalized,
            self.pre_write_cosine_means,
            self.post_write_cosine_means,
            self.key_pairwise_cosine_means,
            self.value_pairwise_cosine_means,
            self.delta_pairwise_cosine_means,
            self.write_norms,
            self.memory_norms,
            self.readout_shares,
            self.valid_token_counts,
            self.bank_record_counts,
        )
        if batch_size <= 0 or any(len(values) != batch_size for values in aligned):
            raise ValueError("memory write audit fields must align to the owner batch")
        if self.write_versions_observed != self.write_versions_before:
            raise ValueError("current Support was not observed with its before-write memory")
        if not self.next_only_verified:
            raise ValueError("Meta-TTT write failed the next-chunk-only audit")
        cosines = (
            *self.pre_write_cosine_means,
            *self.post_write_cosine_means,
            *self.key_pairwise_cosine_means,
            *self.value_pairwise_cosine_means,
            *self.delta_pairwise_cosine_means,
        )
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in cosines):
            raise ValueError("memory write cosine audits must be finite in [-1, 1]")
        norms = (
            *self.eta_sums,
            *self.write_norms,
            *self.memory_norms,
            *self.readout_shares,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in norms):
            raise ValueError("memory write audit norms must be finite and non-negative")
        if any(type(value) is not int or value < 0 for value in self.slots_written):
            raise ValueError("memory write slots_written must be non-negative integers")


@dataclass(frozen=True, slots=True)
class TruncatedSegmentAudit:
    """One bounded truncated-graph segment and its local backward boundary."""

    training_mode: str
    segment_index: int
    support_start_index: int
    support_end_index: int
    support_count: int
    query_count: int
    pre_write_cosine_mean: float
    eta_sum: float
    renormalized_count: int
    raw_query_cotangent_sum_norm: float
    clipped_query_cotangent_sum_norm: float
    deferred_vjp_norm: float
    query_roles: tuple[str, ...]
    query_weights: tuple[float, ...]
    write_version_before_segment: tuple[int, ...]
    write_version_at_query: tuple[int, ...]
    write_attempt_count: int
    write_count: int
    skip_count: int
    skip_reason_counts: tuple[tuple[str, int], ...]
    backward_applied: bool
    includes_query_backward: bool
    truncated: bool
    truncate_audits: tuple[MemoryTruncateAudit, ...]

    def __post_init__(self) -> None:
        if self.training_mode not in {"meta_ttt", "outer_only", "no_write"}:
            raise ValueError("A5 segment training mode must be meta_ttt, outer_only, or no_write")
        integers = (
            self.segment_index,
            self.support_start_index,
            self.support_end_index,
            self.support_count,
            self.query_count,
            self.write_attempt_count,
            self.write_count,
            self.skip_count,
            self.renormalized_count,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("truncated segment indices/counts must be non-negative integers")
        if self.support_count <= 0:
            raise ValueError("a truncated segment must contain at least one Support")
        if self.query_count <= 0:
            raise ValueError("a truncated segment must contain at least one Meta Query")
        if self.support_end_index - self.support_start_index + 1 != self.support_count:
            raise ValueError("truncated segment Support range does not match its count")
        if not math.isfinite(self.pre_write_cosine_mean) or not (
            -1.0 <= self.pre_write_cosine_mean <= 1.0
        ):
            raise ValueError("truncated segment pre-write cosine must be finite in [-1, 1]")
        if not math.isfinite(self.eta_sum) or self.eta_sum < 0.0:
            raise ValueError("truncated segment eta sum must be finite and non-negative")
        cotangent_norms = (
            self.raw_query_cotangent_sum_norm,
            self.clipped_query_cotangent_sum_norm,
            self.deferred_vjp_norm,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in cotangent_norms):
            raise ValueError("truncated segment cotangent norms must be finite and non-negative")
        if self.clipped_query_cotangent_sum_norm != self.deferred_vjp_norm:
            raise ValueError("deferred VJP norm must equal the clipped Query cotangent sum norm")
        if (
            len(self.query_roles) != self.query_count
            or len(self.query_weights) != self.query_count
            or any(role not in {"intermediate", "final"} for role in self.query_roles)
            or self.query_weights != (1.0,) * self.query_count
        ):
            raise ValueError("truncated segment must carry unit-weight Meta Queries")
        if self.write_attempt_count != self.write_count + self.skip_count:
            raise ValueError("segment write attempts must equal writes plus skips")
        if sum(count for _, count in self.skip_reason_counts) != self.skip_count:
            raise ValueError("segment skip-reason counts must equal skip count")
        if self.training_mode == "no_write":
            if self.write_attempt_count or self.write_count or self.skip_count:
                raise ValueError("no-write segments cannot attempt memory writes")
            if self.pre_write_cosine_mean != 0.0 or self.eta_sum != 0.0:
                raise ValueError("no-write segments cannot report write statistics")
        else:
            expected_mode = "meta_ttt" if self.write_count > 0 else "outer_only"
            if self.training_mode != expected_mode:
                raise ValueError("A5 segment training mode disagrees with its memory writes")
        if not self.write_version_before_segment or len(self.write_version_before_segment) != len(
            self.write_version_at_query
        ):
            raise ValueError("segment write-version boundaries must align")
        if not self.write_version_at_query or any(
            type(value) is not int or value < 0 for value in self.write_version_at_query
        ):
            raise ValueError("truncated segment Query write versions must be non-negative integers")
        flags = (self.backward_applied, self.includes_query_backward, self.truncated)
        if any(type(value) is not bool for value in flags):
            raise TypeError("truncated segment flags must be bool")
        if not self.backward_applied:
            raise ValueError("every truncated segment must contribute one backward collective")
        if not self.includes_query_backward:
            raise ValueError("every A5 segment must close with one Meta Query")
        if self.truncated != bool(self.truncate_audits):
            raise ValueError("segment truncation flag and audits disagree")


@dataclass(frozen=True, slots=True)
class CounterfactualAuditRequest:
    """One deterministic diagnostic Query selection for the current optimizer step."""

    optimizer_step: int
    query_selector: int
    queries_per_rank: int = 1

    def __post_init__(self) -> None:
        if self.optimizer_step <= 0 or self.query_selector < 0:
            raise ValueError("counterfactual audit request indices must be positive/non-negative")
        if type(self.queries_per_rank) is not int or self.queries_per_rank < 1:
            raise ValueError("counterfactual queries_per_rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class CounterfactualReferenceAudit:
    reference: str
    adapted_loss: float
    reference_loss: float
    gain_abs: float
    gain_rel: float
    descent_cosine: float

    def __post_init__(self) -> None:
        if self.reference not in {"episode_zero", "segment_start"}:
            raise ValueError("counterfactual reference is invalid")
        if any(
            not math.isfinite(value)
            for value in (
                self.adapted_loss,
                self.reference_loss,
                self.gain_abs,
                self.gain_rel,
                self.descent_cosine,
            )
        ):
            raise ValueError("counterfactual audit values must be finite")
        if self.adapted_loss < 0.0 or self.reference_loss < 0.0:
            raise ValueError("counterfactual Query losses must be non-negative")
        if not -1.0 <= self.descent_cosine <= 1.0:
            raise ValueError("counterfactual descent cosine must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class QueryCounterfactualAudit:
    optimizer_step: int
    references: tuple[CounterfactualReferenceAudit, ...]

    def __post_init__(self) -> None:
        if self.optimizer_step <= 0:
            raise ValueError("counterfactual optimizer step must be positive")
        if tuple(item.reference for item in self.references) != (
            "episode_zero",
            "segment_start",
        ):
            raise ValueError("counterfactual Query audit must contain both ordered references")


@dataclass(frozen=True, slots=True)
class QueryCotangentClipAudit:
    raw_joint_norm: float
    clipped_joint_norm: float
    clip_scale: float
    clipped: bool
    max_norm: float

    def __post_init__(self) -> None:
        values = (self.raw_joint_norm, self.clipped_joint_norm, self.clip_scale, self.max_norm)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Query cotangent clip audit values must be finite and non-negative")
        if self.max_norm <= 0.0:
            raise ValueError("Query cotangent clip audit requires a positive max_norm")
        if self.clip_scale > 1.0 or self.clipped_joint_norm > self.max_norm * (1.0 + 1.0e-6):
            raise ValueError("Query cotangent clip audit exceeds the configured norm cap")
        if self.clipped != (self.clip_scale < 1.0):
            raise ValueError("Query cotangent clipped flag disagrees with its scale")


@dataclass(frozen=True, slots=True)
class TruncatedQueryPointAudit:
    """Adapted-only Query audit used by the production A5 path."""

    query_index: int
    query_role: str
    query_weight: float
    support_count: int
    weighted_outer_loss: float
    task_name: str
    case_id: str
    query_time: float
    observation_end_time: float
    write_versions: tuple[int, ...]
    metrics: QueryMetricSnapshot
    prefill_count: int
    observation_immutable: bool
    proxy_gradient_norms: tuple[float, ...]
    proxy_gradient_joint_norm_raw: float
    proxy_gradient_joint_norm_clipped: float
    proxy_gradient_clip_scale: float
    proxy_gradient_clipped: bool
    proxy_gradient_max_norm: float
    proxy_gradient_status: str
    proxy_storage_isolated: bool
    proxy_max_abs_value_drift: float
    counterfactual: QueryCounterfactualAudit | None = None

    def __post_init__(self) -> None:
        if type(self.query_index) is not int or self.query_index < 0:
            raise ValueError("truncated Query index must be a non-negative integer")
        if self.query_role not in {"intermediate", "final"} or self.query_weight != 1.0:
            raise ValueError("truncated Query role/weight is invalid")
        if type(self.support_count) is not int or not 1 <= self.support_count <= 8:
            raise ValueError("truncated Query must supervise 1-8 Supports")
        if not math.isfinite(self.weighted_outer_loss) or self.weighted_outer_loss < 0.0:
            raise ValueError("truncated Query weighted Outer loss must be finite and non-negative")
        if not self.task_name or not self.case_id:
            raise ValueError("truncated Query task/case identifiers must be non-empty")
        if self.query_time < self.observation_end_time:
            raise ValueError("truncated Query audit exposes future observation")
        if self.prefill_count != 1:
            raise ValueError("each production Query must execute exactly one prefill")
        if not self.observation_immutable:
            raise ValueError("production Query answer mutated its observation")
        if not self.proxy_gradient_norms or any(
            not math.isfinite(value) or value < 0.0 for value in self.proxy_gradient_norms
        ):
            raise ValueError("Query proxy gradient norms must be finite and non-negative")
        expected_joint = math.sqrt(math.fsum(value * value for value in self.proxy_gradient_norms))
        if not math.isclose(
            self.proxy_gradient_joint_norm_raw,
            expected_joint,
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise ValueError("Query proxy raw joint norm disagrees with per-matrix norms")
        QueryCotangentClipAudit(
            raw_joint_norm=self.proxy_gradient_joint_norm_raw,
            clipped_joint_norm=self.proxy_gradient_joint_norm_clipped,
            clip_scale=self.proxy_gradient_clip_scale,
            clipped=self.proxy_gradient_clipped,
            max_norm=self.proxy_gradient_max_norm,
        )
        gradient_nonzero = any(value > 0.0 for value in self.proxy_gradient_norms)
        if gradient_nonzero != (self.proxy_gradient_status == "nonzero"):
            raise ValueError("Query proxy gradient status disagrees with measured norms")
        if self.proxy_gradient_status not in {
            "nonzero",
            "zero_padding",
            "zero_fast_objective_satisfied",
            "zero_no_valid_retrieval_bag",
            "zero_no_fast_dependent_term",
        }:
            raise ValueError("Query proxy gradient status is invalid")
        if not self.proxy_storage_isolated:
            raise ValueError("Query proxy memory matrices must use isolated storage")
        if (
            not math.isfinite(self.proxy_max_abs_value_drift)
            or self.proxy_max_abs_value_drift != 0.0
        ):
            raise ValueError("Query proxy memory values must exactly match authoritative M_after")


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeAudit:
    """Bounded-memory evidence for an otherwise unbounded numeric memory trajectory."""

    associative_contract: str
    adaptation_mode: str
    ttt_enabled: bool
    associative_valid_count: int
    loss_weight: float
    support_count: int
    query_count: int
    diagnostic_query_count: int
    zero_support_query_count: int
    support_segments_without_query: int
    insufficient_inter_query_gap: bool
    prewarm_count: int
    truncation_horizon: int
    segment_count: int
    backward_count: int
    query_backward_count: int
    deferred_vjp_backward_count: int
    truncation_count: int
    maximum_retained_support_graphs: int
    write_attempt_count: int
    write_count: int
    skip_count: int
    readout_target_cosine_mean: float
    post_write_cosine_mean: float
    readout_share_mean: float
    memory_norm_max: float
    write_norm_mean: float
    eta_sum_mean: float
    renormalized_fraction: float
    slots_written_total: int
    bank_record_count: int
    empty_bank_count: int
    parameter_versions_unchanged_before_outer_step: bool
    bank_context_detached: bool
    support_supervision_reachable: bool
    training_counterfactual_executed: bool
    diagnostic_counterfactual_executed: bool
    counterfactual_audited_query_count: int
    segments: tuple[TruncatedSegmentAudit, ...]
    writes: tuple[MemoryWriteAudit, ...]
    queries: tuple[TruncatedQueryPointAudit, ...]

    def __post_init__(self) -> None:
        if self.adaptation_mode not in _ADAPTATION_MODES:
            raise ValueError("A5 adaptation mode is invalid")
        if type(self.ttt_enabled) is not bool:
            raise TypeError("A5 TTT enabled audit must be bool")
        if type(self.associative_valid_count) is not int or self.associative_valid_count < 0:
            raise ValueError("A5 associative valid count must be a non-negative integer")
        if self.ttt_enabled != (self.adaptation_mode == "meta_ttt"):
            raise ValueError("A5 TTT enabled audit disagrees with adaptation mode")
        if type(self.diagnostic_counterfactual_executed) is not bool:
            raise TypeError("diagnostic counterfactual flag must be bool")
        if (
            type(self.counterfactual_audited_query_count) is not int
            or not 0 <= self.counterfactual_audited_query_count <= self.query_count
            or self.diagnostic_counterfactual_executed
            != (self.counterfactual_audited_query_count > 0)
        ):
            raise ValueError("diagnostic counterfactual Query count is invalid")
        if self.loss_weight not in (0.0, 1.0):
            raise ValueError("A5 episode audit loss weight must be deterministic zero or one")
        if self.truncation_horizon <= 0:
            raise ValueError("truncation horizon must be positive")
        if type(self.insufficient_inter_query_gap) is not bool:
            raise TypeError("A5 insufficient-gap audit must be bool")
        if self.maximum_retained_support_graphs > self.truncation_horizon:
            raise ValueError("A5 retained more than K Support graphs")
        cosines = (self.readout_target_cosine_mean, self.post_write_cosine_mean)
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in cosines):
            raise ValueError("A5 memory cosine audits must be finite in [-1, 1]")
        nonnegative_finite = (
            self.readout_share_mean,
            self.memory_norm_max,
            self.write_norm_mean,
            self.eta_sum_mean,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative_finite):
            raise ValueError("A5 memory scale audits must be finite and non-negative")
        if (
            not math.isfinite(self.renormalized_fraction)
            or not 0.0 <= self.renormalized_fraction <= 1.0
        ):
            raise ValueError("A5 eta renormalized fraction must be in [0, 1]")
        if (
            type(self.slots_written_total) is not int
            or type(self.bank_record_count) is not int
            or type(self.empty_bank_count) is not int
            or min(self.slots_written_total, self.bank_record_count, self.empty_bank_count) < 0
        ):
            raise ValueError("A5 memory audit counts must be non-negative integers")
        expected_write_audits = self.support_count if self.ttt_enabled else 0
        if len(self.writes) != expected_write_audits or len(self.queries) != self.query_count:
            raise ValueError("A5 detailed audit counts drifted")
        if not self.ttt_enabled and (
            self.write_attempt_count
            or self.write_count
            or self.skip_count
            or self.associative_valid_count
            or self.maximum_retained_support_graphs
            or self.slots_written_total
            or any(segment.training_mode != "no_write" for segment in self.segments)
        ):
            raise ValueError("no-write audit contains Meta-TTT activity")
        query_offset = 0
        for segment in self.segments:
            queries = self.queries[query_offset : query_offset + segment.query_count]
            if len(queries) != segment.query_count or any(
                segment.support_count != query.support_count for query in queries
            ):
                raise ValueError("A5 segment and Query bundle Support counts drifted")
            query_offset += segment.query_count
        if query_offset != self.query_count:
            raise ValueError("A5 segment Query bundle counts drifted")
        if not self.parameter_versions_unchanged_before_outer_step:
            raise ValueError("outer parameters changed before the episode-level optimizer step")
        if self.training_counterfactual_executed:
            raise ValueError("the production A5 path must not execute no-write counterfactuals")

    @property
    def meta_ttt_segment_count(self) -> int:
        return sum(segment.training_mode == "meta_ttt" for segment in self.segments)

    @property
    def outer_only_segment_count(self) -> int:
        return sum(segment.training_mode == "outer_only" for segment in self.segments)

    @property
    def no_write_segment_count(self) -> int:
        return sum(segment.training_mode == "no_write" for segment in self.segments)


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeOutput:
    """Detached logging values returned after all segment backward calls have completed."""

    total: Tensor
    query_loss: Tensor
    final_fast_states: tuple[FastMemoryState, ...]
    final_runtime: BatchRuntimeState
    audit: TruncatedMetaTTTEpisodeAudit

    def __post_init__(self) -> None:
        values = (self.total, self.query_loss)
        if any(value.ndim != 0 or value.dtype != torch.float32 for value in values):
            raise ValueError("truncated A5 logging losses must be detached FP32 scalars")
        if any(value.requires_grad or value.grad_fn is not None for value in values):
            raise ValueError("truncated A5 output must not retain completed autograd graphs")
        if any(not bool(torch.isfinite(value).item()) for value in values):
            raise ValueError("truncated A5 logging losses must be finite")
        if not torch.equal(self.total, self.query_loss):
            raise ValueError("truncated A5 total must equal Query outer loss exactly")


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
        raw_support_visual_batcher: RawSupportVisualBatcher | None = None,
        support_visual_batch_size: int = 1,
        query_activation_offload: bool = False,
        outer_composer: OfficialWeakOuterLossComposer | None = None,
        adaptation_mode: str = "meta_ttt",
    ) -> None:
        if not isinstance(config, ProjectConfig):
            raise TypeError("Meta-TTT runner requires validated ProjectConfig")
        if not isinstance(model, StateTTTModel):
            raise TypeError("Meta-TTT runner requires StateTTTModel")
        self.config = config
        self.model = model
        self.fast_controller = fast_controller
        self.runtime_resetter = runtime_resetter
        self.query_loss_builder = query_loss_builder or StageAQueryLossBuilder()
        if type(query_encoder_reuse) is not bool:
            raise TypeError("query_encoder_reuse must be bool")
        self.query_encoder_reuse = query_encoder_reuse
        if raw_support_visual_batcher is not None and not callable(raw_support_visual_batcher):
            raise TypeError("raw_support_visual_batcher must be callable")
        if type(support_visual_batch_size) is not int or support_visual_batch_size <= 0:
            raise ValueError("support_visual_batch_size must be a positive integer")
        self.raw_support_visual_batcher = raw_support_visual_batcher
        self.support_visual_batch_size = support_visual_batch_size
        if type(query_activation_offload) is not bool:
            raise TypeError("query_activation_offload must be bool")
        self.query_activation_offload = query_activation_offload
        self.outer_composer = outer_composer or OfficialWeakOuterLossComposer(
            config.loss.official_weak_balance
        )
        if adaptation_mode == _LEGACY_STATIC_W0:
            raise ValueError(
                "A5 adaptation mode 'static_w0' was renamed; use 'no_write' "
                "(the memory stays zero and the memory-interface parameters freeze)"
            )
        if adaptation_mode not in _ADAPTATION_MODES:
            raise ValueError("A5 adaptation mode must be meta_ttt or no_write")
        self.adaptation_mode = adaptation_mode
        self.ttt_enabled = adaptation_mode == "meta_ttt"
        if not self.ttt_enabled:
            for parameter in self.fast_controller.collect_associative_parameters():
                parameter.requires_grad_(False)
        self.last_balance_audit: OfficialWeakBalanceAudit | None = None
        audit_config = config.a5.counterfactual_audit
        if audit_config.enabled and (
            audit_config.interval_steps <= 0 or audit_config.queries_per_rank <= 0
        ):
            raise ValueError("A5 counterfactual audit requires positive interval and query count")

    def run_truncated(
        self,
        episode: MetaTTTEpisode,
        *,
        backward: Callable[[Tensor, bool], None] | None = None,
        backward_gradient_scale: float = 1.0,
        episode_loss_weight: float = 1.0,
        counterfactual_audit: CounterfactualAuditRequest | None = None,
    ) -> TruncatedMetaTTTEpisodeOutput:
        """Run one Query-aligned deferred-VJP closure per bounded Support segment."""

        self._validate_truncated_episode(episode)
        if (
            not isinstance(backward_gradient_scale, int | float)
            or not math.isfinite(float(backward_gradient_scale))
            or float(backward_gradient_scale) <= 0.0
        ):
            raise ValueError("backward_gradient_scale must be finite and positive")
        if (
            not isinstance(episode_loss_weight, int | float)
            or not math.isfinite(float(episode_loss_weight))
            or float(episode_loss_weight) not in (0.0, 1.0)
        ):
            raise ValueError("A5 episode loss weight must be deterministic zero or one")
        episode_weight = float(episode_loss_weight)
        audit_config = self.config.a5.counterfactual_audit
        if counterfactual_audit is not None:
            if not audit_config.enabled:
                raise ValueError("counterfactual audit request requires enabled diagnostic config")
            if not self.ttt_enabled:
                raise ValueError("counterfactual Query audit requires Meta-TTT")
            if not isinstance(counterfactual_audit, CounterfactualAuditRequest):
                raise TypeError("counterfactual audit request is invalid")
        if counterfactual_audit is None:
            selected_query_indices: frozenset[int] = frozenset()
        else:
            total_queries = len(episode.query_points)
            selected_query_indices = frozenset(
                (counterfactual_audit.query_selector + offset) % total_queries
                for offset in range(min(counterfactual_audit.queries_per_rank, total_queries))
            )
        backward_fn = backward or _plain_backward
        self.model.train()
        adapted = self._reset_trajectory(episode.owner, differentiable=True)
        tracked_parameters = _unique_parameters(tuple(self.model.parameters()))
        versions_before = tuple(parameter._version for parameter in tracked_parameters)
        horizon = self.config.a5.truncation_horizon
        support_count = len(episode.support_chunks)
        write_audits: list[MemoryWriteAudit] = []
        segment_audits: list[TruncatedSegmentAudit] = []
        query_audits: list[TruncatedQueryPointAudit] = []
        maximum_retained = 0
        support_offset = 0
        device = adapted.fast_states[0].m.device
        query_loss_detached = torch.zeros((), dtype=torch.float32, device=device)
        pre_write_cosine_sum = 0.0
        post_write_cosine_sum = 0.0
        write_norm_sum = 0.0
        eta_sum_total = 0.0
        readout_share_sum = 0.0
        readout_share_count = 0
        memory_norm_max = 0.0
        renormalized_write_count = 0
        slots_written_total = 0
        bank_record_count = 0
        empty_bank_count = 0
        support_lifecycle = PrefillLifecycle(episode.owner)
        query_offload_budget = (
            QueryActivationOffloadBudget.from_environment()
            if self.query_activation_offload
            else None
        )

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
                    with_grad=self.ttt_enabled,
                )
                prewarm_query = (
                    first_segment_query.detached()
                    if isinstance(first_segment_query.value, QueryEncoderOutput)
                    else first_segment_query
                )
                prewarm_query.validate_for(prewarm.request.query_input)
        prewarm_observation, prewarm_fast_audit = self._observe(
            prewarm,
            adapted,
            support_lifecycle,
            seed=episode.seed,
            with_grad=False,
            prepared_query=prewarm_query,
        )
        if any(prewarm_fast_audit.write_versions):
            raise ValueError("the no-write prewarm must observe the zero-memory generation")
        adapted.runtime = _runtime_from_observation(prewarm_observation, episode.owner)
        episode_zero_reference = (
            _zero_reference_fast_states(adapted.fast_states)
            if counterfactual_audit is not None and episode_weight == 1.0
            else None
        )
        del prewarm_observation

        query_offset = 0
        for segment_index, (segment_length, segment_query_count) in enumerate(
            zip(
                episode.segment_lengths,
                episode.segment_query_counts,
                strict=True,
            )
        ):
            segment_start = support_offset
            segment_write_start = len(write_audits)
            write_version_before_segment = tuple(
                state.write_version for state in adapted.fast_states
            )
            segment_start_reference = (
                _snapshot_reference_fast_states(adapted.fast_states)
                if counterfactual_audit is not None and episode_weight == 1.0
                else None
            )
            queries = episode.query_points[query_offset : query_offset + segment_query_count]
            query_roles = episode.query_roles[query_offset : query_offset + segment_query_count]
            query_weights = episode.query_weights[query_offset : query_offset + segment_query_count]
            raw_segment = episode.support_chunks[support_offset : support_offset + segment_length]
            active_segment = raw_segment
            if self.raw_support_visual_batcher is not None and self.support_visual_batch_size > 1:
                with _seeded_rng(
                    episode.seed + support_offset + 1,
                    adapted.fast_states,
                ):
                    prepared_segment = self.raw_support_visual_batcher(
                        raw_segment,
                        self.support_visual_batch_size,
                    )
                self._validate_prepared_segment(raw_segment, prepared_segment)
                active_segment = prepared_segment
            segment_retained_writes = 0
            segment_pre_write_cosine_sum = 0.0
            segment_pre_write_cosine_count = 0
            segment_eta_sum = 0.0
            segment_renormalized_count = 0
            segment_query = first_segment_query if segment_index == 0 else None
            for segment_offset, chunk in enumerate(active_segment):
                support_index = support_offset + segment_offset
                if self.query_encoder_reuse and segment_query is None:
                    segment_query = self._prepare_query(
                        chunk,
                        adapted,
                        with_grad=self.ttt_enabled,
                    )
                elif segment_query is not None:
                    segment_query.validate_for(chunk.request.query_input)
                observation, fast_audit = self._observe(
                    chunk,
                    adapted,
                    support_lifecycle,
                    seed=episode.seed + support_index + 1,
                    with_grad=self.ttt_enabled,
                    prepared_query=segment_query,
                )
                adapted.runtime = _runtime_from_observation(observation, episode.owner)
                if not self.ttt_enabled:
                    if tuple(state.write_version for state in adapted.fast_states) != (
                        write_version_before_segment
                    ):
                        raise RuntimeError("no-write Support changed memory write versions")
                    del observation
                    continue
                intermediates = observation.soft_intermediates.fast_associative
                if intermediates is None:
                    raise RuntimeError("Meta-TTT Support did not capture associative tensors")
                spatial = observation.soft_intermediates.spatial
                if spatial is None:
                    raise RuntimeError("Meta-TTT Support did not produce spatial slot state")
                write_batch = self.fast_controller.prepare_write(intermediates, spatial)
                before_versions = tuple(state.write_version for state in adapted.fast_states)
                results = apply_memory_writes(
                    fast_states=adapted.fast_states,
                    batch=write_batch,
                )
                adapted.fast_states = tuple(result.fast_state for result in results)
                after_versions = tuple(state.write_version for state in adapted.fast_states)
                write_audits.append(
                    _make_memory_write_audit(
                        support_index=support_index,
                        chunk=chunk,
                        before_versions=before_versions,
                        observed_fast_audit=fast_audit,
                        results=results,
                        runtime=adapted.runtime,
                        after_versions=after_versions,
                    )
                )
                segment_retained_writes += 1
                maximum_retained = max(maximum_retained, segment_retained_writes)
                for result in results:
                    if result.did_write:
                        pre_write_cosine_sum += result.pre_write_cosine_mean
                        post_write_cosine_sum += result.post_write_cosine_mean
                        write_norm_sum += result.write_norm
                        eta_sum_total += result.eta_sum
                        segment_pre_write_cosine_sum += result.pre_write_cosine_mean
                        segment_pre_write_cosine_count += 1
                        segment_eta_sum += result.eta_sum
                        slots_written_total += result.slots_written
                        renormalized_write_count += int(result.eta_renormalized)
                        segment_renormalized_count += int(result.eta_renormalized)
                    memory_norm_max = max(memory_norm_max, result.memory_norm)
                for share in fast_audit.readout_share_norms:
                    readout_share_sum += share
                    readout_share_count += 1
                counts = tuple(int(value.item()) for value in intermediates.bank_record_counts)
                bank_record_count += sum(counts)
                empty_bank_count += sum(count == 0 for count in counts)
                del results, write_batch, intermediates, spatial, observation

            query_runtime_snapshot = adapted.runtime
            # Pre-query snapshot: the dedup base must not see the query chunk's own
            # commit, or the current slots would match themselves and zero the novelty.
            query_dedup = O2DedupContext.from_identity_states(
                query_runtime_snapshot.identity_bank_states
            )
            authoritative_fast_states = query_runtime_snapshot.fast_states
            accumulated_gradients: tuple[Tensor, ...] | None = None
            raw_accumulated_gradients: tuple[Tensor, ...] | None = None
            write_versions = tuple(state.write_version for state in authoritative_fast_states)
            for bundle_offset, (query, query_role, query_weight) in enumerate(
                zip(queries, query_roles, query_weights, strict=True)
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
                    calibration_observation, _ = self._observe(
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

                proxy_pairs = tuple(
                    make_query_proxy_fast_state(state) for state in authoritative_fast_states
                )
                proxy_states = tuple(state for state, _ in proxy_pairs)
                proxy_audits = tuple(audit for _, audit in proxy_pairs)
                query_trajectory = _Trajectory(
                    _fork_retrieval_runtime(query_runtime_snapshot).with_fast_states(proxy_states)
                )
                query_lifecycle = PrefillLifecycle(episode.owner)
                prepared_query = (
                    self._prepare_query(query.chunk, query_trajectory, with_grad=True)
                    if self.query_encoder_reuse
                    else None
                )
                activation_scope = query_activation_context(
                    self.query_activation_offload,
                    shared_budget=query_offload_budget,
                )
                with activation_scope:
                    observation, _ = self._observe(
                        query.chunk,
                        query_trajectory,
                        query_lifecycle,
                        seed=episode.seed + 10_000 + global_query_index,
                        with_grad=True,
                        prepared_query=prepared_query,
                    )
                    observation_versions = _tensor_version_signature(observation)
                    output = self._answer(
                        query,
                        observation,
                        query_lifecycle,
                        fast_states=proxy_states,
                        with_grad=True,
                    )
                immutable = observation_versions == _tensor_version_signature(observation)
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
                captured_gradients, proxy_gradient_norms = _capture_query_proxy_gradients(
                    proxy_states,
                    backward_gradient_scale=float(backward_gradient_scale),
                )
                clipped_gradients, cotangent_clip_audit = _clip_query_proxy_gradients(
                    captured_gradients,
                    max_norm=self.config.a5.query_meta_gradient.max_norm,
                    epsilon=self.config.a5.query_meta_gradient.epsilon,
                )
                query_counterfactual: QueryCounterfactualAudit | None = None
                if (
                    global_query_index in selected_query_indices
                    and episode_weight == 1.0
                    and counterfactual_audit is not None
                ):
                    if episode_zero_reference is None or segment_start_reference is None:
                        raise RuntimeError(
                            "counterfactual reference memory states were not captured"
                        )
                    adapted_audit_loss = float(
                        (query_weight * objective.outer.outer).detach().float().item()
                    )
                    reference_items: list[CounterfactualReferenceAudit] = []
                    for reference_name, reference_states in (
                        ("episode_zero", episode_zero_reference),
                        ("segment_start", segment_start_reference),
                    ):
                        reference_loss = self._counterfactual_query_loss(
                            query=query,
                            query_weight=query_weight,
                            query_runtime_snapshot=query_runtime_snapshot,
                            reference_states=reference_states,
                            balance_audit=balance_audit,
                            seed=episode.seed + 10_000 + global_query_index,
                            dedup=query_dedup,
                        )
                        gain_abs = reference_loss - adapted_audit_loss
                        reference_items.append(
                            CounterfactualReferenceAudit(
                                reference=reference_name,
                                adapted_loss=adapted_audit_loss,
                                reference_loss=reference_loss,
                                gain_abs=gain_abs,
                                gain_rel=gain_abs / max(abs(reference_loss), 1.0e-6),
                                descent_cosine=_query_descent_cosine(
                                    authoritative_fast_states,
                                    reference_states,
                                    captured_gradients,
                                ),
                            )
                        )
                    query_counterfactual = QueryCounterfactualAudit(
                        optimizer_step=counterfactual_audit.optimizer_step,
                        references=tuple(reference_items),
                    )
                    trace_event(
                        "a5_diagnostic_counterfactual",
                        optimizer_step=counterfactual_audit.optimizer_step,
                        query_index=global_query_index,
                        query_role=query_role,
                        support_count=segment_length,
                        segment_index=segment_index,
                        references=tuple(
                            {
                                "reference": item.reference,
                                "adapted_loss": item.adapted_loss,
                                "reference_loss": item.reference_loss,
                                "gain_abs": item.gain_abs,
                                "gain_rel": item.gain_rel,
                                "descent_cosine": item.descent_cosine,
                            }
                            for item in reference_items
                        ),
                    )
                if accumulated_gradients is None:
                    accumulated_gradients = clipped_gradients
                    raw_accumulated_gradients = captured_gradients
                else:
                    accumulated_gradients = tuple(
                        total + current
                        for total, current in zip(
                            accumulated_gradients,
                            clipped_gradients,
                            strict=True,
                        )
                    )
                    if raw_accumulated_gradients is None:
                        raise RuntimeError("raw Query cotangent accumulator disappeared")
                    raw_accumulated_gradients = tuple(
                        total + current
                        for total, current in zip(
                            raw_accumulated_gradients,
                            captured_gradients,
                            strict=True,
                        )
                    )
                if isinstance(activation_scope, QueryActivationOffloadScope):
                    activation_scope.release()
                if balance_audit is not None and gradient_statistics is not None:
                    balance_audit = self.outer_composer.commit_streamed_gradients(
                        (gradient_statistics,),
                        balance_audit,
                    )
                    self.last_balance_audit = balance_audit
                maximum_proxy_drift = max(
                    audit.max_abs_value_drift for audit in proxy_audits
                )
                query_audits.append(
                    TruncatedQueryPointAudit(
                        query_index=global_query_index,
                        query_role=query_role,
                        query_weight=query_weight,
                        support_count=segment_length,
                        weighted_outer_loss=float(query_loss.detach().item()),
                        task_name=query.task_name,
                        case_id=query.case_id,
                        query_time=query.query_time,
                        observation_end_time=query.chunk.end_time,
                        write_versions=write_versions,
                        metrics=objective.metrics,
                        prefill_count=query_lifecycle.audit().prefill_count,
                        observation_immutable=immutable,
                        proxy_gradient_norms=proxy_gradient_norms,
                        proxy_gradient_joint_norm_raw=cotangent_clip_audit.raw_joint_norm,
                        proxy_gradient_joint_norm_clipped=(
                            cotangent_clip_audit.clipped_joint_norm
                        ),
                        proxy_gradient_clip_scale=cotangent_clip_audit.clip_scale,
                        proxy_gradient_clipped=cotangent_clip_audit.clipped,
                        proxy_gradient_max_norm=cotangent_clip_audit.max_norm,
                        proxy_gradient_status=_proxy_gradient_status(
                            proxy_gradient_norms,
                            objective.metrics,
                            episode_weight=episode_weight,
                        ),
                        proxy_storage_isolated=all(
                            audit.storage_isolated for audit in proxy_audits
                        ),
                        proxy_max_abs_value_drift=maximum_proxy_drift,
                        counterfactual=query_counterfactual,
                    )
                )
                for state in proxy_states:
                    state.m.grad = None
                adapted.runtime = query_runtime_snapshot
                del (
                    captured_gradients,
                    clipped_gradients,
                    cotangent_clip_audit,
                    query_counterfactual,
                    gradient_statistics,
                    query_loss,
                    objective,
                    output,
                    observation,
                    prepared_query,
                    query_lifecycle,
                    query_trajectory,
                    activation_scope,
                    proxy_states,
                    proxy_audits,
                    proxy_pairs,
                    balanced_outer,
                )

            if accumulated_gradients is None or raw_accumulated_gradients is None:
                raise RuntimeError("A5 Query bundle produced no proxy cotangent")
            raw_query_cotangent_sum_norm = _cotangent_norm_float(
                raw_accumulated_gradients
            )
            clipped_query_cotangent_sum_norm = _cotangent_norm_float(
                accumulated_gradients
            )
            deferred_vjp_norm = clipped_query_cotangent_sum_norm
            deferred_vjp = deferred_fast_vjp_loss(
                authoritative_fast_states,
                accumulated_gradients,
            )
            backward_fn(deferred_vjp, False)
            truncate_audits = self._truncate_trajectory(adapted)
            segment_writes = write_audits[segment_write_start:]
            segment_write_attempts = sum(len(audit.did_write) for audit in segment_writes)
            segment_write_count = sum(sum(audit.did_write) for audit in segment_writes)
            segment_skip_reasons: dict[str, int] = {}
            for audit in segment_writes:
                for did_write, reason in zip(
                    audit.did_write,
                    audit.skip_reasons,
                    strict=True,
                ):
                    if not did_write:
                        key = reason or "unspecified"
                        segment_skip_reasons[key] = segment_skip_reasons.get(key, 0) + 1
            segment_training_mode = (
                ("meta_ttt" if segment_write_count > 0 else "outer_only")
                if self.ttt_enabled
                else "no_write"
            )
            segment_pre_write_cosine = (
                segment_pre_write_cosine_sum / segment_pre_write_cosine_count
                if segment_pre_write_cosine_count
                else 0.0
            )
            segment_audits.append(
                TruncatedSegmentAudit(
                    training_mode=segment_training_mode,
                    segment_index=segment_index,
                    support_start_index=segment_start,
                    support_end_index=segment_start + segment_length - 1,
                    support_count=segment_length,
                    query_count=segment_query_count,
                    pre_write_cosine_mean=segment_pre_write_cosine,
                    eta_sum=segment_eta_sum,
                    renormalized_count=segment_renormalized_count,
                    raw_query_cotangent_sum_norm=raw_query_cotangent_sum_norm,
                    clipped_query_cotangent_sum_norm=clipped_query_cotangent_sum_norm,
                    deferred_vjp_norm=deferred_vjp_norm,
                    query_roles=query_roles,
                    query_weights=query_weights,
                    write_version_before_segment=write_version_before_segment,
                    write_version_at_query=write_versions,
                    write_attempt_count=segment_write_attempts,
                    write_count=segment_write_count,
                    skip_count=segment_write_attempts - segment_write_count,
                    skip_reason_counts=tuple(sorted(segment_skip_reasons.items())),
                    backward_applied=True,
                    includes_query_backward=True,
                    truncated=True,
                    truncate_audits=truncate_audits,
                )
            )
            trace_event(
                "a5_query_aligned_segment_released",
                video_ids=episode.owner.video_ids,
                trajectory_ids=episode.owner.trajectory_ids,
                segment_index=segment_index,
                segment_count=len(episode.segment_lengths),
                query_roles=query_roles,
                query_count=segment_query_count,
                support_count=segment_length,
                query_weights=query_weights,
                diagnostic_query_count=episode.diagnostic_query_count,
                training_mode=segment_training_mode,
                ttt_enabled=self.ttt_enabled,
                write_version_delta=max(write_versions) - max(write_version_before_segment),
                write_attempt_count=segment_write_attempts,
                write_count=segment_write_count,
                skip_count=segment_write_attempts - segment_write_count,
                skip_reason_counts=tuple(sorted(segment_skip_reasons.items())),
                raw_query_cotangent_sum_norm=raw_query_cotangent_sum_norm,
                clipped_query_cotangent_sum_norm=clipped_query_cotangent_sum_norm,
                deferred_vjp_norm=deferred_vjp_norm,
                cuda_allocated_bytes=(
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
                cuda_reserved_bytes=(
                    torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                ),
                offload_live_bytes=(
                    query_offload_budget.claimed_bytes if query_offload_budget is not None else 0
                ),
            )
            support_offset += segment_length
            query_offset += segment_query_count
            del (
                accumulated_gradients,
                raw_accumulated_gradients,
                deferred_vjp,
                segment_query,
                segment_writes,
            )

        if support_offset != support_count:
            raise RuntimeError("A5 supervised segments did not consume every Support")
        if query_offset != len(episode.query_points):
            raise RuntimeError("A5 supervised Query bundles did not consume every Query")
        versions_after = tuple(parameter._version for parameter in tracked_parameters)
        attempted = sum(len(audit.did_write) for audit in write_audits)
        written = sum(sum(audit.did_write) for audit in write_audits)
        detached_query = query_loss_detached.detach().clone()
        detached_total = detached_query.detach().clone()
        readout_target_cosine_mean = pre_write_cosine_sum / written if written else 0.0
        post_write_cosine_mean = post_write_cosine_sum / written if written else 0.0
        write_norm_mean = write_norm_sum / written if written else 0.0
        eta_sum_mean = eta_sum_total / written if written else 0.0
        renormalized_fraction = renormalized_write_count / written if written else 0.0
        readout_share_mean = (
            readout_share_sum / readout_share_count if readout_share_count else 0.0
        )

        query_count = len(episode.query_points)
        trace_event(
            "a5_memory_numerical_audit",
            video_ids=episode.owner.video_ids,
            trajectory_ids=episode.owner.trajectory_ids,
            support_count=support_count,
            query_count=query_count,
            adaptation_mode=self.adaptation_mode,
            ttt_enabled=self.ttt_enabled,
            associative_contract=ASSOCIATIVE_CONTRACT,
            associative_valid_count=written,
            readout_target_cosine_mean=readout_target_cosine_mean,
            post_write_cosine_mean=post_write_cosine_mean,
            readout_share_mean=readout_share_mean,
            memory_norm_max=memory_norm_max,
            write_norm_mean=write_norm_mean,
            eta_sum_mean=eta_sum_mean,
            renormalized_fraction=renormalized_fraction,
            slots_written_total=slots_written_total,
            bank_record_count=bank_record_count,
            empty_bank_count=empty_bank_count,
        )
        episode_audit = TruncatedMetaTTTEpisodeAudit(
            associative_contract=ASSOCIATIVE_CONTRACT,
            adaptation_mode=self.adaptation_mode,
            ttt_enabled=self.ttt_enabled,
            associative_valid_count=written,
            loss_weight=episode_weight,
            support_count=support_count,
            query_count=query_count,
            diagnostic_query_count=episode.diagnostic_query_count,
            zero_support_query_count=0,
            support_segments_without_query=0,
            insufficient_inter_query_gap=episode.insufficient_inter_query_gap,
            prewarm_count=1,
            truncation_horizon=horizon,
            segment_count=len(segment_audits),
            backward_count=query_count + len(segment_audits),
            query_backward_count=query_count,
            deferred_vjp_backward_count=len(segment_audits),
            truncation_count=len(segment_audits),
            maximum_retained_support_graphs=maximum_retained,
            write_attempt_count=attempted,
            write_count=written,
            skip_count=attempted - written,
            readout_target_cosine_mean=readout_target_cosine_mean,
            post_write_cosine_mean=post_write_cosine_mean,
            readout_share_mean=readout_share_mean,
            memory_norm_max=memory_norm_max,
            write_norm_mean=write_norm_mean,
            eta_sum_mean=eta_sum_mean,
            renormalized_fraction=renormalized_fraction,
            slots_written_total=slots_written_total,
            bank_record_count=bank_record_count,
            empty_bank_count=empty_bank_count,
            parameter_versions_unchanged_before_outer_step=versions_before == versions_after,
            bank_context_detached=True,
            support_supervision_reachable=False,
            training_counterfactual_executed=False,
            diagnostic_counterfactual_executed=any(
                query.counterfactual is not None for query in query_audits
            ),
            counterfactual_audited_query_count=sum(
                query.counterfactual is not None for query in query_audits
            ),
            segments=tuple(segment_audits),
            writes=tuple(write_audits),
            queries=tuple(query_audits),
        )
        return TruncatedMetaTTTEpisodeOutput(
            total=detached_total,
            query_loss=detached_query,
            final_fast_states=adapted.fast_states,
            final_runtime=adapted.runtime,
            audit=episode_audit,
        )

    def _validate_truncated_episode(self, episode: MetaTTTEpisode) -> None:
        stage = self.config.a5
        if episode.prewarm_chunk is None:
            raise ValueError("A5 production requires an explicit S0 no-update prewarm chunk")
        if not episode.query_points:
            raise ValueError("A5 production requires at least one Meta Query")
        if len(episode.segment_query_counts) != len(episode.segment_lengths):
            raise ValueError("A5 production requires one Query bundle per Support segment")
        if any(length > stage.truncation_horizon for length in episode.segment_lengths):
            raise ValueError("A5 production segment exceeds the configured truncation horizon")
        if episode.seed != stage.seed:
            raise ValueError("A5 episode seed must equal the fixed Stage C seed")

    @staticmethod
    def _truncate_trajectory(
        trajectory: _Trajectory,
    ) -> tuple[MemoryTruncateAudit, ...]:
        pairs = tuple(truncate_memory_state(state) for state in trajectory.fast_states)
        trajectory.fast_states = tuple(state for state, _ in pairs)
        values = tuple(state.m for state in trajectory.fast_states)
        if len({tensor_storage_key(value) for value in values}) != len(values):
            raise ValueError("truncated batched memory states must remain storage-isolated")
        return tuple(audit for _, audit in pairs)

    def _reset_trajectory(
        self,
        owner: RuntimeOwner,
        *,
        differentiable: bool,
    ) -> _Trajectory:
        runtime = self.runtime_resetter(owner)
        if not isinstance(runtime, BatchRuntimeState):
            raise TypeError("Meta-TTT runtime resetter must return BatchRuntimeState")
        runtime.validate_for(owner)
        fast_states = tuple(
            self.fast_controller.reset_fast_state(differentiable=differentiable)
            for _ in owner.video_ids
        )
        if any(
            state.write_version or state.write_count or state.skip_count for state in fast_states
        ):
            raise ValueError("fresh Meta-TTT memory states must reset all counters")
        if any(
            state.m.device.type != "meta" and bool(torch.count_nonzero(state.m.detach()))
            for state in fast_states
        ):
            raise ValueError("fresh Meta-TTT memory states must start at exact zero")
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
    ) -> tuple[ObservationChunkOutput, FastTTTForwardAudit]:
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
        audit = self.fast_controller.last_audit
        if not isinstance(audit, FastTTTForwardAudit) or not audit.used_runtime_state:
            raise ValueError("Meta-TTT observe did not consume the managed FastMemoryState")
        expected = tuple(state.write_version for state in trajectory.fast_states)
        if audit.write_versions != expected:
            raise ValueError("Fast Adapter audit version disagrees with the bound trajectory")
        return output, audit

    def _prepare_query(
        self,
        chunk: MetaCausalChunk,
        trajectory: _Trajectory,
        *,
        with_grad: bool,
    ) -> PreparedQueryOutput:
        query_input = chunk.request.query_input
        if query_reuse_key(query_input) != query_reuse_key(chunk.query_input):
            raise ValueError("Meta-TTT chunk Query metadata drifted before reuse")
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

    @staticmethod
    def _validate_prepared_segment(
        source: tuple[MetaCausalChunk, ...],
        prepared: tuple[MetaCausalChunk, ...],
    ) -> None:
        if len(prepared) != len(source):
            raise ValueError("raw visual batcher changed the K-segment length")
        for before, after in zip(source, prepared, strict=True):
            if (
                before.start_time != after.start_time
                or before.end_time != after.end_time
                or before.query_input != after.query_input
                or before.request.owner != after.request.owner
                or before.request.query_input != after.request.query_input
                or before.request.inference != after.request.inference
                or before.request.runtime_state is not after.request.runtime_state
                or before.request.bank_states is not after.request.bank_states
            ):
                raise ValueError("raw visual batcher may replace only request.video_input")

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

    def _counterfactual_query_loss(
        self,
        *,
        query: MetaTTTQueryPoint,
        query_weight: float,
        query_runtime_snapshot: BatchRuntimeState,
        reference_states: tuple[FastMemoryState, ...],
        balance_audit: OfficialWeakBalanceAudit | None,
        seed: int,
        dedup: O2DedupContext | None = None,
    ) -> float:
        """Evaluate one no-grad memory reference without committing any runtime state."""

        reference = _Trajectory(
            _fork_retrieval_runtime(query_runtime_snapshot).with_fast_states(reference_states)
        )
        lifecycle = PrefillLifecycle(query.chunk.request.owner)
        if not isinstance(self.fast_controller, nn.Module):
            raise TypeError("counterfactual audit requires an nn.Module fast controller")
        with (
            _temporarily_clear_module_parameter_gradients(self.fast_controller),
            _seeded_rng(seed, reference_states),
            torch.no_grad(),
        ):
            prepared_query = (
                self._prepare_query(query.chunk, reference, with_grad=False)
                if self.query_encoder_reuse
                else None
            )
            observation, _ = self._observe(
                query.chunk,
                reference,
                lifecycle,
                seed=seed,
                with_grad=False,
                prepared_query=prepared_query,
            )
            output = self._answer(
                query,
                observation,
                lifecycle,
                fast_states=reference.fast_states,
                with_grad=False,
            )
            objective = self._query_objective(query, output, dedup=dedup)
            if balance_audit is not None:
                outer = self.outer_composer.compose_one_from_audit(
                    objective.answer,
                    cast(OfficialWeakStateLossOutput, objective.state),
                    query_count=1,
                    audit=balance_audit,
                )
                objective = replace(objective, outer=outer)
            value = query_weight * objective.outer.outer
        scalar = float(value.detach().float().item())
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError("counterfactual Query loss must be finite and non-negative")
        if lifecycle.audit().prefill_count != 1:
            raise ValueError("counterfactual reference must execute exactly one prefill")
        return scalar

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
        if not objectives:
            raise ValueError("Meta-TTT requires at least one Query objective")
        official = tuple(
            isinstance(objective.state, OfficialWeakStateLossOutput) for objective in objectives
        )
        if not any(official):
            self.last_balance_audit = None
            return objectives
        if not all(official):
            raise ValueError("official-weak balancing cannot mix dense Query losses")
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


def _make_memory_write_audit(
    *,
    support_index: int,
    chunk: MetaCausalChunk,
    before_versions: tuple[int, ...],
    observed_fast_audit: FastTTTForwardAudit,
    results: tuple[MemoryWriteResult, ...],
    runtime: object,
    after_versions: tuple[int, ...],
) -> MemoryWriteAudit:
    expected_after = tuple(
        before + int(result.did_write)
        for before, result in zip(before_versions, results, strict=True)
    )
    next_only = (
        observed_fast_audit.write_versions == before_versions and after_versions == expected_after
    )
    return MemoryWriteAudit(
        support_index=support_index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        write_versions_before=before_versions,
        write_versions_observed=observed_fast_audit.write_versions,
        write_versions_after=after_versions,
        did_write=tuple(result.did_write for result in results),
        skip_reasons=tuple(
            None if result.skip_reason is None else result.skip_reason.value for result in results
        ),
        slots_written=tuple(result.slots_written for result in results),
        eta_sums=tuple(result.eta_sum for result in results),
        eta_renormalized=tuple(result.eta_renormalized for result in results),
        pre_write_cosine_means=tuple(result.pre_write_cosine_mean for result in results),
        post_write_cosine_means=tuple(result.post_write_cosine_mean for result in results),
        key_pairwise_cosine_means=tuple(result.key_pairwise_cosine_mean for result in results),
        value_pairwise_cosine_means=tuple(result.value_pairwise_cosine_mean for result in results),
        delta_pairwise_cosine_means=tuple(result.delta_pairwise_cosine_mean for result in results),
        write_norms=tuple(result.write_norm for result in results),
        memory_norms=tuple(result.memory_norm for result in results),
        readout_shares=observed_fast_audit.readout_share_norms,
        valid_token_counts=observed_fast_audit.valid_token_counts,
        bank_record_counts=observed_fast_audit.bank_record_counts,
        runtime_detached=not _contains_grad_tensor(runtime),
        next_only_verified=next_only,
    )


def _query_metrics(
    answer: AnswerLossOutput,
    state: StateLossOutput | OfficialWeakStateLossOutput,
) -> QueryMetricSnapshot:
    common = (
        ("loss/answer", _term_float(answer.loss)),
        ("loss/state", float(state.total.detach().item())),
        ("answer/token_accuracy", _term_float(answer.teacher_forced_token_accuracy)),
        ("answer/number_token_accuracy", _term_float(answer.number_token_accuracy)),
        ("answer/exact_match", _term_float(answer.answer_exact_match)),
        ("reader/exact_count_accuracy", _term_float(answer.reader_exact_count_accuracy)),
    )
    if isinstance(state, OfficialWeakStateLossOutput):
        state_metrics = (
            ("state/task", _weak_term_float(state.task)),
            ("state/operator", _weak_term_float(state.operator)),
            ("state/retrieval", _weak_term_float(state.retrieval)),
            ("state/time", _weak_term_float(state.time)),
            *state.audit.metrics(),
        )
    else:
        state_metrics = (
            ("state/o1", _term_float(state.o1)),
            ("state/o2", _term_float(state.o2)),
            ("state/e1", _term_float(state.e1)),
            ("state/e2", _term_float(state.e2)),
        )
    return QueryMetricSnapshot(metrics=(*common, *state_metrics))


def _proxy_gradient_status(
    norms: tuple[float, ...],
    metrics: QueryMetricSnapshot,
    *,
    episode_weight: float,
) -> str:
    """Classify a zero Query-to-memory cotangent without changing the loss path."""

    if any(value > 0.0 for value in norms):
        return "nonzero"
    if episode_weight == 0.0:
        return "zero_padding"
    values = dict(metrics.metrics)
    task = values.get("state/task")
    valid_bags = values.get("retrieval/valid_bag_rows")
    if task is not None and abs(task) <= 1.0e-12:
        return "zero_fast_objective_satisfied"
    if valid_bags is not None and valid_bags <= 0.0:
        return "zero_no_valid_retrieval_bag"
    return "zero_no_fast_dependent_term"


def _weak_term_float(term: object) -> float | None:
    value = getattr(term, "value", None)
    valid_rows = getattr(term, "valid_rows", None)
    if not isinstance(value, Tensor) or type(valid_rows) is not int:
        raise TypeError("official-weak metric source must expose value/valid_rows")
    return float(value.detach().item()) if valid_rows > 0 else None


def _term_float(term: object) -> float | None:
    value = getattr(term, "value", None)
    valid = getattr(term, "row_valid_mask", None)
    if not isinstance(value, Tensor) or not isinstance(valid, Tensor):
        raise TypeError("metric source must expose a typed LossTerm value/mask")
    return float(value.detach().item()) if bool(valid.any().item()) else None


def _runtime_from_observation(
    observation: ObservationChunkOutput,
    owner: RuntimeOwner,
) -> BatchRuntimeState:
    runtime = observation.runtime_state
    if not isinstance(runtime, BatchRuntimeState):
        raise TypeError("Meta-TTT observation must return BatchRuntimeState")
    runtime.validate_for(owner)
    if tuple(observation.bank_states) != runtime.bank_states:
        raise ValueError("Meta-TTT observation Bank states disagree with runtime rows")
    return runtime


def _snapshot_reference_fast_states(
    states: Sequence[FastMemoryState],
) -> tuple[FastMemoryState, ...]:
    """Copy numeric memory state into isolated leaves with no edge to the Support graph."""

    snapshots: list[FastMemoryState] = []
    for state in states:
        snapshots.append(
            FastMemoryState(
                m=state.m.detach().clone().requires_grad_(True),
                write_version=state.write_version,
                write_count=state.write_count,
                skip_count=state.skip_count,
                differentiable=False,
            )
        )
    return tuple(snapshots)


def _zero_reference_fast_states(
    states: Sequence[FastMemoryState],
) -> tuple[FastMemoryState, ...]:
    """Build the exact M=0 counterfactual reference for each batch row."""

    return tuple(
        FastMemoryState(
            m=torch.zeros_like(state.m.detach()).requires_grad_(True),
            write_version=0,
            write_count=0,
            skip_count=0,
            differentiable=False,
        )
        for state in states
    )


@contextmanager
def _temporarily_clear_module_parameter_gradients(
    module: nn.Module,
) -> Iterator[None]:
    """Permit a read-only online-state binding without losing accumulated outer gradients."""

    parameters = tuple(module.parameters())
    saved_gradients = tuple(parameter.grad for parameter in parameters)
    for parameter in parameters:
        parameter.grad = None
    try:
        yield
    finally:
        for parameter, gradient in zip(parameters, saved_gradients, strict=True):
            if parameter.grad is not None:
                raise RuntimeError(
                    "read-only counterfactual forward unexpectedly produced module gradients"
                )
            parameter.grad = gradient


def _query_descent_cosine(
    adapted_states: Sequence[FastMemoryState],
    reference_states: Sequence[FastMemoryState],
    gradients: Sequence[Tensor],
) -> float:
    adapted = tuple(value for state in adapted_states for value in state.fast_parameters)
    references = tuple(value for state in reference_states for value in state.fast_parameters)
    captured = tuple(gradients)
    if not adapted or len(adapted) != len(references) or len(adapted) != len(captured):
        raise ValueError("counterfactual cosine tensors must be non-empty and aligned")
    dot = 0.0
    gradient_squared = 0.0
    delta_squared = 0.0
    for current, reference, gradient in zip(adapted, references, captured, strict=True):
        if current.shape != reference.shape or current.shape != gradient.shape:
            raise ValueError("counterfactual cosine tensor shapes drifted")
        negative_gradient = -gradient.detach().float()
        delta = current.detach().float() - reference.detach().float()
        dot += float((negative_gradient * delta).sum(dtype=torch.float64).item())
        gradient_squared += float(negative_gradient.square().sum(dtype=torch.float64).item())
        delta_squared += float(delta.square().sum(dtype=torch.float64).item())
    denominator = math.sqrt(gradient_squared * delta_squared)
    cosine = 0.0 if denominator == 0.0 else dot / denominator
    if not math.isfinite(cosine):
        raise ValueError("counterfactual descent cosine is non-finite")
    return max(-1.0, min(1.0, cosine))


def _unique_parameters(parameters: Sequence[nn.Parameter]) -> tuple[nn.Parameter, ...]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return tuple(result)


def _plain_backward(loss: Tensor, retain_graph: bool = False) -> None:
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise TypeError("segment backward requires one scalar Tensor")
    if not loss.requires_grad:
        raise ValueError("segment loss must remain connected to the Outer graph")
    loss.backward(retain_graph=retain_graph)


def _capture_query_proxy_gradients(
    states: Sequence[FastMemoryState],
    *,
    backward_gradient_scale: float,
) -> tuple[tuple[Tensor, ...], tuple[float, ...]]:
    """Capture unscaled Query cotangents before releasing its isolated proxy state."""

    values = tuple(value for state in states for value in state.fast_parameters)
    if not values:
        raise ValueError("Query proxy gradient capture requires memory matrices")
    gradients: list[Tensor] = []
    norms: list[float] = []
    for value in values:
        gradient = value.grad
        if gradient is None:
            raise ValueError("Query loss did not produce a gradient for every proxy memory matrix")
        unscaled = gradient.detach().float().clone().div_(backward_gradient_scale)
        if not bool(torch.isfinite(unscaled).all()):
            raise ValueError("Query proxy gradient must be finite after backward unscale")
        gradients.append(unscaled)
        norms.append(float(torch.linalg.vector_norm(unscaled).cpu().item()))
    return tuple(gradients), tuple(norms)


def _clip_query_proxy_gradients(
    gradients: Sequence[Tensor],
    *,
    max_norm: float,
    epsilon: float,
) -> tuple[tuple[Tensor, ...], QueryCotangentClipAudit]:
    """Clip one Query's complete FP32 cotangent before segment-level summation."""

    values = tuple(gradients)
    if not values or any(
        gradient.dtype != torch.float32
        or gradient.requires_grad
        or gradient.grad_fn is not None
        for gradient in values
    ):
        raise ValueError("Query cotangent clipping requires detached FP32 gradients")
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("Query cotangent max_norm must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Query cotangent epsilon must be finite and positive")
    raw_norm = _cotangent_norm_float(values)
    scale = min(1.0, max_norm / max(raw_norm, epsilon))
    clipped = tuple(gradient * scale for gradient in values)
    clipped_norm = _cotangent_norm_float(clipped)
    return clipped, QueryCotangentClipAudit(
        raw_joint_norm=raw_norm,
        clipped_joint_norm=clipped_norm,
        clip_scale=scale,
        clipped=scale < 1.0,
        max_norm=max_norm,
    )


def _cotangent_norm_float(gradients: Sequence[Tensor]) -> float:
    values = tuple(gradients)
    if not values:
        raise ValueError("Query cotangent norm requires at least one tensor")
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
    result = math.sqrt(math.fsum(squared_sums))
    if not math.isfinite(result):
        raise ValueError("Query cotangent norm must be finite")
    return result


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

    if (
        current.shape != reference.shape
        or current.dtype != reference.dtype
        or current.device != reference.device
        or not torch.is_floating_point(current)
    ):
        raise ValueError("Query signature reanchor requires aligned floating tensors")
    current_fp32 = current.detach().float()
    reference_fp32 = reference.detach().float()
    tolerance = max(5.0e-4, 2.0 * float(torch.finfo(current.dtype).eps))
    cosine = torch.nn.functional.cosine_similarity(current_fp32, reference_fp32, dim=-1)
    if not torch.allclose(current_fp32, reference_fp32, atol=tolerance, rtol=0.0) or bool(
        torch.any(cosine < 0.999)
    ):
        maximum_delta = float((current_fp32 - reference_fp32).abs().max().item())
        minimum_cosine = float(cosine.min().item())
        raise ValueError(
            "Query signature changed semantically within one episode "
            f"(max_abs_delta={maximum_delta:.6g}, min_cosine={minimum_cosine:.6g})"
        )
    # Forward value is exactly the authoritative cache signature.  The zero-valued residual
    # preserves an identity gradient to the freshly recomputed Query graph.
    return reference.detach() + (current - current.detach())


def _contains_grad_tensor(value: object, seen: set[int] | None = None) -> bool:
    active = set() if seen is None else seen
    if id(value) in active:
        return False
    active.add(id(value))
    if isinstance(value, Tensor):
        return value.requires_grad or value.grad_fn is not None
    if isinstance(value, Mapping):
        return any(_contains_grad_tensor(item, active) for item in value.values())
    if isinstance(value, tuple | list):
        return any(_contains_grad_tensor(item, active) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_grad_tensor(getattr(value, field.name), active) for field in fields(value)
        )
    return False


def _fork_retrieval_runtime(runtime: BatchRuntimeState) -> BatchRuntimeState:
    """Isolate mutable retrieval rings while retaining functional Bank/FSM state."""

    return BatchRuntimeState(
        tuple(
            replace(row, retrieval_history=history.fork())
            for row, history in zip(runtime.rows, runtime.retrieval_histories, strict=True)
        )
    )


def _contains_tensor(value: object, seen: set[int] | None = None) -> bool:
    active = set() if seen is None else seen
    if id(value) in active:
        return False
    active.add(id(value))
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item, active) for item in value.values())
    if isinstance(value, tuple | list):
        return any(_contains_tensor(item, active) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, field.name), active) for field in fields(value))
    return False


def _tensor_version_signature(value: object) -> tuple[tuple[int, int], ...]:
    found: list[tuple[int, int]] = []
    _collect_tensor_versions(value, found, set())
    return tuple(sorted(found))


def _collect_tensor_versions(
    value: object,
    found: list[tuple[int, int]],
    seen: set[int],
) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Tensor):
        found.append((id(value), value._version))
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_tensor_versions(item, found, seen)
    elif isinstance(value, tuple | list):
        for item in value:
            _collect_tensor_versions(item, found, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_tensor_versions(getattr(value, field.name), found, seen)
