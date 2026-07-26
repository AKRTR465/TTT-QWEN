"""P16/P17 causal Meta-TTT episode orchestration and engineering audits.

Inputs: resettable model/runtime factories, causal Support/Query chunks, typed query labels,
and the frozen P14 loss/functional-SGD contracts.
Outputs: an after-update outer objective, per-video next-only fast generations, detached overlap
snapshots, before/after query metrics, and bounded graph/lifecycle audits.
Forbidden: Support labels, batch-scalar inner updates, in-place fast mutation, first-order training,
cross-video runtime reuse, observe-after-prefill, or carrying differentiable runtime snapshots.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import pairwise
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    FastReanchorAudit,
    FastTTTForwardAudit,
    FastWeightsState,
    OptimizerRuntimeState,
    deferred_fast_vjp_loss,
    make_query_proxy_fast_state,
    reanchor_fast_state,
)
from ttt_svcbench_qwen.functional_sgd import (
    FunctionalSGDResult,
    functional_sgd_steps_from_ttt,
    reset_optimizer_state,
)
from ttt_svcbench_qwen.input_composer import ComposedInput, map_teacher_forced_targets
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    AnswerLossOutput,
    E1ConsistencyInput,
    E2ConsistencyInput,
    EventConsistencyInput,
    IdentityConsistencyInput,
    IdentityPairStatus,
    OuterLossInput,
    OuterLossOutput,
    ReaderCountMetricInput,
    StateLossInput,
    StateLossOutput,
    TemporalPredictionInput,
    TemporalPredictor,
    TTTLossInput,
    TTTLossOutput,
    compute_answer_loss,
    compute_outer_loss,
    compute_state_loss,
    compute_ttt_loss,
)
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BatchRuntimeState,
    ObservationChunkOutput,
    ObservationChunkRequest,
    OnlineOverlapSnapshot,
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
from ttt_svcbench_qwen.stage_a_runtime import StageAWriteAudit
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakStateLossOutput,
    OfficialWeakTargetBuilder,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_encoder import TemporalEncoderOutput
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.tensor_contracts import tensor_storage_key
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageASupervisionBatch,
)
from ttt_svcbench_qwen.training_context import (
    QueryActivationOffloadBudget,
    QueryActivationOffloadScope,
    query_activation_context,
)

_SUPPORTED_TERMS = ("pred", "identity", "event")
_CI_Z_95 = 1.959963984540054


class FastStateController(Protocol):
    """Subset of :class:`FastTTTAdapter` needed by a managed meta episode."""

    last_audit: FastTTTForwardAudit | None

    def reset_fast_state(
        self,
        state: FastWeightsState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastWeightsState: ...

    def use_fast_state(
        self,
        state: FastWeightsState | Sequence[FastWeightsState],
    ) -> AbstractContextManager[object]: ...

    def collect_meta_fast_parameters(self) -> tuple[nn.Parameter, nn.Parameter]: ...


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
            or len(self.segment_lengths) != len(self.query_points)
            or any(
                type(length) is not int or length < 1 or length > 8
                for length in self.segment_lengths
            )
            or sum(self.segment_lengths) != support_count
        ):
            raise ValueError("Meta-TTT requires one 1-8 Support segment per Meta Query")
        expected_roles = (
            ("final",)
            if len(self.query_points) == 1
            else ("intermediate", "final")
        )
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
        offset = 0
        previous_query_time: float | None = None
        for length, query in zip(self.segment_lengths, self.query_points, strict=True):
            segment = self.support_chunks[offset : offset + length]
            if segment[-1].end_time >= query.query_time:
                raise ValueError("each Meta Query must follow every Support in its segment")
            if previous_query_time is not None and any(
                chunk.end_time <= previous_query_time for chunk in segment
            ):
                raise ValueError("later segment Supports must advance beyond the prior Meta Query")
            previous_query_time = query.query_time
            offset += length


@dataclass(frozen=True, slots=True)
class CrossChunkMatchAudit:
    previous_available: bool
    snapshot_detached: bool
    snapshot_storage_isolated: bool
    position_causal: bool
    authoritative_identity_update_evidence: bool
    identity_decision_storage_free: bool
    authoritative_identity_decision_counts: tuple[int, ...]
    identity_matched_counts: tuple[int, ...]
    identity_duplicate_counts: tuple[int, ...]
    identity_low_confidence_counts: tuple[int, ...]
    e1_overlap_counts: tuple[int, ...]
    e2_overlap_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        flags = (
            self.previous_available,
            self.snapshot_detached,
            self.snapshot_storage_isolated,
            self.position_causal,
            self.authoritative_identity_update_evidence,
            self.identity_decision_storage_free,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("cross-chunk match flags must be bool")
        lengths = {
            len(self.identity_matched_counts),
            len(self.authoritative_identity_decision_counts),
            len(self.identity_duplicate_counts),
            len(self.identity_low_confidence_counts),
            len(self.e1_overlap_counts),
            len(self.e2_overlap_counts),
        }
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("cross-chunk match counts must align to one non-empty batch")
        counts = (
            *self.identity_matched_counts,
            *self.authoritative_identity_decision_counts,
            *self.identity_duplicate_counts,
            *self.identity_low_confidence_counts,
            *self.e1_overlap_counts,
            *self.e2_overlap_counts,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("cross-chunk match counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class TTTInputBuildResult:
    inputs: TTTLossInput
    snapshot: OnlineOverlapSnapshot
    audit: CrossChunkMatchAudit


class CausalOverlapTTTInputBuilder:
    """Build P14 inputs from real adjacent outputs and detached exact-position snapshots."""

    def __init__(self, config: ProjectConfig) -> None:
        self.match_threshold = float(config.observation_heads.o2.match_threshold)
        self.ambiguity_margin = float(config.observation_heads.o2.match_ambiguity_margin)

    def __call__(
        self,
        output: ObservationChunkOutput,
        *,
        previous: OnlineOverlapSnapshot | None,
        current_end_time: float,
        enabled_terms: tuple[str, ...],
    ) -> TTTInputBuildResult:
        observations = _typed_observations(output)
        temporal = output.temporal
        if not isinstance(temporal, TemporalEncoderOutput):
            raise TypeError("Meta-TTT requires typed TemporalEncoderOutput")
        if tuple(dict.fromkeys(enabled_terms)) != enabled_terms or any(
            term not in _SUPPORTED_TERMS for term in enabled_terms
        ):
            raise ValueError("Meta-TTT enabled terms must be unique pred/identity/event names")
        snapshot = OnlineOverlapSnapshot.capture(output, end_time=current_end_time)
        if previous is not None:
            if previous.owner != output.owner:
                raise ValueError("overlap snapshot owner changed across chunks")
            if previous.end_time >= current_end_time:
                raise ValueError("overlap snapshot must precede the current chunk end")

        identity, identity_counts = self._identity_input(
            observations,
            previous if "identity" in enabled_terms else None,
        )
        event, event_counts = self._event_input(
            observations,
            previous if "event" in enabled_terms else None,
        )
        previous_available = previous is not None
        snapshot_tensors = () if previous is None else previous.tensors
        snapshot_detached = all(
            not value.requires_grad and value.grad_fn is None for value in snapshot_tensors
        )
        snapshot_isolated = len({tensor_storage_key(value) for value in snapshot_tensors}) == len(
            snapshot_tensors
        )
        hard_audit = output.state_audit
        if isinstance(hard_audit, StageAWriteAudit):
            authoritative_identity = True
            identity_decisions = hard_audit.identity_decisions
        else:
            authoritative_identity = False
            identity_decisions = ()
        decision_counts = (
            tuple(len(values) for values in identity_decisions)
            if authoritative_identity
            else (0,) * observations.o2.identity.shape[0]
        )
        decision_storage_free = not _contains_tensor(identity_decisions)
        if "identity" in enabled_terms and not authoritative_identity:
            raise ValueError(
                "identity consistency requires authoritative IdentityUpdateResult decision audit"
            )
        return TTTInputBuildResult(
            inputs=TTTLossInput(
                temporal=TemporalPredictionInput(
                    hidden=temporal.hidden,
                    valid_mask=temporal.valid_mask,
                    position_ids=temporal.position_ids,
                ),
                identity=identity,
                event=event,
            ),
            snapshot=snapshot,
            audit=CrossChunkMatchAudit(
                previous_available=previous_available,
                snapshot_detached=snapshot_detached,
                snapshot_storage_isolated=snapshot_isolated,
                position_causal=previous is None or previous.end_time < current_end_time,
                authoritative_identity_update_evidence=authoritative_identity,
                identity_decision_storage_free=decision_storage_free,
                authoritative_identity_decision_counts=decision_counts,
                identity_matched_counts=identity_counts[0],
                identity_duplicate_counts=identity_counts[1],
                identity_low_confidence_counts=identity_counts[2],
                e1_overlap_counts=event_counts,
                e2_overlap_counts=event_counts,
            ),
        )

    def _identity_input(
        self,
        current: ObservationOutputs,
        previous: OnlineOverlapSnapshot | None,
    ) -> tuple[IdentityConsistencyInput, tuple[tuple[int, ...], ...]]:
        batch_size, current_width = current.o2.valid_mask.shape
        if previous is None:
            statuses = torch.full(
                (batch_size, max(current_width, 1)),
                int(IdentityPairStatus.PADDING),
                dtype=torch.int64,
                device=current.o2.identity.device,
            )
            indices = torch.full_like(statuses, -1)
            positions = torch.full_like(statuses, -1)
            timestamps = torch.full(
                statuses.shape,
                -1.0,
                dtype=current.o2.timestamps.dtype,
                device=current.o2.identity.device,
            )
            result = IdentityConsistencyInput(
                current_predictions=current.o2.identity,
                previous_targets=current.o2.identity.detach().clone(),
                current_valid_mask=current.o2.valid_mask,
                previous_valid_mask=current.o2.valid_mask.detach().clone(),
                current_indices=indices,
                previous_indices=indices.clone(),
                statuses=statuses,
                current_position_ids=positions,
                previous_position_ids=positions.clone(),
                current_timestamps=timestamps,
                previous_timestamps=timestamps.clone(),
            )
            zeros = (0,) * batch_size
            return result, (zeros, zeros, zeros)

        if previous.identity.device != current.o2.identity.device:
            raise ValueError("identity overlap snapshots must share the current device")
        pair_width = max(current_width, 1)
        current_indices = torch.full(
            (batch_size, pair_width),
            -1,
            dtype=torch.int64,
            device=current.o2.identity.device,
        )
        previous_indices = current_indices.clone()
        statuses = torch.full_like(current_indices, int(IdentityPairStatus.PADDING))
        current_positions = current_indices.clone()
        previous_positions = current_indices.clone()
        current_times = torch.full(
            current_indices.shape,
            -1.0,
            dtype=current.o2.timestamps.dtype,
            device=current.o2.identity.device,
        )
        previous_times = current_times.clone()
        matched_counts: list[int] = []
        duplicate_counts: list[int] = []
        low_counts: list[int] = []
        for row in range(batch_size):
            decisions = self._match_identity_row(current, previous, row)
            matched = duplicates = low = 0
            for pair, (current_index, previous_index, status) in enumerate(decisions):
                current_indices[row, pair] = current_index
                previous_indices[row, pair] = previous_index
                statuses[row, pair] = int(status)
                if status not in (IdentityPairStatus.PADDING, IdentityPairStatus.INVALID_SOURCE):
                    current_positions[row, pair] = current.o2.position_ids[row, current_index]
                    previous_positions[row, pair] = previous.identity_position_ids[
                        row, previous_index
                    ]
                    current_times[row, pair] = current.o2.timestamps[row, current_index]
                    previous_times[row, pair] = previous.identity_timestamps[row, previous_index]
                matched += status is IdentityPairStatus.MATCHED
                duplicates += status is IdentityPairStatus.DUPLICATE
                low += status is IdentityPairStatus.LOW_CONFIDENCE
            matched_counts.append(matched)
            duplicate_counts.append(duplicates)
            low_counts.append(low)
        result = IdentityConsistencyInput(
            current_predictions=current.o2.identity,
            previous_targets=previous.identity,
            current_valid_mask=current.o2.valid_mask,
            previous_valid_mask=previous.identity_valid_mask,
            current_indices=current_indices,
            previous_indices=previous_indices,
            statuses=statuses,
            current_position_ids=current_positions,
            previous_position_ids=previous_positions,
            current_timestamps=current_times,
            previous_timestamps=previous_times,
        )
        return result, (
            tuple(matched_counts),
            tuple(duplicate_counts),
            tuple(low_counts),
        )

    def _match_identity_row(
        self,
        current: ObservationOutputs,
        previous: OnlineOverlapSnapshot,
        row: int,
    ) -> tuple[tuple[int, int, IdentityPairStatus], ...]:
        current_valid = torch.nonzero(current.o2.valid_mask[row], as_tuple=False).flatten().tolist()
        previous_valid = (
            torch.nonzero(previous.identity_valid_mask[row], as_tuple=False).flatten().tolist()
        )
        decisions: list[tuple[int, int, IdentityPairStatus]] = []
        claims: dict[int, list[tuple[int, float, float]]] = {}
        for current_index in current_valid:
            temporal_candidates = [
                previous_index
                for previous_index in previous_valid
                if int(current.o2.position_ids[row, current_index].item())
                == int(previous.identity_position_ids[row, previous_index].item())
                and abs(
                    float(current.o2.timestamps[row, current_index].item())
                    - float(previous.identity_timestamps[row, previous_index].item())
                )
                <= 1.0e-6
            ]
            if not temporal_candidates:
                decisions.append((-1, -1, IdentityPairStatus.INVALID_SOURCE))
                continue
            current_value = current.o2.identity[row, current_index].detach().float()
            scores = [
                float(
                    torch.dot(current_value, previous.identity[row, previous_index].float()).item()
                )
                for previous_index in temporal_candidates
            ]
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            best_offset = order[0]
            best_index = temporal_candidates[best_offset]
            best_score = scores[best_offset]
            second_score = scores[order[1]] if len(order) > 1 else -1.0
            if best_score < self.match_threshold or (
                len(order) > 1 and best_score - second_score <= self.ambiguity_margin
            ):
                decisions.append((current_index, best_index, IdentityPairStatus.LOW_CONFIDENCE))
                continue
            claims.setdefault(best_index, []).append((current_index, best_score, second_score))

        claimed_current = {value[0] for values in claims.values() for value in values}
        for previous_index, values in claims.items():
            ordered = sorted(values, key=lambda item: (-item[1], item[0]))
            winner = ordered[0]
            decisions.append((winner[0], previous_index, IdentityPairStatus.MATCHED))
            decisions.extend(
                (loser[0], previous_index, IdentityPairStatus.DUPLICATE) for loser in ordered[1:]
            )
        decisions.sort(key=lambda item: (item[0] < 0, item[0], item[1]))
        if len(decisions) > max(len(current_valid), 1):  # pragma: no cover - defensive
            raise RuntimeError("identity matcher emitted more pairs than current slots")
        if any(
            current_index >= 0
            and current_index not in claimed_current
            and status is IdentityPairStatus.MATCHED
            for current_index, _, status in decisions
        ):
            raise RuntimeError("identity matcher lost its one-to-one claim bookkeeping")
        return tuple(decisions)

    def _event_input(
        self,
        current: ObservationOutputs,
        previous: OnlineOverlapSnapshot | None,
    ) -> tuple[EventConsistencyInput, tuple[int, ...]]:
        if previous is None:
            batch_size = current.e1.valid_mask.shape[0]
            width = max(int(current.e1.valid_mask.shape[1]), 1)
            pair_mask = torch.zeros(
                (batch_size, width), dtype=torch.bool, device=current.e1.valid_mask.device
            )
            positions = torch.full(pair_mask.shape, -1, dtype=torch.int64, device=pair_mask.device)
            timestamps = torch.full(
                pair_mask.shape,
                -1.0,
                dtype=current.e1.timestamps.dtype,
                device=pair_mask.device,
            )
            e1_current = _pad_probability_width(current.e1.probabilities, width)
            e2_event_current = _pad_probability_width(current.e2.event_probabilities, width)
            e2_phase_current = _pad_probability_width(current.e2.phase_probabilities, width)
            event = EventConsistencyInput(
                e1=E1ConsistencyInput(
                    current_probabilities=e1_current,
                    previous_target_probabilities=e1_current.detach().clone(),
                    pair_mask=pair_mask,
                    alignment_mask=pair_mask.clone(),
                    current_position_ids=positions,
                    previous_position_ids=positions.clone(),
                    current_timestamps=timestamps,
                    previous_timestamps=timestamps.clone(),
                ),
                e2=E2ConsistencyInput(
                    current_event_probabilities=e2_event_current,
                    previous_event_target_probabilities=e2_event_current.detach().clone(),
                    current_phase_probabilities=e2_phase_current,
                    previous_phase_target_probabilities=e2_phase_current.detach().clone(),
                    pair_mask=pair_mask.clone(),
                    alignment_mask=pair_mask.clone(),
                    current_position_ids=positions.clone(),
                    previous_position_ids=positions.clone(),
                    current_timestamps=timestamps.clone(),
                    previous_timestamps=timestamps.clone(),
                ),
            )
            return event, (0,) * batch_size

        if previous.e1_probabilities.device != current.e1.probabilities.device:
            raise ValueError("event overlap snapshots must share the current device")
        pairs = _match_event_positions(current, previous)
        pair_width = max(max((len(row) for row in pairs), default=0), 1)
        batch_size = current.e1.valid_mask.shape[0]
        pair_mask = torch.zeros(
            (batch_size, pair_width), dtype=torch.bool, device=current.e1.valid_mask.device
        )
        alignment = pair_mask.clone()
        current_positions = torch.full(
            pair_mask.shape, -1, dtype=torch.int64, device=pair_mask.device
        )
        previous_positions = current_positions.clone()
        current_times = torch.full(
            pair_mask.shape,
            -1.0,
            dtype=current.e1.timestamps.dtype,
            device=pair_mask.device,
        )
        previous_times = current_times.clone()
        e1_current = torch.zeros(
            (batch_size, pair_width, 3),
            dtype=current.e1.probabilities.dtype,
            device=pair_mask.device,
        )
        e1_previous = torch.zeros_like(e1_current)
        e2_event_current = torch.zeros(
            (batch_size, pair_width, 4),
            dtype=current.e2.event_probabilities.dtype,
            device=pair_mask.device,
        )
        e2_event_previous = torch.zeros_like(e2_event_current)
        e2_phase_current = torch.zeros_like(e2_event_current)
        e2_phase_previous = torch.zeros_like(e2_event_current)
        counts: list[int] = []
        for row, row_pairs in enumerate(pairs):
            counts.append(len(row_pairs))
            for pair, (current_index, previous_index, time_aligned) in enumerate(row_pairs):
                pair_mask[row, pair] = True
                alignment[row, pair] = time_aligned
                current_positions[row, pair] = current.e1.position_ids[row, current_index]
                previous_positions[row, pair] = previous.event_position_ids[row, previous_index]
                current_times[row, pair] = current.e1.timestamps[row, current_index]
                previous_times[row, pair] = previous.event_timestamps[row, previous_index]
                e1_current[row, pair] = current.e1.probabilities[row, current_index]
                e1_previous[row, pair] = previous.e1_probabilities[row, previous_index]
                e2_event_current[row, pair] = current.e2.event_probabilities[row, current_index]
                e2_event_previous[row, pair] = previous.e2_event_probabilities[row, previous_index]
                e2_phase_current[row, pair] = current.e2.phase_probabilities[row, current_index]
                e2_phase_previous[row, pair] = previous.e2_phase_probabilities[row, previous_index]
        event = EventConsistencyInput(
            e1=E1ConsistencyInput(
                current_probabilities=e1_current,
                previous_target_probabilities=e1_previous,
                pair_mask=pair_mask,
                alignment_mask=alignment,
                current_position_ids=current_positions,
                previous_position_ids=previous_positions,
                current_timestamps=current_times,
                previous_timestamps=previous_times,
            ),
            e2=E2ConsistencyInput(
                current_event_probabilities=e2_event_current,
                previous_event_target_probabilities=e2_event_previous,
                current_phase_probabilities=e2_phase_current,
                previous_phase_target_probabilities=e2_phase_previous,
                pair_mask=pair_mask.clone(),
                alignment_mask=alignment.clone(),
                current_position_ids=current_positions.clone(),
                previous_position_ids=previous_positions.clone(),
                current_timestamps=current_times.clone(),
                previous_timestamps=previous_times.clone(),
            ),
        )
        return event, tuple(counts)


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
class InnerUpdateAudit:
    support_index: int
    start_time: float
    end_time: float
    fast_versions_before: tuple[int, ...]
    fast_versions_observed: tuple[int, ...]
    fast_versions_after: tuple[int, ...]
    did_update: tuple[bool, ...]
    skip_reasons: tuple[str | None, ...]
    gradient_norms: tuple[float | None, ...]
    update_norms: tuple[float, ...]
    pred_valid_counts: tuple[int, ...]
    identity_valid_counts: tuple[int, ...]
    e1_valid_counts: tuple[int, ...]
    e2_valid_counts: tuple[int, ...]
    match: CrossChunkMatchAudit
    runtime_detached: bool
    next_only_verified: bool

    def __post_init__(self) -> None:
        batch_size = len(self.fast_versions_before)
        aligned = (
            self.fast_versions_observed,
            self.fast_versions_after,
            self.did_update,
            self.skip_reasons,
            self.gradient_norms,
            self.update_norms,
            self.pred_valid_counts,
            self.identity_valid_counts,
            self.e1_valid_counts,
            self.e2_valid_counts,
        )
        if batch_size <= 0 or any(len(values) != batch_size for values in aligned):
            raise ValueError("Inner update audit fields must align to the owner batch")
        if self.fast_versions_observed != self.fast_versions_before:
            raise ValueError("current Support was not observed with its before-update weights")
        if not self.next_only_verified:
            raise ValueError("Meta-TTT update failed the next-chunk-only audit")


@dataclass(frozen=True, slots=True)
class TruncatedSegmentAudit:
    """One bounded second-order graph segment and its local backward boundary."""

    segment_index: int
    support_start_index: int
    support_end_index: int
    support_count: int
    auxiliary_loss: float
    deferred_vjp_norm: float
    query_role: str
    query_weight: float
    fast_version_at_query: tuple[int, ...]
    backward_applied: bool
    includes_query_backward: bool
    reanchored: bool
    reanchor_audits: tuple[FastReanchorAudit, ...]

    def __post_init__(self) -> None:
        integers = (
            self.segment_index,
            self.support_start_index,
            self.support_end_index,
            self.support_count,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("truncated segment indices/counts must be non-negative integers")
        if self.support_count <= 0:
            raise ValueError("a truncated segment must contain at least one Support")
        if self.support_end_index - self.support_start_index + 1 != self.support_count:
            raise ValueError("truncated segment Support range does not match its count")
        if not math.isfinite(self.auxiliary_loss) or self.auxiliary_loss < 0.0:
            raise ValueError("truncated segment auxiliary loss must be finite and non-negative")
        if not math.isfinite(self.deferred_vjp_norm) or self.deferred_vjp_norm < 0.0:
            raise ValueError("truncated segment deferred VJP norm must be finite and non-negative")
        if self.query_role not in {"intermediate", "final"} or self.query_weight != 1.0:
            raise ValueError("truncated segment must carry one unit-weight Meta Query")
        if not self.fast_version_at_query or any(
            type(value) is not int or value < 0 for value in self.fast_version_at_query
        ):
            raise ValueError("truncated segment Query fast versions must be non-negative integers")
        flags = (self.backward_applied, self.includes_query_backward, self.reanchored)
        if any(type(value) is not bool for value in flags):
            raise TypeError("truncated segment flags must be bool")
        if not self.backward_applied:
            raise ValueError("every truncated segment must contribute one backward collective")
        if not self.includes_query_backward:
            raise ValueError("every A5 segment must close with one Meta Query")
        if self.reanchored != bool(self.reanchor_audits):
            raise ValueError("segment re-anchor flag and audits disagree")


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
    fast_versions: tuple[int, ...]
    metrics: QueryMetricSnapshot
    prefill_count: int
    observation_immutable: bool
    proxy_gradient_norms: tuple[float, ...]
    proxy_storage_isolated: bool
    proxy_max_abs_value_drift: float

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
        if not self.proxy_storage_isolated:
            raise ValueError("Query proxy fast matrices must use isolated storage")
        if (
            not math.isfinite(self.proxy_max_abs_value_drift)
            or self.proxy_max_abs_value_drift != 0.0
        ):
            raise ValueError("Query proxy fast values must exactly match authoritative W_after")


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeAudit:
    """Bounded-memory evidence for an otherwise unbounded numeric fast trajectory."""

    active_terms: tuple[str, ...]
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
    update_attempt_count: int
    update_count: int
    skip_count: int
    parameter_versions_unchanged_before_outer_step: bool
    overlap_graph_detached: bool
    support_supervision_reachable: bool
    training_counterfactual_executed: bool
    segments: tuple[TruncatedSegmentAudit, ...]
    updates: tuple[InnerUpdateAudit, ...]
    queries: tuple[TruncatedQueryPointAudit, ...]

    def __post_init__(self) -> None:
        if self.loss_weight not in (0.0, 1.0):
            raise ValueError("A5 episode audit loss weight must be deterministic zero or one")
        if self.prewarm_count != 1:
            raise ValueError("A5 production episodes require exactly one no-update prewarm")
        if self.truncation_horizon <= 0:
            raise ValueError("truncation horizon must be positive")
        expected_segments = self.query_count
        expected_backwards = 2 * expected_segments
        if self.segment_count != expected_segments:
            raise ValueError("A5 graph segment count must equal Meta Query count")
        if self.query_backward_count != self.query_count:
            raise ValueError("A5 must backward every Query point exactly once")
        if self.deferred_vjp_backward_count != self.segment_count:
            raise ValueError("A5 must execute one deferred fast-state VJP per segment")
        if self.backward_count != expected_backwards:
            raise ValueError("A5 streamed Query/segment backward count drifted")
        if self.truncation_count != self.segment_count:
            raise ValueError("A5 must re-anchor after every supervised segment")
        if (
            type(self.diagnostic_query_count) is not int
            or self.diagnostic_query_count < 0
            or self.zero_support_query_count != 0
            or self.support_segments_without_query != 0
        ):
            raise ValueError("A5 Query/Support alignment audit failed")
        if type(self.insufficient_inter_query_gap) is not bool:
            raise TypeError("A5 insufficient-gap audit must be bool")
        if self.maximum_retained_support_graphs > self.truncation_horizon:
            raise ValueError("A5 retained more than K Support graphs")
        if self.update_attempt_count != self.update_count + self.skip_count:
            raise ValueError("A5 update attempts must equal accepted plus skipped")
        if len(self.segments) != self.segment_count:
            raise ValueError("A5 segment audit count drifted")
        if len(self.updates) != self.support_count or len(self.queries) != self.query_count:
            raise ValueError("A5 detailed audit counts drifted")
        if any(
            segment.support_count != query.support_count
            for segment, query in zip(self.segments, self.queries, strict=True)
        ):
            raise ValueError("A5 segment and Meta Query Support counts drifted")
        if not self.parameter_versions_unchanged_before_outer_step:
            raise ValueError("outer parameters changed before the episode-level optimizer step")
        if not self.overlap_graph_detached:
            raise ValueError("A5 overlap snapshots retained an autograd graph")
        if self.support_supervision_reachable:
            raise ValueError("Support labels became reachable from the A5 inner path")
        if self.training_counterfactual_executed:
            raise ValueError("the production A5 path must not execute static-W0 counterfactuals")


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeOutput:
    """Detached logging values returned after all segment backward calls have completed."""

    total: Tensor
    query_loss: Tensor
    support_auxiliary_loss: Tensor
    final_fast_states: tuple[FastWeightsState, ...]
    final_optimizer_states: tuple[OptimizerRuntimeState, ...]
    final_runtime: BatchRuntimeState
    audit: TruncatedMetaTTTEpisodeAudit

    def __post_init__(self) -> None:
        values = (self.total, self.query_loss, self.support_auxiliary_loss)
        if any(value.ndim != 0 or value.dtype != torch.float32 for value in values):
            raise ValueError("truncated A5 logging losses must be detached FP32 scalars")
        if any(value.requires_grad or value.grad_fn is not None for value in values):
            raise ValueError("truncated A5 output must not retain completed autograd graphs")
        if any(not bool(torch.isfinite(value).item()) for value in values):
            raise ValueError("truncated A5 logging losses must be finite")
        expected = self.query_loss + self.support_auxiliary_loss
        if not torch.allclose(self.total, expected, atol=1.0e-7, rtol=1.0e-7):
            raise ValueError("truncated A5 total must equal Query plus normalized Support loss")


@dataclass(slots=True)
class _Trajectory:
    runtime: BatchRuntimeState

    @property
    def fast_states(self) -> tuple[FastWeightsState, ...]:
        return self.runtime.fast_states

    @fast_states.setter
    def fast_states(self, values: tuple[FastWeightsState, ...]) -> None:
        self.runtime = self.runtime.with_fast_states(values)

    @property
    def optimizer_states(self) -> tuple[OptimizerRuntimeState, ...]:
        return self.runtime.optimizer_states

    @optimizer_states.setter
    def optimizer_states(self, values: tuple[OptimizerRuntimeState, ...]) -> None:
        self.runtime = self.runtime.with_fast_states(self.fast_states, values)


class MetaTTTEpisodeRunner:
    """Run the production A5 trajectory with bounded second-order graph segments."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        model: StateTTTModel,
        fast_controller: FastStateController,
        predictor: TemporalPredictor,
        runtime_resetter: EpisodeRuntimeResetter,
        ttt_input_builder: CausalOverlapTTTInputBuilder | None = None,
        query_loss_builder: MetaQueryLossBuilder | None = None,
        query_encoder_reuse: bool = False,
        raw_support_visual_batcher: RawSupportVisualBatcher | None = None,
        support_visual_batch_size: int = 1,
        query_activation_offload: bool = False,
        outer_composer: OfficialWeakOuterLossComposer | None = None,
    ) -> None:
        if not isinstance(config, ProjectConfig):
            raise TypeError("Meta-TTT runner requires validated ProjectConfig")
        if not isinstance(model, StateTTTModel):
            raise TypeError("Meta-TTT runner requires StateTTTModel")
        self.config = config
        self.model = model
        self.fast_controller = fast_controller
        self.predictor = predictor
        self.runtime_resetter = runtime_resetter
        self.enabled_terms = _SUPPORTED_TERMS
        self.ttt_input_builder = ttt_input_builder or CausalOverlapTTTInputBuilder(config)
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
        self.last_balance_audit: OfficialWeakBalanceAudit | None = None
        if config.fast_ttt.optimizer.meta_gradient_mode != "full_second_order":
            raise ValueError("the training runner only permits full_second_order inner updates")

    def run_truncated(
        self,
        episode: MetaTTTEpisode,
        *,
        backward: Callable[[Tensor, bool], None] | None = None,
        backward_gradient_scale: float = 1.0,
        episode_loss_weight: float = 1.0,
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
        backward_fn = backward or _plain_backward
        self.model.train()
        self.predictor.train()
        adapted = self._reset_trajectory(episode.owner, differentiable=True)
        tracked_parameters = _unique_parameters(
            (*self.model.parameters(), *self.predictor.parameters())
        )
        versions_before = tuple(parameter._version for parameter in tracked_parameters)
        horizon = self.config.a5.truncation_horizon
        support_count = len(episode.support_chunks)
        auxiliary_scale = float(self.config.loss.auxiliary_outer_weight) / support_count
        update_audits: list[InnerUpdateAudit] = []
        segment_audits: list[TruncatedSegmentAudit] = []
        query_audits: list[TruncatedQueryPointAudit] = []
        maximum_retained = 0
        support_offset = 0
        device = adapted.fast_states[0].w_t_1.device
        query_loss_detached = torch.zeros((), dtype=torch.float32, device=device)
        support_total_detached = torch.zeros((), dtype=torch.float32, device=device)
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
                    with_grad=True,
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
        if any(prewarm_fast_audit.fast_versions):
            raise ValueError("the no-update prewarm must observe the initial W0 generation")
        adapted.runtime = _runtime_from_observation(prewarm_observation, episode.owner)
        previous_snapshot = OnlineOverlapSnapshot.capture(
            prewarm_observation,
            end_time=prewarm.end_time,
        )
        del prewarm_observation

        for segment_index, (segment_length, query, query_role, query_weight) in enumerate(
            zip(
                episode.segment_lengths,
                episode.query_points,
                episode.query_roles,
                episode.query_weights,
                strict=True,
            )
        ):
            segment_start = support_offset
            raw_segment = episode.support_chunks[
                support_offset : support_offset + segment_length
            ]
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
            segment_outputs: list[TTTLossOutput] = []
            segment_query = first_segment_query if segment_index == 0 else None
            for segment_offset, chunk in enumerate(active_segment):
                support_index = support_offset + segment_offset
                if self.query_encoder_reuse and segment_query is None:
                    segment_query = self._prepare_query(chunk, adapted, with_grad=True)
                elif segment_query is not None:
                    segment_query.validate_for(chunk.request.query_input)
                observation, fast_audit = self._observe(
                    chunk,
                    adapted,
                    support_lifecycle,
                    seed=episode.seed + support_index + 1,
                    with_grad=True,
                    prepared_query=segment_query,
                )
                adapted.runtime = _runtime_from_observation(observation, episode.owner)
                built = self.ttt_input_builder(
                    observation,
                    previous=previous_snapshot,
                    current_end_time=chunk.end_time,
                    enabled_terms=self.enabled_terms,
                )
                ttt_output = compute_ttt_loss(self.predictor, built.inputs)
                _validate_variant_loss_terms(ttt_output, self.enabled_terms)
                before_versions = tuple(state.fast_version for state in adapted.fast_states)
                results = functional_sgd_steps_from_ttt(
                    ttt_output=ttt_output,
                    fast_states=adapted.fast_states,
                    optimizer_config=self.config.fast_ttt.optimizer,
                    optimizer_states=adapted.optimizer_states,
                )
                adapted.fast_states = tuple(result.fast_state for result in results)
                adapted.optimizer_states = tuple(result.optimizer_state for result in results)
                after_versions = tuple(state.fast_version for state in adapted.fast_states)
                update_audits.append(
                    _make_inner_update_audit(
                        support_index=support_index,
                        chunk=chunk,
                        before_versions=before_versions,
                        observed_fast_audit=fast_audit,
                        results=results,
                        ttt_output=ttt_output,
                        match=built.audit,
                        runtime=adapted.runtime,
                        after_versions=after_versions,
                    )
                )
                previous_snapshot = built.snapshot
                segment_outputs.append(ttt_output)
                maximum_retained = max(maximum_retained, len(segment_outputs))
                support_total_detached = (
                    support_total_detached + ttt_output.total.detach().to(torch.float32)
                )
                del results, ttt_output, built, observation

            query_runtime_snapshot = adapted.runtime
            balance_audit: OfficialWeakBalanceAudit | None = None
            if all(query.supervision.official_weak) or bool(
                getattr(self.query_loss_builder, "streamed_balance_calibration", False)
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
                    seed=episode.seed + 10_000 + segment_index,
                    with_grad=False,
                    prepared_query=calibration_prepared_query,
                )
                calibration_output = self._answer(
                    query,
                    calibration_observation,
                    calibration_lifecycle,
                    with_grad=False,
                )
                self._balance_query_objectives(
                    (self._query_objective(query, calibration_output, ()),),
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

            authoritative_fast_states = query_runtime_snapshot.fast_states
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
                    seed=episode.seed + 10_000 + segment_index,
                    with_grad=True,
                    prepared_query=prepared_query,
                )
                observation_versions = _tensor_version_signature(observation)
                output = self._answer(query, observation, query_lifecycle, with_grad=True)
            immutable = observation_versions == _tensor_version_signature(observation)
            objective = self._query_objective(query, output, ())
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
            if isinstance(activation_scope, QueryActivationOffloadScope):
                activation_scope.release()
            if balance_audit is not None and gradient_statistics is not None:
                balance_audit = self.outer_composer.commit_streamed_gradients(
                    (gradient_statistics,),
                    balance_audit,
                )
                self.last_balance_audit = balance_audit
            deferred_vjp_norm = float(
                torch.stack(
                    tuple(
                        gradient.detach().to(torch.float32).square().sum()
                        for gradient in captured_gradients
                    )
                )
                .sum()
                .sqrt()
                .item()
            )
            deferred_vjp = deferred_fast_vjp_loss(
                authoritative_fast_states,
                captured_gradients,
            )
            segment_loss = (
                episode_weight
                * auxiliary_scale
                * torch.stack(tuple(item.total for item in segment_outputs)).sum()
            )
            backward_fn(segment_loss + deferred_vjp, False)
            maximum_proxy_drift = max(
                value for audit in proxy_audits for value in audit.max_abs_value_drift
            )
            fast_versions = tuple(state.fast_version for state in proxy_states)
            query_audits.append(
                TruncatedQueryPointAudit(
                    query_index=segment_index,
                    query_role=query_role,
                    query_weight=query_weight,
                    support_count=segment_length,
                    weighted_outer_loss=float(query_loss.detach().item()),
                    task_name=query.task_name,
                    case_id=query.case_id,
                    query_time=query.query_time,
                    observation_end_time=query.chunk.end_time,
                    fast_versions=fast_versions,
                    metrics=objective.metrics,
                    prefill_count=query_lifecycle.audit().prefill_count,
                    observation_immutable=immutable,
                    proxy_gradient_norms=proxy_gradient_norms,
                    proxy_storage_isolated=all(
                        audit.storage_isolated for audit in proxy_audits
                    ),
                    proxy_max_abs_value_drift=maximum_proxy_drift,
                )
            )
            for state in proxy_states:
                state.w_t_1.grad = None
                state.w_t_2.grad = None
            adapted.runtime = query_runtime_snapshot
            reanchor_audits = self._reanchor_trajectory(adapted)
            segment_audits.append(
                TruncatedSegmentAudit(
                    segment_index=segment_index,
                    support_start_index=segment_start,
                    support_end_index=segment_start + segment_length - 1,
                    support_count=segment_length,
                    auxiliary_loss=float(segment_loss.detach().item()),
                    deferred_vjp_norm=deferred_vjp_norm,
                    query_role=query_role,
                    query_weight=query_weight,
                    fast_version_at_query=fast_versions,
                    backward_applied=True,
                    includes_query_backward=True,
                    reanchored=True,
                    reanchor_audits=reanchor_audits,
                )
            )
            trace_event(
                "a5_query_aligned_segment_released",
                segment_index=segment_index,
                segment_count=len(episode.segment_lengths),
                query_role=query_role,
                support_count=segment_length,
                query_weight=query_weight,
                diagnostic_query_count=episode.diagnostic_query_count,
                deferred_vjp_norm=deferred_vjp_norm,
                cuda_allocated_bytes=(
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
                cuda_reserved_bytes=(
                    torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                ),
                offload_live_bytes=(
                    query_offload_budget.claimed_bytes
                    if query_offload_budget is not None
                    else 0
                ),
            )
            support_offset += segment_length
            del (
                captured_gradients,
                deferred_vjp,
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
                segment_loss,
                segment_outputs,
                segment_query,
            )

        if support_offset != support_count:
            raise RuntimeError("A5 supervised segments did not consume every Support")
        versions_after = tuple(parameter._version for parameter in tracked_parameters)
        attempted = sum(len(audit.did_update) for audit in update_audits)
        updated = sum(sum(audit.did_update) for audit in update_audits)
        detached_query = query_loss_detached.detach().clone()
        detached_auxiliary = (
            episode_weight * auxiliary_scale * support_total_detached
        ).detach().clone()
        detached_total = (detached_query + detached_auxiliary).detach().clone()
        query_count = len(episode.query_points)
        audit = TruncatedMetaTTTEpisodeAudit(
            active_terms=self.enabled_terms,
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
            backward_count=2 * query_count,
            query_backward_count=query_count,
            deferred_vjp_backward_count=query_count,
            truncation_count=len(segment_audits),
            maximum_retained_support_graphs=maximum_retained,
            update_attempt_count=attempted,
            update_count=updated,
            skip_count=attempted - updated,
            parameter_versions_unchanged_before_outer_step=versions_before == versions_after,
            overlap_graph_detached=all(item.match.snapshot_detached for item in update_audits),
            support_supervision_reachable=False,
            training_counterfactual_executed=False,
            segments=tuple(segment_audits),
            updates=tuple(update_audits),
            queries=tuple(query_audits),
        )
        return TruncatedMetaTTTEpisodeOutput(
            total=detached_total,
            query_loss=detached_query,
            support_auxiliary_loss=detached_auxiliary,
            final_fast_states=adapted.fast_states,
            final_optimizer_states=adapted.optimizer_states,
            final_runtime=adapted.runtime,
            audit=audit,
        )

    def _validate_truncated_episode(self, episode: MetaTTTEpisode) -> None:
        stage = self.config.a5
        if episode.prewarm_chunk is None:
            raise ValueError("A5 production requires an explicit S0 no-update prewarm chunk")
        if not episode.query_points:
            raise ValueError("A5 production requires at least one Meta Query")
        if len(episode.query_points) != len(episode.segment_lengths):
            raise ValueError("A5 production requires one Meta Query per Support segment")
        if any(length > stage.truncation_horizon for length in episode.segment_lengths):
            raise ValueError("A5 production segment exceeds the configured truncation horizon")
        if episode.seed != stage.seed:
            raise ValueError("A5 episode seed must equal the fixed Stage C seed")
        if self.config.fast_ttt.optimizer.momentum != 0.0:
            raise ValueError("truncated A5 currently requires stateless momentum=0 Inner SGD")

    @staticmethod
    def _reanchor_trajectory(
        trajectory: _Trajectory,
    ) -> tuple[FastReanchorAudit, ...]:
        pairs = tuple(reanchor_fast_state(state) for state in trajectory.fast_states)
        trajectory.fast_states = tuple(state for state, _ in pairs)
        values = tuple(value for state in trajectory.fast_states for value in state.fast_parameters)
        if len({tensor_storage_key(value) for value in values}) != len(values):
            raise ValueError("re-anchored batched fast states must remain storage-isolated")
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
        optimizer_states = tuple(
            reset_optimizer_state(self.config.fast_ttt.optimizer) for _ in owner.video_ids
        )
        if any(
            state.fast_version or state.update_count or state.skip_count for state in fast_states
        ):
            raise ValueError("fresh Meta-TTT fast states must reset all counters")
        if any(state.optimizer_name != "sgd" for state in optimizer_states):
            raise ValueError("fresh Meta-TTT optimizer state must use SGD")
        return _Trajectory(runtime.with_fast_states(fast_states, optimizer_states))

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
            raise ValueError("Meta-TTT observe did not consume the managed FastWeightsState")
        expected = tuple(state.fast_version for state in trajectory.fast_states)
        if audit.fast_versions != expected:
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
        if (
            isinstance(output, QueryEncoderOutput)
            and cache is not None
            and not reset_soft_state
        ):
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
        with_grad: bool,
    ) -> StateTTTModelOutput:
        answer = query.answer
        request = AnswerQueryRequest(
            owner=observation.owner,
            observation=observation,
            base_input_ids=answer.base_input_ids,
            base_attention_mask=answer.base_attention_mask,
            pixel_values_videos=answer.pixel_values_videos,
            video_grid_thw=answer.video_grid_thw,
            tokenizer=answer.tokenizer,
            embedding_owner=answer.embedding_owner,
            rope_indexer=answer.rope_indexer,
            qwen_kwargs=answer.qwen_kwargs,
        )
        with torch.set_grad_enabled(with_grad):
            return self.model.prefill_answer(
                self.model.prepare_answer(request, lifecycle),
                lifecycle,
            )

    def _query_objective(
        self,
        query: MetaTTTQueryPoint,
        output: StateTTTModelOutput,
        support: tuple[TTTLossOutput, ...],
    ) -> MetaQueryObjective:
        with trace_cuda_phase("outer_loss", stage="a5_query"):
            inputs = self.query_loss_builder(
                output,
                answer=query.answer,
                supervision=query.supervision,
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
                    support_ttt=support,
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
            auxiliary = objective.outer.auxiliary_ttt
            total = outer.outer + objective.outer.auxiliary_weight * auxiliary.value
            balanced_outer = replace(outer, auxiliary_ttt=auxiliary, total=total)
            outputs.append(
                replace(
                    objective,
                    outer=balanced_outer,
                )
            )
        return tuple(outputs)


def _make_inner_update_audit(
    *,
    support_index: int,
    chunk: MetaCausalChunk,
    before_versions: tuple[int, ...],
    observed_fast_audit: FastTTTForwardAudit,
    results: tuple[FunctionalSGDResult, ...],
    ttt_output: TTTLossOutput,
    match: CrossChunkMatchAudit,
    runtime: object,
    after_versions: tuple[int, ...],
) -> InnerUpdateAudit:
    expected_after = tuple(
        before + int(result.did_update)
        for before, result in zip(before_versions, results, strict=True)
    )
    next_only = (
        observed_fast_audit.fast_versions == before_versions and after_versions == expected_after
    )
    return InnerUpdateAudit(
        support_index=support_index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        fast_versions_before=before_versions,
        fast_versions_observed=observed_fast_audit.fast_versions,
        fast_versions_after=after_versions,
        did_update=tuple(result.did_update for result in results),
        skip_reasons=tuple(
            None if result.skip_reason is None else result.skip_reason.value for result in results
        ),
        gradient_norms=tuple(result.gradient_norm for result in results),
        update_norms=tuple(result.update_norm for result in results),
        pred_valid_counts=tuple(int(value.item()) for value in ttt_output.pred.valid_counts),
        identity_valid_counts=tuple(
            int(value.item()) for value in ttt_output.identity.valid_counts
        ),
        e1_valid_counts=tuple(int(value.item()) for value in ttt_output.e1_event.valid_counts),
        e2_valid_counts=tuple(int(value.item()) for value in ttt_output.e2_event.valid_counts),
        match=match,
        runtime_detached=not _contains_grad_tensor(runtime),
        next_only_verified=next_only,
    )


def _validate_variant_loss_terms(
    output: TTTLossOutput,
    enabled_terms: tuple[str, ...],
) -> None:
    if "identity" not in enabled_terms and bool(output.identity.valid_counts.any().item()):
        raise ValueError("disabled identity loss produced valid inner terms")
    if "event" not in enabled_terms and bool(output.event.valid_counts.any().item()):
        raise ValueError("disabled event loss produced valid inner terms")
    expected = output.pred.per_row
    if "identity" in enabled_terms:
        expected = expected + 0.5 * output.identity.per_row
    if "event" in enabled_terms:
        expected = expected + 0.5 * output.event.per_row
    if not torch.allclose(output.per_row_total, expected, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("active Meta-TTT terms do not equal the audited variant objective")


def _typed_observations(output: ObservationChunkOutput) -> ObservationOutputs:
    if not isinstance(output.observations, ObservationOutputs):
        raise TypeError("Meta-TTT overlap builder requires ObservationOutputs")
    return output.observations


def _match_event_positions(
    current: ObservationOutputs,
    previous: OnlineOverlapSnapshot,
) -> tuple[tuple[tuple[int, int, bool], ...], ...]:
    rows: list[tuple[tuple[int, int, bool], ...]] = []
    for row in range(current.e1.valid_mask.shape[0]):
        previous_by_position = {
            int(previous.event_position_ids[row, index].item()): index
            for index in torch.nonzero(previous.event_valid_mask[row], as_tuple=False)
            .flatten()
            .tolist()
        }
        pairs: list[tuple[int, int, bool]] = []
        seen: set[int] = set()
        for current_index in (
            torch.nonzero(current.e1.valid_mask[row], as_tuple=False).flatten().tolist()
        ):
            position = int(current.e1.position_ids[row, current_index].item())
            previous_index = previous_by_position.get(position)
            if previous_index is None:
                continue
            if position in seen:
                raise ValueError("event overlap position must be unique per current row")
            seen.add(position)
            time_aligned = (
                abs(
                    float(current.e1.timestamps[row, current_index].item())
                    - float(previous.event_timestamps[row, previous_index].item())
                )
                <= 1.0e-6
            )
            pairs.append((current_index, previous_index, time_aligned))
        rows.append(tuple(pairs))
    return tuple(rows)


def _pad_probability_width(values: Tensor, width: int) -> Tensor:
    if values.shape[1] == width:
        return values
    if values.shape[1] > width:
        return values[:, :width]
    padding = torch.zeros(
        (values.shape[0], width - values.shape[1], values.shape[2]),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat((values, padding), dim=1)


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
    states: Sequence[FastWeightsState],
    *,
    backward_gradient_scale: float,
) -> tuple[tuple[Tensor, ...], tuple[float, ...]]:
    """Capture unscaled Query cotangents before releasing its isolated proxy state."""

    values = tuple(value for state in states for value in state.fast_parameters)
    if not values:
        raise ValueError("Query proxy gradient capture requires fast matrices")
    gradients: list[Tensor] = []
    norms: list[float] = []
    for value in values:
        gradient = value.grad
        if gradient is None:
            raise ValueError("Query loss did not produce a gradient for every proxy fast matrix")
        unscaled = gradient.detach().float().clone().div_(backward_gradient_scale)
        if not bool(torch.isfinite(unscaled).all()):
            raise ValueError("Query proxy gradient must be finite after backward unscale")
        gradients.append(unscaled)
        norms.append(float(torch.linalg.vector_norm(unscaled).cpu().item()))
    return tuple(gradients), tuple(norms)


class _SeededRNG:
    def __init__(self, seed: int, devices: tuple[int, ...]) -> None:
        self.seed = seed
        self.context = torch.random.fork_rng(devices=list(devices))

    def __enter__(self) -> None:
        self.context.__enter__()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.context.__exit__(exc_type, exc, traceback)


def _seeded_rng(seed: int, states: Sequence[FastWeightsState]) -> _SeededRNG:
    devices = tuple(
        sorted(
            {
                cast(int, state.w_t_1.device.index)
                for state in states
                if state.w_t_1.device.type == "cuda" and state.w_t_1.device.index is not None
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
