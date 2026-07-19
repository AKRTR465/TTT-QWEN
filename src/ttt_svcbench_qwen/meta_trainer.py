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

from ttt_svcbench_qwen.config import (
    MetaTTTVariant,
    ProjectConfig,
)
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.fast_ttt import (
    FastReanchorAudit,
    FastTTTForwardAudit,
    FastWeightsState,
    OptimizerRuntimeState,
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
    PrefillLifecycle,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
)
from ttt_svcbench_qwen.observation_heads import ObservationOutputs
from ttt_svcbench_qwen.query_encoder import QueryEncoderOutput
from ttt_svcbench_qwen.stage_a_runtime import StageAWriteAudit
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakStateLossOutput,
    OfficialWeakTargetBuilder,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_encoder import TemporalEncoderOutput
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageASupervisionBatch,
)

_SUPPORTED_TERMS = ("pred", "identity", "event")
_LEGACY_FULL_GRAPH_BOUND = 8
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

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Meta-TTT episode seed must be a non-negative integer")
        support_count = len(self.support_chunks)
        if support_count < 1:
            raise ValueError("Meta-TTT episodes require at least one Support chunk")
        if not self.query_points:
            raise ValueError("Meta-TTT episodes require at least one later Query point")
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
        if query_ends[0] <= support_ends[-1]:
            raise ValueError("the first Query observation must be later than all Support chunks")
        if any(right <= left for left, right in pairwise(query_ends)):
            raise ValueError("Query observation end times must advance strictly")
        if any(right <= left for left, right in pairwise(query_times)):
            raise ValueError("Query points must advance strictly in causal time")


@dataclass(frozen=True, slots=True)
class DetachedOverlapSnapshot:
    """Minimal previous-chunk sources; every tensor is detached and storage-isolated."""

    owner: RuntimeOwner
    end_time: float
    identity: Tensor
    identity_valid_mask: Tensor
    identity_position_ids: Tensor
    identity_timestamps: Tensor
    e1_probabilities: Tensor
    e2_event_probabilities: Tensor
    e2_phase_probabilities: Tensor
    event_valid_mask: Tensor
    event_position_ids: Tensor
    event_timestamps: Tensor

    def __post_init__(self) -> None:
        tensors = self.tensors
        if any(value.requires_grad or value.grad_fn is not None for value in tensors):
            raise ValueError("overlap snapshots must be detached from autograd")
        materialized = tuple(value for value in tensors if value.device.type != "meta")
        if len({_storage_key(value) for value in materialized}) != len(materialized):
            raise ValueError("overlap snapshot tensors must use isolated storage")
        if not math.isfinite(self.end_time) or self.end_time < 0.0:
            raise ValueError("overlap snapshot end_time must be finite and non-negative")

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.identity,
            self.identity_valid_mask,
            self.identity_position_ids,
            self.identity_timestamps,
            self.e1_probabilities,
            self.e2_event_probabilities,
            self.e2_phase_probabilities,
            self.event_valid_mask,
            self.event_position_ids,
            self.event_timestamps,
        )

    @classmethod
    def capture(
        cls,
        output: ObservationChunkOutput,
        *,
        end_time: float,
    ) -> DetachedOverlapSnapshot:
        observations = _typed_observations(output)
        return cls(
            owner=output.owner,
            end_time=end_time,
            identity=observations.o2.identity.detach().clone(),
            identity_valid_mask=observations.o2.valid_mask.detach().clone(),
            identity_position_ids=observations.o2.position_ids.detach().clone(),
            identity_timestamps=observations.o2.timestamps.detach().clone(),
            e1_probabilities=observations.e1.probabilities.detach().clone(),
            e2_event_probabilities=observations.e2.event_probabilities.detach().clone(),
            e2_phase_probabilities=observations.e2.phase_probabilities.detach().clone(),
            event_valid_mask=observations.e1.valid_mask.detach().clone(),
            event_position_ids=observations.e1.position_ids.detach().clone(),
            event_timestamps=observations.e1.timestamps.detach().clone(),
        )


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
    snapshot: DetachedOverlapSnapshot
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
        previous: DetachedOverlapSnapshot | None,
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
        snapshot = DetachedOverlapSnapshot.capture(output, end_time=current_end_time)
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
        snapshot_isolated = len({_storage_key(value) for value in snapshot_tensors}) == len(
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
        previous: DetachedOverlapSnapshot | None,
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
        previous: DetachedOverlapSnapshot,
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
        previous: DetachedOverlapSnapshot | None,
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
class QueryPointAudit:
    query_index: int
    task_name: str
    case_id: str
    query_time: float
    observation_end_time: float
    before_fast_versions: tuple[int, ...]
    after_fast_versions: tuple[int, ...]
    before: QueryMetricSnapshot
    after: QueryMetricSnapshot
    before_prefill_count: int
    after_prefill_count: int
    independent_lifecycles: bool
    observation_immutable: bool

    def __post_init__(self) -> None:
        if self.query_time < self.observation_end_time:
            raise ValueError("Query audit exposes future observation")
        if self.before_prefill_count != 1 or self.after_prefill_count != 1:
            raise ValueError("each Query branch must execute exactly one prefill")
        if not self.independent_lifecycles or not self.observation_immutable:
            raise ValueError("Query branches require isolated lifecycle and immutable observation")


@dataclass(frozen=True, slots=True)
class MetaTTTEpisodeAudit:
    variant: MetaTTTVariant
    active_terms: tuple[str, ...]
    support_count: int
    query_count: int
    runtime_reset_count: int
    fast_reset_count: int
    optimizer_reset_count: int
    update_attempt_count: int
    update_count: int
    skip_count: int
    parameter_versions_unchanged_during_inner: bool
    overlap_graph_detached: bool
    retained_support_graph_count: int
    graph_bound: int
    trajectory_reuse_strategy: str
    support_supervision_reachable: bool
    updates: tuple[InnerUpdateAudit, ...]
    queries: tuple[QueryPointAudit, ...]

    def __post_init__(self) -> None:
        if self.support_supervision_reachable:
            raise ValueError("Support labels became reachable from the Meta-TTT inner path")
        if not self.parameter_versions_unchanged_during_inner:
            raise ValueError("checkpointed parameters changed during functional inner updates")
        if not self.overlap_graph_detached:
            raise ValueError("cross-chunk overlap snapshots retained an autograd graph")
        if self.retained_support_graph_count > self.graph_bound or self.graph_bound != 8:
            raise ValueError("Meta-TTT graph lifetime exceeded the explicit 8-Support bound")
        if self.update_attempt_count != self.update_count + self.skip_count:
            raise ValueError("inner attempted updates must equal accepted plus skipped")
        if len(self.updates) != self.support_count or len(self.queries) != self.query_count:
            raise ValueError("Meta-TTT audit counts do not match detailed entries")
        if self.trajectory_reuse_strategy != "causal_replay_isolated_prefill":
            raise ValueError("Meta-TTT trajectory reuse strategy drifted")


@dataclass(frozen=True, slots=True)
class MetaTTTEpisodeOutput:
    variant: MetaTTTVariant
    total: Tensor
    support_ttt: tuple[TTTLossOutput, ...]
    query_objectives: tuple[MetaQueryObjective, ...]
    final_fast_states: tuple[FastWeightsState, ...]
    final_optimizer_states: tuple[OptimizerRuntimeState, ...]
    final_runtime: BatchRuntimeState
    audit: MetaTTTEpisodeAudit

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or self.total.dtype != torch.float32:
            raise ValueError("Meta-TTT episode total must be an FP32 scalar")
        if not bool(torch.isfinite(self.total.detach()).item()):
            raise ValueError("Meta-TTT episode total must be finite")
        expected = torch.stack(tuple(query.outer.total for query in self.query_objectives)).mean()
        if not torch.allclose(self.total.detach(), expected.detach(), atol=1.0e-7, rtol=1.0e-7):
            raise ValueError("multi-Query objective must be the mean of pointwise outer totals")


@dataclass(frozen=True, slots=True)
class TruncatedSegmentAudit:
    """One bounded second-order graph segment and its local backward boundary."""

    segment_index: int
    support_start_index: int
    support_end_index: int
    support_count: int
    auxiliary_loss: float
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
        flags = (self.backward_applied, self.includes_query_backward, self.reanchored)
        if any(type(value) is not bool for value in flags):
            raise TypeError("truncated segment flags must be bool")
        if not self.backward_applied:
            raise ValueError("every truncated segment must contribute one backward collective")
        if self.reanchored != bool(self.reanchor_audits):
            raise ValueError("segment re-anchor flag and audits disagree")


@dataclass(frozen=True, slots=True)
class TruncatedQueryPointAudit:
    """Adapted-only Query audit used by the production A5 path."""

    query_index: int
    task_name: str
    case_id: str
    query_time: float
    observation_end_time: float
    fast_versions: tuple[int, ...]
    metrics: QueryMetricSnapshot
    prefill_count: int
    observation_immutable: bool

    def __post_init__(self) -> None:
        if type(self.query_index) is not int or self.query_index < 0:
            raise ValueError("truncated Query index must be a non-negative integer")
        if not self.task_name or not self.case_id:
            raise ValueError("truncated Query task/case identifiers must be non-empty")
        if self.query_time < self.observation_end_time:
            raise ValueError("truncated Query audit exposes future observation")
        if self.prefill_count != 1:
            raise ValueError("each production Query must execute exactly one prefill")
        if not self.observation_immutable:
            raise ValueError("production Query answer mutated its observation")


@dataclass(frozen=True, slots=True)
class TruncatedMetaTTTEpisodeAudit:
    """Bounded-memory evidence for an otherwise unbounded numeric fast trajectory."""

    variant: MetaTTTVariant
    active_terms: tuple[str, ...]
    support_count: int
    query_count: int
    prewarm_count: int
    truncation_horizon: int
    segment_count: int
    backward_count: int
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
        if self.variant is not MetaTTTVariant.A5:
            raise ValueError("the production truncated path is defined only for A5")
        if self.prewarm_count != 1:
            raise ValueError("A5 production episodes require exactly one no-update prewarm")
        if self.truncation_horizon <= 0:
            raise ValueError("truncation horizon must be positive")
        expected_segments = math.ceil(self.support_count / self.truncation_horizon)
        expected_truncations = self.support_count // self.truncation_horizon
        if self.segment_count != expected_segments or self.backward_count != expected_segments:
            raise ValueError("A5 must execute exactly one backward per graph segment")
        if self.truncation_count != expected_truncations:
            raise ValueError("A5 truncation count must follow processed Support steps")
        if self.maximum_retained_support_graphs > self.truncation_horizon:
            raise ValueError("A5 retained more than K Support graphs")
        if self.update_attempt_count != self.update_count + self.skip_count:
            raise ValueError("A5 update attempts must equal accepted plus skipped")
        if len(self.segments) != self.segment_count:
            raise ValueError("A5 segment audit count drifted")
        if len(self.updates) != self.support_count or len(self.queries) != self.query_count:
            raise ValueError("A5 detailed audit counts drifted")
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
    """Run A3/A4/A5 with a static-W0 counterfactual and a differentiable adapted path."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        model: StateTTTModel,
        fast_controller: FastStateController,
        predictor: TemporalPredictor,
        runtime_resetter: EpisodeRuntimeResetter,
        variant: MetaTTTVariant,
        ttt_input_builder: CausalOverlapTTTInputBuilder | None = None,
        query_loss_builder: MetaQueryLossBuilder | None = None,
    ) -> None:
        if not isinstance(config, ProjectConfig):
            raise TypeError("Meta-TTT runner requires validated ProjectConfig")
        if not isinstance(model, StateTTTModel):
            raise TypeError("Meta-TTT runner requires StateTTTModel")
        if not isinstance(variant, MetaTTTVariant):
            raise TypeError("Meta-TTT runner variant must be MetaTTTVariant")
        self.config = config
        self.model = model
        self.fast_controller = fast_controller
        self.predictor = predictor
        self.runtime_resetter = runtime_resetter
        self.variant = variant
        self.enabled_terms = enabled_terms_for(config, variant)
        self.ttt_input_builder = ttt_input_builder or CausalOverlapTTTInputBuilder(config)
        self.query_loss_builder = query_loss_builder or StageAQueryLossBuilder()
        if config.fast_ttt.optimizer.meta_gradient_mode != "full_second_order":
            raise ValueError("the training runner only permits full_second_order inner updates")

    def __call__(self, episode: MetaTTTEpisode) -> MetaTTTEpisodeOutput:
        self._validate_episode_for_variant(episode)
        self.model.train()
        self.predictor.train()
        adapted = self._reset_trajectory(episode.owner, differentiable=True)
        baseline = self._reset_trajectory(episode.owner, differentiable=False)
        _assert_trajectory_isolation(adapted, baseline)
        tracked_parameters = _unique_parameters(
            (*self.model.parameters(), *self.predictor.parameters())
        )
        versions_before = tuple(parameter._version for parameter in tracked_parameters)
        support_outputs: list[TTTLossOutput] = []
        update_audits: list[InnerUpdateAudit] = []
        previous_snapshot: DetachedOverlapSnapshot | None = None
        adapted_lifecycle = PrefillLifecycle(episode.owner)
        baseline_lifecycle = PrefillLifecycle(episode.owner)

        for support_index, chunk in enumerate(episode.support_chunks):
            adapted_observation, adapted_fast_audit = self._observe(
                chunk,
                adapted,
                adapted_lifecycle,
                seed=episode.seed + support_index,
                with_grad=True,
            )
            baseline_observation, _ = self._observe(
                chunk,
                baseline,
                baseline_lifecycle,
                seed=episode.seed + support_index,
                with_grad=False,
            )
            adapted.runtime = _runtime_from_observation(adapted_observation, episode.owner)
            baseline.runtime = _runtime_from_observation(baseline_observation, episode.owner)
            built = self.ttt_input_builder(
                adapted_observation,
                previous=previous_snapshot,
                current_end_time=chunk.end_time,
                enabled_terms=self.enabled_terms,
            )
            ttt_output = compute_ttt_loss(self.predictor, built.inputs)
            _validate_variant_loss_terms(ttt_output, self.enabled_terms)
            versions_before_update = tuple(state.fast_version for state in adapted.fast_states)
            results = functional_sgd_steps_from_ttt(
                ttt_output=ttt_output,
                fast_states=adapted.fast_states,
                optimizer_config=self.config.fast_ttt.optimizer,
                optimizer_states=adapted.optimizer_states,
            )
            adapted.fast_states = tuple(result.fast_state for result in results)
            adapted.optimizer_states = tuple(result.optimizer_state for result in results)
            versions_after_update = tuple(state.fast_version for state in adapted.fast_states)
            update_audits.append(
                _make_inner_update_audit(
                    support_index=support_index,
                    chunk=chunk,
                    before_versions=versions_before_update,
                    observed_fast_audit=adapted_fast_audit,
                    results=results,
                    ttt_output=ttt_output,
                    match=built.audit,
                    runtime=adapted.runtime,
                    after_versions=versions_after_update,
                )
            )
            support_outputs.append(ttt_output)
            previous_snapshot = built.snapshot

        query_objectives: list[MetaQueryObjective] = []
        query_audits: list[QueryPointAudit] = []
        for query_index, query in enumerate(episode.query_points):
            seed = episode.seed + 10_000 + query_index
            after_lifecycle = PrefillLifecycle(episode.owner)
            before_lifecycle = PrefillLifecycle(episode.owner)
            if after_lifecycle is before_lifecycle:
                raise RuntimeError("Query counterfactual lifecycles unexpectedly alias")
            after_observation, _ = self._observe(
                query.chunk,
                adapted,
                after_lifecycle,
                seed=seed,
                with_grad=True,
            )
            before_observation, _ = self._observe(
                query.chunk,
                baseline,
                before_lifecycle,
                seed=seed,
                with_grad=False,
            )
            adapted.runtime = _runtime_from_observation(after_observation, episode.owner)
            baseline.runtime = _runtime_from_observation(before_observation, episode.owner)
            after_versions = _tensor_version_signature(after_observation)
            before_versions = _tensor_version_signature(before_observation)
            after_output = self._answer(query, after_observation, after_lifecycle, with_grad=True)
            before_output = self._answer(
                query, before_observation, before_lifecycle, with_grad=False
            )
            observation_immutable = after_versions == _tensor_version_signature(
                after_observation
            ) and before_versions == _tensor_version_signature(before_observation)
            after_objective = self._query_objective(query, after_output, tuple(support_outputs))
            with torch.no_grad():
                before_inputs = self.query_loss_builder(
                    before_output,
                    answer=query.answer,
                    supervision=query.supervision,
                )
                before_answer = compute_answer_loss(before_inputs.answer)
                before_state = (
                    before_inputs.state
                    if isinstance(before_inputs.state, OfficialWeakStateLossOutput)
                    else compute_state_loss(before_inputs.state)
                )
                before_metrics = _query_metrics(before_answer, before_state)
            query_objectives.append(after_objective)
            query_audits.append(
                QueryPointAudit(
                    query_index=query_index,
                    task_name=query.task_name,
                    case_id=query.case_id,
                    query_time=query.query_time,
                    observation_end_time=query.chunk.end_time,
                    before_fast_versions=tuple(
                        state.fast_version for state in baseline.fast_states
                    ),
                    after_fast_versions=tuple(state.fast_version for state in adapted.fast_states),
                    before=before_metrics,
                    after=after_objective.metrics,
                    before_prefill_count=before_lifecycle.audit().prefill_count,
                    after_prefill_count=after_lifecycle.audit().prefill_count,
                    independent_lifecycles=before_lifecycle is not after_lifecycle,
                    observation_immutable=observation_immutable,
                )
            )

        total = torch.stack(tuple(query.outer.total for query in query_objectives)).mean()
        versions_after = tuple(parameter._version for parameter in tracked_parameters)
        attempted = sum(len(audit.did_update) for audit in update_audits)
        updated = sum(sum(audit.did_update) for audit in update_audits)
        skipped = attempted - updated
        audit = MetaTTTEpisodeAudit(
            variant=self.variant,
            active_terms=self.enabled_terms,
            support_count=len(episode.support_chunks),
            query_count=len(episode.query_points),
            runtime_reset_count=2,
            fast_reset_count=2 * len(episode.owner.video_ids),
            optimizer_reset_count=2 * len(episode.owner.video_ids),
            update_attempt_count=attempted,
            update_count=updated,
            skip_count=skipped,
            parameter_versions_unchanged_during_inner=versions_before == versions_after,
            overlap_graph_detached=all(update.match.snapshot_detached for update in update_audits),
            retained_support_graph_count=len(support_outputs),
            graph_bound=_LEGACY_FULL_GRAPH_BOUND,
            trajectory_reuse_strategy="causal_replay_isolated_prefill",
            support_supervision_reachable=False,
            updates=tuple(update_audits),
            queries=tuple(query_audits),
        )
        return MetaTTTEpisodeOutput(
            variant=self.variant,
            total=total,
            support_ttt=tuple(support_outputs),
            query_objectives=tuple(query_objectives),
            final_fast_states=adapted.fast_states,
            final_optimizer_states=adapted.optimizer_states,
            final_runtime=adapted.runtime,
            audit=audit,
        )

    def run_truncated(
        self,
        episode: MetaTTTEpisode,
        *,
        backward: Callable[[Tensor], None] | None = None,
    ) -> TruncatedMetaTTTEpisodeOutput:
        """Run production A5 with exact second order inside K-step graph segments.

        ``backward`` is injectable so LLaMA-Factory/Accelerate/DeepSpeed can own the
        distributed backward call.  It is invoked exactly ``ceil(T / K)`` times;
        this method never executes an Outer optimizer step.
        """

        self._validate_truncated_episode(episode)
        backward_fn = backward or _plain_backward
        self.model.train()
        self.predictor.train()
        adapted = self._reset_trajectory(episode.owner, differentiable=True)
        tracked_parameters = _unique_parameters(
            (*self.model.parameters(), *self.predictor.parameters())
        )
        versions_before = tuple(parameter._version for parameter in tracked_parameters)
        horizon = self.config.stage_c.truncation_horizon
        support_count = len(episode.support_chunks)
        auxiliary_scale = float(self.config.loss.auxiliary_outer_weight) / support_count
        update_audits: list[InnerUpdateAudit] = []
        segment_audits: list[TruncatedSegmentAudit] = []
        segment_outputs: list[TTTLossOutput] = []
        support_total_detached: Tensor | None = None
        maximum_retained = 0
        lifecycle = PrefillLifecycle(episode.owner)

        prewarm = cast(MetaCausalChunk, episode.prewarm_chunk)
        prewarm_observation, prewarm_fast_audit = self._observe(
            prewarm,
            adapted,
            lifecycle,
            seed=episode.seed,
            with_grad=False,
        )
        if any(prewarm_fast_audit.fast_versions):
            raise ValueError("the no-update prewarm must observe the initial W0 generation")
        adapted.runtime = _runtime_from_observation(prewarm_observation, episode.owner)
        previous_snapshot = DetachedOverlapSnapshot.capture(
            prewarm_observation,
            end_time=prewarm.end_time,
        )
        del prewarm_observation

        for support_index, chunk in enumerate(episode.support_chunks):
            observation, fast_audit = self._observe(
                chunk,
                adapted,
                lifecycle,
                seed=episode.seed + support_index + 1,
                with_grad=True,
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
            detached_total = ttt_output.total.detach()
            support_total_detached = (
                detached_total
                if support_total_detached is None
                else support_total_detached + detached_total
            )

            boundary_reached = len(segment_outputs) == horizon
            final_support = support_index + 1 == support_count
            if boundary_reached and not final_support:
                segment_loss = (
                    auxiliary_scale
                    * torch.stack(tuple(output.total for output in segment_outputs)).sum()
                )
                backward_fn(segment_loss)
                reanchor_audits = self._reanchor_trajectory(adapted)
                segment_audits.append(
                    TruncatedSegmentAudit(
                        segment_index=len(segment_audits),
                        support_start_index=support_index + 1 - len(segment_outputs),
                        support_end_index=support_index,
                        support_count=len(segment_outputs),
                        auxiliary_loss=float(segment_loss.detach().item()),
                        backward_applied=True,
                        includes_query_backward=False,
                        reanchored=True,
                        reanchor_audits=reanchor_audits,
                    )
                )
                segment_outputs.clear()
                del segment_loss, results, ttt_output, built, observation

        if not segment_outputs or support_total_detached is None:
            raise RuntimeError("A5 final graph segment unexpectedly has no Support")

        query_objectives: list[MetaQueryObjective] = []
        query_audits: list[TruncatedQueryPointAudit] = []
        for query_index, query in enumerate(episode.query_points):
            query_lifecycle = PrefillLifecycle(episode.owner)
            observation, _ = self._observe(
                query.chunk,
                adapted,
                query_lifecycle,
                seed=episode.seed + 10_000 + query_index,
                with_grad=True,
            )
            adapted.runtime = _runtime_from_observation(observation, episode.owner)
            observation_versions = _tensor_version_signature(observation)
            output = self._answer(query, observation, query_lifecycle, with_grad=True)
            immutable = observation_versions == _tensor_version_signature(observation)
            objective = self._query_objective(query, output, ())
            query_objectives.append(objective)
            query_audits.append(
                TruncatedQueryPointAudit(
                    query_index=query_index,
                    task_name=query.task_name,
                    case_id=query.case_id,
                    query_time=query.query_time,
                    observation_end_time=query.chunk.end_time,
                    fast_versions=tuple(state.fast_version for state in adapted.fast_states),
                    metrics=objective.metrics,
                    prefill_count=query_lifecycle.audit().prefill_count,
                    observation_immutable=immutable,
                )
            )

        query_loss = torch.stack(tuple(item.outer.outer for item in query_objectives)).mean()
        final_segment_loss = (
            auxiliary_scale * torch.stack(tuple(output.total for output in segment_outputs)).sum()
        )
        final_loss = query_loss + final_segment_loss
        backward_fn(final_loss)
        final_reanchor_audits = self._reanchor_trajectory(adapted)
        final_support_index = support_count - 1
        segment_audits.append(
            TruncatedSegmentAudit(
                segment_index=len(segment_audits),
                support_start_index=final_support_index + 1 - len(segment_outputs),
                support_end_index=final_support_index,
                support_count=len(segment_outputs),
                auxiliary_loss=float(final_segment_loss.detach().item()),
                backward_applied=True,
                includes_query_backward=True,
                reanchored=True,
                reanchor_audits=final_reanchor_audits,
            )
        )

        versions_after = tuple(parameter._version for parameter in tracked_parameters)
        attempted = sum(len(audit.did_update) for audit in update_audits)
        updated = sum(sum(audit.did_update) for audit in update_audits)
        detached_query = query_loss.detach().clone()
        detached_auxiliary = (auxiliary_scale * support_total_detached).detach().clone()
        detached_total = (detached_query + detached_auxiliary).detach().clone()
        audit = TruncatedMetaTTTEpisodeAudit(
            variant=self.variant,
            active_terms=self.enabled_terms,
            support_count=support_count,
            query_count=len(episode.query_points),
            prewarm_count=1,
            truncation_horizon=horizon,
            segment_count=len(segment_audits),
            backward_count=len(segment_audits),
            truncation_count=support_count // horizon,
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
        stage = self.config.stage_c
        if self.variant is not MetaTTTVariant.A5 or stage.active_variant is not MetaTTTVariant.A5:
            raise ValueError("the truncated production entrypoint requires active A5")
        if not stage.direct_from_stage_a or stage.meta_gradient_mode != "truncated_second_order":
            raise ValueError("A5 production must transition directly from A2 in truncated mode")
        if stage.maximum_support_chunks is not None or stage.support_chunk_schedule:
            raise ValueError("A5 production cannot cap or enumerate Support counts")
        if episode.prewarm_chunk is None or stage.prewarm_support_chunks != 1:
            raise ValueError("A5 production requires an explicit S0 no-update prewarm chunk")
        if len(episode.query_points) < stage.minimum_query_points:
            raise ValueError("A5 production requires multiple later Query points")
        if episode.seed != stage.seed:
            raise ValueError("A5 episode seed must equal the fixed Stage C seed")
        if self.config.fast_ttt.optimizer.momentum != 0.0:
            raise ValueError("truncated A5 currently requires stateless momentum=0 Inner SGD")
        if stage.training_counterfactual_enabled:
            raise ValueError("static-W0 counterfactuals are validation-only in production A5")

    @staticmethod
    def _reanchor_trajectory(
        trajectory: _Trajectory,
    ) -> tuple[FastReanchorAudit, ...]:
        pairs = tuple(reanchor_fast_state(state) for state in trajectory.fast_states)
        trajectory.fast_states = tuple(state for state, _ in pairs)
        values = tuple(value for state in trajectory.fast_states for value in state.fast_parameters)
        if len({_storage_key(value) for value in values}) != len(values):
            raise ValueError("re-anchored batched fast states must remain storage-isolated")
        return tuple(audit for _, audit in pairs)

    def _validate_episode_for_variant(self, episode: MetaTTTEpisode) -> None:
        support_count = len(episode.support_chunks)
        query_count = len(episode.query_points)
        if self.variant is MetaTTTVariant.A3:
            if support_count != self.config.stage_b.support_chunks:
                raise ValueError("A3 requires the configured single Support chunk")
            if query_count < self.config.stage_b.minimum_query_points:
                raise ValueError("A3 requires at least the configured Query count")
            if episode.seed != self.config.stage_b.seed:
                raise ValueError("A3 episode seed must equal the fixed Stage B seed")
        else:
            if self.variant not in self.config.stage_c.variants:
                raise ValueError("Stage C variant is not present in the isolated ablation set")
            schedule = self.config.stage_c.support_chunk_schedule
            if schedule and support_count not in schedule:
                raise ValueError(
                    "Stage C Support count is outside the configured ablation schedule"
                )
            if not schedule and support_count > _LEGACY_FULL_GRAPH_BOUND:
                raise ValueError(
                    "unbounded Stage C episodes must use the truncated training entrypoint"
                )
            if query_count < self.config.stage_c.minimum_query_points:
                raise ValueError("Stage C requires multiple later Query points")
            if episode.seed != self.config.stage_c.seed:
                raise ValueError("Stage C episode seed must equal the fixed Stage C seed")

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
    ) -> tuple[ObservationChunkOutput, FastTTTForwardAudit]:
        request = replace(
            chunk.request,
            runtime_state=trajectory.runtime,
            bank_states=trajectory.runtime.bank_states,
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
            return self.model.answer_query(request, lifecycle)

    def _query_objective(
        self,
        query: MetaTTTQueryPoint,
        output: StateTTTModelOutput,
        support: tuple[TTTLossOutput, ...],
    ) -> MetaQueryObjective:
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
        return MetaQueryObjective(answer, state, outer, _query_metrics(answer, state))


def enabled_terms_for(
    config: ProjectConfig,
    variant: MetaTTTVariant,
) -> tuple[str, ...]:
    """Return the explicit ablation surface; A4/A5 differ only by ``event``."""

    terms: tuple[str, ...]
    if variant is MetaTTTVariant.A3:
        terms = config.stage_b.enabled_ttt_terms
    elif variant is MetaTTTVariant.A4:
        terms = config.stage_c.a4_enabled_ttt_terms
    elif variant is MetaTTTVariant.A5:
        terms = config.stage_c.a5_enabled_ttt_terms
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported Meta-TTT variant: {variant}")
    if tuple(term for term in _SUPPORTED_TERMS if term in terms) != terms:
        raise ValueError("Meta-TTT terms must be an ordered subset of pred/identity/event")
    return terms


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
    previous: DetachedOverlapSnapshot,
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


def _assert_trajectory_isolation(adapted: _Trajectory, baseline: _Trajectory) -> None:
    if adapted.runtime is baseline.runtime:
        raise ValueError("adapted and baseline runtimes must be independently reset")
    if any(
        left is right
        for left, right in zip(
            adapted.runtime.bank_states, baseline.runtime.bank_states, strict=True
        )
    ):
        raise ValueError("adapted and baseline Bank states must be independently reset")
    fast_values = tuple(
        value
        for trajectory in (adapted, baseline)
        for state in trajectory.fast_states
        for value in state.fast_parameters
    )
    if len({_storage_key(value) for value in fast_values}) != len(fast_values):
        raise ValueError("adapted/baseline transient fast weights must use isolated storage")


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


def _plain_backward(loss: Tensor) -> None:
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise TypeError("segment backward requires one scalar Tensor")
    if not loss.requires_grad:
        raise ValueError("segment loss must remain connected to the Outer graph")
    loss.backward()


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


def _storage_key(value: Tensor) -> tuple[str, int | None, int]:
    return (
        value.device.type,
        value.device.index,
        int(value.untyped_storage().data_ptr()),
    )
