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
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import (
    MetaTTTVariant,
    ProjectConfig,
)
from ttt_svcbench_qwen.data import assert_runtime_payload_safe
from ttt_svcbench_qwen.fast_ttt import (
    FastTTTForwardAudit,
    FastWeightsState,
    OptimizerRuntimeState,
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
from ttt_svcbench_qwen.stage_a_targets import StageATargetBuilder, TargetProvenance
from ttt_svcbench_qwen.state_encoder import TemporalEncoderOutput
from ttt_svcbench_qwen.state_retriever import RetrieverOutput
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageASupervisionBatch,
)

_SUPPORTED_TERMS = ("pred", "identity", "event")
_MAX_SUPPORT_CHUNKS = 8
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


@dataclass(frozen=True, slots=True)
class MetaModelRuntime:
    """One freshly reset hard/runtime trajectory, separate from fast/SGD state."""

    runtime_state: object
    bank_states: tuple[object, ...]

    def validate_for(self, owner: RuntimeOwner) -> None:
        if self.runtime_state is None:
            raise ValueError("Meta-TTT runtime resetter returned no runtime state")
        if len(self.bank_states) != len(owner.video_ids):
            raise ValueError("Meta-TTT reset Bank states must align to the owner batch")
        if any(state is None for state in self.bank_states):
            raise ValueError("Meta-TTT reset Bank states cannot contain None")


class EpisodeRuntimeResetter(Protocol):
    def __call__(self, owner: RuntimeOwner) -> MetaModelRuntime: ...


@dataclass(frozen=True, slots=True)
class MetaCausalChunk:
    """One model observation plus independently audited label-free runtime payload."""

    request: ObservationChunkRequest
    start_time: float
    end_time: float
    runtime_payload: Mapping[str, object]

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
        assert_runtime_payload_safe(self.runtime_payload, layer="Meta-TTT Support/Query")


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

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Meta-TTT episode seed must be a non-negative integer")
        support_count = len(self.support_chunks)
        if support_count < 1 or support_count > _MAX_SUPPORT_CHUNKS:
            raise ValueError("Meta-TTT episodes require between 1 and 8 Support chunks")
        if not self.query_points:
            raise ValueError("Meta-TTT episodes require at least one later Query point")
        chunks = (*self.support_chunks, *(query.chunk for query in self.query_points))
        if any(chunk.request.owner != self.owner for chunk in chunks):
            raise ValueError("all Meta-TTT requests must share the episode owner")
        batch_size = len(self.owner.video_ids)
        if any(query.answer.base_input_ids.shape[0] != batch_size for query in self.query_points):
            raise ValueError("all Meta-TTT Query rows must align to the owner batch")
        support_ends = tuple(chunk.end_time for chunk in self.support_chunks)
        if any(right <= left for left, right in pairwise(support_ends)):
            raise ValueError("Support chunk end times must advance strictly")
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
    state: StateLossInput


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
        if supervision.state is None:
            raise ValueError("Meta-TTT Query points require explicit State supervision")
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
            state=self.target_builder(
                output.observations,
                output.query,
                output.retrieval,
                supervision.state,
            ),
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
    state: StateLossOutput
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
    final_runtime: MetaModelRuntime
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
class MetaOuterStepAudit:
    optimizer_step_applied: bool
    skip_reason: str | None
    gradient_norm: float | None
    meta_fast_gradient_norms: tuple[float, float] | None
    meta_fast_delta_norms: tuple[float, float]
    transient_fast_in_optimizer: bool

    def __post_init__(self) -> None:
        if self.optimizer_step_applied != (self.skip_reason is None):
            raise ValueError("Meta-TTT outer step and skip reason disagree")
        if self.transient_fast_in_optimizer:
            raise ValueError("transient per-video W_t entered the Outer optimizer")
        values = (
            *self.meta_fast_delta_norms,
            *((self.gradient_norm,) if self.gradient_norm is not None else ()),
            *(self.meta_fast_gradient_norms or ()),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Meta-TTT outer gradient/delta norms must be finite and non-negative")
        if self.optimizer_step_applied:
            if self.meta_fast_gradient_norms is None or min(self.meta_fast_gradient_norms) <= 0.0:
                raise ValueError("an applied Meta-TTT step requires gradients on both W0 matrices")
            if min(self.meta_fast_delta_norms) <= 0.0:
                raise ValueError("an applied Meta-TTT step must change both W0 matrices")


@dataclass(frozen=True, slots=True)
class MetaTTTTrainingStepOutput:
    episode: MetaTTTEpisodeOutput
    global_step: int
    audit: MetaOuterStepAudit

    def __post_init__(self) -> None:
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("Meta-TTT global_step must be a non-negative integer")


class MetaTTTTrainer:
    """Consume the episode objective immediately and apply one audited Outer step."""

    def __init__(
        self,
        *,
        runner: MetaTTTEpisodeRunner,
        optimizer: torch.optim.Optimizer,
        outer_grad_clip_norm: float,
    ) -> None:
        if not isinstance(runner, MetaTTTEpisodeRunner):
            raise TypeError("MetaTTTTrainer requires MetaTTTEpisodeRunner")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("MetaTTTTrainer requires a torch optimizer")
        if not math.isfinite(outer_grad_clip_norm) or outer_grad_clip_norm <= 0.0:
            raise ValueError("Outer gradient clip norm must be positive and finite")
        self.runner = runner
        self.optimizer = optimizer
        self.outer_grad_clip_norm = outer_grad_clip_norm
        self.global_step = 0
        self._optimizer_parameters = _optimizer_parameters(optimizer)
        meta_fast = runner.fast_controller.collect_meta_fast_parameters()
        optimizer_ids = {id(parameter) for parameter in self._optimizer_parameters}
        if any(id(parameter) not in optimizer_ids for parameter in meta_fast):
            raise ValueError("Outer optimizer must own both meta-learned W0 matrices")

    def train_step(self, episode: MetaTTTEpisode) -> MetaTTTTrainingStepOutput:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.runner(episode)
        transient_ids = {
            id(value) for state in output.final_fast_states for value in state.fast_parameters
        }
        transient_in_optimizer = any(
            id(parameter) in transient_ids for parameter in self._optimizer_parameters
        )
        if transient_in_optimizer:
            raise ValueError("Outer optimizer captured transient per-video W_t")
        meta_fast = self.runner.fast_controller.collect_meta_fast_parameters()
        before = tuple(parameter.detach().clone() for parameter in meta_fast)
        output.total.backward()
        gradients = tuple(parameter.grad for parameter in self._optimizer_parameters)
        present = tuple(value for value in gradients if value is not None)
        finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
        skip_reason: str | None = None
        gradient_norm: float | None = None
        meta_norms: tuple[float, float] | None = None
        if finite:
            if any(parameter.grad is None for parameter in meta_fast):
                finite = False
                skip_reason = "missing_meta_fast_gradient"
            else:
                meta_norms = cast(
                    tuple[float, float],
                    tuple(
                        float(cast(Tensor, parameter.grad).detach().float().norm().item())
                        for parameter in meta_fast
                    ),
                )
                if min(meta_norms) <= 0.0:
                    finite = False
                    skip_reason = "zero_meta_fast_gradient"
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(
                self._optimizer_parameters,
                self.outer_grad_clip_norm,
                error_if_nonfinite=False,
            )
            gradient_norm = float(norm.detach().float().item())
            if math.isfinite(gradient_norm):
                self.optimizer.step()
                self.global_step += 1
            else:
                finite = False
                skip_reason = "nonfinite_clipped_gradient"
        if not finite:
            if skip_reason is None:
                skip_reason = "no_gradient" if not present else "nonfinite_gradient"
            self.optimizer.zero_grad(set_to_none=True)
        deltas = cast(
            tuple[float, float],
            tuple(
                float((parameter.detach() - start).float().norm().item())
                for parameter, start in zip(meta_fast, before, strict=True)
            ),
        )
        return MetaTTTTrainingStepOutput(
            episode=output,
            global_step=self.global_step,
            audit=MetaOuterStepAudit(
                optimizer_step_applied=finite,
                skip_reason=skip_reason,
                gradient_norm=gradient_norm,
                meta_fast_gradient_norms=meta_norms,
                meta_fast_delta_norms=deltas,
                transient_fast_in_optimizer=transient_in_optimizer,
            ),
        )


@dataclass(slots=True)
class _Trajectory:
    runtime: MetaModelRuntime
    fast_states: tuple[FastWeightsState, ...]
    optimizer_states: tuple[OptimizerRuntimeState, ...]


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
            adapted.runtime = MetaModelRuntime(
                adapted_observation.runtime_state, adapted_observation.bank_states
            )
            baseline.runtime = MetaModelRuntime(
                baseline_observation.runtime_state, baseline_observation.bank_states
            )
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
                    runtime=adapted.runtime.runtime_state,
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
            adapted.runtime = MetaModelRuntime(
                after_observation.runtime_state, after_observation.bank_states
            )
            baseline.runtime = MetaModelRuntime(
                before_observation.runtime_state, before_observation.bank_states
            )
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
                before_state = compute_state_loss(before_inputs.state)
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
            graph_bound=_MAX_SUPPORT_CHUNKS,
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
            if support_count not in self.config.stage_c.support_chunk_schedule:
                raise ValueError("Stage C Support count must be one of the audited 1/4/8 schedule")
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
        if not isinstance(runtime, MetaModelRuntime):
            raise TypeError("Meta-TTT runtime resetter must return MetaModelRuntime")
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
        return _Trajectory(runtime, fast_states, optimizer_states)

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
            runtime_state=trajectory.runtime.runtime_state,
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
        state = compute_state_loss(inputs.state)
        outer = compute_outer_loss(
            OuterLossInput(answer_after=answer, state_after=state, support_ttt=support)
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


@dataclass(frozen=True, slots=True)
class VariantIsolationAudit:
    a3_terms: tuple[str, ...]
    a4_terms: tuple[str, ...]
    a5_terms: tuple[str, ...]
    a4_minus_a3: tuple[str, ...]
    a5_minus_a4: tuple[str, ...]
    isolated: bool

    def __post_init__(self) -> None:
        if not self.isolated:
            raise ValueError("A3/A4/A5 objective increments are not isolated")


def audit_variant_isolation(config: ProjectConfig) -> VariantIsolationAudit:
    a3 = enabled_terms_for(config, MetaTTTVariant.A3)
    a4 = enabled_terms_for(config, MetaTTTVariant.A4)
    a5 = enabled_terms_for(config, MetaTTTVariant.A5)
    a4_delta = tuple(term for term in a4 if term not in a3)
    a5_delta = tuple(term for term in a5 if term not in a4)
    return VariantIsolationAudit(
        a3_terms=a3,
        a4_terms=a4,
        a5_terms=a5,
        a4_minus_a3=a4_delta,
        a5_minus_a4=a5_delta,
        isolated=a3 == ("pred",)
        and a4 == ("pred", "identity")
        and a5 == ("pred", "identity", "event")
        and a4_delta == ("identity",)
        and a5_delta == ("event",),
    )


class MetaGradientReferenceMode(StrEnum):
    """Diagnostic only; the episode runner always uses full second order."""

    FIRST_ORDER = "first_order_reference"
    FULL_SECOND_ORDER = "full_second_order_reference"


class TwoMatrixLoss(Protocol):
    def __call__(self, first: Tensor, second: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class MetaGradientReferenceResult:
    mode: MetaGradientReferenceMode
    support_loss: Tensor
    query_loss: Tensor
    next_parameters: tuple[Tensor, Tensor]
    meta_gradients: tuple[Tensor, Tensor]


def run_meta_gradient_reference(
    *,
    initial_parameters: tuple[Tensor, Tensor],
    support_loss: TwoMatrixLoss,
    query_loss: TwoMatrixLoss,
    learning_rate: float,
    mode: MetaGradientReferenceMode,
) -> MetaGradientReferenceResult:
    """Small-tensor FOMAML/full-MAML reference; never used by the training runner."""

    if not isinstance(mode, MetaGradientReferenceMode):
        raise TypeError("meta-gradient reference mode must be explicit")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("meta-gradient reference learning_rate must be positive and finite")
    parameters = initial_parameters
    if any(
        value.ndim == 0
        or not torch.is_floating_point(value)
        or not value.requires_grad
        or not value.is_leaf
        for value in parameters
    ):
        raise ValueError("meta-gradient reference inputs must be floating gradient leaves")
    inner = support_loss(*parameters)
    _validate_reference_scalar(inner, "support")
    full = mode is MetaGradientReferenceMode.FULL_SECOND_ORDER
    gradients_raw = torch.autograd.grad(inner, parameters, create_graph=full)
    gradients = cast(tuple[Tensor, Tensor], gradients_raw)
    if not full:
        gradients = (gradients[0].detach(), gradients[1].detach())
    next_parameters = (
        parameters[0] - learning_rate * gradients[0],
        parameters[1] - learning_rate * gradients[1],
    )
    outer = query_loss(*next_parameters)
    _validate_reference_scalar(outer, "query")
    meta_raw = torch.autograd.grad(outer, parameters)
    meta = cast(tuple[Tensor, Tensor], meta_raw)
    return MetaGradientReferenceResult(mode, inner, outer, next_parameters, meta)


@dataclass(frozen=True, slots=True)
class SyntheticAblationRecord:
    case_id: str
    task_name: str
    metric_name: str
    variant: MetaTTTVariant
    value: float
    failure_cases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not self.task_name or not self.metric_name:
            raise ValueError("synthetic ablation keys must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("synthetic ablation values must be finite")
        if any(not value for value in self.failure_cases):
            raise ValueError("synthetic failure-case identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class SyntheticAblationComparison:
    baseline: MetaTTTVariant
    candidate: MetaTTTVariant
    task_name: str
    metric_name: str
    sample_count: int
    mean_delta: float
    ci95_low: float
    ci95_high: float
    failure_cases: tuple[str, ...]
    scientific_claim_allowed: bool = False

    def __post_init__(self) -> None:
        if self.scientific_claim_allowed:
            raise ValueError("synthetic engineering ablations cannot support scientific claims")
        values = (self.mean_delta, self.ci95_low, self.ci95_high)
        if self.sample_count <= 0 or any(not math.isfinite(value) for value in values):
            raise ValueError("synthetic ablation comparison statistics are invalid")


def compare_synthetic_ablations(
    records: Sequence[SyntheticAblationRecord],
) -> tuple[SyntheticAblationComparison, ...]:
    """Paired A4-vs-A3 and A5-vs-A4 engineering deltas with normal 95% CIs."""

    values = tuple(records)
    if not values:
        raise ValueError("synthetic ablation comparison requires records")
    by_key = {
        (item.case_id, item.task_name, item.metric_name, item.variant): item for item in values
    }
    if len(by_key) != len(values):
        raise ValueError(
            "synthetic ablation records must have unique case/task/metric/variant keys"
        )
    comparisons: list[SyntheticAblationComparison] = []
    for baseline, candidate in (
        (MetaTTTVariant.A3, MetaTTTVariant.A4),
        (MetaTTTVariant.A4, MetaTTTVariant.A5),
    ):
        groups = sorted({(item.task_name, item.metric_name) for item in values})
        for task_name, metric_name in groups:
            paired: list[tuple[SyntheticAblationRecord, SyntheticAblationRecord]] = []
            case_ids = sorted(
                {
                    item.case_id
                    for item in values
                    if item.task_name == task_name and item.metric_name == metric_name
                }
            )
            for case_id in case_ids:
                left = by_key.get((case_id, task_name, metric_name, baseline))
                right = by_key.get((case_id, task_name, metric_name, candidate))
                if left is not None and right is not None:
                    paired.append((left, right))
            if not paired:
                continue
            deltas = tuple(right.value - left.value for left, right in paired)
            mean = math.fsum(deltas) / len(deltas)
            if len(deltas) == 1:
                half_width = 0.0
            else:
                variance = math.fsum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
                half_width = _CI_Z_95 * math.sqrt(variance / len(deltas))
            failures = tuple(
                sorted(
                    {
                        failure
                        for pair in paired
                        for record in pair
                        for failure in record.failure_cases
                    }
                )
            )
            comparisons.append(
                SyntheticAblationComparison(
                    baseline=baseline,
                    candidate=candidate,
                    task_name=task_name,
                    metric_name=metric_name,
                    sample_count=len(deltas),
                    mean_delta=mean,
                    ci95_low=mean - half_width,
                    ci95_high=mean + half_width,
                    failure_cases=failures,
                )
            )
    if not comparisons:
        raise ValueError("no paired A3/A4 or A4/A5 synthetic ablation records were found")
    return tuple(comparisons)


def render_synthetic_ablation_report(
    comparisons: Sequence[SyntheticAblationComparison],
) -> str:
    """Render a compact audit artifact with an unavoidable non-scientific disclaimer."""

    values = tuple(comparisons)
    if not values:
        raise ValueError("cannot render an empty synthetic ablation report")
    lines = [
        "# P17 synthetic engineering ablation report",
        "",
        "> Synthetic/tiny engineering evidence only; no scientific gain or SVCBench accuracy "
        "claim is allowed.",
        "",
        "| comparison | task | metric | n | mean delta | 95% CI | failures |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for item in values:
        failures = ", ".join(item.failure_cases) if item.failure_cases else "none"
        lines.append(
            f"| {item.candidate.value} vs {item.baseline.value} | {item.task_name} | "
            f"{item.metric_name} | {item.sample_count} | {item.mean_delta:.6f} | "
            f"[{item.ci95_low:.6f}, {item.ci95_high:.6f}] | {failures} |"
        )
    return "\n".join(lines) + "\n"


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


def _query_metrics(answer: AnswerLossOutput, state: StateLossOutput) -> QueryMetricSnapshot:
    return QueryMetricSnapshot(
        metrics=(
            ("loss/answer", _term_float(answer.loss)),
            ("loss/state", float(state.total.detach().item())),
            ("answer/token_accuracy", _term_float(answer.teacher_forced_token_accuracy)),
            ("answer/number_token_accuracy", _term_float(answer.number_token_accuracy)),
            ("answer/exact_match", _term_float(answer.answer_exact_match)),
            ("reader/exact_count_accuracy", _term_float(answer.reader_exact_count_accuracy)),
            ("state/o1", _term_float(state.o1)),
            ("state/o2", _term_float(state.o2)),
            ("state/e1", _term_float(state.e1)),
            ("state/e2", _term_float(state.e2)),
        )
    )


def _term_float(term: object) -> float | None:
    value = getattr(term, "value", None)
    valid = getattr(term, "row_valid_mask", None)
    if not isinstance(value, Tensor) or not isinstance(valid, Tensor):
        raise TypeError("metric source must expose a typed LossTerm value/mask")
    return float(value.detach().item()) if bool(valid.any().item()) else None


def _assert_trajectory_isolation(adapted: _Trajectory, baseline: _Trajectory) -> None:
    if adapted.runtime.runtime_state is baseline.runtime.runtime_state:
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


def _unique_parameters(parameters: Sequence[nn.Parameter]) -> tuple[nn.Parameter, ...]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return tuple(result)


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for value in group["params"]:
            if not isinstance(value, nn.Parameter):
                raise TypeError("Outer optimizer may contain only nn.Parameter values")
            if id(value) in seen:
                raise ValueError("Outer optimizer parameter groups cannot contain aliases")
            values.append(value)
            seen.add(id(value))
    if not values:
        raise ValueError("Outer optimizer cannot be empty")
    return tuple(values)


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


def _validate_reference_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0 or not torch.is_floating_point(value) or not value.requires_grad:
        raise ValueError(f"meta-gradient {name} loss must be a differentiable scalar")
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f"meta-gradient {name} loss must be finite")
