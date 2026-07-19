"""P15 hard-state rollout with a differentiable semantic write branch.

Inputs: typed soft observation outputs, encoder states, query routing, and a reset Stage A runtime.
Outputs: detached hard Bank/Identity/FSM state plus gradient-carrying semantic projections.
Forbidden: Inner SGD, transient fast weights, labels, future chunks, or checkpointed runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.identity_bank import (
    IdentityBank,
    IdentityBankRuntimeState,
    IdentityObservationDecision,
)
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    ObservationChunkRequest,
    RuntimeOwner,
)
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E2RuntimeState,
    ObservationOutputs,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_EVENT_KIND,
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    QueryEncoderOutput,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E2EventKind,
    HeadType,
    StateBankRuntimeState,
    StructuredStateBank,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
    TemporalEncoderOutput,
)


@dataclass(frozen=True, slots=True)
class StageABatchRuntime:
    """External per-owner state; it is deliberately not an ``nn.Module``."""

    owner: RuntimeOwner
    next_chunk_index: int
    slot_states: tuple[SpatialSlotRuntimeState | None, ...]
    temporal_cache: TemporalCache | None
    e1_states: tuple[E1RuntimeState | None, ...]
    e2_states: tuple[E2RuntimeState | None, ...]
    state_bank_states: tuple[StateBankRuntimeState, ...]
    identity_bank_states: tuple[IdentityBankRuntimeState, ...]
    inner_sgd_attempted: int = 0
    inner_sgd_updated: int = 0
    inner_sgd_skipped: int = 0

    def __post_init__(self) -> None:
        batch_size = len(self.owner.video_ids)
        aligned = (
            self.slot_states,
            self.e1_states,
            self.e2_states,
            self.state_bank_states,
            self.identity_bank_states,
        )
        if any(len(values) != batch_size for values in aligned):
            raise ValueError("Stage A runtime fields must align to the owner batch")
        if type(self.next_chunk_index) is not int or self.next_chunk_index < 0:
            raise ValueError("Stage A next_chunk_index must be non-negative")
        counters = (
            self.inner_sgd_attempted,
            self.inner_sgd_updated,
            self.inner_sgd_skipped,
        )
        if any(type(value) is not int or value != 0 for value in counters):
            raise ValueError("Stage A forbids every Inner-SGD attempt, update, and skip")
        for row, (video_id, trajectory_id) in enumerate(
            zip(self.owner.video_ids, self.owner.trajectory_ids, strict=True)
        ):
            bank = self.state_bank_states[row]
            identity = self.identity_bank_states[row]
            if (bank.video_id, bank.trajectory_id) != (video_id, trajectory_id):
                raise ValueError("Stage A State Bank ownership mismatch")
            if (identity.video_id, identity.trajectory_id) != (video_id, trajectory_id):
                raise ValueError("Stage A Identity Bank ownership mismatch")
            slot = self.slot_states[row]
            if slot is not None and slot.video_id != video_id:
                raise ValueError("Stage A slot-state ownership mismatch")
            for name, state in (("E1", self.e1_states[row]), ("E2", self.e2_states[row])):
                if state is not None and (state.video_id, state.trajectory_id) != (
                    video_id,
                    trajectory_id,
                ):
                    raise ValueError(f"Stage A {name} ownership mismatch")
        if self.temporal_cache is not None and (
            self.temporal_cache.video_ids != self.owner.video_ids
            or self.temporal_cache.trajectory_ids != self.owner.trajectory_ids
        ):
            raise ValueError("Stage A temporal-cache ownership mismatch")


@dataclass(frozen=True, slots=True)
class StageASoftWriteOutput:
    """Semantic Projector outputs retained only for outer-loss gradients."""

    o1_semantics: Tensor
    o1_present_mask: Tensor
    o2_semantics: Tensor
    o2_present_mask: Tensor
    e1_semantics: Tensor
    e1_present_mask: Tensor
    e2_semantics: Tensor
    e2_present_mask: Tensor
    source_policy: tuple[tuple[str, str], ...] = (
        ("o1", "valid_slot_mean_v1"),
        ("o2", "per_valid_slot_v1"),
        ("e1", "per_valid_tubelet_v1"),
        ("e2", "per_valid_tubelet_v1"),
    )

    def __post_init__(self) -> None:
        batch_size = self.o1_semantics.shape[0] if self.o1_semantics.ndim == 2 else -1
        if self.o1_semantics.shape != (batch_size, 512):
            raise ValueError("O1 soft semantics must be [B, 512]")
        if self.o1_present_mask.shape != (batch_size,):
            raise ValueError("O1 semantic mask must be [B]")
        pairs = (
            (self.o2_semantics, self.o2_present_mask, "O2"),
            (self.e1_semantics, self.e1_present_mask, "E1"),
            (self.e2_semantics, self.e2_present_mask, "E2"),
        )
        for values, mask, name in pairs:
            if values.ndim != 3 or values.shape[0] != batch_size or values.shape[-1] != 512:
                raise ValueError(f"{name} soft semantics must be [B, N, 512]")
            if mask.shape != values.shape[:2]:
                raise ValueError(f"{name} semantic mask must be [B, N]")
        tensors = (
            self.o1_semantics,
            self.o1_present_mask,
            self.o2_semantics,
            self.o2_present_mask,
            self.e1_semantics,
            self.e1_present_mask,
            self.e2_semantics,
            self.e2_present_mask,
        )
        reference = self.o1_semantics
        if any(tensor.device != reference.device for tensor in tensors):
            raise ValueError("Stage A soft semantics must share one device")
        for mask in (
            self.o1_present_mask,
            self.o2_present_mask,
            self.e1_present_mask,
            self.e2_present_mask,
        ):
            if mask.dtype is not torch.bool:
                raise TypeError("Stage A semantic masks must be bool")
        if not all(torch.is_floating_point(tensor) for tensor in tensors[::2]):
            raise TypeError("Stage A semantic values must be floating")
        if reference.device.type != "meta":
            for values, mask in (
                (self.o1_semantics, self.o1_present_mask),
                (self.o2_semantics, self.o2_present_mask),
                (self.e1_semantics, self.e1_present_mask),
                (self.e2_semantics, self.e2_present_mask),
            ):
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Stage A soft semantics must be finite")
                if bool(torch.any(values[~mask] != 0.0)):
                    raise ValueError("invalid Stage A semantic sources must be zero")
                valid = values[mask]
                if valid.numel():
                    norms = torch.linalg.vector_norm(valid.float(), dim=-1)
                    norm_tolerance = max(
                        5.0e-4,
                        2.0 * float(torch.finfo(valid.dtype).eps),
                    )
                    if not torch.allclose(
                        norms,
                        torch.ones_like(norms),
                        atol=norm_tolerance,
                        rtol=0.0,
                    ):
                        raise ValueError("valid Stage A semantics must have unit norm")


@dataclass(frozen=True, slots=True)
class StageAWriteAudit:
    chunk_index: int
    head_types: tuple[HeadType | None, ...]
    bank_versions_before: tuple[int, ...]
    bank_versions_after: tuple[int, ...]
    record_counts_after: tuple[int, ...]
    identity_counts_after: tuple[int, ...]
    identity_decisions: tuple[tuple[IdentityObservationDecision, ...], ...]
    skipped_rows: tuple[int, ...]
    inner_sgd_attempted: int = 0
    inner_sgd_updated: int = 0
    inner_sgd_skipped: int = 0

    def __post_init__(self) -> None:
        batch_size = len(self.head_types)
        fields = (
            self.bank_versions_before,
            self.bank_versions_after,
            self.record_counts_after,
            self.identity_counts_after,
            self.identity_decisions,
        )
        if batch_size <= 0 or any(len(values) != batch_size for values in fields):
            raise ValueError("Stage A write audit must align to one non-empty batch")
        if any(
            value != 0
            for value in (
                self.inner_sgd_attempted,
                self.inner_sgd_updated,
                self.inner_sgd_skipped,
            )
        ):
            raise ValueError("Stage A write audit cannot report Inner SGD activity")


class StageABankWriter:
    """Commit hard state while preserving a separate differentiable semantic branch."""

    def __init__(self, state_bank: StructuredStateBank, identity_bank: IdentityBank) -> None:
        if not isinstance(state_bank, StructuredStateBank) or not isinstance(
            identity_bank, IdentityBank
        ):
            raise TypeError("StageABankWriter requires typed State and Identity Banks")
        self.state_bank = state_bank
        self.identity_bank = identity_bank

    def reset(self, owner: RuntimeOwner) -> StageABatchRuntime:
        banks = tuple(
            self.state_bank.reset(video_id, trajectory_id)
            for video_id, trajectory_id in zip(
                owner.video_ids,
                owner.trajectory_ids,
                strict=True,
            )
        )
        identities = tuple(
            self.identity_bank.reset(video_id, trajectory_id, hot_cache_enabled=False)
            for video_id, trajectory_id in zip(
                owner.video_ids,
                owner.trajectory_ids,
                strict=True,
            )
        )
        empty = (None,) * len(owner.video_ids)
        return StageABatchRuntime(
            owner=owner,
            next_chunk_index=0,
            slot_states=empty,
            temporal_cache=None,
            e1_states=empty,
            e2_states=empty,
            state_bank_states=banks,
            identity_bank_states=identities,
        )

    def __call__(
        self,
        observations: object,
        spatial: object,
        temporal: object,
        query: object,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput:
        if not isinstance(observations, ObservationOutputs):
            raise TypeError("Stage A writer requires ObservationOutputs")
        if not isinstance(spatial, SpatialEncoderOutput) or not isinstance(
            temporal, TemporalEncoderOutput
        ):
            raise TypeError("Stage A writer requires typed spatial/temporal outputs")
        if not isinstance(query, QueryEncoderOutput):
            raise TypeError("Stage A writer requires QueryEncoderOutput")
        runtime = request.runtime_state
        if not isinstance(runtime, StageABatchRuntime):
            raise TypeError("Stage A writer requires StageABatchRuntime")
        if runtime.owner != request.owner:
            raise ValueError("Stage A writer request owner does not match runtime")
        if len(request.bank_states) != len(runtime.state_bank_states) or any(
            provided is not authoritative
            for provided, authoritative in zip(
                request.bank_states,
                runtime.state_bank_states,
                strict=True,
            )
        ):
            raise ValueError("Stage A request must carry the authoritative Bank states")
        if spatial.next_states is None:
            raise ValueError("Stage A spatial output must return detached next_states")

        soft = self._project_soft(spatial, temporal, observations)
        next_banks = list(runtime.state_bank_states)
        next_identities = list(runtime.identity_bank_states)
        identity_decisions: list[tuple[IdentityObservationDecision, ...]] = [
            () for _ in runtime.identity_bank_states
        ]
        skipped: list[int] = []
        for row, operator in enumerate(query.hard_operators):
            head = OPERATOR_TO_HEAD_TYPE[operator]
            state = next_banks[row]
            if head is HeadType.O1:
                mask = observations.o1.valid_mask[row]
                if not bool(mask.any().item()):
                    skipped.append(row)
                    continue
                timestamp = float(observations.o1.timestamps[row, mask][0].item())
                position_id = int(observations.o1.position_ids[row, mask][0].item())
                has_o1 = any(record.head_type is HeadType.O1 for record in state.records)
                next_banks[row] = self.state_bank.update_o1(
                    state,
                    observations.o1,
                    soft.o1_semantics[row],
                    observation_timestamp=timestamp,
                    observation_position_id=position_id,
                    row=row,
                    set_baseline=operator is Operator.O1_DELTA and not has_o1,
                    slot_overflow_count=int(spatial.active_slot_overflow_count[row].item()),
                )
            elif head is HeadType.O2:
                result = self.identity_bank.update_row(
                    next_identities[row],
                    self.state_bank,
                    state,
                    observations.o2,
                    soft.o2_semantics[row],
                    row=row,
                    chunk_index=runtime.next_chunk_index,
                )
                next_identities[row] = result.identity_state
                next_banks[row] = result.state_bank_state
                identity_decisions[row] = result.decisions
            elif head is HeadType.E1:
                event_kind = OPERATOR_TO_EVENT_KIND[operator]
                if not isinstance(event_kind, E1EventKind):
                    raise RuntimeError("E1 operator lost its event-kind mapping")
                next_banks[row] = self.state_bank.update_e1(
                    state,
                    observations.e1,
                    soft.e1_semantics,
                    event_kind=event_kind,
                    row=row,
                )
            elif head is HeadType.E2:
                event_kind = OPERATOR_TO_EVENT_KIND[operator]
                if not isinstance(event_kind, E2EventKind):
                    raise RuntimeError("E2 operator lost its event-kind mapping")
                next_banks[row] = self.state_bank.update_e2(
                    state,
                    observations.e2,
                    soft.e2_semantics,
                    event_kind=event_kind,
                    row=row,
                )
            else:
                skipped.append(row)

        next_runtime = StageABatchRuntime(
            owner=request.owner,
            next_chunk_index=runtime.next_chunk_index + 1,
            slot_states=cast(tuple[SpatialSlotRuntimeState | None, ...], spatial.next_states),
            temporal_cache=temporal.cache,
            e1_states=cast(tuple[E1RuntimeState | None, ...], observations.e1.next_states),
            e2_states=cast(tuple[E2RuntimeState | None, ...], observations.e2.next_states),
            state_bank_states=tuple(next_banks),
            identity_bank_states=tuple(next_identities),
        )
        audit = StageAWriteAudit(
            chunk_index=runtime.next_chunk_index,
            head_types=tuple(OPERATOR_TO_HEAD_TYPE[value] for value in query.hard_operators),
            bank_versions_before=tuple(state.version for state in runtime.state_bank_states),
            bank_versions_after=tuple(state.version for state in next_banks),
            record_counts_after=tuple(len(state.records) for state in next_banks),
            identity_counts_after=tuple(state.unique_count for state in next_identities),
            identity_decisions=tuple(identity_decisions),
            skipped_rows=tuple(skipped),
        )
        return BankWriteOutput(
            runtime_state=next_runtime,
            bank_states=tuple(next_banks),
            audit=audit,
            soft_write=soft,
        )

    def _project_soft(
        self,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        observations: ObservationOutputs,
    ) -> StageASoftWriteOutput:
        slot_mask = spatial.slot_valid_mask
        slot_count = slot_mask.sum(dim=1, keepdim=True).clamp_min(1)
        o1_source = (spatial.slots * slot_mask.unsqueeze(-1).to(dtype=spatial.slots.dtype)).sum(
            dim=1
        ) / slot_count.to(dtype=spatial.slots.dtype)
        o1_present = slot_mask.any(dim=1)
        o1 = self.state_bank.project(o1_source, HeadType.O1)
        o1 = torch.where(o1_present.unsqueeze(-1), o1, 0.0)

        o2 = self.state_bank.project(spatial.slots, HeadType.O2)
        o2 = torch.where(slot_mask.unsqueeze(-1), o2, 0.0)
        time_mask = temporal.valid_mask
        e1 = self.state_bank.project(temporal.hidden, HeadType.E1)
        e2 = self.state_bank.project(temporal.hidden, HeadType.E2)
        e1 = torch.where(time_mask.unsqueeze(-1), e1, 0.0)
        e2 = torch.where(time_mask.unsqueeze(-1), e2, 0.0)
        if not torch.equal(slot_mask, observations.o1.valid_mask) or not torch.equal(
            slot_mask, observations.o2.valid_mask
        ):
            raise ValueError("Stage A spatial sources must align with O1/O2 masks")
        if not torch.equal(time_mask, observations.e1.valid_mask) or not torch.equal(
            time_mask, observations.e2.valid_mask
        ):
            raise ValueError("Stage A temporal sources must align with E1/E2 masks")
        return StageASoftWriteOutput(
            o1_semantics=o1,
            o1_present_mask=o1_present,
            o2_semantics=o2,
            o2_present_mask=slot_mask.clone(),
            e1_semantics=e1,
            e1_present_mask=time_mask.clone(),
            e2_semantics=e2,
            e2_present_mask=time_mask.clone(),
        )
