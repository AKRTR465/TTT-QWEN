"""P15 hard-state rollout with a differentiable semantic write branch.

Inputs: typed soft observation outputs, encoder states, query routing, and a reset Stage A runtime.
Outputs: detached hard Bank/Identity/FSM state plus gradient-carrying semantic projections.
Forbidden: memory writes, transient memory state, labels, future chunks, or checkpointed
runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.identity_bank import IdentityBank
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    BatchRuntimeState,
    ObservationChunkRequest,
    RuntimeOwner,
    TrajectoryRuntimeState,
)
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E2RuntimeState,
    ObservationOutputs,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_EVENT_KIND,
    OPERATOR_TO_HEAD_TYPE,
    OPERATORS,
    Operator,
    QueryEncoderOutput,
)
from ttt_svcbench_qwen.runtime_metrics import trace_cuda_phase
from ttt_svcbench_qwen.state_bank import (
    RETRIEVAL_HEAD_ORDER,
    E1EventKind,
    E2EventKind,
    HeadType,
    RetrievalHistoryAppendBatch,
    StructuredStateBank,
    TensorizedRetrievalHistory,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalEncoderOutput,
)


@dataclass(frozen=True, slots=True)
class StageASoftWriteOutput:
    """Projected write semantics plus raw sources for detached retrieval history."""

    o1_semantics: Tensor
    o1_present_mask: Tensor
    o2_semantics: Tensor
    o2_present_mask: Tensor
    e1_semantics: Tensor
    e1_present_mask: Tensor
    e2_semantics: Tensor
    e2_present_mask: Tensor
    o1_sources: Tensor
    o2_sources: Tensor
    e1_sources: Tensor
    e2_sources: Tensor


class StageABankWriter:
    """Commit hard state while preserving a separate differentiable semantic branch."""

    def __init__(self, state_bank: StructuredStateBank, identity_bank: IdentityBank) -> None:
        self.state_bank = state_bank
        self.identity_bank = identity_bank

    def reset(self, owner: RuntimeOwner) -> BatchRuntimeState:
        banks = tuple(
            self.state_bank.reset(video_id, trajectory_id)
            for video_id, trajectory_id in zip(
                owner.video_ids,
                owner.trajectory_ids,
                strict=True,
            )
        )
        identities = tuple(
            self.identity_bank.reset(video_id, trajectory_id)
            for video_id, trajectory_id in zip(
                owner.video_ids,
                owner.trajectory_ids,
                strict=True,
            )
        )
        return BatchRuntimeState(
            tuple(
                TrajectoryRuntimeState(
                    owner=RuntimeOwner((video_id,), (trajectory_id,)),
                    next_chunk_index=0,
                    slot_state=None,
                    temporal_cache=None,
                    e1_state=None,
                    e2_state=None,
                    state_bank=bank,
                    identity_bank=identity,
                    retrieval_history=TensorizedRetrievalHistory(
                        video_id,
                        trajectory_id,
                        capacity_per_head=self.state_bank.config.retrieval_history_capacity_per_head,
                        source_dim=self.state_bank.config.retrieval_history_source_dim,
                        dtype=next(self.state_bank.semantic_projector.parameters()).dtype,
                        device=next(self.state_bank.semantic_projector.parameters()).device,
                    ),
                )
                for video_id, trajectory_id, bank, identity in zip(
                    owner.video_ids,
                    owner.trajectory_ids,
                    banks,
                    identities,
                    strict=True,
                )
            )
        )

    def __call__(
        self,
        observations: ObservationOutputs,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput:
        runtime = request.runtime_state

        soft = self._project_soft(spatial, temporal, observations)
        next_banks = list(runtime.state_bank_states)
        next_identities = list(runtime.identity_bank_states)
        for row, operator in enumerate(query.hard_operators):
            head = OPERATOR_TO_HEAD_TYPE[operator]
            if request.retrieval_history_write_enabled:
                with trace_cuda_phase("retrieval_history_write"):
                    history = cast(
                        TensorizedRetrievalHistory, runtime.rows[row].retrieval_history
                    )
                    self._append_all_head_retrieval_history_tensorized(
                        history, observations, soft, row=row
                    )
            state = next_banks[row]
            if head is HeadType.O1:
                mask = observations.o1.valid_mask[row]
                if not bool(mask.any().item()):
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
            elif head is HeadType.E1:
                next_banks[row] = self.state_bank.update_e1(
                    state,
                    observations.e1,
                    soft.e1_semantics,
                    event_kind=cast(E1EventKind, OPERATOR_TO_EVENT_KIND[operator]),
                    row=row,
                )
            elif head is HeadType.E2:
                next_banks[row] = self.state_bank.update_e2(
                    state,
                    observations.e2,
                    soft.e2_semantics,
                    event_kind=cast(E2EventKind, OPERATOR_TO_EVENT_KIND[operator]),
                    row=row,
                )

        slot_states = cast(tuple[SpatialSlotRuntimeState | None, ...], spatial.next_states)
        e1_states = cast(tuple[E1RuntimeState | None, ...], observations.e1.next_states)
        e2_states = cast(tuple[E2RuntimeState | None, ...], observations.e2.next_states)
        next_runtime = BatchRuntimeState(
            tuple(
                replace(
                    previous,
                    next_chunk_index=runtime.next_chunk_index + 1,
                    slot_state=slot_state,
                    temporal_cache=temporal.cache,
                    e1_state=e1_state,
                    e2_state=e2_state,
                    state_bank=bank,
                    identity_bank=identity,
                )
                for previous, slot_state, e1_state, e2_state, bank, identity in zip(
                    runtime.rows,
                    slot_states,
                    e1_states,
                    e2_states,
                    next_banks,
                    next_identities,
                    strict=True,
                )
            )
        )
        return BankWriteOutput(
            runtime_state=next_runtime,
            bank_states=tuple(next_banks),
            audit=None,
            soft_write=soft,
        )

    def _append_all_head_retrieval_history_tensorized(
        self,
        history: TensorizedRetrievalHistory,
        observations: ObservationOutputs,
        soft: StageASoftWriteOutput,
        *,
        row: int,
    ) -> None:
        """Collect O1/all-O2/E1/E2 once and issue a single vectorized ring write."""

        device = soft.o1_sources.device
        o1_present = soft.o1_present_mask[row : row + 1]
        o2_present = soft.o2_present_mask[row]
        e1_present = soft.e1_present_mask[row].any().reshape(1)
        e2_present = soft.e2_present_mask[row].any().reshape(1)
        present = torch.cat((o1_present, o2_present, e1_present, e2_present))
        sources = torch.cat(
            (
                soft.o1_sources[row : row + 1],
                soft.o2_sources[row],
                soft.e1_sources[row : row + 1],
                soft.e2_sources[row : row + 1],
            ),
            dim=0,
        )
        o2_width = soft.o2_sources.shape[1]
        head_codes = torch.cat(
            (
                torch.full(
                    (1,), RETRIEVAL_HEAD_ORDER.index(HeadType.O1), dtype=torch.int64, device=device
                ),
                torch.full(
                    (o2_width,),
                    RETRIEVAL_HEAD_ORDER.index(HeadType.O2),
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (1,), RETRIEVAL_HEAD_ORDER.index(HeadType.E1), dtype=torch.int64, device=device
                ),
                torch.full(
                    (1,), RETRIEVAL_HEAD_ORDER.index(HeadType.E2), dtype=torch.int64, device=device
                ),
            )
        )
        operator_codes = torch.cat(
            (
                torch.full(
                    (1,), OPERATORS.index(Operator.O1_SNAP), dtype=torch.int64, device=device
                ),
                torch.full(
                    (o2_width,),
                    OPERATORS.index(Operator.O2_UNIQUE),
                    dtype=torch.int64,
                    device=device,
                ),
                torch.full(
                    (1,), OPERATORS.index(Operator.E1_ACTION), dtype=torch.int64, device=device
                ),
                torch.full(
                    (1,), OPERATORS.index(Operator.E2_EPISODE), dtype=torch.int64, device=device
                ),
            )
        )
        timestamps = torch.cat(
            (
                torch.full((1,), -1.0, dtype=torch.float64, device=device),
                observations.o2.timestamps[row].to(dtype=torch.float64),
                torch.full((2,), -1.0, dtype=torch.float64, device=device),
            )
        )
        time_ranges = torch.full((o2_width + 3, 2), -1.0, dtype=torch.float64, device=device)
        time_ranges[0] = _time_range_tensor(
            observations.o1.timestamps[row], observations.o1.valid_mask[row]
        )
        time_ranges[o2_width + 1] = _time_range_tensor(
            observations.e1.timestamps[row], observations.e1.valid_mask[row]
        )
        time_ranges[o2_width + 2] = _time_range_tensor(
            observations.e2.timestamps[row], observations.e2.valid_mask[row]
        )
        indices = torch.nonzero(present, as_tuple=False).flatten()
        count = indices.numel()
        valid = torch.ones((count,), dtype=torch.bool, device=device)
        history.append_many(
            RetrievalHistoryAppendBatch(
                sources=sources.index_select(0, indices),
                head_codes=head_codes.index_select(0, indices),
                operator_codes=operator_codes.index_select(0, indices),
                timestamps=timestamps.index_select(0, indices),
                time_ranges=time_ranges.index_select(0, indices),
                valid_mask=valid,
                eligible_mask=valid.clone(),
            )
        )

    def _project_soft(
        self,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        observations: ObservationOutputs,
    ) -> StageASoftWriteOutput:
        slot_mask = observations.o1.valid_mask
        slot_count = slot_mask.sum(dim=1, keepdim=True).clamp_min(1)
        o1_source = (spatial.slots * slot_mask.unsqueeze(-1).to(dtype=spatial.slots.dtype)).sum(
            dim=1
        ) / slot_count.to(dtype=spatial.slots.dtype)
        o1_present = slot_mask.any(dim=1)
        o1_source = torch.where(o1_present.unsqueeze(-1), o1_source, 0.0)
        o2_source = torch.where(slot_mask.unsqueeze(-1), spatial.slots, 0.0)
        o1 = self.state_bank.project(o1_source, HeadType.O1)
        o1 = torch.where(o1_present.unsqueeze(-1), o1, 0.0)

        o2 = self.state_bank.project(spatial.slots, HeadType.O2)
        o2 = torch.where(slot_mask.unsqueeze(-1), o2, 0.0)
        time_mask = temporal.valid_mask
        time_count = time_mask.sum(dim=1, keepdim=True).clamp_min(1)
        time_source = (
            temporal.hidden * time_mask.unsqueeze(-1).to(dtype=temporal.hidden.dtype)
        ).sum(dim=1) / time_count.to(dtype=temporal.hidden.dtype)
        time_present = time_mask.any(dim=1)
        time_source = torch.where(time_present.unsqueeze(-1), time_source, 0.0)
        e1 = self.state_bank.project(temporal.hidden, HeadType.E1)
        e2 = self.state_bank.project(temporal.hidden, HeadType.E2)
        e1 = torch.where(time_mask.unsqueeze(-1), e1, 0.0)
        e2 = torch.where(time_mask.unsqueeze(-1), e2, 0.0)
        return StageASoftWriteOutput(
            o1_semantics=o1,
            o1_present_mask=o1_present,
            o2_semantics=o2,
            o2_present_mask=slot_mask.clone(),
            e1_semantics=e1,
            e1_present_mask=time_mask.clone(),
            e2_semantics=e2,
            e2_present_mask=time_mask.clone(),
            o1_sources=o1_source.detach(),
            o2_sources=o2_source.detach(),
            e1_sources=time_source.detach(),
            e2_sources=time_source.detach(),
        )


def _time_range_tensor(timestamps: Tensor, mask: Tensor) -> Tensor:
    selected = timestamps.masked_fill(~mask, float("inf"))
    start = selected.min().to(dtype=torch.float64)
    end = timestamps.masked_fill(~mask, float("-inf")).max().to(dtype=torch.float64)
    return torch.stack((start, end))
