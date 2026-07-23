from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from tests.support import parameter_count
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
    StreamReplayAudit,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E1Payload,
    E2EventKind,
    E2Payload,
    E2Phase,
    HeadType,
    O1Payload,
    O1SlotState,
    SemanticProjector,
    StateRecord,
    StateRecordKind,
    StructuredStateBank,
    build_state_bank,
    clone_state_record,
)

EXACT_PROJECTOR_PARAMETERS = 1_316_864
HIDDEN_DIM = 768
SEMANTIC_DIM = 512


@pytest.fixture(scope="module")
def bank() -> StructuredStateBank:
    torch.manual_seed(20260717)
    module = build_state_bank(load_config())
    module.eval()
    return module


def _unit_semantic(index: int = 0, *, requires_grad: bool = False) -> Tensor:
    value = torch.zeros(SEMANTIC_DIM)
    value[index] = 1.0
    return value.requires_grad_(requires_grad)


def _empty_e1_state() -> E1RuntimeState:
    return E1RuntimeState(
        video_id="observation-video",
        trajectory_id="observation-trajectory",
        query_signature=torch.zeros(SEMANTIC_DIM),
        projected_history=torch.zeros(0, 512),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )


def _empty_e2_state() -> E2RuntimeState:
    return E2RuntimeState(
        video_id="observation-video",
        trajectory_id="observation-trajectory",
        query_signature=torch.zeros(SEMANTIC_DIM),
        hidden=torch.zeros(2, HIDDEN_DIM),
        checkpoint_hidden=torch.zeros(0, 2, HIDDEN_DIM),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )


def _o1_output(
    probabilities: Tensor,
    *,
    timestamp: float,
    position_id: int,
    valid_mask: Tensor | None = None,
) -> O1SoftOutput:
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    batch_size, slot_count = probabilities.shape[:2]
    mask = (
        torch.ones(batch_size, slot_count, dtype=torch.bool) if valid_mask is None else valid_mask
    )
    timestamps = torch.where(
        mask,
        torch.full(mask.shape, timestamp, dtype=torch.float64),
        torch.full(mask.shape, -1.0, dtype=torch.float64),
    )
    positions = torch.where(
        mask,
        torch.full(mask.shape, position_id, dtype=torch.int64),
        torch.full(mask.shape, -1, dtype=torch.int64),
    )
    soft_count = (probabilities[..., 0] * probabilities[..., 1] * probabilities[..., 2] * mask).sum(
        dim=1
    )
    return O1SoftOutput(
        logits=torch.zeros_like(probabilities),
        probabilities=probabilities,
        soft_count=soft_count,
        valid_mask=mask,
        timestamps=timestamps,
        position_ids=positions,
    )


def _e1_output(
    probabilities: Tensor,
    timestamps: Tensor,
    position_ids: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> E1SoftOutput:
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if valid_mask is not None and valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    mask = (
        torch.ones(probabilities.shape[:2], dtype=torch.bool) if valid_mask is None else valid_mask
    )
    probabilities = probabilities.masked_fill(~mask.unsqueeze(-1), 0.0)
    timestamps = torch.where(mask, timestamps, torch.full_like(timestamps, -1.0))
    position_ids = torch.where(mask, position_ids, torch.full_like(position_ids, -1))
    return E1SoftOutput(
        logits=torch.zeros_like(probabilities),
        probabilities=probabilities,
        valid_mask=mask,
        timestamps=timestamps.to(dtype=torch.float64),
        position_ids=position_ids.to(dtype=torch.int64),
        next_states=tuple(_empty_e1_state() for _ in range(probabilities.shape[0])),
        audit=StreamReplayAudit(
            "e1",
            tuple(probabilities.shape[1] for _ in range(probabilities.shape[0])),
            tuple(0 for _ in range(probabilities.shape[0])),
            tuple(0 for _ in range(probabilities.shape[0])),
        ),
    )


def _e2_output(
    event_probabilities: Tensor,
    phase_indices: Tensor,
    timestamps: Tensor,
    position_ids: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> E2SoftOutput:
    if event_probabilities.ndim == 2:
        event_probabilities = event_probabilities.unsqueeze(0)
    if phase_indices.ndim == 1:
        phase_indices = phase_indices.unsqueeze(0)
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if valid_mask is not None and valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    mask = (
        torch.ones(event_probabilities.shape[:2], dtype=torch.bool)
        if valid_mask is None
        else valid_mask
    )
    event_probabilities = event_probabilities.masked_fill(~mask.unsqueeze(-1), 0.0)
    timestamps = torch.where(mask, timestamps, torch.full_like(timestamps, -1.0))
    position_ids = torch.where(mask, position_ids, torch.full_like(position_ids, -1))
    phase_probabilities = torch.nn.functional.one_hot(phase_indices, num_classes=4).float()
    phase_probabilities = phase_probabilities.masked_fill(~mask.unsqueeze(-1), 0.0)
    return E2SoftOutput(
        event_logits=torch.zeros_like(event_probabilities),
        phase_logits=torch.zeros_like(phase_probabilities),
        event_probabilities=event_probabilities,
        phase_probabilities=phase_probabilities,
        valid_mask=mask,
        timestamps=timestamps.to(dtype=torch.float64),
        position_ids=position_ids.to(dtype=torch.int64),
        next_states=tuple(_empty_e2_state() for _ in range(event_probabilities.shape[0])),
        audit=StreamReplayAudit(
            "e2",
            tuple(event_probabilities.shape[1] for _ in range(event_probabilities.shape[0])),
            tuple(0 for _ in range(event_probabilities.shape[0])),
            tuple(0 for _ in range(event_probabilities.shape[0])),
        ),
    )


def _candidate(
    name: str = "candidate-0",
    *,
    first_seen: float = 0.0,
    last_seen: float | None = None,
) -> CandidateIdentity:
    prototype = torch.zeros(256)
    prototype[0] = 1.0
    resolved_last_seen = first_seen if last_seen is None else last_seen
    return CandidateIdentity(
        name,
        prototype,
        1,
        8,
        0.8,
        first_seen=first_seen,
        last_seen=resolved_last_seen,
    )


def _confirmed(
    name: str = "identity-0",
    *,
    first_seen: float = 0.0,
    last_seen: float = 1.0,
) -> ConfirmedIdentity:
    prototype = torch.zeros(256)
    prototype[1] = 1.0
    return ConfirmedIdentity(name, prototype, first_seen, last_seen, 2)


def test_meta_topology_exact_parameter_count_builder_and_state_dict_boundary() -> None:
    config = load_config()
    with torch.device("meta"):
        module = build_state_bank(config)

    assert isinstance(module, StructuredStateBank)
    assert isinstance(module.semantic_projector, SemanticProjector)
    assert parameter_count(module.semantic_projector) == EXACT_PROJECTOR_PARAMETERS
    assert module.semantic_projector.head_type_embeddings.weight.shape == (4, HIDDEN_DIM)
    assert module.semantic_projector.hidden_projection.in_features == HIDDEN_DIM
    assert module.semantic_projector.hidden_projection.out_features == 1024
    assert module.semantic_projector.output_projection.out_features == SEMANTIC_DIM
    assert set(dict(module.named_children())) == {"semantic_projector"}
    assert not any(isinstance(child, nn.Dropout) for child in module.modules())
    assert not tuple(module.named_buffers())
    assert set(module.state_dict()) == {
        "semantic_projector.head_type_embeddings.weight",
        "semantic_projector.input_norm.weight",
        "semantic_projector.input_norm.bias",
        "semantic_projector.hidden_projection.weight",
        "semantic_projector.hidden_projection.bias",
        "semantic_projector.output_projection.weight",
        "semantic_projector.output_projection.bias",
    }


def test_projector_normalization_zero_fallback_head_conditioning_and_gradients(
    bank: StructuredStateBank,
) -> None:
    one_dimensional = bank.project(torch.randn(HIDDEN_DIM), HeadType.O1)
    assert one_dimensional.shape == (SEMANTIC_DIM,)
    torch.testing.assert_close(
        torch.linalg.vector_norm(one_dimensional),
        torch.tensor(1.0),
    )

    source = torch.randn(4, HIDDEN_DIM, generator=torch.Generator().manual_seed(7))
    source.requires_grad_(True)
    projected = bank.project(source, tuple(HeadType))

    assert projected.shape == (4, SEMANTIC_DIM)
    assert bool(torch.isfinite(projected).all())
    torch.testing.assert_close(
        torch.linalg.vector_norm(projected.float(), dim=-1),
        torch.ones(4),
    )
    assert not torch.allclose(projected[0], projected[1])
    weights = torch.linspace(0.5, 1.5, SEMANTIC_DIM)
    (projected * weights).sum().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert float(source.grad.abs().sum()) > 0.0
    assert bank.semantic_projector.head_type_embeddings.weight.grad is not None
    bank.zero_grad(set_to_none=True)

    projector = SemanticProjector(load_config().state_bank.semantic_projector)
    with torch.no_grad():
        projector.output_projection.weight.zero_()
        projector.output_projection.bias.zero_()
    fallback = projector(torch.zeros(2, HIDDEN_DIM), (HeadType.O1, HeadType.E2))
    expected = torch.zeros(2, SEMANTIC_DIM)
    expected[:, 0] = 1.0
    torch.testing.assert_close(fallback, expected)


def test_bfloat16_projector_and_hard_append_normalize_in_float32() -> None:
    torch.manual_seed(20260718)
    module = build_state_bank(load_config()).to(dtype=torch.bfloat16)
    module.eval()
    projected = module.project(torch.randn(HIDDEN_DIM, dtype=torch.bfloat16), HeadType.O2)

    assert projected.shape == (SEMANTIC_DIM,)
    assert projected.dtype is torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(projected),
        torch.tensor(1.0),
    )

    empty_view = module.view((module.reset("video-bf16-empty", "trajectory-bf16-empty"),))
    assert empty_view.embeddings.shape == (1, 0, SEMANTIC_DIM)
    assert empty_view.embeddings.dtype is torch.float32

    state = module.append_record(
        module.reset("video-bf16", "trajectory-bf16"),
        head_type=HeadType.O2,
        semantic_embedding=torch.linspace(-1.0, 1.0, SEMANTIC_DIM, dtype=torch.bfloat16),
        timestamp=0.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-bf16"),
    )
    stored = state.records[0].semantic_embedding
    assert stored.dtype is torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(stored),
        torch.tensor(1.0),
    )


def test_all_five_payloads_record_xor_head_and_detach_contracts() -> None:
    payloads = (
        (HeadType.O1, O1Payload(0, 0, (), baseline_initialized=False)),
        (HeadType.O2, _candidate(first_seen=1.0)),
        (HeadType.O2, _confirmed(first_seen=2.0, last_seen=2.0)),
        (HeadType.E1, E1Payload(E1EventKind.ACTION, 0, (), 0.0)),
        (
            HeadType.E2,
            E2Payload(E2EventKind.PERIODIC, 0, E2Phase.INACTIVE, (), ()),
        ),
    )
    records = []
    for index, (head_type, payload) in enumerate(payloads):
        records.append(
            StateRecord(
                record_id=f"record-{index}",
                video_id="video-a",
                trajectory_id="trajectory-a",
                head_type=head_type,
                semantic_embedding=_unit_semantic(index),
                timestamp=float(index),
                time_range=None,
                valid=True,
                confidence=0.9,
                payload=payload,
            )
        )
    assert tuple(record.head_type for record in records) == tuple(item[0] for item in payloads)

    with pytest.raises(ValueError, match="exactly one"):
        replace(records[0], timestamp=None, time_range=None)
    with pytest.raises(ValueError, match="exactly one"):
        replace(records[0], time_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="unit L2"):
        replace(records[0], semantic_embedding=torch.zeros(SEMANTIC_DIM))
    with pytest.raises(ValueError, match="detached"):
        replace(records[0], semantic_embedding=_unit_semantic(requires_grad=True))
    with pytest.raises(ValueError, match="head_type"):
        replace(records[0], head_type=HeadType.E1)
    grad_candidate = CandidateIdentity(
        "candidate-grad",
        torch.ones(256, requires_grad=True),
        1,
        8,
        0.8,
    )
    with pytest.raises(ValueError, match="detached"):
        replace(records[1], payload=grad_candidate)


def test_state_record_time_contract_blocks_future_payload_evidence() -> None:
    o1_payload = O1Payload(
        current_visible_count=1,
        baseline_count=0,
        active_slot_ids=(0,),
        slot_states=(
            O1SlotState(
                slot_id=0,
                is_object=True,
                is_target=True,
                visible=True,
                enter=False,
                exit=False,
                last_timestamp=5.0,
                last_position_id=2,
                confidence=0.9,
            ),
        ),
        last_timestamp=5.0,
        last_position_id=2,
        update_count=1,
    )
    e1_payload = E1Payload(
        event_kind=E1EventKind.ACTION,
        event_count=1,
        recent_event_times=(3.0,),
        cooldown_until=8.0,
        active=True,
        armed=False,
        candidate_start=4.0,
        last_timestamp=5.0,
        last_position_id=2,
    )
    e2_payload = E2Payload(
        event_kind=E2EventKind.PERIODIC,
        completed_count=1,
        phase=E2Phase.COMPLETED,
        completed_intervals=((1.0, 4.0),),
        recent_event_times=(4.0,),
        last_timestamp=5.0,
        last_position_id=2,
    )

    def aggregate_record(
        head_type: HeadType, payload: O1Payload | E1Payload | E2Payload
    ) -> StateRecord:
        return StateRecord(
            record_id=f"record-{head_type.value}",
            video_id="video-time-contract",
            trajectory_id="trajectory-time-contract",
            head_type=head_type,
            semantic_embedding=_unit_semantic(),
            timestamp=5.0,
            time_range=None,
            valid=True,
            confidence=0.9,
            payload=payload,
        )

    o1_record = aggregate_record(HeadType.O1, o1_payload)
    e1_record = aggregate_record(HeadType.E1, e1_payload)
    e2_record = aggregate_record(HeadType.E2, e2_payload)

    with pytest.raises(ValueError, match="last_timestamp must match"):
        replace(o1_record, payload=replace(o1_payload, last_timestamp=4.0))
    with pytest.raises(ValueError, match="O1 slot time"):
        future_slot = replace(o1_payload.slot_states[0], last_timestamp=6.0)
        replace(o1_record, payload=replace(o1_payload, slot_states=(future_slot,)))
    with pytest.raises(ValueError, match="require a point timestamp"):
        replace(o1_record, timestamp=None, time_range=(0.0, 5.0))

    with pytest.raises(ValueError, match="last_timestamp must match"):
        replace(e1_record, payload=replace(e1_payload, last_timestamp=4.0))
    with pytest.raises(ValueError, match="E1 event time"):
        replace(e1_record, payload=replace(e1_payload, recent_event_times=(6.0,)))
    with pytest.raises(ValueError, match="E1 candidate time"):
        replace(e1_record, payload=replace(e1_payload, candidate_start=6.0))

    with pytest.raises(ValueError, match="last_timestamp must match"):
        replace(e2_record, payload=replace(e2_payload, last_timestamp=4.0))
    with pytest.raises(ValueError, match="E2 interval time"):
        replace(e2_record, payload=replace(e2_payload, completed_intervals=((1.0, 6.0),)))
    with pytest.raises(ValueError, match="E2 event time"):
        replace(e2_record, payload=replace(e2_payload, recent_event_times=(6.0,)))
    with pytest.raises(ValueError, match="E2 candidate time"):
        active_payload = E2Payload(
            event_kind=E2EventKind.PERIODIC,
            completed_count=0,
            phase=E2Phase.ACTIVE,
            completed_intervals=(),
            recent_event_times=(),
            current_start=6.0,
            last_timestamp=5.0,
            last_position_id=2,
        )
        replace(e2_record, payload=active_payload)


def test_o2_record_time_contract_preserves_first_seen_and_allows_later_last_seen() -> None:
    point_candidate = _candidate("candidate-time", first_seen=2.0, last_seen=5.0)
    point_confirmed = _confirmed("identity-time", first_seen=2.0, last_seen=5.0)
    point_record = StateRecord(
        record_id="record-point",
        video_id="video-o2-time",
        trajectory_id="trajectory-o2-time",
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=point_confirmed,
    )
    StateRecord(
        record_id="record-candidate-point",
        video_id="video-o2-time",
        trajectory_id="trajectory-o2-time",
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(1),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=point_candidate,
    )
    range_record = replace(
        point_record,
        record_id="record-range",
        timestamp=None,
        time_range=(2.0, 5.0),
    )

    assert isinstance(range_record.payload, ConfirmedIdentity)
    assert range_record.payload.last_seen == 5.0
    with pytest.raises(ValueError, match="point record timestamp"):
        replace(point_record, timestamp=3.0)
    with pytest.raises(ValueError, match="range record boundaries"):
        replace(range_record, time_range=(1.0, 5.0))
    with pytest.raises(ValueError, match="range record boundaries"):
        replace(range_record, time_range=(2.0, 6.0))


def test_functional_crud_tombstones_snapshot_clear_and_release(
    bank: StructuredStateBank,
) -> None:
    fresh = bank.reset("video-crud", "trajectory-crud")
    semantic = _unit_semantic(requires_grad=True)
    first = bank.append_record(
        fresh,
        head_type=HeadType.O2,
        semantic_embedding=semantic,
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate(first_seen=1.0),
    )
    assert not fresh.records
    assert first.records[0].record_id == "record-00000000"
    assert not first.records[0].semantic_embedding.requires_grad
    assert first.records[0].semantic_embedding.untyped_storage().data_ptr() != (
        semantic.untyped_storage().data_ptr()
    )

    first_payload = first.records[0].payload
    assert isinstance(first_payload, CandidateIdentity)
    replacement = replace(
        first.records[0],
        timestamp=2.0,
        confidence=0.9,
        payload=replace(first_payload, first_seen=2.0, last_seen=2.0),
    )
    updated = bank.update_record(first, replacement)
    assert first.records[0].timestamp == 1.0
    assert updated.records[0].timestamp == 2.0
    snapshot = bank.snapshot(updated)
    assert snapshot is not updated
    assert snapshot.video_id == updated.video_id
    assert snapshot.trajectory_id == updated.trajectory_id
    assert snapshot.audit_log == updated.audit_log
    assert snapshot.issued_record_ids == updated.issued_record_ids
    assert snapshot.next_record_sequence == updated.next_record_sequence
    assert snapshot.released == updated.released
    assert snapshot.version == updated.version
    snapshot_record = snapshot.records[0]
    updated_record = updated.records[0]
    assert (
        snapshot_record.record_id,
        snapshot_record.video_id,
        snapshot_record.trajectory_id,
        snapshot_record.head_type,
        snapshot_record.timestamp,
        snapshot_record.time_range,
        snapshot_record.valid,
        snapshot_record.confidence,
    ) == (
        updated_record.record_id,
        updated_record.video_id,
        updated_record.trajectory_id,
        updated_record.head_type,
        updated_record.timestamp,
        updated_record.time_range,
        updated_record.valid,
        updated_record.confidence,
    )
    torch.testing.assert_close(
        snapshot_record.semantic_embedding,
        updated_record.semantic_embedding,
    )
    assert snapshot_record.semantic_embedding.untyped_storage().data_ptr() != (
        updated_record.semantic_embedding.untyped_storage().data_ptr()
    )
    assert isinstance(snapshot_record.payload, CandidateIdentity)
    assert isinstance(updated_record.payload, CandidateIdentity)
    assert (
        snapshot_record.payload.candidate_id,
        snapshot_record.payload.observation_count,
        snapshot_record.payload.ttl_remaining,
        snapshot_record.payload.confidence,
    ) == (
        updated_record.payload.candidate_id,
        updated_record.payload.observation_count,
        updated_record.payload.ttl_remaining,
        updated_record.payload.confidence,
    )
    torch.testing.assert_close(
        snapshot_record.payload.identity_prototype,
        updated_record.payload.identity_prototype,
    )
    assert snapshot_record.payload.identity_prototype.untyped_storage().data_ptr() != (
        updated_record.payload.identity_prototype.untyped_storage().data_ptr()
    )

    with pytest.raises(ValueError, match="explicit invalidation"):
        bank.update_record(updated, replace(updated.records[0], valid=False))

    invalidated = bank.invalidate_record(
        updated,
        updated.records[0].record_id,
        audit_timestamp=2.5,
        reason="test",
    )
    assert invalidated.records[0].valid is False
    duplicate_invalidation = bank.invalidate_record(
        invalidated,
        invalidated.records[0].record_id,
        audit_timestamp=3.0,
        reason="test-again",
    )
    assert duplicate_invalidation.audit_log[-1].action == "invalidate_duplicate"
    with pytest.raises(ValueError, match="terminal"):
        bank.update_record(invalidated, invalidated.records[0])

    cleared = bank.clear(invalidated)
    assert not cleared.records
    assert cleared.issued_record_ids == ("record-00000000",)
    second = bank.append_record(
        cleared,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(2),
        timestamp=4.0,
        time_range=None,
        valid=True,
        confidence=0.7,
        payload=_confirmed(first_seen=4.0, last_seen=4.0),
    )
    assert second.records[0].record_id == "record-00000001"

    released = bank.release(second)
    assert released.released and not released.records and not released.issued_record_ids
    for operation in (
        lambda: bank.records_for(released),
        lambda: bank.snapshot(released),
        lambda: bank.view((released,)),
        lambda: bank.clear(released),
    ):
        with pytest.raises(ValueError, match="released"):
            operation()


def test_o2_lifecycle_bridge_links_records_and_freezes_confirmed_first_seen(
    bank: StructuredStateBank,
) -> None:
    prototype = torch.zeros(256)
    prototype[3] = 1.0
    candidate = CandidateIdentity(
        candidate_id="candidate-bridge",
        identity_prototype=prototype,
        observation_count=1,
        ttl_remaining=8,
        confidence=0.8,
        first_seen=1.0,
        last_seen=1.0,
        first_seen_position_id=4,
        last_seen_position_id=4,
        last_reliable_chunk_index=0,
        reliable_streak=1,
        semantic_record_id=None,
    )
    fresh = bank.reset("video-o2-bridge", "trajectory-o2-bridge")
    with_candidate, candidate_record = bank.append_o2_candidate(
        fresh,
        semantic_embedding=_unit_semantic(3, requires_grad=True),
        candidate=candidate,
        confidence=0.8,
    )
    assert not fresh.records
    assert candidate_record.timestamp == 1.0
    assert isinstance(candidate_record.payload, CandidateIdentity)
    assert candidate_record.payload.semantic_record_id == candidate_record.record_id
    assert not candidate_record.payload.identity_prototype.requires_grad

    updated_candidate = replace(
        candidate_record.payload,
        observation_count=2,
        last_seen=2.0,
        last_seen_position_id=8,
        last_reliable_chunk_index=1,
        reliable_streak=2,
    )
    candidate_updated = bank.update_o2_candidate(
        with_candidate,
        candidate=updated_candidate,
        semantic_embedding=_unit_semantic(4),
        confidence=0.9,
        audit_timestamp=2.0,
    )
    assert candidate_updated.records[0].timestamp == 1.0
    assert with_candidate.records[0].confidence == 0.8

    confirmed_draft = ConfirmedIdentity(
        identity_id="identity-bridge",
        identity_prototype=updated_candidate.identity_prototype,
        first_seen=updated_candidate.first_seen,
        last_seen=updated_candidate.last_seen,
        observation_count=updated_candidate.observation_count,
        semantic_record_id=None,
        prototype_version=0,
        first_seen_position_id=updated_candidate.first_seen_position_id,
        last_seen_position_id=updated_candidate.last_seen_position_id,
    )
    promoted, confirmed_record = bank.promote_o2_candidate(
        candidate_updated,
        candidate_record.record_id,
        semantic_embedding=_unit_semantic(5),
        confirmed=confirmed_draft,
        confidence=0.9,
        audit_timestamp=2.0,
    )
    assert candidate_updated.records[0].valid is True
    assert promoted.records[0].valid is False
    assert isinstance(confirmed_record.payload, ConfirmedIdentity)
    assert confirmed_record.payload.semantic_record_id == confirmed_record.record_id
    assert confirmed_record.timestamp == confirmed_draft.first_seen == 1.0
    view = bank.view((promoted,), HeadType.O2)
    assert view.record_kinds == ((StateRecordKind.O2_CANDIDATE, StateRecordKind.O2_CONFIRMED),)
    assert view.retrieval_eligible_mask.tolist() == [[False, True]]

    updated_confirmed = replace(
        confirmed_record.payload,
        last_seen=3.0,
        last_seen_position_id=12,
        observation_count=3,
        prototype_version=1,
    )
    confirmed_updated = bank.update_o2_confirmed(
        promoted,
        confirmed=updated_confirmed,
        semantic_embedding=_unit_semantic(6),
        confidence=0.95,
        audit_timestamp=3.0,
    )
    stored_confirmed = confirmed_updated.records[1]
    assert stored_confirmed.timestamp == 1.0
    assert isinstance(stored_confirmed.payload, ConfirmedIdentity)
    assert stored_confirmed.payload.last_seen == 3.0
    assert stored_confirmed.payload.prototype_version == 1

    with pytest.raises(ValueError, match="cannot change first_seen"):
        bank.update_o2_confirmed(
            promoted,
            confirmed=replace(confirmed_record.payload, first_seen=0.5),
            semantic_embedding=_unit_semantic(6),
            confidence=0.95,
            audit_timestamp=3.0,
        )
    with pytest.raises(ValueError, match="cannot be promoted"):
        bank.promote_o2_candidate(
            promoted,
            candidate_record.record_id,
            semantic_embedding=_unit_semantic(5),
            confirmed=confirmed_draft,
            confidence=0.9,
            audit_timestamp=2.0,
        )


def test_dynamic_batched_view_keeps_present_and_record_valid_masks_separate(
    bank: StructuredStateBank,
) -> None:
    first = bank.reset("video-view-a", "trajectory-view-a")
    first = bank.append_record(
        first,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(0),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-a", first_seen=1.0),
    )
    first = bank.append_record(
        first,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(1),
        timestamp=None,
        time_range=(1.0, 2.0),
        valid=True,
        confidence=0.9,
        payload=_confirmed("identity-a", first_seen=1.0, last_seen=2.0),
    )
    first = bank.invalidate_record(
        first,
        first.records[0].record_id,
        audit_timestamp=2.5,
        reason="view-invalid",
    )
    second = bank.reset("video-view-b", "trajectory-view-b")
    second = bank.append_record(
        second,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(2),
        timestamp=3.0,
        time_range=None,
        valid=True,
        confidence=0.7,
        payload=_candidate("candidate-b", first_seen=3.0),
    )
    third = bank.reset("video-view-c", "trajectory-view-c")
    parameter_shapes = tuple(parameter.shape for parameter in bank.parameters())

    view = bank.view((first, second, third))
    assert view.embeddings.shape == (3, 2, SEMANTIC_DIM)
    assert view.present_mask.tolist() == [[True, True], [True, False], [False, False]]
    assert view.record_valid_mask.tolist() == [[False, True], [True, False], [False, False]]
    assert view.record_kinds == (
        (StateRecordKind.O2_CANDIDATE, StateRecordKind.O2_CONFIRMED),
        (StateRecordKind.O2_CANDIDATE, None),
        (None, None),
    )
    assert view.retrieval_eligible_mask.tolist() == [
        [False, True],
        [False, False],
        [False, False],
    ]
    assert view.n_state.tolist() == [2, 1, 0]
    assert view.owner_record_counts.tolist() == [2, 1, 0]
    assert view.video_ids == ("video-view-a", "video-view-b", "video-view-c")
    assert view.trajectory_ids == (
        "trajectory-view-a",
        "trajectory-view-b",
        "trajectory-view-c",
    )
    assert view.bank_versions == (first.version, second.version, third.version)
    assert view.record_ids[0] == tuple(record.record_id for record in first.records)
    assert (
        tuple(record.record_id if record is not None else None for record in view.cloned_records[0])
        == view.record_ids[0]
    )
    assert torch.count_nonzero(view.embeddings[~view.present_mask]) == 0
    assert torch.all(view.timestamps[~view.present_mask] == -1.0)
    assert torch.all(view.time_ranges[~view.present_mask] == -1.0)
    assert tuple(parameter.shape for parameter in bank.parameters()) == parameter_shapes
    original = first.records[0].semantic_embedding.clone()
    view.embeddings[0, 0].zero_()
    torch.testing.assert_close(first.records[0].semantic_embedding, original)
    cloned = view.cloned_records[0][0]
    assert cloned is not None
    cloned.semantic_embedding.zero_()
    torch.testing.assert_close(first.records[0].semantic_embedding, original)

    empty_view = bank.view(
        (
            bank.reset("video-empty-a", "trajectory-empty-a"),
            bank.reset("video-empty-b", "trajectory-empty-b"),
        )
    )
    assert empty_view.embeddings.shape == (2, 0, SEMANTIC_DIM)
    assert empty_view.retrieval_eligible_mask.shape == (2, 0)
    assert empty_view.owner_record_counts.tolist() == [0, 0]
    assert empty_view.cloned_records == ((), ())
    with pytest.raises(ValueError, match="owners must be unique"):
        bank.view((first, first))


def test_batched_view_supports_row_wise_head_partitions_and_none_rows(
    bank: StructuredStateBank,
) -> None:
    first = bank.append_record(
        bank.reset("video-row-a", "trajectory-row-a"),
        head_type=HeadType.O1,
        semantic_embedding=_unit_semantic(0),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=O1Payload(0, 0, ()),
    )
    first = bank.append_record(
        first,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(1),
        timestamp=1.5,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-row-a", first_seen=1.5),
    )
    first = bank.append_record(
        first,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(2),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.95,
        payload=_confirmed("identity-row-a", first_seen=2.0, last_seen=2.0),
    )
    second = bank.append_record(
        bank.reset("video-row-b", "trajectory-row-b"),
        head_type=HeadType.E1,
        semantic_embedding=_unit_semantic(3),
        timestamp=3.0,
        time_range=None,
        valid=True,
        confidence=0.7,
        payload=E1Payload(E1EventKind.ACTION, 0, (), 0.0),
    )
    third = bank.append_record(
        bank.reset("video-row-c", "trajectory-row-c"),
        head_type=HeadType.O1,
        semantic_embedding=_unit_semantic(4),
        timestamp=4.0,
        time_range=None,
        valid=True,
        confidence=0.6,
        payload=O1Payload(0, 0, ()),
    )

    view = bank.view(
        (first, second, third),
        (HeadType.O2, HeadType.E1, None),
    )

    assert view.embeddings.shape == (3, 2, SEMANTIC_DIM)
    assert view.n_state.tolist() == [2, 1, 0]
    assert view.owner_record_counts.tolist() == [3, 1, 1]
    assert view.present_mask.tolist() == [[True, True], [True, False], [False, False]]
    assert view.head_types == (
        (HeadType.O2, HeadType.O2),
        (HeadType.E1, None),
        (None, None),
    )
    assert view.record_kinds == (
        (StateRecordKind.O2_CANDIDATE, StateRecordKind.O2_CONFIRMED),
        (StateRecordKind.E1_AGGREGATE, None),
        (None, None),
    )
    assert view.retrieval_eligible_mask.tolist() == [
        [False, True],
        [True, False],
        [False, False],
    ]
    assert all(record is None for record in view.cloned_records[2])

    scalar_view = bank.view((first, second, third), HeadType.O2)
    assert scalar_view.n_state.tolist() == [2, 0, 0]
    unfiltered_view = bank.view((first, second, third), None)
    assert unfiltered_view.n_state.tolist() == [3, 1, 1]
    assert torch.equal(unfiltered_view.n_state, unfiltered_view.owner_record_counts)

    with pytest.raises(ValueError, match="head filters must match the batch size"):
        bank.view((first, second), (HeadType.O1,))
    with pytest.raises(TypeError, match="must contain HeadType or None"):
        bank.view((first,), ("o1",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="released State Bank runtime"):
        bank.view((bank.release(third),), (None,))


def test_state_bank_view_strictly_validates_aligned_snapshot_metadata_and_storage(
    bank: StructuredStateBank,
) -> None:
    state = bank.append_record(
        bank.reset("video-view-contract", "trajectory-view-contract"),
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(0),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-view-contract", first_seen=1.0),
    )
    state = bank.append_record(
        state,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(0),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=_confirmed("identity-view-contract", first_seen=2.0, last_seen=2.0),
    )
    view = bank.view((state,), HeadType.O2)
    first = view.cloned_records[0][0]
    second = view.cloned_records[0][1]
    assert first is not None and second is not None

    isolated = clone_state_record(first)
    assert isolated.semantic_embedding.untyped_storage().data_ptr() != (
        first.semantic_embedding.untyped_storage().data_ptr()
    )
    isolated.semantic_embedding.zero_()
    assert torch.count_nonzero(first.semantic_embedding) == 1
    with pytest.raises(TypeError, match="requires a StateRecord"):
        clone_state_record("not-a-record")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="owner metadata is inconsistent"):
        replace(view, video_ids=("video-wrong",))
    with pytest.raises(ValueError, match="record ID metadata is inconsistent"):
        replace(view, record_ids=(("record-wrong", view.record_ids[0][1]),))
    with pytest.raises(ValueError, match="record head metadata is inconsistent"):
        replace(view, head_types=((HeadType.O1, HeadType.O2),))
    with pytest.raises(ValueError, match="record kind metadata is inconsistent"):
        replace(
            view,
            record_kinds=((StateRecordKind.O2_CONFIRMED, StateRecordKind.O2_CONFIRMED),),
        )

    invalid_mask = view.record_valid_mask.clone()
    invalid_mask[0, 0] = False
    with pytest.raises(ValueError, match="record validity metadata is inconsistent"):
        replace(view, record_valid_mask=invalid_mask)

    shared_second = replace(second, semantic_embedding=first.semantic_embedding)
    shared_embeddings = view.embeddings.clone()
    shared_embeddings[0, 1] = first.semantic_embedding
    with pytest.raises(ValueError, match="must not share mutable tensor storage"):
        replace(
            view,
            embeddings=shared_embeddings,
            cloned_records=((first, shared_second),),
        )


def test_batched_view_rejects_cross_owner_shared_tensor_storage(
    bank: StructuredStateBank,
) -> None:
    first = bank.append_record(
        bank.reset("video-shared-a", "trajectory-shared-a"),
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-shared", first_seen=1.0),
    )
    shared_record = replace(
        first.records[0],
        video_id="video-shared-b",
        trajectory_id="trajectory-shared-b",
    )
    second = replace(
        first,
        video_id="video-shared-b",
        trajectory_id="trajectory-shared-b",
        records=(shared_record,),
    )

    with pytest.raises(ValueError, match="must not share mutable tensor storage"):
        bank.view((first, second))


def test_aggregate_partitions_reject_duplicate_generic_append(
    bank: StructuredStateBank,
) -> None:
    aggregate_payloads = (
        (HeadType.O1, O1Payload(0, 0, (), baseline_initialized=False)),
        (HeadType.E1, E1Payload(E1EventKind.ACTION, 0, (), 0.0)),
        (
            HeadType.E2,
            E2Payload(E2EventKind.PERIODIC, 0, E2Phase.INACTIVE, (), ()),
        ),
    )
    for index, (head_type, payload) in enumerate(aggregate_payloads):
        state = bank.append_record(
            bank.reset(f"video-aggregate-{index}", f"trajectory-aggregate-{index}"),
            head_type=head_type,
            semantic_embedding=_unit_semantic(index),
            timestamp=0.0,
            time_range=None,
            valid=True,
            confidence=0.8,
            payload=payload,
        )
        with pytest.raises(ValueError, match="partition already has its aggregate record"):
            bank.append_record(
                state,
                head_type=head_type,
                semantic_embedding=_unit_semantic(index + 1),
                timestamp=1.0,
                time_range=None,
                valid=True,
                confidence=0.7,
                payload=payload,
            )


def test_event_kind_provenance_is_required_and_frozen(
    bank: StructuredStateBank,
) -> None:
    e1_observation = _e1_output(
        torch.zeros(1, 3),
        torch.tensor([0.0]),
        torch.tensor([0]),
    )
    e2_observation = _e2_output(
        torch.zeros(1, 4),
        torch.tensor([0]),
        torch.tensor([0.0]),
        torch.tensor([0]),
    )
    semantics = _unit_semantic().reshape(1, 1, -1)

    e1_states = tuple(
        bank.update_e1(
            bank.reset(f"video-e1-{kind.value}", f"trajectory-e1-{kind.value}"),
            e1_observation,
            semantics,
            event_kind=kind,
        )
        for kind in E1EventKind
    )
    e2_states = tuple(
        bank.update_e2(
            bank.reset(f"video-e2-{kind.value}", f"trajectory-e2-{kind.value}"),
            e2_observation,
            semantics,
            event_kind=kind,
        )
        for kind in E2EventKind
    )
    assert tuple(state.records[0].payload.event_kind for state in e1_states) == tuple(E1EventKind)
    assert tuple(state.records[0].payload.event_kind for state in e2_states) == tuple(E2EventKind)
    assert dict(e1_states[0].audit_log[-1].details)["event_kind"] == "action"
    assert dict(e2_states[0].audit_log[-1].details)["event_kind"] == "periodic"

    with pytest.raises(ValueError, match="E1 event kind cannot change"):
        bank.update_e1(
            e1_states[0],
            e1_observation,
            semantics,
            event_kind=E1EventKind.TRANSIT,
        )
    with pytest.raises(ValueError, match="E2 event kind cannot change"):
        bank.update_e2(
            e2_states[0],
            e2_observation,
            semantics,
            event_kind=E2EventKind.EPISODE,
        )

    e1_record = e1_states[0].records[0]
    assert isinstance(e1_record.payload, E1Payload)
    with pytest.raises(ValueError, match="replacement cannot change E1 event kind"):
        bank.update_record(
            e1_states[0],
            replace(
                e1_record,
                payload=replace(e1_record.payload, event_kind=E1EventKind.TRANSIT),
            ),
        )
    e2_record = e2_states[0].records[0]
    assert isinstance(e2_record.payload, E2Payload)
    with pytest.raises(ValueError, match="replacement cannot change E2 event kind"):
        bank.update_record(
            e2_states[0],
            replace(
                e2_record,
                payload=replace(e2_record.payload, event_kind=E2EventKind.EPISODE),
            ),
        )


def test_first_aggregate_with_older_cross_head_time_keeps_audit_monotonic(
    bank: StructuredStateBank,
) -> None:
    state = bank.append_record(
        bank.reset("video-audit", "trajectory-audit"),
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(),
        timestamp=100.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-audit", first_seen=100.0),
    )
    state = bank.update_o1(
        state,
        _o1_output(
            torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.1, 0.9]]),
            timestamp=1.0,
            position_id=0,
        ),
        _unit_semantic(1),
        observation_timestamp=1.0,
        observation_position_id=0,
    )
    state = bank.update_e1(
        state,
        _e1_output(torch.zeros(1, 3), torch.tensor([2.0]), torch.tensor([0])),
        _unit_semantic(2).reshape(1, 1, -1),
        event_kind=E1EventKind.ACTION,
    )
    state = bank.update_e2(
        state,
        _e2_output(
            torch.zeros(1, 4),
            torch.tensor([0]),
            torch.tensor([3.0]),
            torch.tensor([0]),
        ),
        _unit_semantic(3).reshape(1, 1, -1),
        event_kind=E2EventKind.PERIODIC,
    )

    audit_timestamps = tuple(entry.timestamp for entry in state.audit_log)
    assert audit_timestamps == tuple(sorted(audit_timestamps))
    assert set(audit_timestamps) == {100.0}
    assert {entry.action for entry in state.audit_log} >= {
        "o1_update",
        "e1_fsm_update",
        "e2_fsm_update",
    }


def test_o1_explicit_baseline_zero_slot_row_metadata_overflow_and_idempotency(
    bank: StructuredStateBank,
) -> None:
    state = bank.reset("video-o1", "trajectory-o1")
    visible = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.1, 0.1, 0.5],
            [0.9, 0.9, 0.9, 0.8, 0.1, 0.9],
        ]
    )
    state = bank.update_o1(
        state,
        _o1_output(visible, timestamp=1.75, position_id=7),
        _unit_semantic(),
        observation_timestamp=1.75,
        observation_position_id=7,
    )
    first_record_id = state.records[0].record_id
    first_payload = state.records[0].payload
    assert isinstance(first_payload, O1Payload)
    assert first_payload.current_visible_count == 2
    assert first_payload.baseline_initialized is False

    empty_mask = torch.zeros(1, 2, dtype=torch.bool)
    state = bank.update_o1(
        state,
        _o1_output(
            torch.zeros(1, 2, 6),
            timestamp=2.75,
            position_id=11,
            valid_mask=empty_mask,
        ),
        _unit_semantic(1),
        observation_timestamp=2.75,
        observation_position_id=11,
        set_baseline=True,
        slot_overflow_count=2,
    )
    baseline_payload = state.records[0].payload
    assert isinstance(baseline_payload, O1Payload)
    assert state.records[0].record_id == first_record_id
    assert baseline_payload.current_visible_count == baseline_payload.baseline_count == 2
    assert baseline_payload.active_slot_ids == first_payload.active_slot_ids
    assert baseline_payload.slot_states == first_payload.slot_states
    assert baseline_payload.baseline_initialized
    assert baseline_payload.baseline_position_id == 11
    assert baseline_payload.last_position_id == 11
    assert baseline_payload.last_spatial_overflow_count == 2
    assert dict(state.audit_log[-1].details)["slot_overflow_delta"] == 2

    one_visible = torch.tensor(
        [
            [0.9, 0.9, 0.9, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
        ]
    )
    state = bank.update_o1(
        state,
        _o1_output(one_visible, timestamp=3.75, position_id=15),
        _unit_semantic(2),
        observation_timestamp=3.75,
        observation_position_id=15,
        slot_overflow_count=5,
    )
    payload = state.records[0].payload
    assert isinstance(payload, O1Payload)
    assert payload.current_visible_count == 1 and payload.baseline_count == 2
    assert payload.update_count == 3
    assert dict(state.audit_log[-1].details)["slot_overflow_delta"] == 3

    drifted_evidence = one_visible.flip(0)
    drifted = bank.update_o1(
        state,
        _o1_output(drifted_evidence, timestamp=9.75, position_id=15),
        _unit_semantic(3),
        observation_timestamp=9.75,
        observation_position_id=15,
        slot_overflow_count=8,
    )
    drifted_payload = drifted.records[0].payload
    assert isinstance(drifted_payload, O1Payload)
    assert drifted_payload == replace(payload, last_spatial_overflow_count=8)
    torch.testing.assert_close(
        drifted.records[0].semantic_embedding,
        state.records[0].semantic_embedding,
    )
    drift_audit = dict(drifted.audit_log[-1].details)
    assert drifted.audit_log[-1].action == "o1_duplicate_position"
    assert drift_audit["evidence_comparable"] is True
    assert drift_audit["timestamp_drift"] is True
    assert drift_audit["slot_evidence_drift_count"] == 2
    assert drift_audit["semantic_drift"] is True
    assert drift_audit["slot_overflow_delta"] == 3

    duplicate = bank.update_o1(
        state,
        _o1_output(one_visible, timestamp=2.75, position_id=11),
        _unit_semantic(3),
        observation_timestamp=2.75,
        observation_position_id=11,
        slot_overflow_count=5,
    )
    assert duplicate.records[0].payload == payload
    assert duplicate.audit_log[-1].action == "o1_duplicate_position"
    with pytest.raises(ValueError, match="only be initialized once"):
        bank.update_o1(
            state,
            _o1_output(one_visible, timestamp=4.75, position_id=19),
            _unit_semantic(),
            observation_timestamp=4.75,
            observation_position_id=19,
            set_baseline=True,
            slot_overflow_count=5,
        )
    with pytest.raises(ValueError, match="cannot decrease"):
        bank.update_o1(
            state,
            _o1_output(one_visible, timestamp=4.75, position_id=19),
            _unit_semantic(),
            observation_timestamp=4.75,
            observation_position_id=19,
            slot_overflow_count=4,
        )


def test_o1_exit_evidence_forces_slot_not_visible(bank: StructuredStateBank) -> None:
    state = bank.update_o1(
        bank.reset("video-o1-exit", "trajectory-o1-exit"),
        _o1_output(
            torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.9, 0.9]]),
            timestamp=0.0,
            position_id=0,
        ),
        _unit_semantic(),
        observation_timestamp=0.0,
        observation_position_id=0,
    )
    payload = state.records[0].payload
    assert isinstance(payload, O1Payload)
    assert payload.current_visible_count == 0
    assert payload.active_slot_ids == ()
    assert len(payload.slot_states) == 1
    assert payload.slot_states[0].exit is True
    assert payload.slot_states[0].visible is False


def test_fresh_all_invalid_e1_and_e2_outputs_do_not_enter_bank(
    bank: StructuredStateBank,
) -> None:
    valid_mask = torch.zeros(3, dtype=torch.bool)
    semantics = _unit_semantic().reshape(1, 1, -1).expand(1, 3, -1)

    fresh_e1 = bank.reset("video-e1-invalid", "trajectory-e1-invalid")
    unchanged_e1 = bank.update_e1(
        fresh_e1,
        _e1_output(
            torch.zeros(3, 3),
            torch.arange(3, dtype=torch.float32),
            torch.arange(3),
            valid_mask=valid_mask,
        ),
        semantics,
        event_kind=E1EventKind.ACTION,
    )
    assert unchanged_e1 is fresh_e1
    assert not unchanged_e1.records and not unchanged_e1.audit_log
    assert unchanged_e1.version == 0

    fresh_e2 = bank.reset("video-e2-invalid", "trajectory-e2-invalid")
    unchanged_e2 = bank.update_e2(
        fresh_e2,
        _e2_output(
            torch.zeros(3, 4),
            torch.zeros(3, dtype=torch.int64),
            torch.arange(3, dtype=torch.float32),
            torch.arange(3),
            valid_mask=valid_mask,
        ),
        semantics,
        event_kind=E2EventKind.PERIODIC,
    )
    assert unchanged_e2 is fresh_e2
    assert not unchanged_e2.records and not unchanged_e2.audit_log
    assert unchanged_e2.version == 0


def test_exact_fp32_fsm_thresholds_with_non_prefix_valid_masks(
    bank: StructuredStateBank,
) -> None:
    e1_valid = torch.tensor([True, False, True, False, True])
    e1_probabilities = torch.tensor(
        [
            [0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.7, 0.7, 0.7],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    e1_semantics = _unit_semantic().reshape(1, 1, -1).expand(1, 5, -1)
    e1_state = bank.update_e1(
        bank.reset("video-e1-threshold", "trajectory-e1-threshold"),
        _e1_output(
            e1_probabilities,
            torch.tensor([0.0, 9.0, 0.1, 9.0, 0.2]),
            torch.tensor([0, 99, 1, 99, 2]),
            valid_mask=e1_valid,
        ),
        e1_semantics,
        event_kind=E1EventKind.ACTION,
    )
    e1_payload = e1_state.records[0].payload
    assert isinstance(e1_payload, E1Payload)
    assert e1_payload.event_count == 1
    assert e1_payload.recent_event_times == pytest.approx((0.1,))
    assert e1_payload.active is False and e1_payload.armed is True
    assert e1_payload.last_position_id == 2

    e2_valid = torch.tensor([True, False, True, False, True, False, True])
    e2_events = torch.tensor(
        [
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    e2_semantics = _unit_semantic(1).reshape(1, 1, -1).expand(1, 7, -1)
    e2_state = bank.update_e2(
        bank.reset("video-e2-threshold", "trajectory-e2-threshold"),
        _e2_output(
            e2_events,
            torch.tensor([1, 0, 2, 0, 3, 0, 0]),
            torch.tensor([0.0, 9.0, 0.1, 9.0, 0.2, 9.0, 0.3]),
            torch.tensor([0, 99, 1, 99, 2, 99, 3]),
            valid_mask=e2_valid,
        ),
        e2_semantics,
        event_kind=E2EventKind.PERIODIC,
    )
    e2_payload = e2_state.records[0].payload
    assert isinstance(e2_payload, E2Payload)
    assert e2_payload.completed_count == 1
    torch.testing.assert_close(
        torch.tensor(e2_payload.completed_intervals),
        torch.tensor(((0.0, 0.2),)),
    )
    assert e2_payload.phase is E2Phase.INACTIVE
    assert e2_payload.rearm_suppression_count == 0
    assert e2_payload.last_position_id == 3


def test_e1_hysteresis_cooldown_nms_overlap_and_gap_fail_closed(
    bank: StructuredStateBank,
) -> None:
    probabilities = torch.tensor(
        [
            [0.8, 0.0, 0.0],
            [0.9, 0.8, 0.8],
            [0.9, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.9, 0.8, 0.8],
        ]
    )
    timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
    positions = torch.arange(7)
    semantics = torch.stack([_unit_semantic(index % 4) for index in range(7)]).unsqueeze(0)
    state = bank.update_e1(
        bank.reset("video-e1", "trajectory-e1"),
        _e1_output(probabilities, timestamps, positions),
        semantics,
        event_kind=E1EventKind.ACTION,
    )
    payload = state.records[0].payload
    assert isinstance(payload, E1Payload)
    assert payload.event_count == 2
    assert payload.recent_event_times == pytest.approx((0.1, 0.8))
    assert payload.duplicate_suppression_count == 1
    assert payload.cooldown_hit_count == 1
    assert payload.nms_suppression_count == 0
    assert payload.active is False and payload.armed is False

    overlap = bank.update_e1(
        state,
        _e1_output(probabilities[-2:], timestamps[-2:], positions[-2:]),
        semantics[:, -2:],
        event_kind=E1EventKind.ACTION,
    )
    overlap_payload = overlap.records[0].payload
    assert isinstance(overlap_payload, E1Payload)
    assert overlap_payload.event_count == payload.event_count
    assert overlap_payload.recent_event_times == payload.recent_event_times
    assert overlap_payload.active == payload.active
    assert overlap_payload.armed == payload.armed
    assert overlap_payload.candidate_start == payload.candidate_start
    assert overlap_payload.last_timestamp == payload.last_timestamp
    assert overlap_payload.last_position_id == payload.last_position_id
    assert overlap_payload.duplicate_suppression_count == (payload.duplicate_suppression_count + 2)
    assert overlap.audit_log[-1].action == "e1_overlap_ignored"
    assert dict(overlap.audit_log[-1].details)["duplicate_positions"] == 2
    with pytest.raises(ValueError, match="gaps"):
        bank.update_e1(
            state,
            _e1_output(torch.tensor([[0.2, 0.0, 0.0]]), torch.tensor([1.0]), torch.tensor([8])),
            _unit_semantic().reshape(1, 1, -1),
            event_kind=E1EventKind.ACTION,
        )

    nms_state = bank.append_record(
        bank.reset("video-e1-nms", "trajectory-e1-nms"),
        head_type=HeadType.E1,
        semantic_embedding=_unit_semantic(),
        timestamp=1.1,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=E1Payload(
            E1EventKind.ACTION,
            1,
            (1.0,),
            0.0,
            active=True,
            armed=False,
            candidate_start=1.1,
            last_timestamp=1.1,
            last_position_id=1,
        ),
    )
    nms_state = bank.update_e1(
        nms_state,
        _e1_output(torch.tensor([[0.9, 0.8, 0.8]]), torch.tensor([1.2]), torch.tensor([2])),
        _unit_semantic().reshape(1, 1, -1),
        event_kind=E1EventKind.ACTION,
    )
    nms_payload = nms_state.records[0].payload
    assert isinstance(nms_payload, E1Payload)
    assert nms_payload.event_count == 1 and nms_payload.nms_suppression_count == 1


def test_e2_phase_gated_transitions_rearm_overlap_and_conflicts(
    bank: StructuredStateBank,
) -> None:
    events = torch.tensor(
        [
            [0.7, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.8],
        ]
    )
    phases = torch.tensor([1, 2, 3, 3, 0, 1, 2, 3])
    timestamps = torch.arange(8, dtype=torch.float32) / 4.0
    positions = torch.arange(8)
    semantics = torch.stack([_unit_semantic(index % 4) for index in range(8)]).unsqueeze(0)
    state = bank.update_e2(
        bank.reset("video-e2", "trajectory-e2"),
        _e2_output(events, phases, timestamps, positions),
        semantics,
        event_kind=E2EventKind.PERIODIC,
    )
    payload = state.records[0].payload
    assert isinstance(payload, E2Payload)
    assert payload.completed_count == 2
    torch.testing.assert_close(
        torch.tensor(payload.completed_intervals),
        torch.tensor(((0.0, 0.5), (1.25, 1.75))),
    )
    assert payload.phase is E2Phase.COMPLETED
    assert payload.rearm_suppression_count == 1

    overlap = bank.update_e2(
        state,
        _e2_output(events[-2:], phases[-2:], timestamps[-2:], positions[-2:]),
        semantics[:, -2:],
        event_kind=E2EventKind.PERIODIC,
    )
    overlap_payload = overlap.records[0].payload
    assert isinstance(overlap_payload, E2Payload)
    assert overlap_payload.completed_count == payload.completed_count
    assert overlap_payload.completed_intervals == payload.completed_intervals
    assert overlap_payload.phase is payload.phase
    assert overlap_payload.current_start == payload.current_start
    assert overlap_payload.last_timestamp == payload.last_timestamp
    assert overlap_payload.last_position_id == payload.last_position_id
    assert overlap_payload.duplicate_suppression_count == (payload.duplicate_suppression_count + 2)
    assert overlap.audit_log[-1].action == "e2_overlap_ignored"
    assert dict(overlap.audit_log[-1].details)["duplicate_positions"] == 2
    with pytest.raises(ValueError, match="gaps"):
        bank.update_e2(
            state,
            _e2_output(
                torch.zeros(1, 4),
                torch.tensor([0]),
                torch.tensor([2.5]),
                torch.tensor([10]),
            ),
            _unit_semantic().reshape(1, 1, -1),
            event_kind=E2EventKind.PERIODIC,
        )

    conflict = bank.update_e2(
        bank.reset("video-e2-conflict", "trajectory-e2-conflict"),
        _e2_output(
            torch.tensor([[0.9, 0.9, 0.9, 0.9]]),
            torch.tensor([3]),
            torch.tensor([0.0]),
            torch.tensor([0]),
        ),
        _unit_semantic().reshape(1, 1, -1),
        event_kind=E2EventKind.PERIODIC,
    )
    conflict_payload = conflict.records[0].payload
    assert isinstance(conflict_payload, E2Payload)
    assert conflict_payload.phase is E2Phase.INACTIVE
    assert conflict_payload.completed_count == 0 and conflict_payload.conflict_count == 1


def test_e1_and_e2_history_513_eviction_preserves_exact_totals(
    bank: StructuredStateBank,
) -> None:
    event_count = 513
    e1_probabilities: list[list[float]] = []
    e1_times: list[float] = []
    for index in range(event_count):
        base = float(index)
        e1_probabilities.extend(([0.8, 0.0, 0.0], [0.9, 0.8, 0.8], [0.2, 0.0, 0.0]))
        e1_times.extend((base, base + 0.1, base + 0.2))
    e1_tensor = torch.tensor(e1_probabilities)
    e1_positions = torch.arange(e1_tensor.shape[0])
    e1_semantics = _unit_semantic().reshape(1, 1, -1).expand(1, e1_tensor.shape[0], -1)
    e1_state = bank.update_e1(
        bank.reset("video-e1-history", "trajectory-e1-history"),
        _e1_output(e1_tensor, torch.tensor(e1_times), e1_positions),
        e1_semantics,
        event_kind=E1EventKind.ACTION,
    )
    e1_payload = e1_state.records[0].payload
    assert isinstance(e1_payload, E1Payload)
    assert e1_payload.event_count == event_count
    assert len(e1_payload.recent_event_times) == 512
    assert e1_payload.history_eviction_count == 1 and e1_payload.history_truncated
    assert e1_payload.recent_event_times[0] == pytest.approx(1.1)

    e2_events: list[list[float]] = []
    e2_phases: list[int] = []
    e2_times: list[float] = []
    for index in range(event_count):
        base = float(index)
        e2_events.extend(
            ([0.7, 0.0, 0.0, 0.0], [0.0, 0.8, 0.7, 0.0], [0.0, 0.0, 0.0, 0.8], [0.0] * 4)
        )
        e2_phases.extend((1, 2, 3, 0))
        e2_times.extend((base, base + 0.1, base + 0.2, base + 0.3))
    e2_tensor = torch.tensor(e2_events)
    e2_positions = torch.arange(e2_tensor.shape[0])
    e2_semantics = _unit_semantic().reshape(1, 1, -1).expand(1, e2_tensor.shape[0], -1)
    e2_state = bank.update_e2(
        bank.reset("video-e2-history", "trajectory-e2-history"),
        _e2_output(
            e2_tensor,
            torch.tensor(e2_phases),
            torch.tensor(e2_times),
            e2_positions,
        ),
        e2_semantics,
        event_kind=E2EventKind.PERIODIC,
    )
    e2_payload = e2_state.records[0].payload
    assert isinstance(e2_payload, E2Payload)
    assert e2_payload.completed_count == event_count
    assert len(e2_payload.completed_intervals) == event_count
    assert len(e2_payload.recent_event_times) == 512
    assert e2_payload.history_eviction_count == 1 and e2_payload.history_truncated
    assert e2_payload.recent_event_times[0] == pytest.approx(1.2)


def test_hard_write_detaches_without_breaking_soft_semantic_or_observation_gradients(
    bank: StructuredStateBank,
) -> None:
    bank.zero_grad(set_to_none=True)
    source = torch.randn(1, HIDDEN_DIM, requires_grad=True)
    soft_semantic = bank.project(source, HeadType.O1)[0]
    probabilities = torch.tensor(
        [[[0.9, 0.9, 0.9, 0.1, 0.1, 0.9]]],
        requires_grad=True,
    )
    observation = _o1_output(probabilities, timestamp=0.0, position_id=0)
    state = bank.update_o1(
        bank.reset("video-grad", "trajectory-grad"),
        observation,
        soft_semantic,
        observation_timestamp=0.0,
        observation_position_id=0,
        set_baseline=True,
    )
    stored = state.records[0].semantic_embedding
    assert not stored.requires_grad and stored.grad_fn is None
    assert stored.untyped_storage().data_ptr() != soft_semantic.untyped_storage().data_ptr()
    assert soft_semantic.requires_grad and probabilities.requires_grad

    weights = torch.linspace(0.5, 1.5, SEMANTIC_DIM)
    ((soft_semantic * weights).sum() + observation.probabilities.sum()).backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert probabilities.grad is not None and bool(torch.isfinite(probabilities.grad).all())
    assert float(source.grad.abs().sum()) > 0.0
    assert float(probabilities.grad.abs().sum()) > 0.0
    assert bank.semantic_projector.output_projection.weight.grad is not None
    assert all("record" not in key and "runtime" not in key for key in bank.state_dict())
