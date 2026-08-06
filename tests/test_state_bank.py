from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from tests.support import parameter_count
from tests.support.runtime_factories import make_e1_state, make_e2_state
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E1Payload,
    E2EventKind,
    E2Payload,
    E2Phase,
    HeadType,
    O1Payload,
    SemanticProjector,
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
    return make_e1_state(video_id="observation-video", trajectory_id="observation-trajectory")


def _empty_e2_state() -> E2RuntimeState:
    return make_e2_state(video_id="observation-video", trajectory_id="observation-trajectory")


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


def test_functional_append_update_invalidate_and_release(bank: StructuredStateBank) -> None:
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
    assert fresh.version == 0 and first.version == 1
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
    assert updated.version == first.version + 1

    invalidated = bank.invalidate_record(
        updated,
        updated.records[0].record_id,
        audit_timestamp=2.5,
        reason="test",
    )
    assert invalidated.records[0].valid is False
    assert updated.records[0].valid is True
    duplicate_invalidation = bank.invalidate_record(
        invalidated,
        invalidated.records[0].record_id,
        audit_timestamp=3.0,
        reason="test-again",
    )
    assert duplicate_invalidation.audit_log[-1].action == "invalidate_duplicate"
    assert duplicate_invalidation.records[0].valid is False

    second = bank.append_record(
        invalidated,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(2),
        timestamp=4.0,
        time_range=None,
        valid=True,
        confidence=0.7,
        payload=_confirmed(first_seen=4.0, last_seen=4.0),
    )
    assert second.records[1].record_id == "record-00000001"
    assert second.issued_record_ids == ("record-00000000", "record-00000001")

    released = bank.release(second)
    assert released.released and not released.records and not released.issued_record_ids
    assert released.version == second.version + 1
    assert second.records and not second.released


def test_state_bank_isolates_records_per_video(bank: StructuredStateBank) -> None:
    """Per-video isolation: one owner's writes never leak into another owner's state."""

    first = bank.append_record(
        bank.reset("video-iso-a", "trajectory-iso-a"),
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(0),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_confirmed("identity-iso-a", first_seen=1.0, last_seen=1.0),
    )
    second = bank.reset("video-iso-b", "trajectory-iso-b")
    second = bank.append_record(
        second,
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(1),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=_confirmed("identity-iso-b", first_seen=2.0, last_seen=2.0),
    )

    assert len(first.records) == len(second.records) == 1
    assert {record.video_id for record in first.records} == {"video-iso-a"}
    assert {record.video_id for record in second.records} == {"video-iso-b"}

    released_first = bank.release(first)
    assert not released_first.records
    assert len(second.records) == 1 and not second.released

    view = bank.view((first, second))
    assert view.video_ids == ("video-iso-a", "video-iso-b")
    assert view.n_state.tolist() == [1, 1]
    assert view.embeddings[0].untyped_storage().data_ptr() != (
        first.records[0].semantic_embedding.untyped_storage().data_ptr()
    )
    assert not torch.allclose(view.embeddings[0, 0], view.embeddings[1, 0])


def test_clone_state_record_isolates_tensor_storage(bank: StructuredStateBank) -> None:
    state = bank.append_record(
        bank.reset("video-clone", "trajectory-clone"),
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(0),
        timestamp=1.0,
        time_range=None,
        valid=True,
        confidence=0.8,
        payload=_candidate("candidate-clone", first_seen=1.0),
    )
    record = state.records[0]
    isolated = clone_state_record(record)
    assert isolated.record_id == record.record_id
    assert isolated.semantic_embedding.untyped_storage().data_ptr() != (
        record.semantic_embedding.untyped_storage().data_ptr()
    )
    isolated.semantic_embedding.zero_()
    assert torch.count_nonzero(record.semantic_embedding) == 1
    assert isinstance(isolated.payload, CandidateIdentity)
    assert isinstance(record.payload, CandidateIdentity)
    assert isolated.payload.identity_prototype.untyped_storage().data_ptr() != (
        record.payload.identity_prototype.untyped_storage().data_ptr()
    )


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


def test_o1_baseline_slot_carry_over_and_position_monotonic_idempotency(
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

    duplicate = bank.update_o1(
        state,
        _o1_output(one_visible, timestamp=2.75, position_id=11),
        _unit_semantic(3),
        observation_timestamp=2.75,
        observation_position_id=11,
        slot_overflow_count=5,
    )
    assert duplicate is state
    assert duplicate.records[0].payload == payload


def test_e1_hysteresis_and_cooldown_with_overlap_ignored(bank: StructuredStateBank) -> None:
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
    assert payload.active is False and payload.armed is False

    overlap = bank.update_e1(
        state,
        _e1_output(probabilities[-2:], timestamps[-2:], positions[-2:]),
        semantics[:, -2:],
        event_kind=E1EventKind.ACTION,
    )
    assert overlap is state


def test_e2_phase_gated_transitions_overlap_and_conflicting_evidence(
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

    overlap = bank.update_e2(
        state,
        _e2_output(events[-2:], phases[-2:], timestamps[-2:], positions[-2:]),
        semantics[:, -2:],
        event_kind=E2EventKind.PERIODIC,
    )
    assert overlap is state

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
    assert conflict_payload.completed_count == 0


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
