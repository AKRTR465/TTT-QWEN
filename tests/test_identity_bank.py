from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

import ttt_svcbench_qwen.identity_bank as identity_bank_types
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.identity_bank import (
    CandidateIdentity,
    ConfirmedChunk,
    ConfirmedIdentity,
    ExactMatchStatus,
    HotCacheEntry,
    IdentityBank,
    IdentityBankRuntimeState,
    IdentityDecisionStatus,
    IdentityUpdateResult,
    build_identity_bank,
    evaluate_identity_quality,
)
from ttt_svcbench_qwen.observation_heads import O2SoftOutput
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    StateBankRuntimeState,
    StateRecord,
    StateRecordKind,
    StructuredStateBank,
    build_state_bank,
)

IDENTITY_DIM = 256
SEMANTIC_DIM = 512


@pytest.fixture
def banks() -> tuple[IdentityBank, StructuredStateBank]:
    config = load_config()
    return build_identity_bank(config), build_state_bank(config)


def _random_unit_vectors(count: int, *, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    vectors = F.normalize(
        torch.randn(count, IDENTITY_DIM, generator=generator, dtype=torch.float32),
        dim=-1,
    )
    if count > 1:
        off_diagonal = (vectors @ vectors.T).masked_fill(
            torch.eye(count, dtype=torch.bool),
            -1.0,
        )
        assert float(off_diagonal.max()) < 0.8
    return vectors


def _unit_identity(index: int = 0, *, requires_grad: bool = False) -> Tensor:
    value = torch.zeros(IDENTITY_DIM)
    value[index] = 1.0
    return value.requires_grad_(requires_grad)


def _unit_semantics(
    count: int,
    *,
    offset: int = 0,
    requires_grad: bool = False,
) -> Tensor:
    values = torch.zeros(count, SEMANTIC_DIM)
    rows = torch.arange(count)
    values[rows, (rows + offset) % SEMANTIC_DIM] = 1.0
    return values.requires_grad_(requires_grad)


def _o2_output(
    identities: Tensor,
    *,
    novelty: float,
    match_confidence: float,
    position_id: int,
    timestamp: float | None = None,
    valid_mask: Tensor | None = None,
) -> O2SoftOutput:
    if identities.ndim == 2:
        identities = identities.unsqueeze(0)
    batch_size, slot_count = identities.shape[:2]
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, slot_count, dtype=torch.bool)
    elif valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    probabilities = torch.empty(batch_size, slot_count, 2, dtype=identities.dtype)
    probabilities[..., 0] = novelty
    probabilities[..., 1] = match_confidence
    probabilities = probabilities.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    logits = torch.zeros_like(probabilities)
    resolved_timestamp = float(position_id) if timestamp is None else timestamp
    timestamps = torch.full(
        (batch_size, slot_count),
        resolved_timestamp,
        dtype=torch.float64,
    )
    timestamps = torch.where(valid_mask, timestamps, torch.full_like(timestamps, -1.0))
    position_ids = torch.full(
        (batch_size, slot_count),
        position_id,
        dtype=torch.int64,
    )
    position_ids = torch.where(valid_mask, position_ids, torch.full_like(position_ids, -1))
    safe_identities = identities.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    return O2SoftOutput(
        identity=safe_identities,
        score_logits=logits,
        score_probabilities=probabilities,
        valid_mask=valid_mask,
        timestamps=timestamps,
        position_ids=position_ids,
    )


def _update(
    bank: IdentityBank,
    identity_state: IdentityBankRuntimeState,
    state_bank: StructuredStateBank,
    state_state: StateBankRuntimeState,
    identities: Tensor,
    semantics: Tensor,
    *,
    novelty: float,
    match_confidence: float,
    position_id: int,
    chunk_index: int | None = None,
) -> IdentityUpdateResult:
    observation = _o2_output(
        identities,
        novelty=novelty,
        match_confidence=match_confidence,
        position_id=position_id,
    )
    return bank.update_row(
        identity_state,
        state_bank,
        state_state,
        observation,
        semantics,
        row=0,
        chunk_index=position_id if chunk_index is None else chunk_index,
    )


def _reset_pair(
    bank: IdentityBank,
    state_bank: StructuredStateBank,
    *,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState]:
    return (
        bank.reset(video_id, trajectory_id, hot_cache_enabled=False),
        state_bank.reset(video_id, trajectory_id),
    )


def _linked_candidate(
    index: int,
    prototype: Tensor,
    *,
    confidence: float = 0.95,
    ttl_remaining: int = 8,
    position_id: int = 0,
    chunk_index: int = 0,
) -> CandidateIdentity:
    return CandidateIdentity(
        candidate_id=f"candidate-{index:08d}",
        identity_prototype=prototype.detach().to(dtype=torch.float32, device="cpu").clone(),
        observation_count=1,
        ttl_remaining=ttl_remaining,
        confidence=confidence,
        first_seen=float(position_id),
        last_seen=float(position_id),
        first_seen_position_id=position_id,
        last_seen_position_id=position_id,
        last_reliable_chunk_index=chunk_index,
        reliable_streak=1,
        semantic_record_id=f"record-{index:08d}",
    )


def _candidate_runtime(
    bank: IdentityBank,
    state_bank: StructuredStateBank,
    prototypes: Tensor,
    *,
    confidence: float = 0.95,
    ttl_remaining: int = 8,
    candidate_capacity: int | None = None,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState]:
    count = int(prototypes.shape[0])
    if candidate_capacity is None:
        candidate_capacity = max(64, ((count + 63) // 64) * 64)
    candidates = tuple(
        _linked_candidate(
            index,
            prototype,
            confidence=confidence,
            ttl_remaining=ttl_remaining,
        )
        for index, prototype in enumerate(prototypes)
    )
    records = tuple(
        StateRecord(
            record_id=candidate.semantic_record_id or "",
            video_id=video_id,
            trajectory_id=trajectory_id,
            head_type=HeadType.O2,
            semantic_embedding=_unit_semantics(1, offset=index)[0].clone(),
            timestamp=candidate.first_seen,
            time_range=None,
            valid=True,
            confidence=candidate.confidence,
            payload=replace(
                candidate,
                identity_prototype=candidate.identity_prototype.detach().clone(),
            ),
        )
        for index, candidate in enumerate(candidates)
    )
    base = bank.reset(video_id, trajectory_id, hot_cache_enabled=False)
    identity_state = replace(
        base,
        candidates=candidates,
        candidate_capacity=candidate_capacity,
        next_candidate_sequence=count,
        issued_candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        last_chunk_index=0,
    )
    state_state = StateBankRuntimeState(
        video_id=video_id,
        trajectory_id=trajectory_id,
        records=records,
        audit_log=(),
        issued_record_ids=tuple(record.record_id for record in records),
        next_record_sequence=count,
    )
    return identity_state, state_state


def _confirmed_chunk(
    prototypes: Tensor,
    *,
    id_offset: int,
    capacity: int = 256,
) -> ConfirmedChunk:
    size = int(prototypes.shape[0])
    assert 0 <= size <= capacity
    values = torch.zeros(capacity, IDENTITY_DIM, dtype=torch.float32)
    values[:size] = prototypes.detach().to(dtype=torch.float32, device="cpu")
    occupied = torch.zeros(capacity, dtype=torch.bool)
    occupied[:size] = True
    identity_ids = tuple(
        f"identity-{id_offset + index:08d}" if index < size else None for index in range(capacity)
    )
    record_ids = tuple(
        f"record-{id_offset + index:08d}" if index < size else None for index in range(capacity)
    )
    first_seen = torch.full((capacity,), -1.0, dtype=torch.float64)
    last_seen = torch.full((capacity,), -1.0, dtype=torch.float64)
    first_seen[:size] = 0.0
    last_seen[:size] = 1.0
    counts = torch.zeros(capacity, dtype=torch.int64)
    counts[:size] = 2
    first_positions = torch.full((capacity,), -1, dtype=torch.int64)
    last_positions = torch.full((capacity,), -1, dtype=torch.int64)
    first_positions[:size] = 0
    last_positions[:size] = 1
    return ConfirmedChunk(
        prototypes=values,
        occupied=occupied,
        identity_ids=identity_ids,
        first_seen=first_seen,
        last_seen=last_seen,
        observation_counts=counts,
        first_seen_position_ids=first_positions,
        last_seen_position_ids=last_positions,
        semantic_record_ids=record_ids,
        prototype_versions=torch.zeros(capacity, dtype=torch.int64),
    )


def _confirmed_runtime(
    bank: IdentityBank,
    prototypes: Tensor,
    *,
    hot_cache_enabled: bool = False,
    cached_indices: tuple[int, ...] = (),
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
) -> IdentityBankRuntimeState:
    chunks = tuple(
        _confirmed_chunk(prototypes[start : start + 256], id_offset=start)
        for start in range(0, int(prototypes.shape[0]), 256)
    )
    base = bank.reset(
        video_id,
        trajectory_id,
        hot_cache_enabled=hot_cache_enabled,
        hot_device="cpu" if hot_cache_enabled else None,
    )
    cache = tuple(
        HotCacheEntry(
            identity_id=f"identity-{index:08d}",
            identity_prototype=prototypes[index]
            .detach()
            .to(dtype=torch.bfloat16, device="cpu")
            .clone(),
            last_accessed_position_id=index + 1,
        )
        for index in cached_indices
    )
    count = int(prototypes.shape[0])
    return replace(
        base,
        confirmed_chunks=chunks,
        hot_cache=cache,
        next_identity_sequence=count,
        issued_identity_ids=tuple(f"identity-{index:08d}" for index in range(count)),
        last_chunk_index=1,
    )


def _confirmed_state_pair(
    bank: IdentityBank,
    prototypes: Tensor,
) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState]:
    identity_state = _confirmed_runtime(bank, prototypes)
    records = tuple(
        StateRecord(
            record_id=f"record-{index:08d}",
            video_id=identity_state.video_id,
            trajectory_id=identity_state.trajectory_id,
            head_type=HeadType.O2,
            semantic_embedding=_unit_semantics(1, offset=index)[0].clone(),
            timestamp=0.0,
            time_range=None,
            valid=True,
            confidence=0.95,
            payload=ConfirmedIdentity(
                identity_id=f"identity-{index:08d}",
                identity_prototype=prototype.detach().to(dtype=torch.float32).clone(),
                first_seen=0.0,
                last_seen=1.0,
                observation_count=2,
                semantic_record_id=f"record-{index:08d}",
                first_seen_position_id=0,
                last_seen_position_id=1,
            ),
        )
        for index, prototype in enumerate(prototypes)
    )
    state_state = StateBankRuntimeState(
        video_id=identity_state.video_id,
        trajectory_id=identity_state.trajectory_id,
        records=records,
        audit_log=(),
        issued_record_ids=tuple(record.record_id for record in records),
        next_record_sequence=len(records),
    )
    return identity_state, state_state


def test_candidate_capacity_grows_64_to_512_and_513th_is_explicit_overflow(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    vectors = _random_unit_vectors(513, seed=20260714)
    first_identity, first_state = _candidate_runtime(
        bank,
        state_bank,
        vectors[:64],
        candidate_capacity=64,
    )
    assert len(first_identity.candidates) == 64
    assert first_identity.candidate_capacity == 64

    second = _update(
        bank,
        first_identity,
        state_bank,
        first_state,
        vectors[64:65],
        _unit_semantics(1, offset=64),
        novelty=0.95,
        match_confidence=0.05,
        position_id=1,
    )
    assert len(second.identity_state.candidates) == 65
    assert second.identity_state.candidate_capacity == 128

    full_identity, full_state = _candidate_runtime(
        bank,
        state_bank,
        vectors[:512],
        candidate_capacity=512,
    )
    assert len(full_identity.candidates) == 512
    assert full_identity.candidate_capacity == 512
    retained_ids = tuple(candidate.candidate_id for candidate in full_identity.candidates)

    overflow = _update(
        bank,
        full_identity,
        state_bank,
        full_state,
        vectors[512:],
        _unit_semantics(1, offset=512),
        novelty=0.95,
        match_confidence=0.05,
        position_id=1,
    )
    assert len(overflow.identity_state.candidates) == 512
    assert overflow.identity_state.candidate_capacity == 512
    assert overflow.identity_state.candidate_overflow_count == 1
    assert tuple(candidate.candidate_id for candidate in overflow.identity_state.candidates) == (
        retained_ids
    )


def test_candidate_ttl_expires_at_zero_after_eight_unmatched_positions(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state, state_state = _reset_pair(bank, state_bank)
    created = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        _unit_identity(0).unsqueeze(0),
        _unit_semantics(1),
        novelty=0.95,
        match_confidence=0.05,
        position_id=0,
    )
    assert created.identity_state.candidates[0].ttl_remaining == 8
    replay = _update(
        bank,
        created.identity_state,
        state_bank,
        created.state_bank_state,
        _unit_identity(0).unsqueeze(0),
        _unit_semantics(1),
        novelty=0.05,
        match_confidence=0.95,
        position_id=0,
        chunk_index=0,
    )
    assert replay.identity_state is created.identity_state
    assert replay.state_bank_state is created.state_bank_state
    assert replay.decisions[0].status is IdentityDecisionStatus.REPLAY_IGNORED
    same_position_new_chunk = _update(
        bank,
        created.identity_state,
        state_bank,
        created.state_bank_state,
        _unit_identity(0).unsqueeze(0),
        _unit_semantics(1),
        novelty=0.05,
        match_confidence=0.95,
        position_id=0,
        chunk_index=1,
    )
    assert same_position_new_chunk.identity_state is created.identity_state
    assert same_position_new_chunk.state_bank_state is created.state_bank_state
    assert same_position_new_chunk.decisions[0].reason == "same_committed_position"
    current = created
    for position in range(1, 8):
        current = _update(
            bank,
            current.identity_state,
            state_bank,
            current.state_bank_state,
            _unit_identity(1).unsqueeze(0),
            _unit_semantics(1, offset=position),
            novelty=0.1,
            match_confidence=0.1,
            position_id=position,
        )
    assert len(current.identity_state.candidates) == 1
    assert current.identity_state.candidates[0].ttl_remaining == 1

    expired = _update(
        bank,
        current.identity_state,
        state_bank,
        current.state_bank_state,
        _unit_identity(1).unsqueeze(0),
        _unit_semantics(1, offset=8),
        novelty=0.1,
        match_confidence=0.1,
        position_id=8,
    )
    assert not expired.identity_state.candidates
    assert expired.identity_state.candidate_expired_count == 1
    candidate_records = [
        record
        for record in expired.state_bank_state.records
        if isinstance(record.payload, identity_bank_types.CandidateIdentity)
    ]
    assert len(candidate_records) == 1
    assert candidate_records[0].valid is False


def test_low_confidence_prune_precedes_full_store_admission(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    vectors = _random_unit_vectors(513, seed=20260715)
    identity_state, state_state = _candidate_runtime(
        bank,
        state_bank,
        vectors[:512],
        candidate_capacity=512,
    )
    low = replace(identity_state.candidates[0], confidence=0.49)
    identity_state = replace(
        identity_state,
        candidates=(low,) + identity_state.candidates[1:],
    )
    low_record_id = low.semantic_record_id
    records = tuple(
        replace(
            record,
            confidence=0.49,
            payload=replace(
                low,
                identity_prototype=low.identity_prototype.detach().clone(),
            ),
        )
        if record.record_id == low_record_id
        else record
        for record in state_state.records
    )
    state_state = replace(state_state, records=records)

    admitted = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        vectors[512:],
        _unit_semantics(1, offset=512),
        novelty=0.95,
        match_confidence=0.05,
        position_id=1,
    )
    assert admitted.decisions[0].status is IdentityDecisionStatus.CANDIDATE_CREATED
    assert len(admitted.identity_state.candidates) == 512
    assert admitted.identity_state.candidate_low_confidence_pruned_count == 1
    assert admitted.identity_state.candidate_overflow_count == 0
    assert all(
        candidate.candidate_id != low.candidate_id
        for candidate in admitted.identity_state.candidates
    )
    assert any(
        candidate.candidate_id == "candidate-00000512"
        for candidate in admitted.identity_state.candidates
    )
    by_id = {record.record_id: record for record in admitted.state_bank_state.records}
    assert low_record_id is not None
    assert by_id[low_record_id].valid is False
    actions = tuple(entry.action for entry in admitted.identity_state.audit_log)
    assert actions.index("candidate_low_confidence_prune") < actions.index("candidate_created")


def test_two_distinct_reliable_positions_promote_candidate_and_link_records(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state, state_state = _reset_pair(bank, state_bank)
    prototype = _unit_identity(0).unsqueeze(0)
    semantic = _unit_semantics(1)
    candidate = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        prototype,
        semantic,
        novelty=0.95,
        match_confidence=0.05,
        position_id=0,
    )
    candidate_id = candidate.identity_state.candidates[0].candidate_id
    candidate_record_id = candidate.identity_state.candidates[0].semantic_record_id

    promoted = _update(
        bank,
        candidate.identity_state,
        state_bank,
        candidate.state_bank_state,
        prototype,
        semantic,
        novelty=0.05,
        match_confidence=0.95,
        position_id=1,
    )
    assert not promoted.identity_state.candidates
    assert promoted.identity_state.unique_count == 1
    assert len(promoted.identity_state.confirmed) == 1
    confirmed = promoted.identity_state.confirmed[0]
    assert confirmed.observation_count == 2
    assert confirmed.first_seen_position_id == 0
    assert confirmed.last_seen_position_id == 1
    assert confirmed.semantic_record_id != candidate_record_id
    assert candidate_id in promoted.identity_state.issued_candidate_ids

    by_id = {record.record_id: record for record in promoted.state_bank_state.records}
    assert by_id[candidate_record_id].valid is False
    assert isinstance(by_id[candidate_record_id].payload, identity_bank_types.CandidateIdentity)
    assert by_id[confirmed.semantic_record_id].valid is True
    assert isinstance(
        by_id[confirmed.semantic_record_id].payload,
        identity_bank_types.ConfirmedIdentity,
    )
    view = state_bank.view((promoted.state_bank_state,), head_type=HeadType.O2)
    kinds = tuple(kind for kind in view.record_kinds[0] if kind is not None)
    assert kinds == (StateRecordKind.O2_CANDIDATE, StateRecordKind.O2_CONFIRMED)
    assert view.retrieval_eligible_mask.tolist() == [[False, True]]


def test_257th_confirmed_allocates_second_cpu_chunk_without_losing_first_256(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    prototypes = _random_unit_vectors(257, seed=20260716)
    identity_state, state_state = _confirmed_state_pair(bank, prototypes[:256])
    before = {
        identity.identity_id: identity.identity_prototype.clone()
        for identity in identity_state.confirmed
    }
    candidate = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        prototypes[256:].clone(),
        _unit_semantics(1, offset=256),
        novelty=0.95,
        match_confidence=0.05,
        position_id=2,
        chunk_index=2,
    )
    assert candidate.decisions[0].scanned_confirmed_count == 256
    promoted = _update(
        bank,
        candidate.identity_state,
        state_bank,
        candidate.state_bank_state,
        prototypes[256:].clone(),
        _unit_semantics(1, offset=256),
        novelty=0.05,
        match_confidence=0.95,
        position_id=3,
        chunk_index=3,
    )
    assert promoted.decisions[0].status is IdentityDecisionStatus.PROMOTED
    assert promoted.identity_state.unique_count == 257
    assert promoted.identity_state.confirmed_capacity == 512
    assert len(promoted.identity_state.confirmed_chunks) == 2
    assert all(
        chunk.prototypes.device.type == "cpu" and chunk.prototypes.dtype == torch.float32
        for chunk in promoted.identity_state.confirmed_chunks
    )
    for identity_id, expected in before.items():
        torch.testing.assert_close(
            bank.confirmed_by_id(promoted.identity_state, identity_id).identity_prototype,
            expected,
            rtol=0.0,
            atol=0.0,
        )
    newest = bank.confirmed_by_id(promoted.identity_state, "identity-00000256")
    assert newest.first_seen == 2.0
    assert newest.last_seen == 3.0
    assert newest.observation_count == 2


def test_same_identity_one_hundred_times_counts_once_and_updates_existing_record(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state, state_state = _reset_pair(bank, state_bank)
    prototype = _unit_identity(0).unsqueeze(0)
    semantic = _unit_semantics(1)
    result = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        prototype,
        semantic,
        novelty=0.95,
        match_confidence=0.05,
        position_id=0,
    )
    statuses = [result.decisions[0].status]
    for position in range(1, 100):
        result = _update(
            bank,
            result.identity_state,
            state_bank,
            result.state_bank_state,
            prototype,
            semantic,
            novelty=0.05,
            match_confidence=0.95,
            position_id=position,
        )
        statuses.append(result.decisions[0].status)
    assert statuses.count(IdentityDecisionStatus.CANDIDATE_CREATED) == 1
    assert statuses.count(IdentityDecisionStatus.PROMOTED) == 1
    assert statuses.count(IdentityDecisionStatus.CONFIRMED_UPDATED) == 98
    assert result.identity_state.unique_count == 1
    assert not result.identity_state.candidates
    confirmed = result.identity_state.confirmed[0]
    assert confirmed.observation_count == 100
    assert confirmed.first_seen == 0.0
    assert confirmed.last_seen == 99.0
    assert confirmed.prototype_version == 98
    valid_confirmed_records = [
        record
        for record in result.state_bank_state.records
        if record.valid and isinstance(record.payload, ConfirmedIdentity)
    ]
    assert len(valid_confirmed_records) == 1


def test_confirmed_prototype_uses_normalized_ema_and_refreshes_cache_copy(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state = bank.reset(
        "video-a",
        "trajectory-a",
        hot_cache_enabled=True,
        hot_device="cpu",
    )
    state_state = state_bank.reset("video-a", "trajectory-a")
    initial = _unit_identity(0)
    semantic = _unit_semantics(1)
    first = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        initial.unsqueeze(0),
        semantic,
        novelty=0.95,
        match_confidence=0.05,
        position_id=0,
    )
    promoted = _update(
        bank,
        first.identity_state,
        state_bank,
        first.state_bank_state,
        initial.unsqueeze(0),
        semantic,
        novelty=0.05,
        match_confidence=0.95,
        position_id=1,
    )
    observation = F.normalize(
        0.9 * _unit_identity(0) + (1.0 - 0.9**2) ** 0.5 * _unit_identity(1), dim=0
    )
    updated = _update(
        bank,
        promoted.identity_state,
        state_bank,
        promoted.state_bank_state,
        observation.unsqueeze(0),
        semantic,
        novelty=0.05,
        match_confidence=0.95,
        position_id=2,
    )
    expected = F.normalize(0.9 * initial + 0.1 * observation, dim=0)
    confirmed = updated.identity_state.confirmed[0]
    torch.testing.assert_close(confirmed.identity_prototype, expected, rtol=1.0e-6, atol=1.0e-6)
    assert confirmed.prototype_version == 1
    cache = updated.identity_state.hot_cache
    assert len(cache) == 1 and cache[0].prototype_version == 1
    torch.testing.assert_close(
        cache[0].identity_prototype.float(),
        expected,
        rtol=5.0e-3,
        atol=5.0e-3,
    )
    assert cache[0].identity_prototype.untyped_storage().data_ptr() != (
        updated.identity_state.confirmed_chunks[0].prototypes.untyped_storage().data_ptr()
    )
    details = dict(updated.identity_state.audit_log[-2].details)
    assert details["old_prototype_checksum"] != details["new_prototype_checksum"]


def test_exact_near_tie_conflict_is_fail_closed_and_updates_neither_identity(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    residual = (1.0 - 0.9**2) ** 0.5
    left = F.normalize(0.9 * _unit_identity(0) + residual * _unit_identity(1), dim=0)
    right = F.normalize(0.9 * _unit_identity(0) - residual * _unit_identity(1), dim=0)
    identity_state, state_state = _confirmed_state_pair(bank, torch.stack((left, right)))
    before = tuple(identity.observation_count for identity in identity_state.confirmed)
    conflict = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        _unit_identity(0).unsqueeze(0),
        _unit_semantics(1),
        novelty=0.05,
        match_confidence=0.95,
        position_id=2,
        chunk_index=2,
    )
    assert conflict.decisions[0].status is IdentityDecisionStatus.MATCH_CONFLICT
    assert conflict.decisions[0].reason == "confirmed_near_tie"
    assert conflict.decisions[0].scanned_confirmed_count == 2
    assert conflict.identity_state.match_conflict_count == 1
    assert (
        tuple(identity.observation_count for identity in conflict.identity_state.confirmed)
        == before
    )
    assert conflict.state_bank_state is state_state


def test_threshold_boundaries_are_inclusive_for_scores_and_cosine(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state, state_state = _reset_pair(bank, state_bank)
    prototype = _unit_identity(0).unsqueeze(0)
    candidate = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        prototype,
        _unit_semantics(1),
        novelty=0.5,
        match_confidence=0.49,
        position_id=0,
    )
    assert candidate.decisions[0].status is IdentityDecisionStatus.CANDIDATE_CREATED
    assert candidate.identity_state.candidates[0].confidence == pytest.approx(0.5)
    promoted = _update(
        bank,
        candidate.identity_state,
        state_bank,
        candidate.state_bank_state,
        prototype,
        _unit_semantics(1),
        novelty=0.49,
        match_confidence=0.5,
        position_id=1,
    )
    assert promoted.decisions[0].status is IdentityDecisionStatus.PROMOTED

    exact_boundary = F.normalize(0.8 * _unit_identity(0) + 0.6 * _unit_identity(1), dim=0)
    match = bank.exact_match(
        promoted.identity_state,
        exact_boundary,
        access_position_id=2,
        use_hot_cache=False,
    )
    assert match.matches[0].status is ExactMatchStatus.MATCHED
    assert match.matches[0].score == pytest.approx(0.8, abs=1.0e-6)


def test_hot_cache_hit_never_masks_better_full_cpu_exact_match(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, _ = banks
    query = _unit_identity(0)
    true_cpu = F.normalize(
        0.99 * query + (1.0 - 0.99**2) ** 0.5 * _unit_identity(1),
        dim=0,
    )
    cached_but_worse = F.normalize(
        0.90 * query + (1.0 - 0.90**2) ** 0.5 * _unit_identity(2),
        dim=0,
    )
    state = _confirmed_runtime(
        bank,
        torch.stack((true_cpu, cached_but_worse)),
        hot_cache_enabled=True,
        cached_indices=(1,),
    )

    cache_on = bank.exact_match(state, query, access_position_id=10, use_hot_cache=True)
    cache_off = bank.exact_match(state, query, access_position_id=10, use_hot_cache=False)
    for result in (cache_on, cache_off):
        decision = result.matches[0]
        assert decision.status is ExactMatchStatus.MATCHED
        assert decision.identity_id == "identity-00000000"
        assert decision.score == pytest.approx(0.99, abs=1.0e-6)
        assert decision.scanned_confirmed_count == 2
    assert cache_on.matches[0].cache_hit is False
    assert cache_off.matches[0].cache_hit is False
    assert {entry.identity_id for entry in cache_on.state.hot_cache} == {
        "identity-00000000",
        "identity-00000001",
    }
    assert tuple(entry.identity_id for entry in cache_off.state.hot_cache) == ("identity-00000001",)


def test_hot_cache_miss_warms_from_cpu_and_lru_eviction_preserves_truth(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, _ = banks
    prototypes = _random_unit_vectors(257, seed=20260718)
    state = _confirmed_runtime(
        bank,
        prototypes,
        hot_cache_enabled=True,
        cached_indices=tuple(range(1, 257)),
    )
    assert state.unique_count == 257
    assert len(state.hot_cache) == 256
    result = bank.exact_match(
        state,
        prototypes[0],
        access_position_id=300,
        use_hot_cache=True,
    )
    decision = result.matches[0]
    assert decision.status is ExactMatchStatus.MATCHED
    assert decision.identity_id == "identity-00000000"
    assert decision.cache_hit is False
    assert decision.scanned_confirmed_count == 257
    cache_ids = {entry.identity_id for entry in result.state.hot_cache}
    assert len(cache_ids) == 256
    assert "identity-00000000" in cache_ids
    assert "identity-00000001" not in cache_ids
    assert result.state.unique_count == 257
    torch.testing.assert_close(
        bank.confirmed_by_id(result.state, "identity-00000001").identity_prototype,
        prototypes[1],
        rtol=0.0,
        atol=0.0,
    )


def test_owner_storage_snapshot_clear_and_release_are_isolated_and_fail_closed(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    prototype = _unit_identity(0).unsqueeze(0)
    left, left_state = _candidate_runtime(
        bank,
        state_bank,
        prototype,
        video_id="video-left",
        trajectory_id="trajectory-left",
    )
    right, right_state = _candidate_runtime(
        bank,
        state_bank,
        prototype,
        video_id="video-right",
        trajectory_id="trajectory-right",
    )
    assert left.candidates[0].identity_prototype.untyped_storage().data_ptr() != (
        right.candidates[0].identity_prototype.untyped_storage().data_ptr()
    )
    snapshot = bank.snapshot(left)
    restored = bank.restore(snapshot)
    assert restored.video_id == left.video_id
    assert restored.trajectory_id == left.trajectory_id
    assert restored.candidates[0].identity_prototype.untyped_storage().data_ptr() != (
        left.candidates[0].identity_prototype.untyped_storage().data_ptr()
    )

    updated_left = _update(
        bank,
        left,
        state_bank,
        left_state,
        prototype,
        _unit_semantics(1),
        novelty=0.05,
        match_confidence=0.95,
        position_id=1,
    )
    assert updated_left.identity_state.unique_count == 1
    assert right.unique_count == 0 and len(right.candidates) == 1
    with pytest.raises(ValueError, match="owner identifiers"):
        _update(
            bank,
            left,
            state_bank,
            right_state,
            prototype,
            _unit_semantics(1),
            novelty=0.05,
            match_confidence=0.95,
            position_id=1,
        )

    cleared = bank.clear(updated_left.identity_state)
    assert not cleared.candidates and cleared.unique_count == 0
    assert cleared.candidate_capacity == 64 and cleared.confirmed_capacity == 256
    assert cleared.issued_candidate_ids == updated_left.identity_state.issued_candidate_ids
    released = bank.release(updated_left.identity_state)
    assert released.released
    assert not released.candidates and not released.confirmed_chunks and not released.hot_cache
    assert released.candidate_capacity == 0 and released.hot_cache_capacity == 0
    for operation in (
        lambda: bank.snapshot(released),
        lambda: bank.clear(released),
        lambda: bank.exact_match(released, prototype[0], access_position_id=2),
    ):
        with pytest.raises(ValueError, match="released"):
            operation()


def test_hard_writes_detach_clone_and_do_not_break_soft_gradients_or_state_dict(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, state_bank = banks
    identity_state, state_state = _reset_pair(bank, state_bank)
    identity_leaf = _unit_identity(0, requires_grad=True)
    semantic_leaf = _unit_semantics(1, requires_grad=True)
    before_keys = tuple(state_bank.state_dict())
    result = _update(
        bank,
        identity_state,
        state_bank,
        state_state,
        identity_leaf.unsqueeze(0),
        semantic_leaf,
        novelty=0.95,
        match_confidence=0.05,
        position_id=0,
    )
    stored_candidate = result.identity_state.candidates[0].identity_prototype
    stored_record = result.state_bank_state.records[0]
    for tensor in (
        stored_candidate,
        stored_record.semantic_embedding,
        stored_record.payload.identity_prototype,
    ):
        assert not tensor.requires_grad and tensor.grad_fn is None
    assert (
        stored_candidate.untyped_storage().data_ptr() != identity_leaf.untyped_storage().data_ptr()
    )
    assert stored_record.semantic_embedding.untyped_storage().data_ptr() != (
        semantic_leaf.untyped_storage().data_ptr()
    )
    soft_loss = identity_leaf.square().sum() + semantic_leaf.square().sum()
    soft_loss.backward()
    assert identity_leaf.grad is not None and torch.isfinite(identity_leaf.grad).all()
    assert semantic_leaf.grad is not None and torch.isfinite(semantic_leaf.grad).all()
    assert tuple(state_bank.state_dict()) == before_keys
    assert not isinstance(bank, torch.nn.Module)


def test_offline_duplicate_and_missed_new_metrics_are_explicit_and_label_free_runtime() -> None:
    metrics = evaluate_identity_quality(
        ("A", "A", "B", "C", "C"),
        ("identity-1", "identity-2", None, "identity-3", "identity-3"),
    )
    assert metrics.duplicate_excess_count == 1
    assert metrics.duplicate_denominator == 3
    assert metrics.duplicate_rate == pytest.approx(1.0 / 3.0)
    assert metrics.missed_new_count == 1
    assert metrics.ground_truth_identity_denominator == 3
    assert metrics.missed_new_identity_rate == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="equal length"):
        evaluate_identity_quality(("A",), (None, None))


def test_ann_is_disabled_and_every_decision_reports_full_cpu_scan(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, _ = banks
    assert bank.confirmed_config.exact_search is True
    assert bank.confirmed_config.ann_enabled is False
    prototypes = _random_unit_vectors(17, seed=20260719)
    state = _confirmed_runtime(bank, prototypes)
    result = bank.exact_match(
        state,
        prototypes[[0, 8, 16]],
        access_position_id=2,
        use_hot_cache=False,
    )
    assert result.search_mode == "exact"
    assert result.ann_enabled is False
    assert all(match.scanned_confirmed_count == 17 for match in result.matches)
    assert tuple(match.identity_id for match in result.matches) == (
        "identity-00000000",
        "identity-00000008",
        "identity-00000016",
    )
