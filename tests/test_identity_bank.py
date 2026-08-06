from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from tests.support.runtime_factories import make_state_record
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.identity_bank import (
    CandidateIdentity,
    ConfirmedIdentity,
    IdentityBank,
    IdentityBankRuntimeState,
    IdentityDecisionStatus,
    IdentityUpdateResult,
    _append_confirmed,
    _relevance_ema,
    _update_confirmed,
    build_identity_bank,
)
from ttt_svcbench_qwen.observation_heads import O2SoftOutput
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    StateBankRuntimeState,
    StructuredStateBank,
    build_state_bank,
)

IDENTITY_DIM = 256
SEMANTIC_DIM = 512
NEW = (0.95, 0.05)
MATCH = (0.05, 0.95)
WEAK = (0.1, 0.1)


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
    off_diagonal = (vectors @ vectors.T).masked_fill(torch.eye(count, dtype=torch.bool), -1.0)
    assert float(off_diagonal.max()) < 0.8
    return vectors


def _unit_identity(index: int = 0, *, requires_grad: bool = False) -> Tensor:
    value = torch.zeros(IDENTITY_DIM)
    value[index] = 1.0
    return value.requires_grad_(requires_grad)


def _unit_semantics(offset: int = 0, *, requires_grad: bool = False) -> Tensor:
    values = torch.zeros(1, SEMANTIC_DIM)
    values[0, offset % SEMANTIC_DIM] = 1.0
    return values.requires_grad_(requires_grad)


class _Driver:
    """Thread one owner row through ``IdentityBank.update_row`` without per-call boilerplate."""

    def __init__(
        self,
        banks: tuple[IdentityBank, StructuredStateBank],
        *,
        video_id: str = "video-a",
        trajectory_id: str = "trajectory-a",
    ) -> None:
        self.bank, self.state_bank = banks
        self.video_id = video_id
        self.trajectory_id = trajectory_id
        self.identity = self.bank.reset(video_id, trajectory_id)
        self.state = self.state_bank.reset(video_id, trajectory_id)

    def step(
        self,
        identities: Tensor,
        position_id: int,
        scores: tuple[float, float] = NEW,
        *,
        semantics: Tensor | None = None,
        chunk_index: int | None = None,
        commit: bool = True,
    ) -> IdentityUpdateResult:
        if identities.ndim == 2:
            identities = identities.unsqueeze(0)
        shape = identities.shape[:2]
        probabilities = torch.empty((*shape, 2), dtype=identities.dtype)
        probabilities[..., 0], probabilities[..., 1] = scores
        observation = O2SoftOutput(
            identity=identities,
            score_logits=torch.zeros_like(probabilities),
            score_probabilities=probabilities,
            valid_mask=torch.ones(shape, dtype=torch.bool),
            timestamps=torch.full(shape, float(position_id), dtype=torch.float64),
            position_ids=torch.full(shape, position_id, dtype=torch.int64),
        )
        result = self.bank.update_row(
            self.identity,
            self.state_bank,
            self.state,
            observation,
            _unit_semantics(position_id) if semantics is None else semantics,
            row=0,
            chunk_index=position_id if chunk_index is None else chunk_index,
        )
        if commit:
            self.identity, self.state = result.identity_state, result.state_bank_state
        return result

    def seed_records(
        self,
        payloads: tuple[CandidateIdentity, ...] | tuple[ConfirmedIdentity, ...],
        *,
        confidence: float,
    ) -> None:
        records = tuple(
            make_state_record(
                payload.semantic_record_id or "",
                HeadType.O2,
                replace(payload, identity_prototype=payload.identity_prototype.clone()),
                semantic_embedding=_unit_semantics(index)[0].clone(),
                video_id=self.video_id,
                trajectory_id=self.trajectory_id,
                timestamp=0.0,
                confidence=confidence,
            )
            for index, payload in enumerate(payloads)
        )
        self.state = StateBankRuntimeState(
            video_id=self.video_id,
            trajectory_id=self.trajectory_id,
            records=records,
            audit_log=(),
            issued_record_ids=tuple(record.record_id for record in records),
            next_record_sequence=len(records),
        )

    def seed_candidates(self, prototypes: Tensor, *, confidence: float = 0.95) -> None:
        candidates = tuple(
            CandidateIdentity(
                candidate_id=f"candidate-{index:08d}",
                identity_prototype=prototype.to(dtype=torch.float32).clone(),
                observation_count=1,
                ttl_remaining=8,
                confidence=confidence,
                last_reliable_chunk_index=0,
                reliable_streak=1,
                semantic_record_id=f"record-{index:08d}",
            )
            for index, prototype in enumerate(prototypes)
        )
        self.identity = replace(
            self.identity,
            candidates=candidates,
            next_candidate_sequence=len(candidates),
            issued_candidate_ids=tuple(item.candidate_id for item in candidates),
            last_chunk_index=0,
        )
        self.seed_records(candidates, confidence=confidence)

    def seed_confirmed(self, prototypes: Tensor) -> None:
        confirmed = tuple(
            ConfirmedIdentity(
                identity_id=f"identity-{index:08d}",
                identity_prototype=prototype.to(dtype=torch.float32).clone(),
                first_seen=0.0,
                last_seen=1.0,
                observation_count=2,
                semantic_record_id=f"record-{index:08d}",
                first_seen_position_id=0,
                last_seen_position_id=1,
            )
            for index, prototype in enumerate(prototypes)
        )
        self.identity = replace(
            self.identity,
            confirmed=confirmed,
            next_identity_sequence=len(confirmed),
            issued_identity_ids=tuple(item.identity_id for item in confirmed),
            last_chunk_index=1,
        )
        self.seed_records(confirmed, confidence=0.95)


def _promote_single(
    banks: tuple[IdentityBank, StructuredStateBank],
    prototype: Tensor,
    *,
    video_id: str,
    trajectory_id: str,
) -> IdentityBankRuntimeState:
    driver = _Driver(banks, video_id=video_id, trajectory_id=trajectory_id)
    driver.step(prototype.unsqueeze(0), 0, NEW)
    promoted = driver.step(prototype.unsqueeze(0), 1, MATCH)
    assert promoted.decisions[0].status is IdentityDecisionStatus.PROMOTED
    return promoted.identity_state


def test_full_candidate_store_prunes_low_confidence_before_rejecting_new(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    vectors = _random_unit_vectors(513, seed=20260715)
    driver.seed_candidates(vectors[:512])
    retained_ids = tuple(item.candidate_id for item in driver.identity.candidates)
    assert len(retained_ids) == 512

    rejected = driver.step(vectors[512:], 1, NEW, commit=False)
    assert rejected.decisions[0].status is IdentityDecisionStatus.OVERFLOW_REJECTED
    assert tuple(c.candidate_id for c in rejected.identity_state.candidates) == retained_ids

    low = replace(driver.identity.candidates[0], confidence=0.49)
    low_record_id = low.semantic_record_id
    assert low_record_id is not None
    driver.identity = replace(driver.identity, candidates=(low,) + driver.identity.candidates[1:])
    driver.state = replace(
        driver.state,
        records=tuple(
            replace(
                record,
                confidence=0.49,
                payload=replace(low, identity_prototype=low.identity_prototype.clone()),
            )
            if record.record_id == low_record_id
            else record
            for record in driver.state.records
        ),
    )

    admitted = driver.step(vectors[512:], 1, NEW)
    assert admitted.decisions[0].status is IdentityDecisionStatus.CANDIDATE_CREATED
    assert len(admitted.identity_state.candidates) == 512
    assert admitted.identity_state.candidate_low_confidence_pruned_count == 1
    ids = tuple(item.candidate_id for item in admitted.identity_state.candidates)
    assert low.candidate_id not in ids and "candidate-00000512" in ids
    by_id = {record.record_id: record for record in admitted.state_bank_state.records}
    assert by_id[low_record_id].valid is False


def test_candidate_ttl_expires_at_zero_after_eight_unmatched_positions(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    created = driver.step(_unit_identity(0).unsqueeze(0), 0, NEW)
    assert created.identity_state.candidates[0].ttl_remaining == 8

    for chunk_index in (0, 1):
        replay = driver.step(
            _unit_identity(0).unsqueeze(0), 0, MATCH, chunk_index=chunk_index, commit=False
        )
        assert replay.identity_state is created.identity_state
        assert replay.state_bank_state is created.state_bank_state
        assert replay.decisions[0].status is IdentityDecisionStatus.REPLAY_IGNORED

    for position in range(1, 8):
        aged = driver.step(_unit_identity(1).unsqueeze(0), position, WEAK)
    assert len(aged.identity_state.candidates) == 1
    assert aged.identity_state.candidates[0].ttl_remaining == 1

    expired = driver.step(_unit_identity(1).unsqueeze(0), 8, WEAK)
    assert not expired.identity_state.candidates
    assert expired.identity_state.candidate_expired_count == 1
    tombstones = [
        record
        for record in expired.state_bank_state.records
        if isinstance(record.payload, CandidateIdentity)
    ]
    assert len(tombstones) == 1 and tombstones[0].valid is False


def test_two_distinct_reliable_positions_promote_candidate_and_link_records(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    prototype = _unit_identity(0).unsqueeze(0)
    candidate = driver.step(prototype, 0, NEW)
    assert candidate.decisions[0].status is IdentityDecisionStatus.CANDIDATE_CREATED
    candidate_id = candidate.identity_state.candidates[0].candidate_id
    candidate_record_id = candidate.identity_state.candidates[0].semantic_record_id
    assert candidate_record_id is not None

    promoted = driver.step(prototype, 1, MATCH)
    assert promoted.decisions[0].status is IdentityDecisionStatus.PROMOTED
    assert not promoted.identity_state.candidates
    assert promoted.identity_state.unique_count == 1
    confirmed = promoted.identity_state.confirmed[0]
    assert confirmed.observation_count == 2
    assert confirmed.first_seen_position_id == 0
    assert confirmed.last_seen_position_id == 1
    assert confirmed.semantic_record_id not in (None, candidate_record_id)
    assert candidate_id in promoted.identity_state.issued_candidate_ids

    by_id = {record.record_id: record for record in promoted.state_bank_state.records}
    assert by_id[candidate_record_id].valid is False
    assert isinstance(by_id[candidate_record_id].payload, CandidateIdentity)
    assert confirmed.semantic_record_id is not None
    assert by_id[confirmed.semantic_record_id].valid is True
    assert isinstance(by_id[confirmed.semantic_record_id].payload, ConfirmedIdentity)

    view = driver.state_bank.view((promoted.state_bank_state,), head_type=HeadType.O2)
    payloads = tuple(type(r.payload) for r in view.cloned_records[0] if r is not None)
    assert payloads == (CandidateIdentity, ConfirmedIdentity)
    assert view.retrieval_eligible_mask.tolist() == [[False, True]]


def test_same_identity_one_hundred_times_counts_once_and_updates_existing_record(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    prototype = _unit_identity(0).unsqueeze(0)
    statuses = [driver.step(prototype, 0, NEW).decisions[0].status]
    for position in range(1, 100):
        statuses.append(driver.step(prototype, position, MATCH).decisions[0].status)
    assert statuses.count(IdentityDecisionStatus.CANDIDATE_CREATED) == 1
    assert statuses.count(IdentityDecisionStatus.PROMOTED) == 1
    assert statuses.count(IdentityDecisionStatus.CONFIRMED_UPDATED) == 98
    assert driver.identity.unique_count == 1
    assert not driver.identity.candidates
    confirmed = driver.identity.confirmed[0]
    assert confirmed.observation_count == 100
    assert (confirmed.first_seen, confirmed.last_seen) == (0.0, 99.0)
    assert confirmed.prototype_version == 98
    valid_confirmed = [
        record
        for record in driver.state.records
        if record.valid and isinstance(record.payload, ConfirmedIdentity)
    ]
    assert len(valid_confirmed) == 1


def test_confirmed_prototype_uses_normalized_ema(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    initial = _unit_identity(0)
    driver.step(initial.unsqueeze(0), 0, NEW)
    driver.step(initial.unsqueeze(0), 1, MATCH)
    observation = F.normalize(
        0.9 * _unit_identity(0) + (1.0 - 0.9**2) ** 0.5 * _unit_identity(1), dim=0
    )
    updated = driver.step(observation.unsqueeze(0), 2, MATCH)
    assert updated.decisions[0].status is IdentityDecisionStatus.CONFIRMED_UPDATED
    expected = F.normalize(0.9 * initial + 0.1 * observation, dim=0)
    confirmed = updated.identity_state.confirmed[0]
    torch.testing.assert_close(confirmed.identity_prototype, expected, rtol=1.0e-6, atol=1.0e-6)
    assert confirmed.prototype_version == 1


def test_exact_near_tie_conflict_is_fail_closed_and_updates_neither_identity(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    residual = (1.0 - 0.9**2) ** 0.5
    left = F.normalize(0.9 * _unit_identity(0) + residual * _unit_identity(1), dim=0)
    right = F.normalize(0.9 * _unit_identity(0) - residual * _unit_identity(1), dim=0)
    driver.seed_confirmed(torch.stack((left, right)))
    before_state = driver.state
    before = tuple(item.observation_count for item in driver.identity.confirmed)

    conflict = driver.step(_unit_identity(0).unsqueeze(0), 2, MATCH, chunk_index=2)
    assert conflict.decisions[0].status is IdentityDecisionStatus.MATCH_CONFLICT
    assert tuple(i.observation_count for i in conflict.identity_state.confirmed) == before
    assert conflict.state_bank_state is before_state


def test_score_threshold_boundaries_are_inclusive(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    prototype = _unit_identity(0).unsqueeze(0)
    candidate = driver.step(prototype, 0, (0.5, 0.49))
    assert candidate.decisions[0].status is IdentityDecisionStatus.CANDIDATE_CREATED
    assert candidate.identity_state.candidates[0].confidence == pytest.approx(0.5)
    promoted = driver.step(prototype, 1, (0.49, 0.5))
    assert promoted.decisions[0].status is IdentityDecisionStatus.PROMOTED


def test_bank_state_is_isolated_per_video_and_release_clears_only_its_owner(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    bank, _ = banks
    left = _promote_single(
        banks, _unit_identity(0), video_id="video-left", trajectory_id="trajectory-left"
    )
    right = _promote_single(
        banks, _unit_identity(1), video_id="video-right", trajectory_id="trajectory-right"
    )
    assert (left.video_id, right.video_id) == ("video-left", "video-right")
    assert len(left.confirmed) == 1 and len(right.confirmed) == 1
    torch.testing.assert_close(left.confirmed[0].identity_prototype, _unit_identity(0))
    torch.testing.assert_close(right.confirmed[0].identity_prototype, _unit_identity(1))
    storages = [
        {i.identity_prototype.untyped_storage().data_ptr() for i in state.confirmed}
        for state in (left, right)
    ]
    assert not (storages[0] & storages[1])

    released = bank.release(left)
    assert released.released and released.video_id == "video-left"
    assert not released.confirmed and not released.candidates
    assert len(right.confirmed) == 1 and not right.released


def test_hard_writes_detach_clone_and_do_not_break_soft_gradients_or_state_dict(
    banks: tuple[IdentityBank, StructuredStateBank],
) -> None:
    driver = _Driver(banks)
    identity_leaf = _unit_identity(0, requires_grad=True)
    semantic_leaf = _unit_semantics(requires_grad=True)
    before_keys = tuple(driver.state_bank.state_dict())
    result = driver.step(identity_leaf.unsqueeze(0), 0, NEW, semantics=semantic_leaf)
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
    (identity_leaf.square().sum() + semantic_leaf.square().sum()).backward()
    assert identity_leaf.grad is not None and torch.isfinite(identity_leaf.grad).all()
    assert semantic_leaf.grad is not None and torch.isfinite(semantic_leaf.grad).all()
    assert tuple(driver.state_bank.state_dict()) == before_keys
    assert not isinstance(driver.bank, torch.nn.Module)


def test_relevance_survives_confirmed_storage_roundtrip() -> None:
    base = ConfirmedIdentity(
        identity_id="identity-relevance",
        identity_prototype=_unit_identity(3),
        first_seen=0.0,
        last_seen=1.0,
        observation_count=2,
        semantic_record_id="record-relevance",
        first_seen_position_id=0,
        last_seen_position_id=1,
        relevance=0.7,
    )
    store = _append_confirmed((), base)
    assert store[0].relevance == pytest.approx(0.7)

    store = _update_confirmed(
        store,
        replace(base, last_seen=2.0, observation_count=3, prototype_version=1, relevance=0.9),
    )
    assert len(store) == 1
    assert store[0].relevance == pytest.approx(0.9)
    assert store[0].prototype_version == 1

    assert _relevance_ema(0.5, 0.9, 0.9) == pytest.approx(0.54)
    assert _relevance_ema(1.0, 1.0, 0.9) == 1.0
    assert _relevance_ema(0.0, 0.0, 0.9) == 0.0
