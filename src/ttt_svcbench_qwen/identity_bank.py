"""Implement the detached O2 Candidate/Confirmed identity lifecycle.

Inputs: O2 soft observations, semantic embeddings, causal owner metadata, and chunk index.
Outputs: functional CPU-FP32 identity state, linked O2 records, exact decisions, and audit.
Forbidden: q_target retrieval, labels in runtime, ANN, silent overwrite, or cache truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig

if TYPE_CHECKING:
    from ttt_svcbench_qwen.observation_heads import O2SoftOutput
    from ttt_svcbench_qwen.state_bank import (
        StateBankRuntimeState,
        StructuredStateBank,
    )


IDENTITY_DIM = 256
SEMANTIC_DIM = 512


class IdentityDecisionStatus(StrEnum):
    INVALID = "invalid"
    REPLAY_IGNORED = "replay_ignored"
    SIGNAL_CONFLICT = "signal_conflict"
    MATCH_CONFLICT = "match_conflict"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_UPDATED = "candidate_updated"
    PROMOTED = "promoted"
    CONFIRMED_UPDATED = "confirmed_updated"
    OVERFLOW_REJECTED = "overflow_rejected"


class ExactMatchStatus(StrEnum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    candidate_id: str
    identity_prototype: Tensor
    observation_count: int
    ttl_remaining: int
    confidence: float
    first_seen: float = 0.0
    last_seen: float = 0.0
    first_seen_position_id: int = 0
    last_seen_position_id: int = 0
    last_reliable_chunk_index: int = 0
    reliable_streak: int = 1
    semantic_record_id: str | None = None
    relevance: float = 0.5


@dataclass(frozen=True, slots=True)
class ConfirmedIdentity:
    identity_id: str
    identity_prototype: Tensor
    first_seen: float
    last_seen: float
    observation_count: int
    semantic_record_id: str | None = None
    prototype_version: int = 0
    first_seen_position_id: int = 0
    last_seen_position_id: int = 0
    relevance: float = 0.5



@dataclass(frozen=True, slots=True)
class IdentityBankRuntimeState:
    video_id: str = "unowned-video"
    trajectory_id: str = "unowned-trajectory"
    candidates: tuple[CandidateIdentity, ...] = ()
    confirmed: tuple[ConfirmedIdentity, ...] = ()
    next_candidate_sequence: int = 0
    next_identity_sequence: int = 0
    issued_candidate_ids: tuple[str, ...] = ()
    issued_identity_ids: tuple[str, ...] = ()
    candidate_expired_count: int = 0
    candidate_low_confidence_pruned_count: int = 0
    signal_conflict_count: int = 0
    last_chunk_index: int = -1
    last_committed_position_id: int = -1
    released: bool = False
    version: int = 0

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def unique_count(self) -> int:
        return len(self.confirmed)


@dataclass(frozen=True, slots=True)
class IdentityObservationDecision:
    slot_index: int
    position_id: int
    timestamp: float
    status: IdentityDecisionStatus
    candidate_id: str | None = None
    identity_id: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityUpdateResult:
    identity_state: IdentityBankRuntimeState
    state_bank_state: StateBankRuntimeState
    decisions: tuple[IdentityObservationDecision, ...]


@dataclass(frozen=True, slots=True)
class _Match:
    status: ExactMatchStatus
    entry_id: str | None
    score: float | None
    ambiguous_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Observation:
    slot_index: int
    timestamp: float
    position_id: int
    identity: Tensor
    semantic: Tensor
    novelty: float
    match_confidence: float
    confidence: float
    relevance: float


class IdentityBank:
    """Parameter-free functional O2 identity operator."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.o2_config = config.observation_heads.o2
        self.candidate_config = config.state_bank.candidate_store
        self.confirmed_config = config.state_bank.confirmed_store

    def reset(
        self,
        video_id: str,
        trajectory_id: str,
    ) -> IdentityBankRuntimeState:
        return IdentityBankRuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
        )

    def release(self, state: IdentityBankRuntimeState) -> IdentityBankRuntimeState:
        return IdentityBankRuntimeState(
            video_id=state.video_id,
            trajectory_id=state.trajectory_id,
            released=True,
            version=state.version + 1,
        )

    def confirmed_by_id(
        self, state: IdentityBankRuntimeState, identity_id: str
    ) -> ConfirmedIdentity:
        return _clone_confirmed(
            next(
                identity
                for identity in state.confirmed
                if identity.identity_id == identity_id
            )
        )

    def update_row(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observation: O2SoftOutput,
        semantic_embeddings: Tensor,
        *,
        row: int,
        chunk_index: int,
    ) -> IdentityUpdateResult:
        """Commit one owner row exactly once for a monotonically increasing chunk index."""

        observations = self._extract_observations(observation, semantic_embeddings, row)
        committed_positions = {item.position_id for item in observations}
        committed_position = next(iter(committed_positions), None)
        same_position_replay = (
            committed_position is not None
            and committed_position == identity_state.last_committed_position_id
        )
        if chunk_index == identity_state.last_chunk_index or same_position_replay:
            replay_decisions = tuple(
                IdentityObservationDecision(
                    slot_index=item.slot_index,
                    position_id=item.position_id,
                    timestamp=item.timestamp,
                    status=IdentityDecisionStatus.REPLAY_IGNORED,
                )
                for item in observations
            )
            return IdentityUpdateResult(identity_state, state_state, replay_decisions)

        next_identity = identity_state
        next_state_bank = state_state
        decisions: dict[int, IdentityObservationDecision] = {}
        eligible: list[_Observation] = []
        for item in observations:
            signal = self._signal_kind(item)
            if signal is None:
                next_identity = replace(
                    next_identity,
                    signal_conflict_count=next_identity.signal_conflict_count + 1,
                    version=next_identity.version + 1,
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.SIGNAL_CONFLICT,
                    next_identity.unique_count,
                    reason="novelty_and_match_confidence_are_both_high_or_both_low",
                )
            else:
                eligible.append(item)

        next_identity, next_state_bank, confirmed_decisions, remaining = (
            self._assign_and_update_confirmed(
                next_identity,
                state_bank,
                next_state_bank,
                eligible,
            )
        )
        decisions.update(confirmed_decisions)
        next_identity, next_state_bank, candidate_decisions, remaining = (
            self._assign_and_update_candidates(
                next_identity,
                state_bank,
                next_state_bank,
                remaining,
                chunk_index,
            )
        )
        decisions.update(candidate_decisions)

        matched_candidate_ids = {
            decision.candidate_id
            for decision in candidate_decisions.values()
            if decision.status
            in {IdentityDecisionStatus.CANDIDATE_UPDATED, IdentityDecisionStatus.PROMOTED}
            and decision.candidate_id is not None
        }
        next_identity, next_state_bank = self._age_and_prune_candidates(
            next_identity,
            state_bank,
            next_state_bank,
            matched_candidate_ids,
            chunk_index,
            observations,
        )
        for item in sorted(remaining, key=lambda value: (value.position_id, value.slot_index)):
            next_identity, next_state_bank, decision = self._create_or_reject_candidate(
                next_identity,
                state_bank,
                next_state_bank,
                item,
                chunk_index,
            )
            decisions[item.slot_index] = decision
        next_identity = replace(
            next_identity,
            last_chunk_index=chunk_index,
            last_committed_position_id=(
                committed_position
                if committed_position is not None
                else identity_state.last_committed_position_id
            ),
            version=next_identity.version + 1,
        )
        ordered_decisions = tuple(decisions[index] for index in sorted(decisions))
        return IdentityUpdateResult(next_identity, next_state_bank, ordered_decisions)

    def _extract_observations(
        self, observation: O2SoftOutput, semantic_embeddings: Tensor, row: int
    ) -> tuple[_Observation, ...]:
        items: list[_Observation] = []
        for slot_index in range(observation.identity.shape[1]):
            if not bool(observation.valid_mask[row, slot_index]):
                continue
            timestamp = float(observation.timestamps[row, slot_index].item())
            position_id = int(observation.position_ids[row, slot_index].item())
            novelty = float(observation.score_probabilities[row, slot_index, 0].float().item())
            match_confidence = float(
                observation.score_probabilities[row, slot_index, 1].float().item()
            )
            confidence = max(novelty, match_confidence)
            items.append(
                _Observation(
                    slot_index=slot_index,
                    timestamp=timestamp,
                    position_id=position_id,
                    identity=_hard_identity(observation.identity[row, slot_index]),
                    semantic=semantic_embeddings[slot_index],
                    novelty=novelty,
                    match_confidence=match_confidence,
                    confidence=confidence,
                    relevance=float(observation.relevance[row, slot_index].float().item()),
                )
            )
        return tuple(sorted(items, key=lambda item: (item.position_id, item.slot_index)))

    def _signal_kind(self, item: _Observation) -> str | None:
        novelty_high = item.novelty >= self.o2_config.novelty_threshold
        match_high = item.match_confidence >= self.o2_config.match_confidence_threshold
        if novelty_high == match_high:
            return None
        return "new" if novelty_high else "match"

    def _assign_and_update_confirmed(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observations: Sequence[_Observation],
    ) -> tuple[
        IdentityBankRuntimeState,
        StateBankRuntimeState,
        dict[int, IdentityObservationDecision],
        tuple[_Observation, ...],
    ]:
        matches = {
            item.slot_index: self._match_confirmed(identity_state, item.identity)
            for item in observations
        }
        decisions: dict[int, IdentityObservationDecision] = {}
        remaining: list[_Observation] = []
        claims: dict[str, list[tuple[_Observation, _Match]]] = {}
        next_identity = identity_state
        next_state_bank = state_state
        for item in observations:
            match = matches[item.slot_index]
            if match.status is ExactMatchStatus.AMBIGUOUS:
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    similarity=match.score,
                    reason="confirmed_near_tie",
                )
            elif match.status is ExactMatchStatus.MATCHED and match.entry_id is not None:
                claims.setdefault(match.entry_id, []).append((item, match))
            else:
                remaining.append(item)
        winners: list[tuple[_Observation, _Match]] = []
        for identity_id, group in claims.items():
            ordered = sorted(
                group,
                key=lambda pair: (
                    -cast(float, pair[1].score),
                    -pair[0].match_confidence,
                    pair[0].slot_index,
                    identity_id,
                ),
            )
            winners.append(ordered[0])
            for loser, loser_match in ordered[1:]:
                decisions[loser.slot_index] = self._decision(
                    loser,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    identity_id=identity_id,
                    similarity=loser_match.score,
                    reason="one_to_one_confirmed_claim",
                )
        for item, match in sorted(
            winners, key=lambda pair: (pair[0].position_id, pair[0].slot_index)
        ):
            assert match.entry_id is not None
            previous = self.confirmed_by_id(next_identity, match.entry_id)
            updated = replace(
                previous,
                identity_prototype=_prototype_ema(
                    previous.identity_prototype,
                    item.identity,
                    self.o2_config.prototype_ema,
                ),
                last_seen=item.timestamp,
                last_seen_position_id=item.position_id,
                observation_count=previous.observation_count + 1,
                prototype_version=previous.prototype_version + 1,
                relevance=_relevance_ema(
                    previous.relevance,
                    item.relevance,
                    self.o2_config.prototype_ema,
                ),
            )
            next_state_bank = state_bank.update_o2_confirmed(
                next_state_bank,
                confirmed=updated,
                semantic_embedding=item.semantic,
                confidence=item.confidence,
                audit_timestamp=item.timestamp,
            )
            next_identity = self._replace_confirmed(next_identity, updated, item)
            decisions[item.slot_index] = self._decision(
                item,
                IdentityDecisionStatus.CONFIRMED_UPDATED,
                identity_state.unique_count,
                identity_id=updated.identity_id,
                similarity=match.score,
            )
        return next_identity, next_state_bank, decisions, tuple(remaining)

    def _assign_and_update_candidates(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        observations: Sequence[_Observation],
        chunk_index: int,
    ) -> tuple[
        IdentityBankRuntimeState,
        StateBankRuntimeState,
        dict[int, IdentityObservationDecision],
        tuple[_Observation, ...],
    ]:
        matches = {
            item.slot_index: self._match_candidates(identity_state, item.identity)
            for item in observations
        }
        decisions: dict[int, IdentityObservationDecision] = {}
        remaining: list[_Observation] = []
        claims: dict[str, list[tuple[_Observation, _Match]]] = {}
        next_identity = identity_state
        next_state_bank = state_state
        for item in observations:
            match = matches[item.slot_index]
            if match.status is ExactMatchStatus.AMBIGUOUS:
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    similarity=match.score,
                    reason="candidate_near_tie",
                )
            elif match.status is ExactMatchStatus.MATCHED and match.entry_id is not None:
                claims.setdefault(match.entry_id, []).append((item, match))
            else:
                remaining.append(item)
        winners: list[tuple[_Observation, _Match]] = []
        for candidate_id, group in claims.items():
            ordered = sorted(
                group,
                key=lambda pair: (
                    -cast(float, pair[1].score),
                    -pair[0].match_confidence,
                    pair[0].slot_index,
                    candidate_id,
                ),
            )
            winners.append(ordered[0])
            for loser, loser_match in ordered[1:]:
                decisions[loser.slot_index] = self._decision(
                    loser,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    candidate_id=candidate_id,
                    similarity=loser_match.score,
                    reason="one_to_one_candidate_claim",
                )
        for item, match in sorted(
            winners, key=lambda pair: (pair[0].position_id, pair[0].slot_index)
        ):
            assert match.entry_id is not None
            previous = _candidate_by_id(next_identity, match.entry_id)
            reliable = item.confidence >= self.o2_config.reliability_threshold
            if reliable and chunk_index == previous.last_reliable_chunk_index + 1:
                reliable_streak = previous.reliable_streak + 1
            elif reliable and chunk_index > previous.last_reliable_chunk_index:
                reliable_streak = 1
            else:
                reliable_streak = previous.reliable_streak
            updated = replace(
                previous,
                identity_prototype=_prototype_ema(
                    previous.identity_prototype,
                    item.identity,
                    self.o2_config.prototype_ema,
                ),
                observation_count=previous.observation_count + 1,
                ttl_remaining=self.candidate_config.ttl_chunks,
                confidence=self.o2_config.prototype_ema * previous.confidence
                + (1.0 - self.o2_config.prototype_ema) * item.confidence,
                last_seen=item.timestamp,
                last_seen_position_id=item.position_id,
                last_reliable_chunk_index=chunk_index
                if reliable
                else previous.last_reliable_chunk_index,
                reliable_streak=reliable_streak,
                relevance=_relevance_ema(
                    previous.relevance,
                    item.relevance,
                    self.o2_config.prototype_ema,
                ),
            )
            if reliable_streak >= self.o2_config.confirmation_observations:
                next_identity, next_state_bank, confirmed = self._promote_candidate(
                    next_identity,
                    state_bank,
                    next_state_bank,
                    updated,
                    item,
                )
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.PROMOTED,
                    identity_state.unique_count,
                    candidate_id=updated.candidate_id,
                    identity_id=confirmed.identity_id,
                    similarity=match.score,
                )
            else:
                next_state_bank = state_bank.update_o2_candidate(
                    next_state_bank,
                    candidate=updated,
                    semantic_embedding=item.semantic,
                    confidence=updated.confidence,
                    audit_timestamp=item.timestamp,
                )
                next_identity = _replace_candidate(next_identity, updated, item)
                decisions[item.slot_index] = self._decision(
                    item,
                    IdentityDecisionStatus.CANDIDATE_UPDATED,
                    next_identity.unique_count,
                    candidate_id=updated.candidate_id,
                    similarity=match.score,
                )
        return next_identity, next_state_bank, decisions, tuple(remaining)

    def _create_or_reject_candidate(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        item: _Observation,
        chunk_index: int,
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState, IdentityObservationDecision]:
        dynamic_match = self._match_candidates(identity_state, item.identity)
        if dynamic_match.status is not ExactMatchStatus.UNMATCHED:
            return (
                identity_state,
                state_state,
                self._decision(
                    item,
                    IdentityDecisionStatus.MATCH_CONFLICT,
                    identity_state.unique_count,
                    candidate_id=dynamic_match.entry_id,
                    similarity=dynamic_match.score,
                    reason="same_chunk_candidate_claim",
                ),
            )
        next_identity = identity_state
        next_state_bank = state_state
        if len(next_identity.candidates) >= self.candidate_config.hard_limit:
            return (
                next_identity,
                next_state_bank,
                self._decision(
                    item,
                    IdentityDecisionStatus.OVERFLOW_REJECTED,
                    next_identity.unique_count,
                    reason="candidate_hard_limit_reached",
                ),
            )
        candidate_id = f"candidate-{next_identity.next_candidate_sequence:08d}"

        draft = CandidateIdentity(
            candidate_id=candidate_id,
            identity_prototype=item.identity,
            observation_count=1,
            ttl_remaining=self.candidate_config.ttl_chunks,
            confidence=item.confidence,
            first_seen=item.timestamp,
            last_seen=item.timestamp,
            first_seen_position_id=item.position_id,
            last_seen_position_id=item.position_id,
            last_reliable_chunk_index=chunk_index,
            reliable_streak=1 if item.confidence >= self.o2_config.reliability_threshold else 0,
            semantic_record_id=None,
            relevance=item.relevance,
        )
        next_state_bank, record = state_bank.append_o2_candidate(
            next_state_bank,
            semantic_embedding=item.semantic,
            candidate=draft,
            confidence=item.confidence,
        )
        linked = cast(CandidateIdentity, record.payload)
        next_identity = replace(
            next_identity,
            candidates=next_identity.candidates + (_clone_candidate(linked),),
            next_candidate_sequence=next_identity.next_candidate_sequence + 1,
            issued_candidate_ids=next_identity.issued_candidate_ids + (candidate_id,),
            version=next_identity.version + 1,
        )
        return (
            next_identity,
            next_state_bank,
            self._decision(
                item,
                IdentityDecisionStatus.CANDIDATE_CREATED,
                next_identity.unique_count,
                candidate_id=candidate_id,
            ),
        )

    def _promote_candidate(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        candidate: CandidateIdentity,
        item: _Observation,
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState, ConfirmedIdentity]:
        identity_id = f"identity-{identity_state.next_identity_sequence:08d}"
        draft = ConfirmedIdentity(
            identity_id=identity_id,
            identity_prototype=candidate.identity_prototype,
            first_seen=candidate.first_seen,
            last_seen=candidate.last_seen,
            observation_count=candidate.observation_count,
            semantic_record_id=None,
            prototype_version=0,
            first_seen_position_id=candidate.first_seen_position_id,
            last_seen_position_id=candidate.last_seen_position_id,
            relevance=candidate.relevance,
        )
        assert candidate.semantic_record_id is not None
        next_state_bank, record = state_bank.promote_o2_candidate(
            state_state,
            candidate.semantic_record_id,
            semantic_embedding=item.semantic,
            confirmed=draft,
            confidence=candidate.confidence,
            audit_timestamp=item.timestamp,
        )
        linked = cast(ConfirmedIdentity, record.payload)
        remaining = tuple(
            _clone_candidate(value)
            for value in identity_state.candidates
            if value.candidate_id != candidate.candidate_id
        )
        next_identity = replace(
            identity_state,
            candidates=remaining,
            confirmed=_append_confirmed(identity_state.confirmed, linked),
            next_identity_sequence=identity_state.next_identity_sequence + 1,
            issued_identity_ids=identity_state.issued_identity_ids + (identity_id,),
            version=identity_state.version + 1,
        )
        return next_identity, next_state_bank, linked

    def _age_and_prune_candidates(
        self,
        identity_state: IdentityBankRuntimeState,
        state_bank: StructuredStateBank,
        state_state: StateBankRuntimeState,
        refreshed_candidate_ids: set[str],
        chunk_index: int,
        observations: Sequence[_Observation],
    ) -> tuple[IdentityBankRuntimeState, StateBankRuntimeState]:
        timestamp = max((item.timestamp for item in observations), default=float(chunk_index))
        kept: list[CandidateIdentity] = []
        expired: list[CandidateIdentity] = []
        low_confidence: list[CandidateIdentity] = []
        for candidate in identity_state.candidates:
            if candidate.candidate_id in refreshed_candidate_ids:
                aged = candidate
            else:
                aged = replace(candidate, ttl_remaining=max(candidate.ttl_remaining - 1, 0))
            if aged.ttl_remaining == 0:
                expired.append(aged)
            elif aged.confidence < self.candidate_config.low_confidence_threshold:
                low_confidence.append(aged)
            else:
                kept.append(aged)
        low_confidence.sort(
            key=lambda candidate: (
                candidate.confidence,
                candidate.last_seen_position_id,
                candidate.candidate_id,
            )
        )
        next_state_bank = state_state
        for candidate, reason in (
            *((candidate, "ttl_expired") for candidate in expired),
            *((candidate, "low_confidence") for candidate in low_confidence),
        ):
            assert candidate.semantic_record_id is not None
            next_state_bank = state_bank.invalidate_o2_candidate(
                next_state_bank,
                candidate.semantic_record_id,
                audit_timestamp=timestamp,
                reason=reason,
            )
        if not expired and not low_confidence and tuple(kept) == identity_state.candidates:
            return identity_state, next_state_bank
        return (
            replace(
                identity_state,
                candidates=tuple(_clone_candidate(candidate) for candidate in kept),
                candidate_expired_count=identity_state.candidate_expired_count + len(expired),
                candidate_low_confidence_pruned_count=(
                    identity_state.candidate_low_confidence_pruned_count + len(low_confidence)
                ),
                version=identity_state.version + 1,
            ),
            next_state_bank,
        )

    def _match_confirmed(self, state: IdentityBankRuntimeState, query: Tensor) -> _Match:
        identities = state.confirmed
        if not identities:
            return _Match(ExactMatchStatus.UNMATCHED, None, None)
        prototypes = torch.stack([identity.identity_prototype for identity in identities])
        scores = prototypes @ query
        return _select_match(
            tuple(identity.identity_id for identity in identities),
            scores,
            self.o2_config.match_threshold,
            self.o2_config.match_ambiguity_margin,
        )

    def _match_candidates(self, state: IdentityBankRuntimeState, query: Tensor) -> _Match:
        if not state.candidates:
            return _Match(ExactMatchStatus.UNMATCHED, None, None)
        prototypes = torch.stack([candidate.identity_prototype for candidate in state.candidates])
        scores = prototypes @ query
        return _select_match(
            tuple(candidate.candidate_id for candidate in state.candidates),
            scores,
            self.candidate_config.match_threshold,
            self.o2_config.match_ambiguity_margin,
        )

    def _replace_confirmed(
        self,
        state: IdentityBankRuntimeState,
        confirmed: ConfirmedIdentity,
        item: _Observation,
    ) -> IdentityBankRuntimeState:
        return replace(
            state,
            confirmed=_update_confirmed(state.confirmed, confirmed),
            version=state.version + 1,
        )

    def _decision(
        self,
        item: _Observation,
        status: IdentityDecisionStatus,
        scanned: int,
        *,
        candidate_id: str | None = None,
        identity_id: str | None = None,
        similarity: float | None = None,
        reason: str | None = None,
    ) -> IdentityObservationDecision:
        return IdentityObservationDecision(
            slot_index=item.slot_index,
            position_id=item.position_id,
            timestamp=item.timestamp,
            status=status,
            candidate_id=candidate_id,
            identity_id=identity_id,
        )


def build_identity_bank(config: ProjectConfig | None = None) -> IdentityBank:
    return IdentityBank(config)


def _append_confirmed(
    confirmed_store: tuple[ConfirmedIdentity, ...],
    confirmed: ConfirmedIdentity,
) -> tuple[ConfirmedIdentity, ...]:
    stored = replace(confirmed, identity_prototype=_hard_identity(confirmed.identity_prototype))
    return tuple(_clone_confirmed(item) for item in confirmed_store) + (stored,)


def _update_confirmed(
    confirmed_store: tuple[ConfirmedIdentity, ...], confirmed: ConfirmedIdentity
) -> tuple[ConfirmedIdentity, ...]:
    stored = replace(confirmed, identity_prototype=_hard_identity(confirmed.identity_prototype))
    return tuple(
        stored if item.identity_id == confirmed.identity_id else _clone_confirmed(item)
        for item in confirmed_store
    )


def _select_match(
    entry_ids: tuple[str, ...],
    scores: Tensor,
    threshold: float,
    ambiguity_margin: float,
) -> _Match:
    ordered = sorted(
        (
            (float(score.item()), entry_id)
            for score, entry_id in zip(scores, entry_ids, strict=True)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_id = ordered[0]
    if best_score < threshold:
        return _Match(ExactMatchStatus.UNMATCHED, None, best_score)
    if len(ordered) > 1:
        second_score, second_id = ordered[1]
        if second_score >= threshold and best_score - second_score <= ambiguity_margin:
            return _Match(
                ExactMatchStatus.AMBIGUOUS,
                None,
                best_score,
                tuple(sorted((best_id, second_id))),
            )
    return _Match(ExactMatchStatus.MATCHED, best_id, best_score)


def _prototype_ema(old: Tensor, observation: Tensor, decay: float) -> Tensor:
    return _normalize_identity(decay * old.float() + (1.0 - decay) * observation.float())


def _relevance_ema(old: float, observation: float, decay: float) -> float:
    value = decay * old + (1.0 - decay) * observation
    return min(1.0, max(0.0, value))


def _hard_identity(identity: Tensor) -> Tensor:
    return _normalize_identity(identity.detach().to(device="cpu", dtype=torch.float32, copy=True))


def _normalize_identity(identity: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(identity.float())
    if float(norm.item()) <= 1.0e-8:
        output = torch.zeros(IDENTITY_DIM, dtype=torch.float32, device="cpu")
        output[0] = 1.0
        return output
    return F.normalize(identity.float(), dim=0, eps=1.0e-8).to(device="cpu", copy=True).detach()


def _clone_candidate(candidate: CandidateIdentity) -> CandidateIdentity:
    return replace(candidate, identity_prototype=candidate.identity_prototype.detach().clone())


def _clone_confirmed(confirmed: ConfirmedIdentity) -> ConfirmedIdentity:
    return replace(confirmed, identity_prototype=confirmed.identity_prototype.detach().clone())


def _replace_candidate(
    state: IdentityBankRuntimeState, candidate: CandidateIdentity, item: _Observation
) -> IdentityBankRuntimeState:
    candidates = tuple(
        _clone_candidate(candidate if value.candidate_id == candidate.candidate_id else value)
        for value in state.candidates
    )
    return replace(
        state,
        candidates=candidates,
        version=state.version + 1,
    )


def _candidate_by_id(state: IdentityBankRuntimeState, candidate_id: str) -> CandidateIdentity:
    return _clone_candidate(
        next(
            candidate
            for candidate in state.candidates
            if candidate.candidate_id == candidate_id
        )
    )
