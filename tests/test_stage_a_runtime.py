from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import Tensor

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.identity_bank import IdentityDecisionStatus, build_identity_bank
from ttt_svcbench_qwen.model import BatchRuntimeState, ObservationChunkRequest, RuntimeOwner
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
    StreamReplayAudit,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    OperatorRouterOutput,
    QueryEmbeddingOutput,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeResolverLogits,
    TimeResolverOutput,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.stage_a_runtime import (
    StageABankWriter,
    StageASoftWriteOutput,
    StageAWriteAudit,
)
from ttt_svcbench_qwen.state_bank import HeadType, build_state_bank
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
    TemporalEncoderOutput,
)
from ttt_svcbench_qwen.state_reader import DeterministicStateReader, ReaderResult
from ttt_svcbench_qwen.state_retriever import build_state_retriever


class _NumberTokenizer:
    name_or_path = "synthetic-number-tokenizer"
    vocab_size = 256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(value) for value in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value) for value in token_ids)


def _cache(owner: RuntimeOwner, hidden: Tensor, query: Tensor) -> TemporalCache:
    batch_size, width = hidden.shape[:2]
    kv_shape = (batch_size, 12, width, 64)
    replay_shape = (batch_size, 12, 0, 64)
    timestamps = torch.arange(width, dtype=torch.float64).expand(batch_size, -1).clone()
    positions = torch.arange(width, dtype=torch.int64).expand(batch_size, -1).clone()
    return TemporalCache(
        hidden=hidden.detach().clone(),
        layer_keys=tuple(torch.zeros(kv_shape) for _ in range(6)),
        layer_values=tuple(torch.zeros(kv_shape) for _ in range(6)),
        replay_layer_keys=tuple(torch.zeros(replay_shape) for _ in range(6)),
        replay_layer_values=tuple(torch.zeros(replay_shape) for _ in range(6)),
        timestamps=timestamps,
        replay_timestamps=torch.empty((batch_size, 0), dtype=torch.float64),
        position_ids=positions,
        replay_position_ids=torch.empty((batch_size, 0), dtype=torch.int64),
        valid_mask=torch.ones((batch_size, width), dtype=torch.bool),
        replay_valid_mask=torch.empty((batch_size, 0), dtype=torch.bool),
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
        query_signatures=query.detach().clone(),
        total_seen=torch.full((batch_size,), width, dtype=torch.int64),
    )


def _spatial(owner: RuntimeOwner, slots: Tensor) -> SpatialEncoderOutput:
    batch_size, width = slots.shape[:2]
    mask = torch.ones((batch_size, width), dtype=torch.bool)
    next_states = tuple(
        SpatialSlotRuntimeState(
            video_id=owner.video_ids[row],
            slots=slots[row].detach().clone(),
            slot_valid_mask=mask[row].clone(),
            slot_confidence=torch.ones(width),
            active_slot_overflow_count=0,
            overflow_event_count=0,
            processed_tubelets=3,
        )
        for row in range(batch_size)
    )
    return SpatialEncoderOutput(
        slots=slots,
        slot_valid_mask=mask,
        active_slot_overflow_count=torch.zeros(batch_size, dtype=torch.int64),
        slot_confidence=torch.ones((batch_size, width)),
        next_states=next_states,
    )


def _stream_states(
    owner: RuntimeOwner, query: Tensor, width: int
) -> tuple[tuple[E1RuntimeState, ...], tuple[E2RuntimeState, ...]]:
    timestamps = torch.arange(width, dtype=torch.float64)
    positions = torch.arange(width, dtype=torch.int64)
    e1 = tuple(
        E1RuntimeState(
            video_id=owner.video_ids[row],
            trajectory_id=owner.trajectory_ids[row],
            query_signature=query[row].detach().clone(),
            projected_history=torch.zeros((width, 512)),
            timestamps=timestamps.clone(),
            position_ids=positions.clone(),
            total_seen=width,
        )
        for row in range(len(owner.video_ids))
    )
    e2_items: list[E2RuntimeState] = []
    for row in range(len(owner.video_ids)):
        checkpoints = torch.zeros((width, 2, 768))
        e2_items.append(
            E2RuntimeState(
                video_id=owner.video_ids[row],
                trajectory_id=owner.trajectory_ids[row],
                query_signature=query[row].detach().clone(),
                hidden=checkpoints[-1].clone(),
                checkpoint_hidden=checkpoints,
                timestamps=timestamps.clone(),
                position_ids=positions.clone(),
                total_seen=width,
            )
        )
    return e1, tuple(e2_items)


def _observations(
    owner: RuntimeOwner,
    spatial: SpatialEncoderOutput,
    temporal: TemporalEncoderOutput,
    query: Tensor,
) -> ObservationOutputs:
    batch_size, slots = spatial.slot_valid_mask.shape
    width = temporal.valid_mask.shape[1]
    slot_times = torch.full((batch_size, slots), float(width - 1), dtype=torch.float64)
    slot_positions = torch.full((batch_size, slots), width - 1, dtype=torch.int64)
    o1_logits = torch.full((batch_size, slots, 6), 5.0, requires_grad=True)
    o1_probabilities = torch.sigmoid(o1_logits)
    o1 = O1SoftOutput(
        logits=o1_logits,
        probabilities=o1_probabilities,
        soft_count=(o1_probabilities[..., :3].prod(dim=-1) * spatial.slot_valid_mask).sum(dim=1),
        valid_mask=spatial.slot_valid_mask.clone(),
        timestamps=slot_times,
        position_ids=slot_positions,
    )
    identities = torch.nn.functional.normalize(
        torch.randn((batch_size, slots, 256), requires_grad=True), dim=-1
    )
    score_logits = torch.empty((batch_size, slots, 2), requires_grad=True)
    with torch.no_grad():
        score_logits[..., 0] = 5.0
        score_logits[..., 1] = -5.0
    o2 = O2SoftOutput(
        identity=identities,
        score_logits=score_logits,
        score_probabilities=torch.sigmoid(score_logits),
        valid_mask=spatial.slot_valid_mask.clone(),
        timestamps=slot_times.clone(),
        position_ids=slot_positions.clone(),
    )
    e1_states, e2_states = _stream_states(owner, query, width)
    timestamps = temporal.timestamps.clone()
    positions = temporal.position_ids.clone()
    e1_logits = torch.full((batch_size, width, 3), 5.0, requires_grad=True)
    e1 = E1SoftOutput(
        logits=e1_logits,
        probabilities=torch.sigmoid(e1_logits),
        valid_mask=temporal.valid_mask.clone(),
        timestamps=timestamps,
        position_ids=positions,
        next_states=e1_states,
        audit=StreamReplayAudit(
            "e1",
            (width,) * batch_size,
            (0,) * batch_size,
            (width,) * batch_size,
        ),
    )
    event_logits = torch.full((batch_size, width, 4), 5.0, requires_grad=True)
    phase_logits = torch.zeros((batch_size, width, 4), requires_grad=True)
    e2 = E2SoftOutput(
        event_logits=event_logits,
        phase_logits=phase_logits,
        event_probabilities=torch.sigmoid(event_logits),
        phase_probabilities=torch.softmax(phase_logits, dim=-1),
        valid_mask=temporal.valid_mask.clone(),
        timestamps=timestamps.clone(),
        position_ids=positions.clone(),
        next_states=e2_states,
        audit=StreamReplayAudit(
            "e2",
            (width,) * batch_size,
            (0,) * batch_size,
            (width,) * batch_size,
        ),
    )
    return ObservationOutputs(o1=o1, o2=o2, e1=e1, e2=e2)


def _query(owner: RuntimeOwner) -> QueryEncoderOutput:
    batch_size = len(owner.video_ids)
    operators = (
        Operator.O1_SNAP,
        Operator.O2_UNIQUE,
        Operator.E1_ACTION,
        Operator.E2_PERIODIC,
    )
    query = torch.nn.functional.normalize(torch.randn((batch_size, 512)), dim=-1)
    embeddings = QueryEmbeddingOutput(
        token_states=torch.zeros((batch_size, 1, 768)),
        pooling_weights=torch.ones((batch_size, 1)),
        q_target=query,
        q_operator=query.clone(),
        q_time=query.clone(),
        padding_mask=torch.zeros((batch_size, 1), dtype=torch.bool),
    )
    raw = torch.tensor([tuple(Operator).index(value) for value in operators])
    logits = torch.full((batch_size, 9), -5.0)
    logits[torch.arange(batch_size), raw] = 5.0
    route = OperatorRouterOutput(
        logits=logits,
        confidence=torch.ones(batch_size),
        raw_indices=raw,
        hard_operators=operators,
        head_types=tuple(OPERATOR_TO_HEAD_TYPE[value] for value in operators),
        confidence_gate_applied=False,
    )
    time_logits = TimeResolverLogits(
        mode_logits=torch.zeros((batch_size, 4)),
        mode_confidence=torch.ones(batch_size),
        mode_indices=torch.ones(batch_size, dtype=torch.int64),
        span_start_logits=torch.zeros((batch_size, 1)),
        span_end_logits=torch.zeros((batch_size, 1)),
        padding_mask=torch.zeros((batch_size, 1), dtype=torch.bool),
    )
    resolutions = tuple(
        TimeResolution(
            window=TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True),
            status=TimeResolutionStatus.OK,
            reason="synthetic_explicit",
            mode_confidence=1.0,
            numeric_span=None,
            parsed_values_seconds=(),
            used_operator_default=True,
        )
        for _ in range(batch_size)
    )
    return QueryEncoderOutput(
        embeddings=embeddings,
        route=route,
        time=TimeResolverOutput(time_logits, resolutions),
        hard_operators=operators,
        head_types=route.head_types,
    )


def test_stage_a_writer_runs_four_hard_heads_and_keeps_soft_projector_gradient() -> None:
    torch.manual_seed(15)
    owner = RuntimeOwner(
        ("video-o1", "video-o2", "video-e1", "video-e2"),
        ("trajectory-o1", "trajectory-o2", "trajectory-e1", "trajectory-e2"),
    )
    query = _query(owner)
    slots = torch.randn((4, 2, 768), requires_grad=True)
    hidden = torch.randn((4, 3, 768), requires_grad=True)
    spatial = _spatial(owner, slots)
    temporal = TemporalEncoderOutput(
        hidden=hidden,
        timestamps=torch.arange(3, dtype=torch.float64).expand(4, -1).clone(),
        position_ids=torch.arange(3, dtype=torch.int64).expand(4, -1).clone(),
        valid_mask=torch.ones((4, 3), dtype=torch.bool),
        cache=_cache(owner, hidden, query.q_target),
    )
    observations = _observations(owner, spatial, temporal, query.q_target)
    state_bank = build_state_bank(load_config())
    writer = StageABankWriter(state_bank, build_identity_bank(load_config()))
    runtime = writer.reset(owner)
    result = writer(
        observations,
        spatial,
        temporal,
        query,
        ObservationChunkRequest(
            owner=owner,
            video_input="synthetic",
            query_input="synthetic",
            runtime_state=runtime,
            bank_states=runtime.state_bank_states,
            inference=False,
        ),
    )

    assert isinstance(result.runtime_state, BatchRuntimeState)
    assert result.runtime_state.next_chunk_index == 1
    assert isinstance(result.audit, StageAWriteAudit)
    assert result.audit.head_types == (HeadType.O1, HeadType.O2, HeadType.E1, HeadType.E2)
    assert len(result.audit.identity_decisions) == 4
    assert result.audit.identity_decisions[0] == ()
    assert result.audit.identity_decisions[1]
    assert result.audit.identity_decisions[2:] == ((), ())
    assert isinstance(result.soft_write, StageASoftWriteOutput)
    assert all(state.records for state in result.bank_states)
    assert all(
        not record.semantic_embedding.requires_grad and record.semantic_embedding.grad_fn is None
        for state in result.bank_states
        for record in state.records
    )

    soft = result.soft_write
    assert all(
        not source.requires_grad and source.grad_fn is None
        for source in (
            soft.o1_sources,
            soft.o2_sources,
            soft.e1_sources,
            soft.e2_sources,
        )
    )
    soft_loss = (
        soft.o1_semantics.square().sum()
        + soft.o2_semantics.square().sum()
        + soft.e1_semantics.square().sum()
        + soft.e2_semantics.square().sum()
    )
    soft_loss.backward()
    projector_grads = tuple(
        parameter.grad for parameter in state_bank.semantic_projector.parameters()
    )
    assert all(value is not None and torch.isfinite(value).all() for value in projector_grads)
    assert any(
        float(value.abs().sum().item()) > 0.0 for value in projector_grads if value is not None
    )

    retriever = build_state_retriever(load_config())
    pre_write_view = state_bank.retrieval_view(
        runtime.state_bank_states,
        tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in query.hard_operators),
    )
    pre_write_retrieval = retriever(
        state_bank,
        pre_write_view,
        query,
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
    )
    assert pre_write_retrieval.n_state.tolist() == [0, 0, 0, 0]
    history_view = state_bank.retrieval_view(
        result.bank_states,
        tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in query.hard_operators),
    )
    history_retrieval = retriever(
        state_bank,
        history_view,
        query,
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
    )
    assert history_retrieval.n_state.tolist() == [1, 0, 1, 1]
    assert all(
        not record.semantic_source.requires_grad and record.semantic_source.grad_fn is None
        for state in result.bank_states
        for record in state.retrieval_history
    )
    reader_results = DeterministicStateReader(_NumberTokenizer()).read_bank(
        state_bank,
        result.bank_states,
        query,
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
    )
    assert len(reader_results) == 4
    assert all(isinstance(value, ReaderResult) for value in reader_results)
    assert tuple(value.operator for value in reader_results) == query.hard_operators


def test_stage_a_soft_write_masks_carried_slots_without_new_temporal_positions() -> None:
    owner = RuntimeOwner(("video",), ("trajectory",))
    slots = torch.randn((1, 2, 768), requires_grad=True)
    hidden = torch.zeros((1, 1, 768), requires_grad=True)
    query = torch.randn((1, 512))
    spatial = _spatial(owner, slots)
    temporal_mask = torch.zeros((1, 1), dtype=torch.bool)
    temporal = TemporalEncoderOutput(
        hidden=hidden,
        timestamps=torch.full((1, 1), -1.0, dtype=torch.float64),
        position_ids=torch.full((1, 1), -1, dtype=torch.int64),
        valid_mask=temporal_mask,
        cache=_cache(owner, hidden, query),
    )
    slot_mask = torch.zeros_like(spatial.slot_valid_mask)
    observations = SimpleNamespace(
        o1=SimpleNamespace(valid_mask=slot_mask),
        o2=SimpleNamespace(valid_mask=slot_mask.clone()),
        e1=SimpleNamespace(valid_mask=temporal_mask),
        e2=SimpleNamespace(valid_mask=temporal_mask.clone()),
    )
    writer = StageABankWriter(
        build_state_bank(load_config()),
        build_identity_bank(load_config()),
    )

    soft = writer._project_soft(spatial, temporal, observations)  # type: ignore[arg-type]

    assert not bool(soft.o1_present_mask.any().item())
    assert not bool(soft.o2_present_mask.any().item())
    assert torch.count_nonzero(soft.o1_semantics) == 0
    assert torch.count_nonzero(soft.o2_semantics) == 0
    assert torch.count_nonzero(soft.o1_sources) == 0
    assert torch.count_nonzero(soft.o2_sources) == 0


def test_o2_history_is_written_only_when_candidates_promote() -> None:
    torch.manual_seed(20260720)
    owner = RuntimeOwner(
        ("video-o1", "video-o2", "video-e1", "video-e2"),
        ("trajectory-o1", "trajectory-o2", "trajectory-e1", "trajectory-e2"),
    )
    query = _query(owner)
    slots = torch.randn((4, 2, 768))
    hidden = torch.randn((4, 3, 768))
    spatial = _spatial(owner, slots)
    temporal = TemporalEncoderOutput(
        hidden=hidden,
        timestamps=torch.arange(3, dtype=torch.float64).expand(4, -1).clone(),
        position_ids=torch.arange(3, dtype=torch.int64).expand(4, -1).clone(),
        valid_mask=torch.ones((4, 3), dtype=torch.bool),
        cache=_cache(owner, hidden, query.q_target),
    )
    observations = _observations(owner, spatial, temporal, query.q_target)
    state_bank = build_state_bank(load_config())
    writer = StageABankWriter(state_bank, build_identity_bank(load_config()))
    runtime = writer.reset(owner)
    request = ObservationChunkRequest(
        owner=owner,
        video_input="synthetic",
        query_input="synthetic",
        runtime_state=runtime,
        bank_states=runtime.state_bank_states,
        inference=False,
    )
    first = writer(observations, spatial, temporal, query, request)
    assert first.bank_states[1].retrieval_history == ()
    assert all(
        decision.status is IdentityDecisionStatus.CANDIDATE_CREATED
        for decision in first.audit.identity_decisions[1]
    )

    second_observations = replace(
        observations,
        o1=replace(
            observations.o1,
            timestamps=observations.o1.timestamps + 3.0,
            position_ids=observations.o1.position_ids + 3,
        ),
        o2=replace(
            observations.o2,
            timestamps=observations.o2.timestamps + 3.0,
            position_ids=observations.o2.position_ids + 3,
        ),
        e1=replace(
            observations.e1,
            timestamps=observations.e1.timestamps + 3.0,
            position_ids=observations.e1.position_ids + 3,
        ),
        e2=replace(
            observations.e2,
            timestamps=observations.e2.timestamps + 3.0,
            position_ids=observations.e2.position_ids + 3,
        ),
    )
    second = writer(
        second_observations,
        spatial,
        temporal,
        query,
        replace(
            request,
            runtime_state=first.runtime_state,
            bank_states=first.bank_states,
        ),
    )

    assert all(
        decision.status is IdentityDecisionStatus.PROMOTED
        for decision in second.audit.identity_decisions[1]
    )
    o2_history = second.bank_states[1].retrieval_history
    assert len(o2_history) == 2
    assert all(record.head_type is HeadType.O2 for record in o2_history)
    assert all(record.lifecycle_id is not None for record in o2_history)
    assert all(record.retrieval_eligible for record in o2_history)
    assert [len(second.bank_states[index].records) for index in (0, 2, 3)] == [1, 1, 1]
    assert [
        len(second.bank_states[index].retrieval_history) for index in (0, 2, 3)
    ] == [2, 2, 2]


def test_stage_a_runtime_has_no_fast_or_optimizer_state() -> None:
    owner = RuntimeOwner(("video",), ("trajectory",))
    writer = StageABankWriter(build_state_bank(load_config()), build_identity_bank(load_config()))
    runtime = writer.reset(owner)
    assert all(row.fast_weights is None and row.optimizer is None for row in runtime.rows)
