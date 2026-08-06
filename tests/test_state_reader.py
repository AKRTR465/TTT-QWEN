from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tests.support import parameter_count
from tests.support.runtime_factories import make_state_record
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.identity_bank import ConfirmedIdentity
from ttt_svcbench_qwen.query_encoder import (
    Operator,
    TimeResolution,
    TimeResolutionStatus,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import (
    E1EventKind,
    E1Payload,
    E2EventKind,
    E2Payload,
    E2Phase,
    HeadType,
    O1Payload,
    StateRecord,
)
from ttt_svcbench_qwen.state_reader import (
    DeterministicStateReader,
    ReaderResult,
    ReaderStatus,
    StateResampler,
    StateResamplerOutput,
    build_state_reader,
    build_state_resampler,
    serialize_number_token_ids,
)
from ttt_svcbench_qwen.state_retriever import (
    RetrievalFilterAudit,
    RetrievalReason,
    RetrievalStatus,
    RetrieverOutput,
)

SEMANTIC_DIM = 512
IDENTITY_DIM = 256
EXACT_RESAMPLER_PARAMETERS = 14_722_048
TOKENIZER_FILES = {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
_TIME_STATUS_FOR = {
    RetrievalStatus.OK: TimeResolutionStatus.OK,
    RetrievalStatus.EMPTY: TimeResolutionStatus.OK,
    RetrievalStatus.UNSUPPORTED: TimeResolutionStatus.UNSUPPORTED,
    RetrievalStatus.INVALID: TimeResolutionStatus.INVALID,
}


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config()


def _tokenizer_snapshot(config: ProjectConfig) -> Path:
    override = os.environ.get("TTT_SVCBENCH_TOKENIZER_SNAPSHOT")
    if override:
        return Path(override)
    cache_roots = []
    if hf_home := os.environ.get("HF_HOME"):
        cache_roots.append(Path(hf_home) / "hub")
    cache_roots.extend(
        (Path.home() / ".cache" / "huggingface" / "hub", Path("F:/huggingface_cache/hub"))
    )
    relative = Path("models--Qwen--Qwen3-VL-8B-Instruct") / "snapshots" / config.model.revision
    for root in cache_roots:
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    return cache_roots[-1] / relative


@pytest.fixture(scope="module")
def number_tokenizer(config: ProjectConfig) -> PreTrainedTokenizerBase:
    snapshot = _tokenizer_snapshot(config)
    assert snapshot.is_dir(), (
        "the pinned tokenizer-only snapshot is required; P12 tests must not download 8B weights"
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


@pytest.fixture(scope="module")
def resampler(config: ProjectConfig) -> StateResampler:
    torch.manual_seed(20260714)
    module = build_state_resampler(config)
    module.eval()
    return module


@pytest.fixture(scope="module")
def reader(
    config: ProjectConfig,
    number_tokenizer: PreTrainedTokenizerBase,
) -> DeterministicStateReader:
    return build_state_reader(config, tokenizer=number_tokenizer)


def _unit_semantic(index: int = 0) -> Tensor:
    value = torch.zeros(SEMANTIC_DIM, dtype=torch.float32)
    value[index % SEMANTIC_DIM] = 1.0
    return value


def _confirmed_record(sequence: int, *, first_seen: float = 1.0) -> StateRecord:
    record_id = f"o2-{sequence:08d}"
    identity_prototype = torch.zeros(IDENTITY_DIM, dtype=torch.float32)
    identity_prototype[sequence % IDENTITY_DIM] = 1.0
    return make_state_record(
        record_id,
        HeadType.O2,
        ConfirmedIdentity(
            identity_id=f"identity-{sequence:08d}",
            identity_prototype=identity_prototype,
            first_seen=first_seen,
            last_seen=first_seen,
            observation_count=2,
            semantic_record_id=record_id,
        ),
        semantic_embedding=_unit_semantic(sequence),
        timestamp=first_seen,
    )


def _o1_record(current_count: int, baseline_count: int) -> StateRecord:
    return make_state_record(
        "o1-aggregate",
        HeadType.O1,
        O1Payload(
            current_visible_count=current_count,
            baseline_count=baseline_count,
            active_slot_ids=tuple(range(current_count)),
            baseline_initialized=True,
            baseline_position_id=0,
        ),
        semantic_embedding=_unit_semantic(401),
        timestamp=10.0,
    )


def _e1_record(
    event_kind: E1EventKind,
    event_times: tuple[float, ...],
    *,
    event_count: int | None = None,
) -> StateRecord:
    return make_state_record(
        f"e1-{event_kind.value}",
        HeadType.E1,
        E1Payload(
            event_kind=event_kind,
            event_count=len(event_times) if event_count is None else event_count,
            recent_event_times=event_times,
            cooldown_until=0.0,
        ),
        semantic_embedding=_unit_semantic(402),
        timestamp=10.0,
    )


def _e2_record(
    event_kind: E2EventKind,
    intervals: tuple[tuple[float, float], ...],
    *,
    phase: E2Phase = E2Phase.COMPLETED,
    current_start: float | None = None,
) -> StateRecord:
    return make_state_record(
        f"e2-{event_kind.value}",
        HeadType.E2,
        E2Payload(
            event_kind=event_kind,
            completed_count=len(intervals),
            phase=phase,
            completed_intervals=intervals,
            recent_event_times=(),
            current_start=current_start,
        ),
        semantic_embedding=_unit_semantic(403),
        timestamp=10.0,
    )


def _resolution(
    *,
    mode: TimeWindowMode = TimeWindowMode.HISTORY,
    query_time: float = 10.0,
    start_time: float | None = 0.0,
    end_time: float | None = None,
    status: TimeResolutionStatus = TimeResolutionStatus.OK,
) -> TimeResolution:
    return TimeResolution(
        window=TimeWindow(
            mode=mode,
            query_time=query_time,
            start_time=start_time,
            end_time=query_time if end_time is None else end_time,
            valid=status is TimeResolutionStatus.OK,
        ),
        status=status,
    )


def _retrieval(
    rows: Sequence[Sequence[StateRecord]],
    *,
    candidate_rows: Sequence[Sequence[StateRecord]] | None = None,
    statuses: Sequence[RetrievalStatus] | None = None,
    reasons: Sequence[RetrievalReason] | None = None,
    hard_operators: Sequence[Operator] | None = None,
    time_resolutions: Sequence[TimeResolution] | None = None,
) -> RetrieverOutput:
    """Build one aligned RetrieverOutput the Resampler and Reader both consume.

    ``rows`` are the selected records per batch row; ``candidate_rows`` widens the
    candidate axis so the packing path sees ``n_state > n_retrieved``.
    """
    selected_rows = tuple(tuple(row) for row in rows)
    batch = len(selected_rows)
    candidate_per_row = (
        selected_rows if candidate_rows is None else tuple(tuple(row) for row in candidate_rows)
    )
    row_statuses = (
        tuple(statuses)
        if statuses is not None
        else tuple(RetrievalStatus.OK if row else RetrievalStatus.EMPTY for row in selected_rows)
    )
    row_reasons = (
        tuple(reasons)
        if reasons is not None
        else tuple(
            RetrievalReason.MATCHED if row else RetrievalReason.EMPTY_BANK for row in selected_rows
        )
    )
    head_defaults = {
        HeadType.O1: Operator.O1_SNAP,
        HeadType.O2: Operator.O2_UNIQUE,
        HeadType.E1: Operator.E1_ACTION,
        HeadType.E2: Operator.E2_PERIODIC,
    }
    operators = (
        tuple(hard_operators)
        if hard_operators is not None
        else tuple(
            head_defaults[row[0].head_type] if row else Operator.O1_SNAP
            for row in candidate_per_row
        )
    )
    resolutions = (
        tuple(time_resolutions)
        if time_resolutions is not None
        else tuple(_resolution(status=_TIME_STATUS_FOR[status]) for status in row_statuses)
    )
    assert (
        len(candidate_per_row)
        == len(row_statuses)
        == len(row_reasons)
        == len(operators)
        == len(resolutions)
        == batch
    )

    width = max((len(row) for row in candidate_per_row), default=0)
    embeddings = torch.zeros(batch, width, SEMANTIC_DIM, dtype=torch.float32)
    scores = torch.zeros(batch, width, dtype=torch.float32)
    present_mask = torch.zeros(batch, width, dtype=torch.bool)
    eligible_mask = torch.zeros_like(present_mask)
    selected_mask = torch.zeros_like(present_mask)
    candidate_ids: list[tuple[str | None, ...]] = []
    candidate_snapshots: list[tuple[StateRecord | None, ...]] = []
    selected_ids: list[tuple[str, ...]] = []
    selected_scores: list[tuple[float, ...]] = []
    selected_records: list[tuple[StateRecord, ...]] = []
    n_state = torch.zeros(batch, dtype=torch.int64)
    n_retrieved = torch.zeros(batch, dtype=torch.int64)

    for row, chosen in enumerate(selected_rows):
        assert (row_statuses[row] is RetrievalStatus.OK) == bool(chosen)
        owned = tuple(
            replace(record, video_id=f"video-{row}", trajectory_id=f"trajectory-{row}")
            for record in candidate_per_row[row]
        )
        chosen_ids = {record.record_id for record in chosen}
        for column, record in enumerate(owned):
            embeddings[row, column] = record.semantic_embedding
            scores[row, column] = 0.8 if record.record_id in chosen_ids else 0.1
            present_mask[row, column] = True
            eligible_mask[row, column] = record.valid
            selected_mask[row, column] = record.record_id in chosen_ids
        padding = (None,) * (width - len(owned))
        candidate_ids.append(tuple(record.record_id for record in owned) + padding)
        candidate_snapshots.append(owned + padding)
        # Canonical evidence order is score-descending then record_id; every selected
        # record carries the same synthetic score, so this reduces to a record_id sort.
        canonical = tuple(
            sorted(
                (record for record in owned if record.record_id in chosen_ids),
                key=lambda record: record.record_id,
            )
        )
        assert len(canonical) == len(chosen_ids)
        selected_ids.append(tuple(record.record_id for record in canonical))
        selected_scores.append((0.8,) * len(canonical))
        selected_records.append(canonical)
        n_state[row] = len(owned)
        n_retrieved[row] = len(canonical)

    return RetrieverOutput(
        selected_record_ids=tuple(selected_ids),
        selected_scores=tuple(selected_scores),
        selected_records=tuple(selected_records),
        candidate_record_ids=tuple(candidate_ids),
        candidate_records=tuple(candidate_snapshots),
        candidate_head_types=tuple(
            tuple(record.head_type if record is not None else None for record in row)
            for row in candidate_snapshots
        ),
        state_embeddings=embeddings,
        scores=scores,
        present_mask=present_mask,
        record_valid_mask=eligible_mask.clone(),
        retrieval_eligible_mask=eligible_mask,
        causal_mask=present_mask.clone(),
        predicted_head_mask=present_mask.clone(),
        selected_mask=selected_mask,
        status=row_statuses,
        reason=row_reasons,
        hard_operators=operators,
        time_resolutions=resolutions,
        n_state=n_state,
        n_retrieved=n_retrieved,
        audit=tuple(
            RetrievalFilterAudit(
                n_state=int(n_state[row].item()),
                selected_count=int(n_retrieved[row].item()),
            )
            for row in range(batch)
        ),
        video_ids=tuple(f"video-{row}" for row in range(batch)),
        trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch)),
        bank_video_ids=tuple(f"video-{row}" for row in range(batch)),
        bank_trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch)),
        bank_versions=tuple(range(batch)),
    )


def _read_one(
    reader: DeterministicStateReader,
    operator: Operator,
    resolution: TimeResolution,
    records: Sequence[StateRecord],
) -> ReaderResult:
    retrieval = _retrieval(
        (tuple(records),),
        hard_operators=(operator,),
        time_resolutions=(resolution,),
    )
    results = reader.read(retrieval)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ReaderResult)
    return result


def _assert_number_tokens(
    tokenizer: PreTrainedTokenizerBase,
    result: ReaderResult,
    expected_count: int,
) -> None:
    """Pin the product contract: the Answer path consumes Reader-emitted number ids."""

    assert result.status is ReaderStatus.OK or result.status is ReaderStatus.EMPTY
    assert result.exact_count == expected_count
    assert result.number_token_ids == serialize_number_token_ids(tokenizer, expected_count)
    assert result.number_token_ids != ()
    assert tokenizer.decode(
        list(result.number_token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == str(expected_count)
    assert dict(result.audit_fields)["number_text"] == str(expected_count)


def test_meta_topology_and_exact_state_resampler_parameter_count(config: ProjectConfig) -> None:
    with torch.device("meta"):
        module = build_state_resampler(config)

    assert isinstance(module, StateResampler)
    assert module.q_state.shape == (16, 512)
    assert module.empty_record_embedding.shape == (512,)
    assert len(module.layers) == 3
    assert parameter_count(module) == EXACT_RESAMPLER_PARAMETERS


@pytest.mark.parametrize("record_count", (0, 3, 30, 300))
def test_state_resampler_keeps_fixed_sixteen_token_shape_for_dynamic_records(
    resampler: StateResampler,
    record_count: int,
) -> None:
    retrieval = _retrieval((tuple(_confirmed_record(index) for index in range(record_count)),))
    output = resampler(torch.randn(1, SEMANTIC_DIM), retrieval)

    assert isinstance(output, StateResamplerOutput)
    assert output.hidden_states.shape == (1, 16, 512)
    assert output.state_tokens.shape == (1, 16, 4096)
    assert output.selected_record_ids == retrieval.selected_record_ids
    assert output.state_token_valid_mask.tolist() == [True]
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.state_tokens).all()


def test_resampler_is_permutation_invariant_reads_past_sixteen_and_never_mutates_inputs(
    resampler: StateResampler,
) -> None:
    records = tuple(_confirmed_record(index) for index in range(30))
    forward = _retrieval((records,))
    reverse = _retrieval((tuple(reversed(records)),))
    beyond_sixteen = _retrieval(
        (records[:-1] + (replace(records[-1], semantic_embedding=_unit_semantic(500)),),)
    )
    q_target = torch.randn(1, SEMANTIC_DIM)
    snapshot = forward.state_embeddings.clone()

    forward_output = resampler(q_target, forward)
    reverse_output = resampler(q_target, reverse)
    changed_output = resampler(q_target, beyond_sixteen)

    # Canonical evidence axis: candidate order cannot move the State Tokens.
    assert forward_output.selected_record_ids == reverse_output.selected_record_ids
    assert forward_output.selected_record_ids[0] == tuple(
        sorted(record.record_id for record in records)
    )
    torch.testing.assert_close(forward_output.hidden_states, reverse_output.hidden_states)
    torch.testing.assert_close(forward_output.state_tokens, reverse_output.state_tokens)
    # No Top-K truncation: the 30th record still moves the output.
    assert not torch.allclose(forward_output.state_tokens, changed_output.state_tokens)
    torch.testing.assert_close(forward.state_embeddings, snapshot)


def test_resampler_packs_only_the_noncontiguous_selected_subset_of_a_wide_candidate_axis(
    resampler: StateResampler,
) -> None:
    candidates = tuple(_confirmed_record(index) for index in range(6))
    selected = (candidates[1], candidates[4])
    full_axis = _retrieval((selected,), candidate_rows=(candidates,))
    selected_only = _retrieval((selected,))
    q_target = torch.randn(1, SEMANTIC_DIM)

    full_output = resampler(q_target, full_axis)
    selected_output = resampler(q_target, selected_only)

    assert int(full_axis.n_state[0].item()) == 6
    assert int(full_axis.n_retrieved[0].item()) == 2
    assert full_axis.selected_mask[0].tolist() == [False, True, False, False, True, False]
    assert full_output.selected_record_ids == selected_output.selected_record_ids
    torch.testing.assert_close(full_output.hidden_states, selected_output.hidden_states)
    torch.testing.assert_close(full_output.state_tokens, selected_output.state_tokens)


def test_resampler_packs_tensor_only_retrieval_candidate_axis(
    resampler: StateResampler,
) -> None:
    records = []
    for sequence in range(6):
        record_id = f"retrieval-{sequence:08d}"
        original = _confirmed_record(sequence)
        assert isinstance(original.payload, ConfirmedIdentity)
        records.append(
            replace(
                original,
                record_id=record_id,
                payload=replace(original.payload, semantic_record_id=record_id),
            )
        )
    candidates = tuple(records)
    eager = _retrieval(((candidates[1], candidates[4]),), candidate_rows=(candidates,))
    lazy = replace(
        eager,
        candidate_record_ids=((None,) * len(candidates),),
        candidate_records=((None,) * len(candidates),),
        candidate_head_types=((None,) * len(candidates),),
    )
    q_target = torch.randn(1, SEMANTIC_DIM)

    eager_output = resampler(q_target, eager)
    lazy_output = resampler(q_target, lazy)

    assert lazy_output.selected_record_ids == eager_output.selected_record_ids
    torch.testing.assert_close(lazy_output.hidden_states, eager_output.hidden_states)
    torch.testing.assert_close(lazy_output.state_tokens, eager_output.state_tokens)


def test_resampler_fail_closed_status_mask_separates_empty_from_unknown(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260716)
    module = build_state_resampler(config)
    retrieval = _retrieval(
        ((_confirmed_record(0),), (), (), ()),
        statuses=(
            RetrievalStatus.OK,
            RetrievalStatus.EMPTY,
            RetrievalStatus.UNSUPPORTED,
            RetrievalStatus.INVALID,
        ),
        reasons=(
            RetrievalReason.MATCHED,
            RetrievalReason.EMPTY_BANK,
            RetrievalReason.UNSUPPORTED_OPERATOR,
            RetrievalReason.INVALID_TIME,
        ),
        hard_operators=(
            Operator.O2_UNIQUE,
            Operator.O1_SNAP,
            Operator.UNSUPPORTED,
            Operator.UNSUPPORTED,
        ),
        time_resolutions=(
            _resolution(),
            _resolution(mode=TimeWindowMode.NOW, start_time=None),
            _resolution(status=TimeResolutionStatus.UNSUPPORTED),
            _resolution(status=TimeResolutionStatus.INVALID),
        ),
    )
    q_target = torch.randn(4, SEMANTIC_DIM, requires_grad=True)

    output = module(q_target, retrieval)

    assert output.retrieval_status == retrieval.status
    assert output.state_token_valid_mask.tolist() == [True, True, False, False]
    assert output.hidden_states[2:].eq(0.0).all()
    assert output.state_tokens[2:].eq(0.0).all()

    output.state_tokens.square().sum().backward()
    assert q_target.grad is not None
    assert q_target.grad[:2].abs().sum() > 0
    assert q_target.grad[2:].eq(0.0).all()
    assert module.empty_record_embedding.grad is not None
    assert module.empty_record_embedding.grad.abs().sum() > 0


def test_resampler_backpropagates_to_query_empty_embedding_and_every_parameter(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260715)
    module = build_state_resampler(config)
    retrieval = _retrieval(((_confirmed_record(0), _confirmed_record(1)), ()))
    q_target = torch.randn(2, SEMANTIC_DIM, requires_grad=True)

    output = module(q_target, retrieval)
    assert output.selected_record_ids[1] == ()
    assert output.state_token_valid_mask.tolist() == [True, True]
    assert torch.isfinite(output.state_tokens).all()
    (output.hidden_states.square().mean() + output.state_tokens.square().mean()).backward()

    assert q_target.grad is not None
    assert torch.isfinite(q_target.grad).all() and q_target.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_pinned_tokenizer_fixture_is_offline_and_strictly_roundtrips_numbers(
    config: ProjectConfig,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    snapshot = _tokenizer_snapshot(config)
    assert {path.name for path in snapshot.iterdir() if path.is_file()} >= TOKENIZER_FILES
    assert type(number_tokenizer).__name__ == config.state_reader.tokenizer_class
    assert number_tokenizer.vocab_size == config.state_reader.tokenizer_vocab_size
    expected_ids = {0: (15,), 7: (22,), 42: (19, 17), -3: (12, 18), 300: (18, 15, 15)}
    for value, expected in expected_ids.items():
        token_ids = serialize_number_token_ids(number_tokenizer, value)
        assert token_ids == expected
        assert number_tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) == str(value)


def test_reader_o1_snap_and_signed_o1_delta_emit_number_tokens(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    snap = _read_one(
        reader,
        Operator.O1_SNAP,
        _resolution(mode=TimeWindowMode.NOW, start_time=None),
        (_o1_record(3, 5),),
    )
    delta = _read_one(
        reader,
        Operator.O1_DELTA,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_o1_record(2, 5),),
    )

    _assert_number_tokens(number_tokenizer, snap, 3)
    _assert_number_tokens(number_tokenizer, delta, -3)
    assert snap.operator is Operator.O1_SNAP
    assert delta.operator is Operator.O1_DELTA


def test_reader_o2_counts_confirmed_identities_and_reports_canonical_evidence(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    records = tuple(
        _confirmed_record(index, first_seen=first_seen)
        for index, first_seen in enumerate((1.0, 7.0, 10.0))
    )
    unique = _read_one(reader, Operator.O2_UNIQUE, _resolution(), records)
    gain = _read_one(
        reader,
        Operator.O2_GAIN,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        records[1:],
    )

    _assert_number_tokens(number_tokenizer, unique, 3)
    _assert_number_tokens(number_tokenizer, gain, 2)
    assert unique.selected_record_ids == tuple(sorted(record.record_id for record in records))
    assert gain.selected_record_ids == tuple(sorted(record.record_id for record in records[1:]))
    assert unique.time_window.mode is TimeWindowMode.HISTORY
    audit = dict(unique.audit_fields)
    assert audit["operator"] == Operator.O2_UNIQUE.value
    assert audit["n_state"] == 3
    assert audit["n_retrieved"] == 3
    assert audit["reader_reason"] == "exact_typed_payload_arithmetic"


def test_reader_e1_uses_cumulative_history_and_a_closed_bounded_window(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    action = _e1_record(E1EventKind.ACTION, (5.0, 8.0, 9.0), event_count=6)

    history = _read_one(reader, Operator.E1_ACTION, _resolution(), (action,))
    bounded = _read_one(
        reader,
        Operator.E1_ACTION,
        _resolution(mode=TimeWindowMode.EXPLICIT_RANGE, start_time=5.0, end_time=8.0),
        (action,),
    )
    transit = _read_one(
        reader,
        Operator.E1_TRANSIT,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_e1_record(E1EventKind.TRANSIT, (4.0, 5.0, 10.0)),),
    )

    # HISTORY reads the cumulative counter; bounded windows count retained times,
    # inclusive on both edges.
    _assert_number_tokens(number_tokenizer, history, 6)
    _assert_number_tokens(number_tokenizer, bounded, 2)
    _assert_number_tokens(number_tokenizer, transit, 2)


def test_reader_e2_anchors_on_completion_end_and_ignores_the_active_interval(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    periodic = _read_one(
        reader,
        Operator.E2_PERIODIC,
        _resolution(),
        (_e2_record(E2EventKind.PERIODIC, ((0.0, 2.0), (2.0, 5.0), (8.0, 10.0))),),
    )
    # (2.0, 5.0) ends inside [5.0, 10.0] even though it starts before it.
    episode = _read_one(
        reader,
        Operator.E2_EPISODE,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_e2_record(E2EventKind.EPISODE, ((0.0, 2.0), (2.0, 5.0), (5.0, 7.0), (8.0, 10.0))),),
    )
    # An in-flight interval has no completion end, so it is never counted.
    active = _read_one(
        reader,
        Operator.E2_PERIODIC,
        _resolution(mode=TimeWindowMode.EXPLICIT_RANGE, start_time=5.0, end_time=8.0),
        (
            _e2_record(
                E2EventKind.PERIODIC,
                ((0.0, 5.0), (5.0, 8.0), (8.0, 9.0)),
                phase=E2Phase.ACTIVE,
                current_start=9.5,
            ),
        ),
    )

    _assert_number_tokens(number_tokenizer, periodic, 3)
    _assert_number_tokens(number_tokenizer, episode, 3)
    _assert_number_tokens(number_tokenizer, active, 2)


def test_reader_propagates_empty_unsupported_and_invalid_without_fabricating_counts(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    retrieval = _retrieval(
        ((), (), ()),
        statuses=(
            RetrievalStatus.EMPTY,
            RetrievalStatus.UNSUPPORTED,
            RetrievalStatus.INVALID,
        ),
        reasons=(
            RetrievalReason.EMPTY_BANK,
            RetrievalReason.UNSUPPORTED_OPERATOR,
            RetrievalReason.INVALID_TIME,
        ),
        hard_operators=(Operator.O1_SNAP, Operator.UNSUPPORTED, Operator.UNSUPPORTED),
        time_resolutions=(
            _resolution(mode=TimeWindowMode.NOW, start_time=None),
            _resolution(status=TimeResolutionStatus.UNSUPPORTED),
            _resolution(status=TimeResolutionStatus.INVALID),
        ),
    )
    results = reader.read(retrieval)

    assert tuple(result.status for result in results) == (
        ReaderStatus.EMPTY,
        ReaderStatus.UNSUPPORTED,
        ReaderStatus.INVALID,
    )
    # A reliably empty Bank is an exact zero with real number tokens, not a refusal.
    _assert_number_tokens(number_tokenizer, results[0], 0)
    for result in results[1:]:
        assert result.exact_count is None
        assert result.number_token_ids == ()


def test_reader_status_precedence_fails_closed_on_cross_layer_contradictions(
    reader: DeterministicStateReader,
) -> None:
    reliable_time = _resolution(mode=TimeWindowMode.NOW, start_time=None)
    retrieval = _retrieval(
        ((), (), ()),
        statuses=(
            RetrievalStatus.EMPTY,
            RetrievalStatus.UNSUPPORTED,
            RetrievalStatus.INVALID,
        ),
        reasons=(
            RetrievalReason.EMPTY_BANK,
            RetrievalReason.DEGENERATE_QUERY,
            RetrievalReason.INVALID_TIME,
        ),
        hard_operators=(Operator.O1_SNAP, Operator.O1_SNAP, Operator.O1_SNAP),
        time_resolutions=(
            _resolution(status=TimeResolutionStatus.INVALID),
            reliable_time,
            reliable_time,
        ),
    )

    results = reader.read(retrieval)

    # An "empty" retrieval whose time window is invalid must not become an exact zero.
    assert results[0].status is ReaderStatus.INVALID
    assert dict(results[0].audit_fields)["reader_reason"] == "inconsistent_empty_query_metadata"
    assert results[0].exact_count is None
    assert results[0].number_token_ids == ()
    assert results[1].status is ReaderStatus.UNSUPPORTED
    assert results[2].status is ReaderStatus.INVALID


def test_reader_surface_has_no_label_channel_and_results_are_frozen(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    forbidden = {
        "answer",
        "count",
        "ground_truth",
        "ground_truth_count",
        "occurrence_times",
        "counting_type",
        "counting_subtype",
    }
    assert forbidden.isdisjoint(inspect.signature(reader.read).parameters)
    assert (tuple(reader.parameters()) if isinstance(reader, nn.Module) else ()) == ()

    resolution = _resolution(mode=TimeWindowMode.NOW, start_time=None)
    retrieval = _retrieval(
        ((_o1_record(3, 1),),),
        hard_operators=(Operator.O1_SNAP,),
        time_resolutions=(resolution,),
    )
    result = reader.read(retrieval)[0]

    assert result.exact_count == 3
    assert reader.audit_number_tokens(result) == 3
    assert result.number_token_ids != serialize_number_token_ids(number_tokenizer, 99)
    with pytest.raises(FrozenInstanceError):
        result.exact_count = 99
    with pytest.raises(TypeError):
        reader.read(retrieval, (Operator.O1_SNAP,), (resolution,), ground_truth_count=99)


def test_state_token_mutation_cannot_change_deterministic_reader_count(
    reader: DeterministicStateReader,
    resampler: StateResampler,
) -> None:
    retrieval = _retrieval((tuple(_confirmed_record(index) for index in range(3)),))
    resolution = _resolution()
    before = reader.read(retrieval, (Operator.O2_UNIQUE,), (resolution,))[0]
    output = resampler(torch.randn(1, SEMANTIC_DIM), retrieval)

    with torch.no_grad():
        output.state_tokens.fill_(12345.0)
        output.hidden_states.mul_(-7.0)
    after = reader.read(retrieval, (Operator.O2_UNIQUE,), (resolution,))[0]

    assert before.exact_count == after.exact_count == 3
    assert before.number_token_ids == after.number_token_ids
    assert before.selected_record_ids == after.selected_record_ids
