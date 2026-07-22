from __future__ import annotations

import inspect
import math
import os
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tests.support import parameter_count
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.identity_bank import CandidateIdentity, ConfirmedIdentity
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
    decode_number_token_ids,
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
PINNED_TOKENIZER_BYTES = 11_491_943
PINNED_TOKENIZER_MANIFEST_SHA256 = (
    "ccd18347b6d6714d91d4c55b37ff05e473a0f8e84fbcba2bda1401a9572f44c3"
)
TOKENIZER_FILES = {
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
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
        (
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("F:/huggingface_cache/hub"),
        )
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


def _unit_identity(index: int = 0) -> Tensor:
    value = torch.zeros(IDENTITY_DIM, dtype=torch.float32)
    value[index % IDENTITY_DIM] = 1.0
    return value


def _confirmed_record(
    sequence: int,
    *,
    first_seen: float = 1.0,
    semantic_index: int | None = None,
) -> StateRecord:
    record_id = f"o2-{sequence:08d}"
    return StateRecord(
        record_id=record_id,
        video_id="video-0",
        trajectory_id="trajectory-0",
        head_type=HeadType.O2,
        semantic_embedding=_unit_semantic(sequence if semantic_index is None else semantic_index),
        timestamp=first_seen,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=ConfirmedIdentity(
            identity_id=f"identity-{sequence:08d}",
            identity_prototype=_unit_identity(sequence),
            first_seen=first_seen,
            last_seen=first_seen,
            observation_count=2,
            semantic_record_id=record_id,
        ),
    )


def _o1_record(
    current_count: int,
    baseline_count: int,
    *,
    baseline_initialized: bool = True,
) -> StateRecord:
    return StateRecord(
        record_id="o1-aggregate",
        video_id="video-0",
        trajectory_id="trajectory-0",
        head_type=HeadType.O1,
        semantic_embedding=_unit_semantic(401),
        timestamp=10.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=O1Payload(
            current_visible_count=current_count,
            baseline_count=baseline_count,
            active_slot_ids=tuple(range(current_count)),
            baseline_initialized=baseline_initialized,
            baseline_position_id=0 if baseline_initialized else None,
        ),
    )


def _e1_record(
    event_kind: E1EventKind,
    event_times: tuple[float, ...],
    *,
    event_count: int | None = None,
    history_eviction_count: int = 0,
) -> StateRecord:
    resolved_count = len(event_times) if event_count is None else event_count
    return StateRecord(
        record_id=f"e1-{event_kind.value}",
        video_id="video-0",
        trajectory_id="trajectory-0",
        head_type=HeadType.E1,
        semantic_embedding=_unit_semantic(402),
        timestamp=10.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=E1Payload(
            event_kind=event_kind,
            event_count=resolved_count,
            recent_event_times=event_times,
            cooldown_until=0.0,
            history_eviction_count=history_eviction_count,
        ),
    )


def _e2_record(
    event_kind: E2EventKind,
    intervals: tuple[tuple[float, float], ...],
) -> StateRecord:
    return StateRecord(
        record_id=f"e2-{event_kind.value}",
        video_id="video-0",
        trajectory_id="trajectory-0",
        head_type=HeadType.E2,
        semantic_embedding=_unit_semantic(403),
        timestamp=10.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=E2Payload(
            event_kind=event_kind,
            completed_count=len(intervals),
            phase=E2Phase.COMPLETED,
            completed_intervals=intervals,
            recent_event_times=(),
        ),
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
        reason=f"synthetic-{status.value}",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
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
    normalized_rows = tuple(tuple(row) for row in rows)
    batch_size = len(normalized_rows)
    normalized_candidates = (
        normalized_rows if candidate_rows is None else tuple(tuple(row) for row in candidate_rows)
    )
    assert len(normalized_candidates) == batch_size
    if statuses is None:
        normalized_statuses = tuple(
            RetrievalStatus.OK if row else RetrievalStatus.EMPTY for row in normalized_rows
        )
    else:
        normalized_statuses = tuple(statuses)
    if reasons is None:
        normalized_reasons = tuple(
            RetrievalReason.MATCHED if row else RetrievalReason.EMPTY_BANK
            for row in normalized_rows
        )
    else:
        normalized_reasons = tuple(reasons)
    if hard_operators is None:
        head_defaults = {
            HeadType.O1: Operator.O1_SNAP,
            HeadType.O2: Operator.O2_UNIQUE,
            HeadType.E1: Operator.E1_ACTION,
            HeadType.E2: Operator.E2_PERIODIC,
        }
        normalized_operators = tuple(
            head_defaults[candidates[0].head_type] if candidates else Operator.O1_SNAP
            for candidates in normalized_candidates
        )
    else:
        normalized_operators = tuple(hard_operators)
    if time_resolutions is None:
        normalized_resolutions = tuple(
            _resolution(
                status=(
                    TimeResolutionStatus.UNSUPPORTED
                    if status is RetrievalStatus.UNSUPPORTED
                    else TimeResolutionStatus.INVALID
                    if status is RetrievalStatus.INVALID
                    else TimeResolutionStatus.OK
                )
            )
            for status in normalized_statuses
        )
    else:
        normalized_resolutions = tuple(time_resolutions)
    assert len(normalized_statuses) == batch_size
    assert len(normalized_reasons) == batch_size
    assert len(normalized_operators) == batch_size
    assert len(normalized_resolutions) == batch_size

    max_records = max((len(row) for row in normalized_candidates), default=0)
    embeddings = torch.zeros(batch_size, max_records, SEMANTIC_DIM, dtype=torch.float32)
    scores = torch.zeros(batch_size, max_records, dtype=torch.float32)
    present_mask = torch.zeros(batch_size, max_records, dtype=torch.bool)
    record_valid_mask = torch.zeros_like(present_mask)
    retrieval_eligible_mask = torch.zeros_like(present_mask)
    selected_mask = torch.zeros_like(present_mask)
    candidate_ids: list[tuple[str | None, ...]] = []
    candidate_snapshots: list[tuple[StateRecord | None, ...]] = []
    selected_ids: list[tuple[str, ...]] = []
    selected_scores: list[tuple[float, ...]] = []
    selected_records: list[tuple[StateRecord, ...]] = []
    audits: list[RetrievalFilterAudit] = []
    n_state = torch.zeros(batch_size, dtype=torch.int64)
    n_retrieved = torch.zeros(batch_size, dtype=torch.int64)

    for row_index, raw_selected_records in enumerate(normalized_rows):
        status = normalized_statuses[row_index]
        assert (status is RetrievalStatus.OK) == bool(raw_selected_records)
        video_id = f"video-{row_index}"
        trajectory_id = f"trajectory-{row_index}"
        candidates = tuple(
            replace(record, video_id=video_id, trajectory_id=trajectory_id)
            for record in normalized_candidates[row_index]
        )
        candidate_by_id = {record.record_id: record for record in candidates}
        records = tuple(candidate_by_id[record.record_id] for record in raw_selected_records)
        selected_id_set = {record.record_id for record in records}
        for column, record in enumerate(candidates):
            embeddings[row_index, column] = record.semantic_embedding
            scores[row_index, column] = 0.8 if record.record_id in selected_id_set else 0.1
            present_mask[row_index, column] = True
            record_valid_mask[row_index, column] = record.valid
            retrieval_eligible_mask[row_index, column] = record.valid and not isinstance(
                record.payload, CandidateIdentity
            )
            selected_mask[row_index, column] = record.record_id in selected_id_set
        ids_on_candidate_axis = tuple(record.record_id for record in candidates)
        candidate_ids.append(ids_on_candidate_axis + (None,) * (max_records - len(candidates)))
        candidate_snapshots.append(candidates + (None,) * (max_records - len(candidates)))
        canonical_records = tuple(
            sorted(
                records,
                key=lambda record: (
                    -float(scores[row_index, ids_on_candidate_axis.index(record.record_id)].item()),
                    record.record_id,
                ),
            )
        )
        canonical_ids = tuple(record.record_id for record in canonical_records)
        selected_ids.append(canonical_ids)
        score_by_id = {
            record.record_id: float(scores[row_index, column].item())
            for column, record in enumerate(candidates)
        }
        selected_scores.append(tuple(score_by_id[record_id] for record_id in canonical_ids))
        selected_records.append(canonical_records)
        n_state[row_index] = len(candidates)
        n_retrieved[row_index] = len(records)
        audits.append(
            RetrievalFilterAudit(
                n_state=len(candidates),
                head_partition_excluded_count=0,
                query_rejected_count=0,
                owner_mismatch_count=0,
                invalid_count=0,
                retrieval_ineligible_count=0,
                future_count=0,
                outside_window_count=0,
                below_similarity_count=len(candidates) - len(records),
                selected_count=len(records),
            )
        )

    return RetrieverOutput(
        selected_record_ids=tuple(selected_ids),
        selected_scores=tuple(selected_scores),
        selected_records=tuple(selected_records),
        candidate_record_ids=tuple(candidate_ids),
        candidate_records=tuple(candidate_snapshots),
        state_embeddings=embeddings,
        scores=scores,
        present_mask=present_mask,
        record_valid_mask=record_valid_mask,
        retrieval_eligible_mask=retrieval_eligible_mask,
        causal_mask=present_mask.clone(),
        selected_mask=selected_mask,
        status=normalized_statuses,
        reason=normalized_reasons,
        hard_operators=normalized_operators,
        time_resolutions=normalized_resolutions,
        n_state=n_state,
        n_retrieved=n_retrieved,
        audit=tuple(audits),
        video_ids=tuple(f"video-{row}" for row in range(batch_size)),
        trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch_size)),
        bank_video_ids=tuple(f"video-{row}" for row in range(batch_size)),
        bank_trajectory_ids=tuple(f"trajectory-{row}" for row in range(batch_size)),
        bank_versions=tuple(range(batch_size)),
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


def _assert_number_roundtrip(
    tokenizer: PreTrainedTokenizerBase,
    result: ReaderResult,
) -> None:
    assert result.exact_count is not None
    decoded = decode_number_token_ids(tokenizer, result.number_token_ids)
    assert decoded == result.exact_count
    assert tokenizer.decode(
        list(result.number_token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == str(result.exact_count)
    assert tuple(tokenizer.encode(str(decoded), add_special_tokens=False)) == (
        result.number_token_ids
    )
    audit = dict(result.audit_fields)
    assert audit["computed_exact_count"] == result.exact_count
    assert audit["number_text"] == str(result.exact_count)
    assert audit["input_record_count"] == len(result.selected_record_ids)
    assert audit["n_retrieved"] == len(result.selected_record_ids)


def test_meta_topology_and_exact_state_resampler_parameter_count(
    config: ProjectConfig,
) -> None:
    with torch.device("meta"):
        module = build_state_resampler(config)

    assert isinstance(module, StateResampler)
    assert module.q_state.shape == (16, 512)
    assert module.empty_record_embedding.shape == (512,)
    assert len(module.layers) == 3
    assert parameter_count(module) == EXACT_RESAMPLER_PARAMETERS
    assert sum(parameter.numel() for parameter in module.parameters()) == (
        EXACT_RESAMPLER_PARAMETERS
    )


def test_p12_builders_reject_missing_config_tokenizer_and_contract_drift(
    config: ProjectConfig,
    number_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="validated ProjectConfig"):
        build_state_resampler()
    with pytest.raises(ValueError, match="validated ProjectConfig"):
        build_state_reader(tokenizer=number_tokenizer)
    with pytest.raises(ValueError, match="pinned tokenizer"):
        build_state_reader(config)

    bad_resampler = config.state_resampler.model_copy(update={"parameter_count": 14_720_000})
    bad_resampler_config = config.model_copy(update={"state_resampler": bad_resampler})
    with pytest.raises(ValueError, match="State Resampler parameter_count"):
        build_state_resampler(bad_resampler_config)

    bad_reader = config.state_reader.model_copy(update={"ground_truth_input_forbidden": False})
    bad_reader_config = config.model_copy(update={"state_reader": bad_reader})
    with pytest.raises(ValueError, match="State Reader ground_truth_input_forbidden"):
        build_state_reader(bad_reader_config, tokenizer=number_tokenizer)

    fake_type = type("Qwen2TokenizerFast", (), {})
    fake_tokenizer = fake_type()
    fake_tokenizer.name_or_path = str(tmp_path)
    fake_tokenizer.vocab_size = config.state_reader.tokenizer_vocab_size
    with pytest.raises(ValueError, match="missing merges.txt"):
        build_state_reader(config, tokenizer=fake_tokenizer)


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
    assert output.record_mask.shape == (1, record_count)
    assert output.cross_attention_weights.shape == (1, 16, record_count)
    assert output.selected_attention_mass.shape == (1, 16)
    assert output.selected_record_ids == retrieval.selected_record_ids
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.state_tokens).all()


def test_mixed_ragged_attention_mask_mass_and_empty_row_are_exact(
    resampler: StateResampler,
) -> None:
    counts = (3, 0, 30)
    retrieval = _retrieval(
        tuple(tuple(_confirmed_record(index) for index in range(count)) for count in counts)
    )
    output = resampler(torch.randn(3, SEMANTIC_DIM), retrieval)

    assert output.record_mask.shape == (3, 30)
    assert output.cross_attention_weights.shape == (3, 16, 30)
    torch.testing.assert_close(
        output.record_mask.sum(dim=1),
        torch.tensor(counts, dtype=torch.int64),
    )
    expanded_mask = output.record_mask[:, None, :].expand_as(output.cross_attention_weights)
    assert output.cross_attention_weights.masked_select(~expanded_mask).eq(0.0).all()
    expected_mass = torch.tensor([1.0, 0.0, 1.0])[:, None].expand(-1, 16)
    torch.testing.assert_close(output.selected_attention_mass, expected_mass)
    torch.testing.assert_close(output.cross_attention_weights.sum(dim=-1), expected_mass)
    assert torch.isfinite(output.cross_attention_weights).all()


def test_cross_attention_matches_frozen_scaled_fp32_masked_softmax_formula(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260717)
    layer = build_state_resampler(config).to(dtype=torch.bfloat16).layers[0]
    queries = torch.randn(2, 16, SEMANTIC_DIM, dtype=torch.bfloat16)
    records = torch.randn(2, 4, SEMANTIC_DIM, dtype=torch.bfloat16)
    mask = torch.tensor(
        [[True, True, True, False], [True, False, False, False]],
        dtype=torch.bool,
    )

    _, observed = layer(queries, records, mask)

    normalized = layer.self_norm(queries)
    self_queries = layer._split_heads(layer.self_q(normalized))
    self_keys = layer._split_heads(layer.self_k(normalized))
    self_values = layer._split_heads(layer.self_v(normalized))
    self_logits = torch.matmul(self_queries.float(), self_keys.float().transpose(-1, -2))
    self_weights = torch.softmax(self_logits / math.sqrt(layer.head_dim), dim=-1)
    self_context = torch.matmul(self_weights.to(self_values.dtype), self_values)
    after_self = queries + layer.self_out(layer._merge_heads(self_context))
    cross_queries = layer._split_heads(layer.cross_q(layer.cross_norm(after_self)))
    cross_keys = layer._split_heads(layer.cross_k(records))
    logits = torch.matmul(cross_queries.float(), cross_keys.float().transpose(-1, -2))
    logits = logits / math.sqrt(layer.head_dim)
    valid_pairs = mask[:, None, None, :]
    logits = logits.masked_fill(~valid_pairs, torch.finfo(torch.float32).min)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(valid_pairs, weights, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = weights.mean(dim=1)

    assert observed.dtype is torch.float32
    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)
    assert observed[:, :, -1].eq(0.0).all()
    uniform = torch.full_like(observed[0, :, :3], 1.0 / 3.0)
    assert not torch.allclose(observed[0, :, :3], uniform)


def test_all_empty_batch_has_zero_width_audit_no_nan_and_trainable_empty_embedding(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260714)
    module = build_state_resampler(config)
    retrieval = _retrieval(((), ()))
    q_target = torch.randn(2, SEMANTIC_DIM, requires_grad=True)
    output = module(q_target, retrieval)

    assert output.record_mask.shape == (2, 0)
    assert output.cross_attention_weights.shape == (2, 16, 0)
    assert output.selected_record_ids == ((), ())
    assert output.selected_attention_mass.eq(0.0).all()
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.state_tokens).all()

    output.state_tokens.square().mean().backward()
    assert module.empty_record_embedding.grad is not None
    assert torch.isfinite(module.empty_record_embedding.grad).all()
    assert module.empty_record_embedding.grad.abs().sum() > 0
    assert q_target.grad is not None and q_target.grad.abs().sum() > 0


def test_resampler_is_candidate_permutation_invariant_and_uses_canonical_selected_axis(
    resampler: StateResampler,
) -> None:
    records = tuple(_confirmed_record(index) for index in range(30))
    forward = _retrieval((records,))
    reverse = _retrieval((tuple(reversed(records)),))
    q_target = torch.randn(1, SEMANTIC_DIM)

    forward_output = resampler(q_target, forward)
    reverse_output = resampler(q_target, reverse)

    assert forward_output.selected_record_ids == reverse_output.selected_record_ids
    assert forward_output.selected_record_ids[0] == tuple(
        sorted(record.record_id for record in records)
    )
    torch.testing.assert_close(forward_output.hidden_states, reverse_output.hidden_states)
    torch.testing.assert_close(forward_output.state_tokens, reverse_output.state_tokens)
    torch.testing.assert_close(
        forward_output.cross_attention_weights,
        reverse_output.cross_attention_weights,
    )


def test_resampler_packs_only_noncontiguous_selected_subset_when_n_state_exceeds_n_ret(
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
    assert full_output.record_mask.shape == (1, 2)
    assert full_output.selected_record_ids == selected_output.selected_record_ids
    torch.testing.assert_close(full_output.hidden_states, selected_output.hidden_states)
    torch.testing.assert_close(full_output.state_tokens, selected_output.state_tokens)
    torch.testing.assert_close(
        full_output.cross_attention_weights,
        selected_output.cross_attention_weights,
    )


def test_resampler_fail_closed_status_mask_separates_empty_from_unknown(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260716)
    module = build_state_resampler(config)
    operators = (
        Operator.O2_UNIQUE,
        Operator.O1_SNAP,
        Operator.UNSUPPORTED,
        Operator.UNSUPPORTED,
    )
    resolutions = (
        _resolution(),
        _resolution(mode=TimeWindowMode.NOW, start_time=None),
        _resolution(status=TimeResolutionStatus.UNSUPPORTED),
        _resolution(status=TimeResolutionStatus.INVALID),
    )
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
        hard_operators=operators,
        time_resolutions=resolutions,
    )
    q_target = torch.randn(4, SEMANTIC_DIM, requires_grad=True)

    output = module(q_target, retrieval)

    assert output.retrieval_status == retrieval.status
    assert output.state_token_valid_mask.tolist() == [True, True, False, False]
    assert output.hidden_states[2:].eq(0.0).all()
    assert output.state_tokens[2:].eq(0.0).all()
    assert output.selected_attention_mass[0].eq(1.0).all()
    assert output.selected_attention_mass[1:].eq(0.0).all()

    output.state_tokens.square().sum().backward()
    assert q_target.grad is not None
    assert q_target.grad[:2].abs().sum() > 0
    assert q_target.grad[2:].eq(0.0).all()
    assert module.empty_record_embedding.grad is not None
    assert module.empty_record_embedding.grad.abs().sum() > 0


def test_resampler_uses_records_beyond_sixteen_and_never_aliases_inputs(
    resampler: StateResampler,
) -> None:
    records = tuple(_confirmed_record(index) for index in range(30))
    changed = records[:-1] + (replace(records[-1], semantic_embedding=_unit_semantic(500)),)
    original_retrieval = _retrieval((records,))
    changed_retrieval = _retrieval((changed,))
    q_target = torch.randn(1, SEMANTIC_DIM)
    original_snapshot = original_retrieval.state_embeddings.clone()

    original = resampler(q_target, original_retrieval)
    modified = resampler(q_target, changed_retrieval)

    assert not torch.allclose(original.hidden_states, modified.hidden_states)
    assert not torch.allclose(original.state_tokens, modified.state_tokens)
    assert original.hidden_states.untyped_storage().data_ptr() != (
        original_retrieval.state_embeddings.untyped_storage().data_ptr()
    )
    assert original.state_tokens.untyped_storage().data_ptr() != (
        original.hidden_states.untyped_storage().data_ptr()
    )
    assert original.hidden_states.untyped_storage().data_ptr() != (
        q_target.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(original_retrieval.state_embeddings, original_snapshot)


def test_resampler_backpropagates_to_query_and_every_trainable_parameter(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(20260715)
    module = build_state_resampler(config)
    retrieval = _retrieval(((_confirmed_record(0), _confirmed_record(1), _confirmed_record(2)), ()))
    q_target = torch.randn(2, SEMANTIC_DIM, requires_grad=True)

    output = module(q_target, retrieval)
    (output.hidden_states.square().mean() + output.state_tokens.square().mean()).backward()

    assert q_target.grad is not None
    assert torch.isfinite(q_target.grad).all() and q_target.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_resampler_rejects_wrong_query_shape_batch_and_non_finite_input(
    resampler: StateResampler,
) -> None:
    retrieval = _retrieval(((_confirmed_record(0),),))

    with pytest.raises(ValueError, match=r"\[B, 512\]"):
        resampler(torch.randn(1, 511), retrieval)
    with pytest.raises(ValueError, match="batch"):
        resampler(torch.randn(2, 512), retrieval)
    invalid = torch.randn(1, 512)
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        resampler(invalid, retrieval)

    mixed_dtype = resampler(torch.randn(1, 512, dtype=torch.bfloat16), retrieval)
    assert mixed_dtype.hidden_states.dtype is resampler.q_state.dtype


def test_pinned_tokenizer_fixture_is_small_offline_and_strictly_roundtrips_numbers(
    config: ProjectConfig,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    snapshot = _tokenizer_snapshot(config)
    files = {path.name for path in snapshot.iterdir() if path.is_file()}
    assert files >= TOKENIZER_FILES
    assert sum((snapshot / filename).stat().st_size for filename in TOKENIZER_FILES) == (
        PINNED_TOKENIZER_BYTES
    )
    manifest = sha256()
    for filename in sorted(TOKENIZER_FILES):
        manifest.update(filename.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update((snapshot / filename).read_bytes())
        manifest.update(b"\0")
    assert manifest.hexdigest() == PINNED_TOKENIZER_MANIFEST_SHA256
    assert config.state_reader.tokenizer_manifest_sha256 == manifest.hexdigest()
    assert type(number_tokenizer).__name__ == config.state_reader.tokenizer_class
    assert number_tokenizer.vocab_size == config.state_reader.tokenizer_vocab_size
    expected_ids = {
        0: (15,),
        7: (22,),
        42: (19, 17),
        -3: (12, 18),
        300: (18, 15, 15),
    }
    for value, expected in expected_ids.items():
        token_ids = serialize_number_token_ids(number_tokenizer, value)
        assert token_ids == expected
        assert decode_number_token_ids(number_tokenizer, token_ids) == value


def test_reader_o1_snap_uses_current_visible_count(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.O1_SNAP,
        _resolution(mode=TimeWindowMode.NOW, start_time=None),
        (_o1_record(3, 5),),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 3
    assert dict(result.audit_fields)["operand_current_visible_count"] == 3
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_o1_delta_supports_signed_exact_count(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.O1_DELTA,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_o1_record(2, 5),),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == -3
    audit = dict(result.audit_fields)
    assert audit["baseline_policy"] == "fixed_baseline_v1"
    assert audit["operand_current_visible_count"] == 2
    assert audit["operand_baseline_count"] == 5
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_o2_unique_counts_distinct_confirmed_identities_at_query_time(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    records = tuple(
        _confirmed_record(index, first_seen=first_seen)
        for index, first_seen in enumerate((1.0, 7.0, 10.0))
    )
    result = _read_one(reader, Operator.O2_UNIQUE, _resolution(), records)

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 3
    assert result.selected_record_ids == tuple(sorted(record.record_id for record in records))
    assert dict(result.audit_fields)["matched_first_seen_count"] == 3
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_o2_gain_uses_closed_first_seen_window_boundaries(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    records = tuple(
        _confirmed_record(index, first_seen=first_seen)
        for index, first_seen in enumerate((5.0, 10.0))
    )
    result = _read_one(
        reader,
        Operator.O2_GAIN,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        records,
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 2
    assert dict(result.audit_fields)["matched_first_seen_count"] == 2
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_e1_action_uses_action_completion_history(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.E1_ACTION,
        _resolution(),
        (_e1_record(E1EventKind.ACTION, (1.0, 5.0, 10.0)),),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 3
    assert dict(result.audit_fields)["operand_cumulative_event_count"] == 3
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_rejects_o2_records_that_contradict_retriever_time_filters(
    reader: DeterministicStateReader,
) -> None:
    future = _read_one(
        reader,
        Operator.O2_UNIQUE,
        _resolution(mode=TimeWindowMode.HISTORY),
        (_confirmed_record(900, first_seen=11.0),),
    )
    outside = _read_one(
        reader,
        Operator.O2_GAIN,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_confirmed_record(901, first_seen=4.0),),
    )

    assert future.status is ReaderStatus.INVALID
    assert dict(future.audit_fields)["reader_reason"] == "o2_future_identity_reached_reader"
    assert outside.status is ReaderStatus.INVALID
    assert dict(outside.audit_fields)["reader_reason"] == "o2_identity_outside_gain_window"


def test_reader_e1_transit_uses_closed_completion_time_window(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.E1_TRANSIT,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (_e1_record(E1EventKind.TRANSIT, (4.0, 5.0, 10.0)),),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 2
    assert dict(result.audit_fields)["matched_completion_count"] == 2
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_e2_periodic_counts_completed_interval_endpoints(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.E2_PERIODIC,
        _resolution(),
        (_e2_record(E2EventKind.PERIODIC, ((0.0, 2.0), (2.0, 5.0), (8.0, 10.0))),),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 3
    assert dict(result.audit_fields)["matched_completion_end_count"] == 3
    _assert_number_roundtrip(number_tokenizer, result)


def test_reader_e2_episode_filters_by_completion_end_not_interval_start(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    result = _read_one(
        reader,
        Operator.E2_EPISODE,
        _resolution(mode=TimeWindowMode.RECENT, start_time=5.0),
        (
            _e2_record(
                E2EventKind.EPISODE,
                ((0.0, 2.0), (2.0, 5.0), (5.0, 7.0), (8.0, 10.0)),
            ),
        ),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 3
    assert dict(result.audit_fields)["operand_completed_interval_count"] == 4
    _assert_number_roundtrip(number_tokenizer, result)


@pytest.mark.parametrize(
    ("operator", "record"),
    (
        (Operator.E1_ACTION, _e1_record(E1EventKind.TRANSIT, (1.0,))),
        (Operator.E1_TRANSIT, _e1_record(E1EventKind.ACTION, (1.0,))),
        (Operator.E2_PERIODIC, _e2_record(E2EventKind.EPISODE, ((0.0, 1.0),))),
        (Operator.E2_EPISODE, _e2_record(E2EventKind.PERIODIC, ((0.0, 1.0),))),
    ),
)
def test_reader_event_kind_mismatch_fails_closed_as_invalid(
    reader: DeterministicStateReader,
    operator: Operator,
    record: StateRecord,
) -> None:
    result = _read_one(reader, operator, _resolution(), (record,))

    assert result.status is ReaderStatus.INVALID
    assert result.exact_count is None
    assert result.number_token_ids == ()
    assert "event_kind_mismatch" in dict(result.audit_fields)["reader_reason"]


def test_reader_rejects_truncated_e1_history_for_an_unsafe_bounded_window(
    reader: DeterministicStateReader,
) -> None:
    record = _e1_record(
        E1EventKind.ACTION,
        (5.0, 9.0),
        event_count=3,
        history_eviction_count=1,
    )
    result = _read_one(
        reader,
        Operator.E1_ACTION,
        _resolution(mode=TimeWindowMode.RECENT, start_time=4.0),
        (record,),
    )

    assert result.status is ReaderStatus.INVALID
    assert result.exact_count is None
    assert dict(result.audit_fields)["reader_reason"] == "e1_window_history_truncated"


def test_reader_e1_truncated_history_uses_cumulative_and_safe_boundary_is_closed(
    reader: DeterministicStateReader,
) -> None:
    record = _e1_record(
        E1EventKind.ACTION,
        (5.0, 8.0, 9.0),
        event_count=6,
        history_eviction_count=3,
    )

    history = _read_one(reader, Operator.E1_ACTION, _resolution(), (record,))
    bounded = _read_one(
        reader,
        Operator.E1_ACTION,
        _resolution(
            mode=TimeWindowMode.EXPLICIT_RANGE,
            start_time=5.0,
            end_time=8.0,
        ),
        (record,),
    )

    assert history.status is ReaderStatus.OK
    assert history.exact_count == 6
    assert dict(history.audit_fields)["operand_history_eviction_count"] == 3
    assert bounded.status is ReaderStatus.OK
    assert bounded.exact_count == 2
    assert dict(bounded.audit_fields)["matched_completion_count"] == 2


def test_reader_rejects_candidate_duplicate_identity_and_duplicate_aggregate(
    reader: DeterministicStateReader,
) -> None:
    confirmed = _confirmed_record(0)
    candidate = replace(
        _confirmed_record(1),
        payload=CandidateIdentity(
            candidate_id="candidate-1",
            identity_prototype=_unit_identity(1),
            observation_count=1,
            ttl_remaining=8,
            confidence=0.9,
            first_seen=1.0,
            last_seen=1.0,
            semantic_record_id="o2-00000001",
        ),
    )
    duplicate_identity = replace(
        _confirmed_record(2),
        payload=replace(
            _confirmed_record(2).payload,
            identity_id=confirmed.payload.identity_id,
        ),
    )
    second_o1 = replace(_o1_record(2, 1), record_id="o1-aggregate-copy")

    candidate_result = _read_one(reader, Operator.O2_UNIQUE, _resolution(), (candidate,))
    duplicate_result = _read_one(
        reader,
        Operator.O2_UNIQUE,
        _resolution(),
        (confirmed, duplicate_identity),
    )
    aggregate_result = _read_one(
        reader,
        Operator.O1_SNAP,
        _resolution(mode=TimeWindowMode.NOW, start_time=None),
        (_o1_record(2, 1), second_o1),
    )

    assert dict(candidate_result.audit_fields)["reader_reason"] == "o2_candidate_reached_reader"
    assert dict(duplicate_result.audit_fields)["reader_reason"] == "duplicate_confirmed_identity"
    assert (
        dict(aggregate_result.audit_fields)["reader_reason"]
        == "o1_aggregate_requires_exactly_one_record"
    )


def test_reader_e2_excludes_upper_outside_and_active_incomplete_interval(
    reader: DeterministicStateReader,
) -> None:
    record = _e2_record(
        E2EventKind.PERIODIC,
        ((0.0, 5.0), (5.0, 8.0), (8.0, 9.0)),
    )
    assert isinstance(record.payload, E2Payload)
    active = replace(
        record,
        payload=replace(record.payload, phase=E2Phase.ACTIVE, current_start=9.5),
    )
    result = _read_one(
        reader,
        Operator.E2_PERIODIC,
        _resolution(
            mode=TimeWindowMode.EXPLICIT_RANGE,
            start_time=5.0,
            end_time=8.0,
        ),
        (active,),
    )

    assert result.status is ReaderStatus.OK
    assert result.exact_count == 2
    audit = dict(result.audit_fields)
    assert audit["operand_completed_interval_count"] == 3
    assert audit["operand_active_interval_present"] is True
    assert audit["matched_completion_end_count"] == 2


def test_reader_propagates_empty_unsupported_and_invalid_without_fabricating_counts(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    operators = (Operator.O1_SNAP, Operator.UNSUPPORTED, Operator.UNSUPPORTED)
    resolutions = (
        _resolution(mode=TimeWindowMode.NOW, start_time=None),
        _resolution(status=TimeResolutionStatus.UNSUPPORTED),
        _resolution(status=TimeResolutionStatus.INVALID),
    )
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
        hard_operators=operators,
        time_resolutions=resolutions,
    )
    results = reader.read(retrieval)

    assert tuple(result.status for result in results) == (
        ReaderStatus.EMPTY,
        ReaderStatus.UNSUPPORTED,
        ReaderStatus.INVALID,
    )
    assert results[0].exact_count == 0
    _assert_number_roundtrip(number_tokenizer, results[0])
    for result in results[1:]:
        assert result.exact_count is None
        assert result.number_token_ids == ()


def test_reader_status_precedence_fails_closed_on_cross_layer_contradictions(
    reader: DeterministicStateReader,
) -> None:
    invalid_time = _resolution(status=TimeResolutionStatus.INVALID)
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
        time_resolutions=(invalid_time, reliable_time, reliable_time),
    )

    results = reader.read(retrieval)

    assert results[0].status is ReaderStatus.INVALID
    assert dict(results[0].audit_fields)["reader_reason"] == "inconsistent_empty_query_metadata"
    assert results[1].status is ReaderStatus.UNSUPPORTED
    assert results[2].status is ReaderStatus.INVALID


def test_reader_invalid_state_and_input_lengths_fail_closed(
    reader: DeterministicStateReader,
) -> None:
    resolution = _resolution(mode=TimeWindowMode.RECENT, start_time=5.0)
    retrieval = _retrieval(
        ((_o1_record(1, 0, baseline_initialized=False),),),
        hard_operators=(Operator.O1_DELTA,),
        time_resolutions=(resolution,),
    )
    result = reader.read(retrieval)[0]
    assert result.status is ReaderStatus.INVALID
    assert dict(result.audit_fields)["reader_reason"] == "o1_baseline_uninitialized"

    with pytest.raises(ValueError, match="hard_operators"):
        reader.read(retrieval, (), (resolution,))
    with pytest.raises(ValueError, match="time_resolutions"):
        reader.read(retrieval, (Operator.O1_DELTA,), ())


def test_reader_rejects_operator_and_window_rewrites_after_retrieval(
    reader: DeterministicStateReader,
) -> None:
    resolution = _resolution(mode=TimeWindowMode.NOW, start_time=None)
    retrieval = _retrieval(
        ((_o1_record(3, 1),),),
        hard_operators=(Operator.O1_SNAP,),
        time_resolutions=(resolution,),
    )

    with pytest.raises(ValueError, match="hard_operators.*provenance"):
        reader.read(retrieval, (Operator.O1_DELTA,), (resolution,))
    with pytest.raises(ValueError, match="time_resolutions.*provenance"):
        reader.read(
            retrieval,
            (Operator.O1_SNAP,),
            (_resolution(mode=TimeWindowMode.HISTORY),),
        )


def test_retriever_snapshot_rejects_typed_payload_replacement(
    reader: DeterministicStateReader,
) -> None:
    resolution = _resolution(mode=TimeWindowMode.NOW, start_time=None)
    retrieval = _retrieval(
        ((_o1_record(3, 1),),),
        hard_operators=(Operator.O1_SNAP,),
        time_resolutions=(resolution,),
    )
    original = retrieval.selected_records[0][0]
    replacement = replace(original, payload=_o1_record(9, 1).payload)

    with pytest.raises(ValueError, match="selected typed records.*candidate snapshot"):
        replace(retrieval, selected_records=((replacement,),))

def test_number_tokens_are_reader_owned_gt_substitution_is_detected_and_result_is_frozen(
    reader: DeterministicStateReader,
    number_tokenizer: PreTrainedTokenizerBase,
) -> None:
    resolution = _resolution(mode=TimeWindowMode.NOW, start_time=None)
    retrieval = _retrieval(
        ((_o1_record(3, 1),),),
        hard_operators=(Operator.O1_SNAP,),
        time_resolutions=(resolution,),
    )
    result = reader.read(retrieval)[0]
    fake_ground_truth = 99
    fake_ids = serialize_number_token_ids(number_tokenizer, fake_ground_truth)

    assert result.exact_count == 3
    assert result.number_token_ids != fake_ids
    assert decode_number_token_ids(number_tokenizer, fake_ids) != result.exact_count
    assert reader.audit_number_tokens(result) == 3
    with pytest.raises(ValueError, match="authoritative exact_count"):
        reader.audit_number_tokens(replace(result, number_token_ids=fake_ids))
    with pytest.raises(ValueError, match="do not reproduce exact_count"):
        replace(result, exact_count=fake_ground_truth, number_token_ids=fake_ids)
    replacements = {
        "operand_current_visible_count": fake_ground_truth,
        "computed_exact_count": fake_ground_truth,
        "number_text": str(fake_ground_truth),
    }
    forged = replace(
        result,
        exact_count=fake_ground_truth,
        number_token_ids=fake_ids,
        audit_fields=tuple(
            (key, replacements.get(key, value)) for key, value in result.audit_fields
        ),
    )
    assert reader.audit_number_tokens(forged) == fake_ground_truth
    with pytest.raises(ValueError, match="authoritative retrieved-record arithmetic"):
        reader.audit_results(retrieval, (forged,))
    assert reader.audit_results(retrieval, (result,)) == (result,)
    with pytest.raises(FrozenInstanceError):
        result.exact_count = fake_ground_truth
    with pytest.raises(TypeError):
        reader.read(
            retrieval,
            (Operator.O1_SNAP,),
            (resolution,),
            ground_truth_count=fake_ground_truth,
        )


def test_reader_surface_has_no_labels_or_learned_counter_parameters(
    reader: DeterministicStateReader,
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
    parameters = tuple(reader.parameters()) if isinstance(reader, nn.Module) else ()
    assert parameters == ()


def test_state_token_mutation_cannot_change_deterministic_reader_count(
    reader: DeterministicStateReader,
    resampler: StateResampler,
) -> None:
    records = tuple(_confirmed_record(index) for index in range(3))
    retrieval = _retrieval((records,))
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
