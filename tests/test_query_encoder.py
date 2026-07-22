from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import QueryEncoderConfig, load_config
from ttt_svcbench_qwen.data import (
    RUNTIME_DENYLIST,
    RuntimeQueryInput,
    assert_runtime_payload_safe,
    extract_explicit_time_values,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_EVENT_KIND,
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    OperatorRouter,
    QueryEmbeddingEncoder,
    QueryEncoder,
    QueryEncoderInput,
    QueryEncoderSupervision,
    TimeResolutionStatus,
    TimeResolverLogits,
    TimeWindowMode,
    TimeWindowResolver,
    build_query_encoder,
    embed_question_tokens,
    operator_router_parameter_count,
    query_embedding_parameter_count,
    time_resolver_parameter_count,
)
from ttt_svcbench_qwen.query_tokens import QuestionTokenBatch, QuestionTokenSpan
from ttt_svcbench_qwen.state_bank import E1EventKind, E2EventKind, HeadType


def make_char_token_batch(questions: tuple[str, ...]) -> QuestionTokenBatch:
    width = max(len(question) for question in questions)
    input_ids = torch.zeros(len(questions), width, dtype=torch.int64)
    attention_mask = torch.zeros_like(input_ids)
    offset_mapping = torch.zeros(len(questions), width, 2, dtype=torch.int64)
    spans = []
    for row, question in enumerate(questions):
        length = len(question)
        input_ids[row, :length] = torch.arange(1, length + 1)
        attention_mask[row, :length] = 1
        offset_mapping[row, :length, 0] = torch.arange(length)
        offset_mapping[row, :length, 1] = torch.arange(1, length + 1)
        spans.append(QuestionTokenSpan(0, length))
    return QuestionTokenBatch(
        questions=questions,
        input_ids=input_ids,
        attention_mask=attention_mask,
        padding_mask=attention_mask == 0,
        offset_mapping=offset_mapping,
        spans=tuple(spans),
        source_fields=("question",) * len(questions),
    )


def make_query_input(
    questions: tuple[str, ...],
    query_times: tuple[float, ...],
    *,
    embedding_dim: int = 16,
    explicit_time_values: tuple[tuple[float, ...], ...] | None = None,
) -> QueryEncoderInput:
    tokens = make_char_token_batch(questions)
    torch.manual_seed(7)
    embeddings = torch.randn(*tokens.input_ids.shape, embedding_dim)
    values = (
        explicit_time_values
        if explicit_time_values is not None
        else tuple(extract_explicit_time_values(question) for question in questions)
    )
    return QueryEncoderInput(
        question_embeddings=embeddings,
        question_tokens=tokens,
        query_time=torch.tensor(query_times, dtype=torch.float32),
        explicit_time_values=values,
    )


def make_tiny_query_config() -> QueryEncoderConfig:
    return QueryEncoderConfig(
        input_dim=16,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        ffn_dim=16,
        dropout=0.0,
        output_dim=4,
        bidirectional=True,
        position_encoding="sinusoidal",
        pooling="learned_attention",
    )


def make_time_logits(
    query_input: QueryEncoderInput,
    modes: tuple[TimeWindowMode, ...],
    *,
    confidence: float = 1.0,
    pointer_char_spans: tuple[tuple[int, int] | None, ...] | None = None,
) -> TimeResolverLogits:
    batch_size, width = query_input.question_tokens.input_ids.shape
    mode_logits = torch.zeros(batch_size, 4)
    mode_indices = torch.tensor(
        [tuple(TimeWindowMode).index(mode) for mode in modes],
        dtype=torch.int64,
    )
    mode_logits.scatter_(1, mode_indices.unsqueeze(1), 8.0)
    minimum = torch.finfo(torch.float32).min
    start = torch.zeros(batch_size, width).masked_fill(query_input.padding_mask, minimum)
    end = torch.zeros(batch_size, width).masked_fill(query_input.padding_mask, minimum)
    if pointer_char_spans is not None:
        if len(pointer_char_spans) != batch_size:
            raise ValueError("pointer_char_spans must contain one entry per batch item")
        for row, span in enumerate(pointer_char_spans):
            if span is None:
                continue
            span_start, span_end = span
            if not 0 <= span_start < span_end <= query_input.question_tokens.spans[row].end:
                raise ValueError("pointer test span must be a valid character-token interval")
            start[row, span_start] = 10.0
            end[row, span_end - 1] = 10.0
    return TimeResolverLogits(
        mode_logits=mode_logits,
        mode_confidence=torch.full((batch_size,), confidence),
        mode_indices=mode_indices,
        span_start_logits=start,
        span_end_logits=end,
        padding_mask=query_input.padding_mask,
    )


def test_tiny_backbone_is_bidirectional_and_padding_invariant() -> None:
    torch.manual_seed(0)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config()).eval()
    embeddings = torch.randn(2, 5, 16)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )

    baseline = encoder(embeddings, padding_mask)
    poisoned_padding = embeddings.clone()
    poisoned_padding[1, 3:] = 1.0e6
    after_padding_change = encoder(poisoned_padding, padding_mask)
    changed_future = embeddings.clone()
    changed_future[0, -1] += 10.0
    after_future_change = encoder(changed_future, padding_mask)
    changed_prefix = embeddings.clone()
    changed_prefix[0, 0] -= 10.0
    after_prefix_change = encoder(changed_prefix, padding_mask)

    assert torch.equal(baseline.pooling_weights[padding_mask], torch.zeros(2))
    assert torch.allclose(baseline.pooling_weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(
        baseline.token_states[1, :3],
        after_padding_change.token_states[1, :3],
    )
    assert torch.allclose(baseline.q_target[1], after_padding_change.q_target[1])
    assert not torch.allclose(baseline.token_states[0, 0], after_future_change.token_states[0, 0])
    assert not torch.allclose(baseline.token_states[0, -1], after_prefix_change.token_states[0, -1])
    for query_embedding in (baseline.q_target, baseline.q_operator, baseline.q_time):
        assert query_embedding.shape == (2, 4)
        assert torch.isfinite(query_embedding).all()
        assert torch.allclose(query_embedding.norm(dim=-1), torch.ones(2), atol=1.0e-6)


def test_bfloat16_autocast_keeps_pooling_weights_normalized_in_float32() -> None:
    torch.manual_seed(11)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config()).eval()
    embeddings = torch.randn(2, 17, 16)
    padding_mask = torch.tensor(
        [
            [False] * 17,
            [False] * 7 + [True] * 10,
        ]
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(embeddings, padding_mask)

    assert output.token_states.dtype == torch.bfloat16
    assert output.pooling_weights.dtype == torch.float32
    assert torch.equal(output.pooling_weights[padding_mask], torch.zeros(10))
    assert torch.allclose(
        output.pooling_weights.sum(dim=1),
        torch.ones(2),
        atol=1.0e-6,
        rtol=0.0,
    )
    for embedding in (output.q_target, output.q_operator, output.q_time):
        assert embedding.dtype == torch.bfloat16
        assert torch.isfinite(embedding).all()


def test_sinusoidal_positions_make_token_order_observable_without_adding_parameters() -> None:
    torch.manual_seed(4)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config()).eval()
    embeddings = torch.randn(1, 5, 16)
    padding_mask = torch.zeros(1, 5, dtype=torch.bool)

    original = encoder(embeddings, padding_mask)
    reversed_order = encoder(embeddings.flip(1), padding_mask)

    assert not torch.allclose(original.q_target, reversed_order.q_target)
    assert query_embedding_parameter_count(encoder) == query_embedding_parameter_count(
        QueryEmbeddingEncoder(make_tiny_query_config())
    )


def test_attention_calls_use_only_padding_mask_and_are_explicitly_noncausal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = QueryEmbeddingEncoder(make_tiny_query_config())
    captured: list[dict[str, object]] = []
    for layer in encoder.transformer.layers:
        attention = layer.self_attn
        original_forward = attention.forward

        def capture_forward(
            *args: object,
            _original: object = original_forward,
            **kwargs: object,
        ) -> object:
            captured.append(dict(kwargs))
            return _original(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(attention, "forward", capture_forward)

    padding_mask = torch.tensor([[False, False, False, True]])
    encoder(torch.randn(1, 4, 16), padding_mask)

    assert len(captured) == 2
    for call in captured:
        assert call["attn_mask"] is None
        assert call["is_causal"] is False
        key_padding_mask = call["key_padding_mask"]
        assert isinstance(key_padding_mask, Tensor)
        if key_padding_mask.dtype == torch.bool:
            assert torch.equal(key_padding_mask, padding_mask)
        else:
            assert torch.equal(torch.isneginf(key_padding_mask), padding_mask)


def test_query_encoder_structure_freezes_gelu_no_final_norm_and_biasless_pool_scorer() -> None:
    encoder = QueryEmbeddingEncoder(make_tiny_query_config())

    assert encoder.transformer.norm is None
    assert encoder.pool_scorer.bias is None
    assert len(encoder.transformer.layers) == 2
    for layer in encoder.transformer.layers:
        assert layer.norm_first is True
        assert layer.activation is torch.nn.functional.gelu
    for head in (encoder.target_head, encoder.operator_head, encoder.time_head):
        assert isinstance(head[1], nn.GELU)
        assert not any(isinstance(module, nn.LayerNorm) for module in head)


def test_backbone_rejects_all_padding_bad_masks_and_non_finite_inputs() -> None:
    encoder = QueryEmbeddingEncoder(make_tiny_query_config())
    embeddings = torch.zeros(1, 3, 16)

    with pytest.raises(ValueError, match="at least one"):
        encoder(embeddings, torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        encoder(embeddings, torch.zeros(1, 3))
    bad = embeddings.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        encoder(bad, torch.zeros(1, 3, dtype=torch.bool))


def test_three_embedding_heads_are_independent_and_receive_gradients() -> None:
    torch.manual_seed(1)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config())
    head_parameter_ids = [
        {id(parameter) for parameter in head.parameters()}
        for head in (encoder.target_head, encoder.operator_head, encoder.time_head)
    ]
    assert not (head_parameter_ids[0] & head_parameter_ids[1])
    assert not (head_parameter_ids[0] & head_parameter_ids[2])
    assert not (head_parameter_ids[1] & head_parameter_ids[2])

    embeddings = torch.randn(2, 4, 16, requires_grad=True)
    output = encoder(embeddings, torch.zeros(2, 4, dtype=torch.bool))
    weights = torch.arange(1, 5, dtype=embeddings.dtype)
    loss = (
        (output.q_target * weights).sum()
        + (output.q_operator * weights.flip(0)).sum()
        + (output.q_time * weights.square()).sum()
    )
    loss.backward()

    assert encoder.input_projection.weight.grad is not None
    for head in (encoder.target_head, encoder.operator_head, encoder.time_head):
        final_weight = head[-1].weight
        assert final_weight.grad is not None
        assert torch.isfinite(final_weight.grad).all()
        assert final_weight.grad.abs().sum() > 0


def test_perturbing_one_embedding_head_does_not_change_the_other_two() -> None:
    torch.manual_seed(2)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config()).eval()
    embeddings = torch.randn(2, 4, 16)
    padding_mask = torch.zeros(2, 4, dtype=torch.bool)
    with torch.no_grad():
        before = encoder(embeddings, padding_mask)
        encoder.target_head[-1].bias[0].add_(1.0)
        after = encoder(embeddings, padding_mask)

    assert not torch.allclose(before.q_target, after.q_target)
    assert torch.equal(before.q_operator, after.q_operator)
    assert torch.equal(before.q_time, after.q_time)


def test_query_embedding_parameter_budget_is_exact_on_meta_device() -> None:
    with torch.device("meta"):
        encoder = QueryEmbeddingEncoder(load_config().query_encoder)

    assert query_embedding_parameter_count(encoder) == 36_026_112
    assert query_embedding_parameter_count(encoder) / 1_000_000 == pytest.approx(
        36.03,
        abs=0.005,
    )


def test_operator_router_covers_all_nine_classes_and_head_mapping() -> None:
    config = load_config().operator_router.model_copy(update={"confidence_threshold": 0.2})
    router = OperatorRouter(config)
    with torch.no_grad():
        router.prototypes.zero_()
        router.prototypes[:, : len(Operator)] = torch.eye(len(Operator))
        router.log_temperature.zero_()
    queries = torch.zeros(len(Operator), 512)
    queries[:, : len(Operator)] = torch.eye(len(Operator))

    output = router(queries, apply_confidence_gate=True)
    scaled = router(queries * 7.0, apply_confidence_gate=True)

    assert output.hard_operators == tuple(Operator)
    assert output.head_types == tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in Operator)
    assert output.head_types[:2] == (HeadType.O1, HeadType.O1)
    assert output.head_types[2:4] == (HeadType.O2, HeadType.O2)
    assert output.head_types[4:6] == (HeadType.E1, HeadType.E1)
    assert output.head_types[6:8] == (HeadType.E2, HeadType.E2)
    assert output.head_types[8] is None
    assert tuple(OPERATOR_TO_EVENT_KIND[operator] for operator in Operator) == (
        None,
        None,
        None,
        None,
        E1EventKind.ACTION,
        E1EventKind.TRANSIT,
        E2EventKind.PERIODIC,
        E2EventKind.EPISODE,
        None,
    )
    assert torch.allclose(output.logits, scaled.logits)
    assert torch.equal(output.confidence, torch.softmax(output.logits, dim=-1).max(dim=-1).values)
    assert router.temperature.item() > 0.0
    assert operator_router_parameter_count(router) == 4_609

    low_confidence = router(torch.zeros(1, 512), apply_confidence_gate=True)
    assert low_confidence.hard_operators == (Operator.UNSUPPORTED,)


def test_uncalibrated_router_keeps_raw_logits_but_gates_inference() -> None:
    router = OperatorRouter(load_config().operator_router)
    query = torch.randn(3, 512)

    training_route = router(query, apply_confidence_gate=False)
    inference_route = router(query, apply_confidence_gate=True)

    assert torch.equal(training_route.logits, inference_route.logits)
    assert inference_route.hard_operators == (Operator.UNSUPPORTED,) * 3
    assert all(head_type is None for head_type in inference_route.head_types)


def test_time_network_shapes_masks_parameter_count_and_gradients() -> None:
    resolver = TimeWindowResolver(load_config().time_resolver)
    q_time = torch.randn(2, 512, requires_grad=True)
    token_states = torch.randn(2, 5, 768, requires_grad=True)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )

    output = resolver(q_time, token_states, padding_mask)

    assert output.mode_logits.shape == (2, 4)
    assert output.span_start_logits.shape == output.span_end_logits.shape == (2, 5)
    minimum = torch.finfo(token_states.dtype).min
    assert torch.equal(output.span_start_logits[padding_mask], torch.full((2,), minimum))
    assert torch.equal(output.span_end_logits[padding_mask], torch.full((2,), minimum))
    loss = (
        output.mode_logits.sum()
        + output.span_start_logits[~padding_mask].sum()
        + output.span_end_logits[~padding_mask].sum()
    )
    loss.backward()
    assert resolver.mode_classifier[0].weight.grad is not None
    assert resolver.span_start.weight.grad is not None
    assert resolver.span_end.weight.grad is not None
    assert time_resolver_parameter_count(resolver) == 133_894


def test_default_now_and_history_windows_follow_operator_semantics() -> None:
    query_input = make_query_input(
        ("How many now?", "How many have appeared so far?"),
        (10.0, 12.0),
    )
    resolver = TimeWindowResolver(load_config().time_resolver)
    logits = make_time_logits(
        query_input,
        (TimeWindowMode.NOW, TimeWindowMode.HISTORY),
    )

    output = resolver.resolve(
        logits,
        query_input,
        (Operator.O1_SNAP, Operator.O2_UNIQUE),
        apply_confidence_gate=False,
    )

    now, history = output.resolutions
    assert now.status is TimeResolutionStatus.OK
    assert now.window.start_time is None
    assert now.window.end_time == now.window.query_time == 10.0
    assert now.used_operator_default is True
    assert history.status is TimeResolutionStatus.OK
    assert history.window.start_time == 0.0
    assert history.window.end_time == history.window.query_time == 12.0


@pytest.mark.parametrize(
    ("operator", "expected_mode", "expected_status"),
    [
        (Operator.O1_SNAP, TimeWindowMode.NOW, TimeResolutionStatus.OK),
        (Operator.O1_DELTA, TimeWindowMode.RECENT, TimeResolutionStatus.INVALID),
        (Operator.O2_UNIQUE, TimeWindowMode.HISTORY, TimeResolutionStatus.OK),
        (Operator.O2_GAIN, TimeWindowMode.RECENT, TimeResolutionStatus.INVALID),
        (Operator.E1_ACTION, TimeWindowMode.HISTORY, TimeResolutionStatus.OK),
        (Operator.E1_TRANSIT, TimeWindowMode.HISTORY, TimeResolutionStatus.OK),
        (Operator.E2_PERIODIC, TimeWindowMode.HISTORY, TimeResolutionStatus.OK),
        (Operator.E2_EPISODE, TimeWindowMode.HISTORY, TimeResolutionStatus.OK),
    ],
)
def test_all_eight_operator_default_time_semantics_are_explicit(
    operator: Operator,
    expected_mode: TimeWindowMode,
    expected_status: TimeResolutionStatus,
) -> None:
    query_input = make_query_input(("How many?",), (10.0,))
    resolver = TimeWindowResolver(load_config().time_resolver)
    logits = make_time_logits(query_input, (expected_mode,))

    resolution = resolver.resolve(
        logits,
        query_input,
        (operator,),
        apply_confidence_gate=False,
    ).resolutions[0]

    assert resolution.window.mode is expected_mode
    assert resolution.status is expected_status
    if expected_mode is TimeWindowMode.RECENT:
        assert resolution.reason == "recent_requires_one_duration"
    else:
        assert resolution.used_operator_default is True


@pytest.mark.parametrize(
    ("question", "query_time", "explicit_values", "numeric_text", "expected_start"),
    [
        (
            "Count events in the last 2 minutes and 3 seconds",
            200.0,
            (120.0, 3.0),
            "2 minutes and 3 seconds",
            77.0,
        ),
        ("过去 10 秒内发生了几次？", 20.0, (10.0,), "10 秒", 10.0),
    ],
)
def test_recent_time_parser_supports_english_chinese_and_compound_units(
    question: str,
    query_time: float,
    explicit_values: tuple[float, ...],
    numeric_text: str,
    expected_start: float,
) -> None:
    query_input = make_query_input(
        (question,),
        (query_time,),
        explicit_time_values=(explicit_values,),
    )
    resolver = TimeWindowResolver(load_config().time_resolver)
    numeric_start = question.index(numeric_text)
    logits = make_time_logits(
        query_input,
        (TimeWindowMode.RECENT,),
        pointer_char_spans=((numeric_start, numeric_start + len(numeric_text)),),
    )

    output = resolver.resolve(
        logits,
        query_input,
        (Operator.O1_DELTA,),
        apply_confidence_gate=False,
    ).resolutions[0]

    assert output.status is TimeResolutionStatus.OK
    assert output.window.mode is TimeWindowMode.RECENT
    assert output.window.start_time == expected_start
    assert output.window.end_time == query_time
    assert output.numeric_span is not None


@pytest.mark.parametrize(
    "question",
    ["from 2 to 8 seconds", "从 2 到 8 秒"],
)
def test_explicit_range_parser_supports_shared_english_and_chinese_units(
    question: str,
) -> None:
    query_input = make_query_input(
        (question,),
        (10.0,),
    )
    resolver = TimeWindowResolver(load_config().time_resolver)
    numeric_start = question.index("2")
    logits = make_time_logits(
        query_input,
        (TimeWindowMode.EXPLICIT_RANGE,),
        pointer_char_spans=((numeric_start, len(question)),),
    )

    output = resolver.resolve(
        logits,
        query_input,
        (Operator.E2_EPISODE,),
        apply_confidence_gate=False,
    ).resolutions[0]

    assert output.status is TimeResolutionStatus.OK
    assert output.window.mode is TimeWindowMode.EXPLICIT_RANGE
    assert output.window.start_time == 2.0
    assert output.window.end_time == 8.0
    assert output.parsed_values_seconds == (2.0, 8.0)


@pytest.mark.parametrize(
    ("question", "query_time", "operator", "numeric_text", "reason"),
    [
        (
            "How many changed?",
            10.0,
            Operator.O1_DELTA,
            None,
            "recent_requires_one_duration",
        ),
        ("in the last 2 hours", 10.0, Operator.O1_DELTA, None, "unsupported_time_unit"),
        (
            "How many are visible in the last 2 days?",
            10.0,
            Operator.O1_SNAP,
            None,
            "unsupported_time_unit",
        ),
        ("in the last 0 seconds", 10.0, Operator.O1_DELTA, "0 seconds", "must_be_positive"),
        (
            "in the last 15 seconds",
            10.0,
            Operator.O1_DELTA,
            "15 seconds",
            "starts_before_video",
        ),
        ("from 8 to 2 seconds", 10.0, Operator.E1_ACTION, "8 to 2 seconds", "is_reversed"),
        (
            "from 2 to 12 seconds",
            10.0,
            Operator.E1_ACTION,
            "2 to 12 seconds",
            "after_query_time",
        ),
        (
            "last 2 seconds and past 3 seconds",
            10.0,
            Operator.E1_ACTION,
            None,
            "ambiguous_time_expression",
        ),
    ],
)
def test_time_parser_rejects_missing_invalid_reversed_future_and_ambiguous_windows(
    question: str,
    query_time: float,
    operator: Operator,
    numeric_text: str | None,
    reason: str,
) -> None:
    query_input = make_query_input((question,), (query_time,))
    resolver = TimeWindowResolver(load_config().time_resolver)
    pointer_span = None
    if numeric_text is not None:
        start = question.index(numeric_text)
        pointer_span = (start, start + len(numeric_text))
    logits = make_time_logits(
        query_input,
        (TimeWindowMode.RECENT,),
        pointer_char_spans=(pointer_span,),
    )

    output = resolver.resolve(
        logits,
        query_input,
        (operator,),
        apply_confidence_gate=False,
    ).resolutions[0]

    assert output.status is TimeResolutionStatus.INVALID
    assert output.window.valid is False
    assert reason in output.reason
    assert output.window.end_time == query_time


def test_time_parser_rejects_pointer_order_outside_span_and_explicit_value_mismatch() -> None:
    question = "in the last 10 seconds"
    query_input = make_query_input((question,), (20.0,))
    resolver = TimeWindowResolver(load_config().time_resolver)

    reversed_pointer = make_time_logits(query_input, (TimeWindowMode.RECENT,))
    reversed_pointer.span_start_logits[0, question.index("seconds") + len("seconds") - 1] = 10.0
    reversed_pointer.span_end_logits[0, question.index("10")] = 10.0
    reversed_result = resolver.resolve(
        reversed_pointer,
        query_input,
        (Operator.O1_DELTA,),
        apply_confidence_gate=False,
    ).resolutions[0]
    assert reversed_result.status is TimeResolutionStatus.INVALID
    assert reversed_result.reason == "pointer_order_invalid"

    outside_pointer = make_time_logits(
        query_input,
        (TimeWindowMode.RECENT,),
        pointer_char_spans=((0, 2),),
    )
    outside_result = resolver.resolve(
        outside_pointer,
        query_input,
        (Operator.O1_DELTA,),
        apply_confidence_gate=False,
    ).resolutions[0]
    assert outside_result.status is TimeResolutionStatus.INVALID
    assert outside_result.reason == "pointer_outside_numeric_expression"

    for partial_text in ("10", "seconds"):
        partial_start = question.index(partial_text)
        partial_pointer = make_time_logits(
            query_input,
            (TimeWindowMode.RECENT,),
            pointer_char_spans=((partial_start, partial_start + len(partial_text)),),
        )
        partial_result = resolver.resolve(
            partial_pointer,
            query_input,
            (Operator.O1_DELTA,),
            apply_confidence_gate=False,
        ).resolutions[0]
        assert partial_result.status is TimeResolutionStatus.INVALID
        assert partial_result.reason == "pointer_does_not_cover_numeric_expression"

    mismatched_input = make_query_input(
        (question,),
        (20.0,),
        explicit_time_values=((11.0,),),
    )
    aligned_pointer = make_time_logits(
        mismatched_input,
        (TimeWindowMode.RECENT,),
        pointer_char_spans=((question.index("10"), len(question)),),
    )
    mismatch_result = resolver.resolve(
        aligned_pointer,
        mismatched_input,
        (Operator.O1_DELTA,),
        apply_confidence_gate=False,
    ).resolutions[0]
    assert mismatch_result.status is TimeResolutionStatus.INVALID
    assert mismatch_result.reason == "explicit_time_values_mismatch"


@pytest.mark.parametrize(
    "bad_values",
    [(), (120.0,), (120.0, 3.0, 1.0), (3.0, 120.0), (123.0,)],
)
def test_explicit_time_component_metadata_rejects_missing_extra_reordered_or_aggregated_values(
    bad_values: tuple[float, ...],
) -> None:
    question = "Count events in the last 2 minutes and 3 seconds"
    query_input = make_query_input(
        (question,),
        (200.0,),
        explicit_time_values=(bad_values,),
    )
    numeric_text = "2 minutes and 3 seconds"
    numeric_start = question.index(numeric_text)
    logits = make_time_logits(
        query_input,
        (TimeWindowMode.RECENT,),
        pointer_char_spans=((numeric_start, numeric_start + len(numeric_text)),),
    )

    resolution = (
        TimeWindowResolver(load_config().time_resolver)
        .resolve(
            logits,
            query_input,
            (Operator.O1_DELTA,),
            apply_confidence_gate=False,
        )
        .resolutions[0]
    )

    assert resolution.status is TimeResolutionStatus.INVALID
    assert resolution.reason == "explicit_time_values_mismatch"


def test_time_confidence_gate_requires_calibration_and_mode_agreement() -> None:
    config = load_config().time_resolver.model_copy(update={"confidence_threshold": 0.8})
    resolver = TimeWindowResolver(config)
    query_input = make_query_input(("How many now?",), (5.0,))

    low = make_time_logits(query_input, (TimeWindowMode.NOW,), confidence=0.7)
    low_result = resolver.resolve(
        low,
        query_input,
        (Operator.O1_SNAP,),
        apply_confidence_gate=True,
    ).resolutions[0]
    assert low_result.status is TimeResolutionStatus.UNSUPPORTED

    wrong_mode = make_time_logits(query_input, (TimeWindowMode.HISTORY,), confidence=0.9)
    mismatch = resolver.resolve(
        wrong_mode,
        query_input,
        (Operator.O1_SNAP,),
        apply_confidence_gate=True,
    ).resolutions[0]
    assert mismatch.status is TimeResolutionStatus.UNSUPPORTED
    assert "mismatch" in mismatch.reason

    correct = make_time_logits(query_input, (TimeWindowMode.NOW,), confidence=0.9)
    accepted = resolver.resolve(
        correct,
        query_input,
        (Operator.O1_SNAP,),
        apply_confidence_gate=True,
    ).resolutions[0]
    assert accepted.status is TimeResolutionStatus.OK


def test_uncalibrated_time_resolver_keeps_logits_but_gates_inference() -> None:
    resolver = TimeWindowResolver(load_config().time_resolver)
    query_input = make_query_input(("How many now?",), (5.0,))
    logits = make_time_logits(query_input, (TimeWindowMode.NOW,), confidence=1.0)

    training_result = resolver.resolve(
        logits,
        query_input,
        (Operator.O1_SNAP,),
        apply_confidence_gate=False,
    ).resolutions[0]
    inference_result = resolver.resolve(
        logits,
        query_input,
        (Operator.O1_SNAP,),
        apply_confidence_gate=True,
    ).resolutions[0]

    assert training_result.status is TimeResolutionStatus.OK
    assert inference_result.status is TimeResolutionStatus.UNSUPPORTED
    assert inference_result.reason == "uncalibrated_or_low_time_confidence"


def test_runtime_payload_and_token_provenance_reject_every_label_field() -> None:
    tokens = make_char_token_batch(("How many?",))
    embeddings = torch.randn(1, tokens.input_ids.shape[1], 16)
    payload: dict[str, object] = {
        "video": (Path("synthetic.mp4"),),
        "question": tokens.questions,
        "query_time": torch.tensor([3.0]),
        "explicit_time_values": ((),),
    }

    query = RuntimeQueryInput(
        "video-0",
        "trajectory-0",
        "query-0",
        0,
        Path("synthetic.mp4"),
        "How many?",
        3.0,
        (),
    )
    safe = QueryEncoderInput.from_runtime_queries(embeddings, tokens, (query,))
    assert safe.question_tokens.questions == ("How many?",)
    for denied_field in RUNTIME_DENYLIST:
        poisoned = {**payload, denied_field: "forbidden"}
        with pytest.raises(ValueError, match="denied fields"):
            assert_runtime_payload_safe(poisoned, layer="JSON")
    with pytest.raises(ValueError, match="canonical question"):
        QueryEncoderInput.from_runtime_queries(
            embeddings,
            tokens,
            (replace(query, question="different"),),
        )
    with pytest.raises(ValueError, match="question-only provenance"):
        replace(tokens, source_fields=("answer",))


def test_appended_answer_token_cannot_extend_beyond_canonical_question() -> None:
    tokens = make_char_token_batch(("abc",))
    with pytest.raises(ValueError, match="beyond the canonical question"):
        QuestionTokenBatch(
            questions=tokens.questions,
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones(1, 4, dtype=torch.int64),
            padding_mask=torch.zeros(1, 4, dtype=torch.bool),
            offset_mapping=torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]]),
            spans=(QuestionTokenSpan(0, 4),),
            source_fields=("question",),
        )


def test_query_time_and_runtime_explicit_values_must_be_finite() -> None:
    tokens = make_char_token_batch(("How many?",))
    embeddings = torch.zeros(1, tokens.input_ids.shape[1], 16)
    with pytest.raises(ValueError, match="finite and non-negative"):
        QueryEncoderInput(embeddings, tokens, torch.tensor([torch.nan]), ((),))
    with pytest.raises(ValueError, match="finite and non-negative"):
        QueryEncoderInput(embeddings, tokens, torch.tensor([1.0]), ((torch.inf,),))


class FakeEmbeddingOnlyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 4096)
        self.decoder_called = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, *_args: object, **_kwargs: object) -> Tensor:
        self.decoder_called = True
        raise AssertionError("the full decoder must not run for question encoding")


def test_question_embedding_helper_never_calls_full_qwen_decoder() -> None:
    model = FakeEmbeddingOnlyQwen()
    tokens = make_char_token_batch(("question",))

    embeddings = embed_question_tokens(model, tokens, load_config())

    assert embeddings.shape == (1, len("question"), 4096)
    assert model.decoder_called is False


def test_full_p4_composition_outputs_512_embeddings_and_forces_invalid_time_unsupported() -> None:
    query_input = make_query_input(("abc",), (5.0,), embedding_dim=4096)
    model = build_query_encoder(load_config()).eval()
    with torch.no_grad():
        encoded = model.embedding_encoder(query_input.question_embeddings, query_input.padding_mask)
        model.operator_router.prototypes.copy_(-encoded.q_operator.repeat(len(Operator), 1))
        model.operator_router.prototypes[0].copy_(encoded.q_operator[0])
        snap = model(query_input, inference=False)
        model.operator_router.prototypes.copy_(-encoded.q_operator.repeat(len(Operator), 1))
        model.operator_router.prototypes[1].copy_(encoded.q_operator[0])
        delta = model(query_input, inference=False)

    assert snap.q_target.shape == snap.q_operator.shape == snap.q_time.shape == (1, 512)
    assert snap.embeddings.token_states.shape == (1, 3, 768)
    assert snap.route.hard_operators == (Operator.O1_SNAP,)
    assert snap.hard_operators == (Operator.O1_SNAP,)
    assert snap.head_types == (HeadType.O1,)
    assert snap.time.resolutions[0].window.mode is TimeWindowMode.NOW
    assert delta.route.hard_operators == (Operator.O1_DELTA,)
    assert delta.hard_operators == (Operator.UNSUPPORTED,)
    assert delta.time.resolutions[0].status is TimeResolutionStatus.INVALID


def test_supervision_is_separate_from_forward_signature() -> None:
    supervision = QueryEncoderSupervision(
        operator_targets=torch.tensor([0, 8]),
        time_mode_targets=torch.tensor([0, 3]),
        span_start_targets=torch.tensor([-100, 1]),
        span_end_targets=torch.tensor([-100, 2]),
    )

    assert supervision.operator_targets.shape == (2,)
    assert "supervision" not in inspect.signature(QueryEncoder.forward).parameters


def test_query_encoder_eval_defaults_to_uncalibrated_inference_gate() -> None:
    query_input = make_query_input(("How many now?",), (5.0,), embedding_dim=4096)
    model = build_query_encoder(load_config()).eval()

    with torch.no_grad():
        output = model(query_input)

    assert output.route.confidence_gate_applied is True
    assert output.hard_operators == (Operator.UNSUPPORTED,)


@pytest.mark.parametrize(
    ("start_targets", "end_targets", "reason"),
    [
        ([-1], [0], "non-negative"),
        ([-100], [0], "ignored together"),
        ([2], [1], "start <= end"),
    ],
)
def test_supervision_rejects_invalid_or_inconsistent_pointer_targets(
    start_targets: list[int],
    end_targets: list[int],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        QueryEncoderSupervision(
            operator_targets=torch.tensor([0]),
            time_mode_targets=torch.tensor([0]),
            span_start_targets=torch.tensor(start_targets),
            span_end_targets=torch.tensor(end_targets),
        )
