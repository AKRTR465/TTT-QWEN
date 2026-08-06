from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from tests.support import parameter_count
from ttt_svcbench_qwen.config import QueryEncoderConfig, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput, extract_explicit_time_values
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_EVENT_KIND,
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    OperatorRouter,
    QueryEmbeddingEncoder,
    QueryEncoderInput,
    TimeResolution,
    TimeResolutionStatus,
    TimeResolverLogits,
    TimeWindowMode,
    TimeWindowResolver,
    build_query_encoder,
    embed_question_tokens,
)
from ttt_svcbench_qwen.query_tokens import QuestionTokenBatch
from ttt_svcbench_qwen.state_bank import E1EventKind, E2EventKind, HeadType


def make_char_token_batch(questions: tuple[str, ...]) -> QuestionTokenBatch:
    width = max(len(question) for question in questions)
    input_ids = torch.zeros(len(questions), width, dtype=torch.int64)
    attention_mask = torch.zeros_like(input_ids)
    for row, question in enumerate(questions):
        length = len(question)
        input_ids[row, :length] = torch.arange(1, length + 1)
        attention_mask[row, :length] = 1
    return QuestionTokenBatch(
        questions=questions,
        input_ids=input_ids,
        attention_mask=attention_mask,
        padding_mask=attention_mask == 0,
    )


def make_query_input(
    questions: tuple[str, ...],
    query_times: tuple[float, ...],
    *,
    embedding_dim: int = 16,
) -> QueryEncoderInput:
    tokens = make_char_token_batch(questions)
    torch.manual_seed(7)
    embeddings = torch.randn(*tokens.input_ids.shape, embedding_dim)
    return QueryEncoderInput(
        question_embeddings=embeddings,
        question_tokens=tokens,
        query_time=torch.tensor(query_times, dtype=torch.float32),
        explicit_time_values=tuple(
            extract_explicit_time_values(question) for question in questions
        ),
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


def make_time_logits(query_input: QueryEncoderInput) -> TimeResolverLogits:
    """Build a shape-correct logits bundle; `resolve` reads only its batch size."""

    batch_size, width = query_input.question_tokens.input_ids.shape
    minimum = torch.finfo(torch.float32).min
    padding_mask = query_input.padding_mask
    return TimeResolverLogits(
        mode_logits=torch.zeros(batch_size, len(TimeWindowMode)),
        mode_confidence=torch.ones(batch_size),
        mode_indices=torch.zeros(batch_size, dtype=torch.int64),
        span_start_logits=torch.zeros(batch_size, width).masked_fill(padding_mask, minimum),
        span_end_logits=torch.zeros(batch_size, width).masked_fill(padding_mask, minimum),
        padding_mask=padding_mask,
    )


def resolve_one(question: str, query_time: float, operator: Operator) -> TimeResolution:
    query_input = make_query_input((question,), (query_time,))
    resolver = TimeWindowResolver(load_config().time_resolver)
    return resolver.resolve(
        make_time_logits(query_input),
        query_input,
        (operator,),
    ).resolutions[0]


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


def test_bfloat16_autocast_keeps_query_embeddings_finite_and_padding_safe() -> None:
    torch.manual_seed(11)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config()).eval()
    embeddings = torch.randn(2, 17, 16)
    padding_mask = torch.tensor([[False] * 17, [False] * 7 + [True] * 10])
    poisoned = embeddings.clone()
    poisoned[1, 7:] = 1.0e6

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(embeddings, padding_mask)
        poisoned_output = encoder(poisoned, padding_mask)

    assert output.token_states.dtype == torch.bfloat16
    assert torch.equal(output.q_target[1], poisoned_output.q_target[1])
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
    assert parameter_count(encoder) == parameter_count(
        QueryEmbeddingEncoder(make_tiny_query_config())
    )


def test_three_embedding_heads_are_independent_and_receive_gradients() -> None:
    torch.manual_seed(1)
    encoder = QueryEmbeddingEncoder(make_tiny_query_config())
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

    assert parameter_count(encoder) == 36_026_112
    assert parameter_count(encoder) / 1_000_000 == pytest.approx(36.03, abs=0.005)


def test_operator_router_covers_all_nine_classes_and_head_mapping() -> None:
    router = OperatorRouter(load_config().operator_router)
    with torch.no_grad():
        router.prototypes.zero_()
        router.prototypes[:, : len(Operator)] = torch.eye(len(Operator))
        router.log_temperature.zero_()
    queries = torch.zeros(len(Operator), 512, requires_grad=True)
    with torch.no_grad():
        queries[:, : len(Operator)] = torch.eye(len(Operator))

    output = router(queries)
    scaled = router(queries * 7.0)
    output.logits.sum().backward()

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
    assert torch.equal(output.raw_indices, torch.arange(len(Operator)))
    assert torch.allclose(output.logits, scaled.logits)
    assert torch.equal(output.confidence, torch.softmax(output.logits, dim=-1).max(dim=-1).values)
    assert parameter_count(router) == 4_609
    # route.logits is the supervised quantity: it must stay differentiable.
    assert output.logits.requires_grad is True
    assert router.prototypes.grad is not None
    assert router.prototypes.grad.abs().sum() > 0


def test_router_temperature_divides_the_cosine_logits() -> None:
    router = OperatorRouter(load_config().operator_router)
    with torch.no_grad():
        router.prototypes.zero_()
        router.prototypes[:, : len(Operator)] = torch.eye(len(Operator))
    queries = torch.zeros(len(Operator), 512)
    queries[:, : len(Operator)] = torch.eye(len(Operator))

    with torch.no_grad():
        router.log_temperature.fill_(0.0)
        unit = router(queries)
        router.log_temperature.fill_(float(torch.tensor(2.0).log()))
        halved = router(queries)

    assert router.temperature.item() == pytest.approx(2.0)
    assert torch.allclose(unit.logits.diagonal(), torch.ones(len(Operator)), atol=1.0e-6)
    assert torch.allclose(halved.logits, unit.logits / 2.0, atol=1.0e-6)


def test_time_network_shapes_masks_parameter_count_and_gradients() -> None:
    resolver = TimeWindowResolver(load_config().time_resolver)
    q_time = torch.randn(2, 512, requires_grad=True)
    token_states = torch.randn(2, 5, 768, requires_grad=True)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )

    output = resolver(q_time, token_states, padding_mask)

    assert output.mode_logits.shape == (2, len(TimeWindowMode))
    assert output.span_start_logits.shape == output.span_end_logits.shape == (2, 5)
    minimum = torch.finfo(token_states.dtype).min
    assert torch.equal(output.span_start_logits[padding_mask], torch.full((2,), minimum))
    assert torch.equal(output.span_end_logits[padding_mask], torch.full((2,), minimum))
    # mode_logits / span_*_logits are the supervised quantities.
    loss = (
        output.mode_logits.sum()
        + output.span_start_logits[~padding_mask].sum()
        + output.span_end_logits[~padding_mask].sum()
    )
    loss.backward()
    assert resolver.mode_classifier[0].weight.grad is not None
    assert resolver.span_start.weight.grad is not None
    assert resolver.span_end.weight.grad is not None
    assert parameter_count(resolver) == 133_894


@pytest.mark.parametrize(
    ("operator", "expected_mode", "expected_start"),
    [
        (Operator.O1_SNAP, TimeWindowMode.NOW, None),
        (Operator.O1_DELTA, TimeWindowMode.RECENT, 0.0),
        (Operator.O2_UNIQUE, TimeWindowMode.HISTORY, 0.0),
        (Operator.O2_GAIN, TimeWindowMode.RECENT, 0.0),
        (Operator.E1_ACTION, TimeWindowMode.HISTORY, 0.0),
        (Operator.E1_TRANSIT, TimeWindowMode.HISTORY, 0.0),
        (Operator.E2_PERIODIC, TimeWindowMode.HISTORY, 0.0),
        (Operator.E2_EPISODE, TimeWindowMode.HISTORY, 0.0),
    ],
)
def test_all_eight_operator_default_time_semantics_are_explicit(
    operator: Operator,
    expected_mode: TimeWindowMode,
    expected_start: float | None,
) -> None:
    resolution = resolve_one("How many?", 10.0, operator)

    assert resolution.window.mode is expected_mode
    assert resolution.status is TimeResolutionStatus.OK
    assert resolution.window.valid is True
    assert resolution.window.start_time == expected_start
    assert resolution.window.end_time == resolution.window.query_time == 10.0


@pytest.mark.parametrize(
    ("question", "query_time", "operator", "expected_mode", "expected_start", "expected_end"),
    [
        # Recent windows: English compound units, Chinese, and a duration longer
        # than the elapsed video (start clamped to zero).
        (
            "Count events in the last 2 minutes and 3 seconds",
            200.0,
            Operator.O1_DELTA,
            TimeWindowMode.RECENT,
            77.0,
            200.0,
        ),
        ("过去 10 秒内发生了几次？", 20.0, Operator.O1_DELTA, TimeWindowMode.RECENT, 10.0, 20.0),
        ("in the last 15 seconds", 10.0, Operator.O1_DELTA, TimeWindowMode.RECENT, 0.0, 10.0),
        # Explicit ranges: shared English/Chinese units, reversed endpoints get
        # ordered, and an end past the query time is clamped into the prefix.
        ("from 2 to 8 seconds", 10.0, Operator.E2_EPISODE, TimeWindowMode.EXPLICIT_RANGE, 2.0, 8.0),
        ("从 2 到 8 秒", 10.0, Operator.E2_EPISODE, TimeWindowMode.EXPLICIT_RANGE, 2.0, 8.0),
        ("from 8 to 2 seconds", 10.0, Operator.E1_ACTION, TimeWindowMode.EXPLICIT_RANGE, 2.0, 8.0),
        (
            "from 2 to 12 seconds",
            10.0,
            Operator.E1_ACTION,
            TimeWindowMode.EXPLICIT_RANGE,
            2.0,
            10.0,
        ),
    ],
)
def test_parsed_time_windows_are_ordered_and_clamped_into_the_causal_prefix(
    question: str,
    query_time: float,
    operator: Operator,
    expected_mode: TimeWindowMode,
    expected_start: float,
    expected_end: float,
) -> None:
    output = resolve_one(question, query_time, operator)

    assert output.status is TimeResolutionStatus.OK
    assert output.window.valid is True
    assert output.window.mode is expected_mode
    assert output.window.start_time == expected_start
    assert output.window.end_time == expected_end
    assert output.window.end_time <= query_time


def test_resolve_is_per_row_and_unsupported_operators_get_invalid_windows() -> None:
    query_input = make_query_input(("How many now?", "What colour is it?"), (5.0, 7.0))
    resolver = TimeWindowResolver(load_config().time_resolver)

    supported, unsupported = resolver.resolve(
        make_time_logits(query_input),
        query_input,
        (Operator.O1_SNAP, Operator.UNSUPPORTED),
    ).resolutions

    assert supported.status is TimeResolutionStatus.OK
    assert supported.window.mode is TimeWindowMode.NOW
    assert supported.window.end_time == supported.window.query_time == 5.0
    assert unsupported.status is TimeResolutionStatus.UNSUPPORTED
    assert unsupported.window.valid is False
    assert unsupported.window.start_time is None
    assert unsupported.window.end_time == unsupported.window.query_time == 7.0


def test_runtime_queries_bind_query_time_and_explicit_values() -> None:
    tokens = make_char_token_batch(("How many?",))
    embeddings = torch.randn(1, tokens.input_ids.shape[1], 16)
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

    bound = QueryEncoderInput.from_runtime_queries(embeddings, tokens, (query,))

    assert bound.question_tokens.questions == ("How many?",)
    assert bound.query_time.tolist() == [3.0]
    assert bound.explicit_time_values == ((),)


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


def test_full_p4_composition_outputs_512_embeddings_and_routed_time_windows() -> None:
    query_input = make_query_input(("abc",), (5.0,), embedding_dim=4096)
    model = build_query_encoder(load_config()).eval()
    with torch.no_grad():
        encoded = model.embedding_encoder(query_input.question_embeddings, query_input.padding_mask)
        model.operator_router.prototypes.copy_(-encoded.q_operator.repeat(len(Operator), 1))
        model.operator_router.prototypes[0].copy_(encoded.q_operator[0])
        snap = model(query_input)
        model.operator_router.prototypes.copy_(-encoded.q_operator.repeat(len(Operator), 1))
        model.operator_router.prototypes[1].copy_(encoded.q_operator[0])
        delta = model(query_input)

    assert snap.q_target.shape == snap.q_operator.shape == snap.q_time.shape == (1, 512)
    assert snap.embeddings.token_states.shape == (1, 3, 768)
    assert snap.route.hard_operators == (Operator.O1_SNAP,)
    assert snap.hard_operators == (Operator.O1_SNAP,)
    assert snap.head_types == (HeadType.O1,)
    assert snap.operator_logits is snap.route.logits
    assert snap.time.logits.mode_logits.shape == (1, len(TimeWindowMode))
    assert snap.time.resolutions[0].window.mode is TimeWindowMode.NOW
    assert delta.route.hard_operators == (Operator.O1_DELTA,)
    assert delta.hard_operators == (Operator.O1_DELTA,)
    assert delta.head_types == (HeadType.O1,)
    assert delta.time.resolutions[0].window.mode is TimeWindowMode.RECENT
    assert delta.time.resolutions[0].status is TimeResolutionStatus.OK
