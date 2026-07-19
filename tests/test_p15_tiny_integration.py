from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from test_stage_a_runtime import _cache, _observations, _query, _spatial
from ttt_svcbench_qwen.config import StageAVariant, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.identity_bank import build_identity_bank
from ttt_svcbench_qwen.input_composer import EXACT_NUMBER_INSTRUCTION, compose_inputs
from ttt_svcbench_qwen.model import (
    BatchRuntimeState,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
    VisualStageOutput,
)
from ttt_svcbench_qwen.query_encoder import Operator, QueryEncoderOutput
from ttt_svcbench_qwen.stage_a_runtime import (
    StageABankWriter,
    StageASoftWriteOutput,
)
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    E1TargetLabels,
    E2TargetLabels,
    O1TargetLabels,
    O2TargetLabels,
    QueryTargetLabels,
    StageATargetBatch,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import HeadType, StructuredStateBank, build_state_bank
from ttt_svcbench_qwen.state_encoder import TemporalEncoderOutput
from ttt_svcbench_qwen.state_reader import DeterministicStateReader, ReaderResult
from ttt_svcbench_qwen.state_retriever import (
    EmbeddingStateRetriever,
    RetrievalStatus,
    build_state_retriever,
)
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageAEpisodeInputs,
    StageAEpisodeRunner,
    StageASupervisionBatch,
    StageATrainingBatch,
    StageATypedForwardAdapter,
    compute_stage_a_losses,
)


class _Tokenizer:
    name_or_path = "synthetic-p15-tokenizer"
    pad_token_id = 0

    def __init__(self) -> None:
        self.tokens = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            "<|video_pad|>": 2,
            "<|im_start|>": 3,
            "user": 4,
            "question": 5,
            "assistant": 6,
            "\n": 7,
            "answer": 8,
            "<|vision_start|>": 9,
            "<|vision_end|>": 10,
            "instruction-a": 11,
            "instruction-b": 12,
            "-": 13,
            **{str(value): 14 + value for value in range(10)},
        }
        self.additional_special_tokens: list[str] = []

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def add_special_tokens(
        self,
        special_tokens_dict: Mapping[str, object],
        replace_additional_special_tokens: bool = True,
    ) -> int:
        raw = special_tokens_dict["additional_special_tokens"]
        assert isinstance(raw, list)
        if replace_additional_special_tokens:
            self.additional_special_tokens = []
        added = 0
        for value in raw:
            token = str(value)
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.tokens.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == EXACT_NUMBER_INSTRUCTION:
            return [self.tokens["instruction-a"], self.tokens["instruction-b"]]
        return [self.tokens[value] for value in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        inverse = {value: key for key, value in self.tokens.items()}
        return "".join(inverse[int(value)] for value in token_ids)


class _TinyRopeIndexer:
    def get_rope_index(
        self,
        input_ids: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del image_grid_thw, video_grid_thw
        assert input_ids is not None and attention_mask is not None
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        position_ids = positions.unsqueeze(0).expand(3, -1, -1).clone()
        rope_deltas = (positions.max(dim=-1).values + 1 - input_ids.shape[1]).unsqueeze(1)
        return position_ids, rope_deltas


class _TinyQwenPrefill(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 128)
        self.prefill_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def resize_token_embeddings(self, new_num_tokens: int, **_kwargs: object) -> nn.Embedding:
        old = self.embedding
        replacement = nn.Embedding(new_num_tokens, old.embedding_dim)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[: old.num_embeddings].copy_(old.weight)
        self.embedding = replacement
        return replacement

    def forward(self, request: QwenPrefillRequest) -> SimpleNamespace:
        self.prefill_calls += 1
        assert isinstance(request.input_ids, Tensor)
        hidden = self.embedding(request.input_ids)
        if isinstance(request.state_tokens, Tensor):
            hidden = hidden + request.state_tokens.mean(dim=1, keepdim=True)
        return SimpleNamespace(logits=self.lm_head(hidden))


class _ForbiddenDecode:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _model_inputs: object) -> object:
        self.calls += 1
        raise AssertionError("Stage A must never enter decode")


class _VisualStage(nn.Module):
    def forward(self, request: ObservationChunkRequest) -> VisualStageOutput:
        return VisualStageOutput(request.video_input, prepared_video_features="tiny-video")


class _FastStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        assert isinstance(visual.value, Tensor)
        return replace(visual, value=visual.value * self.scale)


class _QueryStage(nn.Module):
    def __init__(self, output: QueryEncoderOutput) -> None:
        super().__init__()
        self.output = output

    def forward(self, _query_input: object, *, inference: bool) -> QueryEncoderOutput:
        assert inference is False
        return self.output


class _SpatialStage(nn.Module):
    def __init__(self, slots: Tensor) -> None:
        super().__init__()
        self.slots = nn.Parameter(slots)

    def forward(
        self,
        _visual: VisualStageOutput,
        _query: object,
        request: ObservationChunkRequest,
    ) -> object:
        return _spatial(request.owner, self.slots)


class _TemporalStage(nn.Module):
    def __init__(self, hidden: Tensor) -> None:
        super().__init__()
        self.hidden = nn.Parameter(hidden)

    def forward(
        self,
        _visual: VisualStageOutput,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput:
        width = self.hidden.shape[1]
        return TemporalEncoderOutput(
            hidden=self.hidden,
            timestamps=torch.arange(width, dtype=torch.float64).expand(4, -1).clone(),
            position_ids=torch.arange(width, dtype=torch.int64).expand(4, -1).clone(),
            valid_mask=torch.ones((4, width), dtype=torch.bool),
            cache=_cache(request.owner, self.hidden, query.q_target),
        )


class _ObservationStage(nn.Module):
    def forward(
        self,
        spatial: object,
        temporal: object,
        query: QueryEncoderOutput,
        request: ObservationChunkRequest,
    ) -> object:
        return _observations(request.owner, spatial, temporal, query.q_target)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _TinyResamplerOutput:
    state_tokens: Tensor
    state_token_valid_mask: Tensor
    selected_record_ids: tuple[tuple[str, ...], ...]
    retrieval_status: tuple[RetrievalStatus, ...]


class _TinyResampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(512, 8)

    def forward(self, q_target: Tensor, retrieval: object) -> _TinyResamplerOutput:
        assert hasattr(retrieval, "selected_record_ids") and hasattr(retrieval, "status")
        tokens = self.projection(q_target).unsqueeze(1).expand(-1, 16, -1)
        statuses = tuple(retrieval.status)
        valid = torch.tensor(
            [status in (RetrievalStatus.OK, RetrievalStatus.EMPTY) for status in statuses],
            dtype=torch.bool,
            device=tokens.device,
        )
        return _TinyResamplerOutput(
            state_tokens=torch.where(valid[:, None, None], tokens, 0.0),
            state_token_valid_mask=valid,
            selected_record_ids=tuple(retrieval.selected_record_ids),
            retrieval_status=statuses,
        )


class _CaptureMetrics:
    def __init__(self) -> None:
        self.output: StateTTTModelOutput | None = None

    def __call__(
        self,
        output: StateTTTModelOutput,
        _supervision: StageASupervisionBatch,
    ) -> tuple[tuple[tuple[str, float | None], ...], tuple[str, ...]]:
        self.output = output
        return (), ()


def _state_targets() -> StageATargetBatch:
    synthetic = TargetProvenance.SYNTHETIC_EXPLICIT
    identities = torch.zeros((1, 2, 256))
    identities[0, 0, 0] = 1.0
    identities[0, 1, 1] = 1.0
    return StageATargetBatch(
        o1=O1TargetLabels(
            row_indices=torch.tensor([0]),
            targets=torch.ones((1, 2, 6)),
            slot_mask=torch.ones((1, 2), dtype=torch.bool),
            provenance=(synthetic,),
        ),
        o2=O2TargetLabels(
            row_indices=torch.tensor([1]),
            identity_targets=identities,
            score_targets=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            slot_mask=torch.ones((1, 2), dtype=torch.bool),
            provenance=(synthetic,),
        ),
        e1=E1TargetLabels(
            row_indices=torch.tensor([2]),
            targets=torch.ones((1, 3, 3)),
            time_mask=torch.ones((1, 3), dtype=torch.bool),
            provenance=(synthetic,),
        ),
        e2=E2TargetLabels(
            row_indices=torch.tensor([3]),
            event_targets=torch.ones((1, 3, 4)),
            phase_targets=torch.tensor([[0, 1, 2]]),
            time_mask=torch.ones((1, 3), dtype=torch.bool),
            provenance=(synthetic,),
        ),
        query=QueryTargetLabels(
            operator_targets=torch.tensor(
                [
                    tuple(Operator).index(Operator.O1_SNAP),
                    tuple(Operator).index(Operator.O2_UNIQUE),
                    tuple(Operator).index(Operator.E1_ACTION),
                    tuple(Operator).index(Operator.E2_PERIODIC),
                ]
            ),
            time_mode_targets=torch.ones(4, dtype=torch.int64),
            span_start_targets=torch.zeros(4, dtype=torch.int64),
            span_end_targets=torch.zeros(4, dtype=torch.int64),
            operator_provenance=(synthetic,) * 4,
            time_provenance=(synthetic,) * 4,
            span_provenance=(synthetic,) * 4,
        ),
    )


def test_tiny_a2_runs_real_hard_state_reader_composer_and_state_answer_loss() -> None:
    torch.manual_seed(15)
    config = load_config()
    owner = RuntimeOwner(
        ("video-o1", "video-o2", "video-e1", "video-e2"),
        ("trajectory-o1", "trajectory-o2", "trajectory-e1", "trajectory-e2"),
    )
    state_bank: StructuredStateBank = build_state_bank(config)
    identity_bank = build_identity_bank(config)
    retriever: EmbeddingStateRetriever = build_state_retriever(config)
    writer = StageABankWriter(state_bank, identity_bank)
    runtime = writer.reset(owner)
    slots = torch.randn((4, 2, 768))
    hidden = torch.randn((4, 3, 768))
    with torch.no_grad():
        retrieval_queries = torch.stack(
            (
                state_bank.project(slots[0].mean(dim=0), HeadType.O1),
                state_bank.project(slots[1, 0], HeadType.O2),
                state_bank.project(hidden[2, -1], HeadType.E1),
                state_bank.project(hidden[3, -1], HeadType.E2),
            )
        )
    query = _query(owner)
    query = replace(query, embeddings=replace(query.embeddings, q_target=retrieval_queries))
    tokenizer = _Tokenizer()
    qwen = _TinyQwenPrefill(len(tokenizer))
    decoder = _ForbiddenDecode()
    capture = _CaptureMetrics()
    model = StateTTTModel(
        config,
        ModelComponents(
            visual_stage=_VisualStage(),
            query_encoder=_QueryStage(query),
            composer=compose_inputs,
            qwen_prefill=qwen,
            qwen_decode=decoder,
            fast_adapter=_FastStage(),
            spatial_encoder=_SpatialStage(slots),
            temporal_encoder=_TemporalStage(hidden),
            observation_heads=_ObservationStage(),
            state_bank=state_bank,
            bank_writer=writer,
            retriever=retriever,
            reader=DeterministicStateReader(tokenizer),
            resampler=_TinyResampler(),
        ),
        ModelFeatureFlags(),
    )
    runner = StageAEpisodeRunner(
        model=model,
        variant=StageAVariant.A2,
        metric_builder=capture,
    )
    forward = StageATypedForwardAdapter(StageAVariant.A2, runner)

    base_row = (3, 4, 2, 5, 1, 3, 6, 7, 8, 1)
    base_input_ids = torch.tensor([base_row] * 4, dtype=torch.int64)
    base_attention_mask = torch.ones_like(base_input_ids)
    base_labels = torch.full_like(base_input_ids, -100)
    base_labels[:, 8:] = base_input_ids[:, 8:]
    missing = TargetProvenance.MISSING
    synthetic = TargetProvenance.SYNTHETIC_EXPLICIT
    supervision = StageASupervisionBatch(
        answer=AnswerTargetLabels(
            base_labels=base_labels,
            base_number_token_mask=torch.zeros_like(base_input_ids, dtype=torch.bool),
            target_counts=torch.full((4,), -100, dtype=torch.int64),
            answer_provenance=(synthetic,) * 4,
            count_provenance=(missing,) * 4,
        ),
        state=_state_targets(),
    )
    episode = StageAEpisodeInputs(
        owner=owner,
        observation_requests=(
            ObservationChunkRequest(
                owner=owner,
                video_input=torch.ones((4, 1)),
                query_input="synthetic-explicit-query",
                runtime_state=runtime,
                bank_states=runtime.state_bank_states,
                inference=False,
            ),
        ),
        answer=StageAEpisodeAnswerInputs(
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            pixel_values_videos=torch.ones((4, 1)),
            video_grid_thw=torch.ones((4, 3), dtype=torch.int64),
            tokenizer=tokenizer,
            embedding_owner=qwen,
            rope_indexer=_TinyRopeIndexer(),
        ),
    )
    batch = StageATrainingBatch(
        runtime_queries=tuple(
            RuntimeQueryInput(
                video_id=video_id,
                trajectory_id=trajectory_id,
                query_id=f"query-{row}",
                query_index=row,
                video=Path(f"video-{row}.mp4"),
                question="how many",
                query_time=2.0,
                explicit_time_values=(),
            )
            for row, (video_id, trajectory_id) in enumerate(
                zip(
                owner.video_ids,
                owner.trajectory_ids,
                strict=True,
                )
            )
        ),
        model_inputs=episode,
        supervision=supervision,
    )

    adapted = forward(batch, training=True)
    adapted.audit.validate_for(StageAVariant.A2)
    losses = compute_stage_a_losses(
        StageAVariant.A2,
        answer=adapted.answer_loss_input,
        state=adapted.state_loss_input,
    )
    assert losses.state is not None
    assert torch.equal(losses.total, losses.state.total + losses.answer.loss.value)
    assert torch.isfinite(losses.total)
    losses.total.backward()

    output = capture.output
    assert output is not None
    assert qwen.prefill_calls == 1
    assert decoder.calls == 0
    assert adapted.audit.decode_step_count == 0
    assert (
        adapted.audit.inner_sgd_attempted,
        adapted.audit.inner_sgd_updated,
        adapted.audit.inner_sgd_skipped,
    ) == (0, 0, 0)
    assert isinstance(output.runtime_state, BatchRuntimeState)
    assert output.runtime_state.next_chunk_index == 1
    assert all(
        row.fast_weights is None and row.optimizer is None for row in output.runtime_state.rows
    )
    hard_head_types = tuple(
        state.records[0].head_type for state in output.runtime_state.state_bank_states
    )
    assert hard_head_types == (
        HeadType.O1,
        HeadType.O2,
        HeadType.E1,
        HeadType.E2,
    )
    assert all(
        not record.semantic_embedding.requires_grad and record.semantic_embedding.grad_fn is None
        for state in output.runtime_state.state_bank_states
        for record in state.records
    )
    assert len(output.reader) == 4
    assert all(isinstance(result, ReaderResult) for result in output.reader)
    assert tuple(result.operator for result in output.reader) == query.hard_operators

    mapped_labels = adapted.answer_loss_input.labels
    assert output.composed.number_position_mask.any()
    assert torch.all(mapped_labels[output.composed.state_position_mask] == -100)
    assert torch.all(mapped_labels[output.composed.number_position_mask] == -100)
    assert torch.equal(
        adapted.answer_loss_input.number_token_mask
        & (output.composed.state_position_mask | output.composed.number_position_mask),
        torch.zeros_like(adapted.answer_loss_input.number_token_mask),
    )

    soft = output.soft_intermediates.state_write
    assert isinstance(soft, StageASoftWriteOutput)
    assert all(
        value.requires_grad
        for value in (soft.o1_semantics, soft.o2_semantics, soft.e1_semantics, soft.e2_semantics)
    )
    state_bank.semantic_projector.zero_grad(set_to_none=True)
    (
        soft.o1_semantics.square().mean()
        + soft.o2_semantics.square().mean()
        + soft.e1_semantics.square().mean()
        + soft.e2_semantics.square().mean()
    ).backward()
    projector_gradients = tuple(
        parameter.grad for parameter in state_bank.semantic_projector.parameters()
    )
    assert all(value is not None and torch.isfinite(value).all() for value in projector_gradients)
    assert any(
        float(value.abs().sum().item()) > 0.0 for value in projector_gradients if value is not None
    )
