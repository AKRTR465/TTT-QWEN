from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import MetaTTTVariant, ProjectConfig, load_config
from ttt_svcbench_qwen.fast_ttt import FastTTTForwardAudit, FastWeightsState
from ttt_svcbench_qwen.identity_bank import (
    IdentityDecisionStatus,
    IdentityObservationDecision,
)
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    O1StateTarget,
    StateLossInput,
    TemporalPredictor,
)
from ttt_svcbench_qwen.meta_trainer import (
    MetaCausalChunk,
    MetaGradientReferenceMode,
    MetaModelRuntime,
    MetaQueryLossInput,
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    MetaTTTQueryPoint,
    MetaTTTTrainer,
    StageAQueryLossBuilder,
    SyntheticAblationRecord,
    audit_variant_isolation,
    compare_synthetic_ablations,
    render_synthetic_ablation_report,
    run_meta_gradient_reference,
)
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
    VisualStageOutput,
)
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
from ttt_svcbench_qwen.stage_a_runtime import StageAWriteAudit
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    StageATargetBatch,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import HeadType
from ttt_svcbench_qwen.state_encoder import TemporalCache, TemporalEncoderOutput
from ttt_svcbench_qwen.trainer import StageAEpisodeAnswerInputs, StageASupervisionBatch


@dataclass(frozen=True, slots=True)
class _VideoChunk:
    features: Tensor
    timestamps: Tensor
    position_ids: Tensor
    valid_mask: Tensor
    identity_timestamp: float = 0.0
    identity_position_id: int = 0


@dataclass(frozen=True, slots=True)
class _TinyRuntime:
    owner: RuntimeOwner
    next_chunk_index: int


@dataclass(frozen=True, slots=True)
class _TinyBank:
    version: int


class _RuntimeResetter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, owner: RuntimeOwner) -> MetaModelRuntime:
        self.calls += 1
        return MetaModelRuntime(
            _TinyRuntime(owner, 0), tuple(_TinyBank(0) for _ in owner.video_ids)
        )


class _TinyFastController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        first = torch.zeros((768, 768), dtype=torch.float32)
        second = torch.zeros_like(first)
        first[0, 0] = 0.2
        second[0, 0] = 0.3
        self.w0_1 = nn.Parameter(first)
        self.w0_2 = nn.Parameter(second)
        self._active: tuple[FastWeightsState, ...] | None = None
        self.last_audit: FastTTTForwardAudit | None = None

    def reset_fast_state(
        self,
        state: FastWeightsState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastWeightsState:
        del state
        mode = bool(differentiable)
        if mode:
            w0_1: Tensor = self.w0_1
            w0_2: Tensor = self.w0_2
            w_t_1 = self.w0_1.clone()
            w_t_2 = self.w0_2.clone()
        else:
            w0_1 = self.w0_1.detach().clone()
            w0_2 = self.w0_2.detach().clone()
            w_t_1 = w0_1.clone().requires_grad_(True)
            w_t_2 = w0_2.clone().requires_grad_(True)
        return FastWeightsState(w0_1, w0_2, w_t_1, w_t_2, 0, 0, 0, mode)

    @contextmanager
    def use_fast_state(
        self,
        state: FastWeightsState | Sequence[FastWeightsState],
    ) -> Iterator[object]:
        if self._active is not None:
            raise RuntimeError("tiny fast binding is not re-entrant")
        self._active = (state,) if isinstance(state, FastWeightsState) else tuple(state)
        try:
            yield self
        finally:
            self._active = None

    def collect_meta_fast_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.w0_1, self.w0_2)

    def forward(
        self,
        visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        if self._active is None or not isinstance(visual.value, Tensor):
            raise RuntimeError("tiny fast stage requires one managed binding")
        gains = torch.stack([state.w_t_1[0, 0] * state.w_t_2[0, 0] for state in self._active])
        adapted = visual.value + gains[:, None, None]
        residual = adapted - visual.value
        self.last_audit = FastTTTForwardAudit(
            fast_versions=tuple(state.fast_version for state in self._active),
            update_counts=tuple(state.update_count for state in self._active),
            valid_token_counts=tuple(adapted.shape[1] for _ in self._active),
            used_runtime_state=True,
            w_t_1_norms=tuple(float(state.w_t_1.detach().norm()) for state in self._active),
            w_t_2_norms=tuple(float(state.w_t_2.detach().norm()) for state in self._active),
            input_norms=tuple(
                float(visual.value[row].detach().norm()) for row in range(len(gains))
            ),
            residual_norms=tuple(float(residual[row].detach().norm()) for row in range(len(gains))),
        )
        return replace(visual, value=adapted, prepared_video_features=adapted)


class _VisualStage(nn.Module):
    def forward(self, request: ObservationChunkRequest) -> VisualStageOutput:
        if not isinstance(request.video_input, _VideoChunk):
            raise TypeError("tiny visual stage requires _VideoChunk")
        return VisualStageOutput(
            value=request.video_input.features,
            prepared_video_features=request.video_input.features,
        )


class _QueryStage(nn.Module):
    def forward(self, query_input: object, *, inference: bool) -> object:
        if inference or not isinstance(query_input, Tensor):
            raise ValueError("tiny query stage requires a training Tensor")
        return SimpleNamespace(q_target=query_input)


class _SpatialStage:
    def __call__(
        self,
        visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> object:
        return visual.value


class _TemporalStage:
    def __call__(
        self,
        visual: VisualStageOutput,
        _query: object,
        request: ObservationChunkRequest,
    ) -> TemporalEncoderOutput:
        if not isinstance(visual.value, Tensor) or not isinstance(request.video_input, _VideoChunk):
            raise TypeError("tiny temporal stage inputs are invalid")
        payload = request.video_input
        return TemporalEncoderOutput(
            hidden=visual.value,
            timestamps=payload.timestamps,
            position_ids=payload.position_ids,
            valid_mask=payload.valid_mask,
            cache=_empty_cache(request.owner, visual.value),
        )


class _ObservationStage:
    def __init__(self) -> None:
        self.outputs: list[ObservationOutputs] = []

    def __call__(
        self,
        _spatial: object,
        temporal: object,
        _query: object,
        request: ObservationChunkRequest,
    ) -> ObservationOutputs:
        if not isinstance(temporal, TemporalEncoderOutput) or not isinstance(
            request.video_input, _VideoChunk
        ):
            raise TypeError("tiny observation stage inputs are invalid")
        hidden = temporal.hidden
        batch_size, width = hidden.shape[:2]
        slot_logits = hidden.mean(dim=1, keepdim=True)[..., :6]
        slot_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=hidden.device)
        slot_times = torch.full(
            (batch_size, 1),
            request.video_input.identity_timestamp,
            dtype=torch.float64,
            device=hidden.device,
        )
        slot_positions = torch.full(
            (batch_size, 1),
            request.video_input.identity_position_id,
            dtype=torch.int64,
            device=hidden.device,
        )
        identity = F.normalize(hidden.mean(dim=1, keepdim=True)[..., :256].float(), dim=-1)
        identity = identity.to(dtype=hidden.dtype)
        score_logits = hidden.mean(dim=1, keepdim=True)[..., :2]
        e1_logits = hidden[..., :3]
        e2_event_logits = hidden[..., :4]
        e2_phase_logits = hidden[..., 4:8]
        output = ObservationOutputs(
            o1=O1SoftOutput(
                logits=slot_logits,
                probabilities=torch.sigmoid(slot_logits),
                soft_count=torch.sigmoid(slot_logits[..., :3]).prod(dim=-1).sum(dim=1),
                valid_mask=slot_mask,
                timestamps=slot_times,
                position_ids=slot_positions,
            ),
            o2=O2SoftOutput(
                identity=identity,
                score_logits=score_logits,
                score_probabilities=torch.sigmoid(score_logits),
                valid_mask=slot_mask.clone(),
                timestamps=slot_times.clone(),
                position_ids=slot_positions.clone(),
            ),
            e1=E1SoftOutput(
                logits=e1_logits,
                probabilities=torch.sigmoid(e1_logits),
                valid_mask=temporal.valid_mask,
                timestamps=temporal.timestamps,
                position_ids=temporal.position_ids,
                next_states=tuple(
                    _e1_state(request.owner, row, temporal.position_ids, temporal.timestamps)
                    for row in range(batch_size)
                ),
                audit=StreamReplayAudit(
                    "e1", (width,) * batch_size, (0,) * batch_size, (width,) * batch_size
                ),
            ),
            e2=E2SoftOutput(
                event_logits=e2_event_logits,
                phase_logits=e2_phase_logits,
                event_probabilities=torch.sigmoid(e2_event_logits),
                phase_probabilities=torch.softmax(e2_phase_logits.float(), dim=-1).to(hidden.dtype),
                valid_mask=temporal.valid_mask.clone(),
                timestamps=temporal.timestamps.clone(),
                position_ids=temporal.position_ids.clone(),
                next_states=tuple(
                    _e2_state(request.owner, row, temporal.position_ids, temporal.timestamps)
                    for row in range(batch_size)
                ),
                audit=StreamReplayAudit(
                    "e2", (width,) * batch_size, (0,) * batch_size, (width,) * batch_size
                ),
            ),
        )
        self.outputs.append(output)
        return output


class _BankWriter:
    def __call__(
        self,
        _observations: object,
        _spatial: object,
        _temporal: object,
        _query: object,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput:
        if not isinstance(request.runtime_state, _TinyRuntime):
            raise TypeError("tiny writer requires _TinyRuntime")
        runtime = request.runtime_state
        next_runtime = _TinyRuntime(runtime.owner, runtime.next_chunk_index + 1)
        next_banks = tuple(
            _TinyBank(bank.version + 1) if isinstance(bank, _TinyBank) else _TinyBank(1)
            for bank in request.bank_states
        )
        decisions = tuple(
            (
                IdentityObservationDecision(
                    slot_index=0,
                    position_id=0,
                    timestamp=0.0,
                    status=IdentityDecisionStatus.CONFIRMED_UPDATED,
                    identity_id=f"identity-{row}",
                    similarity=1.0,
                    novelty=0.0,
                    match_confidence=1.0,
                    scanned_confirmed_count=1,
                ),
            )
            for row in range(len(request.owner.video_ids))
        )
        audit = StageAWriteAudit(
            chunk_index=runtime.next_chunk_index,
            head_types=(HeadType.O2,) * len(request.owner.video_ids),
            bank_versions_before=tuple(
                bank.version if isinstance(bank, _TinyBank) else 0 for bank in request.bank_states
            ),
            bank_versions_after=tuple(bank.version for bank in next_banks),
            record_counts_after=(1,) * len(next_banks),
            identity_counts_after=(1,) * len(next_banks),
            identity_decisions=decisions,
            skipped_rows=(),
        )
        return BankWriteOutput(next_runtime, next_banks, audit)


class _Retriever:
    def retrieve_query(
        self,
        _state_bank: object,
        states: Sequence[object],
        _query: object,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> object:
        if len(states) != len(video_ids) or len(states) != len(trajectory_ids):
            raise ValueError("tiny retrieval ownership mismatch")
        return tuple(state.version if isinstance(state, _TinyBank) else 0 for state in states)


@dataclass(frozen=True, slots=True)
class _ReaderResult:
    exact_count: int


class _Reader:
    def __init__(self) -> None:
        self.calls: list[tuple[_ReaderResult, ...]] = []

    def read(self, retrieval: object) -> Sequence[object]:
        if not isinstance(retrieval, tuple):
            raise TypeError("tiny Reader requires tuple retrieval")
        results = tuple(_ReaderResult(int(value)) for value in retrieval)
        self.calls.append(results)
        return results

    def audit_results(
        self,
        _retrieval: object,
        results: Sequence[object],
    ) -> Sequence[object]:
        return results

    def audit_number_tokens(self, result: object) -> int | None:
        return result.exact_count if isinstance(result, _ReaderResult) else None


class _Composer:
    def __init__(self) -> None:
        self.input_ids: list[Tensor] = []

    def __call__(self, **kwargs: object) -> object:
        input_ids = kwargs["base_input_ids"]
        attention = kwargs["base_attention_mask"]
        if not isinstance(input_ids, Tensor) or not isinstance(attention, Tensor):
            raise TypeError("tiny Composer inputs must be tensors")
        self.input_ids.append(input_ids.detach().clone())
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=torch.arange(input_ids.shape[1]).expand_as(input_ids),
            rope_deltas=torch.zeros((input_ids.shape[0], 1), dtype=torch.int64),
            state_position_mask=None,
        )


class _Qwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prepared_features: list[Tensor] = []

    def forward(self, request: QwenPrefillRequest) -> object:
        if not isinstance(request.input_ids, Tensor) or not isinstance(
            request.prepared_video_features, Tensor
        ):
            raise TypeError("tiny Qwen inputs must be tensors")
        self.prepared_features.append(request.prepared_video_features.detach().clone())
        score = request.prepared_video_features.float().mean(dim=(1, 2))
        zeros = torch.zeros_like(score)
        row = torch.stack((score, -score, zeros), dim=-1)
        logits = row[:, None, :].expand(-1, request.input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class _Decode:
    def __call__(self, _inputs: object) -> object:
        raise AssertionError("Meta-TTT training must not decode")


class _TinyPredictor(TemporalPredictor):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.input_dim = 768
        self.output_dim = 768
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden * self.scale


class _TinyQueryLossBuilder:
    def __call__(
        self,
        output: StateTTTModelOutput,
        *,
        answer: StageAEpisodeAnswerInputs,
        supervision: StageASupervisionBatch,
    ) -> MetaQueryLossInput:
        del answer
        if not isinstance(output.answer_logits, Tensor) or not isinstance(
            output.observations, ObservationOutputs
        ):
            raise TypeError("tiny Query output has the wrong type")
        labels = supervision.answer.base_labels.to(output.answer_logits.device)
        if labels.shape != output.answer_logits.shape[:2]:
            raise ValueError("tiny Query labels must align to Qwen logits")
        number_mask = torch.zeros_like(labels, dtype=torch.bool)
        o1 = output.observations.o1
        return MetaQueryLossInput(
            answer=AnswerLossInput(output.answer_logits, labels, number_mask),
            state=StateLossInput(
                batch_size=output.answer_logits.shape[0],
                o1=O1StateTarget(
                    row_indices=torch.arange(output.answer_logits.shape[0]),
                    logits=o1.logits,
                    targets=torch.zeros_like(o1.logits),
                    slot_mask=o1.valid_mask,
                ),
            ),
        )


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config()


def _system(
    config: ProjectConfig,
    variant: MetaTTTVariant,
) -> tuple[MetaTTTEpisodeRunner, _TinyFastController, _TinyPredictor, _RuntimeResetter]:
    fast = _TinyFastController()
    predictor = _TinyPredictor()
    resetter = _RuntimeResetter()
    reader = _Reader()
    model = StateTTTModel(
        config,
        ModelComponents(
            visual_stage=_VisualStage(),
            query_encoder=_QueryStage(),
            fast_adapter=fast,
            spatial_encoder=_SpatialStage(),
            temporal_encoder=_TemporalStage(),
            observation_heads=_ObservationStage(),
            state_bank=object(),
            bank_writer=_BankWriter(),
            retriever=_Retriever(),
            reader=reader,
            composer=_Composer(),
            qwen_prefill=_Qwen(),
            qwen_decode=_Decode(),
        ),
        ModelFeatureFlags(
            fast_enabled=True,
            bank_enabled=True,
            reader_enabled=True,
            state_tokens_enabled=False,
        ),
    )
    runner = MetaTTTEpisodeRunner(
        config=config,
        model=model,
        fast_controller=fast,
        predictor=predictor,
        runtime_resetter=resetter,
        variant=variant,
        query_loss_builder=_TinyQueryLossBuilder(),
    )
    return runner, fast, predictor, resetter


def _episode(
    config: ProjectConfig,
    variant: MetaTTTVariant,
    *,
    support_count: int,
    query_count: int,
    invalid_first_support: bool = False,
) -> MetaTTTEpisode:
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    supports = tuple(
        _chunk(
            owner,
            chunk_index=index,
            end_time=float(index + 1),
            width=1 if index == 0 and invalid_first_support else 2,
        )
        for index in range(support_count)
    )
    queries = tuple(
        MetaTTTQueryPoint(
            chunk=_chunk(
                owner,
                chunk_index=support_count + index,
                end_time=float(support_count + index + 1),
                width=2,
            ),
            query_time=float(support_count + index + 1),
            answer=_answer_inputs(),
            supervision=_supervision(),
            task_name="synthetic-count",
            case_id=f"case-{index}",
        )
        for index in range(query_count)
    )
    seed = config.stage_b.seed if variant is MetaTTTVariant.A3 else config.stage_c.seed
    return MetaTTTEpisode(owner, supports, queries, seed)


def _chunk(
    owner: RuntimeOwner,
    *,
    chunk_index: int,
    end_time: float,
    width: int,
) -> MetaCausalChunk:
    positions = torch.arange(chunk_index, chunk_index + width, dtype=torch.int64).unsqueeze(0)
    timestamps = positions.to(torch.float64)
    base = 0.05 + 0.01 * positions.to(torch.float32)
    features = base.unsqueeze(-1).expand(1, width, 768).clone()
    payload = _VideoChunk(
        features=features,
        timestamps=timestamps,
        position_ids=positions,
        valid_mask=torch.ones((1, width), dtype=torch.bool),
    )
    runtime_payload: Mapping[str, object] = {
        "video": "synthetic-video",
        "question": "how many",
        "query_time": end_time,
        "explicit_time_values": (),
    }
    return MetaCausalChunk(
        request=ObservationChunkRequest(
            owner=owner,
            video_input=payload,
            query_input=torch.zeros((1, 512)),
            runtime_state=object(),
            bank_states=(object(),),
            inference=False,
        ),
        start_time=max(0.0, end_time - 1.0),
        end_time=end_time,
        runtime_payload=runtime_payload,
    )


def _answer_inputs() -> StageAEpisodeAnswerInputs:
    return StageAEpisodeAnswerInputs(
        base_input_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        base_attention_mask=torch.ones((1, 3), dtype=torch.int64),
        pixel_values_videos=None,
        video_grid_thw=None,
        tokenizer=object(),
        embedding_owner=object(),
        rope_indexer=object(),
    )


def _supervision(label: int = 0) -> StageASupervisionBatch:
    synthetic = TargetProvenance.SYNTHETIC_EXPLICIT
    return StageASupervisionBatch(
        answer=AnswerTargetLabels(
            base_labels=torch.full((1, 3), label, dtype=torch.int64),
            base_number_token_mask=torch.zeros((1, 3), dtype=torch.bool),
            target_counts=torch.tensor([0], dtype=torch.int64),
            answer_provenance=(synthetic,),
            count_provenance=(synthetic,),
        ),
        state=StageATargetBatch(),
    )


def _empty_cache(owner: RuntimeOwner, reference: Tensor) -> TemporalCache:
    batch_size = len(owner.video_ids)
    empty_hidden = torch.zeros((batch_size, 0, 768), dtype=reference.dtype)
    empty_kv = tuple(torch.zeros((batch_size, 12, 0, 64), dtype=reference.dtype) for _ in range(6))
    replay_kv = tuple(torch.zeros((batch_size, 12, 0, 64), dtype=reference.dtype) for _ in range(6))
    return TemporalCache(
        hidden=empty_hidden,
        layer_keys=empty_kv,
        layer_values=tuple(value.clone() for value in empty_kv),
        replay_layer_keys=replay_kv,
        replay_layer_values=tuple(value.clone() for value in replay_kv),
        timestamps=torch.zeros((batch_size, 0), dtype=torch.float64),
        replay_timestamps=torch.zeros((batch_size, 0), dtype=torch.float64),
        position_ids=torch.zeros((batch_size, 0), dtype=torch.int64),
        replay_position_ids=torch.zeros((batch_size, 0), dtype=torch.int64),
        valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool),
        replay_valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool),
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
        query_signatures=torch.zeros((batch_size, 512), dtype=reference.dtype),
        total_seen=torch.zeros(batch_size, dtype=torch.int64),
    )


def _e1_state(
    owner: RuntimeOwner,
    row: int,
    positions: Tensor,
    timestamps: Tensor,
) -> E1RuntimeState:
    total_seen = int(positions[row, -1].item()) + 1
    length = min(total_seen, 66)
    start = total_seen - length
    state_positions = torch.arange(start, total_seen, dtype=torch.int64)
    state_times = state_positions.to(torch.float64)
    return E1RuntimeState(
        video_id=owner.video_ids[row],
        trajectory_id=owner.trajectory_ids[row],
        query_signature=torch.zeros(512),
        projected_history=torch.zeros((length, 512)),
        timestamps=state_times,
        position_ids=state_positions,
        total_seen=total_seen,
    )


def _e2_state(
    owner: RuntimeOwner,
    row: int,
    positions: Tensor,
    timestamps: Tensor,
) -> E2RuntimeState:
    del timestamps
    total_seen = int(positions[row, -1].item()) + 1
    length = min(total_seen, 5)
    start = total_seen - length
    state_positions = torch.arange(start, total_seen, dtype=torch.int64)
    checkpoint = torch.zeros((length, 2, 768))
    return E2RuntimeState(
        video_id=owner.video_ids[row],
        trajectory_id=owner.trajectory_ids[row],
        query_signature=torch.zeros(512),
        hidden=checkpoint[-1].clone(),
        checkpoint_hidden=checkpoint,
        timestamps=state_positions.to(torch.float64),
        position_ids=state_positions,
        total_seen=total_seen,
    )


def _graph_node_count(value: Tensor) -> int:
    root = value.grad_fn
    if root is None:
        return 0
    stack = [root]
    seen: set[int] = set()
    retained_nodes: list[object] = []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        # Keep Python wrappers alive while traversing. Otherwise CPython may
        # reuse an id for a later autograd node and make the count flaky.
        retained_nodes.append(node)
        stack.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    return len(retained_nodes)


def test_variant_isolation_and_explicit_first_order_reference(config: ProjectConfig) -> None:
    audit = audit_variant_isolation(config)
    assert audit.a4_minus_a3 == ("identity",)
    assert audit.a5_minus_a4 == ("event",)

    def support(first: Tensor, second: Tensor) -> Tensor:
        return 0.5 * (first.square() + 3.0 * second.square()).sum()

    def query(first: Tensor, second: Tensor) -> Tensor:
        return 0.5 * (first + second).square().sum()

    first_order = run_meta_gradient_reference(
        initial_parameters=(
            torch.tensor([1.2], dtype=torch.float64, requires_grad=True),
            torch.tensor([-0.7], dtype=torch.float64, requires_grad=True),
        ),
        support_loss=support,
        query_loss=query,
        learning_rate=0.1,
        mode=MetaGradientReferenceMode.FIRST_ORDER,
    )
    full = run_meta_gradient_reference(
        initial_parameters=(
            torch.tensor([1.2], dtype=torch.float64, requires_grad=True),
            torch.tensor([-0.7], dtype=torch.float64, requires_grad=True),
        ),
        support_loss=support,
        query_loss=query,
        learning_rate=0.1,
        mode=MetaGradientReferenceMode.FULL_SECOND_ORDER,
    )
    adapted_sum = (1.0 - 0.1) * 1.2 + (1.0 - 0.3) * -0.7
    assert torch.allclose(
        full.meta_gradients[0], torch.tensor([adapted_sum * 0.9], dtype=torch.float64)
    )
    assert torch.allclose(
        full.meta_gradients[1], torch.tensor([adapted_sum * 0.7], dtype=torch.float64)
    )
    assert torch.allclose(
        first_order.meta_gradients[0], torch.tensor([adapted_sum], dtype=torch.float64)
    )
    assert torch.allclose(
        first_order.meta_gradients[1], torch.tensor([adapted_sum], dtype=torch.float64)
    )
    assert not torch.equal(first_order.meta_gradients[0], full.meta_gradients[0])


def test_a3_runner_and_outer_step_reach_both_meta_fast_matrices(config: ProjectConfig) -> None:
    runner, fast, predictor, resetter = _system(config, MetaTTTVariant.A3)
    episode = _episode(config, MetaTTTVariant.A3, support_count=1, query_count=1)
    optimizer = torch.optim.SGD((*fast.parameters(), *predictor.parameters()), lr=0.05)
    trainer = MetaTTTTrainer(runner=runner, optimizer=optimizer, outer_grad_clip_norm=1.0)
    output = trainer.train_step(episode)

    assert output.global_step == 1
    assert output.audit.optimizer_step_applied
    assert output.audit.meta_fast_gradient_norms is not None
    assert min(output.audit.meta_fast_gradient_norms) > 0.0
    assert min(output.audit.meta_fast_delta_norms) > 0.0
    assert output.episode.audit.active_terms == ("pred",)
    assert output.episode.audit.update_count == 1
    assert output.episode.audit.updates[0].fast_versions_before == (0,)
    assert output.episode.audit.updates[0].fast_versions_after == (1,)
    assert output.episode.audit.queries[0].after_fast_versions == (1,)
    assert output.episode.audit.queries[0].before_fast_versions == (0,)
    assert resetter.calls == 2


def test_invalid_support_skips_update_but_keeps_causal_query(config: ProjectConfig) -> None:
    runner, _, _, _ = _system(config, MetaTTTVariant.A3)
    episode = _episode(
        config,
        MetaTTTVariant.A3,
        support_count=1,
        query_count=1,
        invalid_first_support=True,
    )
    output = runner(episode)
    update = output.audit.updates[0]
    assert update.did_update == (False,)
    assert update.skip_reasons == ("insufficient_time",)
    assert update.fast_versions_after == (0,)
    assert output.final_fast_states[0].skip_count == 1
    assert output.audit.query_count == 1
    output.total.backward()


def test_stage_c_invalid_chunk_skips_then_later_supports_continue(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    output = runner(
        _episode(
            config,
            MetaTTTVariant.A5,
            support_count=4,
            query_count=2,
            invalid_first_support=True,
        )
    )
    assert output.audit.updates[0].did_update == (False,)
    assert output.audit.updates[0].skip_reasons == ("insufficient_time",)
    assert [update.fast_versions_before for update in output.audit.updates] == [
        (0,),
        (0,),
        (1,),
        (2,),
    ]
    assert output.final_fast_states[0].update_count == 3
    assert output.final_fast_states[0].skip_count == 1
    assert len(output.audit.queries) == 2


@pytest.mark.parametrize("support_count", [1, 4, 8])
def test_a5_support_schedule_is_bounded_and_next_only(
    config: ProjectConfig,
    support_count: int,
) -> None:
    runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    output = runner(_episode(config, MetaTTTVariant.A5, support_count=support_count, query_count=2))
    assert output.audit.support_count == support_count
    assert output.audit.retained_support_graph_count == support_count
    assert output.audit.graph_bound == 8
    assert output.audit.update_attempt_count == support_count
    assert all(update.next_only_verified for update in output.audit.updates)
    assert output.final_fast_states[0].update_count == support_count
    if support_count > 1:
        assert sum(output.audit.updates[1].e1_valid_counts) > 0
        assert sum(output.audit.updates[1].e2_valid_counts) > 0
        assert output.audit.updates[1].match.snapshot_detached
        assert output.audit.updates[1].match.snapshot_storage_isolated
        assert output.audit.updates[1].match.authoritative_identity_update_evidence
        assert output.audit.updates[1].match.identity_decision_storage_free


def test_a4_a5_terms_multi_query_and_repeatability(config: ProjectConfig) -> None:
    a4_runner, _, _, _ = _system(config, MetaTTTVariant.A4)
    a5_runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    a4_episode = _episode(config, MetaTTTVariant.A4, support_count=4, query_count=2)
    a5_episode = _episode(config, MetaTTTVariant.A5, support_count=4, query_count=2)
    a4 = a4_runner(a4_episode)
    a5 = a5_runner(a5_episode)
    repeated = a5_runner(a5_episode)

    assert a4.audit.active_terms == ("pred", "identity")
    assert a5.audit.active_terms == ("pred", "identity", "event")
    assert all(sum(update.e1_valid_counts) == 0 for update in a4.audit.updates)
    assert sum(a5.audit.updates[1].e1_valid_counts) > 0
    assert len(a5.query_objectives) == 2
    assert all(query.independent_lifecycles for query in a5.audit.queries)
    assert all(query.observation_immutable for query in a5.audit.queries)
    assert torch.equal(a5.total.detach(), repeated.total.detach())
    assert torch.equal(
        a5.final_fast_states[0].w_t_1.detach(),
        repeated.final_fast_states[0].w_t_1.detach(),
    )
    assert torch.allclose(
        a5.total,
        torch.stack(tuple(query.outer.total for query in a5.query_objectives)).mean(),
    )


def test_eight_support_graph_is_released_and_does_not_grow(config: ProjectConfig) -> None:
    runner, _, _, _ = _system(config, MetaTTTVariant.A5)

    first_episode = _episode(config, MetaTTTVariant.A5, support_count=8, query_count=2)
    first = runner(first_episode)
    first_node_count = _graph_node_count(first.total)
    first_total = first.total
    first_support = first.support_ttt[0].per_row_total
    first_total_ref = weakref.ref(first_total)
    first_support_ref = weakref.ref(first_support)
    first.total.backward()
    observations = runner.model.components.observation_heads
    assert isinstance(observations, _ObservationStage)
    observations.outputs.clear()
    del first_total, first_support, first, first_episode
    gc.collect()
    assert first_total_ref() is None
    assert first_support_ref() is None

    second_episode = _episode(config, MetaTTTVariant.A5, support_count=8, query_count=2)
    second = runner(second_episode)
    second_node_count = _graph_node_count(second.total)
    assert second_node_count == first_node_count
    second_total = second.total
    second_support = second.support_ttt[0].per_row_total
    second_total_ref = weakref.ref(second_total)
    second_support_ref = weakref.ref(second_support)
    second.total.backward()
    observations.outputs.clear()
    del second_total, second_support, second, second_episode
    gc.collect()
    assert second_total_ref() is None
    assert second_support_ref() is None


def test_query_future_and_prefill_reuse_are_rejected(config: ProjectConfig) -> None:
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    chunk = _chunk(owner, chunk_index=1, end_time=2.0, width=2)
    with pytest.raises(ValueError, match="future"):
        MetaTTTQueryPoint(
            chunk=chunk,
            query_time=1.5,
            answer=_answer_inputs(),
            supervision=_supervision(),
            task_name="task",
            case_id="case",
        )
    runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    output = runner(_episode(config, MetaTTTVariant.A5, support_count=1, query_count=2))
    assert all(query.before_prefill_count == 1 for query in output.audit.queries)
    assert all(query.after_prefill_count == 1 for query in output.audit.queries)


def test_later_query_labels_cannot_change_earlier_query_path(config: ProjectConfig) -> None:
    clean_runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    changed_runner, _, _, _ = _system(config, MetaTTTVariant.A5)
    clean_episode = _episode(config, MetaTTTVariant.A5, support_count=4, query_count=2)
    changed_episode = replace(
        clean_episode,
        query_points=(
            clean_episode.query_points[0],
            replace(clean_episode.query_points[1], supervision=_supervision(label=1)),
        ),
    )
    clean = clean_runner(clean_episode)
    changed = changed_runner(changed_episode)

    assert torch.equal(
        clean.query_objectives[0].outer.total.detach(),
        changed.query_objectives[0].outer.total.detach(),
    )
    assert clean.audit.queries[0].before == changed.audit.queries[0].before
    assert clean.audit.queries[0].after == changed.audit.queries[0].after
    clean_observations = clean_runner.model.components.observation_heads
    changed_observations = changed_runner.model.components.observation_heads
    assert isinstance(clean_observations, _ObservationStage)
    assert isinstance(changed_observations, _ObservationStage)
    first_query_call = 2 * len(clean_episode.support_chunks)
    assert torch.equal(
        clean_observations.outputs[first_query_call].o1.logits.detach(),
        changed_observations.outputs[first_query_call].o1.logits.detach(),
    )
    clean_reader = clean_runner.model.components.reader
    changed_reader = changed_runner.model.components.reader
    assert isinstance(clean_reader, _Reader)
    assert isinstance(changed_reader, _Reader)
    assert clean_reader.calls[0] == changed_reader.calls[0]
    clean_composer = clean_runner.model.components.composer
    changed_composer = changed_runner.model.components.composer
    assert isinstance(clean_composer, _Composer)
    assert isinstance(changed_composer, _Composer)
    assert torch.equal(clean_composer.input_ids[0], changed_composer.input_ids[0])
    assert not torch.equal(
        clean.query_objectives[1].outer.total.detach(),
        changed.query_objectives[1].outer.total.detach(),
    )


@pytest.mark.parametrize("denied", ["answer", "count", "occurrence_times"])
def test_support_label_poison_is_rejected_before_any_model_call(
    config: ProjectConfig,
    denied: str,
) -> None:
    runner, fast, _, resetter = _system(config, MetaTTTVariant.A3)
    del runner
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    clean = _chunk(owner, chunk_index=0, end_time=1.0, width=2)
    poisoned = dict(clean.runtime_payload)
    poisoned[denied] = "forbidden"
    with pytest.raises(ValueError, match="denied fields"):
        replace(clean, runtime_payload=poisoned)
    assert resetter.calls == 0
    assert fast.last_audit is None


def test_stage_a_query_builder_stays_available_as_production_adapter() -> None:
    assert isinstance(StageAQueryLossBuilder(), StageAQueryLossBuilder)


def test_synthetic_ablation_report_has_paired_ci_failures_and_disclaimer() -> None:
    records = tuple(
        SyntheticAblationRecord(
            case_id=f"case-{case}",
            task_name="count",
            metric_name="answer/exact_match",
            variant=variant,
            value=float(case) + offset,
            failure_cases=("synthetic-hard-case",) if case == 1 else (),
        )
        for case in range(3)
        for variant, offset in (
            (MetaTTTVariant.A3, 0.0),
            (MetaTTTVariant.A4, 0.1),
            (MetaTTTVariant.A5, 0.15),
        )
    )
    comparisons = compare_synthetic_ablations(records)
    assert [(item.baseline, item.candidate) for item in comparisons] == [
        (MetaTTTVariant.A3, MetaTTTVariant.A4),
        (MetaTTTVariant.A4, MetaTTTVariant.A5),
    ]
    assert all(item.sample_count == 3 for item in comparisons)
    assert all("synthetic-hard-case" in item.failure_cases for item in comparisons)
    report = render_synthetic_ablation_report(comparisons)
    assert "Synthetic/tiny engineering evidence only" in report
    assert "no scientific gain" in report
    assert "95% CI" in report
