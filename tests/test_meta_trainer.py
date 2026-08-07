from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import ttt_svcbench_qwen.meta_trainer as meta_trainer_module
from tests.support.runtime_factories import (
    make_e1_state,
    make_e2_state,
    make_temporal_cache,
)
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput, assert_runtime_payload_safe
from ttt_svcbench_qwen.fast_ttt import (
    AssociativeTTTIntermediates,
    FastAssociativeContext,
    FastMemoryState,
    FastTTTForwardAudit,
    MemoryWriteBatch,
)
from ttt_svcbench_qwen.identity_bank import build_identity_bank
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    O1StateTarget,
    StateLossInput,
)
from ttt_svcbench_qwen.meta_trainer import (
    MetaCausalChunk,
    MetaQueryLossBuilder,
    MetaQueryLossInput,
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    MetaTTTQueryPoint,
    _reanchor_query_signature,
)
from ttt_svcbench_qwen.model import (
    BankWriteOutput,
    BatchRuntimeState,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
    TrajectoryRuntimeState,
    VisualStageOutput,
    query_dropout_seed,
)
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
)
from ttt_svcbench_qwen.query_encoder import Operator
from ttt_svcbench_qwen.stage_a_runtime import StageASoftWriteOutput
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    OfficialWeakLossAudit,
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
    StageATargetBatch,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    RetrievalHistoryView,
    TensorizedRetrievalHistory,
    build_state_bank,
)
from ttt_svcbench_qwen.state_encoder import TemporalCache, TemporalEncoderOutput
from ttt_svcbench_qwen.trainer import StageAEpisodeAnswerInputs, StageASupervisionBatch


@dataclass(frozen=True, slots=True)
class _VideoChunk:
    features: Tensor
    timestamps: Tensor
    position_ids: Tensor
    valid_mask: Tensor
    associative_valid_mask: Tensor | None = None
    identity_timestamp: float = 0.0
    identity_position_id: int = 0


class _RuntimeResetter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, owner: RuntimeOwner) -> BatchRuntimeState:
        self.calls += 1
        state_bank = build_state_bank(load_config())
        identity_bank = build_identity_bank(load_config())
        parameter = next(state_bank.semantic_projector.parameters())
        return BatchRuntimeState(
            tuple(
                TrajectoryRuntimeState(
                    owner=RuntimeOwner((video_id,), (trajectory_id,)),
                    next_chunk_index=0,
                    slot_state=None,
                    temporal_cache=None,
                    e1_state=None,
                    e2_state=None,
                    state_bank=state_bank.reset(video_id, trajectory_id),
                    identity_bank=identity_bank.reset(video_id, trajectory_id),
                    retrieval_history=TensorizedRetrievalHistory(
                        video_id,
                        trajectory_id,
                        capacity_per_head=state_bank.config.retrieval_history_capacity_per_head,
                        source_dim=state_bank.config.retrieval_history_source_dim,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ),
                )
                for video_id, trajectory_id in zip(
                    owner.video_ids,
                    owner.trajectory_ids,
                    strict=True,
                )
            )
        )


class _TinyFastController(nn.Module):
    """Minimal FastStateController analog with a real per-video memory matrix."""

    def __init__(self) -> None:
        super().__init__()
        first = torch.zeros((768, 768), dtype=torch.float32)
        second = torch.zeros_like(first)
        first[0, 0] = 0.2
        second[0, 0] = 0.3
        self.w0_1 = nn.Parameter(first)
        self.w0_2 = nn.Parameter(second)
        self._active: tuple[FastMemoryState, ...] | None = None
        self._context: FastAssociativeContext | None = None
        self._intermediates: AssociativeTTTIntermediates | None = None
        self.value_scale = nn.Parameter(torch.tensor(0.5))
        self.context_scale = nn.Parameter(torch.tensor(0.01))
        self.eta_raw = nn.Parameter(torch.tensor(0.0))
        self.last_audit: FastTTTForwardAudit | None = None

    def reset_fast_state(
        self,
        state: FastMemoryState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastMemoryState:
        del state
        return FastMemoryState(
            m=torch.zeros((768, 768), dtype=torch.float32, requires_grad=True),
            write_version=0,
            write_count=0,
            skip_count=0,
            differentiable=bool(differentiable),
        )

    @contextmanager
    def use_fast_state(
        self,
        state: FastMemoryState | Sequence[FastMemoryState],
    ) -> Iterator[object]:
        if self._active is not None:
            raise RuntimeError("tiny fast binding is not re-entrant")
        states = (state,) if isinstance(state, FastMemoryState) else tuple(state)
        if not states[0].differentiable and any(
            parameter.grad is not None for parameter in self.parameters()
        ):
            raise ValueError("clear stale module gradients before binding online Fast TTT state")
        self._active = states
        try:
            yield self
        finally:
            self._active = None

    def collect_meta_fast_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.w0_1, self.w0_2)

    def collect_associative_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.value_scale, self.context_scale, self.eta_raw)

    def prepare_write(
        self,
        intermediates: AssociativeTTTIntermediates,
        spatial: object,
    ) -> MemoryWriteBatch:
        del spatial
        keys_tokens = intermediates.keys.float()
        valid = intermediates.valid_mask
        counts = valid.sum(dim=1)
        denominator = counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
        pooled = (keys_tokens * valid.unsqueeze(-1).to(torch.float32)).sum(dim=1) / denominator
        unit = pooled * torch.rsqrt(pooled.square().sum(dim=-1, keepdim=True) + 1.0e-12)
        slot_mask = (counts > 0).unsqueeze(-1)
        slot_scale = slot_mask.to(torch.float32)
        keys = (unit * slot_scale).unsqueeze(1)
        etas = torch.sigmoid(self.eta_raw).reshape(1, 1).expand(keys_tokens.shape[0], 1)
        etas = etas * slot_scale
        return MemoryWriteBatch(
            keys=keys,
            values=keys.clone(),
            etas=etas,
            slot_mask=slot_mask,
            beta=torch.tensor(0.01, dtype=torch.float32),
            eta_renormalized=(False,) * keys_tokens.shape[0],
        )

    @contextmanager
    def use_associative_context(
        self,
        context: FastAssociativeContext,
    ) -> Iterator[object]:
        if self._context is not None:
            raise RuntimeError("tiny associative context is not re-entrant")
        self._context = context
        try:
            yield self
        finally:
            self._context = None

    def consume_associative_intermediates(self) -> AssociativeTTTIntermediates | None:
        value = self._intermediates
        self._intermediates = None
        return value

    def forward(
        self,
        visual: VisualStageOutput,
        _query: object,
        _request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        if self._active is None or not isinstance(visual.value, Tensor):
            raise RuntimeError("tiny fast stage requires one managed binding")
        static_gain = self.w0_1[0, 0] * self.w0_2[0, 0]
        gains = torch.stack([static_gain + state.m[0, 0] for state in self._active])
        adapted = visual.value + gains[:, None, None]
        context = self._context
        if context is None:
            raise RuntimeError("tiny Fast forward requires an associative context")
        context_term = (
            context.combined_query.mean(dim=-1, keepdim=True).unsqueeze(1)
            * self.context_scale
        )
        keys = visual.value * self.value_scale + context_term
        predictions = keys + gains[:, None, None]
        payload = _request.video_input
        if not isinstance(payload, _VideoChunk):
            raise TypeError("tiny Fast forward requires a _VideoChunk payload")
        associative_mask = payload.associative_valid_mask
        valid_mask = (
            payload.valid_mask if associative_mask is None else associative_mask
        ).to(device=keys.device)
        self._intermediates = AssociativeTTTIntermediates(
            keys=keys,
            predictions=predictions,
            valid_mask=valid_mask,
            bank_record_counts=context.bank_record_counts,
            bank_versions=context.bank_versions,
        )
        self.last_audit = FastTTTForwardAudit(
            write_versions=tuple(state.write_version for state in self._active),
            write_counts=tuple(state.write_count for state in self._active),
            valid_token_counts=tuple(adapted.shape[1] for _ in self._active),
            used_runtime_state=True,
            bank_record_counts=tuple(int(value.item()) for value in context.bank_record_counts),
            readout_share_norms=tuple(
                float(state.m.detach()[0, 0].abs()) for state in self._active
            ),
        )
        return replace(visual, value=adapted)


class _VisualStage(nn.Module):
    def forward(self, request: ObservationChunkRequest) -> VisualStageOutput:
        if not isinstance(request.video_input, _VideoChunk):
            raise TypeError("tiny visual stage requires _VideoChunk")
        return VisualStageOutput(
            value=request.video_input.features,
        )


class _QueryStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, query_input: object, *, inference: bool) -> object:
        if inference or not isinstance(query_input, RuntimeQueryInput):
            raise ValueError("tiny query stage requires a training RuntimeQueryInput")
        self.calls += 1
        return SimpleNamespace(
            q_target=torch.zeros((1, 512)),
            hard_operators=(Operator.O1_SNAP,),
            head_types=(HeadType.O1,),
        )


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
            cache=_cache(
                request.owner,
                visual.value,
                payload.timestamps,
                payload.position_ids,
                payload.valid_mask,
            ),
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
                    _e1_state(request.owner, row, temporal.position_ids)
                    for row in range(batch_size)
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
                    _e2_state(request.owner, row, temporal.position_ids)
                    for row in range(batch_size)
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
        if not isinstance(request.runtime_state, BatchRuntimeState):
            raise TypeError("tiny writer requires BatchRuntimeState")
        runtime = request.runtime_state
        if not isinstance(_temporal, TemporalEncoderOutput) or not isinstance(
            _observations, ObservationOutputs
        ):
            raise TypeError("tiny writer requires typed temporal/observation outputs")
        next_banks = tuple(replace(bank, version=bank.version + 1) for bank in runtime.bank_states)
        next_runtime = BatchRuntimeState(
            tuple(
                replace(
                    row,
                    next_chunk_index=runtime.next_chunk_index + 1,
                    temporal_cache=_temporal.cache,
                    e1_state=_observations.e1.next_states[index],
                    e2_state=_observations.e2.next_states[index],
                    state_bank=next_banks[index],
                )
                for index, row in enumerate(runtime.rows)
            )
        )
        valid = _temporal.valid_mask
        counts = valid.sum(dim=1).clamp_min(1).to(_temporal.hidden.dtype)
        source = (
            (_temporal.hidden * valid.unsqueeze(-1).to(_temporal.hidden.dtype)).sum(dim=1)
            / counts.unsqueeze(-1)
        )
        present = valid.any(dim=1)
        o2_present = _observations.o2.valid_mask
        soft_write = StageASoftWriteOutput(
            o1_semantics=torch.zeros(
                source.shape[0], 512, dtype=source.dtype, device=source.device
            ),
            o1_present_mask=present,
            o2_semantics=torch.zeros(
                *o2_present.shape,
                512,
                dtype=source.dtype,
                device=source.device,
            ),
            o2_present_mask=o2_present,
            e1_semantics=torch.zeros(
                *valid.shape,
                512,
                dtype=source.dtype,
                device=source.device,
            ),
            e1_present_mask=valid,
            e2_semantics=torch.zeros(
                *valid.shape,
                512,
                dtype=source.dtype,
                device=source.device,
            ),
            e2_present_mask=valid,
            o1_sources=source,
            o2_sources=source.unsqueeze(1).expand(-1, o2_present.shape[1], -1),
            e1_sources=source,
            e2_sources=source,
        )
        return BankWriteOutput(next_runtime, next_banks, None, soft_write=soft_write)


class _Retriever:
    def __call__(
        self,
        _state_bank: object,
        history: RetrievalHistoryView,
        _query: object,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> object:
        versions = history.bank_versions
        if len(versions) != len(video_ids) or len(versions) != len(trajectory_ids):
            raise ValueError("tiny retrieval ownership mismatch")
        return SimpleNamespace(audit=versions, versions=versions)


@dataclass(frozen=True, slots=True)
class _ReaderResult:
    exact_count: int


class _Reader:
    def __init__(self) -> None:
        self.calls: list[tuple[_ReaderResult, ...]] = []

    def read(self, retrieval: object) -> Sequence[object]:
        versions = getattr(retrieval, "versions", None)
        if not isinstance(versions, tuple):
            raise TypeError("tiny Reader requires typed retrieval versions")
        results = tuple(_ReaderResult(int(value)) for value in versions)
        self.calls.append(results)
        return results

    def read_bank(
        self,
        _state_bank: object,
        states: Sequence[object],
        _query: object,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> Sequence[object]:
        if len(states) != len(video_ids) or len(states) != len(trajectory_ids):
            raise ValueError("tiny Reader ownership mismatch")
        results = tuple(_ReaderResult(int(state.version)) for state in states)
        self.calls.append(results)
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
    def __init__(self, fast_controller: _TinyFastController) -> None:
        super().__init__()
        object.__setattr__(self, "_fast_controller", fast_controller)
        self.answer_features: list[Tensor] = []
        self.answer_write_versions: list[tuple[int, ...]] = []

    def forward(self, request: QwenPrefillRequest) -> object:
        if not isinstance(request.input_ids, Tensor):
            raise TypeError("tiny Qwen inputs must be tensors")
        fast_controller = self._fast_controller
        active = fast_controller._active
        if active is None:
            raise RuntimeError("tiny Answer prefill requires one managed fast-state binding")
        self.answer_features.append(request.pixel_values_videos.detach().clone())
        self.answer_write_versions.append(tuple(state.write_version for state in active))
        gains = torch.stack(tuple(state.m[0, 0] for state in active))
        score = request.pixel_values_videos.float().mean().reshape(1) + gains - gains.detach()
        zeros = torch.zeros_like(score)
        row = torch.stack((score, -score, zeros), dim=-1)
        logits = row[:, None, :].expand(-1, request.input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class _TinyQueryLossBuilder:
    def __init__(self, *, answer_connected: bool = True, state_connected: bool = True) -> None:
        self.answer_connected = answer_connected
        self.state_connected = state_connected

    def __call__(
        self,
        output: StateTTTModelOutput,
        *,
        answer: StageAEpisodeAnswerInputs,
        supervision: StageASupervisionBatch,
        dedup: object | None = None,
    ) -> MetaQueryLossInput:
        del answer, dedup
        if not isinstance(output.answer_logits, Tensor) or not isinstance(
            output.observations, ObservationOutputs
        ):
            raise TypeError("tiny Query output has the wrong type")
        labels = supervision.answer.base_labels.to(output.answer_logits.device)
        if labels.shape != output.answer_logits.shape[:2]:
            raise ValueError("tiny Query labels must align to Qwen logits")
        number_mask = torch.zeros_like(labels, dtype=torch.bool)
        o1 = output.observations.o1
        answer_logits = (
            output.answer_logits if self.answer_connected else output.answer_logits.detach()
        )
        state_mask = o1.valid_mask if self.state_connected else torch.zeros_like(o1.valid_mask)
        return MetaQueryLossInput(
            answer=AnswerLossInput(answer_logits, labels, number_mask),
            state=StateLossInput(
                batch_size=output.answer_logits.shape[0],
                o1=O1StateTarget(
                    row_indices=torch.arange(output.answer_logits.shape[0]),
                    logits=o1.logits,
                    targets=torch.zeros_like(o1.logits),
                    slot_mask=state_mask,
                ),
            ),
        )


class _TinyOfficialWeakQueryLossBuilder:
    streamed_balance_calibration = True

    def __call__(
        self,
        output: StateTTTModelOutput,
        *,
        answer: StageAEpisodeAnswerInputs,
        supervision: StageASupervisionBatch,
        dedup: object | None = None,
    ) -> MetaQueryLossInput:
        del answer, dedup
        if not isinstance(output.answer_logits, Tensor) or not isinstance(
            output.observations, ObservationOutputs
        ):
            raise TypeError("tiny official-weak Query output has the wrong type")
        labels = supervision.answer.base_labels.to(output.answer_logits.device)
        number_mask = torch.zeros_like(labels, dtype=torch.bool)
        anchor = output.observations.o1.logits.float().square().mean() + 1.0
        terms = tuple(
            OfficialWeakLossTerm(value=anchor * factor, valid_rows=1)
            for factor in (1.0, 2.0, 3.0, 4.0)
        )
        state = OfficialWeakStateLossOutput(
            task=terms[0],
            operator=terms[1],
            retrieval=terms[2],
            time=terms[3],
            total=torch.stack(tuple(term.value for term in terms)).sum(),
            audit=OfficialWeakLossAudit(
                labels_joined_after_forward=True,
                runtime_payload_reused_for_labels=False,
                identity_target_fabricated=False,
                unique_retrieval_id_fabricated=False,
                future_occurrences_ignored=0,
                retrieval_bag_sizes=(1,),
            ),
        )
        return MetaQueryLossInput(
            answer=AnswerLossInput(output.answer_logits, labels, number_mask),
            state=state,
        )


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config()


def _system(
    config: ProjectConfig,
    *,
    query_loss_builder: MetaQueryLossBuilder | None = None,
    query_encoder_reuse: bool = False,
) -> tuple[MetaTTTEpisodeRunner, _TinyFastController, _TinyFastController, _RuntimeResetter]:
    fast = _TinyFastController()
    resetter = _RuntimeResetter()
    reader = _Reader()
    qwen = _Qwen(fast)
    model = StateTTTModel(
        config,
        ModelComponents(
            visual_stage=_VisualStage(),
            query_encoder=_QueryStage(),
            fast_adapter=fast,
            spatial_encoder=_SpatialStage(),
            temporal_encoder=_TemporalStage(),
            observation_heads=_ObservationStage(),
            state_bank=build_state_bank(config),
            bank_writer=_BankWriter(),
            retriever=_Retriever(),
            reader=reader,
            composer=_Composer(),
            qwen_prefill=qwen,
            qwen_generate=qwen,  # type: ignore[arg-type]
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
        runtime_resetter=resetter,
        query_loss_builder=query_loss_builder or _TinyQueryLossBuilder(),
        query_encoder_reuse=query_encoder_reuse,
    )
    return runner, fast, fast, resetter


def _truncated_episode(
    config: ProjectConfig,
    *,
    support_count: int,
    invalid_first_support: bool = False,
    diagnostic_query_count: int = 0,
) -> MetaTTTEpisode:
    if not 1 <= support_count <= 16:
        raise ValueError("support-aligned test episode supports must be in [1, 16]")
    segment_lengths = (support_count,) if support_count <= 8 else (8, support_count - 8)
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    prewarm = _chunk(owner, chunk_index=0, end_time=0.5, width=2)
    supports: list[MetaCausalChunk] = []
    queries: list[MetaTTTQueryPoint] = []
    current_time = 0.5
    chunk_index = 1
    for segment_index, segment_length in enumerate(segment_lengths):
        for _ in range(segment_length):
            current_time += 1.0
            supports.append(
                _chunk(
                    owner,
                    chunk_index=chunk_index,
                    end_time=current_time,
                    width=2,
                    valid=not (not supports and invalid_first_support),
                )
            )
            chunk_index += 1
        current_time += 0.5
        query_chunk = _chunk(
            owner,
            chunk_index=chunk_index,
            end_time=current_time,
            width=2,
        )
        queries.append(
            MetaTTTQueryPoint(
                chunk=query_chunk,
                query_time=current_time,
                answer=_answer_inputs(),
                supervision=_supervision(),
                task_name="synthetic-count",
                case_id=f"case-{segment_index}",
            )
        )
        query_input = query_chunk.query_input
        segment_start = len(supports) - segment_length
        supports[segment_start:] = [
            replace(
                chunk,
                request=replace(chunk.request, query_input=query_input),
                query_input=query_input,
            )
            for chunk in supports[segment_start:]
        ]
        chunk_index += 1
    return MetaTTTEpisode(
        owner=owner,
        support_chunks=tuple(supports),
        query_points=tuple(queries),
        seed=config.a5.seed,
        prewarm_chunk=prewarm,
        segment_lengths=segment_lengths,
        query_roles=(("final",) if len(segment_lengths) == 1 else ("intermediate", "final")),
        query_weights=(1.0,) * len(segment_lengths),
        diagnostic_query_count=diagnostic_query_count,
    )


def _with_shared_query_key(episode: MetaTTTEpisode) -> MetaTTTEpisode:
    if episode.prewarm_chunk is None:
        raise ValueError("shared Query helper requires a truncated episode")
    reference = episode.support_chunks[0].query_input
    shared = replace(
        reference,
        query_id="shared-query",
        question="shared question",
        episode_nonce=17,
    )

    def bind(chunk: MetaCausalChunk) -> MetaCausalChunk:
        return replace(
            chunk,
            request=replace(chunk.request, query_input=shared),
            query_input=shared,
        )

    return replace(
        episode,
        prewarm_chunk=bind(episode.prewarm_chunk),
        support_chunks=tuple(bind(chunk) for chunk in episode.support_chunks),
        query_points=tuple(
            replace(query, chunk=bind(query.chunk)) for query in episode.query_points
        ),
    )


def _chunk(
    owner: RuntimeOwner,
    *,
    chunk_index: int,
    end_time: float,
    width: int,
    valid: bool = True,
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
        associative_valid_mask=torch.full((1, width), valid, dtype=torch.bool),
    )
    query_input = RuntimeQueryInput(
        video_id=owner.video_ids[0],
        trajectory_id=owner.trajectory_ids[0],
        query_id=f"query-{chunk_index}",
        query_index=chunk_index,
        video=Path("synthetic-video.mp4"),
        question="how many",
        query_time=end_time,
        explicit_time_values=(),
    )
    return MetaCausalChunk(
        request=ObservationChunkRequest(
            owner=owner,
            video_input=payload,
            query_input=query_input,
            runtime_state=object(),
            bank_states=(object(),),
            inference=False,
        ),
        start_time=max(0.0, end_time - 1.0),
        end_time=end_time,
        query_input=query_input,
    )


def _answer_inputs() -> StageAEpisodeAnswerInputs:
    return StageAEpisodeAnswerInputs(
        base_input_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        base_attention_mask=torch.ones((1, 3), dtype=torch.int64),
        pixel_values_videos=torch.ones((8, 4)),
        video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
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


def _cache(
    owner: RuntimeOwner,
    reference: Tensor,
    timestamps: Tensor,
    position_ids: Tensor,
    valid_mask: Tensor,
) -> TemporalCache:
    return make_temporal_cache(
        hidden=reference,
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
        timestamps=timestamps,
        position_ids=position_ids,
        valid_mask=valid_mask,
        total_seen=position_ids[:, -1].clone() + 1,
    )


def _e1_state(owner: RuntimeOwner, row: int, positions: Tensor) -> E1RuntimeState:
    return make_e1_state(
        video_id=owner.video_ids[row],
        trajectory_id=owner.trajectory_ids[row],
        total_seen=int(positions[row, -1].item()) + 1,
    )


def _e2_state(owner: RuntimeOwner, row: int, positions: Tensor) -> E2RuntimeState:
    return make_e2_state(
        video_id=owner.video_ids[row],
        trajectory_id=owner.trajectory_ids[row],
        total_seen=int(positions[row, -1].item()) + 1,
    )


def _counting_backward() -> tuple[list[float], object]:
    """Collect one entry per backward call; the schedule replaces the deleted audit."""

    values: list[float] = []

    def backward(loss: Tensor, retain_graph: bool) -> None:
        assert not retain_graph
        values.append(float(loss.detach()))
        loss.backward()

    return values, backward


def test_meta_episode_supports_stay_strictly_inside_the_causal_prefix(
    config: ProjectConfig,
) -> None:
    """Cross-segment time monotonicity: no Support may reach or pass its own Query."""

    episode = _truncated_episode(config, support_count=16)
    offset = 0
    previous_query_time = float(episode.prewarm_chunk.end_time)  # type: ignore[union-attr]
    for segment_length, query in zip(
        episode.segment_lengths,
        episode.query_points,
        strict=True,
    ):
        segment = episode.support_chunks[offset : offset + segment_length]
        assert [chunk.end_time for chunk in segment] == sorted(
            chunk.end_time for chunk in segment
        )
        assert segment[0].start_time >= previous_query_time - 1.0
        assert min(chunk.end_time for chunk in segment) > previous_query_time
        assert segment[-1].end_time < query.query_time
        assert query.chunk.end_time <= query.query_time
        previous_query_time = query.query_time
        offset += segment_length


def test_stage_c_invalid_chunk_skips_then_later_supports_continue(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(config)
    output = runner.run_truncated(
        _truncated_episode(config, support_count=4, invalid_first_support=True)
    )
    assert output.audit.write_count == 3
    assert output.audit.skip_count == 1
    assert output.final_fast_states[0].write_count == 3
    assert output.final_fast_states[0].skip_count == 1
    assert output.final_fast_states[0].write_version == 3
    assert output.audit.query_count == 1


def test_a5_all_skipped_segment_still_closes_with_its_query(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(config)
    backward_values, backward = _counting_backward()

    output = runner.run_truncated(
        _truncated_episode(config, support_count=1, invalid_first_support=True),
        backward=backward,  # type: ignore[arg-type]
    )

    assert output.audit.segment_count == 1
    assert output.audit.write_count == 0
    assert output.audit.skip_count == 1
    assert output.audit.query_count == 1
    assert output.audit.loss_weight == 1.0
    # One Query backward plus the segment's single deferred VJP closure.
    assert len(backward_values) == 2
    assert output.final_fast_states[0].write_version == 0


@pytest.mark.parametrize("support_count", [1, 4, 8])
def test_a5_support_schedule_is_bounded_and_next_only(
    config: ProjectConfig,
    support_count: int,
) -> None:
    runner, _, _, _ = _system(config)
    output = runner.run_truncated(_truncated_episode(config, support_count=support_count))
    assert output.audit.segment_count == 1
    assert output.audit.write_count == support_count
    assert output.audit.skip_count == 0
    assert output.audit.associative_valid_count == support_count
    assert output.final_fast_states[0].write_count == support_count
    assert output.final_fast_states[0].write_version == support_count


def test_truncated_a5_two_k8_segments_each_close_with_a_query(
    config: ProjectConfig,
) -> None:
    runner, fast, associative, resetter = _system(config)
    episode = _truncated_episode(config, support_count=16)
    backward_values, backward = _counting_backward()
    output = runner.run_truncated(episode, backward=backward)  # type: ignore[arg-type]

    assert output.audit.segment_count == 2
    assert output.audit.query_count == 2
    assert output.audit.write_count == 16
    assert output.audit.skip_count == 0
    # Two Query backwards plus one deferred VJP closure per K=8 segment.
    assert len(backward_values) == 4
    assert output.final_fast_states[0].write_version == 16
    assert output.final_fast_states[0].write_count == 16
    assert output.query_loss.item() == pytest.approx(output.total.item())
    # The final memory was truncated at the last segment boundary: a leaf with no
    # retained Support graph, values preserved bitwise.
    assert output.final_fast_states[0].m.is_leaf
    assert output.final_fast_states[0].m.grad_fn is None
    assert output.total.grad_fn is None and not output.total.requires_grad
    assert fast.w0_1.grad is not None and float(fast.w0_1.grad.norm()) > 0.0
    assert fast.w0_2.grad is not None and float(fast.w0_2.grad.norm()) > 0.0
    assert associative.value_scale.grad is not None
    assert float(associative.value_scale.grad.abs()) > 0.0
    assert associative.eta_raw.grad is not None
    assert float(associative.eta_raw.grad.abs()) > 0.0
    assert resetter.calls == 1


def test_truncated_a5_query_bundle_sums_proxy_gradients_then_closes_once(
    config: ProjectConfig,
) -> None:
    runner, fast, _, _ = _system(config)
    base = _truncated_episode(config, support_count=8)
    first = base.query_points[0]
    second_chunk = _chunk(
        base.owner,
        chunk_index=99,
        end_time=first.query_time + 0.5,
        width=2,
    )
    second = replace(
        first,
        chunk=second_chunk,
        query_time=first.query_time + 0.5,
        case_id="case-bundle-final",
    )
    episode = replace(
        base,
        query_points=(first, second),
        segment_query_counts=(2,),
        query_roles=("intermediate", "final"),
        query_weights=(1.0, 1.0),
    )
    backward_values, backward = _counting_backward()

    output = runner.run_truncated(episode, backward=backward)  # type: ignore[arg-type]

    assert output.audit.segment_count == 1
    assert output.audit.query_count == 2
    assert output.audit.write_count == 8
    assert output.audit.skip_count == 0
    # Two bundled Query backwards, then exactly one deferred VJP closure.
    assert len(backward_values) == 3
    assert fast.w0_1.grad is not None and float(fast.w0_1.grad.norm()) > 0.0


def test_query_cotangents_clip_independently_then_sum_without_averaging() -> None:
    ordinary = (
        torch.tensor([0.3], dtype=torch.float32),
        torch.tensor([0.4], dtype=torch.float32),
    )
    small = (
        torch.tensor([0.03], dtype=torch.float32),
        torch.tensor([0.04], dtype=torch.float32),
    )
    extreme = (
        torch.tensor([6.0], dtype=torch.float32),
        torch.tensor([8.0], dtype=torch.float32),
    )
    originals = tuple(
        tuple(gradient.clone() for gradient in query)
        for query in (ordinary, small, extreme)
    )

    clipped_queries = []
    for query in (ordinary, small, extreme):
        clipped = meta_trainer_module._clip_query_proxy_gradients(
            query,
            max_norm=1.0,
            epsilon=1.0e-12,
        )
        clipped_queries.append(clipped)

    raw_norms = [
        meta_trainer_module._cotangent_norm_float(query)
        for query in (ordinary, small, extreme)
    ]
    clipped_norms = [
        meta_trainer_module._cotangent_norm_float(query) for query in clipped_queries
    ]
    assert raw_norms == pytest.approx([0.5, 0.05, 10.0])
    assert clipped_norms == pytest.approx([0.5, 0.05, 1.0])

    segment_sum = tuple(
        sum(query[matrix_index] for query in clipped_queries)
        for matrix_index in range(2)
    )
    torch.testing.assert_close(segment_sum[0], torch.tensor([0.93]))
    torch.testing.assert_close(segment_sum[1], torch.tensor([1.24]))
    # Per-Query clipping is not a segment-level clip: the sum may exceed max_norm.
    assert meta_trainer_module._cotangent_norm_float(segment_sum) > 1.0

    for query, original in zip((ordinary, small, extreme), originals, strict=True):
        for gradient, expected in zip(query, original, strict=True):
            torch.testing.assert_close(gradient, expected)


def test_query_cotangent_clip_norm_is_config_driven(config: ProjectConfig) -> None:
    gradients = (torch.tensor([30.0, 40.0], dtype=torch.float32),)

    clipped = meta_trainer_module._clip_query_proxy_gradients(
        gradients,
        max_norm=10.0,
        epsilon=1.0e-12,
    )

    assert meta_trainer_module._cotangent_norm_float(gradients) == pytest.approx(50.0)
    assert meta_trainer_module._cotangent_norm_float(clipped) == pytest.approx(10.0)
    torch.testing.assert_close(clipped[0], torch.tensor([6.0, 8.0]))
    # The production clip norm is the configured one, not a hard-coded constant.
    configured = meta_trainer_module._clip_query_proxy_gradients(
        gradients,
        max_norm=config.a5.query_meta_gradient.max_norm,
        epsilon=config.a5.query_meta_gradient.epsilon,
    )
    assert meta_trainer_module._cotangent_norm_float(configured) == pytest.approx(
        min(50.0, config.a5.query_meta_gradient.max_norm)
    )


def test_query_cotangent_clipping_does_not_rescale_direct_outer_gradient() -> None:
    direct_outer = nn.Parameter(torch.tensor(2.0))
    query_proxies = (
        nn.Parameter(torch.tensor([6.0, 8.0])),
        nn.Parameter(torch.tensor([60.0, 80.0])),
    )
    clipped_queries = []
    for direct_coefficient, proxy in zip((3.0, 5.0), query_proxies, strict=True):
        (direct_outer * direct_coefficient + proxy.square().sum()).backward()
        assert proxy.grad is not None
        clipped = meta_trainer_module._clip_query_proxy_gradients(
            (proxy.grad.detach().float().clone(),),
            max_norm=1.0,
            epsilon=1.0e-12,
        )
        clipped_queries.append(clipped[0])
        proxy.grad = None

    assert direct_outer.grad is not None
    torch.testing.assert_close(direct_outer.grad, torch.tensor(8.0))
    assert all(float(torch.linalg.vector_norm(value)) <= 1.0 + 1.0e-6 for value in clipped_queries)


def test_query_proxy_gradient_equals_answer_plus_state_components(
    config: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_capture = meta_trainer_module._capture_query_proxy_gradients
    captured: list[tuple[Tensor, ...]] = []

    def capture(
        states: Sequence[FastMemoryState],
        *,
        backward_gradient_scale: float,
    ) -> tuple[Tensor, ...]:
        gradients = original_capture(
            states,
            backward_gradient_scale=backward_gradient_scale,
        )
        captured.append(tuple(gradient.detach().clone() for gradient in gradients))
        return gradients

    monkeypatch.setattr(meta_trainer_module, "_capture_query_proxy_gradients", capture)

    def run(builder: _TinyQueryLossBuilder) -> tuple[Tensor, ...]:
        captured.clear()
        torch.manual_seed(101)
        runner, _, _, _ = _system(config, query_loss_builder=builder)
        runner.run_truncated(_truncated_episode(config, support_count=4))
        assert len(captured) == 1
        return tuple(gradient.clone() for gradient in captured[0])

    answer = run(_TinyQueryLossBuilder(state_connected=False))
    state = run(_TinyQueryLossBuilder(answer_connected=False))
    total = run(_TinyQueryLossBuilder())

    assert all(float(torch.linalg.vector_norm(gradient).item()) > 0.0 for gradient in answer)
    assert all(float(torch.linalg.vector_norm(gradient).item()) > 0.0 for gradient in state)
    for combined, answer_gradient, state_gradient in zip(total, answer, state, strict=True):
        torch.testing.assert_close(
            combined,
            answer_gradient + state_gradient,
            rtol=1.0e-5,
            atol=1.0e-7,
        )


def test_truncated_a5_exact_k_waits_for_query_before_truncation(config: ProjectConfig) -> None:
    runner, fast, _, _ = _system(config)
    backward_values, backward = _counting_backward()

    output = runner.run_truncated(
        _truncated_episode(config, support_count=8),
        backward=backward,  # type: ignore[arg-type]
    )

    assert output.audit.segment_count == 1
    assert output.audit.write_count == 8
    # A K=8 segment closes with exactly one Query backward and one deferred VJP.
    assert len(backward_values) == 2
    assert output.final_fast_states[0].write_version == 8
    assert fast.w0_1.grad is not None and float(fast.w0_1.grad.norm()) > 0.0
    assert fast.w0_2.grad is not None and float(fast.w0_2.grad.norm()) > 0.0


def test_truncated_queries_and_deferred_vjp_never_retain_local_graph(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(config)
    retain_flags: list[bool] = []
    graph_bearing: list[bool] = []

    def backward(loss: Tensor, retain_graph: bool) -> None:
        retain_flags.append(retain_graph)
        graph_bearing.append(loss.grad_fn is not None)
        loss.backward(retain_graph=retain_graph)

    output = runner.run_truncated(
        _truncated_episode(config, support_count=9, diagnostic_query_count=4),
        backward=backward,
    )

    # Two segments, each one Query backward plus one deferred VJP, none retained.
    assert retain_flags == [False] * 4
    # Every backward really consumed a local graph; nothing was a detached no-op.
    assert graph_bearing == [True] * 4
    # The truncation point is a fresh leaf, so no Support graph survived the episode.
    assert output.final_fast_states[0].m.is_leaf
    assert output.final_fast_states[0].m.grad_fn is None
    assert output.final_fast_states[0].m.requires_grad


def test_zero_weight_padding_keeps_backward_schedule_but_contributes_zero(
    config: ProjectConfig,
) -> None:
    runner, fast, associative, _ = _system(config)
    backward_values, backward = _counting_backward()

    output = runner.run_truncated(
        _truncated_episode(config, support_count=9, diagnostic_query_count=3),
        backward=backward,  # type: ignore[arg-type]
        episode_loss_weight=0.0,
    )

    assert output.audit.loss_weight == 0.0
    # The 4-rank sampler pads with zero-weight episodes: the backward schedule must
    # stay identical to a weighted rank so the collective never desynchronizes.
    assert len(backward_values) == 4
    assert backward_values == pytest.approx([0.0] * 4)
    assert output.total == pytest.approx(0.0)
    assert output.query_loss == pytest.approx(0.0)
    assert fast.w0_1.grad is not None and torch.count_nonzero(fast.w0_1.grad) == 0
    assert fast.w0_2.grad is not None and torch.count_nonzero(fast.w0_2.grad) == 0
    assert associative.value_scale.grad is not None
    assert torch.count_nonzero(associative.value_scale.grad) == 0


def test_truncated_a5_ema_balance_composes_all_queries_once(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(
        config,
        query_loss_builder=_TinyOfficialWeakQueryLossBuilder(),
    )

    output = runner.run_truncated(_truncated_episode(config, support_count=9))

    assert runner.last_balance_audit is not None
    assert tuple(term.name for term in runner.last_balance_audit.terms) == (
        "task",
        "operator",
        "retrieval",
        "time",
    )
    assert all(term.global_valid_count > 0 for term in runner.last_balance_audit.terms)
    assert output.audit.query_count == 2
    qwen = runner.model.components.qwen_prefill
    assert isinstance(qwen, _Qwen)
    # One calibration prefill and one graph-bearing prefill per Query, each reading
    # the memory generation produced by its own segment (M_{t-1} recurrence order).
    assert qwen.answer_write_versions == [(8,), (8,), (9,), (9,)]


def test_truncated_a5_reuses_one_query_graph_per_segment_and_final_key(
    config: ProjectConfig,
) -> None:
    serial, serial_fast, _, _ = _system(config)
    reused, reused_fast, _, _ = _system(
        config,
        query_encoder_reuse=True,
    )
    episode = _with_shared_query_key(_truncated_episode(config, support_count=16))

    serial_output = serial.run_truncated(episode)
    reused_output = reused.run_truncated(episode)

    serial_query = serial.model.components.query_encoder
    reused_query = reused.model.components.query_encoder
    assert isinstance(serial_query, _QueryStage)
    assert isinstance(reused_query, _QueryStage)
    assert serial_query.calls == 21  # prewarm + 16 Supports + two calibrated Meta Queries
    # Every segment and Meta Query receives one fresh grad-bearing Query graph.
    assert reused_query.calls == 6
    assert torch.equal(serial_output.total, reused_output.total)
    assert torch.equal(serial_output.query_loss, reused_output.query_loss)
    assert torch.equal(serial_fast.w0_1.grad, reused_fast.w0_1.grad)
    assert torch.equal(serial_fast.w0_2.grad, reused_fast.w0_2.grad)


def test_query_signature_reanchor_preserves_value_and_fresh_gradient() -> None:
    reference = F.normalize(torch.randn(2, 512), dim=-1).to(torch.bfloat16)
    current = reference.detach().clone().requires_grad_(True)
    anchored = _reanchor_query_signature(current, reference)

    assert torch.equal(anchored.detach(), reference)
    anchored.float().sum().backward()
    assert current.grad is not None
    assert torch.equal(current.grad, torch.ones_like(current))


def test_query_dropout_seed_is_stable_for_same_question_within_episode() -> None:
    base = RuntimeQueryInput(
        video_id="video",
        trajectory_id="trajectory",
        query_id="query:0",
        query_index=0,
        video=Path("video.mp4"),
        question="How many objects have appeared so far?",
        query_time=10.0,
        explicit_time_values=(),
        episode_nonce=7,
    )
    later = replace(
        base,
        query_id="query:1",
        query_index=1,
        query_time=20.0,
    )
    different_question = replace(
        later,
        question="How many actions have completed so far?",
    )
    different_episode = replace(later, episode_nonce=8)

    assert query_dropout_seed(base) == query_dropout_seed(later)
    assert query_dropout_seed(base) != query_dropout_seed(different_question)
    assert query_dropout_seed(base) != query_dropout_seed(different_episode)


def test_truncated_a5_does_not_reuse_different_final_query_ids(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(
        config,
        query_encoder_reuse=True,
    )
    episode = _truncated_episode(config, support_count=9)
    runner.run_truncated(episode)

    query = runner.model.components.query_encoder
    assert isinstance(query, _QueryStage)
    assert query.calls == 7  # prewarm + two segments + calibration and Meta Query pairs


def test_later_query_labels_cannot_change_earlier_query_path(config: ProjectConfig) -> None:
    clean_runner, _, _, _ = _system(config)
    changed_runner, _, _, _ = _system(config)
    clean_episode = _truncated_episode(config, support_count=9)
    changed_episode = replace(
        clean_episode,
        query_points=(
            clean_episode.query_points[0],
            replace(clean_episode.query_points[1], supervision=_supervision(label=1)),
        ),
    )
    clean_values, clean_backward = _counting_backward()
    changed_values, changed_backward = _counting_backward()
    clean_runner.run_truncated(clean_episode, backward=clean_backward)  # type: ignore[arg-type]
    changed_runner.run_truncated(changed_episode, backward=changed_backward)  # type: ignore[arg-type]

    # Backward schedule is [query 0, deferred 0, query 1, deferred 1].
    assert len(clean_values) == len(changed_values) == 4
    assert clean_values[0] == pytest.approx(changed_values[0])
    assert clean_values[2] != pytest.approx(changed_values[2])


def test_later_support_cannot_change_earlier_intermediate_query(
    config: ProjectConfig,
) -> None:
    torch.manual_seed(17)
    clean_runner, _, _, _ = _system(config)
    torch.manual_seed(17)
    changed_runner, _, _, _ = _system(config)
    clean_episode = _truncated_episode(config, support_count=9)
    changed_supports = list(clean_episode.support_chunks)
    later = changed_supports[-1]
    payload = later.request.video_input
    assert isinstance(payload, _VideoChunk)
    changed_features = payload.features.clone()
    changed_features[..., 1] += 100.0
    changed_payload = replace(payload, features=changed_features)
    changed_supports[-1] = replace(
        later,
        request=replace(later.request, video_input=changed_payload),
    )
    changed_episode = replace(
        clean_episode,
        support_chunks=tuple(changed_supports),
    )
    clean_values, clean_backward = _counting_backward()
    changed_values, changed_backward = _counting_backward()

    clean_runner.run_truncated(clean_episode, backward=clean_backward)  # type: ignore[arg-type]
    changed_runner.run_truncated(changed_episode, backward=changed_backward)  # type: ignore[arg-type]

    # Chunk t only ever reads M_{t-1}: the second segment's Support cannot move the
    # first segment's Query loss, but must move its own.
    assert clean_values[0] == pytest.approx(changed_values[0])
    assert clean_values[2] != pytest.approx(changed_values[2])


def test_answer_only_proxy_gradient_trains_the_memory_interface(
    config: ProjectConfig,
) -> None:
    runner, _, associative, _ = _system(
        config,
        query_loss_builder=_TinyQueryLossBuilder(state_connected=False),
    )

    output = runner.run_truncated(_truncated_episode(config, support_count=4))

    assert output.audit.write_count == 4
    assert associative.value_scale.grad is not None
    assert float(torch.linalg.vector_norm(associative.value_scale.grad).item()) > 0.0
    assert associative.eta_raw.grad is not None
    assert float(torch.linalg.vector_norm(associative.eta_raw.grad).item()) > 0.0


@pytest.mark.parametrize("denied", ["answer", "count", "occurrence_times"])
def test_support_label_poison_is_rejected_before_any_model_call(
    config: ProjectConfig,
    denied: str,
) -> None:
    runner, fast, _, resetter = _system(config)
    del runner
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    clean = _chunk(owner, chunk_index=0, end_time=1.0, width=2)
    poisoned = dict(clean.query_input.as_payload())
    poisoned[denied] = "forbidden"
    with pytest.raises(ValueError, match="denied fields"):
        assert_runtime_payload_safe(poisoned, layer="Meta-TTT Support/Query")
    assert resetter.calls == 0
    assert fast.last_audit is None
