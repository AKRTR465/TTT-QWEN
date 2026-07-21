from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import MetaTTTVariant, ProjectConfig, load_config
from ttt_svcbench_qwen.data import RuntimeQueryInput, assert_runtime_payload_safe
from ttt_svcbench_qwen.fast_ttt import FastTTTForwardAudit, FastWeightsState
from ttt_svcbench_qwen.identity_bank import (
    IdentityDecisionStatus,
    IdentityObservationDecision,
    build_identity_bank,
)
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    O1StateTarget,
    StateLossInput,
    TemporalPredictor,
)
from ttt_svcbench_qwen.meta_trainer import (
    MetaCausalChunk,
    MetaQueryLossBuilder,
    MetaQueryLossInput,
    MetaTTTEpisode,
    MetaTTTEpisodeRunner,
    MetaTTTQueryPoint,
    StageAQueryLossBuilder,
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
    OfficialWeakLossAudit,
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
    StageATargetBatch,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_bank import HeadType, build_state_bank
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


class _RuntimeResetter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, owner: RuntimeOwner) -> BatchRuntimeState:
        self.calls += 1
        state_bank = build_state_bank(load_config())
        identity_bank = build_identity_bank(load_config())
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
                    identity_bank=identity_bank.reset(
                        video_id,
                        trajectory_id,
                        hot_cache_enabled=False,
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
        return SimpleNamespace(q_target=torch.zeros((1, 512)))


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
            bank_versions_before=tuple(bank.version for bank in runtime.bank_states),
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
        versions = tuple(state.version for state in states)
        return SimpleNamespace(audit=versions, versions=versions)

    def retrieve_query_history(
        self,
        state_bank: object,
        states: Sequence[object],
        query: object,
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> object:
        return self.retrieve_query(
            state_bank,
            states,
            query,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
        )


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

    def audit_results(
        self,
        _retrieval: object,
        results: Sequence[object],
    ) -> Sequence[object]:
        return results

    def audit_bank_results(
        self,
        _state_bank: object,
        _states: Sequence[object],
        _query: object,
        results: Sequence[object],
        *,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ) -> Sequence[object]:
        if len(video_ids) != len(trajectory_ids):
            raise ValueError("tiny Reader ownership mismatch")
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
        self.answer_features: list[Tensor] = []

    def forward(self, request: QwenPrefillRequest) -> object:
        if not isinstance(request.input_ids, Tensor):
            raise TypeError("tiny Qwen inputs must be tensors")
        self.answer_features.append(request.pixel_values_videos.detach().clone())
        score = request.pixel_values_videos.float().mean().reshape(1)
        zeros = torch.zeros_like(score)
        row = torch.stack((score, -score, zeros), dim=-1)
        logits = row[:, None, :].expand(-1, request.input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


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


class _TinyOfficialWeakQueryLossBuilder:
    streamed_balance_calibration = True

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
    variant: MetaTTTVariant,
    *,
    query_loss_builder: MetaQueryLossBuilder | None = None,
    query_encoder_reuse: bool = False,
    raw_support_visual_batcher: object | None = None,
    support_visual_batch_size: int = 1,
) -> tuple[MetaTTTEpisodeRunner, _TinyFastController, _TinyPredictor, _RuntimeResetter]:
    fast = _TinyFastController()
    predictor = _TinyPredictor()
    resetter = _RuntimeResetter()
    reader = _Reader()
    qwen = _Qwen()
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
        predictor=predictor,
        runtime_resetter=resetter,
        variant=variant,
        query_loss_builder=query_loss_builder or _TinyQueryLossBuilder(),
        query_encoder_reuse=query_encoder_reuse,
        raw_support_visual_batcher=raw_support_visual_batcher,  # type: ignore[arg-type]
        support_visual_batch_size=support_visual_batch_size,
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


def _truncated_episode(
    config: ProjectConfig,
    *,
    support_count: int,
    query_count: int = 2,
) -> MetaTTTEpisode:
    base = _episode(
        config,
        MetaTTTVariant.A5,
        support_count=support_count,
        query_count=query_count,
    )
    prewarm = _chunk(base.owner, chunk_index=0, end_time=0.5, width=2)
    supports = tuple(
        _chunk(
            base.owner,
            chunk_index=index + 1,
            end_time=float(index) + 1.5,
            width=2,
        )
        for index in range(support_count)
    )
    queries = tuple(
        replace(
            query,
            chunk=_chunk(
                base.owner,
                chunk_index=support_count + index + 1,
                end_time=float(support_count + index) + 2.0,
                width=2,
            ),
            query_time=float(support_count + index) + 2.0,
        )
        for index, query in enumerate(base.query_points)
    )
    return replace(base, prewarm_chunk=prewarm, support_chunks=supports, query_points=queries)


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
    batch_size = len(owner.video_ids)
    width = reference.shape[1]
    hidden = reference.detach().clone()
    empty_kv = tuple(
        torch.zeros((batch_size, 12, width, 64), dtype=reference.dtype) for _ in range(6)
    )
    replay_kv = tuple(torch.zeros((batch_size, 12, 0, 64), dtype=reference.dtype) for _ in range(6))
    return TemporalCache(
        hidden=hidden,
        layer_keys=empty_kv,
        layer_values=tuple(value.clone() for value in empty_kv),
        replay_layer_keys=replay_kv,
        replay_layer_values=tuple(value.clone() for value in replay_kv),
        timestamps=timestamps.clone(),
        replay_timestamps=torch.zeros((batch_size, 0), dtype=torch.float64),
        position_ids=position_ids.clone(),
        replay_position_ids=torch.zeros((batch_size, 0), dtype=torch.int64),
        valid_mask=valid_mask.clone(),
        replay_valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool),
        video_ids=owner.video_ids,
        trajectory_ids=owner.trajectory_ids,
        query_signatures=torch.zeros((batch_size, 512), dtype=reference.dtype),
        total_seen=position_ids[:, -1].clone() + 1,
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


def test_truncated_a5_t17_k8_has_two_numeric_truncations_and_bounded_graphs(
    config: ProjectConfig,
) -> None:
    runner, fast, predictor, resetter = _system(config, MetaTTTVariant.A5)
    episode = _truncated_episode(config, support_count=17)
    output = runner.run_truncated(episode)

    assert output.audit.support_count == 17
    assert output.audit.truncation_horizon == 8
    assert output.audit.truncation_count == 2
    assert output.audit.segment_count == 3
    assert output.audit.backward_count == 4
    assert output.audit.query_backward_count == 2
    assert output.audit.maximum_retained_support_graphs == 8
    assert [segment.support_count for segment in output.audit.segments] == [8, 8, 1]
    assert [segment.includes_query_backward for segment in output.audit.segments] == [
        False,
        False,
        True,
    ]
    assert all(segment.reanchored for segment in output.audit.segments)
    assert output.final_fast_states[0].fast_version == 17
    assert output.final_fast_states[0].update_count == 17
    assert output.final_fast_states[0].w_t_1.grad_fn is not None
    assert output.final_fast_states[0].w_t_2.grad_fn is not None
    assert output.total.grad_fn is None and not output.total.requires_grad
    assert fast.w0_1.grad is not None and float(fast.w0_1.grad.norm()) > 0.0
    assert fast.w0_2.grad is not None and float(fast.w0_2.grad.norm()) > 0.0
    assert predictor.scale.grad is not None and float(predictor.scale.grad.abs()) > 0.0
    assert resetter.calls == 1


def test_truncated_a5_batches_raw_visuals_only_within_each_k_segment(
    config: ProjectConfig,
) -> None:
    calls: list[tuple[int, int]] = []

    def raw_batcher(
        chunks: tuple[MetaCausalChunk, ...],
        batch_size: int,
    ) -> tuple[MetaCausalChunk, ...]:
        calls.append((len(chunks), batch_size))
        return chunks

    runner, _, _, _ = _system(
        config,
        MetaTTTVariant.A5,
        raw_support_visual_batcher=raw_batcher,
        support_visual_batch_size=2,
    )

    output = runner.run_truncated(_truncated_episode(config, support_count=17))

    assert calls == [(8, 2), (8, 2), (1, 2)]
    assert [segment.support_count for segment in output.audit.segments] == [8, 8, 1]
    assert all(update.next_only_verified for update in output.audit.updates)
    assert output.final_fast_states[0].fast_version == 17


def test_truncated_a5_exact_k_waits_for_query_before_reanchor(config: ProjectConfig) -> None:
    runner, fast, _, _ = _system(config, MetaTTTVariant.A5)

    output = runner.run_truncated(_truncated_episode(config, support_count=8))

    assert output.audit.segment_count == 1
    assert output.audit.backward_count == 2
    assert output.audit.query_backward_count == 2
    assert output.audit.truncation_count == 1
    assert output.audit.segments[0].support_count == 8
    assert output.audit.segments[0].includes_query_backward
    assert output.audit.segments[0].reanchored
    assert fast.w0_1.grad is not None and float(fast.w0_1.grad.norm()) > 0.0
    assert fast.w0_2.grad is not None and float(fast.w0_2.grad.norm()) > 0.0


def test_sequential_multi_query_backward_matches_one_shot_mean() -> None:
    initial = torch.tensor([0.75, -0.25], dtype=torch.float64)

    one_shot = initial.clone().requires_grad_(True)
    shared_one_shot = torch.sin(one_shot.square())
    cases = ((1.0, 0.2), (2.0, -0.3), (-0.5, 0.7))
    losses = tuple((shared_one_shot * scale - target).square().sum() for scale, target in cases)
    torch.stack(losses).mean().backward()
    expected = one_shot.grad.detach().clone()

    streamed = initial.clone().requires_grad_(True)
    shared_streamed = torch.sin(streamed.square())
    query_count = 3
    for index, (scale, target) in enumerate(cases):
        loss = (shared_streamed * scale - target).square().sum() / float(query_count)
        loss.backward(retain_graph=index + 1 < query_count)

    assert streamed.grad is not None
    assert torch.allclose(streamed.grad, expected, atol=1.0e-12, rtol=1.0e-12)


def test_truncated_a5_instant_equal_composes_all_queries_once(
    config: ProjectConfig,
) -> None:
    raw = config.model_dump(mode="json")
    raw["loss"]["official_weak_balance"]["mode"] = "instant_equal"
    raw["loss"]["official_weak_balance"]["experimental"] = True
    raw["loss"]["official_weak_balance"]["scale_min"] = 0.1
    raw["loss"]["official_weak_balance"]["scale_max"] = 10.0
    instant_config = ProjectConfig.model_validate(raw)
    runner, _, _, _ = _system(
        instant_config,
        MetaTTTVariant.A5,
        query_loss_builder=_TinyOfficialWeakQueryLossBuilder(),
    )

    output = runner.run_truncated(
        _truncated_episode(instant_config, support_count=1, query_count=2)
    )

    assert runner.last_balance_audit is not None
    assert runner.last_balance_audit.auxiliary_to_answer_ratio <= 0.3
    assert len(output.audit.queries) == 2
    assert all(
        query.metrics.value("loss/aux_to_answer_ratio") is not None
        for query in output.audit.queries
    )


def test_truncated_a5_reuses_one_query_graph_per_segment_and_final_key(
    config: ProjectConfig,
) -> None:
    serial, serial_fast, _, _ = _system(config, MetaTTTVariant.A5)
    reused, reused_fast, _, _ = _system(
        config,
        MetaTTTVariant.A5,
        query_encoder_reuse=True,
    )
    episode = _with_shared_query_key(_truncated_episode(config, support_count=17))

    serial_output = serial.run_truncated(episode)
    reused_output = reused.run_truncated(episode)

    serial_query = serial.model.components.query_encoder
    reused_query = reused.model.components.query_encoder
    assert isinstance(serial_query, _QueryStage)
    assert isinstance(reused_query, _QueryStage)
    assert serial_query.calls == 20  # prewarm + 17 Supports + 2 final Queries
    assert reused_query.calls == 5  # prewarm + three K segments + one final key
    assert torch.equal(serial_output.total, reused_output.total)
    assert torch.equal(serial_output.query_loss, reused_output.query_loss)
    assert torch.equal(serial_fast.w0_1.grad, reused_fast.w0_1.grad)
    assert torch.equal(serial_fast.w0_2.grad, reused_fast.w0_2.grad)


def test_truncated_a5_does_not_reuse_different_final_query_ids(
    config: ProjectConfig,
) -> None:
    runner, _, _, _ = _system(
        config,
        MetaTTTVariant.A5,
        query_encoder_reuse=True,
    )
    episode = _truncated_episode(config, support_count=1, query_count=2)
    runner.run_truncated(episode)

    query = runner.model.components.query_encoder
    assert isinstance(query, _QueryStage)
    assert query.calls == 4  # prewarm + one segment + two distinct final keys


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
    runner, fast, _, resetter = _system(config, MetaTTTVariant.A5)
    del runner
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    clean = _chunk(owner, chunk_index=0, end_time=1.0, width=2)
    poisoned = dict(clean.query_input.as_payload())
    poisoned[denied] = "forbidden"
    with pytest.raises(ValueError, match="denied fields"):
        assert_runtime_payload_safe(poisoned, layer="Meta-TTT Support/Query")
    assert resetter.calls == 0
    assert fast.last_audit is None


def test_stage_a_query_builder_stays_available_as_production_adapter() -> None:
    assert isinstance(StageAQueryLossBuilder(), StageAQueryLossBuilder)
