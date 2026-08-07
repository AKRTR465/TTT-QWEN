from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.support import make_test_model as build_model
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.identity_bank import build_identity_bank
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BankWriteOutput,
    BatchRuntimeState,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    PrefillLifecycle,
    QwenGenerateOutput,
    QwenGenerateRequest,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    TrajectoryRuntimeState,
    VisualStageOutput,
)
from ttt_svcbench_qwen.query_encoder import Operator
from ttt_svcbench_qwen.state_bank import (
    RetrievalHistoryView,
    StateBankRuntimeState,
    TensorizedRetrievalHistory,
)


class _StateBankComponent:
    pass


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config()


@dataclass
class SpySuite:
    events: list[str]
    reader_results: tuple[object, ...]
    retrieval: object
    resampler_output: object
    composed: object
    prefill_output: object
    composer_request: dict[str, object] | None = None
    prefill_request: QwenPrefillRequest | None = None

    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        self.events.append("visual")
        assert request.video_input == "video-input"
        return VisualStageOutput("main-visual", "visual-audit")

    def query(self, query_input: object, *, inference: bool) -> object:
        self.events.append("query")
        assert query_input == "query-input"
        assert inference is True
        return SimpleNamespace(q_target="q-target", hard_operators=(Operator.O1_SNAP,))

    def fast(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        self.events.append("fast")
        assert visual.value == "main-visual"
        assert query.q_target == "q-target"
        assert isinstance(request.runtime_state, BatchRuntimeState)
        assert request.runtime_state.owner == request.owner
        return VisualStageOutput("adapted-main", "fast-audit")

    def spatial(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> object:
        self.events.append("spatial")
        assert visual.value == "adapted-main"
        return "spatial-soft"

    def temporal(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> object:
        self.events.append("temporal")
        assert visual.value == "adapted-main"
        return "temporal-soft"

    def heads(
        self,
        spatial: object,
        temporal: object,
        query: object,
        request: ObservationChunkRequest,
    ) -> object:
        self.events.append("heads")
        assert (spatial, temporal) == ("spatial-soft", "temporal-soft")
        return "observation-soft"

    def write_bank(
        self,
        observations: object,
        spatial: object,
        temporal: object,
        query: object,
        request: ObservationChunkRequest,
    ) -> BankWriteOutput:
        self.events.append("bank")
        assert observations == "observation-soft"
        assert (spatial, temporal) == ("spatial-soft", "temporal-soft")
        return BankWriteOutput("runtime-1", ("bank-1",), "bank-audit")

    def __call__(
        self,
        state_bank: object,
        history: RetrievalHistoryView,
        query: object,
        *,
        video_ids: Any,
        trajectory_ids: Any,
    ) -> object:
        self.events.append("retriever.history")
        assert isinstance(state_bank, _StateBankComponent)
        assert history.bank_versions == (0,)
        assert tuple(video_ids) == ("video-a",)
        assert tuple(trajectory_ids) == ("trajectory-a",)
        return self.retrieval

    def read_bank(
        self,
        state_bank: object,
        states: Any,
        query: object,
        *,
        video_ids: Any,
        trajectory_ids: Any,
    ) -> tuple[object, ...]:
        self.events.append("reader.bank")
        assert isinstance(state_bank, _StateBankComponent)
        assert tuple(states) == ("bank-1",)
        assert tuple(video_ids) == ("video-a",)
        assert tuple(trajectory_ids) == ("trajectory-a",)
        return self.reader_results

    def resample(self, q_target: object, retrieval: object) -> object:
        self.events.append("resampler")
        assert q_target == "q-target"
        assert retrieval is self.retrieval
        return self.resampler_output

    def compose(self, **kwargs: object) -> object:
        self.events.append("composer")
        self.composer_request = kwargs
        return self.composed

    def prefill(self, request: QwenPrefillRequest) -> object:
        self.events.append("qwen.prefill")
        self.prefill_request = request
        return self.prefill_output

    def generate(self, request: QwenGenerateRequest) -> QwenGenerateOutput:
        self.events.append("qwen.generate")
        return QwenGenerateOutput("answer", torch.tensor([[1]], dtype=torch.int64))


def make_suite() -> SpySuite:
    retrieval = SimpleNamespace(
        selected_record_ids=(("record-1",),),
        status=("ok",),
        audit=("retrieval-audit",),
    )
    reader_result = SimpleNamespace(
        selected_record_ids=("record-1",),
        exact_count=2,
        number_token_ids=(17,),
    )
    resampler = SimpleNamespace(
        state_tokens="state-tokens",
        state_token_valid_mask="state-valid",
        selected_record_ids=(("record-1",),),
        retrieval_status=("ok",),
    )
    composed = SimpleNamespace(
        input_ids="composed-ids",
        inputs_embeds="audit-only-embeds",
        attention_mask="composed-mask",
        state_position_mask="state-mask",
        position_ids="composer-position-audit",
        rope_deltas="composer-rope-audit",
    )
    return SpySuite(
        events=[],
        reader_results=(reader_result,),
        retrieval=retrieval,
        resampler_output=resampler,
        composed=composed,
        prefill_output=SimpleNamespace(logits="answer-logits", past_key_values="cache"),
    )


def make_components(suite: SpySuite, **updates: object) -> ModelComponents:
    values: dict[str, object] = {
        "visual_stage": suite.visual,
        "query_encoder": suite.query,
        "composer": suite.compose,
        "qwen_prefill": suite.prefill,
        "qwen_generate": suite.generate,
        "fast_adapter": suite.fast,
        "spatial_encoder": suite.spatial,
        "temporal_encoder": suite.temporal,
        "observation_heads": suite.heads,
        "state_bank": _StateBankComponent(),
        "bank_writer": suite.write_bank,
        "retriever": suite,
        "reader": suite,
        "resampler": suite.resample,
    }
    values.update(updates)
    return ModelComponents(**values)  # type: ignore[arg-type]


def make_owner(name: str = "a") -> RuntimeOwner:
    return RuntimeOwner((f"video-{name}",), (f"trajectory-{name}",))


def make_observation_request(owner: RuntimeOwner) -> ObservationChunkRequest:
    video_id = owner.video_ids[0]
    trajectory_id = owner.trajectory_ids[0]
    runtime = BatchRuntimeState(
        (
            TrajectoryRuntimeState(
                owner=owner,
                next_chunk_index=0,
                slot_state=None,
                temporal_cache=None,
                e1_state=None,
                e2_state=None,
                state_bank=StateBankRuntimeState(video_id, trajectory_id, (), ()),
                identity_bank=build_identity_bank(load_config()).reset(video_id, trajectory_id),
                retrieval_history=TensorizedRetrievalHistory(
                    video_id,
                    trajectory_id,
                    capacity_per_head=8,
                    source_dim=768,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                ),
            ),
        )
    )
    return ObservationChunkRequest(
        owner=owner,
        video_input="video-input",
        query_input="query-input",
        runtime_state=runtime,
        bank_states=("bank-0",),
    )


def make_answer_request(owner: RuntimeOwner, observation: object) -> AnswerQueryRequest:
    return AnswerQueryRequest(
        owner=owner,
        observation=observation,  # type: ignore[arg-type]
        base_input_ids="base-ids",
        base_attention_mask="base-mask",
        pixel_values_videos=torch.ones((8, 4)),
        video_grid_thw=torch.tensor([[2, 2, 2]], dtype=torch.int64),
        tokenizer="tokenizer",
        embedding_owner="embedding-owner",
        rope_indexer="rope-indexer",
        qwen_kwargs=(("use_cache", True),),
    )


def run_observation(
    model: StateTTTModel,
    owner: RuntimeOwner,
    lifecycle: PrefillLifecycle,
) -> object:
    return model.observe_chunk(make_observation_request(owner), lifecycle)


def run_answer(
    model: StateTTTModel,
    request: AnswerQueryRequest,
    lifecycle: PrefillLifecycle,
) -> object:
    return model.prefill_answer(model.prepare_answer(request, lifecycle), lifecycle)


def test_model_registers_each_injected_module_identity_once_and_never_runtime(
    config: ProjectConfig,
) -> None:
    shared_qwen = nn.Linear(2, 2)
    components = ModelComponents(
        visual_stage=shared_qwen,  # type: ignore[arg-type]
        query_encoder=shared_qwen,  # type: ignore[arg-type]
        composer=shared_qwen,  # type: ignore[arg-type]
        qwen_prefill=shared_qwen,  # type: ignore[arg-type]
        qwen_generate=shared_qwen,  # type: ignore[arg-type]
    )
    flags = ModelFeatureFlags(
        fast_enabled=False,
        bank_enabled=False,
        reader_enabled=False,
        state_tokens_enabled=False,
    )

    model = build_model(config, components=components, feature_flags=flags)
    lifecycle = PrefillLifecycle(make_owner())

    assert isinstance(model, nn.Module)
    assert tuple(model.component_modules) == ("visual_stage",)
    assert set(model.state_dict()) == {
        "component_modules.visual_stage.weight",
        "component_modules.visual_stage.bias",
    }
    assert len(tuple(model.parameters())) == 2
    assert all("runtime" not in key and "lifecycle" not in key for key in model.state_dict())
    assert lifecycle not in tuple(model.modules())


def test_observe_chunk_runs_the_visual_and_state_pipeline(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)

    output = run_observation(model, owner, lifecycle)

    assert suite.events == ["query", "visual", "fast", "spatial", "temporal", "heads", "bank"]
    assert output.visual.value == "adapted-main"
    assert output.runtime_state == "runtime-1"
    assert output.bank_states == ("bank-1",)
    assert output.soft_intermediates.adapted_visual == "adapted-main"
    assert output.soft_intermediates.spatial == "spatial-soft"
    assert output.soft_intermediates.temporal == "temporal-soft"
    assert output.soft_intermediates.observations == "observation-soft"


def test_answer_query_retrieves_resamples_composes_and_prefills(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    observation = run_observation(model, owner, lifecycle)
    suite.events.clear()

    output = run_answer(model, make_answer_request(owner, observation), lifecycle)

    assert suite.events == [
        "retriever.history",
        "reader.bank",
        "resampler",
        "composer",
        "qwen.prefill",
    ]
    assert output.answer_logits == "answer-logits"
    assert output.reader == suite.reader_results
    assert output.retrieval is suite.retrieval
    assert output.resampler is suite.resampler_output
    assert output.runtime_state == "runtime-1"

    composer_request = suite.composer_request
    assert composer_request is not None
    assert composer_request["reader_results"] == suite.reader_results
    assert composer_request["state_tokens"] == "state-tokens"
    assert composer_request["state_token_valid_mask"] == "state-valid"
    assert composer_request["include_state"] is True
    assert composer_request["include_number"] is True

    prefill = suite.prefill_request
    assert prefill is not None
    assert prefill.input_ids == "composed-ids"
    assert prefill.attention_mask == "composed-mask"
    assert torch.equal(prefill.pixel_values_videos, torch.ones((8, 4)))
    assert torch.equal(prefill.video_grid_thw, torch.tensor([[2, 2, 2]]))
    assert not hasattr(prefill, "prepared_video_features")
    assert prefill.state_position_mask == "state-mask"
    assert prefill.state_tokens == "state-tokens"
    assert not hasattr(prefill, "inputs_embeds")


def test_disabled_features_are_not_called_and_are_reported_as_absent(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    flags = ModelFeatureFlags(
        fast_enabled=False,
        bank_enabled=False,
        reader_enabled=False,
        state_tokens_enabled=False,
    )
    components = make_components(
        suite,
        fast_adapter=None,
        spatial_encoder=None,
        temporal_encoder=None,
        observation_heads=None,
        state_bank=None,
        bank_writer=None,
        retriever=None,
        reader=None,
        resampler=None,
    )
    model = build_model(config, components=components, feature_flags=flags)
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)

    observation = run_observation(model, owner, lifecycle)
    answer = run_answer(model, make_answer_request(owner, observation), lifecycle)

    assert suite.events == ["query", "visual", "composer", "qwen.prefill"]
    assert observation.visual.value == "main-visual"
    assert observation.spatial is None
    assert observation.observations is None
    assert answer.retrieval is None
    assert answer.reader == ()
    assert answer.resampler is None
    assert suite.composer_request is not None
    assert suite.composer_request["reader_results"] == ()
    assert suite.composer_request["state_tokens"] is None
    assert suite.composer_request["include_state"] is False
    assert suite.composer_request["include_number"] is False

