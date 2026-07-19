from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BankWriteOutput,
    LifecycleError,
    LifecyclePhase,
    ModelComponents,
    ModelFeatureFlags,
    ObservationChunkRequest,
    PrefillLifecycle,
    QwenGenerateOutput,
    QwenGenerateRequest,
    QwenPrefillRequest,
    RuntimeOwner,
    StateTTTModel,
    VisualStageOutput,
    assert_training_number_agreement,
    build_model,
    evaluate_number_agreement,
)


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
    audit_replacement: tuple[object, ...] | None = None

    def visual(self, request: ObservationChunkRequest) -> VisualStageOutput:
        self.events.append("visual")
        assert request.video_input == "video-input"
        return VisualStageOutput("main-visual", "prepared-main-deepstack", "visual-audit")

    def query(self, query_input: object, *, inference: bool) -> object:
        self.events.append("query")
        assert query_input == "query-input"
        assert inference is True
        return SimpleNamespace(q_target="q-target")

    def fast(
        self,
        visual: VisualStageOutput,
        query: object,
        request: ObservationChunkRequest,
    ) -> VisualStageOutput:
        self.events.append("fast")
        assert visual.value == "main-visual"
        assert query.q_target == "q-target"
        assert request.runtime_state == "runtime-0"
        return VisualStageOutput("adapted-main", "prepared-main-deepstack", "fast-audit")

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

    def retrieve_query(
        self,
        state_bank: object,
        states: Any,
        query: object,
        *,
        video_ids: Any,
        trajectory_ids: Any,
    ) -> object:
        self.events.append("retriever")
        assert state_bank == "state-bank-component"
        assert tuple(states) == ("bank-1",)
        assert tuple(video_ids) == ("video-a",)
        assert tuple(trajectory_ids) == ("trajectory-a",)
        return self.retrieval

    def read(self, retrieval: object) -> tuple[object, ...]:
        self.events.append("reader.read")
        assert retrieval is self.retrieval
        return self.reader_results

    def audit_results(
        self,
        retrieval: object,
        results: Any,
    ) -> tuple[object, ...]:
        self.events.append("reader.audit")
        assert retrieval is self.retrieval
        assert tuple(results) == self.reader_results
        return self.reader_results if self.audit_replacement is None else self.audit_replacement

    def audit_number_tokens(self, result: object) -> int:
        self.events.append("reader.number")
        assert result is self.reader_results[0]
        return 2

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
        "state_bank": "state-bank-component",
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
    return ObservationChunkRequest(
        owner=owner,
        video_input="video-input",
        query_input="query-input",
        runtime_state="runtime-0",
        bank_states=("bank-0",),
    )


def make_answer_request(owner: RuntimeOwner, observation: object) -> AnswerQueryRequest:
    return AnswerQueryRequest(
        owner=owner,
        observation=observation,  # type: ignore[arg-type]
        base_input_ids="base-ids",
        base_attention_mask="base-mask",
        pixel_values_videos="pixels",
        video_grid_thw="grid",
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


def test_build_model_validates_feature_dependencies_before_any_stage(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    with pytest.raises(ValueError, match="validated ProjectConfig"):
        build_model(components=make_components(suite))
    with pytest.raises(ValueError, match="explicit ModelComponents"):
        build_model(config)
    with pytest.raises(ValueError, match="fast_adapter"):
        build_model(config, components=make_components(suite, fast_adapter=None))
    with pytest.raises(ValueError, match="reader"):
        build_model(config, components=make_components(suite, reader=None))
    with pytest.raises(ValueError, match="resampler"):
        build_model(config, components=make_components(suite, resampler=None))
    with pytest.raises(ValueError, match="Reader requires"):
        ModelFeatureFlags(bank_enabled=False, reader_enabled=True, state_tokens_enabled=False)
    assert suite.events == []


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


def test_observe_chunk_is_composition_only_and_returns_soft_intermediates(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)

    output = run_observation(model, owner, lifecycle)

    assert suite.events == ["visual", "query", "fast", "spatial", "temporal", "heads", "bank"]
    assert output.visual.value == "adapted-main"
    assert output.runtime_state == "runtime-1"
    assert output.bank_states == ("bank-1",)
    assert output.state_audit == "bank-audit"
    assert output.soft_intermediates.adapted_visual == "adapted-main"
    assert output.soft_intermediates.spatial == "spatial-soft"
    assert output.soft_intermediates.temporal == "temporal-soft"
    assert output.soft_intermediates.observations == "observation-soft"
    assert output.lifecycle.phase is LifecyclePhase.READY
    assert output.lifecycle.observation_count == 1


def test_soft_observation_recompute_boundary_cannot_duplicate_hard_commit(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    request = make_observation_request(owner)

    soft = model.observe_chunk_soft(request)

    assert suite.events == ["visual", "query", "fast", "spatial", "temporal", "heads"]
    assert "bank" not in suite.events
    assert lifecycle.audit().observation_count == 0
    output = model.commit_observation(request, soft, lifecycle)
    assert output.runtime_state == "runtime-1"
    assert suite.events.count("bank") == 1
    assert soft.commit_guard.committed

    with pytest.raises(LifecycleError, match="already committed"):
        model.commit_observation(request, soft, lifecycle)
    assert suite.events.count("bank") == 1


def test_answer_query_audits_same_retrieval_before_resampler_and_native_prefill(
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
        "retriever",
        "reader.read",
        "reader.audit",
        "reader.number",
        "resampler",
        "composer",
        "qwen.prefill",
    ]
    assert output.answer_logits == "answer-logits"
    assert output.reader == suite.reader_results
    assert output.retrieval is suite.retrieval
    assert output.resampler is suite.resampler_output
    assert output.runtime_state == "runtime-1"
    assert output.lifecycle.phase is LifecyclePhase.PREFILLED
    assert output.lifecycle.prefill_count == 1

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
    assert prefill.pixel_values_videos == "pixels"
    assert prefill.video_grid_thw == "grid"
    assert prefill.prepared_video_features == "prepared-main-deepstack"
    assert prefill.state_position_mask == "state-mask"
    assert prefill.state_tokens == "state-tokens"
    assert prefill.composer_position_ids_audit == "composer-position-audit"
    assert prefill.composer_rope_deltas_audit == "composer-rope-audit"
    assert not hasattr(prefill, "inputs_embeds")


def test_prefill_is_one_shot_and_observe_is_forbidden_after_it(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    observation = run_observation(model, owner, lifecycle)
    run_answer(model, make_answer_request(owner, observation), lifecycle)
    event_count = len(suite.events)

    with pytest.raises(LifecycleError, match="exactly once"):
        run_answer(model, make_answer_request(owner, observation), lifecycle)
    with pytest.raises(LifecycleError, match="forbidden after prefill"):
        model.observe_chunk(make_observation_request(owner), lifecycle)
    assert len(suite.events) == event_count


def test_cross_owner_observation_fails_closed(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    other = make_owner("b")
    lifecycle = PrefillLifecycle(owner)

    with pytest.raises(LifecycleError, match="owner"):
        model.observe_chunk(make_observation_request(other), lifecycle)
    assert suite.events == []


def test_reader_rewrite_blocks_resampler_composer_and_marks_lifecycle_failed(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    replacement = SimpleNamespace(
        selected_record_ids=("record-1",),
        exact_count=999,
        number_token_ids=(999,),
    )
    suite.audit_replacement = (replacement,)
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    observation = run_observation(model, owner, lifecycle)
    suite.events.clear()

    with pytest.raises(ValueError, match="unchanged authoritative"):
        model.prepare_answer(make_answer_request(owner, observation), lifecycle)

    assert suite.events == ["retriever", "reader.read", "reader.audit"]
    assert lifecycle.audit().phase is LifecyclePhase.READY


def test_resampler_provenance_mismatch_blocks_composer_and_prefill(
    config: ProjectConfig,
) -> None:
    suite = make_suite()
    suite.resampler_output = SimpleNamespace(
        state_tokens="state-tokens",
        state_token_valid_mask="state-valid",
        selected_record_ids=(("different-record",),),
        retrieval_status=("ok",),
    )
    model = build_model(config, components=make_components(suite))
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    observation = run_observation(model, owner, lifecycle)
    suite.events.clear()

    with pytest.raises(ValueError, match="same Retriever"):
        model.prepare_answer(make_answer_request(owner, observation), lifecycle)

    assert suite.events[-1] == "resampler"
    assert "composer" not in suite.events
    assert "qwen.prefill" not in suite.events
    assert lifecycle.audit().phase is LifecyclePhase.READY


def test_prefill_failure_is_terminal_and_cannot_repeat_state_reads(
    config: ProjectConfig,
) -> None:
    suite = make_suite()

    def failing_prefill(request: QwenPrefillRequest) -> object:
        suite.events.append("qwen.prefill")
        raise RuntimeError("synthetic prefill failure")

    model = build_model(
        config,
        components=make_components(suite, qwen_prefill=failing_prefill),
    )
    owner = make_owner()
    lifecycle = PrefillLifecycle(owner)
    observation = run_observation(model, owner, lifecycle)
    suite.events.clear()

    with pytest.raises(RuntimeError, match="synthetic prefill failure"):
        run_answer(model, make_answer_request(owner, observation), lifecycle)
    event_count = len(suite.events)
    assert lifecycle.audit().phase is LifecyclePhase.FAILED
    with pytest.raises(LifecycleError, match="reset"):
        run_answer(model, make_answer_request(owner, observation), lifecycle)
    assert len(suite.events) == event_count


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

    assert suite.events == ["visual", "query", "composer", "qwen.prefill"]
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


def test_qwen_kwargs_cannot_override_composer_or_native_visual_fields() -> None:
    owner = make_owner()
    suite = make_suite()
    observation = SimpleNamespace(owner=owner)
    base = dict(
        owner=owner,
        observation=observation,
        base_input_ids="ids",
        base_attention_mask="mask",
        pixel_values_videos="pixels",
        video_grid_thw="grid",
        tokenizer="tokenizer",
        embedding_owner="embedding",
        rope_indexer="rope",
    )
    with pytest.raises(ValueError, match="P13-owned"):
        AnswerQueryRequest(**base, qwen_kwargs=(("inputs_embeds", "forbidden"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="P13-owned"):
        AnswerQueryRequest(
            **base,
            qwen_kwargs=(("prepared_video_features", "forbidden"),),
        )  # type: ignore[arg-type]
    assert suite.events == []


def test_reader_number_agreement_is_independent_and_training_mismatch_is_blocked() -> None:
    results = (
        SimpleNamespace(exact_count=2),
        SimpleNamespace(exact_count=0),
        SimpleNamespace(exact_count=None),
        SimpleNamespace(exact_count=-3),
    )

    metrics = evaluate_number_agreement(results, (2, 7, 999, None))

    assert metrics.comparable_rows == 3
    assert metrics.matched_rows == 1
    assert metrics.mismatched_rows == 1
    assert metrics.missing_rows == 1
    assert metrics.accuracy == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="authoritative Reader"):
        assert_training_number_agreement(results, (2, 7, None, -3))
    assert_training_number_agreement(results, (2, 0, None, -3))
