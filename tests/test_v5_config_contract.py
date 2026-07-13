from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ttt_svcbench_qwen.config import (
    CalibrationStatus,
    ProjectConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model_state_ttt_8b.yaml"


def load_raw_config() -> dict[str, Any]:
    raw: object = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_v5_yaml_passes_strong_validation_and_serializes_completely() -> None:
    config = load_config(CONFIG_PATH)
    serialized = json.loads(config.model_dump_json())

    assert serialized["spec_version"] == (
        "state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval"
    )
    assert serialized["model"]["revision"] == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert set(serialized) == set(ProjectConfig.model_fields)


def test_v5_base_and_deepstack_contract() -> None:
    model = load_config().model

    assert model.base_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert model.transformers_version == "4.57.1"
    assert model.vision.model_dump() == {
        "depth": 27,
        "hidden_size": 1152,
        "num_heads": 16,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "output_size": 4096,
        "deepstack_visual_indexes": (8, 16, 24),
    }
    assert model.llm.num_layers == 36
    assert model.llm.hidden_size == 4096
    assert all(model.online_freeze.model_dump().values())


def test_p2_data_and_checkpoint_video_processor_contract() -> None:
    config = load_config()

    assert config.data.video_directory == "data/videos"
    assert config.data.group_key_fields == ("source_dataset", "video_path")
    assert config.data.group_k_folds == 5
    assert config.data.fold_seed == 42
    assert config.data.runtime_allowlist == (
        "video",
        "question",
        "query_time",
        "explicit_time_values",
    )
    assert config.data.runtime_denylist == (
        "answer",
        "count",
        "occurrence_times",
        "counting_type",
        "counting_subtype",
    )
    video = config.video_preprocessing
    assert (video.frames_per_chunk, video.stride_frames, video.sample_fps) == (16, 8, 2.0)
    assert video.causal_boundary == "right_closed"
    assert (video.processor_shortest_edge, video.processor_longest_edge) == (4096, 25_165_824)
    assert (video.patch_size, video.temporal_patch_size, video.spatial_merge_size) == (16, 2, 2)


def test_v5_fast_and_inner_sgd_contract() -> None:
    fast = load_config().fast_ttt

    assert (fast.input_dim, fast.bottleneck_dim, fast.output_dim) == (4096, 768, 4096)
    assert fast.fast_bias is False
    assert fast.residual_scale == 0.1
    assert fast.fast_matrix_count == 2
    assert fast.online_parameter_count == 2 * 768 * 768 == 1_179_648
    assert fast.optimizer.model_dump() == {
        "name": "sgd",
        "learning_rate": 1.0e-4,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "steps_per_chunk": 1,
        "grad_clip_norm": 1.0,
        "reset_per_video": True,
    }


def test_v5_encoder_head_and_capacity_contracts() -> None:
    config = load_config()

    assert config.spatial_encoder.hidden_dim == 768
    assert config.spatial_encoder.active_slots == 32
    assert config.spatial_encoder.max_active_slots == 64
    assert config.temporal_encoder.num_layers == 6
    assert config.temporal_encoder.cache_tubelets == 64
    assert config.observation_heads.o1.hidden_dims == (1024, 1024)
    assert config.observation_heads.o1.output_dim == 6
    assert config.observation_heads.o2.identity_dim == 256
    assert config.observation_heads.e1.dilations == (1, 2, 4, 8, 16)
    assert config.observation_heads.e2.num_layers == 2
    assert config.state_bank.confirmed_store.initial_capacity == 256
    assert config.state_bank.confirmed_store.growth_chunk == 256
    assert config.state_bank.confirmed_store.hard_limit is None
    assert config.state_bank.confirmed_store.gpu_hot_capacity == 256
    assert config.state_bank.candidate_store.initial_capacity == 64
    assert config.state_bank.candidate_store.hard_limit == 512
    assert config.state_bank.event_history_capacity == 512


def test_v5_query_retrieval_resampler_and_loss_contracts() -> None:
    config = load_config()

    assert config.query_encoder.model_dump(exclude={"pooling", "bidirectional"}) == {
        "input_dim": 4096,
        "hidden_dim": 768,
        "num_layers": 4,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "dropout": 0.1,
        "output_dim": 512,
    }
    assert len(config.operator_router.prototypes) == 9
    assert config.operator_router.prototypes[-1] == "unsupported"
    assert config.retriever.record_similarity_threshold == 0.35
    assert config.retriever.top_k is None
    assert config.retriever.ann_enabled is False
    assert config.state_resampler.num_queries == 16
    assert config.state_resampler.output_dim == 4096
    assert config.predictor.model_dump() == {
        "input_dim": 768,
        "hidden_dim": 1536,
        "output_dim": 768,
    }
    assert config.loss.model_dump() == {
        "pred_weight": 1.0,
        "identity_weight": 0.5,
        "event_weight": 0.5,
        "o1_unlabeled_weight": 0.0,
        "auxiliary_outer_weight": 0.1,
    }


def test_v5_parameter_budget_matches_architecture_rounding() -> None:
    budget = load_config().parameter_budget
    component_total = sum(
        (
            budget.fast_ttt_adapter_millions,
            budget.spatial_encoder_millions,
            budget.temporal_encoder_millions,
            budget.query_encoder_millions,
            budget.o1_millions,
            budget.o2_millions,
            budget.e1_millions,
            budget.e2_millions,
            budget.semantic_projector_millions,
            budget.predictor_millions,
            budget.state_resampler_millions,
            budget.router_resolver_empty_millions,
        )
    )

    assert abs(component_total - 156.83) <= budget.rounding_tolerance_millions
    assert budget.online_fast_matrices_millions == 1.179648


def test_uncalibrated_thresholds_block_formal_evaluation() -> None:
    config = load_config()
    assert config.evaluation.formal_evaluation_enabled is False
    assert config.retriever.threshold_status is CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED

    raw = load_raw_config()
    raw["evaluation"]["formal_evaluation_enabled"] = True
    with pytest.raises(ValidationError, match="formal evaluation requires every threshold"):
        ProjectConfig.model_validate(raw)


Mutation = Callable[[dict[str, Any]], None]


def set_nested(*path_and_value: object) -> Mutation:
    *path, value = path_and_value

    def mutate(raw: dict[str, Any]) -> None:
        target: dict[str, Any] = raw
        for key in path[:-1]:
            target = target[str(key)]
        target[str(path[-1])] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_nested("fast_ttt", "bottleneck_dim", 512), "bottleneck_dim must be 768"),
        (set_nested("spatial_encoder", "active_slots", 16), "active_slots must be 32"),
        (set_nested("state_resampler", "num_queries", 8), "num_queries must be 16"),
        (set_nested("fast_ttt", "optimizer", "momentum", 0.9), "momentum must be 0.0"),
        (set_nested("retriever", "top_k", 16), "retriever.top_k must be None"),
        (set_nested("model", "vision", "deepstack_visual_indexes", [7, 15, 23]), "deepstack"),
        (set_nested("query_encoder", "num_heads", 10), "num_heads must be 12"),
    ],
)
def test_v3_values_and_illegal_combinations_fail_before_startup(
    mutate: Mutation, message: str
) -> None:
    raw = copy.deepcopy(load_raw_config())
    mutate(raw)

    with pytest.raises(ValidationError, match=message):
        ProjectConfig.model_validate(raw)


def test_unknown_config_keys_are_rejected() -> None:
    raw = load_raw_config()
    raw["surprise_gate"] = {"enabled": True}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProjectConfig.model_validate(raw)


def test_active_docs_and_tests_no_longer_claim_v3_is_the_current_v5_config() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    active_config_and_tests = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in (ROOT / "configs", ROOT / "tests")
        for path in sorted(directory.glob("*"))
        if path.suffix in {".py", ".yaml"}
    )

    assert "P1 已实现并有契约测试" in readme
    assert "P1 已验证边界" in decisions
    assert not (ROOT / "tests" / "test_v3_architecture_config.py").exists()
    legacy_yaml_value = "bottleneck_dim" + ": 512"
    legacy_test_assertion = 'bottleneck_dim"]' + " == 512"
    assert legacy_yaml_value not in active_config_and_tests
    assert legacy_test_assertion not in active_config_and_tests
