from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ttt_svcbench_qwen.config import (
    CONFIG_SCHEMA_VERSION,
    SPEC_VERSION,
    ProjectConfig,
    load_config,
)

CONFIG_PATH = Path("configs/model_state_ttt_8b.yaml")


def _raw() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_schema13_slot_memory_and_robust_query_contract_roundtrips() -> None:
    config = load_config(CONFIG_PATH)
    assert config.config_schema_version == CONFIG_SCHEMA_VERSION == 13
    assert (
        config.spec_version
        == SPEC_VERSION
        == "state_ttt_qwen3vl8b_slot_memory_delta_v1"
    )
    assert config.fast_memory.write_rule == "parallel_delta_rule"
    assert config.fast_memory.key_source == "gated_probe_over_token_keys"
    assert config.fast_memory.value_source == "spatial_slot_state_detached"
    assert config.fast_memory.memory_dtype == "float32"
    assert config.fast_memory.zero_init_per_video is True
    assert config.fast_memory.eta_chunk_budget == 1.0
    assert config.a5.query_meta_gradient.mode == "per_query_global_norm_clip_sum"
    assert config.a5.query_meta_gradient.max_norm == 10.0
    assert config.a5.query_meta_gradient.epsilon == 1.0e-12
    assert config.a5.counterfactual_audit.references == ("episode_zero", "segment_start")
    assert config.associative_ttt.contract == "bank_conditioned_slot_memory_v3"
    assert config.associative_ttt.bank_embedding_dim == 512
    assert config.associative_ttt.key_dim == 768
    assert config.associative_ttt.bank_empty_policy == "zero"
    assert config.fast_ttt.online_parameter_count == 589_824
    assert config.a5.warmup.max_steps == 128
    assert config.a5.warmup.linear_warmup_steps == 4
    assert config.a5.warmup.state_learning_rate == 1.0e-5
    assert ProjectConfig.model_validate(config.model_dump()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract", "legacy"),
        ("contract", "bank_conditioned_state_write_v2"),
        ("bank_embedding_dim", 256),
        ("key_dim", 512),
        ("bank_empty_policy", "learned"),
    ],
)
def test_associative_contract_drift_fails_before_startup(field: str, value: object) -> None:
    raw = _raw()
    associative = deepcopy(raw["associative_ttt"])
    assert isinstance(associative, dict)
    associative[field] = value
    raw["associative_ttt"] = associative
    with pytest.raises((ValidationError, ValueError)):
        ProjectConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("write_rule", "sequential_delta_rule"),
        ("key_source", "raw_slot_state"),
        ("value_source", "spatial_slot_state_live"),
        ("eta_chunk_budget", 1.5),
        ("forget_beta_max", 0.9),
        ("eta_gate_init", 0.5),
        ("memory_dtype", "bfloat16"),
        ("zero_init_per_video", False),
    ],
)
def test_fast_memory_contract_drift_fails_before_startup(field: str, value: object) -> None:
    raw = _raw()
    memory = deepcopy(raw["fast_memory"])
    assert isinstance(memory, dict)
    memory[field] = value
    raw["fast_memory"] = memory
    with pytest.raises((ValidationError, ValueError)):
        ProjectConfig.model_validate(raw)


def test_removed_predictor_and_manual_loss_fields_are_forbidden() -> None:
    raw = _raw()
    raw["predictor"] = {"input_dim": 768}
    with pytest.raises(ValidationError, match="predictor"):
        ProjectConfig.model_validate(raw)

    raw = _raw()
    loss = deepcopy(raw["loss"])
    assert isinstance(loss, dict)
    loss["pred_weight"] = 1.0
    raw["loss"] = loss
    with pytest.raises(ValidationError, match="pred_weight"):
        ProjectConfig.model_validate(raw)


def test_associative_optimizer_budget_matches_qwen_and_state_rss() -> None:
    config = load_config(CONFIG_PATH)
    caps = config.outer_gradient_control.max_grad_norm
    reference = 5.0e-6
    assert config.a5.optimizer.associative_learning_rate * caps.associative == pytest.approx(
        reference
    )
    state_names = ("state_shared", "state_task", "state_router_time", "state_retrieval")
    rss = sum(
        (config.a5.optimizer.state_learning_rate * getattr(caps, name)) ** 2
        for name in state_names
    ) ** 0.5
    assert rss == pytest.approx(reference)


def test_unknown_config_keys_are_rejected() -> None:
    raw = _raw()
    raw["unknown"] = True
    with pytest.raises(ValidationError, match="unknown"):
        ProjectConfig.model_validate(raw)


def test_removed_lttt_production_symbols_do_not_exist() -> None:
    source_root = Path("src/ttt_svcbench_qwen")
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    for removed in (
        "class TemporalPredictor",
        "class CausalOverlapTTTInputBuilder",
        "IdentityConsistencyInput",
        "EventConsistencyInput",
        "online_overlap_memory",
        "compute_ttt_outer_auxiliary_loss",
    ):
        assert removed not in source
