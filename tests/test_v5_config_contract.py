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


def test_schema12_state_write_fast_and_robust_query_contract_roundtrips() -> None:
    config = load_config(CONFIG_PATH)
    assert config.config_schema_version == CONFIG_SCHEMA_VERSION == 12
    assert (
        config.spec_version
        == SPEC_VERSION
        == "state_ttt_qwen3vl8b_state_write_associative_v3"
    )
    assert config.fast_ttt.optimizer.fast_weight_dtype == "float32"
    assert config.fast_ttt.optimizer.fast_core_dtype == "float32"
    assert config.a5.query_meta_gradient.mode == "per_query_global_norm_clip_sum"
    assert config.a5.query_meta_gradient.max_norm == 1.0
    assert config.a5.query_meta_gradient.epsilon == 1.0e-12
    assert config.associative_ttt.contract == "bank_conditioned_state_write_v2"
    assert config.associative_ttt.bank_embedding_dim == 512
    assert config.associative_ttt.key_dim == config.associative_ttt.target_dim == 768
    assert config.associative_ttt.bank_empty_policy == "zero"
    assert (
        config.associative_ttt.target_source
        == "predicted_active_head_soft_write_stopgrad"
    )
    assert config.associative_ttt.unsupported_target_policy == "skip"
    assert config.associative_ttt.loss == "normalized_fp32_cosine"
    assert config.a5.warmup.max_steps == 128
    assert config.a5.warmup.linear_warmup_steps == 4
    assert config.a5.warmup.state_learning_rate == 1.0e-5
    assert ProjectConfig.model_validate(config.model_dump()) == config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract", "legacy"),
        ("bank_embedding_dim", 256),
        ("key_dim", 512),
        ("target_dim", 512),
        ("bank_empty_policy", "learned"),
        ("target_source", "adapted"),
        ("unsupported_target_policy", "zero"),
        ("loss", "cosine"),
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
