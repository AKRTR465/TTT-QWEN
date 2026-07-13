from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model_state_ttt_8b.yaml"


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def test_v3_online_update_contract() -> None:
    config = load_config()
    fast_ttt = config["fast_ttt"]

    assert fast_ttt["bottleneck_dim"] == 512
    assert fast_ttt["residual_scale"] == 0.1
    assert fast_ttt["update_order"] == "observe_state_then_update_for_next_chunk"
    assert fast_ttt["optimizer"] == "sgd"
    assert fast_ttt["learning_rate"] == 1.0e-4
    assert fast_ttt["momentum"] == 0.0
    assert fast_ttt["weight_decay"] == 0.0
    assert fast_ttt["steps_per_chunk"] == 1
    assert fast_ttt["grad_clip_norm"] == 1.0
    assert fast_ttt["reset_per_video"] is True
    assert "surprise_gate" not in config


def test_v3_online_freeze_contract() -> None:
    model = load_config()["model"]

    assert model["freeze_vision_online"] is True
    assert model["freeze_merger_online"] is True
    assert model["freeze_deepstack_online"] is True
    assert model["freeze_llm_online"] is True


def test_v3_ttt_loss_contract() -> None:
    ttt_loss = load_config()["ttt_loss"]

    assert ttt_loss == {
        "temporal_prediction_weight": 1.0,
        "identity_overlap_weight": 0.5,
        "event_overlap_weight": 0.5,
        "o1_unlabeled_weight": 0.0,
        "auxiliary_outer_weight": 0.1,
    }


def test_v3_scalable_identity_bank_contract() -> None:
    o2 = load_config()["o2"]
    confirmed = o2["confirmed_store"]
    candidates = o2["candidate_store"]

    assert "max_confirmed_identities" not in o2
    assert confirmed["initial_capacity"] == 128
    assert confirmed["growth_chunk"] == 128
    assert confirmed["hard_limit"] is None
    assert confirmed["storage_device"] == "cpu"
    assert confirmed["storage_dtype"] == "float32"
    assert confirmed["gpu_hot_capacity"] == 128

    assert "max_candidates" not in o2
    assert candidates["initial_capacity"] == 32
    assert candidates["growth_chunk"] == 32
    assert candidates["hard_limit"] == 256
    assert candidates["ttl_chunks"] == 8
    assert candidates["overflow_policy"] == "prune_stale_then_low_confidence"

    matching = o2["matching"]
    assert matching["search_full_confirmed_store"] is True
    assert matching["exact_search_threshold"] == 2048
    assert matching["ann_enabled"] is False
