from __future__ import annotations

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.inference import main as inference_main
from ttt_svcbench_qwen.llamafactory_trainer import main as training_main
from ttt_svcbench_qwen.production_runtime import build_runtime

V6_ARCHITECTURE_SNAPSHOT = {
    "spec_version": "state_ttt_qwen3vl8b_high_capacity_sgd_v6_retrieval_history",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
    "vision": {
        "output_size": 4096,
        "deepstack_visual_indexes": (8, 16, 24),
    },
    "fast_ttt": {
        "dimensions": (4096, 768, 4096),
        "online_parameter_count": 1_179_648,
        "update_order": "observe_state_then_update_for_next_chunk",
    },
    "state_encoders": {
        "spatial": (768, 2, 32, 64, 24_815_360),
        "temporal": (768, 6, 64, 48_438_272),
    },
    "observation_heads": {
        "o1": 2_632_710,
        "o2": 2_499_843,
        "e1": 9_717_252,
        "e2": 7_293_449,
    },
    "query_and_reader": {
        "query_layers": 4,
        "query_output_dim": 512,
        "state_token_count": 16,
        "state_token_output_dim": 4096,
        "signed_exact_count": True,
    },
    "loss": {
        "pred_weight": 1.0,
        "identity_weight": 0.5,
        "event_weight": 0.5,
        "o1_unlabeled_weight": 0.0,
        "auxiliary_outer_weight": 0.1,
    },
}


def test_v6_architecture_snapshot_is_unchanged() -> None:
    config = load_config()
    actual = {
        "spec_version": config.spec_version,
        "base_model": config.model.base_model,
        "vision": {
            "output_size": config.model.vision.output_size,
            "deepstack_visual_indexes": config.model.vision.deepstack_visual_indexes,
        },
        "fast_ttt": {
            "dimensions": (
                config.fast_ttt.input_dim,
                config.fast_ttt.bottleneck_dim,
                config.fast_ttt.output_dim,
            ),
            "online_parameter_count": config.fast_ttt.online_parameter_count,
            "update_order": config.fast_ttt.update_order,
        },
        "state_encoders": {
            "spatial": (
                config.spatial_encoder.hidden_dim,
                config.spatial_encoder.stages,
                config.spatial_encoder.active_slots,
                config.spatial_encoder.max_active_slots,
                24_815_360,
            ),
            "temporal": (
                config.temporal_encoder.hidden_dim,
                config.temporal_encoder.num_layers,
                config.temporal_encoder.cache_tubelets,
                config.temporal_encoder.parameter_count,
            ),
        },
        "observation_heads": {
            "o1": config.observation_heads.o1.parameter_count,
            "o2": config.observation_heads.o2.parameter_count,
            "e1": config.observation_heads.e1.parameter_count,
            "e2": config.observation_heads.e2.parameter_count,
        },
        "query_and_reader": {
            "query_layers": config.query_encoder.num_layers,
            "query_output_dim": config.query_encoder.output_dim,
            "state_token_count": config.state_resampler.num_queries,
            "state_token_output_dim": config.state_resampler.output_dim,
            "signed_exact_count": config.state_reader.signed_exact_count,
        },
        "loss": {
            "pred_weight": config.loss.pred_weight,
            "identity_weight": config.loss.identity_weight,
            "event_weight": config.loss.event_weight,
            "o1_unlabeled_weight": config.loss.o1_unlabeled_weight,
            "auxiliary_outer_weight": config.loss.auxiliary_outer_weight,
        },
    }
    assert actual == V6_ARCHITECTURE_SNAPSHOT


def test_production_entrypoints_remain_importable() -> None:
    assert callable(training_main)
    assert callable(build_runtime)
    assert callable(inference_main)
