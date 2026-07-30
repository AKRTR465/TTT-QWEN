from __future__ import annotations

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.inference import main as inference_main
from ttt_svcbench_qwen.llamafactory_trainer import main as training_main
from ttt_svcbench_qwen.production_runtime import build_runtime

V13_ARCHITECTURE_SNAPSHOT = {
    "spec_version": "state_ttt_qwen3vl8b_slot_memory_delta_v1",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
    "vision": {
        "output_size": 4096,
        "deepstack_visual_indexes": (8, 16, 24),
    },
    "fast_ttt": {
        "dimensions": (4096, 768, 4096),
        "online_parameter_count": 589_824,
        "update_order": "observe_state_then_update_for_next_chunk",
    },
    "fast_memory": {
        "write_rule": "parallel_delta_rule",
        "key_source": "gated_probe_over_token_keys",
        "value_source": "spatial_slot_state_detached",
        "eta_max_per_slot": 0.25,
        "eta_chunk_budget": 1.0,
        "forget_beta_max": 0.1,
        "read_gate_shape": "per_channel",
        "memory_dtype": "float32",
        "zero_init_per_video": True,
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
    "associative_ttt": {
        "contract": "bank_conditioned_slot_memory_v3",
        "bank_embedding_dim": 512,
        "key_dim": 768,
        "bank_empty_policy": "zero",
    },
    "query_loss": {
        "operator_weight": 1.0,
        "retrieval_weight": 1.0,
        "time_weight": 1.0,
    },
    "query_meta_gradient": {
        "mode": "per_query_global_norm_clip_sum",
        "max_norm": 10.0,
        "epsilon": 1.0e-12,
    },
}


def test_v13_architecture_snapshot_is_unchanged() -> None:
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
        "fast_memory": {
            "write_rule": config.fast_memory.write_rule,
            "key_source": config.fast_memory.key_source,
            "value_source": config.fast_memory.value_source,
            "eta_max_per_slot": config.fast_memory.eta_max_per_slot,
            "eta_chunk_budget": config.fast_memory.eta_chunk_budget,
            "forget_beta_max": config.fast_memory.forget_beta_max,
            "read_gate_shape": config.fast_memory.read_gate_shape,
            "memory_dtype": config.fast_memory.memory_dtype,
            "zero_init_per_video": config.fast_memory.zero_init_per_video,
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
        "associative_ttt": {
            "contract": config.associative_ttt.contract,
            "bank_embedding_dim": config.associative_ttt.bank_embedding_dim,
            "key_dim": config.associative_ttt.key_dim,
            "bank_empty_policy": config.associative_ttt.bank_empty_policy,
        },
        "query_loss": {
            "operator_weight": config.loss.operator_weight,
            "retrieval_weight": config.loss.retrieval_weight,
            "time_weight": config.loss.time_weight,
        },
        "query_meta_gradient": {
            "mode": config.a5.query_meta_gradient.mode,
            "max_norm": config.a5.query_meta_gradient.max_norm,
            "epsilon": config.a5.query_meta_gradient.epsilon,
        },
    }
    assert actual == V13_ARCHITECTURE_SNAPSHOT


def test_production_entrypoints_remain_importable() -> None:
    assert callable(training_main)
    assert callable(build_runtime)
    assert callable(inference_main)
