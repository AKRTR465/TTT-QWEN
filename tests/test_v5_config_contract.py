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
    assert fast.rms_norm_eps == 1.0e-6
    assert fast.slow_projection_bias is True
    assert fast.fast_bias is False
    assert fast.fast_initialization == "xavier_uniform"
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

    spatial = config.spatial_encoder
    assert spatial.model_dump() == {
        "input_dim": 4096,
        "hidden_dim": 768,
        "stages": 2,
        "num_heads": 12,
        "head_dim": 64,
        "refinements_per_stage": 3,
        "ffn_dim": 3072,
        "active_slots": 32,
        "max_active_slots": 64,
        "query_dim": 512,
        "layer_norm_eps": 1.0e-5,
        "slot_initialization": "shared_seed_plus_fixed_sinusoidal_codes",
        "attention_normalization": "softmax_slots_then_normalize_tokens",
        "attention_epsilon": 1.0e-8,
        "confidence_mode": "attention_occupancy",
        "overflow_policy": "preserve_existing_reject_excess",
        "slot_valid_mask": True,
        "log_overflow": True,
    }
    assert spatial.num_heads * spatial.head_dim == spatial.hidden_dim
    assert config.temporal_encoder.model_dump() == {
        "input_dim": 4096,
        "hidden_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "dropout": 0.1,
        "position_encoding": "absolute_sinusoidal",
        "layer_norm_eps": 1.0e-5,
        "activation": "gelu",
        "pre_norm": True,
        "attention_projection_bias": True,
        "strict_causal": True,
        "causal_includes_self": True,
        "causal_window_includes_current": True,
        "cache_tubelets": 64,
        "cache_mode": "layerwise_kv",
        "position_id_mode": "explicit_global",
        "overlap_policy": "replay_replace",
        "overlap_tubelets": 4,
        "replay_context_tubelets": 3,
        "cache_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_cache_default": True,
        "query_dim": 512,
        "parameter_count": 48_438_272,
    }
    heads = config.observation_heads
    assert heads.model_dump(exclude={"o1", "o2", "e1", "e2"}) == {
        "temporal_input_conditioning": "inherited_query_conditioned_h_t",
        "raw_logits": True,
        "debug_probabilities": True,
        "output_valid_mask": True,
        "output_timestamps": True,
        "output_position_ids": True,
        "invalid_output_policy": "zero_tensors_negative_one_metadata",
        "online_frozen": True,
        "online_forward_no_grad": False,
        "detach_inputs": False,
        "hard_state_mutation": False,
    }
    assert heads.o1.model_dump() == {
        "input_dim": 768,
        "query_dim": 512,
        "film_dim": 1536,
        "hidden_dims": (1024, 1024),
        "output_dim": 6,
        "output_names": ("object", "target", "visible", "enter", "exit", "confidence"),
        "layer_norm_eps": 1.0e-5,
        "film_mode": "one_plus_scale_and_shift",
        "activation": "silu",
        "dropout": 0.0,
        "linear_bias": True,
        "parameter_count": 2_632_710,
        "object_threshold": 0.5,
        "target_threshold": 0.5,
        "visible_threshold": 0.5,
        "enter_threshold": 0.5,
        "exit_threshold": 0.5,
        "confidence_threshold": 0.5,
        "baseline_policy": "explicit_set_once_per_trajectory",
        "count_update_policy": "recompute_from_full_slot_state",
        "committed_position_policy": "idempotent_preserve_and_audit_drift",
        "threshold_status": CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
    }
    assert heads.o2.model_dump() == {
        "input_dim": 768,
        "hidden_dims": (1024, 1024),
        "identity_dim": 256,
        "score_dim": 2,
        "score_names": ("novelty", "match_confidence"),
        "layer_norm_eps": 1.0e-5,
        "activation": "silu",
        "dropout": 0.0,
        "linear_bias": True,
        "identity_normalization": "l2_fp32_unit_basis_fallback",
        "normalization_eps": 1.0e-8,
        "parameter_count": 2_103_042,
        "prototype_ema": 0.9,
        "confirmation_observations": 2,
        "match_threshold": 0.8,
        "novelty_threshold": 0.5,
        "match_confidence_threshold": 0.5,
        "reliability_threshold": 0.5,
        "candidate_low_confidence_threshold": 0.5,
        "match_ambiguity_margin": 1.0e-6,
        "threshold_status": CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
    }
    assert heads.e1.model_dump() == {
        "input_dim": 768,
        "channels": 512,
        "num_layers": 5,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16),
        "output_dim": 3,
        "output_names": ("eventness", "completion", "transition"),
        "layer_norm_eps": 1.0e-5,
        "activation": "silu_filter_sigmoid_gate",
        "strict_causal": True,
        "batch_norm": False,
        "dropout": 0.0,
        "convolution_bias": True,
        "causal_padding": "left",
        "receptive_field": 63,
        "streaming_state_mode": "projected_history",
        "overlap_tubelets": 4,
        "history_tubelets": 66,
        "state_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_runtime_default": True,
        "parameter_count": 9_584_643,
        "tau_on": 0.7,
        "tau_off": 0.3,
        "completion_threshold": 0.7,
        "transition_threshold": 0.7,
        "min_gap_seconds": 0.5,
        "fsm_policy": "eventness_hysteresis_completion_transition",
        "cooldown_nms_source": "min_gap_seconds",
        "committed_position_policy": "idempotent_ignore_and_audit",
        "threshold_status": CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
    }
    assert heads.e2.model_dump() == {
        "input_dim": 768,
        "hidden_dim": 768,
        "num_layers": 2,
        "event_output_dim": 4,
        "phase_output_dim": 4,
        "event_names": ("start", "active", "end", "complete"),
        "phase_names": ("inactive", "active", "end_candidate", "completed"),
        "layer_norm_eps": 1.0e-5,
        "bidirectional": False,
        "batch_first": True,
        "bias": True,
        "dropout": 0.0,
        "streaming_state_mode": "hidden_with_rollback_checkpoints",
        "overlap_tubelets": 4,
        "checkpoint_tubelets": 5,
        "state_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_runtime_default": True,
        "parameter_count": 7_094_792,
        "start_threshold": 0.6,
        "end_threshold": 0.6,
        "complete_threshold": 0.7,
        "rearm_max_event_probability": 0.5,
        "rearm_phase": "inactive",
        "completed_hold_positions": 1,
        "fsm_policy": "phase_gated_single_transition_per_position",
        "active_evidence_policy": "diagnostic_and_phase_consistency_only",
        "committed_position_policy": "idempotent_ignore_and_audit",
        "threshold_status": CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
    }
    bank = config.state_bank
    assert bank.semantic_projector.model_dump() == {
        "input_dim": 768,
        "hidden_dim": 1024,
        "output_dim": 512,
        "head_type_count": 4,
        "head_types": ("o1", "o2", "e1", "e2"),
        "layer_norm_eps": 1.0e-5,
        "activation": "silu",
        "dropout": 0.0,
        "linear_bias": True,
        "normalization_dtype": "float32",
        "normalization_eps": 1.0e-8,
        "zero_norm_fallback": "first_unit_basis",
        "parameter_count": 1_316_864,
        "included_in_model_state_dict": True,
        "included_in_outer_optimizer": True,
        "included_in_inner_optimizer": False,
        "online_frozen": True,
        "online_forward_no_grad": False,
        "detach_inputs": False,
    }
    bank_contract = bank.model_dump(
        exclude={"semantic_projector", "confirmed_store", "candidate_store"}
    )
    assert bank_contract == {
        "semantic_dim": 512,
        "identity_dim": 256,
        "event_history_capacity": 512,
        "isolation_keys": ("video_id", "trajectory_id", "head_type"),
        "hard_updates_no_grad": True,
        "detach_before_write": True,
        "runtime_in_model_state_dict": False,
        "runtime_registered_parameters": False,
        "runtime_registered_buffers": False,
        "runtime_in_outer_optimizer": False,
        "runtime_in_inner_optimizer": False,
        "snapshot_separate_from_model_checkpoint": True,
        "record_time_metadata_policy": "exactly_one",
        "record_id_policy": "trajectory_monotonic_never_reuse",
        "aggregate_record_heads": ("o1", "e1", "e2"),
        "aggregate_update_mode": "functional_replace",
        "committed_position_policy": "idempotent_ignore_and_audit",
        "o2_p9_policy": "generic_crud_only_p10_owns_lifecycle",
        "o2_lifecycle_owner": "identity_bank_p10",
        "o2_candidate_retrieval_eligible": False,
        "o2_confirmed_retrieval_eligible": True,
        "dynamic_view_padding": "batch_max",
        "n_state_definition": "owner_head_present_records_before_filters",
        "event_kind_provenance": "hard_operator_frozen_per_aggregate",
    }
    assert bank.confirmed_store.model_dump() == {
        "initial_capacity": 256,
        "growth_chunk": 256,
        "hard_limit": None,
        "storage_device": "cpu",
        "storage_dtype": "float32",
        "exact_search": True,
        "ann_enabled": False,
        "gpu_hot_capacity": 256,
        "hot_cache_enabled": True,
        "hot_cache_device": "cuda",
        "hot_cache_dtype": "bfloat16",
        "eviction_policy": "lru_position_then_identity_id",
    }
    assert bank.candidate_store.model_dump() == {
        "initial_capacity": 64,
        "growth_chunk": 64,
        "hard_limit": 512,
        "ttl_chunks": 8,
        "match_threshold": 0.8,
        "reliability_threshold": 0.5,
        "low_confidence_threshold": 0.5,
        "ttl_refresh_policy": "reset_to_full_on_reliable_match",
        "ttl_aging_policy": (
            "match_first_then_unmatched_decrement_once_per_new_committed_position_"
            "remove_at_zero_end"
        ),
        "promotion_policy": "two_reliable_distinct_consecutive_committed_positions",
        "overflow_policy": "expire_then_low_confidence_then_reject",
        "prune_order": (
            "expired",
            "low_confidence",
            "confidence_asc",
            "last_position_id_asc",
            "candidate_id_asc",
            "reject_new",
        ),
    }


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
        "position_encoding": "sinusoidal",
    }
    assert len(config.operator_router.prototypes) == 9
    assert config.operator_router.prototypes[-1] == "unsupported"
    assert config.operator_router.temperature_initial == 1.0
    assert config.operator_router.temperature_trainable is True
    assert config.operator_router.confidence_threshold is None
    assert config.operator_router.threshold_status is CalibrationStatus.CALIBRATION_REQUIRED
    assert config.time_resolver.confidence_threshold is None
    assert config.time_resolver.threshold_status is CalibrationStatus.CALIBRATION_REQUIRED
    assert config.retriever.record_similarity_threshold == 0.35
    assert config.retriever.similarity_dtype == "float32"
    assert config.retriever.normalization_eps == 1.0e-8
    assert config.retriever.zero_query_policy == "unsupported"
    assert config.retriever.threshold_comparison == "greater_than_or_equal"
    assert config.retriever.record_confidence_threshold is None
    assert config.retriever.operator_head_types == (
        "o1",
        "o1",
        "o2",
        "o2",
        "e1",
        "e1",
        "e2",
        "e2",
        None,
    )
    assert config.retriever.filter_order == (
        "invalid",
        "retrieval_ineligible",
        "future",
        "outside_window",
        "below_similarity",
    )
    assert config.retriever.selection_order == ("score_desc", "record_id_asc")
    assert config.retriever.owner_mismatch_status == "invalid"
    assert config.retriever.aggregate_time_policy == "causal_availability_only_window_in_reader"
    assert config.retriever.atomic_window_boundary == "closed"
    assert config.retriever.metrics_policy == "offline_ground_truth_runtime_label_free"
    assert config.retriever.top_k is None
    assert config.retriever.ann_enabled is False
    assert config.state_resampler.num_queries == 16
    assert config.state_resampler.output_dim == 4096
    assert config.state_resampler.model_dump() == {
        "num_queries": 16,
        "num_layers": 3,
        "num_heads": 8,
        "head_dim": 64,
        "ffn_dim": 2048,
        "hidden_dim": 512,
        "output_dim": 4096,
        "layer_norm_eps": 1.0e-5,
        "activation": "gelu",
        "dropout": 0.0,
        "attention_bias": True,
        "output_projection_bias": True,
        "attention_softmax_dtype": "float32",
        "empty_record_embedding": True,
        "empty_record_policy": "internal_trainable_kv_external_zero_width",
        "attention_audit": "final_layer_mean_heads_selected_mass",
        "parameter_count": 14_722_048,
    }
    assert config.state_reader.model_dump() == {
        "signed_exact_count": True,
        "empty_exact_count": 0,
        "status_propagation": "retriever_exact",
        "o1_delta_policy": "fixed_baseline_v1",
        "o2_identity_key": "identity_id",
        "point_window_boundary": "closed",
        "e1_history_policy": "cumulative_or_retained_completion_times",
        "e1_truncated_window_status": "invalid",
        "e2_window_anchor": "completion_end",
        "event_kind_mismatch_status": "invalid",
        "number_text_format": "canonical_ascii_signed_decimal",
        "tokenizer_add_special_tokens": False,
        "tokenizer_roundtrip_required": True,
        "tokenizer_class": "Qwen2TokenizerFast",
        "tokenizer_vocab_size": 151_643,
        "tokenizer_required_files": (
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        "tokenizer_manifest_sha256": (
            "ccd18347b6d6714d91d4c55b37ff05e473a0f8e84fbcba2bda1401a9572f44c3"
        ),
        "ground_truth_input_forbidden": True,
    }
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

    assert budget.spatial_encoder_millions == 24.81536
    assert budget.temporal_encoder_millions == 48.438272
    assert (
        budget.o1_millions,
        budget.o2_millions,
        budget.e1_millions,
        budget.e2_millions,
    ) == (2.632710, 2.103042, 9.584643, 7.094792)
    assert budget.semantic_projector_millions == 1.316864
    assert budget.new_modules_total_millions == 156.715683
    assert (
        abs(component_total - budget.new_modules_total_millions)
        <= budget.rounding_tolerance_millions
    )
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
        (
            set_nested("fast_ttt", "rms_norm_eps", 1.0e-5),
            "fast_ttt.rms_norm_eps must be 1e-06",
        ),
        (
            set_nested("fast_ttt", "slow_projection_bias", False),
            "fast_ttt.slow_projection_bias must be True",
        ),
        (
            set_nested("fast_ttt", "fast_initialization", "zeros"),
            "fast_ttt.fast_initialization must be 'xavier_uniform'",
        ),
        (set_nested("spatial_encoder", "active_slots", 16), "active_slots must be 32"),
        (
            set_nested("spatial_encoder", "layer_norm_eps", 1.0e-6),
            "spatial_encoder.layer_norm_eps",
        ),
        (
            set_nested("spatial_encoder", "slot_initialization", "learned_slot_bank"),
            "spatial_encoder.slot_initialization",
        ),
        (
            set_nested("spatial_encoder", "attention_normalization", "softmax_tokens"),
            "spatial_encoder.attention_normalization",
        ),
        (
            set_nested("spatial_encoder", "attention_epsilon", 1.0e-6),
            "spatial_encoder.attention_epsilon",
        ),
        (
            set_nested("spatial_encoder", "confidence_mode", "learned_head"),
            "spatial_encoder.confidence_mode",
        ),
        (
            set_nested("spatial_encoder", "overflow_policy", "replace_low_confidence"),
            "spatial_encoder.overflow_policy",
        ),
        (
            set_nested("parameter_budget", "spatial_encoder_millions", 24.88),
            "parameter budget|spatial encoder budget",
        ),
        (
            set_nested("temporal_encoder", "position_encoding", "learned_absolute"),
            "temporal_encoder.position_encoding",
        ),
        (
            set_nested("temporal_encoder", "layer_norm_eps", 1.0e-6),
            "temporal_encoder.layer_norm_eps",
        ),
        (
            set_nested("temporal_encoder", "activation", "relu"),
            "temporal_encoder.activation",
        ),
        (
            set_nested("temporal_encoder", "pre_norm", False),
            "temporal_encoder.pre_norm",
        ),
        (
            set_nested("temporal_encoder", "attention_projection_bias", False),
            "temporal_encoder.attention_projection_bias",
        ),
        (
            set_nested("temporal_encoder", "causal_includes_self", False),
            "temporal_encoder.causal_includes_self",
        ),
        (
            set_nested("temporal_encoder", "causal_window_includes_current", False),
            "temporal_encoder.causal_window_includes_current",
        ),
        (
            set_nested("temporal_encoder", "cache_mode", "final_hidden"),
            "temporal_encoder.cache_mode",
        ),
        (
            set_nested("temporal_encoder", "position_id_mode", "chunk_local"),
            "temporal_encoder.position_id_mode",
        ),
        (
            set_nested("temporal_encoder", "overlap_policy", "append_duplicates"),
            "temporal_encoder.overlap_policy",
        ),
        (
            set_nested("temporal_encoder", "overlap_tubelets", 3),
            "temporal_encoder.overlap_tubelets",
        ),
        (
            set_nested("temporal_encoder", "replay_context_tubelets", 2),
            "temporal_encoder.replay_context_tubelets",
        ),
        (
            set_nested("temporal_encoder", "cache_owner_keys", ["video_id"]),
            "temporal_encoder.cache_owner_keys",
        ),
        (
            set_nested("temporal_encoder", "detach_cache_default", False),
            "temporal_encoder.detach_cache_default",
        ),
        (
            set_nested("temporal_encoder", "parameter_count", 48_487_424),
            "temporal_encoder.parameter_count",
        ),
        (
            set_nested("observation_heads", "raw_logits", False),
            "observation_heads.raw_logits",
        ),
        (
            set_nested("observation_heads", "o1", "film_mode", "direct_scale_shift"),
            "observation_heads.o1.film_mode",
        ),
        (
            set_nested("observation_heads", "o1", "parameter_count", 2_630_000),
            "observation_heads.o1.parameter_count",
        ),
        (
            set_nested("observation_heads", "o1", "object_threshold", 0.6),
            "observation_heads.o1.object_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "identity_normalization", "l2"),
            "observation_heads.o2.identity_normalization",
        ),
        (
            set_nested("observation_heads", "o2", "prototype_ema", 0.8),
            "o2.prototype_ema",
        ),
        (
            set_nested("observation_heads", "o2", "confirmation_observations", 3),
            "o2.confirmation_observations",
        ),
        (
            set_nested("observation_heads", "o2", "match_threshold", 0.75),
            "o2.match_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "novelty_threshold", 0.6),
            "o2.novelty_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "match_confidence_threshold", 0.6),
            "o2.match_confidence_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "reliability_threshold", 0.6),
            "o2.reliability_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "candidate_low_confidence_threshold", 0.4),
            "o2.candidate_low_confidence_threshold",
        ),
        (
            set_nested("observation_heads", "o2", "match_ambiguity_margin", 1.0e-5),
            "o2.match_ambiguity_margin",
        ),
        (
            set_nested("observation_heads", "o2", "threshold_status", "calibrated"),
            "o2.threshold_status",
        ),
        (
            set_nested("observation_heads", "e1", "history_tubelets", 62),
            "observation_heads.e1.history_tubelets",
        ),
        (
            set_nested("observation_heads", "e1", "completion_threshold", 0.6),
            "observation_heads.e1.completion_threshold",
        ),
        (
            set_nested("observation_heads", "e2", "checkpoint_tubelets", 4),
            "observation_heads.e2.checkpoint_tubelets",
        ),
        (
            set_nested("observation_heads", "e2", "rearm_max_event_probability", 0.4),
            "observation_heads.e2.rearm_max_event_probability",
        ),
        (
            set_nested("state_bank", "semantic_projector", "parameter_count", 1_320_000),
            "state_bank.semantic_projector.parameter_count",
        ),
        (
            set_nested("state_bank", "semantic_projector", "included_in_model_state_dict", False),
            "state_bank.semantic_projector.included_in_model_state_dict",
        ),
        (
            set_nested("state_bank", "runtime_in_model_state_dict", True),
            "state_bank.runtime_in_model_state_dict",
        ),
        (
            set_nested("state_bank", "record_time_metadata_policy", "timestamp_or_time_range"),
            "state_bank.record_time_metadata_policy",
        ),
        (
            set_nested("state_bank", "confirmed_store", "exact_search", False),
            "state_bank.confirmed_store.exact_search",
        ),
        (
            set_nested("state_bank", "confirmed_store", "ann_enabled", True),
            "state_bank.confirmed_store.ann_enabled",
        ),
        (
            set_nested("state_bank", "confirmed_store", "hot_cache_dtype", "float32"),
            "state_bank.confirmed_store.hot_cache_dtype",
        ),
        (
            set_nested("state_bank", "confirmed_store", "eviction_policy", "fifo"),
            "state_bank.confirmed_store.eviction_policy",
        ),
        (
            set_nested("state_bank", "candidate_store", "ttl_chunks", 7),
            "state_bank.candidate_store.ttl_chunks",
        ),
        (
            set_nested("state_bank", "candidate_store", "match_threshold", 0.75),
            "state_bank.candidate_store.match_threshold",
        ),
        (
            set_nested("state_bank", "candidate_store", "low_confidence_threshold", 0.4),
            "state_bank.candidate_store.low_confidence_threshold",
        ),
        (
            set_nested("state_bank", "candidate_store", "overflow_policy", "overwrite"),
            "state_bank.candidate_store.overflow_policy",
        ),
        (
            set_nested(
                "state_bank",
                "candidate_store",
                "prune_order",
                ["low_confidence", "expired", "reject_new"],
            ),
            "state_bank.candidate_store.prune_order",
        ),
        (
            set_nested("state_bank", "o2_lifecycle_owner", "state_bank_p9"),
            "state_bank.o2_lifecycle_owner",
        ),
        (
            set_nested("state_bank", "o2_candidate_retrieval_eligible", True),
            "state_bank.o2_candidate_retrieval_eligible",
        ),
        (
            set_nested("state_bank", "event_kind_provenance", "ground_truth_label"),
            "state_bank.event_kind_provenance",
        ),
        (
            set_nested("parameter_budget", "temporal_encoder_millions", 48.49),
            "parameter budget components|temporal encoder budget",
        ),
        (
            set_nested("parameter_budget", "o1_millions", 2.63),
            "parameter budget components|O1 budget",
        ),
        (
            set_nested("parameter_budget", "semantic_projector_millions", 1.32),
            "parameter budget components|Semantic Projector budget",
        ),
        (
            set_nested("parameter_budget", "new_modules_total_millions", 156.75536),
            "parameter budget components",
        ),
        (set_nested("state_resampler", "num_queries", 8), "num_queries must be 16"),
        (
            set_nested("state_resampler", "layer_norm_eps", 1.0e-6),
            "state_resampler.layer_norm_eps",
        ),
        (
            set_nested("state_resampler", "activation", "silu"),
            "state_resampler.activation",
        ),
        (
            set_nested("state_resampler", "dropout", 0.1),
            "state_resampler.dropout",
        ),
        (
            set_nested("state_resampler", "attention_bias", False),
            "state_resampler.attention_bias",
        ),
        (
            set_nested("state_resampler", "output_projection_bias", False),
            "state_resampler.output_projection_bias",
        ),
        (
            set_nested("state_resampler", "attention_softmax_dtype", "float16"),
            "state_resampler.attention_softmax_dtype",
        ),
        (
            set_nested("state_resampler", "empty_record_embedding", False),
            "state_resampler.empty_record_embedding",
        ),
        (
            set_nested("state_resampler", "empty_record_policy", "masked_softmax"),
            "state_resampler.empty_record_policy",
        ),
        (
            set_nested("state_resampler", "attention_audit", "none"),
            "state_resampler.attention_audit",
        ),
        (
            set_nested("state_resampler", "parameter_count", 14_720_000),
            "state_resampler.parameter_count",
        ),
        (
            set_nested("state_reader", "signed_exact_count", False),
            "state_reader.signed_exact_count",
        ),
        (
            set_nested("state_reader", "empty_exact_count", 1),
            "state_reader.empty_exact_count",
        ),
        (
            set_nested("state_reader", "status_propagation", "infer_from_length"),
            "state_reader.status_propagation",
        ),
        (
            set_nested("state_reader", "o1_delta_policy", "window_delta"),
            "state_reader.o1_delta_policy",
        ),
        (
            set_nested("state_reader", "e1_truncated_window_status", "empty"),
            "state_reader.e1_truncated_window_status",
        ),
        (
            set_nested("state_reader", "e2_window_anchor", "interval_start"),
            "state_reader.e2_window_anchor",
        ),
        (
            set_nested("state_reader", "number_text_format", "localized"),
            "state_reader.number_text_format",
        ),
        (
            set_nested("state_reader", "tokenizer_add_special_tokens", True),
            "state_reader.tokenizer_add_special_tokens",
        ),
        (
            set_nested("state_reader", "ground_truth_input_forbidden", False),
            "state_reader.ground_truth_input_forbidden",
        ),
        (set_nested("fast_ttt", "optimizer", "momentum", 0.9), "momentum must be 0.0"),
        (
            set_nested("retriever", "similarity_dtype", "float16"),
            "retriever.similarity_dtype",
        ),
        (
            set_nested("retriever", "normalization_eps", 1.0e-6),
            "retriever.normalization_eps",
        ),
        (
            set_nested("retriever", "zero_query_policy", "first_unit_basis"),
            "retriever.zero_query_policy",
        ),
        (
            set_nested("retriever", "threshold_comparison", "greater_than"),
            "retriever.threshold_comparison",
        ),
        (
            set_nested("retriever", "record_confidence_threshold", 0.5),
            "retriever.record_confidence_threshold",
        ),
        (
            set_nested(
                "retriever",
                "operator_head_types",
                ["o1", "o1", "o2", "o2", "e1", "e1", "e2", "e2", "o1"],
            ),
            "retriever.operator_head_types",
        ),
        (
            set_nested(
                "retriever",
                "filter_order",
                [
                    "retrieval_ineligible",
                    "invalid",
                    "future",
                    "outside_window",
                    "below_similarity",
                ],
            ),
            "retriever.filter_order",
        ),
        (
            set_nested("retriever", "selection_order", ["record_id_asc"]),
            "retriever.selection_order",
        ),
        (
            set_nested("retriever", "owner_mismatch_status", "empty"),
            "retriever.owner_mismatch_status",
        ),
        (
            set_nested("retriever", "aggregate_time_policy", "record_window"),
            "retriever.aggregate_time_policy",
        ),
        (
            set_nested("retriever", "atomic_window_boundary", "half_open"),
            "retriever.atomic_window_boundary",
        ),
        (
            set_nested("retriever", "metrics_policy", "runtime_ground_truth"),
            "retriever.metrics_policy",
        ),
        (set_nested("retriever", "top_k", 16), "retriever.top_k must be None"),
        (set_nested("retriever", "ann_enabled", True), "retriever.ann_enabled must be False"),
        (set_nested("model", "vision", "deepstack_visual_indexes", [7, 15, 23]), "deepstack"),
        (set_nested("query_encoder", "num_heads", 10), "num_heads must be 12"),
        (
            set_nested("query_encoder", "pooling", "mean"),
            "query_encoder.pooling must be 'learned_attention'",
        ),
        (
            set_nested("query_encoder", "position_encoding", "none"),
            "query_encoder.position_encoding must be 'sinusoidal'",
        ),
        (
            set_nested("operator_router", "temperature_initial", 0.5),
            "temperature_initial must be 1.0",
        ),
        (
            set_nested("operator_router", "confidence_threshold", 0.5),
            "operator_router.confidence_threshold must be None",
        ),
        (
            set_nested("time_resolver", "hidden_dim", 128),
            "time_resolver.hidden_dim must be 256",
        ),
        (
            set_nested("time_resolver", "pointer_heads", 1),
            "time_resolver.pointer_heads must be 2",
        ),
        (
            set_nested("time_resolver", "confidence_threshold", 0.5),
            "time_resolver.confidence_threshold must be None",
        ),
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
