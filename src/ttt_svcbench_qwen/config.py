"""Load and validate the frozen slot-memory project configuration.

Inputs: one UTF-8 YAML file describing the frozen schema-14 contract.
Outputs: an immutable, fully validated :class:`ProjectConfig`.
Forbidden: model forward logic, training logic, secret values, or platform absolute paths.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "state_ttt_qwen3vl8b_slot_memory_delta_v1"
CONFIG_SCHEMA_VERSION = 14
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TRANSFORMERS_VERSION = "4.57.1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "model_state_ttt_8b.yaml"

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class FrozenModel(BaseModel):  # type: ignore[misc]
    """Base for immutable configuration objects that reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationStatus(StrEnum):
    """Whether a threshold is suitable for frozen formal evaluation."""

    BOOTSTRAP_CALIBRATION_REQUIRED = "bootstrap_calibration_required"
    CALIBRATION_REQUIRED = "calibration_required"
    CALIBRATED = "calibrated"


class AuditLevel(StrEnum):
    """Runtime integrity work retained at each production audit level."""

    OFF = "off"
    BOUNDARY = "boundary"
    FULL = "full"


class DataConfig(FrozenModel):
    grouped_annotation_file: str
    flat_annotation_file: str
    video_directory: str
    group_key_fields: tuple[str, ...]
    group_k_folds: PositiveInt
    fold_seed: NonNegativeInt
    runtime_allowlist: tuple[str, ...]
    runtime_denylist: tuple[str, ...]
    official_clean_selection_forbidden: bool


class VideoPreprocessingConfig(FrozenModel):
    sample_fps: PositiveFloat
    frames_per_chunk: PositiveInt
    stride_frames: PositiveInt
    causal_boundary: str
    processor_shortest_edge: PositiveInt
    processor_longest_edge: PositiveInt
    patch_size: PositiveInt
    temporal_patch_size: PositiveInt
    spatial_merge_size: PositiveInt
    pad_value: float
    full_tubelet_required_for_state: bool


class VisionConfig(FrozenModel):
    depth: PositiveInt
    hidden_size: PositiveInt
    num_heads: PositiveInt
    patch_size: PositiveInt
    temporal_patch_size: PositiveInt
    spatial_merge_size: PositiveInt
    output_size: PositiveInt
    deepstack_visual_indexes: tuple[int, ...]


class LLMConfig(FrozenModel):
    num_layers: PositiveInt
    hidden_size: PositiveInt


class OnlineFreezeConfig(FrozenModel):
    vision: bool
    merger: bool
    deepstack: bool
    llm: bool


class ModelConfig(FrozenModel):
    base_model: str
    revision: str
    transformers_version: str
    vision: VisionConfig
    llm: LLMConfig
    online_freeze: OnlineFreezeConfig


class FastTTTConfig(FrozenModel):
    input_dim: PositiveInt
    bottleneck_dim: PositiveInt
    output_dim: PositiveInt
    residual_scale: PositiveFloat
    rms_norm_eps: PositiveFloat
    slow_projection_bias: bool
    fast_bias: bool
    fast_initialization: str
    fast_matrix_count: PositiveInt
    online_parameter_count: PositiveInt
    update_order: str


class FastMemoryConfig(FrozenModel):
    """Zero-initialized per-video delta-rule slot memory replacing inner SGD."""

    write_rule: Literal["parallel_delta_rule"]
    key_source: Literal["gated_probe_over_token_keys"]
    value_source: Literal["spatial_slot_state_detached"]
    eta_max_per_slot: PositiveFloat = Field(le=1.0)
    eta_chunk_budget: PositiveFloat = Field(le=1.0)
    eta_gate_hidden_dim: PositiveInt
    eta_gate_init: PositiveFloat
    forget_beta_max: PositiveFloat = Field(le=0.5)
    forget_beta_init: PositiveFloat
    read_gate_init: PositiveFloat
    read_gate_shape: Literal["per_channel"]
    memory_dtype: Literal["float32"]
    zero_init_per_video: Literal[True]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_gate_bounds(self) -> Self:
        if self.eta_gate_init >= self.eta_max_per_slot:
            raise ValueError("fast_memory.eta_gate_init must be below eta_max_per_slot")
        if self.forget_beta_init >= self.forget_beta_max:
            raise ValueError("fast_memory.forget_beta_init must be below forget_beta_max")
        # The K-step BPTT contraction bound requires sum(eta) <= 2 * (1 - beta_max).
        if self.eta_chunk_budget > 2.0 * (1.0 - self.forget_beta_max):
            raise ValueError("fast_memory eta budget violates the write contraction bound")
        return self


class SpatialEncoderConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    stages: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    refinements_per_stage: PositiveInt
    ffn_dim: PositiveInt
    active_slots: PositiveInt
    max_active_slots: PositiveInt
    query_dim: PositiveInt
    layer_norm_eps: PositiveFloat
    slot_initialization: str
    attention_normalization: str
    attention_epsilon: PositiveFloat
    confidence_mode: str
    overflow_policy: str
    slot_valid_mask: bool
    log_overflow: bool


class TemporalEncoderConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    ffn_dim: PositiveInt
    dropout: Probability
    position_encoding: str
    layer_norm_eps: PositiveFloat
    activation: str
    pre_norm: bool
    attention_projection_bias: bool
    strict_causal: bool
    causal_includes_self: bool
    causal_window_includes_current: bool
    cache_tubelets: PositiveInt
    cache_mode: str
    position_id_mode: str
    overlap_policy: str
    overlap_tubelets: PositiveInt
    replay_context_tubelets: PositiveInt
    cache_owner_keys: tuple[str, ...]
    detach_cache_default: bool
    query_dim: PositiveInt
    parameter_count: PositiveInt


class O1Config(FrozenModel):
    input_dim: PositiveInt
    query_dim: PositiveInt
    film_dim: PositiveInt
    hidden_dims: tuple[int, ...]
    output_dim: PositiveInt
    output_names: tuple[str, ...]
    layer_norm_eps: PositiveFloat
    film_mode: str
    activation: str
    dropout: Probability
    linear_bias: bool
    parameter_count: PositiveInt
    object_threshold: Probability
    target_threshold: Probability
    visible_threshold: Probability
    enter_threshold: Probability
    exit_threshold: Probability
    confidence_threshold: Probability
    baseline_policy: str
    count_update_policy: str
    committed_position_policy: str
    threshold_status: CalibrationStatus


class O2Config(FrozenModel):
    input_dim: PositiveInt
    hidden_dims: tuple[int, ...]
    identity_dim: PositiveInt
    score_dim: PositiveInt
    score_names: tuple[str, ...]
    layer_norm_eps: PositiveFloat
    activation: str
    dropout: Probability
    linear_bias: bool
    identity_normalization: str
    normalization_eps: PositiveFloat
    parameter_count: PositiveInt
    prototype_ema: Probability
    confirmation_observations: PositiveInt
    match_threshold: Probability
    novelty_threshold: Probability
    match_confidence_threshold: Probability
    reliability_threshold: Probability
    candidate_low_confidence_threshold: Probability
    match_ambiguity_margin: PositiveFloat
    threshold_status: CalibrationStatus
    relevance_gate_mode: Literal["audit_only", "enforce"]
    relevance_threshold: Probability | None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate_relevance_gate(self) -> O2Config:
        if self.relevance_gate_mode == "enforce" and self.relevance_threshold is None:
            raise ValueError("O2 relevance enforce mode requires a calibrated threshold")
        return self


class E1Config(FrozenModel):
    input_dim: PositiveInt
    channels: PositiveInt
    num_layers: PositiveInt
    kernel_size: PositiveInt
    dilations: tuple[int, ...]
    output_dim: PositiveInt
    output_names: tuple[str, ...]
    layer_norm_eps: PositiveFloat
    activation: str
    strict_causal: bool
    batch_norm: bool
    dropout: Probability
    convolution_bias: bool
    causal_padding: str
    receptive_field: PositiveInt
    streaming_state_mode: str
    overlap_tubelets: PositiveInt
    history_tubelets: PositiveInt
    state_owner_keys: tuple[str, ...]
    detach_runtime_default: bool
    parameter_count: PositiveInt
    tau_on: Probability
    tau_off: Probability
    completion_threshold: Probability
    transition_threshold: Probability
    min_gap_seconds: NonNegativeFloat
    fsm_policy: str
    cooldown_nms_source: str
    committed_position_policy: str
    threshold_status: CalibrationStatus


class E2Config(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    num_layers: PositiveInt
    event_output_dim: PositiveInt
    phase_output_dim: PositiveInt
    event_names: tuple[str, ...]
    phase_names: tuple[str, ...]
    layer_norm_eps: PositiveFloat
    bidirectional: bool
    batch_first: bool
    bias: bool
    dropout: Probability
    streaming_state_mode: str
    overlap_tubelets: PositiveInt
    checkpoint_tubelets: PositiveInt
    state_owner_keys: tuple[str, ...]
    detach_runtime_default: bool
    parameter_count: PositiveInt
    start_threshold: Probability
    end_threshold: Probability
    complete_threshold: Probability
    rearm_max_event_probability: Probability
    rearm_phase: str
    completed_hold_positions: PositiveInt
    fsm_policy: str
    active_evidence_policy: str
    committed_position_policy: str
    threshold_status: CalibrationStatus


class ObservationHeadsConfig(FrozenModel):
    temporal_input_conditioning: str
    raw_logits: bool
    debug_probabilities: bool
    output_valid_mask: bool
    output_timestamps: bool
    output_position_ids: bool
    invalid_output_policy: str
    online_frozen: bool
    online_forward_no_grad: bool
    detach_inputs: bool
    hard_state_mutation: bool
    o1: O1Config
    o2: O2Config
    e1: E1Config
    e2: E2Config


class SemanticProjectorConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    output_dim: PositiveInt
    head_type_count: PositiveInt
    head_types: tuple[str, ...]
    layer_norm_eps: PositiveFloat
    activation: str
    dropout: Probability
    linear_bias: bool
    normalization_dtype: str
    normalization_eps: PositiveFloat
    zero_norm_fallback: str
    parameter_count: PositiveInt
    included_in_model_state_dict: bool
    included_in_outer_optimizer: bool
    included_in_inner_optimizer: bool
    online_frozen: bool
    online_forward_no_grad: bool
    detach_inputs: bool


class ConfirmedStoreConfig(FrozenModel):
    initial_capacity: PositiveInt
    growth_chunk: PositiveInt
    hard_limit: PositiveInt | None
    storage_device: str
    storage_dtype: str
    exact_search: bool
    ann_enabled: bool
    gpu_hot_capacity: PositiveInt
    hot_cache_enabled: bool
    hot_cache_device: str
    hot_cache_dtype: str
    eviction_policy: str


class CandidateStoreConfig(FrozenModel):
    initial_capacity: PositiveInt
    growth_chunk: PositiveInt
    hard_limit: PositiveInt
    ttl_chunks: PositiveInt
    match_threshold: Probability
    reliability_threshold: Probability
    low_confidence_threshold: Probability
    ttl_refresh_policy: str
    ttl_aging_policy: str
    promotion_policy: str
    overflow_policy: str
    prune_order: tuple[str, ...]


class StateBankConfig(FrozenModel):
    semantic_dim: PositiveInt
    identity_dim: PositiveInt
    semantic_projector: SemanticProjectorConfig
    confirmed_store: ConfirmedStoreConfig
    candidate_store: CandidateStoreConfig
    event_history_capacity: PositiveInt
    retrieval_history_capacity_per_head: PositiveInt
    retrieval_history_source_dim: PositiveInt
    isolation_keys: tuple[str, ...]
    hard_updates_no_grad: bool
    detach_before_write: bool
    runtime_in_model_state_dict: bool
    runtime_registered_parameters: bool
    runtime_registered_buffers: bool
    runtime_in_outer_optimizer: bool
    runtime_in_inner_optimizer: bool
    snapshot_separate_from_model_checkpoint: bool
    record_time_metadata_policy: str
    record_id_policy: str
    aggregate_record_heads: tuple[str, ...]
    aggregate_update_mode: str
    committed_position_policy: str
    o2_p9_policy: str
    o2_lifecycle_owner: str
    o2_candidate_retrieval_eligible: bool
    o2_confirmed_retrieval_eligible: bool
    dynamic_view_padding: str
    n_state_definition: str
    event_kind_provenance: str


class QueryEncoderConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    ffn_dim: PositiveInt
    dropout: Probability
    output_dim: PositiveInt
    bidirectional: bool
    position_encoding: str
    pooling: str


class OperatorRouterConfig(FrozenModel):
    prototypes: tuple[str, ...]
    output_dim: PositiveInt
    temperature_initial: PositiveFloat
    temperature_trainable: bool
    confidence_threshold: Probability | None
    threshold_status: CalibrationStatus


class TimeResolverConfig(FrozenModel):
    modes: tuple[str, ...]
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    mode_count: PositiveInt
    token_hidden_dim: PositiveInt
    pointer_heads: PositiveInt
    confidence_threshold: Probability | None
    threshold_status: CalibrationStatus


class RetrieverConfig(FrozenModel):
    semantic_dim: PositiveInt
    record_similarity_threshold: Probability
    threshold_status: CalibrationStatus
    similarity_dtype: str
    normalization_eps: PositiveFloat
    zero_query_policy: str
    threshold_comparison: str
    record_confidence_threshold: Probability | None
    operator_head_types: tuple[str | None, ...]
    filter_order: tuple[str, ...]
    selection_order: tuple[str, ...]
    owner_mismatch_status: str
    aggregate_time_policy: str
    atomic_window_boundary: str
    metrics_policy: str
    score_chunk_size: PositiveInt
    top_k: PositiveInt | None
    ann_enabled: bool


class StateResamplerConfig(FrozenModel):
    num_queries: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    ffn_dim: PositiveInt
    hidden_dim: PositiveInt
    output_dim: PositiveInt
    layer_norm_eps: PositiveFloat
    activation: str
    dropout: Probability
    attention_bias: bool
    output_projection_bias: bool
    attention_softmax_dtype: str
    empty_record_embedding: bool
    empty_record_policy: str
    attention_audit: str
    parameter_count: PositiveInt


class StateReaderConfig(FrozenModel):
    signed_exact_count: bool
    empty_exact_count: NonNegativeInt
    status_propagation: str
    o1_delta_policy: str
    o2_identity_key: str
    point_window_boundary: str
    e1_history_policy: str
    e1_truncated_window_status: str
    e2_window_anchor: str
    event_kind_mismatch_status: str
    number_text_format: str
    tokenizer_add_special_tokens: bool
    tokenizer_roundtrip_required: bool
    tokenizer_class: str
    tokenizer_vocab_size: PositiveInt
    tokenizer_required_files: tuple[str, ...]
    tokenizer_manifest_sha256: str
    ground_truth_input_forbidden: bool


class InputComposerConfig(FrozenModel):
    state_token_count: PositiveInt
    special_tokens: tuple[str, ...]
    special_token_ids: tuple[NonNegativeInt, ...]
    tokenizer_base_length: PositiveInt
    tokenizer_extended_length: PositiveInt
    tokenizer_revision: str
    model_embedding_rows: PositiveInt
    embedding_initialization: str
    embedding_source_token_ids: tuple[NonNegativeInt, ...]
    initialize_input_and_output_embeddings: bool
    padding_side: str
    state_payload_statuses: tuple[str, ...]
    invalid_payload_policy: str
    payload_order: tuple[str, ...]
    prefill_once: bool
    generation_num_beams: PositiveInt


class AssociativeTTTConfig(FrozenModel):
    """Frozen Bank-conditioned key-context contract feeding the slot memory."""

    contract: Literal["bank_conditioned_slot_memory_v3"]
    bank_embedding_dim: PositiveInt
    key_dim: PositiveInt
    bank_empty_policy: Literal["zero"]


class OfficialWeakBalanceConfig(FrozenModel):
    """Formal EMA Answer-reference composition for the four official-weak terms."""

    group_weight: Probability
    answer_reference_floor: PositiveFloat = 0.1
    scale_min: PositiveFloat
    scale_max: PositiveFloat
    epsilon: PositiveFloat
    ema_beta: Probability = 0.99
    grad_ema_beta: Probability = 0.99
    grad_scale_min: PositiveFloat = 0.1
    grad_scale_max: PositiveFloat = 10.0

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scale_bounds(self) -> Self:
        if self.scale_min > self.scale_max:
            raise ValueError("official-weak scale_min cannot exceed scale_max")
        if self.grad_scale_min > self.grad_scale_max:
            raise ValueError("official-weak grad_scale_min cannot exceed grad_scale_max")
        return self


class LossConfig(FrozenModel):
    operator_weight: NonNegativeFloat
    retrieval_weight: NonNegativeFloat
    time_weight: NonNegativeFloat
    answer_causal_shift: bool
    answer_ignore_index: int
    official_weak_balance: OfficialWeakBalanceConfig


class OuterGradientControlMode(StrEnum):
    PER_GROUP_L2_EQUAL_UPDATE_CAP = "per_group_l2_equal_update_cap"


class OuterNonfinitePolicy(StrEnum):
    SKIP_UPDATE = "skip_update"


class OuterMaxGradNormConfig(FrozenModel):
    qwen: PositiveFloat
    fast_slow: PositiveFloat
    state_shared: PositiveFloat
    state_task: PositiveFloat
    state_router_time: PositiveFloat
    state_retrieval: PositiveFloat
    w0: PositiveFloat
    associative: PositiveFloat


class OuterGradientControlConfig(FrozenModel):
    """Fixed A2/A5 Outer gradient caps; audit state never affects optimization."""

    mode: OuterGradientControlMode
    max_grad_norm: OuterMaxGradNormConfig
    nonfinite_policy: OuterNonfinitePolicy
    audit_steps: PositiveInt


class A2OptimizerConfig(FrozenModel):
    qwen_learning_rate: PositiveFloat
    state_learning_rate: PositiveFloat
    w0_learning_rate: PositiveFloat


class A2TrainingConfig(FrozenModel):
    optimizer: A2OptimizerConfig


class A5OptimizerConfig(FrozenModel):
    """Non-Qwen A5 parameter-group learning rates owned by State-TTT."""

    fast_slow_learning_rate: PositiveFloat
    state_learning_rate: PositiveFloat
    w0_learning_rate: PositiveFloat
    associative_learning_rate: PositiveFloat


class A5CounterfactualAuditConfig(FrozenModel):
    """No-grad Query references used only for diagnostic causal-effect measurement."""

    enabled: bool = False
    interval_steps: PositiveInt = 8
    queries_per_rank: PositiveInt = 1
    references: tuple[Literal["episode_zero"], Literal["segment_start"]] = (
        "episode_zero",
        "segment_start",
    )


class A5QueryMetaGradientConfig(FrozenModel):
    """Robust aggregation applied only to Query fast-weight cotangents."""

    mode: Literal["per_query_global_norm_clip_sum"]
    max_norm: PositiveFloat
    epsilon: PositiveFloat


class A5WarmupConfig(FrozenModel):
    """Independent Memory/State handoff stage; values are part of schema-14."""

    max_steps: Literal[256]
    linear_warmup_steps: Literal[4]
    fast_slow_learning_rate: NonNegativeFloat
    state_learning_rate: PositiveFloat
    w0_learning_rate: NonNegativeFloat
    associative_learning_rate: PositiveFloat
    bundle_schema_version: Literal[2]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_warmup_contract(self) -> Self:
        expected = (0.0, 1.0e-5, 0.0, 5.0e-5)
        actual = (
            float(self.fast_slow_learning_rate),
            float(self.state_learning_rate),
            float(self.w0_learning_rate),
            float(self.associative_learning_rate),
        )
        if actual != expected:
            raise ValueError("A5 Memory/State warmup learning rates drifted from schema-14")
        return self


class A5TrainingConfig(FrozenModel):
    """Direct A5 contract with K-step truncation and one episode seed."""

    truncation_horizon: PositiveInt
    seed: NonNegativeInt
    optimizer: A5OptimizerConfig
    warmup: A5WarmupConfig
    query_meta_gradient: A5QueryMetaGradientConfig
    counterfactual_audit: A5CounterfactualAuditConfig = A5CounterfactualAuditConfig()


class InferenceRuntimeConfig(FrozenModel):
    """Production inference audit cost selected at the process boundary."""

    audit_level: AuditLevel


_FROZEN_CONTRACT: dict[str, object] = {
    "spec_version": SPEC_VERSION,
    "config_schema_version": CONFIG_SCHEMA_VERSION,
    "data": {
        "grouped_annotation_file": "data/vcbench_data.jsonl",
        "flat_annotation_file": "data/vcbench_eval.jsonl",
        "video_directory": "data/videos",
        "group_key_fields": ("source_dataset", "video_path"),
        "group_k_folds": 5,
        "fold_seed": 42,
        "runtime_allowlist": ("video", "question", "query_time", "explicit_time_values"),
        "runtime_denylist": (
            "answer",
            "count",
            "occurrence_times",
            "counting_type",
            "counting_subtype",
        ),
        "official_clean_selection_forbidden": True,
    },
    "video_preprocessing": {
        "sample_fps": 2.0,
        "frames_per_chunk": 16,
        "stride_frames": 8,
        "causal_boundary": "right_closed",
        "processor_shortest_edge": 4096,
        "processor_longest_edge": 25_165_824,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "pad_value": 0.0,
        "full_tubelet_required_for_state": True,
    },
    "model": {
        "base_model": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "vision": {
            "depth": 27,
            "hidden_size": 1152,
            "num_heads": 16,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "output_size": 4096,
            "deepstack_visual_indexes": (8, 16, 24),
        },
        "llm": {
            "num_layers": 36,
            "hidden_size": 4096,
        },
        "online_freeze": {
            "vision": True,
            "merger": True,
            "deepstack": True,
            "llm": True,
        },
    },
    "fast_ttt": {
        "input_dim": 4096,
        "bottleneck_dim": 768,
        "output_dim": 4096,
        "residual_scale": 0.1,
        "rms_norm_eps": 1.0e-6,
        "slow_projection_bias": True,
        "fast_bias": False,
        "fast_initialization": "xavier_uniform",
        "fast_matrix_count": 1,
        "online_parameter_count": 589_824,
        "update_order": "observe_state_then_update_for_next_chunk",
    },
    "fast_memory": {
        "write_rule": "parallel_delta_rule",
        "key_source": "gated_probe_over_token_keys",
        "value_source": "spatial_slot_state_detached",
        "eta_chunk_budget": 1.0,
        "read_gate_shape": "per_channel",
        "memory_dtype": "float32",
        "zero_init_per_video": True,
    },
    "spatial_encoder": {
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
        "attention_epsilon": 1.0e-8,
        "slot_valid_mask": True,
        "log_overflow": True,
    },
    "state_bank": {
        "confirmed_store": {
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
        },
        "candidate_store": {
            "initial_capacity": 64,
            "growth_chunk": 64,
            "hard_limit": 512,
            "ttl_chunks": 8,
            "match_threshold": 0.8,
            "reliability_threshold": 0.5,
            "low_confidence_threshold": 0.5,
            "ttl_refresh_policy": "reset_to_full_on_reliable_match",
            "ttl_aging_policy": (
                "match_first_then_unmatched_decrement_once_per_new_committed_position_remove_at_zero_end"
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
        },
        "o2_lifecycle_owner": "identity_bank_p10",
        "o2_candidate_retrieval_eligible": False,
        "o2_confirmed_retrieval_eligible": True,
        "event_kind_provenance": "hard_operator_frozen_per_aggregate",
    },
    "query_encoder": {
        "input_dim": 4096,
        "hidden_dim": 768,
        "num_layers": 4,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "dropout": 0.1,
        "output_dim": 512,
        "bidirectional": True,
        "position_encoding": "sinusoidal",
        "pooling": "learned_attention",
    },
    "operator_router": {
        "output_dim": 512,
        "temperature_initial": 1.0,
        "temperature_trainable": True,
        "confidence_threshold": None,
        "threshold_status": CalibrationStatus.CALIBRATION_REQUIRED,
    },
    "time_resolver": {
        "input_dim": 512,
        "hidden_dim": 256,
        "mode_count": 4,
        "token_hidden_dim": 768,
        "pointer_heads": 2,
        "confidence_threshold": None,
        "threshold_status": CalibrationStatus.CALIBRATION_REQUIRED,
    },
    "retriever": {
        "semantic_dim": 512,
        "record_similarity_threshold": 0.35,
        "similarity_dtype": "float32",
        "normalization_eps": 1.0e-8,
        "zero_query_policy": "unsupported",
        "threshold_comparison": "greater_than_or_equal",
        "record_confidence_threshold": None,
        "filter_order": (
            "invalid",
            "retrieval_ineligible",
            "future",
            "outside_window",
            "below_similarity",
        ),
        "selection_order": ("score_desc", "record_id_asc"),
        "owner_mismatch_status": "invalid",
        "aggregate_time_policy": "causal_availability_only_window_in_reader",
        "atomic_window_boundary": "closed",
        "metrics_policy": "offline_ground_truth_runtime_label_free",
        "top_k": None,
        "ann_enabled": False,
    },
    "state_resampler": {
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
    },
    "state_reader": {
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
    },
    "input_composer": {
        "state_token_count": 16,
        "special_tokens": (
            "<|state_start|>",
            "<|state_pad|>",
            "<|state_end|>",
            "<|number_start|>",
            "<|number_end|>",
        ),
        "special_token_ids": (151_669, 151_670, 151_671, 151_672, 151_673),
        "tokenizer_base_length": 151_669,
        "tokenizer_extended_length": 151_674,
        "tokenizer_revision": BASE_MODEL_REVISION,
        "model_embedding_rows": 151_936,
        "embedding_initialization": "fp32_mean_of_vision_start_video_pad_vision_end_then_cast",
        "embedding_source_token_ids": (151_652, 151_656, 151_653),
        "initialize_input_and_output_embeddings": True,
        "padding_side": "left",
        "state_payload_statuses": ("ok", "empty"),
        "invalid_payload_policy": "omit_state_and_number",
        "payload_order": (
            "system_user_question_video",
            "state",
            "number",
            "user_end",
            "assistant_generation_prefix",
        ),
        "prefill_once": True,
        "generation_num_beams": 1,
    },
    "associative_ttt": {
        "contract": "bank_conditioned_slot_memory_v3",
        "bank_embedding_dim": 512,
        "key_dim": 768,
        "bank_empty_policy": "zero",
    },
    "loss": {
        "operator_weight": 1.0,
        "retrieval_weight": 1.0,
        "time_weight": 1.0,
        "answer_causal_shift": True,
        "answer_ignore_index": -100,
        "official_weak_balance": {
            "group_weight": 0.4,
            "answer_reference_floor": 0.1,
            "scale_min": 0.001,
            "scale_max": 20.0,
            "epsilon": 1.0e-8,
            "ema_beta": 0.99,
            "grad_ema_beta": 0.99,
            "grad_scale_min": 0.1,
            "grad_scale_max": 10.0,
        },
    },
    "outer_gradient_control": {
        "max_grad_norm": {
            "qwen": 1.0,
            "state_shared": 0.05,
            "state_task": 0.05,
            "state_router_time": 0.05,
            "state_retrieval": 0.05,
            "associative": 0.1,
            "w0": 0.1,
        },
        "mode": OuterGradientControlMode.PER_GROUP_L2_EQUAL_UPDATE_CAP,
        "nonfinite_policy": OuterNonfinitePolicy.SKIP_UPDATE,
        "audit_steps": 32,
    },
    "a2": {
        "optimizer": {
            "qwen_learning_rate": 1.0e-5,
            "state_learning_rate": 1.0e-4,
            "w0_learning_rate": 1.0e-4,
        },
    },
    "a5": {
        "truncation_horizon": 8,
        "seed": 42,
        "optimizer": {
            "state_learning_rate": 5.0e-5,
            "w0_learning_rate": 5.0e-5,
            "associative_learning_rate": 5.0e-5,
        },
    },
    "inference": {
        "audit_level": AuditLevel.BOUNDARY,
    },
    "observation_heads": {
        "o1": {
            "output_names": ("object", "target", "visible", "enter", "exit", "confidence"),
            "object_threshold": 0.5,
            "target_threshold": 0.5,
            "visible_threshold": 0.5,
            "enter_threshold": 0.5,
            "exit_threshold": 0.5,
            "confidence_threshold": 0.5,
            "baseline_policy": "explicit_set_once_per_trajectory",
            "count_update_policy": "recompute_from_full_slot_state",
            "committed_position_policy": "idempotent_preserve_and_audit_drift",
        },
        "o2": {
            "score_names": ("novelty", "match_confidence"),
            "prototype_ema": 0.9,
            "confirmation_observations": 2,
            "match_threshold": 0.8,
            "novelty_threshold": 0.5,
            "match_confidence_threshold": 0.5,
            "reliability_threshold": 0.5,
            "candidate_low_confidence_threshold": 0.5,
            "match_ambiguity_margin": 1.0e-6,
            "threshold_status": CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
            "relevance_gate_mode": "audit_only",
            "relevance_threshold": None,
        },
        "e1": {
            "output_names": ("eventness", "completion", "transition"),
            "tau_on": 0.7,
            "tau_off": 0.3,
            "completion_threshold": 0.7,
            "transition_threshold": 0.7,
            "min_gap_seconds": 0.5,
            "fsm_policy": "eventness_hysteresis_completion_transition",
            "cooldown_nms_source": "min_gap_seconds",
            "committed_position_policy": "idempotent_ignore_and_audit",
        },
        "e2": {
            "event_names": ("start", "active", "end", "complete"),
            "phase_names": ("inactive", "active", "end_candidate", "completed"),
            "start_threshold": 0.6,
            "end_threshold": 0.6,
            "complete_threshold": 0.7,
            "rearm_max_event_probability": 0.5,
            "rearm_phase": "inactive",
            "completed_hold_positions": 1,
            "fsm_policy": "phase_gated_single_transition_per_position",
            "active_evidence_policy": "diagnostic_and_phase_consistency_only",
            "committed_position_policy": "idempotent_ignore_and_audit",
        },
    },
}


def _validate_frozen_contract(node: object, contract: dict[str, object], prefix: str) -> None:
    for name, expected in contract.items():
        path = f"{prefix}{name}"
        actual = getattr(node, name)
        if isinstance(expected, dict):
            _validate_frozen_contract(actual, expected, f"{path}.")
        elif actual != expected:
            raise ValueError(f"{path} must be {expected!r}; got {actual!r}")


class ProjectConfig(FrozenModel):
    """Schema-13 production configuration with cross-component contract validation."""

    spec_version: str
    config_schema_version: Literal[14]
    data: DataConfig
    video_preprocessing: VideoPreprocessingConfig
    model: ModelConfig
    fast_ttt: FastTTTConfig
    fast_memory: FastMemoryConfig
    spatial_encoder: SpatialEncoderConfig
    temporal_encoder: TemporalEncoderConfig
    observation_heads: ObservationHeadsConfig
    state_bank: StateBankConfig
    query_encoder: QueryEncoderConfig
    operator_router: OperatorRouterConfig
    time_resolver: TimeResolverConfig
    retriever: RetrieverConfig
    state_resampler: StateResamplerConfig
    state_reader: StateReaderConfig
    input_composer: InputComposerConfig
    associative_ttt: AssociativeTTTConfig
    loss: LossConfig
    outer_gradient_control: OuterGradientControlConfig
    a2: A2TrainingConfig
    a5: A5TrainingConfig
    inference: InferenceRuntimeConfig

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_production_contract(self) -> Self:
        _validate_frozen_contract(self, _FROZEN_CONTRACT, "")
        if self.a5.query_meta_gradient.mode != "per_query_global_norm_clip_sum":
            raise ValueError(
                "a5.query_meta_gradient.mode must be 'per_query_global_norm_clip_sum'"
            )
        self._validate_attention_dimensions()
        self._validate_video_preprocessing_contract()
        self._validate_head_contracts()
        self._validate_state_and_query_contracts()
        self._validate_eta_budget_headroom()
        return self

    def _validate_eta_budget_headroom(self) -> None:
        """Refuse an eta gate that saturates its chunk budget at initialization.

        ``eta`` is gated per slot and then renormalized so each chunk's sum stays
        within ``eta_chunk_budget``.  With ``active_slots`` slots each starting at
        ``eta_gate_init`` the initial sum is their product, so if that exceeds the
        budget the renormalizer fires on *every* write from step zero.  Two
        consequences, neither of which anything else catches: the documented G4
        pipeline-health gate requires a renormalization rate below 20% and would
        report 100% forever, and the learned gate can then only set the relative
        distribution of eta across slots, never the total, so "outer-learned
        write gain" is half disabled by construction.

        The product is the initial sum *exactly* only because the eta gate's output
        projection is zero-initialized (``fast_ttt``).  With a Xavier weight there
        the gate's data term adds a seed-dependent offset to every logit, so this
        guard would bound an intent rather than the realized sum -- the reason the
        first attempt at G4 lowered ``eta_gate_init`` and changed nothing.

        This guard is cross-section (spatial x fast_memory), which is why it
        cannot live on ``FastMemoryConfig`` and why the drift went unnoticed.
        """

        slots = self.spatial_encoder.active_slots
        initial_sum = float(slots) * self.fast_memory.eta_gate_init
        if initial_sum > self.fast_memory.eta_chunk_budget:
            raise ValueError(
                "active_slots * fast_memory.eta_gate_init "
                f"({slots} * {self.fast_memory.eta_gate_init} = {initial_sum:g}) exceeds "
                f"fast_memory.eta_chunk_budget ({self.fast_memory.eta_chunk_budget}); "
                "the eta renormalizer would saturate from step zero and G4 could never pass"
            )

    def _validate_video_preprocessing_contract(self) -> None:
        video = self.video_preprocessing
        vision = self.model.vision
        if video.patch_size != vision.patch_size:
            raise ValueError("video_preprocessing.patch_size must match model.vision.patch_size")
        if video.temporal_patch_size != vision.temporal_patch_size:
            raise ValueError(
                "video_preprocessing.temporal_patch_size must match "
                "model.vision.temporal_patch_size"
            )
        if video.spatial_merge_size != vision.spatial_merge_size:
            raise ValueError(
                "video_preprocessing.spatial_merge_size must match model.vision.spatial_merge_size"
            )
        if video.frames_per_chunk % video.temporal_patch_size != 0:
            raise ValueError("frames_per_chunk must be divisible by temporal_patch_size")
        if video.stride_frames > video.frames_per_chunk:
            raise ValueError("stride_frames cannot exceed frames_per_chunk")

    def _validate_attention_dimensions(self) -> None:
        attention = (
            (
                "spatial_encoder",
                self.spatial_encoder.hidden_dim,
                self.spatial_encoder.num_heads,
                self.spatial_encoder.head_dim,
            ),
            (
                "temporal_encoder",
                self.temporal_encoder.hidden_dim,
                self.temporal_encoder.num_heads,
                self.temporal_encoder.head_dim,
            ),
            (
                "query_encoder",
                self.query_encoder.hidden_dim,
                self.query_encoder.num_heads,
                self.query_encoder.head_dim,
            ),
            (
                "state_resampler",
                self.state_resampler.hidden_dim,
                self.state_resampler.num_heads,
                self.state_resampler.head_dim,
            ),
        )
        for name, hidden_dim, num_heads, head_dim in attention:
            if hidden_dim % num_heads != 0:
                raise ValueError(f"{name}.hidden_dim must be divisible by num_heads")
            if hidden_dim // num_heads != head_dim:
                raise ValueError(f"{name}.head_dim must equal hidden_dim // num_heads")

    def _validate_head_contracts(self) -> None:
        heads = self.observation_heads
        e1_receptive_field = 1 + (heads.e1.kernel_size - 1) * sum(heads.e1.dilations)
        if heads.e1.receptive_field != e1_receptive_field:
            raise ValueError("observation_heads.e1 receptive field does not match its dilations")
        if heads.e1.history_tubelets != (e1_receptive_field - 1 + heads.e1.overlap_tubelets):
            raise ValueError(
                "observation_heads.e1 streaming history must cover context and overlap"
            )
        if heads.e2.checkpoint_tubelets != heads.e2.overlap_tubelets + 1:
            raise ValueError(
                "observation_heads.e2 rollback checkpoints must cover overlap plus anchor"
            )
        if heads.e1.completion_threshold != heads.e1.tau_on or (
            heads.e1.transition_threshold != heads.e1.tau_on
        ):
            raise ValueError("P9 E1 completion/transition thresholds must reuse tau_on")

    def _validate_state_and_query_contracts(self) -> None:
        prototypes = (
            "o1-snap",
            "o1-delta",
            "o2-unique",
            "o2-gain",
            "e1-action",
            "e1-transit",
            "e2-periodic",
            "e2-episode",
            "unsupported",
        )
        if self.operator_router.prototypes != prototypes:
            raise ValueError("operator_router.prototypes must contain the frozen 9 operators")
        expected_retriever_heads = (
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
        if self.retriever.operator_head_types != expected_retriever_heads:
            raise ValueError("retriever.operator_head_types must align with the frozen 9 operators")
        state_resampler = self.state_resampler
        attention_parameters = (
            4 * state_resampler.hidden_dim * state_resampler.hidden_dim
            + 4 * state_resampler.hidden_dim
        )
        ffn_parameters = (
            2 * state_resampler.hidden_dim * state_resampler.ffn_dim
            + state_resampler.ffn_dim
            + state_resampler.hidden_dim
        )
        norm_parameters = 3 * 2 * state_resampler.hidden_dim
        expected_resampler_parameters = (
            state_resampler.num_layers
            * (2 * attention_parameters + ffn_parameters + norm_parameters)
            + state_resampler.num_queries * state_resampler.hidden_dim
            + state_resampler.hidden_dim
            + state_resampler.hidden_dim * state_resampler.output_dim
            + state_resampler.output_dim
        )
        if state_resampler.parameter_count != expected_resampler_parameters:
            raise ValueError("state_resampler.parameter_count must match the frozen P12 topology")
        if self.time_resolver.modes != ("now", "history", "recent", "explicit_range"):
            raise ValueError("time_resolver.modes must contain the frozen 4 modes")
        if self.state_bank.isolation_keys != ("video_id", "trajectory_id", "head_type"):
            raise ValueError("state_bank.isolation_keys must isolate video, trajectory, and head")
        o2 = self.observation_heads.o2
        candidate = self.state_bank.candidate_store
        if candidate.match_threshold != o2.match_threshold:
            raise ValueError("O2 and Candidate identity match thresholds must agree")
        if candidate.reliability_threshold != o2.reliability_threshold:
            raise ValueError("O2 and Candidate reliability thresholds must agree")
        if candidate.low_confidence_threshold != o2.candidate_low_confidence_threshold:
            raise ValueError("O2 and Candidate low-confidence thresholds must agree")
        projector = self.state_bank.semantic_projector
        if projector.head_type_count != len(projector.head_types):
            raise ValueError("semantic projector head_type_count must match head_types")
        if projector.output_dim != self.state_bank.semantic_dim:
            raise ValueError("semantic projector output must match state_bank.semantic_dim")
        if projector.input_dim != self.state_bank.retrieval_history_source_dim:
            raise ValueError(
                "semantic projector input must match retrieval history source dimension"
            )
        projector_parameter_count = (
            projector.head_type_count * projector.input_dim
            + 2 * projector.input_dim
            + projector.input_dim * projector.hidden_dim
            + projector.hidden_dim
            + projector.hidden_dim * projector.output_dim
            + projector.output_dim
        )
        if projector.parameter_count != projector_parameter_count:
            raise ValueError(
                "semantic projector parameter_count does not match its frozen topology"
            )
        if self.fast_ttt.online_parameter_count != (
            self.fast_ttt.fast_matrix_count * self.fast_ttt.bottleneck_dim**2
        ):
            raise ValueError("fast_ttt.online_parameter_count does not match one memory matrix")
        if self.associative_ttt.key_dim != self.fast_ttt.bottleneck_dim:
            raise ValueError("associative_ttt.key_dim must equal the fast bottleneck dimension")
        if self.spatial_encoder.active_slots > self.spatial_encoder.max_active_slots:
            raise ValueError("spatial_encoder.active_slots cannot exceed max_active_slots")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read one UTF-8 YAML file and reject missing, unknown, or invalid values."""

    config_path = Path(path)
    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return cast(ProjectConfig, ProjectConfig.model_validate(raw))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and print the schema-14 configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    print(load_config(args.config).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
