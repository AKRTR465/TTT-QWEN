"""Load and validate the frozen v5 project configuration.

Inputs: UTF-8 YAML plus environment-variable *names* for model, data, cache, and outputs.
Outputs: an immutable, fully validated :class:`ProjectConfig` and environment summaries.
Forbidden: model forward logic, training logic, secret values, or platform absolute paths.
"""

from __future__ import annotations

import argparse
import platform
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self, cast

import torch
import transformers
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "state_ttt_qwen3vl8b_high_capacity_sgd_v6_retrieval_history"
CONFIG_SCHEMA_VERSION = 4
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


class PathsConfig(FrozenModel):
    model_root_env: str
    svcbench_root_env: str
    hf_home_env: str
    output_root_env: str


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


class InnerSGDConfig(FrozenModel):
    name: str
    learning_rate: PositiveFloat
    momentum: NonNegativeFloat
    weight_decay: NonNegativeFloat
    steps_per_chunk: PositiveInt
    grad_clip_norm: PositiveFloat
    reset_per_video: bool
    meta_gradient_mode: str


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
    optimizer: InnerSGDConfig


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


class PredictorConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    output_dim: PositiveInt
    layer_norm_eps: PositiveFloat
    activation: str
    linear_bias: bool
    parameter_count: PositiveInt


class OfficialWeakBalanceMode(StrEnum):
    """Composition policy for labeled official-weak outer losses."""

    LEGACY_SUM = "legacy_sum"
    INSTANT_EQUAL = "instant_equal"
    EMA_ANSWER_REF = "ema_answer_ref"


class OfficialWeakBalanceConfig(FrozenModel):
    """Answer-dominant composition for the four official-weak terms."""

    mode: OfficialWeakBalanceMode
    group_weight: Probability
    scale_min: PositiveFloat
    scale_max: PositiveFloat
    epsilon: PositiveFloat
    ema_beta: Probability = 0.99

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scale_bounds(self) -> Self:
        if self.scale_min > self.scale_max:
            raise ValueError("official-weak scale_min cannot exceed scale_max")
        return self


class LossConfig(FrozenModel):
    pred_weight: NonNegativeFloat
    identity_weight: NonNegativeFloat
    event_weight: NonNegativeFloat
    o1_unlabeled_weight: NonNegativeFloat
    operator_weight: NonNegativeFloat
    retrieval_weight: NonNegativeFloat
    time_weight: NonNegativeFloat
    auxiliary_outer_weight: NonNegativeFloat
    answer_causal_shift: bool
    answer_ignore_index: int
    official_weak_balance: OfficialWeakBalanceConfig


class StageAVariant(StrEnum):
    """P15 ablations that are legal before any Meta-TTT episode."""

    A1 = "a1"
    A2 = "a2"


class StageAOptimizerConfig(FrozenModel):
    name: str
    qwen_learning_rate: PositiveFloat
    state_learning_rate: PositiveFloat
    w0_learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    betas: tuple[Probability, Probability]
    epsilon: PositiveFloat
    grad_clip_norm: PositiveFloat


class StageACheckpointConfig(FrozenModel):
    format: str
    trainable_only: bool
    include_optimizer: bool
    include_scheduler: bool
    include_rng: bool
    save_full_model: bool
    save_runtime_state: bool
    save_every_epochs: PositiveInt
    selection_policy: str


class StageATrainingConfig(FrozenModel):
    """Production full-unfreeze A2 before the direct A5 transition."""

    variant: StageAVariant
    inner_sgd_enabled: bool
    fast_adapter_mode: str
    qwen_strategy: str
    qwen_parameter_allowlist: tuple[str, ...]
    trainable_components: tuple[str, ...]
    predictor_trainable: bool
    epochs: PositiveInt
    per_device_train_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    world_size: PositiveInt
    global_batch_size: PositiveInt
    loss_terms: tuple[str, ...]
    supervision_provenance: str
    load_best_model_at_end: bool
    balanced_task_sampling: bool
    synthetic_engineering_gate_only: bool
    seed: NonNegativeInt
    optimizer: StageAOptimizerConfig
    checkpoint: StageACheckpointConfig

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_production_a2(self) -> Self:
        if self.variant is not StageAVariant.A2:
            raise ValueError("stage_a.variant must be a2 in the production path")
        if self.inner_sgd_enabled:
            raise ValueError("stage_a.inner_sgd_enabled must be false during A2")
        if self.qwen_strategy != "full_unfreeze_qwen3_vl_8b":
            raise ValueError("production A2 must fully unfreeze Qwen3-VL-8B")
        if self.predictor_trainable:
            raise ValueError("Predictor remains frozen during A2")
        if self.epochs != 8 or self.load_best_model_at_end:
            raise ValueError("production A2 uses eight epochs and the final checkpoint")
        if self.global_batch_size != (
            self.per_device_train_batch_size * self.gradient_accumulation_steps * self.world_size
        ):
            raise ValueError("A2 global batch must equal per-device batch * GA * world size")
        if self.loss_terms != ("state", "answer"):
            raise ValueError("A2 loss must contain only state and answer")
        if self.supervision_provenance != "official_weak":
            raise ValueError("production A2 requires official_weak state supervision")
        if self.synthetic_engineering_gate_only:
            raise ValueError("production A2 cannot be marked synthetic-only")
        return self


class MetaTTTVariant(StrEnum):
    """Explicit P16/P17 Meta-TTT ablations; never selected implicitly."""

    A3 = "a3"
    A4 = "a4"
    A5 = "a5"


class StageBTrainingConfig(FrozenModel):
    """Single-Support, prediction-only Meta-TTT engineering contract."""

    variant: MetaTTTVariant
    support_chunks: PositiveInt
    minimum_query_points: PositiveInt
    enabled_ttt_terms: tuple[str, ...]
    inner_sgd_enabled: bool
    update_effect: str
    auxiliary_outer_weight: NonNegativeFloat
    reset_per_episode: bool
    compare_before_after: bool
    meta_gradient_mode: str
    reuse_strategy: str
    synthetic_engineering_gate_only: bool
    seed: NonNegativeInt


class StageCOptimizerConfig(FrozenModel):
    """Non-Qwen A5 parameter-group learning rates owned by State-TTT."""

    state_learning_rate: PositiveFloat
    w0_learning_rate: PositiveFloat
    predictor_learning_rate: PositiveFloat


class StageCTrainingConfig(FrozenModel):
    """Direct A5 contract with unbounded numeric state and bounded meta gradients."""

    active_variant: MetaTTTVariant
    variants: tuple[MetaTTTVariant, ...]
    a4_enabled_ttt_terms: tuple[str, ...]
    a5_enabled_ttt_terms: tuple[str, ...]
    support_chunk_schedule: tuple[PositiveInt, ...]
    maximum_support_chunks: PositiveInt | None
    minimum_query_points: PositiveInt
    multi_query_enabled: bool
    detach_overlap_snapshots: bool
    detach_runtime_between_chunks: bool
    update_effect: str
    reuse_strategy: str
    direct_from_stage_a: bool
    meta_gradient_mode: str
    truncation_horizon: PositiveInt
    reanchor_to_w0: bool
    segment_auxiliary_backward: bool
    training_counterfactual_enabled: bool
    prewarm_support_chunks: NonNegativeInt
    outer_step_scope: str
    synthetic_engineering_gate_only: bool
    seed: NonNegativeInt
    optimizer: StageCOptimizerConfig

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_incremental_variants(self) -> Self:
        if self.variants != (MetaTTTVariant.A4, MetaTTTVariant.A5):
            raise ValueError("stage_c.variants must be exactly ('a4', 'a5')")
        if self.active_variant not in self.variants:
            raise ValueError("stage_c.active_variant must select a4 or a5")
        if self.a4_enabled_ttt_terms != ("pred", "identity"):
            raise ValueError("stage_c.a4_enabled_ttt_terms must contain only pred and identity")
        if self.a5_enabled_ttt_terms != (*self.a4_enabled_ttt_terms, "event"):
            raise ValueError("stage_c.a5_enabled_ttt_terms must differ from A4 only by event")
        if self.direct_from_stage_a and self.active_variant is not MetaTTTVariant.A5:
            raise ValueError("direct Stage A transition must enter A5")
        if self.meta_gradient_mode != "truncated_second_order":
            raise ValueError("Stage C production mode must use truncated_second_order")
        if self.maximum_support_chunks is not None:
            raise ValueError("Stage C production must not cap the numeric Support stream")
        if not self.reanchor_to_w0 or not self.segment_auxiliary_backward:
            raise ValueError("truncated Stage C requires segment backward and W0 re-anchoring")
        if self.prewarm_support_chunks != 1:
            raise ValueError("Stage C requires exactly one no-update prewarm chunk")
        if self.outer_step_scope != "episode":
            raise ValueError("Stage C outer optimizer scope must be one complete episode")
        return self

    @property
    def enabled_ttt_terms(self) -> tuple[str, ...]:
        return (
            self.a4_enabled_ttt_terms
            if self.active_variant is MetaTTTVariant.A4
            else self.a5_enabled_ttt_terms
        )


class InferenceRuntimeConfig(FrozenModel):
    """Managed per-video lifecycle and immutable-generation contract."""

    reset_per_video: bool
    update_effect: str
    prefill_once: bool
    decode_state_immutable: bool
    release_on_exception: bool
    audit_level: AuditLevel
    repeat_query_policy: str
    record_skip_reasons: bool
    synthetic_engineering_gate_only: bool


class EvaluationConfig(FrozenModel):
    formal_evaluation_enabled: bool
    official_clean_tuning_forbidden: bool


class ParameterBudgetConfig(FrozenModel):
    fast_ttt_adapter_millions: PositiveFloat
    online_fast_matrices_millions: PositiveFloat
    spatial_encoder_millions: PositiveFloat
    temporal_encoder_millions: PositiveFloat
    query_encoder_millions: PositiveFloat
    o1_millions: PositiveFloat
    o2_millions: PositiveFloat
    e1_millions: PositiveFloat
    e2_millions: PositiveFloat
    semantic_projector_millions: PositiveFloat
    predictor_millions: PositiveFloat
    state_resampler_millions: PositiveFloat
    router_resolver_empty_millions: PositiveFloat
    new_modules_total_millions: PositiveFloat
    rounding_tolerance_millions: PositiveFloat


class ProjectConfig(FrozenModel):
    """Complete v5 configuration with cross-component contract validation."""

    spec_version: str
    config_schema_version: int
    paths: PathsConfig
    data: DataConfig
    video_preprocessing: VideoPreprocessingConfig
    model: ModelConfig
    fast_ttt: FastTTTConfig
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
    predictor: PredictorConfig
    loss: LossConfig
    stage_a: StageATrainingConfig
    stage_b: StageBTrainingConfig
    stage_c: StageCTrainingConfig
    inference: InferenceRuntimeConfig
    evaluation: EvaluationConfig
    parameter_budget: ParameterBudgetConfig

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_v5_contract(self) -> Self:
        checks: tuple[tuple[str, object, object], ...] = (
            ("spec_version", self.spec_version, SPEC_VERSION),
            ("config_schema_version", self.config_schema_version, CONFIG_SCHEMA_VERSION),
            ("paths.model_root_env", self.paths.model_root_env, "QWEN_MODEL_ROOT"),
            ("paths.svcbench_root_env", self.paths.svcbench_root_env, "SVCBENCH_ROOT"),
            ("paths.hf_home_env", self.paths.hf_home_env, "HF_HOME"),
            ("paths.output_root_env", self.paths.output_root_env, "OUTPUT_ROOT"),
            (
                "data.grouped_annotation_file",
                self.data.grouped_annotation_file,
                "data/vcbench_data.jsonl",
            ),
            (
                "data.flat_annotation_file",
                self.data.flat_annotation_file,
                "data/vcbench_eval.jsonl",
            ),
            ("data.video_directory", self.data.video_directory, "data/videos"),
            ("data.group_key_fields", self.data.group_key_fields, ("source_dataset", "video_path")),
            ("data.group_k_folds", self.data.group_k_folds, 5),
            ("data.fold_seed", self.data.fold_seed, 42),
            (
                "data.runtime_allowlist",
                self.data.runtime_allowlist,
                ("video", "question", "query_time", "explicit_time_values"),
            ),
            (
                "data.runtime_denylist",
                self.data.runtime_denylist,
                ("answer", "count", "occurrence_times", "counting_type", "counting_subtype"),
            ),
            (
                "data.official_clean_selection_forbidden",
                self.data.official_clean_selection_forbidden,
                True,
            ),
            ("video_preprocessing.sample_fps", self.video_preprocessing.sample_fps, 2.0),
            ("video_preprocessing.frames_per_chunk", self.video_preprocessing.frames_per_chunk, 16),
            ("video_preprocessing.stride_frames", self.video_preprocessing.stride_frames, 8),
            (
                "video_preprocessing.causal_boundary",
                self.video_preprocessing.causal_boundary,
                "right_closed",
            ),
            (
                "video_preprocessing.processor_shortest_edge",
                self.video_preprocessing.processor_shortest_edge,
                4096,
            ),
            (
                "video_preprocessing.processor_longest_edge",
                self.video_preprocessing.processor_longest_edge,
                25_165_824,
            ),
            ("video_preprocessing.patch_size", self.video_preprocessing.patch_size, 16),
            (
                "video_preprocessing.temporal_patch_size",
                self.video_preprocessing.temporal_patch_size,
                2,
            ),
            (
                "video_preprocessing.spatial_merge_size",
                self.video_preprocessing.spatial_merge_size,
                2,
            ),
            ("video_preprocessing.pad_value", self.video_preprocessing.pad_value, 0.0),
            (
                "video_preprocessing.full_tubelet_required_for_state",
                self.video_preprocessing.full_tubelet_required_for_state,
                True,
            ),
            ("model.base_model", self.model.base_model, BASE_MODEL_ID),
            ("model.revision", self.model.revision, BASE_MODEL_REVISION),
            ("model.transformers_version", self.model.transformers_version, TRANSFORMERS_VERSION),
            ("model.vision.depth", self.model.vision.depth, 27),
            ("model.vision.hidden_size", self.model.vision.hidden_size, 1152),
            ("model.vision.num_heads", self.model.vision.num_heads, 16),
            ("model.vision.patch_size", self.model.vision.patch_size, 16),
            ("model.vision.temporal_patch_size", self.model.vision.temporal_patch_size, 2),
            ("model.vision.spatial_merge_size", self.model.vision.spatial_merge_size, 2),
            ("model.vision.output_size", self.model.vision.output_size, 4096),
            (
                "model.vision.deepstack_visual_indexes",
                self.model.vision.deepstack_visual_indexes,
                (8, 16, 24),
            ),
            ("model.llm.num_layers", self.model.llm.num_layers, 36),
            ("model.llm.hidden_size", self.model.llm.hidden_size, 4096),
            ("model.online_freeze.vision", self.model.online_freeze.vision, True),
            ("model.online_freeze.merger", self.model.online_freeze.merger, True),
            ("model.online_freeze.deepstack", self.model.online_freeze.deepstack, True),
            ("model.online_freeze.llm", self.model.online_freeze.llm, True),
            ("fast_ttt.input_dim", self.fast_ttt.input_dim, 4096),
            ("fast_ttt.bottleneck_dim", self.fast_ttt.bottleneck_dim, 768),
            ("fast_ttt.output_dim", self.fast_ttt.output_dim, 4096),
            ("fast_ttt.residual_scale", self.fast_ttt.residual_scale, 0.1),
            ("fast_ttt.rms_norm_eps", self.fast_ttt.rms_norm_eps, 1.0e-6),
            ("fast_ttt.slow_projection_bias", self.fast_ttt.slow_projection_bias, True),
            ("fast_ttt.fast_bias", self.fast_ttt.fast_bias, False),
            ("fast_ttt.fast_initialization", self.fast_ttt.fast_initialization, "xavier_uniform"),
            ("fast_ttt.fast_matrix_count", self.fast_ttt.fast_matrix_count, 2),
            ("fast_ttt.online_parameter_count", self.fast_ttt.online_parameter_count, 1_179_648),
            (
                "fast_ttt.update_order",
                self.fast_ttt.update_order,
                "observe_state_then_update_for_next_chunk",
            ),
            ("fast_ttt.optimizer.name", self.fast_ttt.optimizer.name, "sgd"),
            ("fast_ttt.optimizer.learning_rate", self.fast_ttt.optimizer.learning_rate, 1.0e-4),
            ("fast_ttt.optimizer.momentum", self.fast_ttt.optimizer.momentum, 0.0),
            ("fast_ttt.optimizer.weight_decay", self.fast_ttt.optimizer.weight_decay, 0.0),
            ("fast_ttt.optimizer.steps_per_chunk", self.fast_ttt.optimizer.steps_per_chunk, 1),
            ("fast_ttt.optimizer.grad_clip_norm", self.fast_ttt.optimizer.grad_clip_norm, 1.0),
            ("fast_ttt.optimizer.reset_per_video", self.fast_ttt.optimizer.reset_per_video, True),
            (
                "fast_ttt.optimizer.meta_gradient_mode",
                self.fast_ttt.optimizer.meta_gradient_mode,
                "full_second_order",
            ),
            ("spatial_encoder.input_dim", self.spatial_encoder.input_dim, 4096),
            ("spatial_encoder.hidden_dim", self.spatial_encoder.hidden_dim, 768),
            ("spatial_encoder.stages", self.spatial_encoder.stages, 2),
            ("spatial_encoder.num_heads", self.spatial_encoder.num_heads, 12),
            ("spatial_encoder.head_dim", self.spatial_encoder.head_dim, 64),
            (
                "spatial_encoder.refinements_per_stage",
                self.spatial_encoder.refinements_per_stage,
                3,
            ),
            ("spatial_encoder.ffn_dim", self.spatial_encoder.ffn_dim, 3072),
            ("spatial_encoder.active_slots", self.spatial_encoder.active_slots, 32),
            ("spatial_encoder.max_active_slots", self.spatial_encoder.max_active_slots, 64),
            ("spatial_encoder.query_dim", self.spatial_encoder.query_dim, 512),
            ("spatial_encoder.layer_norm_eps", self.spatial_encoder.layer_norm_eps, 1.0e-5),
            (
                "spatial_encoder.slot_initialization",
                self.spatial_encoder.slot_initialization,
                "shared_seed_plus_fixed_sinusoidal_codes",
            ),
            (
                "spatial_encoder.attention_normalization",
                self.spatial_encoder.attention_normalization,
                "softmax_slots_then_normalize_tokens",
            ),
            (
                "spatial_encoder.attention_epsilon",
                self.spatial_encoder.attention_epsilon,
                1.0e-8,
            ),
            (
                "spatial_encoder.confidence_mode",
                self.spatial_encoder.confidence_mode,
                "attention_occupancy",
            ),
            (
                "spatial_encoder.overflow_policy",
                self.spatial_encoder.overflow_policy,
                "preserve_existing_reject_excess",
            ),
            ("spatial_encoder.slot_valid_mask", self.spatial_encoder.slot_valid_mask, True),
            ("spatial_encoder.log_overflow", self.spatial_encoder.log_overflow, True),
            ("temporal_encoder.input_dim", self.temporal_encoder.input_dim, 4096),
            ("temporal_encoder.hidden_dim", self.temporal_encoder.hidden_dim, 768),
            ("temporal_encoder.num_layers", self.temporal_encoder.num_layers, 6),
            ("temporal_encoder.num_heads", self.temporal_encoder.num_heads, 12),
            ("temporal_encoder.head_dim", self.temporal_encoder.head_dim, 64),
            ("temporal_encoder.ffn_dim", self.temporal_encoder.ffn_dim, 3072),
            ("temporal_encoder.dropout", self.temporal_encoder.dropout, 0.1),
            (
                "temporal_encoder.position_encoding",
                self.temporal_encoder.position_encoding,
                "absolute_sinusoidal",
            ),
            (
                "temporal_encoder.layer_norm_eps",
                self.temporal_encoder.layer_norm_eps,
                1.0e-5,
            ),
            ("temporal_encoder.activation", self.temporal_encoder.activation, "gelu"),
            ("temporal_encoder.pre_norm", self.temporal_encoder.pre_norm, True),
            (
                "temporal_encoder.attention_projection_bias",
                self.temporal_encoder.attention_projection_bias,
                True,
            ),
            ("temporal_encoder.strict_causal", self.temporal_encoder.strict_causal, True),
            (
                "temporal_encoder.causal_includes_self",
                self.temporal_encoder.causal_includes_self,
                True,
            ),
            (
                "temporal_encoder.causal_window_includes_current",
                self.temporal_encoder.causal_window_includes_current,
                True,
            ),
            ("temporal_encoder.cache_tubelets", self.temporal_encoder.cache_tubelets, 64),
            (
                "temporal_encoder.cache_mode",
                self.temporal_encoder.cache_mode,
                "layerwise_kv",
            ),
            (
                "temporal_encoder.position_id_mode",
                self.temporal_encoder.position_id_mode,
                "explicit_global",
            ),
            (
                "temporal_encoder.overlap_policy",
                self.temporal_encoder.overlap_policy,
                "replay_replace",
            ),
            (
                "temporal_encoder.overlap_tubelets",
                self.temporal_encoder.overlap_tubelets,
                4,
            ),
            (
                "temporal_encoder.replay_context_tubelets",
                self.temporal_encoder.replay_context_tubelets,
                3,
            ),
            (
                "temporal_encoder.cache_owner_keys",
                self.temporal_encoder.cache_owner_keys,
                ("video_id", "trajectory_id", "query_signature"),
            ),
            (
                "temporal_encoder.detach_cache_default",
                self.temporal_encoder.detach_cache_default,
                True,
            ),
            ("temporal_encoder.query_dim", self.temporal_encoder.query_dim, 512),
            (
                "temporal_encoder.parameter_count",
                self.temporal_encoder.parameter_count,
                48_438_272,
            ),
            ("state_bank.semantic_dim", self.state_bank.semantic_dim, 512),
            ("state_bank.identity_dim", self.state_bank.identity_dim, 256),
            (
                "state_bank.semantic_projector.input_dim",
                self.state_bank.semantic_projector.input_dim,
                768,
            ),
            (
                "state_bank.semantic_projector.hidden_dim",
                self.state_bank.semantic_projector.hidden_dim,
                1024,
            ),
            (
                "state_bank.semantic_projector.output_dim",
                self.state_bank.semantic_projector.output_dim,
                512,
            ),
            (
                "state_bank.semantic_projector.head_types",
                self.state_bank.semantic_projector.head_types,
                ("o1", "o2", "e1", "e2"),
            ),
            (
                "state_bank.semantic_projector.head_type_count",
                self.state_bank.semantic_projector.head_type_count,
                4,
            ),
            (
                "state_bank.semantic_projector.layer_norm_eps",
                self.state_bank.semantic_projector.layer_norm_eps,
                1.0e-5,
            ),
            (
                "state_bank.semantic_projector.activation",
                self.state_bank.semantic_projector.activation,
                "silu",
            ),
            (
                "state_bank.semantic_projector.dropout",
                self.state_bank.semantic_projector.dropout,
                0.0,
            ),
            (
                "state_bank.semantic_projector.linear_bias",
                self.state_bank.semantic_projector.linear_bias,
                True,
            ),
            (
                "state_bank.semantic_projector.normalization_dtype",
                self.state_bank.semantic_projector.normalization_dtype,
                "float32",
            ),
            (
                "state_bank.semantic_projector.normalization_eps",
                self.state_bank.semantic_projector.normalization_eps,
                1.0e-8,
            ),
            (
                "state_bank.semantic_projector.zero_norm_fallback",
                self.state_bank.semantic_projector.zero_norm_fallback,
                "first_unit_basis",
            ),
            (
                "state_bank.semantic_projector.parameter_count",
                self.state_bank.semantic_projector.parameter_count,
                1_316_864,
            ),
            (
                "state_bank.semantic_projector.included_in_model_state_dict",
                self.state_bank.semantic_projector.included_in_model_state_dict,
                True,
            ),
            (
                "state_bank.semantic_projector.included_in_outer_optimizer",
                self.state_bank.semantic_projector.included_in_outer_optimizer,
                True,
            ),
            (
                "state_bank.semantic_projector.included_in_inner_optimizer",
                self.state_bank.semantic_projector.included_in_inner_optimizer,
                False,
            ),
            (
                "state_bank.semantic_projector.online_frozen",
                self.state_bank.semantic_projector.online_frozen,
                True,
            ),
            (
                "state_bank.semantic_projector.online_forward_no_grad",
                self.state_bank.semantic_projector.online_forward_no_grad,
                False,
            ),
            (
                "state_bank.semantic_projector.detach_inputs",
                self.state_bank.semantic_projector.detach_inputs,
                False,
            ),
            (
                "state_bank.confirmed_store.initial_capacity",
                self.state_bank.confirmed_store.initial_capacity,
                256,
            ),
            (
                "state_bank.confirmed_store.growth_chunk",
                self.state_bank.confirmed_store.growth_chunk,
                256,
            ),
            (
                "state_bank.confirmed_store.hard_limit",
                self.state_bank.confirmed_store.hard_limit,
                None,
            ),
            (
                "state_bank.confirmed_store.storage_device",
                self.state_bank.confirmed_store.storage_device,
                "cpu",
            ),
            (
                "state_bank.confirmed_store.storage_dtype",
                self.state_bank.confirmed_store.storage_dtype,
                "float32",
            ),
            (
                "state_bank.confirmed_store.exact_search",
                self.state_bank.confirmed_store.exact_search,
                True,
            ),
            (
                "state_bank.confirmed_store.ann_enabled",
                self.state_bank.confirmed_store.ann_enabled,
                False,
            ),
            (
                "state_bank.confirmed_store.gpu_hot_capacity",
                self.state_bank.confirmed_store.gpu_hot_capacity,
                256,
            ),
            (
                "state_bank.confirmed_store.hot_cache_enabled",
                self.state_bank.confirmed_store.hot_cache_enabled,
                True,
            ),
            (
                "state_bank.confirmed_store.hot_cache_device",
                self.state_bank.confirmed_store.hot_cache_device,
                "cuda",
            ),
            (
                "state_bank.confirmed_store.hot_cache_dtype",
                self.state_bank.confirmed_store.hot_cache_dtype,
                "bfloat16",
            ),
            (
                "state_bank.confirmed_store.eviction_policy",
                self.state_bank.confirmed_store.eviction_policy,
                "lru_position_then_identity_id",
            ),
            (
                "state_bank.candidate_store.initial_capacity",
                self.state_bank.candidate_store.initial_capacity,
                64,
            ),
            (
                "state_bank.candidate_store.growth_chunk",
                self.state_bank.candidate_store.growth_chunk,
                64,
            ),
            (
                "state_bank.candidate_store.hard_limit",
                self.state_bank.candidate_store.hard_limit,
                512,
            ),
            (
                "state_bank.candidate_store.ttl_chunks",
                self.state_bank.candidate_store.ttl_chunks,
                8,
            ),
            (
                "state_bank.candidate_store.match_threshold",
                self.state_bank.candidate_store.match_threshold,
                0.8,
            ),
            (
                "state_bank.candidate_store.reliability_threshold",
                self.state_bank.candidate_store.reliability_threshold,
                0.5,
            ),
            (
                "state_bank.candidate_store.low_confidence_threshold",
                self.state_bank.candidate_store.low_confidence_threshold,
                0.5,
            ),
            (
                "state_bank.candidate_store.ttl_refresh_policy",
                self.state_bank.candidate_store.ttl_refresh_policy,
                "reset_to_full_on_reliable_match",
            ),
            (
                "state_bank.candidate_store.ttl_aging_policy",
                self.state_bank.candidate_store.ttl_aging_policy,
                "match_first_then_unmatched_decrement_once_per_new_committed_position_remove_at_zero_end",
            ),
            (
                "state_bank.candidate_store.promotion_policy",
                self.state_bank.candidate_store.promotion_policy,
                "two_reliable_distinct_consecutive_committed_positions",
            ),
            (
                "state_bank.candidate_store.overflow_policy",
                self.state_bank.candidate_store.overflow_policy,
                "expire_then_low_confidence_then_reject",
            ),
            (
                "state_bank.candidate_store.prune_order",
                self.state_bank.candidate_store.prune_order,
                (
                    "expired",
                    "low_confidence",
                    "confidence_asc",
                    "last_position_id_asc",
                    "candidate_id_asc",
                    "reject_new",
                ),
            ),
            ("state_bank.event_history_capacity", self.state_bank.event_history_capacity, 512),
            (
                "state_bank.retrieval_history_capacity_per_head",
                self.state_bank.retrieval_history_capacity_per_head,
                512,
            ),
            (
                "state_bank.retrieval_history_source_dim",
                self.state_bank.retrieval_history_source_dim,
                768,
            ),
            ("state_bank.hard_updates_no_grad", self.state_bank.hard_updates_no_grad, True),
            ("state_bank.detach_before_write", self.state_bank.detach_before_write, True),
            (
                "state_bank.runtime_in_model_state_dict",
                self.state_bank.runtime_in_model_state_dict,
                False,
            ),
            (
                "state_bank.runtime_registered_parameters",
                self.state_bank.runtime_registered_parameters,
                False,
            ),
            (
                "state_bank.runtime_registered_buffers",
                self.state_bank.runtime_registered_buffers,
                False,
            ),
            (
                "state_bank.runtime_in_outer_optimizer",
                self.state_bank.runtime_in_outer_optimizer,
                False,
            ),
            (
                "state_bank.runtime_in_inner_optimizer",
                self.state_bank.runtime_in_inner_optimizer,
                False,
            ),
            (
                "state_bank.snapshot_separate_from_model_checkpoint",
                self.state_bank.snapshot_separate_from_model_checkpoint,
                True,
            ),
            (
                "state_bank.record_time_metadata_policy",
                self.state_bank.record_time_metadata_policy,
                "exactly_one",
            ),
            (
                "state_bank.record_id_policy",
                self.state_bank.record_id_policy,
                "trajectory_monotonic_never_reuse",
            ),
            (
                "state_bank.aggregate_record_heads",
                self.state_bank.aggregate_record_heads,
                ("o1", "e1", "e2"),
            ),
            (
                "state_bank.aggregate_update_mode",
                self.state_bank.aggregate_update_mode,
                "functional_replace",
            ),
            (
                "state_bank.committed_position_policy",
                self.state_bank.committed_position_policy,
                "idempotent_ignore_and_audit",
            ),
            (
                "state_bank.o2_p9_policy",
                self.state_bank.o2_p9_policy,
                "generic_crud_only_p10_owns_lifecycle",
            ),
            (
                "state_bank.o2_lifecycle_owner",
                self.state_bank.o2_lifecycle_owner,
                "identity_bank_p10",
            ),
            (
                "state_bank.o2_candidate_retrieval_eligible",
                self.state_bank.o2_candidate_retrieval_eligible,
                False,
            ),
            (
                "state_bank.o2_confirmed_retrieval_eligible",
                self.state_bank.o2_confirmed_retrieval_eligible,
                True,
            ),
            (
                "state_bank.dynamic_view_padding",
                self.state_bank.dynamic_view_padding,
                "batch_max",
            ),
            (
                "state_bank.n_state_definition",
                self.state_bank.n_state_definition,
                "owner_head_present_records_before_filters",
            ),
            (
                "state_bank.event_kind_provenance",
                self.state_bank.event_kind_provenance,
                "hard_operator_frozen_per_aggregate",
            ),
            ("query_encoder.input_dim", self.query_encoder.input_dim, 4096),
            ("query_encoder.hidden_dim", self.query_encoder.hidden_dim, 768),
            ("query_encoder.num_layers", self.query_encoder.num_layers, 4),
            ("query_encoder.num_heads", self.query_encoder.num_heads, 12),
            ("query_encoder.head_dim", self.query_encoder.head_dim, 64),
            ("query_encoder.ffn_dim", self.query_encoder.ffn_dim, 3072),
            ("query_encoder.dropout", self.query_encoder.dropout, 0.1),
            ("query_encoder.output_dim", self.query_encoder.output_dim, 512),
            ("query_encoder.bidirectional", self.query_encoder.bidirectional, True),
            (
                "query_encoder.position_encoding",
                self.query_encoder.position_encoding,
                "sinusoidal",
            ),
            ("query_encoder.pooling", self.query_encoder.pooling, "learned_attention"),
            ("operator_router.output_dim", self.operator_router.output_dim, 512),
            (
                "operator_router.temperature_initial",
                self.operator_router.temperature_initial,
                1.0,
            ),
            (
                "operator_router.temperature_trainable",
                self.operator_router.temperature_trainable,
                True,
            ),
            (
                "operator_router.confidence_threshold",
                self.operator_router.confidence_threshold,
                None,
            ),
            (
                "operator_router.threshold_status",
                self.operator_router.threshold_status,
                CalibrationStatus.CALIBRATION_REQUIRED,
            ),
            ("time_resolver.input_dim", self.time_resolver.input_dim, 512),
            ("time_resolver.hidden_dim", self.time_resolver.hidden_dim, 256),
            ("time_resolver.mode_count", self.time_resolver.mode_count, 4),
            ("time_resolver.token_hidden_dim", self.time_resolver.token_hidden_dim, 768),
            ("time_resolver.pointer_heads", self.time_resolver.pointer_heads, 2),
            (
                "time_resolver.confidence_threshold",
                self.time_resolver.confidence_threshold,
                None,
            ),
            (
                "time_resolver.threshold_status",
                self.time_resolver.threshold_status,
                CalibrationStatus.CALIBRATION_REQUIRED,
            ),
            ("retriever.semantic_dim", self.retriever.semantic_dim, 512),
            (
                "retriever.record_similarity_threshold",
                self.retriever.record_similarity_threshold,
                0.35,
            ),
            ("retriever.similarity_dtype", self.retriever.similarity_dtype, "float32"),
            ("retriever.normalization_eps", self.retriever.normalization_eps, 1.0e-8),
            ("retriever.zero_query_policy", self.retriever.zero_query_policy, "unsupported"),
            (
                "retriever.threshold_comparison",
                self.retriever.threshold_comparison,
                "greater_than_or_equal",
            ),
            (
                "retriever.record_confidence_threshold",
                self.retriever.record_confidence_threshold,
                None,
            ),
            (
                "retriever.filter_order",
                self.retriever.filter_order,
                (
                    "invalid",
                    "retrieval_ineligible",
                    "future",
                    "outside_window",
                    "below_similarity",
                ),
            ),
            (
                "retriever.selection_order",
                self.retriever.selection_order,
                ("score_desc", "record_id_asc"),
            ),
            (
                "retriever.owner_mismatch_status",
                self.retriever.owner_mismatch_status,
                "invalid",
            ),
            (
                "retriever.aggregate_time_policy",
                self.retriever.aggregate_time_policy,
                "causal_availability_only_window_in_reader",
            ),
            (
                "retriever.atomic_window_boundary",
                self.retriever.atomic_window_boundary,
                "closed",
            ),
            (
                "retriever.metrics_policy",
                self.retriever.metrics_policy,
                "offline_ground_truth_runtime_label_free",
            ),
            ("retriever.top_k", self.retriever.top_k, None),
            ("retriever.ann_enabled", self.retriever.ann_enabled, False),
            ("state_resampler.num_queries", self.state_resampler.num_queries, 16),
            ("state_resampler.num_layers", self.state_resampler.num_layers, 3),
            ("state_resampler.num_heads", self.state_resampler.num_heads, 8),
            ("state_resampler.head_dim", self.state_resampler.head_dim, 64),
            ("state_resampler.ffn_dim", self.state_resampler.ffn_dim, 2048),
            ("state_resampler.hidden_dim", self.state_resampler.hidden_dim, 512),
            ("state_resampler.output_dim", self.state_resampler.output_dim, 4096),
            ("state_resampler.layer_norm_eps", self.state_resampler.layer_norm_eps, 1.0e-5),
            ("state_resampler.activation", self.state_resampler.activation, "gelu"),
            ("state_resampler.dropout", self.state_resampler.dropout, 0.0),
            ("state_resampler.attention_bias", self.state_resampler.attention_bias, True),
            (
                "state_resampler.output_projection_bias",
                self.state_resampler.output_projection_bias,
                True,
            ),
            (
                "state_resampler.attention_softmax_dtype",
                self.state_resampler.attention_softmax_dtype,
                "float32",
            ),
            (
                "state_resampler.empty_record_embedding",
                self.state_resampler.empty_record_embedding,
                True,
            ),
            (
                "state_resampler.empty_record_policy",
                self.state_resampler.empty_record_policy,
                "internal_trainable_kv_external_zero_width",
            ),
            (
                "state_resampler.attention_audit",
                self.state_resampler.attention_audit,
                "final_layer_mean_heads_selected_mass",
            ),
            ("state_resampler.parameter_count", self.state_resampler.parameter_count, 14_722_048),
            ("state_reader.signed_exact_count", self.state_reader.signed_exact_count, True),
            ("state_reader.empty_exact_count", self.state_reader.empty_exact_count, 0),
            (
                "state_reader.status_propagation",
                self.state_reader.status_propagation,
                "retriever_exact",
            ),
            (
                "state_reader.o1_delta_policy",
                self.state_reader.o1_delta_policy,
                "fixed_baseline_v1",
            ),
            ("state_reader.o2_identity_key", self.state_reader.o2_identity_key, "identity_id"),
            (
                "state_reader.point_window_boundary",
                self.state_reader.point_window_boundary,
                "closed",
            ),
            (
                "state_reader.e1_history_policy",
                self.state_reader.e1_history_policy,
                "cumulative_or_retained_completion_times",
            ),
            (
                "state_reader.e1_truncated_window_status",
                self.state_reader.e1_truncated_window_status,
                "invalid",
            ),
            (
                "state_reader.e2_window_anchor",
                self.state_reader.e2_window_anchor,
                "completion_end",
            ),
            (
                "state_reader.event_kind_mismatch_status",
                self.state_reader.event_kind_mismatch_status,
                "invalid",
            ),
            (
                "state_reader.number_text_format",
                self.state_reader.number_text_format,
                "canonical_ascii_signed_decimal",
            ),
            (
                "state_reader.tokenizer_add_special_tokens",
                self.state_reader.tokenizer_add_special_tokens,
                False,
            ),
            (
                "state_reader.tokenizer_roundtrip_required",
                self.state_reader.tokenizer_roundtrip_required,
                True,
            ),
            (
                "state_reader.tokenizer_class",
                self.state_reader.tokenizer_class,
                "Qwen2TokenizerFast",
            ),
            (
                "state_reader.tokenizer_vocab_size",
                self.state_reader.tokenizer_vocab_size,
                151_643,
            ),
            (
                "state_reader.tokenizer_required_files",
                self.state_reader.tokenizer_required_files,
                ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
            ),
            (
                "state_reader.tokenizer_manifest_sha256",
                self.state_reader.tokenizer_manifest_sha256,
                "ccd18347b6d6714d91d4c55b37ff05e473a0f8e84fbcba2bda1401a9572f44c3",
            ),
            (
                "state_reader.ground_truth_input_forbidden",
                self.state_reader.ground_truth_input_forbidden,
                True,
            ),
            ("input_composer.state_token_count", self.input_composer.state_token_count, 16),
            (
                "input_composer.special_tokens",
                self.input_composer.special_tokens,
                (
                    "<|state_start|>",
                    "<|state_pad|>",
                    "<|state_end|>",
                    "<|number_start|>",
                    "<|number_end|>",
                ),
            ),
            (
                "input_composer.special_token_ids",
                self.input_composer.special_token_ids,
                (151_669, 151_670, 151_671, 151_672, 151_673),
            ),
            (
                "input_composer.tokenizer_base_length",
                self.input_composer.tokenizer_base_length,
                151_669,
            ),
            (
                "input_composer.tokenizer_extended_length",
                self.input_composer.tokenizer_extended_length,
                151_674,
            ),
            (
                "input_composer.tokenizer_revision",
                self.input_composer.tokenizer_revision,
                BASE_MODEL_REVISION,
            ),
            (
                "input_composer.model_embedding_rows",
                self.input_composer.model_embedding_rows,
                151_936,
            ),
            (
                "input_composer.embedding_initialization",
                self.input_composer.embedding_initialization,
                "fp32_mean_of_vision_start_video_pad_vision_end_then_cast",
            ),
            (
                "input_composer.embedding_source_token_ids",
                self.input_composer.embedding_source_token_ids,
                (151_652, 151_656, 151_653),
            ),
            (
                "input_composer.initialize_input_and_output_embeddings",
                self.input_composer.initialize_input_and_output_embeddings,
                True,
            ),
            ("input_composer.padding_side", self.input_composer.padding_side, "left"),
            (
                "input_composer.state_payload_statuses",
                self.input_composer.state_payload_statuses,
                ("ok", "empty"),
            ),
            (
                "input_composer.invalid_payload_policy",
                self.input_composer.invalid_payload_policy,
                "omit_state_and_number",
            ),
            (
                "input_composer.payload_order",
                self.input_composer.payload_order,
                (
                    "system_user_question_video",
                    "state",
                    "number",
                    "user_end",
                    "assistant_generation_prefix",
                ),
            ),
            ("input_composer.prefill_once", self.input_composer.prefill_once, True),
            (
                "input_composer.generation_num_beams",
                self.input_composer.generation_num_beams,
                1,
            ),
            ("predictor.input_dim", self.predictor.input_dim, 768),
            ("predictor.hidden_dim", self.predictor.hidden_dim, 1536),
            ("predictor.output_dim", self.predictor.output_dim, 768),
            ("predictor.layer_norm_eps", self.predictor.layer_norm_eps, 1.0e-5),
            ("predictor.activation", self.predictor.activation, "silu"),
            ("predictor.linear_bias", self.predictor.linear_bias, True),
            ("predictor.parameter_count", self.predictor.parameter_count, 2_363_136),
            ("loss.pred_weight", self.loss.pred_weight, 1.0),
            ("loss.identity_weight", self.loss.identity_weight, 0.5),
            ("loss.event_weight", self.loss.event_weight, 0.5),
            ("loss.o1_unlabeled_weight", self.loss.o1_unlabeled_weight, 0.0),
            ("loss.operator_weight", self.loss.operator_weight, 1.0),
            ("loss.retrieval_weight", self.loss.retrieval_weight, 1.0),
            ("loss.time_weight", self.loss.time_weight, 1.0),
            ("loss.auxiliary_outer_weight", self.loss.auxiliary_outer_weight, 0.1),
            ("loss.answer_causal_shift", self.loss.answer_causal_shift, True),
            ("loss.answer_ignore_index", self.loss.answer_ignore_index, -100),
            (
                "loss.official_weak_balance.group_weight",
                self.loss.official_weak_balance.group_weight,
                0.3,
            ),
            (
                "loss.official_weak_balance.scale_min",
                self.loss.official_weak_balance.scale_min,
                (
                    0.001
                    if self.loss.official_weak_balance.mode
                    is OfficialWeakBalanceMode.EMA_ANSWER_REF
                    else 0.1
                ),
            ),
            (
                "loss.official_weak_balance.scale_max",
                self.loss.official_weak_balance.scale_max,
                (
                    20.0
                    if self.loss.official_weak_balance.mode
                    is OfficialWeakBalanceMode.EMA_ANSWER_REF
                    else 10.0
                ),
            ),
            (
                "loss.official_weak_balance.epsilon",
                self.loss.official_weak_balance.epsilon,
                1.0e-8,
            ),
            (
                "loss.official_weak_balance.ema_beta",
                self.loss.official_weak_balance.ema_beta,
                0.99,
            ),
            ("stage_a.variant", self.stage_a.variant, StageAVariant.A2),
            ("stage_a.inner_sgd_enabled", self.stage_a.inner_sgd_enabled, False),
            (
                "stage_a.fast_adapter_mode",
                self.stage_a.fast_adapter_mode,
                "static_w0_no_inner_sgd",
            ),
            (
                "stage_a.qwen_strategy",
                self.stage_a.qwen_strategy,
                "full_unfreeze_qwen3_vl_8b",
            ),
            ("stage_a.qwen_parameter_allowlist", self.stage_a.qwen_parameter_allowlist, ()),
            (
                "stage_a.trainable_components",
                self.stage_a.trainable_components,
                (
                    "qwen_vit",
                    "qwen_main_merger",
                    "qwen_deepstack_mergers",
                    "qwen_decoder_36",
                    "fast_adapter_w0",
                    "query_encoder",
                    "spatial_encoder",
                    "temporal_encoder",
                    "observation_heads",
                    "state_bank",
                    "resampler",
                ),
            ),
            ("stage_a.predictor_trainable", self.stage_a.predictor_trainable, False),
            ("stage_a.epochs", self.stage_a.epochs, 8),
            (
                "stage_a.per_device_train_batch_size",
                self.stage_a.per_device_train_batch_size,
                1,
            ),
            (
                "stage_a.gradient_accumulation_steps",
                self.stage_a.gradient_accumulation_steps,
                4,
            ),
            ("stage_a.world_size", self.stage_a.world_size, 4),
            ("stage_a.global_batch_size", self.stage_a.global_batch_size, 16),
            ("stage_a.loss_terms", self.stage_a.loss_terms, ("state", "answer")),
            (
                "stage_a.supervision_provenance",
                self.stage_a.supervision_provenance,
                "official_weak",
            ),
            (
                "stage_a.load_best_model_at_end",
                self.stage_a.load_best_model_at_end,
                False,
            ),
            ("stage_a.balanced_task_sampling", self.stage_a.balanced_task_sampling, True),
            (
                "stage_a.synthetic_engineering_gate_only",
                self.stage_a.synthetic_engineering_gate_only,
                False,
            ),
            ("stage_a.seed", self.stage_a.seed, 42),
            ("stage_a.optimizer.name", self.stage_a.optimizer.name, "adamw"),
            (
                "stage_a.optimizer.qwen_learning_rate",
                self.stage_a.optimizer.qwen_learning_rate,
                1.0e-5,
            ),
            (
                "stage_a.optimizer.state_learning_rate",
                self.stage_a.optimizer.state_learning_rate,
                1.0e-4,
            ),
            (
                "stage_a.optimizer.w0_learning_rate",
                self.stage_a.optimizer.w0_learning_rate,
                1.0e-4,
            ),
            ("stage_a.optimizer.weight_decay", self.stage_a.optimizer.weight_decay, 0.01),
            ("stage_a.optimizer.betas", self.stage_a.optimizer.betas, (0.9, 0.999)),
            ("stage_a.optimizer.epsilon", self.stage_a.optimizer.epsilon, 1.0e-8),
            ("stage_a.optimizer.grad_clip_norm", self.stage_a.optimizer.grad_clip_norm, 1.0),
            (
                "stage_a.checkpoint.format",
                self.stage_a.checkpoint.format,
                "full_model_optimizer_scheduler_rng_v1",
            ),
            ("stage_a.checkpoint.trainable_only", self.stage_a.checkpoint.trainable_only, False),
            (
                "stage_a.checkpoint.include_optimizer",
                self.stage_a.checkpoint.include_optimizer,
                True,
            ),
            (
                "stage_a.checkpoint.include_scheduler",
                self.stage_a.checkpoint.include_scheduler,
                True,
            ),
            ("stage_a.checkpoint.include_rng", self.stage_a.checkpoint.include_rng, True),
            ("stage_a.checkpoint.save_full_model", self.stage_a.checkpoint.save_full_model, True),
            (
                "stage_a.checkpoint.save_runtime_state",
                self.stage_a.checkpoint.save_runtime_state,
                False,
            ),
            (
                "stage_a.checkpoint.save_every_epochs",
                self.stage_a.checkpoint.save_every_epochs,
                2,
            ),
            (
                "stage_a.checkpoint.selection_policy",
                self.stage_a.checkpoint.selection_policy,
                "final_epoch",
            ),
            ("stage_b.variant", self.stage_b.variant, MetaTTTVariant.A3),
            ("stage_b.support_chunks", self.stage_b.support_chunks, 1),
            ("stage_b.minimum_query_points", self.stage_b.minimum_query_points, 1),
            ("stage_b.enabled_ttt_terms", self.stage_b.enabled_ttt_terms, ("pred",)),
            ("stage_b.inner_sgd_enabled", self.stage_b.inner_sgd_enabled, True),
            ("stage_b.update_effect", self.stage_b.update_effect, "next_chunk_only"),
            (
                "stage_b.auxiliary_outer_weight",
                self.stage_b.auxiliary_outer_weight,
                0.1,
            ),
            ("stage_b.reset_per_episode", self.stage_b.reset_per_episode, True),
            ("stage_b.compare_before_after", self.stage_b.compare_before_after, True),
            (
                "stage_b.meta_gradient_mode",
                self.stage_b.meta_gradient_mode,
                "meta_full_second_order",
            ),
            (
                "stage_b.reuse_strategy",
                self.stage_b.reuse_strategy,
                "causal_replay_isolated_prefill",
            ),
            (
                "stage_b.synthetic_engineering_gate_only",
                self.stage_b.synthetic_engineering_gate_only,
                True,
            ),
            ("stage_b.seed", self.stage_b.seed, 42),
            (
                "stage_c.variants",
                self.stage_c.variants,
                (MetaTTTVariant.A4, MetaTTTVariant.A5),
            ),
            (
                "stage_c.a4_enabled_ttt_terms",
                self.stage_c.a4_enabled_ttt_terms,
                ("pred", "identity"),
            ),
            (
                "stage_c.a5_enabled_ttt_terms",
                self.stage_c.a5_enabled_ttt_terms,
                ("pred", "identity", "event"),
            ),
            (
                "stage_c.support_chunk_schedule",
                self.stage_c.support_chunk_schedule,
                (),
            ),
            ("stage_c.maximum_support_chunks", self.stage_c.maximum_support_chunks, None),
            ("stage_c.minimum_query_points", self.stage_c.minimum_query_points, 2),
            ("stage_c.multi_query_enabled", self.stage_c.multi_query_enabled, True),
            (
                "stage_c.detach_overlap_snapshots",
                self.stage_c.detach_overlap_snapshots,
                True,
            ),
            (
                "stage_c.detach_runtime_between_chunks",
                self.stage_c.detach_runtime_between_chunks,
                True,
            ),
            ("stage_c.update_effect", self.stage_c.update_effect, "next_chunk_only"),
            (
                "stage_c.reuse_strategy",
                self.stage_c.reuse_strategy,
                "causal_replay_isolated_prefill",
            ),
            ("stage_c.direct_from_stage_a", self.stage_c.direct_from_stage_a, True),
            (
                "stage_c.meta_gradient_mode",
                self.stage_c.meta_gradient_mode,
                "truncated_second_order",
            ),
            ("stage_c.truncation_horizon", self.stage_c.truncation_horizon, 8),
            ("stage_c.reanchor_to_w0", self.stage_c.reanchor_to_w0, True),
            (
                "stage_c.segment_auxiliary_backward",
                self.stage_c.segment_auxiliary_backward,
                True,
            ),
            (
                "stage_c.training_counterfactual_enabled",
                self.stage_c.training_counterfactual_enabled,
                False,
            ),
            ("stage_c.prewarm_support_chunks", self.stage_c.prewarm_support_chunks, 1),
            ("stage_c.outer_step_scope", self.stage_c.outer_step_scope, "episode"),
            (
                "stage_c.synthetic_engineering_gate_only",
                self.stage_c.synthetic_engineering_gate_only,
                False,
            ),
            ("stage_c.seed", self.stage_c.seed, 42),
            (
                "stage_c.optimizer.state_learning_rate",
                self.stage_c.optimizer.state_learning_rate,
                5.0e-5,
            ),
            (
                "stage_c.optimizer.w0_learning_rate",
                self.stage_c.optimizer.w0_learning_rate,
                5.0e-5,
            ),
            (
                "stage_c.optimizer.predictor_learning_rate",
                self.stage_c.optimizer.predictor_learning_rate,
                5.0e-5,
            ),
            ("inference.reset_per_video", self.inference.reset_per_video, True),
            ("inference.update_effect", self.inference.update_effect, "next_chunk_only"),
            ("inference.prefill_once", self.inference.prefill_once, True),
            (
                "inference.decode_state_immutable",
                self.inference.decode_state_immutable,
                True,
            ),
            ("inference.release_on_exception", self.inference.release_on_exception, True),
            ("inference.audit_level", self.inference.audit_level, AuditLevel.BOUNDARY),
            (
                "inference.repeat_query_policy",
                self.inference.repeat_query_policy,
                "explicit_new_or_retry",
            ),
            ("inference.record_skip_reasons", self.inference.record_skip_reasons, True),
            (
                "inference.synthetic_engineering_gate_only",
                self.inference.synthetic_engineering_gate_only,
                True,
            ),
            (
                "evaluation.official_clean_tuning_forbidden",
                self.evaluation.official_clean_tuning_forbidden,
                True,
            ),
        )
        for path, actual, expected in checks:
            if actual != expected:
                raise ValueError(f"{path} must be {expected!r}; got {actual!r}")

        self._validate_attention_dimensions()
        self._validate_video_preprocessing_contract()
        self._validate_head_contracts()
        self._validate_state_and_query_contracts()
        self._validate_calibration_gate()
        self._validate_parameter_budget()
        return self

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
        expected: tuple[tuple[str, object, object], ...] = (
            (
                "temporal_input_conditioning",
                heads.temporal_input_conditioning,
                "inherited_query_conditioned_h_t",
            ),
            ("raw_logits", heads.raw_logits, True),
            ("debug_probabilities", heads.debug_probabilities, True),
            ("output_valid_mask", heads.output_valid_mask, True),
            ("output_timestamps", heads.output_timestamps, True),
            ("output_position_ids", heads.output_position_ids, True),
            (
                "invalid_output_policy",
                heads.invalid_output_policy,
                "zero_tensors_negative_one_metadata",
            ),
            ("online_frozen", heads.online_frozen, True),
            ("online_forward_no_grad", heads.online_forward_no_grad, False),
            ("detach_inputs", heads.detach_inputs, False),
            ("hard_state_mutation", heads.hard_state_mutation, False),
            ("o1.input_dim", heads.o1.input_dim, 768),
            ("o1.query_dim", heads.o1.query_dim, 512),
            ("o1.film_dim", heads.o1.film_dim, 1536),
            ("o1.hidden_dims", heads.o1.hidden_dims, (1024, 1024)),
            ("o1.output_dim", heads.o1.output_dim, 6),
            (
                "o1.output_names",
                heads.o1.output_names,
                ("object", "target", "visible", "enter", "exit", "confidence"),
            ),
            ("o1.layer_norm_eps", heads.o1.layer_norm_eps, 1.0e-5),
            ("o1.film_mode", heads.o1.film_mode, "one_plus_scale_and_shift"),
            ("o1.activation", heads.o1.activation, "silu"),
            ("o1.dropout", heads.o1.dropout, 0.0),
            ("o1.linear_bias", heads.o1.linear_bias, True),
            ("o1.parameter_count", heads.o1.parameter_count, 2_632_710),
            ("o1.object_threshold", heads.o1.object_threshold, 0.5),
            ("o1.target_threshold", heads.o1.target_threshold, 0.5),
            ("o1.visible_threshold", heads.o1.visible_threshold, 0.5),
            ("o1.enter_threshold", heads.o1.enter_threshold, 0.5),
            ("o1.exit_threshold", heads.o1.exit_threshold, 0.5),
            ("o1.confidence_threshold", heads.o1.confidence_threshold, 0.5),
            (
                "o1.baseline_policy",
                heads.o1.baseline_policy,
                "explicit_set_once_per_trajectory",
            ),
            (
                "o1.count_update_policy",
                heads.o1.count_update_policy,
                "recompute_from_full_slot_state",
            ),
            (
                "o1.committed_position_policy",
                heads.o1.committed_position_policy,
                "idempotent_preserve_and_audit_drift",
            ),
            ("o2.input_dim", heads.o2.input_dim, 768),
            ("o2.hidden_dims", heads.o2.hidden_dims, (1024, 1024)),
            ("o2.identity_dim", heads.o2.identity_dim, 256),
            ("o2.score_dim", heads.o2.score_dim, 2),
            ("o2.score_names", heads.o2.score_names, ("novelty", "match_confidence")),
            ("o2.layer_norm_eps", heads.o2.layer_norm_eps, 1.0e-5),
            ("o2.activation", heads.o2.activation, "silu"),
            ("o2.dropout", heads.o2.dropout, 0.0),
            ("o2.linear_bias", heads.o2.linear_bias, True),
            (
                "o2.identity_normalization",
                heads.o2.identity_normalization,
                "l2_fp32_unit_basis_fallback",
            ),
            ("o2.normalization_eps", heads.o2.normalization_eps, 1.0e-8),
            ("o2.parameter_count", heads.o2.parameter_count, 2_499_843),
            ("o2.prototype_ema", heads.o2.prototype_ema, 0.9),
            ("o2.confirmation_observations", heads.o2.confirmation_observations, 2),
            ("o2.match_threshold", heads.o2.match_threshold, 0.8),
            ("o2.novelty_threshold", heads.o2.novelty_threshold, 0.5),
            (
                "o2.match_confidence_threshold",
                heads.o2.match_confidence_threshold,
                0.5,
            ),
            ("o2.reliability_threshold", heads.o2.reliability_threshold, 0.5),
            (
                "o2.candidate_low_confidence_threshold",
                heads.o2.candidate_low_confidence_threshold,
                0.5,
            ),
            (
                "o2.match_ambiguity_margin",
                heads.o2.match_ambiguity_margin,
                1.0e-6,
            ),
            (
                "o2.threshold_status",
                heads.o2.threshold_status,
                CalibrationStatus.BOOTSTRAP_CALIBRATION_REQUIRED,
            ),
            ("e1.input_dim", heads.e1.input_dim, 768),
            ("e1.channels", heads.e1.channels, 512),
            ("e1.num_layers", heads.e1.num_layers, 5),
            ("e1.kernel_size", heads.e1.kernel_size, 3),
            ("e1.dilations", heads.e1.dilations, (1, 2, 4, 8, 16)),
            ("e1.output_dim", heads.e1.output_dim, 3),
            (
                "e1.output_names",
                heads.e1.output_names,
                ("eventness", "completion", "transition"),
            ),
            ("e1.layer_norm_eps", heads.e1.layer_norm_eps, 1.0e-5),
            ("e1.activation", heads.e1.activation, "silu_filter_sigmoid_gate"),
            ("e1.strict_causal", heads.e1.strict_causal, True),
            ("e1.batch_norm", heads.e1.batch_norm, False),
            ("e1.dropout", heads.e1.dropout, 0.0),
            ("e1.convolution_bias", heads.e1.convolution_bias, True),
            ("e1.causal_padding", heads.e1.causal_padding, "left"),
            ("e1.receptive_field", heads.e1.receptive_field, 63),
            ("e1.streaming_state_mode", heads.e1.streaming_state_mode, "projected_history"),
            ("e1.overlap_tubelets", heads.e1.overlap_tubelets, 4),
            ("e1.history_tubelets", heads.e1.history_tubelets, 66),
            (
                "e1.state_owner_keys",
                heads.e1.state_owner_keys,
                ("video_id", "trajectory_id", "query_signature"),
            ),
            ("e1.detach_runtime_default", heads.e1.detach_runtime_default, True),
            ("e1.parameter_count", heads.e1.parameter_count, 9_717_252),
            ("e1.tau_on", heads.e1.tau_on, 0.7),
            ("e1.tau_off", heads.e1.tau_off, 0.3),
            ("e1.completion_threshold", heads.e1.completion_threshold, 0.7),
            ("e1.transition_threshold", heads.e1.transition_threshold, 0.7),
            ("e1.min_gap_seconds", heads.e1.min_gap_seconds, 0.5),
            (
                "e1.fsm_policy",
                heads.e1.fsm_policy,
                "eventness_hysteresis_completion_transition",
            ),
            ("e1.cooldown_nms_source", heads.e1.cooldown_nms_source, "min_gap_seconds"),
            (
                "e1.committed_position_policy",
                heads.e1.committed_position_policy,
                "idempotent_ignore_and_audit",
            ),
            ("e2.input_dim", heads.e2.input_dim, 768),
            ("e2.hidden_dim", heads.e2.hidden_dim, 768),
            ("e2.num_layers", heads.e2.num_layers, 2),
            ("e2.event_output_dim", heads.e2.event_output_dim, 4),
            ("e2.phase_output_dim", heads.e2.phase_output_dim, 4),
            ("e2.event_names", heads.e2.event_names, ("start", "active", "end", "complete")),
            (
                "e2.phase_names",
                heads.e2.phase_names,
                ("inactive", "active", "end_candidate", "completed"),
            ),
            ("e2.layer_norm_eps", heads.e2.layer_norm_eps, 1.0e-5),
            ("e2.bidirectional", heads.e2.bidirectional, False),
            ("e2.batch_first", heads.e2.batch_first, True),
            ("e2.bias", heads.e2.bias, True),
            ("e2.dropout", heads.e2.dropout, 0.0),
            (
                "e2.streaming_state_mode",
                heads.e2.streaming_state_mode,
                "hidden_with_rollback_checkpoints",
            ),
            ("e2.overlap_tubelets", heads.e2.overlap_tubelets, 4),
            ("e2.checkpoint_tubelets", heads.e2.checkpoint_tubelets, 5),
            (
                "e2.state_owner_keys",
                heads.e2.state_owner_keys,
                ("video_id", "trajectory_id", "query_signature"),
            ),
            ("e2.detach_runtime_default", heads.e2.detach_runtime_default, True),
            ("e2.parameter_count", heads.e2.parameter_count, 7_293_449),
            ("e2.start_threshold", heads.e2.start_threshold, 0.6),
            ("e2.end_threshold", heads.e2.end_threshold, 0.6),
            ("e2.complete_threshold", heads.e2.complete_threshold, 0.7),
            (
                "e2.rearm_max_event_probability",
                heads.e2.rearm_max_event_probability,
                0.5,
            ),
            ("e2.rearm_phase", heads.e2.rearm_phase, "inactive"),
            ("e2.completed_hold_positions", heads.e2.completed_hold_positions, 1),
            (
                "e2.fsm_policy",
                heads.e2.fsm_policy,
                "phase_gated_single_transition_per_position",
            ),
            (
                "e2.active_evidence_policy",
                heads.e2.active_evidence_policy,
                "diagnostic_and_phase_consistency_only",
            ),
            (
                "e2.committed_position_policy",
                heads.e2.committed_position_policy,
                "idempotent_ignore_and_audit",
            ),
        )
        for path, actual, required in expected:
            if actual != required:
                raise ValueError(f"observation_heads.{path} must be {required!r}; got {actual!r}")

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
            raise ValueError("fast_ttt.online_parameter_count does not match two fast matrices")
        if self.spatial_encoder.active_slots > self.spatial_encoder.max_active_slots:
            raise ValueError("spatial_encoder.active_slots cannot exceed max_active_slots")

    def _validate_calibration_gate(self) -> None:
        statuses = (
            self.observation_heads.o1.threshold_status,
            self.observation_heads.o2.threshold_status,
            self.observation_heads.e1.threshold_status,
            self.observation_heads.e2.threshold_status,
            self.operator_router.threshold_status,
            self.time_resolver.threshold_status,
            self.retriever.threshold_status,
        )
        if self.evaluation.formal_evaluation_enabled and any(
            status is not CalibrationStatus.CALIBRATED for status in statuses
        ):
            raise ValueError("formal evaluation requires every threshold status to be calibrated")

    def _validate_parameter_budget(self) -> None:
        budget = self.parameter_budget
        components = (
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
        if (
            abs(sum(components) - budget.new_modules_total_millions)
            > budget.rounding_tolerance_millions
        ):
            raise ValueError(
                "parameter budget components exceed the architecture rounding tolerance"
            )
        exact_fast_millions = self.fast_ttt.online_parameter_count / 1_000_000
        if abs(exact_fast_millions - budget.online_fast_matrices_millions) > 1.0e-9:
            raise ValueError("online fast parameter budget must use the exact matrix count")
        exact_spatial_millions = 24_815_360 / 1_000_000
        if abs(exact_spatial_millions - budget.spatial_encoder_millions) > 1.0e-9:
            raise ValueError("spatial encoder budget must use the exact P6 parameter count")
        exact_temporal_millions = self.temporal_encoder.parameter_count / 1_000_000
        if abs(exact_temporal_millions - budget.temporal_encoder_millions) > 1.0e-9:
            raise ValueError("temporal encoder budget must use the exact P7 parameter count")
        exact_head_budgets = (
            (self.observation_heads.o1.parameter_count, budget.o1_millions, "O1"),
            (self.observation_heads.o2.parameter_count, budget.o2_millions, "O2"),
            (self.observation_heads.e1.parameter_count, budget.e1_millions, "E1"),
            (self.observation_heads.e2.parameter_count, budget.e2_millions, "E2"),
        )
        for parameter_count, millions, name in exact_head_budgets:
            if abs(parameter_count / 1_000_000 - millions) > 1.0e-9:
                raise ValueError(f"{name} budget must use the exact P8 parameter count")
        exact_projector_millions = self.state_bank.semantic_projector.parameter_count / 1_000_000
        if abs(exact_projector_millions - budget.semantic_projector_millions) > 1.0e-9:
            raise ValueError("Semantic Projector budget must use the exact P9 parameter count")
        exact_total_millions = 157_443_750 / 1_000_000
        if abs(exact_total_millions - budget.new_modules_total_millions) > 1.0e-9:
            raise ValueError("new module budget must use the frozen P9 component total")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read one UTF-8 YAML file and reject missing, unknown, or invalid values."""

    config_path = Path(path)
    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return cast(ProjectConfig, ProjectConfig.model_validate(raw))


def environment_summary() -> dict[str, object]:
    """Return the local runtime identity without resolving data or model paths."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and print the frozen v5 configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    print(load_config(args.config).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
