"""Load the slot-memory project configuration.

Inputs: one UTF-8 YAML file describing the project configuration.
Outputs: an immutable :class:`ProjectConfig`.
Forbidden: model forward logic, training logic, secret values, or platform absolute paths.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TRANSFORMERS_VERSION = "4.57.1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "model_state_ttt_8b.yaml"


class FrozenModel(BaseModel):  # type: ignore[misc]
    """Base for immutable configuration objects that ignore unknown keys."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class AuditLevel(StrEnum):
    """Runtime integrity work retained at each production audit level."""

    OFF = "off"
    BOUNDARY = "boundary"
    FULL = "full"


class VideoPreprocessingConfig(FrozenModel):
    sample_fps: float
    frames_per_chunk: int
    stride_frames: int
    processor_shortest_edge: int
    processor_longest_edge: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    pad_value: float
    full_tubelet_required_for_state: bool


class VisionConfig(FrozenModel):
    depth: int
    hidden_size: int
    num_heads: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    output_size: int
    deepstack_visual_indexes: tuple[int, ...]


class LLMConfig(FrozenModel):
    num_layers: int
    hidden_size: int


class ModelConfig(FrozenModel):
    base_model: str
    revision: str
    transformers_version: str
    vision: VisionConfig
    llm: LLMConfig


class FastTTTConfig(FrozenModel):
    input_dim: int
    bottleneck_dim: int
    output_dim: int
    residual_scale: float
    rms_norm_eps: float
    slow_projection_bias: bool
    fast_bias: bool
    fast_initialization: str
    fast_matrix_count: int
    online_parameter_count: int


class FastMemoryConfig(FrozenModel):
    """Zero-initialized per-video delta-rule slot memory replacing inner SGD."""

    write_rule: Literal["parallel_delta_rule"]
    eta_max_per_slot: float
    eta_chunk_budget: float
    eta_gate_hidden_dim: int
    eta_gate_init: float
    forget_beta_max: float
    forget_beta_init: float
    read_gate_init: float
    memory_dtype: Literal["float32"]
    zero_init_per_video: Literal[True]


class SpatialEncoderConfig(FrozenModel):
    input_dim: int
    hidden_dim: int
    stages: int
    num_heads: int
    head_dim: int
    refinements_per_stage: int
    ffn_dim: int
    active_slots: int
    max_active_slots: int
    query_dim: int
    layer_norm_eps: float
    attention_epsilon: float
    overflow_policy: str
    slot_valid_mask: bool
    log_overflow: bool


class TemporalEncoderConfig(FrozenModel):
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    dropout: float
    position_encoding: str
    layer_norm_eps: float
    activation: str
    strict_causal: bool
    cache_tubelets: int
    overlap_tubelets: int
    replay_context_tubelets: int
    query_dim: int


class O1Config(FrozenModel):
    input_dim: int
    query_dim: int
    film_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int
    layer_norm_eps: float
    film_mode: str
    activation: str
    dropout: float
    parameter_count: int
    object_threshold: float
    target_threshold: float
    visible_threshold: float
    enter_threshold: float
    exit_threshold: float
    confidence_threshold: float


class O2Config(FrozenModel):
    input_dim: int
    hidden_dims: tuple[int, ...]
    identity_dim: int
    score_dim: int
    score_names: tuple[str, ...]
    layer_norm_eps: float
    activation: str
    dropout: float
    identity_normalization: str
    normalization_eps: float
    parameter_count: int
    prototype_ema: float
    confirmation_observations: int
    match_threshold: float
    novelty_threshold: float
    match_confidence_threshold: float
    reliability_threshold: float
    candidate_low_confidence_threshold: float
    match_ambiguity_margin: float


class E1Config(FrozenModel):
    input_dim: int
    channels: int
    num_layers: int
    kernel_size: int
    dilations: tuple[int, ...]
    output_dim: int
    layer_norm_eps: float
    activation: str
    strict_causal: bool
    batch_norm: bool
    dropout: float
    convolution_bias: bool
    causal_padding: str
    receptive_field: int
    streaming_state_mode: str
    overlap_tubelets: int
    history_tubelets: int
    state_owner_keys: tuple[str, ...]
    detach_runtime_default: bool
    parameter_count: int
    tau_on: float
    tau_off: float
    completion_threshold: float
    transition_threshold: float
    min_gap_seconds: float
    cooldown_nms_source: str


class E2Config(FrozenModel):
    input_dim: int
    hidden_dim: int
    num_layers: int
    event_output_dim: int
    phase_output_dim: int
    event_names: tuple[str, ...]
    phase_names: tuple[str, ...]
    layer_norm_eps: float
    bidirectional: bool
    batch_first: bool
    bias: bool
    dropout: float
    streaming_state_mode: str
    overlap_tubelets: int
    checkpoint_tubelets: int
    state_owner_keys: tuple[str, ...]
    detach_runtime_default: bool
    parameter_count: int
    start_threshold: float
    end_threshold: float
    complete_threshold: float
    rearm_max_event_probability: float
    rearm_phase: str
    completed_hold_positions: int


class ObservationHeadsConfig(FrozenModel):
    o1: O1Config
    o2: O2Config
    e1: E1Config
    e2: E2Config


class SemanticProjectorConfig(FrozenModel):
    input_dim: int
    hidden_dim: int
    output_dim: int
    head_type_count: int
    head_types: tuple[str, ...]
    layer_norm_eps: float
    activation: str
    dropout: float
    normalization_dtype: str
    normalization_eps: float


class ConfirmedStoreConfig(FrozenModel):
    initial_capacity: int
    growth_chunk: int
    hard_limit: int | None
    storage_device: str
    storage_dtype: str
    exact_search: bool
    ann_enabled: bool
    gpu_hot_capacity: int
    hot_cache_enabled: bool
    hot_cache_device: str
    hot_cache_dtype: str


class CandidateStoreConfig(FrozenModel):
    initial_capacity: int
    growth_chunk: int
    hard_limit: int
    ttl_chunks: int
    match_threshold: float
    reliability_threshold: float
    low_confidence_threshold: float
    overflow_policy: str


class StateBankConfig(FrozenModel):
    semantic_dim: int
    identity_dim: int
    semantic_projector: SemanticProjectorConfig
    confirmed_store: ConfirmedStoreConfig
    candidate_store: CandidateStoreConfig
    event_history_capacity: int
    retrieval_history_capacity_per_head: int
    retrieval_history_source_dim: int


class QueryEncoderConfig(FrozenModel):
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    dropout: float
    output_dim: int
    bidirectional: bool
    position_encoding: str
    pooling: str


class OperatorRouterConfig(FrozenModel):
    prototypes: tuple[str, ...]
    output_dim: int
    temperature_initial: float
    temperature_trainable: bool
    confidence_threshold: float | None


class TimeResolverConfig(FrozenModel):
    modes: tuple[str, ...]
    input_dim: int
    hidden_dim: int
    mode_count: int
    token_hidden_dim: int
    pointer_heads: int
    confidence_threshold: float | None


class RetrieverConfig(FrozenModel):
    semantic_dim: int
    record_similarity_threshold: float
    similarity_dtype: str
    normalization_eps: float
    record_confidence_threshold: float | None
    score_chunk_size: int
    top_k: int | None


class StateResamplerConfig(FrozenModel):
    num_queries: int
    num_layers: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    hidden_dim: int
    output_dim: int
    layer_norm_eps: float
    activation: str
    dropout: float
    attention_bias: bool
    output_projection_bias: bool
    attention_softmax_dtype: str


class StateReaderConfig(FrozenModel):
    signed_exact_count: bool
    empty_exact_count: int
    o2_identity_key: str
    tokenizer_class: str
    tokenizer_vocab_size: int


class AssociativeTTTConfig(FrozenModel):
    """Bank-conditioned key-context contract feeding the slot memory."""

    bank_embedding_dim: int
    key_dim: int


class OfficialWeakBalanceConfig(FrozenModel):
    """Formal EMA Answer-reference composition for the four official-weak terms."""

    group_weight: float
    answer_reference_floor: float = 0.1
    scale_min: float
    scale_max: float
    epsilon: float
    ema_beta: float = 0.99
    grad_ema_beta: float = 0.99
    grad_scale_min: float = 0.1
    grad_scale_max: float = 10.0


class LossConfig(FrozenModel):
    operator_weight: float
    retrieval_weight: float
    time_weight: float
    official_weak_balance: OfficialWeakBalanceConfig


class OuterGradientControlMode(StrEnum):
    PER_GROUP_L2_EQUAL_UPDATE_CAP = "per_group_l2_equal_update_cap"


class OuterMaxGradNormConfig(FrozenModel):
    qwen: float
    fast_slow: float
    state_shared: float
    state_task: float
    state_router_time: float
    state_retrieval: float
    w0: float
    associative: float


class OuterGradientControlConfig(FrozenModel):
    """Fixed A2/A5 Outer gradient caps."""

    mode: OuterGradientControlMode
    max_grad_norm: OuterMaxGradNormConfig


class A2OptimizerConfig(FrozenModel):
    qwen_learning_rate: float
    state_learning_rate: float
    w0_learning_rate: float


class A2TrainingConfig(FrozenModel):
    optimizer: A2OptimizerConfig


class A5OptimizerConfig(FrozenModel):
    """Non-Qwen A5 parameter-group learning rates owned by State-TTT."""

    fast_slow_learning_rate: float
    state_learning_rate: float
    w0_learning_rate: float
    associative_learning_rate: float


class A5QueryMetaGradientConfig(FrozenModel):
    """Robust aggregation applied only to Query fast-weight cotangents."""

    max_norm: float
    epsilon: float


class A5WarmupConfig(FrozenModel):
    """Independent Memory/State handoff stage."""

    max_steps: Literal[256]
    linear_warmup_steps: Literal[4]
    fast_slow_learning_rate: float
    state_learning_rate: float
    w0_learning_rate: float
    associative_learning_rate: float


class A5TrainingConfig(FrozenModel):
    """Direct A5 contract with K-step truncation and one episode seed."""

    truncation_horizon: int
    seed: int
    optimizer: A5OptimizerConfig
    warmup: A5WarmupConfig
    query_meta_gradient: A5QueryMetaGradientConfig


class InferenceRuntimeConfig(FrozenModel):
    """Production inference audit cost selected at the process boundary."""

    audit_level: AuditLevel


class ProjectConfig(FrozenModel):
    """Production configuration for the slot-memory State-TTT model."""

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
    associative_ttt: AssociativeTTTConfig
    loss: LossConfig
    outer_gradient_control: OuterGradientControlConfig
    a2: A2TrainingConfig
    a5: A5TrainingConfig
    inference: InferenceRuntimeConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read one UTF-8 YAML file into a :class:`ProjectConfig`."""

    config_path = Path(path)
    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return cast(ProjectConfig, ProjectConfig.model_validate(raw))
