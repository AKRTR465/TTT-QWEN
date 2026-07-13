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

SPEC_VERSION = "state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval"
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


class PathsConfig(FrozenModel):
    model_root_env: str
    svcbench_root_env: str
    hf_home_env: str
    output_root_env: str


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


class FastTTTConfig(FrozenModel):
    input_dim: PositiveInt
    bottleneck_dim: PositiveInt
    output_dim: PositiveInt
    residual_scale: PositiveFloat
    fast_bias: bool
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
    strict_causal: bool
    cache_tubelets: PositiveInt
    query_dim: PositiveInt


class O1Config(FrozenModel):
    input_dim: PositiveInt
    query_dim: PositiveInt
    film_dim: PositiveInt
    hidden_dims: tuple[int, ...]
    output_dim: PositiveInt
    threshold_status: CalibrationStatus


class O2Config(FrozenModel):
    input_dim: PositiveInt
    hidden_dims: tuple[int, ...]
    identity_dim: PositiveInt
    score_dim: PositiveInt
    prototype_ema: Probability
    confirmation_observations: PositiveInt
    match_threshold: Probability | None
    threshold_status: CalibrationStatus


class E1Config(FrozenModel):
    input_dim: PositiveInt
    channels: PositiveInt
    num_layers: PositiveInt
    kernel_size: PositiveInt
    dilations: tuple[int, ...]
    output_dim: PositiveInt
    tau_on: Probability
    tau_off: Probability
    min_gap_seconds: NonNegativeFloat
    threshold_status: CalibrationStatus


class E2Config(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    num_layers: PositiveInt
    event_output_dim: PositiveInt
    phase_output_dim: PositiveInt
    start_threshold: Probability
    end_threshold: Probability
    complete_threshold: Probability
    threshold_status: CalibrationStatus


class ObservationHeadsConfig(FrozenModel):
    o1: O1Config
    o2: O2Config
    e1: E1Config
    e2: E2Config


class SemanticProjectorConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    output_dim: PositiveInt


class ConfirmedStoreConfig(FrozenModel):
    initial_capacity: PositiveInt
    growth_chunk: PositiveInt
    hard_limit: PositiveInt | None
    storage_device: str
    storage_dtype: str
    gpu_hot_capacity: PositiveInt


class CandidateStoreConfig(FrozenModel):
    initial_capacity: PositiveInt
    growth_chunk: PositiveInt
    hard_limit: PositiveInt
    ttl_chunks: PositiveInt
    overflow_policy: str


class StateBankConfig(FrozenModel):
    semantic_dim: PositiveInt
    identity_dim: PositiveInt
    semantic_projector: SemanticProjectorConfig
    confirmed_store: ConfirmedStoreConfig
    candidate_store: CandidateStoreConfig
    event_history_capacity: PositiveInt
    isolation_keys: tuple[str, ...]
    hard_updates_no_grad: bool
    detach_before_write: bool
    included_in_state_dict: bool


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
    pooling: str


class OperatorRouterConfig(FrozenModel):
    prototypes: tuple[str, ...]
    output_dim: PositiveInt
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
    empty_record_embedding: bool


class PredictorConfig(FrozenModel):
    input_dim: PositiveInt
    hidden_dim: PositiveInt
    output_dim: PositiveInt


class LossConfig(FrozenModel):
    pred_weight: NonNegativeFloat
    identity_weight: NonNegativeFloat
    event_weight: NonNegativeFloat
    o1_unlabeled_weight: NonNegativeFloat
    auxiliary_outer_weight: NonNegativeFloat


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
    paths: PathsConfig
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
    predictor: PredictorConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    parameter_budget: ParameterBudgetConfig

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_v5_contract(self) -> Self:
        checks: tuple[tuple[str, object, object], ...] = (
            ("spec_version", self.spec_version, SPEC_VERSION),
            ("paths.model_root_env", self.paths.model_root_env, "QWEN_MODEL_ROOT"),
            ("paths.svcbench_root_env", self.paths.svcbench_root_env, "SVCBENCH_ROOT"),
            ("paths.hf_home_env", self.paths.hf_home_env, "HF_HOME"),
            ("paths.output_root_env", self.paths.output_root_env, "OUTPUT_ROOT"),
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
            ("fast_ttt.fast_bias", self.fast_ttt.fast_bias, False),
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
            ("spatial_encoder.slot_valid_mask", self.spatial_encoder.slot_valid_mask, True),
            ("spatial_encoder.log_overflow", self.spatial_encoder.log_overflow, True),
            ("temporal_encoder.input_dim", self.temporal_encoder.input_dim, 4096),
            ("temporal_encoder.hidden_dim", self.temporal_encoder.hidden_dim, 768),
            ("temporal_encoder.num_layers", self.temporal_encoder.num_layers, 6),
            ("temporal_encoder.num_heads", self.temporal_encoder.num_heads, 12),
            ("temporal_encoder.head_dim", self.temporal_encoder.head_dim, 64),
            ("temporal_encoder.ffn_dim", self.temporal_encoder.ffn_dim, 3072),
            ("temporal_encoder.dropout", self.temporal_encoder.dropout, 0.1),
            ("temporal_encoder.strict_causal", self.temporal_encoder.strict_causal, True),
            ("temporal_encoder.cache_tubelets", self.temporal_encoder.cache_tubelets, 64),
            ("temporal_encoder.query_dim", self.temporal_encoder.query_dim, 512),
            ("state_bank.semantic_dim", self.state_bank.semantic_dim, 512),
            ("state_bank.identity_dim", self.state_bank.identity_dim, 256),
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
                "state_bank.confirmed_store.gpu_hot_capacity",
                self.state_bank.confirmed_store.gpu_hot_capacity,
                256,
            ),
            (
                "state_bank.candidate_store.initial_capacity",
                self.state_bank.candidate_store.initial_capacity,
                64,
            ),
            (
                "state_bank.candidate_store.hard_limit",
                self.state_bank.candidate_store.hard_limit,
                512,
            ),
            ("state_bank.event_history_capacity", self.state_bank.event_history_capacity, 512),
            ("query_encoder.input_dim", self.query_encoder.input_dim, 4096),
            ("query_encoder.hidden_dim", self.query_encoder.hidden_dim, 768),
            ("query_encoder.num_layers", self.query_encoder.num_layers, 4),
            ("query_encoder.num_heads", self.query_encoder.num_heads, 12),
            ("query_encoder.head_dim", self.query_encoder.head_dim, 64),
            ("query_encoder.ffn_dim", self.query_encoder.ffn_dim, 3072),
            ("query_encoder.dropout", self.query_encoder.dropout, 0.1),
            ("query_encoder.output_dim", self.query_encoder.output_dim, 512),
            ("query_encoder.bidirectional", self.query_encoder.bidirectional, True),
            ("operator_router.output_dim", self.operator_router.output_dim, 512),
            ("retriever.semantic_dim", self.retriever.semantic_dim, 512),
            (
                "retriever.record_similarity_threshold",
                self.retriever.record_similarity_threshold,
                0.35,
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
            ("predictor.input_dim", self.predictor.input_dim, 768),
            ("predictor.hidden_dim", self.predictor.hidden_dim, 1536),
            ("predictor.output_dim", self.predictor.output_dim, 768),
            ("loss.pred_weight", self.loss.pred_weight, 1.0),
            ("loss.identity_weight", self.loss.identity_weight, 0.5),
            ("loss.event_weight", self.loss.event_weight, 0.5),
            ("loss.o1_unlabeled_weight", self.loss.o1_unlabeled_weight, 0.0),
            ("loss.auxiliary_outer_weight", self.loss.auxiliary_outer_weight, 0.1),
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
        self._validate_head_contracts()
        self._validate_state_and_query_contracts()
        self._validate_calibration_gate()
        self._validate_parameter_budget()
        return self

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
        expected: tuple[tuple[str, object, object], ...] = (
            ("o1.input_dim", self.observation_heads.o1.input_dim, 768),
            ("o1.query_dim", self.observation_heads.o1.query_dim, 512),
            ("o1.film_dim", self.observation_heads.o1.film_dim, 1536),
            ("o1.hidden_dims", self.observation_heads.o1.hidden_dims, (1024, 1024)),
            ("o1.output_dim", self.observation_heads.o1.output_dim, 6),
            ("o2.input_dim", self.observation_heads.o2.input_dim, 768),
            ("o2.hidden_dims", self.observation_heads.o2.hidden_dims, (1024, 1024)),
            ("o2.identity_dim", self.observation_heads.o2.identity_dim, 256),
            ("o2.score_dim", self.observation_heads.o2.score_dim, 2),
            ("e1.input_dim", self.observation_heads.e1.input_dim, 768),
            ("e1.channels", self.observation_heads.e1.channels, 512),
            ("e1.num_layers", self.observation_heads.e1.num_layers, 5),
            ("e1.kernel_size", self.observation_heads.e1.kernel_size, 3),
            ("e1.dilations", self.observation_heads.e1.dilations, (1, 2, 4, 8, 16)),
            ("e1.output_dim", self.observation_heads.e1.output_dim, 3),
            ("e2.input_dim", self.observation_heads.e2.input_dim, 768),
            ("e2.hidden_dim", self.observation_heads.e2.hidden_dim, 768),
            ("e2.num_layers", self.observation_heads.e2.num_layers, 2),
            ("e2.event_output_dim", self.observation_heads.e2.event_output_dim, 4),
            ("e2.phase_output_dim", self.observation_heads.e2.phase_output_dim, 4),
        )
        for path, actual, required in expected:
            if actual != required:
                raise ValueError(f"observation_heads.{path} must be {required!r}; got {actual!r}")

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
        if self.time_resolver.modes != ("now", "history", "recent", "explicit_range"):
            raise ValueError("time_resolver.modes must contain the frozen 4 modes")
        if self.state_bank.isolation_keys != ("video_id", "trajectory_id", "head_type"):
            raise ValueError("state_bank.isolation_keys must isolate video, trajectory, and head")
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
