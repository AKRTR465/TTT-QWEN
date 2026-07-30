from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import ttt_svcbench_qwen.llamafactory_trainer as trainer_module
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter
from ttt_svcbench_qwen.llamafactory_trainer import (
    CheckpointPolicy,
    OuterParameterAudit,
    ProductionStage,
    ProductionTrainerRuntime,
    SegmentBackwardController,
    TTTQwenTrainerMixin,
    _A2AuditAccumulator,
    _aggregate_operator_diagnostics,
    _checkpoint_policy_from_environment,
    _ControlledDeepSpeedEngineWrapper,
    _disable_smoke_checkpoints,
    _publish_epoch_two_four_checkpoints,
    _reset_a2_to_a5_associative,
    _reset_a2_to_a5_balance,
    _validate_checkpoint_tree,
    _validate_resume_balance_schema,
    make_production_outer_optimizer_factory,
    resolve_same_stage_resume,
)
from ttt_svcbench_qwen.outer_gradient_control import OuterGradientController
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakBalanceAudit,
    OfficialWeakOuterLossComposer,
    OfficialWeakTermBalanceMetrics,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    LlamaFactoryCheckoutAudit,
    LlamaFactorySymbols,
    ProductionTTTConfig,
    QwenOuterTrainabilityConfig,
    audit_outer_checkpoint_boundary,
    configure_qwen_outer_trainability,
    fully_unfreeze_qwen,
    initialize_outer_model_from_a2,
    load_outer_checkpoint,
    load_training_yaml,
)
from ttt_svcbench_qwen.production_runtime import (
    ProductionOuterModel,
    QueryObservationSpec,
    SupportChunkSpec,
    _build_runtime_preprocess_cache,
    _decode_query_targets_grouped,
    _decode_targets_with_seek,
    _decode_uniform_interval,
    _llamafactory_uniform_frame_indices,
    _query_chunk_spec,
    _resize_to_pixel_budget,
    _TargetSeekUnavailable,
    _uniform_target_times,
    _video_pixel_bounds,
    build_inference_runtime_bundle,
)
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakLossAudit,
    OperatorDiagnosticAudit,
)

ROOT = Path(__file__).resolve().parents[1]

_A2_EXTENSION_FIELDS: dict[str, object] = {
    "stage": "a2",
    "project_config": "configs/model_state_ttt_8b.yaml",
    "dataset_manifest": "manifest.json",
    "support_prefetch_depth": 2,
    "support_decode_coalesce": True,
    "support_materialization": "dataloader_episode",
    "state_query_visual_mode": "recent_chunk",
    "state_query_max_frames": 16,
    "answer_query_visual_mode": "causal_prefix",
    "answer_query_max_frames": 256,
    "state_query_cache_mode": "inherit",
    "answer_query_cache_mode": "disabled",
    "preprocess_cache_mode": "read_write",
    "preprocess_cache_miss_policy": "decode",
    "preprocess_cache_root_env": "TTT_PREPROCESS_CACHE_ROOT",
    "preprocess_cache_max_gb": 200.0,
    "preprocess_cache_dtype": "float32",
}


class _OuterToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.p_context = nn.Linear(4, 4)


def test_trainer_main_destroys_initialized_process_group_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[bool] = []
    monkeypatch.setattr(trainer_module, "_run_main", lambda _argv: 0)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    assert trainer_module.main(["config.yaml"]) == 0
    assert destroyed == [True]


def test_trainer_main_destroys_initialized_process_group_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[bool] = []

    def fail(_argv: list[str] | None) -> int:
        raise RuntimeError("training failed")

    monkeypatch.setattr(trainer_module, "_run_main", fail)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        trainer_module.main(["config.yaml"])
    assert destroyed == [True]


def test_trainer_main_skips_process_group_teardown_when_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "_run_main", lambda _argv: 0)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def unexpected_destroy() -> None:
        raise AssertionError("uninitialized process group must not be destroyed")

    monkeypatch.setattr(torch.distributed, "destroy_process_group", unexpected_destroy)

    assert trainer_module.main(["config.yaml"]) == 0


def test_a5_outer_parameter_audit_allows_partial_qwen_with_full_state_training() -> None:
    audit = OuterParameterAudit(
        stage=ProductionStage.A5,
        total_parameter_count=100,
        trainable_parameter_count=70,
        qwen_parameter_count=60,
        qwen_trainable_count=30,
        non_qwen_parameter_count=40,
        non_qwen_trainable_count=40,
        associative_parameter_count=10,
        associative_trainable_count=10,
        transient_parameter_names=(),
        backbone_registered=True,
    )

    assert audit.qwen_trainable_count < audit.qwen_parameter_count


def test_a5_outer_parameter_audit_rejects_frozen_state_parameter() -> None:
    with pytest.raises(ValueError, match="every state, W0, and Associative"):
        OuterParameterAudit(
            stage=ProductionStage.A5,
            total_parameter_count=100,
            trainable_parameter_count=69,
            qwen_parameter_count=60,
            qwen_trainable_count=30,
            non_qwen_parameter_count=40,
            non_qwen_trainable_count=39,
            associative_parameter_count=10,
            associative_trainable_count=10,
            transient_parameter_names=(),
            backbone_registered=True,
        )


def test_static_w0_outer_parameter_audit_requires_frozen_associative() -> None:
    audit = OuterParameterAudit(
        stage=ProductionStage.A5,
        a5_adaptation_mode="static_w0",
        total_parameter_count=100,
        trainable_parameter_count=60,
        qwen_parameter_count=60,
        qwen_trainable_count=30,
        non_qwen_parameter_count=40,
        non_qwen_trainable_count=30,
        associative_parameter_count=10,
        associative_trainable_count=0,
        transient_parameter_names=(),
        backbone_registered=True,
    )

    assert audit.associative_trainable_count == 0
    with pytest.raises(ValueError, match="Associative must remain frozen"):
        OuterParameterAudit(
            stage=ProductionStage.A5,
            a5_adaptation_mode="static_w0",
            total_parameter_count=100,
            trainable_parameter_count=61,
            qwen_parameter_count=60,
            qwen_trainable_count=30,
            non_qwen_parameter_count=40,
            non_qwen_trainable_count=31,
            associative_parameter_count=10,
            associative_trainable_count=1,
            transient_parameter_names=(),
            backbone_registered=True,
        )


def test_runtime_preprocess_cache_honors_explicit_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setenv("TEST_PREPROCESS_CACHE_ROOT", str(root))
    monkeypatch.setenv("TTT_PREPROCESS_CACHE_NAMESPACE", "statequery-v1")
    backbone = SimpleNamespace(
        model_args=SimpleNamespace(model_name_or_path="model", revision="main"),
        processor=object(),
        project_config=SimpleNamespace(
            video_preprocessing=SimpleNamespace(
                processor_shortest_edge=256,
                processor_longest_edge=131072,
            )
        ),
    )
    config = SimpleNamespace(
        preprocess_cache_mode="readonly",
        preprocess_cache_miss_policy="error",
        preprocess_cache_root_env="TEST_PREPROCESS_CACHE_ROOT",
        preprocess_cache_max_gb=1,
        preprocess_cache_dtype="float32",
    )

    cache = _build_runtime_preprocess_cache(backbone, config)

    assert cache is not None
    assert cache.namespace == "statequery-v1"


def test_inference_bundle_rejects_processor_without_qwen_tokenizer_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "qwen"
    model_root.mkdir()
    processor = SimpleNamespace(
        apply_chat_template=lambda *_args, **_kwargs: "",
        video_processor=lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime.transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: processor,
    )

    def reject_tokenizer_fallback(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("AutoTokenizer fallback must not be used")

    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime.transformers.AutoTokenizer.from_pretrained",
        reject_tokenizer_fallback,
    )

    with pytest.raises(TypeError, match="requires the Qwen3-VL tokenizer"):
        build_inference_runtime_bundle(
            model_root=model_root,
            checkpoint=tmp_path / "unused-checkpoint",
            device="cpu",
            dtype=torch.float32,
        )


class _GroupedOuterToy(nn.Module):
    def __init__(self, qwen: nn.Module, *, associative_trainable: bool) -> None:
        super().__init__()
        self.qwen = qwen
        self.state_model = nn.Module()
        self.state_model.component_modules = nn.ModuleDict(
            {
                "spatial_encoder": nn.Linear(4, 4),
                "observation_heads": nn.Linear(4, 4),
                "query_encoder": _GroupedQueryToy(),
                "state_bank": _GroupedStateBankToy(),
            }
        )
        self.fast_adapter = build_fast_ttt_adapter(load_config())
        for parameter in self.fast_adapter.collect_associative_parameters():
            parameter.requires_grad_(associative_trainable)


class _GroupedQueryToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_head = nn.Linear(4, 4)
        self.operator_router = nn.Linear(4, 4)
        self.time_resolver = nn.Linear(4, 4)


class _GroupedStateBankToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_projector = nn.Linear(4, 4)


def _grouped_bundle(
    tmp_path: Path,
    project: ProjectConfig,
    *,
    adaptation_mode: str = "meta_ttt",
    associative_trainable: bool | None = None,
) -> tuple[LlamaFactoryBackboneBundle, nn.Module]:
    qwen = nn.Linear(4, 4)
    checkout = tmp_path / "lf"
    checkout.mkdir()
    symbols = LlamaFactorySymbols(
        get_train_args=lambda *_args, **_kwargs: (),
        load_tokenizer=lambda *_args, **_kwargs: {},
        load_model=lambda *_args, **_kwargs: qwen,
        trainer_base=object,
        checkout=LlamaFactoryCheckoutAudit(checkout, "523f801", False, True),
    )
    bundle = LlamaFactoryBackboneBundle(
        model=qwen,
        tokenizer=object(),
        processor=None,
        model_args=object(),
        data_args=object(),
        training_args=SimpleNamespace(
            learning_rate=5.0e-6,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1.0e-8,
            weight_decay=0.01,
            seed=42,
            data_seed=42,
        ),
        finetuning_args=object(),
        generating_args=object(),
        project_config=project,
        ttt_config=ProductionTTTConfig(
            stage="a5",
            a5_adaptation_mode=adaptation_mode,
            warmup_bundle="warmup" if adaptation_mode == "meta_ttt" else None,
            project_config="configs/model_state_ttt_8b.yaml",
            dataset_manifest="manifest.json",
            initialize_from_a2_checkpoint="a2-final",
            support_prefetch_depth=2,
            support_decode_coalesce=True,
            support_materialization="segment_double_buffer",
            segment_prefetch_depth=1,
            state_query_visual_mode="recent_chunk",
            state_query_max_frames=16,
            answer_query_visual_mode="causal_prefix",
            answer_query_max_frames=256,
            state_query_cache_mode="inherit",
            answer_query_cache_mode="disabled",
            preprocess_cache_mode="read_write",
            preprocess_cache_miss_policy="decode",
            preprocess_cache_root_env="TTT_PREPROCESS_CACHE_ROOT",
            preprocess_cache_max_gb=200.0,
            preprocess_cache_dtype="float32",
        ),
        symbols=symbols,
    )
    if associative_trainable is None:
        associative_trainable = adaptation_mode == "meta_ttt"
    return bundle, _GroupedOuterToy(qwen, associative_trainable=associative_trainable)


class _QwenOwnerToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_embed = nn.Linear(2, 2)
        self.visual.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(27)])
        self.visual.merger = nn.Linear(2, 2)
        self.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(8, 2)
        self.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(36)])
        self.language_model.norm = nn.LayerNorm(2)


def _checkpoint_balance_state(
    *,
    schema: int = 7,
    ema_dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    return {
        "official_weak_balancer.ema_values": torch.zeros(5, dtype=ema_dtype),
        "official_weak_balancer.ema_valid": torch.zeros(5, dtype=torch.bool),
        "official_weak_balancer.ema_update_counts": torch.zeros(5, dtype=torch.int64),
        "official_weak_balancer.gradient_ema_values": torch.zeros(4, dtype=torch.float64),
        "official_weak_balancer.gradient_ema_valid": torch.zeros(4, dtype=torch.bool),
        "official_weak_balancer.gradient_ema_update_counts": torch.zeros(4, dtype=torch.int64),
        "official_weak_balancer.balance_schema_version": torch.tensor(schema, dtype=torch.int64),
    }


def test_outer_checkpoint_loader_accepts_only_exact_safetensors(tmp_path: Path) -> None:
    source = _OuterToy()
    checkpoint = tmp_path / "outer.safetensors"
    save_file(source.state_dict(), checkpoint)
    target = _OuterToy()
    target.requires_grad_(False)
    for parameter in target.parameters():
        parameter.zero_()

    audit = load_outer_checkpoint(target, checkpoint)

    assert audit.format == "safetensors"
    assert audit.tensor_count == len(source.state_dict())
    assert all(
        torch.equal(target.state_dict()[key], value) for key, value in source.state_dict().items()
    )
    torch.save(source.state_dict(), tmp_path / "outer.bin")
    with pytest.raises(ValueError, match="safetensors"):
        load_outer_checkpoint(target, tmp_path / "outer.bin")
    bad = dict(source.state_dict())
    bad["temporal_cache.hidden"] = torch.zeros(1)
    save_file(bad, tmp_path / "bad.safetensors")
    with pytest.raises(ValueError, match="exactly match"):
        load_outer_checkpoint(target, tmp_path / "bad.safetensors")
    removed_controller = dict(source.state_dict())
    removed_controller["step_controller.weight"] = torch.zeros((1, 7))
    save_file(removed_controller, tmp_path / "learned-step.safetensors")
    with pytest.raises(ValueError, match="exactly match"):
        load_outer_checkpoint(target, tmp_path / "learned-step.safetensors")


def test_production_outer_checkpoint_owns_ema_balance_state() -> None:
    config = load_config()
    qwen = nn.Linear(2, 2)
    balancer = OfficialWeakOuterLossComposer(config.loss.official_weak_balance)
    outer = ProductionOuterModel(nn.Linear(2, 2), qwen, balancer)

    keys = set(audit_outer_checkpoint_boundary(outer))

    assert "official_weak_balancer.ema_values" in keys
    assert "official_weak_balancer.ema_valid" in keys
    assert "official_weak_balancer.ema_update_counts" in keys
    assert "official_weak_balancer.gradient_ema_values" in keys
    assert "official_weak_balancer.gradient_ema_valid" in keys
    assert "official_weak_balancer.gradient_ema_update_counts" in keys
    assert "official_weak_balancer.balance_schema_version" in keys


def test_a2_to_a5_resets_loss_and_gradient_ema() -> None:
    config = load_config()
    balancer = OfficialWeakOuterLossComposer(config.loss.official_weak_balance)
    balancer.ema_values.fill_(3.0)
    balancer.ema_valid.fill_(True)
    balancer.ema_update_counts.fill_(4)
    balancer.gradient_ema_values.fill_(5.0)
    balancer.gradient_ema_valid.fill_(True)
    balancer.gradient_ema_update_counts.fill_(6)
    outer = ProductionOuterModel(nn.Linear(2, 2), nn.Linear(2, 2), balancer)

    _reset_a2_to_a5_balance(outer)

    assert not bool(balancer.ema_valid.any())
    assert not bool(balancer.gradient_ema_valid.any())
    assert not bool(balancer.ema_update_counts.any())
    assert not bool(balancer.gradient_ema_update_counts.any())
    assert int(balancer.balance_schema_version.item()) == 7


def test_same_stage_resume_rejects_old_balance_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    save_file(
        {"official_weak_balancer.ema_values": torch.zeros(5)},
        checkpoint / "model.safetensors",
    )
    with pytest.raises(ValueError, match="missing required tensor"):
        _validate_resume_balance_schema(checkpoint)

    save_file(_checkpoint_balance_state(schema=6), checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="requires schema 7"):
        _validate_resume_balance_schema(checkpoint)

    save_file(
        _checkpoint_balance_state(ema_dtype=torch.bfloat16),
        checkpoint / "model.safetensors",
    )
    with pytest.raises(ValueError, match="must be torch.float64"):
        _validate_resume_balance_schema(checkpoint)

    save_file(_checkpoint_balance_state(), checkpoint / "model.safetensors")
    _validate_resume_balance_schema(checkpoint)


def test_balance_schema_validator_reads_all_sharded_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    state = _checkpoint_balance_state()
    keys = tuple(state)
    first_keys = keys[:3]
    second_keys = keys[3:]
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    save_file({key: state[key] for key in first_keys}, checkpoint / first_name)
    save_file({key: state[key] for key in second_keys}, checkpoint / second_name)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **{key: first_name for key in first_keys},
                    **{key: second_name for key in second_keys},
                }
            }
        ),
        encoding="utf-8",
    )

    _validate_resume_balance_schema(checkpoint)


def test_a2_audit_accumulator_aggregates_all_microbatches_and_flushes() -> None:
    accumulator = _A2AuditAccumulator()
    for index, answer in enumerate((1.0, 3.0, 5.0, 7.0)):
        terms = tuple(
            OfficialWeakTermBalanceMetrics(
                name=name,
                raw_global_mean=torch.tensor(answer + term_index, dtype=torch.float64),
                scale=torch.tensor(1.0 + term_index, dtype=torch.float64),
                aligned_global_mean=torch.tensor(answer + term_index, dtype=torch.float64),
                weighted_global_mean=torch.tensor(0.01 * answer, dtype=torch.float64),
                global_valid_count=torch.tensor(
                    0.0 if name == "retrieval" and index % 2 == 0 else 1.0,
                    dtype=torch.float64,
                ),
                scale_clamped=torch.tensor(index % 2 == 0),
                loss_scale=torch.tensor(1.0, dtype=torch.float64),
                gradient_scale=torch.tensor(1.0, dtype=torch.float64),
                raw_gradient_rms=torch.tensor(answer, dtype=torch.float64),
                ema_gradient_rms=torch.tensor(answer + 0.5, dtype=torch.float64),
            )
            for term_index, name in enumerate(("task", "operator", "retrieval", "time"))
        )
        balance = OfficialWeakBalanceAudit(
            answer_global_mean=torch.tensor(answer, dtype=torch.float64),
            answer_global_count=torch.tensor(1.0, dtype=torch.float64),
            state_global_mean=torch.tensor(0.1 * answer, dtype=torch.float64),
            terms=terms,
            auxiliary_to_answer_ratio=torch.tensor(0.1, dtype=torch.float64),
            group_guard=torch.tensor(0.8 + 0.05 * index, dtype=torch.float64),
            group_guard_active=torch.tensor(index < 2),
            group_guard_reference=torch.tensor(2.0, dtype=torch.float64),
            group_guard_reference_floored=torch.tensor(index == 0),
            state_to_reference_ratio=torch.tensor(0.1, dtype=torch.float64),
            state_to_current_answer_ratio=torch.tensor(0.1, dtype=torch.float64),
            ema_means=tuple(
                torch.tensor(answer + offset, dtype=torch.float64) for offset in range(5)
            ),
            ema_update_counts=tuple(torch.tensor(index + 1) for _ in range(5)),
            gradient_ema_rms=tuple(
                torch.tensor(answer + offset, dtype=torch.float64) for offset in range(4)
            ),
            gradient_ema_update_counts=tuple(torch.tensor(index + 1) for _ in range(4)),
        )
        weak = OfficialWeakLossAudit(
            labels_joined_after_forward=True,
            runtime_payload_reused_for_labels=False,
            identity_target_fabricated=False,
            unique_retrieval_id_fabricated=False,
            future_occurrences_ignored=0,
            retrieval_bag_sizes=(1,),
            retrieval_valid_bag_rows=1,
        )
        accumulator.add(balance, weak)

    metrics = accumulator.flush()

    assert metrics["loss/ga_microbatch_count"] == 4.0
    assert metrics["loss/answer"] == pytest.approx(4.0)
    assert metrics["loss/state"] == pytest.approx(0.4)
    assert metrics["loss/raw/task"] == pytest.approx(4.0)
    assert metrics["loss/global_valid_count/retrieval"] == 2.0
    assert metrics["loss/group_guard_active"] == pytest.approx(0.5)
    assert metrics["loss/group_guard_reference_floored"] == pytest.approx(0.25)
    assert metrics["loss/ema/answer"] == pytest.approx(7.0)
    assert metrics["loss/ema_updates/answer"] == 4.0
    assert metrics["retrieval/valid_bag_rows"] == 4.0
    assert accumulator.flush() == {}


def test_operator_diagnostics_aggregate_confusion_before_macro_recall() -> None:
    audits: list[OfficialWeakLossAudit] = []
    rows = ((0, 0, 0), (1, 1, 8), (0, 1, 1), (1, 1, 1))
    for target, raw_prediction, effective_prediction in rows:
        raw = [0] * 72
        effective = [0] * 72
        support = [0] * 8
        loss_sums = [0.0] * 8
        raw[target * 9 + raw_prediction] = 1
        effective[target * 9 + effective_prediction] = 1
        support[target] = 1
        loss_sums[target] = 1.0 + target
        audits.append(
            OfficialWeakLossAudit(
                labels_joined_after_forward=True,
                runtime_payload_reused_for_labels=False,
                identity_target_fabricated=False,
                unique_retrieval_id_fabricated=False,
                future_occurrences_ignored=0,
                retrieval_bag_sizes=(),
                operator_diagnostics=OperatorDiagnosticAudit(
                    raw_confusion=tuple(raw),
                    effective_confusion=tuple(effective),
                    class_loss_sums=tuple(loss_sums),
                    class_support=tuple(support),
                    confidence_sum=0.8,
                    entropy_sum=1.0,
                    temperature_sum=1.0,
                    temperature_count=1,
                ),
            )
        )
    metrics: dict[str, float] = {}
    _aggregate_operator_diagnostics(metrics, audits)

    assert metrics["operator/support/o1-snap"] == 2.0
    assert metrics["operator/support/o1-delta"] == 2.0
    assert metrics["operator/micro_accuracy"] == pytest.approx(0.5)
    assert metrics["operator/macro_recall"] == pytest.approx(0.5)
    assert metrics["operator/predicted_unsupported_rate"] == pytest.approx(0.25)
    assert metrics["operator/effective_confusion/o1-delta/unsupported"] == 1.0
    assert metrics["operator/temperature"] == pytest.approx(1.0)


def test_a2_yaml_runs_four_epochs_and_keeps_only_the_final_checkpoint(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml"
    )

    assert native["num_train_epochs"] == 4.0
    assert native["save_strategy"] == "epoch"
    assert "save_steps" not in native
    assert native["save_total_limit"] == 1
    assert native["save_only_model"] is False
    assert native["video_max_pixels"] == 131_072
    assert extension.stage == "a2"
    assert set(extension.model_dump(exclude_none=True)) == {
        "stage",
        "a5_adaptation_mode",
        "a5_phase",
        "project_config",
        "dataset_manifest",
        "qwen_outer_trainability",
        "visual_cost_index",
        "support_prefetch_depth",
        "support_decode_coalesce",
        "support_materialization",
        "prepared_episode_max_bytes",
        "support_visual_batch_size",
        "query_encoder_reuse",
        "query_frame_sampling",
        "query_sample_fps",
        "state_query_visual_mode",
        "state_query_max_frames",
        "answer_query_visual_mode",
        "answer_query_max_frames",
        "query_decode_max_groups",
        "state_query_cache_mode",
        "answer_query_cache_mode",
        "query_activation_offload",
        "preprocess_cache_mode",
        "preprocess_cache_miss_policy",
        "preprocess_cache_root_env",
        "preprocess_cache_max_gb",
        "preprocess_cache_dtype",
        "visual_cost_mode",
        "runtime_trace_mode",
        "segment_prefetch_depth",
        "semantic_projector_delta_audit_steps",
        "a5_parameter_delta_audit_steps",
        "operator_diagnostics_interval",
    }


def test_fullprefix256_yaml_matches_qwen_visual_budget_and_dynamic_graph_zero1(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml"
    )

    assert native["video_fps"] == 2.0
    assert native["video_maxlen"] == 256
    assert native["cutoff_len"] == 16_384
    assert native["deepspeed"] == "configs/h200/deepspeed_zero1_dynamic_graph.json"
    assert native["per_device_train_batch_size"] == 1
    assert native["gradient_accumulation_steps"] == 4
    assert native["dataloader_num_workers"] == 2
    assert native["dataloader_prefetch_factor"] == 2
    assert native["save_strategy"] == "epoch"
    assert native["save_total_limit"] == 1
    assert native["max_grad_norm"] == 0.0
    assert extension.state_query_visual_mode == "recent_chunk"
    assert extension.state_query_max_frames == 16
    assert extension.answer_query_visual_mode == "causal_prefix"
    assert extension.answer_query_max_frames == 256
    assert extension.query_decode_max_groups == 16
    assert extension.state_query_cache_mode == "inherit"
    assert extension.answer_query_cache_mode == "disabled"
    assert extension.cached_query_roles == frozenset(("state_query",))
    assert extension.visual_cost_mode == "exact_tokens_then_runtime"
    assert extension.visual_cost_index == "/tmp/visual_cost_index.json"


def test_semantic_repair_train_split_recipe_saves_only_epochs_two_and_four(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "RUN_ROOT": "/tmp/run",
        "SVCBENCH_DATASET_MANIFEST": "/tmp/dataset_manifest.json",
        "VISUAL_COST_INDEX": "/tmp/state16_answer256_schema4.json",
    }.items():
        monkeypatch.setenv(key, value)

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml"
    )

    assert native["num_train_epochs"] == 4.0
    assert native["save_strategy"] == "steps"
    assert native["save_steps"] == 0.5
    assert native["save_total_limit"] == 2
    assert native["resume_from_checkpoint"] is None
    assert extension.stage == "a2"
    assert extension.state_query_visual_mode == "recent_chunk"
    assert extension.state_query_max_frames == 16
    assert extension.answer_query_visual_mode == "causal_prefix"
    assert extension.answer_query_max_frames == 256
    assert extension.state_query_cache_mode == "inherit"
    assert extension.answer_query_cache_mode == "disabled"
    assert extension.query_cache_enabled("state_query")
    assert not extension.query_cache_enabled("answer_query")
    assert extension.preprocess_cache_mode == "readonly"


def test_dual_query_visual_config_is_required_and_legacy_is_rejected() -> None:
    legacy_fields = {
        key: value
        for key, value in _A2_EXTENSION_FIELDS.items()
        if not key.startswith(("state_query_", "answer_query_"))
    }
    with pytest.raises(ValueError, match="Field required"):
        ProductionTTTConfig(**legacy_fields)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ProductionTTTConfig(
            **_A2_EXTENSION_FIELDS,
            query_visual_mode="causal_prefix",
            query_max_frames=256,
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ProductionTTTConfig(**_A2_EXTENSION_FIELDS, query_cache_mode="inherit")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ProductionTTTConfig(**_A2_EXTENSION_FIELDS, query_decode_strategy="legacy_seek")
    with pytest.raises(ValueError, match="support_materialization"):
        ProductionTTTConfig(
            **{**_A2_EXTENSION_FIELDS, "support_materialization": "trainer_prefetch"}
        )
    with pytest.raises(ValueError, match="A2 requires full"):
        ProductionTTTConfig(
            **_A2_EXTENSION_FIELDS,
            qwen_outer_trainability={
                "mode": "partial",
                "vision_freeze_first_blocks": 13,
                "decoder_train_last_layers": 8,
                "train_vision_patch_embed": False,
                "train_main_merger": True,
                "train_deepstack_mergers": True,
                "train_language_model_norm": True,
                "train_input_embeddings": False,
                "train_lm_head": False,
            },
        )


def test_split_query_specs_bound_state_to_16_and_answer_to_256(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    config = ProductionTTTConfig(
        stage="a2",
        project_config="configs/model_state_ttt_8b.yaml",
        dataset_manifest="manifest.json",
        support_prefetch_depth=2,
        support_decode_coalesce=True,
        support_materialization="dataloader_episode",
        state_query_visual_mode="recent_chunk",
        state_query_max_frames=16,
        answer_query_visual_mode="causal_prefix",
        answer_query_max_frames=256,
        state_query_cache_mode="inherit",
        answer_query_cache_mode="disabled",
        preprocess_cache_mode="read_write",
        preprocess_cache_miss_policy="decode",
        preprocess_cache_root_env="TTT_PREPROCESS_CACHE_ROOT",
        preprocess_cache_max_gb=200.0,
        preprocess_cache_dtype="float32",
    )
    state = _query_chunk_spec(
        "q:state_query",
        video,
        20.0,
        reset_soft_state=False,
        config=config,
        role="state_query",
    )
    answer = _query_chunk_spec(
        "q:answer_query",
        video,
        20.0,
        reset_soft_state=False,
        config=config,
        role="answer_query",
    )

    assert (state.start_time, state.end_time, state.maximum_frames) == (12.0, 20.0, 16)
    assert state.observation_role == "state_query"
    assert (answer.start_time, answer.end_time, answer.maximum_frames) == (0.0, 20.0, 256)
    assert answer.observation_role == "answer_query"


def test_fullprefix256_trace_override_requires_run_root(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("TTT_DATALOADER_TRACE", "1")
    monkeypatch.delenv("RUN_ROOT", raising=False)

    with pytest.raises(ValueError, match="requires RUN_ROOT"):
        load_training_yaml(ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml")


def test_fullprefix256_trace_and_cost_preflight_overrides(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "SVCBENCH_DATASET_MANIFEST": "/tmp/dataset_manifest.json",
        "TTT_DATALOADER_TRACE": "1",
        "RUN_ROOT": "/tmp/run",
        "TTT_VISUAL_COST_PREFLIGHT": "1",
        "TTT_SMOKE_MAX_STEPS": "1",
    }.items():
        monkeypatch.setenv(key, value)

    _, extension = load_training_yaml(ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml")

    assert extension.runtime_trace_mode == "cuda"
    assert Path(extension.runtime_trace_dir or "") == Path("/tmp/run/runtime_trace")
    assert extension.visual_cost_mode == "proxy"
    assert extension.visual_cost_index is None


def test_fullprefix256_cost_preflight_requires_explicit_smoke(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("TTT_VISUAL_COST_PREFLIGHT", "1")
    monkeypatch.delenv("TTT_SMOKE_MAX_STEPS", raising=False)

    with pytest.raises(ValueError, match="explicit smoke run"):
        load_training_yaml(ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml")


def test_a2_lazy_ga_fetch_pulls_each_microbatch_only_when_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "ttt_svcbench_qwen.llamafactory_trainer.trace_event",
        lambda event, **fields: events.append((event, fields)),
    )
    owner = SimpleNamespace(ttt_runtime=SimpleNamespace(stage=ProductionStage.A2))
    pulled: list[int] = []

    def source():
        for index in range(4):
            pulled.append(index)
            yield {"prepared_a2": index}

    batches, num_items = TTTQwenTrainerMixin.get_batch_samples(
        owner, iter(source()), 4, torch.device("cpu")
    )

    assert num_items is None
    assert len(batches) == 4
    assert pulled == []
    iterator = iter(batches)
    assert next(iterator) == {"prepared_a2": 0}
    assert pulled == [0]
    assert [name for name, _ in events] == ["a2_ga_microbatch_fetch"]
    assert list(iterator) == [{"prepared_a2": index} for index in range(1, 4)]
    assert pulled == [0, 1, 2, 3]
    assert [name for name, _ in events] == [
        "a2_ga_microbatch_fetch",
        "a2_ga_microbatch_fetch",
        "a2_ga_microbatch_fetch",
        "a2_ga_microbatch_fetch",
        "a2_ga_group_fetch",
    ]
    assert events[-1][1]["fetched_batches"] == 4


def test_a2_lazy_ga_fails_closed_on_transformers_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.llamafactory_trainer.transformers.__version__", "4.58.0")
    owner = SimpleNamespace(ttt_runtime=SimpleNamespace(stage=ProductionStage.A2))

    with pytest.raises(RuntimeError, match="pinned to Transformers 4.57.1"):
        TTTQwenTrainerMixin.get_batch_samples(
            owner, iter(({"prepared_a2": 0},)), 1, torch.device("cpu")
        )


def test_a2_runtime_cost_observation_includes_collate_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeA2Record:
        query = SimpleNamespace(runtime=SimpleNamespace(query_id="query-1"))

    observations: list[tuple[str, float]] = []
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("ttt_svcbench_qwen.llamafactory_trainer.A2QueryRecord", _FakeA2Record)
    monkeypatch.setattr(
        "ttt_svcbench_qwen.llamafactory_trainer.trace_event",
        lambda event, **fields: events.append((event, fields)),
    )
    owner = SimpleNamespace(
        ttt_runtime=SimpleNamespace(stage=ProductionStage.A2),
        _ttt_train_sampler=SimpleNamespace(
            observe_runtime_cost=lambda record_id, seconds: observations.append(
                (record_id, seconds)
            )
        ),
    )
    prepared = SimpleNamespace(
        record=_FakeA2Record(),
        preparation=SimpleNamespace(collate_seconds=7.5),
    )

    TTTQwenTrainerMixin._observe_runtime_cost(owner, {"prepared_a2": prepared}, 2.5)

    assert observations == [("query-1", 10.0)]
    assert events == [
        (
            "runtime_cost_observation",
            {
                "record_id": "query-1",
                "preparation_seconds": 7.5,
                "training_seconds": 2.5,
                "seconds": 10.0,
            },
        )
    ]


def test_a2_fullprefix_uses_dynamic_graph_safe_zero1_profile() -> None:
    root = Path(__file__).parents[1]
    text = (root / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml").read_text(encoding="utf-8")
    assert "deepspeed: configs/h200/deepspeed_zero1_dynamic_graph.json" in text

    profile = json.loads(
        (root / "configs/h200/deepspeed_zero1_dynamic_graph.json").read_text(encoding="utf-8")
    )
    zero = profile["zero_optimization"]
    assert zero["stage"] == 1
    assert zero["overlap_comm"] is False
    assert zero["reduce_scatter"] is False
    assert zero["round_robin_gradients"] is False
    assert zero["ignore_unused_parameters"] is False
    assert profile["gradient_clipping"] == 0.0


def test_h200_entries_default_to_fullprefix256_profiles() -> None:
    root = Path(__file__).parents[1]
    train = (root / "scripts/h200/train_a2_a5.sh").read_text(encoding="utf-8")
    launch = (root / "scripts/h200/launch_4gpu.sh").read_text(encoding="utf-8")
    for text in (train, launch):
        assert "a2_qwen3vl8b_fullprefix256_4gpu.yaml" in text
        assert "a5_meta_ttt_k8_fullprefix256_4gpu.yaml" in text
    for removed in (
        "configs/h200/a2_qwen3vl8b_full_4gpu.yaml",
        "configs/h200/a2_qwen3vl8b_full_4gpu_120g.yaml",
        "configs/h200/a5_meta_ttt_k8_4gpu.yaml",
        "scripts/h200/launch_qwen3vl8b_ttt_a2_full4.sh",
        "scripts/h200/launch_qwen3vl8b_ttt_a5_k8_full4.sh",
        "scripts/h200/train_a2_allsvcbench_4epoch.sh",
    ):
        assert not (root / removed).exists()


def test_h200_training_entries_disable_shortest_first_by_default() -> None:
    root = Path(__file__).parents[1]
    train = (root / "scripts/h200/train_a2_a5.sh").read_text(encoding="utf-8")
    benchmark = (root / "scripts/h200/benchmark_fullprefix256_8step.sh").read_text(encoding="utf-8")

    default_assignment = 'TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"'
    assert default_assignment in train
    assert default_assignment in benchmark
    assert "export TTT_SMOKE_SHORTEST_FIRST=1" not in benchmark


def test_production_video_pixel_bounds_use_model_arguments_and_keep_tokens_dynamic() -> None:
    bounds = _video_pixel_bounds(
        SimpleNamespace(
            model_args=SimpleNamespace(video_min_pixels=786_432, video_max_pixels=1_048_576),
            data_args=SimpleNamespace(video_min_pixels=16 * 16, video_max_pixels=16 * 16),
        )
    )
    assert bounds == (786_432, 1_048_576)

    frames = torch.zeros((2, 3, 360, 640), dtype=torch.uint8)
    resized = _resize_to_pixel_budget(
        frames,
        minimum_pixels=bounds[0],
        maximum_pixels=bounds[1],
    )
    assert resized.shape[-2:] == (672, 1184)
    assert 786_432 <= resized.shape[-2] * resized.shape[-1] <= 1_048_576


def test_long_interval_decoder_seeks_targets_without_retaining_all_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "long.mp4"
    path.touch()
    counters = {"decoded": 0, "converted": 0, "seeks": 0}
    fps = 30
    total_frames = 2_050 * fps

    class _Frame:
        def __init__(self, index: int) -> None:
            self.time = index / fps

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            counters["converted"] += 1
            return np.zeros((2, 3, 3), dtype=np.uint8)

    stream = SimpleNamespace(time_base=Fraction(1, fps))

    class _Container:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(video=[stream])
            self.cursor = 0

        def __enter__(self) -> _Container:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def seek(self, offset: int, **_kwargs: object) -> None:
            counters["seeks"] += 1
            # Emulate backward seek to a keyframe at most one second earlier.
            self.cursor = max(0, offset - offset % fps)

        def decode(self, _stream: object) -> Iterator[_Frame]:
            for index in range(self.cursor, total_frames):
                counters["decoded"] += 1
                yield _Frame(index)

    monkeypatch.setattr("ttt_svcbench_qwen.production_runtime.av.open", lambda _path: _Container())
    spec = SupportChunkSpec(
        chunk_id="long-support",
        video_path=path,
        start_time=0.0,
        end_time=2_048.0,
        maximum_frames=16,
        query_time=4_449.0,
    )

    frames, timestamps = _decode_uniform_interval(spec, sample_fps=2.0)

    assert frames.shape == (16, 3, 2, 3)
    assert timestamps.shape == (16,)
    assert bool(torch.all(timestamps[1:] > timestamps[:-1]))
    assert counters["seeks"] == 16
    assert counters["converted"] == 16
    assert counters["decoded"] < 16 * (fps + 3)


def test_query_prefix_allows_256_frames_without_relaxing_support_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video.mp4"
    path.touch()
    query = QueryObservationSpec(
        chunk_id="query",
        video_path=path,
        start_time=0.0,
        end_time=663.0,
        maximum_frames=256,
        query_time=663.0,
        sampling_fps=2.0,
    )

    targets = _uniform_target_times(query, query.sampling_fps)

    assert len(targets) == 256
    assert targets[0] == 0.0
    assert targets[-1] == 663.0
    assert all(value <= query.query_time for value in targets)
    with pytest.raises(ValueError, match="Support chunks permit"):
        SupportChunkSpec("support", path, 0.0, 8.0, 256, 8.0)


def test_query_uniform_indices_match_llamafactory_523f801_reference() -> None:
    indices = _llamafactory_uniform_frame_indices(
        total_frames=1_989,
        duration=663.0,
        video_fps=2.0,
        video_maxlen=256,
    )

    reference = tuple(int(value) for value in np.linspace(0, 1_988, 256).astype(np.int32).tolist())
    assert indices == reference
    assert len(indices) == 256


def test_grouped_query_decode_matches_legacy_frames_with_at_most_sixteen_seeks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "long-query.mp4"
    path.touch()
    fps = 12
    total_frames = 664 * fps
    counters = {"seeks": 0}

    class _Frame:
        def __init__(self, index: int) -> None:
            self.index = index
            self.time = index / fps

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            value = np.zeros((2, 2, 3), dtype=np.uint8)
            value[0, 0] = (self.index & 255, (self.index >> 8) & 255, 0)
            return value

    stream = SimpleNamespace(time_base=Fraction(1, fps))

    class _Container:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(video=[stream])
            self.cursor = 0

        def __enter__(self) -> _Container:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def seek(self, offset: int, **_kwargs: object) -> None:
            counters["seeks"] += 1
            self.cursor = max(0, offset - offset % fps)

        def decode(self, _stream: object) -> Iterator[_Frame]:
            for index in range(self.cursor, total_frames):
                yield _Frame(index)

    monkeypatch.setattr("ttt_svcbench_qwen.production_runtime.av.open", lambda _path: _Container())
    query = QueryObservationSpec(
        chunk_id="query",
        video_path=path,
        start_time=0.0,
        end_time=663.0,
        maximum_frames=256,
        query_time=663.0,
        sampling_fps=2.0,
        decode_max_groups=16,
    )
    targets = _uniform_target_times(query, query.sampling_fps)

    legacy_frames, legacy_timestamps = _decode_targets_with_seek(query, targets)
    legacy_seek_count = counters["seeks"]
    counters["seeks"] = 0
    grouped_frames, grouped_timestamps = _decode_query_targets_grouped(
        query, targets, max_groups=16
    )

    assert legacy_seek_count == 256
    assert counters["seeks"] == 16
    assert grouped_timestamps == legacy_timestamps
    assert len(grouped_frames) == len(legacy_frames) == 256
    assert all(
        torch.equal(grouped, legacy)
        for grouped, legacy in zip(grouped_frames, legacy_frames, strict=True)
    )


def test_grouped_query_decode_falls_back_to_one_streaming_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "nonseekable.mp4"
    path.touch()
    query = QueryObservationSpec(
        "query",
        path,
        0.0,
        32.0,
        64,
        32.0,
    )
    calls: list[str] = []
    targets = [float(index) * 32.0 / 63.0 for index in range(64)]
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._llamafactory_query_target_times",
        lambda _spec, _fps: targets,
    )

    def unavailable(*_args: object, **_kwargs: object) -> tuple[list[torch.Tensor], list[float]]:
        calls.append("grouped")
        raise _TargetSeekUnavailable("no timestamp index")

    def streaming(_spec: object, values: list[float]) -> tuple[list[torch.Tensor], list[float]]:
        calls.append("streaming")
        return [torch.zeros((3, 2, 2), dtype=torch.uint8) for _ in values], values

    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._decode_query_targets_grouped", unavailable
    )
    monkeypatch.setattr("ttt_svcbench_qwen.production_runtime._decode_targets_streaming", streaming)
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._decode_targets_with_seek",
        lambda *_args, **_kwargs: pytest.fail("legacy per-target seek must not run"),
    )

    frames, timestamps = _decode_uniform_interval(query, query.sampling_fps)

    assert calls == ["grouped", "streaming"]
    assert frames.shape[0] == 64
    assert timestamps.shape == (64,)


def test_short_interval_decoder_streams_once_instead_of_seeking_every_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "short.mp4"
    path.touch()
    calls: list[str] = []

    def decode(name: str):
        def inner(_spec: SupportChunkSpec, targets: list[float]):
            calls.append(name)
            frames = [torch.zeros((3, 2, 3), dtype=torch.uint8) for _ in targets]
            return frames, list(targets)

        return inner

    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._decode_targets_streaming",
        decode("stream"),
    )
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_runtime._decode_targets_with_seek",
        decode("seek"),
    )
    short = SupportChunkSpec("short", path, 10.0, 18.0, 16, 20.0)
    long = SupportChunkSpec("long", path, 10.0, 42.0, 8, 50.0)

    short_frames, _ = _decode_uniform_interval(short, sample_fps=2.0)
    long_frames, _ = _decode_uniform_interval(long, sample_fps=2.0)

    assert calls == ["stream", "seek"]
    assert short_frames.shape[0] == 16
    assert long_frames.shape[0] == 8


def test_outer_model_forces_non_reentrant_gradient_checkpointing() -> None:
    class _CheckpointingQwen(nn.Module):
        supports_gradient_checkpointing = True

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.calls: list[dict[str, object] | None] = []

        def gradient_checkpointing_enable(
            self, *, gradient_checkpointing_kwargs: dict[str, object] | None = None
        ) -> None:
            self.calls.append(gradient_checkpointing_kwargs)

        def gradient_checkpointing_disable(self) -> None:
            pass

    qwen = _CheckpointingQwen()
    outer = ProductionOuterModel(nn.Linear(1, 1), qwen)
    outer.gradient_checkpointing_enable({"use_reentrant": True, "preserve_rng_state": False})

    assert qwen.calls == [{"use_reentrant": False, "preserve_rng_state": False}]


def test_training_yaml_expands_required_environment_and_keeps_a5_fresh(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("A2_CHECKPOINT", "/tmp/a2-final")
    monkeypatch.setenv("A5_WARMUP_BUNDLE", "/tmp/a5-warmup")

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a5_meta_ttt_k8_fullprefix256_4gpu.yaml"
    )

    assert native["resume_from_checkpoint"] is None
    assert native["output_dir"] == "/tmp/output"
    assert extension.initialize_from_a2_checkpoint == "/tmp/a2-final"
    assert extension.stage == "a5"
    assert extension.a5_phase == "main"
    assert extension.warmup_bundle == "/tmp/a5-warmup"

    monkeypatch.delenv("A2_CHECKPOINT")
    with pytest.raises(ValueError, match="unresolved environment variables"):
        load_training_yaml(ROOT / "configs/h200/a5_meta_ttt_k8_fullprefix256_4gpu.yaml")


def test_a5_partial_qwen_yaml_selects_vit_half_and_decoder_last_eight(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("A2_CHECKPOINT", "/tmp/a2-final")
    monkeypatch.setenv("A5_WARMUP_BUNDLE", "/tmp/a5-warmup")

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"
    )
    policy = extension.qwen_outer_trainability

    assert native["finetuning_type"] == "full"
    assert native["num_train_epochs"] == 4.0
    assert native["save_strategy"] == "no"
    assert native["save_total_limit"] == 1
    assert native["save_only_model"] is False
    assert native["deepspeed"] == "configs/h200/deepspeed_zero1_dynamic_graph.json"
    assert extension.stage == "a5"
    assert extension.a5_phase == "main"
    assert extension.warmup_bundle == "/tmp/a5-warmup"
    assert policy.mode == "partial"
    assert policy.vision_freeze_first_blocks == 13
    assert policy.decoder_train_last_layers == 8
    assert not policy.train_vision_patch_embed
    assert policy.train_main_merger
    assert policy.train_deepstack_mergers
    assert policy.train_language_model_norm
    assert not policy.train_input_embeddings
    assert not policy.train_lm_head

    launcher = (ROOT / "scripts/h200/train_a5_vithalf_decoder8.sh").read_text(encoding="utf-8")
    assert 'TTT_CHECKPOINT_POLICY="atomic_final_only"' in launcher
    assert 'TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"' in launcher
    assert 'TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"' in launcher
    assert "[[ $# -eq 3 ]] || usage" in launcher
    assert "<a5_warmup_bundle>" in launcher
    assert "<dataset_manifest.json>" in launcher


def test_a5_associative_lttt_finalonly_launcher_contract(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "SVCBENCH_DATASET_MANIFEST": "/tmp/v4_manifest.json",
        "A2_CHECKPOINT": "/tmp/a2-final",
        "A5_WARMUP_BUNDLE": "/tmp/a5-warmup",
    }.items():
        monkeypatch.setenv(key, value)

    native, extension = load_training_yaml(
        ROOT / "configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"
    )
    launcher = (ROOT / "scripts/h200/train_a5_associative_lttt_finalonly.sh").read_text(
        encoding="utf-8"
    )

    assert native["num_train_epochs"] == 4.0
    assert extension.stage == "a5"
    assert extension.a5_adaptation_mode == "meta_ttt"
    assert extension.a5_phase == "main"
    assert extension.warmup_bundle == "/tmp/a5-warmup"
    assert extension.initialize_from_a2_checkpoint == "/tmp/a2-final"
    assert 'TTT_A5_ADAPTATION_MODE="meta_ttt"' in launcher
    assert 'TTT_CHECKPOINT_POLICY="atomic_final_only"' in launcher
    assert "a5_dense_querybundle_train_support_statequery_fp16_v4" in launcher
    assert 'TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"' in launcher
    assert 'TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"' in launcher
    assert "[[ $# -eq 3 ]] || usage" in launcher
    assert 'train_a2_a5.sh" a5 "$1" "$3"' in launcher


def test_a5_fast_state_warmup_yaml_and_launcher_are_restart_only(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/v4_manifest.json")
    monkeypatch.setenv("A2_CHECKPOINT", "/tmp/a2-final")

    native, extension = load_training_yaml(ROOT / "configs/h200/a5_fast_state_warmup_128_4gpu.yaml")
    launcher = (ROOT / "scripts/h200/train_a5_fast_state_warmup.sh").read_text(encoding="utf-8")

    assert native["max_steps"] == 128
    assert native["warmup_steps"] == 4
    assert native["lr_scheduler_type"] == "cosine"
    assert native["save_strategy"] == "no"
    assert native["resume_from_checkpoint"] is None
    assert extension.a5_phase == "fast_state_warmup"
    assert extension.warmup_bundle is None
    assert extension.qwen_outer_trainability.mode == "frozen"
    assert "[[ $# -eq 2 ]] || usage" in launcher
    assert "a5_warmup_bundle only" in launcher
    assert 'TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"' in launcher


def test_a5_static_w0_yaml_and_launcher_match_meta_ttt_data_contract(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "SVCBENCH_DATASET_MANIFEST": "/tmp/v4_manifest.json",
        "A2_CHECKPOINT": "/tmp/a2-final",
        "A5_WARMUP_BUNDLE": "/tmp/a5-warmup",
    }.items():
        monkeypatch.setenv(key, value)

    meta_native, meta = load_training_yaml(
        ROOT / "configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"
    )
    static_native, static = load_training_yaml(
        ROOT / "configs/h200/a5_static_w0_k8_vithalf_decoder8_4gpu.yaml"
    )

    assert meta.a5_adaptation_mode == "meta_ttt"
    assert static.a5_adaptation_mode == "static_w0"
    assert static.stage == meta.stage == "a5"
    assert static.dataset_manifest == meta.dataset_manifest == "/tmp/v4_manifest.json"
    assert static.initialize_from_a2_checkpoint == meta.initialize_from_a2_checkpoint
    assert static.qwen_outer_trainability == meta.qwen_outer_trainability
    for key in (
        "num_train_epochs",
        "seed",
        "data_seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "deepspeed",
    ):
        assert static_native[key] == meta_native[key]

    launcher = (ROOT / "scripts/h200/train_a5_static_w0_ablation.sh").read_text(encoding="utf-8")
    assert 'TTT_A5_ADAPTATION_MODE="static_w0"' in launcher
    assert "a5_dense_querybundle_train_support_statequery_fp16_v4" in launcher
    assert 'TTT_CHECKPOINT_POLICY="atomic_final_only"' in launcher
    assert 'TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"' in launcher
    assert "[[ $# -eq 2 ]] || usage" in launcher


def test_training_yaml_rejects_unknown_extension_keys_and_invalid_stage_checkpoint(
    tmp_path: Path,
) -> None:
    def write(extension: dict[str, object]) -> Path:
        path = tmp_path / "training.yaml"
        lines = ["model_name_or_path: model", "ttt_qwen:"]
        lines.extend(f"  {key}: {json.dumps(value)}" for key, value in extension.items())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_training_yaml(write({**_A2_EXTENSION_FIELDS, "inner_learning_rate": 1.0e-4}))
    with pytest.raises(ValueError, match="A2 must not initialize"):
        load_training_yaml(
            write({**_A2_EXTENSION_FIELDS, "initialize_from_a2_checkpoint": "checkpoint"})
        )
    with pytest.raises(ValueError, match="A5 requires initialize"):
        load_training_yaml(write({**_A2_EXTENSION_FIELDS, "stage": "a5"}))


def test_full_unfreeze_accepts_qwen_module_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _QwenOwnerToy()
    wrapper = nn.Module()
    wrapper.model = owner
    wrapper.requires_grad_(False)
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_factory.assert_qwen_runtime_structure",
        lambda _owner, _config: None,
    )

    audit = fully_unfreeze_qwen(wrapper, load_config())

    assert audit.decoder_layer_count == 36
    assert audit.all_qwen_parameters_trainable
    assert all(parameter.requires_grad for parameter in wrapper.parameters())


def test_partial_qwen_trainability_freezes_vit_prefix_and_decoder_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _QwenOwnerToy()
    wrapper = nn.Module()
    wrapper.model = owner
    wrapper.lm_head = nn.Linear(2, 8, bias=False)
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_factory.assert_qwen_runtime_structure",
        lambda _owner, _config: None,
    )
    policy = QwenOuterTrainabilityConfig(
        mode="partial",
        vision_freeze_first_blocks=13,
        decoder_train_last_layers=8,
        train_vision_patch_embed=False,
        train_main_merger=True,
        train_deepstack_mergers=True,
        train_language_model_norm=True,
        train_input_embeddings=False,
        train_lm_head=False,
    )

    audit = configure_qwen_outer_trainability(wrapper, load_config(), policy)

    assert audit.mode == "partial"
    assert audit.frozen_vision_block_indices == tuple(range(13))
    assert audit.trainable_vision_block_indices == tuple(range(13, 27))
    assert audit.frozen_decoder_layer_indices == tuple(range(28))
    assert audit.trainable_decoder_layer_indices == tuple(range(28, 36))
    assert not audit.vision_patch_embed_trainable
    assert audit.main_merger_trainable
    assert audit.deepstack_mergers_trainable
    assert audit.language_model_norm_trainable
    assert not audit.input_embeddings_trainable
    assert not audit.lm_head_trainable
    assert audit.trainable_parameters < audit.total_parameters
    assert all(
        not parameter.requires_grad
        for block in owner.visual.blocks[:13]
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in owner.visual.blocks[13:]
        for parameter in block.parameters()
    )
    assert all(
        not parameter.requires_grad
        for layer in owner.language_model.layers[:28]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in owner.language_model.layers[28:]
        for parameter in layer.parameters()
    )


def test_frozen_qwen_trainability_is_bitwise_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _QwenOwnerToy()
    wrapper = nn.Module()
    wrapper.model = owner
    wrapper.lm_head = nn.Linear(2, 8, bias=False)
    before = {name: value.detach().clone() for name, value in wrapper.state_dict().items()}
    monkeypatch.setattr(
        "ttt_svcbench_qwen.production_factory.assert_qwen_runtime_structure",
        lambda _owner, _config: None,
    )
    policy = QwenOuterTrainabilityConfig(
        mode="frozen",
        vision_freeze_first_blocks=27,
        decoder_train_last_layers=0,
        train_vision_patch_embed=False,
        train_main_merger=False,
        train_deepstack_mergers=False,
        train_language_model_norm=False,
        train_input_embeddings=False,
        train_lm_head=False,
    )

    audit = configure_qwen_outer_trainability(wrapper, load_config(), policy)

    assert audit.mode == "frozen"
    assert audit.trainable_parameters == 0
    assert audit.frozen_vision_block_indices == tuple(range(27))
    assert audit.frozen_decoder_layer_indices == tuple(range(36))
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())
    assert all(torch.equal(before[name], value) for name, value in wrapper.state_dict().items())


def test_a2_weight_initialization_is_strict_and_excludes_runtime_state(tmp_path: Path) -> None:
    torch.manual_seed(41)
    source = _OuterToy()
    checkpoint = tmp_path / "a2-final"
    checkpoint.mkdir()
    save_file(
        {name: value.detach().clone() for name, value in source.state_dict().items()},
        str(checkpoint / "model.safetensors"),
    )
    torch.manual_seed(42)
    target = _OuterToy()

    audit = initialize_outer_model_from_a2(target, checkpoint)

    assert audit.format == "safetensors"
    assert audit.tensor_count == len(target.state_dict())
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.state_dict()
    )

    class _BadOuter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("visual_cache", torch.ones(1))

    with pytest.raises(ValueError, match="transient/hard runtime"):
        audit_outer_checkpoint_boundary(_BadOuter())


@pytest.mark.parametrize("checkpoint_format", ("single", "sharded"))
def test_legacy_a2_to_a5_profile_allows_removed_modules_and_new_associative_state(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    target = _GroupedOuterToy(nn.Linear(4, 4), associative_trainable=True)
    with torch.no_grad():
        target.fast_adapter.p_context.weight.fill_(1.0)
        if target.fast_adapter.p_context.bias is not None:
            target.fast_adapter.p_context.bias.fill_(1.0)
    state = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if ".p_context." not in name and "associative_contract_version" not in name
    }
    state["fast_adapter.p_value.weight"] = torch.zeros(1)
    state["fast_adapter.p_value.bias"] = torch.zeros(1)
    state["fast_adapter.predictor.weight"] = torch.zeros(1)
    state["fast_adapter.predictor.bias"] = torch.zeros(1)
    checkpoint = tmp_path / "a2-final"
    checkpoint.mkdir()
    if checkpoint_format == "single":
        save_file(state, checkpoint / "model.safetensors")
    else:
        items = tuple(sorted(state.items()))
        split = len(items) // 2
        shards = (items[:split], items[split:])
        weight_map: dict[str, str] = {}
        for index, shard in enumerate(shards, start=1):
            filename = f"model-{index:05d}-of-00002.safetensors"
            save_file(dict(shard), checkpoint / filename)
            weight_map.update({name: filename for name, _value in shard})
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )

    audit = initialize_outer_model_from_a2(target, checkpoint)

    assert audit.format == (
        "safetensors" if checkpoint_format == "single" else "sharded_safetensors"
    )
    assert not audit.missing_keys
    assert not audit.unexpected_keys
    assert torch.count_nonzero(target.fast_adapter.p_context.weight) > 0
    _reset_a2_to_a5_associative(target)
    assert torch.count_nonzero(target.fast_adapter.p_context.weight) == 0
    if target.fast_adapter.p_context.bias is not None:
        assert torch.count_nonzero(target.fast_adapter.p_context.bias) == 0


@pytest.mark.parametrize("invalid_boundary", ("missing", "unexpected"))
def test_legacy_a2_to_a5_profile_rejects_unlisted_boundary_drift(
    tmp_path: Path,
    invalid_boundary: str,
) -> None:
    target = _GroupedOuterToy(nn.Linear(4, 4), associative_trainable=True)
    state = {name: value.detach().clone() for name, value in target.state_dict().items()}
    if invalid_boundary == "missing":
        key = next(
            name
            for name in state
            if ".p_context." not in name and "associative_contract_version" not in name
        )
        del state[key]
    else:
        state["fast_adapter.unlisted_legacy.weight"] = torch.zeros(1)
    checkpoint = tmp_path / "a2-final"
    checkpoint.mkdir()
    save_file(state, checkpoint / "model.safetensors")

    with pytest.raises(ValueError, match="does not exactly match"):
        initialize_outer_model_from_a2(target, checkpoint)


def test_production_runtime_defers_optimizer_and_sampler_to_central_bridge() -> None:
    model = _OuterToy()
    model.p_context.requires_grad_(False)
    runtime = ProductionTrainerRuntime(
        stage=ProductionStage.A2,
        model=model,
        train_dataset=(1,),
        eval_dataset=None,
        data_collator=lambda rows: rows,
        stage_a_loss_step=lambda _model, _inputs: model.backbone.weight.sum(),
    )
    assert runtime.stage is ProductionStage.A2
    assert runtime.optimizer_factory is None
    assert runtime.train_sampler_factory is None


def _resume_run(
    tmp_path: Path,
    name: str,
    *,
    run_config: dict[str, object] | None,
) -> Path:
    run = tmp_path / "runs" / name
    checkpoint = run / "checkpoints" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    if run_config is not None:
        (run / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    return checkpoint


def test_same_stage_resume_is_distinct_from_a2_to_a5_initialization(tmp_path: Path) -> None:
    checkpoint = _resume_run(
        tmp_path,
        "0715_010203_a5",
        run_config={
            "stage": "a5",
            "config_schema_version": 12,
            "associative_ttt_contract": "bank_conditioned_state_write_v2",
        },
    )

    assert resolve_same_stage_resume(str(checkpoint), ProductionStage.A5) == checkpoint
    with pytest.raises(ValueError, match="stage does not match"):
        resolve_same_stage_resume(str(checkpoint), ProductionStage.A2)
    with pytest.raises(ValueError, match="adaptation mode"):
        resolve_same_stage_resume(
            str(checkpoint),
            ProductionStage.A5,
            a5_adaptation_mode="static_w0",
        )

    orphan = _resume_run(tmp_path, "orphan", run_config=None)
    with pytest.raises(FileNotFoundError, match="run_config"):
        resolve_same_stage_resume(str(orphan), ProductionStage.A5)


def test_same_stage_resume_accepts_only_matching_static_w0_mode(tmp_path: Path) -> None:
    checkpoint = _resume_run(
        tmp_path,
        "static-w0",
        run_config={
            "stage": "a5",
            "a5_adaptation_mode": "static_w0",
            "config_schema_version": 12,
            "associative_ttt_contract": "bank_conditioned_state_write_v2",
        },
    )

    assert (
        resolve_same_stage_resume(
            str(checkpoint),
            ProductionStage.A5,
            a5_adaptation_mode="static_w0",
        )
        == checkpoint
    )
    with pytest.raises(ValueError, match="adaptation mode"):
        resolve_same_stage_resume(str(checkpoint), ProductionStage.A5)


def test_same_stage_resume_rejects_legacy_associative_contract(
    tmp_path: Path,
) -> None:
    checkpoint = _resume_run(
        tmp_path,
        "learned-step",
        run_config={
            "stage": "a5",
            "a5_adaptation_mode": "meta_ttt",
            "config_schema_version": 9,
        },
    )

    with pytest.raises(ValueError, match="schema-12 state-write associative"):
        resolve_same_stage_resume(str(checkpoint), ProductionStage.A5)


def test_a5_global_sample_sequence_hash_is_mode_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Record:
        def __init__(self, episode_id: str) -> None:
            self.episode_id = episode_id

    class _Sampler:
        def __init__(self) -> None:
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __iter__(self) -> Iterator[int]:
            return iter((0, 1) if self.epoch == 0 else (1, 0))

    records = (_Record("episode-a"), _Record("episode-b"))
    monkeypatch.setattr(trainer_module, "A5EpisodeRecord", _Record)
    digest, count = trainer_module._a5_global_sample_sequence_sha256(
        records,
        lambda _dataset, _rank, _world_size: _Sampler(),
        epoch_count=2.0,
    )
    expected = hashlib.sha256(
        b"0\tepisode-a\n0\tepisode-b\n1\tepisode-b\n1\tepisode-a\n"
    ).hexdigest()

    assert digest == expected
    assert count == 4


def test_deepspeed_segment_backward_steps_only_after_all_segments() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))

    class _Engine:
        def __init__(self) -> None:
            self.backward_values: list[float] = []
            self.boundaries: list[bool] = []
            self.step_calls = 0

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
            self.boundaries.append(is_boundary)

        def backward(self, loss: torch.Tensor) -> None:
            self.backward_values.append(float(loss.detach()))
            loss.backward()

        def step(self) -> None:
            self.step_calls += 1

    engine = _Engine()
    accelerator = SimpleNamespace(
        distributed_type="DistributedType.DEEPSPEED",
        deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
    )
    controller = SegmentBackwardController(accelerator, nn.Linear(1, 1), expected_count=3)

    controller.backward(parameter.square())
    controller.backward(parameter.square() * 2.0)
    assert engine.step_calls == 0
    controller.backward(parameter.square() * 3.0)
    assert engine.step_calls == 0

    controller.finalize()

    assert engine.backward_values == [4.0, 8.0, 12.0]
    assert engine.boundaries == [False, False, True]
    assert engine.step_calls == 1
    assert parameter.grad is not None
    assert float(parameter.grad) == pytest.approx(24.0)
    with pytest.raises(RuntimeError, match="more than once"):
        controller.finalize()


def test_deepspeed_segment_controller_requires_partition_query() -> None:
    class _Engine:
        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            del is_boundary

        @staticmethod
        def backward(loss: torch.Tensor) -> None:
            del loss

        @staticmethod
        def step() -> None:
            return None

    accelerator = SimpleNamespace(
        distributed_type="DistributedType.DEEPSPEED",
        deepspeed_engine_wrapped=SimpleNamespace(engine=_Engine()),
    )
    with pytest.raises(TypeError, match="partition"):
        SegmentBackwardController(accelerator, nn.Linear(1, 1), expected_count=1)


def test_a5_segment_backward_anchors_every_trainable_parameter_on_every_call() -> None:
    class _ConditionalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.always = nn.Parameter(torch.tensor(2.0))
            self.conditional = nn.Parameter(torch.tensor(3.0))
            self.frozen = nn.Parameter(torch.tensor(4.0), requires_grad=False)

    model = _ConditionalModel()

    class _Engine:
        def __init__(self) -> None:
            self.boundaries: list[bool] = []
            self.backward_gradients: list[tuple[float, float, None]] = []

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
            self.boundaries.append(is_boundary)

        def backward(self, loss: torch.Tensor, **kwargs: object) -> None:
            loss.backward(**kwargs)
            assert model.always.grad is not None
            assert model.conditional.grad is not None
            self.backward_gradients.append(
                (
                    float(model.always.grad),
                    float(model.conditional.grad),
                    model.frozen.grad,
                )
            )
            model.always.grad = None
            model.conditional.grad = None

        @staticmethod
        def step() -> None:
            return None

    engine = _Engine()
    controller = SegmentBackwardController(
        SimpleNamespace(
            distributed_type="DistributedType.DEEPSPEED",
            deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
        ),
        model,
        expected_count=2,
    )

    controller.backward(model.always.square())
    controller.backward(model.conditional.square())
    controller.finalize()

    assert engine.boundaries == [False, True]
    assert engine.backward_gradients == [
        (4.0, 0.0, None),
        (0.0, 6.0, None),
    ]


def test_a5_rank_stable_optimizer_anchor_excludes_always_used_qwen_group() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen = nn.Linear(1, 1, bias=False)
            self.state = nn.Linear(1, 1, bias=False)
            self.p_context = nn.Linear(1, 1, bias=False)

    model = _Model()
    gradient_controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen", "state_shared", "associative"),
    )

    class _Engine:
        def __init__(self) -> None:
            self.optimizer = object()

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            assert is_boundary

        @staticmethod
        def backward(loss: torch.Tensor, **kwargs: object) -> None:
            loss.backward(**kwargs)

        @staticmethod
        def step() -> None:
            return None

    controller = SegmentBackwardController(
        SimpleNamespace(
            distributed_type="DistributedType.DEEPSPEED",
            deepspeed_engine_wrapped=SimpleNamespace(engine=_Engine()),
        ),
        model,
        expected_count=1,
        gradient_controller=gradient_controller,
    )

    assert controller._rank_stable_parameters == (
        model.state.weight,
        model.p_context.weight,
    )


@pytest.mark.parametrize("diverge", (False, True))
def test_a5_rank_stable_hook_order_audit_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    diverge: bool,
) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen = nn.Linear(1, 1, bias=False)
            self.state = nn.Linear(1, 1, bias=False)
            self.p_context = nn.Linear(1, 1, bias=False)

    model = _Model()
    gradient_controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen", "state_shared", "associative"),
    )
    step_calls = 0

    class _Engine:
        optimizer = object()

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            assert is_boundary

        @staticmethod
        def backward(loss: torch.Tensor, **kwargs: object) -> None:
            loss.backward(**kwargs)

        @staticmethod
        def step() -> None:
            nonlocal step_calls
            step_calls += 1

    def fake_all_gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
        outputs[0].copy_(value)
        outputs[1].copy_(value if not diverge or value.numel() == 1 else value.flip(0))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    controller = SegmentBackwardController(
        SimpleNamespace(
            distributed_type="DistributedType.DEEPSPEED",
            deepspeed_engine_wrapped=SimpleNamespace(engine=_Engine()),
        ),
        model,
        expected_count=1,
        gradient_controller=gradient_controller,
    )

    if diverge:
        with pytest.raises(RuntimeError, match="hook order diverged"):
            controller.backward(model.qwen.weight.square().sum())
        assert controller.backward_count == 0
    else:
        controller.backward(model.qwen.weight.square().sum())
        assert controller.backward_count == 1

    assert step_calls == 0


def test_a5_zero1_rank_audit_allows_order_drift_with_identical_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen = nn.Linear(1, 1, bias=False)
            self.state = nn.Linear(1, 1, bias=False)
            self.p_context = nn.Linear(1, 1, bias=False)

    model = _Model()
    gradient_controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen", "state_shared", "associative"),
    )

    class _Engine:
        optimizer = object()

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return False

        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            assert is_boundary

        @staticmethod
        def backward(loss: torch.Tensor, **kwargs: object) -> None:
            loss.backward(**kwargs)

        @staticmethod
        def step() -> None:
            return None

    def fake_all_gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
        outputs[0].copy_(value)
        outputs[1].copy_(value if value.numel() == 1 else value.flip(0))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    controller = SegmentBackwardController(
        SimpleNamespace(
            distributed_type="DistributedType.DEEPSPEED",
            deepspeed_engine_wrapped=SimpleNamespace(engine=_Engine()),
        ),
        model,
        expected_count=1,
        gradient_controller=gradient_controller,
    )

    controller.backward(model.qwen.weight.square().sum())

    assert controller.backward_count == 1
    assert controller._requires_exact_rank_hook_order is False


def test_a2_controlled_wrapper_clips_only_at_the_final_ga_boundary() -> None:
    events: list[object] = []

    class _GradientController:
        def apply_deepspeed(self, optimizer: object) -> None:
            events.append(("clip", optimizer))

    class _Engine:
        optimizer = object()

        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            events.append(("boundary", is_boundary))

        @staticmethod
        def backward(loss: torch.Tensor, **_kwargs: object) -> None:
            events.append(("backward", float(loss)))

        @staticmethod
        def step() -> None:
            events.append("step")

        @staticmethod
        def get_global_grad_norm() -> float:
            return 1.0

    engine = _Engine()
    wrapper = _ControlledDeepSpeedEngineWrapper(
        engine,
        _GradientController(),  # type: ignore[arg-type]
    )

    wrapper.backward(torch.tensor(1.0), sync_gradients=False)
    wrapper.backward(torch.tensor(2.0), sync_gradients=True)

    assert events == [
        ("boundary", False),
        ("backward", 1.0),
        ("boundary", True),
        ("backward", 2.0),
        ("clip", engine.optimizer),
        "step",
    ]


def test_a2_compute_loss_sanitizes_middle_ga_microbatch_without_dropping_backward() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    factors = iter((1.0, float("nan"), 2.0))

    class _Step:
        def __call__(self, _model: nn.Module, _inputs: object) -> torch.Tensor:
            return parameter * next(factors)

    gradient_controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )
    owner = SimpleNamespace(
        ttt_runtime=SimpleNamespace(
            stage=ProductionStage.A2,
            stage_a_loss_step=_Step(),
            gradient_controller=gradient_controller,
        )
    )
    backward_count = 0
    for _ in range(3):
        loss = TTTQwenTrainerMixin.compute_loss(owner, nn.Linear(1, 1), {})
        assert torch.isfinite(loss)
        loss.backward()
        backward_count += 1

    assert backward_count == 3
    assert parameter.grad is not None


def test_a5_segment_controller_clips_after_all_backward_calls_before_step() -> None:
    events: list[str] = []

    class _GradientController:
        def apply_deepspeed(self, _optimizer: object) -> None:
            events.append("clip")

    class _Engine:
        optimizer = object()

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        @staticmethod
        def set_gradient_accumulation_boundary(*, is_boundary: bool) -> None:
            events.append(f"boundary:{is_boundary}")

        @staticmethod
        def backward(_loss: torch.Tensor, **_kwargs: object) -> None:
            events.append("backward")

        @staticmethod
        def step() -> None:
            events.append("step")

    engine = _Engine()
    accelerator = SimpleNamespace(
        distributed_type="DistributedType.DEEPSPEED",
        deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
    )
    controller = SegmentBackwardController(
        accelerator,
        nn.Linear(1, 1),
        expected_count=2,
        gradient_controller=_GradientController(),  # type: ignore[arg-type]
    )

    controller.backward(torch.tensor(1.0, requires_grad=True))
    controller.backward(torch.tensor(2.0, requires_grad=True))
    controller.finalize()

    assert events == [
        "boundary:False",
        "backward",
        "boundary:True",
        "backward",
        "clip",
        "step",
    ]


def test_a5_nonfinite_segment_preserves_backward_parity_and_skips_episode_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameter = nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0e-4, "group_name": "associative"}]
    )

    class _Zero:
        def __init__(self) -> None:
            self.optimizer = optimizer
            self.averaged_gradients = {0: [parameter.grad]}
            self.params_in_partition = [[parameter]]
            self.real_dp_process_group = [None]
            self.loss_scale = 1.0
            self.partition_gradients = True
            self.clip_grad = 0.0

        @staticmethod
        def get_grad_norm_direct(gradients: list[torch.Tensor], _params: object) -> torch.Tensor:
            return (
                torch.stack([gradient.double().square().sum() for gradient in gradients])
                .sum()
                .sqrt()
            )

        def has_overflow(self, *, partition_gradients: bool) -> bool:
            assert partition_gradients
            return not bool(torch.isfinite(parameter.grad).all())

    zero = _Zero()

    class _Engine:
        def __init__(self) -> None:
            self.optimizer = zero
            self.boundaries: list[bool] = []
            self.backward_calls = 0
            self.step_calls = 0
            self.scheduler_steps = 0

        @staticmethod
        def zero_optimization_partition_gradients() -> bool:
            return True

        def set_gradient_accumulation_boundary(self, *, is_boundary: bool) -> None:
            self.boundaries.append(is_boundary)

        def backward(self, loss: torch.Tensor, **kwargs: object) -> None:
            loss.backward(**kwargs)
            self.backward_calls += 1

        def step(self) -> None:
            self.step_calls += 1
            if not zero.has_overflow(partition_gradients=True):
                optimizer.step()
                self.scheduler_steps += 1

    engine = _Engine()
    gradient_controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("associative",),
    )
    controller = SegmentBackwardController(
        SimpleNamespace(
            distributed_type="DistributedType.DEEPSPEED",
            deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
        ),
        nn.Linear(1, 1),
        expected_count=3,
        gradient_controller=gradient_controller,
    )
    before = parameter.detach().clone()

    controller.backward(parameter * 1.0)
    controller.backward(parameter * float("nan"))
    controller.backward(parameter * 2.0)
    with pytest.warns(RuntimeWarning, match="A5 backward 1"):
        controller.finalize()

    assert engine.boundaries == [False, False, True]
    assert engine.backward_calls == 3
    assert engine.step_calls == 1
    assert engine.scheduler_steps == 0
    assert torch.equal(parameter.detach(), before)
    assert gradient_controller.last_audit is not None
    assert gradient_controller.last_audit.skipped_update_count == 1
    assert gradient_controller.last_audit.nonfinite_loss_sources == ("A5 backward 1",)


def test_atomic_final_checkpoint_validation_requires_model_and_resume_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / ".final-checkpoint.incomplete"
    resume = checkpoint / "resume_state"
    resume.mkdir(parents=True)
    save_file({"weight": torch.ones(1)}, str(checkpoint / "model.safetensors"))
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    (resume / "random_states_0.pkl").write_bytes(b"state")

    _validate_checkpoint_tree(checkpoint)

    (resume / "random_states_0.pkl").unlink()
    with pytest.raises(RuntimeError, match="resume state"):
        _validate_checkpoint_tree(checkpoint)


def _write_standard_checkpoint(
    checkpoint: Path,
    *,
    global_step: int,
    max_steps: int,
    epoch: float,
) -> None:
    checkpoint.mkdir()
    save_file({"weight": torch.ones(1)}, str(checkpoint / "model.safetensors"))
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "max_steps": max_steps, "epoch": epoch}),
        encoding="utf-8",
    )


def test_epoch_two_four_checkpoint_policy_publishes_two_resumable_checkpoints(
    tmp_path: Path,
) -> None:
    _write_standard_checkpoint(
        tmp_path / "checkpoint-464", global_step=464, max_steps=928, epoch=2.0
    )
    _write_standard_checkpoint(
        tmp_path / "checkpoint-928", global_step=928, max_steps=928, epoch=4.0
    )

    published = _publish_epoch_two_four_checkpoints(tmp_path)

    assert published == {
        2: tmp_path / "epoch-2-checkpoint",
        4: tmp_path / "epoch-4-checkpoint",
    }
    assert all(path.is_dir() for path in published.values())
    assert not tuple(tmp_path.glob("checkpoint-*"))


def test_checkpoint_policy_environment_defaults_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TTT_CHECKPOINT_POLICY", raising=False)
    assert _checkpoint_policy_from_environment() is CheckpointPolicy.ATOMIC_FINAL_ONLY
    monkeypatch.setenv("TTT_CHECKPOINT_POLICY", "epoch_2_and_epoch_4")
    assert _checkpoint_policy_from_environment() is CheckpointPolicy.EPOCH_2_AND_EPOCH_4
    monkeypatch.setenv("TTT_CHECKPOINT_POLICY", "unknown")
    with pytest.raises(ValueError, match="TTT_CHECKPOINT_POLICY"):
        _checkpoint_policy_from_environment()


def test_explicit_smoke_disables_all_periodic_checkpoints() -> None:
    class _Strategy(StrEnum):
        STEPS = "steps"
        NO = "no"

    arguments = SimpleNamespace(save_strategy=_Strategy.STEPS, save_steps=0.5)

    _disable_smoke_checkpoints(arguments)

    assert arguments.save_strategy is _Strategy.NO
    assert arguments.save_steps == 0


@pytest.mark.parametrize(
    ("stage", "adaptation_mode", "associative_trainable", "expected_lrs"),
    [
        (
            ProductionStage.A2,
            "meta_ttt",
            False,
            {
                "qwen": 1.0e-5,
                "state_shared": 1.0e-4,
                "state_task": 1.0e-4,
                "state_router_time": 1.0e-4,
                "state_retrieval": 1.0e-4,
                "w0": 1.0e-4,
            },
        ),
        (
            ProductionStage.A5,
            "meta_ttt",
            True,
            {
                "qwen": 5.0e-6,
                "fast_slow": 5.0e-5,
                "state_shared": 5.0e-5,
                "state_task": 5.0e-5,
                "state_router_time": 5.0e-5,
                "state_retrieval": 5.0e-5,
                "w0": 5.0e-5,
                "associative": 5.0e-5,
            },
        ),
        (
            ProductionStage.A5,
            "static_w0",
            False,
            {
                "qwen": 5.0e-6,
                "fast_slow": 5.0e-5,
                "state_shared": 5.0e-5,
                "state_task": 5.0e-5,
                "state_router_time": 5.0e-5,
                "state_retrieval": 5.0e-5,
                "w0": 5.0e-5,
            },
        ),
    ],
)
def test_central_outer_optimizer_has_exact_stage_groups(
    tmp_path: Path,
    stage: ProductionStage,
    adaptation_mode: str,
    associative_trainable: bool,
    expected_lrs: dict[str, float],
) -> None:
    bundle, model = _grouped_bundle(
        tmp_path,
        load_config(),
        adaptation_mode=adaptation_mode,
        associative_trainable=associative_trainable,
    )

    optimizer = make_production_outer_optimizer_factory(
        bundle,
        stage,
        a5_adaptation_mode=adaptation_mode,
    )(model)

    actual_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert actual_lrs == expected_lrs
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}


def test_canonical_a5_builds_equal_budget_production_optimizer(
    tmp_path: Path,
) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)

    optimizer = make_production_outer_optimizer_factory(
        bundle,
        ProductionStage.A5,
    )(model)

    groups = {str(group["group_name"]): group for group in optimizer.param_groups}
    caps = project.outer_gradient_control.max_grad_norm
    assert "step_controller" not in groups
    assert float(groups["fast_slow"]["lr"]) * float(caps.fast_slow) == pytest.approx(5.0e-6)
    assert float(groups["w0"]["lr"]) * float(caps.w0) == pytest.approx(5.0e-6)
    assert float(groups["associative"]["lr"]) * float(caps.associative) == pytest.approx(5.0e-6)


def test_warmup_optimizer_excludes_qwen_and_updates_fast_state_groups(
    tmp_path: Path,
) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)
    bundle.model.requires_grad_(False)
    optimizer = make_production_outer_optimizer_factory(
        bundle,
        ProductionStage.A5,
        a5_phase="fast_state_warmup",
    )(model)
    groups = {str(group["group_name"]): group for group in optimizer.param_groups}
    delta_auditor = trainer_module._A5ParameterGroupStepAuditor(
        model,
        delta_audit_steps=3,
        group_names=("fast_slow",),
    )

    assert "qwen" not in groups
    assert {id(parameter) for parameter in delta_auditor.parameters["fast_slow"]} == {
        id(parameter) for parameter in model.fast_adapter.collect_slow_parameters()
    }
    assert {name: float(group["lr"]) for name, group in groups.items()} == {
        "fast_slow": 5.0e-5,
        "state_shared": 1.0e-5,
        "state_task": 1.0e-5,
        "state_router_time": 1.0e-5,
        "state_retrieval": 1.0e-5,
        "w0": 5.0e-5,
        "associative": 5.0e-5,
    }
    qwen_before = {
        name: value.detach().clone() for name, value in bundle.model.state_dict().items()
    }
    representatives = {name: group["params"][0] for name, group in groups.items()}
    representative_before = {
        name: parameter.detach().clone() for name, parameter in representatives.items()
    }
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.float().mean() for parameter in representatives.values())
        loss.backward()
        optimizer.step()

    assert all(
        torch.equal(qwen_before[name], value) for name, value in bundle.model.state_dict().items()
    )
    assert all(
        not torch.equal(representative_before[name], parameter)
        for name, parameter in representatives.items()
    )


def test_warmup_qwen_bitwise_baseline_starts_after_framework_prepare() -> None:
    qwen = nn.Sequential(nn.Linear(8, 8, bias=False), nn.LayerNorm(8))
    qwen.register_buffer("position_counter", torch.arange(4, dtype=torch.int64))
    qwen.requires_grad_(False)
    auditor = trainer_module._WarmupQwenBitwiseAuditor(qwen)

    # A framework conversion before the first Trainer step belongs outside the
    # frozen-training interval.
    qwen.to(dtype=torch.bfloat16)
    auditor.capture_post_prepare_baseline(global_step=0)
    audit = auditor.finalize(device=torch.device("cpu"))

    assert audit.baseline_stage == "post_deepspeed_prepare_pre_first_training_step"
    assert audit.baseline_global_step == 0
    assert audit.baseline_sha256 == audit.final_sha256
    assert audit.changed_tensor_count == 0
    assert audit.local_unchanged
    assert audit.all_ranks_unchanged


def test_warmup_qwen_bitwise_audit_reports_parameter_and_buffer_drift(
    tmp_path: Path,
) -> None:
    qwen = nn.Linear(8, 8, bias=False)
    qwen.register_buffer("state_counter", torch.zeros(2), persistent=False)
    qwen.requires_grad_(False)
    auditor = trainer_module._WarmupQwenBitwiseAuditor(qwen)
    auditor.capture_post_prepare_baseline(global_step=0)

    with torch.no_grad():
        qwen.weight[0, 0].add_(1.0)
        qwen.state_counter[0].add_(1.0)
    audit = auditor.finalize(device=torch.device("cpu"))
    changed = {change.name: change for change in audit.changed_tensors}

    assert not audit.local_unchanged
    assert not audit.all_ranks_unchanged
    assert audit.changed_parameter_count == 1
    assert audit.changed_buffer_count == 1
    assert changed["weight"].change_type == "content"
    assert changed["state_counter"].change_type == "content"
    rank_path, canonical_path = trainer_module._write_warmup_qwen_bitwise_audit(
        artifact_root=tmp_path,
        audit=audit,
    )
    assert rank_path.is_file()
    assert canonical_path == tmp_path / "qwen_bitwise_audit.json"
    persisted = json.loads(rank_path.read_text(encoding="utf-8"))
    assert persisted["changed_tensor_count"] == 2
    assert {item["name"] for item in persisted["changed_tensors"]} == {
        "weight",
        "state_counter",
    }
    with pytest.raises(RuntimeError, match="post-DeepSpeed"):
        trainer_module._assert_warmup_qwen_bitwise_unchanged(audit)


def test_warmup_qwen_bitwise_auditor_rejects_late_or_trainable_baseline() -> None:
    trainable = nn.Linear(4, 4)
    auditor = trainer_module._WarmupQwenBitwiseAuditor(trainable)
    with pytest.raises(RuntimeError, match="trainable parameters"):
        auditor.capture_post_prepare_baseline(global_step=0)

    frozen = nn.Linear(4, 4).requires_grad_(False)
    late = trainer_module._WarmupQwenBitwiseAuditor(frozen)
    with pytest.raises(RuntimeError, match="before optimizer step one"):
        late.capture_post_prepare_baseline(global_step=1)


def test_warmup_bundle_is_non_qwen_atomic_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    source = {
        "a2_checkpoint_sha256": "a2-hash",
        "code_commit": "commit",
        "code_dirty": False,
        "project_config_sha256": "config-hash",
        "dataset_manifest_sha256": "manifest-hash",
        "seed": 42,
        "data_seed": 42,
    }
    monkeypatch.setattr(
        trainer_module,
        "_warmup_source_manifest",
        lambda **_kwargs: dict(source),
    )
    qwen_sha256 = trainer_module._module_bitwise_sha256(bundle.model)
    expected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name in trainer_module._warmup_bundle_allowlist(model, bundle.model)
    }

    bundle_path, manifest = trainer_module._publish_warmup_bundle(
        model=model,
        qwen_model=bundle.model,
        backbone=bundle,
        artifact_root=artifact_root,
        global_step=128,
        qwen_sha256=qwen_sha256,
    )

    assert bundle_path.name == "a5_warmup_bundle"
    assert not (artifact_root / ".a5_warmup_bundle.incomplete").exists()
    assert not any(name.startswith("qwen.") for name in manifest["parameter_allowlist"])
    assert not any("transient_w_t" in name for name in manifest["parameter_allowlist"])
    assert manifest["qwen_bitwise_sha256"] == qwen_sha256
    main_config = bundle.ttt_config.model_copy(update={"warmup_bundle": str(bundle_path)})
    main_bundle = replace(bundle, ttt_config=main_config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in expected:
                parameter.zero_()
        for name, buffer in model.named_buffers():
            if name in expected:
                buffer.zero_()
    audit = trainer_module._load_warmup_bundle(
        model=model,
        qwen_model=bundle.model,
        backbone=main_bundle,
    )
    assert audit["tensor_count"] == len(expected)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())

    monkeypatch.setattr(
        trainer_module,
        "_warmup_source_manifest",
        lambda **_kwargs: {**source, "a2_checkpoint_sha256": "wrong-a2"},
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        trainer_module._load_warmup_bundle(
            model=model,
            qwen_model=bundle.model,
            backbone=main_bundle,
        )


def test_warmup_bundle_can_publish_prepared_cpu_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(
        trainer_module,
        "_warmup_source_manifest",
        lambda **_kwargs: {
            "a2_checkpoint_sha256": "a2-hash",
            "code_commit": "commit",
            "code_dirty": False,
            "project_config_sha256": "config-hash",
            "dataset_manifest_sha256": "manifest-hash",
            "seed": 42,
            "data_seed": 42,
        },
    )
    prepared = trainer_module._prepare_warmup_bundle_tensors(model, bundle.model)
    allowlist, tensors = prepared

    assert tuple(sorted(tensors)) == allowlist
    assert all(value.device.type == "cpu" for value in tensors.values())
    assert not any(name.startswith("qwen.") for name in allowlist)
    bundle_path, manifest = trainer_module._publish_warmup_bundle(
        model=model,
        qwen_model=bundle.model,
        backbone=bundle,
        artifact_root=artifact_root,
        global_step=128,
        qwen_sha256=trainer_module._module_bitwise_sha256(bundle.model),
        prepared_bundle=prepared,
    )

    assert bundle_path.is_dir()
    assert manifest["parameter_allowlist"] == list(allowlist)
    with pytest.raises(ValueError, match="keys do not match"):
        trainer_module._publish_warmup_bundle(
            model=model,
            qwen_model=bundle.model,
            backbone=bundle,
            artifact_root=tmp_path / "other-artifacts",
            global_step=128,
            qwen_sha256="qwen-hash",
            prepared_bundle=(allowlist, {**tensors, "unexpected": torch.ones(1)}),
        )


def test_all_ranks_true_uses_local_value_without_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert trainer_module._all_ranks_true(True, device=torch.device("cpu"))
    assert not trainer_module._all_ranks_true(False, device=torch.device("cpu"))


def test_optimizer_rejects_noncanonical_budget_drift(tmp_path: Path) -> None:
    base = load_config()
    wrong_project = base.model_copy(
        update={
            "a5": base.a5.model_copy(
                update={
                    "optimizer": base.a5.optimizer.model_copy(
                        update={"associative_learning_rate": 1.0e-4}
                    )
                }
            )
        }
    )
    bundle, model = _grouped_bundle(tmp_path, wrong_project)

    with pytest.raises(ValueError, match="must remain aligned"):
        make_production_outer_optimizer_factory(bundle, ProductionStage.A5)(model)


def test_outer_optimizer_rejects_removed_step_controller_parameters(
    tmp_path: Path,
) -> None:
    bundle, model = _grouped_bundle(tmp_path, load_config())
    model.step_controller = nn.Linear(7, 1)

    with pytest.raises(ValueError, match="step-controller parameters were removed"):
        make_production_outer_optimizer_factory(
            bundle,
            ProductionStage.A5,
            a5_adaptation_mode="meta_ttt",
        )(model)
