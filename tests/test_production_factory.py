from __future__ import annotations

import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import ttt_svcbench_qwen.llamafactory_trainer as trainer_module
from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.fast_ttt import ASSOCIATIVE_CONTRACT_VERSION, build_fast_ttt_adapter
from ttt_svcbench_qwen.llamafactory_trainer import (
    ProductionStage,
    _disable_smoke_checkpoints,
    _validate_checkpoint_tree,
    make_production_outer_optimizer_factory,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    LlamaFactorySymbols,
    ProductionTTTConfig,
    initialize_outer_model_from_a2,
    load_training_yaml,
)
from ttt_svcbench_qwen.production_runtime import (
    QueryObservationSpec,
    SupportChunkSpec,
    _llamafactory_uniform_frame_indices,
    _query_chunk_spec,
    _uniform_target_times,
)
from ttt_svcbench_qwen.state_encoder import SpatialObjectEncoder

ROOT = Path(__file__).resolve().parents[1]
A2_YAML = ROOT / "configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml"


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


def _grouped_bundle(
    tmp_path: Path,
    project: ProjectConfig,
    *,
    adaptation_mode: str = "meta_ttt",
    associative_trainable: bool | None = None,
) -> tuple[LlamaFactoryBackboneBundle, nn.Module]:
    qwen = nn.Linear(4, 4)
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
            preprocess_cache_root_env="TTT_PREPROCESS_CACHE_ROOT",
            preprocess_cache_max_gb=200.0,
            preprocess_cache_dtype="float32",
        ),
        symbols=LlamaFactorySymbols(
            get_train_args=lambda *_args, **_kwargs: (),
            load_tokenizer=lambda *_args, **_kwargs: {},
            load_model=lambda *_args, **_kwargs: qwen,
            trainer_base=object,
        ),
    )
    if associative_trainable is None:
        associative_trainable = adaptation_mode == "meta_ttt"
    return bundle, _GroupedOuterToy(qwen, associative_trainable=associative_trainable)


def test_a2_yaml_runs_four_epochs_and_keeps_only_the_final_checkpoint(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")

    native, extension = load_training_yaml(A2_YAML)

    assert native["num_train_epochs"] == 4.0
    assert native["save_strategy"] == "epoch"
    assert "save_steps" not in native
    assert native["save_total_limit"] == 1
    assert native["save_only_model"] is False
    assert native["video_max_pixels"] == 131_072
    assert extension.stage == "a2"


def test_fullprefix256_yaml_matches_qwen_visual_budget_and_dynamic_graph_zero1(
    h200_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")

    native, extension = load_training_yaml(A2_YAML)

    assert native["video_fps"] == 2.0
    assert native["video_maxlen"] == 256
    assert native["cutoff_len"] == 16_384
    assert native["deepspeed"] == "configs/h200/deepspeed_zero1_dynamic_graph.json"
    assert native["per_device_train_batch_size"] == 1
    assert native["gradient_accumulation_steps"] == 4
    assert native["dataloader_num_workers"] == 2
    assert native["dataloader_prefetch_factor"] == 2
    assert native["max_grad_norm"] == 0.0
    assert extension.state_query_visual_mode == "recent_chunk"
    assert extension.state_query_max_frames == 16
    assert extension.answer_query_visual_mode == "causal_prefix"
    assert extension.answer_query_max_frames == 256
    assert extension.query_decode_max_groups == 16
    assert extension.state_query_cache_mode == "inherit"
    assert extension.answer_query_cache_mode == "disabled"
    assert extension.cached_query_roles == frozenset(("state_query",))
    assert extension.visual_cost_mode == "proxy"


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


def test_query_prefix_samples_256_causal_frames(tmp_path: Path) -> None:
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
    # Causal prefix: no sampled frame may sit after the query time.
    assert all(value <= query.query_time for value in targets)
    # A support chunk stays inside its own window, so its last target never
    # crosses the chunk end even when the frame budget is generous.
    support = SupportChunkSpec("support", path, 0.0, 8.0, 16, 8.0)
    support_targets = _uniform_target_times(support, 2.0)
    assert support_targets[0] == 0.0
    assert support_targets[-1] == 8.0
    assert all(0.0 <= value <= 8.0 for value in support_targets)


def test_query_uniform_indices_match_llamafactory_523f801_reference() -> None:
    indices = _llamafactory_uniform_frame_indices(
        total_frames=1_989,
        duration=663.0,
        video_fps=2.0,
        video_maxlen=256,
    )

    reference = tuple(int(value) for value in np.linspace(0, 1_988, 256).astype(np.int32).tolist())
    assert indices == reference


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
            "no_write",
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


def test_canonical_a5_builds_equal_budget_production_optimizer(tmp_path: Path) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)

    optimizer = make_production_outer_optimizer_factory(bundle, ProductionStage.A5)(model)

    groups = {str(group["group_name"]): group for group in optimizer.param_groups}
    caps = project.outer_gradient_control.max_grad_norm
    assert "step_controller" not in groups
    assert float(groups["fast_slow"]["lr"]) * float(caps.fast_slow) == pytest.approx(5.0e-6)
    assert float(groups["w0"]["lr"]) * float(caps.w0) == pytest.approx(5.0e-6)
    assert float(groups["associative"]["lr"]) * float(caps.associative) == pytest.approx(5.0e-6)


def test_warmup_optimizer_trains_only_memory_interface_and_state_groups(
    tmp_path: Path,
) -> None:
    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)
    bundle.model.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen optimizer groups"):
        make_production_outer_optimizer_factory(
            bundle,
            ProductionStage.A5,
            a5_phase="fast_state_warmup",
        )(model)

    trainer_module._configure_fast_state_warmup_trainability(model, bundle.model)
    optimizer = make_production_outer_optimizer_factory(
        bundle,
        ProductionStage.A5,
        a5_phase="fast_state_warmup",
    )(model)
    groups = {str(group["group_name"]): group for group in optimizer.param_groups}

    assert "qwen" not in groups
    assert {name: float(group["lr"]) for name, group in groups.items()} == {
        "state_shared": 1.0e-5,
        "state_task": 1.0e-5,
        "state_router_time": 1.0e-5,
        "state_retrieval": 1.0e-5,
        "associative": 5.0e-5,
    }
    qwen_before = {
        name: value.detach().clone() for name, value in bundle.model.state_dict().items()
    }
    frozen_fast_parameters = (
        *model.fast_adapter.collect_slow_parameters(),
        *model.fast_adapter.collect_meta_fast_parameters(),
    )
    assert all(not parameter.requires_grad for parameter in frozen_fast_parameters)
    representatives = {name: group["params"][0] for name, group in groups.items()}
    representative_before = {
        name: parameter.detach().clone() for name, parameter in representatives.items()
    }
    frozen_before = tuple(
        (parameter, parameter.detach().clone()) for parameter in frozen_fast_parameters
    )
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert not {id(parameter) for parameter in frozen_fast_parameters} & owned
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.float().mean() for parameter in representatives.values())
        loss.backward()
        optimizer.step()

    assert all(
        torch.equal(qwen_before[name], value) for name, value in bundle.model.state_dict().items()
    )
    assert all(torch.equal(before, parameter) for parameter, before in frozen_before)
    assert all(
        not torch.equal(representative_before[name], parameter)
        for name, parameter in representatives.items()
    )


def test_outer_optimizer_rejects_the_transient_per_video_memory(tmp_path: Path) -> None:
    bundle, model = _grouped_bundle(tmp_path, load_config())
    model.register_parameter(
        "m",
        nn.Parameter(torch.zeros((4, 4), dtype=torch.float32)),
    )

    with pytest.raises(ValueError, match="transient per-video memory"):
        make_production_outer_optimizer_factory(
            bundle,
            ProductionStage.A5,
            a5_adaptation_mode="meta_ttt",
        )(model)


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


def test_a2_initialization_requires_the_complete_outer_checkpoint(tmp_path: Path) -> None:
    source = nn.Linear(4, 4)
    checkpoint = tmp_path / "a2"
    checkpoint.mkdir()
    save_file(source.state_dict(), str(checkpoint / "model.safetensors"))
    target = nn.Linear(4, 4)

    initialize_outer_model_from_a2(target, checkpoint)
    assert all(
        torch.equal(source.state_dict()[name], value) for name, value in target.state_dict().items()
    )

    save_file({"weight": source.weight.detach()}, str(checkpoint / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Missing key"):
        initialize_outer_model_from_a2(nn.Linear(4, 4), checkpoint)


def test_explicit_smoke_disables_all_periodic_checkpoints() -> None:
    class _Strategy(StrEnum):
        STEPS = "steps"
        NO = "no"

    arguments = SimpleNamespace(save_strategy=_Strategy.STEPS, save_steps=0.5)

    _disable_smoke_checkpoints(arguments)

    assert arguments.save_strategy is _Strategy.NO
    assert arguments.save_steps == 0


def test_bundle_contract_version_pin_matches_the_runtime_contract() -> None:
    """The trainer's bundle pin is a deliberate duplicate of the runtime version.

    Publish stamps the trainer constant into the bundle manifest and load
    compares against it, while the adapter carries the runtime constant in its
    ``memory_contract_version`` buffer.  If one moves without the other, either
    new bundles record a stale revision or old bundles keep loading across a
    key-semantics change -- both silent.  Pin them together.
    """

    assert (
        trainer_module._WARMUP_BUNDLE_ASSOCIATIVE_CONTRACT_VERSION == ASSOCIATIVE_CONTRACT_VERSION
    )


def test_warmup_bundle_allowlist_excludes_non_persistent_buffers() -> None:
    """A ``persistent=False`` buffer must not break publication.

    ``_grouped_bundle`` builds a synthetic model whose Qwen stand-in is an
    ``nn.Linear``, so no real spatial encoder -- and therefore no non-persistent
    buffer -- ever reached the allowlist in tests.  Production does have one:
    ``SpatialObjectEncoder`` registers ``slot_codes`` with ``persistent=False``,
    its name matches none of ``_WARMUP_BUNDLE_EXCLUDED_TOKENS``, and it is absent
    from ``state_dict``.  That combination failed the allowlist's subset check
    *after* a full 256-step warmup had trained, losing the whole run's product.
    """

    spatial = SpatialObjectEncoder(load_config().spatial_encoder)
    assert "slot_codes" in dict(spatial.named_buffers())
    assert "slot_codes" not in spatial.state_dict()

    class _Carrier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial = spatial
            self.trained = nn.Linear(4, 4)
            self.register_buffer("kept", torch.zeros(2), persistent=True)

    model = _Carrier()
    qwen = nn.Linear(4, 4)
    allowlist = trainer_module._warmup_bundle_allowlist(model, qwen)

    assert "spatial.slot_codes" not in allowlist
    assert "trained.weight" in allowlist
    assert "kept" in allowlist
    assert "spatial.shared_slot_seed" in allowlist
    # Every surviving name must be saveable, which is the property the check owns.
    assert set(allowlist) <= set(model.state_dict())


def test_warmup_bundle_is_non_qwen_atomic_and_fail_closed(tmp_path: Path) -> None:
    """The A2->A5 handoff round-trip: publish, reload, and refuse to overwrite."""

    project = load_config()
    bundle, model = _grouped_bundle(tmp_path, project)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    prepared_allowlist, prepared_tensors = trainer_module._prepare_warmup_bundle_tensors(
        model,
        bundle.model,
    )
    assert tuple(sorted(prepared_tensors)) == prepared_allowlist
    assert all(value.device.type == "cpu" for value in prepared_tensors.values())
    expected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name in prepared_allowlist
    }

    bundle_path, manifest = trainer_module._publish_warmup_bundle(
        model=model,
        qwen_model=bundle.model,
        backbone=bundle,
        artifact_root=artifact_root,
        global_step=project.a5.warmup.max_steps,
    )

    assert bundle_path.name == "a5_warmup_bundle"
    assert not (artifact_root / ".a5_warmup_bundle.incomplete").exists()
    assert not any(name.startswith("qwen.") for name in manifest["parameter_allowlist"])
    assert not any("transient_w_t" in name for name in manifest["parameter_allowlist"])
    assert manifest["parameter_allowlist"] == list(prepared_allowlist)
    assert manifest["optimizer_steps"] == project.a5.warmup.max_steps

    # Publication is atomic and refuses to clobber an existing handoff.
    with pytest.raises(FileExistsError, match="warmup handoff bundle"):
        trainer_module._publish_warmup_bundle(
            model=model,
            qwen_model=bundle.model,
            backbone=bundle,
            artifact_root=artifact_root,
            global_step=project.a5.warmup.max_steps,
        )

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
    assert audit["bundle_sha256"] == manifest["bundle_sha256"]
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())

    manifest_path = bundle_path / "manifest.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["associative_contract_version"] = ASSOCIATIVE_CONTRACT_VERSION - 1
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="associative_contract_version"):
        trainer_module._load_warmup_bundle(
            model=model,
            qwen_model=bundle.model,
            backbone=main_bundle,
        )
