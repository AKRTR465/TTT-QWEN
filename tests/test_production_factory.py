from __future__ import annotations

import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.llamafactory_trainer import (
    ProductionStage,
    ProductionTrainerRuntime,
    SegmentBackwardController,
    make_production_outer_optimizer_factory,
    resolve_same_stage_resume,
)
from ttt_svcbench_qwen.production_factory import (
    LlamaFactoryBackboneBundle,
    LlamaFactoryCheckoutAudit,
    LlamaFactorySymbols,
    ProductionTTTConfig,
    audit_outer_checkpoint_boundary,
    fully_unfreeze_qwen,
    initialize_outer_model_from_a2,
    load_outer_checkpoint,
    load_training_yaml,
)
from ttt_svcbench_qwen.production_runtime import (
    CurrentChunkSpec,
    ProductionOuterModel,
    _decode_uniform_interval,
    _resize_to_pixel_budget,
    _video_pixel_bounds,
)


class _OuterToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.predictor = nn.Linear(4, 4)


class _GroupedOuterToy(nn.Module):
    def __init__(self, qwen: nn.Module, *, predictor_trainable: bool) -> None:
        super().__init__()
        self.qwen = qwen
        self.state = nn.Linear(4, 4)
        self.w0_1 = nn.Parameter(torch.ones(4, 4))
        self.w0_2 = nn.Parameter(torch.ones(4, 4))
        self.predictor = nn.Linear(4, 4)
        self.predictor.requires_grad_(predictor_trainable)


class _QwenOwnerToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.stem = nn.Linear(2, 2)
        self.visual.merger = nn.Linear(2, 2)
        self.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(2, 2)])
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(36)])


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
        torch.equal(target.state_dict()[key], value)
        for key, value in source.state_dict().items()
    )
    torch.save(source.state_dict(), tmp_path / "outer.bin")
    with pytest.raises(ValueError, match="safetensors"):
        load_outer_checkpoint(target, tmp_path / "outer.bin")
    bad = dict(source.state_dict())
    bad["temporal_cache.hidden"] = torch.zeros(1)
    save_file(bad, tmp_path / "bad.safetensors")
    with pytest.raises(ValueError, match="exactly match"):
        load_outer_checkpoint(target, tmp_path / "bad.safetensors")


def test_a2_yaml_runs_four_epochs_and_only_saves_the_final_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("MODEL", "/tmp/qwen3vl8b")
    monkeypatch.setenv("DATASET_DIR", "/tmp/svcbench")
    monkeypatch.setenv("DATASET_NAME", "svcbench_qwen3vl_sft")

    native, extension = load_training_yaml(root / "configs/h200/a2_qwen3vl8b_full_4gpu.yaml")

    assert native["num_train_epochs"] == 4.0
    assert native["save_strategy"] == "no"
    assert "save_steps" not in native
    assert "save_total_limit" not in native
    assert native["save_only_model"] is False
    assert native["video_max_pixels"] == 131_072
    assert extension.stage == "a2"
    assert set(extension.model_dump(exclude_none=True)) == {
        "stage",
        "project_config",
        "dataset_manifest",
            "support_prefetch_depth",
            "support_decode_coalesce",
            "support_materialization",
            "prepared_episode_max_bytes",
            "support_visual_batch_size",
            "query_encoder_reuse",
            "preprocess_cache_mode",
            "preprocess_cache_miss_policy",
        "preprocess_cache_root_env",
        "preprocess_cache_max_gb",
            "preprocess_cache_dtype",
            "visual_cost_mode",
            "runtime_trace_mode",
            "segment_prefetch_depth",
        }


def test_a2_uses_dynamic_graph_safe_zero1_profile() -> None:
    root = Path(__file__).parents[1]
    for yaml_name in (
        "a2_qwen3vl8b_full_4gpu.yaml",
        "a2_qwen3vl8b_full_4gpu_120g.yaml",
    ):
        text = (root / "configs/h200" / yaml_name).read_text(encoding="utf-8")
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


def test_h200_a2_entry_defaults_to_bounded_dynamic_visual_tokens() -> None:
    root = Path(__file__).parents[1]
    launcher = (root / "scripts/h200/train_a2_a5.sh").read_text(encoding="utf-8")
    assert 'YAML="${YAML:-$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_full_4gpu.yaml}"' in launcher


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
    spec = CurrentChunkSpec(
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


def test_short_interval_decoder_streams_once_instead_of_seeking_every_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "short.mp4"
    path.touch()
    calls: list[str] = []

    def decode(name: str):
        def inner(_spec: CurrentChunkSpec, targets: list[float]):
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
    short = CurrentChunkSpec("short", path, 10.0, 18.0, 16, 20.0)
    long = CurrentChunkSpec("long", path, 10.0, 42.0, 8, 50.0)

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
    outer = ProductionOuterModel(nn.Linear(1, 1), nn.Linear(1, 1), qwen)
    outer.gradient_checkpointing_enable({"use_reentrant": True, "preserve_rng_state": False})

    assert qwen.calls == [{"use_reentrant": False, "preserve_rng_state": False}]


def test_training_yaml_expands_required_environment_and_keeps_a5_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("SVCBENCH_DATASET_MANIFEST", "/tmp/dataset_manifest.json")
    monkeypatch.setenv("A2_CHECKPOINT", "/tmp/a2-final")
    monkeypatch.setenv("MODEL", "/tmp/qwen3vl8b")
    monkeypatch.setenv("DATASET_DIR", "/tmp/svcbench")
    monkeypatch.setenv("DATASET_NAME", "svcbench_qwen3vl_sft")

    native, extension = load_training_yaml(root / "configs/h200/a5_meta_ttt_k8_4gpu.yaml")

    assert native["resume_from_checkpoint"] is None
    assert native["output_dir"] == "/tmp/output"
    assert extension.initialize_from_a2_checkpoint == "/tmp/a2-final"
    assert extension.stage == "a5"

    monkeypatch.delenv("A2_CHECKPOINT")
    with pytest.raises(ValueError, match="unresolved environment variables"):
        load_training_yaml(root / "configs/h200/a5_meta_ttt_k8_4gpu.yaml")


def test_training_yaml_rejects_unknown_extension_keys_and_invalid_stage_checkpoint(
    tmp_path: Path,
) -> None:
    base = {
        "stage": "a2",
        "project_config": "configs/model_state_ttt_8b.yaml",
        "dataset_manifest": "manifest.json",
        "support_prefetch_depth": 2,
        "support_decode_coalesce": True,
        "preprocess_cache_mode": "read_write",
        "preprocess_cache_miss_policy": "decode",
        "preprocess_cache_root_env": "TTT_PREPROCESS_CACHE_ROOT",
        "preprocess_cache_max_gb": 200.0,
        "preprocess_cache_dtype": "float32",
    }

    def write(extension: dict[str, object]) -> Path:
        path = tmp_path / "training.yaml"
        lines = ["model_name_or_path: model", "ttt_qwen:"]
        lines.extend(f"  {key}: {json.dumps(value)}" for key, value in extension.items())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_training_yaml(write({**base, "inner_learning_rate": 1.0e-4}))
    with pytest.raises(ValueError, match="A2 must not initialize"):
        load_training_yaml(write({**base, "initialize_from_a2_checkpoint": "checkpoint"}))
    with pytest.raises(ValueError, match="A5 requires initialize"):
        load_training_yaml(write({**base, "stage": "a5"}))


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


def test_production_runtime_defers_optimizer_and_sampler_to_central_bridge() -> None:
    model = _OuterToy()
    model.predictor.requires_grad_(False)
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


def test_same_stage_resume_is_distinct_from_a2_to_a5_initialization(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "0715_010203_a5"
    checkpoint = run / "checkpoints" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (run / "run_config.json").write_text('{"stage": "a5"}', encoding="utf-8")

    assert resolve_same_stage_resume(str(checkpoint), ProductionStage.A5) == checkpoint
    with pytest.raises(ValueError, match="stage does not match"):
        resolve_same_stage_resume(str(checkpoint), ProductionStage.A2)

    orphan = tmp_path / "checkpoint-orphan"
    orphan.mkdir()
    (orphan / "trainer_state.json").write_text("{}", encoding="utf-8")
    (orphan / "scheduler.pt").write_bytes(b"scheduler")
    (orphan / "optimizer.pt").write_bytes(b"optimizer")
    with pytest.raises(FileNotFoundError, match="run_config"):
        resolve_same_stage_resume(str(orphan), ProductionStage.A5)


def test_deepspeed_segment_backward_steps_only_after_all_segments() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))

    class _Engine:
        def __init__(self) -> None:
            self.backward_values: list[float] = []
            self.boundaries: list[bool] = []
            self.step_calls = 0

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


@pytest.mark.parametrize(
    ("stage", "predictor_trainable", "expected_lrs"),
    [
        (ProductionStage.A2, False, {"qwen": 1.0e-5, "state": 1.0e-4, "w0": 1.0e-4}),
        (
            ProductionStage.A5,
            True,
            {"qwen": 5.0e-6, "state": 5.0e-5, "w0": 5.0e-5, "predictor": 5.0e-5},
        ),
    ],
)
def test_central_outer_optimizer_has_exact_stage_groups(
    tmp_path: Path,
    stage: ProductionStage,
    predictor_trainable: bool,
    expected_lrs: dict[str, float],
) -> None:
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
        ),
        finetuning_args=object(),
        generating_args=object(),
        project_config=load_config(),
        ttt_config=ProductionTTTConfig(
            stage="a5",
            project_config="configs/model_state_ttt_8b.yaml",
            dataset_manifest="manifest.json",
            initialize_from_a2_checkpoint="a2-final",
            support_prefetch_depth=2,
            support_decode_coalesce=True,
            preprocess_cache_mode="read_write",
            preprocess_cache_miss_policy="decode",
            preprocess_cache_root_env="TTT_PREPROCESS_CACHE_ROOT",
            preprocess_cache_max_gb=200.0,
            preprocess_cache_dtype="float32",
        ),
        symbols=symbols,
    )
    model = _GroupedOuterToy(qwen, predictor_trainable=predictor_trainable)

    optimizer = make_production_outer_optimizer_factory(bundle, stage)(model)

    actual_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert actual_lrs == expected_lrs
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
