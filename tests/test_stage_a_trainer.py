from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from ttt_svcbench_qwen.config import StageAVariant, load_config
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    O1StateTarget,
    OperatorLossInput,
    ReaderCountMetricInput,
    StateLossInput,
)
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    StageATargetBatch,
    TargetProvenance,
)
from ttt_svcbench_qwen.trainer import (
    REQUIRED_A2_METRICS,
    StageAExecutionAudit,
    StageAForwardOutput,
    StageASupervisionBatch,
    StageATrainingBatch,
    TrainingStage,
    TrainingStepOutput,
    build_balanced_stage_a_indices,
    build_trainer,
    compute_stage_a_losses,
    load_stage_a_checkpoint,
)


class _ToyStageAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.component_modules = nn.ModuleDict(
            {
                "fast_adapter": nn.Linear(4, 4),
                "query_encoder": nn.Linear(4, 9),
                "spatial_encoder": nn.Linear(4, 4),
                "temporal_encoder": nn.Linear(4, 4),
                "observation_heads": nn.Linear(4, 6),
                "state_bank": nn.Linear(4, 4),
                "resampler": nn.Linear(4, 5),
                "qwen_prefill": nn.Linear(4, 5),
                "predictor": nn.Linear(4, 4),
            }
        )


class _ToyForward:
    def __init__(self, model: _ToyStageAModel) -> None:
        self.model = model

    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAForwardOutput:
        values = batch.model_inputs
        assert isinstance(values, Tensor)
        modules = self.model.component_modules
        hidden = torch.tanh(modules["fast_adapter"](values))
        spatial = torch.tanh(modules["spatial_encoder"](hidden))
        temporal = torch.tanh(modules["temporal_encoder"](spatial))
        semantic = torch.tanh(modules["state_bank"](temporal))
        answer_row = modules["resampler"](semantic)
        answer_logits = answer_row.unsqueeze(1).expand(-1, 3, -1)
        labels = torch.tensor([[-100, 1, 2]] * values.shape[0], dtype=torch.int64)
        number_mask = torch.tensor([[False, False, True]] * values.shape[0])
        counts = torch.arange(values.shape[0], dtype=torch.int64)
        answer = AnswerLossInput(
            logits=answer_logits,
            labels=labels,
            number_token_mask=number_mask,
            reader_counts=ReaderCountMetricInput(
                counts,
                counts.clone(),
                torch.ones_like(counts).bool(),
            ),
        )
        head_logits = modules["observation_heads"](spatial).unsqueeze(1)
        o1 = O1StateTarget(
            row_indices=torch.arange(values.shape[0]),
            logits=head_logits,
            targets=torch.ones_like(head_logits),
            slot_mask=torch.ones(head_logits.shape[:2], dtype=torch.bool),
        )
        operator = OperatorLossInput(
            logits=modules["query_encoder"](values),
            targets=torch.arange(values.shape[0], dtype=torch.int64) % 9,
            valid_mask=torch.ones(values.shape[0], dtype=torch.bool),
        )
        metrics = tuple(
            (name, None if name == "retrieval/precision" else 0.5) for name in REQUIRED_A2_METRICS
        )
        return StageAForwardOutput(
            answer_loss_input=answer,
            state_loss_input=StateLossInput(
                batch_size=values.shape[0],
                o1=o1,
                operator=operator,
            ),
            audit=StageAExecutionAudit(
                row_count=values.shape[0],
                observed_chunk_count=values.shape[0] * 2,
                hard_state_row_count=values.shape[0],
                query_router_row_count=values.shape[0],
                time_resolver_row_count=values.shape[0],
                retrieval_row_count=values.shape[0],
                reader_result_count=values.shape[0],
                bank_reset_count=values.shape[0],
                bank_write_count=values.shape[0] * 2,
                cache_advance_count=values.shape[0] * 2,
                fsm_rollout_count=2,
            ),
            metrics=metrics,
            failure_cases=("synthetic-reader-llm-disagreement-row-3",),
        )


def _batch(*, leaked: bool = False) -> StageATrainingBatch:
    payloads = tuple(
        {
            "video": Path(f"synthetic-{row}.mp4"),
            "question": f"synthetic question {row}",
            "query_time": 2.0,
            "explicit_time_values": (),
            **({"count": 3} if leaked else {}),
        }
        for row in range(4)
    )
    return StageATrainingBatch(
        runtime_payloads=payloads,
        model_inputs=torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16.0,
        supervision=StageASupervisionBatch(
            answer=AnswerTargetLabels(
                base_labels=torch.tensor([[-100, 1, 2]] * 4),
                base_number_token_mask=torch.tensor([[False, False, True]] * 4),
                target_counts=torch.arange(4, dtype=torch.int64),
                answer_provenance=(TargetProvenance.SYNTHETIC_EXPLICIT,) * 4,
                count_provenance=(TargetProvenance.SYNTHETIC_EXPLICIT,) * 4,
            ),
            state=StageATargetBatch(),
        ),
    )


def _answer_input() -> AnswerLossInput:
    logits = torch.zeros((1, 3, 5), requires_grad=True)
    return AnswerLossInput(
        logits=logits,
        labels=torch.tensor([[-100, 1, 2]]),
        number_token_mask=torch.tensor([[False, False, True]]),
    )


def test_stage_a_loss_has_no_ttt_and_keeps_a1_a2_objectives_exact() -> None:
    answer = _answer_input()
    a1 = compute_stage_a_losses(StageAVariant.A1, answer=answer, state=None)
    assert a1.state is None
    assert torch.equal(a1.total, a1.answer.loss.value)

    logits = torch.zeros((1, 1, 6), requires_grad=True)
    state_input = StateLossInput(
        batch_size=1,
        o1=O1StateTarget(
            row_indices=torch.tensor([0]),
            logits=logits,
            targets=torch.ones_like(logits),
            slot_mask=torch.ones((1, 1), dtype=torch.bool),
        ),
    )
    a2 = compute_stage_a_losses(StageAVariant.A2, answer=answer, state=state_input)
    assert a2.state is not None
    assert torch.equal(a2.total, a2.state.total + a2.answer.loss.value)
    with pytest.raises(ValueError, match="A1 cannot receive State"):
        compute_stage_a_losses(StageAVariant.A1, answer=answer, state=state_input)


def test_stage_a_optimizer_owns_static_w0_and_state_allowlist_only() -> None:
    model = _ToyStageAModel()
    trainer = build_trainer(config=load_config(), model=model, forward_step=_ToyForward(model))
    selected = trainer.parameter_audit.trainable_names
    assert any("fast_adapter" in name for name in selected)
    assert all("predictor" not in name and "qwen_prefill" not in name for name in selected)
    actual = {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }
    expected = {id(parameter) for name, parameter in model.named_parameters() if name in selected}
    assert actual == expected


def test_stage_a_step_updates_only_allowlist_and_reports_na_as_none() -> None:
    torch.manual_seed(15)
    model = _ToyStageAModel()
    trainer = build_trainer(config=load_config(), model=model, forward_step=_ToyForward(model))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    output = trainer.train_step(_batch())
    assert output.stage is TrainingStage.A
    assert output.global_step == 1
    assert output.audit is not None and output.audit.optimizer_step_applied
    metrics = dict(output.metrics)
    assert metrics["retrieval/precision"] is None
    assert metrics["reader/exact_count_accuracy"] == 1.0
    for name, parameter in model.named_parameters():
        changed = not torch.equal(before[name], parameter)
        assert changed == (name in trainer.parameter_audit.trainable_names)


def test_stage_a_nonfinite_gradient_skips_without_partial_parameter_update() -> None:
    model = _ToyStageAModel()
    trainer = build_trainer(config=load_config(), model=model, forward_step=_ToyForward(model))
    parameter = dict(model.named_parameters())[trainer.parameter_audit.trainable_names[0]]
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    output = trainer.train_step(_batch())
    handle.remove()
    assert output.global_step == 0
    assert output.audit is not None
    assert output.audit.skip_reason == "nonfinite_gradient"
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())


def test_stage_a_runtime_payload_leak_and_inner_sgd_audit_fail_closed() -> None:
    model = _ToyStageAModel()
    trainer = build_trainer(config=load_config(), model=model, forward_step=_ToyForward(model))
    with pytest.raises(ValueError, match="denied fields"):
        trainer.train_step(_batch(leaked=True))
    with pytest.raises(ValueError, match="Inner SGD"):
        StageAExecutionAudit(
            row_count=1,
            observed_chunk_count=1,
            hard_state_row_count=1,
            query_router_row_count=1,
            time_resolver_row_count=1,
            retrieval_row_count=1,
            reader_result_count=1,
            bank_reset_count=1,
            bank_write_count=1,
            cache_advance_count=1,
            fsm_rollout_count=1,
            inner_sgd_attempted=1,
        ).validate_for(StageAVariant.A2)


def test_balanced_stage_a_sampler_is_seeded_and_balances_four_families() -> None:
    tasks = ("o1", "o1", "o1", "o2", "e1", "e1", "e2")
    first = build_balanced_stage_a_indices(tasks, seed=42)
    second = build_balanced_stage_a_indices(tasks, seed=42)
    assert first == second
    counts = {name: 0 for name in ("o1", "o2", "e1", "e2")}
    for index in first:
        counts[tasks[index]] += 1
    assert set(counts.values()) == {3}


def test_stage_a_checkpoint_roundtrip_is_trainable_only_and_restores_rng(tmp_path: Path) -> None:
    torch.manual_seed(15)
    random.seed(15)
    config = load_config()
    model = _ToyStageAModel()
    trainer = build_trainer(config=config, model=model, forward_step=_ToyForward(model))
    trainer.train_step(_batch())
    saved = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in trainer.parameter_audit.trainable_names
    }
    checkpoint = trainer.save_checkpoint(
        tmp_path / "checkpoint",
        metrics={"validation_total_loss": 1.0},
        fold=0,
        dataset_revision="synthetic-p15-v1",
        annotation_sha256="a" * 64,
        architecture_sha256="b" * 64,
        git_commit="synthetic-test-commit",
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.add_(10.0)
    random.random()
    torch.rand(3)
    restored_step = load_stage_a_checkpoint(
        checkpoint,
        model=model,
        optimizer=trainer.optimizer,
        config=config,
        parameter_audit=trainer.parameter_audit,
        variant=StageAVariant.A2,
    )
    assert restored_step == 1
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    assert all(torch.equal(saved[name], dict(model.named_parameters())[name]) for name in saved)

    tensors = load_file(str(checkpoint / "trainable.safetensors"))
    assert set(tensors) == set(trainer.parameter_audit.trainable_names)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, sort_keys=True).lower()
    for forbidden in ("transient_w_t", "state_bank_runtime", "temporal_cache"):
        assert forbidden in serialized
    assert manifest["variant"] == "a2"
    assert set(manifest["artifacts"]) == {"trainable.safetensors", "training_state.pt"}


def test_training_step_output_rejects_stage_loss_type_mismatch() -> None:
    with pytest.raises(ValueError, match="Stage A output requires"):
        TrainingStepOutput(
            stage=TrainingStage.A,
            losses=object(),  # type: ignore[arg-type]
            global_step=0,
            metrics=(),
            checkpoint_path=None,
        )
