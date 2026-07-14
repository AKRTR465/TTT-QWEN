from __future__ import annotations

import json
from pathlib import Path

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.p15_artifacts import (
    P15FailureExample,
    evaluate_p15_exit_gate,
    write_p15_artifact_bundle,
)
from ttt_svcbench_qwen.trainer import REQUIRED_A2_METRICS


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "trainable.safetensors").write_bytes(b"tiny-trainable-only")
    (root / "training_state.pt").write_bytes(b"tiny-optimizer-rng")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "trainable_safetensors_plus_pt_state_v1",
                "variant": "a2",
                "excluded_runtime": ["transient_W_t", "temporal_cache", "fsm_runtime"],
            }
        ),
        encoding="utf-8",
    )
    return root


def _metrics() -> dict[str, float | None]:
    values = {name: 0.0 for name in REQUIRED_A2_METRICS}
    values.update(
        {
            "reader/exact_count_accuracy": 1.0,
            "reader/llm_number_disagreement_rate": 0.0,
        }
    )
    return values


def _audit(architecture_sha256: str) -> dict[str, object]:
    return {
        "inner_sgd_attempted": 0,
        "inner_sgd_updated": 0,
        "inner_sgd_skipped": 0,
        "functional_sgd_reachable": False,
        "predictor_reachable": False,
        "transient_w_t_present": False,
        "finite_loss": True,
        "finite_gradients": True,
        "bank_reset_ok": True,
        "cache_isolation_ok": True,
        "fsm_rollout_ok": True,
        "checkpoint_roundtrip_ok": True,
        "hard_head_rows": {"o1": 1, "o2": 1, "e1": 1, "e2": 1},
        "expected_architecture_sha256": architecture_sha256,
    }


def test_p15_artifact_bundle_is_utf8_hashed_and_opens_p16_gate(tmp_path: Path) -> None:
    architecture_sha256 = "a" * 64
    bundle = write_p15_artifact_bundle(
        tmp_path / "artifacts",
        config=load_config(),
        metrics=_metrics(),
        audit=_audit(architecture_sha256),
        failure_examples=(
            P15FailureExample(
                "synthetic-missing-label",
                "missing_provenance",
                True,
                "Missing dense labels remain invalid and cannot become zero labels.",
            ),
        ),
        freeze_strategy=(
            "Synthetic engineering gate: Qwen is frozen; static W0 and explicit state modules "
            "are the only outer-training candidates. No scientific convergence claim."
        ),
        checkpoint_directory=_checkpoint(tmp_path / "checkpoint"),
        architecture_sha256=architecture_sha256,
        git_commit="synthetic-worktree",
    )
    result = evaluate_p15_exit_gate(
        bundle,
        expected_architecture_sha256=architecture_sha256,
    )
    assert result.p16_allowed
    assert not result.reasons
    assert len(result.artifact_hashes) == 7
    for path in bundle.iterdir():
        path.read_text(encoding="utf-8", errors="strict")


def test_p15_exit_gate_fails_closed_on_unhandled_case_and_checkpoint_drift(
    tmp_path: Path,
) -> None:
    architecture_sha256 = "b" * 64
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    bundle = write_p15_artifact_bundle(
        tmp_path / "artifacts",
        config=load_config(),
        metrics=_metrics(),
        audit=_audit(architecture_sha256),
        failure_examples=(P15FailureExample("case", "negative", False, "not handled"),),
        freeze_strategy="Synthetic Qwen frozen engineering gate.",
        checkpoint_directory=checkpoint,
        architecture_sha256=architecture_sha256,
        git_commit="synthetic-worktree",
    )
    (checkpoint / "training_state.pt").write_bytes(b"drift")
    result = evaluate_p15_exit_gate(
        bundle,
        expected_architecture_sha256=architecture_sha256,
    )
    assert not result.p16_allowed
    assert "unhandled_failure_example" in result.reasons
    assert "checkpoint_artifact_mismatch:training_state.pt" in result.reasons
