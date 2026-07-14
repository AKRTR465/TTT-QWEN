"""Write and verify the compact P15 evidence bundle and P16 exit gate.

Inputs: validated config, metrics, runtime audits, handled failures, and a tiny checkpoint.
Outputs: UTF-8 artifacts with hashes plus a fail-closed P16 eligibility result.
Forbidden: full-model/runtime state, unhandled failures, TTT activity, or scientific claims.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from ttt_svcbench_qwen.config import ProjectConfig, StageAVariant
from ttt_svcbench_qwen.trainer import REQUIRED_A2_METRICS

_REQUIRED_FILES = (
    "config-snapshot.yaml",
    "metrics.json",
    "audit.json",
    "failure-examples.json",
    "freeze-strategy.md",
    "checkpoint-manifest.json",
    "bundle-manifest.json",
)


@dataclass(frozen=True, slots=True)
class P15FailureExample:
    case_id: str
    category: str
    handled: bool
    detail: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.category or not self.detail:
            raise ValueError("P15 failure examples require non-empty text fields")
        if type(self.handled) is not bool:
            raise TypeError("P15 failure handled flag must be bool")


@dataclass(frozen=True, slots=True)
class P15ExitGateResult:
    p16_allowed: bool
    reasons: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.p16_allowed != (not self.reasons):
            raise ValueError("P15 exit-gate result and reasons disagree")


def write_p15_artifact_bundle(
    root: str | Path,
    *,
    config: ProjectConfig,
    metrics: Mapping[str, float | None],
    audit: Mapping[str, object],
    failure_examples: Sequence[P15FailureExample],
    freeze_strategy: str,
    checkpoint_directory: str | Path,
    architecture_sha256: str,
    git_commit: str,
) -> Path:
    """Write small reviewable artifacts; the checkpoint remains outside the Git bundle."""

    if config.stage_a.variant is not StageAVariant.A2:
        raise ValueError("P15 artifact bundle is the A2 exit gate")
    if not config.stage_a.synthetic_engineering_gate_only:
        raise ValueError("P15 low-space bundle must remain an explicit synthetic gate")
    if not freeze_strategy.strip() or not architecture_sha256 or not git_commit:
        raise ValueError("P15 artifact provenance text must be non-empty")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_directory)
    checkpoint_files = tuple(
        checkpoint / name
        for name in ("trainable.safetensors", "training_state.pt", "manifest.json")
    )
    if any(not path.is_file() for path in checkpoint_files):
        raise FileNotFoundError("P15 checkpoint is missing one or more compact artifacts")

    snapshot = yaml.safe_dump(
        config.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
    )
    (output / "config-snapshot.yaml").write_text(snapshot, encoding="utf-8")
    _write_json(
        output / "metrics.json",
        {
            "schema": "p15_stage_a_metrics_v1",
            "stage": "stage_a",
            "variant": "a2",
            "synthetic_engineering_gate_only": True,
            "metrics": dict(metrics),
        },
    )
    _write_json(
        output / "audit.json",
        {
            "schema": "p15_stage_a_audit_v1",
            "architecture_sha256": architecture_sha256,
            "git_commit": git_commit,
            **dict(audit),
        },
    )
    _write_json(
        output / "failure-examples.json",
        {
            "schema": "p15_failure_examples_v1",
            "cases": [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "handled": case.handled,
                    "detail": case.detail,
                }
                for case in failure_examples
            ],
        },
    )
    (output / "freeze-strategy.md").write_text(
        freeze_strategy.rstrip() + "\n",
        encoding="utf-8",
    )
    checkpoint_payload = {
        "schema": "p15_checkpoint_reference_v1",
        "directory": str(checkpoint.resolve()),
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in checkpoint_files
        },
    }
    _write_json(output / "checkpoint-manifest.json", checkpoint_payload)
    hashes = {
        name: _sha256_file(output / name)
        for name in _REQUIRED_FILES
        if name != "bundle-manifest.json"
    }
    _write_json(
        output / "bundle-manifest.json",
        {
            "schema": "p15_artifact_bundle_v1",
            "files": hashes,
        },
    )
    return output


def evaluate_p15_exit_gate(
    root: str | Path,
    *,
    expected_architecture_sha256: str,
    minimum_reader_accuracy: float = 1.0,
) -> P15ExitGateResult:
    """Prove every P15 artifact/gate before allowing P16 construction."""

    output = Path(root)
    reasons: list[str] = []
    for name in _REQUIRED_FILES:
        if not (output / name).is_file():
            reasons.append(f"missing_artifact:{name}")
    if reasons:
        return P15ExitGateResult(False, tuple(reasons), ())
    try:
        snapshot = yaml.safe_load((output / "config-snapshot.yaml").read_text(encoding="utf-8"))
        metrics_payload = _read_json(output / "metrics.json")
        audit = _read_json(output / "audit.json")
        failures = _read_json(output / "failure-examples.json")
        checkpoint = _read_json(output / "checkpoint-manifest.json")
        bundle = _read_json(output / "bundle-manifest.json")
        freeze_text = (output / "freeze-strategy.md").read_text(encoding="utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError, OSError) as error:
        return P15ExitGateResult(False, (f"artifact_parse_error:{type(error).__name__}",), ())

    stage_a = snapshot.get("stage_a", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(stage_a, dict) or stage_a.get("variant") != "a2":
        reasons.append("config_not_a2")
    if not stage_a.get("synthetic_engineering_gate_only"):
        reasons.append("synthetic_gate_marker_missing")
    if "synthetic" not in freeze_text.lower() or "qwen" not in freeze_text.lower():
        reasons.append("freeze_strategy_incomplete")

    raw_metrics = metrics_payload.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        reasons.append("metrics_schema_invalid")
        raw_metrics = {}
    required_metrics = (*REQUIRED_A2_METRICS, "reader/exact_count_accuracy")
    missing_metrics = tuple(name for name in required_metrics if name not in raw_metrics)
    if missing_metrics:
        reasons.append(f"metrics_missing:{','.join(missing_metrics)}")
    reader_accuracy = raw_metrics.get("reader/exact_count_accuracy")
    if not isinstance(reader_accuracy, (int, float)) or reader_accuracy < minimum_reader_accuracy:
        reasons.append("reader_not_stable")
    disagreement = raw_metrics.get("reader/llm_number_disagreement_rate")
    if disagreement != 0.0:
        reasons.append("unresolved_reader_llm_disagreement")

    if audit.get("architecture_sha256") != expected_architecture_sha256:
        reasons.append("architecture_hash_mismatch")
    for counter in ("inner_sgd_attempted", "inner_sgd_updated", "inner_sgd_skipped"):
        if audit.get(counter) != 0:
            reasons.append(f"ttt_activity:{counter}")
    for forbidden in ("functional_sgd_reachable", "predictor_reachable", "transient_w_t_present"):
        if audit.get(forbidden) is not False:
            reasons.append(f"forbidden_runtime:{forbidden}")
    for required_true in (
        "finite_loss",
        "finite_gradients",
        "bank_reset_ok",
        "cache_isolation_ok",
        "fsm_rollout_ok",
        "checkpoint_roundtrip_ok",
    ):
        if audit.get(required_true) is not True:
            reasons.append(f"audit_failed:{required_true}")
    hard_rows = audit.get("hard_head_rows")
    if not isinstance(hard_rows, dict) or any(
        type(hard_rows.get(name)) is not int or hard_rows[name] <= 0
        for name in ("o1", "o2", "e1", "e2")
    ):
        reasons.append("four_head_rollout_incomplete")

    cases = failures.get("cases")
    if not isinstance(cases, list) or not cases:
        reasons.append("failure_examples_missing")
    elif any(not isinstance(case, dict) or case.get("handled") is not True for case in cases):
        reasons.append("unhandled_failure_example")

    files = checkpoint.get("files")
    if not isinstance(files, dict):
        reasons.append("checkpoint_reference_invalid")
    else:
        for name, metadata in files.items():
            if not isinstance(metadata, dict):
                reasons.append(f"checkpoint_metadata_invalid:{name}")
                continue
            path = Path(str(checkpoint.get("directory", ""))) / name
            if (
                not path.is_file()
                or metadata.get("size_bytes") != path.stat().st_size
                or metadata.get("sha256") != _sha256_file(path)
            ):
                reasons.append(f"checkpoint_artifact_mismatch:{name}")
    bundle_files = bundle.get("files")
    if not isinstance(bundle_files, dict):
        reasons.append("bundle_manifest_invalid")
    else:
        for name, expected_hash in bundle_files.items():
            path = output / name
            if not path.is_file() or _sha256_file(path) != expected_hash:
                reasons.append(f"bundle_hash_mismatch:{name}")
    artifact_hashes = tuple(
        (name, _sha256_file(output / name)) for name in _REQUIRED_FILES if (output / name).is_file()
    )
    return P15ExitGateResult(not reasons, tuple(reasons), artifact_hashes)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise json.JSONDecodeError("top-level JSON must be an object", str(value), 0)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
