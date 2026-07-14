"""Fail-closed evidence bundles for the P16, P17, and P18 stage gates.

The bundle is deliberately data-only.  Training and inference code supplies a frozen
configuration snapshot, numeric metrics, a structured runtime audit, handled negative
cases, a human-readable report, and command outputs.  This module writes those inputs as
UTF-8, hashes every file, and independently revalidates the stage-specific contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TypeGuard

import yaml

_FIXED_FILES = (
    "config-snapshot.yaml",
    "metrics.json",
    "audit.json",
    "failure-cases.json",
    "report.md",
    "command-evidence.json",
    "bundle-manifest.json",
)
_BUNDLE_SCHEMA = "stage_gate_artifact_bundle_v1"
_METRICS_SCHEMA = "stage_gate_metrics_v1"
_AUDIT_SCHEMA = "stage_gate_audit_v1"
_FAILURE_SCHEMA = "stage_gate_failure_cases_v1"
_COMMAND_SCHEMA = "stage_gate_command_evidence_v1"
_SAFE_COMMAND_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GateStage(StrEnum):
    """Stages supported by the shared evidence contract."""

    P16 = "p16"
    P17 = "p17"
    P18 = "p18"


@dataclass(frozen=True, slots=True)
class StageFailureCase:
    """One deliberately exercised failure path."""

    case_id: str
    category: str
    handled: bool
    detail: str

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.category.strip() or not self.detail.strip():
            raise ValueError("stage failure cases require non-empty text fields")
        if type(self.handled) is not bool:
            raise TypeError("stage failure handled flag must be bool")


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    """Captured command result whose stdout and stderr become hashed bundle files."""

    name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str = ""

    def __post_init__(self) -> None:
        if _SAFE_COMMAND_NAME.fullmatch(self.name) is None:
            raise ValueError("command evidence name must be a safe lower-case slug")
        if not self.command.strip():
            raise ValueError("command evidence requires a non-empty command")
        if type(self.exit_code) is not int:
            raise TypeError("command evidence exit_code must be int")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("command evidence stdout/stderr must be text")


@dataclass(frozen=True, slots=True)
class StageGateResult:
    """A stage passes only when no structural, semantic, or integrity reason remains."""

    stage: GateStage
    passed: bool
    reasons: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.passed != (not self.reasons):
            raise ValueError("stage gate result and reasons disagree")


def write_stage_gate_bundle(
    root: str | Path,
    *,
    stage: GateStage,
    config_snapshot: Mapping[str, object],
    metrics: Mapping[str, object],
    audit: Mapping[str, object],
    failure_cases: Sequence[StageFailureCase],
    report: str,
    command_evidence: Sequence[CommandEvidence],
) -> Path:
    """Write a reviewable UTF-8 bundle without deciding whether its evidence passes."""

    if not isinstance(stage, GateStage):
        raise TypeError("stage must be a GateStage")
    if not report.strip():
        raise ValueError("stage gate report must be non-empty")
    if not command_evidence:
        raise ValueError("stage gate bundle requires command evidence")
    names = tuple(item.name for item in command_evidence)
    if len(set(names)) != len(names):
        raise ValueError("command evidence names must be unique")

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    command_root = output / "command-logs"
    command_root.mkdir(parents=True, exist_ok=True)

    snapshot_text = yaml.safe_dump(
        dict(config_snapshot),
        allow_unicode=True,
        sort_keys=False,
    )
    (output / "config-snapshot.yaml").write_text(snapshot_text, encoding="utf-8")
    _write_json(
        output / "metrics.json",
        {"schema": _METRICS_SCHEMA, "stage": stage.value, "metrics": dict(metrics)},
    )
    _write_json(
        output / "audit.json",
        {"schema": _AUDIT_SCHEMA, "stage": stage.value, "audit": dict(audit)},
    )
    _write_json(
        output / "failure-cases.json",
        {
            "schema": _FAILURE_SCHEMA,
            "stage": stage.value,
            "cases": [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "handled": case.handled,
                    "detail": case.detail,
                }
                for case in failure_cases
            ],
        },
    )
    (output / "report.md").write_text(report.rstrip() + "\n", encoding="utf-8")

    commands: list[dict[str, object]] = []
    command_paths: list[Path] = []
    for evidence in command_evidence:
        outputs: dict[str, object] = {}
        for stream_name, content in (("stdout", evidence.stdout), ("stderr", evidence.stderr)):
            path = command_root / f"{evidence.name}.{stream_name}.log"
            path.write_text(content, encoding="utf-8")
            command_paths.append(path)
            outputs[stream_name] = {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        commands.append(
            {
                "name": evidence.name,
                "command": evidence.command,
                "exit_code": evidence.exit_code,
                "outputs": outputs,
            }
        )
    _write_json(
        output / "command-evidence.json",
        {"schema": _COMMAND_SCHEMA, "stage": stage.value, "commands": commands},
    )

    manifest_paths = [
        output / name for name in _FIXED_FILES if name != "bundle-manifest.json"
    ] + command_paths
    hashes = {
        path.relative_to(output).as_posix(): _sha256_file(path)
        for path in sorted(manifest_paths, key=lambda item: item.as_posix())
    }
    _write_json(
        output / "bundle-manifest.json",
        {"schema": _BUNDLE_SCHEMA, "stage": stage.value, "files": hashes},
    )
    return output


def evaluate_stage_gate_bundle(
    root: str | Path,
    *,
    stage: GateStage,
    expected_config_sha256: str,
) -> StageGateResult:
    """Re-parse, integrity-check, and semantically validate one stage bundle."""

    if not isinstance(stage, GateStage):
        raise TypeError("stage must be a GateStage")
    output = Path(root)
    reasons: list[str] = []
    if _SHA256.fullmatch(expected_config_sha256) is None:
        raise ValueError("expected_config_sha256 must be a lower-case SHA-256 digest")
    for name in _FIXED_FILES:
        if not (output / name).is_file():
            reasons.append(f"missing_artifact:{name}")
    if reasons:
        return StageGateResult(stage, False, tuple(reasons), ())

    try:
        snapshot = _read_yaml_object(output / "config-snapshot.yaml")
        metrics_payload = _read_json_object(output / "metrics.json")
        audit_payload = _read_json_object(output / "audit.json")
        failure_payload = _read_json_object(output / "failure-cases.json")
        command_payload = _read_json_object(output / "command-evidence.json")
        manifest = _read_json_object(output / "bundle-manifest.json")
        report = (output / "report.md").read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError, OSError, ValueError) as error:
        return StageGateResult(
            stage,
            False,
            (f"artifact_parse_error:{type(error).__name__}",),
            (),
        )

    if _sha256_file(output / "config-snapshot.yaml") != expected_config_sha256:
        reasons.append("config_snapshot_hash_mismatch")
    _validate_header(metrics_payload, _METRICS_SCHEMA, stage, "metrics", reasons)
    _validate_header(audit_payload, _AUDIT_SCHEMA, stage, "audit", reasons)
    _validate_header(failure_payload, _FAILURE_SCHEMA, stage, "failure_cases", reasons)
    _validate_header(command_payload, _COMMAND_SCHEMA, stage, "commands", reasons)
    _validate_header(manifest, _BUNDLE_SCHEMA, stage, "bundle", reasons)

    gate_config = _as_mapping(snapshot.get("stage_gate"))
    if gate_config is None:
        reasons.append("config_stage_gate_missing")
        gate_config = {}
    elif gate_config.get("stage") != stage.value:
        reasons.append("config_stage_mismatch")

    metrics = _as_mapping(metrics_payload.get("metrics"))
    if metrics is None:
        reasons.append("metrics_schema_invalid")
        metrics = {}
    audit = _as_mapping(audit_payload.get("audit"))
    if audit is None:
        reasons.append("audit_schema_invalid")
        audit = {}

    if stage is GateStage.P16:
        _validate_p16_config(gate_config, reasons)
        _validate_p16_metrics(metrics, reasons)
        _validate_p16_audit(audit, reasons)
    elif stage is GateStage.P17:
        _validate_p17_config(gate_config, reasons)
        _validate_p17_metrics(metrics, reasons)
        _validate_p17_audit(audit, reasons)
    else:
        _validate_p18_config(gate_config, reasons)
        _validate_p18_metrics(metrics, reasons)
        _validate_p18_audit(audit, reasons)

    _validate_failure_cases(failure_payload, stage, reasons)
    command_paths = _validate_commands(command_payload, stage, output, reasons)
    _validate_report(report, stage, reasons)
    _validate_bundle_manifest(manifest, output, command_paths, reasons)

    unique_reasons = tuple(dict.fromkeys(reasons))
    artifact_hashes = tuple(
        (path.relative_to(output).as_posix(), _sha256_file(path))
        for path in sorted(output.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )
    return StageGateResult(stage, not unique_reasons, unique_reasons, artifact_hashes)


def _validate_p16_config(config: Mapping[str, object], reasons: list[str]) -> None:
    expected: tuple[tuple[str, object, str], ...] = (
        ("variant", "a3", "p16_config_not_a3"),
        ("support_chunks", 1, "p16_config_support_not_one"),
        ("minimum_query_points", 1, "p16_config_query_point_missing"),
        ("inner_steps_per_valid_chunk", 1, "p16_config_inner_step_not_one"),
        ("update_effect", "next_chunk_only", "p16_config_update_not_next_only"),
        ("support_uses_labels", False, "p16_config_support_labels_not_disabled"),
        ("reset_each_episode", True, "p16_config_reset_not_enabled"),
        ("outer_aux_pred_weight", 0.1, "p16_config_pred_aux_weight_invalid"),
    )
    for key, value, reason in expected:
        if not _same_scalar(config.get(key), value):
            reasons.append(reason)
    weights = _as_mapping(config.get("inner_loss_weights"))
    if weights is None or not _exact_numeric_weights(
        weights, {"pred": 1.0, "id": 0.0, "event": 0.0}
    ):
        reasons.append("p16_config_inner_loss_not_pred_only")


def _validate_p16_metrics(metrics: Mapping[str, object], reasons: list[str]) -> None:
    required = (
        "query/before_answer_loss",
        "query/after_answer_loss",
        "query/before_state_loss",
        "query/after_state_loss",
        "inner/update_norm",
        "inner/gradient_norm",
        "inner/skip_rate",
        "inner/updates_per_video",
    )
    _require_finite_metrics(metrics, required, "p16", reasons)
    _validate_rate(metrics.get("inner/skip_rate"), "p16_skip_rate_invalid", reasons)
    _validate_nonnegative(
        metrics.get("inner/updates_per_video"), "p16_updates_per_video_invalid", reasons
    )


def _validate_p16_audit(audit: Mapping[str, object], reasons: list[str]) -> None:
    support = _section(audit, "support", "p16_support_audit_missing", reasons)
    if support.get("count") != 1:
        reasons.append("p16_single_support_not_verified")
    for key, reason in (
        ("label_free", "p16_support_label_leakage_not_excluded"),
        ("hard_state_before_inner_loss", "p16_observe_state_order_not_verified"),
        ("pred_only", "p16_pred_only_not_verified"),
    ):
        _require_true(support, key, reason, reasons)
    if not _is_positive_int(support.get("query_points")):
        reasons.append("p16_followup_query_missing")

    update = _section(audit, "update", "p16_update_audit_missing", reasons)
    if update.get("steps_on_valid_support") != 1:
        reasons.append("p16_one_step_not_verified")
    for key, reason in (
        ("only_two_fast_matrices_changed", "p16_fast_parameter_boundary_failed"),
        ("next_chunk_only", "p16_next_only_not_verified"),
        ("current_support_not_recomputed", "p16_support_recompute_not_excluded"),
        ("norms_recorded", "p16_update_norm_audit_missing"),
        ("skip_reason_recorded", "p16_skip_reason_audit_missing"),
    ):
        _require_true(update, key, reason, reasons)

    outer = _section(audit, "outer", "p16_outer_audit_missing", reasons)
    for key, reason in (
        ("after_update_query_loss", "p16_after_update_query_not_verified"),
        ("w0_meta_gradient", "p16_w0_meta_gradient_missing_or_false"),
        ("first_order_gradient_check", "p16_first_order_gradient_check_failed"),
        ("second_order_gradient_check", "p16_second_order_gradient_check_failed"),
        ("before_after_metrics_recorded", "p16_before_after_metrics_missing"),
    ):
        _require_true(outer, key, reason, reasons)
    if not _same_scalar(outer.get("pred_auxiliary_weight"), 0.1):
        reasons.append("p16_outer_pred_aux_weight_invalid")

    reset = _section(audit, "reset", "p16_reset_audit_missing", reasons)
    components = _as_mapping(reset.get("components"))
    required_components = (
        "fast_weights",
        "sgd_state",
        "temporal_cache",
        "slot_state",
        "state_bank",
        "fsm",
        "audit",
    )
    if components is None or any(components.get(name) is not True for name in required_components):
        reasons.append("p16_full_reset_not_verified")
    _require_true(reset, "cross_episode_isolation", "p16_cross_episode_pollution", reasons)

    invalid = _section(audit, "invalid_support", "p16_invalid_support_audit_missing", reasons)
    for key, reason in (
        ("no_valid_time_skipped", "p16_invalid_support_skip_failed"),
        ("episode_continued", "p16_invalid_support_broke_episode"),
        ("hard_state_protocol_preserved", "p16_invalid_support_state_protocol_failed"),
    ):
        _require_true(invalid, key, reason, reasons)
    _require_true(audit, "fixed_seed_repeatable", "p16_fixed_seed_not_repeatable", reasons)


def _validate_p17_config(config: Mapping[str, object], reasons: list[str]) -> None:
    if config.get("support_chunk_schedule") != [1, 4, 8]:
        reasons.append("p17_config_support_schedule_invalid")
    expected: tuple[tuple[str, object, str], ...] = (
        (
            "maximum_inner_steps_per_valid_chunk",
            1,
            "p17_config_inner_step_not_one",
        ),
        ("multiple_query_points", True, "p17_config_multi_query_disabled"),
        ("cross_chunk_pred_graph", False, "p17_config_cross_chunk_graph_enabled"),
        ("new_video_starts_from_w0", True, "p17_config_w0_reset_disabled"),
        ("query_time_causal", True, "p17_config_query_causality_disabled"),
    )
    for key, value, reason in expected:
        if not _same_scalar(config.get(key), value):
            reasons.append(reason)

    variants = _as_mapping(config.get("variants"))
    a4 = _as_mapping(variants.get("a4")) if variants is not None else None
    a5 = _as_mapping(variants.get("a5")) if variants is not None else None
    if a4 is None or a5 is None:
        reasons.append("p17_config_a4_a5_missing")
        return
    if a4.get("independent") is not True or a5.get("independent") is not True:
        reasons.append("p17_config_a4_a5_not_independent")
    a4_weights = _as_mapping(a4.get("inner_loss_weights"))
    a5_weights = _as_mapping(a5.get("inner_loss_weights"))
    if a4_weights is None or not _exact_numeric_weights(
        a4_weights, {"pred": 1.0, "id": 0.5, "event": 0.0}
    ):
        reasons.append("p17_config_a4_weights_invalid")
    if a5_weights is None or not _exact_numeric_weights(
        a5_weights, {"pred": 1.0, "id": 0.5, "event": 0.5}
    ):
        reasons.append("p17_config_a5_weights_invalid")
    shared_keys = (set(a4) | set(a5)) - {"inner_loss_weights"}
    if any(a4.get(key) != a5.get(key) for key in shared_keys):
        reasons.append("p17_config_a4_a5_unexpected_drift")


def _validate_p17_metrics(metrics: Mapping[str, object], reasons: list[str]) -> None:
    required = (
        "a4/query_before_loss",
        "a4/query_after_loss",
        "a5/query_before_loss",
        "a5/query_after_loss",
        "identity/duplicate_rate",
        "identity/missed_new_rate",
        "event/e1_overlap_loss",
        "event/e2_overlap_loss",
        "inner/skip_rate",
        "ablation/a4_vs_a3_delta",
        "ablation/a4_vs_a3_ci_low",
        "ablation/a4_vs_a3_ci_high",
        "ablation/a5_vs_a4_delta",
        "ablation/a5_vs_a4_ci_low",
        "ablation/a5_vs_a4_ci_high",
    )
    _require_finite_metrics(metrics, required, "p17", reasons)
    for name in ("identity/duplicate_rate", "identity/missed_new_rate", "inner/skip_rate"):
        _validate_rate(metrics.get(name), f"p17_rate_invalid:{name}", reasons)
    for comparison in ("a4_vs_a3", "a5_vs_a4"):
        delta = metrics.get(f"ablation/{comparison}_delta")
        low = metrics.get(f"ablation/{comparison}_ci_low")
        high = metrics.get(f"ablation/{comparison}_ci_high")
        if not (
            _is_finite_number(delta)
            and _is_finite_number(low)
            and _is_finite_number(high)
            and float(low) <= float(delta) <= float(high)
        ):
            reasons.append(f"p17_ablation_ci_invalid:{comparison}")


def _validate_p17_audit(audit: Mapping[str, object], reasons: list[str]) -> None:
    variants = _section(audit, "variants", "p17_variant_audit_missing", reasons)
    for key, reason in (
        ("a4_independent_run", "p17_a4_independent_run_failed"),
        ("a5_independent_run", "p17_a5_independent_run_failed"),
        ("config_diff_only_event", "p17_variant_config_drift"),
    ):
        _require_true(variants, key, reason, reasons)

    identity = _section(audit, "identity", "p17_identity_audit_missing", reasons)
    for key, reason in (
        ("causal_overlap_metadata", "p17_identity_overlap_not_causal"),
        ("reliable_match_mask", "p17_identity_match_mask_failed"),
        ("no_match_invalid", "p17_identity_no_match_invalid_failed"),
        ("stop_gradient", "p17_identity_stop_gradient_failed"),
    ):
        _require_true(identity, key, reason, reasons)

    event = _section(audit, "event", "p17_event_audit_missing", reasons)
    for key, reason in (
        ("e1_overlap_recorded", "p17_e1_overlap_missing"),
        ("e2_overlap_recorded", "p17_e2_overlap_missing"),
        ("event_phase_masks", "p17_event_phase_mask_failed"),
        ("mse_kl_verified", "p17_event_loss_not_verified"),
        ("fsm_detached", "p17_fsm_detach_failed"),
    ):
        _require_true(event, key, reason, reasons)

    support = _section(audit, "support", "p17_support_audit_missing", reasons)
    timelines = _as_mapping(support.get("timelines"))
    if timelines is None or any(timelines.get(str(count)) is not True for count in (1, 4, 8)):
        reasons.append("p17_support_1_4_8_timeline_failed")
    for key, reason in (
        ("max_one_step", "p17_support_step_boundary_failed"),
        ("next_chunk_only", "p17_support_next_only_failed"),
        ("invalid_chunk_continues", "p17_invalid_chunk_progression_failed"),
        ("fast_state_timeline_consistent", "p17_fast_state_timeline_inconsistent"),
    ):
        _require_true(support, key, reason, reasons)

    queries = _section(audit, "queries", "p17_query_audit_missing", reasons)
    for key, reason in (
        ("multiple", "p17_multiple_queries_not_verified"),
        ("causal_state_only", "p17_query_future_state_leakage"),
        ("later_label_isolation", "p17_later_query_label_leakage"),
        ("per_query_losses_recorded", "p17_per_query_losses_missing"),
        ("trajectory_policy_recorded", "p17_query_trajectory_policy_missing"),
    ):
        _require_true(queries, key, reason, reasons)

    graph = _section(audit, "graph", "p17_graph_audit_missing", reasons)
    for key, reason in (
        ("cross_chunk_pred_graph_released", "p17_cross_chunk_graph_not_released"),
        ("bounded_lifetime", "p17_graph_lifetime_unbounded"),
    ):
        _require_true(graph, key, reason, reasons)
    ablation = _section(audit, "ablation", "p17_ablation_audit_missing", reasons)
    for key, reason in (
        ("confidence_intervals_recorded", "p17_ablation_ci_missing"),
        ("before_after_per_task_recorded", "p17_ablation_detail_missing"),
    ):
        _require_true(ablation, key, reason, reasons)


def _validate_p18_config(config: Mapping[str, object], reasons: list[str]) -> None:
    expected: tuple[tuple[str, object, str], ...] = (
        ("mode", "inference", "p18_config_not_inference"),
        ("runtime_scope", "per_video", "p18_config_not_per_video"),
        ("labels_allowed", False, "p18_config_labels_allowed"),
        ("query_time_causal", True, "p18_config_query_causality_disabled"),
        ("prefill_count_per_query", 1, "p18_config_prefill_not_once"),
        ("update_steps_per_valid_chunk", 1, "p18_config_update_step_not_one"),
    )
    for key, value, reason in expected:
        if not _same_scalar(config.get(key), value):
            reasons.append(reason)
    if config.get("decode_mutable_state") != ["llm_kv_cache"]:
        reasons.append("p18_config_decode_mutability_invalid")


def _validate_p18_metrics(metrics: Mapping[str, object], reasons: list[str]) -> None:
    required = (
        "runtime/videos",
        "runtime/chunks",
        "runtime/queries",
        "runtime/updates",
        "runtime/skips",
        "runtime/prefill_calls",
        "runtime/decode_steps",
        "reader/status_invalid",
        "reader/status_unsupported",
        "reader/status_empty",
        "reader/status_ok",
    )
    _require_finite_metrics(metrics, required, "p18", reasons)
    for name in required:
        _validate_nonnegative(metrics.get(name), f"p18_metric_negative:{name}", reasons)
    for status in ("invalid", "unsupported", "empty", "ok"):
        value = metrics.get(f"reader/status_{status}")
        if not _is_finite_number(value) or float(value) < 1:
            reasons.append(f"p18_reader_status_not_exercised:{status}")
    queries = metrics.get("runtime/queries")
    prefill = metrics.get("runtime/prefill_calls")
    if not (
        _is_finite_number(queries)
        and _is_finite_number(prefill)
        and float(queries) > 0
        and float(prefill) == float(queries)
    ):
        reasons.append("p18_prefill_query_count_mismatch")


def _validate_p18_audit(audit: Mapping[str, object], reasons: list[str]) -> None:
    reset = _section(audit, "reset", "p18_reset_audit_missing", reasons)
    components = _as_mapping(reset.get("components"))
    required_components = (
        "fast_weights",
        "sgd_state",
        "temporal_cache",
        "slot_state",
        "state_bank",
        "identity_candidate",
        "identity_confirmed",
        "identity_hot_cache",
        "o1_fsm",
        "e1_fsm",
        "e2_fsm",
        "event_histories",
        "reader_audit",
        "gru_hidden",
        "fast_version",
        "update_counters",
    )
    if components is None or any(components.get(name) is not True for name in required_components):
        reasons.append("p18_full_reset_not_verified")
    before = reset.get("before_checksum")
    after = reset.get("after_checksum")
    expected_empty = reset.get("expected_empty_checksum")
    initial = reset.get("video_initial_checksums")
    if not (
        _valid_sha256(before)
        and _valid_sha256(after)
        and _valid_sha256(expected_empty)
        and before != after
        and after == expected_empty
        and isinstance(initial, list)
        and len(initial) >= 2
        and all(value == expected_empty for value in initial)
    ):
        reasons.append("p18_reset_checksum_contract_failed")
    for key, reason in (
        ("no_cross_video_residue", "p18_cross_video_residue"),
        ("first_chunk_w0_verified", "p18_first_chunk_w0_failed"),
        ("first_chunk_empty_bank_verified", "p18_first_chunk_bank_not_empty"),
    ):
        _require_true(reset, key, reason, reasons)

    chunks = audit.get("chunk_timeline")
    _validate_p18_chunk_timeline(chunks, reasons)
    expected_observe_steps = [
        "causal_crop",
        "vit_merger",
        "fast_adapter",
        "state_encoder",
        "soft_decoders",
        "hard_state_update",
        "ttt_loss",
        "functional_sgd",
    ]
    if audit.get("observe_pipeline_steps") != expected_observe_steps:
        reasons.append("p18_chunk_pipeline_order_invalid")
    _require_true(audit, "causal_crop_verified", "p18_causal_crop_not_verified", reasons)

    query = _section(audit, "query", "p18_query_audit_missing", reasons)
    expected_query_steps = [
        "query_encoder",
        "operator",
        "time_resolver",
        "retriever",
        "reader",
        "resampler",
        "composer",
        "llm",
    ]
    if query.get("pipeline_steps") != expected_query_steps:
        reasons.append("p18_query_pipeline_order_invalid")
    prefill_counts = query.get("prefill_counts")
    if not (
        isinstance(prefill_counts, list)
        and prefill_counts
        and all(value == 1 for value in prefill_counts)
    ):
        reasons.append("p18_prefill_once_failed")
    statuses = _as_mapping(query.get("reader_statuses"))
    if statuses is None or any(
        statuses.get(status) is not True for status in ("invalid", "unsupported", "empty", "ok")
    ):
        reasons.append("p18_reader_status_coverage_failed")
    for key, reason in (
        ("reader_results_saved", "p18_reader_result_not_saved"),
        ("selected_records_saved", "p18_selected_records_not_saved"),
        ("state_attention_saved", "p18_state_attention_not_saved"),
        ("final_text_saved", "p18_final_text_not_saved"),
        ("replayable_count_sources", "p18_count_source_not_replayable"),
    ):
        _require_true(query, key, reason, reasons)
    future = _as_mapping(query.get("future_invariance"))
    if future is None or not (
        future.get("verified") is True
        and _same_valid_hashes(
            future.get("baseline_reader_sha256"), future.get("perturbed_reader_sha256")
        )
        and _same_valid_hashes(
            future.get("baseline_input_sha256"), future.get("perturbed_input_sha256")
        )
    ):
        reasons.append("p18_future_invariance_failed")

    decode = _section(audit, "decode", "p18_decode_audit_missing", reasons)
    if decode.get("mutable_state") != ["llm_kv_cache"]:
        reasons.append("p18_decode_mutability_failed")
    if not _same_valid_hashes(
        decode.get("state_checksum_before"), decode.get("state_checksum_after")
    ):
        reasons.append("p18_decode_state_changed")
    for key, reason in (
        ("multi_token_verified", "p18_decode_multi_token_not_verified"),
        ("retry_semantics_recorded", "p18_retry_semantics_missing"),
    ):
        _require_true(decode, key, reason, reasons)

    release = _section(audit, "release", "p18_release_audit_missing", reasons)
    for key, reason in (
        ("abort_safe", "p18_abort_release_failed"),
        ("exception_safe", "p18_exception_release_failed"),
        ("next_video_clean", "p18_release_polluted_next_video"),
    ):
        _require_true(release, key, reason, reasons)
    cli = _section(audit, "cli", "p18_cli_audit_missing", reasons)
    for key, reason in (
        ("available", "p18_cli_missing"),
        ("label_fields_rejected", "p18_cli_label_boundary_failed"),
        ("entrypoint_smoke_passed", "p18_cli_smoke_failed"),
    ):
        _require_true(cli, key, reason, reasons)


def _validate_p18_chunk_timeline(value: object, reasons: list[str]) -> None:
    if not isinstance(value, list) or len(value) < 2:
        reasons.append("p18_chunk_timeline_missing")
        return
    previous_end: float | None = None
    previous_version: int | None = None
    saw_update = False
    saw_skip = False
    for index, raw_entry in enumerate(value):
        entry = _as_mapping(raw_entry)
        if entry is None:
            reasons.append(f"p18_chunk_timeline_invalid:{index}")
            continue
        start = entry.get("start_time")
        end = entry.get("end_time")
        before = entry.get("fast_version_before")
        after = entry.get("fast_version_after")
        updated = entry.get("updated")
        skip_reason = entry.get("skip_reason")
        if not (
            isinstance(entry.get("chunk_id"), (str, int))
            and _is_finite_number(start)
            and _is_finite_number(end)
            and float(start) < float(end)
            and _is_nonnegative_int(before)
            and _is_nonnegative_int(after)
            and type(updated) is bool
            and entry.get("hard_state_updated") is True
            and entry.get("update_effect") == "next_chunk_only"
        ):
            reasons.append(f"p18_chunk_timeline_invalid:{index}")
            continue
        if previous_end is not None and float(start) < previous_end:
            reasons.append(f"p18_chunk_time_regressed:{index}")
        if previous_version is not None and before != previous_version:
            reasons.append(f"p18_chunk_fast_version_discontinuous:{index}")
        if updated is True:
            saw_update = True
            if after != before + 1 or skip_reason not in (None, ""):
                reasons.append(f"p18_chunk_update_audit_invalid:{index}")
        else:
            saw_skip = True
            if after != before or not isinstance(skip_reason, str) or not skip_reason.strip():
                reasons.append(f"p18_chunk_skip_audit_invalid:{index}")
        previous_end = float(end)
        previous_version = int(after)
    if not saw_update:
        reasons.append("p18_chunk_update_not_exercised")
    if not saw_skip:
        reasons.append("p18_chunk_skip_not_exercised")


def _validate_failure_cases(
    payload: Mapping[str, object], stage: GateStage, reasons: list[str]
) -> None:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        reasons.append("failure_cases_missing")
        return
    categories: set[str] = set()
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _as_mapping(raw_case)
        if case is None:
            reasons.append(f"failure_case_invalid:{index}")
            continue
        case_id = case.get("case_id")
        category = case.get("category")
        detail = case.get("detail")
        if not (
            isinstance(case_id, str)
            and case_id.strip()
            and isinstance(category, str)
            and category.strip()
            and isinstance(detail, str)
            and detail.strip()
        ):
            reasons.append(f"failure_case_invalid:{index}")
            continue
        if case_id in case_ids:
            reasons.append(f"failure_case_duplicate:{case_id}")
        case_ids.add(case_id)
        categories.add(category)
        if case.get("handled") is not True:
            reasons.append(f"failure_case_unhandled:{case_id}")
    required_categories = {
        GateStage.P16: {"invalid_support_skip", "nonfinite_inner_loss_skip"},
        GateStage.P17: {"identity_no_overlap", "invalid_chunk_skip", "late_query_leakage"},
        GateStage.P18: {
            "label_rejected",
            "future_frame_rejected",
            "abort_release",
            "unsupported_reader",
        },
    }[stage]
    for category in sorted(required_categories - categories):
        reasons.append(f"failure_case_category_missing:{category}")


def _validate_commands(
    payload: Mapping[str, object],
    stage: GateStage,
    output: Path,
    reasons: list[str],
) -> set[str]:
    raw_commands = payload.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        reasons.append("command_evidence_missing")
        return set()
    seen_names: set[str] = set()
    paths: set[str] = set()
    for index, raw_command in enumerate(raw_commands):
        command = _as_mapping(raw_command)
        if command is None:
            reasons.append(f"command_evidence_invalid:{index}")
            continue
        name = command.get("name")
        command_text = command.get("command")
        if not isinstance(name, str) or _SAFE_COMMAND_NAME.fullmatch(name) is None:
            reasons.append(f"command_name_invalid:{index}")
            continue
        if name in seen_names:
            reasons.append(f"command_name_duplicate:{name}")
        seen_names.add(name)
        if not isinstance(command_text, str) or not command_text.strip():
            reasons.append(f"command_text_missing:{name}")
        exit_code = command.get("exit_code")
        if type(exit_code) is not int or exit_code != 0:
            reasons.append(f"command_failed:{name}")
        outputs = _as_mapping(command.get("outputs"))
        if outputs is None:
            reasons.append(f"command_outputs_missing:{name}")
            continue
        for stream_name in ("stdout", "stderr"):
            metadata = _as_mapping(outputs.get(stream_name))
            if metadata is None:
                reasons.append(f"command_output_missing:{name}:{stream_name}")
                continue
            relative = metadata.get("path")
            if not isinstance(relative, str) or not _safe_bundle_path(relative):
                reasons.append(f"command_output_path_invalid:{name}:{stream_name}")
                continue
            paths.add(relative)
            path = output / PurePosixPath(relative)
            expected_size = metadata.get("size_bytes")
            expected_hash = metadata.get("sha256")
            if not path.is_file():
                reasons.append(f"command_output_missing:{name}:{stream_name}")
                continue
            try:
                path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                reasons.append(f"command_output_not_utf8:{name}:{stream_name}")
            if (
                not _is_nonnegative_int(expected_size)
                or path.stat().st_size != expected_size
                or not _valid_sha256(expected_hash)
                or _sha256_file(path) != expected_hash
            ):
                reasons.append(f"command_output_hash_mismatch:{name}:{stream_name}")
    required_names = {"pytest", "ruff", "mypy"}
    if stage is GateStage.P18:
        required_names.add("cli")
    for name in sorted(required_names - seen_names):
        reasons.append(f"command_evidence_required:{name}")
    return paths


def _validate_report(report: str, stage: GateStage, reasons: list[str]) -> None:
    lower = report.lower()
    if not report.strip():
        reasons.append("report_empty")
    if stage.value not in lower:
        reasons.append("report_stage_missing")
    if "audit" not in lower and "审计" not in report:
        reasons.append("report_audit_summary_missing")


def _validate_bundle_manifest(
    manifest: Mapping[str, object],
    output: Path,
    command_paths: set[str],
    reasons: list[str],
) -> None:
    files = _as_mapping(manifest.get("files"))
    if files is None:
        reasons.append("bundle_manifest_invalid")
        return
    required = {name for name in _FIXED_FILES if name != "bundle-manifest.json"} | command_paths
    manifest_names = set(files)
    for name in sorted(required - manifest_names):
        reasons.append(f"bundle_manifest_entry_missing:{name}")
    actual_names = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    for name in sorted(actual_names - manifest_names):
        reasons.append(f"bundle_unmanifested_file:{name}")
    for name in sorted(manifest_names - actual_names):
        reasons.append(f"bundle_manifest_stale_entry:{name}")
    for name, expected_hash in files.items():
        if not isinstance(name, str) or not _safe_bundle_path(name):
            reasons.append(f"bundle_path_invalid:{name}")
            continue
        path = output / PurePosixPath(name)
        if (
            not path.is_file()
            or not _valid_sha256(expected_hash)
            or _sha256_file(path) != expected_hash
        ):
            reasons.append(f"bundle_hash_mismatch:{name}")


def _validate_header(
    payload: Mapping[str, object],
    schema: str,
    stage: GateStage,
    label: str,
    reasons: list[str],
) -> None:
    if payload.get("schema") != schema:
        reasons.append(f"{label}_schema_mismatch")
    if payload.get("stage") != stage.value:
        reasons.append(f"{label}_stage_mismatch")


def _require_finite_metrics(
    metrics: Mapping[str, object],
    names: Sequence[str],
    stage: str,
    reasons: list[str],
) -> None:
    for name in names:
        if name not in metrics:
            reasons.append(f"{stage}_metric_missing:{name}")
        elif not _is_finite_number(metrics[name]):
            reasons.append(f"{stage}_metric_not_finite:{name}")


def _validate_rate(value: object, reason: str, reasons: list[str]) -> None:
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        reasons.append(reason)


def _validate_nonnegative(value: object, reason: str, reasons: list[str]) -> None:
    if not _is_finite_number(value) or float(value) < 0:
        reasons.append(reason)


def _section(
    parent: Mapping[str, object], key: str, reason: str, reasons: list[str]
) -> Mapping[str, object]:
    value = _as_mapping(parent.get(key))
    if value is None:
        reasons.append(reason)
        return {}
    return value


def _require_true(mapping: Mapping[str, object], key: str, reason: str, reasons: list[str]) -> None:
    if mapping.get(key) is not True:
        reasons.append(reason)


def _same_scalar(value: object, expected: object) -> bool:
    if type(expected) is bool or type(expected) is int:
        return type(value) is type(expected) and value == expected
    if isinstance(expected, float):
        return _is_finite_number(value) and float(value) == expected
    return value == expected


def _exact_numeric_weights(value: Mapping[str, object], expected: Mapping[str, float]) -> bool:
    if set(value) != set(expected):
        return False
    for name, weight in expected.items():
        actual = value[name]
        if not _is_finite_number(actual) or float(actual) != weight:
            return False
    return True


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_int(value: object) -> TypeGuard[int]:
    return type(value) is int and value > 0


def _is_nonnegative_int(value: object) -> TypeGuard[int]:
    return type(value) is int and value >= 0


def _valid_sha256(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _same_valid_hashes(left: object, right: object) -> bool:
    return _valid_sha256(left) and _valid_sha256(right) and left == right


def _safe_bundle_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and bool(path.parts) and ".." not in path.parts


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8", errors="strict"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("top-level JSON value must be an object with string keys")
    return value


def _read_yaml_object(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("top-level YAML value must be an object with string keys")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
