from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ttt_svcbench_qwen.stage_gate_artifacts import (
    CommandEvidence,
    GateStage,
    StageFailureCase,
    evaluate_stage_gate_bundle,
    write_stage_gate_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(stage: GateStage) -> dict[str, object]:
    if stage is GateStage.P16:
        gate: dict[str, object] = {
            "stage": "p16",
            "variant": "a3",
            "support_chunks": 1,
            "minimum_query_points": 1,
            "inner_loss_weights": {"pred": 1.0, "id": 0.0, "event": 0.0},
            "inner_steps_per_valid_chunk": 1,
            "update_effect": "next_chunk_only",
            "support_uses_labels": False,
            "reset_each_episode": True,
            "outer_aux_pred_weight": 0.1,
        }
    elif stage is GateStage.P17:
        gate = {
            "stage": "p17",
            "variants": {
                "a4": {
                    "independent": True,
                    "inner_loss_weights": {"pred": 1.0, "id": 0.5, "event": 0.0},
                },
                "a5": {
                    "independent": True,
                    "inner_loss_weights": {"pred": 1.0, "id": 0.5, "event": 0.5},
                },
            },
            "support_chunk_schedule": [1, 4, 8],
            "maximum_inner_steps_per_valid_chunk": 1,
            "multiple_query_points": True,
            "cross_chunk_pred_graph": False,
            "new_video_starts_from_w0": True,
            "query_time_causal": True,
        }
    else:
        gate = {
            "stage": "p18",
            "mode": "inference",
            "runtime_scope": "per_video",
            "labels_allowed": False,
            "query_time_causal": True,
            "prefill_count_per_query": 1,
            "update_steps_per_valid_chunk": 1,
            "decode_mutable_state": ["llm_kv_cache"],
        }
    return {
        "spec_version": "synthetic_stage_gate_test_v1",
        "unicode_note": "合成输入，不含真实视频",
        "stage_gate": gate,
    }


def _metrics(stage: GateStage) -> dict[str, object]:
    if stage is GateStage.P16:
        return {
            "query/before_answer_loss": 1.2,
            "query/after_answer_loss": 1.0,
            "query/before_state_loss": 0.8,
            "query/after_state_loss": 0.7,
            "inner/update_norm": 0.02,
            "inner/gradient_norm": 0.4,
            "inner/skip_rate": 0.25,
            "inner/updates_per_video": 0.75,
        }
    if stage is GateStage.P17:
        return {
            "a4/query_before_loss": 1.0,
            "a4/query_after_loss": 0.9,
            "a5/query_before_loss": 1.0,
            "a5/query_after_loss": 0.85,
            "identity/duplicate_rate": 0.1,
            "identity/missed_new_rate": 0.2,
            "event/e1_overlap_loss": 0.3,
            "event/e2_overlap_loss": 0.4,
            "inner/skip_rate": 0.125,
            "ablation/a4_vs_a3_delta": 0.03,
            "ablation/a4_vs_a3_ci_low": 0.01,
            "ablation/a4_vs_a3_ci_high": 0.05,
            "ablation/a5_vs_a4_delta": 0.02,
            "ablation/a5_vs_a4_ci_low": -0.01,
            "ablation/a5_vs_a4_ci_high": 0.04,
        }
    return {
        "runtime/videos": 2.0,
        "runtime/chunks": 4.0,
        "runtime/queries": 4.0,
        "runtime/updates": 2.0,
        "runtime/skips": 2.0,
        "runtime/prefill_calls": 4.0,
        "runtime/decode_steps": 12.0,
        "reader/status_invalid": 1.0,
        "reader/status_unsupported": 1.0,
        "reader/status_empty": 1.0,
        "reader/status_ok": 1.0,
    }


def _audit(stage: GateStage) -> dict[str, object]:
    if stage is GateStage.P16:
        return {
            "support": {
                "count": 1,
                "query_points": 1,
                "label_free": True,
                "hard_state_before_inner_loss": True,
                "pred_only": True,
            },
            "update": {
                "steps_on_valid_support": 1,
                "only_two_fast_matrices_changed": True,
                "next_chunk_only": True,
                "current_support_not_recomputed": True,
                "norms_recorded": True,
                "skip_reason_recorded": True,
            },
            "outer": {
                "after_update_query_loss": True,
                "w0_meta_gradient": True,
                "first_order_gradient_check": True,
                "second_order_gradient_check": True,
                "before_after_metrics_recorded": True,
                "pred_auxiliary_weight": 0.1,
            },
            "reset": {
                "components": {
                    "fast_weights": True,
                    "sgd_state": True,
                    "temporal_cache": True,
                    "slot_state": True,
                    "state_bank": True,
                    "fsm": True,
                    "audit": True,
                },
                "cross_episode_isolation": True,
            },
            "invalid_support": {
                "no_valid_time_skipped": True,
                "episode_continued": True,
                "hard_state_protocol_preserved": True,
            },
            "fixed_seed_repeatable": True,
        }
    if stage is GateStage.P17:
        return {
            "variants": {
                "a4_independent_run": True,
                "a5_independent_run": True,
                "config_diff_only_event": True,
            },
            "identity": {
                "causal_overlap_metadata": True,
                "reliable_match_mask": True,
                "no_match_invalid": True,
                "stop_gradient": True,
            },
            "event": {
                "e1_overlap_recorded": True,
                "e2_overlap_recorded": True,
                "event_phase_masks": True,
                "mse_kl_verified": True,
                "fsm_detached": True,
            },
            "support": {
                "timelines": {"1": True, "4": True, "8": True},
                "max_one_step": True,
                "next_chunk_only": True,
                "invalid_chunk_continues": True,
                "fast_state_timeline_consistent": True,
            },
            "queries": {
                "multiple": True,
                "causal_state_only": True,
                "later_label_isolation": True,
                "per_query_losses_recorded": True,
                "trajectory_policy_recorded": True,
            },
            "graph": {
                "cross_chunk_pred_graph_released": True,
                "bounded_lifetime": True,
            },
            "ablation": {
                "confidence_intervals_recorded": True,
                "before_after_per_task_recorded": True,
            },
        }
    return {
        "reset": {
            "components": {
                "fast_weights": True,
                "sgd_state": True,
                "temporal_cache": True,
                "slot_state": True,
                "state_bank": True,
                "identity_candidate": True,
                "identity_confirmed": True,
                "identity_hot_cache": True,
                "o1_fsm": True,
                "e1_fsm": True,
                "e2_fsm": True,
                "event_histories": True,
                "reader_audit": True,
                "gru_hidden": True,
                "fast_version": True,
                "update_counters": True,
            },
            "before_checksum": "a" * 64,
            "after_checksum": "b" * 64,
            "expected_empty_checksum": "b" * 64,
            "video_initial_checksums": ["b" * 64, "b" * 64],
            "no_cross_video_residue": True,
            "first_chunk_w0_verified": True,
            "first_chunk_empty_bank_verified": True,
        },
        "chunk_timeline": [
            {
                "chunk_id": "chunk-0",
                "start_time": 0.0,
                "end_time": 1.0,
                "fast_version_before": 0,
                "fast_version_after": 1,
                "updated": True,
                "skip_reason": None,
                "hard_state_updated": True,
                "update_effect": "next_chunk_only",
            },
            {
                "chunk_id": "chunk-1",
                "start_time": 1.0,
                "end_time": 2.0,
                "fast_version_before": 1,
                "fast_version_after": 1,
                "updated": False,
                "skip_reason": "no_valid_target",
                "hard_state_updated": True,
                "update_effect": "next_chunk_only",
            },
        ],
        "observe_pipeline_steps": [
            "causal_crop",
            "vit_merger",
            "fast_adapter",
            "state_encoder",
            "soft_decoders",
            "hard_state_update",
            "ttt_loss",
            "functional_sgd",
        ],
        "causal_crop_verified": True,
        "query": {
            "pipeline_steps": [
                "query_encoder",
                "operator",
                "time_resolver",
                "retriever",
                "reader",
                "resampler",
                "composer",
                "llm",
            ],
            "prefill_counts": [1, 1, 1, 1],
            "reader_statuses": {
                "invalid": True,
                "unsupported": True,
                "empty": True,
                "ok": True,
            },
            "reader_results_saved": True,
            "selected_records_saved": True,
            "state_attention_saved": True,
            "final_text_saved": True,
            "replayable_count_sources": True,
            "future_invariance": {
                "verified": True,
                "baseline_reader_sha256": "c" * 64,
                "perturbed_reader_sha256": "c" * 64,
                "baseline_input_sha256": "d" * 64,
                "perturbed_input_sha256": "d" * 64,
            },
        },
        "decode": {
            "mutable_state": ["llm_kv_cache"],
            "state_checksum_before": "e" * 64,
            "state_checksum_after": "e" * 64,
            "multi_token_verified": True,
            "retry_semantics_recorded": True,
        },
        "release": {"abort_safe": True, "exception_safe": True, "next_video_clean": True},
        "cli": {
            "available": True,
            "label_fields_rejected": True,
            "entrypoint_smoke_passed": True,
        },
    }


def _failure_cases(stage: GateStage) -> tuple[StageFailureCase, ...]:
    categories = {
        GateStage.P16: ("invalid_support_skip", "nonfinite_inner_loss_skip"),
        GateStage.P17: ("identity_no_overlap", "invalid_chunk_skip", "late_query_leakage"),
        GateStage.P18: (
            "label_rejected",
            "future_frame_rejected",
            "abort_release",
            "unsupported_reader",
        ),
    }[stage]
    return tuple(
        StageFailureCase(
            case_id=f"{stage.value}-{index}",
            category=category,
            handled=True,
            detail=f"Synthetic {category} path was rejected or skipped without state pollution.",
        )
        for index, category in enumerate(categories)
    )


def _commands(stage: GateStage) -> tuple[CommandEvidence, ...]:
    names = ["pytest", "ruff", "mypy"]
    if stage is GateStage.P18:
        names.append("cli")
    return tuple(
        CommandEvidence(
            name=name,
            command=f"synthetic-{name} --stage {stage.value}",
            exit_code=0,
            stdout=f"{name} passed：合成证据\n",
        )
        for name in names
    )


def _write_valid(root: Path, stage: GateStage) -> Path:
    return write_stage_gate_bundle(
        root,
        stage=stage,
        config_snapshot=_config(stage),
        metrics=_metrics(stage),
        audit=_audit(stage),
        failure_cases=_failure_cases(stage),
        report=f"# {stage.value.upper()} audit report\n\n合成审计证据完整。",
        command_evidence=_commands(stage),
    )


@pytest.mark.parametrize("stage", tuple(GateStage))
def test_each_stage_bundle_is_utf8_hashed_and_passes(stage: GateStage, tmp_path: Path) -> None:
    bundle = _write_valid(tmp_path / stage.value, stage)
    result = evaluate_stage_gate_bundle(
        bundle,
        stage=stage,
        expected_config_sha256=_sha256(bundle / "config-snapshot.yaml"),
    )

    assert result.passed, result.reasons
    assert not result.reasons
    assert {name for name, _ in result.artifact_hashes} == {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    for path in bundle.rglob("*"):
        if path.is_file():
            path.read_text(encoding="utf-8", errors="strict")

    command_payload = json.loads((bundle / "command-evidence.json").read_text(encoding="utf-8"))
    for command in command_payload["commands"]:
        for metadata in command["outputs"].values():
            assert _sha256(bundle / metadata["path"]) == metadata["sha256"]


def test_gate_fails_closed_on_missing_artifact_and_hashed_file_drift(tmp_path: Path) -> None:
    missing = _write_valid(tmp_path / "missing", GateStage.P16)
    expected_missing_hash = _sha256(missing / "config-snapshot.yaml")
    (missing / "audit.json").unlink()
    missing_result = evaluate_stage_gate_bundle(
        missing,
        stage=GateStage.P16,
        expected_config_sha256=expected_missing_hash,
    )
    assert not missing_result.passed
    assert "missing_artifact:audit.json" in missing_result.reasons

    drifted = _write_valid(tmp_path / "drifted", GateStage.P16)
    expected_drifted_hash = _sha256(drifted / "config-snapshot.yaml")
    (drifted / "report.md").write_text("# P16 audit report\n\ndrift\n", encoding="utf-8")
    drift_result = evaluate_stage_gate_bundle(
        drifted,
        stage=GateStage.P16,
        expected_config_sha256=expected_drifted_hash,
    )
    assert not drift_result.passed
    assert "bundle_hash_mismatch:report.md" in drift_result.reasons


def test_p16_gate_rejects_incomplete_meta_ttt_audit(tmp_path: Path) -> None:
    config = _config(GateStage.P16)
    gate = config["stage_gate"]
    assert isinstance(gate, dict)
    gate["support_chunks"] = 2
    audit = _audit(GateStage.P16)
    outer = audit["outer"]
    update = audit["update"]
    invalid = audit["invalid_support"]
    assert isinstance(outer, dict) and isinstance(update, dict) and isinstance(invalid, dict)
    outer.pop("w0_meta_gradient")
    outer["second_order_gradient_check"] = False
    update["next_chunk_only"] = False
    invalid["no_valid_time_skipped"] = False
    bundle = write_stage_gate_bundle(
        tmp_path,
        stage=GateStage.P16,
        config_snapshot=config,
        metrics=_metrics(GateStage.P16),
        audit=audit,
        failure_cases=_failure_cases(GateStage.P16),
        report="# P16 audit report\n\nSynthetic negative bundle.",
        command_evidence=_commands(GateStage.P16),
    )
    result = evaluate_stage_gate_bundle(
        bundle,
        stage=GateStage.P16,
        expected_config_sha256=_sha256(bundle / "config-snapshot.yaml"),
    )
    assert not result.passed
    assert {
        "p16_config_support_not_one",
        "p16_w0_meta_gradient_missing_or_false",
        "p16_second_order_gradient_check_failed",
        "p16_next_only_not_verified",
        "p16_invalid_support_skip_failed",
    }.issubset(result.reasons)


def test_p17_gate_rejects_variant_timeline_graph_and_ci_drift(tmp_path: Path) -> None:
    config = _config(GateStage.P17)
    gate = config["stage_gate"]
    assert isinstance(gate, dict)
    variants = gate["variants"]
    assert isinstance(variants, dict) and isinstance(variants["a5"], dict)
    variants["a5"]["independent"] = False
    audit = _audit(GateStage.P17)
    support = audit["support"]
    queries = audit["queries"]
    graph = audit["graph"]
    assert isinstance(support, dict) and isinstance(support["timelines"], dict)
    assert isinstance(queries, dict) and isinstance(graph, dict)
    support["timelines"]["8"] = False
    queries["multiple"] = False
    graph["cross_chunk_pred_graph_released"] = False
    metrics = _metrics(GateStage.P17)
    metrics["ablation/a4_vs_a3_delta"] = 0.5
    bundle = write_stage_gate_bundle(
        tmp_path,
        stage=GateStage.P17,
        config_snapshot=config,
        metrics=metrics,
        audit=audit,
        failure_cases=_failure_cases(GateStage.P17),
        report="# P17 audit report\n\nSynthetic negative bundle.",
        command_evidence=_commands(GateStage.P17),
    )
    result = evaluate_stage_gate_bundle(
        bundle,
        stage=GateStage.P17,
        expected_config_sha256=_sha256(bundle / "config-snapshot.yaml"),
    )
    assert not result.passed
    assert {
        "p17_config_a4_a5_not_independent",
        "p17_config_a4_a5_unexpected_drift",
        "p17_support_1_4_8_timeline_failed",
        "p17_multiple_queries_not_verified",
        "p17_cross_chunk_graph_not_released",
        "p17_ablation_ci_invalid:a4_vs_a3",
    }.issubset(result.reasons)


def test_p18_gate_rejects_reset_timeline_query_decode_release_and_cli_drift(
    tmp_path: Path,
) -> None:
    audit = copy.deepcopy(_audit(GateStage.P18))
    reset = audit["reset"]
    chunks = audit["chunk_timeline"]
    query = audit["query"]
    decode = audit["decode"]
    release = audit["release"]
    cli = audit["cli"]
    assert all(isinstance(value, dict) for value in (reset, query, decode, release, cli))
    assert isinstance(reset, dict) and isinstance(reset["components"], dict)
    assert isinstance(chunks, list) and isinstance(chunks[1], dict)
    assert isinstance(query, dict) and isinstance(query["future_invariance"], dict)
    assert isinstance(query["reader_statuses"], dict)
    assert isinstance(decode, dict) and isinstance(release, dict) and isinstance(cli, dict)
    reset["components"]["reader_audit"] = False
    reset["video_initial_checksums"] = ["b" * 64, "f" * 64]
    chunks[1]["fast_version_before"] = 7
    query["prefill_counts"] = [1, 2]
    query["reader_statuses"]["empty"] = False
    query["future_invariance"]["perturbed_input_sha256"] = "f" * 64
    decode["state_checksum_after"] = "f" * 64
    release["exception_safe"] = False
    cli["entrypoint_smoke_passed"] = False
    bundle = write_stage_gate_bundle(
        tmp_path,
        stage=GateStage.P18,
        config_snapshot=_config(GateStage.P18),
        metrics=_metrics(GateStage.P18),
        audit=audit,
        failure_cases=_failure_cases(GateStage.P18),
        report="# P18 audit report\n\nSynthetic negative bundle.",
        command_evidence=_commands(GateStage.P18),
    )
    result = evaluate_stage_gate_bundle(
        bundle,
        stage=GateStage.P18,
        expected_config_sha256=_sha256(bundle / "config-snapshot.yaml"),
    )
    assert not result.passed
    assert {
        "p18_full_reset_not_verified",
        "p18_reset_checksum_contract_failed",
        "p18_chunk_fast_version_discontinuous:1",
        "p18_prefill_once_failed",
        "p18_reader_status_coverage_failed",
        "p18_future_invariance_failed",
        "p18_decode_state_changed",
        "p18_exception_release_failed",
        "p18_cli_smoke_failed",
    }.issubset(result.reasons)


def test_gate_rejects_unhandled_failure_failed_command_and_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    failures = list(_failure_cases(GateStage.P16))
    failures[0] = StageFailureCase(
        case_id=failures[0].case_id,
        category=failures[0].category,
        handled=False,
        detail=failures[0].detail,
    )
    commands = list(_commands(GateStage.P16))
    commands[0] = CommandEvidence("pytest", "synthetic-pytest", 1, "failed\n")
    bundle = write_stage_gate_bundle(
        tmp_path,
        stage=GateStage.P16,
        config_snapshot=_config(GateStage.P16),
        metrics=_metrics(GateStage.P16),
        audit=_audit(GateStage.P16),
        failure_cases=failures,
        report="# P16 audit report\n\nSynthetic negative bundle.",
        command_evidence=commands,
    )
    result = evaluate_stage_gate_bundle(
        bundle,
        stage=GateStage.P16,
        expected_config_sha256="0" * 64,
    )
    assert not result.passed
    assert "config_snapshot_hash_mismatch" in result.reasons
    assert f"failure_case_unhandled:{failures[0].case_id}" in result.reasons
    assert "command_failed:pytest" in result.reasons
