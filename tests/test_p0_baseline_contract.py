from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "docs" / "p0"


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_p0_spec_lock_matches_architecture_and_baseline() -> None:
    architecture_path = ROOT / "ARCHITECTURE.md"
    architecture = read_utf8(architecture_path)
    spec_lock = read_utf8(P0 / "spec-lock.md")
    architecture_hash = hashlib.sha256(architecture_path.read_bytes()).hexdigest()

    assert "state_ttt_qwen3vl8b_high_capacity_sgd_v5_embedding_retrieval" in architecture
    assert "> 修订日期：2026-07-14" in architecture
    assert "> 状态：PARTIALLY IMPLEMENTED / P0-P15 ENGINEERING-VERIFIED" in architecture
    assert f"ARCHITECTURE_SHA256 | `{architecture_hash}`" in spec_lock
    assert "7f0185f8136faf88cc59e5ba2ec7309c36f8d013" in spec_lock
    assert "c66d2675c153ce306248b2b97913ff41f162fd3bb8a7514c6ca75888c12b8df2" in spec_lock


def test_p0_separates_fixed_prohibited_and_experimental_contracts() -> None:
    spec_lock = read_utf8(P0 / "spec-lock.md")

    assert "## v5 固定项" in spec_lock
    assert "## 第一版禁止项" in spec_lock
    assert "## 实验待定项" in spec_lock
    assert "旧 v3 运行事实隔离" in spec_lock
    assert "bottleneck 512、16 slots、8 State Token" in spec_lock


def test_p0_baseline_command_logs_are_present_and_green() -> None:
    expected = {
        "uv-sync-frozen.log": "Checked 74 packages",
        "pytest-q.log": "5 passed",
        "ruff-check.log": "All checks passed!",
        "mypy-src.log": "Success: no issues found in 2 source files",
        "environment-summary.log": "transformers: 4.57.1",
        "model-revision.log": "revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "baseline-hashes.log": "baseline_commit: 7f0185f8136faf88cc59e5ba2ec7309c36f8d013",
    }

    for filename, marker in expected.items():
        assert marker in read_utf8(P0 / "evidence" / "commands" / filename)


def test_p0_traceability_matches_todo_appendix_and_covers_chapters_0_to_22() -> None:
    traceability = read_utf8(P0 / "requirements-traceability.md")
    todo = read_utf8(ROOT / "TODO.md")
    appendix_d = todo.split("## 附录 D.", maxsplit=1)[1].split("## 附录 E.", maxsplit=1)[0]

    todo_sources = set(re.findall(r"^\| `([^`]+)` \|", appendix_d, flags=re.MULTILINE))
    trace_rows = [line for line in traceability.splitlines() if line.startswith("| ARCH-")]
    trace_sources = {line.split("|")[2].strip().strip("`") for line in trace_rows}
    top_level = {int(re.match(r"\d+", source).group()) for source in trace_sources}

    assert trace_sources == todo_sources
    assert top_level == set(range(23))
    assert len({line.split("|")[1].strip() for line in trace_rows}) == len(trace_rows)
    assert all(" P" in line and ("tests/" in line or "P" in line) for line in trace_rows)


def test_p0_stage_status_has_separate_planned_and_verified_columns() -> None:
    policy = read_utf8(P0 / "execution-policy.md")
    stage_rows = [line for line in policy.splitlines() if re.match(r"^\| P\d+ \|", line)]

    assert "| 阶段 | 计划设计 | 已验证实现 |" in policy
    assert len(stage_rows) == 23
    assert "<spec_version>__fold-<fold>__seed-<seed>__model-<revision12>__ttt-<on|off>" in policy
    for directory in ("config/", "logs/", "checkpoints/", "metrics/", "audit/", "failures/"):
        assert directory in policy


def test_p0_review_template_contains_forbidden_and_gate_checks() -> None:
    template = read_utf8(ROOT / ".github" / "pull_request_template.md")

    assert "## 第一版禁止项" in template
    assert "Surprise Gate" in template
    assert "固定 Top-K" in template
    assert "未改造 DeepStack" in template
    assert "query_time 之后" in template
    assert "全部 `pytest`、`ruff` 和 `mypy`" in template


def test_runtime_sources_do_not_hardcode_platform_absolute_paths() -> None:
    forbidden = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:root|home|mnt|data|workspace)/)")
    runtime_files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "configs").glob("*.yaml")),
    ]

    for path in runtime_files:
        assert forbidden.search(read_utf8(path)) is None, path.relative_to(ROOT)


def test_all_p0_text_artifacts_are_strict_utf8() -> None:
    paths = [
        ROOT / ".github" / "pull_request_template.md",
        *sorted(P0.rglob("*.md")),
        *sorted((P0 / "evidence" / "commands").glob("*.log")),
    ]

    for path in paths:
        path.read_bytes().decode("utf-8", errors="strict")
