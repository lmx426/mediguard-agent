"""Regression tests for local Evidence Agent evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = PROJECT_ROOT / "agent_evaluation" / "evidence_agent" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_eval  # noqa: E402


def test_eval_select_run_analysis_ignores_stale_current_analysis() -> None:
    run_analysis = {"run_id": "arun_old", "status": "complete"}
    current_analysis = {"run_id": "arun_other", "status": "complete"}

    analysis, source, stale = run_eval.select_run_analysis(
        run_analysis=run_analysis,
        current_analysis=current_analysis,
        run_id="arun_new",
    )

    assert analysis is None
    assert source == "stale_ignored"
    assert stale is True


def test_eval_summary_separates_conditional_and_effective_quality() -> None:
    good_validation = {
        "quality_available": True,
        "schema_pass": True,
        "citation_valid_rate": 1.0,
        "required_evidence_coverage_rate": 0.5,
        "sensitive_leak": False,
        "prohibited_conclusion": False,
    }
    missing_validation = run_eval.validate_analysis(None, {})

    summary = run_eval.summarize(
        [
            {
                "status": "complete",
                "latency_ms": 1000,
                "model_call_count": 1,
                "tool_call_count": 1,
                "validation": good_validation,
            },
            {
                "status": "failed",
                "latency_ms": 1000,
                "model_call_count": 1,
                "tool_call_count": 1,
                "validation": missing_validation,
            },
        ]
    )

    assert summary["complete_rate"] == 0.5
    assert summary["schema_pass_rate"] == 0.5
    assert summary["quality_available_rate"] == 0.5
    assert summary["citation_valid_rate_avg"] == 1.0
    assert summary["required_evidence_coverage_rate_avg"] == 0.5
    assert summary["effective_citation_valid_rate_avg"] == 0.5
    assert summary["effective_required_evidence_coverage_rate_avg"] == 0.25
