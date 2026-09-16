"""Compare business required-point scoring with RAGAS-style claim flow.

The script drives the Case Agent once per eval row, then scores the same
response and the same adopted policy contexts with independent metric views:

- business_required_points: the current project judge over audited required /
  optional answer points.
- ragas_claim_flow: a project JSON judge that follows the RAGAS factual
  correctness flow by decomposing reference and response into atomic facts and
  computing TP / FP / FN without using audited required / optional points.
- native_ragas: optional native RAGAS AnswerCorrectness and Faithfulness with
  no deepseek_formula fallback.

This is an offline evaluation harness only. It does not change retrieval,
case facts, risk scores, rules, or human audit decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.paths import DEFAULT_CORPUS_ROOT
from src.backend.scripts.evaluate_case_agent_ragas_generation import (
    GenerationMetricEvaluator,
    append_jsonl,
    build_ragas_input,
    build_runtime_container,
    evaluate_one_row,
    fmt,
    install_gold_filter_injection,
    load_existing_results,
    metric_point_statements,
    reference_for_generation_eval,
    resolve_actor,
    resolve_case_id,
    selected_eval_rows,
    shutdown_container,
    validate_online_chain_region_contract,
)
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    load_env_file,
)


DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_RESULTS_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_case_agent_ragas_comparison_results.jsonl"
)
DEFAULT_RAGAS_INPUTS_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_case_agent_ragas_comparison_inputs.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_case_agent_ragas_comparison_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_case_agent_ragas_comparison_report.md"
)
EVALUABLE_STATUSES = {"completed", "degraded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run current Case Agent answers once and compare business "
            "required-point scoring with RAGAS-style claim-flow scoring."
        )
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--ragas-inputs-jsonl", type=Path, default=DEFAULT_RAGAS_INPUTS_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--actor-username", default="default_auditor")
    parser.add_argument("--active-stage", default="evidence_review")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--response-scope", choices=["primary", "full"], default="primary")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-sessions", action="store_true")
    parser.add_argument(
        "--inject-gold-filters",
        action="store_true",
        help="Inject eval-row gold filters into L3; omit for online filter inference.",
    )
    parser.add_argument(
        "--allow-regionless-online-questions",
        action="store_true",
        help="Allow online-chain rows whose question has no explicit jurisdiction.",
    )
    parser.add_argument(
        "--native-reference-source",
        choices=["required", "full", "reference"],
        default="required",
        help=(
            "Reference passed to native RAGAS. 'required' uses "
            "reference_answer_required first; 'full' uses reference_answer_full; "
            "'reference' uses reference_answer."
        ),
    )
    parser.add_argument(
        "--claim-flow-reference-source",
        choices=["required", "full", "reference"],
        default="required",
        help=(
            "Reference passed to the RAGAS-style claim-flow judge. This judge "
            "does not read required_answer_points / optional_answer_points."
        ),
    )
    parser.add_argument(
        "--native-answer-correctness-mode",
        choices=["factual", "ragas_default"],
        default="factual",
        help="Native RAGAS AnswerCorrectness mode; factual uses weights [1, 0].",
    )
    parser.add_argument(
        "--run-native-ragas",
        action="store_true",
        help=(
            "Also try the native RAGAS package. Disabled by default because some "
            "thinking/reasoning judge models reject RAGAS tool_choice calls."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--fail-on-native-error",
        action="store_true",
        help="Exit non-zero if any native RAGAS row cannot be scored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    write_reports(report, args.report_json, args.report_md)
    native = report["summary"]["native_ragas"]
    business = report["summary"]["business_required_points"]
    claim_flow = report["summary"]["ragas_claim_flow"]
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "completed_count": report["summary"]["run_status_counts"].get(
                    "completed",
                    0,
                ),
                "business_answer_correctness_avg": business.get(
                    "answer_correctness_avg",
                ),
                "business_faithfulness_avg": business.get("faithfulness_avg"),
                "ragas_claim_flow_answer_correctness_avg": claim_flow.get(
                    "answer_correctness_avg",
                ),
                "ragas_claim_flow_faithfulness_avg": claim_flow.get("faithfulness_avg"),
                "native_answer_correctness_avg": native.get("answer_correctness_avg"),
                "native_faithfulness_avg": native.get("faithfulness_avg"),
                "native_evaluated_count": native.get("evaluated_count"),
                "results_jsonl": str(args.results_jsonl),
                "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.run_native_ragas and args.fail_on_native_error and native.get("failed_count", 0):
        raise SystemExit(2)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    rows = selected_eval_rows(args)
    validate_online_chain_region_contract(args, rows)
    existing = load_existing_results(args.results_jsonl) if not args.no_resume else {}

    container = None
    restore_gold_filter_injection: Callable[[], None] | None = None
    results: list[dict[str, Any]] = []
    ragas_inputs: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        container = build_runtime_container()
        service = container.case_agent
        if service is None:
            raise RuntimeError(
                "Case Agent is not available: "
                + json.dumps(container.case_agent_status, ensure_ascii=False)
            )
        if args.inject_gold_filters:
            restore_gold_filter_injection = install_gold_filter_injection(service)
        actor = resolve_actor(container, args.actor_username)
        case_id = resolve_case_id(container, args.case_id)
        business_evaluator = GenerationMetricEvaluator(business_metric_args(args))
        claim_flow_evaluator = GenerationMetricEvaluator(claim_flow_metric_args(args))
        native_evaluator = (
            GenerationMetricEvaluator(native_metric_args(args))
            if args.run_native_ragas
            else None
        )

        total = len(rows)
        for index, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id") or f"row_{index}")
            if query_id in existing:
                result = existing[query_id]
                results.append(result)
                ragas_input = result.get("ragas_input")
                if isinstance(ragas_input, dict):
                    ragas_inputs.append(ragas_input)
                print_progress(args, index, total, result, resumed=True)
                continue

            result = evaluate_one_row(
                business_args(args),
                row=row,
                service=service,
                actor=actor,
                case_id=case_id,
                evaluator=business_evaluator,
            )
            business_metrics = result.pop("metrics", {})
            claim_flow_metrics = score_ragas_claim_flow(
                claim_flow_evaluator,
                args=args,
                row=row,
                result=result,
            )
            native_metrics = score_native_ragas(
                native_evaluator,
                args=args,
                row=row,
                result=result,
            )
            result["schema_version"] = "policy_case_agent_ragas_comparison_eval_v1"
            result["metrics"] = {
                "business_required_points": business_metrics,
                "ragas_claim_flow": claim_flow_metrics,
                "native_ragas": native_metrics,
            }
            result["native_reference_source"] = args.native_reference_source
            result["native_reference"] = native_reference(row, args.native_reference_source)
            result["claim_flow_reference_source"] = args.claim_flow_reference_source
            result["claim_flow_reference"] = native_reference(
                row,
                args.claim_flow_reference_source,
            )
            result["gold_required_points"] = metric_point_statements(
                row,
                "required_answer_points",
            )
            result["gold_optional_points"] = metric_point_statements(
                row,
                "optional_answer_points",
            )
            result["ragas_input"] = build_comparison_ragas_input(row, result, args)
            results.append(result)
            ragas_inputs.append(result["ragas_input"])
            append_jsonl(args.results_jsonl, [result])
            append_jsonl(args.ragas_inputs_jsonl, [result["ragas_input"]])
            print_progress(args, index, total, result, resumed=False)
    finally:
        if restore_gold_filter_injection is not None:
            restore_gold_filter_injection()
        shutdown_container(container)

    summary = summarize_comparison(results)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "results_jsonl": str(args.results_jsonl),
        "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
        "query_count": len(results),
        "offset": args.offset,
        "limit": args.limit,
        "response_scope": args.response_scope,
        "native_reference_source": args.native_reference_source,
        "claim_flow_reference_source": args.claim_flow_reference_source,
        "native_answer_correctness_mode": args.native_answer_correctness_mode,
        "runs_native_ragas": bool(args.run_native_ragas),
        "elapsed_seconds": time.perf_counter() - started_all,
        "summary": summary,
        "sample_results": slim_results(results[:20]),
        "boundary": {
            "drives_case_agent_service": True,
            "runs_policy_rag_mcp_through_expert_agent": True,
            "uses_same_agent_response_for_both_metrics": True,
            "uses_same_policy_contexts_for_both_metrics": True,
            "business_metric_uses_required_optional_points": True,
            "claim_flow_metric_uses_required_optional_points": False,
            "claim_flow_metric_decomposes_reference_and_response": True,
            "native_ragas_uses_required_optional_points": False,
            "native_ragas_allows_deepseek_formula_fallback": False,
            "injects_gold_filters_to_l3": bool(args.inject_gold_filters),
            "requires_region_terms_without_gold_filter_injection": (
                not args.allow_regionless_online_questions
            ),
            "makes_audit_decisions": False,
        },
    }


def business_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.metric_backend = "deepseek_formula"
    copied.answer_correctness_mode = "factual"
    copied.skip_ragas = False
    copied.ragas_native_only = False
    copied.force_ragas_native = False
    return copied


def business_metric_args(args: argparse.Namespace) -> argparse.Namespace:
    return business_args(args)


def claim_flow_metric_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.metric_backend = "deepseek_formula"
    copied.answer_correctness_mode = "factual"
    copied.skip_ragas = False
    copied.ragas_native_only = False
    copied.force_ragas_native = False
    return copied


def native_metric_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.metric_backend = "ragas"
    copied.answer_correctness_mode = args.native_answer_correctness_mode
    copied.skip_ragas = False
    copied.ragas_native_only = True
    copied.force_ragas_native = True
    apply_ragas_judge_env_aliases()
    return copied


def apply_ragas_judge_env_aliases() -> None:
    aliases = {
        "MEDIGUARD_RAGAS_JUDGE_API_KEY": "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
        "MEDIGUARD_RAGAS_JUDGE_BASE_URL": "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
        "MEDIGUARD_RAGAS_JUDGE_MODEL": "MEDIGUARD_POLICY_RAG_EVAL_MODEL",
    }
    for source, target in aliases.items():
        value = os.environ.get(source)
        if value:
            os.environ[target] = value


def score_native_ragas(
    evaluator: GenerationMetricEvaluator | None,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    if evaluator is None:
        return native_metric_error("native_ragas_skipped")
    run_status = str(result.get("run_status") or "")
    response_text = str(result.get("agent_response") or "")
    context_texts = [
        str(context.get("text") or context.get("excerpt") or "").strip()
        for context in result.get("retrieved_contexts") or []
        if isinstance(context, dict)
        and str(context.get("text") or context.get("excerpt") or "").strip()
    ]
    if run_status not in EVALUABLE_STATUSES:
        return native_metric_error(f"run_status_{run_status}")
    if not response_text.strip():
        return native_metric_error("empty_agent_response")
    if not context_texts:
        return native_metric_error("no_policy_context")
    try:
        metrics = evaluator.score(
            user_input=str(row.get("question") or ""),
            reference=native_reference(row, args.native_reference_source),
            reference_answer_full=str(
                row.get("reference_answer_full") or row.get("reference_answer") or ""
            ),
            required_answer_points=[],
            optional_answer_points=[],
            response=response_text,
            retrieved_contexts=context_texts,
        )
    except Exception as exc:
        return native_metric_error(
            "native_ragas_scoring_failed",
            error=f"{exc.__class__.__name__}: {str(exc)[:500]}",
        )
    metrics["metric_backend"] = "ragas"
    metrics["ragas_native"] = True
    metrics["reference_source"] = args.native_reference_source
    return metrics


def score_ragas_claim_flow(
    evaluator: GenerationMetricEvaluator,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    run_status = str(result.get("run_status") or "")
    response_text = str(result.get("agent_response") or "")
    context_texts = [
        str(context.get("text") or context.get("excerpt") or "").strip()
        for context in result.get("retrieved_contexts") or []
        if isinstance(context, dict)
        and str(context.get("text") or context.get("excerpt") or "").strip()
    ]
    if run_status not in EVALUABLE_STATUSES:
        return claim_flow_metric_error(f"run_status_{run_status}")
    if not response_text.strip():
        return claim_flow_metric_error("empty_agent_response")
    if not context_texts:
        return claim_flow_metric_error("no_policy_context")
    try:
        metrics = evaluator.score(
            user_input=str(row.get("question") or ""),
            reference=native_reference(row, args.claim_flow_reference_source),
            reference_answer_full=str(
                row.get("reference_answer_full") or row.get("reference_answer") or ""
            ),
            required_answer_points=[],
            optional_answer_points=[],
            response=response_text,
            retrieved_contexts=context_texts,
        )
    except Exception as exc:
        return claim_flow_metric_error(
            "ragas_claim_flow_scoring_failed",
            error=f"{exc.__class__.__name__}: {str(exc)[:500]}",
        )
    metrics["metric_backend"] = "deepseek_formula"
    metrics["ragas_native"] = False
    metrics["ragas_flow"] = "reference_response_claim_decomposition"
    metrics["uses_required_optional_points"] = False
    metrics["reference_source"] = args.claim_flow_reference_source
    return metrics


def claim_flow_metric_error(reason: str, *, error: str = "") -> dict[str, Any]:
    return {
        "evaluable": False,
        "reason": reason,
        "metric_backend": "deepseek_formula",
        "ragas_native": False,
        "ragas_flow": "reference_response_claim_decomposition",
        "uses_required_optional_points": False,
        "answer_correctness": None,
        "faithfulness": None,
        "error": error,
    }


def native_metric_error(reason: str, *, error: str = "") -> dict[str, Any]:
    return {
        "evaluable": False,
        "reason": reason,
        "metric_backend": "ragas",
        "ragas_native": True,
        "answer_correctness": None,
        "faithfulness": None,
        "error": error,
    }


def native_reference(row: dict[str, Any], source: str) -> str:
    if source == "full":
        return str(
            row.get("reference_answer_full")
            or row.get("reference_answer")
            or row.get("reference_answer_required")
            or ""
        )
    if source == "reference":
        return str(row.get("reference_answer") or row.get("reference_answer_full") or "")
    return reference_for_generation_eval(row)


def build_comparison_ragas_input(
    row: dict[str, Any],
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = build_ragas_input(
        row,
        response_text=str(result.get("agent_response") or ""),
        contexts=[
            context
            for context in result.get("retrieved_contexts") or []
            if isinstance(context, dict)
        ],
    )
    payload["native_reference_source"] = args.native_reference_source
    payload["native_reference"] = native_reference(row, args.native_reference_source)
    payload["claim_flow_reference_source"] = args.claim_flow_reference_source
    payload["claim_flow_reference"] = native_reference(
        row,
        args.claim_flow_reference_source,
    )
    return payload


def summarize_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    run_status_counts = Counter(str(item.get("run_status") or "unknown") for item in results)
    return {
        "query_count": len(results),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "retrieved_context_count_avg": average(
            item.get("retrieved_context_count") for item in results
        ),
        "business_required_points": metric_summary(
            metric_at(item, "business_required_points") for item in results
        ),
        "ragas_claim_flow": metric_summary(
            metric_at(item, "ragas_claim_flow") for item in results
        ),
        "native_ragas": metric_summary(metric_at(item, "native_ragas") for item in results),
        "by_question_type": grouped_comparison_summary(results, "question_type"),
        "by_bundle_type": grouped_comparison_summary(results, "bundle_type"),
    }


def grouped_comparison_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get(key) or "unknown")].append(item)
    return {
        group: {
            "query_count": len(items),
            "business_answer_correctness_avg": average(
                metric_at(item, "business_required_points").get("answer_correctness")
                for item in items
                if metric_at(item, "business_required_points").get("evaluable") is True
            ),
            "claim_flow_answer_correctness_avg": average(
                metric_at(item, "ragas_claim_flow").get("answer_correctness")
                for item in items
                if metric_at(item, "ragas_claim_flow").get("evaluable") is True
            ),
            "native_answer_correctness_avg": average(
                metric_at(item, "native_ragas").get("answer_correctness")
                for item in items
                if metric_at(item, "native_ragas").get("evaluable") is True
            ),
            "business_faithfulness_avg": average(
                metric_at(item, "business_required_points").get("faithfulness")
                for item in items
                if metric_at(item, "business_required_points").get("evaluable") is True
            ),
            "claim_flow_faithfulness_avg": average(
                metric_at(item, "ragas_claim_flow").get("faithfulness")
                for item in items
                if metric_at(item, "ragas_claim_flow").get("evaluable") is True
            ),
            "native_faithfulness_avg": average(
                metric_at(item, "native_ragas").get("faithfulness")
                for item in items
                if metric_at(item, "native_ragas").get("evaluable") is True
            ),
        }
        for group, items in sorted(grouped.items())
    }


def metric_summary(metrics_iter: Iterable[dict[str, Any]]) -> dict[str, Any]:
    metrics = list(metrics_iter)
    evaluated = [item for item in metrics if item.get("evaluable") is True]
    reasons = Counter(str(item.get("reason") or "ok") for item in metrics)
    backends = Counter(str(item.get("metric_backend") or "none") for item in metrics)
    return {
        "evaluated_count": len(evaluated),
        "failed_count": len(metrics) - len(evaluated),
        "metric_reason_counts": dict(sorted(reasons.items())),
        "metric_backend_counts": dict(sorted(backends.items())),
        "answer_correctness_avg": average(
            item.get("answer_correctness") for item in evaluated
        ),
        "faithfulness_avg": average(item.get("faithfulness") for item in evaluated),
        "required_point_recall_avg": average(
            item.get("required_point_recall") for item in evaluated
        ),
        "optional_point_coverage_avg": average(
            item.get("optional_point_coverage") for item in evaluated
        ),
        "fp_count_total": sum_metric_count(evaluated, "fp_count"),
        "fn_required_count_total": sum_metric_count(evaluated, "fn_required_count"),
    }


def metric_at(result: dict[str, Any], name: str) -> dict[str, Any]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    value = metrics.get(name)
    return value if isinstance(value, dict) else {}


def average(values: Iterable[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def sum_metric_count(metrics: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in metrics:
        value = item.get(key)
        if isinstance(value, int):
            total += value
    return total


def slim_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slimmed = []
    for item in results:
        business = metric_at(item, "business_required_points")
        claim_flow = metric_at(item, "ragas_claim_flow")
        native = metric_at(item, "native_ragas")
        slimmed.append(
            {
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "run_status": item.get("run_status"),
                "answer_display_mode": item.get("answer_display_mode"),
                "retrieved_context_count": item.get("retrieved_context_count"),
                "business_required_points": business,
                "ragas_claim_flow": claim_flow,
                "native_ragas": native,
                "native_reference_source": item.get("native_reference_source"),
                "claim_flow_reference_source": item.get("claim_flow_reference_source"),
                "gold_required_points": item.get("gold_required_points", []),
                "gold_optional_points": item.get("gold_optional_points", []),
                "agent_response_preview": str(item.get("agent_response") or "")[:320],
            }
        )
    return slimmed


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    business = summary.get("business_required_points") or {}
    claim_flow = summary.get("ragas_claim_flow") or {}
    native = summary.get("native_ragas") or {}
    lines = [
        "# Case Agent RAGAS Comparison Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report compares the current business required-point judge with native RAGAS on the same Case Agent responses and the same adopted policy contexts. It does not make audit decisions.",
        "",
        "## Scope",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Response scope: `{report.get('response_scope')}`",
        f"- Claim-flow reference source: `{report.get('claim_flow_reference_source')}`",
        f"- Native reference source: `{report.get('native_reference_source')}`",
        f"- Native answer correctness mode: `{report.get('native_answer_correctness_mode')}`",
        f"- Run native RAGAS package: `{report.get('runs_native_ragas')}`",
        f"- Row results: `{report.get('results_jsonl')}`",
        f"- RAGAS inputs: `{report.get('ragas_inputs_jsonl')}`",
        "",
        "## Overall",
        "",
        "| Metric | Evaluated | Answer Correctness | Faithfulness | Required Recall | Optional Coverage | FP | FN required |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Business required points | {business.get('evaluated_count')} | "
            f"{fmt(business.get('answer_correctness_avg'))} | "
            f"{fmt(business.get('faithfulness_avg'))} | "
            f"{fmt(business.get('required_point_recall_avg'))} | "
            f"{fmt(business.get('optional_point_coverage_avg'))} | "
            f"{business.get('fp_count_total')} | {business.get('fn_required_count_total')} |"
        ),
        (
            f"| RAGAS claim flow | {claim_flow.get('evaluated_count')} | "
            f"{fmt(claim_flow.get('answer_correctness_avg'))} | "
            f"{fmt(claim_flow.get('faithfulness_avg'))} | n/a | n/a | "
            f"{claim_flow.get('fp_count_total')} | {claim_flow.get('fn_required_count_total')} |"
        ),
        (
            f"| Native RAGAS | {native.get('evaluated_count')} | "
            f"{fmt(native.get('answer_correctness_avg'))} | "
            f"{fmt(native.get('faithfulness_avg'))} | n/a | n/a | n/a | n/a |"
        ),
        "",
        f"- RAGAS claim-flow failed rows: {claim_flow.get('failed_count')}",
        f"- RAGAS claim-flow reason counts: `{claim_flow.get('metric_reason_counts')}`",
        f"- Native RAGAS failed rows: {native.get('failed_count')}",
        f"- Native RAGAS reason counts: `{native.get('metric_reason_counts')}`",
        f"- Native RAGAS backend counts: `{native.get('metric_backend_counts')}`",
        f"- Retrieved context count avg: {fmt(summary.get('retrieved_context_count_avg'))}",
        f"- Run status counts: `{summary.get('run_status_counts')}`",
        "",
        "## Metric Definitions",
        "",
        "- Business required points: current audited required/optional answer-point judge. Required points drive TP/FN; correct optional points do not count as FP or FN.",
        "- RAGAS claim flow: LLM judge decomposes the reference answer and Agent response into atomic factual claims, then computes TP/FP/FN and F1. It does not use audited required/optional points.",
        "- Native RAGAS: native AnswerCorrectness and Faithfulness. It does not read gold required/optional points and does not fall back to `deepseek_formula`.",
        "",
        "## By Question Type",
        "",
        "| Question type | Queries | Business AC | Claim-flow AC | Native AC | Business Faith | Claim-flow Faith | Native Faith |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in (summary.get("by_question_type") or {}).items():
        lines.append(
            f"| {group} | {item.get('query_count')} | "
            f"{fmt(item.get('business_answer_correctness_avg'))} | "
            f"{fmt(item.get('claim_flow_answer_correctness_avg'))} | "
            f"{fmt(item.get('native_answer_correctness_avg'))} | "
            f"{fmt(item.get('business_faithfulness_avg'))} | "
            f"{fmt(item.get('claim_flow_faithfulness_avg'))} | "
            f"{fmt(item.get('native_faithfulness_avg'))} |"
        )
    lines.extend(
        [
            "",
            "## By Bundle Type",
            "",
            "| Bundle type | Queries | Business AC | Claim-flow AC | Native AC | Business Faith | Claim-flow Faith | Native Faith |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, item in (summary.get("by_bundle_type") or {}).items():
        lines.append(
            f"| {group} | {item.get('query_count')} | "
            f"{fmt(item.get('business_answer_correctness_avg'))} | "
            f"{fmt(item.get('claim_flow_answer_correctness_avg'))} | "
            f"{fmt(item.get('native_answer_correctness_avg'))} | "
            f"{fmt(item.get('business_faithfulness_avg'))} | "
            f"{fmt(item.get('claim_flow_faithfulness_avg'))} | "
            f"{fmt(item.get('native_faithfulness_avg'))} |"
        )
    lines.extend(["", "## Sample Results", ""])
    for item in report.get("sample_results") or []:
        business_metrics = item.get("business_required_points") or {}
        claim_flow_metrics = item.get("ragas_claim_flow") or {}
        native_metrics = item.get("native_ragas") or {}
        lines.extend(
            [
                f"### {item.get('query_id')}",
                "",
                f"- Question: {item.get('question')}",
                f"- Run status: `{item.get('run_status')}`",
                f"- Context count: {item.get('retrieved_context_count')}",
                f"- Business AC/Faith: {fmt(business_metrics.get('answer_correctness'))} / {fmt(business_metrics.get('faithfulness'))}",
                f"- RAGAS claim-flow AC/Faith: {fmt(claim_flow_metrics.get('answer_correctness'))} / {fmt(claim_flow_metrics.get('faithfulness'))}",
                f"- Native RAGAS AC/Faith: {fmt(native_metrics.get('answer_correctness'))} / {fmt(native_metrics.get('faithfulness'))}",
                f"- Claim-flow reason: `{claim_flow_metrics.get('reason')}`",
                f"- Native reason: `{native_metrics.get('reason')}`",
                f"- Answer preview: {item.get('agent_response_preview')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def print_progress(
    args: argparse.Namespace,
    index: int,
    total: int,
    result: dict[str, Any],
    *,
    resumed: bool,
) -> None:
    every = max(1, args.progress_every)
    if index != 1 and index % every != 0 and index != total:
        return
    business = metric_at(result, "business_required_points")
    claim_flow = metric_at(result, "ragas_claim_flow")
    native = metric_at(result, "native_ragas")
    print(
        "[case-agent-ragas-comparison] "
        f"progress={index}/{total} "
        f"query_id={result.get('query_id')} "
        f"run={result.get('run_status')} "
        f"contexts={result.get('retrieved_context_count')} "
        f"business_ac={fmt(business.get('answer_correctness'))} "
        f"claim_flow_ac={fmt(claim_flow.get('answer_correctness'))} "
        f"native_ac={fmt(native.get('answer_correctness'))} "
        f"native_faith={fmt(native.get('faithfulness'))} "
        f"{'resumed' if resumed else 'new'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
