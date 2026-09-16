"""Evaluate online L3 filter generation against the audited Policy RAG labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.backend.application.agent.expert_agent.service.orchestrator import (
    ExpertAgentService,
)
from src.backend.application.agent.expert_agent.service.policy_filter_contract import (
    PolicyFilters,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    PolicyFilterOutputError,
)
from src.backend.application.agent.runtime.gateways.deepseek import (
    DeepSeekModelGateway,
)
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    DEFAULT_EVAL_ROOT,
    DEFAULT_REPORT_ROOT,
    first_env,
    load_env_file,
    write_jsonl,
)


DEFAULT_SOURCE = (
    DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_RESULTS = DEFAULT_EVAL_ROOT / "policy_filter_generation_results.jsonl"
DEFAULT_REPORT_JSON = DEFAULT_REPORT_ROOT / "policy_filter_generation_report.json"
DEFAULT_REPORT_MD = DEFAULT_REPORT_ROOT / "policy_filter_generation_report.md"
FILTER_FIELDS = ("jurisdiction", "policy_domain", "content_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate registered LLM PolicyFilter generation on audited labels."
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    api_key = first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_DEEPSEEK_API_KEY",
        "MEDIGUARD_CASE_AGENT_DEEPSEEK_API_KEY",
        "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
        "MEDIGUARD_DEEPSEEK_API_KEY",
    )
    if not api_key:
        raise RuntimeError("DeepSeek API key for PolicyFilter generation is missing")
    base_url = args.base_url or first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_BASE_URL",
        "MEDIGUARD_CASE_AGENT_BASE_URL",
        "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
        "MEDIGUARD_LLM_BASE_URL",
    ) or "https://api.deepseek.com"
    model = args.model or first_env(
        "MEDIGUARD_EXPERT_ANALYSIS_MODEL",
        "MEDIGUARD_CASE_AGENT_MODEL",
        "MEDIGUARD_POLICY_RAG_EVAL_MODEL",
        "MEDIGUARD_LLM_MODEL",
    ) or "deepseek-v4-flash"
    gateway = DeepSeekModelGateway(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=args.timeout_seconds,
        thinking_enabled=False,
        reasoning_effort="low",
    )
    rows = read_jsonl(args.source_jsonl)
    selected = rows[max(args.offset, 0) :]
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]

    service = ExpertAgentService()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        query_id = str(row.get("query_id") or "")
        question = str(row.get("question") or "")
        try:
            gold = canonical_filters(row.get("filters") or {})
            plan = service.generate_policy_retrieval_plan(
                user_question=question,
                model_gateway=gateway,
            )
            generated = canonical_filters(plan.get("filters") or {})
            comparison = compare_filters(gold, generated)
            diagnostics = plan.get("filter_diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
            result = {
                "query_id": query_id,
                "status": "scored",
                "user_input": question,
                "gold_filters": gold,
                "generated_filters": generated,
                "exact_match": comparison["exact_match"],
                "field_exact": comparison["field_exact"],
                "extra_values": comparison["extra_values"],
                "missing_values": comparison["missing_values"],
                "valid_on_first_call": bool(
                    diagnostics.get("valid_on_first_call", True)
                ),
                "repair_count": int(diagnostics.get("repair_count") or 0),
                "filter_reason": diagnostics.get("filter_reason"),
            }
        except Exception as exc:
            result = {
                "query_id": query_id,
                "status": "error",
                "user_input": question,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc)[:500],
            }
            if isinstance(exc, PolicyFilterOutputError):
                result["validation_errors"] = exc.errors
        results.append(result)
        if args.progress_every > 0 and (
            index % args.progress_every == 0 or index == len(selected)
        ):
            print(
                f"[policy-filter-eval] {index}/{len(selected)} "
                f"query={query_id} status={result['status']} "
                f"exact={result.get('exact_match', 'n/a')}",
                flush=True,
            )

    write_jsonl(args.results_jsonl, results)
    report = build_report(
        results,
        source=args.source_jsonl,
        model=model,
        results_path=args.results_jsonl,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def canonical_filters(value: dict[str, Any]) -> dict[str, Any]:
    return PolicyFilters.model_validate(value).model_dump(mode="json")


def compare_filters(
    gold: dict[str, Any],
    generated: dict[str, Any],
) -> dict[str, Any]:
    field_exact: dict[str, bool] = {}
    extra: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for field in FILTER_FIELDS:
        gold_values = {str(item) for item in gold.get(field) or []}
        generated_values = {str(item) for item in generated.get(field) or []}
        field_exact[field] = gold_values == generated_values
        extra[field] = sorted(generated_values - gold_values)
        missing[field] = sorted(gold_values - generated_values)
    return {
        "exact_match": all(field_exact.values()),
        "field_exact": field_exact,
        "extra_values": extra,
        "missing_values": missing,
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    source: Path,
    model: str,
    results_path: Path,
) -> dict[str, Any]:
    scored = [row for row in results if row.get("status") == "scored"]
    field_metrics: dict[str, dict[str, Any]] = {}
    for field in FILTER_FIELDS:
        exact_count = sum(
            bool((row.get("field_exact") or {}).get(field)) for row in scored
        )
        tp = fp = fn = 0
        for row in scored:
            gold = set((row.get("gold_filters") or {}).get(field) or [])
            generated = set((row.get("generated_filters") or {}).get(field) or [])
            tp += len(gold.intersection(generated))
            fp += len(generated - gold)
            fn += len(gold - generated)
        field_metrics[field] = {
            "exact_count": exact_count,
            "exact_rate": exact_count / len(scored) if scored else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "extra_count": fp,
            "missing_count": fn,
        }
    exact_count = sum(bool(row.get("exact_match")) for row in scored)
    repaired = [int(row.get("repair_count") or 0) for row in scored]
    return {
        "status": "ok" if len(scored) == len(results) else "partial",
        "query_count": len(results),
        "scored_count": len(scored),
        "error_count": len(results) - len(scored),
        "model": model,
        "source_jsonl": str(source),
        "results_jsonl": str(results_path),
        "exact_match_count": exact_count,
        "exact_match_rate": exact_count / len(scored) if scored else None,
        "valid_on_first_call_rate": (
            sum(bool(row.get("valid_on_first_call")) for row in scored) / len(scored)
            if scored
            else None
        ),
        "average_repair_count": mean(repaired) if repaired else None,
        "field_metrics": field_metrics,
        "status_counts": dict(Counter(str(row.get("status")) for row in results)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PolicyFilter Generation Evaluation",
        "",
        f"- Status: `{report['status']}`",
        f"- Queries: {report['query_count']}",
        f"- Model: `{report['model']}`",
        f"- Exact match: {format_rate(report.get('exact_match_rate'))}",
        f"- Valid on first call: {format_rate(report.get('valid_on_first_call_rate'))}",
        "",
        "| Field | Exact | Precision | Recall | Extra | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, item in report.get("field_metrics", {}).items():
        lines.append(
            f"| {field} | {format_rate(item.get('exact_rate'))} | "
            f"{format_rate(item.get('precision'))} | {format_rate(item.get('recall'))} | "
            f"{item.get('extra_count', 0)} | {item.get('missing_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "Gold filters are used only for comparison and are not passed to the LLM.",
            "Policy RAG MCP, chunking, ranking, answer slots, and Ragas metrics are unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[policy-filter-eval] failed: {exc}", file=sys.stderr)
        raise
