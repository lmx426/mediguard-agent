"""Compare two Case Agent retrieval runs against audited evidence groups.

This analysis does not calculate Ragas metrics. It reads official metric values
already written by the Native Ragas evaluator and separately measures audited
evidence-group retrieval coverage from node ids or context anchors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_REPORT_ROOT,
)


DEFAULT_GOLD = (
    DEFAULT_EVAL_ROOT / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_BASELINE = (
    DEFAULT_EVAL_ROOT / "policy_case_agent_ragas_current_online_197_results.jsonl"
)
DEFAULT_CANDIDATE = (
    DEFAULT_EVAL_ROOT / "policy_case_agent_ragas_dual_rrf_197_results.jsonl"
)
DEFAULT_REPORT_JSON = DEFAULT_REPORT_ROOT / "policy_filter_retrieval_ab_report.json"
DEFAULT_REPORT_MD = DEFAULT_REPORT_ROOT / "policy_filter_retrieval_ab_report.md"
HIT_K_VALUES = (6, 8, 10, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strict-only and dual-RRF Case Agent retrieval runs."
    )
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(args.gold_jsonl)
    baseline_rows = read_jsonl(args.baseline_jsonl)
    candidate_rows = read_jsonl(args.candidate_jsonl)
    report = build_ab_report(
        gold_rows=gold_rows,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_path=args.baseline_jsonl,
        candidate_path=args.candidate_jsonl,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_ab_report(
    *,
    gold_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    gold_by_id = {str(row.get("query_id")): row for row in gold_rows}
    baseline = summarize_run(gold_by_id=gold_by_id, result_rows=baseline_rows)
    candidate = summarize_run(gold_by_id=gold_by_id, result_rows=candidate_rows)
    comparable = sorted(
        set(gold_by_id)
        & {str(row.get("query_id")) for row in baseline_rows}
        & {str(row.get("query_id")) for row in candidate_rows}
    )
    return {
        "status": "complete",
        "gold_query_count": len(gold_by_id),
        "comparable_query_count": len(comparable),
        "baseline_path": str(baseline_path or ""),
        "candidate_path": str(candidate_path or ""),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: _delta(candidate.get(key), baseline.get(key))
            for key in (
                "any_required_group_hit_rate",
                "all_required_groups_hit_rate",
                "required_group_mean_recall",
                "required_evidence_group_filter_retention_rate",
                "gold_chunk_any_hit_rate",
                "gold_chunk_all_group_hit_rate",
                "answer_correctness_average",
                "faithfulness_average",
                "metric_error_rate",
                "no_context_rate",
                "wrong_jurisdiction_context_rate",
                "context_count_avg",
                "context_chars_avg",
                "latency_p95_seconds",
            )
        },
        "nested_delta": {
            "hit_at_k": {
                key: _delta(
                    (candidate.get("hit_at_k") or {}).get(key),
                    (baseline.get("hit_at_k") or {}).get(key),
                )
                for key in {str(k) for k in HIT_K_VALUES}
            },
            "all_required_groups_hit_at_k": {
                key: _delta(
                    (candidate.get("all_required_groups_hit_at_k") or {}).get(key),
                    (baseline.get("all_required_groups_hit_at_k") or {}).get(key),
                )
                for key in {str(k) for k in HIT_K_VALUES}
            },
        },
        "boundary": {
            "ragas_metrics_recomputed": False,
            "gold_filters_injected": False,
            "coverage_method": "node_id_or_audited_anchor_terms",
            "hit_at_k_uses_result_order": True,
            "retrieved_context_metadata_available": False,
        },
    }


def summarize_run(
    *,
    gold_by_id: dict[str, dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = 0
    any_hit = 0
    all_hit = 0
    recalls: list[float] = []
    no_context = 0
    wrong_jurisdiction_contexts = 0
    context_total = 0
    context_counts: list[int] = []
    context_chars: list[int] = []
    latencies: list[float] = []
    ac_values: list[float] = []
    faith_values: list[float] = []
    metric_error_rows = 0
    status_counts: dict[str, int] = {}
    total_required_groups = 0
    total_required_groups_matched = 0
    any_gold_chunk_hit = 0
    all_gold_chunk_group_hit = 0
    hit_at_k_counts = {k: 0 for k in HIT_K_VALUES}
    all_group_hit_at_k_counts = {k: 0 for k in HIT_K_VALUES}
    diagnostics_accumulator: dict[str, list[float]] = {
        "strict_candidate_count": [],
        "recall_candidate_count": [],
        "deduplicated_candidate_count": [],
        "rerank_input_count": [],
        "final_context_count": [],
        "evidence_group_count": [],
        "strict_only_count": [],
        "recall_only_count": [],
        "branch_overlap_count": [],
    }
    fallback_recall_only_count = 0
    for row in result_rows:
        query_id = str(row.get("query_id") or "")
        gold = gold_by_id.get(query_id)
        if gold is None:
            continue
        total += 1
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        contexts = retrieved_contexts(row)
        node_ids = retrieved_node_ids(row)
        if not contexts and not node_ids:
            no_context += 1
        context_total += len(contexts)
        context_counts.append(len(contexts))
        context_chars.append(sum(len(context) for context in contexts))
        wrong_jurisdiction_contexts += wrong_jurisdiction_count(
            gold=gold,
            contexts=contexts,
        )
        groups = required_evidence_groups(gold)
        matched = sum(1 for group in groups if evidence_group_matches(group=group, contexts=contexts, node_ids=node_ids))
        total_required_groups += len(groups)
        total_required_groups_matched += matched
        if groups:
            recall = matched / len(groups)
            recalls.append(recall)
            if matched > 0:
                any_hit += 1
            if matched == len(groups):
                all_hit += 1
        if any_gold_evidence_matches(gold=gold, contexts=contexts, node_ids=node_ids):
            any_gold_chunk_hit += 1
        if groups and all(
            evidence_group_matches(group=group, contexts=contexts, node_ids=node_ids)
            for group in groups
        ):
            all_gold_chunk_group_hit += 1
        for k in HIT_K_VALUES:
            contexts_at_k = contexts[:k]
            if any_gold_evidence_matches(gold=gold, contexts=contexts_at_k, node_ids=node_ids):
                hit_at_k_counts[k] += 1
            if groups and all(
                evidence_group_matches(group=group, contexts=contexts_at_k, node_ids=node_ids)
                for group in groups
            ):
                all_group_hit_at_k_counts[k] += 1
        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            latencies.append(float(elapsed))
        metric_values = row.get("metrics")
        metric_values = metric_values if isinstance(metric_values, dict) else {}
        _append_metric(ac_values, metric_values.get("answer_correctness"))
        _append_metric(faith_values, metric_values.get("faithfulness"))
        if any_metric_error(metric_values):
            metric_error_rows += 1
        diagnostics = retrieval_diagnostics(row)
        for key, values in diagnostics_accumulator.items():
            value = diagnostics.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if diagnostics.get("fallback_mode") == "recall_only":
            fallback_recall_only_count += 1

    return {
        "query_count": total,
        "status_counts": status_counts,
        "any_required_group_hit_rate": _rate(any_hit, total),
        "all_required_groups_hit_rate": _rate(all_hit, total),
        "required_group_mean_recall": mean(recalls) if recalls else None,
        "required_evidence_group_filter_retention_rate": _rate(
            total_required_groups_matched,
            total_required_groups,
        ),
        "gold_chunk_any_hit_rate": _rate(any_gold_chunk_hit, total),
        "gold_chunk_all_group_hit_rate": _rate(all_gold_chunk_group_hit, total),
        "hit_at_k": {
            str(k): _rate(hit_at_k_counts[k], total)
            for k in HIT_K_VALUES
        },
        "all_required_groups_hit_at_k": {
            str(k): _rate(all_group_hit_at_k_counts[k], total)
            for k in HIT_K_VALUES
        },
        "no_context_rate": _rate(no_context, total),
        "context_count_avg": mean(context_counts) if context_counts else None,
        "context_chars_avg": mean(context_chars) if context_chars else None,
        "wrong_jurisdiction_context_rate": _rate(
            wrong_jurisdiction_contexts,
            context_total,
        ),
        "answer_correctness_scored": len(ac_values),
        "answer_correctness_average": mean(ac_values) if ac_values else None,
        "faithfulness_scored": len(faith_values),
        "faithfulness_average": mean(faith_values) if faith_values else None,
        "metric_error_rows": metric_error_rows,
        "metric_error_rate": _rate(metric_error_rows, total),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "retrieval_diagnostics_avg": {
            key: (mean(values) if values else None)
            for key, values in diagnostics_accumulator.items()
        },
        "recall_only_fallback_count": fallback_recall_only_count,
    }


def required_evidence_groups(row: dict[str, Any]) -> list[dict[str, Any]]:
    groups = row.get("gold_evidence_groups")
    return [
        group
        for group in (groups if isinstance(groups, list) else [])
        if isinstance(group, dict) and group.get("required") is not False
    ]


def evidence_group_matches(
    *,
    group: dict[str, Any],
    contexts: list[str],
    node_ids: set[str],
) -> bool:
    gold_node_ids = {
        str(value)
        for value in group.get("gold_node_ids") or []
        if str(value)
    }
    if gold_node_ids & node_ids:
        return True
    anchor_sets: list[list[str]] = []
    for evidence_ref in group.get("evidence_refs") or []:
        if not isinstance(evidence_ref, dict):
            continue
        anchors = [
            str(value)
            for value in evidence_ref.get("anchor_terms") or []
            if len(str(value).strip()) >= 2
        ]
        if anchors:
            anchor_sets.append(anchors)
    for context in contexts:
        for anchors in anchor_sets:
            required_matches = 1 if len(anchors) == 1 else 2
            if sum(1 for anchor in anchors if anchor in context) >= required_matches:
                return True
    return False


def any_gold_evidence_matches(
    *,
    gold: dict[str, Any],
    contexts: list[str],
    node_ids: set[str],
) -> bool:
    return any(
        evidence_group_matches(group=group, contexts=contexts, node_ids=node_ids)
        for group in required_evidence_groups(gold)
    )


def retrieved_contexts(row: dict[str, Any]) -> list[str]:
    ragas_input = row.get("ragas_input")
    ragas_input = ragas_input if isinstance(ragas_input, dict) else {}
    values = ragas_input.get("retrieved_contexts") or row.get("retrieved_contexts")
    return [str(value) for value in values or [] if str(value).strip()]


def retrieved_node_ids(row: dict[str, Any]) -> set[str]:
    values = row.get("retrieved_node_ids") or []
    return {str(value) for value in values if str(value).strip()}


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def retrieval_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("retrieval_diagnostics")
    if isinstance(direct, dict):
        return direct
    for key in ("mcp_safe_summary", "safe_summary"):
        payload = row.get(key)
        if isinstance(payload, dict) and isinstance(payload.get("retrieval_diagnostics"), dict):
            return payload["retrieval_diagnostics"]
    return {}


def any_metric_error(metrics: dict[str, Any]) -> bool:
    return any(
        isinstance(payload, dict) and payload.get("status") == "metric_error"
        for payload in metrics.values()
    )


def wrong_jurisdiction_count(
    *,
    gold: dict[str, Any],
    contexts: list[str],
) -> int:
    expected = set(_string_list((gold.get("filters") or {}).get("jurisdiction")))
    if not expected:
        return 0
    wrong = 0
    for context in contexts:
        jurisdiction = inferred_context_jurisdiction(context)
        if jurisdiction and jurisdiction not in expected and jurisdiction != "national":
            wrong += 1
    return wrong


def inferred_context_jurisdiction(context: str) -> str:
    text = str(context or "")
    for field in ("地区", "jurisdiction"):
        match = re.search(rf"{field}\s*[:：]\s*(national|beijing|shanghai)", text)
        if match:
            return match.group(1)
    if "北京市" in text or "北京医保" in text:
        return "beijing"
    if "上海市" in text or "上海医保" in text:
        return "shanghai"
    if "国家医保" in text or "国家医疗保障局" in text or "全国" in text:
        return "national"
    return ""


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    delta = report["delta"]
    nested_delta = report.get("nested_delta") or {}
    rows = [
        ("Any required group hit", "any_required_group_hit_rate"),
        ("All required groups hit", "all_required_groups_hit_rate"),
        ("Required group mean recall", "required_group_mean_recall"),
        ("Required group retention", "required_evidence_group_filter_retention_rate"),
        ("Gold chunk any hit", "gold_chunk_any_hit_rate"),
        ("Gold chunk all-group hit", "gold_chunk_all_group_hit_rate"),
        ("Hit@6", ("hit_at_k", "6")),
        ("Hit@8", ("hit_at_k", "8")),
        ("Hit@10", ("hit_at_k", "10")),
        ("Hit@12", ("hit_at_k", "12")),
        ("Answer Correctness", "answer_correctness_average"),
        ("Faithfulness", "faithfulness_average"),
        ("Metric error rate", "metric_error_rate"),
        ("No context", "no_context_rate"),
        ("Wrong jurisdiction", "wrong_jurisdiction_context_rate"),
        ("Avg contexts", "context_count_avg"),
        ("Avg chars", "context_chars_avg"),
        ("Latency p95 seconds", "latency_p95_seconds"),
    ]
    lines = [
        "# Policy Filter Retrieval A/B Report",
        "",
        f"- Comparable queries: `{report['comparable_query_count']}`",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        if isinstance(key, tuple):
            metric_key, sub_key = key
            lines.append(
                f"| {label} | {_fmt((baseline.get(metric_key) or {}).get(sub_key))} | "
                f"{_fmt((candidate.get(metric_key) or {}).get(sub_key))} | "
                f"{_fmt((nested_delta.get(metric_key) or {}).get(sub_key))} |"
            )
            continue
        lines.append(
            f"| {label} | {_fmt(baseline.get(key))} | "
            f"{_fmt(candidate.get(key))} | {_fmt(delta.get(key))} |"
        )
    lines.extend(
        [
            "",
            "Ragas values are read from the official Native Ragas output; this script does not recompute metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _append_metric(target: list[float], payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("status") != "scored":
        return
    value = payload.get("value")
    if isinstance(value, (int, float)):
        target.append(float(value))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _delta(candidate: Any, baseline: Any) -> float | None:
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return float(candidate) - float(baseline)
    return None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
