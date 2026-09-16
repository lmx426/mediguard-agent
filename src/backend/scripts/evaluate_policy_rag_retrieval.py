"""Evaluate Policy RAG retrieval quality on confirmed closed-loop cases."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.search_policy_evidence_uc import (
    SearchPolicyEvidenceUseCase,
)
from src.backend.application.policy_rag.schemas import RETRIEVAL_PROFILES
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bge_reranker import BgePolicyReranker
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
)
from src.backend.infrastructure.policy_rag.retriever_factory import (
    build_policy_retriever,
)
from src.backend.scripts.policy_rag_eval_cases import (
    CORE_CASE_QUERIES,
)


TOP_K_VALUES = [3, 5, 10]
DEFAULT_FETCH_K = 20
DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_2.jsonl"
)
DEFAULT_CORE_QUERIES_JSONL = DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_core_queries.jsonl"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_retrieval_quality_report.json"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_retrieval_quality_report.md"
DEFAULT_AUDIT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_eval_standard_audit.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Policy RAG retrieval quality.")
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument(
        "--retrieval-strategy",
        choices=tuple(RETRIEVAL_PROFILES),
        default="dense",
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--core-queries-jsonl", type=Path, default=DEFAULT_CORE_QUERIES_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--standard-audit-report",
        type=Path,
        default=DEFAULT_AUDIT_REPORT_MD,
    )
    return parser.parse_args()


def build_use_case(args: argparse.Namespace) -> SearchPolicyEvidenceUseCase:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    retriever = build_policy_retriever(
        strategy=args.retrieval_strategy,
        embedder=embedder,
    )
    reranker = BgePolicyReranker(
        cache_folder=args.cache_folder,
        device=args.reranker_device or args.device,
        allow_unavailable=True,
    )
    return SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
        retrieval_strategy=args.retrieval_strategy,
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    write_jsonl(args.core_queries_jsonl, CORE_CASE_QUERIES)
    eval_queries = load_jsonl(args.eval_set_jsonl)
    use_case = build_use_case(args)

    mode_reports = []
    for mode in [
        {"name": f"{args.retrieval_strategy}_only", "rerank": False},
        {"name": f"{args.retrieval_strategy}_with_reranker", "rerank": True},
    ]:
        query_reports = []
        total_queries = len(eval_queries)
        print(
            f"[policy-rag-eval] mode={mode['name']} queries={total_queries}",
            flush=True,
        )
        for index, query in enumerate(eval_queries, start=1):
            started = time.perf_counter()
            response = use_case.search(
                {
                    "question": query["question"],
                    "filters": query["filters"],
                    "top_k": max(TOP_K_VALUES),
                    "fetch_k": args.fetch_k,
                    "rerank": mode["rerank"],
                }
            )
            elapsed = time.perf_counter() - started
            query_reports.append(evaluate_query(query, response, elapsed_seconds=elapsed))
            if index == 1 or index % 10 == 0 or index == total_queries:
                print(
                    f"[policy-rag-eval] mode={mode['name']} "
                    f"progress={index}/{total_queries}",
                    flush=True,
                )
        mode_reports.append(
            {
                "mode": mode["name"],
                "rerank": mode["rerank"],
                "query_count": len(query_reports),
                "queries": query_reports,
                "summary": summarize_mode(query_reports),
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "core_queries_jsonl": str(args.core_queries_jsonl),
        "fetch_k": args.fetch_k,
        "retrieval_strategy": args.retrieval_strategy,
        "top_k_values": TOP_K_VALUES,
        "modes": mode_reports,
        "passed": all(mode["summary"]["supported_queries_with_results"] for mode in mode_reports),
        "eval_schema_version": "policy_rag_eval_v1_2",
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_llm_judge": False,
            "calculates_token_usage": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def evaluate_query(
    query: dict[str, Any],
    response: dict[str, Any],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    evidence = list(response.get("evidence") or [])
    expected_status = str(query.get("expected_status") or "supported")
    evidence_status = _evidence_status(query)
    gap_type = str(query.get("gap_type") or "none")
    coverage_gap_expected = evidence_status == "none"
    warnings_text = " ".join(str(item) for item in response.get("warnings") or []).lower()
    metadata_filter_applied = bool(response.get("metadata_filter_applied", True))
    requested_filters = dict(query.get("filters") or {})
    applied_filters = dict(response.get("filters_used") or {})
    rerank_info = dict(response.get("rerank") or {})
    if metadata_filter_applied:
        unsupported_filters = sorted(
            set(requested_filters) - set(applied_filters)
        )
    else:
        unsupported_filters = []
    coverage_gap_behavior = _classify_coverage_gap(
        query=query,
        evidence=evidence,
        warnings_text=warnings_text,
        unsupported_filters=unsupported_filters,
    )
    coverage_gap_detected = coverage_gap_behavior in {
        "no_result",
        "unsupported_filter",
        "related_but_insufficient",
    }
    per_k = {
        str(k): metrics_at_k(
            query,
            response,
            k=k,
            evidence_status=evidence_status,
        )
        for k in TOP_K_VALUES
    }
    return {
        "case_id": query["case_id"],
        "query_id": query["query_id"],
        "question": query["question"],
        "filters": query["filters"],
        "expected_status": expected_status,
        "evidence_status": evidence_status,
        "gap_type": gap_type,
        "metric_eligibility": query.get("metric_eligibility") or {},
        "known_limitations": query.get("known_limitations", []),
        "coverage_gap_reason": query.get("coverage_gap_reason"),
        "elapsed_seconds": elapsed_seconds,
        "response_status": response.get("status"),
        "retrieval_mode": response.get("retrieval_mode"),
        "metadata_filter_applied": metadata_filter_applied,
        "filters_requested": requested_filters,
        "filters_applied": applied_filters,
        "rerank_requested": bool(rerank_info.get("requested")),
        "rerank_conditional": bool(rerank_info.get("conditional")),
        "rerank_condition_met": bool(rerank_info.get("condition_met")),
        "rerank_triggered": bool(rerank_info.get("triggered")),
        "rerank_trigger_reason": rerank_info.get("trigger_reason"),
        "result_count": response.get("result_count"),
        "warnings": response.get("warnings"),
        "coverage_gap_expected": coverage_gap_expected,
        "coverage_gap_detected": coverage_gap_detected,
        "coverage_gap_behavior": coverage_gap_behavior,
        "unsupported_filters": unsupported_filters,
        "per_k": per_k,
        "top_source_ids": [item.get("source_id") for item in evidence[:5]],
        "top_evidence": evidence[:3],
    }


def _classify_coverage_gap(
    *,
    query: dict[str, Any],
    evidence: list[dict[str, Any]],
    warnings_text: str,
    unsupported_filters: list[str],
) -> str:
    """Distinguish a true gap from related evidence without a complete rule."""

    if _evidence_status(query) != "none":
        return "not_applicable"
    if unsupported_filters:
        return "unsupported_filter"
    if (
        not evidence
        or "no results" in warnings_text
        or "coverage" in warnings_text
        or "unavailable" in warnings_text
    ):
        return "no_result"
    if any(
        _matches_requested_filters(item, dict(query.get("filters") or {}))
        for item in evidence
    ):
        return "related_but_insufficient"
    return "unrelated_results"


def metrics_at_k(
    query: dict[str, Any],
    response: dict[str, Any],
    *,
    k: int,
    evidence_status: str,
) -> dict[str, Any]:
    evidence = list(response.get("evidence") or [])[:k]
    expected_groups = list(query.get("expected_source_groups") or [])
    eligible_groups = _eligible_groups(query, expected_groups, evidence_status)
    group_hits = {
        group["group"]: any(_group_source_hit(item, group) for item in evidence)
        for group in expected_groups
    }
    content_group_hits = {
        group["group"]: any(
            _group_content_hit(item, group, query) for item in evidence
        )
        for group in expected_groups
    }
    group_count = len(eligible_groups)
    hit_group_count = sum(
        1 for group in eligible_groups if group_hits.get(group["group"]) is True
    )
    content_hit_group_count = sum(
        1
        for group in eligible_groups
        if content_group_hits.get(group["group"]) is True
    )
    relevant_count = sum(
        1 for item in evidence if _matches_any_group_source(item, eligible_groups)
    )
    content_relevant_count = sum(
        1
        for item in evidence
        if _matches_any_group_content(item, eligible_groups, query)
    )
    nonempty = len(evidence)
    filter_match_count = sum(
        1
        for item in evidence
        if _matches_requested_filters(item, dict(query.get("filters") or {}))
    )
    gold_node_ids = _gold_node_ids_for_groups(query, eligible_groups)
    retrieved_node_ids_ordered = [
        str(item.get("node_id") or "")
        for item in evidence
        if str(item.get("node_id") or "")
    ]
    retrieved_node_ids = set(retrieved_node_ids_ordered)
    exact_node_hit_count = len(retrieved_node_ids.intersection(gold_node_ids))
    exact_node_recall_value = (
        exact_node_hit_count / len(gold_node_ids) if gold_node_ids else None
    )
    first_gold_rank = next(
        (
            rank
            for rank, node_id in enumerate(retrieved_node_ids_ordered, start=1)
            if node_id in gold_node_ids
        ),
        None,
    )
    chunk_hit_value = 1.0 if first_gold_rank is not None else 0.0
    chunk_mrr_value = (
        1.0 / first_gold_rank if first_gold_rank is not None else 0.0
    )
    if not gold_node_ids:
        chunk_hit_value = None
        chunk_mrr_value = None
    content_anchor_recall_value = (
        content_hit_group_count / group_count if group_count else None
    )
    content_anchor_precision_value = content_relevant_count / k
    if evidence_status == "full":
        recall = hit_group_count / group_count if group_count else None
        precision = relevant_count / k
        content_recall = content_anchor_recall_value
        content_precision = content_anchor_precision_value
        exact_node_recall = exact_node_recall_value
        content_anchor_recall = content_anchor_recall_value
        content_anchor_precision = content_anchor_precision_value
        partial_recall = None
        partial_precision = None
        partial_content_recall = None
        partial_content_precision = None
        partial_exact_node_recall = None
        partial_content_anchor_recall = None
        partial_content_anchor_precision = None
    elif evidence_status == "partial":
        recall = None
        precision = None
        content_recall = None
        content_precision = None
        exact_node_recall = None
        content_anchor_recall = None
        content_anchor_precision = None
        partial_recall = hit_group_count / group_count if group_count else None
        partial_precision = relevant_count / k
        partial_content_recall = content_anchor_recall_value
        partial_content_precision = content_anchor_precision_value
        partial_exact_node_recall = exact_node_recall_value
        partial_content_anchor_recall = content_anchor_recall_value
        partial_content_anchor_precision = content_anchor_precision_value
    else:
        recall = None
        precision = None
        content_recall = None
        content_precision = None
        exact_node_recall = None
        content_anchor_recall = None
        content_anchor_precision = None
        partial_recall = None
        partial_precision = None
        partial_content_recall = None
        partial_content_precision = None
        partial_exact_node_recall = None
        partial_content_anchor_recall = None
        partial_content_anchor_precision = None
    return {
        "k": k,
        "result_count_at_k": len(evidence),
        "group_hits": group_hits,
        "content_group_hits": content_group_hits,
        "hit_group_count": hit_group_count,
        "content_hit_group_count": content_hit_group_count,
        "expected_group_count": group_count,
        "recall": recall,
        "precision": precision,
        "content_recall": content_recall,
        "content_precision": content_precision,
        "exact_node_recall": exact_node_recall,
        "content_anchor_recall": content_anchor_recall,
        "content_anchor_precision": content_anchor_precision,
        "partial_recall": partial_recall,
        "partial_precision": partial_precision,
        "partial_content_recall": partial_content_recall,
        "partial_content_precision": partial_content_precision,
        "partial_exact_node_recall": partial_exact_node_recall,
        "partial_content_anchor_recall": partial_content_anchor_recall,
        "partial_content_anchor_precision": partial_content_anchor_precision,
        "metric_evidence_status": evidence_status,
        "relevant_count": relevant_count,
        "content_relevant_count": content_relevant_count,
        "exact_node_hit_count": exact_node_hit_count,
        "gold_exact_node_count": len(gold_node_ids),
        "gold_chunk_count": len(gold_node_ids),
        "chunk_hit_count": exact_node_hit_count,
        "chunk_recall": exact_node_recall_value,
        "chunk_hit": chunk_hit_value,
        "chunk_mrr": chunk_mrr_value,
        "first_gold_rank": first_gold_rank,
        "citeable_rate": (
            sum(1 for item in evidence if item.get("can_cite_as_policy_basis") is True)
            / nonempty
            if nonempty
            else 0.0
        ),
        "source_url_rate": (
            sum(1 for item in evidence if item.get("source_url")) / nonempty
            if nonempty
            else 0.0
        ),
        "filter_precision": filter_match_count / nonempty if nonempty else 0.0,
    }


def _evidence_status(query: dict[str, Any]) -> str:
    value = str(query.get("evidence_status") or "").strip()
    if value in {"full", "partial", "none"}:
        return value
    return {
        "supported": "full",
        "partial_supported": "partial",
        "coverage_gap": "none",
    }.get(str(query.get("expected_status") or ""), "none")


def _eligible_groups(
    query: dict[str, Any],
    expected_groups: list[dict[str, Any]],
    evidence_status: str,
) -> list[dict[str, Any]]:
    if evidence_status == "full":
        return expected_groups
    if evidence_status != "partial":
        return []

    audited_groups = {
        str(group.get("group") or ""): group
        for group in query.get("evidence_groups") or []
    }
    return [
        group
        for group in expected_groups
        if audited_groups.get(str(group.get("group") or ""), {}).get("status")
        == "available"
    ]


def _matches_requested_filters(
    item: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    jurisdictions = set(filters.get("jurisdiction") or [])
    policy_domains = set(filters.get("policy_domain") or [])
    if jurisdictions and item.get("jurisdiction") not in jurisdictions:
        return False
    if policy_domains and item.get("policy_domain") not in policy_domains:
        return False
    return True


def _candidate_source_aliases(item: dict[str, Any]) -> set[str]:
    aliases = {
        str(item.get("source_id") or ""),
        str(item.get("doc_id") or ""),
    }
    return {alias for alias in aliases if alias}


def _group_sources(group: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            group.get("acceptable_sources")
            or group.get("match_any")
            or []
        )
        if str(value)
    }


def _group_source_hit(item: dict[str, Any], group: dict[str, Any]) -> bool:
    aliases = _candidate_source_aliases(item)
    expected = _group_sources(group)
    if aliases.intersection(expected):
        return True
    text = str(item.get("text") or "")
    return any(source_id in text for source_id in expected)


def _anchor_set_hit(text: str, anchor_set: list[str]) -> bool:
    normalized = str(text or "")
    return bool(anchor_set) and all(str(anchor) in normalized for anchor in anchor_set)


def _group_content_hit(
    item: dict[str, Any],
    group: dict[str, Any],
    query: dict[str, Any] | None = None,
) -> bool:
    if not _group_source_hit(item, group):
        return False
    anchor_sets = list(group.get("content_anchor_sets") or [])
    if not anchor_sets:
        return True
    return any(_anchor_set_hit(str(item.get("text") or ""), anchors) for anchors in anchor_sets)


def _matches_any_group_source(
    item: dict[str, Any],
    groups: list[dict[str, Any]],
) -> bool:
    return any(_group_source_hit(item, group) for group in groups)


def _matches_any_group_content(
    item: dict[str, Any],
    groups: list[dict[str, Any]],
    query: dict[str, Any] | None = None,
) -> bool:
    return any(_group_content_hit(item, group, query) for group in groups)


def _gold_node_ids_for_group(
    query: dict[str, Any],
    group_name: str,
) -> set[str]:
    node_ids: set[str] = set()
    for ref in query.get("gold_evidence_refs") or []:
        if str(ref.get("group") or "") != group_name:
            continue
        node_id = str(ref.get("node_id") or "").strip()
        if node_id:
            node_ids.add(node_id)
    if node_ids:
        return node_ids
    for group in query.get("evidence_groups") or []:
        if str(group.get("group") or "") != group_name:
            continue
        node_ids.update(
            str(node_id)
            for node_id in group.get("gold_node_ids") or []
            if str(node_id)
        )
    return node_ids


def _gold_node_ids_for_groups(
    query: dict[str, Any],
    groups: list[dict[str, Any]],
) -> set[str]:
    node_ids: set[str] = set()
    for group in groups:
        node_ids.update(
            _gold_node_ids_for_group(query, str(group.get("group") or ""))
        )
    return node_ids


def summarize_mode(query_reports: list[dict[str, Any]]) -> dict[str, Any]:
    full_queries = [item for item in query_reports if item.get("evidence_status") == "full"]
    partial_queries = [
        item for item in query_reports if item.get("evidence_status") == "partial"
    ]
    coverage_gap_queries = [
        item for item in query_reports if item.get("evidence_status") == "none"
    ]
    available_queries = full_queries + partial_queries
    per_k_summary = {}
    for k in TOP_K_VALUES:
        key = str(k)
        recall_values = [
            item["per_k"][key]["recall"]
            for item in full_queries
            if item["per_k"][key]["recall"] is not None
        ]
        precision_values = [
            item["per_k"][key]["precision"]
            for item in full_queries
            if item["per_k"][key]["precision"] is not None
        ]
        content_recall_values = [
            item["per_k"][key]["content_recall"]
            for item in full_queries
            if item["per_k"][key]["content_recall"] is not None
        ]
        content_precision_values = [
            item["per_k"][key]["content_precision"]
            for item in full_queries
            if item["per_k"][key]["content_precision"] is not None
        ]
        exact_node_recall_values = [
            item["per_k"][key]["exact_node_recall"]
            for item in full_queries
            if item["per_k"][key]["exact_node_recall"] is not None
        ]
        content_anchor_recall_values = [
            item["per_k"][key]["content_anchor_recall"]
            for item in full_queries
            if item["per_k"][key]["content_anchor_recall"] is not None
        ]
        content_anchor_precision_values = [
            item["per_k"][key]["content_anchor_precision"]
            for item in full_queries
            if item["per_k"][key]["content_anchor_precision"] is not None
        ]
        partial_recall_values = [
            item["per_k"][key]["partial_recall"]
            for item in partial_queries
            if item["per_k"][key]["partial_recall"] is not None
        ]
        partial_precision_values = [
            item["per_k"][key]["partial_precision"]
            for item in partial_queries
            if item["per_k"][key]["partial_precision"] is not None
        ]
        partial_content_recall_values = [
            item["per_k"][key]["partial_content_recall"]
            for item in partial_queries
            if item["per_k"][key]["partial_content_recall"] is not None
        ]
        partial_content_precision_values = [
            item["per_k"][key]["partial_content_precision"]
            for item in partial_queries
            if item["per_k"][key]["partial_content_precision"] is not None
        ]
        partial_exact_node_recall_values = [
            item["per_k"][key]["partial_exact_node_recall"]
            for item in partial_queries
            if item["per_k"][key]["partial_exact_node_recall"] is not None
        ]
        partial_content_anchor_recall_values = [
            item["per_k"][key]["partial_content_anchor_recall"]
            for item in partial_queries
            if item["per_k"][key]["partial_content_anchor_recall"] is not None
        ]
        partial_content_anchor_precision_values = [
            item["per_k"][key]["partial_content_anchor_precision"]
            for item in partial_queries
            if item["per_k"][key]["partial_content_anchor_precision"] is not None
        ]
        chunk_recall_values = [
            item["per_k"][key]["chunk_recall"]
            for item in available_queries
            if item["per_k"][key]["chunk_recall"] is not None
        ]
        chunk_hit_values = [
            item["per_k"][key]["chunk_hit"]
            for item in available_queries
            if item["per_k"][key]["chunk_hit"] is not None
        ]
        chunk_mrr_values = [
            item["per_k"][key]["chunk_mrr"]
            for item in available_queries
            if item["per_k"][key]["chunk_mrr"] is not None
        ]
        per_k_summary[key] = {
            "recall_avg": _avg(recall_values),
            "precision_avg": _avg(precision_values),
            "content_recall_avg": _avg(content_recall_values),
            "content_precision_avg": _avg(content_precision_values),
            "exact_node_recall_avg": _avg(exact_node_recall_values),
            "content_anchor_recall_avg": _avg(content_anchor_recall_values),
            "content_anchor_precision_avg": _avg(
                content_anchor_precision_values
            ),
            "partial_recall_avg": _avg(partial_recall_values),
            "partial_precision_avg": _avg(partial_precision_values),
            "partial_content_recall_avg": _avg(partial_content_recall_values),
            "partial_content_precision_avg": _avg(
                partial_content_precision_values
            ),
            "partial_exact_node_recall_avg": _avg(
                partial_exact_node_recall_values
            ),
            "partial_content_anchor_recall_avg": _avg(
                partial_content_anchor_recall_values
            ),
            "partial_content_anchor_precision_avg": _avg(
                partial_content_anchor_precision_values
            ),
            "chunk_recall_avg": _avg(chunk_recall_values),
            "chunk_hit_avg": _avg(chunk_hit_values),
            "chunk_mrr_avg": _avg(chunk_mrr_values),
            "chunk_metric_query_count": len(chunk_recall_values),
            "citeable_rate_avg": _avg(
                [item["per_k"][key]["citeable_rate"] for item in query_reports]
            ),
            "source_url_rate_avg": _avg(
                [item["per_k"][key]["source_url_rate"] for item in query_reports]
            ),
            "filter_precision_avg": _avg(
                [item["per_k"][key]["filter_precision"] for item in query_reports]
            ),
            "full_recall_query_count": sum(
                1
                for item in full_queries
                if item["per_k"][key]["recall"] == 1.0
            ),
            "full_query_count": len(full_queries),
            "partial_query_count": len(partial_queries),
            "retrieval_available_query_count": len(available_queries),
        }
    elapsed = [float(item["elapsed_seconds"]) for item in query_reports]
    query_count = len(query_reports)
    coverage_available_count = len(available_queries)
    rerank_requested_count = sum(
        1 for item in query_reports if item.get("rerank_requested")
    )
    rerank_triggered_count = sum(
        1 for item in query_reports if item.get("rerank_triggered")
    )
    rerank_trigger_reasons = Counter(
        str(item.get("rerank_trigger_reason") or "unknown")
        for item in query_reports
        if item.get("rerank_requested")
    )
    return {
        "query_count": len(query_reports),
        "full_query_count": len(full_queries),
        "partial_query_count": len(partial_queries),
        "retrieval_available_query_count": len(available_queries),
        "coverage_gap_query_count": len(coverage_gap_queries),
        "coverage_gap_detected_count": sum(
            1 for item in coverage_gap_queries if item.get("coverage_gap_detected")
        ),
        "coverage_gap_detection_rate": (
            sum(1 for item in coverage_gap_queries if item.get("coverage_gap_detected"))
            / len(coverage_gap_queries)
            if coverage_gap_queries
            else None
        ),
        "coverage_gap_leakage_count": sum(
            1
            for item in coverage_gap_queries
            if not item.get("coverage_gap_detected")
        ),
        "coverage_gap_leakage_rate": (
            sum(
                1
                for item in coverage_gap_queries
                if not item.get("coverage_gap_detected")
            )
            / len(coverage_gap_queries)
            if coverage_gap_queries
            else None
        ),
        "coverage_rate": coverage_available_count / query_count if query_count else 0.0,
        "full_answerability_rate": len(full_queries) / query_count if query_count else 0.0,
        "supported_queries_with_results": all(
            int(item.get("result_count") or 0) > 0 for item in full_queries
        ),
        "avg_seconds": statistics.fmean(elapsed) if elapsed else 0.0,
        "rerank_requested_count": rerank_requested_count,
        "rerank_triggered_count": rerank_triggered_count,
        "rerank_trigger_rate": (
            rerank_triggered_count / rerank_requested_count
            if rerank_requested_count
            else None
        ),
        "rerank_trigger_reasons": dict(rerank_trigger_reasons),
        "per_k": per_k_summary,
    }


def _avg(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"No evaluation queries found in {path}")
    return rows


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def write_standard_audit_report(
    eval_queries: list[dict[str, Any]],
    eval_path: Path,
    report_path: Path,
) -> None:
    status_counts: dict[str, int] = {}
    for query in eval_queries:
        status = str(query.get("expected_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Policy RAG Evaluation Standard Audit",
        "",
        "This file records the manually confirmed v1.2 evaluation standard. "
        "It is based on the retrieved node正文、结构化表格字段和官方来源 metadata; "
        "document names alone are not treated as content evidence.",
        "",
        f"- Eval set: `{eval_path}`",
        f"- Query count: {len(eval_queries)}",
        f"- Status counts: {status_counts}",
        "- `full`: current corpus contains complete clause/table content and enters strict Recall/Precision.",
        "- `partial`: current corpus supports only the verification direction or partial fields and enters separate partial metrics.",
        "- `none`: the requested evidence or retrieval capability is currently unavailable and enters only coverage-gap metrics.",
        "- `reference_only`: reference material may support discovery, but is not a binding policy basis.",
        "",
        "| Query | Evidence status | Gap type | Strict eligible | Partial eligible | Standard basis |",
        "|---|---|---|---:|---:|---|",
    ]
    for query in eval_queries:
        basis = str(query.get("standard_basis") or "").replace("|", "\\|").replace("\n", " ")
        evidence_status = _evidence_status(query)
        eligibility = query.get("metric_eligibility") or {}
        lines.append(
            f"| `{query.get('query_id')}` | `{evidence_status}` | "
            f"`{query.get('gap_type', 'none')}` | "
            f"{'yes' if eligibility.get('strict_retrieval') else 'no'} | "
            f"{'yes' if eligibility.get('partial_retrieval') else 'no'} | {basis} |"
        )
        limitations = query.get("known_limitations") or []
        if limitations:
            lines.append("")
            lines.append(f"  - Limitations: {'；'.join(str(item) for item in limitations)}")
        gap_reason = query.get("coverage_gap_reason")
        if gap_reason:
            lines.append("")
            lines.append(f"  - Coverage gap: {gap_reason}")
        for group in query.get("expected_source_groups") or []:
            sources = ", ".join(sorted(_group_sources(group)))
            anchors = " / ".join(
                " + ".join(str(anchor) for anchor in anchor_set)
                for anchor_set in group.get("content_anchor_sets") or []
            )
            lines.append(
                f"  - Evidence group `{group.get('group')}`: sources `{sources}`; "
                f"anchors `{anchors}`."
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy RAG Retrieval Quality Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report evaluates retrieval only. It does not use an LLM judge, calculate token usage, connect to Case Agent, or make audit decisions.",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Retrieval strategy: `{report.get('retrieval_strategy', 'dense')}`",
        f"- Top K values: {report.get('top_k_values')}",
        "",
        "## Metric Eligibility",
        "",
        "- Strict Recall/Precision and strict content metrics use `evidence_status=full` only.",
        "- Partial evidence metrics use `evidence_status=partial` only.",
        "- `evidence_status=none` is excluded from retrieval metrics and reported as coverage-gap behavior.",
        "- `exact_node_recall` checks the exact audited node IDs.",
        "- `content_anchor_recall` checks equivalent retrieved nodes by source plus正文 anchor terms.",
        "- Coverage-gap behavior distinguishes `no_result`, `unsupported_filter`, and `related_but_insufficient`.",
        "",
    ]
    for mode in report.get("modes") or []:
        summary = mode.get("summary") or {}
        lines.extend(
            [
                f"## {mode.get('mode')}",
                "",
                f"- Rerank: {mode.get('rerank')}",
                f"- Query count: {summary.get('query_count')}",
                f"- Full evidence query count: {summary.get('full_query_count')}",
                f"- Partial evidence query count: {summary.get('partial_query_count')}",
                f"- Retrieval-available query count: {summary.get('retrieval_available_query_count')}",
                f"- Coverage gap query count: {summary.get('coverage_gap_query_count')}",
                f"- Coverage gap detected count: {summary.get('coverage_gap_detected_count')}",
                f"- Coverage gap detection rate: {_fmt_pct(summary.get('coverage_gap_detection_rate'))}",
                f"- Coverage gap leakage rate: {_fmt_pct(summary.get('coverage_gap_leakage_rate'))}",
                f"- Coverage rate: {_fmt_pct(summary.get('coverage_rate'))}",
                f"- Full answerability rate: {_fmt_pct(summary.get('full_answerability_rate'))}",
                f"- Avg seconds: {summary.get('avg_seconds'):.3f}",
                "",
                "| K | Chunk Recall@K | Hit@K | MRR@K | Strict Recall | Strict Precision | Exact Node Recall | Content Anchor Recall | Content Anchor Precision | Partial Recall | Partial Precision | Partial Exact Node Recall | Partial Content Anchor Recall | Partial Content Anchor Precision | Citeable Rate | Source URL Rate | Filter Precision | Full Recall |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for k, item in (summary.get("per_k") or {}).items():
            lines.append(
                f"| {k} | {_fmt(item.get('chunk_recall_avg'))} | {_fmt(item.get('chunk_hit_avg'))} | {_fmt(item.get('chunk_mrr_avg'))} | {_fmt(item.get('recall_avg'))} | {_fmt(item.get('precision_avg'))} | {_fmt(item.get('exact_node_recall_avg'))} | {_fmt(item.get('content_anchor_recall_avg'))} | {_fmt(item.get('content_anchor_precision_avg'))} | {_fmt(item.get('partial_recall_avg'))} | {_fmt(item.get('partial_precision_avg'))} | {_fmt(item.get('partial_exact_node_recall_avg'))} | {_fmt(item.get('partial_content_anchor_recall_avg'))} | {_fmt(item.get('partial_content_anchor_precision_avg'))} | {_fmt(item.get('citeable_rate_avg'))} | {_fmt(item.get('source_url_rate_avg'))} | {_fmt(item.get('filter_precision_avg'))} | {item.get('full_recall_query_count')}/{item.get('full_query_count')} |"
            )
        lines.extend(["", "### Query Details", ""])
        for query in mode.get("queries") or []:
            k5 = query["per_k"]["5"]
            lines.extend(
                [
                    f"#### {query.get('query_id')}",
                    "",
                    f"- Case: {query.get('case_id')}",
                    f"- Expected status: {query.get('expected_status')}",
                    f"- Evidence status: {query.get('evidence_status')}",
                    f"- Gap type: {query.get('gap_type')}",
                    f"- Result count: {query.get('result_count')}",
                    f"- Top source ids: `{query.get('top_source_ids')}`",
                    f"- Recall@5: {_fmt(k5.get('recall'))}",
                    f"- Precision@5: {_fmt(k5.get('precision'))}",
                    f"- Exact Node Recall@5: {_fmt(k5.get('exact_node_recall'))}",
                    f"- Chunk Recall@5: {_fmt(k5.get('chunk_recall'))}",
                    f"- Hit@5: {_fmt(k5.get('chunk_hit'))}",
                    f"- MRR@5: {_fmt(k5.get('chunk_mrr'))}",
                    f"- First gold rank@5: {k5.get('first_gold_rank')}",
                    f"- Content Anchor Recall@5: {_fmt(k5.get('content_anchor_recall'))}",
                    f"- Content Anchor Precision@5: {_fmt(k5.get('content_anchor_precision'))}",
                    f"- Partial Recall@5: {_fmt(k5.get('partial_recall'))}",
                    f"- Partial Precision@5: {_fmt(k5.get('partial_precision'))}",
                    f"- Partial Exact Node Recall@5: {_fmt(k5.get('partial_exact_node_recall'))}",
                    f"- Partial Content Anchor Recall@5: {_fmt(k5.get('partial_content_anchor_recall'))}",
                    f"- Partial Content Anchor Precision@5: {_fmt(k5.get('partial_content_anchor_precision'))}",
                    f"- Filter precision@5: {_fmt(k5.get('filter_precision'))}",
                    f"- Coverage gap expected/detected: {query.get('coverage_gap_expected')} / {query.get('coverage_gap_detected')}",
                    f"- Coverage gap behavior: `{query.get('coverage_gap_behavior')}`",
                    f"- Unsupported filters: `{query.get('unsupported_filters')}`",
                    "",
                ]
            )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def main() -> None:
    args = parse_args()
    eval_queries = load_jsonl(args.eval_set_jsonl)
    write_standard_audit_report(
        eval_queries,
        args.eval_set_jsonl,
        args.standard_audit_report,
    )
    report = run_eval(args)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "eval_set_jsonl": report["eval_set_jsonl"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "modes": [
                    {
                        "mode": mode["mode"],
                        "summary": mode["summary"],
                    }
                    for mode in report["modes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
