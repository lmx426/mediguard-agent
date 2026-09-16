"""Evaluate raw Dense retrieval against raw BM25-RRF hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.schemas import PolicyRagFilters
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bge_reranker import BgePolicyReranker
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyRetriever
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
)
from src.backend.infrastructure.policy_rag.raw_hybrid_retriever import (
    RawHybridPolicyRetriever,
)
from src.backend.scripts.evaluate_policy_rag_retrieval import (
    TOP_K_VALUES,
    evaluate_query,
    load_jsonl,
    summarize_mode,
    write_standard_audit_report,
)


DEFAULT_FETCH_K = 20
DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_3_combined_224.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_retrieval_quality_report_v1_3_raw_dense_vs_hybrid.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_retrieval_quality_report_v1_3_raw_dense_vs_hybrid.md"
)
DEFAULT_AUDIT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_eval_standard_audit_v1_3_raw_dense_vs_hybrid.md"
)
DEFAULT_RERANK_MARGIN_THRESHOLD = 0.08
DEFAULT_CONDITIONAL_RERANK_TOP_K = 5


class _RawRetrievalModeConfig(dict[str, object]):
    """Typed dict-like config for raw retrieval evaluation modes."""


def _mode_config(
    *,
    retriever: object,
    reranker: BgePolicyReranker | None,
    rerank_policy: str,
) -> _RawRetrievalModeConfig:
    return _RawRetrievalModeConfig(
        retriever=retriever,
        reranker=reranker,
        rerank_policy=rerank_policy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw Dense against raw BM25-RRF hybrid retrieval."
    )
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--include-reranker", action="store_true")
    parser.add_argument(
        "--only-hybrid",
        action="store_true",
        help="Evaluate only metadata-filtered FAISS+bge-m3+BM25 RRF retrieval.",
    )
    parser.add_argument("--only-hybrid-reranker", action="store_true")
    parser.add_argument(
        "--conditional-reranker",
        action="store_true",
        help=(
            "Apply reranker only when the hybrid candidate set looks uncertain "
            "or crowded by one source."
        ),
    )
    parser.add_argument(
        "--rerank-margin-threshold",
        type=float,
        default=DEFAULT_RERANK_MARGIN_THRESHOLD,
        help=(
            "Trigger conditional reranking when the relative top-K RRF score "
            "margin is below this value."
        ),
    )
    parser.add_argument(
        "--disable-metadata-filters",
        action="store_true",
        help="Do not apply request metadata filters during retrieval.",
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--standard-audit-report",
        type=Path,
        default=DEFAULT_AUDIT_REPORT_MD,
    )
    return parser.parse_args()


def build_retrievers(
    args: argparse.Namespace,
) -> dict[str, _RawRetrievalModeConfig]:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense = FaissPolicyRetriever(embedder=embedder)
    bm25 = Bm25PolicyRetriever()
    hybrid = RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=bm25,
    )
    retrievers: dict[str, _RawRetrievalModeConfig] = {
        "raw_dense": _mode_config(
            retriever=dense,
            reranker=None,
            rerank_policy="none",
        ),
        "raw_hybrid_bm25_rrf": _mode_config(
            retriever=hybrid,
            reranker=None,
            rerank_policy="none",
        ),
    }
    if args.only_hybrid and not args.include_reranker:
        return {
            "raw_hybrid_bm25_rrf": _mode_config(
                retriever=hybrid,
                reranker=None,
                rerank_policy="none",
            )
        }
    if args.include_reranker:
        reranker = BgePolicyReranker(
            cache_folder=args.cache_folder,
            device=args.reranker_device or args.device,
            allow_unavailable=False,
        )
        rerank_policy = "conditional" if args.conditional_reranker else "always"
        rerank_mode_name = (
            "raw_hybrid_bm25_rrf_with_conditional_reranker"
            if args.conditional_reranker
            else "raw_hybrid_bm25_rrf_with_reranker"
        )
        if args.only_hybrid_reranker:
            return {
                rerank_mode_name: _mode_config(
                    retriever=hybrid,
                    reranker=reranker,
                    rerank_policy=rerank_policy,
                )
            }
        retrievers[rerank_mode_name] = _mode_config(
            retriever=hybrid,
            reranker=reranker,
            rerank_policy=rerank_policy,
        )
    return retrievers


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    eval_queries = load_jsonl(args.eval_set_jsonl)
    retrievers = build_retrievers(args)
    mode_reports: list[dict[str, Any]] = []

    for mode_name, mode_config in retrievers.items():
        retriever = mode_config["retriever"]
        reranker = mode_config["reranker"]
        rerank_policy = str(mode_config["rerank_policy"])
        query_reports: list[dict[str, Any]] = []
        total_queries = len(eval_queries)
        print(
            f"[policy-rag-raw-eval] mode={mode_name} queries={total_queries}",
            flush=True,
        )
        for index, query in enumerate(eval_queries, start=1):
            requested_filters = _normalized_filters(query.get("filters"))
            applied_filters = (
                {} if args.disable_metadata_filters else requested_filters
            )
            started = time.perf_counter()
            candidates = retriever.retrieve(
                question=str(query["question"]),
                filters=applied_filters,
                fetch_k=args.fetch_k,
            )
            rerank_requested = reranker is not None
            rerank_condition_met = False
            rerank_triggered = False
            rerank_trigger_reason = "reranker_not_configured"
            if reranker is not None:
                if rerank_policy == "conditional":
                    rerank_condition_met, rerank_trigger_reason = _should_rerank_raw_hybrid(
                        candidates=candidates,
                        top_k=DEFAULT_CONDITIONAL_RERANK_TOP_K,
                        margin_threshold=args.rerank_margin_threshold,
                    )
                    rerank_triggered = rerank_condition_met
                else:
                    rerank_condition_met = True
                    rerank_triggered = True
                    rerank_trigger_reason = "always_on"
            if rerank_triggered and reranker is not None:
                candidates = reranker.rerank(
                    question=str(query["question"]),
                    candidates=candidates,
                )
            elapsed = time.perf_counter() - started
            response = _raw_response(
                question=str(query["question"]),
                filters_requested=requested_filters,
                filters_used=applied_filters,
                metadata_filter_applied=not args.disable_metadata_filters,
                candidates=candidates,
                retrieval_mode=_retrieval_mode(
                    retriever=retriever,
                    mode_name=mode_name,
                    rerank_applied=rerank_triggered,
                ),
                rerank_requested=rerank_requested,
                rerank_condition_met=rerank_condition_met,
                rerank_triggered=rerank_triggered,
                rerank_trigger_reason=rerank_trigger_reason,
                rerank_policy=rerank_policy,
            )
            query_reports.append(
                evaluate_query(query, response, elapsed_seconds=elapsed)
            )
            if index == 1 or index % 10 == 0 or index == total_queries:
                print(
                    f"[policy-rag-raw-eval] mode={mode_name} "
                    f"progress={index}/{total_queries}",
                    flush=True,
                )
        mode_reports.append(
            {
                "mode": mode_name,
                "rerank": rerank_requested,
                "rerank_policy": rerank_policy,
                "query_count": len(query_reports),
                "queries": query_reports,
                "summary": summarize_mode(query_reports),
            }
        )

    return {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "fetch_k": args.fetch_k,
        "top_k_values": TOP_K_VALUES,
        "retrieval_scope": (
            "raw_retrieval_only_no_metadata_filters"
            if args.disable_metadata_filters
            else "raw_retrieval_only"
        ),
        "metadata_filter_applied": not args.disable_metadata_filters,
        "conditional_reranker": bool(args.conditional_reranker),
        "conditional_reranker_top_k": DEFAULT_CONDITIONAL_RERANK_TOP_K,
        "rerank_margin_threshold": args.rerank_margin_threshold,
        "post_retrieval_optimizations": {
            "group_aware": False,
            "source_deduplication": False,
            "source_cap": False,
            "reranker": bool(args.include_reranker),
            "conditional_reranker": bool(args.conditional_reranker),
        },
        "bm25": {
            "algorithm": "Okapi BM25",
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "Chinese character n-grams 2/3/4 plus ASCII and number tokens",
        },
        "fusion": {
            "algorithm": "Reciprocal Rank Fusion",
            "rrf_k": 60,
            "dense_weight": 1.0,
            "lexical_weight": 1.0,
        },
        "reranker": {
            "enabled": bool(args.include_reranker),
            "conditional": bool(args.conditional_reranker),
            "conditional_top_k": DEFAULT_CONDITIONAL_RERANK_TOP_K,
            "margin_threshold": args.rerank_margin_threshold,
            "model": (
                "BAAI/bge-reranker-v2-m3"
                if args.include_reranker
                else None
            ),
        },
        "modes": mode_reports,
        "passed": all(
            mode["summary"]["supported_queries_with_results"]
            for mode in mode_reports
        ),
        "eval_schema_version": "policy_rag_eval_v1_3_raw_retrieval",
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_llm_judge": False,
            "calculates_token_usage": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def _normalized_filters(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    return PolicyRagFilters.from_mapping(payload).normalized().to_dict()


def _raw_response(
    *,
    question: str,
    filters_requested: dict[str, object],
    filters_used: dict[str, object],
    metadata_filter_applied: bool,
    candidates: list[Any],
    retrieval_mode: str,
    rerank_requested: bool,
    rerank_condition_met: bool,
    rerank_triggered: bool,
    rerank_trigger_reason: str,
    rerank_policy: str,
) -> dict[str, Any]:
    evidence = [
        _evidence_payload(
            candidate,
            rank=index,
            rerank_applied=rerank_triggered,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    return {
        "status": "ok",
        "retrieval_mode": retrieval_mode,
        "question": question,
        "filters_requested": filters_requested,
        "filters_used": filters_used,
        "metadata_filter_applied": metadata_filter_applied,
        "rerank": {
            "requested": rerank_requested,
            "conditional": rerank_policy == "conditional",
            "condition_met": rerank_condition_met,
            "triggered": rerank_triggered,
            "trigger_reason": rerank_trigger_reason,
            "applied": rerank_triggered,
            "policy": rerank_policy,
        },
        "result_count": len(evidence),
        "evidence": evidence,
        "warnings": (
            ["raw retrieval returned no results"] if not evidence else []
        ),
    }


def _evidence_payload(
    candidate: Any,
    *,
    rank: int,
    rerank_applied: bool,
) -> dict[str, Any]:
    metadata = candidate.metadata
    score = (
        candidate.rerank_score
        if rerank_applied and candidate.rerank_score is not None
        else candidate.faiss_score
    )
    return {
        "rank": rank,
        "score": score,
        "score_type": (
            "rerank_score"
            if rerank_applied
            else metadata.get("score_type") or "faiss_score"
        ),
        "faiss_score": candidate.faiss_score,
        "rerank_score": candidate.rerank_score,
        "dense_score": metadata.get("dense_score"),
        "lexical_score": metadata.get("lexical_score"),
        "hybrid_score": metadata.get("hybrid_score"),
        "retrieval_strategy": metadata.get("retrieval_strategy"),
        "node_id": candidate.node_id,
        "text": candidate.text,
        "retrieval_groups": [],
        "title": metadata.get("title"),
        "source_url": metadata.get("source_url"),
        "source_id": metadata.get("source_id"),
        "jurisdiction": metadata.get("jurisdiction"),
        "policy_domain": metadata.get("policy_domain"),
        "content_type": metadata.get("content_type"),
        "doc_type": metadata.get("doc_type"),
        "doc_id": metadata.get("doc_id"),
        "section_heading": metadata.get("section_heading"),
        "chunk_index": metadata.get("chunk_index"),
        "evidence_role": metadata.get("evidence_role"),
        "can_cite_as_policy_basis": metadata.get(
            "can_cite_as_policy_basis"
        ),
    }


def _retrieval_mode(
    *,
    retriever: object,
    mode_name: str,
    rerank_applied: bool,
) -> str:
    base_mode = str(getattr(retriever, "retrieval_mode_name", mode_name))
    if rerank_applied:
        return f"{base_mode}_with_bge_reranker_v2_m3"
    return base_mode


def _should_rerank_raw_hybrid(
    *,
    candidates: list[Any],
    top_k: int,
    margin_threshold: float,
) -> tuple[bool, str]:
    if len(candidates) <= top_k:
        return False, "candidate_count_not_above_top_k"
    ranked = sorted(candidates, key=lambda item: item.faiss_score, reverse=True)
    top_sources = {_source_key(item) for item in ranked[:top_k]}
    all_sources = {_source_key(item) for item in ranked}
    if len(top_sources) == 1 and len(all_sources) > 1:
        return True, "single_source_crowding"
    if len(all_sources) <= 1:
        return False, "single_source_candidate_pool"
    kth_index = min(max(top_k - 1, 0), len(ranked) - 1)
    top_score = float(ranked[0].faiss_score)
    kth_score = float(ranked[kth_index].faiss_score)
    relative_margin = (top_score - kth_score) / max(abs(top_score), 1e-9)
    if relative_margin < margin_threshold:
        return True, "low_relative_score_margin"
    return False, "multi_source_confident_results"


def _source_key(candidate: Any) -> str:
    metadata = candidate.metadata
    return str(
        metadata.get("source_id")
        or metadata.get("doc_id")
        or candidate.node_id.split("::", 1)[0]
    )


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_markdown_report(report), encoding="utf-8", newline="\n")


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy RAG Raw Retrieval Quality Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report compares raw Dense retrieval with raw BM25-RRF hybrid retrieval.",
        "It excludes group-aware planning, source de-duplication, and source caps.",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Top K values: {report.get('top_k_values')}",
        f"- Metadata filters applied: {report.get('metadata_filter_applied')}",
        f"- Retrieval scope: `{report.get('retrieval_scope')}`",
        f"- Conditional reranker: {report.get('conditional_reranker')}",
        f"- Rerank margin threshold: {report.get('rerank_margin_threshold')}",
        f"- BM25: `{report.get('bm25')}`",
        f"- Fusion: `{report.get('fusion')}`",
        f"- Reranker: `{report.get('reranker')}`",
        "",
    ]
    for mode in report.get("modes") or []:
        summary = mode.get("summary") or {}
        lines.extend(
            [
                f"## {mode.get('mode')}",
                "",
                f"- Retrieval mode: `{mode.get('mode')}`",
                f"- Rerank: `{mode.get('rerank')}`",
                f"- Query count: {summary.get('query_count')}",
                f"- Full evidence query count: {summary.get('full_query_count')}",
                f"- Partial evidence query count: {summary.get('partial_query_count')}",
                f"- Coverage gap query count: {summary.get('coverage_gap_query_count')}",
                f"- Coverage gap detection rate: {_fmt_pct(summary.get('coverage_gap_detection_rate'))}",
                f"- Coverage gap leakage rate: {_fmt_pct(summary.get('coverage_gap_leakage_rate'))}",
                f"- Coverage rate: {_fmt_pct(summary.get('coverage_rate'))}",
                f"- Full answerability rate: {_fmt_pct(summary.get('full_answerability_rate'))}",
                f"- Avg seconds: {summary.get('avg_seconds'):.3f}",
                f"- Rerank trigger rate: {_fmt_pct(summary.get('rerank_trigger_rate'))}",
                f"- Rerank triggered/requested: {summary.get('rerank_triggered_count')} / {summary.get('rerank_requested_count')}",
                f"- Rerank trigger reasons: `{summary.get('rerank_trigger_reasons')}`",
                "",
                "| K | Chunk Recall@K | Hit@K | MRR@K | Strict Recall | Strict Precision | Exact Node Recall | Content Anchor Recall | Content Anchor Precision | Partial Recall | Partial Precision | Citeable Rate | Source URL Rate | Filter Precision |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for k, item in (summary.get("per_k") or {}).items():
            lines.append(
                f"| {k} | {_fmt(item.get('chunk_recall_avg'))} | {_fmt(item.get('chunk_hit_avg'))} | {_fmt(item.get('chunk_mrr_avg'))} | {_fmt(item.get('recall_avg'))} | {_fmt(item.get('precision_avg'))} | {_fmt(item.get('exact_node_recall_avg'))} | {_fmt(item.get('content_anchor_recall_avg'))} | {_fmt(item.get('content_anchor_precision_avg'))} | {_fmt(item.get('partial_recall_avg'))} | {_fmt(item.get('partial_precision_avg'))} | {_fmt(item.get('citeable_rate_avg'))} | {_fmt(item.get('source_url_rate_avg'))} | {_fmt(item.get('filter_precision_avg'))} |"
            )
        lines.extend(["", "### Query Details", ""])
        for query in mode.get("queries") or []:
            k5 = query["per_k"]["5"]
            lines.extend(
                [
                    f"#### {query.get('query_id')}",
                    "",
                    f"- Case: {query.get('case_id')}",
                    f"- Evidence status: {query.get('evidence_status')}",
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
                    f"- Filter precision@5: {_fmt(k5.get('filter_precision'))}",
                    f"- Coverage gap expected/detected: {query.get('coverage_gap_expected')} / {query.get('coverage_gap_detected')}",
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
