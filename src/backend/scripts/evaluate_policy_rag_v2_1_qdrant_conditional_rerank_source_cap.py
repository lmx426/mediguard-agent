"""Evaluate conditional reranker plus source-cap on Qdrant hybrid retrieval.

This is an experiment-only script. It keeps the same metadata-filtered
Qdrant+bge-m3+BM25+RRF first-stage retrieval and compares post-retrieval
orchestration variants with node-level metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bge_reranker import BgePolicyReranker
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyRetriever
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
)
from src.backend.infrastructure.policy_rag.qdrant_retriever import QdrantPolicyRetriever
from src.backend.infrastructure.policy_rag.raw_hybrid_retriever import (
    RawHybridPolicyRetriever,
)
from src.backend.scripts.evaluate_policy_rag_v2_1_retrieval import (
    DEFAULT_FETCH_K,
    TOP_K_VALUES,
    context_from_candidate,
    evaluate_query,
    fmt,
    normalize_filters,
    parse_top_k_values,
    read_jsonl,
    summarize,
)


DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_1_entity_fixed_from300_audited.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_qdrant_conditional_rerank_source_cap_node_contrast_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_qdrant_conditional_rerank_source_cap_node_contrast_report.md"
)
DEFAULT_CONDITIONAL_TOP_K = 5
DEFAULT_RERANK_MARGIN_THRESHOLD = 0.08
DEFAULT_SOURCE_CAP = 2


@dataclass(frozen=True, slots=True)
class ModeConfig:
    name: str
    source_cap_enabled: bool
    reranker_enabled: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qdrant hybrid baseline with source-cap and conditional "
            "bge-reranker post-processing."
        )
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--top-k-values", default="3,5,10")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--source-cap", type=int, default=DEFAULT_SOURCE_CAP)
    parser.add_argument(
        "--conditional-top-k",
        type=int,
        default=DEFAULT_CONDITIONAL_TOP_K,
        help="Top-K window used to detect same-source crowding and low margin.",
    )
    parser.add_argument(
        "--rerank-margin-threshold",
        type=float,
        default=DEFAULT_RERANK_MARGIN_THRESHOLD,
        help="Trigger reranker when relative score margin is below this value.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path(os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"])
        if os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_PATH")
        else None,
        help="Use embedded local Qdrant storage.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_URL"),
        help="Use a running Qdrant HTTP service. Ignored when --qdrant-path is set.",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qdrant_path is not None:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"] = str(args.qdrant_path)
    if args.collection:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION"] = str(args.collection)
    if args.qdrant_url and args.qdrant_path is None:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_URL"] = str(args.qdrant_url)

    top_k_values = parse_top_k_values(args.top_k_values)
    report = run_eval(args, top_k_values=top_k_values)
    write_contrast_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "top_k_values": top_k_values,
                "qdrant_mode": report["qdrant_mode"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "modes": [
                    {
                        "mode": mode["mode"],
                        "summary": mode["summary"]["by_k"],
                        "rerank_trigger_rate": mode["postprocess"].get(
                            "rerank_trigger_rate"
                        ),
                    }
                    for mode in report["modes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_eval(args: argparse.Namespace, *, top_k_values: list[int]) -> dict[str, Any]:
    eval_rows = read_jsonl(args.eval_set_jsonl)
    if args.offset:
        eval_rows = eval_rows[args.offset :]
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    retriever = build_retriever(args)
    reranker = BgePolicyReranker(
        cache_folder=args.cache_folder,
        device=args.reranker_device or args.device,
        allow_unavailable=False,
    )
    max_top_k = max(top_k_values)
    modes = [
        ModeConfig(
            name="qdrant_hybrid_baseline",
            source_cap_enabled=False,
            reranker_enabled=False,
        ),
        ModeConfig(
            name=f"qdrant_hybrid_source_cap_{args.source_cap}",
            source_cap_enabled=True,
            reranker_enabled=False,
        ),
        ModeConfig(
            name=f"qdrant_hybrid_conditional_reranker_source_cap_{args.source_cap}",
            source_cap_enabled=True,
            reranker_enabled=True,
        ),
    ]
    query_reports_by_mode: dict[str, list[dict[str, Any]]] = {
        mode.name: [] for mode in modes
    }
    postprocess_by_mode: dict[str, dict[str, Any]] = {
        mode.name: {
            "rerank_requested_count": 0,
            "rerank_triggered_count": 0,
            "rerank_trigger_reasons": Counter(),
            "source_cap_changed_count": 0,
        }
        for mode in modes
    }

    total = len(eval_rows)
    for index, row in enumerate(eval_rows, start=1):
        filters = normalize_filters(row.get("filters"))
        started = time.perf_counter()
        base_candidates = retriever.retrieve(
            question=str(row["question"]),
            filters=filters,
            fetch_k=max(args.fetch_k, max_top_k),
        )
        retrieval_elapsed = time.perf_counter() - started
        baseline_ids = [candidate.node_id for candidate in base_candidates[:max_top_k]]

        for mode in modes:
            mode_started = time.perf_counter()
            candidates = list(base_candidates)
            rerank_applied = False
            rerank_triggered = False
            rerank_trigger_reason = "reranker_not_requested"
            if mode.reranker_enabled:
                postprocess_by_mode[mode.name]["rerank_requested_count"] += 1
                rerank_triggered, rerank_trigger_reason = should_rerank(
                    candidates=candidates,
                    top_k=args.conditional_top_k,
                    margin_threshold=args.rerank_margin_threshold,
                )
                postprocess_by_mode[mode.name]["rerank_trigger_reasons"][
                    rerank_trigger_reason
                ] += 1
                if rerank_triggered:
                    postprocess_by_mode[mode.name]["rerank_triggered_count"] += 1
                    candidates = reranker.rerank(
                        question=str(row["question"]),
                        candidates=candidates,
                    )
                    rerank_applied = True

            if mode.source_cap_enabled:
                candidates = select_with_source_cap(
                    candidates,
                    limit=max_top_k,
                    source_cap=args.source_cap,
                    rerank_applied=rerank_applied,
                )
                selected_ids = [candidate.node_id for candidate in candidates[:max_top_k]]
                if selected_ids != baseline_ids:
                    postprocess_by_mode[mode.name]["source_cap_changed_count"] += 1

            elapsed = retrieval_elapsed + (time.perf_counter() - mode_started)
            contexts = [
                context_from_candidate_with_postprocess(
                    candidate,
                    rank=rank,
                    rerank_applied=rerank_applied,
                    source_cap_enabled=mode.source_cap_enabled,
                )
                for rank, candidate in enumerate(candidates[:max_top_k], start=1)
            ]
            query_report = evaluate_query(
                row,
                retrieved_contexts=contexts,
                judge_result=None,
                elapsed_seconds=elapsed,
                top_k_values=top_k_values,
            )
            query_report["postprocess"] = {
                "source_cap_enabled": mode.source_cap_enabled,
                "source_cap": args.source_cap if mode.source_cap_enabled else None,
                "rerank_requested": mode.reranker_enabled,
                "rerank_triggered": rerank_triggered,
                "rerank_trigger_reason": rerank_trigger_reason,
                "rerank_applied": rerank_applied,
            }
            query_reports_by_mode[mode.name].append(query_report)

        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"[policy-rag-v2.1-cond-rerank-source-cap] progress={index}/{total}",
                flush=True,
            )

    mode_reports = []
    for mode in modes:
        query_reports = query_reports_by_mode[mode.name]
        postprocess = normalize_postprocess_summary(
            postprocess_by_mode[mode.name],
            query_count=len(query_reports),
        )
        mode_reports.append(
            {
                "mode": mode.name,
                "query_count": len(query_reports),
                "retrieval_strategy": mode_retrieval_strategy(
                    mode=mode,
                    source_cap=args.source_cap,
                ),
                "queries": query_reports,
                "summary": summarize(query_reports, top_k_values=top_k_values),
                "postprocess": postprocess,
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "query_count": len(eval_rows),
        "fetch_k": args.fetch_k,
        "top_k_values": top_k_values,
        "conditional_top_k": args.conditional_top_k,
        "rerank_margin_threshold": args.rerank_margin_threshold,
        "source_cap": args.source_cap,
        "qdrant_mode": "local_path" if args.qdrant_path else "url",
        "qdrant_path": str(args.qdrant_path) if args.qdrant_path else None,
        "qdrant_url": None if args.qdrant_path else args.qdrant_url,
        "collection": getattr(retriever._dense_retriever, "collection_name", None),
        "reranker_model": reranker.model_name,
        "modes": mode_reports,
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_llm_judge_for_context_metrics": False,
            "generates_final_rag_answer": False,
            "calculates_answer_correctness": False,
            "calculates_faithfulness": False,
            "calculates_citation_accuracy": False,
            "makes_audit_decisions": False,
            "changes_production_retriever": False,
        },
    }


def build_retriever(args: argparse.Namespace) -> RawHybridPolicyRetriever:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense = QdrantPolicyRetriever(
        url=args.qdrant_url,
        collection_name=args.collection,
        embedder=embedder,
    )
    lexical = Bm25PolicyRetriever()
    return RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )


def should_rerank(
    *,
    candidates: list[RetrievedPolicyCandidate],
    top_k: int,
    margin_threshold: float,
) -> tuple[bool, str]:
    if len(candidates) <= top_k:
        return False, "candidate_count_not_above_top_k"
    ranked = sorted(candidates, key=lambda item: item.faiss_score, reverse=True)
    top_sources = {_source_key(item) for item in ranked[:top_k]}
    all_sources = {_source_key(item) for item in ranked}
    if len(all_sources) <= 1:
        return False, "single_source_candidate_pool"
    if len(top_sources) == 1:
        return True, "single_source_crowding"
    kth_index = min(max(top_k - 1, 0), len(ranked) - 1)
    top_score = float(ranked[0].faiss_score)
    kth_score = float(ranked[kth_index].faiss_score)
    relative_margin = (top_score - kth_score) / max(abs(top_score), 1e-9)
    if relative_margin < margin_threshold:
        return True, "low_relative_score_margin"
    return False, "multi_source_confident_results"


def select_with_source_cap(
    candidates: list[RetrievedPolicyCandidate],
    *,
    limit: int,
    source_cap: int,
    rerank_applied: bool,
) -> list[RetrievedPolicyCandidate]:
    ranked = sorted(
        candidates,
        key=lambda item: item.score(rerank_applied=rerank_applied),
        reverse=True,
    )
    selected: list[RetrievedPolicyCandidate] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    for candidate in ranked:
        source = _source_key(candidate)
        if source_counts.get(source, 0) >= source_cap:
            continue
        append_source_capped_candidate(
            candidate,
            selected=selected,
            selected_ids=selected_ids,
            source_counts=source_counts,
        )
        if len(selected) >= limit:
            return selected

    # Soft cap: preserve TopK size when the candidate pool has too few sources.
    for candidate in ranked:
        if candidate.node_id in selected_ids:
            continue
        append_source_capped_candidate(
            replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "source_cap_relaxed_fill": True,
                },
            ),
            selected=selected,
            selected_ids=selected_ids,
            source_counts=source_counts,
        )
        if len(selected) >= limit:
            break
    return selected


def append_source_capped_candidate(
    candidate: RetrievedPolicyCandidate,
    *,
    selected: list[RetrievedPolicyCandidate],
    selected_ids: set[str],
    source_counts: dict[str, int],
) -> None:
    selected.append(candidate)
    selected_ids.add(candidate.node_id)
    source = _source_key(candidate)
    source_counts[source] = source_counts.get(source, 0) + 1


def context_from_candidate_with_postprocess(
    candidate: RetrievedPolicyCandidate,
    *,
    rank: int,
    rerank_applied: bool,
    source_cap_enabled: bool,
) -> dict[str, Any]:
    context = context_from_candidate(candidate, rank=rank)
    if rerank_applied and candidate.rerank_score is not None:
        context["score"] = candidate.rerank_score
        context["score_type"] = "rerank_score"
    else:
        context["score_type"] = candidate.metadata.get("score_type") or "hybrid_score"
    context["faiss_score"] = candidate.faiss_score
    context["rerank_score"] = candidate.rerank_score
    context["source_cap_enabled"] = source_cap_enabled
    context["source_cap_relaxed_fill"] = bool(
        candidate.metadata.get("source_cap_relaxed_fill")
    )
    return context


def normalize_postprocess_summary(
    payload: dict[str, Any],
    *,
    query_count: int,
) -> dict[str, Any]:
    triggered = int(payload.get("rerank_triggered_count") or 0)
    requested = int(payload.get("rerank_requested_count") or 0)
    return {
        "rerank_requested_count": requested,
        "rerank_triggered_count": triggered,
        "rerank_trigger_rate": triggered / requested if requested else 0.0,
        "rerank_trigger_reasons": dict(payload.get("rerank_trigger_reasons") or {}),
        "source_cap_changed_count": int(payload.get("source_cap_changed_count") or 0),
        "source_cap_changed_rate": (
            int(payload.get("source_cap_changed_count") or 0) / query_count
            if query_count
            else 0.0
        ),
    }


def mode_retrieval_strategy(*, mode: ModeConfig, source_cap: int) -> str:
    base = "metadata_filter_qdrant_bge_m3_bm25_rrf"
    if mode.source_cap_enabled:
        base = f"{base}_soft_source_cap_{source_cap}"
    if mode.reranker_enabled:
        base = f"{base}_conditional_bge_reranker_v2_m3"
    return base


def _source_key(candidate: RetrievedPolicyCandidate) -> str:
    metadata = candidate.metadata
    return str(
        metadata.get("source_id")
        or metadata.get("doc_id")
        or candidate.node_id.split("::", 1)[0]
    )


def write_contrast_reports(
    report: dict[str, Any],
    report_json: Path,
    report_md: Path,
) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    baseline = report["modes"][0]
    lines = [
        "# Policy RAG v2.1 Qdrant Conditional Reranker + Source Cap Contrast",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report evaluates retrieval only. It compares post-retrieval orchestration on the same Qdrant hybrid candidate pool.",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Top K values: {report.get('top_k_values')}",
        f"- Qdrant mode/path: `{report.get('qdrant_mode')}` / `{report.get('qdrant_path')}`",
        f"- Collection: `{report.get('collection')}`",
        f"- Source cap: {report.get('source_cap')} (soft fill keeps TopK size when sources are insufficient)",
        f"- Conditional reranker topK: {report.get('conditional_top_k')}",
        f"- Rerank margin threshold: {report.get('rerank_margin_threshold')}",
        f"- Reranker model: `{report.get('reranker_model')}`",
        "",
        "## Overall Metrics",
        "",
        "| Mode | K | Recall@K | Delta Recall | Hit@K | Delta Hit | MRR@K | Delta MRR | Avg Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_by_k = (baseline.get("summary") or {}).get("by_k") or {}
    for mode in report.get("modes") or []:
        summary = mode.get("summary") or {}
        avg_seconds = summary.get("avg_seconds")
        for k, item in (summary.get("by_k") or {}).items():
            base_item = baseline_by_k.get(k) or {}
            lines.append(
                f"| `{mode.get('mode')}` | {k} | {fmt(item.get('recall'))} | "
                f"{fmt_delta(item.get('recall'), base_item.get('recall'))} | "
                f"{fmt(item.get('hit'))} | {fmt_delta(item.get('hit'), base_item.get('hit'))} | "
                f"{fmt(item.get('mrr'))} | {fmt_delta(item.get('mrr'), base_item.get('mrr'))} | "
                f"{fmt(avg_seconds)} |"
            )

    lines.extend(
        [
            "",
            "## Postprocess Stats",
            "",
            "| Mode | Rerank Triggered / Requested | Trigger Rate | Trigger Reasons | Source Cap Changed | Source Cap Changed Rate |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for mode in report.get("modes") or []:
        post = mode.get("postprocess") or {}
        lines.append(
            f"| `{mode.get('mode')}` | "
            f"{post.get('rerank_triggered_count')} / {post.get('rerank_requested_count')} | "
            f"{fmt(post.get('rerank_trigger_rate'))} | "
            f"`{post.get('rerank_trigger_reasons')}` | "
            f"{post.get('source_cap_changed_count')} | "
            f"{fmt(post.get('source_cap_changed_rate'))} |"
        )

    lines.extend(["", "## Bundle Type K5 Metrics", ""])
    lines.extend(
        [
            "| Mode | Bundle Type | Count | Recall@5 | Hit@5 | MRR@5 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for mode in report.get("modes") or []:
        by_bundle = (mode.get("summary") or {}).get("by_bundle_type") or {}
        for bundle_type, group in by_bundle.items():
            k5 = (group.get("by_k") or {}).get("5") or {}
            lines.append(
                f"| `{mode.get('mode')}` | `{bundle_type}` | {group.get('query_count')} | "
                f"{fmt(k5.get('recall'))} | {fmt(k5.get('hit'))} | {fmt(k5.get('mrr'))} |"
            )

    lines.extend(["", "## Sample Query Details", ""])
    for mode in report.get("modes") or []:
        lines.extend([f"### {mode.get('mode')}", ""])
        for item in (mode.get("queries") or [])[:10]:
            k5 = item["per_k"].get("5") or {}
            post = item.get("postprocess") or {}
            lines.extend(
                [
                    f"#### {item.get('query_id')}",
                    "",
                    f"- Question: {item.get('question')}",
                    f"- Bundle/question type: `{item.get('bundle_type')}` / `{item.get('question_type')}`",
                    f"- Retrieved node ids: `{(item.get('retrieved_node_ids') or [])[:5]}`",
                    f"- Recall@5 / Hit@5 / MRR@5: {fmt(k5.get('recall'))} / {fmt(k5.get('hit'))} / {fmt(k5.get('mrr'))}",
                    f"- Rerank triggered/reason: {post.get('rerank_triggered')} / `{post.get('rerank_trigger_reason')}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def fmt_delta(value: Any, baseline: Any) -> str:
    if value is None or baseline is None:
        return "n/a"
    delta = float(value) - float(baseline)
    return f"{delta:+.3f}"


if __name__ == "__main__":
    main()
