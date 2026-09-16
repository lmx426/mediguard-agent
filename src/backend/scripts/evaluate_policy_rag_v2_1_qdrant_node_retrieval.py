"""Evaluate Qdrant-backed Policy RAG retrieval with node-level metrics only.

This is a comparison script for the vector database backend. It keeps the same
metadata filters, bge-m3 query embedding, BM25, RRF fusion, TopK and fetchK
shape as the FAISS hybrid evaluator, but replaces FAISS dense recall with
Qdrant dense recall.
"""

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

from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
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
    parse_top_k_values,
    read_jsonl,
    summarize,
    write_reports,
)


DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_1_entity_fixed_from300_audited.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_qdrant_hybrid_node_eval_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_qdrant_hybrid_node_eval_report.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v2.1 Policy RAG Qdrant hybrid retrieval with node metrics only."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--top-k-values", default="3,5,10")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path(os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"])
        if os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_PATH")
        else None,
        help="Use embedded local Qdrant storage at this path.",
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
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "top_k_values": top_k_values,
                "qdrant_mode": report["qdrant_mode"],
                "qdrant_path": report["qdrant_path"],
                "qdrant_url": report["qdrant_url"],
                "collection": report["collection"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "summary": report["summary"],
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
    max_top_k = max(top_k_values)
    query_reports: list[dict[str, Any]] = []
    total = len(eval_rows)
    for index, row in enumerate(eval_rows, start=1):
        started = time.perf_counter()
        filters = normalize_filters_from_eval(row)
        candidates = retriever.retrieve(
            question=str(row["question"]),
            filters=filters,
            fetch_k=max(args.fetch_k, max_top_k),
        )
        retrieved_contexts = [
            context_from_candidate(candidate, rank=rank)
            for rank, candidate in enumerate(candidates[:max_top_k], start=1)
        ]
        query_reports.append(
            evaluate_query(
                row,
                retrieved_contexts=retrieved_contexts,
                judge_result=None,
                elapsed_seconds=time.perf_counter() - started,
                top_k_values=top_k_values,
            )
        )
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"[policy-rag-v2.1-qdrant-node-eval] progress={index}/{total}",
                flush=True,
            )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "query_count": len(query_reports),
        "fetch_k": args.fetch_k,
        "top_k_values": top_k_values,
        "retrieval_strategy": "metadata_filter_qdrant_bge_m3_bm25_rrf",
        "qdrant_mode": "local_path" if args.qdrant_path else "url",
        "qdrant_path": str(args.qdrant_path) if args.qdrant_path else None,
        "qdrant_url": None if args.qdrant_path else args.qdrant_url,
        "collection": getattr(retriever._dense_retriever, "collection_name", None),
        "judge_enabled": False,
        "judge_model": None,
        "judge_cache_path": None,
        "queries": query_reports,
        "summary": summarize(query_reports, top_k_values=top_k_values),
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_llm_judge_for_context_metrics": False,
            "generates_final_rag_answer": False,
            "calculates_answer_correctness": False,
            "calculates_faithfulness": False,
            "calculates_citation_accuracy": False,
            "makes_audit_decisions": False,
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


def normalize_filters_from_eval(row: dict[str, Any]) -> dict[str, Any]:
    from src.backend.scripts.evaluate_policy_rag_v2_1_retrieval import normalize_filters

    return normalize_filters(row.get("filters"))


if __name__ == "__main__":
    main()
