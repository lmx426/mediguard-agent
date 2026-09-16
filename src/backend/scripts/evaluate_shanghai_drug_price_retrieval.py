"""Evaluate retrieval for Shanghai drug product price reference nodes.

This is a retrieval-only evaluation for the newly added reference data. It
does not run reranking, LLM generation, MCP tools, Case Agent integration, or
audit decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.scripts.query_policy_faiss_index import (  # noqa: E402
    load_runtime as load_dense_runtime,
    retrieve as dense_retrieve,
)
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder  # noqa: E402
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever  # noqa: E402
from src.backend.infrastructure.policy_rag.hybrid_retriever import HybridPolicyRetriever  # noqa: E402
from src.backend.infrastructure.policy_rag.node_store import PolicyNodeStore  # noqa: E402


DEFAULT_RAG_READY_ROOT = (
    PROJECT_ROOT / "src" / "backend" / "policy_corpus" / "rag_ready"
)
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus" / "reports"
DEFAULT_DRUG_NODES = (
    DEFAULT_RAG_READY_ROOT / "nodes" / "shanghai_drug_price_nodes_20260820_full.jsonl"
)
DEFAULT_COMBINED_NODES = (
    DEFAULT_RAG_READY_ROOT
    / "nodes"
    / "policy_nodes_with_shanghai_drug_price_20260820_full.jsonl"
)
DEFAULT_INDEX_DIR = (
    DEFAULT_RAG_READY_ROOT
    / "indexes"
    / "faiss_bge_m3_flat_ip_with_shanghai_drug_price_20260820_full"
)
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "policy.index"
DEFAULT_INDEX_NODE_IDS = DEFAULT_INDEX_DIR / "index_node_ids.jsonl"
DEFAULT_INDEX_MANIFEST = DEFAULT_INDEX_DIR / "index_manifest.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"
DEFAULT_EVAL_SET = (
    DEFAULT_RAG_READY_ROOT
    / "eval"
    / "shanghai_drug_price_retrieval_eval_20260820_full.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_REPORT_ROOT / "shanghai_drug_price_retrieval_eval_20260820_full.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_REPORT_ROOT / "shanghai_drug_price_retrieval_eval_20260820_full.md"
)

DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 30
DEFAULT_SAMPLE_SIZE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-nodes", type=Path, default=DEFAULT_DRUG_NODES)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--index-node-ids", type=Path, default=DEFAULT_INDEX_NODE_IDS)
    parser.add_argument("--index-manifest", type=Path, default=DEFAULT_INDEX_MANIFEST)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_COMBINED_NODES)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--strategy", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--force-eval-set", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.fetch_k < args.top_k:
        raise ValueError("--fetch-k must be >= --top-k")

    eval_set = load_or_build_eval_set(args)
    report = evaluate(eval_set, args)
    write_report(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": report["status"],
                "strategy": report["strategy"],
                "query_count": report["query_count"],
                "hit_at_k": report["metrics"]["hit_at_k"],
                "mrr_at_k": report["metrics"]["mrr_at_k"],
                "recall_at_k": report["metrics"]["recall_at_k"],
                "precision_at_k": report["metrics"]["precision_at_k"],
                "report_json": str(args.report_json),
                "eval_set": str(args.eval_set),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_or_build_eval_set(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.eval_set.exists() and not args.force_eval_set:
        return load_jsonl(args.eval_set)
    drug_nodes = load_jsonl(args.drug_nodes)
    eval_rows = build_eval_rows(drug_nodes, sample_size=args.sample_size)
    write_jsonl(args.eval_set, eval_rows)
    return eval_rows


def build_eval_rows(
    node_rows: list[dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_drugs: set[str] = set()
    for node in sorted(
        node_rows,
        key=lambda item: (
            str((item.get("metadata") or {}).get("drug_name") or ""),
            str((item.get("metadata") or {}).get("manufacturer") or ""),
            str((item.get("metadata") or {}).get("specification") or ""),
        ),
    ):
        metadata = dict(node.get("metadata") or {})
        drug_name = str(metadata.get("drug_name") or "").strip()
        manufacturer = str(metadata.get("manufacturer") or "").strip()
        specification = str(metadata.get("specification") or "").strip()
        drug_code = str(metadata.get("drug_code") or "").strip()
        if not drug_name or not manufacturer or not specification:
            continue
        if drug_name in seen_drugs and len(candidates) < sample_size // 2:
            continue
        seen_drugs.add(drug_name)
        template_index = len(candidates) % 3
        if template_index == 0:
            question = (
                f"上海{drug_name}（{manufacturer}，{specification}）"
                "上周医保药店周均价是多少？"
            )
        elif template_index == 1:
            question = (
                f"上海医保药店中，{manufacturer}生产的{drug_name}"
                f"{specification}价格区间是多少？"
            )
        else:
            question = (
                f"药品编码{drug_code}对应的上海上周药店价格参考是多少？"
                if drug_code
                else f"上海{manufacturer}{drug_name}{specification}价格参考是多少？"
            )
        candidates.append(
            {
                "query_id": f"sh_drug_price_{len(candidates) + 1:03d}",
                "question": question,
                "gold_node_ids": [node["id_"]],
                "gold_drug_name": drug_name,
                "gold_manufacturer": manufacturer,
                "gold_specification": specification,
                "gold_drug_code": drug_code,
                "filters": {
                    "jurisdiction": ["shanghai"],
                    "policy_domain": ["drug_product_price_reference"],
                    "content_type": ["table_row"],
                    "can_cite_as_policy_basis": False,
                },
                "question_type": "drug_product_price_exact_lookup",
            }
        )
        if len(candidates) >= sample_size:
            break
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Only built {len(candidates)} eval rows from {len(node_rows)} drug nodes"
        )
    return candidates


def evaluate(eval_set: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    dense_runtime = None
    hybrid_retriever = None
    if args.strategy == "hybrid":
        hybrid_retriever = build_hybrid_retriever(args)
    else:
        dense_runtime = load_dense_runtime(args)
    rows: list[dict[str, Any]] = []
    for item in eval_set:
        filters = dict(item.get("filters") or {})
        if hybrid_retriever is not None:
            results = hybrid_results(
                retriever=hybrid_retriever,
                question=str(item["question"]),
                filters=filters,
                top_k=args.top_k,
                fetch_k=args.fetch_k,
            )
        else:
            if dense_runtime is None:
                raise RuntimeError("Dense runtime was not initialized")
            index, node_ids, nodes, embed_model = dense_runtime
            retrieval = dense_retrieve(
                question=str(item["question"]),
                index=index,
                node_ids=node_ids,
                nodes=nodes,
                embed_model=embed_model,
                top_k=args.top_k,
                fetch_k=args.fetch_k,
                jurisdictions=set(filters.get("jurisdiction") or []),
                policy_domains=set(filters.get("policy_domain") or []),
                content_types=set(filters.get("content_type") or []),
                include_retrieval_hints=False,
            )
            results = list(retrieval.get("results") or [])
        gold_node_ids = {str(node_id) for node_id in item.get("gold_node_ids") or []}
        retrieved_node_ids = [str(result["node_id"]) for result in results]
        hit_ranks = [
            index + 1
            for index, node_id in enumerate(retrieved_node_ids)
            if node_id in gold_node_ids
        ]
        first_hit_rank = hit_ranks[0] if hit_ranks else None
        matched_count = len(set(retrieved_node_ids).intersection(gold_node_ids))
        rows.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "gold_node_ids": sorted(gold_node_ids),
                "retrieved_node_ids": retrieved_node_ids,
                "hit": bool(hit_ranks),
                "first_hit_rank": first_hit_rank,
                "recall_at_k": matched_count / len(gold_node_ids)
                if gold_node_ids
                else 0.0,
                "precision_at_k": matched_count / args.top_k,
                "mrr_at_k": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
                "top_result": results[0] if results else None,
            }
        )

    metrics = aggregate_metrics(rows, top_k=args.top_k)
    return {
        "status": "ok",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set": str(args.eval_set),
        "index_path": str(args.index_path),
        "nodes": str(args.nodes),
        "strategy": args.strategy,
        "query_count": len(rows),
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "metrics": metrics,
        "rows": rows,
        "boundary": {
            "retrieval_only": True,
            "runs_reranker": False,
            "runs_llm_generation": False,
            "integrates_mcp": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def aggregate_metrics(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    if not rows:
        return {}
    hit_values = [1.0 if row["hit"] else 0.0 for row in rows]
    recall_values = [float(row["recall_at_k"]) for row in rows]
    precision_values = [float(row["precision_at_k"]) for row in rows]
    mrr_values = [float(row["mrr_at_k"]) for row in rows]
    domain_ok = [
        bool(
            (row.get("top_result") or {}).get("policy_domain")
            == "drug_product_price_reference"
        )
        for row in rows
    ]
    source_url_ok = [
        str((row.get("top_result") or {}).get("source_url") or "").startswith("http")
        for row in rows
    ]
    cite_false_ok = [
        (row.get("top_result") or {}).get("can_cite_as_policy_basis") is False
        for row in rows
    ]
    return {
        "top_k": top_k,
        "hit_at_k": round(avg(hit_values), 4),
        "recall_at_k": round(avg(recall_values), 4),
        "precision_at_k": round(avg(precision_values), 4),
        "mrr_at_k": round(avg(mrr_values), 4),
        "top1_domain_precision": round(avg([1.0 if item else 0.0 for item in domain_ok]), 4),
        "top1_source_url_rate": round(avg([1.0 if item else 0.0 for item in source_url_ok]), 4),
        "top1_reference_flag_rate": round(avg([1.0 if item else 0.0 for item in cite_false_ok]), 4),
        "first_hit_rank_distribution": dict(
            Counter(str(row["first_hit_rank"] or "miss") for row in rows)
        ),
    }


def build_hybrid_retriever(args: argparse.Namespace) -> HybridPolicyRetriever:
    embedder = BgeQueryEmbedder(
        model_name=args.model_name,
        max_length=args.max_length,
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense_retriever = FaissPolicyRetriever(
        index_path=args.index_path,
        index_node_ids_path=args.index_node_ids,
        index_manifest_path=args.index_manifest,
        nodes_path=args.nodes,
        embedder=embedder,
    )
    node_store = PolicyNodeStore(args.nodes)
    return HybridPolicyRetriever(
        dense_retriever=dense_retriever,
        node_store=node_store,
    )


def hybrid_results(
    *,
    retriever: HybridPolicyRetriever,
    question: str,
    filters: dict[str, object],
    top_k: int,
    fetch_k: int,
) -> list[dict[str, Any]]:
    candidates = retriever.retrieve(
        question=question,
        filters=filters,
        fetch_k=fetch_k,
    )[:top_k]
    results: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        metadata = dict(candidate.metadata)
        results.append(
            {
                "rank": rank,
                "score": candidate.faiss_score,
                "node_id": candidate.node_id,
                "title": metadata.get("title"),
                "jurisdiction": metadata.get("jurisdiction"),
                "policy_domain": metadata.get("policy_domain"),
                "content_type": metadata.get("content_type"),
                "evidence_role": metadata.get("evidence_role"),
                "can_cite_as_policy_basis": metadata.get("can_cite_as_policy_basis"),
                "source_url": metadata.get("source_url"),
                "source_id": metadata.get("source_id"),
                "doc_id": metadata.get("doc_id"),
                "doc_type": metadata.get("doc_type"),
                "section_heading": metadata.get("section_heading"),
                "chunk_index": metadata.get("chunk_index"),
                "text": candidate.text,
            }
        )
    return results


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")


def write_report(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Shanghai Drug Price Retrieval Eval",
        "",
        f"Updated at: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Query count: {report['query_count']}",
        f"- Strategy: `{report['strategy']}`",
        f"- Top K: {report['top_k']}",
        f"- Fetch K: {report['fetch_k']}",
        f"- Hit@K: {metrics['hit_at_k']:.4f}",
        f"- Recall@K: {metrics['recall_at_k']:.4f}",
        f"- Precision@K: {metrics['precision_at_k']:.4f}",
        f"- MRR@K: {metrics['mrr_at_k']:.4f}",
        f"- Top1 domain precision: {metrics['top1_domain_precision']:.4f}",
        f"- Top1 source URL rate: {metrics['top1_source_url_rate']:.4f}",
        f"- Top1 reference flag rate: {metrics['top1_reference_flag_rate']:.4f}",
        f"- First hit ranks: `{json.dumps(metrics['first_hit_rank_distribution'], ensure_ascii=False)}`",
        "",
        "## Misses",
        "",
    ]
    misses = [row for row in report["rows"] if not row["hit"]]
    if not misses:
        lines.append("- None")
    else:
        for row in misses:
            lines.append(f"- `{row['query_id']}` {row['question']}")
    lines.extend(["", "## Sample Results", ""])
    for row in report["rows"][:10]:
        top = row.get("top_result") or {}
        lines.extend(
            [
                f"### {row['query_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Hit: {row['hit']}",
                f"- First hit rank: {row['first_hit_rank']}",
                f"- Top node: `{top.get('node_id')}`",
                f"- Top title: {top.get('title')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- Retrieval only.",
            "- No reranker or LLM judge was used.",
            "- No MCP runtime path was switched by this evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
