"""Query the local FAISS policy RAG index and run smoke tests.

This script is for retrieval validation only. It embeds query text with the
same bge-m3 model, searches the local FAISS index, and returns source-grounded
chunks. It does not run reranking, LLM generation, MCP tools, Case Agent
integration, or audit decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_INDEX_DIR = DEFAULT_RAG_READY_ROOT / "indexes" / "faiss_bge_m3_flat_ip"
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "policy.index"
DEFAULT_INDEX_NODE_IDS = DEFAULT_INDEX_DIR / "index_node_ids.jsonl"
DEFAULT_INDEX_MANIFEST = DEFAULT_INDEX_DIR / "index_manifest.json"
DEFAULT_NODES = DEFAULT_RAG_READY_ROOT / "nodes" / "policy_nodes.jsonl"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_faiss_retrieval_smoke_report.md"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_faiss_retrieval_smoke_report.json"

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 50

SMOKE_QUERIES: list[dict[str, Any]] = [
    {
        "id": "remote_manual_reimbursement_materials",
        "question": "北京参保人在上海异地门急诊手工报销需要什么材料？",
        "expected_terms": ["北京", "异地", "报销"],
    },
    {
        "id": "remote_settlement_principle",
        "question": "跨省异地就医为什么是就医地目录、参保地政策？",
        "expected_terms": ["就医地", "参保地", "政策"],
    },
    {
        "id": "shanghai_ct_price",
        "question": "上海胸部 CT 平扫价格目录怎么查？",
        "expected_terms": ["上海", "CT", "价格"],
    },
    {
        "id": "amoxicillin_catalog",
        "question": "阿莫西林是否在医保药品目录中？",
        "expected_terms": ["阿莫西林", "药品", "目录"],
    },
    {
        "id": "beijing_manual_reimbursement",
        "question": "北京手工报销需要哪些材料？",
        "expected_terms": ["北京", "手工", "材料"],
    },
]


@dataclass(frozen=True, slots=True)
class NodeRecord:
    node_id: str
    text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local policy FAISS index or run retrieval smoke tests."
    )
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--index-node-ids", type=Path, default=DEFAULT_INDEX_NODE_IDS)
    parser.add_argument("--index-manifest", type=Path, default=DEFAULT_INDEX_MANIFEST)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--jurisdiction", action="append", default=[])
    parser.add_argument("--policy-domain", action="append", default=[])
    parser.add_argument("--content-type", action="append", default=[])
    parser.add_argument("--include-retrieval-hints", action="store_true")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write report files for custom --question runs as well as smoke tests.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def load_index_node_ids(path: Path) -> list[str]:
    node_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            embedding_index = payload.get("embedding_index")
            if embedding_index != len(node_ids):
                raise ValueError(
                    f"Non-sequential embedding_index at {path}:{line_no}: {embedding_index}"
                )
            node_id = str(payload.get("node_id") or "").strip()
            if not node_id:
                raise ValueError(f"Missing node_id at {path}:{line_no}")
            node_ids.append(node_id)
    return node_ids


def load_nodes(path: Path) -> dict[str, NodeRecord]:
    nodes: dict[str, NodeRecord] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            node = TextNode.from_dict(payload)
            if not node.id_:
                raise ValueError(f"TextNode at {path}:{line_no} is missing id_")
            nodes[node.id_] = NodeRecord(
                node_id=node.id_,
                text=node.text or "",
                metadata=dict(node.metadata or {}),
            )
    return nodes


def build_embed_model(args: argparse.Namespace) -> HuggingFaceEmbedding:
    if args.cache_folder:
        args.cache_folder.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=args.model_name,
        max_length=args.max_length,
        normalize=True,
        cache_folder=str(args.cache_folder) if args.cache_folder else None,
        device=args.device,
        trust_remote_code=False,
        show_progress_bar=False,
    )


def embed_query(embed_model: HuggingFaceEmbedding, question: str) -> np.ndarray:
    embedding = np.asarray(embed_model.get_query_embedding(question), dtype=np.float32)
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D query embedding, got shape {embedding.shape}")
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = embedding / norm
    return np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)


def record_matches_filters(
    record: NodeRecord,
    *,
    jurisdictions: set[str],
    policy_domains: set[str],
    content_types: set[str],
    include_retrieval_hints: bool,
) -> bool:
    metadata = record.metadata
    if jurisdictions and str(metadata.get("jurisdiction") or "") not in jurisdictions:
        return False
    if policy_domains and str(metadata.get("policy_domain") or "") not in policy_domains:
        return False
    if content_types and str(metadata.get("content_type") or "") not in content_types:
        return False
    if (
        not include_retrieval_hints
        and metadata.get("evidence_role") == "retrieval_hint"
    ):
        return False
    return True


def result_payload(
    *,
    rank: int,
    score: float,
    node_id: str,
    record: NodeRecord,
) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "rank": rank,
        "score": score,
        "node_id": node_id,
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
        "text": record.text,
    }


def retrieve(
    *,
    question: str,
    index: faiss.Index,
    node_ids: list[str],
    nodes: dict[str, NodeRecord],
    embed_model: HuggingFaceEmbedding,
    top_k: int,
    fetch_k: int,
    jurisdictions: set[str],
    policy_domains: set[str],
    content_types: set[str],
    include_retrieval_hints: bool,
) -> dict[str, Any]:
    query_vector = embed_query(embed_model, question)
    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension {query_vector.shape[1]} does not match index dimension {index.d}"
        )
    search_k = max(fetch_k, top_k)
    scores, indexes = index.search(query_vector, search_k)
    results: list[dict[str, Any]] = []
    for score, embedding_index in zip(scores[0].tolist(), indexes[0].tolist(), strict=True):
        if embedding_index < 0:
            continue
        node_id = node_ids[int(embedding_index)]
        record = nodes.get(node_id)
        if record is None:
            continue
        if not record_matches_filters(
            record,
            jurisdictions=jurisdictions,
            policy_domains=policy_domains,
            content_types=content_types,
            include_retrieval_hints=include_retrieval_hints,
        ):
            continue
        results.append(
            result_payload(
                rank=len(results) + 1,
                score=float(score),
                node_id=node_id,
                record=record,
            )
        )
        if len(results) >= top_k:
            break
    return {
        "question": question,
        "top_k": top_k,
        "fetch_k": search_k,
        "result_count": len(results),
        "results": results,
    }


def term_hit_count(results: list[dict[str, Any]], expected_terms: list[str]) -> int:
    haystack = "\n".join(
        str(result.get("title") or "") + "\n" + str(result.get("text") or "")
        for result in results
    )
    return sum(1 for term in expected_terms if term in haystack)


def smoke_validation(
    query: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    results = list(retrieval.get("results") or [])
    expected_terms = list(query.get("expected_terms") or [])
    citeable_count = sum(
        1 for result in results if result.get("can_cite_as_policy_basis") is True
    )
    source_url_count = sum(1 for result in results if result.get("source_url"))
    top_score = float(results[0]["score"]) if results else None
    expected_hit_count = term_hit_count(results, expected_terms)
    return {
        "query_id": query["id"],
        "result_count": len(results),
        "citeable_result_count": citeable_count,
        "source_url_count": source_url_count,
        "top_score": top_score,
        "expected_terms": expected_terms,
        "expected_term_hit_count": expected_hit_count,
        "passed": bool(
            results
            and citeable_count >= 1
            and source_url_count >= 1
            and top_score is not None
            and np.isfinite(top_score)
        ),
    }


def load_runtime(args: argparse.Namespace) -> tuple[faiss.Index, list[str], dict[str, NodeRecord], HuggingFaceEmbedding]:
    index_manifest = read_json(args.index_manifest)
    if index_manifest.get("status") != "complete":
        raise RuntimeError(f"Index manifest is not complete: {args.index_manifest}")
    index = faiss.read_index(str(args.index_path))
    node_ids = load_index_node_ids(args.index_node_ids)
    if index.ntotal != len(node_ids):
        raise RuntimeError(
            f"Index ntotal {index.ntotal} does not match node-id map {len(node_ids)}"
        )
    nodes = load_nodes(args.nodes)
    missing_nodes = [node_id for node_id in node_ids if node_id not in nodes]
    if missing_nodes:
        raise RuntimeError(f"Node store is missing {len(missing_nodes)} indexed node IDs")
    embed_model = build_embed_model(args)
    return index, node_ids, nodes, embed_model


def run_queries(args: argparse.Namespace) -> dict[str, Any]:
    index, node_ids, nodes, embed_model = load_runtime(args)
    questions = list(args.question)
    if args.smoke_test:
        questions.extend(query["question"] for query in SMOKE_QUERIES)
    if not questions:
        raise ValueError("Provide --question or --smoke-test.")

    jurisdictions = set(args.jurisdiction)
    policy_domains = set(args.policy_domain)
    content_types = set(args.content_type)
    retrievals = [
        retrieve(
            question=question,
            index=index,
            node_ids=node_ids,
            nodes=nodes,
            embed_model=embed_model,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            jurisdictions=jurisdictions,
            policy_domains=policy_domains,
            content_types=content_types,
            include_retrieval_hints=args.include_retrieval_hints,
        )
        for question in questions
    ]

    smoke_results: list[dict[str, Any]] = []
    if args.smoke_test:
        smoke_offset = len(args.question)
        for query, retrieval in zip(SMOKE_QUERIES, retrievals[smoke_offset:], strict=True):
            smoke_results.append(smoke_validation(query, retrieval))

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "retrieval_mode": "faiss_flat_ip_bge_m3",
        "index_path": str(args.index_path),
        "index_node_ids": str(args.index_node_ids),
        "nodes": str(args.nodes),
        "model_name": args.model_name,
        "cache_folder": str(args.cache_folder) if args.cache_folder else None,
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "filters": {
            "jurisdiction": sorted(jurisdictions),
            "policy_domain": sorted(policy_domains),
            "content_type": sorted(content_types),
            "include_retrieval_hints": args.include_retrieval_hints,
        },
        "query_count": len(retrievals),
        "queries": retrievals,
        "smoke_test": {
            "enabled": args.smoke_test,
            "passed": all(item["passed"] for item in smoke_results) if args.smoke_test else None,
            "results": smoke_results,
        },
        "boundary": {
            "retrieval_only": True,
            "runs_reranker": False,
            "runs_llm_generation": False,
            "integrates_mcp": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    smoke = report.get("smoke_test") or {}
    lines = [
        "# Policy FAISS Retrieval Smoke Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report validates local FAISS retrieval only. It does not run reranking, LLM generation, MCP tools, Case Agent integration, or audit decisions.",
        "",
        "## Summary",
        "",
        f"- Retrieval mode: `{report.get('retrieval_mode')}`",
        f"- Model: `{report.get('model_name')}`",
        f"- Index path: `{report.get('index_path')}`",
        f"- Node store: `{report.get('nodes')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Top K: {report.get('top_k')}",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Smoke test enabled: {smoke.get('enabled')}",
        f"- Smoke test passed: {smoke.get('passed')}",
        "",
    ]
    if smoke.get("enabled"):
        lines.extend(["## Smoke Validation", ""])
        for item in smoke.get("results") or []:
            lines.extend(
                [
                    f"### {item.get('query_id')}",
                    "",
                    f"- Passed: {item.get('passed')}",
                    f"- Result count: {item.get('result_count')}",
                    f"- Citeable result count: {item.get('citeable_result_count')}",
                    f"- Source URL count: {item.get('source_url_count')}",
                    f"- Top score: {item.get('top_score')}",
                    f"- Expected term hits: {item.get('expected_term_hit_count')} / {len(item.get('expected_terms') or [])}",
                    "",
                ]
            )
    lines.extend(["## Query Results", ""])
    for query in report.get("queries") or []:
        lines.extend([f"### {query.get('question')}", ""])
        for result in query.get("results") or []:
            text = str(result.get("text") or "").replace("\n", " ")
            if len(text) > 260:
                text = text[:260] + "..."
            lines.extend(
                [
                    f"- Rank {result.get('rank')} | score={result.get('score'):.6f} | `{result.get('jurisdiction')}` / `{result.get('policy_domain')}` / `{result.get('content_type')}`",
                    f"  Title: {result.get('title')}",
                    f"  Source: {result.get('source_url')}",
                    f"  Text: {text}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- No reranker was used.",
            "- No LLM answer was generated.",
            "- No Agent or audit workflow was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.fetch_k <= 0:
        raise ValueError("--fetch-k must be positive")
    report = run_queries(args)
    should_write_report = args.smoke_test or args.write_report
    if should_write_report:
        write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "retrieval_mode": report["retrieval_mode"],
                "query_count": report["query_count"],
                "smoke_test_passed": report["smoke_test"]["passed"],
                "report_json": str(args.report_json) if should_write_report else None,
                "report_md": str(args.report_md) if should_write_report else None,
                "queries": [
                    {
                        "question": query["question"],
                        "result_count": query["result_count"],
                        "top_result": query["results"][0] if query["results"] else None,
                    }
                    for query in report["queries"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
