"""Stdio MCP server for read-only policy RAG evidence search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from src.backend.application.policy_rag.schemas import (
    DEFAULT_FETCH_K,
    FILTER_STRATEGY_SINGLE,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_TOP_K,
    RETRIEVAL_PROFILES,
)
from src.backend.application.policy_rag.search_policy_evidence_uc import (
    SearchPolicyEvidenceUseCase,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_MODEL_CACHE
from src.backend.infrastructure.policy_rag.retriever_factory import (
    build_policy_retriever,
)


mcp = FastMCP("policy-rag")
_USE_CASE: SearchPolicyEvidenceUseCase | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MediGuard Policy RAG MCP server over stdio."
    )
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument(
        "--retrieval-strategy",
        choices=tuple(RETRIEVAL_PROFILES),
        default=DEFAULT_RETRIEVAL_STRATEGY,
    )
    parser.add_argument(
        "--enable-reranker",
        dest="enable_reranker",
        action="store_true",
        help="Load bge-reranker-v2-m3 for conditional rerank requests.",
    )
    parser.add_argument(
        "--no-reranker",
        dest="enable_reranker",
        action="store_false",
        help="Keep the reranker unloaded. This is the default.",
    )
    parser.set_defaults(enable_reranker=False)
    return parser.parse_args()


def default_runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        cache_folder=DEFAULT_MODEL_CACHE,
        device=None,
        reranker_device=None,
        retrieval_strategy=DEFAULT_RETRIEVAL_STRATEGY,
        enable_reranker=False,
    )


def build_use_case(args: argparse.Namespace) -> SearchPolicyEvidenceUseCase:
    from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
    from src.backend.infrastructure.policy_rag.bge_reranker import BgePolicyReranker

    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    retriever = build_policy_retriever(
        strategy=args.retrieval_strategy,
        embedder=embedder,
    )
    reranker = None
    if args.enable_reranker:
        reranker = BgePolicyReranker(
            cache_folder=args.cache_folder,
            device=args.reranker_device or args.device,
            allow_unavailable=True,
        )
        if not reranker.available:
            print(
                f"Policy RAG reranker unavailable: {reranker.unavailable_reason}",
                file=sys.stderr,
            )
    return SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
        retrieval_strategy=args.retrieval_strategy,
    )


def get_use_case() -> SearchPolicyEvidenceUseCase:
    if _USE_CASE is None:
        return build_use_case(default_runtime_args())
    return _USE_CASE


@mcp.tool(name="policy_rag_search")
def policy_rag_search(
    question: str,
    filters: dict[str, Any],
    recall_filters: dict[str, Any] | None = None,
    filter_strategy: str = FILTER_STRATEGY_SINGLE,
    adaptive_top_k: bool = False,
    allow_broad_filters: bool = False,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    rerank: bool = False,
) -> dict[str, Any]:
    """Search policy evidence with explicit metadata filters.

    This tool is retrieval-only. It does not read case facts, calculate risk,
    or make reimbursement, refusal, punishment, or fraud decisions.
    """

    return get_use_case().search(
        {
            "question": question,
            "filters": filters,
            "recall_filters": recall_filters,
            "filter_strategy": filter_strategy,
            "adaptive_top_k": adaptive_top_k,
            "allow_broad_filters": allow_broad_filters,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "rerank": rerank,
        }
    )


def main() -> None:
    global _USE_CASE
    args = parse_args()
    _USE_CASE = build_use_case(args)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
