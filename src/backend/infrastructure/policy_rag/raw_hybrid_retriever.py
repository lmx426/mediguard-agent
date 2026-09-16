"""Raw hybrid retrieval for Dense-vs-Hybrid experiments."""

from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyRetriever
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever
from src.backend.infrastructure.policy_rag.paths import DEFAULT_NODES_PATH


DEFAULT_RRF_K = 60


class RawHybridPolicyRetriever:
    """Fuse one dense pass and one BM25 pass without result orchestration."""

    retrieval_mode_name = "raw_hybrid_dense_bge_m3_bm25_rrf"

    def __init__(
        self,
        *,
        dense_retriever: FaissPolicyRetriever | None = None,
        lexical_retriever: Bm25PolicyRetriever | None = None,
        embedder: BgeQueryEmbedder | None = None,
        nodes_path: Path = DEFAULT_NODES_PATH,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._dense_retriever = dense_retriever or FaissPolicyRetriever(
            embedder=embedder
        )
        self._lexical_retriever = lexical_retriever or Bm25PolicyRetriever(
            nodes_path=nodes_path
        )
        self._rrf_k = int(rrf_k)

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        dense_candidates = self._dense_retriever.retrieve(
            question=question,
            filters=filters,
            fetch_k=fetch_k,
        )
        lexical_candidates = self._lexical_retriever.retrieve(
            question=question,
            filters=filters,
            fetch_k=fetch_k,
        )
        return _rrf_fuse(
            dense_candidates=dense_candidates,
            lexical_candidates=lexical_candidates,
            fetch_k=fetch_k,
            rrf_k=self._rrf_k,
        )


def _rrf_fuse(
    *,
    dense_candidates: list[RetrievedPolicyCandidate],
    lexical_candidates: list[RetrievedPolicyCandidate],
    fetch_k: int,
    rrf_k: int,
) -> list[RetrievedPolicyCandidate]:
    """Fuse by node_id; same-node overlap is fusion, not source de-duplication."""

    candidates_by_node: dict[str, RetrievedPolicyCandidate] = {}
    hybrid_scores: DefaultDict[str, float] = defaultdict(float)
    dense_scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}

    for rank, candidate in enumerate(dense_candidates, start=1):
        node_id = candidate.node_id
        candidates_by_node.setdefault(node_id, candidate)
        dense_scores[node_id] = candidate.faiss_score
        dense_ranks[node_id] = rank
        hybrid_scores[node_id] += 1.0 / (rrf_k + rank)

    for rank, candidate in enumerate(lexical_candidates, start=1):
        node_id = candidate.node_id
        candidates_by_node.setdefault(node_id, candidate)
        lexical_scores[node_id] = candidate.faiss_score
        lexical_ranks[node_id] = rank
        hybrid_scores[node_id] += 1.0 / (rrf_k + rank)

    fused: list[RetrievedPolicyCandidate] = []
    for node_id, candidate in candidates_by_node.items():
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "dense_score": dense_scores.get(node_id),
                "lexical_score": lexical_scores.get(node_id),
                "hybrid_score": hybrid_scores[node_id],
                "retrieval_strategy": "raw_hybrid_dense_bm25_rrf",
                "score_type": "hybrid_score",
                "rrf_k": rrf_k,
            }
        )
        fused.append(
            replace(
                candidate,
                metadata=metadata,
                faiss_score=hybrid_scores[node_id],
                rerank_score=None,
            )
        )

    fused.sort(
        key=lambda item: (
            item.faiss_score,
            -dense_ranks.get(item.node_id, 10_000),
            -lexical_ranks.get(item.node_id, 10_000),
        ),
        reverse=True,
    )
    return fused[:fetch_k]
