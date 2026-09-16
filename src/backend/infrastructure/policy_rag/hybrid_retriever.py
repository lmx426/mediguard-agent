"""Hybrid dense and lexical retriever for policy RAG evidence search."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, DefaultDict

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyIndex
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever
from src.backend.infrastructure.policy_rag.node_store import (
    PolicyNodeStore,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_NODES_PATH


DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_LEXICAL_WEIGHT = 1.0
EXACT_CODE_MATCH_RRF_BONUS = 1.0


class HybridPolicyRetriever:
    """Combine bge-m3 FAISS recall with local lexical recall using RRF."""

    retrieval_mode_name = "hybrid_dense_bge_m3_lexical_rrf"

    def __init__(
        self,
        *,
        dense_retriever: Any | None = None,
        embedder: BgeQueryEmbedder | None = None,
        nodes_path: Path = DEFAULT_NODES_PATH,
        node_store: PolicyNodeStore | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ) -> None:
        self._dense_retriever = dense_retriever or FaissPolicyRetriever(
            embedder=embedder
        )
        self._node_store = node_store or PolicyNodeStore(nodes_path)
        self._lexical_index = Bm25PolicyIndex(self._node_store)
        self._rrf_k = int(rrf_k)
        self._dense_weight = float(dense_weight)
        self._lexical_weight = float(lexical_weight)
        dense_mode = str(
            getattr(self._dense_retriever, "retrieval_mode_name", "dense_bge_m3")
        )
        self.retrieval_mode_name = f"hybrid_{dense_mode}_lexical_rrf"

    @property
    def node_count(self) -> int:
        return self._node_store.node_count

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
        lexical_candidates = self._lexical_index.retrieve(
            question=question,
            filters=filters,
            fetch_k=fetch_k,
        )
        return _fuse_rrf(
            dense_candidates=dense_candidates,
            lexical_candidates=lexical_candidates,
            fetch_k=fetch_k,
            rrf_k=self._rrf_k,
            dense_weight=self._dense_weight,
            lexical_weight=self._lexical_weight,
        )


def _fuse_rrf(
    *,
    dense_candidates: list[RetrievedPolicyCandidate],
    lexical_candidates: list[RetrievedPolicyCandidate],
    fetch_k: int,
    rrf_k: int,
    dense_weight: float,
    lexical_weight: float,
) -> list[RetrievedPolicyCandidate]:
    candidates_by_key: dict[tuple[str, object], RetrievedPolicyCandidate] = {}
    hybrid_scores: DefaultDict[tuple[str, object], float] = defaultdict(float)
    dense_scores: dict[tuple[str, object], float] = {}
    lexical_scores: dict[tuple[str, object], float] = {}
    dense_ranks: dict[tuple[str, object], int] = {}
    lexical_ranks: dict[tuple[str, object], int] = {}
    exact_code_matches: dict[tuple[str, object], bool] = {}

    for rank, candidate in enumerate(dense_candidates, start=1):
        key = _candidate_key(candidate)
        candidates_by_key.setdefault(key, candidate)
        dense_scores[key] = candidate.faiss_score
        dense_ranks[key] = rank
        hybrid_scores[key] += dense_weight / (rrf_k + rank)

    for rank, candidate in enumerate(lexical_candidates, start=1):
        key = _candidate_key(candidate)
        candidates_by_key.setdefault(key, candidate)
        lexical_scores[key] = candidate.faiss_score
        lexical_ranks[key] = rank
        exact_code_matches[key] = candidate.metadata.get("exact_code_match") is True
        hybrid_scores[key] += lexical_weight / (rrf_k + rank)
        if exact_code_matches[key]:
            hybrid_scores[key] += EXACT_CODE_MATCH_RRF_BONUS / (rrf_k + 1)

    fused: list[RetrievedPolicyCandidate] = []
    for key, candidate in candidates_by_key.items():
        hybrid_score = hybrid_scores[key]
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "dense_score": dense_scores.get(key),
                "lexical_score": lexical_scores.get(key),
                "hybrid_score": hybrid_score,
                "retrieval_strategy": "hybrid_dense_lexical_rrf",
                "score_type": "hybrid_score",
                "exact_code_match": exact_code_matches.get(key)
                or metadata.get("exact_code_match"),
            }
        )
        fused.append(
            replace(
                candidate,
                metadata=metadata,
                faiss_score=hybrid_score,
                rerank_score=None,
            )
        )

    fused.sort(
        key=lambda item: (
            item.faiss_score,
            -dense_ranks.get(_candidate_key(item), 10_000),
            -lexical_ranks.get(_candidate_key(item), 10_000),
        ),
        reverse=True,
    )
    return fused[:fetch_k]


def _source_key(candidate: RetrievedPolicyCandidate) -> str:
    return str(
        candidate.metadata.get("source_id")
        or candidate.metadata.get("doc_id")
        or candidate.node_id.split("::", 1)[0]
    )


def _candidate_key(candidate: RetrievedPolicyCandidate) -> tuple[str, object]:
    chunk_index = candidate.metadata.get("chunk_index")
    return (
        _source_key(candidate),
        chunk_index if chunk_index is not None else candidate.node_id,
    )
