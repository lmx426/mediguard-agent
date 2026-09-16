"""Reciprocal-rank fusion across strict and recall-safe retrieval branches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import DefaultDict

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate


DEFAULT_BRANCH_RRF_K = 60


def fuse_branch_rankings(
    *,
    strict_candidates: list[RetrievedPolicyCandidate],
    recall_candidates: list[RetrievedPolicyCandidate],
    limit: int,
    rrf_k: int = DEFAULT_BRANCH_RRF_K,
) -> list[RetrievedPolicyCandidate]:
    """Fuse two independently ranked lists using equal-weight RRF."""

    candidates: dict[tuple[str, object], RetrievedPolicyCandidate] = {}
    scores: DefaultDict[tuple[str, object], float] = defaultdict(float)
    branches: DefaultDict[tuple[str, object], list[str]] = defaultdict(list)
    branch_ranks: DefaultDict[tuple[str, object], dict[str, int]] = defaultdict(dict)

    for branch_name, ranked in (
        ("strict", strict_candidates),
        ("recall", recall_candidates),
    ):
        for rank, candidate in enumerate(ranked, start=1):
            key = _candidate_key(candidate)
            previous = candidates.get(key)
            if previous is None or candidate.faiss_score > previous.faiss_score:
                candidates[key] = candidate
            scores[key] += 1.0 / (rrf_k + rank)
            if branch_name not in branches[key]:
                branches[key].append(branch_name)
            branch_ranks[key][branch_name] = rank

    fused: list[RetrievedPolicyCandidate] = []
    for key, candidate in candidates.items():
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "pre_branch_fusion_score": candidate.faiss_score,
                "branch_rrf_score": scores[key],
                "retrieval_branches": list(branches[key]),
                "branch_ranks": dict(branch_ranks[key]),
                "score_type": "branch_rrf_score",
            }
        )
        fused.append(
            replace(
                candidate,
                metadata=metadata,
                faiss_score=scores[key],
                rerank_score=None,
            )
        )

    fused.sort(
        key=lambda item: (
            item.faiss_score,
            "strict" in item.metadata.get("retrieval_branches", []),
        ),
        reverse=True,
    )
    return fused[: max(1, int(limit))]


def _candidate_key(candidate: RetrievedPolicyCandidate) -> tuple[str, object]:
    source = str(
        candidate.metadata.get("source_id")
        or candidate.metadata.get("doc_id")
        or candidate.node_id.split("::", 1)[0]
    )
    chunk_index = candidate.metadata.get("chunk_index")
    return (source, chunk_index if chunk_index is not None else candidate.node_id)
