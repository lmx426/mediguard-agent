"""Factory for versioned Policy RAG retrieval profiles."""

from __future__ import annotations

import os
from typing import Any

from src.backend.application.policy_rag.schemas import (
    RETRIEVAL_STRATEGY_DENSE,
    RETRIEVAL_STRATEGY_HYBRID,
    normalize_retrieval_strategy,
)
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever
from src.backend.infrastructure.policy_rag.hybrid_retriever import HybridPolicyRetriever


VECTOR_BACKEND_FAISS = "faiss"
VECTOR_BACKEND_QDRANT = "qdrant"
DEFAULT_VECTOR_BACKEND = VECTOR_BACKEND_FAISS


def build_policy_retriever(
    *,
    strategy: object,
    embedder: BgeQueryEmbedder,
) -> Any:
    """Build the configured retrieval implementation for Policy RAG MCP."""

    normalized = normalize_retrieval_strategy(strategy)
    vector_backend = normalize_vector_backend(
        os.environ.get("MEDIGUARD_POLICY_RAG_VECTOR_BACKEND")
    )
    dense_retriever = build_dense_retriever(
        vector_backend=vector_backend,
        embedder=embedder,
    )
    if normalized == RETRIEVAL_STRATEGY_HYBRID:
        return HybridPolicyRetriever(dense_retriever=dense_retriever)
    if normalized == RETRIEVAL_STRATEGY_DENSE:
        return dense_retriever
    raise ValueError(f"Unsupported policy RAG retrieval strategy: {strategy!r}")


def build_dense_retriever(
    *,
    vector_backend: object,
    embedder: BgeQueryEmbedder,
) -> Any:
    backend = normalize_vector_backend(vector_backend)
    if backend == VECTOR_BACKEND_QDRANT:
        from src.backend.infrastructure.policy_rag.qdrant_retriever import (
            QdrantPolicyRetriever,
        )

        return QdrantPolicyRetriever(embedder=embedder)
    return FaissPolicyRetriever(embedder=embedder)


def normalize_vector_backend(value: object) -> str:
    backend = str(value or DEFAULT_VECTOR_BACKEND).strip().lower()
    if backend in {VECTOR_BACKEND_FAISS, VECTOR_BACKEND_QDRANT}:
        return backend
    return DEFAULT_VECTOR_BACKEND
