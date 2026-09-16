"""FAISS-backed dense retriever for the policy RAG MCP service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.node_store import PolicyNodeStore
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_INDEX_MANIFEST_PATH,
    DEFAULT_INDEX_NODE_IDS_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_NODES_PATH,
)


class FaissPolicyRetriever:
    """Searches the local bge-m3 FAISS index and applies metadata filters."""

    def __init__(
        self,
        *,
        index_path: Path = DEFAULT_INDEX_PATH,
        index_node_ids_path: Path = DEFAULT_INDEX_NODE_IDS_PATH,
        index_manifest_path: Path = DEFAULT_INDEX_MANIFEST_PATH,
        nodes_path: Path = DEFAULT_NODES_PATH,
        embedder: BgeQueryEmbedder | None = None,
    ) -> None:
        self._index_path = index_path
        self._index_node_ids_path = index_node_ids_path
        self._index_manifest_path = index_manifest_path
        self._nodes_path = nodes_path
        self._manifest = _read_json(index_manifest_path)
        if self._manifest.get("status") != "complete":
            raise RuntimeError(f"Policy FAISS index is not complete: {index_manifest_path}")
        self._index = faiss.read_index(str(index_path))
        self._index_node_ids = _load_index_node_ids(index_node_ids_path)
        if self._index.ntotal != len(self._index_node_ids):
            raise RuntimeError(
                f"Policy FAISS index ntotal {self._index.ntotal} does not match "
                f"node map {len(self._index_node_ids)}"
            )
        self._node_store = PolicyNodeStore(nodes_path)
        self._node_store.require_all(self._index_node_ids)
        self._embedder = embedder or BgeQueryEmbedder()

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def node_count(self) -> int:
        return self._index.ntotal

    @property
    def embedding_dim(self) -> int:
        return self._index.d

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        query_vector = self._embedder.embed_query(question)
        if query_vector.shape[1] != self._index.d:
            raise ValueError(
                f"Query embedding dimension {query_vector.shape[1]} does not match "
                f"FAISS index dimension {self._index.d}"
            )

        search_k = min(self._index.ntotal, max(fetch_k * 5, fetch_k))
        candidates: list[RetrievedPolicyCandidate] = []
        while True:
            candidates = self._search_and_filter(
                query_vector=query_vector,
                filters=filters,
                search_k=search_k,
                fetch_k=fetch_k,
            )
            if len(candidates) >= fetch_k or search_k >= self._index.ntotal:
                return candidates
            search_k = min(self._index.ntotal, max(search_k * 2, search_k + fetch_k))

    def _search_and_filter(
        self,
        *,
        query_vector: Any,
        filters: dict[str, object],
        search_k: int,
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        scores, indexes = self._index.search(query_vector, search_k)
        candidates: list[RetrievedPolicyCandidate] = []
        for score, embedding_index in zip(
            scores[0].tolist(),
            indexes[0].tolist(),
            strict=True,
        ):
            if embedding_index < 0:
                continue
            node_id = self._index_node_ids[int(embedding_index)]
            record = self._node_store.get(node_id)
            if record is None:
                continue
            if not self._node_store.matches_filters(record, filters):
                continue
            candidates.append(
                RetrievedPolicyCandidate(
                    node_id=node_id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    faiss_score=float(score),
                )
            )
            if len(candidates) >= fetch_k:
                break
        return candidates


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_index_node_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(str(path))
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
                    f"Non-sequential embedding_index at {path}:{line_no}: "
                    f"{embedding_index}"
                )
            node_id = str(payload.get("node_id") or "").strip()
            if not node_id:
                raise ValueError(f"Missing node_id at {path}:{line_no}")
            node_ids.append(node_id)
    if not node_ids:
        raise RuntimeError(f"No policy index node IDs loaded from {path}")
    return node_ids
