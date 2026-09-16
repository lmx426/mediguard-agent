"""Qdrant-backed dense retriever for the policy RAG runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.node_store import PolicyNodeStore
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_MODEL_CACHE,
    DEFAULT_NODES_PATH,
    DEFAULT_QDRANT_MANIFEST_PATH,
)


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_COLLECTION = "mediguard_policy_bge_m3"


class QdrantPolicyRetriever:
    """Search a Qdrant collection and apply payload filters server-side."""

    retrieval_mode_name = "dense_qdrant_bge_m3"

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
        manifest_path: Path = DEFAULT_QDRANT_MANIFEST_PATH,
        nodes_path: Path = DEFAULT_NODES_PATH,
        embedder: BgeQueryEmbedder | None = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
    ) -> None:
        QdrantClient = _load_qdrant_client()
        self._url = url or os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_URL") or DEFAULT_QDRANT_URL
        self._path = os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_PATH") or None
        self._collection_name = (
            collection_name
            or os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION")
            or DEFAULT_QDRANT_COLLECTION
        )
        self._manifest_path = manifest_path
        self._manifest = _read_optional_json(manifest_path)
        if self._path:
            self._client = QdrantClient(path=self._path)
        else:
            self._client = QdrantClient(
                url=self._url,
                api_key=api_key or os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_API_KEY") or None,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
        self._node_store = PolicyNodeStore(nodes_path)
        self._embedder = embedder or BgeQueryEmbedder(cache_folder=DEFAULT_MODEL_CACHE)
        self._ensure_collection_available()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        query_vector = self._embedder.embed_query(question)[0].tolist()
        qdrant_filter = _qdrant_filter(filters)
        points = _query_points(
            client=self._client,
            collection_name=self._collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=fetch_k,
        )

        candidates: list[RetrievedPolicyCandidate] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            node_id = str(payload.get("node_id") or "").strip()
            if not node_id:
                continue
            text = str(payload.get("text") or "")
            metadata = dict(payload.get("metadata") or {})
            if not text or not metadata:
                record = self._node_store.get(node_id)
                if record is None:
                    continue
                text = text or record.text
                metadata = metadata or dict(record.metadata)
            metadata.update(
                {
                    "qdrant_point_id": str(getattr(point, "id", "")),
                    "qdrant_score": float(getattr(point, "score", 0.0) or 0.0),
                    "retrieval_strategy": "dense_qdrant_bge_m3",
                    "score_type": "qdrant_score",
                }
            )
            candidates.append(
                RetrievedPolicyCandidate(
                    node_id=node_id,
                    text=text,
                    metadata=metadata,
                    faiss_score=float(getattr(point, "score", 0.0) or 0.0),
                )
            )
        return candidates[:fetch_k]

    def _ensure_collection_available(self) -> None:
        try:
            self._client.get_collection(self._collection_name)
        except Exception as exc:  # pragma: no cover - depends on local Qdrant
            location = (
                f"local path {self._path!r}"
                if self._path
                else f"Qdrant URL {self._url!r}"
            )
            raise RuntimeError(
                "Qdrant policy collection is not available. "
                "Run build_qdrant_policy_index.py first or set "
                f"MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION correctly. Location: {location}."
            ) from exc


def _load_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qdrant-client is required for Qdrant policy retrieval. "
            "Install it with: python -m pip install qdrant-client"
        ) from exc
    return QdrantClient


def _load_qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qdrant-client is required for Qdrant policy retrieval. "
            "Install it with: python -m pip install qdrant-client"
        ) from exc
    return models


def _qdrant_filter(filters: dict[str, object]) -> Any:
    models = _load_qdrant_models()
    must = []
    for key in ("jurisdiction", "policy_domain"):
        values = [str(item) for item in filters.get(key) or [] if str(item)]
        if not values:
            continue
        if len(values) == 1:
            match = models.MatchValue(value=values[0])
        else:
            match = models.MatchAny(any=values)
        must.append(models.FieldCondition(key=f"metadata.{key}", match=match))

    cite_filter = filters.get("can_cite_as_policy_basis")
    if cite_filter is not None:
        must.append(
            models.FieldCondition(
                key="metadata.can_cite_as_policy_basis",
                match=models.MatchValue(value=bool(cite_filter)),
            )
        )
    return models.Filter(must=must) if must else None


def _query_points(
    *,
    client: Any,
    collection_name: str,
    query_vector: list[float],
    query_filter: Any,
    limit: int,
) -> list[Any]:
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(getattr(result, "points", result) or [])
    return list(
        client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        or []
    )


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
