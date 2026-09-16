"""Concrete Mem0 adapter kept behind the MediGuard memory port."""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
from typing import Any

from ..application.ports.case_memory import MemoMemoryPort, MemoProjectionError
from ..domain.case_memory.entities import (
    MemoryRecallRequest,
    MemoryRecord,
    MemoryVectorProjection,
)


class Mem0MemoMemoryAdapter(MemoMemoryPort):
    def __init__(self, *, enabled: bool, database_url: str, runtime_dir: Path, llm_model: str, llm_base_url: str, llm_api_key: str, embedding_model: str, embedding_cache_dir: Path, projection_collection: str = "mediguard_case_memory") -> None:
        self.enabled = enabled
        self._database_url = database_url
        self._runtime_dir = runtime_dir
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._embedding_model = embedding_model
        self._embedding_cache_dir = embedding_cache_dir
        self._projection_collection = projection_collection
        self._memory: Any | None = None
        self.last_error: str | None = None

    def _get_memory(self) -> Any | None:
        if not self.enabled:
            return None
        if self._memory is not None:
            return self._memory
        try:
            from mem0 import Memory

            self._runtime_dir.mkdir(parents=True, exist_ok=True)
            embedding_model, local_files_only = self._resolve_embedding_model()
            config = {
                "history_db_path": str(self._runtime_dir / "history.db"),
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "connection_string": self._database_url.replace("+psycopg", ""),
                        "collection_name": self._projection_collection,
                        "embedding_model_dims": 1024,
                        "hnsw": True,
                    },
                },
                "llm": {
                    "provider": "deepseek",
                    "config": {
                        "model": self._llm_model,
                        "api_key": self._llm_api_key,
                        "deepseek_base_url": self._llm_base_url,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": embedding_model,
                        "model_kwargs": {
                            "cache_folder": str(self._embedding_cache_dir),
                            "local_files_only": local_files_only,
                        },
                    },
                },
                "custom_instructions": (
                    "Only create a short redacted memory summary. Never include RES, "
                    "identity fields, prompts, scores, source documents or raw case facts."
                ),
            }
            self._memory = Memory.from_config(
                config
            )
            return self._memory
        except Exception as exc:  # pragma: no cover - depends on optional Mem0 runtime
            self.last_error = exc.__class__.__name__
            raise MemoProjectionError("Mem0 initialization failed") from exc

    def add_active_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> str | None:
        client = self._get_memory()
        if client is None:
            return None
        try:
            result = client.add(
                projection.embedding_text,
                user_id=self._entity_id(memory.scope.scope_id),
                infer=False,
                metadata=self._metadata(memory, projection),
                expiration_date=self._expiration_date(memory.expires_at),
            )
            items = result.get("results", []) if isinstance(result, dict) else result
            if isinstance(items, list) and items:
                return str(items[0].get("id") or "") or None
        except Exception as exc:  # pragma: no cover
            self.last_error = exc.__class__.__name__
            raise MemoProjectionError("Mem0 add projection failed") from exc
        raise MemoProjectionError("Mem0 add projection returned no id")

    def search(
        self,
        request: MemoryRecallRequest,
        *,
        query_text: str,
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            client = self._get_memory()
        except MemoProjectionError:
            # Recall degrades to governed SQL/BM25/Graph candidates.
            return []
        if client is None:
            return []
        try:
            query = query_text or str(request.task_context)
            scope_ids = [
                request.scope.scope_id,
                *(scope.scope_id for scope in request.scope_fallbacks),
            ]
            if "global" not in scope_ids:
                scope_ids.append("global")
            hits: list[dict[str, Any]] = []
            for scope_id in dict.fromkeys(scope_ids):
                result = client.search(
                    query,
                    top_k=request.max_items * 4,
                    filters={
                        **metadata_filter,
                        "memory_type": request.memory_type.value,
                        "user_id": self._entity_id(scope_id),
                    },
                    rerank=False,
                    show_expired=False,
                )
                items = result.get("results", result) if isinstance(result, dict) else result
                for item in items if isinstance(items, list) else []:
                    metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                    case_memory_id = str(metadata.get("case_memory_id") or "")
                    if case_memory_id:
                        hits.append(
                            {
                                "case_memory_id": case_memory_id,
                                "memo_memory_id": str(item.get("id") or ""),
                                "score": float(item.get("score") or 0.0),
                            }
                        )
            deduped: dict[str, dict[str, Any]] = {}
            for hit in hits:
                existing = deduped.get(hit["case_memory_id"])
                if existing is None or hit["score"] > existing["score"]:
                    deduped[hit["case_memory_id"]] = hit
            return sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
        except Exception as exc:  # pragma: no cover
            self.last_error = exc.__class__.__name__
            return []

    def update_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> None:
        client = self._get_memory()
        if client is None or not memory.memo_memory_id:
            return
        try:
            client.update(
                memory.memo_memory_id,
                text=projection.embedding_text,
                metadata=self._metadata(memory, projection),
                expiration_date=self._expiration_date(memory.expires_at),
            )
        except Exception as exc:  # pragma: no cover
            self.last_error = exc.__class__.__name__
            raise MemoProjectionError("Mem0 update projection failed") from exc

    def delete_projection(self, memory: MemoryRecord) -> None:
        client = self._get_memory()
        if client is None or not memory.memo_memory_id:
            return
        try:
            client.delete(memory.memo_memory_id)
        except Exception as exc:  # pragma: no cover
            self.last_error = exc.__class__.__name__
            raise MemoProjectionError("Mem0 delete projection failed") from exc

    @staticmethod
    def _metadata(
        memory: MemoryRecord,
        projection: MemoryVectorProjection | None = None,
    ) -> dict[str, Any]:
        payload = memory.payload
        structured = payload.get("structured_content") if isinstance(payload.get("structured_content"), dict) else {}
        recommended = payload.get("recommended_action") if isinstance(payload.get("recommended_action"), dict) else {}
        stable = payload.get("stable_preferences") if isinstance(payload.get("stable_preferences"), dict) else {}
        information_needs = (
            structured.get("information_needs")
            or recommended.get("information_needs")
            or stable.get("information_needs")
            or stable.get("policy_search", {}).get("information_needs", [])
        )
        metadata = {
            "case_memory_id": memory.memory_id,
            "memory_type": memory.memory_type.value,
            "memory_level": memory.memory_level.value,
            "status": memory.status.value,
            "scope_type": memory.scope.scope_type,
            "scope_id_hash": hashlib.sha256(memory.scope.scope_id.encode()).hexdigest(),
            "consumer_tags": memory.allowed_consumers,
            "policy_domain": payload.get("policy_domain"),
            "jurisdiction": payload.get("jurisdiction"),
            "information_need_tags": information_needs if isinstance(information_needs, list) else [str(information_needs)],
            "freshness_bucket": round(memory.freshness_score, 1),
            "importance_bucket": round(memory.importance_score, 1),
            "created_at": memory.created_at.isoformat(),
        }
        if projection is not None:
            metadata.update(projection.metadata)
            metadata["projection_version"] = projection.projection_version
        return metadata

    @staticmethod
    def _entity_id(scope_id: str) -> str:
        """Keep Mem0 entity IDs opaque while preserving deterministic scope routing."""

        return f"mediguard_scope_{hashlib.sha256(scope_id.encode('utf-8')).hexdigest()[:32]}"

    @staticmethod
    def _expiration_date(value: datetime | None) -> str | None:
        return value.date().isoformat() if value is not None else None

    def _resolve_embedding_model(self) -> tuple[str, bool]:
        """Use an existing HuggingFace snapshot when available, without network HEAD calls."""

        model_dir = self._embedding_cache_dir / (
            "models--" + self._embedding_model.replace("/", "--")
        )
        ref_file = model_dir / "refs" / "main"
        if ref_file.is_file():
            revision = ref_file.read_text(encoding="utf-8").strip()
            snapshot = model_dir / "snapshots" / revision
            if revision and (snapshot / "config.json").is_file():
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                return str(snapshot), True
        return self._embedding_model, False
