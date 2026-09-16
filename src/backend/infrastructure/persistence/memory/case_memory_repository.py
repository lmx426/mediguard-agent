"""In-memory implementations used by UI smoke tests and local fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any

from ....application.ports.case_memory import (
    EntityGraphProjectionRepository,
    MemoMemoryPort,
    MemoryEventRepository,
    MemoryOutboxRepository,
    MemoryRepository,
)
from ....domain.case_memory.entities import MemoryRecord, MemoryRecallRequest
from ....domain.case_memory.entities import MemoryQueryProfile, MemoryVectorProjection
from ....application.case_memory.retrieval import (
    MemoryGraphProjectionRegistry,
    MemoryQueryBuilder,
)


def _memory_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", str(value)):
        normalized = token.casefold()
        if len(normalized) <= 1:
            continue
        tokens.add(normalized)
        if re.search(r"[\u4e00-\u9fff]", normalized):
            tokens.update(normalized[index : index + 2] for index in range(len(normalized) - 1))
    return tokens


class InMemoryMemoryRepository(MemoryRepository):
    def __init__(self) -> None:
        self._items: dict[str, MemoryRecord] = {}

    def save(self, memory: MemoryRecord) -> MemoryRecord:
        self._items[memory.memory_id] = memory
        return memory

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._items.get(memory_id)

    def get_many(self, memory_ids) -> list[MemoryRecord]:
        return [self._items[memory_id] for memory_id in memory_ids if memory_id in self._items]

    def list(self, *, status: str | None = None, scope_type: str | None = None, scope_id: str | None = None, memory_type: str | None = None, limit: int | None = None) -> list[MemoryRecord]:
        rows = list(self._items.values())
        if status is not None:
            rows = [row for row in rows if row.status.value == status]
        if scope_type is not None:
            rows = [row for row in rows if row.scope.scope_type == scope_type]
        if scope_id is not None:
            rows = [row for row in rows if row.scope.scope_id == scope_id]
        if memory_type is not None:
            rows = [row for row in rows if row.memory_type.value == memory_type]
        rows = sorted(rows, key=lambda row: row.created_at, reverse=True)
        return rows[:limit] if limit is not None else rows

    def update(self, memory: MemoryRecord) -> MemoryRecord:
        self._items[memory.memory_id] = memory
        return memory


class InMemoryMemoryEventRepository(MemoryEventRepository):
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def append(self, memory_id: str, event_type: str, actor_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        event = {"event_id": f"mevent_{len(self._events) + 1}", "memory_id": memory_id, "event_type": event_type, "actor_id": actor_id, "payload": payload, "created_at": datetime.now(timezone.utc)}
        self._events.append(event)
        return event

    def list_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        return [event for event in self._events if event["memory_id"] == memory_id]


class InMemoryMemoryOutboxRepository(MemoryOutboxRepository):
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def enqueue(self, event_type: str, aggregate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        idempotency_key = f"{event_type}:{aggregate_id}:{payload_hash}"
        for event in self._events:
            if event.get("idempotency_key") == idempotency_key:
                return event
        event = {"event_id": f"outbox_{len(self._events) + 1}", "event_type": event_type, "aggregate_id": aggregate_id, "payload": payload, "status": "pending", "retry_count": 0, "next_attempt_at": datetime.now(timezone.utc), "locked_until": None}
        event["idempotency_key"] = idempotency_key
        self._events.append(event)
        return event

    def claim_next_batch(self, limit: int = 50) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        rows = [
            event
            for event in self._events
            if event["status"] in {"pending", "retrying", "processing"}
            and event.get("next_attempt_at", now) <= now
            and (event.get("locked_until") is None or event["locked_until"] <= now)
        ][:limit]
        for event in rows:
            event["status"] = "processing"
            event["locked_until"] = now + timedelta(minutes=5)
        return rows

    def mark_processed(self, event_id: str) -> None:
        for event in self._events:
            if event["event_id"] == event_id:
                event["status"] = "processed"
                event["locked_until"] = None

    def mark_failed(self, event_id: str, error: str) -> None:
        for event in self._events:
            if event["event_id"] == event_id:
                event["retry_count"] += 1
                event["last_error"] = error
                event["locked_until"] = None
                if event["retry_count"] >= 5:
                    event["status"] = "dead_letter"
                else:
                    event["status"] = "retrying"
                    event["next_attempt_at"] = datetime.now(timezone.utc) + timedelta(minutes=min(30, 2 ** event["retry_count"]))

    def status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._events:
            status = str(event.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts


class InMemoryMemoMemoryAdapter(MemoMemoryPort):
    """Deterministic fallback with the same boundary as the Mem0 adapter."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(type(self), "enabled", False))
        self._items: dict[str, MemoryRecord] = {}
        self._texts: dict[str, str] = {}

    def add_active_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> str | None:
        self._items[memory.memory_id] = memory
        self._texts[memory.memory_id] = projection.embedding_text
        return memory.memory_id

    def search(
        self,
        request: MemoryRecallRequest,
        *,
        query_text: str,
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_tokens = _memory_tokens(query_text)
        rows = []
        for memory in self._items.values():
            if memory.memory_type != request.memory_type or memory.status.value != "active":
                continue
            text_tokens = _memory_tokens(self._texts.get(memory.memory_id, memory.summary))
            score = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            rows.append({"case_memory_id": memory.memory_id, "score": float(score)})
        return sorted(rows, key=lambda row: row["score"], reverse=True)

    def update_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> None:
        self._items[memory.memory_id] = memory
        self._texts[memory.memory_id] = projection.embedding_text

    def delete_projection(self, memory: MemoryRecord) -> None:
        self._items.pop(memory.memory_id, None)
        self._texts.pop(memory.memory_id, None)


class InMemoryEntityGraphProjectionRepository(EntityGraphProjectionRepository):
    def __init__(self) -> None:
        self._links: dict[str, set[str]] = {}
        self._items: dict[str, str] = {}
        self._anchors: dict[str, set[str]] = {}
        self._registry = MemoryGraphProjectionRegistry()

    def upsert(self, memory: MemoryRecord) -> None:
        self.delete_projection(memory.memory_id)
        anchors = {relation.anchor_key for relation in self._registry.relations(memory)}
        for other_id, other_anchors in list(self._anchors.items()):
            if other_id != memory.memory_id and anchors & other_anchors:
                self._links.setdefault(memory.memory_id, set()).add(other_id)
                self._links.setdefault(other_id, set()).add(memory.memory_id)
        self._items[memory.memory_id] = memory.memory_type.value
        self._anchors[memory.memory_id] = anchors

    def expand_candidates(
        self,
        request: MemoryRecallRequest,
        seed_memory_ids: list[str],
        *,
        query_profile: MemoryQueryProfile | None = None,
    ) -> list[str]:
        expanded: set[str] = set()
        for memory_id in seed_memory_ids:
            expanded.update(self._links.get(memory_id, set()))
        profile = query_profile or MemoryQueryBuilder().build(request)
        query_anchors = set(profile.graph_anchor_keys)
        if query_anchors:
            for memory_id, anchors in self._anchors.items():
                if anchors & query_anchors:
                    expanded.add(memory_id)
        return sorted(expanded)

    def delete_projection(self, memory_id: str) -> None:
        for linked_id in self._links.pop(memory_id, set()):
            self._links.get(linked_id, set()).discard(memory_id)
        self._items.pop(memory_id, None)
        self._anchors.pop(memory_id, None)
        self._links.pop(memory_id, None)
        for values in self._links.values():
            values.discard(memory_id)
