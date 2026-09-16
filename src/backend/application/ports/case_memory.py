"""Ports for the CaseMemoryService anti-corruption boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from ...domain.case_memory.entities import (
    MemoryQueryProfile,
    MemoryRecord,
    MemoryRecallRequest,
    MemoryVectorProjection,
)


class MemoProjectionError(RuntimeError):
    """A governed projection write failed and must be retried by the Outbox."""


class MemoryRepository(Protocol):
    def save(self, memory: MemoryRecord) -> MemoryRecord: ...
    def get(self, memory_id: str) -> MemoryRecord | None: ...
    def get_many(self, memory_ids: Iterable[str]) -> list[MemoryRecord]: ...
    def list(self, *, status: str | None = None, scope_type: str | None = None, scope_id: str | None = None, memory_type: str | None = None, limit: int | None = None) -> list[MemoryRecord]: ...
    def update(self, memory: MemoryRecord) -> MemoryRecord: ...


class MemoryEventRepository(Protocol):
    def append(self, memory_id: str, event_type: str, actor_id: str | None, payload: dict[str, Any]) -> dict[str, Any]: ...
    def list_for_memory(self, memory_id: str) -> list[dict[str, Any]]: ...


class MemoryOutboxRepository(Protocol):
    def enqueue(self, event_type: str, aggregate_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    def claim_next_batch(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def mark_processed(self, event_id: str) -> None: ...
    def mark_failed(self, event_id: str, error: str) -> None: ...
    def status(self) -> dict[str, int]: ...


class MemoMemoryPort(Protocol):
    """MediGuard-neutral Mem0 port. No Mem0 types cross this interface."""

    def add_active_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> str | None: ...
    def search(
        self,
        request: MemoryRecallRequest,
        *,
        query_text: str,
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]: ...
    def update_projection(
        self,
        memory: MemoryRecord,
        projection: MemoryVectorProjection,
    ) -> None: ...
    def delete_projection(self, memory: MemoryRecord) -> None: ...


class EntityGraphProjectionRepository(Protocol):
    def upsert(self, memory: MemoryRecord) -> None: ...
    def expand_candidates(
        self,
        request: MemoryRecallRequest,
        seed_memory_ids: Iterable[str],
        *,
        query_profile: MemoryQueryProfile | None = None,
    ) -> list[str]: ...
    def delete_projection(self, memory_id: str) -> None: ...
