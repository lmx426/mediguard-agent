"""SQL repositories for the CaseMemoryService truth ledger and projections."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .....application.ports.case_memory import (
    EntityGraphProjectionRepository,
    MemoMemoryPort,
    MemoryEventRepository,
    MemoryOutboxRepository,
    MemoryRepository,
)
from .....application.case_memory.retrieval import (
    MemoryGraphProjectionRegistry,
    MemoryQueryBuilder,
)
from .....domain.case_memory.entities import (
    MemoryQueryProfile,
    MemoryRecord,
    MemoryRecallRequest,
)
from ..models import (
    CaseMemoryIndexORM,
    CaseMemoryORM,
    MemoryEventORM,
    MemoryGraphEdgeORM,
    MemoryGraphNodeORM,
    MemoryOutboxORM,
)
from ..session import session_scope


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _to_memory(row: CaseMemoryORM) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row.memory_id,
        memory_type=row.memory_type,
        memory_level=row.memory_level,
        scope={"scope_type": row.scope_type, "scope_id": row.scope_id},
        status=row.status,
        payload=row.payload,
        allowed_consumers=row.allowed_consumers,
        consumer_view_policy_id=row.consumer_view_policy_id,
        admission_policy=row.admission_policy,
        admission_confidence=row.admission_confidence,
        shadow_observation_count=row.shadow_observation_count,
        importance_score=row.importance_score,
        freshness_score=row.freshness_score,
        usage_count=row.usage_count,
        success_count=row.success_count,
        conflict_count=row.conflict_count,
        created_at=row.created_at,
        last_verified_at=row.last_verified_at,
        last_used_at=row.last_used_at,
        review_after=row.review_after,
        expires_at=row.expires_at,
        snoozed_until=row.snoozed_until,
        memo_memory_id=row.memo_memory_id,
        projection_sync_status=row.projection_sync_status,
        supersedes_id=row.supersedes_id,
        conflict_with_ids=row.conflict_with_ids,
        source_event_ids=row.source_event_ids,
    )


class SqlMemoryRepository(MemoryRepository):
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def save(self, memory: MemoryRecord) -> MemoryRecord:
        with session_scope(self._session_factory) as session:
            row = CaseMemoryORM(
                memory_id=memory.memory_id,
                memory_type=memory.memory_type.value,
                memory_level=memory.memory_level.value,
                scope_type=memory.scope.scope_type,
                scope_id=memory.scope.scope_id,
                status=memory.status.value,
                payload=memory.payload,
                allowed_consumers=memory.allowed_consumers,
                consumer_view_policy_id=memory.consumer_view_policy_id,
                admission_policy=memory.admission_policy,
                admission_confidence=memory.admission_confidence,
                shadow_observation_count=memory.shadow_observation_count,
                importance_score=memory.importance_score,
                freshness_score=memory.freshness_score,
                usage_count=memory.usage_count,
                success_count=memory.success_count,
                conflict_count=memory.conflict_count,
                created_at=memory.created_at,
                last_verified_at=memory.last_verified_at,
                last_used_at=memory.last_used_at,
                review_after=memory.review_after,
                expires_at=memory.expires_at,
                snoozed_until=memory.snoozed_until,
                memo_memory_id=memory.memo_memory_id,
                projection_sync_status=memory.projection_sync_status,
                supersedes_id=memory.supersedes_id,
                conflict_with_ids=memory.conflict_with_ids,
                source_event_ids=memory.source_event_ids,
            )
            session.add(row)
            session.add(CaseMemoryIndexORM(memory_id=memory.memory_id, search_text=f"{memory.summary} {memory.payload}"))
        return memory

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._session_factory() as session:
            row = session.scalar(select(CaseMemoryORM).where(CaseMemoryORM.memory_id == memory_id))
            return _to_memory(row) if row else None

    def get_many(self, memory_ids: Iterable[str]) -> list[MemoryRecord]:
        ids = list(dict.fromkeys(str(memory_id) for memory_id in memory_ids if memory_id))
        if not ids:
            return []
        with self._session_factory() as session:
            rows = session.scalars(
                select(CaseMemoryORM).where(CaseMemoryORM.memory_id.in_(ids))
            ).all()
            by_id = {row.memory_id: _to_memory(row) for row in rows}
            return [by_id[memory_id] for memory_id in ids if memory_id in by_id]

    def list(self, *, status: str | None = None, scope_type: str | None = None, scope_id: str | None = None, memory_type: str | None = None, limit: int | None = None) -> list[MemoryRecord]:
        with self._session_factory() as session:
            statement = select(CaseMemoryORM).order_by(CaseMemoryORM.created_at.desc())
            if status is not None:
                statement = statement.where(CaseMemoryORM.status == status)
            if scope_type is not None:
                statement = statement.where(CaseMemoryORM.scope_type == scope_type)
            if scope_id is not None:
                statement = statement.where(CaseMemoryORM.scope_id == scope_id)
            if memory_type is not None:
                statement = statement.where(CaseMemoryORM.memory_type == memory_type)
            if limit is not None:
                statement = statement.limit(max(1, limit))
            return [_to_memory(row) for row in session.scalars(statement).all()]

    def update(self, memory: MemoryRecord) -> MemoryRecord:
        with session_scope(self._session_factory) as session:
            row = session.scalar(select(CaseMemoryORM).where(CaseMemoryORM.memory_id == memory.memory_id).with_for_update())
            if row is None:
                raise KeyError(memory.memory_id)
            for field, value in {
                "status": memory.status.value,
                "payload": memory.payload,
                "allowed_consumers": memory.allowed_consumers,
                "consumer_view_policy_id": memory.consumer_view_policy_id,
                "admission_confidence": memory.admission_confidence,
                "shadow_observation_count": memory.shadow_observation_count,
                "importance_score": memory.importance_score,
                "freshness_score": memory.freshness_score,
                "usage_count": memory.usage_count,
                "success_count": memory.success_count,
                "conflict_count": memory.conflict_count,
                "last_verified_at": memory.last_verified_at,
                "last_used_at": memory.last_used_at,
                "review_after": memory.review_after,
                "expires_at": memory.expires_at,
                "snoozed_until": memory.snoozed_until,
                "memo_memory_id": memory.memo_memory_id,
                "projection_sync_status": memory.projection_sync_status,
                "supersedes_id": memory.supersedes_id,
                "conflict_with_ids": memory.conflict_with_ids,
                "source_event_ids": memory.source_event_ids,
            }.items():
                setattr(row, field, value)
            index_row = session.get(CaseMemoryIndexORM, memory.memory_id)
            if index_row is not None:
                index_row.search_text = f"{memory.summary} {memory.payload}"
                index_row.last_indexed_at = datetime.now(timezone.utc)
        return memory


class SqlMemoryEventRepository(MemoryEventRepository):
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def append(self, memory_id: str, event_type: str, actor_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        with session_scope(self._session_factory) as session:
            row = MemoryEventORM(memory_id=memory_id, event_type=event_type, actor_id=actor_id, payload=payload)
            session.add(row)
            session.flush()
            return {"event_id": str(row.id), "memory_id": memory_id, "event_type": event_type, "actor_id": actor_id, "payload": payload, "created_at": row.created_at}

    def list_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.scalars(select(MemoryEventORM).where(MemoryEventORM.memory_id == memory_id).order_by(MemoryEventORM.created_at.asc())).all()
            return [{"event_id": str(row.id), "event_type": row.event_type, "actor_id": row.actor_id, "payload": row.payload, "created_at": row.created_at} for row in rows]


class SqlMemoryOutboxRepository(MemoryOutboxRepository):
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def enqueue(self, event_type: str, aggregate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        idempotency_key = f"{event_type}:{aggregate_id}:{payload_hash}"
        with session_scope(self._session_factory) as session:
            existing = session.scalar(select(MemoryOutboxORM).where(MemoryOutboxORM.idempotency_key == idempotency_key))
            if existing is not None:
                return {"event_id": str(existing.id), "status": existing.status}
            row = MemoryOutboxORM(event_type=event_type, aggregate_id=aggregate_id, idempotency_key=idempotency_key, payload=payload)
            session.add(row)
            session.flush()
            return {"event_id": str(row.id), "status": row.status}

    def claim_next_batch(self, limit: int = 50) -> list[dict[str, Any]]:
        with session_scope(self._session_factory) as session:
            now = datetime.now(timezone.utc)
            rows = session.scalars(
                select(MemoryOutboxORM)
                .where(
                    MemoryOutboxORM.status.in_(["pending", "retrying", "processing"]),
                    MemoryOutboxORM.next_attempt_at <= now,
                    or_(MemoryOutboxORM.locked_until.is_(None), MemoryOutboxORM.locked_until <= now),
                )
                .order_by(MemoryOutboxORM.created_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
            for row in rows:
                row.status = "processing"
                row.locked_until = now + timedelta(minutes=5)
            return [{"event_id": str(row.id), "event_type": row.event_type, "aggregate_id": row.aggregate_id, "payload": row.payload, "status": row.status} for row in rows]

    def mark_processed(self, event_id: str) -> None:
        with session_scope(self._session_factory) as session:
            row = session.get(MemoryOutboxORM, event_id)
            if row:
                row.status = "processed"
                row.processed_at = datetime.now(timezone.utc)
                row.locked_until = None

    def mark_failed(self, event_id: str, error: str) -> None:
        with session_scope(self._session_factory) as session:
            row = session.get(MemoryOutboxORM, event_id)
            if row:
                row.retry_count += 1
                row.last_error = error[:1000]
                row.locked_until = None
                if row.retry_count >= 5:
                    row.status = "dead_letter"
                else:
                    row.status = "retrying"
                    row.next_attempt_at = datetime.now(timezone.utc) + timedelta(minutes=min(30, 2 ** row.retry_count))

    def status(self) -> dict[str, int]:
        with self._session_factory() as session:
            rows = session.execute(
                select(MemoryOutboxORM.status, func.count(MemoryOutboxORM.id)).group_by(
                    MemoryOutboxORM.status
                )
            ).all()
            return {str(status): int(count) for status, count in rows}


class SqlEntityGraphProjectionRepository(EntityGraphProjectionRepository):
    """Rule-owned graph projection; Mem0 remains behind its own port."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory
        self._registry = MemoryGraphProjectionRegistry()

    def _node(self, session: Session, node_type: str, key: str, label: str, attrs: dict[str, Any] | None = None) -> MemoryGraphNodeORM:
        key_hash = _hash(f"{node_type}:{key}")
        row = session.scalar(select(MemoryGraphNodeORM).where(MemoryGraphNodeORM.node_type == node_type, MemoryGraphNodeORM.node_key_hash == key_hash))
        if row is None:
            row = MemoryGraphNodeORM(node_type=node_type, node_key_hash=key_hash, display_label=label[:240], safe_attrs=attrs or {})
            session.add(row)
            session.flush()
        else:
            row.display_label = label[:240]
            if attrs is not None:
                row.safe_attrs = attrs
        return row

    def _edge(self, session: Session, source: MemoryGraphNodeORM, target: MemoryGraphNodeORM, edge_type: str, memory_id: str | None, refs: list[str] | None = None) -> None:
        row = session.scalar(select(MemoryGraphEdgeORM).where(MemoryGraphEdgeORM.source_node_id == source.id, MemoryGraphEdgeORM.target_node_id == target.id, MemoryGraphEdgeORM.edge_type == edge_type))
        if row is None:
            session.add(MemoryGraphEdgeORM(source_node_id=source.id, target_node_id=target.id, edge_type=edge_type, memory_id=memory_id, supporting_refs=refs or [], source="rule", created_by_rule_or_llm="rule"))
        else:
            row.status = "active"
            row.memory_id = memory_id
            row.supporting_refs = refs or row.supporting_refs or []
            row.last_verified_at = datetime.now(timezone.utc)

    def upsert(self, memory: MemoryRecord) -> None:
        with session_scope(self._session_factory) as session:
            session.query(MemoryGraphEdgeORM).filter(
                MemoryGraphEdgeORM.memory_id == memory.memory_id
            ).update({"status": "tombstoned"})
            if memory.memory_level.value == "L0_source_event":
                source = self._node(session, "source_event_anchor", memory.memory_id, "脱敏来源锚点")
                for source_ref in memory.source_event_ids or memory.payload.get("source_ref_ids", []):
                    target = self._node(session, "source_event_anchor", str(source_ref), "脱敏来源锚点")
                    self._edge(session, source, target, "DERIVED_FROM", memory.memory_id, [str(source_ref)])
                return
            memory_node = self._node(
                session,
                "memory",
                memory.memory_id,
                memory.summary,
                {
                    "memory_type": memory.memory_type.value,
                    "memory_level": memory.memory_level.value,
                    "allowed_consumers": list(memory.allowed_consumers),
                },
            )
            self._connect_payload(session, memory_node, memory)
            if memory.supersedes_id:
                self._edge(
                    session,
                    memory_node,
                    self._node(session, "memory", memory.supersedes_id, memory.supersedes_id),
                    "SUPERSEDES",
                    memory.memory_id,
                    [memory.supersedes_id],
                )
            for conflict_id in memory.conflict_with_ids:
                self._edge(
                    session,
                    memory_node,
                    self._node(session, "memory", str(conflict_id), str(conflict_id)),
                    "CONFLICTS_WITH",
                    memory.memory_id,
                    [str(conflict_id)],
                )

    def _connect_payload(self, session: Session, memory_node: MemoryGraphNodeORM, memory: MemoryRecord) -> None:
        payload = memory.payload
        for relation in self._registry.relations(memory):
            self._edge(
                session,
                memory_node,
                self._node(
                    session,
                    relation.node_type,
                    relation.node_key,
                    relation.label,
                ),
                relation.edge_type,
                memory.memory_id,
            )

        if payload.get("supporting_l1_refs"):
            refs = payload["supporting_l1_refs"]
            ref_edge = "SUMMARIZES"
            ref_node_type = "memory"
        elif payload.get("supporting_l2_refs"):
            refs = payload["supporting_l2_refs"]
            ref_edge = "STABILIZED_FROM"
            ref_node_type = "memory"
        elif payload.get("derived_from_l0_refs"):
            refs = payload["derived_from_l0_refs"]
            ref_edge = "DERIVED_FROM"
            ref_node_type = "source_event_anchor"
        else:
            refs = payload.get("source_refs") or memory.source_event_ids or []
            ref_edge = "DERIVED_FROM"
            ref_node_type = "source_event_anchor"
        for ref in refs:
            self._edge(session, memory_node, self._node(session, ref_node_type, str(ref), str(ref)), ref_edge, memory.memory_id, [str(ref)])

    def expand_candidates(
        self,
        request: MemoryRecallRequest,
        seed_memory_ids: Iterable[str],
        *,
        query_profile: MemoryQueryProfile | None = None,
    ) -> list[str]:
        seeds = list(seed_memory_ids)
        profile = query_profile or MemoryQueryBuilder().build(request)
        node_hashes = [
            *[_hash(f"memory:{item}") for item in seeds],
            *[_hash(anchor) for anchor in profile.graph_anchor_keys],
        ]
        if not node_hashes:
            return []
        with self._session_factory() as session:
            seed_nodes = session.scalars(
                select(MemoryGraphNodeORM)
                .where(MemoryGraphNodeORM.node_key_hash.in_(node_hashes))
                .limit(100)
            ).all()
            seed_ids = [node.id for node in seed_nodes]
            if not seed_ids:
                return []
            adjacent_edges = session.scalars(
                select(MemoryGraphEdgeORM).where(
                    or_(MemoryGraphEdgeORM.source_node_id.in_(seed_ids), MemoryGraphEdgeORM.target_node_id.in_(seed_ids)),
                    MemoryGraphEdgeORM.status == "active",
                    MemoryGraphEdgeORM.edge_type.in_(self._registry.EXPANDABLE_EDGE_TYPES),
                ).limit(300)
            ).all()
            related_memory_ids = {
                str(edge.memory_id) for edge in adjacent_edges if edge.memory_id
            }
            neighbor_ids = [
                edge.target_node_id if edge.source_node_id in seed_ids else edge.source_node_id
                for edge in adjacent_edges
            ]
            if not neighbor_ids:
                return list(related_memory_ids)
            related_edges = session.scalars(
                select(MemoryGraphEdgeORM).where(
                    or_(MemoryGraphEdgeORM.source_node_id.in_(neighbor_ids), MemoryGraphEdgeORM.target_node_id.in_(neighbor_ids)),
                    MemoryGraphEdgeORM.status == "active",
                    MemoryGraphEdgeORM.edge_type.in_(self._registry.EXPANDABLE_EDGE_TYPES),
                    MemoryGraphEdgeORM.memory_id.is_not(None),
                ).limit(500)
            ).all()
            related_memory_ids.update(
                str(edge.memory_id) for edge in related_edges if edge.memory_id
            )
            return sorted(related_memory_ids)

    def delete_projection(self, memory_id: str) -> None:
        with session_scope(self._session_factory) as session:
            session.query(MemoryGraphEdgeORM).filter(MemoryGraphEdgeORM.memory_id == memory_id).update({"status": "tombstoned"})
