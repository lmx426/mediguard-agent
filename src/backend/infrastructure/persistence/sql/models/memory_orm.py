"""PostgreSQL truth-ledger and projection tables for case memories."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from . import Base, TimestampMixin, utc_now

# The direct fallback repository is not active yet, so this reserved table stays
# JSONB to match the current Alembic schema. Mem0 uses pgvector in its own store.
MemoryVectorType = JSONB


class CaseMemoryORM(Base, TimestampMixin):
    """MediGuard truth ledger. Only active rows are projected to Mem0."""

    __tablename__ = "case_memories"
    __table_args__ = (
        Index("ix_case_memories_recall", "status", "scope_type", "scope_id", "memory_type", "memory_level"),
        Index("ix_case_memories_freshness", "expires_at", "review_after"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    memory_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(60), nullable=False)
    memory_level: Mapped[str] = mapped_column(String(60), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="candidate")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    allowed_consumers: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    consumer_view_policy_id: Mapped[str] = mapped_column(String(100), default="default", nullable=False)
    admission_policy: Mapped[str] = mapped_column(String(30), default="human_review", nullable=False)
    admission_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    shadow_observation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    freshness_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    conflict_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    memo_memory_id: Mapped[str | None] = mapped_column(String(120))
    projection_sync_status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    supersedes_id: Mapped[str | None] = mapped_column(String(80))
    conflict_with_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_event_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)


class MemoryEventORM(Base):
    __tablename__ = "memory_events"
    __table_args__ = (Index("ix_memory_events_memory_created", "memory_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    memory_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(40), default="case_memory", nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(80))
    source_run_id: Mapped[str | None] = mapped_column(String(120))
    source_event_id: Mapped[str | None] = mapped_column(String(120))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class MemoryOutboxORM(Base):
    __tablename__ = "memory_outbox"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_memory_outbox_idempotency"),
        Index("ix_memory_outbox_status_next", "status", "next_attempt_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(40), default="case_memory", nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(80), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class MemoryEmbeddingORM(Base):
    __tablename__ = "memory_embeddings"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    memory_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    embedding_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(160), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_vector: Mapped[list[float] | None] = mapped_column(MemoryVectorType)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseMemoryIndexORM(Base):
    __tablename__ = "case_memory_index"

    memory_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[str | None] = mapped_column(Text)
    last_indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class MemoryGraphNodeORM(Base, TimestampMixin):
    __tablename__ = "memory_graph_nodes"
    __table_args__ = (UniqueConstraint("node_type", "node_key_hash", name="uq_memory_graph_node_key"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_type: Mapped[str] = mapped_column(String(60), nullable=False)
    node_key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    display_label: Mapped[str] = mapped_column(String(240), nullable=False)
    safe_attrs: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class MemoryGraphEdgeORM(Base, TimestampMixin):
    __tablename__ = "memory_graph_edges"
    __table_args__ = (
        UniqueConstraint("source_node_id", "target_node_id", "edge_type", name="uq_memory_graph_edge"),
        Index("ix_memory_graph_edges_memory", "memory_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_node_id: Mapped[UUID] = mapped_column(ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"), nullable=False)
    target_node_id: Mapped[UUID] = mapped_column(ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"), nullable=False)
    edge_type: Mapped[str] = mapped_column(String(60), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source: Mapped[str] = mapped_column(String(60), default="rule", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="active", nullable=False)
    supporting_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_by_rule_or_llm: Mapped[str] = mapped_column(String(60), default="rule", nullable=False)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
