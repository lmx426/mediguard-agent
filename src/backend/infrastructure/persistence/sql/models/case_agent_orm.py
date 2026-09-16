"""Case Agent ORM models.

These tables store lightweight assistant working memory and audit traces for
single-case sessions. They deliberately do not store raw files, RES, model
reasoning, or formal review decisions.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from . import Base, TimestampMixin, utc_now


class CaseAgentSessionORM(Base, TimestampMixin):
    """A user-owned Case Agent session bound to one case."""

    __tablename__ = "case_agent_sessions"
    __table_args__ = (
        Index("ix_case_agent_sessions_case_actor", "case_id", "actor_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    session_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    current_topic: Mapped[str | None] = mapped_column(String(240))
    pending_tool: Mapped[str | None] = mapped_column(String(120))
    referenced_source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    task_state: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    last_read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)


class CaseAgentMessageORM(Base):
    """A persisted user or assistant message."""

    __tablename__ = "case_agent_messages"
    __table_args__ = (
        Index("ix_case_agent_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_sessions.id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    active_stage: Mapped[str | None] = mapped_column(String(80))
    answer_payload: Mapped[dict | None] = mapped_column(JSONB)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseAgentRunORM(Base, TimestampMixin):
    """One background assistant-turn execution."""

    __tablename__ = "case_agent_runs"
    __table_args__ = (
        Index("ix_case_agent_runs_session_status", "session_id", "status"),
        Index(
            "uq_case_agent_runs_one_active_per_session",
            "session_id",
            unique=True,
            postgresql_where=text("status IN ('created', 'running', 'resuming')"),
        ),
        Index(
            "uq_case_agent_runs_session_client_request",
            "session_id",
            "client_request_id",
            unique=True,
            postgresql_where=text("client_request_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_sessions.id", ondelete="CASCADE"), nullable=False)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    user_message_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_messages.id"), nullable=False)
    assistant_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("case_agent_messages.id"))
    parent_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="SET NULL"))
    resumed_by_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(30), default="created", nullable=False)
    current_node: Mapped[str] = mapped_column(String(80), default="created", nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    degraded_reason: Mapped[str | None] = mapped_column(Text)
    pending_clarification: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    resume_context: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    timeout_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_reason: Mapped[str | None] = mapped_column(String(80))
    client_request_id: Mapped[str | None] = mapped_column(String(120))
    case_context_fingerprint: Mapped[str | None] = mapped_column(String(80))
    lease_owner: Mapped[str | None] = mapped_column(String(120))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CaseAgentEventORM(Base):
    """Replayable Case Agent business event."""

    __tablename__ = "case_agent_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_case_agent_events_run_sequence"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseAgentToolCallORM(Base):
    """Auditable Tool Calling trace for Case Agent."""

    __tablename__ = "case_agent_tool_calls"
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id", name="uq_case_agent_tool_calls_run_call"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(120), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    arguments: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(160), index=True)
    response_ref: Mapped[str | None] = mapped_column(String(200))
    result_summary: Mapped[dict | None] = mapped_column(JSONB)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseAgentNodeExecutionORM(Base):
    """Durable execution record for one graph node."""

    __tablename__ = "case_agent_node_executions"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_case_agent_node_executions_run_sequence"),
        Index("ix_case_agent_node_executions_run_node", "run_id", "node_name"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    input_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    output_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class CaseAgentCheckpointORM(Base):
    """Safe persisted state snapshot after important Case Agent nodes."""

    __tablename__ = "case_agent_checkpoints"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_case_agent_checkpoints_run_sequence"),
        Index("ix_case_agent_checkpoints_run_node", "run_id", "node_name"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("case_agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(String(80), nullable=False)
    state_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    safe_to_resume: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
