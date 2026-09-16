"""Evidence Agent ORM 模型。

包含 Agent 运行记录、事件、检查点、工具调用、分析结果、证据引用等 ORM 映射。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from . import Base, TimestampMixin, utc_now


class AgentRunORM(Base, TimestampMixin):
    """One formal Evidence Agent execution."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_agent_runs_idempotency_key"),
        Index("ix_agent_runs_case_status", "case_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[str] = mapped_column(String(60), unique=True, index=True, nullable=False)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    actor_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(20), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    current_node: Mapped[str] = mapped_column(String(60), default="queued", nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(80), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentEventORM(Base):
    """Replayable, redacted Agent business event."""

    __tablename__ = "agent_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_agent_events_run_sequence"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentCheckpointORM(Base):
    """Redacted resumable state after a graph node."""

    __tablename__ = "agent_checkpoints"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_agent_checkpoints_run_sequence"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(String(60), nullable=False)
    state_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    safe_to_resume: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentToolCallORM(Base):
    """Validated read-only Function Calling trace."""

    __tablename__ = "agent_tool_calls"
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id", name="uq_agent_tool_calls_run_call"),
        UniqueConstraint("run_id", "signature", name="uq_agent_tool_calls_run_signature"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(120), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
    signature: Mapped[str] = mapped_column(String(64), nullable=False)
    arguments: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict | None] = mapped_column(JSONB)
    result_hash: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentEvidenceAnalysisORM(Base):
    """Immutable validated Agent output with current-version projection."""

    __tablename__ = "agent_evidence_analyses"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_agent_evidence_analyses_run"),
        Index("ix_agent_analyses_case_current", "case_id", "is_current"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id: Mapped[str] = mapped_column(String(60), unique=True, index=True, nullable=False)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    generated_notice: Mapped[str] = mapped_column(String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(80), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AgentEvidenceCitationORM(Base):
    """Normalized citations attached to an immutable analysis."""

    __tablename__ = "agent_evidence_citations"
    __table_args__ = (UniqueConstraint("analysis_id", "citation_id", name="uq_agent_citations_analysis_citation"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(ForeignKey("agent_evidence_analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    citation_id: Mapped[str] = mapped_column(String(160), nullable=False)
    source_type: Mapped[str] = mapped_column(String(60), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(180), nullable=False)
    label: Mapped[str] = mapped_column(String(240), nullable=False)
    version: Mapped[str | None] = mapped_column(String(120))
    current_value: Mapped[str | None] = mapped_column(Text)
    threshold: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
