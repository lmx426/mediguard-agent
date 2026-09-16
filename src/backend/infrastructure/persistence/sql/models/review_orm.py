"""审核相关 ORM 模型。

包含人工初审记录、初审附件、审核工作笔记、Trace 留痕节点等 ORM 映射。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base, utc_now


class WorkflowTraceNodeORM(Base):
    """Trace 与九阶段流程留痕。"""

    __tablename__ = "workflow_trace_nodes"
    __table_args__ = (UniqueConstraint("case_id", "node_id", name="uq_workflow_trace_case_node"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(String(120), nullable=False)
    node_name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    node_order: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ReviewORM(Base):
    """人工初审记录。"""

    __tablename__ = "reviews"
    __table_args__ = (UniqueConstraint("case_id", name="uq_reviews_case"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    reviewer_name: Mapped[str] = mapped_column(String(120), nullable=False)
    decision: Mapped[str] = mapped_column(String(160), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    attachments: Mapped[list[ReviewAttachmentORM]] = relationship(
        back_populates="review",
        cascade="all, delete-orphan",
    )


class ReviewAttachmentORM(Base):
    """初审材料登记元数据，不存文件内容。"""

    __tablename__ = "review_attachments"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    review_id: Mapped[UUID] = mapped_column(ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    material_type: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(80), nullable=False)
    remark: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(180))
    file_size: Mapped[int | None] = mapped_column(BigInteger)
    file_type: Mapped[str | None] = mapped_column(String(120))
    file_last_modified: Mapped[str | None] = mapped_column(String(80))

    review: Mapped[ReviewORM] = relationship(back_populates="attachments")


class AuditNoteORM(Base):
    """审核工作笔记。"""

    __tablename__ = "audit_notes"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    note_id: Mapped[str] = mapped_column(String(60), unique=True, index=True, nullable=False)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    author_name: Mapped[str] = mapped_column(String(120), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    materials: Mapped[list[dict]] = mapped_column(JSONB, default=list, nullable=False)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    voided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    voided_by_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    voided_by_name: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
