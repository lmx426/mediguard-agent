"""SQLAlchemy ORM mapping for Caser context sections."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from . import Base, TimestampMixin, utc_now


class CaserContextSectionORM(Base, TimestampMixin):
    """One Caser read-model section for one case."""

    __tablename__ = "caser_context_sections"
    __table_args__ = (
        UniqueConstraint("case_id", "section_key", name="uq_caser_context_case_section"),
        Index("ix_caser_context_case_status", "case_id", "status"),
        Index("ix_caser_context_section_key", "section_key"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_key: Mapped[str] = mapped_column(String(80), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    completeness: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    source_refs: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_versions: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    generated_by: Mapped[str] = mapped_column(String(120), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
