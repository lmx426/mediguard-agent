"""接入与审计 ORM 模型。

包含 CSV 批量接入批次/明细、案件分配、统计材料文档、申诉材料、人工操作审计日志等 ORM 映射。
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from . import Base, utc_now


class AuditActionORM(Base):
    """人工操作审计日志。"""

    __tablename__ = "audit_actions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID | None] = mapped_column(ForeignKey("cases.id", ondelete="SET NULL"), index=True)
    actor_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    action_type: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class IngestBatchORM(Base):
    """CSV 批量接入批次。"""

    __tablename__ = "ingest_batches"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    batch_no: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(240))
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    total_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class IngestBatchItemORM(Base):
    """批次内每行记录处理结果。"""

    __tablename__ = "ingest_batch_items"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("ingest_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    case_id: Mapped[UUID | None] = mapped_column(ForeignKey("cases.id", ondelete="SET NULL"))
    fingerprint: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    source_record: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseAssignmentORM(Base):
    """案件分配记录，支持初审/复审责任链。"""

    __tablename__ = "case_assignments"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    assignee_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    assignment_role: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StatisticalMaterialDocumentORM(Base):
    """统计材料视图快照，由 81 字段派生。"""

    __tablename__ = "statistical_material_documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    document_type: Mapped[str] = mapped_column(String(80), nullable=False)
    visit_date: Mapped[str | None] = mapped_column(String(40))
    institution: Mapped[str | None] = mapped_column(String(120))
    department: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    check_points: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseMaterialSnapshotORM(Base):
    """单案件业务材料视图快照。"""

    __tablename__ = "case_material_snapshots"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    input_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    template_version: Mapped[str] = mapped_column(String(80), nullable=False)
    scenario_code: Mapped[str] = mapped_column(String(80), nullable=False)
    scenario_label: Mapped[str] = mapped_column(String(120), nullable=False)
    evidence_tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    subject_profile: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    materials_payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    generation_status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class CaseMaterialAssetORM(Base):
    """业务材料 PNG 资产元数据。"""

    __tablename__ = "case_material_assets"
    __table_args__ = (
        UniqueConstraint(
            "material_set_id",
            "material_id",
            "asset_id",
            name="uq_case_material_asset_identity",
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    material_set_id: Mapped[UUID] = mapped_column(
        ForeignKey("case_material_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    material_id: Mapped[str] = mapped_column(String(100), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(80), nullable=False)
    file_type: Mapped[str] = mapped_column(String(80), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    display_name: Mapped[str] = mapped_column(String(180), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AppealMaterialORM(Base):
    """申诉材料登记元数据，不存文件内容。"""

    __tablename__ = "appeal_materials"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    registered_by_user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    material_type: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(80), nullable=False)
    remark: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(180))
    file_size: Mapped[int | None] = mapped_column(BigInteger)
    file_type: Mapped[str | None] = mapped_column(String(120))
    file_last_modified: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
