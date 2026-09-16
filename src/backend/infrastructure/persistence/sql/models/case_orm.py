"""案件相关 ORM 模型。

包含案件主表及关联的源记录、指纹、规则结果、模型结果、
PCA 特征分、风险分项、证据包等 ORM 映射。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base, TimestampMixin, utc_now


class CaseORM(Base, TimestampMixin):
    """稽核案件主表，对应当前 CaseDetail / CaseSummary。"""

    __tablename__ = "cases"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[str] = mapped_column(String(40), unique=True, index=True, nullable=False)
    case_title: Mapped[str] = mapped_column(String(240), nullable=False)
    case_type: Mapped[str] = mapped_column(String(120), nullable=False)
    claim_summary: Mapped[str] = mapped_column(Text, nullable=False)
    subject_ref: Mapped[str | None] = mapped_column(String(80))
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    review_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    review_priority: Mapped[str] = mapped_column(String(40), nullable=False)
    evidence_consistency: Mapped[str] = mapped_column(Text, nullable=False)
    model_signal_source: Mapped[str] = mapped_column(Text, nullable=False)
    model_evidence_ref: Mapped[str] = mapped_column(String(160), nullable=False)
    model_signal_reasons: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    rule_pool_version: Mapped[str] = mapped_column(String(80), nullable=False)
    rule_baseline_version: Mapped[str] = mapped_column(String(80), nullable=False)
    expected_recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    claim_amount: Mapped[float | None] = mapped_column(Float)
    assigned_auditor_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    risk_breakdown_total_score: Mapped[int | None] = mapped_column(Integer)
    risk_breakdown_max_score: Mapped[int | None] = mapped_column(Integer)
    risk_breakdown_display_score: Mapped[str | None] = mapped_column(String(40))
    risk_breakdown_level: Mapped[str | None] = mapped_column(String(20))
    risk_breakdown_level_label: Mapped[str | None] = mapped_column(String(40))
    risk_breakdown_cap_note: Mapped[str | None] = mapped_column(Text)

    source_records: Mapped[list[CaseSourceRecordORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    fingerprints: Mapped[list[CaseFingerprintORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    rule_results: Mapped[list[RuleResultORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    fraud_model_results: Mapped[list[FraudModelResultORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    pca_feature_scores: Mapped[list[PCAFeatureScoreORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    risk_score_components: Mapped[list[RiskScoreComponentORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    evidence_packages: Mapped[list[EvidencePackageORM]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )


class CaseSourceRecordORM(Base):
    """完整 81 字段脱敏宽表快照。"""

    __tablename__ = "case_source_records"
    __table_args__ = (UniqueConstraint("case_id", "record_version", name="uq_case_source_record_version"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    source_system: Mapped[str] = mapped_column(String(120), nullable=False)
    record_version: Mapped[str] = mapped_column(String(80), nullable=False)
    source_record: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    input_features: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="source_records")


class CaseFingerprintORM(Base):
    """接入幂等指纹，防止重复建案。"""

    __tablename__ = "case_fingerprints"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    fingerprint: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    source_system: Mapped[str] = mapped_column(String(120), nullable=False)
    record_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="fingerprints")


class RuleResultORM(Base):
    """规则核验结果。"""

    __tablename__ = "rule_results"
    __table_args__ = (UniqueConstraint("case_id", "rule_id", name="uq_rule_results_case_rule"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_id: Mapped[str] = mapped_column(String(40), nullable=False)
    rule_name: Mapped[str] = mapped_column(String(160), nullable=False)
    hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    layer: Mapped[str] = mapped_column(String(60), nullable=False)
    action: Mapped[str] = mapped_column(String(60), nullable=False)
    current_value: Mapped[str] = mapped_column(Text, default="", nullable=False)
    threshold: Mapped[str] = mapped_column(Text, default="", nullable=False)
    business_explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="rule_results")
    check_items: Mapped[list[RuleCheckItemORM]] = relationship(
        back_populates="rule_result",
        cascade="all, delete-orphan",
    )


class RuleCheckItemORM(Base):
    """规则内部核验项。"""

    __tablename__ = "rule_check_items"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    rule_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("rule_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label: Mapped[str] = mapped_column(String(160), nullable=False)
    current_value: Mapped[str] = mapped_column(Text, nullable=False)
    threshold: Mapped[str] = mapped_column(Text, nullable=False)
    hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    layer: Mapped[str] = mapped_column(String(60), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    rule_result: Mapped[RuleResultORM] = relationship(back_populates="check_items")


class FraudModelResultORM(Base):
    """独立模型识别预警结果。"""

    __tablename__ = "fraud_model_results"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    result: Mapped[str] = mapped_column(String(40), nullable=False)
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(160), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    probability: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="fraud_model_results")


class PCAFeatureScoreORM(Base):
    """PCA 综合指标分。"""

    __tablename__ = "pca_feature_scores"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="pca_feature_scores")


class RiskScoreComponentORM(Base):
    """综合风险提示强度分项贡献。"""

    __tablename__ = "risk_score_components"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(40), nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    max_score: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_detail: Mapped[str | None] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="risk_score_components")


class EvidencePackageORM(Base):
    """证据包快照。"""

    __tablename__ = "evidence_packages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id: Mapped[UUID] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    risk_summary: Mapped[str] = mapped_column(Text, nullable=False)
    model_evidence: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    case: Mapped[CaseORM] = relationship(back_populates="evidence_packages")
