"""SQLAlchemy ORM 模型模块。

提供所有数据库表的声明式 ORM 映射。
拆分为多个子模块：case_orm、review_orm、user_orm、agent_orm、case_agent_orm、ingest_orm、memory_orm。

外部使用者可通过 `from infrastructure.persistence.sql.models import Base` 获取声明式基类，
以及所有 ORM 模型类。

Base、TimestampMixin、utc_now 作为公共基础设施在此文件中定义，
各子模块通过 `from . import Base, TimestampMixin, utc_now` 引用。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for created/updated fields."""

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all database tables."""


class TimestampMixin:
    """Common created/updated timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


# 从各子模块导入所有 ORM 模型，保证外部可一步引入
# 导入顺序按表依赖关系排列（被引用表先于引用表）

from .user_orm import RoleORM, UserORM, UserRoleORM  # noqa: E402, F401

from .case_orm import (  # noqa: E402, F401
    CaseFingerprintORM,
    CaseORM,
    CaseSourceRecordORM,
    EvidencePackageORM,
    FraudModelResultORM,
    PCAFeatureScoreORM,
    RiskScoreComponentORM,
    RuleCheckItemORM,
    RuleResultORM,
)

from .review_orm import (  # noqa: E402, F401
    AuditNoteORM,
    ReviewAttachmentORM,
    ReviewORM,
    WorkflowTraceNodeORM,
)

from .agent_orm import (  # noqa: E402, F401
    AgentCheckpointORM,
    AgentEventORM,
    AgentEvidenceAnalysisORM,
    AgentEvidenceCitationORM,
    AgentRunORM,
    AgentToolCallORM,
)

from .case_agent_orm import (  # noqa: E402, F401
    CaseAgentCheckpointORM,
    CaseAgentEventORM,
    CaseAgentMessageORM,
    CaseAgentNodeExecutionORM,
    CaseAgentRunORM,
    CaseAgentSessionORM,
    CaseAgentToolCallORM,
)

from .caser_context_orm import CaserContextSectionORM  # noqa: E402, F401

from .ingest_orm import (  # noqa: E402, F401
    AppealMaterialORM,
    AuditActionORM,
    CaseMaterialAssetORM,
    CaseMaterialSnapshotORM,
    CaseAssignmentORM,
    IngestBatchItemORM,
    IngestBatchORM,
    StatisticalMaterialDocumentORM,
)

from .memory_orm import (  # noqa: E402, F401
    CaseMemoryIndexORM,
    CaseMemoryORM,
    MemoryEmbeddingORM,
    MemoryEventORM,
    MemoryGraphEdgeORM,
    MemoryGraphNodeORM,
    MemoryOutboxORM,
)
