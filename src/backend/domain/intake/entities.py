"""数据接入领域实体模块。

合并自：
- domain/models/ingest.py —— 上游完整脱敏宽表接入模型
- domain/models/feature.py —— 单条统计特征与风险提示结果模型

v1.0 只接收脱敏后的结构化统计特征，不接收身份字段、RES、原始票据或完整医疗记录。
"""

from typing import Any

from pydantic import BaseModel, Field

from ..audit.review.entities import RiskScoreBreakdown
from ..audit.review.workflow_projector import CaseFullResponse


# ============================================================================
# 来自 domain/models/ingest.py —— 上游完整脱敏宽表接入模型
# ============================================================================


class IngestRecordInput(BaseModel):
    """单条完整脱敏宽表记录输入。"""

    case_title: str | None = Field(default=None, description="可选案件标题")
    case_type: str | None = Field(default="上游宽表记录接入", description="可选案件类型")
    source_system: str | None = Field(default="医保结算申报系统", description="来源系统")
    record_version: str | None = Field(default="claim-wide-v1", description="记录版本")
    case_context: dict[str, Any] = Field(
        default_factory=dict,
        description="可选案件业务上下文，用于游客样本和材料仿真；不得包含 RES 或真实身份字段",
    )
    record: dict[str, Any] = Field(description="完整脱敏宽表记录，包含 81 个输入字段")


class IngestSchemaResponse(BaseModel):
    """GET /api/ingest-schema 响应。"""

    required_fields: list[str]
    supported_fields: list[str]
    forbidden_fields: list[str]
    example_json: IngestRecordInput


class IngestRecordSummary(BaseModel):
    """脱敏业务样本记录列表摘要。"""

    record_id: str
    subject_ref: str
    risk_level: str
    risk_score: float
    sample_result: str | None = None
    sample_category: str | None = None
    sample_label: str | None = None
    access_mode: str | None = None
    case_type: str | None = None
    expected_rule_ids: list[str] = Field(default_factory=list)


class IngestRecordResponse(BaseModel):
    """单条脱敏业务样本记录详情。"""

    record_id: str
    record: dict[str, float | str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    case_context: dict[str, Any] = Field(default_factory=dict)


class BatchIngestInput(BaseModel):
    """批量完整脱敏宽表接入输入。"""

    records: list[IngestRecordInput] = Field(
        description="多条完整脱敏宽表记录，按上传顺序排列"
    )


class BatchIngestResponse(BaseModel):
    """批量完整脱敏宽表接入响应。"""

    record_count: int
    cases: list[CaseFullResponse]


# ============================================================================
# 来自 domain/models/feature.py —— 单条统计特征与风险提示结果模型
# ============================================================================


class FeatureRecordInput(BaseModel):
    """单条医保结算统计记录输入。"""

    case_id: str | None = Field(default=None, description="可选案件 ID，兼容输入使用")
    case_title: str | None = Field(default=None, description="可选案件标题")
    case_type: str | None = Field(default="统计特征输入", description="可选案件类型")
    features: dict[str, Any] = Field(description="统计特征键值表")


class FeatureSchemaResponse(BaseModel):
    """GET /api/feature-schema 响应。"""

    required_fields: list[str]
    supported_fields: list[str]
    forbidden_fields: list[str]
    example_json: FeatureRecordInput


class ScoringResult(BaseModel):
    """模型风险信号结果。"""

    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    source: str
    evidence_ref: str
    reasons: list[str] = Field(default_factory=list)
    risk_score_breakdown: RiskScoreBreakdown | None = None
