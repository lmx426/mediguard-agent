"""审核领域实体模块。

合并自：
- domain/models/case.py —— 案件、规则命中、风险分等核心数据结构
- domain/models/auth.py —— 认证用户模型
- domain/models/statistical_materials.py —— 统计材料只读模型
- domain/review.py —— 人工初审和工作笔记的安全校验函数

所有数据均来自 fixture，不包含 RES、不经过 LLM。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .workflow_projector import ReviewAttachment, ReviewInput


# ============================================================================
# 来自 domain/models/case.py —— 案件领域模型
# ============================================================================


class RuleCheckItem(BaseModel):
    """规则内部的单个核验项。"""

    label: str = Field(description="核验项名称")
    current_value: str = Field(description="当前值，面向审核员展示")
    threshold: str = Field(description="阈值或核验口径")
    hit: bool = Field(description="该核验项是否触发")
    layer: str = Field(
        default="risk_signal",
        description="输出层级：data_quality_or_applicability | risk_signal | strong_review_signal",
    )
    severity: str = Field(description="严重度：info | low | medium | high | critical")
    explanation: str = Field(description="业务解释")


class RuleHit(BaseModel):
    """单条确定性规则命中结果。

    Agent 可以读取规则结果，但不能修改规则结果。
    """

    rule_id: str = Field(description="规则唯一标识，如 OP-R001")
    rule_name: str = Field(description="规则名称，如 高频就诊分级规则")
    hit: bool = Field(description="是否命中风险线索或强复核线索")
    severity: str = Field(description="严重度：info | low | medium | high | critical")
    reason: str = Field(description="命中或不命中的简要原因")
    evidence_ref: str = Field(description="证据引用标识，如 rule:OP-R001:visit_frequency")
    version: str = Field(description="规则版本号，如 1.0.0")
    layer: str = Field(
        default="risk_signal",
        description="输出层级：data_quality_or_applicability | risk_signal | strong_review_signal",
    )
    action: str = Field(
        default="MANUAL_REVIEW",
        description="建议动作：MANUAL_REVIEW | REQUEST_SUPPLEMENT | SPECIAL_AUDIT | DATA_QUALITY_CHECK",
    )
    current_value: str = Field(default="", description="当前值摘要")
    threshold: str = Field(default="", description="阈值或核验口径摘要")
    business_explanation: str = Field(default="", description="业务解释")
    check_items: list[RuleCheckItem] = Field(default_factory=list, description="规则内部核验项")


FraudScreeningResult = Literal["suspected", "not_suspected", "not_available"]
RiskScoreComponentKey = Literal[
    "model_warning",
    "rule_check",
    "peer_deviation",
    "data_flow",
]


class FraudScreeningSignal(BaseModel):
    """独立模型识别预警二分类信号。

    该字段不由风险分、风险等级或规则命中推导；当前未接入真实二分类模型时，
    统一返回 not_available，避免把异常筛查强度误读为模型识别结论。
    """

    result: FraudScreeningResult = Field(
        default="not_available",
        description="二分类结果：suspected | not_suspected | not_available",
    )
    label: str = Field(default="未接入", description="前端展示名")
    source: str = Field(
        default="模型识别预警未接入",
        description="独立二分类模型或上游结果来源",
    )
    evidence_ref: str = Field(
        default="fraud_screening:binary:not-configured",
        description="独立二分类结果来源引用",
    )
    reason: str | None = Field(
        default="当前环境未配置独立模型识别预警；模型贡献未参与综合风险提示强度计算。",
        description="结果说明",
    )
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="欺诈正类概率；模型不支持概率或未接入时为空",
    )


class PCAFeatureScore(BaseModel):
    """PCA 降维后的 10 个综合指标分，用于前端事实底座展示。"""

    name: str = Field(description="综合指标名称，来自算法说明书表 4-3")
    score: float = Field(description="该综合指标对应的 PCA 分值")


class RiskScoreComponent(BaseModel):
    """综合风险提示强度的单项贡献。"""

    key: RiskScoreComponentKey = Field(description="贡献项唯一标识")
    label: str = Field(description="贡献项业务名称")
    score: int = Field(ge=0, description="当前贡献分")
    max_score: int = Field(gt=0, description="该贡献项满分")
    summary: str = Field(description="面向审核员的一句话解释")
    details: list[str] = Field(default_factory=list, description="展开说明")
    source_detail: str | None = Field(default=None, description="来源说明")


class RiskScoreBreakdown(BaseModel):
    """综合风险提示强度结构化分解。

    该分数用于人工核验分流，不等同欺诈认定、拒付建议或处罚依据。
    """

    total_score: int = Field(ge=0, le=100, description="综合风险提示强度，100 分制")
    max_score: int = Field(default=100, description="综合风险提示强度满分")
    display_score: str = Field(description="前端展示分数，如 82 / 100")
    level: str = Field(description="风险等级：low | medium | high")
    level_label: str = Field(description="风险等级中文名")
    components: list[RiskScoreComponent] = Field(
        default_factory=list,
        description="贡献项列表",
    )
    cap_note: str | None = Field(
        default=None,
        description="模型无预警封顶等业务口径说明",
    )


class CaseSummary(BaseModel):
    """案件摘要——用于案件列表展示。"""

    case_id: str = Field(description="案件唯一标识，如 CASE-001")
    case_title: str = Field(description="案件标题")
    case_type: str = Field(description="案件类型：门诊 | 住院 | 跨省就医")
    risk_level: str = Field(description="风险等级：low | medium | high | insufficient")
    risk_score: float = Field(ge=0.0, le=1.0, description="综合风险提示强度，范围 0.0 ~ 1.0")
    review_status: str = Field(
        default="pending",
        description="审核办理状态：pending | reviewed",
    )
    rule_signal_count: int = Field(default=0, ge=0, description="命中的规则风险线索数量")
    claim_amount: float | None = Field(default=None, description="申报总费用")


class CaseDetail(CaseSummary):
    """案件详情——工作台展示的完整信息。"""

    claim_summary: str = Field(description="案情描述摘要")
    model_evidence_ref: str = Field(
        default="model:score:v1.0",
        description="模型评分的证据引用标识",
    )
    rule_hits: list[RuleHit] = Field(description="规则命中结果列表")
    expected_recommendation: str = Field(
        description="审核建议方向，来自确定性风险信号和规则结果，不来自 LLM"
    )
    model_signal_source: str = Field(
        default="医保结算风险信号引擎（op-rule-pool-v0.1）",
        description="模型风险信号来源，v1.0 默认为确定性业务风险信号引擎",
    )
    model_signal_reasons: list[str] = Field(
        default_factory=list,
        description="模型风险信号的确定性加分原因",
    )
    review_priority: str = Field(
        default="routine",
        description="人工复核优先级：routine | manual_review | high_priority | special_audit",
    )
    evidence_consistency: str = Field(
        default="模型信号与规则证据待核验",
        description="模型信号与规则证据的一致性摘要",
    )
    subject_ref: str | None = Field(
        default=None,
        description="合成脱敏申报人编号，如 SIM_PERSON_000001",
    )
    rule_pool_version: str = Field(
        default="op-rule-pool-v0.1",
        description="规则池版本",
    )
    rule_baseline_version: str = Field(
        default="mediredata-baseline-v0.1",
        description="聚合阈值基线版本",
    )
    input_features: dict[str, float] = Field(
        default_factory=dict,
        description="脱敏统计特征快照，不含 RES 或身份字段",
    )
    source_record: dict[str, float | str] = Field(
        default_factory=dict,
        description="完整脱敏宽表记录快照，不含 RES，个人编码仅允许合成编码",
    )
    case_context: dict[str, Any] = Field(
        default_factory=dict,
        description="案件业务上下文快照，仅用于样本说明、规则提示和材料仿真，不含 RES 或真实身份字段",
    )
    fraud_screening: FraudScreeningSignal = Field(
        default_factory=FraudScreeningSignal,
        description="独立模型识别预警二分类信号，是综合风险提示强度的模型贡献来源",
    )
    pca_feature_scores: list[PCAFeatureScore] = Field(
        default_factory=list,
        description="PCA 10 个综合指标分，仅用于事实底座展示，不进入证据包",
    )
    risk_score_breakdown: RiskScoreBreakdown | None = Field(
        default=None,
        description="综合风险提示强度分项贡献，前端只读展示",
    )


# ============================================================================
# 来自 domain/models/auth.py —— 认证用户模型
# ============================================================================


class AuthenticatedUser(BaseModel):
    """User identity exposed to application use cases and the frontend."""

    id: str
    username: str
    display_name: str
    department: str | None = None
    roles: list[str] = Field(default_factory=list)


class StoredUser(AuthenticatedUser):
    """User record returned by repositories for authentication checks."""

    status: str
    password_hash: str | None = None
    case_memory_enabled: bool = True

    def to_authenticated(self) -> AuthenticatedUser:
        """将存储用户转换为认证用户视图。"""
        return AuthenticatedUser(
            id=self.id,
            username=self.username,
            display_name=self.display_name,
            department=self.department,
            roles=list(self.roles),
        )


# ============================================================================
# 来自 domain/models/statistical_materials.py —— 统计材料只读模型
# ============================================================================


class MedicalRecordDocument(BaseModel):
    """单份聚合统计材料。

    类名暂时保持兼容，返回内容不表示真实病历或原始医疗记录。
    """

    document_id: str = Field(description="材料唯一标识")
    title: str = Field(description="材料标题")
    document_type: str = Field(description="材料类型")
    visit_date: str = Field(description="就诊日期或统计周期")
    institution: str = Field(description="脱敏机构名称")
    department: str = Field(description="科室")
    status: str = Field(description="材料状态")
    content: str = Field(description="脱敏完整材料正文")
    check_points: list[str] = Field(
        default_factory=list,
        description="供审核员查看的核验要点",
    )
    category_id: str = Field(default="", description="一级材料分类标识")
    category_title: str = Field(default="", description="一级材料分类名称")
    occurred_at: str = Field(default="", description="材料发生时间")
    material_source: str = Field(default="", description="材料来源")
    occurrence_scene: str = Field(default="", description="发生场景")
    material_shape: str = Field(default="文本", description="材料形态")
    summary_items: list[str] = Field(default_factory=list, description="材料摘要项")
    tables: list["BusinessMaterialTable"] = Field(
        default_factory=list,
        description="结构化业务明细表",
    )
    assets: list["BusinessMaterialAsset"] = Field(
        default_factory=list,
        description="受控预览资产",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="材料内部元数据")


class BusinessMaterialAsset(BaseModel):
    """业务材料中的受控预览资产元数据。"""

    asset_id: str = Field(description="资产唯一标识")
    material_id: str = Field(description="所属材料标识")
    title: str = Field(description="资产标题")
    asset_type: Literal["prescription_png", "pharmacy_receipt_png"] = Field(
        description="资产类型"
    )
    file_type: Literal["image/png"] = Field(default="image/png", description="文件类型")
    preview_url: str = Field(default="", description="受控预览地址")
    sha256: str = Field(default="", description="文件哈希")
    size_bytes: int = Field(default=0, ge=0, description="文件大小")


class BusinessMaterialTable(BaseModel):
    """业务材料详情中的结构化表格。"""

    title: str = Field(description="表格标题")
    columns: list[str] = Field(default_factory=list, description="列名")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="行数据")


class BusinessMaterialCategory(BaseModel):
    """业务材料一级分类。"""

    category_id: str = Field(description="分类标识")
    title: str = Field(description="分类名称")
    documents: list[MedicalRecordDocument] = Field(default_factory=list)


class MaterialSubjectProfile(BaseModel):
    """申报人脱敏画像，只展示审核相关事实，不含真实身份字段。"""

    subject_ref: str = Field(description="脱敏申报人编号")
    gender: str = Field(description="性别")
    age_group: str = Field(description="年龄段")
    insurance_type: str = Field(description="参保类型")
    patient_group_tags: list[str] = Field(default_factory=list, description="人群标签")
    chronic_condition_tags: list[str] = Field(default_factory=list, description="慢病标签")
    allergy_history: str = Field(default="未见记录", description="过敏史")
    primary_visit_type: str = Field(default="门诊", description="主要就诊类型")
    registration_status: str = Field(default="缺失", description="挂号状态")


class MaterialAssetLocation(BaseModel):
    """材料资产的内部存储位置，不直接暴露给前端。"""

    material_id: str
    asset_id: str
    asset_type: str
    file_type: str
    storage_path: str
    sha256: str
    size_bytes: int
    display_name: str


class MedicalRecordResponse(BaseModel):
    """统计材料视图响应。"""

    case_id: str
    disclaimer: str = Field(default="")
    notice: str = Field(default="")
    generation_status: Literal["ready", "failed"] = "ready"
    template_version: str = Field(default="")
    subject_profile: MaterialSubjectProfile | None = None
    case_context: dict[str, Any] = Field(
        default_factory=dict,
        description="安全展示用案件上下文，不含真实身份或标签字段",
    )
    categories: list[BusinessMaterialCategory] = Field(default_factory=list)
    documents: list[MedicalRecordDocument] = Field(default_factory=list)


# ============================================================================
# 来自 domain/review.py —— 人工初审和工作笔记的安全校验
# ============================================================================

# 审核备注中禁止出现的敏感术语
FORBIDDEN_REVIEW_TERMS = ("个人编码", "身份证", "姓名", "电话", "住址")


def find_forbidden_content(content: str) -> str | None:
    """查找不允许进入笔记或材料元数据的标签及身份字段。

    参数：
        content: 待检查的文本内容。

    返回：
        匹配到的禁止词，未匹配时返回 None。
    """

    # 使用负向环视确保 RES 作为独立词出现，避免误匹配其他字段名
    if re.search(r"(?i)(?<![a-z0-9])RES(?![a-z0-9])", content):
        return "RES"
    return next(
        (term for term in FORBIDDEN_REVIEW_TERMS if term in content),
        None,
    )


def attachment_metadata_text(attachments: Sequence[ReviewAttachment]) -> str:
    """将材料元数据展开为仅用于安全检查的文本。

    参数：
        attachments: 人工初审材料附件元数据序列。

    返回：
        拼接后的纯文本，供安全检查使用。
    """

    parts: list[str] = []
    for attachment in attachments:
        parts.extend(
            [
                attachment.name,
                attachment.material_type,
                attachment.source,
                attachment.verification_status,
                attachment.remark,
                attachment.file_name or "",
                attachment.file_type or "",
                attachment.file_last_modified or "",
            ]
        )
    return "\n".join(parts)


def review_attachment_text(body: ReviewInput) -> str:
    """将附件元数据展开为仅用于安全检查的文本。

    参数：
        body: 人工审核意见输入。

    返回：
        拼接后的纯文本，供安全检查使用。
    """

    return attachment_metadata_text(body.attachments)
