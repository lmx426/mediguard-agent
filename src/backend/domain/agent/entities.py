"""证据代理领域实体模块。

源自 domain/models/agent.py。

定义受控 Evidence Agent 的请求、运行、分析和账本等实体模型。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ============================================================================
# 类型别名
# ============================================================================

AnalysisType = Literal["comprehensive"]
EvidenceAgentEvalVariant = Literal["A0", "B1", "B2", "B3", "B4", "B5", "C1"]
AIRiskLabel = Literal[
    "存在疑似欺诈风险线索",
    "未发现明确疑似欺诈风险线索",
    "当前证据不足",
]
VerificationRelationType = Literal["risk_score_related", "scan_discovered"]
VerificationPriority = Literal["high", "medium", "low"]
EvidenceReviewRelation = Literal[
    "supports",
    "partially_supports",
    "weakly_supports",
    "insufficient_evidence",
    "inconsistent",
]
ClueReviewStatus = Literal["supported", "needs_review", "unconfirmed"]
RunStatus = Literal["queued", "running", "complete", "partial", "failed"]


# ============================================================================
# 来自 domain/models/agent.py —— 证据代理实体
# ============================================================================


class EvidenceAgentRequest(BaseModel):
    """A formal evidence analysis request, created automatically or by manual regenerate."""

    model_config = ConfigDict(extra="forbid")

    analysis_type: AnalysisType = "comprehensive"
    eval_variant: EvidenceAgentEvalVariant | None = None


class AgentCitation(BaseModel):
    """A resolved, case-scoped source stored with the final analysis."""

    citation_id: str
    source_type: str
    source_ref: str
    label: str
    version: str | None = None
    current_value: str | None = None
    threshold: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStatement(BaseModel):
    """A statement that must cite entries from the evidence ledger."""

    statement: str = Field(min_length=1, max_length=1600)
    source_refs: list[str] = Field(min_length=1, max_length=8)


class SystemRiskPrompt(BaseModel):
    """由后端风险引擎生成并固化到分析结果的只读风险提示快照。"""

    generated_by: Literal["backend_risk_engine"] = "backend_risk_engine"
    risk_level: Literal["high", "medium", "low", "insufficient"]
    risk_level_label: str = Field(min_length=1, max_length=40)
    risk_score: int = Field(ge=0, le=100)
    score_text: str = Field(min_length=1, max_length=80)
    source_summary: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(default_factory=list, max_length=12)


class AgentEvidenceReview(BaseModel):
    """Agent 对系统综合风险提示的证据复核结果。"""

    relation: EvidenceReviewRelation
    label: str = Field(default="证据不足", min_length=1, max_length=40)
    support_level: Literal["高", "中", "低", "证据不足"]
    summary: str = Field(min_length=1, max_length=2000)
    source_refs: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_relation_label(cls, value: Any) -> Any:
        """复核标签由 relation 确定，避免模型生成任意标题。"""

        if not isinstance(value, dict):
            return value
        labels = {
            "supports": "支持",
            "partially_supports": "部分支持",
            "weakly_supports": "支撑较弱",
            "insufficient_evidence": "证据不足",
            "inconsistent": "发现不一致",
        }
        relation = value.get("relation")
        if relation not in labels:
            return value
        return {**value, "label": labels[relation]}


class AgentClueReview(BaseModel):
    """候选线索与当前证据之间的复核关系。"""

    clue_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    status: ClueReviewStatus
    explanation: str = Field(min_length=1, max_length=1200)
    source_refs: list[str] = Field(min_length=1, max_length=8)


class AgentVerificationItem(BaseModel):
    """A non-decisional action for a human reviewer."""

    title: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=800)
    rationale: str = Field(min_length=1, max_length=800)
    relation_type: VerificationRelationType = Field(
        default="scan_discovered",
        description="核验事项来源分类：risk_score_related 表示风险提示关联核验，scan_discovered 表示全局扫描补充发现。",
    )
    priority: VerificationPriority = Field(
        default="medium",
        description="核验事项办理优先级：high / medium / low。",
    )
    source_refs: list[str] = Field(min_length=1, max_length=8)


class AgentSignalReview(BaseModel):
    """How the Agent rechecks signals against available evidence."""

    case_review_hint: str = Field(
        default="",
        max_length=1200,
        description="本案复核提示：说明已识别线索与证据、规则或材料之间的对应关系。",
    )
    case_review_hint_source_refs: list[str] = Field(default_factory=list, max_length=8)
    supported_clues: list[AgentStatement] = Field(
        default_factory=list,
        max_length=12,
        description="已有证据支持的线索。",
    )
    needs_review: list[AgentStatement] = Field(
        default_factory=list,
        max_length=12,
        description="仍需人工复核才能确认是否成立的线索。",
    )
    unconfirmed_items: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="因材料缺失、口径不足或冲突导致暂无法确认的事项。",
    )
    supplementary_review_hints: list[AgentStatement] = Field(
        default_factory=list,
        max_length=12,
        description="面向人工复核的补充核验提示。",
    )


class AgentGeneratedContent(BaseModel):
    """Strict model-facing JSON contract before citations are resolved."""

    status: Literal["complete", "partial"]
    ai_risk_label: AIRiskLabel = Field(
        default="当前证据不足",
        description="AI 对疑似欺诈风险线索是否存在的判断标签；不构成正式定性结论。",
    )
    ai_risk_label_source_refs: list[str] = Field(default_factory=list, max_length=8)
    risk_judgement: str = Field(
        default="当前证据仅支持作为人工核验参考。",
        min_length=1,
        max_length=1200,
        description="疑似欺诈风险线索研判，不是最终定性结论。",
    )
    risk_judgement_source_refs: list[str] = Field(default_factory=list, max_length=8)
    evidence_strength: str = Field(
        default="待人工核验",
        min_length=1,
        max_length=120,
        description="结论支撑度，表示当前证据与后端综合风险提示强度的匹配程度，例如：高 / 中 / 低 / 证据不足。",
    )
    evidence_strength_source_refs: list[str] = Field(default_factory=list, max_length=8)
    key_risk_signals: list[AgentStatement] = Field(default_factory=list, max_length=20)
    human_review_focus: list[AgentVerificationItem] = Field(default_factory=list, max_length=20)
    risk_overview: str = Field(min_length=1, max_length=2000)
    risk_overview_source_refs: list[str] = Field(min_length=1, max_length=8)
    supporting_evidence: list[AgentStatement] = Field(default_factory=list, max_length=20)
    conflicts: list[AgentStatement] = Field(default_factory=list, max_length=20)
    missing_information: list[str] = Field(default_factory=list, max_length=20)
    signal_review: AgentSignalReview = Field(default_factory=AgentSignalReview)
    verification_checklist: list[AgentVerificationItem] = Field(default_factory=list, max_length=20)
    boundary_notice: str = Field(min_length=1, max_length=500)


class ReviewAdvisoryGeneratedContentV2(BaseModel):
    """Review Advisor V2 模型输出契约，只保留一次生成所需的业务语义。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "partial"]
    evidence_review: AgentEvidenceReview
    clue_reviews: list[AgentClueReview] = Field(default_factory=list, max_length=24)
    verification_checklist: list[AgentVerificationItem] = Field(default_factory=list, max_length=20)
    conflicts: list[AgentStatement] = Field(default_factory=list, max_length=20)
    missing_information: list[AgentStatement] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_generated_content(cls, value: Any) -> Any:
        """兼容旧测试网关或滚动升级期间返回的 V1 模型 JSON。"""

        if not isinstance(value, dict) or "evidence_review" in value:
            return value

        refs = list(
            dict.fromkeys(
                value.get("risk_overview_source_refs", [])
                or value.get("evidence_strength_source_refs", [])
                or value.get("risk_judgement_source_refs", [])
            )
        )
        strength = value.get("evidence_strength", "证据不足")
        relation_by_strength = {
            "高": ("supports", "支持"),
            "中": ("partially_supports", "部分支持"),
            "低": ("weakly_supports", "支撑较弱"),
        }
        relation, label = relation_by_strength.get(
            strength,
            ("insufficient_evidence", "证据不足"),
        )
        signal_review = value.get("signal_review") or {}
        clue_reviews: list[dict[str, Any]] = []
        clue_groups = (
            ("supported", signal_review.get("supported_clues") or value.get("supporting_evidence", [])),
            ("needs_review", signal_review.get("needs_review", [])),
        )
        for status, items in clue_groups:
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not item.get("statement"):
                    continue
                clue_reviews.append(
                    {
                        "clue_id": f"legacy:{status}:{index + 1}",
                        "title": str(item["statement"]).split("：", 1)[0][:200],
                        "status": status,
                        "explanation": item["statement"],
                        "source_refs": item.get("source_refs") or refs,
                    }
                )
        for index, item in enumerate(signal_review.get("unconfirmed_items", [])):
            clue_reviews.append(
                {
                    "clue_id": f"legacy:unconfirmed:{index + 1}",
                    "title": str(item).split("：", 1)[0][:200],
                    "status": "unconfirmed",
                    "explanation": str(item),
                    "source_refs": refs,
                }
            )
        missing_information = [
            item
            if isinstance(item, dict)
            else {"statement": str(item), "source_refs": refs}
            for item in value.get("missing_information", [])
        ]
        return {
            "status": value.get("status", "partial"),
            "evidence_review": {
                "relation": relation,
                "label": label,
                "support_level": strength if strength in {"高", "中", "低", "证据不足"} else "证据不足",
                "summary": value.get("risk_overview") or value.get("risk_judgement") or "当前证据不足以完成复核。",
                "source_refs": refs,
            },
            "clue_reviews": clue_reviews,
            "verification_checklist": value.get("verification_checklist", []),
            "conflicts": value.get("conflicts", []),
            "missing_information": missing_information,
        }


class EvidenceAgentAnalysis(BaseModel):
    """Immutable validated Evidence Agent result."""

    analysis_id: str
    run_id: str
    case_id: str
    analysis_type: AnalysisType
    status: Literal["complete", "partial"]
    system_risk_prompt: SystemRiskPrompt | None = None
    evidence_review: AgentEvidenceReview | None = None
    clue_reviews: list[AgentClueReview] = Field(default_factory=list)
    ai_risk_label: AIRiskLabel = "当前证据不足"
    ai_risk_label_source_refs: list[str] = Field(default_factory=list)
    risk_judgement: str = "当前证据仅支持作为人工核验参考。"
    risk_judgement_source_refs: list[str] = Field(default_factory=list)
    evidence_strength: str = "待人工核验"
    evidence_strength_source_refs: list[str] = Field(default_factory=list)
    key_risk_signals: list[AgentStatement] = Field(default_factory=list)
    human_review_focus: list[AgentVerificationItem] = Field(default_factory=list)
    risk_overview: str
    risk_overview_source_refs: list[str]
    supporting_evidence: list[AgentStatement] = Field(default_factory=list)
    conflicts: list[AgentStatement] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    missing_information_details: list[AgentStatement] = Field(default_factory=list)
    signal_review: AgentSignalReview = Field(default_factory=AgentSignalReview)
    verification_checklist: list[AgentVerificationItem] = Field(default_factory=list)
    citations: list[AgentCitation] = Field(default_factory=list)
    boundary_notice: str
    generated_notice: str = "AI生成，仅供人工核验"
    input_fingerprint: str
    model_name: str
    prompt_version: str
    tool_version: str
    created_at: datetime


class AgentRun(BaseModel):
    """Public state of one persisted Agent run."""

    run_id: str
    case_id: str
    actor_id: str
    analysis_type: AnalysisType
    status: RunStatus
    current_node: str
    input_fingerprint: str
    reused: bool = False
    error_code: str | None = None
    error_message: str | None = None
    model_call_count: int = 0
    tool_call_count: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    analysis: EvidenceAgentAnalysis | None = None


class AgentEvent(BaseModel):
    """Persisted business progress event suitable for SSE replay."""

    run_id: str
    sequence: int
    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EvidenceLedgerItem(BaseModel):
    """A tool-produced fact that final model output is allowed to cite."""

    ledger_ref: str
    source_type: str
    source_ref: str
    label: str
    value: Any
    version: str | None = None
    current_value: str | None = None
    threshold: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
