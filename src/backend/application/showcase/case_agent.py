"""Deterministic Case Agent projection for the public interview showcase."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..ports.repositories import CaseRepository
from ...domain.audit.review.evidence_packager import EvidenceService
from ...domain.audit.review.entities import CaseDetail, RuleHit


class ShowcaseCaseAgentSource(BaseModel):
    """A safe source displayed beside one pre-generated answer."""

    source_ref: str
    label: str
    summary: str


class ShowcaseCaseAgentAnswer(BaseModel):
    """One pre-generated interview question and answer."""

    key: str
    question: str
    answer: str
    checks: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class ShowcaseCaseAgentProjection(BaseModel):
    """Read-only Case Agent content that never invokes a live model."""

    case_id: str
    title: str
    generated_notice: str
    boundary_notice: str
    answers: list[ShowcaseCaseAgentAnswer]
    sources: list[ShowcaseCaseAgentSource]


class ShowcaseCaseAgentService:
    """Build stable case assistance from already-computed safe facts."""

    def __init__(self, cases: CaseRepository, evidence: EvidenceService) -> None:
        self._cases = cases
        self._evidence = evidence

    def get_projection(self, case_id: str) -> ShowcaseCaseAgentProjection:
        case = self._cases.get_case(case_id)
        if case is None:
            raise KeyError(case_id)

        evidence = self._evidence.generate(case)
        triggered_rules = self._triggered_rules(case)
        sources = self._sources(case, triggered_rules)
        source_refs = [source.source_ref for source in sources]
        score = (
            case.risk_score_breakdown.display_score
            if case.risk_score_breakdown is not None
            else f"{round(case.risk_score * 100)} / 100"
        )
        rule_summary = (
            "、".join(rule.rule_name for rule in triggered_rules[:3])
            if triggered_rules
            else "未形成明确规则命中"
        )
        model_summary = case.fraud_screening.label or "未接入"
        action_checks = [
            f"{item.title}：{item.action}"
            for item in evidence.review_actions[:4]
        ]
        if not action_checks:
            action_checks = ["核对案件事实、规则适用范围和材料完整性"]
        missing = list(evidence.missing_information[:3])
        missing_text = "；".join(missing) if missing else "当前未标记具体缺失材料"

        return ShowcaseCaseAgentProjection(
            case_id=case.case_id,
            title="案件稽核助手 · 预生成会话",
            generated_notice="预生成演示，未调用实时模型",
            boundary_notice=(
                "助手只整理当前脱敏案件中的系统风险提示、确定性规则和基础证据包；"
                "不保存会话，不修改源数据、模型结果、规则结果或人工决定。"
            ),
            answers=[
                ShowcaseCaseAgentAnswer(
                    key="risk_focus",
                    question="这起案件为什么需要人工关注？",
                    answer=(
                        f"系统综合风险提示强度为 {score}，模型识别预警为“{model_summary}”。"
                        f"当前主要核验线索为：{rule_summary}。这些信号只用于人工分流，"
                        "不能单独形成违规、拒付、处罚或欺诈认定。"
                    ),
                    checks=action_checks[:3],
                    source_refs=source_refs,
                ),
                ShowcaseCaseAgentAnswer(
                    key="next_checks",
                    question="下一步优先核验什么？",
                    answer=(
                        f"建议按基础证据包逐项核对事实与材料。当前缺口提示为：{missing_text}。"
                        "如材料与结构化字段存在冲突，应记录差异并交由审核员判断。"
                    ),
                    checks=action_checks,
                    source_refs=source_refs,
                ),
                ShowcaseCaseAgentAnswer(
                    key="agent_boundary",
                    question="案件助手可以替审核员做决定吗？",
                    answer=(
                        "不可以。案件助手只能组织证据、解释系统提示和列出待核验事项。"
                        "最终处理意见必须由有权限的审核人员基于合法取得的业务材料确认。"
                    ),
                    checks=[
                        "确认回答引用当前案件已有来源",
                        "确认未修改模型分数、规则结果或人工意见",
                        "确认没有自动形成业务裁决",
                    ],
                    source_refs=source_refs,
                ),
            ],
            sources=sources,
        )

    @staticmethod
    def _triggered_rules(case: CaseDetail) -> list[RuleHit]:
        return [
            rule
            for rule in case.rule_hits
            if rule.hit or any(item.hit for item in rule.check_items)
        ]

    @staticmethod
    def _sources(
        case: CaseDetail,
        triggered_rules: list[RuleHit],
    ) -> list[ShowcaseCaseAgentSource]:
        sources = [
            ShowcaseCaseAgentSource(
                source_ref="showcase:risk",
                label="系统综合风险提示",
                summary=(
                    case.risk_score_breakdown.display_score
                    if case.risk_score_breakdown is not None
                    else f"{round(case.risk_score * 100)} / 100"
                ),
            )
        ]
        if case.fraud_screening.result != "not_available":
            probability = case.fraud_screening.probability
            sources.append(
                ShowcaseCaseAgentSource(
                    source_ref="showcase:model",
                    label=f"模型识别预警：{case.fraud_screening.label}",
                    summary=(
                        f"预警概率 {probability:.1%}"
                        if probability is not None
                        else case.fraud_screening.reason
                    ),
                )
            )
        sources.extend(
            ShowcaseCaseAgentSource(
                source_ref=f"showcase:rule:{rule.rule_id}",
                label=f"[{rule.rule_id}] {rule.rule_name}",
                summary=rule.business_explanation or rule.reason,
            )
            for rule in triggered_rules[:6]
        )
        return sources
