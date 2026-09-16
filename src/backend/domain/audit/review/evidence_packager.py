"""确定性证据包模块。

合并自：
- domain/models/evidence.py —— 证据包领域模型（Citation, RuleEvidenceCard, ReviewAction, EvidencePackage）
- domain/evidence.py —— 确定性证据包生成器（EvidenceService + 常量）

v1.0 不使用 LLM，采用基于模型信号和规则结果的模板逻辑。
证据包只组织证据，不做最终裁决。
"""

from pydantic import BaseModel, Field

from .entities import CaseDetail, RuleHit
from .rule_evaluator import LAYER_DATA, LAYER_STRONG

# ============================================================================
# 来自 domain/models/evidence.py —— 确定性证据包领域模型
# ============================================================================


class Citation(BaseModel):
    """证据引用——每条 Agent 建议至少关联一个模型或规则证据来源。"""

    label: str = Field(description="引用标签，如 模型风险分 0.12")
    source: str = Field(description="证据来源类型：model_evidence | rule_evidence | fraud_screening")
    ref: str = Field(description="证据引用标识，如 model:score:v1.0")


class RuleEvidenceCard(BaseModel):
    """面向审核员展示的规则核验卡片。"""

    rule_id: str = Field(description="规则唯一标识")
    rule_name: str = Field(description="规则名称")
    severity: str = Field(description="严重度")
    hit_fact: str = Field(description="命中事实或核验事实")
    review_impact: str = Field(description="对人工审核的影响")
    suggested_action: str = Field(description="建议核验动作")
    supplement_materials: list[str] = Field(
        default_factory=list,
        description="建议补充或重点查看的材料",
    )
    source_ref: str = Field(description="来源引用标识")


class ReviewAction(BaseModel):
    """证据包给审核员的核验动作建议。"""

    title: str = Field(description="动作标题")
    action: str = Field(description="建议动作")
    rationale: str = Field(description="建议原因")
    source_refs: list[str] = Field(
        default_factory=list,
        description="关联的来源引用",
    )


class EvidencePackage(BaseModel):
    """Agent 生成的证据包。

    约束：
    - 只引用模型评分和规则命中证据
    - 不包含最终违规裁决
    - 不给出自动化不予支付或行政处理建议
    - 证据不足时标记 missing_information，不编造额外事实
    """

    case_id: str = Field(description="关联的案件 ID")
    risk_summary: str = Field(description="风险等级摘要文本")
    model_evidence: str = Field(description="模型评分证据描述")
    rule_evidence: list[str] = Field(description="每条命中规则的证据描述列表")
    rule_cards: list[RuleEvidenceCard] = Field(
        default_factory=list,
        description="结构化规则核验卡片",
    )
    review_actions: list[ReviewAction] = Field(
        default_factory=list,
        description="结构化审核动作建议",
    )
    recommendation: str = Field(
        description="审核建议。不包含裁决性结论"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="证据不足时列出的缺失信息，不编造额外依据",
    )
    citations: list[Citation] = Field(
        description="每条建议关联的证据引用列表"
    )
    forbidden_actions: list[str] = Field(
        description="明确禁止的动作列表，Agent 不得执行"
    )


# ============================================================================
# 来自 domain/evidence.py —— 确定性证据包生成器
# ============================================================================

FORBIDDEN_ACTIONS: list[str] = [
    "证据包生成器不得输出最终违规裁决",
    "证据包生成器不得给出自动化不予支付建议",
    "证据包生成器不得给出行政处理建议",
    "审核结论应由人工审核员独立做出",
]

ACTION_LABELS: dict[str, str] = {
    "MANUAL_REVIEW": "建议人工核验",
    "REQUEST_SUPPLEMENT": "要求补充材料",
    "SPECIAL_AUDIT": "准备专项核验材料",
    "DATA_QUALITY_CHECK": "先确认数据完整性",
}

PRIORITY_LABELS: dict[str, str] = {
    "routine": "常规处理 / 抽样核验",
    "manual_review": "建议人工核验",
    "high_priority": "建议优先核验",
    "special_audit": "重点核验 / 专项核验准备",
}


class EvidenceService:
    """确定性证据包生成器。

    基于模型信号和规则结果，使用模板逻辑组织证据包。
    不调用 LLM，不做最终裁决。
    """

    def generate(self, case: CaseDetail) -> EvidencePackage:
        """根据案件详情生成完整的证据包。

        参数：
            case: 案件详情，包含模型评分和规则命中结果。

        返回：
            结构化的证据包，供审核员使用。
        """
        risk_rules = [rule for rule in case.rule_hits if rule.hit]
        notice_rules = [
            rule
            for rule in case.rule_hits
            if any(item.hit and item.layer == LAYER_DATA for item in rule.check_items)
        ]
        triggered_rules = self._unique_rules([*risk_rules, *notice_rules])

        return EvidencePackage(
            case_id=case.case_id,
            risk_summary=self._build_risk_summary(case, risk_rules, notice_rules),
            model_evidence=self._build_model_evidence(case),
            rule_evidence=self._build_rule_evidence(case, risk_rules, notice_rules),
            rule_cards=self._build_rule_cards(case, risk_rules, notice_rules),
            review_actions=self._build_review_actions(case, risk_rules, notice_rules),
            recommendation=self._build_recommendation(case, risk_rules, notice_rules),
            missing_information=self._build_missing_information(case, risk_rules, notice_rules),
            citations=self._build_citations(case, triggered_rules),
            forbidden_actions=FORBIDDEN_ACTIONS,
        )

    @staticmethod
    def _build_risk_summary(
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> str:
        """构建风险摘要文本。"""
        if case.risk_score_breakdown:
            active_components = [
                component.label
                for component in case.risk_score_breakdown.components
                if component.score > 0
            ]
            source_text = (
                "主要来自" + "、".join(active_components[:3])
                if active_components
                else "当前未形成明显贡献来源"
            )
            return (
                f"风险提示强度 {case.risk_score_breakdown.display_score}，"
                f"{case.risk_score_breakdown.level_label}。"
                f"{source_text}；规则核验和同类偏离用于辅助核验。"
            )
        strong_count = len([rule for rule in risk_rules if rule.layer == LAYER_STRONG])
        return (
            f"风险提示强度 {case.risk_score:.2f}，"
            f"发现业务线索 {len(risk_rules)} 类、重点核验线索 {strong_count} 类、"
            f"数据质量或适用范围提示 {len(notice_rules)} 类。"
            f"{case.evidence_consistency}"
        )

    @staticmethod
    def _build_model_evidence(case: CaseDetail) -> str:
        """构建模型证据描述。"""
        fraud_signal = case.fraud_screening
        probability_text = (
            f"预警概率：{fraud_signal.probability:.1%}。"
            if fraud_signal.probability is not None
            else ""
        )
        fraud_evidence = (
            (
                f"模型识别预警：{fraud_signal.label}。"
                f"{probability_text}"
            )
            if fraud_signal.result != "not_available"
            else (
                "模型识别预警未接入；"
                "当前模型贡献未参与本项分值。"
            )
        )
        if case.risk_score_breakdown:
            component_text = "；".join(
                f"{component.label} {component.score}/{component.max_score}"
                for component in case.risk_score_breakdown.components
            )
            cap_note = case.risk_score_breakdown.cap_note or ""
            return (
                f"{case.model_signal_source} 生成综合风险提示强度 "
                f"{case.risk_score_breakdown.display_score}，"
                f"等级：{case.risk_score_breakdown.level_label}。"
                f"贡献构成：{component_text}。"
                f"{fraud_evidence}"
                f"{cap_note}"
                "该提示只用于人工核验分流，不可由 Agent 修改。"
            )
        reasons = "；".join(case.model_signal_reasons) if case.model_signal_reasons else "无额外加分原因"
        return (
            f"{case.model_signal_source} 基于单条脱敏统计宽表记录生成风险提示："
            f"{case.risk_score:.2f}，等级：{case.risk_level}。"
            f"重点关注原因：{reasons}。"
            f"{fraud_evidence}"
            "该提示为确定性异常筛查结果，非 LLM 生成，不可由 Agent 修改。"
        )

    @staticmethod
    def _build_rule_evidence(
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> list[str]:
        """构建规则证据描述列表。"""
        evidence: list[str] = [
            "已完成本案费用、就诊频次、机构和挂号流程等业务核验。"
        ]

        if risk_rules:
            for rule in risk_rules:
                action = ACTION_LABELS.get(rule.action, rule.action)
                evidence.append(
                    f"{rule.rule_name}：{rule.reason}。"
                    f"当前值：{rule.current_value}。建议动作：{action}。"
                )
        else:
            evidence.append("当前没有形成需要单独核验的业务线索。")

        for rule in notice_rules:
            notice_items = [
                item for item in rule.check_items if item.hit and item.layer == LAYER_DATA
            ]
            if not notice_items:
                continue
            evidence.append(
                f"{rule.rule_name}："
                + "；".join(item.explanation for item in notice_items)
            )

        return evidence

    def _build_rule_cards(
        self,
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> list[RuleEvidenceCard]:
        """构建结构化规则核验卡片列表。"""
        triggered_rules = self._unique_rules([*risk_rules, *notice_rules])
        return [
            RuleEvidenceCard(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                severity=rule.severity,
                hit_fact=self._build_hit_fact(rule),
                review_impact=self._build_review_impact(rule),
                suggested_action=ACTION_LABELS.get(rule.action, rule.action),
                supplement_materials=self._suggest_materials(rule),
                source_ref=rule.evidence_ref,
            )
            for rule in triggered_rules
        ]

    def _build_review_actions(
        self,
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> list[ReviewAction]:
        """构建结构化审核动作建议列表。"""
        triggered_rules = self._unique_rules([*risk_rules, *notice_rules])
        refs = [rule.evidence_ref for rule in triggered_rules] or [case.model_evidence_ref]
        actions: list[ReviewAction] = []

        if case.review_priority == "special_audit":
            actions.append(
                ReviewAction(
                    title="准备专项核验材料",
                    action="由审核员确认是否移交上级或专项核验人员继续核验",
                    rationale="当前存在重点核验线索，需要结合申报材料和统计材料进一步确认。",
                    source_refs=refs,
                )
            )
        elif case.review_priority == "high_priority":
            actions.append(
                ReviewAction(
                    title="建议优先核验",
                    action="优先核对命中的业务线索、费用结构和统计材料",
                    rationale="风险提示强度较高或多项规则线索同时出现。",
                    source_refs=refs,
                )
            )
        elif case.review_priority == "manual_review":
            actions.append(
                ReviewAction(
                    title="完成人工核验",
                    action="结合业务核验结果、费用明细和统计材料形成初审意见",
                    rationale="当前存在需要人工确认的筛查或规则线索。",
                    source_refs=refs,
                )
            )
        elif notice_rules and not risk_rules:
            actions.append(
                ReviewAction(
                    title="先核验材料完整性",
                    action="确认记录适用范围、挂号状态和费用字段是否完整",
                    rationale="当前主要是数据质量或适用范围提示，尚未形成明确风险线索。",
                    source_refs=refs,
                )
            )
        else:
            actions.append(
                ReviewAction(
                    title="按常规抽样口径处理",
                    action="可按常规流程处理，或纳入抽样核验",
                    rationale="当前风险提示和规则核验均未发现明显异常。",
                    source_refs=refs,
                )
            )

        if risk_rules:
            actions.append(
                ReviewAction(
                    title="补齐重点材料",
                    action="查看处方、费用明细、挂号记录或诊疗摘要材料中的对应事实",
                    rationale="业务线索需要通过申报材料和统计材料确认是否成立。",
                    source_refs=[rule.evidence_ref for rule in risk_rules],
                )
            )
        return actions

    @staticmethod
    def _build_recommendation(
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> str:
        """构建审核建议文本。"""
        if case.review_priority == "special_audit":
            return (
                "建议准备专项核验材料并由审核员重点确认。"
                "当前存在重点核验线索，需结合申报材料和业务口径进一步确认。"
            )
        if case.review_priority == "high_priority":
            return (
                "建议优先核验。"
                "风险提示强度或多项规则线索提示该案件需要优先查看。"
            )
        if case.review_priority == "manual_review":
            return (
                "建议人工核验。"
                "当前存在筛查关注或单项规则线索，需由审核员结合材料判断。"
            )
        if notice_rules and not risk_rules:
            return (
                "建议标记数据质量问题或要求补充材料。"
                "当前未形成风险线索，但存在数据质量或适用范围提示。"
            )
        return (
            "建议按常规流程处理或进入抽样复核。"
            "当前风险提示和规则核验均未发现明显异常。"
        )

    @staticmethod
    def _build_missing_information(
        case: CaseDetail,
        risk_rules: list[RuleHit],
        notice_rules: list[RuleHit],
    ) -> list[str]:
        """识别缺失信息列表。"""
        missing: list[str] = []
        if case.risk_level in ("medium", "high") and not risk_rules:
            missing.append("筛查提示关注但规则证据不足，建议查看完整申报材料。")
        if len(risk_rules) == 1:
            missing.append(
                f"仅发现 1 类业务线索（{risk_rules[0].rule_name}），建议结合费用清单核验。"
            )
        if any(rule.rule_id == "OP-R003" for rule in risk_rules):
            missing.append("药品费用组合线索需要结合处方、药品明细或长期用药材料人工核验。")
        if any(rule.rule_id == "OP-R005" for rule in risk_rules):
            missing.append("挂号状态组合线索需要补充流程材料或挂号记录核验。")
        if notice_rules:
            missing.append("存在数据质量或适用范围提示，建议先确认记录是否属于门诊普通结算核验范围。")
        return missing

    @staticmethod
    def _build_citations(
        case: CaseDetail,
        triggered_rules: list[RuleHit],
    ) -> list[Citation]:
        """构建证据引用列表。"""
        citations: list[Citation] = [
            Citation(
                label=(
                    f"风险提示强度 {case.risk_score_breakdown.display_score}"
                    if case.risk_score_breakdown
                    else f"风险提示强度 {case.risk_score:.2f}"
                ),
                source="model_evidence",
                ref=case.model_evidence_ref,
            )
        ]
        if case.fraud_screening.result != "not_available":
            citations.append(
                Citation(
                    label=f"模型识别预警 {case.fraud_screening.label}",
                    source="fraud_screening",
                    ref=case.fraud_screening.evidence_ref,
                )
            )
        for rule in triggered_rules:
            citations.append(
                Citation(
                    label=f"规则 [{rule.rule_id}] {rule.rule_name}",
                    source="rule_evidence",
                    ref=rule.evidence_ref,
                )
            )
        return citations

    @staticmethod
    def _build_hit_fact(rule: RuleHit) -> str:
        """构建规则命中事实文本。"""
        if rule.current_value and rule.threshold:
            return f"{rule.reason}；当前值：{rule.current_value}；核验口径：{rule.threshold}"
        if rule.current_value:
            return f"{rule.reason}；当前值：{rule.current_value}"
        return rule.reason

    @staticmethod
    def _build_review_impact(rule: RuleHit) -> str:
        """构建对人工审核的影响说明。"""
        if rule.layer == LAYER_STRONG:
            return "属于重点核验线索，需由审核员重点确认。"
        if rule.layer == LAYER_DATA:
            return "属于数据质量或适用范围提示，需先确认材料完整性。"
        if rule.severity in {"high", "critical"}:
            return "可能影响初审判断，建议优先核验。"
        return "作为人工初审的关注项。"

    @staticmethod
    def _suggest_materials(rule: RuleHit) -> list[str]:
        """根据规则 ID 返回建议补充的材料列表。"""
        by_rule = {
            "OP-R001": ["就诊流水", "同日就诊明细", "跨机构就诊说明"],
            "OP-R002": ["结算单", "基金支付明细", "费用明细"],
            "OP-R003": ["处方记录", "药品费用清单", "长期用药材料"],
            "OP-R004": ["检查治疗项目明细", "诊疗摘要材料"],
            "OP-R005": ["挂号记录", "就诊流程材料"],
            "OP-R006": ["待遇资格材料", "补助支付明细"],
            "OP-R007": ["字段校验记录", "申报材料说明"],
            "OP-R008": ["按前序核验结果查看对应材料"],
            "OP-R009": ["备案记录", "急诊证明材料", "北京参保地待遇政策", "上海就医地目录"],
            "OP-R010": ["急诊病历", "急诊挂号记录", "急诊诊断章", "抢救记录"],
            "OP-R011": ["费用明细", "参保地待遇政策", "就医地药品和诊疗项目目录"],
            "OP-R012": ["政策生效时间", "政策废止状态", "规则库版本记录"],
        }
        return by_rule.get(rule.rule_id, ["相关申报材料"])

    @staticmethod
    def _unique_rules(rules: list[RuleHit]) -> list[RuleHit]:
        """按规则 ID 去重，保留首次出现的规则。"""
        unique: dict[str, RuleHit] = {}
        for rule in rules:
            unique[rule.rule_id] = rule
        return list(unique.values())
