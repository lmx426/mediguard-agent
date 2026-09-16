"""EvidenceService 单元测试。

验证：
- 三种不同风险等级案件的证据包生成
- 禁止词汇（"欺诈""拒赔""处罚"）不出现在 recommendation 中
- 引用完整性
"""

import pytest

from src.backend.domain.audit.review.entities import CaseDetail, FraudScreeningSignal, RuleHit
from src.backend.domain.audit.review.evidence_packager import EvidenceService, FORBIDDEN_ACTIONS


def _make_rule(
    rule_id: str,
    hit: bool = False,
    severity: str = "low",
    reason: str = "测试原因",
    layer: str = "risk_signal",
) -> RuleHit:
    return RuleHit(
        rule_id=rule_id,
        rule_name=f"测试规则{rule_id}",
        hit=hit,
        severity=severity,
        reason=reason,
        evidence_ref=f"rule:{rule_id}:test",
        version="1.0.0",
        layer=layer,
    )


def _make_case(
    case_id: str = "CASE-TEST",
    risk_score: float = 0.5,
    risk_level: str = "medium",
    rule_hits: list[RuleHit] | None = None,
) -> CaseDetail:
    hits = [rule for rule in (rule_hits or []) if rule.hit]
    if risk_level == "high" or len(hits) >= 2:
        review_priority = "high_priority"
    elif risk_level == "medium" or len(hits) == 1:
        review_priority = "manual_review"
    else:
        review_priority = "routine"
    return CaseDetail(
        case_id=case_id,
        case_title="测试案件",
        case_type="门诊",
        claim_summary="测试摘要",
        risk_score=risk_score,
        risk_level=risk_level,
        model_evidence_ref="model:score:v1.0",
        rule_hits=rule_hits or [],
        expected_recommendation="测试建议",
        model_signal_reasons=["基础异常筛查信号 0.10"],
        review_priority=review_priority,
    )


class TestEvidenceService:
    def setup_method(self):
        self.service = EvidenceService()

    # ---- 低风险案件 ----

    def test_low_risk_no_rules_hit(self):
        case = _make_case(
            risk_score=0.12,
            risk_level="low",
            rule_hits=[_make_rule("R001", hit=False)],
        )
        pkg = self.service.generate(case)

        assert pkg.case_id == "CASE-TEST"
        assert "风险提示强度 0.12" in pkg.risk_summary
        assert "0.12" in pkg.model_evidence
        assert len(pkg.citations) == 1  # 只有模型证据引用，无规则命中引用
        assert pkg.citations[0].source == "model_evidence"

    # ---- 高风险案件 ----

    def test_high_risk_multiple_rules_hit(self):
        case = _make_case(
            risk_score=0.87,
            risk_level="high",
            rule_hits=[
                _make_rule("R001", hit=True, severity="high", reason="单日金额异常"),
                _make_rule("R002", hit=True, severity="high", reason="自费比例高"),
                _make_rule("R003", hit=True, severity="medium", reason="费用分布异常"),
            ],
        )
        pkg = self.service.generate(case)

        assert "业务线索 3 类" in pkg.risk_summary
        assert len(pkg.rule_evidence) == 4  # 1 条 summary + 3 条规则
        assert len(pkg.rule_cards) == 3
        assert len(pkg.review_actions) >= 2
        assert pkg.rule_cards[0].suggested_action
        assert len(pkg.citations) == 4  # 1 模型 + 3 规则

    # ---- 证据不足案件 ----

    def test_medium_risk_insufficient_evidence(self):
        case = _make_case(
            risk_score=0.55,
            risk_level="medium",
            rule_hits=[
                _make_rule("R004", hit=True, severity="medium", reason="备案缺失"),
            ],
        )
        pkg = self.service.generate(case)

        assert len(pkg.missing_information) > 0
        assert any("仅发现 1 类业务线索" in m for m in pkg.missing_information)

    # ---- 安全边界 ----

    def test_recommendation_has_no_forbidden_words(self):
        """recommendation 不得包含"欺诈""拒赔""处罚"。"""
        case = _make_case(risk_score=0.87, risk_level="high")
        pkg = self.service.generate(case)

        forbidden = ["欺诈", "拒赔", "处罚"]
        for word in forbidden:
            assert word not in pkg.recommendation, (
                f"recommendation 包含禁止词汇: {word}"
            )

    def test_evidence_package_does_not_expose_res_field_name(self):
        """证据包不得暴露离线标签字段名。"""
        case = _make_case(risk_score=0.87, risk_level="high")
        pkg = self.service.generate(case)

        payload = " ".join(
            [
                pkg.risk_summary,
                pkg.model_evidence,
                pkg.recommendation,
                *pkg.rule_evidence,
                *pkg.missing_information,
                *(card.hit_fact for card in pkg.rule_cards),
                *(action.action for action in pkg.review_actions),
            ]
        )

        assert "RES" not in payload

    def test_forbidden_actions_always_present(self):
        case = _make_case()
        pkg = self.service.generate(case)

        assert len(pkg.forbidden_actions) == len(FORBIDDEN_ACTIONS)
        for action in FORBIDDEN_ACTIONS:
            assert action in pkg.forbidden_actions

    def test_citations_match_evidence(self):
        """每条 citation 都对应模型证据或命中规则。"""
        hit_rules = [
            _make_rule("R001", hit=True, severity="high"),
            _make_rule("R002", hit=False),
        ]
        case = _make_case(risk_score=0.8, risk_level="high", rule_hits=hit_rules)
        pkg = self.service.generate(case)

        # 1 模型引用 + 1 条命中规则引用 = 2
        assert len(pkg.citations) == 2
        assert pkg.citations[0].source == "model_evidence"
        assert pkg.citations[1].source == "rule_evidence"

    def test_unavailable_fraud_screening_is_not_cited_as_result(self):
        """未接入二分类模型时，证据包不输出有预警/无预警结论引用。"""
        case = _make_case(risk_score=0.95, risk_level="high")
        pkg = self.service.generate(case)

        assert case.fraud_screening.result == "not_available"
        assert all(citation.source != "fraud_screening" for citation in pkg.citations)

    def test_independent_fraud_screening_result_gets_own_citation(self):
        """真实二分类结果保留独立来源，并可参与综合风险提示强度。"""
        case = _make_case(risk_score=0.12, risk_level="low")
        case.fraud_screening = FraudScreeningSignal(
            result="suspected",
            label="有预警",
            source="独立二分类模型测试输出",
            evidence_ref="fraud_screening:binary:test",
            reason="测试结果",
            probability=0.783,
        )
        pkg = self.service.generate(case)

        assert any(citation.source == "fraud_screening" for citation in pkg.citations)
        assert "模型识别预警：有预警" in pkg.model_evidence
        assert "预警概率：78.3%" in pkg.model_evidence
        assert "测试结果" not in pkg.model_evidence

    # ---- 规则未命中时 ----

    def test_no_rules_hit_produces_correct_evidence(self):
        case = _make_case(
            risk_score=0.1,
            risk_level="low",
            rule_hits=[
                _make_rule("R001", hit=False, reason="费用在正常范围"),
                _make_rule("R002", hit=False, reason="自费比例正常"),
            ],
        )
        pkg = self.service.generate(case)

        assert any("没有形成需要单独核验的业务线索" in item for item in pkg.rule_evidence)
        assert pkg.rule_cards == []
        assert len(pkg.citations) == 1  # 仅模型引用
