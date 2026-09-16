from src.backend.domain.audit.review.entities import CaseDetail, RuleCheckItem, RuleHit
from src.backend.application.intake.build_case_pipeline import AuditPipelineService
from src.backend.infrastructure.persistence.memory.case_repository import CaseService


def _rule(rule_id: str, hit: bool = False, layer: str = "risk_signal") -> RuleHit:
    return RuleHit(
        rule_id=rule_id,
        rule_name=f"Rule {rule_id}",
        hit=hit,
        severity="medium",
        reason="test",
        evidence_ref=f"rule:{rule_id}:test",
        version="op-rule-pool-v0.1",
        layer=layer,
    )


def _case(case_id: str = "CASE-Q") -> CaseDetail:
    return CaseDetail(
        case_id=case_id,
        case_title="多机构就诊线索",
        case_type="统计特征输入",
        claim_summary="test",
        risk_score=0.73,
        risk_level="high",
        model_evidence_ref="model:signal:test",
        rule_hits=[_rule("OP-R001", hit=True), _rule("OP-R002", hit=False)],
        expected_recommendation="test",
        input_features={"ALL_SUM": 1200.5},
    )


def test_case_summary_includes_review_status_signal_count_and_claim_amount():
    service = CaseService()
    service.add_case(_case("CASE-PENDING"))
    service.add_case(_case("CASE-REVIEWED"))

    summaries = {
        item.case_id: item
        for item in service.list_cases(reviewed_case_ids={"CASE-REVIEWED"})
    }

    assert summaries["CASE-PENDING"].review_status == "pending"
    assert summaries["CASE-REVIEWED"].review_status == "reviewed"
    assert summaries["CASE-PENDING"].rule_signal_count == 1
    assert summaries["CASE-PENDING"].claim_amount == 1200.5


def test_case_title_generation_uses_business_priority():
    title = AuditPipelineService._build_case_title(
        [
            _rule("OP-R001", hit=True),
            _rule("OP-R002", hit=True),
        ]
    )

    assert title == "就诊行为一致性核验线索"


def test_case_title_generation_prefers_strong_review_signal():
    title = AuditPipelineService._build_case_title(
        [
            _rule("OP-R002", hit=True),
            _rule("OP-R003", hit=True, layer="strong_review_signal"),
        ]
    )

    assert title == "药品费用合理性核验线索"


def test_case_title_generation_falls_back_to_routine_title():
    title = AuditPipelineService._build_case_title([_rule("OP-R001", hit=False)])

    assert title == "常规门诊统计记录"


def test_case_title_generation_ignores_non_op_r008_data_notice():
    data_notice = _rule("OP-R003", hit=False, layer="data_quality_or_applicability")
    title = AuditPipelineService._build_case_title([data_notice])

    assert title == "常规门诊统计记录"


def test_case_title_generation_uses_material_data_quality_notice():
    data_notice = _rule("OP-R008", hit=False, layer="data_quality_or_applicability")
    data_notice.check_items = [
        RuleCheckItem(
            label="数据质量",
            current_value="异常",
            threshold="需关注",
            hit=True,
            layer="data_quality_or_applicability",
            severity="high",
            explanation="数据质量提示",
        )
    ]
    title = AuditPipelineService._build_case_title([data_notice])

    assert title == "材料补充提示"
