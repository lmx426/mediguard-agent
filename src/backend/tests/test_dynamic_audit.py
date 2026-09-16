"""单条统计记录动态稽核测试。"""

import pytest

from src.backend.application.intake.build_case_pipeline import AuditPipelineService
from src.backend.domain.audit.review.entities import FraudScreeningSignal, PCAFeatureScore
from src.backend.domain.intake.entities import FeatureRecordInput
from src.backend.domain.audit.review.evidence_packager import EvidenceService
from src.backend.domain.intake.validation_rules import (
    FeatureValidationError,
    FeatureValidationService,
)
from src.backend.domain.audit.review.risk_engine import OperationalRiskSignalProvider
from src.backend.domain.audit.review.rule_evaluator import RuleService
from src.backend.application.fraud.run_screening_uc import FraudModelResult
from src.backend.infrastructure.persistence.memory.case_repository import CaseService
from src.backend.infrastructure.persistence.memory.trace_repository import TraceService


def _features(**overrides):
    features = {
        "月就诊次数_MAX": 2,
        "月就诊医院数_MAX": 1,
        "一天去两家医院的天数": 0,
        "药品在总金额中的占比": 0.25,
        "检查总费用在总金额占比": 0.10,
        "治疗费用在总金额占比": 0.15,
        "是否挂号": 1,
        "ALL_SUM": 385,
        "药品费发生金额_SUM": 96.25,
        "检查费发生金额_SUM": 38.5,
        "治疗费发生金额_SUM": 57.75,
    }
    features.update(overrides)
    return features


class _StubFraudModelService:
    """测试用假模型服务，固定返回二分类预警和 10 个 PCA 分。"""

    def analyze(self, _record):
        return FraudModelResult(
            fraud_screening=FraudScreeningSignal(
                result="suspected",
                label="有预警",
                source="模型识别预警",
                evidence_ref="fraud_screening:binary:test",
                reason="测试输出",
                probability=0.783,
            ),
            pca_feature_scores=[
                PCAFeatureScore(name=f"PCA指标{index}", score=float(index))
                for index in range(1, 11)
            ],
        )


class TestFeatureValidationService:
    def setup_method(self):
        self.service = FeatureValidationService()

    def test_valid_feature_record(self):
        normalized = self.service.validate(FeatureRecordInput(features=_features()))

        assert normalized["月就诊次数_MAX"] == 2
        assert normalized["药品在总金额中的占比"] == 0.25

    def test_missing_required_field(self):
        features = _features()
        del features["ALL_SUM"]

        with pytest.raises(FeatureValidationError) as exc:
            self.service.validate(FeatureRecordInput(features=features))

        assert any("缺少必填字段：ALL_SUM" in e for e in exc.value.errors)

    def test_rejects_non_numeric_value(self):
        with pytest.raises(FeatureValidationError) as exc:
            self.service.validate(
                FeatureRecordInput(features=_features(ALL_SUM="not-a-number"))
            )

        assert any("字段必须为数值：ALL_SUM" in e for e in exc.value.errors)

    def test_rejects_ratio_out_of_range(self):
        with pytest.raises(FeatureValidationError) as exc:
            self.service.validate(
                FeatureRecordInput(features=_features(药品在总金额中的占比=1.2))
            )

        assert any("比例字段必须在 0 到 1 之间" in e for e in exc.value.errors)

    def test_rejects_negative_amount(self):
        with pytest.raises(FeatureValidationError) as exc:
            self.service.validate(
                FeatureRecordInput(features=_features(ALL_SUM=-1))
            )

        assert any("金额字段不得为负数：ALL_SUM" in e for e in exc.value.errors)

    def test_rejects_res_and_identity_fields(self):
        features = _features()
        features["RES"] = 1
        features["个人编码"] = "P001"

        with pytest.raises(FeatureValidationError) as exc:
            self.service.validate(FeatureRecordInput(features=features))

        assert any("RES" in e for e in exc.value.errors)
        assert any("个人编码" in e for e in exc.value.errors)


class TestDynamicScoringAndRules:
    def test_composite_risk_signal_uses_model_rules_peer_and_data_flow(self):
        provider = OperationalRiskSignalProvider()
        rules = RuleService().evaluate(
            _features(
                月就诊次数_MAX=22,
                月就诊医院数_MAX=5,
                一天去两家医院的天数=16,
                药品在总金额中的占比=0.96,
                治疗费用在总金额占比=0.50,
                ALL_SUM=80000,
                药品费发生金额_SUM=76000,
                治疗费发生金额_SUM=40000,
                是否挂号=0,
            )
        )
        high = provider.score(
            _features(
                月就诊次数_MAX=22,
                月就诊医院数_MAX=5,
                一天去两家医院的天数=16,
                药品在总金额中的占比=0.96,
                治疗费用在总金额占比=0.50,
                ALL_SUM=80000,
                药品费发生金额_SUM=76000,
                治疗费发生金额_SUM=40000,
                是否挂号=0,
            ),
            rules=rules,
            fraud_screening=FraudScreeningSignal(
                result="suspected",
                label="有预警",
                source="模型识别预警",
                evidence_ref="fraud_screening:binary:test",
                probability=0.783,
            ),
        )

        assert high.risk_score == 0.85
        assert high.risk_level == "high"
        assert high.risk_score_breakdown is not None
        assert high.risk_score_breakdown.components[0].label == "模型识别预警"
        assert high.risk_score_breakdown.components[0].score == 55
        assert high.risk_score_breakdown.components[0].max_score == 70
        assert high.risk_score_breakdown.components[1].score == 15
        assert high.risk_score_breakdown.components[1].max_score == 15
        assert high.risk_score_breakdown.components[2].score == 10
        assert high.risk_score_breakdown.components[2].max_score == 10

    def test_rule_service_returns_fixed_op_rule_pool(self):
        rules = RuleService().evaluate(
            _features(
                月就诊次数_MAX=22,
                月就诊医院数_MAX=5,
                一天去两家医院的天数=16,
                药品在总金额中的占比=0.96,
                治疗费用在总金额占比=0.50,
                ALL_SUM=80000,
                药品费发生金额_SUM=76000,
                治疗费发生金额_SUM=40000,
                是否挂号=0,
            )
        )

        assert [rule.rule_id for rule in rules] == [
            "OP-R001",
            "OP-R002",
            "OP-R003",
            "OP-R004",
            "OP-R005",
            "OP-R006",
            "OP-R007",
            "OP-R009",
            "OP-R010",
            "OP-R011",
            "OP-R012",
            "OP-R008",
        ]
        assert all(rule.version == "op-rule-pool-v0.3" for rule in rules)
        assert any(rule.layer == "strong_review_signal" for rule in rules)
        assert any(rule.action == "SPECIAL_AUDIT" for rule in rules)
        assert any(rule.current_value and rule.threshold for rule in rules)

    def test_old_standalone_drug_ratio_registration_and_check_ratio_do_not_hit(self):
        rules = RuleService().evaluate(
            _features(
                药品在总金额中的占比=0.91,
                检查总费用在总金额占比=0.70,
                是否挂号=0,
                ALL_SUM=1200,
                药品费发生金额_SUM=1092,
                检查费发生金额_SUM=840,
            )
        )

        by_id = {rule.rule_id: rule for rule in rules}
        assert not by_id["OP-R004"].hit
        assert not by_id["OP-R007"].hit
        assert "OP-R005" in by_id
        assert not any(rule.rule_id == "R003" for rule in rules)


class TestAuditPipelineService:
    def setup_method(self):
        self.case_service = CaseService()
        self.evidence_service = EvidenceService()
        self.trace_service = TraceService()
        self.pipeline = AuditPipelineService(
            case_service=self.case_service,
            evidence_service=self.evidence_service,
            trace_service=self.trace_service,
            validation_service=FeatureValidationService(),
            scoring_provider=OperationalRiskSignalProvider(),
            rule_service=RuleService(),
        )

    def test_pipeline_creates_case_evidence_and_trace(self):
        response = self.pipeline.run(
            FeatureRecordInput(
                case_title="测试动态案件",
                case_type="统计特征输入",
                features=_features(月就诊次数_MAX=12, 药品在总金额中的占比=0.91),
            )
        )

        assert response.case.case_id.startswith("CUSTOM-")
        assert self.case_service.get_case(response.case.case_id) is not None
        assert response.evidence.case_id == response.case.case_id
        assert response.evidence.citations[0].source == "model_evidence"

        assert "trace" not in response.model_dump()

        workflow = self.trace_service.build_workflow(response.case.case_id)
        assert [step.key for step in workflow.steps] == [
            "case_intake",
            "fact_base",
            "rule_check",
            "risk_screening",
            "evidence_package",
            "initial_review",
            "secondary_review",
            "appeal_handling",
            "case_result",
        ]
        assert workflow.current_step == "initial_review"
        assert workflow.steps[5].status == "current"
        assert workflow.steps[6].status == "conditional"
        assert workflow.steps[7].status == "conditional"
        assert workflow.steps[-1].status == "pending"
        rules_step = next(step for step in workflow.steps if step.key == "rule_check")
        assert rules_step.metadata["rule_pool_version"] == "op-rule-pool-v0.3"
        assert rules_step.metadata["total_rules"] == 12

    def test_fraud_screening_is_not_derived_from_risk_level(self):
        low = self.pipeline.run(FeatureRecordInput(features=_features()))
        high = self.pipeline.run(
            FeatureRecordInput(
                features=_features(
                    月就诊次数_MAX=22,
                    月就诊医院数_MAX=5,
                    一天去两家医院的天数=16,
                    药品在总金额中的占比=0.96,
                    治疗费用在总金额占比=0.50,
                    ALL_SUM=80000,
                    药品费发生金额_SUM=76000,
                    治疗费发生金额_SUM=40000,
                    是否挂号=0,
                )
            )
        )

        assert low.case.risk_level == "low"
        assert high.case.risk_level == "medium"
        assert low.case.fraud_screening.result == "not_available"
        assert high.case.fraud_screening.result == "not_available"

    def test_pipeline_evidence_recommendation_has_no_forbidden_words(self):
        response = self.pipeline.run(
            FeatureRecordInput(features=_features(月就诊次数_MAX=12))
        )

        for word in ["欺诈", "拒赔", "处罚"]:
            assert word not in response.evidence.recommendation

    def test_fraud_model_result_contributes_to_composite_risk_score(self):
        pipeline = AuditPipelineService(
            case_service=self.case_service,
            evidence_service=self.evidence_service,
            trace_service=self.trace_service,
            validation_service=FeatureValidationService(),
            scoring_provider=OperationalRiskSignalProvider(),
            rule_service=RuleService(),
            fraud_model_service=_StubFraudModelService(),
        )

        response = pipeline.run(FeatureRecordInput(features=_features()))

        assert response.case.risk_score == 0.6
        assert response.case.risk_level == "medium"
        assert response.case.fraud_screening.result == "suspected"
        assert response.case.fraud_screening.probability == 0.783
        assert response.case.risk_score_breakdown is not None
        assert response.case.risk_score_breakdown.components[0].score == 55
        assert response.case.risk_score_breakdown.components[0].max_score == 70
        assert len(response.case.pca_feature_scores) == 10
        assert "预警概率：78.3%" in response.evidence.model_evidence
        assert "PCA指标1" not in response.evidence.model_evidence

        trace = self.trace_service.get_trace(response.case.case_id)
        fraud_node = next(node for node in trace if node.node_id == "trace:fraud_model_screened")
        assert fraud_node.metadata == {
            "result": "suspected",
            "has_probability": True,
        }

    def test_peer_deviation_scans_all_safe_wide_record_fields(self):
        safe_record = {
            "个人编码": "SIM_PERSON_999999",
            **{
                field: 0
                for field in OperationalRiskSignalProvider().baseline.fields
            },
        }
        safe_record.update(
            {
                "ALL_SUM": 80000,
                "药品费发生金额_SUM": 76000,
                "药品在总金额中的占比": 0.96,
            }
        )

        scoring = OperationalRiskSignalProvider().score(
            _features(
                ALL_SUM=80000,
                药品费发生金额_SUM=76000,
                药品在总金额中的占比=0.96,
            ),
            source_record=safe_record,
        )

        assert scoring.risk_score_breakdown is not None
        peer = scoring.risk_score_breakdown.components[2]
        assert peer.key == "peer_deviation"
        assert "已扫描 80 项安全申报字段" in peer.summary
        assert "RES" not in str(peer.model_dump())
        assert "个人编码" not in str(peer.model_dump())
