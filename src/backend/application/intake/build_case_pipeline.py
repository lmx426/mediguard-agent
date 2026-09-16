"""案件接入与确定性分析用例。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..ports.repositories import (
    CaseRepository,
    NoteRepository,
    ReviewRepository,
    TraceRepository,
)
from ...constants.ingest_fields import INGEST_RECORD_FIELDS
from ...domain.audit.review.entities import CaseDetail
from ...domain.intake.entities import FeatureRecordInput, IngestRecordInput
from ...domain.audit.review.workflow_projector import CaseFullResponse, TraceNode
from ...domain.audit.review.evidence_packager import EvidenceService
from ...domain.intake.validation_rules import FeatureValidationService, IngestValidationError, IngestValidationService
from ...domain.audit.review.risk_engine import OperationalRiskSignalProvider
from ...domain.audit.review.rule_evaluator import LAYER_DATA, LAYER_STRONG, RuleService
from ..fraud.run_screening_uc import FraudModelResult, FraudModelService

if TYPE_CHECKING:
    from ..agent.case_agent.context.refresh_service import CaserContextRefreshService


class _UnavailableFraudModelService:
    """测试或未配置场景下的空模型服务。"""

    @staticmethod
    def analyze(_record: dict[str, float | str] | None) -> FraudModelResult:
        return FraudModelService._unavailable("not_configured")


@dataclass(frozen=True)
class _PreparedCase:
    """尚未写入仓储的完整案件产物。"""

    response: CaseFullResponse
    trace_nodes: list[TraceNode]
    fingerprint: str | None = None


class AuditPipelineService:
    """将脱敏统计记录转成可审核案件。

    类名暂时保留以兼容既有测试和调用方；它在新架构中属于应用用例，
    不再承担路由或基础设施职责。
    """

    TITLE_BY_RULE_ID = {
        "OP-R001": "就诊行为一致性核验线索",
        "OP-R002": "费用与支付结构核验线索",
        "OP-R003": "药品费用合理性核验线索",
        "OP-R004": "检查治疗结构核验线索",
        "OP-R005": "挂号与流程一致性核验线索",
        "OP-R006": "待遇补助口径提示",
        "OP-R007": "申报数据质量提示",
        "OP-R008": "材料补充提示",
        "OP-R009": "异地手工报销备案核验提示",
        "OP-R010": "急诊身份材料核验提示",
        "OP-R011": "参保地待遇与就医地目录口径核验提示",
        "OP-R012": "政策版本冲突核验提示",
    }
    RULE_TITLE_PRIORITY = [
        "OP-R001",
        "OP-R002",
        "OP-R003",
        "OP-R004",
        "OP-R005",
        "OP-R009",
        "OP-R010",
        "OP-R011",
        "OP-R012",
        "OP-R008",
        "OP-R007",
        "OP-R006",
    ]
    LAYER_TITLE_PRIORITY = {
        LAYER_STRONG: 0,
        "risk_signal": 1,
        LAYER_DATA: 2,
    }

    def __init__(
        self,
        case_service: CaseRepository,
        evidence_service: EvidenceService,
        trace_service: TraceRepository,
        validation_service: FeatureValidationService,
        scoring_provider: OperationalRiskSignalProvider,
        rule_service: RuleService,
        fraud_model_service: FraudModelService | None = None,
        review_repository: ReviewRepository | None = None,
        note_repository: NoteRepository | None = None,
    ) -> None:
        self.case_service = case_service
        self.evidence_service = evidence_service
        self.trace_service = trace_service
        self.validation_service = validation_service
        self.scoring_provider = scoring_provider
        self.rule_service = rule_service
        self.fraud_model_service = fraud_model_service or _UnavailableFraudModelService()
        self.review_repository = review_repository
        self.note_repository = note_repository
        self._caser_context_events: CaserContextRefreshService | None = None

    def set_caser_context_events(
        self,
        events: CaserContextRefreshService | None,
    ) -> None:
        """Attach the optional Caser context refresh publisher."""

        self._caser_context_events = events

    def run(
        self,
        record: FeatureRecordInput,
        source: str = "json",
        source_record: dict[str, float | str] | None = None,
        upstream_trace: bool = False,
        case_context: dict[str, Any] | None = None,
    ) -> CaseFullResponse:
        """处理兼容统计特征输入并提交案件。"""

        prepared = self._prepare(
            record,
            source=source,
            source_record=source_record,
            upstream_trace=upstream_trace,
            case_context=case_context,
        )
        self._commit([prepared])
        return prepared.response

    def _prepare(
        self,
        record: FeatureRecordInput,
        source: str,
        source_record: dict[str, float | str] | None,
        upstream_trace: bool,
        case_context: dict[str, Any] | None = None,
        fingerprint: str | None = None,
    ) -> _PreparedCase:
        """完成全部校验和分析，但不修改任何运行时仓储。"""

        features = self.validation_service.validate(record)
        safe_context = self._safe_case_context(case_context)
        rules = self.rule_service.evaluate(
            features,
            source_record=source_record,
            case_context=safe_context,
        )
        fraud_model_result = self.fraud_model_service.analyze(source_record)
        scoring = self.scoring_provider.score(
            features,
            source_record=source_record,
            rules=rules,
            fraud_screening=fraud_model_result.fraud_screening,
        )
        risk_hit_count = len([r for r in rules if r.hit])
        review_priority = self._derive_review_priority(scoring.risk_level, rules)
        evidence_consistency = self._build_evidence_consistency(scoring.risk_level, risk_hit_count)
        subject_ref = self._extract_subject_ref(source_record)

        case_id = record.case_id or self._new_case_id(source)
        case = CaseDetail(
            case_id=case_id,
            case_title=record.case_title or self._build_case_title(rules),
            case_type=record.case_type or "统计特征输入",
            claim_summary=self._build_claim_summary(
                features,
                source,
                source_record,
                safe_context,
            ),
            risk_score=scoring.risk_score,
            risk_level=scoring.risk_level,
            model_evidence_ref=scoring.evidence_ref,
            rule_hits=rules,
            expected_recommendation="由风险提示强度和规则命中生成审核建议。",
            model_signal_source=scoring.source,
            model_signal_reasons=scoring.reasons,
            review_priority=review_priority,
            evidence_consistency=evidence_consistency,
            subject_ref=subject_ref,
            rule_pool_version=rules[0].version if rules else "op-rule-pool-v0.1",
            rule_baseline_version=self.rule_service.baseline.version,
            input_features=features,
            source_record=source_record or {},
            case_context=safe_context,
            fraud_screening=fraud_model_result.fraud_screening,
            pca_feature_scores=fraud_model_result.pca_feature_scores,
            risk_score_breakdown=scoring.risk_score_breakdown,
        )

        trace_nodes = self._build_initial_trace(
            case,
            features,
            scoring.source,
            risk_hit_count,
            upstream_trace=upstream_trace,
            source_record=source_record,
            case_context=safe_context,
        )
        evidence = self.evidence_service.generate(case)
        return _PreparedCase(
            response=CaseFullResponse(
                case=case,
                evidence=evidence,
                review=None,
            ),
            trace_nodes=trace_nodes,
            fingerprint=fingerprint,
        )

    def run_ingest(
        self,
        body: IngestRecordInput,
        ingest_validation_service: IngestValidationService,
    ) -> CaseFullResponse:
        """从完整脱敏宽表记录幂等生成动态稽核案件。"""

        safe_record = ingest_validation_service.validate(body)
        fingerprint = self._ingest_fingerprint(body, safe_record)
        existing = self.case_service.find_by_fingerprint(fingerprint)
        if existing is not None:
            return self._existing_response(existing)

        prepared = self._prepare_safe_ingest(body, safe_record, fingerprint)
        self._commit([prepared])
        return prepared.response

    def run_ingest_batch(
        self,
        bodies: list[IngestRecordInput],
        ingest_validation_service: IngestValidationService,
    ) -> list[CaseFullResponse]:
        """从批量完整脱敏宽表记录生成动态稽核案件。"""
        if not bodies:
            raise IngestValidationError(["报表至少需要包含 1 条完整脱敏宽表记录"])

        safe_records: list[
            tuple[IngestRecordInput, dict[str, float | str], str]
        ] = []
        errors: list[str] = []
        seen_fingerprints: dict[str, int] = {}

        for index, body in enumerate(bodies, start=1):
            try:
                safe_record = ingest_validation_service.validate(body)
            except IngestValidationError as exc:
                errors.extend([f"第 {index} 行：{error}" for error in exc.errors])
                continue

            fingerprint = self._ingest_fingerprint(body, safe_record)
            if fingerprint in seen_fingerprints:
                errors.append(
                    f"第 {index} 行：与第 {seen_fingerprints[fingerprint]} 行为完全重复记录"
                )
            else:
                seen_fingerprints[fingerprint] = index
            safe_records.append((body, safe_record, fingerprint))

        if errors:
            raise IngestValidationError(errors)

        responses: list[CaseFullResponse] = []
        prepared_cases: list[_PreparedCase] = []
        for body, safe_record, fingerprint in safe_records:
            existing = self.case_service.find_by_fingerprint(fingerprint)
            if existing is not None:
                responses.append(self._existing_response(existing))
                continue

            prepared = self._prepare_safe_ingest(body, safe_record, fingerprint)
            prepared_cases.append(prepared)
            responses.append(prepared.response)

        # 所有行均已成功构建后才统一提交，保证批量请求不会出现部分案件。
        self._commit(prepared_cases)
        return responses

    def _prepare_safe_ingest(
        self,
        body: IngestRecordInput,
        safe_record: dict[str, float | str],
        fingerprint: str,
    ) -> _PreparedCase:
        """从已校验宽表构建案件产物，但不写入仓储。"""

        features = {
            field: float(safe_record[field])
            for field in self.validation_service.get_schema().required_fields
        }
        subject_ref = str(safe_record["个人编码"])
        feature_record = FeatureRecordInput(
            case_title=body.case_title or f"上游申报记录 {subject_ref}",
            case_type=body.case_type or "上游宽表记录接入",
            features=features,
        )
        return self._prepare(
            feature_record,
            source=body.source_system or "upstream",
            source_record=safe_record,
            upstream_trace=True,
            case_context=body.case_context,
            fingerprint=fingerprint,
        )

    def _commit(self, prepared_cases: list[_PreparedCase]) -> None:
        """将已完整构建的案件和 Trace 一次性提交到内存仓储。"""

        if not prepared_cases:
            return
        atomic_commit = getattr(self.case_service, "add_cases_with_artifacts", None)
        if callable(atomic_commit):
            atomic_commit(
                [
                    (
                        item.response.case,
                        item.fingerprint,
                        item.response.evidence,
                        item.trace_nodes,
                    )
                    for item in prepared_cases
                ]
            )
            self._publish_caser_events(prepared_cases)
            return
        self.case_service.add_cases(
            [
                (item.response.case, item.fingerprint, item.response.evidence)
                for item in prepared_cases
            ]
        )
        self.trace_service.init_many(
            [
                (item.response.case.case_id, item.trace_nodes)
                for item in prepared_cases
            ]
        )
        self._publish_caser_events(prepared_cases)

    def _publish_caser_events(self, prepared_cases: list[_PreparedCase]) -> None:
        if self._caser_context_events is None:
            return
        for item in prepared_cases:
            case_id = item.response.case.case_id
            self._caser_context_events.publish(
                "case.ingested",
                case_id,
                reason="case committed",
            )
            self._caser_context_events.publish(
                "evidence.built",
                case_id,
                reason="base evidence package committed",
            )

    def _existing_response(self, case: CaseDetail) -> CaseFullResponse:
        """返回已存在案件的当前状态，不重新计算或覆盖人工结果。"""

        review = (
            self.review_repository.get(case.case_id)
            if self.review_repository is not None
            else None
        )
        notes = (
            self.note_repository.list_for_case(case.case_id)
            if self.note_repository is not None
            else []
        )
        return CaseFullResponse(
            case=case,
            evidence=self.evidence_service.generate(case),
            review=review,
            notes=notes,
        )

    @staticmethod
    def _ingest_fingerprint(
        body: IngestRecordInput,
        safe_record: dict[str, float | str],
    ) -> str:
        """生成稳定接入指纹。

        标题不影响业务数据身份；来源系统、记录版本和固定顺序的 81 字段共同
        决定是否为同一条接入记录。
        """

        payload = {
            "source_system": body.source_system or "医保结算申报系统",
            "record_version": body.record_version or "claim-wide-v1",
            "case_context": AuditPipelineService._safe_case_context(body.case_context),
            "record": [
                [field, safe_record[field]]
                for field in INGEST_RECORD_FIELDS
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_case_id(source: str) -> str:
        if "上游" in source or "医保结算" in source or source == "upstream":
            prefix = "INGEST"
        else:
            prefix = "CUSTOM"
        return f"{prefix}-{uuid4().hex[:8].upper()}"

    @classmethod
    def _build_case_title(cls, rules) -> str:
        def sort_key(rule) -> tuple[int, int]:
            layer_rank = cls.LAYER_TITLE_PRIORITY.get(rule.layer, 9)
            try:
                rule_rank = cls.RULE_TITLE_PRIORITY.index(rule.rule_id)
            except ValueError:
                rule_rank = 99
            return layer_rank, rule_rank

        risk_triggered = [rule for rule in rules if rule.hit]
        if risk_triggered:
            primary = sorted(risk_triggered, key=sort_key)[0]
            return cls.TITLE_BY_RULE_ID.get(primary.rule_id, primary.rule_name)

        data_quality_triggered = [
            rule
            for rule in rules
            if rule.rule_id == "OP-R008"
            and any(
                item.hit
                and item.layer == LAYER_DATA
                and item.severity in {"medium", "high", "critical"}
                for item in rule.check_items
            )
        ]
        if data_quality_triggered:
            primary = data_quality_triggered[0]
            return cls.TITLE_BY_RULE_ID.get(primary.rule_id, primary.rule_name)

        return "常规门诊统计记录"

    @staticmethod
    def _build_claim_summary(
        features: dict[str, float],
        source: str,
        source_record: dict[str, float | str] | None = None,
        case_context: dict[str, Any] | None = None,
    ) -> str:
        subject_part = ""
        if source_record and "个人编码" in source_record:
            subject_part = f"脱敏申报人编号={source_record['个人编码']}，"
        context_part = AuditPipelineService._context_summary(case_context)
        return (
            f"该案件由{source}单条脱敏统计特征记录生成。"
            f"{subject_part}"
            f"{context_part}"
            f"月就诊次数_MAX={features['月就诊次数_MAX']:g}，"
            f"月就诊医院数_MAX={features['月就诊医院数_MAX']:g}，"
            f"一天去两家医院的天数={features['一天去两家医院的天数']:g}，"
            f"ALL_SUM={features['ALL_SUM']:g}，"
            f"药品费用占比={features['药品在总金额中的占比']:.2f}。"
        )

    def _build_initial_trace(
        self,
        case: CaseDetail,
        features: dict[str, float],
        scoring_source: str,
        hit_count: int,
        upstream_trace: bool = False,
        source_record: dict[str, float | str] | None = None,
        case_context: dict[str, Any] | None = None,
    ) -> list[TraceNode]:
        record_field_count = len(source_record or features)
        subject_ref = self._extract_subject_ref(source_record) or "未提供"
        data_notice_count = len(
            [
                r for r in case.rule_hits
                if any(item.hit and item.layer == LAYER_DATA for item in r.check_items)
            ]
        )
        strong_count = len([r for r in case.rule_hits if r.layer == LAYER_STRONG])
        actions = sorted(
            {
                r.action
                for r in case.rule_hits
                if r.hit or any(item.hit and item.layer == LAYER_DATA for item in r.check_items)
            }
        )
        return [
            TraceNode(
                node_id="trace:case_received",
                node_name="案件接收",
                status="completed",
                summary=f"接收并生成案件 {case.case_id}：{case.case_title}",
                order=1,
                metadata={
                    "source": "upstream_wide_record" if upstream_trace else "legacy_feature_record",
                    "subject_ref": subject_ref,
                    "field_count": record_field_count,
                },
            ),
            TraceNode(
                node_id="trace:data_integrity_validated",
                node_name="数据完整性校验",
                status="completed",
                summary=(
                    "完整宽表字段、禁止字段和合成申报人编号校验通过"
                    if upstream_trace
                    else "兼容统计特征字段、禁止字段和数值范围校验通过"
                ),
                order=2,
                metadata={
                    "has_wide_record": upstream_trace,
                    "forbidden_fields_blocked": True,
                    "rule_pool_scope": "门诊/普通结算统计异常",
                },
            ),
            TraceNode(
                node_id="trace:rules_evaluated",
                node_name="规则核验",
                status="completed",
                summary=(
                    f"按 {case.rule_pool_version} 检查 {len(case.rule_hits)} 类业务核验场景，"
                    f"业务线索命中 {hit_count} 类，重点核验线索 {strong_count} 类"
                ),
                order=3,
                metadata={
                    "rule_pool_version": case.rule_pool_version,
                    "rule_baseline_version": case.rule_baseline_version,
                    "total_rules": len(case.rule_hits),
                    "risk_signal_count": hit_count,
                    "strong_review_signal_count": strong_count,
                    "data_quality_or_applicability_count": data_notice_count,
                    "suggested_actions": actions,
                },
            ),
            TraceNode(
                node_id="trace:risk_screened",
                node_name="风险筛查",
                status="completed",
                summary=(
                    f"{scoring_source} 生成风险提示强度 "
                    f"{case.risk_score_breakdown.display_score if case.risk_score_breakdown else format(case.risk_score, '.2f')}，"
                    f"风险等级：{case.risk_level}"
                ),
                order=4,
                metadata={
                    "model_signal_source": scoring_source,
                    "risk_score": case.risk_score,
                    "risk_level": case.risk_level,
                    "model_signal_reasons": case.model_signal_reasons,
                    "risk_score_breakdown": (
                        case.risk_score_breakdown.model_dump()
                        if case.risk_score_breakdown
                        else None
                    ),
                },
            ),
            TraceNode(
                node_id="trace:fraud_model_screened",
                node_name="模型识别预警",
                status="completed",
                summary=f"模型识别预警结果：{case.fraud_screening.label}",
                order=5,
                metadata={
                    "result": case.fraud_screening.result,
                    "has_probability": case.fraud_screening.probability is not None,
                },
            ),
            TraceNode(
                node_id="trace:evidence_organized",
                node_name="证据整理",
                status="completed",
                summary="证据包生成器基于风险提示和规则结果整理证据摘要",
                order=6,
                metadata={
                    "evidence_consistency": case.evidence_consistency,
                    "review_priority": case.review_priority,
                    "agent_mode": "deterministic_evidence_generator",
                    "case_context_keys": sorted((case_context or {}).keys()),
                },
            ),
            TraceNode(
                node_id="trace:manual_review_pending",
                node_name="待人工初审",
                status="pending",
                summary="等待审核员提交处理意见",
                order=7,
                metadata={"manual_review_required": True},
            ),
        ]

    @staticmethod
    def _extract_subject_ref(
        source_record: dict[str, float | str] | None,
    ) -> str | None:
        if source_record and "个人编码" in source_record:
            return str(source_record["个人编码"])
        return None

    @staticmethod
    def _safe_case_context(context: dict[str, Any] | None) -> dict[str, Any]:
        forbidden_exact = {"RES"}
        forbidden_fragments = ("\u59d3\u540d", "\u8eab\u4efd\u8bc1", "\u7535\u8bdd", "\u4f4f\u5740")
        if not isinstance(context, dict):
            return {}

        def scrub(value: Any) -> Any:
            if isinstance(value, dict):
                cleaned: dict[str, Any] = {}
                for key, item in value.items():
                    text_key = str(key)
                    if (
                        text_key.upper() in forbidden_exact
                        or any(token in text_key for token in forbidden_fragments)
                    ):
                        continue
                    cleaned[text_key] = scrub(item)
                return cleaned
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value

        return scrub(context)

    @staticmethod
    def _context_summary(context: dict[str, Any] | None) -> str:
        if not context:
            return ""
        parts = []
        mapping = (
            ("insured_region", "参保地"),
            ("treatment_region", "就医地"),
            ("visit_type", "就医类型"),
            ("claim_mode_label", "报销方式"),
            ("claim_mode", "报销方式"),
            ("filing_status", "备案状态"),
            ("direct_settlement", "是否直接结算"),
        )
        seen_labels: set[str] = set()
        for key, label in mapping:
            if label in seen_labels or key not in context:
                continue
            value = context.get(key)
            if value in (None, "", [], {}):
                continue
            parts.append(f"{label}={value}")
            seen_labels.add(label)
        if not parts:
            return ""
        return "业务上下文：" + "，".join(parts) + "。"

    @staticmethod
    def _derive_review_priority(risk_level: str, rules: list) -> str:
        if any(rule.layer == LAYER_STRONG for rule in rules):
            return "special_audit"
        risk_hit_count = len([rule for rule in rules if rule.hit])
        if risk_hit_count >= 2 or risk_level == "high":
            return "high_priority"
        if risk_hit_count == 1 or risk_level == "medium":
            return "manual_review"
        return "routine"

    @staticmethod
    def _build_evidence_consistency(risk_level: str, risk_hit_count: int) -> str:
        model_attention = risk_level in ("high", "medium")
        if model_attention and risk_hit_count >= 2:
            return "综合风险提示与多项规则线索一致，建议人工重点核验。"
        if model_attention and risk_hit_count == 0:
            return "综合风险提示关注，但规则证据不足，建议人工抽查。"
        if not model_attention and risk_hit_count > 0:
            return "综合风险提示较低，但规则发现业务线索，建议人工关注。"
        return "综合风险提示和规则核验均未发现明显异常，建议常规处理或抽样核验。"
