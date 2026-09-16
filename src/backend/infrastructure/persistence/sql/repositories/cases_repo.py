"""PostgreSQL-backed case repository — cases_repo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from .....domain.audit.review.entities import (
    CaseDetail,
    CaseSummary,
    FraudScreeningSignal,
    PCAFeatureScore,
    RiskScoreBreakdown,
    RiskScoreComponent,
    RuleCheckItem,
    RuleHit,
)
from .....domain.audit.review.evidence_packager import EvidencePackage
from .....domain.audit.review.workflow_projector import TraceNode
from ..models import (
    CaseFingerprintORM,
    CaseORM,
    CaseSourceRecordORM,
    EvidencePackageORM,
    FraudModelResultORM,
    PCAFeatureScoreORM,
    RiskScoreComponentORM,
    RuleCheckItemORM,
    RuleResultORM,
    WorkflowTraceNodeORM,
)
from ..session import session_scope


class SqlCaseRepository:
    """Persist cases and generated artifacts in PostgreSQL."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def clear(self) -> None:
        """Delete case-owned runtime data. Intended for tests, not app shutdown."""

        with session_scope(self._session_factory) as session:
            session.execute(delete(CaseORM))

    @property
    def case_count(self) -> int:
        """Return total case count for health checks."""

        with self._session_factory() as session:
            return int(session.scalar(select(func.count(CaseORM.id))) or 0)

    def add_case(
        self,
        case: CaseDetail,
        fingerprint: str | None = None,
        evidence_package: EvidencePackage | None = None,
    ) -> None:
        """Persist one case and its deterministic analysis artifacts."""

        self.add_cases([(case, fingerprint, evidence_package)])

    def add_cases(
        self,
        entries: list[
            tuple[CaseDetail, str | None]
            | tuple[CaseDetail, str | None, EvidencePackage | None]
        ],
    ) -> None:
        """Persist multiple cases in one database transaction."""

        normalized = []
        for entry in entries:
            case, fingerprint, *optional = entry
            normalized.append((case, fingerprint, optional[0] if optional else None, None))
        self._add_cases_with_artifacts(normalized)

    def add_cases_with_artifacts(
        self,
        entries: list[
            tuple[
                CaseDetail,
                str | None,
                EvidencePackage | None,
                list[TraceNode],
            ]
        ],
    ) -> None:
        """Persist cases, evidence snapshots and Trace nodes atomically."""

        self._add_cases_with_artifacts(entries)

    def find_by_fingerprint(self, fingerprint: str) -> CaseDetail | None:
        """Return the existing case matching an ingest fingerprint."""

        with self._session_factory() as session:
            row = session.scalar(
                select(CaseFingerprintORM)
                .where(CaseFingerprintORM.fingerprint == fingerprint)
                .options(selectinload(CaseFingerprintORM.case))
            )
            if row is None:
                return None
            return self._to_detail(session, row.case)

    def list_cases(self, reviewed_case_ids: set[str] | None = None) -> list[CaseSummary]:
        """Return queue summaries without changing the public API shape."""

        reviewed_case_ids = reviewed_case_ids or set()
        with self._session_factory() as session:
            rows = session.scalars(select(CaseORM).order_by(CaseORM.created_at)).all()
            summaries: list[CaseSummary] = []
            for row in rows:
                summaries.append(
                    CaseSummary(
                        case_id=row.case_id,
                        case_title=row.case_title,
                        case_type=row.case_type,
                        risk_level=row.risk_level,
                        risk_score=row.risk_score,
                        review_status=(
                            "reviewed"
                            if row.case_id in reviewed_case_ids or row.review_status == "reviewed"
                            else "pending"
                        ),
                        rule_signal_count=self._rule_signal_count(row),
                        claim_amount=row.claim_amount,
                    )
                )
            return summaries

    def get_case(self, case_id: str) -> CaseDetail | None:
        """Return a case detail by public case id."""

        with self._session_factory() as session:
            row = self._load_case(session, case_id)
            return self._to_detail(session, row) if row is not None else None

    def _add_cases_with_artifacts(
        self,
        entries: list[
            tuple[
                CaseDetail,
                str | None,
                EvidencePackage | None,
                list[TraceNode] | None,
            ]
        ],
    ) -> None:
        with session_scope(self._session_factory) as session:
            for case, fingerprint, evidence_package, trace_nodes in entries:
                existing = session.scalar(
                    select(CaseORM).where(CaseORM.case_id == case.case_id)
                )
                if existing is not None:
                    session.delete(existing)
                    session.flush()

                case_row = self._from_detail(case)
                session.add(case_row)
                session.flush()

                session.add(
                    CaseSourceRecordORM(
                        case_id=case_row.id,
                        source_system=self._source_system(case),
                        record_version="claim-wide-v1",
                        source_record=self._source_record_with_context(case),
                        input_features=case.input_features,
                    )
                )
                if fingerprint:
                    session.add(
                        CaseFingerprintORM(
                            case_id=case_row.id,
                            fingerprint=fingerprint,
                            source_system=self._source_system(case),
                            record_version="claim-wide-v1",
                        )
                    )

                self._add_rule_results(session, case_row.id, case)
                self._add_fraud_result(session, case_row.id, case)
                self._add_pca_scores(session, case_row.id, case)
                self._add_risk_components(session, case_row.id, case)
                if evidence_package is not None:
                    self._add_evidence_package(session, case_row.id, evidence_package)
                if trace_nodes is not None:
                    self._add_trace_nodes(session, case_row.id, trace_nodes)

    def _load_case(self, session: Session, case_id: str) -> CaseORM | None:
        return session.scalar(
            select(CaseORM)
            .where(CaseORM.case_id == case_id)
            .options(
                selectinload(CaseORM.source_records),
                selectinload(CaseORM.rule_results).selectinload(RuleResultORM.check_items),
                selectinload(CaseORM.fraud_model_results),
                selectinload(CaseORM.pca_feature_scores),
                selectinload(CaseORM.risk_score_components),
            )
        )

    def _to_detail(self, session: Session, row: CaseORM) -> CaseDetail:
        row = self._load_case(session, row.case_id) or row
        source = row.source_records[0] if row.source_records else None
        source_record = dict(source.source_record if source else {})
        context_marker = source_record.pop("_case_context", None)
        case_context = context_marker if isinstance(context_marker, dict) else {}
        return CaseDetail(
            case_id=row.case_id,
            case_title=row.case_title,
            case_type=row.case_type,
            claim_summary=row.claim_summary,
            risk_level=row.risk_level,
            risk_score=row.risk_score,
            review_status=row.review_status,
            rule_signal_count=self._rule_signal_count(row),
            claim_amount=row.claim_amount,
            model_evidence_ref=row.model_evidence_ref,
            rule_hits=self._to_rule_hits(row),
            expected_recommendation=row.expected_recommendation,
            model_signal_source=row.model_signal_source,
            model_signal_reasons=row.model_signal_reasons or [],
            review_priority=row.review_priority,
            evidence_consistency=row.evidence_consistency,
            subject_ref=row.subject_ref,
            rule_pool_version=row.rule_pool_version,
            rule_baseline_version=row.rule_baseline_version,
            input_features=(source.input_features if source else {}),
            source_record=source_record,
            case_context=case_context,
            fraud_screening=self._to_fraud_signal(row),
            pca_feature_scores=self._to_pca_scores(row),
            risk_score_breakdown=self._to_risk_breakdown(row),
        )

    def _from_detail(self, case: CaseDetail) -> CaseORM:
        breakdown = case.risk_score_breakdown
        return CaseORM(
            case_id=case.case_id,
            case_title=case.case_title,
            case_type=case.case_type,
            claim_summary=case.claim_summary,
            subject_ref=case.subject_ref,
            risk_level=case.risk_level,
            risk_score=case.risk_score,
            review_status=case.review_status,
            review_priority=case.review_priority,
            evidence_consistency=case.evidence_consistency,
            model_signal_source=case.model_signal_source,
            model_evidence_ref=case.model_evidence_ref,
            model_signal_reasons=case.model_signal_reasons,
            rule_pool_version=case.rule_pool_version,
            rule_baseline_version=case.rule_baseline_version,
            expected_recommendation=case.expected_recommendation,
            claim_amount=self._claim_amount(case),
            risk_breakdown_total_score=breakdown.total_score if breakdown else None,
            risk_breakdown_max_score=breakdown.max_score if breakdown else None,
            risk_breakdown_display_score=breakdown.display_score if breakdown else None,
            risk_breakdown_level=breakdown.level if breakdown else None,
            risk_breakdown_level_label=breakdown.level_label if breakdown else None,
            risk_breakdown_cap_note=breakdown.cap_note if breakdown else None,
        )

    def _add_rule_results(self, session: Session, case_uuid: Any, case: CaseDetail) -> None:
        for position, rule in enumerate(case.rule_hits):
            rule_row = RuleResultORM(
                case_id=case_uuid,
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                hit=rule.hit,
                severity=rule.severity,
                reason=rule.reason,
                evidence_ref=rule.evidence_ref,
                version=rule.version,
                layer=rule.layer,
                action=rule.action,
                current_value=rule.current_value,
                threshold=rule.threshold,
                business_explanation=rule.business_explanation,
                position=position,
            )
            session.add(rule_row)
            session.flush()
            for item_position, item in enumerate(rule.check_items):
                session.add(
                    RuleCheckItemORM(
                        rule_result_id=rule_row.id,
                        label=item.label,
                        current_value=item.current_value,
                        threshold=item.threshold,
                        hit=item.hit,
                        layer=item.layer,
                        severity=item.severity,
                        explanation=item.explanation,
                        position=item_position,
                    )
                )

    def _add_fraud_result(self, session: Session, case_uuid: Any, case: CaseDetail) -> None:
        signal = case.fraud_screening
        session.add(
            FraudModelResultORM(
                case_id=case_uuid,
                result=signal.result,
                label=signal.label,
                source=signal.source,
                evidence_ref=signal.evidence_ref,
                reason=signal.reason,
                probability=signal.probability,
            )
        )

    def _add_pca_scores(self, session: Session, case_uuid: Any, case: CaseDetail) -> None:
        for position, score in enumerate(case.pca_feature_scores):
            session.add(
                PCAFeatureScoreORM(
                    case_id=case_uuid,
                    name=score.name,
                    score=score.score,
                    position=position,
                )
            )

    def _add_risk_components(self, session: Session, case_uuid: Any, case: CaseDetail) -> None:
        if case.risk_score_breakdown is None:
            return
        for position, component in enumerate(case.risk_score_breakdown.components):
            session.add(
                RiskScoreComponentORM(
                    case_id=case_uuid,
                    key=component.key,
                    label=component.label,
                    score=component.score,
                    max_score=component.max_score,
                    summary=component.summary,
                    details=component.details,
                    source_detail=component.source_detail,
                    position=position,
                )
            )

    def _add_evidence_package(
        self,
        session: Session,
        case_uuid: Any,
        evidence_package: EvidencePackage,
    ) -> None:
        session.add(
            EvidencePackageORM(
                case_id=case_uuid,
                risk_summary=evidence_package.risk_summary,
                model_evidence=evidence_package.model_evidence,
                recommendation=evidence_package.recommendation,
                payload=evidence_package.model_dump(mode="json"),
            )
        )

    def _add_trace_nodes(
        self,
        session: Session,
        case_uuid: Any,
        trace_nodes: list[TraceNode],
    ) -> None:
        for node in trace_nodes:
            session.add(
                WorkflowTraceNodeORM(
                    case_id=case_uuid,
                    node_id=node.node_id,
                    node_name=node.node_name,
                    status=node.status,
                    summary=node.summary,
                    node_order=node.order,
                    metadata_json=node.metadata,
                )
            )

    @staticmethod
    def _to_rule_hits(row: CaseORM) -> list[RuleHit]:
        rules = sorted(row.rule_results, key=lambda item: item.position)
        return [
            RuleHit(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                hit=rule.hit,
                severity=rule.severity,
                reason=rule.reason,
                evidence_ref=rule.evidence_ref,
                version=rule.version,
                layer=rule.layer,
                action=rule.action,
                current_value=rule.current_value,
                threshold=rule.threshold,
                business_explanation=rule.business_explanation,
                check_items=[
                    RuleCheckItem(
                        label=item.label,
                        current_value=item.current_value,
                        threshold=item.threshold,
                        hit=item.hit,
                        layer=item.layer,
                        severity=item.severity,
                        explanation=item.explanation,
                    )
                    for item in sorted(rule.check_items, key=lambda check: check.position)
                ],
            )
            for rule in rules
        ]

    @staticmethod
    def _to_fraud_signal(row: CaseORM) -> FraudScreeningSignal:
        if not row.fraud_model_results:
            return FraudScreeningSignal()
        signal = row.fraud_model_results[0]
        return FraudScreeningSignal(
            result=signal.result,
            label=signal.label,
            source=signal.source,
            evidence_ref=signal.evidence_ref,
            reason=signal.reason,
            probability=signal.probability,
        )

    @staticmethod
    def _to_pca_scores(row: CaseORM) -> list[PCAFeatureScore]:
        return [
            PCAFeatureScore(name=item.name, score=item.score)
            for item in sorted(row.pca_feature_scores, key=lambda score: score.position)
        ]

    @staticmethod
    def _to_risk_breakdown(row: CaseORM) -> RiskScoreBreakdown | None:
        if row.risk_breakdown_total_score is None:
            return None
        components = [
            RiskScoreComponent(
                key=item.key,
                label=item.label,
                score=item.score,
                max_score=item.max_score,
                summary=item.summary,
                details=item.details or [],
                source_detail=item.source_detail,
            )
            for item in sorted(row.risk_score_components, key=lambda component: component.position)
        ]
        return RiskScoreBreakdown(
            total_score=row.risk_breakdown_total_score,
            max_score=row.risk_breakdown_max_score or 100,
            display_score=row.risk_breakdown_display_score or f"{row.risk_breakdown_total_score} / 100",
            level=row.risk_breakdown_level or row.risk_level,
            level_label=row.risk_breakdown_level_label or row.risk_level,
            components=components,
            cap_note=row.risk_breakdown_cap_note,
        )

    @staticmethod
    def _rule_signal_count(row: CaseORM) -> int:
        return len([rule for rule in row.rule_results if rule.hit])

    @staticmethod
    def _claim_amount(case: CaseDetail) -> float | None:
        value = case.source_record.get("ALL_SUM") or case.input_features.get("ALL_SUM")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _source_system(case: CaseDetail) -> str:
        return "upstream_wide_record" if case.source_record else "feature_record"

    @staticmethod
    def _source_record_with_context(case: CaseDetail) -> dict:
        source_record = dict(case.source_record or {})
        if case.case_context:
            source_record["_case_context"] = case.case_context
        return source_record
