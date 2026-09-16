"""Deterministic Review Advisor projection for the public showcase."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json

from ..ports.repositories import CaseRepository
from ...domain.agent.entities import (
    AgentCitation,
    AgentClueReview,
    AgentEvidenceReview,
    AgentRun,
    AgentSignalReview,
    AgentStatement,
    AgentVerificationItem,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
    SystemRiskPrompt,
)
from ...domain.audit.review.evidence_packager import EvidenceService
from ...domain.audit.review.entities import AuthenticatedUser, CaseDetail, RuleHit


class ShowcaseReviewAdvisorService:
    """Expose a stable, labelled analysis without calling an LLM."""

    def __init__(self, cases: CaseRepository, evidence: EvidenceService) -> None:
        self._cases = cases
        self._evidence = evidence
        self._runs: dict[str, AgentRun] = {}

    def start(
        self,
        case_id: str,
        _request: EvidenceAgentRequest,
        actor: AuthenticatedUser,
    ) -> AgentRun:
        return self._build_run(case_id, actor.id)

    def get_run(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def get_latest_run(
        self,
        case_id: str,
        _analysis_type: str = "comprehensive",
    ) -> AgentRun | None:
        try:
            return self._build_run(case_id, "showcase")
        except KeyError:
            return None

    def get_current_analysis(
        self,
        case_id: str,
        _analysis_type: str = "comprehensive",
    ) -> EvidenceAgentAnalysis | None:
        run = self.get_latest_run(case_id, _analysis_type)
        return run.analysis if run is not None else None

    @staticmethod
    def list_events(_run_id: str, _after_sequence: int = 0) -> list:
        return []

    @staticmethod
    def set_caser_context_events(_events) -> None:
        return None

    @staticmethod
    def shutdown() -> None:
        return None

    def _build_run(self, case_id: str, actor_id: str) -> AgentRun:
        case = self._cases.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        analysis = self._build_analysis(case)
        now = analysis.created_at
        run = AgentRun(
            run_id=analysis.run_id,
            case_id=case_id,
            actor_id=actor_id,
            analysis_type="comprehensive",
            status=analysis.status,
            current_node="showcase_precomputed",
            input_fingerprint=analysis.input_fingerprint,
            reused=True,
            model_call_count=0,
            tool_call_count=0,
            created_at=now,
            updated_at=now,
            completed_at=now,
            analysis=analysis,
        )
        self._runs[run.run_id] = run
        return run

    def _build_analysis(self, case: CaseDetail) -> EvidenceAgentAnalysis:
        evidence = self._evidence.generate(case)
        citations, citation_ids = self._build_citations(case)
        triggered = self._triggered_rules(case)
        risk_refs = ["showcase:risk"]
        if case.fraud_screening.result != "not_available":
            risk_refs.append("showcase:model")
        risk_refs.extend(
            citation_ids[rule.evidence_ref]
            for rule in triggered
            if rule.evidence_ref in citation_ids
        )
        risk_refs = list(dict.fromkeys(risk_refs))[:8]

        relation, label, support = self._evidence_relation(case, triggered)
        summary = self._review_summary(case, triggered)
        clue_reviews = [
            AgentClueReview(
                clue_id=f"showcase:{rule.rule_id}",
                title=rule.rule_name,
                status="supported" if rule.hit else "needs_review",
                explanation=(
                    f"{rule.business_explanation or rule.reason} "
                    "该线索只用于人工核验，不构成处理结论。"
                ),
                source_refs=[citation_ids[rule.evidence_ref]],
            )
            for rule in triggered
            if rule.evidence_ref in citation_ids
        ]
        verification = [
            AgentVerificationItem(
                title=item.title,
                action=item.action,
                rationale=item.rationale,
                relation_type="risk_score_related",
                priority=(
                    "high"
                    if case.risk_level == "high"
                    else "medium"
                    if case.risk_level == "medium"
                    else "low"
                ),
                source_refs=self._map_refs(item.source_refs, citation_ids, risk_refs),
            )
            for item in evidence.review_actions
        ]
        missing_details = [
            AgentStatement(statement=item, source_refs=risk_refs)
            for item in evidence.missing_information
        ]
        system_prompt = self._system_risk_prompt(case, risk_refs)
        fingerprint = sha256(
            json.dumps(
                case.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        now = datetime.now(timezone.utc)
        suffix = case.case_id.lower().replace("_", "-")
        ai_label = (
            "存在疑似欺诈风险线索"
            if case.fraud_screening.result == "suspected" and any(rule.hit for rule in triggered)
            else "未发现明确疑似欺诈风险线索"
            if case.fraud_screening.result == "not_suspected"
            else "当前证据不足"
        )
        supported = [
            AgentStatement(
                statement=f"{item.title}：{item.explanation}",
                source_refs=item.source_refs,
            )
            for item in clue_reviews
            if item.status == "supported"
        ]
        needs_review = [
            AgentStatement(
                statement=f"{item.title}：{item.explanation}",
                source_refs=item.source_refs,
            )
            for item in clue_reviews
            if item.status != "supported"
        ]
        return EvidenceAgentAnalysis(
            analysis_id=f"showcase-analysis-{suffix}",
            run_id=f"showcase-run-{suffix}",
            case_id=case.case_id,
            analysis_type="comprehensive",
            status="complete",
            system_risk_prompt=system_prompt,
            evidence_review=AgentEvidenceReview(
                relation=relation,
                label=label,
                support_level=support,
                summary=summary,
                source_refs=risk_refs,
            ),
            clue_reviews=clue_reviews,
            ai_risk_label=ai_label,
            ai_risk_label_source_refs=risk_refs,
            risk_judgement=summary,
            risk_judgement_source_refs=risk_refs,
            evidence_strength=support,
            evidence_strength_source_refs=risk_refs,
            key_risk_signals=supported,
            human_review_focus=verification,
            risk_overview=summary,
            risk_overview_source_refs=risk_refs,
            supporting_evidence=supported,
            conflicts=[],
            missing_information=list(evidence.missing_information),
            missing_information_details=missing_details,
            signal_review=AgentSignalReview(
                case_review_hint=summary,
                case_review_hint_source_refs=risk_refs,
                supported_clues=supported,
                needs_review=needs_review,
                unconfirmed_items=list(evidence.missing_information),
                supplementary_review_hints=[],
            ),
            verification_checklist=verification,
            citations=citations,
            boundary_notice=(
                "该内容为面试展示版预生成结果，未调用实时模型；"
                "只整理现有脱敏事实和确定性规则，不替代人工审核决定。"
            ),
            generated_notice="预生成演示，未调用实时模型",
            input_fingerprint=fingerprint,
            model_name="showcase-precomputed",
            prompt_version="showcase-review-advisor-v1",
            tool_version="showcase-read-only-v1",
            created_at=now,
        )

    @staticmethod
    def _triggered_rules(case: CaseDetail) -> list[RuleHit]:
        return [
            rule
            for rule in case.rule_hits
            if rule.hit or any(item.hit for item in rule.check_items)
        ]

    @staticmethod
    def _evidence_relation(
        case: CaseDetail,
        triggered: list[RuleHit],
    ) -> tuple[str, str, str]:
        risk_hits = [rule for rule in triggered if rule.hit]
        if case.fraud_screening.result == "suspected" and risk_hits:
            return "supports", "支持", "高"
        if risk_hits:
            return "partially_supports", "部分支持", "中"
        if case.fraud_screening.result == "not_suspected":
            return "supports", "支持", "中"
        return "insufficient_evidence", "证据不足", "证据不足"

    @staticmethod
    def _review_summary(case: CaseDetail, triggered: list[RuleHit]) -> str:
        risk_hits = [rule for rule in triggered if rule.hit]
        score = (
            case.risk_score_breakdown.display_score
            if case.risk_score_breakdown
            else f"{round(case.risk_score * 100)} / 100"
        )
        if risk_hits:
            names = "、".join(rule.rule_name for rule in risk_hits[:3])
            return (
                f"系统综合风险提示强度为 {score}，模型与确定性规则共同形成核验线索。"
                f"当前主要由{name}提供支撑，仍需结合材料由审核员确认。"
            )
        return (
            f"系统综合风险提示强度为 {score}，当前未形成明确规则风险命中。"
            "现有证据支持按常规流程或抽样口径处理，最终意见仍由审核员确认。"
        )

    @staticmethod
    def _system_risk_prompt(case: CaseDetail, refs: list[str]) -> SystemRiskPrompt:
        breakdown = case.risk_score_breakdown
        score = breakdown.total_score if breakdown else round(case.risk_score * 100)
        level = case.risk_level if case.risk_level in {"high", "medium", "low", "insufficient"} else "insufficient"
        labels = {"high": "高风险", "medium": "中风险", "low": "低风险", "insufficient": "证据不足"}
        return SystemRiskPrompt(
            risk_level=level,
            risk_level_label=breakdown.level_label if breakdown else labels[level],
            risk_score=score,
            score_text=breakdown.display_score if breakdown else f"{score} / 100",
            source_summary=(
                [item.label for item in breakdown.components if item.score > 0]
                if breakdown
                else ["确定性风险信号"]
            ),
            source_refs=refs,
        )

    @staticmethod
    def _build_citations(
        case: CaseDetail,
    ) -> tuple[list[AgentCitation], dict[str, str]]:
        citations = [
            AgentCitation(
                citation_id="showcase:risk",
                source_type="risk_breakdown",
                source_ref=case.model_evidence_ref,
                label="系统综合风险提示",
                current_value=(
                    case.risk_score_breakdown.display_score
                    if case.risk_score_breakdown
                    else f"{round(case.risk_score * 100)} / 100"
                ),
                metadata={"showcase": True},
            )
        ]
        mapping = {case.model_evidence_ref: "showcase:risk"}
        if case.fraud_screening.result != "not_available":
            citations.append(
                AgentCitation(
                    citation_id="showcase:model",
                    source_type="model_warning",
                    source_ref=case.fraud_screening.evidence_ref,
                    label=f"模型识别预警：{case.fraud_screening.label}",
                    current_value=(
                        f"{case.fraud_screening.probability:.1%}"
                        if case.fraud_screening.probability is not None
                        else case.fraud_screening.label
                    ),
                    metadata={"showcase": True, "precomputed": True},
                )
            )
            mapping[case.fraud_screening.evidence_ref] = "showcase:model"
        for rule in ShowcaseReviewAdvisorService._triggered_rules(case):
            citation_id = f"showcase:rule:{rule.rule_id}"
            citations.append(
                AgentCitation(
                    citation_id=citation_id,
                    source_type="rule_result",
                    source_ref=rule.evidence_ref,
                    label=f"[{rule.rule_id}] {rule.rule_name}",
                    version=rule.version,
                    current_value=rule.current_value,
                    threshold=rule.threshold,
                    metadata={"showcase": True},
                )
            )
            mapping[rule.evidence_ref] = citation_id
        return citations, mapping

    @staticmethod
    def _map_refs(
        refs: list[str],
        mapping: dict[str, str],
        fallback: list[str],
    ) -> list[str]:
        resolved = [mapping[ref] for ref in refs if ref in mapping]
        return list(dict.fromkeys(resolved or fallback))[:8]
