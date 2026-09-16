"""Resolve Policy RAG MCP filters for L3 Expert Analysis.

The resolver mirrors the goldset philosophy: filters are explicit business
retrieval boundaries, not free-form model output. Offline goldsets can derive
filters from known evidence metadata; online L3 must infer them from the user
question, semantic hints, case facts, and caller-provided task filters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.backend.application.agent.expert_agent.service.policy_filter_contract import (
    PolicyFilters,
    PolicyRetrievalPlanDraft,
    repair_json_object,
)


SUPPORTED_CONTENT_TYPES = {
    "policy_text",
    "table_row",
}

SUPPORTED_POLICY_DOMAINS = {
    "benefit",
    "chronic_disease_long_prescription",
    "designated_institution",
    "drug_catalog",
    "drug_product_price_reference",
    "emergency",
    "fund_supervision",
    "manual_reimbursement",
    "medical_service_price",
    "remote_medical",
    "remote_medical_manual_reimbursement",
    "shanghai_payment_scope",
    "special_disease_filing",
    "special_disease_scope",
}

SUPPORTED_JURISDICTIONS = {
    "national",
    "beijing",
    "shanghai",
}


@dataclass(slots=True)
class PolicyFilterResolverInput:
    user_question: str
    semantic_frame: dict[str, Any] = field(default_factory=dict)
    fact_bundle: list[dict[str, Any]] = field(default_factory=list)
    case_state_summary: dict[str, Any] = field(default_factory=dict)
    task_filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyFilterResolverResult:
    policy_question: str
    filters: dict[str, Any]
    filter_source: str
    confidence: float
    warnings: list[str] = field(default_factory=list)


class PolicyFilterOutputError(ValueError):
    """Structured LLM filter output could not satisfy the registered contract."""

    def __init__(self, message: str, *, errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.errors = errors or [{"type": "invalid_filter_output", "msg": message}]


class PolicyFilterResolver:
    """Infer and normalize the MCP retrieval boundary for a policy question."""

    def resolve_llm_plan(
        self,
        raw_output: str | dict[str, Any],
    ) -> dict[str, Any]:
        """Repair and validate one complete LLM retrieval-plan submission.

        This is the production path for generated filters.  It does not infer,
        merge, or append metadata values after the model has selected them.
        """

        try:
            payload = repair_json_object(raw_output)
            plan = PolicyRetrievalPlanDraft.model_validate(payload)
        except ValidationError as exc:
            raise PolicyFilterOutputError(
                "LLM policy filter output failed Pydantic validation",
                errors=exc.errors(include_url=False, include_input=False),
            ) from exc
        except Exception as exc:
            raise PolicyFilterOutputError(str(exc)) from exc
        return plan.model_dump(mode="json")

    def resolve_validated_filters(
        self,
        filters: dict[str, Any],
        *,
        policy_question: str,
        confidence: float = 0.8,
        filter_source: str = "llm_registered",
        allow_broad: bool = False,
    ) -> PolicyFilterResolverResult:
        """Normalize a validated filter set without adding business labels."""

        try:
            normalized = normalize_policy_filters(filters)
            if allow_broad and not normalized.get("jurisdiction") and not normalized.get("policy_domain"):
                validated = PolicyFilters.model_construct(
                    jurisdiction=list(normalized.get("jurisdiction") or []),
                    policy_domain=list(normalized.get("policy_domain") or []),
                    content_type=list(normalized.get("content_type") or []),
                    can_cite_as_policy_basis=bool(
                        normalized.get("can_cite_as_policy_basis", True)
                    ),
                )
            else:
                validated = PolicyFilters.model_validate(normalized)
        except ValidationError as exc:
            raise PolicyFilterOutputError(
                "Policy filters failed Pydantic validation",
                errors=exc.errors(include_url=False, include_input=False),
            ) from exc
        return PolicyFilterResolverResult(
            policy_question=str(policy_question or "").strip(),
            filters=validated.model_dump(mode="json"),
            filter_source=filter_source,
            confidence=max(0.0, min(float(confidence), 1.0)),
            warnings=[],
        )

    def resolve(self, payload: PolicyFilterResolverInput) -> PolicyFilterResolverResult:
        question = str(payload.user_question or "").strip()
        source_parts: list[str] = []
        warnings: list[str] = []

        filters = normalize_policy_filters(payload.task_filters)
        if _force_policy_filters(payload.task_filters) and filters:
            if "content_type" not in filters:
                _apply_content_type_defaults(question, filters)
            _apply_contract_defaults(filters)
            return PolicyFilterResolverResult(
                policy_question=_policy_question_for_template(question),
                filters=normalize_policy_filters(filters),
                filter_source="task_filters:forced",
                confidence=1.0,
                warnings=[],
            )
        if filters:
            source_parts.append("task_filters")

        semantic_filters = _filters_from_semantic_frame(payload.semantic_frame)
        if semantic_filters:
            filters = merge_policy_filters(filters, semantic_filters)
            source_parts.append("semantic_frame")

        case_filters = _filters_from_case_context(
            fact_bundle=payload.fact_bundle,
            case_state_summary=payload.case_state_summary,
            question=question,
        )
        if case_filters:
            filters = merge_policy_filters(filters, case_filters)
            source_parts.append("case_context")

        question_filters = _filters_from_question(question)
        if question_filters:
            filters = merge_policy_filters(filters, question_filters)
            source_parts.append("question_rules")

        template = _goldset_style_template(question, filters)
        if template is not None:
            previous_domains = set(_string_list(filters.get("policy_domain")))
            filters = template["filters"]
            source_parts.append(f"template:{template['name']}")
            removed_domains = previous_domains.difference(
                set(_string_list(filters.get("policy_domain")))
            )
            if removed_domains:
                warnings.append(
                    "template_narrowed_policy_domain:"
                    + ",".join(sorted(removed_domains))
                )

        filters = normalize_policy_filters(filters)
        _apply_question_domain_constraints(question, filters)
        _apply_content_type_defaults(question, filters)
        _apply_contract_defaults(filters)
        warnings.extend(_filter_warnings(filters))

        confidence = _confidence_for(filters, source_parts, warnings)
        policy_question = _policy_question_for_template(question)
        return PolicyFilterResolverResult(
            policy_question=policy_question,
            filters=filters,
            filter_source="+".join(dict.fromkeys(source_parts)) or "unresolved",
            confidence=confidence,
            warnings=list(dict.fromkeys(warnings)),
        )


def normalize_policy_filters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    filters: dict[str, Any] = {}
    for key in ("jurisdiction", "policy_domain", "content_type"):
        values = _string_list(value.get(key))
        if key == "jurisdiction":
            values = [
                jurisdiction
                for item in values
                if (jurisdiction := _normalize_jurisdiction(item))
            ]
        if key == "policy_domain":
            values = [
                domain
                for item in values
                for domain in _normalize_policy_domain(item)
            ]
        if key == "content_type":
            values = [
                content_type
                for item in values
                for content_type in _normalize_content_type(item)
            ]
        if values:
            filters[key] = list(dict.fromkeys(values))
    cite = value.get("can_cite_as_policy_basis")
    if cite is not None:
        filters["can_cite_as_policy_basis"] = _bool(cite, bool(cite))
    return filters


def merge_policy_filters(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    extra = extra or {}
    for key in ("jurisdiction", "policy_domain", "content_type"):
        values = _string_list(merged.get(key)) + _string_list(extra.get(key))
        if key == "jurisdiction":
            values = [
                jurisdiction
                for item in values
                if (jurisdiction := _normalize_jurisdiction(item))
            ]
        if key == "policy_domain":
            values = [
                domain
                for item in values
                for domain in _normalize_policy_domain(item)
            ]
        if key == "content_type":
            values = [
                content_type
                for item in values
                for content_type in _normalize_content_type(item)
            ]
        if values:
            merged[key] = list(dict.fromkeys(values))
    cite = extra.get("can_cite_as_policy_basis", merged.get("can_cite_as_policy_basis"))
    if cite is not None:
        merged["can_cite_as_policy_basis"] = _bool(cite, bool(cite))
    return merged


def _filters_from_semantic_frame(frame: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(frame, dict):
        return {}
    filters: dict[str, Any] = {}
    slots = frame.get("slots") if isinstance(frame.get("slots"), dict) else {}
    values = [
        frame.get("policy_domain"),
        frame.get("scenario"),
        frame.get("action"),
        slots.get("policy_domain"),
        slots.get("scenario"),
    ]
    domains: list[str] = []
    for value in values:
        domains.extend(_normalize_policy_domain(str(value or "")))
    if domains:
        filters["policy_domain"] = list(dict.fromkeys(domains))
    jurisdictions: list[str] = []
    for value in (frame.get("jurisdiction"), slots.get("jurisdiction")):
        jurisdictions.extend(_string_list(value))
    if jurisdictions:
        filters["jurisdiction"] = jurisdictions
    return normalize_policy_filters(filters)


def _filters_from_case_context(
    *,
    fact_bundle: list[dict[str, Any]],
    case_state_summary: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    raw_text = " ".join(
        [
            json.dumps(fact_bundle, ensure_ascii=False, default=str),
            json.dumps(case_state_summary, ensure_ascii=False, default=str),
        ]
    )
    filters: dict[str, Any] = {}
    jurisdictions: list[str] = []
    if any(token in raw_text for token in ("参保地", "insured_region", "北京", "北京市")):
        if "北京" in raw_text or "北京市" in raw_text or "beijing" in raw_text:
            jurisdictions.append("beijing")
    if any(token in raw_text for token in ("就医地", "treatment_region", "上海", "上海市")):
        if "上海" in raw_text or "上海市" in raw_text or "shanghai" in raw_text:
            jurisdictions.append("shanghai")
    if any(token in question + raw_text for token in ("跨省", "异地")):
        jurisdictions.insert(0, "national")
    if jurisdictions:
        filters["jurisdiction"] = list(dict.fromkeys(jurisdictions))

    domains: list[str] = []
    scope_text = question + raw_text
    special_disease_query = _looks_like_special_disease_query(question)
    if any(token in scope_text for token in ("跨省", "异地")) or (
        "备案" in scope_text and not special_disease_query
    ):
        domains.append("remote_medical")
    if special_disease_query:
        domains.extend(["special_disease_filing", "special_disease_scope"])
    if any(token in raw_text for token in ("manual_reimbursement", "手工报销", "零星报销")):
        domains.append("manual_reimbursement")
    if domains:
        filters["policy_domain"] = list(dict.fromkeys(domains))
    return normalize_policy_filters(filters)


def _filters_from_question(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {"can_cite_as_policy_basis": True}
    jurisdictions: list[str] = []
    if any(token in question for token in ("北京", "北京市")):
        jurisdictions.append("beijing")
    if any(token in question for token in ("上海", "上海市")):
        jurisdictions.append("shanghai")
    if any(token in question for token in ("国家", "全国", "跨省", "异地")):
        jurisdictions.insert(0, "national")

    domains: list[str] = []
    if _looks_like_drug_price_reference_query(question):
        return {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_product_price_reference"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": False,
        }
    if _looks_like_remote_benefit_split_query(question):
        return {
            "jurisdiction": list(dict.fromkeys(jurisdictions or ["national"])),
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        }
    if _looks_like_remote_emergency_manual_flow_query(question):
        return {
            "jurisdiction": list(dict.fromkeys(jurisdictions or ["national"])),
            "policy_domain": [
                "remote_medical",
                "remote_medical_manual_reimbursement",
                "manual_reimbursement",
                "emergency",
            ],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        }
    if _looks_like_manual_reimbursement_scenario_query(question):
        return {
            "jurisdiction": list(dict.fromkeys(jurisdictions or ["beijing"])),
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["table_row", "policy_text"],
            "can_cite_as_policy_basis": True,
        }
    if _looks_like_shanghai_payment_scope_verification_query(question):
        return {
            "jurisdiction": ["shanghai"],
            "policy_domain": [
                "drug_catalog",
                "medical_service_price",
                "shanghai_payment_scope",
            ],
            "content_type": ["table_row", "policy_text"],
            "can_cite_as_policy_basis": True,
        }
    if _looks_like_manual_reimbursement_service_query(question):
        result = {
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        }
        if jurisdictions:
            result["jurisdiction"] = list(dict.fromkeys(jurisdictions))
        return result
    if _looks_like_remote_settlement_management_query(question):
        domains.append("remote_medical")
    if _looks_like_shanghai_service_facility_scope_query(question):
        jurisdictions = ["shanghai"]
        domains.append("shanghai_payment_scope")
    if _looks_like_designated_institution_query(question):
        domains.append("designated_institution")
    if _looks_like_fund_supervision_query(question):
        domains.append("fund_supervision")
    if _looks_like_chronic_long_prescription_query(question):
        domains.append("chronic_disease_long_prescription")
    if _looks_like_benefit_policy_query(question):
        domains.append("benefit")
    if _looks_like_special_disease_filing_query(question):
        domains.append("special_disease_filing")
    if _looks_like_special_disease_scope_query(question):
        domains.append("special_disease_scope")
    if _looks_like_negotiated_drug_double_channel_query(question):
        domains.append("drug_catalog")
        if "上海" in question or "上海市" in question:
            domains.append("shanghai_payment_scope")
    if _looks_like_special_disease_query(question):
        domains.extend(["special_disease_filing", "special_disease_scope"])
        if any(token in question for token in ("长期处方", "长处方", "续方")):
            domains.append("chronic_disease_long_prescription")
    if (
        not _looks_like_remote_settlement_management_query(question)
        and (
            any(token in question for token in ("异地", "跨省"))
            or (
        "备案" in question and not _looks_like_special_disease_query(question)
            )
        )
    ):
        domains.extend(["remote_medical", "remote_medical_manual_reimbursement"])
    if any(token in question for token in ("手工报销", "零星报销", "报销材料")):
        domains.append("manual_reimbursement")
    if any(token in question for token in ("急诊", "门急诊")) and not _looks_like_shanghai_service_facility_scope_query(question):
        domains.append("emergency")
    if _looks_like_consumable_payment_scope_query(question):
        domains.append("shanghai_payment_scope")
    if _looks_like_drug_catalog_query(question):
        domains.append("drug_catalog")
    if _looks_like_medical_service_price_query(question):
        domains.append("medical_service_price")
    if jurisdictions:
        filters["jurisdiction"] = list(dict.fromkeys(jurisdictions))
    if domains:
        filters["policy_domain"] = list(dict.fromkeys(domains))
    return normalize_policy_filters(filters)


def _goldset_style_template(
    question: str,
    filters: dict[str, Any],
) -> dict[str, Any] | None:
    if _looks_like_remote_benefit_split_query(question):
        jurisdictions = _string_list(filters.get("jurisdiction"))
        normalized = [
            jurisdiction
            for item in jurisdictions
            if (jurisdiction := _normalize_jurisdiction(item))
        ]
        if "national" not in normalized:
            normalized.insert(0, "national")
        return {
            "name": "remote_benefit_split",
            "filters": {
                "jurisdiction": list(dict.fromkeys(normalized or ["national"])),
                "policy_domain": ["remote_medical"],
                "content_type": ["policy_text"],
                "can_cite_as_policy_basis": True,
            },
        }
    if _looks_like_remote_emergency_manual_flow_query(question):
        jurisdictions = _string_list(filters.get("jurisdiction"))
        normalized = [
            jurisdiction
            for item in jurisdictions
            if (jurisdiction := _normalize_jurisdiction(item))
        ]
        if "national" not in normalized:
            normalized.insert(0, "national")
        return {
            "name": "remote_emergency_manual_flow",
            "filters": {
                "jurisdiction": list(dict.fromkeys(normalized or ["national"])),
                "policy_domain": [
                    "remote_medical",
                    "remote_medical_manual_reimbursement",
                    "manual_reimbursement",
                    "emergency",
                ],
                "content_type": ["policy_text", "table_row"],
                "can_cite_as_policy_basis": True,
            },
        }
    if _looks_like_manual_reimbursement_scenario_query(question):
        explicit_jurisdictions = _explicit_jurisdictions_from_question(question)
        return {
            "name": "manual_reimbursement_scenario_slots",
            "filters": {
                "jurisdiction": explicit_jurisdictions or ["beijing"],
                "policy_domain": ["manual_reimbursement"],
                "content_type": ["table_row", "policy_text"],
                "can_cite_as_policy_basis": True,
            },
        }
    if _looks_like_shanghai_payment_scope_verification_query(question):
        return {
            "name": "shanghai_payment_scope_verification",
            "filters": {
                "jurisdiction": ["shanghai"],
                "policy_domain": [
                    "drug_catalog",
                    "medical_service_price",
                    "shanghai_payment_scope",
                ],
                "content_type": ["table_row", "policy_text"],
                "can_cite_as_policy_basis": True,
            },
        }
    if _looks_like_manual_reimbursement_service_query(question):
        explicit_jurisdictions = _explicit_jurisdictions_from_question(question)
        filters_for_template: dict[str, Any] = {
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        }
        if explicit_jurisdictions:
            filters_for_template["jurisdiction"] = explicit_jurisdictions
        return {
            "name": "manual_reimbursement_service_guide",
            "filters": filters_for_template,
        }
    return None


def _apply_content_type_defaults(question: str, filters: dict[str, Any]) -> None:
    existing = _string_list(filters.get("content_type"))
    domains = set(_string_list(filters.get("policy_domain")))
    required: list[str] = []
    if _looks_like_manual_reimbursement_service_query(question):
        required.append("policy_text")
    if "drug_product_price_reference" in domains:
        required = ["table_row"]
    elif "drug_catalog" in domains:
        if _looks_like_drug_catalog_policy_text_query(question):
            required.append("policy_text")
        elif _looks_like_drug_catalog_table_row_query(question):
            required.append("table_row")
        elif not existing:
            required.extend(["table_row", "policy_text"])
    if "medical_service_price" in domains:
        if _looks_like_table_row_fact_query(question):
            required.append("table_row")
        elif not existing:
            required.extend(["table_row", "policy_text"])
    if "designated_institution" in domains:
        required.append("table_row")
    if "shanghai_payment_scope" in domains:
        if _looks_like_consumable_payment_scope_query(question) and _looks_like_table_row_fact_query(question):
            required.extend(["table_row", "policy_text"])
        else:
            required.append("policy_text")
    if domains.intersection(
        {
            "benefit",
            "chronic_disease_long_prescription",
            "fund_supervision",
            "manual_reimbursement",
            "remote_medical",
            "remote_medical_manual_reimbursement",
            "special_disease_filing",
            "special_disease_scope",
        }
    ):
        required.append("policy_text")
    if not existing and not required:
        if _looks_like_remote_benefit_split_query(question):
            required.append("policy_text")
        elif _looks_like_remote_emergency_manual_flow_query(question):
            required.extend(["policy_text", "table_row"])
        elif domains:
            required.extend(["policy_text", "table_row"])
    if existing or required:
        ordered = [
            item for item in dict.fromkeys(required + existing)
            if item in SUPPORTED_CONTENT_TYPES
        ]
        if (
            "table_row" in ordered
            and (
                _looks_like_manual_reimbursement_scenario_query(question)
                or _looks_like_shanghai_payment_scope_verification_query(question)
                or _looks_like_table_row_fact_query(question)
            )
        ):
            ordered = ["table_row", *[item for item in ordered if item != "table_row"]]
        filters["content_type"] = ordered


def _apply_question_domain_constraints(question: str, filters: dict[str, Any]) -> None:
    domains = _string_list(filters.get("policy_domain"))
    if not domains:
        return
    compact = re.sub(r"\s+", "", str(question or ""))
    if (
        _looks_like_special_disease_query(compact)
        and not any(token in compact for token in ("跨省", "异地就医", "异地"))
    ):
        domains = [
            domain
            for domain in domains
            if domain not in {"remote_medical", "remote_medical_manual_reimbursement"}
        ]
        domains.extend(["special_disease_filing", "special_disease_scope"])
    filters["policy_domain"] = list(dict.fromkeys(domains))


def _apply_contract_defaults(filters: dict[str, Any]) -> None:
    if "can_cite_as_policy_basis" not in filters:
        filters["can_cite_as_policy_basis"] = True


def _filter_warnings(filters: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not filters.get("jurisdiction") and not filters.get("policy_domain"):
        warnings.append("filters_too_broad")
    if len(_string_list(filters.get("policy_domain"))) > 3:
        warnings.append("policy_domain_broad")
    if len(_string_list(filters.get("jurisdiction"))) > 3:
        warnings.append("jurisdiction_broad")
    return warnings


def _confidence_for(
    filters: dict[str, Any],
    source_parts: list[str],
    warnings: list[str],
) -> float:
    if "template:remote_benefit_split" in source_parts:
        return 0.95
    if "template:remote_emergency_manual_flow" in source_parts:
        return 0.92
    if "template:manual_reimbursement_scenario_slots" in source_parts:
        return 0.91
    if "template:shanghai_payment_scope_verification" in source_parts:
        return 0.9
    if "template:manual_reimbursement_service_guide" in source_parts:
        return 0.88
    if "question_rules" in source_parts and filters.get("policy_domain"):
        return 0.86 if filters.get("jurisdiction") else 0.78
    if "case_context" in source_parts and filters.get("jurisdiction"):
        return 0.82
    if "task_filters" in source_parts and not warnings:
        return 0.8
    if filters.get("policy_domain") or filters.get("jurisdiction"):
        return 0.65
    return 0.35


def _policy_question_for_template(question: str) -> str:
    if _looks_like_chronic_long_prescription_query(question) and _looks_like_benefit_policy_query(question):
        return (
            "基本医保待遇和慢性病长处方组合核验：城乡居民医保参保范围、新生儿待遇等待期、"
            "家庭医生首诊转诊、外省市目录标准、慢病药品品种规格衔接、BJ-GBI医事服务费损失补偿、月度通报"
        )
    if _looks_like_remote_settlement_management_query(question):
        return (
            "跨省异地就医直接结算管理规则：银行手续费和银行票据工本费不得从医保基金列支、"
            "预付金黄色预警红色预警和紧急调增、一次性跨省住院总费用费用协查"
        )
    if _looks_like_shanghai_service_facility_scope_query(question):
        return "上海基本医疗保险医疗服务设施范围：住院床位费、急诊观察室床位费、基金支付范围和政策有效期"
    if _looks_like_fund_supervision_query(question):
        return "医保基金使用监督管理规则：不属于基金支付范围、拒不配合调查暂停联网结算、涉嫌骗取医保基金处理程序"
    if _looks_like_chronic_long_prescription_query(question):
        return "北京门诊慢性病长处方政策：慢病药品品种规格衔接、BJ-GBI医事服务费损失补偿、月度通报、高血压糖尿病按人头付费"
    if _looks_like_benefit_policy_query(question):
        return "基本医保待遇和参保规则：城乡居民医保参保范围、新生儿待遇等待期、外埠户籍配偶材料、家庭医生首诊转诊、外省市目录标准"
    if _looks_like_special_disease_filing_query(question):
        return "北京门诊特殊病备案规则：特殊病种备案申报表、本市和异地备案办理路径、住院期间限制、外埠户籍连续缴费条件、病种名称调整"
    if _looks_like_special_disease_scope_query(question):
        return "北京门诊特殊疾病范围规则：新增门诊特殊疾病病种、报销范围、备案审核后享受待遇、未备案或非选定机构费用不纳入"
    if _looks_like_negotiated_drug_double_channel_query(question):
        return "医保药品目录谈判药品和双通道规则：协议期内谈判药品乙类管理、电子处方、不得受一品两规药占比总额限制影响"
    if _looks_like_remote_benefit_split_query(question):
        return (
            "跨省异地就医就医地目录和参保地待遇分工规则："
            "就医地支付范围、参保地起付标准、支付比例、最高支付限额"
        )
    if _looks_like_remote_emergency_manual_flow_query(question):
        return (
            "北京参保人异地急诊、备案和手工报销处理流程："
            "备案状态、急诊例外、直接结算、手工报销材料、参保地经办口径"
        )
    if _looks_like_manual_reimbursement_scenario_query(question):
        return (
            "北京手工报销场景化材料和结算凭证：外埠就医材料、"
            "定点医药机构记账结算凭证、急诊就医未出示社保卡或医保电子凭证、"
            "个人全额垫付、收据、处方、诊断证明、单位或便民服务中心到区医保经办机构手工报销"
        )
    if _looks_like_shanghai_payment_scope_verification_query(question):
        return (
            "上海医保支付范围核验：药品目录医保类别和限定支付范围、"
            "胸部 CT 医疗服务项目支付范围、医用耗材纳入医保支付范围和支付办法"
        )
    if _looks_like_manual_reimbursement_service_query(question):
        return "零星报销或手工报销门诊费用的必要材料和法定办结时限服务指南依据"
    if _looks_like_designated_institution_query(question):
        return question[:1200]
    return question[:1200]


def _looks_like_remote_benefit_split_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_remote_context = any(token in compact for token in ("跨省", "异地就医", "异地"))
    has_split_parties = "就医地" in compact and "参保地" in compact
    has_split_topic = any(
        token in compact
        for token in (
            "目录",
            "支付范围",
            "待遇",
            "支付比例",
            "起付",
            "最高支付限额",
            "分工",
            "怎么分",
            "如何分",
            "区分",
        )
    )
    return has_remote_context and has_split_parties and has_split_topic


def _looks_like_remote_emergency_manual_flow_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_remote_context = any(token in compact for token in ("跨省", "异地就医", "异地"))
    has_process_intent = any(
        token in compact
        for token in ("如何处理", "怎么处理", "怎么办", "处理流程", "办理", "路径", "怎么走")
    )
    topic_count = sum(
        bool(any(token in compact for token in group))
        for group in (
            ("急诊", "急诊抢救", "门急诊"),
            ("备案", "补备案"),
            ("手工报销", "零星报销", "报销"),
        )
    )
    return has_remote_context and has_process_intent and topic_count >= 2


def _looks_like_shanghai_payment_scope_verification_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or "上海" not in compact:
        return False
    has_scope_intent = any(
        token in compact
        for token in ("支付范围", "医保支付", "报销范围", "如何核验", "怎么核验", "核验")
    )
    topic_count = sum(
        bool(any(token in compact for token in group))
        for group in (
            ("药品", "药物", "用药", "药品目录"),
            ("CT", "ct", "胸部CT", "胸部ct", "检查", "影像"),
            ("医用耗材", "耗材"),
        )
    )
    return has_scope_intent and topic_count >= 2


def _looks_like_manual_reimbursement_service_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_reimbursement = any(
        token in compact
        for token in ("零星报销", "手工报销", "门诊费用报销", "门诊医疗费用报销", "门诊报销")
    )
    has_service_need = any(
        token in compact
        for token in (
            "必要材料",
            "申请材料",
            "材料清单",
            "需要哪些材料",
            "应要求提供哪些",
            "法定办结时限",
            "办结时限",
            "办理时限",
            "多少工作日",
            "服务指南",
            "办事指南",
        )
    )
    return has_reimbursement and has_service_need


def _looks_like_manual_reimbursement_scenario_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    scenario_count = sum(
        bool(any(token in compact for token in group))
        for group in (
            ("外埠就医", "外地就医", "易地安置"),
            ("记账结算", "记帐结算", "定点医药机构", "定点医疗机构", "定点零售药店"),
            ("急诊就医", "急诊留观", "未出示社保卡", "医保电子凭证", "没带医保凭证"),
        )
    )
    asks_material_or_voucher = any(token in compact for token in ("材料", "结算凭证", "审核结算凭证", "核验"))
    return scenario_count >= 2 and asks_material_or_voucher


def _looks_like_designated_institution_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "定点医疗机构",
            "定点医药机构",
            "定点零售药店",
            "定点药店",
            "定点医院",
            "机构编码",
            "药店编码",
            "医保定点",
            "社区卫生服务",
            "医务室",
        )
    )


def _looks_like_fund_supervision_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "基金监管",
            "监督检查",
            "重点监督检查",
            "拒不配合",
            "暂停联网结算",
            "锁卡",
            "骗取基金",
            "涉嫌骗保",
            "违法违规",
            "不属于基金支付范围",
            "不属于医疗保障基金支付范围",
            "追回基金",
            "异常情形审核",
        )
    )


def _looks_like_chronic_long_prescription_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "长期处方",
            "长处方",
            "慢性病长",
            "慢病长",
            "慢性病患者",
            "高血压",
            "糖尿病",
            "BJ-GBI",
            "医事服务费",
            "月度通报",
            "按人头付费",
            "品种规格",
            "品规",
            "医联体",
        )
    )


def _looks_like_benefit_policy_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "城乡居民医保",
            "城乡居民基本医疗保险",
            "城乡老年人",
            "参保范围",
            "参保资格",
            "新生儿",
            "等待期",
            "待遇起始",
            "外埠户籍配偶",
            "家庭医生",
            "首诊转诊",
            "外省市目录",
            "待遇享受",
        )
    )


def _looks_like_special_disease_filing_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "特殊病备案",
            "特殊病种备案",
            "门诊特殊病备案",
            "门诊特殊疾病备案",
            "备案申报表",
            "医保办公室",
            "医保办",
            "病种名称",
            "中重度哮喘",
            "外埠户籍",
            "24个月",
        )
    )


def _looks_like_special_disease_scope_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "特殊疾病范围",
            "门诊特殊疾病范围",
            "新增病种",
            "重性精神病",
            "肺动脉高压",
            "耐多药结核",
            "尼曼匹克",
            "特发性肺纤维化",
            "未备案",
            "选定特殊病种定点医疗机构",
        )
    )


def _looks_like_remote_settlement_management_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "银行手续费",
            "银行票据",
            "工本费",
            "预付金",
            "黄色预警",
            "红色预警",
            "紧急调增",
            "费用协查",
            "一次性跨省住院",
            "跨省住院总费用",
            "国家跨省异地就医管理子系统",
        )
    )


def _looks_like_shanghai_service_facility_scope_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return "上海" in compact and any(
        token in compact
        for token in (
            "医疗服务设施",
            "住院床位费",
            "急诊观察室床位费",
            "床位费",
            "实施期限",
            "有效期",
        )
    )


def _looks_like_negotiated_drug_double_channel_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "协议期内谈判药品",
            "谈判药品",
            "双通道",
            "电子处方",
            "一品两规",
            "药占比",
            "总额限制",
        )
    )


def _explicit_jurisdictions_from_question(text: str) -> list[str]:
    jurisdictions: list[str] = []
    if any(token in text for token in ("北京", "北京市")):
        jurisdictions.append("beijing")
    if any(token in text for token in ("上海", "上海市")):
        jurisdictions.append("shanghai")
    if any(token in text for token in ("国家", "全国", "跨省", "异地")):
        jurisdictions.insert(0, "national")
    return list(dict.fromkeys(jurisdictions))


def _looks_like_consumable_payment_scope_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(compact) and any(token in compact for token in ("医用耗材", "耗材"))


def _looks_like_medical_service_price_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    compact_lower = compact.lower()
    if not compact:
        return False
    if _looks_like_shanghai_service_facility_scope_query(compact):
        return False
    if _looks_like_fund_supervision_query(compact):
        return False
    return (
        any(token in compact for token in ("医疗服务价格", "医疗服务项目", "诊疗项目", "收费", "价格", "项目"))
        or "ct" in compact_lower
        or any(token in compact for token in ("检查", "检验", "影像"))
    )


def _looks_like_special_disease_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "门诊特殊疾病",
            "特殊疾病",
            "门诊特病",
            "门诊慢特病",
            "慢特病",
            "门特",
            "特殊病",
            "特病",
            "慢病",
            "病种范围",
            "病种备案",
            "病种核验",
        )
    )


def _looks_like_drug_catalog_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or _looks_like_remote_benefit_split_query(compact):
        return False
    if _looks_like_remote_emergency_manual_flow_query(compact):
        return False
    if _looks_like_special_disease_query(compact) and not _looks_like_table_row_fact_query(compact):
        return False
    if _looks_like_chronic_long_prescription_query(compact) and not _looks_like_table_row_fact_query(compact):
        return False
    if _looks_like_drug_price_reference_query(compact):
        return False
    drug_markers = (
        "药品",
        "药物",
        "用药",
        "药品目录",
        "医保药品目录",
        "限定支付范围",
        "药品编码",
        "通用名",
        "阿莫西林",
        "布洛芬",
        "阿托伐他汀",
        "二甲双胍",
        "氨氯地平",
        "缬沙坦",
    )
    return any(marker in compact for marker in drug_markers)


def _looks_like_drug_catalog_table_row_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return _looks_like_table_row_fact_query(compact) and not _looks_like_drug_catalog_policy_text_query(compact)


def _looks_like_drug_catalog_policy_text_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    if _looks_like_negotiated_drug_double_channel_query(compact):
        return True
    if any(
        token in compact
        for token in (
            "普通片剂",
            "剂型包括",
            "剂型包含",
            "包含哪些剂型",
            "包括哪些剂型",
            "哪些剂型",
            "剂型归类",
            "归类说明",
            "目录说明",
            "剂型说明",
            "剂型不单列",
            "通用名称",
            "药品目录说明",
        )
    ):
        return True
    return bool(
        _looks_like_drug_catalog_query(compact)
        and any(token in compact for token in ("如何理解", "怎么理解", "说明", "解释", "规则"))
        and not _looks_like_table_row_fact_query(compact)
    )


def _looks_like_table_row_fact_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "医保类别",
            "目录编号",
            "药品编码",
            "项目编码",
            "机构编码",
            "甲类",
            "乙类",
            "单价",
            "价格是多少",
            "收费标准",
            "计价单位",
            "本地支付比例",
            "支付比例是多少",
            "报销比例是多少",
            "编码是什么",
            "编号是什么",
            "类别是什么",
        )
    )


def _looks_like_drug_price_reference_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or "上海" not in compact:
        return False
    if _has_drug_catalog_payment_intent(compact):
        return False
    price_markers = (
        "药价",
        "价格",
        "均价",
        "平均价",
        "平均价格",
        "周均价",
        "价格区间",
        "多少钱",
    )
    drug_markers = (
        "药",
        "药品",
        "药店",
        "布洛芬",
        "阿莫西林",
        "阿托伐他汀",
        "二甲双胍",
        "氨氯地平",
        "缬沙坦",
        "连花清瘟",
        "抗病毒口服液",
        "头孢",
        "胶囊",
        "片",
        "口服液",
        "颗粒",
    )
    return any(marker in compact for marker in price_markers) and any(
        marker in compact for marker in drug_markers
    )


def _has_drug_catalog_payment_intent(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "医保类别",
            "目录编号",
            "医保目录",
            "药品目录",
            "医保药品目录",
            "支付比例",
            "本地支付比例",
            "报销比例",
            "甲类",
            "乙类",
            "限定支付范围",
        )
    )


def _force_policy_filters(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return _bool(
        value.get("force_policy_filters", value.get("_force_policy_filters")),
        False,
    )


def _normalize_policy_domain(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text in SUPPORTED_POLICY_DOMAINS:
        return [text]
    lowered = text.lower()
    alias_map = {
        "basic_medical_insurance": [],
        "medical_insurance": [],
        "医疗保障": [],
        "医疗保险": [],
        "基本医疗保险": [],
        "医保政策": [],
        "医保": [],
        "异地就医": ["remote_medical"],
        "跨省异地就医": ["remote_medical"],
        "remote_settlement": ["remote_medical"],
        "异地结算": ["remote_medical"],
        "跨省结算": ["remote_medical"],
        "outpatient": ["manual_reimbursement"],
        "outpatient_reimbursement": ["manual_reimbursement"],
        "manual_reimbursement": ["manual_reimbursement"],
        "零星报销": ["manual_reimbursement"],
        "门诊报销": ["manual_reimbursement"],
        "普通门诊": ["manual_reimbursement"],
        "普通门诊报销": ["manual_reimbursement"],
        "医疗保险报销": ["manual_reimbursement"],
        "manual_reimbursement": ["manual_reimbursement"],
        "emergency_rescue": ["emergency"],
        "emergency_non_observation": ["emergency"],
        "定点机构": ["designated_institution"],
        "定点医疗机构": ["designated_institution"],
        "定点医药机构": ["designated_institution"],
        "定点零售药店": ["designated_institution"],
        "医保定点": ["designated_institution"],
        "基金监管": ["fund_supervision"],
        "监督检查": ["fund_supervision"],
        "异常审核": ["fund_supervision"],
        "异常情形审核": ["fund_supervision"],
        "长处方": ["chronic_disease_long_prescription"],
        "长期处方": ["chronic_disease_long_prescription"],
        "慢性病长处方": ["chronic_disease_long_prescription"],
        "慢病长处方": ["chronic_disease_long_prescription"],
        "special_disease_filing": ["special_disease_filing"],
        "特殊病备案": ["special_disease_filing"],
        "特殊病种备案": ["special_disease_filing"],
        "special_disease_scope": ["special_disease_scope"],
        "特殊疾病范围": ["special_disease_scope"],
        "门诊特殊疾病范围": ["special_disease_scope"],
        "medical_consumable": ["shanghai_payment_scope"],
        "medical_consumables": ["shanghai_payment_scope"],
        "medical_consumable_catalog": ["shanghai_payment_scope"],
        "payment_scope": ["shanghai_payment_scope"],
        "shanghai_payment": ["shanghai_payment_scope"],
        "insurance_payment": ["shanghai_payment_scope"],
        "medical_insurance_payment": ["shanghai_payment_scope"],
        "medical_service": ["medical_service_price"],
        "medical_service_item": ["medical_service_price"],
        "diagnostic_item_catalog": ["medical_service_price"],
        "medical_service_price": ["medical_service_price"],
        "ct": ["medical_service_price"],
        "胸部ct": ["medical_service_price"],
        "胸部CT": ["medical_service_price"],
        "检查项目": ["medical_service_price"],
        "诊疗项目": ["medical_service_price"],
        "医疗服务项目": ["medical_service_price"],
        "price": ["medical_service_price"],
        "benefit": ["benefit"],
        "drug": ["drug_catalog"],
        "drug_catalog": ["drug_catalog"],
        "medical_insurance_drug_catalog": ["drug_catalog"],
        "医保目录": ["drug_catalog"],
        "drug_price": ["drug_product_price_reference"],
        "drug_product_price": ["drug_product_price_reference"],
        "drug_product_price_reference": ["drug_product_price_reference"],
        "price_reference": ["drug_product_price_reference"],
        "catalog": ["drug_catalog"],
        "registration_process": ["remote_medical"],
    }
    if lowered in alias_map:
        return [
            domain for domain in alias_map[lowered]
            if domain in SUPPORTED_POLICY_DOMAINS
        ]
    if _looks_like_designated_institution_query(text):
        return ["designated_institution"]
    if _looks_like_fund_supervision_query(text):
        return ["fund_supervision"]
    if _looks_like_chronic_long_prescription_query(text):
        return ["chronic_disease_long_prescription"]
    if _looks_like_benefit_policy_query(text):
        return ["benefit"]
    if _looks_like_special_disease_filing_query(text):
        return ["special_disease_filing"]
    if _looks_like_special_disease_scope_query(text):
        return ["special_disease_scope"]
    if _looks_like_remote_settlement_management_query(text):
        return ["remote_medical"]
    if _looks_like_shanghai_service_facility_scope_query(text):
        return ["shanghai_payment_scope"]
    if _looks_like_negotiated_drug_double_channel_query(text):
        domains = ["drug_catalog"]
        if "上海" in text:
            domains.append("shanghai_payment_scope")
        return domains
    if any(token in text for token in ("医用耗材", "耗材", "支付范围", "医保支付", "支付办法")):
        return ["shanghai_payment_scope"]
    if _looks_like_drug_price_reference_query(text):
        return ["drug_product_price_reference"]
    if _looks_like_special_disease_query(text):
        domains = ["special_disease_filing", "special_disease_scope"]
        if any(token in text for token in ("长期处方", "长处方", "续方")):
            domains.append("chronic_disease_long_prescription")
        return domains
    if _looks_like_medical_service_price_query(text):
        return ["medical_service_price"]
    if any(token in text for token in ("药品", "药品目录", "医保目录", "目录")):
        return ["drug_catalog"]
    if any(token in text for token in ("待遇", "起付线", "支付比例", "封顶线")):
        return ["benefit"]
    if any(token in text for token in ("手工报销", "零星报销", "门诊报销", "普通门诊", "报销材料")):
        return ["manual_reimbursement"]
    if any(token in text for token in ("急诊", "门急诊", "急诊抢救")):
        return ["emergency"]
    if any(token in text for token in ("异地", "跨省")) or (
        "备案" in text and not _looks_like_special_disease_query(text)
    ):
        return ["remote_medical"]
    return []


def _normalize_content_type(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in SUPPORTED_CONTENT_TYPES:
        return [lowered]
    alias_map = {
        "policy": ["policy_text"],
        "policy_file": ["policy_text"],
        "policy_document": ["policy_text"],
        "policy_interpretation": ["policy_text"],
        "policy_pdf": ["policy_text"],
        "document_markdown": ["policy_text"],
        "application/pdf": ["policy_text"],
        "政策": ["policy_text"],
        "政策文件": ["policy_text"],
        "政策文本": ["policy_text"],
        "政策原文": ["policy_text"],
        "原文": ["policy_text"],
        "notice": ["policy_text"],
        "catalog_notice": ["policy_text"],
        "通知": ["policy_text"],
        "规范性文件": ["policy_text"],
        "规范": ["policy_text"],
        "办法": ["policy_text"],
        "规定": ["policy_text"],
        "service": ["policy_text"],
        "guide": ["policy_text"],
        "service guide": ["policy_text"],
        "办事指南": ["policy_text"],
        "服务指南": ["policy_text"],
        "办理指南": ["policy_text"],
        "经办指南": ["policy_text"],
        "faq": ["policy_text"],
        "qa": ["policy_text"],
        "问答": ["policy_text"],
        "常见问题": ["policy_text"],
        "table": ["table_row"],
        "table_row": ["table_row"],
        "table_rows_jsonl": ["table_row"],
        "table_page_text_jsonl": ["table_row"],
        "table_csv": ["table_row"],
        "price_table": ["table_row"],
        "material_chain_table": ["table_row"],
        "catalog": ["table_row"],
        "code_database": ["table_row"],
        "drug_catalog": ["table_row"],
        "表格": ["table_row"],
        "表格行": ["table_row"],
        "目录表": ["table_row"],
        "医保药品目录": ["table_row"],
        "药品目录": ["table_row"],
        "清单": ["table_row"],
    }
    if lowered in alias_map:
        return alias_map[lowered]
    matches: list[str] = []
    for alias, normalized in alias_map.items():
        if alias and (alias in text or alias in lowered):
            matches.extend(normalized)
    return list(dict.fromkeys(matches))


def _normalize_jurisdiction(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    mapping = {
        "北京": "beijing",
        "北京市": "beijing",
        "上海": "shanghai",
        "上海市": "shanghai",
        "国家": "national",
        "全国": "national",
        "中国": "national",
        "中华人民共和国": "national",
        "cn": "national",
        "china": "national",
    }
    normalized = mapping.get(text, mapping.get(text.lower(), text.lower()))
    if normalized in SUPPORTED_JURISDICTIONS:
        return normalized
    return ""


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [
            item.strip().strip("\"'")
            for item in re.split(r"[,，、;；|]+", text)
            if item.strip().strip("\"'")
        ]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_string_list(str(item)))
        return items
    return [str(value).strip()]


def _bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return fallback
