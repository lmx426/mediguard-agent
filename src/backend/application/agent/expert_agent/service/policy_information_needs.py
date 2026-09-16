"""Dynamic answer-need resolution for policy expert answers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
)
from src.backend.application.agent.expert_agent.service.policy_answer_slots import (
    SLOT_REGISTRY,
    slot_answer_action,
    slot_label,
    slot_policy_domains,
    slot_required_fields,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    merge_policy_filters,
    normalize_policy_filters,
)


AnswerMode = Literal[
    "fact_lookup",
    "list",
    "process_rule",
    "policy_explanation",
    "comparison",
]


def normalize_information_needs(value: Any) -> list[str]:
    raw = value
    if isinstance(value, dict):
        raw = value.get("information_needs") or value.get("answer_slots") or value.get("question_slots")
    if raw in (None, "", [], {}):
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                candidate = item.get("need_text") or item.get("label") or item.get("question_span") or item.get("slot_id")
            else:
                candidate = item
            text = str(candidate or "").strip()
            if text:
                items.append(text)
    else:
        text = str(raw).strip()
        items = [text] if text else []
    normalized: list[str] = []
    for item in items:
        need = str(item).strip()
        if need in SLOT_REGISTRY:
            need = slot_label(need)
        if need and need not in normalized:
            normalized.append(need)
    return normalized[:12]


def normalize_answer_mode(value: Any, question: str, information_needs: list[str] | None = None) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"fact_lookup", "list", "process_rule", "policy_explanation", "comparison"}:
        return candidate
    compact = _compact(question)
    needs = information_needs or []
    if "比较" in compact or "对比" in compact:
        return "comparison"
    if any(token in compact for token in ("列出", "有哪些", "哪些", "清单", "目录")) and len(needs) <= 3:
        return "list"
    if any(token in compact for token in ("规则", "流程", "条件", "办理", "如何", "怎么", "步骤", "时限")):
        return "process_rule"
    if any(token in compact for token in ("为什么", "解释", "说明", "含义")):
        return "policy_explanation"
    if len(needs) == 1 and any(token in compact for token in ("是什么", "多少", "哪一个", "哪个")):
        return "fact_lookup"
    return "process_rule"


def build_information_need_requirements(
    *,
    user_question: str,
    information_needs: list[str] | None,
    filters: dict[str, Any] | None,
    answer_mode: AnswerMode = "process_rule",
) -> list[AnswerRequirement]:
    """Resolve dynamic information needs into stable internal requirements."""

    normalized_filters = normalize_policy_filters(filters or {})
    raw_needs = normalize_information_needs(information_needs or []) or _split_question(user_question)
    if not raw_needs and str(user_question or "").strip():
        raw_needs = [str(user_question).strip()]

    requirements: list[AnswerRequirement] = []
    seen_need_ids: set[str] = set()
    for need_text in raw_needs[:12]:
        need_kind = infer_need_kind(need_text, user_question=user_question, filters=normalized_filters)
        definition = SLOT_REGISTRY.get(need_kind) or SLOT_REGISTRY["policy_basis"]
        need_id = _stable_need_id(
            need_kind=need_kind,
            need_text=need_text,
            answer_mode=answer_mode,
            user_question=user_question,
            filters=normalized_filters,
        )
        if need_id in seen_need_ids:
            continue
        seen_need_ids.add(need_id)
        need_filters = merge_policy_filters(
            normalized_filters,
            definition.retrieval_profile.as_filters(),
        )
        requirements.append(
            AnswerRequirement(
                requirement_id=need_id,
                slot_id=need_kind,
                label=need_text,
                answer_mode=answer_mode,
                question_span=need_text[:240] if need_text else str(user_question)[:240],
                required=True,
                scenario_id=definition.scenario_id,
                required_fields=_required_fields_for_need(need_kind, user_question or need_text),
                answer_action=slot_answer_action(need_kind),
                filters={**need_filters, "need_kind": need_kind},
                fact_schema=definition.extraction_profile.fact_schema,
                extractor_id=definition.extraction_profile.extractor_id,
            )
        )

    if not requirements:
        definition = SLOT_REGISTRY["policy_basis"]
        requirements.append(
            AnswerRequirement(
                requirement_id=_stable_need_id(
                    need_kind="policy_basis",
                    need_text=str(user_question or "").strip() or "政策依据",
                    answer_mode=answer_mode,
                    user_question=user_question,
                    filters=normalized_filters,
                ),
                slot_id="policy_basis",
                label=str(user_question or "").strip()[:240] or "政策依据",
                answer_mode=answer_mode,
                question_span=str(user_question or "")[:240],
                required=True,
                scenario_id=definition.scenario_id,
                required_fields=list(definition.required_fields),
                answer_action=definition.answer_action,
                filters={**normalized_filters, "need_kind": "policy_basis"},
                fact_schema=definition.extraction_profile.fact_schema,
                extractor_id=definition.extraction_profile.extractor_id,
            )
        )
    return requirements


def infer_need_kind(
    need_text: str,
    *,
    user_question: str,
    filters: dict[str, Any] | None = None,
) -> str:
    compact_need = _compact(need_text)
    compact_question = _compact(user_question)
    domains = set(_string_list((filters or {}).get("policy_domain")))
    matched: list[str] = []
    for slot_id, definition in SLOT_REGISTRY.items():
        if any(pattern and pattern in compact_need for pattern in definition.question_patterns):
            matched.append(slot_id)
    if matched:
        return _prefer_need_kind(matched, compact_need)
    if any(token in compact_need for token in ("必要材料", "申请材料", "报销材料", "材料目录", "核验哪些材料")):
        return "required_materials"
    if any(token in compact_need for token in ("法定办结时限", "承诺办结时限", "办结时限", "办理时限", "多少工作日", "多久办结")):
        return "statutory_processing_time"
    if any(token in compact_need for token in ("药品", "药物", "用药", "药品目录", "医保药品")) and "drug_product_price_reference" not in domains:
        return "drug_catalog"
    if any(token in compact_need for token in ("医疗服务价格", "医疗服务项目", "诊疗项目", "计价单位", "收费标准", "CT", "ct")):
        return "medical_service_price"
    if any(token in compact_need for token in ("医用耗材", "耗材")):
        return "consumable_payment_scope"
    if any(token in compact_need for token in ("定点机构", "定点医院", "定点药店", "机构编码", "定点状态")):
        return "designated_institution"
    if any(token in compact_need for token in ("手工报销", "零星报销", "报销", "费用报销")) and "异地" not in compact_need:
        return "manual_reimbursement"
    if any(token in compact_need for token in ("异地", "跨省", "备案")) and "直接结算" in compact_need:
        return "remote_benefit_split"
    if any(token in compact_need for token in ("基金监管", "监督检查", "拒不配合", "骗取基金", "不属于基金支付范围")):
        return "fund_supervision_policy"
    if any(token in compact_need for token in ("特殊病", "特殊疾病", "门诊特殊疾病", "慢特病")):
        return "special_disease_scope_policy" if any(token in compact_need for token in ("范围", "报销", "支付条件")) else "special_disease_filing_policy"
    if any(token in compact_need for token in ("长处方", "长期处方", "慢病", "高血压", "糖尿病")):
        return "chronic_long_prescription_policy"
    if any(token in compact_need for token in ("谈判药品", "双通道", "电子处方")):
        return "negotiated_drug_double_channel"
    if any(token in compact_need for token in ("住院床位费", "急诊观察室床位费", "医疗服务设施")):
        return "shanghai_service_facility_scope"
    if any(token in compact_need for token in ("支付规则", "支付口径", "支付范围", "待遇", "分工")) and any(
        token in compact_need for token in ("跨省", "异地就医", "异地")
    ):
        return "remote_benefit_split"
    if any(token in compact_need for token in ("备案成功后", "备案后", "哪些机构就医", "定点医药机构", "定点医疗机构", "统筹地区", "就医机构范围")):
        return "remote_filing_institution_scope"
    if domains:
        for slot_id in SLOT_REGISTRY:
            if slot_id in domains:
                return slot_id
    return "policy_basis"


def information_need_prompt_contract() -> str:
    return json.dumps(
        {
            "answer_mode_values": [
                "fact_lookup",
                "list",
                "process_rule",
                "policy_explanation",
                "comparison",
            ],
            "rules": [
                "answer_mode 描述回答形态，不是检索领域。",
                "information_needs 只写用户真正需要的回答点，不要写固定槽位名。",
                "复杂问题拆成多个信息需求；每个信息需求尽量短而具体。",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def information_need_terms(need_kind: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((slot_label(need_kind), *slot_coverage_terms(need_kind))))


def information_need_policy_domains(need_kind: str) -> tuple[str, ...]:
    return slot_policy_domains(need_kind)


def information_need_required_fields(need_kind: str, question: str) -> list[str]:
    definition = SLOT_REGISTRY.get(need_kind)
    if definition is None:
        return []
    return _required_fields_for_need(need_kind, question)


def information_need_answer_action(need_kind: str) -> str:
    return slot_answer_action(need_kind)


def information_need_label(need_kind: str) -> str:
    return slot_label(need_kind)


def _required_fields_for_need(need_kind: str, question: str) -> list[str]:
    definition = SLOT_REGISTRY.get(need_kind)
    if definition is None:
        return []
    compact = _compact(str(question or ""))
    if need_kind == "remote_benefit_split" and _looks_like_remote_direct_settlement_payment_question(question):
        return [
            "直接结算费用范围",
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
            "门诊慢特病病种范围",
        ]
    if need_kind == "remote_settlement_management":
        fields: list[str] = []
        if any(token in compact for token in ("银行手续费", "银行票据", "工本费")):
            fields.append("银行费用不得列支基金")
        if any(token in compact for token in ("预付金", "黄色预警", "红色预警", "紧急调增")):
            fields.extend(["预付金黄色预警", "预付金红色预警", "紧急调增流程"])
        if any(token in compact for token in ("费用协查", "一次性跨省住院", "3万元", "三万元")):
            fields.append("费用协查信息")
        return fields or list(definition.required_fields)
    return list(definition.required_fields)


def _prefer_need_kind(slot_ids: list[str], compact: str) -> str:
    deduped = list(dict.fromkeys(slot_ids))
    scenario_slot_ids = {
        "foreign_treatment_manual_reimbursement",
        "account_settlement_voucher",
        "emergency_manual_reimbursement_materials",
        "remote_self_pay_filing_manual_reimbursement",
        "remote_emergency_observation_reimbursement",
    }
    if len(scenario_slot_ids.intersection(deduped)) >= 2:
        deduped = [
            slot_id for slot_id in deduped
            if slot_id not in {"required_materials", "manual_reimbursement", "remote_filing"}
        ]
    if "remote_emergency_observation_reimbursement" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id not in {"emergency_manual_reimbursement_materials", "remote_filing"}]
    if "remote_self_pay_filing_manual_reimbursement" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id not in {"remote_filing", "manual_reimbursement"}]
    if "remote_filing_institution_scope" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id != "remote_filing"]
    if "remote_benefit_split" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id not in {"remote_medical", "benefit_params", "policy_basis"}]
    if "special_disease_filing_policy" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id != "remote_filing"]
    if "chronic_long_prescription_policy" in deduped and "drug_catalog" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id != "drug_catalog"]
    if "benefit_policy" in deduped and "drug_catalog" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id != "drug_catalog"]
    if "shanghai_service_facility_scope" in deduped:
        deduped = [slot_id for slot_id in deduped if slot_id not in {"medical_service_price", "consumable_payment_scope"}]
    concrete = [slot_id for slot_id in deduped if not (SLOT_REGISTRY.get(slot_id).broad if SLOT_REGISTRY.get(slot_id) else False)]
    if concrete:
        keep_broad = [slot_id for slot_id in deduped if slot_id not in {"manual_reimbursement", "policy_basis"}]
        deduped = list(dict.fromkeys(keep_broad))
    return deduped[0] if deduped else "policy_basis"


def _split_question(question: str) -> list[str]:
    compact = str(question or "").strip()
    if not compact:
        return []
    parts = [
        part.strip(" ，,；;。！？?!")
        for part in re.split(r"[？?。；;\n]+", compact)
        if part.strip(" ，,；;。！？?!")
    ]
    if len(parts) <= 1:
        raw = re.split(r"[、，]", compact)
        split_parts = [part.strip() for part in raw if part.strip()]
        if len(split_parts) > 1:
            parts = split_parts
    if len(parts) <= 1:
        raw = re.split(r"[和及与]", compact)
        split_parts = [part.strip(" ，,；;。！？?!") for part in raw if part.strip(" ，,；;。！？?!")]
        if len(split_parts) > 1:
            parts = split_parts
    expanded: list[str] = []
    for part in parts:
        if not part:
            continue
        sub_parts = [
            item.strip(" ，,；;。！？?!")
            for item in re.split(r"[和及与]", part)
            if item.strip(" ，,；;。！？?!")
        ]
        if len(sub_parts) > 1:
            expanded.extend(sub_parts)
        else:
            expanded.append(part)
    parts = expanded or parts
    return [part[:240] for part in parts][:12]


def _stable_need_id(
    *,
    need_kind: str,
    need_text: str,
    answer_mode: str,
    user_question: str,
    filters: dict[str, Any],
) -> str:
    raw = json.dumps(
        {
            "kind": need_kind,
            "need_text": need_text,
            "answer_mode": answer_mode,
            "user_question": user_question,
            "filters": filters,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{need_kind}:{digest}"


def _looks_like_remote_direct_settlement_payment_question(question: str) -> bool:
    compact = _compact(str(question or ""))
    if not compact:
        return False
    return (
        any(token in compact for token in ("跨省", "异地就医", "异地"))
        and "直接结算" in compact
        and any(token in compact for token in ("医疗费用支付规则", "费用支付规则", "支付规则", "支付口径", "怎样支付", "如何支付"))
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]


resolve_information_needs = build_information_need_requirements
infer_answer_mode_from_question = normalize_answer_mode
