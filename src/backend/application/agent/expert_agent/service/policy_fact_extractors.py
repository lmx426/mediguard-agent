"""Deterministic fact extraction from normalized Policy RAG evidence."""

from __future__ import annotations

import re
from typing import Any

from src.backend.application.agent.expert_agent.schemas import PolicyEvidence
from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
    EvidenceMatch,
    ExtractedFact,
    NormalizedPolicyEvidence,
)
from src.backend.application.agent.expert_agent.service.policy_answer_slots import (
    slot_coverage_terms,
    slot_policy_domains,
)


MATERIAL_NAMES = (
    "医疗保险卡损坏告知单",
    "住院期间的医疗费用清单",
    "北京市医疗保险手工报销(外埠就医)费用申报结算明细表",
    "北京市医疗保险手工报销(外埠就医)费用审核结算凭证",
    "北京市医疗保险门急诊(药店)费用审核结算凭证",
    "北京市医疗保险住院费用审核结算凭证",
    "新发与补（换）社会保障卡证明",
    "被委托人身份证",
    "疾病诊断证明书",
    "医疗费专用收据",
    "住院费用结算单",
    "住院费用结算清单",
    "出院(观)小结",
    "医保电子凭证",
    "有效身份证件",
    "费用清单",
    "诊断证明",
    "诊疗证明",
    "处方底方",
    "专用收据",
    "检查报告",
    "病史资料",
    "社会保障卡",
    "社保卡",
    "医保卡",
    "银行卡",
    "身份证",
    "收据",
    "处方",
)

DRUG_DOSAGE_TERMS = (
    "普通片剂",
    "片剂",
    "片",
    "胶囊",
    "胶囊剂",
    "颗粒",
    "颗粒剂",
    "口服",
    "口服液",
    "滴眼",
    "滴眼剂",
    "注射",
    "注射剂",
    "缓释",
    "控释",
    "肠溶",
    "薄膜衣",
    "糖衣",
    "咀嚼",
    "分散",
    "丸",
    "散",
    "栓",
    "贴",
)


def normalize_policy_evidence(
    evidence: list[PolicyEvidence],
) -> list[NormalizedPolicyEvidence]:
    """Convert adopted PolicyEvidence objects into stable internal evidence."""

    normalized: list[NormalizedPolicyEvidence] = []
    for item in evidence:
        content = str(item.excerpt or "").strip()
        if not content:
            continue
        normalized.append(
            NormalizedPolicyEvidence(
                evidence_id=item.evidence_ref,
                source_ref=item.source_ref,
                title=item.title,
                content=content,
                jurisdiction=item.jurisdiction,
                policy_domain=item.policy_domain,
                content_type=item.content_type,
                source_url=item.source_url,
                version=item.version,
                metadata=item.metadata,
            )
        )
    return normalized


def extract_facts_for_requirements(
    requirements: list[AnswerRequirement],
    evidence: list[NormalizedPolicyEvidence],
) -> tuple[list[ExtractedFact], list[EvidenceMatch]]:
    """Extract facts and requirement-evidence coverage matches."""

    facts: list[ExtractedFact] = []
    matches: list[EvidenceMatch] = []
    fact_index = 1
    match_index = 1
    for requirement in requirements:
        candidates = _candidate_evidence(requirement, evidence)
        req_facts: list[ExtractedFact] = []
        for candidate in candidates[:8]:
            extracted = _extract_from_evidence(requirement, candidate, fact_index)
            if extracted:
                req_facts.extend(extracted)
                fact_index += len(extracted)
                matches.append(
                    EvidenceMatch(
                        match_id=f"match_{match_index}",
                        requirement_id=requirement.requirement_id,
                        evidence_id=candidate.evidence_id,
                        coverage_status="covered",
                        matched_spans=[
                            fact.evidence_text for fact in extracted[:3]
                        ],
                        score=1.0,
                        reason="extractable_fact_found",
                    )
                )
                match_index += 1
            elif _evidence_candidate_score(requirement, candidate) >= 0.55:
                matches.append(
                    EvidenceMatch(
                        match_id=f"match_{match_index}",
                        requirement_id=requirement.requirement_id,
                        evidence_id=candidate.evidence_id,
                        coverage_status="partial",
                        matched_spans=[
                            _best_snippet(candidate.content, slot_coverage_terms(requirement.slot_id))
                        ],
                        score=0.55,
                        reason="topic_match_without_extractable_fact",
                    )
                )
                match_index += 1
        facts.extend(req_facts)
    return _deduplicate_facts(facts), matches


def _extract_from_evidence(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    extractor_id = requirement.extractor_id
    if extractor_id == "remote_benefit_split":
        return _extract_remote_benefit_split(requirement, evidence, start_index)
    if extractor_id == "material_list":
        return _extract_material_list(requirement, evidence, start_index)
    if extractor_id == "manual_reimbursement_scenario":
        return _extract_manual_reimbursement_scenario(requirement, evidence, start_index)
    if extractor_id == "remote_self_pay_filing_manual_reimbursement":
        return _extract_remote_self_pay_filing_manual_reimbursement(
            requirement,
            evidence,
            start_index,
        )
    if extractor_id == "remote_emergency_observation_reimbursement":
        return _extract_remote_emergency_observation_reimbursement(
            requirement,
            evidence,
            start_index,
        )
    if extractor_id == "processing_time":
        return _extract_processing_time(requirement, evidence, start_index)
    if extractor_id == "drug_catalog":
        return _extract_drug_catalog(requirement, evidence, start_index)
    if extractor_id == "medical_service_price":
        return _extract_medical_service_price(requirement, evidence, start_index)
    if extractor_id == "consumable_payment_scope":
        return _extract_consumable_payment_scope(requirement, evidence, start_index)
    if extractor_id == "benefit_params":
        return _extract_benefit_params(requirement, evidence, start_index)
    if extractor_id == "designated_institution":
        return _extract_designated_institution(requirement, evidence, start_index)
    if extractor_id == "special_disease_condition":
        return _extract_special_disease_condition(requirement, evidence, start_index)
    if extractor_id == "required_field_policy_rule":
        return _extract_required_field_policy_rule(requirement, evidence, start_index)
    if requirement.slot_id == "remote_filing_institution_scope":
        return _extract_remote_filing_institution_scope(requirement, evidence, start_index)
    return _extract_policy_rule_span(requirement, evidence, start_index)


def _extract_remote_benefit_split(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _compact(text)
    if not (
        "就医地" in compact
        and "参保地" in compact
        and any(term in compact for term in ("支付范围", "目录"))
        and any(term in compact for term in ("起付标准", "支付比例", "最高支付限额", "待遇"))
    ):
        return []
    snippet = _best_snippet(
        text,
        (
            "住院",
            "普通门诊",
            "门诊慢特病",
            "就医地",
            "参保地",
            "支付范围",
            "起付标准",
            "支付比例",
            "最高支付限额",
            "病种范围",
        ),
    )
    facts: list[ExtractedFact] = []
    if all(term in compact for term in ("住院", "普通门诊", "门诊慢特病")):
        facts.append(
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="direct_settlement_expense_scope",
                value={
                    "直接结算费用范围": ["住院", "普通门诊", "门诊慢特病医疗费用"],
                },
                display_text="跨省异地就医直接结算覆盖住院、普通门诊和门诊慢特病医疗费用。",
                evidence_text=snippet,
                confidence=0.96,
            )
        )
    facts.extend(
        [
            _fact(
                start_index + len(facts),
                requirement,
                evidence,
                fact_type="payment_scope_owner",
                value={
                    "owner": "就医地",
                    "items": ["医保药品", "医疗服务项目", "医用耗材"],
                    "就医地支付范围": "医保药品、医疗服务项目和医用耗材等支付范围原则上按就医地规定执行",
                },
                display_text="医保药品、医疗服务项目和医用耗材等支付范围原则上按就医地规定执行。",
                evidence_text=snippet,
                confidence=0.96,
            ),
            _fact(
                start_index + len(facts) + 1,
                requirement,
                evidence,
                fact_type="benefit_policy_owner",
                value={
                    "owner": "参保地",
                    "items": ["起付标准", "支付比例", "最高支付限额", "门诊慢特病病种范围"],
                    "参保地起付标准": "基本医疗保险基金起付标准按参保地规定执行",
                    "参保地支付比例": "基本医疗保险基金支付比例按参保地规定执行",
                    "参保地最高支付限额": "基本医疗保险基金最高支付限额按参保地规定执行",
                    "门诊慢特病病种范围": "门诊慢特病病种范围按参保地政策执行",
                },
                display_text="基金起付标准、支付比例、最高支付限额和门诊慢特病病种范围等待遇参数按参保地政策执行。",
                evidence_text=snippet,
                confidence=0.96,
            ),
        ]
    )
    return facts


def _extract_material_list(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    required, non_required = _extract_material_statuses(text)
    facts: list[ExtractedFact] = []
    snippet = _best_snippet(text, ("材料名称", "材料必要性", "申请材料", "必要", "非必要"))
    if required:
        facts.append(
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="material_list",
                value={"required_items": required},
                display_text=f"必要材料包括{_join_cn(required)}。",
                evidence_text=snippet,
                confidence=0.9,
            )
        )
    if non_required and any(token in _compact(requirement.question_span) for token in ("非必要", "可不提交", "无需提交")):
        facts.append(
            _fact(
                start_index + len(facts),
                requirement,
                evidence,
                fact_type="non_required_material_list",
                value={"non_required_items": non_required[:8]},
                display_text=f"材料目录中标注为非必要的材料包括{_join_cn(non_required[:5])}等。",
                evidence_text=snippet,
                confidence=0.86,
            )
        )
    if facts:
        return facts
    fallback = _extract_materials_from_sentence(text)
    if not fallback:
        return []
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="material_list",
            value={"mentioned_items": fallback},
            display_text=f"材料要求中提到{_join_cn(fallback)}。",
            evidence_text=_best_snippet(text, tuple(fallback)),
            confidence=0.74,
        )
    ]


def _extract_manual_reimbursement_scenario(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _compact(text)
    slot_id = requirement.slot_id
    if slot_id == "foreign_treatment_manual_reimbursement":
        required_terms = ("外埠", "诊疗证明", "处方底方", "费用清单", "费用收据")
        if not all(term in compact for term in required_terms):
            return []
        snippet = _best_snippet(
            text,
            ("外埠", "诊疗证明", "处方底方", "费用清单", "费用收据", "申报结算明细表"),
        )
        return [
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="foreign_treatment_manual_reimbursement",
                value={
                    "scenario": "外埠就医",
                    "required_items": ["诊疗证明", "处方底方", "费用清单", "费用收据"],
                    "voucher": "北京市医疗保险手工报销(外埠就医)费用申报结算明细表",
                },
                display_text=(
                    "外埠就医手工报销需持外埠定点医疗机构的诊疗证明、处方底方、"
                    "费用清单和费用收据，并填写《北京市医疗保险手工报销(外埠就医)"
                    "费用申报结算明细表》等报区、县医保中心审核结算。"
                ),
                evidence_text=snippet,
                confidence=0.94,
            )
        ]
    if slot_id == "account_settlement_voucher":
        if not (
            "定点医疗机构" in compact
            and "定点零售药店" in compact
            and ("个人帐户" in compact or "个人账户" in compact)
            and "记账" in compact
            and "门急诊" in compact
            and "审核结算凭证" in compact
        ):
            return []
        snippet = _best_snippet(
            text,
            ("定点医疗机构", "定点零售药店", "个人帐户", "个人账户", "记账", "门急诊", "审核结算凭证"),
        )
        return [
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="account_settlement_voucher",
                value={
                    "scenario": "定点医药机构记账结算",
                    "voucher": "北京市医疗保险门急诊(药店)费用审核结算凭证",
                },
                display_text=(
                    "定点医疗机构和定点零售药店对个人账户支付部分记账结算时，"
                    "需填写《北京市医疗保险门急诊(药店)费用审核结算凭证》，"
                    "并与区、县医保中心结算。"
                ),
                evidence_text=snippet,
                confidence=0.94,
            )
        ]
    if slot_id == "emergency_manual_reimbursement_materials":
        if (
            any(term in compact for term in ("未出示社保卡", "医保电子凭证", "没带医保凭证", "全额垫付"))
            and "收据" in compact
            and "处方" in compact
            and "诊断证明" in compact
            and "手工报销" in compact
        ):
            snippet = _best_snippet(
                text,
                ("社保卡", "医保电子凭证", "全额垫付", "收据", "处方", "诊断证明", "手工报销"),
            )
            return [
                _fact(
                    start_index,
                    requirement,
                    evidence,
                    fact_type="emergency_manual_reimbursement_materials",
                    value={
                        "scenario": "急诊就医",
                        "required_items": ["收据", "处方", "诊断证明"],
                        "submission_channel": "单位或便民服务中心向区医保经办机构申请手工报销",
                    },
                    display_text=(
                        "急诊就医未出示社保卡或医保电子凭证时，需个人先全额垫付，"
                        "保留医院开具的收据、处方、诊断证明等材料，交由单位或便民"
                        "服务中心到区医保经办机构进行手工报销。"
                    ),
                    evidence_text=snippet,
                    confidence=0.9,
                )
            ]
        if (
            "留观" in _compact(requirement.question_span)
            and "急诊留观" in compact
            and "门急诊" in compact
            and "审核结算凭证" in compact
        ):
            snippet = _best_snippet(
                text,
                ("急诊留观", "门急诊", "审核结算凭证", "收入院证明", "处方底方", "专用收据"),
            )
            return [
                _fact(
                    start_index,
                    requirement,
                    evidence,
                    fact_type="emergency_observation_voucher",
                    value={
                        "scenario": "急诊留观",
                        "required_items": ["收入院证明", "处方底方", "专用收据"],
                        "voucher": "北京市医疗保险门急诊(药店)费用审核结算凭证",
                    },
                    display_text=(
                        "急诊留观并收入院前7日发生的费用，由用人单位汇总填写"
                        "《北京市医疗保险门急诊(药店)费用审核结算凭证》，并附"
                        "收入院证明、处方底方和专用收据报区、县医保中心审核结算。"
                    ),
                    evidence_text=snippet,
                    confidence=0.86,
                )
            ]
    return []


def _extract_remote_self_pay_filing_manual_reimbursement(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _field_match_text(text)
    if not (
        ("自费结算" in compact or "自行垫付" in compact or "全额垫付" in compact)
        and ("补办备案" in compact or "补办备案手续" in compact)
        and "参保地" in compact
        and "手工报销" in compact
    ):
        return []
    snippet = _best_snippet(
        text,
        ("自费结算", "补办备案", "参保地", "医保手工报销", "手工报销"),
    )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="remote_self_pay_filing_manual_reimbursement",
            value={
                "scenario": "异地就医出院自费结算后补办备案",
                "settlement": "出院自费结算",
                "filing": "按参保地规定补办备案手续",
                "reimbursement_path": "申请医保手工报销",
            },
            display_text=(
                "异地就医出院自费结算后，可按参保地规定补办备案手续，"
                "再申请医保手工报销。"
            ),
            evidence_text=snippet,
            confidence=0.92,
        )
    ]


def _extract_remote_emergency_observation_reimbursement(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _field_match_text(text)
    if not (
        "急诊留观" in compact
        and ("不能实现异地直接结算" in compact or "暂不能实现异地直接结算" in compact or "暂不能直接结算" in compact)
        and "票据" in compact
        and "报销材料" in compact
        and ("社保所" in compact or "单位" in compact)
        and "医保经办机构" in compact
        and ("住院标准报销" in compact or "按住院标准" in compact)
    ):
        return []
    snippet = _best_snippet(
        text,
        ("急诊留观", "暂不能实现异地直接结算", "票据", "相关报销材料", "社保所", "住院标准报销"),
    )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="remote_emergency_observation_reimbursement",
            value={
                "scenario": "异地急诊留观",
                "direct_settlement": "暂不能实现异地直接结算",
                "required_materials": "异地就医票据及相关报销材料",
                "submission_channel": "本人所属单位或社保所提交",
                "handler": "所属区医保经办机构手工报销",
                "payment_standard": "按住院标准报销",
            },
            display_text=(
                "异地急诊留观费用暂不能实现异地直接结算，参保人员可将异地就医"
                "票据及相关报销材料交给本人所属单位或社保所，由单位或社保所"
                "向所属区医保经办机构申请手工报销，并按住院标准报销。"
            ),
            evidence_text=snippet,
            confidence=0.92,
        )
    ]


def _extract_processing_time(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    facts: list[ExtractedFact] = []
    statutory = _first_unique(
        re.findall(r"法定办结时限(?:说明)?\s*[:：]?\s*(\d+)\s*(?:个)?\s*[\(（]?\s*工作日", text)
    )
    if statutory:
        facts.append(
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="statutory_processing_time",
                value={"days": statutory, "unit": "工作日"},
                display_text=f"法定办结时限为{statutory}个工作日。",
                evidence_text=_best_snippet(text, ("法定办结时限", statutory, "工作日")),
                confidence=0.94,
            )
        )
    step_match = re.search(
        r"审查\s*(\d+)\s*[^。；;]{0,80}?决定\s*(\d+)\s*",
        text,
    )
    if step_match:
        review_days, decision_days = step_match.groups()
        facts.append(
            _fact(
                start_index + len(facts),
                requirement,
                evidence,
                fact_type="processing_step_time",
                value={"review_days": review_days, "decision_days": decision_days, "unit": "工作日"},
                display_text=f"办理流程中审查环节为{review_days}个工作日，决定环节为{decision_days}个工作日。",
                evidence_text=_best_snippet(text, ("审查", review_days, "决定", decision_days)),
                confidence=0.88,
            )
        )
    return facts


def _extract_drug_catalog(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    fields = _field_map(evidence.content)
    category = (
        fields.get("医保类别")
        or fields.get("类别")
        or fields.get("insurance_class")
        or fields.get("catalog_class")
        or ""
    )
    code = (
        fields.get("目录编号")
        or fields.get("编号")
        or fields.get("catalog_no")
        or fields.get("normalized_catalog_no")
        or ""
    )
    name = (
        fields.get("药品名称")
        or fields.get("通用名")
        or fields.get("注册名称")
        or fields.get("drug_name")
        or fields.get("generic_name")
        or ""
    )
    dosage = (
        fields.get("剂型")
        or fields.get("剂型/规格")
        or fields.get("dosage_form")
        or fields.get("dosage")
        or ""
    )
    local_ratio = (
        fields.get("本地支付比例")
        or fields.get("支付比例")
        or fields.get("local_payment_policy")
        or fields.get("local_payment_ratio")
        or ""
    )
    remark = (
        fields.get("备注")
        or fields.get("限定支付范围")
        or fields.get("remark")
        or fields.get("payment_limit")
        or ""
    )

    if not (category or code or name):
        row = _parse_compact_drug_row(evidence.content)
        category = row.get("category", "")
        code = row.get("code", "")
        name = row.get("name", "")
        dosage = row.get("dosage", "")
        local_ratio = row.get("local_ratio", "")

    if not (category or code or name or local_ratio or remark):
        return _extract_drug_catalog_guidance(requirement, evidence, start_index)

    if evidence.content_type == "table_row" and not _row_identity_matches_question(
        name=name,
        code=code,
        dosage=dosage,
        question=requirement.question_span,
    ):
        return []

    normalized_category = _format_catalog_category(category)
    display_parts: list[str] = []
    subject = _drug_subject_for_question(name=name, dosage=dosage, question=requirement.question_span)
    if normalized_category:
        display_parts.append(f"{subject}的医保类别为{normalized_category}")
    if code:
        display_parts.append(f"目录编号为{code}")
    if local_ratio:
        display_parts.append(f"本地支付比例为{local_ratio}")
    if remark and _question_asks_drug_remark(requirement.question_span):
        display_parts.append(f"备注或限定支付范围为{remark}")
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="drug_catalog_row",
            value={
                "name": name,
                "dosage": dosage,
                "category": normalized_category,
                "catalog_code": code,
                "local_ratio": local_ratio,
                "remark": remark,
            },
            display_text="，".join(display_parts) + "。",
            evidence_text=_best_snippet(evidence.content, (name, code, category, local_ratio)),
            confidence=0.9,
        )
    ]


def _extract_drug_catalog_guidance(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    compact = _compact(evidence.content)
    asks_dosage_definition = _question_asks_drug_dosage_definition(requirement.question_span)
    if not any(
        term in compact
        for term in ("医保药品", "药品目录", "限定支付范围", "医保类别", "甲类", "乙类", "普通片剂", "剂型")
    ):
        return []
    dosage_definition = _extract_drug_dosage_definition(requirement.question_span, evidence.content)
    if dosage_definition:
        return [
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="drug_catalog_dosage_definition",
                value={"policy_text": dosage_definition},
                display_text=dosage_definition,
                evidence_text=_best_snippet(
                    evidence.content,
                    ("普通片剂", "剂型", "素片", "糖衣片", "薄膜衣片", "分散片", "咀嚼片"),
                ),
                confidence=0.86,
            )
        ]
    if asks_dosage_definition:
        return []
    snippet = _best_snippet(
        evidence.content,
        ("医保药品", "药品目录", "医保类别", "限定支付范围", "诊断", "治疗", "病情"),
    )
    display = "药品支付范围应核验药品是否纳入医保药品目录、医保类别以及限定支付范围。"
    if any(term in compact for term in ("诊断", "治疗", "病情", "适应症")):
        display = (
            "药品支付范围除核验医保目录、医保类别和限定支付范围外，还应核验诊断、"
            "治疗与病情或说明书适应症是否匹配。"
        )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="drug_catalog_policy_guidance",
            value={"policy_text": snippet},
            display_text=display,
            evidence_text=snippet,
            confidence=0.72,
        )
    ]


def _extract_medical_service_price(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    fields = _field_map(evidence.content)
    name = fields.get("项目名称") or fields.get("项目") or ""
    code = fields.get("编码") or fields.get("项目编码") or ""
    unit = fields.get("计价单位") or ""
    price = fields.get("收费标准") or fields.get("价格") or fields.get("单价") or ""
    insurance = fields.get("医保类别") or ""
    description = fields.get("内容说明") or fields.get("项目内涵") or ""
    if not (name or code or unit or price or description):
        return _extract_medical_service_guidance(requirement, evidence, start_index)
    if evidence.content_type == "table_row" and not _service_row_matches_question(
        name=name,
        code=code,
        text=evidence.content,
        question=requirement.question_span,
    ):
        return []
    parts: list[str] = []
    subject = name or "该医疗服务项目"
    if code:
        parts.append(f"编码为{code}")
    if unit:
        parts.append(f"计价单位为{unit}")
    if price:
        parts.append(f"收费标准为{price}")
    if insurance:
        parts.append(f"医保类别为{insurance}")
    if description:
        parts.append(f"内容说明为{_clip(description, 120)}")
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="medical_service_price_row",
            value={
                "name": name,
                "code": code,
                "unit": unit,
                "price": price,
                "insurance_category": insurance,
                "description": description,
            },
            display_text=f"{subject}" + ("的" if parts else "") + "，".join(parts) + "。",
            evidence_text=_best_snippet(evidence.content, (name, code, unit, price)),
            confidence=0.9,
        )
    ]


def _extract_medical_service_guidance(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    compact = _compact(evidence.content)
    if not any(term.lower() in compact.lower() for term in ("CT", "医疗服务项目", "诊疗项目", "计价单位", "项目内涵")):
        return []
    snippet = _best_snippet(
        evidence.content,
        ("CT", "医疗服务项目", "诊疗项目", "计价单位", "收费标准", "项目内涵", "报告"),
    )
    if not _service_row_matches_question(
        name="",
        code="",
        text=evidence.content,
        question=requirement.question_span,
    ):
        return []
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="medical_service_price_guidance",
            value={"policy_text": snippet},
            display_text=(
                "医疗服务项目支付范围和价格核验应核对项目名称、编码、计价单位、"
                "收费标准以及项目内涵或报告要求。"
            ),
            evidence_text=snippet,
            confidence=0.72,
        )
    ]


def _extract_consumable_payment_scope(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    fields = _field_map(evidence.content)
    name = (
        fields.get("耗材名称")
        or fields.get("医用耗材名称")
        or fields.get("项目名称")
        or fields.get("名称")
        or ""
    )
    code = fields.get("编码") or fields.get("耗材编码") or fields.get("项目编码") or ""
    category = fields.get("医保类别") or fields.get("类别") or fields.get("支付类别") or ""
    payment_method = fields.get("支付办法") or fields.get("支付范围") or fields.get("基金支付范围") or ""
    self_pay = fields.get("先自负比例") or fields.get("个人先自负比例") or ""

    if evidence.content_type == "table_row":
        if not _row_identity_matches_question(
            name=name,
            code=code,
            question=requirement.question_span,
        ):
            return []
        parts: list[str] = []
        if code:
            parts.append(f"编码为{code}")
        if category:
            parts.append(f"医保类别为{category}")
        if payment_method:
            parts.append(f"支付办法或支付范围为{payment_method}")
        if self_pay:
            parts.append(f"先自负比例为{self_pay}")
        if not parts:
            return []
        return [
            _fact(
                start_index,
                requirement,
                evidence,
                fact_type="consumable_payment_scope_row",
                value={
                    "name": name,
                    "code": code,
                    "category": category,
                    "payment_method": payment_method,
                    "self_pay_ratio": self_pay,
                },
                display_text=f"{name or '该医用耗材'}的" + "，".join(parts) + "。",
                evidence_text=_best_snippet(evidence.content, (name, code, category, payment_method, self_pay)),
                confidence=0.86,
            )
        ]

    compact = _compact(evidence.content)
    if not any(term in compact for term in ("医用耗材", "耗材")):
        return []
    if not any(term in compact for term in ("支付范围", "基金支付", "支付办法", "先自负", "医保类别", "甲类", "乙类")):
        return []
    snippet = _best_snippet(
        evidence.content,
        ("医用耗材", "耗材", "支付范围", "基金支付", "支付办法", "先自负", "医保类别"),
    )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="consumable_payment_scope_guidance",
            value={"policy_text": snippet},
            display_text=(
                "医用耗材支付范围应核验是否纳入医保支付范围，并按对应支付办法、"
                "医保分类或先自负比例执行。"
            ),
            evidence_text=snippet,
            confidence=0.74,
        )
    ]


def _extract_benefit_params(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    snippet = _best_snippet(evidence.content, ("起付", "支付比例", "最高支付限额", "封顶"))
    compact = _compact(snippet)
    if not any(term in compact for term in ("起付", "支付比例", "报销比例", "最高支付限额", "封顶")):
        return []
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="benefit_parameter",
            value={"policy_text": snippet},
            display_text=_ensure_sentence(_clip(snippet, 220)),
            evidence_text=snippet,
            confidence=0.72,
        )
    ]


def _extract_designated_institution(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    fields = _field_map(evidence.content)
    metadata = evidence.metadata or {}
    name = (
        fields.get("机构名称")
        or fields.get("医疗机构名称")
        or fields.get("药店名称")
        or fields.get("名称")
        or fields.get("institution_name")
        or fields.get("pharmacy_name")
        or fields.get("medical_institution_name")
        or str(metadata.get("institution_name") or "")
        or str(metadata.get("pharmacy_name") or "")
        or str(metadata.get("medical_institution_name") or "")
        or str(metadata.get("name") or "")
        or ""
    )
    code = (
        fields.get("编码")
        or fields.get("机构编码")
        or fields.get("药店编码")
        or fields.get("institution_code")
        or fields.get("pharmacy_code")
        or fields.get("medical_institution_code")
        or str(metadata.get("institution_code") or "")
        or str(metadata.get("pharmacy_code") or "")
        or str(metadata.get("medical_institution_code") or "")
        or str(metadata.get("code") or "")
    )
    address = (
        fields.get("地址")
        or fields.get("机构地址")
        or fields.get("address")
        or str(metadata.get("address") or "")
        or str(metadata.get("institution_address") or "")
        or ""
    )
    status = (
        fields.get("状态")
        or fields.get("定点状态")
        or fields.get("status")
        or str(metadata.get("status") or "")
        or str(metadata.get("institution_status") or "")
        or ""
    )
    district = fields.get("district") or fields.get("所属区") or str(metadata.get("district") or "")
    institution_type = (
        fields.get("institution_type")
        or fields.get("机构类型")
        or str(metadata.get("institution_type") or "")
    )
    if not (name or code or address or status or district):
        return []
    if _designated_question_has_specific_entity(requirement.question_span) and not _row_identity_matches_question(
        name=name,
        code=code,
        question=requirement.question_span,
    ):
        return []
    parts = []
    if code:
        parts.append(f"编码为{code}")
    if status:
        parts.append(f"状态为{status}")
    if institution_type:
        parts.append(f"类型为{institution_type}")
    if district:
        parts.append(f"所属区为{district}")
    if address:
        parts.append(f"地址为{address}")
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="designated_institution_row",
            value={
                "name": name,
                "code": code,
                "address": address,
                "status": status,
                "district": district,
                "institution_type": institution_type,
            },
            display_text=f"{name or '该定点机构'}" + ("的" if parts else "") + "，".join(parts) + "。",
            evidence_text=_best_snippet(evidence.content, (name, code, status, district)),
            confidence=0.86,
        )
    ]


def _extract_special_disease_condition(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _compact(text)
    if not (
        "类风湿关节炎" in compact
        and "DMARDs" in text
        and "3-6个月" in compact
        and "疾病活动度下降低于50%" in compact
        and "风湿病专科医师处方" in compact
    ):
        return []
    snippet = _best_snippet(
        text,
        ("类风湿关节炎", "DMARDs", "3-6个月", "疾病活动度下降低于50%", "风湿病专科医师处方"),
    )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="special_disease_payment_condition",
            value={
                "disease": "类风湿关节炎",
                "conditions": [
                    "诊断明确",
                    "经传统DMARDs治疗3-6个月",
                    "疾病活动度下降低于50%",
                    "需风湿病专科医师处方",
                ],
            },
            display_text=(
                "诊断明确的类风湿关节炎需经传统DMARDs治疗3-6个月且疾病活动度"
                "下降低于50%，并需风湿病专科医师处方，方可纳入医保支付。"
            ),
            evidence_text=snippet,
            confidence=0.96,
        )
    ]


def _extract_remote_filing_institution_scope(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    text = evidence.content
    compact = _compact(text)
    if not (
        "备案" in compact
        and "统筹地区" in compact
        and "定点医药机构" in compact
    ):
        return []
    snippet = _best_snippet(
        text,
        ("备案", "就医地", "统筹地区", "所有定点医药机构", "就医结算"),
    )
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="remote_filing_institution_scope",
            value={
                "scope": "就医地所在统筹地区内所有定点医药机构",
                "备案到就医地统筹地区": "原则上只需备案到就医地所在统筹地区",
                "统筹地区内所有定点医药机构": "备案成功后可在该统筹地区内所有定点医药机构就医",
                "按规定就医结算": "按规定就医结算",
            },
            display_text=(
                "本市参保人员办理跨省异地就医备案时，原则上只需备案到就医地"
                "所在统筹地区，备案成功后可在该统筹地区内所有定点医药机构"
                "按规定就医结算。"
            ),
            evidence_text=snippet,
            confidence=0.94,
        )
    ]


def _extract_required_field_policy_rule(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    terms = tuple(
        dict.fromkeys(
            [
                *slot_coverage_terms(requirement.slot_id),
                *[
                    alias
                    for field in requirement.required_fields
                    for alias in _required_field_aliases(field, {})
                ],
            ]
        )
    )
    snippet = _best_snippet(evidence.content, terms or slot_coverage_terms(requirement.slot_id))
    if not snippet:
        return []
    matched_fields = _matched_required_fields(
        requirement,
        {"policy_text": snippet},
        snippet,
        evidence.content,
    )
    if requirement.required_fields and not matched_fields:
        return []
    if not requirement.required_fields and not any(term and term in snippet for term in terms):
        return []
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type=requirement.fact_schema,
            value={
                "policy_text": snippet,
                "matched_required_fields": matched_fields,
            },
            display_text=_ensure_sentence(_clip(snippet, 360)),
            evidence_text=snippet,
            confidence=0.86 if matched_fields else 0.74,
        )
    ]


def _extract_policy_rule_span(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    start_index: int,
) -> list[ExtractedFact]:
    terms = slot_coverage_terms(requirement.slot_id)
    snippet = _best_snippet(evidence.content, terms)
    if not snippet or not any(term and term in snippet for term in terms):
        return []
    return [
        _fact(
            start_index,
            requirement,
            evidence,
            fact_type="policy_rule_span",
            value={"policy_text": snippet},
            display_text=_ensure_sentence(_clip(snippet, 260)),
            evidence_text=snippet,
            confidence=0.68,
        )
    ]


def _candidate_evidence(
    requirement: AnswerRequirement,
    evidence: list[NormalizedPolicyEvidence],
) -> list[NormalizedPolicyEvidence]:
    scored = [
        (item, _evidence_candidate_score(requirement, item))
        for item in evidence
    ]
    return [
        item for item, score in sorted(scored, key=lambda pair: pair[1], reverse=True)
        if score >= 0.25
    ]


def _evidence_candidate_score(
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
) -> float:
    score = 0.0
    domains = set(slot_policy_domains(requirement.slot_id))
    if not domains or (evidence.policy_domain or "") in domains:
        score += 0.45
    content = evidence.title + "\n" + evidence.content
    terms = [term for term in slot_coverage_terms(requirement.slot_id) if term]
    if terms:
        matched = sum(1 for term in terms if term in content)
        score += min(0.45, matched / max(len(terms), 1) * 0.45)
    else:
        score += 0.25
    if requirement.slot_id in {"drug_catalog", "medical_service_price", "consumable_payment_scope", "designated_institution"}:
        if evidence.content_type == "table_row":
            score += 0.1
    if requirement.slot_id == "special_disease_payment_condition":
        content = evidence.title + "\n" + evidence.content
        compact = _compact(content)
        if evidence.policy_domain == "special_disease_scope":
            score += 0.2
        if "类风湿关节炎" in compact and "DMARDs" in content:
            score += 0.2
    if requirement.slot_id == "remote_benefit_split" and "就医地" in content and "参保地" in content:
        score += 0.15
    return min(score, 1.0)


def _extract_material_statuses(text: str) -> tuple[list[str], list[str]]:
    required: list[tuple[int, str]] = []
    non_required: list[tuple[int, str]] = []
    for name in MATERIAL_NAMES:
        status = _material_status(text, name)
        if status is None:
            continue
        position = text.find(name)
        if status == "required":
            required.append((position, name))
        else:
            non_required.append((position, name))
    return (
        _drop_subsumed_material_names(_ordered_unique(required)),
        _drop_subsumed_material_names(_ordered_unique(non_required)),
    )


def _material_status(text: str, name: str) -> str | None:
    for match in re.finditer(re.escape(name), text):
        start = match.start()
        prefix = text[max(0, start - 6):start]
        if name == "身份证" and "被委托人" in prefix:
            continue
        window = text[start:start + 100]
        non_index = window.find("非必要")
        necessary_index = window.find("必要")
        if non_index >= 0 and (necessary_index < 0 or non_index <= necessary_index):
            return "non_required"
        if necessary_index >= 0:
            return "required"
    return None


def _extract_materials_from_sentence(text: str) -> list[str]:
    snippets = [
        sentence for sentence in _sentences(text)
        if "材料" in sentence or "收据" in sentence or "处方" in sentence
    ]
    found: list[tuple[int, str]] = []
    for snippet in snippets[:4]:
        for name in MATERIAL_NAMES:
            if name in snippet:
                found.append((text.find(name), name))
    return _drop_subsumed_material_names(_ordered_unique(found))[:8]


def _field_map(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([^:\n：]{1,28})[:：]\s*([^\n]+)", text):
        key = match.group(1).strip()
        value = match.group(2).strip(" ；。")
        if key and value:
            fields[key] = value
    return fields


def _parse_compact_drug_row(text: str) -> dict[str, str]:
    compact = re.sub(r"\s+", " ", text).strip()
    ratio_match = re.search(r"(?P<local_ratio>\d+(?:\.\d+)?%)\s*$", compact)
    local_ratio = ratio_match.group("local_ratio") if ratio_match else ""
    body = compact[: ratio_match.start()].rstrip() if ratio_match else compact
    match = re.search(
        r"(?P<category>[甲乙])\s+(?P<code>\d{1,6})\s+"
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·（）()]+)"
        r"(?:\s+(?P<dosage>[\u4e00-\u9fffA-Za-z0-9·（）()]+))?$",
        body,
    )
    if not match:
        return {}
    row = match.groupdict(default="")
    row["local_ratio"] = local_ratio
    return row


def _row_identity_matches_question(
    *,
    name: str,
    code: str,
    question: str,
    dosage: str = "",
) -> bool:
    compact_question = _compact(question).lower()
    compact_name = _compact(name).lower()
    compact_code = _compact(code).lower()
    compact_dosage = _compact(dosage).lower()
    if compact_code and compact_code in compact_question:
        return True
    if not compact_name or len(compact_name) < 2:
        return False
    name_parts = [
        part for part in re.split(r"[、,，/]+", compact_name)
        if part
    ]
    base_name_variants = list(dict.fromkeys([compact_name, *name_parts]))
    combined_variants = []
    if compact_dosage:
        clean_dosage = re.sub(r"[（(].*?[）)]", "", compact_dosage)
        for variant_name in base_name_variants:
            combined_variants.extend([variant_name + compact_dosage, variant_name + clean_dosage])
        if any(
            variant and len(variant) >= 2 and (
                variant in compact_question or compact_question in variant
            )
            for variant in combined_variants
        ):
            return True
        dosage_sensitive_question = any(
            _compact(term).lower() in compact_question for term in DRUG_DOSAGE_TERMS
        )
        if dosage_sensitive_question and compact_dosage not in compact_question:
            return False
    name_variants = []
    for variant_name in base_name_variants:
        name_variants.extend([variant_name, re.sub(r"[（(].*?[）)]", "", variant_name)])
    return any(
        variant and len(variant) >= 2 and (
            variant in compact_question or compact_question in variant
        )
        for variant in name_variants
    )


def _designated_question_has_specific_entity(question: str) -> bool:
    compact = _compact(question)
    if not compact:
        return False
    if re.search(r"[“\"'‘][^”\"'’]{2,40}[”\"'’]", str(question or "")):
        return True
    return any(
        token in compact
        for token in (
            "医院",
            "卫生服务中心",
            "卫生服务站",
            "医务室",
            "门诊部",
            "诊所",
            "药店",
            "药房",
            "医药",
            "有限公司",
            "普通合伙",
        )
    ) and not any(token in compact for token in ("哪些", "所有", "名单", "清单", "列表"))


def _extract_drug_dosage_definition(question: str, text: str) -> str:
    if not _question_asks_drug_dosage_definition(question):
        return ""
    paren_match = re.search(r"普通片剂[（(](?P<items>[^）)]{2,260})[）)]", text)
    if paren_match:
        items = paren_match.group("items").strip(" ：:，,")
        return _ensure_sentence(f"普通片剂包括{items}")
    match = re.search(
        r"普通片剂(?:包括|包含)[:：]?\s*(?P<items>[^。；;\n]{2,260})",
        text,
    )
    if match:
        items = match.group("items").strip(" ：:，,")
        return _ensure_sentence(f"普通片剂包括{items}")
    snippet = _best_snippet(text, ("普通片剂", "剂型", "片剂", "素片", "糖衣片", "薄膜衣片"))
    return _ensure_sentence(_clip(snippet, 260)) if "普通片剂" in snippet else ""


def _question_asks_drug_dosage_definition(question: str) -> bool:
    compact_question = _compact(question)
    return any(
        token in compact_question
        for token in ("普通片剂", "包含哪些剂型", "包括哪些剂型", "哪些剂型", "剂型归类", "剂型说明")
    )


def _drug_subject_for_question(*, name: str, dosage: str, question: str) -> str:
    compact_question = _compact(question).lower()
    compact_dosage = _compact(dosage).lower()
    name_parts = [
        part for part in re.split(r"[、,，/]+", str(name or ""))
        if part
    ]
    candidates = list(dict.fromkeys([str(name or ""), *name_parts]))
    for candidate in candidates:
        compact_candidate = _compact(candidate).lower()
        if not compact_candidate:
            continue
        if compact_dosage and f"{compact_candidate}{compact_dosage}" in compact_question:
            return f"{candidate}{dosage}"
        if compact_candidate in compact_question:
            return candidate + (f"（{dosage}）" if dosage and dosage not in candidate else "")
    if name:
        return name + (f"（{dosage}）" if dosage and dosage not in name else "")
    return "该药品"


def _question_asks_drug_remark(question: str) -> bool:
    compact = _compact(question)
    return any(
        token in compact
        for token in (
            "备注",
            "限定支付范围",
            "限定支付条件",
            "支付限制",
            "限制条件",
            "适应症",
            "方可支付",
            "哪些条件",
        )
    )


def _service_row_matches_question(*, name: str, code: str, text: str, question: str) -> bool:
    if _row_identity_matches_question(name=name, code=code, question=question):
        return True
    compact_question = _compact(question).lower()
    compact_text = _compact(text).lower()
    if "ct" in compact_question and (
        "ct" in compact_text or "计算机体层" in compact_text or "断层扫描" in compact_text
    ):
        return True
    if any(term in compact_question for term in ("胸部", "肺部")) and any(
        term in compact_text for term in ("胸部", "肺部", "ct", "计算机体层")
    ):
        return True
    service_entities = ("血常规", "b超", "彩超", "核磁", "磁共振", "mri", "x线", "dr")
    return any(term in compact_question and term in compact_text for term in service_entities)


def _fact(
    index: int,
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    *,
    fact_type: str,
    value: dict[str, Any],
    display_text: str,
    evidence_text: str,
    confidence: float,
) -> ExtractedFact:
    fact_value = dict(value or {})
    fact_value.setdefault("scenario_id", requirement.scenario_id)
    fact_value.setdefault("required_fields", list(requirement.required_fields))
    fact_value.setdefault(
        "matched_required_fields",
        _matched_required_fields(
            requirement,
            fact_value,
            display_text,
            evidence_text,
            evidence.content,
        ),
    )
    fact_value.setdefault("answer_action", requirement.answer_action)
    fact_value.setdefault("quote_verified", True)
    fact_value.setdefault(
        "support_role",
        "direct_answer" if confidence >= 0.75 else "supporting",
    )
    return ExtractedFact(
        fact_id=f"fact_{index}",
        requirement_id=requirement.requirement_id,
        slot_id=requirement.slot_id,
        fact_type=fact_type,
        value=fact_value,
        display_text=_ensure_sentence(_clip(display_text, 500)),
        evidence_refs=[evidence.evidence_id],
        source_refs=[evidence.source_ref],
        evidence_text=_clip(evidence_text, 800),
        confidence=confidence,
    )


def _matched_required_fields(
    requirement: AnswerRequirement,
    value: dict[str, Any],
    *texts: str,
) -> list[str]:
    required_fields = [str(item).strip() for item in requirement.required_fields if str(item).strip()]
    if not required_fields:
        return []
    value_text = _value_match_text(value)
    haystack = _field_match_text("\n".join([*texts, value_text]))
    matched: list[str] = []
    for field in required_fields:
        aliases = _required_field_aliases(field, value)
        if any(alias and _field_match_text(alias) in haystack for alias in aliases):
            matched.append(field)
    return list(dict.fromkeys(matched))


def _required_field_aliases(field: str, value: dict[str, Any]) -> tuple[str, ...]:
    aliases: dict[str, tuple[str, ...]] = {
        "直接结算费用范围": (
            "直接结算的住院、普通门诊和门诊慢特病医疗费用",
            "住院、普通门诊和门诊慢特病医疗费用",
            "住院普通门诊门诊慢特病医疗费用",
            "住院",
            "普通门诊",
            "门诊慢特病",
        ),
        "就医地支付范围": ("就医地支付范围", "就医地规定的支付范围", "支付范围原则上按就医地", "执行就医地规定的支付范围"),
        "参保地起付标准": ("参保地起付标准", "参保地规定的基本医疗保险基金起付标准", "起付标准"),
        "参保地支付比例": ("参保地支付比例", "参保地规定的基本医疗保险基金支付比例", "支付比例"),
        "参保地最高支付限额": ("参保地最高支付限额", "参保地规定的基本医疗保险基金最高支付限额", "最高支付限额"),
        "门诊慢特病病种范围": ("门诊慢特病病种范围", "门诊慢特病病种范围等有关政策", "病种范围"),
        "备案到就医地统筹地区": ("备案到就医地所在统筹地区", "备案到就医地所在的统筹地区", "就医地所在统筹地区"),
        "统筹地区内所有定点医药机构": ("统筹地区内所有定点医药机构", "所有定点医药机构"),
        "按规定就医结算": ("按规定就医结算", "相关规定就医结算", "就医结算"),
        "出院自费结算": ("出院自费结算", "自费结算", "自行垫付", "全额垫付"),
        "补办备案手续": ("补办备案手续", "补办备案", "补备案"),
        "参保地规定": ("参保地规定", "按参保地规定", "参保地政策"),
        "医保手工报销": ("医保手工报销", "手工报销"),
        "暂不能直接结算": ("暂不能直接结算", "暂不能实现异地直接结算", "不能实现异地直接结算"),
        "异地就医票据及相关报销材料": ("异地就医票据及相关报销材料", "票据及相关报销材料", "票据", "报销材料"),
        "单位或社保所提交": ("单位或社保所", "所属单位", "社保所", "单位"),
        "区医保经办机构手工报销": ("区医保经办机构申请手工报销", "区医保经办机构", "医保经办机构", "手工报销"),
        "住院标准报销": ("住院标准报销", "按住院标准", "按照住院标准"),
        "区县医保中心结算": ("区县医保中心", "区、县医保中心", "医保中心结算", "进行结算"),
        "手工报销路径": ("手工报销", "手工报销路径", "医保经办机构"),
        "诊断明确": ("诊断明确",),
        "传统DMARDs治疗3-6个月": ("传统DMARDs治疗3-6个月", "DMARDs治疗3-6个月", "DMARDs"),
        "疾病活动度下降低于50%": ("疾病活动度下降低于50%", "下降低于50%"),
        "风湿病专科医师处方": ("风湿病专科医师处方", "专科医师处方"),
        "药品名称": ("药品名称", "通用名", str(value.get("name") or "")),
        "医保类别": ("医保类别", str(value.get("category") or ""), str(value.get("insurance_category") or "")),
        "目录编号": ("目录编号", "编号", str(value.get("catalog_code") or "")),
        "项目名称": ("项目名称", str(value.get("name") or "")),
        "编码": ("编码", str(value.get("code") or "")),
        "计价单位": ("计价单位", str(value.get("unit") or "")),
        "收费标准": ("收费标准", "价格", str(value.get("price") or "")),
        "机构名称": ("机构名称", "药店名称", str(value.get("name") or "")),
        "状态": ("状态", "定点状态", str(value.get("status") or "")),
        "必要材料清单": ("必要材料", "required_items", "mentioned_items"),
        "法定办结时限": ("法定办结时限", "工作日", str(value.get("days") or "")),
        "起付标准": ("起付标准", "起付线"),
        "支付比例": ("支付比例", "报销比例"),
        "最高支付限额": ("最高支付限额", "封顶线"),
        "支付范围": ("支付范围", "基金支付"),
        "支付办法": ("支付办法", "先自负", "医保类别"),
        "政策依据": ("政策依据", "规定", "按照", "执行"),
        "备案规则": ("备案", "补办", "视同已备案"),
        "手工报销规则": ("手工报销", "零星报销", "报销"),
        "个人账户支付": ("个人账户支付", "个人帐户支付", "个人账户", "个人帐户"),
        "记账结算": ("记账结算", "记帐结算", "记账"),
        "门急诊费用审核结算凭证": ("门急诊费用审核结算凭证", "门急诊(药店)费用审核结算凭证", "审核结算凭证"),
        "银行费用不得列支基金": ("银行手续费", "银行票据工本费", "不得从基金中列支", "不列入基金支出"),
        "预付金黄色预警": ("预付金", "黄色预警", "70%"),
        "预付金红色预警": ("预付金", "红色预警", "90%"),
        "紧急调增流程": ("紧急调增", "预付金", "清算资金"),
        "费用协查信息": ("费用协查", "一次性跨省住院", "总费用超过3万元", "国家跨省异地就医管理子系统"),
        "待遇或参保规则": ("待遇", "参保人员范围", "参保范围", "城乡居民基本医疗保险", "医疗保险待遇"),
        "城乡老年人参保范围": ("城乡老年人", "男年满60周岁", "女年满50周岁", "无其它基本医疗保障"),
        "新生儿待遇起始": ("新生儿", "待遇享受", "待遇起始", "出生"),
        "待遇等待期": ("等待期", "待遇等待", "待遇享受"),
        "外埠户籍配偶参保材料": ("外埠户籍配偶", "配偶", "申请材料", "居住证"),
        "家庭医生签约首诊转诊": ("家庭医生签约", "首诊转诊", "转诊手续"),
        "外省市医疗费用目录标准": ("外省市", "国家及本市基本医疗保险有关规定", "医疗费用", "目录"),
        "慢性病长处方规则": ("长处方", "慢性病", "慢性病患者", "长期用药需求"),
        "慢病药品品种规格衔接": ("品种规格", "医联体", "用药衔接", "慢性病常用药品"),
        "医事服务费损失补偿": ("BJ-GBI", "医事服务费", "损失补偿", "年终清算"),
        "长处方月度通报": ("月度通报", "考核评分", "长处方政策落实"),
        "高血压糖尿病按人头付费": ("高血压", "糖尿病", "按人头付费"),
        "基金监管规则": ("监督检查", "基金使用", "服务协议", "智能监管", "异常情形审核"),
        "不属于基金支付范围处理": ("不属于医疗保障基金支付范围", "不予支付", "追回", "基金支付范围"),
        "拒不配合调查处置": ("拒不配合", "暂停联网结算", "锁卡", "重点监督检查"),
        "骗取基金处理程序": ("骗取医疗保障基金", "涉嫌骗保", "违法违规", "行政处罚"),
        "特殊病备案规则": ("特殊病种备案", "备案申报表", "医保办公室", "医疗保险经办机构"),
        "特殊病备案办理路径": ("特殊病种备案申报表", "本人选定", "定点医院", "医疗保险办公室", "参保区医疗保险经办机构"),
        "住院期间不得备案": ("住院期间", "办理出院手续后", "方可办理特殊病备案"),
        "外埠户籍特殊病备案条件": ("外埠户籍", "连续缴纳医疗保险费满24个月", "可办理门诊特殊病备案"),
        "特殊病备案名称调整": ("备案名称调整", "中重度哮喘生物制剂治疗", "中重度过敏性哮喘"),
        "特殊疾病范围规则": ("门诊特殊疾病范围", "新增门诊特殊疾病", "报销范围", "备案审核"),
        "新增门诊特殊疾病病种": ("重性精神病", "肺动脉高压", "耐多药结核", "C型尼曼匹克病", "中重度过敏性哮喘", "特发性肺纤维化"),
        "门诊特殊疾病报销范围": ("门诊特殊疾病报销范围", "门诊检查", "治疗", "相关药品", "基本医疗保险支付范围及标准"),
        "备案审核后享受待遇": ("备案审核", "享受门诊特殊疾病报销待遇", "未进行备案审核", "不纳入"),
        "上海医疗服务设施范围规则": ("医疗服务设施", "基金支付范围", "住院床位费", "急诊观察室床位费"),
        "住院床位费纳入范围": ("住院床位费", "基金支付范围", "支付标准"),
        "急诊观察室床位费纳入范围": ("急诊观察室床位费", "基金支付范围", "支付标准"),
        "政策有效期": ("有效期", "实施期限", "2026年7月31日"),
        "谈判药品或双通道规则": ("协议期内谈判药品", "谈判药品", "双通道", "电子处方"),
        "协议期内谈判药品乙类管理": ("协议期内谈判药品", "乙类", "基金支付范围"),
        "双通道药品供应约束": ("双通道", "电子处方", "一品两规", "药占比", "总额限制"),
    }
    direct_value = str(value.get(field) or "")
    return tuple(dict.fromkeys((field, direct_value, *aliases.get(field, ()))))


def _value_match_text(value: dict[str, Any]) -> str:
    fragments: list[str] = []
    for item in value.values():
        if isinstance(item, list):
            fragments.extend(str(part) for part in item)
        elif isinstance(item, dict):
            fragments.extend(str(part) for part in item.values())
        else:
            fragments.append(str(item))
    return " ".join(fragments)


def _field_match_text(text: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()《》“”\"'‘’\[\]【】/\\-]+", "", str(text or ""))


def _deduplicate_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    deduped: list[ExtractedFact] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        key = (fact.slot_id, re.sub(r"\s+", "", fact.display_text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _best_snippet(text: str, terms: tuple[str, ...] | list[str]) -> str:
    sentences = _sentences(text)
    if not sentences:
        return _clip(text, 360)
    scored = []
    clean_terms = [term for term in terms if term]
    for sentence in sentences:
        score = sum(1 for term in clean_terms if term in sentence)
        scored.append((score, len(sentence), sentence))
    best = max(scored, key=lambda item: (item[0], -item[1]))[2]
    return _clip(best, 360)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。；;])\s*", normalized)
    if len(parts) <= 1:
        return [normalized]
    return [part.strip() for part in parts if part.strip()]


def _first_unique(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _ordered_unique(items: list[tuple[int, str]]) -> list[str]:
    result: list[str] = []
    for _, name in sorted(items, key=lambda item: item[0] if item[0] >= 0 else 999999):
        if name not in result:
            result.append(name)
    return result


def _drop_subsumed_material_names(items: list[str]) -> list[str]:
    result: list[str] = []
    for name in items:
        if any(name != other and name in other for other in items):
            continue
        result.append(name)
    return result


def _format_catalog_category(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if value in {"甲", "乙"}:
        return f"{value}类"
    return value


def _join_cn(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "和" + items[-1]


def _ensure_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.endswith(("。", "；", ";", "！", "？")):
        return text
    return text + "。"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
