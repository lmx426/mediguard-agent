"""Sentence-window evidence judging and extraction for L3 policy text."""

from __future__ import annotations

import re
from typing import Any

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


TEXTUAL_CONTENT_TYPES = {"policy_text", "faq", "service_guide"}
DIRECT_SUPPORT_ROLE = "direct_answer"
SUPPORTING_ROLE = "supporting"


def build_sentence_windows(
    *,
    requirements: list[AnswerRequirement],
    evidence: list[NormalizedPolicyEvidence],
    filters: dict[str, Any],
    user_question: str,
    radius: int = 1,
) -> list[dict[str, Any]]:
    """Build slot-specific sentence windows from textual evidence chunks."""

    windows: list[dict[str, Any]] = []
    for requirement in requirements:
        if _structured_requirement(requirement):
            continue
        for item in evidence:
            if str(item.content_type or "") not in TEXTUAL_CONTENT_TYPES:
                continue
            sentences = _sentences(item.content)
            if not sentences:
                continue
            for index, sentence in enumerate(sentences):
                window_text = " ".join(
                    sentences[max(0, index - radius): min(len(sentences), index + radius + 1)]
                ).strip()
                if not window_text:
                    continue
                score, signals = rule_pre_score(
                    requirement=requirement,
                    evidence=item,
                    window_text=window_text,
                    filters=filters,
                    user_question=user_question,
                )
                if score < 0.18:
                    continue
                windows.append(
                    {
                        "window_id": f"{item.evidence_id}#w{index + 1:02d}:{requirement.need_id}",
                        "node_id": item.metadata.get("node_id") or item.evidence_id,
                        "evidence_id": item.evidence_id,
                        "source_ref": item.source_ref,
                        "source_url": item.source_url,
                        "title": item.title,
                        "need_id": requirement.need_id,
                        "need_text": requirement.need_text,
                        "answer_mode": requirement.answer_mode,
                        "slot_id": requirement.slot_id,
                        "requirement_id": requirement.requirement_id,
                        "slot_label": requirement.label,
                        "scenario_id": requirement.scenario_id,
                        "required_fields": list(requirement.required_fields),
                        "answer_action": requirement.answer_action,
                        "window_text": window_text[:1200],
                        "sentence_index": index,
                        "content_type": item.content_type,
                        "policy_domain": item.policy_domain,
                        "jurisdiction": item.jurisdiction,
                        "pre_score": round(score, 3),
                        "score_signals": signals,
                    }
                )
    return windows


def rule_pre_score(
    *,
    requirement: AnswerRequirement,
    evidence: NormalizedPolicyEvidence,
    window_text: str,
    filters: dict[str, Any],
    user_question: str,
) -> tuple[float, dict[str, Any]]:
    """Cheap coarse scoring. Final evidence status is decided after quote validation."""

    text = f"{evidence.title}\n{window_text}"
    compact_text = _compact(text)
    terms = [term for term in slot_coverage_terms(requirement.slot_id) if term]
    matched_terms = [term for term in terms if _compact(term) in compact_text]

    score = 0.0
    if terms:
        score += min(0.42, len(matched_terms) / max(len(terms), 1) * 0.42)
    else:
        score += 0.12

    slot_domains = set(slot_policy_domains(requirement.slot_id))
    evidence_domain = str(evidence.policy_domain or "")
    if not slot_domains or evidence_domain in slot_domains:
        score += 0.22

    requested_domains = set(_string_list(filters.get("policy_domain")))
    if not requested_domains or evidence_domain in requested_domains:
        score += 0.08

    requirement_content_types = set(_string_list(requirement.filters.get("content_type")))
    evidence_content_type = str(evidence.content_type or "")
    if not requirement_content_types or evidence_content_type in requirement_content_types:
        score += 0.08

    requested_jurisdictions = set(_string_list(filters.get("jurisdiction")))
    evidence_jurisdiction = str(evidence.jurisdiction or "")
    if not requested_jurisdictions or evidence_jurisdiction in requested_jurisdictions or evidence_jurisdiction == "national":
        score += 0.08

    question_entities = _question_entities(user_question)
    matched_entities = [entity for entity in question_entities if _compact(entity) in compact_text]
    if matched_entities:
        score += min(0.12, len(matched_entities) / max(len(question_entities), 1) * 0.12)

    text_len = len(window_text)
    if text_len > 900:
        score *= 0.8
    elif text_len > 600:
        score *= 0.9

    return min(score, 1.0), {
        "matched_terms": matched_terms[:12],
        "matched_entities": matched_entities[:8],
        "policy_domain_match": not slot_domains or evidence_domain in slot_domains,
        "content_type_match": not requirement_content_types or evidence_content_type in requirement_content_types,
        "jurisdiction_match": not requested_jurisdictions or evidence_jurisdiction in requested_jurisdictions or evidence_jurisdiction == "national",
    }


def select_candidate_windows(
    windows: list[dict[str, Any]],
    *,
    per_slot: int = 6,
    max_total: int = 24,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        grouped.setdefault(str(window.get("requirement_id") or window.get("slot_id") or ""), []).append(window)

    selected: list[dict[str, Any]] = []
    for _, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (
                float(item.get("pre_score") or 0.0),
                -int(item.get("sentence_index") or 0),
            ),
            reverse=True,
        )
        seen_keys: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for item in ordered:
            key = (
                str(item.get("evidence_id") or ""),
                re.sub(r"\s+", "", str(item.get("window_text") or "")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
            if len(deduped) >= per_slot:
                break
        selected.extend(deduped)
    return sorted(selected, key=lambda item: float(item.get("pre_score") or 0.0), reverse=True)[:max_total]


def windows_for_prompt(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bounded payload sent to the LLM judge."""

    return [
        {
            "window_id": item.get("window_id"),
            "need_id": item.get("need_id") or item.get("slot_id"),
            "need_text": item.get("need_text") or item.get("slot_label"),
            "slot_id": item.get("slot_id"),
            "slot_label": item.get("slot_label"),
            "answer_mode": item.get("answer_mode"),
            "scenario_id": item.get("scenario_id"),
            "required_fields": item.get("required_fields") or [],
            "answer_action": item.get("answer_action"),
            "window_text": item.get("window_text"),
            "title": item.get("title"),
            "jurisdiction": item.get("jurisdiction"),
            "policy_domain": item.get("policy_domain"),
            "content_type": item.get("content_type"),
            "pre_score": item.get("pre_score"),
            "score_signals": item.get("score_signals"),
        }
        for item in windows[:24]
    ]


def extract_verified_facts_from_judgement(
    *,
    requirements: list[AnswerRequirement],
    candidate_windows: list[dict[str, Any]],
    payload: dict[str, Any],
    fact_start_index: int = 1,
    match_start_index: int = 1,
) -> tuple[list[ExtractedFact], list[EvidenceMatch], dict[str, Any]]:
    """Coerce LLM slot-window judgements into quote-verified facts."""

    windows_by_id = {
        str(item.get("window_id") or ""): item
        for item in candidate_windows
        if item.get("window_id")
    }
    requirements_by_id = {item.requirement_id: item for item in requirements}
    requirements_by_slot = {item.slot_id: item for item in requirements}

    facts: list[ExtractedFact] = []
    matches: list[EvidenceMatch] = []
    rejected: list[dict[str, Any]] = []
    raw_results = _raw_window_results(payload)
    for result in raw_results[:48]:
        if not isinstance(result, dict):
            continue
        window_id = str(result.get("window_id") or "")
        window = windows_by_id.get(window_id)
        if window is None:
            rejected.append({"window_id": window_id, "reason": "unknown_window"})
            continue
        slot_id = str(
            result.get("need_id")
            or result.get("slot_id")
            or window.get("need_id")
            or window.get("slot_id")
            or ""
        )
        requirement = (
            requirements_by_id.get(str(window.get("requirement_id") or ""))
            or requirements_by_slot.get(slot_id)
        )
        if requirement is None:
            rejected.append({"window_id": window_id, "reason": "unknown_slot"})
            continue
        score = _float_between(result.get("llm_score") or result.get("score"), 0.0, 1.0, 0.0)
        support_role = _support_role(result.get("support_role"))
        if score < 0.55 or support_role == "irrelevant":
            rejected.append({"window_id": window_id, "reason": "low_score_or_irrelevant"})
            continue
        window_text = str(window.get("window_text") or "")
        fact_items = result.get("facts")
        if isinstance(fact_items, dict):
            fact_items = [fact_items]
        if not isinstance(fact_items, list):
            fact_items = []
        accepted_for_match: list[str] = []
        for fact_item in fact_items[:6]:
            if not isinstance(fact_item, dict):
                continue
            fact_text = _ensure_sentence(str(fact_item.get("fact") or "").strip())
            quote = str(
                fact_item.get("evidence_quote")
                or fact_item.get("quote")
                or fact_item.get("evidence_text")
                or ""
            ).strip()
            if not fact_text or not quote:
                rejected.append({"window_id": window_id, "reason": "empty_fact_or_quote"})
                continue
            if not quote_verified(quote, window_text):
                rejected.append(
                    {
                        "window_id": window_id,
                        "reason": "quote_not_found",
                        "quote": quote[:160],
                    }
                )
                continue
            matched_fields = _matched_required_fields(
                requirement,
                window_text,
                fact_text,
                quote,
                *[
                    str(item)
                    for item in _string_list(
                        fact_item.get("matched_required_fields")
                        or fact_item.get("covered_fields")
                    )
                ],
            )
            verified_support_role = support_role
            if (
                verified_support_role == DIRECT_SUPPORT_ROLE
                and requirement.required_fields
                and not matched_fields
            ):
                verified_support_role = SUPPORTING_ROLE
            fact_id = f"fact_{fact_start_index + len(facts)}"
            evidence_id = str(window.get("evidence_id") or "")
            source_ref = str(window.get("source_ref") or "")
            facts.append(
                ExtractedFact(
                    fact_id=fact_id,
                    requirement_id=requirement.requirement_id,
                    slot_id=requirement.slot_id,
                    fact_type=f"{requirement.fact_schema}_slot_window",
                    value={
                        "support_role": verified_support_role,
                        "llm_score": score,
                        "quote_verified": True,
                        "window_id": window_id,
                        "pre_score": window.get("pre_score"),
                        "scenario_id": requirement.scenario_id,
                        "required_fields": list(requirement.required_fields),
                        "matched_required_fields": matched_fields,
                        "answer_action": requirement.answer_action,
                    },
                    display_text=fact_text[:500],
                    evidence_refs=[evidence_id],
                    source_refs=[source_ref],
                    evidence_text=quote[:800],
                    confidence=score,
                )
            )
            accepted_for_match.append(quote[:240])
        if accepted_for_match:
            matches.append(
                EvidenceMatch(
                    match_id=f"match_{match_start_index + len(matches)}",
                    requirement_id=requirement.requirement_id,
                    evidence_id=str(window.get("evidence_id") or ""),
                    coverage_status=(
                        "covered"
                        if verified_support_role == DIRECT_SUPPORT_ROLE and score >= 0.75
                        else "partial"
                    ),
                    matched_spans=accepted_for_match[:3],
                    score=score,
                    reason=f"slot_window_{verified_support_role}",
                )
            )

    facts = _deduplicate_facts(facts)
    diagnostics = {
        "raw_result_count": len(raw_results),
        "verified_fact_count": len(facts),
        "verified_match_count": len(matches),
        "rejected_count": len(rejected),
        "rejected": rejected[:12],
    }
    return facts, matches, diagnostics


def quote_verified(quote: str, text: str) -> bool:
    """Return True only when the LLM quote is present in the source window."""

    quote_norm = _compact(quote)
    text_norm = _compact(text)
    return bool(quote_norm) and quote_norm in text_norm


def reindex_facts_and_matches(
    facts: list[ExtractedFact],
    matches: list[EvidenceMatch],
    *,
    fact_start_index: int = 1,
    match_start_index: int = 1,
) -> tuple[list[ExtractedFact], list[EvidenceMatch]]:
    fact_id_map: dict[str, str] = {}
    reindexed_facts: list[ExtractedFact] = []
    for offset, fact in enumerate(facts):
        new_id = f"fact_{fact_start_index + offset}"
        fact_id_map[fact.fact_id] = new_id
        reindexed_facts.append(fact.model_copy(update={"fact_id": new_id}))

    reindexed_matches: list[EvidenceMatch] = []
    for offset, match in enumerate(matches):
        reindexed_matches.append(
            match.model_copy(update={"match_id": f"match_{match_start_index + offset}"})
        )
    _ = fact_id_map
    return reindexed_facts, reindexed_matches


def textual_evidence(evidence: list[NormalizedPolicyEvidence]) -> list[NormalizedPolicyEvidence]:
    return [item for item in evidence if str(item.content_type or "") in TEXTUAL_CONTENT_TYPES]


def structured_evidence(evidence: list[NormalizedPolicyEvidence]) -> list[NormalizedPolicyEvidence]:
    return [item for item in evidence if str(item.content_type or "") == "table_row"]


def _raw_window_results(payload: dict[str, Any]) -> list[Any]:
    for key in ("window_results", "results", "slot_window_results", "judgements", "judgments"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _structured_requirement(requirement: AnswerRequirement) -> bool:
    return requirement.extractor_id in {
        "drug_catalog",
        "medical_service_price",
        "consumable_payment_scope",
        "designated_institution",
    } and "policy_text" not in _string_list(requirement.filters.get("content_type"))


def _support_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"direct", "direct_answer", "answer", "covered"}:
        return DIRECT_SUPPORT_ROLE
    if normalized in {"supporting", "support", "partial", "background"}:
        return SUPPORTING_ROLE
    return "irrelevant"


def _matched_required_fields(
    requirement: AnswerRequirement,
    *texts: str,
) -> list[str]:
    required_fields = [str(item).strip() for item in requirement.required_fields if str(item).strip()]
    if not required_fields:
        return []
    haystack = _field_match_text("\n".join(texts))
    matched: list[str] = []
    for field in required_fields:
        aliases = _required_field_aliases(field)
        if any(alias and _field_match_text(alias) in haystack for alias in aliases):
            matched.append(field)
    return list(dict.fromkeys(matched))


def _required_field_aliases(field: str) -> tuple[str, ...]:
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
        "个人账户支付": ("个人账户支付", "个人帐户支付", "个人账户", "个人帐户"),
        "记账结算": ("记账结算", "记帐结算", "记账"),
        "门急诊费用审核结算凭证": ("门急诊费用审核结算凭证", "门急诊(药店)费用审核结算凭证", "审核结算凭证"),
        "法定办结时限": ("法定办结时限", "工作日"),
        "政策依据": ("政策依据", "规定", "按照", "执行"),
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
    return tuple(dict.fromkeys((field, *aliases.get(field, ()))))


def _question_entities(question: str) -> list[str]:
    compact = _compact(question)
    entities: list[str] = []
    for match in re.finditer(r"[“\"'‘]([^”\"'’]{2,30})[”\"'’]", str(question or "")):
        entities.append(match.group(1))
    for token in (
        "北京",
        "上海",
        "跨省",
        "异地就医",
        "直接结算",
        "备案",
        "外埠就医",
        "记账结算",
        "急诊就医",
        "类风湿关节炎",
        "DMARDs",
        "基金监管",
        "拒不配合",
        "长处方",
        "长期处方",
        "城乡居民医保",
        "城乡老年人",
        "特殊病备案",
        "特殊疾病范围",
        "预付金",
        "费用协查",
        "银行手续费",
        "住院床位费",
        "急诊观察室床位费",
        "双通道",
        "谈判药品",
    ):
        if token in compact:
            entities.append(token)
    return list(dict.fromkeys(entities))[:12]


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？；;\n])\s*", normalized) if part.strip()]
    if len(parts) > 1:
        return parts
    if len(normalized) > 220:
        comma_parts = [part.strip() for part in re.split(r"(?<=，)\s*", normalized) if part.strip()]
        if len(comma_parts) > 1:
            return comma_parts
        return [normalized[index:index + 180] for index in range(0, len(normalized), 160)]
    return parts


def _deduplicate_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    result: list[ExtractedFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.slot_id, _compact(fact.display_text), _compact(fact.evidence_text))
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _ensure_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else text + "。"


def _float_between(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, minimum), maximum)


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _field_match_text(text: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()《》“”\"'‘’\[\]【】/\\-]+", "", str(text or ""))
