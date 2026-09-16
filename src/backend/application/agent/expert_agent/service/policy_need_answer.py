"""Fast information-need-native evidence matching and answer composition."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.backend.application.agent.expert_agent.schemas import PolicyEvidence
from src.backend.application.agent.expert_agent.service.policy_need_contracts import (
    ClaimCitation,
    InformationNeedRequirement,
    NeedAnswer,
    NeedClaim,
    NeedEvidenceMatch,
    NeedFact,
    NeedFirstPolicyAnswer,
    NormalizedPolicyEvidence,
)


TEXTUAL_CONTENT_TYPES = {"policy_text", "faq", "service_guide"}
INTERNAL_FIELD_MARKERS = (
    "source_id",
    "node_id",
    "field_key",
    "row_id",
    "case_relevance",
    "content_type",
    "资料标题:",
    "来源ID:",
    "metadata",
)
METADATA_LINE_PREFIXES = (
    "资料标题",
    "地区",
    "政策领域",
    "来源ID",
    "source_id",
    "source_url",
    "node_id",
    "field_key",
    "row_id",
    "case_relevance",
    "dataset",
    "fetched_at",
    "官方来源",
    "page",
)
QUESTION_NOISE = (
    "请问",
    "请告诉我",
    "告诉我",
    "需要",
    "哪些",
    "什么",
    "如何",
    "怎么",
    "是否",
    "分别",
    "相关",
    "政策",
    "规定",
    "要求",
    "审核",
    "核验",
)
BUSINESS_TERMS = (
    "手工报销",
    "报销材料",
    "材料清单",
    "费用收据",
    "费用清单",
    "处方底方",
    "诊疗证明",
    "结算凭证",
    "门诊",
    "住院",
    "急诊留观",
    "急诊就医",
    "外埠就医",
    "异地就医",
    "跨省",
    "直接结算",
    "备案",
    "起付标准",
    "支付比例",
    "最高支付限额",
    "医保类别",
    "目录编号",
    "分类",
    "剂型",
    "限定支付",
    "机构编码",
    "定点机构",
    "收费标准",
    "计价单位",
    "办理时限",
    "工作日",
    "特殊疾病",
    "慢特病",
    "长处方",
    "基金监管",
)
SUBJECT_TERM_GROUPS = (
    ("药品", "药物", "用药", "药品目录", "医保类别", "剂型", "限定支付"),
    ("医用耗材", "耗材"),
    ("CT", "ct", "医疗服务项目", "诊疗项目", "计价单位", "收费标准"),
    ("定点机构", "定点医院", "定点药店", "机构编码"),
)
MATERIAL_TERMS = (
    "诊疗证明",
    "诊断证明",
    "处方底方",
    "处方",
    "费用清单",
    "费用明细",
    "费用收据",
    "专用收据",
    "结算凭证",
    "身份证",
    "社会保障卡",
    "社保卡",
    "申请表",
    "审批单",
)


def build_information_need_requirements(
    *,
    user_question: str,
    information_needs: list[str],
    filters: dict[str, Any],
    answer_mode: str,
) -> list[InformationNeedRequirement]:
    """Build stable requirements without classifying information points."""

    requirements: list[InformationNeedRequirement] = []
    seen: list[str] = []
    for raw_need in information_needs[:12]:
        need_text = str(raw_need or "").strip()
        if not need_text or any(
            _information_needs_equivalent(need_text, existing)
            for existing in seen
        ):
            continue
        seen.append(need_text)
        index = len(requirements) + 1
        digest = hashlib.sha256(need_text.encode("utf-8")).hexdigest()[:10]
        requirements.append(
            InformationNeedRequirement(
                need_id=f"need_{index:02d}_{digest}",
                need_text=need_text[:240],
                question_span=need_text[:300],
                answer_mode=_answer_mode(answer_mode),
                required=True,
                filters=dict(filters or {}),
            )
        )
    if not requirements:
        fallback = str(user_question or "").strip() or "相关政策信息"
        digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:10]
        requirements.append(
            InformationNeedRequirement(
                need_id=f"need_01_{digest}",
                need_text=fallback[:240],
                question_span=fallback[:300],
                answer_mode=_answer_mode(answer_mode),
                required=True,
                filters=dict(filters or {}),
            )
        )
    return requirements


def normalize_policy_evidence(evidence: list[PolicyEvidence]) -> list[NormalizedPolicyEvidence]:
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


def build_information_need_answer(
    *,
    user_question: str,
    information_needs: list[str],
    answer_mode: str,
    filters: dict[str, Any],
    evidence: list[PolicyEvidence],
) -> NeedFirstPolicyAnswer:
    requirements = build_information_need_requirements(
        user_question=user_question,
        information_needs=information_needs,
        filters=filters,
        answer_mode=answer_mode,
    )
    normalized = normalize_policy_evidence(evidence)
    facts: list[NeedFact] = []
    matches: list[NeedEvidenceMatch] = []
    for requirement in requirements:
        need_facts, need_matches = _resolve_need(
            requirement=requirement,
            user_question=user_question,
            evidence=normalized,
            fact_start=len(facts) + 1,
            match_start=len(matches) + 1,
        )
        facts.extend(need_facts)
        matches.extend(need_matches)

    facts = _allocate_unique_facts(requirements, facts)
    selected_evidence = {
        (fact.need_id, evidence_ref)
        for fact in facts
        for evidence_ref in fact.evidence_refs
    }
    needs_with_facts = {fact.need_id for fact in facts}
    matches = [
        match
        for match in matches
        if (match.need_id, match.evidence_id) in selected_evidence
        or (
            match.need_id not in needs_with_facts
            and match.coverage_status == "partial"
        )
    ]
    facts = [
        fact.model_copy(update={"fact_id": f"fact_{index}"})
        for index, fact in enumerate(facts, start=1)
    ]
    matches = [
        match.model_copy(update={"match_id": f"match_{index}"})
        for index, match in enumerate(matches, start=1)
    ]

    need_answers, claims, citations = _compose_need_answers(requirements, facts)
    answer_markdown = render_information_need_markdown(
        need_answers,
        citations,
        answer_mode=answer_mode,
    )
    source_refs = list(
        dict.fromkeys(
            source_ref
            for answer in need_answers
            for source_ref in answer.source_refs
            if source_ref
        )
    )[:20]
    supported = sum(answer.status == "supported" for answer in need_answers)
    partial = sum(answer.status == "partial" for answer in need_answers)
    conflicted = any(answer.status == "conflicted" for answer in need_answers)
    if conflicted:
        support_status = "conflicted"
    elif supported == len(need_answers) and need_answers:
        support_status = "supported"
    elif supported or partial:
        support_status = "partial"
    else:
        support_status = "unsupported"
    claim_plan = [
        {
            "need_id": answer.need_id,
            "need_text": answer.need_text,
            "generation": (
                "allow"
                if answer.status == "supported"
                else "allow_with_qualification"
                if answer.status == "partial"
                else "deny"
            ),
            "support_status": answer.status,
            "fact_refs": answer.fact_refs,
            "source_refs": answer.source_refs,
        }
        for answer in need_answers
    ]
    expert_answer = answer_markdown or "当前没有足够可引用的政策证据回答该问题。"
    return NeedFirstPolicyAnswer(
        answer_mode=_answer_mode(answer_mode),
        information_needs=[item.need_text for item in requirements],
        expert_answer=expert_answer[:2000],
        answer_markdown=answer_markdown[:4000],
        answer_requirements=requirements,
        evidence_matches=matches[:80],
        extracted_facts=facts[:60],
        need_answers=need_answers,
        claims=claims,
        claim_plan=claim_plan,
        citations=citations,
        source_refs=source_refs,
        support_status=support_status,
    )


def _resolve_need(
    *,
    requirement: InformationNeedRequirement,
    user_question: str,
    evidence: list[NormalizedPolicyEvidence],
    fact_start: int,
    match_start: int,
) -> tuple[list[NeedFact], list[NeedEvidenceMatch]]:
    candidates: list[tuple[float, NeedFact]] = []
    partial_matches: list[NeedEvidenceMatch] = []
    for item in evidence:
        if not _jurisdiction_compatible(requirement.filters, item.jurisdiction):
            continue
        if str(item.content_type or "") == "table_row":
            fact = _fact_from_table_row(requirement, item, user_question)
        elif str(item.content_type or "") in TEXTUAL_CONTENT_TYPES or not item.content_type:
            fact = _fact_from_text(requirement, item, user_question)
        else:
            fact = None
        if fact is not None:
            candidates.append((fact.confidence, fact))
            continue
        score = _relevance_score(requirement.need_text, f"{item.title}\n{item.content}")
        if score >= 0.22:
            partial_matches.append(
                NeedEvidenceMatch(
                    match_id=f"match_{match_start + len(partial_matches)}",
                    need_id=requirement.need_id,
                    evidence_id=item.evidence_id,
                    coverage_status="partial",
                    support_role="supporting",
                    matched_spans=[],
                    score=min(score, 0.74),
                    reason="topic_match_without_user_safe_fact",
                )
            )

    selected: list[NeedFact] = []
    seen_text: set[str] = set()
    for _, fact in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = _compact(fact.user_facing_text)
        if not key or key in seen_text:
            continue
        seen_text.add(key)
        selected.append(fact.model_copy(update={"fact_id": f"fact_{fact_start + len(selected)}"}))
        if len(selected) >= 6:
            break
    matches = [
        NeedEvidenceMatch(
            match_id=f"match_{match_start + index}",
            need_id=requirement.need_id,
            evidence_id=fact.evidence_refs[0],
            coverage_status="covered" if fact.confidence >= 0.72 else "partial",
            support_role=fact.support_role,
            matched_spans=[fact.evidence_quote[:300]],
            score=fact.confidence,
            reason=f"content_driven_{fact.fact_type}",
        )
        for index, fact in enumerate(selected)
    ]
    if not matches:
        matches = partial_matches[:4]
    return selected, matches


def _fact_from_table_row(
    requirement: InformationNeedRequirement,
    evidence: NormalizedPolicyEvidence,
    user_question: str,
) -> NeedFact | None:
    fields = _field_map(evidence.content)
    need_text = requirement.need_text
    need_context = f"{user_question}\n{need_text}"
    scope_text = "\n".join(
        str(fields.get(key) or "")
        for key in ("required_when", "evidence_text", "scope", "remark", "remarks")
    )
    if _scope_conflict(need_context, scope_text or evidence.content):
        return None

    material_name = _field(fields, "material_name", "材料名称")
    if material_name:
        if not _asks_materials(need_context):
            return None
        if not _material_row_supports_need(need_context, fields, material_name):
            return None
        text = f"需提交{_clean_value(material_name)}。"
        return _need_fact(
            requirement,
            evidence,
            fact_type="material_row",
            value={"items": [_clean_value(material_name)]},
            user_text=text,
            quote=_field_quote(evidence.content, "material_name", material_name),
            confidence=0.93,
        )

    drug_name = _field(fields, "drug_name", "药品名称", "name")
    catalog_no = _field(fields, "catalog_no", "normalized_catalog_no", "目录编号")
    if drug_name and _row_entity_matches(need_context, drug_name, catalog_no):
        values = {
            "name": _clean_value(drug_name),
            "classification": _clean_value(_field(fields, "classification_name", "分类名称")),
            "insurance_class": _clean_value(_field(fields, "insurance_class", "医保类别")),
            "catalog_no": _clean_value(catalog_no),
            "dosage_form": _clean_value(_field(fields, "dosage_form", "剂型")),
            "remark": _clean_value(_field(fields, "remark", "remarks", "限定支付")),
        }
        text = _drug_text(need_context, values)
        return _need_fact(
            requirement,
            evidence,
            fact_type="drug_catalog_row",
            value=values,
            user_text=text,
            quote=_business_quote(evidence.content, values.values()),
            confidence=0.96,
        ) if text else None

    institution_name = _field(fields, "institution_name", "机构名称", "pharmacy_name")
    institution_code = _field(fields, "institution_code", "机构编码", "code")
    if institution_name and _row_entity_matches(need_context, institution_name, institution_code):
        values = {
            "name": _clean_value(institution_name),
            "code": _clean_value(institution_code),
            "district": _clean_value(_field(fields, "district", "区县")),
            "institution_type": _clean_value(_field(fields, "institution_type", "机构类型")),
            "grade": _clean_value(_field(fields, "grade", "等级")),
        }
        details = [f"机构编码为{values['code']}" if values["code"] else ""]
        if values["institution_type"]:
            details.append(f"机构类型为{values['institution_type']}")
        if values["grade"]:
            details.append(f"等级为{values['grade']}")
        text = f"{values['name']}在当前定点机构数据中有记录"
        if any(details):
            text += "，" + "，".join(item for item in details if item)
        text += "。"
        return _need_fact(
            requirement,
            evidence,
            fact_type="designated_institution_row",
            value=values,
            user_text=text,
            quote=_business_quote(evidence.content, values.values()),
            confidence=0.97,
        )

    service_name = _field(fields, "service_name", "item_name", "project_name", "项目名称")
    service_code = _field(fields, "service_code", "item_code", "project_code", "编码")
    if service_name and _row_entity_matches(need_context, service_name, service_code):
        values = {
            "name": _clean_value(service_name),
            "code": _clean_value(service_code),
            "unit": _clean_value(_field(fields, "unit", "pricing_unit", "计价单位")),
            "price": _clean_value(_field(fields, "price", "standard_price", "收费标准")),
        }
        parts = [values["name"]]
        if values["code"]:
            parts.append(f"编码为{values['code']}")
        if values["unit"]:
            parts.append(f"计价单位为{values['unit']}")
        if values["price"]:
            parts.append(f"收费标准为{values['price']}")
        return _need_fact(
            requirement,
            evidence,
            fact_type="medical_service_row",
            value=values,
            user_text="，".join(parts) + "。",
            quote=_business_quote(evidence.content, values.values()),
            confidence=0.96,
        )
    return None


def _fact_from_text(
    requirement: InformationNeedRequirement,
    evidence: NormalizedPolicyEvidence,
    user_question: str,
) -> NeedFact | None:
    sentences = _sentences(evidence.content)
    need_context = f"{user_question}\n{requirement.need_text}"
    best: tuple[float, str, str] | None = None
    for index, sentence in enumerate(sentences):
        if _looks_internal(sentence):
            continue
        window = " ".join(sentences[max(0, index - 1): min(len(sentences), index + 2)])
        if _scope_conflict(need_context, sentence):
            continue
        if _subject_conflict(requirement.need_text, window):
            continue
        score = _relevance_score(need_context, f"{evidence.title}\n{window}")
        score += _filter_bonus(requirement.filters, evidence)
        if score < 0.3:
            continue
        if best is None or score > best[0]:
            best = (min(score, 0.92), sentence, window)
    if best is None:
        return None
    score, sentence, window = best
    user_text = _user_text_from_sentence(need_context, sentence, window)
    if not user_text or _looks_internal(user_text):
        return None
    return _need_fact(
        requirement,
        evidence,
        fact_type="sentence_window",
        value={"window_radius": 1},
        user_text=user_text,
        quote=sentence,
        confidence=max(0.62, score),
    )


def _compose_need_answers(
    requirements: list[InformationNeedRequirement],
    facts: list[NeedFact],
) -> tuple[list[NeedAnswer], list[NeedClaim], list[ClaimCitation]]:
    facts_by_need: dict[str, list[NeedFact]] = {}
    for fact in facts:
        facts_by_need.setdefault(fact.need_id, []).append(fact)
    answers: list[NeedAnswer] = []
    claims: list[NeedClaim] = []
    citations: list[ClaimCitation] = []
    citation_by_claim_source: dict[tuple[str, str], ClaimCitation] = {}
    for requirement in requirements:
        need_facts = facts_by_need.get(requirement.need_id, [])
        if not need_facts:
            answers.append(
                NeedAnswer(
                    need_id=requirement.need_id,
                    need_text=requirement.need_text,
                    status="missing",
                    answer_text="当前可引用证据不足以回答该信息点。",
                    missing_reason="no_direct_user_safe_evidence",
                )
            )
            continue
        answer_text, used_facts = _merge_fact_text(requirement.need_text, need_facts)
        if not answer_text or _looks_internal(answer_text):
            answers.append(
                NeedAnswer(
                    need_id=requirement.need_id,
                    need_text=requirement.need_text,
                    status="missing",
                    answer_text="当前证据无法安全整理为面向用户的结论。",
                    missing_reason="user_visible_output_gate_rejected",
                )
            )
            continue
        fact_refs = [fact.fact_id for fact in used_facts]
        source_refs = list(
            dict.fromkeys(ref for fact in used_facts for ref in fact.source_refs if ref)
        )[:8]
        claim_id = f"claim_{len(claims) + 1}"
        citation_ids: list[str] = []
        for source_ref in source_refs:
            citation_key = (claim_id, source_ref)
            citation = citation_by_claim_source.get(citation_key)
            if citation is None:
                citation = ClaimCitation(
                    citation_id=f"cit_{len(citation_by_claim_source) + 1}",
                    label=len(citation_by_claim_source) + 1,
                    claim_id=claim_id,
                    fact_refs=fact_refs[:12],
                    evidence_refs=list(
                        dict.fromkeys(
                            ref for fact in used_facts for ref in fact.evidence_refs if ref
                        )
                    )[:12],
                    source_refs=[source_ref],
                )
                citation_by_claim_source[citation_key] = citation
                citations.append(citation)
            citation_ids.append(citation.citation_id)
        status = "supported" if max(fact.confidence for fact in used_facts) >= 0.70 else "partial"
        answers.append(
            NeedAnswer(
                need_id=requirement.need_id,
                need_text=requirement.need_text,
                status=status,
                answer_text=answer_text,
                fact_refs=fact_refs[:12],
                source_refs=source_refs,
                citation_ids=citation_ids[:8],
            )
        )
        claims.append(
            NeedClaim(
                claim_id=claim_id,
                need_id=requirement.need_id,
                need_text=requirement.need_text,
                text=answer_text,
                fact_refs=fact_refs[:12],
                source_refs=source_refs,
                citation_ids=citation_ids[:8],
                support_status=status,
            )
        )
    return answers, claims, citations


def _merge_fact_text(need_text: str, facts: list[NeedFact]) -> tuple[str, list[NeedFact]]:
    material_items: list[str] = []
    material_facts: list[NeedFact] = []
    for fact in facts:
        raw_items = fact.value.get("items")
        if isinstance(raw_items, list):
            cleaned_items = [_clean_value(item) for item in raw_items if _clean_value(item)]
            if cleaned_items:
                material_items.extend(cleaned_items)
                material_facts.append(fact)
    if material_items and _asks_materials(need_text):
        unique = list(dict.fromkeys(material_items))[:12]
        return f"需要提交{'、'.join(unique)}。", material_facts[:12]
    texts: list[str] = []
    used_facts: list[NeedFact] = []
    seen: set[str] = set()
    for fact in facts:
        text = _ensure_sentence(fact.user_facing_text)
        key = _compact(text)
        if not text or _looks_internal(text) or key in seen:
            continue
        seen.add(key)
        texts.append(text)
        used_facts.append(fact)
        if len(texts) >= 2:
            break
    return " ".join(texts)[:900], used_facts


def render_information_need_markdown(
    answers: list[NeedAnswer],
    citations: list[ClaimCitation],
    *,
    answer_mode: str = "list",
) -> str:
    """Render user-facing text while keeping need-level citation closure."""

    citation_by_id = {item.citation_id: item for item in citations}
    rendered: list[str] = []
    for answer in answers:
        labels = [
            citation_by_id[citation_id].label
            for citation_id in answer.citation_ids
            if citation_id in citation_by_id
        ]
        suffix = "".join(f"[{label}]" for label in labels)
        rendered.append(f"{answer.answer_text}{suffix}")
    if answer_mode in {"comparison", "policy_explanation"}:
        return " ".join(rendered)
    multiple = len(rendered) > 1
    return "\n".join(
        f"{index}. {text}" if multiple else text
        for index, text in enumerate(rendered, start=1)
    )


def _user_text_from_sentence(need_text: str, sentence: str, window: str) -> str:
    if _asks_materials(need_text):
        items = [term for term in MATERIAL_TERMS if term in window]
        if items:
            return f"需要提交{'、'.join(list(dict.fromkeys(items))[:12])}。"
    cleaned = _clean_policy_text(sentence)
    if len(cleaned) > 420:
        clauses = [item.strip() for item in re.split(r"[；;]", cleaned) if item.strip()]
        cleaned = "；".join(clauses[:2])
    return _ensure_sentence(cleaned[:560]) if cleaned else ""


def _need_fact(
    requirement: InformationNeedRequirement,
    evidence: NormalizedPolicyEvidence,
    *,
    fact_type: str,
    value: dict[str, Any],
    user_text: str,
    quote: str,
    confidence: float,
) -> NeedFact:
    return NeedFact(
        fact_id="pending",
        need_id=requirement.need_id,
        fact_type=fact_type,
        value=value,
        user_facing_text=_ensure_sentence(user_text),
        evidence_refs=[evidence.evidence_id],
        source_refs=[evidence.source_ref],
        evidence_quote=str(quote or "")[:1000],
        support_role="direct_answer",
        confidence=min(max(confidence, 0.0), 1.0),
    )


def _drug_text(need_text: str, values: dict[str, str]) -> str:
    name = values.get("name") or "该药品"
    parts: list[str] = []
    if "分类" in need_text and values.get("classification"):
        parts.append(f"分类为{values['classification']}")
    if any(token in need_text for token in ("医保类别", "甲类", "乙类")) and values.get("insurance_class"):
        category = values["insurance_class"]
        parts.append(f"医保类别为{category if category.endswith('类') else category + '类'}")
    if any(token in need_text for token in ("目录编号", "编号")) and values.get("catalog_no"):
        parts.append(f"目录编号为{values['catalog_no']}")
    if "剂型" in need_text and values.get("dosage_form"):
        parts.append(f"剂型为{values['dosage_form']}")
    if any(token in need_text for token in ("限定支付", "备注")) and values.get("remark"):
        parts.append(f"限定支付说明为{values['remark']}")
    if not parts:
        for key, label in (
            ("classification", "分类"),
            ("insurance_class", "医保类别"),
            ("catalog_no", "目录编号"),
        ):
            if values.get(key):
                value = values[key]
                if key == "insurance_class" and not value.endswith("类"):
                    value += "类"
                parts.append(f"{label}为{value}")
    return f"{name}{'，'.join(parts)}。" if parts else ""


def _filter_bonus(filters: dict[str, Any], evidence: NormalizedPolicyEvidence) -> float:
    bonus = 0.0
    jurisdictions = set(_string_list(filters.get("jurisdiction")))
    if not jurisdictions or evidence.jurisdiction in jurisdictions or evidence.jurisdiction == "national":
        bonus += 0.08
    domains = set(_string_list(filters.get("policy_domain")))
    if not domains or evidence.policy_domain in domains:
        bonus += 0.06
    content_types = set(_string_list(filters.get("content_type")))
    if not content_types or evidence.content_type in content_types:
        bonus += 0.04
    return bonus


def _information_needs_equivalent(left: str, right: str) -> bool:
    left_normalized = _compact(_strip_noise(left))
    right_normalized = _compact(_strip_noise(right))
    if not left_normalized or not right_normalized:
        return False
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    if len(shorter) >= 8 and shorter in longer:
        return True
    left_bigrams = _bigrams(left_normalized)
    right_bigrams = _bigrams(right_normalized)
    union = left_bigrams | right_bigrams
    return bool(union) and len(left_bigrams & right_bigrams) / len(union) >= 0.82


def _allocate_unique_facts(
    requirements: list[InformationNeedRequirement],
    facts: list[NeedFact],
) -> list[NeedFact]:
    """Assign an identical user-visible fact to the best matching need only."""

    requirement_by_id = {item.need_id: item for item in requirements}
    grouped: dict[str, list[tuple[int, NeedFact]]] = {}
    for index, fact in enumerate(facts):
        grouped.setdefault(_compact(fact.user_facing_text), []).append((index, fact))

    keep_indexes: set[int] = set()
    for group in grouped.values():
        need_ids = {fact.need_id for _, fact in group}
        if len(need_ids) <= 1:
            keep_indexes.update(index for index, _ in group)
            continue
        best_index, _best_fact = max(
            group,
            key=lambda item: (
                _relevance_score(
                    requirement_by_id.get(item[1].need_id, requirements[0]).need_text,
                    f"{item[1].user_facing_text}\n{item[1].evidence_quote}",
                ),
                item[1].confidence,
                -item[0],
            ),
        )
        keep_indexes.add(best_index)
    return [fact for index, fact in enumerate(facts) if index in keep_indexes]


def _material_row_supports_need(
    need_text: str,
    fields: dict[str, str],
    material_name: str,
) -> bool:
    """Require table rows to support the relation asked by the information need."""

    need = _compact(need_text)
    business_text = _compact(
        "\n".join(
            [
                material_name,
                _field(fields, "required_when", "适用条件"),
                _field(fields, "evidence_text", "证据文本"),
                _field(fields, "scope", "适用范围"),
                _field(fields, "remark", "remarks", "备注"),
            ]
        )
    )
    asks_replacement = any(
        token in need for token in ("替代", "代替", "缺少", "缺失", "补正", "补充核验")
    )
    if asks_replacement:
        explicit_relation = any(
            token in business_text
            for token in ("替代", "代替", "缺少时", "缺失时", "可补充", "补正材料")
        )
        if not explicit_relation:
            return False
        requested_objects = [
            token
            for token in ("挂号", "门诊病历", "病历", "票据", "收据", "处方", "诊断证明")
            if token in need
        ]
        if requested_objects and not any(token in business_text for token in requested_objects):
            return False
    if "核验" in need and any(token in need for token in ("挂号", "病历", "缺少", "替代")):
        if not any(token in business_text for token in ("核验", "审核", "查验", "确认")):
            return False
    return True


def _relevance_score(need_text: str, evidence_text: str) -> float:
    need = _compact(need_text)
    text = _compact(evidence_text)
    if not need or not text:
        return 0.0
    terms = [term for term in BUSINESS_TERMS if term in need]
    exact = [term for term in terms if term in text]
    score = min(0.5, len(exact) * 0.16)
    need_bigrams = _bigrams(_strip_noise(need))
    text_bigrams = _bigrams(text)
    if need_bigrams:
        score += min(0.42, len(need_bigrams.intersection(text_bigrams)) / len(need_bigrams) * 0.7)
    for quoted in re.findall(r"[《“](.*?)[》”]", need_text):
        if _compact(quoted) and _compact(quoted) in text:
            score += 0.25
    for code in re.findall(r"[A-Za-z0-9★()（）-]{3,}", need_text):
        if _compact(code).lower() in text.lower():
            score += 0.2
    return min(score, 1.0)


def _scope_conflict(need_text: str, evidence_text: str) -> bool:
    need = _compact(need_text)
    evidence = _compact(evidence_text)
    qualifiers = (
        (("异地", "外埠", "跨省", "易地"), ("异地", "外埠", "跨省", "易地"), ("本地", "普通门诊")),
        (("急诊留观", "急诊就医", "急诊抢救"), ("急诊留观", "急诊就医", "急诊抢救"), ("普通门诊", "住院")),
        (("特殊疾病", "慢特病", "门特"), ("特殊疾病", "慢特病", "门特"), ("普通门诊", "住院")),
    )
    for need_terms, evidence_terms, broad_terms in qualifiers:
        if (
            not any(term in need for term in need_terms)
            and any(term in evidence for term in evidence_terms)
            and not any(term in evidence for term in broad_terms)
        ):
            return True
    if "门诊" in need and "住院" in evidence and "门诊" not in evidence:
        return True
    if "住院" in need and "门诊" in evidence and "住院" not in evidence:
        return True
    return False


def _subject_conflict(need_text: str, evidence_text: str) -> bool:
    need = _compact(need_text)
    evidence = _compact(evidence_text)
    need_groups = [
        index
        for index, terms in enumerate(SUBJECT_TERM_GROUPS)
        if any(term in need for term in terms)
    ]
    if not need_groups:
        return False
    if any(
        term in evidence
        for index in need_groups
        for term in SUBJECT_TERM_GROUPS[index]
    ):
        return False
    return any(
        term in evidence
        for index, terms in enumerate(SUBJECT_TERM_GROUPS)
        if index not in need_groups
        for term in terms
    )


def _asks_materials(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("材料", "提交", "票据", "凭证", "清单", "收据", "处方"))


def _jurisdiction_compatible(
    filters: dict[str, Any],
    evidence_jurisdiction: str | None,
) -> bool:
    requested = set(_string_list(filters.get("jurisdiction")))
    evidence = str(evidence_jurisdiction or "").strip().lower()
    if not requested or not evidence or evidence == "national":
        return True
    return evidence in requested


def _row_entity_matches(question: str, name: str, code: str) -> bool:
    compact_question = _compact(question).lower()
    compact_name = _compact(name).lower()
    compact_code = _compact(code).lower()
    if compact_code and compact_code in compact_question:
        return True
    if compact_name and compact_name in compact_question:
        return True
    parts = [item for item in re.split(r"[、,，/\s]+", compact_name) if len(item) >= 2]
    return bool(parts) and any(item in compact_question for item in parts)


def _field_map(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([^:\n：]{1,40})[:：]\s*([^\n]+)", str(text or "")):
        key = match.group(1).strip()
        value = match.group(2).strip(" ；。")
        if key and value:
            fields[key] = value
    return fields


def _field(fields: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = fields.get(key)
        if value:
            return str(value)
    return ""


def _field_quote(text: str, key: str, value: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}[:：]\s*{re.escape(value)}\s*$", text)
    return match.group(0).strip() if match else value


def _business_quote(text: str, values: Any) -> str:
    selected = [str(value) for value in values if str(value or "").strip()]
    lines = [line.strip() for line in str(text or "").splitlines()]
    matched = [line for line in lines if any(value in line for value in selected)]
    return "\n".join(matched[:8]) or "；".join(selected[:8])


def _sentences(text: str) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in raw.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or _metadata_line(line):
            continue
        lines.append(line)
    parts: list[str] = []
    for line in lines:
        split = [item.strip() for item in re.split(r"(?<=[。！？；;])\s*", line) if item.strip()]
        parts.extend(split or [line])
    return parts[:80]


def _metadata_line(line: str) -> bool:
    head = line.split(":", 1)[0].split("：", 1)[0].strip()
    return head in METADATA_LINE_PREFIXES


def _clean_policy_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"^(?:答复|答|回复)[:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?:主办|承办|政府网站标识码|ICP备案序号)[:：].*$", "", cleaned)
    return cleaned.strip(" ；;")


def _looks_internal(text: str) -> bool:
    compact = str(text or "").lower()
    return any(marker.lower() in compact for marker in INTERNAL_FIELD_MARKERS)


def _clean_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ；;。")


def _answer_mode(value: str) -> str:
    allowed = {"fact_lookup", "list", "process_rule", "policy_explanation", "comparison"}
    return value if value in allowed else "process_rule"


def _strip_noise(text: str) -> str:
    output = text
    for token in QUESTION_NOISE:
        output = output.replace(token, "")
    return output


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9★]+", "", str(text or ""))
    return {compact[index:index + 2] for index in range(max(len(compact) - 1, 0))}


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _ensure_sentence(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""
    return value if value.endswith(("。", "！", "？", ".", "!", "?")) else value + "。"


def _compact(text: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()《》“”\"'‘’\[\]【】/\\-]+", "", str(text or ""))
