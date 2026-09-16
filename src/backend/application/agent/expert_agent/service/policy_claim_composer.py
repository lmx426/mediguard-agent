"""Claim-first answer composition for L3 Policy Expert."""

from __future__ import annotations

import re
from typing import Any

from src.backend.application.agent.expert_agent.schemas import (
    AnswerabilityCheck,
    ExpertAnalysisTask,
    PolicyEvidence,
)
from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerClaim,
    AnswerRequirement,
    ClaimCitation,
    ClaimFirstPolicyAnswer,
    EvidenceMatch,
    ExtractedFact,
)
from src.backend.application.agent.expert_agent.service.policy_answer_slots import (
    resolve_answer_requirements,
    normalize_answer_slot_ids,
)
from src.backend.application.agent.expert_agent.service.policy_information_needs import (
    build_information_need_requirements,
    normalize_answer_mode,
    normalize_information_needs,
)
from src.backend.application.agent.expert_agent.service.policy_fact_extractors import (
    extract_facts_for_requirements,
    normalize_policy_evidence,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    normalize_policy_filters,
)


def build_claim_first_policy_answer(
    *,
    task: ExpertAnalysisTask,
    evidence: list[PolicyEvidence],
    answerability: AnswerabilityCheck,
    question_slots: list[dict[str, Any]],
    information_needs: list[str] | None = None,
    filters: dict[str, Any],
    precomputed_requirements: list[AnswerRequirement] | None = None,
    precomputed_facts: list[ExtractedFact] | None = None,
    precomputed_matches: list[EvidenceMatch] | None = None,
    coverage_result: dict[str, Any] | None = None,
) -> ClaimFirstPolicyAnswer:
    """Build a deterministic, fact-grounded policy answer draft."""

    normalized_evidence = normalize_policy_evidence(evidence)
    slot_hints = _slot_hints_from_question_slots(
        question_slots,
        normalize_answer_slot_ids(answerability.covered_slots),
    )
    explicit_information_needs = normalize_information_needs(information_needs or [])
    requirements = precomputed_requirements
    if requirements is None:
        if explicit_information_needs:
            requirements = build_information_need_requirements(
                user_question=task.user_question,
                information_needs=explicit_information_needs,
                filters=normalize_policy_filters(filters),
                answer_mode=normalize_answer_mode(None, task.user_question, explicit_information_needs),
            )
        else:
            if slot_hints:
                requirements = resolve_answer_requirements(
                    user_question=task.user_question,
                    question_slots=slot_hints,
                    filters=normalize_policy_filters(filters),
                )
            else:
                requirements = build_information_need_requirements(
                    user_question=task.user_question,
                    information_needs=[str(task.user_question).strip() or "policy_question"],
                    filters=normalize_policy_filters(filters),
                    answer_mode=normalize_answer_mode(None, task.user_question, []),
                )
    if precomputed_facts is not None:
        facts = precomputed_facts
        matches = precomputed_matches or []
    else:
        facts, matches = extract_facts_for_requirements(requirements, normalized_evidence)
    information_needs = _information_needs_from_requirements(
        requirements,
        question_slots,
        task.user_question,
    )
    answer_mode = _answer_mode_from_question(task.user_question, information_needs)
    claim_plan = _claim_plan_from_coverage(coverage_result, requirements, facts)
    facts_for_claims = _facts_for_claim_plan(facts, claim_plan)
    claims, citations = _claims_and_citations(requirements, facts_for_claims, claim_plan)
    claims, citations = _filter_claims_to_direct_answer_scope(
        user_question=task.user_question,
        claims=claims,
        citations=citations,
    )
    if not claims and evidence and answerability.answerability != "insufficient":
        claims, citations = _fallback_claims_from_evidence(
            task=task,
            evidence=normalized_evidence,
            answerability=answerability,
            requirements=requirements,
        )
    claims, citations = _deduplicate_claims_and_renumber(claims, citations)
    markdown = _render_answer_markdown(claims, citations)
    source_refs = _source_refs_from_claims(claims) or [
        item.source_ref for item in evidence if item.source_ref
    ][:20]

    support_status = _support_status(
        requirements,
        claims,
        answerability,
        claim_plan,
        user_question=task.user_question,
    )
    if not markdown:
        markdown = _fallback_policy_answer(task, evidence, answerability)
    elif support_status == "partial" and "不作确定性结论" not in markdown:
        markdown = (
            markdown.rstrip()
            + " 当前可引用证据仅覆盖部分核验点，未覆盖部分不作确定性结论。"
        )
    return ClaimFirstPolicyAnswer(
        answer_mode=answer_mode,
        information_needs=information_needs,
        expert_answer=markdown[:2000],
        answer_markdown=markdown[:4000],
        answer_requirements=requirements,
        evidence_matches=matches,
        extracted_facts=facts,
        claims=claims,
        claim_plan=claim_plan,
        citations=citations,
        source_refs=source_refs[:20],
        support_status=support_status,
    )


def _slot_hints_from_question_slots(
    question_slots: list[dict[str, Any]],
    answerability_slots: list[str],
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in question_slots:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or "").strip()
        if slot_id and slot_id not in seen:
            hint: dict[str, Any] = {"slot_id": slot_id}
            required_fields = item.get("required_fields")
            if isinstance(required_fields, list) and required_fields:
                hint["required_fields"] = [
                    str(field).strip()
                    for field in required_fields
                    if str(field).strip()
                ]
            hints.append(hint)
            seen.add(slot_id)
    for slot_id in answerability_slots:
        if not slot_id or slot_id in seen:
            continue
        hints.append({"slot_id": slot_id})
        seen.add(slot_id)
    return hints


def _fallback_claims_from_evidence(
    *,
    task: ExpertAnalysisTask,
    evidence: list[NormalizedPolicyEvidence],
    answerability: AnswerabilityCheck,
    requirements: list[AnswerRequirement],
) -> tuple[list[AnswerClaim], list[ClaimCitation]]:
    if not evidence:
        return [], []
    fallback_text = _fallback_policy_answer(task, [PolicyEvidence.model_validate({
        "evidence_ref": item.evidence_id,
        "source_ref": item.source_ref,
        "title": item.title,
        "excerpt": item.content,
        "jurisdiction": item.jurisdiction,
        "policy_domain": item.policy_domain,
        "content_type": item.content_type,
        "source_url": item.source_url,
        "version": item.version,
        "used_for": "policy_answer",
        "metadata": item.metadata,
    }) for item in evidence], answerability)
    if not fallback_text:
        fallback_text = _ensure_sentence(_clip(evidence[0].content, 260))
    if not fallback_text:
        return [], []
    requirement = requirements[0] if requirements else None
    slot_id = requirement.slot_id if requirement is not None else "policy_basis"
    citation = ClaimCitation(
        citation_id="cit_1",
        label=1,
        claim_id="claim_1",
        fact_refs=[],
        evidence_refs=[],
        source_refs=[evidence[0].source_ref],
    )
    claim = AnswerClaim(
        claim_id="claim_1",
        requirement_id=requirement.requirement_id if requirement is not None else "req_1",
        slot_id=slot_id,
        text=fallback_text[:600],
        fact_refs=[],
        source_refs=[evidence[0].source_ref],
        citation_ids=["cit_1"],
        support_status="supported" if answerability.answerability == "answerable" else "partial",
    )
    return [claim], [citation]


def _information_needs_from_requirements(
    requirements: list[AnswerRequirement],
    question_slots: list[dict[str, Any]],
    user_question: str,
) -> list[str]:
    needs: list[str] = []
    for item in requirements:
        text = str(
            getattr(item, "need_text", "")
            or getattr(item, "label", "")
            or getattr(item, "question_span", "")
            or ""
        ).strip()
        if text:
            needs.append(text)
    if not needs:
        for item in question_slots:
            if not isinstance(item, dict):
                continue
            text = str(
                item.get("need_text")
                or item.get("label")
                or item.get("question_span")
                or item.get("slot_id")
                or ""
            ).strip()
            if text:
                needs.append(text)
    if not needs and str(user_question or "").strip():
        needs.append(str(user_question).strip())
    return list(dict.fromkeys(needs))[:12]


def _answer_mode_from_question(question: str, information_needs: list[str]) -> str:
    compact = _compact(question)
    if "比较" in compact or "对比" in compact:
        return "comparison"
    if any(token in compact for token in ("列出", "有哪些", "哪些", "清单", "目录")) and len(information_needs) <= 3:
        return "list"
    if any(token in compact for token in ("规则", "流程", "条件", "办理", "如何", "怎么", "步骤", "时限")):
        return "process_rule"
    if any(token in compact for token in ("为什么", "解释", "说明", "含义")):
        return "policy_explanation"
    if len(information_needs) == 1 and any(token in compact for token in ("是什么", "多少", "哪一个", "哪个")):
        return "fact_lookup"
    return "process_rule"


def _claims_and_citations(
    requirements: list[AnswerRequirement],
    facts: list[ExtractedFact],
    claim_plan: list[dict[str, Any]] | None = None,
) -> tuple[list[AnswerClaim], list[ClaimCitation]]:
    facts_by_requirement: dict[str, list[ExtractedFact]] = {}
    for fact in facts:
        facts_by_requirement.setdefault(fact.requirement_id, []).append(fact)

    plan_by_requirement = _plan_by_requirement(claim_plan or [])
    citations_by_source: dict[str, ClaimCitation] = {}
    claims: list[AnswerClaim] = []
    for requirement in requirements:
        plan_item = plan_by_requirement.get(requirement.requirement_id)
        if plan_item and plan_item.get("generation") == "deny":
            continue
        req_facts = facts_by_requirement.get(requirement.requirement_id, [])
        if not req_facts:
            continue
        for fact in req_facts[:5]:
            claim_text = _claim_text_for_requirement(requirement, fact)
            if not claim_text:
                continue
            claim_id = f"claim_{len(claims) + 1}"
            citation_ids: list[str] = []
            for source_ref in fact.source_refs[:4]:
                citation = citations_by_source.get(source_ref)
                if citation is None:
                    citation = ClaimCitation(
                        citation_id=f"cit_{len(citations_by_source) + 1}",
                        label=len(citations_by_source) + 1,
                        claim_id=claim_id,
                        fact_refs=[fact.fact_id],
                        evidence_refs=fact.evidence_refs[:4],
                        source_refs=[source_ref],
                    )
                    citations_by_source[source_ref] = citation
                else:
                    citation.fact_refs = list(dict.fromkeys([*citation.fact_refs, fact.fact_id]))[:12]
                    citation.evidence_refs = list(
                        dict.fromkeys([*citation.evidence_refs, *fact.evidence_refs])
                    )[:12]
                    if citation.claim_id is None:
                        citation.claim_id = claim_id
                citation_ids.append(citation.citation_id)
            claims.append(
                AnswerClaim(
                    claim_id=claim_id,
                    requirement_id=requirement.requirement_id,
                    slot_id=requirement.slot_id,
                    text=claim_text,
                    fact_refs=[fact.fact_id],
                    source_refs=fact.source_refs[:8],
                    citation_ids=list(dict.fromkeys(citation_ids))[:8],
                    support_status=(
                        "partial"
                        if plan_item and plan_item.get("generation") == "allow_with_qualification"
                        else "supported"
                    ),
                )
            )

    citations = sorted(citations_by_source.values(), key=lambda item: item.label)
    return claims[:30], citations[:30]


def _claim_plan_from_coverage(
    coverage_result: dict[str, Any] | None,
    requirements: list[AnswerRequirement],
    facts: list[ExtractedFact],
) -> list[dict[str, Any]]:
    if not isinstance(coverage_result, dict):
        return []
    raw_plan = coverage_result.get("claim_plan")
    if not isinstance(raw_plan, list):
        raw_plan = []
    requirements_by_slot = {item.slot_id: item for item in requirements}
    facts_by_slot: dict[str, list[str]] = {}
    for fact in facts:
        facts_by_slot.setdefault(fact.slot_id, []).append(fact.fact_id)

    plan: list[dict[str, Any]] = []
    for raw_item in raw_plan[:30]:
        if not isinstance(raw_item, dict):
            continue
        slot_id = str(raw_item.get("slot_id") or "")
        requirement = requirements_by_slot.get(slot_id)
        if requirement is None:
            continue
        generation = str(raw_item.get("generation") or "").strip()
        if generation not in {"allow", "allow_with_qualification", "deny"}:
            status = str(raw_item.get("support_status") or raw_item.get("coverage_status") or "")
            generation = (
                "allow"
                if status == "covered"
                else "allow_with_qualification"
                if status == "partial"
                else "deny"
            )
        fact_refs = _string_list(raw_item.get("fact_refs")) or facts_by_slot.get(slot_id, [])
        direct_fact_refs = _string_list(raw_item.get("direct_fact_refs"))
        background_fact_refs = _string_list(raw_item.get("background_fact_refs"))
        plan.append(
            {
                "slot_id": slot_id,
                "requirement_id": str(raw_item.get("requirement_id") or requirement.requirement_id),
                "label": str(raw_item.get("label") or requirement.label),
                "scenario_id": str(raw_item.get("scenario_id") or requirement.scenario_id),
                "answer_action": str(raw_item.get("answer_action") or requirement.answer_action),
                "required_fields": _string_list(raw_item.get("required_fields") or requirement.required_fields),
                "covered_fields": _string_list(raw_item.get("covered_fields")),
                "partial_fields": _string_list(raw_item.get("partial_fields")),
                "missing_fields": _string_list(raw_item.get("missing_fields")),
                "fact_refs": list(dict.fromkeys(fact_refs))[:8],
                "direct_fact_refs": list(dict.fromkeys(direct_fact_refs))[:8],
                "background_fact_refs": list(dict.fromkeys(background_fact_refs))[:8],
                "evidence_refs": _string_list(raw_item.get("evidence_refs"))[:8],
                "generation": generation,
                "support_status": str(raw_item.get("support_status") or raw_item.get("coverage_status") or ""),
            }
        )
    return plan


def _facts_for_claim_plan(
    facts: list[ExtractedFact],
    claim_plan: list[dict[str, Any]],
) -> list[ExtractedFact]:
    if not claim_plan:
        return facts
    allowed_slots: set[str] = set()
    allowed_fact_refs_by_slot: dict[str, set[str]] = {}
    required_fields_by_slot: dict[str, set[str]] = {}
    for item in claim_plan:
        generation = str(item.get("generation") or "")
        slot_id = str(item.get("slot_id") or "")
        if generation == "deny" or not slot_id:
            continue
        allowed_slots.add(slot_id)
        refs = _string_list(item.get("direct_fact_refs")) or _string_list(item.get("fact_refs"))
        if refs:
            allowed_fact_refs_by_slot.setdefault(slot_id, set()).update(refs)
        required_fields_by_slot.setdefault(slot_id, set()).update(
            _string_list(item.get("covered_fields"))
            or _string_list(item.get("required_fields"))
        )

    filtered: list[ExtractedFact] = []
    blocked_roles = {"supporting_background", "distractor", "irrelevant"}
    for fact in facts:
        role = str(fact.value.get("support_role") or "").strip()
        if role in blocked_roles:
            continue
        if fact.slot_id not in allowed_slots:
            continue
        slot_refs = allowed_fact_refs_by_slot.get(fact.slot_id, set())
        if slot_refs and fact.fact_id not in slot_refs:
            matched_fields = set(_string_list(fact.value.get("matched_required_fields")))
            required_fields = required_fields_by_slot.get(fact.slot_id, set())
            if not matched_fields.intersection(required_fields):
                continue
        filtered.append(fact)
    return filtered


def _plan_by_requirement(
    claim_plan: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in claim_plan:
        requirement_id = str(item.get("requirement_id") or "")
        if requirement_id and requirement_id not in result:
            result[requirement_id] = item
    return result


def _claim_text_for_requirement(
    requirement: AnswerRequirement,
    fact: ExtractedFact,
) -> str:
    text = _ensure_sentence(fact.display_text)
    if requirement.slot_id == "drug_catalog" and _question_asks_drug_dosage_definition(
        requirement.question_span
    ):
        scoped = (
            _drug_dosage_definition_from_text(fact.display_text)
            or _drug_dosage_definition_from_text(fact.evidence_text)
        )
        return scoped
    return text


def _drug_dosage_definition_from_text(text: str) -> str:
    raw = str(text or "")
    paren_match = re.search(r"普通片剂[（(](?P<items>[^）)]{2,260})[）)]", raw)
    if paren_match:
        items = paren_match.group("items")
        return _ensure_sentence(f"普通片剂包括{_clean_dosage_items(items)}")
    match = re.search(r"普通片剂(?:包括|包含)[:：]?\s*(?P<items>[^。；;\n]{2,260})", raw)
    if not match:
        return ""
    items = _clean_dosage_items(match.group("items"))
    return _ensure_sentence(f"普通片剂包括{items}") if items else ""


def _clean_dosage_items(value: str) -> str:
    items = str(value or "").strip(" ：:，,、。；;")
    items = re.split(
        r"[，,、]?(?:硬胶囊|软胶囊|肠溶胶囊|缓释片|口服溶液剂|外用溶液剂|气雾剂|注射剂)",
        items,
        maxsplit=1,
    )[0]
    return items.strip(" ：:，,、。；;")


def _question_asks_drug_dosage_definition(question: str) -> bool:
    compact = _compact(question)
    return any(
        token in compact
        for token in ("普通片剂", "包含哪些剂型", "包括哪些剂型", "哪些剂型", "剂型归类", "剂型说明")
    )


def _deduplicate_claims_and_renumber(
    claims: list[AnswerClaim],
    citations: list[ClaimCitation],
) -> tuple[list[AnswerClaim], list[ClaimCitation]]:
    if not claims:
        return claims, citations
    merged: list[AnswerClaim] = []
    seen: dict[str, int] = {}
    for claim in claims:
        key = _claim_key(claim.text)
        if key in seen:
            index = seen[key]
            existing = merged[index]
            merged[index] = existing.model_copy(
                update={
                    "fact_refs": list(dict.fromkeys([*existing.fact_refs, *claim.fact_refs]))[:8],
                    "source_refs": list(dict.fromkeys([*existing.source_refs, *claim.source_refs]))[:8],
                    "citation_ids": list(dict.fromkeys([*existing.citation_ids, *claim.citation_ids]))[:8],
                }
            )
            continue
        seen[key] = len(merged)
        merged.append(claim.model_copy(update={"claim_id": f"claim_{len(merged) + 1}"}))
    return _renumber_citations(merged, citations)


def _claim_key(text: str) -> str:
    return _compact(str(text or "").strip(" 。；;,.，"))


def _filter_claims_to_direct_answer_scope(
    *,
    user_question: str,
    claims: list[AnswerClaim],
    citations: list[ClaimCitation],
) -> tuple[list[AnswerClaim], list[ClaimCitation]]:
    """Keep the main answer focused on explicitly asked question slots.

    Policy RAG often retrieves strongly related background clauses. Those
    clauses remain available in policy_evidence, but the primary expert answer
    should not promote them into claims unless the user actually asked for that
    scenario.
    """

    if not claims:
        return claims, citations
    compact = _compact(user_question)
    asks_manual_reimbursement = any(
        token in compact
        for token in (
            "手工报销",
            "零星报销",
            "报销材料",
            "申请报销",
            "自费结算",
            "垫付",
            "外埠就医",
        )
    )
    asks_emergency = any(token in compact for token in ("急诊", "门急诊", "留观"))
    asks_remote_benefit_split = (
        (
            any(token in compact for token in ("就医地", "参保地"))
            and any(
                token in compact
                for token in ("目录", "支付范围", "待遇", "起付", "支付比例", "最高支付限额", "分工")
            )
        )
        or (
            any(token in compact for token in ("跨省", "异地就医", "异地"))
            and "直接结算" in compact
            and any(token in compact for token in ("医疗费用支付规则", "费用支付规则", "支付规则", "如何支付", "怎样支付"))
        )
    )
    asks_special_disease = any(
        token in compact
        for token in ("门诊特殊疾病", "特殊疾病", "门诊慢特病", "慢特病", "门特", "特殊病", "特病")
    )
    asks_long_term_remote = any(token in compact for token in ("长期居住", "常驻", "长期异地"))
    asks_temporary_out_payment_ratio = any(
        token in compact
        for token in ("临时外出", "转诊", "报销比例", "支付比例降幅", "降幅", "待遇降低", "降低多少")
    )

    filtered: list[AnswerClaim] = []
    for claim in claims:
        text = _compact(claim.text)
        if (
            not asks_manual_reimbursement
            and any(token in text for token in ("手工报销", "社保所", "所属单位", "单位(社保所)", "单位（社保所）"))
        ):
            continue
        if (
            not asks_emergency
            and any(token in text for token in ("急诊留观", "门急诊", "急诊抢救"))
        ):
            continue
        if (
            not asks_remote_benefit_split
            and "就医地" in text
            and "参保地" in text
            and any(token in text for token in ("支付范围", "起付标准", "支付比例", "最高支付限额"))
        ):
            continue
        if (
            not asks_remote_benefit_split
            and "直接结算" in text
            and any(token in text for token in ("覆盖住院", "普通门诊", "门诊慢特病", "费用范围"))
        ):
            continue
        if (
            not asks_special_disease
            and claim.slot_id in {"special_disease_filing", "special_disease_scope", "policy_basis"}
            and any(token in text for token in ("特殊病", "门诊特殊疾病", "门诊慢特病"))
        ):
            continue
        if not asks_long_term_remote and "长期居住" in text:
            continue
        if (
            not asks_temporary_out_payment_ratio
            and any(token in text for token in ("降幅", "不超过10个百分点", "不超过20个百分点", "低于参保地"))
        ):
            continue
        filtered.append(claim)

    if not filtered:
        return claims, citations
    return _renumber_citations(filtered, citations)


def _renumber_citations(
    claims: list[AnswerClaim],
    citations: list[ClaimCitation],
) -> tuple[list[AnswerClaim], list[ClaimCitation]]:
    old_by_source: dict[str, ClaimCitation] = {}
    for citation in citations:
        for source_ref in citation.source_refs:
            old_by_source[source_ref] = citation

    next_label = 1
    citation_by_source: dict[str, ClaimCitation] = {}
    new_claims: list[AnswerClaim] = []
    for claim in claims:
        citation_ids: list[str] = []
        for source_ref in claim.source_refs[:4]:
            citation = citation_by_source.get(source_ref)
            if citation is None:
                old = old_by_source.get(source_ref)
                citation = ClaimCitation(
                    citation_id=f"cit_{next_label}",
                    label=next_label,
                    claim_id=claim.claim_id,
                    fact_refs=list(dict.fromkeys(
                        [*claim.fact_refs, *(old.fact_refs if old else [])]
                    ))[:12],
                    evidence_refs=list(old.evidence_refs if old else [])[:12],
                    source_refs=[source_ref],
                )
                citation_by_source[source_ref] = citation
                next_label += 1
            citation_ids.append(citation.citation_id)
        new_claims.append(claim.model_copy(update={"citation_ids": citation_ids}))
    return new_claims, sorted(citation_by_source.values(), key=lambda item: item.label)


def _render_answer_markdown(
    claims: list[AnswerClaim],
    citations: list[ClaimCitation],
) -> str:
    if not claims:
        return ""
    citation_by_id = {citation.citation_id: citation for citation in citations}
    rendered: list[str] = []
    for claim in claims:
        labels = [
            citation_by_id[citation_id].label
            for citation_id in claim.citation_ids
            if citation_id in citation_by_id
        ]
        suffix = "".join(f"[{label}]" for label in labels)
        text = claim.text.rstrip()
        rendered.append(f"{text}{suffix}")
    return " ".join(rendered)


def _fallback_policy_answer(
    task: ExpertAnalysisTask,
    evidence: list[PolicyEvidence],
    answerability: AnswerabilityCheck,
) -> str:
    if not evidence:
        return "当前没有足够可引用的政策证据回答该问题。"
    titles = "、".join(item.title for item in evidence[:3] if item.title)
    prefix = "根据已检索到的政策依据"
    if titles:
        prefix += f"（{titles}）"
    if answerability.answerability == "partial":
        covered = "、".join(answerability.covered_slots[:4]) or "部分政策依据"
        missing = "、".join(answerability.missing_slots[:4]) or "其余关键政策口径"
        return (
            f"{prefix}，当前可支持{covered}；但{missing}证据不足，"
            "相关部分不作确定性结论。"
        )
    return f"{prefix}，已找到与“{task.user_question}”相关的可引用政策证据。"


def _support_status(
    requirements: list[AnswerRequirement],
    claims: list[AnswerClaim],
    answerability: AnswerabilityCheck,
    claim_plan: list[dict[str, Any]] | None = None,
    *,
    user_question: str = "",
) -> str:
    if not claims:
        return "unsupported"
    if claim_plan:
        answered = {
            claim.requirement_id for claim in claims
            if claim.support_status in {"supported", "partial"}
        }
        required_plan = [
            item for item in claim_plan
            if str(item.get("requirement_id") or "")
            and (
                _slot_asked_by_question(user_question, str(item.get("slot_id") or ""))
                or str(item.get("requirement_id") or "") in answered
            )
        ]
        if any(
            str(item.get("generation") or "") == "deny"
            and str(item.get("requirement_id") or "") not in answered
            for item in required_plan
        ):
            return "partial"
        if any(
            str(item.get("generation") or "") == "allow_with_qualification"
            and str(item.get("requirement_id") or "") not in answered
            for item in required_plan
        ):
            return "partial"
        planned_requirements = {
            str(item.get("requirement_id") or "")
            for item in required_plan
            if str(item.get("generation") or "") in {"allow", "allow_with_qualification"}
        }
        if planned_requirements and planned_requirements.issubset(answered):
            return "supported"
        return "partial"
    answered = {claim.requirement_id for claim in claims if claim.support_status == "supported"}
    required = {
        requirement.requirement_id for requirement in requirements
        if requirement.required
    }
    if required and required.issubset(answered) and answerability.answerability != "partial":
        return "supported"
    if required and required.issubset(answered):
        return "supported"
    return "partial"


def _slot_asked_by_question(user_question: str, slot_id: str) -> bool:
    compact = _compact(user_question)
    if not compact:
        return True
    asks_remote_filing_scope = (
        any(token in compact for token in ("备案成功", "备案后", "办理备案"))
        and any(token in compact for token in ("哪些机构", "哪些定点", "定点医药机构", "定点医疗机构", "统筹地区"))
    )
    asks_remote_self_pay = (
        any(token in compact for token in ("自费结算", "自行垫付", "全额垫付"))
        and any(token in compact for token in ("补办备案", "补办备案手续"))
        and "手工报销" in compact
    )
    asks_remote_emergency_observation = any(token in compact for token in ("急诊留观", "留观费用", "异地急诊留观"))
    asks_manual_scenario = any(token in compact for token in ("外埠就医", "外地就医", "易地安置")) or (
        "记账" in compact
        and any(token in compact for token in ("定点医药机构", "定点医疗机构", "定点零售药店"))
    ) or any(token in compact for token in ("急诊就医", "未出示社保卡", "医保电子凭证", "没带医保凭证"))
    if slot_id == "remote_benefit_split":
        return (
            (
                any(token in compact for token in ("就医地", "参保地"))
                and any(
                    token in compact
                    for token in ("目录", "支付范围", "待遇", "起付", "支付比例", "最高支付限额", "分工")
                )
            )
            or (
                any(token in compact for token in ("跨省", "异地就医", "异地"))
                and "直接结算" in compact
                and any(token in compact for token in ("医疗费用支付规则", "费用支付规则", "支付规则", "如何支付", "怎样支付"))
            )
        )
    if slot_id == "remote_filing_institution_scope":
        return asks_remote_filing_scope
    if slot_id == "foreign_treatment_manual_reimbursement":
        return any(token in compact for token in ("外埠就医", "外地就医", "易地安置"))
    if slot_id == "account_settlement_voucher":
        return "记账" in compact and any(token in compact for token in ("定点医药机构", "定点医疗机构", "定点零售药店"))
    if slot_id == "emergency_manual_reimbursement_materials":
        return any(token in compact for token in ("急诊就医", "未出示社保卡", "医保电子凭证", "没带医保凭证"))
    if slot_id == "remote_self_pay_filing_manual_reimbursement":
        return asks_remote_self_pay
    if slot_id == "remote_emergency_observation_reimbursement":
        return asks_remote_emergency_observation
    if slot_id == "remote_settlement_management":
        return any(
            token in compact
            for token in ("银行手续费", "银行票据", "工本费", "预付金", "黄色预警", "红色预警", "紧急调增", "费用协查")
        )
    if slot_id == "benefit_policy":
        return any(
            token in compact
            for token in ("城乡居民医保", "城乡老年人", "参保范围", "参保资格", "新生儿", "等待期", "外埠户籍配偶", "家庭医生", "首诊转诊", "外省市目录")
        )
    if slot_id == "chronic_long_prescription_policy":
        return any(
            token in compact
            for token in ("长期处方", "长处方", "慢性病", "高血压", "糖尿病", "BJ-GBI", "医事服务费", "月度通报", "品种规格", "医联体")
        )
    if slot_id == "fund_supervision_policy":
        return any(
            token in compact
            for token in ("基金监管", "监督检查", "拒不配合", "暂停联网结算", "锁卡", "骗取基金", "涉嫌骗保", "不属于基金支付范围", "异常情形审核")
        )
    if slot_id == "special_disease_filing_policy":
        return any(
            token in compact
            for token in ("特殊病备案", "特殊病种备案", "门诊特殊病备案", "门诊特殊疾病备案", "备案申报表", "医保办公室", "医保办", "病种名称", "中重度哮喘", "外埠户籍", "24个月")
        )
    if slot_id == "special_disease_scope_policy":
        return any(
            token in compact
            for token in ("特殊疾病范围", "门诊特殊疾病范围", "新增病种", "重性精神病", "肺动脉高压", "未备案", "报销范围")
        )
    if slot_id == "shanghai_service_facility_scope":
        return any(token in compact for token in ("医疗服务设施", "住院床位费", "急诊观察室床位费", "床位费", "实施期限", "有效期"))
    if slot_id == "negotiated_drug_double_channel":
        return any(token in compact for token in ("协议期内谈判药品", "谈判药品", "双通道", "电子处方", "一品两规", "药占比", "总额限制"))
    if slot_id == "designated_institution":
        return any(token in compact for token in ("定点机构", "定点医院", "定点药店", "定点医药机构", "定点医疗机构", "定点零售药店", "机构编码", "药店编码", "定点状态"))
    if slot_id == "manual_reimbursement":
        return (
            any(token in compact for token in ("手工报销", "零星报销", "报销材料", "申请报销"))
            and not asks_manual_scenario
            and not asks_remote_self_pay
            and not asks_remote_emergency_observation
        )
    if slot_id == "remote_filing":
        return (
            any(token in compact for token in ("备案", "补备案", "急诊例外", "急诊抢救", "视同备案", "视同已备案"))
            and not asks_remote_filing_scope
            and not asks_remote_self_pay
            and not asks_remote_emergency_observation
        )
    if slot_id == "required_materials":
        return any(token in compact for token in ("必要材料", "申请材料", "提交哪些材料", "需要哪些材料", "核验哪些材料", "报销材料"))
    if slot_id == "statutory_processing_time":
        return any(token in compact for token in ("办结时限", "办理时限", "多少工作日", "多久办结"))
    if slot_id == "drug_catalog":
        return any(token in compact for token in ("药品", "药物", "医保药品", "目录编号", "医保类别", "限定支付"))
    if slot_id == "medical_service_price":
        return any(token in compact for token in ("医疗服务", "诊疗项目", "收费标准", "计价单位", "CT", "ct"))
    if slot_id == "consumable_payment_scope":
        return any(token in compact for token in ("医用耗材", "耗材", "支付范围", "支付办法"))
    if slot_id == "benefit_params":
        return any(token in compact for token in ("起付线", "起付标准", "支付比例", "报销比例", "封顶线", "最高支付限额", "待遇参数"))
    if slot_id == "special_disease_payment_condition":
        return any(token in compact for token in ("特殊疾病", "门诊慢特病", "类风湿", "DMARDs", "限定支付条件", "方可支付"))
    if slot_id == "policy_basis":
        has_specific_slot = any(
            (
                asks_remote_filing_scope,
                asks_remote_self_pay,
                asks_remote_emergency_observation,
                asks_manual_scenario,
                any(
                    token in compact
                    for token in (
                        "药品",
                        "医疗服务",
                        "医用耗材",
                        "特殊疾病",
                        "门诊慢特病",
                        "基金监管",
                        "监督检查",
                        "长期处方",
                        "长处方",
                        "城乡居民医保",
                        "定点医疗机构",
                        "定点零售药店",
                        "床位费",
                        "双通道",
                    )
                ),
            )
        )
        return (
            not has_specific_slot
            and any(token in compact for token in ("政策", "依据", "规定", "规则", "口径", "要点"))
        )
    return True


def _source_refs_from_claims(claims: list[AnswerClaim]) -> list[str]:
    refs: list[str] = []
    for claim in claims:
        refs.extend(claim.source_refs)
    return list(dict.fromkeys(ref for ref in refs if ref))


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _ensure_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else text + "。"
