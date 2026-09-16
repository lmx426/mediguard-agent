"""Build a question-focused answer context for Caser generation."""

from __future__ import annotations

import json
from typing import Any

from src.backend.application.agent.case_agent.artifacts import artifact_results
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentSource,
)

from ._errors import state_error


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "case_number": ("案件编号", "编号", "case_number"),
    "case_type": ("案件类型", "类型", "case_type"),
    "review_status": (
        "审核状态",
        "审核通过",
        "通过了",
        "通过了吗",
        "过了吗",
        "是否通过",
        "有没有通过",
        "已通过",
        "人工初审",
        "人工复审",
        "申诉",
        "状态",
        "review_status",
    ),
    "claimant_ref": ("脱敏申报人编号", "申报人编号", "claimant_ref"),
    "access_method": ("接入方式", "来源方式", "access_method"),
    "insured_region": ("参保地", "参保地区", "在哪里参保", "在哪参保", "哪里参保", "insured_region"),
    "treatment_region": ("就医地", "治疗地", "在哪里就医", "在哪就医", "哪里就医", "treatment_region"),
    "visit_type": ("就医类型", "visit_type"),
    "claim_mode": ("报销方式", "claim_mode"),
    "direct_settlement": ("直接结算", "直结", "direct_settlement"),
    "filing_status": ("备案状态", "备案", "filing_status"),
    "emergency_material_status": ("急诊材料", "急诊材料状态", "emergency_material_status"),
    "diagnosis": ("诊断", "诊断结论", "diagnosis"),
    "claimant_code": ("申报人编码", "申报人编号", "claimant_code"),
    "gender": ("性别", "申报人性别", "gender"),
    "age_group": ("年龄段", "年龄", "age_group"),
    "insurance_type": ("参保类型", "医保类型", "insurance_type"),
    "chronic_condition_tags": ("慢病标签", "慢病", "chronic_condition_tags"),
    "allergy_history": ("过敏史", "过敏", "allergy_history"),
    "risk_level": ("风险等级", "是否高风险", "risk_level"),
    "overall_strength": ("综合风险提示强度", "风险评分", "综合风险评分", "overall_strength"),
    "申报总费用": ("申报总费用", "总费用"),
    "本次审批金额": ("本次审批金额", "审批金额"),
    "统筹支付金额": ("统筹支付金额", "统筹支付"),
    "个人账户金额": ("个人账户金额", "个人账户"),
    "药品费": ("药品费",),
    "检查费": ("检查费",),
    "治疗费": ("治疗费",),
    "医用材料费": ("医用材料费",),
}


SECTION_LABELS = {
    "case_basic_info": "案件基础信息",
    "claimant_profile": "申报人基础信息",
    "material_overview": "业务材料总览",
    "medical_materials": "就诊诊疗材料",
    "prescription_materials": "处方购药材料",
    "settlement_materials": "费用结算材料",
    "statistics_report": "统计报表",
    "risk_score": "综合风险评分",
    "evidence_package": "基础证据包",
    "case_judgement": "案件研判",
    "verification_items": "待人工核验事项",
    "rule_verification": "业务规则核验清单",
    "policy_knowledge": "政策知识库",
    "policy_expert": "政策专家分析",
    "case_context_observation": "L3 补充案件事实观察",
}


def build_answer_context_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create an answer-specific fact package from capability results."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="build_answer_context",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            run.run_id,
            "answer_context_building",
            "正在整理回答事实",
            {},
        )
        state["answer_context"] = _build_answer_context(state)
        state["next_action"] = "resolve_answer_policy"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _build_answer_context(state: dict[str, Any]) -> dict[str, Any]:
    semantics = state.get("query_semantics", {})
    if not isinstance(semantics, dict):
        semantics = {}
    user_goal = str(semantics.get("user_goal") or "")
    user_message = state.get("user_message")
    question = str(getattr(user_message, "content", "") or "")
    if not user_goal:
        user_goal = question
    granularity = _normalize_granularity(semantics.get("granularity"))
    answer_context: dict[str, Any] = {
        "user_goal": user_goal,
        "granularity": granularity,
        "facts": [],
        "lists": [],
        "details": [],
        "analysis_inputs": [],
        "notices": [],
        "source_refs": [],
    }

    if state.get("answer_strategy") == "reuse_previous_answer":
        _append_reused_previous_answer_context(answer_context, state)
        answer_context["source_refs"] = list(dict.fromkeys(answer_context["source_refs"]))
        if not any(answer_context[key] for key in ("facts", "lists", "details", "analysis_inputs", "notices")):
            answer_context["notices"].append(
                {
                    "type": "missing_previous_answer",
                    "message": "当前没有找到可复用的上一轮回答。",
                }
            )
        return _clean_empty(answer_context)

    result_items = artifact_results(state) or state.get("capability_results", [])
    for capability, result in result_items:
        payload = getattr(result, "payload", {})
        if not isinstance(payload, dict):
            continue
        if getattr(result, "status", "") != "success":
            _append_notice(answer_context, capability, payload, result)
            continue
        section_key = str(payload.get("section_key") or capability)
        section_payload = payload.get("payload")
        if not isinstance(section_payload, dict):
            section_payload = payload
        source_refs = [
            ref for ref in getattr(result, "source_refs", [])
            if isinstance(ref, str) and ref
        ]
        answer_context["source_refs"].extend(source_refs)
        _append_section_context(
            answer_context,
            section_key=section_key,
            capability=str(capability),
            payload=section_payload,
            user_goal=user_goal,
            question=question,
            granularity=granularity,
            source_refs=source_refs,
        )

    answer_context["source_refs"] = list(dict.fromkeys(answer_context["source_refs"]))
    if not any(answer_context[key] for key in ("facts", "lists", "details", "analysis_inputs", "notices")):
        answer_context["notices"].append(
            {
                "type": "empty_context",
                "message": "当前没有可用于回答该问题的可采信事实。",
            }
        )
    return _clean_empty(answer_context)


def _append_notice(
    answer_context: dict[str, Any],
    capability: str,
    payload: dict[str, Any],
    result: Any,
) -> None:
    inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    summary = inner.get("answerability_summary") if isinstance(inner, dict) else None
    status = str(inner.get("status") or getattr(result, "status", "unavailable")) if isinstance(inner, dict) else str(getattr(result, "status", "unavailable"))
    message = payload.get("message")
    if not message and isinstance(inner, dict):
        message = inner.get("expert_answer")
    if not message:
        message = (
            "当前没有足够可引用的政策证据回答该问题。"
            if capability == "ask_policy_expert" and status == "insufficient"
            else "该能力当前不可用。"
        )
    answer_context["notices"].append(
        {
            "capability": capability,
            "status": status,
            "error_code": getattr(result, "error_code", None),
            "message": message,
            "section_key": payload.get("section_key"),
            "answerability_summary": summary if isinstance(summary, dict) else None,
        }
    )


def _append_section_context(
    answer_context: dict[str, Any],
    *,
    section_key: str,
    capability: str,
    payload: dict[str, Any],
    user_goal: str,
    question: str,
    granularity: str,
    source_refs: list[str],
) -> None:
    if granularity == "single_field":
        fields = _extract_single_fields(section_key, payload, user_goal or question)
        if fields:
            answer_context["facts"].extend(
                _fact(section_key, label, value, source_refs)
                for label, value in fields
            )
            return

    if section_key == "case_basic_info":
        answer_context["facts"].extend(
            _facts_from_mapping(section_key, _case_basic_info_payload(payload), source_refs)
        )
        return
    if section_key == "claimant_profile":
        answer_context["facts"].extend(_facts_from_mapping(section_key, payload, source_refs))
        return
    if section_key == "material_overview":
        _append_material_overview(answer_context, payload, source_refs)
        return
    if section_key == "medical_materials":
        _append_material_details(answer_context, section_key, payload, source_refs)
        return
    if section_key == "prescription_materials":
        _append_prescription_details(answer_context, payload, source_refs)
        return
    if section_key == "settlement_materials":
        _append_settlement_details(answer_context, payload, source_refs)
        return
    if section_key == "statistics_report":
        _append_statistics(answer_context, payload, user_goal or question, source_refs)
        return
    if section_key == "risk_score":
        _append_risk_score(answer_context, payload, source_refs)
        return
    if section_key == "evidence_package":
        _append_evidence_package(answer_context, payload, source_refs)
        return
    if section_key == "policy_expert":
        _append_policy_expert(answer_context, payload, source_refs)
        return
    if section_key in {"case_judgement", "verification_items", "rule_verification", "policy_knowledge"}:
        answer_context["analysis_inputs"].append(
            {
                "section": SECTION_LABELS.get(section_key, section_key),
                "capability": capability,
                "data": _trim_payload(payload, 6000),
                "source_refs": source_refs,
            }
        )
        return
    answer_context["details"].append(
        {
            "section": SECTION_LABELS.get(section_key, section_key),
            "data": _trim_payload(payload, 4000),
            "source_refs": source_refs,
        }
    )


def _extract_single_fields(
    section_key: str,
    payload: dict[str, Any],
    text: str,
) -> list[tuple[str, Any]]:
    keys = _field_keys_for_text(text)
    fields: list[tuple[str, Any]] = []
    if section_key == "case_basic_info":
        fields.extend(_pick_fields(_case_basic_info_payload(payload), keys))
        if fields:
            return fields
    if section_key == "settlement_materials":
        summary = payload.get("settlement_summary")
        if isinstance(summary, dict):
            fields.extend(_pick_fields(summary, keys))
        if fields:
            return fields
    if section_key == "risk_score":
        risk_fields = {
            "overall_strength": payload.get("overall_strength"),
            "risk_level": payload.get("risk_level"),
        }
        fields.extend(_pick_fields(risk_fields, keys))
        if fields:
            return fields
    fields.extend(_pick_fields(payload, keys))
    return fields


def _field_keys_for_text(text: str) -> list[str]:
    matched: list[str] = []
    for key, aliases in FIELD_ALIASES.items():
        if any(alias and alias in text for alias in aliases):
            matched.append(key)
    return list(dict.fromkeys(matched))


def _pick_fields(payload: dict[str, Any], keys: list[str]) -> list[tuple[str, Any]]:
    if not keys:
        return []
    result: list[tuple[str, Any]] = []
    for key in keys:
        if key in payload and payload[key] not in (None, "", [], {}):
            result.append((_label_for_key(key), _display_value(key, payload[key])))
    return result


def _facts_from_mapping(
    section_key: str,
    payload: dict[str, Any],
    source_refs: list[str],
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"meta", "source_ref", "payload_hash"} or value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            value = _trim_payload(value, 1200)
        facts.append(_fact(section_key, _label_for_key(key), value, source_refs))
    return facts[:12]


def _fact(
    section_key: str,
    label: str,
    value: Any,
    source_refs: list[str],
) -> dict[str, Any]:
    return {
        "section": SECTION_LABELS.get(section_key, section_key),
        "label": label,
        "value": value,
        "source_refs": source_refs[:4],
    }


def _append_material_overview(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    groups = [
        group for group in payload.get("groups", [])
        if isinstance(group, dict)
    ]
    answer_context["facts"].append(
        _fact("material_overview", "材料总数", payload.get("material_total") or len(groups), source_refs)
    )
    for group in groups[:8]:
        materials = [
            {
                "材料名称": item.get("name"),
                "发生时间": item.get("occurred_at"),
                "材料来源": item.get("source"),
                "材料形态": item.get("shape"),
                "状态": item.get("status"),
            }
            for item in group.get("materials", [])[:20]
            if isinstance(item, dict)
        ]
        answer_context["lists"].append(
            {
                "section": "业务材料总览",
                "title": group.get("title") or group.get("category_id") or "材料分组",
                "count": group.get("count") or len(materials),
                "items": materials,
                "source_refs": source_refs[:4],
            }
        )


def _append_material_details(
    answer_context: dict[str, Any],
    section_key: str,
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    for material in _materials(payload)[:8]:
        answer_context["details"].append(
            {
                "section": SECTION_LABELS.get(section_key, section_key),
                "title": material.get("name") or material.get("material_id"),
                "basic_info": material.get("basic_info"),
                "content": material.get("content"),
                "visit_details": material.get("visit_details", [])[:40],
                "check_points": material.get("check_points", [])[:12],
                "source_refs": _refs_for_item(material, source_refs),
            }
        )


def _append_prescription_details(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    for material in _materials(payload)[:8]:
        rows = [
            row for row in material.get("prescription_drug_details", [])
            if isinstance(row, dict)
        ]
        answer_context["details"].append(
            {
                "section": "处方购药材料",
                "title": material.get("name") or material.get("material_id"),
                "basic_info": material.get("basic_info"),
                "content": material.get("content"),
                "prescription_drug_details": rows[:100],
                "image_assets": material.get("image_assets", [])[:8],
                "check_points": material.get("check_points", [])[:12],
                "source_refs": _refs_for_item(material, source_refs),
            }
        )


def _append_settlement_details(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    summary = payload.get("settlement_summary")
    if isinstance(summary, dict):
        answer_context["facts"].extend(_facts_from_mapping("settlement_materials", summary, source_refs))
    items = [
        item for item in payload.get("non_drug_fee_items", [])
        if isinstance(item, dict)
    ]
    if items:
        answer_context["lists"].append(
            {
                "section": "费用结算材料",
                "title": "非药品费用业务明细",
                "count": len(items),
                "items": items[:100],
                "source_refs": source_refs[:4],
            }
        )
    _append_material_details(answer_context, "settlement_materials", payload, source_refs)


def _append_statistics(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    user_goal: str,
    source_refs: list[str],
) -> None:
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        answer_context["facts"].append(
            _fact("statistics_report", "统计指标数量", payload.get("metric_count"), source_refs)
        )
        return
    selected = []
    focus_terms = _focus_terms(user_goal)
    for group_name, metrics in groups.items():
        if not isinstance(metrics, list):
            continue
        matched = [
            metric for metric in metrics
            if isinstance(metric, dict) and _metric_matches(metric, focus_terms)
        ]
        if not matched and _group_matches(group_name, focus_terms):
            matched = [metric for metric in metrics if isinstance(metric, dict)]
        if not matched and not focus_terms:
            matched = [metric for metric in metrics if isinstance(metric, dict)][:8]
        if matched:
            selected.append(
                {
                    "group": group_name,
                    "metrics": matched[:40],
                }
            )
    answer_context["analysis_inputs"].append(
        {
            "section": "统计报表",
            "metric_count": payload.get("metric_count"),
            "metric_baseline_status": payload.get("metric_baseline_status"),
            "selected_groups": selected[:6],
            "source_refs": source_refs[:4],
        }
    )


def _append_risk_score(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    for key in ("overall_strength", "risk_level"):
        if payload.get(key) not in (None, ""):
            answer_context["facts"].append(
                _fact("risk_score", _label_for_key(key), payload.get(key), source_refs)
            )
    components = []
    for key in ("model_warning", "rule_check", "peer_deviation", "data_flow"):
        value = payload.get(key)
        if isinstance(value, dict):
            components.append(value)
    answer_context["analysis_inputs"].append(
        {
            "section": "综合风险评分",
            "components": components[:6],
            "source_refs": source_refs[:4],
        }
    )


def _append_evidence_package(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    answer_context["analysis_inputs"].append(
        {
            "section": "基础证据包",
            "base_summary": payload.get("base_summary"),
            "source_basis": payload.get("source_basis"),
            "source_refs": source_refs[:4],
        }
    )
    clues = [
        clue for clue in payload.get("discovered_clues", [])
        if isinstance(clue, dict)
    ]
    if clues:
        answer_context["lists"].append(
            {
                "section": "基础证据包",
                "title": "系统已发现线索",
                "count": len(clues),
                "items": clues[:20],
                "source_refs": source_refs[:4],
            }
        )


def _append_policy_expert(
    answer_context: dict[str, Any],
    payload: dict[str, Any],
    source_refs: list[str],
) -> None:
    evidence = [
        {
            "source_ref": item.get("source_ref"),
            "title": item.get("title"),
            "jurisdiction": item.get("jurisdiction"),
            "policy_domain": item.get("policy_domain"),
            "content_type": item.get("content_type"),
            "excerpt": item.get("excerpt"),
        }
        for item in payload.get("policy_evidence", [])[:5]
        if isinstance(item, dict)
    ]
    answer_context["analysis_inputs"].append(
        {
            "section": "政策专家分析",
            "expert_answer": payload.get("expert_answer"),
            "case_facts_used": payload.get("case_facts_used", []),
            "policy_evidence": evidence,
            "audit_suggestions": payload.get("audit_suggestions", []),
            "answerability_summary": payload.get("answerability_summary"),
            "limits": payload.get("limits"),
            "source_refs": source_refs[:5],
        }
    )


def _append_reused_previous_answer_context(
    answer_context: dict[str, Any],
    state: dict[str, Any],
) -> None:
    recent_messages = state.get("recent_messages", [])
    previous = _reusable_assistant_answer_for_ref(
        recent_messages,
        str(state.get("reuse_answer_ref") or ""),
    ) or _latest_reusable_assistant_answer(recent_messages)
    if previous is None:
        answer_context["notices"].append(
            {
                "type": "missing_previous_answer",
                "message": "当前没有找到可复用的上一轮回答。",
            }
        )
        return

    previous_message, previous_answer = previous
    source_refs = _reuse_source_refs(state, previous_answer)
    sources = _sources_for_refs(previous_answer, source_refs)
    if sources:
        _merge_available_sources(state, sources)
        source_refs = [source.source_ref for source in sources]
    else:
        source_refs = []
    answer_context["source_refs"].extend(source_refs)
    answer_context["analysis_inputs"].append(
        {
            "section": "上一轮回答",
            "mode": "reuse_previous_answer",
            "rewrite_mode": str(state.get("answer_rewrite_mode") or "explain"),
            "rewrite_instruction": str(getattr(state.get("user_message"), "content", "") or "")[:500],
            "previous_answer_ref": _answer_ref_for_message(previous_message),
            "previous_answer": _previous_answer_text(previous_answer, previous_message),
            "previous_source_refs": source_refs[:8],
            "guardrails": [
                "只基于上一轮回答解释、举例或改写。",
                "不得新增政策事实或案件事实。",
                "不得调用新的 L1、L2 或 L3 能力。",
                "引用只能来自上一轮回答已经使用的 source_ref。",
            ],
            "source_refs": source_refs[:8],
        }
    )


def _latest_reusable_assistant_answer(recent_messages: Any) -> tuple[Any, Any] | None:
    for message in reversed(list(recent_messages or [])):
        reusable = _reusable_assistant_answer(message)
        if reusable is not None:
            return reusable
    return None


def _reusable_assistant_answer_for_ref(
    recent_messages: Any,
    answer_ref: str,
) -> tuple[Any, Any] | None:
    if not answer_ref.startswith("answer:"):
        return None
    message_id = answer_ref.removeprefix("answer:")
    for message in reversed(list(recent_messages or [])):
        if str(getattr(message, "message_id", "") or "") == message_id:
            return _reusable_assistant_answer(message)
    return None


def _reusable_assistant_answer(message: Any) -> tuple[Any, Any] | None:
    if str(getattr(message, "role", "") or "") != "assistant":
        return None
    answer_payload = getattr(message, "answer_payload", None)
    if answer_payload is None:
        return (
            (message, None)
            if str(getattr(message, "content", "") or "").strip()
            else None
        )
    if isinstance(answer_payload, dict):
        display_mode = str(answer_payload.get("display_mode") or "plain")
        metadata = (
            answer_payload.get("metadata")
            if isinstance(answer_payload.get("metadata"), dict)
            else {}
        )
        content_blocks = answer_payload.get("content_blocks") or []
    else:
        display_mode = str(getattr(answer_payload, "display_mode", "plain") or "plain")
        metadata = getattr(answer_payload, "metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        content_blocks = getattr(answer_payload, "content_blocks", []) or []
    if display_mode in {"unavailable", "error"}:
        return None
    status = str(metadata.get("status") or metadata.get("intent") or "").lower()
    if status in {
        "clarification_required",
        "unavailable",
        "error",
        "failed",
        "waiting_for_user",
    }:
        return None
    if metadata.get("missing_slots"):
        return None
    if not content_blocks and not str(getattr(message, "content", "") or "").strip():
        return None
    return message, answer_payload


def _answer_ref_for_message(message: Any) -> str:
    message_id = str(getattr(message, "message_id", "") or "")
    return f"answer:{message_id}" if message_id else ""


def _previous_answer_text(answer: Any, message: Any) -> str:
    if isinstance(answer, CaseAgentAnswer):
        return answer.plain_text[:3000]
    if isinstance(answer, dict):
        blocks = [
            str(block.get("text") or "").strip()
            for block in answer.get("content_blocks", [])
            if isinstance(block, dict) and str(block.get("text") or "").strip()
        ]
        if blocks:
            return "\n".join(blocks)[:3000]
    return str(getattr(message, "content", "") or "")[:3000]


def _reuse_source_refs(state: dict[str, Any], answer: Any) -> list[str]:
    refs = [
        str(ref) for ref in list(state.get("reuse_source_refs") or [])
        if isinstance(ref, str) and ref
    ]
    if isinstance(answer, CaseAgentAnswer):
        for block in answer.content_blocks:
            refs.extend(str(ref) for ref in block.source_refs if ref)
        refs.extend(str(source.source_ref) for source in answer.sources if source.source_ref)
    elif isinstance(answer, dict):
        for block in answer.get("content_blocks", []):
            if isinstance(block, dict):
                refs.extend(
                    str(ref) for ref in list(block.get("source_refs") or [])
                    if isinstance(ref, str) and ref
                )
        for source in answer.get("sources", []):
            if isinstance(source, dict) and source.get("source_ref"):
                refs.append(str(source["source_ref"]))
    return list(dict.fromkeys(refs))[:20]


def _sources_for_refs(answer: Any, source_refs: list[str]) -> list[CaseAgentSource]:
    allowed = set(source_refs)
    sources: list[CaseAgentSource] = []
    if isinstance(answer, CaseAgentAnswer):
        for source in answer.sources:
            if not allowed or source.source_ref in allowed:
                sources.append(source)
    elif isinstance(answer, dict):
        for source in answer.get("sources", []):
            if not isinstance(source, dict):
                continue
            try:
                parsed = CaseAgentSource.model_validate(source)
            except Exception:
                continue
            if not allowed or parsed.source_ref in allowed:
                sources.append(parsed)
    seen: set[str] = set()
    deduped: list[CaseAgentSource] = []
    for source in sources:
        if source.source_ref in seen:
            continue
        seen.add(source.source_ref)
        deduped.append(source)
    return deduped[:20]


def _merge_available_sources(state: dict[str, Any], sources: list[CaseAgentSource]) -> None:
    existing = [
        source for source in state.get("available_sources", [])
        if isinstance(source, CaseAgentSource)
    ]
    by_ref = {source.source_ref: source for source in existing}
    for source in sources:
        by_ref.setdefault(source.source_ref, source)
    state["available_sources"] = list(by_ref.values())


def _materials(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in payload.get("materials", [])
        if isinstance(item, dict)
    ]


def _refs_for_item(item: dict[str, Any], fallback: list[str]) -> list[str]:
    refs: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("source_ref")
            if isinstance(ref, str) and ref:
                refs.append(ref)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(item)
    return list(dict.fromkeys(refs or fallback))[:4]


def _label_for_key(key: str) -> str:
    labels = {
        "case_number": "案件编号",
        "case_type": "案件类型",
        "review_status": "审核状态",
        "claimant_ref": "脱敏申报人编号",
        "access_method": "接入方式",
        "insured_region": "参保地",
        "treatment_region": "就医地",
        "visit_type": "就医类型",
        "claim_mode": "报销方式",
        "direct_settlement": "直接结算",
        "filing_status": "备案状态",
        "emergency_material_status": "急诊材料状态",
        "diagnosis": "诊断",
        "claimant_code": "申报人编码",
        "gender": "性别",
        "age_group": "年龄段",
        "insurance_type": "参保类型",
        "chronic_condition_tags": "慢病标签",
        "allergy_history": "过敏史",
        "patient_group_tags": "申报人特征标签",
        "overall_strength": "综合风险提示强度",
        "risk_level": "风险等级",
    }
    return labels.get(key, key)


def _display_value(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return "是" if value else "否"
    text = str(value)
    if key == "claim_mode":
        return {
            "manual_reimbursement": "手工报销",
            "manual_upload_review": "手工上传材料复核",
            "post_settlement_review": "医保结算后复核",
            "direct_settlement": "直接结算",
        }.get(text, text)
    if key == "review_status":
        return {
            "pending": "待人工初审",
            "reviewed": "已完成人工初审",
            "approved": "审核通过",
            "rejected": "审核未通过",
            "closed": "已形成案件处理结果",
        }.get(text, text)
    if key == "filing_status":
        return {
            "unknown": "状态不明",
            "missing": "缺失",
            "not_applicable": "不适用",
            "filed": "已备案",
            "online_filing_effective": "线上备案即时生效",
            "filed_or_emergency_exception_supported": "已备案或急诊例外材料支持",
        }.get(text, text)
    if key == "emergency_material_status":
        return {
            "unknown": "状态不明",
            "missing": "缺失",
            "missing_or_unclear": "缺失或不清晰",
            "present": "已上传",
            "present_but_needs_verification": "已上传，待核验",
            "ordinary_outpatient_only": "仅普通门诊材料",
            "clear": "材料清晰",
            "not_applicable": "不适用",
        }.get(text, text)
    return value


def _case_basic_info_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base = dict(payload)
    case_context = base.get("case_context")
    if not isinstance(case_context, dict):
        return base
    for key in (
        "insured_region",
        "treatment_region",
        "visit_type",
        "claim_mode",
        "direct_settlement",
        "filing_status",
        "emergency_material_status",
        "diagnosis",
    ):
        if base.get(key) in (None, "", [], {}):
            value = case_context.get(key)
            if value not in (None, "", [], {}):
                base[key] = value
    return base


def _normalize_granularity(value: Any) -> str:
    granularity = str(value or "").strip()
    if granularity in {
        "single_field",
        "field_group",
        "list",
        "detail",
        "overview",
        "analysis",
    }:
        return granularity
    return "overview"


def _focus_terms(text: str) -> list[str]:
    terms: list[str] = []
    if any(token in text for token in ("费用", "金额", "结算", "申报", "审批")):
        terms.extend(["费用", "金额", "SUM", "ALL_SUM", "审批"])
    if any(token in text for token in ("药", "处方", "购药")):
        terms.extend(["药", "drug"])
    if any(token in text for token in ("就诊", "诊疗", "医院", "机构")):
        terms.extend(["就诊", "医院", "机构", "visit"])
    if any(token in text for token in ("支付", "统筹", "个人账户")):
        terms.extend(["支付", "统筹", "个人账户"])
    if any(token in text for token in ("占比", "比例", "结构")):
        terms.extend(["占比", "比例", "ratio"])
    if any(token in text for token in ("偏高", "偏离", "高于", "明显")):
        terms.extend(["偏高", "偏离", "高于", "明显"])
    return list(dict.fromkeys(terms))


def _group_matches(group_name: str, terms: list[str]) -> bool:
    markers = {
        "fee_statistics": ["费用", "金额", "结算", "申报", "审批"],
        "drug_statistics": ["药", "处方", "购药"],
        "visit_statistics": ["就诊", "医院", "机构"],
        "visit_day_statistics": ["就诊"],
        "payment_statistics": ["支付", "统筹", "个人账户"],
        "fee_structure_statistics": ["占比", "比例", "结构"],
    }
    return any(term in markers.get(group_name, []) for term in terms)


def _metric_matches(metric: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return False
    text = json.dumps(
        {
            "feature_name": metric.get("feature_name"),
            "hint": metric.get("hint"),
            "判断": metric.get("判断"),
        },
        ensure_ascii=False,
        default=str,
    )
    return any(term in text for term in terms)


def _trim_payload(value: Any, limit: int) -> Any:
    raw = json.dumps(value, ensure_ascii=False, default=str)
    if len(raw) <= limit:
        return value
    return {"summary": raw[:limit], "truncated": True}


def _clean_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_clean_empty(item) for item in value if item not in (None, "", [], {})]
    return value
