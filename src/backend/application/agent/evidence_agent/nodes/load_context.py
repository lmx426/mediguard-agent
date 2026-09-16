"""Context loading node for Evidence Agent."""

from __future__ import annotations

import time
from typing import Any

from src.backend.domain.agent.entities import EvidenceAgentRequest
from ..review_brief import REVIEW_BRIEF_VERSION, build_review_brief
from ..prompts.loader import build_system_prompt, build_user_prompt
from ..tools.registry import ToolExecutionError, tool_signature


def load_context_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Load the default evidence ledger and build the initial model context."""

    run_id = state["run_id"]
    case_id = state["case_id"]
    service._repository.update_run(run_id, status="running", current_node="load_context")
    case = service._cases.get_case(case_id)
    if case is None:
        return _state_error(state, "case_not_found", "Case not found")

    service._repository.append_event(
        run_id,
        "ledger_prefetch_started",
        "Prefetching the default evidence ledger",
        {"tool": "get_evidence_ledger"},
    )
    result, items, fallback_used = _prefetch_default_ledger(service, state)
    if result is None:
        return _state_error(state, "base_prefetch_failed", "Default evidence prefetch failed")

    ledger = {item.ledger_ref: item.model_dump(mode="json") for item in items}
    request = EvidenceAgentRequest.model_validate(state["request"])
    eval_variant = request.eval_variant or "A0"
    evidence_ledger = result.get("data", result)
    prompt_evidence_ledger = _prompt_ledger_view(evidence_ledger, eval_variant)
    tool_catalog = service._tools.catalog_for_prompt(stage="post_prefetch")
    state["ledger"] = ledger
    state["system_risk_prompt"] = evidence_ledger.get("system_risk_prompt")
    state["tool_catalog"] = tool_catalog
    state["messages"] = [
        {"role": "system", "content": build_system_prompt(eval_variant=eval_variant)},
        {
            "role": "user",
            "content": build_user_prompt(
                case_id=case_id,
                request=request,
                evidence_ledger=prompt_evidence_ledger,
                ledger_refs=list(ledger),
                tool_catalog=tool_catalog,
                eval_variant=eval_variant,
            ),
        },
    ]
    service._repository.append_event(
        run_id,
        "ledger_prefetched",
        "Default evidence ledger loaded",
        {
            "source_count": len(ledger),
            "fallback_used": fallback_used,
            "eval_variant": eval_variant,
            **_prompt_ledger_payload(prompt_evidence_ledger, eval_variant),
        },
    )
    service._repository.append_event(run_id, "base_loaded", "Base evidence package loaded")
    service._checkpoint(state, "load_context")
    return state


def _prompt_ledger_view(evidence_ledger: dict[str, Any], eval_variant: str) -> dict[str, Any]:
    if eval_variant == "B3":
        return _lightweight_ledger_view(evidence_ledger)
    if eval_variant == "C1":
        return build_review_brief(evidence_ledger)
    return evidence_ledger


def _prompt_ledger_payload(prompt_evidence_ledger: dict[str, Any], eval_variant: str) -> dict[str, Any]:
    if eval_variant == "B3":
        return {"ledger_view": "lightweight"}
    if eval_variant == "C1":
        prefetch_summary = prompt_evidence_ledger.get("prefetch_summary") or {}
        return {
            "ledger_view": "review_brief",
            "review_brief_builder": REVIEW_BRIEF_VERSION,
            "brief_candidate_count": prefetch_summary.get("brief_candidate_count"),
            "brief_fact_count": prefetch_summary.get("brief_fact_count"),
            "brief_citation_manifest_count": prefetch_summary.get(
                "brief_citation_manifest_count"
            ),
        }
    return {"ledger_view": "full"}


def _lightweight_ledger_view(evidence_ledger: dict[str, Any]) -> dict[str, Any]:
    """Build a compact prompt view while keeping the full ledger in state."""

    return {
        "case_id": evidence_ledger.get("case_id"),
        "ledger_version": evidence_ledger.get("ledger_version"),
        "scope": "lightweight_review_brief",
        "agent_task": evidence_ledger.get("agent_task"),
        "case_context": evidence_ledger.get("case_context"),
        "system_risk_prompt": evidence_ledger.get("system_risk_prompt"),
        "clue_candidates": [
            _compact_item(
                item,
                (
                    "clue_id",
                    "title",
                    "category",
                    "statement",
                    "status",
                    "source_refs",
                ),
            )
            for item in evidence_ledger.get("clue_candidates", [])
            if isinstance(item, dict)
        ],
        "key_fact_summaries": [
            _compact_item(
                item,
                ("fact_id", "category", "statement", "source_refs"),
            )
            for item in evidence_ledger.get("normalized_facts", [])
            if isinstance(item, dict)
        ],
        "known_gaps": [
            _compact_item(item, ("gap_id", "statement", "source_refs"))
            for item in evidence_ledger.get("known_gaps", [])
            if isinstance(item, dict)
        ],
        "drilldown_index": evidence_ledger.get("drilldown_index", {}),
        "evidence_reference_index": evidence_ledger.get("evidence_reference_index", []),
        "generation_guidance": {
            **dict(evidence_ledger.get("generation_guidance") or {}),
            "default_context": "lightweight_view",
            "tool_call_policy": "默认视图不足以形成证据闭环时，可按穿透索引调用细粒度工具补查。",
        },
        "prefetch_summary": evidence_ledger.get("prefetch_summary", {}),
    }


def _compact_item(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: item[key] for key in keys if key in item}


def _prefetch_default_ledger(
    service: Any,
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[Any], bool]:
    """Run framework-level ledger prefetch with a minimal fallback."""

    try:
        result, items = _run_prefetch_tool(service, state, "get_evidence_ledger")
        return result, items, False
    except Exception as exc:
        service._repository.append_event(
            state["run_id"],
            "prefetch_failed",
            "Default evidence ledger failed; falling back to base evidence package",
            {
                "tool": "get_evidence_ledger",
                "error_code": getattr(exc, "code", "prefetch_failed"),
            },
        )
        try:
            result, items = _run_prefetch_tool(service, state, "get_base_evidence_package")
        except Exception:
            return None, [], True
        case = service._cases.get_case(state["case_id"])
        if case is None:
            return None, [], True
        minimal = {
            "data": {
                "case_id": state["case_id"],
                "ledger_version": "evidence-ledger-v2-minimal",
                "scope": "fallback_review_brief",
                "agent_task": {
                    "objective": "复核当前证据是否支撑系统综合风险提示，并整理仍需人工介入的核验处置事项",
                    "fixed_boundaries": [
                        "不得重新计算或修改风险分",
                        "不得认定欺诈、拒付、处罚或审核通过",
                    ],
                },
                "system_risk_prompt": _fallback_system_risk_prompt(case),
                "normalized_facts": [
                    {
                        "fact_id": "fact:fallback-base-evidence",
                        "category": "base_evidence",
                        "statement": result.get("data", result).get("risk_summary", "已加载最小基础证据包。"),
                        "source_refs": [item.ledger_ref for item in items[:1]],
                    }
                ],
                "clue_candidates": [],
                "known_gaps": [],
                "drilldown_index": {"rules": [], "materials": [], "fields": []},
                "generation_guidance": {
                    "default_action": "generate_first",
                    "decision_owner": "model",
                    "reason": "Evidence Ledger 预取失败，当前仅提供最小基础证据包。",
                    "tool_call_policy": "如具体事实缺口影响证据复核判断，可调用当前可用的细粒度工具补查。",
                    "when_to_call_tools": ["需要确认影响复核关系的具体规则、字段或材料事实"],
                    "when_not_to_call_tools": ["不要重复读取基础证据包", "不要为了凑齐引用调用工具"],
                },
                "prefetch_summary": {
                    "ledger_item_count": len(items),
                    "fallback_used": True,
                },
            },
            "disclaimer": result.get("disclaimer"),
        }
        return minimal, items, True


def _fallback_system_risk_prompt(case: Any) -> dict[str, Any]:
    """在 Ledger 预取失败时保留后端风险提示的最小只读快照。"""

    breakdown = case.risk_score_breakdown
    score = breakdown.total_score if breakdown else round(case.risk_score * 100)
    level = breakdown.level if breakdown else case.risk_level
    level_label = breakdown.level_label if breakdown else {
        "high": "高风险",
        "medium": "中风险",
        "low": "低风险",
        "insufficient": "证据不足",
    }.get(level, str(level))
    return {
        "generated_by": "backend_risk_engine",
        "risk_level": level,
        "risk_level_label": level_label,
        "risk_score": score,
        "score_text": breakdown.display_score if breakdown else f"{score} / 100",
        "source_summary": [item.label for item in breakdown.components] if breakdown else [],
        "source_refs": [case.model_evidence_ref],
    }


def _run_prefetch_tool(
    service: Any,
    state: dict[str, Any],
    tool_name: str,
) -> tuple[dict[str, Any], list[Any]]:
    """Execute and persist one framework-level read-only tool call."""

    run_id = state["run_id"]
    args = {"case_id": state["case_id"]}
    tool_call_id = f"system_prefetch:{tool_name}"
    signature = tool_signature(tool_name, args)
    if signature not in state["seen_signatures"]:
        state["seen_signatures"].append(signature)

    state["tool_call_count"] += 1
    service._repository.update_run(
        run_id,
        current_node="load_context",
        tool_call_count=state["tool_call_count"],
    )

    start = time.monotonic()
    try:
        result, items, validated_args = service._tools.execute(
            current_case_id=state["case_id"],
            tool_name=tool_name,
            raw_arguments=args,
        )
        record_result = {
            "ok": True,
            "data": result,
            "evidence_items": [item.model_dump(mode="json") for item in items],
        }
        service._repository.record_tool_call(
            run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=validated_args,
            result=record_result,
            status="success",
            error_code=None,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
        return result, items
    except ToolExecutionError as exc:
        service._repository.record_tool_call(
            run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=args,
            result=None,
            status="failed",
            error_code=exc.code,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
        raise
    except Exception:
        service._repository.record_tool_call(
            run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=args,
            result=None,
            status="failed",
            error_code="prefetch_failed",
            latency_ms=int((time.monotonic() - start) * 1000),
        )
        raise


def _state_error(state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    """Set an error route on the Agent state."""

    state["error_code"] = code
    state["error_message"] = message
    state["next_action"] = "fail"
    return state
