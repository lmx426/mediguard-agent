"""Resolve deterministic answer policy after capability execution."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.artifacts import artifact_results
from src.backend.application.agent.case_agent.tools.registry import (
    is_l2_or_l3_capability,
    l1_section_for_capability,
    normalize_capability_name,
)

from ._errors import state_error


VALID_DISPLAY_MODES = {"plain", "grounded", "unavailable", "error"}


def resolve_answer_policy_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Decide answer mode, citation policy and capability status deterministically."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="resolve_answer_policy",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        policy = _build_answer_policy(state)
        state["answer_policy"] = policy
        state["context_plan"] = {
            **(state.get("context_plan") or {}),
            "display_mode": policy["display_mode"],
            "requires_citation": policy["requires_citation"],
            "source_policy": policy["source_policy"],
            "capability_policy": policy["capability_policy"],
            "capabilities": state.get("execution_plan", []),
        }
        service._repository.append_event(
            run.run_id,
            "answer_policy_resolved",
            "Answer policy resolved",
            {
                "display_mode": policy["display_mode"],
                "capability_policy": policy["capability_policy"],
                "requires_citation": policy["requires_citation"],
            },
        )
        if getattr(service, "_fast_mode_enabled", False):
            service._apply_fast_context_digest(state)
        state["next_action"] = "resolve_answer_style"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _build_answer_policy(state: dict[str, Any]) -> dict[str, Any]:
    shortcut = state.get("shortcut") or {}
    shortcut_display_mode = str(shortcut.get("display_mode") or "")
    if shortcut_display_mode in VALID_DISPLAY_MODES:
        return _policy(
            display_mode=shortcut_display_mode,
            capability_policy="shortcut",
            source_policy="none",
            requires_citation=False,
            reason="Shortcut answer uses deterministic product capability status.",
            capabilities_used=[],
            unavailable_capabilities=[],
            not_ready_sections=[],
        )

    execution_plan = [
        item for item in state.get("execution_plan", [])
        if isinstance(item, dict)
    ]
    if state.get("answer_strategy") == "reuse_previous_answer":
        source_refs = [
            ref for ref in (state.get("answer_context") or {}).get("source_refs", [])
            if isinstance(ref, str) and ref
        ]
        if source_refs:
            return _policy(
                display_mode="grounded",
                capability_policy="reuse_previous_answer",
                source_policy="required",
                requires_citation=True,
                reason="This answer rewrites the previous answer and reuses its closed source refs.",
                capabilities_used=["reuse_previous_answer"],
                unavailable_capabilities=[],
                not_ready_sections=[],
            )
        return _policy(
            display_mode="plain",
            capability_policy="reuse_previous_answer",
            source_policy="none",
            requires_citation=False,
            reason="This answer rewrites the previous plain answer without new capability execution.",
            capabilities_used=["reuse_previous_answer"],
            unavailable_capabilities=[],
            not_ready_sections=[],
        )
    if not execution_plan:
        return _policy(
            display_mode="plain",
            capability_policy="no_tool",
            source_policy="none",
            requires_citation=False,
            reason="No capability execution is required for this answer.",
            capabilities_used=[],
            unavailable_capabilities=[],
            not_ready_sections=[],
        )

    raw_results = artifact_results(state) or state.get("capability_results", [])
    capability_results = [
        (name, result)
        for name, result in raw_results
        if isinstance(name, str) and name != "case_context_observation"
    ]
    planned_capabilities = [
        normalize_capability_name(str(item.get("capability") or item.get("name") or ""))
        for item in execution_plan
    ]
    l1_capabilities = [
        capability for capability in planned_capabilities
        if l1_section_for_capability(capability) is not None
    ]
    l2_l3_capabilities = [
        capability for capability in planned_capabilities
        if is_l2_or_l3_capability(capability)
    ]
    successful_l1 = [
        name for name, result in capability_results
        if l1_section_for_capability(name) is not None
        and getattr(result, "status", "") == "success"
        and getattr(result, "source_refs", [])
    ]
    successful_grounded = [
        name for name, result in capability_results
        if getattr(result, "status", "") == "success"
        and getattr(result, "source_refs", [])
    ]
    unavailable_capabilities = [
        name for name, result in capability_results
        if getattr(result, "status", "") == "unavailable"
    ]
    expert_evidence_insufficient = _expert_evidence_insufficient(capability_results)
    expert_evidence_insufficient_with_sources = _expert_evidence_insufficient_with_sources(
        capability_results
    )
    failed_capabilities = [
        name for name, result in capability_results
        if getattr(result, "status", "") == "failed"
    ]
    not_ready_sections = _not_ready_sections(state, capability_results)

    if expert_evidence_insufficient_with_sources:
        return _policy(
            display_mode="grounded",
            capability_policy="expert_evidence_insufficient",
            source_policy="required",
            requires_citation=True,
            reason=(
                "Policy Expert returned partial citable evidence but judged the "
                "question insufficiently answerable."
            ),
            capabilities_used=expert_evidence_insufficient_with_sources,
            unavailable_capabilities=[],
            not_ready_sections=not_ready_sections,
        )
    if successful_l1:
        return _policy(
            display_mode="grounded",
            capability_policy=(
                "partial_capability"
                if unavailable_capabilities or failed_capabilities or not_ready_sections
                else "grounded_capability"
            ),
            source_policy="required",
            requires_citation=True,
            reason="At least one L1 capability returned grounded sources.",
            capabilities_used=successful_l1,
            unavailable_capabilities=unavailable_capabilities,
            not_ready_sections=not_ready_sections,
        )
    if successful_grounded:
        return _policy(
            display_mode="grounded",
            capability_policy=(
                "partial_capability"
                if unavailable_capabilities or failed_capabilities or not_ready_sections
                else "grounded_capability"
            ),
            source_policy="required",
            requires_citation=True,
            reason="At least one capability returned grounded sources.",
            capabilities_used=successful_grounded,
            unavailable_capabilities=unavailable_capabilities,
            not_ready_sections=not_ready_sections,
        )
    if expert_evidence_insufficient:
        return _policy(
            display_mode="unavailable",
            capability_policy="expert_evidence_insufficient",
            source_policy="none",
            requires_citation=False,
            reason="Policy Expert ran but did not return enough citable evidence.",
            capabilities_used=[],
            unavailable_capabilities=[],
            not_ready_sections=not_ready_sections,
        )
    if l2_l3_capabilities and not l1_capabilities:
        return _policy(
            display_mode="unavailable",
            capability_policy="capability_unavailable",
            source_policy="none",
            requires_citation=False,
            reason="Only reserved L2/L3 capabilities were planned.",
            capabilities_used=[],
            unavailable_capabilities=unavailable_capabilities or l2_l3_capabilities,
            not_ready_sections=not_ready_sections,
        )
    if unavailable_capabilities or not_ready_sections:
        return _policy(
            display_mode="unavailable",
            capability_policy="data_not_ready",
            source_policy="none",
            requires_citation=False,
            reason="Required L1 data is not ready or unavailable.",
            capabilities_used=[],
            unavailable_capabilities=unavailable_capabilities,
            not_ready_sections=not_ready_sections,
        )
    if failed_capabilities:
        return _policy(
            display_mode="error",
            capability_policy="capability_failed",
            source_policy="none",
            requires_citation=False,
            reason="Capability execution failed without usable sources.",
            capabilities_used=[],
            unavailable_capabilities=failed_capabilities,
            not_ready_sections=not_ready_sections,
        )
    return _policy(
        display_mode="plain",
        capability_policy="no_usable_source",
        source_policy="none",
        requires_citation=False,
        reason="No usable source was returned.",
        capabilities_used=[],
        unavailable_capabilities=unavailable_capabilities,
        not_ready_sections=not_ready_sections,
    )


def _not_ready_sections(
    state: dict[str, Any],
    capability_results: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    not_ready: list[dict[str, Any]] = []
    for item in state.get("section_status", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "")
        if status and status != "ready":
            not_ready.append(item)
    for name, result in capability_results:
        payload = getattr(result, "payload", {})
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "not_ready":
            not_ready.append(
                {
                    "capability": name,
                    "section_key": payload.get("section_key"),
                    "status": payload.get("section_status") or "not_ready",
                    "completeness": payload.get("completeness"),
                    "error_code": getattr(result, "error_code", None),
                }
            )
    deduped: dict[str, dict[str, Any]] = {}
    for item in not_ready:
        key = str(item.get("section_key") or item.get("capability") or item)
        deduped[key] = item
    return list(deduped.values())


def _expert_evidence_insufficient(capability_results: list[tuple[str, Any]]) -> bool:
    for name, result in capability_results:
        if normalize_capability_name(name) != "ask_policy_expert":
            continue
        payload = getattr(result, "payload", {})
        if not isinstance(payload, dict):
            continue
        inner = payload.get("payload")
        if not isinstance(inner, dict):
            inner = payload
        status = str(inner.get("status") or payload.get("status") or "")
        error_code = str(getattr(result, "error_code", "") or "")
        if status == "insufficient" or error_code == "expert_evidence_insufficient":
            return True
        summary = inner.get("answerability_summary")
        if isinstance(summary, dict) and summary.get("answerability") == "insufficient":
            return True
    return False


def _expert_evidence_insufficient_with_sources(
    capability_results: list[tuple[str, Any]],
) -> list[str]:
    capabilities: list[str] = []
    for name, result in capability_results:
        normalized = normalize_capability_name(name)
        if normalized != "ask_policy_expert":
            continue
        if not getattr(result, "source_refs", []):
            continue
        payload = getattr(result, "payload", {})
        if not isinstance(payload, dict):
            continue
        inner = payload.get("payload")
        if not isinstance(inner, dict):
            inner = payload
        status = str(inner.get("status") or payload.get("status") or "")
        error_code = str(getattr(result, "error_code", "") or "")
        summary = inner.get("answerability_summary")
        summary_insufficient = (
            isinstance(summary, dict)
            and summary.get("answerability") == "insufficient"
        )
        if (
            status == "insufficient"
            or error_code == "expert_evidence_insufficient"
            or summary_insufficient
        ):
            capabilities.append(normalized)
    return list(dict.fromkeys(capabilities))


def _policy(
    *,
    display_mode: str,
    capability_policy: str,
    source_policy: str,
    requires_citation: bool,
    reason: str,
    capabilities_used: list[str],
    unavailable_capabilities: list[str],
    not_ready_sections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "display_mode": display_mode,
        "requires_citation": requires_citation,
        "source_policy": source_policy,
        "capability_policy": capability_policy,
        "no_general_knowledge_fallback": display_mode != "plain",
        "reason": reason,
        "capabilities_used": list(dict.fromkeys(capabilities_used)),
        "unavailable_capabilities": list(dict.fromkeys(unavailable_capabilities)),
        "not_ready_sections": not_ready_sections,
    }
