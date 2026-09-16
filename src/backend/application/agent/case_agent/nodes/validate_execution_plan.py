"""Validate the model-generated Caser execution plan."""

from __future__ import annotations

import json
import re
from typing import Any

from src.backend.application.agent.case_agent.tools.registry import (
    is_known_capability,
    is_l2_or_l3_capability,
    l1_section_for_capability,
    normalize_capability_name,
)
from src.backend.domain.case_memory.entities import MemoryType

from ._errors import state_error


FIELD_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,80}$")
FORBIDDEN_ARGUMENT_MARKERS = (
    "select ",
    " insert ",
    " update ",
    " delete ",
    " drop ",
    " from ",
    " join ",
    "../",
    "..\\",
    ":\\",
    "/etc/",
)


def validate_execution_plan_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Validate plan whitelist, case binding and section readiness."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="validate_execution_plan",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            run.run_id,
            "execution_plan_validating",
            "Checking Caser capability plan",
            {},
        )

        if state.get("missing_slots"):
            state["query_semantics"] = _normalized_query_semantics(
                state.get("query_semantics", {}),
                intent="clarification_required",
                missing_slots=state.get("missing_slots", []),
            )
            state["intent"] = "clarification_required"
            state["next_action"] = "ask_clarification"
            return state

        execution_plan = state.get("execution_plan", [])
        if not isinstance(execution_plan, list):
            raise ValueError("execution_plan must be a list")

        validated: list[dict[str, Any]] = []
        section_status: list[dict[str, Any]] = []
        for index, item in enumerate(execution_plan[:8], start=1):
            if not isinstance(item, dict):
                raise ValueError("execution plan item must be an object")
            raw_capability = str(item.get("capability") or "")
            capability = normalize_capability_name(raw_capability)
            if not is_known_capability(capability):
                raise ValueError(f"unknown capability: {raw_capability}")
            step = _validate_step(item.get("step"), index)
            args = dict(item.get("arguments") or {})
            if capability == "ask_policy_expert":
                _merge_policy_memory_hint(
                    args,
                    service.memory_hint_for_node(
                        state,
                        consumer="policy_filter_resolver",
                        memory_type=MemoryType.POLICY_SEARCH,
                        task_context={"question": str(args.get("question") or "")},
                    ),
                )
            args["case_id"] = _validate_case_id(args.get("case_id"), run.case_id)
            args["fields"] = _validate_fields(args.get("fields", []))
            args["filters"] = _validate_filters(args.get("filters", {}))
            args["limit"] = _validate_limit(args.get("limit", 30))
            _validate_no_forbidden_arguments(args)
            next_item = {
                "step": step,
                "layer": _validate_layer(item.get("layer"), capability),
                "capability": capability,
                "name": capability,
                "arguments": args,
                "depends_on": _validate_depends_on(item.get("depends_on", []), current_step=step),
                "reason": str(item.get("reason") or "")[:300],
            }
            validated.append(next_item)
            section_key = l1_section_for_capability(capability)
            if section_key is not None:
                status = service._tools.section_status(
                    case_id=run.case_id,
                    capability_name=capability,
                )
                section_status.append(status)

        validated.sort(key=lambda item: item["step"])
        state["execution_plan"] = validated
        state["query_semantics"] = _normalized_query_semantics(
            state.get("query_semantics", {}),
            intent=_infer_intent(
                state.get("query_semantics", {}).get("intent"),
                validated,
            ),
            missing_slots=[],
        )
        state["intent"] = state["query_semantics"]["intent"]
        state["context_plan"] = {
            "capabilities": validated,
        }
        state["section_status"] = section_status
        state["next_action"] = "execution_dag_builder" if validated else "build_answer_context"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _merge_policy_memory_hint(arguments: dict[str, Any], pack: dict[str, Any]) -> None:
    """Use approved historical filters as defaults; current request values win."""

    memory_filters: dict[str, Any] = {}
    memory_lists: dict[str, list[Any]] = {}
    memory_guidance: dict[str, Any] = {}
    for item in pack.get("tool_param_hints", []) if isinstance(pack, dict) else []:
        if not isinstance(item, dict):
            continue
        structured = item.get("structured_content")
        if isinstance(structured, dict):
            if isinstance(structured.get("filters"), dict):
                memory_filters.update(structured["filters"])
                memory_guidance["filters"] = dict(memory_filters)
            for key in (
                "information_needs",
                "coverage_requirements",
                "evidence_focus",
                "citation_expectations",
                "avoid_claims",
            ):
                value = structured.get(key)
                if isinstance(value, list):
                    memory_lists.setdefault(key, []).extend(value)
                    memory_guidance[key] = list(dict.fromkeys(memory_lists[key]))[:32]
        if isinstance(item.get("recommended_action"), dict):
            action = item["recommended_action"]
            if isinstance(action.get("filters"), dict):
                memory_filters.update(action["filters"])
                memory_guidance["filters"] = dict(memory_filters)
            for key in (
                "information_needs",
                "coverage_requirements",
                "evidence_focus",
                "citation_expectations",
                "avoid_claims",
            ):
                if isinstance(action.get(key), list):
                    memory_lists.setdefault(key, []).extend(action[key])
                    memory_guidance[key] = list(dict.fromkeys(memory_lists[key]))[:32]
    current_filters = arguments.get("filters")
    arguments["filters"] = {
        **memory_filters,
        **(current_filters if isinstance(current_filters, dict) else {}),
    }
    for key, values in memory_lists.items():
        current = arguments.get(key)
        current_values = current if isinstance(current, list) else []
        merged = list(dict.fromkeys([*values, *current_values]))
        if merged:
            arguments[key] = merged[:32]
            memory_guidance[key] = merged[:32]
    if memory_guidance:
        arguments["memory_guidance"] = memory_guidance


def _validate_case_id(value: Any, current_case_id: str) -> str:
    if value in (None, "", "__CURRENT_CASE__"):
        return current_case_id
    if str(value) != current_case_id:
        raise ValueError("planned case_id does not match current case")
    return current_case_id


def _validate_step(value: Any, fallback: int) -> int:
    try:
        step = int(value)
    except (TypeError, ValueError):
        step = fallback
    return max(1, min(step, 8))


def _validate_layer(value: Any, capability: str) -> str:
    layer = str(value or "").upper()
    if layer not in {"L1", "L2", "L3"}:
        layer = "L3" if is_l2_or_l3_capability(capability) else "L1"
    if l1_section_for_capability(capability) is not None:
        return "L1"
    if is_l2_or_l3_capability(capability):
        return "L3" if capability.startswith("ask_") else "L2"
    return layer


def _validate_fields(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("fields must be a list")
    fields: list[str] = []
    for item in value[:32]:
        if not isinstance(item, str):
            continue
        field = item.strip()
        if not FIELD_PATTERN.match(field):
            continue
        fields.append(field)
    return fields


def _validate_filters(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("filters must be an object")
    cleaned: dict[str, Any] = {}
    for key, item in list(value.items())[:12]:
        if not isinstance(key, str) or not FIELD_PATTERN.match(key):
            raise ValueError("filter names must be simple top-level identifiers")
        if isinstance(item, (dict, list)):
            raise ValueError("nested filters are not supported in v1")
        cleaned[key] = item
    return cleaned


def _validate_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    return max(1, min(limit, 100))


def _validate_depends_on(value: Any, *, current_step: int) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("depends_on must be a list")
    depends_on: list[str] = []
    for item in value[:8]:
        if item in (None, ""):
            continue
        text = str(item).strip()
        if not text:
            continue
        if text.isdigit():
            step_number = int(text)
            if step_number >= current_step:
                raise ValueError("depends_on can only reference earlier steps")
            text = f"step_{step_number}"
        elif re.match(r"^step_[1-8]$", text):
            step_number = int(text.split("_", 1)[1])
            if step_number >= current_step:
                raise ValueError("depends_on can only reference earlier steps")
        else:
            raise ValueError("depends_on items must be step numbers or step_N references")
        depends_on.append(text)
    return list(dict.fromkeys(depends_on))


def _validate_no_forbidden_arguments(arguments: dict[str, Any]) -> None:
    raw = json.dumps(arguments, ensure_ascii=False, default=str).lower()
    if any(marker in raw for marker in FORBIDDEN_ARGUMENT_MARKERS):
        raise ValueError("execution plan contains forbidden argument markers")


def _normalized_query_semantics(
    value: Any,
    *,
    intent: str,
    missing_slots: list[Any],
) -> dict[str, Any]:
    semantics = dict(value or {}) if isinstance(value, dict) else {}
    return {
        "intent": intent,
        "user_goal": str(semantics.get("user_goal") or "")[:500],
        "granularity": _normalize_granularity(semantics.get("granularity")),
        "missing_slots": [str(item)[:80] for item in missing_slots[:8]],
    }


def _infer_intent(raw_intent: Any, execution_plan: list[dict[str, Any]]) -> str:
    intent = str(raw_intent or "").strip()
    if intent in {"general_help", "case_task", "expert_task", "clarification_required"}:
        return intent
    if not execution_plan:
        return "general_help"
    if any(is_l2_or_l3_capability(item["capability"]) for item in execution_plan):
        return "expert_task"
    return "case_task"


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
