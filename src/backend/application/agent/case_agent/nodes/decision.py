"""Caser decision and planning layer nodes."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
    _fallback_general_help_plan,
    _parse_plan,
    _recent_context_for_planner,
)
from src.backend.application.agent.case_agent.prompts.planning import (
    build_business_semantic_planning_prompt,
)
from src.backend.application.agent.case_agent.schemas.perception import (
    PlannerLLMContext,
)
from src.backend.application.agent.case_agent.schemas.plan import (
    CaserBusinessSemanticPlan,
    CaserCapabilityPlanItem,
)
from src.backend.application.agent.case_agent.tools.registry import (
    manifest_item_for_capability,
    normalize_capability_name,
)
from src.backend.domain.case_memory.entities import MemoryType

from ._errors import state_error


def load_perceptual_state_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Read the single handoff object from the perception layer."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="load_perceptual_state",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        if not isinstance(state.get("perceptual_state"), dict):
            raise ValueError("perceptual_state is missing")
        service._repository.append_event(
            run.run_id,
            "perceptual_state_loaded",
            "已读取 PerceptualState",
            {
                "intent": (state.get("perceptual_state") or {}).get("semantic", {}).get("intent"),
                "decision_context_ref": state.get("decision_context_ref") or "",
            },
        )
        state["next_action"] = "decision_readiness_checker"
        return state
    except Exception as exc:
        return state_error(state, exc)


def decision_readiness_checker_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Check slots, confidence and governance readiness before planning."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="decision_readiness_checker",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        perceptual_state = state.get("perceptual_state") or {}
        readiness = perceptual_state.get("readiness_signal") or {}
        if readiness.get("fail_closed"):
            state["next_action"] = "fail_closed_plan"
        elif readiness.get("need_clarification"):
            state["next_action"] = "ask_clarification_plan"
        else:
            state["next_action"] = "decision_precheck"
        service._repository.append_event(
            run.run_id,
            "decision_readiness_checked",
            "已判断是否可规划",
            {
                "readiness": readiness.get("status"),
                "next_action": state["next_action"],
            },
        )
        return state
    except Exception as exc:
        return state_error(state, exc)


def ask_clarification_plan_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create a clarification plan from perceptual missing slots."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="ask_clarification_plan",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        missing_slots = list(state.get("missing_slots") or [])
        state["planning_output"] = {
            "query_semantics": {
                "intent": "clarification_required",
                "user_goal": _question(state),
                "granularity": "overview",
                "missing_slots": missing_slots,
            },
            "execution_plan": [],
        }
        state["next_action"] = "plan_normalizer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def fail_closed_plan_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Route a governance-blocked decision to fail-closed handling."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="fail_closed_plan",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        state["error_code"] = state.get("error_code") or "DecisionNotReady"
        state["error_message"] = state.get("error_message") or "Decision layer cannot safely plan this request."
        state["next_action"] = "fail_closed"
        return state
    except Exception as exc:
        return state_error(state, exc)


def decision_precheck_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Choose deterministic, heuristic or LLM planning branch."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="decision_precheck",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        if _should_reuse_previous_answer(state):
            strategy = "reuse_previous_answer"
            state.update(_previous_answer_reuse_state(state))
            state["planning_output"] = _reuse_previous_answer_plan(state)
            state["next_action"] = "plan_normalizer"
            service._repository.append_event(
                run.run_id,
                "previous_answer_reuse_planned",
                "已规划复用上一轮回答和引用",
                {
                    "reuse_answer_ref": state.get("reuse_answer_ref") or "",
                    "reuse_source_ref_count": len(state.get("reuse_source_refs") or []),
                    "rewrite_mode": state.get("answer_rewrite_mode") or "",
                },
            )
        elif state.get("rule_candidate_plan") is not None:
            strategy = "rule_based"
            state["next_action"] = "rule_based_planner"
        elif _should_use_heuristic_planner(state):
            strategy = "heuristic"
            state["next_action"] = "read_context_slice"
        else:
            strategy = "llm_planner"
            state["next_action"] = "planner_llm_context_builder"
        service._repository.append_event(
            run.run_id,
            "planner_strategy_selected",
            "已选择决策规划策略",
            {
                "strategy": strategy,
                "next_action": state["next_action"],
            },
        )
        return state
    except Exception as exc:
        return state_error(state, exc)


def rule_based_planner_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Generate an ExecutionPlan from deterministic perception signals."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="rule_based_planner",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        state["planning_output"] = state.get("rule_candidate_plan") or _fallback_general_help_plan().model_dump(
            mode="json"
        )
        service._repository.append_event(
            run.run_id,
            "rule_based_plan_created",
            "规则规划已生成 ExecutionPlan",
            {
                "step_count": len(state.get("planning_output", {}).get("execution_plan", [])),
            },
        )
        state["next_action"] = "plan_normalizer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def read_context_slice_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Read a small context slice from DecisionContextRef."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="read_context_slice",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        decision_context = state.get("decision_context") or {}
        state["decision_context_slice"] = {
            "context_ref": state.get("decision_context_ref") or "",
            "current_topic": decision_context.get("working_memory_context", {}).get("current_topic", ""),
            "source_refs": decision_context.get("domain_artifact_context", {}).get("source_ref", [])[:8],
        }
        service._repository.append_event(
            run.run_id,
            "decision_context_slice_loaded",
            "已读取决策上下文切片",
            {
                "context_ref": state["decision_context_slice"]["context_ref"],
                "source_ref_count": len(state["decision_context_slice"]["source_refs"]),
            },
        )
        state["next_action"] = "heuristic_planner"
        return state
    except Exception as exc:
        return state_error(state, exc)


def heuristic_planner_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Fallback deterministic planner for non-LLM multi-step routes."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="heuristic_planner",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        if _semantic_requires_policy_expert(state):
            state["planning_output"] = _heuristic_policy_expert_plan(state)
        else:
            state["planning_output"] = _fallback_general_help_plan().model_dump(mode="json")
        service._repository.append_event(
            run.run_id,
            "heuristic_plan_created",
            "启发式规划已生成 ExecutionPlan",
            {
                "step_count": len(state.get("planning_output", {}).get("execution_plan", [])),
            },
        )
        state["next_action"] = "plan_normalizer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def planner_llm_context_builder_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Project DecisionContext into PlannerLLMContext without a second GSSC."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="planner_llm_context_builder",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        decision_context = state.get("decision_context") or {}
        perceptual_state = state.get("perceptual_state") or {}
        context = PlannerLLMContext(
            task_context={
                "user_question": _planner_question(state),
                "semantic": perceptual_state.get("semantic") or {},
                "slots_focus": perceptual_state.get("slots_focus") or {},
            },
            scope_context={
                "case_id": state["run"].case_id,
                "current_case_only": True,
            },
            memory_context={
                **(decision_context.get("working_memory_context") or {}),
                "long_term_memory": service.memory_hint_for_node(
                    state,
                    consumer="decision_planner",
                    memory_type=MemoryType.DECISION_PLAN,
                    task_context={"question": _planner_question(state)},
                ),
            },
            artifact_context=decision_context.get("domain_artifact_context") or {},
            capability_context={
                "allowed_layers": ["L1", "L2", "L3"],
                "max_steps": 8,
            },
            governance_context=decision_context.get("governance_context") or {},
            output_contract={
                "schema": "CaserBusinessSemanticPlan",
                "only_output_steps": True,
            },
        )
        state["planner_llm_context"] = context.model_dump(mode="json")
        service._repository.append_event(
            run.run_id,
            "planner_llm_context_built",
            "已生成 PlannerLLMContext",
            {
                "context_ref": state.get("decision_context_ref") or "",
                "allowed_layers": context.capability_context.get("allowed_layers", []),
            },
        )
        state["next_action"] = "llm_planner"
        return state
    except Exception as exc:
        return state_error(state, exc)


def llm_planner_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Produce a structured ExecutionPlan from PerceptualState and DecisionContext."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="llm_planner",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            run.run_id,
            "llm_planning",
            "正在进行 LLM 规划",
            {"model": service._classifier_model},
        )
        response = service._gateway.complete(
            messages=[
                {
                    "role": "system",
                    "content": build_business_semantic_planning_prompt(run.case_id),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_question": _planner_question(state),
                            "recent_context": _recent_context_for_planner(
                                state.get("recent_messages", []),
                                state.get("session"),
                            ),
                            "planner_llm_context": state.get("planner_llm_context") or {},
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            tools=[],
            require_json=True,
            model=service._classifier_model,
            thinking_enabled=False,
            max_tokens=getattr(service, "_planner_max_tokens", 768),
            timeout_seconds=getattr(service, "_planner_timeout_seconds", 15),
        )
        state["model_call_count"] = state.get("model_call_count", 0) + 1
        recorder = getattr(service, "record_model_call_metrics", None)
        model_metrics = (
            recorder(state, "llm_planner", response)
            if callable(recorder)
            else dict(response.metrics)
        )
        state["planning_output"] = _parse_plan(response.content or "{}").model_dump(mode="json")
        service._repository.append_event(
            run.run_id,
            "llm_plan_created",
            "LLM 规划已生成 ExecutionPlan",
            {
                "source": "llm_planner",
                "step_count": len(state.get("planning_output", {}).get("execution_plan", [])),
                "model_metrics": model_metrics,
            },
        )
        state["next_action"] = "plan_normalizer"
        return state
    except (json.JSONDecodeError, ValidationError) as exc:
        state["planning_output"] = _fallback_general_help_plan().model_dump(mode="json")
        state["planning_error"] = f"{exc.__class__.__name__}: {exc}"
        state["next_action"] = "plan_normalizer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def plan_normalizer_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner branch output back to the existing execution contract."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="plan_normalizer",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        plan = CaserBusinessSemanticPlan.model_validate(
            state.get("planning_output") or _fallback_general_help_plan().model_dump(mode="json")
        )
        semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
        if not plan.query_semantics.information_needs:
            inferred_needs = semantic.get("information_needs") or semantic.get("target_objects") or []
            plan = plan.model_copy(
                update={
                    "query_semantics": plan.query_semantics.model_copy(
                        update={
                            "information_needs": [
                                str(item)[:80]
                                for item in inferred_needs[:12]
                                if str(item or "").strip()
                            ]
                        }
                    )
                }
            )
        plan, optimization = _optimize_plan(plan)
        payload = plan.model_dump(mode="json")
        state.update(payload)
        state["plan_optimization"] = optimization
        state["intent"] = plan.query_semantics.intent
        state["intent_confidence"] = state.get("intent_confidence", 0.85)
        state["slots"] = state.get("slots", {})
        state["missing_slots"] = plan.query_semantics.missing_slots
        state["context_plan"] = {
            "capabilities": [
                {
                    "step": item.step,
                    "name": item.capability,
                    "arguments": item.arguments,
                    "layer": item.layer,
                    "depends_on": item.depends_on,
                    "covers": item.covers,
                    "reason": item.reason,
                }
                for item in plan.execution_plan
            ],
        }
        service._repository.append_event(
            run.run_id,
            "execution_plan_normalized",
            "已生成结构化执行计划",
            {
                "intent": state["intent"],
                "step_count": len(plan.execution_plan),
                "removed_capabilities": optimization["removed_capabilities"],
            },
        )
        state["next_action"] = "validate_execution_plan"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _optimize_plan(
    plan: CaserBusinessSemanticPlan,
) -> tuple[CaserBusinessSemanticPlan, dict[str, Any]]:
    """Remove duplicate or non-covering read-only steps from dependency-free plans."""

    unique: list[CaserCapabilityPlanItem] = []
    seen: set[str] = set()
    removed: list[str] = []
    for item in plan.execution_plan:
        capability = normalize_capability_name(item.capability)
        if capability in seen:
            removed.append(capability)
            continue
        seen.add(capability)
        manifest = manifest_item_for_capability(capability)
        covers = item.covers
        if not covers and manifest is not None:
            covers = list(
                manifest.information_needs
                or ((manifest.section_key,) if manifest.section_key else (capability,))
            )
        unique.append(
            item.model_copy(
                update={
                    "capability": capability,
                    "covers": covers,
                }
            )
        )

    needs = {
        str(item).strip().lower()
        for item in plan.query_semantics.information_needs
        if str(item or "").strip()
    }
    can_minimize = bool(
        needs
        and len(unique) > 1
        and all(item.layer == "L1" and not item.depends_on and item.covers for item in unique)
    )
    selected = unique
    if can_minimize:
        remaining = list(unique)
        uncovered = set(needs)
        picked: list[CaserCapabilityPlanItem] = []
        while uncovered and remaining:
            ranked = sorted(
                remaining,
                key=lambda item: (
                    -len({str(value).strip().lower() for value in item.covers} & uncovered),
                    _capability_context_cost(item.capability),
                    item.step,
                ),
            )
            best = ranked[0]
            gain = {str(value).strip().lower() for value in best.covers} & uncovered
            if not gain:
                break
            picked.append(best)
            uncovered -= gain
            remaining.remove(best)
        if picked and not uncovered:
            picked_names = {item.capability for item in picked}
            removed.extend(item.capability for item in unique if item.capability not in picked_names)
            selected = [item for item in unique if item.capability in picked_names]

    normalized = [
        item.model_copy(update={"step": index})
        for index, item in enumerate(selected, start=1)
    ]
    return (
        plan.model_copy(update={"execution_plan": normalized}),
        {
            "information_needs": sorted(needs),
            "removed_capabilities": list(dict.fromkeys(removed)),
        },
    )


def _capability_context_cost(capability: str) -> int:
    manifest = manifest_item_for_capability(capability)
    return manifest.estimated_context_chars if manifest is not None else 10000


def _question(state: dict[str, Any]) -> str:
    user_message = state.get("user_message")
    return str(getattr(user_message, "content", "") or "")


def _planner_question(state: dict[str, Any]) -> str:
    semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
    rewritten = str(semantic.get("rewritten_query") or "").strip()
    return rewritten or _question(state)


def _should_use_heuristic_planner(state: dict[str, Any]) -> bool:
    perceptual_state = state.get("perceptual_state") or {}
    semantic = perceptual_state.get("semantic") or {}
    readiness = perceptual_state.get("readiness_signal") or {}
    if readiness.get("need_clarification") or readiness.get("fail_closed"):
        return False
    if _semantic_requires_policy_expert(state):
        return True
    if semantic.get("intent") == "general_help":
        return True
    if semantic.get("target_layer_hint") == "none" and semantic.get("evidence_need") == "none":
        return True
    return False


def _semantic_requires_policy_expert(state: dict[str, Any]) -> bool:
    semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
    if not isinstance(semantic, dict):
        return False
    focus = semantic.get("focus_candidate") if isinstance(semantic.get("focus_candidate"), dict) else {}
    values = [
        semantic.get("evidence_need"),
        focus.get("evidence_need"),
        semantic.get("target_layer_hint"),
        focus.get("target_layer_hint"),
        semantic.get("capability_hint"),
        focus.get("capability_hint"),
        semantic.get("action"),
        focus.get("action"),
    ]
    normalized = {str(value or "").strip() for value in values if value not in (None, "")}
    return bool(
        normalized.intersection(
            {
                "policy_evidence",
                "L3",
                "ask_policy_expert",
                "query_policy_basis",
                "policy_basis_query",
            }
        )
    )


def _heuristic_policy_expert_plan(state: dict[str, Any]) -> dict[str, Any]:
    question = _planner_question(state)
    run = state["run"]
    arguments: dict[str, Any] = {
        "case_id": run.case_id,
        "question": question,
        "goal": question,
    }
    semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
    focus = semantic.get("focus_candidate") if isinstance(semantic.get("focus_candidate"), dict) else {}
    filters = (
        semantic.get("filters")
        or semantic.get("policy_filters")
        or focus.get("filters")
        or focus.get("policy_filters")
    )
    if isinstance(filters, dict) and filters:
        arguments["filters"] = filters
    return CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "expert_task",
                "user_goal": question or "查询政策依据",
                "granularity": "analysis",
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L3",
                    "capability": "ask_policy_expert",
                    "arguments": arguments,
                    "depends_on": [],
                    "reason": "感知层已标记需要 policy_evidence / L3，由启发式规划生成 Policy Expert 执行步骤。",
                }
            ],
        }
    ).model_dump(mode="json")


def _should_reuse_previous_answer(state: dict[str, Any]) -> bool:
    payload = _previous_answer_reuse_state(state)
    if not payload.get("reuse_answer_ref"):
        return False
    semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
    action = str(semantic.get("action") or "")
    evidence_need = str(semantic.get("evidence_need") or "")
    requires_new = semantic.get("requires_new_evidence")
    return (
        payload.get("answer_strategy") == "reuse_previous_answer"
        or action == "explain_previous_answer"
        or evidence_need == "previous_answer"
        or requires_new is False
    )


def _previous_answer_reuse_state(state: dict[str, Any]) -> dict[str, Any]:
    perceptual_state = state.get("perceptual_state") or {}
    semantic = perceptual_state.get("semantic") or {}
    refs = perceptual_state.get("context_refs") or {}
    slots = state.get("slots") or {}
    source_refs = (
        semantic.get("reuse_source_refs")
        or slots.get("reuse_source_refs")
        or refs.get("source_ref")
        or []
    )
    if not isinstance(source_refs, list):
        source_refs = []
    return {
        "answer_strategy": slots.get("answer_strategy")
        or ("reuse_previous_answer" if semantic.get("evidence_need") == "previous_answer" else ""),
        "reuse_answer_ref": semantic.get("reuse_answer_ref")
        or slots.get("reuse_answer_ref")
        or refs.get("answer_ref")
        or "",
        "reuse_source_refs": [
            str(ref) for ref in source_refs[:20]
            if isinstance(ref, str) and ref
        ],
        "answer_rewrite_mode": semantic.get("rewrite_mode")
        or slots.get("rewrite_mode")
        or "explain",
    }


def _reuse_previous_answer_plan(state: dict[str, Any]) -> dict[str, Any]:
    rewrite_mode = str(state.get("answer_rewrite_mode") or "explain")
    goals = {
        "translate_zh": "将上一轮回答准确改写为中文",
        "translate_en": "将上一轮回答准确改写为英文",
        "example": "用一个简单直观的例子说明上一轮回答",
        "simplify": "用更通俗简洁的方式解释上一轮回答",
        "paraphrase": "换句话解释上一轮回答",
        "summarize": "简短总结上一轮回答",
        "list": "将上一轮回答整理为清晰列表",
        "expand": "在不增加新事实的前提下展开解释上一轮回答",
        "explain": "继续解释上一轮回答",
    }
    return {
        "query_semantics": {
            "intent": "case_task",
            "user_goal": goals.get(rewrite_mode, goals["explain"]),
            "granularity": "analysis",
            "missing_slots": [],
        },
        "execution_plan": [],
    }
