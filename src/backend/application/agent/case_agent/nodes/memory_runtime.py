"""Case Agent hooks for governed long-term memory runtime loading."""

from __future__ import annotations

from typing import Any

from src.backend.domain.case_memory.entities import MemoryType

from ._errors import state_error


def build_memory_retrieval_plan_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create a backend-owned post-intent memory plan."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="build_memory_retrieval_plan",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        semantic = (state.get("perceptual_state") or {}).get("semantic") or {}
        layer = str(semantic.get("target_layer_hint") or "none").upper()
        items: list[dict[str, Any]] = []
        intent = str(semantic.get("intent") or state.get("intent") or "")
        if (
            intent not in {"", "general_help", "clarification_required"}
            and str(semantic.get("source") or "") != "rule_precheck"
        ):
            items.append(
                {
                    "memory_type": MemoryType.INTENT_ROUTE.value,
                    "consumers": ["intent_router"],
                    "max_items": 3,
                    "wait_timeout_ms": 0,
                    "mode": "shadow",
                }
            )
        if layer == "L2":
            items.append(
                {
                    "memory_type": MemoryType.DECISION_PLAN.value,
                    "consumers": ["decision_planner"],
                    "max_items": 3,
                    "wait_timeout_ms": 300,
                }
            )
        elif layer == "L3":
            items.append(
                {
                    "memory_type": MemoryType.POLICY_SEARCH.value,
                    "consumers": ["policy_filter_resolver", "expert_analysis"],
                    "max_items": 3,
                    "wait_timeout_ms": 500,
                }
            )
        state["memory_retrieval_plan"] = {
            "version": "v1",
            "source": "deterministic_layer_mapping",
            "target_layer": layer,
            "intent_shadow_only": True,
            "items": items,
        }
        service._repository.append_event(
            run.run_id,
            "memory_retrieval_plan_built",
            "已按最终意图生成请求级记忆计划",
            {
                "target_layer": layer,
                "memory_types": [item["memory_type"] for item in items],
            },
        )
        state["next_action"] = "start_request_memory_prefetch"
        return state
    except Exception as exc:
        return state_error(state, exc)


def start_request_memory_prefetch_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Submit the planned memory work without waiting for retrieval."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="start_request_memory_prefetch",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        status = service.start_request_memory_prefetch(state)
        state["request_memory_status"] = status
        service._repository.append_event(
            run.run_id,
            "memory_request_prefetch_started",
            "已启动意图相关的请求级记忆预取",
            status,
        )
        state["next_action"] = "load_perceptual_state"
        return state
    except Exception as exc:
        return state_error(state, exc)


def prefetch_request_memory_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Prefetch request-level candidates after the input and basic entities are known."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="memory_prefetch",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        status = service.prefetch_request_memory(state)
        state["request_memory_status"] = status
        service._repository.append_event(
            run.run_id,
            "memory_request_prefetched",
            "已完成请求级记忆候选批预取",
            status,
        )
        state["next_action"] = "semantic_intent_perception"
        return state
    except Exception as exc:
        return state_error(state, exc)


def memory_hint_for_node(
    service: Any,
    state: dict[str, Any],
    *,
    consumer: str,
    memory_type: MemoryType,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a view-trimmed hint pack, falling back to one targeted recall if needed."""

    return service.memory_hint_for_node(
        state,
        consumer=consumer,
        memory_type=memory_type,
        task_context=task_context,
    )
