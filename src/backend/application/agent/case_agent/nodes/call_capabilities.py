"""Case Agent controlled capability execution node."""

from __future__ import annotations

from typing import Any

from ._errors import state_error


def call_capabilities_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Execute backend-resolved capabilities and collect available sources."""

    try:
        run = state["run"]
        plan = state.get("context_plan", {})
        capabilities = state.get("execution_plan") or plan.get("capabilities", [])
        service._repository.update_run(
            run.run_id,
            current_node="call_capabilities",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        results = []
        for capability in sorted(capabilities, key=_capability_step):
            capability_name = _capability_name(capability)
            arguments = _capability_arguments(capability, run.case_id)
            service._repository.append_event(
                run.run_id,
                "capability_running",
                "正在读取相关资料",
                {"capability": capability_name},
            )
            if capability_name == "ask_policy_expert":
                expert_result = service._run_expert_capability(
                    run.run_id,
                    "policy_analysis",
                    state,
                    tool_name=capability_name,
                    arguments=arguments,
                )
                result = service._expert_result_to_tool_result(
                    case_id=run.case_id,
                    capability=capability_name,
                    result=expert_result,
                )
            else:
                result = service._run_tool(
                    run.run_id,
                    run.case_id,
                    f"system_capability:{capability_name}",
                    capability_name,
                    arguments,
                )
            state["tool_call_count"] = state.get("tool_call_count", 0) + 1
            results.append((capability_name, result))

        state["capability_results"] = results
        state["available_sources"] = service._collect_available_sources(results)
        state["next_action"] = "build_answer_context"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _capability_name(capability: Any) -> str:
    if isinstance(capability, dict):
        return str(capability.get("name") or capability.get("capability") or "")
    return str(capability)


def _capability_step(capability: Any) -> int:
    if not isinstance(capability, dict):
        return 0
    try:
        return int(capability.get("step") or 0)
    except (TypeError, ValueError):
        return 0


def _capability_arguments(capability: Any, case_id: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {"case_id": case_id}
    if isinstance(capability, dict) and isinstance(capability.get("arguments"), dict):
        arguments.update(capability["arguments"])
    return arguments
