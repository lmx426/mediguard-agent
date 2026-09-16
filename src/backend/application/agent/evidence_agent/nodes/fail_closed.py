"""失败终止节点。

负责在 Agent 运行出现不可恢复错误时记录失败状态并优雅终止。
"""

from __future__ import annotations

from typing import Any


def fail_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """处理 Agent 运行失败并记录失败事件。

    Args:
        service: ReviewAdvisorService 实例，通过它访问仓储。
        state: 当前 AgentState。

    Returns:
        更新后的 AgentState。
    """

    _fail(service, state["run_id"], state.get("error_code", "agent_failed"), state.get("error_message", "Agent failed"))
    return state


def _fail(service: Any, run_id: str, code: str, message: str) -> None:
    """记录 Agent 运行失败并更新仓储状态。

    Args:
        service: ReviewAdvisorService 实例。
        run_id: Agent 运行标识。
        code: 错误代码。
        message: 错误描述，最多 500 字符。
    """

    service._repository.update_run(
        run_id,
        status="failed",
        current_node="failed",
        error_code=code,
        error_message=message[:500],
    )
    service._repository.append_event(
        run_id,
        "failed",
        "Review Advisor run failed; base evidence remains available",
        {"error_code": code},
    )


def _state_error(state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    """设置错误状态并路由到 fail 节点。

    Args:
        state: 当前 AgentState。
        code: 错误代码。
        message: 错误描述。

    Returns:
        更新后的 AgentState，next_action 设为 fail。
    """

    state["error_code"] = code
    state["error_message"] = message
    state["next_action"] = "fail"
    return state
