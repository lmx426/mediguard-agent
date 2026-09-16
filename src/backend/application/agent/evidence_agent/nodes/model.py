"""模型调用节点。

负责调用 LLM 网关完成证据工具选择或最终 JSON 生成。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.domain.agent.entities import ReviewAdvisoryGeneratedContentV2
from ..prompts.loader import build_structure_repair_prompt


MODEL_COMPLETED_METRIC_KEYS = (
    "model",
    "latency_ms",
    "message_count",
    "input_chars",
    "tool_schema_count",
    "tool_schema_chars",
    "require_json",
    "thinking_enabled",
    "reasoning_effort",
    "output_chars",
    "reasoning_chars",
    "response_tool_call_count",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def model_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """调用 LLM 模型进行下一步推理。

    Args:
        service: ReviewAdvisorService 实例，通过它访问网关、设置等依赖。
        state: 当前 AgentState。

    Returns:
        更新后的 AgentState，包含模型响应和下一步路由。
    """

    # 超时检查
    if service._timed_out(state):
        return _state_error(state, "timeout", "Agent timed out")
    # 模型调用预算检查
    if state["model_call_count"] >= service._settings.evidence_agent_max_model_calls:
        return _state_error(state, "model_budget_exceeded", "Model call budget exceeded")
    run_id = state["run_id"]
    eval_variant = state.get("request", {}).get("eval_variant") or "A0"
    state["model_call_count"] += 1
    service._repository.update_run(
        run_id,
        current_node="model",
        model_call_count=state["model_call_count"],
        tool_call_count=state["tool_call_count"],
    )
    service._repository.append_event(
        run_id,
        "generating" if state.get("ledger") else "planning",
        "Agent is organizing evidence analysis"
        if state.get("ledger")
        else "Agent is selecting evidence tools",
    )
    # 判断是否需要强制输出最终 JSON
    force_final_json = (
        state.get("force_final_json")
        or state["model_call_count"] >= service._settings.evidence_agent_max_model_calls - 1
    )
    direct_json_without_tools = eval_variant == "B4"
    tools_for_model = (
        []
        if force_final_json or direct_json_without_tools
        else service._tools.schemas_for_model(stage="post_prefetch")
    )
    require_json = bool(force_final_json or direct_json_without_tools)
    # 调用模型网关
    response = service._gateway.complete(
        messages=state["messages"],
        tools=tools_for_model,
        require_json=require_json,
        thinking_enabled=False if eval_variant == "B1" else None,
    )
    if not isinstance(response, ModelResponse):
        raise TypeError("ModelGateway returned an unsupported response")
    service._repository.append_event(
        run_id,
        "model_completed",
        "Model call completed with redacted timing metrics",
        _model_completed_payload(
            response=response,
            model_call_index=state["model_call_count"],
            force_final_json=bool(force_final_json),
            require_json=require_json,
            eval_variant=eval_variant,
        ),
    )
    state["pending_response"] = response
    # 如果模型返回了工具调用
    if response.tool_calls:
        state["messages"].append(response.ephemeral_assistant_message())
        state["planning_rounds"] += 1
        if state["planning_rounds"] > service._settings.evidence_agent_max_planning_rounds:
            return _state_error(state, "planning_budget_exceeded", "Planning round budget exceeded")
        state["next_action"] = "tools"
        return state

    # 检查模型内容是否为空
    if not response.content:
        return _state_error(state, "empty_model_output", "Model returned empty output")
    # 尝试解析模型输出为最终 JSON
    try:
        payload = json.loads(response.content)
        state["final_content"] = ReviewAdvisoryGeneratedContentV2.model_validate(payload).model_dump(mode="json")
    except (json.JSONDecodeError, ValidationError) as exc:
        # 尝试结构修复
        if _can_repair_structure_error(state, service._settings):
            state["structure_repair_count"] = state.get("structure_repair_count", 0) + 1
            state["force_final_json"] = True
            state["messages"].append(
                {
                    "role": "user",
                    "content": build_structure_repair_prompt(
                        error_message=exc.__class__.__name__,
                        ledger=state["ledger"],
                        eval_variant=eval_variant,
                    ),
                }
            )
            service._repository.append_event(
                run_id,
                "repairing_structure",
                "Model output did not match the required schema; retrying one controlled JSON rewrite",
                {"error_code": "invalid_model_output"},
            )
            service._checkpoint(state, "structure_repair")
            state["next_action"] = "model"
            return state
        return _state_error(state, "invalid_model_output", f"Model output structure validation failed: {exc.__class__.__name__}")
    state["next_action"] = "finalize"
    return state


def _can_repair_structure_error(state: dict[str, Any], settings: Any) -> bool:
    """判断是否可以重试修复模型输出的结构错误。"""
    return (
        state.get("structure_repair_count", 0) < 1
        and state["model_call_count"] < settings.evidence_agent_max_model_calls
    )


def _model_completed_payload(
    *,
    response: ModelResponse,
    model_call_index: int,
    force_final_json: bool,
    require_json: bool,
    eval_variant: str,
) -> dict[str, Any]:
    metrics = response.metrics if isinstance(response.metrics, dict) else {}
    payload: dict[str, Any] = {
        "model_call_index": model_call_index,
        "force_final_json": force_final_json,
        "require_json": require_json,
        "eval_variant": eval_variant,
    }
    for key in MODEL_COMPLETED_METRIC_KEYS:
        if key in metrics:
            payload[key] = _safe_metric_value(metrics[key])

    payload.setdefault("latency_ms", None)
    payload.setdefault("input_chars", None)
    payload.setdefault("tool_schema_count", None)
    payload.setdefault("tool_schema_chars", None)
    payload.setdefault("output_chars", len(response.content or ""))
    payload.setdefault("reasoning_chars", len(response.reasoning_content or ""))
    payload.setdefault("prompt_tokens", None)
    payload.setdefault("completion_tokens", None)
    payload.setdefault("total_tokens", None)
    payload.setdefault("response_tool_call_count", len(response.tool_calls))
    return payload


def _safe_metric_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _state_error(state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    """设置错误状态并路由到 fail 节点。"""
    state["error_code"] = code
    state["error_message"] = message
    state["next_action"] = "fail"
    return state
