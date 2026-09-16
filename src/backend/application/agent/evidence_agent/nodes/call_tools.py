"""工具调用节点。

负责将模型请求的工具调用转发给 ReviewAdvisorToolRegistry，
收集证据并更新账本。
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.backend.application.agent.runtime.gateways.base import ModelResponse
from ..prompts.loader import build_collected_evidence_prompt
from ..tools.registry import ToolExecutionError, tool_signature


def tool_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """执行模型请求的工具调用并收集证据。

    Args:
        service: ReviewAdvisorService 实例，通过它访问工具注册表、仓储等依赖。
        state: 当前 AgentState。

    Returns:
        更新后的 AgentState，包含工具执行结果和更新的证据账本。
    """

    response: ModelResponse = state["pending_response"]
    eval_variant = state.get("request", {}).get("eval_variant") or "A0"
    if not response.tool_calls:
        state["next_action"] = "model"
        return state
    run_id = state["run_id"]
    added = 0
    for call in response.tool_calls:
        # 工具调用预算检查
        if state["tool_call_count"] >= service._settings.evidence_agent_max_tool_calls:
            return _state_error(state, "tool_budget_exceeded", "Tool call budget exceeded")
        state["tool_call_count"] += 1
        start = time.monotonic()
        if not service._tools.is_model_callable(call.name, stage="post_prefetch"):
            result = {
                "ok": False,
                "error": {
                    "code": "tool_not_available_in_stage",
                    "message": "该工具已由框架预取或不属于当前阶段可调用工具，请依据 tool_catalog 选择细粒度补查工具。",
                },
            }
            service._repository.record_tool_call(
                run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments={},
                result=None,
                status="failed",
                error_code="tool_not_available_in_stage",
                latency_ms=int((time.monotonic() - start) * 1000),
            )
            state["messages"].append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
            )
            state["argument_error_count"] += 1
            continue
        # 尝试解析工具参数为 JSON
        try:
            raw_args = json.loads(call.arguments)
        except json.JSONDecodeError:
            raw_args = {"_invalid_json": call.arguments[:200]}
        # 重复调用检测
        signature = tool_signature(call.name, raw_args)
        if signature in state["seen_signatures"]:
            state["repeat_count"] += 1
            result = {"ok": False, "error": {"code": "repeated_call", "message": "Repeated tool call blocked"}}
            state["messages"].append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
            )
            if state["repeat_count"] > 1:
                return _state_error(state, "repeated_call", "Repeated tool call budget exceeded")
            continue
        state["seen_signatures"].append(signature)
        service._repository.append_event(
            run_id,
            "tool_running",
            "Reading read-only business evidence",
            {"tool": call.name},
        )
        # 执行工具调用
        try:
            data, items, validated_args = service._tools.execute(
                current_case_id=state["case_id"],
                tool_name=call.name,
                raw_arguments=call.arguments,
            )
            before = len(state["ledger"])
            for item in items:
                state["ledger"][item.ledger_ref] = item.model_dump(mode="json")
            added += len(state["ledger"]) - before
            result = {
                "ok": True,
                "data": data,
                "evidence_items": [item.model_dump(mode="json") for item in items],
            }
            service._repository.record_tool_call(
                run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=validated_args,
                result=result,
                status="success",
                error_code=None,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        except ToolExecutionError as exc:
            state["argument_error_count"] += 1
            result = {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
            service._repository.record_tool_call(
                run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=raw_args,
                result=None,
                status="failed",
                error_code=exc.code,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
            if state["argument_error_count"] > 2:
                return _state_error(state, "tool_argument_repair_exceeded", "Tool argument repair budget exceeded")
        state["messages"].append(
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)}
        )
    # 判断证据收集是否饱和
    state["no_new_evidence_rounds"] = (
        state["no_new_evidence_rounds"] + 1 if added == 0 else 0
    )
    collection_saturated = state["no_new_evidence_rounds"] >= 2
    if collection_saturated:
        state["force_final_json"] = True
        state["messages"].append(
            {
                "role": "user",
                "content": build_collected_evidence_prompt(
                    state["ledger"],
                    eval_variant=eval_variant,
                ),
            }
        )
    service._repository.update_run(
        run_id,
        current_node="tools",
        model_call_count=state["model_call_count"],
        tool_call_count=state["tool_call_count"],
    )
    service._repository.append_event(
        run_id,
        "ledger_updated",
        "Evidence ledger updated",
        {"source_count": len(state["ledger"])},
    )
    if collection_saturated:
        service._repository.append_event(
            run_id,
            "evidence_saturated",
            "Evidence collection saturated; generating final JSON",
            {"source_count": len(state["ledger"])},
        )
    # 预算耗尽时强制收集证据
    if (
        state["tool_call_count"] >= service._settings.evidence_agent_max_tool_calls
        or state["model_call_count"] == service._settings.evidence_agent_max_model_calls - 1
    ):
        state["messages"].append(
            {
                "role": "user",
                "content": build_collected_evidence_prompt(
                    state["ledger"],
                    eval_variant=eval_variant,
                ),
            }
        )
    service._checkpoint(state, "tools")
    state["next_action"] = "model"
    return state


def _state_error(state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    """设置错误状态并路由到 fail 节点。"""
    state["error_code"] = code
    state["error_message"] = message
    state["next_action"] = "fail"
    return state
