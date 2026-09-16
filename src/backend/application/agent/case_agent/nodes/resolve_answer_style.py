"""Resolve answer expression style for Caser generation."""

from __future__ import annotations

from typing import Any

from ._errors import state_error


def resolve_answer_style_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Turn query granularity into concise generation rules."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="resolve_answer_style",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        style = _build_answer_style(state)
        session_memory_reader = getattr(service, "session_memory_context", None)
        memory_pack = session_memory_reader(state) if callable(session_memory_reader) else {}
        style["memory_prompt_contexts"] = memory_pack.get("prompt_contexts", [])
        style["memory_trace_refs"] = memory_pack.get("trace_refs", [])
        state["answer_style_policy"] = style
        service._repository.append_event(
            run.run_id,
            "answer_style_resolved",
            "正在确定回答方式",
            {
                "granularity": style["granularity"],
                "answer_tone": style["answer_tone"],
                "max_blocks": style["max_blocks"],
            },
        )
        state["next_action"] = "generate_answer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _build_answer_style(state: dict[str, Any]) -> dict[str, Any]:
    semantics = state.get("query_semantics", {})
    if not isinstance(semantics, dict):
        semantics = {}
    answer_policy = state.get("answer_policy", {})
    if not isinstance(answer_policy, dict):
        answer_policy = {}
    display_mode = answer_policy.get("display_mode") or state.get("context_plan", {}).get("display_mode") or "plain"
    granularity = _normalize_granularity(semantics.get("granularity"))
    if state.get("answer_strategy") == "reuse_previous_answer":
        return {
            "granularity": "analysis",
            "answer_tone": "previous_answer_rewrite",
            "max_blocks": 2,
            "max_sentences_per_block": 3,
            "include_interpretation": True,
            "include_next_focus": False,
            "forbid_extra_fields": True,
            "business_expression": "只基于上一轮回答做通俗解释、举例或改写，复用上一轮引用，不新增政策或案件事实。",
        }
    if display_mode in {"unavailable", "error"}:
        return {
            "granularity": granularity,
            "answer_tone": "friendly_degraded",
            "max_blocks": 1,
            "max_sentences_per_block": 1,
            "include_interpretation": False,
            "include_next_focus": False,
            "forbid_extra_fields": True,
            "business_expression": "只说明能力状态或数据状态，不补通用知识，不展开内部错误。",
        }
    if display_mode == "plain":
        return {
            "granularity": granularity,
            "answer_tone": "concise_product_help",
            "max_blocks": 2,
            "max_sentences_per_block": 2,
            "include_interpretation": False,
            "include_next_focus": False,
            "forbid_extra_fields": True,
            "business_expression": "只回答用户问到的功能说明或会话说明，不引用案件资料。",
        }
    if granularity == "single_field":
        return {
            "granularity": granularity,
            "answer_tone": "direct_fact",
            "max_blocks": 1,
            "max_sentences_per_block": 1,
            "include_interpretation": False,
            "include_next_focus": False,
            "forbid_extra_fields": True,
            "business_expression": "只回答请求字段和值，不顺带输出同 section 其他字段。",
        }
    if granularity == "field_group":
        return {
            "granularity": granularity,
            "answer_tone": "compact_fact_group",
            "max_blocks": 2,
            "max_sentences_per_block": 2,
            "include_interpretation": False,
            "include_next_focus": False,
            "forbid_extra_fields": False,
            "business_expression": "按字段组简洁说明，避免长报告。",
        }
    if granularity == "list":
        return {
            "granularity": granularity,
            "answer_tone": "complete_list",
            "max_blocks": 3,
            "max_sentences_per_block": 2,
            "include_interpretation": False,
            "include_next_focus": False,
            "forbid_extra_fields": False,
            "business_expression": "优先完整列出用户要求的清单或明细，不改写成泛泛摘要。",
        }
    if granularity == "detail":
        return {
            "granularity": granularity,
            "answer_tone": "focused_detail",
            "max_blocks": 4,
            "max_sentences_per_block": 2,
            "include_interpretation": True,
            "include_next_focus": False,
            "forbid_extra_fields": False,
            "business_expression": "围绕指定材料、规则或线索展开必要详情。",
        }
    if granularity == "analysis":
        return {
            "granularity": granularity,
            "answer_tone": "audit_analysis",
            "max_blocks": 4,
            "max_sentences_per_block": 2,
            "include_interpretation": True,
            "include_next_focus": True,
            "forbid_extra_fields": False,
            "business_expression": "基于已有事实解释风险、偏离或核验含义；不形成业务处置结论。",
        }
    return {
        "granularity": "overview",
        "answer_tone": "business_overview",
        "max_blocks": 3,
        "max_sentences_per_block": 2,
        "include_interpretation": True,
        "include_next_focus": False,
        "forbid_extra_fields": False,
        "business_expression": "给出简洁业务摘要，避免数据库字段堆叠。",
    }


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
