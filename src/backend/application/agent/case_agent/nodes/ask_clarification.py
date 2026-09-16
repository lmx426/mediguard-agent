"""Case Agent clarification node."""

from __future__ import annotations

from typing import Any

from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentContentBlock,
)


SLOT_LABELS = {
    "region": "适用地区",
    "policy_subject": "政策对象",
    "query_topic": "查询主题",
    "drug_name": "药品名称",
    "comparison_dimension": "同类口径比较维度",
    "business_module": "需要查询的业务内容",
}


def ask_clarification_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create a short clarification answer when key slots are missing."""

    run = state["run"]
    service._repository.update_run(
        run.run_id,
        current_node="ask_clarification",
        model_call_count=state.get("model_call_count", 0),
        tool_call_count=state.get("tool_call_count", 0),
    )
    missing = state.get("missing_slots", [])
    labels = list(dict.fromkeys(SLOT_LABELS.get(slot, "必要信息") for slot in missing))
    text = (
        f"为了继续处理这个问题，请补充：{'、'.join(labels)}。"
        if labels
        else "请换一种更具体的问法，我再继续帮你核验。"
    )
    state["answer"] = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[CaseAgentContentBlock(text=text)],
        sources=[],
        fallback_notice="",
        metadata={
            "intent": state.get("intent", "unknown"),
            "requires_citation": False,
            "capabilities_used": [],
            "missing_slots": missing,
        },
    )
    state["pending_clarification"] = {
        "intent": state.get("intent", "unknown"),
        "missing_slots": missing,
        "prompt": text,
        "slots": state.get("slots", {}),
        "task_id": str((state.get("task_state") or {}).get("task_id") or ""),
        "semantic": {
            key: value
            for key, value in ((state.get("perceptual_state") or {}).get("semantic") or {}).items()
            if key
            in {
                "intent",
                "action",
                "answer_shape",
                "target_layer_hint",
                "target_objects",
                "information_needs",
                "rewritten_query",
                "evidence_need",
            }
        },
        "user_message_id": state.get("user_message").message_id
        if state.get("user_message") is not None
        else None,
    }
    service._repository.append_event(
        run.run_id,
        "clarification_requested",
        "需要补充关键信息",
        {"missing_slots": missing},
    )
    state["completion_status"] = "waiting_for_user"
    state["next_action"] = "persist_result"
    return state
