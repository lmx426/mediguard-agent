"""Expert Agent unavailable result node."""

from __future__ import annotations

from typing import Any


def return_unavailable_node(state: dict[str, Any]) -> dict[str, Any]:
    """Return a controlled unavailable result until expert RAG is connected."""

    task_type = state.get("expert_task_type", "unknown")
    state["status"] = "unavailable"
    state["message"] = "专家能力尚未接入"
    state["result"] = {
        "status": "unavailable",
        "expert_task_type": task_type,
        "message": "专家能力尚未接入",
        "evidence": [],
    }
    return state
