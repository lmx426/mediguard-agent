"""Case Agent fail-closed node."""

from __future__ import annotations

from typing import Any


def fail_closed_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Convert any runtime failure into a safe fallback answer."""

    message = state.get("error_message") or "Case Agent failed"
    state["answer"] = service._fallback_answer(message, "error")
    state["completion_status"] = "degraded"
    state["next_action"] = "persist_result"
    return state
