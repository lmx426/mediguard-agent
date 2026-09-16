"""Case Agent node error helpers."""

from __future__ import annotations

from typing import Any


def state_error(state: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """Route a node failure to fail-closed handling."""

    state["error"] = exc
    state["error_code"] = exc.__class__.__name__
    state["error_message"] = str(exc)
    state["next_action"] = "fail_closed"
    return state
