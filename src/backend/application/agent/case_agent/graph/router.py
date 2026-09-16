"""Case Agent StateGraph routers."""

from __future__ import annotations

from typing import Any


def route_next(state: dict[str, Any]) -> str:
    """Route by the node-selected next action."""

    return state.get("next_action", "fail_closed")
