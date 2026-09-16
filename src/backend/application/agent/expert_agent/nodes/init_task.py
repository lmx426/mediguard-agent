"""Expert Agent task initialization."""

from __future__ import annotations

from typing import Any


def init_task_node(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize the expert task state."""

    state.setdefault("input_factors", {})
    state.setdefault("status", "queued")
    return state
