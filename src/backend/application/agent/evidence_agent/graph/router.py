"""StateGraph 条件路由函数。

每个函数根据 AgentState 中的 next_action 字段决定下一步节点。
"""

from __future__ import annotations

from typing import Any


def route_after_model(state: dict[str, Any]) -> str:
    """模型调用后的条件路由。

    可能返回：tools, finalize, fail。
    """
    return state["next_action"]


def route_after_tools(state: dict[str, Any]) -> str:
    """工具调用后的条件路由。

    可能返回：model, fail。
    """
    return state["next_action"]


def route_after_finalize(state: dict[str, Any]) -> str:
    """结果终审后的条件路由。

    可能返回：model, fail, completed。
    """
    return state["next_action"]
