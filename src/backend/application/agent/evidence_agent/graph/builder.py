"""Review Advisor LangGraph builder."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.evidence_agent.graph.router import (
    route_after_finalize,
    route_after_model,
    route_after_tools,
)
from src.backend.application.agent.evidence_agent.graph.state import AgentState
from src.backend.application.agent.evidence_agent.nodes.call_tools import tool_node
from src.backend.application.agent.evidence_agent.nodes.fail_closed import fail_node
from src.backend.application.agent.evidence_agent.nodes.finalize import finalize_node
from src.backend.application.agent.evidence_agent.nodes.load_context import (
    load_context_node,
)
from src.backend.application.agent.evidence_agent.nodes.model import model_node


def build_review_advisor_graph(service: Any):
    """Build the Review Advisor StateGraph without changing behavior."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Review Advisor 需要 langgraph 包。请安装 environment.yml 中的依赖。"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("load_context", lambda s: load_context_node(service, s))
    graph.add_node("model", lambda s: model_node(service, s))
    graph.add_node("tools", lambda s: tool_node(service, s))
    graph.add_node("finalize", lambda s: finalize_node(service, s))
    graph.add_node("fail", lambda s: fail_node(service, s))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "model")
    graph.add_conditional_edges(
        "model",
        route_after_model,
        {"model": "model", "tools": "tools", "finalize": "finalize", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"model": "model", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {"model": "model", "fail": "fail", "completed": END},
    )
    graph.add_edge("fail", END)
    return graph.compile()


build_evidence_agent_graph = build_review_advisor_graph
