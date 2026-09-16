"""Expert Analysis graph builder."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.expert_agent.graph.state import ExpertAgentState


def build_expert_agent_graph(service: Any):
    """Build the controlled L3 Expert Analysis compiled graph."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Expert Agent 需要 langgraph 包。请安装 environment.yml 中的依赖。"
        ) from exc

    graph = StateGraph(ExpertAgentState)
    graph.add_node("init_task", service._init_task_node)
    graph.add_node("load_task_context", service._load_task_context_node)
    graph.add_node("task_understanding", service._task_understanding_node)
    graph.add_node("case_context_adapter", service._case_context_adapter_node)
    graph.add_node("policy_tool_adapter", service._policy_tool_adapter_node)
    graph.add_node("hard_gate", service._hard_gate_node)
    graph.add_node("need_evidence_resolution", service._need_evidence_resolution_node)
    graph.add_node("answerability_check", service._answerability_check_node)
    graph.add_node("follow_up_retrieval", service._follow_up_retrieval_node)
    graph.add_node("expert_synthesis", service._expert_synthesis_node)
    graph.add_node("return_unavailable", service._return_unavailable_node)
    graph.set_entry_point("init_task")
    graph.add_conditional_edges(
        "init_task",
        _route_next,
        {
            "load_task_context": "load_task_context",
            "return_unavailable": "return_unavailable",
        },
    )
    graph.add_edge("load_task_context", "task_understanding")
    graph.add_conditional_edges(
        "task_understanding",
        _route_next,
        {
            "case_context_adapter": "case_context_adapter",
            "policy_tool_adapter": "policy_tool_adapter",
            "return_unavailable": "return_unavailable",
        },
    )
    graph.add_edge("case_context_adapter", "policy_tool_adapter")
    graph.add_edge("policy_tool_adapter", "hard_gate")
    graph.add_edge("hard_gate", "need_evidence_resolution")
    graph.add_edge("need_evidence_resolution", "answerability_check")
    graph.add_conditional_edges(
        "answerability_check",
        _route_next,
        {
            "follow_up_retrieval": "follow_up_retrieval",
            "question_rewrite": "follow_up_retrieval",
            "expert_synthesis": "expert_synthesis",
            "return_unavailable": "return_unavailable",
        },
    )
    graph.add_conditional_edges(
        "follow_up_retrieval",
        _route_next,
        {
            "policy_tool_adapter": "policy_tool_adapter",
            "expert_synthesis": "expert_synthesis",
            "return_unavailable": "return_unavailable",
        },
    )
    graph.add_edge("expert_synthesis", END)
    graph.add_edge("return_unavailable", END)
    return graph.compile()


def _route_next(state: dict[str, Any]) -> str:
    return str(state.get("next_action") or "return_unavailable")
