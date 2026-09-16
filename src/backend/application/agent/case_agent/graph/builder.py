"""Case Agent LangGraph builder."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.graph.router import route_next
from src.backend.application.agent.case_agent.graph.state import CaseAgentState
from src.backend.application.agent.case_agent.nodes.ask_clarification import (
    ask_clarification_node,
)
from src.backend.application.agent.case_agent.nodes.build_answer_context import (
    build_answer_context_node,
)
from src.backend.application.agent.case_agent.nodes.call_capabilities import (
    call_capabilities_node,
)
from src.backend.application.agent.case_agent.nodes.decision import (
    ask_clarification_plan_node,
    decision_precheck_node,
    decision_readiness_checker_node,
    fail_closed_plan_node,
    heuristic_planner_node,
    llm_planner_node,
    load_perceptual_state_node,
    plan_normalizer_node,
    planner_llm_context_builder_node,
    read_context_slice_node,
    rule_based_planner_node,
)
from src.backend.application.agent.case_agent.nodes.execution_dag import (
    call_expert_analysis_node,
    dag_executor_node,
    dispatch_step_node,
    execute_l1_step_node,
    execute_l2_step_node,
    execution_dag_builder_node,
    materialize_expert_task_node,
)
from src.backend.application.agent.case_agent.nodes.fail_closed import fail_closed_node
from src.backend.application.agent.case_agent.nodes.generate_answer import (
    generate_answer_node,
)
from src.backend.application.agent.case_agent.nodes.load_session import (
    load_session_node,
)
from src.backend.application.agent.case_agent.nodes.memory_runtime import (
    build_memory_retrieval_plan_node,
    start_request_memory_prefetch_node,
)
from src.backend.application.agent.case_agent.nodes.perception import (
    build_perceptual_state_node,
    fast_rule_entity_perception_node,
    intent_example_biencoder_node,
    llm_semantic_parser_node,
)
from src.backend.application.agent.case_agent.nodes.persist_result import (
    persist_result_node,
)
from src.backend.application.agent.case_agent.nodes.resolve_answer_policy import (
    resolve_answer_policy_node,
)
from src.backend.application.agent.case_agent.nodes.resolve_answer_style import (
    resolve_answer_style_node,
)
from src.backend.application.agent.case_agent.nodes.validate_answer import (
    validate_answer_node,
)
from src.backend.application.agent.case_agent.nodes.validate_execution_plan import (
    validate_execution_plan_node,
)


def build_case_agent_graph(service: Any):
    """Build the Caser-oriented Case Agent StateGraph."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Case Agent requires langgraph.") from exc

    graph = StateGraph(CaseAgentState)
    graph.add_node(
        "load_session",
        lambda s: service._invoke_node("load_session", lambda state: load_session_node(service, state), s),
    )
    graph.add_node(
        "fast_rule_entity_perception",
        lambda s: service._invoke_node(
            "fast_rule_entity_perception",
            lambda state: fast_rule_entity_perception_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "intent_example_biencoder",
        lambda s: service._invoke_node(
            "intent_example_biencoder",
            lambda state: intent_example_biencoder_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "llm_semantic_parser",
        lambda s: service._invoke_node(
            "llm_semantic_parser",
            lambda state: llm_semantic_parser_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "build_perceptual_state",
        lambda s: service._invoke_node(
            "build_perceptual_state",
            lambda state: build_perceptual_state_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "build_memory_retrieval_plan",
        lambda s: service._invoke_node(
            "build_memory_retrieval_plan",
            lambda state: build_memory_retrieval_plan_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "start_request_memory_prefetch",
        lambda s: service._invoke_node(
            "start_request_memory_prefetch",
            lambda state: start_request_memory_prefetch_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "load_perceptual_state",
        lambda s: service._invoke_node(
            "load_perceptual_state",
            lambda state: load_perceptual_state_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "decision_readiness_checker",
        lambda s: service._invoke_node(
            "decision_readiness_checker",
            lambda state: decision_readiness_checker_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "ask_clarification_plan",
        lambda s: service._invoke_node(
            "ask_clarification_plan",
            lambda state: ask_clarification_plan_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "fail_closed_plan",
        lambda s: service._invoke_node("fail_closed_plan", lambda state: fail_closed_plan_node(service, state), s),
    )
    graph.add_node(
        "decision_precheck",
        lambda s: service._invoke_node(
            "decision_precheck",
            lambda state: decision_precheck_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "rule_based_planner",
        lambda s: service._invoke_node(
            "rule_based_planner",
            lambda state: rule_based_planner_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "read_context_slice",
        lambda s: service._invoke_node(
            "read_context_slice",
            lambda state: read_context_slice_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "heuristic_planner",
        lambda s: service._invoke_node(
            "heuristic_planner",
            lambda state: heuristic_planner_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "planner_llm_context_builder",
        lambda s: service._invoke_node(
            "planner_llm_context_builder",
            lambda state: planner_llm_context_builder_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "llm_planner",
        lambda s: service._invoke_node("llm_planner", lambda state: llm_planner_node(service, state), s),
    )
    graph.add_node(
        "plan_normalizer",
        lambda s: service._invoke_node("plan_normalizer", lambda state: plan_normalizer_node(service, state), s),
    )
    graph.add_node(
        "validate_execution_plan",
        lambda s: service._invoke_node(
            "validate_execution_plan",
            lambda state: validate_execution_plan_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "ask_clarification",
        lambda s: service._invoke_node("ask_clarification", lambda state: ask_clarification_node(service, state), s),
    )
    graph.add_node(
        "call_capabilities",
        lambda s: service._invoke_node("call_capabilities", lambda state: call_capabilities_node(service, state), s),
    )
    graph.add_node(
        "execution_dag_builder",
        lambda s: service._invoke_node(
            "execution_dag_builder",
            lambda state: execution_dag_builder_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "dag_executor",
        lambda s: service._invoke_node("dag_executor", lambda state: dag_executor_node(service, state), s),
    )
    graph.add_node(
        "dispatch_step",
        lambda s: service._invoke_node("dispatch_step", lambda state: dispatch_step_node(service, state), s),
    )
    graph.add_node(
        "execute_l1_step",
        lambda s: service._invoke_node("execute_l1_step", lambda state: execute_l1_step_node(service, state), s),
    )
    graph.add_node(
        "execute_l2_step",
        lambda s: service._invoke_node("execute_l2_step", lambda state: execute_l2_step_node(service, state), s),
    )
    graph.add_node(
        "materialize_expert_task",
        lambda s: service._invoke_node(
            "materialize_expert_task",
            lambda state: materialize_expert_task_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "call_expert_analysis",
        lambda s: service._invoke_node(
            "call_expert_analysis",
            lambda state: call_expert_analysis_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "resolve_answer_policy",
        lambda s: service._invoke_node(
            "resolve_answer_policy",
            lambda state: resolve_answer_policy_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "build_answer_context",
        lambda s: service._invoke_node(
            "build_answer_context",
            lambda state: build_answer_context_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "resolve_answer_style",
        lambda s: service._invoke_node(
            "resolve_answer_style",
            lambda state: resolve_answer_style_node(service, state),
            s,
        ),
    )
    graph.add_node(
        "generate_answer",
        lambda s: service._invoke_node("generate_answer", lambda state: generate_answer_node(service, state), s),
    )
    graph.add_node(
        "validate_answer",
        lambda s: service._invoke_node("validate_answer", lambda state: validate_answer_node(service, state), s),
    )
    graph.add_node(
        "persist_result",
        lambda s: service._invoke_node("persist_result", lambda state: persist_result_node(service, state), s),
    )
    graph.add_node(
        "fail_closed",
        lambda s: service._invoke_node("fail_closed", lambda state: fail_closed_node(service, state), s),
    )

    graph.set_entry_point("load_session")
    graph.add_conditional_edges(
        "load_session",
        route_next,
        {
            "fast_rule_entity_perception": "fast_rule_entity_perception",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "fast_rule_entity_perception",
        route_next,
        {
            "build_perceptual_state": "build_perceptual_state",
            "intent_example_biencoder": "intent_example_biencoder",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "intent_example_biencoder",
        route_next,
        {
            "build_perceptual_state": "build_perceptual_state",
            "llm_semantic_parser": "llm_semantic_parser",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "llm_semantic_parser",
        route_next,
        {"build_perceptual_state": "build_perceptual_state", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "build_perceptual_state",
        route_next,
        {
            "build_memory_retrieval_plan": "build_memory_retrieval_plan",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "build_memory_retrieval_plan",
        route_next,
        {
            "start_request_memory_prefetch": "start_request_memory_prefetch",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "start_request_memory_prefetch",
        route_next,
        {"load_perceptual_state": "load_perceptual_state", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "load_perceptual_state",
        route_next,
        {"decision_readiness_checker": "decision_readiness_checker", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "decision_readiness_checker",
        route_next,
        {
            "ask_clarification_plan": "ask_clarification_plan",
            "fail_closed_plan": "fail_closed_plan",
            "decision_precheck": "decision_precheck",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "ask_clarification_plan",
        route_next,
        {"plan_normalizer": "plan_normalizer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "fail_closed_plan",
        route_next,
        {"fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "decision_precheck",
        route_next,
        {
            "rule_based_planner": "rule_based_planner",
            "read_context_slice": "read_context_slice",
            "planner_llm_context_builder": "planner_llm_context_builder",
            "plan_normalizer": "plan_normalizer",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "rule_based_planner",
        route_next,
        {"plan_normalizer": "plan_normalizer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "read_context_slice",
        route_next,
        {"heuristic_planner": "heuristic_planner", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "heuristic_planner",
        route_next,
        {"plan_normalizer": "plan_normalizer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "planner_llm_context_builder",
        route_next,
        {"llm_planner": "llm_planner", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "llm_planner",
        route_next,
        {"plan_normalizer": "plan_normalizer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "plan_normalizer",
        route_next,
        {
            "validate_execution_plan": "validate_execution_plan",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "validate_execution_plan",
        route_next,
        {
            "ask_clarification": "ask_clarification",
            "execution_dag_builder": "execution_dag_builder",
            "build_answer_context": "build_answer_context",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_edge("ask_clarification", "persist_result")
    graph.add_conditional_edges(
        "execution_dag_builder",
        route_next,
        {"dag_executor": "dag_executor", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "dag_executor",
        route_next,
        {
            "dispatch_step": "dispatch_step",
            "build_answer_context": "build_answer_context",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "dispatch_step",
        route_next,
        {
            "execute_l1_step": "execute_l1_step",
            "execute_l2_step": "execute_l2_step",
            "materialize_expert_task": "materialize_expert_task",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "execute_l1_step",
        route_next,
        {"dag_executor": "dag_executor", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "execute_l2_step",
        route_next,
        {"dag_executor": "dag_executor", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "materialize_expert_task",
        route_next,
        {"call_expert_analysis": "call_expert_analysis", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "call_expert_analysis",
        route_next,
        {"dag_executor": "dag_executor", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "call_capabilities",
        route_next,
        {
            "build_answer_context": "build_answer_context",
            "resolve_answer_policy": "resolve_answer_policy",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_conditional_edges(
        "build_answer_context",
        route_next,
        {"resolve_answer_policy": "resolve_answer_policy", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "resolve_answer_policy",
        route_next,
        {"resolve_answer_style": "resolve_answer_style", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "resolve_answer_style",
        route_next,
        {"generate_answer": "generate_answer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "generate_answer",
        route_next,
        {"validate_answer": "validate_answer", "fail_closed": "fail_closed"},
    )
    graph.add_conditional_edges(
        "validate_answer",
        route_next,
        {
            "generate_answer": "generate_answer",
            "persist_result": "persist_result",
            "fail_closed": "fail_closed",
        },
    )
    graph.add_edge("fail_closed", "persist_result")
    graph.add_edge("persist_result", END)
    return graph.compile()
