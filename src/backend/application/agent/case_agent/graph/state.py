"""Case Agent LangGraph state."""

from __future__ import annotations

from typing import Any, TypedDict

from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentMessage,
    CaseAgentRun,
    CaseAgentSession,
    CaseAgentSource,
)


class CaseAgentState(TypedDict, total=False):
    """Runtime state passed between Case Agent StateGraph nodes."""

    run_id: str
    actor_id: str
    run: CaseAgentRun
    session: CaseAgentSession
    task_state: dict[str, Any]
    turn_relation: dict[str, Any]
    recent_messages: list[CaseAgentMessage]
    user_message: CaseAgentMessage
    messages: list[dict[str, Any]]
    final_response: Any
    capability_results: list[tuple[str, Any]]
    available_sources: list[CaseAgentSource]
    input_envelope: dict[str, Any]
    entity_frame: dict[str, Any]
    semantic_intent_signal: dict[str, Any]
    semantic_hint_pack: dict[str, Any]
    light_semantic_frame: dict[str, Any]
    biencoder_frame: dict[str, Any]
    llm_semantic_result: dict[str, Any]
    deep_semantic_frame: dict[str, Any]
    semantic_frame: dict[str, Any]
    context_need: dict[str, Any]
    minimal_planning_context: dict[str, Any]
    decision_context: dict[str, Any]
    decision_context_ref: str
    perception_contract: dict[str, Any]
    perceptual_state: dict[str, Any]
    perception_metrics: dict[str, Any]
    model_call_metrics: list[dict[str, Any]]
    perception_started_epoch_ms: int
    decision_context_slice: dict[str, Any]
    planner_llm_context: dict[str, Any]
    rule_candidate_plan: dict[str, Any]
    planning_output: dict[str, Any]
    plan_optimization: dict[str, Any]
    shortcut: dict[str, Any]
    resume_context: dict[str, Any]
    session_memory_context: dict[str, Any]
    memory_retrieval_plan: dict[str, Any]
    request_memory_status: dict[str, Any]
    memory_consumption_status: dict[str, Any]
    memory_hint_packs: dict[str, Any]
    memory_fallbacks: list[dict[str, Any]]
    failure_recovery_hint: dict[str, Any]
    intent_memory_shadow: dict[str, Any]
    context_digest: dict[str, Any]
    digest_source_refs: list[str]
    intent: str
    intent_confidence: float
    slots: dict[str, Any]
    missing_slots: list[str]
    query_semantics: dict[str, Any]
    execution_plan: list[dict[str, Any]]
    execution_dag: dict[str, Any]
    current_dag_step: dict[str, Any]
    artifact_store: dict[str, Any]
    artifact_refs: list[str]
    current_expert_task: Any
    section_status: list[dict[str, Any]]
    planning_error: str
    context_plan: dict[str, Any]
    answer_policy: dict[str, Any]
    answer_context: dict[str, Any]
    answer_style_policy: dict[str, Any]
    answer_strategy: str
    reuse_answer_ref: str
    reuse_source_refs: list[str]
    answer_rewrite_mode: str
    answer: CaseAgentAnswer
    assistant_message_id: str
    model_call_count: int
    tool_call_count: int
    validation_retry_count: int
    validation_error: str
    validation_repair_succeeded: bool
    validation_recovery_mode: str
    next_action: str
    completion_status: str
    pending_clarification: dict[str, Any]
    error: Exception
    error_code: str
    error_message: str
