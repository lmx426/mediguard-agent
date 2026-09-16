"""Expert Analysis sub-agent state."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.backend.application.agent.expert_agent.schemas import (
    AnswerabilityCheck,
    ExpertAnalysisResult,
    ExpertAnalysisTask,
)


ExpertTaskType = Literal[
    "policy_analysis",
    "drug_rationality",
    "precedent_matching",
    "evidence_deep_dive",
    "document_interpretation",
]


class ExpertAgentState(TypedDict, total=False):
    """State flowing through the Expert Analysis compiled subgraph."""

    task_id: str
    parent_run_id: str
    case_id: str
    expert_task_type: ExpertTaskType
    task: ExpertAnalysisTask
    input_factors: dict[str, Any]
    runtime: dict[str, Any]
    task_context: dict[str, Any]
    retrieval_plan: dict[str, Any]
    policy_request: dict[str, Any]
    policy_observation: dict[str, Any]
    policy_observations: list[dict[str, Any]]
    mcp_safe_summary: dict[str, Any]
    mcp_safe_summaries: list[dict[str, Any]]
    hard_gate: dict[str, Any]
    evidence_normalizer: dict[str, Any]
    answer_mode: str
    information_needs: list[str]
    answer_requirements: list[dict[str, Any]]
    sentence_windows: list[dict[str, Any]]
    candidate_windows: list[dict[str, Any]]
    need_evidence_resolution: dict[str, Any]
    verified_need_facts: list[dict[str, Any]]
    need_evidence_matches: list[dict[str, Any]]
    need_answers: list[dict[str, Any]]
    need_first_answer: dict[str, Any]
    coverage_result: dict[str, Any]
    llm_semantic_coverage: dict[str, Any]
    generation_policy: dict[str, str]
    answerability_check: AnswerabilityCheck
    case_context_observations: list[dict[str, Any]]
    case_facts_used: list[dict[str, Any]]
    adopted_policy_evidence: list[dict[str, Any]]
    result_model: ExpertAnalysisResult
    status: str
    message: str
    result: dict[str, Any]
    source_refs: list[str]
    model_call_count: int
    tool_call_count: int
    retrieval_round_count: int
    rewrite_count: int
    next_action: str
    error_code: str
    error_message: str
