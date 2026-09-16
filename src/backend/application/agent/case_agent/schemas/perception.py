"""Caser perception and decision context schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SemanticSource = Literal[
    "rule_precheck",
    "intent_example_biencoder",
    "llm_semantic_parser",
    "fallback",
]
ReadinessStatus = Literal[
    "ready_to_plan",
    "soft_plan_allowed",
    "need_clarification",
    "fail_closed",
]
LayerHint = Literal["L1", "L2", "L3", "none"]


class InputEnvelope(BaseModel):
    """Normalized user input passed into the perception layer."""

    raw_text: str = Field(default="", max_length=1200)
    normalized_text: str = Field(default="", max_length=1200)
    channel: str = "chat"
    attachments_ref: list[str] = Field(default_factory=list, max_length=12)
    event_ref: str = ""


class EntityFrame(BaseModel):
    """Early rule-extracted entities and slot candidates."""

    explicit_entities: dict[str, Any] = Field(default_factory=dict)
    slot_candidates: dict[str, Any] = Field(default_factory=dict)
    referenced_source_refs: list[str] = Field(default_factory=list, max_length=20)


class SemanticFrame(BaseModel):
    """Merged semantic frame before context assembly."""

    source: SemanticSource = "fallback"
    utterance_type: str = "question"
    business_intent: str = "general_help"
    answer_shape: str = "overview"
    evidence_need: str = "none"
    focus_candidate: dict[str, Any] = Field(default_factory=dict)
    slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticParseResult(BaseModel):
    """Pure semantic parser output. It must not contain an execution plan."""

    intent: str = Field(default="general_help", max_length=80)
    action: str = Field(default="general_help", max_length=80)
    answer_shape: str = Field(default="overview", max_length=40)
    target_layer_hint: LayerHint = "none"
    target_objects: list[str] = Field(default_factory=list, max_length=8)
    information_needs: list[str] = Field(default_factory=list, max_length=12)
    rewritten_query: str = Field(default="", max_length=500)
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    need_clarification: bool = False
    safety_flags: list[str] = Field(default_factory=list, max_length=8)


class MinimalPlanningContext(BaseModel):
    """Small context used by deterministic planning."""

    question: str = Field(default="", max_length=1200)
    case_id: str = ""
    session_id: str = ""
    actor_id: str = ""
    governance_min: dict[str, Any] = Field(default_factory=dict)
    current_topic: str = Field(default="", max_length=300)
    referenced_refs: list[str] = Field(default_factory=list, max_length=20)


class DecisionContext(BaseModel):
    """Perception GSSC output used as the planning context base."""

    context_ref: str
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    governance_context: dict[str, Any] = Field(default_factory=dict)
    domain_artifact_context: dict[str, Any] = Field(default_factory=dict)
    working_memory_context: dict[str, Any] = Field(default_factory=dict)
    context_summary: dict[str, Any] = Field(default_factory=dict)


class PerceptualState(BaseModel):
    """Single handoff object from perception to decision."""

    semantic: dict[str, Any] = Field(default_factory=dict)
    slots_focus: dict[str, Any] = Field(default_factory=dict)
    context_refs: dict[str, Any] = Field(default_factory=dict)
    decision_context_ref: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    readiness_signal: dict[str, Any] = Field(default_factory=dict)
    minimal_context: dict[str, Any] = Field(default_factory=dict)


class PlannerLLMContext(BaseModel):
    """Context projected from DecisionContext for LLM planning."""

    task_context: dict[str, Any] = Field(default_factory=dict)
    scope_context: dict[str, Any] = Field(default_factory=dict)
    memory_context: dict[str, Any] = Field(default_factory=dict)
    artifact_context: dict[str, Any] = Field(default_factory=dict)
    capability_context: dict[str, Any] = Field(default_factory=dict)
    governance_context: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
