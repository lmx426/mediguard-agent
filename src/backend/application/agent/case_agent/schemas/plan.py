"""Caser semantic planning schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CaserPlanLayer = Literal["L1", "L2", "L3"]
CaserPlannerIntent = Literal[
    "general_help",
    "case_task",
    "expert_task",
    "clarification_required",
]
CaserAnswerGranularity = Literal[
    "single_field",
    "field_group",
    "list",
    "detail",
    "overview",
    "analysis",
]


class CaserQuerySemantics(BaseModel):
    """Planner's understanding of the user question."""

    intent: CaserPlannerIntent = "general_help"
    user_goal: str = Field(default="", max_length=500)
    granularity: CaserAnswerGranularity = "overview"
    information_needs: list[str] = Field(default_factory=list, max_length=12)
    missing_slots: list[str] = Field(default_factory=list, max_length=8)


class CaserCapabilityPlanItem(BaseModel):
    """One backend capability requested by the planner."""

    step: int = Field(default=1, ge=1, le=8)
    layer: CaserPlanLayer
    capability: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str | int] = Field(default_factory=list, max_length=8)
    covers: list[str] = Field(default_factory=list, max_length=12)
    reason: str = Field(default="", max_length=300)


class CaserBusinessSemanticPlan(BaseModel):
    """Strict JSON plan emitted by the model planner."""

    query_semantics: CaserQuerySemantics = Field(default_factory=CaserQuerySemantics)
    execution_plan: list[CaserCapabilityPlanItem] = Field(default_factory=list, max_length=8)
