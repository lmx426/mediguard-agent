"""Structured contracts for the Expert Analysis sub-agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


ExpertStatus = Literal["ok", "partial", "insufficient", "unavailable", "failed"]


class ExpertAnalysisBudget(BaseModel):
    """Runtime budget granted by the parent Caser run."""

    max_model_calls: int = Field(default=6, ge=1, le=8)
    max_tool_calls: int = Field(default=6, ge=1, le=12)
    max_retrieval_rounds: int = Field(default=3, ge=1, le=5)
    max_llm_rewrite_calls: int = Field(default=1, ge=0, le=2)
    timeout_ms: int = Field(default=20000, ge=1000, le=120000)


class ExpertAnalysisConstraints(BaseModel):
    """Non-negotiable boundaries inherited from Caser governance."""

    read_only: bool = True
    no_final_audit_decision: bool = True
    must_cite_evidence: bool = True
    no_material_gap_diff: bool = True


class ExpertAnalysisTask(BaseModel):
    """Minimal task package passed from Caser to the L3 sub-agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=120)
    parent_run_id: str = Field(min_length=1, max_length=120)
    case_id: str = Field(min_length=1, max_length=120)
    capability: str = Field(default="ask_policy_expert", max_length=80)
    expert_task_type: str = Field(default="policy_analysis", max_length=80)
    goal: str = Field(default="", max_length=500)
    user_question: str = Field(min_length=1, max_length=2000)
    fact_bundle: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    context_refs: list[str] = Field(default_factory=list, max_length=40)
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "policy.search_text",
            "policy.search_version",
            "case_context.query",
        ],
        max_length=12,
    )
    constraints: ExpertAnalysisConstraints = Field(default_factory=ExpertAnalysisConstraints)
    budget: ExpertAnalysisBudget = Field(default_factory=ExpertAnalysisBudget)
    filters: dict[str, Any] = Field(default_factory=dict)
    memory_guidance: dict[str, Any] = Field(default_factory=dict)


class PolicyEvidence(BaseModel):
    """Policy evidence finally adopted by Expert Analysis."""

    evidence_ref: str = Field(min_length=1, max_length=220)
    source_ref: str = Field(min_length=1, max_length=220)
    title: str = Field(default="", max_length=240)
    excerpt: str = Field(default="", max_length=1200)
    jurisdiction: str | None = Field(default=None, max_length=120)
    policy_domain: str | None = Field(default=None, max_length=120)
    content_type: str | None = Field(default=None, max_length=80)
    source_url: str | None = Field(default=None, max_length=500)
    version: str | None = Field(default=None, max_length=120)
    used_for: str = Field(default="policy_answer", max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseFactUsed(BaseModel):
    """Case fact used only as policy-query context."""

    label: str = Field(min_length=1, max_length=120)
    value: Any
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    used_for: str = Field(default="policy_query_context", max_length=120)


class ExpertAuditSuggestion(BaseModel):
    """Policy-grounded audit workflow suggestion, not a final decision."""

    type: str = Field(default="policy_based_verification", max_length=120)
    text: str = Field(min_length=1, max_length=800)
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class AnswerabilityCheck(BaseModel):
    """LLM coverage judgment inside the L3 ReAct loop."""

    model_config = ConfigDict(populate_by_name=True)

    answerability: Literal["answerable", "partial", "insufficient"] = "insufficient"
    covered_needs: list[str] = Field(
        default_factory=list,
        max_length=12,
        validation_alias=AliasChoices("covered_needs", "covered_slots"),
    )
    missing_needs: list[str] = Field(
        default_factory=list,
        max_length=12,
        validation_alias=AliasChoices("missing_needs", "missing_slots"),
    )
    usable_evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    next_action: Literal["synthesize", "rewrite", "insufficient"] = "insufficient"
    reason_code: str = Field(default="unknown", max_length=120)
    reason: str = Field(default="", max_length=500)

    @property
    def covered_slots(self) -> list[str]:
        """Read-only compatibility for historical in-process callers."""

        return self.covered_needs

    @property
    def missing_slots(self) -> list[str]:
        """Read-only compatibility for historical in-process callers."""

        return self.missing_needs


class ExpertAnalysisResult(BaseModel):
    """Structured result returned to Caser; never directly final user output."""

    status: ExpertStatus
    answer_mode: str = Field(default="process_rule", max_length=80)
    information_needs: list[str] = Field(default_factory=list, max_length=12)
    pipeline_version: str = Field(default="information_need_v2", max_length=80)
    expert_answer: str = Field(default="", max_length=2000)
    answer_markdown: str = Field(default="", max_length=4000)
    answer_requirements: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    evidence_matches: list[dict[str, Any]] = Field(default_factory=list, max_length=40)
    extracted_facts: list[dict[str, Any]] = Field(default_factory=list, max_length=40)
    need_answers: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    claims: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    claim_plan: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    citations: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    case_facts_used: list[CaseFactUsed] = Field(default_factory=list, max_length=20)
    policy_evidence: list[PolicyEvidence] = Field(default_factory=list, max_length=20)
    material_gaps: list[dict[str, Any]] = Field(default_factory=list, max_length=0)
    audit_suggestions: list[ExpertAuditSuggestion] = Field(default_factory=list, max_length=12)
    answerability_summary: AnswerabilityCheck | None = None
    coverage_result: dict[str, Any] = Field(default_factory=dict)
    generation_policy: dict[str, str] = Field(default_factory=dict)
    filter_diagnostics: dict[str, Any] = Field(default_factory=dict)
    limits: str = Field(
        default=(
            "当前 Policy Expert 仅提供政策口径和可引用依据，不基于仿真材料形成"
            "当前案件材料缺口差集，不形成拒付、处罚、欺诈或最终审核结论。"
        ),
        max_length=800,
    )


class PolicySearchRequest(BaseModel):
    """Request sent through the policy_tool_adapter to Policy RAG MCP."""

    question: str = Field(min_length=1, max_length=1200)
    filters: dict[str, Any] = Field(default_factory=dict)
    recall_filters: dict[str, Any] | None = None
    filter_strategy: Literal["single", "dual_rrf"] = "single"
    adaptive_top_k: bool = False
    allow_broad_filters: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    fetch_k: int = Field(default=40, ge=1, le=200)
    rerank: bool = False
