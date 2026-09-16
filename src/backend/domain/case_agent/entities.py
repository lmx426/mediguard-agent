"""Case Agent domain entities.

The Case Agent is a lightweight, case-bound assistant. It may persist
session-level working memory and trace artifacts, but it must not mutate
formal case facts, rules, risk signals, Evidence Agent analyses, workflow, or
final human review decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CaseAgentRunStatus = Literal[
    "created",
    "running",
    "waiting_for_user",
    "resuming",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "degraded",
]
CaseAgentMessageRole = Literal["user", "assistant"]
CaseAgentSessionStatus = Literal["active", "archived"]
CaseAgentDisplayMode = Literal["plain", "grounded", "unavailable", "error"]
CaseAgentAttentionType = Literal["none", "new_result", "needs_input", "degraded", "failed"]
CaseAgentTaskStatus = Literal["idle", "active", "waiting_user", "completed", "error"]
CaseAgentTurnRelation = Literal[
    "new_task",
    "supplement",
    "correction",
    "clarification_response",
    "answer_rewrite",
    "continuation",
]
CaseAgentIntent = Literal[
    "general_help",
    "case_basic_info_query",
    "claimant_profile_query",
    "material_overview_query",
    "medical_materials_query",
    "prescription_materials_query",
    "settlement_materials_query",
    "statistics_report_query",
    "risk_score_query",
    "evidence_package_query",
    "case_judgement_query",
    "verification_items_query",
    "rule_verification_query",
    "policy_expert_query",
    "drug_clinical_expert_query",
    "precedent_expert_query",
    "complex_material_expert_query",
    "case_snapshot_query",
    "risk_rule_query",
    "business_material_catalog_query",
    "business_material_detail_query",
    "case_review_analysis_query",
    "domain_expert_query",
    "unknown",
]


class CaseAgentSourceDetailField(BaseModel):
    """One displayable field in a cited source detail panel."""

    label: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=500)


class CaseAgentSourceDetail(BaseModel):
    """Generic detail payload for a source cited by Case Agent."""

    fields: list[CaseAgentSourceDetailField] = Field(default_factory=list, max_length=12)


class CaseAgentSource(BaseModel):
    """A case-scoped source that answer blocks may cite."""

    source_ref: str = Field(min_length=1, max_length=200)
    source_type: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    version: str | None = Field(default=None, max_length=80)
    detail: CaseAgentSourceDetail = Field(default_factory=CaseAgentSourceDetail)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseAgentContentBlock(BaseModel):
    """One visible answer text block with optional inline source references."""

    text: str = Field(min_length=1, max_length=1200)
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class CaseAgentClaim(BaseModel):
    """One sentence-level factual claim in a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=80)
    need_id: str = Field(default="", max_length=80)
    need_text: str = Field(default="", max_length=240)
    text: str = Field(min_length=1, max_length=600)
    fact_refs: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    citation_ids: list[str] = Field(default_factory=list, max_length=8)
    support_status: str = Field(default="supported", max_length=40)


class CaseAgentCitation(BaseModel):
    """Citation marker rendered from answer markdown to a concrete source."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1, max_length=80)
    label: int = Field(ge=1, le=99)
    claim_id: str | None = Field(default=None, max_length=80)
    fact_refs: list[str] = Field(default_factory=list, max_length=12)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class CaseAgentAnswer(BaseModel):
    """Lightweight structured answer rendered by the frontend."""

    model_config = ConfigDict(extra="forbid")

    display_mode: CaseAgentDisplayMode = "plain"
    content_blocks: list[CaseAgentContentBlock] = Field(default_factory=list, min_length=1, max_length=8)
    sources: list[CaseAgentSource] = Field(default_factory=list, max_length=20)
    answer_markdown: str | None = Field(default=None, max_length=4000)
    claims: list[CaseAgentClaim] = Field(default_factory=list, max_length=30)
    citations: list[CaseAgentCitation] = Field(default_factory=list, max_length=30)
    fallback_notice: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        """Return text content for message previews and copy actions."""

        if self.answer_markdown:
            return self.answer_markdown
        return "\n".join(block.text for block in self.content_blocks)


class CaseAgentMessage(BaseModel):
    """One persisted Case Agent message."""

    message_id: str
    session_id: str
    role: CaseAgentMessageRole
    content: str
    active_stage: str | None = None
    answer_payload: CaseAgentAnswer | None = None
    source_refs: list[str] = Field(default_factory=list)
    created_at: datetime


class CaseAgentCompletedStep(BaseModel):
    """One bounded, reusable conclusion from a completed conversation turn."""

    message_id: str = Field(default="", max_length=80)
    intent: str = Field(default="unknown", max_length=80)
    subtask: str = Field(default="", max_length=120)
    answer_ref: str = Field(default="", max_length=120)
    capabilities: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(default_factory=list, max_length=12)
    completed_at: datetime | None = None


class CaseAgentTaskState(BaseModel):
    """Validated cross-turn task state without raw prompts or tool payloads."""

    schema_version: str = Field(default="case-agent-task-state-v1", max_length=40)
    state_version: int = Field(default=0, ge=0)
    task_id: str = Field(default="", max_length=80)
    status: CaseAgentTaskStatus = "idle"
    core_intent: str = Field(default="unknown", max_length=80)
    goal: str = Field(default="", max_length=300)
    current_subtask: str = Field(default="", max_length=120)
    workflow_stage: str | None = Field(default=None, max_length=80)
    confirmed_slots: dict[str, Any] = Field(default_factory=dict)
    pending_clarification: dict[str, Any] = Field(default_factory=dict)
    completed_steps: list[CaseAgentCompletedStep] = Field(default_factory=list, max_length=8)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    last_turn_relation: CaseAgentTurnRelation = "new_task"
    last_user_message_id: str | None = Field(default=None, max_length=80)
    last_turn_created_at: datetime | None = None
    updated_at: datetime | None = None


class CaseAgentRunSummary(BaseModel):
    """Latest run state embedded in the lightweight session list."""

    run_id: str
    status: CaseAgentRunStatus
    effective_status: str
    current_node: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None


class CaseAgentSession(BaseModel):
    """Session-level working memory, scoped by user + case + session."""

    session_id: str
    case_id: str
    actor_id: str
    title: str
    status: CaseAgentSessionStatus = "active"
    session_summary: str = ""
    current_topic: str | None = None
    pending_tool: str | None = None
    referenced_source_refs: list[str] = Field(default_factory=list)
    task_state: CaseAgentTaskState = Field(default_factory=CaseAgentTaskState, exclude=True)
    latest_run: CaseAgentRunSummary | None = None
    attention_type: CaseAgentAttentionType = "none"
    has_unread_activity: bool = False
    last_read_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[CaseAgentMessage] = Field(default_factory=list)


class CaseAgentRun(BaseModel):
    """One assistant turn execution."""

    run_id: str
    session_id: str
    case_id: str
    actor_id: str
    user_message_id: str
    assistant_message_id: str | None = None
    parent_run_id: str | None = None
    resumed_by_run_id: str | None = None
    status: CaseAgentRunStatus
    current_node: str
    error_code: str | None = None
    error_message: str | None = None
    degraded_reason: str | None = None
    pending_clarification: dict[str, Any] = Field(default_factory=dict)
    resume_context: dict[str, Any] = Field(default_factory=dict)
    model_name: str
    model_call_count: int = 0
    tool_call_count: int = 0
    timeout_at: datetime | None = None
    started_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancel_reason: str | None = None
    client_request_id: str | None = None
    case_context_fingerprint: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class CaseAgentEvent(BaseModel):
    """Replayable business event for SSE."""

    run_id: str
    sequence: int
    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CaseAgentToolCall(BaseModel):
    """Auditable tool-call trace."""

    run_id: str
    sequence: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result_summary: dict[str, Any] | None = None
    source_refs: list[str] = Field(default_factory=list)
    status: Literal["success", "failed", "unavailable"]
    error_code: str | None = None
    latency_ms: int = 0
    created_at: datetime


class CaseAgentSendMessageInput(BaseModel):
    """User message payload."""

    content: str = Field(min_length=1, max_length=2000)
    active_stage: str | None = Field(default=None, max_length=80)
    client_request_id: str | None = Field(default=None, min_length=8, max_length=120)


class CaseAgentMarkReadInput(BaseModel):
    """Advance one session's read watermark through a specific terminal run."""

    through_run_id: str = Field(min_length=1, max_length=80)


class CaseAgentCreateSessionInput(BaseModel):
    """Session creation payload."""

    title: str | None = Field(default=None, max_length=80)


class CaseAgentRenameSessionInput(BaseModel):
    """Session title update payload."""

    title: str = Field(min_length=1, max_length=80)


class CaseAgentAdoptNoteInput(BaseModel):
    """Human-confirmed adoption request."""

    content: str = Field(min_length=1, max_length=2000)
    source_refs: list[str] = Field(default_factory=list, max_length=20)
