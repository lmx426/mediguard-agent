"""Governed long-term memory entities.

The domain intentionally uses maturity (L0-L3) as the only hierarchy. The
memory type is a consumer-purpose label, not a semantic/entity/procedural
classification axis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryLevel(StrEnum):
    L0 = "L0_source_event"
    L1 = "L1_atomic_memory"
    L2 = "L2_scenario_memory"
    L3 = "L3_stable_profile_or_playbook"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SHADOW = "shadow"
    REJECTED = "rejected"
    SNOOZED = "snoozed"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    TOMBSTONED = "tombstoned"
    CONFLICT_REVIEW = "conflict_review"


class MemoryType(StrEnum):
    INTENT_ROUTE = "intent_route_hint"
    POLICY_SEARCH = "policy_search_hint"
    FAILURE = "failure_hint"
    ANSWER_STYLE = "answer_style_hint"
    DECISION_PLAN = "decision_plan_hint"


class MemoryScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_type: Literal["auditor", "team", "department", "global"] = "auditor"
    scope_id: str = "global"


class L0SourceEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    source_run_id: str | None = None
    source_event_id: str
    node: str
    validation_codes: list[str] = Field(default_factory=list)
    source_ref_ids: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)


class L1AtomicMemoryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    structured_content: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""
    source_refs: list[str] = Field(default_factory=list)
    derived_from_l0_refs: list[str] = Field(default_factory=list)


class L2ScenarioMemoryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_summary: str
    applicable_conditions: list[str] = Field(default_factory=list)
    recommended_action: dict[str, Any] = Field(default_factory=dict)
    exception_rules: list[str] = Field(default_factory=list)
    supporting_l1_refs: list[str] = Field(default_factory=list)


class L3StableProfileOrPlaybookPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_or_playbook_summary: str
    stable_preferences: dict[str, Any] = Field(default_factory=dict)
    standard_steps: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    supporting_l2_refs: list[str] = Field(default_factory=list)
    override_rules: list[str] = Field(default_factory=list)


PayloadModel = (
    L0SourceEventPayload
    | L1AtomicMemoryPayload
    | L2ScenarioMemoryPayload
    | L3StableProfileOrPlaybookPayload
)


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_type: MemoryType
    memory_level: MemoryLevel
    scope: MemoryScope
    status: MemoryStatus = MemoryStatus.CANDIDATE
    payload: dict[str, Any] = Field(default_factory=dict)
    allowed_consumers: list[str] = Field(default_factory=list)
    consumer_view_policy_id: str = "default"
    admission_policy: str = "human_review"
    admission_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    shadow_observation_count: int = Field(default=0, ge=0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified_at: datetime | None = None
    last_used_at: datetime | None = None
    review_after: datetime | None = None
    expires_at: datetime | None = None
    snoozed_until: datetime | None = None
    memo_memory_id: str | None = None
    projection_sync_status: str = "pending"
    supersedes_id: str | None = None
    conflict_with_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_level_payload(self) -> "MemoryRecord":
        expected = {
            MemoryLevel.L0: L0SourceEventPayload,
            MemoryLevel.L1: L1AtomicMemoryPayload,
            MemoryLevel.L2: L2ScenarioMemoryPayload,
            MemoryLevel.L3: L3StableProfileOrPlaybookPayload,
        }[self.memory_level]
        parsed = expected.model_validate(self.payload)
        self.payload = parsed.model_dump(mode="json")
        if self.memory_level == MemoryLevel.L0 and self.status not in {
            MemoryStatus.CANDIDATE,
            MemoryStatus.ARCHIVED,
        }:
            raise ValueError("L0 source events cannot become active recall memories")
        if any(term in str(self.payload) for term in ("RES", "身份证", "姓名", "电话", "住址")):
            raise ValueError("memory payload contains forbidden sensitive content")
        return self

    @property
    def summary(self) -> str:
        return str(
            self.payload.get("summary")
            or self.payload.get("scenario_summary")
            or self.payload.get("profile_or_playbook_summary")
            or self.payload.get("event_type")
            or ""
        )


class MemoryAdmissionDecision(BaseModel):
    policy: Literal["auto_active", "shadow", "human_review"]
    status: MemoryStatus
    reason: str
    importance_score: float = Field(ge=0.0, le=1.0)


class MemoryRecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    consumer: str
    memory_type: MemoryType
    scope: MemoryScope
    scope_fallbacks: list[MemoryScope] = Field(default_factory=list, max_length=4)
    task_context: dict[str, Any] = Field(default_factory=dict)
    memory_level_override: MemoryLevel | None = None
    max_items: int = Field(default=3, ge=1, le=10)


class MemoryVectorProjection(BaseModel):
    """Safe, deterministic text projected to the vector backend."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    embedding_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    projection_version: str = "v2"


class MemoryQueryProfile(BaseModel):
    """Type-specific retrieval query derived from transient task context."""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType
    query_text: str = ""
    match_profile: dict[str, Any] = Field(default_factory=dict)
    exact_filters: dict[str, list[str]] = Field(default_factory=dict)
    graph_anchor_keys: list[str] = Field(default_factory=list)


class ConsumerViewPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    consumer: str
    memory_type: MemoryType
    memory_level: MemoryLevel
    control_fields: list[str] = Field(default_factory=list)
    prompt_fields: list[str] = Field(default_factory=list)
    tool_param_fields: list[str] = Field(default_factory=list)
    trace_fields: list[str] = Field(default_factory=list)
    max_items: int = Field(default=3, ge=1, le=10)

    @property
    def allowed_fields(self) -> set[str]:
        return set(self.control_fields + self.prompt_fields + self.tool_param_fields + self.trace_fields)

    def trim(self, payload: dict[str, Any], fields: list[str] | None = None) -> dict[str, Any]:
        """Return only registered fields, supporting dotted nested paths."""

        trimmed: dict[str, Any] = {}
        for path in fields if fields is not None else self.allowed_fields:
            current: Any = payload
            parts = path.split(".")
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is None:
                continue
            target = trimmed
            for part in parts[:-1]:
                existing = target.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    target[part] = existing
                target = existing
            target[parts[-1]] = current
        return trimmed


class MemoryHintPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    consumer: str
    memory_type: MemoryType
    control_hints: list[dict[str, Any]] = Field(default_factory=list)
    prompt_contexts: list[dict[str, Any]] = Field(default_factory=list)
    tool_param_hints: list[dict[str, Any]] = Field(default_factory=list)
    trace_refs: list[dict[str, Any]] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    retrieved_levels: list[MemoryLevel] = Field(default_factory=list)


class MemoryPresentationItem(BaseModel):
    memory_id: str
    memory_type: MemoryType
    memory_level: MemoryLevel
    status: MemoryStatus
    summary: str
    source: list[str] = Field(default_factory=list)
    created_at: datetime
    last_verified_at: datetime | None = None
    confidence: float
    observation_count: int = Field(default=1, ge=1)
    scope: MemoryScope
    allowed_consumers: list[str]
    freshness_score: float
    importance_score: float
    projection_sync_status: str
    creation_mode: Literal["system_extracted", "system_consolidated"]
    activation_mode: Literal["human_confirmed", "auto_active", "pending_review", "shadow"]
    activated_at: datetime | None = None
    activated_by: str | None = None
    archived_at: datetime | None = None
    archived_by: str | None = None


class MemorySafeDetail(MemoryPresentationItem):
    structured_action: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""
    source_refs: list[str] = Field(default_factory=list)
    status_events: list[dict[str, Any]] = Field(default_factory=list)


class MemoryPreview(BaseModel):
    memory_id: str
    consumer: str
    node: str | None = None
    available: bool
    memory_level: MemoryLevel | None = None
    allowed_fields: list[str] = Field(default_factory=list)
    delivery_locations: list[str] = Field(default_factory=list)
    hint_pack: MemoryHintPack | None = None
    reason: str | None = None
