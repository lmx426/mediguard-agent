"""Safe API contracts for case-memory governance and preview."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...domain.case_memory.entities import (
    MemoryHintPack,
    MemoryLevel,
    MemoryPreview,
    MemoryPresentationItem,
    MemorySafeDetail,
    MemoryScope,
    MemoryType,
)


class MemoryActionInput(BaseModel):
    action: Literal[
        "confirm",
        "auto_activate",
        "reject",
        "snooze",
        "archive",
        "revoke",
        "restore",
        "supersede",
        "mark_conflict",
        "tombstone",
    ]
    snoozed_until: datetime | None = None
    related_memory_id: str | None = Field(default=None, max_length=80)


class MemoryCaptureInput(BaseModel):
    memory_type: MemoryType
    memory_level: MemoryLevel
    payload: dict[str, Any]
    scope: MemoryScope = Field(default_factory=MemoryScope)
    source_event_ids: list[str] = Field(default_factory=list)
    allowed_consumers: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    feature: str | None = None


class MemoryPreviewInput(BaseModel):
    consumer: str
    task_context: dict[str, Any] = Field(default_factory=dict)


class MemoryBatchRecallInput(BaseModel):
    request_id: str
    consumer: str
    memory_type: MemoryType
    scope: MemoryScope = Field(default_factory=MemoryScope)
    task_context: dict[str, Any] = Field(default_factory=dict)


class PersonalMemoryPreferenceInput(BaseModel):
    enabled: bool


class PersonalMemoryPreferenceResponse(BaseModel):
    enabled: bool


__all__ = [
    "MemoryActionInput",
    "MemoryBatchRecallInput",
    "MemoryCaptureInput",
    "MemoryHintPack",
    "MemoryLevel",
    "MemoryPreview",
    "MemoryPreviewInput",
    "MemoryPresentationItem",
    "MemorySafeDetail",
    "MemoryType",
    "PersonalMemoryPreferenceInput",
    "PersonalMemoryPreferenceResponse",
]
