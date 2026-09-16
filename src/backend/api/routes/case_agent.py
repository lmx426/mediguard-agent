"""Case Agent routes."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..dependencies import (
    AppContainer,
    get_case_agent,
    get_container,
    get_current_user,
)
from ...application.agent.case_agent.service import CaseAgentService
from ...application.showcase import (
    ShowcaseCaseAgentProjection,
    ShowcaseCaseAgentService,
)
from ...domain.audit.review.entities import AuthenticatedUser
from ...domain.audit.review.workflow_projector import AuditNote
from ...domain.case_agent.entities import (
    CaseAgentAdoptNoteInput,
    CaseAgentCreateSessionInput,
    CaseAgentMarkReadInput,
    CaseAgentRenameSessionInput,
    CaseAgentRun,
    CaseAgentSendMessageInput,
    CaseAgentSession,
)
from ...domain.caser_context.entities import CaserContextSection

router = APIRouter(tags=["case-agent"])
TERMINAL_STATUSES = {
    "waiting_for_user",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "degraded",
}


class CaserContextRebuildInput(BaseModel):
    """Manual Caser context rebuild request."""

    section_keys: list[str] | None = Field(default=None, max_length=12)


@router.get("/case-agent/status")
def case_agent_status(container: AppContainer = Depends(get_container)) -> dict:
    """Return Case Agent readiness without leaking secrets."""

    return container.case_agent_status


@router.get(
    "/cases/{case_id}/case-agent/showcase",
    response_model=ShowcaseCaseAgentProjection,
)
def get_showcase_case_agent(
    case_id: str,
    _actor: AuthenticatedUser = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> ShowcaseCaseAgentProjection:
    """Return the labelled, read-only Case Agent interview projection."""

    if not container.settings.showcase_mode:
        raise HTTPException(status_code=404, detail="Showcase Case Agent 未启用")
    try:
        return ShowcaseCaseAgentService(
            container.cases,
            container.evidence,
        ).get_projection(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"案件 {case_id} 不存在") from exc


@router.post(
    "/cases/{case_id}/caser/context/rebuild",
    response_model=list[CaserContextSection],
)
def rebuild_caser_context(
    case_id: str,
    body: CaserContextRebuildInput | None = None,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> list[CaserContextSection]:
    """Manually rebuild Caser View Store sections for one case."""

    return service.rebuild_caser_context(
        case_id,
        actor,
        section_keys=body.section_keys if body is not None else None,
    )


@router.get(
    "/cases/{case_id}/caser/context/sections",
    response_model=list[CaserContextSection],
)
def list_caser_context_sections(
    case_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> list[CaserContextSection]:
    """List all Caser section statuses and payloads for one case."""

    return service.list_caser_context_sections(case_id, actor)


@router.get(
    "/cases/{case_id}/caser/context/sections/{section_key}",
    response_model=CaserContextSection,
)
def get_caser_context_section(
    case_id: str,
    section_key: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaserContextSection:
    """Read one Caser section for one case."""

    return service.get_caser_context_section(case_id, section_key, actor)


@router.get(
    "/cases/{case_id}/case-agent/sessions",
    response_model=list[CaseAgentSession],
)
def list_case_agent_sessions(
    case_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> list[CaseAgentSession]:
    """List current user's sessions for one case."""

    return service.list_sessions(case_id, actor)


@router.post(
    "/cases/{case_id}/case-agent/sessions",
    response_model=CaseAgentSession,
)
def create_case_agent_session(
    case_id: str,
    body: CaseAgentCreateSessionInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentSession:
    """Create a new isolated Case Agent session for this case."""

    return service.create_session(case_id, body, actor)


@router.get(
    "/case-agent/sessions/{session_id}",
    response_model=CaseAgentSession,
)
def get_case_agent_session(
    session_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentSession:
    """Read one session with messages."""

    return service.get_session(session_id, actor)


@router.patch(
    "/case-agent/sessions/{session_id}",
    response_model=CaseAgentSession,
)
def rename_case_agent_session(
    session_id: str,
    body: CaseAgentRenameSessionInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentSession:
    """Rename one current user's active Case Agent session."""

    return service.rename_session(session_id, body, actor)


@router.delete("/case-agent/sessions/{session_id}")
def archive_case_agent_session(
    session_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> dict[str, bool]:
    """Archive one current user's Case Agent session."""

    return service.archive_session(session_id, actor)


@router.post(
    "/case-agent/sessions/{session_id}/restore",
    response_model=CaseAgentSession,
)
def restore_case_agent_session(
    session_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentSession:
    """Restore one archived Case Agent session for undo."""

    return service.restore_session(session_id, actor)


@router.post(
    "/case-agent/sessions/{session_id}/read",
    response_model=CaseAgentSession,
)
def mark_case_agent_session_read(
    session_id: str,
    body: CaseAgentMarkReadInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentSession:
    """Advance the read watermark through one exact terminal run."""

    return service.mark_session_read(session_id, body, actor)


@router.post(
    "/case-agent/sessions/{session_id}/messages",
    response_model=CaseAgentRun,
)
def send_case_agent_message(
    session_id: str,
    body: CaseAgentSendMessageInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentRun:
    """Append a user message and start a background assistant turn."""

    return service.send_message(session_id, body, actor)


@router.get("/case-agent/runs/{run_id}", response_model=CaseAgentRun)
def get_case_agent_run(
    run_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentRun:
    """Read one Case Agent run."""

    return service.get_run(run_id, actor)


@router.post("/case-agent/runs/{run_id}/cancel", response_model=CaseAgentRun)
def cancel_case_agent_run(
    run_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> CaseAgentRun:
    """Request an idempotent cooperative stop for one current user's run."""

    return service.cancel_run(run_id, actor)


@router.get("/case-agent/runs/{run_id}/events")
def stream_case_agent_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> StreamingResponse:
    """SSE event stream for one Case Agent run."""

    service.get_run(run_id, actor)
    cursor = max(after, int(last_event_id or 0))
    return StreamingResponse(
        _event_stream(service, run_id, actor, cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/case-agent/sessions/{session_id}/messages/{message_id}/adopt-note",
    response_model=AuditNote,
)
def adopt_case_agent_note(
    session_id: str,
    message_id: str,
    body: CaseAgentAdoptNoteInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: CaseAgentService = Depends(get_case_agent),
) -> AuditNote:
    """Save a human-confirmed assistant draft as an audit work note."""

    return service.adopt_note(session_id, message_id, body, actor)


def _event_stream(
    service: CaseAgentService,
    run_id: str,
    actor: AuthenticatedUser,
    cursor: int,
) -> Iterator[str]:
    last_heartbeat = time.monotonic()
    while True:
        events = service.list_events(run_id, cursor)
        for event in events:
            cursor = event.sequence
            payload = event.model_dump(mode="json")
            yield (
                f"id: {event.sequence}\n"
                f"event: {event.event_type}\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )
        try:
            run = service.get_run(run_id, actor)
        except Exception:
            break
        if run.status in TERMINAL_STATUSES and not events:
            break
        if time.monotonic() - last_heartbeat >= 15:
            yield ": keep-alive\n\n"
            last_heartbeat = time.monotonic()
        time.sleep(0.5)
