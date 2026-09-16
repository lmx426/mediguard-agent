"""Case-memory governance, recall and projection maintenance endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi import HTTPException, status

from ..dependencies import get_case_memory, get_current_user
from ..schemas.memory import (
    MemoryActionInput,
    MemoryBatchRecallInput,
    MemoryCaptureInput,
    MemoryPreview,
    MemoryPreviewInput,
    PersonalMemoryPreferenceInput,
    PersonalMemoryPreferenceResponse,
)
from ...domain.audit.review.entities import AuthenticatedUser
from ...domain.case_memory.entities import MemoryHintPack, MemoryRecallRequest, MemoryScope

router = APIRouter(prefix="/memory", tags=["case-memory"])


def _scope(actor: AuthenticatedUser, scope_type: str | None, scope_id: str | None) -> MemoryScope:
    resolved_type = scope_type or "auditor"
    resolved_id = scope_id or (actor.id if resolved_type == "auditor" else "global")
    if resolved_type == "auditor" and resolved_id != actor.id and "admin" not in actor.roles:
        resolved_id = actor.id
    if resolved_type in {"team", "department", "global"} and "admin" not in actor.roles:
        resolved_type, resolved_id = "auditor", actor.id
    return MemoryScope(scope_type=resolved_type, scope_id=resolved_id)


@router.get("/pending-count")
def pending_count(
    scope_type: str | None = Query(default=None),
    scope_id: str | None = Query(default=None),
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
) -> dict[str, int]:
    return {"count": service.pending_count(_scope(actor, scope_type, scope_id))}


@router.get("/preference", response_model=PersonalMemoryPreferenceResponse)
def get_personal_memory_preference(
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
) -> PersonalMemoryPreferenceResponse:
    return PersonalMemoryPreferenceResponse(
        enabled=service.personal_memory_enabled(actor.id)
    )


@router.put("/preference", response_model=PersonalMemoryPreferenceResponse)
def update_personal_memory_preference(
    body: PersonalMemoryPreferenceInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
) -> PersonalMemoryPreferenceResponse:
    return PersonalMemoryPreferenceResponse(
        enabled=service.set_personal_memory_enabled(actor.id, body.enabled)
    )


@router.get("/candidates")
def list_candidates(
    scope_type: str | None = Query(default=None),
    scope_id: str | None = Query(default=None),
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    return service.presentation_items(scope=_scope(actor, scope_type, scope_id), status="candidate")


@router.get("/active")
def list_active(
    memory_type: str | None = Query(default=None),
    scope_type: str | None = Query(default=None),
    scope_id: str | None = Query(default=None),
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    return service.presentation_items(scope=_scope(actor, scope_type, scope_id), status="active", memory_type=memory_type)


@router.get("/archived")
def list_archived(
    memory_type: str | None = Query(default=None),
    scope_type: str | None = Query(default=None),
    scope_id: str | None = Query(default=None),
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    return service.presentation_items(scope=_scope(actor, scope_type, scope_id), status="archived", memory_type=memory_type)


@router.post("/capture")
def capture_memory(
    body: MemoryCaptureInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    if "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="manual memory capture requires admin role")
    scope = body.scope
    memory = service.capture(
        memory_type=body.memory_type,
        level=body.memory_level,
        scope=scope,
        payload=body.payload,
        source_event_ids=body.source_event_ids,
        allowed_consumers=body.allowed_consumers,
        confidence=body.confidence,
        feature=body.feature,
    )
    return service._safe_item(memory)


@router.post("/recall", response_model=MemoryHintPack)
def recall_memory(
    body: MemoryBatchRecallInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
) -> MemoryHintPack:
    if "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="raw memory recall requires admin role")
    scope = _scope(actor, body.scope.scope_type, body.scope.scope_id)
    return service.recall(MemoryRecallRequest(request_id=body.request_id, consumer=body.consumer, memory_type=body.memory_type, scope=scope, task_context=body.task_context))


@router.post("/outbox/process")
def process_outbox(
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    if "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory maintenance requires admin role")
    return service.process_pending_projections()


@router.post("/reconcile")
def reconcile_memory(
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    if "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory maintenance requires admin role")
    return service.reconcile()


@router.get("/{memory_id}")
def get_memory_detail(
    memory_id: str,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    memory = service.repository.get(memory_id)
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
    if memory.scope.scope_type == "auditor" and memory.scope.scope_id != actor.id and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory is outside the current auditor scope")
    if memory.scope.scope_type == "department" and memory.scope.scope_id != actor.department and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory is outside the current department scope")
    if memory.scope.scope_type in {"team", "global"} and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="shared memory detail requires admin role")
    return service.safe_detail(memory_id, actor.id)


@router.post("/{memory_id}/actions")
def memory_action(
    memory_id: str,
    body: MemoryActionInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
):
    current = service.repository.get(memory_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
    if current.scope.scope_type != "auditor" and "admin" not in actor.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="shared-scope memory governance requires admin role",
        )
    if body.action in {"auto_activate", "supersede", "mark_conflict", "tombstone"} and "admin" not in actor.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this memory governance action requires admin role",
        )
    memory = service.govern(
        memory_id,
        body.action,
        actor.id,
        body.snoozed_until,
        body.related_memory_id,
    )
    return {
        "memory_id": memory.memory_id,
        "status": memory.status,
        "projection": {"status": "queued"},
    }


@router.post("/{memory_id}/preview", response_model=MemoryPreview)
def preview_memory(
    memory_id: str,
    body: MemoryPreviewInput,
    actor: AuthenticatedUser = Depends(get_current_user),
    service=Depends(get_case_memory),
) -> MemoryPreview:
    memory = service.repository.get(memory_id)
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory not found")
    if memory.scope.scope_type == "auditor" and memory.scope.scope_id != actor.id and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory is outside the current auditor scope")
    if memory.scope.scope_type == "department" and memory.scope.scope_id != actor.department and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="memory is outside the current department scope")
    if memory.scope.scope_type in {"team", "global"} and "admin" not in actor.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="shared memory preview requires admin role")
    return service.preview(memory_id, body.consumer, body.task_context, actor.id)
