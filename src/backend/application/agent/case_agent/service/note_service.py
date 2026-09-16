"""Case Agent note adoption service."""

from __future__ import annotations

from src.backend.application.audit.review.manage_notes_uc import ManageNotesUseCase
from src.backend.application.ports.case_agent import CaseAgentRepository
from src.backend.core.exceptions import ResourceConflictError, ResourceNotFoundError
from src.backend.domain.audit.review.entities import AuthenticatedUser
from src.backend.domain.audit.review.workflow_projector import AuditNote, AuditNoteInput
from src.backend.domain.case_agent.entities import CaseAgentAdoptNoteInput


class CaseAgentNoteService:
    """Persist human-confirmed Case Agent drafts as audit work notes."""

    def __init__(
        self,
        *,
        repository: CaseAgentRepository,
        manage_notes: ManageNotesUseCase,
    ) -> None:
        self._repository = repository
        self._manage_notes = manage_notes

    def adopt_note(
        self,
        session_id: str,
        message_id: str,
        body: CaseAgentAdoptNoteInput,
        actor: AuthenticatedUser,
    ) -> AuditNote:
        session = self._repository.get_session(session_id=session_id, actor_id=actor.id)
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        assistant_message = next(
            (
                message
                for message in session.messages
                if message.message_id == message_id and message.role == "assistant"
            ),
            None,
        )
        if assistant_message is None:
            raise ResourceNotFoundError("待采纳的助手消息不存在")
        if (
            assistant_message.answer_payload is not None
            and assistant_message.answer_payload.metadata.get("case_context_stale") is True
        ):
            raise ResourceConflictError(
                "案件信息已更新，请重新核验后再采纳工作笔记",
                code="case_agent_answer_stale",
            )
        note_input = AuditNoteInput(
            source="case_agent_adopted",
            content=body.content.strip(),
            source_refs=list(dict.fromkeys(body.source_refs)),
            materials=[],
        )
        return self._manage_notes.add(session.case_id, note_input, actor)
