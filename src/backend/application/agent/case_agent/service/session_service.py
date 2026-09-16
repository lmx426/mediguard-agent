"""Case Agent session lifecycle service."""

from __future__ import annotations

from datetime import datetime

from src.backend.application.ports.case_agent import CaseAgentRepository
from src.backend.core.exceptions import (
    BusinessValidationError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from src.backend.domain.audit.review.entities import (
    AuthenticatedUser,
    find_forbidden_content,
)
from src.backend.domain.case_agent.entities import (
    CaseAgentCreateSessionInput,
    CaseAgentRenameSessionInput,
    CaseAgentSession,
)


class CaseAgentSessionService:
    """Handle Case Agent session CRUD without touching graph execution."""

    def __init__(self, *, repository: CaseAgentRepository, model_name: str) -> None:
        self._repository = repository
        self._model_name = model_name

    def list_sessions(self, case_id: str, actor: AuthenticatedUser) -> list[CaseAgentSession]:
        return self._repository.list_sessions(case_id=case_id, actor_id=actor.id)

    def create_session(
        self,
        case_id: str,
        body: CaseAgentCreateSessionInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentSession:
        title = (body.title or f"会话 {datetime.now().strftime('%m-%d %H:%M')}").strip()
        try:
            return self._repository.create_session(
                case_id=case_id,
                actor_id=actor.id,
                title=title[:80],
                model_name=self._model_name,
            )
        except KeyError as exc:
            raise ResourceNotFoundError(f"案件 {case_id} 不存在") from exc

    def get_session(self, session_id: str, actor: AuthenticatedUser) -> CaseAgentSession:
        session = self._repository.get_session(session_id=session_id, actor_id=actor.id)
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        return session

    def archive_session(self, session_id: str, actor: AuthenticatedUser) -> dict[str, bool]:
        active_run = self._repository.get_active_run_for_session(
            session_id=session_id,
            actor_id=actor.id,
        )
        if active_run is not None:
            raise ResourceConflictError(
                "会话仍有任务正在处理，请先停止任务",
                code="case_agent_session_busy",
                context={
                    "active_run_id": active_run.run_id,
                    "active_run_status": (
                        "cancelling" if active_run.cancel_requested_at is not None else active_run.status
                    ),
                },
            )
        archived = self._repository.archive_session(session_id=session_id, actor_id=actor.id)
        if not archived:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        return {"archived": True}

    def restore_session(self, session_id: str, actor: AuthenticatedUser) -> CaseAgentSession:
        session = self._repository.restore_session(session_id=session_id, actor_id=actor.id)
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        return session

    def rename_session(
        self,
        session_id: str,
        body: CaseAgentRenameSessionInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentSession:
        title = body.title.strip()
        if not title:
            raise BusinessValidationError(["Case Agent 会话标题不能为空"])
        forbidden = find_forbidden_content(title)
        if forbidden:
            raise BusinessValidationError([f"Case Agent 会话标题不得包含禁止字段或身份信息：{forbidden}"])
        session = self._repository.rename_session(
            session_id=session_id,
            actor_id=actor.id,
            title=title[:80],
        )
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        return session
