"""PostgreSQL repository for Case Agent state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .....application.ports.case_agent import CaseAgentRepository
from .....application.agent.case_agent.memory.task_state import (
    merge_persisted_task_state,
    task_state_update_is_stale,
)
from .....domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentEvent,
    CaseAgentMessage,
    CaseAgentRun,
    CaseAgentRunSummary,
    CaseAgentSession,
    CaseAgentTaskState,
)
from .....core.exceptions import ResourceConflictError
from ..models import (
    CaseAgentCheckpointORM,
    CaseAgentEventORM,
    CaseAgentMessageORM,
    CaseAgentNodeExecutionORM,
    CaseAgentRunORM,
    CaseAgentSessionORM,
    CaseAgentToolCallORM,
    CaseORM,
    UserORM,
)
from ..session import session_scope


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_persisted_answer_payload(payload: Any) -> Any:
    """Adapt legacy persisted claims without weakening the current write contract."""

    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return normalized
    claims: list[Any] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            claims.append(raw_claim)
            continue
        claim = dict(raw_claim)
        legacy_requirement_id = str(claim.pop("requirement_id", "") or "").strip()
        legacy_slot_id = str(claim.pop("slot_id", "") or "").strip()
        claim.pop("need_kind", None)
        if not str(claim.get("need_id") or "").strip():
            claim["need_id"] = (legacy_requirement_id or legacy_slot_id)[:80]
        if not str(claim.get("need_text") or "").strip():
            claim["need_text"] = str(claim.get("text") or "")[:240]
        claims.append(claim)
    normalized["claims"] = claims
    return normalized


class SqlCaseAgentRepository(CaseAgentRepository):
    """Persist lightweight Case Agent sessions, runs and events in PostgreSQL."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_session(
        self,
        *,
        case_id: str,
        actor_id: str,
        title: str,
        model_name: str,
    ) -> CaseAgentSession:
        with session_scope(self._session_factory) as session:
            case_row = self._case_row(session, case_id, lock=True)
            if case_row is None:
                raise KeyError(case_id)
            actor = self._user_row(session, actor_id)
            if actor is None:
                raise KeyError(actor_id)
            row = CaseAgentSessionORM(
                session_id=f"csess_{uuid4().hex}",
                case_id=case_row.id,
                actor_id=actor.id,
                title=title,
                model_name=model_name,
            )
            session.add(row)
            session.flush()
            return self._to_session(session, row, include_messages=True)

    def list_sessions(self, *, case_id: str, actor_id: str) -> list[CaseAgentSession]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(CaseAgentSessionORM)
                .join(CaseORM, CaseORM.id == CaseAgentSessionORM.case_id)
                .join(UserORM, UserORM.id == CaseAgentSessionORM.actor_id)
                .where(
                    CaseORM.case_id == case_id,
                    UserORM.id == UUID(actor_id),
                    CaseAgentSessionORM.status == "active",
                )
                .order_by(
                    CaseAgentSessionORM.updated_at.desc(),
                    CaseAgentSessionORM.id.desc(),
                )
            ).all()
            return [self._to_session(session, row, include_messages=False) for row in rows]

    def get_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        include_messages: bool = True,
    ) -> CaseAgentSession | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(CaseAgentSessionORM)
                .join(UserORM, UserORM.id == CaseAgentSessionORM.actor_id)
                .where(
                    CaseAgentSessionORM.session_id == session_id,
                    UserORM.id == UUID(actor_id),
                    CaseAgentSessionORM.status == "active",
                )
            )
            return self._to_session(session, row, include_messages=include_messages) if row is not None else None

    def archive_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> bool:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True, status=None)
            if row is None:
                return False
            if row.status != "archived":
                row.status = "archived"
                row.updated_at = datetime.now(timezone.utc)
            return True

    def restore_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentSession | None:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True, status=None)
            if row is None:
                return None
            if row.status != "active":
                row.status = "active"
                row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_session(session, row, include_messages=True)

    def rename_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        title: str,
    ) -> CaseAgentSession | None:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                return None
            row.title = title
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_session(session, row, include_messages=True)

    def append_user_message(
        self,
        *,
        session_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
    ) -> CaseAgentMessage:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                raise KeyError(session_id)
            message = CaseAgentMessageORM(
                message_id=f"cmsg_{uuid4().hex}",
                session_id=row.id,
                actor_id=row.actor_id,
                role="user",
                content=content,
                active_stage=active_stage,
                source_refs=[],
            )
            session.add(message)
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_message(message, row.session_id)

    def create_turn(
        self,
        *,
        session_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
        model_name: str,
        client_request_id: str,
        case_context_fingerprint: str | None = None,
    ) -> tuple[CaseAgentMessage, CaseAgentRun, bool]:
        """Atomically append a user message and create its serialized session run."""

        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                raise KeyError(session_id)

            existing = session.scalar(
                select(CaseAgentRunORM).where(
                    CaseAgentRunORM.session_id == row.id,
                    CaseAgentRunORM.client_request_id == client_request_id,
                )
            )
            if existing is not None:
                existing_message = session.get(CaseAgentMessageORM, existing.user_message_id)
                if existing_message is None:
                    raise KeyError(existing.run_id)
                return (
                    self._to_message(existing_message, row.session_id),
                    self._to_run(session, existing),
                    False,
                )

            active = session.scalar(
                select(CaseAgentRunORM)
                .where(
                    CaseAgentRunORM.session_id == row.id,
                    CaseAgentRunORM.status.in_(("created", "running", "resuming")),
                )
                .order_by(CaseAgentRunORM.created_at.desc(), CaseAgentRunORM.id.desc())
                .with_for_update()
            )
            if active is not None:
                raise ResourceConflictError(
                    "当前会话正在处理上一条问题",
                    code="case_agent_session_busy",
                    context={
                        "active_run_id": active.run_id,
                        "active_run_status": self._effective_run_status(active),
                    },
                )

            parent = session.scalar(
                select(CaseAgentRunORM)
                .where(
                    CaseAgentRunORM.session_id == row.id,
                    CaseAgentRunORM.actor_id == row.actor_id,
                    CaseAgentRunORM.status == "waiting_for_user",
                    CaseAgentRunORM.resumed_by_run_id.is_(None),
                )
                .order_by(CaseAgentRunORM.updated_at.desc(), CaseAgentRunORM.id.desc())
                .with_for_update()
            )
            user_message = CaseAgentMessageORM(
                message_id=f"cmsg_{uuid4().hex}",
                session_id=row.id,
                actor_id=row.actor_id,
                role="user",
                content=content,
                active_stage=active_stage,
                source_refs=[],
            )
            session.add(user_message)
            session.flush()
            resume_context = (
                {
                    "parent_run_id": parent.run_id,
                    "pending_clarification": parent.pending_clarification or {},
                    "parent_user_message_id": session.scalar(
                        select(CaseAgentMessageORM.message_id).where(
                            CaseAgentMessageORM.id == parent.user_message_id
                        )
                    ),
                }
                if parent is not None
                else {}
            )
            run = CaseAgentRunORM(
                run_id=f"crun_{uuid4().hex}",
                session_id=row.id,
                case_id=row.case_id,
                actor_id=row.actor_id,
                user_message_id=user_message.id,
                parent_run_id=parent.id if parent is not None else None,
                status="created",
                current_node="created",
                resume_context=resume_context,
                model_name=model_name,
                client_request_id=client_request_id,
                case_context_fingerprint=case_context_fingerprint,
            )
            session.add(run)
            session.flush()
            if parent is not None:
                parent.resumed_by_run_id = run.id
                parent.updated_at = datetime.now(timezone.utc)
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return (
                self._to_message(user_message, row.session_id),
                self._to_run(session, run),
                True,
            )

    def append_assistant_message(
        self,
        *,
        session_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
        answer_payload: CaseAgentAnswer,
        source_refs: list[str],
    ) -> CaseAgentMessage:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                raise KeyError(session_id)
            message = CaseAgentMessageORM(
                message_id=f"cmsg_{uuid4().hex}",
                session_id=row.id,
                actor_id=row.actor_id,
                role="assistant",
                content=content,
                active_stage=active_stage,
                answer_payload=answer_payload.model_dump(mode="json"),
                source_refs=list(dict.fromkeys(source_refs)),
            )
            session.add(message)
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_message(message, row.session_id)

    def list_recent_messages(
        self,
        *,
        session_id: str,
        actor_id: str,
        limit: int = 8,
    ) -> list[CaseAgentMessage]:
        with self._session_factory() as session:
            row = self._session_row(session, session_id, actor_id, lock=False)
            if row is None:
                return []
            cancelled_user_messages = select(CaseAgentRunORM.user_message_id).where(
                CaseAgentRunORM.status == "cancelled"
            )
            rows = session.scalars(
                select(CaseAgentMessageORM)
                .where(CaseAgentMessageORM.session_id == row.id)
                .where(
                    or_(
                        CaseAgentMessageORM.role != "user",
                        CaseAgentMessageORM.id.not_in(cancelled_user_messages),
                    )
                )
                .order_by(CaseAgentMessageORM.created_at.desc())
                .limit(limit)
            ).all()
            return [self._to_message(message, row.session_id) for message in reversed(rows)]

    def create_run(
        self,
        *,
        session_id: str,
        actor_id: str,
        user_message_id: str,
        model_name: str,
        parent_run_id: str | None = None,
        resume_context: dict[str, Any] | None = None,
    ) -> CaseAgentRun:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                raise KeyError(session_id)
            user_message = session.scalar(
                select(CaseAgentMessageORM).where(
                    CaseAgentMessageORM.message_id == user_message_id,
                    CaseAgentMessageORM.session_id == row.id,
                    CaseAgentMessageORM.role == "user",
                )
            )
            if user_message is None:
                raise KeyError(user_message_id)
            parent_row = None
            if parent_run_id:
                parent_row = self._run_row(session, parent_run_id, lock=True)
                if parent_row is None or parent_row.session_id != row.id or parent_row.actor_id != row.actor_id:
                    raise KeyError(parent_run_id)
            run = CaseAgentRunORM(
                run_id=f"crun_{uuid4().hex}",
                session_id=row.id,
                case_id=row.case_id,
                actor_id=row.actor_id,
                user_message_id=user_message.id,
                parent_run_id=parent_row.id if parent_row is not None else None,
                status="created",
                current_node="created",
                resume_context=resume_context or {},
                model_name=model_name,
            )
            session.add(run)
            row.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_run(session, run)

    def get_run(self, *, run_id: str, actor_id: str) -> CaseAgentRun | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(CaseAgentRunORM)
                .join(CaseAgentSessionORM, CaseAgentSessionORM.id == CaseAgentRunORM.session_id)
                .where(
                    CaseAgentRunORM.run_id == run_id,
                    CaseAgentSessionORM.actor_id == UUID(actor_id),
                )
            )
            if row is None:
                return None
            return self._to_run(session, row)

    def get_active_run_for_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentRun | None:
        with self._session_factory() as session:
            session_row = self._session_row(session, session_id, actor_id, lock=False)
            if session_row is None:
                return None
            row = session.scalar(
                select(CaseAgentRunORM)
                .where(
                    CaseAgentRunORM.session_id == session_row.id,
                    CaseAgentRunORM.status.in_(("created", "running", "resuming")),
                )
                .order_by(CaseAgentRunORM.created_at.desc(), CaseAgentRunORM.id.desc())
            )
            return self._to_run(session, row) if row is not None else None

    def list_recoverable_runs(self) -> list[CaseAgentRun]:
        """Return persisted non-terminal runs left behind by the previous process."""

        with self._session_factory() as session:
            rows = list(
                session.scalars(
                    select(CaseAgentRunORM)
                    .where(CaseAgentRunORM.status.in_(("created", "running", "resuming")))
                    .order_by(CaseAgentRunORM.created_at.asc(), CaseAgentRunORM.id.asc())
                )
            )
            return [self._to_run(session, row) for row in rows]

    def request_run_cancel(
        self,
        *,
        run_id: str,
        actor_id: str,
        reason: str,
    ) -> tuple[CaseAgentRun, bool]:
        terminal = {"waiting_for_user", "completed", "failed", "cancelled", "timed_out", "degraded"}
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(CaseAgentRunORM)
                .join(CaseAgentSessionORM, CaseAgentSessionORM.id == CaseAgentRunORM.session_id)
                .where(
                    CaseAgentRunORM.run_id == run_id,
                    CaseAgentSessionORM.actor_id == UUID(actor_id),
                )
                .with_for_update()
            )
            if row is None:
                raise KeyError(run_id)
            if row.status in terminal:
                return self._to_run(session, row), False
            now = datetime.now(timezone.utc)
            changed = row.cancel_requested_at is None
            if changed:
                row.cancel_requested_at = now
                row.cancel_reason = reason[:80]
            if row.status == "created":
                row.status = "cancelled"
                row.current_node = "cancelled"
                row.cancelled_at = now
                row.completed_at = now
            row.updated_at = now
            session_row = session.get(CaseAgentSessionORM, row.session_id)
            if session_row is not None:
                session_row.updated_at = now
            session.flush()
            return self._to_run(session, row), changed

    def is_run_cancel_requested(self, run_id: str) -> bool:
        with self._session_factory() as session:
            row = self._run_row(session, run_id, lock=False)
            return bool(row is not None and row.cancel_requested_at is not None)

    def mark_run_cancelled(self, run_id: str) -> bool:
        terminal = {"waiting_for_user", "completed", "failed", "cancelled", "timed_out", "degraded"}
        with session_scope(self._session_factory) as session:
            row = self._run_row(session, run_id, lock=True)
            if row is None or row.status in terminal:
                return False
            if row.cancel_requested_at is None:
                return False
            now = datetime.now(timezone.utc)
            row.status = "cancelled"
            row.current_node = "cancelled"
            row.cancelled_at = now
            row.completed_at = now
            row.updated_at = now
            session_row = session.get(CaseAgentSessionORM, row.session_id)
            if session_row is not None:
                session_row.updated_at = now
            return True

    def mark_session_read(
        self,
        *,
        session_id: str,
        actor_id: str,
        through_run_id: str,
    ) -> CaseAgentSession | None:
        with session_scope(self._session_factory) as session:
            session_row = self._session_row(session, session_id, actor_id, lock=True)
            if session_row is None:
                return None
            run = session.scalar(
                select(CaseAgentRunORM).where(
                    CaseAgentRunORM.run_id == through_run_id,
                    CaseAgentRunORM.session_id == session_row.id,
                )
            )
            if run is None:
                raise KeyError(through_run_id)
            if run.completed_at is None:
                raise ResourceConflictError(
                    "运行尚未结束，不能标记为已读",
                    code="case_agent_run_not_terminal",
                    context={"active_run_id": run.run_id},
                )
            if session_row.last_read_at is None or run.completed_at > session_row.last_read_at:
                session_row.last_read_at = run.completed_at
            session.flush()
            return self._to_session(session, session_row, include_messages=True)

    def finalize_run_with_answer(
        self,
        *,
        run_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
        answer_payload: CaseAgentAnswer,
        source_refs: list[str],
        status: str,
        current_node: str,
        error_code: str | None,
        error_message: str | None,
        degraded_reason: str | None,
        pending_clarification: dict[str, Any],
        model_call_count: int,
        tool_call_count: int,
    ) -> CaseAgentMessage | None:
        """Atomically win the cancel/final-answer race and persist one answer."""

        terminal = {"waiting_for_user", "completed", "failed", "cancelled", "timed_out", "degraded"}
        with session_scope(self._session_factory) as session:
            row = self._run_row(session, run_id, lock=True)
            if row is None or row.actor_id != UUID(actor_id):
                raise KeyError(run_id)
            if row.status in terminal:
                return None
            now = datetime.now(timezone.utc)
            if row.cancel_requested_at is not None:
                row.status = "cancelled"
                row.current_node = "cancelled"
                row.cancelled_at = now
                row.completed_at = now
                row.updated_at = now
                return None
            session_row = session.get(CaseAgentSessionORM, row.session_id)
            if session_row is None or session_row.status != "active":
                raise KeyError(row.session_id)
            message = CaseAgentMessageORM(
                message_id=f"cmsg_{uuid4().hex}",
                session_id=session_row.id,
                actor_id=session_row.actor_id,
                role="assistant",
                content=content,
                active_stage=active_stage,
                answer_payload=answer_payload.model_dump(mode="json"),
                source_refs=list(dict.fromkeys(source_refs)),
            )
            session.add(message)
            session.flush()
            row.status = status
            row.current_node = current_node
            row.assistant_message_id = message.id
            row.error_code = error_code
            row.error_message = error_message
            row.degraded_reason = degraded_reason
            row.pending_clarification = pending_clarification
            row.model_call_count = model_call_count
            row.tool_call_count = tool_call_count
            row.completed_at = now
            row.updated_at = now
            session_row.updated_at = now
            session.flush()
            return self._to_message(message, session_row.session_id)

    def find_waiting_run_for_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentRun | None:
        with self._session_factory() as session:
            session_row = self._session_row(session, session_id, actor_id, lock=False)
            if session_row is None:
                return None
            row = session.scalar(
                select(CaseAgentRunORM)
                .where(
                    CaseAgentRunORM.session_id == session_row.id,
                    CaseAgentRunORM.actor_id == session_row.actor_id,
                    CaseAgentRunORM.status == "waiting_for_user",
                    CaseAgentRunORM.resumed_by_run_id.is_(None),
                )
                .order_by(CaseAgentRunORM.updated_at.desc(), CaseAgentRunORM.id.desc())
            )
            return self._to_run(session, row) if row is not None else None

    def mark_run_resumed(
        self,
        *,
        parent_run_id: str,
        resumed_by_run_id: str,
        actor_id: str,
    ) -> None:
        with session_scope(self._session_factory) as session:
            parent = self._run_row(session, parent_run_id, lock=True)
            child = self._run_row(session, resumed_by_run_id, lock=False)
            if parent is None or child is None or parent.actor_id != UUID(actor_id):
                return
            if parent.session_id != child.session_id or parent.actor_id != child.actor_id:
                return
            parent.resumed_by_run_id = child.id
            parent.updated_at = datetime.now(timezone.utc)

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_node: str | None = None,
        assistant_message_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        degraded_reason: str | None = None,
        pending_clarification: dict[str, Any] | None = None,
        model_call_count: int | None = None,
        tool_call_count: int | None = None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            row = self._run_row(session, run_id, lock=True)
            if row is None:
                raise KeyError(run_id)
            if status is not None:
                status = {"queued": "created", "complete": "completed"}.get(status, status)
                row.status = status
                if status in {"running", "resuming"} and row.started_at is None:
                    row.started_at = datetime.now(timezone.utc)
                if status in {
                    "waiting_for_user",
                    "completed",
                    "failed",
                    "cancelled",
                    "timed_out",
                    "degraded",
                }:
                    row.completed_at = datetime.now(timezone.utc)
                    if status == "cancelled":
                        row.cancelled_at = row.completed_at
            if current_node is not None:
                row.current_node = current_node
            if assistant_message_id is not None:
                assistant = session.scalar(
                    select(CaseAgentMessageORM).where(
                        CaseAgentMessageORM.message_id == assistant_message_id,
                        CaseAgentMessageORM.session_id == row.session_id,
                    )
                )
                if assistant is not None:
                    row.assistant_message_id = assistant.id
            row.error_code = error_code
            row.error_message = error_message
            if degraded_reason is not None:
                row.degraded_reason = degraded_reason
            if pending_clarification is not None:
                row.pending_clarification = pending_clarification
            if model_call_count is not None:
                row.model_call_count = model_call_count
            if tool_call_count is not None:
                row.tool_call_count = tool_call_count
            session_row = session.scalar(
                select(CaseAgentSessionORM).where(CaseAgentSessionORM.id == row.session_id)
                .with_for_update()
            )
            if session_row is not None:
                session_row.updated_at = datetime.now(timezone.utc)

    def update_working_memory(
        self,
        *,
        session_id: str,
        actor_id: str,
        session_summary: str | None = None,
        current_topic: str | None = None,
        pending_tool: str | None = None,
        referenced_source_refs: list[str] | None = None,
        task_state: dict[str, Any] | None = None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            row = self._session_row(session, session_id, actor_id, lock=True)
            if row is None:
                raise KeyError(session_id)
            if task_state is not None and task_state_update_is_stale(
                row.task_state,
                task_state,
            ):
                return
            if session_summary is not None:
                row.session_summary = session_summary
            if current_topic is not None:
                row.current_topic = current_topic
            if pending_tool is not None:
                row.pending_tool = pending_tool
            if referenced_source_refs is not None:
                row.referenced_source_refs = list(dict.fromkeys(referenced_source_refs))
            if task_state is not None:
                row.task_state = merge_persisted_task_state(row.task_state, task_state)
            row.updated_at = datetime.now(timezone.utc)

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> CaseAgentEvent:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(CaseAgentEventORM.sequence), 0)).where(
                        CaseAgentEventORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            row = CaseAgentEventORM(
                run_id=run.id,
                sequence=sequence,
                event_type=event_type,
                message=message,
                payload=payload or {},
            )
            session.add(row)
            session.flush()
            return CaseAgentEvent(
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                message=message,
                payload=row.payload,
                created_at=row.created_at,
            )

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[CaseAgentEvent]:
        with self._session_factory() as session:
            run = self._run_row(session, run_id)
            if run is None:
                return []
            rows = session.scalars(
                select(CaseAgentEventORM)
                .where(
                    CaseAgentEventORM.run_id == run.id,
                    CaseAgentEventORM.sequence > after_sequence,
                )
                .order_by(CaseAgentEventORM.sequence)
            ).all()
            return [
                CaseAgentEvent(
                    run_id=run_id,
                    sequence=row.sequence,
                    event_type=row.event_type,
                    message=row.message,
                    payload=row.payload,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def start_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> int:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(CaseAgentToolCallORM.sequence), 0)).where(
                        CaseAgentToolCallORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            now = datetime.now(timezone.utc)
            session.add(
                CaseAgentToolCallORM(
                    run_id=run.id,
                    sequence=sequence,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    idempotency_key=self._tool_idempotency_key(
                        run_id=run_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    ),
                    result_summary={},
                    source_refs=[],
                    status="running",
                    started_at=now,
                    latency_ms=0,
                )
            )
            return sequence

    def finish_tool_call(
        self,
        run_id: str,
        *,
        sequence: int,
        result_summary: dict[str, Any] | None,
        source_refs: list[str],
        status: str,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=False)
            if run is None:
                raise KeyError(run_id)
            row = session.scalar(
                select(CaseAgentToolCallORM)
                .where(
                    CaseAgentToolCallORM.run_id == run.id,
                    CaseAgentToolCallORM.sequence == sequence,
                )
                .with_for_update()
            )
            if row is None:
                raise KeyError(f"{run_id}:{sequence}")
            row.result_summary = result_summary
            row.source_refs = list(dict.fromkeys(source_refs))
            row.status = status
            row.error_code = error_code
            row.latency_ms = latency_ms
            row.response_ref = self._tool_response_ref(
                status=status,
                error_code=error_code,
                source_refs=source_refs,
                result_summary=result_summary,
            )

    def record_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: dict[str, Any] | None,
        source_refs: list[str],
        status: str,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        sequence = self.start_tool_call(
            run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        self.finish_tool_call(
            run_id,
            sequence=sequence,
            result_summary=result_summary,
            source_refs=source_refs,
            status=status,
            error_code=error_code,
            latency_ms=latency_ms,
        )

    def start_node_execution(
        self,
        run_id: str,
        *,
        node_name: str,
        input_snapshot: dict[str, Any],
    ) -> int:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(CaseAgentNodeExecutionORM.sequence), 0)).where(
                        CaseAgentNodeExecutionORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            session.add(
                CaseAgentNodeExecutionORM(
                    run_id=run.id,
                    sequence=sequence,
                    node_name=node_name,
                    status="running",
                    input_snapshot=input_snapshot,
                    output_snapshot={},
                    latency_ms=0,
                )
            )
            return sequence

    def finish_node_execution(
        self,
        run_id: str,
        *,
        sequence: int,
        output_snapshot: dict[str, Any],
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=False)
            if run is None:
                raise KeyError(run_id)
            row = session.scalar(
                select(CaseAgentNodeExecutionORM)
                .where(
                    CaseAgentNodeExecutionORM.run_id == run.id,
                    CaseAgentNodeExecutionORM.sequence == sequence,
                )
                .with_for_update()
            )
            if row is None:
                raise KeyError(f"{run_id}:{sequence}")
            finished_at = datetime.now(timezone.utc)
            row.status = "completed"
            row.output_snapshot = output_snapshot
            row.finished_at = finished_at
            row.latency_ms = max(0, int((finished_at - row.started_at).total_seconds() * 1000))

    def fail_node_execution(
        self,
        run_id: str,
        *,
        sequence: int,
        error_code: str,
        error_message: str,
        output_snapshot: dict[str, Any] | None = None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=False)
            if run is None:
                raise KeyError(run_id)
            row = session.scalar(
                select(CaseAgentNodeExecutionORM)
                .where(
                    CaseAgentNodeExecutionORM.run_id == run.id,
                    CaseAgentNodeExecutionORM.sequence == sequence,
                )
                .with_for_update()
            )
            if row is None:
                raise KeyError(f"{run_id}:{sequence}")
            finished_at = datetime.now(timezone.utc)
            row.status = "failed"
            row.output_snapshot = output_snapshot or {}
            row.error_code = error_code
            row.error_message = error_message[:500]
            row.finished_at = finished_at
            row.latency_ms = max(0, int((finished_at - row.started_at).total_seconds() * 1000))

    def save_checkpoint(
        self,
        run_id: str,
        *,
        node_name: str,
        state_snapshot: dict[str, Any],
        source_refs: list[str],
        safe_to_resume: bool = True,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(CaseAgentCheckpointORM.sequence), 0)).where(
                        CaseAgentCheckpointORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            session.add(
                CaseAgentCheckpointORM(
                    run_id=run.id,
                    sequence=sequence,
                    node_name=node_name,
                    state_snapshot=state_snapshot,
                    source_refs=list(dict.fromkeys(source_refs)),
                    safe_to_resume=safe_to_resume,
                )
            )

    def get_latest_checkpoint(
        self,
        run_id: str,
        *,
        node_name: str | None = None,
    ) -> dict[str, Any] | None:
        with self._session_factory() as session:
            run = self._run_row(session, run_id, lock=False)
            if run is None:
                return None
            statement = select(CaseAgentCheckpointORM).where(
                CaseAgentCheckpointORM.run_id == run.id
            )
            if node_name is not None:
                statement = statement.where(CaseAgentCheckpointORM.node_name == node_name)
            row = session.scalar(
                statement.order_by(
                    CaseAgentCheckpointORM.sequence.desc(),
                    CaseAgentCheckpointORM.id.desc(),
                )
            )
            return row.state_snapshot if row is not None else None

    @staticmethod
    def _tool_idempotency_key(
        *,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        return _canonical_hash(
            {
                "run_id": run_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )

    @staticmethod
    def _tool_response_ref(
        *,
        status: str,
        error_code: str | None,
        source_refs: list[str],
        result_summary: dict[str, Any] | None,
    ) -> str:
        return (
            "case-agent-tool-response:"
            + _canonical_hash(
                {
                    "status": status,
                    "error_code": error_code,
                    "source_refs": source_refs,
                    "result_summary": result_summary,
                }
            )[:24]
        )

    @staticmethod
    def _run_row(session: Session, run_id: str, lock: bool = False) -> CaseAgentRunORM | None:
        statement = select(CaseAgentRunORM).where(CaseAgentRunORM.run_id == run_id)
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    @staticmethod
    def _case_row(session: Session, case_id: str, lock: bool = False) -> CaseORM | None:
        statement = select(CaseORM).where(CaseORM.case_id == case_id)
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    @staticmethod
    def _user_row(session: Session, actor_id: str) -> UserORM | None:
        try:
            return session.get(UserORM, UUID(actor_id))
        except ValueError:
            return None

    def _session_row(
        self,
        session: Session,
        session_id: str,
        actor_id: str,
        lock: bool = False,
        status: str | None = "active",
    ) -> CaseAgentSessionORM | None:
        statement = (
            select(CaseAgentSessionORM)
            .join(UserORM, UserORM.id == CaseAgentSessionORM.actor_id)
            .where(
                CaseAgentSessionORM.session_id == session_id,
                UserORM.id == UUID(actor_id),
            )
        )
        if status is not None:
            statement = statement.where(CaseAgentSessionORM.status == status)
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    def _to_session(
        self,
        session: Session,
        row: CaseAgentSessionORM | None,
        include_messages: bool = True,
    ) -> CaseAgentSession | None:
        if row is None:
            return None
        case_id = session.scalar(select(CaseORM.case_id).where(CaseORM.id == row.case_id))
        messages: list[CaseAgentMessage] = []
        if include_messages:
            rows = session.scalars(
                select(CaseAgentMessageORM)
                .where(CaseAgentMessageORM.session_id == row.id)
                .order_by(CaseAgentMessageORM.created_at)
            ).all()
            messages = [self._to_message(message, row.session_id) for message in rows]
        latest_run_row = session.scalar(
            select(CaseAgentRunORM)
            .where(CaseAgentRunORM.session_id == row.id)
            .order_by(CaseAgentRunORM.created_at.desc(), CaseAgentRunORM.id.desc())
            .limit(1)
        )
        latest_run = self._to_run_summary(latest_run_row) if latest_run_row is not None else None
        attention_type = self._attention_type(latest_run_row)
        attention_at = latest_run_row.completed_at if latest_run_row is not None else None
        has_unread_activity = bool(
            attention_type != "none"
            and attention_at is not None
            and (row.last_read_at is None or attention_at > row.last_read_at)
        )
        return CaseAgentSession(
            session_id=row.session_id,
            case_id=str(case_id or ""),
            actor_id=str(row.actor_id),
            title=row.title,
            status=row.status,
            session_summary=row.session_summary,
            current_topic=row.current_topic,
            pending_tool=row.pending_tool,
            referenced_source_refs=list(row.referenced_source_refs or []),
            task_state=CaseAgentTaskState.model_validate(row.task_state or {}),
            latest_run=latest_run,
            attention_type=attention_type,
            has_unread_activity=has_unread_activity,
            last_read_at=row.last_read_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            messages=messages,
        )

    @staticmethod
    def _to_message(row: CaseAgentMessageORM, public_session_id: str) -> CaseAgentMessage:
        answer_payload = (
            CaseAgentAnswer.model_validate(
                _normalize_persisted_answer_payload(row.answer_payload)
            )
            if row.answer_payload is not None
            else None
        )
        return CaseAgentMessage(
            message_id=row.message_id,
            session_id=public_session_id,
            role=row.role,
            content=row.content,
            active_stage=row.active_stage,
            answer_payload=answer_payload,
            source_refs=list(row.source_refs or []),
            created_at=row.created_at,
        )

    def _to_run(self, session: Session, row: CaseAgentRunORM) -> CaseAgentRun:
        session_row = session.get(CaseAgentSessionORM, row.session_id)
        case_id = session.scalar(select(CaseORM.case_id).where(CaseORM.id == row.case_id))
        user_message_id = session.scalar(
            select(CaseAgentMessageORM.message_id).where(CaseAgentMessageORM.id == row.user_message_id)
        )
        assistant_message_id = (
            session.scalar(
                select(CaseAgentMessageORM.message_id).where(CaseAgentMessageORM.id == row.assistant_message_id)
            )
            if row.assistant_message_id is not None
            else None
        )
        parent_run_id = (
            session.scalar(select(CaseAgentRunORM.run_id).where(CaseAgentRunORM.id == row.parent_run_id))
            if row.parent_run_id is not None
            else None
        )
        resumed_by_run_id = (
            session.scalar(select(CaseAgentRunORM.run_id).where(CaseAgentRunORM.id == row.resumed_by_run_id))
            if row.resumed_by_run_id is not None
            else None
        )
        return CaseAgentRun(
            run_id=row.run_id,
            session_id=session_row.session_id if session_row is not None else "",
            case_id=str(case_id or ""),
            actor_id=str(row.actor_id),
            user_message_id=str(user_message_id or ""),
            assistant_message_id=str(assistant_message_id) if assistant_message_id else None,
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            resumed_by_run_id=str(resumed_by_run_id) if resumed_by_run_id else None,
            status=row.status,
            current_node=row.current_node,
            error_code=row.error_code,
            error_message=row.error_message,
            degraded_reason=row.degraded_reason,
            pending_clarification=dict(row.pending_clarification or {}),
            resume_context=dict(row.resume_context or {}),
            model_name=row.model_name,
            model_call_count=row.model_call_count,
            tool_call_count=row.tool_call_count,
            timeout_at=row.timeout_at,
            started_at=row.started_at,
            cancel_requested_at=row.cancel_requested_at,
            cancelled_at=row.cancelled_at,
            cancel_reason=row.cancel_reason,
            client_request_id=row.client_request_id,
            case_context_fingerprint=row.case_context_fingerprint,
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
        )

    @staticmethod
    def _effective_run_status(row: CaseAgentRunORM) -> str:
        if row.cancel_requested_at is not None and row.status in {"created", "running", "resuming"}:
            return "cancelling"
        return row.status

    @classmethod
    def _to_run_summary(cls, row: CaseAgentRunORM) -> CaseAgentRunSummary:
        return CaseAgentRunSummary(
            run_id=row.run_id,
            status=row.status,
            effective_status=cls._effective_run_status(row),
            current_node=row.current_node,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            cancel_requested_at=row.cancel_requested_at,
        )

    @staticmethod
    def _attention_type(row: CaseAgentRunORM | None) -> str:
        if row is None:
            return "none"
        if row.status == "completed":
            return "new_result"
        if row.status == "waiting_for_user":
            return "needs_input"
        if row.status == "degraded":
            return "degraded"
        if row.status in {"failed", "timed_out"}:
            return "failed"
        return "none"
