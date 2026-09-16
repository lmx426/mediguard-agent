"""PostgreSQL-backed audit note repository — notes_repo."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .....domain.audit.review.entities import AuthenticatedUser
from .....domain.audit.review.workflow_projector import AuditNote, ReviewAttachment
from ..models import AuditActionORM, AuditNoteORM, CaseORM, UserORM
from ..session import get_or_create_default_auditor, session_scope


class SqlNoteRepository:
    """Persist manual/assistant/external audit notes."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def clear(self) -> None:
        """Delete audit notes. Intended for tests."""

        with session_scope(self._session_factory) as session:
            session.execute(delete(AuditNoteORM))

    def list_for_case(self, case_id: str) -> list[AuditNote]:
        """Return all notes for a public case id."""

        with self._session_factory() as session:
            case = self._case(session, case_id)
            if case is None:
                return []
            rows = session.scalars(
                select(AuditNoteORM)
                .where(AuditNoteORM.case_id == case.id)
                .order_by(AuditNoteORM.created_at)
            ).all()
            return [
                self._to_note(row)
                for row in rows
            ]

    def add(
        self,
        case_id: str,
        note: AuditNote,
        actor: AuthenticatedUser | None = None,
    ) -> None:
        """Append a note without changing case review status."""

        with session_scope(self._session_factory) as session:
            case = self._case(session, case_id)
            if case is None:
                raise ValueError(f"case {case_id} does not exist")
            author = (
                self._actor_user(session, actor)
                if actor is not None
                else get_or_create_default_auditor(session)
            )
            note_id = note.note_id or f"NOTE-{uuid4().hex[:8].upper()}"
            session.add(
                AuditNoteORM(
                    note_id=note_id,
                    case_id=case.id,
                    author_id=author.id,
                    author_name=note.author or author.display_name,
                    source=note.source,
                    content=note.content,
                    source_refs=list(note.source_refs),
                    materials=[
                        material.model_dump(mode="json")
                        for material in note.materials
                    ],
                )
            )
            session.add(
                AuditActionORM(
                    case_id=case.id,
                    actor_id=author.id,
                    action_type="note_added",
                    summary="新增审核工作笔记",
                    payload={
                        "note_id": note_id,
                        "source": note.source,
                        "source_refs": list(note.source_refs),
                        "material_count": len(note.materials),
                    },
                )
            )

    def delete(
        self,
        case_id: str,
        note_id: str,
        actor: AuthenticatedUser | None = None,
    ) -> AuditNote | None:
        """Delete a note row."""

        with session_scope(self._session_factory) as session:
            case = self._case(session, case_id)
            if case is None:
                return None
            row = session.scalar(
                select(AuditNoteORM).where(
                    AuditNoteORM.case_id == case.id,
                    AuditNoteORM.note_id == note_id,
                )
            )
            if row is None:
                return None
            deleted_note = self._to_note(row)
            actor_user = (
                self._actor_user(session, actor)
                if actor is not None
                else get_or_create_default_auditor(session)
            )
            session.add(
                AuditActionORM(
                    case_id=case.id,
                    actor_id=actor_user.id,
                    action_type="note_deleted",
                    summary="删除审核工作笔记",
                    payload={
                        "note_id": note_id,
                        "source": row.source,
                    },
                )
            )
            session.delete(row)
            return deleted_note

    @staticmethod
    def _case(session: Session, case_id: str) -> CaseORM | None:
        return session.scalar(select(CaseORM).where(CaseORM.case_id == case_id))

    @staticmethod
    def _to_note(row: AuditNoteORM) -> AuditNote:
        return AuditNote(
            note_id=row.note_id,
            author=row.author_name,
            source=row.source,
            content=row.content,
            source_refs=list(row.source_refs or []),
            materials=[
                ReviewAttachment(**item)
                for item in (row.materials or [])
            ],
            created_at=row.created_at.isoformat(),
            status=row.status,
            voided_at=row.voided_at.isoformat() if row.voided_at else None,
            voided_by=row.voided_by_name,
        )

    @staticmethod
    def _actor_user(session: Session, actor: AuthenticatedUser) -> UserORM:
        try:
            actor_id = UUID(actor.id)
        except ValueError as exc:
            raise ValueError("authenticated actor id is invalid") from exc
        user = session.get(UserORM, actor_id)
        if user is None:
            raise ValueError("authenticated actor does not exist")
        return user
