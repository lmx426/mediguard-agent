"""PostgreSQL-backed review repository — reviews_repo."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from .....domain.audit.review.entities import AuthenticatedUser
from .....domain.audit.review.workflow_projector import ReviewAttachment, ReviewDecision
from ..models import (
    AuditActionORM,
    CaseORM,
    ReviewAttachmentORM,
    ReviewORM,
    UserORM,
)
from ..session import get_or_create_default_auditor, session_scope


class SqlReviewRepository:
    """Persist one first-review decision per case."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def clear(self) -> None:
        """Delete review rows. Intended for tests."""

        with session_scope(self._session_factory) as session:
            session.execute(delete(ReviewORM))
            session.execute(
                delete(AuditActionORM).where(
                    AuditActionORM.action_type == "review_submitted"
                )
            )
            session.execute(
                CaseORM.__table__.update().values(review_status="pending")
            )

    def get(self, case_id: str) -> ReviewDecision | None:
        """Return the stored review decision by public case id."""

        with self._session_factory() as session:
            row = self._review(session, case_id)
            if row is None:
                return None
            return self._to_decision(row)

    def save(
        self,
        case_id: str,
        review: ReviewDecision,
        actor: AuthenticatedUser | None = None,
    ) -> None:
        """Save or replace the current first-review decision."""

        with session_scope(self._session_factory) as session:
            case = self._case(session, case_id)
            if case is None:
                raise ValueError(f"case {case_id} does not exist")

            reviewer = (
                self._actor_user(session, actor)
                if actor is not None
                else get_or_create_default_auditor(session)
            )
            reviewer_name = review.reviewer or reviewer.display_name
            row = session.scalar(
                select(ReviewORM).where(ReviewORM.case_id == case.id)
            )
            if row is None:
                row = ReviewORM(
                    case_id=case.id,
                    reviewer_id=reviewer.id,
                    reviewer_name=reviewer_name,
                    decision=review.decision,
                    reason=review.reason,
                    submitted_at=self._parse_time(review.submitted_at),
                )
                session.add(row)
                session.flush()
            else:
                row.reviewer_id = reviewer.id
                row.reviewer_name = reviewer_name
                row.decision = review.decision
                row.reason = review.reason
                row.submitted_at = self._parse_time(review.submitted_at)
                row.attachments.clear()
                session.flush()

            for attachment in review.attachments:
                row.attachments.append(self._from_attachment(attachment))

            case.review_status = "reviewed"
            session.add(
                AuditActionORM(
                    case_id=case.id,
                    actor_id=reviewer.id,
                    action_type="review_submitted",
                    summary=f"提交人工初审：{review.decision}",
                    payload=review.model_dump(mode="json"),
                )
            )

    def case_ids(self) -> set[str]:
        """Return public case ids that have submitted reviews."""

        with self._session_factory() as session:
            rows = session.execute(
                select(CaseORM.case_id)
                .join(ReviewORM, ReviewORM.case_id == CaseORM.id)
                .order_by(CaseORM.created_at)
            ).all()
            return {row[0] for row in rows}

    @staticmethod
    def _from_attachment(attachment: ReviewAttachment) -> ReviewAttachmentORM:
        return ReviewAttachmentORM(
            name=attachment.name,
            material_type=attachment.material_type,
            source=attachment.source,
            verification_status=attachment.verification_status,
            remark=attachment.remark,
            file_name=attachment.file_name,
            file_size=attachment.file_size,
            file_type=attachment.file_type,
            file_last_modified=attachment.file_last_modified,
        )

    @staticmethod
    def _to_decision(row: ReviewORM) -> ReviewDecision:
        return ReviewDecision(
            reviewer=row.reviewer_name,
            decision=row.decision,
            reason=row.reason,
            submitted_at=row.submitted_at.isoformat(),
            attachments=[
                ReviewAttachment(
                    name=item.name,
                    material_type=item.material_type,
                    source=item.source,
                    verification_status=item.verification_status,
                    remark=item.remark,
                    file_name=item.file_name,
                    file_size=item.file_size,
                    file_type=item.file_type,
                    file_last_modified=item.file_last_modified,
                )
                for item in row.attachments
            ],
        )

    @staticmethod
    def _parse_time(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _case(session: Session, case_id: str) -> CaseORM | None:
        return session.scalar(select(CaseORM).where(CaseORM.case_id == case_id))

    @staticmethod
    def _review(session: Session, case_id: str) -> ReviewORM | None:
        return session.scalar(
            select(ReviewORM)
            .join(CaseORM, ReviewORM.case_id == CaseORM.id)
            .where(CaseORM.case_id == case_id)
            .options(selectinload(ReviewORM.attachments))
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
