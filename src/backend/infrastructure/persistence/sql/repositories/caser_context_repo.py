"""PostgreSQL repository for Caser context sections."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.backend.application.ports.caser_context import CaserContextRepository
from src.backend.domain.caser_context.entities import CaserContextSection

from ..models import CaseORM, CaserContextSectionORM
from ..session import session_scope


class SqlCaserContextRepository(CaserContextRepository):
    """Persist Caser read-model sections in PostgreSQL."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def upsert_section(
        self,
        *,
        case_id: str,
        section_key: str,
        schema_version: str,
        status: str,
        completeness: str,
        payload: dict[str, Any],
        payload_hash: str,
        source_refs: list[str],
        source_versions: dict[str, str],
        generated_by: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> CaserContextSection:
        with session_scope(self._session_factory) as session:
            case_row = self._case_row(session, case_id)
            if case_row is None:
                raise KeyError(case_id)
            row = self._section_row(session, case_row.id, section_key, lock=True)
            now = datetime.now(timezone.utc)
            if row is None:
                row = CaserContextSectionORM(
                    case_id=case_row.id,
                    section_key=section_key,
                    schema_version=schema_version,
                    status=status,
                    completeness=completeness,
                    payload=payload,
                    payload_hash=payload_hash,
                    source_refs=list(dict.fromkeys(source_refs)),
                    source_versions=source_versions,
                    generated_by=generated_by,
                    generated_at=now,
                    refreshed_at=now,
                    error_code=error_code,
                    error_message=(error_message or "")[:1000] if error_message else None,
                )
                session.add(row)
            else:
                row.schema_version = schema_version
                row.status = status
                row.completeness = completeness
                row.payload = payload
                row.payload_hash = payload_hash
                row.source_refs = list(dict.fromkeys(source_refs))
                row.source_versions = source_versions
                row.generated_by = generated_by
                row.generated_at = now
                row.refreshed_at = now
                row.error_code = error_code
                row.error_message = (error_message or "")[:1000] if error_message else None
            session.flush()
            return self._to_section(session, row)

    def mark_sections_building(
        self,
        *,
        case_id: str,
        section_keys: list[str],
        generated_by: str,
    ) -> None:
        if not section_keys:
            return
        with session_scope(self._session_factory) as session:
            case_row = self._case_row(session, case_id)
            if case_row is None:
                raise KeyError(case_id)
            now = datetime.now(timezone.utc)
            for section_key in section_keys:
                row = self._section_row(session, case_row.id, section_key, lock=True)
                if row is None:
                    session.add(
                        CaserContextSectionORM(
                            case_id=case_row.id,
                            section_key=section_key,
                            schema_version="caser-context-v1",
                            status="building",
                            completeness="not_available",
                            payload={},
                            payload_hash="",
                            source_refs=[],
                            source_versions={},
                            generated_by=generated_by,
                            generated_at=now,
                            refreshed_at=now,
                        )
                    )
                else:
                    row.status = "building"
                    row.completeness = "not_available"
                    row.generated_by = generated_by
                    row.refreshed_at = now

    def get_section(self, *, case_id: str, section_key: str) -> CaserContextSection | None:
        with self._session_factory() as session:
            case_row = self._case_row(session, case_id)
            if case_row is None:
                return None
            row = self._section_row(session, case_row.id, section_key, lock=False)
            return self._to_section(session, row) if row is not None else None

    def list_sections(self, *, case_id: str) -> list[CaserContextSection]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(CaserContextSectionORM)
                .join(CaseORM, CaseORM.id == CaserContextSectionORM.case_id)
                .where(CaseORM.case_id == case_id)
                .order_by(CaserContextSectionORM.section_key)
            ).all()
            return [self._to_section(session, row) for row in rows]

    @staticmethod
    def _case_row(session: Session, case_id: str) -> CaseORM | None:
        return session.scalar(select(CaseORM).where(CaseORM.case_id == case_id))

    @staticmethod
    def _section_row(
        session: Session,
        internal_case_id: Any,
        section_key: str,
        *,
        lock: bool,
    ) -> CaserContextSectionORM | None:
        statement = select(CaserContextSectionORM).where(
            CaserContextSectionORM.case_id == internal_case_id,
            CaserContextSectionORM.section_key == section_key,
        )
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    @staticmethod
    def _to_section(session: Session, row: CaserContextSectionORM) -> CaserContextSection:
        case_id = session.scalar(select(CaseORM.case_id).where(CaseORM.id == row.case_id))
        return CaserContextSection(
            case_id=str(case_id or ""),
            section_key=row.section_key,
            schema_version=row.schema_version,
            status=row.status,
            completeness=row.completeness,
            payload=dict(row.payload or {}),
            payload_hash=row.payload_hash,
            source_refs=list(row.source_refs or []),
            source_versions=dict(row.source_versions or {}),
            generated_by=row.generated_by,
            generated_at=row.generated_at,
            refreshed_at=row.refreshed_at,
            error_code=row.error_code,
            error_message=row.error_message,
        )
