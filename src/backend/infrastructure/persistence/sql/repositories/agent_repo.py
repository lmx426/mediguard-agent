"""PostgreSQL repository for formal Evidence Agent state — agent_repo."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .....application.ports.agent import EvidenceAgentRepository
from .....domain.agent.entities import (
    AgentEvent,
    AgentRun,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
)
from ..models import (
    AgentCheckpointORM,
    AgentEventORM,
    AgentEvidenceAnalysisORM,
    AgentEvidenceCitationORM,
    AgentRunORM,
    AgentToolCallORM,
    CaseORM,
)
from ..session import session_scope


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SqlEvidenceAgentRepository(EvidenceAgentRepository):
    """Store every formal Agent artifact in PostgreSQL."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_or_reuse_run(
        self,
        *,
        case_id: str,
        actor_id: str,
        request: EvidenceAgentRequest,
        input_fingerprint: str,
        config_fingerprint: str,
        model_name: str,
        prompt_version: str,
        tool_version: str,
    ) -> AgentRun:
        scope = {
            "case_id": case_id,
            "analysis_type": request.analysis_type,
            "input_fingerprint": input_fingerprint,
            "config_fingerprint": config_fingerprint,
        }
        idempotency_key = _canonical_hash(scope)
        with session_scope(self._session_factory) as session:
            case_row = session.scalar(
                select(CaseORM)
                .where(CaseORM.case_id == case_id)
                .with_for_update()
            )
            if case_row is None:
                raise KeyError(case_id)

            existing = session.scalar(
                select(AgentRunORM).where(AgentRunORM.idempotency_key == idempotency_key)
            )
            if existing is not None and existing.status != "failed":
                return self._to_run(session, existing, reused=True)
            if existing is not None:
                existing.idempotency_key = _canonical_hash(
                    {"failed_run": existing.run_id, "previous": idempotency_key}
                )
                session.flush()

            active = session.scalar(
                select(AgentRunORM)
                .where(
                    AgentRunORM.case_id == case_row.id,
                    AgentRunORM.analysis_type == request.analysis_type,
                    AgentRunORM.status.in_(["queued", "running"]),
                )
                .with_for_update()
            )
            if active is not None:
                return self._to_run(session, active, reused=True)

            row = AgentRunORM(
                run_id=f"arun_{uuid4().hex}",
                case_id=case_row.id,
                actor_id=UUID(actor_id),
                analysis_type=request.analysis_type,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                idempotency_key=idempotency_key,
                status="queued",
                current_node="queued",
                model_name=model_name,
                prompt_version=prompt_version,
                tool_version=tool_version,
            )
            session.add(row)
            session.flush()
            return self._to_run(session, row)

    def get_run(self, run_id: str) -> AgentRun | None:
        with self._session_factory() as session:
            row = self._run_row(session, run_id)
            return self._to_run(session, row) if row is not None else None

    def get_latest_run(
        self,
        case_id: str,
        analysis_type: str = "comprehensive",
    ) -> AgentRun | None:
        """Return the active run for a case, falling back to the latest run."""

        with self._session_factory() as session:
            base = (
                select(AgentRunORM)
                .join(CaseORM, CaseORM.id == AgentRunORM.case_id)
                .where(
                    CaseORM.case_id == case_id,
                    AgentRunORM.analysis_type == analysis_type,
                )
            )
            active = session.scalar(
                base.where(AgentRunORM.status.in_(["queued", "running"]))
                .order_by(AgentRunORM.created_at.desc())
                .limit(1)
            )
            if active is not None:
                return self._to_run(session, active)
            latest = session.scalar(base.order_by(AgentRunORM.created_at.desc()).limit(1))
            return self._to_run(session, latest) if latest is not None else None

    def get_current_analysis(
        self,
        case_id: str,
    ) -> EvidenceAgentAnalysis | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(AgentEvidenceAnalysisORM)
                .join(CaseORM, CaseORM.id == AgentEvidenceAnalysisORM.case_id)
                .where(
                    CaseORM.case_id == case_id,
                    AgentEvidenceAnalysisORM.analysis_type == "comprehensive",
                    AgentEvidenceAnalysisORM.is_current.is_(True),
                )
                .order_by(AgentEvidenceAnalysisORM.created_at.desc())
            )
            return EvidenceAgentAnalysis.model_validate(row.payload) if row is not None else None

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_node: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        model_call_count: int | None = None,
        tool_call_count: int | None = None,
    ) -> None:
        with session_scope(self._session_factory) as session:
            row = self._run_row(session, run_id, lock=True)
            if row is None:
                raise KeyError(run_id)
            now = datetime.now(timezone.utc)
            if status is not None:
                row.status = status
                if status == "running" and row.started_at is None:
                    row.started_at = now
                if status in {"complete", "partial", "failed"}:
                    row.completed_at = now
            if current_node is not None:
                row.current_node = current_node
            row.error_code = error_code
            row.error_message = error_message
            if model_call_count is not None:
                row.model_call_count = model_call_count
            if tool_call_count is not None:
                row.tool_call_count = tool_call_count

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(AgentEventORM.sequence), 0)).where(
                        AgentEventORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            row = AgentEventORM(
                run_id=run.id,
                sequence=sequence,
                event_type=event_type,
                message=message,
                payload=payload or {},
            )
            session.add(row)
            session.flush()
            return AgentEvent(
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                message=message,
                payload=row.payload,
                created_at=row.created_at,
            )

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[AgentEvent]:
        with self._session_factory() as session:
            run = self._run_row(session, run_id)
            if run is None:
                return []
            rows = session.scalars(
                select(AgentEventORM)
                .where(
                    AgentEventORM.run_id == run.id,
                    AgentEventORM.sequence > after_sequence,
                )
                .order_by(AgentEventORM.sequence)
            ).all()
            return [
                AgentEvent(
                    run_id=run_id,
                    sequence=row.sequence,
                    event_type=row.event_type,
                    message=row.message,
                    payload=row.payload,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def save_checkpoint(
        self,
        run_id: str,
        node_name: str,
        state: dict[str, Any],
        safe_to_resume: bool = True,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(AgentCheckpointORM.sequence), 0)).where(
                        AgentCheckpointORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            session.add(
                AgentCheckpointORM(
                    run_id=run.id,
                    sequence=sequence,
                    node_name=node_name,
                    state_json=state,
                    state_hash=_canonical_hash(state),
                    safe_to_resume=safe_to_resume,
                )
            )

    def record_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any] | None,
        status: str,
        error_code: str | None,
        latency_ms: int,
    ) -> None:
        signature = _canonical_hash({"tool": tool_name, "arguments": arguments})
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, run_id, lock=True)
            if run is None:
                raise KeyError(run_id)
            existing = session.scalar(
                select(AgentToolCallORM).where(
                    AgentToolCallORM.run_id == run.id,
                    AgentToolCallORM.signature == signature,
                )
            )
            if existing is not None:
                return
            sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(AgentToolCallORM.sequence), 0)).where(
                        AgentToolCallORM.run_id == run.id
                    )
                )
                or 0
            ) + 1
            session.add(
                AgentToolCallORM(
                    run_id=run.id,
                    sequence=sequence,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    signature=signature,
                    arguments=arguments,
                    result=result,
                    result_hash=_canonical_hash(result) if result is not None else None,
                    status=status,
                    error_code=error_code,
                    latency_ms=latency_ms,
                )
            )

    def find_tool_result(self, run_id: str, signature: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            run = self._run_row(session, run_id)
            if run is None:
                return None
            row = session.scalar(
                select(AgentToolCallORM).where(
                    AgentToolCallORM.run_id == run.id,
                    AgentToolCallORM.signature == signature,
                    AgentToolCallORM.status == "success",
                )
            )
            return row.result if row is not None else None

    def save_analysis(self, analysis: EvidenceAgentAnalysis) -> None:
        with session_scope(self._session_factory) as session:
            run = self._run_row(session, analysis.run_id, lock=True)
            if run is None:
                raise KeyError(analysis.run_id)
            existing = session.scalar(
                select(AgentEvidenceAnalysisORM).where(AgentEvidenceAnalysisORM.run_id == run.id)
            )
            if existing is not None:
                return
            session.execute(
                update(AgentEvidenceAnalysisORM)
                .where(
                    AgentEvidenceAnalysisORM.case_id == run.case_id,
                    AgentEvidenceAnalysisORM.analysis_type == analysis.analysis_type,
                    AgentEvidenceAnalysisORM.is_current.is_(True),
                )
                .values(is_current=False)
            )
            row = AgentEvidenceAnalysisORM(
                analysis_id=analysis.analysis_id,
                run_id=run.id,
                case_id=run.case_id,
                analysis_type=analysis.analysis_type,
                status=analysis.status,
                input_fingerprint=analysis.input_fingerprint,
                is_current=True,
                payload=analysis.model_dump(mode="json"),
                generated_notice=analysis.generated_notice,
                model_name=analysis.model_name,
                prompt_version=analysis.prompt_version,
                tool_version=analysis.tool_version,
                created_at=analysis.created_at,
            )
            session.add(row)
            session.flush()
            for citation in analysis.citations:
                session.add(
                    AgentEvidenceCitationORM(
                        analysis_id=row.id,
                        citation_id=citation.citation_id,
                        source_type=citation.source_type,
                        source_ref=citation.source_ref,
                        label=citation.label,
                        version=citation.version,
                        current_value=citation.current_value,
                        threshold=citation.threshold,
                        metadata_json=citation.metadata,
                    )
                )

    @staticmethod
    def _run_row(session: Session, run_id: str, lock: bool = False) -> AgentRunORM | None:
        statement = select(AgentRunORM).where(AgentRunORM.run_id == run_id)
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    def _to_run(self, session: Session, row: AgentRunORM, reused: bool = False) -> AgentRun:
        case_id = session.scalar(select(CaseORM.case_id).where(CaseORM.id == row.case_id))
        analysis_row = session.scalar(
            select(AgentEvidenceAnalysisORM).where(AgentEvidenceAnalysisORM.run_id == row.id)
        )
        analysis = (
            EvidenceAgentAnalysis.model_validate(analysis_row.payload)
            if analysis_row is not None
            else None
        )
        return AgentRun(
            run_id=row.run_id,
            case_id=case_id or "",
            actor_id=str(row.actor_id),
            analysis_type=row.analysis_type,
            status=row.status,
            current_node=row.current_node,
            input_fingerprint=row.input_fingerprint,
            reused=reused,
            error_code=row.error_code,
            error_message=row.error_message,
            model_call_count=row.model_call_count,
            tool_call_count=row.tool_call_count,
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
            analysis=analysis,
        )
