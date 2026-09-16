"""PostgreSQL-backed Trace repository — traces_repo."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from .....domain.audit.review.workflow_projector import ReviewDecision, TraceNode, WorkflowResponse, build_workflow_response
from ..models import CaseORM, WorkflowTraceNodeORM
from ..session import session_scope


class SqlTraceRepository:
    """Persist Trace nodes and rebuild the nine-stage workflow from PostgreSQL."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def clear(self) -> None:
        """Delete all Trace rows. Intended for tests."""

        with session_scope(self._session_factory) as session:
            session.execute(delete(WorkflowTraceNodeORM))

    def init_from_fixture(self, case_id: str, initial_nodes: list[TraceNode]) -> None:
        """Replace a case Trace timeline with initial nodes."""

        self.init_many([(case_id, initial_nodes)])

    def init_many(self, entries: list[tuple[str, list[TraceNode]]]) -> None:
        """Persist multiple Trace timelines in one transaction."""

        with session_scope(self._session_factory) as session:
            for public_case_id, nodes in entries:
                case = self._case(session, public_case_id)
                if case is None:
                    continue
                session.execute(
                    delete(WorkflowTraceNodeORM).where(
                        WorkflowTraceNodeORM.case_id == case.id
                    )
                )
                for node in nodes:
                    session.add(self._from_node(case.id, node))

    def get_trace(self, case_id: str) -> list[TraceNode]:
        """Return all Trace nodes for a public case id."""

        with self._session_factory() as session:
            case = self._case(session, case_id)
            if case is None:
                return []
            rows = session.scalars(
                select(WorkflowTraceNodeORM)
                .where(WorkflowTraceNodeORM.case_id == case.id)
                .order_by(WorkflowTraceNodeORM.node_order)
            ).all()
            return [self._to_node(row) for row in rows]

    def append_node(
        self,
        case_id: str,
        node_id: str,
        node_name: str,
        summary: str,
    ) -> TraceNode:
        """Append one completed Trace node to a case timeline."""

        with session_scope(self._session_factory) as session:
            case = self._case(session, case_id)
            if case is None:
                raise ValueError(f"case {case_id} does not exist")
            next_order = (
                session.scalar(
                    select(func.max(WorkflowTraceNodeORM.node_order)).where(
                        WorkflowTraceNodeORM.case_id == case.id
                    )
                )
                or 0
            ) + 1
            node = TraceNode(
                node_id=node_id,
                node_name=node_name,
                status="completed",
                summary=summary,
                order=next_order,
            )
            session.add(self._from_node(case.id, node))
            return node

    def update_pending_to_completed(
        self,
        case_id: str,
        pending_node_id: str,
    ) -> TraceNode | None:
        """Mark a pending Trace node as completed."""

        with session_scope(self._session_factory) as session:
            case = self._case(session, case_id)
            if case is None:
                return None
            row = session.scalar(
                select(WorkflowTraceNodeORM).where(
                    WorkflowTraceNodeORM.case_id == case.id,
                    WorkflowTraceNodeORM.node_id == pending_node_id,
                    WorkflowTraceNodeORM.status == "pending",
                )
            )
            if row is None:
                return None
            row.status = "completed"
            return self._to_node(row)

    def build_workflow(
        self,
        case_id: str,
        review: ReviewDecision | None = None,
    ) -> WorkflowResponse:
        """Build the frontend workflow response from stored Trace nodes."""

        return build_workflow_response(case_id, self.get_trace(case_id), review)

    @staticmethod
    def _from_node(case_uuid, node: TraceNode) -> WorkflowTraceNodeORM:
        return WorkflowTraceNodeORM(
            case_id=case_uuid,
            node_id=node.node_id,
            node_name=node.node_name,
            status=node.status,
            summary=node.summary,
            node_order=node.order,
            metadata_json=node.metadata,
        )

    @staticmethod
    def _to_node(row: WorkflowTraceNodeORM) -> TraceNode:
        return TraceNode(
            node_id=row.node_id,
            node_name=row.node_name,
            status=row.status,
            summary=row.summary,
            order=row.node_order,
            metadata=row.metadata_json or {},
        )

    @staticmethod
    def _case(session: Session, case_id: str) -> CaseORM | None:
        return session.scalar(select(CaseORM).where(CaseORM.case_id == case_id))
