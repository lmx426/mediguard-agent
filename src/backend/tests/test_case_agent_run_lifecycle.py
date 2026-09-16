"""Focused tests for Case Agent concurrent-run cancellation boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.backend.application.agent.case_agent.nodes.persist_result import persist_result_node
from src.backend.application.agent.case_agent.service.note_service import CaseAgentNoteService
from src.backend.application.agent.case_agent.service.orchestrator import (
    CaseAgentRunCancelled,
    CaseAgentService,
)
from src.backend.core.exceptions import ResourceConflictError
from src.backend.domain.audit.review.entities import AuthenticatedUser
from src.backend.domain.case_agent.entities import (
    CaseAgentAdoptNoteInput,
    CaseAgentAnswer,
    CaseAgentContentBlock,
    CaseAgentMessage,
    CaseAgentSession,
)


def test_node_wrapper_stops_before_cancelled_node_executes() -> None:
    class Repository:
        def is_run_cancel_requested(self, run_id: str) -> bool:
            return run_id == "crun_cancelled"

    called = False
    service = CaseAgentService.__new__(CaseAgentService)
    service._repository = Repository()

    def node(state):
        nonlocal called
        called = True
        return state

    with pytest.raises(CaseAgentRunCancelled):
        service._invoke_node(
            "generate_answer",
            node,
            {"run_id": "crun_cancelled", "actor_id": "actor"},
        )

    assert called is False


def test_persist_result_drops_answer_when_cancel_wins_finalization() -> None:
    class Repository:
        def __init__(self) -> None:
            self.events: list[tuple[str, str]] = []
            self.memory_updated = False

        def update_run(self, run_id: str, **kwargs) -> None:
            assert run_id == "crun_1"

        def finalize_run_with_answer(self, **kwargs):
            assert kwargs["run_id"] == "crun_1"
            return None

        def is_run_cancel_requested(self, run_id: str) -> bool:
            return run_id == "crun_1"

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append((run_id, event_type))

        def update_working_memory(self, **kwargs) -> None:
            self.memory_updated = True

    repository = Repository()
    service = SimpleNamespace(
        _repository=repository,
        _answer_source_refs=lambda _answer: [],
        case_context_fingerprint=lambda _case_id: None,
    )
    run = SimpleNamespace(
        run_id="crun_1",
        session_id="csess_1",
        case_id="CASE-1",
        case_context_fingerprint=None,
    )
    session = SimpleNamespace(session_id="csess_1", session_summary="")
    state = {
        "run_id": run.run_id,
        "actor_id": "actor",
        "run": run,
        "session": session,
        "user_message": SimpleNamespace(content="问题", active_stage="evidence_package"),
        "answer": CaseAgentAnswer(content_blocks=[CaseAgentContentBlock(text="不应保存")]),
        "completion_status": "completed",
    }

    result = persist_result_node(service, state)

    assert result["completion_status"] == "cancelled"
    assert repository.memory_updated is False
    assert repository.events == [("crun_1", "cancelled")]


def test_stale_answer_cannot_be_adopted_as_work_note() -> None:
    now = datetime.now(timezone.utc)
    answer = CaseAgentAnswer(
        content_blocks=[CaseAgentContentBlock(text="旧版本回答")],
        metadata={"case_context_stale": True},
    )
    message = CaseAgentMessage(
        message_id="cmsg_1",
        session_id="csess_1",
        role="assistant",
        content="旧版本回答",
        answer_payload=answer,
        created_at=now,
    )
    session = CaseAgentSession(
        session_id="csess_1",
        case_id="CASE-1",
        actor_id="actor",
        title="会话",
        created_at=now,
        updated_at=now,
        messages=[message],
    )

    class Repository:
        def get_session(self, **kwargs):
            return session

    class Notes:
        def add(self, *args, **kwargs):
            raise AssertionError("stale answer must not be saved")

    service = CaseAgentNoteService(repository=Repository(), manage_notes=Notes())
    actor = AuthenticatedUser(
        id="actor",
        username="auditor",
        display_name="审核员",
        roles=["auditor"],
    )

    with pytest.raises(ResourceConflictError) as exc_info:
        service.adopt_note(
            "csess_1",
            "cmsg_1",
            CaseAgentAdoptNoteInput(content="工作笔记", source_refs=[]),
            actor,
        )

    assert exc_info.value.code == "case_agent_answer_stale"


def test_service_restart_requeues_persisted_active_runs() -> None:
    class Repository:
        def __init__(self) -> None:
            self.updates: list[tuple[str, dict[str, str]]] = []
            self.events: list[tuple[str, str, dict[str, str]]] = []

        def list_recoverable_runs(self):
            return [
                SimpleNamespace(
                    run_id="crun_recover",
                    actor_id="actor",
                    status="running",
                    cancel_requested_at=None,
                )
            ]

        def update_run(self, run_id: str, **kwargs) -> None:
            self.updates.append((run_id, kwargs))

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append((run_id, event_type, payload or {}))

    repository = Repository()
    submitted: list[tuple[str, str]] = []
    service = CaseAgentService.__new__(CaseAgentService)
    service._repository = repository
    service._submit_run = lambda run_id, actor_id: submitted.append((run_id, actor_id))

    recovered = service.recover_persisted_runs()

    assert recovered == 1
    assert repository.updates == [
        ("crun_recover", {"status": "created", "current_node": "recovery_queued"})
    ]
    assert repository.events == [
        ("crun_recover", "recovered", {"previous_status": "running"})
    ]
    assert submitted == [("crun_recover", "actor")]
