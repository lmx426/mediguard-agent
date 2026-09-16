"""Case Agent repository port."""

from __future__ import annotations

from typing import Any, Protocol

from ...domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentEvent,
    CaseAgentMessage,
    CaseAgentRun,
    CaseAgentSession,
)


class CaseAgentRepository(Protocol):
    """PostgreSQL-backed Case Agent state."""

    def create_session(
        self,
        *,
        case_id: str,
        actor_id: str,
        title: str,
        model_name: str,
    ) -> CaseAgentSession: ...

    def list_sessions(self, *, case_id: str, actor_id: str) -> list[CaseAgentSession]: ...

    def get_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        include_messages: bool = True,
    ) -> CaseAgentSession | None: ...

    def archive_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> bool: ...

    def restore_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentSession | None: ...

    def rename_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        title: str,
    ) -> CaseAgentSession | None: ...

    def append_user_message(
        self,
        *,
        session_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
    ) -> CaseAgentMessage: ...

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
    ) -> tuple[CaseAgentMessage, CaseAgentRun, bool]: ...

    def append_assistant_message(
        self,
        *,
        session_id: str,
        actor_id: str,
        content: str,
        active_stage: str | None,
        answer_payload: CaseAgentAnswer,
        source_refs: list[str],
    ) -> CaseAgentMessage: ...

    def list_recent_messages(
        self,
        *,
        session_id: str,
        actor_id: str,
        limit: int = 8,
    ) -> list[CaseAgentMessage]: ...

    def create_run(
        self,
        *,
        session_id: str,
        actor_id: str,
        user_message_id: str,
        model_name: str,
        parent_run_id: str | None = None,
        resume_context: dict[str, Any] | None = None,
    ) -> CaseAgentRun: ...

    def get_run(self, *, run_id: str, actor_id: str) -> CaseAgentRun | None: ...

    def get_active_run_for_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentRun | None: ...

    def list_recoverable_runs(self) -> list[CaseAgentRun]: ...

    def request_run_cancel(
        self,
        *,
        run_id: str,
        actor_id: str,
        reason: str,
    ) -> tuple[CaseAgentRun, bool]: ...

    def is_run_cancel_requested(self, run_id: str) -> bool: ...

    def mark_run_cancelled(self, run_id: str) -> bool: ...

    def mark_session_read(
        self,
        *,
        session_id: str,
        actor_id: str,
        through_run_id: str,
    ) -> CaseAgentSession | None: ...

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
    ) -> CaseAgentMessage | None: ...

    def find_waiting_run_for_session(
        self,
        *,
        session_id: str,
        actor_id: str,
    ) -> CaseAgentRun | None: ...

    def mark_run_resumed(
        self,
        *,
        parent_run_id: str,
        resumed_by_run_id: str,
        actor_id: str,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> CaseAgentEvent: ...

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[CaseAgentEvent]: ...

    def start_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> int: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def start_node_execution(
        self,
        run_id: str,
        *,
        node_name: str,
        input_snapshot: dict[str, Any],
    ) -> int: ...

    def finish_node_execution(
        self,
        run_id: str,
        *,
        sequence: int,
        output_snapshot: dict[str, Any],
    ) -> None: ...

    def fail_node_execution(
        self,
        run_id: str,
        *,
        sequence: int,
        error_code: str,
        error_message: str,
        output_snapshot: dict[str, Any] | None = None,
    ) -> None: ...

    def save_checkpoint(
        self,
        run_id: str,
        *,
        node_name: str,
        state_snapshot: dict[str, Any],
        source_refs: list[str],
        safe_to_resume: bool = True,
    ) -> None: ...

    def get_latest_checkpoint(
        self,
        run_id: str,
        *,
        node_name: str | None = None,
    ) -> dict[str, Any] | None: ...
