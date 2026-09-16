"""Case Agent session loading node."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.memory.working_memory import (
    RECENT_MESSAGE_LIMIT,
)
from src.backend.application.agent.case_agent.memory.task_state import (
    classify_turn_relation,
    task_state_payload,
)
from src.backend.core.exceptions import ResourceNotFoundError

from ._errors import state_error


def load_session_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Load run, session, recent messages, and the current user message."""

    try:
        run = service._repository.get_run(
            run_id=state["run_id"],
            actor_id=state["actor_id"],
        )
        if run is None:
            raise ResourceNotFoundError("Case Agent 运行不存在")

        service._repository.update_run(
            run.run_id,
            status="running",
            current_node="load_session",
        )
        session = service._repository.get_session(
            session_id=run.session_id,
            actor_id=state["actor_id"],
        )
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        recent = service._repository.list_recent_messages(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            limit=RECENT_MESSAGE_LIMIT,
        )
        user_message = next(
            (message for message in recent if message.message_id == run.user_message_id),
            recent[-1],
        )
        task_state = task_state_payload(getattr(session, "task_state", None))
        turn_relation = classify_turn_relation(
            user_message.content,
            task_state=task_state,
            resume_context=run.resume_context,
        )
        if task_state.get("task_id"):
            task_state = {**task_state, "status": "active"}
        session_memory_context = service.bootstrap_session_memory(
            {
                **state,
                "run": run,
                "session": session,
                "user_message": user_message,
            }
        )
        state.update(
            {
                "run": run,
                "session": session,
                "task_state": task_state,
                "turn_relation": turn_relation,
                "recent_messages": recent,
                "user_message": user_message,
                "capability_results": [],
                "available_sources": [],
                "session_memory_context": session_memory_context,
                "validation_retry_count": state.get("validation_retry_count", 0),
                "next_action": "fast_rule_entity_perception",
            }
        )
        return state
    except Exception as exc:
        return state_error(state, exc)
