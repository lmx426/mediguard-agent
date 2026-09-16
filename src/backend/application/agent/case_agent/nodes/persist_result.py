"""Case Agent result persistence node."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.memory.summary import next_summary
from src.backend.application.agent.case_agent.memory.task_state import (
    build_next_task_state,
)


def persist_result_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Persist assistant message, working memory and final run status."""

    answer = state["answer"]
    run = state.get("run")
    if run is None:
        run = service._repository.get_run(
            run_id=state["run_id"],
            actor_id=state["actor_id"],
        )
        state["run"] = run
    if run is None:
        return state

    session = state.get("session")
    if session is None:
        session = service._repository.get_session(
            session_id=run.session_id,
            actor_id=state["actor_id"],
        )
        state["session"] = session
    if session is None:
        service._repository.update_run(
            run.run_id,
            status="failed",
            current_node="failed",
            error_code=state.get("error_code", "session_not_found"),
            error_message=state.get("error_message", "Case Agent 会话不存在")[:500],
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        return state

    service._repository.update_run(run.run_id, current_node="persisting")
    completion_status = state.get("completion_status")
    failed = completion_status == "failed"
    waiting_for_user = completion_status == "waiting_for_user"
    degraded = completion_status == "degraded"
    user_message = state.get("user_message")
    run_context_fingerprint = getattr(run, "case_context_fingerprint", None)
    fingerprint_loader = getattr(service, "case_context_fingerprint", None)
    if run_context_fingerprint and callable(fingerprint_loader):
        current_fingerprint = fingerprint_loader(run.case_id)
        if current_fingerprint and current_fingerprint != run_context_fingerprint:
            answer.metadata = {**answer.metadata, "case_context_stale": True}
            answer.fallback_notice = (
                "案件信息在任务执行期间已更新；本回答基于较早的案件上下文，请重新核验后再采纳。"
            )
    answer_source_refs = [] if failed else service._answer_source_refs(answer)
    final_status = (
        "waiting_for_user"
        if waiting_for_user
        else "degraded"
        if degraded
        else "failed"
        if failed
        else "completed"
    )
    final_node = {
        "waiting_for_user": "waiting_for_user",
        "degraded": "degraded",
        "failed": "failed",
        "completed": "complete",
    }[final_status]
    finalize = getattr(service._repository, "finalize_run_with_answer", None)
    finalized_atomically = callable(finalize)
    if finalized_atomically:
        assistant = finalize(
            run_id=run.run_id,
            actor_id=state["actor_id"],
            content=answer.plain_text,
            active_stage=None if failed or user_message is None else user_message.active_stage,
            answer_payload=answer,
            source_refs=answer_source_refs,
            status=final_status,
            current_node=final_node,
            error_code=(state.get("error_code", "CaseAgentRuntimeError") if failed or degraded else None),
            error_message=(state.get("error_message", "")[:500] if failed or degraded else None),
            degraded_reason=(state.get("error_code", "CaseAgentRuntimeError") if degraded else None),
            pending_clarification=state.get("pending_clarification", {}) if waiting_for_user else {},
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        if assistant is None:
            if service._repository.is_run_cancel_requested(run.run_id):
                service._repository.append_event(
                    run.run_id,
                    "cancelled",
                    "任务已停止，未生成或保存助手回答",
                    {"reason": "user_requested"},
                )
                state["completion_status"] = "cancelled"
            return state
    else:
        assistant = service._repository.append_assistant_message(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            content=answer.plain_text,
            active_stage=None if failed or user_message is None else user_message.active_stage,
            answer_payload=answer,
            source_refs=answer_source_refs,
        )
    answer_ref = f"answer:{assistant.message_id}"
    if waiting_for_user and user_message is not None:
        service._repository.update_working_memory(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            current_topic=user_message.content[:120],
            pending_tool="clarification",
            referenced_source_refs=answer_source_refs,
            task_state=build_next_task_state(
                state,
                answer_ref="",
                source_refs=answer_source_refs,
                status="waiting_user",
            ),
        )
        if not finalized_atomically:
            service._repository.update_run(
                run.run_id,
                status="waiting_for_user",
                current_node="waiting_for_user",
                assistant_message_id=assistant.message_id,
                pending_clarification=state.get("pending_clarification", {}),
                model_call_count=state.get("model_call_count", 0),
                tool_call_count=state.get("tool_call_count", 0),
            )
        service._repository.append_event(
            run.run_id,
            "waiting_for_user",
            "等待审核人员补充信息",
            {
                "assistant_message_id": assistant.message_id,
                "pending_clarification": state.get("pending_clarification", {}),
            },
        )
    elif not failed and not degraded and user_message is not None:
        service._repository.update_working_memory(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            session_summary=next_summary(
                session.session_summary,
                user_message.content,
                answer.plain_text,
            ),
            current_topic=user_message.content[:120],
            pending_tool="",
            referenced_source_refs=answer_source_refs,
            task_state=build_next_task_state(
                state,
                answer_ref=answer_ref,
                source_refs=answer_source_refs,
                status="completed",
            ),
        )
        if not finalized_atomically:
            service._repository.update_run(
                run.run_id,
                status="completed",
                current_node="complete",
                assistant_message_id=assistant.message_id,
                model_call_count=state.get("model_call_count", 0),
                tool_call_count=state.get("tool_call_count", 0),
            )
        service._repository.append_event(
            run.run_id,
            "complete",
            "结构化回答已生成",
            {"assistant_message_id": assistant.message_id},
        )
        shadow_evaluator = getattr(service, "record_intent_memory_shadow", None)
        if callable(shadow_evaluator):
            try:
                shadow_evaluator(state)
            except Exception:
                pass
        capture = getattr(service, "enqueue_completed_run_memory_capture", None)
        if callable(capture):
            try:
                capture(state)
            except Exception:
                pass
    elif degraded:
        service._repository.update_working_memory(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            task_state=build_next_task_state(
                state,
                answer_ref="",
                source_refs=[],
                status="error",
            ),
        )
        if not finalized_atomically:
            service._repository.update_run(
                run.run_id,
                status="degraded",
                current_node="degraded",
                assistant_message_id=assistant.message_id,
                error_code=state.get("error_code", "CaseAgentRuntimeError"),
                error_message=state.get("error_message", "")[:500],
                degraded_reason=state.get("error_code", "CaseAgentRuntimeError"),
                model_call_count=state.get("model_call_count", 0),
                tool_call_count=state.get("tool_call_count", 0),
            )
        service._repository.append_event(
            run.run_id,
            "degraded",
            "案件助手已安全降级",
            {"error_type": state.get("error_code", "CaseAgentRuntimeError")},
        )
    else:
        service._repository.update_working_memory(
            session_id=session.session_id,
            actor_id=state["actor_id"],
            task_state=build_next_task_state(
                state,
                answer_ref="",
                source_refs=[],
                status="error",
            ),
        )
        if not finalized_atomically:
            service._repository.update_run(
                run.run_id,
                status="failed",
                current_node="failed",
                assistant_message_id=assistant.message_id,
                error_code=state.get("error_code", "CaseAgentRuntimeError"),
                error_message=state.get("error_message", "")[:500],
                model_call_count=state.get("model_call_count", 0),
                tool_call_count=state.get("tool_call_count", 0),
            )
        service._repository.append_event(
            run.run_id,
            "failed",
            "案件助手生成失败，已安全降级",
            {"error_type": state.get("error_code", "CaseAgentRuntimeError")},
        )
    state["assistant_message_id"] = assistant.message_id
    return state
