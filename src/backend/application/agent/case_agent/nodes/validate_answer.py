"""Case Agent answer validation node."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.case_agent.safety.guardrails import (
    CaseAgentSafetyError,
)
from src.backend.application.agent.case_agent.answer_assembler import (
    build_deterministic_answer,
)
from src.backend.domain.case_memory.entities import MemoryType

from ._errors import state_error


REPAIRABLE_ERRORS = {
    "invalid_json",
    "invalid_schema",
    "citation_invalid",
    "citation_missing_for_grounded",
}


def validate_answer_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Validate the model output structure, citations and safety boundaries."""

    try:
        service._repository.update_run(
            state["run"].run_id,
            current_node="validating",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            state["run"].run_id,
            "validating",
            "正在校验结构化回答和引用边界",
            {},
        )
        answer = service._parse_answer(state["final_response"], state)
        state["answer"] = answer
        if state.get("validation_retry_count", 0) > 0 and state.get("validation_error"):
            state["validation_repair_succeeded"] = True
            state.setdefault("validation_recovery_mode", "second_model_call")
        state["completion_status"] = "complete"
        state["next_action"] = "persist_result"
        return state
    except CaseAgentSafetyError as exc:
        if exc.code in REPAIRABLE_ERRORS:
            hint_reader = getattr(service, "memory_hint_for_node", None)
            if callable(hint_reader):
                try:
                    state["failure_recovery_hint"] = hint_reader(
                        state,
                        consumer="recovery_handler",
                        memory_type=MemoryType.FAILURE,
                        task_context={
                            "node": "validate_answer",
                            "validation_codes": [exc.code],
                            "error_category": _error_category(exc.code),
                            "component_version": "case_agent_answer_contract_v1",
                            "trigger_conditions": [
                                "repairable_validation_error",
                                "single_attempt_budget",
                            ],
                        },
                    )
                except Exception:
                    state["failure_recovery_hint"] = {}
        recovered = build_deterministic_answer(state, recovery=True)
        if exc.code in REPAIRABLE_ERRORS and recovered is not None:
            state["answer"] = recovered
            state["validation_error"] = f"{exc.code}: {exc}"
            state["validation_repair_succeeded"] = True
            state["validation_recovery_mode"] = "deterministic_controlled_context"
            state["completion_status"] = "complete"
            state["next_action"] = "persist_result"
            service._repository.append_event(
                state["run"].run_id,
                "answer_recovered_from_controlled_context",
                "模型回答结构异常，已使用受控工具结果恢复",
                {"error_code": exc.code},
            )
            return state
        if (
            exc.code in REPAIRABLE_ERRORS
            and state.get("answer_strategy") != "reuse_previous_answer"
            and state.get("validation_retry_count", 0) < 1
        ):
            state["validation_retry_count"] = state.get("validation_retry_count", 0) + 1
            state["validation_error"] = f"{exc.code}: {exc}"
            state["validation_recovery_mode"] = "second_model_call"
            service._repository.append_event(
                state["run"].run_id,
                "repairing_answer",
                "回答未通过结构或引用校验，正在重写",
                {"error_code": exc.code},
            )
            state["next_action"] = "generate_answer"
            return state
        return state_error(state, exc)
    except Exception as exc:
        return state_error(state, exc)


def _error_category(error_code: str) -> str:
    if error_code in {"invalid_json", "invalid_schema"}:
        return "response_structure"
    if error_code in {"citation_invalid", "citation_missing_for_grounded"}:
        return "citation_integrity"
    return "answer_validation"
