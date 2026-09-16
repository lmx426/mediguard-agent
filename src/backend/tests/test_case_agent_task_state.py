"""Case Agent structured cross-turn task-state tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.backend.application.agent.case_agent.memory.task_state import (
    build_next_task_state,
    classify_turn_relation,
    detect_answer_rewrite_mode,
    is_answer_rewrite_only,
    merge_persisted_task_state,
    task_state_update_is_stale,
)
from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
    _recent_context_for_planner,
)
from src.backend.application.agent.case_agent.nodes.decision import (
    decision_precheck_node,
    plan_normalizer_node,
)
from src.backend.application.agent.case_agent.nodes.generate_answer import (
    generate_answer_node,
)
from src.backend.application.agent.case_agent.nodes.perception import (
    build_perceptual_state_node,
    fast_rule_entity_perception_node,
)
from src.backend.application.agent.case_agent.nodes.validate_execution_plan import (
    validate_execution_plan_node,
)
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentCompletedStep,
    CaseAgentContentBlock,
    CaseAgentSession,
    CaseAgentSource,
    CaseAgentTaskState,
)
from src.backend.application.agent.case_agent.safety import validate_answer
from src.backend.application.agent.runtime.gateways.base import ModelResponse


class _Repository:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def update_run(self, run_id: str, **kwargs) -> None:
        self.run_update = {"run_id": run_id, **kwargs}

    def append_event(self, run_id: str, event_type: str, message: str, payload=None):
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }
        )


def _message(message_id: str, role: str, content: str, *, answer=None):
    return SimpleNamespace(
        message_id=message_id,
        role=role,
        content=content,
        source_refs=["case:source-1"] if answer is not None else [],
        answer_payload=answer,
        active_stage="evidence_review",
        created_at=datetime.now(timezone.utc),
    )


def test_turn_relation_allows_answer_rewrite_during_pending_clarification() -> None:
    task_state = CaseAgentTaskState(
        task_id="ctask_1",
        status="waiting_user",
        pending_clarification={"missing_slots": ["drug_name"]},
    )

    relation = classify_turn_relation(
        "用中文告诉我",
        task_state=task_state,
        resume_context={"pending_clarification": {"missing_slots": ["drug_name"]}},
    )

    assert relation["kind"] == "answer_rewrite"
    assert relation["confidence"] == 0.98


def test_turn_relation_keeps_genuine_pending_clarification_response() -> None:
    task_state = CaseAgentTaskState(
        task_id="ctask_1",
        status="waiting_user",
        pending_clarification={"missing_slots": ["drug_name"]},
    )

    relation = classify_turn_relation(
        "药品名称是阿司匹林",
        task_state=task_state,
        resume_context={"pending_clarification": {"missing_slots": ["drug_name"]}},
    )

    assert relation["kind"] == "clarification_response"
    assert relation["confidence"] == 0.99


def test_turn_relation_detects_translation_rewrite() -> None:
    relation = classify_turn_relation(
        "请用中文告诉我",
        task_state=CaseAgentTaskState(task_id="ctask_1", status="completed"),
    )

    assert relation["kind"] == "answer_rewrite"


def test_answer_rewrite_detection_covers_direct_summary_phrases() -> None:
    cases = {
        "简单回答我": "simplify",
        "简单告诉我": "simplify",
        "简单直接告诉我上述答案": "simplify",
        "请直接告诉我上述答案": "simplify",
        "用个例子告诉我": "example",
        "用个简单例子告诉我": "example",
        "用个例子说一下": "example",
        "用简单例子说明": "example",
        "只说结论": "summarize",
        "用一句话回答上面的问题": "summarize",
        "概括上一轮回答": "summarize",
        "扩充上一轮回答": "expand",
    }

    for text, expected_mode in cases.items():
        assert detect_answer_rewrite_mode(text) == expected_mode
        assert is_answer_rewrite_only(text) is True
        relation = classify_turn_relation(
            text,
            task_state=CaseAgentTaskState(task_id="ctask_1", status="completed"),
        )
        assert relation["kind"] == "answer_rewrite"


def test_answer_rewrite_detection_rejects_new_business_question() -> None:
    text = "请简单回答北京异地就医报销比例是多少"

    assert detect_answer_rewrite_mode(text) == "simplify"
    assert is_answer_rewrite_only(text) is False
    relation = classify_turn_relation(
        text,
        task_state=CaseAgentTaskState(task_id="ctask_1", status="completed"),
    )

    assert relation["kind"] == "continuation"


def test_fast_perception_reuses_task_state_answer_for_chinese_translation() -> None:
    previous_answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="The record was ingested from the upstream wide table.",
                source_refs=["case:source-1"],
            )
        ],
        sources=[],
    )
    task_state = CaseAgentTaskState(
        state_version=3,
        task_id="ctask_1",
        status="completed",
        core_intent="case_task",
        context_snapshot={
            "answer_ref": "answer:msg-answer",
            "source_refs": ["case:source-1"],
        },
    )
    user_message = _message("msg-user", "user", "请用中文告诉我")
    state = {
        "run": SimpleNamespace(
            run_id="crun_translate",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            current_topic="接入方式",
            session_summary="",
            referenced_source_refs=["case:source-1"],
            task_state=task_state,
        ),
        "task_state": task_state.model_dump(mode="json"),
        "turn_relation": {"kind": "answer_rewrite", "confidence": 0.98},
        "user_message": user_message,
        "recent_messages": [
            _message("msg-answer", "assistant", previous_answer.plain_text, answer=previous_answer),
            user_message,
        ],
        "actor_id": "actor-001",
        "model_call_count": 0,
        "tool_call_count": 0,
    }
    service = SimpleNamespace(_repository=_Repository())

    state = fast_rule_entity_perception_node(service, state)
    assert state["next_action"] == "build_perceptual_state"
    state = build_perceptual_state_node(service, state)

    semantic = state["perceptual_state"]["semantic"]
    assert semantic["reuse_answer_ref"] == "answer:msg-answer"
    assert semantic["rewrite_mode"] == "translate_zh"
    assert semantic["requires_new_evidence"] is False
    assert state["model_call_count"] == 0


def test_fast_perception_reuses_previous_answer_for_simple_direct_followup() -> None:
    previous_answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="北京门诊手工报销需要提交费用收据、费用明细和相关诊疗材料。",
                source_refs=["policy-evidence:materials-1"],
            )
        ],
        sources=[],
    )
    task_state = CaseAgentTaskState(
        state_version=4,
        task_id="ctask_policy",
        status="completed",
        core_intent="expert_task",
        context_snapshot={
            "answer_ref": "answer:msg-policy-answer",
            "source_refs": ["policy-evidence:materials-1"],
        },
    )
    user_message = _message(
        "msg-simple-followup",
        "user",
        "简单直接告诉我上述答案",
    )
    previous_message = _message(
        "msg-policy-answer",
        "assistant",
        previous_answer.plain_text,
        answer=previous_answer,
    )
    previous_message.source_refs = ["policy-evidence:materials-1"]
    clarification_answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[CaseAgentContentBlock(text="请补充需要查询的业务内容。")],
        sources=[],
        metadata={
            "intent": "clarification_required",
            "missing_slots": ["business_module"],
        },
    )
    clarification_message = _message(
        "msg-clarification",
        "assistant",
        clarification_answer.plain_text,
        answer=clarification_answer,
    )
    clarification_message.source_refs = []
    state = {
        "run": SimpleNamespace(
            run_id="crun_simple_followup",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            current_topic="北京手工报销",
            session_summary="",
            referenced_source_refs=["policy-evidence:materials-1"],
            task_state=task_state,
        ),
        "task_state": task_state.model_dump(mode="json"),
        "turn_relation": {"kind": "answer_rewrite", "confidence": 0.98},
        "user_message": user_message,
        "recent_messages": [previous_message, clarification_message, user_message],
        "actor_id": "actor-001",
        "model_call_count": 0,
        "tool_call_count": 0,
    }
    repository = _Repository()
    service = SimpleNamespace(_repository=repository)

    state = fast_rule_entity_perception_node(service, state)
    assert state["next_action"] == "build_perceptual_state"
    state = build_perceptual_state_node(service, state)

    semantic = state["perceptual_state"]["semantic"]
    assert semantic["evidence_need"] == "previous_answer"
    assert semantic["rewrite_mode"] == "simplify"
    assert semantic["requires_new_evidence"] is False
    assert semantic["target_layer_hint"] == "none"
    assert semantic["reuse_answer_ref"] == "answer:msg-policy-answer"
    assert semantic["reuse_source_refs"] == ["policy-evidence:materials-1"]
    assert state["model_call_count"] == 0
    assert any(
        event["event_type"] == "fast_rule_entity_perception_complete"
        and event["payload"].get("rule_kind") == "answer_rewrite_followup"
        for event in repository.events
    )

    state = decision_precheck_node(service, state)
    state = plan_normalizer_node(service, state)
    state = validate_execution_plan_node(service, state)

    assert state["answer_strategy"] == "reuse_previous_answer"
    assert state["execution_plan"] == []
    assert state["next_action"] == "build_answer_context"
    assert state["model_call_count"] == 0
    assert state["tool_call_count"] == 0


def test_fast_perception_uses_pending_clarification_slot() -> None:
    task_state = CaseAgentTaskState(
        state_version=2,
        task_id="ctask_1",
        status="waiting_user",
        core_intent="expert_task",
        pending_clarification={"missing_slots": ["drug_name"]},
    )
    user_message = _message("msg-user", "user", "阿司匹林")
    pending = {
        "intent": "expert_task",
        "missing_slots": ["drug_name"],
        "slots": {},
        "semantic": {
            "intent": "expert_task",
            "action": "query_policy_basis",
            "answer_shape": "analysis",
            "target_layer_hint": "L3",
            "target_objects": ["drug"],
            "information_needs": ["用药政策依据"],
            "rewritten_query": "查询药品用药政策依据",
            "evidence_need": "expert_evidence",
        },
    }
    state = {
        "run": SimpleNamespace(
            run_id="crun_clarification",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            current_topic="药品政策",
            session_summary="",
            referenced_source_refs=[],
            task_state=task_state,
        ),
        "task_state": task_state.model_dump(mode="json"),
        "turn_relation": {"kind": "clarification_response", "confidence": 0.99},
        "resume_context": {"pending_clarification": pending},
        "user_message": user_message,
        "recent_messages": [user_message],
        "actor_id": "actor-001",
        "model_call_count": 0,
        "tool_call_count": 0,
    }
    service = SimpleNamespace(_repository=_Repository())

    state = fast_rule_entity_perception_node(service, state)
    state = build_perceptual_state_node(service, state)

    assert state["slots"]["drug_name"] == "阿司匹林"
    assert state["missing_slots"] == []
    assert state["perceptual_state"]["semantic"]["target_layer_hint"] == "L3"
    assert state["model_call_count"] == 0


def test_task_state_rewrite_preserves_task_and_updates_answer_ref() -> None:
    now = datetime.now(timezone.utc)
    current = CaseAgentTaskState(
        state_version=4,
        task_id="ctask_existing",
        status="completed",
        core_intent="case_task",
        goal="查询案件接入方式",
        completed_steps=[
            CaseAgentCompletedStep(
                message_id="msg-old",
                intent="case_task",
                subtask="query_case_basic_info",
                answer_ref="answer:msg-old-answer",
                completed_at=now,
            )
        ],
        context_snapshot={"answer_ref": "answer:msg-old-answer"},
    )
    state = {
        "session": SimpleNamespace(task_state=current),
        "turn_relation": {"kind": "answer_rewrite"},
        "user_message": SimpleNamespace(
            message_id="msg-new",
            created_at=now + timedelta(seconds=1),
            active_stage="case_intake",
        ),
        "intent": "case_task",
        "answer_strategy": "reuse_previous_answer",
        "answer_rewrite_mode": "translate_zh",
        "execution_plan": [],
        "completion_status": "complete",
    }

    result = CaseAgentTaskState.model_validate(
        build_next_task_state(
            state,
            answer_ref="answer:msg-new-answer",
            source_refs=["case:source-1"],
            status="completed",
        )
    )

    assert result.task_id == "ctask_existing"
    assert result.core_intent == "case_task"
    assert result.current_subtask == "rewrite:translate_zh"
    assert result.context_snapshot["answer_ref"] == "answer:msg-new-answer"
    assert len(result.completed_steps) == 2


def test_task_state_correction_invalidates_old_completed_steps() -> None:
    now = datetime.now(timezone.utc)
    current = CaseAgentTaskState(
        state_version=2,
        task_id="ctask_existing",
        status="completed",
        core_intent="case_task",
        completed_steps=[
            CaseAgentCompletedStep(
                message_id="msg-old",
                answer_ref="answer:msg-old-answer",
                completed_at=now,
            )
        ],
        context_snapshot={"answer_ref": "answer:msg-old-answer"},
    )
    state = {
        "session": SimpleNamespace(task_state=current),
        "turn_relation": {"kind": "correction"},
        "user_message": SimpleNamespace(
            message_id="msg-correction",
            created_at=now + timedelta(seconds=1),
            active_stage="case_intake",
        ),
        "intent": "case_task",
        "execution_plan": [{"capability": "query_case_basic_info"}],
        "completion_status": "complete",
    }

    result = CaseAgentTaskState.model_validate(
        build_next_task_state(
            state,
            answer_ref="answer:msg-corrected",
            source_refs=["case:source-2"],
            status="completed",
        )
    )

    assert result.task_id == "ctask_existing"
    assert len(result.completed_steps) == 1
    assert result.completed_steps[0].answer_ref == "answer:msg-corrected"
    assert result.context_snapshot["source_refs"] == ["case:source-2"]


def test_persisted_task_state_rejects_older_turn_overwrite() -> None:
    now = datetime.now(timezone.utc)
    current = CaseAgentTaskState(
        state_version=5,
        task_id="ctask_new",
        last_user_message_id="msg-new",
        last_turn_created_at=now,
    )
    incoming = CaseAgentTaskState(
        state_version=2,
        task_id="ctask_old",
        last_user_message_id="msg-old",
        last_turn_created_at=now - timedelta(seconds=10),
    )

    merged = CaseAgentTaskState.model_validate(
        merge_persisted_task_state(current, incoming)
    )

    assert merged.task_id == "ctask_new"
    assert merged.state_version == 5
    assert merged.last_user_message_id == "msg-new"
    assert task_state_update_is_stale(current, incoming) is True


def test_persisted_task_state_increments_version_for_newer_turn() -> None:
    now = datetime.now(timezone.utc)
    current = CaseAgentTaskState(
        state_version=5,
        task_id="ctask_current",
        last_turn_created_at=now,
    )
    incoming = CaseAgentTaskState(
        state_version=2,
        task_id="ctask_current",
        last_turn_created_at=now + timedelta(seconds=1),
    )

    merged = CaseAgentTaskState.model_validate(
        merge_persisted_task_state(current, incoming)
    )

    assert merged.state_version == 6


def test_session_task_state_is_not_exposed_by_api_serialization() -> None:
    now = datetime.now(timezone.utc)
    session = CaseAgentSession(
        session_id="csess_1",
        case_id="CASE-001",
        actor_id="actor-1",
        title="测试会话",
        task_state=CaseAgentTaskState(task_id="ctask_private"),
        created_at=now,
        updated_at=now,
    )

    assert "task_state" not in session.model_dump(mode="json")


def test_planner_context_uses_two_messages_when_task_state_is_available() -> None:
    task_state = CaseAgentTaskState(
        state_version=1,
        task_id="ctask_1",
        status="active",
        core_intent="case_task",
    )
    messages = [
        _message(f"msg-{index}", "user" if index % 2 == 0 else "assistant", f"消息 {index}")
        for index in range(6)
    ]
    session = SimpleNamespace(
        current_topic="案件审核",
        referenced_source_refs=[],
        task_state=task_state,
    )

    context = _recent_context_for_planner(messages, session)

    assert len(context["recent_messages"]) == 2
    assert context["structured_task_state"]["task_id"] == "ctask_1"


def test_previous_answer_rewrite_uses_one_text_call_and_deterministic_structure() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return ModelResponse(content="该记录来自上游完整宽表接入。")

    source = CaseAgentSource(
        source_ref="case:source-1",
        source_type="case",
        title="案件接入信息",
    )
    gateway = Gateway()

    def fail_if_full_generation_context_is_used(_state):
        raise AssertionError("previous-answer rewrite must not load normal conversation context")

    service = SimpleNamespace(
        _repository=_Repository(),
        _fast_mode_enabled=False,
        _gateway=gateway,
        _generator_model="deepseek-chat",
        _answer_max_tokens=1536,
        _answer_timeout_seconds=30,
        _build_generation_messages=fail_if_full_generation_context_is_used,
    )
    state = {
        "run": SimpleNamespace(run_id="crun_rewrite"),
        "intent": "case_task",
        "answer_strategy": "reuse_previous_answer",
        "answer_rewrite_mode": "translate_zh",
        "answer_context": {
            "source_refs": ["case:source-1"],
            "analysis_inputs": [
                {
                    "mode": "reuse_previous_answer",
                    "rewrite_mode": "translate_zh",
                    "previous_answer": "The record came from the upstream wide table.",
                    "rewrite_instruction": "请用中文告诉我",
                    "previous_source_refs": ["case:source-1"],
                }
            ],
        },
        "available_sources": [source],
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    result = generate_answer_node(service, state)
    answer = CaseAgentAnswer.model_validate_json(result["final_response"].content)

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["require_json"] is False
    assert gateway.calls[0]["max_tokens"] == 384
    assert gateway.calls[0]["timeout_seconds"] == 8.0
    assert len(gateway.calls[0]["messages"]) == 2
    assert "The record came from the upstream wide table." in gateway.calls[0]["messages"][1]["content"]
    assert "previous_answer" in gateway.calls[0]["messages"][1]["content"]
    assert "上一轮用户原问题" not in gateway.calls[0]["messages"][1]["content"]
    assert result["model_call_count"] == 1
    assert result["final_response"].metrics["rewrite_text_wrapped"] is True
    assert answer.content_blocks[0].text == "该记录来自上游完整宽表接入。"
    assert answer.content_blocks[0].source_refs == ["case:source-1"]
    assert answer.metadata["deterministic_structure_wrap"] is True
    validate_answer(answer, [source])
