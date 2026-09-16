"""Structured, bounded Case Agent state shared across conversation turns."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.backend.domain.case_agent.entities import (
    CaseAgentCompletedStep,
    CaseAgentTaskState,
)


_CORRECTION_MARKERS = (
    "不是",
    "更正",
    "纠正",
    "改成",
    "我说错了",
    "应该是",
    "应为",
    "其实是",
)
_NEW_TASK_MARKERS = (
    "换个问题",
    "另一个问题",
    "新的问题",
    "重新开始",
    "接下来问",
    "先不说这个",
    "不聊这个了",
)
_SUPPLEMENT_MARKERS = (
    "补充",
    "另外",
    "还有",
    "同时",
    "再加上",
)

_ANSWER_REWRITE_MARKERS_BY_MODE = (
    (
        "translate_zh",
        ("用中文告诉我", "中文告诉我", "翻译成中文", "用中文", "中文解释"),
    ),
    (
        "translate_en",
        ("用英文告诉我", "英文告诉我", "翻译成英文", "用英文", "英文解释"),
    ),
    (
        "example",
        (
            "用一个简单直观的例子说明",
            "用个简单例子告诉我",
            "用个例子告诉我",
            "用个例子说一下",
            "用简单例子说明",
            "举个例子",
            "举例说明",
            "例子说明",
            "给个例子",
            "换个例子",
        ),
    ),
    (
        "simplify",
        (
            "简单回答我",
            "简单回答",
            "简单告诉我",
            "简单直接告诉我",
            "简单点回答",
            "简单点说",
            "简单点",
            "简要回答",
            "简要说",
            "简明回答",
            "直接回答我",
            "直接回答",
            "直接告诉我答案",
            "直接告诉我",
            "直接给我答案",
            "直接给答案",
            "直接说答案",
            "通俗点",
            "通俗说明",
            "简单说",
            "简单说明",
            "直观说明",
            "说人话",
        ),
    ),
    ("paraphrase", ("换句话说", "换句话解释")),
    (
        "summarize",
        (
            "用一句话回答",
            "一句话回答",
            "一句话概括",
            "只说结论",
            "只要结论",
            "直接给结论",
            "提炼结论",
            "概括一下",
            "概括",
            "简短一点",
            "简洁一点",
            "缩短",
            "总结一下",
            "总结",
        ),
    ),
    ("list", ("列成清单", "整理成列表", "列表说明", "分点说")),
    (
        "expand",
        (
            "展开说",
            "详细解释",
            "详细一点",
            "扩充一下",
            "扩充说明",
            "扩充",
            "扩展一下",
            "扩展说明",
            "扩展",
            "再详细说说",
            "再解释一下",
            "再说明一下",
        ),
    ),
)

_ANSWER_REWRITE_MARKERS = tuple(
    sorted(
        {
            marker
            for _, mode_markers in _ANSWER_REWRITE_MARKERS_BY_MODE
            for marker in mode_markers
        },
        key=len,
        reverse=True,
    )
)

_ANSWER_REFERENCE_PATTERNS = (
    re.compile(
        r"(?:上面|上述|刚才|之前|前面|上一轮|上轮)"
        r"(?:关于[^，。！？,.!?]{0,40})?的?"
        r"(?:回答|答案|内容|结果|结论|问题)"
    ),
    re.compile(r"(?:这个|这段|该)(?:回答|答案|内容|结果|结论|问题)"),
)

_ANSWER_REWRITE_FILLERS = (
    "麻烦",
    "拜托",
    "请",
    "帮我",
    "给我",
    "简单",
    "简要",
    "简明",
    "把",
    "将",
    "再",
    "直接",
    "重新",
    "改写",
    "回答",
    "告诉",
    "说明",
    "解释",
    "概括",
    "总结",
    "说",
    "写",
    "成",
    "一下",
    "下",
    "即可",
    "就行",
    "就好",
    "上一轮",
    "上轮",
    "上面",
    "上述",
    "刚才",
    "之前",
    "前面",
    "这个",
    "这段",
    "内容",
    "答案",
    "结果",
    "结论",
    "问题",
    "原回答",
    "原答案",
    "我",
    "用",
    "的",
    "吧",
    "呢",
    "吗",
    "呀",
    "啊",
)


def task_state_payload(value: Any) -> dict[str, Any]:
    """Return one validated task-state payload from a model or mapping."""

    if isinstance(value, CaseAgentTaskState):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return CaseAgentTaskState.model_validate(value).model_dump(mode="json")
    return CaseAgentTaskState().model_dump(mode="json")


def detect_answer_rewrite_mode(text: str) -> str | None:
    """Recognize style-only follow-ups that can reuse the latest answer."""

    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return None
    for mode, markers in _ANSWER_REWRITE_MARKERS_BY_MODE:
        if any(marker in compact for marker in markers):
            return mode
    return None


def is_answer_rewrite_only(text: str) -> bool:
    """Return whether a turn changes presentation without adding a new question."""

    compact = re.sub(r"\s+", "", str(text or ""))
    if detect_answer_rewrite_mode(compact) is None:
        return False

    residual = compact
    for pattern in _ANSWER_REFERENCE_PATTERNS:
        residual = pattern.sub("", residual)
    for marker in _ANSWER_REWRITE_MARKERS:
        residual = residual.replace(marker, "")
    for filler in sorted(_ANSWER_REWRITE_FILLERS, key=len, reverse=True):
        residual = residual.replace(filler, "")
    residual = re.sub(r"[，。！？、；：,.!?;:'\"“”‘’（）()\[\]【】<>《》]+", "", residual)
    return not residual


def classify_turn_relation(
    text: str,
    *,
    task_state: dict[str, Any] | CaseAgentTaskState,
    resume_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how the new user turn relates to the active task."""

    current = CaseAgentTaskState.model_validate(task_state or {})
    compact = re.sub(r"\s+", "", str(text or ""))
    resume = resume_context or {}
    if is_answer_rewrite_only(compact):
        return _relation("answer_rewrite", 0.98, "style_only_followup")
    pending = resume.get("pending_clarification") or current.pending_clarification
    if isinstance(pending, dict) and pending.get("missing_slots"):
        return _relation("clarification_response", 0.99, "pending_clarification")
    if any(marker in compact for marker in _NEW_TASK_MARKERS):
        return _relation("new_task", 0.98, "explicit_task_switch")
    if any(marker in compact for marker in _CORRECTION_MARKERS):
        return _relation("correction", 0.9, "explicit_correction")
    if not current.task_id:
        return _relation("new_task", 1.0, "no_active_task")
    if any(marker in compact for marker in _SUPPLEMENT_MARKERS):
        return _relation("supplement", 0.75, "supplement_marker")
    return _relation("continuation", 0.7, "active_task")


def compact_task_state(value: Any) -> dict[str, Any]:
    """Project only the fields needed by perception and planning prompts."""

    state = CaseAgentTaskState.model_validate(value or {})
    return {
        "state_version": state.state_version,
        "task_id": state.task_id,
        "status": state.status,
        "core_intent": state.core_intent,
        "goal": state.goal[:240],
        "current_subtask": state.current_subtask,
        "workflow_stage": state.workflow_stage,
        "confirmed_slots": _safe_mapping(state.confirmed_slots),
        "pending_clarification": _safe_pending(state.pending_clarification),
        "context_snapshot": {
            "answer_ref": str(state.context_snapshot.get("answer_ref") or "")[:120],
            "source_refs": _safe_refs(state.context_snapshot.get("source_refs")),
            "last_intent": str(state.context_snapshot.get("last_intent") or "")[:80],
            "last_capabilities": _safe_strings(
                state.context_snapshot.get("last_capabilities"), 8, 120
            ),
        },
        "last_turn_relation": state.last_turn_relation,
    }


def merge_persisted_task_state(current_value: Any, incoming_value: Any) -> dict[str, Any]:
    """Increment state version while preventing an older turn from overwriting a newer one."""

    current = CaseAgentTaskState.model_validate(current_value or {})
    incoming = CaseAgentTaskState.model_validate(incoming_value or {})
    if task_state_update_is_stale(current, incoming):
        return current.model_dump(mode="json")
    return incoming.model_copy(
        update={"state_version": current.state_version + 1}
    ).model_dump(mode="json")


def task_state_update_is_stale(current_value: Any, incoming_value: Any) -> bool:
    """Return whether an incoming turn is older than the persisted session state."""

    current = CaseAgentTaskState.model_validate(current_value or {})
    incoming = CaseAgentTaskState.model_validate(incoming_value or {})
    return bool(
        current.last_turn_created_at is not None
        and incoming.last_turn_created_at is not None
        and incoming.last_turn_created_at < current.last_turn_created_at
    )


def build_next_task_state(
    state: dict[str, Any],
    *,
    answer_ref: str,
    source_refs: list[str],
    status: str,
) -> dict[str, Any]:
    """Build the next validated task state after one persisted turn."""

    session = state.get("session")
    current = CaseAgentTaskState.model_validate(
        getattr(session, "task_state", None) or state.get("task_state") or {}
    )
    relation_payload = state.get("turn_relation") or {}
    relation = str(relation_payload.get("kind") or "continuation")
    if relation not in {
        "new_task",
        "supplement",
        "correction",
        "clarification_response",
        "answer_rewrite",
        "continuation",
    }:
        relation = "continuation"

    user_message = state.get("user_message")
    message_id = str(getattr(user_message, "message_id", "") or "")
    created_at = getattr(user_message, "created_at", None)
    workflow_stage = getattr(user_message, "active_stage", None)
    resolved_intent = str(state.get("intent") or current.core_intent or "unknown")[:80]
    capabilities = _capabilities(state)
    subtask = _subtask(state, capabilities)
    is_new_task = (
        relation == "new_task"
        or not current.task_id
        or (
            relation == "continuation"
            and current.core_intent not in {"", "unknown"}
            and resolved_intent not in {"", "unknown", current.core_intent}
        )
    )

    confirmed_slots = {} if is_new_task else dict(current.confirmed_slots)
    confirmed_slots.update(_safe_mapping(state.get("slots") or {}))
    completed_steps = [] if is_new_task or relation == "correction" else list(current.completed_steps)
    previous_snapshot = {} if is_new_task or relation == "correction" else dict(current.context_snapshot)

    goal = _goal(state)
    if not is_new_task and relation in {
        "supplement",
        "correction",
        "clarification_response",
        "answer_rewrite",
        "continuation",
    }:
        goal = current.goal or goal
    core_intent = resolved_intent if is_new_task else current.core_intent

    pending = state.get("pending_clarification") if status == "waiting_user" else {}
    if status == "completed" and answer_ref:
        completed_steps.append(
            CaseAgentCompletedStep(
                message_id=message_id,
                intent=resolved_intent,
                subtask=subtask,
                answer_ref=answer_ref[:120],
                capabilities=capabilities,
                source_refs=_safe_refs(source_refs),
                completed_at=datetime.now(timezone.utc),
            )
        )
        completed_steps = completed_steps[-8:]
        previous_snapshot = {
            "answer_ref": answer_ref[:120],
            "source_refs": _safe_refs(source_refs),
            "last_intent": resolved_intent,
            "last_capabilities": capabilities,
        }

    next_state = current.model_copy(
        update={
            "task_id": f"ctask_{uuid4().hex}" if is_new_task else current.task_id,
            "status": status,
            "core_intent": core_intent or resolved_intent,
            "goal": goal[:300],
            "current_subtask": subtask,
            "workflow_stage": workflow_stage or current.workflow_stage,
            "confirmed_slots": confirmed_slots,
            "pending_clarification": _safe_pending(pending),
            "completed_steps": completed_steps,
            "context_snapshot": previous_snapshot,
            "last_turn_relation": relation,
            "last_user_message_id": message_id or current.last_user_message_id,
            "last_turn_created_at": created_at or datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return CaseAgentTaskState.model_validate(next_state.model_dump()).model_dump(mode="json")


def _relation(kind: str, confidence: float, reason: str) -> dict[str, Any]:
    return {"kind": kind, "confidence": confidence, "reason": reason}


def _goal(state: dict[str, Any]) -> str:
    semantics = state.get("query_semantics") or {}
    if isinstance(semantics, dict) and semantics.get("user_goal"):
        return str(semantics["user_goal"])[:300]
    perceptual = (state.get("perceptual_state") or {}).get("semantic") or {}
    return str(perceptual.get("rewritten_query") or "")[:300]


def _capabilities(state: dict[str, Any]) -> list[str]:
    values = []
    for item in state.get("execution_plan", []) or []:
        if isinstance(item, dict):
            values.append(str(item.get("capability") or item.get("name") or ""))
    return list(dict.fromkeys(value[:120] for value in values if value))[:8]


def _subtask(state: dict[str, Any], capabilities: list[str]) -> str:
    if state.get("answer_strategy") == "reuse_previous_answer":
        return f"rewrite:{str(state.get('answer_rewrite_mode') or 'explain')[:80]}"
    if state.get("completion_status") == "waiting_for_user":
        return "clarification"
    return capabilities[0] if capabilities else str(state.get("intent") or "general_help")[:120]


def _safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, item in list(value.items())[:16]:
        name = str(key)[:80]
        if isinstance(item, (str, int, float, bool)) or item is None:
            safe[name] = str(item)[:200] if isinstance(item, str) else item
        elif isinstance(item, list):
            safe[name] = _safe_strings(item, 8, 120)
    return safe


def _safe_pending(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "intent": str(value.get("intent") or "")[:80],
        "missing_slots": _safe_strings(value.get("missing_slots"), 8, 80),
        "slots": _safe_mapping(value.get("slots")),
        "task_id": str(value.get("task_id") or "")[:80],
    }


def _safe_refs(value: Any) -> list[str]:
    return _safe_strings(value, 12, 200)


def _safe_strings(value: Any, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item)[:item_limit] for item in value if item))[:limit]
