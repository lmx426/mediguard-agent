"""Caser perception layer nodes."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
    _deterministic_expert_plan,
    _deterministic_followup_plan,
    _deterministic_single_field_plan,
    _recent_context_for_planner,
    capability_plan_from_hint,
)
from src.backend.application.agent.case_agent.memory.task_state import (
    compact_task_state,
    detect_answer_rewrite_mode,
    is_answer_rewrite_only,
)
from src.backend.application.agent.case_agent.prompts.planning import (
    build_semantic_parse_prompt,
)
from src.backend.application.agent.case_agent.schemas.perception import (
    DecisionContext,
    EntityFrame,
    InputEnvelope,
    MinimalPlanningContext,
    PerceptualState,
    SemanticFrame,
    SemanticParseResult,
)
from src.backend.application.agent.case_agent.shortcut import (
    detect_recent_answer_count_shortcut,
    detect_shortcut,
)
from ._errors import state_error


REGION_ALIASES = {
    "北京": "beijing",
    "上海": "shanghai",
    "全国": "national",
    "国家": "national",
}


def fast_rule_entity_perception_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic perception fast path with one durable node write."""

    started = time.perf_counter()
    try:
        run = state["run"]
        user_message = state["user_message"]
        service._repository.update_run(
            run.run_id,
            current_node="fast_rule_entity_perception",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        raw_text = str(getattr(user_message, "content", "") or "")
        envelope = InputEnvelope(
            raw_text=raw_text,
            normalized_text=_normalize_message(raw_text),
            channel="chat",
            attachments_ref=[],
            event_ref=getattr(user_message, "message_id", ""),
        )
        frame = EntityFrame(
            explicit_entities=_extract_entities(raw_text),
            slot_candidates=_extract_slot_candidates(raw_text),
            referenced_source_refs=_extract_referenced_source_refs(raw_text, state),
        )
        state["input_envelope"] = envelope.model_dump(mode="json")
        state["entity_frame"] = frame.model_dump(mode="json")
        state["semantic_intent_signal"] = {
            "normalized_text": envelope.normalized_text,
            "has_question": bool(envelope.normalized_text),
            "current_topic": str(getattr(state.get("session"), "current_topic", "") or "")[:300],
        }
        state["perception_started_epoch_ms"] = int(time.time() * 1000)
        rule_result = _apply_fast_rule_semantics(state)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        state.setdefault("perception_metrics", {})["fast_rule_entity_ms"] = latency_ms
        service._repository.append_event(
            run.run_id,
            "fast_rule_entity_perception_complete",
            "已完成快速规则与实体感知",
            {
                "latency_ms": latency_ms,
                "rule_hit": rule_result["rule_hit"],
                "rule_kind": rule_result["rule_kind"],
                "turn_relation": (state.get("turn_relation") or {}).get("kind"),
                "entity_keys": list(frame.explicit_entities.keys()),
                "next_action": state["next_action"],
            },
        )
        return state
    except Exception as exc:
        return state_error(state, exc)


def build_perceptual_state_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Build the complete perception handoff with one normalization pass."""

    started = time.perf_counter()
    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="build_perceptual_state",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        semantic_source = state.get("light_semantic_frame") or state.get("deep_semantic_frame") or {}
        entity_frame = state.get("entity_frame") or {}
        slots = dict(semantic_source.get("slots") or {})
        slots.update(entity_frame.get("slot_candidates") or {})
        semantic = SemanticFrame(
            source=semantic_source.get("source") or "fallback",
            utterance_type=semantic_source.get("utterance_type") or "question",
            business_intent=semantic_source.get("business_intent") or "general_help",
            answer_shape=semantic_source.get("answer_shape") or "overview",
            evidence_need=semantic_source.get("evidence_need") or "none",
            focus_candidate=semantic_source.get("focus_candidate") or {},
            slots=slots,
            missing_slots=semantic_source.get("missing_slots") or [],
            confidence=float(semantic_source.get("confidence") or 0.0),
        )
        confidence = semantic.confidence
        if semantic.source == "rule_precheck":
            confidence = max(confidence, 0.98)
        elif semantic.source == "llm_semantic_parser":
            confidence = max(confidence, 0.82)
        if semantic.missing_slots:
            confidence = min(confidence, 0.5)
        semantic = semantic.model_copy(update={"confidence": max(0.0, min(confidence, 1.0))})

        minimal = _minimal_context(state)
        semantic_payload = semantic.model_dump(mode="json")
        needs_full_context = _needs_full_context(state, semantic_payload)
        context_need = {
            "mode": "gssc" if needs_full_context else "minimal",
            "packs": _context_packs_for_semantic(semantic_payload) if needs_full_context else [],
        }
        decision_context = (
            _build_decision_context(state, context_need=context_need)
            if needs_full_context
            else {}
        )
        decision_context_ref = str(decision_context.get("context_ref") or "")
        constraints = _governance_constraints()
        missing_slots = [str(item)[:80] for item in list(semantic.missing_slots)[:8]]
        readiness_status = "need_clarification" if missing_slots else "ready_to_plan"
        readiness = {
            "status": readiness_status,
            "ready_to_plan": readiness_status == "ready_to_plan",
            "soft_plan_allowed": readiness_status == "ready_to_plan",
            "need_clarification": readiness_status == "need_clarification",
            "fail_closed": False,
            "reasons": ["missing_slots"] if missing_slots else [],
        }
        session = state.get("session")
        context_refs = {
            "case_ref": run.case_id,
            "session_ref": run.session_id,
            "answer_ref": decision_context.get("domain_artifact_context", {}).get("answer_ref"),
            "artifact_ref": state.get("artifact_refs", []),
            "source_ref": _recent_source_refs(state.get("recent_messages", []), session),
            "evidence_ref": [],
        }
        perceptual_state = PerceptualState(
            semantic={
                "intent": semantic.business_intent,
                "action": semantic.focus_candidate.get("action") or "",
                "answer_shape": semantic.answer_shape,
                "target_layer_hint": semantic.focus_candidate.get("target_layer_hint") or "none",
                "target_objects": semantic.focus_candidate.get("target_objects") or [],
                "information_needs": semantic.focus_candidate.get("information_needs") or [],
                "rewritten_query": semantic.focus_candidate.get("rewritten_query")
                or minimal.question,
                "evidence_need": semantic.evidence_need,
                "requires_new_evidence": semantic.slots.get("requires_new_evidence", True),
                "reuse_answer_ref": semantic.slots.get("reuse_answer_ref") or "",
                "reuse_source_refs": semantic.slots.get("reuse_source_refs") or [],
                "rewrite_mode": semantic.slots.get("rewrite_mode") or "",
                "confidence": semantic.confidence,
                "source": semantic.source,
            },
            slots_focus={
                "entities": entity_frame.get("explicit_entities") or {},
                "object_ref": semantic.focus_candidate,
                "missing_slots": missing_slots,
            },
            context_refs=context_refs,
            decision_context_ref=decision_context_ref,
            constraints=constraints,
            readiness_signal=readiness,
            minimal_context=minimal.model_dump(mode="json"),
        )
        contract = {
            "semantic_frame": semantic_payload,
            "entity_frame": entity_frame,
            "minimal_context": minimal.model_dump(mode="json"),
            "decision_context_ref": decision_context_ref,
            "decision_context": decision_context,
            "constraints": constraints,
        }
        state.update(
            {
                "semantic_frame": semantic_payload,
                "slots": slots,
                "missing_slots": missing_slots,
                "intent_confidence": semantic.confidence,
                "minimal_planning_context": minimal.model_dump(mode="json"),
                "context_need": context_need,
                "decision_context": decision_context,
                "decision_context_ref": decision_context_ref,
                "perception_contract": contract,
                "perceptual_state": perceptual_state.model_dump(mode="json"),
                "intent": semantic.business_intent,
                "next_action": "build_memory_retrieval_plan",
            }
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        total_latency_ms = max(
            0,
            int(time.time() * 1000) - int(state.get("perception_started_epoch_ms") or 0),
        )
        metrics = state.setdefault("perception_metrics", {})
        metrics["build_perceptual_state_ms"] = latency_ms
        metrics["total_latency_ms"] = total_latency_ms
        metrics["semantic_source"] = semantic.source
        service._repository.append_event(
            run.run_id,
            "perceptual_state_ready",
            "感知层已形成可规划信号",
            {
                "intent": semantic.business_intent,
                "confidence": semantic.confidence,
                "readiness": readiness_status,
                "semantic_source": semantic.source,
                "decision_context_mode": context_need["mode"],
                "latency_ms": latency_ms,
                "total_latency_ms": total_latency_ms,
            },
        )
        return state
    except Exception as exc:
        return state_error(state, exc)


def input_ingestion_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create the InputEnvelope for the perception layer."""

    try:
        run = state["run"]
        user_message = state["user_message"]
        service._repository.update_run(
            run.run_id,
            current_node="input_ingestion",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        raw_text = str(getattr(user_message, "content", "") or "")
        envelope = InputEnvelope(
            raw_text=raw_text,
            normalized_text=_normalize_message(raw_text),
            channel="chat",
            attachments_ref=[],
            event_ref=getattr(user_message, "message_id", ""),
        )
        state["input_envelope"] = envelope.model_dump(mode="json")
        service._repository.append_event(
            run.run_id,
            "input_ingestion",
            "已完成输入适配",
            {"message_id": envelope.event_ref},
        )
        state["next_action"] = "early_entity_extractor"
        return state
    except Exception as exc:
        return state_error(state, exc)


def early_entity_extractor_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Extract simple entities before semantic perception."""

    try:
        run = state["run"]
        envelope = state.get("input_envelope") or {}
        text = str(envelope.get("raw_text") or "")
        service._repository.update_run(
            run.run_id,
            current_node="early_entity_extractor",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        frame = EntityFrame(
            explicit_entities=_extract_entities(text),
            slot_candidates=_extract_slot_candidates(text),
            referenced_source_refs=_extract_referenced_source_refs(text, state),
        )
        state["entity_frame"] = frame.model_dump(mode="json")
        service._repository.append_event(
            run.run_id,
            "early_entity_extracted",
            "已抽取问题中的基础实体",
            {
                "entity_keys": list(frame.explicit_entities.keys()),
                "referenced_source_ref_count": len(frame.referenced_source_refs),
            },
        )
        state["next_action"] = "memory_prefetch"
        return state
    except Exception as exc:
        return state_error(state, exc)


def semantic_intent_perception_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Prepare lightweight semantic perception signals."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="semantic_intent_perception",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        envelope = state.get("input_envelope") or {}
        state["semantic_intent_signal"] = {
            "normalized_text": envelope.get("normalized_text") or "",
            "has_question": bool(str(envelope.get("normalized_text") or "").strip()),
            "current_topic": str(getattr(state.get("session"), "current_topic", "") or "")[:300],
        }
        service._repository.append_event(
            run.run_id,
            "semantic_intent_perception",
            "正在识别业务意图",
            {},
        )
        state["next_action"] = "rule_precheck"
        return state
    except Exception as exc:
        return state_error(state, exc)


def rule_precheck_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic semantic shortcuts before Bi-Encoder or LLM."""

    try:
        run = state["run"]
        user_message = state["user_message"]
        service._repository.update_run(
            run.run_id,
            current_node="rule_precheck",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        text = str(getattr(user_message, "content", "") or "")
        shortcut = detect_shortcut(text)
        if shortcut is None:
            shortcut = detect_recent_answer_count_shortcut(
                text,
                state.get("recent_messages", []),
            )
        if shortcut is not None:
            from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
                _shortcut_plan,
            )

            plan = _shortcut_plan(shortcut)
            state["shortcut"] = shortcut.to_state()
            state["rule_candidate_plan"] = plan.model_dump(mode="json")
            state["light_semantic_frame"] = _semantic_frame_from_plan(
                plan.model_dump(mode="json"),
                source="rule_precheck",
                confidence=1.0,
                evidence_need="none",
            )
            service._repository.append_event(
                run.run_id,
                "rule_precheck_hit",
                "规则语义已命中",
                {"capability_count": len(plan.execution_plan), "shortcut": shortcut.kind},
            )
            state["next_action"] = "slot_merger"
            return state

        rewrite_followup = _answer_rewrite_followup_signal(state)
        if rewrite_followup is not None:
            state["light_semantic_frame"] = rewrite_followup
            service._repository.append_event(
                run.run_id,
                "answer_rewrite_followup_detected",
                "已识别为上一轮回答的解释或举例追问",
                {
                    "rewrite_mode": rewrite_followup.get("slots", {}).get("rewrite_mode"),
                    "reuse_answer_ref": rewrite_followup.get("slots", {}).get("reuse_answer_ref"),
                    "reuse_source_ref_count": len(
                        rewrite_followup.get("slots", {}).get("reuse_source_refs") or []
                    ),
                },
            )
            state["next_action"] = "slot_merger"
            return state

        clarification_response = _clarification_response_signal(state)
        if clarification_response is not None:
            state["light_semantic_frame"] = clarification_response
            state["next_action"] = "slot_merger"
            return state

        plan = _deterministic_expert_plan(run.case_id, text)
        if plan is None:
            plan = _deterministic_followup_plan(
                run.case_id,
                text,
                state.get("recent_messages", []),
                state.get("session"),
            )
        if plan is None:
            plan = _deterministic_single_field_plan(run.case_id, text)
        if plan is not None:
            plan_payload = plan.model_dump(mode="json")
            state["rule_candidate_plan"] = plan_payload
            state["light_semantic_frame"] = _semantic_frame_from_plan(
                plan_payload,
                source="rule_precheck",
                confidence=1.0,
                evidence_need=_evidence_need_for_plan(plan_payload),
            )
            service._repository.append_event(
                run.run_id,
                "rule_precheck_hit",
                "规则语义已命中",
                {"capability_count": len(plan.execution_plan)},
            )
            state["next_action"] = "slot_merger"
            return state

        service._repository.append_event(
            run.run_id,
            "rule_precheck_miss",
            "规则语义未命中，进入意图样例检索",
            {},
        )
        state["next_action"] = "intent_example_biencoder"
        return state
    except Exception as exc:
        return state_error(state, exc)


def intent_example_biencoder_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Run optional intent example retrieval.

    The lightweight BERT index is a prepared but not-yet-connected asset. This
    node is present in the runtime graph and safely falls back when no matcher
    is injected.
    """

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="intent_example_biencoder",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        matcher = getattr(service, "_intent_biencoder", None)
        envelope = state.get("input_envelope") or {}
        result = None
        service._repository.append_event(
            run.run_id,
            "intent_example_biencoder_matching",
            "正在检索意图样例",
            {
                "connected": matcher is not None,
                "backend": getattr(matcher, "metadata", {}).get("backend")
                if matcher is not None
                else "",
            },
        )
        if matcher is not None and callable(getattr(matcher, "match", None)):
            result = matcher.match(str(envelope.get("normalized_text") or ""))
        if isinstance(result, dict) and result.get("decision") == "accept":
            state["biencoder_frame"] = result
            direct_plan = capability_plan_from_hint(
                run.case_id,
                str(envelope.get("normalized_text") or ""),
                result,
            )
            if direct_plan is not None:
                state["rule_candidate_plan"] = direct_plan.model_dump(mode="json")
            state["light_semantic_frame"] = {
                "source": "intent_example_biencoder",
                "utterance_type": "question",
                "business_intent": str(result.get("business_intent") or "case_task"),
                "answer_shape": str(result.get("answer_shape") or "overview"),
                "evidence_need": str(result.get("evidence_need") or "case_fact"),
                "focus_candidate": {
                    "matched_intent_id": result.get("matched_intent_id") or "",
                    "target_layer_hint": result.get("target_layer_hint") or "none",
                    "target_objects": [result.get("matched_intent_id")]
                    if result.get("matched_intent_id")
                    else [],
                    "capability_hint": result.get("capability_hint"),
                    "ability_layer": result.get("ability_layer"),
                    "information_needs": result.get("information_needs") or [],
                },
                "slots": {
                    "capability_hint": result.get("capability_hint"),
                    "matched_intent_id": result.get("matched_intent_id"),
                },
                "missing_slots": [],
                "confidence": float(result.get("confidence") or 0.0),
            }
            service._repository.append_event(
                run.run_id,
                "intent_example_biencoder_accepted",
                "意图样例检索已采纳",
                {
                    "business_intent": state["light_semantic_frame"]["business_intent"],
                    "confidence": state["light_semantic_frame"]["confidence"],
                    "matched_intent_id": result.get("matched_intent_id") or "",
                    "target_layer_hint": result.get("target_layer_hint") or "none",
                    "backend": result.get("backend") or "",
                    "direct_plan": direct_plan is not None,
                },
            )
            state["next_action"] = "build_perceptual_state"
            return state

        state["biencoder_frame"] = result if isinstance(result, dict) else {
            "source": "intent_example_biencoder",
            "decision": "fallback",
            "reason": "intent example index is not connected",
        }
        state["semantic_hint_pack"] = _semantic_hint_pack(state)
        service._repository.append_event(
            run.run_id,
            "intent_example_biencoder_fallback",
            "意图样例检索未采纳，进入 LLM 语义兜底",
            {
                "connected": matcher is not None,
                "decision": state["biencoder_frame"].get("decision"),
                "reason": state["biencoder_frame"].get("reason"),
                "matched_intent_id": state["biencoder_frame"].get("matched_intent_id") or "",
                "confidence": state["biencoder_frame"].get("confidence") or 0.0,
                "backend": state["biencoder_frame"].get("backend") or "",
            },
        )
        state["next_action"] = "llm_semantic_parser"
        return state
    except Exception as exc:
        return state_error(state, exc)


def light_semantic_context_assembly_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded hint pack for LLM semantic parsing."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="light_semantic_context_assembly",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        hint_pack = _semantic_hint_pack(state, service)
        state["semantic_hint_pack"] = hint_pack
        service._repository.append_event(
            run.run_id,
            "light_semantic_context_assembled",
            "已完成轻量语义上下文装配",
            {
                "active_ref_count": len(hint_pack.get("active_reference_objects", [])),
                "candidate_field_count": len(
                    hint_pack.get("slot_repair_hints", {}).get("candidate_fields", [])
                ),
            },
        )
        state["next_action"] = "llm_semantic_parser"
        return state
    except Exception as exc:
        return state_error(state, exc)


def llm_semantic_parser_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Use the lightweight parser prompt to produce pure semantic structure."""

    try:
        run = state["run"]
        user_message = state["user_message"]
        service._repository.update_run(
            run.run_id,
            current_node="llm_semantic_parser",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            run.run_id,
            "llm_semantic_parsing",
            "正在进行 LLM 语义解析",
            {"model": getattr(service, "_classifier_model", "")},
        )
        response = service._gateway.complete(
            messages=[
                {
                    "role": "system",
                    "content": build_semantic_parse_prompt(run.case_id),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_question": user_message.content,
                            "semantic_hint_pack": state.get("semantic_hint_pack")
                            or _semantic_hint_pack(state, service),
                            "instruction": (
                                "只做语义解析、指代消解和补槽；不要生成 execution_plan，"
                                "不要回答用户问题。"
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            tools=[],
            require_json=True,
            model=service._classifier_model,
            thinking_enabled=False,
            max_tokens=getattr(service, "_semantic_max_tokens", 384),
            timeout_seconds=getattr(service, "_semantic_timeout_seconds", 12),
        )
        state["model_call_count"] = state.get("model_call_count", 0) + 1
        recorder = getattr(service, "record_model_call_metrics", None)
        model_metrics = (
            recorder(state, "llm_semantic_parser", response)
            if callable(recorder)
            else dict(response.metrics)
        )
        semantic_result = _parse_semantic_result(response.content or "{}")
        semantic_payload = semantic_result.model_dump(mode="json")
        state["llm_semantic_result"] = semantic_payload
        state["deep_semantic_frame"] = _semantic_frame_from_semantic_result(
            semantic_payload,
            source="llm_semantic_parser",
        )
        service._repository.append_event(
            run.run_id,
            "llm_semantic_parsed",
            "LLM 语义解析已完成",
            {
                "intent": semantic_payload.get("intent"),
                "target_layer_hint": semantic_payload.get("target_layer_hint"),
                "model_metrics": model_metrics,
            },
        )
        state["next_action"] = "build_perceptual_state"
        return state
    except (json.JSONDecodeError, ValidationError) as exc:
        semantic_payload = _fallback_semantic_result().model_dump(mode="json")
        state["llm_semantic_result"] = semantic_payload
        state["deep_semantic_frame"] = _semantic_frame_from_semantic_result(
            semantic_payload,
            source="fallback",
        )
        state["planning_error"] = f"{exc.__class__.__name__}: {exc}"
        service._repository.append_event(
            run.run_id,
            "llm_semantic_parser_failed",
            "LLM 语义解析未形成有效结构，已安全降级",
            {"error": state["planning_error"][:300]},
        )
        state["next_action"] = "build_perceptual_state"
        return state
    except Exception as exc:
        return state_error(state, exc)


def slot_merger_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Merge entity and semantic frames into a stable SemanticFrame."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="slot_merger",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        semantic = (
            state.get("light_semantic_frame")
            or state.get("deep_semantic_frame")
            or {}
        )
        entity_frame = state.get("entity_frame") or {}
        slots = dict(semantic.get("slots") or {})
        slots.update(entity_frame.get("slot_candidates") or {})
        merged = SemanticFrame(
            source=semantic.get("source") or "fallback",
            utterance_type=semantic.get("utterance_type") or "question",
            business_intent=semantic.get("business_intent") or "general_help",
            answer_shape=semantic.get("answer_shape") or "overview",
            evidence_need=semantic.get("evidence_need") or "none",
            focus_candidate=semantic.get("focus_candidate") or {},
            slots=slots,
            missing_slots=semantic.get("missing_slots") or [],
            confidence=float(semantic.get("confidence") or 0.0),
        )
        state["semantic_frame"] = merged.model_dump(mode="json")
        state["slots"] = slots
        state["missing_slots"] = merged.missing_slots
        state["intent_confidence"] = merged.confidence
        service._repository.append_event(
            run.run_id,
            "slot_merged",
            "已合并槽位与语义",
            {
                "semantic_source": merged.source,
                "missing_slot_count": len(merged.missing_slots),
            },
        )
        state["next_action"] = "context_need_resolver"
        return state
    except Exception as exc:
        return state_error(state, exc)


def context_need_resolver_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the perception layer needs the single GSSC assembly."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="context_need_resolver",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        minimal = _minimal_context(state)
        state["minimal_planning_context"] = minimal.model_dump(mode="json")
        semantic = state.get("semantic_frame") or {}
        needs_full_context = _needs_full_context(state, semantic)
        state["context_need"] = {
            "mode": "gssc" if needs_full_context else "minimal",
            "packs": _context_packs_for_semantic(semantic) if needs_full_context else [],
        }
        service._repository.append_event(
            run.run_id,
            "context_need_resolved",
            "已判断规划所需上下文包",
            state["context_need"],
        )
        service._repository.append_event(
            run.run_id,
            "minimal_planning_context_built",
            "已生成 MinimalPlanningContext",
            {
                "case_id": minimal.case_id,
                "current_topic": minimal.current_topic[:120],
            },
        )
        state["next_action"] = "perception_gssc" if needs_full_context else "perception_contract_builder"
        return state
    except Exception as exc:
        return state_error(state, exc)


def perception_gssc_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Build the single DecisionContext with a lightweight GSSC pass."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="perception_gssc",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        context_ref = f"decision_context:{run.run_id}"
        session = state.get("session")
        user_message = state.get("user_message")
        recent = state.get("recent_messages", []) or []
        decision_context = DecisionContext(
            context_ref=context_ref,
            runtime_context={
                "run_id": run.run_id,
                "trace_id": run.run_id,
                "session_id": run.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "route_version": "caser-perception-decision-v0.1",
            },
            governance_context={
                "actor_id": state.get("actor_id"),
                "tenant": "local",
                "role": "auditor",
                "permissions": ["case_agent:read"],
                "current_case_only": True,
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
            },
            domain_artifact_context={
                "case_ref": run.case_id,
                "case_state_summary": {
                    "session_title": getattr(session, "title", ""),
                    "active_stage": getattr(user_message, "active_stage", None),
                },
                "l1_result_ref": [],
                "l2_result_ref": [],
                "l3_result_ref": [],
                "answer_ref": _latest_answer_ref(recent),
                "source_ref": _recent_source_refs(recent, session),
                "evidence_ref": [],
            },
            working_memory_context={
                "recent_window": _recent_message_window(
                    recent,
                    limit=_recent_context_limit(state),
                ),
                "rolling_summary": str(getattr(session, "session_summary", "") or "")[:800],
                "current_topic": str(getattr(session, "current_topic", "") or "")[:300],
                "structured_task_state": compact_task_state(state.get("task_state") or {}),
                "turn_relation": state.get("turn_relation") or {},
                "long_memory_index": None,
                "vector_memory_refs": [],
            },
            context_summary={
                "context_need": state.get("context_need"),
                "semantic_source": (state.get("semantic_frame") or {}).get("source"),
            },
        )
        state["decision_context"] = decision_context.model_dump(mode="json")
        state["decision_context_ref"] = context_ref
        service._repository.append_event(
            run.run_id,
            "perception_gssc_complete",
            "已完成感知层上下文装配",
            {
                "context_ref": context_ref,
                "packs": state.get("context_need", {}).get("packs", []),
            },
        )
        state["next_action"] = "perception_contract_builder"
        return state
    except Exception as exc:
        return state_error(state, exc)


def perception_contract_builder_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Build the raw perception contract before quality gates."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="perception_contract_builder",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        state["perception_contract"] = {
            "semantic_frame": state.get("semantic_frame") or {},
            "entity_frame": state.get("entity_frame") or {},
            "minimal_context": state.get("minimal_planning_context") or {},
            "decision_context_ref": state.get("decision_context_ref") or "",
            "decision_context": state.get("decision_context") or {},
            "constraints": _governance_constraints(),
        }
        state["next_action"] = "state_normalizer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def state_normalizer_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Normalize perception contract schema and defaults."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="state_normalizer",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        contract = dict(state.get("perception_contract") or {})
        semantic = SemanticFrame.model_validate(contract.get("semantic_frame") or {})
        minimal = MinimalPlanningContext.model_validate(
            contract.get("minimal_context") or _minimal_context(state).model_dump(mode="json")
        )
        contract["semantic_frame"] = semantic.model_dump(mode="json")
        contract["minimal_context"] = minimal.model_dump(mode="json")
        contract["constraints"] = {
            **_governance_constraints(),
            **dict(contract.get("constraints") or {}),
        }
        state["perception_contract"] = contract
        service._repository.append_event(
            run.run_id,
            "perception_state_normalized",
            "已统一感知状态 schema 与默认值",
            {
                "semantic_source": semantic.source,
                "missing_slot_count": len(semantic.missing_slots),
            },
        )
        state["next_action"] = "confidence_calibrator"
        return state
    except Exception as exc:
        return state_error(state, exc)


def confidence_calibrator_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Calibrate a single confidence value for decision readiness."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="confidence_calibrator",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        contract = dict(state.get("perception_contract") or {})
        semantic = dict(contract.get("semantic_frame") or {})
        confidence = float(semantic.get("confidence") or 0.0)
        if semantic.get("source") == "rule_precheck":
            confidence = max(confidence, 0.98)
        elif semantic.get("source") == "llm_semantic_parser":
            confidence = max(confidence, 0.82)
        if semantic.get("missing_slots"):
            confidence = min(confidence, 0.5)
        contract["semantic_frame"] = {
            **semantic,
            "confidence": max(0.0, min(confidence, 1.0)),
        }
        state["perception_contract"] = contract
        state["intent_confidence"] = contract["semantic_frame"]["confidence"]
        service._repository.append_event(
            run.run_id,
            "perception_confidence_calibrated",
            "已校准感知置信度",
            {"confidence": state["intent_confidence"]},
        )
        state["next_action"] = "readiness_signal_builder"
        return state
    except Exception as exc:
        return state_error(state, exc)


def readiness_signal_builder_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Create the final PerceptualState handoff."""

    try:
        run = state["run"]
        service._repository.update_run(
            run.run_id,
            current_node="readiness_signal_builder",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        contract = dict(state.get("perception_contract") or {})
        semantic = dict(contract.get("semantic_frame") or {})
        entity = dict(contract.get("entity_frame") or {})
        minimal = dict(contract.get("minimal_context") or {})
        missing_slots = [
            str(item)[:80] for item in list(semantic.get("missing_slots") or [])[:8]
        ]
        readiness_status = "need_clarification" if missing_slots else "ready_to_plan"
        readiness = {
            "status": readiness_status,
            "ready_to_plan": readiness_status == "ready_to_plan",
            "soft_plan_allowed": readiness_status == "ready_to_plan",
            "need_clarification": readiness_status == "need_clarification",
            "fail_closed": False,
            "reasons": ["missing_slots"] if missing_slots else [],
        }
        context_refs = {
            "case_ref": run.case_id,
            "session_ref": run.session_id,
            "answer_ref": (state.get("decision_context") or {})
            .get("domain_artifact_context", {})
            .get("answer_ref"),
            "artifact_ref": state.get("artifact_refs", []),
            "source_ref": _recent_source_refs(
                state.get("recent_messages", []),
                state.get("session"),
            ),
            "evidence_ref": [],
        }
        perceptual_state = PerceptualState(
            semantic={
                "intent": semantic.get("business_intent") or "general_help",
                "action": (semantic.get("focus_candidate") or {}).get("action") or "",
                "answer_shape": semantic.get("answer_shape") or "overview",
                "target_layer_hint": (semantic.get("focus_candidate") or {}).get("target_layer_hint")
                or "none",
                "target_objects": (semantic.get("focus_candidate") or {}).get("target_objects") or [],
                "rewritten_query": (semantic.get("focus_candidate") or {}).get("rewritten_query")
                or minimal.get("question")
                or "",
                "evidence_need": semantic.get("evidence_need") or "none",
                "requires_new_evidence": (semantic.get("slots") or {}).get(
                    "requires_new_evidence",
                    True,
                ),
                "reuse_answer_ref": (semantic.get("slots") or {}).get("reuse_answer_ref") or "",
                "reuse_source_refs": (semantic.get("slots") or {}).get("reuse_source_refs") or [],
                "rewrite_mode": (semantic.get("slots") or {}).get("rewrite_mode") or "",
                "confidence": semantic.get("confidence") or 0.0,
                "source": semantic.get("source") or "fallback",
            },
            slots_focus={
                "entities": entity.get("explicit_entities") or {},
                "object_ref": semantic.get("focus_candidate") or {},
                "missing_slots": missing_slots,
            },
            context_refs=context_refs,
            decision_context_ref=state.get("decision_context_ref") or "",
            constraints=contract.get("constraints") or _governance_constraints(),
            readiness_signal=readiness,
            minimal_context=minimal,
        )
        state["perceptual_state"] = perceptual_state.model_dump(mode="json")
        state["intent"] = perceptual_state.semantic["intent"]
        state["missing_slots"] = missing_slots
        state["next_action"] = "load_perceptual_state"
        service._repository.append_event(
            run.run_id,
            "readiness_signal_built",
            "已生成 readiness_signal",
            {
                "readiness": readiness_status,
                "missing_slot_count": len(missing_slots),
            },
        )
        service._repository.append_event(
            run.run_id,
            "perceptual_state_ready",
            "感知层已形成可规划信号",
            {
                "intent": state["intent"],
                "confidence": state.get("intent_confidence", 0.0),
                "readiness": readiness_status,
            },
        )
        return state
    except Exception as exc:
        return state_error(state, exc)


def _normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _extract_entities(text: str) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    rule_ids = sorted(set(match.upper() for match in re.findall(r"\bOP-R\d{3,}\b", text, re.I)))
    if rule_ids:
        entities["rule_id"] = rule_ids
    material_ids = re.findall(r"\b(?:BM|CL|MAT|CUSTOM)-[A-Za-z0-9_-]+\b", text)
    if material_ids:
        entities["material_id"] = material_ids[:8]
    regions = [value for label, value in REGION_ALIASES.items() if label in text]
    if regions:
        entities["region"] = list(dict.fromkeys(regions))
    dates = re.findall(r"\b\d{4}[-年]\d{1,2}(?:[-月]\d{1,2}日?)?\b", text)
    if dates:
        entities["date"] = dates[:8]
    source_numbers = re.findall(r"(?:引用|证据|来源)\s*([0-9]{1,2})", text)
    if source_numbers:
        entities["source_number"] = source_numbers[:8]
    return entities


def _extract_slot_candidates(text: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    if "政策" in text or "规定" in text or "口径" in text:
        slots["policy_domain"] = "medical_insurance"
    if "药" in text:
        slots["domain_object"] = "drug"
    if "异地" in text:
        slots["scenario"] = "remote_settlement"
    return slots


def _extract_referenced_source_refs(text: str, state: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    session = state.get("session")
    refs.extend(
        str(ref) for ref in list(getattr(session, "referenced_source_refs", []) or [])[:12]
    )
    if any(marker in text for marker in ("引用", "证据", "来源", "第")):
        for message in reversed(list(state.get("recent_messages", []) or [])):
            refs.extend(str(ref) for ref in list(getattr(message, "source_refs", []) or [])[:12])
            answer_payload = getattr(message, "answer_payload", None)
            if answer_payload is not None:
                try:
                    refs.extend(str(source.source_ref) for source in answer_payload.sources)
                except AttributeError:
                    pass
            if refs:
                break
    return list(dict.fromkeys(ref for ref in refs if ref))


def _semantic_hint_pack(state: dict[str, Any], service: Any | None = None) -> dict[str, Any]:
    run = state["run"]
    user_message = state.get("user_message")
    session = state.get("session")
    recent_messages = list(state.get("recent_messages", []) or [])
    current_question = str(getattr(user_message, "content", "") or "")
    recent_context = _recent_context_for_planner(recent_messages, session)
    return {
        "current_input": {
            "raw_question": current_question[:500],
            "normalized_question": _normalize_message(current_question)[:500],
            "is_follow_up": _looks_like_follow_up_question(current_question),
        },
        "runtime_scope": {
            "case_id": run.case_id,
            "session_id": str(getattr(run, "session_id", "") or ""),
            "current_case_only": True,
        },
        "conversation_focus": {
            "current_topic": recent_context.get("current_topic", ""),
            "last_user_question": _last_message_content(recent_messages, "user")[:220],
            "last_answer_summary": _last_message_content(recent_messages, "assistant")[:260],
            "referenced_source_refs": recent_context.get("referenced_source_refs", [])[:8],
        },
        "structured_task_state": compact_task_state(state.get("task_state") or {}),
        "turn_relation": state.get("turn_relation") or {},
        "intent_example_signal": _intent_example_signal(state.get("biencoder_frame") or {}),
        "active_reference_objects": _active_reference_objects(recent_messages, session),
        "slot_repair_hints": {
            "candidate_fields": _candidate_fields_for_question(
                current_question,
                str(getattr(session, "current_topic", "") or ""),
            ),
            "explicit_entities": (state.get("entity_frame") or {}).get("explicit_entities", {}),
            "slot_candidates": (state.get("entity_frame") or {}).get("slot_candidates", {}),
        },
        "capability_routing_hint": {
            "L1": "查询当前案件事实，例如就诊、材料、费用、结算、状态、申报人信息。",
            "L2": "基于案件事实做汇总、计数、解释、比对、风险原因或审核建议整理。",
            "L3": "查询政策依据、医保规定、报销口径、专家政策分析。",
        },
        "long_term_memory": {},
        "governance_min": _governance_constraints(),
    }


def _apply_fast_rule_semantics(state: dict[str, Any]) -> dict[str, Any]:
    """Apply the deterministic rule tier without another durable graph node."""

    run = state["run"]
    text = str(getattr(state.get("user_message"), "content", "") or "")
    shortcut = detect_shortcut(text)
    if shortcut is None:
        shortcut = detect_recent_answer_count_shortcut(
            text,
            state.get("recent_messages", []),
        )
    if shortcut is not None:
        from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
            _shortcut_plan,
        )

        plan = _shortcut_plan(shortcut)
        state["shortcut"] = shortcut.to_state()
        state["rule_candidate_plan"] = plan.model_dump(mode="json")
        state["light_semantic_frame"] = _semantic_frame_from_plan(
            state["rule_candidate_plan"],
            source="rule_precheck",
            confidence=1.0,
            evidence_need="none",
        )
        state["next_action"] = "build_perceptual_state"
        return {"rule_hit": True, "rule_kind": str(shortcut.kind)}

    rewrite_followup = _answer_rewrite_followup_signal(state)
    if rewrite_followup is not None:
        state["light_semantic_frame"] = rewrite_followup
        state["next_action"] = "build_perceptual_state"
        return {"rule_hit": True, "rule_kind": "answer_rewrite_followup"}

    clarification_response = _clarification_response_signal(state)
    if clarification_response is not None:
        state["light_semantic_frame"] = clarification_response
        state["next_action"] = "build_perceptual_state"
        return {"rule_hit": True, "rule_kind": "clarification_response"}

    plan = _deterministic_expert_plan(run.case_id, text)
    if plan is None:
        plan = _deterministic_followup_plan(
            run.case_id,
            text,
            state.get("recent_messages", []),
            state.get("session"),
        )
    if plan is None:
        plan = _deterministic_single_field_plan(run.case_id, text)
    if plan is not None:
        plan_payload = plan.model_dump(mode="json")
        state["rule_candidate_plan"] = plan_payload
        state["light_semantic_frame"] = _semantic_frame_from_plan(
            plan_payload,
            source="rule_precheck",
            confidence=1.0,
            evidence_need=_evidence_need_for_plan(plan_payload),
        )
        state["next_action"] = "build_perceptual_state"
        return {"rule_hit": True, "rule_kind": "deterministic_plan"}

    state["next_action"] = "intent_example_biencoder"
    return {"rule_hit": False, "rule_kind": "none"}


def _build_decision_context(
    state: dict[str, Any],
    *,
    context_need: dict[str, Any],
) -> dict[str, Any]:
    """Build the bounded perception-to-decision context in memory."""

    run = state["run"]
    session = state.get("session")
    user_message = state.get("user_message")
    recent = state.get("recent_messages", []) or []
    context_ref = f"decision_context:{run.run_id}"
    return DecisionContext(
        context_ref=context_ref,
        runtime_context={
            "run_id": run.run_id,
            "trace_id": run.run_id,
            "session_id": run.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route_version": "caser-perception-decision-v0.2",
        },
        governance_context={
            "actor_id": state.get("actor_id"),
            "tenant": "local",
            "role": "auditor",
            "permissions": ["case_agent:read"],
            "current_case_only": True,
            "read_only": True,
            "no_final_audit_decision": True,
            "must_cite_evidence": True,
        },
        domain_artifact_context={
            "case_ref": run.case_id,
            "case_state_summary": {
                "session_title": getattr(session, "title", ""),
                "active_stage": getattr(user_message, "active_stage", None),
            },
            "l1_result_ref": [],
            "l2_result_ref": [],
            "l3_result_ref": [],
            "answer_ref": _latest_answer_ref(recent),
            "source_ref": _recent_source_refs(recent, session),
            "evidence_ref": [],
        },
        working_memory_context={
            "recent_window": _recent_message_window(
                recent,
                limit=_recent_context_limit(state),
            ),
            "rolling_summary": str(getattr(session, "session_summary", "") or "")[:800],
            "current_topic": str(getattr(session, "current_topic", "") or "")[:300],
            "structured_task_state": compact_task_state(state.get("task_state") or {}),
            "turn_relation": state.get("turn_relation") or {},
            "long_memory_index": None,
            "vector_memory_refs": [],
        },
        context_summary={
            "context_need": context_need,
            "semantic_source": (
                state.get("light_semantic_frame")
                or state.get("deep_semantic_frame")
                or {}
            ).get("source"),
        },
    ).model_dump(mode="json")


def _intent_example_signal(frame: dict[str, Any]) -> dict[str, Any]:
    candidates = frame.get("top_candidates") or frame.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    return {
        "hit_status": str(frame.get("decision") or "fallback")[:40],
        "reason": str(frame.get("reason") or "")[:120],
        "top_candidates": [
            {
                "intent": str(item.get("intent") or item.get("business_intent") or "")[:80],
                "layer_hint": str(item.get("layer_hint") or item.get("layer") or "")[:20],
                "score": item.get("score") or item.get("confidence"),
                "matched_example": str(item.get("matched_example") or item.get("text") or "")[:120],
            }
            for item in candidates[:3]
            if isinstance(item, dict)
        ],
    }


def _active_reference_objects(recent_messages: list[Any], session: Any) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    session_refs = list(getattr(session, "referenced_source_refs", []) or [])[:8]
    for ref in session_refs:
        objects.append(
            {
                "ref": str(ref)[:120],
                "type": _source_type_from_ref(str(ref)),
                "business_module": _business_module_from_ref(str(ref)),
                "label": str(ref)[:120],
            }
        )
    for message in reversed(recent_messages[-6:]):
        answer_payload = getattr(message, "answer_payload", None)
        if answer_payload is None:
            continue
        try:
            for source in list(answer_payload.sources or [])[:8]:
                ref = str(getattr(source, "source_ref", "") or "")
                if not ref:
                    continue
                objects.append(
                    {
                        "ref": ref[:120],
                        "type": str(getattr(source, "source_type", "") or _source_type_from_ref(ref))[:80],
                        "business_module": _business_module_from_ref(ref),
                        "label": str(getattr(source, "title", "") or ref)[:160],
                    }
                )
        except AttributeError:
            continue
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in objects:
        ref = str(item.get("ref") or "")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        deduped.append(item)
    return deduped[:8]


def _last_message_content(recent_messages: list[Any], role: str) -> str:
    for message in reversed(recent_messages):
        if str(getattr(message, "role", "") or "") != role:
            continue
        return str(getattr(message, "content", "") or "")
    return ""


def _looks_like_follow_up_question(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) <= 12 and any(marker in compact for marker in ("这个", "它", "为什么", "多少", "几项", "哪些")):
        return True
    return any(
        marker in compact
        for marker in (
            "为什么这么高",
            "为什么会触发",
            "为什么触发",
            "这个是什么意思",
            "这个呢",
            "它呢",
            "一共有几项",
            "举个例子",
            "举例说明",
            "例子说明",
            "简单说明",
            "简单说",
            "通俗点",
            "通俗说明",
            "直观说明",
            "换句话说",
            "再解释一下",
        )
    )


def _answer_rewrite_followup_signal(state: dict[str, Any]) -> dict[str, Any] | None:
    user_message = state.get("user_message")
    text = str(getattr(user_message, "content", "") or "")
    if not _looks_like_answer_rewrite_followup(text):
        return None
    preferred_answer_ref = str(
        (
            compact_task_state(state.get("task_state") or {})
            .get("context_snapshot", {})
            .get("answer_ref")
        )
        or ""
    )
    previous = _assistant_message_for_answer_ref(
        state.get("recent_messages", []),
        preferred_answer_ref,
    ) or _latest_assistant_message_with_answer(state.get("recent_messages", []))
    if previous is None:
        return None

    message_id = str(getattr(previous, "message_id", "") or "")
    answer_ref = f"answer:{message_id}" if message_id else _latest_answer_ref([previous])
    if not answer_ref:
        return None
    source_refs = _source_refs_from_message(previous)
    rewrite_mode = _answer_rewrite_mode(text)
    return {
        "source": "rule_precheck",
        "utterance_type": "question",
        "business_intent": "case_task",
        "answer_shape": "analysis",
        "evidence_need": "previous_answer",
        "focus_candidate": {
            "target_layer_hint": "none",
            "target_objects": [answer_ref],
            "rewritten_query": _rewrite_previous_answer_question(rewrite_mode),
            "action": "explain_previous_answer",
            "rewrite_mode": rewrite_mode,
        },
        "slots": {
            "answer_strategy": "reuse_previous_answer",
            "requires_new_evidence": False,
            "reuse_answer_ref": answer_ref,
            "reuse_source_refs": source_refs,
            "rewrite_mode": rewrite_mode,
        },
        "missing_slots": [],
        "confidence": 0.96,
    }


def _looks_like_answer_rewrite_followup(text: str) -> bool:
    return is_answer_rewrite_only(text)


def _latest_assistant_message_with_answer(recent_messages: Any) -> Any | None:
    for message in reversed(list(recent_messages or [])):
        if _assistant_message_is_reusable(message):
            return message
    return None


def _assistant_message_for_answer_ref(recent_messages: Any, answer_ref: str) -> Any | None:
    if not answer_ref.startswith("answer:"):
        return None
    message_id = answer_ref.removeprefix("answer:")
    for message in reversed(list(recent_messages or [])):
        if str(getattr(message, "message_id", "") or "") != message_id:
            continue
        return message if _assistant_message_is_reusable(message) else None
    return None


def _assistant_message_is_reusable(message: Any) -> bool:
    if str(getattr(message, "role", "") or "") != "assistant":
        return False
    answer = getattr(message, "answer_payload", None)
    if answer is None:
        return bool(str(getattr(message, "content", "") or "").strip())
    if isinstance(answer, dict):
        display_mode = str(answer.get("display_mode") or "plain")
        metadata = answer.get("metadata") if isinstance(answer.get("metadata"), dict) else {}
        content_blocks = answer.get("content_blocks") or []
    else:
        display_mode = str(getattr(answer, "display_mode", "plain") or "plain")
        metadata = getattr(answer, "metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        content_blocks = getattr(answer, "content_blocks", []) or []
    if display_mode in {"unavailable", "error"}:
        return False
    status = str(metadata.get("status") or metadata.get("intent") or "").lower()
    if status in {
        "clarification_required",
        "unavailable",
        "error",
        "failed",
        "waiting_for_user",
    }:
        return False
    if metadata.get("missing_slots"):
        return False
    return bool(content_blocks or str(getattr(message, "content", "") or "").strip())


def _source_refs_from_message(message: Any) -> list[str]:
    refs: list[str] = []
    refs.extend(str(ref) for ref in list(getattr(message, "source_refs", []) or []) if ref)
    answer_payload = getattr(message, "answer_payload", None)
    if answer_payload is not None:
        try:
            for block in answer_payload.content_blocks:
                refs.extend(str(ref) for ref in list(block.source_refs or []) if ref)
            refs.extend(str(source.source_ref) for source in list(answer_payload.sources or []))
        except AttributeError:
            pass
    return list(dict.fromkeys(ref for ref in refs if ref))[:20]


def _answer_rewrite_mode(text: str) -> str:
    return detect_answer_rewrite_mode(text) or "explain"


def _rewrite_previous_answer_question(rewrite_mode: str) -> str:
    if rewrite_mode == "translate_zh":
        return "将上一轮回答准确改写为中文。"
    if rewrite_mode == "translate_en":
        return "将上一轮回答准确改写为英文。"
    if rewrite_mode == "example":
        return "用一个简单直观的例子说明上一轮回答。"
    if rewrite_mode == "simplify":
        return "用更通俗简洁的方式解释上一轮回答。"
    if rewrite_mode == "paraphrase":
        return "换句话解释上一轮回答。"
    if rewrite_mode == "summarize":
        return "简短总结上一轮回答。"
    if rewrite_mode == "list":
        return "将上一轮回答整理为清晰列表。"
    if rewrite_mode == "expand":
        return "在不增加新事实的前提下展开解释上一轮回答。"
    return "继续解释上一轮回答。"


def _clarification_response_signal(state: dict[str, Any]) -> dict[str, Any] | None:
    relation = state.get("turn_relation") or {}
    if relation.get("kind") != "clarification_response":
        return None
    pending = (state.get("resume_context") or {}).get("pending_clarification")
    if not isinstance(pending, dict):
        pending = (state.get("task_state") or {}).get("pending_clarification")
    if not isinstance(pending, dict):
        return None
    missing = [str(item) for item in pending.get("missing_slots", []) if item]
    user_message = state.get("user_message")
    answer = str(getattr(user_message, "content", "") or "").strip()
    if len(missing) != 1 or not answer or len(answer) > 200:
        return None

    semantic = pending.get("semantic") if isinstance(pending.get("semantic"), dict) else {}
    task_state = compact_task_state(state.get("task_state") or {})
    intent = str(
        semantic.get("intent")
        or pending.get("intent")
        or task_state.get("core_intent")
        or "case_task"
    )
    if intent not in {"general_help", "case_task", "expert_task"}:
        intent = str(task_state.get("core_intent") or "case_task")
    if intent not in {"general_help", "case_task", "expert_task"}:
        intent = "case_task"

    slots = dict(pending.get("slots") or {})
    slots[missing[0]] = answer
    rewritten = str(semantic.get("rewritten_query") or task_state.get("goal") or "")
    return {
        "source": "rule_precheck",
        "utterance_type": "clarification_response",
        "business_intent": intent,
        "answer_shape": semantic.get("answer_shape") or "overview",
        "evidence_need": semantic.get("evidence_need") or "case_context",
        "focus_candidate": {
            "target_layer_hint": semantic.get("target_layer_hint") or "none",
            "target_objects": semantic.get("target_objects") or [],
            "information_needs": semantic.get("information_needs") or [],
            "rewritten_query": f"{rewritten}；补充{missing[0]}：{answer}"[:500],
            "action": semantic.get("action") or "query_fact",
        },
        "slots": slots,
        "missing_slots": [],
        "confidence": 0.98,
    }


def _candidate_fields_for_question(question: str, current_topic: str) -> list[str]:
    text = f"{question} {current_topic}"
    fields: list[str] = []
    if any(marker in text for marker in ("风险", "高", "评分", "分数")):
        fields.extend(["risk_score", "risk_level", "overall_strength", "risk_related_tasks"])
    if any(marker in text for marker in ("规则", "触发", "命中", "核验")):
        fields.extend(["rule_id", "rule_status", "verification_reason", "suggested_action"])
    if any(marker in text for marker in ("材料", "就诊", "诊疗", "医院", "处方", "结算")):
        fields.extend(["material_id", "material_type", "material_status", "occurred_at", "institution"])
    if any(marker in text for marker in ("政策", "规定", "依据", "口径")):
        fields.extend(["policy_domain", "jurisdiction", "service_date", "scenario"])
    return list(dict.fromkeys(fields))[:8]


def _source_type_from_ref(ref: str) -> str:
    if ref.startswith("policy") or ref.startswith("rag_policy"):
        return "policy_evidence"
    if ref.startswith("rule:"):
        return "rule_check"
    if ref.startswith("material:"):
        return "business_fact"
    if ref.startswith("risk:"):
        return "business_fact"
    return "source_ref"


def _business_module_from_ref(ref: str) -> str:
    if ref.startswith("risk:"):
        return "综合风险评分"
    if ref.startswith("rule:"):
        return "业务规则核验"
    if ref.startswith("material:"):
        return "材料与就诊记录"
    if ref.startswith("policy") or ref.startswith("rag_policy"):
        return "政策依据"
    return "当前会话引用"


def _parse_semantic_result(content: str) -> SemanticParseResult:
    text = content.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1].strip()
    payload = json.loads(text or "{}")
    payload = _normalize_semantic_payload(payload)
    return SemanticParseResult.model_validate(payload)


def _normalize_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    if isinstance(payload.get("query_semantics"), dict):
        payload = _semantic_payload_from_legacy_plan(payload)
    target_layer_hint = _normalize_layer_hint(payload.get("target_layer_hint"))
    missing_slots = payload.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = []
    filled_slots = payload.get("filled_slots")
    if not isinstance(filled_slots, dict):
        filled_slots = {}
    target_objects = payload.get("target_objects")
    if not isinstance(target_objects, list):
        target_objects = []
    information_needs = payload.get("information_needs")
    if not isinstance(information_needs, list):
        information_needs = target_objects
    safety_flags = payload.get("safety_flags")
    if not isinstance(safety_flags, list):
        safety_flags = []
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = max(0.0, min(confidence_value, 1.0))
    return {
        "intent": _normalize_semantic_intent(
            payload.get("intent"),
            target_layer_hint=target_layer_hint,
            missing_slots=missing_slots,
        ),
        "action": str(payload.get("action") or "general_help")[:80],
        "answer_shape": _normalize_answer_shape(payload.get("answer_shape")),
        "target_layer_hint": target_layer_hint,
        "target_objects": [str(item)[:80] for item in target_objects[:8] if str(item or "").strip()],
        "information_needs": [
            str(item)[:80]
            for item in information_needs[:12]
            if str(item or "").strip()
        ],
        "rewritten_query": str(payload.get("rewritten_query") or "")[:500],
        "filled_slots": filled_slots,
        "missing_slots": [str(item)[:80] for item in missing_slots[:8]],
        "confidence": confidence_value,
        "need_clarification": bool(payload.get("need_clarification") or missing_slots),
        "safety_flags": [str(item)[:80] for item in safety_flags[:8]],
    }


def _semantic_payload_from_legacy_plan(payload: dict[str, Any]) -> dict[str, Any]:
    semantics = dict(payload.get("query_semantics") or {})
    execution_plan = payload.get("execution_plan")
    layer_hint = _target_layer_from_execution_plan(execution_plan if isinstance(execution_plan, list) else [])
    return {
        "intent": semantics.get("intent") or "general_help",
        "action": _action_from_granularity(semantics.get("granularity")),
        "answer_shape": semantics.get("granularity") or "overview",
        "target_layer_hint": layer_hint,
        "target_objects": [],
        "information_needs": semantics.get("information_needs") or [],
        "rewritten_query": semantics.get("user_goal") or "",
        "filled_slots": {},
        "missing_slots": semantics.get("missing_slots") or [],
        "confidence": 0.7,
        "need_clarification": bool(semantics.get("missing_slots")),
        "safety_flags": [],
    }


def _target_layer_from_execution_plan(execution_plan: list[Any]) -> str:
    layers = [str(item.get("layer") or "").upper() for item in execution_plan if isinstance(item, dict)]
    if "L3" in layers:
        return "L3"
    if "L2" in layers:
        return "L2"
    if "L1" in layers:
        return "L1"
    return "none"


def _normalize_layer_hint(value: Any) -> str:
    layer = str(value or "").upper()
    return layer if layer in {"L1", "L2", "L3"} else "none"


def _normalize_semantic_intent(value: Any, *, target_layer_hint: str, missing_slots: list[Any]) -> str:
    intent = str(value or "").strip()
    if intent in {"general_help", "case_task", "expert_task", "clarification_required"}:
        return intent
    if missing_slots:
        return "clarification_required"
    if target_layer_hint == "L3":
        return "expert_task"
    if target_layer_hint in {"L1", "L2"}:
        return "case_task"
    return "general_help"


def _normalize_answer_shape(value: Any) -> str:
    answer_shape = str(value or "").strip()
    if answer_shape in {"single_field", "field_group", "list", "detail", "overview", "analysis"}:
        return answer_shape
    return "overview"


def _action_from_granularity(value: Any) -> str:
    granularity = str(value or "")
    if granularity == "list":
        return "list_records"
    if granularity == "single_field":
        return "query_fact"
    if granularity == "analysis":
        return "explain_reason"
    return "general_help"


def _fallback_semantic_result() -> SemanticParseResult:
    return SemanticParseResult(
        intent="general_help",
        action="general_help",
        answer_shape="overview",
        target_layer_hint="none",
        target_objects=[],
        information_needs=[],
        rewritten_query="",
        filled_slots={},
        missing_slots=[],
        confidence=0.0,
        need_clarification=False,
        safety_flags=[],
    )


def _semantic_frame_from_semantic_result(
    semantic_payload: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    layer_hint = _normalize_layer_hint(semantic_payload.get("target_layer_hint"))
    slots = dict(semantic_payload.get("filled_slots") or {})
    rewritten_query = str(semantic_payload.get("rewritten_query") or "").strip()
    if rewritten_query:
        slots["rewritten_query"] = rewritten_query
    return {
        "source": source,
        "utterance_type": "question",
        "business_intent": semantic_payload.get("intent") or "general_help",
        "answer_shape": semantic_payload.get("answer_shape") or "overview",
        "evidence_need": _evidence_need_for_layer(
            layer_hint,
            str(semantic_payload.get("intent") or ""),
        ),
        "focus_candidate": {
            "target_layer_hint": layer_hint,
            "target_objects": list(semantic_payload.get("target_objects") or [])[:8],
            "information_needs": list(semantic_payload.get("information_needs") or [])[:12],
            "rewritten_query": rewritten_query,
            "action": semantic_payload.get("action") or "",
        },
        "slots": slots,
        "missing_slots": semantic_payload.get("missing_slots") or [],
        "confidence": semantic_payload.get("confidence") or 0.0,
    }


def _evidence_need_for_layer(layer_hint: str, intent: str) -> str:
    if layer_hint == "L3" or intent == "expert_task":
        return "policy_evidence"
    if layer_hint == "L2":
        return "derived_fact"
    if layer_hint == "L1":
        return "case_fact"
    return "none"


def _semantic_frame_from_plan(
    plan_payload: dict[str, Any],
    *,
    source: str,
    confidence: float,
    evidence_need: str,
) -> dict[str, Any]:
    semantics = dict(plan_payload.get("query_semantics") or {})
    execution_plan = plan_payload.get("execution_plan")
    layer_hint = _target_layer_from_execution_plan(
        execution_plan if isinstance(execution_plan, list) else []
    )
    return {
        "source": source,
        "utterance_type": "question",
        "business_intent": semantics.get("intent") or "general_help",
        "answer_shape": semantics.get("granularity") or "overview",
        "evidence_need": evidence_need,
        "focus_candidate": {
            "target_layer_hint": layer_hint,
            "target_objects": [],
            "rewritten_query": semantics.get("user_goal") or "",
            "action": _action_from_granularity(semantics.get("granularity")),
        },
        "slots": {},
        "missing_slots": semantics.get("missing_slots") or [],
        "confidence": confidence,
    }


def _evidence_need_for_plan(plan_payload: dict[str, Any]) -> str:
    plan = plan_payload.get("execution_plan")
    if not isinstance(plan, list) or not plan:
        return "none"
    layers = {str(item.get("layer") or "").upper() for item in plan if isinstance(item, dict)}
    if "L3" in layers:
        return "policy_evidence"
    if "L2" in layers:
        return "derived_fact"
    return "case_fact"


def _minimal_context(state: dict[str, Any]) -> MinimalPlanningContext:
    run = state["run"]
    session = state.get("session")
    user_message = state.get("user_message")
    return MinimalPlanningContext(
        question=str(getattr(user_message, "content", "") or ""),
        case_id=run.case_id,
        session_id=run.session_id,
        actor_id=str(state.get("actor_id") or ""),
        governance_min=_governance_constraints(),
        current_topic=str(getattr(session, "current_topic", "") or "")[:300],
        referenced_refs=list(getattr(session, "referenced_source_refs", []) or [])[:20],
    )


def _needs_full_context(state: dict[str, Any], semantic: dict[str, Any]) -> bool:
    if state.get("llm_semantic_result"):
        return True
    if semantic.get("missing_slots"):
        return False
    if semantic.get("evidence_need") in {"previous_answer", "source_ref"}:
        return True
    return False


def _context_packs_for_semantic(semantic: dict[str, Any]) -> list[str]:
    packs = ["runtime", "governance"]
    evidence_need = semantic.get("evidence_need")
    if evidence_need in {"case_fact", "derived_fact", "policy_evidence"}:
        packs.append("domain_artifacts")
    if evidence_need in {"previous_answer", "source_ref"} or semantic.get("source") == "llm_semantic_parser":
        packs.append("working_memory")
    return list(dict.fromkeys(packs))


def _governance_constraints() -> dict[str, bool]:
    return {
        "read_only": True,
        "current_case_only": True,
        "no_final_audit_decision": True,
        "must_cite_evidence": True,
    }


def _latest_answer_ref(recent_messages: Any) -> str | None:
    for message in reversed(list(recent_messages or [])):
        if getattr(message, "role", "") == "assistant":
            message_id = str(getattr(message, "message_id", "") or "")
            return f"answer:{message_id}" if message_id else None
    return None


def _recent_source_refs(recent_messages: Any, session: Any) -> list[str]:
    refs: list[str] = []
    refs.extend(str(ref) for ref in list(getattr(session, "referenced_source_refs", []) or [])[:12])
    for message in reversed(list(recent_messages or [])[-8:]):
        refs.extend(str(ref) for ref in list(getattr(message, "source_refs", []) or [])[:12])
        answer_payload = getattr(message, "answer_payload", None)
        if answer_payload is not None:
            try:
                refs.extend(str(source.source_ref) for source in answer_payload.sources)
            except AttributeError:
                pass
    return list(dict.fromkeys(ref for ref in refs if ref))[:20]


def _recent_context_limit(state: dict[str, Any]) -> int:
    task_state = compact_task_state(state.get("task_state") or {})
    return 2 if task_state.get("task_id") else 6


def _recent_message_window(
    recent_messages: Any,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    window: list[dict[str, Any]] = []
    for message in list(recent_messages or [])[-max(1, min(limit, 6)):]:
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        window.append(
            {
                "role": getattr(message, "role", ""),
                "message_id": getattr(message, "message_id", ""),
                "content": content[:500],
                "source_refs": list(getattr(message, "source_refs", []) or [])[:8],
            }
        )
    return window
