"""结果终审节点。

负责校验模型生成的证据分析内容，包括引文验证、数值核对、
安全边界检查和边界声明注入，最终保存分析结果。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.backend.domain.agent.entities import (
    AgentCitation,
    AgentSignalReview,
    AgentStatement,
    EvidenceAgentAnalysis,
    EvidenceLedgerItem,
    ReviewAdvisoryGeneratedContentV2,
    SystemRiskPrompt,
)
from ..prompts.loader import build_validation_repair_prompt
from ..safety import AgentOutputValidationError, validate_generated_content, validate_generated_content_relaxed


def finalize_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """对模型生成的证据分析进行终审校验并保存。

    Args:
        service: ReviewAdvisorService 实例，通过它访问仓储、设置等依赖。
        state: 当前 AgentState。

    Returns:
        更新后的 AgentState，包含 completed 或 fail 的下一步路由。
    """

    run_id = state["run_id"]
    eval_variant = state.get("request", {}).get("eval_variant") or "A0"
    service._repository.update_run(run_id, current_node="validating")
    service._repository.append_event(run_id, "validating", "Validating structure, citations, and safety boundaries")
    generated = ReviewAdvisoryGeneratedContentV2.model_validate(state["final_content"])
    # 反序列化证据账本
    ledger = {
        ref: EvidenceLedgerItem.model_validate(item)
        for ref, item in state["ledger"].items()
    }
    try:
        # 根据设置选择严格或宽松校验
        if service._settings.evidence_agent_strict_local_validation:
            citations = validate_generated_content(generated, ledger)
        else:
            citations = validate_generated_content_relaxed(generated, ledger)
    except AgentOutputValidationError as exc:
        # 尝试验证修复
        if _can_repair_validation_error(state, exc, service._settings):
            state["validation_repair_count"] = state.get("validation_repair_count", 0) + 1
            state["force_final_json"] = True
            state["messages"].append(
                {
                    "role": "user",
                    "content": build_validation_repair_prompt(
                        error_code=exc.code,
                        error_message=str(exc),
                        ledger=state["ledger"],
                        eval_variant=eval_variant,
                    ),
                }
            )
            service._repository.append_event(
                run_id,
                "repairing",
                "Agent output failed local validation; retrying one controlled rewrite",
                {"error_code": exc.code},
            )
            service._checkpoint(state, "validation_repair")
            state["next_action"] = "model"
            return state
        return _state_error(state, exc.code, str(exc))
    # 获取 Agent 运行记录
    run = service._repository.get_run(run_id)
    if run is None:
        return _state_error(state, "run_not_found", "Agent run not found")
    system_risk_prompt = SystemRiskPrompt.model_validate(state["system_risk_prompt"])
    legacy_signal_review = _legacy_signal_review(generated)
    clue_statements = [
        AgentStatement(
            statement=f"{item.title}：{item.explanation}",
            source_refs=item.source_refs,
        )
        for item in generated.clue_reviews
    ]
    supporting_evidence = [
        statement
        for statement, item in zip(clue_statements, generated.clue_reviews, strict=True)
        if item.status == "supported"
    ]
    ai_risk_label = _legacy_ai_risk_label(
        risk_level=system_risk_prompt.risk_level,
        relation=generated.evidence_review.relation,
    )
    boundary_notice = "本研判用于辅助人工审核，不构成正式定性或处理结论，不替代审核人员决定。"
    # 构建分析结果
    analysis = EvidenceAgentAnalysis(
        analysis_id=f"eana_{uuid4().hex}",
        run_id=run_id,
        case_id=state["case_id"],
        analysis_type=run.analysis_type,
        status=generated.status,
        system_risk_prompt=system_risk_prompt,
        evidence_review=generated.evidence_review,
        clue_reviews=generated.clue_reviews,
        ai_risk_label=ai_risk_label,
        ai_risk_label_source_refs=generated.evidence_review.source_refs,
        risk_judgement=generated.evidence_review.label,
        risk_judgement_source_refs=generated.evidence_review.source_refs,
        evidence_strength=generated.evidence_review.support_level,
        evidence_strength_source_refs=generated.evidence_review.source_refs,
        key_risk_signals=clue_statements,
        human_review_focus=[],
        risk_overview=generated.evidence_review.summary,
        risk_overview_source_refs=generated.evidence_review.source_refs,
        supporting_evidence=supporting_evidence,
        conflicts=generated.conflicts,
        missing_information=[item.statement for item in generated.missing_information],
        missing_information_details=generated.missing_information,
        signal_review=legacy_signal_review,
        verification_checklist=generated.verification_checklist,
        citations=citations,
        boundary_notice=boundary_notice,
        input_fingerprint=state["input_fingerprint"],
        model_name=service._settings.llm_model,
        prompt_version=service._settings.evidence_agent_prompt_version,
        tool_version=service._settings.evidence_agent_tool_version,
        created_at=datetime.now(timezone.utc),
    )
    # 保存分析结果并更新运行状态
    service._repository.save_analysis(analysis)
    service._repository.update_run(
        run_id,
        status=generated.status,
        current_node="completed",
        model_call_count=state["model_call_count"],
        tool_call_count=state["tool_call_count"],
    )
    service._repository.append_event(
        run_id,
        generated.status,
        "Review Advisor analysis completed" if generated.status == "complete" else "Review Advisor generated a partial analysis",
        {"analysis_id": analysis.analysis_id},
    )
    if generated.status in {"complete", "partial"}:
        publish_caser = getattr(service, "_publish_caser_context_event", None)
        if callable(publish_caser):
            publish_caser("review_advisor.completed", state["case_id"])
    service._checkpoint(state, "completed")
    state["next_action"] = "completed"
    return state


def _can_repair_validation_error(
    state: dict[str, Any],
    exc: AgentOutputValidationError,
    settings: Any,
) -> bool:
    """判断是否可以重试修复输出验证错误。

    Args:
        state: 当前 AgentState。
        exc: 验证异常实例。
        settings: 应用设置对象。

    Returns:
        True 表示可以重试，False 表示不可修复。
    """

    return (
        exc.code in {
            "numeric_mismatch",
            "citation_missing",
            "citation_invalid",
            "prohibited_conclusion",
            "evidence_review_inconsistent",
            "sensitive_output",
        }
        and state.get("validation_repair_count", 0) < 1
        and state["model_call_count"] < settings.evidence_agent_max_model_calls
    )


def _state_error(state: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    """设置错误状态并路由到 fail 节点。"""
    state["error_code"] = code
    state["error_message"] = message
    state["next_action"] = "fail"
    return state


def _legacy_signal_review(
    generated: ReviewAdvisoryGeneratedContentV2,
) -> AgentSignalReview:
    """将 V2 线索复核映射到旧 API 字段，供历史前端和持久化结果兼容。"""

    groups: dict[str, list[AgentStatement]] = {
        "supported": [],
        "needs_review": [],
        "unconfirmed": [],
    }
    for item in generated.clue_reviews:
        groups[item.status].append(
            AgentStatement(
                statement=f"{item.title}：{item.explanation}",
                source_refs=item.source_refs,
            )
        )
    return AgentSignalReview(
        supported_clues=groups["supported"],
        needs_review=groups["needs_review"],
        unconfirmed_items=[item.statement for item in groups["unconfirmed"]],
        supplementary_review_hints=[],
    )


def _legacy_ai_risk_label(*, risk_level: str, relation: str) -> str:
    """仅为旧 API 字段提供确定性兼容值，不作为前端研判结论展示。"""

    if relation in {"insufficient_evidence", "inconsistent"}:
        return "当前证据不足"
    if risk_level == "low" and relation == "supports":
        return "未发现明确疑似欺诈风险线索"
    return "存在疑似欺诈风险线索"
