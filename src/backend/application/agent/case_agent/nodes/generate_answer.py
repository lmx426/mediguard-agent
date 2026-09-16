"""Case Agent final answer generation node."""

from __future__ import annotations
import json
import re
from typing import Any

from src.backend.application.agent.case_agent.artifacts import artifact_results
from src.backend.application.agent.case_agent.answer_assembler import (
    build_deterministic_answer,
)
from src.backend.application.agent.case_agent.prompts.final_answer import (
    build_final_answer_prompt,
)
from src.backend.application.agent.case_agent.shortcut import (
    CaseAgentShortcut,
    build_shortcut_answer,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentCitation,
    CaseAgentClaim,
    CaseAgentContentBlock,
    CaseAgentSource,
)

from ._errors import state_error


def generate_answer_node(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Generate the final structured JSON answer."""

    try:
        service._repository.update_run(
            state["run"].run_id,
            current_node="generating",
            model_call_count=state.get("model_call_count", 0),
            tool_call_count=state.get("tool_call_count", 0),
        )
        service._repository.append_event(
            state["run"].run_id,
            "generating",
            "正在生成结构化回答",
            {},
        )

        if getattr(service, "_fast_mode_enabled", False) and state.get("shortcut"):
            answer = build_shortcut_answer(CaseAgentShortcut(**state["shortcut"]))
            final_response = ModelResponse(content=answer.model_dump_json())
        elif deterministic_answer := _policy_expert_answer_from_state(state):
            final_response = ModelResponse(
                content=deterministic_answer.model_dump_json(),
                metrics={
                    "deterministic_handoff": True,
                    "capability": "ask_policy_expert",
                    "source_ref_count": len(
                        deterministic_answer.content_blocks[0].source_refs
                    ),
                    "model_call_skipped": True,
                },
            )
        elif deterministic_answer := build_deterministic_answer(state):
            final_response = ModelResponse(
                content=deterministic_answer.model_dump_json(),
                metrics={
                    "deterministic_answer": True,
                    "capabilities": deterministic_answer.metadata.get("capabilities_used", []),
                    "model_call_skipped": True,
                },
            )
        elif state.get("answer_strategy") == "reuse_previous_answer":
            rewrite_response = service._gateway.complete(
                messages=_previous_answer_rewrite_messages(state),
                tools=[],
                require_json=False,
                model=service._generator_model,
                thinking_enabled=False,
                max_tokens=min(getattr(service, "_answer_max_tokens", 1536), 384),
                timeout_seconds=min(
                    float(getattr(service, "_answer_timeout_seconds", 30)),
                    8.0,
                ),
            )
            state["model_call_count"] = state.get("model_call_count", 0) + 1
            recorder = getattr(service, "record_model_call_metrics", None)
            if callable(recorder):
                recorder(state, "generate_answer_rewrite", rewrite_response)
            rewrite_answer = _wrap_previous_answer_rewrite(
                state,
                rewrite_response.content or "",
            )
            final_response = ModelResponse(
                content=rewrite_answer.model_dump_json(),
                metrics={
                    **dict(rewrite_response.metrics),
                    "rewrite_text_wrapped": True,
                    "rewrite_mode": state.get("answer_rewrite_mode") or "explain",
                    "model_call_count": 1,
                },
            )
        else:
            final_response = service._gateway.complete(
                messages=service._build_generation_messages(state)
                + [
                    {
                        "role": "system",
                        "content": build_final_answer_prompt(
                            display_mode=(
                                state.get("answer_policy", {}).get("display_mode")
                                or state.get("context_plan", {}).get("display_mode")
                                or "plain"
                            ),
                            fast_mode_enabled=getattr(service, "_fast_mode_enabled", False),
                            draft_only=True,
                        ),
                    }
                ],
                tools=[],
                require_json=True,
                model=service._generator_model,
                thinking_enabled=False,
                max_tokens=getattr(service, "_answer_max_tokens", 1536),
                timeout_seconds=getattr(service, "_answer_timeout_seconds", 30),
            )
            state["model_call_count"] = state.get("model_call_count", 0) + 1
            recorder = getattr(service, "record_model_call_metrics", None)
            if callable(recorder):
                recorder(state, "generate_answer", final_response)

        state["final_response"] = final_response
        state["next_action"] = "validate_answer"
        return state
    except Exception as exc:
        return state_error(state, exc)


def _policy_expert_answer_from_state(state: dict[str, Any]) -> CaseAgentAnswer | None:
    """Build final answer directly from a structured Policy Expert result."""

    policy = state.get("answer_policy")
    if not isinstance(policy, dict) or policy.get("display_mode") != "grounded":
        return None

    available_sources = {
        source.source_ref: source
        for source in state.get("available_sources", [])
        if isinstance(source, CaseAgentSource)
    }
    if not available_sources:
        return None

    result_items = artifact_results(state) or state.get("capability_results", [])
    for capability, result in result_items:
        if capability != "ask_policy_expert" or getattr(result, "status", "") != "success":
            continue
        payload = getattr(result, "payload", {})
        if not isinstance(payload, dict) or payload.get("section_key") != "policy_expert":
            continue
        expert_payload = payload.get("payload")
        if not isinstance(expert_payload, dict):
            continue
        answer_text = str(expert_payload.get("expert_answer") or "").strip()
        answer_markdown = str(expert_payload.get("answer_markdown") or "").strip()
        claims = _policy_claims(expert_payload.get("claims"), available_sources)
        citations = _policy_citations(expert_payload.get("citations"), available_sources)
        answer_markdown, claims, citations = _reconcile_policy_output(
            answer_markdown=answer_markdown,
            claims=claims,
            citations=citations,
            need_answers=expert_payload.get("need_answers"),
            answer_mode=expert_payload.get("answer_mode"),
        )
        contract_refs = [
            ref
            for item in [*claims, *citations]
            for ref in item.source_refs
            if ref in available_sources
        ]
        source_refs = list(dict.fromkeys(contract_refs)) or _policy_expert_source_refs(
            expert_payload,
            getattr(result, "source_refs", []),
            available_sources,
        )
        if answer_markdown:
            answer_text = re.sub(r"\[\d+\]", "", answer_markdown).strip()
        if not answer_text or not source_refs:
            continue

        sources = [available_sources[ref] for ref in source_refs]
        content_blocks = [
            CaseAgentContentBlock(
                text=_clip(answer_text, 1200),
                source_refs=source_refs[:8],
            )
        ]
        suggestion_text = _audit_suggestions_text(
            expert_payload.get("audit_suggestions"),
            source_refs,
        )
        if suggestion_text:
            content_blocks.append(
                CaseAgentContentBlock(
                    text=suggestion_text,
                    source_refs=source_refs,
                )
            )

        insufficient_policy_notice = ""
        if (
            str(expert_payload.get("status") or "") == "insufficient"
            or policy.get("capability_policy") == "expert_evidence_insufficient"
        ):
            insufficient_policy_notice = (
                "政策证据不足，当前仅展示已检索到的可引用依据，"
                "未使用通用知识补充政策事实。"
            )

        return CaseAgentAnswer(
            display_mode="grounded",
            content_blocks=content_blocks,
            sources=sources,
            answer_markdown=_clip(answer_markdown, 4000) if answer_markdown else None,
            claims=claims,
            citations=citations,
            fallback_notice=insufficient_policy_notice,
            metadata={
                "intent": state.get("intent") or "policy_expert_query",
                "requires_citation": True,
                "capabilities_used": ["ask_policy_expert"],
                "deterministic_handoff": True,
                "expert_task_id": payload.get("expert_task_id"),
                "answerability_summary": expert_payload.get("answerability_summary"),
                "claim_count": len(claims),
                "citation_count": len(citations),
                "extracted_fact_count": len(
                    expert_payload.get("extracted_facts")
                    if isinstance(expert_payload.get("extracted_facts"), list)
                    else []
                ),
                "limits": expert_payload.get("limits") or payload.get("limits"),
            },
        )
    return None


def _policy_expert_source_refs(
    expert_payload: dict[str, Any],
    result_refs: Any,
    available_sources: dict[str, CaseAgentSource],
) -> list[str]:
    refs: list[str] = []
    for key in ("source_refs",):
        value = expert_payload.get(key)
        if isinstance(value, list):
            refs.extend(ref for ref in value if isinstance(ref, str))
    citation_items = expert_payload.get("citations", [])
    if not isinstance(citation_items, list):
        citation_items = []
    for item in citation_items:
        if isinstance(item, dict) and isinstance(item.get("source_refs"), list):
            refs.extend(ref for ref in item["source_refs"] if isinstance(ref, str))
    claim_items = expert_payload.get("claims", [])
    if not isinstance(claim_items, list):
        claim_items = []
    for item in claim_items:
        if isinstance(item, dict) and isinstance(item.get("source_refs"), list):
            refs.extend(ref for ref in item["source_refs"] if isinstance(ref, str))
    if isinstance(result_refs, list):
        refs.extend(ref for ref in result_refs if isinstance(ref, str))
    return [
        ref for ref in dict.fromkeys(refs)
        if ref in available_sources
    ][:20]


def _policy_claims(
    value: Any,
    available_sources: dict[str, CaseAgentSource],
) -> list[CaseAgentClaim]:
    if not isinstance(value, list):
        return []
    claims: list[CaseAgentClaim] = []
    for index, item in enumerate(value[:30], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        source_refs = [
            ref for ref in item.get("source_refs", [])
            if isinstance(ref, str) and ref in available_sources
        ][:8]
        if not source_refs:
            continue
        claims.append(
            CaseAgentClaim(
                claim_id=str(item.get("claim_id") or f"claim_{index}")[:80],
                need_id=str(item.get("need_id") or item.get("requirement_id") or "")[:80],
                need_text=str(item.get("need_text") or item.get("label") or "")[:240],
                text=_clip(text, 600),
                fact_refs=[
                    str(ref)[:80] for ref in item.get("fact_refs", [])
                    if isinstance(ref, str)
                ][:8],
                source_refs=source_refs,
                citation_ids=[
                    str(ref)[:80] for ref in item.get("citation_ids", [])
                    if isinstance(ref, str)
                ][:8],
                support_status=str(item.get("support_status") or "supported")[:40],
            )
        )
    return claims


def _policy_citations(
    value: Any,
    available_sources: dict[str, CaseAgentSource],
) -> list[CaseAgentCitation]:
    if not isinstance(value, list):
        return []
    citations: list[CaseAgentCitation] = []
    for index, item in enumerate(value[:30], start=1):
        if not isinstance(item, dict):
            continue
        source_refs = [
            ref for ref in item.get("source_refs", [])
            if isinstance(ref, str) and ref in available_sources
        ][:8]
        if not source_refs:
            continue
        label = item.get("label")
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            label_int = index
        citations.append(
            CaseAgentCitation(
                citation_id=str(item.get("citation_id") or f"cit_{index}")[:80],
                label=min(max(label_int, 1), 99),
                claim_id=(
                    str(item["claim_id"])[:80]
                    if isinstance(item.get("claim_id"), str)
                    else None
                ),
                fact_refs=[
                    str(ref)[:80] for ref in item.get("fact_refs", [])
                    if isinstance(ref, str)
                ][:12],
                evidence_refs=[
                    str(ref)[:220] for ref in item.get("evidence_refs", [])
                    if isinstance(ref, str)
                ][:12],
                source_refs=source_refs,
            )
        )
    return citations


def _reconcile_policy_output(
    *,
    answer_markdown: str,
    claims: list[CaseAgentClaim],
    citations: list[CaseAgentCitation],
    need_answers: Any,
    answer_mode: Any = None,
) -> tuple[str, list[CaseAgentClaim], list[CaseAgentCitation]]:
    """Rebuild display markdown after source filtering to preserve citation closure."""

    unique_claims: list[CaseAgentClaim] = []
    seen_claims: set[tuple[str, tuple[str, ...]]] = set()
    for claim in claims:
        signature = (
            re.sub(r"\s+", "", claim.text),
            tuple(sorted(claim.source_refs)),
        )
        if signature in seen_claims:
            continue
        seen_claims.add(signature)
        unique_claims.append(claim)
    claims = unique_claims
    valid_claim_ids = {claim.claim_id for claim in claims}
    if valid_claim_ids:
        citations = [
            citation
            for citation in citations
            if citation.claim_id is None or citation.claim_id in valid_claim_ids
        ]
    citations = [
        citation.model_copy(update={"label": index})
        for index, citation in enumerate(citations, start=1)
    ]
    citations_by_claim: dict[str, list[CaseAgentCitation]] = {}
    for citation in citations:
        if citation.claim_id:
            citations_by_claim.setdefault(citation.claim_id, []).append(citation)
    claims = [
        claim.model_copy(
            update={
                "citation_ids": [
                    citation.citation_id
                    for citation in citations_by_claim.get(claim.claim_id, [])
                ][:8]
            }
        )
        for claim in claims
    ]

    markdown_labels = {
        int(value) for value in re.findall(r"\[(\d+)\]", answer_markdown)
    }
    valid_labels = {citation.label for citation in citations}
    if (
        not isinstance(need_answers, list)
        and answer_markdown
        and markdown_labels
        and markdown_labels == valid_labels
    ):
        return answer_markdown, claims, citations

    claim_by_need = {claim.need_id: claim for claim in claims if claim.need_id}
    rendered: list[tuple[str, list[int]]] = []
    if isinstance(need_answers, list):
        for item in need_answers:
            if not isinstance(item, dict):
                continue
            need_id = str(item.get("need_id") or "")
            claim = claim_by_need.get(need_id)
            status = str(item.get("status") or "")
            if claim is not None:
                labels = [
                    citation.label
                    for citation in citations_by_claim.get(claim.claim_id, [])
                ]
                rendered.append((claim.text, labels))
            elif status in {"missing", "conflicted"}:
                text = str(item.get("answer_text") or "").strip()
                if text:
                    rendered.append((text, []))
    if not rendered:
        for claim in claims:
            rendered.append(
                (
                    claim.text,
                    [
                        citation.label
                        for citation in citations_by_claim.get(claim.claim_id, [])
                    ],
                )
            )
    if rendered:
        visible: list[str] = []
        seen_text: set[str] = set()
        integrated = str(answer_mode or "") in {"comparison", "policy_explanation"}
        multiple = len(rendered) > 1 and not integrated
        for text, labels in rendered:
            key = re.sub(r"\s+", "", text)
            if not key or key in seen_text:
                continue
            seen_text.add(key)
            prefix = f"{len(visible) + 1}. " if multiple else ""
            suffix = "".join(f"[{label}]" for label in labels)
            visible.append(f"{prefix}{text}{suffix}")
        return (" ".join(visible) if integrated else "\n".join(visible)), claims, citations

    sanitized = re.sub(
        r"\[(\d+)\]",
        lambda match: match.group(0) if int(match.group(1)) in valid_labels else "",
        answer_markdown,
    ).strip()
    return sanitized, claims, citations


def _audit_suggestions_text(value: Any, allowed_refs: list[str]) -> str:
    if not isinstance(value, list) or not value:
        return ""
    allowed = set(allowed_refs)
    suggestions: list[str] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        refs = [
            ref for ref in item.get("source_refs", [])
            if isinstance(ref, str) and ref in allowed
        ]
        if text and refs:
            suggestions.append(text)
    if not suggestions:
        return ""
    return _clip("人工核验参考：" + "；".join(suggestions), 1200)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _previous_answer_rewrite_prompt(mode: str) -> str:
    instructions = {
        "translate_zh": "将上一轮回答准确改写为中文。",
        "translate_en": "将上一轮回答准确改写为英文。",
        "example": (
            "输出一个明确标注为‘假设示例’的具体场景，使用某参保人、参保地、就医地等中性角色，"
            "只把上一轮回答已有的步骤或关系串起来；不能只是换词复述原文。"
        ),
        "simplify": "用更通俗简洁的表达重写上一轮回答。",
        "paraphrase": "换一种说法重写上一轮回答。",
        "summarize": "只提炼上一轮回答的核心结论，删除重复背景和展开说明。",
        "list": "将上一轮回答整理为清晰列表。",
        "expand": "只展开上一轮回答已经包含的关系、原因或步骤，不增加新的政策事实。",
        "explain": "继续解释上一轮回答。",
    }
    return (
        "你只负责改写受控上下文中的 previous_answer。"
        f"{instructions.get(mode, instructions['explain'])}"
        "不得新增案件事实、政策事实、数值、来源或审核结论。"
        "只输出最终正文，不输出 JSON、Markdown 代码围栏、字段名或解释过程。"
    )


def _previous_answer_rewrite_messages(state: dict[str, Any]) -> list[dict[str, Any]]:
    rewrite_input: dict[str, Any] = {}
    context = state.get("answer_context") or {}
    for item in context.get("analysis_inputs", []):
        if isinstance(item, dict) and item.get("mode") == "reuse_previous_answer":
            rewrite_input = {
                "rewrite_mode": str(
                    item.get("rewrite_mode")
                    or state.get("answer_rewrite_mode")
                    or "explain"
                ),
                "previous_answer": str(item.get("previous_answer") or "")[:3000],
                "rewrite_instruction": str(item.get("rewrite_instruction") or "")[:500],
                "allowed_source_refs": [
                    str(ref)
                    for ref in list(item.get("previous_source_refs") or [])[:8]
                    if ref
                ],
            }
            break
    if not rewrite_input:
        rewrite_input = {
            "rewrite_mode": str(state.get("answer_rewrite_mode") or "explain"),
            "previous_answer": "",
            "rewrite_instruction": "",
            "allowed_source_refs": [],
        }
    return [
        {
            "role": "system",
            "content": _previous_answer_rewrite_prompt(rewrite_input["rewrite_mode"]),
        },
        {
            "role": "user",
            "content": "受控改写输入："
            + json.dumps(rewrite_input, ensure_ascii=False, default=str),
        },
    ]


def _wrap_previous_answer_rewrite(
    state: dict[str, Any],
    model_text: str,
) -> CaseAgentAnswer:
    context = state.get("answer_context") or {}
    available_sources = [
        source
        for source in state.get("available_sources", [])
        if isinstance(source, CaseAgentSource)
    ]
    available_refs = {source.source_ref for source in available_sources}
    source_refs = [
        str(ref)
        for ref in context.get("source_refs", [])
        if isinstance(ref, str) and ref in available_refs
    ][:8]
    text = _clean_rewrite_text(model_text)
    if not text:
        for item in context.get("analysis_inputs", []):
            if isinstance(item, dict) and item.get("mode") == "reuse_previous_answer":
                text = str(item.get("previous_answer") or "").strip()
                if text:
                    break
    if not text:
        text = "上一轮回答暂时无法改写，请重新提出具体的表达要求。"

    blocks = [
        CaseAgentContentBlock(
            text=chunk,
            source_refs=source_refs,
        )
        for chunk in _split_rewrite_blocks(text)
    ]
    sources = [source for source in available_sources if source.source_ref in source_refs]
    display_mode = "grounded" if source_refs and sources else "plain"
    return CaseAgentAnswer(
        display_mode=display_mode,
        content_blocks=blocks,
        sources=sources,
        fallback_notice="",
        metadata={
            "intent": state.get("intent") or "case_task",
            "requires_citation": display_mode == "grounded",
            "capabilities_used": ["reuse_previous_answer"],
            "rewrite_mode": state.get("answer_rewrite_mode") or "explain",
            "deterministic_structure_wrap": True,
        },
    )


def _clean_rewrite_text(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text[:2400]


def _split_rewrite_blocks(text: str) -> list[str]:
    if len(text) <= 1200:
        return [text]
    return [text[:1200], text[1200:2400]]
