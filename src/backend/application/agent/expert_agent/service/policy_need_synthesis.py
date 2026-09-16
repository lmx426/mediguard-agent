"""One-shot language synthesis for complex information-need policy answers."""

from __future__ import annotations

import json
import re
from typing import Any

from src.backend.application.agent.expert_agent.service.policy_need_contracts import (
    NeedFirstPolicyAnswer,
)
from src.backend.application.agent.expert_agent.service.policy_need_answer import (
    render_information_need_markdown,
)


INTERNAL_MARKERS = (
    "source_id",
    "node_id",
    "field_key",
    "row_id",
    "case_relevance",
    "content_type",
    "资料标题",
    "来源ID",
    "metadata",
)
UNCERTAINTY_MARKERS = (
    "证据未明确说明",
    "未明确说明",
    "证据不足",
    "无法据此判断",
    "无法判断",
    "无法回答",
    "未提供",
)


def should_batch_synthesize(answer: NeedFirstPolicyAnswer) -> bool:
    answerable = [
        item for item in answer.need_answers
        if item.status in {"supported", "partial"} and item.fact_refs
    ]
    if len(answerable) < 2 or len(answer.claims) < 2:
        return False
    has_textual_fact = any(
        fact.fact_type == "sentence_window" for fact in answer.extracted_facts
    )
    return has_textual_fact or answer.answer_mode in {
        "process_rule",
        "policy_explanation",
        "comparison",
    }


def build_batch_synthesis_messages(
    *,
    user_question: str,
    answer: NeedFirstPolicyAnswer,
) -> list[dict[str, Any]]:
    fact_by_id = {fact.fact_id: fact for fact in answer.extracted_facts}
    needs = []
    for item in answer.need_answers:
        if item.status not in {"supported", "partial"} or not item.fact_refs:
            continue
        facts = [fact_by_id[ref] for ref in item.fact_refs if ref in fact_by_id]
        needs.append(
            {
                "need_id": item.need_id,
                "need_text": item.need_text,
                "support_status": item.status,
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "fact_text": fact.user_facing_text,
                        "evidence_quote": fact.evidence_quote,
                    }
                    for fact in facts
                ],
            }
        )
    payload = {
        "user_question": user_question,
        "information_needs": needs,
        "output_contract": {
            "answers": [
                {
                    "need_id": "必须逐字复制输入 need_id",
                    "answer_text": "直接回答该信息点的一到两句中文",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "你是 Policy Expert 的批量回答整理节点。只允许使用输入 facts 和 "
                "evidence_quote 中已经出现的事实，对全部信息点各写一条直接、自然、简洁的回答。"
                "不要复制问答标题，不要输出内部字段，不要新增条件、材料、时间、比例、流程或建议；"
                "不得把相关证据改写成替代、因果或充分条件。partial 必须保留证据边界；"
                "supported 信息点必须直接回答，不得写成‘证据未明确说明’或‘无法判断’。"
                "不同信息点不得返回相同句子。不要添加引用编号或 Markdown。只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        },
    ]


def apply_batch_synthesis(
    answer: NeedFirstPolicyAnswer,
    payload: dict[str, Any],
) -> NeedFirstPolicyAnswer | None:
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list):
        return None
    expected = {
        item.need_id: item
        for item in answer.need_answers
        if item.status in {"supported", "partial"} and item.fact_refs
    }
    submitted: dict[str, str] = {}
    for item in raw_answers:
        if not isinstance(item, dict):
            return None
        need_id = str(item.get("need_id") or "").strip()
        text = str(item.get("answer_text") or "").strip()
        if need_id not in expected or need_id in submitted:
            return None
        submitted[need_id] = text
    if set(submitted) != set(expected):
        return None

    facts_by_id = {fact.fact_id: fact for fact in answer.extracted_facts}
    normalized_answers: dict[str, str] = {}
    seen_text: set[str] = set()
    for need_id, text in submitted.items():
        need_answer = expected[need_id]
        support_parts = [need_answer.need_text]
        for fact_ref in need_answer.fact_refs:
            fact = facts_by_id.get(fact_ref)
            if fact is not None:
                support_parts.extend((fact.user_facing_text, fact.evidence_quote))
        normalized = _validated_synthesis_text(text, "\n".join(support_parts))
        if normalized is None:
            return None
        if (
            need_answer.status == "supported"
            and any(marker in normalized for marker in UNCERTAINTY_MARKERS)
        ):
            return None
        signature = _compact(normalized)
        if signature in seen_text:
            return None
        seen_text.add(signature)
        normalized_answers[need_id] = normalized

    next_need_answers = [
        item.model_copy(
            update={"answer_text": normalized_answers.get(item.need_id, item.answer_text)}
        )
        for item in answer.need_answers
    ]
    next_claims = [
        claim.model_copy(
            update={"text": normalized_answers.get(claim.need_id, claim.text)}
        )
        for claim in answer.claims
    ]
    markdown = render_information_need_markdown(
        next_need_answers,
        answer.citations,
        answer_mode=answer.answer_mode,
    )
    return answer.model_copy(
        update={
            "expert_answer": markdown[:2000],
            "answer_markdown": markdown[:4000],
            "need_answers": next_need_answers,
            "claims": next_claims,
        }
    )


def _validated_synthesis_text(text: str, support_text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not 4 <= len(normalized) <= 600:
        return None
    if re.search(r"\[\d+\]", normalized):
        return None
    lowered = normalized.lower()
    if any(marker.lower() in lowered for marker in INTERNAL_MARKERS):
        return None
    support = _compact(support_text).lower()
    for token in re.findall(r"\d+(?:\.\d+)?%?|《[^》]{1,80}》|[A-Za-z]+\d+[A-Za-z0-9-]*", normalized):
        if _compact(token).lower() not in support:
            return None
    answer_bigrams = _bigrams(normalized)
    support_bigrams = _bigrams(support_text)
    if answer_bigrams and len(answer_bigrams & support_bigrams) / len(answer_bigrams) < 0.12:
        return None
    if not normalized.endswith(("。", "！", "？", ".", "!", "?")):
        normalized += "。"
    return normalized


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))
    return {compact[index:index + 2] for index in range(max(len(compact) - 1, 0))}


def _compact(text: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()《》“”\"'‘’\[\]【】/\\-]+", "", str(text or ""))


__all__ = [
    "apply_batch_synthesis",
    "build_batch_synthesis_messages",
    "should_batch_synthesize",
]
