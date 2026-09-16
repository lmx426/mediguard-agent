"""Question rewrite helpers for need-based policy retrieval."""

from __future__ import annotations

import re
from typing import Any, Literal

RewriteMode = Literal["split", "narrow", "normalize"]


def rewrite_question_and_information_needs(
    *,
    original_question: str,
    answer_mode: str,
    information_needs: list[str],
    missing_needs: list[str] | None = None,
    failed_evidence_summary: list[str] | None = None,
) -> dict[str, Any]:
    compact_question = str(original_question or "").strip()
    normalized_needs = _normalize_needs(missing_needs or information_needs or [compact_question])
    if answer_mode == "fact_lookup":
        rewrite_mode: RewriteMode = "normalize"
        rewritten_needs = normalized_needs[:1]
    elif answer_mode == "list":
        rewrite_mode = "split"
        rewritten_needs = normalized_needs
    elif answer_mode == "comparison":
        rewrite_mode = "split"
        rewritten_needs = normalized_needs
    elif answer_mode == "policy_explanation":
        rewrite_mode = "narrow"
        rewritten_needs = normalized_needs[:3]
    else:
        rewrite_mode = "split" if len(normalized_needs) > 1 else "narrow"
        rewritten_needs = normalized_needs[:4]

    rewritten_question = compact_question
    if failed_evidence_summary:
        rewritten_question = compact_question
    return {
        "rewritten_question": rewritten_question,
        "rewritten_information_needs": rewritten_needs,
        "rewrite_mode": rewrite_mode,
    }


def _normalize_needs(values: list[str]) -> list[str]:
    needs: list[str] = []
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "")).strip(" ，,；;。！？?!")
        if text and text not in needs:
            needs.append(text)
    return needs
