"""Need-based sentence-window evidence judging for policy answers."""

from __future__ import annotations

from typing import Any

from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
    EvidenceMatch,
    ExtractedFact,
    NormalizedPolicyEvidence,
)
from src.backend.application.agent.expert_agent.service.policy_slot_window import (
    build_sentence_windows as build_need_windows,
    extract_verified_facts_from_judgement,
    quote_verified,
    reindex_facts_and_matches,
    select_candidate_windows,
    structured_evidence,
    textual_evidence,
    windows_for_prompt,
)


def build_sentence_windows(
    *,
    requirements: list[AnswerRequirement],
    evidence: list[NormalizedPolicyEvidence],
    filters: dict[str, Any],
    user_question: str,
    radius: int = 1,
) -> list[dict[str, Any]]:
    return build_need_windows(
        requirements=requirements,
        evidence=evidence,
        filters=filters,
        user_question=user_question,
        radius=radius,
    )
