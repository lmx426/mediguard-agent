"""Build a recall-safe retrieval branch without changing LLM strict filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.backend.domain.policy_rag_taxonomy import CONTENT_TYPES


@dataclass(frozen=True, slots=True)
class RecallFilterProjection:
    filters: dict[str, Any]
    reasons: tuple[str, ...] = field(default_factory=tuple)
    explicit_jurisdictions: tuple[str, ...] = field(default_factory=tuple)


class RecallFilterProjector:
    """Project only trustworthy hard boundaries into a second search branch.

    This class never repairs or overwrites the strict LLM output. It removes
    uncertain constraints for a separate recall branch and derives explicit
    jurisdiction only from literal user wording.
    """

    def project(
        self,
        *,
        user_question: str,
        strict_filters: dict[str, Any],
    ) -> RecallFilterProjection:
        explicit = _explicit_jurisdictions(user_question)
        jurisdictions = _recall_jurisdictions(explicit)
        cite_value = strict_filters.get("can_cite_as_policy_basis")
        if not isinstance(cite_value, bool):
            cite_value = None

        reasons = [
            "policy_domain_softened_for_recall",
            "content_type_broadened_to_registered_chunk_types",
        ]
        if explicit:
            reasons.append("explicit_jurisdiction_retained")
        else:
            reasons.append("inferred_jurisdiction_removed")

        return RecallFilterProjection(
            filters={
                "jurisdiction": jurisdictions,
                "policy_domain": [],
                "content_type": list(CONTENT_TYPES),
                "can_cite_as_policy_basis": cite_value,
            },
            reasons=tuple(reasons),
            explicit_jurisdictions=tuple(explicit),
        )


def _explicit_jurisdictions(question: str) -> list[str]:
    text = str(question or "")
    output: list[str] = []
    if "北京" in text:
        output.append("beijing")
    if "上海" in text:
        output.append("shanghai")
    if any(token in text for token in ("全国", "国家级", "国家政策", "国家规定")):
        output.append("national")
    return output


def _recall_jurisdictions(explicit: list[str]) -> list[str]:
    if not explicit:
        return []
    output = list(explicit)
    if any(value in {"beijing", "shanghai"} for value in explicit):
        if "national" not in output:
            output.append("national")
    return output
