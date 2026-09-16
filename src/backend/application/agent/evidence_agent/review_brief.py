"""Deterministic prompt brief for Review Advisor evaluation variants.

The builder keeps the full Evidence Ledger in Agent state for validation and
tool drilldown. It only changes the model-facing prompt representation used by
the eval-only C1 variant.
"""

from __future__ import annotations

from typing import Any


REVIEW_BRIEF_VERSION = "review-brief-v1"

MAX_CANDIDATE_CLUES = 10
MAX_IMPORTANT_FACTS = 24
MAX_KNOWN_GAPS = 8
MAX_MANIFEST_REFS = 48


def build_review_brief(evidence_ledger: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, task-sufficient view from an Evidence Ledger."""

    if not isinstance(evidence_ledger, dict):
        return {
            "scope": "task_sufficient_review_brief_v1",
            "builder_version": REVIEW_BRIEF_VERSION,
            "case_id": None,
            "candidate_clues": [],
            "citation_manifest": {},
            "generation_guidance": {
                "default_context": "review_brief",
                "tool_call_policy": "Use available tools only when the brief leaves a material gap.",
            },
        }

    reference_by_ref = _reference_by_ref(evidence_ledger)
    facts = _dict_items(evidence_ledger.get("normalized_facts"))
    facts_by_ref = _facts_by_ref(facts)
    system_risk_prompt = evidence_ledger.get("system_risk_prompt") or {}
    system_refs = _clean_refs(system_risk_prompt.get("source_refs"))
    known_gaps = _compact_known_gaps(evidence_ledger.get("known_gaps"))
    clue_candidates = _candidate_clues(
        evidence_ledger=evidence_ledger,
        facts_by_ref=facts_by_ref,
        known_gaps=known_gaps,
        system_refs=system_refs,
    )
    selected_refs = _selected_refs(
        system_refs=system_refs,
        clue_candidates=clue_candidates,
        known_gaps=known_gaps,
        facts=facts,
    )
    important_facts = _important_facts(facts, selected_refs)
    selected_refs = _limit_refs(
        _unique_refs(
            selected_refs,
            _refs_from_items(important_facts),
        ),
        MAX_MANIFEST_REFS,
    )

    guidance = dict(evidence_ledger.get("generation_guidance") or {})
    guidance.update(
        {
            "default_context": "review_brief",
            "brief_builder_version": REVIEW_BRIEF_VERSION,
            "tool_call_policy": (
                "The review brief is the first source of truth for final JSON. "
                "Call drilldown tools only when a specific rule, safe field, or "
                "statistical material detail can change the evidence relation, "
                "clue status, conflict, or human verification item."
            ),
        }
    )

    return {
        "case_id": evidence_ledger.get("case_id"),
        "ledger_version": evidence_ledger.get("ledger_version"),
        "scope": "task_sufficient_review_brief_v1",
        "builder_version": REVIEW_BRIEF_VERSION,
        "representation_goal": (
            "Backend-prepared review representation for one final advisory JSON. "
            "Full Evidence Ledger remains available to backend validation."
        ),
        "agent_task": evidence_ledger.get("agent_task"),
        "case_context": evidence_ledger.get("case_context"),
        "system_risk_prompt": _compact_system_risk_prompt(system_risk_prompt),
        "candidate_clues": clue_candidates,
        "important_facts": important_facts,
        "known_gaps": known_gaps,
        "citation_manifest": _citation_manifest(reference_by_ref, facts_by_ref, known_gaps, selected_refs),
        "drilldown_index": _compact_drilldown_index(evidence_ledger.get("drilldown_index")),
        "generation_guidance": guidance,
        "prefetch_summary": {
            **dict(evidence_ledger.get("prefetch_summary") or {}),
            "brief_builder_version": REVIEW_BRIEF_VERSION,
            "brief_candidate_count": len(clue_candidates),
            "brief_fact_count": len(important_facts),
            "brief_citation_manifest_count": len(selected_refs),
        },
    }


def _candidate_clues(
    *,
    evidence_ledger: dict[str, Any],
    facts_by_ref: dict[str, list[dict[str, Any]]],
    known_gaps: list[dict[str, Any]],
    system_refs: list[str],
) -> list[dict[str, Any]]:
    clues: list[dict[str, Any]] = []
    counter_refs = _counter_rule_refs(evidence_ledger.get("drilldown_index"))
    for item in _dict_items(evidence_ledger.get("clue_candidates"))[:MAX_CANDIDATE_CLUES]:
        support_refs = _clean_refs(item.get("source_refs"))
        related_refs = _related_fact_refs(support_refs, facts_by_ref)
        gap_refs = _refs_from_items(known_gaps[:3])
        eligible_refs = _limit_refs(
            _unique_refs(support_refs, related_refs, system_refs, gap_refs),
            12,
        )
        clues.append(
            {
                "clue_id": item.get("clue_id"),
                "title": item.get("title") or item.get("statement") or item.get("clue_id"),
                "origin": item.get("origin"),
                "current_status": item.get("status"),
                "basis_summary": item.get("basis_summary") or item.get("statement"),
                "support_refs": support_refs,
                "context_refs": _limit_refs(_unique_refs(related_refs, system_refs), 8),
                "counter_refs": _limit_refs(counter_refs, 3),
                "known_gaps": known_gaps[:3],
                "suggested_checks": _suggested_checks(item),
                "eligible_refs": eligible_refs,
            }
        )
    return clues


def _important_facts(facts: list[dict[str, Any]], selected_refs: list[str]) -> list[dict[str, Any]]:
    selected_set = set(selected_refs)
    prioritized: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    category_rank = {
        "model_warning": 0,
        "rule_result": 1,
        "risk_component": 2,
        "material_summary": 3,
    }
    ordered = sorted(
        facts,
        key=lambda item: category_rank.get(str(item.get("category")), 9),
    )
    for fact in ordered:
        compact = _compact_fact(fact)
        if not compact:
            continue
        refs = set(compact.get("source_refs") or [])
        if refs & selected_set:
            prioritized.append(compact)
        else:
            fallback.append(compact)
    return [*prioritized, *fallback][:MAX_IMPORTANT_FACTS]


def _compact_fact(item: dict[str, Any]) -> dict[str, Any]:
    refs = _clean_refs(item.get("source_refs"))
    if not refs:
        return {}
    payload: dict[str, Any] = {
        "fact_id": item.get("fact_id"),
        "category": item.get("category"),
        "statement": item.get("statement"),
        "source_refs": refs,
    }
    value_summary = item.get("value_summary")
    if isinstance(value_summary, dict):
        payload["value_summary"] = {
            key: value
            for key, value in value_summary.items()
            if value is not None and key in {"reason", "severity", "current_value", "threshold", "status"}
        }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _compact_known_gaps(value: Any) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for item in _dict_items(value)[:MAX_KNOWN_GAPS]:
        refs = _clean_refs(item.get("source_refs"))
        if not refs:
            continue
        gaps.append(
            {
                "gap_id": item.get("gap_id"),
                "description": item.get("description") or item.get("statement"),
                "source_refs": refs,
            }
        )
    return gaps


def _compact_system_risk_prompt(value: Any) -> dict[str, Any] | Any:
    if not isinstance(value, dict):
        return value
    return {
        key: value.get(key)
        for key in (
            "generated_by",
            "risk_level",
            "risk_level_label",
            "score_text",
            "source_summary",
            "source_refs",
        )
        if key in value
    }


def _compact_drilldown_index(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"rules": [], "materials": [], "fields": []}
    return {
        "rules": [
            _keep_keys(item, ("rule_id", "rule_name", "hit", "severity", "evidence_ref"))
            for item in _dict_items(value.get("rules"))[:24]
        ],
        "materials": [
            _keep_keys(item, ("document_id", "title", "document_type"))
            for item in _dict_items(value.get("materials"))[:10]
        ],
        "fields": [
            _keep_keys(item, ("field_name", "status", "source_ref"))
            for item in _dict_items(value.get("fields"))[:24]
        ],
    }


def _citation_manifest(
    reference_by_ref: dict[str, dict[str, Any]],
    facts_by_ref: dict[str, list[dict[str, Any]]],
    known_gaps: list[dict[str, Any]],
    selected_refs: list[str],
) -> dict[str, dict[str, Any]]:
    gap_by_ref = {
        ref: gap
        for gap in known_gaps
        for ref in _clean_refs(gap.get("source_refs"))
    }
    manifest: dict[str, dict[str, Any]] = {}
    for ref in selected_refs:
        reference = reference_by_ref.get(ref, {})
        first_fact = (facts_by_ref.get(ref) or [{}])[0]
        gap = gap_by_ref.get(ref, {})
        manifest[ref] = {
            "source_type": reference.get("source_type"),
            "label": reference.get("label") or first_fact.get("statement") or gap.get("description") or ref,
            "current_value": reference.get("current_value"),
            "threshold": reference.get("threshold"),
            "short_fact": first_fact.get("statement") or gap.get("description"),
        }
        manifest[ref] = {
            key: value
            for key, value in manifest[ref].items()
            if value not in (None, "", [], {})
        }
    return manifest


def _reference_by_ref(evidence_ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for item in _dict_items(evidence_ledger.get("evidence_reference_index")):
        ref = item.get("source_ref")
        if isinstance(ref, str) and ref:
            references[ref] = item
    return references


def _facts_by_ref(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        for ref in _clean_refs(fact.get("source_refs")):
            grouped.setdefault(ref, []).append(_compact_fact(fact))
    return grouped


def _selected_refs(
    *,
    system_refs: list[str],
    clue_candidates: list[dict[str, Any]],
    known_gaps: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> list[str]:
    clue_refs = _unique_refs(
        _refs_from_items(clue_candidates, key="eligible_refs"),
        _refs_from_items(clue_candidates, key="support_refs"),
        _refs_from_items(clue_candidates, key="context_refs"),
        _refs_from_items(clue_candidates, key="counter_refs"),
    )
    fact_refs = _refs_from_items(_important_facts(facts, _unique_refs(system_refs, clue_refs)))
    gap_refs = _refs_from_items(known_gaps)
    return _limit_refs(
        _unique_refs(system_refs, clue_refs, fact_refs, gap_refs),
        MAX_MANIFEST_REFS,
    )


def _related_fact_refs(
    support_refs: list[str],
    facts_by_ref: dict[str, list[dict[str, Any]]],
) -> list[str]:
    categories = {
        str(fact.get("category"))
        for ref in support_refs
        for fact in facts_by_ref.get(ref, [])
        if fact.get("category")
    }
    if not categories:
        return []
    refs: list[str] = []
    for ref, facts in facts_by_ref.items():
        if ref in support_refs:
            continue
        if any(str(fact.get("category")) in categories for fact in facts):
            refs.append(ref)
    return _limit_refs(refs, 5)


def _counter_rule_refs(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    refs: list[str] = []
    for item in _dict_items(value.get("rules")):
        if item.get("hit") is True:
            continue
        if item.get("severity") not in {"high", "critical", "medium"}:
            continue
        ref = item.get("evidence_ref")
        if isinstance(ref, str) and ref:
            refs.append(ref)
    return _limit_refs(_unique_refs(refs), 4)


def _suggested_checks(item: dict[str, Any]) -> list[str]:
    origin = str(item.get("origin") or "")
    clue_id = str(item.get("clue_id") or "")
    if "rule" in clue_id:
        return [
            "Check whether the rule hit fact, current value, and threshold are enough to support the clue.",
            "If the brief lacks rule detail, use get_rule_details for the exact rule.",
        ]
    if "model" in clue_id:
        return [
            "Check whether model warning evidence aligns with rule hits and risk components.",
            "Do not restate the backend risk score as an Agent conclusion.",
        ]
    if origin == "risk_score":
        return [
            "Check whether this risk contribution has cited business evidence.",
            "Use safe field or material drilldown only if it changes the review relation.",
        ]
    return ["Write only evidence-backed clue review and human verification items."]


def _refs_from_items(items: list[dict[str, Any]], *, key: str = "source_refs") -> list[str]:
    return _unique_refs(*(_clean_refs(item.get(key)) for item in items))


def _clean_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _unique_refs(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    refs: list[str] = []
    for group in groups:
        for ref in group:
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
    return refs


def _limit_refs(refs: list[str], limit: int) -> list[str]:
    return refs[:limit]


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _keep_keys(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: item[key] for key in keys if key in item and item[key] is not None}
