"""Deterministic Case Agent answer assembly from controlled context."""

from __future__ import annotations

import json
from typing import Any

from src.backend.application.agent.case_agent.tools.registry import (
    manifest_item_for_capability,
    normalize_capability_name,
)
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentContentBlock,
    CaseAgentSource,
)


DIRECT_GRANULARITIES = {"single_field", "field_group", "list", "overview"}


def build_deterministic_answer(
    state: dict[str, Any],
    *,
    recovery: bool = False,
) -> CaseAgentAnswer | None:
    """Build a grounded answer when controlled tool output already satisfies the request."""

    policy = state.get("answer_policy") or {}
    if policy.get("display_mode") != "grounded":
        return None
    if state.get("answer_strategy") == "reuse_previous_answer":
        return None

    capabilities = _planned_capabilities(state)
    if not capabilities:
        return None
    manifests = [manifest_item_for_capability(name) for name in capabilities]
    if not recovery and any(
        manifest is None
        or manifest.layer != "L1"
        or manifest.status != "available"
        or not manifest.can_answer_directly
        for manifest in manifests
    ):
        return None

    context = state.get("answer_context") or {}
    semantics = state.get("query_semantics") or {}
    granularity = str(semantics.get("granularity") or "overview")
    if not recovery and granularity not in DIRECT_GRANULARITIES:
        return None
    if context.get("notices"):
        return None

    available_sources = [
        source
        for source in state.get("available_sources", [])
        if isinstance(source, CaseAgentSource)
    ]
    available_refs = {source.source_ref for source in available_sources}
    context_refs = [
        ref
        for ref in context.get("source_refs", [])
        if isinstance(ref, str) and ref in available_refs
    ]
    if not context_refs:
        return None

    max_blocks = max(
        1,
        min(8, int((state.get("answer_style_policy") or {}).get("max_blocks") or 4)),
    )
    blocks = _render_blocks(
        context,
        manifests=manifests,
        granularity=granularity,
        default_refs=context_refs[:8],
        max_blocks=max_blocks,
    )
    blocks = [
        CaseAgentContentBlock(
            text=block["text"][:1200],
            source_refs=[ref for ref in block["source_refs"] if ref in available_refs][:8],
        )
        for block in blocks
        if block.get("text") and block.get("source_refs")
    ]
    if not blocks:
        return None

    used_refs = {
        ref
        for block in blocks
        for ref in block.source_refs
    }
    sources = [source for source in available_sources if source.source_ref in used_refs]
    if not sources:
        return None
    return CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=blocks,
        sources=sources,
        fallback_notice="",
        metadata={
            "intent": state.get("intent") or semantics.get("intent") or "case_task",
            "requires_citation": True,
            "capabilities_used": capabilities,
            "assembled_by": "deterministic_answer_assembler",
            "recovery": recovery,
        },
    )


def _planned_capabilities(state: dict[str, Any]) -> list[str]:
    capabilities = [
        normalize_capability_name(str(item.get("capability") or item.get("name") or ""))
        for item in state.get("execution_plan", [])
        if isinstance(item, dict)
    ]
    return list(dict.fromkeys(item for item in capabilities if item))


def _render_blocks(
    context: dict[str, Any],
    *,
    manifests: list[Any],
    granularity: str,
    default_refs: list[str],
    max_blocks: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if granularity == "overview" and len(manifests) == 1:
        summary = str(getattr(manifests[0], "business_summary", "") or "").strip()
        if summary:
            blocks.append({"text": summary, "source_refs": default_refs})

    facts = [item for item in context.get("facts", []) if isinstance(item, dict)]
    rendered_facts = [
        f"{str(item.get('label') or '信息')}：{_render_value(item.get('value'))}"
        for item in facts[:12]
        if item.get("value") not in (None, "", [], {})
    ]
    if rendered_facts and len(blocks) < max_blocks:
        fact_refs = _refs_from_items(facts, default_refs)
        prefix = "当前案件：" if granularity == "overview" else ""
        blocks.append(
            {
                "text": (prefix + "；".join(rendered_facts) + "。").strip(),
                "source_refs": fact_refs,
            }
        )

    for item in context.get("lists", []):
        if len(blocks) >= max_blocks or not isinstance(item, dict):
            break
        title = str(item.get("title") or item.get("section") or "清单")
        rows = [row for row in item.get("items", []) if isinstance(row, dict)]
        rendered_rows = [
            f"{index}. {_render_mapping(row)}"
            for index, row in enumerate(rows[:20], start=1)
            if _render_mapping(row)
        ]
        if not rendered_rows:
            continue
        count = item.get("count")
        count_text = f"（共 {count} 项）" if count not in (None, "") else ""
        text = f"{title}{count_text}：\n" + "\n".join(rendered_rows)
        blocks.append(
            {
                "text": text[:1200],
                "source_refs": _refs_from_items([item], default_refs),
            }
        )
    return blocks[:max_blocks]


def _refs_from_items(items: list[dict[str, Any]], default_refs: list[str]) -> list[str]:
    refs: list[str] = []
    for item in items:
        refs.extend(
            str(ref)
            for ref in item.get("source_refs", [])
            if isinstance(ref, str) and ref
        )
    return list(dict.fromkeys(refs or default_refs))[:8]


def _render_mapping(value: dict[str, Any]) -> str:
    pairs = []
    for key, item in value.items():
        if item in (None, "", [], {}) or str(key).endswith("_ref"):
            continue
        pairs.append(f"{key}：{_render_value(item)}")
    return "，".join(pairs)[:500]


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        return "、".join(_render_value(item) for item in value[:12])
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)[:500]
    return str(value)
