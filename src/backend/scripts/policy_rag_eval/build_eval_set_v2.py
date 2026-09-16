"""Build a draft Policy RAG v2 goldset from mapped candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_NODES_PATH,
    DEFAULT_V2_DRAFT_PATH,
    DEFAULT_V2_MAPPED_PATH,
    SCHEMA_EVAL_V2,
    derive_filters_from_refs,
    is_generic_answer_point,
    load_policy_nodes,
    read_jsonl,
    stable_id,
    versioned_eval_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build draft policy_rag_eval_v2 goldset records."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--version-suffix", default=None)
    parser.add_argument("--mapped-jsonl", type=Path, default=None)
    parser.add_argument("--draft-jsonl", type=Path, default=None)
    args = parser.parse_args()
    if args.mapped_jsonl is None:
        args.mapped_jsonl = (
            versioned_eval_path("mapped", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_MAPPED_PATH
        )
    if args.draft_jsonl is None:
        args.draft_jsonl = (
            versioned_eval_path("draft", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_DRAFT_PATH
        )
    return args


def main() -> None:
    args = parse_args()
    nodes = load_policy_nodes(args.nodes_jsonl)
    mapped_candidates = read_jsonl(args.mapped_jsonl)
    rows = [
        build_eval_row(candidate, nodes, index)
        for index, candidate in enumerate(mapped_candidates, start=1)
    ]
    write_jsonl(args.draft_jsonl, rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "draft_count": len(rows),
                "draft_path": str(args.draft_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_eval_row(
    candidate: dict[str, Any],
    nodes: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    mapped_contexts = list(candidate.get("mapped_contexts") or [])
    answer_points = normalize_answer_points(
        candidate.get("answer_points"),
        candidate.get("reference_answer"),
        mapped_contexts,
    )
    groups = [
        build_group_for_answer_point(point, mapped_contexts, group_index)
        for group_index, point in enumerate(answer_points, start=1)
    ]
    gold_refs = dedupe_refs(
        ref for group in groups for ref in group.get("evidence_refs") or []
    )
    proposed_filters = candidate.get("proposed_filters")
    filters = (
        normalize_filters(proposed_filters)
        or dict(candidate.get("derived_filters") or {})
        or derive_filters_from_refs(gold_refs, nodes)
    )
    query_id = f"v2_{index:06d}"
    missing_required_groups = [
        group["group_id"] for group in groups if group["required"] and not group["gold_node_ids"]
    ]
    evidence_status = "full" if not missing_required_groups and gold_refs else "none"
    return {
        "schema_version": SCHEMA_EVAL_V2,
        "query_id": query_id,
        "candidate_id": candidate.get("candidate_id"),
        "case_id": "policy_rag_v2",
        "bundle_type": candidate.get("bundle_type") or "unknown",
        "source_context_count": candidate.get("source_context_count"),
        "question": str(candidate.get("question") or "").strip(),
        "question_type": str(candidate.get("question_type") or "single_hop"),
        "filters": filters,
        "reference_answer": str(candidate.get("reference_answer") or "").strip(),
        "answer_points": answer_points,
        "gold_evidence_groups": groups,
        "expected_source_groups": expected_source_groups_from_groups(groups),
        "gold_evidence_refs": gold_refs,
        "canonical_source_ids": sorted(
            {str(ref.get("source_id") or "") for ref in gold_refs if ref.get("source_id")}
        ),
        "evidence_status": evidence_status,
        "expected_status": "supported" if evidence_status == "full" else "coverage_gap",
        "metric_eligibility": {
            "strict_retrieval": evidence_status == "full",
            "partial_retrieval": False,
            "gap_detection": evidence_status == "none",
        },
        "generated_by": candidate.get("generated_by") or "ragas_deepseek",
        "generation_model": candidate.get("generation_model"),
        "audit_status": "pending",
        "standard_evidence_audit": {
            "verified": False,
            "query_evidence_status": evidence_status,
            "matched_group_count": 0,
            "group_count": len(groups),
        },
        "mapping_summary": candidate.get("mapping_summary") or {},
        "source_bundle_id": candidate.get("source_bundle_id"),
        "build_id": stable_id(
            "evalv2",
            candidate.get("candidate_id"),
            candidate.get("question"),
        ),
    }


def normalize_answer_points(
    raw_points: object,
    reference_answer: object,
    mapped_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context_ids = [
        str(context.get("context_id") or "")
        for context in mapped_contexts
        if str(context.get("context_id") or "")
    ]
    points: list[dict[str, Any]] = []
    if isinstance(raw_points, list):
        for raw in raw_points:
            if isinstance(raw, dict):
                statement = str(raw.get("statement") or raw.get("text") or "").strip()
                raw_context_ids = [
                    str(item)
                    for item in raw.get("context_ids", [])
                    if str(item) in context_ids
                ]
                required = bool(raw.get("required", True))
            else:
                statement = str(raw).strip()
                raw_context_ids = []
                required = True
            if statement and not is_generic_answer_point(statement):
                points.append(
                    {
                        "point_id": f"p{len(points) + 1}",
                        "statement": statement,
                        "required": required,
                        "context_ids": raw_context_ids or context_ids,
                    }
                )
    if points:
        return points[:4]

    answer = str(reference_answer or "").strip()
    statements = [
        item.strip()
        for item in re.split(r"[。；;]\s*", answer)
        if item.strip()
    ]
    if not statements and answer:
        statements = [answer]
    fallback_points: list[dict[str, Any]] = []
    for statement in statements[:4]:
        if is_generic_answer_point(statement):
            continue
        fallback_points.append(
            {
                "point_id": f"p{len(fallback_points) + 1}",
                "statement": statement,
                "required": True,
                "context_ids": context_ids,
            }
        )
    return fallback_points


def build_group_for_answer_point(
    point: dict[str, Any],
    mapped_contexts: list[dict[str, Any]],
    group_index: int,
) -> dict[str, Any]:
    point_context_ids = set(point.get("context_ids") or [])
    selected_contexts = [
        context
        for context in mapped_contexts
        if str(context.get("context_id") or "") in point_context_ids
    ]
    if not selected_contexts:
        selected_contexts = mapped_contexts
    refs = [
        evidence_ref_from_context(context, point_id=str(point["point_id"]))
        for context in selected_contexts
        if context.get("node_id")
    ]
    refs = dedupe_refs(refs)
    group_id = f"g{group_index}"
    return {
        "group_id": group_id,
        "group_name": group_name_from_statement(str(point.get("statement") or ""), group_index),
        "answer_point_ids": [str(point["point_id"])],
        "match_mode": "any",
        "required": bool(point.get("required", True)),
        "gold_node_ids": sorted(
            {str(ref.get("node_id") or "") for ref in refs if ref.get("node_id")}
        ),
        "evidence_refs": refs,
    }


def evidence_ref_from_context(context: dict[str, Any], *, point_id: str) -> dict[str, Any]:
    return {
        "answer_point_id": point_id,
        "node_id": context.get("node_id"),
        "source_id": context.get("source_id"),
        "doc_id": context.get("doc_id"),
        "source_url": context.get("source_url"),
        "evidence_text": context.get("evidence_text"),
        "anchor_terms": context.get("anchor_terms") or [],
        "support_role": "primary",
        "node_mapping_method": context.get("node_mapping_method"),
        "node_mapping_confidence": context.get("node_mapping_confidence"),
    }


def expected_source_groups_from_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for group in groups:
        refs = group.get("evidence_refs") or []
        outputs.append(
            {
                "group": group["group_id"],
                "acceptable_sources": sorted(
                    {str(ref.get("source_id") or "") for ref in refs if ref.get("source_id")}
                ),
                "content_anchor_sets": [
                    [str(term) for term in ref.get("anchor_terms") or []]
                    for ref in refs
                    if ref.get("anchor_terms")
                ],
                "document_roles": sorted(
                    {
                        str(ref.get("support_role") or "primary")
                        for ref in refs
                    }
                ),
            }
        )
    return outputs


def dedupe_refs(refs: Any) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ref in refs:
        key = (
            str(ref.get("answer_point_id") or ""),
            str(ref.get("node_id") or ""),
            str(ref.get("evidence_text") or "")[:80],
        )
        unique[key] = dict(ref)
    return list(unique.values())


def normalize_filters(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    outputs: dict[str, list[str]] = {}
    for key in ("jurisdiction", "policy_domain", "content_type"):
        raw = value.get(key)
        if isinstance(raw, list):
            cleaned = sorted({str(item) for item in raw if str(item)})
            if cleaned:
                outputs[key] = cleaned
    return outputs


def group_name_from_statement(statement: str, group_index: int) -> str:
    text = re.sub(r"\s+", "", statement)
    return text[:18] or f"证据组{group_index}"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v2-build] failed: {exc}", file=sys.stderr)
        raise
