"""Audit and accept Policy RAG v2 goldset records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_NODES_PATH,
    DEFAULT_V2_AUDIT_REPORT_PATH,
    DEFAULT_V2_AUDITED_PATH,
    DEFAULT_V2_DRAFT_PATH,
    SCHEMA_EVAL_V2,
    extract_table_search_entities,
    is_generic_answer_point,
    is_table_like_node,
    is_table_query_missing_concrete_entity,
    is_title_style_question,
    is_citeable_node,
    load_policy_nodes,
    read_jsonl,
    statement_supported_by_evidence,
    table_entity_in_question,
    text_contains,
    versioned_eval_path,
    write_jsonl,
)


TARGET_200_BUNDLE_RATIOS = [
    ("single_clause", 0.30),
    ("table_lookup", 0.25),
    ("same_doc_multi_clause", 0.15),
    ("cross_doc_policy_combo", 0.225),
    ("case_like_policy_query", 0.075),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit draft policy_rag_eval_v2 records and write accepted goldset."
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--version-suffix", default=None)
    parser.add_argument("--draft-jsonl", type=Path, default=None)
    parser.add_argument("--audited-jsonl", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--min-mapping-confidence", type=float, default=0.75)
    parser.add_argument("--target-accepted", type=int, default=None)
    args = parser.parse_args()
    if args.draft_jsonl is None:
        args.draft_jsonl = (
            versioned_eval_path("draft", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_DRAFT_PATH
        )
    if args.audited_jsonl is None:
        args.audited_jsonl = (
            versioned_eval_path("audited", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_AUDITED_PATH
        )
    if args.report_md is None:
        args.report_md = (
            versioned_eval_path("audit_report", args.version_suffix)
            if args.version_suffix
            else DEFAULT_V2_AUDIT_REPORT_PATH
        )
    return args


def main() -> None:
    args = parse_args()
    nodes = load_policy_nodes(args.nodes_jsonl)
    rows = read_jsonl(args.draft_jsonl)
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_query_ids: set[str] = set()

    for row in rows:
        result = audit_row(
            row,
            nodes,
            seen_questions=seen_questions,
            seen_query_ids=seen_query_ids,
            min_mapping_confidence=args.min_mapping_confidence,
        )
        if result["accepted"]:
            accepted_row = dict(row)
            accepted_row["audit_status"] = "accepted"
            accepted_row["evidence_status"] = "full"
            accepted_row["expected_status"] = "supported"
            accepted_row["metric_eligibility"] = {
                "strict_retrieval": True,
                "partial_retrieval": False,
                "gap_detection": False,
            }
            accepted_row["standard_evidence_audit"] = {
                "verified": True,
                "query_evidence_status": "full",
                "matched_group_count": len(
                    accepted_row.get("gold_evidence_groups") or []
                ),
                "group_count": len(accepted_row.get("gold_evidence_groups") or []),
                "checks": result["checks"],
            }
            eligible.append(accepted_row)
        else:
            rejected.append(
                {
                    "query_id": row.get("query_id"),
                    "candidate_id": row.get("candidate_id"),
                    "question": row.get("question"),
                    "failures": result["failures"],
                    "warnings": result["warnings"],
                }
            )

    audited, held_out = select_accepted_rows(
        eligible,
        target_accepted=args.target_accepted,
    )
    write_jsonl(args.audited_jsonl, audited)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        build_report(
            source_rows=rows,
            eligible_rows=eligible,
            accepted_rows=audited,
            rejected_rows=rejected,
            held_out_rows=held_out,
            report_path=args.report_md,
            audited_path=args.audited_jsonl,
            target_accepted=args.target_accepted,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "draft_count": len(rows),
                "accepted_count": len(audited),
                "eligible_count": len(eligible),
                "rejected_count": len(rejected),
                "held_out_count": len(held_out),
                "audited_path": str(args.audited_jsonl),
                "report_path": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def audit_row(
    row: dict[str, Any],
    nodes: dict[str, Any],
    *,
    seen_questions: set[str],
    seen_query_ids: set[str],
    min_mapping_confidence: float,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    query_id = str(row.get("query_id") or "").strip()
    question = str(row.get("question") or "").strip()
    if row.get("schema_version") != SCHEMA_EVAL_V2:
        failures.append("schema_version is not policy_rag_eval_v2")
    if not query_id:
        failures.append("missing query_id")
    elif query_id in seen_query_ids:
        failures.append("duplicate query_id")
    else:
        seen_query_ids.add(query_id)
    if len(question) < 8:
        failures.append("question is too short or missing")
    elif is_title_style_question(question):
        failures.append("question exposes policy title or starts from document title")
    elif question in seen_questions:
        failures.append("duplicate question")
    else:
        seen_questions.add(question)
    if not str(row.get("reference_answer") or "").strip():
        failures.append("missing reference_answer")

    answer_points: dict[str, str] = {}
    for point in row.get("answer_points") or []:
        if not isinstance(point, dict):
            continue
        point_id = str(point.get("point_id") or "")
        statement = str(point.get("statement") or "").strip()
        if not point_id:
            continue
        answer_points[point_id] = statement
        if is_generic_answer_point(statement):
            failures.append(f"answer point {point_id} is generic or unsupported")
    if not answer_points:
        failures.append("missing answer_points")

    groups = list(row.get("gold_evidence_groups") or [])
    if not groups:
        failures.append("missing gold_evidence_groups")
    covered_points: set[str] = set()
    verified_group_count = 0
    for group in groups:
        group_id = str(group.get("group_id") or "")
        refs = list(group.get("evidence_refs") or [])
        node_ids = [str(item) for item in group.get("gold_node_ids") or [] if str(item)]
        point_ids = [
            str(item) for item in group.get("answer_point_ids") or [] if str(item)
        ]
        if not group_id:
            failures.append("group missing group_id")
        if not point_ids:
            failures.append(f"{group_id}: missing answer_point_ids")
        for point_id in point_ids:
            if point_id not in answer_points:
                failures.append(f"{group_id}: unknown answer_point_id {point_id}")
            else:
                covered_points.add(point_id)
        if group.get("required", True) and not node_ids:
            failures.append(f"{group_id}: required group has no gold_node_ids")
        if not refs:
            failures.append(f"{group_id}: missing evidence_refs")
        verified_refs = 0
        for ref in refs:
            ref_failures = audit_ref(
                ref,
                nodes,
                min_mapping_confidence=min_mapping_confidence,
            )
            if ref_failures:
                failures.extend(f"{group_id}: {item}" for item in ref_failures)
            else:
                verified_refs += 1
        evidence_texts = [str(ref.get("evidence_text") or "") for ref in refs]
        for point_id in point_ids:
            statement = answer_points.get(point_id, "")
            if statement and not statement_supported_by_evidence(statement, evidence_texts):
                failures.append(f"{group_id}: answer point {point_id} is not supported by evidence_text")
        if verified_refs:
            verified_group_count += 1

    missing_points = set(answer_points) - covered_points
    for point_id in sorted(missing_points):
        failures.append(f"answer point {point_id} is not covered by any evidence group")

    filter_failures = audit_filters(row, nodes)
    failures.extend(filter_failures)
    table_entity_failures, table_entity_checks = audit_table_query_entities(row, nodes)
    failures.extend(table_entity_failures)
    checks["verified_group_count"] = verified_group_count
    checks["answer_point_count"] = len(answer_points)
    checks["filter_check"] = "passed" if not filter_failures else "failed"
    checks["source_url_required"] = True
    checks["table_query_entity_check"] = table_entity_checks
    return {
        "accepted": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def audit_ref(
    ref: dict[str, Any],
    nodes: dict[str, Any],
    *,
    min_mapping_confidence: float,
) -> list[str]:
    failures: list[str] = []
    node_id = str(ref.get("node_id") or "").strip()
    node = nodes.get(node_id)
    if node is None:
        return [f"node_id not found: {node_id}"]
    if not is_citeable_node(node):
        failures.append(f"node is not citeable: {node_id}")
    source_url = str(ref.get("source_url") or "").strip()
    if not source_url:
        failures.append(f"missing source_url for {node_id}")
    elif source_url != str(node.metadata.get("source_url") or ""):
        failures.append(f"source_url mismatch for {node_id}")
    evidence_text = str(ref.get("evidence_text") or "").strip()
    if not evidence_text:
        failures.append(f"missing evidence_text for {node_id}")
    elif not text_contains(node.text, evidence_text):
        failures.append(f"evidence_text not found in node.text for {node_id}")
    confidence = ref.get("node_mapping_confidence")
    if confidence is None:
        failures.append(f"missing node_mapping_confidence for {node_id}")
    elif float(confidence) < min_mapping_confidence:
        failures.append(
            f"low node_mapping_confidence for {node_id}: {float(confidence):.3f}"
        )
    method = str(ref.get("node_mapping_method") or "")
    if method not in {"metadata_direct", "exact_text", "semantic_match"}:
        failures.append(f"invalid node_mapping_method for {node_id}: {method}")
    return failures


def audit_table_query_entities(
    row: dict[str, Any],
    nodes: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    question = str(row.get("question") or "")
    question_type = str(row.get("question_type") or "")
    bundle_type = str(row.get("bundle_type") or "")
    refs = list(row.get("gold_evidence_refs") or [])
    all_refs_with_nodes: list[tuple[dict[str, Any], Any]] = []
    table_refs: list[tuple[dict[str, Any], Any]] = []
    for ref in refs:
        node = nodes.get(str(ref.get("node_id") or ""))
        if node is None:
            continue
        all_refs_with_nodes.append((ref, node))
        if is_table_like_node(node):
            table_refs.append((ref, node))

    requires_check = (
        question_type == "table_lookup"
        or bundle_type == "table_lookup"
        or bool(table_refs)
    )
    refs_to_check = (
        table_refs
        if table_refs
        else all_refs_with_nodes
        if question_type == "table_lookup" or bundle_type == "table_lookup"
        else []
    )
    checks: dict[str, Any] = {
        "required": requires_check,
        "table_ref_count": len(table_refs),
        "matched_ref_count": 0,
        "missing_ref_count": 0,
        "sample_expected_entities": [],
    }
    if not requires_check:
        return [], checks

    failures: list[str] = []
    if is_table_query_missing_concrete_entity(question):
        failures.append("table lookup question uses generic placeholder instead of searchable entity")

    if not refs_to_check:
        failures.append("table lookup question has no table-like gold evidence")
        return failures, checks

    for ref, node in refs_to_check:
        entities = extract_table_search_entities(
            question=question,
            evidence_text=str(ref.get("evidence_text") or ""),
            node=node,
            anchor_terms=ref.get("anchor_terms") or [],
        )
        matched = [entity for entity in entities if table_entity_in_question(entity, question)]
        if entities and len(checks["sample_expected_entities"]) < 8:
            checks["sample_expected_entities"].extend(
                entity
                for entity in entities[:3]
                if entity not in checks["sample_expected_entities"]
            )
        if matched:
            checks["matched_ref_count"] += 1
            continue
        checks["missing_ref_count"] += 1
        if entities:
            preview = " / ".join(entities[:5])
            failures.append(
                "table lookup question missing searchable entity from gold evidence: "
                f"{preview}"
            )
        else:
            failures.append("table lookup gold evidence has no extractable searchable entity")

    return failures, checks


def select_accepted_rows(
    rows: list[dict[str, Any]],
    *,
    target_accepted: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_accepted is None or target_accepted <= 0 or len(rows) <= target_accepted:
        return rows, []
    quotas = bundle_target_counts(target_accepted)
    by_bundle: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bundle.setdefault(str(row.get("bundle_type") or "unknown"), []).append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bundle_type, quota in quotas.items():
        for row in by_bundle.get(bundle_type, [])[:quota]:
            selected.append(row)
            selected_ids.add(str(row.get("query_id") or row.get("candidate_id") or id(row)))

    if len(selected) < target_accepted:
        for row in rows:
            row_id = str(row.get("query_id") or row.get("candidate_id") or id(row))
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)
            if len(selected) >= target_accepted:
                break
    selected = selected[:target_accepted]
    selected_ids = {
        str(row.get("query_id") or row.get("candidate_id") or id(row)) for row in selected
    }
    held_out = [
        row
        for row in rows
        if str(row.get("query_id") or row.get("candidate_id") or id(row)) not in selected_ids
    ]
    return selected, held_out


def bundle_target_counts(target_accepted: int) -> dict[str, int]:
    raw_counts: list[tuple[str, float, int]] = []
    assigned = 0
    for bundle_type, ratio in TARGET_200_BUNDLE_RATIOS:
        exact = target_accepted * ratio
        count = int(exact)
        raw_counts.append((bundle_type, exact - count, count))
        assigned += count
    remainder = max(0, target_accepted - assigned)
    raw_counts.sort(key=lambda item: item[1], reverse=True)
    counts = {bundle_type: count for bundle_type, _, count in raw_counts}
    for bundle_type, _, _ in raw_counts[:remainder]:
        counts[bundle_type] += 1
    return {
        bundle_type: counts.get(bundle_type, 0)
        for bundle_type, _ in TARGET_200_BUNDLE_RATIOS
    }


def audit_filters(row: dict[str, Any], nodes: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    filters = row.get("filters") or {}
    if not isinstance(filters, dict):
        return ["filters must be an object"]
    refs = list(row.get("gold_evidence_refs") or [])
    for key in ("jurisdiction", "policy_domain", "content_type"):
        expected_values = {str(item) for item in filters.get(key) or [] if str(item)}
        if not expected_values:
            continue
        ref_values = {
            str(nodes[str(ref.get("node_id"))].metadata.get(key) or "")
            for ref in refs
            if str(ref.get("node_id") or "") in nodes
            and str(nodes[str(ref.get("node_id"))].metadata.get(key) or "")
        }
        if not ref_values:
            failures.append(f"filter {key} cannot be derived from gold refs")
        elif not ref_values.issubset(expected_values):
            failures.append(
                f"filter {key} does not cover gold refs: "
                f"filters={sorted(expected_values)}, refs={sorted(ref_values)}"
            )
    return failures


def build_report(
    *,
    source_rows: list[dict[str, Any]],
    eligible_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    held_out_rows: list[dict[str, Any]],
    report_path: Path,
    audited_path: Path,
    target_accepted: int | None,
) -> str:
    accepted_domains = Counter(
        domain
        for row in accepted_rows
        for domain in (row.get("filters") or {}).get("policy_domain", [])
    )
    accepted_types = Counter(row.get("question_type") for row in accepted_rows)
    accepted_bundles = Counter(row.get("bundle_type") for row in accepted_rows)
    rejected_reasons = Counter(
        str(reason).split(":")[0]
        for row in rejected_rows
        for reason in row.get("failures") or []
    )
    avg_chunks = average(
        len({str(ref.get("node_id") or "") for ref in row.get("gold_evidence_refs") or [] if ref.get("node_id")})
        for row in accepted_rows
    )
    avg_points = average(len(row.get("answer_points") or []) for row in accepted_rows)
    avg_nodes_per_point = average(
        len(group.get("gold_node_ids") or [])
        for row in accepted_rows
        for group in row.get("gold_evidence_groups") or []
    )
    lines = [
        "# Policy RAG Eval Set v2 Audit Report",
        "",
        "This report audits candidate questions against local policy node text.",
        "RAGAS/DeepSeek output is treated as candidate data only; accepted rows must bind to real node IDs and exact local evidence text.",
        "",
        f"- Draft rows: {len(source_rows)}",
        f"- Quality-eligible rows before quota: {len(eligible_rows)}",
        f"- Target accepted rows: {target_accepted if target_accepted else 'not capped'}",
        f"- Accepted rows: {len(accepted_rows)}",
        f"- Rejected rows: {len(rejected_rows)}",
        f"- Held-out rows by quota: {len(held_out_rows)}",
        f"- Audited goldset: `{audited_path}`",
        f"- Report: `{report_path}`",
        f"- Avg gold chunks per accepted query: {avg_chunks:.2f}",
        f"- Avg answer points per accepted query: {avg_points:.2f}",
        f"- Avg node IDs per evidence group: {avg_nodes_per_point:.2f}",
        "",
        "## Accepted Bundle Types",
        "",
        "| Bundle type | Count |",
        "|---|---:|",
    ]
    for bundle_type, count in accepted_bundles.most_common():
        lines.append(f"| `{bundle_type}` | {count} |")
    lines.extend([
        "",
        "## Accepted Question Types",
        "",
        "| Question type | Count |",
        "|---|---:|",
    ])
    for question_type, count in accepted_types.most_common():
        lines.append(f"| `{question_type}` | {count} |")
    lines.extend(["", "## Accepted Policy Domains", "", "| Policy domain | Count |", "|---|---:|"])
    for domain, count in accepted_domains.most_common():
        lines.append(f"| `{domain}` | {count} |")
    lines.extend(["", "## Manual Spot Check Sample", "", "| Query | Question | Evidence groups |", "|---|---|---:|"])
    for row in accepted_rows[:30]:
        question = str(row.get("question") or "").replace("|", "\\|")
        lines.append(
            f"| `{row.get('query_id')}` | {question} | "
            f"{len(row.get('gold_evidence_groups') or [])} |"
        )
    if rejected_reasons:
        lines.extend(["", "## Rejection Reason Summary", "", "| Reason | Count |", "|---|---:|"])
        for reason, count in rejected_reasons.most_common(30):
            escaped_reason = reason.replace("|", "\\|")
            lines.append(f"| {escaped_reason} | {count} |")
    if rejected_rows:
        lines.extend(["", "## Rejected Rows", "", "| Query | Candidate | Failures |", "|---|---|---|"])
        for row in rejected_rows[:80]:
            failures = "；".join(str(item) for item in row.get("failures") or [])
            failures = failures.replace("|", "\\|")
            lines.append(
                f"| `{row.get('query_id')}` | `{row.get('candidate_id')}` | {failures} |"
            )
    return "\n".join(lines) + "\n"


def average(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v2-audit] failed: {exc}", file=sys.stderr)
        raise
