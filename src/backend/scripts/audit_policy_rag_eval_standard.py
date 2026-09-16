"""Audit that the policy RAG evaluation standard points to real corpus evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_NODES_PATH,
)


DEFAULT_EVAL_SET = DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1.jsonl"
DEFAULT_ENRICHED_EVAL_SET = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_rag_eval_set_v1_2.jsonl"
)
DEFAULT_REPORT = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_eval_standard_evidence_audit.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify evaluation standards against actual policy nodes."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument(
        "--enriched-eval-set-jsonl",
        type=Path,
        default=DEFAULT_ENRICHED_EVAL_SET,
    )
    parser.add_argument("--nodes-jsonl", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def load_node_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            node_id = str(payload.get("id_") or "").strip()
            text = str(payload.get("text") or "")
            metadata = payload.get("metadata") or {}
            if not node_id or not text.strip() or not isinstance(metadata, dict):
                raise ValueError(f"Invalid policy node at {path}:{line_no}")
            source_id = str(metadata.get("source_id") or "").strip()
            doc_id = str(metadata.get("doc_id") or "").strip()
            record = {
                "node_id": node_id,
                "text": text,
                "metadata": metadata,
            }
            for alias in {source_id, doc_id}:
                if alias:
                    by_source[alias].append(record)
    return by_source


def anchor_hit(text: str, anchors: list[str]) -> bool:
    return bool(anchors) and all(str(anchor) in text for anchor in anchors)


def _record_source_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("source_id") or metadata.get("doc_id") or "").strip()


def _representative_matches(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one verified node per official source for each anchor set.

    The evaluation standard should cover equivalent official sources without
    allowing the first source in the candidate list to crowd out the others.
    """

    representatives: dict[str, dict[str, Any]] = {}
    for record in matches:
        source_key = _record_source_key(record)
        node_id = str(record.get("node_id") or "")
        key = source_key or node_id
        if key and key not in representatives:
            representatives[key] = record
    return list(representatives.values())


def _match_scope(group: dict[str, Any]) -> str:
    roles = {
        str(role)
        for role in group.get("document_roles") or []
        if str(role)
    }
    if any(
        role in roles
        for role in {
            "table_row",
            "catalog_rows",
            "price_table",
            "price_table_csv",
            "medical_service_price_rows",
            "institution_catalog",
            "material_chain_table",
        }
    ):
        return "same_row"
    return "same_clause"


def _gap_type(query: dict[str, Any]) -> str:
    explicit = str(query.get("gap_type") or "").strip()
    if explicit:
        return explicit
    query_id = str(query.get("query_id") or "")
    if str(query.get("expected_status") or "") != "coverage_gap":
        return "none"
    if query_id.startswith("case_4_"):
        return "capability_gap"
    return "corpus_gap"


def audit_query(
    query: dict[str, Any],
    node_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected_status = str(query.get("expected_status") or "unknown")
    groups: list[dict[str, Any]] = []
    for group in query.get("expected_source_groups") or []:
        sources = [
            str(item)
            for item in (
                group.get("acceptable_sources")
                or group.get("match_any")
                or []
            )
            if str(item)
        ]
        candidates = [
            record
            for source in sources
            for record in node_index.get(source, [])
        ]
        anchor_sets = list(group.get("content_anchor_sets") or [])
        matched_anchor_sets: list[list[str]] = []
        matched_node_ids: list[str] = []
        matched_evidence_refs: list[dict[str, Any]] = []
        for anchors in anchor_sets:
            matches = [
                record
                for record in candidates
                if anchor_hit(str(record["text"]), list(anchors))
            ]
            if matches:
                matched_anchor_sets.append(list(anchors))
                for record in _representative_matches(matches):
                    node_id = str(record["node_id"])
                    matched_node_ids.append(node_id)
                    metadata = record["metadata"]
                    matched_evidence_refs.append(
                        {
                            "group": str(group.get("group") or ""),
                            "node_id": node_id,
                            "source_id": str(metadata.get("source_id") or ""),
                            "doc_id": str(metadata.get("doc_id") or ""),
                            "anchor": list(anchors),
                            "match_scope": _match_scope(group),
                    }
                )
        if not anchor_sets and candidates:
            for record in _representative_matches(candidates):
                node_id = str(record["node_id"])
                matched_node_ids.append(node_id)
                metadata = record["metadata"]
                matched_evidence_refs.append(
                    {
                        "group": str(group.get("group") or ""),
                        "node_id": node_id,
                        "source_id": str(metadata.get("source_id") or ""),
                        "doc_id": str(metadata.get("doc_id") or ""),
                        "anchor": [],
                        "match_scope": _match_scope(group),
                    }
                )
        verified = bool(candidates) and (
            bool(matched_anchor_sets) if anchor_sets else True
        )
        groups.append(
            {
                "group": group.get("group"),
                "acceptable_sources": sources,
                "source_present": bool(candidates),
                "anchors_defined": bool(anchor_sets),
                "matched_anchor_sets": matched_anchor_sets,
                "matched_node_ids": sorted(set(matched_node_ids)),
                "matched_source_ids": sorted(
                    {
                        str(ref.get("source_id") or "")
                        for ref in matched_evidence_refs
                        if str(ref.get("source_id") or "")
                    }
                ),
                "matched_evidence_refs": _dedupe_refs(matched_evidence_refs),
                "verified": verified,
                "evidence_status": "available" if verified else "missing",
            }
        )

    if expected_status == "coverage_gap":
        verified = not any(group["verified"] for group in groups)
    else:
        verified = bool(groups) and all(group["verified"] for group in groups)

    return {
        "query_id": query.get("query_id"),
        "case_id": query.get("case_id"),
        "expected_status": expected_status,
        "verified": verified,
        "groups": groups,
        "coverage_gap_reason": query.get("coverage_gap_reason"),
        "gap_type": _gap_type(query),
    }


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ref in refs:
        key = (
            str(ref.get("node_id") or ""),
            str(ref.get("source_id") or ""),
            json.dumps(ref.get("anchor") or [], ensure_ascii=False),
        )
        unique[key] = ref
    return list(unique.values())


def enrich_query(
    query: dict[str, Any],
    audit_row: dict[str, Any],
) -> dict[str, Any]:
    expected_status = str(query.get("expected_status") or "unknown")
    if expected_status == "supported" and audit_row["verified"]:
        evidence_status = "full"
    elif expected_status == "partial_supported" and audit_row["verified"]:
        evidence_status = "partial"
    else:
        evidence_status = "none"

    evidence_groups: list[dict[str, Any]] = []
    gold_refs: list[dict[str, Any]] = []
    canonical_sources: set[str] = set()
    for group in audit_row["groups"]:
        refs = list(group.get("matched_evidence_refs") or [])
        gold_refs.extend(refs)
        canonical_sources.update(
            str(ref.get("source_id") or "")
            for ref in refs
            if str(ref.get("source_id") or "")
        )
        evidence_groups.append(
            {
                "group": group.get("group"),
                "status": group.get("evidence_status"),
                "matched_node_ids": group.get("matched_node_ids") or [],
                "matched_source_ids": group.get("matched_source_ids") or [],
                "matched_anchor_sets": group.get("matched_anchor_sets") or [],
                "gold_node_ids": group.get("matched_node_ids") or [],
            }
        )

    enriched = dict(query)
    enriched["schema_version"] = "policy_rag_eval_v1_2"
    enriched["evidence_status"] = evidence_status
    enriched["gap_type"] = audit_row["gap_type"]
    enriched["metric_eligibility"] = {
        "strict_retrieval": evidence_status == "full",
        "partial_retrieval": evidence_status == "partial",
        "gap_detection": evidence_status == "none",
    }
    enriched["evidence_groups"] = evidence_groups
    enriched["gold_evidence_refs"] = _dedupe_refs(gold_refs)
    enriched["canonical_source_ids"] = sorted(canonical_sources)
    enriched["standard_evidence_audit"] = {
        "verified": audit_row["verified"],
        "query_evidence_status": evidence_status,
        "matched_group_count": sum(
            1 for group in audit_row["groups"] if group["verified"]
        ),
        "group_count": len(audit_row["groups"]),
    }
    return enriched


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    eval_queries: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    eval_path: Path,
    nodes_path: Path,
    enriched_path: Path,
) -> str:
    verified_count = sum(1 for row in audit_rows if row["verified"])
    status_counts = {
        "full": sum(
            1
            for query, row in zip(eval_queries, audit_rows, strict=True)
            if str(query.get("expected_status") or "") == "supported"
            and row["verified"]
        ),
        "partial": sum(
            1
            for query, row in zip(eval_queries, audit_rows, strict=True)
            if str(query.get("expected_status") or "") == "partial_supported"
            and row["verified"]
        ),
        "none": sum(
            1
            for query, row in zip(eval_queries, audit_rows, strict=True)
            if str(query.get("expected_status") or "") == "coverage_gap"
        ),
    }
    lines = [
        "# Policy RAG Evaluation Standard Evidence Audit",
        "",
        "This report verifies that the confirmed evaluation standard references "
        "actual indexed node text and metadata. It does not use file names alone "
        "as evidence and does not use an LLM judge.",
        "",
        f"- Eval set: `{eval_path}`",
        f"- Enriched eval set: `{enriched_path}`",
        f"- Nodes: `{nodes_path}`",
        f"- Query count: {len(eval_queries)}",
        f"- Verified query count: {verified_count}/{len(audit_rows)}",
        f"- Evidence status counts: `{status_counts}`",
        "",
        "| Query | Expected status | Evidence status | Gap type | Verified | Evidence groups |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in audit_rows:
        group_count = len(row["groups"])
        expected_status = str(row["expected_status"])
        evidence_status = (
            "full"
            if expected_status == "supported" and row["verified"]
            else "partial"
            if expected_status == "partial_supported" and row["verified"]
            else "none"
        )
        lines.append(
            f"| `{row['query_id']}` | `{expected_status}` | `{evidence_status}` | "
            f"`{row['gap_type']}` | {'yes' if row['verified'] else 'NO'} | "
            f"{group_count} |"
        )
        for group in row["groups"]:
            matched = ", ".join(group["matched_node_ids"]) or "none"
            anchors = " / ".join(
                " + ".join(str(anchor) for anchor in anchor_set)
                for anchor_set in group["matched_anchor_sets"]
            ) or "none"
            lines.append(
                f"  - `{group['group']}`: source_present="
                f"`{group['source_present']}`, anchors=`{anchors}`, "
                f"sources=`{', '.join(group.get('matched_source_ids') or [])}`, "
                f"node_ids=`{matched}`, status=`{group['evidence_status']}`"
            )
        if row["coverage_gap_reason"]:
            lines.append(f"  - Coverage gap reason: {row['coverage_gap_reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    eval_queries = load_jsonl(args.eval_set_jsonl)
    node_index = load_node_index(args.nodes_jsonl)
    audit_rows = [audit_query(query, node_index) for query in eval_queries]
    enriched_queries = [
        enrich_query(query, audit_row)
        for query, audit_row in zip(eval_queries, audit_rows, strict=True)
    ]
    write_jsonl(args.enriched_eval_set_jsonl, enriched_queries)
    report = build_report(
        eval_queries,
        audit_rows,
        eval_path=args.eval_set_jsonl,
        nodes_path=args.nodes_jsonl,
        enriched_path=args.enriched_eval_set_jsonl,
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report, encoding="utf-8", newline="\n")
    failed = [row["query_id"] for row in audit_rows if not row["verified"]]
    print(
        json.dumps(
            {
                "query_count": len(audit_rows),
                "verified_query_count": len(audit_rows) - len(failed),
                "failed_query_ids": failed,
                "enriched_eval_set_jsonl": str(args.enriched_eval_set_jsonl),
                "report_md": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
