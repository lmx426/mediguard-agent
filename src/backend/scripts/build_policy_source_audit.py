"""Audit RAG-ready policy sources before chunk generation.

This script is an ingestion gate for the policy RAG corpus. It validates source
provenance, default-ingest boundaries, local file existence, document metadata,
and table chunk traceability before any embedding or vector-store work happens.
It does not create chunks, embeddings, indexes, or audit decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_RAG_READY_ROOT = DEFAULT_CORPUS_ROOT / "rag_ready"
DEFAULT_INGESTION_MANIFEST = (
    DEFAULT_RAG_READY_ROOT / "metadata" / "lightrag_ingestion_manifest.jsonl"
)
DEFAULT_EXCLUDED_MANIFEST = (
    DEFAULT_RAG_READY_ROOT / "metadata" / "lightrag_excluded_sources.jsonl"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_RAG_READY_ROOT / "metadata" / "rag_source_manifest.jsonl"
DEFAULT_ACCEPTANCE_REVIEW = (
    DEFAULT_RAG_READY_ROOT / "metadata" / "source_acceptance_review.jsonl"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_chunk_source_audit_report.md"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_chunk_source_audit_report.json"
)


ALLOWED_DEFAULT_RESOURCE_TYPES = {"markdown_document", "jsonl_chunks"}
ALLOWED_JURISDICTIONS = {"national", "beijing", "shanghai", "multi"}
BLOCKED_ROLE_MARKERS = ("trace_", "reference_", "interpretation_")
VALID_LEGAL_WEIGHTS = {
    "binding",
    "catalog",
    "service_guide",
    "reference_only",
    "derived_lookup_hint",
    "unknown",
}


@dataclass(slots=True)
class AuditRecord:
    index: int
    status: str
    source_id: str
    title: str
    jurisdiction: str
    policy_domain: str
    resource_type: str
    rag_ingestion_role: str
    legal_weight: str
    ingestion_path: str
    source_url: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "source_id": self.source_id,
            "title": self.title,
            "jurisdiction": self.jurisdiction,
            "policy_domain": self.policy_domain,
            "resource_type": self.resource_type,
            "rag_ingestion_role": self.rag_ingestion_role,
            "legal_weight": self.legal_weight,
            "ingestion_path": self.ingestion_path,
            "source_url": self.source_url,
            "failures": self.failures,
            "warnings": self.warnings,
            "notes": self.notes,
            "measured": self.measured,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit default policy RAG sources before chunk generation."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ingestion-manifest", type=Path, default=DEFAULT_INGESTION_MANIFEST)
    parser.add_argument("--excluded-manifest", type=Path, default=DEFAULT_EXCLUDED_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--acceptance-review", type=Path, default=DEFAULT_ACCEPTANCE_REVIEW)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path} at line {line_no}")
            records.append(payload)
    return records


def read_jsonl_head(path: Path, *, limit: int = 3) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            if len(records) >= limit:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records, total


def parse_front_matter(text: str) -> tuple[dict[str, str], str, bool]:
    if not text.startswith("---"):
        return {}, text, False
    match = re.match(r"^---\s*\n(?P<meta>[\s\S]*?)\n---\s*\n?(?P<body>[\s\S]*)$", text)
    if not match:
        return {}, text, False
    metadata: dict[str, str] = {}
    for line in match.group("meta").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or ":" not in item:
            continue
        key, value = item.split(":", 1)
        clean_value = value.strip()
        if (
            len(clean_value) >= 2
            and clean_value[0] == clean_value[-1]
            and clean_value[0] in {"'", '"'}
        ):
            clean_value = clean_value[1:-1]
        metadata[key.strip()] = clean_value
    return metadata, match.group("body"), True


def source_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = str(record.get("source_id") or "").strip()
        if source_id and source_id not in index:
            index[source_id] = record
    return index


def acceptance_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = str(record.get("source_id") or "").strip()
        if source_id and source_id not in index:
            index[source_id] = record
    return index


def infer_legal_weight(
    *,
    manifest_record: dict[str, Any],
    source_record: dict[str, Any] | None,
    front_matter: dict[str, str] | None,
    sample_records: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    for source_name, value in (
        ("manifest", manifest_record.get("legal_weight")),
        ("rag_source_manifest", (source_record or {}).get("legal_weight")),
        ("front_matter", (front_matter or {}).get("legal_weight")),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return normalized, source_name

    for sample in sample_records or []:
        normalized = str(sample.get("legal_weight") or "").strip()
        if normalized:
            return normalized, "jsonl_sample"

    role = str(manifest_record.get("rag_ingestion_role") or "")
    if "catalog_table_primary" in role:
        return "catalog", "inferred_from_role"
    if "price_table_primary" in role or "directory_table_primary" in role:
        return "catalog", "inferred_from_role"
    if "business_rule_table_primary" in role or "benefit_table_primary" in role:
        return "service_guide", "inferred_from_role"
    if "core_document_table_context" in role:
        return "catalog", "inferred_from_role"
    return "unknown", "missing"


def audit_record(
    *,
    index: int,
    record: dict[str, Any],
    project_root: Path,
    excluded_paths: set[str],
    source_records_by_id: dict[str, dict[str, Any]],
    acceptance_by_id: dict[str, dict[str, Any]],
) -> AuditRecord:
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    measured: dict[str, Any] = {}

    source_id = str(record.get("source_id") or "").strip()
    title = str(record.get("title") or "").strip()
    jurisdiction = str(record.get("jurisdiction") or "").strip()
    policy_domain = str(record.get("policy_domain") or "").strip()
    resource_type = str(record.get("resource_type") or "").strip()
    role = str(record.get("rag_ingestion_role") or "").strip()
    ingestion_path = str(record.get("ingestion_path") or "").strip()
    source_url = str(record.get("source_url") or "").strip()

    if record.get("default_ingest") is not True:
        failures.append("default_ingest is not true")
    if not source_id:
        failures.append("missing source_id")
    if not title:
        failures.append("missing title")
    if jurisdiction not in ALLOWED_JURISDICTIONS:
        failures.append(f"unexpected jurisdiction: {jurisdiction or '<missing>'}")
    if not policy_domain:
        failures.append("missing policy_domain")
    if resource_type not in ALLOWED_DEFAULT_RESOURCE_TYPES:
        failures.append(f"unsupported default resource_type: {resource_type or '<missing>'}")
    if not role:
        failures.append("missing rag_ingestion_role")
    if any(marker in role for marker in BLOCKED_ROLE_MARKERS):
        failures.append(f"default manifest includes blocked role: {role}")
    if not source_url.startswith(("http://", "https://")) and role != "graph_hint":
        failures.append("missing or invalid source_url")
    elif not source_url.startswith(("http://", "https://")) and role == "graph_hint":
        warnings.append("manifest source_url is empty; graph-hint rows must carry source_url")
    if not ingestion_path:
        failures.append("missing ingestion_path")
    if ingestion_path in excluded_paths:
        failures.append("same ingestion_path also appears in excluded sources")

    local_path = project_root / ingestion_path if ingestion_path else None
    if local_path is None or not local_path.exists():
        failures.append("local ingestion_path does not exist")
    elif local_path.is_dir():
        failures.append("ingestion_path is a directory, expected file")
    else:
        measured["file_bytes"] = local_path.stat().st_size

    source_record = source_records_by_id.get(source_id)
    acceptance_record = acceptance_by_id.get(source_id)
    front_matter: dict[str, str] | None = None
    sample_records: list[dict[str, Any]] | None = None

    if source_record is None:
        warnings.append("source_id not found in rag_source_manifest; metadata will rely on ingestion file")
    else:
        status = str(source_record.get("acceptance_status") or "")
        if status not in {"basic_satisfied", "fully_satisfied"}:
            failures.append(f"source acceptance status is not RAG-ready: {status or '<missing>'}")

    if acceptance_record is None:
        warnings.append("source_id not found in source_acceptance_review")
    else:
        if acceptance_record.get("can_enter_rag_ready") is False:
            failures.append("source_acceptance_review marks source as not enterable")
        elif "can_enter_rag_ready" not in acceptance_record:
            notes.append("acceptance review uses legacy format without can_enter_rag_ready")
        status = str(acceptance_record.get("acceptance_status") or "")
        if status not in {"basic_satisfied", "fully_satisfied"}:
            failures.append(f"acceptance_review status is not RAG-ready: {status or '<missing>'}")

    if local_path is not None and local_path.exists() and local_path.is_file():
        if resource_type == "markdown_document":
            text = local_path.read_text(encoding="utf-8", errors="replace")
            front_matter, body, has_front_matter = parse_front_matter(text)
            measured["body_chars"] = len(body)
            measured["heading_count"] = len(re.findall(r"(?m)^#{1,6}\s+", body))
            measured["page_anchor_count"] = len(re.findall(r"(?m)^##\s+Page\s+\d+", body))
            if not has_front_matter:
                failures.append("markdown file has no front matter")
            else:
                for key in (
                    "doc_id",
                    "source_id",
                    "title",
                    "jurisdiction",
                    "policy_domain",
                    "doc_type",
                    "legal_weight",
                    "source_url",
                ):
                    if not str(front_matter.get(key) or "").strip():
                        failures.append(f"markdown front matter missing {key}")
                if front_matter.get("source_id") and front_matter.get("source_id") != source_id:
                    failures.append("markdown front matter source_id differs from manifest")
                if front_matter.get("source_url") and front_matter.get("source_url") != source_url:
                    warnings.append("markdown front matter source_url differs from manifest")
            if len(body.strip()) < 180:
                warnings.append("markdown body is very short; keep only if source is an FAQ or compact guide")
            if measured["page_anchor_count"]:
                notes.append("page anchors retained for traceability")
        elif resource_type == "jsonl_chunks":
            sample_records, actual_rows = read_jsonl_head(local_path)
            measured["actual_rows"] = actual_rows
            declared_rows = record.get("row_count") or record.get("chunk_count")
            if declared_rows is not None:
                try:
                    declared_int = int(declared_rows)
                    measured["declared_rows"] = declared_int
                    if declared_int != actual_rows:
                        warnings.append(
                            f"declared row/chunk count {declared_int} differs from actual {actual_rows}"
                        )
                except (TypeError, ValueError):
                    warnings.append("declared row/chunk count is not numeric")
            if actual_rows == 0:
                failures.append("jsonl chunk file is empty")
            for row_no, sample in enumerate(sample_records, start=1):
                if role == "graph_hint":
                    for key in ("alias_id", "alias", "canonical_term", "source_url"):
                        if not str(sample.get(key) or "").strip():
                            failures.append(f"graph hint sample row {row_no} missing {key}")
                    if str(sample.get("source_url") or "").startswith(("http://", "https://")):
                        notes.append("graph hint sample rows carry source_url")
                    continue
                content = str(sample.get("content") or sample.get("text") or "").strip()
                if not content:
                    failures.append(f"jsonl sample row {row_no} has no content/text")
                for key in (
                    "source_id",
                    "title",
                    "jurisdiction",
                    "policy_domain",
                    "source_url",
                    "rag_ingestion_role",
                ):
                    if not str(sample.get(key) or "").strip():
                        failures.append(f"jsonl sample row {row_no} missing {key}")
                if sample.get("source_id") and sample.get("source_id") != source_id:
                    warnings.append(f"jsonl sample row {row_no} source_id differs from manifest")

    legal_weight, legal_weight_source = infer_legal_weight(
        manifest_record=record,
        source_record=source_record,
        front_matter=front_matter,
        sample_records=sample_records,
    )
    measured["legal_weight_source"] = legal_weight_source
    if legal_weight not in VALID_LEGAL_WEIGHTS:
        failures.append(f"unexpected legal_weight: {legal_weight}")
    if legal_weight == "unknown":
        warnings.append("legal_weight is unknown")
    if legal_weight_source == "inferred_from_role":
        warnings.append("legal_weight inferred from ingestion role; prefer explicit source metadata")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return AuditRecord(
        index=index,
        status=status,
        source_id=source_id,
        title=title,
        jurisdiction=jurisdiction,
        policy_domain=policy_domain,
        resource_type=resource_type,
        rag_ingestion_role=role,
        legal_weight=legal_weight,
        ingestion_path=ingestion_path,
        source_url=source_url,
        failures=failures,
        warnings=warnings,
        notes=notes,
        measured=measured,
    )


def build_markdown_report(records: list[AuditRecord], *, generated_at: str) -> str:
    status_counts = Counter(record.status for record in records)
    jurisdiction_counts = Counter(record.jurisdiction for record in records)
    role_counts = Counter(record.rag_ingestion_role for record in records)
    legal_weight_counts = Counter(record.legal_weight for record in records)
    resource_type_counts = Counter(record.resource_type for record in records)
    fail_count = status_counts.get("FAIL", 0)
    block_chunk_generation = fail_count > 0

    lines: list[str] = [
        "# Policy Chunk Source Audit Report",
        "",
        f"Updated at: {generated_at}",
        "",
        "This report validates the default RAG ingestion manifest before generating `policy_chunks.jsonl`. It does not create chunks, embeddings, vector indexes, or audit conclusions.",
        "",
        "## Summary",
        "",
        f"- Default-ingest records audited: {len(records)}",
        f"- PASS: {status_counts.get('PASS', 0)}",
        f"- WARN: {status_counts.get('WARN', 0)}",
        f"- FAIL: {status_counts.get('FAIL', 0)}",
        f"- Block chunk generation: {'yes' if block_chunk_generation else 'no'}",
        "",
        "## Distribution",
        "",
        "### Jurisdiction",
        "",
    ]
    for key, count in sorted(jurisdiction_counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Resource Type", ""])
    for key, count in sorted(resource_type_counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Legal Weight", ""])
    for key, count in sorted(legal_weight_counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Ingestion Role", ""])
    for key, count in sorted(role_counts.items()):
        lines.append(f"- `{key}`: {count}")

    lines.extend(
        [
            "",
            "## Source Gate Table",
            "",
            "| # | Status | Jurisdiction | Domain | Role | Legal weight | Title | Notes |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for record in records:
        notes = []
        if record.failures:
            notes.extend(f"FAIL: {item}" for item in record.failures)
        if record.warnings:
            notes.extend(f"WARN: {item}" for item in record.warnings)
        if record.notes:
            notes.extend(f"NOTE: {item}" for item in record.notes)
        note_text = "<br>".join(escape_table(item) for item in notes) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.index),
                    record.status,
                    escape_table(record.jurisdiction),
                    escape_table(record.policy_domain),
                    escape_table(record.rag_ingestion_role),
                    escape_table(record.legal_weight),
                    escape_table(record.title),
                    note_text,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if block_chunk_generation:
        lines.append(
            "Do not generate `policy_chunks.jsonl` until all FAIL records are fixed or explicitly excluded from the default ingestion manifest."
        )
    else:
        lines.append(
            "No blocking source-governance failure was found. `policy_chunks.jsonl` can be generated using the confirmed chunk strategy, while WARN records should be handled as review notes."
        )
    lines.append("")
    return "\n".join(lines)


def escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    ingestion_records = load_jsonl(args.ingestion_manifest)
    excluded_records = load_jsonl(args.excluded_manifest)
    source_records = load_jsonl(args.source_manifest)
    acceptance_records = load_jsonl(args.acceptance_review)

    excluded_paths = {
        str(record.get("ingestion_path") or "").strip()
        for record in excluded_records
        if str(record.get("ingestion_path") or "").strip()
    }
    source_records_by_id = source_index(source_records)
    acceptance_by_id = acceptance_index(acceptance_records)

    audit_records = [
        audit_record(
            index=index,
            record=record,
            project_root=project_root,
            excluded_paths=excluded_paths,
            source_records_by_id=source_records_by_id,
            acceptance_by_id=acceptance_by_id,
        )
        for index, record in enumerate(ingestion_records, start=1)
    ]

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": generated_at,
        "input_manifest": str(args.ingestion_manifest),
        "records_audited": len(audit_records),
        "status_counts": dict(Counter(record.status for record in audit_records)),
        "block_chunk_generation": any(record.status == "FAIL" for record in audit_records),
        "records": [record.to_json() for record in audit_records],
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        build_markdown_report(audit_records, generated_at=generated_at),
        encoding="utf-8",
        newline="\n",
    )

    print(json.dumps(report["status_counts"], ensure_ascii=False, sort_keys=True))
    print(f"block_chunk_generation={report['block_chunk_generation']}")
    print(f"report_md={args.report_md}")
    print(f"report_json={args.report_json}")


if __name__ == "__main__":
    main()
