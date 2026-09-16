"""Build standard policy RAG chunks from the accepted ingestion manifest.

The script reads only the default ingestion manifest and writes a single
`policy_chunks.jsonl` file for later embedding. It does not build embeddings,
vector indexes, MCP tools, or audit decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
DEFAULT_SOURCE_MANIFEST = DEFAULT_RAG_READY_ROOT / "metadata" / "rag_source_manifest.jsonl"
DEFAULT_SOURCE_AUDIT = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_chunk_source_audit_report.json"
)
DEFAULT_OUTPUT_CHUNKS = DEFAULT_RAG_READY_ROOT / "chunks" / "policy_chunks.jsonl"
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_chunk_build_report.md"
DEFAULT_REPORT_JSON = DEFAULT_CORPUS_ROOT / "reports" / "policy_chunk_build_report.json"

CHUNK_STRATEGY_ID = "policy_chunk_strategy_v0.1"
MIN_TEXT_CHARS = 20
PAGE_HEADING_RE = re.compile(r"^#{1,6}\s+Page\s+(\d+)\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ARTICLE_BOUNDARY_RE = re.compile(
    r"^(第[一二三四五六七八九十百千万\d]+条|[一二三四五六七八九十]+、|（[一二三四五六七八九十\d]+）|\([一二三四五六七八九十\d]+\)|\d+[.、])"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。；;！？!?])")
BLOCKED_ROLE_MARKERS = ("trace_", "reference_", "interpretation_")


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    profile: str
    chunk_size: int
    hard_max: int
    overlap: int
    split_method: str


POLICY_TEXT = ChunkPolicy(
    profile="policy_text",
    chunk_size=900,
    hard_max=1200,
    overlap=120,
    split_method="heading_article_sentence",
)
GUIDE_FAQ = ChunkPolicy(
    profile="service_guide_faq",
    chunk_size=650,
    hard_max=900,
    overlap=80,
    split_method="guide_section_sentence",
)
TABLE_CONTEXT = ChunkPolicy(
    profile="core_document_table_context",
    chunk_size=700,
    hard_max=1000,
    overlap=100,
    split_method="table_context_sentence",
)
ROW_LEVEL = ChunkPolicy(
    profile="row_level",
    chunk_size=0,
    hard_max=0,
    overlap=0,
    split_method="one_jsonl_row_one_chunk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build policy_chunks.jsonl from accepted RAG-ready sources."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ingestion-manifest", type=Path, default=DEFAULT_INGESTION_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-audit-report", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--output-chunks", type=Path, default=DEFAULT_OUTPUT_CHUNKS)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument(
        "--allow-without-source-audit",
        action="store_true",
        help="Allow chunk generation when the source audit report is missing.",
    )
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


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(?P<meta>[\s\S]*?)\n---\s*\n?(?P<body>[\s\S]*)$", text)
    if not match:
        return {}, text
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
    return metadata, match.group("body")


def source_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = str(record.get("source_id") or "").strip()
        if source_id and source_id not in index:
            index[source_id] = record
    return index


def require_source_audit(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    if not path.exists():
        if allow_missing:
            return {
                "available": False,
                "block_chunk_generation": False,
                "warning": "source audit report missing but explicitly allowed",
            }
        raise FileNotFoundError(
            f"Source audit report is required before chunk generation: {path}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("block_chunk_generation"):
        raise RuntimeError(
            "Source audit report blocks chunk generation; fix FAIL records first."
        )
    return report


def choose_policy(record: dict[str, Any], front_matter: dict[str, str] | None) -> ChunkPolicy:
    role = str(record.get("rag_ingestion_role") or "")
    doc_type = str((front_matter or {}).get("doc_type") or record.get("doc_type") or "")
    domain = str(record.get("policy_domain") or "")
    title = str(record.get("title") or "")

    if role == "core_document_table_context":
        return TABLE_CONTEXT
    if doc_type in {"service_guide", "faq"}:
        return GUIDE_FAQ
    if "faq" in role or "常见问题" in title or "办理" in title or "材料" in title:
        return GUIDE_FAQ
    if domain in {"manual_reimbursement"} and "guide" in str(record.get("source_id") or ""):
        return GUIDE_FAQ
    return POLICY_TEXT


def infer_legal_weight(
    *,
    record: dict[str, Any],
    source_record: dict[str, Any] | None,
    front_matter: dict[str, str] | None,
    row_payload: dict[str, Any] | None = None,
) -> str:
    for value in (
        record.get("legal_weight"),
        (front_matter or {}).get("legal_weight"),
        (row_payload or {}).get("legal_weight"),
        (source_record or {}).get("legal_weight"),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return normalized

    role = str(record.get("rag_ingestion_role") or "")
    if role in {"catalog_table_primary", "price_table_primary", "directory_table_primary"}:
        return "catalog"
    if role in {"business_rule_table_primary", "benefit_table_primary"}:
        return "service_guide"
    if role == "graph_hint":
        return "derived_lookup_hint"
    if role == "core_document_table_context":
        return "catalog"
    return "unknown"


def build_base_metadata(
    *,
    record: dict[str, Any],
    source_record: dict[str, Any] | None,
    front_matter: dict[str, str] | None,
    policy: ChunkPolicy,
    source_path: str,
    content_type: str,
    legal_weight: str,
) -> dict[str, Any]:
    metadata = {
        "source_id": str(record.get("source_id") or "").strip(),
        "title": str(record.get("title") or "").strip(),
        "jurisdiction": str(record.get("jurisdiction") or "").strip(),
        "policy_domain": str(record.get("policy_domain") or "").strip(),
        "legal_weight": legal_weight,
        "rag_ingestion_role": str(record.get("rag_ingestion_role") or "").strip(),
        "content_type": content_type,
        "resource_type": str(record.get("resource_type") or "").strip(),
        "source_url": str(record.get("source_url") or "").strip(),
        "source_path": source_path,
        "chunk_strategy": CHUNK_STRATEGY_ID,
        "chunk_profile": policy.profile,
        "chunk_size": policy.chunk_size,
        "chunk_overlap": policy.overlap,
        "chunk_hard_max": policy.hard_max,
        "chunk_unit": "chinese_char",
        "split_method": policy.split_method,
    }
    for key in (
        "doc_id",
        "doc_type",
        "publish_date",
        "effective_date",
        "expiry_date",
        "status",
        "document_no",
        "sha256",
    ):
        value = (front_matter or {}).get(key)
        if not value:
            value = (source_record or {}).get(key)
        if value not in (None, ""):
            metadata[key] = value
    return metadata


def normalize_block(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def markdown_units(body: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    page_refs: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    paragraph_lines: list[str] = []
    paragraph_pages: set[str] = set()
    seen_blocks: set[str] = set()

    def current_heading() -> str:
        return " > ".join(heading for _, heading in heading_stack[-3:])

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_pages
        text = cleanup_markdown_text("\n".join(paragraph_lines))
        paragraph_lines = []
        if not text:
            paragraph_pages = set()
            return
        dedupe_key = normalize_block(text)
        if len(dedupe_key) > 30:
            if dedupe_key in seen_blocks:
                paragraph_pages = set()
                return
            seen_blocks.add(dedupe_key)
        units.append(
            {
                "text": text,
                "section_heading": current_heading(),
                "page_refs": sorted(paragraph_pages or set(page_refs[-1:])),
            }
        )
        paragraph_pages = set()

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        page_match = PAGE_HEADING_RE.match(stripped)
        if page_match:
            flush_paragraph()
            page_refs.append(f"Page {page_match.group(1)}")
            continue
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading = cleanup_markdown_text(heading_match.group(2))
            heading_stack = [(item_level, item) for item_level, item in heading_stack if item_level < level]
            heading_stack.append((level, heading))
            units.append(
                {
                    "text": heading,
                    "section_heading": current_heading(),
                    "page_refs": sorted(set(page_refs[-1:])),
                    "is_heading": True,
                }
            )
            continue
        if ARTICLE_BOUNDARY_RE.match(stripped):
            flush_paragraph()
        paragraph_lines.append(stripped)
        if page_refs:
            paragraph_pages.add(page_refs[-1])
    flush_paragraph()
    return units


def cleanup_markdown_text(text: str) -> str:
    text = re.sub(r"(?m)^\|?\s*[-:|]{3,}\s*\|?\s*$", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_text(text: str, *, hard_max: int) -> list[str]:
    if len(text) <= hard_max:
        return [text]
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        return [text[index : index + hard_max] for index in range(0, len(text), hard_max)]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > hard_max:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(sentence[index : index + hard_max] for index in range(0, len(sentence), hard_max))
            continue
        candidate = f"{current}{sentence}" if not current else f"{current} {sentence}"
        if len(candidate) > hard_max and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    boundary = max(tail.rfind("。"), tail.rfind("；"), tail.rfind(";"), tail.rfind(" "))
    if boundary > 20:
        return tail[boundary + 1 :].strip()
    return tail.strip()


def build_markdown_chunks(
    *,
    record: dict[str, Any],
    source_record: dict[str, Any] | None,
    path: Path,
    source_path: str,
    sequence_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    front_matter, body = parse_front_matter(raw_text)
    policy = choose_policy(record, front_matter)
    legal_weight = infer_legal_weight(
        record=record,
        source_record=source_record,
        front_matter=front_matter,
    )
    base = build_base_metadata(
        record=record,
        source_record=source_record,
        front_matter=front_matter,
        policy=policy,
        source_path=source_path,
        content_type="policy_text",
        legal_weight=legal_weight,
    )
    units = [
        unit
        for unit in markdown_units(body)
        if len(str(unit.get("text") or "").strip()) >= MIN_TEXT_CHARS
    ]
    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_pages: set[str] = set()
    current_heading = ""
    overlap_prefix = ""

    def flush() -> None:
        nonlocal current_parts, current_pages, current_heading, overlap_prefix
        text = cleanup_markdown_text(" ".join(part for part in current_parts if part.strip()))
        if len(text) < MIN_TEXT_CHARS:
            current_parts = []
            current_pages = set()
            return
        chunk_index = len(chunks) + 1
        chunk_id = f"{base['source_id']}::md::{chunk_index:04d}"
        chunks.append(
            {
                **base,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "source_sequence": sequence_start,
                "section_heading": current_heading,
                "page_refs": sorted(current_pages),
                "text": text,
                "text_chars": len(text),
            }
        )
        overlap_prefix = tail_overlap(text, policy.overlap)
        current_parts = [overlap_prefix] if overlap_prefix else []
        current_pages = set()

    for unit in units:
        unit_text = str(unit.get("text") or "").strip()
        section_heading = str(unit.get("section_heading") or "").strip()
        pages = set(unit.get("page_refs") or [])
        for piece in split_long_text(unit_text, hard_max=policy.hard_max):
            if not current_heading:
                current_heading = section_heading
            candidate_len = len(cleanup_markdown_text(" ".join([*current_parts, piece])))
            current_len = len(cleanup_markdown_text(" ".join(current_parts)))
            should_flush = False
            if current_parts and candidate_len > policy.hard_max:
                should_flush = True
            elif (
                current_parts
                and candidate_len > policy.chunk_size
                and current_len >= max(200, policy.chunk_size // 2)
            ):
                should_flush = True
            if should_flush:
                flush()
                current_heading = section_heading
            current_parts.append(piece)
            current_pages.update(pages)
            if len(cleanup_markdown_text(" ".join(current_parts))) >= policy.hard_max:
                flush()
                current_heading = section_heading
    flush()

    source_summary = {
        "source_id": base["source_id"],
        "title": base["title"],
        "resource_type": "markdown_document",
        "chunk_profile": policy.profile,
        "input_units": len(units),
        "chunk_count": len(chunks),
    }
    return chunks, source_summary


def build_graph_hint_text(row: dict[str, Any], record: dict[str, Any]) -> str:
    related = row.get("related_terms") or []
    if isinstance(related, list):
        related_text = "、".join(str(item) for item in related)
    else:
        related_text = str(related)
    return "\n".join(
        item
        for item in [
            f"资料标题: {record.get('title')}",
            "资料类型: 检索别名提示，不是政策依据",
            f"地区: {row.get('jurisdiction') or record.get('jurisdiction')}",
            f"政策领域: {row.get('policy_domain') or record.get('policy_domain')}",
            f"别名: {row.get('alias')}",
            f"规范名称: {row.get('canonical_term')}",
            f"相关词: {related_text}",
            f"证据说明: {row.get('evidence_note')}",
            f"官方来源: {row.get('source_url') or record.get('source_url')}",
        ]
        if str(item).strip() and not str(item).endswith(": None")
    )


def stable_row_scope(source_path: str) -> str:
    """Build a compact, stable scope so row-level IDs stay unique per input file."""
    stem = Path(source_path).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not normalized:
        normalized = "source"
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:48]}_{digest}"


def build_jsonl_chunks(
    *,
    record: dict[str, Any],
    source_record: dict[str, Any] | None,
    path: Path,
    source_path: str,
    sequence_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    role = str(record.get("rag_ingestion_role") or "")
    content_type = "graph_hint" if role == "graph_hint" else "table_row"
    row_scope = stable_row_scope(source_path)
    row_number = 0
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row_number += 1
            row = json.loads(stripped)
            if not isinstance(row, dict):
                continue
            text = str(row.get("content") or row.get("text") or "").strip()
            if role == "graph_hint":
                text = build_graph_hint_text(row, record)
            if not text:
                continue
            policy = ROW_LEVEL
            legal_weight = infer_legal_weight(
                record=record,
                source_record=source_record,
                front_matter=None,
                row_payload=row,
            )
            row_url = str(row.get("source_url") or record.get("source_url") or "").strip()
            base = build_base_metadata(
                record={**record, "source_url": row_url or record.get("source_url")},
                source_record=source_record,
                front_matter=None,
                policy=policy,
                source_path=source_path,
                content_type=content_type,
                legal_weight=legal_weight,
            )
            original_chunk_id = str(row.get("chunk_id") or row.get("row_id") or row.get("alias_id") or "")
            chunk_index = len(chunks) + 1
            chunk = {
                **base,
                "chunk_id": f"{base['source_id']}::row::{row_scope}::{chunk_index:06d}",
                "chunk_index": chunk_index,
                "source_sequence": sequence_start,
                "row_scope": row_scope,
                "source_row_number": row_number,
                "source_line_number": line_no,
                "original_chunk_id": original_chunk_id,
                "page_refs": page_refs_for_row(row),
                "text": text,
                "text_chars": len(text),
            }
            for key in (
                "row_number",
                "item_code",
                "item_name",
                "drug_name",
                "catalog_no",
                "insurance_class",
                "institution_name",
                "pharmacy_name",
                "alias",
                "canonical_term",
                "source_table_path",
            ):
                if row.get(key) not in (None, ""):
                    chunk[key] = row[key]
            chunks.append(chunk)

    source_summary = {
        "source_id": str(record.get("source_id") or ""),
        "title": str(record.get("title") or ""),
        "resource_type": "jsonl_chunks",
        "chunk_profile": "row_level",
        "input_units": row_number,
        "chunk_count": len(chunks),
    }
    return chunks, source_summary


def page_refs_for_row(row: dict[str, Any]) -> list[str]:
    value = row.get("page_number") or row.get("page") or row.get("page_ref")
    if value in (None, ""):
        return []
    return [f"Page {value}" if str(value).isdigit() else str(value)]


def validate_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    empty = [chunk["chunk_id"] for chunk in chunks if not str(chunk.get("text") or "").strip()]
    missing_url = [chunk["chunk_id"] for chunk in chunks if not str(chunk.get("source_url") or "").startswith(("http://", "https://")) and chunk.get("content_type") != "graph_hint"]
    blocked_roles = [
        chunk["chunk_id"]
        for chunk in chunks
        if any(marker in str(chunk.get("rag_ingestion_role") or "") for marker in BLOCKED_ROLE_MARKERS)
    ]
    over_hard = [
        chunk["chunk_id"]
        for chunk in chunks
        if int(chunk.get("chunk_hard_max") or 0) > 0
        and int(chunk.get("text_chars") or 0) > int(chunk.get("chunk_hard_max") or 0) + 20
    ]
    chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    duplicate_chunk_ids = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if chunk_id and count > 1
    ]
    return {
        "empty_chunks": empty,
        "missing_source_url": missing_url,
        "blocked_roles": blocked_roles,
        "over_hard_max": over_hard,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "is_valid": not (empty or missing_url or blocked_roles or over_hard or duplicate_chunk_ids),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy Chunk Build Report",
        "",
        f"Updated at: {report['generated_at']}",
        "",
        "This report describes `policy_chunks.jsonl` generation. It does not build embeddings, vector indexes, MCP tools, or audit decisions.",
        "",
        "## Summary",
        "",
        f"- Input manifest records: {report['input_records']}",
        f"- Sources processed: {report['sources_processed']}",
        f"- Total chunks: {report['total_chunks']}",
        f"- Output: `{report['output_chunks']}`",
        f"- Strategy: `{CHUNK_STRATEGY_ID}`",
        f"- Source audit gate: {report['source_audit_gate']}",
        f"- Output valid: {'yes' if report['validation']['is_valid'] else 'no'}",
        "",
        "## Chunk Strategy",
        "",
        "| Data type | chunk size | hard max | overlap | method |",
        "|---|---:|---:|---:|---|",
        "| Policy text MD | 900 | 1200 | 120 | heading/article first, sentence fallback |",
        "| Fund supervision MD | 900 | 1200 | 120 | article-aware policy text profile |",
        "| Service guide / FAQ MD | 650 | 900 | 80 | guide/section sentence split |",
        "| core_document_table_context MD | 700 | 1000 | 100 | context text only; tables stay row-level |",
        "| JSONL tables | row-level | not split | 0 | one JSONL row per chunk |",
        "| graph hints | row-level | not split | 0 | retrieval hint only, not policy basis |",
        "",
        "## Distribution",
        "",
        "### Content Type",
        "",
    ]
    for key, count in sorted(report["by_content_type"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Jurisdiction", ""])
    for key, count in sorted(report["by_jurisdiction"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "### Ingestion Role", ""])
    for key, count in sorted(report["by_rag_ingestion_role"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            "| # | Chunks | Type | Profile | Source | Title |",
            "|---:|---:|---|---|---|---|",
        ]
    )
    for index, source in enumerate(report["source_summaries"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(source["chunk_count"]),
                    escape_table(source["resource_type"]),
                    escape_table(source["chunk_profile"]),
                    escape_table(source["source_id"]),
                    escape_table(source["title"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Validation", ""])
    validation = report["validation"]
    for key in (
        "empty_chunks",
        "missing_source_url",
        "blocked_roles",
        "over_hard_max",
        "duplicate_chunk_ids",
    ):
        values = validation[key]
        lines.append(f"- `{key}`: {len(values)}")
    lines.append("")
    return "\n".join(lines)


def escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_audit = require_source_audit(
        args.source_audit_report,
        allow_missing=args.allow_without_source_audit,
    )
    manifest_records = [
        record
        for record in load_jsonl(args.ingestion_manifest)
        if record.get("default_ingest") is True
    ]
    source_records_by_id = source_index(load_jsonl(args.source_manifest))

    chunks: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_sequence, record in enumerate(manifest_records, start=1):
        source_id = str(record.get("source_id") or "")
        source_record = source_records_by_id.get(source_id)
        ingestion_path = str(record.get("ingestion_path") or "")
        local_path = project_root / ingestion_path
        if not local_path.exists():
            raise FileNotFoundError(f"Ingestion path does not exist: {ingestion_path}")
        resource_type = str(record.get("resource_type") or "")
        role = str(record.get("rag_ingestion_role") or "")
        if any(marker in role for marker in BLOCKED_ROLE_MARKERS):
            raise RuntimeError(f"Blocked role in default manifest: {role}")
        if resource_type == "markdown_document":
            source_chunks, summary = build_markdown_chunks(
                record=record,
                source_record=source_record,
                path=local_path,
                source_path=ingestion_path,
                sequence_start=source_sequence,
            )
        elif resource_type == "jsonl_chunks":
            source_chunks, summary = build_jsonl_chunks(
                record=record,
                source_record=source_record,
                path=local_path,
                source_path=ingestion_path,
                sequence_start=source_sequence,
            )
        else:
            raise RuntimeError(f"Unsupported default resource_type: {resource_type}")
        chunks.extend(source_chunks)
        source_summaries.append(summary)

    validation = validate_chunks(chunks)
    if not validation["is_valid"]:
        raise RuntimeError(f"Generated chunks failed validation: {validation}")

    args.output_chunks.parent.mkdir(parents=True, exist_ok=True)
    with args.output_chunks.open("w", encoding="utf-8", newline="\n") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": generated_at,
        "input_manifest": str(args.ingestion_manifest),
        "input_records": len(manifest_records),
        "sources_processed": len(source_summaries),
        "total_chunks": len(chunks),
        "output_chunks": str(args.output_chunks),
        "chunk_strategy": CHUNK_STRATEGY_ID,
        "source_audit_gate": {
            "available": bool(source_audit),
            "block_chunk_generation": source_audit.get("block_chunk_generation"),
            "status_counts": source_audit.get("status_counts"),
        },
        "by_content_type": dict(Counter(chunk.get("content_type") for chunk in chunks)),
        "by_jurisdiction": dict(Counter(chunk.get("jurisdiction") for chunk in chunks)),
        "by_policy_domain": dict(Counter(chunk.get("policy_domain") for chunk in chunks)),
        "by_rag_ingestion_role": dict(Counter(chunk.get("rag_ingestion_role") for chunk in chunks)),
        "source_summaries": source_summaries,
        "length_stats": length_stats(chunks),
        "validation": validation,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        build_markdown_report(report),
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "input_records": report["input_records"],
                "sources_processed": report["sources_processed"],
                "total_chunks": report["total_chunks"],
                "by_content_type": report["by_content_type"],
                "output_chunks": report["output_chunks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def length_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("content_type") or "unknown")].append(int(chunk.get("text_chars") or 0))
    stats: dict[str, Any] = {}
    for key, values in grouped.items():
        sorted_values = sorted(values)
        stats[key] = {
            "count": len(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": round(sum(values) / len(values), 2),
            "p50": percentile(sorted_values, 0.5),
            "p90": percentile(sorted_values, 0.9),
        }
    return stats


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * ratio))))
    return values[index]


if __name__ == "__main__":
    main()
