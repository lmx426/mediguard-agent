"""Manifest models for the policy RAG source corpus.

The manifest records provenance for raw pages and downloaded attachments. It is
not a reimbursement rule engine and must not be used to produce final audit
decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_serializer


Jurisdiction = Literal["national", "beijing", "shanghai", "unknown"]
SourceStatus = Literal["active", "expired", "replaced", "unknown"]
DocType = Literal[
    "policy",
    "policy_interpretation",
    "catalog",
    "code_database",
    "service_guide",
    "faq",
    "typical_case",
    "price_table",
    "notice",
    "other",
]
LegalWeight = Literal[
    "binding",
    "catalog",
    "service_guide",
    "reference_only",
    "unknown",
]


class PolicyAttachmentRecord(BaseModel):
    """Downloaded source attachment tracked without parsing in stage 1."""

    url: HttpUrl
    filename: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    content_type: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    fetched_at: datetime

    @field_serializer("url")
    def serialize_url(self, value: HttpUrl) -> str:
        return str(value)

    @field_serializer("fetched_at")
    def serialize_fetched_at(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat()


class PolicySourceManifestRecord(BaseModel):
    """One source-page record for the policy corpus."""

    source_id: str = Field(min_length=3)
    title: str = Field(min_length=1)
    jurisdiction: Jurisdiction
    policy_domain: str = Field(min_length=1)
    url: HttpUrl
    issuing_authority: str | None = None
    publish_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    status: SourceStatus = "unknown"
    version_note: str = Field(
        default="Dates and replacement relation require deterministic extraction or manual review.",
        min_length=1,
    )
    supersedes: list[str] = Field(default_factory=list)
    doc_type: DocType = "other"
    legal_weight: LegalWeight = "unknown"
    case_tags: list[str] = Field(default_factory=list)
    raw_path: str = Field(min_length=1)
    clean_path: str | None = None
    sha256: str = Field(min_length=64, max_length=64)
    fetched_at: datetime
    attachments: list[PolicyAttachmentRecord] = Field(default_factory=list)
    crawl_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("url")
    def serialize_url(self, value: HttpUrl) -> str:
        return str(value)

    @field_serializer("fetched_at")
    def serialize_fetched_at(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[PolicySourceManifestRecord]:
    """Load and validate a JSONL manifest file."""

    records: list[PolicySourceManifestRecord] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(
                    PolicySourceManifestRecord.model_validate_json(stripped)
                )
            except ValidationError as exc:
                raise ValueError(f"Invalid manifest record at line {line_no}: {exc}") from exc
    return records


def append_manifest_record(path: Path, record: PolicySourceManifestRecord) -> None:
    """Append one manifest record as compact UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json")
    with path.open("a", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        file_obj.write("\n")


def write_manifest_records(
    path: Path, records: list[PolicySourceManifestRecord]
) -> None:
    """Rewrite a JSONL manifest with compact UTF-8 records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            payload = record.model_dump(mode="json")
            file_obj.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            file_obj.write("\n")


def upsert_manifest_record(path: Path, record: PolicySourceManifestRecord) -> None:
    """Insert or replace a record by source ID or canonical URL."""

    records = load_manifest(path)
    canonical_url = str(record.url).rstrip("/")
    replaced = False
    next_records: list[PolicySourceManifestRecord] = []

    for existing in records:
        same_source = existing.source_id == record.source_id
        same_url = str(existing.url).rstrip("/") == canonical_url
        if same_source or same_url:
            if not replaced:
                next_records.append(record)
                replaced = True
            continue
        next_records.append(existing)

    if not replaced:
        next_records.append(record)

    write_manifest_records(path, next_records)


def validate_manifest_file(path: Path) -> tuple[int, list[str]]:
    """Return record count and duplicate-source warnings for a manifest file."""

    records = load_manifest(path)
    seen: set[str] = set()
    warnings: list[str] = []
    for record in records:
        if record.source_id in seen:
            warnings.append(f"Duplicate source_id: {record.source_id}")
        seen.add(record.source_id)
    return len(records), warnings
