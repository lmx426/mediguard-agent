"""Structured policy-version metadata search for the Policy RAG MCP."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_VERSION_INDEX_PATH,
    DEFAULT_VERSION_REVIEW_INDEX_PATH,
)


@dataclass(frozen=True, slots=True)
class PolicyVersionRecord:
    doc_id: str
    source_id: str
    title: str
    jurisdiction: str
    policy_domain: str
    doc_type: str
    legal_weight: str
    source_url: str
    document_no: str
    publish_date: str
    effective_date: str
    expiry_date: str
    status: str
    official_status_text: str
    needs_manual_review: bool
    rag_path: str | None
    review_note: str | None

    @property
    def search_text(self) -> str:
        return " ".join(
            value
            for value in (
                self.title,
                self.document_no,
                self.jurisdiction,
                self.policy_domain,
                self.status,
                self.official_status_text,
                self.publish_date,
                self.effective_date,
                self.expiry_date,
            )
            if value
        )

    def to_evidence(self, *, rank: int, score: float) -> dict[str, Any]:
        return {
            "rank": rank,
            "score": score,
            "score_type": "version_metadata_match",
            "node_id": f"version::{self.source_id}",
            "text": self.search_text,
            "title": self.title,
            "source_url": self.source_url,
            "source_id": self.source_id,
            "jurisdiction": self.jurisdiction,
            "policy_domain": self.policy_domain,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "document_no": self.document_no,
            "publish_date": self.publish_date,
            "effective_date": self.effective_date,
            "expiry_date": self.expiry_date,
            "status": self.status,
            "official_status_text": self.official_status_text,
            "needs_manual_review": self.needs_manual_review,
            "rag_path": self.rag_path,
            "review_note": self.review_note,
            "evidence_role": "version_metadata",
            "can_cite_as_policy_basis": False,
        }


class PolicyVersionStore:
    """Loads normalized version metadata and supports structured filtering."""

    def __init__(
        self,
        *,
        index_path: Path = DEFAULT_VERSION_INDEX_PATH,
        review_index_path: Path = DEFAULT_VERSION_REVIEW_INDEX_PATH,
    ) -> None:
        self._index_path = index_path
        self._review_index_path = review_index_path
        self._records = self._load_records(index_path, review_index_path)
        if not self._records:
            raise RuntimeError("No policy version metadata records were loaded")

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def index_path(self) -> Path:
        return self._index_path

    def search(
        self,
        *,
        question: str,
        filters: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        records = [
            record
            for record in self._records
            if self._matches_filters(record, filters)
        ]
        scored = [
            (self._score(question, record), record)
            for record in records
        ]
        scored.sort(
            key=lambda item: (
                item[0],
                _date_sort_value(item[1].effective_date),
            ),
            reverse=True,
        )
        return [
            record.to_evidence(rank=index, score=score)
            for index, (score, record) in enumerate(scored[:top_k], start=1)
        ]

    def _matches_filters(
        self,
        record: PolicyVersionRecord,
        filters: dict[str, Any],
    ) -> bool:
        if not _matches_list(record.jurisdiction, filters.get("jurisdiction")):
            return False
        if not _matches_list(record.policy_domain, filters.get("policy_domain")):
            return False
        if not _matches_list(record.source_id, filters.get("source_id")):
            return False
        status_values = filters.get("status")
        if status_values:
            normalized_status = _normalize_status_values(status_values)
            if record.status not in normalized_status:
                return False
        valid_on = str(filters.get("valid_on") or "").strip()
        if valid_on and not _is_valid_on(record, valid_on):
            return False
        return True

    @staticmethod
    def _score(question: str, record: PolicyVersionRecord) -> float:
        query_terms = _terms(question)
        if not query_terms:
            return 0.0
        haystack = record.search_text
        hits = sum(1 for term in query_terms if term in haystack)
        return hits / len(query_terms)

    @staticmethod
    def _load_records(
        index_path: Path,
        review_index_path: Path,
    ) -> list[PolicyVersionRecord]:
        primary = _read_jsonl(index_path)
        review = {
            str(item.get("source_id") or ""): item
            for item in _read_jsonl(review_index_path)
            if item.get("source_id")
        }
        merged: dict[str, dict[str, Any]] = {}
        for item in primary:
            source_id = str(item.get("source_id") or "").strip()
            if source_id:
                merged[source_id] = dict(item)
        for source_id, review_item in review.items():
            if source_id not in merged:
                merged[source_id] = dict(review_item)
            else:
                for key, value in review_item.items():
                    if merged[source_id].get(key) in (None, "") and value not in (None, ""):
                        merged[source_id][key] = value
        records: list[PolicyVersionRecord] = []
        for item in merged.values():
            records.append(
                PolicyVersionRecord(
                    doc_id=str(item.get("doc_id") or ""),
                    source_id=str(item.get("source_id") or ""),
                    title=str(item.get("title") or ""),
                    jurisdiction=str(item.get("jurisdiction") or ""),
                    policy_domain=str(item.get("policy_domain") or ""),
                    doc_type=str(item.get("doc_type") or ""),
                    legal_weight=str(item.get("legal_weight") or ""),
                    source_url=str(item.get("source_url") or ""),
                    document_no=str(item.get("document_no") or ""),
                    publish_date=str(item.get("publish_date") or ""),
                    effective_date=str(item.get("effective_date") or ""),
                    expiry_date=str(item.get("expiry_date") or ""),
                    status=_normalize_status(
                        item.get("status"),
                        item.get("status_inferred"),
                        item.get("official_status_text"),
                    ),
                    official_status_text=str(
                        item.get("official_status_text") or ""
                    ),
                    needs_manual_review=bool(
                        item.get("needs_manual_review")
                        or item.get("status_inferred") == "unknown_needs_manual_review"
                    ),
                    rag_path=(
                        str(item.get("rag_path"))
                        if item.get("rag_path")
                        else None
                    ),
                    review_note=(
                        str(item.get("review_note"))
                        if item.get("review_note")
                        else None
                    ),
                )
            )
        return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _matches_list(value: str, filter_value: Any) -> bool:
    if not filter_value:
        return True
    values = [str(item) for item in filter_value] if isinstance(filter_value, list) else [str(filter_value)]
    return value in values


def _normalize_status_values(value: Any) -> set[str]:
    values = value if isinstance(value, list) else [value]
    return {_normalize_status(item, None, None) for item in values}


def _normalize_status(
    status: Any,
    inferred: Any,
    official: Any,
) -> str:
    raw_values = [
        str(value or "").strip().lower()
        for value in (status, inferred, official)
    ]
    for value in raw_values:
        if value in {"active", "有效", "current"}:
            return "active"
        if value in {"expired", "失效", "废止"}:
            return "expired"
        if value in {"replaced", "替代"}:
            return "replaced"
    return "unknown"


def _is_valid_on(record: PolicyVersionRecord, valid_on: str) -> bool:
    try:
        target = date.fromisoformat(valid_on)
    except ValueError:
        return False
    if not record.effective_date:
        return False
    try:
        effective = date.fromisoformat(record.effective_date)
    except ValueError:
        return False
    if effective > target:
        return False
    if record.expiry_date:
        try:
            if date.fromisoformat(record.expiry_date) < target:
                return False
        except ValueError:
            return False
    return record.status not in {"expired", "replaced"}


def _terms(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text or "").strip().lower()
    if not normalized:
        return []
    if re.search(r"[\u4e00-\u9fff]", normalized):
        return [normalized[index : index + 2] for index in range(max(1, len(normalized) - 1))]
    return [item for item in re.split(r"[^a-z0-9_]+", normalized) if item]


def _date_sort_value(value: str) -> str:
    return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or "") else ""
