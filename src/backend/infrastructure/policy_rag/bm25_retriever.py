"""Standard BM25 lexical retrieval over serialized policy nodes."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
from src.backend.infrastructure.policy_rag.node_store import (
    PolicyNodeRecord,
    PolicyNodeStore,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_NODES_PATH


DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
EXACT_CODE_MATCH_BOOST = 1000.0

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[\u4e00-\u9fff]+")
_ASCII_SPLIT_RE = re.compile(r"[-_./]+")
_CODE_KEY_RE = re.compile(r"(?:^|_)(?:code|id|编码)(?:$|_)", re.IGNORECASE)
_CODE_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{5,}$")


@dataclass(frozen=True, slots=True)
class _Bm25Document:
    record: PolicyNodeRecord
    term_counts: Counter[str]
    document_length: int


def tokenize_policy_text(text: str) -> list[str]:
    """Tokenize policy text deterministically for the local BM25 index."""

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(str(text).lower()):
        value = match.group(0)
        if value.isascii():
            tokens.append(value)
            tokens.extend(part for part in _ASCII_SPLIT_RE.split(value) if part)
            continue
        tokens.extend(_chinese_ngrams(value))
    return tokens


class Bm25PolicyIndex:
    """Okapi BM25 index with an inverted index and metadata filtering."""

    def __init__(
        self,
        node_store: PolicyNodeStore,
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")

        self._node_store = node_store
        self._k1 = float(k1)
        self._b = float(b)
        self._documents: list[_Bm25Document] = []
        self._document_frequency: Counter[str] = Counter()
        self._inverted_index: dict[str, list[int]] = {}

        for document_index, record in enumerate(node_store.records()):
            term_counts = Counter(tokenize_policy_text(_lexical_text(record)))
            document = _Bm25Document(
                record=record,
                term_counts=term_counts,
                document_length=sum(term_counts.values()),
            )
            self._documents.append(document)
            for term in term_counts:
                self._document_frequency[term] += 1
                self._inverted_index.setdefault(term, []).append(document_index)

        document_lengths = [
            document.document_length for document in self._documents
        ]
        self._avg_document_length = (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 1.0
        )
        self._idf = {
            term: math.log(
                1.0
                + (
                    len(self._documents) - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )
            for term, document_frequency in self._document_frequency.items()
        }

    @property
    def corpus_size(self) -> int:
        return len(self._documents)

    @property
    def average_document_length(self) -> float:
        return self._avg_document_length

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        query_terms = set(tokenize_policy_text(question))
        if not query_terms:
            return []

        candidate_indexes: set[int] = set()
        for term in query_terms:
            candidate_indexes.update(self._inverted_index.get(term, []))

        candidates: list[RetrievedPolicyCandidate] = []
        for document_index in candidate_indexes:
            document = self._documents[document_index]
            if not self._node_store.matches_filters(document.record, filters):
                continue
            exact_code_boost = _exact_code_match_boost(document.record, question)
            score = self._score(document, query_terms) + exact_code_boost
            if score <= 0:
                continue
            metadata = dict(document.record.metadata)
            metadata.update(
                {
                    "lexical_score": score,
                    "retrieval_strategy": "lexical_bm25_okapi",
                    "score_type": "lexical_score",
                    "bm25_k1": self._k1,
                    "bm25_b": self._b,
                    "exact_code_match": exact_code_boost > 0,
                    "exact_code_match_boost": exact_code_boost,
                }
            )
            candidates.append(
                RetrievedPolicyCandidate(
                    node_id=document.record.node_id,
                    text=document.record.text,
                    metadata=metadata,
                    faiss_score=score,
                )
            )

        candidates.sort(key=lambda item: item.faiss_score, reverse=True)
        return candidates[:fetch_k]

    def _score(
        self,
        document: _Bm25Document,
        query_terms: set[str],
    ) -> float:
        document_length = max(document.document_length, 1)
        average_length = max(self._avg_document_length, 1.0)
        length_normalization = (
            1.0
            - self._b
            + self._b * document_length / average_length
        )
        score = 0.0
        for term in query_terms:
            term_frequency = document.term_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            numerator = term_frequency * (self._k1 + 1.0)
            denominator = term_frequency + self._k1 * length_normalization
            score += self._idf[term] * numerator / denominator
        return score


class Bm25PolicyRetriever:
    """Loads the local policy node store and serves standard BM25 retrieval."""

    retrieval_mode_name = "lexical_bm25_okapi"

    def __init__(
        self,
        *,
        nodes_path: Path = DEFAULT_NODES_PATH,
        node_store: PolicyNodeStore | None = None,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self._node_store = node_store or PolicyNodeStore(nodes_path)
        self._index = Bm25PolicyIndex(self._node_store, k1=k1, b=b)

    @property
    def node_count(self) -> int:
        return self._node_store.node_count

    @property
    def index(self) -> Bm25PolicyIndex:
        return self._index

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        return self._index.retrieve(
            question=question,
            filters=filters,
            fetch_k=fetch_k,
        )


def _lexical_text(record: PolicyNodeRecord) -> str:
    metadata = record.metadata
    pieces = [
        record.text,
        str(metadata.get("title") or ""),
        str(metadata.get("section_heading") or ""),
    ]
    return "\n".join(piece for piece in pieces if piece)


def _exact_code_match_boost(record: PolicyNodeRecord, question: str) -> float:
    normalized_question = _normalize_code_text(question)
    if not normalized_question:
        return 0.0
    for key, value in record.metadata.items():
        if not _CODE_KEY_RE.search(str(key)):
            continue
        code = _normalize_code_text(str(value or ""))
        if len(code) < 6 or not _CODE_VALUE_RE.match(str(value or "").strip()):
            continue
        if code in normalized_question:
            return EXACT_CODE_MATCH_BOOST
    return 0.0


def _normalize_code_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value or "")).lower()


def _chinese_ngrams(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    if len(text) <= 12:
        tokens.append(text)
    for size in (2, 3, 4):
        if len(text) < size:
            continue
        tokens.extend(
            text[index : index + size]
            for index in range(len(text) - size + 1)
        )
    if len(text) == 1:
        tokens.append(text)
    return tokens
