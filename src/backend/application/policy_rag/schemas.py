"""Schemas for policy RAG evidence retrieval.

The application layer owns the business boundary: this module describes policy
evidence search requests and responses, not reimbursement decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.backend.domain.policy_rag_taxonomy import (
    CONTENT_TYPES,
    JURISDICTIONS,
    POLICY_DOMAINS,
)


DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 40
DEFAULT_MULTI_GROUP_FETCH_K = 12
DEFAULT_BRANCH_POOL_SIZE = 40
DEFAULT_BRANCH_FUSION_LIMIT = 30
DEFAULT_FINAL_CONTEXT_CAP = 12
MAX_TOP_K = 20
MAX_FETCH_K = 200

FILTER_STRATEGY_SINGLE = "single"
FILTER_STRATEGY_DUAL_RRF = "dual_rrf"
FILTER_STRATEGIES = {FILTER_STRATEGY_SINGLE, FILTER_STRATEGY_DUAL_RRF}

CONTENT_TYPE_ALIASES = {
    "policy": "policy_text",
    "policy_file": "policy_text",
    "policy_document": "policy_text",
    "policy_interpretation": "policy_text",
    "policy_pdf": "policy_text",
    "document_markdown": "policy_text",
    "application/pdf": "policy_text",
    "政策": "policy_text",
    "政策文件": "policy_text",
    "政策正文": "policy_text",
    "政策文本": "policy_text",
    "table": "table_row",
    "table_row": "table_row",
    "表格": "table_row",
    "表格行": "table_row",
    "目录表": "table_row",
    "结构化表格": "table_row",
}

MODE = "policy_rag_mcp_v1"
RETRIEVAL_STRATEGY_DENSE = "dense"
RETRIEVAL_STRATEGY_HYBRID = "hybrid"
DEFAULT_RETRIEVAL_STRATEGY = RETRIEVAL_STRATEGY_HYBRID
RETRIEVAL_PROFILE_DENSE = "dense_faiss_bge_m3_v1"
RETRIEVAL_PROFILE_HYBRID = "hybrid_group_aware_strict_v3_bge_m3_bm25_rrf_v1"
RETRIEVAL_PROFILES = {
    RETRIEVAL_STRATEGY_DENSE: RETRIEVAL_PROFILE_DENSE,
    RETRIEVAL_STRATEGY_HYBRID: RETRIEVAL_PROFILE_HYBRID,
}
RETRIEVAL_MODE_WITH_RERANK = "dense_faiss_bge_m3_with_bge_reranker_v2_m3"
RETRIEVAL_MODE_FAISS_ONLY = "dense_faiss_bge_m3"
RETRIEVAL_MODE_GROUP_AWARE = "group_aware_dense_faiss_bge_m3"
RETRIEVAL_MODE_GROUP_AWARE_STRICT_V3 = "group_aware_strict_v3_dense_faiss_bge_m3"
RETRIEVAL_MODE_GROUP_AWARE_WITH_RERANK = (
    "group_aware_dense_faiss_bge_m3_with_conditional_bge_reranker_v2_m3"
)

DEFAULT_LIMITS = [
    "Only policy evidence retrieval results are returned.",
    "This service does not read L1/L2 case facts.",
    "This service does not generate final audit conclusions.",
    "This service does not decide refusal, payment, punishment, or fraud.",
    "This service does not modify case state, rule hits, risk scores, or human audit decisions.",
]


class PolicyRagRequestError(ValueError):
    """Raised when a policy RAG search request violates the v1 contract."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


def normalize_retrieval_strategy(value: object) -> str:
    strategy = str(value or DEFAULT_RETRIEVAL_STRATEGY).strip().lower()
    if strategy not in RETRIEVAL_PROFILES:
        return DEFAULT_RETRIEVAL_STRATEGY
    return strategy


def retrieval_profile_for_strategy(value: object) -> str:
    return RETRIEVAL_PROFILES[normalize_retrieval_strategy(value)]


@dataclass(slots=True)
class PolicyRagFilters:
    """Registered metadata filters supplied by the caller."""

    jurisdiction: list[str] = field(default_factory=list)
    policy_domain: list[str] = field(default_factory=list)
    content_type: list[str] = field(default_factory=list)
    can_cite_as_policy_basis: bool | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "PolicyRagFilters":
        if payload is None:
            raise PolicyRagRequestError(
                "filters_required",
                "policy_rag_search requires explicit filters.",
            )
        if not isinstance(payload, dict):
            raise PolicyRagRequestError(
                "invalid_filters",
                "filters must be a JSON object.",
            )
        return cls(
            jurisdiction=_string_list(payload.get("jurisdiction")),
            policy_domain=_string_list(payload.get("policy_domain")),
            content_type=_string_list(payload.get("content_type")),
            can_cite_as_policy_basis=_optional_bool(
                payload.get("can_cite_as_policy_basis")
            ),
        )

    def normalized(self, *, preserve_none_cite: bool = False) -> "PolicyRagFilters":
        content_type = _unique_nonempty(
            _normalize_content_type(value) for value in self.content_type
        )
        return PolicyRagFilters(
            jurisdiction=_unique_nonempty(self.jurisdiction),
            policy_domain=_unique_nonempty(self.policy_domain),
            content_type=content_type or ["policy_text", "table_row"],
            can_cite_as_policy_basis=(
                None
                if preserve_none_cite and self.can_cite_as_policy_basis is None
                else True
                if self.can_cite_as_policy_basis is None
                else self.can_cite_as_policy_basis
            ),
        )

    def validate(self, *, allow_broad: bool = False) -> None:
        self._validate_values("jurisdiction", self.jurisdiction, JURISDICTIONS)
        self._validate_values("policy_domain", self.policy_domain, POLICY_DOMAINS)
        self._validate_values("content_type", self.content_type, CONTENT_TYPES)
        if not allow_broad and not self.jurisdiction and not self.policy_domain:
            raise PolicyRagRequestError(
                "filters_too_broad",
                "filters must include jurisdiction or policy_domain.",
            )

    @staticmethod
    def _validate_values(
        field_name: str,
        values: list[str],
        allowed: tuple[str, ...],
    ) -> None:
        unknown = [value for value in values if value not in allowed]
        if unknown:
            raise PolicyRagRequestError(
                "unregistered_filter_value",
                f"{field_name} contains unregistered values: {unknown}.",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "policy_domain": self.policy_domain,
            "content_type": self.content_type,
            "can_cite_as_policy_basis": self.can_cite_as_policy_basis,
        }


@dataclass(slots=True)
class PolicyRagSearchRequest:
    question: str
    filters: PolicyRagFilters
    recall_filters: PolicyRagFilters | None = None
    filter_strategy: str = FILTER_STRATEGY_SINGLE
    adaptive_top_k: bool = False
    allow_broad_filters: bool = False
    top_k: int = DEFAULT_TOP_K
    fetch_k: int = DEFAULT_FETCH_K
    rerank: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PolicyRagSearchRequest":
        question = str(payload.get("question") or "").strip()
        if not question:
            raise PolicyRagRequestError(
                "question_required",
                "policy_rag_search requires a non-empty question.",
            )
        filter_strategy = _filter_strategy(payload.get("filter_strategy"))
        filters = PolicyRagFilters.from_mapping(payload.get("filters")).normalized()
        allow_broad_filters = _bool(payload.get("allow_broad_filters"), default=False)
        filters.validate(allow_broad=allow_broad_filters)
        recall_filters: PolicyRagFilters | None = None
        if payload.get("recall_filters") is not None:
            recall_filters = PolicyRagFilters.from_mapping(
                payload.get("recall_filters")
            ).normalized(preserve_none_cite=True)
            recall_filters.validate(allow_broad=True)
        if filter_strategy == FILTER_STRATEGY_DUAL_RRF and recall_filters is None:
            raise PolicyRagRequestError(
                "recall_filters_required",
                "dual_rrf requires explicit recall_filters.",
            )
        return cls(
            question=question,
            filters=filters,
            recall_filters=recall_filters,
            filter_strategy=filter_strategy,
            adaptive_top_k=_bool(payload.get("adaptive_top_k"), default=False),
            allow_broad_filters=allow_broad_filters,
            top_k=_positive_int(payload.get("top_k"), DEFAULT_TOP_K, "top_k"),
            fetch_k=_positive_int(payload.get("fetch_k"), DEFAULT_FETCH_K, "fetch_k"),
            rerank=_bool(payload.get("rerank"), default=False),
        ).normalized()

    def normalized(self) -> "PolicyRagSearchRequest":
        top_k = min(max(1, int(self.top_k)), MAX_TOP_K)
        fetch_k = min(max(top_k, int(self.fetch_k)), MAX_FETCH_K)
        return PolicyRagSearchRequest(
            question=self.question,
            filters=self.filters,
            recall_filters=self.recall_filters,
            filter_strategy=self.filter_strategy,
            adaptive_top_k=self.adaptive_top_k,
            allow_broad_filters=self.allow_broad_filters,
            top_k=top_k,
            fetch_k=fetch_k,
            rerank=self.rerank,
        )


@dataclass(frozen=True, slots=True)
class RetrievedPolicyCandidate:
    node_id: str
    text: str
    metadata: dict[str, Any]
    faiss_score: float
    rerank_score: float | None = None
    retrieval_groups: tuple[str, ...] = ()

    def score(self, *, rerank_applied: bool) -> float:
        if rerank_applied and self.rerank_score is not None:
            return self.rerank_score
        return self.faiss_score


def invalid_request_response(error: PolicyRagRequestError) -> dict[str, Any]:
    return {
        "status": "invalid_request",
        "error": error.error,
        "message": error.message,
        "limits": DEFAULT_LIMITS,
    }


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise PolicyRagRequestError(
            "invalid_filter_value",
            "filter values must be strings or string arrays.",
        )
    values: list[str] = []
    for item in value:
        item_text = str(item or "").strip()
        if item_text:
            values.append(item_text)
    return values


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _normalize_content_type(value: str) -> str:
    text = str(value or "").strip()
    return CONTENT_TYPE_ALIASES.get(text, CONTENT_TYPE_ALIASES.get(text.lower(), text.lower()))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _bool(value, default=False)


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise PolicyRagRequestError(
        "invalid_boolean",
        "boolean fields must be true or false.",
    )


def _positive_int(value: Any, default: int, field_name: str) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyRagRequestError(
            "invalid_integer",
            f"{field_name} must be a positive integer.",
        ) from exc
    if parsed <= 0:
        raise PolicyRagRequestError(
            "invalid_integer",
            f"{field_name} must be a positive integer.",
        )
    return parsed


def _filter_strategy(value: Any) -> str:
    strategy = str(value or FILTER_STRATEGY_SINGLE).strip().lower()
    if strategy not in FILTER_STRATEGIES:
        raise PolicyRagRequestError(
            "invalid_filter_strategy",
            f"filter_strategy must be one of {sorted(FILTER_STRATEGIES)}.",
        )
    return strategy
