from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.backend.application.policy_rag.search_policy_evidence_uc import (
    SearchPolicyEvidenceUseCase,
)
from src.backend.application.policy_rag.schemas import (
    PolicyRagFilters,
    RetrievedPolicyCandidate,
)
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyIndex
from src.backend.infrastructure.policy_rag.hybrid_retriever import HybridPolicyRetriever
from src.backend.infrastructure.policy_rag.node_store import PolicyNodeRecord
from src.backend.infrastructure.policy_rag.raw_hybrid_retriever import (
    RawHybridPolicyRetriever,
)


@dataclass
class _FakeRetriever:
    responses: dict[tuple[str, ...], list[RetrievedPolicyCandidate]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        domains = tuple(filters.get("policy_domain") or [])
        self.calls.append(
            {
                "question": question,
                "filters": filters,
                "fetch_k": fetch_k,
            }
        )
        return list(self.responses.get(domains, []))


class _FakeReranker:
    available = True
    unavailable_reason = None
    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(
        self,
        *,
        question: str,
        candidates: list[RetrievedPolicyCandidate],
    ) -> list[RetrievedPolicyCandidate]:
        self.calls.append([candidate.node_id for candidate in candidates])
        return sorted(
            candidates,
            key=lambda candidate: candidate.faiss_score,
            reverse=True,
        )


def _candidate(
    node_id: str,
    *,
    source_id: str,
    domain: str,
    score: float,
    chunk_index: int,
    text: str | None = None,
    doc_type: str = "policy",
) -> RetrievedPolicyCandidate:
    return RetrievedPolicyCandidate(
        node_id=node_id,
        text=text or f"{domain} evidence",
        metadata={
            "source_id": source_id,
            "doc_id": source_id,
            "policy_domain": domain,
            "jurisdiction": "beijing",
            "content_type": "policy_text",
            "doc_type": doc_type,
            "can_cite_as_policy_basis": True,
            "source_url": "https://example.test/policy",
            "chunk_index": chunk_index,
        },
        faiss_score=score,
    )


def test_multi_group_retrieval_keeps_both_evidence_groups_and_caps_source() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog",): [
                _candidate(
                    "drug-1",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.98,
                    chunk_index=1,
                ),
                _candidate(
                    "drug-2",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.97,
                    chunk_index=2,
                ),
                _candidate(
                    "drug-3",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.96,
                    chunk_index=3,
                ),
            ],
            ("medical_service_price",): [
                _candidate(
                    "price-1",
                    source_id="price-source",
                    domain="medical_service_price",
                    score=0.80,
                    chunk_index=1,
                ),
            ],
        }
    )
    reranker = _FakeReranker()
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
    )

    response = use_case.search(
        {
            "question": "阿莫西林和胸部CT价格如何核验？",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog", "medical_service_price"],
            },
            "top_k": 3,
            "fetch_k": 12,
            "rerank": True,
        }
    )

    assert response["retrieval_strategy"] == "hybrid"
    assert response["retrieval_profile"] == "hybrid_group_aware_strict_v3_bge_m3_bm25_rrf_v1"
    assert response["retrieval_plan"]["profile"] == "strict_v3"
    assert response["retrieval_plan"]["strategy"] == "multi_group_strict_v3_hybrid"
    assert response["rerank"]["triggered"] is True
    assert reranker.calls == [["price-1", "drug-1", "drug-2", "drug-3"]]
    evidence = response["evidence"]
    assert {item["retrieval_groups"][0] for item in evidence} == {
        "drug_catalog",
        "medical_service_price",
    }
    assert sum(item["source_id"] == "drug-source" for item in evidence) <= 2


def test_multi_group_selection_keeps_group_coverage_and_allows_global_fill() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog",): [
                _candidate(
                    f"drug-{index}",
                    source_id=f"drug-source-{index}",
                    domain="drug_catalog",
                    score=0.99 - index * 0.01,
                    chunk_index=index,
                )
                for index in range(1, 5)
            ],
            ("medical_service_price",): [
                _candidate(
                    f"price-{index}",
                    source_id=f"price-source-{index}",
                    domain="medical_service_price",
                    score=0.70 - index * 0.01,
                    chunk_index=index,
                )
                for index in range(1, 4)
            ],
        }
    )
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=None,
    )

    response = use_case.search(
        {
            "question": "Check drug catalog and CT price evidence.",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog", "medical_service_price"],
            },
            "top_k": 4,
            "fetch_k": 12,
            "rerank": False,
        }
    )

    assert sum(
        "drug_catalog" in item["retrieval_groups"] for item in response["evidence"]
    ) >= 2
    assert sum(
        "medical_service_price" in item["retrieval_groups"]
        for item in response["evidence"]
    ) >= 1


def test_strict_v3_does_not_split_unclear_multi_domain_query() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog", "medical_service_price"): [
                _candidate(
                    "combined-1",
                    source_id="combined-source",
                    domain="drug_catalog",
                    score=0.90,
                    chunk_index=1,
                )
            ],
            ("drug_catalog",): [
                _candidate(
                    "drug-1",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.99,
                    chunk_index=1,
                )
            ],
            ("medical_service_price",): [
                _candidate(
                    "price-1",
                    source_id="price-source",
                    domain="medical_service_price",
                    score=0.98,
                    chunk_index=1,
                )
            ],
        }
    )
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=None,
    )

    response = use_case.search(
        {
            "question": "医保政策资料查询",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog", "medical_service_price"],
            },
            "top_k": 3,
            "fetch_k": 12,
            "rerank": False,
        }
    )

    assert response["retrieval_plan"]["strategy"] == "single_group_strict_v3_hybrid"
    assert [call["filters"]["policy_domain"] for call in retriever.calls] == [
        ["drug_catalog", "medical_service_price"]
    ]
    assert response["evidence"][0]["node_id"] == "combined-1"


def test_single_group_uses_dense_score_without_v3_anchor_boost() -> None:
    retriever = _FakeRetriever(
        {
            ("medical_service_price",): [
                _candidate(
                    "generic-price",
                    source_id="price-source",
                    domain="medical_service_price",
                    score=0.99,
                    chunk_index=1,
                    text="generic outpatient service evidence",
                    doc_type="policy",
                ),
                _candidate(
                    "ct-price",
                    source_id="price-source",
                    domain="medical_service_price",
                    score=0.70,
                    chunk_index=2,
                    text="CT price table evidence with unit and project meaning",
                    doc_type="price_table",
                ),
            ]
        }
    )
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=None,
    )

    response = use_case.search(
        {
            "question": "How should CT price and billing unit be checked?",
            "filters": {
                "jurisdiction": ["shanghai"],
                "policy_domain": ["medical_service_price"],
            },
            "top_k": 1,
            "fetch_k": 12,
            "rerank": False,
        }
    )

    assert response["evidence"][0]["node_id"] == "generic-price"


def test_hybrid_retrieval_uses_lexical_match_to_rescue_exact_candidate() -> None:
    node_store = _FakeNodeStore(
        [
            PolicyNodeRecord(
                node_id="generic-price",
                text="普通门诊收费材料说明。",
                metadata=_metadata(
                    source_id="price-source",
                    domain="medical_service_price",
                    chunk_index=1,
                ),
            ),
            PolicyNodeRecord(
                node_id="ct-price",
                text="胸部 CT 平扫项目应核验计价单位和项目内涵。",
                metadata=_metadata(
                    source_id="price-source",
                    domain="medical_service_price",
                    chunk_index=2,
                ),
            ),
        ]
    )
    dense_retriever = _FakeDenseRetriever(
        [
            _candidate(
                "generic-price",
                source_id="price-source",
                domain="medical_service_price",
                score=0.99,
                chunk_index=1,
                text="普通门诊收费材料说明。",
            ),
            _candidate(
                "ct-price",
                source_id="price-source",
                domain="medical_service_price",
                score=0.50,
                chunk_index=2,
                text="胸部 CT 平扫项目应核验计价单位和项目内涵。",
            ),
        ]
    )
    retriever = HybridPolicyRetriever(
        dense_retriever=dense_retriever,
        node_store=node_store,
    )

    candidates = retriever.retrieve(
        question="胸部 CT 平扫计价单位如何核验？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["medical_service_price"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
        fetch_k=2,
    )

    assert candidates[0].node_id == "ct-price"
    assert candidates[0].metadata["retrieval_strategy"] == "hybrid_dense_lexical_rrf"
    assert candidates[0].metadata["dense_score"] == 0.50
    assert candidates[0].metadata["lexical_score"] > 0
    assert candidates[0].metadata["hybrid_score"] == candidates[0].faiss_score


def test_standard_bm25_applies_metadata_filters_and_exposes_parameters() -> None:
    node_store = _FakeNodeStore(
        [
            PolicyNodeRecord(
                node_id="drug-1",
                text="阿莫西林属于医保药品目录药品。",
                metadata=_metadata(
                    source_id="drug-source",
                    domain="drug_catalog",
                    chunk_index=1,
                ),
            ),
            PolicyNodeRecord(
                node_id="price-1",
                text="胸部 CT 平扫应核验计价单位。",
                metadata=_metadata(
                    source_id="price-source",
                    domain="medical_service_price",
                    chunk_index=1,
                ),
            ),
        ]
    )
    index = Bm25PolicyIndex(node_store)

    candidates = index.retrieve(
        question="阿莫西林医保药品目录",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
        fetch_k=5,
    )

    assert [candidate.node_id for candidate in candidates] == ["drug-1"]
    assert candidates[0].metadata["retrieval_strategy"] == "lexical_bm25_okapi"
    assert candidates[0].metadata["bm25_k1"] == 1.2
    assert candidates[0].metadata["bm25_b"] == 0.75


def test_raw_hybrid_does_not_apply_source_cap() -> None:
    candidates = [
        _candidate(
            f"chunk-{index}",
            source_id="same-source",
            domain="drug_catalog",
            score=0.99 - index * 0.01,
            chunk_index=index,
        )
        for index in range(1, 4)
    ]
    lexical = _FakeLexicalRetriever(candidates)
    dense = _FakeDenseRetriever(candidates)
    retriever = RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        question="药品目录",
        filters={"policy_domain": ["drug_catalog"]},
        fetch_k=3,
    )

    assert len(results) == 3
    assert {item.metadata["source_id"] for item in results} == {"same-source"}
    assert all(item.metadata["retrieval_strategy"] == "raw_hybrid_dense_bm25_rrf" for item in results)


@dataclass
class _FakeDenseRetriever:
    candidates: list[RetrievedPolicyCandidate]

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        return self.candidates[:fetch_k]


@dataclass
class _FakeLexicalRetriever:
    candidates: list[RetrievedPolicyCandidate]

    def retrieve(
        self,
        *,
        question: str,
        filters: dict[str, object],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        return self.candidates[:fetch_k]


class _FakeNodeStore:
    def __init__(self, records: list[PolicyNodeRecord]) -> None:
        self._records = records

    @property
    def node_count(self) -> int:
        return len(self._records)

    def records(self) -> tuple[PolicyNodeRecord, ...]:
        return tuple(self._records)

    def matches_filters(
        self,
        record: PolicyNodeRecord,
        filters: dict[str, object],
    ) -> bool:
        metadata = record.metadata
        jurisdictions = {str(value) for value in filters.get("jurisdiction") or []}
        domains = {str(value) for value in filters.get("policy_domain") or []}
        content_types = {str(value) for value in filters.get("content_type") or []}
        if jurisdictions and metadata.get("jurisdiction") not in jurisdictions:
            return False
        if domains and metadata.get("policy_domain") not in domains:
            return False
        if content_types and metadata.get("content_type") not in content_types:
            return False
        cite_filter = filters.get("can_cite_as_policy_basis")
        return cite_filter is None or metadata.get(
            "can_cite_as_policy_basis"
        ) == cite_filter


def _metadata(
    *,
    source_id: str,
    domain: str,
    chunk_index: int,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "doc_id": source_id,
        "policy_domain": domain,
        "jurisdiction": "beijing",
        "content_type": "policy_text",
        "doc_type": "policy",
        "can_cite_as_policy_basis": True,
        "source_url": "https://example.test/policy",
        "chunk_index": chunk_index,
    }
def test_simple_group_does_not_call_reranker_when_dense_results_are_confident() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog",): [
                _candidate(
                    "drug-1",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.99,
                    chunk_index=1,
                ),
                _candidate(
                    "drug-2",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.70,
                    chunk_index=2,
                ),
                _candidate(
                    "drug-3",
                    source_id="drug-source",
                    domain="drug_catalog",
                    score=0.60,
                    chunk_index=3,
                ),
            ]
        }
    )
    reranker = _FakeReranker()
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
    )

    response = use_case.search(
        {
            "question": "阿莫西林是否在医保药品目录？",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog"],
            },
            "top_k": 3,
            "fetch_k": 12,
            "rerank": True,
        }
    )

    assert response["rerank"]["triggered"] is False
    assert response["rerank"]["trigger_reason"] in {
        "candidate_count_not_above_top_k",
        "single_group_confident_dense_results",
    }
    assert response["result_count"] == 3
    assert reranker.calls == []


def test_medium_query_caps_cross_encoder_input_at_twelve_candidates() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog",): [
                _candidate(
                    f"drug-{index}",
                    source_id=f"drug-source-{index}",
                    domain="drug_catalog",
                    score=0.90 - index * 0.001,
                    chunk_index=index,
                )
                for index in range(20)
            ]
        }
    )
    reranker = _FakeReranker()
    use_case = SearchPolicyEvidenceUseCase(retriever=retriever, reranker=reranker)

    response = use_case.search(
        {
            "question": "药品目录中多个相近结果如何核验？",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog"],
            },
            "top_k": 5,
            "fetch_k": 24,
            "rerank": True,
        }
    )

    assert len(reranker.calls[0]) == 12
    assert response["rerank"]["candidate_limit"] == 12
    assert response["retrieval_diagnostics"]["rerank_input_count"] == 12


def test_complex_query_caps_cross_encoder_input_at_sixteen_candidates() -> None:
    retriever = _FakeRetriever(
        {
            ("remote_medical",): [
                _candidate(
                    f"remote-{index}",
                    source_id=f"remote-source-{index}",
                    domain="remote_medical",
                    score=0.90 - index * 0.001,
                    chunk_index=index,
                )
                for index in range(24)
            ]
        }
    )
    reranker = _FakeReranker()
    use_case = SearchPolicyEvidenceUseCase(retriever=retriever, reranker=reranker)

    response = use_case.search(
        {
            "question": "跨省异地就医待遇如何比较？",
            "filters": {
                "jurisdiction": ["national"],
                "policy_domain": ["remote_medical"],
            },
            "top_k": 5,
            "fetch_k": 40,
            "rerank": True,
        }
    )

    assert len(reranker.calls[0]) == 16
    assert response["rerank"]["candidate_limit"] == 16
    assert response["retrieval_diagnostics"]["rerank_input_count"] == 16


def test_duplicate_source_chunk_merges_group_labels() -> None:
    retriever = _FakeRetriever(
        {
            ("remote_medical", "remote_medical_manual_reimbursement"): [
                _candidate(
                    "shared",
                    source_id="same-source",
                    domain="remote_medical",
                    score=0.90,
                    chunk_index=1,
                )
            ],
            ("manual_reimbursement", "remote_medical_manual_reimbursement"): [
                _candidate(
                    "shared",
                    source_id="same-source",
                    domain="remote_medical",
                    score=0.91,
                    chunk_index=1,
                )
            ],
        }
    )
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=None,
    )

    response = use_case.search(
        {
            "question": "异地急诊手工报销如何处理？",
            "filters": {
                "jurisdiction": ["national", "beijing"],
                "policy_domain": [
                    "remote_medical",
                    "manual_reimbursement",
                    "remote_medical_manual_reimbursement",
                ],
            },
            "top_k": 5,
            "fetch_k": 12,
            "rerank": False,
        }
    )

    assert response["result_count"] == 1
    assert set(response["evidence"][0]["retrieval_groups"]) == {
        "remote_medical",
        "manual_reimbursement",
    }


def test_dual_rrf_recall_branch_rescues_candidate_blocked_by_strict_domain() -> None:
    retriever = _FakeRetriever(
        {
            ("benefit",): [
                _candidate(
                    "strict-benefit",
                    source_id="strict-source",
                    domain="benefit",
                    score=0.90,
                    chunk_index=1,
                )
            ],
            (): [
                _candidate(
                    "recall-manual",
                    source_id="recall-source",
                    domain="manual_reimbursement",
                    score=0.85,
                    chunk_index=1,
                )
            ],
        }
    )
    reranker = _FakeReranker()
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
    )

    response = use_case.search(
        {
            "question": "北京参保人手工报销需要哪些材料？",
            "filters": {
                "jurisdiction": ["national"],
                "policy_domain": ["benefit"],
                "content_type": ["policy_text"],
                "can_cite_as_policy_basis": True,
            },
            "recall_filters": {
                "jurisdiction": ["beijing", "national"],
                "policy_domain": [],
                "content_type": ["policy_text", "table_row"],
                "can_cite_as_policy_basis": True,
            },
            "filter_strategy": "dual_rrf",
            "adaptive_top_k": True,
            "top_k": 5,
            "fetch_k": 40,
            "rerank": True,
        }
    )

    assert response["status"] == "ok"
    assert response["top_k"] == 6
    assert response["rerank"]["trigger_reason"] == "dual_branch_fusion"
    assert {item["node_id"] for item in response["evidence"]} == {
        "strict-benefit",
        "recall-manual",
    }
    recalled = next(
        item for item in response["evidence"]
        if item["node_id"] == "recall-manual"
    )
    assert recalled["retrieval_branches"] == ["recall"]
    diagnostics = response["retrieval_diagnostics"]
    assert diagnostics["strict_only_count"] == 1
    assert diagnostics["recall_only_count"] == 1
    assert diagnostics["branch_fusion_limit"] == 30


def test_adaptive_top_k_uses_two_group_budget_and_two_per_group() -> None:
    retriever = _FakeRetriever(
        {
            ("drug_catalog",): [
                _candidate(
                    f"drug-adaptive-{index}",
                    source_id=f"drug-adaptive-source-{index}",
                    domain="drug_catalog",
                    score=0.99 - index * 0.01,
                    chunk_index=index,
                )
                for index in range(1, 5)
            ],
            ("medical_service_price",): [
                _candidate(
                    f"price-adaptive-{index}",
                    source_id=f"price-adaptive-source-{index}",
                    domain="medical_service_price",
                    score=0.89 - index * 0.01,
                    chunk_index=index,
                )
                for index in range(1, 5)
            ],
            (): [],
        }
    )
    use_case = SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=_FakeReranker(),
    )

    response = use_case.search(
        {
            "question": "同时核验阿莫西林目录和胸部CT项目价格。",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["drug_catalog", "medical_service_price"],
                "content_type": ["policy_text", "table_row"],
                "can_cite_as_policy_basis": True,
            },
            "recall_filters": {
                "jurisdiction": ["beijing", "national"],
                "policy_domain": [],
                "content_type": ["policy_text", "table_row"],
                "can_cite_as_policy_basis": True,
            },
            "filter_strategy": "dual_rrf",
            "adaptive_top_k": True,
            "fetch_k": 40,
            "rerank": True,
        }
    )

    assert response["retrieval_diagnostics"]["evidence_group_count"] == 2
    assert response["top_k"] == 8
    assert response["result_count"] == 8
    assert sum(
        "drug_catalog" in item["retrieval_groups"]
        for item in response["evidence"]
    ) >= 2
    assert sum(
        "medical_service_price" in item["retrieval_groups"]
        for item in response["evidence"]
    ) >= 2
    assert all(call["fetch_k"] == 12 for call in retriever.calls)


def test_final_candidate_gate_excludes_graph_hint_from_dual_branch() -> None:
    graph_hint = _candidate(
        "graph-hint",
        source_id="hint-source",
        domain="benefit",
        score=0.99,
        chunk_index=1,
    )
    graph_hint.metadata["policy_domain"] = "graph_hints"
    graph_hint.metadata["content_type"] = "graph_hint"
    graph_hint.metadata["can_cite_as_policy_basis"] = False
    retriever = _FakeRetriever(
        {
            ("benefit",): [],
            (): [graph_hint],
        }
    )
    use_case = SearchPolicyEvidenceUseCase(retriever=retriever, reranker=None)

    response = use_case.search(
        {
            "question": "北京门诊待遇政策是什么？",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["benefit"],
                "can_cite_as_policy_basis": True,
            },
            "recall_filters": {
                "jurisdiction": ["beijing", "national"],
                "policy_domain": [],
                "content_type": ["policy_text", "table_row"],
                "can_cite_as_policy_basis": True,
            },
            "filter_strategy": "dual_rrf",
            "adaptive_top_k": True,
            "rerank": True,
        }
    )

    assert response["result_count"] == 0


def test_allow_broad_filters_supports_recall_only_general_search() -> None:
    retriever = _FakeRetriever(
        {
            (): [
                _candidate(
                    "broad-1",
                    source_id="broad-source",
                    domain="manual_reimbursement",
                    score=0.91,
                    chunk_index=1,
                )
            ]
        }
    )
    use_case = SearchPolicyEvidenceUseCase(retriever=retriever, reranker=None)

    response = use_case.search(
        {
            "question": "审核参保人零星报销门诊费用时，应要求提供哪些必要材料？",
            "filters": {},
            "allow_broad_filters": True,
            "top_k": 5,
            "fetch_k": 40,
            "rerank": False,
        }
    )

    assert response["status"] == "ok"
    assert response["result_count"] == 1
    assert response["retrieval_plan"]["groups"][0]["name"] == "general_policy"
    assert response["evidence"][0]["node_id"] == "broad-1"


def test_policy_pdf_content_type_alias_is_normalized_before_validation() -> None:
    filters = PolicyRagFilters.from_mapping(
        {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["shanghai_payment_scope"],
            "content_type": ["policy_pdf", "policy_text"],
            "can_cite_as_policy_basis": True,
        }
    ).normalized()

    filters.validate()

    assert filters.content_type == ["policy_text"]


def test_content_type_is_used_as_strict_retrieval_filter() -> None:
    retriever = _FakeRetriever(
        {
            ("shanghai_payment_scope",): [
                _candidate(
                    "payment-scope",
                    source_id="payment-source",
                    domain="shanghai_payment_scope",
                    score=0.91,
                    chunk_index=1,
                )
            ]
        }
    )
    use_case = SearchPolicyEvidenceUseCase(retriever=retriever, reranker=None)

    response = use_case.search(
        {
            "question": "上海药品和诊疗项目支付范围如何核验？",
            "filters": {
                "jurisdiction": ["shanghai"],
                "policy_domain": ["shanghai_payment_scope"],
                "content_type": ["policy_pdf", "policy_text"],
                "can_cite_as_policy_basis": True,
            },
            "top_k": 5,
            "fetch_k": 20,
            "rerank": False,
        }
    )

    assert response["status"] == "ok"
    assert response["requested_filters"]["content_type"] == ["policy_text"]
    assert response["filters_used"]["content_type"] == ["policy_text"]
    assert response["retrieval_diagnostics"]["ignored_filter_fields"] == []
    assert all(call["filters"]["content_type"] == ["policy_text"] for call in retriever.calls)
