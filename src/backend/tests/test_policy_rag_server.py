from __future__ import annotations

from src.backend.mcp import policy_rag_server


class _DummyUseCase:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {"status": "ok", "payload": payload}


def test_policy_rag_search_forwards_dual_rrf_arguments(monkeypatch) -> None:
    dummy = _DummyUseCase()
    monkeypatch.setattr(policy_rag_server, "_USE_CASE", dummy, raising=False)

    response = policy_rag_server.policy_rag_search(
        question="北京参保人异地手工报销需要哪些材料？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
        recall_filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": [],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
        filter_strategy="dual_rrf",
        adaptive_top_k=True,
        allow_broad_filters=False,
        top_k=5,
        fetch_k=40,
        rerank=True,
    )

    assert response["status"] == "ok"
    assert dummy.calls[0]["filter_strategy"] == "dual_rrf"
    assert dummy.calls[0]["adaptive_top_k"] is True
    assert dummy.calls[0]["recall_filters"] == {
        "jurisdiction": ["beijing", "national"],
        "policy_domain": [],
        "content_type": ["policy_text", "table_row"],
        "can_cite_as_policy_basis": True,
    }


def test_policy_rag_search_keeps_single_branch_defaults(monkeypatch) -> None:
    dummy = _DummyUseCase()
    monkeypatch.setattr(policy_rag_server, "_USE_CASE", dummy, raising=False)

    response = policy_rag_server.policy_rag_search(
        question="北京参保人手工报销需要哪些材料？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
    )

    assert response["status"] == "ok"
    assert dummy.calls[0]["filter_strategy"] == "single"
    assert dummy.calls[0]["adaptive_top_k"] is False
    assert dummy.calls[0]["recall_filters"] is None


def test_policy_rag_search_forwards_allow_broad_filters(monkeypatch) -> None:
    dummy = _DummyUseCase()
    monkeypatch.setattr(policy_rag_server, "_USE_CASE", dummy, raising=False)

    response = policy_rag_server.policy_rag_search(
        question="审核参保人零星报销门诊费用时，应要求提供哪些必要材料？",
        filters={},
        allow_broad_filters=True,
    )

    assert response["status"] == "ok"
    assert dummy.calls[0]["allow_broad_filters"] is True
