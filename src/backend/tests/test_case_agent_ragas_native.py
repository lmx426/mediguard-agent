from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.backend.scripts.build_case_agent_ragas_goldset import (
    EXPECTED_COUNT,
    EXPECTED_FIELDS,
    build_goldset,
)
from src.backend.scripts.evaluate_case_agent_ragas_native import (
    collect_policy_contexts,
    NativeRagasScorer,
    load_and_validate_goldset,
    metric_scored,
    validate_judge_model,
)


def test_goldset_builder_emits_only_three_required_fields() -> None:
    source_rows = [
        {
            "query_id": f"q-{index}",
            "question": f"question-{index}",
            "reference_answer_required": f"reference-{index}",
            "reference_answer_full": "must not be copied",
            "RES": "must not be copied",
        }
        for index in range(EXPECTED_COUNT)
    ]

    result = build_goldset(source_rows)

    assert len(result) == EXPECTED_COUNT
    assert all(set(row) == EXPECTED_FIELDS for row in result)
    assert result[0] == {
        "query_id": "q-0",
        "user_input": "question-0",
        "reference": "reference-0",
    }


def test_goldset_loader_rejects_extra_fields(tmp_path) -> None:
    path = tmp_path / "goldset.jsonl"
    row = {"query_id": "q-1", "user_input": "q", "reference": "a", "extra": 1}
    path.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(EXPECTED_COUNT)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly query_id"):
        load_and_validate_goldset(path)


def test_native_scorer_passes_metric_specific_official_inputs() -> None:
    class FakeMetric:
        def __init__(self, value: float) -> None:
            self.value = value
            self.calls: list[dict[str, object]] = []

        async def ascore(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(value=self.value)

    answer_correctness = FakeMetric(0.8)
    faithfulness = FakeMetric(0.9)
    scorer = NativeRagasScorer(
        llm=object(),
        answer_correctness=answer_correctness,
        faithfulness=faithfulness,
        judge_model="deepseek-chat",
        embedding_model="BAAI/bge-m3",
        timeout=10.0,
    )

    metrics = asyncio.run(
        scorer.score(
            user_input="question",
            response="answer",
            reference="reference",
            retrieved_contexts=["context"],
        )
    )

    assert answer_correctness.calls == [
        {"user_input": "question", "response": "answer", "reference": "reference"}
    ]
    assert faithfulness.calls == [
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["context"],
        }
    ]
    assert metrics["answer_correctness"] == {"status": "scored", "value": 0.8}
    assert metrics["faithfulness"] == {"status": "scored", "value": 0.9}


def test_faithfulness_without_context_is_not_scored() -> None:
    class FakeMetric:
        def __init__(self) -> None:
            self.called = False

        async def ascore(self, **kwargs):
            self.called = True
            return SimpleNamespace(value=1.0)

    faithfulness = FakeMetric()
    scorer = NativeRagasScorer(
        llm=object(),
        answer_correctness=FakeMetric(),
        faithfulness=faithfulness,
        judge_model="deepseek-chat",
        embedding_model="BAAI/bge-m3",
        timeout=10.0,
    )

    metrics = asyncio.run(
        scorer.score(
            user_input="question",
            response="answer",
            reference="reference",
            retrieved_contexts=[],
        )
    )

    assert metrics["faithfulness"] == {
        "status": "not_evaluable",
        "reason": "no_retrieved_contexts",
        "value": None,
    }
    assert faithfulness.called is False


def test_collect_policy_contexts_reads_expert_checkpoint_evidence() -> None:
    snapshot = {
        "adopted_policy_evidence": [
            {
                "evidence_ref": "policy-node:1",
                "source_ref": "policy-source:1",
                "title": "上海门诊零星报销",
                "excerpt": "门诊零星报销需提供身份证、社保卡和医疗费专用收据。",
                "jurisdiction": "shanghai",
                "policy_domain": "manual_reimbursement",
                "content_type": "policy_text",
                "source_url": "https://example.test/policy",
                "version": "v1",
                "metadata": {"node_id": "node-1"},
            }
        ]
    }

    class FakeRepository:
        _session_factory = None

        def get_latest_checkpoint(self, run_id: str, node_name: str | None = None):
            _ = run_id
            if node_name in {None, "expert_analysis:expert_synthesis"}:
                return snapshot
            return None

    service = SimpleNamespace(_repository=FakeRepository())

    contexts = collect_policy_contexts(service, run_id="run-1", answer=None)

    assert contexts == [
        {
            "rank": 1,
            "source_ref": "policy-source:1",
            "evidence_ref": "policy-node:1",
            "node_id": "node-1",
            "source_id": None,
            "source_url": "https://example.test/policy",
            "title": "上海门诊零星报销",
            "text": "门诊零星报销需提供身份证、社保卡和医疗费专用收据。",
        }
    ]


def test_non_finite_ragas_value_is_an_error() -> None:
    result = metric_scored(SimpleNamespace(value=float("nan")))
    assert result["status"] == "metric_error"
    assert result["value"] is None


@pytest.mark.parametrize("model", ["deepseek-reasoner", "deepseek-r1", "deepseek-v4-flash"])
def test_native_judge_rejects_known_incompatible_models(model: str) -> None:
    with pytest.raises(RuntimeError, match="non-thinking DeepSeek judge"):
        validate_judge_model(model)
