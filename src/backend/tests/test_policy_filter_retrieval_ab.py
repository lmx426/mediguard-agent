from __future__ import annotations

from src.backend.scripts.evaluate_policy_filter_retrieval_ab import (
    build_ab_report,
    evidence_group_matches,
)


def _gold() -> dict[str, object]:
    return {
        "query_id": "q1",
        "gold_evidence_groups": [
            {
                "group_id": "g1",
                "required": True,
                "gold_node_ids": ["node-1"],
                "evidence_refs": [
                    {"anchor_terms": ["法定办结时限", "30个工作日", "零星报销"]}
                ],
            }
        ],
    }


def _result(*, contexts: list[str], ac: float, faith: float) -> dict[str, object]:
    return {
        "query_id": "q1",
        "status": "scored",
        "ragas_input": {"retrieved_contexts": contexts},
        "metrics": {
            "answer_correctness": {"status": "scored", "value": ac},
            "faithfulness": {"status": "scored", "value": faith},
        },
        "elapsed_seconds": 10.0,
    }


def test_evidence_group_match_accepts_node_id_or_two_audited_anchors() -> None:
    group = _gold()["gold_evidence_groups"][0]

    assert evidence_group_matches(
        group=group,
        contexts=[],
        node_ids={"node-1"},
    )
    assert evidence_group_matches(
        group=group,
        contexts=["零星报销的法定办结时限另行规定"],
        node_ids=set(),
    )
    assert not evidence_group_matches(
        group=group,
        contexts=["只出现零星报销一个锚点"],
        node_ids=set(),
    )


def test_ab_report_reads_native_ragas_values_without_recomputing() -> None:
    report = build_ab_report(
        gold_rows=[_gold()],
        baseline_rows=[_result(contexts=[], ac=0.3, faith=1.0)],
        candidate_rows=[
            _result(
                contexts=["零星报销法定办结时限为30个工作日"],
                ac=0.8,
                faith=1.0,
            )
        ],
    )

    assert report["baseline"]["all_required_groups_hit_rate"] == 0.0
    assert report["candidate"]["all_required_groups_hit_rate"] == 1.0
    assert report["delta"]["answer_correctness_average"] == 0.5
    assert report["boundary"]["ragas_metrics_recomputed"] is False
