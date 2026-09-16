from __future__ import annotations

from src.backend.scripts.evaluate_policy_rag_v2_1_three_way import (
    MODE_DENSE,
    MODE_FILTERED_HYBRID,
    MODE_STRICT_V3,
    aggregate_context_rows,
    build_cache_key,
    node_metrics,
    summarize_three_way,
)


def _context(node_id: str, text: str = "evidence") -> dict[str, object]:
    return {"node_id": node_id, "text": text}


def test_node_metrics_use_distinct_gold_nodes_and_first_gold_rank() -> None:
    metrics = node_metrics(
        retrieved_contexts=[
            _context("noise"),
            _context("gold-a"),
            _context("gold-a"),
            _context("gold-b"),
        ],
        gold_node_ids={"gold-a", "gold-b"},
        k=4,
    )

    assert metrics["recall"] == 1.0
    assert metrics["hit"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["hit_node_count"] == 2


def test_cache_key_changes_when_retrieved_context_changes() -> None:
    common = {
        "mode": MODE_FILTERED_HYBRID,
        "query_id": "q1",
        "context_k": 5,
        "question": "问题",
        "reference": "参考答案",
    }
    first = build_cache_key(contexts=[_context("n1", "a")], **common)
    second = build_cache_key(contexts=[_context("n1", "b")], **common)

    assert first != second


def test_three_way_summary_keeps_context_metrics_separate_from_node_metrics() -> None:
    modes = {
        mode: {
            "per_k": {
                "5": {
                    "recall": 1.0 if mode != MODE_DENSE else 0.5,
                    "hit": 1.0,
                    "mrr": 1.0 if mode == MODE_STRICT_V3 else 0.5,
                }
            },
            "context_metrics": {
                "5": {
                    "context_precision": {"status": "scored", "value": 0.8},
                    "context_recall": {"status": "scored", "value": 0.9},
                }
            },
        }
        for mode in (MODE_DENSE, MODE_FILTERED_HYBRID, MODE_STRICT_V3)
    }
    rows = [{"modes": modes}, {"modes": modes}]

    summary = summarize_three_way(
        rows,
        top_k_values=[5],
        context_k_values=[5],
    )

    assert summary["by_mode"][MODE_STRICT_V3]["node_by_k"]["5"]["mrr"] == 1.0
    assert (
        summary["by_mode"][MODE_FILTERED_HYBRID]["context_by_k"]["5"][
            "context_recall"
        ]
        == 0.9
    )
    assert summary["deltas"]["strict_v3_minus_filtered_hybrid"]["mrr"] == 0.5


def test_three_way_mode_order_is_stable() -> None:
    assert (MODE_DENSE, MODE_FILTERED_HYBRID, MODE_STRICT_V3) == (
        "question_dense_bge_m3",
        "question_filters_hybrid_rrf",
        "question_filters_strict_v3_hybrid_rrf",
    )


def test_aggregate_context_rows_reports_scored_and_errors() -> None:
    result = aggregate_context_rows(
        [
            {
                "context_precision": {"status": "scored", "value": 0.8},
                "context_recall": {"status": "scored", "value": 0.6},
            },
            {
                "context_precision": {
                    "status": "metric_error",
                    "value": None,
                },
                "context_recall": {"status": "scored", "value": 1.0},
            },
        ]
    )

    assert result["context_precision"] == 0.8
    assert result["context_recall"] == 0.8
    assert result["precision_scored_count"] == 1
    assert result["recall_scored_count"] == 2
    assert result["metric_error_count"] == 1
