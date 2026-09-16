"""Compare three Policy RAG retrieval paths on one fixed goldset.

Modes:
1. question -> FAISS + bge-m3 without metadata filters or BM25;
2. question + gold filters -> metadata filter + dense/bm25 RRF;
3. question + gold filters -> production strict-v3 orchestration on top of
   metadata-filtered dense/bm25 RRF.

This is an offline retrieval evaluation. It does not generate final answers,
change production state, calculate risk scores, or make audit decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.search_policy_evidence_uc import (
    SearchPolicyEvidenceUseCase,
)
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bm25_retriever import Bm25PolicyRetriever
from src.backend.infrastructure.policy_rag.faiss_retriever import FaissPolicyRetriever
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
)
from src.backend.infrastructure.policy_rag.raw_hybrid_retriever import (
    RawHybridPolicyRetriever,
)
from src.backend.scripts.evaluate_policy_rag_v2_1_retrieval import (
    append_jsonl,
    context_from_candidate,
    gold_nodes,
    normalize_filters,
    parse_top_k_values,
)
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    read_jsonl,
)
from src.backend.scripts.policy_rag_eval.native_context_metrics import (
    NativeContextMetricsScorer,
)


MODE_DENSE = "question_dense_bge_m3"
MODE_FILTERED_HYBRID = "question_filters_hybrid_rrf"
MODE_STRICT_V3 = "question_filters_strict_v3_hybrid_rrf"
MODE_ORDER = (MODE_DENSE, MODE_FILTERED_HYBRID, MODE_STRICT_V3)
MODE_LABELS = {
    MODE_DENSE: "question -> bge-m3",
    MODE_FILTERED_HYBRID: "question + filters -> metadata + bge-m3/BM25/RRF",
    MODE_STRICT_V3: "question + filters -> metadata + hybrid RRF + strict-v3",
}

DEFAULT_EVAL_SET = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_1_entity_fixed_from300_audited.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_v2_1_three_way_eval_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_v2_1_three_way_eval_report.md"
)
DEFAULT_JUDGE_CACHE = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_three_way_ragas_context_cache.jsonl"
)
DEFAULT_RAGAS_INPUTS = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_v2_1_three_way_ragas_inputs.jsonl"
)
DEFAULT_FETCH_K = 20
DEFAULT_TOP_K_VALUES = "3,5,10"
DEFAULT_CONTEXT_K_VALUES = "5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a three-way Policy RAG retrieval comparison with native Ragas context metrics."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--judge-cache-jsonl", type=Path, default=DEFAULT_JUDGE_CACHE)
    parser.add_argument("--ragas-inputs-jsonl", type=Path, default=DEFAULT_RAGAS_INPUTS)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--top-k-values", default=DEFAULT_TOP_K_VALUES)
    parser.add_argument(
        "--context-k-values",
        default=DEFAULT_CONTEXT_K_VALUES,
        help="Top-K values passed to native Ragas. Defaults to 5 to control Judge calls.",
    )
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-resume-judge", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    write_report_files(report, args.report_json, args.report_md)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "query_count": report["query_count"],
                "modes": list(MODE_ORDER),
                "top_k_values": report["top_k_values"],
                "context_k_values": report["context_k_values"],
                "judge_enabled": report["judge_enabled"],
                "judge_model": report["judge_model"],
                "ragas_version": report["ragas_version"],
                "summary_top5": summary["by_mode_top5"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "judge_cache_jsonl": str(args.judge_cache_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "ok" and not args.allow_partial:
        raise SystemExit(2)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.fetch_k <= 0:
        raise ValueError("fetch-k must be positive")
    top_k_values = parse_top_k_values(args.top_k_values)
    context_k_values = parse_top_k_values(args.context_k_values)
    if any(k > max(top_k_values) for k in context_k_values):
        raise ValueError("context-k-values cannot exceed the largest top-k value")

    rows = read_jsonl(args.eval_set_jsonl)
    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("The selected goldset slice is empty")

    scorer = None
    ragas_version = None
    judge_model = None
    if not args.skip_judge:
        scorer = NativeContextMetricsScorer.from_environment(
            env_file=args.env_file,
            judge_model=args.judge_model,
            timeout=args.judge_timeout,
            temperature=args.judge_temperature,
        )
        ragas_version = scorer.ragas_version
        judge_model = scorer.judge_model

    cache = {} if args.no_resume_judge else load_judge_cache(args.judge_cache_jsonl)
    args.judge_cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.ragas_inputs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense = FaissPolicyRetriever(embedder=embedder)
    hybrid = RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=Bm25PolicyRetriever(),
    )
    strict_use_case = SearchPolicyEvidenceUseCase(
        retriever=hybrid,
        reranker=None,
        retrieval_strategy="hybrid",
    )

    max_top_k = max(top_k_values)
    query_reports: list[dict[str, Any]] = []
    metric_error_count = 0
    strict_applied_count = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"Goldset row {index} has an empty question")
        filters = normalize_filters(row.get("filters"))

        dense_candidates = dense.retrieve(
            question=question,
            filters={},
            fetch_k=max(args.fetch_k, max_top_k),
        )
        hybrid_candidates = hybrid.retrieve(
            question=question,
            filters=filters,
            fetch_k=max(args.fetch_k, max_top_k),
        )
        strict_response = strict_use_case.search(
            {
                "question": question,
                "filters": filters,
                "top_k": max_top_k,
                "fetch_k": args.fetch_k,
                "rerank": False,
            }
        )
        if strict_response.get("status") != "ok":
            raise RuntimeError(
                f"strict-v3 returned {strict_response.get('status')}: "
                f"{strict_response.get('message') or strict_response.get('error')}"
            )
        strict_plan = strict_response.get("retrieval_plan") or {}
        if len(strict_plan.get("groups") or []) > 1:
            strict_applied_count += 1

        mode_contexts = {
            MODE_DENSE: [
                context_from_candidate(candidate, rank=rank)
                for rank, candidate in enumerate(
                    dense_candidates[:max_top_k], start=1
                )
            ],
            MODE_FILTERED_HYBRID: [
                context_from_candidate(candidate, rank=rank)
                for rank, candidate in enumerate(
                    hybrid_candidates[:max_top_k], start=1
                )
            ],
            MODE_STRICT_V3: normalize_response_contexts(
                strict_response.get("evidence") or []
            ),
        }
        mode_reports: dict[str, Any] = {}
        reference = reference_for_row(row)
        for mode in MODE_ORDER:
            contexts = mode_contexts[mode]
            metric_by_k = {
                str(k): node_metrics(
                    retrieved_contexts=contexts,
                    gold_node_ids=gold_nodes(row),
                    k=k,
                )
                for k in top_k_values
            }
            context_metrics: dict[str, Any] = {}
            if scorer is not None:
                for k in context_k_values:
                    cache_key = build_cache_key(
                        mode=mode,
                        query_id=str(row.get("query_id") or ""),
                        context_k=k,
                        question=question,
                        reference=reference,
                        contexts=contexts[:k],
                    )
                    cached = cache.get(cache_key)
                    if cached is None:
                        input_row = {
                            "schema_version": "policy_rag_v2_1_three_way_ragas_input_v1",
                            "cache_key": cache_key,
                            "mode": mode,
                            "query_id": row.get("query_id"),
                            "context_k": k,
                            "user_input": question,
                            "reference": reference,
                            "retrieved_contexts": [
                                str(context.get("text") or "")
                                for context in contexts[:k]
                            ],
                        }
                        append_jsonl(args.ragas_inputs_jsonl, [input_row])
                        scored = scorer.score_sync(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=input_row["retrieved_contexts"],
                        )
                        cached = {
                            "schema_version": "policy_rag_v2_1_three_way_ragas_cache_v1",
                            "cache_key": cache_key,
                            "mode": mode,
                            "query_id": row.get("query_id"),
                            "context_k": k,
                            "input_fingerprint": cache_key.rsplit(":", 1)[-1],
                            **scored,
                        }
                        append_jsonl(args.judge_cache_jsonl, [cached])
                        cache[cache_key] = cached
                    context_metrics[str(k)] = {
                        "context_precision": cached.get("context_precision"),
                        "context_recall": cached.get("context_recall"),
                    }
                    metric_error_count += count_metric_errors(context_metrics[str(k)])

            mode_reports[mode] = {
                "label": MODE_LABELS[mode],
                "filters_source": "none" if mode == MODE_DENSE else "goldset",
                "filters_used": {} if mode == MODE_DENSE else filters,
                "retrieved_contexts": contexts,
                "retrieved_node_ids": [
                    str(context.get("node_id") or "") for context in contexts
                ],
                "per_k": metric_by_k,
                "context_metrics": context_metrics,
                "retrieval_plan": strict_plan if mode == MODE_STRICT_V3 else None,
            }

        query_reports.append(
            {
                "query_id": row.get("query_id"),
                "question": question,
                "question_type": row.get("question_type"),
                "bundle_type": row.get("bundle_type"),
                "gold_filters": filters,
                "gold_node_ids": sorted(gold_nodes(row)),
                "reference": reference,
                "modes": mode_reports,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if index == 1 or index % max(1, args.progress_every) == 0 or index == total:
            print(
                f"[policy-rag-three-way] progress={index}/{total} "
                f"judge={'on' if scorer is not None else 'off'}",
                flush=True,
            )

    summary = summarize_three_way(
        query_reports,
        top_k_values=top_k_values,
        context_k_values=context_k_values,
    )
    status = "ok" if metric_error_count == 0 else "partial"
    return {
        "status": status,
        "schema_version": "policy_rag_v2_1_three_way_eval_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "query_count": len(query_reports),
        "offset": args.offset,
        "limit": args.limit,
        "fetch_k": args.fetch_k,
        "top_k_values": top_k_values,
        "context_k_values": context_k_values,
        "retrieval_backend": "faiss",
        "embedding_model": "BAAI/bge-m3",
        "judge_enabled": scorer is not None,
        "judge_model": judge_model,
        "ragas_version": ragas_version,
        "judge_cache_jsonl": str(args.judge_cache_jsonl),
        "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
        "strict_v3_applied_count": strict_applied_count,
        "metric_error_count": metric_error_count,
        "queries": query_reports,
        "summary": summary,
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_gold_filters_for_modes": [MODE_FILTERED_HYBRID, MODE_STRICT_V3],
            "dense_mode_ignores_filters": True,
            "uses_native_ragas_context_metrics": scorer is not None,
            "generates_final_rag_answer": False,
            "calculates_answer_correctness": False,
            "calculates_faithfulness": False,
            "makes_audit_decisions": False,
            "changes_production_retriever": False,
        },
    }


def reference_for_row(row: dict[str, Any]) -> str:
    return str(
        row.get("reference_answer_required")
        or row.get("reference_answer")
        or row.get("reference_answer_full")
        or ""
    ).strip()


def normalize_response_contexts(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for rank, item in enumerate(evidence, start=1):
        payload = dict(item)
        payload["rank"] = rank
        payload["text"] = str(payload.get("text") or "")
        contexts.append(payload)
    return contexts


def node_metrics(
    *,
    retrieved_contexts: list[dict[str, Any]],
    gold_node_ids: set[str],
    k: int,
) -> dict[str, Any]:
    top_contexts = retrieved_contexts[:k]
    retrieved_ids = [str(item.get("node_id") or "") for item in top_contexts]
    hits = [node_id for node_id in retrieved_ids if node_id in gold_node_ids]
    first_rank = next(
        (
            rank
            for rank, node_id in enumerate(retrieved_ids, start=1)
            if node_id in gold_node_ids
        ),
        None,
    )
    return {
        "k": k,
        "result_count": len(top_contexts),
        "recall": len(set(hits)) / len(gold_node_ids) if gold_node_ids else None,
        "hit": 1.0 if first_rank is not None else 0.0,
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_gold_rank": first_rank,
        "gold_node_count": len(gold_node_ids),
        "hit_node_count": len(set(hits)),
    }


def build_cache_key(
    *,
    mode: str,
    query_id: str,
    context_k: int,
    question: str,
    reference: str,
    contexts: list[dict[str, Any]],
) -> str:
    fingerprint_payload = {
        "mode": mode,
        "query_id": query_id,
        "context_k": context_k,
        "question": question,
        "reference": reference,
        "contexts": [
            {
                "node_id": context.get("node_id"),
                "text": context.get("text"),
            }
            for context in contexts
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"three_way_v1:{mode}:{query_id}:{context_k}:{fingerprint}"


def load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("cache_key") or "")
        if key:
            cache[key] = row
    return cache


def count_metric_errors(metrics: dict[str, Any]) -> int:
    return sum(
        1
        for value in metrics.values()
        if isinstance(value, dict) and value.get("status") == "metric_error"
    )


def summarize_three_way(
    query_reports: list[dict[str, Any]],
    *,
    top_k_values: list[int],
    context_k_values: list[int],
) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for mode in MODE_ORDER:
        by_k: dict[str, Any] = {}
        context_by_k: dict[str, Any] = {}
        mode_rows = [item["modes"][mode] for item in query_reports]
        for k in top_k_values:
            metrics = [row["per_k"][str(k)] for row in mode_rows]
            by_k[str(k)] = aggregate_metric_rows(metrics)
        for k in context_k_values:
            metrics = [
                row["context_metrics"].get(str(k))
                for row in mode_rows
                if str(k) in row["context_metrics"]
            ]
            context_by_k[str(k)] = aggregate_context_rows(metrics)
        by_mode[mode] = {
            "label": MODE_LABELS[mode],
            "node_by_k": by_k,
            "context_by_k": context_by_k,
        }

    primary_k = 5 if 5 in top_k_values else top_k_values[0]
    primary_context_k = 5 if 5 in context_k_values else context_k_values[0]
    deltas: dict[str, Any] = {}
    for left, right, name in (
        (MODE_DENSE, MODE_FILTERED_HYBRID, "filtered_hybrid_minus_dense"),
        (MODE_FILTERED_HYBRID, MODE_STRICT_V3, "strict_v3_minus_filtered_hybrid"),
    ):
        deltas[name] = {
            metric: delta(
                by_mode[left]["node_by_k"][str(primary_k)].get(metric),
                by_mode[right]["node_by_k"][str(primary_k)].get(metric),
            )
            for metric in ("recall", "hit", "mrr")
        }
        deltas[name].update(
            {
                "context_precision": delta(
                    by_mode[left]["context_by_k"].get(str(primary_context_k), {}).get(
                        "context_precision"
                    ),
                    by_mode[right]["context_by_k"].get(str(primary_context_k), {}).get(
                        "context_precision"
                    ),
                ),
                "context_recall": delta(
                    by_mode[left]["context_by_k"].get(str(primary_context_k), {}).get(
                        "context_recall"
                    ),
                    by_mode[right]["context_by_k"].get(str(primary_context_k), {}).get(
                        "context_recall"
                    ),
                ),
            }
        )
    return {
        "by_mode": by_mode,
        "by_mode_top5": {
            mode: {
                "node": by_mode[mode]["node_by_k"].get(str(primary_k)),
                "context": by_mode[mode]["context_by_k"].get(str(primary_context_k)),
            }
            for mode in MODE_ORDER
        },
        "primary_node_k": primary_k,
        "primary_context_k": primary_context_k,
        "deltas": deltas,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "recall": average(row.get("recall") for row in rows),
        "hit": average(row.get("hit") for row in rows),
        "mrr": average(row.get("mrr") for row in rows),
        "full_node_recall_count": sum(row.get("recall") == 1.0 for row in rows),
        "hit_count": sum(row.get("hit") == 1.0 for row in rows),
    }


def aggregate_context_rows(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    precision = [
        row["context_precision"].get("value")
        for row in rows
        if row and isinstance(row.get("context_precision"), dict)
    ]
    recall = [
        row["context_recall"].get("value")
        for row in rows
        if row and isinstance(row.get("context_recall"), dict)
    ]
    return {
        "query_count": len(rows),
        "context_precision": average(precision),
        "context_recall": average(recall),
        "precision_scored_count": sum(value is not None for value in precision),
        "recall_scored_count": sum(value is not None for value in recall),
        "metric_error_count": sum(count_metric_errors(row or {}) for row in rows),
    }


def average(values: Iterable[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def write_report_files(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Policy RAG v2.1 Three-Way Retrieval Evaluation",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This is an offline retrieval-only comparison. Modes 2 and 3 use the goldset filters; mode 1 deliberately ignores filters. Context metrics are native Ragas reference-based metrics.",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Node K values: `{report.get('top_k_values')}`",
        f"- Ragas K values: `{report.get('context_k_values')}`",
        f"- Dense backend: `{report.get('retrieval_backend')}`",
        f"- Judge: `{report.get('judge_model')}` / Ragas `{report.get('ragas_version')}`",
        f"- Strict-v3 applied rows: {report.get('strict_v3_applied_count')}",
        "",
        "## Overall Node Metrics",
        "",
        "| Mode | K | Recall@K | Hit@K | MRR@K | Full Recall | Hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODE_ORDER:
        for k, metrics in summary["by_mode"][mode]["node_by_k"].items():
            lines.append(
                f"| {MODE_LABELS[mode]} | {k} | {fmt(metrics.get('recall'))} | "
                f"{fmt(metrics.get('hit'))} | {fmt(metrics.get('mrr'))} | "
                f"{metrics.get('full_node_recall_count')} | {metrics.get('hit_count')} |"
            )
    lines.extend(
        [
            "",
            "## Native Ragas Context Metrics",
            "",
            "| Mode | K | Context Precision | Context Recall | Precision Scored | Recall Scored | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODE_ORDER:
        for k, metrics in summary["by_mode"][mode]["context_by_k"].items():
            lines.append(
                f"| {MODE_LABELS[mode]} | {k} | {fmt(metrics.get('context_precision'))} | "
                f"{fmt(metrics.get('context_recall'))} | {metrics.get('precision_scored_count')} | "
                f"{metrics.get('recall_scored_count')} | {metrics.get('metric_error_count')} |"
            )
    lines.extend(["", "## Primary Deltas", "", "```json"])
    lines.append(json.dumps(summary["deltas"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Sample Details", ""])
    for row in report.get("queries", [])[:20]:
        lines.extend(
            [
                f"### {row.get('query_id')}",
                "",
                f"- Question: {row.get('question')}",
                f"- Gold filters: `{json.dumps(row.get('gold_filters'), ensure_ascii=False)}`",
            ]
        )
        for mode in MODE_ORDER:
            mode_row = row["modes"][mode]
            k = str(summary["primary_node_k"])
            node = mode_row["per_k"].get(k) or {}
            context = mode_row["context_metrics"].get(
                str(summary["primary_context_k"]), {}
            )
            lines.append(
                f"- {MODE_LABELS[mode]}: nodes=`{mode_row.get('retrieved_node_ids', [])[:5]}`, "
                f"Recall={fmt(node.get('recall'))}, MRR={fmt(node.get('mrr'))}, "
                f"CP={fmt((context.get('context_precision') or {}).get('value'))}, "
                f"CR={fmt((context.get('context_recall') or {}).get('value'))}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
