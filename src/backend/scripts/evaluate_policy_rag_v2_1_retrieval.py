"""Evaluate Policy RAG v2.1 retrieval with node metrics and DeepSeek judge.

This script evaluates retrieval only. It does not generate final RAG answers
and does not make audit decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.schemas import PolicyRagFilters
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
from src.backend.scripts.policy_rag_eval.common import (
    load_deepseek_config,
    parse_json_object,
    read_jsonl,
    write_jsonl,
)


TOP_K_VALUES = [3, 5, 10]
DEFAULT_FETCH_K = 20
DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_1_200_from300_audited.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_v2_1_retrieval_eval_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_v2_1_retrieval_eval_report.md"
)
DEFAULT_JUDGE_JSONL = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_v2_1_retrieval_judge_results.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v2.1 Policy RAG retrieval with DeepSeek context judge."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--judge-jsonl", type=Path, default=DEFAULT_JUDGE_JSONL)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--top-k-values", default="3,5,10")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-resume-judge", action="store_true")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-timeout", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = parse_top_k_values(args.top_k_values)
    report = run_eval(args, top_k_values=top_k_values)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "top_k_values": top_k_values,
                "judge_enabled": report["judge_enabled"],
                "judge_cache_path": str(args.judge_jsonl),
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_eval(args: argparse.Namespace, *, top_k_values: list[int]) -> dict[str, Any]:
    eval_rows = read_jsonl(args.eval_set_jsonl)
    if args.offset:
        eval_rows = eval_rows[args.offset :]
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    retriever = build_retriever(args)
    judge_cache = load_judge_cache(args.judge_jsonl) if not args.no_resume_judge else {}
    judge_client = None if args.skip_judge else DeepSeekContextJudge(args)
    max_top_k = max(top_k_values)

    query_reports: list[dict[str, Any]] = []
    judge_outputs_to_write: list[dict[str, Any]] = []
    total = len(eval_rows)
    for index, row in enumerate(eval_rows, start=1):
        started = time.perf_counter()
        filters = normalize_filters(row.get("filters"))
        candidates = retriever.retrieve(
            question=str(row["question"]),
            filters=filters,
            fetch_k=max(args.fetch_k, max_top_k),
        )
        retrieved_contexts = [
            context_from_candidate(candidate, rank=rank)
            for rank, candidate in enumerate(candidates[:max_top_k], start=1)
        ]
        elapsed = time.perf_counter() - started

        judge_result: dict[str, Any] | None = None
        if judge_client is not None:
            cached = judge_cache.get(str(row.get("query_id") or ""))
            if cached is not None:
                judge_result = cached
            else:
                judge_result = judge_client.judge(
                    row=row,
                    retrieved_contexts=retrieved_contexts,
                )
                judge_outputs_to_write.append(judge_result)
                append_jsonl(args.judge_jsonl, [judge_result])

        query_reports.append(
            evaluate_query(
                row,
                retrieved_contexts=retrieved_contexts,
                judge_result=judge_result,
                elapsed_seconds=elapsed,
                top_k_values=top_k_values,
            )
        )
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"[policy-rag-v2.1-eval] progress={index}/{total} "
                f"judge={'off' if args.skip_judge else 'on'}",
                flush=True,
            )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "query_count": len(query_reports),
        "fetch_k": args.fetch_k,
        "top_k_values": top_k_values,
        "retrieval_strategy": "metadata_filter_faiss_bge_m3_bm25_rrf",
        "judge_enabled": not args.skip_judge,
        "judge_model": None if args.skip_judge else judge_client.model,
        "judge_cache_path": str(args.judge_jsonl),
        "queries": query_reports,
        "summary": summarize(query_reports, top_k_values=top_k_values),
        "boundary": {
            "runs_retrieval_quality_eval": True,
            "uses_llm_judge_for_context_metrics": not args.skip_judge,
            "generates_final_rag_answer": False,
            "calculates_answer_correctness": False,
            "calculates_faithfulness": False,
            "calculates_citation_accuracy": False,
            "makes_audit_decisions": False,
        },
    }


def build_retriever(args: argparse.Namespace) -> RawHybridPolicyRetriever:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense = FaissPolicyRetriever(embedder=embedder)
    bm25 = Bm25PolicyRetriever()
    return RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=bm25,
    )


def evaluate_query(
    row: dict[str, Any],
    *,
    retrieved_contexts: list[dict[str, Any]],
    judge_result: dict[str, Any] | None,
    elapsed_seconds: float,
    top_k_values: list[int],
) -> dict[str, Any]:
    gold_node_ids = gold_nodes(row)
    per_k = {
        str(k): metrics_at_k(
            retrieved_contexts=retrieved_contexts,
            gold_node_ids=gold_node_ids,
            judge_result=judge_result,
            k=k,
        )
        for k in top_k_values
    }
    return {
        "query_id": row.get("query_id"),
        "candidate_id": row.get("candidate_id"),
        "bundle_type": row.get("bundle_type"),
        "question_type": row.get("question_type"),
        "policy_domains": list((row.get("filters") or {}).get("policy_domain") or []),
        "content_types": list((row.get("filters") or {}).get("content_type") or []),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "gold_node_ids": sorted(gold_node_ids),
        "retrieved_node_ids": [context["node_id"] for context in retrieved_contexts],
        "retrieved_contexts": retrieved_contexts,
        "elapsed_seconds": elapsed_seconds,
        "judge": judge_result,
        "per_k": per_k,
    }


def metrics_at_k(
    *,
    retrieved_contexts: list[dict[str, Any]],
    gold_node_ids: set[str],
    judge_result: dict[str, Any] | None,
    k: int,
) -> dict[str, Any]:
    contexts_at_k = retrieved_contexts[:k]
    retrieved_node_ids = [str(context.get("node_id") or "") for context in contexts_at_k]
    hits = [node_id for node_id in retrieved_node_ids if node_id in gold_node_ids]
    first_gold_rank = next(
        (
            rank
            for rank, node_id in enumerate(retrieved_node_ids, start=1)
            if node_id in gold_node_ids
        ),
        None,
    )
    node_recall = len(set(hits)) / len(gold_node_ids) if gold_node_ids else None
    hit_at_k = 1.0 if first_gold_rank is not None else 0.0
    mrr_at_k = 1.0 / first_gold_rank if first_gold_rank is not None else 0.0
    context_recall = None
    context_precision = None
    useful_flags: list[int] = []
    if judge_result is not None:
        context_recall = context_recall_at_k(judge_result, k)
        useful_flags = usefulness_flags_at_k(judge_result, k)
        context_precision = context_precision_at_k(useful_flags)
    return {
        "k": k,
        "result_count": len(contexts_at_k),
        "recall": node_recall,
        "hit": hit_at_k,
        "mrr": mrr_at_k,
        "first_gold_rank": first_gold_rank,
        "gold_node_count": len(gold_node_ids),
        "hit_node_count": len(set(hits)),
        "context_recall": context_recall,
        "context_precision": context_precision,
        "useful_context_flags": useful_flags,
    }


def context_recall_at_k(judge_result: dict[str, Any], k: int) -> float:
    claim_support = list(judge_result.get("claim_support") or [])
    if not claim_support:
        return 0.0
    supported = 0
    for item in claim_support:
        ranks = [
            int(rank)
            for rank in item.get("supporting_context_ranks") or []
            if str(rank).isdigit()
        ]
        if bool(item.get("supported")) and any(rank <= k for rank in ranks):
            supported += 1
    return supported / len(claim_support)


def usefulness_flags_at_k(judge_result: dict[str, Any], k: int) -> list[int]:
    by_rank = {
        int(item.get("rank")): 1 if bool(item.get("useful")) else 0
        for item in judge_result.get("context_usefulness") or []
        if str(item.get("rank") or "").isdigit()
    }
    return [by_rank.get(rank, 0) for rank in range(1, k + 1)]


def context_precision_at_k(useful_flags: list[int]) -> float:
    useful_count = sum(useful_flags)
    if useful_count <= 0:
        return 0.0
    running_useful = 0
    weighted_precision = 0.0
    for index, flag in enumerate(useful_flags, start=1):
        if flag:
            running_useful += 1
            weighted_precision += running_useful / index
    return weighted_precision / useful_count


class DeepSeekContextJudge:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The openai package is required for DeepSeek judge.") from exc

        config = load_deepseek_config(args.env_file)
        self.model = config["model"]
        self._client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=args.judge_timeout,
            max_retries=1,
        )
        self._temperature = args.judge_temperature

    def judge(
        self,
        *,
        row: dict[str, Any],
        retrieved_contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "query_id": row.get("query_id"),
            "question": row.get("question"),
            "reference_answer": row.get("reference_answer"),
            "retrieved_contexts": [
                {
                    "rank": context["rank"],
                    "node_id": context["node_id"],
                    "source_id": context.get("source_id"),
                    "source_url": context.get("source_url"),
                    "text": truncate_text(str(context.get("text") or ""), 1400),
                }
                for context in retrieved_contexts
            ],
        }
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": judge_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
            stream=False,
        )
        parsed = parse_json_object(response.choices[0].message.content or "")
        return normalize_judge_result(
            parsed,
            query_id=str(row.get("query_id") or ""),
            model=self.model,
            retrieved_context_count=len(retrieved_contexts),
        )


def judge_system_prompt() -> str:
    return (
        "你是医保政策 RAG 检索评测裁判，只评估检索上下文是否支撑参考答案。"
        "不要使用外部知识，不要补充政策结论，不要判断审核通过/拒付/处罚。"
        "输出必须是 JSON object。"
        "任务1：把 reference_answer 拆成 1-6 条 atomic factual claims，"
        "每条 claim 只能包含一个可核验事实。"
        "任务2：判断每条 claim 是否能从 retrieved_contexts 直接支持或合理归因；"
        "必须给出 supporting_context_ranks。若没有支撑，supported=false 且 ranks=[]。"
        "任务3：判断每个 retrieved context 是否 useful；只要它支撑至少一个 claim，或包含回答该问题必要事实，就 useful=true。"
        "返回格式："
        "{\"claims\":[{\"claim_id\":\"c1\",\"statement\":\"...\"}],"
        "\"claim_support\":[{\"claim_id\":\"c1\",\"supported\":true,\"supporting_context_ranks\":[1]}],"
        "\"context_usefulness\":[{\"rank\":1,\"useful\":true,\"supported_claim_ids\":[\"c1\"]}]}"
    )


def normalize_judge_result(
    payload: dict[str, Any],
    *,
    query_id: str,
    model: str,
    retrieved_context_count: int,
) -> dict[str, Any]:
    claims = []
    for index, item in enumerate(list(payload.get("claims") or [])[:6], start=1):
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        claims.append(
            {
                "claim_id": str(item.get("claim_id") or f"c{index}"),
                "statement": statement,
            }
        )
    valid_claim_ids = {claim["claim_id"] for claim in claims}
    claim_support = []
    for claim in claims:
        raw = next(
            (
                item
                for item in payload.get("claim_support") or []
                if isinstance(item, dict)
                and str(item.get("claim_id") or "") == claim["claim_id"]
            ),
            {},
        )
        ranks = normalized_ranks(raw.get("supporting_context_ranks"), retrieved_context_count)
        claim_support.append(
            {
                "claim_id": claim["claim_id"],
                "supported": bool(raw.get("supported")) and bool(ranks),
                "supporting_context_ranks": ranks,
            }
        )

    context_usefulness = []
    usefulness_by_rank = {
        int(item.get("rank")): item
        for item in payload.get("context_usefulness") or []
        if isinstance(item, dict) and str(item.get("rank") or "").isdigit()
    }
    for rank in range(1, retrieved_context_count + 1):
        raw = usefulness_by_rank.get(rank, {})
        supported_claim_ids = [
            str(value)
            for value in raw.get("supported_claim_ids") or []
            if str(value) in valid_claim_ids
        ]
        context_usefulness.append(
            {
                "rank": rank,
                "useful": bool(raw.get("useful")) and bool(supported_claim_ids),
                "supported_claim_ids": supported_claim_ids,
            }
        )
    return {
        "schema_version": "policy_rag_v2_1_context_judge_v1",
        "query_id": query_id,
        "judge_model": model,
        "claims": claims,
        "claim_support": claim_support,
        "context_usefulness": context_usefulness,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def normalized_ranks(value: object, max_rank: int) -> list[int]:
    if not isinstance(value, list):
        return []
    ranks: list[int] = []
    for item in value:
        try:
            rank = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= rank <= max_rank and rank not in ranks:
            ranks.append(rank)
    return sorted(ranks)


def summarize(
    query_reports: list[dict[str, Any]],
    *,
    top_k_values: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "query_count": len(query_reports),
        "avg_seconds": average(item["elapsed_seconds"] for item in query_reports),
        "by_k": {},
        "by_question_type": grouped_summary(query_reports, "question_type", top_k_values),
        "by_bundle_type": grouped_summary(query_reports, "bundle_type", top_k_values),
        "by_policy_domain": multi_value_grouped_summary(query_reports, "policy_domains", top_k_values),
        "by_content_type": multi_value_grouped_summary(query_reports, "content_types", top_k_values),
    }
    for k in top_k_values:
        summary["by_k"][str(k)] = aggregate_at_k(query_reports, k)
    return summary


def grouped_summary(
    query_reports: list[dict[str, Any]],
    key: str,
    top_k_values: list[int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in query_reports:
        grouped[str(item.get(key) or "unknown")].append(item)
    return {
        group: {
            "query_count": len(items),
            "by_k": {str(k): aggregate_at_k(items, k) for k in top_k_values},
        }
        for group, items in sorted(grouped.items())
    }


def multi_value_grouped_summary(
    query_reports: list[dict[str, Any]],
    key: str,
    top_k_values: list[int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in query_reports:
        values = item.get(key) or ["unknown"]
        for value in values:
            grouped[str(value or "unknown")].append(item)
    return {
        group: {
            "query_count": len(items),
            "by_k": {str(k): aggregate_at_k(items, k) for k in top_k_values},
        }
        for group, items in sorted(grouped.items())
    }


def aggregate_at_k(query_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    metrics = [item["per_k"][str(k)] for item in query_reports]
    return {
        "recall": average(item.get("recall") for item in metrics),
        "hit": average(item.get("hit") for item in metrics),
        "mrr": average(item.get("mrr") for item in metrics),
        "context_recall": average(item.get("context_recall") for item in metrics),
        "context_precision": average(item.get("context_precision") for item in metrics),
        "full_node_recall_count": sum(1 for item in metrics if item.get("recall") == 1.0),
        "hit_count": sum(1 for item in metrics if item.get("hit") == 1.0),
    }


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy RAG v2.1 Retrieval Eval Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report evaluates retrieval only. Node metrics are exact node-id metrics. Context metrics use DeepSeek Judge in a RAGAS-style flow when enabled.",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Retrieval strategy: `{report.get('retrieval_strategy')}`",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Top K values: {report.get('top_k_values')}",
        f"- Judge enabled: {report.get('judge_enabled')}",
        f"- Judge model: `{report.get('judge_model')}`",
        f"- Judge cache: `{report.get('judge_cache_path')}`",
        "",
        "## Overall Metrics",
        "",
        "| K | Recall@K | Hit@K | MRR@K | Context Recall@K | Context Precision@K | Full Node Recall | Hits |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, item in (report.get("summary") or {}).get("by_k", {}).items():
        lines.append(
            f"| {k} | {fmt(item.get('recall'))} | {fmt(item.get('hit'))} | {fmt(item.get('mrr'))} | "
            f"{fmt(item.get('context_recall'))} | {fmt(item.get('context_precision'))} | "
            f"{item.get('full_node_recall_count')} | {item.get('hit_count')} |"
        )
    lines.extend(["", "## Sample Query Details", ""])
    for item in (report.get("queries") or [])[:30]:
        k5 = item["per_k"].get("5") or item["per_k"][str(report["top_k_values"][0])]
        lines.extend(
            [
                f"### {item.get('query_id')}",
                "",
                f"- Question: {item.get('question')}",
                f"- Bundle/question type: `{item.get('bundle_type')}` / `{item.get('question_type')}`",
                f"- Gold node count: {len(item.get('gold_node_ids') or [])}",
                f"- Retrieved node ids: `{(item.get('retrieved_node_ids') or [])[:5]}`",
                f"- Recall@5: {fmt(k5.get('recall'))}",
                f"- Hit@5: {fmt(k5.get('hit'))}",
                f"- MRR@5: {fmt(k5.get('mrr'))}",
                f"- Context Recall@5: {fmt(k5.get('context_recall'))}",
                f"- Context Precision@5: {fmt(k5.get('context_precision'))}",
                "",
            ]
        )
        judge = item.get("judge") or {}
        if judge.get("claims"):
            lines.append("Claims:")
            for claim in judge.get("claims") or []:
                lines.append(f"- `{claim.get('claim_id')}` {claim.get('statement')}")
            lines.append("")
    return "\n".join(lines) + "\n"


def context_from_candidate(candidate: Any, *, rank: int) -> dict[str, Any]:
    metadata = candidate.metadata
    return {
        "rank": rank,
        "node_id": candidate.node_id,
        "text": candidate.text,
        "score": candidate.faiss_score,
        "source_id": metadata.get("source_id"),
        "source_url": metadata.get("source_url"),
        "title": metadata.get("title"),
        "jurisdiction": metadata.get("jurisdiction"),
        "policy_domain": metadata.get("policy_domain"),
        "content_type": metadata.get("content_type"),
        "doc_type": metadata.get("doc_type"),
        "can_cite_as_policy_basis": metadata.get("can_cite_as_policy_basis"),
    }


def gold_nodes(row: dict[str, Any]) -> set[str]:
    node_ids: set[str] = set()
    for group in row.get("gold_evidence_groups") or []:
        node_ids.update(
            str(node_id)
            for node_id in group.get("gold_node_ids") or []
            if str(node_id)
        )
    if node_ids:
        return node_ids
    return {
        str(ref.get("node_id"))
        for ref in row.get("gold_evidence_refs") or []
        if str(ref.get("node_id") or "")
    }


def normalize_filters(payload: object) -> dict[str, Any]:
    return PolicyRagFilters.from_mapping(payload if isinstance(payload, dict) else {}).normalized().to_dict()


def parse_top_k_values(value: str) -> list[int]:
    values = []
    for piece in str(value or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        parsed = int(piece)
        if parsed <= 0:
            raise ValueError("top K values must be positive")
        values.append(parsed)
    return sorted(set(values)) or TOP_K_VALUES


def load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {
        str(row.get("query_id") or ""): row
        for row in rows
        if str(row.get("query_id") or "")
    }


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False))
            file_obj.write("\n")


def truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]"


def average(values: Iterable[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
