"""Evaluate an experimental multi-hop group-aware Policy RAG retriever.

This is an experiment-only script. It does not change the production Policy RAG
MCP retriever or the baseline v2.1 retrieval evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.schemas import RetrievedPolicyCandidate
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
    DeepSeekContextJudge,
    append_jsonl,
    context_from_candidate,
    evaluate_query,
    load_judge_cache,
    normalize_filters,
    parse_top_k_values,
    read_jsonl,
    summarize,
    write_reports,
)


TOP_K_VALUES = [3, 5, 10]
DEFAULT_FETCH_K = 20
DEFAULT_GROUP_FETCH_K = 8
DEFAULT_RRF_K = 60
VECTOR_BACKEND_FAISS = "faiss"
VECTOR_BACKEND_QDRANT = "qdrant"
DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_1_entity_fixed_from300_audited.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_multihop_group_aware_eval_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_multihop_group_aware_eval_report.md"
)
DEFAULT_JUDGE_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "reports"
    / "policy_rag_v2_1_multihop_group_aware_judge_results.jsonl"
)

MULTIHOP_BUNDLE_TYPES = {
    "cross_doc_policy_combo",
    "case_like_policy_query",
}

DOMAIN_QUERY_HINTS = {
    "benefit": "待遇政策 起付标准 支付比例 最高支付限额 封顶线 参保身份",
    "chronic_disease_long_prescription": "慢病 长期处方 续方 家庭医生签约 开药时长",
    "designated_institution": "定点医疗机构 定点零售药店 机构名称 编码 地址 区 等级",
    "drug_catalog": "医保药品目录 药品名称 甲类 乙类 限定支付范围 备注",
    "fund_supervision": "基金监管 违规行为 重复收费 分解收费 超标准收费 经办处理",
    "manual_reimbursement": "手工报销 零星报销 材料 票据 处方 病历 办理时限",
    "medical_service_price": "医疗服务价格 项目名称 项目编码 计价单位 收费标准 项目内涵",
    "remote_medical": "异地就医 备案 直接结算 急诊抢救 就医地 参保地",
    "remote_medical_manual_reimbursement": "异地手工报销 补备案 急诊例外 报销材料 申报时限",
    "shanghai_payment_scope": "上海支付范围 药品 诊疗项目 医用耗材 甲类 乙类 自付比例",
    "special_disease_filing": "门诊特殊疾病备案 病种范围 备案材料 定点机构",
    "special_disease_scope": "门诊特殊疾病 报销范围 限定支付 条件 专科医师",
}


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    group: str
    question: str
    filters: dict[str, Any]
    weight: float


class MultiHopGroupAwareRetriever:
    """Deterministic query decomposition plus group-aware RRF merge."""

    retrieval_mode_name = "multihop_group_aware_hybrid_rrf_v1"

    def __init__(
        self,
        *,
        base_retriever: RawHybridPolicyRetriever,
        group_fetch_k: int,
        max_domain_groups: int,
        max_clause_subqueries: int,
        strict_decomposition: bool = False,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._base_retriever = base_retriever
        self._group_fetch_k = max(1, int(group_fetch_k))
        self._max_domain_groups = max(0, int(max_domain_groups))
        self._max_clause_subqueries = max(0, int(max_clause_subqueries))
        self._strict_decomposition = bool(strict_decomposition)
        self._rrf_k = max(1, int(rrf_k))

    def retrieve_for_row(
        self,
        *,
        row: dict[str, Any],
        question: str,
        filters: dict[str, Any],
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        if not should_use_multihop_strategy(row):
            return self._base_retriever.retrieve(
                question=question,
                filters=filters,
                fetch_k=fetch_k,
            )

        plans = self._build_plans(row=row, question=question, filters=filters)
        plan_results: list[tuple[RetrievalPlan, list[RetrievedPolicyCandidate]]] = []
        for plan in plans:
            candidates = self._base_retriever.retrieve(
                question=plan.question,
                filters=plan.filters,
                fetch_k=max(self._group_fetch_k, min(fetch_k, self._group_fetch_k)),
            )
            plan_results.append((plan, candidates))
        return self._merge_plans(plan_results, fetch_k=fetch_k)

    def _build_plans(
        self,
        *,
        row: dict[str, Any],
        question: str,
        filters: dict[str, Any],
    ) -> list[RetrievalPlan]:
        plans: list[RetrievalPlan] = []
        bundle_type = str(row.get("bundle_type") or "")
        domain_values = [
            str(value)
            for value in (filters.get("policy_domain") or [])
            if str(value).strip()
        ]
        clauses = split_query_clauses(question)

        if self._strict_decomposition:
            if bundle_type == "cross_doc_policy_combo":
                for domain in domain_values[: self._max_domain_groups]:
                    narrowed = dict(filters)
                    narrowed["policy_domain"] = [domain]
                    hint = DOMAIN_QUERY_HINTS.get(domain, domain)
                    plans.append(
                        RetrievalPlan(
                            group=f"domain:{domain}",
                            question=f"{question}\n检索重点：{hint}",
                            filters=narrowed,
                            weight=1.05,
                        )
                    )
                for index, clause in enumerate(
                    clauses[: max(0, min(self._max_clause_subqueries, 1))],
                    start=1,
                ):
                    plans.append(
                        RetrievalPlan(
                            group=f"clause:{index}",
                            question=f"{clause}\n原始问题：{question}",
                            filters=dict(filters),
                            weight=0.75,
                        )
                    )
            elif bundle_type == "case_like_policy_query":
                if len(domain_values) >= 2 and len(clauses) >= 2:
                    for domain in domain_values[: self._max_domain_groups]:
                        narrowed = dict(filters)
                        narrowed["policy_domain"] = [domain]
                        hint = DOMAIN_QUERY_HINTS.get(domain, domain)
                        plans.append(
                            RetrievalPlan(
                                group=f"domain:{domain}",
                                question=f"{question}\n检索重点：{hint}",
                                filters=narrowed,
                                weight=1.05,
                            )
                        )
                    for index, clause in enumerate(
                        clauses[: max(0, min(self._max_clause_subqueries, 1))],
                        start=1,
                    ):
                        plans.append(
                            RetrievalPlan(
                                group=f"clause:{index}",
                                question=f"{clause}\n原始问题：{question}",
                                filters=dict(filters),
                                weight=0.75,
                            )
                        )
        else:
            for domain in domain_values[: self._max_domain_groups]:
                narrowed = dict(filters)
                narrowed["policy_domain"] = [domain]
                hint = DOMAIN_QUERY_HINTS.get(domain, domain)
                plans.append(
                    RetrievalPlan(
                        group=f"domain:{domain}",
                        question=f"{question}\n检索重点：{hint}",
                        filters=narrowed,
                        weight=1.2,
                    )
                )

            for index, clause in enumerate(clauses[: self._max_clause_subqueries], start=1):
                plans.append(
                    RetrievalPlan(
                        group=f"clause:{index}",
                        question=f"{clause}\n原始问题：{question}",
                        filters=dict(filters),
                        weight=0.9,
                    )
                )

        plans.append(
            RetrievalPlan(
                group="base",
                question=question,
                filters=dict(filters),
                weight=1.0,
            )
        )
        return plans

    def _merge_plans(
        self,
        plan_results: list[tuple[RetrievalPlan, list[RetrievedPolicyCandidate]]],
        *,
        fetch_k: int,
    ) -> list[RetrievedPolicyCandidate]:
        candidates_by_node: dict[str, RetrievedPolicyCandidate] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        groups_by_node: defaultdict[str, set[str]] = defaultdict(set)
        ranks_by_node: defaultdict[str, list[int]] = defaultdict(list)

        for plan, candidates in plan_results:
            for rank, candidate in enumerate(candidates, start=1):
                node_id = candidate.node_id
                candidates_by_node.setdefault(node_id, candidate)
                scores[node_id] += plan.weight / (self._rrf_k + rank)
                groups_by_node[node_id].add(plan.group)
                ranks_by_node[node_id].append(rank)

        fused: list[RetrievedPolicyCandidate] = []
        for node_id, candidate in candidates_by_node.items():
            retrieval_groups = tuple(sorted(groups_by_node[node_id]))
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "retrieval_strategy": self.retrieval_mode_name,
                    "score_type": "multihop_group_rrf_score",
                    "multihop_group_rrf_score": scores[node_id],
                    "multihop_retrieval_groups": list(retrieval_groups),
                    "multihop_best_sub_rank": min(ranks_by_node[node_id]),
                    "multihop_group_count": len(retrieval_groups),
                    "rrf_k": self._rrf_k,
                }
            )
            fused.append(
                replace(
                    candidate,
                    metadata=metadata,
                    faiss_score=scores[node_id],
                    rerank_score=None,
                    retrieval_groups=retrieval_groups,
                )
            )

        fused.sort(
            key=lambda item: (
                item.faiss_score,
                item.metadata.get("multihop_group_count") or 0,
                -(item.metadata.get("multihop_best_sub_rank") or 10_000),
            ),
            reverse=True,
        )
        return group_round_robin_select(
            fused,
            group_order=[plan.group for plan, _ in plan_results],
            fetch_k=fetch_k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v2.1 Policy RAG multi-hop group-aware retrieval."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--judge-jsonl", type=Path, default=DEFAULT_JUDGE_JSONL)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--group-fetch-k", type=int, default=DEFAULT_GROUP_FETCH_K)
    parser.add_argument("--max-domain-groups", type=int, default=5)
    parser.add_argument("--max-clause-subqueries", type=int, default=2)
    parser.add_argument(
        "--strict-decomposition",
        action="store_true",
        help="Only decompose multi-hop rows when they are clearly multi-sentence, multi-domain, or cross-doc.",
    )
    parser.add_argument(
        "--vector-backend",
        choices=[VECTOR_BACKEND_FAISS, VECTOR_BACKEND_QDRANT],
        default=VECTOR_BACKEND_FAISS,
        help="Dense vector backend for the group-aware experiment.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path(os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"])
        if os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_PATH")
        else None,
        help="Use embedded local Qdrant storage when --vector-backend=qdrant.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_URL"),
        help="Use a running Qdrant HTTP service when --vector-backend=qdrant and no path is set.",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION"),
        help="Qdrant collection name when --vector-backend=qdrant.",
    )
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
    configure_vector_backend(args)
    top_k_values = parse_top_k_values(args.top_k_values)
    report = run_eval(args, top_k_values=top_k_values)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "top_k_values": top_k_values,
                "vector_backend": report["vector_backend"],
                "strict_decomposition": report["strict_decomposition"],
                "qdrant_mode": report.get("qdrant_mode"),
                "qdrant_path": report.get("qdrant_path"),
                "qdrant_url": report.get("qdrant_url"),
                "collection": report.get("collection"),
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
    total = len(eval_rows)
    multihop_count = 0
    for index, row in enumerate(eval_rows, start=1):
        started = time.perf_counter()
        filters = normalize_filters(row.get("filters"))
        if should_use_multihop_strategy(row):
            multihop_count += 1
        candidates = retriever.retrieve_for_row(
            row=row,
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
                f"[policy-rag-v2.1-multihop-eval] progress={index}/{total} "
                f"judge={'off' if args.skip_judge else 'on'}",
                flush=True,
            )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "query_count": len(query_reports),
        "fetch_k": args.fetch_k,
        "group_fetch_k": args.group_fetch_k,
        "max_domain_groups": args.max_domain_groups,
        "max_clause_subqueries": args.max_clause_subqueries,
        "strict_decomposition": args.strict_decomposition,
        "top_k_values": top_k_values,
        "retrieval_strategy": (
            f"metadata_filter_{args.vector_backend}_bge_m3_bm25_rrf_multihop_group_aware_"
            f"{'strict_v3' if args.strict_decomposition else 'v1'}"
        ),
        "vector_backend": args.vector_backend,
        "qdrant_mode": "local_path"
        if args.vector_backend == VECTOR_BACKEND_QDRANT and args.qdrant_path
        else ("url" if args.vector_backend == VECTOR_BACKEND_QDRANT else None),
        "qdrant_path": str(args.qdrant_path)
        if args.vector_backend == VECTOR_BACKEND_QDRANT and args.qdrant_path
        else None,
        "qdrant_url": args.qdrant_url
        if args.vector_backend == VECTOR_BACKEND_QDRANT and not args.qdrant_path
        else None,
        "collection": args.collection if args.vector_backend == VECTOR_BACKEND_QDRANT else None,
        "multihop_strategy_applied_count": multihop_count,
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
            "changes_production_retriever": False,
        },
    }


def build_retriever(args: argparse.Namespace) -> MultiHopGroupAwareRetriever:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    dense: Any
    if args.vector_backend == VECTOR_BACKEND_QDRANT:
        from src.backend.infrastructure.policy_rag.qdrant_retriever import (
            QdrantPolicyRetriever,
        )

        dense = QdrantPolicyRetriever(
            url=args.qdrant_url,
            collection_name=args.collection,
            embedder=embedder,
        )
    else:
        dense = FaissPolicyRetriever(embedder=embedder)
    bm25 = Bm25PolicyRetriever()
    raw = RawHybridPolicyRetriever(
        dense_retriever=dense,
        lexical_retriever=bm25,
    )
    return MultiHopGroupAwareRetriever(
        base_retriever=raw,
        group_fetch_k=args.group_fetch_k,
        max_domain_groups=args.max_domain_groups,
        max_clause_subqueries=args.max_clause_subqueries,
        strict_decomposition=args.strict_decomposition,
    )


def configure_vector_backend(args: argparse.Namespace) -> None:
    if args.vector_backend != VECTOR_BACKEND_QDRANT:
        return
    if args.qdrant_path is not None:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_PATH"] = str(args.qdrant_path)
    if args.qdrant_url and args.qdrant_path is None:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_URL"] = str(args.qdrant_url)
    if args.collection:
        os.environ["MEDIGUARD_POLICY_RAG_QDRANT_COLLECTION"] = str(args.collection)


def should_use_multihop_strategy(row: dict[str, Any]) -> bool:
    return (
        str(row.get("question_type") or "") == "multi_hop"
        or str(row.get("bundle_type") or "") in MULTIHOP_BUNDLE_TYPES
    )


def split_query_clauses(question: str) -> list[str]:
    raw_pieces = []
    buffer = ""
    for char in str(question or ""):
        buffer += char
        if char in "。；;？！?\n":
            raw_pieces.append(buffer.strip("。；;？！?\n "))
            buffer = ""
    if buffer.strip():
        raw_pieces.append(buffer.strip())

    clauses: list[str] = []
    for piece in raw_pieces:
        for sub_piece in piece.split("，"):
            clause = sub_piece.strip(" ,，")
            if len(clause) >= 10 and clause not in clauses:
                clauses.append(clause)
    return clauses


def group_round_robin_select(
    candidates: list[RetrievedPolicyCandidate],
    *,
    group_order: list[str],
    fetch_k: int,
) -> list[RetrievedPolicyCandidate]:
    selected: list[RetrievedPolicyCandidate] = []
    selected_ids: set[str] = set()
    ordered_groups = [
        group
        for group in group_order
        if group != "base"
    ] + ["base"]
    ordered_groups = list(dict.fromkeys(ordered_groups))

    while len(selected) < fetch_k:
        added = False
        for group in ordered_groups:
            candidate = next(
                (
                    item
                    for item in candidates
                    if item.node_id not in selected_ids
                    and group in set(item.metadata.get("multihop_retrieval_groups") or [])
                ),
                None,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.node_id)
            added = True
            if len(selected) >= fetch_k:
                break
        if not added:
            break

    if len(selected) < fetch_k:
        for candidate in candidates:
            if candidate.node_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.node_id)
            if len(selected) >= fetch_k:
                break
    return selected[:fetch_k]


if __name__ == "__main__":
    main()
