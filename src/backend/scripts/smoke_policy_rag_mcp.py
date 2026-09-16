"""Smoke-test the Policy RAG MCP use case without starting an MCP client."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.application.policy_rag.schemas import (
    DEFAULT_FETCH_K,
    DEFAULT_RETRIEVAL_STRATEGY,
    DEFAULT_TOP_K,
    RETRIEVAL_PROFILES,
)
from src.backend.application.policy_rag.search_policy_evidence_uc import (
    SearchPolicyEvidenceUseCase,
)
from src.backend.infrastructure.policy_rag.bge_embedder import BgeQueryEmbedder
from src.backend.infrastructure.policy_rag.bge_reranker import BgePolicyReranker
from src.backend.infrastructure.policy_rag.paths import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_MODEL_CACHE,
)
from src.backend.infrastructure.policy_rag.retriever_factory import (
    build_policy_retriever,
)


DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_smoke_report.json"
)
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_smoke_report.md"

SMOKE_REQUESTS: list[dict[str, Any]] = [
    {
        "id": "beijing_manual_reimbursement",
        "question": "北京参保人门急诊手工报销需要提交哪些材料？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
    },
    {
        "id": "national_remote_principle",
        "question": "跨省异地就医为什么要区分就医地目录和参保地政策？",
        "filters": {
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical"],
        },
    },
    {
        "id": "drug_catalog",
        "question": "阿莫西林和布洛芬缓释胶囊在医保药品目录中如何核验？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["drug_catalog"],
        },
    },
    {
        "id": "shanghai_ct_price",
        "question": "上海胸部 CT 平扫医疗服务价格或诊疗项目如何核验？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["medical_service_price"],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Policy RAG MCP use-case smoke queries."
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument(
        "--retrieval-strategy",
        choices=tuple(RETRIEVAL_PROFILES),
        default=DEFAULT_RETRIEVAL_STRATEGY,
    )
    parser.add_argument(
        "--rerank",
        dest="no_rerank",
        action="store_false",
        help="Request conditional reranking and load the reranker.",
    )
    parser.add_argument(
        "--no-rerank",
        dest="no_rerank",
        action="store_true",
        help="Do not load or request the reranker. This is the default.",
    )
    parser.set_defaults(no_rerank=True)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def build_use_case(args: argparse.Namespace) -> SearchPolicyEvidenceUseCase:
    embedder = BgeQueryEmbedder(
        cache_folder=args.cache_folder,
        device=args.device,
    )
    retriever = build_policy_retriever(
        strategy=args.retrieval_strategy,
        embedder=embedder,
    )
    reranker = None
    if not args.no_rerank:
        reranker = BgePolicyReranker(
            cache_folder=args.cache_folder,
            device=args.reranker_device or args.device,
            allow_unavailable=True,
        )
    return SearchPolicyEvidenceUseCase(
        retriever=retriever,
        reranker=reranker,
        retrieval_strategy=args.retrieval_strategy,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    use_case = build_use_case(args)
    responses = []
    for request in SMOKE_REQUESTS:
        response = use_case.search(
            {
                "question": request["question"],
                "filters": request["filters"],
                "top_k": args.top_k,
                "fetch_k": args.fetch_k,
                "rerank": not args.no_rerank,
            }
        )
        responses.append(
            {
                "id": request["id"],
                "question": request["question"],
                "filters": request["filters"],
                "status": response.get("status"),
                "retrieval_strategy": response.get("retrieval_strategy"),
                "retrieval_profile": response.get("retrieval_profile"),
                "retrieval_mode": response.get("retrieval_mode"),
                "result_count": response.get("result_count"),
                "rerank": response.get("rerank"),
                "warnings": response.get("warnings"),
                "top_evidence": list(response.get("evidence") or [])[:2],
            }
        )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "retrieval_strategy": args.retrieval_strategy,
        "rerank_requested": not args.no_rerank,
        "passed": all(
            item["status"] == "ok" and int(item.get("result_count") or 0) > 0
            for item in responses
        ),
        "requests": responses,
        "boundary": {
            "uses_mcp_use_case": True,
            "starts_mcp_stdio_server": False,
            "runs_llm_generation": False,
            "integrates_case_agent": False,
            "makes_audit_decisions": False,
        },
    }


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    report_md.write_text(build_markdown_report(report), encoding="utf-8", newline="\n")


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy RAG MCP Smoke Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report validates the Policy RAG MCP application use case. It does not generate audit conclusions or connect to Case Agent.",
        "",
        "## Summary",
        "",
        f"- Passed: {report.get('passed')}",
        f"- Rerank requested: {report.get('rerank_requested')}",
        f"- Top K: {report.get('top_k')}",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Retrieval strategy: `{report.get('retrieval_strategy', 'dense')}`",
        "",
        "## Requests",
        "",
    ]
    for item in report.get("requests") or []:
        rerank = item.get("rerank") or {}
        lines.extend(
            [
                f"### {item.get('id')}",
                "",
                f"- Question: {item.get('question')}",
                f"- Filters: `{json.dumps(item.get('filters'), ensure_ascii=False)}`",
                f"- Status: {item.get('status')}",
                f"- Retrieval strategy: `{item.get('retrieval_strategy')}`",
                f"- Retrieval profile: `{item.get('retrieval_profile')}`",
                f"- Retrieval mode: `{item.get('retrieval_mode')}`",
                f"- Result count: {item.get('result_count')}",
                f"- Rerank applied: {rerank.get('applied')}",
                f"- Warnings: {item.get('warnings')}",
                "",
            ]
        )
        for evidence in item.get("top_evidence") or []:
            text = str(evidence.get("text") or "").replace("\n", " ")
            if len(text) > 220:
                text = text[:220] + "..."
            lines.extend(
                [
                    f"- Rank {evidence.get('rank')} | score={evidence.get('score')} | {evidence.get('title')}",
                    f"  Source: {evidence.get('source_url')}",
                    f"  Text: {text}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- Retrieval only.",
            "- No L1/L2 case data is read.",
            "- No final reimbursement, refusal, punishment, or fraud decision is made.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = run_smoke(args)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "requests": [
                    {
                        "id": item["id"],
                        "status": item["status"],
                        "result_count": item["result_count"],
                        "retrieval_mode": item["retrieval_mode"],
                        "rerank": item["rerank"],
                    }
                    for item in report["requests"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
