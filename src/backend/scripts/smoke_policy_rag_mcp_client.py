"""Smoke-test Policy RAG through a real stdio MCP client."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.backend.application.policy_rag.schemas import (
    DEFAULT_RETRIEVAL_STRATEGY,
    RETRIEVAL_PROFILES,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_CORPUS_ROOT
from src.backend.scripts.policy_rag_eval_cases import CORE_CASE_QUERIES


DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_client_smoke_report.json"
)
DEFAULT_REPORT_MD = DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_client_smoke_report.md"

DRUG_PRICE_SMOKE_QUERIES: list[dict[str, Any]] = [
    {
        "case_id": "drug_price_reference",
        "query_id": "shanghai_ibuprofen_price_reference",
        "question": "上海布洛芬缓释胶囊上周医保药店价格区间是多少？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_product_price_reference"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": False,
        },
    },
    {
        "case_id": "drug_price_reference",
        "query_id": "shanghai_atorvastatin_price_reference",
        "question": "上海阿托伐他汀钙片上周医保药店周均价是多少？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_product_price_reference"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": False,
        },
    },
    {
        "case_id": "drug_price_reference",
        "query_id": "shanghai_drug_code_price_reference",
        "question": "药品编码XC10BAY348A001010179034对应的上海上周药店价格参考是多少？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_product_price_reference"],
            "content_type": ["table_row"],
            "can_cite_as_policy_basis": False,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Policy RAG MCP client smoke tests.")
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument(
        "--retrieval-strategy",
        choices=tuple(RETRIEVAL_PROFILES),
        default=DEFAULT_RETRIEVAL_STRATEGY,
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--server-reranker", action="store_true")
    parser.add_argument("--include-drug-price-smoke", action="store_true")
    parser.add_argument("--only-drug-price-smoke", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


async def run_client_smoke(args: argparse.Namespace) -> dict[str, Any]:
    server_args = [
        "-m",
        "src.backend.mcp.policy_rag_server",
        "--retrieval-strategy",
        args.retrieval_strategy,
    ]
    if args.server_reranker:
        server_args.append("--enable-reranker")
    else:
        server_args.append("--no-reranker")
    server = StdioServerParameters(
        command=args.python,
        args=server_args,
        cwd=PROJECT_ROOT,
        env=_offline_environment(),
        encoding="utf-8",
        encoding_error_handler="replace",
    )

    started_at = time.perf_counter()
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=args.timeout_seconds),
        ) as session:
            await session.initialize()
            startup_seconds = time.perf_counter() - started_at
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            query_results = []
            queries = selected_queries(args)
            for query in queries:
                payload = {
                    "question": query["question"],
                    "filters": query["filters"],
                    "top_k": args.top_k,
                    "fetch_k": args.fetch_k,
                    "rerank": args.rerank,
                }
                query_started = time.perf_counter()
                result = await session.call_tool(
                    "policy_rag_search",
                    payload,
                    read_timeout_seconds=timedelta(seconds=args.timeout_seconds),
                )
                elapsed = time.perf_counter() - query_started
                response = _extract_tool_response(result)
                query_results.append(
                    {
                        "case_id": query["case_id"],
                        "query_id": query["query_id"],
                        "question": query["question"],
                        "filters": query["filters"],
                        "elapsed_seconds": elapsed,
                        "status": response.get("status"),
                        "result_count": response.get("result_count"),
                        "retrieval_strategy": response.get("retrieval_strategy"),
                        "retrieval_profile": response.get("retrieval_profile"),
                        "retrieval_mode": response.get("retrieval_mode"),
                        "rerank": response.get("rerank"),
                        "warnings": response.get("warnings"),
                        "structure_checks": _structure_checks(response),
                        "top_evidence": list(response.get("evidence") or [])[:2],
                    }
                )

            invalid_filters_response = _extract_tool_response(
                await session.call_tool(
                    "policy_rag_search",
                    {
                        "question": "这个请求故意不传 filters，用于验证 schema。",
                        "filters": {},
                        "top_k": 5,
                        "fetch_k": 20,
                        "rerank": False,
                    },
                    read_timeout_seconds=timedelta(seconds=args.timeout_seconds),
                )
            )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "server_command": args.python,
        "server_args": server_args,
        "startup_seconds": startup_seconds,
        "tool_names": tool_names,
        "only_policy_rag_search_tool": tool_names == ["policy_rag_search"],
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "retrieval_strategy": args.retrieval_strategy,
        "rerank_requested": args.rerank,
        "server_reranker_loaded": args.server_reranker,
        "query_count": len(query_results),
        "queries": query_results,
        "invalid_request_check": invalid_filters_response,
        "passed": bool(
            tool_names == ["policy_rag_search"]
            and invalid_filters_response.get("status") == "invalid_request"
            and all(
                item["status"] == "ok"
                and int(item.get("result_count") or 0) > 0
                and all(item["structure_checks"].values())
                for item in query_results
            )
        ),
        "boundary": {
            "uses_real_stdio_mcp_client": True,
            "integrates_case_agent": False,
            "runs_llm_generation": False,
            "makes_audit_decisions": False,
        },
    }


def _extract_tool_response(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError("Tool response did not include JSON content")


def selected_queries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.only_drug_price_smoke:
        return list(DRUG_PRICE_SMOKE_QUERIES)
    queries = list(CORE_CASE_QUERIES)
    if args.include_drug_price_smoke:
        queries.extend(DRUG_PRICE_SMOKE_QUERIES)
    return queries


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _structure_checks(response: dict[str, Any]) -> dict[str, bool]:
    evidence = list(response.get("evidence") or [])
    return {
        "has_evidence": bool(evidence),
        "has_full_text": all(bool(item.get("text")) for item in evidence),
        "has_source_id": all(bool(item.get("source_id")) for item in evidence),
        "has_source_url": all(bool(item.get("source_url")) for item in evidence),
        "has_policy_domain": all(bool(item.get("policy_domain")) for item in evidence),
        "has_filters_used": bool(response.get("filters_used")),
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
        "# Policy RAG MCP Client Smoke Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report uses a real stdio MCP client. It does not connect to Case Agent or generate audit conclusions.",
        "",
        "## Summary",
        "",
        f"- Passed: {report.get('passed')}",
        f"- Startup seconds: {report.get('startup_seconds'):.3f}",
        f"- Tools: `{report.get('tool_names')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Top K: {report.get('top_k')}",
        f"- Fetch K: {report.get('fetch_k')}",
        f"- Retrieval strategy: `{report.get('retrieval_strategy')}`",
        f"- Rerank requested: {report.get('rerank_requested')}",
        f"- Server reranker loaded: {report.get('server_reranker_loaded')}",
        "",
        "## Queries",
        "",
    ]
    for item in report.get("queries") or []:
        lines.extend(
            [
                f"### {item.get('query_id')}",
                "",
                f"- Case: {item.get('case_id')}",
                f"- Status: {item.get('status')}",
                f"- Result count: {item.get('result_count')}",
                f"- Retrieval strategy: `{item.get('retrieval_strategy')}`",
                f"- Retrieval profile: `{item.get('retrieval_profile')}`",
                f"- Retrieval mode: `{item.get('retrieval_mode')}`",
                f"- Elapsed seconds: {item.get('elapsed_seconds'):.3f}",
                f"- Structure checks: `{item.get('structure_checks')}`",
                "",
            ]
        )
        for evidence in item.get("top_evidence") or []:
            text = str(evidence.get("text") or "").replace("\n", " ")
            if len(text) > 180:
                text = text[:180] + "..."
            lines.extend(
                [
                    f"- Rank {evidence.get('rank')} | {evidence.get('source_id')} | {evidence.get('policy_domain')}",
                    f"  Source: {evidence.get('source_url')}",
                    f"  Text: {text}",
                ]
            )
        lines.append("")
    invalid = report.get("invalid_request_check") or {}
    lines.extend(
        [
            "## Invalid Request Check",
            "",
            f"- Status: {invalid.get('status')}",
            f"- Error: {invalid.get('error')}",
            f"- Message: {invalid.get('message')}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_client_smoke(args))
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "startup_seconds": report["startup_seconds"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "query_count": report["query_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
