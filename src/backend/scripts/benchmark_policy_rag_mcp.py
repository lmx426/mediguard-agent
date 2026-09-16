"""Single-process Policy RAG MCP performance baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
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

from src.backend.infrastructure.policy_rag.paths import DEFAULT_CORPUS_ROOT
from src.backend.scripts.policy_rag_eval_cases import CORE_CASE_QUERIES


DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_performance_baseline.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_rag_mcp_performance_baseline.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Policy RAG MCP hot queries.")
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dense_report = await _run_session(
        python=args.python,
        server_reranker=False,
        scenarios=[
            {"name": "dense_only_fetch20", "rerank": False, "fetch_k": 20, "top_k": 5}
        ],
        timeout_seconds=args.timeout_seconds,
    )
    rerank_report = await _run_session(
        python=args.python,
        server_reranker=True,
        scenarios=[
            {"name": "rerank_fetch20", "rerank": True, "fetch_k": 20, "top_k": 5},
            {"name": "rerank_fetch40", "rerank": True, "fetch_k": 40, "top_k": 5},
        ],
        timeout_seconds=args.timeout_seconds,
    )
    sessions = [dense_report, rerank_report]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "case_query_count": len(CORE_CASE_QUERIES),
        "sessions": sessions,
        "passed": all(session["passed"] for session in sessions),
        "boundary": {
            "uses_real_stdio_mcp_client": True,
            "runs_concurrency_test": False,
            "calculates_token_usage": False,
            "runs_llm_generation": False,
            "makes_audit_decisions": False,
        },
    }


async def _run_session(
    *,
    python: str,
    server_reranker: bool,
    scenarios: list[dict[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    server_args = ["-m", "src.backend.mcp.policy_rag_server"]
    if not server_reranker:
        server_args.append("--no-reranker")
    server = StdioServerParameters(
        command=python,
        args=server_args,
        cwd=PROJECT_ROOT,
        env=_offline_environment(),
        encoding="utf-8",
        encoding_error_handler="replace",
    )
    started = time.perf_counter()
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=timeout_seconds),
        ) as session:
            await session.initialize()
            startup_seconds = time.perf_counter() - started
            scenario_reports = []
            for scenario in scenarios:
                measurements = []
                for query in CORE_CASE_QUERIES:
                    payload = {
                        "question": query["question"],
                        "filters": query["filters"],
                        "top_k": scenario["top_k"],
                        "fetch_k": scenario["fetch_k"],
                        "rerank": scenario["rerank"],
                    }
                    query_started = time.perf_counter()
                    result = await session.call_tool(
                        "policy_rag_search",
                        payload,
                        read_timeout_seconds=timedelta(seconds=timeout_seconds),
                    )
                    elapsed = time.perf_counter() - query_started
                    response = _extract_tool_response(result)
                    measurements.append(
                        {
                            "case_id": query["case_id"],
                            "query_id": query["query_id"],
                            "elapsed_seconds": elapsed,
                            "status": response.get("status"),
                            "result_count": response.get("result_count"),
                            "retrieval_mode": response.get("retrieval_mode"),
                            "rerank": response.get("rerank"),
                            "warnings": response.get("warnings"),
                        }
                    )
                scenario_reports.append(_scenario_summary(scenario, measurements))
    return {
        "server_reranker_loaded": server_reranker,
        "server_args": server_args,
        "startup_seconds": startup_seconds,
        "scenarios": scenario_reports,
        "passed": all(scenario["passed"] for scenario in scenario_reports),
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


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _scenario_summary(
    scenario: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    elapsed = [float(item["elapsed_seconds"]) for item in measurements]
    return {
        "name": scenario["name"],
        "top_k": scenario["top_k"],
        "fetch_k": scenario["fetch_k"],
        "rerank": scenario["rerank"],
        "query_count": len(measurements),
        "avg_seconds": statistics.fmean(elapsed),
        "p50_seconds": _percentile(elapsed, 50),
        "p95_seconds": _percentile(elapsed, 95),
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "error_count": sum(1 for item in measurements if item.get("status") != "ok"),
        "all_result_counts": [item.get("result_count") for item in measurements],
        "measurements": measurements,
        "passed": all(
            item.get("status") == "ok" and item.get("result_count") == scenario["top_k"]
            for item in measurements
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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
        "# Policy RAG MCP Performance Baseline",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report measures cold startup and single-process hot queries. It does not run concurrency tests or LLM generation.",
        "",
        f"Passed: {report.get('passed')}",
        "",
    ]
    for session in report.get("sessions") or []:
        lines.extend(
            [
                "## Session",
                "",
                f"- Server reranker loaded: {session.get('server_reranker_loaded')}",
                f"- Startup seconds: {session.get('startup_seconds'):.3f}",
                f"- Passed: {session.get('passed')}",
                "",
            ]
        )
        for scenario in session.get("scenarios") or []:
            lines.extend(
                [
                    f"### {scenario.get('name')}",
                    "",
                    f"- Rerank: {scenario.get('rerank')}",
                    f"- Top K: {scenario.get('top_k')}",
                    f"- Fetch K: {scenario.get('fetch_k')}",
                    f"- Avg seconds: {scenario.get('avg_seconds'):.3f}",
                    f"- P50 seconds: {scenario.get('p50_seconds'):.3f}",
                    f"- P95 seconds: {scenario.get('p95_seconds'):.3f}",
                    f"- Min / Max seconds: {scenario.get('min_seconds'):.3f} / {scenario.get('max_seconds'):.3f}",
                    f"- Error count: {scenario.get('error_count')}",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
                "sessions": [
                    {
                        "server_reranker_loaded": session["server_reranker_loaded"],
                        "startup_seconds": session["startup_seconds"],
                        "scenarios": [
                            {
                                "name": scenario["name"],
                                "avg_seconds": scenario["avg_seconds"],
                                "p50_seconds": scenario["p50_seconds"],
                                "p95_seconds": scenario["p95_seconds"],
                                "passed": scenario["passed"],
                            }
                            for scenario in session["scenarios"]
                        ],
                    }
                    for session in report["sessions"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
