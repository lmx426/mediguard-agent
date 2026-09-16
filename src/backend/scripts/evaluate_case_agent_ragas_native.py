"""Evaluate Case Agent answers with native Ragas Collections metrics.

This is an offline-only evaluator. It drives the existing Case Agent service
path, captures the answer and the contexts actually used by that run, and
calls the official Ragas ``ascore`` APIs. There is intentionally no formula
judge or metric fallback in this module.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HF_CACHE = PROJECT_ROOT / "runtime" / "hf_cache"
if DEFAULT_HF_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.backend.api.dependencies import build_container
from src.backend.core.config import Settings
from src.backend.domain.audit.review.entities import AuthenticatedUser
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentCreateSessionInput,
    CaseAgentSendMessageInput,
)
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    load_env_file,
    read_jsonl,
)


DEFAULT_GOLDSET = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "eval"
    / "policy_case_agent_ragas_goldset_197.jsonl"
)
DEFAULT_INPUTS = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "eval"
    / "policy_case_agent_ragas_native_inputs.jsonl"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "eval"
    / "policy_case_agent_ragas_native_results.jsonl"
)
DEFAULT_REPORT_JSON = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "reports"
    / "policy_case_agent_ragas_native_report.json"
)
DEFAULT_REPORT_MD = (
    PROJECT_ROOT
    / "src"
    / "backend"
    / "policy_corpus"
    / "reports"
    / "policy_case_agent_ragas_native_report.md"
)

EXPECTED_GOLDSET_FIELDS = {"query_id", "user_input", "reference"}
EXPECTED_GOLDSET_COUNT = 197
EVALUABLE_RUN_STATUSES = {"completed", "degraded"}
TERMINAL_STATUSES = {
    "waiting_for_user",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "degraded",
}
CHECKPOINT_CONTEXT_NODES = (
    "validate_answer",
    "generate_answer",
    "resolve_answer_style",
    "resolve_answer_policy",
    "build_answer_context",
    "call_capabilities",
)
EXPERT_CHECKPOINT_CONTEXT_NODES = (
    "expert_analysis:policy_tool_adapter",
    "expert_analysis:hard_gate",
    "expert_analysis:slot_window_judge",
    "expert_analysis:answerability_check",
    "expert_analysis:expert_synthesis",
    "expert_analysis:return_unavailable",
    "expert_analysis:follow_up_retrieval",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Case Agent generation with native Ragas metrics."
    )
    parser.add_argument("--goldset-jsonl", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--ragas-inputs-jsonl", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--actor-username", default="default_auditor")
    parser.add_argument("--active-stage", default="evidence_review")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--response-scope", choices=["primary", "full"], default="primary")
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Device for the Ragas embedding model; default cpu avoids CUDA OOM on local eval runs.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-sessions", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success even when rows are not evaluable or a metric errors.",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    write_reports(report, args.report_json, args.report_md)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "query_count": report["query_count"],
                "answer_correctness_scored": summary["answer_correctness"]["scored_count"],
                "faithfulness_scored": summary["faithfulness"]["scored_count"],
                "metric_backend": report["metric_backend"],
                "ragas_version": report["ragas_version"],
                "results_jsonl": str(args.results_jsonl),
                "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "ok" and not args.allow_partial:
        raise SystemExit(2)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    goldset = load_and_validate_goldset(args.goldset_jsonl)
    selected = goldset[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.no_resume:
        reset_jsonl(args.results_jsonl)
        reset_jsonl(args.ragas_inputs_jsonl)
    existing = load_existing_results(args.results_jsonl) if not args.no_resume else {}

    ragas_version = installed_ragas_version()
    scorer = NativeRagasScorer.from_environment(
        env_file=args.env_file,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        timeout=args.judge_timeout,
        temperature=args.judge_temperature,
    )

    container = None
    results: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        container = build_runtime_container()
        service = container.case_agent
        if service is None:
            raise RuntimeError(
                "Case Agent is not available: "
                + json.dumps(container.case_agent_status, ensure_ascii=False)
            )
        actor = resolve_actor(container, args.actor_username)
        case_id = resolve_case_id(container, args.case_id)

        for index, row in enumerate(selected, start=1):
            query_id = row["query_id"]
            if query_id in existing:
                result = existing[query_id]
                results.append(result)
                if isinstance(result.get("ragas_input"), dict):
                    inputs.append(result["ragas_input"])
                print_progress(args, index, len(selected), result, resumed=True)
                continue

            result = evaluate_one_row(
                args,
                row=row,
                service=service,
                actor=actor,
                case_id=case_id,
                scorer=scorer,
            )
            results.append(result)
            inputs.append(result["ragas_input"])
            append_jsonl(args.results_jsonl, [result])
            append_jsonl(args.ragas_inputs_jsonl, [result["ragas_input"]])
            print_progress(args, index, len(selected), result, resumed=False)
    finally:
        shutdown_container(container)

    summary = summarize(results)
    status = "ok" if summary["non_scored_metric_count"] == 0 else "partial"
    return {
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "goldset_jsonl": str(args.goldset_jsonl),
        "results_jsonl": str(args.results_jsonl),
        "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
        "query_count": len(results),
        "offset": args.offset,
        "limit": args.limit,
        "response_scope": args.response_scope,
        "metric_backend": "ragas_native",
        "ragas_version": ragas_version,
        "judge_model": scorer.judge_model,
        "embedding_model": scorer.embedding_model,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
        "boundary": {
            "uses_three_field_goldset": True,
            "drives_case_agent_service": True,
            "uses_actual_agent_response": True,
            "uses_actual_policy_contexts": True,
            "injects_gold_contexts": False,
            "uses_native_ragas_collections": True,
            "allows_formula_fallback": False,
            "makes_audit_decisions": False,
        },
    }


def load_and_validate_goldset(path: Path) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_GOLDSET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GOLDSET_COUNT} Goldset rows, received {len(rows)}."
        )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if set(row) != EXPECTED_GOLDSET_FIELDS:
            raise ValueError(
                f"Goldset row {index} must contain exactly "
                "query_id, user_input, reference."
            )
        query_id = str(row["query_id"] or "").strip()
        user_input = str(row["user_input"] or "").strip()
        reference = str(row["reference"] or "").strip()
        if not query_id or query_id in seen:
            raise ValueError(f"Invalid or duplicate query_id at row {index}: {query_id}")
        if not user_input:
            raise ValueError(f"Empty user_input at row {query_id}")
        if not reference:
            raise ValueError(f"Empty reference at row {query_id}")
        seen.add(query_id)
        result.append(
            {"query_id": query_id, "user_input": user_input, "reference": reference}
        )
    return result


def build_runtime_container() -> Any:
    settings = Settings()
    container = build_container(settings)
    container.fixtures.load_all()
    return container


def resolve_actor(container: Any, username: str) -> AuthenticatedUser:
    user = container.users.get_by_username(username)
    if user is None:
        raise RuntimeError(f"Auditor user not found: {username}")
    return user.to_authenticated()


def resolve_case_id(container: Any, case_id: str) -> str:
    if case_id:
        return case_id
    cases = container.list_cases.execute()
    if not cases:
        raise RuntimeError("No cases available for Case Agent evaluation.")
    return str(cases[0].case_id)


def evaluate_one_row(
    args: argparse.Namespace,
    *,
    row: dict[str, str],
    service: Any,
    actor: AuthenticatedUser,
    case_id: str,
    scorer: "NativeRagasScorer",
) -> dict[str, Any]:
    query_id = row["query_id"]
    started = time.perf_counter()
    session_id = ""
    run_id = ""
    run_status = "not_started"
    response = ""
    contexts: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    try:
        session = service.create_session(
            case_id,
            CaseAgentCreateSessionInput(title=f"Native Ragas {query_id}"),
            actor,
        )
        session_id = str(session.session_id)
        run = service.send_message(
            session.session_id,
            CaseAgentSendMessageInput(
                content=row["user_input"],
                active_stage=args.active_stage,
            ),
            actor,
        )
        run_id = str(run.run_id)
        run = wait_for_run(
            service,
            run_id=run_id,
            actor=actor,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
        )
        run_status = str(run.status)
        answer = load_run_answer(service, session_id=session_id, run=run, actor=actor)
        response = answer_text(answer, scope=args.response_scope)
        contexts = collect_policy_contexts(service, run_id=run_id, answer=answer)
    except Exception as exc:
        error = {"type": exc.__class__.__name__, "message": str(exc)[:500]}

    ragas_input = {
        "query_id": query_id,
        "user_input": row["user_input"],
        "reference": row["reference"],
        "response": response,
        "retrieved_contexts": context_texts(contexts),
    }

    if error is not None:
        metrics = {
            "answer_correctness": metric_not_evaluable("agent_run_failed"),
            "faithfulness": metric_not_evaluable("agent_run_failed"),
        }
        status = "agent_error"
    elif run_status not in EVALUABLE_RUN_STATUSES or not response:
        reason = "empty_agent_response" if not response else f"run_status_{run_status}"
        metrics = {
            "answer_correctness": metric_not_evaluable(reason),
            "faithfulness": metric_not_evaluable(reason),
        }
        status = "not_evaluable"
    else:
        metrics = scorer.score_sync(
            user_input=row["user_input"],
            response=response,
            reference=row["reference"],
            retrieved_contexts=ragas_input["retrieved_contexts"],
        )
        status = "scored" if any(
            metric.get("status") == "scored" for metric in metrics.values()
        ) else "metric_error"

    result = {
        "schema_version": "case_agent_ragas_native_eval_v1",
        "query_id": query_id,
        "status": status,
        "run_status": run_status,
        "case_id": case_id,
        "session_id": session_id,
        "run_id": run_id,
        "response": response,
        "retrieved_context_count": len(ragas_input["retrieved_contexts"]),
        "retrieved_node_ids": [str(context.get("node_id") or "") for context in contexts if str(context.get("node_id") or "").strip()],
        "ragas_input": ragas_input,
        "metrics": metrics,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if session_id and not args.keep_sessions:
        try:
            service.archive_session(session_id, actor)
        except Exception:
            pass
    return result


def wait_for_run(
    service: Any,
    *,
    run_id: str,
    actor: AuthenticatedUser,
    timeout_seconds: float,
    poll_interval: float,
) -> Any:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    run = service.get_run(run_id, actor)
    while run.status not in TERMINAL_STATUSES:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Case Agent run timed out: {run_id}")
        time.sleep(max(0.2, poll_interval))
        run = service.get_run(run_id, actor)
    return run


def load_run_answer(
    service: Any,
    *,
    session_id: str,
    run: Any,
    actor: AuthenticatedUser,
) -> CaseAgentAnswer | None:
    session = service.get_session(session_id, actor)
    assistant_id = getattr(run, "assistant_message_id", None)
    if assistant_id:
        for message in session.messages:
            if message.message_id == assistant_id:
                return message.answer_payload
    for message in reversed(session.messages):
        if message.role == "assistant":
            return message.answer_payload
    return None


def answer_text(answer: CaseAgentAnswer | None, *, scope: str) -> str:
    if answer is None:
        return ""
    if scope == "primary" and answer.content_blocks:
        return str(answer.content_blocks[0].text or "").strip()
    return answer.plain_text.strip()


def collect_policy_contexts(
    service: Any,
    *,
    run_id: str,
    answer: CaseAgentAnswer | None,
) -> list[dict[str, Any]]:
    contexts = contexts_from_sql_tool_calls(service, run_id=run_id)
    if not contexts:
        contexts = contexts_from_checkpoints(service, run_id=run_id)
    contexts = dedup_contexts(contexts)
    if answer is not None:
        source_refs = answer_source_refs(answer)
        if source_refs:
            cited = [
                context
                for context in contexts
                if context.get("source_ref") in source_refs
                or context.get("evidence_ref") in source_refs
            ]
            if cited:
                contexts = cited
    return [{**context, "rank": index} for index, context in enumerate(contexts[:12], 1)]


def contexts_from_sql_tool_calls(service: Any, *, run_id: str) -> list[dict[str, Any]]:
    repository = getattr(service, "_repository", None)
    session_factory = getattr(repository, "_session_factory", None)
    if not callable(session_factory):
        return []
    try:
        from sqlalchemy import select
        from src.backend.infrastructure.persistence.sql.models import (
            CaseAgentRunORM,
            CaseAgentToolCallORM,
        )

        with session_factory() as session:
            run_row = session.scalar(
                select(CaseAgentRunORM).where(CaseAgentRunORM.run_id == run_id)
            )
            if run_row is None:
                return []
            rows = session.scalars(
                select(CaseAgentToolCallORM)
                .where(CaseAgentToolCallORM.run_id == run_row.id)
                .order_by(CaseAgentToolCallORM.sequence)
            ).all()
            contexts: list[dict[str, Any]] = []
            for row in rows:
                if row.tool_name == "ask_policy_expert":
                    contexts.extend(contexts_from_expert_payload(row.result_summary))
            return contexts
    except Exception:
        return []


def contexts_from_expert_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = [
        payload.get("policy_evidence"),
        payload.get("adopted_policy_evidence"),
    ]
    wrapped = payload.get("payload")
    if isinstance(wrapped, dict):
        candidates.extend(
            [
                wrapped.get("policy_evidence"),
                wrapped.get("adopted_policy_evidence"),
            ]
        )
    contexts: list[dict[str, Any]] = []
    for evidence in candidates:
        contexts.extend(contexts_from_policy_evidence(evidence))
    return contexts


def contexts_from_checkpoints(service: Any, *, run_id: str) -> list[dict[str, Any]]:
    repository = getattr(service, "_repository", None)
    get_latest_checkpoint = getattr(repository, "get_latest_checkpoint", None)
    if not callable(get_latest_checkpoint):
        return []
    snapshots: list[dict[str, Any]] = []
    for node_name in (*CHECKPOINT_CONTEXT_NODES, *EXPERT_CHECKPOINT_CONTEXT_NODES):
        snapshot = get_latest_checkpoint(run_id, node_name=node_name)
        if isinstance(snapshot, dict):
            snapshots.append(snapshot)
    latest = get_latest_checkpoint(run_id)
    if isinstance(latest, dict):
        snapshots.append(latest)

    contexts: list[dict[str, Any]] = []
    for snapshot in snapshots:
        source_by_ref = sources_by_ref(snapshot.get("available_sources"))
        contexts.extend(
            contexts_from_answer_context(
                snapshot.get("answer_context"), source_by_ref=source_by_ref
            )
        )
        contexts.extend(contexts_from_capability_results(snapshot.get("capability_results")))
        contexts.extend(
            contexts_from_policy_evidence(
                snapshot.get("adopted_policy_evidence"),
                source_by_ref=source_by_ref,
            )
        )
    return contexts


def contexts_from_policy_evidence(
    value: Any,
    *,
    source_by_ref: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    contexts: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_ref = str(item.get("source_ref") or item.get("evidence_ref") or "")
        source = source_by_ref.get(source_ref, {}) if source_by_ref else {}
        source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        text = str(
            item.get("excerpt")
            or item.get("text")
            or item.get("content")
            or item.get("window_text")
            or ""
        ).strip()
        if not text:
            continue
        contexts.append(
            {
                "rank": index,
                "source_ref": source_ref,
                "evidence_ref": str(item.get("evidence_ref") or source_ref),
                "node_id": metadata.get("node_id") or source_metadata.get("node_id"),
                "source_id": metadata.get("source_id") or source_metadata.get("source_id"),
                "source_url": item.get("source_url")
                or metadata.get("source_url")
                or source_metadata.get("source_url"),
                "title": item.get("title") or source.get("title"),
                "text": text,
            }
        )
    return contexts


def contexts_from_answer_context(
    value: Any,
    *,
    source_by_ref: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    contexts: list[dict[str, Any]] = []
    for section in value.get("analysis_inputs") or []:
        if not isinstance(section, dict) or section.get("section") != "政策专家分析":
            continue
        for index, item in enumerate(section.get("policy_evidence") or [], 1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("excerpt") or "").strip()
            if not text:
                continue
            source_ref = str(item.get("source_ref") or "")
            source = source_by_ref.get(source_ref, {})
            metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
            contexts.append(
                {
                    "rank": index,
                    "source_ref": source_ref,
                    "evidence_ref": metadata.get("evidence_ref") or source_ref,
                    "node_id": metadata.get("node_id"),
                    "source_id": metadata.get("source_id"),
                    "source_url": metadata.get("source_url"),
                    "title": item.get("title") or source.get("title"),
                    "text": text,
                }
            )
    return contexts


def sources_by_ref(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(item["source_ref"]): item
        for item in value
        if isinstance(item, dict) and item.get("source_ref")
    }


def contexts_from_capability_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    contexts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("capability") != "ask_policy_expert":
            continue
        payload = item.get("payload_summary")
        if not isinstance(payload, dict):
            continue
        contexts.extend(contexts_from_expert_payload(payload.get("payload") or payload))
    return contexts


def answer_source_refs(answer: CaseAgentAnswer) -> set[str]:
    refs: list[str] = []
    for block in answer.content_blocks:
        refs.extend(block.source_refs)
    refs.extend(source.source_ref for source in answer.sources)
    return {ref for ref in refs if ref}


def dedup_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for context in contexts:
        text = str(context.get("text") or "").strip()
        if not text:
            continue
        key = (str(context.get("node_id") or context.get("source_ref") or ""), text[:160])
        if key in seen:
            continue
        seen.add(key)
        output.append(context)
    return output


def context_texts(contexts: list[dict[str, Any]]) -> list[str]:
    return [
        text
        for context in contexts
        if (text := str(context.get("text") or context.get("excerpt") or "").strip())
    ]


class NativeRagasScorer:
    """Thin adapter around the official Ragas Collections metrics."""

    def __init__(
        self,
        *,
        llm: Any,
        answer_correctness: Any,
        faithfulness: Any,
        judge_model: str,
        embedding_model: str,
        timeout: float,
    ) -> None:
        self._answer_correctness = answer_correctness
        self._faithfulness = faithfulness
        self.judge_model = judge_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self._llm = llm

    @classmethod
    def from_environment(
        cls,
        *,
        env_file: Path,
        judge_model: str,
        embedding_model: str,
        embedding_device: str | None,
        timeout: float,
        temperature: float,
    ) -> "NativeRagasScorer":
        load_env_file(env_file)
        api_key = first_env(
            "MEDIGUARD_RAGAS_EVAL_API_KEY",
            "MEDIGUARD_RAGAS_JUDGE_API_KEY",
            "MEDIGUARD_POLICY_RAG_EVAL_DEEPSEEK_API_KEY",
            "MEDIGUARD_DEEPSEEK_API_KEY",
        )
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key is missing. Set MEDIGUARD_RAGAS_EVAL_API_KEY."
            )
        base_url = (
            first_env(
                "MEDIGUARD_RAGAS_EVAL_BASE_URL",
                "MEDIGUARD_RAGAS_JUDGE_BASE_URL",
                "MEDIGUARD_POLICY_RAG_EVAL_BASE_URL",
            )
            or "https://api.deepseek.com"
        )
        model = (
            judge_model
            or first_env(
                "MEDIGUARD_RAGAS_EVAL_MODEL",
                "MEDIGUARD_RAGAS_JUDGE_MODEL",
            )
            or "deepseek-chat"
        )
        validate_judge_model(model)
        require_ragas_04()

        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerCorrectness, Faithfulness

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )
        ragas_llm = llm_factory(
            model,
            provider="openai",
            client=client,
            temperature=temperature,
        )
        embeddings = HuggingFaceEmbeddings(
            model=resolve_embedding_model(embedding_model),
            device=embedding_device or None,
        )
        return cls(
            llm=ragas_llm,
            answer_correctness=AnswerCorrectness(
                llm=ragas_llm,
                embeddings=embeddings,
            ),
            faithfulness=Faithfulness(llm=ragas_llm),
            judge_model=model,
            embedding_model=embedding_model,
            timeout=timeout,
        )

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[str, dict[str, Any]]:
        answer_correctness = await self._score_answer_correctness(
            user_input=user_input,
            response=response,
            reference=reference,
        )
        faithfulness = await self._score_faithfulness(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        return {
            "answer_correctness": answer_correctness,
            "faithfulness": faithfulness,
        }

    async def _score_answer_correctness(
        self, *, user_input: str, response: str, reference: str
    ) -> dict[str, Any]:
        try:
            result = await self._answer_correctness.ascore(
                user_input=user_input,
                response=response,
                reference=reference,
            )
            return metric_scored(result)
        except Exception as exc:
            return metric_error(exc)

    async def _score_faithfulness(
        self, *, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> dict[str, Any]:
        if not retrieved_contexts:
            return metric_not_evaluable("no_retrieved_contexts")
        try:
            result = await self._faithfulness.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            return metric_scored(result)
        except Exception as exc:
            return metric_error(exc)

    def score_sync(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        return asyncio.run(self.score(**kwargs))


def validate_judge_model(model: str) -> None:
    lowered = model.lower()
    risky_markers = ("reasoner", "thinking", "r1", "v4")
    if any(marker in lowered for marker in risky_markers):
        raise RuntimeError(
            f"Native Ragas requires a non-thinking DeepSeek judge; refusing model {model!r}. "
            "Use deepseek-chat for MEDIGUARD_RAGAS_EVAL_MODEL."
        )


def resolve_embedding_model(model: str) -> str:
    """Prefer the repository's immutable local snapshot for offline evaluation."""
    if model != "BAAI/bge-m3":
        return model
    refs = DEFAULT_HF_CACHE / "models--BAAI--bge-m3" / "refs" / "main"
    if not refs.exists():
        return model
    revision = refs.read_text(encoding="utf-8").strip()
    snapshot = DEFAULT_HF_CACHE / "models--BAAI--bge-m3" / "snapshots" / revision
    return str(snapshot) if snapshot.exists() else model


def require_ragas_04() -> None:
    version = installed_ragas_version()
    parts = version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if (major, minor) < (0, 4):
        raise RuntimeError(
            f"Native Ragas evaluator requires ragas>=0.4; installed {version}."
        )


def installed_ragas_version() -> str:
    try:
        ensure_ragas_import_compatibility()
        import ragas

        return str(ragas.__version__)
    except Exception as exc:
        raise RuntimeError("Ragas is not installed in the evaluation environment.") from exc


def ensure_ragas_import_compatibility() -> None:
    """Bridge Ragas 0.4.3's legacy optional Vertex AI import.

    Ragas 0.4.3 imports ``langchain_community.chat_models.vertexai`` while
    initializing its LLM base module. Recent ``langchain-community`` wheels no
    longer ship that compatibility module. This evaluator uses the OpenAI
    adapter for DeepSeek, so a marker class is sufficient for Ragas' type
    registry and does not participate in metric scoring.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801 - matches the upstream class name
        pass

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module


def metric_scored(result: Any) -> dict[str, Any]:
    value = getattr(result, "value", None)
    if value is None or not math.isfinite(float(value)):
        return metric_error(ValueError("Ragas returned a non-finite MetricResult.value."))
    return {"status": "scored", "value": float(value)}


def metric_not_evaluable(reason: str) -> dict[str, Any]:
    return {"status": "not_evaluable", "reason": reason, "value": None}


def metric_error(exc: Exception) -> dict[str, Any]:
    return {
        "status": "metric_error",
        "reason": "native_ragas_call_failed",
        "error_type": exc.__class__.__name__,
        "error_message": str(exc)[:500],
        "value": None,
    }


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row.get("query_id")): row
        for row in read_jsonl(path)
        if row.get("schema_version") == "case_agent_ragas_native_eval_v1"
        and row.get("query_id")
    }


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "unknown") for row in results)
    run_status_counts = Counter(str(row.get("run_status") or "unknown") for row in results)
    metric_error_count = sum(
        1
        for row in results
        for metric in (row.get("metrics") or {}).values()
        if isinstance(metric, dict) and metric.get("status") == "metric_error"
    )
    not_evaluable_count = sum(
        1
        for row in results
        for metric in (row.get("metrics") or {}).values()
        if isinstance(metric, dict) and metric.get("status") == "not_evaluable"
    )
    return {
        "row_status_counts": dict(status_counts),
        "run_status_counts": dict(run_status_counts),
        "blocking_error_count": int(status_counts.get("agent_error", 0) + metric_error_count),
        "metric_error_count": metric_error_count,
        "not_evaluable_count": not_evaluable_count,
        "non_scored_metric_count": metric_error_count + not_evaluable_count,
        "answer_correctness": summarize_metric(results, "answer_correctness"),
        "faithfulness": summarize_metric(results, "faithfulness"),
        "retrieved_context_count_avg": average(
            row.get("retrieved_context_count") for row in results
        ),
    }


def summarize_metric(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [
        float((row.get("metrics") or {}).get(name, {}).get("value"))
        for row in results
        if (row.get("metrics") or {}).get(name, {}).get("status") == "scored"
        and (row.get("metrics") or {}).get(name, {}).get("value") is not None
    ]
    return {
        "scored_count": len(values),
        "not_evaluable_count": sum(
            1
            for row in results
            if (row.get("metrics") or {}).get(name, {}).get("status") == "not_evaluable"
        ),
        "metric_error_count": sum(
            1
            for row in results
            if (row.get("metrics") or {}).get(name, {}).get("status") == "metric_error"
        ),
        "average": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def average(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def print_progress(
    args: argparse.Namespace,
    index: int,
    total: int,
    result: dict[str, Any],
    *,
    resumed: bool,
) -> None:
    every = max(1, int(args.progress_every))
    if index != total and index % every:
        return
    metrics = result.get("metrics") or {}
    ac = metrics.get("answer_correctness", {}).get("value")
    faith = metrics.get("faithfulness", {}).get("value")
    print(
        f"[native-ragas] {index}/{total} query={result.get('query_id')} "
        f"status={result.get('status')} ac={fmt(ac)} faith={fmt(faith)} "
        f"resumed={resumed}",
        flush=True,
    )


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_md.write_text(build_markdown_report(report), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    ac = summary["answer_correctness"]
    faith = summary["faithfulness"]
    return "\n".join(
        [
            "# Case Agent Native Ragas Eval Report",
            "",
            f"- Status: `{report['status']}`",
            f"- Goldset: `{report['goldset_jsonl']}`",
            f"- Query count: {report['query_count']}",
            f"- Ragas version: `{report['ragas_version']}`",
            f"- Judge model: `{report['judge_model']}`",
            f"- Embedding model: `{report['embedding_model']}`",
            f"- Metric backend: `{report['metric_backend']}`",
            "",
            "## Metrics",
            "",
            "| Metric | Scored | Not evaluable | Errors | Average | Median | Stdev |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Answer Correctness | {ac['scored_count']} | {ac['not_evaluable_count']} | {ac['metric_error_count']} | {fmt(ac['average'])} | {fmt(ac['median'])} | {fmt(ac['stdev'])} |",
            f"| Faithfulness | {faith['scored_count']} | {faith['not_evaluable_count']} | {faith['metric_error_count']} | {fmt(faith['average'])} | {fmt(faith['median'])} | {fmt(faith['stdev'])} |",
            "",
            "## Run Status",
            "",
            f"- Row status counts: `{summary['row_status_counts']}`",
            f"- Run status counts: `{summary['run_status_counts']}`",
            f"- Native Ragas metric errors: {summary['metric_error_count']}",
            f"- Average retrieved context count: {fmt(summary['retrieved_context_count_avg'])}",
            "",
            "## Boundary",
            "",
            "- Goldset contains only `query_id`, `user_input`, and `reference`.",
            "- `response` and `retrieved_contexts` come from the actual Case Agent run.",
            "- Faithfulness and Answer Correctness use official Collections `ascore` calls.",
            "- No custom formula judge or metric fallback is used.",
            "- This report does not make audit, payment, punishment, or fraud decisions.",
            "",
        ]
    )


def shutdown_container(container: Any | None) -> None:
    if container is None:
        return
    try:
        if getattr(container, "case_agent", None) is not None:
            container.case_agent.shutdown()
        elif getattr(container, "caser_context", None) is not None:
            container.caser_context.shutdown()
        if getattr(container, "evidence_agent", None) is not None:
            container.evidence_agent.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
