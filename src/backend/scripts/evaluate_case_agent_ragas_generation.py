"""Evaluate Case Agent generated answers with RAGAS generation metrics.

This script drives the existing Case Agent service with policy goldset
questions, collects the final assistant answer and the actual policy contexts
used by the agent, then scores Answer Correctness and Faithfulness.

It is an offline evaluation harness only. It does not change policy retrieval,
case facts, rule hits, risk scores, or human audit decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.api.dependencies import build_container
from src.backend.core.config import Settings
from src.backend.domain.audit.review.entities import AuthenticatedUser
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentCreateSessionInput,
    CaseAgentSendMessageInput,
)
from src.backend.infrastructure.policy_rag.paths import DEFAULT_CORPUS_ROOT
from src.backend.scripts.policy_rag_eval.common import (
    DEFAULT_ENV_FILE,
    load_deepseek_config,
    load_env_file,
    parse_json_object,
    read_jsonl,
    write_jsonl,
)


DEFAULT_EVAL_SET_JSONL = (
    DEFAULT_CORPUS_ROOT
    / "eval"
    / "policy_rag_eval_set_v2_2_required_optional_audited.jsonl"
)
DEFAULT_RESULTS_JSONL = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_case_agent_ragas_generation_results.jsonl"
)
DEFAULT_RAGAS_INPUTS_JSONL = (
    DEFAULT_CORPUS_ROOT / "eval" / "policy_case_agent_ragas_inputs.jsonl"
)
DEFAULT_REPORT_JSON = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_case_agent_ragas_generation_report.json"
)
DEFAULT_REPORT_MD = (
    DEFAULT_CORPUS_ROOT / "reports" / "policy_case_agent_ragas_generation_report.md"
)

TERMINAL_STATUSES = {
    "waiting_for_user",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "degraded",
}
EVALUABLE_STATUSES = {"completed", "degraded"}
CHECKPOINT_CONTEXT_NODES = (
    "validate_answer",
    "generate_answer",
    "resolve_answer_style",
    "resolve_answer_policy",
    "build_answer_context",
    "call_capabilities",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current Case Agent answers with RAGAS Answer Correctness and Faithfulness."
    )
    parser.add_argument("--eval-set-jsonl", type=Path, default=DEFAULT_EVAL_SET_JSONL)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--ragas-inputs-jsonl", type=Path, default=DEFAULT_RAGAS_INPUTS_JSONL)
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
    parser.add_argument(
        "--metric-backend",
        choices=["ragas", "deepseek_formula", "auto"],
        default="auto",
        help=(
            "auto tries native RAGAS first and falls back to the project JSON judge; "
            "ragas requests native RAGAS but still falls back unless --ragas-native-only is set; "
            "deepseek_formula uses the same formulas with a project JSON judge."
        ),
    )
    parser.add_argument(
        "--answer-correctness-mode",
        choices=["factual", "ragas_default"],
        default="factual",
        help="factual uses RAGAS weights [1,0]; ragas_default uses [0.75,0.25] and requires embeddings.",
    )
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-sessions", action="store_true")
    parser.add_argument(
        "--inject-gold-filters",
        action="store_true",
        help=(
            "Inject each eval row's gold filters into L3 ExpertAnalysisTask.filters. "
            "Use this only for fixed-filter generation evaluation; omit it to test "
            "the real online filter inference path."
        ),
    )
    parser.add_argument(
        "--allow-regionless-online-questions",
        action="store_true",
        help=(
            "Allow online-chain evaluation rows whose question text has no explicit "
            "jurisdiction. By default, rows without region terms fail fast unless "
            "--inject-gold-filters is used."
        ),
    )
    parser.add_argument(
        "--ragas-native-only",
        action="store_true",
        help="Disable fallback when native RAGAS scoring fails.",
    )
    parser.add_argument(
        "--force-ragas-native",
        action="store_true",
        help=(
            "Force native RAGAS calls even when the configured DeepSeek model is "
            "known to reject RAGAS structured tool_choice requests."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(args)
    write_reports(report, args.report_json, args.report_md)
    print(
        json.dumps(
            {
                "status": "ok",
                "query_count": report["query_count"],
                "completed_count": report["summary"]["run_status_counts"].get("completed", 0),
                "answer_correctness_avg": report["summary"].get("answer_correctness_avg"),
                "faithfulness_avg": report["summary"].get("faithfulness_avg"),
                "metric_backend": report["metric_backend"],
                "results_jsonl": str(args.results_jsonl),
                "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
                "report_json": str(args.report_json),
                "report_md": str(args.report_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    rows = selected_eval_rows(args)
    validate_online_chain_region_contract(args, rows)
    existing = load_existing_results(args.results_jsonl) if not args.no_resume else {}

    container = None
    service = None
    evaluator: GenerationMetricEvaluator | None = None
    results: list[dict[str, Any]] = []
    ragas_inputs: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    restore_gold_filter_injection: Callable[[], None] | None = None
    try:
        container = build_runtime_container()
        service = container.case_agent
        if service is None:
            raise RuntimeError(
                "Case Agent is not available: "
                + json.dumps(container.case_agent_status, ensure_ascii=False)
            )
        if args.inject_gold_filters:
            restore_gold_filter_injection = install_gold_filter_injection(service)
        actor = resolve_actor(container, args.actor_username)
        case_id = resolve_case_id(container, args.case_id)
        evaluator = None if args.skip_ragas else GenerationMetricEvaluator(args)

        total = len(rows)
        for index, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id") or f"row_{index}")
            if query_id in existing:
                result = existing[query_id]
                results.append(result)
                ragas_input = result.get("ragas_input")
                if isinstance(ragas_input, dict):
                    ragas_inputs.append(ragas_input)
                print_progress(args, index, total, result, resumed=True)
                continue

            result = evaluate_one_row(
                args,
                row=row,
                service=service,
                actor=actor,
                case_id=case_id,
                evaluator=evaluator,
            )
            results.append(result)
            ragas_inputs.append(result["ragas_input"])
            append_jsonl(args.results_jsonl, [result])
            append_jsonl(args.ragas_inputs_jsonl, [result["ragas_input"]])
            print_progress(args, index, total, result, resumed=False)
    finally:
        if restore_gold_filter_injection is not None:
            restore_gold_filter_injection()
        shutdown_container(container)

    summary = summarize(results)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set_jsonl": str(args.eval_set_jsonl),
        "results_jsonl": str(args.results_jsonl),
        "ragas_inputs_jsonl": str(args.ragas_inputs_jsonl),
        "query_count": len(results),
        "offset": args.offset,
        "limit": args.limit,
        "response_scope": args.response_scope,
        "metric_backend": "skipped" if args.skip_ragas else args.metric_backend,
        "answer_correctness_mode": args.answer_correctness_mode,
        "elapsed_seconds": time.perf_counter() - started_all,
        "summary": summary,
        "sample_results": slim_results(results[:20]),
        "boundary": {
            "uses_current_goldset": True,
            "drives_case_agent_service": True,
            "matches_frontend_backend_chain_after_post": True,
            "runs_policy_rag_mcp_through_expert_agent": True,
            "uses_actual_agent_response": True,
            "uses_actual_policy_contexts": True,
            "injects_gold_filters_to_l3": bool(args.inject_gold_filters),
            "requires_region_terms_without_gold_filter_injection": (
                not args.allow_regionless_online_questions
            ),
            "attempts_ragas_native_metrics": (
                not args.skip_ragas and args.metric_backend in {"ragas", "auto"}
            ),
            "allows_deepseek_formula_fallback": (
                not args.skip_ragas
                and args.metric_backend in {"auto", "ragas"}
                and not args.ragas_native_only
            ),
            "uses_ragas_native_metrics": (
                (summary.get("actual_metric_backend_counts") or {}).get("ragas", 0) > 0
            ),
            "uses_deepseek_formula_metrics": (
                (summary.get("actual_metric_backend_counts") or {}).get("deepseek_formula", 0)
                > 0
            ),
            "calculates_answer_correctness": not args.skip_ragas,
            "calculates_faithfulness": not args.skip_ragas,
            "makes_audit_decisions": False,
        },
    }


def selected_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.eval_set_jsonl)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def validate_online_chain_region_contract(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    if args.inject_gold_filters or args.allow_regionless_online_questions:
        return
    missing = [
        str(row.get("query_id") or index)
        for index, row in enumerate(rows, start=1)
        if not question_has_explicit_region(str(row.get("question") or ""))
    ]
    if not missing:
        return
    sample = ", ".join(missing[:10])
    raise RuntimeError(
        "Online-chain generation evaluation requires explicit jurisdiction terms "
        "in every question so PolicyFilterResolver can infer filters from the same "
        "signal a real user would provide. Missing region rows: "
        f"{sample}. Add 北京/上海/国家/跨省/异地 to the questions, or rerun with "
        "--inject-gold-filters to evaluate generation under fixed gold filters."
    )


def question_has_explicit_region(question: str) -> bool:
    compact = str(question or "").replace(" ", "")
    return any(
        token in compact
        for token in (
            "北京",
            "北京市",
            "上海",
            "上海市",
            "国家",
            "全国",
            "跨省",
            "异地",
        )
    )


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
    row: dict[str, Any],
    service: Any,
    actor: AuthenticatedUser,
    case_id: str,
    evaluator: "GenerationMetricEvaluator | None",
) -> dict[str, Any]:
    query_id = str(row.get("query_id") or "")
    started = time.perf_counter()
    session_id = ""
    run_id = ""
    run_status = "not_started"
    run_error_code = None
    run_error_message = None
    gold_filters_injected = False
    try:
        if args.inject_gold_filters:
            raw_filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
            gold_filters = dict(raw_filters or {})
            if gold_filters:
                gold_filters["force_policy_filters"] = True
                setattr(service, "_policy_rag_eval_gold_filters", gold_filters)
                gold_filters_injected = True
            else:
                setattr(service, "_policy_rag_eval_gold_filters", {})
        session = service.create_session(
            case_id,
            CaseAgentCreateSessionInput(title=f"RAGAS {query_id or 'eval'}"),
            actor,
        )
        session_id = session.session_id
        run = service.send_message(
            session.session_id,
            CaseAgentSendMessageInput(
                content=str(row.get("question") or ""),
                active_stage=args.active_stage,
            ),
            actor,
        )
        run_id = run.run_id
        run = wait_for_run(
            service,
            run_id=run_id,
            actor=actor,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
        )
        run_status = run.status
        run_error_code = run.error_code
        run_error_message = run.error_message

        answer = load_run_answer(service, session_id=session_id, run=run, actor=actor)
        response_text = answer_text(answer, scope=args.response_scope)
        contexts = collect_policy_contexts(service, run_id=run_id, answer=answer)
        filter_diagnostics = collect_filter_diagnostics(service, run_id=run_id)
        ragas_input = build_ragas_input(row, response_text=response_text, contexts=contexts)
        metric_result = score_generation(
            evaluator,
            row=row,
            response_text=response_text,
            contexts=contexts,
            run_status=run_status,
        )
        result = {
            "schema_version": "policy_case_agent_ragas_generation_eval_v1",
            "query_id": query_id,
            "question": row.get("question"),
            "reference_answer": row.get("reference_answer"),
            "bundle_type": row.get("bundle_type"),
            "question_type": row.get("question_type"),
            "filters": row.get("filters") or {},
            "gold_filters": row.get("filters") or {},
            "gold_filters_injected": gold_filters_injected,
            "filter_diagnostics": filter_diagnostics,
            "llm_planner_filters": filter_diagnostics.get("llm_planner_filters", {}),
            "resolver_filters": filter_diagnostics.get("resolver_filters", {}),
            "retrieval_request_filters": filter_diagnostics.get(
                "retrieval_request_filters",
                {},
            ),
            "case_id": case_id,
            "session_id": session_id,
            "run_id": run_id,
            "run_status": run_status,
            "run_error_code": run_error_code,
            "run_error_message": run_error_message,
            "answer_display_mode": getattr(answer, "display_mode", None) if answer else None,
            "capabilities_used": answer.metadata.get("capabilities_used", []) if answer else [],
            "response_scope": args.response_scope,
            "agent_response": response_text,
            "retrieved_context_count": len(contexts),
            "retrieved_contexts": contexts,
            "ragas_input": ragas_input,
            "metrics": metric_result,
            "elapsed_seconds": time.perf_counter() - started,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        return result
    except Exception as exc:
        return {
            "schema_version": "policy_case_agent_ragas_generation_eval_v1",
            "query_id": query_id,
            "question": row.get("question"),
            "reference_answer": row.get("reference_answer"),
            "bundle_type": row.get("bundle_type"),
            "question_type": row.get("question_type"),
            "filters": row.get("filters") or {},
            "gold_filters": row.get("filters") or {},
            "gold_filters_injected": gold_filters_injected,
            "filter_diagnostics": {},
            "llm_planner_filters": {},
            "resolver_filters": {},
            "retrieval_request_filters": {},
            "case_id": case_id,
            "session_id": session_id,
            "run_id": run_id,
            "run_status": run_status,
            "run_error_code": run_error_code or exc.__class__.__name__,
            "run_error_message": str(exc)[:500],
            "answer_display_mode": None,
            "capabilities_used": [],
            "response_scope": args.response_scope,
            "agent_response": "",
            "retrieved_context_count": 0,
            "retrieved_contexts": [],
            "ragas_input": build_ragas_input(row, response_text="", contexts=[]),
            "metrics": {
                "evaluable": False,
                "reason": "agent_run_failed",
                "answer_correctness": None,
                "faithfulness": None,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    finally:
        if args.inject_gold_filters:
            try:
                setattr(service, "_policy_rag_eval_gold_filters", {})
            except Exception:
                pass
        if session_id and not args.keep_sessions:
            try:
                service.archive_session(session_id, actor)
            except Exception:
                pass


def install_gold_filter_injection(service: Any) -> Callable[[], None]:
    original = getattr(service, "_materialize_expert_task", None)
    if not callable(original):
        return lambda: None

    def wrapped_materialize_expert_task(*args: Any, **kwargs: Any) -> Any:
        task = original(*args, **kwargs)
        filters = getattr(service, "_policy_rag_eval_gold_filters", None)
        if isinstance(filters, dict) and filters:
            return task.model_copy(update={"filters": dict(filters)})
        return task

    setattr(service, "_materialize_expert_task", wrapped_materialize_expert_task)

    def restore() -> None:
        setattr(service, "_materialize_expert_task", original)

    return restore


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


def build_ragas_input(
    row: dict[str, Any],
    *,
    response_text: str,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    required_points = metric_point_statements(row, "required_answer_points")
    optional_points = metric_point_statements(row, "optional_answer_points")
    return {
        "query_id": row.get("query_id"),
        "user_input": row.get("question"),
        "reference": reference_for_generation_eval(row),
        "reference_answer_full": row.get("reference_answer_full")
        or row.get("reference_answer"),
        "required_answer_points": required_points,
        "optional_answer_points": optional_points,
        "response": response_text,
        "retrieved_contexts": [
            str(context.get("text") or context.get("excerpt") or "").strip()
            for context in contexts
            if str(context.get("text") or context.get("excerpt") or "").strip()
        ],
    }


def reference_for_generation_eval(row: dict[str, Any]) -> str:
    return str(row.get("reference_answer_required") or row.get("reference_answer") or "")


def metric_point_statements(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list):
        return []
    statements: list[str] = []
    for item in value:
        if isinstance(item, dict):
            statement = str(item.get("statement") or "").strip()
        else:
            statement = str(item or "").strip()
        if statement:
            statements.append(statement)
    return statements


def score_generation(
    evaluator: "GenerationMetricEvaluator | None",
    *,
    row: dict[str, Any],
    response_text: str,
    contexts: list[dict[str, Any]],
    run_status: str,
) -> dict[str, Any]:
    if evaluator is None:
        return {
            "evaluable": False,
            "reason": "metric_scoring_skipped",
            "answer_correctness": None,
            "faithfulness": None,
        }
    if run_status not in EVALUABLE_STATUSES:
        return {
            "evaluable": False,
            "reason": f"run_status_{run_status}",
            "answer_correctness": None,
            "faithfulness": None,
        }
    if not response_text.strip():
        return {
            "evaluable": False,
            "reason": "empty_agent_response",
            "answer_correctness": None,
            "faithfulness": None,
        }
    context_texts = [
        str(context.get("text") or context.get("excerpt") or "").strip()
        for context in contexts
        if str(context.get("text") or context.get("excerpt") or "").strip()
    ]
    if not context_texts:
        return {
            "evaluable": False,
            "reason": "no_policy_context",
            "answer_correctness": None,
            "faithfulness": None,
        }
    try:
        return evaluator.score(
            user_input=str(row.get("question") or ""),
            reference=reference_for_generation_eval(row),
            reference_answer_full=str(
                row.get("reference_answer_full") or row.get("reference_answer") or ""
            ),
            required_answer_points=metric_point_statements(row, "required_answer_points"),
            optional_answer_points=metric_point_statements(row, "optional_answer_points"),
            response=response_text,
            retrieved_contexts=context_texts,
        )
    except Exception as exc:
        return {
            "evaluable": False,
            "reason": "metric_scoring_failed",
            "answer_correctness": None,
            "faithfulness": None,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc)[:500],
        }


class GenerationMetricEvaluator:
    """Score generation metrics with RAGAS or an equivalent DeepSeek formula judge."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = args.metric_backend
        self.answer_correctness_mode = args.answer_correctness_mode
        self.ragas_native_only = args.ragas_native_only
        self.force_ragas_native = args.force_ragas_native
        self._deepseek = load_deepseek_config(args.env_file)
        self._judge_timeout = args.judge_timeout
        self._judge_temperature = args.judge_temperature
        self._ragas_llm = None
        self._ragas_answer_correctness = None
        self._ragas_faithfulness = None
        self._ragas_skip_reason = ""
        self._ragas_setup_error: Exception | None = None
        self._formula_judge = None
        if self.backend in {"ragas", "auto"}:
            if self._should_skip_native_ragas(args):
                self._ragas_skip_reason = (
                    "configured_judge_model_rejects_ragas_tool_choice"
                )
            else:
                try:
                    self._setup_ragas()
                except Exception as exc:
                    if self.ragas_native_only:
                        raise
                    self._ragas_setup_error = exc
        if self.backend in {"deepseek_formula", "auto"}:
            self._formula_judge = DeepSeekRagasFormulaJudge(
                config=self._deepseek,
                timeout=args.judge_timeout,
                temperature=args.judge_temperature,
            )

    def _should_skip_native_ragas(self, args: argparse.Namespace) -> bool:
        if self.ragas_native_only or self.force_ragas_native:
            return False
        if os.environ.get("MEDIGUARD_POLICY_RAG_EVAL_FORCE_RAGAS_NATIVE") == "1":
            return False
        return deepseek_model_rejects_ragas_tool_choice(self._deepseek)

    def _setup_ragas(self) -> None:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerCorrectness, Faithfulness

        client = AsyncOpenAI(
            api_key=self._deepseek["api_key"],
            base_url=self._deepseek["base_url"],
            timeout=self._judge_timeout,
            max_retries=1,
        )
        self._ragas_llm = llm_factory(
            self._deepseek["model"],
            client=client,
            temperature=self._judge_temperature,
        )
        weights = [1.0, 0.0]
        embeddings = None
        if self.answer_correctness_mode == "ragas_default":
            weights = [0.75, 0.25]
            embeddings = build_ragas_embeddings()
        self._ragas_answer_correctness = AnswerCorrectness(
            llm=self._ragas_llm,
            embeddings=embeddings,
            weights=weights,
        )
        self._ragas_faithfulness = Faithfulness(llm=self._ragas_llm)

    def score(
        self,
        *,
        user_input: str,
        reference: str,
        response: str,
        retrieved_contexts: list[str],
        reference_answer_full: str = "",
        required_answer_points: list[str] | None = None,
        optional_answer_points: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.backend == "ragas":
            if not self._native_ragas_ready():
                return self._score_with_formula_judge(
                    user_input=user_input,
                    reference=reference,
                    reference_answer_full=reference_answer_full,
                    required_answer_points=required_answer_points or [],
                    optional_answer_points=optional_answer_points or [],
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    ragas_error=self._native_ragas_unavailable_error(),
                )
            try:
                return asyncio.run(
                    self._score_with_ragas(
                        user_input=user_input,
                        reference=reference,
                        reference_answer_full=reference_answer_full,
                        required_answer_points=required_answer_points or [],
                        optional_answer_points=optional_answer_points or [],
                        response=response,
                        retrieved_contexts=retrieved_contexts,
                    )
                )
            except Exception as exc:
                if self.ragas_native_only:
                    raise
                return self._score_with_formula_judge(
                    user_input=user_input,
                    reference=reference,
                    reference_answer_full=reference_answer_full,
                    required_answer_points=required_answer_points or [],
                    optional_answer_points=optional_answer_points or [],
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    ragas_error=exc,
                )
        if self.backend == "deepseek_formula":
            return self._score_with_formula_judge(
                user_input=user_input,
                reference=reference,
                reference_answer_full=reference_answer_full,
                required_answer_points=required_answer_points or [],
                optional_answer_points=optional_answer_points or [],
                response=response,
                retrieved_contexts=retrieved_contexts,
                ragas_error=None,
            )
        if not self._native_ragas_ready():
            return self._score_with_formula_judge(
                user_input=user_input,
                reference=reference,
                reference_answer_full=reference_answer_full,
                required_answer_points=required_answer_points or [],
                optional_answer_points=optional_answer_points or [],
                response=response,
                retrieved_contexts=retrieved_contexts,
                ragas_error=self._native_ragas_unavailable_error(),
            )
        try:
            return asyncio.run(
                self._score_with_ragas(
                    user_input=user_input,
                    reference=reference,
                    reference_answer_full=reference_answer_full,
                    required_answer_points=required_answer_points or [],
                    optional_answer_points=optional_answer_points or [],
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
            )
        except Exception as exc:
            return self._score_with_formula_judge(
                user_input=user_input,
                reference=reference,
                reference_answer_full=reference_answer_full,
                required_answer_points=required_answer_points or [],
                optional_answer_points=optional_answer_points or [],
                response=response,
                retrieved_contexts=retrieved_contexts,
                ragas_error=exc,
            )

    def _native_ragas_ready(self) -> bool:
        return (
            self._ragas_answer_correctness is not None
            and self._ragas_faithfulness is not None
        )

    def _native_ragas_unavailable_error(self) -> Exception:
        if self._ragas_setup_error is not None:
            return self._ragas_setup_error
        reason = self._ragas_skip_reason or "native_ragas_not_initialized"
        return NativeRagasUnavailable(reason)

    async def _score_with_ragas(
        self,
        *,
        user_input: str,
        reference: str,
        response: str,
        retrieved_contexts: list[str],
        reference_answer_full: str = "",
        required_answer_points: list[str] | None = None,
        optional_answer_points: list[str] | None = None,
    ) -> dict[str, Any]:
        answer_correctness = await self._ragas_answer_correctness.ascore(
            user_input=user_input,
            response=response,
            reference=reference,
        )
        faithfulness = await self._ragas_faithfulness.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        return {
            "evaluable": True,
            "reason": None,
            "metric_backend": "ragas",
            "judge_model": self._deepseek["model"],
            "answer_correctness": clean_score(getattr(answer_correctness, "value", None)),
            "faithfulness": clean_score(getattr(faithfulness, "value", None)),
            "answer_correctness_mode": self.answer_correctness_mode,
            "ragas_native": True,
        }

    def _score_with_formula_judge(
        self,
        *,
        user_input: str,
        reference: str,
        reference_answer_full: str,
        required_answer_points: list[str],
        optional_answer_points: list[str],
        response: str,
        retrieved_contexts: list[str],
        ragas_error: Exception | None,
    ) -> dict[str, Any]:
        if self._formula_judge is None:
            self._formula_judge = DeepSeekRagasFormulaJudge(
                config=self._deepseek,
                timeout=self._judge_timeout,
                temperature=self._judge_temperature,
            )
        judged = self._formula_judge.judge(
            user_input=user_input,
            reference=reference,
            reference_answer_full=reference_answer_full,
            required_answer_points=required_answer_points,
            optional_answer_points=optional_answer_points,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        judged["metric_backend"] = "deepseek_formula"
        judged["ragas_native"] = False
        if ragas_error is not None:
            judged["ragas_fallback_reason"] = f"{ragas_error.__class__.__name__}: {str(ragas_error)[:300]}"
        return judged


class NativeRagasUnavailable(RuntimeError):
    """Native RAGAS was intentionally skipped before making an API call."""


def deepseek_model_rejects_ragas_tool_choice(config: dict[str, str]) -> bool:
    base_url = str(config.get("base_url") or "").lower()
    model = str(config.get("model") or "").lower()
    if "deepseek" not in base_url and "deepseek" not in model:
        return False
    risky_markers = ("reasoner", "thinking", "r1", "v4")
    return any(marker in model for marker in risky_markers)


class DeepSeekRagasFormulaJudge:
    """Fallback judge that applies RAGAS-compatible formulas to JSON verdicts."""

    def __init__(
        self,
        *,
        config: dict[str, str],
        timeout: float,
        temperature: float,
    ) -> None:
        from openai import OpenAI

        self.model = config["model"]
        self._client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=timeout,
            max_retries=1,
        )
        self._temperature = temperature

    def judge(
        self,
        *,
        user_input: str,
        reference: str,
        reference_answer_full: str,
        required_answer_points: list[str],
        optional_answer_points: list[str],
        response: str,
        retrieved_contexts: list[str],
    ) -> dict[str, Any]:
        payload = {
            "user_input": user_input,
            "reference": reference,
            "reference_answer_full": reference_answer_full,
            "required_answer_points": required_answer_points,
            "optional_answer_points": optional_answer_points,
            "response": response,
            "retrieved_contexts": [
                {"rank": index, "text": text[:1800]}
                for index, text in enumerate(retrieved_contexts, start=1)
            ],
        }
        raw = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formula_judge_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
            stream=False,
        )
        parsed = parse_json_object(raw.choices[0].message.content or "")
        return normalize_formula_judge(
            parsed,
            model=self.model,
            required_answer_points=required_answer_points,
            optional_answer_points=optional_answer_points,
        )


def formula_judge_prompt() -> str:
    return (
        "你是医保政策 RAG 生成评测裁判。只能根据输入的 reference、response、retrieved_contexts 做评测，"
        "不要使用外部知识，不要补充政策结论，不要判断审核通过/拒付/处罚。输出必须是 JSON object。"
        "任务 A：按 RAGAS Answer Correctness 思路评分。"
        "如果 required_answer_points 非空，只把 required_answer_points 作为必须覆盖的事实点；"
        "required 缺失才算 FN。optional_answer_points 是允许补充事实：漏答 optional 不算 FN，"
        "正确提到 optional 不算 FP，也不加 TP；错误提到 optional 才算 FP。"
        "如果 required_answer_points 为空，则退回 RAGAS 常规做法：把 reference 和 response 分别拆成原子事实声明，"
        "把 response 声明归为 TP/FP，把 reference 中未覆盖声明归为 FN。"
        "TP 表示 response 正确覆盖 required/reference 事实；FP 表示 response 多说、错说或 reference/retrieved_contexts 不支持；"
        "FN 表示 required/reference 中应该回答但 response 漏掉。"
        "任务 B：按 RAGAS Faithfulness 思路，把 response 拆成原子事实声明，判断每条声明是否能从 retrieved_contexts 直接支持或合理归因。"
        "返回格式："
        "{\"answer_correctness\":{\"response_statements\":[\"...\"],\"reference_statements\":[\"...\"],"
        "\"required_points\":[{\"statement\":\"...\",\"covered\":true}],"
        "\"optional_points\":[{\"statement\":\"...\",\"covered\":false}],"
        "\"TP\":[\"...\"],\"FP\":[\"...\"],\"FN\":[\"...\"]},"
        "\"faithfulness\":{\"statements\":[{\"statement\":\"...\",\"supported\":true,"
        "\"supporting_context_ranks\":[1]}]}}"
    )


def normalize_formula_judge(
    payload: dict[str, Any],
    *,
    model: str,
    required_answer_points: list[str] | None = None,
    optional_answer_points: list[str] | None = None,
) -> dict[str, Any]:
    correctness = payload.get("answer_correctness") if isinstance(payload, dict) else {}
    faithfulness = payload.get("faithfulness") if isinstance(payload, dict) else {}
    if not isinstance(correctness, dict):
        correctness = {}
    if not isinstance(faithfulness, dict):
        faithfulness = {}
    required_points = clean_string_list(required_answer_points or [])
    optional_points = clean_string_list(optional_answer_points or [])
    if required_points:
        required_verdicts = normalize_point_verdicts(
            correctness.get("required_points"),
            required_points,
        )
        if required_verdicts:
            tp = [
                item["statement"]
                for item in required_verdicts
                if item.get("covered") is True
            ]
            fn = [
                item["statement"]
                for item in required_verdicts
                if item.get("covered") is not True
            ]
        else:
            tp = clean_string_list(correctness.get("TP"))
            fn = clean_string_list(correctness.get("FN"))
        optional_verdicts = normalize_point_verdicts(
            correctness.get("optional_points"),
            optional_points,
        )
    else:
        tp = clean_string_list(correctness.get("TP"))
        fn = clean_string_list(correctness.get("FN"))
        required_verdicts = []
        optional_verdicts = normalize_point_verdicts(
            correctness.get("optional_points"),
            optional_points,
        )
    fp = clean_string_list(correctness.get("FP"))
    answer_correctness = f1_from_counts(len(tp), len(fp), len(fn))
    required_point_recall = (
        len(tp) / len(required_verdicts)
        if required_verdicts
        else (len(tp) / (len(tp) + len(fn)) if len(tp) + len(fn) else None)
    )
    optional_covered_count = sum(
        1 for item in optional_verdicts if item.get("covered") is True
    )
    optional_point_coverage = (
        optional_covered_count / len(optional_verdicts) if optional_verdicts else None
    )

    statements = []
    supported = 0
    for item in faithfulness.get("statements") or []:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        ranks = [
            int(rank)
            for rank in item.get("supporting_context_ranks") or []
            if str(rank).isdigit()
        ]
        is_supported = bool(item.get("supported")) and bool(ranks)
        supported += 1 if is_supported else 0
        statements.append(
            {
                "statement": statement,
                "supported": is_supported,
                "supporting_context_ranks": sorted(set(ranks)),
            }
        )
    faithfulness_score = supported / len(statements) if statements else None
    return {
        "evaluable": True,
        "reason": None,
        "judge_model": model,
        "answer_correctness": answer_correctness,
        "faithfulness": faithfulness_score,
        "answer_correctness_counts": {
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
        },
        "required_point_recall": required_point_recall,
        "optional_point_coverage": optional_point_coverage,
        "fp_count": len(fp),
        "fn_required_count": len(fn),
        "answer_correctness_details": {
            "response_statements": clean_string_list(correctness.get("response_statements")),
            "reference_statements": clean_string_list(correctness.get("reference_statements")),
            "required_points": required_verdicts,
            "optional_points": optional_verdicts,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        },
        "faithfulness_counts": {
            "supported": supported,
            "total": len(statements),
        },
        "faithfulness_details": statements,
        "answer_correctness_mode": "factual",
    }


def normalize_point_verdicts(value: Any, source_points: list[str]) -> list[dict[str, Any]]:
    if not source_points:
        return []
    raw_items = value if isinstance(value, list) else []
    raw_by_statement = {
        compact_statement(str(item.get("statement") or "")): item
        for item in raw_items
        if isinstance(item, dict) and str(item.get("statement") or "").strip()
    }
    verdicts: list[dict[str, Any]] = []
    for index, source_statement in enumerate(source_points):
        raw_item = raw_by_statement.get(compact_statement(source_statement))
        if raw_item is None and index < len(raw_items):
            raw_item = raw_items[index]
        if isinstance(raw_item, dict):
            statement = source_statement
            covered = raw_item.get("covered") is True
        else:
            statement = source_statement
            covered = False
        verdicts.append(
            {
                "statement": statement or source_statement,
                "covered": covered,
            }
        )
    return verdicts


def compact_statement(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isalnum())


def build_ragas_embeddings() -> Any:
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


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
    return [
        {**context, "rank": index}
        for index, context in enumerate(contexts[:8], start=1)
    ]


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
                if row.tool_name != "ask_policy_expert":
                    continue
                contexts.extend(contexts_from_expert_payload(row.result_summary))
            return contexts
    except Exception:
        return []


def collect_filter_diagnostics(service: Any, *, run_id: str) -> dict[str, Any]:
    repository = getattr(service, "_repository", None)
    session_factory = getattr(repository, "_session_factory", None)
    if not callable(session_factory):
        return {}
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
                return {}
            rows = session.scalars(
                select(CaseAgentToolCallORM)
                .where(CaseAgentToolCallORM.run_id == run_row.id)
                .order_by(CaseAgentToolCallORM.sequence)
            ).all()
            for row in rows:
                if row.tool_name != "ask_policy_expert":
                    continue
                diagnostics = filter_diagnostics_from_expert_payload(row.result_summary)
                if diagnostics:
                    return diagnostics
    except Exception:
        return {}
    return {}


def filter_diagnostics_from_expert_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    diagnostics = payload.get("filter_diagnostics")
    if isinstance(diagnostics, dict):
        return diagnostics
    wrapped = payload.get("payload")
    if isinstance(wrapped, dict):
        diagnostics = wrapped.get("filter_diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return {}


def contexts_from_expert_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    evidence = payload.get("policy_evidence")
    if not isinstance(evidence, list):
        wrapped = payload.get("payload")
        if isinstance(wrapped, dict):
            evidence = wrapped.get("policy_evidence")
    contexts: list[dict[str, Any]] = []
    for index, item in enumerate(evidence or [], start=1):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        text = str(item.get("excerpt") or item.get("text") or "").strip()
        if not text:
            continue
        contexts.append(
            {
                "rank": index,
                "source_ref": item.get("source_ref") or item.get("evidence_ref"),
                "evidence_ref": item.get("evidence_ref") or item.get("source_ref"),
                "node_id": metadata.get("node_id"),
                "source_id": metadata.get("source_id"),
                "source_url": item.get("source_url"),
                "title": item.get("title"),
                "jurisdiction": item.get("jurisdiction"),
                "policy_domain": item.get("policy_domain"),
                "content_type": item.get("content_type"),
                "text": text,
            }
        )
    return contexts


def contexts_from_checkpoints(service: Any, *, run_id: str) -> list[dict[str, Any]]:
    repository = getattr(service, "_repository", None)
    get_latest_checkpoint = getattr(repository, "get_latest_checkpoint", None)
    if not callable(get_latest_checkpoint):
        return []
    snapshots: list[dict[str, Any]] = []
    for node_name in CHECKPOINT_CONTEXT_NODES:
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
                snapshot.get("answer_context"),
                source_by_ref=source_by_ref,
            )
        )
        contexts.extend(contexts_from_capability_results(snapshot.get("capability_results")))
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
        if not isinstance(section, dict):
            continue
        if str(section.get("section") or "") != "政策专家分析":
            continue
        for index, item in enumerate(section.get("policy_evidence") or [], start=1):
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
                    "jurisdiction": item.get("jurisdiction") or metadata.get("jurisdiction"),
                    "policy_domain": item.get("policy_domain") or metadata.get("policy_domain"),
                    "content_type": item.get("content_type"),
                    "text": text,
                }
            )
    return contexts


def contexts_from_capability_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    contexts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("capability") != "ask_policy_expert":
            continue
        payload = item.get("payload_summary")
        if isinstance(payload, dict) and not payload.get("truncated"):
            contexts.extend(contexts_from_expert_payload(payload.get("payload") or payload))
    return contexts


def sources_by_ref(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    sources: dict[str, dict[str, Any]] = {}
    for item in value:
        if isinstance(item, dict) and item.get("source_ref"):
            sources[str(item["source_ref"])] = item
    return sources


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
        key = (
            str(context.get("node_id") or context.get("source_ref") or ""),
            text[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(context)
    return output


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
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


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clean_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(1.0, number))


def f1_from_counts(tp: int, fp: int, fn: int, *, beta: float = 1.0) -> float:
    precision = 1.0 if tp + fp == 0 and fn == 0 else (tp / (tp + fp) if tp + fp else 0.0)
    recall = 1.0 if tp + fn == 0 and fp == 0 else (tp / (tp + fn) if tp + fn else 0.0)
    if precision + recall == 0:
        return 0.0
    beta_squared = beta**2
    return float(
        (1 + beta_squared)
        * (precision * recall)
        / (beta_squared * precision + recall)
    )


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    run_status_counts = Counter(str(item.get("run_status") or "unknown") for item in results)
    reasons = Counter(str((item.get("metrics") or {}).get("reason") or "ok") for item in results)
    actual_backends = Counter(
        str((item.get("metrics") or {}).get("metric_backend") or "none")
        for item in results
    )
    native_ragas_count = actual_backends.get("ragas", 0)
    fallback_count = sum(
        1
        for item in results
        if (item.get("metrics") or {}).get("ragas_fallback_reason")
    )
    evaluated = [
        item for item in results
        if (item.get("metrics") or {}).get("evaluable") is True
    ]
    return {
        "query_count": len(results),
        "evaluated_count": len(evaluated),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "metric_reason_counts": dict(sorted(reasons.items())),
        "actual_metric_backend_counts": dict(sorted(actual_backends.items())),
        "native_ragas_count": native_ragas_count,
        "ragas_fallback_count": fallback_count,
        "answer_correctness_avg": average(
            (item.get("metrics") or {}).get("answer_correctness")
            for item in evaluated
        ),
        "faithfulness_avg": average(
            (item.get("metrics") or {}).get("faithfulness")
            for item in evaluated
        ),
        "retrieved_context_count_avg": average(
            item.get("retrieved_context_count") for item in results
        ),
        "required_point_recall_avg": average(
            (item.get("metrics") or {}).get("required_point_recall")
            for item in evaluated
        ),
        "optional_point_coverage_avg": average(
            (item.get("metrics") or {}).get("optional_point_coverage")
            for item in evaluated
        ),
        "fp_count_total": sum_metric_count(evaluated, "fp_count"),
        "fn_required_count_total": sum_metric_count(evaluated, "fn_required_count"),
        "by_question_type": grouped_metric_summary(results, "question_type"),
        "by_bundle_type": grouped_metric_summary(results, "bundle_type"),
    }


def grouped_metric_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item.get(key) or "unknown")].append(item)
    return {
        group: {
            "query_count": len(items),
            "evaluated_count": sum(
                1 for item in items
                if (item.get("metrics") or {}).get("evaluable") is True
            ),
            "answer_correctness_avg": average(
                (item.get("metrics") or {}).get("answer_correctness")
                for item in items
                if (item.get("metrics") or {}).get("evaluable") is True
            ),
            "faithfulness_avg": average(
                (item.get("metrics") or {}).get("faithfulness")
                for item in items
                if (item.get("metrics") or {}).get("evaluable") is True
            ),
            "required_point_recall_avg": average(
                (item.get("metrics") or {}).get("required_point_recall")
                for item in items
                if (item.get("metrics") or {}).get("evaluable") is True
            ),
            "optional_point_coverage_avg": average(
                (item.get("metrics") or {}).get("optional_point_coverage")
                for item in items
                if (item.get("metrics") or {}).get("evaluable") is True
            ),
        }
        for group, items in sorted(grouped.items())
    }


def average(values: Iterable[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def sum_metric_count(results: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in results:
        value = (item.get("metrics") or {}).get(key)
        if isinstance(value, int):
            total += value
    return total


def slim_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slimmed = []
    for item in results:
        contexts = item.get("retrieved_contexts") or []
        slimmed.append(
            {
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "run_status": item.get("run_status"),
                "answer_display_mode": item.get("answer_display_mode"),
                "retrieved_context_count": item.get("retrieved_context_count"),
                "metrics": item.get("metrics"),
                "agent_response_preview": str(item.get("agent_response") or "")[:240],
                "context_previews": [
                    {
                        "rank": context.get("rank"),
                        "node_id": context.get("node_id"),
                        "source_id": context.get("source_id"),
                        "policy_domain": context.get("policy_domain"),
                        "text": str(context.get("text") or "")[:180],
                    }
                    for context in contexts[:3]
                    if isinstance(context, dict)
                ],
            }
        )
    return slimmed


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
    summary = report.get("summary") or {}
    lines = [
        "# Case Agent RAGAS Generation Eval Report",
        "",
        f"Updated at: {report.get('generated_at')}",
        "",
        "This report evaluates the current Case Agent final answer, not retrieval alone. It does not make audit decisions.",
        "",
        "## Scope",
        "",
        f"- Eval set: `{report.get('eval_set_jsonl')}`",
        f"- Query count: {report.get('query_count')}",
        f"- Response scope: `{report.get('response_scope')}`",
        f"- Inject gold filters to L3: `{(report.get('boundary') or {}).get('injects_gold_filters_to_l3')}`",
        f"- Require region terms without gold filter injection: `{(report.get('boundary') or {}).get('requires_region_terms_without_gold_filter_injection')}`",
        f"- Metric backend: `{report.get('metric_backend')}`",
        f"- Actual metric backend counts: `{summary.get('actual_metric_backend_counts')}`",
        f"- Answer Correctness mode: `{report.get('answer_correctness_mode')}`",
        f"- RAGAS inputs: `{report.get('ragas_inputs_jsonl')}`",
        f"- Row results: `{report.get('results_jsonl')}`",
        "",
        "## Overall",
        "",
        f"- Evaluated count: {summary.get('evaluated_count')}",
        f"- Answer Correctness avg: {fmt(summary.get('answer_correctness_avg'))}",
        f"- Faithfulness avg: {fmt(summary.get('faithfulness_avg'))}",
        f"- Required point recall avg: {fmt(summary.get('required_point_recall_avg'))}",
        f"- Optional point coverage avg: {fmt(summary.get('optional_point_coverage_avg'))}",
        f"- FP count total: {summary.get('fp_count_total')}",
        f"- FN required count total: {summary.get('fn_required_count_total')}",
        f"- Retrieved context count avg: {fmt(summary.get('retrieved_context_count_avg'))}",
        f"- Run status counts: `{summary.get('run_status_counts')}`",
        f"- Metric reason counts: `{summary.get('metric_reason_counts')}`",
        f"- Native RAGAS rows: {summary.get('native_ragas_count')}",
        f"- RAGAS fallback rows: {summary.get('ragas_fallback_count')}",
        "",
        "## Metric Definitions",
        "",
        "- Answer Correctness follows RAGAS factual correctness. For v2.2 rows, FN is computed only from required_answer_points; optional_answer_points may be omitted without penalty and are not TP when correctly included.",
        "- Faithfulness follows RAGAS groundedness: split response into statements and check whether each statement is supported by retrieved contexts; score = supported statements / all response statements.",
        "",
        "## By Question Type",
        "",
        "| Question type | Queries | Evaluated | Answer Correctness | Faithfulness | Required Recall | Optional Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in (summary.get("by_question_type") or {}).items():
        lines.append(
            f"| {group} | {item.get('query_count')} | {item.get('evaluated_count')} | "
            f"{fmt(item.get('answer_correctness_avg'))} | {fmt(item.get('faithfulness_avg'))} | "
            f"{fmt(item.get('required_point_recall_avg'))} | {fmt(item.get('optional_point_coverage_avg'))} |"
        )
    lines.extend(
        [
            "",
            "## By Bundle Type",
            "",
            "| Bundle type | Queries | Evaluated | Answer Correctness | Faithfulness | Required Recall | Optional Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, item in (summary.get("by_bundle_type") or {}).items():
        lines.append(
            f"| {group} | {item.get('query_count')} | {item.get('evaluated_count')} | "
            f"{fmt(item.get('answer_correctness_avg'))} | {fmt(item.get('faithfulness_avg'))} | "
            f"{fmt(item.get('required_point_recall_avg'))} | {fmt(item.get('optional_point_coverage_avg'))} |"
        )
    lines.extend(["", "## Sample Results", ""])
    for item in report.get("sample_results") or []:
        metrics = item.get("metrics") or {}
        lines.extend(
            [
                f"### {item.get('query_id')}",
                "",
                f"- Question: {item.get('question')}",
                f"- Run status: `{item.get('run_status')}`",
                f"- Display mode: `{item.get('answer_display_mode')}`",
                f"- Context count: {item.get('retrieved_context_count')}",
                f"- Answer Correctness: {fmt(metrics.get('answer_correctness'))}",
                f"- Faithfulness: {fmt(metrics.get('faithfulness'))}",
                f"- Required point recall: {fmt(metrics.get('required_point_recall'))}",
                f"- Optional point coverage: {fmt(metrics.get('optional_point_coverage'))}",
                f"- FP/FN required: {metrics.get('fp_count')}/{metrics.get('fn_required_count')}",
                f"- Metric backend: `{metrics.get('metric_backend')}`",
                f"- Reason: `{metrics.get('reason')}`",
                f"- Answer preview: {item.get('agent_response_preview')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def print_progress(
    args: argparse.Namespace,
    index: int,
    total: int,
    result: dict[str, Any],
    *,
    resumed: bool,
) -> None:
    every = max(1, args.progress_every)
    if index != 1 and index % every != 0 and index != total:
        return
    metrics = result.get("metrics") or {}
    print(
        "[case-agent-ragas-generation] "
        f"progress={index}/{total} "
        f"query_id={result.get('query_id')} "
        f"run={result.get('run_status')} "
        f"contexts={result.get('retrieved_context_count')} "
        f"ac={fmt(metrics.get('answer_correctness'))} "
        f"faith={fmt(metrics.get('faithfulness'))} "
        f"{'resumed' if resumed else 'new'}",
        flush=True,
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
