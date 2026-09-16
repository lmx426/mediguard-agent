"""Case Agent lightweight orchestrator."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from src.backend.application.agent.case_agent.artifacts import (
    artifact_results,
    clear_runtime_artifacts,
)
from src.backend.application.agent.case_agent.graph.builder import build_case_agent_graph
from src.backend.application.agent.case_agent.context.refresh_service import (
    CaserContextRefreshService,
)
from src.backend.application.agent.case_agent.memory.summary import next_summary
from src.backend.application.agent.case_agent.memory.task_state import task_state_payload
from src.backend.application.agent.case_agent.nodes.fail_closed import fail_closed_node
from src.backend.application.agent.case_agent.nodes.persist_result import persist_result_node
from src.backend.application.agent.case_agent.prompts.system import SYSTEM_BOUNDARY
from src.backend.application.agent.case_agent.safety.guardrails import (
    CaseAgentSafetyError,
    assert_safe_text,
    validate_answer,
)
from src.backend.application.agent.case_agent.service.note_service import (
    CaseAgentNoteService,
)
from src.backend.application.agent.case_agent.service.session_service import (
    CaseAgentSessionService,
)
from src.backend.application.agent.case_agent.tools.registry import (
    CaseAgentToolRegistry,
    CaseAgentToolResult,
)
from src.backend.application.agent.expert_agent.service.orchestrator import (
    ExpertAgentResult,
    ExpertAgentService,
)
from src.backend.application.agent.expert_agent.schemas import (
    ExpertAnalysisBudget,
    ExpertAnalysisConstraints,
    ExpertAnalysisTask,
)
from src.backend.application.agent.expert_agent.tools.case_context_gateway import (
    CaseContextGateway,
)
from src.backend.application.agent.expert_agent.tools.policy_rag_mcp import (
    PolicyRagMcpClient,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.application.audit.review.manage_notes_uc import ManageNotesUseCase
from src.backend.application.case_memory.service import CaseMemoryService
from src.backend.application.ports.agent_gateway import ModelGateway
from src.backend.application.ports.case_agent import CaseAgentRepository
from src.backend.core.exceptions import BusinessValidationError, ResourceNotFoundError
from src.backend.domain.audit.review.entities import AuthenticatedUser, find_forbidden_content
from src.backend.domain.audit.review.workflow_projector import AuditNote
from src.backend.domain.case_agent.entities import (
    CaseAgentAdoptNoteInput,
    CaseAgentAnswer,
    CaseAgentContentBlock,
    CaseAgentCreateSessionInput,
    CaseAgentEvent,
    CaseAgentMarkReadInput,
    CaseAgentRenameSessionInput,
    CaseAgentRun,
    CaseAgentSendMessageInput,
    CaseAgentSession,
    CaseAgentSource,
    CaseAgentSourceDetail,
    CaseAgentSourceDetailField,
)
from src.backend.domain.caser_context.entities import CaserContextSection
from src.backend.domain.case_memory.entities import (
    MemoryHintPack,
    MemoryRecallRequest,
    MemoryScope,
    MemoryType,
)


class CaseAgentRunCancelled(RuntimeError):
    """Internal control-flow signal for a persisted cooperative cancellation."""


class CaseAgentService:
    """Lightweight, case-bound assistant service."""

    def __init__(
        self,
        *,
        repository: CaseAgentRepository,
        gateway: ModelGateway,
        tools: CaseAgentToolRegistry,
        expert_agent: ExpertAgentService,
        manage_notes: ManageNotesUseCase,
        model_name: str,
        classifier_model: str,
        generator_model: str,
        max_concurrency: int,
        max_tool_calls: int,
        fast_mode_enabled: bool = True,
        caser_context: CaserContextRefreshService | None = None,
        policy_rag_client: PolicyRagMcpClient | None = None,
        policy_rag_timeout_ms: int = 20000,
        expert_gateway: ModelGateway | None = None,
        intent_biencoder: Any | None = None,
        case_memory: CaseMemoryService | None = None,
        semantic_timeout_seconds: int = 12,
        planner_timeout_seconds: int = 15,
        answer_timeout_seconds: int = 30,
        semantic_max_tokens: int = 384,
        planner_max_tokens: int = 768,
        answer_max_tokens: int = 1536,
    ) -> None:
        self._repository = repository
        self._gateway = gateway
        self._expert_gateway = expert_gateway
        self._tools = tools
        self._expert_agent = expert_agent
        self._policy_rag_client = policy_rag_client
        self._policy_rag_timeout_ms = max(1000, int(policy_rag_timeout_ms))
        self._intent_biencoder = intent_biencoder
        self._case_memory = case_memory
        self._manage_notes = manage_notes
        self._caser_context = caser_context
        self._model_name = model_name
        self._classifier_model = classifier_model
        self._generator_model = generator_model
        self._max_tool_calls = max_tool_calls
        self._fast_mode_enabled = fast_mode_enabled
        self._semantic_timeout_seconds = max(3, int(semantic_timeout_seconds))
        self._planner_timeout_seconds = max(3, int(planner_timeout_seconds))
        self._answer_timeout_seconds = max(5, int(answer_timeout_seconds))
        self._semantic_max_tokens = max(128, int(semantic_max_tokens))
        self._planner_max_tokens = max(256, int(planner_max_tokens))
        self._answer_max_tokens = max(512, int(answer_max_tokens))
        self._sessions = CaseAgentSessionService(
            repository=repository,
            model_name=model_name,
        )
        self._notes = CaseAgentNoteService(
            repository=repository,
            manage_notes=manage_notes,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="case-agent",
        )
        self._memory_executor = ThreadPoolExecutor(
            max_workers=max(2, min(4, max_concurrency)),
            thread_name_prefix="case-memory",
        )
        self._policy_rag_prewarm_submitted = False
        self._policy_rag_prewarm_summary: dict[str, Any] | None = None
        self._session_memory_cache: dict[str, dict[str, Any]] = {}
        self._session_memory_cached_at: dict[str, float] = {}
        self._session_memory_futures: dict[str, Future[None]] = {}
        self._request_memory_futures: dict[tuple[str, str], Future[dict[str, Any]]] = {}
        self._actor_memory_departments: dict[str, str] = {}
        self._session_memory_lock = RLock()
        self._request_memory_lock = RLock()
        self._run_futures: dict[str, Future[None]] = {}
        self._run_futures_lock = RLock()
        self._graph = build_case_agent_graph(self)

    def list_sessions(self, case_id: str, actor: AuthenticatedUser) -> list[CaseAgentSession]:
        return self._sessions.list_sessions(case_id, actor)

    def create_session(
        self,
        case_id: str,
        body: CaseAgentCreateSessionInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentSession:
        return self._sessions.create_session(case_id, body, actor)

    def get_session(self, session_id: str, actor: AuthenticatedUser) -> CaseAgentSession:
        return self._sessions.get_session(session_id, actor)

    def archive_session(self, session_id: str, actor: AuthenticatedUser) -> dict[str, bool]:
        return self._sessions.archive_session(session_id, actor)

    def restore_session(self, session_id: str, actor: AuthenticatedUser) -> CaseAgentSession:
        return self._sessions.restore_session(session_id, actor)

    def rename_session(
        self,
        session_id: str,
        body: CaseAgentRenameSessionInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentSession:
        return self._sessions.rename_session(session_id, body, actor)

    def mark_session_read(
        self,
        session_id: str,
        body: CaseAgentMarkReadInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentSession:
        try:
            session = self._repository.mark_session_read(
                session_id=session_id,
                actor_id=actor.id,
                through_run_id=body.through_run_id,
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Case Agent 会话或运行不存在") from exc
        if session is None:
            raise ResourceNotFoundError("Case Agent 会话不存在")
        return session

    def send_message(
        self,
        session_id: str,
        body: CaseAgentSendMessageInput,
        actor: AuthenticatedUser,
    ) -> CaseAgentRun:
        content = body.content.strip()
        if actor.department:
            self._actor_memory_departments[actor.id] = actor.department
        forbidden = find_forbidden_content(content)
        if forbidden:
            raise BusinessValidationError([f"案件助手问题不得包含禁止字段或身份信息：{forbidden}"])
        try:
            assert_safe_text(content, allow_decision_request=True)
        except CaseAgentSafetyError as exc:
            raise BusinessValidationError([str(exc)]) from exc
        session = self.get_session(session_id, actor)
        client_request_id = body.client_request_id or f"server_{uuid4().hex}"
        try:
            user_message, run, created = self._repository.create_turn(
                session_id=session_id,
                actor_id=actor.id,
                content=content,
                active_stage=body.active_stage,
                model_name=self._model_name,
                client_request_id=client_request_id,
                case_context_fingerprint=self.case_context_fingerprint(session.case_id),
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Case Agent 会话不存在") from exc
        if not created:
            self._submit_run(run.run_id, actor.id)
            return run
        if run.parent_run_id is not None:
            self._repository.append_event(
                run.parent_run_id,
                "resumed",
                "审核人员已补充信息，追问已接续到新运行",
                {"resumed_by_run_id": run.run_id},
            )
        self._repository.append_event(
            run.run_id,
            "message_received",
            "已接收审核人员问题",
            {
                "message_id": user_message.message_id,
                "active_stage": body.active_stage,
                "parent_run_id": run.parent_run_id,
            },
        )
        if run.parent_run_id is not None:
            self._repository.append_event(
                run.run_id,
                "resuming",
                "正在接续上一轮追问",
                {"parent_run_id": run.parent_run_id},
            )
        self._submit_run(run.run_id, actor.id)
        return self._repository.get_run(run_id=run.run_id, actor_id=actor.id) or run

    def cancel_run(self, run_id: str, actor: AuthenticatedUser) -> CaseAgentRun:
        try:
            run, changed = self._repository.request_run_cancel(
                run_id=run_id,
                actor_id=actor.id,
                reason="user_requested",
            )
        except KeyError as exc:
            raise ResourceNotFoundError("Case Agent 运行不存在") from exc
        if changed:
            event_type = "cancelled" if run.status == "cancelled" else "cancel_requested"
            message = "任务已停止" if run.status == "cancelled" else "已收到停止请求，将在当前步骤结束后停止"
            self._repository.append_event(
                run.run_id,
                event_type,
                message,
                {"reason": "user_requested"},
            )
        if run.status == "cancelled":
            with self._run_futures_lock:
                future = self._run_futures.get(run.run_id)
            if future is not None:
                future.cancel()
        return self._repository.get_run(run_id=run.run_id, actor_id=actor.id) or run

    def _submit_run(self, run_id: str, actor_id: str) -> None:
        run = self._repository.get_run(run_id=run_id, actor_id=actor_id)
        if run is None or run.status in {
            "waiting_for_user",
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "degraded",
        }:
            return
        with self._run_futures_lock:
            existing = self._run_futures.get(run_id)
            if existing is not None and not existing.done():
                return
            future = self._executor.submit(self._execute_run, run_id, actor_id)
            self._run_futures[run_id] = future
            future.add_done_callback(lambda _future, rid=run_id: self._forget_run_future(rid))

    def _forget_run_future(self, run_id: str) -> None:
        with self._run_futures_lock:
            self._run_futures.pop(run_id, None)

    def recover_persisted_runs(self) -> int:
        """Resume non-terminal runs after a single-process service restart.

        PostgreSQL remains authoritative. Runs that had already entered execution
        are restarted from the graph entry because Case Agent tools are read-only;
        persisted events keep the recovery visible to the auditor.
        """

        lister = getattr(self._repository, "list_recoverable_runs", None)
        if not callable(lister):
            return 0
        recovered = 0
        for run in lister():
            if run.cancel_requested_at is not None:
                self._mark_run_cancelled(run.run_id, reason="service_recovery")
                continue
            if run.status in {"running", "resuming"}:
                self._repository.update_run(
                    run.run_id,
                    status="created",
                    current_node="recovery_queued",
                )
            self._repository.append_event(
                run.run_id,
                "recovered",
                "服务恢复后已重新加入执行队列",
                {"previous_status": run.status},
            )
            self._submit_run(run.run_id, run.actor_id)
            recovered += 1
        return recovered

    def case_context_fingerprint(self, case_id: str) -> str | None:
        """Return a stable digest of the current safe Caser sections."""

        if self._caser_context is None:
            return None
        try:
            sections = self._caser_context.list_sections(case_id)
        except Exception:
            return None
        payload = [
            {
                "section_key": section.section_key,
                "payload_hash": section.payload_hash,
                "source_versions": section.source_versions,
            }
            for section in sorted(sections, key=lambda item: item.section_key)
        ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_run(self, run_id: str, actor: AuthenticatedUser) -> CaseAgentRun:
        run = self._repository.get_run(run_id=run_id, actor_id=actor.id)
        if run is None:
            raise ResourceNotFoundError("Case Agent 运行不存在")
        return run

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[CaseAgentEvent]:
        return self._repository.list_events(run_id, after_sequence)

    def adopt_note(
        self,
        session_id: str,
        message_id: str,
        body: CaseAgentAdoptNoteInput,
        actor: AuthenticatedUser,
    ) -> AuditNote:
        return self._notes.adopt_note(session_id, message_id, body, actor)

    def rebuild_caser_context(
        self,
        case_id: str,
        actor: AuthenticatedUser,
        section_keys: list[str] | None = None,
    ) -> list[CaserContextSection]:
        _ = actor
        if self._caser_context is None:
            raise ResourceNotFoundError("Caser context service is not available")
        return self._caser_context.rebuild_case(
            case_id,
            section_keys=section_keys,
            generated_by="api:manual",
        )

    def list_caser_context_sections(
        self,
        case_id: str,
        actor: AuthenticatedUser,
    ) -> list[CaserContextSection]:
        _ = actor
        if self._caser_context is None:
            raise ResourceNotFoundError("Caser context service is not available")
        return self._caser_context.list_sections(case_id)

    def get_caser_context_section(
        self,
        case_id: str,
        section_key: str,
        actor: AuthenticatedUser,
    ) -> CaserContextSection:
        _ = actor
        if self._caser_context is None:
            raise ResourceNotFoundError("Caser context service is not available")
        return self._caser_context.get_section(case_id, section_key)

    def prewarm_policy_rag(self) -> bool:
        """Preload the read-only Policy RAG runtime in the background."""

        if self._policy_rag_prewarm_submitted:
            return False
        prewarm = getattr(self._policy_rag_client, "prewarm", None)
        if not callable(prewarm):
            return False
        self._policy_rag_prewarm_submitted = True
        self._executor.submit(self._prewarm_policy_rag_runtime, prewarm)
        return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
        self._memory_executor.shutdown(wait=False, cancel_futures=False)
        if self._case_memory is not None:
            self._case_memory.cache.clear_all()
        if self._caser_context is not None:
            self._caser_context.shutdown()

    def bootstrap_session_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return ready Session memory immediately and warm it in the background."""

        if self._case_memory is None:
            return {"enabled": False, "prompt_contexts": [], "memory_ids": []}
        session = state.get("session")
        run = state.get("run")
        session_id = str(getattr(session, "session_id", "") or "")
        if not session_id or run is None:
            return {"enabled": False, "prompt_contexts": [], "memory_ids": []}
        with self._session_memory_lock:
            cached = self._session_memory_cache.get(session_id)
            cached_at = self._session_memory_cached_at.get(session_id, 0.0)
            if cached is not None and time.monotonic() - cached_at <= 300:
                return cached
            if cached is not None:
                self._session_memory_cache.pop(session_id, None)
                self._session_memory_cached_at.pop(session_id, None)
            future = self._session_memory_futures.get(session_id)
            if future is None:
                future = self._memory_executor.submit(
                    self._warm_session_memory,
                    session_id,
                    str(state.get("actor_id") or ""),
                    str(getattr(session, "current_topic", "") or "")[:300],
                )
                self._session_memory_futures[session_id] = future
                if future.done():
                    self._session_memory_futures.pop(session_id, None)
        return {
            "enabled": True,
            "status": "in_flight",
            "prompt_contexts": [],
            "memory_ids": [],
        }

    def session_memory_context(self, state: dict[str, Any]) -> dict[str, Any]:
        """Read the Session cache without waiting for its background future."""

        session_id = str(getattr(state.get("session"), "session_id", "") or "")
        with self._session_memory_lock:
            cached = self._session_memory_cache.get(session_id)
            cached_at = self._session_memory_cached_at.get(session_id, 0.0)
        if cached is not None and time.monotonic() - cached_at <= 300:
            return cached
        current = state.get("session_memory_context")
        if isinstance(current, dict):
            return current
        return {"enabled": self._case_memory is not None, "status": "not_requested", "prompt_contexts": [], "memory_ids": []}

    def _warm_session_memory(
        self,
        session_id: str,
        actor_id: str,
        current_topic: str,
    ) -> None:
        result = {
            "enabled": True,
            "status": "empty",
            "prompt_contexts": [],
            "trace_refs": [],
            "memory_ids": [],
        }
        try:
            if self._case_memory is not None:
                pack = self._case_memory.recall(
                    MemoryRecallRequest(
                        request_id=f"session:{session_id}",
                        consumer="answer_generator",
                        memory_type=MemoryType.ANSWER_STYLE,
                        scope=MemoryScope(scope_type="auditor", scope_id=actor_id),
                        scope_fallbacks=self._memory_scope_fallbacks(actor_id),
                        task_context={
                            "session_id": session_id,
                            "current_topic": current_topic,
                        },
                        max_items=2,
                    )
                )
                result = {
                    "enabled": True,
                    "status": "hit" if pack.memory_ids else "empty",
                    "prompt_contexts": pack.prompt_contexts[:2],
                    "trace_refs": pack.trace_refs[:2],
                    "memory_ids": list(dict.fromkeys(pack.memory_ids)),
                }
        except Exception:
            result["status"] = "failed"
        finally:
            with self._session_memory_lock:
                self._session_memory_cache[session_id] = result
                self._session_memory_cached_at[session_id] = time.monotonic()
                self._session_memory_futures.pop(session_id, None)

    def start_request_memory_prefetch(self, state: dict[str, Any]) -> dict[str, Any]:
        """Start only the post-intent memory types selected by the deterministic plan."""

        plan = state.get("memory_retrieval_plan") or {}
        items = plan.get("items") if isinstance(plan, dict) else []
        if self._case_memory is None or not isinstance(items, list) or not items:
            return {"enabled": self._case_memory is not None, "types": {}}

        run = state["run"]
        context = self._memory_task_context(state)
        scope = MemoryScope(scope_type="auditor", scope_id=str(state.get("actor_id") or ""))
        statuses: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                memory_type = MemoryType(str(item.get("memory_type") or ""))
            except ValueError:
                continue
            consumers = {
                str(value)
                for value in list(item.get("consumers") or [])
                if str(value)
            }
            if not consumers:
                continue
            status = self._case_memory.cache.begin(run.run_id, memory_type)
            statuses[memory_type.value] = status
            if status != "in_flight":
                continue
            request = MemoryRecallRequest(
                request_id=run.run_id,
                consumer=sorted(consumers)[0],
                memory_type=memory_type,
                scope=scope,
                scope_fallbacks=self._memory_scope_fallbacks(str(state.get("actor_id") or "")),
                task_context=context,
                max_items=max(1, min(8, int(item.get("max_items") or 3))),
            )
            key = (run.run_id, memory_type.value)
            with self._request_memory_lock:
                if key in self._request_memory_futures:
                    continue
                self._request_memory_futures[key] = self._memory_executor.submit(
                    self._prefetch_planned_memory,
                    request,
                    consumers,
                )
        return {"enabled": True, "request_id": run.run_id, "types": statuses}

    def _prefetch_planned_memory(
        self,
        request: MemoryRecallRequest,
        consumers: set[str],
    ) -> dict[str, Any]:
        try:
            if self._case_memory is None:
                return {"status": "failed", "count": 0}
            count = self._case_memory.prefetch_type(
                request,
                allowed_consumers=consumers,
            )
            return {
                "status": self._case_memory.cache.status(request.request_id, request.memory_type),
                "count": count,
            }
        except Exception:
            if self._case_memory is not None:
                self._case_memory.cache.mark_terminal(
                    request.request_id,
                    request.memory_type,
                    "failed",
                )
            return {"status": "failed", "count": 0}

    def prefetch_request_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prefetch candidates once after the question and basic entities are known."""

        if self._case_memory is None:
            return {"enabled": False, "candidate_counts": {}}
        run = state["run"]
        context = self._memory_task_context(state)
        counts = self._case_memory.prefetch_request(
            request_id=run.run_id,
            scope=MemoryScope(scope_type="auditor", scope_id=str(state.get("actor_id") or "")),
            task_context=context,
        )
        return {"enabled": True, "request_id": run.run_id, "candidate_counts": counts}

    def memory_hint_for_node(
        self,
        state: dict[str, Any],
        *,
        consumer: str,
        memory_type: MemoryType,
        task_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Consume a planned memory type with a bounded wait and no empty-result retry."""

        if self._case_memory is None:
            return MemoryHintPack(
                request_id=str(state.get("run_id") or ""),
                consumer=consumer,
                memory_type=memory_type,
            ).model_dump(mode="json")
        packs = state.setdefault("memory_hint_packs", {})
        key = f"{consumer}:{memory_type.value}"
        cached = packs.get(key)
        if isinstance(cached, dict):
            return cached
        run = state["run"]
        session = state.get("session")
        context = self._memory_task_context(state, task_context)
        request = MemoryRecallRequest(
            request_id=run.run_id,
            consumer=consumer,
            memory_type=memory_type,
            scope=MemoryScope(scope_type="auditor", scope_id=str(state.get("actor_id") or "")),
            scope_fallbacks=self._memory_scope_fallbacks(str(state.get("actor_id") or "")),
            task_context=context,
        )
        wait_started = time.perf_counter()
        status = self._case_memory.cache.status(run.run_id, memory_type)
        wait_budget_ms = self._memory_wait_budget_ms(state, memory_type, consumer)
        if status == "in_flight" and wait_budget_ms > 0:
            with self._request_memory_lock:
                future = self._request_memory_futures.get((run.run_id, memory_type.value))
            if future is not None:
                try:
                    future.result(timeout=wait_budget_ms / 1000)
                except FutureTimeoutError:
                    self._case_memory.cache.mark_terminal(run.run_id, memory_type, "timed_out")
                except Exception:
                    self._case_memory.cache.mark_terminal(run.run_id, memory_type, "failed")
            status = self._case_memory.cache.status(run.run_id, memory_type)

        if status == "not_requested" and memory_type == MemoryType.FAILURE:
            pack = self._case_memory.recall(request)
            state.setdefault("memory_fallbacks", []).append(
                {"consumer": consumer, "memory_type": memory_type.value, "reason": "failure_recovery"}
            )
            status = "hit" if pack.memory_ids else "empty"
        elif status == "not_requested":
            pack = MemoryHintPack(
                request_id=run.run_id,
                consumer=consumer,
                memory_type=memory_type,
            )
        else:
            pack = self._case_memory.recall_cached(request)
        payload = pack.model_dump(mode="json")
        packs[key] = payload
        wait_ms = round((time.perf_counter() - wait_started) * 1000, 3)
        state.setdefault("request_memory_status", {}).setdefault("types", {})[
            memory_type.value
        ] = status
        state.setdefault("memory_consumption_status", {})[key] = {
            "status": status,
            "wait_ms": wait_ms,
            "wait_budget_ms": wait_budget_ms,
            "memory_count": len(pack.memory_ids),
        }
        if pack.memory_ids:
            for memory_id in pack.memory_ids:
                self._case_memory.mark_used(memory_id)
        return payload

    def record_intent_memory_shadow(self, state: dict[str, Any]) -> dict[str, Any]:
        """Record non-blocking intent-memory agreement without changing routing."""

        run = state.get("run")
        if self._case_memory is None or run is None:
            return {"status": "disabled", "memory_ids": []}
        status = self._case_memory.cache.status(run.run_id, MemoryType.INTENT_ROUTE)
        if status not in {"hit", "empty"}:
            return {"status": status, "memory_ids": []}
        request = MemoryRecallRequest(
            request_id=run.run_id,
            consumer="intent_router",
            memory_type=MemoryType.INTENT_ROUTE,
            scope=MemoryScope(scope_type="auditor", scope_id=str(state.get("actor_id") or "")),
            scope_fallbacks=self._memory_scope_fallbacks(str(state.get("actor_id") or "")),
            task_context=self._memory_task_context(state),
            max_items=3,
        )
        pack = self._case_memory.recall_cached(request)
        suggested_tags: list[str] = []
        for hint in pack.control_hints:
            for section_name in ("structured_content", "recommended_action", "stable_preferences"):
                section = hint.get(section_name)
                if not isinstance(section, dict):
                    continue
                suggested_tags.extend(self._safe_memory_values(section.get("route_tags")))
        current_intent = str(state.get("intent") or "")
        result = {
            "status": "hit" if pack.memory_ids else "empty",
            "mode": "shadow",
            "current_intent": current_intent,
            "suggested_route_tags": list(dict.fromkeys(suggested_tags))[:8],
            "agrees_with_current_route": current_intent in set(suggested_tags),
            "memory_ids": list(pack.memory_ids),
        }
        state["intent_memory_shadow"] = result
        self._repository.append_event(
            run.run_id,
            "intent_memory_shadow_evaluated",
            "已完成意图记忆影子评估，未改变当前路由",
            {
                "memory_count": len(pack.memory_ids),
                "suggestion_count": len(result["suggested_route_tags"]),
                "agrees_with_current_route": result["agrees_with_current_route"],
            },
        )
        return result

    def _memory_task_context(
        self,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the transient, type-neutral task profile used by all memory queries."""

        perceptual = state.get("perceptual_state") or {}
        semantic = dict(perceptual.get("semantic") or {})
        semantic_frame = state.get("semantic_frame") or {}
        focus = semantic_frame.get("focus_candidate") or {}
        query_semantics = state.get("query_semantics") or {}
        semantic.setdefault("intent", state.get("intent") or semantic_frame.get("business_intent"))
        semantic.setdefault("action", focus.get("action"))
        semantic.setdefault("target_objects", focus.get("target_objects"))
        semantic.setdefault("information_needs", focus.get("information_needs"))
        semantic.setdefault("evidence_need", semantic_frame.get("evidence_need"))
        semantic.setdefault("answer_shape", semantic_frame.get("answer_shape"))
        semantic.setdefault("target_layer_hint", focus.get("target_layer_hint"))

        overrides = dict(extra or {})
        filters: dict[str, Any] = {}
        for value in (
            semantic.get("filters"),
            focus.get("filters"),
            focus.get("policy_filters"),
            overrides.get("filters"),
        ):
            filters.update(self._safe_policy_filters(value))
        explicit_entities = (state.get("entity_frame") or {}).get("explicit_entities") or {}
        if isinstance(explicit_entities, dict):
            filters.update(self._safe_policy_filters(explicit_entities))

        task_type = str(
            overrides.get("task_type")
            or semantic.get("intent")
            or query_semantics.get("intent")
            or ""
        )[:80]
        action = str(overrides.get("action") or semantic.get("action") or "")[:80]
        target_objects = self._safe_memory_values(
            overrides.get("target_objects")
            or semantic.get("target_objects")
            or query_semantics.get("target_objects")
        )
        information_needs = self._safe_memory_values(
            overrides.get("information_needs")
            or semantic.get("information_needs")
            or query_semantics.get("information_needs")
        )
        evidence_need = str(
            overrides.get("evidence_need") or semantic.get("evidence_need") or ""
        )[:80]
        required_capabilities = self._safe_memory_values(
            overrides.get("required_capabilities")
            or [
                item.get("capability") or item.get("name")
                for item in state.get("execution_plan", [])
                if isinstance(item, dict)
            ]
        )
        target_layer = str(semantic.get("target_layer_hint") or "none")
        task_complexity = str(overrides.get("task_complexity") or "")
        if not task_complexity:
            task_complexity = (
                "multi_step"
                if len(required_capabilities) > 1
                or len(information_needs) > 1
                or target_layer in {"L2", "L3"}
                else "single_step"
            )
        task_pattern = str(
            overrides.get("task_pattern")
            or self._memory_task_pattern(task_type, action, target_layer, evidence_need)
        )[:240]
        user_message = state.get("user_message")
        context: dict[str, Any] = {
            "question": str(getattr(user_message, "content", "") or "")[:800],
            "task_type": task_type,
            "intent": task_type,
            "task_pattern": task_pattern,
            "task_complexity": task_complexity,
            "action": action,
            "target_objects": target_objects,
            "evidence_need": evidence_need,
            "information_needs": information_needs,
            "required_capabilities": required_capabilities,
            "scenario_type": str(
                overrides.get("scenario_type")
                or query_semantics.get("granularity")
                or semantic.get("answer_shape")
                or "all"
            )[:80],
            "filters": filters,
        }
        jurisdiction = filters.get("jurisdiction") or filters.get("region")
        if jurisdiction not in (None, ""):
            context["jurisdiction"] = jurisdiction
        policy_domain = filters.get("policy_domain") or filters.get("topic")
        if policy_domain not in (None, ""):
            context["policy_domain"] = policy_domain
        context.update(overrides)
        return context

    @staticmethod
    def _memory_task_pattern(
        task_type: str,
        action: str,
        target_layer: str,
        evidence_need: str,
    ) -> str:
        parts = [
            f"intent={task_type}" if task_type else "",
            f"action={action}" if action else "",
            f"layer={target_layer}" if target_layer and target_layer != "none" else "",
            f"evidence={evidence_need}" if evidence_need and evidence_need != "none" else "",
        ]
        return "|".join(part for part in parts if part)

    def _memory_scope_fallbacks(self, actor_id: str) -> list[MemoryScope]:
        department = self._actor_memory_departments.get(actor_id)
        if not department:
            return []
        return [MemoryScope(scope_type="department", scope_id=department)]

    @staticmethod
    def _memory_wait_budget_ms(
        state: dict[str, Any],
        memory_type: MemoryType,
        consumer: str,
    ) -> int:
        plan = state.get("memory_retrieval_plan") or {}
        for item in plan.get("items", []) if isinstance(plan, dict) else []:
            if not isinstance(item, dict):
                continue
            if item.get("memory_type") != memory_type.value:
                continue
            if consumer not in list(item.get("consumers") or []):
                continue
            return max(0, min(1000, int(item.get("wait_timeout_ms") or 0)))
        return 0

    def _handle_node_failure(self, state: dict[str, Any], node_name: str) -> None:
        """Final node failures are trace events, not reusable long-term memories."""

        state["failure_recovery_hint"] = {}

    def enqueue_completed_run_memory_capture(self, state: dict[str, Any]) -> int:
        """Queue deterministic, redacted observations after a successful run."""

        run = state.get("run")
        if self._case_memory is None or run is None:
            return 0
        observations: list[dict[str, Any]] = []
        failure = self._successful_repair_observation(state, run.run_id)
        if failure is not None:
            observations.append(failure)
        route = self._intent_route_observation(state, run.run_id)
        if route is not None:
            observations.append(route)
        policy = self._policy_search_observation(state, run.run_id)
        if policy is not None:
            observations.append(policy)
        plan = self._decision_plan_observation(state, run.run_id)
        if plan is not None:
            observations.append(plan)
        style = self._answer_style_observation(state, run.run_id)
        if style is not None:
            observations.append(style)
        if not observations:
            return 0
        self._case_memory.enqueue_capture_request(
            source_run_id=str(run.run_id),
            scope=MemoryScope(scope_type="auditor", scope_id=str(state.get("actor_id") or run.actor_id)),
            observations=observations,
        )
        return len(observations)

    def _successful_repair_observation(
        self,
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        if (
            int(state.get("validation_retry_count") or 0) < 1
            and not state.get("validation_repair_succeeded")
        ):
            return None
        error_code = str(state.get("validation_error") or "").split(":", 1)[0]
        if error_code not in {
            "invalid_json",
            "invalid_schema",
            "citation_invalid",
            "citation_missing_for_grounded",
        }:
            return None
        error_category = self._validation_error_category(error_code)
        recovery_mode = str(state.get("validation_recovery_mode") or "second_model_call")
        action_payload = {
            "repair_strategy": recovery_mode,
            "avoid": "不得保留无法通过结构或引用校验的内容",
            "recover": "仅使用当前受控证据上下文恢复，并再次执行完整校验",
        }
        return {
            "memory_type": MemoryType.FAILURE.value,
            "event": {
                "event_type": "answer_validation_repaired",
                "source_event_id": f"{run_id}:answer_validation_repaired:{error_code}",
                "node": "validate_answer",
                "validation_codes": [error_code],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": f"回答出现 {error_code} 时，受控重写后通过二次校验",
                "structured_content": {
                    "match_profile": {
                        "node": "validate_answer",
                        "validation_codes": [error_code],
                        "error_category": error_category,
                        "component_version": "case_agent_answer_contract_v1",
                        "trigger_conditions": ["repairable_validation_error", "single_attempt_budget"],
                    },
                    "action_payload": action_payload,
                    "repair_strategy": recovery_mode,
                    "validation_code": error_code,
                    "error_category": error_category,
                    "component_version": "case_agent_answer_contract_v1",
                    "avoid": action_payload["avoid"],
                    "recover": action_payload["recover"],
                    "attempt_count": 1,
                    "validation": "passed",
                },
                "detail": "只保存校验码、修复动作和通过结果，不保存失败回答、Prompt 或案件事实。",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["recovery_handler"],
            "confidence": 0.95,
            "feature": "deterministic_structure_repair",
        }

    def _intent_route_observation(
        self,
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        confidence = float(state.get("intent_confidence") or 0.0)
        intent = str(state.get("intent") or "")
        if confidence < 0.9 or intent in {"", "general_help", "clarification_required"}:
            return None
        capabilities = [
            str(item.get("capability") or item.get("name") or "")
            for item in state.get("execution_plan", [])
            if isinstance(item, dict) and str(item.get("capability") or item.get("name") or "")
        ][:8]
        context = self._memory_task_context(state, {"required_capabilities": capabilities})
        match_profile = {
            key: context.get(key)
            for key in (
                "task_type",
                "task_pattern",
                "action",
                "target_objects",
                "evidence_need",
            )
            if context.get(key) not in (None, "", [])
        }
        action_payload = {
            "intent": intent,
            "route_tags": [intent],
            "capabilities": capabilities,
        }
        return {
            "memory_type": MemoryType.INTENT_ROUTE.value,
            "event": {
                "event_type": "high_confidence_route_completed",
                "source_event_id": f"{run_id}:route:{intent}",
                "node": "validate_execution_plan",
                "validation_codes": ["route_completed"],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": f"高置信度意图 {intent} 使用路由 {', '.join(capabilities) or 'direct_answer'}",
                "structured_content": {
                    "match_profile": match_profile,
                    "action_payload": action_payload,
                    "route_tags": [intent],
                    "capabilities": capabilities,
                },
                "detail": "由已完成请求的受控路由结果提取，不保存用户原始问题。",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["intent_router"],
            "confidence": confidence,
            "feature": "intent_route",
        }

    def _policy_search_observation(
        self,
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        if not self._capability_succeeded(state, "ask_policy_expert"):
            return None
        policy_step = next(
            (
                item for item in state.get("execution_plan", [])
                if isinstance(item, dict)
                and str(item.get("capability") or item.get("name") or "") == "ask_policy_expert"
            ),
            None,
        )
        if policy_step is None:
            return None
        arguments = policy_step.get("arguments") if isinstance(policy_step.get("arguments"), dict) else {}
        filters = self._safe_policy_filters(arguments.get("filters"))
        needs = self._safe_memory_values(
            arguments.get("information_needs")
            or state.get("query_semantics", {}).get("information_needs")
        )
        if not filters and not needs:
            return None
        context = self._memory_task_context(
            state,
            {"filters": filters, "information_needs": needs},
        )
        match_profile = {
            key: context.get(key)
            for key in (
                "task_type",
                "task_pattern",
                "action",
                "target_objects",
                "evidence_need",
                "jurisdiction",
                "policy_domain",
                "information_needs",
            )
            if context.get(key) not in (None, "", [])
        }
        action_payload = {
            "filters": filters,
            "information_needs": needs,
            "coverage_requirements": [f"回答覆盖：{item}" for item in needs],
            "evidence_focus": ["policy_clause", "service_guide"],
            "citation_expectations": ["使用当前检索得到的可核验政策来源"],
            "avoid_claims": ["不得把历史检索偏好当作当前政策结论"],
        }
        summary_terms = [str(value) for value in filters.values()] + needs
        summary = "、".join(summary_terms[:8])
        return {
            "memory_type": MemoryType.POLICY_SEARCH.value,
            "event": {
                "event_type": "validated_policy_search_completed",
                "source_event_id": f"{run_id}:policy_search",
                "node": "call_expert_analysis",
                "validation_codes": ["policy_result_available"],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": f"政策检索使用已验证的过滤条件和信息需求：{summary}",
                "structured_content": {
                    "match_profile": match_profile,
                    "action_payload": action_payload,
                    **action_payload,
                },
                "detail": "来自成功政策分析的检索方案，只保存治理后的过滤条件和回答信息点。",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["policy_filter_resolver", "expert_analysis"],
            "confidence": max(0.7, float(state.get("intent_confidence") or 0.0)),
            "feature": "policy_search",
        }

    def _decision_plan_observation(
        self,
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        if state.get("planning_error"):
            return None
        execution_plan = [item for item in state.get("execution_plan", []) if isinstance(item, dict)]
        if len(execution_plan) < 2:
            return None
        steps = [str(item.get("capability") or item.get("name") or "") for item in execution_plan]
        steps = [item for item in steps if item][:8]
        if len(steps) < 2:
            return None
        dependencies = {
            str(item.get("capability") or item.get("name") or ""): [
                str(value) for value in item.get("depends_on", [])[:8]
            ]
            for item in execution_plan
            if item.get("depends_on")
        }
        context = self._memory_task_context(
            state,
            {
                "required_capabilities": steps,
                "task_complexity": "multi_step",
            },
        )
        match_profile = {
            key: context.get(key)
            for key in (
                "task_type",
                "task_pattern",
                "task_complexity",
                "action",
                "target_objects",
                "evidence_need",
                "required_capabilities",
            )
            if context.get(key) not in (None, "", [])
        }
        action_payload = {
            "planning_steps": steps,
            "dependencies": dependencies,
            "constraints": ["只调用白名单能力", "保持当前案件隔离"],
        }
        return {
            "memory_type": MemoryType.DECISION_PLAN.value,
            "event": {
                "event_type": "validated_plan_completed",
                "source_event_id": f"{run_id}:decision_plan",
                "node": "validate_execution_plan",
                "validation_codes": ["plan_completed"],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": f"已验证的多步骤规划：{' -> '.join(steps)}",
                "structured_content": {
                    "match_profile": match_profile,
                    "action_payload": action_payload,
                    **action_payload,
                },
                "detail": "仅保留能力顺序和依赖，不保存参数、案件事实或模型推理。",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["decision_planner"],
            "confidence": max(0.7, float(state.get("intent_confidence") or 0.0)),
            "feature": "decision_plan",
        }

    def _answer_style_observation(
        self,
        state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        user_message = state.get("user_message")
        content = str(getattr(user_message, "content", "") or "")
        if not content:
            return None
        preferences: dict[str, Any] = {}
        labels: list[str] = []
        if re.search(r"先(?:说|给|写)?结论|结论前置|先讲结论", content):
            preferences["conclusion_first"] = True
            labels.append("结论前置")
        if re.search(r"(?:简洁|简短|精简|少写一点|不要展开)", content):
            preferences["verbosity"] = "concise"
            labels.append("简洁回答")
        elif re.search(r"(?:详细|展开说明|具体说明|多写一点)", content):
            preferences["verbosity"] = "detailed"
            labels.append("详细回答")
        if re.search(r"(?:分点|逐条|列表|要点形式)", content):
            preferences["list_format"] = True
            labels.append("分点组织")
        if re.search(r"(?:用表格|表格形式)", content):
            preferences["table_format"] = True
            labels.append("表格组织")
        elif re.search(r"(?:不要表格|不用表格)", content):
            preferences["table_format"] = False
            labels.append("不使用表格")
        if not preferences:
            return None
        context = self._memory_task_context(state)
        scenario_type = str(context.get("scenario_type") or "all")
        action_payload = {
            "style_preferences": preferences,
            "override_rules": ["当前用户明确要求优先于历史风格偏好"],
        }
        return {
            "memory_type": MemoryType.ANSWER_STYLE.value,
            "event": {
                "event_type": "explicit_answer_style_requested",
                "source_event_id": f"{run_id}:answer_style",
                "node": "resolve_answer_style",
                "validation_codes": ["explicit_user_preference"],
                "source_ref_ids": [],
            },
            "memory": {
                "summary": f"用户明确偏好：{'、'.join(labels)}",
                "structured_content": {
                    "match_profile": {"scenario_type": scenario_type},
                    "action_payload": action_payload,
                    "style_preferences": preferences,
                },
                "detail": "由用户明确表达的回答组织要求提取，不保存原始对话。",
                "source_refs": [run_id],
            },
            "allowed_consumers": ["answer_generator"],
            "confidence": 0.95,
            "feature": "style_preference",
        }

    @staticmethod
    def _capability_succeeded(state: dict[str, Any], capability: str) -> bool:
        for name, result in artifact_results(state) or state.get("capability_results", []):
            if str(name) == capability and str(getattr(result, "status", "")) == "success":
                return True
        return False

    @staticmethod
    def _validation_error_category(error_code: str) -> str:
        if error_code in {"invalid_json", "invalid_schema"}:
            return "response_structure"
        if error_code in {"citation_invalid", "citation_missing_for_grounded"}:
            return "citation_integrity"
        return "answer_validation"

    @staticmethod
    def _safe_policy_filters(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        allowed_keys = {
            "jurisdiction",
            "region",
            "policy_domain",
            "content_type",
            "document_no",
            "issuing_authority",
            "status",
            "topic",
            "source_type",
        }
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:12]:
            if key not in allowed_keys or isinstance(item, (dict, list)):
                continue
            text = str(item).strip()[:80]
            if text and find_forbidden_content(text) is None:
                result[key] = text
        return result

    @staticmethod
    def _safe_memory_values(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value[:12]:
            text = str(item).strip()[:80]
            if text and find_forbidden_content(text) is None and text not in result:
                result.append(text)
        return result

    def _prewarm_policy_rag_runtime(self, prewarm: Any) -> None:
        try:
            summary = prewarm()
            self._policy_rag_prewarm_summary = summary if isinstance(summary, dict) else {}
        except Exception:
            self._policy_rag_prewarm_summary = {"status": "failed"}
            return

    def _invoke_node(self, node_name: str, node: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Run one graph node with durable execution and checkpoint records."""

        run_id = str(state.get("run_id") or getattr(state.get("run"), "run_id", ""))
        if run_id:
            self._raise_if_run_cancelled(run_id)
        execution_sequence: int | None = None
        if run_id:
            try:
                execution_sequence = self._repository.start_node_execution(
                    run_id,
                    node_name=node_name,
                    input_snapshot=self._node_snapshot(state),
                )
            except Exception:
                execution_sequence = None
        try:
            output = node(state)
        except Exception as exc:
            if run_id and execution_sequence is not None:
                try:
                    self._repository.fail_node_execution(
                        run_id,
                        sequence=execution_sequence,
                        error_code=exc.__class__.__name__,
                        error_message=str(exc),
                        output_snapshot=self._node_snapshot(state),
                    )
                except Exception:
                    pass
            raise

        node_failed = bool(
            output.get("error")
            and output.get("next_action") == "fail_closed"
            and node_name != "fail_closed"
        )
        if run_id and execution_sequence is not None:
            try:
                if node_failed:
                    self._handle_node_failure(output, node_name)
                    self._repository.fail_node_execution(
                        run_id,
                        sequence=execution_sequence,
                        error_code=str(output.get("error_code") or "CaseAgentNodeError"),
                        error_message=str(output.get("error_message") or ""),
                        output_snapshot=self._node_snapshot(output),
                    )
                else:
                    self._repository.finish_node_execution(
                        run_id,
                        sequence=execution_sequence,
                        output_snapshot=self._node_snapshot(output),
                    )
            except Exception:
                pass
        if run_id and not node_failed:
            try:
                self._repository.save_checkpoint(
                    run_id,
                    node_name=node_name,
                    state_snapshot=self._checkpoint_snapshot(output),
                    source_refs=self._state_source_refs(output),
                    safe_to_resume=node_name
                    in {
                        "load_session",
                        "fast_rule_entity_perception",
                        "build_perceptual_state",
                        "build_memory_retrieval_plan",
                        "start_request_memory_prefetch",
                        "business_semantic_planner",
                        "input_ingestion",
                        "early_entity_extractor",
                        "memory_prefetch",
                        "semantic_intent_perception",
                        "rule_precheck",
                        "intent_example_biencoder",
                        "light_semantic_context_assembly",
                        "llm_semantic_parser",
                        "slot_merger",
                        "context_need_resolver",
                        "perception_gssc",
                        "perception_contract_builder",
                        "state_normalizer",
                        "confidence_calibrator",
                        "readiness_signal_builder",
                        "load_perceptual_state",
                        "decision_readiness_checker",
                        "decision_precheck",
                        "rule_based_planner",
                        "read_context_slice",
                        "heuristic_planner",
                        "planner_llm_context_builder",
                        "llm_planner",
                        "plan_normalizer",
                        "validate_execution_plan",
                        "call_capabilities",
                        "execution_dag_builder",
                        "dag_executor",
                        "dispatch_step",
                        "execute_l1_step",
                        "execute_l2_step",
                        "materialize_expert_task",
                        "call_expert_analysis",
                        "build_answer_context",
                        "resolve_answer_policy",
                        "resolve_answer_style",
                        "generate_answer",
                        "validate_answer",
                    },
                )
            except Exception:
                pass
        if run_id and node_name != "persist_result":
            self._raise_if_run_cancelled(run_id)
        return output

    def _execute_run(self, run_id: str, actor_id: str) -> None:
        run = self._repository.get_run(run_id=run_id, actor_id=actor_id)
        if run is None:
            return
        if run.status in {"waiting_for_user", "completed", "failed", "cancelled", "timed_out", "degraded"}:
            return
        if run.cancel_requested_at is not None:
            self._mark_run_cancelled(run_id)
            return
        if run.parent_run_id is not None:
            self._repository.update_run(
                run_id,
                status="resuming",
                current_node="resuming",
            )
            run = self._repository.get_run(run_id=run_id, actor_id=actor_id) or run
        state = {
            "run_id": run_id,
            "actor_id": actor_id,
            "run": run,
            "resume_context": run.resume_context,
            "model_call_count": 0,
            "tool_call_count": 0,
            "capability_results": [],
            "available_sources": [],
            "artifact_store": {"artifacts": {}, "by_layer": {"L1": [], "L2": [], "L3": []}},
            "artifact_refs": [],
            "validation_retry_count": 0,
            "next_action": "load_session",
            "memory_hint_packs": {},
            "memory_fallbacks": [],
            "memory_retrieval_plan": {},
            "request_memory_status": {"enabled": self._case_memory is not None, "types": {}},
            "memory_consumption_status": {},
            "failure_recovery_hint": {},
            "intent_memory_shadow": {},
            "validation_repair_succeeded": False,
            "validation_recovery_mode": "",
            "perception_metrics": {},
            "model_call_metrics": [],
        }
        try:
            self._graph.invoke(state)
        except CaseAgentRunCancelled:
            self._mark_run_cancelled(run_id)
        except Exception as exc:
            state["error"] = exc
            state["error_code"] = exc.__class__.__name__
            state["error_message"] = str(exc)
            try:
                persist_result_node(self, fail_closed_node(self, state))
            except Exception:
                # Last-ditch fail-closed path: never leak stack traces.
                self._repository.update_run(
                    run_id,
                    status="failed",
                    current_node="failed",
                    error_code=exc.__class__.__name__,
                    error_message=str(exc)[:500],
                    model_call_count=state.get("model_call_count", 0),
                    tool_call_count=state.get("tool_call_count", 0),
                )
        finally:
            if self._case_memory is not None:
                self._case_memory.cache.clear(run_id)
            with self._request_memory_lock:
                for key in [key for key in self._request_memory_futures if key[0] == run_id]:
                    self._request_memory_futures.pop(key, None)
            clear_runtime_artifacts(run_id)

    def _raise_if_run_cancelled(self, run_id: str) -> None:
        checker = getattr(self._repository, "is_run_cancel_requested", None)
        if callable(checker) and checker(run_id):
            raise CaseAgentRunCancelled(run_id)

    def _mark_run_cancelled(self, run_id: str, *, reason: str = "user_requested") -> None:
        marker = getattr(self._repository, "mark_run_cancelled", None)
        if not callable(marker):
            return
        changed = marker(run_id)
        if changed:
            self._repository.append_event(
                run_id,
                "cancelled",
                "任务已停止，未生成或保存助手回答",
                {"reason": reason},
            )

    def _run_tool(
        self,
        run_id: str,
        case_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> CaseAgentToolResult:
        parsed_arguments = self._parse_tool_arguments_for_trace(arguments)
        tool_sequence: int | None = None
        self._repository.append_event(
            run_id,
            "tool_running",
            f"正在调用工具：{tool_name}",
            {"tool_name": tool_name},
        )
        start_tool_call = getattr(self._repository, "start_tool_call", None)
        finish_tool_call = getattr(self._repository, "finish_tool_call", None)
        if callable(start_tool_call) and callable(finish_tool_call):
            tool_sequence = start_tool_call(
                run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=parsed_arguments,
            )
        try:
            result = self._tools.execute(
                current_case_id=case_id,
                tool_name=tool_name,
                raw_arguments=arguments,
            )
        except Exception as exc:
            if tool_sequence is not None and callable(finish_tool_call):
                finish_tool_call(
                    run_id,
                    sequence=tool_sequence,
                    result_summary={},
                    source_refs=[],
                    status="failed",
                    error_code=exc.__class__.__name__,
                    latency_ms=0,
                )
            raise
        self._persist_tool_result(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=parsed_arguments,
            result=result,
            tool_sequence=tool_sequence,
        )
        event_type = "tool_unavailable" if result.status == "unavailable" else "tool_complete"
        if result.status == "failed":
            event_type = "tool_failed"
        self._repository.append_event(
            run_id,
            event_type,
            f"工具调用完成：{tool_name}",
            {
                "tool_name": tool_name,
                "status": result.status,
                "source_refs": result.source_refs[:8],
                "error_code": result.error_code,
            },
        )
        return result

    def _persist_tool_result(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: CaseAgentToolResult,
        tool_sequence: int | None,
    ) -> None:
        result_summary = self._summarize_tool_payload(result.payload)
        finish_tool_call = getattr(self._repository, "finish_tool_call", None)
        if tool_sequence is not None and callable(finish_tool_call):
            finish_tool_call(
                run_id,
                sequence=tool_sequence,
                result_summary=result_summary,
                source_refs=result.source_refs,
                status=result.status,
                error_code=result.error_code,
                latency_ms=result.latency_ms,
            )
            return
        self._repository.record_tool_call(
            run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary,
            source_refs=result.source_refs,
            status=result.status,
            error_code=result.error_code,
            latency_ms=result.latency_ms,
        )

    def _run_expert_capability(
        self,
        run_id: str,
        expert_task_type: str,
        state: dict[str, Any],
        *,
        tool_name: str = "ask_policy_expert",
        arguments: dict[str, Any] | None = None,
    ) -> ExpertAgentResult:
        run = state["run"]
        tool_arguments = arguments or {
            "case_id": run.case_id,
            "expert_task_type": expert_task_type,
            "slots": state.get("slots", {}),
        }
        task = self._materialize_expert_task(
            run_id=run_id,
            case_id=run.case_id,
            tool_name=tool_name,
            expert_task_type=expert_task_type,
            state=state,
            arguments=tool_arguments,
        )
        return self._call_expert_analysis_task(
            run_id=run_id,
            expert_task_type=expert_task_type,
            state=state,
            task=task,
            tool_name=tool_name,
        )

    def _call_expert_analysis_task(
        self,
        *,
        run_id: str,
        expert_task_type: str,
        state: dict[str, Any],
        task: ExpertAnalysisTask,
        tool_name: str = "ask_policy_expert",
    ) -> ExpertAgentResult:
        run = state["run"]
        tool_call_id = (
            f"system_capability:{tool_name}:{state.get('tool_call_count', 0) + 1}"
        )
        tool_sequence: int | None = None
        start_tool_call = getattr(self._repository, "start_tool_call", None)
        finish_tool_call = getattr(self._repository, "finish_tool_call", None)
        if callable(start_tool_call) and callable(finish_tool_call):
            tool_sequence = start_tool_call(
                run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=task.model_dump(mode="json"),
            )
        try:
            result = self._expert_agent.run(
                expert_task_type=expert_task_type,
                task=task,
                input_factors={
                    "case_id": run.case_id,
                    "parent_run_id": run_id,
                    "intent": state.get("intent"),
                    "slots": state.get("slots", {}),
                    "question": state["user_message"].content,
                    "semantic_frame": state.get("semantic_frame", {}),
                    "case_state_summary": (
                        (
                            state.get("decision_context", {})
                            .get("domain_artifact_context", {})
                            .get("case_state_summary", {})
                        )
                        if isinstance(state.get("decision_context"), dict)
                        else {}
                    ),
                    "fact_bundle": getattr(task, "fact_bundle", []),
                    "source_refs": getattr(task, "context_refs", []),
                    "artifact_refs": state.get("artifact_refs", []),
                },
                model_gateway=self._expert_gateway,
                policy_rag_client=self._policy_rag_client,
                case_context_gateway=CaseContextGateway(tools=self._tools),
                repository=self._repository,
            )
        except Exception as exc:
            if tool_sequence is not None and callable(finish_tool_call):
                finish_tool_call(
                    run_id,
                    sequence=tool_sequence,
                    result_summary={},
                    source_refs=[],
                    status="failed",
                    error_code=exc.__class__.__name__,
                    latency_ms=0,
                )
            raise
        result_status = (
            "success"
            if result.status in {"ok", "partial"}
            else "failed"
            if result.status == "failed"
            else "unavailable"
        )
        result_error_code = (
            None
            if result_status == "success"
            else "expert_capability_failed"
            if result.status == "failed"
            else "expert_evidence_insufficient"
            if result.status == "insufficient"
            else "expert_capability_unavailable"
        )
        if tool_sequence is not None and callable(finish_tool_call):
            finish_tool_call(
                run_id,
                sequence=tool_sequence,
                result_summary=result.payload,
                source_refs=result.source_refs,
                status=result_status,
                error_code=result_error_code,
                latency_ms=0,
            )
        else:
            self._repository.record_tool_call(
                run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=task.model_dump(mode="json"),
                result_summary=result.payload,
                source_refs=result.source_refs,
                status=result_status,
                error_code=result_error_code,
                latency_ms=0,
            )
        self._repository.append_event(
            run_id,
            "capability_unavailable"
            if result_status == "unavailable"
            else "capability_failed"
            if result_status == "failed"
            else "capability_complete",
            "专家能力证据不足或不可用" if result_status == "unavailable" else "专家能力调用完成",
            {
                "expert_task_type": expert_task_type,
                "status": result.status,
                "task_id": result.task_id,
                "source_refs": result.source_refs[:8],
            },
        )
        return result

    def _materialize_expert_task(
        self,
        *,
        run_id: str,
        case_id: str,
        tool_name: str,
        expert_task_type: str,
        state: dict[str, Any],
        arguments: dict[str, Any],
    ) -> ExpertAnalysisTask:
        user_message = state.get("user_message")
        question = str(
            arguments.get("question")
            or getattr(user_message, "content", "")
            or "请查询相关政策口径。"
        )
        semantics = state.get("query_semantics") if isinstance(state.get("query_semantics"), dict) else {}
        goal = str(arguments.get("goal") or semantics.get("user_goal") or question)
        fact_bundle = [
            item for item in arguments.get("fact_bundle", [])
            if isinstance(item, dict)
        ]
        context_refs = [
            str(item) for item in arguments.get("source_refs", [])
            if item
        ]
        return ExpertAnalysisTask(
            task_id=f"l3_policy_{uuid4().hex[:12]}",
            parent_run_id=run_id,
            case_id=case_id,
            capability=tool_name,
            expert_task_type=expert_task_type,
            goal=goal[:500],
            user_question=question[:2000],
            fact_bundle=fact_bundle[:16],
            context_refs=context_refs[:40],
            allowed_tools=[
                "policy.search_text",
                "policy.search_version",
                "case_context.query",
            ],
            constraints=ExpertAnalysisConstraints(
                read_only=True,
                no_final_audit_decision=True,
                must_cite_evidence=True,
                no_material_gap_diff=True,
            ),
            budget=ExpertAnalysisBudget(
                max_model_calls=4,
                max_tool_calls=min(max(1, self._max_tool_calls), 6),
                max_retrieval_rounds=3,
                max_llm_rewrite_calls=1,
                timeout_ms=self._policy_rag_timeout_ms,
            ),
            filters=dict(arguments.get("filters") or {}),
            memory_guidance=(
                dict(arguments.get("memory_guidance") or {})
                if isinstance(arguments.get("memory_guidance"), dict)
                else {}
            ),
        )

    def _expert_result_to_tool_result(
        self,
        *,
        case_id: str,
        capability: str,
        result: ExpertAgentResult,
    ) -> CaseAgentToolResult:
        payload = result.payload if isinstance(result.payload, dict) else {}
        all_policy_evidence = [
            item for item in payload.get("policy_evidence", [])
            if isinstance(item, dict)
        ][:20]
        used_source_refs = self._policy_payload_used_source_refs(payload)
        policy_evidence = (
            [
                item
                for item in all_policy_evidence
                if item.get("source_ref") in used_source_refs
            ][:20]
            if used_source_refs
            else all_policy_evidence[:5]
        )
        if policy_evidence:
            payload = {**payload, "policy_evidence": policy_evidence}
        sources = [
            self._policy_evidence_source(item)
            for item in policy_evidence
            if isinstance(item.get("source_ref"), str) and item.get("source_ref")
        ]
        source_refs = [source.source_ref for source in sources]
        has_citable_policy_evidence = bool(source_refs)
        status = (
            "success"
            if (
                result.status in {"ok", "partial"}
                or result.status == "insufficient"
            )
            and has_citable_policy_evidence
            else "failed"
            if result.status == "failed"
            else "unavailable"
        )
        error_code = (
            "expert_capability_failed"
            if status == "failed"
            else "expert_evidence_insufficient"
            if result.status == "insufficient"
            else "expert_capability_unavailable"
            if status == "unavailable"
            else None
        )
        return CaseAgentToolResult(
            payload={
                "status": payload.get("status") or result.status,
                "case_id": case_id,
                "capability": capability,
                "section_key": "policy_expert",
                "payload": payload,
                "expert_task_id": result.task_id,
                "safe_summary": result.safe_summary,
                "no_general_knowledge_fallback": True,
                "limits": payload.get("limits"),
            },
            source_refs=source_refs,
            sources=sources,
            status=status,
            error_code=error_code,
        )

    @staticmethod
    def _policy_evidence_source(item: dict[str, Any]) -> CaseAgentSource:
        row_fields = CaseAgentService._policy_structured_fields(item.get("excerpt"))
        fields = []
        for label, key in (("地区", "jurisdiction"),):
            value = item.get(key)
            if value not in (None, "", [], {}):
                fields.append(
                    CaseAgentSourceDetailField(label=label, value=str(value)[:500])
                )
        excerpt = CaseAgentService._policy_display_excerpt(item, row_fields)
        if excerpt:
            fields.append(
                CaseAgentSourceDetailField(label="证据片段", value=excerpt[:500])
            )
        row_source_url = str(row_fields.get("source_url") or "").strip()
        source_url = (
            row_source_url
            if row_source_url.startswith(("https://", "http://"))
            else item.get("source_url")
        )
        title = str(
            row_fields.get("source_title")
            or row_fields.get("policy_title")
            or row_fields.get("政策名称")
            or row_fields.get("文件名称")
            or item.get("title")
            or "政策证据"
        )[:200]
        if str(item.get("content_type") or "") == "table_row" and any(
            marker in title.lower() for marker in ("字段", "结构化表", "table row", "dataset")
        ):
            title = {
                "beijing": "北京市医保政策原文",
                "national": "国家医保政策原文",
            }.get(str(item.get("jurisdiction") or "").lower(), "医保政策原文")
        row_version = (
            row_fields.get("publish_date")
            or row_fields.get("effective_date")
            or row_fields.get("发布日期")
            or row_fields.get("生效日期")
        )
        version = (
            row_version
            if row_version
            else None
            if row_source_url
            else item.get("version")
        )
        return CaseAgentSource(
            source_ref=str(item["source_ref"]),
            source_type="policy_rag",
            title=title,
            version=str(version)[:80] if version else None,
            detail=CaseAgentSourceDetail(fields=fields[:12]),
            metadata={
                "evidence_ref": item.get("evidence_ref"),
                "source_url": source_url,
                "policy_domain": item.get("policy_domain"),
                "jurisdiction": item.get("jurisdiction"),
            },
        )

    @staticmethod
    def _policy_payload_used_source_refs(payload: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        for key in ("claims", "citations", "need_answers"):
            items = payload.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                source_refs = item.get("source_refs")
                if isinstance(source_refs, list):
                    refs.extend(str(ref) for ref in source_refs if isinstance(ref, str) and ref)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _policy_structured_fields(value: Any) -> dict[str, str]:
        fields: dict[str, str] = {}
        for match in re.finditer(
            r"(?m)^\s*([^:\n：]{1,40})[:：]\s*([^\n]+)",
            str(value or ""),
        ):
            key = match.group(1).strip()
            field_value = match.group(2).strip()
            if key and field_value:
                fields[key] = field_value
        return fields

    @staticmethod
    def _policy_display_excerpt(
        item: dict[str, Any],
        row_fields: dict[str, str],
    ) -> str:
        if str(item.get("content_type") or "") == "table_row":
            raw = row_fields.get("evidence_text") or row_fields.get("证据文本") or ""
        else:
            raw = str(item.get("excerpt") or "")
        lines = []
        metadata_prefixes = {
            "资料标题", "地区", "政策领域", "来源ID", "source_id", "source_url",
            "node_id", "field_key", "row_id", "case_relevance", "content_type",
            "dataset", "fetched_at", "官方来源", "page", "extraction_method",
        }
        for raw_line in str(raw or "").replace("\r", "\n").split("\n"):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            head = line.split(":", 1)[0].split("：", 1)[0].strip()
            if head in metadata_prefixes:
                continue
            lines.append(line)
        text = re.sub(r"^(?:答复|答|回复)[:：]\s*", "", " ".join(lines)).strip()
        if len(text) <= 500:
            return text
        clipped = text[:500]
        sentence_end = max(clipped.rfind("。"), clipped.rfind("；"), clipped.rfind("！"), clipped.rfind("？"))
        return clipped[: sentence_end + 1] if sentence_end >= 120 else clipped.rstrip() + "..."

    def _node_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._safe_state_snapshot(
            state,
            include_capability_payloads=False,
            include_final_response=False,
        )

    def _checkpoint_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._safe_state_snapshot(
            state,
            include_capability_payloads=True,
            include_final_response=True,
        )

    def _safe_state_snapshot(
        self,
        state: dict[str, Any],
        *,
        include_capability_payloads: bool,
        include_final_response: bool,
    ) -> dict[str, Any]:
        run = state.get("run")
        session = state.get("session")
        user_message = state.get("user_message")
        snapshot: dict[str, Any] = {
            "run_id": state.get("run_id"),
            "actor_id": state.get("actor_id"),
            "run": self._json_safe(run) if run is not None else None,
            "session": (
                {
                    "session_id": session.session_id,
                    "case_id": session.case_id,
                    "actor_id": session.actor_id,
                    "title": session.title,
                    "session_summary": session.session_summary,
                    "current_topic": session.current_topic,
                    "pending_tool": session.pending_tool,
                    "referenced_source_refs": session.referenced_source_refs,
                    "task_state": task_state_payload(
                        getattr(session, "task_state", None)
                    ),
                }
                if session is not None
                else None
            ),
            "user_message": (
                {
                    "message_id": user_message.message_id,
                    "active_stage": user_message.active_stage,
                    "content": self._clip(user_message.content, 1000),
                }
                if user_message is not None
                else None
            ),
            "intent": state.get("intent"),
            "intent_confidence": state.get("intent_confidence"),
            "slots": state.get("slots", {}),
            "missing_slots": state.get("missing_slots", []),
            "input_envelope": state.get("input_envelope", {}),
            "entity_frame": state.get("entity_frame", {}),
            "semantic_frame": state.get("semantic_frame", {}),
            "context_need": state.get("context_need", {}),
            "minimal_planning_context": state.get("minimal_planning_context", {}),
            "decision_context_ref": state.get("decision_context_ref", ""),
            "decision_context": state.get("decision_context", {}),
            "perceptual_state": state.get("perceptual_state", {}),
            "perception_metrics": state.get("perception_metrics", {}),
            "model_call_metrics": state.get("model_call_metrics", []),
            "decision_context_slice": state.get("decision_context_slice", {}),
            "planner_llm_context": state.get("planner_llm_context", {}),
            "query_semantics": state.get("query_semantics", {}),
            "execution_plan": state.get("execution_plan", []),
            "execution_dag": state.get("execution_dag", {}),
            "current_dag_step": state.get("current_dag_step", {}),
            "artifact_refs": state.get("artifact_refs", []),
            "artifact_store": self._artifact_store_snapshot(
                state.get("artifact_store", {}),
                include_payloads=include_capability_payloads,
            ),
            "section_status": state.get("section_status", []),
            "planning_error": state.get("planning_error", ""),
            "plan_optimization": state.get("plan_optimization", {}),
            "context_plan": state.get("context_plan", {}),
            "answer_policy": state.get("answer_policy", {}),
            "answer_context": state.get("answer_context", {}),
            "answer_style_policy": state.get("answer_style_policy", {}),
            "answer_strategy": state.get("answer_strategy", ""),
            "reuse_answer_ref": state.get("reuse_answer_ref", ""),
            "reuse_source_refs": state.get("reuse_source_refs", []),
            "answer_rewrite_mode": state.get("answer_rewrite_mode", ""),
            "task_state": state.get("task_state", {}),
            "turn_relation": state.get("turn_relation", {}),
            "shortcut": state.get("shortcut", {}),
            "resume_context": state.get("resume_context", {}),
            "session_memory_context": state.get("session_memory_context", {}),
            "memory_retrieval_plan": state.get("memory_retrieval_plan", {}),
            "request_memory_status": state.get("request_memory_status", {}),
            "memory_consumption_status": state.get("memory_consumption_status", {}),
            "memory_hint_packs": state.get("memory_hint_packs", {}),
            "memory_fallbacks": state.get("memory_fallbacks", []),
            "failure_recovery_hint": state.get("failure_recovery_hint", {}),
            "context_digest": state.get("context_digest", {}),
            "digest_source_refs": state.get("digest_source_refs", []),
            "available_sources": [
                source.model_dump(mode="json")
                for source in state.get("available_sources", [])
            ],
            "answer": (
                state["answer"].model_dump(mode="json")
                if isinstance(state.get("answer"), CaseAgentAnswer)
                else None
            ),
            "model_call_count": state.get("model_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "validation_retry_count": state.get("validation_retry_count", 0),
            "validation_error": state.get("validation_error", ""),
            "next_action": state.get("next_action"),
            "completion_status": state.get("completion_status"),
            "error_code": state.get("error_code"),
            "error_message": self._clip(state.get("error_message"), 500),
        }
        snapshot["capability_results"] = self._capability_results_snapshot(
            state.get("capability_results", []),
            include_payload_summary=include_capability_payloads,
        )
        if include_final_response and isinstance(state.get("final_response"), ModelResponse):
            final_response = state["final_response"]
            snapshot["final_response"] = {
                "content": final_response.content,
                "tool_call_count": len(final_response.tool_calls),
                "metrics": final_response.metrics,
            }
        return self._json_safe(snapshot)

    @staticmethod
    def record_model_call_metrics(
        state: dict[str, Any],
        node_name: str,
        response: ModelResponse,
    ) -> dict[str, Any]:
        """Persist only bounded operational metrics, never model reasoning or content."""

        allowed = {
            "model",
            "latency_ms",
            "message_count",
            "input_chars",
            "tool_schema_count",
            "tool_schema_chars",
            "require_json",
            "thinking_enabled",
            "reasoning_effort",
            "output_chars",
            "reasoning_chars",
            "thinking_contract_violation",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "max_tokens",
            "timeout_seconds",
        }
        entry = {
            "node": node_name,
            **{
                key: value
                for key, value in response.metrics.items()
                if key in allowed
            },
        }
        state.setdefault("model_call_metrics", []).append(entry)
        return entry

    def _artifact_store_snapshot(
        self,
        store: Any,
        *,
        include_payloads: bool,
    ) -> dict[str, Any]:
        if not isinstance(store, dict):
            return {"artifacts": {}, "by_layer": {}}
        artifacts = store.get("artifacts") if isinstance(store.get("artifacts"), dict) else {}
        safe_artifacts: dict[str, Any] = {}
        for ref, artifact in artifacts.items():
            if not isinstance(artifact, dict):
                continue
            item = {
                "artifact_ref": artifact.get("artifact_ref") or ref,
                "artifact_type": artifact.get("artifact_type"),
                "layer": artifact.get("layer"),
                "step_id": artifact.get("step_id"),
                "step": artifact.get("step"),
                "capability": artifact.get("capability"),
                "status": artifact.get("status"),
                "error_code": artifact.get("error_code"),
                "source_refs": artifact.get("source_refs", []),
                "latency_ms": artifact.get("latency_ms"),
            }
            if include_payloads:
                payload_summary = artifact.get("payload_summary")
                if payload_summary is None and isinstance(artifact.get("payload"), dict):
                    payload_summary = self._summarize_tool_payload(artifact.get("payload", {}))
                if payload_summary is not None:
                    item["payload_summary"] = payload_summary
                item["source_count"] = artifact.get("source_count") or len(artifact.get("sources", []) or [])
            safe_artifacts[str(ref)] = item
        return {
            "artifacts": safe_artifacts,
            "by_layer": store.get("by_layer", {}),
        }

    def _capability_results_snapshot(
        self,
        results: Any,
        *,
        include_payload_summary: bool,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not isinstance(results, list):
            return items
        for item in results:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            name, result = item
            entry = {
                "capability": name,
                "status": getattr(result, "status", ""),
                "source_refs": getattr(result, "source_refs", []),
                "error_code": getattr(result, "error_code", None),
                "latency_ms": getattr(result, "latency_ms", 0),
            }
            if include_payload_summary:
                payload = getattr(result, "payload", {})
                if isinstance(payload, dict):
                    entry["payload_summary"] = self._summarize_tool_payload(payload)
            items.append(entry)
        return items

    def _state_source_refs(self, state: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        for source in state.get("available_sources", []):
            if getattr(source, "source_ref", None):
                refs.append(source.source_ref)
        answer = state.get("answer")
        if isinstance(answer, CaseAgentAnswer):
            refs.extend(self._answer_source_refs(answer))
        refs.extend(state.get("digest_source_refs", []))
        refs.extend(
            ref for ref in state.get("artifact_refs", [])
            if isinstance(ref, str) and ref
        )
        artifact_store = state.get("artifact_store", {})
        artifacts = (
            artifact_store.get("artifacts", {})
            if isinstance(artifact_store, dict)
            else {}
        )
        if isinstance(artifacts, dict):
            for artifact in artifacts.values():
                if not isinstance(artifact, dict):
                    continue
                refs.extend(
                    ref for ref in artifact.get("source_refs", [])
                    if isinstance(ref, str) and ref
                )
        return list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref))

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, ModelResponse):
            return {
                "content": value.content,
                "tool_calls": [asdict(item) for item in value.tool_calls],
                "metrics": value.metrics,
            }
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if is_dataclass(value):
            return self._json_safe(asdict(value))
        if isinstance(value, dict):
            return {
                str(key): self._json_safe(item)
                for key, item in value.items()
                if key != "reasoning_content"
            }
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, str):
            return self._clip(value, 8000)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return str(value)

    def _apply_fast_context_digest(self, state: dict[str, Any]) -> None:
        """Build a compact model context while keeping full source details."""

        if state.get("context_plan", {}).get("display_mode") != "grounded":
            return
        digest, requested_refs = self._build_context_digest(state)
        if not digest:
            return

        sources = state.get("available_sources", [])
        source_by_ref = {source.source_ref: source for source in sources}
        filtered_refs = [
            ref for ref in dict.fromkeys(requested_refs) if ref in source_by_ref
        ]
        if filtered_refs:
            state["available_sources"] = [source_by_ref[ref] for ref in filtered_refs]
            state["digest_source_refs"] = filtered_refs
        else:
            state["digest_source_refs"] = [source.source_ref for source in sources]
        state["context_digest"] = digest

    def _build_context_digest(self, state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        payload_by_name: dict[str, dict[str, Any]] = {}
        refs_by_name: dict[str, list[str]] = {}
        result_items = artifact_results(state) or state.get("capability_results", [])
        for name, result in result_items:
            payload = getattr(result, "payload", {})
            if isinstance(payload, dict):
                payload_by_name[name] = payload
                refs_by_name[name] = [
                    str(ref)
                    for ref in getattr(result, "source_refs", [])
                    if isinstance(ref, str) and ref
                ]

        if not payload_by_name:
            return {}, []

        refs: list[str] = []
        digest: dict[str, Any] = {}
        user_goal = ""
        query_semantics = state.get("query_semantics", {})
        if isinstance(query_semantics, dict):
            user_goal = str(query_semantics.get("user_goal") or "")
        user_message = state.get("user_message")
        if not user_goal and user_message is not None:
            user_goal = str(getattr(user_message, "content", "") or "")

        for capability, capability_payload in payload_by_name.items():
            section_key = capability_payload.get("section_key") or capability
            section_payload = capability_payload.get("payload")
            if not isinstance(section_payload, dict):
                section_payload = capability_payload
            refs.extend(refs_by_name.get(capability, []))
            refs.extend(
                str(ref)
                for ref in self._payload_source_refs(section_payload)
                if isinstance(ref, str)
            )
            digest[f"{section_key}_digest"] = {
                "capability": capability_payload.get("capability") or capability,
                "section_key": section_key,
                "status": capability_payload.get("status"),
                "completeness": capability_payload.get("completeness"),
                "user_goal": user_goal,
                "summary": self._compact_caser_section(section_key, section_payload, user_goal=user_goal),
            }

        return digest, refs

    @classmethod
    def _compact_caser_section(
        cls,
        section_key: str,
        payload: dict[str, Any],
        *,
        user_goal: str = "",
    ) -> dict[str, Any] | str:
        if section_key == "case_basic_info":
            return {
                "case_number": payload.get("case_number"),
                "case_type": payload.get("case_type"),
                "review_status": payload.get("review_status"),
                "claimant_ref": payload.get("claimant_ref"),
                "access_method": payload.get("access_method"),
            }
        if section_key == "claimant_profile":
            return {
                "claimant_code": payload.get("claimant_code"),
                "gender": payload.get("gender"),
                "age_group": payload.get("age_group"),
                "insurance_type": payload.get("insurance_type"),
                "chronic_condition_tags": _preserve_short_list(payload.get("chronic_condition_tags"), 8),
                "allergy_history": _normalize_scalar_or_list(payload.get("allergy_history")),
                "patient_group_tags": _preserve_short_list(payload.get("patient_group_tags"), 8),
            }
        if section_key == "material_overview":
            return {
                "material_total": payload.get("material_total"),
                "groups": [
                    {
                        "category_id": item.get("category_id"),
                        "title": item.get("title"),
                        "count": item.get("count"),
                        "materials": [
                            {
                                "material_id": material.get("material_id"),
                                "name": material.get("name"),
                                "occurred_at": material.get("occurred_at"),
                                "source": material.get("source"),
                                "shape": material.get("shape"),
                                "status": material.get("status"),
                                "operation": material.get("operation"),
                                "source_ref": material.get("source_ref"),
                            }
                            for material in item.get("materials", [])[:8]
                            if isinstance(material, dict)
                        ],
                    }
                    for item in payload.get("groups", [])[:6]
                    if isinstance(item, dict)
                ],
            }
        if section_key == "medical_materials":
            return _compact_medical_materials(payload)
        if section_key == "prescription_materials":
            return _compact_prescription_materials(payload)
        if section_key == "settlement_materials":
            return _compact_settlement_materials(payload)
        if section_key == "statistics_report":
            return _compact_statistics_report(payload, user_goal=user_goal)
        if section_key == "risk_score":
            return {
                "overall_strength": payload.get("overall_strength"),
                "risk_level": payload.get("risk_level"),
                "model_warning": _compact_risk_component(payload.get("model_warning")),
                "rule_check": _compact_risk_component(payload.get("rule_check")),
                "peer_deviation": _compact_risk_component(payload.get("peer_deviation")),
                "data_flow": _compact_risk_component(payload.get("data_flow")),
                "components": [
                    _compact_risk_component(item)
                    for item in payload.get("components", [])[:6]
                    if isinstance(item, dict)
                ],
            }
        if section_key == "evidence_package":
            return {
                "base_summary": payload.get("base_summary"),
                "discovered_clues": [
                    {
                        "name": item.get("name"),
                        "status": item.get("status"),
                        "source": item.get("source"),
                    }
                    for item in payload.get("discovered_clues", [])[:12]
                    if isinstance(item, dict)
                ],
                "source_basis": payload.get("source_basis"),
            }
        if section_key == "rule_verification":
            return {
                "summary": payload.get("summary"),
                "rules": [
                    _compact_rule(item)
                    for item in payload.get("rules", [])[:8]
                    if isinstance(item, dict)
                ],
            }
        return cls._compact_mapping(payload, 2400)

    def _build_generation_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        session = state["session"]
        recent = state.get("recent_messages", [])
        user_message = state["user_message"]
        fast_grounded = bool(
            self._fast_mode_enabled
            and (state.get("answer_context") or state.get("context_digest"))
            and (
                state.get("answer_policy", {}).get("display_mode")
                or state.get("context_plan", {}).get("display_mode")
            ) == "grounded"
        )
        if fast_grounded:
            context_payload = {
                "intent": state.get("intent"),
                "query_semantics": state.get("query_semantics", {}),
                "answer_policy": {
                    key: value
                    for key, value in (state.get("answer_policy") or {}).items()
                    if key
                    in {
                        "display_mode",
                        "requires_citation",
                        "capability_policy",
                        "capabilities_used",
                    }
                },
                "answer_style_policy": state.get("answer_style_policy", {}),
                "answer_context": self._bounded_prompt_value(
                    state.get("answer_context", {}),
                    8000,
                ),
                "context_digest": self._bounded_prompt_value(
                    state.get("context_digest", {}),
                    8000,
                ),
                "allowed_source_refs": list(state.get("digest_source_refs") or [])[:20],
                "available_source_index": [
                    {
                        "source_ref": source.source_ref,
                        "source_type": source.source_type,
                        "title": source.title,
                        "version": source.version,
                    }
                    for source in state.get("available_sources", [])[:20]
                ],
                "validation_error": self._clip(state.get("validation_error", ""), 300),
                "failure_recovery_hint": self._bounded_prompt_value(
                    state.get("failure_recovery_hint", {}),
                    1800,
                ),
            }
        else:
            context_payload = {
                "intent": state.get("intent"),
                "slots": state.get("slots", {}),
                "query_semantics": state.get("query_semantics", {}),
                "execution_plan": state.get("execution_plan", []),
                "section_status": state.get("section_status", []),
                "planning_error": state.get("planning_error", ""),
                "context_plan": state.get("context_plan", {}),
                "answer_policy": state.get("answer_policy", {}),
                "answer_context": self._bounded_prompt_value(
                    state.get("answer_context", {}),
                    8000,
                ),
                "answer_style_policy": state.get("answer_style_policy", {}),
                "answer_strategy": state.get("answer_strategy", ""),
                "working_memory": {
                    "session_summary": session.session_summary,
                    "current_topic": session.current_topic,
                    "referenced_source_refs": session.referenced_source_refs,
                    "structured_task_state": task_state_payload(
                        getattr(session, "task_state", None)
                    ),
                },
                "available_sources": [
                    source.model_dump(mode="json")
                    for source in state.get("available_sources", [])[:20]
                ],
                "validation_error": self._clip(state.get("validation_error", ""), 300),
                "failure_recovery_hint": self._bounded_prompt_value(
                    state.get("failure_recovery_hint", {}),
                    1800,
                ),
            }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_BOUNDARY},
            {
                "role": "system",
                "content": "本轮受控上下文：" + json.dumps(context_payload, ensure_ascii=False, default=str),
            },
        ]
        recent_limit = 2 if task_state_payload(
            getattr(session, "task_state", None)
        ).get("task_id") else 6
        for message in recent[-recent_limit:]:
            if message.message_id == user_message.message_id:
                continue
            messages.append({"role": message.role, "content": message.content[:1200]})
        messages.append({"role": "user", "content": user_message.content})
        return messages

    @staticmethod
    def _bounded_prompt_value(value: Any, max_chars: int) -> Any:
        """Keep one prompt section within a deterministic serialized budget."""

        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return value
        return {
            "summary": serialized[:max_chars],
            "truncated": True,
            "original_chars": len(serialized),
        }

    def _parse_answer(
        self,
        response: ModelResponse,
        state: dict[str, Any],
    ) -> CaseAgentAnswer:
        try:
            raw_json = self._extract_answer_json(response.content or "{}")
            payload = json.loads(raw_json)
            payload, normalization_reasons = self._normalize_answer_payload(payload, state)
            answer = CaseAgentAnswer.model_validate(payload)
            validate_answer(answer, state.get("available_sources", []))
            if normalization_reasons:
                try:
                    self._repository.append_event(
                        state["run"].run_id,
                        "answer_normalized",
                        "模型回答结构已进行轻量修正",
                        {"repairs": normalization_reasons[:8]},
                    )
                except Exception:
                    pass
            return answer
        except json.JSONDecodeError as exc:
            raise CaseAgentSafetyError("invalid_json", "模型输出不是有效 JSON") from exc
        except ValidationError as exc:
            raise CaseAgentSafetyError("invalid_schema", "模型输出不符合回答协议") from exc

    @staticmethod
    def _extract_answer_json(content: str) -> str:
        """Extract a JSON object from a provider response."""

        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            candidate = text[match.start():]
            try:
                _, end = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            return candidate[:end].strip()
        return text

    @staticmethod
    def _normalize_answer_payload(
        payload: Any,
        state: dict[str, Any],
    ) -> tuple[Any, list[str]]:
        """Repair deterministic schema drift without inventing case facts."""

        if not isinstance(payload, dict):
            return payload, []

        normalized = dict(payload)
        reasons: list[str] = []
        if (
            "content_blocks" not in normalized
            and isinstance(normalized.get("text"), str)
            and normalized.get("text", "").strip()
        ):
            normalized = {
                "content_blocks": [
                    {
                        "text": normalized.get("text"),
                        "source_refs": normalized.get("source_refs")
                        if isinstance(normalized.get("source_refs"), list)
                        else [],
                    }
                ]
            }
            reasons.append("wrapped_top_level_content_block")
        allowed_top_level = {
            "display_mode",
            "content_blocks",
            "sources",
            "answer_markdown",
            "claims",
            "citations",
            "fallback_notice",
            "metadata",
        }
        extra_fields = sorted(key for key in normalized if key not in allowed_top_level)
        if extra_fields:
            for key in extra_fields:
                normalized.pop(key, None)
            reasons.append("removed_extra_top_level_fields")

        for key, default in (
            ("content_blocks", []),
            ("sources", []),
            ("answer_markdown", None),
            ("claims", []),
            ("citations", []),
            ("fallback_notice", ""),
            ("metadata", {}),
        ):
            if key not in normalized:
                normalized[key] = default
                reasons.append(f"filled_missing_{key}")

        expected_mode = str((state.get("answer_policy") or {}).get("display_mode") or "plain")
        if normalized.get("display_mode") not in {"plain", "grounded", "unavailable", "error"}:
            normalized["display_mode"] = expected_mode
            reasons.append("filled_display_mode_from_policy")
        if not normalized.get("metadata"):
            policy = state.get("answer_policy") or {}
            semantics = state.get("query_semantics") or {}
            normalized["metadata"] = {
                "intent": state.get("intent") or semantics.get("intent") or "general_help",
                "requires_citation": bool(policy.get("requires_citation")),
                "capabilities_used": list(policy.get("capabilities_used") or []),
            }
            reasons.append("filled_metadata_from_policy")

        if not isinstance(normalized.get("metadata"), dict):
            normalized["metadata"] = {}
            reasons.append("coerced_metadata")
        if not isinstance(normalized.get("fallback_notice"), str):
            normalized["fallback_notice"] = ""
            reasons.append("coerced_fallback_notice")
        if normalized.get("answer_markdown") is not None and not isinstance(
            normalized.get("answer_markdown"), str
        ):
            normalized["answer_markdown"] = None
            reasons.append("coerced_answer_markdown")
        elif isinstance(normalized.get("answer_markdown"), str):
            markdown = normalized["answer_markdown"].strip()
            clipped = markdown[:4000]
            if clipped != normalized["answer_markdown"]:
                reasons.append("normalized_answer_markdown")
            normalized["answer_markdown"] = clipped or None
        if not isinstance(normalized.get("claims"), list):
            normalized["claims"] = []
            reasons.append("coerced_claims")
        if not isinstance(normalized.get("citations"), list):
            normalized["citations"] = []
            reasons.append("coerced_citations")

        blocks, block_reasons = CaseAgentService._normalize_content_blocks(
            normalized.get("content_blocks")
        )
        if block_reasons:
            normalized["content_blocks"] = blocks
            reasons.extend(block_reasons)

        display_mode = normalized.get("display_mode", "plain")
        if display_mode in {"plain", "unavailable", "error"}:
            if normalized.get("sources"):
                normalized["sources"] = []
                reasons.append("cleared_sources_for_non_grounded")
            for block in normalized.get("content_blocks", []):
                if isinstance(block, dict) and block.get("source_refs"):
                    block["source_refs"] = []
                    reasons.append("cleared_source_refs_for_non_grounded")
            if normalized.get("claims"):
                normalized["claims"] = []
                reasons.append("cleared_claims_for_non_grounded")
            if normalized.get("citations"):
                normalized["citations"] = []
                reasons.append("cleared_citations_for_non_grounded")
        elif display_mode == "grounded":
            sources, source_reasons = CaseAgentService._normalize_grounded_sources(
                normalized.get("sources"),
                state.get("available_sources", []),
            )
            if source_reasons:
                normalized["sources"] = sources
                reasons.extend(source_reasons)
            if not normalized.get("sources"):
                referenced = {
                    ref
                    for block in normalized.get("content_blocks", [])
                    if isinstance(block, dict)
                    for ref in block.get("source_refs", [])
                    if isinstance(ref, str) and ref
                }
                materialized = [
                    source.model_dump(mode="json")
                    for source in state.get("available_sources", [])
                    if isinstance(source, CaseAgentSource) and source.source_ref in referenced
                ]
                if materialized:
                    normalized["sources"] = materialized
                    reasons.append("materialized_sources_from_block_refs")

        return normalized, list(dict.fromkeys(reasons))

    @staticmethod
    def _normalize_content_blocks(value: Any) -> tuple[Any, list[str]]:
        reasons: list[str] = []
        if isinstance(value, str):
            return [{"text": value[:1200], "source_refs": []}], ["coerced_content_blocks_string"]
        if not isinstance(value, list):
            return value, reasons

        normalized_blocks: list[Any] = []
        changed = False
        for block in value:
            if isinstance(block, str):
                text = block.strip()
                if text:
                    normalized_blocks.append({"text": text[:1200], "source_refs": []})
                    changed = True
                continue
            if not isinstance(block, dict):
                changed = True
                continue

            next_block = dict(block)
            if not isinstance(next_block.get("text"), str):
                normalized_blocks.append(next_block)
                continue
            text = next_block["text"].strip()
            if not text:
                changed = True
                continue
            if len(text) > 1200:
                next_block["text"] = text[:1200]
                changed = True
            elif text != next_block["text"]:
                next_block["text"] = text
                changed = True
            refs = next_block.get("source_refs", [])
            if not isinstance(refs, list):
                next_block["source_refs"] = []
                changed = True
            else:
                clean_refs = [ref for ref in refs if isinstance(ref, str) and ref][:8]
                if clean_refs != refs:
                    next_block["source_refs"] = clean_refs
                    changed = True
            normalized_blocks.append(next_block)

        if changed:
            reasons.append("normalized_content_blocks")
            return normalized_blocks, reasons
        return value, reasons

    @staticmethod
    def _normalize_grounded_sources(
        value: Any,
        available_sources: list[CaseAgentSource],
    ) -> tuple[Any, list[str]]:
        if not isinstance(value, list):
            return value, []

        available = {
            source.source_ref: source.model_dump(mode="json")
            for source in available_sources
        }
        normalized_sources: list[Any] = []
        changed = False
        required_keys = {"source_ref", "source_type", "title", "version", "detail", "metadata"}

        for source in value:
            if not isinstance(source, dict):
                normalized_sources.append(source)
                continue
            ref = source.get("source_ref")
            if isinstance(ref, str) and ref in available and (
                set(source.keys()) <= {"source_ref"} or not required_keys.issubset(source.keys())
            ):
                normalized_sources.append(available[ref])
                changed = True
                continue
            normalized_sources.append(
                CaseAgentService._normalize_source_detail_fields(source)
            )

        if changed or normalized_sources != value:
            return normalized_sources, ["normalized_grounded_sources"]
        return value, []

    @staticmethod
    def _normalize_source_detail_fields(source: dict[str, Any]) -> dict[str, Any]:
        next_source = dict(source)
        detail = next_source.get("detail")
        if not isinstance(detail, dict):
            return next_source
        fields = detail.get("fields")
        if not isinstance(fields, list):
            return next_source
        clean_fields: list[dict[str, str]] = []
        for field in fields[:12]:
            if not isinstance(field, dict):
                continue
            label = field.get("label")
            value = field.get("value")
            if label in (None, "") or value in (None, ""):
                continue
            clean_fields.append({"label": str(label)[:80], "value": str(value)[:500]})
        next_source["detail"] = {**detail, "fields": clean_fields}
        return next_source

    @staticmethod
    def _collect_available_sources(results: list[tuple[str, Any]]) -> list:
        sources = []
        seen: set[str] = set()
        for _name, result in results:
            for source in getattr(result, "sources", []):
                if source.source_ref not in seen:
                    seen.add(source.source_ref)
                    sources.append(source)
        return sources

    @staticmethod
    def _answer_source_refs(answer: CaseAgentAnswer) -> list[str]:
        refs: list[str] = []
        for block in answer.content_blocks:
            refs.extend(block.source_refs)
        for claim in answer.claims:
            refs.extend(claim.source_refs)
        for citation in answer.citations:
            refs.extend(citation.source_refs)
        for source in answer.sources:
            refs.append(source.source_ref)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _clip(value: Any, limit: int) -> str:
        if value in (None, ""):
            return ""
        text = str(value).strip()
        return text if len(text) <= limit else text[:limit] + "..."

    @classmethod
    def _compact_list(cls, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value[:limit]:
            if item in (None, ""):
                continue
            items.append(cls._clip(item, 180))
        return items

    @classmethod
    def _compact_mapping(cls, value: Any, limit: int) -> dict[str, Any] | str:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            return cls._clip(value, limit)

        preferred = [
            "summary",
            "conclusion",
            "overall_assessment",
            "evidence_strength",
            "relationship",
            "case_review_summary",
            "evidence_review_summary",
            "key_findings",
            "supported_points",
            "gaps",
            "recommendations",
            "next_steps",
        ]
        compact: dict[str, Any] = {}
        for key in preferred:
            if key not in value:
                continue
            item = value[key]
            if isinstance(item, list):
                compact[key] = cls._compact_list(item, 6)
            elif isinstance(item, dict):
                compact[key] = cls._clip(
                    json.dumps(item, ensure_ascii=False, default=str),
                    500,
                )
            else:
                compact[key] = cls._clip(item, 500)

        if compact:
            return compact
        return cls._clip(json.dumps(value, ensure_ascii=False, default=str), limit)

    @staticmethod
    def _risk_contributors(value: Any) -> list[str]:
        if not isinstance(value, dict):
            return []
        candidates: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                score = item.get("score") or item.get("points") or item.get("value")
                label = item.get("label") or item.get("name") or key
                if score not in (None, ""):
                    candidates.append(f"{label}: {score}")
                elif item.get("reason"):
                    candidates.append(f"{label}: {item.get('reason')}")
            elif item not in (None, ""):
                candidates.append(f"{key}: {item}")
        return candidates[:5]

    @classmethod
    def _payload_source_refs(cls, payload: dict[str, Any]) -> list[str]:
        refs: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                source_ref = value.get("source_ref")
                if isinstance(source_ref, str) and source_ref:
                    refs.append(source_ref)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _mentions_work_notes(text: str) -> bool:
        return any(term in text for term in ("工作笔记", "笔记", "人工记录", "审核记录"))

    @staticmethod
    def _fallback_answer(message: str, answer_type: str) -> CaseAgentAnswer:
        notice = CaseAgentService._safe_fallback_uncertainty(message)
        return CaseAgentAnswer(
            display_mode="error" if answer_type == "error" else "unavailable",
            content_blocks=[
                CaseAgentContentBlock(
                    text="当前未形成可采信的案件助手回答。"
                )
            ],
            sources=[],
            fallback_notice=notice,
            metadata={
                "intent": "unknown",
                "requires_citation": False,
                "capabilities_used": [],
                "error_type": answer_type,
            },
        )

    @staticmethod
    def _safe_fallback_uncertainty(message: str) -> str:
        lowered = message.lower()
        if "invalid authentication" in lowered or "incorrect api key" in lowered or "401" in lowered:
            return (
                "Case Agent 模型认证失败：请检查 .env.local 中的 Case Agent 模型 API Key "
                "是否有效，并确认 MEDIGUARD_CASE_AGENT_BASE_URL 与 Key 所属平台一致。"
            )
        if "rate limit" in lowered or "429" in lowered:
            return "Case Agent 模型调用触发限流或额度限制，请稍后重试或检查平台额度。"
        if "connection error" in lowered or "ssl" in lowered or "eof" in lowered:
            return "Case Agent 模型连接失败，请检查本机网络、代理或证书配置。"
        if (
            "prohibited_conclusion" in lowered
            or "安全边界" in message
            or "裁决性" in message
            or "越权结论" in message
        ):
            return "当前回答未通过安全边界校验，请换一种问法或查看当前案件已有资料。"
        if (
            "invalid_json" in lowered
            or "invalid_schema" in lowered
            or "citation_invalid" in lowered
            or "citation_missing_for_grounded" in lowered
            or "unknown capability" in lowered
            or "execution plan" in lowered
            or "field names must be simple" in lowered
            or "filter names must be simple" in lowered
            or "fields must be a list" in lowered
            or "filters must be an object" in lowered
            or "模型输出" in message
            or "回答引用" in message
        ):
            return "当前回答未通过结构或引用校验，请稍后重试或查看当前案件已有资料。"
        return "Case Agent 本次处理未能完成，请稍后重试；当前案件资料与人工审核流程不受影响。"

    @staticmethod
    def _summarize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        if len(raw) <= 2400:
            return payload
        return {
            "status": payload.get("status"),
            "summary": raw[:2400],
            "truncated": True,
        }

    @staticmethod
    def _parse_tool_arguments_for_trace(arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"_raw": arguments[:500], "_parse_error": "invalid_json"}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return dict(arguments)

    @staticmethod
    def _next_summary(previous: str, question: str, answer: str) -> str:
        return next_summary(previous, question, answer)


def _normalize_scalar_or_list(value: Any) -> Any:
    if isinstance(value, list):
        return _preserve_short_list(value, 8)
    if value in (None, ""):
        return "unknown"
    return value


def _preserve_short_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value[:limit] if item not in (None, "")]


def _compact_medical_materials(payload: dict[str, Any]) -> dict[str, Any]:
    materials = [
        item for item in payload.get("materials", [])[:8]
        if isinstance(item, dict)
    ]
    return {
        "material_count": len(payload.get("materials", []) or []),
        "materials": [
            {
                "material_id": item.get("material_id"),
                "name": item.get("name"),
                "basic_info": item.get("basic_info"),
                "content": item.get("content"),
                "visit_details": item.get("visit_details", [])[:30],
                "check_points": item.get("check_points", [])[:12],
            }
            for item in materials
        ],
    }


def _compact_prescription_materials(payload: dict[str, Any]) -> dict[str, Any]:
    materials = [
        item for item in payload.get("materials", [])[:8]
        if isinstance(item, dict)
    ]
    return {
        "material_count": len(payload.get("materials", []) or []),
        "materials": [
            {
                "material_id": item.get("material_id"),
                "name": item.get("name"),
                "basic_info": item.get("basic_info"),
                "content": item.get("content"),
                "prescription_drug_details": item.get("prescription_drug_details", [])[:100],
                "image_assets": [
                    {
                        "title": asset.get("title"),
                        "asset_type": asset.get("asset_type"),
                        "file_type": asset.get("file_type"),
                    }
                    for asset in item.get("image_assets", [])[:8]
                    if isinstance(asset, dict)
                ],
                "check_points": item.get("check_points", [])[:12],
            }
            for item in materials
        ],
    }


def _compact_settlement_materials(payload: dict[str, Any]) -> dict[str, Any]:
    materials = [
        item for item in payload.get("materials", [])[:8]
        if isinstance(item, dict)
    ]
    return {
        "material_count": len(payload.get("materials", []) or []),
        "settlement_summary": payload.get("settlement_summary"),
        "non_drug_fee_items": payload.get("non_drug_fee_items", [])[:80],
        "materials": [
            {
                "material_id": item.get("material_id"),
                "name": item.get("name"),
                "basic_info": item.get("basic_info"),
                "content": item.get("content"),
                "tables": item.get("tables", [])[:5],
                "check_points": item.get("check_points", [])[:12],
            }
            for item in materials
        ],
    }


def _compact_statistics_report(payload: dict[str, Any], *, user_goal: str) -> dict[str, Any]:
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        return {
            "metric_count": payload.get("metric_count"),
            "metric_baseline_status": payload.get("metric_baseline_status"),
            "groups": {},
        }
    selected: dict[str, list[dict[str, Any]]] = {}
    terms = _statistics_focus_terms(user_goal)
    for group_name, metrics in groups.items():
        if not isinstance(metrics, list):
            continue
        group_items = [
            metric for metric in metrics
            if isinstance(metric, dict) and _metric_matches_goal(metric, terms)
        ]
        if not group_items and _group_matches_goal(group_name, terms):
            group_items = [metric for metric in metrics if isinstance(metric, dict)][:20]
        if group_items:
            selected[group_name] = group_items[:40]

    if not selected:
        for group_name, metrics in groups.items():
            if isinstance(metrics, list):
                selected[group_name] = [
                    metric for metric in metrics[:12]
                    if isinstance(metric, dict)
                ]
    return {
        "metric_count": payload.get("metric_count"),
        "metric_baseline_status": payload.get("metric_baseline_status"),
        "groups": selected,
    }


def _statistics_focus_terms(user_goal: str) -> list[str]:
    text = str(user_goal or "")
    terms: list[str] = []
    if any(token in text for token in ("费用", "金额", "结算", "申报", "审批")):
        terms.extend(["费用", "金额", "SUM", "费", "ALL_SUM", "审批"])
    if any(token in text for token in ("药", "处方", "购药")):
        terms.extend(["药", "drug"])
    if any(token in text for token in ("就诊", "诊疗", "医院", "机构")):
        terms.extend(["就诊", "医院", "机构", "visit"])
    if any(token in text for token in ("支付", "统筹", "个人账户")):
        terms.extend(["支付", "统筹", "个人账户"])
    if any(token in text for token in ("占比", "比例", "结构")):
        terms.extend(["占比", "比例", "ratio"])
    if any(token in text for token in ("偏高", "偏离", "高于", "明显")):
        terms.extend(["偏高", "偏离", "高于", "明显"])
    return list(dict.fromkeys(terms))


def _group_matches_goal(group_name: str, terms: list[str]) -> bool:
    if not terms:
        return False
    group_markers = {
        "fee_statistics": ["费用", "金额", "结算", "申报", "审批"],
        "drug_statistics": ["药", "处方", "购药"],
        "visit_statistics": ["就诊", "医院", "机构"],
        "visit_day_statistics": ["就诊", "天数"],
        "payment_statistics": ["支付", "统筹", "个人账户"],
        "fee_structure_statistics": ["占比", "比例", "结构"],
    }
    return any(term in group_markers.get(group_name, []) for term in terms)


def _metric_matches_goal(metric: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return False
    text = json.dumps(
        {
            "feature_name": metric.get("feature_name"),
            "hint": metric.get("hint"),
        },
        ensure_ascii=False,
        default=str,
    )
    return any(term in text for term in terms)


def _compact_risk_component(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "key": value.get("key"),
        "label": value.get("label"),
        "score": value.get("score"),
        "max_score": value.get("max_score"),
        "summary": value.get("summary"),
        "details": value.get("details", [])[:8] if isinstance(value.get("details"), list) else [],
        "source_detail": value.get("source_detail"),
        "signal": value.get("signal") if isinstance(value.get("signal"), dict) else None,
    }


def _compact_rule(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": item.get("rule_id"),
        "rule_name": item.get("rule_name"),
        "rule_status": item.get("rule_status"),
        "attention_level": item.get("attention_level"),
        "verification_description": item.get("verification_description"),
        "verification_result": item.get("verification_result"),
        "basis": item.get("basis", [])[:4] if isinstance(item.get("basis"), list) else [],
        "check_items": item.get("check_items", [])[:12] if isinstance(item.get("check_items"), list) else [],
    }
