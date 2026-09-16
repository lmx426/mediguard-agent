"""Controlled Expert Analysis sub-agent orchestrator."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from src.backend.application.agent.expert_agent.graph.builder import (
    build_expert_agent_graph,
)
from src.backend.application.agent.expert_agent.graph.state import ExpertTaskType
from src.backend.application.agent.expert_agent.schemas import (
    AnswerabilityCheck,
    CaseFactUsed,
    ExpertAnalysisBudget,
    ExpertAnalysisConstraints,
    ExpertAnalysisResult,
    ExpertAnalysisTask,
    ExpertAuditSuggestion,
    PolicyEvidence,
    PolicySearchRequest,
)
from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
    EvidenceMatch,
    ExtractedFact,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    PolicyFilterOutputError,
    PolicyFilterResolver,
    PolicyFilterResolverInput,
    merge_policy_filters,
    normalize_policy_filters,
)
from src.backend.application.agent.expert_agent.service.policy_filter_catalog import (
    policy_filter_prompt_contract,
)
from src.backend.application.agent.expert_agent.service.policy_filter_contract import (
    policy_retrieval_plan_tool_schema,
)
from src.backend.application.agent.expert_agent.service.policy_information_needs import (
    build_information_need_requirements,
    information_need_prompt_contract,
)
from src.backend.application.agent.expert_agent.service.recall_filter_projector import (
    RecallFilterProjector,
)
from src.backend.application.agent.expert_agent.service.policy_answer_slots import (
    SLOT_REGISTRY,
    normalize_answer_slot_ids,
    resolve_answer_requirements,
    slot_answer_action,
    slot_coverage_terms,
    slot_label,
    slot_policy_domains,
    slot_required_fields,
    slot_scenario_id,
)
from src.backend.application.agent.expert_agent.service.policy_fact_extractors import (
    extract_facts_for_requirements,
    normalize_policy_evidence,
)
from src.backend.application.agent.expert_agent.service.policy_need_window import (
    build_sentence_windows,
    extract_verified_facts_from_judgement,
    reindex_facts_and_matches,
    select_candidate_windows,
    structured_evidence,
    textual_evidence,
    windows_for_prompt,
)
from src.backend.application.agent.expert_agent.service.policy_question_rewriter import (
    rewrite_question_and_information_needs,
)
from src.backend.application.agent.expert_agent.service.policy_claim_composer import (
    build_claim_first_policy_answer,
)
from src.backend.application.agent.expert_agent.service.policy_need_answer import (
    build_information_need_answer,
)
from src.backend.application.agent.expert_agent.service.policy_need_contracts import (
    NeedFirstPolicyAnswer,
)
from src.backend.application.agent.expert_agent.service.policy_need_synthesis import (
    apply_batch_synthesis,
    build_batch_synthesis_messages,
    should_batch_synthesize,
)
from src.backend.application.agent.expert_agent.tools.case_context_gateway import (
    CaseContextGateway,
    CaseContextObservation,
)
from src.backend.application.agent.expert_agent.tools.policy_rag_mcp import (
    PolicyRagMcpClient,
    UnavailablePolicyRagMcpClient,
    call_policy_rag,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.application.ports.agent_gateway import ModelGateway


POLICY_LIMITS = (
    "当前 Policy Expert 仅提供政策口径和可引用依据，不基于仿真材料形成"
    "当前案件材料缺口差集，不形成拒付、处罚、欺诈或最终审核结论。"
)
POLICY_RAG_TOP_K = 5
POLICY_RAG_SIMPLE_FETCH_K = 20
POLICY_RAG_MEDIUM_FETCH_K = 24
POLICY_RAG_COMPLEX_FETCH_K = 40
POLICY_ANSWER_PIPELINE = str(
    os.environ.get("MEDIGUARD_POLICY_ANSWER_PIPELINE") or "information_need_v2"
).strip()
POLICY_BATCH_SYNTHESIS_ENABLED = str(
    os.environ.get("MEDIGUARD_POLICY_BATCH_SYNTHESIS_ENABLED") or "true"
).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class ExpertAgentResult:
    """Normalized expert result returned to the parent Case Agent."""

    task_id: str
    expert_task_type: ExpertTaskType | str
    status: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    source_refs: list[str] = field(default_factory=list)
    safe_summary: dict[str, Any] = field(default_factory=dict)


class ExpertAgentService:
    """Deep-Agents-style isolated L3 sub-agent.

    The parent Caser graph passes a small ExpertAnalysisTask and runtime
    adapters. The sub-agent never inherits the full conversation history, never
    writes business decisions, and returns only a structured expert fragment.
    """

    def __init__(self) -> None:
        self._graph = build_expert_agent_graph(self)
        self._filter_resolver = PolicyFilterResolver()
        self._recall_filter_projector = RecallFilterProjector()

    def run(
        self,
        *,
        expert_task_type: ExpertTaskType | str,
        input_factors: dict[str, Any] | None = None,
        task: ExpertAnalysisTask | dict[str, Any] | None = None,
        model_gateway: ModelGateway | None = None,
        policy_rag_client: PolicyRagMcpClient | None = None,
        case_context_gateway: CaseContextGateway | None = None,
        repository: Any | None = None,
    ) -> ExpertAgentResult:
        input_factors = input_factors or {}
        task_model = self._coerce_task(
            task=task,
            expert_task_type=str(expert_task_type),
            input_factors=input_factors,
        )
        initial_state: dict[str, Any] = {
            "task_id": task_model.task_id,
            "parent_run_id": task_model.parent_run_id,
            "case_id": task_model.case_id,
            "expert_task_type": task_model.expert_task_type,
            "task": task_model,
            "input_factors": input_factors,
            "runtime": {
                "model_gateway": model_gateway,
                "policy_rag_client": policy_rag_client
                or UnavailablePolicyRagMcpClient(),
                "case_context_gateway": case_context_gateway,
                "repository": repository,
            },
            "model_call_count": 0,
            "tool_call_count": 0,
            "retrieval_round_count": 0,
            "rewrite_count": 0,
            "case_context_observations": [],
            "case_facts_used": [],
            "policy_observations": [],
            "mcp_safe_summaries": [],
            "adopted_policy_evidence": [],
            "source_refs": [],
            "next_action": "init_task",
        }
        try:
            state = self._graph.invoke(initial_state)
        except Exception as exc:
            state = {
                **initial_state,
                "status": "failed",
                "message": "Policy Expert 执行失败，已安全关闭。",
                "error_code": exc.__class__.__name__,
                "error_message": str(exc)[:500],
            }
            state = self._return_unavailable_node(state)

        payload = state.get("result") if isinstance(state.get("result"), dict) else {}
        return ExpertAgentResult(
            task_id=task_model.task_id,
            expert_task_type=task_model.expert_task_type,
            status=str(state.get("status") or payload.get("status") or "unavailable"),
            message=str(state.get("message") or payload.get("expert_answer") or ""),
            payload=payload,
            source_refs=[
                ref for ref in state.get("source_refs", [])
                if isinstance(ref, str) and ref
            ],
            safe_summary=self._safe_state_snapshot(state),
        )

    def generate_policy_retrieval_plan(
        self,
        *,
        user_question: str,
        model_gateway: ModelGateway,
        fact_bundle: list[dict[str, Any]] | None = None,
        semantic_frame: dict[str, Any] | None = None,
        case_state_summary: dict[str, Any] | None = None,
        input_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the production task-understanding filter path without retrieval."""

        task = ExpertAnalysisTask(
            task_id=f"filter_eval_{uuid4().hex}",
            parent_run_id="filter_generation",
            case_id="filter_generation",
            goal="生成 Policy RAG MCP 检索计划",
            user_question=str(user_question or "").strip(),
            fact_bundle=list(fact_bundle or [])[:16],
            context_refs=[],
            allowed_tools=["policy.search_text", "policy.search_version"],
            budget=ExpertAnalysisBudget(
                max_model_calls=2,
                max_tool_calls=1,
                max_retrieval_rounds=1,
                max_llm_rewrite_calls=0,
                timeout_ms=120000,
            ),
            filters=dict(input_filters or {}),
        )
        state: dict[str, Any] = {
            "task": task,
            "input_factors": {
                "semantic_frame": dict(semantic_frame or {}),
                "case_state_summary": dict(case_state_summary or {}),
            },
            "runtime": {
                "model_gateway": model_gateway,
                "repository": None,
            },
            "model_call_count": 0,
            "tool_call_count": 0,
            "retrieval_round_count": 0,
            "rewrite_count": 0,
            "source_refs": [],
        }
        result_state = self._task_understanding_node(state)
        plan = result_state.get("retrieval_plan")
        if not isinstance(plan, dict):
            raise PolicyFilterOutputError(
                str(result_state.get("error_code") or "filter_generation_failed"),
                errors=[
                    item
                    for item in result_state.get("filter_generation_errors", [])
                    if isinstance(item, dict)
                ],
            )
        return plan

    def _init_task_node(self, state: dict[str, Any]) -> dict[str, Any]:
        try:
            task = self._task(state)
        except ValidationError as exc:
            state.update(
                {
                    "status": "failed",
                    "message": "ExpertAnalysisTask 不符合结构化协议。",
                    "error_code": "invalid_expert_task",
                    "error_message": str(exc)[:500],
                    "next_action": "return_unavailable",
                }
            )
            return state
        if task.capability != "ask_policy_expert" or task.expert_task_type != "policy_analysis":
            state.update(
                {
                    "status": "unavailable",
                    "message": "当前版本仅接入 ask_policy_expert / policy_analysis。",
                    "error_code": "unsupported_expert_task",
                    "next_action": "return_unavailable",
                }
            )
            return state
        state["task"] = task
        state["task_id"] = task.task_id
        state["parent_run_id"] = task.parent_run_id
        state["case_id"] = task.case_id
        state["expert_task_type"] = task.expert_task_type
        runtime = state.get("runtime")
        if isinstance(runtime, dict):
            strict_need_window_llm = bool(runtime.get("strict_need_window_llm")) or _bool(
                os.getenv("MEDIGUARD_POLICY_EXPERT_STRICT_NEED_WINDOW_LLM"),
                False,
            )
            runtime.setdefault("skip_slot_window_llm", not strict_need_window_llm)
        state["status"] = "running"
        state["next_action"] = "load_task_context"
        self._record_event(
            state,
            "expert_analysis_started",
            "Policy Expert 子任务已开始",
            {
                "task_id": task.task_id,
                "case_id": task.case_id,
                "allowed_tools": task.allowed_tools,
            },
        )
        self._save_checkpoint(state, "init_task")
        return state

    def _load_task_context_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        state["task_context"] = {
            "task_id": task.task_id,
            "parent_run_id": task.parent_run_id,
            "case_id": task.case_id,
            "goal": task.goal,
            "user_question": task.user_question,
            "fact_bundle": task.fact_bundle,
            "context_refs": task.context_refs,
            "allowed_tools": task.allowed_tools,
            "constraints": task.constraints.model_dump(mode="json"),
            "budget": task.budget.model_dump(mode="json"),
            "filters": task.filters,
            "memory_guidance": task.memory_guidance,
        }
        state["next_action"] = "task_understanding"
        self._save_checkpoint(state, "load_task_context")
        return state

    def _task_understanding_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        prompt_payload = {
            "goal": task.goal,
            "user_question": task.user_question,
            "fact_bundle": task.fact_bundle,
            "context_refs": task.context_refs,
            "input_filters": task.filters,
            "historical_memory_guidance": task.memory_guidance,
            "semantic_frame": self._input_factors(state).get("semantic_frame") or {},
            "case_state_summary": self._input_factors(state).get("case_state_summary") or {},
            "allowed_tools": task.allowed_tools,
            "constraints": task.constraints.model_dump(mode="json"),
        }
        if _has_forced_policy_filters(task.filters):
            forced_resolution = self._resolve_policy_filters(
                task,
                task_filters=task.filters,
            )
            information_needs = _infer_information_needs_from_question(
                task.user_question,
                forced_resolution.filters,
            )
            planned = {
                "policy_question": forced_resolution.policy_question,
                "answer_mode": _infer_answer_mode_from_question(
                    task.user_question,
                    information_needs,
                ),
                "information_needs": information_needs,
                "filters": forced_resolution.filters,
                "fetch_k": POLICY_RAG_COMPLEX_FETCH_K,
                "rerank": False,
                "need_case_context": False,
                "case_context_requests": [],
                "reason": "forced_policy_filters",
                "filter_reason": "trusted caller supplied forced filters",
                "filter_confidence": 1.0,
                "_filter_generation": {
                    "source": "task_filters:forced",
                    "repair_count": 0,
                    "valid_on_first_call": True,
                },
            }
        else:
            planned = self._llm_policy_retrieval_plan(
                state,
                node_name="task_understanding",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是医保政策 Expert Analysis 的任务理解节点。"
                            "只生成检索计划，不回答用户，不形成审核结论。"
                            "必须调用 submit_policy_retrieval_plan 提交完整计划。"
                            "核心契约是 answer_mode + information_needs，"
                            "answer_mode 只能取 fact_lookup/list/process_rule/"
                            "policy_explanation/comparison。"
                            "information_needs 必须拆成用户真正需要回答的子问题，"
                            "多问可返回多个信息需求；不要把它写成固定槽位名。"
                            "filters 是传给 Policy RAG MCP 的精确检索边界："
                            "只选择回答问题确实需要的值，不得为提高召回追加相似领域。"
                            "案件上下文只作为模型判断依据，后续程序不会把它并入 filters。"
                            "复合问题可返回多个 information_needs；Policy RAG MCP "
                            "会负责多领域分组。"
                            "最终上下文由系统按证据组动态选择 6 到 12 条，"
                            "fetch_k 和 rerank 由编排层按查询复杂度确定；"
                            "不要通过扩大 filters 提高召回。"
                            "\n以下是回答契约：\n"
                            + information_need_prompt_contract()
                            + "\n"
                            + "\n以下是注册的 Filter Contract：\n"
                            + policy_filter_prompt_contract()
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt_payload,
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
            )
        if planned is None:
            state.update(
                {
                    "status": "insufficient",
                    "message": "Policy Expert 未能生成有效的政策检索边界。",
                    "error_code": "filter_generation_failed",
                    "next_action": "return_unavailable",
                }
            )
            self._save_checkpoint(state, "task_understanding")
            return state
        plan = self._normalize_retrieval_plan(task, planned, state=state)
        state["retrieval_plan"] = plan
        state["next_action"] = (
            "case_context_adapter"
            if plan.get("need_case_context")
            else "policy_tool_adapter"
        )
        self._record_event(
            state,
            "expert_task_understood",
            "Policy Expert 已生成政策检索计划",
            {
                "task_id": task.task_id,
                "need_case_context": bool(plan.get("need_case_context")),
                "filter_keys": sorted(
                    key for key in plan.get("filters", {}).keys()
                    if isinstance(key, str)
                )
                if isinstance(plan.get("filters"), dict)
                else [],
                "top_k": plan.get("top_k"),
                "fetch_k": plan.get("fetch_k"),
                "rerank": plan.get("rerank"),
                "retrieval_complexity": plan.get("retrieval_complexity"),
                "rerank_candidate_limit": plan.get("rerank_candidate_limit"),
                "answer_mode": plan.get("answer_mode"),
                "information_needs": plan.get("information_needs", []),
                "filter_source": plan.get("filter_source"),
                "filter_confidence": plan.get("filter_confidence"),
                "filter_warnings": plan.get("filter_warnings", []),
            },
        )
        self._save_checkpoint(state, "task_understanding")
        return state

    def _case_context_adapter_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        gateway = self._runtime(state).get("case_context_gateway")
        if "case_context.query" not in task.allowed_tools or gateway is None:
            state["next_action"] = "policy_tool_adapter"
            self._save_checkpoint(state, "case_context_adapter")
            return state

        requests = [
            item for item in state.get("retrieval_plan", {}).get("case_context_requests", [])
            if isinstance(item, dict)
        ]
        if not requests:
            requests = [
                {
                    "capability": "query_case_basic_info",
                    "arguments": {"case_id": task.case_id, "limit": 30},
                }
            ]
        observations = list(state.get("case_context_observations", []))
        for request in requests[:3]:
            if not self._can_use_tool(state):
                break
            capability = str(request.get("capability") or "query_case_basic_info")
            arguments = dict(request.get("arguments") or {})
            try:
                observation: CaseContextObservation = gateway.execute(
                    current_case_id=task.case_id,
                    capability=capability,
                    arguments=arguments,
                )
            except Exception as exc:
                observation = CaseContextObservation(
                    capability=capability,
                    payload={
                        "status": "failed",
                        "message": "案件上下文补查失败。",
                    },
                    source_refs=[],
                    status="failed",
                    error_code=exc.__class__.__name__,
                )
            state["tool_call_count"] = state.get("tool_call_count", 0) + 1
            observations.append(
                {
                    "capability": observation.capability,
                    "status": observation.status,
                    "payload": observation.payload,
                    "source_refs": observation.source_refs,
                    "error_code": observation.error_code,
                }
            )
        state["case_context_observations"] = observations
        state["case_facts_used"] = [
            item.model_dump(mode="json")
            for item in self._case_facts_from_task_and_observations(task, observations)
        ]
        state["retrieval_plan"] = self._enrich_plan_with_case_context(
            task,
            state.get("retrieval_plan", {}),
            observations,
        )
        state["next_action"] = "policy_tool_adapter"
        self._record_event(
            state,
            "expert_case_context_observed",
            "Policy Expert 已读取必要案件上下文",
            {
                "task_id": task.task_id,
                "observation_count": len(observations),
                "source_refs": self._source_refs_from_case_observations(observations)[:8],
            },
        )
        self._save_checkpoint(state, "case_context_adapter")
        return state

    def _policy_tool_adapter_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        if not {"policy.search_text", "policy.search_version"} & set(task.allowed_tools):
            state.update(
                {
                    "status": "unavailable",
                    "message": "Policy Expert 未获授权调用政策检索工具。",
                    "error_code": "policy_tool_not_allowed",
                    "next_action": "return_unavailable",
                }
            )
            return state
        if not self._can_use_tool(state):
            state.update(
                {
                    "status": "insufficient",
                    "message": "Policy Expert 工具调用预算已耗尽。",
                    "error_code": "tool_budget_exhausted",
                    "next_action": "return_unavailable",
                }
            )
            return state

        request = self._policy_request_from_plan(task, state.get("retrieval_plan", {}))
        state["policy_request"] = request.model_dump(mode="json")
        retrieval_plan = state.get("retrieval_plan", {})
        client = self._runtime(state).get("policy_rag_client") or UnavailablePolicyRagMcpClient()
        self._record_event(
            state,
            "expert_policy_rag_calling",
            "Policy Expert 正在请求 Policy RAG MCP",
            {
                "task_id": task.task_id,
                "top_k": request.top_k,
                "fetch_k": request.fetch_k,
                "rerank": request.rerank,
                "filter_strategy": request.filter_strategy,
                "adaptive_top_k": request.adaptive_top_k,
                "filter_keys": sorted(request.filters.keys()),
                "filter_source": (
                    retrieval_plan.get("filter_source")
                    if isinstance(retrieval_plan, dict)
                    else None
                ),
                "filter_confidence": (
                    retrieval_plan.get("filter_confidence")
                    if isinstance(retrieval_plan, dict)
                    else None
                ),
                "filter_warnings": (
                    retrieval_plan.get("filter_warnings", [])
                    if isinstance(retrieval_plan, dict)
                    else []
                ),
            },
        )
        raw_response, safe_summary = call_policy_rag(
            client=client,
            request=request,
            timeout_ms=task.budget.timeout_ms,
        )
        state["tool_call_count"] = state.get("tool_call_count", 0) + 1
        state["retrieval_round_count"] = state.get("retrieval_round_count", 0) + 1
        state["mcp_safe_summary"] = safe_summary
        state.setdefault("mcp_safe_summaries", []).append(safe_summary)
        state["policy_observation"] = {
            "status": raw_response.get("status"),
            "evidence": raw_response.get("evidence")
            if isinstance(raw_response.get("evidence"), list)
            else [],
            "warnings": raw_response.get("warnings")
            if isinstance(raw_response.get("warnings"), list)
            else [],
            "limits": raw_response.get("limits")
            if isinstance(raw_response.get("limits"), list)
            else [],
            "error": raw_response.get("error"),
            "message": raw_response.get("message"),
            "result_count": raw_response.get("result_count"),
            "top_k": raw_response.get("top_k"),
            "filter_strategy": raw_response.get("filter_strategy"),
            "retrieval_diagnostics": raw_response.get("retrieval_diagnostics")
            if isinstance(raw_response.get("retrieval_diagnostics"), dict)
            else {},
        }
        state.setdefault("policy_observations", []).append(
            {
                "request": request.model_dump(mode="json"),
                "safe_summary": safe_summary,
                "status": state["policy_observation"].get("status"),
            }
        )
        state["next_action"] = "hard_gate"
        self._record_event(
            state,
            "expert_policy_rag_called",
            "Policy Expert 已请求 Policy RAG MCP",
            {
                "task_id": task.task_id,
                "mcp_safe_summary": safe_summary,
            },
        )
        self._save_checkpoint(state, "policy_tool_adapter")
        return state

    def _hard_gate_node(self, state: dict[str, Any]) -> dict[str, Any]:
        observation = state.get("policy_observation", {})
        candidates = observation.get("evidence") if isinstance(observation, dict) else []
        adopted = list(state.get("adopted_policy_evidence", []))
        retrieval_plan = state.get("retrieval_plan", {})
        top_k = _int_between(
            observation.get("top_k") if isinstance(observation, dict) else None,
            1,
            12,
            POLICY_RAG_TOP_K,
        )
        candidate_count = len(candidates) if isinstance(candidates, list) else 0
        newly_adopted: list[dict[str, Any]] = []
        for index, item in enumerate(candidates or [], start=1):
            if not isinstance(item, dict):
                continue
            evidence = self._policy_evidence_from_candidate(item, index)
            if evidence is None:
                continue
            newly_adopted.append(evidence.model_dump(mode="json"))
        merge_pool = (
            [*newly_adopted, *adopted]
            if state.get("retrieval_round_count", 0) > 1
            else [*adopted, *newly_adopted]
        )
        adopted = []
        for item in merge_pool:
            if not isinstance(item, dict):
                continue
            if any(
                existing.get("evidence_ref") == item.get("evidence_ref")
                for existing in adopted
            ):
                continue
            adopted.append(item)
        adopted = adopted[:top_k]
        state["adopted_policy_evidence"] = adopted
        state["source_refs"] = list(
            dict.fromkeys(
                ref for ref in [
                    item.get("source_ref") for item in adopted
                    if isinstance(item, dict)
                ]
                if isinstance(ref, str) and ref
            )
        )
        state["hard_gate"] = {
            "status": "pass" if adopted else "insufficient",
            "adopted_evidence_count": len(adopted),
            "mcp_status": observation.get("status") if isinstance(observation, dict) else None,
            "warnings": observation.get("warnings", []) if isinstance(observation, dict) else [],
        }
        state["evidence_normalizer"] = {
            "candidate_count": candidate_count,
            "normalized_count": len(adopted),
            "dropped_count": max(candidate_count - len(adopted), 0),
            "rules": [
                "drop_empty_excerpt",
                "drop_non_citable_except_drug_price_reference",
                "normalize_or_drop_content_type",
                "normalize_or_drop_policy_domain",
            ],
        }
        state["next_action"] = "need_evidence_resolution"
        self._record_event(
            state,
            "expert_evidence_normalized",
            "Policy Expert 已统一政策证据格式",
            {
                "task_id": self._task(state).task_id,
                "candidate_count": candidate_count,
                "adopted_evidence_count": len(adopted),
                "dropped_count": max(candidate_count - len(adopted), 0),
                "top_k": top_k,
            },
        )
        self._save_checkpoint(state, "hard_gate")
        return state

    def _need_evidence_resolution_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Resolve evidence directly against information needs for new runs."""

        if POLICY_ANSWER_PIPELINE != "information_need_v2":
            return self._slot_window_judge_node(state)
        task = self._task(state)
        evidence = [
            PolicyEvidence.model_validate(item)
            for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        filters = normalize_policy_filters(
            (state.get("retrieval_plan") or {}).get("filters") or task.filters
        )
        information_needs = _normalize_information_needs(
            (state.get("retrieval_plan") or {}).get("information_needs")
            or state.get("information_needs")
        )
        if not information_needs:
            information_needs = _infer_information_needs_from_question(
                task.user_question,
                filters,
            ) or [str(task.user_question).strip() or "相关政策信息"]
        answer_mode = _normalize_answer_mode(
            (state.get("retrieval_plan") or {}).get("answer_mode"),
            task.user_question,
            information_needs,
        )
        need_answer = build_information_need_answer(
            user_question=task.user_question,
            information_needs=information_needs,
            answer_mode=answer_mode,
            filters=filters,
            evidence=evidence,
        )
        payload = need_answer.model_dump(mode="json")
        requested_jurisdictions = set(_string_list(filters.get("jurisdiction")))
        compatible_evidence_count = sum(
            _jurisdiction_compatible(requested_jurisdictions, item.jurisdiction)
            for item in evidence
        )
        wrong_jurisdiction = bool(evidence) and not compatible_evidence_count and bool(
            requested_jurisdictions
        )
        state["answer_mode"] = answer_mode
        state["information_needs"] = information_needs
        state["answer_requirements"] = payload.get("answer_requirements", [])
        state["verified_need_facts"] = payload.get("extracted_facts", [])
        state["need_evidence_matches"] = payload.get("evidence_matches", [])
        state["need_answers"] = payload.get("need_answers", [])
        state["need_first_answer"] = payload
        state["need_evidence_resolution"] = {
            "status": "ready",
            "pipeline_version": need_answer.pipeline_version,
            "requirement_count": len(need_answer.answer_requirements),
            "verified_fact_count": len(need_answer.extracted_facts),
            "verified_match_count": len(need_answer.evidence_matches),
            "supported_need_count": sum(
                item.status == "supported" for item in need_answer.need_answers
            ),
            "partial_need_count": sum(
                item.status == "partial" for item in need_answer.need_answers
            ),
            "missing_need_count": sum(
                item.status == "missing" for item in need_answer.need_answers
            ),
            "post_retrieval_model_calls": 0,
            "compatible_evidence_count": compatible_evidence_count,
            "wrong_jurisdiction": wrong_jurisdiction,
        }
        state["next_action"] = "answerability_check"
        self._record_event(
            state,
            "expert_need_evidence_resolved",
            "Policy Expert 已按回答信息点完成证据解析",
            {
                "task_id": task.task_id,
                **state["need_evidence_resolution"],
            },
        )
        self._save_checkpoint(state, "need_evidence_resolution")
        return state

    def _slot_window_judge_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        evidence_models = [
            PolicyEvidence.model_validate(item)
            for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        filters = normalize_policy_filters(
            (state.get("retrieval_plan") or {}).get("filters") or task.filters
        )
        question_slots = self._build_question_slots(task, state)
        information_needs = _normalize_information_needs(
            (state.get("retrieval_plan") or {}).get("information_needs")
        )
        if not information_needs:
            information_needs = _infer_information_needs_from_question(
                task.user_question,
                filters,
            )
        answer_mode = _normalize_answer_mode(
            (state.get("retrieval_plan") or {}).get("answer_mode"),
            task.user_question,
            information_needs,
        )
        requirements = build_information_need_requirements(
            user_question=task.user_question,
            information_needs=information_needs,
            filters=filters,
            answer_mode=answer_mode,
        )
        normalized_evidence = normalize_policy_evidence(evidence_models)
        structured_items = structured_evidence(normalized_evidence)
        text_items = textual_evidence(normalized_evidence)

        structured_facts, structured_matches = extract_facts_for_requirements(
            requirements,
            structured_items,
        )
        structured_facts, structured_matches = reindex_facts_and_matches(
            structured_facts,
            structured_matches,
            fact_start_index=1,
            match_start_index=1,
        )
        if structured_items:
            self._record_event(
                state,
                "expert_structured_facts_extracted",
                "Policy Expert 已从表格证据规则抽取事实",
                {
                    "task_id": task.task_id,
                    "structured_evidence_count": len(structured_items),
                    "structured_fact_count": len(structured_facts),
                    "structured_match_count": len(structured_matches),
                },
            )

        windows = build_sentence_windows(
            requirements=requirements,
            evidence=text_items,
            filters=filters,
            user_question=task.user_question,
        )
        if text_items:
            self._record_event(
                state,
                "expert_sentence_windows_built",
                "Policy Expert 已完成政策正文切句和窗口构建",
                {
                    "task_id": task.task_id,
                    "textual_evidence_count": len(text_items),
                    "sentence_window_count": len(windows),
                    "window_radius": 1,
                },
            )
        candidates = select_candidate_windows(windows, per_slot=6, max_total=24)
        if windows:
            self._record_event(
                state,
                "expert_candidate_windows_selected",
                "Policy Expert 已完成规则粗筛和候选窗口选择",
                {
                    "task_id": task.task_id,
                    "sentence_window_count": len(windows),
                    "candidate_window_count": len(candidates),
                    "per_slot": 6,
                    "max_total": 24,
                },
            )
        fallback_text_facts, fallback_text_matches = extract_facts_for_requirements(
            requirements,
            text_items,
        )
        fallback_text_facts, fallback_text_matches = reindex_facts_and_matches(
            fallback_text_facts,
            fallback_text_matches,
            fact_start_index=len(structured_facts) + 1,
            match_start_index=len(structured_matches) + 1,
        )

        judge_payload: dict[str, Any] = {"window_results": []}
        extract_payload: dict[str, Any] = {"window_results": []}
        llm_facts: list[Any] = []
        llm_matches: list[Any] = []
        diagnostics: dict[str, Any] = {
            "raw_result_count": 0,
            "verified_fact_count": 0,
            "verified_match_count": 0,
            "rejected_count": 0,
            "rejected": [],
        }
        use_slot_window_llm = not bool((state.get("runtime") or {}).get("skip_slot_window_llm"))
        if candidates and use_slot_window_llm:
            judge_payload = self._llm_json(
                state,
                node_name="need_window_judge",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是医保政策 L3 的 Need-Window Judge 节点。"
                            "你只判断给定 sentence window 是否能支撑指定 information_need，"
                            "不要生成事实，不要补充窗口外知识，不要输出未问背景。"
                            "support_role 只能是 direct_answer / supporting / irrelevant。"
                            "只有当该 window 可直接回答该信息需求时使用 direct_answer；"
                            "泛化背景或相邻说明只能用 supporting。"
                            "输出 JSON：{window_results:[{need_id,window_id,llm_score,support_role}]}。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_question": task.user_question,
                                "answer_requirements": [
                                    item.model_dump(mode="json") for item in requirements
                                ],
                                "candidate_windows": windows_for_prompt(candidates),
                                "constraints": {
                                    "direct_answer_threshold": 0.75,
                                    "partial_threshold": 0.55,
                                    "must_quote_from_window": True,
                                    "do_not_expand_unasked_background": True,
                                },
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                fallback={"window_results": []},
            )
            judged_windows = _judged_windows_from_payload(
                candidate_windows=candidates,
                payload=judge_payload,
            )
            extract_payload = self._llm_json(
                state,
                node_name="need_window_extract",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是医保政策 L3 的 Need-Window Extract 节点。"
                            "你只对已通过 Judge 的 sentence window 抽取可写入答案的事实。"
                            "不要生成最终回答，不要补充窗口外知识，不要输出未问背景。"
                            "support_role 只能是 direct_answer / supporting / irrelevant。"
                            "only direct_answer 适合直接写入答案；supporting 只能用于背景补充。"
                            "facts[].matched_required_fields 只能填写 evidence_quote 原文覆盖的字段；"
                            "每个 facts[].evidence_quote 必须是 window_text 中连续出现的原文片段，"
                            "不能改写、不能摘要、不能跨窗口拼接。"
                            "输出 JSON：{window_results:[{need_id,window_id,llm_score,"
                            "support_role,facts:[{fact,evidence_quote,matched_required_fields}]}]}。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_question": task.user_question,
                                "answer_requirements": [
                                    item.model_dump(mode="json") for item in requirements
                                ],
                                "candidate_windows": windows_for_prompt(judged_windows),
                                "judge_results": judge_payload.get("window_results", []),
                                "constraints": {
                                    "must_quote_from_window": True,
                                    "do_not_expand_unasked_background": True,
                                },
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                fallback={"window_results": []},
            )
            llm_facts, llm_matches, diagnostics = extract_verified_facts_from_judgement(
                requirements=requirements,
                candidate_windows=judged_windows,
                payload=extract_payload,
                fact_start_index=len(structured_facts) + 1,
                match_start_index=len(structured_matches) + 1,
            )
        elif candidates:
            judged_windows = candidates[:24]

        if llm_facts:
            text_facts = _merge_slot_facts_for_required_field_coverage(
                requirements=requirements,
                primary_facts=llm_facts,
                fallback_facts=fallback_text_facts,
            )
            text_matches = _merge_evidence_matches(llm_matches, fallback_text_matches)
            extraction_mode = (
                "llm_slot_window_with_deterministic_field_backfill"
                if len(text_facts) > len(llm_facts)
                else "llm_slot_window"
            )
        else:
            text_facts = fallback_text_facts
            text_matches = fallback_text_matches
            extraction_mode = "deterministic_text_fallback"
        field_backfill_fact_count = max(len(text_facts) - len(llm_facts), 0) if llm_facts else 0

        verified_facts = [*structured_facts, *text_facts]
        verified_matches = [*structured_matches, *text_matches]
        state["question_slots"] = question_slots
        state["answer_mode"] = answer_mode
        state["information_needs"] = information_needs
        state["answer_requirements"] = [
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in requirements
        ]
        state["sentence_windows"] = windows[:80]
        state["candidate_windows"] = candidates[:24]
        state["verified_slot_facts"] = [
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in verified_facts
        ]
        state["slot_window_matches"] = [
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in verified_matches
        ]
        slot_window_summary = {
            "status": "ready",
            "extraction_mode": extraction_mode,
            "requirement_count": len(requirements),
            "structured_fact_count": len(structured_facts),
            "sentence_window_count": len(windows),
            "candidate_window_count": len(candidates),
            "llm_verified_fact_count": len(llm_facts),
            "fallback_text_fact_count": len(fallback_text_facts),
            "field_backfill_fact_count": field_backfill_fact_count,
            "verified_fact_count": len(verified_facts),
            "verified_match_count": len(verified_matches),
            "quote_validator": {
                "enabled": True,
                **diagnostics,
            },
        }
        state["slot_window_judge"] = (
            slot_window_summary if use_slot_window_llm else {}
        )
        state["next_action"] = "answerability_check"
        if candidates:
            self._record_event(
                state,
                "expert_slot_window_judged",
                "Policy Expert 已完成句子窗口证据判断与事实抽取",
                {
                    "task_id": task.task_id,
                    "extraction_mode": extraction_mode,
                    "sentence_window_count": len(windows),
                    "candidate_window_count": len(candidates),
                    "verified_fact_count": len(verified_facts),
                    "field_backfill_fact_count": field_backfill_fact_count,
                    "quote_rejected_count": diagnostics.get("rejected_count", 0),
                },
            )
        if structured_items or candidates or fallback_text_facts:
            self._record_event(
                state,
                "expert_quote_validation_completed",
                "Policy Expert 已校验证据原文引用",
                {
                    "task_id": task.task_id,
                    "structured_fact_count": len(structured_facts),
                    "llm_verified_fact_count": len(llm_facts),
                    "fallback_text_fact_count": len(fallback_text_facts),
                    "field_backfill_fact_count": field_backfill_fact_count,
                    "verified_fact_count": len(verified_facts),
                    "quote_rejected_count": diagnostics.get("rejected_count", 0),
                },
            )
        self._save_checkpoint(state, "slot_window_judge")
        return state

    def _answerability_check_node(self, state: dict[str, Any]) -> dict[str, Any]:
        if POLICY_ANSWER_PIPELINE == "information_need_v2":
            if not state.get("need_first_answer"):
                state = self._need_evidence_resolution_node(state)
            return self._need_answerability_check_node(state)
        task = self._task(state)
        evidence = [
            item for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        question_slots = self._build_question_slots(task, state)
        slot_meta_by_slot_id = {
            str(item.get("slot_id") or ""): item
            for item in question_slots
            if isinstance(item, dict) and str(item.get("slot_id") or "")
        }
        raw_coverage_requirements = [
            item for item in state.get("answer_requirements", [])
            if isinstance(item, dict)
        ]
        if raw_coverage_requirements:
            coverage_requirements = []
            for item in raw_coverage_requirements:
                slot_id = str(item.get("slot_id") or "")
                base_slot = slot_meta_by_slot_id.get(slot_id, {})
                merged = {**base_slot, **item}
                if not merged.get("requirement_id"):
                    merged["requirement_id"] = str(item.get("requirement_id") or item.get("need_id") or slot_id)
                coverage_requirements.append(merged)
        else:
            coverage_requirements = question_slots
        coverage_result = self._slot_coverage_gate(task, state, evidence, coverage_requirements)
        if (
            int(coverage_result.get("supported_slot_count") or 0) == 0
            and question_slots
            and raw_coverage_requirements
        ):
            existing_keys = {
                str(item.get("requirement_id") or item.get("need_id") or item.get("slot_id") or "")
                for item in coverage_requirements
                if isinstance(item, dict)
            }
            supplemental_slots = []
            for slot in question_slots:
                if not isinstance(slot, dict):
                    continue
                slot_id = str(slot.get("slot_id") or "")
                if not slot_id or slot_id in existing_keys:
                    continue
                merged = dict(slot)
                merged.setdefault("requirement_id", slot_id)
                supplemental_slots.append(merged)
            if supplemental_slots:
                expanded_requirements = [*coverage_requirements, *supplemental_slots]
                expanded_coverage_result = self._slot_coverage_gate(
                    task,
                    state,
                    evidence,
                    expanded_requirements,
                )
                if int(expanded_coverage_result.get("supported_slot_count") or 0) > 0:
                    coverage_requirements = expanded_requirements
                    coverage_result = expanded_coverage_result
        fallback_semantic_coverage = _fallback_semantic_coverage(coverage_result)
        information_needs = _normalize_information_needs(
            (state.get("retrieval_plan") or {}).get("information_needs")
        )
        if not information_needs:
            information_needs = [
                str(item.get("label") or item.get("need_text") or item.get("question_span") or item.get("slot_id") or "")
                for item in coverage_requirements
                if isinstance(item, dict)
            ]
            information_needs = [item for item in information_needs if item]
        prompt_payload = {
            "user_question": task.user_question,
            "goal": task.goal,
            "case_facts_used": state.get("case_facts_used", []),
            "information_needs": information_needs,
            "question_slots": question_slots,
            "answer_requirements": coverage_requirements,
            "coverage_result": coverage_result,
            "policy_evidence": self._evidence_for_prompt(evidence),
            "constraints": task.constraints.model_dump(mode="json"),
            "rewrite_count": state.get("rewrite_count", 0),
        }
        state["question_slots"] = question_slots
        state["coverage_result"] = coverage_result
        state["generation_policy"] = _generation_policy_from_coverage(coverage_result)
        self._record_event(
            state,
            "expert_answerability_checking",
            "Policy Expert 正在进行证据覆盖门禁",
            {
                "task_id": task.task_id,
                "evidence_count": len(evidence),
                "rewrite_count": state.get("rewrite_count", 0),
                "slot_count": len(coverage_requirements),
                "coverage_score": coverage_result.get("coverage_score"),
            },
        )
        if state.get("slot_window_judge"):
            semantic_coverage = fallback_semantic_coverage
        else:
            payload = self._llm_json(
                state,
                node_name="answerability_check",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 Policy Expert 内部 llm_semantic_coverage 节点。"
                            "你只做语义辅助，不做最终 answerability 裁决。"
                            "不得把已有可引用证据判成不存在，不得把低质量证据升级成强证据，"
                            "不得新增政策事实。输出 JSON：supported_slots, weak_slots, "
                            "missing_slots, usable_evidence_refs, coverage_comment。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt_payload, ensure_ascii=False, default=str),
                    },
                ],
                fallback=fallback_semantic_coverage,
            )
            semantic_coverage = _coerce_semantic_coverage(payload, fallback_semantic_coverage)
        check = self._answerability_decision(
            state,
            evidence=evidence,
            coverage_result=coverage_result,
            semantic_coverage=semantic_coverage,
        )
        state["answerability_check"] = check
        state["llm_semantic_coverage"] = semantic_coverage
        if check.next_action == "synthesize" and evidence:
            state["next_action"] = "expert_synthesis"
        elif (
            check.next_action == "rewrite"
            and state.get("rewrite_count", 0) < task.budget.max_llm_rewrite_calls
            and state.get("retrieval_round_count", 0) < task.budget.max_retrieval_rounds
        ):
            state["next_action"] = "question_rewrite"
        elif evidence and check.answerability == "partial":
            state["next_action"] = "expert_synthesis"
        else:
            state.update(
                {
                    "status": "insufficient",
                    "message": "未检索到足够、可引用的政策证据回答该问题。",
                    "error_code": check.reason_code or "policy_evidence_insufficient",
                    "next_action": "return_unavailable",
                }
            )
        self._record_event(
            state,
            "expert_answerability_checked",
            "Policy Expert 已完成证据覆盖决策",
            {
                **check.model_dump(mode="json"),
                "coverage_result": coverage_result,
                "llm_semantic_coverage": semantic_coverage,
                "generation_policy": state.get("generation_policy", {}),
            },
        )
        self._save_checkpoint(state, "answerability_check")
        return state

    def _need_answerability_check_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decide coverage from NeedAnswer statuses without slot inference or LLM."""

        task = self._task(state)
        raw_answer = state.get("need_first_answer")
        try:
            need_answer = NeedFirstPolicyAnswer.model_validate(raw_answer)
        except ValidationError:
            state.update(
                {
                    "status": "insufficient",
                    "message": "信息点证据解析结果无效，Policy Expert 不生成政策回答。",
                    "error_code": "invalid_need_evidence_resolution",
                    "next_action": "return_unavailable",
                }
            )
            return state
        covered = [
            item.need_text
            for item in need_answer.need_answers
            if item.status in {"supported", "partial"}
        ]
        missing = [
            item.need_text
            for item in need_answer.need_answers
            if item.status in {"missing", "conflicted"}
        ]
        has_answer = bool(covered)
        all_supported = bool(need_answer.need_answers) and all(
            item.status == "supported" for item in need_answer.need_answers
        )
        can_rewrite = (
            state.get("rewrite_count", 0) < task.budget.max_llm_rewrite_calls
            and state.get("retrieval_round_count", 0) < task.budget.max_retrieval_rounds
        )
        wrong_jurisdiction = bool(
            (state.get("need_evidence_resolution") or {}).get("wrong_jurisdiction")
        )
        if all_supported:
            check = AnswerabilityCheck(
                answerability="answerable",
                covered_needs=covered,
                missing_needs=[],
                usable_evidence_refs=need_answer.source_refs,
                next_action="synthesize",
                reason_code="all_information_needs_supported",
                reason="All required information needs have direct, user-safe evidence.",
            )
        elif has_answer:
            check = AnswerabilityCheck(
                answerability="partial",
                covered_needs=covered,
                missing_needs=missing,
                usable_evidence_refs=need_answer.source_refs,
                next_action="synthesize",
                reason_code="partial_information_need_coverage",
                reason=(
                    "Some information needs are supported only partially or remain missing."
                ),
            )
        else:
            check = AnswerabilityCheck(
                answerability="insufficient",
                covered_needs=[],
                missing_needs=missing or need_answer.information_needs,
                usable_evidence_refs=[],
                next_action="rewrite" if can_rewrite else "insufficient",
                reason_code=(
                    "wrong_jurisdiction"
                    if wrong_jurisdiction
                    else "no_information_need_coverage"
                ),
                reason=(
                    "Retrieved evidence does not match the requested jurisdiction."
                    if wrong_jurisdiction
                    else "No information need has direct, user-safe evidence."
                ),
            )
        coverage_result = {
            "pipeline_version": need_answer.pipeline_version,
            "need_status": {
                item.need_id: {
                    "need_id": item.need_id,
                    "need_text": item.need_text,
                    "status": item.status,
                    "fact_refs": item.fact_refs,
                    "source_refs": item.source_refs,
                    "missing_reason": item.missing_reason,
                }
                for item in need_answer.need_answers
            },
            "covered_need_count": len(covered),
            "missing_need_count": len(missing),
            "supported_need_count": sum(
                item.status == "supported" for item in need_answer.need_answers
            ),
            "coverage_ratio": round(
                len(covered) / max(len(need_answer.need_answers), 1), 3
            ),
            "usable_evidence_refs": need_answer.source_refs,
            "citable_evidence_count": len(
                {
                    ref
                    for item in need_answer.extracted_facts
                    for ref in item.evidence_refs
                }
            ),
            "claim_plan": need_answer.claim_plan,
            "coverage_basis": "information_need_facts",
            "decision_reason": "information_need_coverage_gate",
            "wrong_jurisdiction": wrong_jurisdiction,
        }
        state["answerability_check"] = check
        state["coverage_result"] = coverage_result
        state["generation_policy"] = {
            item.need_id: (
                "allow"
                if item.status == "supported"
                else "allow_with_qualification"
                if item.status == "partial"
                else "deny"
            )
            for item in need_answer.need_answers
        }
        if check.next_action == "synthesize":
            state["next_action"] = "expert_synthesis"
        elif check.next_action == "rewrite":
            state["next_action"] = "question_rewrite"
        else:
            state.update(
                {
                    "status": "insufficient",
                    "message": "未检索到能够直接回答这些信息点的可引用政策证据。",
                    "error_code": check.reason_code,
                    "next_action": "expert_synthesis",
                }
            )
        self._record_event(
            state,
            "expert_need_coverage_checked",
            "Policy Expert 已完成信息点覆盖判断",
            {
                "task_id": task.task_id,
                **check.model_dump(mode="json"),
                "coverage_result": coverage_result,
            },
        )
        self._save_checkpoint(state, "answerability_check")
        return state

    def _follow_up_retrieval_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        if state.get("rewrite_count", 0) >= task.budget.max_llm_rewrite_calls:
            state["next_action"] = "expert_synthesis" if state.get("adopted_policy_evidence") else "return_unavailable"
            return state
        current_plan = state.get("retrieval_plan", {})
        answerability = self._answerability_model_dump(state)
        self._record_event(
            state,
            "expert_query_rewriting",
            "Policy Expert 正在重写政策检索问题",
            {
                "task_id": task.task_id,
                "rewrite_count": state.get("rewrite_count", 0) + 1,
                "missing_need_count": len(
                    answerability.get("missing_needs")
                    or answerability.get("missing_slots")
                    or []
                )
                if isinstance(
                    answerability.get("missing_needs")
                    or answerability.get("missing_slots"),
                    list,
                )
                else 0,
            },
        )
        payload = self._llm_policy_retrieval_plan(
            state,
            node_name="follow_up_retrieval",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 Policy Expert 的一次性检索改写节点。"
                        "只允许基于 missing_needs 和 failed_evidence_summary 改写 "
                        "policy_question、information_needs 或收窄 filters，"
                        "不要扩大到未授权领域。必须调用 submit_policy_retrieval_plan "
                        "提交完整计划；不得只返回修改字段。"
                        "top_k 只是系统内部的种子值，最终上下文由系统按证据组"
                        "自适应到 6/8/10/12 条。\n"
                        + information_need_prompt_contract()
                        + "\n"
                        + policy_filter_prompt_contract()
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_plan": current_plan,
                            "answerability": answerability,
                            "user_question": task.user_question,
                            "failed_evidence_summary": _string_list(
                                current_plan.get("filter_warnings")
                            )[:4],
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
        )
        state["rewrite_count"] = state.get("rewrite_count", 0) + 1
        if payload is None:
            state.update(
                {
                    "error_code": "filter_generation_failed",
                    "next_action": "expert_synthesis"
                    if state.get("adopted_policy_evidence")
                    else "return_unavailable",
                }
            )
            self._save_checkpoint(state, "follow_up_retrieval")
            return state
        rewritten = rewrite_question_and_information_needs(
            original_question=task.user_question,
            answer_mode=str(
                payload.get("answer_mode")
                or current_plan.get("answer_mode")
                or "process_rule"
            ),
            information_needs=_normalize_information_needs(
                payload.get("information_needs")
                or current_plan.get("information_needs")
            ),
            missing_needs=_normalize_information_needs(
                answerability.get("missing_needs")
                or answerability.get("missing_slots")
                or []
            ),
            failed_evidence_summary=_string_list(current_plan.get("filter_warnings"))[:4],
        )
        state["retrieval_plan"] = self._normalize_retrieval_plan(
            task,
            payload,
            state=state,
        )
        state["retrieval_plan"]["rewritten_question"] = rewritten.get("rewritten_question")
        state["retrieval_plan"]["rewritten_information_needs"] = rewritten.get(
            "rewritten_information_needs", []
        )
        state["retrieval_plan"]["rewrite_mode"] = rewritten.get("rewrite_mode")
        state["retrieval_plan"]["need_case_context"] = False
        state["next_action"] = "policy_tool_adapter"
        self._record_event(
            state,
            "expert_query_rewritten",
            "Policy Expert 已重写政策检索问题",
            {
                "task_id": task.task_id,
                "rewrite_count": state.get("rewrite_count", 0),
                "reason": state["retrieval_plan"].get("reason"),
            },
        )
        self._save_checkpoint(state, "follow_up_retrieval")
        return state

    def _need_synthesis_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Optionally polish complex need answers once, then return verified output."""

        task = self._task(state)
        evidence = [
            PolicyEvidence.model_validate(item)
            for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        try:
            need_answer = NeedFirstPolicyAnswer.model_validate(
                state.get("need_first_answer")
            )
        except ValidationError:
            state.update(
                {
                    "status": "failed",
                    "message": "信息点回答结果无效，Policy Expert 已安全停止。",
                    "error_code": "invalid_need_first_answer",
                    "next_action": "return_unavailable",
                }
            )
            return self._return_unavailable_node(state)
        answerability = self._coerce_answerability(
            self._answerability_model_dump(state),
            {
                "answerability": "insufficient",
                "covered_needs": [],
                "missing_needs": need_answer.information_needs,
                "usable_evidence_refs": [],
                "next_action": "insufficient",
                "reason_code": "missing_need_coverage",
                "reason": "Information-need coverage is unavailable.",
            },
        )
        synthesis_status = "not_needed"
        post_retrieval_model_calls = 0
        if POLICY_BATCH_SYNTHESIS_ENABLED and should_batch_synthesize(need_answer):
            before_calls = int(state.get("model_call_count") or 0)
            synthesis_payload = self._llm_json(
                state,
                node_name="information_need_batch_synthesis",
                messages=build_batch_synthesis_messages(
                    user_question=task.user_question,
                    answer=need_answer,
                ),
                fallback={"answers": []},
                thinking_enabled=False,
                temperature=0.1,
                max_tokens=1200,
                timeout_seconds=min(12.0, max(3.0, task.budget.timeout_ms / 1000)),
            )
            post_retrieval_model_calls = max(
                int(state.get("model_call_count") or 0) - before_calls,
                0,
            )
            synthesized = apply_batch_synthesis(need_answer, synthesis_payload)
            if synthesized is not None:
                need_answer = synthesized
                synthesis_status = "applied"
                state["need_first_answer"] = need_answer.model_dump(mode="json")
                state["need_answers"] = [
                    item.model_dump(mode="json") for item in need_answer.need_answers
                ]
            else:
                synthesis_status = (
                    "validation_fallback"
                    if post_retrieval_model_calls
                    else "gateway_unavailable_fallback"
                )
        payload = need_answer.model_dump(mode="json")
        result = ExpertAnalysisResult(
            status=(
                "ok"
                if need_answer.support_status == "supported"
                else "partial"
                if need_answer.support_status in {"partial", "conflicted"}
                else "insufficient"
            ),
            pipeline_version=need_answer.pipeline_version,
            answer_mode=need_answer.answer_mode,
            information_needs=need_answer.information_needs,
            expert_answer=need_answer.expert_answer,
            answer_markdown=need_answer.answer_markdown,
            answer_requirements=payload.get("answer_requirements", []),
            evidence_matches=payload.get("evidence_matches", []),
            extracted_facts=payload.get("extracted_facts", []),
            need_answers=payload.get("need_answers", []),
            claims=payload.get("claims", []),
            claim_plan=payload.get("claim_plan", []),
            citations=payload.get("citations", []),
            case_facts_used=[
                CaseFactUsed.model_validate(item)
                for item in state.get("case_facts_used", [])
                if isinstance(item, dict)
            ],
            policy_evidence=evidence,
            material_gaps=[],
            audit_suggestions=[],
            answerability_summary=answerability,
            coverage_result=state.get("coverage_result", {}),
            generation_policy=state.get("generation_policy", {}),
            filter_diagnostics={
                **dict(
                    (state.get("retrieval_plan") or {}).get("filter_diagnostics")
                    or {}
                ),
                "answer_pipeline": need_answer.pipeline_version,
                "post_retrieval_model_calls": post_retrieval_model_calls,
                "batch_synthesis": synthesis_status,
            },
            limits=POLICY_LIMITS,
        )
        state["result_model"] = result
        state["result"] = result.model_dump(mode="json")
        state["status"] = result.status
        state["message"] = result.expert_answer
        state["source_refs"] = need_answer.source_refs
        state["next_action"] = ""
        self._record_event(
            state,
            "expert_analysis_completed",
            "Policy Expert 已按信息点生成结构化专家结果",
            {
                "task_id": task.task_id,
                "status": result.status,
                "pipeline_version": need_answer.pipeline_version,
                "need_answer_count": len(need_answer.need_answers),
                "claim_count": len(need_answer.claims),
                "source_refs": need_answer.source_refs[:8],
                "batch_synthesis": synthesis_status,
                "post_retrieval_model_calls": post_retrieval_model_calls,
            },
        )
        self._save_checkpoint(state, "expert_synthesis")
        return state

    def _expert_synthesis_node(self, state: dict[str, Any]) -> dict[str, Any]:
        if POLICY_ANSWER_PIPELINE == "information_need_v2":
            return self._need_synthesis_node(state)
        task = self._task(state)
        evidence = [
            PolicyEvidence.model_validate(item)
            for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        if task.constraints.must_cite_evidence and not evidence:
            state.update(
                {
                    "status": "insufficient",
                    "message": "没有可引用政策证据，Policy Expert 不生成专家回答。",
                    "error_code": "no_citable_policy_evidence",
                    "next_action": "return_unavailable",
                }
            )
            return self._return_unavailable_node(state)

        case_facts = [
            CaseFactUsed.model_validate(item)
            for item in state.get("case_facts_used", [])
            if isinstance(item, dict)
        ]
        answerability = self._coerce_answerability(
            self._answerability_model_dump(state),
            self._deterministic_answerability(
                state,
                has_evidence=bool(evidence),
            ).model_dump(mode="json"),
        )
        precomputed_requirements = [
            AnswerRequirement.model_validate(item)
            for item in state.get("answer_requirements", [])
            if isinstance(item, dict)
        ] or None
        precomputed_facts = [
            ExtractedFact.model_validate(item)
            for item in state.get("verified_slot_facts", [])
            if isinstance(item, dict)
        ] or None
        precomputed_matches = [
            EvidenceMatch.model_validate(item)
            for item in state.get("slot_window_matches", [])
            if isinstance(item, dict)
        ] or None
        claim_first_answer = build_claim_first_policy_answer(
            task=task,
            evidence=evidence,
            answerability=answerability,
            question_slots=[
                item for item in state.get("question_slots", [])
                if isinstance(item, dict)
            ],
            information_needs=_normalize_information_needs(
                (state.get("retrieval_plan") or {}).get("information_needs")
                or state.get("information_needs")
            ),
            filters=(state.get("retrieval_plan") or {}).get("filters") or task.filters,
            precomputed_requirements=precomputed_requirements,
            precomputed_facts=precomputed_facts,
            precomputed_matches=precomputed_matches,
            coverage_result=state.get("coverage_result", {}),
        )
        if claim_first_answer.support_status == "supported" and answerability.answerability == "partial":
            answerability = answerability.model_copy(
                update={
                    "answerability": "answerable",
                    "missing_slots": [],
                    "next_action": "synthesize",
                    "reason_code": "claim_first_requirements_covered",
                    "reason": "Claim-first extraction covered all required answer requirements.",
                }
            )
        claim_first_payload = claim_first_answer.model_dump(mode="json")
        fallback_answer = (
            self._deterministic_expert_answer(task, evidence, answerability)
            if _has_drug_price_reference_evidence(evidence)
            else claim_first_answer.expert_answer
        )
        if claim_first_answer.claims:
            self._record_event(
                state,
                "expert_claims_composed",
                "Policy Expert 已将验证事实组织为可引用 claim",
                {
                    "task_id": task.task_id,
                    "claim_count": len(claim_first_answer.claims),
                    "citation_count": len(claim_first_answer.citations),
                    "support_status": claim_first_answer.support_status,
                },
            )
            if claim_first_answer.answer_markdown:
                self._record_event(
                    state,
                    "expert_markdown_rendered",
                    "Policy Expert 已生成句内引用 Markdown",
                    {
                        "task_id": task.task_id,
                        "claim_count": len(claim_first_answer.claims),
                        "citation_count": len(claim_first_answer.citations),
                    },
                )
        self._record_event(
            state,
            "expert_synthesizing",
            "Policy Expert 正在生成结构化专家结果",
            {
                "task_id": task.task_id,
                "evidence_count": len(evidence),
                "model_required": not _has_drug_price_reference_evidence(evidence),
                "claim_first_fact_count": len(claim_first_answer.extracted_facts),
                "claim_first_claim_count": len(claim_first_answer.claims),
            },
        )
        payload = {
            "expert_answer": claim_first_answer.expert_answer,
            "answer_markdown": claim_first_answer.answer_markdown,
            "answer_requirements": claim_first_payload.get("answer_requirements", []),
            "evidence_matches": claim_first_payload.get("evidence_matches", []),
            "extracted_facts": claim_first_payload.get("extracted_facts", []),
            "claims": claim_first_payload.get("claims", []),
            "claim_plan": claim_first_payload.get("claim_plan", []),
            "citations": claim_first_payload.get("citations", []),
            "audit_suggestions": [],
        }
        expert_answer = str(payload.get("expert_answer") or fallback_answer)[:2000]
        answer_markdown = str(payload.get("answer_markdown") or expert_answer)[:4000]
        if answerability.answerability == "partial" and any(
            value == "deny" for value in (state.get("generation_policy") or {}).values()
        ) and claim_first_answer.support_status != "supported":
            expert_answer = fallback_answer[:2000]
            answer_markdown = claim_first_answer.answer_markdown[:4000] or expert_answer
        result = ExpertAnalysisResult(
            status=(
                "partial"
                if answerability.answerability == "partial"
                and claim_first_answer.support_status != "supported"
                else "ok"
            ),
            pipeline_version="legacy_slot_v1",
            answer_mode=claim_first_answer.answer_mode,
            information_needs=claim_first_answer.information_needs,
            expert_answer=expert_answer,
            answer_markdown=answer_markdown,
            answer_requirements=[
                item for item in payload.get("answer_requirements", [])
                if isinstance(item, dict)
            ],
            evidence_matches=[
                item for item in payload.get("evidence_matches", [])
                if isinstance(item, dict)
            ],
            extracted_facts=[
                item for item in payload.get("extracted_facts", [])
                if isinstance(item, dict)
            ],
            claims=[
                item for item in payload.get("claims", [])
                if isinstance(item, dict)
            ],
            claim_plan=[
                item for item in payload.get("claim_plan", [])
                if isinstance(item, dict)
            ],
            citations=[
                item for item in payload.get("citations", [])
                if isinstance(item, dict)
            ],
            case_facts_used=case_facts,
            policy_evidence=evidence,
            material_gaps=[],
            audit_suggestions=self._coerce_audit_suggestions(
                payload.get("audit_suggestions"),
                [item.source_ref for item in evidence],
            ),
            answerability_summary=answerability,
            coverage_result=state.get("coverage_result", {}),
            generation_policy=state.get("generation_policy", {}),
            filter_diagnostics=state.get("retrieval_plan", {}).get("filter_diagnostics", {}),
            limits=POLICY_LIMITS,
        )
        state["result_model"] = result
        state["result"] = result.model_dump(mode="json")
        state["status"] = result.status
        state["message"] = result.expert_answer
        state["source_refs"] = [item.source_ref for item in evidence]
        state["next_action"] = ""
        self._record_event(
            state,
            "expert_analysis_completed",
            "Policy Expert 已生成结构化专家结果",
            {
                "task_id": task.task_id,
                "status": result.status,
                "source_refs": state["source_refs"][:8],
                "answerability": answerability.answerability,
            },
        )
        self._save_checkpoint(state, "expert_synthesis")
        return state

    def _return_unavailable_node(self, state: dict[str, Any]) -> dict[str, Any]:
        task = self._task(state)
        evidence = [
            PolicyEvidence.model_validate(item)
            for item in state.get("adopted_policy_evidence", [])
            if isinstance(item, dict)
        ]
        answerability = self._coerce_answerability(
            self._answerability_model_dump(state),
            {
                "answerability": "insufficient",
                "covered_slots": [],
                "missing_slots": ["policy_evidence"],
                "usable_evidence_refs": [],
                "next_action": "insufficient",
                "reason_code": state.get("error_code") or "unavailable",
                "reason": state.get("message") or "Policy Expert 当前不可用。",
            },
        )
        status = str(state.get("status") or "unavailable")
        if status not in {"insufficient", "unavailable", "failed"}:
            status = "unavailable"
        retained_evidence = evidence if status == "insufficient" else []
        if retained_evidence and not answerability.usable_evidence_refs:
            answerability = answerability.model_copy(
                update={
                    "usable_evidence_refs": [
                        item.evidence_ref for item in retained_evidence
                    ][:20]
                }
            )
        expert_answer = str(state.get("message") or "").strip()
        if status == "insufficient" and retained_evidence:
            expert_answer = expert_answer or self._insufficient_expert_answer(
                task,
                retained_evidence,
                answerability,
            )
        else:
            expert_answer = expert_answer or "当前没有足够可引用的政策证据回答该问题。"
        result = ExpertAnalysisResult(
            status=status,  # type: ignore[arg-type]
            pipeline_version=POLICY_ANSWER_PIPELINE or "information_need_v2",
            answer_mode=str((state.get("retrieval_plan") or {}).get("answer_mode") or "process_rule"),
            information_needs=_normalize_information_needs(
                (state.get("retrieval_plan") or {}).get("information_needs")
            ),
            expert_answer=expert_answer[:2000],
            case_facts_used=[
                CaseFactUsed.model_validate(item)
                for item in state.get("case_facts_used", [])
                if isinstance(item, dict)
            ],
            policy_evidence=retained_evidence,
            material_gaps=[],
            audit_suggestions=[],
            answerability_summary=answerability,
            coverage_result=state.get("coverage_result", {}),
            generation_policy=state.get("generation_policy", {}),
            filter_diagnostics=state.get("retrieval_plan", {}).get("filter_diagnostics", {}),
            limits=POLICY_LIMITS,
        )
        state["result_model"] = result
        state["result"] = result.model_dump(mode="json")
        state["status"] = result.status
        state["message"] = result.expert_answer
        state["source_refs"] = [item.source_ref for item in retained_evidence]
        self._record_event(
            state,
            "expert_analysis_unavailable",
            "Policy Expert 已安全关闭",
            {
                "task_id": task.task_id,
                "status": result.status,
                "error_code": state.get("error_code"),
            },
        )
        self._save_checkpoint(state, "return_unavailable")
        return state

    @staticmethod
    def _insufficient_expert_answer(
        task: ExpertAnalysisTask,
        evidence: list[PolicyEvidence],
        answerability: AnswerabilityCheck,
    ) -> str:
        titles = list(
            dict.fromkeys(
                item.title for item in evidence
                if isinstance(item.title, str) and item.title
            )
        )[:3]
        covered = "、".join(answerability.covered_slots[:4]) or "部分相关政策依据"
        missing = "、".join(answerability.missing_slots[:4]) or "回答该问题所需的关键政策口径"
        title_text = "；已有证据包括：" + "、".join(titles) if titles else ""
        return (
            f"当前检索到的可引用政策证据只能覆盖{covered}，"
            f"不足以完整回答“{task.user_question}”；仍缺少{missing}。"
            f"{title_text}。因此本助手不会使用通用知识补充政策事实。"
        )

    def _coerce_task(
        self,
        *,
        task: ExpertAnalysisTask | dict[str, Any] | None,
        expert_task_type: str,
        input_factors: dict[str, Any],
    ) -> ExpertAnalysisTask:
        if isinstance(task, ExpertAnalysisTask):
            return task
        if isinstance(task, dict):
            return ExpertAnalysisTask.model_validate(task)
        task_id = str(input_factors.get("task_id") or f"extask_{uuid4().hex}")
        parent_run_id = str(input_factors.get("parent_run_id") or "standalone")
        case_id = str(input_factors.get("case_id") or "unknown")
        question = str(
            input_factors.get("user_question")
            or input_factors.get("question")
            or "请查询相关政策口径。"
        )
        return ExpertAnalysisTask(
            task_id=task_id,
            parent_run_id=parent_run_id,
            case_id=case_id,
            expert_task_type=expert_task_type,
            goal=str(input_factors.get("goal") or "查询政策口径")[:500],
            user_question=question[:2000],
            fact_bundle=[
                item for item in input_factors.get("fact_bundle", [])
                if isinstance(item, dict)
            ][:16],
            context_refs=[
                str(item) for item in input_factors.get("source_refs", [])
                if item
            ][:40],
            filters=dict(input_factors.get("filters") or {}),
        )

    @staticmethod
    def _task(state: dict[str, Any]) -> ExpertAnalysisTask:
        task = state.get("task")
        if isinstance(task, ExpertAnalysisTask):
            return task
        return ExpertAnalysisTask.model_validate(task)

    @staticmethod
    def _runtime(state: dict[str, Any]) -> dict[str, Any]:
        runtime = state.get("runtime")
        return runtime if isinstance(runtime, dict) else {}

    @staticmethod
    def _input_factors(state: dict[str, Any]) -> dict[str, Any]:
        factors = state.get("input_factors")
        return factors if isinstance(factors, dict) else {}

    def _resolve_policy_filters(
        self,
        task: ExpertAnalysisTask,
        *,
        semantic_frame: dict[str, Any] | None = None,
        case_state_summary: dict[str, Any] | None = None,
        fact_bundle: list[dict[str, Any]] | None = None,
        task_filters: dict[str, Any] | None = None,
    ):
        return self._filter_resolver.resolve(
            PolicyFilterResolverInput(
                user_question=task.user_question,
                semantic_frame=semantic_frame or {},
                fact_bundle=fact_bundle if fact_bundle is not None else task.fact_bundle,
                case_state_summary=case_state_summary or {},
                task_filters=task_filters if task_filters is not None else task.filters,
            )
        )

    def _llm_json(
        self,
        state: dict[str, Any],
        *,
        node_name: str,
        messages: list[dict[str, Any]],
        fallback: dict[str, Any],
        thinking_enabled: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        task = self._task(state)
        if state.get("model_call_count", 0) >= task.budget.max_model_calls:
            return fallback
        gateway = self._runtime(state).get("model_gateway")
        if gateway is None:
            return fallback
        try:
            response = gateway.complete(
                messages=messages,
                tools=[],
                require_json=True,
                thinking_enabled=thinking_enabled,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            state["model_call_count"] = state.get("model_call_count", 0) + 1
            content = response.content if isinstance(response, ModelResponse) else str(response)
            payload = json.loads(_extract_json_object(content or "{}"))
            return payload if isinstance(payload, dict) else fallback
        except Exception as exc:
            self._record_event(
                state,
                "expert_model_call_failed",
                "Policy Expert 模型节点调用失败，使用安全回退。",
                {
                    "node_name": node_name,
                    "error_type": exc.__class__.__name__,
                },
            )
            return fallback

    def _llm_policy_retrieval_plan(
        self,
        state: dict[str, Any],
        *,
        node_name: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Generate one registered filter plan with at most one same-model repair."""

        task = self._task(state)
        gateway = self._runtime(state).get("model_gateway")
        if gateway is None:
            fallback = self._recall_only_fallback_retrieval_plan(
                task,
                reason="policy_model_gateway_unavailable_recall_only_fallback",
                validation_errors=[
                    {
                        "type": "missing_model_gateway",
                        "msg": "Policy filter model gateway unavailable",
                    }
                ],
                repair_count=0,
            )
            state["filter_generation_errors"] = fallback.get(
                "filter_diagnostics", {}
            ).get("strict_filter_generation_errors", [])
            return fallback
        tool_schema = policy_retrieval_plan_tool_schema()
        validation_errors: list[dict[str, Any]] = []
        raw_output = ""
        for attempt in range(2):
            if state.get("model_call_count", 0) >= task.budget.max_model_calls:
                break
            request_messages = list(messages)
            if attempt:
                request_messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": raw_output[:6000],
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "instruction": (
                                        "上一次提交未通过 Filter Contract。"
                                        "请重新调用 submit_policy_retrieval_plan，"
                                        "返回一份完整计划，不要只返回修补字段。"
                                    ),
                                    "validation_errors": validation_errors,
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    ]
                )
            try:
                response = gateway.complete(
                    messages=request_messages,
                    tools=[tool_schema],
                    require_json=False,
                    thinking_enabled=False,
                    temperature=0.1,
                )
                state["model_call_count"] = state.get("model_call_count", 0) + 1
                raw_output = _policy_plan_raw_output(response)
                plan = self._filter_resolver.resolve_llm_plan(raw_output)
                plan["_filter_generation"] = {
                    "source": "llm_registered",
                    "repair_count": attempt,
                    "valid_on_first_call": attempt == 0,
                }
                return plan
            except PolicyFilterOutputError as exc:
                validation_errors = list(exc.errors)
                self._record_event(
                    state,
                    "expert_filter_plan_invalid",
                    "Policy Expert 检索计划未通过结构校验。",
                    {
                        "node_name": node_name,
                        "attempt": attempt + 1,
                        "error_count": len(validation_errors),
                    },
                )
            except Exception as exc:
                validation_errors = [
                    {
                        "type": exc.__class__.__name__,
                        "msg": str(exc)[:300],
                    }
                ]
                self._record_event(
                    state,
                    "expert_model_call_failed",
                    "Policy Expert 模型节点调用失败。",
                    {
                        "node_name": node_name,
                        "attempt": attempt + 1,
                        "error_type": exc.__class__.__name__,
                    },
                )
        state["filter_generation_errors"] = validation_errors
        fallback = self._recall_only_fallback_retrieval_plan(
            task,
            reason="strict_filter_generation_failed_recall_only_fallback",
            validation_errors=validation_errors,
            repair_count=1,
        )
        self._record_event(
            state,
            "expert_filter_plan_fallback_to_recall_only",
            "Policy Expert 检索计划回退到宽召回边界。",
            {
                "node_name": node_name,
                "validation_error_count": len(validation_errors),
                "filter_source": fallback.get("filter_source"),
            },
        )
        return fallback

    def _deterministic_retrieval_plan(
        self,
        task: ExpertAnalysisTask,
        *,
        semantic_frame: dict[str, Any] | None = None,
        case_state_summary: dict[str, Any] | None = None,
        task_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolution = self._resolve_policy_filters(
            task,
            semantic_frame=semantic_frame,
            case_state_summary=case_state_summary,
            task_filters=task_filters,
        )
        filters = resolution.filters
        need_case_context = (
            "case_context.query" in task.allowed_tools
            and not filters.get("jurisdiction")
        )
        information_needs = _infer_information_needs_from_question(
            task.user_question,
            filters,
        )
        return {
            "policy_question": resolution.policy_question,
            "answer_mode": _infer_answer_mode_from_question(
                task.user_question,
                information_needs,
            ),
            "information_needs": information_needs,
            "filters": filters,
            "top_k": POLICY_RAG_TOP_K,
            "fetch_k": 40,
            "rerank": False,
            "need_case_context": need_case_context,
            "case_context_requests": [
                {
                    "capability": "query_case_basic_info",
                    "arguments": {"case_id": task.case_id, "limit": 30},
                }
            ]
            if need_case_context
            else [],
            "reason": "deterministic_policy_query_plan",
            "filter_source": resolution.filter_source,
            "filter_confidence": resolution.confidence,
            "filter_warnings": resolution.warnings,
        }

    def _recall_only_fallback_retrieval_plan(
        self,
        task: ExpertAnalysisTask,
        *,
        reason: str,
        validation_errors: list[dict[str, Any]] | None = None,
        repair_count: int = 1,
    ) -> dict[str, Any]:
        recall_projection = self._recall_filter_projector.project(
            user_question=task.user_question,
            strict_filters={},
        )
        filters = dict(recall_projection.filters)
        need_case_context = (
            "case_context.query" in task.allowed_tools
            and not filters.get("jurisdiction")
        )
        warnings = [
            "strict_filter_generation_failed",
            "recall_only_fallback_enabled",
        ]
        if validation_errors:
            warnings.append(f"strict_filter_validation_errors:{len(validation_errors)}")
        return {
            "policy_question": str(task.user_question)[:1200],
            "answer_mode": _infer_answer_mode_from_question(
                task.user_question,
                _infer_information_needs_from_question(task.user_question, filters),
            ),
            "information_needs": _infer_information_needs_from_question(
                task.user_question,
                filters,
            ),
            "filters": filters,
            "allow_broad_filters": True,
            "top_k": POLICY_RAG_TOP_K,
            "fetch_k": 40,
            "rerank": True,
            "need_case_context": need_case_context,
            "case_context_requests": [
                {
                    "capability": "query_case_basic_info",
                    "arguments": {"case_id": task.case_id, "limit": 30},
                }
            ]
            if need_case_context
            else [],
            "reason": reason,
            "filter_source": "llm_fallback_recall_only",
            "filter_confidence": 0.0,
            "filter_warnings": list(dict.fromkeys(warnings)),
            "_filter_generation": {
                "source": "recall_only_fallback",
                "repair_count": repair_count,
                "valid_on_first_call": False,
            },
            "filter_diagnostics": {
                "fallback_mode": "recall_only",
                "allow_broad_filters": True,
                "strict_filter_generation_errors": list(validation_errors or []),
                "strict_filters": {},
                "recall_filters": filters,
                "recall_filter_reasons": list(recall_projection.reasons),
                "explicit_jurisdictions": list(
                    recall_projection.explicit_jurisdictions
                ),
            },
        }

    def _normalize_retrieval_plan(
        self,
        task: ExpertAnalysisTask,
        payload: dict[str, Any],
        *,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposed_filters = dict(payload.get("filters") or {})
        allow_broad_filters = _bool(payload.get("allow_broad_filters"), False)
        if _has_forced_policy_filters(task.filters):
            resolution = self._resolve_policy_filters(
                task,
                task_filters=task.filters,
            )
        else:
            resolution = self._filter_resolver.resolve_validated_filters(
                proposed_filters,
                policy_question=str(
                    payload.get("policy_question") or task.user_question
                ),
                confidence=_float_between(
                    payload.get("filter_confidence"),
                    0.0,
                    1.0,
                    0.8,
                ),
                filter_source=str(
                    payload.get("filter_source") or "llm_registered"
                ),
                allow_broad=allow_broad_filters,
            )
        question = str(
            payload.get("policy_question")
            or payload.get("question")
            or resolution.policy_question
            or task.user_question
        )[:1200]
        filters = resolution.filters
        generation = payload.get("_filter_generation")
        generation = generation if isinstance(generation, dict) else {}
        filter_diagnostics = (
            dict(payload.get("filter_diagnostics"))
            if isinstance(payload.get("filter_diagnostics"), dict)
            else {}
        )
        filter_diagnostics.update({
            "task_filters": normalize_policy_filters(task.filters),
            "llm_planner_filters": filters
            if _has_forced_policy_filters(task.filters)
            else proposed_filters,
            "resolver_input_filters": proposed_filters,
            "resolver_filters": filters,
            "filter_source": resolution.filter_source,
            "filter_confidence": resolution.confidence,
            "filter_warnings": list(resolution.warnings),
            "allow_broad_filters": allow_broad_filters,
            "filter_reason": str(
                payload.get("filter_reason") or payload.get("reason") or ""
            )[:500],
            "valid_on_first_call": bool(
                generation.get("valid_on_first_call", True)
            ),
            "repair_count": int(generation.get("repair_count") or 0),
            "resolver_added_values": {},
        })
        recall_projection = self._recall_filter_projector.project(
            user_question=task.user_question,
            strict_filters=filters,
        )
        filter_diagnostics.update(
            {
                "strict_filters": filters,
                "recall_filters": recall_projection.filters,
                "recall_filter_reasons": list(recall_projection.reasons),
                "explicit_jurisdictions": list(
                    recall_projection.explicit_jurisdictions
                ),
                "allow_broad_filters": allow_broad_filters,
            }
        )
        requests = payload.get("case_context_requests")
        if not isinstance(requests, list):
            requests = []
        payload_filter_warnings = payload.get("filter_warnings")
        filter_warnings = []
        if isinstance(payload_filter_warnings, list):
            filter_warnings.extend(
                str(item) for item in payload_filter_warnings if str(item).strip()
            )
        filter_warnings.extend(
            str(item) for item in resolution.warnings if str(item).strip()
        )
        filter_warnings = list(dict.fromkeys(filter_warnings))
        if payload.get("top_k") not in (None, ""):
            requested_top_k = _int_between(
                payload.get("top_k"),
                1,
                20,
                POLICY_RAG_TOP_K,
            )
            if requested_top_k != POLICY_RAG_TOP_K:
                filter_warnings.append("top_k_replaced_by_adaptive_context_policy")
        information_needs = _normalize_information_needs(
            payload.get("information_needs")
        )
        if not information_needs:
            information_needs = _infer_information_needs_from_question(
                task.user_question,
                filters,
            )
        answer_mode = _normalize_answer_mode(
            payload.get("answer_mode"),
            task.user_question,
            information_needs,
        )
        retrieval_budget = _adaptive_policy_retrieval_budget(
            question=task.user_question,
            filters=filters,
            answer_mode=answer_mode,
            information_needs=information_needs,
        )
        requested_fetch_k = _int_between(
            payload.get("fetch_k"),
            1,
            200,
            POLICY_RAG_SIMPLE_FETCH_K,
        )
        if requested_fetch_k != retrieval_budget["fetch_k"]:
            filter_warnings.append("fetch_k_replaced_by_adaptive_policy")
        filter_diagnostics.update(
            {
                "requested_fetch_k": requested_fetch_k,
                "requested_rerank": _bool(payload.get("rerank"), False),
                "retrieval_complexity": retrieval_budget["complexity"],
                "adaptive_fetch_k": retrieval_budget["fetch_k"],
                "adaptive_rerank": retrieval_budget["rerank"],
                "rerank_candidate_limit": retrieval_budget[
                    "rerank_candidate_limit"
                ],
            }
        )
        need_case_context = bool(payload.get("need_case_context", False))
        if need_case_context and not requests:
            requests = [
                {
                    "capability": "query_case_basic_info",
                    "arguments": {"case_id": task.case_id, "limit": 30},
                }
            ]
        return {
            "policy_question": question,
            "filters": filters,
            "recall_filters": recall_projection.filters,
            "filter_strategy": "dual_rrf",
            "adaptive_top_k": True,
            "allow_broad_filters": bool(
                payload.get("allow_broad_filters", False)
            ),
            "top_k": POLICY_RAG_TOP_K,
            "fetch_k": retrieval_budget["fetch_k"],
            "rerank": retrieval_budget["rerank"],
            "retrieval_complexity": retrieval_budget["complexity"],
            "rerank_candidate_limit": retrieval_budget[
                "rerank_candidate_limit"
            ],
            "answer_mode": answer_mode,
            "information_needs": information_needs,
            "need_case_context": need_case_context,
            "case_context_requests": [
                item for item in requests[:3]
                if isinstance(item, dict)
            ],
            "reason": str(payload.get("reason") or "llm_registered_policy_plan")[:300],
            "filter_source": resolution.filter_source,
            "filter_confidence": resolution.confidence,
            "filter_warnings": list(dict.fromkeys(filter_warnings)),
            "filter_diagnostics": filter_diagnostics,
        }

    def _normalize_policy_filters(self, value: Any) -> dict[str, Any]:
        filters = normalize_policy_filters(value)
        if not filters:
            return {}
        if "can_cite_as_policy_basis" not in filters:
            filters["can_cite_as_policy_basis"] = True
        return filters

    def _infer_filters_from_text(self, text: str) -> dict[str, Any]:
        filters: dict[str, Any] = {"can_cite_as_policy_basis": True}
        jurisdictions: list[str] = []
        if "北京" in text:
            jurisdictions.append("beijing")
        if "上海" in text:
            jurisdictions.append("shanghai")
        if any(token in text for token in ("国家", "全国", "跨省", "异地")):
            jurisdictions.append("national")
        domains: list[str] = []
        if _looks_like_drug_price_reference_query(text):
            filters["can_cite_as_policy_basis"] = False
            jurisdictions = ["shanghai"]
            domains.append("drug_product_price_reference")
            filters["content_type"] = ["table_row"]
            filters["jurisdiction"] = jurisdictions
            filters["policy_domain"] = domains
            return filters
        if _looks_like_remote_benefit_split_query(text):
            if "national" not in jurisdictions:
                jurisdictions.insert(0, "national")
            filters["jurisdiction"] = list(dict.fromkeys(jurisdictions))
            filters["policy_domain"] = ["remote_medical"]
            filters["content_type"] = ["policy_text"]
            return filters
        if _looks_like_shanghai_payment_scope_verification_query(text):
            filters["jurisdiction"] = ["shanghai"]
            filters["policy_domain"] = [
                "drug_catalog",
                "medical_service_price",
                "shanghai_payment_scope",
            ]
            filters["content_type"] = ["table_row", "policy_text"]
            return filters
        if _looks_like_manual_reimbursement_service_query(text):
            filters["policy_domain"] = ["manual_reimbursement"]
            filters["content_type"] = ["policy_text"]
            if jurisdictions:
                filters["jurisdiction"] = list(dict.fromkeys(jurisdictions))
            return normalize_policy_filters(filters)
        special_disease_query = _looks_like_special_disease_query(text)
        if special_disease_query:
            domains.extend(["special_disease_filing", "special_disease_scope"])
            if any(token in text for token in ("长期处方", "长处方", "续方")):
                domains.append("chronic_disease_long_prescription")
        if any(token in text for token in ("异地", "跨省")) or (
            "备案" in text and not special_disease_query
        ):
            domains.extend(["remote_medical", "remote_medical_manual_reimbursement"])
        if any(token in text for token in ("手工报销", "零星报销", "报销")):
            domains.append("manual_reimbursement")
        if any(token in text for token in ("急诊", "门急诊")):
            domains.append("emergency")
        if _looks_like_consumable_payment_scope_query(text):
            domains.append("shanghai_payment_scope")
        if _looks_like_drug_catalog_query(text):
            domains.append("drug_catalog")
        if _looks_like_medical_service_price_query(text):
            domains.append("medical_service_price")
        if jurisdictions:
            filters["jurisdiction"] = list(dict.fromkeys(jurisdictions))
        if domains:
            filters["policy_domain"] = list(dict.fromkeys(domains))
        return normalize_policy_filters(filters)

    def _deterministic_answerability_from_evidence(
        self,
        state: dict[str, Any],
        *,
        evidence: list[dict[str, Any]],
    ) -> AnswerabilityCheck | None:
        if not evidence:
            return None
        task = self._task(state)
        question = task.user_question
        question_compact = str(question or "").replace(" ", "")
        if not any(
            token in question_compact
            for token in ("就医地", "参保地", "目录", "待遇", "支付范围", "分工", "跨省异地")
        ):
            return None

        usable_refs: list[str] = []
        for item in evidence:
            text = " ".join(
                str(item.get(key) or "")
                for key in ("title", "excerpt", "policy_domain", "content_type", "jurisdiction")
            )
            if _contains_remote_benefit_split(text):
                ref = str(item.get("evidence_ref") or item.get("source_ref") or "")
                if ref:
                    usable_refs.append(ref)
        if not usable_refs:
            return None
        return AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["就医地支付范围", "参保地待遇政策"],
            missing_slots=[],
            usable_evidence_refs=list(dict.fromkeys(usable_refs))[:20],
            next_action="synthesize",
            reason_code="deterministic_remote_benefit_split",
            reason=(
                "已检索到可引用政策证据，覆盖就医地支付范围与参保地待遇政策分工。"
            ),
        )

    def _enrich_plan_with_case_context(
        self,
        task: ExpertAnalysisTask,
        plan: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Attach context diagnostics without mutating the LLM filter boundary."""

        enriched = dict(plan)
        filters = dict(enriched.get("filters") or {})
        resolution = self._filter_resolver.resolve_validated_filters(
            filters,
            policy_question=str(
                enriched.get("policy_question") or task.user_question
            ),
            confidence=_float_between(
                enriched.get("filter_confidence"),
                0.0,
                1.0,
                0.8,
            ),
            filter_source=str(enriched.get("filter_source") or "llm_registered"),
            allow_broad=_bool(enriched.get("allow_broad_filters"), False),
        )
        enriched["filters"] = resolution.filters
        enriched["need_case_context"] = False
        diagnostics = enriched.get("filter_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics["case_context_observation_count"] = len(observations)
        diagnostics["case_context_filter_mutation"] = False
        enriched["filter_diagnostics"] = diagnostics
        return enriched

    def _policy_request_from_plan(
        self,
        task: ExpertAnalysisTask,
        plan: dict[str, Any],
    ) -> PolicySearchRequest:
        filters = self._normalize_policy_filters(plan.get("filters"))
        request = PolicySearchRequest(
            question=str(plan.get("policy_question") or task.user_question)[:1200],
            filters=filters,
            recall_filters=self._normalize_recall_filters(
                plan.get("recall_filters")
            ),
            filter_strategy=str(plan.get("filter_strategy") or "dual_rrf"),
            adaptive_top_k=_bool(plan.get("adaptive_top_k"), True),
            allow_broad_filters=_bool(plan.get("allow_broad_filters"), False),
            top_k=POLICY_RAG_TOP_K,
            fetch_k=_int_between(plan.get("fetch_k"), 1, 200, 40),
            rerank=_bool(plan.get("rerank"), True),
        )
        diagnostics = plan.get("filter_diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["retrieval_request_filters"] = request.filters
            diagnostics["retrieval_request_recall_filters"] = request.recall_filters
            diagnostics["retrieval_filter_strategy"] = request.filter_strategy
            diagnostics["retrieval_request_question"] = request.question
        return request

    @staticmethod
    def _normalize_recall_filters(value: Any) -> dict[str, Any]:
        payload = dict(value) if isinstance(value, dict) else {}
        return {
            "jurisdiction": _string_list(payload.get("jurisdiction")),
            "policy_domain": [],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": (
                payload.get("can_cite_as_policy_basis")
                if isinstance(payload.get("can_cite_as_policy_basis"), bool)
                else None
            ),
        }

    def _policy_evidence_from_candidate(
        self,
        item: dict[str, Any],
        rank: int,
    ) -> PolicyEvidence | None:
        normalized_filters = normalize_policy_filters(
            {
                "jurisdiction": item.get("jurisdiction"),
                "policy_domain": item.get("policy_domain"),
                "content_type": item.get("content_type"),
                "can_cite_as_policy_basis": item.get("can_cite_as_policy_basis"),
            }
        )
        policy_domains = _string_list(normalized_filters.get("policy_domain"))
        content_types = _string_list(normalized_filters.get("content_type"))
        jurisdictions = _string_list(normalized_filters.get("jurisdiction"))
        raw_policy_domain = str(item.get("policy_domain") or "").strip()
        raw_content_type = str(item.get("content_type") or "").strip()
        policy_domain = policy_domains[0] if policy_domains else ""
        content_type = content_types[0] if content_types else ""
        if raw_policy_domain and not policy_domain:
            return None
        if raw_content_type and not content_type:
            return None
        can_cite_as_policy_basis = _bool(item.get("can_cite_as_policy_basis"), True)
        is_drug_price_reference = policy_domain == "drug_product_price_reference"
        if not can_cite_as_policy_basis and not is_drug_price_reference:
            return None
        node_id = str(item.get("node_id") or f"candidate-{rank}")
        source_id = str(item.get("source_id") or item.get("doc_id") or node_id)
        stable = _stable_ref({"node_id": node_id, "source_id": source_id, "rank": rank})
        source_ref = f"policy-evidence:{stable}"
        excerpt = _clip(str(item.get("text") or item.get("excerpt") or ""), 1000)
        if not excerpt.strip():
            return None
        title = str(item.get("title") or source_id or "政策证据")[:240]
        try:
            return PolicyEvidence(
                evidence_ref=source_ref,
                source_ref=source_ref,
                title=title,
                excerpt=excerpt,
                jurisdiction=_optional_str(jurisdictions[0] if jurisdictions else None, 120),
                policy_domain=_optional_str(policy_domain or None, 120),
                content_type=_optional_str(content_type or None, 80),
                source_url=_optional_str(item.get("source_url"), 500),
                version=_optional_str(item.get("version") or item.get("doc_id"), 120),
                used_for=(
                    "drug_price_reference"
                    if is_drug_price_reference
                    else "policy_answer"
                ),
                metadata={
                    "rank": item.get("rank") or rank,
                    "score": item.get("score"),
                    "score_type": item.get("score_type"),
                    "node_id": node_id,
                    "source_id": source_id,
                    "section_heading": item.get("section_heading"),
                    "retrieval_groups": item.get("retrieval_groups"),
                    "can_cite_as_policy_basis": can_cite_as_policy_basis,
                },
            )
        except ValidationError:
            return None

    def _case_facts_from_task_and_observations(
        self,
        task: ExpertAnalysisTask,
        observations: list[dict[str, Any]],
    ) -> list[CaseFactUsed]:
        facts: list[CaseFactUsed] = []
        for item in task.fact_bundle[:16]:
            label = str(item.get("label") or item.get("field") or item.get("key") or "")
            value = item.get("value")
            if label and value not in (None, "", [], {}):
                facts.append(
                    CaseFactUsed(
                        label=label[:120],
                        value=value,
                        source_refs=[
                            str(ref) for ref in item.get("source_refs", [])
                            if ref
                        ][:8],
                    )
                )
        for observation in observations:
            if observation.get("status") != "success":
                continue
            payload = observation.get("payload")
            if not isinstance(payload, dict):
                continue
            inner = payload.get("payload")
            if not isinstance(inner, dict):
                inner = payload
            context = inner.get("case_context")
            if isinstance(context, dict):
                inner = {**context, **inner}
            source_refs = [
                str(ref) for ref in observation.get("source_refs", [])
                if ref
            ][:8]
            for key in (
                "insured_region",
                "treatment_region",
                "visit_type",
                "claim_mode",
                "filing_status",
                "emergency_material_status",
                "diagnosis",
            ):
                value = inner.get(key)
                if value not in (None, "", [], {}):
                    facts.append(
                        CaseFactUsed(
                            label=_case_fact_label(key),
                            value=value,
                            source_refs=source_refs,
                        )
                    )
        deduped: dict[str, CaseFactUsed] = {}
        for fact in facts:
            deduped[f"{fact.label}:{fact.value}"] = fact
        return list(deduped.values())[:20]

    def _deterministic_answerability(
        self,
        state: dict[str, Any],
        *,
        has_evidence: bool,
    ) -> AnswerabilityCheck:
        if has_evidence:
            refs = [
                str(item.get("evidence_ref"))
                for item in state.get("adopted_policy_evidence", [])
                if isinstance(item, dict) and item.get("evidence_ref")
            ][:20]
            return AnswerabilityCheck(
                answerability="answerable",
                covered_slots=["policy_basis"],
                missing_slots=[],
                usable_evidence_refs=refs,
                next_action="synthesize",
                reason_code="citable_evidence_found",
                reason="Policy RAG returned citable evidence.",
            )
        task = self._task(state)
        can_rewrite = (
            state.get("rewrite_count", 0) < task.budget.max_llm_rewrite_calls
            and state.get("retrieval_round_count", 0) < task.budget.max_retrieval_rounds
        )
        return AnswerabilityCheck(
            answerability="insufficient",
            covered_slots=[],
            missing_slots=["policy_evidence"],
            usable_evidence_refs=[],
            next_action="rewrite" if can_rewrite else "insufficient",
            reason_code="no_citable_evidence",
            reason="No citable policy evidence is available.",
        )

    def _build_question_slots(
        self,
        task: ExpertAnalysisTask,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        question = task.user_question
        filters = normalize_policy_filters(
            (state.get("retrieval_plan") or {}).get("filters") or task.filters
        )
        information_needs = _normalize_information_needs(
            (state.get("retrieval_plan") or {}).get("information_needs")
        )
        if not information_needs:
            information_needs = _infer_information_needs_from_question(question, filters)
        answer_mode = _normalize_answer_mode(
            (state.get("retrieval_plan") or {}).get("answer_mode"),
            question,
            information_needs,
        )
        domains = set(_string_list(filters.get("policy_domain")))
        slots: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            slot_id: str,
            label: str,
            *,
            critical: bool = True,
            weight: float = 1.0,
            keywords: list[str] | None = None,
            policy_domains: list[str] | None = None,
            scenario_id: str | None = None,
            required_fields: list[str] | None = None,
            answer_action: str | None = None,
        ) -> None:
            if slot_id in seen:
                return
            seen.add(slot_id)
            slots.append(
                {
                    "slot_id": slot_id,
                    "label": label,
                    "need_id": slot_id,
                    "need_text": label,
                    "answer_mode": answer_mode,
                    "critical": critical,
                    "weight": weight,
                    "keywords": keywords or [],
                    "policy_domains": policy_domains or [],
                    "scenario_id": (
                        scenario_id if scenario_id is not None else slot_scenario_id(slot_id)
                    ),
                    "required_fields": (
                        required_fields
                        if required_fields is not None
                        else list(slot_required_fields(slot_id))
                    ),
                    "answer_action": answer_action or slot_answer_action(slot_id),
                }
            )

        def add_registered(
            slot_id: str,
            *,
            critical: bool = True,
            weight: float = 1.0,
        ) -> None:
            normalized = normalize_answer_slot_ids([slot_id])
            if not normalized:
                return
            resolved = normalized[0]
            add(
                resolved,
                slot_label(resolved),
                critical=critical,
                weight=weight,
                keywords=list(slot_coverage_terms(resolved)),
                policy_domains=list(slot_policy_domains(resolved)),
                scenario_id=slot_scenario_id(resolved),
                required_fields=_dynamic_slot_required_fields(resolved, question),
                answer_action=slot_answer_action(resolved),
            )

        for slot_id in _information_needs_to_legacy_slots(question, information_needs, filters):
            add_registered(slot_id)

        scenario_slot_ids = _manual_reimbursement_scenario_slot_ids(question)
        for slot_id in scenario_slot_ids:
            add_registered(slot_id, weight=1.2)
        policy_scenario_slot_ids = _policy_scenario_slot_ids(question)
        for slot_id in policy_scenario_slot_ids:
            add_registered(slot_id, weight=1.2)
        has_specific_scenario_slots = bool(scenario_slot_ids or policy_scenario_slot_ids)

        if _looks_like_remote_benefit_split_query(question) or _looks_like_remote_direct_settlement_payment_query(question):
            add(
                "remote_benefit_split",
                "就医地目录与参保地待遇分工",
                critical=True,
                weight=1.2,
                keywords=[
                    "住院",
                    "普通门诊",
                    "门诊慢特病",
                    "就医地",
                    "参保地",
                    "支付范围",
                    "起付",
                    "支付比例",
                    "最高支付限额",
                    "病种范围",
                    "待遇",
                ],
                policy_domains=["remote_medical", "benefit"],
                required_fields=_dynamic_slot_required_fields("remote_benefit_split", question),
            )
        if _looks_like_remote_filing_institution_scope_query(question):
            add_registered("remote_filing_institution_scope", weight=1.15)
        if _looks_like_special_disease_query(question) or domains.intersection(
            {"special_disease_filing", "special_disease_scope"}
        ):
            if "备案" in question or "special_disease_filing" in domains:
                add_registered("special_disease_filing_policy", weight=1.1)
            if any(
                token in question
                for token in ("限定支付条件", "方可支付", "类风湿关节炎", "DMARDs", "风湿病专科医师")
            ):
                add_registered("special_disease_payment_condition", weight=1.2)
            if (
                "special_disease_scope" in domains
                or any(
                    token in question
                    for token in ("特殊疾病范围", "新增病种", "重性精神病", "肺动脉高压", "未备案", "报销范围")
                )
            ):
                add_registered("special_disease_scope_policy", weight=1.1)
        if any(
            token in question
            for token in ("银行手续费", "银行票据", "工本费", "预付金", "黄色预警", "红色预警", "紧急调增", "费用协查")
        ):
            add_registered("remote_settlement_management", weight=1.15)
        if any(
            token in question
            for token in ("城乡居民医保", "城乡老年人", "参保范围", "参保资格", "新生儿", "等待期", "外埠户籍配偶", "家庭医生", "首诊转诊", "外省市目录")
        ) or "benefit" in domains:
            add_registered("benefit_policy", weight=1.1)
        if any(
            token in question
            for token in ("长期处方", "长处方", "慢性病", "高血压", "糖尿病", "BJ-GBI", "医事服务费", "月度通报", "品种规格", "医联体")
        ) or "chronic_disease_long_prescription" in domains:
            add_registered("chronic_long_prescription_policy", weight=1.1)
        if any(
            token in question
            for token in ("基金监管", "监督检查", "拒不配合", "暂停联网结算", "锁卡", "骗取基金", "涉嫌骗保", "不属于基金支付范围", "异常情形审核")
        ) or "fund_supervision" in domains:
            add_registered("fund_supervision_policy", weight=1.1)
        if any(
            token in question
            for token in ("医疗服务设施", "住院床位费", "急诊观察室床位费", "床位费", "实施期限", "有效期")
        ):
            add_registered("shanghai_service_facility_scope", weight=1.1)
        if any(
            token in question
            for token in ("协议期内谈判药品", "谈判药品", "双通道", "电子处方", "一品两规", "药占比", "总额限制")
        ):
            add_registered("negotiated_drug_double_channel", weight=1.1)
        if (
            not has_specific_scenario_slots
            and not (
                _looks_like_remote_benefit_split_query(question)
                or _looks_like_remote_direct_settlement_payment_query(question)
                or _looks_like_remote_filing_institution_scope_query(question)
            )
            and (
            any(token in question for token in ("异地", "跨省"))
            or domains.intersection({"remote_medical", "remote_medical_manual_reimbursement"})
            )
        ):
            add_registered("remote_settlement_management")
        if not has_specific_scenario_slots and (any(token in question for token in ("急诊", "门急诊")) or "emergency" in domains):
            add(
                "emergency_exception",
                "急诊例外",
                critical=True,
                weight=1.2,
                keywords=["急诊", "门急诊", "急诊抢救", "急诊例外", "留观"],
                policy_domains=["emergency", "remote_medical"],
            )
        if not has_specific_scenario_slots and "备案" in question and not _looks_like_special_disease_query(question):
            add(
                "filing_rule",
                "备案规则",
                critical=True,
                weight=1.0,
                keywords=["备案", "异地备案", "补备案", "备案状态"],
                policy_domains=["remote_medical", "remote_medical_manual_reimbursement"],
            )
        if any(
            token in question
            for token in (
                "必要材料",
                "申请材料",
                "提交哪些材料",
                "需要哪些材料",
                "核验哪些材料",
                "应核验哪些材料",
                "需要核验哪些材料",
                "补充哪些材料",
                "报销材料",
                "材料目录",
            )
        ) and not scenario_slot_ids:
            add(
                "required_materials",
                "必要材料",
                critical=True,
                weight=1.2,
                keywords=["材料名称", "材料必要性", "申请材料", "必要", "处方", "费用清单", "收据"],
                policy_domains=["manual_reimbursement", "remote_medical_manual_reimbursement"],
            )
        if any(
            token in question
            for token in ("法定办结时限", "承诺办结时限", "办结时限", "办理时限", "多少工作日", "多久办结")
        ):
            add(
                "statutory_processing_time",
                "法定办结时限",
                critical=True,
                weight=1.1,
                keywords=["法定办结时限", "承诺办结时限", "办理时限", "工作日", "审查", "决定"],
                policy_domains=["manual_reimbursement"],
            )
        if (
            any(token in question for token in ("手工报销", "零星报销", "报销材料"))
            or "manual_reimbursement" in domains
        ) and not scenario_slot_ids:
            add(
                "manual_reimbursement",
                "手工报销",
                critical=True,
                weight=1.0,
                keywords=["手工报销", "零星报销", "报销材料", "经办", "办理", "材料"],
                policy_domains=["manual_reimbursement", "remote_medical_manual_reimbursement"],
            )
        if (
            _looks_like_drug_catalog_query(question)
            or (
                "drug_catalog" in domains
                and not any(token in question for token in ("双通道", "谈判药品", "电子处方", "一品两规"))
            )
        ):
            add(
                "drug_catalog",
                "医保药品目录",
                critical=True,
                weight=1.0,
                keywords=["药品", "药品目录", "医保目录", "医保类别", "目录编号", "限定支付范围"],
                policy_domains=["drug_catalog"],
            )
        if _looks_like_medical_service_price_query(question) or "medical_service_price" in domains:
            add(
                "medical_service_price",
                "医疗服务项目价格",
                critical=True,
                weight=1.0,
                keywords=["胸部", "CT", "医疗服务项目", "医疗服务价格", "诊疗项目", "检查", "影像", "支付范围", "价格"],
                policy_domains=["medical_service_price"],
            )
        if any(token in question for token in ("起付线", "起付标准", "支付比例", "报销比例", "封顶线", "最高支付限额")) or (
            "退休" in question and "待遇" in question
        ):
            add(
                "benefit_params",
                "待遇参数",
                critical=True,
                weight=1.0,
                keywords=["起付标准", "起付线", "支付比例", "报销比例", "最高支付限额", "封顶线", "退休"],
                policy_domains=["benefit", "remote_medical"],
            )
        if _looks_like_consumable_payment_scope_query(question):
            add(
                "consumable_payment_scope",
                "医用耗材支付范围",
                critical=True,
                weight=1.0,
                keywords=["医用耗材", "耗材", "支付范围", "支付办法", "甲类", "乙类", "先自负"],
                policy_domains=["shanghai_payment_scope"],
            )
        if any(token in question for token in ("定点机构", "定点医院", "定点药店", "机构编码", "药店编码", "定点状态")) or (
            "designated_institution" in domains
        ):
            add(
                "designated_institution",
                "定点机构或药店状态",
                critical=True,
                weight=1.0,
                keywords=["机构名称", "药店名称", "编码", "地址", "状态", "定点"],
                policy_domains=["designated_institution"],
            )
        if "drug_product_price_reference" in domains:
            add(
                "drug_price_reference",
                "药品价格参考",
                critical=True,
                weight=1.0,
                keywords=["价格", "均价", "周均价", "价格区间", "药品名称"],
                policy_domains=["drug_product_price_reference"],
            )
        if not slots:
            add(
                "policy_basis",
                "政策依据",
                critical=True,
                weight=1.0,
                keywords=[],
                policy_domains=list(domains),
            )
        return slots[:8]

    def _slot_coverage_gate(
        self,
        task: ExpertAnalysisTask,
        state: dict[str, Any],
        evidence: list[dict[str, Any]],
        slots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        filters = normalize_policy_filters(
            (state.get("retrieval_plan") or {}).get("filters") or task.filters
        )
        requested_jurisdictions = set(_string_list(filters.get("jurisdiction")))
        compatible_evidence: list[dict[str, Any]] = []
        wrong_jurisdiction_refs: list[str] = []
        for item in evidence:
            jurisdiction = str(item.get("jurisdiction") or "")
            if _jurisdiction_compatible(requested_jurisdictions, jurisdiction):
                compatible_evidence.append(item)
            elif item.get("evidence_ref"):
                wrong_jurisdiction_refs.append(str(item.get("evidence_ref")))

        slot_status: dict[str, dict[str, Any]] = {}
        field_coverage_matrix: dict[str, dict[str, Any]] = {}
        claim_plan: list[dict[str, Any]] = []
        supported_refs: list[str] = []
        weighted_score = 0.0
        total_weight = 0.0
        fact_items = [
            item for item in state.get("verified_slot_facts", [])
            if isinstance(item, dict)
        ]
        requirements_by_slot = _requirements_by_slot(state)
        use_fact_coverage = bool(fact_items)
        for slot in slots:
            slot_id = str(slot.get("slot_id") or "")
            requirement_id = str(slot.get("requirement_id") or slot.get("need_id") or "")
            slot_key = requirement_id or slot_id
            weight = float(slot.get("weight") or 1.0)
            total_weight += weight
            requirement = (
                requirements_by_slot.get(slot_key)
                or requirements_by_slot.get(slot_id)
                or {}
            )
            required_fields = (
                _string_list(slot.get("required_fields"))
                or _string_list(requirement.get("required_fields"))
                or list(slot_required_fields(slot_id))
            )
            if use_fact_coverage:
                matches = _slot_fact_matches(slot, fact_items)
            else:
                matches = _slot_evidence_matches(slot, compatible_evidence)
            strong_refs = [
                match["evidence_ref"] for match in matches
                if match["status"] == "strong"
            ]
            weak_refs = [
                match["evidence_ref"] for match in matches
                if match["status"] == "weak"
            ]
            fact_refs = [
                match.get("fact_ref") for match in matches
                if match.get("fact_ref")
            ]
            direct_fact_refs = [
                match.get("fact_ref") for match in matches
                if match.get("fact_ref") and match.get("support_role") == "direct_answer"
            ]
            background_fact_refs = [
                match.get("fact_ref") for match in matches
                if match.get("fact_ref") and match.get("support_role") != "direct_answer"
            ]
            direct_fields: list[str] = []
            partial_fields: list[str] = []
            for match in matches:
                if match.get("status") == "strong":
                    direct_fields.extend(_string_list(match.get("matched_fields")))
                elif match.get("status") == "weak":
                    partial_fields.extend(_string_list(match.get("matched_fields")))
            direct_fields = list(dict.fromkeys(direct_fields))
            partial_fields = [
                field for field in list(dict.fromkeys(partial_fields))
                if field not in direct_fields
            ]
            fields_not_direct = [
                field for field in required_fields
                if field not in direct_fields
            ]
            missing_fields = [
                field for field in required_fields
                if field not in direct_fields and field not in partial_fields
            ]
            if required_fields and not use_fact_coverage:
                direct_fields = required_fields if strong_refs else []
                partial_fields = required_fields if weak_refs and not strong_refs else []
                fields_not_direct = [
                    field for field in required_fields
                    if field not in direct_fields
                ]
                missing_fields = [
                    field for field in required_fields
                    if field not in direct_fields and field not in partial_fields
                ]

            if required_fields and set(required_fields).issubset(set(direct_fields)):
                status = "strong"
                coverage_status = "covered"
                refs = strong_refs[:8] or weak_refs[:8]
                weighted_score += weight
            elif required_fields and (direct_fields or partial_fields or weak_refs):
                status = "weak"
                coverage_status = "partial"
                refs = list(dict.fromkeys([*strong_refs, *weak_refs]))[:8]
                weighted_score += weight * 0.5
            elif not required_fields and strong_refs:
                status = "strong"
                coverage_status = "covered"
                refs = strong_refs[:8]
                weighted_score += weight
            elif not required_fields and weak_refs:
                status = "weak"
                coverage_status = "partial"
                refs = weak_refs[:8]
                weighted_score += weight * 0.5
            else:
                status = "missing"
                coverage_status = "missing"
                refs = []
            supported_refs.extend(refs)
            generation = (
                "allow"
                if status == "strong"
                else "allow_with_qualification"
                if status == "weak"
                else "deny"
            )
            slot_status[slot_key] = {
                "label": slot.get("label") or requirement.get("label") or slot_id or slot_key,
                "requirement_id": requirement_id or slot_key,
                "slot_id": slot_id,
                "status": status,
                "coverage_status": coverage_status,
                "critical": bool(slot.get("critical", True)),
                "weight": weight,
                "evidence_refs": refs,
                "fact_refs": list(dict.fromkeys(fact_refs))[:8],
                "direct_fact_refs": list(dict.fromkeys(direct_fact_refs))[:8],
                "background_fact_refs": list(dict.fromkeys(background_fact_refs))[:8],
                "scenario_id": slot.get("scenario_id") or requirement.get("scenario_id") or slot_scenario_id(slot_id),
                "required_fields": required_fields[:24],
                "covered_fields": direct_fields[:24],
                "partial_fields": partial_fields[:24],
                "fields_not_direct": fields_not_direct[:24],
                "missing_fields": missing_fields[:24],
                "answer_action": slot.get("answer_action") or requirement.get("answer_action") or slot_answer_action(slot_id),
                "generation": generation,
                "coverage_basis": "slot_facts" if use_fact_coverage else "chunk_keywords",
            }
            field_coverage_matrix[slot_key] = {
                "slot_id": slot_id,
                "requirement_id": requirement_id or slot_key,
                "label": slot.get("label") or requirement.get("label") or slot_id or slot_key,
                "scenario_id": slot.get("scenario_id") or requirement.get("scenario_id") or slot_scenario_id(slot_id),
                "required_fields": required_fields[:24],
                "covered_fields": direct_fields[:24],
                "partial_fields": partial_fields[:24],
                "missing_fields": missing_fields[:24],
                "evidence_refs": refs,
                "fact_refs": list(dict.fromkeys(fact_refs))[:8],
                "coverage_status": coverage_status,
            }
            claim_plan.append(
                {
                    "slot_id": slot_id,
                    "requirement_id": requirement_id or str(requirement.get("requirement_id") or slot_key),
                    "label": slot.get("label") or requirement.get("label") or slot_id or slot_key,
                    "scenario_id": slot.get("scenario_id") or requirement.get("scenario_id") or slot_scenario_id(slot_id),
                    "answer_action": slot.get("answer_action") or requirement.get("answer_action") or slot_answer_action(slot_id),
                    "required_fields": required_fields[:24],
                    "covered_fields": direct_fields[:24],
                    "partial_fields": partial_fields[:24],
                    "missing_fields": missing_fields[:24],
                    "fact_refs": list(dict.fromkeys(fact_refs))[:8],
                    "direct_fact_refs": list(dict.fromkeys(direct_fact_refs))[:8],
                    "background_fact_refs": list(dict.fromkeys(background_fact_refs))[:8],
                    "evidence_refs": refs,
                    "generation": generation,
                    "support_status": coverage_status,
                }
            )

        supported_slot_count = sum(
            1 for item in slot_status.values()
            if item.get("status") in {"strong", "weak"}
        )
        covered_need_count = sum(
            1 for item in slot_status.values()
            if item.get("status") == "strong"
        )
        partial_need_count = sum(
            1 for item in slot_status.values()
            if item.get("status") == "weak"
        )
        critical_slots = [
            item for item in slot_status.values()
            if bool(item.get("critical", True))
        ]
        critical_slots_all_strong = bool(critical_slots) and all(
            item.get("status") == "strong" for item in critical_slots
        )
        missing_slots: list[str] = []
        for slot_id, item in slot_status.items():
            label = str(item.get("label") or slot_id)
            missing_fields = _string_list(item.get("missing_fields"))
            if item.get("status") == "missing":
                if missing_fields:
                    missing_slots.append(f"{label}缺少{_join_cn(missing_fields[:4])}")
                else:
                    missing_slots.append(label)
            elif missing_fields:
                missing_slots.append(f"{label}缺少{_join_cn(missing_fields[:4])}")
        confidence_score = weighted_score / total_weight if total_weight else 0.0
        coverage_ratio = (
            (covered_need_count + 0.5 * partial_need_count) / len(slot_status)
            if slot_status
            else 0.0
        )
        wrong_jurisdiction = bool(evidence) and not compatible_evidence and bool(
            requested_jurisdictions
        )
        return {
            "coverage_score": round(confidence_score, 3),
            "coverage_ratio": round(coverage_ratio, 3),
            "confidence_score": round(confidence_score, 3),
            "covered_need_count": covered_need_count,
            "partial_need_count": partial_need_count,
            "slot_status": slot_status,
            "field_coverage_matrix": field_coverage_matrix,
            "claim_plan": claim_plan[:30],
            "supported_slot_count": supported_slot_count,
            "critical_slots_all_strong": critical_slots_all_strong,
            "missing_slots": missing_slots,
            "usable_evidence_refs": list(dict.fromkeys(supported_refs))[:20],
            "citable_evidence_count": len(evidence),
            "compatible_evidence_count": len(compatible_evidence),
            "wrong_jurisdiction": wrong_jurisdiction,
            "wrong_jurisdiction_refs": wrong_jurisdiction_refs[:20],
            "verified_fact_count": len(fact_items),
            "coverage_basis": "slot_facts" if use_fact_coverage else "chunk_keywords",
            "decision_reason": "slot_fact_coverage_gate" if use_fact_coverage else "slot_coverage_gate",
        }

    def _answerability_decision(
        self,
        state: dict[str, Any],
        *,
        evidence: list[dict[str, Any]],
        coverage_result: dict[str, Any],
        semantic_coverage: dict[str, Any],
    ) -> AnswerabilityCheck:
        task = self._task(state)
        can_rewrite = (
            state.get("rewrite_count", 0) < task.budget.max_llm_rewrite_calls
            and state.get("retrieval_round_count", 0) < task.budget.max_retrieval_rounds
        )
        answer_mode = _normalize_answer_mode(
            (state.get("retrieval_plan") or {}).get("answer_mode"),
            task.user_question,
            _normalize_information_needs((state.get("retrieval_plan") or {}).get("information_needs")),
        )
        supported_labels = _coverage_labels_by_status(coverage_result, {"strong", "weak"})
        missing_labels = _coverage_labels_by_status(coverage_result, {"missing"})
        supported_refs = _string_list(coverage_result.get("usable_evidence_refs"))
        coverage_ratio = float(
            coverage_result.get("coverage_ratio")
            or coverage_result.get("coverage_score")
            or 0.0
        )
        confidence_score = float(
            coverage_result.get("confidence_score")
            or coverage_result.get("coverage_score")
            or 0.0
        )
        if int(coverage_result.get("citable_evidence_count") or 0) == 0:
            return AnswerabilityCheck(
                answerability="insufficient",
                covered_slots=[],
                missing_slots=missing_labels or ["policy_evidence"],
                usable_evidence_refs=[],
                next_action="rewrite" if can_rewrite else "insufficient",
                reason_code="no_citable_evidence",
                reason="No citable policy evidence is available.",
            )
        if coverage_result.get("wrong_jurisdiction"):
            return AnswerabilityCheck(
                answerability="insufficient",
                covered_slots=[],
                missing_slots=missing_labels or ["matching_jurisdiction_policy_evidence"],
                usable_evidence_refs=[],
                next_action="rewrite" if can_rewrite else "insufficient",
                reason_code="wrong_jurisdiction",
                reason="Citable evidence exists, but its jurisdiction does not match the requested scope.",
            )
        if int(coverage_result.get("supported_slot_count") or 0) == 0:
            return AnswerabilityCheck(
                answerability="insufficient",
                covered_slots=[],
                missing_slots=missing_labels or ["policy_question_slots"],
                usable_evidence_refs=[],
                next_action="rewrite" if can_rewrite else "insufficient",
                reason_code="no_slot_coverage",
                reason="Citable evidence did not cover any required question slot.",
            )
        semantic_missing = [
            item for item in _string_list(semantic_coverage.get("missing_slots"))
            if item not in supported_labels and item not in missing_labels
        ][:4]

        if coverage_ratio >= 0.85 and confidence_score >= 0.75:
            answerability = "answerable"
            missing = []
            if _looks_like_remote_benefit_split_query(task.user_question):
                reason_code = "deterministic_remote_benefit_split"
                reason = "Evidence covers the remote benefit split rule."
            else:
                reason_code = "coverage_threshold_met"
                reason = "Coverage ratio and confidence score satisfy the direct-answer threshold."
        else:
            answerability = "partial"
            missing = list(dict.fromkeys(missing_labels + semantic_missing))[:12]
            reason_code = "partial_need_coverage"
            reason = "At least one information need has citable evidence, but some needs are weak or missing."
            if int(coverage_result.get("supported_slot_count") or 0) == 0:
                return AnswerabilityCheck(
                    answerability=answerability,
                    covered_slots=supported_labels[:12],
                    missing_slots=missing,
                    usable_evidence_refs=supported_refs[:20],
                    next_action="rewrite" if can_rewrite else "insufficient",
                    reason_code="no_slot_coverage",
                    reason="Citable evidence exists, but it did not cover any required question slot.",
                )
        return AnswerabilityCheck(
            answerability=answerability,
            covered_slots=supported_labels[:12],
            missing_slots=missing,
            usable_evidence_refs=supported_refs[:20],
            next_action="synthesize",
            reason_code=reason_code,
            reason=reason,
        )

    def _coerce_answerability(
        self,
        payload: Any,
        fallback: dict[str, Any],
    ) -> AnswerabilityCheck:
        data = payload if isinstance(payload, dict) else fallback
        try:
            return AnswerabilityCheck.model_validate(data)
        except ValidationError:
            return AnswerabilityCheck.model_validate(fallback)

    def _answerability_model_dump(self, state: dict[str, Any]) -> dict[str, Any]:
        check = state.get("answerability_check")
        if isinstance(check, AnswerabilityCheck):
            return check.model_dump(mode="json")
        if isinstance(check, dict):
            return check
        return {}

    def _deterministic_expert_answer(
        self,
        task: ExpertAnalysisTask,
        evidence: list[PolicyEvidence],
        answerability: AnswerabilityCheck,
    ) -> str:
        if not evidence:
            return "当前没有足够可引用的政策证据回答该问题。"
        titles = "、".join(item.title for item in evidence[:3] if item.title)
        has_drug_price_reference = any(
            item.used_for == "drug_price_reference"
            or item.policy_domain == "drug_product_price_reference"
            for item in evidence
        )
        if has_drug_price_reference:
            return _format_drug_price_reference_answer(task, evidence)
        prefix = "根据已检索到的政策依据"
        if titles:
            prefix += f"（{titles}）"
        if answerability.answerability == "partial":
            covered = "、".join(answerability.covered_slots[:4]) or "部分政策依据"
            missing = "、".join(answerability.missing_slots[:4]) or "其余关键政策口径"
            return (
                f"{prefix}，当前可支持{covered}；"
                f"但{missing}证据不足，相关部分不作确定性结论。"
                f"已有依据片段：{_clip(evidence[0].excerpt, 220)}"
            )
        return (
            f"{prefix}，可围绕用户问题核验相关参考数据。"
            f"核心依据片段：{_clip(evidence[0].excerpt, 260)}"
        )

    def _coerce_audit_suggestions(
        self,
        value: Any,
        allowed_source_refs: list[str],
    ) -> list[ExpertAuditSuggestion]:
        if not isinstance(value, list):
            return []
        allowed = set(allowed_source_refs)
        suggestions: list[ExpertAuditSuggestion] = []
        for item in value[:12]:
            if isinstance(item, str):
                payload = {"text": item}
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                continue
            refs = [
                ref for ref in payload.get("source_refs", [])
                if isinstance(ref, str) and ref in allowed
            ]
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            try:
                suggestions.append(
                    ExpertAuditSuggestion(
                        type=str(payload.get("type") or "policy_based_verification")[:120],
                        text=text[:800],
                        source_refs=refs[:8],
                    )
                )
            except ValidationError:
                continue
        return suggestions

    def _evidence_for_prompt(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "evidence_ref": item.get("evidence_ref"),
                "source_ref": item.get("source_ref"),
                "title": item.get("title"),
                "excerpt": _clip(item.get("excerpt"), 1000),
                "jurisdiction": item.get("jurisdiction"),
                "policy_domain": item.get("policy_domain"),
                "content_type": item.get("content_type"),
                "metadata": item.get("metadata"),
            }
            for item in evidence[:8]
        ]

    def _rewrite_question_deterministically(
        self,
        question: str,
        missing_slots: Any,
    ) -> str:
        missing = "、".join(str(item) for item in missing_slots[:4]) if isinstance(missing_slots, list) else ""
        if missing:
            return f"{question}，重点补充检索：{missing}"
        return f"{question} 政策依据 适用范围 版本"

    def _can_use_tool(self, state: dict[str, Any]) -> bool:
        task = self._task(state)
        return state.get("tool_call_count", 0) < task.budget.max_tool_calls

    def _record_event(
        self,
        state: dict[str, Any],
        event_type: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        repository = self._runtime(state).get("repository")
        append_event = getattr(repository, "append_event", None)
        if not callable(append_event):
            return
        try:
            append_event(
                self._task(state).parent_run_id,
                event_type,
                message,
                payload,
            )
        except Exception:
            return

    def _save_checkpoint(self, state: dict[str, Any], node_name: str) -> None:
        repository = self._runtime(state).get("repository")
        save_checkpoint = getattr(repository, "save_checkpoint", None)
        if not callable(save_checkpoint):
            return
        try:
            save_checkpoint(
                self._task(state).parent_run_id,
                node_name=f"expert_analysis:{node_name}",
                state_snapshot=self._safe_state_snapshot(state),
                source_refs=[
                    ref for ref in state.get("source_refs", [])
                    if isinstance(ref, str) and ref
                ],
                safe_to_resume=True,
            )
        except Exception:
            return

    def _safe_state_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        task = state.get("task")
        task_payload = (
            task.model_dump(mode="json")
            if isinstance(task, ExpertAnalysisTask)
            else {}
        )
        return {
            "task": {
                "task_id": task_payload.get("task_id"),
                "parent_run_id": task_payload.get("parent_run_id"),
                "case_id": task_payload.get("case_id"),
                "capability": task_payload.get("capability"),
                "expert_task_type": task_payload.get("expert_task_type"),
                "goal": task_payload.get("goal"),
                "allowed_tools": task_payload.get("allowed_tools"),
                "constraints": task_payload.get("constraints"),
                "budget": task_payload.get("budget"),
            },
            "status": state.get("status"),
            "message": _clip(state.get("message"), 500),
            "model_call_count": state.get("model_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "retrieval_round_count": state.get("retrieval_round_count", 0),
            "rewrite_count": state.get("rewrite_count", 0),
            "retrieval_plan": {
                key: value
                for key, value in (state.get("retrieval_plan") or {}).items()
                if key != "case_context_requests"
            },
            "policy_request": state.get("policy_request", {}),
            "mcp_safe_summaries": state.get("mcp_safe_summaries", []),
            "hard_gate": state.get("hard_gate", {}),
            "evidence_normalizer": state.get("evidence_normalizer", {}),
            "question_slots": state.get("question_slots", []),
            "answer_requirements": state.get("answer_requirements", []),
            "slot_window_judge": state.get("slot_window_judge", {}),
            "sentence_window_count": len(state.get("sentence_windows", [])),
            "candidate_windows": [
                {
                    "window_id": item.get("window_id"),
                    "slot_id": item.get("slot_id"),
                    "evidence_id": item.get("evidence_id"),
                    "pre_score": item.get("pre_score"),
                    "policy_domain": item.get("policy_domain"),
                    "content_type": item.get("content_type"),
                    "jurisdiction": item.get("jurisdiction"),
                }
                for item in state.get("candidate_windows", [])
                if isinstance(item, dict)
            ][:24],
            "verified_slot_facts": [
                {
                    "fact_id": item.get("fact_id"),
                    "slot_id": item.get("slot_id"),
                    "fact_type": item.get("fact_type"),
                    "evidence_refs": item.get("evidence_refs"),
                    "source_refs": item.get("source_refs"),
                    "confidence": item.get("confidence"),
                    "value": item.get("value"),
                }
                for item in state.get("verified_slot_facts", [])
                if isinstance(item, dict)
            ][:40],
            "coverage_result": state.get("coverage_result", {}),
            "llm_semantic_coverage": state.get("llm_semantic_coverage", {}),
            "generation_policy": state.get("generation_policy", {}),
            "answerability_check": self._answerability_model_dump(state),
            "case_context_observations": [
                self._safe_case_observation(item)
                for item in state.get("case_context_observations", [])
                if isinstance(item, dict)
            ],
            "adopted_policy_evidence": [
                {
                    "evidence_ref": item.get("evidence_ref"),
                    "source_ref": item.get("source_ref"),
                    "title": item.get("title"),
                    "excerpt": _clip(item.get("excerpt"), 1000),
                    "jurisdiction": item.get("jurisdiction"),
                    "policy_domain": item.get("policy_domain"),
                    "content_type": item.get("content_type"),
                    "source_url": item.get("source_url"),
                    "version": item.get("version"),
                    "metadata": item.get("metadata"),
                }
                for item in state.get("adopted_policy_evidence", [])
                if isinstance(item, dict)
            ],
            "source_refs": state.get("source_refs", []),
            "error_code": state.get("error_code"),
        }

    def _safe_case_observation(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload")
        section_key = payload.get("section_key") if isinstance(payload, dict) else None
        inner = payload.get("payload") if isinstance(payload, dict) else None
        keys = list(inner.keys())[:20] if isinstance(inner, dict) else []
        return {
            "capability": item.get("capability"),
            "status": item.get("status"),
            "section_key": section_key,
            "payload_keys": keys,
            "source_refs": item.get("source_refs", [])[:8],
            "error_code": item.get("error_code"),
        }

    @staticmethod
    def _source_refs_from_case_observations(
        observations: list[dict[str, Any]],
    ) -> list[str]:
        refs: list[str] = []
        for observation in observations:
            refs.extend(
                str(ref) for ref in observation.get("source_refs", [])
                if ref
            )
        return list(dict.fromkeys(refs))


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    if start < 0:
        return "{}"
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]


def _policy_plan_raw_output(response: Any) -> str:
    if isinstance(response, ModelResponse):
        for tool_call in response.tool_calls:
            if tool_call.name == "submit_policy_retrieval_plan":
                return str(tool_call.arguments or "")
        return str(response.content or "")
    return str(response or "")


def _has_forced_policy_filters(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return _bool(
        value.get("force_policy_filters", value.get("_force_policy_filters")),
        False,
    )


def _stable_ref(value: Any) -> str:
    import hashlib

    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _fallback_semantic_coverage(coverage_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "supported_slots": _coverage_labels_by_status(coverage_result, {"strong"}),
        "weak_slots": _coverage_labels_by_status(coverage_result, {"weak"}),
        "missing_slots": _coverage_labels_by_status(coverage_result, {"missing"}),
        "usable_evidence_refs": _string_list(coverage_result.get("usable_evidence_refs")),
        "coverage_comment": str(coverage_result.get("decision_reason") or ""),
    }


def _coerce_semantic_coverage(
    payload: Any,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else fallback
    supported = _string_list(data.get("supported_slots") or data.get("covered_slots"))
    weak = _string_list(data.get("weak_slots"))
    missing = _string_list(data.get("missing_slots"))
    refs = _string_list(data.get("usable_evidence_refs"))
    return {
        "supported_slots": supported[:12],
        "weak_slots": weak[:12],
        "missing_slots": missing[:12],
        "usable_evidence_refs": refs[:20],
        "coverage_comment": _clip(
            data.get("coverage_comment") or data.get("reason") or fallback.get("coverage_comment"),
            500,
        ),
    }


def _merge_slot_facts_for_required_field_coverage(
    *,
    requirements: list[AnswerRequirement],
    primary_facts: list[ExtractedFact],
    fallback_facts: list[ExtractedFact],
) -> list[ExtractedFact]:
    """Backfill only facts that cover still-missing required fields."""

    if not primary_facts or not fallback_facts:
        return primary_facts or fallback_facts
    required_by_requirement = {
        item.requirement_id: set(_string_list(item.required_fields))
        for item in requirements
    }
    covered_by_requirement: dict[str, set[str]] = {
        item.requirement_id: set()
        for item in requirements
    }
    for fact in primary_facts:
        fields = set(_string_list(fact.value.get("matched_required_fields")))
        covered_by_requirement.setdefault(fact.requirement_id, set()).update(fields)

    merged: list[ExtractedFact] = list(primary_facts)
    seen = {_fact_merge_key(fact) for fact in merged}
    for requirement in requirements:
        required_fields = required_by_requirement.get(requirement.requirement_id, set())
        if not required_fields:
            continue
        missing = required_fields - covered_by_requirement.get(requirement.requirement_id, set())
        if not missing:
            continue
        for fact in fallback_facts:
            if fact.requirement_id != requirement.requirement_id:
                continue
            fact_fields = set(_string_list(fact.value.get("matched_required_fields")))
            if not fact_fields.intersection(missing):
                continue
            key = _fact_merge_key(fact)
            if key in seen:
                continue
            merged.append(fact)
            seen.add(key)
            covered_by_requirement.setdefault(requirement.requirement_id, set()).update(fact_fields)
            missing = required_fields - covered_by_requirement.get(requirement.requirement_id, set())
            if not missing:
                break
    return _reindex_extracted_facts(merged)


def _merge_evidence_matches(
    primary_matches: list[EvidenceMatch],
    fallback_matches: list[EvidenceMatch],
) -> list[EvidenceMatch]:
    merged: list[EvidenceMatch] = []
    seen: set[tuple[str, str, str]] = set()
    first_number = _match_id_number(
        (primary_matches or fallback_matches)[0].match_id
    ) if (primary_matches or fallback_matches) else 1
    first_number = first_number or 1
    for match in [*primary_matches, *fallback_matches]:
        key = (match.requirement_id, match.evidence_id, match.reason)
        if key in seen:
            continue
        seen.add(key)
        merged.append(match.model_copy(update={"match_id": f"match_{first_number + len(merged)}"}))
    return merged


def _fact_merge_key(fact: ExtractedFact) -> tuple[str, str, str]:
    return (
        fact.requirement_id,
        fact.slot_id,
        re.sub(r"\s+", "", fact.display_text),
    )


def _reindex_extracted_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    if not facts:
        return []
    first_number = _fact_id_number(facts[0].fact_id) or 1
    return [
        fact.model_copy(update={"fact_id": f"fact_{first_number + offset}"})
        for offset, fact in enumerate(facts)
    ]


def _fact_id_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else None


def _match_id_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else None


def _generation_policy_from_coverage(coverage_result: dict[str, Any]) -> dict[str, str]:
    slot_status = coverage_result.get("slot_status")
    if not isinstance(slot_status, dict):
        return {}
    policy: dict[str, str] = {}
    for slot_id, item in slot_status.items():
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status == "strong":
            policy[str(slot_id)] = "allow"
        elif status == "weak":
            policy[str(slot_id)] = "allow_with_qualification"
        else:
            policy[str(slot_id)] = "deny"
    return policy


def _should_follow_up_required_field_coverage(
    *,
    question: str,
    coverage_result: dict[str, Any],
) -> bool:
    if coverage_result.get("coverage_basis") != "slot_facts":
        return False
    slot_status = coverage_result.get("slot_status")
    if not isinstance(slot_status, dict):
        return False
    compact = re.sub(r"\s+", "", str(question or ""))
    if _looks_like_remote_direct_settlement_payment_query(question):
        for slot_id in ("remote_benefit_split", "remote_filing_institution_scope"):
            item = slot_status.get(slot_id)
            if isinstance(item, dict) and _string_list(item.get("missing_fields")):
                return True
    scenario_slots = {
        "foreign_treatment_manual_reimbursement",
        "account_settlement_voucher",
        "emergency_manual_reimbursement_materials",
        "remote_self_pay_filing_manual_reimbursement",
        "remote_emergency_observation_reimbursement",
    }
    explicit_multi_scenario = "分别" in compact or len(scenario_slots.intersection(slot_status.keys())) >= 2
    if explicit_multi_scenario:
        for slot_id in scenario_slots:
            item = slot_status.get(slot_id)
            if not isinstance(item, dict):
                continue
            if item.get("status") == "missing" or _string_list(item.get("missing_fields")):
                return True
    return False


def _coverage_labels_by_status(
    coverage_result: dict[str, Any],
    statuses: set[str],
) -> list[str]:
    slot_status = coverage_result.get("slot_status")
    if not isinstance(slot_status, dict):
        return []
    labels: list[str] = []
    for slot_id, item in slot_status.items():
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") in statuses:
            label = str(item.get("label") or slot_id)
            if "missing" in statuses:
                missing_fields = _string_list(item.get("missing_fields"))
                if missing_fields:
                    label = f"{label}缺少{_join_cn(missing_fields[:4])}"
            labels.append(label)
        elif "missing" in statuses and _string_list(item.get("missing_fields")):
            labels.append(
                f"{str(item.get('label') or slot_id)}缺少{_join_cn(_string_list(item.get('missing_fields'))[:4])}"
            )
    return list(dict.fromkeys(labels))


def _jurisdiction_compatible(
    requested_jurisdictions: set[str],
    evidence_jurisdiction: str | None,
) -> bool:
    evidence = str(evidence_jurisdiction or "").strip().lower()
    if not requested_jurisdictions or not evidence:
        return True
    if evidence == "national":
        return True
    return evidence in requested_jurisdictions


def _slot_evidence_matches(
    slot: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    keywords = [str(item) for item in slot.get("keywords", []) if str(item)]
    domains = set(_string_list(slot.get("policy_domains")))
    slot_id = str(slot.get("slot_id") or "")
    matches: list[dict[str, str]] = []
    for item in evidence:
        ref = str(item.get("evidence_ref") or item.get("source_ref") or "")
        if not ref:
            continue
        policy_domain = str(item.get("policy_domain") or "")
        title = str(item.get("title") or "")
        excerpt = str(item.get("excerpt") or "")
        text = title + "\n" + excerpt
        domain_match = not domains or policy_domain in domains
        text_match = not keywords or any(keyword in text for keyword in keywords)
        if (
            text_match
            and policy_domain == "remote_medical"
            and slot_id in {"emergency_exception", "filing_rule", "manual_reimbursement"}
        ):
            domain_match = True
        if slot_id == "policy_basis" and excerpt:
            domain_match = True
            text_match = True
        if domain_match and text_match:
            matches.append({"evidence_ref": ref, "status": "strong"})
        elif domain_match or (text_match and not domains):
            matches.append({"evidence_ref": ref, "status": "weak"})
    return matches


def _slot_fact_matches(
    slot: dict[str, Any],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slot_id = str(slot.get("slot_id") or "")
    requirement_id = str(slot.get("requirement_id") or slot.get("need_id") or "")
    slot_key = requirement_id or slot_id
    matches: list[dict[str, Any]] = []
    for fact in facts:
        fact_requirement_id = str(fact.get("requirement_id") or fact.get("need_id") or "")
        fact_slot_id = str(fact.get("slot_id") or "")
        if fact_requirement_id and fact_requirement_id != slot_key and fact_slot_id != slot_id:
            continue
        if not fact_requirement_id and fact_slot_id != slot_id:
            continue
        value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
        quote_verified = bool(value.get("quote_verified", True))
        if not quote_verified:
            continue
        score = _float_between(
            fact.get("confidence") or value.get("llm_score"),
            0.0,
            1.0,
            0.0,
        )
        support_role = str(value.get("support_role") or "").strip() or (
            "direct_answer" if score >= 0.75 else "supporting"
        )
        evidence_refs = _string_list(fact.get("evidence_refs"))
        evidence_ref = evidence_refs[0] if evidence_refs else ""
        if not evidence_ref:
            continue
        if support_role == "direct_answer" and score >= 0.75:
            status = "strong"
        elif score >= 0.55:
            status = "weak"
        else:
            continue
        matches.append(
            {
                "evidence_ref": evidence_ref,
                "fact_ref": str(fact.get("fact_id") or ""),
                "status": status,
                "score": score,
                "support_role": support_role,
                "matched_fields": _string_list(value.get("matched_required_fields")),
            }
        )
    return matches


def _requirements_by_slot(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    requirements: dict[str, dict[str, Any]] = {}
    for item in state.get("answer_requirements", []):
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or "")
        requirement_id = str(item.get("requirement_id") or "")
        if requirement_id and requirement_id not in requirements:
            requirements[requirement_id] = item
        if slot_id and slot_id not in requirements:
            requirements[slot_id] = item
    return requirements


def _merge_filters(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    extra = extra or {}
    merged = merge_policy_filters(base, extra)
    cite = extra.get("can_cite_as_policy_basis", merged.get("can_cite_as_policy_basis"))
    merged["can_cite_as_policy_basis"] = (
        True if cite is None else _bool(cite, bool(cite))
    )
    if "content_type" not in merged:
        merged["content_type"] = ["policy_text", "table_row"]
    return merged


def _apply_goldset_style_filter_template(
    question: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Apply online equivalents of policy RAG goldset filter templates."""

    if not _looks_like_remote_benefit_split_query(question):
        return filters

    jurisdictions = _string_list(filters.get("jurisdiction"))
    normalized_jurisdictions = [
        jurisdiction
        for item in jurisdictions
        if (jurisdiction := _normalize_jurisdiction(item))
    ]
    if "national" not in normalized_jurisdictions:
        normalized_jurisdictions.insert(0, "national")

    return {
        **filters,
        "jurisdiction": list(dict.fromkeys(normalized_jurisdictions)),
        "policy_domain": ["remote_medical"],
        "content_type": ["policy_text"],
        "can_cite_as_policy_basis": True,
    }


def _string_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _join_cn(items: list[str]) -> str:
    return "、".join(str(item) for item in items if str(item))


def _normalize_jurisdiction(value: str) -> str:
    filters = normalize_policy_filters({"jurisdiction": [value]})
    values = _string_list(filters.get("jurisdiction"))
    return values[0] if values else ""


def _normalize_policy_domain(value: str) -> list[str]:
    filters = normalize_policy_filters({"policy_domain": [value]})
    return _string_list(filters.get("policy_domain"))


def _looks_like_remote_benefit_split_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_remote_context = any(token in compact for token in ("跨省", "异地就医", "异地"))
    has_split_parties = "就医地" in compact and "参保地" in compact
    has_split_topic = any(
        token in compact
        for token in (
            "目录",
            "支付范围",
            "待遇",
            "支付比例",
            "起付",
            "最高支付限额",
            "分工",
            "怎么分",
            "如何分",
            "区分",
        )
    )
    return has_remote_context and has_split_parties and has_split_topic


def _dynamic_slot_required_fields(slot_id: str, question: str) -> list[str]:
    fields = list(slot_required_fields(slot_id))
    if slot_id == "remote_benefit_split" and _looks_like_remote_direct_settlement_payment_query(question):
        return [
            "直接结算费用范围",
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
            "门诊慢特病病种范围",
        ]
    return fields


def _looks_like_remote_direct_settlement_payment_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_remote_context = any(token in compact for token in ("跨省", "异地就医", "异地"))
    has_direct_settlement = "直接结算" in compact
    asks_payment_rule = any(
        token in compact
        for token in (
            "医疗费用支付规则",
            "费用支付规则",
            "支付规则",
            "支付口径",
            "怎样支付",
            "如何支付",
        )
    )
    return has_remote_context and has_direct_settlement and asks_payment_rule


def _looks_like_remote_filing_institution_scope_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_remote_context = any(token in compact for token in ("跨省", "异地就医", "异地"))
    has_filing_success = any(token in compact for token in ("备案成功", "备案后", "办理备案"))
    asks_institution_scope = any(
        token in compact
        for token in (
            "哪些机构",
            "哪些定点",
            "定点医药机构",
            "定点医疗机构",
            "在哪里就医",
            "在哪些机构就医",
            "统筹地区",
        )
    )
    return has_remote_context and has_filing_success and asks_institution_scope


def _infer_answer_slot_ids_from_question(
    question: str,
    filters: dict[str, Any] | None = None,
) -> list[str]:
    compact = re.sub(r"\s+", "", str(question or ""))
    domains = set(_string_list((filters or {}).get("policy_domain")))
    slot_ids: list[str] = []

    def add(slot_id: str) -> None:
        if slot_id not in slot_ids:
            slot_ids.append(slot_id)

    benefit_split_query = (
        _looks_like_remote_benefit_split_query(question)
        or _looks_like_remote_direct_settlement_payment_query(question)
    )
    scenario_slot_ids = _manual_reimbursement_scenario_slot_ids(question)
    for slot_id in scenario_slot_ids:
        add(slot_id)
    policy_scenario_slot_ids = _policy_scenario_slot_ids(question)
    for slot_id in policy_scenario_slot_ids:
        add(slot_id)
    has_specific_scenario_slots = bool(scenario_slot_ids or policy_scenario_slot_ids)
    if benefit_split_query:
        add("remote_benefit_split")
    if _looks_like_remote_filing_institution_scope_query(question):
        add("remote_filing_institution_scope")
    if any(token in compact for token in ("银行手续费", "银行票据", "工本费", "预付金", "黄色预警", "红色预警", "紧急调增", "费用协查")):
        add("remote_settlement_management")
    if any(token in compact for token in ("城乡居民医保", "城乡老年人", "参保范围", "参保资格", "新生儿", "等待期", "外埠户籍配偶", "家庭医生", "首诊转诊", "外省市目录")) or "benefit" in domains:
        add("benefit_policy")
    if any(token in compact for token in ("长期处方", "长处方", "慢性病", "高血压", "糖尿病", "BJ-GBI", "医事服务费", "月度通报", "品种规格", "医联体")) or "chronic_disease_long_prescription" in domains:
        add("chronic_long_prescription_policy")
    if any(token in compact for token in ("基金监管", "监督检查", "拒不配合", "暂停联网结算", "锁卡", "骗取基金", "涉嫌骗保", "不属于基金支付范围", "异常情形审核")) or "fund_supervision" in domains:
        add("fund_supervision_policy")
    if any(token in compact for token in ("特殊病备案", "特殊病种备案", "门诊特殊病备案", "门诊特殊疾病备案", "备案申报表", "医保办公室", "医保办", "病种名称", "中重度哮喘", "外埠户籍", "24个月")) or "special_disease_filing" in domains:
        add("special_disease_filing_policy")
    if any(token in compact for token in ("特殊疾病范围", "门诊特殊疾病范围", "新增病种", "重性精神病", "肺动脉高压", "未备案", "报销范围")) or "special_disease_scope" in domains:
        if any(token in compact for token in ("限定支付条件", "方可支付", "类风湿关节炎", "DMARDs", "风湿病专科医师")):
            add("special_disease_payment_condition")
        else:
            add("special_disease_scope_policy")
    if any(token in compact for token in ("医疗服务设施", "住院床位费", "急诊观察室床位费", "床位费", "实施期限", "有效期")):
        add("shanghai_service_facility_scope")
    if any(token in compact for token in ("协议期内谈判药品", "谈判药品", "双通道", "电子处方", "一品两规", "药占比", "总额限制")):
        add("negotiated_drug_double_channel")
    if (
        not has_specific_scenario_slots
        and any(token in compact for token in ("备案", "补备案", "急诊", "门急诊", "急诊抢救", "急诊例外"))
        or (
            not has_specific_scenario_slots
            and
            not benefit_split_query
            and any(token in compact for token in ("异地备案", "跨省备案", "异地急诊"))
        )
        or (
            not has_specific_scenario_slots
            and
            not benefit_split_query
            and domains.intersection({"remote_medical", "remote_medical_manual_reimbursement", "emergency"})
        )
    ):
        add("remote_filing")
    if any(
        token in compact
        for token in (
            "必要材料",
            "申请材料",
            "提交哪些材料",
            "需要哪些材料",
            "核验哪些材料",
            "应核验哪些材料",
            "需要核验哪些材料",
            "补充哪些材料",
            "报销材料",
            "材料目录",
        )
    ) and not scenario_slot_ids:
        add("required_materials")
    if any(token in compact for token in ("法定办结时限", "承诺办结时限", "办结时限", "办理时限", "多少工作日", "多久办结")):
        add("statutory_processing_time")
    if (
        any(token in compact for token in ("手工报销", "零星报销", "外埠就医", "报销路径"))
        or "manual_reimbursement" in domains
    ) and not has_specific_scenario_slots:
        add("manual_reimbursement")
    if _looks_like_drug_catalog_query(question) or (
        "drug_catalog" in domains
        and not any(token in compact for token in ("双通道", "谈判药品", "电子处方", "一品两规"))
    ):
        add("drug_catalog")
    if _looks_like_medical_service_price_query(question) or "medical_service_price" in domains:
        add("medical_service_price")
    if _looks_like_consumable_payment_scope_query(question):
        add("consumable_payment_scope")
    if any(token in compact for token in ("起付线", "起付标准", "支付比例", "报销比例", "封顶线", "最高支付限额")):
        add("benefit_params")
    if any(token in compact for token in ("定点机构", "定点医院", "定点药店", "机构编码", "药店编码", "定点状态")) or "designated_institution" in domains:
        add("designated_institution")
    if not slot_ids:
        add("policy_basis")
    return normalize_answer_slot_ids(slot_ids)[:8]


_INFORMATION_NEED_PHRASES: dict[str, str] = {
    "required_materials": "需要哪些必要材料",
    "statutory_processing_time": "法定办结时限是多少",
    "drug_catalog": "相关药品是否纳入医保药品目录",
    "medical_service_price": "相关医疗服务项目价格是多少",
    "consumable_payment_scope": "相关医用耗材是否纳入支付范围",
    "remote_benefit_split": "跨省异地就医直接结算时就医地和参保地分别负责什么",
    "benefit_params": "待遇参数有哪些",
    "remote_filing": "异地就医备案如何办理",
    "remote_filing_institution_scope": "备案成功后可在哪些机构就医",
    "remote_settlement_management": "异地就医结算管理规则是什么",
    "remote_self_pay_filing_manual_reimbursement": "异地自费结算后手工报销如何办理",
    "remote_emergency_observation_reimbursement": "异地急诊留观费用如何报销",
    "foreign_treatment_manual_reimbursement": "外埠就医手工报销需要哪些材料",
    "account_settlement_voucher": "定点医药机构记账结算凭证如何开具",
    "emergency_manual_reimbursement_materials": "急诊就医手工报销需要哪些材料",
    "designated_institution": "定点机构或药店状态如何核验",
    "manual_reimbursement": "手工报销如何办理",
    "policy_basis": "政策依据是什么",
    "benefit_policy": "相关待遇政策是什么",
    "chronic_long_prescription_policy": "慢病长处方政策是什么",
    "fund_supervision_policy": "基金监管规则是什么",
    "special_disease_filing_policy": "特殊病备案如何办理",
    "special_disease_scope_policy": "门诊特殊疾病范围是什么",
    "special_disease_payment_condition": "门诊特殊疾病限定支付条件是什么",
    "shanghai_service_facility_scope": "上海医疗服务设施范围如何核验",
    "negotiated_drug_double_channel": "谈判药品双通道规则是什么",
    "remote_medical": "跨省异地就医支付规则是什么",
    "remote_medical_manual_reimbursement": "跨省异地就医手工报销如何办理",
    "medical_service_price": "医疗服务项目价格是多少",
    "drug_catalog": "相关药品目录如何核验",
}


def _normalize_information_needs(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, dict):
        raw = value.get("information_needs")
    else:
        raw = value
    items: list[str] = []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                candidate = item.get("need_text") or item.get("label") or item.get("text")
            else:
                candidate = item
            text = str(candidate or "").strip()
            if text:
                items.append(text)
    elif raw is not None:
        text = str(raw).strip()
        if text:
            items = [text]
    normalized: list[str] = []
    for item in items:
        need = str(item).strip()
        if need not in normalized:
            normalized.append(need)
    return normalized[:12]


def _infer_answer_mode_from_question(
    question: str,
    information_needs: list[str] | None = None,
) -> str:
    return _normalize_answer_mode(None, question, information_needs or [])


def _legacy_slot_to_information_need(slot_id: str) -> str:
    return _INFORMATION_NEED_PHRASES.get(slot_id, slot_label(slot_id))


def _infer_information_needs_from_question(
    question: str,
    filters: dict[str, Any] | None = None,
) -> list[str]:
    compact = re.sub(r"\s+", "", str(question or "")).strip("。！？?!；; ")
    if not compact:
        return []
    hard_clauses = [
        part.strip(" ，,；;。！？?!")
        for part in re.split(r"[；;。！？?!]+", compact)
        if part.strip(" ，,；;。！？?!")
    ]
    needs: list[str] = []
    for clause in hard_clauses or [compact]:
        enumerated = _expand_enumerated_information_need(clause)
        needs.extend(enumerated or [clause])
    return list(dict.fromkeys(item for item in needs if item))[:12]


def _expand_enumerated_information_need(clause: str) -> list[str]:
    if not re.search(r"、|以及|和|及|与", clause):
        return []
    parts = [
        part.strip(" ，,")
        for part in re.split(r"(?:、|以及|和|及|与)", clause)
        if part.strip(" ，,")
    ]
    if len(parts) <= 1:
        return []
    suffix = _shared_information_need_suffix(parts[-1])
    if not suffix:
        return []
    last_subject = parts[-1][:-len(suffix)].strip(" ，,")
    if not last_subject:
        return []
    subjects = [*parts[:-1], last_subject]
    if any(len(subject) > 40 for subject in subjects):
        return []
    return [f"{subject}{suffix}"[:240] for subject in subjects if subject]


def _shared_information_need_suffix(text: str) -> str:
    markers = (
        "支付范围",
        "报销比例",
        "支付比例",
        "起付标准",
        "最高支付限额",
        "医保类别",
        "目录编号",
        "限定支付",
        "计价单位",
        "收费标准",
        "办理时限",
        "办结时限",
        "需要提交",
        "需要提供",
        "需要核验",
        "应当核验",
        "如何核验",
        "如何办理",
        "是否",
        "有哪些",
        "哪些",
        "是什么",
        "是多少",
    )
    positions = [text.find(marker) for marker in markers if text.find(marker) > 0]
    if not positions:
        return ""
    return text[min(positions):]


def _normalize_answer_mode(
    value: Any,
    question: str,
    information_needs: list[str] | None = None,
) -> str:
    candidate = str(value or "").strip().lower()
    compact = re.sub(r"\s+", "", str(question or ""))
    needs = information_needs or []
    # Explicit comparison wording is stronger than a loose model mode label.
    if any(token in compact for token in ("比较", "对比", "分工", "各自负责", "分别负责")):
        return "comparison"
    if candidate in {"fact_lookup", "list", "process_rule", "policy_explanation", "comparison"}:
        return candidate
    if any(token in compact for token in ("列出", "有哪些", "哪些", "清单", "目录")) and len(needs) <= 3:
        return "list"
    if any(token in compact for token in ("规则", "流程", "条件", "办理", "如何", "怎么", "步骤", "时限")):
        return "process_rule"
    if any(token in compact for token in ("为什么", "解释", "说明", "含义")):
        return "policy_explanation"
    if len(needs) == 1 and any(token in compact for token in ("是什么", "多少", "哪一个", "哪个")):
        return "fact_lookup"
    return "process_rule"


def _adaptive_policy_retrieval_budget(
    *,
    question: str,
    filters: dict[str, Any],
    answer_mode: str,
    information_needs: list[str],
) -> dict[str, Any]:
    """Choose the smallest retrieval budget that preserves query coverage."""

    def _value_count(value: Any) -> int:
        if isinstance(value, (list, tuple, set)):
            return len({str(item).strip() for item in value if str(item).strip()})
        return 1 if str(value or "").strip() else 0

    jurisdiction_count = _value_count(filters.get("jurisdiction"))
    domain_count = _value_count(filters.get("policy_domain"))
    need_count = len({item.strip() for item in information_needs if item.strip()})
    compact_question = re.sub(r"\s+", "", str(question or ""))
    comparison_requested = answer_mode == "comparison" or any(
        token in compact_question
        for token in ("比较", "对比", "分工", "分别适用", "各自适用")
    )

    if comparison_requested or jurisdiction_count > 1:
        return {
            "complexity": "complex",
            "fetch_k": POLICY_RAG_COMPLEX_FETCH_K,
            "rerank": True,
            "rerank_candidate_limit": 16,
        }
    if domain_count > 1 or need_count > 1:
        return {
            "complexity": "medium",
            "fetch_k": POLICY_RAG_MEDIUM_FETCH_K,
            "rerank": True,
            "rerank_candidate_limit": 12,
        }
    return {
        "complexity": "simple",
        "fetch_k": POLICY_RAG_SIMPLE_FETCH_K,
        "rerank": False,
        "rerank_candidate_limit": 0,
    }


def _information_needs_to_legacy_slots(
    question: str,
    information_needs: list[str],
    filters: dict[str, Any] | None = None,
) -> list[str]:
    if not information_needs:
        return _infer_answer_slot_ids_from_question(question, filters)
    slots: list[str] = []
    for need in information_needs:
        slots.extend(_infer_answer_slot_ids_from_question(need, filters))
    if not slots:
        slots = _infer_answer_slot_ids_from_question(question, filters)
    return normalize_answer_slot_ids(slots)[:8]


def _judged_windows_from_payload(
    *,
    candidate_windows: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    windows_by_id = {
        str(item.get("window_id") or ""): item
        for item in candidate_windows
        if item.get("window_id")
    }
    judged: list[dict[str, Any]] = []
    raw_results = payload.get("window_results")
    if not isinstance(raw_results, list):
        return candidate_windows[:24]
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        window_id = str(item.get("window_id") or "")
        window = windows_by_id.get(window_id)
        if window is None:
            continue
        score = _float_between(item.get("llm_score") or item.get("score"), 0.0, 1.0, 0.0)
        role = str(item.get("support_role") or "").strip().lower()
        if role == "irrelevant" or score < 0.55:
            continue
        judged.append(window)
    return judged[:24] or candidate_windows[:24]


def _looks_like_special_disease_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "门诊特殊疾病",
            "特殊疾病",
            "限定支付条件",
            "方可支付",
            "类风湿关节炎",
            "DMARDs",
            "风湿病专科医师",
            "门诊慢特病",
            "慢特病",
            "门特",
            "特殊病",
            "特病",
            "慢病",
            "病种范围",
            "病种备案",
            "病种核验",
        )
    )


def _looks_like_drug_catalog_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or _looks_like_remote_benefit_split_query(compact):
        return False
    if _looks_like_special_disease_query(compact) and not _looks_like_table_row_fact_query(compact):
        return False
    if _looks_like_drug_price_reference_query(compact):
        return False
    drug_markers = (
        "药品",
        "药物",
        "用药",
        "药品目录",
        "医保药品目录",
        "限定支付范围",
        "药品编码",
        "通用名",
    )
    return any(marker in compact for marker in drug_markers)


def _looks_like_table_row_fact_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "医保类别",
            "目录编号",
            "药品编码",
            "项目编码",
            "机构编码",
            "甲类",
            "乙类",
            "单价",
            "价格是多少",
            "收费标准",
            "计价单位",
            "本地支付比例",
            "支付比例是多少",
            "报销比例是多少",
            "编码是什么",
            "编号是什么",
            "类别是什么",
        )
    )


def _looks_like_shanghai_payment_scope_verification_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or "上海" not in compact:
        return False
    has_scope_intent = any(
        token in compact
        for token in ("支付范围", "医保支付", "报销范围", "如何核验", "怎么核验", "核验")
    )
    topic_count = sum(
        bool(any(token in compact for token in group))
        for group in (
            ("药品", "药物", "用药", "药品目录"),
            ("CT", "ct", "胸部CT", "胸部ct", "检查", "影像"),
            ("医用耗材", "耗材"),
        )
    )
    return has_scope_intent and topic_count >= 2


def _looks_like_manual_reimbursement_service_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    has_reimbursement = any(
        token in compact
        for token in ("零星报销", "手工报销", "门诊费用报销", "门诊医疗费用报销", "门诊报销")
    )
    has_service_need = any(
        token in compact
        for token in (
            "必要材料",
            "申请材料",
            "材料清单",
            "需要哪些材料",
            "应要求提供哪些",
            "法定办结时限",
            "办结时限",
            "办理时限",
            "多少工作日",
            "服务指南",
            "办事指南",
        )
    )
    return has_reimbursement and has_service_need


def _looks_like_consumable_payment_scope_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(compact) and any(token in compact for token in ("医用耗材", "耗材"))


def _looks_like_medical_service_price_query(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    compact_lower = compact.lower()
    if not compact:
        return False
    return (
        any(token in compact for token in ("医疗服务价格", "医疗服务项目", "诊疗项目", "收费", "价格", "项目"))
        or "ct" in compact_lower
        or any(token in compact for token in ("检查", "检验", "影像"))
    )


def _looks_like_drug_price_reference_query(text: str) -> bool:
    if not text or "上海" not in text:
        return False
    price_markers = (
        "药价",
        "价格",
        "均价",
        "平均价",
        "平均价格",
        "周均价",
        "价格区间",
        "多少钱",
        "多少",
    )
    drug_markers = (
        "药",
        "药品",
        "药店",
        "布洛芬",
        "阿莫西林",
        "阿托伐他汀",
        "二甲双胍",
        "氨氯地平",
        "缬沙坦",
        "连花清瘟",
        "抗病毒口服液",
        "头孢",
        "胶囊",
        "片",
        "口服液",
        "颗粒",
    )
    return any(marker in text for marker in price_markers) and any(
        marker in text for marker in drug_markers
    )


def _manual_reimbursement_scenario_slot_ids(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return []
    slot_ids: list[str] = []
    if any(token in compact for token in ("外埠就医", "外地就医", "易地安置")):
        slot_ids.append("foreign_treatment_manual_reimbursement")
    if (
        any(token in compact for token in ("记账结算", "记帐结算", "记账"))
        and any(token in compact for token in ("定点医药机构", "定点医疗机构", "定点零售药店"))
    ):
        slot_ids.append("account_settlement_voucher")
    if any(token in compact for token in ("急诊就医", "急诊留观", "未出示社保卡", "医保电子凭证", "没带医保凭证")):
        slot_ids.append("emergency_manual_reimbursement_materials")
    if len(slot_ids) >= 2 or "分别" in compact:
        return list(dict.fromkeys(slot_ids))
    return []


def _policy_scenario_slot_ids(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return []
    slot_ids: list[str] = []
    if (
        any(token in compact for token in ("自费结算", "自行垫付", "全额垫付", "补办备案", "补办备案手续"))
        and any(token in compact for token in ("手工报销", "报销路径", "申请报销"))
    ):
        slot_ids.append("remote_self_pay_filing_manual_reimbursement")
    asks_materials_only = any(
        token in compact
        for token in ("核验哪些材料", "需要哪些材料", "提交哪些材料", "报销材料", "材料清单")
    ) and not any(
        token in compact
        for token in ("直接结算", "暂不能", "不能直接", "报销路径", "怎么报销", "如何报销", "住院标准", "社保所", "经办机构")
    )
    if (
        any(token in compact for token in ("急诊留观", "留观费用", "异地急诊留观"))
        and not asks_materials_only
    ):
        slot_ids.append("remote_emergency_observation_reimbursement")
    return list(dict.fromkeys(slot_ids))


def _has_drug_price_reference_evidence(evidence: list[PolicyEvidence]) -> bool:
    return any(
        item.used_for == "drug_price_reference"
        or item.policy_domain == "drug_product_price_reference"
        for item in evidence
    )


def _format_drug_price_reference_answer(
    task: ExpertAnalysisTask,
    evidence: list[PolicyEvidence],
) -> str:
    rows = [
        row
        for row in (
            _drug_price_reference_row(item)
            for item in evidence
            if item.used_for == "drug_price_reference"
            or item.policy_domain == "drug_product_price_reference"
        )
        if row
    ]
    if not rows:
        return (
            "根据已检索到的上海药品产品价格参考（非医保政策依据），"
            f"当前只能返回原始参考片段：{_clip(evidence[0].excerpt, 260)}"
        )

    query_name = _drug_name_from_query(task.user_question)
    header_name = f"“{query_name}”" if query_name else "该药品"
    if len(rows) == 1:
        row = rows[0]
        return (
            f"根据已检索到的上海药品产品价格参考（非医保政策依据），{header_name}"
            f"匹配到 1 个价格参考产品：{_format_drug_price_row(row)}。"
            "该价格为药店产品价格参考，不作为医保报销政策依据。"
        )

    lines = [
        (
            f"根据已检索到的上海药品产品价格参考（非医保政策依据），{header_name}"
            f"匹配到多个药品产品/规格，不能合并成一个唯一均价；"
            "需要按药品名称、规格、生产企业或药品编码确认具体产品。"
        ),
        "当前召回的前几项参考价如下：",
    ]
    for index, row in enumerate(rows[:5], start=1):
        lines.append(f"{index}. {_format_drug_price_row(row)}。")
    lines.append("该价格为药店产品价格参考，不作为医保报销政策依据。")
    return "\n".join(lines)


def _drug_price_reference_row(evidence: PolicyEvidence) -> dict[str, str] | None:
    fields = _drug_price_reference_fields(evidence.excerpt)
    name = fields.get("药品名称") or fields.get("注册名称")
    avg_price = fields.get("医保药店周均价")
    if not name and not avg_price:
        return None
    return {
        "name": name or evidence.title,
        "registered_name": fields.get("注册名称", ""),
        "code": fields.get("药品编码", ""),
        "company": fields.get("生产企业", ""),
        "spec": fields.get("规格", ""),
        "dosage": fields.get("剂型", ""),
        "package_unit": fields.get("最小包装单位", ""),
        "package_count": fields.get("最小包装数量", ""),
        "avg_price": avg_price,
        "price_range": fields.get("医保药店周价格区间", ""),
        "period": fields.get("价格周期", ""),
    }


def _drug_price_reference_fields(text: str) -> dict[str, str]:
    normalized = str(text or "")
    normalized = normalized.replace("，医保药店周价格区间：", "；医保药店周价格区间：")
    normalized = normalized.replace("。 ", "；").replace("。", "；")
    fields: dict[str, str] = {}
    for part in normalized.split("；"):
        if "：" not in part:
            continue
        key, value = part.split("：", 1)
        key = key.strip()
        value = value.strip(" ；。")
        if key and value:
            fields[key] = value
    return fields


def _format_drug_price_row(row: dict[str, str]) -> str:
    identity_parts = [
        row.get("name", ""),
        row.get("company", ""),
        row.get("spec", ""),
    ]
    identity = " / ".join(part for part in identity_parts if part)
    avg_price = row.get("avg_price") or "未列明"
    price_range = row.get("price_range")
    period = row.get("period")
    detail_parts = [f"医保药店周均价 {avg_price}"]
    if price_range:
        detail_parts.append(f"价格区间 {price_range}")
    if period:
        detail_parts.append(f"价格周期 {period}")
    code = row.get("code")
    if code:
        detail_parts.append(f"药品编码 {code}")
    return f"{identity}：" + "，".join(detail_parts)


def _drug_name_from_query(text: str) -> str:
    candidates = [
        "布洛芬缓释胶囊",
        "精氨酸布洛芬颗粒",
        "布洛芬",
        "阿莫西林",
        "阿托伐他汀",
        "二甲双胍",
        "氨氯地平",
        "缬沙坦",
        "连花清瘟",
        "抗病毒口服液",
        "头孢",
    ]
    for candidate in candidates:
        if candidate in text:
            return candidate
    match = re.search(r"上海(?:的|市)?(.+?)(?:均价|平均价|平均价格|周均价|价格|多少钱|多少)", text)
    if not match:
        return ""
    return match.group(1).strip(" 的药品药店")


def _contains_remote_benefit_split(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    place_patterns = (
        "就医地规定的支付范围",
        "就医地支付范围",
        "就医地目录",
        "就医地医保目录",
        "就医地基本医疗保险支付范围",
        "执行就医地规定的支付范围",
        "按照就医地规定的支付范围",
    )
    insured_patterns = (
        "参保地待遇",
        "参保地起付标准",
        "参保地支付比例",
        "参保地最高支付限额",
        "参保地规定的起付标准",
        "参保地规定的支付比例",
        "参保地规定的最高支付限额",
        "执行参保地规定的起付标准",
        "执行参保地规定的支付比例",
        "执行参保地规定的最高支付限额",
        "参保地规定的基本医疗保险基金",
    )
    has_place_scope = (
        any(pattern in compact for pattern in place_patterns)
        or (
            "执行就医地" in compact
            and any(token in compact for token in ("支付范围", "药品目录", "诊疗项目", "耗材"))
        )
        or
        ("就医地" in compact and any(token in compact for token in ("支付范围", "药品", "服务项目", "耗材", "目录")))
    )
    has_insured_benefit = (
        any(pattern in compact for pattern in insured_patterns)
        or (
            "执行参保地" in compact
            and any(token in compact for token in ("起付标准", "支付比例", "最高支付限额", "待遇"))
        )
        or
        ("参保地" in compact and any(token in compact for token in ("起付", "支付比例", "最高支付限额", "待遇", "基金")))
    )
    return has_place_scope and has_insured_benefit


def _case_fact_label(key: str) -> str:
    return {
        "insured_region": "参保地",
        "treatment_region": "就医地",
        "visit_type": "就医类型",
        "claim_mode": "报销方式",
        "filing_status": "备案状态",
        "emergency_material_status": "急诊材料状态",
        "diagnosis": "诊断",
    }.get(key, key)


def _int_between(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, minimum), maximum)


def _float_between(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, minimum), maximum)


def _bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return fallback


def _optional_str(value: Any, limit: int) -> str | None:
    if value in (None, ""):
        return None
    return str(value)[:limit]


def _clip(value: Any, limit: int) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."
