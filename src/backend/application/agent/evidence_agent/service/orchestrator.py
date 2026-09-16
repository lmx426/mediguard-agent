"""Review Advisor 状态图组装与服务入口。

将 LangGraph StateGraph 的节点、路由和 ReviewAdvisorService 组装为
可对外暴露的受控 Agent 服务。Agent 不可用或输出校验失败时，
基础证据包继续可用（fail-closed）。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.backend.application.agent.evidence_agent.graph.builder import (
    build_review_advisor_graph,
)
from src.backend.application.agent.evidence_agent.graph.state import AgentState
from src.backend.application.agent.evidence_agent.tools.registry import (
    ReviewAdvisorToolRegistry,
)
from src.backend.application.ports.agent import EvidenceAgentRepository
from src.backend.application.ports.agent_gateway import ModelGateway
from src.backend.application.ports.repositories import CaseRepository
from src.backend.core.config import Settings
from src.backend.domain.agent.entities import (
    AgentRun,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
)
from src.backend.domain.audit.review.entities import AuthenticatedUser

if TYPE_CHECKING:
    from src.backend.application.agent.case_agent.context.refresh_service import (
        CaserContextRefreshService,
    )


class ReviewAdvisorService:
    """受控 Review Advisor 应用服务。

    提供自动预生成、人工重新生成、状态查询、当前分析读取和 SSE 事件流能力。
    Agent 运行受多层预算约束（模型调用、工具调用、规划轮数、总超时），
    输出必须通过五道本地安全校验。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        cases: CaseRepository,
        repository: EvidenceAgentRepository,
        gateway: ModelGateway,
        tools: ReviewAdvisorToolRegistry,
    ) -> None:
        """初始化 Review Advisor 服务。

        参数:
            settings: 应用配置，包含所有 Agent 预算阈值。
            cases: 案件仓储，用于读取案件详情。
            repository: Agent 持久化仓储（仅 PostgreSQL 实现）。
            gateway: LLM 模型网关。
            tools: Review Advisor 只读业务工具注册表。
        """

        self._settings = settings
        self._cases = cases
        self._repository = repository
        self._gateway = gateway
        self._tools = tools
        self._run_requests: dict[str, dict[str, Any]] = {}
        self._caser_context_events: CaserContextRefreshService | None = None
        # Agent 运行在线程池中异步执行，不阻塞 HTTP 请求
        self._executor = ThreadPoolExecutor(
            max_workers=settings.evidence_agent_max_concurrency,
            thread_name_prefix="review-advisor",
        )
        # 构建 LangGraph 状态图
        self._graph = build_review_advisor_graph(self)

    def set_caser_context_events(
        self,
        events: CaserContextRefreshService | None,
    ) -> None:
        """Attach the optional Caser context refresh publisher."""

        self._caser_context_events = events

    # ── 公开 API ──────────────────────────────────────

    def start(
        self,
        case_id: str,
        request: EvidenceAgentRequest,
        actor: AuthenticatedUser,
    ) -> AgentRun:
        """启动或复用一次综合证据研判。

        参数:
            case_id: 案件唯一标识。
            request: 综合分析请求。
            actor: 触发分析的审核人员。

        返回:
            AgentRun: 创建或复用的运行记录。

        异常:
            KeyError: 案件不存在。
        """

        case = self._cases.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        self._ensure_eval_variant_allowed(request)

        # 计算输入和配置指纹用于幂等复用
        input_fingerprint = self._input_fingerprint(case.model_dump(mode="json"))
        config_fingerprint = self._config_fingerprint(request.eval_variant)

        run = self._repository.create_or_reuse_run(
            case_id=case_id,
            actor_id=actor.id,
            request=request,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            model_name=self._settings.llm_model,
            prompt_version=self._settings.evidence_agent_prompt_version,
            tool_version=self._settings.evidence_agent_tool_version,
        )

        # 复用已有结果，不重复运行
        if run.reused:
            return run
        self._run_requests[run.run_id] = request.model_dump(mode="json")

        self._repository.append_event(
            run.run_id,
            "run_created",
            "Review Advisor run created",
            {
                "analysis_type": request.analysis_type,
                "eval_variant": self._effective_eval_variant(request),
            },
        )
        self._executor.submit(self._execute, run.run_id)
        return self._repository.get_run(run.run_id) or run

    def get_run(self, run_id: str) -> AgentRun | None:
        """查询一次 Agent 运行的当前状态。

        参数:
            run_id: Agent 运行唯一标识。

        返回:
            AgentRun | None: 运行记录，不存在时返回 None。
        """

        return self._repository.get_run(run_id)

    def get_latest_run(
        self,
        case_id: str,
        analysis_type: str = "comprehensive",
    ) -> AgentRun | None:
        """返回当前案件最近一次或正在执行的 Agent 运行。

        参数:
            case_id: 案件唯一标识。
            analysis_type: 分析类型，当前仅支持 comprehensive。

        返回:
            AgentRun | None: 优先返回 queued/running 运行，否则返回最新运行。
        """

        run = self._repository.get_latest_run(case_id, analysis_type)
        if run is None:
            return None
        if run.analysis is not None and not self._analysis_matches_current_config(run.analysis):
            run.analysis = None
            if run.status not in {"queued", "running"}:
                return None
        return run

    def get_current_analysis(
        self,
        case_id: str,
    ) -> EvidenceAgentAnalysis | None:
        """返回当前案件最新的综合证据研判结果。

        参数:
            case_id: 案件唯一标识。

        返回:
            EvidenceAgentAnalysis | None: 最新分析，不存在时返回 None。
        """

        analysis = self._repository.get_current_analysis(case_id)
        if analysis is None or not self._analysis_matches_current_config(analysis):
            return None
        return analysis

    def list_events(self, run_id: str, after_sequence: int = 0):
        """列出指定游标之后的事件（用于 SSE 推送）。

        参数:
            run_id: Agent 运行唯一标识。
            after_sequence: 从此序号之后开始获取。

        返回:
            list[AgentEvent]: 事件列表。
        """

        return self._repository.list_events(run_id, after_sequence)

    def shutdown(self) -> None:
        """优雅关闭线程池。"""

        self._executor.shutdown(wait=False, cancel_futures=False)

    def _analysis_matches_current_config(self, analysis: EvidenceAgentAnalysis) -> bool:
        """Return whether a saved analysis was generated by the active Agent config."""

        return (
            analysis.model_name == self._settings.llm_model
            and analysis.prompt_version == self._settings.evidence_agent_prompt_version
            and analysis.tool_version == self._settings.evidence_agent_tool_version
        )

    def _is_stale_reused_active_run(self, run: AgentRun) -> bool:
        """检查复用的活跃运行是否已过期。

        参数:
            run: Agent 运行记录。

        返回:
            bool: 运行状态为 running 且超过超时窗口则为 True。
        """

        if not run.reused or run.status != "running":
            return False
        now = datetime.now(timezone.utc)
        stale_seconds = (now - run.updated_at).total_seconds()
        return stale_seconds > self._settings.evidence_agent_timeout_seconds

    # ── 内部执行引擎 ──────────────────────────────────

    def _execute(self, run_id: str) -> None:
        """在线程池中执行一次完整的 Agent 运行。

        参数:
            run_id: Agent 运行唯一标识。
        """

        run = self._repository.get_run(run_id)
        if run is None:
            return

        state: AgentState = {
            "run_id": run_id,
            "case_id": run.case_id,
            "request": self._run_requests.get(
                run_id,
                EvidenceAgentRequest(
                    analysis_type=run.analysis_type,
                ).model_dump(mode="json"),
            ),
            "input_fingerprint": run.input_fingerprint,
            "started_monotonic": time.monotonic(),
            "messages": [],
            "ledger": {},
            "model_call_count": 0,
            "tool_call_count": 0,
            "planning_rounds": 0,
            "argument_error_count": 0,
            "repeat_count": 0,
            "no_new_evidence_rounds": 0,
            "structure_repair_count": 0,
            "validation_repair_count": 0,
            "seen_signatures": [],
            "force_final_json": False,
            "next_action": "load",
        }

        try:
            self._graph.invoke(state)
        except Exception as exc:
            # fail-closed：绝不暴露堆栈或模型原始输出
            self._fail(
                run_id,
                "agent_runtime_error",
                f"Agent runtime failed: {exc.__class__.__name__}",
            )

    # ── 辅助方法 ──────────────────────────────────────

    def _ensure_eval_variant_allowed(self, request: EvidenceAgentRequest) -> None:
        if (
            request.eval_variant not in {None, "A0"}
            and not self._settings.evidence_agent_eval_variants_enabled
        ):
            raise ValueError("Evidence Agent eval variants are disabled")

    def _effective_eval_variant(self, request: EvidenceAgentRequest) -> str:
        if not self._settings.evidence_agent_eval_variants_enabled:
            return "A0"
        return request.eval_variant or "A0"

    def _fail(self, run_id: str, code: str, message: str) -> None:
        """标记运行失败并记录错误事件。

        参数:
            run_id: Agent 运行唯一标识。
            code: 错误代码。
            message: 错误描述（截断到 500 字符）。
        """

        self._repository.update_run(
            run_id,
            status="failed",
            current_node="failed",
            error_code=code,
            error_message=message[:500],
        )
        self._repository.append_event(
            run_id,
            "failed",
            "Review Advisor run failed; base evidence remains available",
            {"error_code": code},
        )

    def _publish_caser_context_event(self, event_type: str, case_id: str) -> None:
        if self._caser_context_events is None:
            return
        self._caser_context_events.publish(
            event_type,
            case_id,
            reason="review advisor run state changed",
        )

    def _checkpoint(self, state: AgentState, node_name: str) -> None:
        """保存安全的运行快照。

        注意：绝不持久化 messages 和 pending_response，
        因为它们可能包含 reasoning_content。

        参数:
            state: 当前运行时状态。
            node_name: 当前节点名。
        """

        safe_state = {
            "case_id": state.get("case_id"),
            "request": state.get("request"),
            "input_fingerprint": state.get("input_fingerprint"),
            "ledger": state.get("ledger"),
            "model_call_count": state.get("model_call_count"),
            "tool_call_count": state.get("tool_call_count"),
            "planning_rounds": state.get("planning_rounds"),
            "seen_signatures": state.get("seen_signatures"),
            "node": node_name,
        }
        self._repository.save_checkpoint(state["run_id"], node_name, safe_state)

    def _timed_out(self, state: AgentState) -> bool:
        """检查运行是否超过总超时预算。

        参数:
            state: 当前运行时状态。

        返回:
            bool: 是否超时。
        """

        elapsed = time.monotonic() - state.get("started_monotonic", 0)
        return elapsed >= self._settings.evidence_agent_timeout_seconds

    @staticmethod
    def _input_fingerprint(case: dict[str, Any]) -> str:
        """计算案件输入指纹，用于判断是否需要重新运行。

        主动剔除个人编码字段后再计算哈希，
        确保脱敏字段不参与指纹。

        参数:
            case: 案件字典。

        返回:
            str: SHA-256 十六进制指纹。
        """

        source = dict(case.get("source_record") or {})
        # 主动剔除个人编码字段
        source.pop("个人编码", None)
        safe = {
            "case_id": case.get("case_id"),
            "risk_score": case.get("risk_score"),
            "risk_level": case.get("risk_level"),
            "risk_breakdown": case.get("risk_score_breakdown"),
            "model_warning": case.get("fraud_screening"),
            "rules": case.get("rule_hits"),
            "source_record": source,
        }
        return hashlib.sha256(
            json.dumps(
                safe, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()

    def _config_fingerprint(self, eval_variant: str | None = None) -> str:
        """计算 Agent 配置指纹。

        当 LLM provider、model、prompt/tool 版本或预算参数变化时，
        指纹变化会触发重新运行。

        返回:
            str: SHA-256 十六进制指纹。
        """

        config = {
            "provider": self._settings.llm_provider,
            "base_url": self._settings.llm_base_url,
            "model": self._settings.llm_model,
            "prompt": self._settings.evidence_agent_prompt_version,
            "tools": self._settings.evidence_agent_tool_version,
            "thinking": self._settings.llm_thinking_enabled,
            "effort": self._settings.llm_reasoning_effort,
            "strict_local_validation": self._settings.evidence_agent_strict_local_validation,
            "budgets": [
                self._settings.evidence_agent_max_model_calls,
                self._settings.evidence_agent_max_tool_calls,
                self._settings.evidence_agent_max_planning_rounds,
            ],
        }
        if self._settings.evidence_agent_eval_variants_enabled:
            config["eval_variant"] = eval_variant or "A0"
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


EvidenceAgentService = ReviewAdvisorService
