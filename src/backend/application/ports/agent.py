"""Evidence Agent 仓储端口。

定义 Agent 运行状态、事件和分析结果的持久化契约。
"""

from __future__ import annotations

from typing import Any, Protocol

from ...domain.agent.entities import (
    AgentEvent,
    AgentRun,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
)


class EvidenceAgentRepository(Protocol):
    """PostgreSQL-backed formal Agent state."""

    def create_or_reuse_run(
        self,
        *,
        case_id: str,
        actor_id: str,
        request: EvidenceAgentRequest,
        input_fingerprint: str,
        config_fingerprint: str,
        model_name: str,
        prompt_version: str,
        tool_version: str,
    ) -> AgentRun: ...

    def get_run(self, run_id: str) -> AgentRun | None: ...

    def get_latest_run(
        self,
        case_id: str,
        analysis_type: str = "comprehensive",
    ) -> AgentRun | None: ...

    def get_current_analysis(
        self,
        case_id: str,
    ) -> EvidenceAgentAnalysis | None: ...

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_node: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        model_call_count: int | None = None,
        tool_call_count: int | None = None,
    ) -> None: ...

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent: ...

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[AgentEvent]: ...

    def save_checkpoint(
        self,
        run_id: str,
        node_name: str,
        state: dict[str, Any],
        safe_to_resume: bool = True,
    ) -> None: ...

    def record_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any] | None,
        status: str,
        error_code: str | None,
        latency_ms: int,
    ) -> None: ...

    def find_tool_result(self, run_id: str, signature: str) -> dict[str, Any] | None: ...

    def save_analysis(self, analysis: EvidenceAgentAnalysis) -> None: ...
