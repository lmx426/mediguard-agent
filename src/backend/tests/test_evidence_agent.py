"""Minimum safety and stability tests for FEATURE-019."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.backend.application.agent.evidence_agent.safety import (
    AgentOutputValidationError,
    validate_generated_content,
    validate_generated_content_relaxed,
)
from src.backend.application.agent.evidence_agent.nodes.finalize import (
    _can_repair_validation_error,
)
from src.backend.application.agent.evidence_agent.prompts.loader import (
    build_validation_repair_prompt,
)
from src.backend.application.agent.evidence_agent.service import EvidenceAgentService
from src.backend.application.agent.evidence_agent.tools import (
    EvidenceAgentToolRegistry,
    ToolExecutionError,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse, ModelToolCall
from src.backend.application.agent.runtime.gateways.deepseek import DeepSeekModelGateway
from src.backend.application.agent.runtime.gateways.fake import ScriptedModelGateway
from src.backend.api.agent_autostart import prestart_evidence_agent
from src.backend.application.audit.review.build_materials_uc import StatisticalMaterialUseCase
from src.backend.core.config import Settings
from src.backend.domain.audit.review.evidence_packager import EvidenceService
from src.backend.domain.agent.entities import (
    AgentEvent,
    AgentEvidenceReview,
    AgentGeneratedContent,
    AgentRun,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
    EvidenceLedgerItem,
)
from src.backend.domain.audit.review.entities import AuthenticatedUser, CaseDetail, RuleHit
from src.backend.infrastructure.persistence.memory.case_repository import CaseService
from src.backend.infrastructure.persistence.memory.material_repository import MemoryMaterialRepository


def build_case(case_id: str = "CASE-AGENT-001") -> CaseDetail:
    return CaseDetail(
        case_id=case_id,
        case_title="Evidence Agent 测试案件",
        case_type="门诊",
        risk_level="medium",
        risk_score=0.5,
        review_status="pending",
        review_priority="manual_review",
        rule_signal_count=0,
        claim_amount=1000,
        claim_summary="脱敏聚合统计记录待人工核验。",
        rule_hits=[],
        expected_recommendation="建议人工核验统计材料。",
        model_signal_source="确定性风险信号引擎",
        model_signal_reasons=["费用结构需核验"],
        evidence_consistency="模型信号与规则证据待核验",
        input_features={
            "ALL_SUM": 1000,
            "月就诊次数_MAX": 4,
            "月就诊医院数_MAX": 1,
            "一天去两家医院的天数": 0,
            "药品在总金额中的占比": 0.5,
            "检查总费用在总金额占比": 0.3,
            "治疗费用在总金额占比": 0.2,
            "是否挂号": 1,
            "药品费发生金额_SUM": 500,
            "检查费发生金额_SUM": 300,
            "治疗费发生金额_SUM": 200,
        },
    )


def build_tools(cases: CaseService) -> EvidenceAgentToolRegistry:
    return EvidenceAgentToolRegistry(
        cases,
        EvidenceService(),
        StatisticalMaterialUseCase(
            material_repository=MemoryMaterialRepository(),
            asset_root=Path("tmp") / "test-agent-materials",
        ),
    )


def build_v2_final_payload(case: CaseDetail) -> dict[str, Any]:
    return {
        "status": "complete",
        "evidence_review": {
            "relation": "partially_supports",
            "label": "ignored",
            "support_level": "\u4e2d",
            "summary": "Available evidence supports manual review.",
            "source_refs": [case.model_evidence_ref],
        },
        "clue_reviews": [
            {
                "clue_id": "clue:model-warning",
                "title": "Model warning",
                "status": "needs_review",
                "explanation": "The warning should be checked against case evidence.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "verification_checklist": [
            {
                "title": "Check evidence",
                "action": "Review the available evidence package.",
                "rationale": "The advisory result is only for human verification.",
                "relation_type": "risk_score_related",
                "priority": "high",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [
            {
                "statement": "Source documents are not available in this dataset.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
    }


def wait_for_agent(repository: FakeAgentRepository, service: EvidenceAgentService) -> None:
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()


def build_actor() -> AuthenticatedUser:
    return AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )


def build_case_with_rules(case_id: str = "CASE-LEDGER-001") -> CaseDetail:
    case = build_case(case_id)
    return case.model_copy(
        update={
            "rule_hits": [
                RuleHit(
                    rule_id="OP-R001",
                    rule_name="命中规则",
                    hit=True,
                    severity="low",
                    reason="命中低强度核验规则。",
                    evidence_ref="rule:OP-R001",
                    version="1.0.0",
                    current_value="4",
                    threshold=">=3",
                ),
                RuleHit(
                    rule_id="OP-R002",
                    rule_name="高严重度未命中规则",
                    hit=False,
                    severity="high",
                    reason="未命中但属于高严重度规则，需要保留详情。",
                    evidence_ref="rule:OP-R002",
                    version="1.0.0",
                    current_value="2",
                    threshold=">=5",
                ),
                RuleHit(
                    rule_id="OP-R003",
                    rule_name="普通未命中规则",
                    hit=False,
                    severity="medium",
                    reason="未命中普通规则，只进入摘要。",
                    evidence_ref="rule:OP-R003",
                    version="1.0.0",
                    current_value="1",
                    threshold=">=8",
                ),
            ],
        }
    )


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(collect_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(collect_keys(child))
        return keys
    return set()


def test_tools_reject_cross_case_and_forbidden_fields() -> None:
    cases = CaseService()
    cases.add_case(build_case())
    tools = build_tools(cases)

    with pytest.raises(ToolExecutionError) as cross_case:
        tools.execute(
            current_case_id="CASE-AGENT-001",
            tool_name="get_risk_breakdown",
            raw_arguments={"case_id": "CASE-OTHER"},
        )
    assert cross_case.value.code == "cross_case_access"

    with pytest.raises(ToolExecutionError) as forbidden:
        tools.execute(
            current_case_id="CASE-AGENT-001",
            tool_name="get_case_fields",
            raw_arguments={"case_id": "CASE-AGENT-001", "fields": ["个人编码", "RES"]},
        )
    assert forbidden.value.code == "forbidden_field"


def test_evidence_ledger_prefetches_default_base_without_raw_materials() -> None:
    cases = CaseService()
    case = build_case_with_rules()
    cases.add_case(case)
    tools = build_tools(cases)

    result, ledger_items, args = tools.execute(
        current_case_id=case.case_id,
        tool_name="get_evidence_ledger",
        raw_arguments={"case_id": case.case_id},
    )

    data = result["data"]
    assert args == {"case_id": case.case_id}
    assert data["ledger_version"] == "evidence-ledger-v2"
    assert data["case_id"] == case.case_id
    assert data["system_risk_prompt"]["generated_by"] == "backend_risk_engine"
    assert data["clue_candidates"]
    assert data["normalized_facts"]
    assert data["generation_guidance"]["decision_owner"] == "model"
    assert data["generation_guidance"]["when_to_call_tools"]
    assert data["generation_guidance"]["when_not_to_call_tools"]
    assert {item["rule_id"] for item in data["drilldown_index"]["rules"]} == {
        "OP-R001", "OP-R002", "OP-R003"
    }
    assert len(ledger_items) == data["prefetch_summary"]["ledger_item_count"]

    material_keys = collect_keys(data["materials_available"])
    assert not {"content", "rows", "sha256", "preview_url", "size_bytes"} & material_keys

    safe_fields = data["safe_field_profile"]
    safe_field_names = {item["field_name"] for item in safe_fields["fields"]}
    assert "RES" not in safe_field_names
    assert "个人编码" not in safe_field_names
    assert "ALL_SUM" in safe_field_names
    assert safe_fields["field_count"] == 39


def test_tool_catalog_and_function_schemas_share_registry_metadata() -> None:
    cases = CaseService()
    case = build_case_with_rules()
    cases.add_case(case)
    tools = build_tools(cases)

    schema_names = {
        item["function"]["name"]
        for item in tools.schemas_for_model(stage="post_prefetch")
    }
    catalog = tools.catalog_for_prompt(stage="post_prefetch")
    catalog_names = {item["name"] for item in catalog["available_tools"]}

    assert schema_names == catalog_names == {
        "get_rule_details",
        "get_case_fields",
        "get_statistical_material",
    }
    assert catalog["prefetched_tools"][0]["name"] == "get_evidence_ledger"
    assert catalog["prefetched_tools"][0]["callable_by_model"] is False
    assert all(item["description"] for item in catalog["available_tools"])
    assert all(item["use_when"] for item in catalog["available_tools"])
    assert all(item["avoid_when"] for item in catalog["available_tools"])


def test_agent_feature_flag_fails_closed_without_affecting_base_api(client) -> None:
    status_response = client.get("/api/agent/status")
    assert status_response.status_code == 200
    assert status_response.json()["enabled"] is False
    assert status_response.json()["available"] is False

    start_response = client.post(
        "/api/cases/CASE-NOT-REQUIRED/evidence-agent/runs",
        json={"analysis_type": "comprehensive"},
    )
    assert start_response.status_code == 503
    assert client.get("/api/cases").status_code == 200


def test_evidence_agent_request_accepts_comprehensive_only() -> None:
    assert EvidenceAgentRequest().analysis_type == "comprehensive"
    assert EvidenceAgentRequest(eval_variant="B3").eval_variant == "B3"
    assert EvidenceAgentRequest(eval_variant="C1").eval_variant == "C1"

    with pytest.raises(ValidationError):
        EvidenceAgentRequest(
            analysis_type="focused",
            focus_type="rule",
            focus_ref="OP-R001",
        )

    with pytest.raises(ValidationError):
        EvidenceAgentRequest(eval_variant="B9")


def test_output_validation_closes_citations_and_blocks_decisions() -> None:
    ledger = {
        "risk:summary": EvidenceLedgerItem(
            ledger_ref="risk:summary",
            source_type="risk_breakdown",
            source_ref="model:score:v1.0",
            label="风险提示",
            value={"level": "medium"},
        )
    }
    valid = AgentGeneratedContent(
        status="complete",
        risk_overview="当前证据形成中等风险提示。",
        risk_overview_source_refs=["risk:summary"],
        supporting_evidence=[],
        conflicts=[],
        missing_information=[],
        signal_review={
            "case_review_hint": "当前线索与风险提示来源存在对应关系，仍需人工复核。",
            "case_review_hint_source_refs": ["risk:summary"],
            "supported_clues": [
                {
                    "statement": "当前线索已有风险提示来源支撑。",
                    "source_refs": ["risk:summary"],
                }
            ],
            "needs_review": [],
            "unconfirmed_items": [],
            "supplementary_review_hints": [],
        },
        verification_checklist=[],
        boundary_notice="仅供人工核验。",
    )
    citations = validate_generated_content(valid, ledger)
    assert citations[0].citation_id == "risk:summary"

    invalid_signal_ref = valid.model_copy(
        update={
            "signal_review": valid.signal_review.model_copy(
                update={"case_review_hint_source_refs": ["other:case"]}
            )
        }
    )
    with pytest.raises(AgentOutputValidationError) as signal_citation_error:
        validate_generated_content(invalid_signal_ref, ledger)
    assert signal_citation_error.value.code == "citation_invalid"

    invalid_ref = valid.model_copy(update={"risk_overview_source_refs": ["other:case"]})
    with pytest.raises(AgentOutputValidationError) as citation_error:
        validate_generated_content(invalid_ref, ledger)
    assert citation_error.value.code == "citation_invalid"

    prohibited = valid.model_copy(update={"risk_overview": "建议审核通过。"})
    with pytest.raises(AgentOutputValidationError) as decision_error:
        validate_generated_content(prohibited, ledger)
    assert decision_error.value.code == "prohibited_conclusion"

    rule_id_text = valid.model_copy(update={"risk_overview": "Rule OP-R001 should be reviewed."})
    validate_generated_content(rule_id_text, ledger)

    suspected_fraud_signal = valid.model_copy(
        update={"risk_judgement": "疑似欺诈线索较强，需审核人员结合材料继续核验。"}
    )
    validate_generated_content(suspected_fraud_signal, ledger)


def test_relaxed_output_validation_skips_citation_and_numeric_checks_only() -> None:
    ledger = {
        "risk:summary": EvidenceLedgerItem(
            ledger_ref="risk:summary",
            source_type="risk_breakdown",
            source_ref="model:score:v1.0",
            label="risk summary",
            value={"level": "medium"},
        )
    }
    content = AgentGeneratedContent(
        status="complete",
        risk_overview="The cited evidence contains 12345.",
        risk_overview_source_refs=["missing:ref"],
        supporting_evidence=[],
        conflicts=[],
        missing_information=[],
        verification_checklist=[],
        boundary_notice="Human review support only.",
    )

    assert validate_generated_content_relaxed(content, ledger) == []

    sensitive = content.model_copy(update={"risk_overview": "Do not expose sk-test."})
    with pytest.raises(AgentOutputValidationError) as sensitive_error:
        validate_generated_content_relaxed(sensitive, ledger)
    assert sensitive_error.value.code == "sensitive_output"

    risk_identifier = content.model_copy(
        update={"risk_overview": "Review clue-risk-summary before forming an opinion."}
    )
    validate_generated_content_relaxed(risk_identifier, ledger)


def test_sensitive_output_allows_one_redacted_controlled_rewrite() -> None:
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_max_model_calls=4,
    )
    error = AgentOutputValidationError(
        "sensitive_output",
        "Agent 输出包含禁止的敏感标记",
    )

    assert _can_repair_validation_error(
        {"validation_repair_count": 0, "model_call_count": 1},
        error,
        settings,
    )
    assert not _can_repair_validation_error(
        {"validation_repair_count": 1, "model_call_count": 2},
        error,
        settings,
    )
    repair_prompt = build_validation_repair_prompt(
        error_code=error.code,
        error_message=str(error),
        ledger={},
    )
    assert "身份核验事项" in repair_prompt
    assert "不要复述上一份回答" in repair_prompt


def test_evidence_review_label_is_derived_from_relation() -> None:
    review = AgentEvidenceReview(
        relation="partially_supports",
        label="任意描述性标题",
        support_level="中",
        summary="当前证据仅形成部分支撑。",
        source_refs=["risk:summary"],
    )

    assert review.label == "部分支持"


def test_deepseek_gateway_returns_redacted_metrics() -> None:
    prompt_text = "private prompt text"
    output_text = '{"status":"complete"}'
    reasoning_text = "private reasoning text"

    class _FakeCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=output_text,
                            reasoning_content=reasoning_text,
                            tool_calls=[],
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                ),
            )

    completions = _FakeCompletions()
    gateway = object.__new__(DeepSeekModelGateway)
    gateway._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    gateway._model = "deepseek-test"
    gateway._thinking_enabled = True
    gateway._reasoning_effort = "low"

    response = gateway.complete(
        messages=[{"role": "system", "content": prompt_text}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_evidence_ledger",
                    "parameters": {"type": "object"},
                },
            }
        ],
        require_json=True,
    )

    assert response.content == output_text
    assert response.reasoning_content == reasoning_text
    assert response.metrics["model"] == "deepseek-test"
    assert response.metrics["message_count"] == 1
    assert response.metrics["tool_schema_count"] == 1
    assert response.metrics["require_json"] is True
    assert response.metrics["thinking_enabled"] is True
    assert response.metrics["reasoning_effort"] == "low"
    assert response.metrics["output_chars"] == len(output_text)
    assert response.metrics["reasoning_chars"] == len(reasoning_text)
    assert response.metrics["prompt_tokens"] == 11
    assert response.metrics["completion_tokens"] == 7
    assert response.metrics["total_tokens"] == 18
    assert response.metrics["latency_ms"] >= 0

    metrics_blob = json.dumps(response.metrics, ensure_ascii=False)
    assert prompt_text not in metrics_blob
    assert output_text not in metrics_blob
    assert reasoning_text not in metrics_blob
    assert "reasoning_content" not in metrics_blob


class FakeAgentRepository:
    """Test double only; production has no in-memory Agent repository."""

    def __init__(self) -> None:
        self.run: AgentRun | None = None
        self.events: list[AgentEvent] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    def create_or_reuse_run(self, **kwargs: Any) -> AgentRun:
        now = datetime.now(timezone.utc)
        request = kwargs["request"]
        self.run = AgentRun(
            run_id=f"arun_{uuid4().hex}",
            case_id=kwargs["case_id"],
            actor_id=kwargs["actor_id"],
            analysis_type=request.analysis_type,
            status="queued",
            current_node="queued",
            input_fingerprint=kwargs["input_fingerprint"],
            created_at=now,
            updated_at=now,
        )
        return self.run

    def get_run(self, run_id: str) -> AgentRun | None:
        return self.run if self.run and self.run.run_id == run_id else None

    def get_latest_run(self, case_id: str, analysis_type: str = "comprehensive") -> AgentRun | None:
        if self.run and self.run.case_id == case_id and self.run.analysis_type == analysis_type:
            return self.run
        return None

    def get_current_analysis(self, *args: Any, **kwargs: Any) -> EvidenceAgentAnalysis | None:
        return self.run.analysis if self.run else None

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        assert self.run and self.run.run_id == run_id
        for key, value in kwargs.items():
            if value is not None and hasattr(self.run, key):
                setattr(self.run, key, value)
        self.run.updated_at = datetime.now(timezone.utc)

    def append_event(self, run_id: str, event_type: str, message: str, payload=None) -> AgentEvent:
        event = AgentEvent(
            run_id=run_id,
            sequence=len(self.events) + 1,
            event_type=event_type,
            message=message,
            payload=payload or {},
            created_at=datetime.now(timezone.utc),
        )
        self.events.append(event)
        return event

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[AgentEvent]:
        return [event for event in self.events if event.sequence > after_sequence]

    def save_checkpoint(self, run_id: str, node_name: str, state: dict, safe_to_resume=True) -> None:
        self.checkpoints.append(state)

    def record_tool_call(self, run_id: str, **kwargs: Any) -> None:
        self.tool_calls.append(kwargs)

    def find_tool_result(self, run_id: str, signature: str):
        return None

    def save_analysis(self, analysis: EvidenceAgentAnalysis) -> None:
        assert self.run
        self.run.analysis = analysis


class _FakePrestartAgent:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.case_ids: list[str] = []

    def start(
        self,
        case_id: str,
        _request: EvidenceAgentRequest,
        _actor: AuthenticatedUser,
    ) -> None:
        self.case_ids.append(case_id)
        if self.fail:
            raise RuntimeError("agent unavailable")


class _FakePrestartContainer:
    def __init__(
        self,
        agent: _FakePrestartAgent | None,
        *,
        eval_variants_enabled: bool = False,
    ) -> None:
        self.evidence_agent = agent
        self.settings = SimpleNamespace(
            evidence_agent_eval_variants_enabled=eval_variants_enabled
        )


class FailingLedgerTools:
    def __init__(self, wrapped: EvidenceAgentToolRegistry) -> None:
        self._wrapped = wrapped

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._wrapped.schemas

    def schemas_for_model(self, *, stage: str) -> list[dict[str, Any]]:
        return self._wrapped.schemas_for_model(stage=stage)

    def catalog_for_prompt(self, *, stage: str) -> dict[str, Any]:
        return self._wrapped.catalog_for_prompt(stage=stage)

    def is_model_callable(self, tool_name: str, *, stage: str) -> bool:
        return self._wrapped.is_model_callable(tool_name, stage=stage)

    def execute(self, **kwargs: Any):
        if kwargs["tool_name"] == "get_evidence_ledger":
            raise ToolExecutionError("ledger_failed", "ledger failed")
        return self._wrapped.execute(**kwargs)


def test_eval_variants_are_disabled_by_default() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
        ),
        cases=cases,
        repository=repository,
        gateway=ScriptedModelGateway([]),
        tools=build_tools(cases),
    )
    try:
        for variant in ("B1", "C1"):
            with pytest.raises(ValueError, match="eval variants are disabled"):
                service.start(
                    case.case_id,
                    EvidenceAgentRequest(eval_variant=variant),
                    build_actor(),
                )
        assert repository.run is None
    finally:
        service.shutdown()


def test_eval_variant_b1_disables_thinking_only_when_enabled() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    gateway = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(build_v2_final_payload(case), ensure_ascii=False))]
    )
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
            evidence_agent_eval_variants_enabled=True,
            evidence_agent_max_concurrency=1,
        ),
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )

    service.start(case.case_id, EvidenceAgentRequest(eval_variant="B1"), build_actor())
    wait_for_agent(repository, service)

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert gateway.requests[0]["thinking_enabled"] is False
    assert gateway.requests[0]["reasoning_effort"] is None
    assert gateway.requests[0]["tools"]
    assert gateway.requests[0]["require_json"] is False
    model_event = next(
        event for event in repository.events if event.event_type == "model_completed"
    )
    assert model_event.payload["eval_variant"] == "B1"


def test_eval_variant_b4_generates_json_without_model_tools() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    gateway = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(build_v2_final_payload(case), ensure_ascii=False))]
    )
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
            evidence_agent_eval_variants_enabled=True,
            evidence_agent_max_concurrency=1,
        ),
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )

    service.start(case.case_id, EvidenceAgentRequest(eval_variant="B4"), build_actor())
    wait_for_agent(repository, service)

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert gateway.requests[0]["tools"] == []
    assert gateway.requests[0]["require_json"] is True
    assert repository.run.tool_call_count == 1
    assert [call["tool_name"] for call in repository.tool_calls] == ["get_evidence_ledger"]


def test_eval_variant_b3_uses_lightweight_prompt_view_with_full_ledger_state() -> None:
    cases = CaseService()
    case = build_case_with_rules()
    cases.add_case(case)
    repository = FakeAgentRepository()
    gateway = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(build_v2_final_payload(case), ensure_ascii=False))]
    )
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
            evidence_agent_eval_variants_enabled=True,
            evidence_agent_max_concurrency=1,
        ),
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )

    service.start(case.case_id, EvidenceAgentRequest(eval_variant="B3"), build_actor())
    wait_for_agent(repository, service)

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    first_user_prompt = gateway.requests[0]["messages"][1]["content"]
    assert "lightweight_review_brief" in first_user_prompt
    assert "default_context" in first_user_prompt
    assert gateway.requests[0]["tools"]
    ledger_event = next(
        event for event in repository.events if event.event_type == "ledger_prefetched"
    )
    assert ledger_event.payload["eval_variant"] == "B3"
    assert ledger_event.payload["ledger_view"] == "lightweight"
    assert repository.checkpoints
    assert len(repository.checkpoints[0]["ledger"]) == ledger_event.payload["source_count"]


def test_eval_variant_c1_uses_review_brief_with_full_ledger_state() -> None:
    cases = CaseService()
    case = build_case_with_rules()
    cases.add_case(case)
    repository = FakeAgentRepository()
    gateway = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(build_v2_final_payload(case), ensure_ascii=False))]
    )
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
            evidence_agent_eval_variants_enabled=True,
            evidence_agent_max_concurrency=1,
        ),
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )

    service.start(case.case_id, EvidenceAgentRequest(eval_variant="C1"), build_actor())
    wait_for_agent(repository, service)

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    first_user_prompt = gateway.requests[0]["messages"][1]["content"]
    assert "task_sufficient_review_brief_v1" in first_user_prompt
    assert "citation_manifest" in first_user_prompt
    assert "context_representation" in first_user_prompt
    assert gateway.requests[0]["tools"]
    ledger_event = next(
        event for event in repository.events if event.event_type == "ledger_prefetched"
    )
    assert ledger_event.payload["eval_variant"] == "C1"
    assert ledger_event.payload["ledger_view"] == "review_brief"
    assert ledger_event.payload["review_brief_builder"] == "review-brief-v1"
    assert ledger_event.payload["brief_candidate_count"] is not None
    assert repository.checkpoints
    assert len(repository.checkpoints[0]["ledger"]) == ledger_event.payload["source_count"]


def test_prestart_evidence_agent_is_fail_open() -> None:
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )

    prestart_evidence_agent(_FakePrestartContainer(None), "CASE-001", actor)

    agent = _FakePrestartAgent()
    prestart_evidence_agent(_FakePrestartContainer(agent), ["CASE-001", "CASE-002"], actor)
    assert agent.case_ids == ["CASE-001", "CASE-002"]

    failing_agent = _FakePrestartAgent(fail=True)
    prestart_evidence_agent(_FakePrestartContainer(failing_agent), "CASE-003", actor)
    assert failing_agent.case_ids == ["CASE-003"]

    eval_agent = _FakePrestartAgent()
    prestart_evidence_agent(
        _FakePrestartContainer(eval_agent, eval_variants_enabled=True),
        "CASE-004",
        actor,
    )
    assert eval_agent.case_ids == []


def test_state_graph_uses_native_function_call_and_never_checkpoints_reasoning() -> None:
    cases = CaseService()
    case = build_case_with_rules()
    cases.add_case(case)
    repository = FakeAgentRepository()
    final_payload = {
        "status": "complete",
        "risk_overview": "当前存在中等风险提示，需由人工继续核验。",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "风险来源已由确定性风险记录支持。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": ["缺少可直接核对的原始申报材料。"],
        "verification_checklist": [
            {
                "title": "核对统计材料",
                "action": "由审核人员查看费用结构材料。",
                "rationale": "当前信息仅为脱敏聚合统计结果。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "仅用于辅助人工核验。",
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content="",
                reasoning_content="ephemeral reasoning that must never be persisted",
                tool_calls=[
                    ModelToolCall(
                        id="call_risk_1",
                        name="get_rule_details",
                        arguments=json.dumps({"case_id": case.case_id, "rule_id": "OP-R001"}),
                    )
                ],
            ),
            ModelResponse(content=json.dumps(final_payload, ensure_ascii=False)),
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="审核员",
        roles=["auditor"],
    )
    run = service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", (
        repository.run.error_code,
        repository.run.error_message,
        [event.event_type for event in repository.events],
    )
    assert repository.run.analysis is not None
    assert repository.tool_calls[0]["tool_call_id"] == "system_prefetch:get_evidence_ledger"
    assert repository.tool_calls[0]["tool_name"] == "get_evidence_ledger"
    assert repository.tool_calls[1]["tool_call_id"] == "call_risk_1"
    second_messages = gateway.requests[1]["messages"]
    assert any(message.get("reasoning_content") for message in second_messages)
    assert any(message.get("role") == "tool" and message.get("tool_call_id") == "call_risk_1" for message in second_messages)
    checkpoint_text = json.dumps(repository.checkpoints, ensure_ascii=False)
    assert "ephemeral reasoning" not in checkpoint_text
    assert "reasoning_content" not in checkpoint_text


def test_state_graph_can_generate_after_default_ledger_without_extra_tool_call() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    final_payload = {
        "status": "complete",
        "evidence_review": {
            "relation": "partially_supports",
            "label": "部分支持",
            "support_level": "中",
            "summary": "当前风险提示已有确定性证据支撑，但关键材料之间仍需人工穿透核对。",
            "source_refs": [case.model_evidence_ref],
        },
        "clue_reviews": [
            {
                "clue_id": "clue:model-warning",
                "title": "模型识别预警",
                "status": "needs_review",
                "explanation": "当前预警可以作为核验入口，仍需结合案件材料确认。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "verification_checklist": [
            {
                "title": "核对证据包",
                "action": "查看当前证据包和统计材料摘要。",
                "rationale": "辅助研判结果仅用于人工核验。",
                "relation_type": "risk_score_related",
                "priority": "high",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [
            {
                "statement": "缺少可直接核对的原始申报材料。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content=json.dumps(final_payload, ensure_ascii=False),
                reasoning_content="private ephemeral reasoning",
                metrics={
                    "model": "fake-model",
                    "latency_ms": 123.45,
                    "message_count": 2,
                    "input_chars": 2048,
                    "tool_schema_count": 3,
                    "tool_schema_chars": 512,
                    "require_json": False,
                    "thinking_enabled": True,
                    "reasoning_effort": "low",
                    "output_chars": 4096,
                    "reasoning_chars": 27,
                    "response_tool_call_count": 0,
                    "finish_reason": "stop",
                    "prompt_tokens": 700,
                    "completion_tokens": 300,
                    "total_tokens": 1000,
                    "content": "must not persist",
                },
            )
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert repository.run.model_call_count == 1
    assert repository.run.tool_call_count == 1
    model_events = [
        event for event in repository.events if event.event_type == "model_completed"
    ]
    assert len(model_events) == 1
    model_payload = model_events[0].payload
    assert model_payload["model_call_index"] == 1
    assert model_payload["latency_ms"] == 123.45
    assert model_payload["input_chars"] == 2048
    assert model_payload["output_chars"] == 4096
    assert model_payload["prompt_tokens"] == 700
    assert model_payload["completion_tokens"] == 300
    assert model_payload["total_tokens"] == 1000
    assert model_payload["response_tool_call_count"] == 0
    assert model_payload["require_json"] is False
    event_blob = json.dumps(model_payload, ensure_ascii=False)
    assert "private ephemeral reasoning" not in event_blob
    assert "must not persist" not in event_blob
    assert "reasoning_content" not in event_blob
    assert [call["tool_call_id"] for call in repository.tool_calls] == [
        "system_prefetch:get_evidence_ledger"
    ]
    assert gateway.requests[0]["tools"]
    assert "default_evidence_ledger" in gateway.requests[0]["messages"][1]["content"]
    assert "tool_catalog" in gateway.requests[0]["messages"][1]["content"]
    assert "generation_guidance" in gateway.requests[0]["messages"][1]["content"]
    assert repository.run.analysis is not None
    assert repository.run.analysis.system_risk_prompt is not None
    assert repository.run.analysis.system_risk_prompt.generated_by == "backend_risk_engine"
    assert repository.run.analysis.evidence_review is not None
    assert repository.run.analysis.evidence_review.relation == "partially_supports"


def test_default_ledger_failure_falls_back_to_base_evidence_package() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    final_payload = {
        "status": "complete",
        "risk_overview": "当前证据提示需由审核人员继续核验。",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "基础证据包仍提供当前案件风险提示来源。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [
            {
                "title": "核对基础证据",
                "action": "查看基础证据包中的风险提示和规则来源。",
                "rationale": "默认 Ledger 失败后已降级到最小证据上下文。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "仅用于辅助人工核验。",
    }
    gateway = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(final_payload, ensure_ascii=False))]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=FailingLedgerTools(build_tools(cases)),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert repository.run.tool_call_count == 2
    assert repository.tool_calls[0]["tool_call_id"] == "system_prefetch:get_evidence_ledger"
    assert repository.tool_calls[0]["status"] == "failed"
    assert repository.tool_calls[1]["tool_call_id"] == "system_prefetch:get_base_evidence_package"
    assert repository.tool_calls[1]["status"] == "success"
    assert any(event.event_type == "prefetch_failed" for event in repository.events)


def test_validation_error_gets_one_controlled_rewrite_without_tools() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    bad_payload = {
        "status": "complete",
        "risk_overview": "The cited evidence contains 0.123456.",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [],
        "boundary_notice": "Human review support only.",
    }
    repaired_payload = {
        "status": "complete",
        "risk_overview": "Current evidence should be checked by a human reviewer.",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "The risk evidence is grounded in the current case ledger.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [
            {
                "title": "Review evidence",
                "action": "Check the evidence package manually.",
                "rationale": "The Agent result is only an auxiliary summary.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "Human review support only.",
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_risk_1",
                        name="get_risk_breakdown",
                        arguments=json.dumps({"case_id": case.case_id}),
                    )
                ],
            ),
            ModelResponse(content=json.dumps(bad_payload, ensure_ascii=False)),
            ModelResponse(content=json.dumps(repaired_payload, ensure_ascii=False)),
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
        evidence_agent_max_model_calls=4,
        evidence_agent_strict_local_validation=True,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert any(event.event_type == "repairing" for event in repository.events)
    assert gateway.requests[2]["tools"] == []


def test_invalid_model_output_gets_one_structure_rewrite_without_tools() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    final_payload = {
        "status": "complete",
        "risk_overview": "Current evidence should be checked by a human reviewer.",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "The risk evidence is grounded in the current case ledger.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [
            {
                "title": "Review evidence",
                "action": "Check the evidence package manually.",
                "rationale": "The Agent result is only an auxiliary summary.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "Human review support only.",
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_risk_1",
                        name="get_risk_breakdown",
                        arguments=json.dumps({"case_id": case.case_id}),
                    )
                ],
            ),
            ModelResponse(content=json.dumps({"risk_overview": "missing required fields"})),
            ModelResponse(content=json.dumps(final_payload, ensure_ascii=False)),
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
        evidence_agent_max_model_calls=4,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert any(event.event_type == "repairing_structure" for event in repository.events)
    assert gateway.requests[2]["tools"] == []
    assert gateway.requests[2]["require_json"] is True


def test_prohibited_conclusion_gets_one_controlled_rewrite() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    bad_payload = {
        "status": "complete",
        "risk_overview": "建议审核通过。",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [],
        "boundary_notice": "Human review support only.",
    }
    repaired_payload = {
        "status": "complete",
        "risk_overview": "当前证据仅提示需要人工继续核验。",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "风险证据来自当前案件证据账本。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [
            {
                "title": "人工核验证据",
                "action": "由审核人员核对当前证据包。",
                "rationale": "Agent 结果仅作为辅助摘要。",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "Human review support only.",
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_risk_1",
                        name="get_risk_breakdown",
                        arguments=json.dumps({"case_id": case.case_id}),
                    )
                ],
            ),
            ModelResponse(content=json.dumps(bad_payload, ensure_ascii=False)),
            ModelResponse(content=json.dumps(repaired_payload, ensure_ascii=False)),
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
        evidence_agent_max_model_calls=4,
        evidence_agent_strict_local_validation=False,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert any(event.event_type == "repairing" for event in repository.events)
    assert gateway.requests[2]["tools"] == []
    assert gateway.requests[2]["require_json"] is True


def test_saturated_evidence_collection_forces_final_json_instead_of_failure() -> None:
    cases = CaseService()
    case = build_case()
    cases.add_case(case)
    repository = FakeAgentRepository()
    final_payload = {
        "status": "complete",
        "risk_overview": "Current evidence should be checked by a human reviewer.",
        "risk_overview_source_refs": [case.model_evidence_ref],
        "supporting_evidence": [
            {
                "statement": "The available evidence is grounded in the current case ledger.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "verification_checklist": [
            {
                "title": "Review evidence",
                "action": "Check the current evidence package manually.",
                "rationale": "The Agent result is only an auxiliary summary.",
                "source_refs": [case.model_evidence_ref],
            }
        ],
        "boundary_notice": "Human review support only.",
    }
    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_risk_1",
                        name="get_risk_breakdown",
                        arguments=json.dumps({"case_id": case.case_id}),
                    )
                ],
            ),
            ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_base_1",
                        name="get_base_evidence_package",
                        arguments=json.dumps({"case_id": case.case_id}),
                    )
                ],
            ),
            ModelResponse(content=json.dumps(final_payload, ensure_ascii=False)),
        ]
    )
    settings = Settings(
        persistence_backend="memory",
        auth_secret_key="x" * 32,
        default_auditor_password="password123",
        evidence_agent_enabled=True,
        evidence_agent_max_concurrency=1,
        evidence_agent_max_model_calls=4,
    )
    service = EvidenceAgentService(
        settings=settings,
        cases=cases,
        repository=repository,
        gateway=gateway,
        tools=build_tools(cases),
    )
    actor = AuthenticatedUser(
        id=str(uuid4()),
        username="auditor",
        display_name="auditor",
        roles=["auditor"],
    )
    service.start(case.case_id, EvidenceAgentRequest(), actor)
    deadline = time.monotonic() + 5
    while repository.run and repository.run.status not in {"complete", "partial", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    service.shutdown()

    assert repository.run is not None
    assert repository.run.status == "complete", repository.run.error_code
    assert any(event.event_type == "evidence_saturated" for event in repository.events)
    assert gateway.requests[2]["tools"] == []
    assert gateway.requests[2]["require_json"] is True


def test_reused_active_run_is_stale_after_timeout_window() -> None:
    cases = CaseService()
    service = EvidenceAgentService(
        settings=Settings(
            persistence_backend="memory",
            auth_secret_key="x" * 32,
            default_auditor_password="password123",
            evidence_agent_enabled=True,
            evidence_agent_timeout_seconds=30,
        ),
        cases=cases,
        repository=FakeAgentRepository(),
        gateway=ScriptedModelGateway([]),
        tools=build_tools(cases),
    )
    now = datetime.now(timezone.utc)
    stale = AgentRun(
        run_id="arun_stale",
        case_id="CASE-AGENT-001",
        actor_id=str(uuid4()),
        analysis_type="comprehensive",
        status="running",
        current_node="model",
        input_fingerprint="fingerprint",
        reused=True,
        created_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(minutes=5),
    )
    fresh = stale.model_copy(update={"run_id": "arun_fresh", "updated_at": now})
    new_run = stale.model_copy(update={"run_id": "arun_new", "reused": False})
    try:
        assert service._is_stale_reused_active_run(stale) is True
        assert service._is_stale_reused_active_run(fresh) is False
        assert service._is_stale_reused_active_run(new_run) is False
    finally:
        service.shutdown()
