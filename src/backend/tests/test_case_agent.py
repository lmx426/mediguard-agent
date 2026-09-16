"""Case Agent lightweight framework tests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.backend.application.agent.case_agent.artifacts import (
    artifact_results,
    clear_runtime_artifacts,
    dependency_artifacts,
    write_step_artifact,
)
from src.backend.application.agent.case_agent.intent_examples import IntentExampleMatcher
from src.backend.application.audit.review.build_materials_uc import StatisticalMaterialUseCase
from src.backend.application.audit.review.get_case_detail_uc import GetCaseDetailUseCase
from src.backend.application.agent.case_agent.nodes.persist_result import (
    persist_result_node,
)
from src.backend.application.agent.case_agent.nodes.call_capabilities import (
    call_capabilities_node,
)
from src.backend.application.agent.case_agent.nodes.business_semantic_planner import (
    business_semantic_planner_node,
    _drug_price_reference_filters,
)
from src.backend.application.agent.case_agent.nodes.decision import (
    decision_precheck_node,
    decision_readiness_checker_node,
    heuristic_planner_node,
    load_perceptual_state_node,
    llm_planner_node,
    plan_normalizer_node,
    rule_based_planner_node,
)
from src.backend.application.agent.case_agent.nodes.build_answer_context import (
    build_answer_context_node,
)
from src.backend.application.agent.case_agent.nodes.execution_dag import (
    call_expert_analysis_node,
    dag_executor_node,
    dispatch_step_node,
    execute_l1_step_node,
    execution_dag_builder_node,
    materialize_expert_task_node,
)
from src.backend.application.agent.case_agent.nodes.generate_answer import (
    _reconcile_policy_output,
    generate_answer_node,
)
from src.backend.application.agent.case_agent.nodes.memory_runtime import (
    build_memory_retrieval_plan_node,
    start_request_memory_prefetch_node,
)
from src.backend.application.agent.case_agent.nodes.perception import (
    build_perceptual_state_node,
    confidence_calibrator_node,
    context_need_resolver_node,
    early_entity_extractor_node,
    fast_rule_entity_perception_node,
    input_ingestion_node,
    intent_example_biencoder_node,
    llm_semantic_parser_node,
    perception_contract_builder_node,
    perception_gssc_node,
    readiness_signal_builder_node,
    rule_precheck_node,
    semantic_intent_perception_node,
    slot_merger_node,
    state_normalizer_node,
)
from src.backend.application.agent.case_agent.nodes.resolve_answer_policy import (
    resolve_answer_policy_node,
)
from src.backend.application.agent.case_agent.nodes.resolve_answer_style import (
    resolve_answer_style_node,
)
from src.backend.application.agent.case_agent.nodes.validate_execution_plan import (
    validate_execution_plan_node,
)
from src.backend.application.agent.case_agent.nodes.validate_answer import (
    validate_answer_node,
)
from src.backend.application.agent.case_agent.context.section_builder import CaserSectionBuilder
from src.backend.application.agent.case_agent.prompts.planning import (
    build_business_semantic_planning_prompt,
)
from src.backend.application.agent.case_agent.prompts.final_answer import (
    build_final_answer_prompt,
)
from src.backend.application.agent.case_agent.safety import (
    CaseAgentSafetyError,
    assert_safe_text,
    validate_answer,
)
from src.backend.application.agent.case_agent.schemas.plan import (
    CaserBusinessSemanticPlan,
)
from src.backend.application.agent.case_agent.service import CaseAgentService
from src.backend.application.agent.case_agent.shortcut import (
    CaseAgentShortcut,
    build_shortcut_answer,
    detect_shortcut,
)
from src.backend.application.agent.case_agent.tools import CaseAgentToolRegistry
from src.backend.application.agent.case_agent.tools.registry import (
    CaseAgentToolResult,
    capability_manifest_for_prompt,
    manifest_item_for_capability,
    normalize_capability_name,
)
from src.backend.domain.case_memory.entities import MemoryHintPack, MemoryType
from src.backend.application.agent.expert_agent import ExpertAgentService
from src.backend.application.agent.expert_agent.service.orchestrator import (
    ExpertAgentResult,
    _adaptive_policy_retrieval_budget,
    _infer_information_needs_from_question,
)
from src.backend.application.agent.expert_agent.schemas import (
    AnswerabilityCheck,
    ExpertAnalysisTask,
    PolicyEvidence,
    PolicySearchRequest,
)
from src.backend.application.agent.expert_agent.service.policy_claim_composer import (
    build_claim_first_policy_answer,
)
from src.backend.application.agent.expert_agent.service.policy_fact_extractors import (
    extract_facts_for_requirements,
)
from src.backend.application.agent.expert_agent.service.policy_answer_contracts import (
    AnswerRequirement,
    ExtractedFact,
    NormalizedPolicyEvidence,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    PolicyFilterResolver,
    PolicyFilterResolverInput,
)
from src.backend.application.agent.expert_agent.service.policy_answer_slots import (
    resolve_answer_requirements,
)
from src.backend.application.agent.expert_agent.service.policy_slot_window import (
    build_sentence_windows,
    extract_verified_facts_from_judgement,
    select_candidate_windows,
)
from src.backend.application.agent.expert_agent.tools.policy_rag_mcp import (
    LazyPolicyRagMcpClient,
    call_policy_rag,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.application.agent.runtime.gateways.fake import ScriptedModelGateway
from src.backend.domain.case_agent.entities import (
    CaseAgentAnswer,
    CaseAgentCitation,
    CaseAgentClaim,
    CaseAgentContentBlock,
    CaseAgentMessage,
    CaseAgentSource,
    CaseAgentSourceDetail,
    CaseAgentSourceDetailField,
)
from src.backend.domain.caser_context.entities import (
    CASER_SECTION_SCHEMA_VERSION,
    CaserContextSection,
)
from src.backend.domain.audit.review.entities import CaseDetail, RuleHit
from src.backend.domain.audit.review.evidence_packager import EvidenceService
from src.backend.infrastructure.persistence.memory.case_repository import CaseService
from src.backend.infrastructure.persistence.memory.material_repository import MemoryMaterialRepository
from src.backend.infrastructure.persistence.memory.note_repository import MemoryNoteRepository
from src.backend.infrastructure.persistence.memory.review_repository import MemoryReviewRepository
from src.backend.infrastructure.persistence.memory.trace_repository import TraceService


def _case(
    case_id: str = "CASE-CASEAGENT-001",
    *,
    case_context: dict[str, object] | None = None,
) -> CaseDetail:
    return CaseDetail(
        case_id=case_id,
        case_title="Case Agent 测试案件",
        case_type="门诊",
        risk_level="medium",
        risk_score=0.5,
        review_status="pending",
        rule_signal_count=1,
        claim_amount=1000,
        claim_summary="脱敏统计记录提示需要人工核验。",
        rule_hits=[
            RuleHit(
                rule_id="OP-R001",
                rule_name="高频就诊核验",
                hit=True,
                severity="medium",
                reason="月就诊次数偏高。",
                evidence_ref="rule:OP-R001:visit_frequency",
                version="1.0.0",
                current_value="8",
                threshold=">=6",
            )
        ],
        expected_recommendation="建议人工核验。",
        model_signal_source="确定性风险信号引擎",
        model_signal_reasons=["就诊频次需核验"],
        evidence_consistency="模型信号与规则证据待核验",
        input_features={"ALL_SUM": 1000, "月就诊次数_MAX": 8},
        case_context=case_context or {},
    )


def _tool_registry(
    case_id: str = "CASE-CASEAGENT-001",
    *,
    evidence_agent: object | None = None,
    policy_knowledge: object | None = None,
    case_context: dict[str, object] | None = None,
) -> CaseAgentToolRegistry:
    cases = CaseService()
    case = _case(case_id, case_context=case_context)
    cases.add_case(case, evidence_package=EvidenceService().generate(case))
    get_detail = GetCaseDetailUseCase(
        cases,
        MemoryReviewRepository(),
        MemoryNoteRepository(),
        TraceService(),
        EvidenceService(),
    )
    statistical_materials = StatisticalMaterialUseCase(
        material_repository=MemoryMaterialRepository(),
        asset_root=Path("tmp/test-case-agent-materials"),
    )
    builder = CaserSectionBuilder(
        get_case_detail=get_detail,
        statistical_materials=statistical_materials,
        evidence_agent=evidence_agent,
    )
    return CaseAgentToolRegistry(
        caser_context=_FakeCaserContext(case_id, builder),
        policy_knowledge=policy_knowledge,
    )


class _FakeCaserContext:
    def __init__(self, case_id: str, builder: CaserSectionBuilder) -> None:
        self.case_id = case_id
        self.sections: dict[str, CaserContextSection] = {}
        for section_key in builder.section_keys:
            built = builder.build(case_id, section_key)
            self.sections[section_key] = CaserContextSection(
                case_id=case_id,
                section_key=section_key,
                schema_version=CASER_SECTION_SCHEMA_VERSION,
                status=built.status,
                completeness=built.completeness,
                payload=built.payload,
                payload_hash=built.payload_hash,
                source_refs=built.source_refs,
                source_versions=built.source_versions,
                generated_by=built.generated_by,
            )

    def get_section(self, case_id: str, section_key: str) -> CaserContextSection:
        if case_id != self.case_id:
            raise KeyError(case_id)
        return self.sections[section_key]


def test_case_agent_fails_closed_when_disabled(client, generated_case_id) -> None:
    status = client.get("/api/case-agent/status")
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert status.json()["available"] is False

    response = client.post(
        f"/api/cases/{generated_case_id}/case-agent/sessions",
        json={"title": "测试会话"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "feature_disabled"


def test_case_agent_normalizes_l1_section_key_to_capability_name() -> None:
    case_id = "CASE-CASEAGENT-001"
    tools = _tool_registry(case_id)

    assert normalize_capability_name("claimant_profile") == "query_claimant_profile"

    result = tools.execute(
        current_case_id=case_id,
        tool_name="claimant_profile",
        raw_arguments={"case_id": case_id},
    )

    assert result.status == "success"
    assert result.payload["capability"] == "query_claimant_profile"
    assert result.payload["section_key"] == "claimant_profile"


def test_case_agent_capability_manifest_guides_planner_with_business_fields() -> None:
    manifest = capability_manifest_for_prompt()
    claimant = manifest_item_for_capability("query_claimant_profile")

    assert claimant is not None
    assert claimant.section_key == "claimant_profile"
    assert "性别" in claimant.fields
    assert "申报人参保类型" in claimant.examples
    assert any(item["capability"] == "query_claimant_profile" for item in manifest)

    prompt = build_business_semantic_planning_prompt("CASE-001")
    assert "query_claimant_profile" in prompt
    assert "申报人性别是什么" in prompt
    assert "section_key 只是内部 View Store 分区" in prompt
    assert '"answer_mode"' not in prompt
    assert '"requires_citation"' not in prompt


def test_case_agent_planner_schema_uses_weak_intent_and_step_plan() -> None:
    plan = CaserBusinessSemanticPlan.model_validate(
        {
            "query_semantics": {
                "intent": "case_task",
                "user_goal": "查询申报人性别",
                "granularity": "single_field",
                "missing_slots": [],
            },
            "execution_plan": [
                {
                    "step": 1,
                    "layer": "L1",
                    "capability": "query_claimant_profile",
                    "arguments": {"case_id": "CASE-001"},
                    "reason": "申报人性别属于申报人基础信息",
                }
            ],
        }
    )

    assert plan.query_semantics.intent == "case_task"
    assert plan.query_semantics.granularity == "single_field"
    assert plan.execution_plan[0].step == 1
    assert plan.execution_plan[0].depends_on == []
    assert not hasattr(plan.query_semantics, "answer_mode")
    assert not hasattr(plan.query_semantics, "requires_citation")


def test_case_agent_planner_routes_case_context_short_field_question_without_llm() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("deterministic field routing should not call the LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_plan_context", case_id="CASE-001"),
        "user_message": SimpleNamespace(content="在哪里参保的"),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "case_task"
    assert next_state["query_semantics"]["granularity"] == "single_field"
    assert next_state["execution_plan"] == [
        {
            "step": 1,
            "layer": "L1",
            "capability": "query_case_basic_info",
            "arguments": {"case_id": "CASE-001"},
            "depends_on": [],
            "reason": "该问题可由当前案件的结构化字段直接回答。",
        }
    ]
    assert next_state["next_action"] == "validate_execution_plan"


def test_case_agent_perception_decision_chain_keeps_simple_l1_question_deterministic() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    service = SimpleNamespace(_repository=PlannerRepository())
    state = {
        "run": SimpleNamespace(
            run_id="crun_perception_decision",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            session_summary="",
            current_topic="",
            referenced_source_refs=[],
        ),
        "user_message": SimpleNamespace(
            message_id="msg-001",
            content="在哪里参保的",
            active_stage=None,
        ),
        "recent_messages": [],
        "actor_id": "actor-001",
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    for node in (
        input_ingestion_node,
        early_entity_extractor_node,
        semantic_intent_perception_node,
        rule_precheck_node,
        slot_merger_node,
        context_need_resolver_node,
        perception_contract_builder_node,
        state_normalizer_node,
        confidence_calibrator_node,
        readiness_signal_builder_node,
        load_perceptual_state_node,
        decision_readiness_checker_node,
        decision_precheck_node,
        rule_based_planner_node,
        plan_normalizer_node,
    ):
        state = node(service, state)

    assert state["model_call_count"] == 0
    assert state["perceptual_state"]["readiness_signal"]["ready_to_plan"] is True
    assert state["perceptual_state"]["decision_context_ref"] == ""
    assert state["execution_plan"][0]["capability"] == "query_case_basic_info"
    assert state["next_action"] == "validate_execution_plan"


def test_case_agent_fast_rule_path_skips_vector_llm_and_task_memory() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            pass

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            pass

    class ForbiddenMatcher:
        def match(self, _text: str) -> dict:
            raise AssertionError("rule hit must not call the intent vector matcher")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _intent_biencoder=ForbiddenMatcher(),
    )
    state = {
        "run": SimpleNamespace(
            run_id="crun_fast_rule",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            session_summary="",
            current_topic="",
            referenced_source_refs=[],
        ),
        "recent_messages": [],
        "actor_id": "actor-001",
        "user_message": SimpleNamespace(
            message_id="msg-001",
            content="在哪里参保的",
            active_stage=None,
        ),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    state = fast_rule_entity_perception_node(service, state)
    assert state["next_action"] == "build_perceptual_state"
    state = build_perceptual_state_node(service, state)
    state = build_memory_retrieval_plan_node(service, state)

    assert state["model_call_count"] == 0
    assert state["memory_retrieval_plan"]["target_layer"] == "L1"
    assert state["memory_retrieval_plan"]["items"] == []
    assert state["perception_metrics"]["total_latency_ms"] >= 0


@pytest.mark.parametrize(
    ("target_layer", "expected_type", "expected_wait_ms"),
    [
        ("L1", None, None),
        ("L2", "decision_plan_hint", 300),
        ("L3", "policy_search_hint", 500),
    ],
)
def test_case_agent_memory_plan_is_deterministic_by_final_layer(
    target_layer: str,
    expected_type: str | None,
    expected_wait_ms: int | None,
) -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            pass

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            pass

    state = {
        "run": SimpleNamespace(run_id=f"crun-plan-{target_layer}"),
        "perceptual_state": {"semantic": {"target_layer_hint": target_layer}},
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    result = build_memory_retrieval_plan_node(
        SimpleNamespace(_repository=PlannerRepository()),
        state,
    )

    items = result["memory_retrieval_plan"]["items"]
    if expected_type is None:
        assert items == []
    else:
        assert items[0]["memory_type"] == expected_type
        assert items[0]["wait_timeout_ms"] == expected_wait_ms


def test_case_agent_perceptual_state_event_precedes_memory_prefetch_start() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[str] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            pass

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(event_type)

    repository = PlannerRepository()
    service = SimpleNamespace(
        _repository=repository,
        start_request_memory_prefetch=lambda _state: {
            "enabled": True,
            "types": {"policy_search_hint": "in_flight"},
        },
    )
    state = {
        "run": SimpleNamespace(
            run_id="crun-order",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            session_summary="",
            current_topic="",
            referenced_source_refs=[],
        ),
        "recent_messages": [],
        "actor_id": "actor-001",
        "user_message": SimpleNamespace(
            message_id="msg-order",
            content="异地就医政策依据是什么",
            active_stage=None,
        ),
        "light_semantic_frame": {
            "source": "intent_example_biencoder",
            "utterance_type": "question",
            "business_intent": "expert_task",
            "answer_shape": "analysis",
            "evidence_need": "policy_evidence",
            "focus_candidate": {"target_layer_hint": "L3"},
            "slots": {},
            "missing_slots": [],
            "confidence": 0.9,
        },
        "entity_frame": {"explicit_entities": {}, "slot_candidates": {}},
        "perception_started_epoch_ms": int(time.time() * 1000),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    state = build_perceptual_state_node(service, state)
    state = build_memory_retrieval_plan_node(service, state)
    start_request_memory_prefetch_node(service, state)

    assert repository.events.index("perceptual_state_ready") < repository.events.index(
        "memory_request_prefetch_started"
    )


def test_case_agent_perception_marks_example_followup_as_previous_answer_reuse() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    source = CaseAgentSource(
        source_ref="policy-evidence:remote-1",
        source_type="policy_rag",
        title="跨省异地就医直接结算通知",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(label="证据片段", value="就医地目录、参保地待遇。"),
            ]
        ),
    )
    previous_answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="跨省异地就医直接结算实行就医地目录、参保地待遇的分工原则。",
                source_refs=["policy-evidence:remote-1"],
            )
        ],
        sources=[source],
    )
    service = SimpleNamespace(_repository=PlannerRepository())
    state = {
        "run": SimpleNamespace(
            run_id="crun_reuse_perception",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "session": SimpleNamespace(
            session_id="session-001",
            title="测试会话",
            session_summary="",
            current_topic="跨省异地就医分工原则",
            referenced_source_refs=[],
        ),
        "user_message": SimpleNamespace(
            message_id="msg-follow",
            content="用个例子告诉我",
            active_stage=None,
        ),
        "recent_messages": [
            SimpleNamespace(
                message_id="msg-answer",
                role="assistant",
                content=previous_answer.plain_text,
                source_refs=["policy-evidence:remote-1"],
                answer_payload=previous_answer,
            ),
            SimpleNamespace(
                message_id="msg-follow",
                role="user",
                content="用个例子告诉我",
                source_refs=[],
                answer_payload=None,
            ),
        ],
        "actor_id": "actor-001",
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    for node in (
        input_ingestion_node,
        early_entity_extractor_node,
        semantic_intent_perception_node,
        rule_precheck_node,
        slot_merger_node,
        context_need_resolver_node,
        perception_gssc_node,
        perception_contract_builder_node,
        state_normalizer_node,
        confidence_calibrator_node,
        readiness_signal_builder_node,
    ):
        state = node(service, state)

    semantic = state["perceptual_state"]["semantic"]
    assert state["model_call_count"] == 0
    assert semantic["action"] == "explain_previous_answer"
    assert semantic["evidence_need"] == "previous_answer"
    assert semantic["requires_new_evidence"] is False
    assert semantic["reuse_answer_ref"] == "answer:msg-answer"
    assert semantic["reuse_source_refs"] == ["policy-evidence:remote-1"]


def test_case_agent_decision_reuses_previous_answer_without_l3_plan() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    service = SimpleNamespace(_repository=PlannerRepository())
    state = {
        "run": SimpleNamespace(run_id="crun_reuse_decision", case_id="CASE-001"),
        "user_message": SimpleNamespace(content="用个例子告诉我"),
        "perceptual_state": {
            "semantic": {
                "intent": "case_task",
                "action": "explain_previous_answer",
                "answer_shape": "analysis",
                "target_layer_hint": "none",
                "rewritten_query": "用一个简单直观的例子说明上一轮回答。",
                "evidence_need": "previous_answer",
                "requires_new_evidence": False,
                "reuse_answer_ref": "answer:msg-answer",
                "reuse_source_refs": ["policy-evidence:remote-1"],
                "rewrite_mode": "example",
            },
            "context_refs": {
                "answer_ref": "answer:msg-answer",
                "source_ref": ["policy-evidence:remote-1"],
            },
            "readiness_signal": {
                "ready_to_plan": True,
                "need_clarification": False,
                "fail_closed": False,
            },
        },
        "slots": {
            "answer_strategy": "reuse_previous_answer",
            "reuse_answer_ref": "answer:msg-answer",
            "reuse_source_refs": ["policy-evidence:remote-1"],
            "rewrite_mode": "example",
        },
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    state = decision_precheck_node(service, state)
    state = plan_normalizer_node(service, state)
    state = validate_execution_plan_node(service, state)

    assert state["answer_strategy"] == "reuse_previous_answer"
    assert state["execution_plan"] == []
    assert state["next_action"] == "build_answer_context"
    assert all(
        event["event_type"] != "llm_plan_created"
        for event in service._repository.events
    )
    assert any(
        event["event_type"] == "previous_answer_reuse_planned"
        for event in service._repository.events
    )


def test_case_agent_intent_example_matcher_accepts_policy_basis_query() -> None:
    matcher = IntentExampleMatcher(
        examples_path=Path("docs/02-planning/intent_biencoder_prep/intent_examples.seed.json"),
        taxonomy_path=Path("docs/02-planning/intent_biencoder_prep/intent_taxonomy.json"),
        backend="ngram",
    )

    result = matcher.match("异地手工报销政策怎么规定")

    assert result["decision"] == "accept"
    assert result["matched_intent_id"] == "policy_basis_query"
    assert result["ability_layer"] == "L3"
    assert result["target_layer_hint"] == "L3"
    assert result["capability_hint"] == "ask_policy_expert"
    assert result["evidence_need"] == "policy_evidence"
    json.dumps(result, ensure_ascii=False)


def test_case_agent_intent_example_matcher_loads_prebuilt_vector_index(tmp_path, monkeypatch) -> None:
    import numpy as np

    encode_calls: list[list[str]] = []

    index_dir = tmp_path / "intent_index"
    index_dir.mkdir()
    examples = [
        {
            "example_id": "policy_basis_query_0001",
            "intent_id": "policy_basis_query",
            "text": "政策依据是什么",
            "normalized_text": "政策依据是什么",
            "ability_layer": "L3",
            "capability_hint": "ask_policy_expert",
            "answer_shape": "policy_analysis",
            "evidence_need": "policy_evidence",
            "status": "available",
        },
        {
            "example_id": "case_basic_info_query_0001",
            "intent_id": "case_basic_info_query",
            "text": "案件类型是什么",
            "normalized_text": "案件类型是什么",
            "ability_layer": "L1",
            "capability_hint": "query_case_basic_info",
            "answer_shape": "single_field",
            "evidence_need": "case_fact",
            "status": "available",
        },
    ]
    (index_dir / "examples.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in examples) + "\n",
        encoding="utf-8",
    )
    np.save(index_dir / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    (index_dir / "manifest.json").write_text(
        json.dumps({"encoder_model": "fake-bge", "example_count": 2}, ensure_ascii=False),
        encoding="utf-8",
    )

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            encode_calls.append([str(text) for text in texts])
            vectors = []
            for text in texts:
                vectors.append([1.0, 0.0] if "政策" in str(text) else [0.0, 1.0])
            return np.asarray(vectors, dtype="float32")

    matcher = IntentExampleMatcher(
        examples_path=Path("docs/02-planning/intent_biencoder_prep/intent_examples.seed.json"),
        taxonomy_path=Path("docs/02-planning/intent_biencoder_prep/intent_taxonomy.json"),
        index_dir=index_dir,
        encoder_model="fake-bge",
        backend="bge",
    )
    monkeypatch.setattr(matcher, "_load_query_encoder", lambda _model_name: FakeEncoder())

    first_prewarm = matcher.prewarm()
    second_prewarm = matcher.prewarm()

    assert first_prewarm["prewarm"]["ready"] is True
    assert first_prewarm["prewarm"]["probe_score_count"] == 2
    assert second_prewarm["prewarm"] == first_prewarm["prewarm"]
    assert len(encode_calls) == 1

    result = matcher.match("政策依据是什么")

    assert len(encode_calls) == 2
    assert result["decision"] == "accept"
    assert result["backend"] == "bge_index"
    assert result["matched_intent_id"] == "policy_basis_query"
    assert result["ability_layer"] == "L3"
    assert result["capability_hint"] == "ask_policy_expert"
    assert result["evidence_need"] == "policy_evidence"


def test_case_agent_intent_example_biencoder_uses_injected_matcher_as_semantic_hint() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class Matcher:
        def match(self, _text: str) -> dict:
            return {
                "source": "intent_example_biencoder",
                "decision": "accept",
                "reason": "high_confidence",
                "business_intent": "expert_task",
                "matched_intent_id": "policy_basis_query",
                "ability_layer": "L3",
                "target_layer_hint": "L3",
                "capability_hint": "ask_policy_expert",
                "answer_shape": "analysis",
                "evidence_need": "policy_evidence",
                "confidence": 0.91,
                "margin": 0.2,
                "backend": "ngram",
                "matched_examples": [],
                "top_candidates": [],
            }

    repository = PlannerRepository()
    service = SimpleNamespace(_repository=repository, _intent_biencoder=Matcher())
    state = {
        "run": SimpleNamespace(run_id="crun_biencoder", case_id="CASE-001"),
        "input_envelope": {"normalized_text": "政策依据是什么"},
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = intent_example_biencoder_node(service, state)

    assert next_state["next_action"] == "build_perceptual_state"
    assert next_state["model_call_count"] == 0
    assert "execution_plan" not in next_state
    assert next_state["light_semantic_frame"]["business_intent"] == "expert_task"
    assert next_state["light_semantic_frame"]["evidence_need"] == "policy_evidence"
    assert next_state["light_semantic_frame"]["focus_candidate"]["target_layer_hint"] == "L3"
    assert next_state["light_semantic_frame"]["focus_candidate"]["capability_hint"] == "ask_policy_expert"
    assert [event["event_type"] for event in repository.events] == [
        "intent_example_biencoder_matching",
        "intent_example_biencoder_accepted",
    ]


def test_case_agent_llm_semantic_parser_emits_running_event_before_model_call() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class PlannerGateway:
        def complete(self, **_kwargs):
            return ModelResponse(
                content=json.dumps(
                    {
                        "intent": "case_task",
                        "action": "query_fact",
                        "answer_shape": "single_field",
                        "target_layer_hint": "L1",
                        "target_objects": ["risk_score"],
                        "rewritten_query": "当前案件风险是多少？",
                        "filled_slots": {"business_module": "risk_score"},
                        "missing_slots": [],
                        "confidence": 0.86,
                        "need_clarification": False,
                        "safety_flags": [],
                    },
                    ensure_ascii=False,
                )
            )

    repository = PlannerRepository()
    service = SimpleNamespace(
        _repository=repository,
        _gateway=PlannerGateway(),
        _classifier_model="test-classifier",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_llm_semantic", case_id="CASE-001"),
        "session": SimpleNamespace(current_topic=""),
        "user_message": SimpleNamespace(content="案件风险是多少"),
        "recent_messages": [],
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = llm_semantic_parser_node(service, state)

    assert next_state["next_action"] == "build_perceptual_state"
    assert next_state["model_call_count"] == 1
    assert "llm_semantic_result" in next_state
    assert "llm_semantic_plan" not in next_state
    assert "execution_plan" not in next_state
    assert next_state["deep_semantic_frame"]["business_intent"] == "case_task"
    assert next_state["deep_semantic_frame"]["focus_candidate"]["target_layer_hint"] == "L1"
    assert [event["event_type"] for event in repository.events] == [
        "llm_semantic_parsing",
        "llm_semantic_parsed",
    ]
    assert repository.events[0]["payload"] == {"model": "test-classifier"}


def test_case_agent_llm_planner_emits_running_event_before_model_call() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class PlannerGateway:
        def __init__(self, repository: PlannerRepository) -> None:
            self._repository = repository

        def complete(self, **_kwargs):
            assert [event["event_type"] for event in self._repository.events] == [
                "llm_planning",
            ]
            return ModelResponse(
                content=json.dumps(
                    {
                        "query_semantics": {
                            "intent": "case_task",
                            "user_goal": "案件风险是多少",
                            "granularity": "single_field",
                            "missing_slots": [],
                        },
                        "execution_plan": [
                            {
                                "step": 1,
                                "layer": "L1",
                                "capability": "query_risk_score",
                                "arguments": {"case_id": "CASE-001"},
                                "depends_on": [],
                                "reason": "查询当前案件风险评分。",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    repository = PlannerRepository()
    service = SimpleNamespace(
        _repository=repository,
        _gateway=PlannerGateway(repository),
        _classifier_model="test-classifier",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_llm_plan", case_id="CASE-001"),
        "session": SimpleNamespace(current_topic=""),
        "user_message": SimpleNamespace(content="案件风险是多少"),
        "recent_messages": [],
        "perceptual_state": {
            "semantic": {
                "rewritten_query": "",
            }
        },
        "planner_llm_context": {},
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = llm_planner_node(service, state)

    assert next_state["next_action"] == "plan_normalizer"
    assert next_state["model_call_count"] == 1
    assert [event["event_type"] for event in repository.events] == [
        "llm_planning",
        "llm_plan_created",
    ]
    assert repository.events[0]["payload"] == {"model": "test-classifier"}


def test_case_agent_heuristic_planner_routes_l3_semantic_to_policy_expert() -> None:
    class PlannerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    repository = PlannerRepository()
    service = SimpleNamespace(_repository=repository)
    question = "北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？备案成功后可在哪些机构就医？"
    state = {
        "run": SimpleNamespace(run_id="crun_heuristic_l3", case_id="CASE-001"),
        "user_message": SimpleNamespace(content=question),
        "model_call_count": 0,
        "tool_call_count": 0,
        "perceptual_state": {
            "semantic": {
                "intent": "general_help",
                "evidence_need": "policy_evidence",
                "rewritten_query": question,
                "focus_candidate": {
                    "target_layer_hint": "L3",
                    "capability_hint": "ask_policy_expert",
                    "action": "query_policy_basis",
                },
            },
            "readiness_signal": {"status": "ready"},
        },
    }

    next_state = heuristic_planner_node(service, state)

    plan = next_state["planning_output"]
    assert next_state["next_action"] == "plan_normalizer"
    assert plan["query_semantics"]["intent"] == "expert_task"
    assert plan["execution_plan"][0]["layer"] == "L3"
    assert plan["execution_plan"][0]["capability"] == "ask_policy_expert"
    assert plan["execution_plan"][0]["arguments"]["question"] == question
    assert repository.events[-1]["payload"]["step_count"] == 1


def test_case_agent_planner_routes_visit_info_question_to_l1_without_llm() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("visit info routing should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_plan_visit", case_id="CASE-001"),
        "user_message": SimpleNamespace(content="申报人都去哪里就诊过"),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "case_task"
    assert next_state["execution_plan"][0]["layer"] == "L1"
    assert next_state["execution_plan"][0]["capability"] == "query_medical_materials"
    assert next_state["next_action"] == "validate_execution_plan"


def test_case_agent_planner_routes_policy_principle_question_to_l3_without_llm() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("policy principle routing should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    question = "跨省异地就医中，就医地目录与参保地待遇如何分工？"
    state = {
        "run": SimpleNamespace(run_id="crun_plan_policy", case_id="CASE-001"),
        "user_message": SimpleNamespace(content=question),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "expert_task"
    assert next_state["query_semantics"]["granularity"] == "analysis"
    assert next_state["execution_plan"] == [
        {
            "step": 1,
            "layer": "L3",
            "capability": "ask_policy_expert",
            "arguments": {
                "case_id": "CASE-001",
                "question": question,
                "goal": question,
            },
            "depends_on": [],
            "reason": "该问题需要查询政策口径，由 Policy Expert 子 Agent 处理。",
        }
    ]
    assert next_state["next_action"] == "validate_execution_plan"


def test_case_agent_planner_resolves_rule_trigger_followup_from_recent_context() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("follow-up rule routing should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_plan_followup", case_id="CASE-001"),
        "session": SimpleNamespace(
            current_topic="OP-R009 规则核验",
            referenced_source_refs=["rule:OP-R009:manual_upload_review"],
        ),
        "recent_messages": [
            SimpleNamespace(
                role="assistant",
                content="规则 OP-R009 提示手工上传材料需要复核。",
                source_refs=["rule:OP-R009:manual_upload_review"],
                answer_payload=None,
            )
        ],
        "user_message": SimpleNamespace(content="为什么会触发呢"),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "case_task"
    assert next_state["query_semantics"]["granularity"] == "analysis"
    assert next_state["missing_slots"] == []
    assert next_state["execution_plan"] == [
        {
            "step": 1,
            "layer": "L1",
            "capability": "query_rule_verification",
            "arguments": {
                "case_id": "CASE-001",
                "filters": {"rule_id": "OP-R009"},
            },
            "depends_on": [],
            "reason": "短追问承接上一轮规则对象，读取当前案件规则核验清单解释触发原因。",
        }
    ]
    assert "OP-R009" in next_state["query_semantics"]["user_goal"]


def test_case_agent_planner_counts_recent_audit_suggestions_without_llm() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("recent count follow-up should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    recent_answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text=(
                    "证据包命中 3 条需关注的异常风险线索：异地手工报销备案/急诊例外核验、"
                    "检查治疗结构核验、材料补充核验。"
                ),
                source_refs=[],
            )
        ],
        sources=[],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_plan_count", case_id="CASE-001"),
        "recent_messages": [
            SimpleNamespace(
                role="assistant",
                content="证据包命中 3 条需关注的异常风险线索。",
                source_refs=[],
                answer_payload=recent_answer,
            )
        ],
        "user_message": SimpleNamespace(content="审核建议一共有几项"),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)
    answer = build_shortcut_answer(CaseAgentShortcut(**next_state["shortcut"]))

    assert next_state["intent"] == "general_help"
    assert next_state["execution_plan"] == []
    assert next_state["next_action"] == "validate_execution_plan"
    assert "3 项" in answer.plain_text
    assert "异地手工报销备案/急诊例外核验" in answer.plain_text


def test_case_agent_planner_routes_case_field_definition_to_case_context() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("case field definition routing should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_plan_field_definition", case_id="CASE-001"),
        "user_message": SimpleNamespace(content="什么是备案状态"),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "case_task"
    assert next_state["query_semantics"]["granularity"] == "analysis"
    assert next_state["execution_plan"][0]["capability"] == "query_case_basic_info"


def test_case_agent_planner_routes_shanghai_drug_price_reference_to_l3_without_llm() -> None:
    class PlannerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlannerGateway:
        def complete(self, **_kwargs):
            raise AssertionError("drug price reference routing should not call planner LLM")

    service = SimpleNamespace(
        _repository=PlannerRepository(),
        _gateway=PlannerGateway(),
        _classifier_model="test-model",
    )
    question = "上海的布洛芬均价是多少"
    state = {
        "run": SimpleNamespace(run_id="crun_plan_drug_price", case_id="CASE-001"),
        "user_message": SimpleNamespace(content=question),
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = business_semantic_planner_node(service, state)

    assert next_state["intent"] == "expert_task"
    assert next_state["execution_plan"][0]["capability"] == "ask_policy_expert"
    args = next_state["execution_plan"][0]["arguments"]
    assert args["question"] == question
    assert args["filters"] == {
        "jurisdiction": "shanghai",
        "policy_domain": "drug_product_price_reference",
        "content_type": "table_row",
        "can_cite_as_policy_basis": False,
    }
    assert next_state["next_action"] == "validate_execution_plan"


def test_case_agent_validate_execution_plan_sorts_by_step_without_answer_policy() -> None:
    class PlanRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class PlanTools:
        def section_status(self, *, case_id: str, capability_name: str) -> dict:
            return {
                "case_id": case_id,
                "capability": capability_name,
                "status": "ready",
            }

    service = SimpleNamespace(_repository=PlanRepository(), _tools=PlanTools())
    state = {
        "run": SimpleNamespace(run_id="crun_plan", case_id="CASE-001"),
        "model_call_count": 0,
        "tool_call_count": 0,
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "查询申报人信息和风险评分",
            "missing_slots": [],
        },
        "execution_plan": [
            {
                "step": 2,
                "layer": "L1",
                "capability": "query_risk_score",
                "arguments": {"case_id": "CASE-001"},
                "reason": "查看风险评分",
            },
            {
                "step": 1,
                "layer": "L1",
                "capability": "claimant_profile",
                "arguments": {"case_id": "CASE-001"},
                "reason": "查看申报人基础信息",
            },
        ],
        "missing_slots": [],
    }

    next_state = validate_execution_plan_node(service, state)

    assert [item["capability"] for item in next_state["execution_plan"]] == [
        "query_claimant_profile",
        "query_risk_score",
    ]
    assert next_state["execution_plan"][0]["step"] == 1
    assert next_state["execution_plan"][0]["depends_on"] == []
    assert next_state["next_action"] == "execution_dag_builder"
    assert "display_mode" not in next_state["context_plan"]


def test_case_agent_validate_execution_plan_drops_nested_fields_instead_of_failing() -> None:
    class PlanRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class PlanTools:
        def section_status(self, *, case_id: str, capability_name: str) -> dict:
            return {
                "case_id": case_id,
                "capability": capability_name,
                "status": "ready",
            }

    service = SimpleNamespace(_repository=PlanRepository(), _tools=PlanTools())
    state = {
        "run": SimpleNamespace(run_id="crun_plan_nested_fields", case_id="CASE-001"),
        "model_call_count": 0,
        "tool_call_count": 0,
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "查询就诊信息",
            "missing_slots": [],
        },
        "execution_plan": [
            {
                "step": 1,
                "layer": "L1",
                "capability": "query_case_basic_info",
                "arguments": {
                    "case_id": "CASE-001",
                    "fields": ["case_context.treatment_region", "case_number"],
                },
                "reason": "查看就诊地",
            }
        ],
        "missing_slots": [],
    }

    next_state = validate_execution_plan_node(service, state)

    assert next_state["next_action"] == "execution_dag_builder"
    assert next_state["execution_plan"][0]["arguments"]["fields"] == ["case_number"]


def test_case_agent_execution_dag_dispatches_l1_and_writes_artifact() -> None:
    class DagRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class DagService:
        def __init__(self) -> None:
            self._repository = DagRepository()

        def _run_tool(self, run_id, case_id, tool_call_id, tool_name, arguments):
            assert run_id == "crun_dag"
            assert case_id == "CASE-001"
            assert tool_name == "query_claimant_profile"
            assert arguments["case_id"] == "CASE-001"
            return CaseAgentToolResult(
                payload={
                    "status": "ok",
                    "section_key": "claimant_profile",
                    "payload": {"gender": "女"},
                },
                source_refs=["case:CASE-001:claimant_profile"],
                sources=[
                    CaseAgentSource(
                        source_ref="case:CASE-001:claimant_profile",
                        source_type="case",
                        title="申报人基础信息",
                    )
                ],
            )

    service = DagService()
    state = {
        "run": SimpleNamespace(run_id="crun_dag", case_id="CASE-001"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L1",
                "capability": "query_claimant_profile",
                "arguments": {"case_id": "CASE-001"},
                "depends_on": [],
            }
        ],
        "model_call_count": 0,
        "tool_call_count": 0,
        "capability_results": [],
        "available_sources": [],
    }

    state = execution_dag_builder_node(service, state)
    assert state["next_action"] == "dag_executor"
    assert state["execution_dag"]["steps"][0]["step_id"] == "step_1"

    state = dag_executor_node(service, state)
    assert state["next_action"] == "dispatch_step"
    assert state["current_dag_step"]["capability"] == "query_claimant_profile"

    state = dispatch_step_node(service, state)
    assert state["next_action"] == "execute_l1_step"

    state = execute_l1_step_node(service, state)
    assert state["next_action"] == "dag_executor"
    assert state["execution_dag"]["steps"][0]["status"] == "completed"
    assert state["artifact_refs"][0].startswith("artifact:l1_result_ref:step_1")
    assert state["capability_results"][0][0] == "query_claimant_profile"
    assert state["available_sources"][0].source_ref == "case:CASE-001:claimant_profile"


def test_case_agent_artifact_state_keeps_only_lightweight_index() -> None:
    large_rows = [{"序号": index, "金额": index * 10} for index in range(160)]
    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "settlement_materials",
            "payload": {
                "settlement_summary": {"申报总费用": 8000},
                "non_drug_fee_items": large_rows,
            },
        },
        source_refs=["material:CASE-001-BM-05"],
        sources=[
            CaseAgentSource(
                source_ref="material:CASE-001-BM-05",
                source_type="material",
                title="费用结算材料",
            )
        ],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_artifact_thin", case_id="CASE-001"),
        "execution_dag": {
            "steps": [
                {
                    "step_id": "step_1",
                    "step": 1,
                    "layer": "L1",
                    "capability": "query_settlement_materials",
                    "status": "completed",
                }
            ]
        },
        "artifact_store": {"artifacts": {}, "by_layer": {"L1": [], "L2": [], "L3": []}},
        "artifact_refs": [],
        "capability_results": [],
        "available_sources": [],
    }

    try:
        artifact_ref = write_step_artifact(
            state,
            step={
                "step_id": "step_1",
                "step": 1,
                "layer": "L1",
                "capability": "query_settlement_materials",
            },
            result=result,
        )

        stored_index = state["artifact_store"]["artifacts"][artifact_ref]
        assert "payload" not in stored_index
        assert "sources" not in stored_index
        assert stored_index["payload_summary"]["payload_omitted"] is True
        assert stored_index["payload_summary"]["payload_bytes"] > 2400

        deps = dependency_artifacts(
            state,
            {"step_id": "step_2", "step": 2, "depends_on": ["step_1"]},
        )
        assert deps[0]["payload"]["payload"]["non_drug_fee_items"][79]["金额"] == 790

        rehydrated = artifact_results(state)
        assert rehydrated[0][1].payload["payload"]["non_drug_fee_items"][0]["序号"] == 0
        assert rehydrated[0][1].sources[0].source_ref == "material:CASE-001-BM-05"
    finally:
        clear_runtime_artifacts(state)


def test_case_agent_materializes_l3_task_from_dependency_artifacts() -> None:
    class L3Repository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append({"event_type": event_type, "payload": payload or {}})

    class L3Service:
        def __init__(self) -> None:
            self._repository = L3Repository()
            self.captured_arguments: dict[str, object] = {}

        def _materialize_expert_task(self, **kwargs):
            self.captured_arguments = kwargs["arguments"]
            return SimpleNamespace(
                task_id="l3_policy_task",
                context_refs=kwargs["arguments"]["source_refs"],
                fact_bundle=kwargs["arguments"]["fact_bundle"],
                allowed_tools=["policy.search_text", "policy.search_version"],
            )

    l1_result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "case_basic_info",
            "payload": {"insured_region": "北京"},
        },
        source_refs=["case:CASE-001:basic"],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_l3_task", case_id="CASE-001"),
        "execution_dag": {
            "steps": [
                {
                    "step_id": "step_1",
                    "step": 1,
                    "layer": "L1",
                    "capability": "query_case_basic_info",
                    "status": "completed",
                },
                {
                    "step_id": "step_2",
                    "step": 2,
                    "layer": "L3",
                    "capability": "ask_policy_expert",
                    "arguments": {"case_id": "CASE-001", "question": "政策怎么规定？"},
                    "depends_on": ["step_1"],
                    "status": "running",
                },
            ]
        },
        "current_dag_step": {
            "step_id": "step_2",
            "step": 2,
            "layer": "L3",
            "capability": "ask_policy_expert",
            "arguments": {"case_id": "CASE-001", "question": "政策怎么规定？"},
            "depends_on": ["step_1"],
        },
        "artifact_store": {"artifacts": {}, "by_layer": {"L1": [], "L2": [], "L3": []}},
        "artifact_refs": [],
        "query_semantics": {"user_goal": "政策怎么规定？"},
        "user_message": SimpleNamespace(content="政策怎么规定？"),
    }
    write_step_artifact(
        state,
        step={
            "step_id": "step_1",
            "step": 1,
            "layer": "L1",
            "capability": "query_case_basic_info",
        },
        result=l1_result,
    )

    service = L3Service()
    next_state = materialize_expert_task_node(service, state)

    assert next_state["next_action"] == "call_expert_analysis"
    assert service.captured_arguments["source_refs"] == ["case:CASE-001:basic"]
    assert service.captured_arguments["fact_bundle"][0]["artifact_ref"].startswith(
        "artifact:l1_result_ref:step_1"
    )


def test_case_agent_materialize_expert_task_uses_configured_policy_rag_timeout() -> None:
    service = CaseAgentService(
        repository=SimpleNamespace(),
        gateway=SimpleNamespace(),
        tools=SimpleNamespace(),
        expert_agent=SimpleNamespace(),
        manage_notes=SimpleNamespace(),
        model_name="case-agent-test",
        classifier_model="classifier-test",
        generator_model="generator-test",
        max_concurrency=1,
        max_tool_calls=9,
        policy_rag_timeout_ms=45000,
    )

    task = service._materialize_expert_task(
        run_id="crun_timeout_budget",
        case_id="CASE-001",
        tool_name="ask_policy_expert",
        expert_task_type="policy_analysis",
        state={
            "run": SimpleNamespace(run_id="crun_timeout_budget", case_id="CASE-001"),
            "user_message": SimpleNamespace(content="上海异地报销规则是什么"),
            "query_semantics": {"user_goal": "上海异地报销规则是什么"},
        },
        arguments={
            "question": "上海异地报销规则是什么",
            "goal": "上海异地报销规则是什么",
            "filters": {"jurisdiction": ["shanghai"]},
            "fact_bundle": [
                {
                    "label": "参保地",
                    "value": "上海",
                    "source_refs": ["case:CASE-001:basic"],
                }
            ],
            "source_refs": ["case:CASE-001:basic"],
        },
    )

    assert task.budget.timeout_ms == 45000
    assert task.budget.max_tool_calls == 6
    assert task.user_question == "上海异地报销规则是什么"
    assert task.filters == {"jurisdiction": ["shanghai"]}


def test_case_agent_planned_memory_prefetch_is_non_blocking() -> None:
    class Cache:
        def __init__(self) -> None:
            self.statuses: dict[tuple[str, MemoryType], str] = {}

        def begin(self, request_id: str, memory_type: MemoryType) -> str:
            key = (request_id, memory_type)
            self.statuses.setdefault(key, "in_flight")
            return self.statuses[key]

        def complete(self, request_id: str, memory_type: MemoryType, _rows) -> str:
            self.statuses[(request_id, memory_type)] = "empty"
            return "empty"

        def status(self, request_id: str, memory_type: MemoryType) -> str:
            return self.statuses.get((request_id, memory_type), "not_requested")

        def mark_terminal(self, request_id: str, memory_type: MemoryType, status: str) -> str:
            self.statuses[(request_id, memory_type)] = status
            return status

        def clear_all(self) -> None:
            self.statuses.clear()

    class SlowCaseMemory:
        def __init__(self) -> None:
            self.cache = Cache()

        def prefetch_type(self, request, *, allowed_consumers=None) -> int:
            time.sleep(0.25)
            self.cache.complete(request.request_id, request.memory_type, [])
            return 0

    memory = SlowCaseMemory()
    service = CaseAgentService(
        repository=SimpleNamespace(),
        gateway=SimpleNamespace(),
        tools=SimpleNamespace(),
        expert_agent=SimpleNamespace(),
        manage_notes=SimpleNamespace(),
        model_name="case-agent-test",
        classifier_model="classifier-test",
        generator_model="generator-test",
        max_concurrency=1,
        max_tool_calls=9,
        case_memory=memory,
    )
    state = {
        "run": SimpleNamespace(
            run_id="crun-async-memory",
            case_id="CASE-001",
            session_id="session-001",
        ),
        "actor_id": "actor-001",
        "user_message": SimpleNamespace(content="分析案件风险"),
        "intent": "case_analysis",
        "entity_frame": {},
        "memory_retrieval_plan": {
            "items": [
                {
                    "memory_type": MemoryType.DECISION_PLAN.value,
                    "consumers": ["decision_planner"],
                    "max_items": 3,
                    "wait_timeout_ms": 300,
                }
            ]
        },
    }

    try:
        started = time.perf_counter()
        result = service.start_request_memory_prefetch(state)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.12
        assert result["types"][MemoryType.DECISION_PLAN.value] == "in_flight"
        future = service._request_memory_futures[
            (state["run"].run_id, MemoryType.DECISION_PLAN.value)
        ]
        future.result(timeout=1)
        assert memory.cache.status(
            state["run"].run_id,
            MemoryType.DECISION_PLAN,
        ) == "empty"
    finally:
        service.shutdown()


def test_case_agent_session_memory_bootstrap_returns_before_recall_finishes() -> None:
    class Cache:
        def clear_all(self) -> None:
            pass

    class SlowCaseMemory:
        def __init__(self) -> None:
            self.cache = Cache()
            self.recall_count = 0

        def recall(self, request):
            self.recall_count += 1
            time.sleep(0.25)
            return MemoryHintPack(
                request_id=request.request_id,
                consumer=request.consumer,
                memory_type=request.memory_type,
            )

    memory = SlowCaseMemory()
    service = CaseAgentService(
        repository=SimpleNamespace(),
        gateway=SimpleNamespace(),
        tools=SimpleNamespace(),
        expert_agent=SimpleNamespace(),
        manage_notes=SimpleNamespace(),
        model_name="case-agent-test",
        classifier_model="classifier-test",
        generator_model="generator-test",
        max_concurrency=1,
        max_tool_calls=9,
        case_memory=memory,
    )
    state = {
        "run": SimpleNamespace(run_id="crun-session-memory"),
        "session": SimpleNamespace(session_id="session-001", current_topic=""),
        "actor_id": "actor-001",
    }

    try:
        started = time.perf_counter()
        result = service.bootstrap_session_memory(state)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.12
        assert result["status"] == "in_flight"
        service._session_memory_futures["session-001"].result(timeout=1)
        assert service.session_memory_context(state)["status"] == "empty"
        assert memory.recall_count == 1
    finally:
        service.shutdown()


def test_validate_answer_loads_failure_memory_for_one_controlled_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []

    class Repository:
        def update_run(self, _run_id: str, **_kwargs) -> None:
            return None

        def append_event(self, _run_id: str, event_type: str, _message: str, payload=None):
            events.append((event_type, payload or {}))

    def parse_answer(_response, _state):
        raise CaseAgentSafetyError("citation_missing_for_grounded", "missing source")

    def memory_hint_for_node(_state, **kwargs):
        calls.append(kwargs)
        return {
            "tool_param_hints": [
                {
                    "structured_content": {
                        "repair_strategy": "second_model_call",
                        "avoid": "不得保留无引用结论",
                        "recover": "仅使用受控证据重写",
                    }
                }
            ],
            "memory_ids": ["mem-failure-1"],
        }

    service = SimpleNamespace(
        _repository=Repository(),
        _parse_answer=parse_answer,
        memory_hint_for_node=memory_hint_for_node,
    )
    monkeypatch.setattr(
        "src.backend.application.agent.case_agent.nodes.validate_answer.build_deterministic_answer",
        lambda _state, recovery=False: None,
    )
    state = {
        "run": SimpleNamespace(run_id="run-validation-memory"),
        "final_response": SimpleNamespace(content="{}"),
        "model_call_count": 1,
        "tool_call_count": 0,
        "validation_retry_count": 0,
        "answer_strategy": "",
    }

    result = validate_answer_node(service, state)

    assert result["next_action"] == "generate_answer"
    assert result["validation_retry_count"] == 1
    assert result["validation_recovery_mode"] == "second_model_call"
    assert result["failure_recovery_hint"]["memory_ids"] == ["mem-failure-1"]
    assert calls[0]["memory_type"] == MemoryType.FAILURE
    assert calls[0]["task_context"]["node"] == "validate_answer"
    assert calls[0]["task_context"]["validation_codes"] == [
        "citation_missing_for_grounded"
    ]
    assert events[-1][0] == "repairing_answer"


def test_case_agent_call_expert_analysis_writes_l3_and_case_context_artifacts() -> None:
    class ExpertCallRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {"event_type": event_type, "payload": payload or {}}

    class ExpertCallService:
        def __init__(self) -> None:
            self._repository = ExpertCallRepository()

        def _call_expert_analysis_task(self, **kwargs):
            return ExpertAgentResult(
                task_id="l3_policy_artifacts",
                expert_task_type="policy_analysis",
                status="ok",
                message="政策专家片段",
                payload={
                    "status": "ok",
                    "expert_answer": "政策专家片段",
                    "case_facts_used": [
                        {
                            "label": "参保地",
                            "value": "北京",
                            "source_refs": ["case:CASE-001:basic"],
                        }
                    ],
                    "policy_evidence": [
                        {
                            "evidence_ref": "policy-evidence:artifacts",
                            "source_ref": "policy-evidence:artifacts",
                            "title": "政策证据",
                            "excerpt": "政策依据片段",
                        }
                    ],
                    "material_gaps": [],
                    "audit_suggestions": [],
                    "limits": "不形成最终审核结论",
                },
                source_refs=["policy-evidence:artifacts"],
                safe_summary={
                    "case_context_observations": [
                        {
                            "capability": "query_case_basic_info",
                            "status": "success",
                            "section_key": "case_basic_info",
                            "payload_keys": ["insured_region"],
                            "source_refs": ["case:CASE-001:basic"],
                        }
                    ]
                },
            )

        def _expert_result_to_tool_result(self, *, case_id, capability, result):
            return CaseAgentToolResult(
                payload={
                    "status": "ok",
                    "section_key": "policy_expert",
                    "payload": result.payload,
                    "expert_task_id": result.task_id,
                },
                source_refs=result.source_refs,
                sources=[
                    CaseAgentSource(
                        source_ref="policy-evidence:artifacts",
                        source_type="policy_rag",
                        title="政策证据",
                    )
                ],
            )

    state = {
        "run": SimpleNamespace(run_id="crun_l3_call", case_id="CASE-001"),
        "current_dag_step": {
            "step_id": "step_2",
            "step": 2,
            "layer": "L3",
            "capability": "ask_policy_expert",
        },
        "current_expert_task": SimpleNamespace(task_id="l3_policy_artifacts"),
        "execution_dag": {
            "steps": [
                {
                    "step_id": "step_2",
                    "step": 2,
                    "layer": "L3",
                    "capability": "ask_policy_expert",
                    "status": "running",
                }
            ],
            "completed_step_ids": [],
        },
        "artifact_store": {"artifacts": {}, "by_layer": {"L1": [], "L2": [], "L3": []}},
        "artifact_refs": [],
        "available_sources": [
            CaseAgentSource(
                source_ref="case:CASE-001:basic",
                source_type="case",
                title="案件基础信息",
            )
        ],
        "capability_results": [],
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = call_expert_analysis_node(ExpertCallService(), state)

    assert next_state["next_action"] == "dag_executor"
    assert any(":l3_result_ref:" in ref for ref in next_state["artifact_refs"])
    assert any(":case_context_observation_ref:" in ref for ref in next_state["artifact_refs"])
    assert next_state["capability_results"][0][0] == "ask_policy_expert"


def test_case_agent_resolve_answer_policy_sets_grounded_for_successful_l1() -> None:
    class PolicyRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    service = SimpleNamespace(
        _repository=PolicyRepository(),
        _fast_mode_enabled=False,
        _apply_fast_context_digest=lambda state: None,
    )
    result = CaseAgentToolResult(
        payload={"status": "ok", "section_key": "claimant_profile"},
        source_refs=["case:CASE-001:claimant_profile"],
        sources=[
            CaseAgentSource(
                source_ref="case:CASE-001:claimant_profile",
                source_type="case",
                title="Claimant profile",
            )
        ],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_policy"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L1",
                "capability": "query_claimant_profile",
                "arguments": {"case_id": "CASE-001"},
            }
        ],
        "capability_results": [("query_claimant_profile", result)],
        "context_plan": {"capabilities": []},
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = resolve_answer_policy_node(service, state)

    assert next_state["answer_policy"]["display_mode"] == "grounded"
    assert next_state["answer_policy"]["requires_citation"] is True
    assert next_state["answer_policy"]["capabilities_used"] == ["query_claimant_profile"]
    assert next_state["context_plan"]["display_mode"] == "grounded"
    assert next_state["next_action"] == "resolve_answer_style"


def test_case_agent_resolve_answer_policy_sets_unavailable_for_l3_only() -> None:
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        ),
        _fast_mode_enabled=False,
        _apply_fast_context_digest=lambda state: None,
    )
    result = CaseAgentToolResult(
        payload={"status": "unavailable", "capability": "ask_policy_expert"},
        status="unavailable",
        error_code="capability_not_connected",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_policy"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L3",
                "capability": "ask_policy_expert",
                "arguments": {"case_id": "CASE-001"},
            }
        ],
        "capability_results": [("ask_policy_expert", result)],
        "context_plan": {"capabilities": []},
    }

    next_state = resolve_answer_policy_node(service, state)

    assert next_state["answer_policy"]["display_mode"] == "unavailable"
    assert next_state["answer_policy"]["source_policy"] == "none"
    assert next_state["answer_policy"]["no_general_knowledge_fallback"] is True
    assert next_state["answer_policy"]["unavailable_capabilities"] == ["ask_policy_expert"]
    assert next_state["next_action"] == "resolve_answer_style"


def test_case_agent_resolve_answer_policy_distinguishes_l3_evidence_insufficient() -> None:
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        ),
        _fast_mode_enabled=False,
        _apply_fast_context_digest=lambda state: None,
    )
    result = CaseAgentToolResult(
        payload={
            "status": "insufficient",
            "capability": "ask_policy_expert",
            "section_key": "policy_expert",
            "payload": {
                "status": "insufficient",
                "expert_answer": "未检索到足够、可引用的政策证据回答该问题。",
                "answerability_summary": {
                    "answerability": "insufficient",
                    "covered_slots": [],
                    "missing_slots": ["参保地待遇分工规则"],
                    "usable_evidence_refs": [],
                    "next_action": "insufficient",
                    "reason_code": "missing_key_rule",
                    "reason": "缺少关键规则。",
                },
            },
        },
        status="unavailable",
        error_code="expert_evidence_insufficient",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_policy_insufficient"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L3",
                "capability": "ask_policy_expert",
                "arguments": {"case_id": "CASE-001"},
            }
        ],
        "capability_results": [("ask_policy_expert", result)],
        "context_plan": {"capabilities": []},
    }

    next_state = resolve_answer_policy_node(service, state)

    assert next_state["answer_policy"]["display_mode"] == "unavailable"
    assert (
        next_state["answer_policy"]["capability_policy"]
        == "expert_evidence_insufficient"
    )
    assert next_state["answer_policy"]["unavailable_capabilities"] == []
    assert "evidence" in next_state["answer_policy"]["reason"].lower()


def test_case_agent_resolve_answer_policy_respects_shortcut_display_mode() -> None:
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        ),
        _fast_mode_enabled=False,
        _apply_fast_context_digest=lambda state: None,
    )
    state = {
        "run": SimpleNamespace(run_id="crun_policy"),
        "shortcut": {"display_mode": "unavailable", "kind": "capability_status"},
        "execution_plan": [],
        "capability_results": [],
        "context_plan": {},
    }

    next_state = resolve_answer_policy_node(service, state)

    assert next_state["answer_policy"]["display_mode"] == "unavailable"
    assert next_state["answer_policy"]["capability_policy"] == "shortcut"
    assert next_state["next_action"] == "resolve_answer_style"


def test_case_agent_resolve_answer_policy_reuses_previous_grounded_answer() -> None:
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        ),
        _fast_mode_enabled=False,
        _apply_fast_context_digest=lambda state: None,
    )
    state = {
        "run": SimpleNamespace(run_id="crun_reuse_policy"),
        "answer_strategy": "reuse_previous_answer",
        "execution_plan": [],
        "capability_results": [],
        "answer_context": {
            "source_refs": ["policy-evidence:remote-1"],
            "analysis_inputs": [
                {
                    "section": "上一轮回答",
                    "mode": "reuse_previous_answer",
                    "previous_source_refs": ["policy-evidence:remote-1"],
                }
            ],
        },
        "context_plan": {"capabilities": []},
    }

    next_state = resolve_answer_policy_node(service, state)

    assert next_state["answer_policy"]["display_mode"] == "grounded"
    assert next_state["answer_policy"]["capability_policy"] == "reuse_previous_answer"
    assert next_state["answer_policy"]["requires_citation"] is True
    assert next_state["context_plan"]["source_policy"] == "required"


def test_case_agent_dispatches_policy_expert_through_l3_subagent() -> None:
    class DispatchRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class DispatchService:
        def __init__(self) -> None:
            self._repository = DispatchRepository()
            self.called_l3 = False
            self.called_tool = False

        def _run_expert_capability(self, run_id, expert_task_type, state, **kwargs):
            self.called_l3 = True
            return ExpertAgentResult(
                task_id="l3_policy_dispatch",
                expert_task_type=expert_task_type,
                status="ok",
                message="ok",
                payload={
                    "status": "ok",
                    "expert_answer": "政策专家片段",
                    "policy_evidence": [
                        {
                            "evidence_ref": "policy-evidence:dispatch",
                            "source_ref": "policy-evidence:dispatch",
                            "title": "政策证据",
                            "excerpt": "政策依据片段",
                        }
                    ],
                    "material_gaps": [],
                    "audit_suggestions": [],
                    "limits": "不形成最终审核结论",
                },
                source_refs=["policy-evidence:dispatch"],
            )

        def _expert_result_to_tool_result(self, *, case_id, capability, result):
            assert case_id == "CASE-001"
            assert capability == "ask_policy_expert"
            return CaseAgentToolResult(
                payload={
                    "status": "ok",
                    "section_key": "policy_expert",
                    "payload": result.payload,
                },
                source_refs=result.source_refs,
                sources=[
                    CaseAgentSource(
                        source_ref="policy-evidence:dispatch",
                        source_type="policy_rag",
                        title="政策证据",
                    )
                ],
            )

        def _run_tool(self, *args, **kwargs):
            self.called_tool = True
            raise AssertionError("ask_policy_expert must not use registry tool path")

        def _collect_available_sources(self, results):
            return CaseAgentService._collect_available_sources(results)

    service = DispatchService()
    state = {
        "run": SimpleNamespace(run_id="crun_dispatch", case_id="CASE-001"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L3",
                "capability": "ask_policy_expert",
                "arguments": {"case_id": "CASE-001", "question": "政策怎么规定？"},
            }
        ],
        "context_plan": {},
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = call_capabilities_node(service, state)

    assert service.called_l3 is True
    assert service.called_tool is False
    assert next_state["capability_results"][0][0] == "ask_policy_expert"
    assert next_state["available_sources"][0].source_type == "policy_rag"
    assert next_state["next_action"] == "build_answer_context"


def test_case_agent_generate_answer_reuses_policy_expert_without_main_model() -> None:
    class AnswerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class ExplodingGateway:
        def complete(self, **_kwargs):
            raise AssertionError("main Case Agent model must not be called")

    source = CaseAgentSource(
        source_ref="policy-evidence:reuse",
        source_type="policy_rag",
        title="跨省异地就医政策证据",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(
                    label="证据片段",
                    value="就医地目录、参保地待遇。",
                )
            ]
        ),
    )
    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "policy_expert",
            "payload": {
                "status": "ok",
                "expert_answer": "跨省异地就医一般按就医地目录、参保地待遇执行。",
                "policy_evidence": [
                    {
                        "source_ref": "policy-evidence:reuse",
                        "title": "跨省异地就医政策证据",
                        "excerpt": "就医地目录、参保地待遇。",
                    }
                ],
                "audit_suggestions": [],
                "limits": "仅提供政策口径，不形成最终审核结论。",
            },
            "expert_task_id": "l3_policy_reuse",
        },
        source_refs=["policy-evidence:reuse"],
        sources=[source],
    )
    service = SimpleNamespace(
        _repository=AnswerRepository(),
        _fast_mode_enabled=True,
        _gateway=ExplodingGateway(),
        _generator_model="deepseek-v4-flash",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_answer"),
        "intent": "policy_expert_query",
        "answer_policy": {
            "display_mode": "grounded",
            "requires_citation": True,
        },
        "context_plan": {"display_mode": "grounded"},
        "capability_results": [("ask_policy_expert", result)],
        "available_sources": [source],
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = generate_answer_node(service, state)

    assert next_state["next_action"] == "validate_answer"
    assert next_state["model_call_count"] == 0
    assert next_state["final_response"].metrics["model_call_skipped"] is True
    answer = CaseAgentAnswer.model_validate_json(next_state["final_response"].content)
    assert answer.metadata["deterministic_handoff"] is True
    assert answer.metadata["capabilities_used"] == ["ask_policy_expert"]
    assert answer.content_blocks[0].source_refs == ["policy-evidence:reuse"]
    validate_answer(answer, [source])


def test_case_agent_policy_expert_handoff_preserves_markdown_citations() -> None:
    class AnswerRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    class ExplodingGateway:
        def complete(self, **_kwargs):
            raise AssertionError("main Case Agent model must not be called")

    source = CaseAgentSource(
        source_ref="policy-evidence:remote",
        source_type="policy_rag",
        title="跨省异地就医政策证据",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(
                    label="证据片段",
                    value="就医地支付范围，参保地待遇参数。",
                )
            ]
        ),
    )
    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "policy_expert",
            "payload": {
                "status": "ok",
                "expert_answer": "支付范围按就医地规定执行。[1] 待遇参数按参保地政策执行。[1]",
                "answer_markdown": "支付范围按就医地规定执行。[1] 待遇参数按参保地政策执行。[1]",
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "requirement_id": "req_1",
                        "slot_id": "remote_benefit_split",
                        "text": "支付范围按就医地规定执行。",
                        "fact_refs": ["fact_1"],
                        "source_refs": ["policy-evidence:remote"],
                        "citation_ids": ["cit_1"],
                        "support_status": "supported",
                    }
                ],
                "citations": [
                    {
                        "citation_id": "cit_1",
                        "label": 1,
                        "claim_id": "claim_1",
                        "fact_refs": ["fact_1"],
                        "evidence_refs": ["policy-node:remote"],
                        "source_refs": ["policy-evidence:remote"],
                    }
                ],
                "policy_evidence": [
                    {
                        "source_ref": "policy-evidence:remote",
                        "title": "跨省异地就医政策证据",
                        "excerpt": "就医地支付范围，参保地待遇参数。",
                    }
                ],
                "audit_suggestions": [],
            },
            "expert_task_id": "l3_policy_markdown",
        },
        source_refs=["policy-evidence:remote"],
        sources=[source],
    )
    service = SimpleNamespace(
        _repository=AnswerRepository(),
        _fast_mode_enabled=True,
        _gateway=ExplodingGateway(),
        _generator_model="deepseek-v4-flash",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_markdown"),
        "intent": "policy_expert_query",
        "answer_policy": {
            "display_mode": "grounded",
            "requires_citation": True,
        },
        "context_plan": {"display_mode": "grounded"},
        "capability_results": [("ask_policy_expert", result)],
        "available_sources": [source],
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = generate_answer_node(service, state)
    answer = CaseAgentAnswer.model_validate_json(next_state["final_response"].content)

    assert answer.answer_markdown == "支付范围按就医地规定执行。[1] 待遇参数按参保地政策执行。[1]"
    assert answer.claims[0].source_refs == ["policy-evidence:remote"]
    assert answer.citations[0].label == 1
    assert answer.metadata["claim_count"] == 1
    validate_answer(answer, [source])


def test_case_agent_preserves_partial_policy_evidence_when_insufficient() -> None:
    class AnswerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    class ExplodingGateway:
        def complete(self, **_kwargs):
            raise AssertionError("main Case Agent model must not be called")

    evidence = {
        "evidence_ref": "policy-node:partial",
        "source_ref": "policy-evidence:partial",
        "title": "跨省异地就医直接结算政策",
        "excerpt": "异地就医备案和直接结算流程相关规定。",
        "jurisdiction": "国家",
        "policy_domain": "异地就医",
        "content_type": "policy",
    }
    expert_result = ExpertAgentResult(
        task_id="l3_policy_partial",
        expert_task_type="policy_analysis",
        status="insufficient",
        message="已有证据涉及备案和直接结算流程，但缺少手工报销材料明细。",
        payload={
            "status": "insufficient",
            "expert_answer": "已有证据涉及备案和直接结算流程，但缺少手工报销材料明细。",
            "policy_evidence": [evidence],
            "answerability_summary": {
                "answerability": "insufficient",
                "covered_slots": ["异地就医备案和直接结算流程"],
                "missing_slots": ["手工报销材料明细"],
                "usable_evidence_refs": ["policy-node:partial"],
                "next_action": "insufficient",
                "reason_code": "missing_material_list",
                "reason": "缺少手工报销材料清单依据。",
            },
            "limits": "仅提供政策口径，不形成最终审核结论。",
        },
        source_refs=["policy-evidence:partial"],
    )
    service = CaseAgentService.__new__(CaseAgentService)
    tool_result = service._expert_result_to_tool_result(
        case_id="CASE-001",
        capability="ask_policy_expert",
        result=expert_result,
    )

    assert tool_result.status == "success"
    assert tool_result.error_code == "expert_evidence_insufficient"
    assert tool_result.source_refs == ["policy-evidence:partial"]

    policy_state = {
        "run": SimpleNamespace(run_id="crun_partial_policy"),
        "execution_plan": [
            {
                "step": 1,
                "layer": "L3",
                "capability": "ask_policy_expert",
                "arguments": {"case_id": "CASE-001"},
            }
        ],
        "capability_results": [("ask_policy_expert", tool_result)],
        "context_plan": {"capabilities": []},
    }
    policy_state = resolve_answer_policy_node(
        SimpleNamespace(
            _repository=AnswerRepository(),
            _fast_mode_enabled=False,
            _apply_fast_context_digest=lambda state: None,
        ),
        policy_state,
    )

    assert policy_state["answer_policy"]["display_mode"] == "grounded"
    assert (
        policy_state["answer_policy"]["capability_policy"]
        == "expert_evidence_insufficient"
    )

    answer_state = {
        **policy_state,
        "intent": "policy_expert_query",
        "available_sources": tool_result.sources,
        "model_call_count": 0,
        "tool_call_count": 1,
    }
    answer_state = generate_answer_node(
        SimpleNamespace(
            _repository=AnswerRepository(),
            _fast_mode_enabled=False,
            _gateway=ExplodingGateway(),
            _generator_model="deepseek-v4-flash",
        ),
        answer_state,
    )

    answer = CaseAgentAnswer.model_validate_json(
        answer_state["final_response"].content
    )
    assert "缺少手工报销材料明细" in answer.content_blocks[0].text
    assert "政策证据不足" in answer.fallback_notice
    assert answer.sources[0].source_ref == "policy-evidence:partial"
    validate_answer(answer, tool_result.sources)


def test_case_agent_build_answer_context_keeps_single_claimant_field() -> None:
    class ContextRepository:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_claimant_profile",
            "section_key": "claimant_profile",
            "payload": {
                "claimant_code": "SIM_PERSON_000001",
                "gender": "女",
                "age_group": "60-69岁",
                "insurance_type": "职工基本医疗保险",
                "allergy_history": "磺胺类药物过敏史",
            },
        },
        source_refs=["case:CASE-001:claimant_profile"],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_context"),
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "申报人性别",
            "granularity": "single_field",
            "missing_slots": [],
        },
        "user_message": SimpleNamespace(content="申报人性别"),
        "capability_results": [("query_claimant_profile", result)],
        "model_call_count": 0,
        "tool_call_count": 1,
    }
    service = SimpleNamespace(_repository=ContextRepository())

    next_state = build_answer_context_node(service, state)

    assert next_state["answer_context"]["facts"] == [
        {
            "section": "申报人基础信息",
            "label": "性别",
            "value": "女",
            "source_refs": ["case:CASE-001:claimant_profile"],
        }
    ]
    assert next_state["next_action"] == "resolve_answer_policy"


def test_case_agent_build_answer_context_routes_review_pass_question_to_status() -> None:
    class ContextRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_case_basic_info",
            "section_key": "case_basic_info",
            "payload": {
                "case_number": "CASE-001",
                "case_type": "门诊",
                "review_status": "reviewed",
            },
        },
        source_refs=["case:CASE-001:basic"],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_review_status"),
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "审核通过了嘛",
            "granularity": "single_field",
            "missing_slots": [],
        },
        "user_message": SimpleNamespace(content="审核通过了嘛"),
        "capability_results": [("query_case_basic_info", result)],
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = build_answer_context_node(
        SimpleNamespace(_repository=ContextRepository()),
        state,
    )

    assert next_state["answer_context"]["facts"] == [
        {
            "section": "案件基础信息",
            "label": "审核状态",
            "value": "已完成人工初审",
            "source_refs": ["case:CASE-001:basic"],
        }
    ]


def test_case_agent_build_answer_context_reuses_previous_answer_sources() -> None:
    class ContextRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {
                "run_id": run_id,
                "event_type": event_type,
                "message": message,
                "payload": payload or {},
            }

    source = CaseAgentSource(
        source_ref="policy-evidence:remote-1",
        source_type="policy_rag",
        title="跨省异地就医直接结算通知",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(label="证据片段", value="就医地目录、参保地待遇。"),
            ]
        ),
    )
    previous_answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="跨省异地就医直接结算实行就医地目录、参保地待遇的分工原则。",
                source_refs=["policy-evidence:remote-1"],
            )
        ],
        sources=[source],
    )
    clarification_answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[CaseAgentContentBlock(text="请补充需要查询的业务内容。")],
        sources=[],
        metadata={
            "intent": "clarification_required",
            "missing_slots": ["business_module"],
        },
    )
    state = {
        "run": SimpleNamespace(run_id="crun_reuse_context"),
        "answer_strategy": "reuse_previous_answer",
        "answer_rewrite_mode": "example",
        "reuse_answer_ref": "answer:msg-answer",
        "reuse_source_refs": ["policy-evidence:remote-1"],
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "用一个简单直观的例子说明上一轮回答",
            "granularity": "analysis",
            "missing_slots": [],
        },
        "user_message": SimpleNamespace(content="用一个简单直观的例子说明"),
        "recent_messages": [
            SimpleNamespace(
                message_id="msg-answer",
                role="assistant",
                content=previous_answer.plain_text,
                source_refs=["policy-evidence:remote-1"],
                answer_payload=previous_answer,
            ),
            SimpleNamespace(
                message_id="msg-clarification",
                role="assistant",
                content=clarification_answer.plain_text,
                source_refs=[],
                answer_payload=clarification_answer,
            ),
            SimpleNamespace(
                message_id="msg-follow",
                role="user",
                content="用一个简单直观的例子说明",
                source_refs=[],
                answer_payload=None,
            ),
        ],
        "capability_results": [],
        "available_sources": [],
        "model_call_count": 0,
        "tool_call_count": 0,
    }

    next_state = build_answer_context_node(
        SimpleNamespace(_repository=ContextRepository()),
        state,
    )

    answer_context = next_state["answer_context"]
    assert answer_context["source_refs"] == ["policy-evidence:remote-1"]
    assert next_state["available_sources"][0].source_ref == "policy-evidence:remote-1"
    previous_input = answer_context["analysis_inputs"][0]
    assert previous_input["section"] == "上一轮回答"
    assert previous_input["mode"] == "reuse_previous_answer"
    assert previous_input["previous_answer_ref"] == "answer:msg-answer"
    assert previous_input["previous_source_refs"] == ["policy-evidence:remote-1"]
    assert "就医地目录、参保地待遇" in previous_input["previous_answer"]


def test_case_agent_l3_uses_dedicated_expert_gateway() -> None:
    class RecordingExpertAgent:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return ExpertAgentResult(
                task_id=str(kwargs["task"].task_id),
                expert_task_type=str(kwargs["expert_task_type"]),
                status="ok",
                message="ok",
                payload={"status": "ok", "expert_answer": "ok"},
                source_refs=["policy:dedicated-gateway"],
            )

    service = CaseAgentService.__new__(CaseAgentService)
    main_gateway = object()
    expert_gateway = object()
    expert_agent = RecordingExpertAgent()
    service._repository = SimpleNamespace(
        record_tool_call=lambda *args, **kwargs: None,
        append_event=lambda *args, **kwargs: None,
    )
    service._gateway = main_gateway
    service._expert_gateway = expert_gateway
    service._tools = object()
    service._expert_agent = expert_agent
    service._policy_rag_client = object()
    service._materialize_expert_task = lambda **kwargs: SimpleNamespace(  # type: ignore[assignment]
        task_id="l3_policy_test",
        parent_run_id=kwargs["run_id"],
        case_id=kwargs["case_id"],
        model_dump=lambda mode="json": {
            "task_id": "l3_policy_test",
            "parent_run_id": kwargs["run_id"],
            "case_id": kwargs["case_id"],
        },
    )

    result = service._run_expert_capability(
        "crun_001",
        "policy_analysis",
        {
            "run": SimpleNamespace(run_id="crun_001", case_id="CASE-001"),
            "user_message": SimpleNamespace(content="上海异地报销规则是什么"),
            "slots": {},
        },
    )

    assert result.status == "ok"
    assert expert_agent.calls
    assert expert_agent.calls[0]["model_gateway"] is expert_gateway
    assert expert_agent.calls[0]["model_gateway"] is not main_gateway


def test_case_agent_l3_does_not_fallback_to_main_gateway() -> None:
    class RecordingExpertAgent:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return ExpertAgentResult(
                task_id=str(kwargs["task"].task_id),
                expert_task_type=str(kwargs["expert_task_type"]),
                status="ok",
                message="ok",
                payload={"status": "ok", "expert_answer": "ok"},
                source_refs=[],
            )

    service = CaseAgentService.__new__(CaseAgentService)
    main_gateway = object()
    expert_agent = RecordingExpertAgent()
    service._repository = SimpleNamespace(
        record_tool_call=lambda *args, **kwargs: None,
        append_event=lambda *args, **kwargs: None,
    )
    service._gateway = main_gateway
    service._expert_gateway = None
    service._tools = object()
    service._expert_agent = expert_agent
    service._policy_rag_client = object()
    service._materialize_expert_task = lambda **kwargs: SimpleNamespace(  # type: ignore[assignment]
        task_id="l3_policy_test",
        parent_run_id=kwargs["run_id"],
        case_id=kwargs["case_id"],
        model_dump=lambda mode="json": {
            "task_id": "l3_policy_test",
            "parent_run_id": kwargs["run_id"],
            "case_id": kwargs["case_id"],
        },
    )

    result = service._run_expert_capability(
        "crun_001",
        "policy_analysis",
        {
            "run": SimpleNamespace(run_id="crun_001", case_id="CASE-001"),
            "user_message": SimpleNamespace(content="policy question"),
            "slots": {},
        },
    )

    assert result.status == "ok"
    assert expert_agent.calls
    assert expert_agent.calls[0]["model_gateway"] is None
    assert expert_agent.calls[0]["model_gateway"] is not main_gateway


def test_case_agent_caps_policy_expert_sources_to_top_five() -> None:
    service = CaseAgentService.__new__(CaseAgentService)
    policy_evidence = [
        {
            "source_ref": f"policy-evidence:{index}",
            "title": f"政策证据 {index}",
            "excerpt": "政策证据片段",
            "jurisdiction": "national",
            "policy_domain": "remote_medical",
            "content_type": "policy_text",
            "source_url": "https://example.test/policy",
        }
        for index in range(1, 9)
    ]
    result = ExpertAgentResult(
        task_id="l3_policy_top_five",
        expert_task_type="policy_analysis",
        status="ok",
        message="ok",
        payload={
            "status": "ok",
            "expert_answer": "政策专家回答。",
            "policy_evidence": policy_evidence,
        },
        source_refs=[item["source_ref"] for item in policy_evidence],
    )

    tool_result = service._expert_result_to_tool_result(
        case_id="CASE-001",
        capability="ask_policy_expert",
        result=result,
    )

    assert tool_result.status == "success"
    assert len(tool_result.source_refs) == 5
    assert len(tool_result.sources) == 5
    assert len(tool_result.payload["payload"]["policy_evidence"]) == 5
    assert tool_result.source_refs == [
        "policy-evidence:1",
        "policy-evidence:2",
        "policy-evidence:3",
        "policy-evidence:4",
        "policy-evidence:5",
    ]


def test_policy_table_row_source_uses_clean_business_excerpt_and_used_sources_only() -> None:
    service = CaseAgentService.__new__(CaseAgentService)
    used_ref = "policy-evidence:used"
    unused_ref = "policy-evidence:unused"
    result = ExpertAgentResult(
        task_id="l3_policy_clean_source",
        expert_task_type="policy_analysis",
        status="ok",
        message="ok",
        payload={
            "status": "ok",
            "expert_answer": "急诊场景需保留诊断证明。[1]",
            "answer_markdown": "急诊场景需保留诊断证明。[1]",
            "claims": [
                {
                    "claim_id": "claim_1",
                    "need_id": "need_1",
                    "need_text": "急诊材料要求",
                    "text": "急诊场景需保留诊断证明。",
                    "source_refs": [used_ref],
                    "citation_ids": ["cit_1"],
                }
            ],
            "citations": [
                {
                    "citation_id": "cit_1",
                    "label": 1,
                    "claim_id": "claim_1",
                    "source_refs": [used_ref],
                }
            ],
            "policy_evidence": [
                {
                    "evidence_ref": used_ref,
                    "source_ref": used_ref,
                    "title": "北京手工报销材料链字段",
                    "content_type": "table_row",
                    "jurisdiction": "beijing",
                    "source_url": "https://example.test/outer",
                    "excerpt": (
                        "资料标题: 北京手工报销材料链字段\n"
                        "政策领域: manual_reimbursement\n"
                        "field_key: emergency_context_marker\n"
                        "material_name: 急诊诊断证明\n"
                        "source_url: https://example.test/row\n"
                        "evidence_text: 未出示医保凭证就医的，需保留收据、处方、诊断证明等材料。\n"
                        "case_relevance: internal-only"
                    ),
                },
                {
                    "evidence_ref": unused_ref,
                    "source_ref": unused_ref,
                    "title": "未使用证据",
                    "content_type": "table_row",
                    "excerpt": "material_name: 费用收据",
                },
            ],
        },
        source_refs=[used_ref, unused_ref],
    )

    tool_result = service._expert_result_to_tool_result(
        case_id="CASE-001",
        capability="ask_policy_expert",
        result=result,
    )

    assert tool_result.source_refs == [used_ref]
    assert len(tool_result.sources) == 1
    fields = {
        field.label: field.value
        for field in tool_result.sources[0].detail.fields
    }
    assert fields["证据片段"] == "未出示医保凭证就医的，需保留收据、处方、诊断证明等材料。"
    assert tool_result.sources[0].title == "北京市医保政策原文"
    assert tool_result.sources[0].version is None
    assert tool_result.sources[0].metadata["source_url"] == "https://example.test/row"
    assert not any(
        marker in fields["证据片段"]
        for marker in ("资料标题", "政策领域", "field_key", "case_relevance", "source_id")
    )


def test_policy_handoff_rebuilds_markdown_after_unavailable_citation_is_removed() -> None:
    class AnswerRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.event = {"run_id": run_id, "event_type": event_type, "payload": payload or {}}

    source = CaseAgentSource(
        source_ref="policy-evidence:available",
        source_type="policy_rag",
        title="可用政策",
        detail=CaseAgentSourceDetail(
            fields=[CaseAgentSourceDetailField(label="证据片段", value="可用政策证据。")]
        ),
    )
    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "section_key": "policy_expert",
            "payload": {
                "status": "ok",
                "expert_answer": "可用结论。[1]\n不可用结论。[2]",
                "answer_markdown": "1. 可用结论。[1]\n2. 不可用结论。[2]",
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "need_id": "need_1",
                        "need_text": "可用信息点",
                        "text": "可用结论。",
                        "source_refs": [source.source_ref],
                        "citation_ids": ["cit_1"],
                    },
                    {
                        "claim_id": "claim_2",
                        "need_id": "need_2",
                        "need_text": "不可用信息点",
                        "text": "不可用结论。",
                        "source_refs": ["policy-evidence:missing"],
                        "citation_ids": ["cit_2"],
                    },
                ],
                "citations": [
                    {
                        "citation_id": "cit_1",
                        "label": 1,
                        "claim_id": "claim_1",
                        "source_refs": [source.source_ref],
                    },
                    {
                        "citation_id": "cit_2",
                        "label": 2,
                        "claim_id": "claim_2",
                        "source_refs": ["policy-evidence:missing"],
                    },
                ],
            },
            "expert_task_id": "l3_policy_citation_closure",
        },
        source_refs=[source.source_ref],
        sources=[source],
    )
    service = SimpleNamespace(
        _repository=AnswerRepository(),
        _fast_mode_enabled=True,
        _gateway=SimpleNamespace(complete=lambda **_: None),
        _generator_model="deepseek-v4-flash",
    )
    state = {
        "run": SimpleNamespace(run_id="crun_citation_closure"),
        "intent": "policy_expert_query",
        "answer_policy": {"display_mode": "grounded", "requires_citation": True},
        "context_plan": {"display_mode": "grounded"},
        "capability_results": [("ask_policy_expert", result)],
        "available_sources": [source],
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = generate_answer_node(service, state)
    answer = CaseAgentAnswer.model_validate_json(next_state["final_response"].content)

    assert answer.answer_markdown == "可用结论。[1]"
    assert [citation.label for citation in answer.citations] == [1]
    assert answer.claims[0].citation_ids == ["cit_1"]
    assert len(answer.claims) == 1


def test_expert_analysis_insufficient_keeps_adopted_policy_evidence() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    task = ExpertAnalysisTask(
        task_id="l3_policy_insufficient",
        parent_run_id="crun_insufficient",
        case_id="CASE-001",
        goal="查询普通异地门诊手工报销材料",
        user_question="普通异地门诊手工报销需要哪些材料？",
    )
    state = {
        "task": task,
        "runtime": {},
        "status": "insufficient",
        "message": "",
        "answerability_check": {
            "answerability": "insufficient",
            "covered_slots": ["异地就医备案和直接结算流程"],
            "missing_slots": ["手工报销材料明细"],
            "usable_evidence_refs": [],
            "next_action": "insufficient",
            "reason_code": "missing_material_list",
            "reason": "缺少手工报销材料清单依据。",
        },
        "case_facts_used": [],
        "adopted_policy_evidence": [
            {
                "evidence_ref": "policy-node:partial",
                "source_ref": "policy-evidence:partial",
                "title": "跨省异地就医直接结算政策",
                "excerpt": "异地就医备案和直接结算流程相关规定。",
                "jurisdiction": "国家",
                "policy_domain": "异地就医",
                "content_type": "policy",
            }
        ],
    }

    next_state = service._return_unavailable_node(state)

    assert next_state["status"] == "insufficient"
    assert next_state["source_refs"] == ["policy-evidence:partial"]
    assert next_state["result"]["policy_evidence"][0]["source_ref"] == "policy-evidence:partial"
    assert (
        next_state["result"]["answerability_summary"]["usable_evidence_refs"]
        == ["policy-node:partial"]
    )
    assert "手工报销材料明细" in next_state["result"]["expert_answer"]


def test_expert_safe_snapshot_keeps_policy_excerpts() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    task = ExpertAnalysisTask(
        task_id="l3_policy_snapshot",
        parent_run_id="crun_snapshot",
        case_id="CASE-001",
        goal="查询零星报销材料",
        user_question="零星报销门诊费用需要哪些材料？",
    )

    snapshot = service._safe_state_snapshot(
        {
            "task": task,
            "status": "ok",
            "message": "ok",
            "adopted_policy_evidence": [
                {
                    "evidence_ref": "policy-node:1",
                    "source_ref": "policy-source:1",
                    "title": "上海门诊零星报销",
                    "excerpt": "门诊零星报销需提供身份证、社保卡和医疗费专用收据。",
                    "jurisdiction": "shanghai",
                    "policy_domain": "manual_reimbursement",
                    "content_type": "policy_text",
                    "source_url": "https://example.test/policy",
                    "version": "v1",
                    "metadata": {"node_id": "node-1"},
                }
            ],
        }
    )

    evidence = snapshot["adopted_policy_evidence"][0]
    assert evidence["excerpt"] == "门诊零星报销需提供身份证、社保卡和医疗费专用收据。"
    assert evidence["source_url"] == "https://example.test/policy"
    assert evidence["version"] == "v1"


def test_expert_analysis_accepts_drug_price_reference_evidence() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)

    evidence = service._policy_evidence_from_candidate(
        {
            "node_id": "node_drug_price_ibuprofen",
            "source_id": "reference_shanghai_drug_product_price_20260820_full",
            "title": "上海药品价格参考：布洛芬缓释胶囊 中美天津史克制药有限公司 0.3g",
            "text": (
                "上海药品产品价格参考。药品名称：布洛芬缓释胶囊；"
                "价格周期：上周，2026-08-10至2026-08-16；"
                "医保药店周均价：13.12元，医保药店周价格区间：0.10至15.30元；"
                "数据用途：药品产品价格参考，不作为医保报销政策依据。"
            ),
            "jurisdiction": "shanghai",
            "policy_domain": "drug_product_price_reference",
            "content_type": "table_row",
            "source_url": "https://bjxt.smiic.net.cn/ypcx/?sessionid=#/pages/drugs-query/search",
            "rank": 1,
            "score": 0.95,
            "can_cite_as_policy_basis": False,
        },
        1,
    )

    assert evidence is not None
    assert evidence.used_for == "drug_price_reference"
    assert evidence.policy_domain == "drug_product_price_reference"
    assert evidence.metadata["can_cite_as_policy_basis"] is False
    assert "13.12元" in evidence.excerpt


def test_expert_analysis_formats_drug_price_reference_answer() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    first = service._policy_evidence_from_candidate(
        {
            "node_id": "node_drug_price_ibuprofen_1",
            "source_id": "reference_shanghai_drug_product_price_20260820_full",
            "title": "上海药品价格参考：布洛芬缓释胶囊 中美天津史克制药有限公司 0.3g",
            "text": (
                "上海药品产品价格参考。 药品名称：布洛芬缓释胶囊；"
                "注册名称：布洛芬缓释胶囊；药品编码：XM01AEB173E003010200969；"
                "生产企业：中美天津史克制药有限公司；规格：0.3g；剂型：缓释胶囊；"
                "最小包装单位：盒；最小包装数量：20；药品类型：西药；价格地区：上海；"
                "价格周期：上周，2026-08-10至2026-08-16；"
                "医保药店周均价：13.12元，医保药店周价格区间：0.1至15.3元；"
                "数据用途：药品产品价格参考，不作为医保报销政策依据。"
            ),
            "jurisdiction": "shanghai",
            "policy_domain": "drug_product_price_reference",
            "content_type": "table_row",
            "source_url": "https://bjxt.smiic.net.cn/ypcx/?sessionid=#/pages/drugs-query/search",
            "rank": 1,
            "score": 0.95,
            "can_cite_as_policy_basis": False,
        },
        1,
    )
    second = service._policy_evidence_from_candidate(
        {
            "node_id": "node_drug_price_ibuprofen_2",
            "source_id": "reference_shanghai_drug_product_price_20260820_full",
            "title": "上海药品价格参考：布洛芬缓释胶囊 北京红林制药有限公司 0.3g",
            "text": (
                "上海药品产品价格参考。 药品名称：布洛芬缓释胶囊；"
                "注册名称：布洛芬缓释胶囊；药品编码：XM01AEB173E003010600044；"
                "生产企业：北京红林制药有限公司；规格：0.3g；剂型：缓释胶囊；"
                "最小包装单位：盒；最小包装数量：30；药品类型：西药；价格地区：上海；"
                "价格周期：上周，2026-08-10至2026-08-16；"
                "医保药店周均价：8.53元，医保药店周价格区间：5.78至9元；"
                "数据用途：药品产品价格参考，不作为医保报销政策依据。"
            ),
            "jurisdiction": "shanghai",
            "policy_domain": "drug_product_price_reference",
            "content_type": "table_row",
            "source_url": "https://bjxt.smiic.net.cn/ypcx/?sessionid=#/pages/drugs-query/search",
            "rank": 2,
            "score": 0.9,
            "can_cite_as_policy_basis": False,
        },
        2,
    )
    assert first is not None
    assert second is not None

    answer = service._deterministic_expert_answer(
        ExpertAnalysisTask(
            task_id="task-drug-price",
            parent_run_id="run-1",
            case_id="CASE-1",
            user_question="上海的布洛芬均价是多少",
            filters={
                "jurisdiction": "shanghai",
                "policy_domain": "drug_product_price_reference",
            },
        ),
        [first, second],
        AnswerabilityCheck(
            answerability="partial",
            covered_slots=["drug_price_reference"],
            missing_slots=[],
            usable_evidence_refs=[first.evidence_ref, second.evidence_ref],
            next_action="synthesize",
            reason_code="reference_evidence_found",
            reason="Drug price reference evidence was found.",
        ),
    )

    assert "不能合并成一个唯一均价" in answer
    assert "13.12元" in answer
    assert "8.53元" in answer
    assert "目前只能形成部分参考信息" not in answer
    assert "不作为医保报销政策依据" in answer


def test_policy_claim_first_remote_benefit_split_extracts_cited_claims() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_remote",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医待遇分工",
        user_question="跨省异地就医中，就医地目录与参保地待遇如何分工？",
        filters={
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:remote-split",
            source_ref="policy-evidence:remote-split",
            title="跨省异地就医直接结算政策",
            excerpt=(
                "跨省异地就医直接结算原则上执行就医地规定的支付范围及有关规定"
                "（基本医疗保险药品、医疗服务项目和医用耗材等支付范围），"
                "执行参保地规定的基本医疗保险基金起付标准、支付比例、最高支付限额、"
                "门诊慢特病病种范围等有关政策。"
            ),
            jurisdiction="national",
            policy_domain="remote_medical",
            content_type="policy_text",
        )
    ]
    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["remote_benefit_split"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-split"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖分工口径。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "医保药品、医疗服务项目和医用耗材等支付范围原则上按就医地规定执行。[1]" in answer.answer_markdown
    assert "基金起付标准、支付比例、最高支付限额和门诊慢特病病种范围等待遇参数按参保地政策执行。[1]" in answer.answer_markdown
    assert len(answer.claims) == 2
    assert answer.citations[0].source_refs == ["policy-evidence:remote-split"]


def test_policy_claim_first_remote_split_omits_unasked_manual_background() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_remote_focused",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算",
        user_question="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？备案成功后可在哪些机构就医？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical", "remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:remote-split",
            source_ref="policy-evidence:remote-split",
            title="跨省异地就医直接结算政策",
            excerpt=(
                "参保人员直接结算的住院、普通门诊和门诊慢特病医疗费用，"
                "原则上执行就医地规定的支付范围及有关规定，执行参保地规定的"
                "基本医疗保险基金起付标准、支付比例、最高支付限额、门诊慢特病病种范围等有关政策。"
            ),
            jurisdiction="national",
            policy_domain="remote_medical",
            content_type="policy_text",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:manual-emergency",
            source_ref="policy-evidence:manual-emergency",
            title="异地急诊留观手工报销问答",
            excerpt=(
                "按照国家统一要求，急诊留观费用暂不能实现异地直接结算，"
                "参保人员可将异地就医票据及相关报销材料交给本人所属单位"
                "（社保所），由单位（社保所）向所属区医保经办机构申请手工报销。"
            ),
            jurisdiction="beijing",
            policy_domain="remote_medical_manual_reimbursement",
            content_type="policy_text",
        ),
    ]
    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="partial",
            covered_slots=["remote_benefit_split"],
            missing_slots=["备案成功后可就医机构范围"],
            usable_evidence_refs=["policy-node:remote-split", "policy-node:manual-emergency"],
            next_action="synthesize",
            reason_code="partial_slot_coverage",
            reason="仅覆盖直接结算支付规则。",
        ),
        question_slots=[
            {"slot_id": "remote_benefit_split"},
            {"slot_id": "remote_filing"},
        ],
        filters=task.filters,
    )

    assert "就医地规定执行" in answer.answer_markdown
    assert "参保地政策执行" in answer.answer_markdown
    assert "手工报销" not in answer.answer_markdown
    assert "急诊留观" not in answer.answer_markdown
    assert "社保所" not in answer.answer_markdown


def test_policy_claim_first_materials_and_time_do_not_dump_service_metadata() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_materials",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询零星报销材料和时限",
        user_question="审核参保人零星报销门诊费用时，应要求提供哪些必要材料？法定办结时限是多少？",
        filters={
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["service_guide"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:materials",
            source_ref="policy-evidence:materials",
            title="门诊零星报销办事指南",
            excerpt=(
                "实施主体 上海市医疗保险事业管理中心 业务办理项编码 12345 "
                "法定办结时限 30(工作日) 30 到现场次数 1 次 "
                "申请材料目录 材料名称 材料必要性 身份证 必要 社保卡 必要 "
                "医疗费专用收据 必要 病史资料 必要 银行卡 必要 医保卡 非必要"
            ),
            jurisdiction="shanghai",
            policy_domain="manual_reimbursement",
            content_type="service_guide",
        )
    ]
    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["required_materials", "statutory_processing_time"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:materials"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖材料和时限。",
        ),
        question_slots=[{"slot_id": "manual_reimbursement"}],
        filters=task.filters,
    )

    assert "必要材料包括" in answer.answer_markdown
    assert "身份证" in answer.answer_markdown
    assert "医疗费专用收据" in answer.answer_markdown
    assert "法定办结时限为30个工作日。[1]" in answer.answer_markdown
    assert "实施主体" not in answer.expert_answer
    assert "业务办理项编码" not in answer.expert_answer


def test_policy_claim_first_broad_shanghai_scope_avoids_random_drug_row() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_shanghai_scope",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询上海支付范围核验",
        user_question="上海药品、胸部 CT 和医用耗材支付范围如何核验？",
        filters={
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_catalog", "medical_service_price", "shanghai_payment_scope"],
            "content_type": ["table_row", "policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:random-drug",
            source_ref="policy-evidence:random-drug",
            title="上海医保药品目录行",
            excerpt="医保类别：乙类\n目录编号：176\n药品名称：水溶性维生素\n剂型：注射剂",
            jurisdiction="shanghai",
            policy_domain="drug_catalog",
            content_type="table_row",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:drug-policy",
            source_ref="policy-evidence:drug-policy",
            title="上海医保药品目录政策说明",
            excerpt="医保药品目录应核验药品目录、医保类别、甲类乙类和限定支付范围。",
            jurisdiction="shanghai",
            policy_domain="drug_catalog",
            content_type="policy_text",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:ct-price",
            source_ref="policy-evidence:ct-price",
            title="上海医疗服务价格项目",
            excerpt="项目名称：胸部 CT 平扫\n编码：CT001\n计价单位：次\n收费标准：520元\n内容说明：包含扫描和出具诊断报告。",
            jurisdiction="shanghai",
            policy_domain="medical_service_price",
            content_type="table_row",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:consumables",
            source_ref="policy-evidence:consumables",
            title="上海医用耗材支付范围政策",
            excerpt="部分医用耗材纳入本市基本医疗保险支付范围，并按基金支付、支付办法、医保类别和先自负比例执行。",
            jurisdiction="shanghai",
            policy_domain="shanghai_payment_scope",
            content_type="policy_text",
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["drug_catalog", "medical_service_price", "consumable_payment_scope"],
            missing_slots=[],
            usable_evidence_refs=[
                "policy-node:drug-policy",
                "policy-node:ct-price",
                "policy-node:consumables",
            ],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖上海支付范围核验。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "水溶性维生素" not in answer.answer_markdown
    assert "药品支付范围应核验药品是否纳入医保药品目录" in answer.answer_markdown
    assert "胸部 CT 平扫的编码为CT001" in answer.answer_markdown
    assert "医用耗材支付范围应核验是否纳入医保支付范围" in answer.answer_markdown


def test_policy_claim_first_drug_catalog_row_extracts_english_jsonl_fields() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_drug_row",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询药品目录字段",
        user_question="处方中的便通片在上海医保药品目录中的医保类别和本地支付比例是多少？",
        filters={
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["table_row"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:wrong-row",
            source_ref="policy-evidence:wrong-row",
            title="上海医保药品目录行",
            excerpt=(
                "insurance_class: 乙\ncatalog_no: 68\ndrug_name: 便通\n"
                "dosage_form: 口服常释剂型\nlocal_payment_policy: 20%\n"
                "raw_line: 乙 68 便通口服常释剂型 20%"
            ),
            jurisdiction="shanghai",
            policy_domain="drug_catalog",
            content_type="table_row",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:drug-row",
            source_ref="policy-evidence:drug-row",
            title="上海医保药品目录行",
            excerpt=(
                "insurance_class: 乙\ncatalog_no: 68\ndrug_name: 便通\n"
                "dosage_form: 片\nlocal_payment_policy: 10%\nremark: (胶囊)\n"
                "raw_line: 乙 68 便通片(胶囊) 10%"
            ),
            jurisdiction="shanghai",
            policy_domain="drug_catalog",
            content_type="table_row",
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["drug_catalog"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:wrong-row", "policy-node:drug-row"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖药品目录字段。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "医保类别为乙类" in answer.answer_markdown
    assert "目录编号为68" in answer.answer_markdown
    assert "本地支付比例为10%" in answer.answer_markdown
    assert "备注或限定支付范围" not in answer.answer_markdown
    assert "20%" not in answer.answer_markdown
    assert "口服常释剂型" not in answer.answer_markdown


def test_policy_claim_first_drug_catalog_policy_text_extracts_dosage_definition() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_drug_dosage",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询药品目录剂型说明",
        user_question="北京医保药品目录中普通片剂包括哪些剂型？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:drug-dosage-policy",
            source_ref="policy-evidence:drug-dosage-policy",
            title="北京医保药品目录说明",
            excerpt="药品目录说明：普通片剂包括素片、糖衣片、薄膜衣片、异形片、分散片、咀嚼片。",
            jurisdiction="beijing",
            policy_domain="drug_catalog",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["drug_catalog"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:drug-dosage-policy"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖剂型说明。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "普通片剂包括素片、糖衣片、薄膜衣片、异形片、分散片、咀嚼片。[1]" in answer.answer_markdown
    assert "医保类别为" not in answer.answer_markdown


def test_policy_claim_first_drug_dosage_definition_trims_adjacent_table_groups() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_drug_dosage_table_groups",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询药品目录剂型归类",
        user_question="请问北京市医保药品目录中的普通片剂具体包含哪些剂型？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:drug-dosage-table",
            source_ref="policy-evidence:drug-dosage-table",
            title="北京医保药品目录说明",
            excerpt=(
                "合并归类的剂型见下表：普通片剂（片、素片、肠溶片、包衣片、"
                "薄膜衣片、糖衣片、浸膏片、分散片、划痕片）、硬胶囊、软胶囊、"
                "缓释片、口服溶液剂。"
            ),
            jurisdiction="beijing",
            policy_domain="drug_catalog",
            content_type="policy_text",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:drug-payment-background",
            source_ref="policy-evidence:drug-payment-background",
            title="北京医保药品支付说明",
            excerpt="药品支付范围应核验药品是否纳入医保药品目录、医保类别以及限定支付范围。",
            jurisdiction="beijing",
            policy_domain="drug_catalog",
            content_type="policy_text",
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["drug_catalog"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:drug-dosage-table"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖剂型说明。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "普通片剂包括片、素片、肠溶片、包衣片、薄膜衣片、糖衣片、浸膏片、分散片、划痕片。[1]" in answer.answer_markdown
    assert "硬胶囊" not in answer.answer_markdown
    assert "药品支付范围" not in answer.answer_markdown
    assert "当前可引用证据仅覆盖部分核验点" not in answer.answer_markdown


def test_policy_claim_first_deduplicates_same_claim_and_merges_citations() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_dedupe",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询补备案后手工报销政策",
        user_question="参保人员跨省异地就医出院自费结算后补办备案，能否申请医保手工报销？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
        },
    )
    requirement = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_filing",
        label="异地备案与急诊例外",
        question_span=task.user_question,
        filters=task.filters,
        fact_schema="policy_rule",
        extractor_id="policy_rule_span",
    )
    facts = [
        ExtractedFact(
            fact_id="fact_1",
            requirement_id="req_1",
            slot_id="remote_filing",
            fact_type="policy_rule_slot_window",
            value={"support_role": "direct_answer", "llm_score": 0.91, "quote_verified": True},
            display_text="跨省异地就医参保人员出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销。",
            evidence_refs=["policy-node:remote-filing-a"],
            source_refs=["policy-evidence:remote-filing-a"],
            evidence_text="出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销",
            confidence=0.91,
        ),
        ExtractedFact(
            fact_id="fact_2",
            requirement_id="req_1",
            slot_id="remote_filing",
            fact_type="policy_rule_slot_window",
            value={"support_role": "direct_answer", "llm_score": 0.9, "quote_verified": True},
            display_text="跨省异地就医参保人员出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销。",
            evidence_refs=["policy-node:remote-filing-b"],
            source_refs=["policy-evidence:remote-filing-b"],
            evidence_text="出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销",
            confidence=0.9,
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=[],
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["remote_filing"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-filing-a"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖补备案后手工报销路径。",
        ),
        question_slots=[],
        filters=task.filters,
        precomputed_requirements=[requirement],
        precomputed_facts=facts,
        precomputed_matches=[],
    )

    assert answer.answer_markdown.count("可以按参保地规定申请医保手工报销") == 1
    assert answer.answer_markdown.endswith("[1][2]")
    assert len(answer.claims) == 1
    assert len(answer.claims[0].source_refs) == 2


def test_policy_claim_first_material_verification_wording_extracts_materials() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_material_wording",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询急诊留观手工报销材料",
        user_question="异地急诊留观费用申请手工报销时，现有政策要求核验哪些材料？",
        filters={
            "policy_domain": ["remote_medical_manual_reimbursement", "manual_reimbursement"],
            "content_type": ["service_guide", "policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:remote-filing",
            source_ref="policy-evidence:remote-filing",
            title="异地就医备案政策",
            excerpt="参保人员出院自费结算后可按参保地规定补办备案手续，符合条件的可申请手工报销。",
            jurisdiction="beijing",
            policy_domain="remote_medical",
            content_type="policy_text",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:manual-materials",
            source_ref="policy-evidence:manual-materials",
            title="北京外埠就医手工报销材料",
            excerpt="申请材料目录 材料名称 材料必要性 诊疗证明 必要 处方底方 必要 费用清单 必要 医疗费专用收据 必要。",
            jurisdiction="beijing",
            policy_domain="manual_reimbursement",
            content_type="service_guide",
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["required_materials"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:manual-materials"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖材料要求。",
        ),
        question_slots=[],
        filters=task.filters,
    )

    assert "必要材料包括" in answer.answer_markdown
    assert "诊疗证明" in answer.answer_markdown
    assert "处方底方" in answer.answer_markdown
    assert "费用清单" in answer.answer_markdown
    assert "政策宣传" not in answer.answer_markdown


def test_policy_claim_first_manual_reimbursement_scenario_slots_answer_by_scenario() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_manual_scenarios",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询北京手工报销场景化材料和凭证",
        user_question="审核北京参保人员手工报销时，涉及外埠就医、定点医药机构记账结算和急诊就医，应分别核验哪些材料或结算凭证？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["table_row", "policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:foreign-treatment",
            source_ref="policy-evidence:foreign-treatment",
            title="北京手工报销材料链字段",
            excerpt=(
                "参保人员按规定在外埠发生的医疗费用，先由个人支付。按规定由基本医疗保险统筹基金支付的，"
                "自医疗费用发生后三个月内由用人单位汇总，持参保人员在外埠定点医疗机构的诊疗证明、"
                "处方底方、费用清单、费用收据，填写《北京市医疗保险手工报销(外埠就医)费用申报结算明细表》，"
                "报区、县医保中心审核结算。"
            ),
            jurisdiction="beijing",
            policy_domain="manual_reimbursement",
            content_type="table_row",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:account-voucher",
            source_ref="policy-evidence:account-voucher",
            title="北京记账结算凭证",
            excerpt=(
                "参保人员在定点医疗机构、定点零售药店发生的按规定应由个人帐户支付的费用，"
                "定点医疗机构和定点零售药店对个人帐户支付部分记账，填写"
                "《北京市医疗保险门急诊(药店)费用审核结算凭证》，与区、县医保中心进行结算。"
            ),
            jurisdiction="beijing",
            policy_domain="manual_reimbursement",
            content_type="table_row",
        ),
        PolicyEvidence(
            evidence_ref="policy-node:emergency-materials",
            source_ref="policy-evidence:emergency-materials",
            title="社保卡就医必看：报销详解",
            excerpt=(
                "急诊就医未出示社保卡或医保电子凭证时，个人需全额垫付医疗费用，"
                "保留好医院开具的收据、处方、诊断证明等材料，交由单位（或便民服务中心）"
                "到区医保经办机构进行手工报销。"
            ),
            jurisdiction="beijing",
            policy_domain="manual_reimbursement",
            content_type="policy_text",
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=[
                "外埠就医手工报销材料",
                "定点医药机构记账结算凭证",
                "急诊就医手工报销材料",
            ],
            missing_slots=[],
            usable_evidence_refs=[
                "policy-node:foreign-treatment",
                "policy-node:account-voucher",
                "policy-node:emergency-materials",
            ],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖三个手工报销场景。",
        ),
        question_slots=[
            {"slot_id": "foreign_treatment_manual_reimbursement"},
            {"slot_id": "account_settlement_voucher"},
            {"slot_id": "emergency_manual_reimbursement_materials"},
        ],
        filters=task.filters,
    )

    assert "外埠就医手工报销需持外埠定点医疗机构的诊疗证明、处方底方、费用清单和费用收据" in answer.answer_markdown
    assert "定点医疗机构和定点零售药店对个人账户支付部分记账结算时" in answer.answer_markdown
    assert "急诊就医未出示社保卡或医保电子凭证时" in answer.answer_markdown
    assert "材料要求中提到" not in answer.answer_markdown
    assert "当前可引用证据仅覆盖部分核验点" not in answer.answer_markdown


def test_policy_slot_window_judge_extracts_quote_verified_remote_split() -> None:
    requirement = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_span="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
        fact_schema="remote_benefit_split",
        extractor_id="remote_benefit_split",
    )
    evidence = NormalizedPolicyEvidence(
        evidence_id="policy-node:remote-direct",
        source_ref="policy-evidence:remote-direct",
        title="跨省异地就医直接结算政策",
        content=(
            "（七）调整本市参保人员特殊病跨省异地就医备案定点医疗机构数量。"
            "跨省异地就医直接结算的住院、普通门诊和门诊慢特病医疗费用，"
            "原则上执行就医地规定的支付范围及有关规定，执行参保地规定的"
            "基本医疗保险基金起付标准、支付比例、最高支付限额、门诊慢特病"
            "病种范围等有关政策。"
        ),
        jurisdiction="beijing",
        policy_domain="remote_medical",
        content_type="policy_text",
    )

    windows = build_sentence_windows(
        requirements=[requirement],
        evidence=[evidence],
        filters=requirement.filters,
        user_question=requirement.question_span,
    )
    candidates = select_candidate_windows(windows, per_slot=2)
    target_window = candidates[0]
    facts, matches, diagnostics = extract_verified_facts_from_judgement(
        requirements=[requirement],
        candidate_windows=candidates,
        payload={
            "window_results": [
                {
                    "slot_id": "remote_benefit_split",
                    "window_id": target_window["window_id"],
                    "llm_score": 0.96,
                    "support_role": "direct_answer",
                    "facts": [
                        {
                            "fact": "医保支付范围及有关规定原则上按就医地规定执行。",
                            "evidence_quote": "执行就医地规定的支付范围及有关规定",
                        },
                        {
                            "fact": "基金起付标准、支付比例、最高支付限额等按参保地规定执行。",
                            "evidence_quote": "执行参保地规定的基本医疗保险基金起付标准、支付比例、最高支付限额",
                        },
                    ],
                },
                {
                    "slot_id": "remote_benefit_split",
                    "window_id": target_window["window_id"],
                    "llm_score": 0.99,
                    "support_role": "direct_answer",
                    "facts": [
                        {
                            "fact": "这条事实没有原文 quote。",
                            "evidence_quote": "原文中不存在的改写片段",
                        }
                    ],
                },
            ]
        },
    )

    assert len(facts) == 2
    assert matches[0].coverage_status == "covered"
    assert diagnostics["rejected_count"] == 1
    assert facts[0].value["quote_verified"] is True


def test_policy_claim_first_direct_settlement_uses_remote_payment_and_institution_slots() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_remote_direct_settlement",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算支付规则和备案后机构范围",
        user_question="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？备案成功后可在哪些机构就医？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:remote-direct",
            source_ref="policy-evidence:remote-direct",
            title="关于进一步做好基本医疗保险跨省异地就医直接结算工作的通知",
            excerpt=(
                "跨省异地就医直接结算的住院、普通门诊和门诊慢特病医疗费用，"
                "原则上执行就医地规定的支付范围及有关规定，执行参保地规定的"
                "基本医疗保险基金起付标准、支付比例、最高支付限额、门诊慢特病"
                "病种范围等有关政策。本市参保人员办理跨省异地就医备案时，"
                "原则上只需备案到就医地所在的统筹地区，备案成功后可在该统筹"
                "地区内所有定点医药机构按规定就医结算。"
            ),
            jurisdiction="beijing",
            policy_domain="remote_medical",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["就医地目录与参保地待遇分工", "备案后就医机构范围"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-direct"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖直接结算支付规则和备案后机构范围。",
        ),
        question_slots=[
            {"slot_id": "remote_benefit_split"},
            {"slot_id": "remote_filing_institution_scope"},
        ],
        filters=task.filters,
    )

    assert "住院、普通门诊和门诊慢特病医疗费用" in answer.answer_markdown
    assert "支付范围原则上按就医地规定执行" in answer.answer_markdown
    assert "起付标准、支付比例、最高支付限额" in answer.answer_markdown
    assert "门诊慢特病病种范围" in answer.answer_markdown
    assert "统筹地区内所有定点医药机构" in answer.answer_markdown
    assert "特殊病跨省异地就医备案定点医疗机构数量" not in answer.answer_markdown


def test_policy_slot_window_backfills_missing_required_fields_from_deterministic_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_remote_direct_backfill",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算支付规则和备案后机构范围",
        user_question="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？备案成功后可在哪些机构就医？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    evidence = PolicyEvidence(
        evidence_ref="policy-node:remote-direct",
        source_ref="policy-evidence:remote-direct",
        title="关于进一步做好基本医疗保险跨省异地就医直接结算工作的通知",
        excerpt=(
            "跨省异地就医直接结算的住院、普通门诊和门诊慢特病医疗费用，"
            "原则上执行就医地规定的支付范围及有关规定，执行参保地规定的"
            "基本医疗保险基金起付标准、支付比例、最高支付限额、门诊慢特病"
            "病种范围等有关政策。本市参保人员办理跨省异地就医备案时，"
            "原则上只需备案到就医地所在的统筹地区，备案成功后可在该统筹"
            "地区内所有定点医药机构按规定就医结算。"
        ),
        jurisdiction="beijing",
        policy_domain="remote_medical",
        content_type="policy_text",
    )

    def fake_llm_json(*args: object, **kwargs: object) -> dict[str, object]:
        messages = kwargs["messages"]
        payload = json.loads(messages[1]["content"])  # type: ignore[index]
        target_window = next(
            item for item in payload["candidate_windows"]
            if item["slot_id"] == "remote_benefit_split"
        )
        return {
            "window_results": [
                {
                    "slot_id": "remote_benefit_split",
                    "window_id": target_window["window_id"],
                    "llm_score": 0.96,
                    "support_role": "direct_answer",
                    "facts": [
                        {
                            "fact": "医保支付范围及有关规定原则上按就医地规定执行。",
                            "evidence_quote": "执行就医地规定的支付范围及有关规定",
                            "matched_required_fields": ["就医地支付范围"],
                        },
                        {
                            "fact": "基金起付标准、支付比例、最高支付限额等按参保地规定执行。",
                            "evidence_quote": "执行参保地规定的基本医疗保险基金起付标准、支付比例、最高支付限额",
                            "matched_required_fields": ["参保地起付标准", "参保地支付比例", "参保地最高支付限额"],
                        },
                    ],
                }
            ]
        }

    monkeypatch.setattr(service, "_llm_json", fake_llm_json)
    state = service._slot_window_judge_node(  # type: ignore[attr-defined]
        {
            "task": task,
            "runtime": {},
            "retrieval_plan": {
                "filters": task.filters,
                "answer_slots": ["remote_benefit_split", "remote_filing_institution_scope"],
            },
            "adopted_policy_evidence": [evidence.model_dump(mode="json")],
        }
    )

    facts = state["verified_slot_facts"]
    field_map: dict[str, set[str]] = {}
    for fact in facts:
        field_map.setdefault(fact["slot_id"], set()).update(
            fact["value"].get("matched_required_fields", [])
        )
    assert "直接结算费用范围" in field_map["remote_benefit_split"]
    assert "门诊慢特病病种范围" in field_map["remote_benefit_split"]
    assert "统筹地区内所有定点医药机构" in field_map["remote_filing_institution_scope"]
    assert state["slot_window_judge"]["extraction_mode"] == "llm_slot_window_with_deterministic_field_backfill"


def test_policy_claim_first_material_list_deduplicates_subsumed_receipt_terms() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_material_dedupe",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询门诊零星报销必要材料",
        user_question="审核参保人零星报销门诊费用时，应要求提供哪些必要材料？",
        filters={
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:outpatient-materials",
            source_ref="policy-evidence:outpatient-materials",
            title="参保人零星报销医疗保险费用的审核、结算：门诊报销",
            excerpt=(
                "材料名称 来源渠道 材料必要性 身份证 必要 社保卡 必要 "
                "医疗费专用收据 必要 疾病诊断证明书 非必要 病史资料 必要 "
                "银行卡 必要"
            ),
            jurisdiction="shanghai",
            policy_domain="manual_reimbursement",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["必要材料"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:outpatient-materials"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖必要材料。",
        ),
        question_slots=[{"slot_id": "required_materials"}],
        filters=task.filters,
    )

    assert "医疗费专用收据" in answer.answer_markdown
    assert "、专用收据" not in answer.answer_markdown
    assert "、收据" not in answer.answer_markdown


def test_policy_claim_first_emergency_visit_does_not_use_observation_voucher_when_not_asked() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_emergency_visit_not_observation",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询急诊就医手工报销材料",
        user_question="审核北京参保人员急诊就医手工报销时，应核验哪些材料？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:emergency-observation",
            source_ref="policy-evidence:emergency-observation",
            title="急诊留观费用审核结算凭证",
            excerpt=(
                "急诊留观并收入院前7日发生的费用，由用人单位汇总填写"
                "《北京市医疗保险门急诊(药店)费用审核结算凭证》，并附"
                "收入院证明、处方底方和专用收据报区、县医保中心审核结算。"
            ),
            jurisdiction="beijing",
            policy_domain="manual_reimbursement",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="partial",
            covered_slots=[],
            missing_slots=["急诊就医手工报销材料"],
            usable_evidence_refs=[],
            next_action="synthesize",
            reason_code="partial",
            reason="未覆盖急诊就医材料。",
        ),
        question_slots=[{"slot_id": "emergency_manual_reimbursement_materials"}],
        filters=task.filters,
    )

    assert "急诊留观并收入院前7日" not in answer.answer_markdown


def test_policy_claim_first_special_disease_condition_extracts_hard_conditions() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_special_disease_condition",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询类风湿关节炎限定支付条件",
        user_question="审核参保人门诊特殊疾病费用时，若其因类风湿关节炎使用新增报销范围的药品，应满足哪些限定支付条件才能纳入医保支付？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_scope"],
            "content_type": ["policy_text"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:special-disease-scope",
            source_ref="policy-evidence:special-disease-scope",
            title="北京市新增基本医疗保险门诊特殊疾病用药报销范围",
            excerpt=(
                "限以下情况方可支付：1.诊断明确的类风湿关节炎经传统DMARDs治疗3-6个月"
                "疾病活动度下降低于50%者；诊断明确的强直性脊柱炎NSAIDs充分治疗3个月"
                "疾病活动度下降低于50%者；并需风湿病专科医师处方。"
            ),
            jurisdiction="beijing",
            policy_domain="special_disease_scope",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["特殊疾病限定支付条件"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:special-disease-scope"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖限定支付条件。",
        ),
        question_slots=[{"slot_id": "special_disease_scope"}],
        filters=task.filters,
    )

    assert "诊断明确的类风湿关节炎需经传统DMARDs治疗3-6个月" in answer.answer_markdown
    assert "疾病活动度下降低于50%" in answer.answer_markdown
    assert "风湿病专科医师处方" in answer.answer_markdown
    assert "药品支付范围应核验药品是否纳入医保药品目录" not in answer.answer_markdown


def test_policy_coverage_gate_builds_required_field_matrix_and_claim_plan() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_plan_matrix",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算支付规则",
        user_question="跨省异地就医直接结算时，医疗费用支付规则是怎样的？",
        filters={
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    requirement = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_span=task.user_question,
        scenario_id="direct_settlement",
        required_fields=[
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
        ],
        answer_action="explain_rule",
        filters=task.filters,
        fact_schema="remote_benefit_split",
        extractor_id="remote_benefit_split",
    )
    fact = ExtractedFact(
        fact_id="fact_1",
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        fact_type="remote_benefit_split_slot_window",
        value={
            "support_role": "direct_answer",
            "llm_score": 0.92,
            "quote_verified": True,
            "matched_required_fields": ["就医地支付范围"],
        },
        display_text="医保支付范围原则上按就医地规定执行。",
        evidence_refs=["policy-node:remote-split"],
        source_refs=["policy-evidence:remote-split"],
        evidence_text="执行就医地规定的支付范围及有关规定",
        confidence=0.92,
    )

    coverage = service._slot_coverage_gate(  # type: ignore[attr-defined]
        task,
        {
            "retrieval_plan": {"filters": task.filters},
            "answer_requirements": [requirement.model_dump(mode="json")],
            "verified_slot_facts": [fact.model_dump(mode="json")],
        },
        [
            {
                "evidence_ref": "policy-node:remote-split",
                "source_ref": "policy-evidence:remote-split",
                "jurisdiction": "national",
                "policy_domain": "remote_medical",
                "content_type": "policy_text",
            }
        ],
        [
            {
                "slot_id": "remote_benefit_split",
                "label": "就医地目录与参保地待遇分工",
                "critical": True,
                "weight": 1.2,
                "scenario_id": "direct_settlement",
                "required_fields": requirement.required_fields,
                "answer_action": "explain_rule",
            }
        ],
    )

    slot_status = coverage["slot_status"]["remote_benefit_split"]
    assert slot_status["status"] == "weak"
    assert slot_status["covered_fields"] == ["就医地支付范围"]
    assert "参保地起付标准" in slot_status["missing_fields"]
    assert coverage["claim_plan"][0]["generation"] == "allow_with_qualification"
    assert coverage["claim_plan"][0]["direct_fact_refs"] == ["fact_1"]


def test_policy_claim_first_uses_claim_plan_to_exclude_denied_background() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_plan_filter",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算支付规则",
        user_question="跨省异地就医直接结算时，医疗费用支付规则是怎样的？",
        filters={
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical", "remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
        },
    )
    remote_req = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_span=task.user_question,
        filters=task.filters,
        fact_schema="remote_benefit_split",
        extractor_id="remote_benefit_split",
    )
    filing_req = AnswerRequirement(
        requirement_id="req_2",
        slot_id="remote_filing",
        label="异地备案与急诊例外",
        question_span=task.user_question,
        filters=task.filters,
        fact_schema="policy_rule",
        extractor_id="policy_rule_span",
    )
    facts = [
        ExtractedFact(
            fact_id="fact_1",
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            fact_type="remote_benefit_split_slot_window",
            value={"support_role": "direct_answer", "llm_score": 0.92, "quote_verified": True},
            display_text="医保支付范围原则上按就医地规定执行。",
            evidence_refs=["policy-node:remote-split"],
            source_refs=["policy-evidence:remote-split"],
            evidence_text="执行就医地规定的支付范围及有关规定",
            confidence=0.92,
        ),
        ExtractedFact(
            fact_id="fact_2",
            requirement_id="req_2",
            slot_id="remote_filing",
            fact_type="policy_rule_slot_window",
            value={"support_role": "direct_answer", "llm_score": 0.91, "quote_verified": True},
            display_text="急诊留观费用可由单位或社保所申请手工报销。",
            evidence_refs=["policy-node:manual-emergency"],
            source_refs=["policy-evidence:manual-emergency"],
            evidence_text="由单位（社保所）向所属区医保经办机构申请手工报销",
            confidence=0.91,
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=[],
        answerability=AnswerabilityCheck(
            answerability="partial",
            covered_slots=["就医地目录与参保地待遇分工"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-split"],
            next_action="synthesize",
            reason_code="partial_slot_coverage",
            reason="仅回答本题支付规则。",
        ),
        question_slots=[],
        filters=task.filters,
        precomputed_requirements=[remote_req, filing_req],
        precomputed_facts=facts,
        precomputed_matches=[],
        coverage_result={
            "claim_plan": [
                {
                    "slot_id": "remote_benefit_split",
                    "requirement_id": "req_1",
                    "generation": "allow",
                    "fact_refs": ["fact_1"],
                    "direct_fact_refs": ["fact_1"],
                    "evidence_refs": ["policy-node:remote-split"],
                },
                {
                    "slot_id": "remote_filing",
                    "requirement_id": "req_2",
                    "generation": "deny",
                    "fact_refs": ["fact_2"],
                    "direct_fact_refs": ["fact_2"],
                    "evidence_refs": ["policy-node:manual-emergency"],
                },
            ]
        },
    )

    assert "就医地规定执行" in answer.answer_markdown
    assert "手工报销" not in answer.answer_markdown
    assert "社保所" not in answer.answer_markdown


def test_policy_claim_first_claim_plan_keeps_all_covered_required_field_facts() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_plan_multi_fact",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询跨省异地就医直接结算支付规则",
        user_question="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    requirement = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_span=task.user_question,
        required_fields=[
            "直接结算费用范围",
            "就医地支付范围",
            "参保地起付标准",
            "参保地支付比例",
            "参保地最高支付限额",
        ],
        filters=task.filters,
        fact_schema="remote_benefit_split",
        extractor_id="remote_benefit_split",
    )
    facts = [
        ExtractedFact(
            fact_id="fact_1",
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            fact_type="direct_settlement_expense_scope",
            value={
                "support_role": "direct_answer",
                "quote_verified": True,
                "matched_required_fields": ["直接结算费用范围"],
            },
            display_text="跨省异地就医直接结算覆盖住院、普通门诊和门诊慢特病医疗费用。",
            evidence_refs=["policy-node:remote-direct"],
            source_refs=["policy-evidence:remote-direct"],
            evidence_text="住院、普通门诊和门诊慢特病医疗费用",
            confidence=0.96,
        ),
        ExtractedFact(
            fact_id="fact_2",
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            fact_type="payment_scope_owner",
            value={
                "support_role": "direct_answer",
                "quote_verified": True,
                "matched_required_fields": [
                    "就医地支付范围",
                    "参保地起付标准",
                    "参保地支付比例",
                    "参保地最高支付限额",
                ],
            },
            display_text="支付范围按就医地规定执行，基金起付标准、支付比例和最高支付限额按参保地规定执行。",
            evidence_refs=["policy-node:remote-direct"],
            source_refs=["policy-evidence:remote-direct"],
            evidence_text="执行就医地规定的支付范围，执行参保地规定的起付标准、支付比例、最高支付限额",
            confidence=0.96,
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=[],
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["就医地目录与参保地待遇分工"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-direct"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖支付规则。",
        ),
        question_slots=[],
        filters=task.filters,
        precomputed_requirements=[requirement],
        precomputed_facts=facts,
        precomputed_matches=[],
        coverage_result={
            "claim_plan": [
                {
                    "slot_id": "remote_benefit_split",
                    "requirement_id": "req_1",
                    "generation": "allow",
                    "fact_refs": ["fact_1"],
                    "direct_fact_refs": ["fact_1"],
                    "covered_fields": [
                        "直接结算费用范围",
                        "就医地支付范围",
                        "参保地起付标准",
                        "参保地支付比例",
                        "参保地最高支付限额",
                    ],
                }
            ]
        },
    )

    assert "住院、普通门诊和门诊慢特病医疗费用" in answer.answer_markdown
    assert "支付范围按就医地规定执行" in answer.answer_markdown
    assert "起付标准、支付比例和最高支付限额按参保地规定执行" in answer.answer_markdown
    assert "当前可引用证据仅覆盖部分核验点" not in answer.answer_markdown


def test_policy_claim_first_partial_note_requires_unanswered_asked_slot() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_plan_no_false_partial",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询支付规则和备案后机构范围",
        user_question="北京参保人员跨省异地就医直接结算时，医疗费用支付规则是怎样的？备案成功后可在哪些机构就医？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
    )
    remote_req = AnswerRequirement(
        requirement_id="req_1",
        slot_id="remote_benefit_split",
        label="就医地目录与参保地待遇分工",
        question_span=task.user_question,
        filters=task.filters,
        fact_schema="remote_benefit_split",
        extractor_id="remote_benefit_split",
    )
    filing_req = AnswerRequirement(
        requirement_id="req_2",
        slot_id="remote_filing_institution_scope",
        label="备案后就医机构范围",
        question_span=task.user_question,
        required_fields=["备案到就医地统筹地区", "统筹地区内所有定点医药机构", "按规定就医结算"],
        filters=task.filters,
        fact_schema="remote_filing_institution_scope",
        extractor_id="policy_rule_span",
    )
    facts = [
        ExtractedFact(
            fact_id="fact_1",
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            fact_type="remote_benefit_split",
            value={"support_role": "direct_answer", "quote_verified": True},
            display_text="参保人员直接结算的住院、普通门诊和门诊慢特病医疗费用，按就医地支付范围和参保地待遇参数执行。",
            evidence_refs=["policy-node:remote-direct"],
            source_refs=["policy-evidence:remote-direct"],
            evidence_text="直接结算的住院、普通门诊和门诊慢特病医疗费用",
            confidence=0.96,
        ),
        ExtractedFact(
            fact_id="fact_2",
            requirement_id="req_2",
            slot_id="remote_filing_institution_scope",
            fact_type="remote_filing_institution_scope",
            value={
                "support_role": "direct_answer",
                "quote_verified": True,
                "matched_required_fields": ["统筹地区内所有定点医药机构", "按规定就医结算"],
            },
            display_text="本市参保人员原则上只需备案到就医地所在统筹地区，备案成功后可在该统筹地区内所有定点医药机构按规定就医结算。",
            evidence_refs=["policy-node:remote-filing"],
            source_refs=["policy-evidence:remote-filing"],
            evidence_text="备案成功后可在该统筹地区内所有定点医药机构按规定就医结算",
            confidence=0.94,
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=[],
        answerability=AnswerabilityCheck(
            answerability="partial",
            covered_slots=["就医地目录与参保地待遇分工", "备案后就医机构范围"],
            missing_slots=["备案后就医机构范围缺少备案到就医地统筹地区"],
            usable_evidence_refs=["policy-node:remote-direct", "policy-node:remote-filing"],
            next_action="synthesize",
            reason_code="partial_slot_coverage",
            reason="字段门禁保守标记为部分覆盖。",
        ),
        question_slots=[],
        filters=task.filters,
        precomputed_requirements=[remote_req, filing_req],
        precomputed_facts=facts,
        precomputed_matches=[],
        coverage_result={
            "claim_plan": [
                {
                    "slot_id": "remote_benefit_split",
                    "requirement_id": "req_1",
                    "generation": "allow",
                    "fact_refs": ["fact_1"],
                    "direct_fact_refs": ["fact_1"],
                },
                {
                    "slot_id": "remote_filing_institution_scope",
                    "requirement_id": "req_2",
                    "generation": "allow_with_qualification",
                    "fact_refs": ["fact_2"],
                    "direct_fact_refs": ["fact_2"],
                    "missing_fields": ["备案到就医地统筹地区"],
                },
            ]
        },
    )

    assert "住院、普通门诊和门诊慢特病医疗费用" in answer.answer_markdown
    assert "统筹地区内所有定点医药机构" in answer.answer_markdown
    assert "当前可引用证据仅覆盖部分核验点" not in answer.answer_markdown
    assert answer.support_status == "supported"


def test_policy_claim_first_filters_unasked_background_even_with_claim_plan() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_plan_scope_filter",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询急诊留观自费结算后补备案和手工报销要点",
        user_question="一位北京参保人员在外地发生急诊留观，当时自费结算，随后按规定补办了跨省异地就医备案，现申请医保手工报销。作为审核员，需要确认哪些政策要点？",
        filters={
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical", "remote_medical_manual_reimbursement"],
            "content_type": ["policy_text", "faq"],
        },
    )
    requirements = [
        AnswerRequirement(
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            label="就医地目录与参保地待遇分工",
            question_span=task.user_question,
            filters=task.filters,
            fact_schema="remote_benefit_split",
            extractor_id="remote_benefit_split",
        ),
        AnswerRequirement(
            requirement_id="req_2",
            slot_id="benefit_params",
            label="待遇参数",
            question_span=task.user_question,
            filters=task.filters,
            fact_schema="benefit_parameter",
            extractor_id="benefit_params",
        ),
        AnswerRequirement(
            requirement_id="req_3",
            slot_id="remote_self_pay_filing_manual_reimbursement",
            label="补办备案后手工报销路径",
            question_span=task.user_question,
            filters=task.filters,
            fact_schema="remote_self_pay_filing_manual_reimbursement",
            extractor_id="remote_self_pay_filing_manual_reimbursement",
        ),
        AnswerRequirement(
            requirement_id="req_4",
            slot_id="remote_emergency_observation_reimbursement",
            label="异地急诊留观报销路径",
            question_span=task.user_question,
            filters=task.filters,
            fact_schema="remote_emergency_observation_reimbursement",
            extractor_id="remote_emergency_observation_reimbursement",
        ),
    ]
    facts = [
        ExtractedFact(
            fact_id="fact_1",
            requirement_id="req_1",
            slot_id="remote_benefit_split",
            fact_type="remote_benefit_split",
            value={"support_role": "direct_answer", "quote_verified": True},
            display_text="参保人员直接结算的住院、普通门诊和门诊慢特病医疗费用，原则上执行就医地支付范围和参保地待遇参数。",
            evidence_refs=["policy-node:remote-direct"],
            source_refs=["policy-evidence:remote-direct"],
            evidence_text="直接结算的住院、普通门诊和门诊慢特病医疗费用",
            confidence=0.96,
        ),
        ExtractedFact(
            fact_id="fact_2",
            requirement_id="req_2",
            slot_id="benefit_params",
            fact_type="benefit_parameter",
            value={"support_role": "direct_answer", "quote_verified": True},
            display_text="异地转诊人员和异地急诊抢救人员支付比例的降幅不超过10个百分点。",
            evidence_refs=["policy-node:ratio"],
            source_refs=["policy-evidence:ratio"],
            evidence_text="支付比例的降幅不超过10个百分点",
            confidence=0.88,
        ),
        ExtractedFact(
            fact_id="fact_3",
            requirement_id="req_3",
            slot_id="remote_self_pay_filing_manual_reimbursement",
            fact_type="remote_self_pay_filing_manual_reimbursement",
            value={"support_role": "direct_answer", "quote_verified": True},
            display_text="异地就医出院自费结算后，可按参保地规定补办备案手续，再申请医保手工报销。",
            evidence_refs=["policy-node:self-pay"],
            source_refs=["policy-evidence:self-pay"],
            evidence_text="出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销",
            confidence=0.92,
        ),
        ExtractedFact(
            fact_id="fact_4",
            requirement_id="req_4",
            slot_id="remote_emergency_observation_reimbursement",
            fact_type="remote_emergency_observation_reimbursement",
            value={"support_role": "direct_answer", "quote_verified": True},
            display_text="异地急诊留观费用暂不能实现异地直接结算，参保人员可将异地就医票据及相关报销材料交给本人所属单位或社保所，由单位或社保所向所属区医保经办机构申请手工报销，并按住院标准报销。",
            evidence_refs=["policy-node:emergency-observation"],
            source_refs=["policy-evidence:emergency-observation"],
            evidence_text="急诊留观费用暂不能实现异地直接结算",
            confidence=0.92,
        ),
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=[],
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["补办备案后手工报销路径", "异地急诊留观报销路径"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:self-pay", "policy-node:emergency-observation"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖问题槽位。",
        ),
        question_slots=[],
        filters=task.filters,
        precomputed_requirements=requirements,
        precomputed_facts=facts,
        precomputed_matches=[],
        coverage_result={
            "claim_plan": [
                {
                    "slot_id": requirement.slot_id,
                    "requirement_id": requirement.requirement_id,
                    "generation": "allow",
                    "fact_refs": [f"fact_{index}"],
                    "direct_fact_refs": [f"fact_{index}"],
                }
                for index, requirement in enumerate(requirements, start=1)
            ]
        },
    )

    assert "出院自费结算后" in answer.answer_markdown
    assert "急诊留观费用暂不能实现异地直接结算" in answer.answer_markdown
    assert "住院、普通门诊和门诊慢特病" not in answer.answer_markdown
    assert "降幅不超过10个百分点" not in answer.answer_markdown
    assert "当前可引用证据仅覆盖部分核验点" not in answer.answer_markdown


def test_policy_claim_first_remote_emergency_observation_extracts_path_claim() -> None:
    task = ExpertAnalysisTask(
        task_id="l3_policy_claim_remote_emergency_observation",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询异地急诊留观报销路径",
        user_question="异地急诊留观费用能不能直接结算？应该通过什么路径报销、按什么标准？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["policy_text", "faq"],
        },
    )
    evidence = [
        PolicyEvidence(
            evidence_ref="policy-node:remote-emergency-observation",
            source_ref="policy-evidence:remote-emergency-observation",
            title="异地急诊留观手工报销问答",
            excerpt=(
                "按照国家统一要求，急诊留观费用暂不能实现异地直接结算。"
                "参保人员可将异地就医票据及相关报销材料交给本人所属单位（社保所），"
                "由单位（社保所）向所属区医保经办机构申请手工报销。"
                "符合规定的急诊留观费用按住院标准报销。"
            ),
            jurisdiction="beijing",
            policy_domain="remote_medical_manual_reimbursement",
            content_type="policy_text",
        )
    ]

    answer = build_claim_first_policy_answer(
        task=task,
        evidence=evidence,
        answerability=AnswerabilityCheck(
            answerability="answerable",
            covered_slots=["异地急诊留观报销路径"],
            missing_slots=[],
            usable_evidence_refs=["policy-node:remote-emergency-observation"],
            next_action="synthesize",
            reason_code="covered",
            reason="已覆盖异地急诊留观报销路径。",
        ),
        question_slots=[{"slot_id": "remote_emergency_observation_reimbursement"}],
        filters=task.filters,
    )

    assert "暂不能实现异地直接结算" in answer.answer_markdown
    assert "本人所属单位或社保所" in answer.answer_markdown
    assert "按住院标准报销" in answer.answer_markdown
    assert "急诊抢救视同备案" not in answer.answer_markdown


def test_expert_information_need_keeps_emergency_observation_material_wording() -> None:
    question = "异地急诊留观费用申请手工报销时，现有政策要求核验哪些材料？"

    information_needs = _infer_information_needs_from_question(question)

    assert information_needs == [question.rstrip("？")]
    assert "required_materials" not in information_needs
    assert "remote_emergency_observation_reimbursement" not in information_needs


def test_expert_question_slots_accept_llm_planned_answer_slots() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)
    task = ExpertAnalysisTask(
        task_id="l3_policy_slots",
        parent_run_id="crun_policy_claim",
        case_id="CASE-001",
        goal="查询上海支付范围核验",
        user_question="上海药品、胸部 CT 和医用耗材支付范围如何核验？",
        filters={
            "jurisdiction": ["shanghai"],
            "policy_domain": ["drug_catalog"],
            "content_type": ["table_row"],
        },
    )

    slots = service._build_question_slots(
        task,
        {
            "retrieval_plan": {
                "filters": task.filters,
                "answer_slots": [
                    "drug_catalog",
                    "medical_service_price",
                    "consumable_payment_scope",
                ],
            }
        },
    )
    slot_ids = [slot["slot_id"] for slot in slots]

    assert "drug_catalog" in slot_ids
    assert "medical_service_price" in slot_ids
    assert "consumable_payment_scope" in slot_ids


def test_expert_analysis_rejects_non_policy_non_drug_reference_evidence() -> None:
    service = ExpertAgentService.__new__(ExpertAgentService)

    evidence = service._policy_evidence_from_candidate(
        {
            "node_id": "node_reference_other",
            "source_id": "reference_other",
            "title": "非政策参考资料",
            "text": "这不是可作为政策依据的资料。",
            "jurisdiction": "shanghai",
            "policy_domain": "policy_interpretation",
            "content_type": "reference",
            "can_cite_as_policy_basis": False,
        },
        1,
    )

    assert evidence is None


def test_case_agent_build_answer_context_keeps_material_sources_and_full_drug_rows() -> None:
    case_id = "CASE-001"
    overview = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_material_overview",
            "section_key": "material_overview",
            "payload": {
                "material_total": 1,
                "groups": [
                    {
                        "title": "处方购药材料",
                        "count": 1,
                        "materials": [
                            {
                                "material_id": "BM-03",
                                "name": "处方购药记录",
                                "occurred_at": "2026-01-01",
                                "source": "院内药房",
                                "shape": "表格 + 图片",
                                "status": "已整理",
                            }
                        ],
                    }
                ],
            },
        },
        source_refs=[f"material:{case_id}-BM-03"],
    )
    prescriptions = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_prescription_materials",
            "section_key": "prescription_materials",
            "payload": {
                "materials": [
                    {
                        "material_id": "BM-03",
                        "name": "处方购药记录",
                        "prescription_drug_details": [
                            {"药品名称": "恩格列净片", "数量": "18盒", "金额": "4709.66"},
                            {"药品名称": "阿托伐他汀钙片", "数量": "10盒", "金额": "520.00"},
                        ],
                    }
                ]
            },
        },
        source_refs=[f"material:{case_id}-BM-03"],
    )
    state = {
        "run": SimpleNamespace(run_id="crun_context"),
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "材料来源分别是什么，并输出所有处方药品明细",
            "granularity": "list",
            "missing_slots": [],
        },
        "user_message": SimpleNamespace(content="材料来源分别是什么，并输出所有处方药品明细"),
        "capability_results": [
            ("query_material_overview", overview),
            ("query_prescription_materials", prescriptions),
        ],
        "model_call_count": 0,
        "tool_call_count": 2,
    }
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        )
    )

    next_state = build_answer_context_node(service, state)

    assert next_state["answer_context"]["lists"][0]["items"][0]["材料来源"] == "院内药房"
    details = next_state["answer_context"]["details"][0]
    assert len(details["prescription_drug_details"]) == 2
    assert details["prescription_drug_details"][0]["药品名称"] == "恩格列净片"


def test_case_agent_resolve_answer_style_single_field_is_direct_fact() -> None:
    service = SimpleNamespace(
        _repository=SimpleNamespace(
            update_run=lambda *args, **kwargs: None,
            append_event=lambda *args, **kwargs: None,
        )
    )
    state = {
        "run": SimpleNamespace(run_id="crun_style"),
        "query_semantics": {"granularity": "single_field"},
        "answer_policy": {"display_mode": "grounded"},
        "model_call_count": 0,
        "tool_call_count": 1,
    }

    next_state = resolve_answer_style_node(service, state)

    assert next_state["answer_style_policy"]["answer_tone"] == "direct_fact"
    assert next_state["answer_style_policy"]["max_blocks"] == 1
    assert next_state["answer_style_policy"]["forbid_extra_fields"] is True
    assert next_state["next_action"] == "generate_answer"


def test_case_agent_new_tools_are_case_bound_and_degrade_safely() -> None:
    case_id = "CASE-CASEAGENT-001"
    tools = _tool_registry(case_id)
    basic = tools.execute(
        current_case_id=case_id,
        tool_name="query_case_basic_info",
        raw_arguments={"case_id": case_id},
    )

    basic_json = json.dumps(basic.payload, ensure_ascii=False)
    assert basic.status == "success"
    assert basic.payload["section_key"] == "case_basic_info"
    assert basic.payload["payload"]["case_number"] == case_id
    assert f"case:{case_id}:basic" in basic.source_refs
    assert "risk_score" not in basic.payload
    assert "case_judgement" not in basic.payload
    assert "RES" not in basic_json

    risk = tools.execute(
        current_case_id=case_id,
        tool_name="query_risk_score",
        raw_arguments={"case_id": case_id},
    )
    risk_json = json.dumps(risk.payload, ensure_ascii=False)
    assert risk.status == "success"
    assert f"risk:{case_id}:model_warning" in risk.source_refs
    assert "fraud" not in risk_json.lower()

    cross_case = tools.execute(
        current_case_id="CASE-OTHER",
        tool_name="query_case_basic_info",
        raw_arguments={"case_id": case_id},
    )
    assert cross_case.status == "failed"
    assert cross_case.error_code == "cross_case_access"

    unavailable = tools.execute(
        current_case_id=case_id,
        tool_name="ask_policy_expert",
        raw_arguments={
            "case_id": case_id,
            "question": "本地政策口径",
        },
    )
    assert unavailable.status == "unavailable"
    assert unavailable.payload["status"] == "unavailable"
    assert unavailable.payload["capability"] == "ask_policy_expert"


def test_case_agent_case_basic_info_exposes_case_context_fields() -> None:
    case_id = "CASE-CASEAGENT-CONTEXT"
    tools = _tool_registry(
        case_id,
        case_context={
            "insured_region": "北京",
            "treatment_region": "上海",
            "visit_type": "异地门诊急诊",
            "claim_mode": "manual_reimbursement",
            "direct_settlement": False,
            "filing_status": "unknown",
            "emergency_material_status": "present_but_needs_verification",
            "diagnosis": "急性上呼吸道感染",
        },
    )

    result = tools.execute(
        current_case_id=case_id,
        tool_name="query_case_basic_info",
        raw_arguments={"case_id": case_id},
    )

    assert result.status == "success"
    payload = result.payload["payload"]
    assert payload["insured_region"] == "北京"
    assert payload["treatment_region"] == "上海"
    assert payload["claim_mode"] == "manual_reimbursement"
    assert payload["filing_status"] == "unknown"
    assert payload["emergency_material_status"] == "present_but_needs_verification"

    class ContextRepository:
        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_update = {"run_id": run_id, **kwargs}

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events = getattr(self, "events", [])
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    state = {
        "run": SimpleNamespace(run_id="crun_context_case_basic"),
        "query_semantics": {
            "intent": "case_task",
            "user_goal": "在哪里参保的",
            "granularity": "single_field",
            "missing_slots": [],
        },
        "user_message": SimpleNamespace(content="在哪里参保的"),
        "capability_results": [("query_case_basic_info", result)],
        "model_call_count": 0,
        "tool_call_count": 1,
    }
    next_state = build_answer_context_node(SimpleNamespace(_repository=ContextRepository()), state)
    assert next_state["answer_context"]["facts"] == [
        {
            "section": "案件基础信息",
            "label": "参保地",
            "value": "北京",
            "source_refs": [f"case:{case_id}:basic"],
        }
    ]


def test_case_agent_l1_source_detail_includes_metadata_and_data_snapshot() -> None:
    case_id = "CASE-CASEAGENT-CONTEXT"
    tools = _tool_registry(
        case_id,
        case_context={
            "insured_region": "北京",
            "treatment_region": "上海",
            "filing_status": "filed",
        },
    )

    result = tools.execute(
        current_case_id=case_id,
        tool_name="query_case_basic_info",
        raw_arguments={"case_id": case_id, "fields": ["case_number", "case_context"]},
    )

    assert result.status == "success"
    assert result.sources
    labels = {field.label: field.value for field in result.sources[0].detail.fields}
    assert labels["业务分区"] == "案件基础信息"
    assert "case_basic_info" in labels["原始元数据"]
    assert "insured_region" in labels["数据快照"]
    assert "case_context" in result.sources[0].metadata["payload_preview"]["data"]


def test_expert_agent_policy_analysis_uses_mcp_and_returns_structured_result() -> None:
    class FakePolicyRagClient:
        def __init__(self) -> None:
            self.calls: list[PolicySearchRequest] = []

        def search(self, request: PolicySearchRequest) -> dict[str, object]:
            self.calls.append(request)
            return {
                "status": "ok",
                "result_count": 1,
                "evidence": [
                    {
                        "rank": 1,
                        "score": 0.91,
                        "score_type": "rerank_score",
                        "node_id": "policy-node-1",
                        "source_id": "BJ-SH-REMOTE-MANUAL",
                        "title": "异地就医手工报销政策摘录",
                        "text": "异地急诊手工报销应结合参保地、就医地、备案或急诊材料等政策条件核验。",
                        "jurisdiction": "beijing",
                        "policy_domain": "remote_medical",
                        "content_type": "policy_text",
                        "can_cite_as_policy_basis": True,
                        "source_url": "https://example.test/policy",
                    }
                ],
                "warnings": [],
                "limits": ["retrieval only"],
            }

    class RecordingRepository:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def append_event(self, run_id: str, event_type: str, message: str, payload=None) -> None:
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "policy_question": "异地急诊手工报销政策条件",
                        "filters": {
                            "jurisdiction": ["beijing"],
                            "policy_domain": ["remote_medical"],
                            "content_type": ["policy_text"],
                            "can_cite_as_policy_basis": True,
                        },
                        "top_k": 3,
                        "fetch_k": 12,
                        "rerank": True,
                        "need_case_context": False,
                        "case_context_requests": [],
                        "reason": "policy question",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "answerability": "answerable",
                        "covered_slots": ["policy_basis"],
                        "missing_slots": [],
                        "usable_evidence_refs": [],
                        "next_action": "synthesize",
                        "reason_code": "covered",
                        "reason": "证据可回答。",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "expert_answer": "政策口径上，应围绕参保地、就医地、备案或急诊材料等条件核验异地急诊手工报销适用性。",
                        "audit_suggestions": [
                            {
                                "type": "policy_based_verification",
                                "text": "按政策证据核验参保地、就医地和急诊材料条件。",
                                "source_refs": [],
                            }
                        ],
                        "material_gaps": [],
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    client = FakePolicyRagClient()
    repository = RecordingRepository()
    service = ExpertAgentService()

    result = service.run(
        expert_task_type="policy_analysis",
        task={
            "task_id": "l3_policy_test",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询异地急诊手工报销政策口径",
            "user_question": "北京参保人在上海异地门诊急诊手工报销需要核验哪些政策条件？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        },
        model_gateway=gateway,
        policy_rag_client=client,
        repository=repository,
    )

    assert result.status == "partial"
    assert result.payload["answerability_summary"]["answerability"] == "partial"
    assert client.calls
    assert "beijing" in client.calls[0].filters["jurisdiction"]
    assert "remote_medical" in client.calls[0].filters["policy_domain"]
    assert result.source_refs
    assert result.payload["material_gaps"] == []
    assert result.payload["policy_evidence"][0]["source_ref"] in result.source_refs
    assert len(gateway.requests) == 1
    event_types = [str(event["event_type"]) for event in repository.events]
    assert event_types.index("expert_policy_rag_calling") < event_types.index(
        "expert_policy_rag_called"
    )
    assert event_types.index("expert_policy_rag_called") < event_types.index(
        "expert_evidence_normalized"
    )
    assert event_types.index("expert_evidence_normalized") < event_types.index(
        "expert_need_evidence_resolved"
    )
    assert event_types.index("expert_need_evidence_resolved") < event_types.index(
        "expert_need_coverage_checked"
    )
    assert event_types.index("expert_need_coverage_checked") < event_types.index(
        "expert_analysis_completed"
    )


def test_expert_agent_answerability_accepts_remote_benefit_split_evidence() -> None:
    policy_clause = (
        "跨省异地就医直接结算的住院、普通门诊和门诊慢特病医疗费用，"
        "原则上执行就医地规定的支付范围及有关规定（基本医疗保险药品、"
        "医疗服务项目和医用耗材等支付范围），执行参保地规定的基本医疗保险"
        "基金起付标准、支付比例、最高支付限额。"
    )

    class FakePolicyRagClient:
        def search(self, request: PolicySearchRequest) -> dict[str, object]:
            return {
                "status": "ok",
                "result_count": 1,
                "evidence": [
                    {
                        "rank": 1,
                        "score": 0.94,
                        "score_type": "dense_score",
                        "node_id": "national-remote-split",
                        "source_id": "policy_national_remote_settlement_20220726",
                        "title": "跨省异地就医直接结算政策",
                        "text": policy_clause,
                        "jurisdiction": "national",
                        "policy_domain": "remote_medical",
                        "content_type": "policy_text",
                        "can_cite_as_policy_basis": True,
                        "source_url": "https://example.test/national-remote",
                    }
                ],
                "warnings": [],
            }

    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "policy_question": "跨省异地就医就医地目录和参保地待遇分工规则",
                        "filters": {
                            "jurisdiction": ["national"],
                            "policy_domain": ["remote_medical"],
                            "content_type": ["policy_text"],
                        },
                        "top_k": 3,
                        "fetch_k": 12,
                        "rerank": False,
                        "need_case_context": False,
                        "case_context_requests": [],
                        "reason": "policy split query",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "answerability": "insufficient",
                        "covered_slots": [],
                        "missing_slots": ["就医地目录", "参保地待遇分工规则"],
                        "usable_evidence_refs": [],
                        "next_action": "insufficient",
                        "reason_code": "missing_key_rule",
                        "reason": "误判为缺少关键规则。",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "expert_answer": (
                            "政策口径上，跨省异地就医直接结算费用原则上执行就医地"
                            "规定的支付范围，待遇支付标准执行参保地规定的起付标准、"
                            "支付比例和最高支付限额。"
                        ),
                        "audit_suggestions": [],
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )

    result = ExpertAgentService().run(
        expert_task_type="policy_analysis",
        task={
            "task_id": "l3_policy_remote_split",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询异地就医就医地目录和参保地待遇分工规则",
            "user_question": "跨省异地就医中，就医地目录与参保地待遇如何分工？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        },
        model_gateway=gateway,
        policy_rag_client=FakePolicyRagClient(),
    )

    assert result.status == "ok"
    assert result.source_refs
    assert result.payload["policy_evidence"]
    assert result.payload["answerability_summary"]["reason_code"] in {
        "all_information_needs_supported",
    }
    assert len(gateway.requests) == 1


def test_expert_analysis_coverage_gate_keeps_partial_evidence_when_llm_rejects() -> None:
    class FakePolicyRagClient:
        def search(self, request: PolicySearchRequest) -> dict[str, object]:
            return {
                "status": "ok",
                "result_count": 1,
                "evidence": [
                    {
                        "rank": 1,
                        "score": 0.88,
                        "score_type": "dense_score",
                        "node_id": "national-remote-filing",
                        "source_id": "policy_national_remote_filing",
                        "title": "异地就医备案政策",
                        "text": "异地就医直接结算需结合参保地、就医地和备案状态进行政策核验。",
                        "jurisdiction": "national",
                        "policy_domain": "remote_medical",
                        "content_type": "policy_text",
                        "can_cite_as_policy_basis": True,
                    }
                ],
                "warnings": [],
            }

    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "policy_question": "普通异地门诊手工报销需要哪些材料？",
                        "filters": {
                            "jurisdiction": ["national"],
                            "policy_domain": ["remote_medical", "manual_reimbursement"],
                            "content_type": ["policy_text"],
                        },
                        "top_k": 5,
                        "fetch_k": 40,
                        "rerank": False,
                        "need_case_context": False,
                        "case_context_requests": [],
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "answerability": "insufficient",
                        "covered_slots": [],
                        "missing_slots": ["手工报销材料明细"],
                        "usable_evidence_refs": [],
                        "next_action": "insufficient",
                        "reason_code": "llm_over_strict",
                        "reason": "误判为完全不可回答。",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "expert_answer": "材料包括身份证、发票、处方等。",
                        "audit_suggestions": [],
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )

    result = ExpertAgentService().run(
        expert_task_type="policy_analysis",
        task={
            "task_id": "l3_policy_partial_coverage",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询普通异地门诊手工报销材料",
            "user_question": "普通异地门诊手工报销需要哪些材料？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        },
        model_gateway=gateway,
        policy_rag_client=FakePolicyRagClient(),
    )

    assert result.status == "insufficient"
    assert result.payload["policy_evidence"]
    assert result.payload["answerability_summary"]["answerability"] == "insufficient"
    assert result.payload["coverage_result"]["supported_need_count"] == 0
    assert all(
        value == "deny" for value in result.payload["generation_policy"].values()
    )
    assert result.payload["pipeline_version"] == "information_need_v2"
    assert result.payload["need_answers"]
    assert all(
        item["status"] == "missing" and not item["source_refs"]
        for item in result.payload["need_answers"]
    )
    assert "当前可引用证据不足以回答该信息点" in result.message
    assert "身份证、发票、处方" not in result.message


def test_expert_analysis_coverage_gate_rejects_wrong_jurisdiction() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask(
        task_id="l3_policy_wrong_jurisdiction",
        parent_run_id="crun_parent",
        case_id="CASE-CASEAGENT-001",
        goal="查询北京门诊慢特病备案和病种范围核验口径",
        user_question="北京门诊慢特病备案和病种范围如何核验？",
        budget={"max_llm_rewrite_calls": 0},
    )
    state = {
        "task": task,
        "runtime": {},
        "retrieval_plan": {
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["special_disease_filing", "special_disease_scope"],
                "content_type": ["policy_text"],
            }
        },
        "adopted_policy_evidence": [
            {
                "evidence_ref": "policy-evidence:shanghai-special-disease",
                "source_ref": "policy-evidence:shanghai-special-disease",
                "title": "上海门诊慢特病备案政策",
                "excerpt": "门诊慢特病备案和病种范围按本市规定核验。",
                "jurisdiction": "shanghai",
                "policy_domain": "special_disease_filing",
                "content_type": "policy_text",
            }
        ],
        "rewrite_count": 0,
        "retrieval_round_count": 0,
    }

    next_state = service._answerability_check_node(state)

    check = next_state["answerability_check"]
    assert check.answerability == "insufficient"
    assert check.reason_code == "wrong_jurisdiction"
    assert next_state["coverage_result"]["wrong_jurisdiction"] is True
    assert next_state["next_action"] == "expert_synthesis"


def test_expert_analysis_remote_benefit_split_filters_follow_goldset_style() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask.model_validate(
        {
            "task_id": "l3_policy_remote_split_filters",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询异地就医就医地目录和参保地待遇分工规则",
            "user_question": "跨省异地就医中，就医地目录与参保地待遇如何分工？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        }
    )

    plan = service._normalize_retrieval_plan(  # type: ignore[attr-defined]
        task,
        {
            "policy_question": "跨省异地就医就医地目录和参保地待遇分工规则",
            "filters": {
                "jurisdiction": ["national"],
                "policy_domain": ["remote_medical", "drug_catalog"],
                "content_type": ["policy_text", "table_row"],
            },
            "top_k": 5,
            "fetch_k": 40,
            "rerank": False,
        },
    )

    assert plan["filters"]["jurisdiction"] == ["national"]
    assert plan["filters"]["policy_domain"] == ["remote_medical", "drug_catalog"]
    assert plan["filters"]["content_type"] == ["policy_text", "table_row"]
    assert plan["filters"]["can_cite_as_policy_basis"] is True
    assert plan["filter_source"] == "llm_registered"
    assert plan["filter_confidence"] == 0.8
    assert plan["filter_warnings"] == []


def test_policy_retrieval_budget_adapts_to_query_complexity() -> None:
    simple = _adaptive_policy_retrieval_budget(
        question="北京阿莫西林属于哪类医保药品？",
        filters={"jurisdiction": ["beijing"], "policy_domain": ["drug_catalog"]},
        answer_mode="fact_lookup",
        information_needs=["医保类别"],
    )
    medium = _adaptive_policy_retrieval_budget(
        question="北京门诊慢特病备案和病种范围如何核验？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_filing", "special_disease_scope"],
        },
        answer_mode="process_rule",
        information_needs=["备案流程", "病种范围"],
    )
    complex_budget = _adaptive_policy_retrieval_budget(
        question="跨省异地就医中，就医地目录与参保地待遇如何分工？",
        filters={
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical", "drug_catalog"],
        },
        answer_mode="comparison",
        information_needs=["就医地目录", "参保地待遇"],
    )

    assert simple == {
        "complexity": "simple",
        "fetch_k": 20,
        "rerank": False,
        "rerank_candidate_limit": 0,
    }
    assert medium == {
        "complexity": "medium",
        "fetch_k": 24,
        "rerank": True,
        "rerank_candidate_limit": 12,
    }
    assert complex_budget == {
        "complexity": "complex",
        "fetch_k": 40,
        "rerank": True,
        "rerank_candidate_limit": 16,
    }


def test_expert_analysis_normalizes_chinese_content_type_and_remote_flow_filters() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask.model_validate(
        {
            "task_id": "l3_policy_remote_emergency_manual_flow",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询北京参保人异地急诊备案和手工报销处理流程",
            "user_question": "北京参保人异地急诊、备案和手工报销如何处理？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        }
    )

    plan = service._normalize_retrieval_plan(  # type: ignore[attr-defined]
        task,
        {
            "policy_question": "北京参保人异地急诊、备案和手工报销如何处理？",
            "filters": {
                "jurisdiction": ["national", "beijing"],
                "policy_domain": [
                    "remote_medical",
                    "remote_medical_manual_reimbursement",
                    "manual_reimbursement",
                    "emergency",
                ],
                "content_type": ["policy_text", "table_row"],
            },
            "top_k": 10,
            "fetch_k": 40,
            "rerank": False,
        },
    )

    assert set(plan["filters"]["jurisdiction"]) == {"national", "beijing"}
    assert plan["filters"]["policy_domain"] == [
        "remote_medical",
        "remote_medical_manual_reimbursement",
        "manual_reimbursement",
        "emergency",
    ]
    assert plan["filters"]["content_type"] == ["policy_text", "table_row"]
    assert plan["filters"]["can_cite_as_policy_basis"] is True
    assert plan["top_k"] == 5
    assert plan["adaptive_top_k"] is True
    assert plan["filter_strategy"] == "dual_rrf"
    assert plan["rerank"] is True
    assert plan["recall_filters"] == {
        "jurisdiction": ["beijing", "national"],
        "policy_domain": [],
        "content_type": ["policy_text", "table_row"],
        "can_cite_as_policy_basis": True,
    }
    assert plan["policy_question"].startswith("北京参保人异地急诊")
    assert plan["filter_source"] == "llm_registered"
    assert "top_k_replaced_by_adaptive_context_policy" in plan["filter_warnings"]


def test_expert_analysis_shanghai_payment_scope_filters_cover_compound_question() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask.model_validate(
        {
            "task_id": "l3_policy_shanghai_payment_scope",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询上海药品、胸部 CT 和医用耗材支付范围核验口径",
            "user_question": "上海药品、胸部 CT 和医用耗材支付范围如何核验？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        }
    )

    plan = service._normalize_retrieval_plan(  # type: ignore[attr-defined]
        task,
        {
            "policy_question": "上海药品、胸部 CT 和医用耗材支付范围如何核验？",
            "filters": {
                "jurisdiction": ["shanghai"],
                "policy_domain": [
                    "drug_catalog",
                    "medical_service_price",
                    "shanghai_payment_scope",
                ],
                "content_type": ["table_row", "policy_text"],
            },
            "top_k": 10,
            "fetch_k": 5,
            "rerank": False,
        },
    )

    assert plan["filters"]["jurisdiction"] == ["shanghai"]
    assert plan["filters"]["policy_domain"] == [
        "drug_catalog",
        "medical_service_price",
        "shanghai_payment_scope",
    ]
    assert plan["filters"]["content_type"] == ["table_row", "policy_text"]
    assert plan["filters"]["can_cite_as_policy_basis"] is True
    assert plan["top_k"] == 5
    assert plan["fetch_k"] == 24
    assert plan["rerank"] is True
    assert plan["retrieval_complexity"] == "medium"
    assert plan["rerank_candidate_limit"] == 12
    assert plan["filter_source"] == "llm_registered"
    assert "top_k_replaced_by_adaptive_context_policy" in plan["filter_warnings"]
    assert "fetch_k_replaced_by_adaptive_policy" in plan["filter_warnings"]


def test_expert_analysis_normalizes_policy_content_type_aliases() -> None:
    service = ExpertAgentService()

    filters = service._normalize_policy_filters(  # type: ignore[attr-defined]
        {
            "jurisdiction": ["全国"],
            "policy_domain": ["异地就医"],
            "content_type": [
                "政策文件、办事指南",
                "['policy_text', 'table_row']",
                "通知",
                "规范性文件",
                "drug_catalog",
                "医保药品目录",
            ],
        }
    )

    assert filters["jurisdiction"] == ["national"]
    assert filters["policy_domain"] == ["remote_medical"]
    assert filters["content_type"] == ["policy_text", "table_row"]
    assert filters["can_cite_as_policy_basis"] is True


def test_expert_analysis_adopts_at_most_top_k_policy_evidence() -> None:
    class FakePolicyRagClient:
        def __init__(self) -> None:
            self.calls: list[PolicySearchRequest] = []

        def search(self, request: PolicySearchRequest) -> dict[str, object]:
            self.calls.append(request)
            evidence = [
                {
                    "rank": index,
                    "score": 1.0 - index / 100,
                    "score_type": "hybrid_score",
                    "node_id": f"policy-node-{index}",
                    "source_id": "policy_national_remote_settlement_20220726",
                    "title": "跨省异地就医直接结算政策",
                    "text": (
                        "跨省异地就医直接结算医疗费用原则上执行就医地"
                        "规定的支付范围，执行参保地规定的起付标准、支付比例、"
                        "最高支付限额。"
                    ),
                    "jurisdiction": "national",
                    "policy_domain": "remote_medical",
                    "content_type": "policy_text",
                    "can_cite_as_policy_basis": True,
                    "source_url": "https://example.test/national-remote",
                }
                for index in range(1, 7)
            ]
            return {
                "status": "ok",
                "result_count": len(evidence),
                "evidence": evidence,
                "warnings": [],
            }

    gateway = ScriptedModelGateway(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "policy_question": "跨省异地就医就医地目录和参保地待遇分工规则",
                        "filters": {
                            "jurisdiction": ["national"],
                            "policy_domain": ["remote_medical"],
                            "content_type": ["policy_text"],
                        },
                        "top_k": 3,
                        "fetch_k": 40,
                        "rerank": False,
                        "need_case_context": False,
                        "case_context_requests": [],
                        "reason": "goldset-style remote split query",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "answerability": "answerable",
                        "covered_slots": ["就医地支付范围", "参保地待遇政策"],
                        "missing_slots": [],
                        "usable_evidence_refs": [],
                        "next_action": "synthesize",
                        "reason_code": "covered",
                        "reason": "证据可回答。",
                    },
                    ensure_ascii=False,
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "expert_answer": "跨省异地就医按就医地支付范围、参保地待遇政策分工。",
                        "audit_suggestions": [],
                        "material_gaps": [],
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    client = FakePolicyRagClient()

    result = ExpertAgentService().run(
        expert_task_type="policy_analysis",
        task={
            "task_id": "l3_policy_top_k",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询异地就医就医地目录和参保地待遇分工规则",
            "user_question": "跨省异地就医中，就医地目录与参保地待遇如何分工？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        },
        model_gateway=gateway,
        policy_rag_client=client,
    )

    assert client.calls[0].top_k == 5
    assert result.status == "ok"
    assert result.source_refs
    assert set(result.source_refs).issubset(
        {item["source_ref"] for item in result.payload["policy_evidence"]}
    )
    assert len(result.payload["policy_evidence"]) == 5


def test_expert_analysis_normalizes_policy_domain_aliases() -> None:
    service = ExpertAgentService()

    filters = service._normalize_policy_filters(  # type: ignore[attr-defined]
        {
            "jurisdiction": ["上海", "CN", "未知地区"],
            "policy_domain": [
                "医用耗材",
                "支付范围",
                "医保支付",
                "医疗保险",
                "门诊报销",
                "医保目录",
                "outpatient",
                "emergency_non_observation",
                "unknown_policy_domain",
            ],
            "content_type": ["政策文件、办事指南"],
        }
    )

    assert filters["jurisdiction"] == ["shanghai", "national"]
    assert filters["policy_domain"] == [
        "shanghai_payment_scope",
        "manual_reimbursement",
        "drug_catalog",
        "emergency",
    ]
    assert filters["content_type"] == ["policy_text"]
    assert filters["can_cite_as_policy_basis"] is True


def test_expert_analysis_compound_payment_scope_uses_partial_covered_evidence() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask.model_validate(
        {
            "task_id": "l3_policy_shanghai_payment_scope_partial",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询上海药品、胸部 CT 和医用耗材支付范围核验口径",
            "user_question": "上海药品、胸部 CT 和医用耗材支付范围如何核验？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        }
    )
    evidence_ref = "policy-evidence:consumable"
    state = {
        "task": task,
        "runtime": {},
        "model_call_count": 0,
        "retrieval_round_count": 1,
        "rewrite_count": 0,
        "retrieval_plan": {
            "filters": {
                "jurisdiction": ["shanghai"],
                "policy_domain": [
                    "drug_catalog",
                    "medical_service_price",
                    "shanghai_payment_scope",
                ],
                "content_type": ["table_row", "policy_text"],
                "can_cite_as_policy_basis": True,
            },
        },
        "adopted_policy_evidence": [
            {
                "evidence_ref": evidence_ref,
                "source_ref": evidence_ref,
                "title": "关于部分医用耗材纳入本市基本医疗保险支付范围并完善支付办法有关事项的通知",
                "excerpt": "将腹壁修复材料等30个医用耗材条目新纳入本市基本医疗保险支付范围，乙类耗材由参保人员先自负20%，其余费用再按规定支付。",
                "jurisdiction": "shanghai",
                "policy_domain": "shanghai_payment_scope",
                "content_type": "policy_text",
            }
        ],
    }

    next_state = service._answerability_check_node(state)  # type: ignore[attr-defined]

    check = next_state["answerability_check"]
    assert check.answerability == "partial"
    assert check.next_action == "synthesize"
    assert check.usable_evidence_refs == [evidence_ref]
    assert any("医用耗材" in item for item in check.covered_needs)
    assert any("药品" in item for item in check.missing_needs)
    assert any("CT" in item for item in check.missing_needs)
    assert next_state["next_action"] == "expert_synthesis"


def test_expert_analysis_special_disease_filters_do_not_become_remote_medical() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask.model_validate(
        {
            "task_id": "l3_policy_special_disease",
            "parent_run_id": "crun_parent",
            "case_id": "CASE-CASEAGENT-001",
            "goal": "查询北京门诊慢特病备案和病种范围核验口径",
            "user_question": "北京门诊慢特病备案和病种范围如何核验？",
            "fact_bundle": [],
            "context_refs": [],
            "allowed_tools": ["policy.search_text", "policy.search_version"],
            "constraints": {
                "read_only": True,
                "no_final_audit_decision": True,
                "must_cite_evidence": True,
                "no_material_gap_diff": True,
            },
            "budget": {
                "max_model_calls": 4,
                "max_tool_calls": 6,
                "max_retrieval_rounds": 3,
                "max_llm_rewrite_calls": 1,
                "timeout_ms": 20000,
            },
        }
    )

    plan = service._normalize_retrieval_plan(  # type: ignore[attr-defined]
        task,
        {
            "policy_question": "北京门诊慢特病备案和病种范围如何核验？",
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": [
                    "special_disease_filing",
                    "special_disease_scope",
                ],
                "content_type": ["policy_text", "table_row"],
            },
            "top_k": 10,
            "fetch_k": 40,
            "rerank": False,
        },
    )

    assert plan["filters"]["jurisdiction"] == ["beijing"]
    assert plan["filters"]["policy_domain"] == [
        "special_disease_filing",
        "special_disease_scope",
    ]
    assert "remote_medical" not in plan["filters"]["policy_domain"]
    assert plan["filters"]["content_type"] == ["policy_text", "table_row"]
    assert plan["top_k"] == 5


def test_policy_filter_resolver_payment_ratio_uses_drug_catalog_not_price_reference() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="处方中的便通片在上海医保药品目录中的医保类别和本地支付比例是多少？"
        )
    )

    assert result.filters["jurisdiction"] == ["shanghai"]
    assert "drug_catalog" in result.filters["policy_domain"]
    assert "drug_product_price_reference" not in result.filters["policy_domain"]
    assert result.filters["content_type"] == ["table_row"]
    assert result.filters["can_cite_as_policy_basis"] is True
    assert _drug_price_reference_filters(
        "处方中的便通片在上海医保药品目录中的医保类别和本地支付比例是多少？"
    ) == {}


def test_policy_filter_resolver_drug_catalog_dosage_definition_uses_policy_text() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="北京医保药品目录中普通片剂包括哪些剂型？"
        )
    )

    assert result.filters["jurisdiction"] == ["beijing"]
    assert result.filters["policy_domain"] == ["drug_catalog"]
    assert result.filters["content_type"] == ["policy_text"]
    assert result.filters["can_cite_as_policy_basis"] is True


def test_policy_filter_resolver_special_disease_wording_maps_scope() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="北京门诊特殊疾病费用的限定支付条件怎么核验？"
        )
    )

    assert "special_disease_scope" in result.filters["policy_domain"]
    assert "special_disease_filing" in result.filters["policy_domain"]


def test_policy_filter_resolver_regionless_manual_service_query_uses_wide_filters() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="审核参保人零星报销门诊费用时，应要求提供哪些必要材料？法定办结时限是多少？",
            task_filters={
                "jurisdiction": ["national"],
                "policy_domain": ["manual_reimbursement"],
                "content_type": ["service_guide"],
            },
        )
    )

    assert "jurisdiction" not in result.filters
    assert result.filters["policy_domain"] == ["manual_reimbursement"]
    assert result.filters["content_type"] == ["policy_text"]
    assert result.filters["can_cite_as_policy_basis"] is True
    assert "template:manual_reimbursement_service_guide" in result.filter_source


def test_policy_filter_resolver_manual_reimbursement_scenario_query_uses_table_rows() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="审核北京参保人员手工报销时，涉及外埠就医、定点医药机构记账结算和急诊就医，应分别核验哪些材料或结算凭证？",
            task_filters={
                "jurisdiction": ["北京"],
                "policy_domain": ["手工报销", "急诊"],
                "content_type": ["政策文件"],
            },
        )
    )

    assert result.filters["jurisdiction"] == ["beijing"]
    assert result.filters["policy_domain"] == ["manual_reimbursement"]
    assert result.filters["content_type"] == ["table_row", "policy_text"]
    assert result.filters["can_cite_as_policy_basis"] is True
    assert "template:manual_reimbursement_scenario_slots" in result.filter_source
    assert "未出示社保卡或医保电子凭证" in result.policy_question
    assert "全额垫付" in result.policy_question
    assert "收据、处方、诊断证明" in result.policy_question
    assert "便民服务中心" in result.policy_question


def test_policy_filter_resolver_forced_filters_skip_question_price_override() -> None:
    result = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(
            user_question="处方中的便通片在上海医保药品目录中的医保类别和本地支付比例是多少？",
            task_filters={
                "jurisdiction": ["shanghai"],
                "policy_domain": ["drug_catalog"],
                "content_type": ["table_row"],
                "can_cite_as_policy_basis": True,
                "force_policy_filters": True,
            },
        )
    )

    assert result.filter_source == "task_filters:forced"
    assert result.confidence == 1.0
    assert result.filters == {
        "jurisdiction": ["shanghai"],
        "policy_domain": ["drug_catalog"],
        "content_type": ["table_row"],
        "can_cite_as_policy_basis": True,
    }


def test_policy_filter_resolver_maps_corpus_backed_policy_domains_to_slots() -> None:
    cases = [
        (
            "跨省异地就医直接结算中银行手续费和银行票据工本费能否由医保基金支付？",
            ["remote_medical"],
            ["policy_text"],
            ["remote_settlement_management"],
        ),
        (
            "北京城乡老年人参加城乡居民医保的参保范围是什么，慢性病长处方用药品种规格怎么衔接？",
            ["chronic_disease_long_prescription", "benefit"],
            ["policy_text"],
            ["benefit_policy", "chronic_long_prescription_policy"],
        ),
        (
            "上海住院床位费和急诊观察室床位费是否属于医保医疗服务设施支付范围？",
            ["shanghai_payment_scope"],
            ["policy_text"],
            ["shanghai_service_facility_scope"],
        ),
        (
            "医保基金监督检查中拒不配合调查会怎么处理？",
            ["fund_supervision"],
            ["policy_text"],
            ["fund_supervision_policy"],
        ),
    ]

    for question, expected_domains, expected_content_types, expected_slots in cases:
        resolved = PolicyFilterResolver().resolve(
            PolicyFilterResolverInput(user_question=question)
        )
        requirements = resolve_answer_requirements(
            user_question=question,
            question_slots=[],
            filters=resolved.filters,
        )

        assert resolved.filters["policy_domain"] == expected_domains
        assert resolved.filters["content_type"] == expected_content_types
        assert [item.slot_id for item in requirements] == expected_slots


def test_policy_required_field_extractor_uses_verified_field_aliases() -> None:
    question = "跨省异地就医直接结算中银行手续费和银行票据工本费能否由医保基金支付？"
    resolved = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(user_question=question)
    )
    requirements = resolve_answer_requirements(
        user_question=question,
        question_slots=[],
        filters=resolved.filters,
    )
    evidence = [
        NormalizedPolicyEvidence(
            evidence_id="policy-node:remote-bank-fee",
            source_ref="policy:remote-settlement",
            title="跨省异地就医直接结算通知",
            content=(
                "跨省异地就医医疗费用结算过程中发生的银行手续费、银行票据工本费"
                "不得从医保基金中列支。各级经办机构应按规定办理清算。"
            ),
            jurisdiction="national",
            policy_domain="remote_medical",
            content_type="policy_text",
            metadata={"node_id": "remote-bank-fee"},
        )
    ]

    facts, matches = extract_facts_for_requirements(requirements, evidence)

    assert [fact.slot_id for fact in facts] == ["remote_settlement_management"]
    assert "不得从医保基金中列支" in facts[0].display_text
    assert facts[0].value["matched_required_fields"] == ["银行费用不得列支基金"]
    assert matches[0].coverage_status == "covered"


def test_designated_institution_extractor_accepts_english_table_metadata_and_name_variants() -> None:
    question = "上海永宁堂大药房(普通合 伙)的药店编码和定点状态是什么？"
    resolved = PolicyFilterResolver().resolve(
        PolicyFilterResolverInput(user_question=question)
    )
    requirements = resolve_answer_requirements(
        user_question=question,
        question_slots=[],
        filters=resolved.filters,
    )
    evidence = [
        NormalizedPolicyEvidence(
            evidence_id="policy-node:sh-pharmacy-001",
            source_ref="policy:shanghai-designated-pharmacy",
            title="本市医疗保障定点零售药店名单",
            content=(
                "pharmacy_name: 上海永宁堂大药房（普通合伙）\n"
                "pharmacy_code: P3101150001\n"
                "status: 正常\n"
                "district: 浦东新区"
            ),
            jurisdiction="shanghai",
            policy_domain="designated_institution",
            content_type="table_row",
            metadata={"pharmacy_name": "上海永宁堂大药房（普通合伙）"},
        )
    ]

    facts, _ = extract_facts_for_requirements(requirements, evidence)

    assert [fact.slot_id for fact in facts] == ["designated_institution"]
    assert "P3101150001" in facts[0].display_text
    assert "正常" in facts[0].display_text


def test_lazy_policy_rag_prewarm_uses_bounded_safe_request() -> None:
    calls: list[dict[str, object]] = []

    def factory():
        def search(payload: dict[str, object]) -> dict[str, object]:
            calls.append(payload)
            return {"status": "ok", "result_count": 0, "evidence": []}

        return search

    client = LazyPolicyRagMcpClient(factory=factory)

    summary = client.prewarm()

    assert summary["status"] == "ok"
    assert summary["result_count"] == 0
    assert len(calls) == 1
    assert calls[0]["top_k"] == 1
    assert calls[0]["fetch_k"] == 8
    assert calls[0]["rerank"] is False


def test_call_policy_rag_times_out_slow_mcp_client() -> None:
    class SlowPolicyRagClient:
        def search(self, request: PolicySearchRequest) -> dict[str, object]:
            _ = request
            time.sleep(5)
            return {"status": "ok", "result_count": 1, "evidence": [{"node_id": "late"}]}

    request = PolicySearchRequest(
        question="北京参保人异地急诊手工报销如何处理？",
        filters={"jurisdiction": ["beijing"], "content_type": ["policy_text"]},
        top_k=5,
        fetch_k=20,
        rerank=False,
    )

    started = time.perf_counter()
    response, summary = call_policy_rag(
        client=SlowPolicyRagClient(),
        request=request,
        timeout_ms=1000,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.5
    assert response["status"] == "unavailable"
    assert response["error"] == "policy_rag_mcp_timeout"
    assert summary["status"] == "unavailable"
    assert summary["error_code"] == "policy_rag_mcp_timeout"
    assert summary["result_count"] == 0


def test_case_agent_business_materials_catalog_does_not_read_full_content() -> None:
    case_id = "CASE-CASEAGENT-001"
    tools = _tool_registry(case_id)

    catalog = tools.execute(
        current_case_id=case_id,
        tool_name="query_material_overview",
        raw_arguments={"case_id": case_id},
    )

    serialized = json.dumps(catalog.payload, ensure_ascii=False)
    assert catalog.status == "success"
    assert catalog.payload["section_key"] == "material_overview"
    assert catalog.payload["payload"]["groups"]
    assert '"content"' not in serialized
    assert '"tables"' not in serialized
    assert all(ref.startswith("material:") for ref in catalog.source_refs)


def test_case_agent_prescription_materials_reads_prescription_section() -> None:
    case_id = "CASE-CASEAGENT-001"
    tools = _tool_registry(case_id)

    detail = tools.execute(
        current_case_id=case_id,
        tool_name="query_prescription_materials",
        raw_arguments={"case_id": case_id},
    )
    assert detail.status == "success"
    assert detail.payload["section_key"] == "prescription_materials"
    assert detail.payload["payload"]["materials"]
    assert "prescription_drug_details" in detail.payload["payload"]["materials"][0]


def test_case_agent_fast_digest_keeps_material_sources_and_prescription_rows() -> None:
    case_id = "CASE-CASEAGENT-001"
    overview = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_material_overview",
            "section_key": "material_overview",
            "completeness": "full",
            "payload": {
                "material_total": 1,
                "groups": [
                    {
                        "category_id": "prescription",
                        "title": "处方购药材料",
                        "count": 1,
                        "materials": [
                            {
                                "material_id": f"{case_id}-BM-03",
                                "name": "处方购药记录",
                                "occurred_at": "2026-01-01",
                                "source": "定点医疗机构 / 院内药房",
                                "shape": "表格 + 图片",
                                "status": "已整理",
                                "operation": "read_only",
                                "source_ref": f"material:{case_id}-BM-03",
                            }
                        ],
                    }
                ],
            },
        },
        source_refs=[f"material:{case_id}-BM-03"],
    )
    prescriptions = CaseAgentToolResult(
        payload={
            "status": "ok",
            "capability": "query_prescription_materials",
            "section_key": "prescription_materials",
            "completeness": "full",
            "payload": {
                "materials": [
                    {
                        "material_id": f"{case_id}-BM-03",
                        "name": "处方购药记录",
                        "basic_info": {"source_ref": f"material:{case_id}-BM-03"},
                        "content": {"raw_material_summary": "药品项目 1 项"},
                        "prescription_drug_details": [
                            {
                                "发生时间": "2026-01-01",
                                "药品名称": "恩格列净片",
                                "规格": "10mg*10片/盒",
                                "数量": "18盒",
                                "单价": "261.65",
                                "金额": "4709.66",
                                "来源": "院内药房",
                                "用法": "口服",
                            }
                        ],
                    }
                ]
            },
        },
        source_refs=[f"material:{case_id}-BM-03"],
    )
    service = CaseAgentService.__new__(CaseAgentService)
    state = {
        "query_semantics": {"user_goal": "材料来源分别是什么，并输出所有处方药品明细"},
        "user_message": SimpleNamespace(content="材料来源分别是什么，并输出所有处方药品明细"),
        "capability_results": [
            ("query_material_overview", overview),
            ("query_prescription_materials", prescriptions),
        ],
    }

    digest, refs = service._build_context_digest(state)

    overview_materials = digest["material_overview_digest"]["summary"]["groups"][0]["materials"]
    prescription_material = digest["prescription_materials_digest"]["summary"]["materials"][0]
    assert overview_materials[0]["source"]
    assert overview_materials[0]["shape"]
    assert overview_materials[0]["occurred_at"]
    assert prescription_material["prescription_drug_details"]
    assert "药品名称" in prescription_material["prescription_drug_details"][0]
    assert refs


def test_case_agent_sources_use_business_titles_not_raw_refs() -> None:
    case_id = "CASE-CASEAGENT-001"
    tools = _tool_registry(case_id)
    evidence = tools.execute(
        current_case_id=case_id,
        tool_name="query_evidence_package",
        raw_arguments={"case_id": case_id},
    )

    assert evidence.sources
    titles = {source.title for source in evidence.sources}
    assert "业务规则核验" in titles
    assert all(":" not in title for title in titles)


def test_case_agent_review_analysis_reader_does_not_trigger_generation() -> None:
    class NoAnalysisAdvisor:
        def __init__(self) -> None:
            self.requested_case_ids: list[str] = []

        def get_current_analysis(self, case_id: str) -> None:
            self.requested_case_ids.append(case_id)
            return None

    case_id = "CASE-CASEAGENT-001"
    advisor = NoAnalysisAdvisor()
    tools = _tool_registry(case_id, evidence_agent=advisor)

    result = tools.execute(
        current_case_id=case_id,
        tool_name="query_case_judgement",
        raw_arguments={"case_id": case_id},
    )

    assert result.status == "unavailable"
    assert result.error_code == "section_not_ready"
    assert advisor.requested_case_ids == [case_id, case_id]


def test_case_agent_safety_blocks_final_decisions_and_sensitive_labels() -> None:
    with pytest.raises(CaseAgentSafetyError):
        assert_safe_text("建议拒付")
    with pytest.raises(CaseAgentSafetyError):
        assert_safe_text("系统自动通过")
    with pytest.raises(CaseAgentSafetyError):
        assert_safe_text("结论为骗保")
    with pytest.raises(CaseAgentSafetyError):
        assert_safe_text("字段 RES 不应出现")

    assert_safe_text("本助手仅提供辅助分析，不形成最终业务处置结论")
    assert_safe_text("支付处理结论需由审核人员确认")
    assert_safe_text("存在需关注的异常风险线索")
    assert_safe_text("能不能给出支付处理结论", allow_decision_request=True)


def test_case_agent_safety_allows_generic_policy_identity_document_terms() -> None:
    source = CaseAgentSource(
        source_ref="policy-evidence:identity-doc",
        source_type="policy_rag",
        title="异地就医备案政策",
    )
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="政策依据显示，参保人员可通过医保电子凭证、有效身份证件或社会保障卡办理备案。",
                source_refs=["policy-evidence:identity-doc"],
            )
        ],
        sources=[source],
    )

    validate_answer(answer, [source])

    with pytest.raises(CaseAgentSafetyError):
        assert_safe_text("参保人身份证号码：110101199001011234")


def test_case_agent_safety_allows_sourced_review_status_fact() -> None:
    source = CaseAgentSource(
        source_ref="case:CASE-001:basic",
        source_type="case",
        title="案件基础信息",
    )
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="当前案件基础信息显示审核状态为审核通过。",
                source_refs=["case:CASE-001:basic"],
            )
        ],
        sources=[source],
    )

    validate_answer(answer, [source])

    unsafe = answer.model_copy(
        update={
            "content_blocks": [
                CaseAgentContentBlock(
                    text="建议审核通过。",
                    source_refs=["case:CASE-001:basic"],
                )
            ]
        }
    )
    with pytest.raises(CaseAgentSafetyError):
        validate_answer(unsafe, [source])


def test_case_agent_shortcut_detects_product_questions_only() -> None:
    greeting = detect_shortcut("你好")
    assert greeting is not None
    assert greeting.kind == "greeting"
    assert greeting.intent == "general_help"

    policy_status = detect_shortcut("政策库现在能查吗？")
    assert policy_status is not None
    assert policy_status.kind == "capability_status"
    assert policy_status.intent == "domain_expert_query"
    assert policy_status.display_mode == "unavailable"

    assert detect_shortcut("当前地区政策怎么规定？") is None
    assert detect_shortcut("阿托伐他汀用在当前诊断下合理吗？") is None
    assert detect_shortcut("当前案件有哪些规则命中和风险提示？") is None


def test_case_agent_shortcut_answer_is_standard_payload() -> None:
    shortcut = detect_shortcut("会话会跨案件共享吗？")
    assert shortcut is not None

    answer = build_shortcut_answer(shortcut)

    assert answer.display_mode == "plain"
    assert answer.sources == []
    assert answer.content_blocks[0].source_refs == []
    assert answer.metadata["shortcut"] is True
    validate_answer(answer, [])


def test_case_agent_node_wrapper_records_execution_and_checkpoint() -> None:
    class DurabilityRepository:
        def __init__(self) -> None:
            self.started: list[dict] = []
            self.finished: list[dict] = []
            self.failed: list[dict] = []
            self.checkpoints: list[dict] = []

        def start_node_execution(self, run_id: str, **kwargs) -> int:
            self.started.append({"run_id": run_id, **kwargs})
            return len(self.started)

        def finish_node_execution(self, run_id: str, **kwargs) -> None:
            self.finished.append({"run_id": run_id, **kwargs})

        def fail_node_execution(self, run_id: str, **kwargs) -> None:
            self.failed.append({"run_id": run_id, **kwargs})

        def save_checkpoint(self, run_id: str, **kwargs) -> None:
            self.checkpoints.append({"run_id": run_id, **kwargs})

    repository = DurabilityRepository()
    service = CaseAgentService.__new__(CaseAgentService)
    service._repository = repository

    state = service._invoke_node(
        "business_semantic_planner",
        lambda current: {**current, "intent": "general_help", "next_action": "validate_execution_plan"},
        {"run_id": "crun_test", "actor_id": "actor_test"},
    )

    assert state["intent"] == "general_help"
    assert repository.started[0]["node_name"] == "business_semantic_planner"
    assert repository.finished[0]["sequence"] == 1
    assert repository.failed == []
    assert repository.checkpoints[0]["node_name"] == "business_semantic_planner"
    assert repository.checkpoints[0]["safe_to_resume"] is True


def test_case_agent_persist_result_marks_clarification_waiting_for_user() -> None:
    class WaitingRepository:
        def __init__(self) -> None:
            self.run_updates: list[dict] = []
            self.memory_updates: list[dict] = []
            self.events: list[dict] = []

        def update_run(self, run_id: str, **kwargs) -> None:
            self.run_updates.append({"run_id": run_id, **kwargs})

        def append_assistant_message(self, **kwargs) -> CaseAgentMessage:
            return CaseAgentMessage(
                message_id="cmsg_assistant",
                session_id=kwargs["session_id"],
                role="assistant",
                content=kwargs["content"],
                active_stage=kwargs["active_stage"],
                answer_payload=kwargs["answer_payload"],
                source_refs=kwargs["source_refs"],
                created_at=datetime.now(timezone.utc),
            )

        def update_working_memory(self, **kwargs) -> None:
            self.memory_updates.append(kwargs)

        def append_event(self, run_id: str, event_type: str, message: str, payload=None):
            self.events.append(
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": payload or {},
                }
            )

    repository = WaitingRepository()
    service = CaseAgentService.__new__(CaseAgentService)
    service._repository = repository
    answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[
            CaseAgentContentBlock(text="为了继续处理这个问题，请补充：药品名称。")
        ],
        metadata={"intent": "domain_expert_query", "missing_slots": ["drug_name"]},
    )
    state = {
        "run": SimpleNamespace(run_id="crun_wait", session_id="csess_wait"),
        "session": SimpleNamespace(session_id="csess_wait"),
        "actor_id": "actor_wait",
        "user_message": SimpleNamespace(content="这个药合理吗？", active_stage=None),
        "answer": answer,
        "completion_status": "waiting_for_user",
        "pending_clarification": {
            "intent": "domain_expert_query",
            "missing_slots": ["drug_name"],
        },
    }

    persist_result_node(service, state)

    final_update = repository.run_updates[-1]
    assert final_update["status"] == "waiting_for_user"
    assert final_update["current_node"] == "waiting_for_user"
    assert final_update["pending_clarification"]["missing_slots"] == ["drug_name"]
    assert repository.memory_updates[-1]["pending_tool"] == "clarification"
    assert repository.events[-1]["event_type"] == "waiting_for_user"


def test_case_agent_fallback_reports_model_auth_failure_safely() -> None:
    answer = CaseAgentService._fallback_answer(
        "Error code: 401 - {'error': {'message': 'Invalid Authentication'}}",
        "error",
    )

    assert answer.display_mode == "error"
    assert "Case Agent 模型认证失败" in answer.fallback_notice
    assert "Invalid Authentication" not in answer.fallback_notice


def test_case_agent_fallback_hides_safety_guardrail_details() -> None:
    answer = CaseAgentService._fallback_answer(
        "prohibited_conclusion: 内容包含裁决性或越权结论",
        "error",
    )

    assert answer.display_mode == "error"
    assert answer.fallback_notice == "当前回答未通过安全边界校验，请换一种问法或查看当前案件已有资料。"
    assert "prohibited_conclusion" not in answer.fallback_notice
    assert "裁决性" not in answer.fallback_notice


def test_case_agent_fallback_hides_plan_field_validation_details() -> None:
    answer = CaseAgentService._fallback_answer(
        "field names must be simple top-level identifiers",
        "error",
    )

    assert answer.display_mode == "error"
    assert answer.fallback_notice == "当前回答未通过结构或引用校验，请稍后重试或查看当前案件已有资料。"
    assert "field names" not in answer.fallback_notice


def test_case_agent_fallback_hides_unknown_internal_error_details() -> None:
    answer = CaseAgentService._fallback_answer("'semantic_intent_perception'", "error")

    assert answer.fallback_notice == (
        "Case Agent 本次处理未能完成，请稍后重试；当前案件资料与人工审核流程不受影响。"
    )
    assert "semantic_intent_perception" not in answer.fallback_notice


def test_case_agent_grounded_answer_closes_inline_sources() -> None:
    source = CaseAgentSource(
        source_ref="case:CASE-001:snapshot",
        source_type="case",
        title="案件摘要",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(label="案件类型", value="门诊"),
                CaseAgentSourceDetailField(label="申报摘要", value="当前案件摘要"),
            ]
        ),
    )
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="已读取当前案件摘要。",
                source_refs=["case:CASE-001:snapshot"],
            ),
        ],
        sources=[source],
    )

    validate_answer(answer, [source])


def test_case_agent_plain_answer_must_not_carry_sources() -> None:
    answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[
            CaseAgentContentBlock(
                text="我可以帮助查询当前案件事实、规则结果和已有 Evidence 分析。",
            )
        ],
    )

    validate_answer(answer, [])


def test_case_agent_general_help_plain_answer_payload_passes_validation() -> None:
    answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[
            CaseAgentContentBlock(
                text=(
                    "当前案件助手可以帮助查询当前案件事实、规则依据、已有 Evidence 分析和审核工作笔记。"
                    "本助手仅基于当前可用数据提供查询和辅助分析，不形成最终业务处置结论。"
                ),
                source_refs=[],
            )
        ],
        sources=[],
        fallback_notice="",
        metadata={
            "intent": "general_help",
            "requires_citation": False,
            "capabilities_used": [],
        },
    )

    validate_answer(answer, [])


def test_case_agent_final_answer_prompt_uses_dynamic_block_policy() -> None:
    prompt = build_final_answer_prompt("grounded", fast_mode_enabled=True)

    assert "单一事实查询只写 1 个 content_block" in prompt
    assert "综合分析最多 4 个 content_blocks" in prompt
    assert "默认输出 3 到 5 个 content_blocks" not in prompt
    assert "fallback_notice 必须为空字符串" in prompt
    assert "interface ExpectedResponse" in prompt
    assert "answer_markdown" in prompt
    assert "citations" in prompt
    assert "禁止输出旧字段或额外字段 brief_answer, citations" not in prompt


def test_case_agent_single_fact_payload_can_use_one_grounded_block() -> None:
    source = CaseAgentSource(
        source_ref="case:CASE-001:claimant_profile",
        source_type="claimant_profile",
        title="申报人基础信息",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(label="性别", value="女"),
            ]
        ),
    )
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="当前申报人性别为女。",
                source_refs=["case:CASE-001:claimant_profile"],
            )
        ],
        sources=[source],
        fallback_notice="",
        metadata={
            "intent": "case_task",
            "requires_citation": True,
            "capabilities_used": ["query_claimant_profile"],
        },
    )

    validate_answer(answer, [source])
    assert len(answer.content_blocks) == 1
    assert answer.fallback_notice == ""


def test_case_agent_normalizes_markdown_json_and_string_blocks() -> None:
    raw = """```json
{
  "display_mode": "plain",
  "content_blocks": ["我可以帮助查询当前案件事实、规则结果和已有 Evidence 分析。"],
  "sources": [],
  "fallback_notice": "",
  "metadata": {"intent": "general_help", "requires_citation": false, "capabilities_used": []},
  "brief_answer": []
}
```"""

    payload = json.loads(CaseAgentService._extract_answer_json(raw))
    normalized, reasons = CaseAgentService._normalize_answer_payload(
        payload,
        {"available_sources": []},
    )
    answer = CaseAgentAnswer.model_validate(normalized)

    assert "removed_extra_top_level_fields" in reasons
    assert "normalized_content_blocks" in reasons
    assert "brief_answer" not in normalized
    assert normalized["content_blocks"][0]["source_refs"] == []
    validate_answer(answer, [])


def test_case_agent_extracts_json_before_trailing_text() -> None:
    raw = """
可以，下面是 JSON：
{
  "display_mode": "plain",
  "content_blocks": [
    {"text": "当前案件助手可以说明会话和能力范围。", "source_refs": []}
  ],
  "sources": [],
  "fallback_notice": "",
  "metadata": {"intent": "general_help", "requires_citation": false, "capabilities_used": []}
}
以上为结构化结果。
"""

    payload = json.loads(CaseAgentService._extract_answer_json(raw))
    normalized, _ = CaseAgentService._normalize_answer_payload(
        payload,
        {"available_sources": []},
    )
    answer = CaseAgentAnswer.model_validate(normalized)

    assert answer.display_mode == "plain"
    assert answer.sources == []
    assert answer.fallback_notice == ""
    validate_answer(answer, [])


def test_case_agent_missing_content_blocks_stays_invalid_schema() -> None:
    normalized, reasons = CaseAgentService._normalize_answer_payload(
        {},
        {"available_sources": []},
    )

    assert normalized["content_blocks"] == []
    assert "filled_missing_content_blocks" in reasons
    with pytest.raises(ValidationError):
        CaseAgentAnswer.model_validate(normalized)


def test_case_agent_unavailable_normalizer_clears_sources_and_source_refs() -> None:
    payload = {
        "display_mode": "unavailable",
        "content_blocks": [
            {
                "text": "当前政策知识库尚未接入，暂不能基于政策库返回可追溯口径。",
                "source_refs": ["case:CASE-001:snapshot"],
            }
        ],
        "sources": [
            {
                "source_ref": "case:CASE-001:snapshot",
                "source_type": "case",
                "title": "案件摘要",
                "version": None,
                "detail": {"fields": []},
                "metadata": {},
            }
        ],
        "fallback_notice": "政策知识库能力未接入。",
        "metadata": {
            "intent": "domain_expert_query",
            "requires_citation": False,
            "capabilities_used": ["expert_agent:policy_analysis"],
        },
    }

    normalized, reasons = CaseAgentService._normalize_answer_payload(
        payload,
        {"available_sources": []},
    )
    answer = CaseAgentAnswer.model_validate(normalized)

    assert "cleared_sources_for_non_grounded" in reasons
    assert "cleared_source_refs_for_non_grounded" in reasons
    assert normalized["sources"] == []
    assert normalized["content_blocks"][0]["source_refs"] == []
    validate_answer(answer, [])


def test_case_agent_grounded_normalizer_completes_sources_from_available_refs() -> None:
    source = CaseAgentSource(
        source_ref="case:CASE-001:snapshot",
        source_type="case",
        title="案件摘要",
        detail=CaseAgentSourceDetail(
            fields=[
                CaseAgentSourceDetailField(label="案件类型", value="门诊"),
            ]
        ),
    )
    payload = {
        "display_mode": "grounded",
        "content_blocks": [
            {
                "text": "当前案件摘要显示该案件为门诊类型。",
                "source_refs": ["case:CASE-001:snapshot"],
            }
        ],
        "sources": [{"source_ref": "case:CASE-001:snapshot"}],
        "fallback_notice": "",
        "metadata": {
            "intent": "case_snapshot_query",
            "requires_citation": True,
            "capabilities_used": ["query_case_basic_info"],
        },
    }

    normalized, reasons = CaseAgentService._normalize_answer_payload(
        payload,
        {"available_sources": [source]},
    )
    answer = CaseAgentAnswer.model_validate(normalized)

    assert "normalized_grounded_sources" in reasons
    assert normalized["sources"][0]["title"] == "案件摘要"
    validate_answer(answer, [source])


def test_case_agent_grounded_missing_citation_still_fails() -> None:
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="已读取当前案件摘要。",
                source_refs=[],
            )
        ],
        sources=[],
    )

    with pytest.raises(CaseAgentSafetyError) as exc:
        validate_answer(answer, [])
    assert exc.value.code == "citation_missing_for_grounded"


def test_case_agent_general_help_allows_boundary_terms_but_blocks_sensitive_labels() -> None:
    answer = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[
            CaseAgentContentBlock(
                text="我可以说明会话、新建按钮和当前助手能力，但不形成拒付、处罚或欺诈认定等结论。",
            )
        ],
        metadata={"intent": "general_help"},
    )

    validate_answer(answer, [])

    sensitive = CaseAgentAnswer(
        display_mode="plain",
        content_blocks=[
            CaseAgentContentBlock(
                text="功能说明中也不能出现字段 RES。",
            )
        ],
        metadata={"intent": "general_help"},
    )

    with pytest.raises(CaseAgentSafetyError):
        validate_answer(sensitive, [])


def test_case_agent_rejects_unknown_inline_source() -> None:
    answer = CaseAgentAnswer(
        display_mode="grounded",
        content_blocks=[
            CaseAgentContentBlock(
                text="已读取当前案件摘要。",
                source_refs=["case:CASE-001:snapshot"],
            )
        ],
        sources=[
            CaseAgentSource(
                source_ref="case:CASE-001:snapshot",
                source_type="case",
                title="案件摘要",
            )
        ],
    )

    with pytest.raises(CaseAgentSafetyError):
        validate_answer(answer, [])


def test_case_agent_fast_context_digest_filters_to_referenced_sources() -> None:
    service = CaseAgentService.__new__(CaseAgentService)
    source_risk = CaseAgentSource(
        source_ref="risk:CASE-001:model_warning",
        source_type="model_warning",
        title="模型风险预警",
    )
    source_rule = CaseAgentSource(
        source_ref="rule:OP-R001:visit_frequency",
        source_type="rule",
        title="规则：高频就诊核验",
    )
    source_unmatched = CaseAgentSource(
        source_ref="rule:OP-R009:unmatched",
        source_type="rule",
        title="未进入摘要的规则",
    )
    result = CaseAgentToolResult(
        payload={
            "status": "ok",
            "risk_rule_context": {
                "case_id": "CASE-001",
                "risk_level": "low",
                "review_priority": "manual_review",
                "risk_signal_summary": "模型信号与规则证据待核验",
                "model_warning": {
                    "label": "有预警",
                    "reason": "就诊频次需要核验",
                    "source_ref": "risk:CASE-001:model_warning",
                },
                "model_signal_reasons": ["就诊频次需要核验"],
                "risk_breakdown": {},
                "rules": [
                    {
                        "rule_id": "OP-R001",
                        "rule_name": "高频就诊核验",
                        "hit": True,
                        "severity": "medium",
                        "rule_action": "MANUAL_REVIEW",
                        "current_value": "8",
                        "threshold": ">=6",
                        "reason": "月就诊次数偏高",
                        "source_ref": "rule:OP-R001:visit_frequency",
                    },
                    {
                        "rule_id": "OP-R009",
                        "rule_name": "未命中规则",
                        "hit": False,
                        "source_ref": "rule:OP-R009:unmatched",
                    },
                ],
            },
        },
        source_refs=[
            "risk:CASE-001:model_warning",
            "rule:OP-R001:visit_frequency",
            "rule:OP-R009:unmatched",
        ],
        sources=[source_risk, source_rule, source_unmatched],
    )
    state = {
        "run": SimpleNamespace(case_id="CASE-001"),
        "user_message": SimpleNamespace(content="当前案件有哪些规则命中和风险提示？"),
        "context_plan": {"display_mode": "grounded"},
        "capability_results": [("query_risk_score", result)],
        "available_sources": [source_risk, source_rule, source_unmatched],
    }

    service._apply_fast_context_digest(state)

    assert "context_digest" in state
    assert "query_risk_score_digest" in state["context_digest"]
    assert [source.source_ref for source in state["available_sources"]] == [
        "risk:CASE-001:model_warning",
        "rule:OP-R001:visit_frequency",
        "rule:OP-R009:unmatched",
    ]


def test_policy_handoff_comparison_rebuilds_continuous_markdown_with_citations() -> None:
    claims = [
        CaseAgentClaim(
            claim_id="claim-directory",
            need_id="need-directory",
            need_text="就医地目录",
            text="支付范围按就医地规定执行。",
            source_refs=["policy:remote"],
        ),
        CaseAgentClaim(
            claim_id="claim-benefit",
            need_id="need-benefit",
            need_text="参保地待遇",
            text="起付标准、支付比例和最高支付限额按参保地规定执行。",
            source_refs=["policy:remote"],
        ),
    ]
    citations = [
        CaseAgentCitation(
            citation_id="citation-directory",
            label=1,
            claim_id="claim-directory",
            source_refs=["policy:remote"],
        ),
        CaseAgentCitation(
            citation_id="citation-benefit",
            label=2,
            claim_id="claim-benefit",
            source_refs=["policy:remote"],
        ),
    ]

    answer_markdown, reconciled_claims, reconciled_citations = _reconcile_policy_output(
        answer_markdown=(
            "1. 支付范围按就医地规定执行。[1]\n"
            "2. 起付标准、支付比例和最高支付限额按参保地规定执行。[2]"
        ),
        claims=claims,
        citations=citations,
        need_answers=[
            {
                "need_id": "need-directory",
                "status": "supported",
                "answer_text": "支付范围按就医地规定执行。",
            },
            {
                "need_id": "need-benefit",
                "status": "supported",
                "answer_text": "起付标准、支付比例和最高支付限额按参保地规定执行。",
            },
        ],
        answer_mode="comparison",
    )

    assert not answer_markdown.startswith("1. ")
    assert "支付范围按就医地规定执行。[1]" in answer_markdown
    assert "起付标准、支付比例和最高支付限额按参保地规定执行。[2]" in answer_markdown
    assert len(reconciled_claims) == 2
    assert [citation.label for citation in reconciled_citations] == [1, 2]
