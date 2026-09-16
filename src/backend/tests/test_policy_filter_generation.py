from __future__ import annotations

import json

import pytest

from src.backend.application.agent.expert_agent.schemas import ExpertAnalysisTask
from src.backend.application.agent.expert_agent.service.orchestrator import (
    ExpertAgentService,
)
from src.backend.application.agent.expert_agent.service.policy_filter_contract import (
    PolicyFilters,
    policy_retrieval_plan_tool_schema,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    PolicyFilterOutputError,
    PolicyFilterResolver,
)
from src.backend.application.agent.expert_agent.service.recall_filter_projector import (
    RecallFilterProjector,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse
from src.backend.application.agent.runtime.gateways.fake import ScriptedModelGateway
from src.backend.scripts.evaluate_policy_filter_generation import compare_filters


def plan_payload(filters: dict[str, object]) -> dict[str, object]:
    return {
        "policy_question": "测试政策问题",
        "filters": filters,
        "information_needs": ["测试问题需要的政策依据"],
        "fetch_k": 40,
        "rerank": False,
        "need_case_context": False,
        "filter_reason": "测试",
        "filter_confidence": 0.9,
    }


def test_policy_filter_resolver_repairs_json_and_normalizes_safe_aliases() -> None:
    raw = (
        '{"policy_question":"测试政策问题","filters":'
        '{"jurisdiction":["国家"],"policy_domain":["基金监管"],'
        '"content_type":["政策正文",],},}'
    )

    plan = PolicyFilterResolver().resolve_llm_plan(raw)

    assert plan["filters"] == {
        "jurisdiction": ["national"],
        "policy_domain": ["fund_supervision"],
        "content_type": ["policy_text"],
        "can_cite_as_policy_basis": True,
    }


def test_policy_filter_contract_rejects_unknown_values_and_extra_fields() -> None:
    resolver = PolicyFilterResolver()

    with pytest.raises(PolicyFilterOutputError):
        resolver.resolve_llm_plan(
            plan_payload(
                {
                    "jurisdiction": ["guangdong"],
                    "policy_domain": ["unknown_domain"],
                    "content_type": ["web_page"],
                    "unexpected": True,
                }
            )
        )


def test_policy_filter_contract_does_not_expand_valid_llm_filters() -> None:
    filters = {
        "jurisdiction": ["national"],
        "policy_domain": ["fund_supervision"],
        "content_type": ["policy_text"],
        "can_cite_as_policy_basis": True,
    }

    plan = PolicyFilterResolver().resolve_llm_plan(plan_payload(filters))

    assert plan["filters"] == filters


def test_registered_filter_plan_uses_same_deepseek_once_for_repair() -> None:
    invalid = plan_payload(
        {
            "jurisdiction": ["national"],
            "policy_domain": ["not_registered"],
            "content_type": ["policy_text"],
        }
    )
    valid = plan_payload(
        {
            "jurisdiction": ["national"],
            "policy_domain": ["fund_supervision"],
            "content_type": ["policy_text"],
        }
    )
    gateway = ScriptedModelGateway(
        [
            ModelResponse(content=json.dumps(invalid, ensure_ascii=False)),
            ModelResponse(content=json.dumps(valid, ensure_ascii=False)),
        ]
    )

    plan = ExpertAgentService().generate_policy_retrieval_plan(
        user_question="医保经办机构可以采取哪些审核方式？",
        model_gateway=gateway,
    )

    assert plan["filters"]["policy_domain"] == ["fund_supervision"]
    assert plan["filter_diagnostics"]["repair_count"] == 1
    assert plan["filter_diagnostics"]["valid_on_first_call"] is False
    assert len(gateway.requests) == 2
    assert gateway.requests[0]["temperature"] == 0.1
    assert gateway.requests[0]["thinking_enabled"] is False
    assert gateway.requests[0]["tools"][0]["function"]["name"] == (
        "submit_policy_retrieval_plan"
    )


def test_registered_filter_plan_stops_after_one_failed_repair() -> None:
    invalid = plan_payload(
        {
            "jurisdiction": ["national"],
            "policy_domain": ["not_registered"],
            "content_type": ["policy_text"],
        }
    )
    gateway = ScriptedModelGateway(
        [
            ModelResponse(content=json.dumps(invalid)),
            ModelResponse(content=json.dumps(invalid)),
        ]
    )

    plan = ExpertAgentService().generate_policy_retrieval_plan(
        user_question="测试问题",
        model_gateway=gateway,
    )

    assert len(gateway.requests) == 2
    assert plan["filter_source"] == "llm_fallback_recall_only"
    assert plan["allow_broad_filters"] is True
    assert plan["filter_confidence"] == 0.0
    assert plan["filters"] == {
        "jurisdiction": [],
        "policy_domain": [],
        "content_type": ["policy_text", "table_row"],
        "can_cite_as_policy_basis": True,
    }
    assert plan["filter_diagnostics"]["fallback_mode"] == "recall_only"
    assert "strict_filter_generation_failed" in plan["filter_warnings"]


def test_case_context_does_not_mutate_generated_filters() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask(
        task_id="filter_case_context",
        parent_run_id="parent",
        case_id="CASE-001",
        user_question="医保经办机构可以采取哪些审核方式？",
    )
    filters = PolicyFilters(
        jurisdiction=["national"],
        policy_domain=["fund_supervision"],
        content_type=["policy_text"],
    ).model_dump(mode="json")
    plan = {
        "policy_question": task.user_question,
        "filters": filters,
        "filter_source": "llm_registered",
        "filter_confidence": 0.9,
        "filter_diagnostics": {},
    }
    observations = [
        {
            "payload": {
                "case_context": {
                    "insured_region": "北京",
                    "treatment_region": "上海",
                    "scenario": "定点医药机构急诊",
                }
            }
        }
    ]

    enriched = service._enrich_plan_with_case_context(  # type: ignore[attr-defined]
        task,
        plan,
        observations,
    )

    assert enriched["filters"] == filters
    assert enriched["filter_diagnostics"]["case_context_filter_mutation"] is False


def test_filter_comparison_uses_set_equality_and_reports_deltas() -> None:
    comparison = compare_filters(
        {
            "jurisdiction": ["beijing"],
            "policy_domain": ["benefit"],
            "content_type": ["policy_text"],
        },
        {
            "jurisdiction": ["beijing"],
            "policy_domain": ["benefit", "special_disease_filing"],
            "content_type": ["policy_text", "service_guide"],
        },
    )

    assert comparison["exact_match"] is False
    assert comparison["extra_values"]["policy_domain"] == [
        "special_disease_filing"
    ]
    assert comparison["extra_values"]["content_type"] == ["service_guide"]


def test_registered_tool_schema_is_generated_from_pydantic_contract() -> None:
    tool = policy_retrieval_plan_tool_schema()

    assert tool["function"]["name"] == "submit_policy_retrieval_plan"
    filters_ref = tool["function"]["parameters"]["properties"]["filters"]["$ref"]
    assert filters_ref.endswith("/$defs/PolicyFilters")


def test_recall_filter_projector_keeps_explicit_local_and_national_scope() -> None:
    strict = {
        "jurisdiction": ["national"],
        "policy_domain": ["benefit"],
        "content_type": ["policy_text"],
        "can_cite_as_policy_basis": True,
    }

    projection = RecallFilterProjector().project(
        user_question="上海参保人申请手工报销需要哪些材料？",
        strict_filters=strict,
    )

    assert strict["jurisdiction"] == ["national"]
    assert projection.filters == {
        "jurisdiction": ["shanghai", "national"],
        "policy_domain": [],
        "content_type": ["policy_text", "table_row"],
        "can_cite_as_policy_basis": True,
    }
    assert projection.explicit_jurisdictions == ("shanghai",)


def test_recall_filter_projector_drops_inferred_jurisdiction_and_domain() -> None:
    projection = RecallFilterProjector().project(
        user_question="参保人在异地就医后如何申请手工报销？",
        strict_filters={
            "jurisdiction": ["national"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
            "can_cite_as_policy_basis": True,
        },
    )

    assert projection.filters["jurisdiction"] == []
    assert projection.filters["policy_domain"] == []
    assert "inferred_jurisdiction_removed" in projection.reasons


def test_filter_contract_rejects_document_genres_as_chunk_content_types() -> None:
    with pytest.raises(PolicyFilterOutputError):
        PolicyFilterResolver().resolve_llm_plan(
            plan_payload(
                {
                    "jurisdiction": ["beijing"],
                    "policy_domain": ["manual_reimbursement"],
                    "content_type": ["service_guide"],
                    "can_cite_as_policy_basis": True,
                }
            )
        )
