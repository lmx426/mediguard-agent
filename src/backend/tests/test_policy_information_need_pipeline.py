from __future__ import annotations

import json
from typing import Any

import pytest

from src.backend.application.agent.expert_agent.schemas import (
    ExpertAnalysisTask,
    PolicyEvidence,
)
from src.backend.application.agent.expert_agent.service.orchestrator import (
    ExpertAgentService,
    _normalize_answer_mode,
)
from src.backend.application.agent.expert_agent.service.policy_filter_resolver import (
    PolicyFilterOutputError,
    PolicyFilterResolver,
)
from src.backend.application.agent.expert_agent.service.policy_need_answer import (
    build_information_need_answer,
)
from src.backend.application.agent.expert_agent.service.policy_need_synthesis import (
    apply_batch_synthesis,
)
from src.backend.application.agent.runtime.gateways.base import ModelResponse


def _evidence(
    evidence_ref: str,
    excerpt: str,
    *,
    content_type: str = "table_row",
    policy_domain: str = "manual_reimbursement",
    source_ref: str | None = None,
) -> PolicyEvidence:
    return PolicyEvidence(
        evidence_ref=evidence_ref,
        source_ref=source_ref or f"source:{evidence_ref}",
        title="测试政策证据",
        excerpt=excerpt,
        jurisdiction="beijing",
        policy_domain=policy_domain,
        content_type=content_type,
        source_url="https://example.test/policy",
    )


def _build(
    *,
    question: str,
    needs: list[str],
    evidence: list[PolicyEvidence],
):
    return build_information_need_answer(
        user_question=question,
        information_needs=needs,
        answer_mode="list" if len(needs) > 1 else "fact_lookup",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text", "table_row"],
            "can_cite_as_policy_basis": True,
        },
        evidence=evidence,
    )


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden for key in value) or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_v2_output_has_one_answer_per_need_and_no_legacy_slot_fields() -> None:
    result = _build(
        question="北京门诊手工报销需要哪些材料，对票据有什么要求？",
        needs=["需要提交的材料清单", "票据和结算凭证要求"],
        evidence=[
            _evidence(
                "ev-material",
                "material_name: 费用收据\nrequired_when: 普通门诊手工报销",
            ),
            _evidence(
                "ev-voucher",
                "material_name: 医疗费用结算凭证\nrequired_when: 普通门诊手工报销",
            ),
        ],
    )

    payload = result.model_dump(mode="json")
    assert len(result.need_answers) == 2
    assert len({item.need_id for item in result.need_answers}) == 2
    assert not _contains_key(
        payload,
        {"slot_id", "need_kind", "answer_slots", "question_slots", "requirement_id"},
    )


def test_ordinary_outpatient_need_rejects_foreign_treatment_material_rows() -> None:
    result = _build(
        question="北京普通门诊手工报销需要提交哪些材料？",
        needs=["普通门诊手工报销材料"],
        evidence=[
            _evidence(
                "ev-foreign",
                "资料标题: 外埠就医材料链\n"
                "来源ID: internal-source\n"
                "field_key: material_name\n"
                "material_name: 外埠就医费用收据\n"
                "required_when: 外埠就医手工报销\n"
                "case_relevance: internal-only",
            )
        ],
    )

    assert result.support_status == "unsupported"
    assert result.need_answers[0].status == "missing"
    assert "外埠就医费用收据" not in result.expert_answer


def test_material_rows_are_composed_as_user_safe_list_without_metadata() -> None:
    result = _build(
        question="北京普通门诊手工报销需要提交哪些材料？",
        needs=["普通门诊手工报销材料清单"],
        evidence=[
            _evidence(
                "ev-receipt",
                "资料标题: 普通门诊材料链\n"
                "来源ID: internal-source\n"
                "field_key: material_name\n"
                "material_name: 医疗费用收据\n"
                "required_when: 普通门诊手工报销\n"
                "case_relevance: internal-only",
            ),
            _evidence(
                "ev-list",
                "material_name: 医疗费用明细\nrequired_when: 普通门诊手工报销",
            ),
        ],
    )

    answer = result.need_answers[0].answer_text
    assert result.need_answers[0].status == "supported"
    assert "医疗费用收据" in answer
    assert "医疗费用明细" in answer
    for marker in ("资料标题", "来源ID", "field_key", "case_relevance"):
        assert marker not in result.expert_answer


def test_material_row_does_not_claim_an_unstated_replacement_relationship() -> None:
    result = _build(
        question="缺少挂号或门诊病历时，手工报销应补充核验哪些材料？",
        needs=["缺少挂号或门诊病历时可补充核验的替代材料清单"],
        evidence=[
            _evidence(
                "ev-emergency-marker",
                "资料标题: 北京手工报销材料链字段\n"
                "field_key: emergency_context_marker\n"
                "material_name: 急诊号别/急诊诊断证明等急诊属性材料\n"
                "required_when: 申请按急诊场景处理，尤其是未直接结算时\n"
                "evidence_text: 未出示医保凭证就医的，需保留收据、处方、诊断证明等材料。",
            )
        ],
    )

    assert result.need_answers[0].status == "missing"
    assert "可以替代" not in result.expert_answer
    assert "需要提交急诊号别" not in result.expert_answer


def test_same_fact_is_not_repeated_across_information_needs() -> None:
    result = _build(
        question="缺少挂号或门诊病历时，手工报销应补充核验哪些材料？",
        needs=[
            "手工报销审核时对挂号、门诊病历等就诊材料的核验要求",
            "缺少挂号或门诊病历时可补充核验的替代材料清单",
        ],
        evidence=[
            _evidence(
                "ev-emergency-marker",
                "field_key: emergency_context_marker\n"
                "material_name: 急诊号别/急诊诊断证明等急诊属性材料\n"
                "required_when: 申请按急诊场景处理，尤其是未直接结算时\n"
                "evidence_text: 未出示医保凭证就医的，需保留收据、处方、诊断证明等材料。",
            )
        ],
    )

    claim_texts = [claim.text for claim in result.claims]
    assert len(claim_texts) == len(set(claim_texts))
    assert result.answer_markdown.count("急诊号别/急诊诊断证明") <= 1


def test_comparison_information_needs_render_as_one_integrated_answer() -> None:
    result = build_information_need_answer(
        user_question="跨省异地就医中，就医地目录与参保地待遇如何分工？",
        information_needs=[
            "跨省异地就医直接结算时，就医地目录负责哪些支付范围（药品、诊疗项目、医用耗材）",
            "跨省异地就医直接结算时，参保地待遇负责哪些内容（起付标准、支付比例、封顶线）",
        ],
        answer_mode="comparison",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical"],
            "content_type": ["policy_text"],
        },
        evidence=[
            _evidence(
                "ev-remote-scope",
                "跨省异地就医直接结算原则上执行就医地规定的支付范围及有关规定。",
                content_type="policy_text",
                policy_domain="remote_medical",
            ),
            _evidence(
                "ev-insured-benefit",
                "执行参保地规定的基本医疗保险基金起付标准、支付比例、最高支付限额等有关政策。",
                content_type="policy_text",
                policy_domain="remote_medical",
            ),
        ],
    )

    assert len(result.need_answers) == 2
    assert not result.answer_markdown.startswith("1. ")
    assert "就医地规定的支付范围" in result.answer_markdown
    assert "参保地规定的基本医疗保险基金起付标准" in result.answer_markdown
    assert "[1]" in result.answer_markdown and "[2]" in result.answer_markdown


@pytest.mark.parametrize(
    ("need", "excerpt", "domain", "expected"),
    [
        (
            "便通片的医保类别和目录编号",
            "drug_name: 便通片\ninsurance_class: 乙\ncatalog_no: 68",
            "drug_catalog",
            "医保类别为乙类",
        ),
        (
            "北京协和医院是否为定点机构，机构编码是什么",
            "institution_name: 北京协和医院\ninstitution_code: H001\ninstitution_type: 医院",
            "designated_institution",
            "机构编码为H001",
        ),
        (
            "胸部CT平扫的计价单位和收费标准",
            "service_name: 胸部CT平扫\nservice_code: CT001\nunit: 次\nprice: 200元",
            "medical_service_price",
            "收费标准为200元",
        ),
    ],
)
def test_known_table_schemas_are_parsed_by_evidence_structure(
    need: str,
    excerpt: str,
    domain: str,
    expected: str,
) -> None:
    result = build_information_need_answer(
        user_question=need,
        information_needs=[need],
        answer_mode="fact_lookup",
        filters={"jurisdiction": ["beijing"], "policy_domain": [domain]},
        evidence=[_evidence("ev-table", excerpt, policy_domain=domain)],
    )

    assert result.need_answers[0].status == "supported"
    assert expected in result.need_answers[0].answer_text


def test_unknown_table_schema_is_not_exposed_as_answer_text() -> None:
    result = _build(
        question="北京普通门诊手工报销需要哪些材料？",
        needs=["普通门诊手工报销材料清单"],
        evidence=[
            _evidence(
                "ev-unknown",
                "unknown_business_key: 内部拼接原文\nsource_id: secret-source",
            )
        ],
    )

    assert result.need_answers[0].status == "missing"
    assert "内部拼接原文" not in result.expert_answer
    assert "secret-source" not in result.expert_answer


def test_claims_and_citations_close_over_need_id_even_when_source_is_shared() -> None:
    shared_source = "source:shared"
    result = _build(
        question="北京门诊手工报销需要哪些材料，对票据有什么要求？",
        needs=["需要提交的材料清单", "票据和结算凭证要求"],
        evidence=[
            _evidence(
                "ev-receipt",
                "material_name: 医疗费用收据\nrequired_when: 普通门诊手工报销",
                source_ref=shared_source,
            ),
            _evidence(
                "ev-voucher",
                "material_name: 医疗费用结算凭证\nrequired_when: 普通门诊手工报销",
                source_ref=shared_source,
            ),
        ],
    )

    claims = {claim.claim_id: claim for claim in result.claims}
    citations = {citation.citation_id: citation for citation in result.citations}
    for answer in result.need_answers:
        assert answer.need_id in {claim.need_id for claim in result.claims}
        for citation_id in answer.citation_ids:
            citation = citations[citation_id]
            claim = claims[citation.claim_id or ""]
            assert claim.need_id == answer.need_id
            assert citation.source_refs == [shared_source]


def test_retrieval_plan_contract_rejects_legacy_answer_slots() -> None:
    with pytest.raises(PolicyFilterOutputError):
        PolicyFilterResolver().resolve_llm_plan(
            {
                "policy_question": "北京普通门诊手工报销需要哪些材料？",
                "information_needs": ["普通门诊手工报销材料清单"],
                "answer_slots": ["required_materials"],
                "filters": {
                    "jurisdiction": ["beijing"],
                    "policy_domain": ["manual_reimbursement"],
                    "content_type": ["policy_text", "table_row"],
                },
            }
        )


def test_normalized_retrieval_plan_contains_information_needs_only() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask(
        task_id="need-native-plan",
        parent_run_id="case-run",
        case_id="CASE-001",
        goal="查询普通门诊手工报销材料",
        user_question="北京普通门诊手工报销需要提交哪些材料？",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
            "content_type": ["policy_text", "table_row"],
            "force_policy_filters": True,
        },
    )

    plan = service._normalize_retrieval_plan(  # type: ignore[attr-defined]
        task,
        {
            "policy_question": task.user_question,
            "answer_mode": "list",
            "information_needs": ["普通门诊手工报销材料清单"],
            "filters": task.filters,
        },
    )

    assert plan["information_needs"] == ["普通门诊手工报销材料清单"]
    assert "answer_slots" not in plan
    assert "question_slots" not in plan


def test_hard_gate_diagnostics_route_to_information_need_resolution() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask(
        task_id="need-native-hard-gate",
        parent_run_id="case-run",
        case_id="CASE-001",
        goal="查询普通门诊手工报销材料",
        user_question="北京普通门诊手工报销需要提交哪些材料？",
    )
    state = {
        "task": task,
        "runtime": {},
        "retrieval_plan": {},
        "policy_observation": {
            "status": "ok",
            "evidence": [],
            "warnings": [],
        },
    }

    gated = service._hard_gate_node(state)  # type: ignore[attr-defined]

    assert gated["next_action"] == "need_evidence_resolution"


def test_insufficient_v2_result_keeps_one_missing_answer_per_need() -> None:
    service = ExpertAgentService()
    task = ExpertAnalysisTask(
        task_id="need-native-insufficient",
        parent_run_id="case-run",
        case_id="CASE-001",
        goal="查询普通门诊手工报销材料",
        user_question="北京普通门诊手工报销需要提交哪些材料？",
        budget={"max_llm_rewrite_calls": 0},
    )
    state = {
        "task": task,
        "runtime": {},
        "retrieval_plan": {
            "answer_mode": "list",
            "information_needs": ["普通门诊手工报销材料清单"],
            "filters": {
                "jurisdiction": ["beijing"],
                "policy_domain": ["manual_reimbursement"],
                "content_type": ["policy_text", "table_row"],
            },
        },
        "adopted_policy_evidence": [],
        "rewrite_count": 0,
        "retrieval_round_count": 1,
    }

    checked = service._answerability_check_node(state)  # type: ignore[attr-defined]
    synthesized = service._expert_synthesis_node(checked)  # type: ignore[attr-defined]

    assert synthesized["status"] == "insufficient"
    assert synthesized["result"]["pipeline_version"] == "information_need_v2"
    assert synthesized["result"]["need_answers"] == [
        {
            "need_id": synthesized["result"]["answer_requirements"][0]["need_id"],
            "need_text": "普通门诊手工报销材料清单",
            "status": "missing",
            "answer_text": "当前可引用证据不足以回答该信息点。",
            "fact_refs": [],
            "source_refs": [],
            "citation_ids": [],
            "missing_reason": "no_direct_user_safe_evidence",
        }
    ]


def test_complex_information_needs_use_one_validated_batch_synthesis_call() -> None:
    evidence = [
        _evidence(
            "ev-filing-before",
            "参保人员跨省异地就医前，应办理跨省异地就医备案。",
            content_type="policy_text",
            policy_domain="remote_medical_manual_reimbursement",
        ),
        _evidence(
            "ev-filing-after",
            "出院自费结算后按规定补办备案手续的，可以按参保地规定申请医保手工报销。",
            content_type="policy_text",
            policy_domain="remote_medical_manual_reimbursement",
        ),
    ]
    need_answer = build_information_need_answer(
        user_question="跨省异地就医备案和自费结算后的手工报销分别如何处理？",
        information_needs=[
            "跨省异地就医前的备案要求",
            "出院自费结算后补备案的手工报销处理",
        ],
        answer_mode="process_rule",
        filters={
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical_manual_reimbursement"],
            "content_type": ["policy_text"],
        },
        evidence=evidence,
    )
    assert len(need_answer.claims) == 2

    class BatchGateway:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return ModelResponse(
                content=json.dumps(
                    {
                        "answers": [
                            {
                                "need_id": need_answer.need_answers[0].need_id,
                                "answer_text": "跨省异地就医前，应先办理备案。",
                            },
                            {
                                "need_id": need_answer.need_answers[1].need_id,
                                "answer_text": "已经自费结算的，按规定补办备案后，可以向参保地申请医保手工报销。",
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    gateway = BatchGateway()
    task = ExpertAnalysisTask(
        task_id="need-native-batch-synthesis",
        parent_run_id="case-run",
        case_id="CASE-001",
        goal="查询异地备案和手工报销流程",
        user_question="跨省异地就医备案和自费结算后的手工报销分别如何处理？",
    )
    state = {
        "task": task,
        "runtime": {"model_gateway": gateway},
        "model_call_count": 0,
        "need_first_answer": need_answer.model_dump(mode="json"),
        "adopted_policy_evidence": [item.model_dump(mode="json") for item in evidence],
        "answerability_check": {
            "answerability": "answerable",
            "covered_needs": need_answer.information_needs,
            "missing_needs": [],
            "usable_evidence_refs": need_answer.source_refs,
            "next_action": "synthesize",
            "reason_code": "covered",
            "reason": "covered",
        },
    }

    result = ExpertAgentService()._need_synthesis_node(state)  # type: ignore[attr-defined]

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["thinking_enabled"] is False
    assert result["result"]["filter_diagnostics"]["post_retrieval_model_calls"] == 1
    assert result["result"]["filter_diagnostics"]["batch_synthesis"] == "applied"
    assert "应先办理备案" in result["result"]["answer_markdown"]
    assert apply_batch_synthesis(
        need_answer,
        {
            "answers": [
                {
                    "need_id": item.need_id,
                    "answer_text": "统筹基金将按99%的新增比例直接支付。",
                }
                for item in need_answer.need_answers
            ]
        },
    ) is None


def test_supported_need_rejects_uncertainty_synthesis_and_keeps_verified_answer() -> None:
    answer = _build(
        question="北京普通门诊手工报销需要哪些材料？",
        needs=["普通门诊手工报销材料清单"],
        evidence=[
            _evidence(
                "ev-receipt",
                "material_name: 医疗费用收据\nrequired_when: 普通门诊手工报销",
            )
        ],
    )

    synthesized = apply_batch_synthesis(
        answer,
        {
            "answers": [
                {
                    "need_id": answer.need_answers[0].need_id,
                    "answer_text": "提供的证据未明确说明需要哪些材料。",
                }
            ]
        },
    )

    assert synthesized is None


def test_comparison_cue_overrides_process_rule_model_label() -> None:
    assert (
        _normalize_answer_mode(
            "process_rule",
            "跨省异地就医中，就医地目录与参保地待遇如何分工？",
            ["就医地目录", "参保地待遇"],
        )
        == "comparison"
    )
