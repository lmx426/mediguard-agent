from __future__ import annotations

import math

from src.backend.scripts.evaluate_case_agent_ragas_generation import (
    build_ragas_input,
    normalize_formula_judge,
)
from src.backend.scripts.evaluate_case_agent_ragas_comparison import (
    native_reference,
    summarize_comparison,
)
from src.backend.scripts.policy_rag_eval.upgrade_eval_set_v2_required_optional import (
    upgrade_row,
)


def test_required_optional_migration_demotes_unasked_process_details() -> None:
    row = {
        "schema_version": "policy_rag_eval_v2",
        "query_id": "v2_000002",
        "question": "审核参保人零星报销门诊费用时，应要求提供哪些必要材料？法定办结时限是多少？",
        "reference_answer": "材料、流程和时限完整答案。",
        "question_type": "single_hop",
        "bundle_type": "single_clause",
        "answer_points": [
            {
                "point_id": "p1",
                "statement": "门诊零星报销的必要材料包括身份证、社保卡、医疗费专用收据、病史资料和银行卡。",
            },
            {
                "point_id": "p2",
                "statement": "医保卡、疾病诊断证明书、死亡证明等属于非必要材料。",
            },
            {
                "point_id": "p3",
                "statement": "审查和决定环节时限各为15个工作日，法定办结时限为30个工作日。",
            },
            {"point_id": "p4", "statement": "办理该事项需到现场1次。"},
        ],
        "gold_evidence_groups": [
            {"group_id": "g1", "answer_point_ids": ["p1"]},
            {"group_id": "g2", "answer_point_ids": ["p2"]},
            {"group_id": "g3", "answer_point_ids": ["p3"]},
            {"group_id": "g4", "answer_point_ids": ["p4"]},
        ],
    }

    upgraded, _audit = upgrade_row(row)

    required = [point["statement"] for point in upgraded["required_answer_points"]]
    optional = [point["statement"] for point in upgraded["optional_answer_points"]]
    assert "门诊零星报销的必要材料包括身份证、社保卡、医疗费专用收据、病史资料和银行卡" in required
    assert "法定办结时限为30个工作日" in required
    assert "医保卡、疾病诊断证明书、死亡证明等属于非必要材料" in optional
    assert "审查和决定环节时限各为15个工作日" in optional
    assert "办理该事项需到现场1次" in optional
    assert "审查" not in upgraded["reference_answer_required"]
    assert "到现场" not in upgraded["reference_answer_required"]


def test_required_optional_migration_demotes_unasked_table_field() -> None:
    row = {
        "schema_version": "policy_rag_eval_v2",
        "query_id": "v2_table",
        "question": "处方中的便通片在上海医保药品目录中的医保类别和本地支付比例是多少？",
        "reference_answer": "便通片完整目录信息。",
        "question_type": "table_lookup",
        "bundle_type": "table_lookup",
        "answer_points": [
            {"point_id": "p1", "statement": "便通片（胶囊）的医保类别为乙类"},
            {"point_id": "p2", "statement": "便通片（胶囊）的目录编号为68"},
            {"point_id": "p3", "statement": "便通片（胶囊）的本地支付比例为10%"},
        ],
        "gold_evidence_groups": [
            {"group_id": "g1", "answer_point_ids": ["p1"]},
            {"group_id": "g2", "answer_point_ids": ["p2"]},
            {"group_id": "g3", "answer_point_ids": ["p3"]},
        ],
    }

    upgraded, _audit = upgrade_row(row)

    required = [point["statement"] for point in upgraded["required_answer_points"]]
    optional = [point["statement"] for point in upgraded["optional_answer_points"]]
    assert "便通片（胶囊）的医保类别为乙类" in required
    assert "便通片（胶囊）的本地支付比例为10%" in required
    assert "便通片（胶囊）的目录编号为68" in optional


def test_required_optional_migration_keeps_asked_alias_fields_required() -> None:
    rows = [
        {
            "question": "如果费用涉及“双通道”药品，需要怎样把握政策要点？审批时限是多久？",
            "question_type": "single_hop",
            "bundle_type": "single_clause",
            "answer_points": [
                {"point_id": "p1", "statement": "双通道药品名单可在上海阳光医药采购网查看。"},
                {
                    "point_id": "p2",
                    "statement": "门诊零星报销的法定及承诺办结时限均为30个工作日，按收到材料齐全、系统受理后起算。",
                },
            ],
        },
        {
            "question": "蛭蛇通络胶囊的报销类别和医保支付标准是多少？",
            "question_type": "table_lookup",
            "bundle_type": "table_lookup",
            "answer_points": [
                {"point_id": "p1", "statement": "蛭蛇通络胶囊属于乙类药品。"},
                {"point_id": "p2", "statement": "蛭蛇通络胶囊的医保支付标准为1.65元(0.5g/粒)。"},
            ],
        },
        {
            "question": "异福口服常释剂型的医保类别和个人自负比例分别是多少？",
            "question_type": "table_lookup",
            "bundle_type": "table_lookup",
            "answer_points": [
                {"point_id": "p1", "statement": "异福口服常释剂型属于乙类药品。"},
                {"point_id": "p2", "statement": "异福口服常释剂型的个人自负比例为10%。"},
            ],
        },
        {
            "question": "它是否在北京市定点零售药店名单中？位于哪个区？",
            "question_type": "table_lookup",
            "bundle_type": "table_lookup",
            "answer_points": [
                {"point_id": "p1", "statement": "该药店是北京市定点零售药店。"},
                {"point_id": "p2", "statement": "该药店位于北京市房山区。"},
            ],
        },
        {
            "question": "请问该药的分类及医保类别是什么？",
            "question_type": "table_lookup",
            "bundle_type": "table_lookup",
            "answer_points": [
                {"point_id": "p1", "statement": "硝呋太尔阴道片的分类为妇科抗感染药和抗菌剂。"},
                {"point_id": "p2", "statement": "硝呋太尔阴道片的医保类别为乙类，目录编号为★(564)。"},
            ],
        },
    ]

    for row in rows:
        row.update(
            {
                "schema_version": "policy_rag_eval_v2",
                "query_id": "alias",
                "reference_answer": "完整答案。",
                "gold_evidence_groups": [],
            }
        )
        upgraded, _audit = upgrade_row(row)
        assert not upgraded["optional_answer_points"]
        assert len(upgraded["required_answer_points"]) == len(row["answer_points"])


def test_ragas_input_uses_required_reference_when_present() -> None:
    row = {
        "query_id": "v2_required",
        "question": "办结时限是多少？",
        "reference_answer": "完整答案。",
        "reference_answer_required": "法定办结时限为30个工作日。",
        "reference_answer_full": "完整答案。",
        "required_answer_points": [{"statement": "法定办结时限为30个工作日"}],
        "optional_answer_points": [{"statement": "到现场次数为1次"}],
    }

    payload = build_ragas_input(
        row,
        response_text="法定办结时限为30个工作日。",
        contexts=[{"text": "法定办结时限 30(工作日)"}],
    )

    assert payload["reference"] == "法定办结时限为30个工作日。"
    assert payload["required_answer_points"] == ["法定办结时限为30个工作日"]
    assert payload["optional_answer_points"] == ["到现场次数为1次"]


def test_formula_judge_counts_only_required_points_for_fn() -> None:
    payload = {
        "answer_correctness": {
            "response_statements": ["法定办结时限为30个工作日"],
            "required_points": [
                {"statement": "法定办结时限为30个工作日", "covered": True}
            ],
            "optional_points": [
                {"statement": "到现场次数为1次", "covered": False}
            ],
            "FP": [],
        },
        "faithfulness": {
            "statements": [
                {
                    "statement": "法定办结时限为30个工作日",
                    "supported": True,
                    "supporting_context_ranks": [1],
                }
            ]
        },
    }

    metrics = normalize_formula_judge(
        payload,
        model="deepseek-test",
        required_answer_points=["法定办结时限为30个工作日"],
        optional_answer_points=["到现场次数为1次"],
    )

    assert metrics["answer_correctness_counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert metrics["fp_count"] == 0
    assert metrics["fn_required_count"] == 0
    assert metrics["required_point_recall"] == 1.0
    assert metrics["optional_point_coverage"] == 0.0
    assert math.isclose(metrics["answer_correctness"], 1.0)


def test_native_ragas_comparison_reference_defaults_to_required_answer() -> None:
    row = {
        "reference_answer": "完整参考答案。",
        "reference_answer_full": "完整参考答案，含可选背景。",
        "reference_answer_required": "只包含必答事实。",
    }

    assert native_reference(row, "required") == "只包含必答事实。"
    assert native_reference(row, "full") == "完整参考答案，含可选背景。"
    assert native_reference(row, "reference") == "完整参考答案。"


def test_native_ragas_comparison_summary_keeps_metric_families_separate() -> None:
    summary = summarize_comparison(
        [
            {
                "run_status": "completed",
                "retrieved_context_count": 5,
                "question_type": "single_hop",
                "bundle_type": "single_clause",
                "metrics": {
                    "business_required_points": {
                        "evaluable": True,
                        "metric_backend": "deepseek_formula",
                        "answer_correctness": 1.0,
                        "faithfulness": 1.0,
                        "required_point_recall": 1.0,
                        "optional_point_coverage": 0.0,
                        "fp_count": 0,
                        "fn_required_count": 0,
                    },
                    "ragas_claim_flow": {
                        "evaluable": True,
                        "metric_backend": "deepseek_formula",
                        "answer_correctness": 0.9,
                        "faithfulness": 1.0,
                        "fp_count": 1,
                        "fn_required_count": 0,
                    },
                    "native_ragas": {
                        "evaluable": True,
                        "metric_backend": "ragas",
                        "answer_correctness": 0.8,
                        "faithfulness": 1.0,
                    },
                },
            }
        ]
    )

    assert summary["business_required_points"]["metric_backend_counts"] == {
        "deepseek_formula": 1
    }
    assert summary["business_required_points"]["answer_correctness_avg"] == 1.0
    assert summary["ragas_claim_flow"]["metric_backend_counts"] == {
        "deepseek_formula": 1
    }
    assert summary["ragas_claim_flow"]["answer_correctness_avg"] == 0.9
    assert summary["native_ragas"]["metric_backend_counts"] == {"ragas": 1}
    assert summary["native_ragas"]["answer_correctness_avg"] == 0.8
