"""Case-based policy RAG evaluation data.

The records in this module are hand-scoped evaluation inputs for retrieval
validation only. They are not audit decisions and do not include case OCR or
L1/L2 statistical facts.
"""

from __future__ import annotations

from typing import Any


CORE_CASE_QUERIES: list[dict[str, Any]] = [
    {
        "case_id": "case_1",
        "query_id": "case_1_core_remote_manual",
        "question": "北京参保人在上海异地急诊手工报销，备案和急诊例外应核验哪些政策？",
        "filters": {
            "jurisdiction": ["national", "beijing", "shanghai"],
            "policy_domain": [
                "remote_medical",
                "manual_reimbursement",
                "remote_medical_manual_reimbursement",
                "shanghai_payment_scope",
                "medical_service_price",
                "drug_catalog",
            ],
        },
    },
    {
        "case_id": "case_2",
        "query_id": "case_2_core_emergency_identity",
        "question": "北京参保人在上海普通门诊被申请为急诊时，急诊例外和手工报销如何核验？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": [
                "remote_medical",
                "manual_reimbursement",
                "remote_medical_manual_reimbursement",
            ],
        },
    },
    {
        "case_id": "case_3",
        "query_id": "case_3_core_benefit_mix",
        "question": "北京参保人在上海就医，为什么不能直接使用上海门诊报销比例？",
        "filters": {
            "jurisdiction": ["national", "beijing", "shanghai"],
            "policy_domain": ["remote_medical", "benefit", "shanghai_payment_scope"],
        },
    },
    {
        "case_id": "case_4",
        "query_id": "case_4_core_version_conflict",
        "question": "2026年5月北京线上备案案件如何判断当前有效政策版本？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["remote_medical"],
        },
    },
    {
        "case_id": "case_5",
        "query_id": "case_5_core_ct_charge",
        "question": "胸部CT平扫、图文报告和胶片费如何核验价格及重复收费风险？",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["medical_service_price", "fund_supervision"],
        },
    },
    {
        "case_id": "case_6",
        "query_id": "case_6_core_drug_catalog",
        "question": "阿莫西林、布洛芬和抗病毒药如何核验医保目录及限定支付范围？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["drug_catalog", "fund_supervision"],
        },
    },
    {
        "case_id": "case_7",
        "query_id": "case_7_core_material_chain",
        "question": "北京门诊购药手工报销需要哪些材料才能闭合就诊事实链？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
    },
    {
        "case_id": "case_8",
        "query_id": "case_8_core_chronic_multi_institution",
        "question": "北京慢病多机构取药如何核验长期处方、定点机构和重复购药风险？",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": [
                "chronic_disease_long_prescription",
                "special_disease_filing",
                "special_disease_scope",
                "designated_institution",
                "fund_supervision",
            ],
        },
    },
]


QUALITY_EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "case_id": "case_1",
        "query_id": "case_1_remote_principle",
        "question": "跨省异地就医中，就医地目录与参保地待遇如何分工？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["remote_medical"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "remote_principle",
                "match_any": [
                    "policy_national_remote_settlement_20220726",
                    "policy_beijing_remote_medical_interpretation_20240619_round3_d1_89da2a401988",
                ],
            }
        ],
    },
    {
        "case_id": "case_1",
        "query_id": "case_1_emergency_manual",
        "question": "北京参保人异地急诊、备案和手工报销如何处理？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": [
                "remote_medical",
                "manual_reimbursement",
                "remote_medical_manual_reimbursement",
            ],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "emergency_remote",
                "match_any": [
                    "faq_beijing_remote_emergency_observation_reimbursement_20240524",
                    "policy_national_remote_settlement_20220726",
                ],
            },
            {
                "group": "manual_reimbursement",
                "match_any": [
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            },
        ],
    },
    {
        "case_id": "case_1",
        "query_id": "case_1_shanghai_payment_scope",
        "question": "上海药品、胸部CT和医用耗材支付范围如何核验？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["shanghai_payment_scope", "medical_service_price"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "shanghai_drug_or_consumable",
                "match_any": [
                    "policy_shanghai_drug_catalog_2024",
                    "medicalrag_policy_shanghai_consumable_payment_scope_20220518_pdf",
                ],
            },
            {
                "group": "shanghai_ct",
                "match_any": ["policy_shanghai_ct_mri_price_norm"],
            },
        ],
    },
    {
        "case_id": "case_2",
        "query_id": "case_2_emergency_vs_outpatient",
        "question": "异地急诊抢救与普通门诊如何区分和适用备案规则？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["remote_medical", "remote_medical_manual_reimbursement"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "emergency_rule",
                "match_any": [
                    "faq_beijing_remote_emergency_observation_reimbursement_20240524",
                    "policy_national_remote_settlement_20220726",
                ],
            }
        ],
    },
    {
        "case_id": "case_2",
        "query_id": "case_2_emergency_material_gap",
        "question": "异地急诊材料不足时，手工报销路径应如何核验？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": [
                "manual_reimbursement",
                "remote_medical_manual_reimbursement",
            ],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "manual_or_emergency_material",
                "match_any": [
                    "faq_beijing_remote_emergency_observation_reimbursement_20240524",
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            }
        ],
    },
    {
        "case_id": "case_2",
        "query_id": "case_2_ordinary_manual_materials",
        "question": "普通异地门诊手工报销需要哪些材料？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "manual_materials",
                "match_any": [
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            }
        ],
    },
    {
        "case_id": "case_3",
        "query_id": "case_3_no_shanghai_ratio",
        "question": "北京参保人在上海门诊就医，为什么不能直接使用上海报销比例？",
        "filters": {
            "jurisdiction": ["national", "beijing", "shanghai"],
            "policy_domain": ["remote_medical", "benefit", "shanghai_payment_scope"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "remote_principle",
                "match_any": [
                    "policy_national_remote_settlement_20220726",
                    "policy_beijing_remote_medical_interpretation_20240619_round3_d1_89da2a401988",
                ],
            }
        ],
    },
    {
        "case_id": "case_3",
        "query_id": "case_3_beijing_retiree_benefit",
        "question": "北京退休职工门急诊待遇应核验哪些参数？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["benefit"],
        },
        "expected_status": "partial_supported",
        "known_limitations": [
            "Current corpus has a brief Beijing employee/retiree benefit guide, but not enough detail for exact amount calculation."
        ],
        "expected_source_groups": [
            {
                "group": "beijing_employee_benefit_overview",
                "match_any": ["catalog_beijing_employee_medical_benefit_20230101"],
            }
        ],
    },
    {
        "case_id": "case_3",
        "query_id": "case_3_shanghai_scope",
        "question": "上海药品和诊疗项目支付范围如何核验？",
        "filters": {
            "jurisdiction": ["shanghai"],
            "policy_domain": ["shanghai_payment_scope", "medical_service_price"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "shanghai_payment_scope",
                "match_any": [
                    "policy_shanghai_drug_catalog_2024",
                    "policy_shanghai_ct_mri_price_norm",
                    "medicalrag_policy_shanghai_consumable_payment_scope_20220518_pdf",
                ],
            }
        ],
    },
    {
        "case_id": "case_4",
        "query_id": "case_4_active_policy_by_service_date",
        "question": "2026年5月上海异地门诊案件应使用哪一版北京备案政策？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical"],
            "status": ["active"],
            "valid_on": "2026-05-01",
        },
        "expected_status": "coverage_gap",
        "coverage_gap_reason": "Version metadata exists locally but is not joined into MCP filters/index yet.",
        "expected_source_groups": [
            {
                "group": "version_metadata",
                "match_any": ["rag_metadata_policy_version_review_index"],
            }
        ],
    },
    {
        "case_id": "case_4",
        "query_id": "case_4_replaced_policy",
        "question": "北京异地备案旧政策是否已经废止或被新政策替代？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["remote_medical"],
            "status": ["active", "expired", "replaced"],
        },
        "expected_status": "coverage_gap",
        "coverage_gap_reason": "MCP v1 does not support version status filtering.",
        "expected_source_groups": [
            {
                "group": "version_metadata",
                "match_any": ["rag_metadata_policy_version_review_index"],
            }
        ],
    },
    {
        "case_id": "case_4",
        "query_id": "case_4_valid_on_filter",
        "question": "政策RAG如何按服务日期筛选有效政策？",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["remote_medical"],
            "valid_on": "2026-05-01",
        },
        "expected_status": "coverage_gap",
        "coverage_gap_reason": "valid_on metadata filtering has not been implemented in MCP v1.",
        "expected_source_groups": [
            {
                "group": "version_metadata",
                "match_any": ["rag_metadata_policy_version_review_index"],
            }
        ],
    },
    {
        "case_id": "case_5",
        "query_id": "case_5_beijing_ct_price",
        "question": "北京胸部CT平扫价格和计价单位如何核验？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["medical_service_price"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "beijing_medical_service_price",
                "match_any": ["catalog_beijing_medical_service_price_with_insurance_20240326"],
            }
        ],
    },
    {
        "case_id": "case_5",
        "query_id": "case_5_ct_report_film_charge",
        "question": "CT平扫、图文报告和胶片费能否分别收费，应查什么价格依据？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["medical_service_price"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "beijing_medical_service_price",
                "match_any": ["catalog_beijing_medical_service_price_with_insurance_20240326"],
            }
        ],
    },
    {
        "case_id": "case_5",
        "query_id": "case_5_fund_supervision_charge",
        "question": "医保基金监管中重复收费、分解收费、超标准收费如何识别？",
        "filters": {
            "jurisdiction": ["national"],
            "policy_domain": ["fund_supervision"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "fund_supervision",
                "match_any": ["policy_national_fund_supervision_rules_20260213"],
            }
        ],
    },
    {
        "case_id": "case_6",
        "query_id": "case_6_drug_catalog_scope",
        "question": "阿莫西林、布洛芬等是否在国家或北京医保药品目录中？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["drug_catalog"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "drug_catalog",
                "match_any": [
                    "policy_national_drug_catalog_2025_with_attachments",
                    "policy_beijing_drug_catalog_execution_20251230",
                ],
            }
        ],
    },
    {
        "case_id": "case_6",
        "query_id": "case_6_limited_payment",
        "question": "医保药品限定支付范围和诊断用药匹配如何核验？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["drug_catalog", "fund_supervision"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "drug_catalog_or_supervision",
                "match_any": [
                    "policy_national_drug_catalog_2025_with_attachments",
                    "policy_beijing_drug_catalog_execution_20251230",
                    "policy_national_fund_supervision_rules_20260213",
                ],
            }
        ],
    },
    {
        "case_id": "case_6",
        "query_id": "case_6_drug_ratio_not_violation",
        "question": "药品费占比高是否足以直接认定医保不合规？",
        "filters": {
            "jurisdiction": ["national", "beijing"],
            "policy_domain": ["drug_catalog", "fund_supervision"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "drug_or_supervision_evidence",
                "match_any": [
                    "policy_national_drug_catalog_2025_with_attachments",
                    "policy_national_fund_supervision_rules_20260213",
                ],
            }
        ],
    },
    {
        "case_id": "case_7",
        "query_id": "case_7_manual_materials",
        "question": "北京门诊手工报销需要提交哪些材料？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "manual_materials",
                "match_any": [
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            }
        ],
    },
    {
        "case_id": "case_7",
        "query_id": "case_7_material_chain",
        "question": "票据、处方、费用明细和门诊病历如何形成手工报销材料链？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "manual_material_chain",
                "match_any": [
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            }
        ],
    },
    {
        "case_id": "case_7",
        "query_id": "case_7_missing_registration",
        "question": "缺少挂号或门诊病历时，手工报销应补充核验哪些材料？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["manual_reimbursement"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "manual_material_gap",
                "match_any": [
                    "policy_beijing_medical_expense_audit_settlement_opinion_20250605",
                    "guide_beijing_manual_reimbursement_materials",
                ],
            }
        ],
    },
    {
        "case_id": "case_8",
        "query_id": "case_8_long_prescription",
        "question": "北京慢病长期处方和续方如何核验？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["chronic_disease_long_prescription"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "long_prescription",
                "match_any": ["policy_beijing_chronic_disease_long_prescription_20240620"],
            }
        ],
    },
    {
        "case_id": "case_8",
        "query_id": "case_8_special_disease",
        "question": "北京门诊慢特病备案和病种范围如何核验？",
        "filters": {
            "jurisdiction": ["beijing"],
            "policy_domain": ["special_disease_filing", "special_disease_scope"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "special_disease",
                "match_any": [
                    "policy_beijing_special_disease_filing_20260106",
                    "policy_beijing_special_disease_scope_20240614",
                    "faq_beijing_special_disease_filing_20250624",
                ],
            }
        ],
    },
    {
        "case_id": "case_8",
        "query_id": "case_8_designated_duplicate",
        "question": "北京慢病多机构取药时，定点机构、重复购药和异常开药如何核验？",
        "filters": {
            "jurisdiction": ["beijing", "national"],
            "policy_domain": ["designated_institution", "fund_supervision"],
        },
        "expected_status": "supported",
        "expected_source_groups": [
            {
                "group": "designated_or_supervision",
                "match_any": [
                    "catalog_beijing_designated_medical_institutions_iframe",
                    "catalog_beijing_designated_pharmacies_iframe",
                    "policy_national_fund_supervision_rules_20260213",
                ],
            }
        ],
    },
]
