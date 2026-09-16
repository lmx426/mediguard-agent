"""业务材料视图 API 集成测试。"""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from src.backend.application.audit.review.build_materials_uc import StatisticalMaterialUseCase
from src.backend.application.audit.review.material_generation import TEMPLATE_VERSION
from src.backend.domain.audit.review.entities import CaseDetail
from src.backend.infrastructure.persistence.memory.material_repository import (
    MemoryMaterialRepository,
)


def test_business_materials_are_generated_from_safe_fields(client, generated_case_id):
    response = client.get(f"/api/cases/{generated_case_id}/statistical-materials")

    assert response.status_code == 200
    payload = response.json()
    text = str(payload)
    assert payload["case_id"] == generated_case_id
    assert payload["generation_status"] == "ready"
    assert payload["template_version"] == TEMPLATE_VERSION
    assert payload["notice"].startswith("本视图基于脱敏申报数据生成业务材料样例")
    assert payload["subject_profile"]["subject_ref"].startswith("SIM_PERSON_")
    assert [category["title"] for category in payload["categories"]] == [
        "就诊诊疗材料",
        "处方购药材料",
        "费用结算材料",
    ]
    assert [len(category["documents"]) for category in payload["categories"]] == [1, 1, 1]
    assert [document["title"] for document in payload["documents"]] == [
        "就诊诊疗记录",
        "处方购药记录",
        "费用结算记录",
    ]

    for forbidden in [
        "RES",
        "身份证",
        "姓名",
        "电话",
        "住址",
        "个人编码",
        "发生场景",
        "风险场景",
        "费用偏高",
        "多机构/多药店购药",
    ]:
        assert forbidden not in text
    for document in payload["documents"]:
        assert "仿真" not in document["title"]
        assert "模拟" not in document["title"]


def test_business_material_drug_and_fee_totals_reconcile(client, generated_case_id):
    case_payload = client.get(f"/api/cases/{generated_case_id}").json()["case"]
    source_record = case_payload["source_record"]
    materials = client.get(
        f"/api/cases/{generated_case_id}/statistical-materials"
    ).json()

    drug_doc = _document(materials, "处方购药记录")
    drug_rows = drug_doc["tables"][0]["rows"]
    expected_drug_total = _money(
        source_record.get("药品费发生金额_SUM")
        or source_record.get("药品费申报金额_SUM")
        or 0
    )
    assert _sum_rows(drug_rows, "金额") == expected_drug_total

    settlement_doc = _document(materials, "费用结算记录")
    summary_rows = settlement_doc["tables"][0]["rows"]
    summary = {row["费用类别"]: _money(row["明细合计"]) for row in summary_rows}
    assert summary["检查费"] == _money(source_record.get("检查费发生金额_SUM") or 0)
    assert summary["治疗费"] == _money(source_record.get("治疗费发生金额_SUM") or 0)
    assert summary["医用材料费"] == _money(source_record.get("医用材料发生金额_SUM") or 0)


def test_business_material_png_assets_are_controlled(client, generated_case_id):
    materials = client.get(
        f"/api/cases/{generated_case_id}/statistical-materials"
    ).json()
    drug_doc = _document(materials, "处方购药记录")
    assets = drug_doc["assets"]

    assert 1 <= len(assets) <= 5
    assert {asset["title"] for asset in assets} == {"处方笺", "药店购药小票"}
    for asset in assets:
        assert asset["preview_url"].startswith(
            f"/api/cases/{generated_case_id}/materials/{drug_doc['document_id']}/assets/"
        )
        assert "runtime" not in asset["preview_url"]
        preview = client.get(asset["preview_url"])
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("image/png")
        assert preview.headers["content-disposition"].startswith("inline")
        assert preview.content.startswith(b"\x89PNG")

        download = client.get(f"{asset['preview_url']}?download=1")
        assert download.status_code == 200
        assert download.headers["content-disposition"].startswith("attachment")


def test_business_material_snapshot_is_reused(client, generated_case_id):
    first = client.get(f"/api/cases/{generated_case_id}/statistical-materials").json()
    second = client.get(f"/api/cases/{generated_case_id}/statistical-materials").json()

    assert second["documents"] == first["documents"]
    assert second["subject_profile"] == first["subject_profile"]


def test_visitor_remote_manual_materials_include_policy_context(client):
    ingest = client.post("/api/ingest-records/SIM_PERSON_011872/push")
    assert ingest.status_code == 200
    case_id = ingest.json()["case"]["case_id"]

    materials = client.get(f"/api/cases/{case_id}/statistical-materials").json()
    text = str(materials)

    assert materials["case_context"]["insured_region"] == "北京"
    assert materials["case_context"]["treatment_region"] == "上海"
    assert materials["case_context"]["claim_mode"] == "manual_reimbursement"
    assert materials["case_context"]["direct_settlement"] is False
    assert materials["case_context"]["filing_status"] == "unknown"
    assert (
        materials["case_context"]["emergency_material_status"]
        == "present_but_needs_verification"
    )
    assert "北京" in text
    assert "上海" in text
    assert "上海市" in text
    assert "上海医保定点药房" in text or "华氏大药房" in text or "上海益丰定点药房" in text
    assert "江宁民康" not in text
    assert "秦淮同仁" not in text
    assert "建邺康宁" not in text
    assert "门诊收费票据" in text
    assert "费用明细" in text
    assert "急诊病历" in text
    for document in materials["documents"]:
        assert "案件上下文" not in document["content"]
        assert "manual_reimbursement" not in document["content"]
    assert "RES" not in text


def test_visitor_high_check_materials_reconstruct_ct_items(client):
    ingest = client.post("/api/ingest-records/SIM_PERSON_011569/push")
    assert ingest.status_code == 200
    case_id = ingest.json()["case"]["case_id"]

    case_payload = client.get(f"/api/cases/{case_id}").json()["case"]
    source_record = case_payload["source_record"]
    materials = client.get(f"/api/cases/{case_id}/statistical-materials").json()
    settlement = _document(materials, "费用结算记录")
    non_drug_rows = settlement["tables"][1]["rows"]

    assert {"胸部 CT 平扫", "图文报告", "胶片费"}.intersection(
        {row["项目名称"] for row in non_drug_rows}
    )
    assert _sum_rows(
        [row for row in non_drug_rows if row["费用类别"] == "检查费"],
        "金额",
    ) == _money(source_record.get("检查费发生金额_SUM") or 0)


def test_stale_business_material_snapshot_is_regenerated(tmp_path):
    repository = MemoryMaterialRepository()
    case = _case_detail("CASE-MATERIAL-STALE")
    repository.save_snapshot(
        case.case_id,
        {
            "case_id": case.case_id,
            "disclaimer": "",
            "notice": "",
            "generation_status": "ready",
            "template_version": "business-materials-v1.2.0",
            "documents": [
                {
                    "document_id": f"{case.case_id}-BM-OLD",
                    "title": "就诊诊疗记录",
                    "document_type": "旧材料",
                    "visit_date": "本次申报周期",
                    "institution": "旧系统",
                    "department": "旧科室",
                    "status": "已整理",
                    "content": "旧版本材料",
                    "check_points": [],
                    "category_id": "clinical",
                    "category_title": "就诊诊疗材料",
                }
            ],
            "categories": [],
        },
        [],
    )
    use_case = StatisticalMaterialUseCase(repository, Path(tmp_path))

    response = use_case.build_for_case(case)
    stored = repository.get_snapshot(case.case_id)

    assert response.template_version == TEMPLATE_VERSION
    assert [document.title for document in response.documents] == [
        "就诊诊疗记录",
        "处方购药记录",
        "费用结算记录",
    ]
    assert stored is not None
    assert stored["template_version"] == TEMPLATE_VERSION


def test_statistical_materials_unknown_case_returns_404(client):
    response = client.get("/api/cases/CASE-UNKNOWN/statistical-materials")

    assert response.status_code == 404


def _document(payload: dict[str, Any], title: str) -> dict[str, Any]:
    return next(document for document in payload["documents"] if document["title"] == title)


def _money(value: Any) -> Decimal:
    return Decimal(str(value).replace(",", "")).quantize(
        Decimal("0.01"),
        ROUND_HALF_UP,
    )


def _sum_rows(rows: list[dict[str, Any]], field: str) -> Decimal:
    return sum((_money(row[field]) for row in rows), Decimal("0.00")).quantize(
        Decimal("0.01"),
        ROUND_HALF_UP,
    )


def _case_detail(case_id: str) -> CaseDetail:
    return CaseDetail(
        case_id=case_id,
        case_title="业务材料测试案件",
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
        subject_ref="SIM_PERSON_999999",
        input_features={
            "ALL_SUM": 1000,
            "月就诊次数_MAX": 4,
            "月就诊医院数_MAX": 1,
            "一天去两家医院的天数": 0,
            "就诊的月数": 1,
            "药品在总金额中的占比": 0.5,
            "检查总费用在总金额占比": 0.3,
            "治疗费用在总金额占比": 0.2,
            "是否挂号": 1,
            "药品费发生金额_SUM": 500,
            "检查费发生金额_SUM": 300,
            "治疗费发生金额_SUM": 200,
            "医用材料发生金额_SUM": 0,
        },
    )
