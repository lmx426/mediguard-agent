"""Deterministic business material generation for the fact-base view."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ....domain.audit.review.entities import (
    BusinessMaterialAsset,
    BusinessMaterialCategory,
    BusinessMaterialTable,
    CaseDetail,
    MaterialAssetLocation,
    MaterialSubjectProfile,
    MedicalRecordDocument,
    MedicalRecordResponse,
)

NOTICE = "本视图基于脱敏申报数据生成业务材料样例，用于辅助查看事实线索，不代表真实个人身份、原始病历或票据。"
TEMPLATE_VERSION = "business-materials-v1.4.3"
MONEY = Decimal("0.01")
PROJECT_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class Scenario:
    code: str
    label: str


@dataclass(frozen=True)
class DrugItem:
    name: str
    spec: str
    usage: str


@dataclass
class MaterialGenerationResult:
    response: MedicalRecordResponse
    assets: list[MaterialAssetLocation]
    metadata: dict[str, Any]


SCENARIOS = {
    "upper_respiratory": Scenario("upper_respiratory", "上呼吸道感染短期购药"),
    "chronic_medication": Scenario("chronic_medication", "慢病长期购药"),
    "multi_org_purchase": Scenario("multi_org_purchase", "就医购药记录组合"),
    "high_check_fee": Scenario("high_check_fee", "检查费用记录组合"),
    "high_treatment_fee": Scenario("high_treatment_fee", "治疗费用记录组合"),
    "material_gap": Scenario("material_gap", "材料记录待补充"),
    "routine_outpatient": Scenario("routine_outpatient", "常规门诊购药记录"),
    "routine_outpatient_complete": Scenario("routine_outpatient_complete", "普通门诊材料齐全"),
    "remote_emergency_manual_complete": Scenario("remote_emergency_manual_complete", "异地门急诊手工报销材料齐全"),
    "remote_emergency_manual_unknown_filing": Scenario("remote_emergency_manual_unknown_filing", "异地门急诊手工报销备案待核验"),
    "remote_emergency_identity_gap": Scenario("remote_emergency_identity_gap", "急诊身份材料待核验"),
    "remote_benefit_catalog_mix": Scenario("remote_benefit_catalog_mix", "参保地待遇与就医地目录混用核验"),
    "policy_version_conflict": Scenario("policy_version_conflict", "政策版本冲突核验"),
    "high_check_fee_ct": Scenario("high_check_fee_ct", "CT相关检查费占比核验"),
    "high_drug_ratio_uri": Scenario("high_drug_ratio_uri", "药品费占比与诊断匹配核验"),
    "registration_gap_purchase": Scenario("registration_gap_purchase", "挂号状态与门诊购药链条核验"),
    "high_frequency_multi_org_chronic": Scenario("high_frequency_multi_org_chronic", "高频多机构慢病取药核验"),
}

DRUG_CATALOG: dict[str, list[DrugItem]] = {
    "upper_respiratory": [
        DrugItem("连花清瘟胶囊", "0.35g*24粒/盒", "口服，一次4粒，一日3次"),
        DrugItem("复方氨酚烷胺胶囊", "12粒/盒", "口服，一次1粒，一日2次"),
        DrugItem("布洛芬缓释胶囊", "0.3g*20粒/盒", "口服，一次1粒，一日2次"),
        DrugItem("头孢克洛缓释片", "0.375g*6片/盒", "口服，一次1片，一日2次"),
    ],
    "chronic_medication": [
        DrugItem("苯磺酸氨氯地平片", "5mg*28片/盒", "口服，一次1片，一日1次"),
        DrugItem("缬沙坦胶囊", "80mg*7粒/盒", "口服，一次1粒，一日1次"),
        DrugItem("二甲双胍片", "0.5g*20片/盒", "口服，一次1片，一日2次"),
        DrugItem("阿卡波糖片", "50mg*30片/盒", "随餐口服，一次1片，一日3次"),
        DrugItem("阿托伐他汀钙片", "20mg*7片/盒", "口服，每晚1片"),
        DrugItem("甘精胰岛素注射液", "3ml:300单位/支", "皮下注射，遵医嘱"),
        DrugItem("恩格列净片", "10mg*10片/盒", "口服，一次1片，一日1次"),
    ],
    "multi_org_purchase": [
        DrugItem("苯磺酸氨氯地平片", "5mg*28片/盒", "口服，一次1片，一日1次"),
        DrugItem("甘精胰岛素注射液", "3ml:300单位/支", "皮下注射，遵医嘱"),
        DrugItem("恩格列净片", "10mg*10片/盒", "口服，一次1片，一日1次"),
        DrugItem("阿托伐他汀钙片", "20mg*7片/盒", "口服，每晚1片"),
        DrugItem("复方丹参滴丸", "27mg*180丸/瓶", "口服，一次10丸，一日3次"),
    ],
    "high_check_fee": [
        DrugItem("盐酸氨溴索口服溶液", "100ml/瓶", "口服，一次10ml，一日3次"),
        DrugItem("孟鲁司特钠片", "10mg*5片/盒", "口服，每晚1片"),
        DrugItem("复方甲氧那明胶囊", "60粒/瓶", "口服，一次2粒，一日3次"),
    ],
    "high_treatment_fee": [
        DrugItem("塞来昔布胶囊", "0.2g*6粒/盒", "口服，一次1粒，一日1次"),
        DrugItem("甲钴胺片", "0.5mg*20片/盒", "口服，一次1片，一日3次"),
        DrugItem("双氯芬酸钠缓释片", "75mg*10片/盒", "口服，一次1片，一日1次"),
    ],
    "material_gap": [
        DrugItem("连花清瘟胶囊", "0.35g*24粒/盒", "口服，一次4粒，一日3次"),
        DrugItem("布洛芬缓释胶囊", "0.3g*20粒/盒", "口服，一次1粒，一日2次"),
    ],
    "routine_outpatient": [
        DrugItem("奥美拉唑肠溶胶囊", "20mg*14粒/盒", "口服，一次1粒，一日1次"),
        DrugItem("蒙脱石散", "3g*10袋/盒", "口服，一次1袋，一日3次"),
        DrugItem("氯雷他定片", "10mg*6片/盒", "口服，一次1片，一日1次"),
    ],
}
URI_DRUG_ITEMS = [
    DrugItem("阿莫西林胶囊", "0.25g*24粒/盒", "口服，一次2粒，一日3次"),
    DrugItem("布洛芬缓释胶囊", "0.3g*20粒/盒", "口服，一次1粒，一日2次"),
    DrugItem("抗病毒口服液", "10ml*10支/盒", "口服，一次10ml，一日3次"),
    DrugItem("复方甘草口服溶液", "100ml/瓶", "口服，一次10ml，一日3次"),
]
for _scenario_code in (
    "remote_emergency_manual_complete",
    "remote_emergency_manual_unknown_filing",
    "remote_emergency_identity_gap",
    "remote_benefit_catalog_mix",
    "policy_version_conflict",
    "high_drug_ratio_uri",
    "registration_gap_purchase",
):
    DRUG_CATALOG[_scenario_code] = list(URI_DRUG_ITEMS)
DRUG_CATALOG["routine_outpatient_complete"] = list(DRUG_CATALOG["routine_outpatient"])
DRUG_CATALOG["high_check_fee_ct"] = [
    DrugItem("布洛芬缓释胶囊", "0.3g*20粒/盒", "口服，一次1粒，一日2次"),
    DrugItem("氯雷他定片", "10mg*6片/盒", "口服，一次1片，一日1次"),
    DrugItem("止咳糖浆", "100ml/瓶", "口服，一次10ml，一日3次"),
]
DRUG_CATALOG["high_frequency_multi_org_chronic"] = list(DRUG_CATALOG["chronic_medication"])

CHECK_ITEMS = ["血常规", "C反应蛋白测定", "尿常规", "心电图", "胸部CT平扫", "肝肾功能"]
CT_CHECK_ITEMS = ["胸部 CT 平扫", "图文报告", "胶片费"]
TREATMENT_ITEMS = ["雾化吸入治疗", "静脉输液观察", "中医定向透药", "红外线治疗", "换药处置"]
MATERIAL_ITEMS = ["一次性输液器", "一次性注射器", "医用敷贴", "雾化面罩", "留置针"]
REGIONAL_INSTITUTIONS: dict[str, list[str]] = {
    "北京": ["北京市朝阳区社区卫生服务中心", "北京协和医院门诊部", "北京市东城区东华门社区卫生服务中心"],
    "上海": ["上海市静安区中心医院", "上海交通大学医学院附属瑞金医院门急诊部", "上海市徐汇区中心医院"],
}
REGIONAL_PHARMACIES: dict[str, list[str]] = {
    "北京": ["北京医保定点药店朝阳门店", "同仁堂定点药店东城店", "北京康民定点药房海淀店"],
    "上海": ["上海医保定点药房静安店", "华氏大药房瑞金医院店", "上海益丰定点药房徐汇店"],
}
DEFAULT_INSTITUTION = REGIONAL_INSTITUTIONS["北京"][0]


class BusinessMaterialGenerator:
    """Build business-looking material samples from safe aggregate fields."""

    def __init__(self, asset_root: Path, api_prefix: str = "/api") -> None:
        self._asset_root = asset_root
        self._api_prefix = api_prefix.rstrip("/")

    def generate(self, case: CaseDetail) -> MaterialGenerationResult:
        record: dict[str, Any] = dict(case.source_record or case.input_features)
        context: dict[str, Any] = dict(case.case_context or {})
        fingerprint = self._fingerprint(case, record)
        rng = random.Random(int(fingerprint[:16], 16))
        scenario = self._select_scenario(record, context)
        subject_profile = self._subject_profile(case, record, scenario, rng)
        service_dates = self._service_dates(record, rng, context)
        institutions, pharmacies = self._regional_providers(context)

        drug_amount = self._drug_amount(record)
        check_amount = self._amount(record, "检查费发生金额_SUM")
        treatment_amount = self._amount(record, "治疗费发生金额_SUM")
        material_amount = self._amount(record, "医用材料发生金额_SUM")

        drug_rows = self._drug_rows(
            scenario=scenario,
            total=drug_amount,
            service_dates=service_dates,
            institutions=institutions,
            pharmacies=pharmacies,
            rng=rng,
        )
        visit_rows = self._visit_rows(
            record=record,
            scenario=scenario,
            service_dates=service_dates,
            drug_rows=drug_rows,
            institutions=institutions,
            pharmacies=pharmacies,
            rng=rng,
        )
        non_drug_tables = self._non_drug_tables(
            scenario=scenario,
            check_amount=check_amount,
            treatment_amount=treatment_amount,
            material_amount=material_amount,
            service_dates=service_dates,
            institutions=institutions,
            rng=rng,
        )

        drug_document_id = f"{case.case_id}-BM-03"
        asset_entries, locations = self._generate_png_assets(
            case=case,
            subject_profile=subject_profile,
            scenario=scenario,
            service_dates=service_dates,
            drug_rows=drug_rows,
            institutions=institutions,
            pharmacies=pharmacies,
            material_id=drug_document_id,
        )

        documents = [
            self._clinical_record_document(
                case=case,
                scenario=scenario,
                subject_profile=subject_profile,
                record=record,
                context=context,
                service_dates=service_dates,
                visit_rows=visit_rows,
            ),
            self._prescription_purchase_document(
                case=case,
                scenario=scenario,
                context=context,
                service_dates=service_dates,
                drug_rows=drug_rows,
                assets=asset_entries,
                document_id=drug_document_id,
            ),
            self._settlement_record_document(
                case=case,
                record=record,
                service_dates=service_dates,
                drug_amount=drug_amount,
                check_amount=check_amount,
                treatment_amount=treatment_amount,
                material_amount=material_amount,
                non_drug_tables=non_drug_tables,
                context=context,
                assets=asset_entries,
            ),
        ]
        categories = self._categories(documents)
        response = MedicalRecordResponse(
            case_id=case.case_id,
            disclaimer=NOTICE,
            notice=NOTICE,
            generation_status="ready",
            template_version=TEMPLATE_VERSION,
            subject_profile=subject_profile,
            case_context=self._display_case_context(context),
            categories=categories,
            documents=documents,
        )
        metadata = {
            "input_fingerprint": fingerprint,
            "template_version": TEMPLATE_VERSION,
            "material_scenario": scenario.code,
            "case_context_keys": sorted(context.keys()),
        }
        return MaterialGenerationResult(response=response, assets=locations, metadata=metadata)

    def _select_scenario(self, record: dict[str, Any], context: dict[str, Any] | None = None) -> Scenario:
        forced = ""
        if isinstance(context, dict):
            forced = str(context.get("material_scenario") or "").strip()
        if forced in SCENARIOS:
            return SCENARIOS[forced]
        missing_keys = [
            "ALL_SUM",
            "药品费发生金额_SUM",
            "检查费发生金额_SUM",
            "治疗费发生金额_SUM",
            "月就诊次数_MAX",
            "是否挂号",
        ]
        if sum(1 for key in missing_keys if self._raw_missing(record.get(key))) >= 2:
            return SCENARIOS["material_gap"]
        cross_days = self._float(record, "一天去两家医院的天数")
        hospital_count = self._float(record, "月就诊医院数_MAX")
        if cross_days >= 1 or hospital_count >= 3:
            return SCENARIOS["multi_org_purchase"]
        check_ratio = self._float(record, "检查总费用在总金额占比")
        if check_ratio >= 0.35:
            return SCENARIOS["high_check_fee"]
        treatment_ratio = self._float(record, "治疗费用在总金额占比")
        if treatment_ratio >= 0.35:
            return SCENARIOS["high_treatment_fee"]
        visit_count = self._float(record, "月就诊次数_MAX")
        drug_amount = self._drug_amount(record)
        if visit_count >= 8 or drug_amount >= Decimal("800.00"):
            return SCENARIOS["chronic_medication"]
        drug_ratio = self._float(record, "药品在总金额中的占比")
        if drug_ratio >= 0.45:
            return SCENARIOS["upper_respiratory"]
        return SCENARIOS["routine_outpatient"]

    def _subject_profile(
        self,
        case: CaseDetail,
        record: dict[str, Any],
        scenario: Scenario,
        rng: random.Random,
    ) -> MaterialSubjectProfile:
        subject_ref = case.subject_ref or f"SIM_SUBJECT_{self._fingerprint(case, record)[:8].upper()}"
        chronic_tags: list[str] = []
        if scenario.code in {"chronic_medication", "high_frequency_multi_org_chronic"}:
            age_group = rng.choice(["56-65岁", "66-75岁", "76-85岁"])
            chronic_tags = rng.sample(["高血压", "2型糖尿病", "血脂异常"], k=2)
        elif scenario.code in {"high_treatment_fee", "high_check_fee", "high_check_fee_ct"}:
            age_group = rng.choice(["36-45岁", "46-55岁", "56-65岁"])
        else:
            age_group = rng.choice(["18-35岁", "36-45岁", "46-55岁", "56-65岁"])
        return MaterialSubjectProfile(
            subject_ref=subject_ref,
            gender=rng.choice(["男", "女"]),
            age_group=age_group,
            insurance_type=rng.choice(["职工基本医疗保险", "城乡居民基本医疗保险"]),
            patient_group_tags=[],
            chronic_condition_tags=chronic_tags,
            allergy_history=rng.choice(["未见记录", "青霉素过敏史", "磺胺类药物过敏史"]),
            primary_visit_type=case.case_type or "门诊",
            registration_status=self._registration_label(record.get("是否挂号")),
        )

    def _service_dates(
        self,
        record: dict[str, Any],
        rng: random.Random,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        visit_date = str((context or {}).get("visit_date") or "").strip()
        if visit_date:
            try:
                base = datetime.fromisoformat(visit_date[:10]).replace(
                    hour=9,
                    minute=30,
                    second=0,
                    microsecond=0,
                )
                offsets = [0, 0, 0, 1, 1, 2]
                return [
                    (base + timedelta(hours=offset + index)).strftime("%Y-%m-%d %H:%M")
                    for index, offset in enumerate(offsets)
                ]
            except ValueError:
                pass
        month_count = max(1, min(6, int(self._float(record, "就诊的月数") or 1)))
        span_days = max(10, month_count * 28 - rng.randint(2, 8))
        start_month = rng.randint(1, max(1, 7 - month_count))
        base = datetime(
            2026,
            start_month,
            rng.randint(3, 12),
            rng.choice([8, 9, 10]),
            rng.choice([5, 12, 24, 36]),
        )
        offsets = [
            0,
            max(1, min(span_days - 7, rng.randint(2, 6))),
            max(3, min(span_days - 5, rng.randint(7, 14))),
            max(5, span_days // 2),
            max(7, span_days - rng.randint(2, 4)),
            span_days,
        ]
        dates: list[str] = []
        for offset in sorted(set(offsets)):
            moment = base + timedelta(days=offset, hours=rng.choice([0, 1, 4, 6]))
            dates.append(moment.strftime("%Y-%m-%d %H:%M"))
        while len(dates) < 6:
            moment = base + timedelta(days=span_days + len(dates), hours=2)
            dates.append(moment.strftime("%Y-%m-%d %H:%M"))
        return dates[:6]

    def _visit_rows(
        self,
        record: dict[str, Any],
        scenario: Scenario,
        service_dates: list[str],
        drug_rows: list[dict[str, Any]],
        institutions: list[str],
        pharmacies: list[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        sources = [rng.choice(institutions), rng.choice(pharmacies)]
        if scenario.code == "multi_org_purchase":
            sources = [institutions[0], pharmacies[0], pharmacies[1]]
        rows: list[dict[str, Any]] = []
        if self._registration_label(record.get("是否挂号")) == "已挂号":
            rows.append(
                {
                    "发生时间": service_dates[0],
                    "服务点": sources[0],
                    "业务动作": "门诊挂号登记",
                    "关联材料": "就诊诊疗记录",
                    "金额": "0.00",
                }
            )
        rows.append(
            {
                "发生时间": service_dates[1],
                "服务点": sources[0],
                "业务动作": "门诊接诊",
                "关联材料": "就诊诊疗记录",
                "金额": "0.00",
            }
        )
        if drug_rows:
            rows.append(
                {
                    "发生时间": service_dates[2],
                    "服务点": sources[-1],
                    "业务动作": "处方流转/购药结算",
                    "关联材料": "处方购药记录",
                    "金额": self._sum_rows(drug_rows, "金额"),
                }
            )
        total_visits = int(self._float(record, "就诊次数_SUM") or self._float(record, "月就诊次数_MAX") or len(rows))
        rows.append(
            {
                "发生时间": service_dates[-2],
                "服务点": "医保结算申报系统",
                "业务动作": f"本周期费用结算归集（{max(total_visits, len(rows))} 次）",
                "关联材料": "费用结算记录",
                "金额": self._fmt_money(self._amount(record, "ALL_SUM")),
            }
        )
        return rows

    def _drug_rows(
        self,
        *,
        scenario: Scenario,
        total: Decimal,
        service_dates: list[str],
        institutions: list[str],
        pharmacies: list[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        if total <= 0:
            return []
        catalog = list(DRUG_CATALOG.get(scenario.code) or DRUG_CATALOG["routine_outpatient"])
        rng.shuffle(catalog)
        if total >= Decimal("3000.00"):
            count = min(len(catalog), 5)
        elif total >= Decimal("800.00"):
            count = min(len(catalog), 4)
        else:
            count = min(len(catalog), 3 if total >= Decimal("80.00") else 2)
        amounts = self._allocate(total, count, rng)
        rows: list[dict[str, Any]] = []
        sources = [institutions[0], pharmacies[0], pharmacies[1]]
        for index, (drug, amount) in enumerate(zip(catalog[:count], amounts, strict=True)):
            if amount >= Decimal("1500.00"):
                qty = rng.randint(12, 30)
            elif amount >= Decimal("500.00"):
                qty = rng.randint(4, 10)
            else:
                qty = 1 if amount < Decimal("30.00") else rng.randint(1, 3)
            unit_price = (amount / Decimal(qty)).quantize(MONEY, ROUND_HALF_UP)
            rows.append(
                {
                    "发生时间": service_dates[min(index + 2, len(service_dates) - 2)],
                    "药品名称": drug.name,
                    "规格": drug.spec,
                    "数量": f"{qty}盒",
                    "单价": self._fmt_money(unit_price),
                    "金额": self._fmt_money(amount),
                    "来源": sources[min(index, len(sources) - 1)],
                    "用法": drug.usage,
                }
            )
        return rows

    def _non_drug_tables(
        self,
        *,
        scenario: Scenario,
        check_amount: Decimal,
        treatment_amount: Decimal,
        material_amount: Decimal,
        service_dates: list[str],
        institutions: list[str],
        rng: random.Random,
    ) -> list[BusinessMaterialTable]:
        fee_rows: list[dict[str, Any]] = []
        check_items = CT_CHECK_ITEMS if scenario.code == "high_check_fee_ct" else CHECK_ITEMS
        fee_rows.extend(
            self._fee_rows("检查费", check_items, check_amount, service_dates[1], institutions[0], rng)
        )
        fee_rows.extend(
            self._fee_rows("治疗费", TREATMENT_ITEMS, treatment_amount, service_dates[3], institutions[0], rng)
        )
        fee_rows.extend(
            self._fee_rows("医用材料费", MATERIAL_ITEMS, material_amount, service_dates[-2], institutions[0], rng)
        )
        summary_rows = [
            {"费用类别": "检查费", "明细合计": self._fmt_money(check_amount)},
            {"费用类别": "治疗费", "明细合计": self._fmt_money(treatment_amount)},
            {"费用类别": "医用材料费", "明细合计": self._fmt_money(material_amount)},
        ]
        return [
            BusinessMaterialTable(
                title="非药品费用业务明细",
                columns=["发生时间", "费用类别", "项目名称", "数量", "单价", "金额", "服务点"],
                rows=fee_rows,
            ),
            BusinessMaterialTable(
                title="费用明细合计",
                columns=["费用类别", "明细合计"],
                rows=summary_rows,
            ),
        ]

    def _fee_rows(
        self,
        category: str,
        catalog: list[str],
        total: Decimal,
        occurred_at: str,
        service_point: str,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        if total <= 0:
            return []
        count = 1 if total < Decimal("80.00") else min(3, len(catalog))
        selected = catalog[:]
        rng.shuffle(selected)
        amounts = self._allocate(total, count, rng)
        rows: list[dict[str, Any]] = []
        for item, amount in zip(selected[:count], amounts, strict=True):
            qty = 1 if amount < Decimal("120.00") else rng.randint(1, 2)
            unit_price = (amount / Decimal(qty)).quantize(MONEY, ROUND_HALF_UP)
            rows.append(
                {
                    "发生时间": occurred_at,
                    "费用类别": category,
                    "项目名称": item,
                    "数量": f"{qty}次",
                    "单价": self._fmt_money(unit_price),
                    "金额": self._fmt_money(amount),
                    "服务点": service_point,
                }
            )
        return rows

    def _clinical_record_document(
        self,
        *,
        case: CaseDetail,
        scenario: Scenario,
        subject_profile: MaterialSubjectProfile,
        record: dict[str, Any],
        context: dict[str, Any],
        service_dates: list[str],
        visit_rows: list[dict[str, Any]],
    ) -> MedicalRecordDocument:
        diagnosis, icd_code, complaint = self._diagnosis(scenario)
        if context.get("diagnosis"):
            diagnosis = str(context["diagnosis"])
        hospital_count = max(1, int(self._float(record, "月就诊医院数_MAX") or 1))
        total_visits = int(self._float(record, "就诊次数_SUM") or len(visit_rows))
        department = self._department_for_context(context)
        clinical_rows = [
            {
                "发生时间": row["发生时间"],
                "机构": row["服务点"],
                "科室": department,
                "记录类型": row["业务动作"],
                "记录摘要": row["关联材料"],
            }
            for row in visit_rows
            if row["服务点"] != "医保结算申报系统"
        ]
        content = (
            f"主诉：{complaint}\n"
            f"诊断：{diagnosis}（{icd_code}）\n"
            f"申报人特征：{subject_profile.gender}，{subject_profile.age_group}\n"
            f"本周期归集就诊记录 {max(total_visits, len(clinical_rows))} 次，"
            f"涉及医疗机构数按宽表统计不超过 {hospital_count} 家。"
        )
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-01",
            title="就诊诊疗记录",
            document_type="就诊诊疗",
            visit_date=self._range_label(service_dates[0], service_dates[-1]),
            institution="定点医疗机构",
            department=department,
            status="已整理",
            category_id="clinical",
            category_title="就诊诊疗材料",
            occurred_at=self._range_label(service_dates[0], service_dates[-1]),
            material_source="定点医疗机构门诊系统 / 挂号结算系统",
            occurrence_scene="",
            material_shape="表格 + 文本",
            content=content,
            summary_items=[
                f"就诊记录 {max(total_visits, len(clinical_rows))} 次",
                f"涉及机构数 {hospital_count} 家以内",
            ],
            check_points=self._check_points(context, "clinical"),
            tables=[
                BusinessMaterialTable(
                    title="挂号与就诊记录",
                    columns=["发生时间", "机构", "科室", "记录类型", "记录摘要"],
                    rows=clinical_rows,
                )
            ],
            metadata=self._material_metadata(context, scenario),
        )

    def _prescription_purchase_document(
        self,
        *,
        case: CaseDetail,
        scenario: Scenario,
        context: dict[str, Any],
        service_dates: list[str],
        drug_rows: list[dict[str, Any]],
        assets: list[BusinessMaterialAsset],
        document_id: str,
    ) -> MedicalRecordDocument:
        total = self._sum_rows(drug_rows, "金额")
        content = (
            f"药品明细合计 {total} 元。"
            "表格展示处方、院内药房或定点药店形成的药品事实；图片附件提供处方笺和购药小票预览。"
        )
        return MedicalRecordDocument(
            document_id=document_id,
            title="处方购药记录",
            document_type="处方购药",
            visit_date=self._range_label(service_dates[2], service_dates[-2]),
            institution="定点医疗机构 / 定点零售药店",
            department="药事服务",
            status="已整理" if drug_rows else "未见药品明细",
            category_id="prescription",
            category_title="处方购药材料",
            occurred_at=self._range_label(service_dates[2], service_dates[-2]),
            material_source="定点医疗机构 / 院内药房 / 定点零售药店",
            occurrence_scene="",
            material_shape="表格 + 图片" if assets else "表格",
            content=content,
            summary_items=[
                f"药品项目 {len(drug_rows)} 项",
                f"药品金额 {total} 元",
            ],
            check_points=self._check_points(context, "prescription"),
            tables=[
                BusinessMaterialTable(
                    title="处方药品明细",
                    columns=["发生时间", "药品名称", "规格", "数量", "单价", "金额", "来源", "用法"],
                    rows=drug_rows,
                )
            ],
            assets=assets,
            metadata=self._material_metadata(context, scenario),
        )

    def _settlement_record_document(
        self,
        *,
        case: CaseDetail,
        record: dict[str, Any],
        service_dates: list[str],
        drug_amount: Decimal,
        check_amount: Decimal,
        treatment_amount: Decimal,
        material_amount: Decimal,
        non_drug_tables: list[BusinessMaterialTable],
        context: dict[str, Any],
        assets: list[BusinessMaterialAsset],
    ) -> MedicalRecordDocument:
        total = self._amount(record, "ALL_SUM")
        approved = self._amount(record, "本次审批金额_SUM")
        pooled = self._amount(record, "统筹支付金额_SUM")
        personal = self._amount(record, "个人账户金额_SUM")
        summary_rows = [
            {"费用类别": "申报总费用", "明细合计": self._fmt_money(total)},
            {"费用类别": "本次审批金额", "明细合计": self._fmt_money(approved)},
            {"费用类别": "统筹支付金额", "明细合计": self._fmt_money(pooled)},
            {"费用类别": "个人账户金额", "明细合计": self._fmt_money(personal)},
            {"费用类别": "药品费", "明细合计": self._fmt_money(drug_amount)},
            {"费用类别": "检查费", "明细合计": self._fmt_money(check_amount)},
            {"费用类别": "治疗费", "明细合计": self._fmt_money(treatment_amount)},
            {"费用类别": "医用材料费", "明细合计": self._fmt_money(material_amount)},
        ]
        completeness_rows = self._attachment_rows(context)
        if not completeness_rows:
            completeness_rows = [
                {"材料名称": "就诊诊疗记录", "状态": "已归集", "材料形态": "表格 + 文本", "来源": "定点医疗机构"},
                {
                    "材料名称": "处方购药记录",
                    "状态": "已归集" if drug_amount > 0 else "未见药品明细",
                    "材料形态": "表格 + 图片" if assets else "表格",
                    "来源": "定点医疗机构 / 定点零售药店",
                },
                {"材料名称": "费用结算记录", "状态": "已归集", "材料形态": "表格", "来源": "医保结算申报系统"},
            ]
        tables = [
            BusinessMaterialTable(
                title="医保结算摘要",
                columns=["费用类别", "明细合计"],
                rows=summary_rows,
            ),
            *non_drug_tables[:1],
            BusinessMaterialTable(
                title="材料完整性清单",
                columns=["材料名称", "状态", "材料形态", "来源"],
                rows=completeness_rows,
            ),
        ]
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-04",
            title="费用结算记录",
            document_type="费用结算",
            visit_date=service_dates[-2],
            institution="医保结算申报系统",
            department="结算窗口",
            status="已整理",
            category_id="settlement",
            category_title="费用结算材料",
            occurred_at=service_dates[-2],
            material_source="医保结算申报系统",
            occurrence_scene="",
            material_shape="表格",
            content=(
                "展示医保结算摘要、非药品费用项目和材料完整性，不重复展示原始宽表字段。"
            ),
            summary_items=[
                f"申报总费用 {self._fmt_money(total)} 元",
            ],
            check_points=self._check_points(context, "settlement"),
            tables=tables,
            metadata=self._material_metadata(context, SCENARIOS.get(str(context.get("material_scenario")), Scenario("auto", "自动场景"))),
        )

    @staticmethod
    def _department_for_context(context: dict[str, Any]) -> str:
        actual_visit_type = str(context.get("actual_material_visit_type") or "")
        visit_type = str(context.get("visit_type") or "")
        if actual_visit_type == "普通门诊":
            return "普通门诊"
        if "急诊" in visit_type:
            return "急诊科"
        return "普通门诊"

    @staticmethod
    def _regional_providers(context: dict[str, Any]) -> tuple[list[str], list[str]]:
        region = str(context.get("treatment_region") or context.get("insured_region") or "北京")
        institutions = next(
            (providers for key, providers in REGIONAL_INSTITUTIONS.items() if key in region),
            REGIONAL_INSTITUTIONS["北京"],
        )
        pharmacies = next(
            (providers for key, providers in REGIONAL_PHARMACIES.items() if key in region),
            REGIONAL_PHARMACIES["北京"],
        )
        return list(institutions), list(pharmacies)

    @staticmethod
    def _display_case_context(context: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = (
            "insured_region",
            "treatment_region",
            "visit_type",
            "claim_mode",
            "direct_settlement",
            "filing_status",
            "emergency_material_status",
            "visit_date",
            "diagnosis",
            "applicant_category",
        )
        return {
            key: context[key]
            for key in allowed_keys
            if context.get(key) not in (None, "", [], {})
        }

    @staticmethod
    def _check_points(context: dict[str, Any], document_scope: str) -> list[str]:
        scenario = str(context.get("material_scenario") or "")
        focus = [
            str(item)
            for item in context.get("policy_check_focus", [])
            if item not in (None, "")
        ]
        common_by_scenario = {
            "remote_emergency_manual_unknown_filing": [
                "核验备案状态是否可追溯",
                "核验急诊材料能否支持急诊例外",
                "区分北京参保地待遇政策与上海就医地目录",
            ],
            "remote_emergency_identity_gap": [
                "核验附件是否存在急诊号别、急诊章或抢救记录",
                "普通门诊病历不能直接支持急诊视同备案",
            ],
            "remote_benefit_catalog_mix": [
                "上海侧目录用于支付范围核验，北京侧政策用于待遇测算",
                "票据中的上海预估比例不能直接作为手工报销最终依据",
            ],
            "policy_version_conflict": [
                "按服务日期核验政策 effective_date、publish_date 和 status",
                "旧政策仅作为历史依据，不直接作为当前有效口径",
            ],
            "high_check_fee_ct": [
                "核对 CT 报告、检查时间、项目名、数量和单价",
                "关注图文报告、胶片费是否存在独立收费依据",
            ],
            "high_drug_ratio_uri": [
                "核验药品目录甲乙类、备注和限定支付范围",
                "核验处方诊断与病历是否支持用药",
            ],
            "registration_gap_purchase": [
                "核验挂号记录、门诊病历、处方和收费票据是否闭合",
            ],
            "high_frequency_multi_org_chronic": [
                "核验处方日期、机构、药品数量和慢病长期处方依据",
                "关注短期多机构相近药品是否存在重复购药核验点",
            ],
        }
        points = [*common_by_scenario.get(scenario, []), *focus]
        if document_scope == "settlement":
            points.append("核对重构费用明细合计与报表聚合字段是否一致")
        unique: list[str] = []
        for point in points:
            if point and point not in unique:
                unique.append(point)
        return unique[:8]

    @staticmethod
    def _attachment_rows(context: dict[str, Any]) -> list[dict[str, str]]:
        profile = context.get("attachment_profile")
        if not isinstance(profile, dict):
            return []
        items = profile.get("items")
        if not isinstance(items, list):
            return []
        type_labels = {
            "outpatient_invoice": "门诊收费票据",
            "fee_detail": "费用明细",
            "prescription": "处方",
            "outpatient_record": "门诊病历",
            "ordinary_outpatient_record": "普通门诊病历",
            "emergency_record": "急诊病历",
            "online_filing_record": "线上备案记录",
            "chest_ct_report": "胸部 CT 报告",
            "registration_record": "挂号记录",
            "multiple_prescriptions": "多张处方",
            "pharmacy_receipts": "药店小票",
            "chronic_outpatient_record": "慢病门诊病历",
        }
        status_labels = {
            "parsed": "已解析",
            "missing": "缺失",
            "missing_or_unclear": "缺失或不清晰",
        }
        rows: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            status = str(item.get("status") or "")
            rows.append(
                {
                    "材料名称": type_labels.get(item_type, item_type or "未命名材料"),
                    "状态": status_labels.get(status, status or "待核验"),
                    "材料形态": "OCR结构化摘要",
                    "来源": "手工上传材料摘要",
                }
            )
        return rows

    @staticmethod
    def _material_metadata(context: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
        return {
            "material_scenario": scenario.code,
            "scenario_label": scenario.label,
            "claim_mode": context.get("claim_mode"),
            "insured_region": context.get("insured_region"),
            "treatment_region": context.get("treatment_region"),
            "attachment_ocr_status": (
                context.get("attachment_profile", {}).get("ocr_status")
                if isinstance(context.get("attachment_profile"), dict)
                else None
            ),
        }

    def _visit_summary_document(
        self,
        case: CaseDetail,
        scenario: Scenario,
        service_dates: list[str],
        visit_rows: list[dict[str, Any]],
    ) -> MedicalRecordDocument:
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-01",
            title="就诊诊疗记录",
            document_type="就诊诊疗",
            visit_date="本次申报周期",
            institution="医保结算申报系统",
            department="门诊/药事服务",
            status="已整理",
            category_id="clinical",
            category_title="就诊诊疗材料",
            occurred_at=service_dates[0],
            material_source="医保结算申报系统 / 定点服务机构回传",
            occurrence_scene="",
            material_shape="表格",
            content="按发生时间汇总本周期挂号、接诊、处方流转和结算申报节点。",
            summary_items=[
                f"归集记录 {len(visit_rows)} 条",
            ],
            tables=[
                BusinessMaterialTable(
                    title="挂号与就诊流水",
                    columns=["发生时间", "服务点", "业务动作", "关联材料", "金额"],
                    rows=visit_rows,
                )
            ],
        )

    def _outpatient_summary_document(
        self,
        case: CaseDetail,
        scenario: Scenario,
        subject_profile: MaterialSubjectProfile,
        service_dates: list[str],
    ) -> MedicalRecordDocument:
        diagnosis, icd_code, complaint = self._diagnosis(scenario)
        content = (
            f"主诉：{complaint}\n"
            f"诊断：{diagnosis}（{icd_code}）\n"
            f"服务点：{DEFAULT_INSTITUTION}\n"
            f"申报人特征：{subject_profile.gender}，{subject_profile.age_group}，"
            f"{'、'.join(subject_profile.patient_group_tags)}\n"
            "记录摘要：门诊接诊后形成处方或费用项目，供审核人员查看事实线索。"
        )
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-02",
            title="门诊病历摘要",
            document_type="门诊摘要",
            visit_date="本次申报周期",
            institution=DEFAULT_INSTITUTION,
            department="普通门诊",
            status="已整理",
            category_id="clinical",
            category_title="就诊诊疗材料",
            occurred_at=service_dates[0],
            material_source="定点医疗机构门诊系统",
            occurrence_scene="",
            material_shape="文本",
            content=content,
            summary_items=[diagnosis, icd_code, DEFAULT_INSTITUTION],
        )

    def _drug_detail_document(
        self,
        *,
        case: CaseDetail,
        scenario: Scenario,
        service_dates: list[str],
        drug_rows: list[dict[str, Any]],
        assets: list[BusinessMaterialAsset],
        document_id: str,
    ) -> MedicalRecordDocument:
        total = self._sum_rows(drug_rows, "金额")
        shape = "表格 + 图片" if assets else "表格"
        content = (
            f"药品明细合计 {total} 元。"
            "处方笺与药店购药小票可能来自不同业务环节，均关联本案件用药事实。"
        )
        return MedicalRecordDocument(
            document_id=document_id,
            title="处方购药记录",
            document_type="处方购药",
            visit_date="本次申报周期",
            institution="定点医疗机构 / 定点零售药店",
            department="药事服务",
            status="已整理" if drug_rows else "未见药品明细",
            category_id="prescription",
            category_title="处方购药材料",
            occurred_at=service_dates[1],
            material_source="定点医疗机构 / 定点零售药店",
            occurrence_scene="",
            material_shape=shape,
            content=content,
            summary_items=[f"药品项目 {len(drug_rows)} 项", f"药品金额 {total} 元"],
            tables=[
                BusinessMaterialTable(
                    title="处方药品明细",
                    columns=["发生时间", "药品名称", "规格", "数量", "单价", "金额", "来源", "用法"],
                    rows=drug_rows,
                )
            ],
            assets=assets,
        )

    def _settlement_document(
        self,
        case: CaseDetail,
        scenario: Scenario,
        service_dates: list[str],
        tables: list[BusinessMaterialTable],
    ) -> MedicalRecordDocument:
        fee_count = sum(len(table.rows) for table in tables[:1])
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-04",
            title="费用结算记录",
            document_type="费用结构",
            visit_date="本次申报周期",
            institution="医保结算申报系统",
            department="结算窗口",
            status="已整理",
            category_id="settlement",
            category_title="费用结算材料",
            occurred_at=service_dates[-1],
            material_source="医保结算申报系统",
            occurrence_scene="",
            material_shape="表格",
            content="按业务费用项目展开检查费、治疗费和医用材料费，不重复展示原始宽表字段。",
            summary_items=[f"非药品费用明细 {fee_count} 条"],
            tables=tables,
        )

    def _completeness_document(
        self,
        case: CaseDetail,
        scenario: Scenario,
        service_dates: list[str],
        assets: list[BusinessMaterialAsset],
    ) -> MedicalRecordDocument:
        image_count = len(assets)
        rows = [
            {"材料名称": "就诊诊疗记录", "状态": "已归集", "材料形态": "表格", "来源": "医保结算申报系统"},
            {"材料名称": "门诊病历摘要", "状态": "已归集", "材料形态": "文本", "来源": "定点医疗机构门诊系统"},
            {
                "材料名称": "处方笺",
                "状态": "已归集" if image_count else "未见图片",
                "材料形态": "PNG",
                "来源": "定点医疗机构门诊系统",
            },
            {
                "材料名称": "药店购药小票",
                "状态": "已归集" if image_count else "未见图片",
                "材料形态": "PNG",
                "来源": "定点零售药店",
            },
            {"材料名称": "费用结算记录", "状态": "已归集", "材料形态": "表格", "来源": "医保结算申报系统"},
        ]
        return MedicalRecordDocument(
            document_id=f"{case.case_id}-BM-05",
            title="材料完整性清单",
            document_type="材料目录",
            visit_date="本次申报周期",
            institution="材料目录",
            department="案件材料",
            status="已整理",
            category_id="settlement",
            category_title="费用结算材料",
            occurred_at=service_dates[-1],
            material_source="材料目录",
            occurrence_scene="",
            material_shape="表格",
            content="仅展示材料是否归集和材料来源，不输出审核判断或处理建议。",
            summary_items=[f"图片材料 {image_count} 张"],
            tables=[
                BusinessMaterialTable(
                    title="材料完整性清单",
                    columns=["材料名称", "状态", "材料形态", "来源"],
                    rows=rows,
                )
            ],
        )

    def _categories(
        self,
        documents: list[MedicalRecordDocument],
    ) -> list[BusinessMaterialCategory]:
        definitions = [
            ("clinical", "就诊诊疗材料"),
            ("prescription", "处方购药材料"),
            ("settlement", "费用结算材料"),
        ]
        return [
            BusinessMaterialCategory(
                category_id=category_id,
                title=title,
                documents=[doc for doc in documents if doc.category_id == category_id],
            )
            for category_id, title in definitions
        ]

    def _generate_png_assets(
        self,
        *,
        case: CaseDetail,
        subject_profile: MaterialSubjectProfile,
        scenario: Scenario,
        service_dates: list[str],
        drug_rows: list[dict[str, Any]],
        institutions: list[str],
        pharmacies: list[str],
        material_id: str,
    ) -> tuple[list[BusinessMaterialAsset], list[MaterialAssetLocation]]:
        if not drug_rows:
            return [], []
        self._asset_root.mkdir(parents=True, exist_ok=True)
        case_dir = self._asset_root / self._safe_path(case.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)

        plans = [
            ("asset-prescription-01", "处方笺", "prescription_png"),
            ("asset-pharmacy-receipt-01", "药店购药小票", "pharmacy_receipt_png"),
        ]
        assets: list[BusinessMaterialAsset] = []
        locations: list[MaterialAssetLocation] = []
        for asset_id, title, asset_type in plans:
            path = case_dir / f"{asset_id}.png"
            if asset_type == "prescription_png":
                self._render_prescription_png(
                    path=path,
                    case=case,
                    subject_profile=subject_profile,
                    scenario=scenario,
                    occurred_at=service_dates[2],
                    drug_rows=drug_rows[:3],
                    institution=institutions[0],
                )
            else:
                self._render_receipt_png(
                    path=path,
                    case=case,
                    occurred_at=service_dates[3],
                    drug_rows=drug_rows[-2:] or drug_rows[:1],
                    pharmacy=pharmacies[0],
                )
            digest = self._sha256(path)
            size = path.stat().st_size
            relative_path = path.relative_to(self._asset_root).as_posix()
            preview_url = (
                f"{self._api_prefix}/cases/{case.case_id}/materials/"
                f"{material_id}/assets/{asset_id}"
            )
            assets.append(
                BusinessMaterialAsset(
                    asset_id=asset_id,
                    material_id=material_id,
                    title=title,
                    asset_type=asset_type,  # type: ignore[arg-type]
                    file_type="image/png",
                    preview_url=preview_url,
                    sha256=digest,
                    size_bytes=size,
                )
            )
            locations.append(
                MaterialAssetLocation(
                    material_id=material_id,
                    asset_id=asset_id,
                    asset_type=asset_type,
                    file_type="image/png",
                    storage_path=relative_path,
                    sha256=digest,
                    size_bytes=size,
                    display_name=f"{title}.png",
                )
            )
        return assets, locations

    def _render_prescription_png(
        self,
        *,
        path: Path,
        case: CaseDetail,
        subject_profile: MaterialSubjectProfile,
        scenario: Scenario,
        occurred_at: str,
        drug_rows: list[dict[str, Any]],
        institution: str,
    ) -> None:
        image, draw, fonts = self._canvas()
        regular, bold, small = fonts
        self._draw_header(draw, "处方笺", regular, bold)
        y = 118
        fields = [
            ("处方号", f"RX-{case.case_id[-6:]}-01"),
            ("日期", occurred_at),
            ("机构", institution),
            ("科室", "普通门诊"),
            ("申报编号", subject_profile.subject_ref),
            ("性别/年龄段", f"{subject_profile.gender} / {subject_profile.age_group}"),
            ("业务类型", "门诊处方"),
        ]
        for index, (label, value) in enumerate(fields):
            x = 70 if index % 2 == 0 else 520
            row_y = y + (index // 2) * 42
            draw.text((x, row_y), f"{label}：{value}", font=regular, fill="#1f2a37")
        table_y = 310
        draw.line((60, table_y - 18, 940, table_y - 18), fill="#94a3b8", width=2)
        headers = ["药品名称", "规格", "数量", "用法"]
        xs = [70, 330, 555, 650]
        for x, header in zip(xs, headers, strict=True):
            draw.text((x, table_y), header, font=bold, fill="#0f172a")
        draw.line((60, table_y + 36, 940, table_y + 36), fill="#cbd5e1", width=1)
        for row_index, row in enumerate(drug_rows):
            row_y = table_y + 58 + row_index * 58
            values = [row["药品名称"], row["规格"], row["数量"], row["用法"]]
            for x, value in zip(xs, values, strict=True):
                draw.text((x, row_y), str(value), font=small, fill="#1f2937")
        draw.line((60, 560, 940, 560), fill="#cbd5e1", width=1)
        draw.text((70, 585), "医师：DR-042", font=regular, fill="#1f2a37")
        draw.text((320, 585), "药师审核：PH-018", font=regular, fill="#1f2a37")
        draw.text((650, 585), "处方状态：已流转", font=regular, fill="#1f2a37")
        image.save(path, "PNG")

    def _render_receipt_png(
        self,
        *,
        path: Path,
        case: CaseDetail,
        occurred_at: str,
        drug_rows: list[dict[str, Any]],
        pharmacy: str,
    ) -> None:
        image, draw, fonts = self._canvas(height=620)
        regular, bold, small = fonts
        self._draw_header(draw, "药店购药小票", regular, bold)
        draw.text((70, 118), f"药店：{pharmacy}", font=regular, fill="#1f2a37")
        draw.text((70, 160), f"流水号：PH-{case.case_id[-6:]}-01", font=regular, fill="#1f2a37")
        draw.text((520, 160), f"时间：{occurred_at}", font=regular, fill="#1f2a37")
        table_y = 235
        headers = ["品名", "规格", "数量", "金额"]
        xs = [70, 350, 610, 770]
        for x, header in zip(xs, headers, strict=True):
            draw.text((x, table_y), header, font=bold, fill="#0f172a")
        draw.line((60, table_y + 36, 940, table_y + 36), fill="#cbd5e1", width=1)
        for row_index, row in enumerate(drug_rows):
            row_y = table_y + 58 + row_index * 62
            values = [row["药品名称"], row["规格"], row["数量"], row["金额"]]
            for x, value in zip(xs, values, strict=True):
                draw.text((x, row_y), str(value), font=small, fill="#1f2937")
        total = self._sum_rows(drug_rows, "金额")
        draw.line((60, 450, 940, 450), fill="#cbd5e1", width=1)
        draw.text((620, 478), f"合计：{total} 元", font=bold, fill="#0f172a")
        draw.text((70, 530), "结算方式：医保电子凭证 / 个人账户", font=regular, fill="#1f2a37")
        draw.text((70, 570), "状态：已联网申报", font=regular, fill="#1f2a37")
        image.save(path, "PNG")

    def _canvas(
        self,
        *,
        width: int = 1000,
        height: int = 680,
    ) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]]:
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, width - 30, height - 30), outline="#94a3b8", width=2)
        regular = self._font(26)
        bold = self._font(34, bold=True)
        small = self._font(22)
        return image, draw, (regular, bold, small)

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        title: str,
        regular: ImageFont.ImageFont,
        bold: ImageFont.ImageFont,
    ) -> None:
        draw.text((70, 54), title, font=bold, fill="#0f172a")
        draw.text((760, 66), "医保定点服务材料", font=regular, fill="#334155")
        draw.line((60, 104, 940, 104), fill="#64748b", width=2)

    def _font(self, size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        font_name = "NotoSansSC-Bold.ttf" if bold else "NotoSansSC-Regular.ttf"
        candidates = [
            PROJECT_ROOT / "src" / "frontend" / "public" / "fonts" / font_name,
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def _diagnosis(self, scenario: Scenario) -> tuple[str, str, str]:
        if scenario.code == "upper_respiratory":
            return "急性上呼吸道感染", "J06.9", "咽痛、鼻塞2天"
        if scenario.code == "chronic_medication":
            return "高血压病；2型糖尿病", "I10；E11.9", "慢病复诊取药"
        if scenario.code == "high_check_fee":
            return "咳嗽待查", "R05", "咳嗽、胸闷3天"
        if scenario.code == "high_treatment_fee":
            return "腰腿痛", "M54.5", "腰部疼痛伴活动受限"
        if scenario.code == "multi_org_purchase":
            return "慢性病续方；上呼吸道感染", "I10；J06.9", "复诊续方与购药"
        if scenario.code == "material_gap":
            return "门诊诊断待补充", "Z03.9", "门诊购药记录待补充"
        return "普通门诊疾病", "Z76.0", "门诊复诊购药"

    def _drug_amount(self, record: dict[str, Any]) -> Decimal:
        amount = self._amount(record, "药品费发生金额_SUM")
        if amount > 0:
            return amount
        return self._amount(record, "药品费申报金额_SUM")

    def _amount(self, record: dict[str, Any], field: str) -> Decimal:
        value = record.get(field)
        if self._raw_missing(value):
            return Decimal("0.00")
        try:
            return max(Decimal(str(value).replace(",", "")).quantize(MONEY, ROUND_HALF_UP), Decimal("0.00"))
        except Exception:
            return Decimal("0.00")

    def _float(self, record: dict[str, Any], field: str) -> float:
        value = record.get(field)
        if self._raw_missing(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _raw_missing(value: Any) -> bool:
        return value is None or value == "" or value == "缺失"

    def _registration_label(self, value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "缺失"
        if number == 1:
            return "已挂号"
        if number == 0:
            return "未挂号"
        return "缺失"

    def _allocate(
        self,
        total: Decimal,
        count: int,
        rng: random.Random,
    ) -> list[Decimal]:
        if total <= 0 or count <= 0:
            return []
        if total < Decimal(count) * MONEY:
            count = 1
        weights = [Decimal(str(rng.uniform(0.75, 1.35))) for _ in range(count)]
        weight_sum = sum(weights)
        amounts = [
            (total * weight / weight_sum).quantize(MONEY, ROUND_HALF_UP)
            for weight in weights
        ]
        diff = total - sum(amounts)
        amounts[-1] = (amounts[-1] + diff).quantize(MONEY, ROUND_HALF_UP)
        return amounts

    @staticmethod
    def _fmt_money(value: Decimal) -> str:
        return f"{value.quantize(MONEY, ROUND_HALF_UP):.2f}"

    @staticmethod
    def _range_label(start: str, end: str) -> str:
        start_date = start[:10]
        end_date = end[:10]
        return start if start_date == end_date else f"{start_date} 至 {end_date}"

    @staticmethod
    def _sum_rows(rows: list[dict[str, Any]], field: str) -> str:
        total = Decimal("0.00")
        for row in rows:
            try:
                total += Decimal(str(row.get(field, "0")).replace(",", ""))
            except Exception:
                continue
        return f"{total.quantize(MONEY, ROUND_HALF_UP):.2f}"

    def _fingerprint(self, case: CaseDetail, record: dict[str, Any]) -> str:
        safe_record = {
            key: value
            for key, value in record.items()
            if key not in {"RES", "姓名", "身份证", "电话", "住址"}
        }
        payload = {
            "case_id": case.case_id,
            "subject_ref": case.subject_ref,
            "record": safe_record,
            "case_context": case.case_context,
            "template_version": TEMPLATE_VERSION,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_path(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
