"""规则评估与基线模块。

合并自：
- domain/rules.py —— 门诊普通结算统计业务核验规则池（RuleService）
- domain/rule_baseline.py —— 规则池聚合阈值基线（RuleBaselineService）

基线只包含聚合阈值，不读取行级标签或生产原始数据。
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

from .entities import RuleCheckItem, RuleHit

# ============================================================================
# 来自 domain/rules.py —— 规则池常量
# ============================================================================

RULE_POOL_VERSION = "op-rule-pool-v0.3"

LAYER_DATA = "data_quality_or_applicability"
LAYER_RISK = "risk_signal"
LAYER_STRONG = "strong_review_signal"

ACTION_MANUAL_REVIEW = "MANUAL_REVIEW"
ACTION_REQUEST_SUPPLEMENT = "REQUEST_SUPPLEMENT"
ACTION_SPECIAL_AUDIT = "SPECIAL_AUDIT"
ACTION_DATA_QUALITY_CHECK = "DATA_QUALITY_CHECK"

SEVERITY_RANK = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# ============================================================================
# 来自 domain/rule_baseline.py —— 聚合阈值基线
# ============================================================================

# 默认基线路径：backend/fixtures/rule_baselines.json
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "fixtures" / "rule_baselines.json"
)


class RuleBaselineService:
    """读取不含行级数据的聚合阈值基线。"""

    def __init__(self, baseline_path: Path | None = None) -> None:
        """初始化基线服务。

        参数：
            baseline_path: 基线 JSON 文件路径。为 None 时使用默认路径。
        """
        self.baseline_path = baseline_path or DEFAULT_BASELINE_PATH
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            self._payload = json.load(f)

    @property
    def version(self) -> str:
        """基线版本号。"""
        return str(self._payload["version"])

    @property
    def row_count(self) -> int:
        """历史样本记录数。"""
        return int(self._payload["row_count"])

    @property
    def field_count(self) -> int:
        """可提供聚合参考的字段数量。"""
        return int(self._payload.get("field_count") or len(self._payload["amount_percentiles"]))

    @property
    def fields(self) -> set[str]:
        """可查询聚合分位的字段集合。"""
        return set(self._payload["amount_percentiles"])

    def percentile(self, field: str, percentile: str) -> float:
        """查询指定字段的百分位值。

        参数：
            field: 字段名。
            percentile: 百分位标识，如 p95、p98、p99。

        返回：
            对应百分位的浮点值。
        """
        return float(self._payload["amount_percentiles"][field][percentile])

    def percentiles(self, field: str) -> dict[str, float]:
        """查询指定字段的全部聚合百分位。"""
        return {
            key: float(value)
            for key, value in self._payload["amount_percentiles"][field].items()
        }

    def band(self, field: str, value: float) -> str | None:
        """返回 value 所在的高分位档位：p95 | p98 | p99。

        参数：
            field: 字段名。
            value: 待查询的数值。

        返回：
            分位档位字符串，低于所有阈值时返回 None。
        """
        field_percentiles = self.percentiles(field)
        p50 = field_percentiles.get("p50")
        if p50 is not None and value <= p50:
            return None
        if value > field_percentiles["p99"]:
            return "p99"
        if value > field_percentiles["p98"]:
            return "p98"
        if value > field_percentiles["p95"]:
            return "p95"
        return None


# ============================================================================
# 来自 domain/rules.py —— 业务核验规则池
# ============================================================================


class RuleService:
    """基于单条完整脱敏宽表和安全案件上下文计算业务核验场景。"""

    def __init__(self, baseline_service: RuleBaselineService | None = None) -> None:
        """初始化规则服务。

        参数：
            baseline_service: 规则基线服务，用于分位阈值查询。为 None 时使用默认基线。
        """
        self.baseline = baseline_service or RuleBaselineService()

    def evaluate(
        self,
        features: dict[str, float],
        source_record: dict[str, float | str] | None = None,
        case_context: dict[str, Any] | None = None,
    ) -> list[RuleHit]:
        """对单条记录执行全部规则核验。

        参数：
            features: 脱敏统计特征键值表。
            source_record: 完整脱敏宽表记录，可选。
            case_context: 离线预处理生成的安全案件上下文，可选。

        返回：
            12 条规则命中结果列表（OP-R001 至 OP-R012）。
        """
        record = source_record or features
        context = case_context or {}
        rules = [
            self._visit_behavior(record),
            self._cost_payment_structure(record),
            self._drug_rationality(record, context=context),
            self._inspection_treatment_structure(record, context=context),
            self._registration_flow(record, context=context),
            self._benefit_subsidy(record),
            self._declaration_quality(record, has_wide_record=source_record is not None),
            self._remote_manual_reimbursement(context),
            self._emergency_material(context),
            self._benefit_catalog_mix(context),
            self._policy_version_conflict(context),
        ]
        rules.append(self._material_supplement(rules))
        return rules

    # ---- OP-R001: 就诊行为一致性核验 ----

    def _visit_behavior(self, record: dict[str, Any]) -> RuleHit:
        """OP-R001 就诊行为一致性核验。

        判断就诊频次、多机构就诊和同日跨机构行为是否形成需要说明的组合场景。
        """
        visit_max = self._number(record, "月就诊次数_MAX")
        hospital_max = self._number(record, "月就诊医院数_MAX")
        cross_days = self._number(record, "一天去两家医院的天数")

        visit_focus = (visit_max or 0.0) >= 10
        hospital_focus = (hospital_max or 0.0) >= 3
        cross_focus = (cross_days or 0.0) >= 10
        focus_count = sum([visit_focus, hospital_focus, cross_focus])
        strong = (cross_days or 0.0) >= 16 or (
            (visit_max or 0.0) >= 17 and (hospital_max or 0.0) >= 4
        )
        hit = strong or focus_count >= 2

        severity = "low"
        layer = LAYER_RISK
        if strong:
            severity = "critical"
            layer = LAYER_STRONG
        elif focus_count >= 2:
            severity = "high" if cross_focus else "medium"

        explanation = (
            "就诊频次、多机构或同日跨机构行为形成组合线索，需结合就诊明细确认原因。"
            if hit
            else "就诊行为未形成需要单独核验的组合线索。"
        )

        return self._rule(
            "OP-R001",
            "就诊行为一致性核验",
            "rule:OP-R001:visit_behavior",
            "判断就诊频次、多机构就诊和同日跨机构行为是否形成需要说明的组合场景。",
            [
                self._item(
                    "就诊行为组合判断",
                    (
                        f"月最高就诊 {self._fmt_number(visit_max)}，"
                        f"月最高医院 {self._fmt_number(hospital_max)}，"
                        f"单日跨医院 {self._fmt_number(cross_days)}"
                    ),
                    "两类及以上就诊行为线索同时出现，或同日跨机构行为达到强关注区间",
                    hit,
                    layer,
                    severity,
                    explanation,
                )
            ],
        )

    # ---- OP-R002: 费用与支付结构核验 ----

    def _cost_payment_structure(self, record: dict[str, Any]) -> RuleHit:
        """OP-R002 费用与支付结构核验。

        判断费用规模和基金、账户、个人负担结构是否组合形成支付核验场景。
        """
        total = self._number(record, "ALL_SUM") or 0.0
        approval = self._number(record, "本次审批金额_SUM") or 0.0
        fund_payment = max(
            self._number(record, "基本统筹基金支付金额_SUM") or 0.0,
            self._number(record, "统筹支付金额_SUM") or 0.0,
        )
        account = self._number(record, "基本个人账户支付_SUM")
        non_account = self._number(record, "非账户支付金额_SUM")

        total_band = self._safe_band("ALL_SUM", total)
        approval_band = self._safe_band("本次审批金额_SUM", approval)
        fund_band = self._safe_band("基本统筹基金支付金额_SUM", fund_payment)
        account_band = self._safe_band("基本个人账户支付_SUM", account)
        fund_ratio = fund_payment / total if total > 0 else None
        non_account_ratio = (non_account or 0.0) / total if total > 0 else None

        amount_focus = any([total_band, approval_band, fund_band])
        payment_focus = bool(
            fund_ratio is not None
            and (fund_ratio >= 0.90 or non_account_ratio is not None and non_account_ratio <= 0.02)
        ) or account_band in ("p98", "p99")
        hit = amount_focus and payment_focus

        severity = "medium"
        if total_band == "p99" or approval_band == "p99":
            severity = "high"
        if hit and fund_ratio is not None and fund_ratio >= 0.95:
            severity = "high"

        return self._rule(
            "OP-R002",
            "费用与支付结构核验",
            "rule:OP-R002:cost_payment_structure",
            "判断费用规模和基金、账户、个人负担结构是否组合形成支付核验场景。",
            [
                self._item(
                    "费用支付组合判断",
                    (
                        f"申报总费用 {self._fmt_money(total)}，"
                        f"审批金额 {self._fmt_money(approval)}，"
                        f"基金占比 {self._fmt_ratio(fund_ratio)}，"
                        f"非账户支付占比 {self._fmt_ratio(non_account_ratio)}，"
                        f"个人账户 {self._fmt_money(account)}"
                    ),
                    "费用或支付金额达到高分位，并叠加基金支付占比偏高、非账户负担偏低或账户支付高分位",
                    hit,
                    LAYER_RISK,
                    severity if hit else "low",
                    (
                        "费用规模与支付结构同时形成关注场景，建议核对结算单和基金支付明细。"
                        if hit
                        else "费用规模或支付结构未形成组合核验场景。"
                    ),
                )
            ],
        )

    # ---- OP-R003: 药品费用合理性核验 ----

    def _drug_rationality(
        self,
        record: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> RuleHit:
        """OP-R003 药品费用合理性核验。

        判断药品费用规模、药品占比和贵重药品费用是否需要结合处方材料确认。
        """
        ratio = self._number(record, "药品在总金额中的占比")
        drug_amount = max(
            self._number(record, "药品费发生金额_SUM") or 0.0,
            self._number(record, "药品费申报金额_SUM") or 0.0,
        )
        precious_drug = self._number(record, "贵重药品发生金额_SUM")
        patent_drug = self._number(record, "中成药费发生金额_SUM")

        amount_band = self._safe_band("药品费发生金额_SUM", drug_amount)
        precious_band = self._safe_band("贵重药品发生金额_SUM", precious_drug)
        patent_band = self._safe_band("中成药费发生金额_SUM", patent_drug)
        structure_hit = bool(
            ratio is not None
            and amount_band is not None
            and ratio >= 0.90
        )
        scenario = self._context_text(context, "material_scenario")
        scenario_drug_hit = bool(
            scenario == "high_drug_ratio_uri"
            and ratio is not None
            and ratio >= 0.75
            and drug_amount > 0
        )
        precious_hit = precious_band is not None
        hit = structure_hit or scenario_drug_hit or precious_hit

        layer = LAYER_RISK
        severity = "medium" if hit else "low"
        if precious_band == "p99" or (ratio is not None and ratio >= 0.95 and amount_band == "p99"):
            layer = LAYER_STRONG
            severity = "critical"
        elif amount_band in ("p98", "p99") or precious_band in ("p98", "p99"):
            severity = "high"

        return self._rule(
            "OP-R003",
            "药品费用合理性核验",
            "rule:OP-R003:drug_rationality",
            "判断药品费用规模、药品占比和贵重药品费用是否需要结合处方材料确认。",
            [
                self._item(
                    "药品费用结构判断",
                    (
                        f"药品费用 {self._fmt_money(drug_amount)}，"
                        f"药品占比 {self._fmt_ratio(ratio)}，"
                        f"贵重药品 {self._fmt_money(precious_drug)}，"
                        f"中成药 {self._fmt_money(patent_drug)}"
                    ),
                    "药品金额达到高分位且药品占比较高，或贵重药品金额达到高分位",
                    hit,
                    layer,
                    severity,
                    (
                        "药品费用结构形成核验场景，建议结合处方、药品清单和长期用药材料确认。"
                        if structure_hit or precious_hit
                        else (
                            "药品费占比高且样本场景提示处方诊断匹配待核验，建议结合处方、病历和目录限定支付范围确认。"
                            if scenario_drug_hit
                            else (
                                "中成药费用达到高分位，仅作为药品结构材料提示。"
                                if patent_band
                                else "药品费用结构未形成需要单独核验的组合场景。"
                            )
                        )
                    ),
                )
            ],
        )

    # ---- OP-R004: 检查治疗结构核验 ----

    def _inspection_treatment_structure(
        self,
        record: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> RuleHit:
        """OP-R004 检查治疗结构核验。

        判断检查、治疗项目结构是否与普通门诊统计场景匹配，必要时提示材料核验。
        """
        check_ratio = self._number(record, "检查总费用在总金额占比")
        check_amount = self._number(record, "检查费发生金额_SUM") or 0.0
        treatment_ratio = self._number(record, "治疗费用在总金额占比")
        treatment_amount = self._number(record, "治疗费发生金额_SUM") or 0.0
        surgery = self._number(record, "手术费发生金额_SUM") or 0.0
        bed = self._number(record, "床位费发生金额_SUM") or 0.0
        material = self._number(record, "医用材料发生金额_SUM") or 0.0

        check_band = self._safe_band("检查费发生金额_SUM", check_amount)
        treatment_band = self._safe_band("治疗费发生金额_SUM", treatment_amount)
        scenario = self._context_text(context, "material_scenario")
        scenario_check_hit = bool(
            scenario == "high_check_fee_ct"
            and check_ratio is not None
            and check_ratio >= 0.45
            and check_amount > 0
        )
        structure_hit = bool(
            check_ratio is not None and check_ratio >= 0.20 and check_band
        ) or bool(
            treatment_ratio is not None and treatment_ratio >= 0.20 and treatment_band
        ) or scenario_check_hit
        scope_notice = any(value > 0 for value in [surgery, bed, material])
        hit = structure_hit or scope_notice

        layer = LAYER_RISK if structure_hit else LAYER_DATA
        severity = "medium" if structure_hit else "info"
        if (
            check_ratio is not None
            and check_ratio >= 0.35
            or treatment_ratio is not None
            and treatment_ratio >= 0.45
        ):
            severity = "high"

        return self._rule(
            "OP-R004",
            "检查治疗结构核验",
            "rule:OP-R004:inspection_treatment_structure",
            "判断检查、治疗项目结构是否与普通门诊统计场景匹配，必要时提示材料核验。",
            [
                self._item(
                    "检查治疗结构判断",
                    (
                        f"检查费用 {self._fmt_money(check_amount)}，"
                        f"检查占比 {self._fmt_ratio(check_ratio)}，"
                        f"治疗费用 {self._fmt_money(treatment_amount)}，"
                        f"治疗占比 {self._fmt_ratio(treatment_ratio)}，"
                        f"手术/床位/材料 {self._fmt_money(surgery + bed + material)}"
                    ),
                    "检查或治疗费用占比与金额同时形成关注，或普通门诊场景出现需确认的手术、床位、材料字段",
                    hit,
                    layer,
                    severity,
                    (
                        "检查治疗结构形成核验场景，建议核对项目明细和诊疗摘要材料。"
                        if structure_hit
                        else (
                            "存在可能超出普通门诊适用范围的字段，建议先确认申报材料场景。"
                            if scope_notice
                            else "检查治疗结构未形成需要单独核验的场景。"
                        )
                    ),
                )
            ],
            default_action=ACTION_REQUEST_SUPPLEMENT,
        )

    # ---- OP-R005: 挂号与流程一致性核验 ----

    def _registration_flow(
        self,
        record: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> RuleHit:
        """OP-R005 挂号与流程一致性核验。

        判断挂号状态是否与费用、频次、多机构就诊行为形成流程一致性核验场景。
        """
        registered = self._number(record, "是否挂号")
        total = self._number(record, "ALL_SUM") or 0.0
        approval = self._number(record, "本次审批金额_SUM") or 0.0
        visit_max = self._number(record, "月就诊次数_MAX") or 0.0
        hospital_max = self._number(record, "月就诊医院数_MAX") or 0.0
        cross_days = self._number(record, "一天去两家医院的天数") or 0.0

        amount_focus = (
            self._safe_band("ALL_SUM", total) is not None
            or self._safe_band("本次审批金额_SUM", approval) is not None
        )
        behavior_focus = visit_max >= 10 or hospital_max >= 3 or cross_days >= 10
        scenario = self._context_text(context, "material_scenario")
        has_settlement_without_registration = bool(registered == 0 and total > 0 and approval > 0)
        scenario_gap = bool(
            scenario == "registration_gap_purchase"
            and registered == 0
            and total > 0
        )
        hit = bool(
            registered == 0
            and (
                amount_focus
                or behavior_focus
                or has_settlement_without_registration
                or scenario_gap
            )
        )
        standalone_notice = registered == 0 and not hit

        return self._rule(
            "OP-R005",
            "挂号与流程一致性核验",
            "rule:OP-R005:registration_flow",
            "判断挂号状态是否与费用、频次、多机构就诊行为形成流程一致性核验场景。",
            [
                self._item(
                    "挂号流程组合判断",
                    (
                        f"挂号状态 {'未挂号' if registered == 0 else '已挂号'}，"
                        f"申报总费用 {self._fmt_money(total)}，"
                        f"月最高就诊 {self._fmt_number(visit_max)}，"
                        f"月最高医院 {self._fmt_number(hospital_max)}，"
                        f"单日跨医院 {self._fmt_number(cross_days)}"
                    ),
                    "未挂号状态叠加已发生费用、审批金额、较高频次、多机构或同日跨机构行为时进入流程材料核验",
                    hit or standalone_notice,
                    LAYER_RISK if hit else LAYER_DATA,
                    "medium" if hit else ("info" if standalone_notice else "low"),
                    (
                        "未挂号状态叠加费用或就诊行为线索，建议补充挂号记录或流程材料。"
                        if hit
                        else (
                            "存在未挂号状态，建议作为流程一致性提示处理。"
                            if standalone_notice
                            else "挂号与就诊流程未形成需要单独核验的场景。"
                        )
                    ),
                )
            ],
            default_action=ACTION_REQUEST_SUPPLEMENT,
        )

    # ---- OP-R006: 待遇补助口径核验 ----

    def _benefit_subsidy(self, record: dict[str, Any]) -> RuleHit:
        """OP-R006 待遇补助口径核验。

        判断救助、优抚、补助和支付字段是否需要按待遇口径确认，不把身份或待遇本身作为风险结论。
        """
        subsidy_fields = [
            "医疗救助个人按比例负担金额_SUM",
            "公务员医疗补助基金支付金额_SUM",
            "城乡救助补助金额_SUM",
            "医疗救助医院申请_SUM",
            "残疾军人补助_SUM",
            "民政救助补助_SUM",
            "城乡优抚补助_SUM",
            "BZ_民政救助",
            "BZ_城乡优抚",
        ]
        triggered_subsidy = [
            field for field in subsidy_fields if (self._number(record, field) or 0.0) > 0
        ]
        fund_payment = max(
            self._number(record, "基本统筹基金支付金额_SUM") or 0.0,
            self._number(record, "统筹支付金额_SUM") or 0.0,
        )
        account = self._number(record, "基本个人账户支付_SUM")
        hit = bool(triggered_subsidy)

        return self._rule(
            "OP-R006",
            "待遇补助口径核验",
            "rule:OP-R006:benefit_subsidy",
            "判断救助、优抚、补助和支付字段是否需要按待遇口径确认，不把身份或待遇本身作为风险结论。",
            [
                self._item(
                    "待遇补助字段判断",
                    (
                        f"涉及字段 {self._join_or_none(triggered_subsidy)}，"
                        f"统筹支付 {self._fmt_money(fund_payment)}，"
                        f"个人账户 {self._fmt_money(account)}"
                    ),
                    "救助、优抚、补助字段存在取值时，提示核对待遇资格和支付口径一致性",
                    hit,
                    LAYER_DATA,
                    "info" if hit else "low",
                    (
                        "存在待遇、救助、优抚或补助相关字段，建议核对待遇资格和支付口径一致性。"
                        if hit
                        else "未发现需要单独提示的待遇补助口径场景。"
                    ),
                )
            ],
            default_action=ACTION_DATA_QUALITY_CHECK,
            force_non_risk=True,
        )

    # ---- OP-R007: 申报数据质量核验 ----

    def _declaration_quality(self, record: dict[str, Any], has_wide_record: bool) -> RuleHit:
        """OP-R007 申报数据质量核验。

        判断申报字段完整性和费用占比反算是否支持后续业务核验。
        """
        total = self._number(record, "ALL_SUM") or 0.0
        ratio_issues: list[str] = []
        ratio_checks = [
            ("药品费发生金额_SUM", "药品在总金额中的占比", "药品费用占比"),
            ("检查费发生金额_SUM", "检查总费用在总金额占比", "检查费用占比"),
            ("治疗费发生金额_SUM", "治疗费用在总金额占比", "治疗费用占比"),
        ]
        for amount_field, ratio_field, label in ratio_checks:
            amount = self._number(record, amount_field)
            ratio = self._number(record, ratio_field)
            if total <= 0 or amount is None or ratio is None:
                continue
            calculated = amount / total
            if abs(calculated - ratio) > 0.05:
                ratio_issues.append(label)

        hit = (not has_wide_record) or bool(ratio_issues)
        current_value = (
            "完整脱敏宽表已接入"
            if has_wide_record
            else "仅收到最小统计特征"
        )
        if ratio_issues:
            current_value += f"，占比反算需确认：{self._join_or_none(ratio_issues)}"

        return self._rule(
            "OP-R007",
            "申报数据质量核验",
            "rule:OP-R007:declaration_quality",
            "判断申报字段完整性和费用占比反算是否支持后续业务核验。",
            [
                self._item(
                    "数据质量判断",
                    current_value,
                    "完整宽表应通过字段、安全和数值校验；费用占比与金额反算偏差不超过 0.05",
                    hit,
                    LAYER_DATA,
                    "info" if hit else "low",
                    (
                        "申报数据存在完整性或占比一致性提示，建议先确认数据质量。"
                        if hit
                        else "申报数据完整性和主要费用占比反算未发现明显问题。"
                    ),
                )
            ],
            default_action=ACTION_DATA_QUALITY_CHECK,
            force_non_risk=True,
        )

    # ---- OP-R008: 材料补充核验 ----

    def _material_supplement(self, rules: list[RuleHit]) -> RuleHit:
        """OP-R008 材料补充核验。

        根据已形成的业务核验场景整理审核员需要优先查看的材料。
        """
        materials = self._materials_for_rules(rules)
        hit = bool(materials)

        return self._rule(
            "OP-R008",
            "材料补充核验",
            "rule:OP-R008:material_supplement",
            "根据已形成的业务核验场景整理审核员需要优先查看的材料。",
            [
                self._item(
                    "需查看材料清单",
                    self._join_or_none(materials),
                    "按命中的业务核验场景补齐或查看对应材料",
                    hit,
                    LAYER_DATA,
                    "info" if hit else "low",
                    (
                        "已根据业务核验场景整理需优先查看的材料。"
                        if hit
                        else "当前没有额外材料补充提示。"
                    ),
                )
            ],
            default_action=ACTION_REQUEST_SUPPLEMENT,
            force_non_risk=True,
        )

    # ---- OP-R009: 异地手工报销备案/急诊例外核验 ----

    def _remote_manual_reimbursement(self, context: dict[str, Any]) -> RuleHit:
        """OP-R009 异地手工报销备案/急诊例外核验。"""

        insured_region = self._context_text(context, "insured_region")
        treatment_region = self._context_text(context, "treatment_region")
        claim_mode = self._context_text(context, "claim_mode")
        visit_type = self._context_text(context, "visit_type")
        filing_status = self._context_text(context, "filing_status")
        emergency_status = self._context_text(context, "emergency_material_status")

        is_remote = self._is_remote_context(context)
        is_manual = claim_mode in {"manual_reimbursement", "manual_upload_review"} or "手工" in self._context_text(
            context,
            "claim_mode_label",
        )
        is_non_direct = self._context_bool(context, "direct_settlement") is False
        filing_unclear = filing_status in {"", "missing", "unknown", "unclear"}
        emergency_needs_review = emergency_status in {
            "missing",
            "missing_or_unclear",
            "present_but_needs_verification",
            "ordinary_outpatient_only",
            "not_established",
        }
        hit = bool(
            is_remote
            and is_manual
            and is_non_direct
            and (filing_unclear or emergency_needs_review)
        )

        return self._rule(
            "OP-R009",
            "异地手工报销备案/急诊例外核验",
            "rule:OP-R009:remote_manual_reimbursement",
            "提示异地就医、手工报销、非直接结算下备案状态和急诊例外材料需要人工核验。",
            [
                self._item(
                    "异地手工报销路径判断",
                    (
                        f"参保地 {insured_region or '缺失'}，"
                        f"就医地 {treatment_region or '缺失'}，"
                        f"就医类型 {visit_type or '缺失'}，"
                        f"报销方式 {claim_mode or '缺失'}，"
                        f"直接结算 {self._context_bool(context, 'direct_settlement')}，"
                        f"备案状态 {filing_status or '缺失'}，"
                        f"急诊材料 {emergency_status or '缺失'}"
                    ),
                    "参保地与就医地不一致、手工报销、非直结，且备案或急诊例外材料仍待核验",
                    hit,
                    LAYER_RISK,
                    "medium" if hit else "low",
                    (
                        "异地手工报销且非直接结算，备案状态或急诊例外材料仍待核验，需结合参保地待遇政策和就医地目录确认。"
                        if hit
                        else "未形成异地手工报销备案/急诊例外核验提示。"
                    ),
                )
            ],
            default_action=ACTION_REQUEST_SUPPLEMENT,
        )

    # ---- OP-R010: 急诊身份材料核验 ----

    def _emergency_material(self, context: dict[str, Any]) -> RuleHit:
        """OP-R010 急诊身份材料核验。"""

        visit_type = self._context_text(context, "visit_type")
        application_visit_type = self._context_text(context, "application_visit_type")
        actual_material_visit_type = self._context_text(context, "actual_material_visit_type")
        emergency_status = self._context_text(context, "emergency_material_status")
        claim_mode = self._context_text(context, "claim_mode")

        claims_emergency = "急诊" in visit_type or "急诊" in application_visit_type
        manual = claim_mode in {"manual_reimbursement", "manual_upload_review"} or "手工" in self._context_text(
            context,
            "claim_mode_label",
        )
        material_gap = emergency_status in {
            "missing",
            "missing_or_unclear",
            "ordinary_outpatient_only",
            "not_established",
        } or actual_material_visit_type == "普通门诊"
        hit = bool(claims_emergency and manual and material_gap)

        return self._rule(
            "OP-R010",
            "急诊身份材料核验",
            "rule:OP-R010:emergency_material",
            "提示申请急诊或急诊例外时，需核验附件是否具有急诊号别、急诊病历、急诊章或抢救记录等证据。",
            [
                self._item(
                    "急诊材料成立性判断",
                    (
                        f"申请就医类型 {application_visit_type or visit_type or '缺失'}，"
                        f"附件体现类型 {actual_material_visit_type or '缺失'}，"
                        f"急诊材料状态 {emergency_status or '缺失'}"
                    ),
                    "申请按急诊处理，但附件仅显示普通门诊或缺少急诊标识时，需要补充急诊材料",
                    hit,
                    LAYER_RISK,
                    "medium" if hit else "low",
                    (
                        "当前附件不足以支持急诊身份或急诊例外成立，建议补充急诊病历、急诊挂号记录或医院急诊证明。"
                        if hit
                        else "未形成急诊身份材料缺口提示。"
                    ),
                )
            ],
            default_action=ACTION_REQUEST_SUPPLEMENT,
        )

    # ---- OP-R011: 参保地待遇 vs 就医地目录混用核验 ----

    def _benefit_catalog_mix(self, context: dict[str, Any]) -> RuleHit:
        """OP-R011 参保地待遇与就医地目录混用核验。"""

        insured_region = self._context_text(context, "insured_region")
        treatment_region = self._context_text(context, "treatment_region")
        calculation_basis = self._context_text(context, "benefit_calculation_basis")
        payment_basis_source = self._context_text(context, "payment_basis_source")
        catalog_basis_source = self._context_text(context, "catalog_basis_source")
        claim_mode = self._context_text(context, "claim_mode")

        remote_manual = self._is_remote_context(context) and claim_mode == "manual_reimbursement"
        uses_treatment_benefit = calculation_basis in {
            "treatment_region_estimate",
            "treatment_region_benefit",
            "unknown",
        } or "上海" in payment_basis_source or "就医地" in payment_basis_source
        has_catalog_signal = bool(catalog_basis_source or treatment_region)
        hit = bool(remote_manual and uses_treatment_benefit and has_catalog_signal)

        return self._rule(
            "OP-R011",
            "参保地待遇与就医地目录混用核验",
            "rule:OP-R011:benefit_catalog_mix",
            "提示异地手工报销中就医地目录和参保地待遇不能混为同一计算口径。",
            [
                self._item(
                    "待遇目录口径判断",
                    (
                        f"参保地 {insured_region or '缺失'}，"
                        f"就医地 {treatment_region or '缺失'}，"
                        f"测算口径 {calculation_basis or '缺失'}，"
                        f"支付依据来源 {payment_basis_source or '缺失'}，"
                        f"目录依据来源 {catalog_basis_source or '缺失'}"
                    ),
                    "异地手工报销中，上海侧目录/预估比例不能直接替代北京参保地待遇测算",
                    hit,
                    LAYER_RISK,
                    "medium" if hit else "low",
                    (
                        "存在参保地待遇和就医地目录混用风险，应区分上海侧支付范围核验与北京参保地待遇测算。"
                        if hit
                        else "未形成参保地待遇与就医地目录混用提示。"
                    ),
                )
            ],
        )

    # ---- OP-R012: 政策版本冲突核验 ----

    def _policy_version_conflict(self, context: dict[str, Any]) -> RuleHit:
        """OP-R012 政策版本冲突核验。"""

        service_date = self._context_date(context, "visit_date")
        rule_policy_publish = self._context_date(context, "rule_policy_publish_date")
        latest_policy_publish = self._context_date(context, "latest_policy_publish_date")
        latest_policy_effective = self._context_date(context, "latest_policy_effective_date")
        recalled_versions = context.get("policy_recall_versions")
        version_count = len(recalled_versions) if isinstance(recalled_versions, list) else 0

        newer_policy = bool(
            rule_policy_publish
            and latest_policy_publish
            and latest_policy_publish > rule_policy_publish
        )
        applies_to_service = bool(
            service_date
            and latest_policy_effective
            and service_date >= latest_policy_effective
        )
        hit = bool(newer_policy and (applies_to_service or version_count >= 2))

        return self._rule(
            "OP-R012",
            "政策版本冲突核验",
            "rule:OP-R012:policy_version_conflict",
            "提示召回政策存在新旧版本或规则库引用政策早于最新生效政策时，需要核验政策生效时间和规则库版本。",
            [
                self._item(
                    "政策版本适用性判断",
                    (
                        f"服务日期 {service_date.isoformat() if service_date else '缺失'}，"
                        f"规则库政策发布日期 {rule_policy_publish.isoformat() if rule_policy_publish else '缺失'}，"
                        f"最新政策发布日期 {latest_policy_publish.isoformat() if latest_policy_publish else '缺失'}，"
                        f"最新政策生效日期 {latest_policy_effective.isoformat() if latest_policy_effective else '缺失'}，"
                        f"召回版本数 {version_count}"
                    ),
                    "服务日期晚于新政策生效日期，且规则库或召回结果存在更旧政策版本",
                    hit,
                    LAYER_RISK,
                    "medium" if hit else "low",
                    (
                        "存在政策版本冲突风险，建议优先核验服务日期对应的有效政策、废止状态和规则库版本。"
                        if hit
                        else "未形成政策版本冲突提示。"
                    ),
                )
            ],
        )

    # ---- 辅助方法 ----

    @staticmethod
    def _number(record: dict[str, Any], field: str) -> float | None:
        """从记录中安全提取数值。"""
        if field not in record:
            return None
        try:
            return float(record[field])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fmt_number(value: float | None) -> str:
        """格式化数值为字符串，缺失时返回"缺失"。"""
        if value is None:
            return "缺失"
        return f"{value:g}"

    @staticmethod
    def _fmt_money(value: float | None) -> str:
        """格式化金额为千分位字符串，缺失时返回"缺失"。"""
        if value is None:
            return "缺失"
        return f"{value:,.2f}"

    @staticmethod
    def _fmt_ratio(value: float | None) -> str:
        """格式化比例为百分号字符串，缺失时返回"缺失"。"""
        if value is None:
            return "缺失"
        return f"{value:.2%}"

    @staticmethod
    def _join_or_none(items: list[str]) -> str:
        """连接字符串列表，空列表时返回"未见相关提示"。"""
        return "、".join(items) if items else "未见相关提示"

    @staticmethod
    def _context_text(context: dict[str, Any] | None, key: str) -> str:
        """从安全案件上下文中提取展示用文本。"""
        if not isinstance(context, dict):
            return ""
        value = context.get(key)
        if value in (None, "", [], {}):
            return ""
        return str(value).strip()

    @staticmethod
    def _context_bool(context: dict[str, Any] | None, key: str) -> bool | None:
        """从安全案件上下文中提取布尔值。"""
        if not isinstance(context, dict) or key not in context:
            return None
        value = context.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    @staticmethod
    def _context_date(context: dict[str, Any] | None, key: str) -> date | None:
        """从安全案件上下文中提取 ISO 日期。"""
        if not isinstance(context, dict):
            return None
        value = context.get(key)
        if isinstance(value, date):
            return value
        if value in (None, ""):
            return None
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    @classmethod
    def _is_remote_context(cls, context: dict[str, Any] | None) -> bool:
        """判断参保地和就医地是否不同。"""
        insured = cls._context_text(context, "insured_region")
        treatment = cls._context_text(context, "treatment_region")
        return bool(insured and treatment and insured != treatment)

    def _safe_band(self, field: str, value: float | None) -> str | None:
        """安全查询分位档位。"""
        if value is None:
            return None
        try:
            return self.baseline.band(field, value)
        except KeyError:
            return None

    @staticmethod
    def _item(
        label: str,
        current_value: str,
        threshold: str,
        hit: bool,
        layer: str,
        severity: str,
        explanation: str,
    ) -> RuleCheckItem:
        """构造规则核验项。"""
        return RuleCheckItem(
            label=label,
            current_value=current_value,
            threshold=threshold,
            hit=hit,
            layer=layer,
            severity=severity,
            explanation=explanation,
        )

    def _rule(
        self,
        rule_id: str,
        rule_name: str,
        evidence_ref: str,
        business_explanation: str,
        items: list[RuleCheckItem],
        default_action: str = ACTION_MANUAL_REVIEW,
        force_non_risk: bool = False,
    ) -> RuleHit:
        """构造规则命中结果。

        根据核验项命中情况自动判断规则层级、动作和严重度。
        """
        triggered = [item for item in items if item.hit]
        risk_items = [
            item
            for item in triggered
            if item.layer in (LAYER_RISK, LAYER_STRONG)
        ]
        hit = bool(risk_items) and not force_non_risk

        if force_non_risk:
            layer = LAYER_DATA
            action = ACTION_DATA_QUALITY_CHECK
        elif any(item.layer == LAYER_STRONG for item in risk_items):
            layer = LAYER_STRONG
            action = ACTION_SPECIAL_AUDIT
        elif risk_items:
            layer = LAYER_RISK
            action = default_action
        elif triggered:
            layer = LAYER_DATA
            action = ACTION_DATA_QUALITY_CHECK
        else:
            layer = LAYER_RISK
            action = default_action

        severity = self._max_severity(triggered or items)
        current_value = "；".join(
            f"{item.label}: {item.current_value}" for item in items
        )
        threshold = "；".join(f"{item.label}: {item.threshold}" for item in items)
        if triggered:
            reason = "；".join(item.explanation for item in triggered)
        else:
            reason = "未形成需要单独核验的业务场景。"

        return RuleHit(
            rule_id=rule_id,
            rule_name=rule_name,
            hit=hit,
            severity=severity,
            reason=reason,
            evidence_ref=evidence_ref,
            version=RULE_POOL_VERSION,
            layer=layer,
            action=action,
            current_value=current_value,
            threshold=threshold,
            business_explanation=business_explanation,
            check_items=items,
        )

    @staticmethod
    def _materials_for_rules(rules: list[RuleHit]) -> list[str]:
        """根据命中的规则汇总需要补充的材料列表。"""
        by_rule = {
            "OP-R001": ["就诊流水", "同日就诊明细", "跨机构就诊说明"],
            "OP-R002": ["结算单", "基金支付明细", "费用明细"],
            "OP-R003": ["处方记录", "药品费用清单", "长期用药材料"],
            "OP-R004": ["检查治疗项目明细", "诊疗摘要材料"],
            "OP-R005": ["挂号记录", "就诊流程材料"],
            "OP-R006": ["待遇资格材料", "补助支付明细"],
            "OP-R007": ["字段校验记录", "申报材料说明"],
            "OP-R009": ["备案记录", "急诊证明材料", "参保地待遇政策", "就医地目录核验结果"],
            "OP-R010": ["急诊病历", "急诊挂号记录", "医院急诊证明", "抢救记录"],
            "OP-R011": ["费用明细", "参保地待遇政策", "就医地药品和诊疗项目目录"],
            "OP-R012": ["政策版本引用记录", "政策生效日期", "规则库版本记录"],
        }
        materials: list[str] = []
        for rule in rules:
            has_prompt = rule.hit or any(item.hit for item in rule.check_items)
            if not has_prompt:
                continue
            for material in by_rule.get(rule.rule_id, []):
                if material not in materials:
                    materials.append(material)
        return materials

    @staticmethod
    def _max_severity(items: list[RuleCheckItem]) -> str:
        """从核验项列表中取最高严重度。"""
        if not items:
            return "low"
        return max(items, key=lambda item: SEVERITY_RANK.get(item.severity, 0)).severity
