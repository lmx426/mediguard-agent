"""综合风险提示强度引擎。

源自 domain/risk.py。

风险提示强度用于人工核验分流，不做欺诈、拒付或处罚结论。
"""

from typing import Any

from src.backend.constants.ingest_fields import INGEST_RECORD_FIELDS

from .entities import (
    FraudScreeningSignal,
    RiskScoreBreakdown,
    RiskScoreComponent,
    RuleHit,
)
from .rule_evaluator import RuleBaselineService
from ...intake.entities import ScoringResult

# 规则层级常量
LAYER_DATA = "data_quality_or_applicability"
LAYER_STRONG = "strong_review_signal"

# 严重度排序权重
SEVERITY_RANK = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

# 风险等级中文标签
LEVEL_LABELS = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}

MODEL_WARNING_MAX_SCORE = 70
RULE_CHECK_MAX_SCORE = 15
PEER_DEVIATION_MAX_SCORE = 10
DATA_FLOW_MAX_SCORE = 5

FIELD_DISPLAY_NAMES = {
    "ALL_SUM": "申报总费用",
    "本次审批金额_SUM": "本次审批金额",
    "统筹支付金额_SUM": "统筹支付金额",
    "基本统筹基金支付金额_SUM": "基本统筹基金支付金额",
    "个人账户金额_SUM": "个人账户金额",
    "基本个人账户支付_SUM": "个人账户支付金额",
    "药品费发生金额_SUM": "药品费用",
    "药品费申报金额_SUM": "药品申报金额",
    "贵重药品发生金额_SUM": "贵重药品费用",
    "检查费发生金额_SUM": "检查费用",
    "检查费申报金额_SUM": "检查申报金额",
    "治疗费发生金额_SUM": "治疗费用",
    "治疗费申报金额_SUM": "治疗申报金额",
    "月就诊次数_MAX": "月最高就诊次数",
    "月就诊医院数_MAX": "月最高就诊医院数",
    "一天去两家医院的天数": "单日跨医院就诊天数",
    "药品在总金额中的占比": "药品占比",
    "检查总费用在总金额占比": "检查占比",
    "治疗费用在总金额占比": "治疗占比",
    "是否挂号": "挂号状态",
}


class OperationalRiskSignalProvider:
    """按模型、规则、同类偏离和数据流程生成综合风险提示强度。"""

    source = "综合风险提示强度计算"
    evidence_ref = "risk_score:composite:v1.0"

    def __init__(self, baseline_service: RuleBaselineService | None = None) -> None:
        """初始化风险信号提供者。

        参数：
            baseline_service: 规则基线服务，用于分位阈值查询。为 None 时使用默认基线。
        """
        self.baseline = baseline_service or RuleBaselineService()

    def score(
        self,
        features: dict[str, float],
        source_record: dict[str, float | str] | None = None,
        rules: list[RuleHit] | None = None,
        fraud_screening: FraudScreeningSignal | None = None,
    ) -> ScoringResult:
        """计算综合风险提示强度。

        参数：
            features: 脱敏统计特征键值表。
            source_record: 完整脱敏宽表记录，可选。
            rules: 规则命中结果列表，可选。
            fraud_screening: 独立模型识别预警信号，可选。

        返回：
            包含风险分、等级和分项贡献的评分结果。
        """
        record: dict[str, Any] = source_record or features
        checked_rules = rules or []
        fraud_signal = fraud_screening or FraudScreeningSignal()

        # 计算四项贡献
        components = [
            self._model_component(fraud_signal),
            self._rule_component(checked_rules),
            self._peer_component(record),
            self._data_flow_component(record, checked_rules),
        ]

        # 汇总并应用封顶规则
        raw_total = min(100, sum(component.score for component in components))
        total = raw_total
        cap_note = None
        if (
            fraud_signal.result == "not_suspected"
            and total >= 70
            and not self._has_strong_rule(checked_rules)
        ):
            total = 69
            cap_note = "模型未触发预警时，综合风险默认不超过中风险；重点规则命中除外。"

        level = self._level(total)
        breakdown = RiskScoreBreakdown(
            total_score=total,
            max_score=100,
            display_score=f"{total} / 100",
            level=level,
            level_label=LEVEL_LABELS[level],
            components=components,
            cap_note=cap_note,
        )
        reasons = [
            f"{component.label} {component.score}/{component.max_score}：{component.summary}"
            for component in components
        ]
        if cap_note:
            reasons.append(cap_note)

        return ScoringResult(
            risk_score=round(total / 100, 2),
            risk_level=level,
            source=self.source,
            evidence_ref=self.evidence_ref,
            reasons=reasons,
            risk_score_breakdown=breakdown,
        )

    @staticmethod
    def _model_component(fraud_signal: FraudScreeningSignal) -> RiskScoreComponent:
        """计算模型识别预警贡献项。

        参数：
            fraud_signal: 独立模型识别预警信号。

        返回：
            模型贡献项。
        """
        max_score = MODEL_WARNING_MAX_SCORE
        probability = fraud_signal.probability
        if probability is not None:
            score = min(max_score, max(0, round(probability * max_score)))
            probability_text = f"{probability:.1%}"
            summary = f"模型返回{fraud_signal.label}，预警概率 {probability_text}。"
            details = [
                f"预警概率：{probability_text}",
                f"模型贡献按预警概率折算，满分 {max_score} 分。",
            ]
        elif fraud_signal.result == "suspected":
            score = max_score
            summary = "模型返回有预警，未返回预警概率。"
            details = ["模型已触发预警，未返回可展示概率。"]
        elif fraud_signal.result == "not_suspected":
            score = 0
            summary = "模型返回无预警，未返回预警概率。"
            details = ["模型未触发预警，且未返回可展示概率。"]
        else:
            score = 0
            summary = "模型识别预警未接入，未参与本项贡献。"
            details = ["当前记录未获得可用的模型预警概率。"]

        return RiskScoreComponent(
            key="model_warning",
            label="模型识别预警",
            score=score,
            max_score=max_score,
            summary=summary,
            details=details,
            source_detail=(
                "XGB 异常申报识别模型；准确率 97.27%，"
                "召回率 98.65%，F1 0.975；样本与当前业务场景一致。"
                if fraud_signal.result != "not_available"
                else "模型识别预警未接入。"
            ),
        )

    @staticmethod
    def _rule_component(rules: list[RuleHit]) -> RiskScoreComponent:
        """计算规则核验贡献项。

        参数：
            rules: 规则命中结果列表。

        返回：
            规则核验贡献项。
        """
        strong_rules = [rule for rule in rules if rule.hit and rule.layer == LAYER_STRONG]
        risk_rules = [rule for rule in rules if rule.hit and rule.layer != LAYER_STRONG]
        prompt_rules = [
            rule
            for rule in rules
            if not rule.hit and any(item.hit and item.layer == LAYER_DATA for item in rule.check_items)
        ]
        raw_score = len(strong_rules) * 6 + len(risk_rules) * 4 + len(prompt_rules) * 1
        score = min(RULE_CHECK_MAX_SCORE, raw_score)
        if score:
            summary = (
                f"命中 {len(strong_rules) + len(risk_rules)} 类业务核验，"
                f"其中 {len(strong_rules)} 类为重点核验。"
            )
        else:
            summary = "当前固定业务规则未形成需要单独核验的线索。"
        details = [
            f"重点核验规则：{len(strong_rules)} 类",
            f"普通命中规则：{len(risk_rules)} 类",
            f"提示类规则：{len(prompt_rules)} 类",
        ]
        if score < raw_score:
            details.append(f"规则核验贡献已按 {RULE_CHECK_MAX_SCORE} 分封顶。")
        return RiskScoreComponent(
            key="rule_check",
            label="规则核验",
            score=score,
            max_score=RULE_CHECK_MAX_SCORE,
            summary=summary,
            details=details,
            source_detail="固定医保业务核验规则池。",
        )

    def _peer_component(self, record: dict[str, Any]) -> RiskScoreComponent:
        """计算同类偏离贡献项。

        参数：
            record: 脱敏宽表记录。

        返回：
            同类偏离贡献项。
        """
        scanned_count, deviations = self._baseline_deviations(record)
        raw_score = sum(points for _, points, _ in deviations)
        score = min(PEER_DEVIATION_MAX_SCORE, raw_score)
        high_count = len([item for item in deviations if item[2] in {"p98", "p99"}])
        details = [detail for detail, _, _ in deviations[:8]]

        if deviations:
            summary = (
                f"已扫描 {scanned_count} 项安全申报字段，"
                f"{len(deviations)} 项高于同类常规范围，"
                f"其中 {high_count} 项处于明显高位。"
            )
        else:
            summary = f"已扫描 {scanned_count} 项安全申报字段，未形成明显同类偏离。"
            details = ["可比字段未进入当前关注区间。"]
        if score == PEER_DEVIATION_MAX_SCORE and len(deviations) > 5:
            details.append(f"同类偏离贡献已按 {PEER_DEVIATION_MAX_SCORE} 分封顶。")
        return RiskScoreComponent(
            key="peer_deviation",
            label="同类偏离",
            score=score,
            max_score=PEER_DEVIATION_MAX_SCORE,
            summary=summary,
            details=details,
            source_detail=(
                f"脱敏历史申报样本聚合分位参考，覆盖 {self.baseline.field_count} 项安全字段，"
                "不含训练标签、身份字段或行级数据。"
            ),
        )

    def _data_flow_component(
        self,
        record: dict[str, Any],
        rules: list[RuleHit],
    ) -> RiskScoreComponent:
        """计算数据/流程完整性贡献项。

        参数：
            record: 脱敏宽表记录。
            rules: 规则命中结果列表。

        返回：
            数据/流程完整性贡献项。
        """
        details: list[str] = []
        score = 0
        registered = self._number(record, "是否挂号")
        amount_band = self._safe_band("ALL_SUM", self._number(record, "ALL_SUM"))
        visit_max = self._number(record, "月就诊次数_MAX") or 0.0
        hospital_max = self._number(record, "月就诊医院数_MAX") or 0.0
        if registered == 0 and (amount_band or visit_max >= 10 or hospital_max >= 3):
            score += 3
            details.append("未挂号状态叠加金额、频次或多机构线索。")

        data_rules = [
            rule
            for rule in rules
            if any(item.hit and item.layer == LAYER_DATA for item in rule.check_items)
        ]
        if data_rules:
            data_points = min(2, len(data_rules))
            score += data_points
            details.append(f"存在 {len(data_rules)} 类数据质量或适用范围提示。")
        if any(rule.rule_id == "OP-R008" for rule in data_rules):
            score += 1
            details.append("材料补充核验提示已生成。")

        score = min(DATA_FLOW_MAX_SCORE, score)
        if details:
            summary = "挂号流程、字段适用范围或材料完整性需结合材料确认。"
        else:
            summary = "当前数据流程未形成额外完整性提示。"
            details = ["挂号状态、字段适用范围和材料提示未形成额外贡献。"]
        return RiskScoreComponent(
            key="data_flow",
            label="数据/流程完整性",
            score=score,
            max_score=DATA_FLOW_MAX_SCORE,
            summary=summary,
            details=details,
            source_detail="案件接入校验、挂号流程和业务材料提示。",
        )

    def _baseline_deviations(self, record: dict[str, Any]) -> tuple[int, list[tuple[str, int, str]]]:
        """检查安全申报字段相对于历史基线的偏离。

        参数：
            record: 脱敏宽表记录。

        返回：
            可比字段数，以及 (描述, 分数, 分位档位) 元组列表。
        """
        deviations: list[tuple[str, int, str]] = []
        scanned_count = 0
        safe_fields = [field for field in INGEST_RECORD_FIELDS if field != "个人编码"]
        for field_name in safe_fields:
            value = self._number(record, field_name)
            if value is None or field_name not in self.baseline.fields:
                continue
            scanned_count += 1
            band = self._safe_band(field_name, value)
            if not band:
                continue
            label = FIELD_DISPLAY_NAMES.get(field_name, field_name)
            value_text = self._format_field_value(field_name, value)
            if band in ("p98", "p99"):
                deviations.append(
                    (
                        f"{label}处于同类明显高位（{band.upper()}），当前值 {value_text}。",
                        3,
                        band,
                    )
                )
            elif band == "p95":
                deviations.append(
                    (
                        f"{label}高于同类常规范围（P95），当前值 {value_text}。",
                        2,
                        band,
                    )
                )
        deviations.sort(key=lambda item: (-self._band_rank(item[2]), -item[1], item[0]))
        return scanned_count, deviations

    @staticmethod
    def _has_strong_rule(rules: list[RuleHit]) -> bool:
        """判断是否存在强复核规则命中。

        参数：
            rules: 规则命中结果列表。

        返回：
            是否存在重点核验规则命中。
        """
        return any(rule.hit and rule.layer == LAYER_STRONG for rule in rules)

    @staticmethod
    def _level(score: int) -> str:
        """根据分数确定风险等级。

        参数：
            score: 0-100 的整数分数。

        返回：
            low | medium | high。
        """
        if score >= 70:
            return "high"
        if score >= 30:
            return "medium"
        return "low"

    @staticmethod
    def _band_rank(band: str | None) -> int:
        """分位档位排序权重。

        参数：
            band: 分位档位字符串（p95, p98, p99）或 None。

        返回：
            排序权重整数。
        """
        return {"p95": 1, "p98": 2, "p99": 3}.get(band or "", 0)

    @staticmethod
    def _number(record: dict[str, Any], field: str) -> float | None:
        """从记录中安全提取数值。

        参数：
            record: 键值对记录。
            field: 字段名。

        返回：
            浮点数值，不存在或无法转换时返回 None。
        """
        if field not in record:
            return None
        try:
            return float(record[field])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_field_value(field: str, value: float) -> str:
        """按字段业务类型格式化当前值。"""
        if "占比" in field:
            return f"{value:.2%}"
        if any(token in field for token in ("金额", "费用", "费", "支付", "账户", "报销", "补助", "救助")):
            return f"{value:,.2f}"
        if "比例" in field:
            return f"{value:.2%}"
        return f"{value:g}"

    def _safe_band(self, field: str, value: float | None) -> str | None:
        """安全查询分位档位。

        参数：
            field: 字段名。
            value: 字段值。

        返回：
            分位档位字符串，不存在时返回 None。
        """
        if value is None:
            return None
        try:
            return self.baseline.band(field, value)
        except KeyError:
            return None


class XGBoostScoringProvider:
    """预留生产模型接入口。

    当前不启用：现有模型链路缺少可靠单条标准化契约。
    """

    def score(self, features: dict[str, float]) -> ScoringResult:
        """预留评分接口，尚未配置。"""
        raise NotImplementedError("XGBoost 单条评分接口尚未配置")
