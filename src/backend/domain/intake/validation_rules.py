"""数据接入校验规则模块。

合并自：
- domain/ingest.py —— 上游完整脱敏宽表接入服务（IngestValidationService）
- domain/feature_validation.py —— 兼容单条统计特征输入的领域校验（FeatureValidationService）
"""

import re
from typing import Any

from ...core.exceptions import BusinessValidationError
from ...constants.ingest_fields import (
    INGEST_BINARY_FIELDS,
    INGEST_FORBIDDEN_FIELDS,
    INGEST_PROPORTION_FIELDS,
    INGEST_RECORD_FIELDS,
)
from ..audit.review.risk_engine import OperationalRiskSignalProvider
from .entities import (
    FeatureRecordInput,
    FeatureSchemaResponse,
    IngestRecordInput,
    IngestRecordSummary,
    IngestSchemaResponse,
)

# ============================================================================
# 来自 domain/feature_validation.py —— 统计特征校验常量
# ============================================================================

# 统计特征必填字段
REQUIRED_FIELDS: list[str] = [
    "月就诊次数_MAX",
    "月就诊医院数_MAX",
    "一天去两家医院的天数",
    "药品在总金额中的占比",
    "检查总费用在总金额占比",
    "治疗费用在总金额占比",
    "是否挂号",
    "ALL_SUM",
    "药品费发生金额_SUM",
    "检查费发生金额_SUM",
    "治疗费发生金额_SUM",
]

# 统计特征支持字段
SUPPORTED_FIELDS: list[str] = list(REQUIRED_FIELDS)

# 统计特征禁止字段
FORBIDDEN_FIELDS: list[str] = [
    "RES",
    "个人编码",
    "身份证",
    "姓名",
    "电话",
    "住址",
]

# 比例字段集合（值需在 0-1 之间）
PROPORTION_FIELDS: set[str] = {
    "药品在总金额中的占比",
    "检查总费用在总金额占比",
    "治疗费用在总金额占比",
}

# 金额字段集合（值不得为负数）
AMOUNT_FIELDS: set[str] = {
    "ALL_SUM",
    "药品费发生金额_SUM",
    "检查费发生金额_SUM",
    "治疗费发生金额_SUM",
}


# ============================================================================
# 来自 domain/feature_validation.py —— FeatureValidationService
# ============================================================================


class FeatureValidationError(BusinessValidationError):
    """字段校验失败。"""

    pass


class FeatureValidationService:
    """校验并标准化单条统计特征输入。"""

    def get_schema(self) -> FeatureSchemaResponse:
        """获取统计特征输入的模式定义。

        返回：
            包含必填字段、支持字段、禁止字段和示例的模式响应。
        """
        return FeatureSchemaResponse(
            required_fields=REQUIRED_FIELDS,
            supported_fields=SUPPORTED_FIELDS,
            forbidden_fields=FORBIDDEN_FIELDS,
            example_json=FeatureRecordInput(
                case_title="自定义单条统计记录",
                case_type="统计特征输入",
                features={
                    "月就诊次数_MAX": 12,
                    "月就诊医院数_MAX": 3,
                    "一天去两家医院的天数": 2,
                    "药品在总金额中的占比": 0.91,
                    "检查总费用在总金额占比": 0.08,
                    "治疗费用在总金额占比": 0.01,
                    "是否挂号": 0,
                    "ALL_SUM": 18000,
                    "药品费发生金额_SUM": 16380,
                    "检查费发生金额_SUM": 1440,
                    "治疗费发生金额_SUM": 180,
                },
            ),
        )

    def validate(self, record: FeatureRecordInput) -> dict[str, float]:
        """返回标准化后的数值特征。

        参数：
            record: 单条统计特征输入。

        返回：
            标准化后的数值特征字典。

        异常：
            FeatureValidationError: 校验失败时抛出。
        """
        errors: list[str] = []
        features = record.features or {}

        # 校验字段名称
        for key in features:
            self._validate_field_name(key, errors)

        # 校验必填字段
        for field in REQUIRED_FIELDS:
            if field not in features:
                errors.append(f"缺少必填字段：{field}")

        # 数值化并校验值域
        normalized: dict[str, float] = {}
        for key, value in features.items():
            if key not in SUPPORTED_FIELDS:
                if not self._is_forbidden_field(key):
                    errors.append(f"不支持的字段：{key}")
                continue
            number = self._to_number(key, value, errors)
            if number is not None:
                normalized[key] = number

        self._validate_values(normalized, errors)

        if errors:
            raise FeatureValidationError(errors)

        return normalized

    @staticmethod
    def _is_forbidden_field(key: str) -> bool:
        """判断字段是否在禁止字段列表中。

        参数：
            key: 字段名。

        返回：
            是否为禁止字段。
        """
        upper_key = key.upper()
        return any(item.upper() in upper_key for item in FORBIDDEN_FIELDS)

    def _validate_field_name(self, key: str, errors: list[str]) -> None:
        """校验单个字段名是否合法。"""
        if self._is_forbidden_field(key):
            errors.append(f"禁止字段不得进入输入：{key}")

    @staticmethod
    def _to_number(key: str, value: Any, errors: list[str]) -> float | None:
        """将字段值转换为浮点数。

        参数：
            key: 字段名。
            value: 原始值。
            errors: 错误列表，校验失败时追加错误信息。

        返回：
            浮点数值，转换失败时返回 None。
        """
        if isinstance(value, bool):
            errors.append(f"字段必须为数值：{key}")
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            errors.append(f"字段必须为数值：{key}")
            return None

    @staticmethod
    def _validate_values(features: dict[str, float], errors: list[str]) -> None:
        """校验标准化后的特征值。

        参数：
            features: 标准化后的特征字典。
            errors: 错误列表，校验失败时追加错误信息。
        """
        for key, value in features.items():
            if key in PROPORTION_FIELDS and not 0 <= value <= 1:
                errors.append(f"比例字段必须在 0 到 1 之间：{key}")
            if key in AMOUNT_FIELDS and value < 0:
                errors.append(f"金额字段不得为负数：{key}")
            if key == "是否挂号" and value not in (0, 1):
                errors.append("是否挂号只能为 0 或 1")
            if key not in PROPORTION_FIELDS and key not in AMOUNT_FIELDS and value < 0:
                errors.append(f"统计字段不得为负数：{key}")


# ============================================================================
# 来自 domain/ingest.py —— IngestValidationService
# ============================================================================


class IngestValidationError(BusinessValidationError):
    """上游脱敏宽表记录校验失败。"""

    pass


class IngestValidationService:
    """校验完整脱敏宽表，并提取稽核所需特征视图。"""

    person_ref_pattern = re.compile(r"^SIM_PERSON_\d{6}$")

    def get_schema(self) -> IngestSchemaResponse:
        """获取完整脱敏宽表输入的模式定义。

        返回：
            包含必填字段、支持字段、禁止字段和示例的模式响应。
        """
        return IngestSchemaResponse(
            required_fields=INGEST_RECORD_FIELDS,
            supported_fields=INGEST_RECORD_FIELDS,
            forbidden_fields=INGEST_FORBIDDEN_FIELDS,
            example_json=IngestRecordInput(
                case_title="上游申报宽表记录",
                case_type="上游宽表记录接入",
                source_system="医保结算申报系统",
                record_version="claim-wide-v1",
                record=self._example_record(),
            ),
        )

    def validate(self, body: IngestRecordInput) -> dict[str, float | str]:
        """校验单条完整脱敏宽表记录。

        参数：
            body: 上游宽表记录输入。

        返回：
            校验通过并标准化后的记录字典。

        异常：
            IngestValidationError: 校验失败时抛出。
        """
        errors: list[str] = []
        record = body.record or {}

        # 校验禁止字段
        for key in record:
            if self._is_forbidden_field(key):
                errors.append(f"禁止字段不得进入输入：{key}")
            elif key not in INGEST_RECORD_FIELDS:
                errors.append(f"不支持的完整宽表字段：{key}")

        # 校验必填字段
        for field in INGEST_RECORD_FIELDS:
            if field not in record:
                errors.append(f"缺少完整宽表字段：{field}")

        normalized: dict[str, float | str] = {}
        if "个人编码" in record:
            person_ref = str(record["个人编码"]).strip()
            if not self.person_ref_pattern.fullmatch(person_ref):
                errors.append("个人编码必须为合成脱敏编码，格式如 SIM_PERSON_000001")
            else:
                normalized["个人编码"] = person_ref

        # 数值化处理
        for field in INGEST_RECORD_FIELDS:
            if field == "个人编码" or field not in record:
                continue
            value = self._to_number(field, record[field], errors)
            if value is not None:
                normalized[field] = value

        self._validate_values(normalized, errors)

        if errors:
            raise IngestValidationError(errors)

        return normalized

    def extract_audit_features(
        self, safe_record: dict[str, float | str]
    ) -> dict[str, float]:
        """从校验通过的记录中提取稽核所需特征。

        参数：
            safe_record: 校验通过的记录字典。

        返回：
            仅包含风险评分所需字段的数值特征字典。
        """
        return {field: float(safe_record[field]) for field in REQUIRED_FIELDS}

    def summarize(
        self, safe_record: dict[str, float | str]
    ) -> IngestRecordSummary:
        """生成记录摘要，包含风险评分。

        参数：
            safe_record: 校验通过的记录字典。

        返回：
            包含风险等级和分值的记录摘要。
        """
        subject_ref = str(safe_record["个人编码"])
        scoring = OperationalRiskSignalProvider().score(
            {
                "月就诊次数_MAX": float(safe_record["月就诊次数_MAX"]),
                "月就诊医院数_MAX": float(safe_record["月就诊医院数_MAX"]),
                "一天去两家医院的天数": float(safe_record["一天去两家医院的天数"]),
                "药品在总金额中的占比": float(safe_record["药品在总金额中的占比"]),
                "检查总费用在总金额占比": float(safe_record["检查总费用在总金额占比"]),
                "治疗费用在总金额占比": float(safe_record["治疗费用在总金额占比"]),
                "是否挂号": float(safe_record["是否挂号"]),
                "ALL_SUM": float(safe_record["ALL_SUM"]),
                "药品费发生金额_SUM": float(safe_record["药品费发生金额_SUM"]),
                "检查费发生金额_SUM": float(safe_record["检查费发生金额_SUM"]),
                "治疗费发生金额_SUM": float(safe_record["治疗费发生金额_SUM"]),
            },
            source_record=safe_record,
        )
        return IngestRecordSummary(
            record_id=subject_ref,
            subject_ref=subject_ref,
            risk_level=scoring.risk_level,
            risk_score=scoring.risk_score,
        )

    @staticmethod
    def _is_forbidden_field(key: str) -> bool:
        """判断字段是否在禁止字段列表中。

        参数：
            key: 字段名。

        返回：
            是否为禁止字段。
        """
        upper_key = key.upper()
        return any(item.upper() in upper_key for item in INGEST_FORBIDDEN_FIELDS)

    @staticmethod
    def _to_number(key: str, value: Any, errors: list[str]) -> float | None:
        """将字段值转换为浮点数。

        参数：
            key: 字段名。
            value: 原始值。
            errors: 错误列表。

        返回：
            浮点数值，转换失败时返回 None。
        """
        if isinstance(value, bool):
            errors.append(f"字段必须为数值：{key}")
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            errors.append(f"字段必须为数值：{key}")
            return None

    @staticmethod
    def _validate_values(
        safe_record: dict[str, float | str], errors: list[str]
    ) -> None:
        """校验标准化后的宽表记录值。

        参数：
            safe_record: 标准化后的记录字典。
            errors: 错误列表。
        """
        for key, value in safe_record.items():
            if key == "个人编码":
                continue
            number = float(value)
            if number < 0:
                errors.append(f"完整宽表数值不得为负数：{key}")
            if key in INGEST_PROPORTION_FIELDS and number > 1:
                errors.append(f"比例字段必须在 0 到 1 之间：{key}")
            if key in INGEST_BINARY_FIELDS and number not in (0, 1):
                errors.append(f"标志字段只能为 0 或 1：{key}")

    @staticmethod
    def _example_record() -> dict[str, float | str]:
        """生成示例宽表记录。

        返回：
            包含典型字段值的示例字典。
        """
        record: dict[str, float | str] = {
            field: 0 for field in INGEST_RECORD_FIELDS
        }
        record.update(
            {
                "个人编码": "SIM_PERSON_000001",
                "一天去两家医院的天数": 2,
                "就诊的月数": 3,
                "月就诊天数_MAX": 14,
                "月就诊天数_AVG": 8.2,
                "月就诊医院数_MAX": 3,
                "月就诊医院数_AVG": 2.1,
                "就诊次数_SUM": 28,
                "月就诊次数_MAX": 12,
                "月就诊次数_AVG": 9.3,
                "月统筹金额_MAX": 7200,
                "月统筹金额_AVG": 4600,
                "月药品金额_MAX": 6500,
                "月药品金额_AVG": 4300,
                "医院编码_NN": 3,
                "顺序号_NN": 28,
                "交易时间DD_NN": 18,
                "交易时间YYYY_NN": 1,
                "交易时间YYYYMM_NN": 3,
                "个人账户金额_SUM": 1200,
                "统筹支付金额_SUM": 13200,
                "ALL_SUM": 18000,
                "药品费发生金额_SUM": 16380,
                "药品费申报金额_SUM": 16380,
                "检查费发生金额_SUM": 1440,
                "检查费申报金额_SUM": 1440,
                "治疗费发生金额_SUM": 180,
                "治疗费申报金额_SUM": 180,
                "本次审批金额_SUM": 13200,
                "药品在总金额中的占比": 0.91,
                "个人支付的药品占比": 0.07,
                "检查总费用在总金额占比": 0.08,
                "个人支付检查费用占比": 0.02,
                "治疗费用在总金额占比": 0.01,
                "个人支付治疗费用占比": 0.01,
                "是否挂号": 0,
            }
        )
        return record
