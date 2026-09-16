"""本地欺诈模型推理编排用例。

该用例负责独立的 fraud_screening 预警与 PCA 10 个综合指标分。
模型依赖、模型文件或预处理资产缺失时返回未接入，不阻塞主业务稽核流程。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any
import warnings

from ...constants.fraud_model_fields import (
    FRAUD_MODEL_FEATURE_FIELDS,
    PCA_FEATURE_SCORE_NAMES,
)
from ...domain.audit.review.entities import FraudScreeningSignal, PCAFeatureScore
from ..ports.model_assets import ModelAssetsLoader


@dataclass(frozen=True)
class FraudModelStatus:
    """模型运行状态，用于 health 排障，不在前端主流程暴露。"""

    enabled: bool
    ready: bool
    status: str

    def model_dump(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "status": self.status,
        }


@dataclass(frozen=True)
class FraudModelResult:
    """单条模型推理结果。"""

    fraud_screening: FraudScreeningSignal
    pca_feature_scores: list[PCAFeatureScore]


@dataclass(frozen=True)
class _ScalerFeature:
    """标准化特征元数据。"""

    name: str
    mean: float
    std: float


class FraudModelService:
    """RF-ERFE-PCA-XGBoost 本地模型适配器。

    将模型资产加载委托给 ModelAssetsLoader 接口，
    自身只负责推理编排和业务校验。
    """

    def __init__(
        self,
        assets_loader: ModelAssetsLoader,
        enabled: bool = True,
    ) -> None:
        """初始化欺诈模型服务。

        Args:
            assets_loader: 模型资产加载器，负责从存储介质读取模型文件。
            enabled: 是否启用模型推理。
        """

        self._assets_loader = assets_loader
        self.enabled = enabled
        self._loaded = False
        self._status = FraudModelStatus(enabled=enabled, ready=False, status="not_loaded")
        self._model: Any | None = None
        self._pandas: Any | None = None
        self._scaler: list[_ScalerFeature] = []
        self._pca_matrix: list[list[float]] = []
        self._load_lock = RLock()
        self._predict_lock = Lock()

    def status(self) -> FraudModelStatus:
        """返回当前模型状态；首次调用会尝试加载模型资产。"""

        self._ensure_loaded()
        return self._status

    def analyze(self, record: dict[str, float | str] | None) -> FraudModelResult:
        """对单条脱敏宽表记录进行独立欺诈模型预警。

        Args:
            record: 已通过接入校验的脱敏 81 字段记录；不得包含 RES。

        Returns:
            FraudModelResult: 包含欺诈预警信号和 PCA 分项评分。
        """

        if not record:
            return self._unavailable("missing_source_record")

        self._ensure_loaded()
        if not self._status.ready:
            return self._unavailable(self._status.status)

        try:
            raw_values = self._extract_feature_values(record)
            scaled_values = self._scale(raw_values)
            pca_values = self._project_pca(scaled_values)
            model_input = self._build_model_input(pca_values)
            with self._predict_lock:
                fraud_screening = self._predict(model_input)
            pca_scores = [
                PCAFeatureScore(name=name, score=round(score, 6))
                for name, score in zip(PCA_FEATURE_SCORE_NAMES, pca_values, strict=True)
            ]
            return FraudModelResult(
                fraud_screening=fraud_screening,
                pca_feature_scores=pca_scores,
            )
        except (KeyError, TypeError, ValueError):
            return self._unavailable("invalid_model_input")
        except Exception:
            return self._unavailable("prediction_failed")

    def _ensure_loaded(self) -> None:
        """尝试加载模型资产，将状态记录到 _status。"""

        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            if not self.enabled:
                self._status = FraudModelStatus(enabled=False, ready=False, status="disabled")
                self._loaded = True
                return

            # 检查依赖
            try:
                importlib.import_module("xgboost")
                importlib.import_module("sklearn")
            except ImportError:
                self._status = FraudModelStatus(enabled=True, ready=False, status="dependency_missing")
                self._loaded = True
                return

            try:
                self._pandas = importlib.import_module("pandas")
            except ImportError:
                self._pandas = None

            try:
                # 通过资产加载器读取模型文件
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*If you are loading a serialized model.*",
                        category=UserWarning,
                    )
                    self._model = self._assets_loader.load_model()
                # 加载并解析标准化统计
                scaler_data = self._assets_loader.load_scaler_stats()
                self._scaler = self._parse_scaler(scaler_data)
                # 加载 PCA 矩阵
                self._pca_matrix = self._assets_loader.load_pca_matrix()
            except Exception:
                self._status = FraudModelStatus(enabled=True, ready=False, status="model_load_failed")
                self._loaded = True
                return

            self._status = FraudModelStatus(enabled=True, ready=True, status="ready")
            self._loaded = True

    @staticmethod
    def _parse_scaler(payload: dict[str, Any]) -> list[_ScalerFeature]:
        """从加载的 scaler 数据中解析标准化特征列表。

        Args:
            payload: load_scaler_stats 返回的字典。

        Returns:
            按 FRAUD_MODEL_FEATURE_FIELDS 顺序排列的 _ScalerFeature 列表。

        Raises:
            ValueError: 特征名称、顺序或数量不匹配，或标准差无效。
        """

        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError("scaler_stats.features must be a list")

        result: list[_ScalerFeature] = []
        for expected_name, item in zip(FRAUD_MODEL_FEATURE_FIELDS, features, strict=True):
            if item.get("name") != expected_name:
                raise ValueError("scaler feature order mismatch")
            mean = float(item["mean"])
            std = float(item["std"])
            if std <= 0:
                raise ValueError(f"invalid std for {expected_name}")
            result.append(_ScalerFeature(name=expected_name, mean=mean, std=std))
        if len(result) != len(FRAUD_MODEL_FEATURE_FIELDS):
            raise ValueError("scaler feature count mismatch")
        return result

    @staticmethod
    def _extract_feature_values(record: dict[str, float | str]) -> list[float]:
        """从记录中按预定义顺序提取特征值。"""
        values: list[float] = []
        for field in FRAUD_MODEL_FEATURE_FIELDS:
            values.append(float(record[field]))
        return values

    def _scale(self, values: list[float]) -> list[float]:
        """Z-score 标准化。"""
        return [
            (value - scaler.mean) / scaler.std
            for value, scaler in zip(values, self._scaler, strict=True)
        ]

    def _project_pca(self, scaled_values: list[float]) -> list[float]:
        """PCA 投影：将 39 个标准化特征降维为 10 个 PCA 综合指标分。"""
        scores: list[float] = []
        for column_index in range(len(PCA_FEATURE_SCORE_NAMES)):
            scores.append(
                sum(
                    scaled_values[row_index] * self._pca_matrix[row_index][column_index]
                    for row_index in range(len(FRAUD_MODEL_FEATURE_FIELDS))
                )
            )
        return scores

    def _build_model_input(self, pca_values: list[float]) -> Any:
        """构建模型输入（优先使用 pandas DataFrame）。"""
        columns = [f"l{index}" for index in range(len(PCA_FEATURE_SCORE_NAMES))]
        if self._pandas is not None:
            return self._pandas.DataFrame([pca_values], columns=columns)
        return [pca_values]

    def _predict(self, model_input: Any) -> FraudScreeningSignal:
        """执行模型预测并构建 FraudScreeningSignal。"""
        if self._model is None:
            raise ValueError("model not loaded")

        prediction = self._first_value(self._model.predict(model_input))
        result = "suspected" if self._is_fraud_class(prediction) else "not_suspected"
        probability: float | None = None
        if hasattr(self._model, "predict_proba"):
            try:
                probability = round(
                    self._fraud_probability(self._model.predict_proba(model_input)),
                    6,
                )
            except (TypeError, ValueError, IndexError):
                probability = None

        return FraudScreeningSignal(
            result=result,
            label="有预警" if result == "suspected" else "无预警",
            source="模型识别预警",
            evidence_ref="fraud_screening:xgboost:rf-erfe-pca",
            reason="本地 RF-ERFE-PCA-XGBoost 模型输出。",
            probability=probability,
        )

    def _fraud_probability(self, probabilities: Any) -> float:
        """从概率数组中提取欺诈类别的概率。"""
        row = list(self._first_value(probabilities))
        classes = list(getattr(self._model, "classes_", [])) if self._model is not None else []
        fraud_index = self._fraud_class_index(classes, len(row))
        probability = float(row[fraud_index])
        if probability < 0 or probability > 1:
            raise ValueError("invalid probability")
        return probability

    @staticmethod
    def _fraud_class_index(classes: list[Any], probability_count: int) -> int:
        """查找标签为 '1'（欺诈）的类别索引。"""
        for index, value in enumerate(classes):
            if str(value) == "1":
                return index
        if probability_count >= 2:
            return 1
        return 0

    @staticmethod
    def _is_fraud_class(value: Any) -> bool:
        """判断预测值是否属于欺诈类别。"""
        try:
            return int(float(value)) == 1
        except (TypeError, ValueError):
            return str(value) == "1"

    @staticmethod
    def _first_value(values: Any) -> Any:
        """从数组/可迭代对象中提取第一个值。"""
        try:
            return values[0]
        except TypeError:
            return list(values)[0]

    @staticmethod
    def _unavailable(status: str) -> FraudModelResult:
        """构造模型不可用时的默认结果。"""
        return FraudModelResult(
            fraud_screening=FraudScreeningSignal(
                result="not_available",
                label="未接入",
                source="模型识别预警未接入",
                evidence_ref="fraud_screening:binary:not-configured",
                reason=status,
                probability=None,
            ),
            pca_feature_scores=[],
        )


class PrecomputedFraudModelService:
    """Serve safe showcase snapshots without loading model binaries."""

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path
        self._records: dict[str, FraudModelResult] | None = None

    def status(self) -> FraudModelStatus:
        try:
            self._ensure_loaded()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return FraudModelStatus(
                enabled=True,
                ready=False,
                status="showcase_fixture_invalid",
            )
        return FraudModelStatus(
            enabled=True,
            ready=True,
            status="showcase_precomputed",
        )

    def analyze(self, record: dict[str, float | str] | None) -> FraudModelResult:
        if not record:
            return FraudModelService._unavailable("missing_source_record")
        try:
            self._ensure_loaded()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return FraudModelService._unavailable("showcase_fixture_invalid")
        record_id = str(record.get("个人编码") or "")
        result = (self._records or {}).get(record_id)
        if result is None:
            return FraudModelService._unavailable("showcase_record_not_configured")
        return result

    def _ensure_loaded(self) -> None:
        if self._records is not None:
            return
        payload = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        raw_records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(raw_records, dict) or not raw_records:
            raise ValueError("showcase model fixture must contain records")
        records: dict[str, FraudModelResult] = {}
        for record_id, raw in raw_records.items():
            if not isinstance(raw, dict):
                raise ValueError("showcase model record must be an object")
            records[str(record_id)] = FraudModelResult(
                fraud_screening=FraudScreeningSignal.model_validate(
                    raw.get("fraud_screening")
                ),
                pca_feature_scores=[
                    PCAFeatureScore.model_validate(item)
                    for item in raw.get("pca_feature_scores", [])
                ],
            )
        self._records = records
