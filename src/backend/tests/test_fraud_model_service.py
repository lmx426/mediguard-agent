"""欺诈模型服务单元测试。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from src.backend.constants.fraud_model_fields import (
    FRAUD_MODEL_FEATURE_FIELDS,
    PCA_FEATURE_SCORE_NAMES,
)
from src.backend.application.fraud.run_screening_uc import (
    FraudModelService,
    FraudModelStatus,
    _ScalerFeature,
)


class _ProbabilityModelPredictsNormal:
    classes_ = [0, 1]

    def predict(self, _model_input):
        return [0]

    def predict_proba(self, _model_input):
        return [[0.03, 0.97]]


class _BinaryOnlyModelPredictsFraud:
    def predict(self, _model_input):
        return [1]


class _MissingAssetLoader:
    def load_model(self):
        raise FileNotFoundError("missing model")


class _CountingAssetLoader:
    def __init__(self) -> None:
        self.load_model_count = 0
        self.load_scaler_count = 0
        self.load_pca_count = 0
        self._lock = Lock()

    def load_model(self):
        with self._lock:
            self.load_model_count += 1
        sleep(0.03)
        return _BinaryOnlyModelPredictsFraud()

    def load_scaler_stats(self):
        with self._lock:
            self.load_scaler_count += 1
        return {
            "features": [
                {"name": field, "mean": 0.0, "std": 1.0}
                for field in FRAUD_MODEL_FEATURE_FIELDS
            ]
        }

    def load_pca_matrix(self):
        with self._lock:
            self.load_pca_count += 1
        return [
            [
                1.0 if column_index == row_index % len(PCA_FEATURE_SCORE_NAMES) else 0.0
                for column_index in range(len(PCA_FEATURE_SCORE_NAMES))
            ]
            for row_index in range(len(FRAUD_MODEL_FEATURE_FIELDS))
        ]


def test_predict_uses_model_binary_output_and_probability_is_display_only(tmp_path):
    service = FraudModelService(tmp_path)
    service._model = _ProbabilityModelPredictsNormal()

    signal = service._predict([[0.0] * 10])

    assert signal.result == "not_suspected"
    assert signal.label == "无预警"
    assert signal.probability == 0.97


def test_predict_without_probability_returns_binary_only(tmp_path):
    service = FraudModelService(tmp_path)
    service._model = _BinaryOnlyModelPredictsFraud()

    signal = service._predict([[0.0] * 10])

    assert signal.result == "suspected"
    assert signal.label == "有预警"
    assert signal.probability is None


def test_analyze_returns_unavailable_when_assets_missing(tmp_path):
    service = FraudModelService(_MissingAssetLoader())
    record = {field: 1.0 for field in FRAUD_MODEL_FEATURE_FIELDS}

    result = service.analyze(record)

    assert result.fraud_screening.result == "not_available"
    assert result.pca_feature_scores == []
    assert service.status().ready is False


def test_concurrent_first_analyze_loads_model_assets_once():
    loader = _CountingAssetLoader()
    service = FraudModelService(loader)
    record = {field: 1.0 for field in FRAUD_MODEL_FEATURE_FIELDS}

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(service.analyze, [record] * 6))

    assert [result.fraud_screening.result for result in results] == ["suspected"] * 6
    assert loader.load_model_count == 1
    assert loader.load_scaler_count == 1
    assert loader.load_pca_count == 1


def test_analyze_generates_ten_pca_scores_when_runtime_assets_loaded(tmp_path):
    service = FraudModelService(tmp_path)
    service._loaded = True
    service._status = FraudModelStatus(enabled=True, ready=True, status="ready")
    service._model = _BinaryOnlyModelPredictsFraud()
    service._scaler = [
        _ScalerFeature(name=field, mean=0.0, std=1.0)
        for field in FRAUD_MODEL_FEATURE_FIELDS
    ]
    service._pca_matrix = [
        [
            1.0 if column_index == row_index % len(PCA_FEATURE_SCORE_NAMES) else 0.0
            for column_index in range(len(PCA_FEATURE_SCORE_NAMES))
        ]
        for row_index in range(len(FRAUD_MODEL_FEATURE_FIELDS))
    ]
    record = {field: 1.0 for field in FRAUD_MODEL_FEATURE_FIELDS}

    result = service.analyze(record)

    assert result.fraud_screening.result == "suspected"
    assert [item.name for item in result.pca_feature_scores] == PCA_FEATURE_SCORE_NAMES
    assert len(result.pca_feature_scores) == 10
