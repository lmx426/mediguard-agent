"""本地模型资产文件 I/O 加载器。

实现 ModelAssetsLoader 协议，从文件系统加载 XGBoost 模型、
PCA 矩阵和标准化统计信息。纯文件 I/O，不包含推理逻辑。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ...constants.fraud_model_fields import (
    FRAUD_MODEL_FEATURE_FIELDS,
    PCA_FEATURE_SCORE_NAMES,
)


class LocalAssetsLoader:
    """从本地文件系统加载 ML 模型资产。

    负责纯文件 I/O，不包含任何业务校验或推理逻辑。
    """

    def __init__(self, assets_dir: Path) -> None:
        """初始化本地资产加载器。

        Args:
            assets_dir: 包含 xgb_model.pkl、selectVecdata.csv 和 scaler_stats.json 的目录。
        """

        self._assets_dir = assets_dir

    def load_model(self) -> Any:
        """加载 XGBoost 模型文件。

        Returns:
            XGBoost 模型对象（依赖 joblib 反序列化）。

        Raises:
            FileNotFoundError: 模型文件不存在。
            ImportError: joblib 或 xgboost 未安装。
        """

        import importlib
        import warnings

        model_path = self._assets_dir / "xgb_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        joblib = importlib.import_module("joblib")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*If you are loading a serialized model.*",
                category=UserWarning,
            )
            return joblib.load(model_path)

    def load_scaler_stats(self) -> dict[str, Any]:
        """加载标准化统计 JSON 文件。

        Returns:
            {"features": [{"name": str, "mean": float, "std": float}, ...]} 格式的字典。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: JSON 格式无效或 features 键缺失。
        """

        scaler_path = self._assets_dir / "scaler_stats.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"标准化统计文件不存在：{scaler_path}")

        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
        if "features" not in payload:
            raise ValueError("scaler_stats.json 缺少 features 键")
        return payload

    def load_pca_matrix(self) -> list[list[float]]:
        """加载 PCA 投影矩阵 CSV 文件。

        Returns:
            二维浮点数列表，shape 为 (39, 10)。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 行数或列数不匹配。
        """

        pca_path = self._assets_dir / "selectVecdata.csv"
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA 矩阵文件不存在：{pca_path}")

        matrix: list[list[float]] = []
        with pca_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                matrix.append([float(value) for value in row])

        if len(matrix) != len(FRAUD_MODEL_FEATURE_FIELDS):
            raise ValueError(f"PCA matrix row count must be {len(FRAUD_MODEL_FEATURE_FIELDS)}")
        if any(len(row) != len(PCA_FEATURE_SCORE_NAMES) for row in matrix):
            raise ValueError(f"PCA matrix column count must be {len(PCA_FEATURE_SCORE_NAMES)}")

        return matrix
