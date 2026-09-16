"""欺诈模型标准化参数脚本测试。"""

import json

from src.backend.constants.fraud_model_fields import FRAUD_MODEL_FEATURE_FIELDS
from tools.generate_fraud_scaler_stats import generate_stats


def test_generate_stats_outputs_only_model_fields_even_when_res_exists(tmp_path):
    input_path = tmp_path / "mediredata.csv"
    header = [*FRAUD_MODEL_FEATURE_FIELDS, "RES"]
    first_row = ["1" for _ in FRAUD_MODEL_FEATURE_FIELDS] + ["0"]
    second_row = ["3" for _ in FRAUD_MODEL_FEATURE_FIELDS] + ["1"]
    input_path.write_text(
        ",".join(header)
        + "\n"
        + ",".join(first_row)
        + "\n"
        + ",".join(second_row)
        + "\n",
        encoding="utf-8",
    )

    payload = generate_stats(input_path, "utf-8")
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["feature_count"] == 39
    assert [item["name"] for item in payload["features"]] == FRAUD_MODEL_FEATURE_FIELDS
    assert "RES" not in serialized
