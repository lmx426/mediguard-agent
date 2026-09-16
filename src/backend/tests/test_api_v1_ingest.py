"""API v1 的 81 字段主业务闭环测试。"""


def _sample_record(client, record_id: str = "SIM_PERSON_000006") -> dict:
    response = client.get(f"/api/ingest-records/{record_id}")
    assert response.status_code == 200
    return response.json()["record"]


def _ingest_body(record: dict) -> dict:
    return {
        "case_type": "上游宽表记录接入",
        "source_system": "医保结算申报系统",
        "record_version": "claim-wide-v1",
        "record": record,
    }


def test_audit_queue_starts_empty_without_builtin_cases(client):
    response = client.get("/api/cases")

    assert response.status_code == 200
    assert response.json() == []


def test_health_reports_fraud_model_status(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["fraud_model"] == {
        "enabled": False,
        "ready": False,
        "status": "disabled",
    }


def test_root_health_alias_reports_fraud_model_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["fraud_model"] == {
        "enabled": False,
        "ready": False,
        "status": "disabled",
    }


def test_app_startup_preloads_fraud_model_before_first_health_check(client):
    assert client.app.state.container.fraud_model._loaded is True


def test_visitor_samples_api_lists_grouped_samples_without_training_label(client):
    response = client.get("/api/visitor-ingest-records")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 10
    assert len([item for item in payload if item["sample_result"] == "normal"]) == 2
    assert len([item for item in payload if item["sample_result"] == "abnormal"]) == 8
    expected_rule_ids = {
        rule_id
        for item in payload
        for rule_id in (item.get("expected_rule_ids") or [])
    }
    assert expected_rule_ids == {
        "OP-R001",
        "OP-R003",
        "OP-R004",
        "OP-R005",
        "OP-R009",
        "OP-R010",
        "OP-R011",
        "OP-R012",
    }
    assert all(item.get("sample_category") for item in payload)
    assert "expected_res" not in str(payload)
    assert "RES" not in str(payload)


def test_high_risk_record_runs_full_business_flow(client):
    record = {
        **_sample_record(client),
        "月就诊次数_MAX": 24,
        "月就诊医院数_MAX": 6,
        "一天去两家医院的天数": 18,
        "ALL_SUM": 100000,
        "药品费发生金额_SUM": 95000,
        "药品费申报金额_SUM": 95000,
        "药品在总金额中的占比": 0.95,
        "是否挂号": 0,
    }
    ingest = client.post("/api/ingest-record", json=_ingest_body(record))

    assert ingest.status_code == 200
    detail = ingest.json()
    case_id = detail["case"]["case_id"]
    assert len(detail["case"]["source_record"]) == 81
    assert detail["case"]["fraud_screening"]["result"] == "not_available"
    assert detail["case"]["fraud_screening"]["probability"] is None
    assert detail["case"]["pca_feature_scores"] == []
    assert detail["case"]["risk_level"] == "medium"
    assert detail["case"]["risk_score_breakdown"]["components"][0]["label"] == "模型识别预警"
    assert detail["case"]["risk_score_breakdown"]["components"][0]["score"] == 0
    assert detail["case"]["risk_score_breakdown"]["components"][1]["score"] == 15
    assert len(
        [rule for rule in detail["case"]["rule_hits"] if rule["hit"]]
    ) >= 2
    assert len(detail["case"]["rule_hits"]) == 12
    assert "RES" not in str(detail)

    workflow = client.get(f"/api/cases/{case_id}/workflow").json()
    assert len(workflow["steps"]) == 9
    assert workflow["current_step"] == "initial_review"

    review = client.post(
        f"/api/cases/{case_id}/review",
        json={
            "reviewer": "审核员",
            "decision": "需进一步调查",
            "reason": "多项确定性业务线索需要进一步人工核验。",
            "attachments": [],
        },
    )
    assert review.status_code == 200

    result = client.get(f"/api/cases/{case_id}").json()
    workflow = client.get(f"/api/cases/{case_id}/workflow").json()
    assert result["review"]["decision"] == "需进一步调查"
    assert workflow["current_step"] == "case_result"
    assert workflow["steps"][-1]["status"] == "recorded"


def test_exact_duplicate_reuses_case_and_preserves_review(client):
    record = _sample_record(client)
    first = client.post("/api/ingest-record", json=_ingest_body(record)).json()
    case_id = first["case"]["case_id"]
    client.post(
        f"/api/cases/{case_id}/review",
        json={
            "reviewer": "审核员A",
            "decision": "需进一步调查",
            "reason": "保留已有人工初审。",
            "attachments": [],
        },
    )

    second_response = client.post(
        "/api/ingest-record",
        json=_ingest_body(record),
    )

    assert second_response.status_code == 200
    second = second_response.json()
    assert second["case"]["case_id"] == case_id
    assert second["review"]["decision"] == "需进一步调查"


def test_changed_record_for_same_subject_creates_new_case(client):
    record = _sample_record(client)
    first = client.post("/api/ingest-record", json=_ingest_body(record)).json()
    changed = {**record, "ALL_SUM": float(record["ALL_SUM"]) + 1}
    second = client.post(
        "/api/ingest-record",
        json=_ingest_body(changed),
    ).json()

    assert first["case"]["case_id"] != second["case"]["case_id"]
    assert first["case"]["subject_ref"] == second["case"]["subject_ref"]


def test_batch_validation_is_atomic(client):
    before = len(client.get("/api/cases").json())
    first = _sample_record(client, "SIM_PERSON_000006")
    invalid = {
        **_sample_record(client, "SIM_PERSON_000007"),
        "RES": 1,
    }

    response = client.post(
        "/api/ingest-records/batch",
        json={"records": [_ingest_body(first), _ingest_body(invalid)]},
    )

    assert response.status_code == 422
    assert "第 2 行" in str(response.json()["detail"])
    assert len(client.get("/api/cases").json()) == before


def test_batch_rejects_exact_duplicate_rows(client):
    before = len(client.get("/api/cases").json())
    record = _sample_record(client)

    response = client.post(
        "/api/ingest-records/batch",
        json={"records": [_ingest_body(record), _ingest_body(record)]},
    )

    assert response.status_code == 422
    assert "完全重复" in str(response.json()["detail"])
    assert len(client.get("/api/cases").json()) == before
