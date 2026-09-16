"""Public interactive demo protection regression tests."""

from fastapi.testclient import TestClient

from src.backend.core.config import BACKEND_DIR, Settings
from src.backend.main import create_app


SAMPLE_IDS = [
    "SIM_PERSON_012345",
    "SIM_PERSON_014632",
    "SIM_PERSON_011569",
    "SIM_PERSON_002204",
    "SIM_PERSON_011394",
]


def _settings(**updates) -> Settings:
    values = {
        "_env_file": None,
        "showcase_mode": False,
        "public_demo_mode": True,
        "persistence_backend": "memory",
        "auth_secret_key": "public-demo-test-secret-key-at-least-32-characters",
        "default_auditor_password": "public-demo-password",
        "auth_cookie_secure": False,
        "fraud_model_enabled": False,
        "evidence_agent_enabled": False,
        "case_agent_enabled": False,
        "case_memory_enabled": False,
        "case_memory_worker_enabled": False,
        "mem0_enabled": False,
        "ingest_records_path": BACKEND_DIR / "fixtures" / "showcase_ingest_records.csv",
        "visitor_ingest_records_path": BACKEND_DIR
        / "fixtures"
        / "showcase_ingest_records.csv",
        "visitor_sample_specs_path": BACKEND_DIR
        / "fixtures"
        / "showcase_sample_specs.json",
    }
    values.update(updates)
    return Settings(**values)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        json={
            "username": "default_auditor",
            "password": "public-demo-password",
        },
    )
    assert response.status_code == 200


def _sample_input(client: TestClient, record_id: str) -> dict:
    response = client.get(f"/api/ingest-records/{record_id}")
    assert response.status_code == 200
    sample = response.json()
    return {
        "case_title": f"公网交互测试 {record_id}",
        "case_type": "公网交互测试",
        "source_system": "合成脱敏样本",
        "record_version": "public-demo-test-v1",
        "case_context": sample.get("case_context") or {},
        "record": sample["record"],
    }


def test_public_demo_is_interactive_and_uses_dynamic_pipeline():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        health = client.get("/health")
        created = client.post(f"/api/ingest-records/{SAMPLE_IDS[0]}/push")

    assert health.status_code == 200
    assert health.json()["showcase"]["enabled"] is False
    assert health.json()["public_demo"] == {
        "enabled": True,
        "interactive": True,
        "max_batch_records": 10,
        "max_cases": 50,
    }
    assert created.status_code == 200
    assert created.json()["case"]["subject_ref"] == SAMPLE_IDS[0]


def test_public_demo_prefers_full_visitor_case_metadata_over_showcase_csv():
    with TestClient(
        create_app(
            _settings(
                ingest_records_path=BACKEND_DIR / "fixtures" / "showcase_ingest_records.csv",
                visitor_ingest_records_path=BACKEND_DIR
                / "fixtures"
                / "visitor_ingest_records.csv",
                visitor_sample_specs_path=BACKEND_DIR
                / "fixtures"
                / "visitor_demo_samples.json",
            )
        )
    ) as client:
        _login(client)
        record = client.get("/api/ingest-records/SIM_PERSON_011872")
        created = client.post("/api/ingest-records/SIM_PERSON_011872/push")

    assert record.status_code == 200
    assert record.json()["metadata"]["case_title"] == "异地门急诊手工报销备案待核验样本"
    assert (
        record.json()["case_context"]["material_scenario"]
        == "remote_emergency_manual_unknown_filing"
    )
    assert created.status_code == 200
    assert created.json()["case"]["case_title"] == "异地门急诊手工报销备案待核验样本"
    assert created.json()["case"]["case_context"]["visitor_sample"]["expected_rule_ids"] == [
        "OP-R009"
    ]


def test_lightweight_memory_mode_reports_available_sql_fallback():
    with TestClient(
        create_app(
            _settings(
                case_memory_enabled=True,
                case_memory_worker_enabled=True,
            )
        )
    ) as client:
        _login(client)
        health = client.get("/health")
        preference = client.get("/api/memory/preference")
        candidates = client.get("/api/memory/candidates")

    assert health.status_code == 200
    assert health.json()["case_memory"] == {
        "enabled": True,
        "available": True,
        "persistence": "memory",
        "engine": "in_memory_sql_graph_bm25",
        "graph_backend": "sql",
        "vector_backend": "disabled",
        "fallback": "in_memory",
        "reason": None,
        "worker_running": False,
        "outbox": {},
        "mem0_enabled": False,
        "mem0_last_error": None,
        "last_error": None,
        "last_reconciliation_at": None,
        "last_consolidation_at": None,
    }
    assert preference.status_code == 200
    assert preference.json() == {"enabled": True}
    assert candidates.status_code == 200


def test_public_demo_rejects_oversized_batch_before_writing():
    with TestClient(create_app(_settings(public_demo_max_batch_records=1))) as client:
        _login(client)
        records = [
            _sample_input(client, SAMPLE_IDS[0]),
            _sample_input(client, SAMPLE_IDS[1]),
        ]
        response = client.post("/api/ingest-records/batch", json={"records": records})
        cases = client.get("/api/cases")

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "public_demo_batch_too_large"
    assert cases.json() == []


def test_public_demo_rejects_ingest_after_case_capacity_is_reached():
    with TestClient(
        create_app(
            _settings(
                public_demo_max_cases=5,
                public_demo_ingest_max_requests=10,
            )
        )
    ) as client:
        _login(client)
        for record_id in SAMPLE_IDS:
            assert client.post(f"/api/ingest-records/{record_id}/push").status_code == 200
        blocked = client.post(f"/api/ingest-records/{SAMPLE_IDS[0]}/push")

    assert blocked.status_code == 503
    assert blocked.json()["detail"]["code"] == "public_demo_case_capacity_reached"


def test_public_demo_rate_limits_expensive_write_categories():
    with TestClient(
        create_app(
            _settings(
                public_demo_ingest_max_requests=1,
                public_demo_rate_window_seconds=300,
            )
        )
    ) as client:
        _login(client)
        first = client.post(f"/api/ingest-records/{SAMPLE_IDS[0]}/push")
        blocked = client.post(f"/api/ingest-records/{SAMPLE_IDS[1]}/push")

    assert first.status_code == 200
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "public_demo_rate_limited"


def test_public_demo_rejects_legacy_feature_ingest():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        response = client.post("/api/audit-record", json={"features": {}})

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "public_demo_legacy_ingest_disabled"


def test_public_demo_rejects_large_request_body():
    settings = _settings(public_demo_max_request_bytes=100_000)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/auth/login",
            content="x" * 100_001,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "public_demo_request_too_large"
