"""Public interview showcase regression tests."""

from fastapi.testclient import TestClient

from src.backend.core.config import Settings
from src.backend.main import create_app


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        showcase_mode=True,
        persistence_backend="memory",
        auth_secret_key="showcase-test-secret-key-at-least-32-characters",
        default_auditor_password="showcase123",
        auth_cookie_secure=False,
        evidence_agent_enabled=False,
        case_agent_enabled=False,
        case_memory_enabled=False,
        case_memory_worker_enabled=False,
        mem0_enabled=False,
    )


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        json={"username": "default_auditor", "password": "showcase123"},
    )
    assert response.status_code == 200


def test_showcase_seeds_five_safe_cases_and_precomputed_model_results():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        cases = client.get("/api/cases")
        samples = client.get("/api/visitor-ingest-records")
        health = client.get("/health")
        details = [
            client.get(f"/api/cases/{item['case_id']}").json()
            for item in cases.json()
        ]

    assert cases.status_code == 200
    assert len(cases.json()) == 5
    assert samples.status_code == 200
    assert len(samples.json()) == 5
    assert all(item["case"]["subject_ref"].startswith("SIM_PERSON_") for item in details)
    assert all("RES" not in item["case"]["source_record"] for item in details)
    assert all(len(item["case"]["source_record"]) == 81 for item in details)
    assert health.json()["showcase"] == {
        "enabled": True,
        "read_only": True,
        "seed_case_count": 5,
    }
    assert health.json()["fraud_model"]["status"] == "showcase_precomputed"


def test_showcase_review_advisor_is_available_and_clearly_labelled():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        case_id = client.get("/api/cases").json()[0]["case_id"]
        status = client.get("/api/agent/status")
        analysis = client.get(f"/api/cases/{case_id}/evidence-agent/current")

    assert status.status_code == 200
    assert status.json()["provider"] == "precomputed"
    assert status.json()["showcase"] is True
    assert analysis.status_code == 200
    assert analysis.json()["generated_notice"] == "预生成演示，未调用实时模型"
    assert analysis.json()["model_name"] == "showcase-precomputed"
    assert analysis.json()["citations"]


def test_showcase_case_agent_is_precomputed_read_only_and_source_grounded():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        case_id = client.get("/api/cases").json()[0]["case_id"]
        status = client.get("/api/case-agent/status")
        projection = client.get(f"/api/cases/{case_id}/case-agent/showcase")

    assert status.status_code == 200
    assert status.json()["provider"] == "precomputed"
    assert status.json()["showcase"] is True
    assert projection.status_code == 200
    payload = projection.json()
    assert payload["generated_notice"] == "预生成演示，未调用实时模型"
    assert len(payload["answers"]) == 3
    assert payload["sources"]
    source_refs = {item["source_ref"] for item in payload["sources"]}
    assert all(
        set(answer["source_refs"]).issubset(source_refs)
        for answer in payload["answers"]
    )
    combined = " ".join(answer["answer"] for answer in payload["answers"])
    assert "建议自动拒付" not in combined
    assert "确认欺诈" not in combined


def test_showcase_blocks_business_writes_after_login():
    with TestClient(create_app(_settings())) as client:
        _login(client)
        case_id = client.get("/api/cases").json()[0]["case_id"]
        responses = [
            client.post("/api/ingest-records/SIM_PERSON_012345/push"),
            client.post(
                f"/api/cases/{case_id}/review",
                json={"decision": "人工复核", "reason": "测试"},
            ),
            client.post(
                f"/api/cases/{case_id}/evidence-agent/runs",
                json={"analysis_type": "comprehensive"},
            ),
            client.put("/api/memory/preference", json={"enabled": False}),
        ]

    assert all(response.status_code == 403 for response in responses)
    assert all(
        response.json()["detail"]["code"] == "showcase_read_only"
        for response in responses
    )


def test_login_rate_limiter_rejects_repeated_failures():
    settings = _settings().model_copy(
        update={"auth_login_max_attempts": 3, "auth_login_window_seconds": 300}
    )
    with TestClient(create_app(settings)) as client:
        for _ in range(3):
            response = client.post(
                "/api/auth/login",
                json={"username": "default_auditor", "password": "wrong-pass"},
            )
            assert response.status_code == 401
        blocked = client.post(
            "/api/auth/login",
            json={"username": "default_auditor", "password": "wrong-pass"},
        )

    assert blocked.status_code == 429
