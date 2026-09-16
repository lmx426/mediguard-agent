"""后端 API 集成测试夹具。"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MEDIGUARD_PERSISTENCE_BACKEND", "memory")
os.environ["MEDIGUARD_CASE_AGENT_ENABLED"] = "false"
os.environ.setdefault(
    "MEDIGUARD_AUTH_SECRET_KEY",
    "test-auth-secret-key-for-mediguard-local-tests",
)
os.environ.setdefault("MEDIGUARD_DEFAULT_AUDITOR_PASSWORD", "mediguard123")

from src.backend.main import create_app


@pytest.fixture
def client(monkeypatch):
    """为每个测试创建独立应用和进程内状态。"""

    monkeypatch.setenv("MEDIGUARD_FRAUD_MODEL_ENABLED", "false")
    monkeypatch.setenv("MEDIGUARD_EVIDENCE_AGENT_ENABLED", "false")
    monkeypatch.setenv("MEDIGUARD_CASE_AGENT_ENABLED", "false")
    with TestClient(create_app()) as test_client:
        response = test_client.post(
            "/api/auth/login",
            json={
                "username": "default_auditor",
                "password": "mediguard123",
            },
        )
        assert response.status_code == 200
        yield test_client


@pytest.fixture
def generated_case_id(client):
    """通过脱敏样本生成一个运行时案件。"""

    response = client.post("/api/ingest-records/SIM_PERSON_000006/push")
    assert response.status_code == 200
    return response.json()["case"]["case_id"]
