"""Authentication API tests."""

from fastapi.testclient import TestClient

from src.backend.main import create_app


def test_login_required_for_business_api(monkeypatch):
    monkeypatch.setenv("MEDIGUARD_PERSISTENCE_BACKEND", "memory")
    monkeypatch.setenv(
        "MEDIGUARD_AUTH_SECRET_KEY",
        "test-auth-secret-key-for-mediguard-local-tests",
    )
    monkeypatch.setenv("MEDIGUARD_DEFAULT_AUDITOR_PASSWORD", "mediguard123")
    with TestClient(create_app()) as client:
        response = client.get("/api/cases")

    assert response.status_code == 401


def test_default_auditor_can_login(monkeypatch):
    monkeypatch.setenv("MEDIGUARD_PERSISTENCE_BACKEND", "memory")
    monkeypatch.setenv(
        "MEDIGUARD_AUTH_SECRET_KEY",
        "test-auth-secret-key-for-mediguard-local-tests",
    )
    monkeypatch.setenv("MEDIGUARD_DEFAULT_AUDITOR_PASSWORD", "mediguard123")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/auth/login",
            json={"username": "default_auditor", "password": "mediguard123"},
        )
        me_response = client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["user"]["username"] == "default_auditor"
    assert me_response.status_code == 200
    assert me_response.json()["display_name"] == "演示审核员"
