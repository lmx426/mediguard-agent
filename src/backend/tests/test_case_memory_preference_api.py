"""API coverage for the reviewer-level cross-case memory switch."""


def test_personal_memory_preference_is_persisted_for_the_current_user(client) -> None:
    initial = client.get("/api/memory/preference")
    assert initial.status_code == 200
    assert initial.json() == {"enabled": True}

    disabled = client.put("/api/memory/preference", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json() == {"enabled": False}

    persisted = client.get("/api/memory/preference")
    assert persisted.status_code == 200
    assert persisted.json() == {"enabled": False}

    enabled = client.put("/api/memory/preference", json={"enabled": True})
    assert enabled.status_code == 200
    assert enabled.json() == {"enabled": True}
