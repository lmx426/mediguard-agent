from __future__ import annotations

import os

from src.backend.infrastructure.policy_rag.device import resolve_policy_model_device


def test_policy_rag_device_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MEDIGUARD_POLICY_RAG_DEVICE", "cpu")

    assert resolve_policy_model_device() == "cpu"


def test_policy_rag_device_requested_value_wins(monkeypatch) -> None:
    monkeypatch.setenv("MEDIGUARD_POLICY_RAG_DEVICE", "cpu")

    assert resolve_policy_model_device("cuda:0") == "cuda:0"
