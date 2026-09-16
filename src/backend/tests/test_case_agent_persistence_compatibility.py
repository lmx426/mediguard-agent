"""Read compatibility tests for persisted Case Agent answer payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.backend.domain.case_agent.entities import CaseAgentAnswer
from src.backend.infrastructure.persistence.sql.repositories.case_agent_repo import (
    SqlCaseAgentRepository,
)


def _legacy_answer_payload() -> dict[str, object]:
    return {
        "display_mode": "grounded",
        "content_blocks": [{"text": "历史政策回答", "source_refs": ["policy:1"]}],
        "sources": [],
        "claims": [
            {
                "claim_id": "claim_1",
                "requirement_id": "remote_filing:705e02672cee",
                "slot_id": "remote_filing",
                "text": "跨省异地就医应按规定办理备案。",
                "fact_refs": ["fact_1"],
                "source_refs": ["policy:1"],
                "citation_ids": ["citation_1"],
                "support_status": "supported",
            }
        ],
        "citations": [
            {
                "citation_id": "citation_1",
                "label": 1,
                "claim_id": "claim_1",
                "fact_refs": ["fact_1"],
                "evidence_refs": ["evidence_1"],
                "source_refs": ["policy:1"],
            }
        ],
        "metadata": {},
    }


def test_current_answer_contract_still_rejects_legacy_claim_fields() -> None:
    with pytest.raises(ValidationError):
        CaseAgentAnswer.model_validate(_legacy_answer_payload())


def test_repository_reads_legacy_claims_as_need_keyed_history() -> None:
    row = SimpleNamespace(
        message_id="cmsg_legacy",
        role="assistant",
        content="历史政策回答",
        active_stage="evidence_package",
        answer_payload=_legacy_answer_payload(),
        source_refs=["policy:1"],
        created_at=datetime.now(timezone.utc),
    )

    message = SqlCaseAgentRepository._to_message(row, "csess_legacy")

    assert message.answer_payload is not None
    assert message.answer_payload.claims[0].need_id == "remote_filing:705e02672cee"
    assert message.answer_payload.claims[0].need_text == "跨省异地就医应按规定办理备案。"
    persisted = message.answer_payload.model_dump(mode="json")
    assert "slot_id" not in persisted["claims"][0]
    assert "requirement_id" not in persisted["claims"][0]
