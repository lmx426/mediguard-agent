from __future__ import annotations

from src.backend.scripts.rebuild_policy_node_metadata_v2 import (
    METADATA_VERSION,
    PolicyNodeMetadataClassification,
    apply_classification,
    classification_input,
    is_retrieval_hint,
    metadata_tool_schema,
)


def _node() -> dict[str, object]:
    return {
        "id_": "policy-node-1",
        "text": "北京市门诊待遇政策正文。",
        "metadata": {
            "title": "北京市门诊待遇政策",
            "source_id": "source-1",
            "jurisdiction": "multi",
            "policy_domain": "graph_hints",
            "content_type": "graph_hint",
            "can_cite_as_policy_basis": False,
            "doc_type": "policy",
            "evidence_role": "policy_evidence",
        },
    }


def test_metadata_classification_input_does_not_expose_legacy_filters() -> None:
    payload = classification_input(_node())

    assert payload["title"] == "北京市门诊待遇政策"
    assert payload["text"] == "北京市门诊待遇政策正文。"
    assert "jurisdiction" not in payload
    assert "policy_domain" not in payload
    assert "content_type" not in payload


def test_metadata_v2_preserves_node_text_and_replaces_registered_filters() -> None:
    node = _node()
    classification = PolicyNodeMetadataClassification(
        jurisdiction="beijing",
        policy_domain="benefit",
        content_type="policy_text",
        can_cite_as_policy_basis=True,
        confidence=0.95,
        reason="北京市政策正文",
    )

    rebuilt = apply_classification(
        node=node,
        classification=classification,
        model="deepseek-chat",
    )

    assert rebuilt["id_"] == node["id_"]
    assert rebuilt["text"] == node["text"]
    metadata = rebuilt["metadata"]
    assert metadata["jurisdiction"] == "beijing"
    assert metadata["policy_domain"] == "benefit"
    assert metadata["content_type"] == "policy_text"
    assert metadata["can_cite_as_policy_basis"] is True
    assert metadata["metadata_version"] == METADATA_VERSION
    assert metadata["metadata_review_status"] == "accepted"


def test_metadata_tool_schema_uses_registered_two_content_types() -> None:
    schema = metadata_tool_schema()["function"]["parameters"]
    content_ref = schema["properties"]["content_type"]["$ref"]
    definition_name = content_ref.rsplit("/", 1)[-1]

    assert schema["$defs"][definition_name]["enum"] == [
        "policy_text",
        "table_row",
    ]


def test_retrieval_hint_is_excluded_from_formal_metadata_v2_nodes() -> None:
    assert is_retrieval_hint(_node()["metadata"]) is True
