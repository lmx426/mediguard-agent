"""Single source of truth for Policy RAG retrieval metadata.

The online query planner, MCP request validation, and offline metadata rebuild
must share these values. Retrieval hints and document genres deliberately stay
outside this contract: they are not policy-evidence chunk types.
"""

from __future__ import annotations


JURISDICTIONS: tuple[str, ...] = (
    "national",
    "beijing",
    "shanghai",
)

POLICY_DOMAINS: tuple[str, ...] = (
    "benefit",
    "chronic_disease_long_prescription",
    "designated_institution",
    "drug_catalog",
    "drug_product_price_reference",
    "emergency",
    "fund_supervision",
    "manual_reimbursement",
    "medical_service_price",
    "remote_medical",
    "remote_medical_manual_reimbursement",
    "shanghai_payment_scope",
    "special_disease_filing",
    "special_disease_scope",
)

CONTENT_TYPES: tuple[str, ...] = (
    "policy_text",
    "table_row",
)

FILTERABLE_METADATA_FIELDS: tuple[str, ...] = (
    "jurisdiction",
    "policy_domain",
    "can_cite_as_policy_basis",
)

RETRIEVAL_HINT_CONTENT_TYPES: frozenset[str] = frozenset({"graph_hint"})
RETRIEVAL_HINT_POLICY_DOMAINS: frozenset[str] = frozenset({"graph_hints"})


def is_registered_filter_value(field_name: str, value: object) -> bool:
    """Return whether one scalar belongs to the registered filter taxonomy."""

    text = str(value or "").strip()
    if field_name == "jurisdiction":
        return text in JURISDICTIONS
    if field_name == "policy_domain":
        return text in POLICY_DOMAINS
    if field_name == "content_type":
        return text in CONTENT_TYPES
    if field_name == "can_cite_as_policy_basis":
        return isinstance(value, bool)
    return False
