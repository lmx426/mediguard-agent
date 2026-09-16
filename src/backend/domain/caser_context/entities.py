"""Caser context section schemas.

The View Store is a Caser-specific read model. It is intentionally separate
from the audit fact tables so the assistant reads a stable, case-bound,
source-aware projection instead of reaching into operational tables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CASER_SECTION_SCHEMA_VERSION = "caser-context-v1"

CaserSectionStatus = Literal["building", "ready", "not_ready", "failed", "stale"]
CaserSectionCompleteness = Literal["full", "partial", "not_available"]

CASER_SECTION_KEYS = [
    "case_basic_info",
    "claimant_profile",
    "material_overview",
    "medical_materials",
    "prescription_materials",
    "settlement_materials",
    "statistics_report",
    "risk_score",
    "evidence_package",
    "case_judgement",
    "verification_items",
    "rule_verification",
]


class CaserSectionMeta(BaseModel):
    """Source and freshness metadata embedded in every section payload."""

    schema_version: str = CASER_SECTION_SCHEMA_VERSION
    case_id: str
    section_key: str
    status: CaserSectionStatus = "ready"
    completeness: CaserSectionCompleteness = "full"
    source_refs: list[str] = Field(default_factory=list)
    source_versions: dict[str, str] = Field(default_factory=dict)
    generated_by: str = "caser_context_builder"
    generated_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class CaserSectionPayload(BaseModel):
    """Base section payload contract."""

    model_config = ConfigDict(extra="forbid")

    meta: CaserSectionMeta


class CaseBasicInfoSection(CaserSectionPayload):
    case_number: str
    case_type: str
    review_status: str
    claimant_ref: str | None = None
    access_method: str = "unknown"
    case_title: str = ""
    claim_summary: str = ""
    claim_amount: float | None = None
    insured_region: str = ""
    treatment_region: str = ""
    visit_type: str = ""
    claim_mode: str = "unknown"
    direct_settlement: bool | None = None
    filing_status: str = "unknown"
    emergency_material_status: str = "unknown"
    diagnosis: str = ""
    case_context: dict[str, Any] = Field(default_factory=dict)


class ClaimantProfileSection(CaserSectionPayload):
    claimant_code: str | None = None
    gender: str = "unknown"
    age_group: str = "unknown"
    insurance_type: str = "unknown"
    chronic_condition_tags: list[str] = Field(default_factory=list)
    allergy_history: str = "unknown"
    patient_group_tags: list[str] = Field(default_factory=list)


class MaterialOverviewSection(CaserSectionPayload):
    material_total: int = 0
    groups: list[dict[str, Any]] = Field(default_factory=list)


class MedicalMaterialsSection(CaserSectionPayload):
    materials: list[dict[str, Any]] = Field(default_factory=list)


class PrescriptionMaterialsSection(CaserSectionPayload):
    materials: list[dict[str, Any]] = Field(default_factory=list)


class SettlementMaterialsSection(CaserSectionPayload):
    materials: list[dict[str, Any]] = Field(default_factory=list)
    settlement_summary: dict[str, Any] = Field(default_factory=dict)
    non_drug_fee_items: list[dict[str, Any]] = Field(default_factory=list)


class StatisticsMetric(BaseModel):
    feature_name: str
    current_value: Any = None
    peer_median: Any = None
    p75: Any = None
    p90: Any = None
    baseline_status: str = "not_available"
    hint: str = ""


class StatisticsReportSection(CaserSectionPayload):
    metric_count: int = 0
    metric_baseline_status: str = "not_available"
    groups: dict[str, list[StatisticsMetric]] = Field(default_factory=dict)


class RiskScoreSection(CaserSectionPayload):
    overall_strength: str = ""
    risk_level: str = ""
    model_warning: dict[str, Any] = Field(default_factory=dict)
    rule_check: dict[str, Any] = Field(default_factory=dict)
    peer_deviation: dict[str, Any] = Field(default_factory=dict)
    data_flow: dict[str, Any] = Field(default_factory=dict)
    components: list[dict[str, Any]] = Field(default_factory=list)


class EvidencePackageSection(CaserSectionPayload):
    base_summary: dict[str, Any] = Field(default_factory=dict)
    discovered_clues: list[dict[str, Any]] = Field(default_factory=list)
    source_basis: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CaseJudgementSection(CaserSectionPayload):
    system_risk_prompt: dict[str, Any] = Field(default_factory=dict)
    evidence_review_result: dict[str, Any] = Field(default_factory=dict)
    review_note: str = ""
    clue_review: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class VerificationItemsSection(CaserSectionPayload):
    risk_related_tasks: list[dict[str, Any]] = Field(default_factory=list)
    global_scan_tasks: list[dict[str, Any]] = Field(default_factory=list)


class RuleVerificationSection(CaserSectionPayload):
    summary: dict[str, Any] = Field(default_factory=dict)
    rules: list[dict[str, Any]] = Field(default_factory=list)


SECTION_MODEL_BY_KEY: dict[str, type[CaserSectionPayload]] = {
    "case_basic_info": CaseBasicInfoSection,
    "claimant_profile": ClaimantProfileSection,
    "material_overview": MaterialOverviewSection,
    "medical_materials": MedicalMaterialsSection,
    "prescription_materials": PrescriptionMaterialsSection,
    "settlement_materials": SettlementMaterialsSection,
    "statistics_report": StatisticsReportSection,
    "risk_score": RiskScoreSection,
    "evidence_package": EvidencePackageSection,
    "case_judgement": CaseJudgementSection,
    "verification_items": VerificationItemsSection,
    "rule_verification": RuleVerificationSection,
}


class CaserContextSection(BaseModel):
    """Persisted Caser section row."""

    case_id: str
    section_key: str
    schema_version: str = CASER_SECTION_SCHEMA_VERSION
    status: CaserSectionStatus
    completeness: CaserSectionCompleteness = "full"
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""
    source_refs: list[str] = Field(default_factory=list)
    source_versions: dict[str, str] = Field(default_factory=dict)
    generated_by: str = "caser_context_builder"
    generated_at: datetime | None = None
    refreshed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


def validate_section_payload(section_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a section payload with the schema assigned to its key."""

    model = SECTION_MODEL_BY_KEY.get(section_key)
    if model is None:
        raise ValueError(f"Unknown Caser section: {section_key}")
    return model.model_validate(payload).model_dump(mode="json")
