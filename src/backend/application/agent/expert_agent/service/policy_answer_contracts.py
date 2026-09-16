"""Claim-first policy answer contracts for L3 Policy Expert."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field


AnswerMode = Literal[
    "fact_lookup",
    "list",
    "process_rule",
    "policy_explanation",
    "comparison",
]

RequirementStatus = Literal[
    "pending",
    "matched",
    "extracted",
    "answered",
    "unsupported",
    "conflicted",
]
EvidenceCoverageStatus = Literal["covered", "partial", "missing"]


class AnswerRequirement(BaseModel):
    """One information need resolved from the question."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(min_length=1, max_length=80)
    slot_id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=160)
    answer_mode: AnswerMode = Field(default="process_rule")
    question_span: str = Field(default="", max_length=240)
    required: bool = True
    scenario_id: str = Field(default="", max_length=120)
    required_fields: list[str] = Field(default_factory=list, max_length=24)
    answer_action: str = Field(default="answer", max_length=80)
    filters: dict[str, Any] = Field(default_factory=dict)
    fact_schema: str = Field(default="policy_rule", max_length=80)
    extractor_id: str = Field(default="policy_rule_span", max_length=120)
    status: RequirementStatus = "pending"

    @computed_field(return_type=str)
    @property
    def need_id(self) -> str:
        return self.requirement_id

    @computed_field(return_type=str)
    @property
    def need_text(self) -> str:
        return self.label

    @computed_field(return_type=str)
    @property
    def need_kind(self) -> str:
        return self.slot_id


class NormalizedPolicyEvidence(BaseModel):
    """Policy RAG evidence normalized before matching and extraction."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=220)
    source_ref: str = Field(min_length=1, max_length=220)
    title: str = Field(default="", max_length=240)
    content: str = Field(default="", max_length=2000)
    jurisdiction: str | None = Field(default=None, max_length=120)
    policy_domain: str | None = Field(default=None, max_length=120)
    content_type: str | None = Field(default=None, max_length=80)
    source_url: str | None = Field(default=None, max_length=500)
    version: str | None = Field(default=None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceMatch(BaseModel):
    """Evidence coverage status for one information need."""

    model_config = ConfigDict(extra="forbid")

    match_id: str = Field(min_length=1, max_length=80)
    requirement_id: str = Field(min_length=1, max_length=80)
    evidence_id: str = Field(min_length=1, max_length=220)
    coverage_status: EvidenceCoverageStatus
    matched_spans: list[str] = Field(default_factory=list, max_length=8)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=240)

    @computed_field(return_type=str)
    @property
    def need_id(self) -> str:
        return self.requirement_id

    @computed_field(return_type=str)
    @property
    def need_text(self) -> str:
        return self.reason

    @computed_field(return_type=str)
    @property
    def need_kind(self) -> str:
        return self.requirement_id


class ExtractedFact(BaseModel):
    """A structured fact extracted from citable evidence."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1, max_length=80)
    requirement_id: str = Field(min_length=1, max_length=80)
    slot_id: str = Field(min_length=1, max_length=120)
    fact_type: str = Field(min_length=1, max_length=120)
    value: dict[str, Any] = Field(default_factory=dict)
    display_text: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)
    source_refs: list[str] = Field(min_length=1, max_length=8)
    evidence_text: str = Field(min_length=1, max_length=800)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @computed_field(return_type=str)
    @property
    def need_id(self) -> str:
        return self.requirement_id

    @computed_field(return_type=str)
    @property
    def need_text(self) -> str:
        return self.display_text

    @computed_field(return_type=str)
    @property
    def need_kind(self) -> str:
        return self.slot_id


class AnswerClaim(BaseModel):
    """One final factual answer statement."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=80)
    requirement_id: str = Field(min_length=1, max_length=80)
    slot_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=600)
    fact_refs: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    citation_ids: list[str] = Field(default_factory=list, max_length=8)
    support_status: Literal["supported", "partial", "unsupported"] = "supported"

    @computed_field(return_type=str)
    @property
    def need_id(self) -> str:
        return self.requirement_id

    @computed_field(return_type=str)
    @property
    def need_text(self) -> str:
        return self.text

    @computed_field(return_type=str)
    @property
    def need_kind(self) -> str:
        return self.slot_id


class ClaimCitation(BaseModel):
    """Clickable citation bound to a claim and its supporting facts."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1, max_length=80)
    label: int = Field(ge=1, le=99)
    claim_id: str | None = Field(default=None, max_length=80)
    fact_refs: list[str] = Field(default_factory=list, max_length=12)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class ClaimFirstPolicyAnswer(BaseModel):
    """Complete deterministic policy answer draft used by synthesis."""

    model_config = ConfigDict(extra="forbid")

    answer_mode: AnswerMode = "process_rule"
    information_needs: list[str] = Field(default_factory=list, max_length=12)
    expert_answer: str = Field(default="", max_length=2000)
    answer_markdown: str = Field(default="", max_length=4000)
    answer_requirements: list[AnswerRequirement] = Field(default_factory=list, max_length=16)
    evidence_matches: list[EvidenceMatch] = Field(default_factory=list, max_length=40)
    extracted_facts: list[ExtractedFact] = Field(default_factory=list, max_length=40)
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=30)
    claim_plan: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    citations: list[ClaimCitation] = Field(default_factory=list, max_length=30)
    source_refs: list[str] = Field(default_factory=list, max_length=20)
    support_status: Literal["supported", "partial", "unsupported"] = "unsupported"
