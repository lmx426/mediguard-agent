"""Caser context refresh service."""

from __future__ import annotations

from typing import Any

from src.backend.application.ports.caser_context import CaserContextRepository
from src.backend.domain.caser_context.entities import (
    CASER_SECTION_KEYS,
    CASER_SECTION_SCHEMA_VERSION,
    CaserContextSection,
)

from .event_bus import CaserContextEvent, CaserContextEventBus
from .section_builder import CaserSectionBuilder


EVENT_SECTION_MAP = {
    "case.ingested": [
        "case_basic_info",
        "claimant_profile",
        "statistics_report",
        "risk_score",
    ],
    "materials.built": [
        "material_overview",
        "medical_materials",
        "prescription_materials",
        "settlement_materials",
    ],
    "evidence.built": [
        "evidence_package",
        "rule_verification",
    ],
    "review_advisor.completed": [
        "case_judgement",
        "verification_items",
    ],
}


class CaserContextRefreshService:
    """Build, store, and inspect Caser context sections."""

    def __init__(
        self,
        *,
        repository: CaserContextRepository,
        builder: CaserSectionBuilder,
        event_bus: CaserContextEventBus,
    ) -> None:
        self._repository = repository
        self._builder = builder
        self._event_bus = event_bus
        self._event_bus.subscribe(self.handle_event)

    @property
    def section_keys(self) -> list[str]:
        return list(CASER_SECTION_KEYS)

    def publish(
        self,
        event_type: str,
        case_id: str,
        *,
        section_keys: list[str] | None = None,
        reason: str = "",
        async_refresh: bool = True,
    ) -> None:
        self._event_bus.publish(
            event_type,
            case_id,
            section_keys=section_keys,
            reason=reason,
            async_refresh=async_refresh,
        )

    def handle_event(self, event: CaserContextEvent) -> None:
        section_keys = event.section_keys or EVENT_SECTION_MAP.get(event.event_type, [])
        if not section_keys:
            return
        try:
            self.rebuild_case(
                event.case_id,
                section_keys=section_keys,
                generated_by=f"event:{event.event_type}",
            )
        except Exception:
            # Event refresh is best-effort. Manual rebuild and normal reads remain available.
            return

    def rebuild_case(
        self,
        case_id: str,
        *,
        section_keys: list[str] | None = None,
        generated_by: str = "manual",
    ) -> list[CaserContextSection]:
        selected = _normalize_section_keys(section_keys)
        self._repository.mark_sections_building(
            case_id=case_id,
            section_keys=selected,
            generated_by=generated_by,
        )
        sections: list[CaserContextSection] = []
        for section_key in selected:
            try:
                built = self._builder.build(case_id, section_key)
                section = self._repository.upsert_section(
                    case_id=case_id,
                    section_key=section_key,
                    schema_version=CASER_SECTION_SCHEMA_VERSION,
                    status=built.status,
                    completeness=built.completeness,
                    payload=built.payload,
                    payload_hash=built.payload_hash,
                    source_refs=built.source_refs,
                    source_versions=built.source_versions,
                    generated_by=generated_by,
                )
            except Exception as exc:
                section = self._repository.upsert_section(
                    case_id=case_id,
                    section_key=section_key,
                    schema_version=CASER_SECTION_SCHEMA_VERSION,
                    status="failed",
                    completeness="not_available",
                    payload={},
                    payload_hash="",
                    source_refs=[],
                    source_versions={},
                    generated_by=generated_by,
                    error_code=exc.__class__.__name__,
                    error_message=str(exc),
                )
            sections.append(section)
        return sections

    def get_section(self, case_id: str, section_key: str) -> CaserContextSection:
        if section_key not in CASER_SECTION_KEYS:
            raise KeyError(section_key)
        section = self._repository.get_section(case_id=case_id, section_key=section_key)
        if section is not None:
            return section
        return _not_ready_section(case_id, section_key)

    def list_sections(self, case_id: str) -> list[CaserContextSection]:
        existing = {
            section.section_key: section
            for section in self._repository.list_sections(case_id=case_id)
        }
        return [
            existing.get(section_key) or _not_ready_section(case_id, section_key)
            for section_key in CASER_SECTION_KEYS
        ]

    def shutdown(self) -> None:
        self._event_bus.shutdown()


def _normalize_section_keys(section_keys: list[str] | None) -> list[str]:
    if section_keys is None:
        return list(CASER_SECTION_KEYS)
    normalized = [key for key in dict.fromkeys(section_keys) if key in CASER_SECTION_KEYS]
    if not normalized:
        raise ValueError("No valid Caser sections selected")
    return normalized


def _not_ready_section(case_id: str, section_key: str) -> CaserContextSection:
    return CaserContextSection(
        case_id=case_id,
        section_key=section_key,
        schema_version=CASER_SECTION_SCHEMA_VERSION,
        status="not_ready",
        completeness="not_available",
        payload={},
        payload_hash="",
        source_refs=[],
        source_versions={},
        generated_by="caser_context_builder",
        error_code="section_not_ready",
        error_message="Caser section has not been built.",
    )
