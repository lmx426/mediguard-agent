"""Repository port for Caser context sections."""

from __future__ import annotations

from typing import Any, Protocol

from src.backend.domain.caser_context.entities import CaserContextSection


class CaserContextRepository(Protocol):
    """Persist and read Caser View Store sections."""

    def upsert_section(
        self,
        *,
        case_id: str,
        section_key: str,
        schema_version: str,
        status: str,
        completeness: str,
        payload: dict[str, Any],
        payload_hash: str,
        source_refs: list[str],
        source_versions: dict[str, str],
        generated_by: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> CaserContextSection: ...

    def mark_sections_building(
        self,
        *,
        case_id: str,
        section_keys: list[str],
        generated_by: str,
    ) -> None: ...

    def get_section(self, *, case_id: str, section_key: str) -> CaserContextSection | None: ...

    def list_sections(self, *, case_id: str) -> list[CaserContextSection]: ...
