"""PostgreSQL-backed business material snapshot repository."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .....domain.audit.review.entities import MaterialAssetLocation
from ..models import CaseMaterialAssetORM, CaseMaterialSnapshotORM, CaseORM
from ..session import session_scope


class SqlMaterialRepository:
    """Persist the current business material snapshot for each case."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def clear(self) -> None:
        with session_scope(self._session_factory) as session:
            session.execute(delete(CaseMaterialSnapshotORM))

    def get_snapshot(self, case_id: str) -> dict | None:
        with self._session_factory() as session:
            row = self._snapshot_row(session, case_id)
            return dict(row.materials_payload) if row is not None else None

    def save_snapshot(
        self,
        case_id: str,
        snapshot: dict,
        assets: list[MaterialAssetLocation],
    ) -> None:
        with session_scope(self._session_factory) as session:
            case_row = session.scalar(
                select(CaseORM)
                .where(CaseORM.case_id == case_id)
                .with_for_update()
            )
            if case_row is None:
                return
            existing = self._snapshot_row(session, case_id)
            if existing is not None:
                session.delete(existing)
                session.flush()

            metadata = snapshot.get("snapshot_metadata", {})
            snapshot_row = CaseMaterialSnapshotORM(
                case_id=case_row.id,
                input_fingerprint=str(metadata.get("input_fingerprint") or ""),
                template_version=str(snapshot.get("template_version") or ""),
                scenario_code=str(metadata.get("scenario_code") or ""),
                scenario_label=str(metadata.get("scenario_label") or ""),
                evidence_tags=list(metadata.get("evidence_tags") or []),
                subject_profile=dict(snapshot.get("subject_profile") or {}),
                materials_payload=snapshot,
                generation_status=str(snapshot.get("generation_status") or "ready"),
            )
            session.add(snapshot_row)
            session.flush()
            for asset in assets:
                session.add(
                    CaseMaterialAssetORM(
                        material_set_id=snapshot_row.id,
                        material_id=asset.material_id,
                        asset_id=asset.asset_id,
                        asset_type=asset.asset_type,
                        file_type=asset.file_type,
                        storage_path=asset.storage_path,
                        sha256=asset.sha256,
                        size_bytes=asset.size_bytes,
                        display_name=asset.display_name,
                    )
                )

    def get_asset(
        self,
        case_id: str,
        material_id: str,
        asset_id: str,
    ) -> MaterialAssetLocation | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(CaseMaterialAssetORM)
                .join(
                    CaseMaterialSnapshotORM,
                    CaseMaterialAssetORM.material_set_id == CaseMaterialSnapshotORM.id,
                )
                .join(CaseORM, CaseMaterialSnapshotORM.case_id == CaseORM.id)
                .where(
                    CaseORM.case_id == case_id,
                    CaseMaterialAssetORM.material_id == material_id,
                    CaseMaterialAssetORM.asset_id == asset_id,
                )
            )
            if row is None:
                return None
            return MaterialAssetLocation(
                material_id=row.material_id,
                asset_id=row.asset_id,
                asset_type=row.asset_type,
                file_type=row.file_type,
                storage_path=row.storage_path,
                sha256=row.sha256,
                size_bytes=row.size_bytes,
                display_name=row.display_name,
            )

    @staticmethod
    def _snapshot_row(
        session: Session,
        case_id: str,
    ) -> CaseMaterialSnapshotORM | None:
        return session.scalar(
            select(CaseMaterialSnapshotORM)
            .join(CaseORM, CaseMaterialSnapshotORM.case_id == CaseORM.id)
            .where(CaseORM.case_id == case_id)
        )
