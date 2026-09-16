"""In-memory business material snapshot repository."""

from __future__ import annotations

from copy import deepcopy

from ....domain.audit.review.entities import MaterialAssetLocation


class MemoryMaterialRepository:
    """Store generated material snapshots for the current process only."""

    def __init__(self) -> None:
        self._snapshots: dict[str, dict] = {}
        self._assets: dict[tuple[str, str, str], MaterialAssetLocation] = {}

    def clear(self) -> None:
        self._snapshots.clear()
        self._assets.clear()

    def get_snapshot(self, case_id: str) -> dict | None:
        snapshot = self._snapshots.get(case_id)
        return deepcopy(snapshot) if snapshot is not None else None

    def save_snapshot(
        self,
        case_id: str,
        snapshot: dict,
        assets: list[MaterialAssetLocation],
    ) -> None:
        self._snapshots[case_id] = deepcopy(snapshot)
        self._assets = {
            key: value
            for key, value in self._assets.items()
            if key[0] != case_id
        }
        for asset in assets:
            self._assets[(case_id, asset.material_id, asset.asset_id)] = asset

    def get_asset(
        self,
        case_id: str,
        material_id: str,
        asset_id: str,
    ) -> MaterialAssetLocation | None:
        return self._assets.get((case_id, material_id, asset_id))
