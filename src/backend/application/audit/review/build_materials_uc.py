"""Business material view use case."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ....application.ports.repositories import MaterialRepository
from ....core.exceptions import ResourceNotFoundError
from ....domain.audit.review.entities import (
    BusinessMaterialCategory,
    CaseDetail,
    MaterialAssetLocation,
    MedicalRecordDocument,
    MedicalRecordResponse,
)
from .material_generation import BusinessMaterialGenerator, NOTICE, TEMPLATE_VERSION

if TYPE_CHECKING:
    from ...agent.case_agent.context.refresh_service import CaserContextRefreshService


class StatisticalMaterialUseCase:
    """Build and retrieve versioned business material snapshots for a case."""

    def __init__(
        self,
        material_repository: MaterialRepository,
        asset_root: Path,
        api_prefix: str = "/api",
    ) -> None:
        self._repository = material_repository
        self._asset_root = asset_root
        self._api_prefix = api_prefix.rstrip("/")
        self._generator = BusinessMaterialGenerator(asset_root, api_prefix)
        self._caser_context_events: CaserContextRefreshService | None = None

    def set_caser_context_events(
        self,
        events: CaserContextRefreshService | None,
    ) -> None:
        """Attach the optional Caser context refresh publisher."""

        self._caser_context_events = events

    def build_for_case(self, case: CaseDetail) -> MedicalRecordResponse:
        snapshot = self._repository.get_snapshot(case.case_id)
        if snapshot is not None and snapshot.get("template_version") == TEMPLATE_VERSION:
            response = MedicalRecordResponse.model_validate(snapshot)
            return self._with_preview_urls(response)

        try:
            generated = self._generator.generate(case)
        except Exception:
            return self._failed_response(case.case_id)

        response = self._with_preview_urls(generated.response)
        payload = response.model_dump(mode="json")
        payload["snapshot_metadata"] = generated.metadata
        self._repository.save_snapshot(case.case_id, payload, generated.assets)
        if self._caser_context_events is not None:
            self._caser_context_events.publish(
                "materials.built",
                case.case_id,
                reason="business material snapshot saved",
            )
        return response

    def resolve_asset(
        self,
        case_id: str,
        material_id: str,
        asset_id: str,
    ) -> tuple[Path, MaterialAssetLocation]:
        asset = self._repository.get_asset(case_id, material_id, asset_id)
        if asset is None:
            raise ResourceNotFoundError("材料图片不存在或不属于当前案件")

        root = self._asset_root.resolve()
        path = (root / asset.storage_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ResourceNotFoundError("材料图片路径不在受控目录内") from exc
        if not path.exists() or not path.is_file():
            raise ResourceNotFoundError("材料图片文件不存在")
        if self._sha256(path) != asset.sha256:
            raise ResourceNotFoundError("材料图片校验失败")
        return path, asset

    def _with_preview_urls(
        self,
        response: MedicalRecordResponse,
    ) -> MedicalRecordResponse:
        response.notice = response.notice or NOTICE
        response.disclaimer = response.disclaimer or NOTICE
        response.template_version = response.template_version or TEMPLATE_VERSION

        def refresh(document: MedicalRecordDocument) -> None:
            for asset in document.assets:
                asset.preview_url = (
                    f"{self._api_prefix}/cases/{response.case_id}/materials/"
                    f"{document.document_id}/assets/{asset.asset_id}"
                )

        for document in response.documents:
            refresh(document)
        response.categories = self._categories(response.documents)
        return response

    @staticmethod
    def _categories(
        documents: list[MedicalRecordDocument],
    ) -> list[BusinessMaterialCategory]:
        definitions = [
            ("clinical", "就诊诊疗材料"),
            ("prescription", "处方购药材料"),
            ("settlement", "费用结算材料"),
        ]
        return [
            BusinessMaterialCategory(
                category_id=category_id,
                title=title,
                documents=[doc for doc in documents if doc.category_id == category_id],
            )
            for category_id, title in definitions
        ]

    @staticmethod
    def _failed_response(case_id: str) -> MedicalRecordResponse:
        document = MedicalRecordDocument(
            document_id=f"{case_id}-BM-FAILED",
            title="业务材料生成状态",
            document_type="系统状态",
            visit_date="本次申报周期",
            institution="业务材料服务",
            department="事实底座",
            status="生成失败",
            content="业务材料生成失败，请继续查看结算申报核验明细。",
            category_id="settlement",
            category_title="费用结算材料",
            occurred_at="本次申报周期",
            material_source="业务材料服务",
            occurrence_scene="",
            material_shape="文本",
        )
        return MedicalRecordResponse(
            case_id=case_id,
            disclaimer=NOTICE,
            notice=NOTICE,
            generation_status="failed",
            template_version=TEMPLATE_VERSION,
            documents=[document],
            categories=[
                BusinessMaterialCategory(
                    category_id="settlement",
                    title="费用结算材料",
                    documents=[document],
                )
            ],
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
