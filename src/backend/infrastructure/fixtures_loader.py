"""本地脱敏样本 fixture 加载器。"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..application.ports.repositories import (
    CaseRepository,
    NoteRepository,
    ReviewRepository,
    TraceRepository,
)
from ..application.intake.build_case_pipeline import AuditPipelineService
from ..constants.ingest_fields import INGEST_RECORD_FIELDS
from ..domain.intake.validation_rules import IngestValidationError, IngestValidationService
from ..domain.intake.entities import (
    IngestRecordInput,
    IngestRecordResponse,
    IngestRecordSummary,
)


class IngestRecordCatalog:
    """从本地 CSV fixture 提供只读脱敏业务样本。"""

    def __init__(
        self,
        fixture_path: Path,
        validation_service: IngestValidationService,
    ) -> None:
        self.fixture_path = fixture_path
        self.validation_service = validation_service
        self._records: dict[str, dict[str, float | str]] = {}

    def load_fixture(self) -> None:
        """加载并验证 CSV fixture 的完整 81 字段。"""

        self._records.clear()
        with open(self.fixture_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != INGEST_RECORD_FIELDS:
                raise IngestValidationError(
                    ["脱敏测试集表头必须与完整 81 字段契约一致"]
                )
            for row in reader:
                safe_record = self.validation_service.validate(
                    IngestRecordInput(record=row)
                )
                self._records[str(safe_record["个人编码"])] = safe_record

    def list_records(self) -> list[IngestRecordSummary]:
        """返回脱敏样本摘要。"""

        return [
            self.validation_service.summarize(record)
            for record in self._records.values()
        ]

    def batch_records(self, record_ids: list[str]) -> list[dict[str, float | str]]:
        """按顺序返回多条脱敏样本。"""

        records: list[dict[str, float | str]] = []
        for record_id in record_ids:
            record = self._records.get(record_id)
            if record is None:
                raise IngestValidationError([f"上游记录 {record_id} 不存在"])
            records.append(record)
        return records

    def get_record(self, record_id: str) -> IngestRecordResponse | None:
        """返回一条完整脱敏样本。"""

        record = self._records.get(record_id)
        if record is None:
            return None
        return IngestRecordResponse(record_id=record_id, record=record)


FORBIDDEN_CONTEXT_KEYS = {"RES", "姓名", "身份证", "电话", "住址"}


@dataclass(frozen=True)
class VisitorSampleSpec:
    """Preprocessed visitor sample metadata, separate from mediredata.csv."""

    record_id: str
    expected_res: int
    sample_result: str
    sample_category: str
    sample_label: str
    access_mode: str
    case_type: str
    case_title: str
    source_system: str = "游客样本仿真（模拟手工上传材料）"
    record_version: str = "claim-wide-v1"
    expected_rule_ids: list[str] = field(default_factory=list)
    case_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VisitorSampleSpec":
        missing = [
            key
            for key in (
                "record_id",
                "expected_res",
                "sample_result",
                "sample_category",
                "sample_label",
                "access_mode",
                "case_type",
                "case_title",
                "case_context",
            )
            if key not in payload
        ]
        if missing:
            raise IngestValidationError([f"游客样本规格缺少字段：{', '.join(missing)}"])
        context = _safe_context(payload.get("case_context") or {})
        metadata = {
            "sample_result": payload["sample_result"],
            "sample_category": payload["sample_category"],
            "sample_label": payload["sample_label"],
            "access_mode": payload["access_mode"],
            "case_type": payload["case_type"],
            "case_title": payload["case_title"],
            "expected_rule_ids": list(payload.get("expected_rule_ids") or []),
        }
        context = {**context, "visitor_sample": metadata}
        return cls(
            record_id=str(payload["record_id"]),
            expected_res=int(payload["expected_res"]),
            sample_result=str(payload["sample_result"]),
            sample_category=str(payload["sample_category"]),
            sample_label=str(payload["sample_label"]),
            access_mode=str(payload["access_mode"]),
            case_type=str(payload["case_type"]),
            case_title=str(payload["case_title"]),
            source_system=str(payload.get("source_system") or "游客样本仿真（模拟手工上传材料）"),
            record_version=str(payload.get("record_version") or "claim-wide-v1"),
            expected_rule_ids=list(payload.get("expected_rule_ids") or []),
            case_context=context,
        )


class VisitorIngestRecordCatalog:
    """Provide allow-listed visitor samples from labelled or safe fixtures."""

    def __init__(
        self,
        fixture_path: Path,
        validation_service: IngestValidationService,
        specs_path: Path | None = None,
    ) -> None:
        self.fixture_path = fixture_path
        self.specs_path = specs_path
        self.validation_service = validation_service
        self._records: dict[str, dict[str, float | str]] = {}
        self._specs: dict[str, VisitorSampleSpec] = {}

    def load_fixture(self) -> None:
        """加载并校验游客演示样本，只保留预处理清单中的固定记录。"""

        self._records.clear()
        self._specs = {
            spec.record_id: spec
            for spec in self._load_specs()
        }
        expected = {
            record_id: spec.expected_res
            for record_id, spec in self._specs.items()
        }
        found: set[str] = set()

        encoding = "utf-8-sig" if self.fixture_path.suffix.lower() == ".csv" else "utf-8"
        try:
            file = open(self.fixture_path, "r", encoding=encoding, newline="")
            reader = csv.DictReader(file)
            labelled_fixture = reader.fieldnames == [*INGEST_RECORD_FIELDS, "RES"]
            safe_fixture = reader.fieldnames == INGEST_RECORD_FIELDS
        except UnicodeDecodeError:
            file = open(self.fixture_path, "r", encoding="gbk", newline="")
            reader = csv.DictReader(file)
            labelled_fixture = reader.fieldnames == [*INGEST_RECORD_FIELDS, "RES"]
            safe_fixture = reader.fieldnames == INGEST_RECORD_FIELDS

        with file:
            if not labelled_fixture and not safe_fixture:
                raise IngestValidationError(
                    ["游客演示样本表头必须严格等于完整 81 字段；本地选样文件可额外包含末列 RES"]
                )

            for row in reader:
                record_id = (row.get(INGEST_RECORD_FIELDS[0]) or "").strip()
                if record_id not in expected:
                    continue

                if labelled_fixture and (row.get("RES") or "").strip() != str(expected[record_id]):
                    raise IngestValidationError(
                        [f"游客演示样本 {record_id} 的标注与预期不一致"]
                    )

                record = {
                    field: row[field]
                    for field in INGEST_RECORD_FIELDS
                }
                safe_record = self.validation_service.validate(
                    IngestRecordInput(record=record)
                )
                self._records[record_id] = safe_record
                found.add(record_id)
                if len(found) == len(self._specs):
                    break

        missing = [
            record_id
            for record_id in self._specs
            if record_id not in found
        ]
        if missing:
            raise IngestValidationError(
                [f"游客演示样本缺少记录：{', '.join(missing)}"]
            )

    def list_records(self) -> list[IngestRecordSummary]:
        """返回游客演示样本摘要。"""

        summaries: list[IngestRecordSummary] = []
        for record_id in self._specs:
            record = self._records.get(record_id)
            if record is None:
                continue
            summary = self.validation_service.summarize(record)
            spec = self._specs[record_id]
            summaries.append(
                summary.model_copy(
                    update={
                        "sample_result": spec.sample_result,
                        "sample_category": spec.sample_category,
                        "sample_label": spec.sample_label,
                        "access_mode": spec.access_mode,
                        "case_type": spec.case_type,
                        "expected_rule_ids": list(spec.expected_rule_ids),
                    }
                )
            )
        return summaries

    def batch_records(self, record_ids: list[str]) -> list[dict[str, float | str]]:
        """按顺序返回多条游客演示样本。"""

        records: list[dict[str, float | str]] = []
        for record_id in record_ids:
            record = self._records.get(record_id)
            if record is None:
                raise IngestValidationError([f"游客样本 {record_id} 不存在"])
            records.append(record)
        return records

    def get_record(self, record_id: str) -> IngestRecordResponse | None:
        """返回一条游客演示样本的完整脱敏宽表。"""

        record = self._records.get(record_id)
        if record is None:
            return None
        spec = self._specs.get(record_id)
        if spec is None:
            return IngestRecordResponse(record_id=record_id, record=record)
        return IngestRecordResponse(
            record_id=record_id,
            record=record,
            metadata={
                "sample_result": spec.sample_result,
                "sample_category": spec.sample_category,
                "sample_label": spec.sample_label,
                "access_mode": spec.access_mode,
                "case_type": spec.case_type,
                "case_title": spec.case_title,
                "source_system": spec.source_system,
                "record_version": spec.record_version,
                "expected_rule_ids": list(spec.expected_rule_ids),
            },
            case_context=spec.case_context,
        )

    def spec_for_record(self, record_id: str) -> VisitorSampleSpec | None:
        return self._specs.get(record_id)

    def _load_specs(self) -> list[VisitorSampleSpec]:
        if self.specs_path is None:
            raise IngestValidationError(["缺少游客样本规格文件路径"])
        with open(self.specs_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        samples = payload.get("samples") if isinstance(payload, dict) else None
        if not isinstance(samples, list) or not samples:
            raise IngestValidationError(["游客样本规格文件必须包含 samples 数组"])
        specs = [
            VisitorSampleSpec.from_mapping(item)
            for item in samples
            if isinstance(item, dict)
        ]
        if len(specs) != len(samples):
            raise IngestValidationError(["游客样本规格存在非对象条目"])
        duplicate_ids = {
            spec.record_id for spec in specs if [item.record_id for item in specs].count(spec.record_id) > 1
        }
        if duplicate_ids:
            raise IngestValidationError([f"游客样本规格存在重复记录：{', '.join(sorted(duplicate_ids))}"])
        return specs


def _safe_context(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise IngestValidationError(["游客样本 case_context 必须是对象"])

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                text_key = str(key)
                if any(forbidden.upper() in text_key.upper() for forbidden in FORBIDDEN_CONTEXT_KEYS):
                    raise IngestValidationError([f"游客样本上下文禁止字段不得进入输入：{text_key}"])
                cleaned[text_key] = scrub(item)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(payload)


class FixturesLoader:
    """初始化脱敏样本目录和全部进程内状态。"""

    def __init__(
        self,
        cases: CaseRepository,
        traces: TraceRepository,
        reviews: ReviewRepository,
        notes: NoteRepository,
        pipeline: AuditPipelineService,
        ingest_records: IngestRecordCatalog,
        visitor_ingest_records: VisitorIngestRecordCatalog | None = None,
        reset_runtime_state: bool = True,
        showcase_record_ids: list[str] | None = None,
    ) -> None:
        self.cases = cases
        self.traces = traces
        self.reviews = reviews
        self.notes = notes
        self.pipeline = pipeline
        self.ingest_records = ingest_records
        self.visitor_ingest_records = visitor_ingest_records
        self.reset_runtime_state = reset_runtime_state
        self.showcase_record_ids = list(showcase_record_ids or [])

    def load_all(self) -> None:
        """清理运行时状态后加载只读脱敏样本目录。

        不预加载固定案件；审核队列只展示用户从样本或 CSV 接入生成的案件。
        """

        if self.reset_runtime_state:
            self.cases.clear()
            self.traces.clear()
            self.reviews.clear()
            self.notes.clear()

        self.ingest_records.load_fixture()
        if self.visitor_ingest_records is not None:
            self.visitor_ingest_records.load_fixture()
        self._seed_showcase_cases()

    def _seed_showcase_cases(self) -> None:
        if not self.showcase_record_ids or self.visitor_ingest_records is None:
            return
        for record_id in self.showcase_record_ids:
            sample = self.visitor_ingest_records.get_record(record_id)
            if sample is None:
                raise IngestValidationError([f"Showcase 演示样本缺少记录：{record_id}"])
            metadata = sample.metadata
            self.pipeline.run_ingest(
                IngestRecordInput(
                    record=sample.record,
                    case_title=str(metadata.get("case_title") or record_id),
                    case_type=str(metadata.get("case_type") or "Showcase 演示案件"),
                    source_system=str(metadata.get("source_system") or "Showcase 安全演示数据"),
                    record_version=str(metadata.get("record_version") or "showcase-v1"),
                    case_context=sample.case_context,
                ),
                self.visitor_ingest_records.validation_service,
            )
