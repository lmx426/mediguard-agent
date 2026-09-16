"""完整脱敏宽表接入测试。"""

import csv
from pathlib import Path

import pytest

from src.backend.domain.intake.entities import IngestRecordInput
from src.backend.application.intake.build_case_pipeline import AuditPipelineService
from src.backend.constants.ingest_fields import INGEST_RECORD_FIELDS
from src.backend.domain.audit.review.evidence_packager import EvidenceService
from src.backend.domain.intake.validation_rules import (
    FeatureValidationError,
    FeatureValidationService,
    IngestValidationError,
    IngestValidationService,
)
from src.backend.domain.audit.review.risk_engine import OperationalRiskSignalProvider
from src.backend.domain.audit.review.rule_evaluator import RuleService
from src.backend.infrastructure.persistence.memory.case_repository import CaseService
from src.backend.infrastructure.persistence.memory.trace_repository import TraceService
from src.backend.infrastructure.fixtures_loader import IngestRecordCatalog, VisitorIngestRecordCatalog


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "business_ingest_records.csv"
)
VISITOR_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "model" / "mediredata.csv"
VISITOR_SPECS_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "visitor_demo_samples.json"
)


def _fixture_record() -> dict[str, str]:
    with open(FIXTURE_PATH, "r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


class TestIngestValidationService:
    def setup_method(self):
        self.service = IngestValidationService()

    def test_schema_uses_full_81_fields_without_res(self):
        schema = self.service.get_schema()

        assert len(schema.required_fields) == 81
        assert schema.required_fields == INGEST_RECORD_FIELDS
        assert "RES" not in schema.required_fields
        assert len(schema.example_json.record) == 81

    def test_validates_full_sanitized_fixture_record(self):
        safe_record = self.service.validate(IngestRecordInput(record=_fixture_record()))

        assert str(safe_record["个人编码"]).startswith("SIM_PERSON_")
        assert isinstance(safe_record["ALL_SUM"], float)
        assert len(safe_record) == 81

    def test_rejects_res_field(self):
        record = _fixture_record()
        record["RES"] = "1"

        with pytest.raises(IngestValidationError) as exc:
            self.service.validate(IngestRecordInput(record=record))

        assert any("RES" in error for error in exc.value.errors)

    def test_rejects_non_synthetic_person_ref(self):
        record = _fixture_record()
        record["个人编码"] = "1000234876"

        with pytest.raises(IngestValidationError) as exc:
            self.service.validate(IngestRecordInput(record=record))

        assert any("SIM_PERSON_000001" in error for error in exc.value.errors)

    def test_extracts_current_audit_feature_view(self):
        safe_record = self.service.validate(IngestRecordInput(record=_fixture_record()))
        features = self.service.extract_audit_features(safe_record)

        assert set(features) == set(FeatureValidationService().get_schema().required_fields)
        assert "个人编码" not in features


class TestIngestRecordService:
    def setup_method(self):
        self.validation_service = IngestValidationService()
        self.service = IngestRecordCatalog(FIXTURE_PATH, self.validation_service)
        self.service.load_fixture()

    def test_lists_fixture_records_without_res(self):
        records = self.service.list_records()

        assert len(records) == 12
        assert records[0].subject_ref.startswith("SIM_PERSON_")
        assert records[0].risk_level in {"low", "medium", "high"}
        assert 0.0 <= records[0].risk_score <= 0.95

    def test_get_record_returns_full_safe_record(self):
        first_id = self.service.list_records()[0].record_id
        record = self.service.get_record(first_id)

        assert record is not None
        assert len(record.record) == 81
        assert "RES" not in record.record

    def test_batch_records_returns_requested_fixture_rows(self):
        record_ids = [item.record_id for item in self.service.list_records()[:2]]

        records = self.service.batch_records(record_ids)

        assert len(records) == 2
        assert all("RES" not in record for record in records)


class TestVisitorIngestRecordService:
    def setup_method(self):
        self.validation_service = IngestValidationService()
        self.service = VisitorIngestRecordCatalog(
            VISITOR_FIXTURE_PATH,
            self.validation_service,
            specs_path=VISITOR_SPECS_PATH,
        )
        self.service.load_fixture()

    def test_lists_preprocessed_visitor_samples_by_result(self):
        records = self.service.list_records()

        assert len(records) == 10
        assert [item.record_id for item in records[:2]] == [
            "SIM_PERSON_012345",
            "SIM_PERSON_006847",
        ]
        assert [item.record_id for item in records[2:]] == [
            "SIM_PERSON_011872",
            "SIM_PERSON_006395",
            "SIM_PERSON_002052",
            "SIM_PERSON_014632",
            "SIM_PERSON_011569",
            "SIM_PERSON_002204",
            "SIM_PERSON_011394",
            "SIM_PERSON_013515",
        ]
        assert len([item for item in records if item.sample_result == "normal"]) == 2
        assert len([item for item in records if item.sample_result == "abnormal"]) == 8
        assert "政策口径核验：异地手工报销" in {
            item.sample_category for item in records
        }

    def test_get_visitor_record_keeps_metadata_separate_from_wide_record(self):
        record = self.service.get_record("SIM_PERSON_011872")

        assert record is not None
        assert len(record.record) == 81
        assert "RES" not in record.record
        assert record.metadata["sample_result"] == "abnormal"
        assert record.metadata["expected_rule_ids"] == ["OP-R009"]
        assert record.case_context["insured_region"] == "北京"
        assert record.case_context["treatment_region"] == "上海"
        assert "expected_res" not in record.metadata
        assert "RES" not in str(record.model_dump())


class TestIngestPipeline:
    def setup_method(self):
        self.case_service = CaseService()
        self.trace_service = TraceService()
        self.ingest_validation_service = IngestValidationService()
        self.pipeline = AuditPipelineService(
            case_service=self.case_service,
            evidence_service=EvidenceService(),
            trace_service=self.trace_service,
            validation_service=FeatureValidationService(),
            scoring_provider=OperationalRiskSignalProvider(),
            rule_service=RuleService(),
        )

    def test_ingest_pipeline_creates_case_with_safe_source_record(self):
        response = self.pipeline.run_ingest(
            IngestRecordInput(record=_fixture_record()),
            self.ingest_validation_service,
        )

        assert response.case.case_id.startswith("INGEST-")
        assert str(response.case.source_record["个人编码"]).startswith("SIM_PERSON_")
        assert response.case.subject_ref == response.case.source_record["个人编码"]
        assert response.case.rule_pool_version == "op-rule-pool-v0.3"
        assert response.case.rule_baseline_version == "mediredata-baseline-v0.2"
        assert len(response.case.rule_hits) == 12
        assert all(rule.rule_id.startswith("OP-R") for rule in response.case.rule_hits)
        assert "RES" not in response.case.source_record
        assert self.case_service.get_case(response.case.case_id) is not None

        assert "trace" not in response.model_dump()

        workflow = self.trace_service.build_workflow(response.case.case_id)
        assert [step.key for step in workflow.steps[:4]] == [
            "case_intake",
            "fact_base",
            "rule_check",
            "risk_screening",
        ]
        assert workflow.steps[5].key == "initial_review"
        assert workflow.steps[5].status == "current"
        rules_step = next(step for step in workflow.steps if step.key == "rule_check")
        assert rules_step.metadata["rule_baseline_version"] == "mediredata-baseline-v0.2"
        assert rules_step.metadata["total_rules"] == 12

    def test_visitor_samples_trigger_expected_primary_rules(self):
        visitor_catalog = VisitorIngestRecordCatalog(
            VISITOR_FIXTURE_PATH,
            self.ingest_validation_service,
            specs_path=VISITOR_SPECS_PATH,
        )
        visitor_catalog.load_fixture()

        expected = {
            "SIM_PERSON_011872": "OP-R009",
            "SIM_PERSON_006395": "OP-R010",
            "SIM_PERSON_002052": "OP-R011",
            "SIM_PERSON_014632": "OP-R012",
            "SIM_PERSON_011569": "OP-R004",
            "SIM_PERSON_002204": "OP-R003",
            "SIM_PERSON_011394": "OP-R005",
            "SIM_PERSON_013515": "OP-R001",
        }

        for record_id, rule_id in expected.items():
            sample = visitor_catalog.get_record(record_id)
            assert sample is not None
            response = self.pipeline.run_ingest(
                IngestRecordInput(
                    record=sample.record,
                    case_title=str(sample.metadata["case_title"]),
                    case_type=str(sample.metadata["case_type"]),
                    source_system=str(sample.metadata["source_system"]),
                    record_version=str(sample.metadata["record_version"]),
                    case_context=sample.case_context,
                ),
                self.ingest_validation_service,
            )
            hit_rules = {rule.rule_id for rule in response.case.rule_hits if rule.hit}
            assert rule_id in hit_rules
            assert response.case.case_context["visitor_sample"]["sample_result"] == "abnormal"

    def test_batch_ingest_creates_cases_all_or_nothing(self):
        records = [
            IngestRecordInput(record={**_fixture_record(), "个人编码": "SIM_PERSON_000001"}),
            IngestRecordInput(record={**_fixture_record(), "个人编码": "SIM_PERSON_000999"}),
        ]

        responses = self.pipeline.run_ingest_batch(
            records,
            self.ingest_validation_service,
        )

        assert len(responses) == 2
        assert self.case_service.case_count == 2
        assert responses[0].case.subject_ref == "SIM_PERSON_000001"
        assert responses[1].case.subject_ref == "SIM_PERSON_000999"

    def test_batch_ingest_rejects_invalid_row_without_partial_cases(self):
        before = self.case_service.case_count
        records = [
            IngestRecordInput(record={**_fixture_record(), "个人编码": "SIM_PERSON_000001"}),
            IngestRecordInput(
                record={**_fixture_record(), "个人编码": "SIM_PERSON_000999", "RES": "1"}
            ),
        ]

        with pytest.raises(IngestValidationError) as exc:
            self.pipeline.run_ingest_batch(records, self.ingest_validation_service)

        assert any("第 2 行" in error for error in exc.value.errors)
        assert self.case_service.case_count == before

    def test_batch_ingest_rejects_duplicate_subject_ref_without_partial_cases(self):
        before = self.case_service.case_count
        records = [
            IngestRecordInput(record={**_fixture_record(), "个人编码": "SIM_PERSON_000001"}),
            IngestRecordInput(record={**_fixture_record(), "个人编码": "SIM_PERSON_000001"}),
        ]

        with pytest.raises(IngestValidationError) as exc:
            self.pipeline.run_ingest_batch(records, self.ingest_validation_service)

        assert any("重复" in error for error in exc.value.errors)
        assert self.case_service.case_count == before

    def test_exact_duplicate_returns_existing_case_without_recalculation(self):
        body = IngestRecordInput(record=_fixture_record())

        first = self.pipeline.run_ingest(body, self.ingest_validation_service)
        second = self.pipeline.run_ingest(body, self.ingest_validation_service)

        assert first.case.case_id == second.case.case_id
        assert self.case_service.case_count == 1

    def test_changed_record_for_same_subject_creates_new_case(self):
        original = IngestRecordInput(record=_fixture_record())
        changed_record = _fixture_record()
        changed_record["ALL_SUM"] = str(float(changed_record["ALL_SUM"]) + 1)
        changed = IngestRecordInput(record=changed_record)

        first = self.pipeline.run_ingest(original, self.ingest_validation_service)
        second = self.pipeline.run_ingest(changed, self.ingest_validation_service)

        assert first.case.case_id != second.case.case_id
        assert self.case_service.case_count == 2
