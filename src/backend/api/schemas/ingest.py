"""上游接入 API 请求/响应模型。

从领域层重导出，不在 API 层新增字段。
"""

from ...domain.intake.entities import (
    BatchIngestInput,
    BatchIngestResponse,
    FeatureRecordInput,
    FeatureSchemaResponse,
    IngestRecordInput,
    IngestRecordResponse,
    IngestRecordSummary,
    IngestSchemaResponse,
)

__all__ = [
    "BatchIngestInput",
    "BatchIngestResponse",
    "FeatureRecordInput",
    "FeatureSchemaResponse",
    "IngestRecordInput",
    "IngestRecordResponse",
    "IngestRecordSummary",
    "IngestSchemaResponse",
]
