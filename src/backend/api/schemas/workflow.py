"""九阶段稽核流程 API 响应模型。

从领域层重导出，不在 API 层新增字段。
"""

from ...domain.audit.review.workflow_projector import WorkflowResponse

__all__ = ["WorkflowResponse"]
