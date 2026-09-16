"""领域实体与稳定数据结构（兼容性重导出）。

本模块保留用于向后兼容，实际实体定义已迁移至业务模块：
- domain/audit/review/entities.py
- domain/audit/review/evidence_packager.py
- domain/audit/review/workflow_projector.py
- domain/intake/entities.py
- domain/agent/entities.py
"""

# Agent 实体 → domain/agent/entities.py
from ..agent.entities import (
    AgentCitation,
    AgentEvent,
    AgentGeneratedContent,
    AgentRun,
    AgentStatement,
    AgentVerificationItem,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
    EvidenceLedgerItem,
)

__all__ = [
    "AgentCitation",
    "AgentEvent",
    "AgentGeneratedContent",
    "AgentRun",
    "AgentStatement",
    "AgentVerificationItem",
    "EvidenceAgentAnalysis",
    "EvidenceAgentRequest",
    "EvidenceLedgerItem",
]
