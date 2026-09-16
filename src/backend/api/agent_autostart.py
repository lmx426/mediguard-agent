"""Review Advisor 自动预生成辅助工具。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..domain.agent.entities import EvidenceAgentRequest
from ..domain.audit.review.entities import AuthenticatedUser

if TYPE_CHECKING:
    from .dependencies import AppContainer

logger = logging.getLogger(__name__)


def prestart_review_advisor(
    container: "AppContainer",
    case_ids: str | list[str],
    actor: AuthenticatedUser,
) -> None:
    """在不影响主流程的前提下预启动研判建议。

    Review Advisor 自身通过事实指纹和配置指纹保证幂等复用。这里仅负责
    在建案/打开案件时尽早触发；Agent 不可用或启动失败时基础证据包继续可用。
    """

    service = container.evidence_agent
    if service is None:
        return
    if getattr(container.settings, "showcase_mode", False):
        return
    if getattr(container.settings, "evidence_agent_eval_variants_enabled", False):
        return

    ids = [case_ids] if isinstance(case_ids, str) else case_ids
    for case_id in ids:
        try:
            service.start(case_id, EvidenceAgentRequest(), actor)
        except Exception as exc:  # pragma: no cover - fail-open safety
            logger.warning(
                "Review Advisor prestart skipped for case %s: %s",
                case_id,
                exc.__class__.__name__,
            )


prestart_evidence_agent = prestart_review_advisor
