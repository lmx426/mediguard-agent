"""Review Advisor 路由。

提供受控 Agent 分析触发、运行状态查询、当前分析读取和 SSE 事件流端点。
Agent 不可用时返回 503，基础证据包继续可用。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ..dependencies import (
    AppContainer,
    get_container,
    get_current_user,
    get_evidence_agent,
)
from ...application.agent.evidence_agent.service import ReviewAdvisorService
from ...domain.agent.entities import (
    AgentRun,
    EvidenceAgentAnalysis,
    EvidenceAgentRequest,
)
from ...domain.audit.review.entities import AuthenticatedUser

router = APIRouter(tags=["agent"])
#: Agent 运行终止状态集合
TERMINAL_STATUSES = {"complete", "partial", "failed"}


@router.get("/agent/status")
def review_advisor_status(
    container: AppContainer = Depends(get_container),
) -> dict:
    """返回 Agent 就绪状态（无需登录，不暴露配置或内部错误）。

    参数:
        container: 应用依赖容器。

    返回:
        dict: 包含 enabled / available / provider / model / reason 等字段。
    """

    return container.evidence_agent_status


@router.post("/cases/{case_id}/evidence-agent/runs", response_model=AgentRun)
def start_review_advisor_run(
    case_id: str,
    body: EvidenceAgentRequest,
    actor: AuthenticatedUser = Depends(get_current_user),
    service: ReviewAdvisorService = Depends(get_evidence_agent),
) -> AgentRun:
    """人工重新生成或复用一次综合证据研判。

    参数:
        case_id: 案件唯一标识。
        body: 综合分析请求。
        actor: 当前登录审核人员。
        service: Review Advisor 服务。

    返回:
        AgentRun: 创建或复用的 Agent 运行记录。

    异常:
        HTTPException 404: 案件不存在。
    """

    try:
        return service.start(case_id, body, actor)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="案件不存在"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get("/cases/{case_id}/evidence-agent/run", response_model=AgentRun | None)
def get_latest_review_advisor_run(
    case_id: str,
    _actor: AuthenticatedUser = Depends(get_current_user),
    service: ReviewAdvisorService = Depends(get_evidence_agent),
) -> AgentRun | None:
    """返回当前案件最近一次或正在执行的综合证据研判运行。

    参数:
        case_id: 案件唯一标识。
        service: Review Advisor 服务。

    返回:
        AgentRun | None: 优先返回未完成运行；不存在时返回 None。
    """

    return service.get_latest_run(case_id)


@router.get("/evidence-agent/runs/{run_id}", response_model=AgentRun)
def get_review_advisor_run(
    run_id: str,
    _actor: AuthenticatedUser = Depends(get_current_user),
    service: ReviewAdvisorService = Depends(get_evidence_agent),
) -> AgentRun:
    """查询一次 Agent 运行的当前状态。

    参数:
        run_id: Agent 运行唯一标识。
        service: Review Advisor 服务。

    返回:
        AgentRun: 运行状态（queued/running/complete/partial/failed）。

    异常:
        HTTPException 404: 运行记录不存在。
    """

    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent 运行不存在"
        )
    return run


@router.get(
    "/cases/{case_id}/evidence-agent/current",
    response_model=EvidenceAgentAnalysis | None,
)
def get_current_review_advisor_analysis(
    case_id: str,
    _actor: AuthenticatedUser = Depends(get_current_user),
    service: ReviewAdvisorService = Depends(get_evidence_agent),
) -> EvidenceAgentAnalysis | None:
    """返回当前案件最新的综合证据研判结果。

    参数:
        case_id: 案件唯一标识。
        service: Review Advisor 服务。

    返回:
        EvidenceAgentAnalysis | None: 最新分析结果，不存在时返回 None。
    """

    return service.get_current_analysis(case_id)


@router.get("/evidence-agent/runs/{run_id}/events")
def stream_review_advisor_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    _actor: AuthenticatedUser = Depends(get_current_user),
    service: ReviewAdvisorService = Depends(get_evidence_agent),
) -> StreamingResponse:
    """SSE 事件流：实时推送 Agent 运行过程中的业务事件。

    参数:
        run_id: Agent 运行唯一标识。
        after: 从指定序号之后开始推送（用于断线重连）。
        last_event_id: SSE 标准断线重连游标。
        service: Review Advisor 服务。

    返回:
        StreamingResponse: text/event-stream 格式的事件流。
    """

    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent 运行不存在"
        )
    cursor = max(after, int(last_event_id or 0))
    return StreamingResponse(
        _event_stream(service, run_id, cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _event_stream(
    service: ReviewAdvisorService,
    run_id: str,
    cursor: int,
) -> Iterator[str]:
    """SSE 事件生成器。

    每 0.5 秒轮询一次新事件；运行终止且无新事件时结束；
    每 15 秒发送一次 keep-alive 注释防止代理超时断开。

    参数:
        service: Review Advisor 服务。
        run_id: Agent 运行唯一标识。
        cursor: 当前事件游标序号。

    Yields:
        str: SSE 格式的事件字符串。
    """

    last_heartbeat = time.monotonic()
    while True:
        events = service.list_events(run_id, cursor)
        for event in events:
            cursor = event.sequence
            payload = event.model_dump(mode="json")
            yield (
                f"id: {event.sequence}\n"
                f"event: {event.event_type}\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )
        run = service.get_run(run_id)
        if run is None or (run.status in TERMINAL_STATUSES and not events):
            break
        if time.monotonic() - last_heartbeat >= 15:
            yield ": keep-alive\n\n"
            last_heartbeat = time.monotonic()
        time.sleep(0.5)
