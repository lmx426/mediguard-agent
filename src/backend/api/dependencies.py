"""FastAPI 依赖容器与集中依赖定义。

构建应用全部运行时依赖的有向无环图（DAG），
按 Settings 选择内存或 PostgreSQL 持久化后端，
并通过 FastAPI Depends 函数暴露给路由层。

这是整个后端 DI 的核心枢纽，所有路由通过此模块获取用例和服务。
"""

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException, Request, status

# ── 端口（Protocol 定义）──────────────────────────────
from ..application.ports.repositories import (
    CaseRepository,
    MaterialRepository,
    NoteRepository,
    ReviewRepository,
    TraceRepository,
    UserRepository,
)

# ── 用例 ─────────────────────────────────────────────
from ..application.intake.build_case_pipeline import AuditPipelineService
from ..application.audit.queue.list_cases_uc import ListCasesUseCase
from ..application.audit.review.get_case_detail_uc import GetCaseDetailUseCase
from ..application.audit.review.submit_review_uc import ReviewCaseUseCase
from ..application.audit.review.manage_notes_uc import ManageNotesUseCase
from ..application.audit.review.build_materials_uc import StatisticalMaterialUseCase

# ── Agent 运行时 ──────────────────────────────────────
from ..application.agent.case_agent.service import CaseAgentService
from ..application.agent.case_agent.context.refresh_service import (
    CaserContextRefreshService,
)
from ..application.agent.evidence_agent.service import ReviewAdvisorService
from ..application.showcase import ShowcaseReviewAdvisorService
from ..application.case_memory.service import CaseMemoryService

# ── 核心配置 ─────────────────────────────────────────
from ..core.config import Settings
from ..core.security import read_auth_token

# ── 领域模型 ─────────────────────────────────────────
from ..domain.audit.review.entities import AuthenticatedUser, StoredUser

# ── 领域服务 ─────────────────────────────────────────
from ..domain.audit.review.evidence_packager import EvidenceService
from ..domain.intake.validation_rules import (
    FeatureValidationService,
    IngestValidationService,
)
from ..domain.audit.review.risk_engine import OperationalRiskSignalProvider
from ..domain.audit.review.rule_evaluator import RuleService

# ── 欺诈模型 ─────────────────────────────────────────
from ..application.fraud.run_screening_uc import (
    FraudModelService,
    PrecomputedFraudModelService,
)

# ── 基础设施 ─────────────────────────────────────────
from ..infrastructure.fixtures_loader import (
    FixturesLoader,
    IngestRecordCatalog,
    VisitorIngestRecordCatalog,
)
from ..infrastructure.model_assets.local_loader import LocalAssetsLoader
from ..infrastructure.persistence.memory.case_repository import CaseService
from ..infrastructure.persistence.memory.material_repository import MemoryMaterialRepository
from ..infrastructure.persistence.memory.note_repository import MemoryNoteRepository
from ..infrastructure.persistence.memory.review_repository import MemoryReviewRepository
from ..infrastructure.persistence.memory.trace_repository import TraceService
from ..infrastructure.persistence.memory.user_repository import MemoryUserRepository
from ..infrastructure.persistence.memory.case_memory_repository import (
    InMemoryEntityGraphProjectionRepository,
    InMemoryMemoMemoryAdapter,
    InMemoryMemoryEventRepository,
    InMemoryMemoryOutboxRepository,
    InMemoryMemoryRepository,
)
from ..infrastructure.mem0_adapter import Mem0MemoMemoryAdapter


@dataclass
class AppContainer:
    """一个应用实例的全部运行时依赖。

    属性:
        settings: 应用配置。
        cases: 案件仓储。
        traces: 流程追踪仓储。
        reviews: 初审记录仓储。
        notes: 工作笔记仓储。
        users: 审核人员账号仓储。
        evidence: 确定性证据包生成服务。
        feature_validation: 统计特征校验服务。
        ingest_validation: 81 字段接入校验服务。
        ingest_records: 脱敏样本记录目录。
        ingest_cases: 建案流水线。
        list_cases: 案件列表用例。
        get_case_detail: 案件详情用例。
        review_case: 人工初审用例。
        manage_notes: 工作笔记管理用例。
        statistical_materials: 统计材料用例。
        fraud_model: 本地欺诈模型推理服务。
        fixtures: 启动时 fixture 加载器。
        database_status: 数据库健康检查回调。
        persistent_storage: 是否使用持久化存储。
        evidence_agent: Review Advisor 服务（兼容旧容器字段，可能为 None）。
        evidence_agent_status: Review Advisor 就绪状态字典。
        case_agent: Case Agent 服务（可能为 None）。
        case_agent_status: Case Agent 就绪状态字典。
    """

    settings: Settings
    cases: CaseRepository
    traces: TraceRepository
    reviews: ReviewRepository
    notes: NoteRepository
    users: UserRepository
    materials: MaterialRepository
    evidence: EvidenceService
    feature_validation: FeatureValidationService
    ingest_validation: IngestValidationService
    ingest_records: IngestRecordCatalog
    visitor_ingest_records: VisitorIngestRecordCatalog
    ingest_cases: AuditPipelineService
    list_cases: ListCasesUseCase
    get_case_detail: GetCaseDetailUseCase
    review_case: ReviewCaseUseCase
    manage_notes: ManageNotesUseCase
    statistical_materials: StatisticalMaterialUseCase
    fraud_model: FraudModelService | PrecomputedFraudModelService
    fixtures: FixturesLoader
    database_status: Callable[[], dict[str, Any]]
    persistent_storage: bool
    evidence_agent: ReviewAdvisorService | ShowcaseReviewAdvisorService | None
    evidence_agent_status: dict[str, Any]
    caser_context: CaserContextRefreshService | None
    caser_context_status: dict[str, Any]
    case_agent: CaseAgentService | None
    case_agent_status: dict[str, Any]
    case_memory: CaseMemoryService | None
    case_memory_status: dict[str, Any]


def build_container(settings: Settings) -> AppContainer:
    """为单个 FastAPI 应用构建隔离的依赖图。

    根据 ``settings.persistence_backend`` 选择内存或 PostgreSQL 实现，
    并依次构建领域服务 → 建案流水线 → 用例 → Agent → Fixtures。

    参数:
        settings: 应用配置对象。

    返回:
        AppContainer: 组装完成的依赖容器。
    """

    # 第一步：选择持久化后端
    (
        cases,
        traces,
        reviews,
        notes,
        users,
        database_status,
        persistent_storage,
        sql_session_factory,
        materials,
    ) = _build_persistence(settings)

    # 第二步：领域服务（纯业务逻辑，无副作用）
    evidence = EvidenceService()
    feature_validation = FeatureValidationService()
    ingest_validation = IngestValidationService()
    risk = OperationalRiskSignalProvider()
    rules = RuleService()

    # 第三步：欺诈模型推理服务（本地 XGBoost）
    if settings.showcase_mode:
        fraud_model = PrecomputedFraudModelService(
            settings.showcase_model_results_path
        )
    else:
        model_assets = LocalAssetsLoader(settings.fraud_model_assets_dir)
        fraud_model = FraudModelService(
            assets_loader=model_assets,
            enabled=settings.fraud_model_enabled,
        )

    # 第四步：脱敏样本记录目录
    ingest_records = IngestRecordCatalog(
        fixture_path=settings.ingest_records_path,
        validation_service=ingest_validation,
    )
    visitor_ingest_records = VisitorIngestRecordCatalog(
        fixture_path=settings.visitor_ingest_records_path,
        validation_service=ingest_validation,
        specs_path=settings.visitor_sample_specs_path,
    )

    # 第五步：建案流水线（编排所有领域服务）
    ingest_cases = AuditPipelineService(
        case_service=cases,
        evidence_service=evidence,
        trace_service=traces,
        validation_service=feature_validation,
        scoring_provider=risk,
        rule_service=rules,
        fraud_model_service=fraud_model,
        review_repository=reviews,
        note_repository=notes,
    )

    # 第六步：用例（薄编排层）
    list_cases = ListCasesUseCase(cases, reviews)
    get_case_detail = GetCaseDetailUseCase(
        cases,
        reviews,
        notes,
        traces,
        evidence,
    )
    review_case = ReviewCaseUseCase(cases, reviews, traces)
    manage_notes = ManageNotesUseCase(cases, notes, reviews)
    statistical_materials = StatisticalMaterialUseCase(
        material_repository=materials,
        asset_root=settings.material_assets_dir,
        api_prefix=settings.api_prefix,
    )

    # 第七步：Review Advisor（条件构建，不可用时不影响基础流程）
    evidence_agent, evidence_agent_status = _build_review_advisor(
        settings=settings,
        cases=cases,
        evidence=evidence,
        statistical_materials=statistical_materials,
        sql_session_factory=sql_session_factory,
        persistent_storage=persistent_storage,
    )
    caser_context, caser_context_status = _build_caser_context(
        get_case_detail=get_case_detail,
        statistical_materials=statistical_materials,
        evidence_agent=evidence_agent,
        sql_session_factory=sql_session_factory,
        persistent_storage=persistent_storage,
    )
    if caser_context is not None:
        set_ingest_events = getattr(ingest_cases, "set_caser_context_events", None)
        if callable(set_ingest_events):
            set_ingest_events(caser_context)
        set_material_events = getattr(statistical_materials, "set_caser_context_events", None)
        if callable(set_material_events):
            set_material_events(caser_context)
        set_review_events = getattr(evidence_agent, "set_caser_context_events", None)
        if callable(set_review_events):
            set_review_events(caser_context)
    case_memory, case_memory_status = _build_case_memory(
        settings=settings,
        sql_session_factory=sql_session_factory,
        persistent_storage=persistent_storage,
        users=users,
    )
    case_agent, case_agent_status = _build_case_agent(
        settings=settings,
        get_case_detail=get_case_detail,
        manage_notes=manage_notes,
        statistical_materials=statistical_materials,
        evidence_agent=evidence_agent,
        caser_context=caser_context,
        sql_session_factory=sql_session_factory,
        persistent_storage=persistent_storage,
        case_memory=case_memory,
    )

    # 第八步：Fixtures 加载器
    fixtures = FixturesLoader(
        cases=cases,
        traces=traces,
        reviews=reviews,
        notes=notes,
        pipeline=ingest_cases,
        ingest_records=ingest_records,
        visitor_ingest_records=visitor_ingest_records,
        reset_runtime_state=not persistent_storage,
        showcase_record_ids=(
            settings.showcase_record_ids if settings.showcase_mode else []
        ),
    )

    return AppContainer(
        settings=settings,
        cases=cases,
        traces=traces,
        reviews=reviews,
        notes=notes,
        users=users,
        materials=materials,
        evidence=evidence,
        feature_validation=feature_validation,
        ingest_validation=ingest_validation,
        ingest_records=ingest_records,
        ingest_cases=ingest_cases,
        list_cases=list_cases,
        get_case_detail=get_case_detail,
        review_case=review_case,
        manage_notes=manage_notes,
        statistical_materials=statistical_materials,
        fraud_model=fraud_model,
        fixtures=fixtures,
        database_status=database_status,
        persistent_storage=persistent_storage,
        evidence_agent=evidence_agent,
        evidence_agent_status=evidence_agent_status,
        caser_context=caser_context,
        caser_context_status=caser_context_status,
        case_agent=case_agent,
        case_agent_status=case_agent_status,
        case_memory=case_memory,
        case_memory_status=case_memory_status,
        visitor_ingest_records=visitor_ingest_records,
    )


def _build_case_memory(
    *,
    settings: Settings,
    sql_session_factory: Any | None,
    persistent_storage: bool,
    users: UserRepository,
) -> tuple[CaseMemoryService | None, dict[str, Any]]:
    """Build the governed memory service without exposing Mem0 types."""

    base_status = {
        "enabled": settings.case_memory_enabled,
        "available": False,
        "persistence": "postgres" if persistent_storage else "memory",
        "engine": (
            "mem0ai/mem0"
            if settings.mem0_enabled and persistent_storage
            else (
                "postgres_sql_graph_bm25"
                if persistent_storage
                else "in_memory_sql_graph_bm25"
            )
        ),
        "graph_backend": settings.mem0_graph_backend,
        "vector_backend": "pgvector" if settings.mem0_enabled and persistent_storage else "disabled",
        "fallback": "sql_graph_bm25" if persistent_storage else "in_memory",
    }
    if not settings.case_memory_enabled:
        return None, {**base_status, "reason": "feature_disabled"}

    if persistent_storage and sql_session_factory is not None:
        from ..infrastructure.persistence.sql.repositories.memory_repo import (
            SqlEntityGraphProjectionRepository,
            SqlMemoryEventRepository,
            SqlMemoryOutboxRepository,
            SqlMemoryRepository,
        )

        repository = SqlMemoryRepository(sql_session_factory)
        events = SqlMemoryEventRepository(sql_session_factory)
        outbox = SqlMemoryOutboxRepository(sql_session_factory)
        graph = SqlEntityGraphProjectionRepository(sql_session_factory)
    else:
        repository = InMemoryMemoryRepository()
        events = InMemoryMemoryEventRepository()
        outbox = InMemoryMemoryOutboxRepository()
        graph = InMemoryEntityGraphProjectionRepository()

    secret = settings.expert_analysis_deepseek_api_key or settings.deepseek_api_key
    api_key = secret.get_secret_value().strip() if secret is not None else ""
    memo: Any = Mem0MemoMemoryAdapter(
        enabled=settings.mem0_enabled and persistent_storage,
        database_url=settings.database_url,
        runtime_dir=settings.mem0_dir,
        llm_model=settings.mem0_llm_model,
        llm_base_url=settings.mem0_llm_base_url,
        llm_api_key=api_key,
        embedding_model=settings.mem0_embedding_model,
        embedding_cache_dir=settings.mem0_embedding_cache_dir,
        projection_collection=settings.mem0_projection_collection,
    )
    if not persistent_storage:
        memo = InMemoryMemoMemoryAdapter()
    service = CaseMemoryService(repository, events, outbox, memo, graph, users)
    return service, {**base_status, "available": True, "reason": None, "mem0_enabled": bool(getattr(memo, "enabled", False))}


def _build_persistence(
    settings: Settings,
) -> tuple[
    CaseRepository,
    TraceRepository,
    ReviewRepository,
    NoteRepository,
    UserRepository,
    Callable[[], dict[str, Any]],
    bool,
    Any | None,
    MaterialRepository,
]:
    """根据配置选择内存或 PostgreSQL 持久化后端。

    参数:
        settings: 应用配置。

    返回:
        tuple: (cases, traces, reviews, notes, users, database_status, persistent_storage, sql_session_factory)

    异常:
        RuntimeError: PostgreSQL 模式缺少必要依赖。
    """

    if settings.persistence_backend == "memory":
        # 进程内内存仓储：服务重启后数据清空
        return (
            CaseService(),
            TraceService(),
            MemoryReviewRepository(),
            MemoryNoteRepository(),
            MemoryUserRepository(settings.default_auditor_password),
            lambda: {
                "enabled": False,
                "backend": "memory",
                "connected": None,
            },
            False,
            None,
            MemoryMaterialRepository(),
        )

    # PostgreSQL 持久化后端
    try:
        from ..infrastructure.persistence.sql.repositories.cases_repo import (
            SqlCaseRepository,
        )
        from ..infrastructure.persistence.sql.repositories.notes_repo import (
            SqlNoteRepository,
        )
        from ..infrastructure.persistence.sql.repositories.reviews_repo import (
            SqlReviewRepository,
        )
        from ..infrastructure.persistence.sql.repositories.traces_repo import (
            SqlTraceRepository,
        )
        from ..infrastructure.persistence.sql.repositories.users_repo import (
            SqlUserRepository,
        )
        from ..infrastructure.persistence.sql.repositories.materials_repo import (
            SqlMaterialRepository,
        )
        from ..infrastructure.persistence.sql.session import (
            create_database_engine,
            create_session_factory,
            database_status,
            seed_default_auditor,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PostgreSQL 持久化需要 sqlalchemy>=2、alembic 和 psycopg[binary]。"
            "请安装 environment.yml 中的依赖后再设置 MEDIGUARD_PERSISTENCE_BACKEND=postgres。"
        ) from exc

    engine = create_database_engine(settings)
    # This checkout does not include the historical Alembic script tree.
    # create_all is idempotent and keeps the memory tables available while
    # the checked-in migration artifact is applied in deployment.
    from ..infrastructure.persistence.sql.models import Base

    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    # 确保默认审核员账号存在
    seed_default_auditor(session_factory, settings)
    return (
        SqlCaseRepository(session_factory),
        SqlTraceRepository(session_factory),
        SqlReviewRepository(session_factory),
        SqlNoteRepository(session_factory),
        SqlUserRepository(session_factory),
        lambda: database_status(engine, settings.persistence_backend),
        True,
        session_factory,
        SqlMaterialRepository(session_factory),
    )


def _build_review_advisor(
    *,
    settings: Settings,
    cases: CaseRepository,
    evidence: EvidenceService,
    statistical_materials: StatisticalMaterialUseCase,
    sql_session_factory: Any | None,
    persistent_storage: bool,
) -> tuple[ReviewAdvisorService | ShowcaseReviewAdvisorService | None, dict[str, Any]]:
    """条件构建 Review Advisor 服务。

    仅在所有前置条件满足时创建 Agent；
    任一条件不满足则返回 None，基础证据包继续可用。

    前置条件（必须全部满足）：
    1. evidence_agent_enabled = True
    2. 使用 PostgreSQL 持久化
    3. llm_provider = "deepseek"
    4. deepseek_api_key 已配置

    参数:
        settings: 应用配置。
        cases: 案件仓储。
        evidence: 证据包生成服务。
        statistical_materials: 统计材料用例。
        sql_session_factory: SQLAlchemy 会话工厂。
        persistent_storage: 是否持久化模式。

    返回:
        tuple: (ReviewAdvisorService | None, status_dict)
    """

    if settings.showcase_mode:
        service = ShowcaseReviewAdvisorService(cases, evidence)
        return service, {
            "enabled": True,
            "available": True,
            "provider": "precomputed",
            "model": "showcase-precomputed",
            "persistence": "postgres" if persistent_storage else "memory",
            "thinking_enabled": False,
            "reason": None,
            "showcase": True,
            "read_only": True,
        }

    provider = settings.review_advisor_provider or settings.llm_provider
    base_url = settings.review_advisor_base_url or settings.llm_base_url
    model = settings.review_advisor_model or settings.llm_model
    secret = settings.review_advisor_deepseek_api_key or settings.deepseek_api_key
    thinking_enabled = (
        settings.llm_thinking_enabled
        if settings.review_advisor_thinking_enabled is None
        else settings.review_advisor_thinking_enabled
    )
    reasoning_effort = (
        settings.review_advisor_reasoning_effort or settings.llm_reasoning_effort
    )

    base_status: dict[str, Any] = {
        "enabled": settings.evidence_agent_enabled,
        "available": False,
        "provider": provider,
        "model": model,
        "persistence": "postgres" if persistent_storage else "memory",
        "thinking_enabled": thinking_enabled,
    }

    # 逐层检查前置条件
    if not settings.evidence_agent_enabled:
        return None, {**base_status, "reason": "feature_disabled"}
    if not persistent_storage or sql_session_factory is None:
        return None, {**base_status, "reason": "postgres_required"}
    if provider != "deepseek":
        return None, {**base_status, "reason": "unsupported_provider"}

    api_key = secret.get_secret_value().strip() if secret is not None else ""
    if not api_key:
        return None, {**base_status, "reason": "api_key_missing"}

    # 条件满足：构建完整 Agent 服务栈
    try:
        from ..application.agent.evidence_agent.tools import ReviewAdvisorToolRegistry
        from ..application.agent.runtime.gateways.deepseek import DeepSeekModelGateway
        from ..infrastructure.persistence.sql.repositories.agent_repo import (
            SqlEvidenceAgentRepository,
        )

        repository = SqlEvidenceAgentRepository(sql_session_factory)
        gateway = DeepSeekModelGateway(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=settings.evidence_agent_timeout_seconds,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        )
        tools = ReviewAdvisorToolRegistry(
            cases,
            evidence,
            statistical_materials,
        )
        service = ReviewAdvisorService(
            settings=settings,
            cases=cases,
            repository=repository,
            gateway=gateway,
            tools=tools,
        )
        return service, {**base_status, "available": True, "reason": None}
    except Exception as exc:
        # Agent 初始化失败不能影响基础业务
        return None, {
            **base_status,
            "reason": "initialization_failed",
            "error_type": exc.__class__.__name__,
        }


def _build_caser_context(
    *,
    get_case_detail: GetCaseDetailUseCase,
    statistical_materials: StatisticalMaterialUseCase,
    evidence_agent: ReviewAdvisorService | None,
    sql_session_factory: Any | None,
    persistent_storage: bool,
) -> tuple[CaserContextRefreshService | None, dict[str, Any]]:
    """Build the Caser context View Store service."""

    base_status: dict[str, Any] = {
        "enabled": persistent_storage,
        "available": False,
        "persistence": "postgres" if persistent_storage else "memory",
        "section_count": 12,
    }
    if not persistent_storage or sql_session_factory is None:
        return None, {**base_status, "reason": "postgres_required"}
    try:
        from ..application.agent.case_agent.context import (
            CaserContextEventBus,
            CaserContextRefreshService,
            CaserSectionBuilder,
        )
        from ..infrastructure.persistence.sql.repositories.caser_context_repo import (
            SqlCaserContextRepository,
        )

        repository = SqlCaserContextRepository(sql_session_factory)
        event_bus = CaserContextEventBus(max_workers=2)
        builder = CaserSectionBuilder(
            get_case_detail=get_case_detail,
            statistical_materials=statistical_materials,
            evidence_agent=evidence_agent,
        )
        service = CaserContextRefreshService(
            repository=repository,
            builder=builder,
            event_bus=event_bus,
        )
        return service, {**base_status, "available": True, "reason": None}
    except Exception as exc:
        return None, {
            **base_status,
            "reason": "initialization_failed",
            "error_type": exc.__class__.__name__,
        }


def _build_case_agent(
    *,
    settings: Settings,
    get_case_detail: GetCaseDetailUseCase,
    manage_notes: ManageNotesUseCase,
    statistical_materials: StatisticalMaterialUseCase,
    evidence_agent: ReviewAdvisorService | None,
    caser_context: CaserContextRefreshService | None,
    sql_session_factory: Any | None,
    persistent_storage: bool,
    case_memory: CaseMemoryService | None,
) -> tuple[CaseAgentService | None, dict[str, Any]]:
    """条件构建 Case Agent 服务。

    第一版仅支持 PostgreSQL 权威存储和 Kimi API。不可用时返回 None，
    不影响基础审核流程、Review Advisor 或工作笔记。
    """

    if settings.showcase_mode:
        return None, {
            "enabled": True,
            "available": True,
            "provider": "precomputed",
            "model": "showcase-precomputed",
            "classifier_model": "showcase-precomputed",
            "generator_model": "showcase-precomputed",
            "persistence": "postgres" if persistent_storage else "memory",
            "fast_mode_enabled": False,
            "intent_biencoder_enabled": False,
            "reason": None,
            "showcase": True,
            "read_only": True,
        }

    provider = settings.case_agent_provider
    classifier_model = settings.case_agent_classifier_model or settings.case_agent_model
    generator_model = settings.case_agent_generator_model or settings.case_agent_model

    def _secret_value(secret: Any | None) -> str:
        if secret is None:
            return ""
        return secret.get_secret_value().strip()

    expert_secret = settings.expert_analysis_deepseek_api_key
    expert_api_key = _secret_value(expert_secret)

    base_status: dict[str, Any] = {
        "enabled": settings.case_agent_enabled,
        "available": False,
        "provider": provider,
        "model": settings.case_agent_model,
        "classifier_model": classifier_model,
        "generator_model": generator_model,
        "persistence": "postgres" if persistent_storage else "memory",
        "fast_mode_enabled": settings.case_agent_fast_mode_enabled,
        "intent_biencoder_enabled": settings.case_agent_intent_biencoder_enabled,
        "intent_biencoder_model": settings.case_agent_intent_encoder_model,
        "intent_biencoder_backend": settings.case_agent_intent_biencoder_backend,
        "intent_biencoder_index_dir": str(settings.case_agent_intent_index_dir),
    }

    if not settings.case_agent_enabled:
        return None, {**base_status, "reason": "feature_disabled"}
    if not persistent_storage or sql_session_factory is None:
        return None, {**base_status, "reason": "postgres_required"}
    if caser_context is None:
        return None, {**base_status, "reason": "caser_context_required"}

    if provider == "deepseek":
        secret = settings.case_agent_deepseek_api_key or settings.deepseek_api_key
    elif provider == "kimi":
        secret = settings.case_agent_kimi_api_key
    else:
        return None, {**base_status, "reason": "unsupported_provider"}

    api_key = _secret_value(secret)
    if not api_key:
        return None, {**base_status, "reason": "api_key_missing"}

    try:
        from ..application.agent.case_agent.intent_examples import IntentExampleMatcher
        from ..application.agent.case_agent.tools import CaseAgentToolRegistry
        from ..application.agent.expert_agent import ExpertAgentService
        from ..application.agent.expert_agent.tools.policy_rag_mcp import (
            LazyPolicyRagMcpClient,
        )
        from ..application.agent.runtime.gateways.deepseek import DeepSeekModelGateway
        from ..application.agent.runtime.gateways.kimi import KimiModelGateway
        from ..infrastructure.persistence.sql.repositories.case_agent_repo import (
            SqlCaseAgentRepository,
        )

        repository = SqlCaseAgentRepository(sql_session_factory)
        if provider == "deepseek":
            gateway = DeepSeekModelGateway(
                api_key=api_key,
                base_url=settings.case_agent_base_url,
                model=settings.case_agent_model,
                timeout_seconds=settings.case_agent_timeout_seconds,
                thinking_enabled=settings.case_agent_thinking_enabled,
                reasoning_effort=settings.case_agent_reasoning_effort,
            )
        else:
            gateway = KimiModelGateway(
                api_key=api_key,
                base_url=settings.case_agent_base_url,
                model=settings.case_agent_model,
                timeout_seconds=settings.case_agent_timeout_seconds,
            )
        expert_gateway = None
        if expert_api_key:
            expert_gateway = DeepSeekModelGateway(
                api_key=expert_api_key,
                base_url=settings.expert_analysis_base_url,
                model=settings.expert_analysis_model,
                timeout_seconds=settings.expert_analysis_timeout_seconds,
                thinking_enabled=settings.expert_analysis_thinking_enabled,
                reasoning_effort=settings.expert_analysis_reasoning_effort,
            )
        tools = CaseAgentToolRegistry(
            caser_context=caser_context,
        )
        intent_biencoder = None
        intent_biencoder_status: dict[str, Any] = {"available": False, "reason": "disabled"}
        if settings.case_agent_intent_biencoder_enabled:
            try:
                intent_biencoder = IntentExampleMatcher(
                    examples_path=settings.case_agent_intent_examples_path,
                    taxonomy_path=settings.case_agent_intent_taxonomy_path,
                    index_dir=settings.case_agent_intent_index_dir,
                    encoder_model=settings.case_agent_intent_encoder_model,
                    model_cache_dir=settings.case_agent_intent_model_cache_dir,
                    backend=settings.case_agent_intent_biencoder_backend,
                    high_confidence_top1=settings.case_agent_intent_high_confidence_top1,
                    high_confidence_margin=settings.case_agent_intent_high_confidence_margin,
                    medium_confidence_top1=settings.case_agent_intent_medium_confidence_top1,
                )
                intent_biencoder_metadata = intent_biencoder.prewarm()
                intent_biencoder_status = {
                    "available": True,
                    "reason": None,
                    "metadata": intent_biencoder_metadata,
                }
            except Exception as exc:
                intent_biencoder = None
                intent_biencoder_status = {
                    "available": False,
                    "reason": "initialization_failed",
                    "error_type": exc.__class__.__name__,
                }
        policy_rag_client = LazyPolicyRagMcpClient(
            enable_reranker=settings.policy_rag_runtime_reranker_enabled,
        )
        expert_agent = ExpertAgentService()
        service = CaseAgentService(
            repository=repository,
            gateway=gateway,
            tools=tools,
            expert_agent=expert_agent,
            manage_notes=manage_notes,
            model_name=settings.case_agent_model,
            classifier_model=classifier_model,
            generator_model=generator_model,
            max_concurrency=settings.case_agent_max_concurrency,
            max_tool_calls=settings.case_agent_max_tool_calls,
            fast_mode_enabled=settings.case_agent_fast_mode_enabled,
            caser_context=caser_context,
            policy_rag_client=policy_rag_client,
            policy_rag_timeout_ms=settings.case_agent_timeout_seconds * 1000,
            expert_gateway=expert_gateway,
            intent_biencoder=intent_biencoder,
            case_memory=case_memory,
            semantic_timeout_seconds=settings.case_agent_semantic_timeout_seconds,
            planner_timeout_seconds=settings.case_agent_planner_timeout_seconds,
            answer_timeout_seconds=settings.case_agent_answer_timeout_seconds,
            semantic_max_tokens=settings.case_agent_semantic_max_tokens,
            planner_max_tokens=settings.case_agent_planner_max_tokens,
            answer_max_tokens=settings.case_agent_answer_max_tokens,
        )
        return service, {
            **base_status,
            "available": True,
            "reason": None,
            "intent_biencoder": intent_biencoder_status,
        }
    except Exception as exc:
        return None, {
            **base_status,
            "reason": "initialization_failed",
            "error_type": exc.__class__.__name__,
        }


# ── FastAPI Depends 函数 ──────────────────────────────


def get_container(request: Request) -> AppContainer:
    """获取当前应用实例的依赖容器。

    参数:
        request: FastAPI Request 对象。

    返回:
        AppContainer: 存储在 app.state 中的依赖容器。
    """

    return request.app.state.container


def get_ingest_cases(request: Request) -> AuditPipelineService:
    """注入建案流水线用例。"""
    return get_container(request).ingest_cases


def get_ingest_validation(request: Request) -> IngestValidationService:
    """注入 81 字段接入校验服务。"""
    return get_container(request).ingest_validation


def get_feature_validation(request: Request) -> FeatureValidationService:
    """注入统计特征校验服务。"""
    return get_container(request).feature_validation


def get_ingest_records(request: Request) -> IngestRecordCatalog:
    """注入脱敏样本记录目录。"""
    return get_container(request).ingest_records


def get_visitor_ingest_records(request: Request) -> VisitorIngestRecordCatalog:
    """注入游客演示样本记录目录。"""

    return get_container(request).visitor_ingest_records


def get_list_cases(request: Request) -> ListCasesUseCase:
    """注入案件列表查询用例。"""
    return get_container(request).list_cases


def get_case_detail(request: Request) -> GetCaseDetailUseCase:
    """注入案件详情查询用例。"""
    return get_container(request).get_case_detail


def get_review_case(request: Request) -> ReviewCaseUseCase:
    """注入人工初审用例。"""
    return get_container(request).review_case


def get_manage_notes(request: Request) -> ManageNotesUseCase:
    """注入工作笔记管理用例。"""
    return get_container(request).manage_notes


def get_statistical_materials(request: Request) -> StatisticalMaterialUseCase:
    """注入统计材料视图用例。"""
    return get_container(request).statistical_materials


def get_current_user(request: Request) -> AuthenticatedUser:
    """解析并校验当前登录审核人员。

    从请求 Cookie 中读取会话令牌，验证后返回审核人员信息。
    未登录或令牌无效时返回 401。

    参数:
        request: FastAPI Request 对象。

    返回:
        AuthenticatedUser: 当前登录审核人员。

    异常:
        HTTPException 401: Cookie 缺失、令牌无效或账号不可用。
    """

    container = get_container(request)
    token = request.cookies.get(container.settings.auth_cookie_name)
    claims = read_auth_token(token, container.settings.auth_secret_key)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录审核人员账号",
        )

    user_id = str(claims.get("sub") or "")
    user = container.users.get_by_id(user_id)
    if not _is_allowed_user(user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
        )
    return user.to_authenticated()


def get_evidence_agent(request: Request) -> ReviewAdvisorService:
    """注入 Review Advisor 服务。

    Agent 不可用时返回 503，不暴露内部配置。

    参数:
        request: FastAPI Request 对象。

    返回:
        ReviewAdvisorService: 就绪的 Agent 服务。

    异常:
        HTTPException 503: Agent 当前不可用。
    """

    container = get_container(request)
    if container.evidence_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Review Advisor 当前不可用，基础证据仍可正常使用",
                "reason": container.evidence_agent_status.get("reason"),
            },
        )
    return container.evidence_agent


def get_case_agent(request: Request) -> CaseAgentService:
    """注入 Case Agent 服务。

    Case Agent 当前只在 PostgreSQL + Kimi 配置齐备时可用。
    """

    container = get_container(request)
    if container.case_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Case Agent 当前不可用，基础审核流程仍可正常使用",
                "reason": container.case_agent_status.get("reason"),
            },
        )
    return container.case_agent


def get_case_memory(request: Request) -> CaseMemoryService:
    """Inject the governed CaseMemoryService."""

    container = get_container(request)
    if container.case_memory is None:
        raise HTTPException(status_code=503, detail="CaseMemoryService 当前不可用")
    return container.case_memory


def _is_allowed_user(user: StoredUser | None) -> bool:
    """校验用户是否有权限进入审核业务。

    参数:
        user: 从仓储中查到的存储用户对象。

    返回:
        bool: 用户存在、状态为 active 且角色包含 auditor/reviewer/admin 之一。
    """

    allowed_roles = {"auditor", "reviewer", "admin"}
    return bool(
        user
        and user.status == "active"
        and allowed_roles.intersection(user.roles)
    )
