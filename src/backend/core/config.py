"""MediGuard 后端全局配置。

所有应用配置通过 ``MEDIGUARD_`` 前缀的环境变量注入，
支持 .env / .env.local 文件本地覆盖。
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 后端项目根目录（core/ 的父目录）
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """从环境变量读取应用配置。

    所有配置项都使用 ``MEDIGUARD_`` 前缀。
    示例：``MEDIGUARD_API_PREFIX=/api``。
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDIGUARD_",
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    # ---- 应用基础 ----
    app_name: str = "MediGuard Agent"
    app_version: str = "1.0.0"
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    ingest_records_path: Path = (
        BACKEND_DIR / "fixtures" / "business_ingest_records.csv"
    )
    visitor_ingest_records_path: Path = (
        BACKEND_DIR.parent.parent / "model" / "mediredata.csv"
    )
    visitor_sample_specs_path: Path = (
        BACKEND_DIR / "fixtures" / "visitor_demo_samples.json"
    )
    showcase_mode: bool = False
    showcase_records_path: Path = (
        BACKEND_DIR / "fixtures" / "showcase_ingest_records.csv"
    )
    showcase_model_results_path: Path = (
        BACKEND_DIR / "fixtures" / "showcase_model_results.json"
    )
    showcase_sample_specs_path: Path = (
        BACKEND_DIR / "fixtures" / "showcase_sample_specs.json"
    )
    showcase_record_ids: list[str] = Field(
        default_factory=lambda: [
            "SIM_PERSON_012345",
            "SIM_PERSON_014632",
            "SIM_PERSON_011569",
            "SIM_PERSON_002204",
            "SIM_PERSON_011394",
        ]
    )
    public_demo_mode: bool = False
    public_demo_max_request_bytes: int = Field(
        default=2_000_000,
        ge=100_000,
        le=20_000_000,
    )
    public_demo_max_batch_records: int = Field(default=10, ge=1, le=100)
    public_demo_max_cases: int = Field(default=50, ge=5, le=1000)
    public_demo_ingest_max_requests: int = Field(default=20, ge=1, le=1000)
    public_demo_agent_max_requests: int = Field(default=30, ge=1, le=1000)
    public_demo_rate_window_seconds: int = Field(default=3600, ge=60, le=86400)
    policy_rag_corpus_path: Path = (
        BACKEND_DIR / "policy_corpus" / "chunks" / "policy_runtime_snippets.jsonl"
    )
    persistence_backend: Literal["memory", "postgres"] = "postgres"
    database_url: str = (
        "postgresql+psycopg://mediguard:mediguard@localhost:5432/mediguard"
    )
    auth_secret_key: str = Field(min_length=32)
    default_auditor_password: str = Field(min_length=8)
    auth_cookie_name: str = "mediguard_session"
    auth_session_hours: int = Field(default=8, ge=1, le=24)
    auth_cookie_secure: bool = False
    auth_login_max_attempts: int = Field(default=5, ge=3, le=20)
    auth_login_window_seconds: int = Field(default=300, ge=60, le=3600)
    fraud_model_enabled: bool = True
    fraud_model_assets_dir: Path = (
        BACKEND_DIR / "model_assets_local"
    )
    evidence_agent_enabled: bool = False
    llm_provider: Literal["deepseek", "fake"] = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    deepseek_api_key: SecretStr | None = None
    llm_thinking_enabled: bool = True
    llm_reasoning_effort: Literal["low", "medium", "high", "max"] = "low"
    expert_analysis_provider: Literal["deepseek"] = "deepseek"
    expert_analysis_base_url: str = "https://api.deepseek.com"
    expert_analysis_model: str = "deepseek-v4-flash"
    expert_analysis_deepseek_api_key: SecretStr | None = None
    expert_analysis_thinking_enabled: bool = True
    expert_analysis_reasoning_effort: Literal["low", "medium", "high", "max"] = "low"
    expert_analysis_timeout_seconds: int = Field(default=90, ge=10, le=120)
    review_advisor_provider: Literal["", "deepseek", "fake"] = ""
    review_advisor_base_url: str = ""
    review_advisor_model: str = ""
    review_advisor_deepseek_api_key: SecretStr | None = None
    review_advisor_thinking_enabled: bool | None = None
    review_advisor_reasoning_effort: Literal["", "low", "medium", "high", "max"] = ""
    review_advisor_enabled: bool | None = None
    review_advisor_timeout_seconds: int | None = Field(default=None, ge=10, le=120)
    review_advisor_max_model_calls: int | None = Field(default=None, ge=1, le=6)
    review_advisor_max_tool_calls: int | None = Field(default=None, ge=1, le=12)
    review_advisor_max_planning_rounds: int | None = Field(default=None, ge=1, le=6)
    review_advisor_max_concurrency: int | None = Field(default=None, ge=1, le=8)
    review_advisor_strict_local_validation: bool | None = None
    review_advisor_eval_variants_enabled: bool | None = None
    evidence_agent_timeout_seconds: int = Field(default=90, ge=10, le=120)
    evidence_agent_max_model_calls: int = Field(default=5, ge=1, le=6)
    evidence_agent_max_tool_calls: int = Field(default=12, ge=1, le=12)
    evidence_agent_max_planning_rounds: int = Field(default=3, ge=1, le=6)
    evidence_agent_max_concurrency: int = Field(default=2, ge=1, le=8)
    evidence_agent_strict_local_validation: bool = True
    evidence_agent_eval_variants_enabled: bool = False
    evidence_agent_prompt_version: str = "review-advisor-prompt-v11-deterministic-label"
    evidence_agent_tool_version: str = "review-advisor-tools-v5-ledger-v2"
    case_agent_enabled: bool = False
    case_agent_provider: Literal["deepseek", "kimi"] = "deepseek"
    case_agent_base_url: str = "https://api.deepseek.com"
    case_agent_model: str = "deepseek-v4-flash"
    case_agent_classifier_model: str = ""
    case_agent_generator_model: str = ""
    case_agent_deepseek_api_key: SecretStr | None = None
    case_agent_kimi_api_key: SecretStr | None = None
    case_agent_thinking_enabled: bool = False
    case_agent_reasoning_effort: Literal["low", "medium", "high", "max"] = "low"
    case_agent_timeout_seconds: int = Field(default=90, ge=10, le=120)
    case_agent_semantic_timeout_seconds: int = Field(default=12, ge=3, le=60)
    case_agent_planner_timeout_seconds: int = Field(default=15, ge=3, le=60)
    case_agent_answer_timeout_seconds: int = Field(default=30, ge=5, le=90)
    case_agent_semantic_max_tokens: int = Field(default=384, ge=128, le=2048)
    case_agent_planner_max_tokens: int = Field(default=768, ge=256, le=4096)
    case_agent_answer_max_tokens: int = Field(default=1536, ge=512, le=8192)
    case_agent_max_concurrency: int = Field(default=2, ge=1, le=8)
    case_agent_max_tool_calls: int = Field(default=5, ge=1, le=12)
    case_agent_fast_mode_enabled: bool = True
    policy_rag_runtime_reranker_enabled: bool = False
    case_agent_intent_biencoder_enabled: bool = True
    case_agent_intent_biencoder_backend: Literal["auto", "bge", "ngram"] = "auto"
    case_agent_intent_encoder_model: str = "BAAI/bge-m3"
    case_agent_intent_model_cache_dir: Path = (
        BACKEND_DIR.parent.parent / "runtime" / "hf_cache"
    )
    case_agent_intent_index_dir: Path = (
        BACKEND_DIR.parent.parent
        / "runtime"
        / "intent_example_index"
        / "caser_intent_examples_v0.2"
    )
    case_agent_intent_examples_path: Path = (
        BACKEND_DIR.parent.parent
        / "docs"
        / "02-planning"
        / "intent_biencoder_prep"
        / "intent_examples.seed.json"
    )
    case_agent_intent_taxonomy_path: Path = (
        BACKEND_DIR.parent.parent
        / "docs"
        / "02-planning"
        / "intent_biencoder_prep"
        / "intent_taxonomy.json"
    )
    case_agent_intent_high_confidence_top1: float = Field(default=0.78, ge=0.0, le=1.0)
    case_agent_intent_high_confidence_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    case_agent_intent_medium_confidence_top1: float = Field(default=0.70, ge=0.0, le=1.0)
    material_assets_dir: Path = (
        BACKEND_DIR.parent.parent / "runtime" / "generated_materials"
    )
    # CaseMemoryService / Mem0 runtime. PostgreSQL remains the truth ledger.
    case_memory_enabled: bool = True
    case_memory_worker_enabled: bool = True
    case_memory_worker_poll_seconds: float = Field(default=2.0, ge=0.25, le=60.0)
    case_memory_reconciliation_seconds: int = Field(default=86400, ge=60)
    case_memory_consolidation_seconds: int = Field(default=3600, ge=60)
    mem0_enabled: bool = True
    mem0_dir: Path = BACKEND_DIR.parent.parent / "runtime" / "mem0"
    mem0_llm_provider: Literal["deepseek"] = "deepseek"
    mem0_llm_model: str = "deepseek-v4-flash"
    mem0_llm_base_url: str = "https://api.deepseek.com"
    mem0_embedding_model: str = "BAAI/bge-m3"
    mem0_embedding_cache_dir: Path = BACKEND_DIR.parent.parent / "runtime" / "hf_cache"
    mem0_graph_backend: Literal["sql", "neo4j"] = "sql"
    mem0_projection_collection: str = "mediguard_case_memory"

    @model_validator(mode="after")
    def resolve_review_advisor_runtime_aliases(self):
        """Prefer Review Advisor env names while keeping Evidence Agent names compatible."""

        explicitly_set = self.model_fields_set
        if self.showcase_mode and "visitor_ingest_records_path" not in explicitly_set:
            self.visitor_ingest_records_path = self.showcase_records_path
        if self.showcase_mode and "visitor_sample_specs_path" not in explicitly_set:
            self.visitor_sample_specs_path = self.showcase_sample_specs_path
        if (
            self.review_advisor_enabled is not None
            and "evidence_agent_enabled" not in explicitly_set
        ):
            self.evidence_agent_enabled = self.review_advisor_enabled
        if (
            self.review_advisor_timeout_seconds is not None
            and "evidence_agent_timeout_seconds" not in explicitly_set
        ):
            self.evidence_agent_timeout_seconds = self.review_advisor_timeout_seconds
        if (
            self.review_advisor_max_model_calls is not None
            and "evidence_agent_max_model_calls" not in explicitly_set
        ):
            self.evidence_agent_max_model_calls = self.review_advisor_max_model_calls
        if (
            self.review_advisor_max_tool_calls is not None
            and "evidence_agent_max_tool_calls" not in explicitly_set
        ):
            self.evidence_agent_max_tool_calls = self.review_advisor_max_tool_calls
        if (
            self.review_advisor_max_planning_rounds is not None
            and "evidence_agent_max_planning_rounds" not in explicitly_set
        ):
            self.evidence_agent_max_planning_rounds = (
                self.review_advisor_max_planning_rounds
            )
        if (
            self.review_advisor_max_concurrency is not None
            and "evidence_agent_max_concurrency" not in explicitly_set
        ):
            self.evidence_agent_max_concurrency = self.review_advisor_max_concurrency
        if (
            self.review_advisor_strict_local_validation is not None
            and "evidence_agent_strict_local_validation" not in explicitly_set
        ):
            self.evidence_agent_strict_local_validation = (
                self.review_advisor_strict_local_validation
            )
        if (
            self.review_advisor_eval_variants_enabled is not None
            and "evidence_agent_eval_variants_enabled" not in explicitly_set
        ):
            self.evidence_agent_eval_variants_enabled = (
                self.review_advisor_eval_variants_enabled
            )
        return self


def get_settings() -> Settings:
    """创建配置对象。

    不使用全局缓存，测试可以为每个应用实例传入独立配置。
    """

    return Settings()
