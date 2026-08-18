import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else BACKEND_DIR / value


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


@dataclass(frozen=True)
class AppSettings:
    upload_dir: Path = field(default_factory=lambda: _env_path("UPLOAD_DIR", "uploads"))
    rag_data_dir: Path = field(default_factory=lambda: _env_path("RAG_DATA_DIR", "data/rag"))
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("CORS_ALLOW_ORIGINS", "*")
    )

    ollama_api_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
    )
    ollama_chat_api_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CHAT_API_URL", "http://127.0.0.1:11434/api/chat")
    )
    ollama_chat_fallback: bool = field(
        default_factory=lambda: _env_bool("OLLAMA_CHAT_FALLBACK", True)
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M")
    )
    ollama_num_predict: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "2800")))
    ollama_num_ctx: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "4096")))
    ollama_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    )
    ollama_think: bool = field(default_factory=lambda: _env_bool("OLLAMA_THINK", False))
    ollama_classifier_seed: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_CLASSIFIER_SEED", "42"))
    )
    ollama_direct_temperature: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_DIRECT_TEMPERATURE", "0.3"))
    )
    route_history_message_chars: int = field(
        default_factory=lambda: int(os.getenv("ROUTE_HISTORY_MESSAGE_CHARS", "600"))
    )
    target_catalog_chars: int = field(
        default_factory=lambda: int(os.getenv("TARGET_CATALOG_CHARS", "3000"))
    )
    target_catalog_document_chars: int = field(
        default_factory=lambda: int(os.getenv("TARGET_CATALOG_DOCUMENT_CHARS", "360"))
    )
    target_descriptor_chars: int = field(
        default_factory=lambda: int(os.getenv("TARGET_DESCRIPTOR_CHARS", "220"))
    )
    target_resolver_max_candidates: int = field(
        default_factory=lambda: int(os.getenv("TARGET_RESOLVER_MAX_CANDIDATES", "5"))
    )
    target_clarification_max_candidates: int = field(
        default_factory=lambda: int(os.getenv("TARGET_CLARIFICATION_MAX_CANDIDATES", "3"))
    )
    document_scope_min_match_score: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_SCOPE_MIN_MATCH_SCORE", "60"))
    )
    document_scope_match_margin: float = field(
        default_factory=lambda: float(os.getenv("DOCUMENT_SCOPE_MATCH_MARGIN", "10"))
    )

    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    rag_min_score: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_SCORE", "0.45")))
    rag_probe_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_PROBE_TOP_K", "3")))
    rag_probe_min_dense_score: float = field(
        default_factory=lambda: float(os.getenv("RAG_PROBE_MIN_DENSE_SCORE", "0.35"))
    )
    rag_rrf_k: int = field(default_factory=lambda: int(os.getenv("RAG_RRF_K", "60")))
    rag_overview_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_OVERVIEW_TOP_K", "8")))
    rag_overview_source_chars: int = field(
        default_factory=lambda: int(os.getenv("RAG_OVERVIEW_SOURCE_CHARS", "900"))
    )
    rag_solve_evidence_per_question: int = field(
        default_factory=lambda: int(os.getenv("RAG_SOLVE_EVIDENCE_PER_QUESTION", "2"))
    )
    rag_solve_max_evidence: int = field(
        default_factory=lambda: int(os.getenv("RAG_SOLVE_MAX_EVIDENCE", "12"))
    )
    rag_session_grace_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("RAG_SESSION_GRACE_TTL_SECONDS", "600"))
    )
    rag_session_hard_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("RAG_SESSION_HARD_TTL_SECONDS", "7200"))
    )
    rag_session_max_documents: int = field(
        default_factory=lambda: int(os.getenv("RAG_SESSION_MAX_DOCUMENTS", "12"))
    )
    rag_session_max_chunks: int = field(
        default_factory=lambda: int(os.getenv("RAG_SESSION_MAX_CHUNKS", "6000"))
    )
    chat_rate_limit: int = field(
        default_factory=lambda: int(os.getenv("CHAT_RATE_LIMIT", "12"))
    )
    chat_rate_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60"))
    )
    upload_rate_limit: int = field(
        default_factory=lambda: int(os.getenv("UPLOAD_RATE_LIMIT", "10"))
    )
    upload_rate_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("UPLOAD_RATE_WINDOW_SECONDS", "600"))
    )
    chat_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CHAT_MAX_CONCURRENCY", "2"))
    )
    upload_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("UPLOAD_MAX_CONCURRENCY", "1"))
    )

    warmup_llm: bool = field(default_factory=lambda: _env_bool("WARMUP_LLM", True))
    warmup_embedding: bool = field(default_factory=lambda: _env_bool("WARMUP_EMBEDDING", True))

    def __post_init__(self) -> None:
        if self.ollama_num_predict <= 0 or self.ollama_num_ctx <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT and OLLAMA_NUM_CTX must be positive.")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive.")
        if (
            self.route_history_message_chars <= 0
            or self.target_catalog_chars <= 0
            or self.target_catalog_document_chars <= 0
            or self.target_descriptor_chars <= 0
            or self.target_resolver_max_candidates <= 0
            or self.target_clarification_max_candidates <= 0
            or self.document_scope_min_match_score <= 0
            or self.document_scope_match_margin <= 0
        ):
            raise ValueError("Route and target context limits must be positive.")
        if (
            self.rag_top_k <= 0
            or self.rag_probe_top_k <= 0
            or self.rag_overview_top_k <= 0
            or self.rag_rrf_k <= 0
            or self.rag_solve_evidence_per_question <= 0
            or self.rag_solve_max_evidence <= 0
        ):
            raise ValueError("RAG top-k settings must be positive.")
        if self.rag_overview_source_chars <= 0:
            raise ValueError("RAG_OVERVIEW_SOURCE_CHARS must be positive.")
        if (
            self.rag_session_grace_ttl_seconds <= 0
            or self.rag_session_hard_ttl_seconds <= self.rag_session_grace_ttl_seconds
        ):
            raise ValueError(
                "RAG session TTLs must be positive and hard TTL must exceed grace TTL."
            )
        if any(
            value <= 0
            for value in (
                self.rag_session_max_documents,
                self.rag_session_max_chunks,
                self.chat_rate_limit,
                self.chat_rate_window_seconds,
                self.upload_rate_limit,
                self.upload_rate_window_seconds,
                self.chat_max_concurrency,
                self.upload_max_concurrency,
            )
        ):
            raise ValueError("Session quotas and request limits must be positive.")
        if not 0 <= self.rag_min_score <= 1 or not 0 <= self.rag_probe_min_dense_score <= 1:
            raise ValueError("RAG score thresholds must be between 0 and 1.")


settings = AppSettings()
