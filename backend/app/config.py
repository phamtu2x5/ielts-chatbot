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
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M")
    )
    ollama_num_predict: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "1200")))
    ollama_num_ctx: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "4096")))
    ollama_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    )
    ollama_think: bool = field(default_factory=lambda: _env_bool("OLLAMA_THINK", False))
    ollama_classifier_seed: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_CLASSIFIER_SEED", "42"))
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

    warmup_llm: bool = field(default_factory=lambda: _env_bool("WARMUP_LLM", True))
    warmup_embedding: bool = field(default_factory=lambda: _env_bool("WARMUP_EMBEDDING", True))

    def __post_init__(self) -> None:
        if self.ollama_num_predict <= 0 or self.ollama_num_ctx <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT and OLLAMA_NUM_CTX must be positive.")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive.")
        if (
            self.rag_top_k <= 0
            or self.rag_probe_top_k <= 0
            or self.rag_overview_top_k <= 0
            or self.rag_rrf_k <= 0
        ):
            raise ValueError("RAG top-k settings must be positive.")
        if self.rag_overview_source_chars <= 0:
            raise ValueError("RAG_OVERVIEW_SOURCE_CHARS must be positive.")
        if not 0 <= self.rag_min_score <= 1 or not 0 <= self.rag_probe_min_dense_score <= 1:
            raise ValueError("RAG score thresholds must be between 0 and 1.")


settings = AppSettings()
