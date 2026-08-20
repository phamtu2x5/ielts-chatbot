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
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000",
        )
    )
    api_auth_required: bool = field(
        default_factory=lambda: _env_bool("API_AUTH_REQUIRED", False)
    )
    api_auth_token: str = field(
        default_factory=lambda: os.getenv("API_AUTH_TOKEN", "").strip()
    )
    debug_payloads: bool = field(default_factory=lambda: _env_bool("DEBUG_PAYLOADS", False))

    ollama_api_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate"
        )
    )
    ollama_chat_api_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_CHAT_API_URL", "http://127.0.0.1:11434/api/chat"
        )
    )
    ollama_chat_fallback: bool = field(
        default_factory=lambda: _env_bool("OLLAMA_CHAT_FALLBACK", True)
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL", "hf.co/Zkare/Chatbot_Ielts_Assistant_v2:Q4_K_M"
        )
    )
    ollama_num_predict: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "2800"))
    )
    ollama_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    )
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
    history_message_chars: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_MESSAGE_CHARS", "600"))
    )

    session_data_dir: Path = field(
        default_factory=lambda: _env_path("SESSION_DATA_DIR", "data/sessions")
    )
    session_grace_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_GRACE_TTL_SECONDS", "300"))
    )
    session_hard_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_HARD_TTL_SECONDS", "1800"))
    )
    chat_rate_limit: int = field(
        default_factory=lambda: int(os.getenv("CHAT_RATE_LIMIT", "30"))
    )
    chat_rate_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60"))
    )
    chat_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CHAT_MAX_CONCURRENCY", "2"))
    )
    warmup_llm: bool = field(default_factory=lambda: _env_bool("WARMUP_LLM", True))

    def __post_init__(self) -> None:
        if self.api_auth_required and len(self.api_auth_token) < 32:
            raise ValueError(
                "API_AUTH_TOKEN must contain at least 32 characters when auth is required."
            )
        if self.api_auth_required and "*" in self.cors_allow_origins:
            raise ValueError("CORS_ALLOW_ORIGINS cannot contain '*' when API auth is required.")
        if self.ollama_num_predict <= 0 or self.ollama_num_ctx <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT and OLLAMA_NUM_CTX must be positive.")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive.")
        if self.history_message_chars <= 0:
            raise ValueError("HISTORY_MESSAGE_CHARS must be positive.")
        if (
            self.session_grace_ttl_seconds <= 0
            or self.session_hard_ttl_seconds <= self.session_grace_ttl_seconds
        ):
            raise ValueError(
                "Session TTLs must be positive and hard TTL must exceed grace TTL."
            )
        if any(
            value <= 0
            for value in (
                self.chat_rate_limit,
                self.chat_rate_window_seconds,
                self.chat_max_concurrency,
            )
        ):
            raise ValueError("Chat request limits must be positive.")


settings = AppSettings()
