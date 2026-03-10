from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/exam_reconciler",
        )
    )
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))

    embeddings_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDINGS_PROVIDER", "openai")
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768"))
    )

    hybrid_alpha: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_ALPHA", "0.7"))
    )
    hybrid_beta: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_BETA", "0.3"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
    )
    retriever_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "10"))
    )

    # LLM Judge (optional — improves low-confidence matches)
    llm_judge_enabled: bool = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_ENABLED", "false").lower()
        == "true"
    )
    llm_judge_model: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_MODEL", "gpt-5-mini-2025-08-07")
    )
    llm_judge_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_BASE_URL", "")
    )
    llm_judge_threshold: float = field(
        default_factory=lambda: float(os.getenv("LLM_JUDGE_THRESHOLD", "0.95"))
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    @property
    def use_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def sql_dir(self) -> Path:
        return Path(__file__).parent.parent / "sql"


def get_settings() -> Settings:
    return Settings()
