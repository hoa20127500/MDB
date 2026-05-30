"""
Application configuration loaded from environment variables via pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the recommendation engine.

    Values are read from environment variables (case-insensitive).
    A `.env` file in the working directory is also supported.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── MongoDB ──────────────────────────────────────────────────────────────
    MONGODB_URI: str = Field(
        ...,
        description="MongoDB Atlas connection URI (e.g. mongodb+srv://...)",
    )

    # ── PostgreSQL sources ───────────────────────────────────────────────────
    ODOO_DSN: str = Field(
        ...,
        description="SQLAlchemy async DSN for the Odoo ERP PostgreSQL database "
        "(e.g. postgresql+asyncpg://user:pass@host/db)",
    )
    VTM_DSN: str = Field(
        ...,
        description="SQLAlchemy async DSN for the VTM PostgreSQL database",
    )

    # ── Embedding model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name used for all embeddings",
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    SCHEDULER_INTERVAL_HOURS: int = Field(
        default=24,
        ge=1,
        description="How often (in hours) the data pipeline runs automatically",
    )

    # ── Recommendation defaults ──────────────────────────────────────────────
    DEFAULT_K: int = Field(
        default=10,
        ge=1,
        description="Default number of recommendations to return when K is not specified",
    )
    MIN_SCORE: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Global minimum similarity score threshold; None means no threshold",
    )

    # ── Pipeline ─────────────────────────────────────────────────────────────
    BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        description="Number of product records processed per MongoDB bulk-write batch",
    )

    EXTRACT_LIMIT: int | None = Field(
        default=None,
        ge=1,
        description="Max products to extract per source per run. None = no limit (all products).",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Using ``lru_cache`` ensures the environment is read only once per process,
    which is important for performance and for predictable test behaviour
    (tests can clear the cache with ``get_settings.cache_clear()``).
    """
    return Settings()  # type: ignore[call-arg]
