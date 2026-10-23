"""Application settings, loaded from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "ai-support-agent"
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'agent.db'}"))
    # "mock" runs fully offline (deterministic, used in tests/CI); "openai" or "anthropic" call real LLM APIs.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "30")))
    webhook_secret: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", "dev-webhook-secret"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "dev-api-key"))
    knowledge_dir: Path = field(
        default_factory=lambda: Path(os.getenv("KNOWLEDGE_DIR", str(BASE_DIR / "data" / "knowledge")))
    )
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "3")))
    agent_max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "4")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
