"""Pydantic-settings reading the same .env as the existing layers.

Single source of truth for runtime config: DATABASE_URL, REDIS_URL, JWT_SECRET,
Google OAuth client, invite-list bootstrap, and a few app constants. Everything
is overridable via env vars; .env loading is done by python-dotenv at import time
of `src/db/database.py` (see _get_dsn) and again here for safety.
"""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # --- Environment ---
    env: str = Field(default="development", validation_alias="ENV")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")

    # --- Database / Redis ---
    database_url: str = Field(validation_alias="DATABASE_URL")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", validation_alias="REDIS_URL")

    # --- Auth ---
    jwt_secret: str = Field(default="dev-only-do-not-use-in-prod", validation_alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 30
    cookie_name: str = "ri_session"
    cookie_secure: bool = Field(default=True, validation_alias="COOKIE_SECURE")

    google_oauth_client_id: str = Field(default="", validation_alias="GOOGLE_OAUTH_CLIENT_ID")
    google_oauth_client_secret: str = Field(default="", validation_alias="GOOGLE_OAUTH_CLIENT_SECRET")
    google_oauth_redirect_uri: str = Field(
        default="http://localhost:8080/auth/google/callback",
        validation_alias="GOOGLE_OAUTH_REDIRECT_URI",
    )

    # Comma-separated emails that bypass the invites-table check.
    bootstrap_admin_emails_raw: str = Field(default="", validation_alias="BOOTSTRAP_ADMIN_EMAILS")

    @property
    def bootstrap_admin_emails(self) -> List[str]:
        return [e.strip().lower() for e in self.bootstrap_admin_emails_raw.split(",") if e.strip()]

    # --- App constants ---
    app_name: str = "researchmate"
    static_ui_dir: str = "src/ui"  # served at "/" in single-origin deploy


_settings: Settings | None = None


def get_settings() -> Settings:
    """Module-level singleton. Imported by routers as a FastAPI dependency."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
