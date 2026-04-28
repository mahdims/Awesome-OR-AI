"""Pydantic-settings reading the same .env as the existing layers.

Single source of truth for runtime config: DATABASE_URL, REDIS_URL, JWT_SECRET,
Google OAuth client, invite-list bootstrap, and a few app constants. Everything
is overridable via env vars; .env loading is done by python-dotenv at import time
of `src/db/database.py` (see _get_dsn) and again here for safety.
"""

from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Sentinel from earlier dev defaults — refusing to start with this prevents
# any deploy that forgets to set JWT_SECRET from accepting attacker-forged
# cookies signed with a known string.
_JWT_SECRET_PLACEHOLDER = "dev-only-do-not-use-in-prod"
_JWT_SECRET_MIN_LEN = 32

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
    # Required. Must be set to a non-placeholder, sufficiently-long value or
    # startup fails (a known fallback would let attackers forge session cookies).
    jwt_secret: str = Field(validation_alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"

    @field_validator("jwt_secret")
    @classmethod
    def _jwt_secret_must_be_real(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "JWT_SECRET is required. Generate one with: openssl rand -hex 64"
            )
        if v.strip() == _JWT_SECRET_PLACEHOLDER:
            raise ValueError(
                f"JWT_SECRET is set to the dev placeholder; refusing to start. "
                f"Generate a real secret: openssl rand -hex 64"
            )
        if len(v) < _JWT_SECRET_MIN_LEN:
            raise ValueError(
                f"JWT_SECRET is shorter than {_JWT_SECRET_MIN_LEN} chars; refusing to start."
            )
        return v
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
