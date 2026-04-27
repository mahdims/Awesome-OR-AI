"""Shared FastAPI dependencies.

Re-exports the DB session and settings; ``current_user`` lands in M2 once
auth is wired. For M1b we ship a stub that returns ``None`` — every M1b
endpoint either is public (/health, /api/init) or simply doesn't enforce
auth yet (the auth router itself populates the cookie when it lands).
"""

from __future__ import annotations

from src.api.db_session import get_session
from src.api.settings import Settings, get_settings

__all__ = ["get_session", "get_settings", "Settings"]
