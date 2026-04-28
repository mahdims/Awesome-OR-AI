"""Health endpoints.

Two levels:
- ``/health``      — process is up. Static. Caddy already serves a copy of this
                     in front of the app for the Better Stack ping.
- ``/health/deep`` — process can talk to Postgres + Redis. Used by deploy
                     scripts and the M2 NotebookLM circuit-breaker check.
"""

from __future__ import annotations

import logging

import redis as _redis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import Settings, get_session, get_settings

router = APIRouter(tags=["health"])
log = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/deep")
def health_deep(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    db_ok = False
    redis_ok = False

    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        # Roll back the failed transaction so the get_session() teardown's
        # session.commit() is a no-op. Without this, /health/deep crashes
        # with InFailedSqlTransaction (500) precisely when the DB is down —
        # the opposite of what a health endpoint should do.
        try:
            session.rollback()
        except Exception:
            pass
        log.warning("db ping failed: %s", exc)

    try:
        client = _redis.from_url(settings.redis_url, socket_timeout=2)
        redis_ok = bool(client.ping())
    except Exception as exc:  # noqa: BLE001
        log.warning("redis ping failed: %s", exc)

    overall = "ok" if (db_ok and redis_ok) else "degraded"
    return {"status": overall, "db": db_ok, "redis": redis_ok}
