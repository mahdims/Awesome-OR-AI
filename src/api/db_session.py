"""Sync SQLAlchemy session for FastAPI request handlers.

We keep the existing ``src/db/database.py`` Database class (raw psycopg) for
the pipeline scripts that already use it; the API layer uses a SQLAlchemy
session because routers will benefit from ORM-style query building once we add
real models in M3+. For M1b the queries are still mostly hand-written SQL.
"""

from __future__ import annotations

from typing import Iterator

from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.settings import get_settings
from src.db.database import _get_dsn  # reuses the +psycopg-strip DSN logic

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _build_engine() -> Engine:
    settings = get_settings()
    dsn = settings.database_url
    # The Database wrapper strips +psycopg for raw psycopg; SQLAlchemy *needs*
    # the +psycopg suffix to pick the right driver, so use the original.
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(dsn, pool_pre_ping=True, future=True)


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine()
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)

        @event.listens_for(_engine, "connect")
        def _on_connect(dbapi_conn, _record):
            # pgvector adapter has to be registered per raw connection.
            register_vector(dbapi_conn)
    return _engine


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yield a session, commit on success, rollback on error."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    with _SessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
