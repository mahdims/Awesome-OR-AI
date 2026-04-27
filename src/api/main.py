"""FastAPI application entrypoint.

Single-origin deploy: this app serves ``/api/*`` and mounts the React UI as
static files at ``/``. Caddy upstream is `api:80` (compose service name).

For M1b we register only the health router; auth + read-only routers land in
weeks 2-3 of the milestone.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from src.api.routers import auth, health, me
from src.api.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(title=settings.app_name, docs_url="/api/docs", openapi_url="/api/openapi.json")

    # SessionMiddleware is required by Authlib's OAuth flow (stores the state
    # token between /auth/google/login and /auth/google/callback).
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.jwt_secret,
        same_site="lax",
        https_only=settings.cookie_secure,
    )

    # Routers — health is public; auth handles its own auth; /api/me requires JWT.
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(me.router)

    # Static UI mount — serves src/ui/* at the root path so the React mockup
    # works without a build step. index.html sits at src/ui/index.html.
    ui_dir = PROJECT_ROOT / settings.static_ui_dir
    if ui_dir.exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


app = create_app()
