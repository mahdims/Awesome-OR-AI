"""Auth endpoints: Google OAuth login + callback + logout.

Flow:
- /auth/google/login  → redirect to Google consent screen.
- /auth/google/callback → token exchange, invite-list check, upsert user,
  issue JWT cookie, redirect to / .
- /auth/logout         → clear cookie.
"""

from __future__ import annotations

import logging

from authlib.integrations.starlette_client import OAuthError
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.auth.google import make_oauth
from src.api.auth.invites import is_email_allowed, mark_invite_accepted
from src.api.auth.jwt import clear_session_cookie, issue_token, set_session_cookie
from src.api.deps import Settings, get_session, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger(__name__)


def _oauth_client(settings: Settings):
    oauth = make_oauth(settings)
    if not hasattr(oauth, "google"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth not configured (GOOGLE_OAUTH_CLIENT_ID/SECRET missing).",
        )
    return oauth


@router.get("/google/login")
async def google_login(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    oauth = _oauth_client(settings)
    return await oauth.google.authorize_redirect(request, settings.google_oauth_redirect_uri)


@router.get("/google/callback")
async def google_callback(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
):
    oauth = _oauth_client(settings)
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        log.warning("OAuth callback failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth error")

    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").lower().strip()
    name = userinfo.get("name") or ""
    avatar_url = userinfo.get("picture") or ""

    if not email or not userinfo.get("email_verified", False):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not verified")

    if not is_email_allowed(email, session, settings):
        log.info("Rejected non-invited login: %s", email)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email not invited")

    # Upsert user.
    row = session.execute(
        text(
            """
            INSERT INTO users (email, name, avatar_url, last_visit_at)
            VALUES (:email, :name, :avatar, now())
            ON CONFLICT (email) DO UPDATE
            SET name = EXCLUDED.name,
                avatar_url = EXCLUDED.avatar_url,
                last_visit_at = now()
            RETURNING id, email, name, avatar_url, role
            """
        ),
        {"email": email, "name": name, "avatar": avatar_url},
    ).mappings().first()
    assert row is not None
    mark_invite_accepted(email, session)

    jwt_token = issue_token(user_id=row["id"], email=row["email"], settings=settings)
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    set_session_cookie(response, jwt_token, settings)
    return response


@router.post("/logout")
def logout(settings: Settings = Depends(get_settings)):
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    clear_session_cookie(response, settings)
    return response
