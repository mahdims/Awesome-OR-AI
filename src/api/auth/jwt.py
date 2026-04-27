"""JWT cookie helpers + current_user dependency.

Cookie: HttpOnly, SameSite=Lax, Secure (prod), name from settings. Payload
keeps it minimal: {sub: user_id, email}. Expiry is a settings constant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.deps import Settings, get_session, get_settings


def issue_token(user_id: int, email: str, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=settings.jwt_expire_days)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def set_session_cookie(response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.jwt_expire_days * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def clear_session_cookie(response, settings: Settings) -> None:
    response.delete_cookie(key=settings.cookie_name, path="/")


def _decode(token: str, settings: Settings) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def current_user_optional(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> Optional[dict]:
    """Returns the logged-in user dict or None. Use for routes that work both ways."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    try:
        claims = _decode(token, settings)
    except JWTError:
        return None
    user_id = int(claims["sub"])
    row = session.execute(
        text("SELECT id, email, name, avatar_url, role FROM users WHERE id = :id"),
        {"id": user_id},
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


def current_user(
    user: Optional[dict] = Depends(current_user_optional),
) -> dict:
    """Returns the logged-in user dict or raises 401."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user
