"""Google OAuth 2.0 client wired via Authlib.

Two endpoints exposed by ``src/api/routers/auth.py``:
- ``GET  /auth/google/login``    — redirects user to Google's consent screen.
- ``GET  /auth/google/callback`` — Google redirects back here with a code.
                                   We exchange it, verify the email, check the
                                   invite list, upsert the user, and issue a
                                   JWT cookie.
"""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth

from src.api.settings import Settings


def make_oauth(settings: Settings) -> OAuth:
    """Build an Authlib OAuth registry holding the Google client.

    Returns an OAuth instance the routers can call (.google.authorize_redirect,
    .google.authorize_access_token). Returns None-equivalent (an OAuth with no
    `google` attribute) if credentials aren't configured — callers must handle.
    """
    oauth = OAuth()
    if settings.google_oauth_client_id and settings.google_oauth_client_secret:
        oauth.register(
            name="google",
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_id=settings.google_oauth_client_id,
            client_secret=settings.google_oauth_client_secret,
            client_kwargs={"scope": "openid email profile"},
        )
    return oauth
