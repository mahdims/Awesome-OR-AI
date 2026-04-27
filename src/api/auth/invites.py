"""Invite-list check.

An email is allowed to sign in if either:
- It appears in ``BOOTSTRAP_ADMIN_EMAILS`` (env var, comma-sep). This is how
  the very first user gets in before any invites row exists.
- It has a row in the ``invites`` table.

On a successful first login we mark the invite as accepted (not strictly
required for auth, but useful for the future invite-management UI in M5).
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.settings import Settings


def is_email_allowed(email: str, session: Session, settings: Settings) -> bool:
    email_lower = email.lower()
    if email_lower in settings.bootstrap_admin_emails:
        return True
    row = session.execute(
        text("SELECT 1 FROM invites WHERE lower(email) = :e"),
        {"e": email_lower},
    ).first()
    return row is not None


def mark_invite_accepted(email: str, session: Session) -> None:
    session.execute(
        text(
            "UPDATE invites SET accepted_at = now() "
            "WHERE lower(email) = :e AND accepted_at IS NULL"
        ),
        {"e": email.lower()},
    )
