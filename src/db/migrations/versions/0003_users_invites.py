"""users + invites tables for Google OAuth + invite-list auth (M1b).

Revision ID: 0003_users_invites
Revises: 0002_rescore_cache
Create Date: 2026-04-27

Two simple tables:
- ``users``: row created on first successful Google OAuth callback. Email is
  unique. ``role`` defaults to 'member'; admin role is granted only via SQL
  for now (UI invite-management lands in M5).
- ``invites``: email allowlist. A row here lets that email log in. Bootstrap
  admins (env var ``BOOTSTRAP_ADMIN_EMAILS``) bypass this table; everyone else
  needs a row.
"""

from alembic import op


revision: str = "0003_users_invites"
down_revision = "0002_rescore_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE users (
            id             BIGSERIAL PRIMARY KEY,
            email          TEXT NOT NULL UNIQUE,
            name           TEXT,
            avatar_url     TEXT,
            role           TEXT NOT NULL DEFAULT 'member',
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_visit_at  TIMESTAMPTZ
        );
        """
    )
    op.execute("CREATE INDEX idx_users_email_lower ON users (lower(email));")

    op.execute(
        """
        CREATE TABLE invites (
            email                TEXT PRIMARY KEY,
            invited_by_user_id   BIGINT REFERENCES users(id) ON DELETE SET NULL,
            accepted_at          TIMESTAMPTZ,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS invites;")
    op.execute("DROP TABLE IF EXISTS users;")
