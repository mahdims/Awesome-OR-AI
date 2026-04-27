"""Per-user state tables (M1b).

Revision ID: 0004_user_state
Revises: 0003_users_invites
Create Date: 2026-04-27

- ``user_paper_state``: reading state + notes per (user, paper).
- ``user_follows``: subdomains the user follows. ``subdomain_id`` is plain
  ``TEXT`` (no FK) — subdomains live in ``research_config/subdomains.yaml``
  until M3 introduces a real ``subdomains`` table.
- ``user_pins``: pinned subdomains for the home page (with ordering).
- ``user_prefs``: density / theme / notification preferences.
"""

from alembic import op


revision: str = "0004_user_state"
down_revision = "0003_users_invites"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE user_paper_state (
            user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            paper_id    TEXT   NOT NULL,
            status      TEXT   NOT NULL CHECK (status IN ('unread','reading','read','discarded')),
            notes       TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, paper_id)
        );
        """
    )
    op.execute("CREATE INDEX idx_user_paper_state_user ON user_paper_state(user_id);")

    op.execute(
        """
        CREATE TABLE user_follows (
            user_id        BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            subdomain_id   TEXT   NOT NULL,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, subdomain_id)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE user_pins (
            user_id        BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            subdomain_id   TEXT   NOT NULL,
            position       INTEGER NOT NULL DEFAULT 0,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, subdomain_id)
        );
        """
    )
    op.execute("CREATE INDEX idx_user_pins_position ON user_pins(user_id, position);")

    op.execute(
        """
        CREATE TABLE user_prefs (
            user_id              BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            density              TEXT NOT NULL DEFAULT 'balanced',
            theme                TEXT NOT NULL DEFAULT 'default',
            notification_prefs   JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_prefs;")
    op.execute("DROP TABLE IF EXISTS user_pins;")
    op.execute("DROP TABLE IF EXISTS user_follows;")
    op.execute("DROP TABLE IF EXISTS user_paper_state;")
