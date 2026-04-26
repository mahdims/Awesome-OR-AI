"""rescore_cache table for layer0 score caching

Revision ID: 0002_rescore_cache
Revises: 0001_initial_schema
Create Date: 2026-04-26

Layer 0's relevance scorer caches per-paper scores so re-runs don't pay LLM
cost for already-seen abstracts. Previously created on-the-fly by
``_init_score_db()`` in fetch_and_score.py; migration now owns it.
"""

from alembic import op


revision: str = "0002_rescore_cache"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE rescore_cache (
            arxiv_id    TEXT NOT NULL,
            category    TEXT NOT NULL,
            title       TEXT,
            abstract    TEXT,
            score       INTEGER,
            score_date  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (arxiv_id, category)
        );
        """
    )
    op.execute("CREATE INDEX idx_rescore_category ON rescore_cache(category);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rescore_cache;")
