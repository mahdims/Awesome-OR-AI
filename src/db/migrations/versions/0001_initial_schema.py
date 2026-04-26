"""initial schema: paper_analyses, citations, fronts, papers, review_updates

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-23

Ports the SQLite schema in src/db/schema.sql to Postgres with type promotions:
- TEXT JSON fields -> JSONB
- BLOB embedding -> vector(768) (pgvector)
- INTEGER 0/1 flags -> BOOLEAN where applicable
- date('now') defaults -> CURRENT_DATE
- INTEGER PRIMARY KEY AUTOINCREMENT -> BIGSERIAL
- json_extract() functional index -> GIN on JSONB column

Also folds in two columns that live code added via ALTER TABLE after the
original schema.sql was written: paper_analyses.venue and paper_analyses.is_relevant.

Schema fixes versus schema.sql:
- FK front_lineage -> research_fronts(front_id) dropped (research_fronts has
  a composite PK; the SQLite FK was a lie the engine happened to tolerate).
- front_lineage.previous_front_id defaults to '' instead of NULL, since
  Postgres rejects NULL PK columns. No existing code writes this table today,
  so the sentinel is a forward-compatible choice rather than a data concern.
"""

from alembic import op


revision: str = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    op.execute(
        """
        CREATE TABLE paper_analyses (
            arxiv_id          TEXT PRIMARY KEY,
            category          TEXT NOT NULL,

            title             TEXT NOT NULL,
            authors           JSONB NOT NULL,
            abstract          TEXT,
            published_date    DATE NOT NULL,

            affiliations      TEXT,
            problem           JSONB NOT NULL,
            methodology       JSONB NOT NULL,
            experiments       JSONB NOT NULL,
            results           JSONB NOT NULL,
            artifacts         JSONB NOT NULL,
            reader_confidence JSONB,

            lineage           JSONB NOT NULL,
            tags              JSONB NOT NULL,
            extensions        JSONB NOT NULL,
            methods_confidence JSONB,

            relevance         JSONB NOT NULL,
            significance      JSONB NOT NULL,
            brief             TEXT NOT NULL,

            analysis_date     TIMESTAMPTZ NOT NULL DEFAULT now(),
            analysis_model    TEXT,
            pdf_hash          TEXT,

            venue             TEXT,

            embedding         vector(768),

            is_relevant       BOOLEAN NOT NULL DEFAULT true
        );
        """
    )
    op.execute("CREATE INDEX idx_category ON paper_analyses(category);")
    op.execute("CREATE INDEX idx_date ON paper_analyses(published_date);")
    op.execute(
        "CREATE INDEX idx_tags_methods ON paper_analyses "
        "USING GIN ((tags -> 'methods') jsonb_path_ops);"
    )

    op.execute(
        """
        CREATE TABLE citations (
            source_paper_id   TEXT NOT NULL,
            target_paper_id   TEXT NOT NULL,
            category          TEXT NOT NULL,
            citation_context  TEXT,
            discovered_date   DATE NOT NULL DEFAULT CURRENT_DATE,
            PRIMARY KEY (source_paper_id, target_paper_id, category)
        );
        """
    )
    op.execute("CREATE INDEX idx_citations_category ON citations(category);")
    op.execute("CREATE INDEX idx_citations_source ON citations(source_paper_id);")
    op.execute("CREATE INDEX idx_citations_target ON citations(target_paper_id);")

    op.execute(
        """
        CREATE TABLE citation_fetch_log (
            arxiv_id     TEXT NOT NULL,
            category     TEXT NOT NULL,
            fetch_date   DATE NOT NULL DEFAULT CURRENT_DATE,
            fetch_mode   TEXT NOT NULL,
            refs_count   INTEGER NOT NULL DEFAULT 0,
            cited_count  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (arxiv_id, category)
        );
        """
    )
    op.execute("CREATE INDEX idx_fetch_log_category ON citation_fetch_log(category);")

    op.execute(
        """
        CREATE TABLE cocitation_edges (
            paper1_id         TEXT NOT NULL,
            paper2_id         TEXT NOT NULL,
            category          TEXT NOT NULL,
            cocitation_count  INTEGER NOT NULL,
            strength          REAL,
            snapshot_date     DATE NOT NULL,
            PRIMARY KEY (paper1_id, paper2_id, category, snapshot_date),
            CHECK (paper1_id < paper2_id)
        );
        """
    )
    op.execute(
        "CREATE INDEX idx_cocitation_category "
        "ON cocitation_edges(category, snapshot_date);"
    )

    op.execute(
        """
        CREATE TABLE research_fronts (
            front_id           TEXT NOT NULL,
            category           TEXT NOT NULL,
            snapshot_date      DATE NOT NULL,

            core_papers        JSONB NOT NULL,
            size               INTEGER NOT NULL,
            internal_density   REAL,

            dominant_methods   JSONB,
            dominant_problems  JSONB,

            growth_rate        REAL,
            stability          REAL,
            status             TEXT,

            name               TEXT,
            summary            TEXT,
            future_directions  JSONB,

            PRIMARY KEY (front_id, snapshot_date)
        );
        """
    )
    op.execute(
        "CREATE INDEX idx_fronts_category "
        "ON research_fronts(category, snapshot_date);"
    )
    op.execute("CREATE INDEX idx_fronts_status ON research_fronts(status);")

    op.execute(
        """
        CREATE TABLE front_lineage (
            current_front_id   TEXT NOT NULL,
            previous_front_id  TEXT NOT NULL DEFAULT '',
            category           TEXT NOT NULL,
            snapshot_date      DATE NOT NULL,
            relationship       TEXT NOT NULL,
            overlap_score      REAL,
            PRIMARY KEY (current_front_id, previous_front_id, snapshot_date)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE bridge_papers (
            paper_id          TEXT NOT NULL,
            category          TEXT NOT NULL,
            snapshot_date     DATE NOT NULL,
            home_front_id     TEXT NOT NULL,
            connected_fronts  JSONB NOT NULL,
            bridge_score      REAL NOT NULL,
            PRIMARY KEY (paper_id, category, snapshot_date)
        );
        """
    )
    op.execute(
        "CREATE INDEX idx_bridge_category "
        "ON bridge_papers(category, snapshot_date);"
    )
    op.execute(
        "CREATE INDEX idx_bridge_score ON bridge_papers(bridge_score DESC);"
    )

    op.execute(
        """
        CREATE TABLE papers (
            arxiv_id     TEXT NOT NULL,
            category     TEXT NOT NULL,
            title        TEXT,
            authors      TEXT,
            date         TEXT,
            affiliation  TEXT,
            venue        TEXT,
            code_url     TEXT,
            fetched_at   DATE NOT NULL DEFAULT CURRENT_DATE,
            PRIMARY KEY (arxiv_id, category)
        );
        """
    )
    op.execute("CREATE INDEX idx_papers_category ON papers(category, date DESC);")

    op.execute(
        """
        CREATE TABLE review_updates (
            update_id      BIGSERIAL PRIMARY KEY,
            category       TEXT NOT NULL,
            update_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
            update_type    TEXT NOT NULL,
            papers_added   INTEGER NOT NULL DEFAULT 0,
            fronts_changed INTEGER NOT NULL DEFAULT 0,
            commit_hash    TEXT,
            summary        TEXT
        );
        """
    )
    op.execute(
        "CREATE INDEX idx_updates_category "
        "ON review_updates(category, update_date);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS review_updates;")
    op.execute("DROP TABLE IF EXISTS papers;")
    op.execute("DROP TABLE IF EXISTS bridge_papers;")
    op.execute("DROP TABLE IF EXISTS front_lineage;")
    op.execute("DROP TABLE IF EXISTS research_fronts;")
    op.execute("DROP TABLE IF EXISTS cocitation_edges;")
    op.execute("DROP TABLE IF EXISTS citation_fetch_log;")
    op.execute("DROP TABLE IF EXISTS citations;")
    op.execute("DROP TABLE IF EXISTS paper_analyses;")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
