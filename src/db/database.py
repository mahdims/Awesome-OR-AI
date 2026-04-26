"""Postgres database wrapper.

Replaces the prior SQLite wrapper. The public method surface is preserved
so Layer 1 / Layer 2 / Layer 3 / scripts do not need invasive rewrites,
with two deliberate behavior changes that callers must adopt:

1.  JSONB round-trips as ``dict`` / ``list`` — not as JSON ``str``.
    Callers that used ``json.loads(row["artifacts"])`` must drop the
    ``json.loads`` call. See the SQL dialect sweep (M1a, step 2).
2.  ``is_relevant`` round-trips as ``bool``, not as ``int`` 0/1.
    Callers doing ``== 1`` must switch to truthiness checks.

Schema is owned by Alembic migrations in ``src/db/migrations/``. This class
does NOT create tables; run ``alembic upgrade head`` once against a fresh
Postgres before using.

Connection string: ``DATABASE_URL`` env var. Expected form
``postgresql+psycopg://user:pass@host:port/db`` (SQLAlchemy-style). We strip
the ``+psycopg`` suffix when handing it to psycopg3 directly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_dsn() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Copy .env.example to .env and fill it in."
        )
    # Accept both SQLAlchemy-style (postgresql+psycopg://...) and native
    # (postgresql://...) URLs; psycopg3 wants the native form.
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


# JSONB columns that the class serializes automatically on insert/update.
_JSON_FIELDS_ANALYSIS = {
    "authors", "problem", "methodology", "experiments", "results",
    "artifacts", "reader_confidence", "lineage", "tags", "extensions",
    "methods_confidence", "relevance", "significance",
}
_JSON_FIELDS_FRONT = {
    "core_papers", "dominant_methods", "dominant_problems", "future_directions",
}
_JSON_FIELDS_BRIDGE = {"connected_fronts"}


def _wrap_json(value: Any) -> Any:
    """Wrap ``value`` for a JSONB column. Accepts dict/list or an already-JSON string."""
    if isinstance(value, (dict, list)):
        return Jsonb(value)
    if isinstance(value, str):
        # Legacy callers may still pass JSON strings. Parse and re-wrap so the
        # column receives JSONB rather than a TEXT literal.
        try:
            return Jsonb(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return Jsonb(value)
    return value


class Database:
    """Thin Postgres wrapper, sync psycopg3.

    Usage::

        with Database() as db:
            rows = db.fetchall("SELECT * FROM papers WHERE category = %s", (cat,))
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or _get_dsn()
        self.conn: Optional[psycopg.Connection] = None

    # --- connection lifecycle ---

    def connect(self) -> "Database":
        self.conn = psycopg.connect(self.dsn, row_factory=dict_row)
        # Required for pgvector list<->vector adaptation on embedding column.
        register_vector(self.conn)
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn is not None:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        self.close()

    def commit(self) -> None:
        assert self.conn is not None, "connect() first"
        self.conn.commit()

    # --- low-level query helpers (paramstyle: pyformat "%s") ---

    def execute(self, query: str, params: Sequence[Any] = ()) -> psycopg.Cursor:
        assert self.conn is not None, "connect() first"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur

    def fetchone(self, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        cur = self.execute(query, params)
        try:
            return cur.fetchone()
        finally:
            cur.close()

    def fetchall(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        cur = self.execute(query, params)
        try:
            return cur.fetchall()
        finally:
            cur.close()

    # === Paper Analyses ===

    def has_analysis(self, arxiv_id: str) -> bool:
        row = self.fetchone(
            "SELECT 1 FROM paper_analyses WHERE arxiv_id = %s", (arxiv_id,)
        )
        return row is not None

    def get_analysis(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        return self.fetchone(
            "SELECT * FROM paper_analyses WHERE arxiv_id = %s", (arxiv_id,)
        )

    def get_embedding(self, arxiv_id: str) -> Optional[List[float]]:
        """Return embedding as a list of floats, or None if not yet generated.

        Behavior change from SQLite: the column is now ``vector(768)`` (pgvector),
        not ``BLOB``. Callers that previously did ``np.frombuffer(blob)`` should
        switch to using the list directly (or ``np.asarray(row)``).
        """
        row = self.fetchone(
            "SELECT embedding FROM paper_analyses WHERE arxiv_id = %s", (arxiv_id,)
        )
        return row["embedding"] if row else None

    def store_embedding(self, arxiv_id: str, embedding: Iterable[float]) -> None:
        """Persist an embedding into the paper_analyses row.

        Accepts any iterable of floats (list, numpy array, etc.). The
        pgvector pg adapter is registered lazily by the pgvector package;
        we pass a plain list which the adapter coerces.
        """
        self.execute(
            "UPDATE paper_analyses SET embedding = %s WHERE arxiv_id = %s",
            (list(embedding), arxiv_id),
        )
        self.commit()

    def store_abstract(self, arxiv_id: str, abstract: str) -> None:
        self.execute(
            "UPDATE paper_analyses SET abstract = %s "
            "WHERE arxiv_id = %s AND (abstract IS NULL OR abstract = '')",
            (abstract, arxiv_id),
        )
        self.commit()

    @staticmethod
    def _compute_is_relevant(relevance: Any, significance: Any) -> bool:
        """Return True if the paper passes the positioning filter.

        Threshold: ``(max(M,P,I) >= 6  AND  sum(M,P,I) >= 18)  OR  must_read``.
        Accepts either a dict (new Postgres path) or a JSON string (legacy
        SQLite callers during the migration window).
        """
        if isinstance(relevance, str):
            try:
                relevance = json.loads(relevance)
            except (json.JSONDecodeError, TypeError):
                relevance = {}
        if isinstance(significance, str):
            try:
                significance = json.loads(significance)
            except (json.JSONDecodeError, TypeError):
                significance = {}
        rel = relevance or {}
        sig = significance or {}
        m = rel.get("methodological", 0)
        p = rel.get("problem", 0)
        i = rel.get("inspirational", 0)
        passes_scores = max(m, p, i) >= 6 and (m + p + i) >= 18
        return bool(passes_scores or sig.get("must_read", False))

    def insert_analysis(self, analysis: Dict[str, Any]) -> None:
        """Upsert a paper analysis row.

        JSONB columns are auto-wrapped. ``is_relevant`` is computed from
        ``relevance`` + ``significance`` and overrides any value in ``analysis``.
        """
        row = {k: v for k, v in analysis.items()}
        row["is_relevant"] = self._compute_is_relevant(
            row.get("relevance"), row.get("significance")
        )
        for field in list(row.keys()):
            if field in _JSON_FIELDS_ANALYSIS:
                row[field] = _wrap_json(row[field])

        columns = list(row.keys())
        col_sql = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in columns if c != "arxiv_id"
        )
        self.execute(
            f"INSERT INTO paper_analyses ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT (arxiv_id) DO UPDATE SET {update_sql}",
            tuple(row.values()),
        )
        self.commit()

    def recompute_relevance_flags(self) -> int:
        """Recompute is_relevant for every paper. Returns rows updated."""
        rows = self.fetchall(
            "SELECT arxiv_id, relevance, significance FROM paper_analyses"
        )
        updated = 0
        for r in rows:
            flag = self._compute_is_relevant(r["relevance"], r["significance"])
            self.execute(
                "UPDATE paper_analyses SET is_relevant = %s WHERE arxiv_id = %s",
                (flag, r["arxiv_id"]),
            )
            updated += 1
        self.commit()
        return updated

    def get_papers_by_category(
        self,
        category: str,
        limit: Optional[int] = None,
        relevant_only: bool = True,
    ) -> List[Dict[str, Any]]:
        relevance_clause = " AND is_relevant = TRUE" if relevant_only else ""
        query = (
            f"SELECT * FROM paper_analyses WHERE category = %s{relevance_clause} "
            "ORDER BY published_date DESC"
        )
        params: List[Any] = [category]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        return self.fetchall(query, tuple(params))

    def get_unanalyzed_papers(
        self, category: str, all_paper_ids: List[str]
    ) -> List[str]:
        if not all_paper_ids:
            return []
        analyzed = self.fetchall(
            "SELECT arxiv_id FROM paper_analyses WHERE arxiv_id = ANY(%s)",
            (list(all_paper_ids),),
        )
        analyzed_ids = {r["arxiv_id"] for r in analyzed}
        return [pid for pid in all_paper_ids if pid not in analyzed_ids]

    # === Citations & Fronts ===

    def insert_citations(self, citations: List[tuple], category: str) -> None:
        assert self.conn is not None, "connect() first"
        cur = self.conn.cursor()
        try:
            cur.executemany(
                "INSERT INTO citations (source_paper_id, target_paper_id, category) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                [(src, tgt, category) for src, tgt in citations],
            )
        finally:
            cur.close()
        self.commit()

    def clear_citations(self, category: str) -> None:
        self.execute("DELETE FROM citations WHERE category = %s", (category,))
        self.commit()

    def get_unfetched_papers(
        self, paper_ids: List[str], category: str, fetch_mode: str
    ) -> List[str]:
        if not paper_ids:
            return []
        rows = self.fetchall(
            "SELECT arxiv_id, fetch_mode FROM citation_fetch_log "
            "WHERE category = %s AND arxiv_id = ANY(%s)",
            (category, list(paper_ids)),
        )
        already_fetched = {
            r["arxiv_id"]
            for r in rows
            if r["fetch_mode"] == "both" or r["fetch_mode"] == fetch_mode
        }
        return [pid for pid in paper_ids if pid not in already_fetched]

    def log_citation_fetch(
        self,
        arxiv_id: str,
        category: str,
        fetch_mode: str,
        refs_count: int = 0,
        cited_count: int = 0,
    ) -> None:
        self.execute(
            "INSERT INTO citation_fetch_log "
            "(arxiv_id, category, fetch_date, fetch_mode, refs_count, cited_count) "
            "VALUES (%s, %s, CURRENT_DATE, %s, %s, %s) "
            "ON CONFLICT (arxiv_id, category) DO UPDATE SET "
            "fetch_date = EXCLUDED.fetch_date, "
            "fetch_mode = EXCLUDED.fetch_mode, "
            "refs_count = EXCLUDED.refs_count, "
            "cited_count = EXCLUDED.cited_count",
            (arxiv_id, category, fetch_mode, refs_count, cited_count),
        )
        self.commit()

    def clear_fetch_log(self, category: str) -> None:
        self.execute(
            "DELETE FROM citation_fetch_log WHERE category = %s", (category,)
        )
        self.commit()

    def clear_fronts_for_snapshot(self, category: str, snapshot_date: str) -> None:
        self.execute(
            "DELETE FROM research_fronts WHERE category = %s AND snapshot_date = %s",
            (category, snapshot_date),
        )
        self.execute(
            "DELETE FROM bridge_papers WHERE category = %s AND snapshot_date = %s",
            (category, snapshot_date),
        )
        self.execute(
            "DELETE FROM cocitation_edges WHERE category = %s AND snapshot_date = %s",
            (category, snapshot_date),
        )
        self.commit()

    def get_latest_fronts(self, category: str) -> List[Dict[str, Any]]:
        return self.fetchall(
            """
            SELECT * FROM research_fronts
            WHERE category = %s AND snapshot_date = (
                SELECT MAX(snapshot_date) FROM research_fronts WHERE category = %s
            )
            ORDER BY size DESC
            """,
            (category, category),
        )

    def insert_front(self, front: Dict[str, Any]) -> None:
        row = {k: v for k, v in front.items()}
        for field in list(row.keys()):
            if field in _JSON_FIELDS_FRONT:
                row[field] = _wrap_json(row[field])

        columns = list(row.keys())
        col_sql = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in columns
            if c not in ("front_id", "snapshot_date")
        )
        self.execute(
            f"INSERT INTO research_fronts ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT (front_id, snapshot_date) DO UPDATE SET {update_sql}",
            tuple(row.values()),
        )
        self.commit()

    def insert_bridge_papers(self, bridges: List[Dict[str, Any]]) -> None:
        for bridge in bridges:
            row = {k: v for k, v in bridge.items()}
            for field in list(row.keys()):
                if field in _JSON_FIELDS_BRIDGE:
                    row[field] = _wrap_json(row[field])

            columns = list(row.keys())
            col_sql = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            update_sql = ", ".join(
                f"{c} = EXCLUDED.{c}"
                for c in columns
                if c not in ("paper_id", "category", "snapshot_date")
            )
            self.execute(
                f"INSERT INTO bridge_papers ({col_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT (paper_id, category, snapshot_date) DO UPDATE SET {update_sql}",
                tuple(row.values()),
            )
        self.commit()

    # === Paper Metadata Enrichment ===

    def update_paper_metadata(
        self,
        arxiv_id: str,
        affiliations: Optional[str] = None,
        code_url: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> None:
        if affiliations:
            self.execute(
                "UPDATE paper_analyses SET affiliations = %s "
                "WHERE arxiv_id = %s AND (affiliations IS NULL OR affiliations = '')",
                (affiliations, arxiv_id),
            )
        if venue:
            self.execute(
                "UPDATE paper_analyses SET venue = %s "
                "WHERE arxiv_id = %s AND (venue IS NULL OR venue = '')",
                (venue, arxiv_id),
            )
        if code_url:
            row = self.fetchone(
                "SELECT artifacts FROM paper_analyses WHERE arxiv_id = %s",
                (arxiv_id,),
            )
            # artifacts is JSONB; psycopg3 returns a dict directly.
            if row and row["artifacts"]:
                artifacts = row["artifacts"]
                if isinstance(artifacts, str):
                    try:
                        artifacts = json.loads(artifacts)
                    except (json.JSONDecodeError, TypeError):
                        artifacts = None
                if isinstance(artifacts, dict) and not artifacts.get("code_url"):
                    artifacts["code_url"] = code_url
                    self.execute(
                        "UPDATE paper_analyses SET artifacts = %s WHERE arxiv_id = %s",
                        (Jsonb(artifacts), arxiv_id),
                    )
        self.commit()

    # === Fetched Papers (intake table) ===

    def upsert_paper(
        self,
        arxiv_id: str,
        category: str,
        title: str = "",
        authors: str = "",
        date: str = "",
        affiliation: str = "",
        venue: str = "",
        code_url: str = "",
    ) -> None:
        self.execute(
            """
            INSERT INTO papers (arxiv_id, category, title, authors, date,
                                affiliation, venue, code_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id, category) DO UPDATE SET
                title       = CASE WHEN EXCLUDED.title       <> '' THEN EXCLUDED.title       ELSE papers.title       END,
                authors     = CASE WHEN EXCLUDED.authors     <> '' THEN EXCLUDED.authors     ELSE papers.authors     END,
                date        = CASE WHEN EXCLUDED.date        <> '' THEN EXCLUDED.date        ELSE papers.date        END,
                affiliation = CASE WHEN EXCLUDED.affiliation <> '' THEN EXCLUDED.affiliation ELSE papers.affiliation END,
                venue       = CASE WHEN EXCLUDED.venue       <> '' THEN EXCLUDED.venue       ELSE papers.venue       END,
                code_url    = CASE WHEN EXCLUDED.code_url    <> '' THEN EXCLUDED.code_url    ELSE papers.code_url    END
            """,
            (
                arxiv_id,
                category,
                title or "",
                authors or "",
                date or "",
                affiliation or "",
                venue or "",
                code_url or "",
            ),
        )

    def get_all_papers(self, category: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        if category:
            rows = self.fetchall(
                "SELECT * FROM papers WHERE category = %s ORDER BY date DESC",
                (category,),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM papers ORDER BY category, date DESC"
            )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(row["category"], []).append(row)
        return result

    def get_unanalyzed_paper_ids(self, category: str) -> List[str]:
        rows = self.fetchall(
            """
            SELECT p.arxiv_id FROM papers p
            WHERE p.category = %s
              AND NOT EXISTS (
                SELECT 1 FROM paper_analyses a WHERE a.arxiv_id = p.arxiv_id
              )
            """,
            (category,),
        )
        return [r["arxiv_id"] for r in rows]

    def migrate_json_to_papers(self, json_path) -> int:
        """One-time backfill: populate `papers` from docs/or-llm-daily.json.

        Safe to call repeatedly — uses ON CONFLICT DO UPDATE (upsert).
        Returns number of papers upserted.
        """
        import re as _re
        from pathlib import Path

        path = Path(json_path)
        if not path.exists():
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for cat, papers in data.items():
            for pid, content in papers.items():
                parts = str(content).split("|")
                try:
                    if len(parts) >= 9:
                        raw_date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
                        affiliation = parts[4].strip()
                        venue = parts[5].strip()
                        raw_code = parts[7].strip()
                    elif len(parts) >= 8:
                        raw_date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
                        affiliation = ""
                        venue = parts[4].strip()
                        raw_code = parts[6].strip()
                    else:
                        raw_date, title, authors = parts[1].strip(), parts[2].strip(), parts[3].strip()
                        affiliation, venue = "", ""
                        raw_code = parts[5].strip()
                except IndexError:
                    continue

                date_clean = _re.sub(r"\*+", "", raw_date).strip()
                title_clean = _re.sub(r"\*+", "", title).strip()
                m = _re.match(r"\[([^\]]+)\]", title_clean)
                if m:
                    title_clean = m.group(1)
                url_m = _re.search(r"\[link\]\((.+?)\)", raw_code)
                code_url = url_m.group(1) if url_m else ""

                self.upsert_paper(
                    pid, cat, title_clean, authors, date_clean,
                    affiliation, venue, code_url,
                )
                count += 1
        self.commit()
        return count

    # === Review Updates ===

    def insert_review_update(
        self,
        category: str,
        update_type: str,
        papers_added: int = 0,
        fronts_changed: int = 0,
        commit_hash: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO review_updates
                (category, update_type, papers_added, fronts_changed, commit_hash, summary)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (category, update_type, papers_added, fronts_changed, commit_hash, summary),
        )
        self.commit()
