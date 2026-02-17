import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent / "research_intelligence.db"

class Database:
    """Thin wrapper for SQLite with JSON serialization helpers."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        # Auto-initialize schema from schema.sql (all CREATE TABLE IF NOT EXISTS — safe for existing DBs)
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            self.conn.executescript(schema_path.read_text())
        # Ensure citation_fetch_log exists (migration for pre-existing databases)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS citation_fetch_log (
                arxiv_id    TEXT NOT NULL,
                category    TEXT NOT NULL,
                fetch_date  DATE NOT NULL DEFAULT (date('now')),
                fetch_mode  TEXT NOT NULL,
                refs_count  INTEGER DEFAULT 0,
                cited_count INTEGER DEFAULT 0,
                PRIMARY KEY (arxiv_id, category)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fetch_log_category "
            "ON citation_fetch_log(category)"
        )
        # Add LLM-enrichment columns to research_fronts if they don't exist yet
        for col in ("name TEXT", "future_directions TEXT"):
            try:
                self.conn.execute(f"ALTER TABLE research_fronts ADD COLUMN {col}")
            except Exception:
                pass  # column already exists
        # Add is_relevant column to paper_analyses if it doesn't exist yet
        try:
            self.conn.execute("ALTER TABLE paper_analyses ADD COLUMN is_relevant INTEGER DEFAULT 1")
        except Exception:
            pass  # column already exists
        return self

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
        self.close()

    def execute(self, query: str, params: tuple = ()):
        return self.conn.execute(query, params)

    def fetchone(self, query: str, params: tuple = ()):
        return self.execute(query, params).fetchone()

    def fetchall(self, query: str, params: tuple = ()):
        return self.execute(query, params).fetchall()

    def commit(self):
        self.conn.commit()

    # === Paper Analyses ===

    def has_analysis(self, arxiv_id: str) -> bool:
        """Check if paper already analyzed."""
        row = self.fetchone(
            "SELECT 1 FROM paper_analyses WHERE arxiv_id = ?",
            (arxiv_id,)
        )
        return row is not None

    def get_analysis(self, arxiv_id: str) -> Optional[Dict]:
        """Retrieve full analysis record."""
        row = self.fetchone(
            "SELECT * FROM paper_analyses WHERE arxiv_id = ?",
            (arxiv_id,)
        )
        return dict(row) if row else None

    @staticmethod
    def _compute_is_relevant(relevance, significance) -> int:
        """Return 1 if paper passes the positioning filter, 0 otherwise.

        Threshold: max(methodological, problem, inspirational) >= 6  OR  must_read == True.
        Accepts either a dict or a JSON string for both arguments.
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
        max_score = max(
            rel.get('methodological', 0),
            rel.get('problem', 0),
            rel.get('inspirational', 0),
        )
        return 1 if (max_score >= 6 or sig.get('must_read', False)) else 0

    def insert_analysis(self, analysis: Dict):
        """Insert new paper analysis."""
        # Convert nested dicts to JSON strings
        json_fields = ['authors', 'problem', 'methodology', 'experiments', 'results',
                       'artifacts', 'reader_confidence', 'lineage', 'tags', 'extensions',
                       'methods_confidence', 'relevance', 'significance']

        analysis_copy = analysis.copy()
        for field in json_fields:
            if field in analysis_copy and isinstance(analysis_copy[field], (dict, list)):
                analysis_copy[field] = json.dumps(analysis_copy[field])

        # Compute relevance filter flag from Positioning agent outputs
        analysis_copy['is_relevant'] = self._compute_is_relevant(
            analysis_copy.get('relevance'),
            analysis_copy.get('significance'),
        )

        columns = ', '.join(analysis_copy.keys())
        placeholders = ', '.join(['?'] * len(analysis_copy))

        self.execute(
            f"INSERT OR REPLACE INTO paper_analyses ({columns}) VALUES ({placeholders})",
            tuple(analysis_copy.values())
        )
        self.commit()

    def recompute_relevance_flags(self) -> int:
        """Recompute is_relevant for all existing papers. Returns number updated."""
        rows = self.fetchall("SELECT arxiv_id, relevance, significance FROM paper_analyses")
        updated = 0
        for row in rows:
            flag = self._compute_is_relevant(row['relevance'], row['significance'])
            self.execute(
                "UPDATE paper_analyses SET is_relevant = ? WHERE arxiv_id = ?",
                (flag, row['arxiv_id'])
            )
            updated += 1
        self.commit()
        return updated

    def get_papers_by_category(self, category: str,
                                limit: Optional[int] = None,
                                relevant_only: bool = True) -> List[Dict]:
        """Get analyzed papers for a category.

        Args:
            relevant_only: If True (default), return only papers where
                is_relevant = 1 (max(M,P,I) >= 6 or must_read).
                Set False to get all papers regardless of filter.
        """
        relevance_clause = " AND is_relevant = 1" if relevant_only else ""
        query = (
            f"SELECT * FROM paper_analyses WHERE category = ?{relevance_clause}"
            " ORDER BY published_date DESC"
        )
        if limit:
            query += f" LIMIT {limit}"
        return [dict(row) for row in self.fetchall(query, (category,))]

    def get_unanalyzed_papers(self, category: str,
                              all_paper_ids: List[str]) -> List[str]:
        """Find papers in JSON but not yet analyzed."""
        if not all_paper_ids:
            return []

        placeholders = ','.join(['?'] * len(all_paper_ids))
        analyzed = self.fetchall(
            f"SELECT arxiv_id FROM paper_analyses WHERE arxiv_id IN ({placeholders})",
            tuple(all_paper_ids)
        )
        analyzed_ids = {row['arxiv_id'] for row in analyzed}
        return [pid for pid in all_paper_ids if pid not in analyzed_ids]

    # === Citations & Fronts ===

    def insert_citations(self, citations: List[tuple], category: str):
        """Bulk insert citations for a category."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO citations (source_paper_id, target_paper_id, category) VALUES (?, ?, ?)",
            [(src, tgt, category) for src, tgt in citations]
        )
        self.commit()

    def clear_citations(self, category: str):
        """Delete all citation edges for a category.

        Used during force-refresh so stale edges from a previous fetch are
        fully removed before re-inserting fresh data.
        """
        self.execute(
            "DELETE FROM citations WHERE category = ?",
            (category,)
        )
        self.commit()

    def get_unfetched_papers(self, paper_ids: List[str], category: str,
                              fetch_mode: str) -> List[str]:
        """
        Return corpus paper IDs that have not yet been fetched from Semantic
        Scholar for this category/mode combination.

        A paper is considered already fetched if:
          - It has a row in citation_fetch_log for this category, AND
          - The logged fetch_mode covers the requested mode
            (a "both" log satisfies a "references" request; the reverse
             does not — "references"-only log won't satisfy "both").
        """
        if not paper_ids:
            return []

        # Fetch existing log entries for this category
        placeholders = ','.join(['?'] * len(paper_ids))
        rows = self.fetchall(
            f"""SELECT arxiv_id, fetch_mode FROM citation_fetch_log
                WHERE category = ? AND arxiv_id IN ({placeholders})""",
            (category, *paper_ids)
        )
        already_fetched = set()
        for row in rows:
            logged_mode = row['fetch_mode']
            # "both" satisfies any mode; "references" only satisfies "references"
            if logged_mode == 'both' or logged_mode == fetch_mode:
                already_fetched.add(row['arxiv_id'])

        return [pid for pid in paper_ids if pid not in already_fetched]

    def log_citation_fetch(self, arxiv_id: str, category: str,
                            fetch_mode: str, refs_count: int = 0,
                            cited_count: int = 0):
        """Record that Semantic Scholar was queried for this corpus paper."""
        self.execute(
            """INSERT OR REPLACE INTO citation_fetch_log
               (arxiv_id, category, fetch_date, fetch_mode, refs_count, cited_count)
               VALUES (?, ?, date('now'), ?, ?, ?)""",
            (arxiv_id, category, fetch_mode, refs_count, cited_count)
        )
        self.commit()

    def clear_fetch_log(self, category: str):
        """Remove all log entries for a category (forces full re-fetch next run)."""
        self.execute(
            "DELETE FROM citation_fetch_log WHERE category = ?",
            (category,)
        )
        self.commit()

    def clear_fronts_for_snapshot(self, category: str, snapshot_date: str):
        """Delete all fronts for a (category, snapshot_date) before re-inserting.

        Called at the start of each detection run so stale fronts from an
        earlier same-day run don't accumulate alongside the new ones.
        Also clears cocitation_edges for the same snapshot so removed edges
        don't persist (INSERT OR REPLACE only upserts, never deletes).
        """
        self.execute(
            "DELETE FROM research_fronts WHERE category = ? AND snapshot_date = ?",
            (category, snapshot_date)
        )
        self.execute(
            "DELETE FROM bridge_papers WHERE category = ? AND snapshot_date = ?",
            (category, snapshot_date)
        )
        self.execute(
            "DELETE FROM cocitation_edges WHERE category = ? AND snapshot_date = ?",
            (category, snapshot_date)
        )
        self.commit()

    def get_latest_fronts(self, category: str) -> List[Dict]:
        """Get most recent front snapshot for a category."""
        query = """
            SELECT * FROM research_fronts
            WHERE category = ? AND snapshot_date = (
                SELECT MAX(snapshot_date) FROM research_fronts WHERE category = ?
            )
            ORDER BY size DESC
        """
        return [dict(row) for row in self.fetchall(query, (category, category))]

    def insert_front(self, front: Dict):
        """Insert research front."""
        # Convert JSON fields
        front_copy = front.copy()
        for field in ['core_papers', 'dominant_methods', 'dominant_problems', 'future_directions']:
            if field in front_copy and isinstance(front_copy[field], list):
                front_copy[field] = json.dumps(front_copy[field])

        columns = ', '.join(front_copy.keys())
        placeholders = ', '.join(['?'] * len(front_copy))
        self.execute(
            f"INSERT OR REPLACE INTO research_fronts ({columns}) VALUES ({placeholders})",
            tuple(front_copy.values())
        )
        self.commit()

    def insert_bridge_papers(self, bridges: List[Dict]):
        """Bulk insert bridge papers."""
        for bridge in bridges:
            # Convert connected_fronts list to JSON
            bridge_copy = bridge.copy()
            if isinstance(bridge_copy.get('connected_fronts'), list):
                bridge_copy['connected_fronts'] = json.dumps(bridge_copy['connected_fronts'])

            columns = ', '.join(bridge_copy.keys())
            placeholders = ', '.join(['?'] * len(bridge_copy))
            self.execute(
                f"INSERT OR REPLACE INTO bridge_papers ({columns}) VALUES ({placeholders})",
                tuple(bridge_copy.values())
            )
        self.commit()

    # === Review Updates ===

    def insert_review_update(self, category: str, update_type: str,
                            papers_added: int = 0, fronts_changed: int = 0,
                            commit_hash: Optional[str] = None,
                            summary: Optional[str] = None):
        """Record a review update event."""
        self.execute(
            """INSERT INTO review_updates (category, update_type, papers_added, fronts_changed, commit_hash, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (category, update_type, papers_added, fronts_changed, commit_hash, summary)
        )
        self.commit()
