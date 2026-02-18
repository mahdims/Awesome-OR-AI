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
        # Add venue column to paper_analyses if it doesn't exist yet
        try:
            self.conn.execute("ALTER TABLE paper_analyses ADD COLUMN venue TEXT")
        except Exception:
            pass  # column already exists
        self.conn.commit()
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

        Threshold: (max(M,P,I) >= 6  AND  sum(M,P,I) >= 18)  OR  must_read == True.

        The dual condition ensures papers score well in at least one dimension
        (max >= 6) AND have sufficient overall relevance (sum >= 18), preventing
        one-dimensional papers (e.g. 10/0/0) from passing. must_read overrides.
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
        m = rel.get('methodological', 0)
        p = rel.get('problem', 0)
        i = rel.get('inspirational', 0)
        max_score = max(m, p, i)
        sum_score = m + p + i
        passes_scores = (max_score >= 6 and sum_score >= 18)
        return 1 if (passes_scores or sig.get('must_read', False)) else 0

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

    # === Paper Metadata Enrichment ===

    def update_paper_metadata(self, arxiv_id: str, affiliations: str = None,
                               code_url: str = None, venue: str = None):
        """Update paper metadata fields (affiliations, venue, code_url). Only fills empty fields."""
        if affiliations:
            self.execute(
                "UPDATE paper_analyses SET affiliations = ? WHERE arxiv_id = ? AND (affiliations IS NULL OR affiliations = '')",
                (affiliations, arxiv_id)
            )
        if venue:
            self.execute(
                "UPDATE paper_analyses SET venue = ? WHERE arxiv_id = ? AND (venue IS NULL OR venue = '')",
                (venue, arxiv_id)
            )
        if code_url:
            # Update code_url inside artifacts JSON
            row = self.fetchone(
                "SELECT artifacts FROM paper_analyses WHERE arxiv_id = ?",
                (arxiv_id,)
            )
            if row and row['artifacts']:
                try:
                    artifacts = json.loads(row['artifacts'])
                    if not artifacts.get('code_url'):
                        artifacts['code_url'] = code_url
                        self.execute(
                            "UPDATE paper_analyses SET artifacts = ? WHERE arxiv_id = ?",
                            (json.dumps(artifacts), arxiv_id)
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
        self.commit()

    # === Fetched Papers (intake table) ===

    def upsert_paper(self, arxiv_id: str, category: str, title: str = "",
                     authors: str = "", date: str = "", affiliation: str = "",
                     venue: str = "", code_url: str = ""):
        """Insert or update a fetched paper in the intake table.

        Only updates non-empty fields; never overwrites with blank values.
        """
        self.execute(
            """INSERT INTO papers (arxiv_id, category, title, authors, date, affiliation, venue, code_url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(arxiv_id, category) DO UPDATE SET
                   title       = CASE WHEN excluded.title != ''       THEN excluded.title       ELSE title END,
                   authors     = CASE WHEN excluded.authors != ''     THEN excluded.authors     ELSE authors END,
                   date        = CASE WHEN excluded.date != ''        THEN excluded.date        ELSE date END,
                   affiliation = CASE WHEN excluded.affiliation != '' THEN excluded.affiliation ELSE affiliation END,
                   venue       = CASE WHEN excluded.venue != ''       THEN excluded.venue       ELSE venue END,
                   code_url    = CASE WHEN excluded.code_url != ''    THEN excluded.code_url    ELSE code_url END""",
            (arxiv_id, category, title or "", authors or "", date or "",
             affiliation or "", venue or "", code_url or "")
        )

    def get_all_papers(self, category: str = None) -> Dict[str, List[Dict]]:
        """Return all fetched papers grouped by category.

        Returns: {category_name: [{arxiv_id, title, authors, date, affiliation, venue, code_url}, ...]}
        """
        if category:
            rows = self.fetchall(
                "SELECT * FROM papers WHERE category = ? ORDER BY date DESC",
                (category,)
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM papers ORDER BY category, date DESC"
            )
        result: Dict[str, List[Dict]] = {}
        for row in rows:
            d = dict(row)
            cat = d['category']
            if cat not in result:
                result[cat] = []
            result[cat].append(d)
        return result

    def get_unanalyzed_paper_ids(self, category: str) -> List[str]:
        """Return arxiv_ids of papers in `papers` table not yet in `paper_analyses`."""
        rows = self.fetchall(
            """SELECT p.arxiv_id FROM papers p
               WHERE p.category = ?
               AND NOT EXISTS (
                   SELECT 1 FROM paper_analyses a WHERE a.arxiv_id = p.arxiv_id
               )""",
            (category,)
        )
        return [row['arxiv_id'] for row in rows]

    def migrate_json_to_papers(self, json_path) -> int:
        """One-time migration: populate `papers` table from docs/or-llm-daily.json.

        Safe to call repeatedly — uses ON CONFLICT DO UPDATE (upsert).
        Returns number of papers upserted.
        """
        import json as _json
        import re as _re
        path = Path(json_path)
        if not path.exists():
            return 0

        with open(path, encoding='utf-8') as f:
            data = _json.load(f)

        count = 0
        for cat, papers in data.items():
            for pid, content in papers.items():
                s = str(content)
                parts = s.split("|")
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

                # Clean date (strip markdown bold)
                date_clean = _re.sub(r'\*+', '', raw_date).strip()
                # Clean title
                title_clean = _re.sub(r'\*+', '', title).strip()
                m = _re.match(r'\[([^\]]+)\]', title_clean)
                if m:
                    title_clean = m.group(1)
                # Extract code URL
                url_m = _re.search(r'\[link\]\((.+?)\)', raw_code)
                code_url = url_m.group(1) if url_m else ""

                self.upsert_paper(pid, cat, title_clean, authors, date_clean,
                                  affiliation, venue, code_url)
                count += 1
        self.commit()
        return count

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
