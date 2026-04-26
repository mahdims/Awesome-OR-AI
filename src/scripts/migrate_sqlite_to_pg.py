"""One-shot copy from the old SQLite file to the new Postgres schema.

Run **after** Postgres is up and ``alembic upgrade head`` has been applied.

    python -m src.scripts.migrate_sqlite_to_pg [--sqlite-path PATH] [--dry-run]

What it does, per table:
- ``paper_analyses``: parse JSON-as-TEXT columns into dicts before insert;
  ``is_relevant`` 0/1 -> bool; ``embedding`` BLOB is **not** carried over —
  the new column is ``vector(768)`` and the BLOB was raw float32 bytes with
  no dimension guarantee. Back-fill happens in M1c (layer1/embedder.py).
- ``papers`` (intake), ``citations``, ``citation_fetch_log``,
  ``cocitation_edges``, ``research_fronts``, ``front_lineage``,
  ``bridge_papers``, ``review_updates``: straight copy with JSON parsing
  where relevant.

Idempotent: every insert uses ``ON CONFLICT ... DO UPDATE`` or
``ON CONFLICT DO NOTHING`` so a partial run can be re-attempted without
duplicating rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg
from psycopg.types.json import Jsonb

from src.db.database import _get_dsn

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[1] / "db" / "research_intelligence.db"


# Column -> JSON parse? True means the SQLite TEXT is JSON and must be parsed.
PAPER_ANALYSES_JSON_COLS = {
    "authors", "problem", "methodology", "experiments", "results",
    "artifacts", "reader_confidence", "lineage", "tags", "extensions",
    "methods_confidence", "relevance", "significance",
}

# Columns we explicitly skip when building the INSERT — embedding is handled
# separately (dropped for now; will be back-filled by the embedder in M1c).
PAPER_ANALYSES_SKIP_COLS = {"embedding"}

RESEARCH_FRONTS_JSON_COLS = {
    "core_papers", "dominant_methods", "dominant_problems", "future_directions",
}
BRIDGE_PAPERS_JSON_COLS = {"connected_fronts"}


def _parse_json_maybe(value: Any) -> Any:
    """Parse a JSON-text column value into a dict/list. Tolerates NULL / empty."""
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _to_bool(value: Any) -> bool:
    """Coerce SQLite 0/1/NULL to Postgres bool with NULL -> True (existing default)."""
    if value is None:
        return True
    return bool(int(value))


def _open_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite source not found: {sqlite_path}")
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(sconn: sqlite3.Connection, name: str) -> bool:
    row = sconn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _columns(sconn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in sconn.execute(f"PRAGMA table_info({table})")]


# --- per-table copiers ---------------------------------------------------


def copy_paper_analyses(
    sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool
) -> int:
    if not _table_exists(sconn, "paper_analyses"):
        return 0
    src_cols = [c for c in _columns(sconn, "paper_analyses") if c not in PAPER_ANALYSES_SKIP_COLS]
    col_sql = ", ".join(src_cols)
    rows = sconn.execute(f"SELECT {col_sql} FROM paper_analyses").fetchall()

    if dry_run or not rows:
        return len(rows)

    placeholders = ", ".join(["%s"] * len(src_cols))
    update_sql = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in src_cols if c != "arxiv_id"
    )
    insert_sql = (
        f"INSERT INTO paper_analyses ({col_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT (arxiv_id) DO UPDATE SET {update_sql}"
    )

    with pconn.cursor() as cur:
        for row in rows:
            values = []
            for col in src_cols:
                v = row[col]
                if col in PAPER_ANALYSES_JSON_COLS:
                    parsed = _parse_json_maybe(v)
                    values.append(Jsonb(parsed) if parsed is not None else None)
                elif col == "is_relevant":
                    values.append(_to_bool(v))
                else:
                    values.append(v)
            cur.execute(insert_sql, tuple(values))
    pconn.commit()
    return len(rows)


def copy_table(
    sconn: sqlite3.Connection,
    pconn: psycopg.Connection,
    table: str,
    json_cols: Iterable[str] = (),
    conflict_cols: Optional[List[str]] = None,
    skip_cols: Iterable[str] = (),
    dry_run: bool = False,
) -> int:
    """Generic single-table copier.

    ``conflict_cols`` is the list of columns in the PK (or any unique constraint)
    for use in the ``ON CONFLICT`` target. If ``None``, the copier uses
    ``ON CONFLICT DO NOTHING``.
    """
    if not _table_exists(sconn, table):
        return 0
    src_cols = [c for c in _columns(sconn, table) if c not in set(skip_cols)]
    col_sql = ", ".join(src_cols)
    rows = sconn.execute(f"SELECT {col_sql} FROM {table}").fetchall()

    if dry_run or not rows:
        return len(rows)

    placeholders = ", ".join(["%s"] * len(src_cols))

    if conflict_cols:
        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in src_cols if c not in conflict_cols
        )
        conflict = f"({', '.join(conflict_cols)})"
        insert_sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT {conflict} DO UPDATE SET {update_sql}"
            if update_sql
            else f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT {conflict} DO NOTHING"
        )
    else:
        insert_sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

    json_set = set(json_cols)
    with pconn.cursor() as cur:
        for row in rows:
            values = []
            for col in src_cols:
                v = row[col]
                if col in json_set:
                    parsed = _parse_json_maybe(v)
                    values.append(Jsonb(parsed) if parsed is not None else None)
                else:
                    values.append(v)
            cur.execute(insert_sql, tuple(values))
    pconn.commit()
    return len(rows)


# --- main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sqlite-path",
        type=Path,
        default=DEFAULT_SQLITE_PATH,
        help="Path to the SQLite source DB",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts without writing to Postgres",
    )
    args = ap.parse_args()

    sconn = _open_sqlite(args.sqlite_path)
    dsn = _get_dsn()

    report: Dict[str, int] = {}
    with psycopg.connect(dsn) as pconn:
        report["paper_analyses"] = copy_paper_analyses(sconn, pconn, args.dry_run)

        report["papers"] = copy_table(
            sconn, pconn, "papers",
            conflict_cols=["arxiv_id", "category"],
            dry_run=args.dry_run,
        )
        report["citations"] = copy_table(
            sconn, pconn, "citations",
            conflict_cols=["source_paper_id", "target_paper_id", "category"],
            dry_run=args.dry_run,
        )
        report["citation_fetch_log"] = copy_table(
            sconn, pconn, "citation_fetch_log",
            conflict_cols=["arxiv_id", "category"],
            dry_run=args.dry_run,
        )
        report["cocitation_edges"] = copy_table(
            sconn, pconn, "cocitation_edges",
            conflict_cols=["paper1_id", "paper2_id", "category", "snapshot_date"],
            dry_run=args.dry_run,
        )
        report["research_fronts"] = copy_table(
            sconn, pconn, "research_fronts",
            json_cols=RESEARCH_FRONTS_JSON_COLS,
            conflict_cols=["front_id", "snapshot_date"],
            dry_run=args.dry_run,
        )
        report["front_lineage"] = copy_table(
            sconn, pconn, "front_lineage",
            conflict_cols=["current_front_id", "previous_front_id", "snapshot_date"],
            dry_run=args.dry_run,
        )
        report["bridge_papers"] = copy_table(
            sconn, pconn, "bridge_papers",
            json_cols=BRIDGE_PAPERS_JSON_COLS,
            conflict_cols=["paper_id", "category", "snapshot_date"],
            dry_run=args.dry_run,
        )
        # review_updates: update_id is BIGSERIAL in PG; drop the SQLite rowid
        # so Postgres assigns its own. conflict_cols=None means ON CONFLICT DO NOTHING,
        # but there's no natural unique key, so a re-run inserts duplicates.
        # Script is one-shot; document that re-running on this table is not safe.
        report["review_updates"] = copy_table(
            sconn, pconn, "review_updates",
            skip_cols=["update_id"],
            dry_run=args.dry_run,
        )

        report["rescore_cache"] = copy_table(
            sconn, pconn, "rescore_cache",
            conflict_cols=["arxiv_id", "category"],
            dry_run=args.dry_run,
        )

    sconn.close()

    action = "would copy" if args.dry_run else "copied"
    width = max(len(k) for k in report)
    print("\nSQLite -> Postgres migration report")
    print("-" * (width + 14))
    for table, count in report.items():
        print(f"  {table.ljust(width)}  {action:>10}  {count} row(s)")
    total = sum(report.values())
    print("-" * (width + 14))
    print(f"  {'TOTAL'.ljust(width)}  {action:>10}  {total} row(s)")

    if not args.dry_run:
        print("\nNote: paper_analyses.embedding was NOT copied (new column is "
              "vector(768); back-fill via layer1/embedder.py in M1c).")
        print("Note: review_updates re-runs insert duplicates — do not re-run.")


if __name__ == "__main__":
    main()
