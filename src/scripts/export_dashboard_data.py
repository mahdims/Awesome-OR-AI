#!/usr/bin/env python3
"""
Export dashboard data from SQLite to docs/dashboard_data.json.

Reads research_intelligence.db and produces a single JSON file consumed
by docs/ui/dashboard.html. Run after Layer 1/2/3 pipeline steps.

Usage:
    python src/scripts/export_dashboard_data.py
    python src/scripts/export_dashboard_data.py --output docs/dashboard_data.json
    python src/scripts/export_dashboard_data.py --db src/db/research_intelligence.db
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

# === Path bootstrap ===
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# === Imports from existing modules (read-only, no modification) ===
from db.database import Database
from config import DB_PATH, PROJECT_ROOT
from layer3.data_collector import (
    _enrich_paper, _enrich_front, _enrich_bridge, _json, _max_relevance,
)
from layer3.smart_prioritization import compute_priority_score

# Optional imports — degrade gracefully if Layer 3 hasn't fully run
try:
    from layer3.framework_genealogy import (
        extract_framework_lineages,
        build_genealogy_tree,
        get_active_frameworks,
        get_framework_summary,
    )
    _HAS_GENEALOGY = True
except ImportError:
    _HAS_GENEALOGY = False

try:
    from layer3.affiliation_analysis import analyze_front_affiliations
    _HAS_AFFILIATIONS = True
except ImportError:
    _HAS_AFFILIATIONS = False

CATEGORIES = [
    "LLMs for Algorithm Design",
    "Generative AI for OR",
    "OR for Generative AI",
]

OUTPUT_PATH = PROJECT_ROOT / "docs" / "dashboard_data.json"


# ── Lookup maps ────────────────────────────────────────────────────────────────

def build_paper_maps(db: Database) -> tuple:
    """
    Build fast lookup maps from latest front snapshots.

    Returns:
        front_map:        paper_id -> front_id
        front_status_map: paper_id -> front.status
        bridge_map:       paper_id -> bridge_score
    """
    front_map = {}
    front_status_map = {}
    bridge_map = {}

    for category in CATEGORIES:
        front_rows = db.get_latest_fronts(category)
        for front in front_rows:
            fid = front["front_id"]
            status = front.get("status", "unknown")
            for pid in _json(front.get("core_papers"), []):
                front_map[pid] = fid
                front_status_map[pid] = status

        # Bridge scores from the same snapshot date as the fronts
        if not front_rows:
            continue
        snapshot_date = front_rows[0].get("snapshot_date", "")
        if not snapshot_date:
            continue
        bridge_rows = db.fetchall(
            "SELECT paper_id, bridge_score FROM bridge_papers WHERE category = ? AND snapshot_date = ?",
            (category, snapshot_date),
        )
        for b in bridge_rows:
            bridge_map[b["paper_id"]] = b["bridge_score"]

    return front_map, front_status_map, bridge_map


# ── Papers ─────────────────────────────────────────────────────────────────────

def export_papers(db: Database, front_map: dict, front_status_map: dict, bridge_map: dict) -> list:
    """Export all relevant paper analyses, enriched with priority/front/bridge info."""
    rows = db.fetchall(
        "SELECT * FROM paper_analyses WHERE is_relevant = 1 ORDER BY published_date DESC"
    )
    papers = []
    for row in rows:
        row = dict(row)
        p = _enrich_paper(row)
        pid = p["arxiv_id"]

        # Attach front / bridge context
        p["front_id"] = front_map.get(pid)
        p["front_status"] = front_status_map.get(pid)
        p["bridge_score"] = bridge_map.get(pid, 0.0)
        p["is_bridge"] = pid in bridge_map

        # Pre-compute priority score (wider window so recency doesn't zero-out most papers)
        p["priority_score"] = round(
            compute_priority_score(
                paper=p,
                front_status=p["front_status"],
                bridge_score=p["bridge_score"],
                days_window=30,
            ),
            2,
        )

        # Include experiments + results for the detail drawer
        # (_enrich_paper omits these; we add them from the raw row)
        p["experiments"] = _json(row.get("experiments"), {})
        p["results"] = _json(row.get("results"), {})
        p["analysis_date"] = (row.get("analysis_date") or "")[:10]

        papers.append(p)

    return papers


# ── Fronts ─────────────────────────────────────────────────────────────────────

def export_fronts(db: Database, papers_by_id: dict) -> list:
    """Export latest research fronts with affiliation + method overlap data."""
    all_fronts = []

    for category in CATEGORIES:
        front_rows = db.get_latest_fronts(category)
        if not front_rows:
            continue

        enriched = [_enrich_front(dict(f)) for f in front_rows]

        # Affiliation analysis per front
        aff_by_front = {}
        if _HAS_AFFILIATIONS:
            try:
                aff_by_front = analyze_front_affiliations(enriched, db)
            except Exception:
                aff_by_front = {}

        # Method overlap between fronts in this category
        front_methods = {}
        for front in enriched:
            methods = set()
            for pid in front.get("core_papers", []):
                p = papers_by_id.get(pid)
                if p:
                    methods.update(p.get("tags", {}).get("methods", []))
            front_methods[front["front_id"]] = methods

        for front in enriched:
            fid = front["front_id"]
            aff_info = aff_by_front.get(fid, {})
            front["top_affiliations"] = aff_info.get("top_affiliations", [])

            overlap = {}
            for other in enriched:
                if other["front_id"] == fid:
                    continue
                shared = sorted(front_methods.get(fid, set()) & front_methods.get(other["front_id"], set()))
                if shared:
                    overlap[other["front_id"]] = shared
            front["method_overlap_with"] = overlap

            all_fronts.append(front)

    return all_fronts


# ── Bridges ────────────────────────────────────────────────────────────────────

def export_bridges(db: Database, papers_by_id: dict) -> list:
    """Export bridge papers with computed verdict. Paper object looked up in browser via papersById."""
    all_bridges = []

    for category in CATEGORIES:
        front_rows = db.get_latest_fronts(category)
        if not front_rows:
            continue

        snapshot_date = front_rows[0].get("snapshot_date", "")
        if not snapshot_date:
            continue

        # Build method sets per front for verdict computation
        enriched_fronts = [_enrich_front(dict(f)) for f in front_rows]
        front_methods = {}
        for front in enriched_fronts:
            methods = set()
            for pid in front.get("core_papers", []):
                p = papers_by_id.get(pid)
                if p:
                    methods.update(p.get("tags", {}).get("methods", []))
            front_methods[front["front_id"]] = methods

        bridge_rows = db.fetchall(
            "SELECT * FROM bridge_papers WHERE category = ? AND snapshot_date = ? ORDER BY bridge_score DESC",
            (category, snapshot_date),
        )

        for row in bridge_rows:
            b = _enrich_bridge(dict(row))
            pid = b["paper_id"]

            # Verdict: method overlap between paper, home front, and connected fronts
            paper_methods = set()
            p = papers_by_id.get(pid)
            if p:
                paper_methods = set(p.get("tags", {}).get("methods", []))

            home_methods = front_methods.get(b["home_front_id"], set())
            overlap_home = bool(paper_methods & home_methods)
            any_cross = any(
                bool(paper_methods & front_methods.get(cid, set()))
                for cid in b.get("connected_fronts", [])
            )

            if overlap_home and any_cross:
                b["verdict"] = "TRUE SYNTHESIS"
            elif any_cross:
                b["verdict"] = "CITING BRIDGE"
            elif overlap_home:
                b["verdict"] = "HOME-ANCHORED"
            else:
                b["verdict"] = "STRUCTURAL"

            all_bridges.append(b)

    return all_bridges


# ── Genealogy ──────────────────────────────────────────────────────────────────

def export_genealogy(db: Database) -> dict:
    """Build framework genealogy tree and wordcloud data."""
    if not _HAS_GENEALOGY:
        return {"frameworks": {}, "wordcloud_data": [], "summary": {}}

    try:
        # Merge lineages across all categories
        all_lineages: dict = {}
        for category in CATEGORIES:
            cat_lineages = extract_framework_lineages(db, category=category)
            for fw, fw_papers in cat_lineages.items():
                if fw not in all_lineages:
                    all_lineages[fw] = []
                existing_ids = {p["arxiv_id"] for p in all_lineages[fw]}
                for p in fw_papers:
                    if p["arxiv_id"] not in existing_ids:
                        all_lineages[fw].append(p)
                        existing_ids.add(p["arxiv_id"])

        # Sort merged paper lists by date
        for fw in all_lineages:
            all_lineages[fw].sort(key=lambda p: p.get("published_date", ""))

        tree = build_genealogy_tree(all_lineages)
        active = get_active_frameworks(all_lineages, days=30)
        active_set = {fw for fw, _ in active}
        summary = get_framework_summary(all_lineages, tree)

        # wordcloud2.js list format: [{text, weight, is_active, must_read_ratio}]
        wordcloud_data = []
        for fw, fw_papers in all_lineages.items():
            must_read_count = sum(1 for p in fw_papers if p.get("must_read", False))
            wordcloud_data.append({
                "text": fw,
                "weight": len(fw_papers),
                "is_active": fw in active_set,
                "must_read_ratio": round(must_read_count / len(fw_papers), 3) if fw_papers else 0.0,
                "must_read_count": must_read_count,
            })

        wordcloud_data.sort(key=lambda d: d["weight"], reverse=True)

        # Annotate tree nodes with is_active flag
        for fw in tree:
            tree[fw]["is_active"] = fw in active_set

        return {
            "frameworks": tree,
            "wordcloud_data": wordcloud_data,
            "summary": summary,
        }
    except Exception as exc:
        print(f"[export] Warning: genealogy export failed: {exc}")
        return {"frameworks": {}, "wordcloud_data": [], "summary": {}}


# ── Category tables ────────────────────────────────────────────────────────────

def export_category_tables(db: Database, papers: list, front_map: dict) -> dict:
    """
    Pre-flatten per-category table rows for the spreadsheet-like tab.

    Uses the `papers` intake table for authors_short (already shortened) and
    code_url fallback (actual URL, not markdown).
    """
    # Build intake-table lookup: arxiv_id -> {authors, code_url}
    intake_rows = db.fetchall("SELECT arxiv_id, authors, code_url FROM papers")
    intake_by_id = {r["arxiv_id"]: dict(r) for r in intake_rows}

    tables = {cat: [] for cat in CATEGORIES}

    for p in papers:
        cat = p["category"]
        if cat not in tables:
            continue

        pid = p["arxiv_id"]
        intake = intake_by_id.get(pid, {})

        # authors_short: prefer intake table (pre-shortened) over computing from JSON array
        authors_short = intake.get("authors") or ""
        if not authors_short:
            authors_list = p.get("authors", [])
            authors_short = (authors_list[0] + " et al.") if len(authors_list) > 1 else (authors_list[0] if authors_list else "")

        # code_url: prefer intake table (actual URL) over artifacts JSON (may be null or markdown)
        code_url = intake.get("code_url") or p.get("artifacts", {}).get("code_url") or None

        rel = p.get("relevance", {})
        sig = p.get("significance", {})
        tags = p.get("tags", {})

        tables[cat].append({
            "arxiv_id": pid,
            "arxiv_url": p["arxiv_url"],
            "title": p["title"],
            "authors_short": authors_short,
            "published_date": p["published_date"],
            "m_score": rel.get("methodological", 0),
            "p_score": rel.get("problem", 0),
            "i_score": rel.get("inspirational", 0),
            "priority_score": p["priority_score"],
            "must_read": bool(sig.get("must_read")),
            "changes_thinking": bool(sig.get("changes_thinking")),
            "team_discussion": bool(sig.get("team_discussion")),
            "front_id": p.get("front_id"),
            "front_name": None,   # filled in second pass after export_fronts()
            "front_status": p.get("front_status"),
            "methods": (tags.get("methods") or [])[:5],
            "problems": (tags.get("problems") or [])[:5],
            "code_url": code_url,
            "brief": p.get("brief", ""),
            "affiliations": p.get("affiliations", ""),
            "analysis_date": p.get("analysis_date", ""),
        })

    return tables


# ── Snapshot stats ─────────────────────────────────────────────────────────────

def build_snapshot(db: Database, papers: list, fronts: list, bridges: list) -> dict:
    """Top-level summary stats shown in the persistent header bar."""
    cutoff_7d = (date.today() - timedelta(days=7)).isoformat()

    by_cat = {}
    for cat in CATEGORIES:
        cat_papers = [p for p in papers if p["category"] == cat]
        cat_fronts = [f for f in fronts if f["category"] == cat]
        cat_bridges = [b for b in bridges if b["category"] == cat]
        by_cat[cat] = {
            "papers": len(cat_papers),
            "fronts": len(cat_fronts),
            "bridges": len(cat_bridges),
        }

    last_update = {}
    for cat in CATEGORIES:
        row = db.fetchone(
            "SELECT MAX(analysis_date) as last_date FROM paper_analyses WHERE category = ? AND is_relevant = 1",
            (cat,),
        )
        last_update[cat] = (row["last_date"] or "")[:10] if row else ""

    return {
        "total_papers": len(papers),
        "total_fronts": len(fronts),
        "total_bridges": len(bridges),
        "must_read_count": sum(1 for p in papers if p["significance"].get("must_read")),
        "changes_thinking_count": sum(1 for p in papers if p["significance"].get("changes_thinking")),
        "emerging_fronts_count": sum(1 for f in fronts if f["status"] == "emerging"),
        "last_7d_papers": sum(1 for p in papers if p["published_date"] >= cutoff_7d),
        "by_category": by_cat,
        "last_update_by_category": last_update,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export dashboard_data.json from SQLite DB")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite DB path")
    args = parser.parse_args()

    print(f"[export] DB:     {args.db}")
    print(f"[export] Output: {args.output}")

    with Database(Path(args.db)) as db:
        print("[export] Building paper/front/bridge lookup maps...")
        front_map, front_status_map, bridge_map = build_paper_maps(db)

        print("[export] Exporting papers...")
        papers = export_papers(db, front_map, front_status_map, bridge_map)
        papers_by_id = {p["arxiv_id"]: p for p in papers}
        print(f"         {len(papers)} papers")

        print("[export] Exporting fronts...")
        fronts = export_fronts(db, papers_by_id)
        front_name_map = {f["front_id"]: f.get("name", "") for f in fronts}
        print(f"         {len(fronts)} fronts")

        print("[export] Exporting bridges...")
        bridges = export_bridges(db, papers_by_id)
        print(f"         {len(bridges)} bridge papers")

        print("[export] Exporting framework genealogy...")
        genealogy = export_genealogy(db)
        print(f"         {len(genealogy['wordcloud_data'])} frameworks")

        print("[export] Building category tables...")
        category_tables = export_category_tables(db, papers, front_map)
        # Second pass: fill front names in table rows
        for cat in CATEGORIES:
            for row in category_tables[cat]:
                fid = row.get("front_id")
                row["front_name"] = front_name_map.get(fid, "") if fid else ""

        print("[export] Building snapshot stats...")
        snapshot = build_snapshot(db, papers, fronts, bridges)

    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "categories": CATEGORIES,
            "last_update_by_category": snapshot["last_update_by_category"],
        },
        "snapshot": snapshot,
        "papers": papers,
        "fronts": fronts,
        "bridges": bridges,
        "genealogy": genealogy,
        "category_tables": category_tables,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, default=str)

    size_kb = out_path.stat().st_size // 1024
    print(f"[export] Done → {out_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
