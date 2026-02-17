#!/usr/bin/env python3
"""
Layer 2 Analysis: Front Characteristics & Bridge Paper Profiling

Two analyses:
  1. Front characteristics — for each detected front:
       - Unique vs. shared methods across fronts
       - Dominant problems, density interpretation
       - Per-paper methodology summary (LLM role, core method)

  2. Top-N bridge paper profiling — for the highest-scoring bridges:
       - Whether the paper is a true methodological synthesizer
         (uses methods from BOTH connected fronts) or merely a citing bridge
       - Layer 1 brief and significance

Usage:
    python src/scripts/layer2_analyze_fronts.py
    python src/scripts/layer2_analyze_fronts.py --category "LLMs for Algorithm Design"
    python src/scripts/layer2_analyze_fronts.py --snapshot-date 2026-02-12
    python src/scripts/layer2_analyze_fronts.py --top-bridges 4
    python src/scripts/layer2_analyze_fronts.py --enrich    # run LLM enrichment first (needs GEMINI_API_KEY)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer2.front_summarizer import summarize_all_fronts


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json(value, default=None):
    """Parse a JSON string stored in the DB, or return default."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _get_latest_snapshot(db: Database, category: str) -> Optional[str]:
    """Return the most recent snapshot_date for a category."""
    row = db.fetchone(
        "SELECT MAX(snapshot_date) AS snap FROM research_fronts WHERE category = ?",
        (category,)
    )
    return row["snap"] if row else None


def _get_fronts(db: Database, category: str, snapshot_date: str) -> list[dict]:
    rows = db.fetchall(
        """SELECT * FROM research_fronts
           WHERE category = ? AND snapshot_date = ?
           ORDER BY size DESC""",
        (category, snapshot_date)
    )
    fronts = []
    for r in rows:
        f = dict(r)
        f["core_papers"] = _json(f.get("core_papers"), [])
        f["dominant_methods"] = _json(f.get("dominant_methods"), [])
        f["dominant_problems"] = _json(f.get("dominant_problems"), [])
        fronts.append(f)
    return fronts


def _get_bridge_papers(db: Database, category: str, snapshot_date: str,
                        top_n: int) -> list[dict]:
    rows = db.fetchall(
        """SELECT * FROM bridge_papers
           WHERE category = ? AND snapshot_date = ?
           ORDER BY bridge_score DESC
           LIMIT ?""",
        (category, snapshot_date, top_n)
    )
    bridges = []
    for r in rows:
        b = dict(r)
        b["connected_fronts"] = _json(b.get("connected_fronts"), [])
        bridges.append(b)
    return bridges


def _get_paper_analysis(db: Database, arxiv_id: str) -> Optional[dict]:
    return db.get_analysis(arxiv_id)


def _methods_of_front(db: Database, front: dict) -> set[str]:
    """All methods (union) across papers in a front."""
    methods = set()
    for pid in front["core_papers"]:
        a = _get_paper_analysis(db, pid)
        if not a:
            continue
        tags = _json(a.get("tags"), {})
        methods.update(tags.get("methods", []))
    return methods


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Front Characteristics
# ──────────────────────────────────────────────────────────────────────────────

def analyze_fronts(db: Database, fronts: list[dict]) -> None:
    separator = "=" * 70

    # Build per-front method sets for uniqueness comparison
    front_method_sets: dict[str, set] = {}
    for f in fronts:
        front_method_sets[f["front_id"]] = _methods_of_front(db, f)

    all_methods_combined = [m for s in front_method_sets.values() for m in s]
    method_global_count = Counter(all_methods_combined)

    print(f"\n{separator}")
    print("FRONT CHARACTERISTICS")
    print(f"{separator}")

    for f in fronts:
        fid = f["front_id"]
        size = f["size"]
        density = f["internal_density"]
        status = f.get("status", "?")

        # Density label
        if density >= 0.95:
            density_label = "fully connected (tight cluster)"
        elif density >= 0.6:
            density_label = "dense (coherent theme)"
        elif density >= 0.3:
            density_label = "moderate (loosely coupled)"
        else:
            density_label = "sparse (broad umbrella)"

        print(f"\n{'─'*60}")
        print(f"  {fid}")
        name = f.get('name') or ''
        if name:
            print(f"  Name  : {name}")
        print(f"  Size: {size} papers   Density: {density:.3f} ({density_label})")
        print(f"  Status: {status}")

        # Method uniqueness
        my_methods = front_method_sets.get(fid, set())
        unique = sorted(m for m in my_methods if method_global_count[m] == 1)
        shared = sorted(m for m in my_methods if method_global_count[m] > 1)

        print(f"\n  Methods:")
        if shared:
            print(f"    Shared across fronts : {', '.join(shared)}")
        if unique:
            print(f"    Unique to this front : {', '.join(unique)}")
        if not shared and not unique:
            print(f"    (no method tags available)")

        # Dominant problems
        if f["dominant_problems"]:
            print(f"\n  Top problems: {', '.join(f['dominant_problems'][:5])}")

        # Per-paper methodology snapshot
        print(f"\n  Papers:")
        for pid in f["core_papers"]:
            a = _get_paper_analysis(db, pid)
            if not a:
                print(f"    [{pid}]  (no Layer 1 analysis)")
                continue

            title = a.get("title", "")[:65]
            pub_date = a.get("published_date", "")[:7]  # YYYY-MM

            meth = _json(a.get("methodology"), {})
            llm_role = meth.get("llm_role", "?")
            core_method = meth.get("core_method", "?")[:40]

            sig = _json(a.get("significance"), {})
            must_read = " [MUST-READ]" if sig.get("must_read") else ""

            print(f"    [{pub_date}] {pid}  {title}...")
            print(f"             LLM role: {llm_role}  |  Core method: {core_method}{must_read}")

    # LLM-enriched summary + future directions per front
    print(f"\n{'─'*60}")
    print(f"  LLM ENRICHMENT (name / summary / future directions)")
    for f in fronts:
        label = f.get('name') or f['front_id'].split('_front_')[-1]
        summary = (f.get('summary') or '').strip()
        fd = _json(f.get('future_directions'), [])

        print(f"\n  ── Front: {label} ──")

        if summary:
            # Print each paragraph indented
            for para in summary.split('\n\n'):
                para = para.strip()
                if para:
                    print(f"    {para}")
        else:
            print(f"    (no summary — run with --enrich to generate)")

        if fd:
            print(f"\n    Future directions:")
            for j, direction in enumerate(fd, 1):
                print(f"      {j}. {direction}")
        else:
            print(f"    Future directions: (none — run with --enrich)")


    # Method overlap matrix between fronts
    if len(fronts) > 1:
        print(f"\n{'─'*60}")
        print(f"  Method Overlap Matrix (shared methods between front pairs)")
        print(f"  {'':>8}", end="")
        short_ids = [f["front_id"].split("_front_")[-1] for f in fronts]
        for sid in short_ids:
            print(f"  F{sid:>2}", end="")
        print()
        for i, fi in enumerate(fronts):
            print(f"  Front {short_ids[i]:>2}", end="")
            for j, fj in enumerate(fronts):
                if i == j:
                    print(f"   — ", end="")
                else:
                    shared_count = len(
                        front_method_sets[fi["front_id"]] &
                        front_method_sets[fj["front_id"]]
                    )
                    print(f"  {shared_count:>3}", end="")
            print()


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Top Bridge Paper Profiling
# ──────────────────────────────────────────────────────────────────────────────

def analyze_bridge_papers(db: Database, bridges: list[dict],
                           fronts: list[dict]) -> None:
    separator = "=" * 70
    front_lookup = {f["front_id"]: f for f in fronts}
    front_method_sets = {f["front_id"]: _methods_of_front(db, f) for f in fronts}

    print(f"\n{separator}")
    print(f"TOP {len(bridges)} BRIDGE PAPERS — SYNTHESIS PROFILING")
    print(f"{separator}")

    for rank, b in enumerate(bridges, 1):
        pid = b["paper_id"]
        score = b["bridge_score"]
        home_id = b["home_front_id"]
        connected = b["connected_fronts"]

        a = _get_paper_analysis(db, pid)

        print(f"\n{'─'*60}")
        print(f"  #{rank}  {pid}  (bridge_score={score:.3f})")

        if a:
            print(f"  Title : {a.get('title', 'N/A')}")
            print(f"  Date  : {a.get('published_date', '?')[:7]}")

        # Front topology
        home_short = home_id.split("_front_")[-1] if home_id else "?"
        conn_short = [c.split("_front_")[-1] for c in connected]
        print(f"  Home front  : Front {home_short}  "
              f"(size={front_lookup.get(home_id, {}).get('size', '?')})")
        print(f"  Connects to : Front(s) {', '.join(conn_short)}")

        if not a:
            print("  (no Layer 1 analysis available)")
            continue

        # Methodological synthesis check
        tags = _json(a.get("tags"), {})
        paper_methods = set(tags.get("methods", []))

        print(f"\n  Synthesis Analysis:")
        print(f"    Paper methods : {', '.join(sorted(paper_methods)) or '(none tagged)'}")

        home_methods = front_method_sets.get(home_id, set())
        overlap_home = paper_methods & home_methods
        print(f"    Home front methods overlap  : "
              f"{', '.join(sorted(overlap_home)) or '(none)'}")

        cross_overlaps = []
        for cid in connected:
            conn_methods = front_method_sets.get(cid, set())
            overlap_conn = paper_methods & conn_methods
            short = cid.split("_front_")[-1]
            cross_overlaps.append((short, overlap_conn))
            print(f"    Front {short} methods overlap     : "
                  f"{', '.join(sorted(overlap_conn)) or '(none)'}")

        # Verdict
        any_cross_overlap = any(ov for _, ov in cross_overlaps)
        if overlap_home and any_cross_overlap:
            verdict = "TRUE SYNTHESIS — uses methods from home and connected fronts"
        elif any_cross_overlap:
            verdict = "CITING BRIDGE (connected front methods, not home front methods)"
        elif overlap_home:
            verdict = "HOME-ANCHORED BRIDGE (cites widely but methods stay within home front)"
        else:
            verdict = "STRUCTURAL BRIDGE (topological only; method overlap unclear)"
        print(f"    → {verdict}")

        # LLM role and core method
        meth = _json(a.get("methodology"), {})
        print(f"\n  Methodology:")
        print(f"    LLM role    : {meth.get('llm_role', '?')}")
        print(f"    Core method : {meth.get('core_method', '?')}")
        print(f"    Novelty     : {meth.get('novelty_claim', '?')[:80]}")

        # Significance
        sig = _json(a.get("significance"), {})
        flags = []
        if sig.get("must_read"):
            flags.append("MUST-READ")
        if sig.get("changes_thinking"):
            flags.append("CHANGES-THINKING")
        if sig.get("team_discussion"):
            flags.append("TEAM-DISCUSSION")
        if flags:
            print(f"  Significance: {' | '.join(flags)}")
            if sig.get("reasoning"):
                print(f"    {sig['reasoning'][:120]}")

        # One-line brief
        brief = a.get("brief", "")
        if brief:
            print(f"  Brief: {brief[:160]}{'...' if len(brief) > 160 else ''}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2 Analysis: Front Characteristics & Bridge Paper Profiling"
    )
    parser.add_argument("--category", help="Category to analyze (default: first in DB)")
    parser.add_argument("--snapshot-date", help="Snapshot date (YYYY-MM-DD, default: latest)")
    parser.add_argument("--top-bridges", type=int, default=4,
                        help="Number of top bridge papers to profile (default: 4)")
    parser.add_argument("--enrich", action="store_true",
                        help="Run LLM enrichment (name + summary + future directions) "
                             "on the loaded fronts before displaying. Requires GEMINI_API_KEY.")
    args = parser.parse_args()

    db = Database()

    # Resolve category
    with db:
        if args.category:
            category = args.category
        else:
            row = db.fetchone(
                "SELECT DISTINCT category FROM research_fronts ORDER BY category LIMIT 1"
            )
            if not row:
                print("[ERROR] No fronts in database. Run layer2_detect_fronts.py first.")
                return 1
            category = row["category"]

        # Resolve snapshot date
        snapshot_date = args.snapshot_date or _get_latest_snapshot(db, category)
        if not snapshot_date:
            print(f"[ERROR] No fronts found for '{category}'.")
            return 1

        fronts = _get_fronts(db, category, snapshot_date)
        bridges = _get_bridge_papers(db, category, snapshot_date, args.top_bridges)

    print(f"\n{'='*70}")
    print(f"LAYER 2 ANALYSIS: {category}")
    print(f"Snapshot: {snapshot_date}   Fronts: {len(fronts)}   "
          f"Top bridges shown: {args.top_bridges}")
    print(f"{'='*70}")

    if not fronts:
        print("[WARN] No fronts to analyze.")
        return 0

    # Optionally run LLM enrichment first
    if args.enrich:
        import os
        if not os.getenv("GEMINI_API_KEY"):
            print("[ERROR] --enrich requires GEMINI_API_KEY to be set.")
            return 1
        print("\n[ENRICH] Generating name + summary + future directions via LLM...")
        fronts = summarize_all_fronts(fronts, update_db=True)

    with db:
        analyze_fronts(db, fronts)
        if bridges:
            analyze_bridge_papers(db, bridges, fronts)
        else:
            print("\n[INFO] No bridge papers found for this snapshot.")

    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
