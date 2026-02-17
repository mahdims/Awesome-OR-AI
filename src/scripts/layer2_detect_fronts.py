#!/usr/bin/env python3
"""
Layer 2 CLI - Bibliometric Analysis & Front Detection

Runs the full Layer 2 pipeline for each category:
1. Build citation graph (Semantic Scholar API)
2. Build co-citation network
3. Detect research fronts (Louvain clustering)
4. Detect bridge papers
5. Generate front summaries (LLM)

Usage:
    python src/scripts/layer2_detect_fronts.py                          # Full pipeline
    python src/scripts/layer2_detect_fronts.py --category "LLMs for Algorithm Design"
    python src/scripts/layer2_detect_fronts.py --skip-citations         # Use cached citations
    python src/scripts/layer2_detect_fronts.py --skip-summaries         # Skip LLM summaries
    python src/scripts/layer2_detect_fronts.py --min-front-size 3       # Larger fronts only
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer2.citation_graph import CitationGraphBuilder
from layer2.front_detection import (
    build_cocitation_network,
    normalize_cocitation_weights,
    detect_fronts,
    enrich_fronts_with_tags,
    compare_with_previous,
    store_cocitation_edges,
)
from layer2.bridge_papers import detect_bridge_papers, store_bridge_papers
from layer2.front_summarizer import summarize_all_fronts


def _parse_json_list(value):
    """Parse a JSON/list value into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    return []


def _has_valid_front_enrichment(front: dict) -> bool:
    """Return True if front already has usable name/summary/future_directions."""
    name = (front.get('name') or '').strip()
    summary = (front.get('summary') or '').strip()
    fd = _parse_json_list(front.get('future_directions'))
    if not name or not summary or not fd:
        return False
    if name.startswith('[Enrichment failed]'):
        return False
    if summary.startswith('[Summary generation failed'):
        return False
    return True


def _front_papers_key(front: dict):
    """Stable key for a front based on its paper set."""
    papers = front.get('core_papers', [])
    if isinstance(papers, str):
        try:
            papers = json.loads(papers)
        except json.JSONDecodeError:
            papers = []
    return tuple(sorted(str(p) for p in papers))


def get_categories_from_db() -> list:
    """Get all categories that have analyzed papers."""
    db = Database()
    with db:
        rows = db.fetchall(
            "SELECT DISTINCT category FROM paper_analyses ORDER BY category"
        )
    return [row['category'] for row in rows]


def run_layer2_pipeline(category: str,
                         skip_citations: bool = False,
                         force_recite: bool = False,
                         skip_summaries: bool = False,
                         min_front_size: int = 2,
                         resolution: float = 1.0,
                         s2_api_key: str = None,
                         snapshot_date: str = None,
                         min_bridge_score: float = 0.5,
                         semantic_weight: float = 1.0,
                         bib_coupling_factor: float = 0.3,
                         min_similarity: float = 0.3) -> dict:
    """
    Run full Layer 2 pipeline for a single category.

    Returns dict with pipeline results.
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    print(f"\n{'#'*70}")
    print(f"# LAYER 2: {category}")
    print(f"# Snapshot: {snapshot_date}")
    print(f"{'#'*70}")

    results = {
        'category': category,
        'snapshot_date': snapshot_date,
        'citation_edges': 0,
        'cocitation_edges': 0,
        'fronts_detected': 0,
        'bridge_papers': 0,
    }

    # Check we have analyzed papers
    db = Database()
    with db:
        papers = db.get_papers_by_category(category)

    if not papers:
        print(f"\n[WARN] No analyzed papers for '{category}'. Run Layer 1 first.")
        return results

    print(f"\nAnalyzed papers: {len(papers)}")

    # Capture existing same-snapshot enrichments before clearing.
    # This avoids repeated LLM calls on reruns when the detected front composition
    # is unchanged.
    existing_enrichment_by_id = {}
    existing_enrichment_by_papers = {}
    if not skip_summaries:
        with db:
            rows = db.fetchall(
                """SELECT front_id, core_papers, name, summary, future_directions
                   FROM research_fronts
                   WHERE category = ? AND snapshot_date = ?""",
                (category, snapshot_date)
            )

        for r in rows:
            row = dict(r)
            if not _has_valid_front_enrichment(row):
                continue
            payload = {
                'name': (row.get('name') or '').strip(),
                'summary': (row.get('summary') or '').strip(),
                'future_directions': _parse_json_list(row.get('future_directions')),
            }
            existing_enrichment_by_id[row['front_id']] = payload
            existing_enrichment_by_papers[_front_papers_key(row)] = payload

        if existing_enrichment_by_id:
            print(f"  Found {len(existing_enrichment_by_id)} reusable same-snapshot front enrichments")

    # Step 1: Build citation graph
    builder = CitationGraphBuilder(s2_api_key=s2_api_key)

    if skip_citations:
        # Pure DB load — no API calls even for unfetched papers
        print(f"\n[STEP 1/5] Loading citation graph from database (--skip-citations)...")
        citation_graph = builder.load_citation_graph_from_db(category)
        if citation_graph.number_of_edges() == 0:
            print("[WARN] No cached citations found. Run without --skip-citations first.")
            return results
    else:
        # Smart incremental fetch: only new papers hit Semantic Scholar
        print(f"\n[STEP 1/5] Building citation graph (incremental)...")
        citation_graph = builder.build_citation_graph(
            category, fetch_mode="both", force_refresh=force_recite
        )

    results['citation_edges'] = citation_graph.number_of_edges()
    print(f"  Citation graph: {citation_graph.number_of_nodes()} nodes, "
          f"{citation_graph.number_of_edges()} edges")

    # Step 2: Build co-citation network (db kept open for semantic edge injection)
    print(f"\n[STEP 2/5] Building co-citation network...")
    corpus_ids = {p['arxiv_id'] for p in papers}
    with db:
        cocitation = build_cocitation_network(
            citation_graph, corpus_ids,
            db=db,
            semantic_weight=semantic_weight,
            bib_coupling_factor=bib_coupling_factor,
            min_similarity=min_similarity,
        )
    cocitation = normalize_cocitation_weights(cocitation)

    results['cocitation_edges'] = cocitation.number_of_edges()
    print(f"  Co-citation network: {cocitation.number_of_nodes()} nodes, "
          f"{cocitation.number_of_edges()} edges")

    # Clear any stale fronts/bridges/cocitation-edges from a previous same-day
    # run BEFORE storing new edges — and before compare_with_previous so that
    # get_latest_fronts returns the real prior-day snapshot.
    with db:
        db.clear_fronts_for_snapshot(category, snapshot_date)

    # Store co-citation edges (snapshot is now clean)
    with db:
        store_cocitation_edges(cocitation, category, snapshot_date, db)

    # Step 3: Detect fronts
    print(f"\n[STEP 3/5] Detecting research fronts (Louvain, resolution={resolution})...")
    fronts = detect_fronts(cocitation, category, snapshot_date,
                           min_front_size, resolution)

    if not fronts:
        print("  [INFO] No fronts detected. Try lowering --min-front-size or check citation coverage.")
        return results

    results['fronts_detected'] = len(fronts)

    # Enrich with Layer 1 tags
    print(f"\n  Enriching with Layer 1 tags...")
    with db:
        fronts = enrich_fronts_with_tags(fronts, db)

    # Compare with previous snapshot (DB now contains only older snapshots)
    print(f"\n  Tracking front evolution...")
    with db:
        fronts = compare_with_previous(fronts, category, db)

    # Reuse previously generated enrichments when front identity/composition matches.
    reused_enrichments = 0
    if existing_enrichment_by_id or existing_enrichment_by_papers:
        for front in fronts:
            payload = existing_enrichment_by_id.get(front['front_id'])
            if payload is None:
                payload = existing_enrichment_by_papers.get(_front_papers_key(front))
            if payload is None:
                continue
            front.update(payload)
            reused_enrichments += 1
        if reused_enrichments:
            print(f"  Reused {reused_enrichments} existing front enrichments (LLM calls avoided)")

    # Store the newly-computed fronts
    with db:
        for front in fronts:
            db.insert_front(front)

    # Step 4: Detect bridge papers
    print(f"\n[STEP 4/5] Detecting bridge papers...")
    bridges = detect_bridge_papers(cocitation, fronts, category, snapshot_date,
                                    min_bridge_score=min_bridge_score)
    results['bridge_papers'] = len(bridges)

    if bridges:
        with db:
            store_bridge_papers(bridges, db)

    # Step 5: Generate front summaries
    if skip_summaries:
        print(f"\n[STEP 5/5] Skipping LLM summaries (--skip-summaries)")
    else:
        fronts_needing_llm = [f for f in fronts if not _has_valid_front_enrichment(f)]
        if not fronts_needing_llm:
            print(f"\n[STEP 5/5] All fronts already enriched (no LLM calls needed)")
        else:
            print(f"\n[STEP 5/5] Generating front summaries with LLM ({len(fronts_needing_llm)}/{len(fronts)} fronts need enrichment)...")
            summarize_all_fronts(fronts_needing_llm, update_db=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {category}")
    print(f"{'='*60}")
    print(f"  Citation edges:    {results['citation_edges']}")
    print(f"  Co-citation edges: {results['cocitation_edges']}")
    print(f"  Research fronts:   {results['fronts_detected']}")
    print(f"  Bridge papers:     {results['bridge_papers']}")

    for front in fronts:
        methods = front.get('dominant_methods', [])
        if isinstance(methods, str):
            methods = json.loads(methods)
        print(f"\n  Front: {front['front_id']}")
        print(f"    Size: {front['size']}, Status: {front.get('status', '?')}")
        print(f"    Methods: {', '.join(methods[:3]) if methods else 'N/A'}")
        if front.get('summary'):
            # Print first line of summary
            first_line = front['summary'].split('\n')[0][:80]
            print(f"    Summary: {first_line}...")

    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Layer 2: Bibliometric Analysis & Front Detection'
    )
    parser.add_argument('--category', help='Process only this category')
    parser.add_argument('--skip-citations', action='store_true',
                        help='Load citation graph from DB only — no API calls at all')
    parser.add_argument('--force-recite', action='store_true',
                        help='Re-fetch ALL corpus papers from Semantic Scholar, ignoring cache')
    parser.add_argument('--skip-summaries', action='store_true',
                        help='Skip LLM summary generation')
    parser.add_argument('--min-front-size', type=int, default=2,
                        help='Minimum papers per front (default: 2)')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Louvain resolution (higher=more communities, default: 1.0)')
    parser.add_argument('--snapshot-date', help='Override snapshot date (YYYY-MM-DD)')
    parser.add_argument('--s2-api-key', help='Semantic Scholar API key (optional, for higher rate limits)')
    parser.add_argument('--min-bridge-score', type=float, default=0.5,
                        help='Min fraction of cross-front edge weight to qualify as bridge paper (default: 0.5)')
    parser.add_argument('--semantic-weight', type=float, default=1.0,
                        help='Weight for semantic similarity edges from Layer 1 data (default: 1.0, 0=citation-only)')
    parser.add_argument('--bib-coupling-factor', type=float, default=0.0,
                        help='Scale for bibliographic coupling edges (default: 0.0, 0=disable)')
    parser.add_argument('--min-similarity', type=float, default=0.9,
                        help='Min IDF-weighted similarity to create a semantic edge (default: 0.9)')
    args = parser.parse_args()

    # Check for Gemini API key if summaries are needed
    if not args.skip_summaries and not os.getenv("GEMINI_API_KEY"):
        print("[WARN] GEMINI_API_KEY not set. Front summaries will be skipped.")
        print("       Set it with: export GEMINI_API_KEY=your_key_here")
        args.skip_summaries = True

    # S2 API key from env if not provided
    s2_api_key = args.s2_api_key or os.getenv("S2_API_KEY")

    print(f"{'='*70}")
    print(f"LAYER 2: BIBLIOMETRIC ANALYSIS & FRONT DETECTION")
    print(f"{'='*70}")

    # Get categories
    if args.category:
        categories = [args.category]
    else:
        categories = get_categories_from_db()

    if not categories:
        print("\n[ERROR] No analyzed papers found in database.")
        print("Run Layer 1 first: python src/scripts/layer1_analyze_new.py")
        return 1

    print(f"Categories to process: {len(categories)}")
    for cat in categories:
        print(f"  - {cat}")

    # Process each category
    all_results = []
    for category in categories:
        result = run_layer2_pipeline(
            category=category,
            skip_citations=args.skip_citations,
            force_recite=args.force_recite,
            skip_summaries=args.skip_summaries,
            min_front_size=args.min_front_size,
            resolution=args.resolution,
            s2_api_key=s2_api_key,
            snapshot_date=args.snapshot_date,
            min_bridge_score=args.min_bridge_score,
            semantic_weight=args.semantic_weight,
            bib_coupling_factor=args.bib_coupling_factor,
            min_similarity=args.min_similarity,
        )
        all_results.append(result)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"LAYER 2 COMPLETE")
    print(f"{'='*70}")
    total_fronts = sum(r['fronts_detected'] for r in all_results)
    total_bridges = sum(r['bridge_papers'] for r in all_results)
    print(f"  Categories processed: {len(all_results)}")
    print(f"  Total fronts detected: {total_fronts}")
    print(f"  Total bridge papers: {total_bridges}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
