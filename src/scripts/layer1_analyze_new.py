#!/usr/bin/env python3
"""
Layer 1 CLI - Analyze New Papers

Analyzes all papers from the existing daily_arxiv.py output that haven't been
analyzed yet. Integrates with docs/or-llm-daily.json.

Usage:
    python src/scripts/layer1_analyze_new.py [--max N] [--category "Category Name"]

Examples:
    python src/scripts/layer1_analyze_new.py                    # Analyze all new papers
    python src/scripts/layer1_analyze_new.py --max 5            # Analyze first 5 papers
    python src/scripts/layer1_analyze_new.py --category "LLMs for Algorithm Design"
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.pipeline import PaperAnalysisPipeline
from db.database import Database

# Fallback JSON path used only when the papers table is empty (first run)
JSON_PATH = Path("docs/or-llm-daily.json")

def parse_pipe_delimited_paper(content: str, arxiv_id: str) -> dict:
    """
    Parse pipe-delimited paper string from JSON.

    Handles 3 format generations:
      - Legacy:  |date|title|authors|arxiv_id|code|           (6 pipes)
      - Current: |date|title|authors|venue|arxiv_id|code|     (8 pipes)
      - Newest:  |date|title|authors|affiliation|venue|arxiv_id|code| (9 pipes)
    """
    parts = content.split('|')

    if len(parts) < 6:
        print(f"  [WARN] Malformed entry for {arxiv_id}: {content[:100]}")
        return None

    try:
        if len(parts) >= 9:
            # Newest: |date|title|authors|affiliation|venue|arxiv_id|code|
            date = parts[1].strip('*').strip()
            title = parts[2].strip('*[]()').split('](')[0].strip()
            authors_str = parts[3].strip()
            affiliation = parts[4].strip()
        elif len(parts) >= 8:
            # Current: |date|title|authors|venue|arxiv_id|code|
            date = parts[1].strip('*').strip()
            title = parts[2].strip('*[]()').split('](')[0].strip()
            authors_str = parts[3].strip()
            affiliation = ''
        else:
            # Legacy: |date|title|authors|arxiv_id|code|
            date = parts[1].strip('*').strip()
            title = parts[2].strip('*[]()').split('](')[0].strip()
            authors_str = parts[3].strip()
            affiliation = ''

        # Extract first author (simple approach)
        authors = [authors_str.split()[0]] if authors_str else ['Unknown']

        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'authors': authors,
            'affiliation': affiliation,
            'abstract': '',  # Will be extracted from PDF
            'published_date': date
        }
    except Exception as e:
        print(f"  [ERROR] Failed to parse {arxiv_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze new papers with Layer 1 MAS')
    parser.add_argument('--max', type=int, help='Maximum papers to analyze')
    parser.add_argument('--category', help='Analyze only this category')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be analyzed without running')
    args = parser.parse_args()

    # Check API key (Gemini is now default)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY=your_key_here")
        print("\nAlternatively, you can configure different models in src/config.py")
        return 1

    print(f"{'='*70}")
    print(f"LAYER 1: DEEP PAPER ANALYSIS")
    print(f"{'='*70}")

    db = Database()

    # One-time migration: if papers table is empty, seed it from JSON
    with db:
        row = db.fetchone("SELECT COUNT(*) as cnt FROM papers")
        if row and row['cnt'] == 0 and JSON_PATH.exists():
            print(f"[MIGRATE] papers table is empty — seeding from {JSON_PATH} ...")
            n = db.migrate_json_to_papers(JSON_PATH)
            print(f"  Migrated {n} papers into papers table")

    # Load paper list from DB (papers table)
    with db:
        all_papers_by_cat = db.get_all_papers()

    if args.category:
        if args.category not in all_papers_by_cat:
            print(f"[WARN] Category not found in papers table: {args.category}")
            # Fallback to JSON
            if JSON_PATH.exists():
                with open(JSON_PATH) as f:
                    data = json.load(f)
                all_papers_by_cat = {k: {p['arxiv_id']: p for p in plist}
                                     for k, plist in db.get_all_papers().items()}
            else:
                return 1
        categories = [args.category]
    else:
        categories = list(all_papers_by_cat.keys())

    print(f"Source: papers table (DB)")
    print(f"Categories: {len(categories)}")
    print(f"{'='*70}\n")

    # Initialize pipeline
    if not args.dry_run:
        pipeline = PaperAnalysisPipeline()

    total_papers = 0
    total_unanalyzed = 0
    papers_to_analyze = []

    # Process each category
    for category in categories:
        cat_papers = all_papers_by_cat.get(category, [])
        print(f"\n{'='*70}")
        print(f"Category: {category}")
        print(f"{'='*70}")

        paper_ids = [p['arxiv_id'] for p in cat_papers]
        total_papers += len(paper_ids)

        with db:
            unanalyzed_ids = db.get_unanalyzed_paper_ids(category)

        print(f"Total papers: {len(paper_ids)}")
        print(f"Already analyzed: {len(paper_ids) - len(unanalyzed_ids)}")
        print(f"Unanalyzed: {len(unanalyzed_ids)}")

        if not unanalyzed_ids:
            print("  [INFO] All papers in this category already analyzed!")
            continue

        total_unanalyzed += len(unanalyzed_ids)

        # Build paper dicts for unanalyzed IDs
        paper_lookup = {p['arxiv_id']: p for p in cat_papers}
        for arxiv_id in unanalyzed_ids:
            p = paper_lookup.get(arxiv_id)
            if p is None:
                continue
            papers_to_analyze.append({
                'arxiv_id':       arxiv_id,
                'title':          p.get('title', ''),
                'authors':        [p.get('authors', 'Unknown').split()[0]] if p.get('authors') else ['Unknown'],
                'affiliation':    p.get('affiliation', ''),
                'abstract':       '',  # fetched from PDF
                'published_date': p.get('date', ''),
                'category':       category,
            })

    # Apply max limit
    if args.max:
        papers_to_analyze = papers_to_analyze[:args.max]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total papers in database: {total_papers}")
    print(f"Papers to analyze: {len(papers_to_analyze)}")
    print(f"{'='*70}\n")

    if not papers_to_analyze:
        print("[INFO] No new papers to analyze. All caught up!")
        return 0

    if args.dry_run:
        print("\n[DRY RUN] Would analyze these papers:")
        for i, paper in enumerate(papers_to_analyze, 1):
            print(f"  {i}. [{paper['category']}] {paper['arxiv_id']}: {paper['title'][:60]}...")
        return 0

    # Run analysis
    print("[START] Beginning analysis...\n")

    results = pipeline.analyze_batch(papers_to_analyze)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Successfully analyzed: {results['success']}")
    print(f"  Skipped (already done): {results['skipped']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total attempted: {len(papers_to_analyze)}")
    print(f"{'='*70}")

    if results['success'] > 0:
        print(f"\n✓ Analysis complete! Database updated with {results['success']} new papers")
        print(f"  Database: {db.db_path}")
    else:
        print(f"\n✗ No papers were successfully analyzed")
        return 1

    # Backfill is_relevant for any papers that pre-date automatic flag computation
    print(f"\n[BACKFILL] Recomputing is_relevant flags for all papers...")
    with db:
        updated = db.recompute_relevance_flags()
    print(f"  is_relevant flag set for {updated} papers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
