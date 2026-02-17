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

JSON_PATH = Path("docs/or-llm-daily.json")

def parse_pipe_delimited_paper(content: str, arxiv_id: str) -> dict:
    """
    Parse pipe-delimited paper string from JSON.

    Format: |date|title|authors|venue|arxiv_id|code_link|
    """
    parts = content.split('|')

    if len(parts) < 6:
        print(f"  [WARN] Malformed entry for {arxiv_id}: {content[:100]}")
        return None

    try:
        date = parts[1].strip('*').strip()
        title = parts[2].strip('*[]()').split('](')[0].strip()
        authors_str = parts[3].strip()

        # Extract first author (simple approach)
        authors = [authors_str.split()[0]] if authors_str else ['Unknown']

        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'authors': authors,
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

    # Load papers from existing JSON
    if not JSON_PATH.exists():
        print(f"ERROR: Paper database not found at {JSON_PATH}")
        print("Run daily_arxiv.py first to collect papers")
        return 1

    with open(JSON_PATH) as f:
        data = json.load(f)

    print(f"{'='*70}")
    print(f"LAYER 1: DEEP PAPER ANALYSIS")
    print(f"{'='*70}")
    print(f"Source: {JSON_PATH}")
    print(f"Categories: {len(data)}")
    print(f"{'='*70}\n")

    # Initialize pipeline (API key read from environment via config)
    if not args.dry_run:
        pipeline = PaperAnalysisPipeline()
        db = Database()
    else:
        db = Database()

    total_papers = 0
    total_unanalyzed = 0
    papers_to_analyze = []

    # Process each category
    categories = [args.category] if args.category else data.keys()

    for category in categories:
        if category not in data:
            print(f"[WARN] Category not found: {category}")
            continue

        papers = data[category]
        print(f"\n{'='*70}")
        print(f"Category: {category}")
        print(f"{'='*70}")

        # Get unanalyzed papers
        paper_ids = list(papers.keys())
        total_papers += len(paper_ids)

        with db:
            unanalyzed = db.get_unanalyzed_papers(category, paper_ids)

        print(f"Total papers: {len(paper_ids)}")
        print(f"Already analyzed: {len(paper_ids) - len(unanalyzed)}")
        print(f"Unanalyzed: {len(unanalyzed)}")

        if not unanalyzed:
            print("  [INFO] All papers in this category already analyzed!")
            continue

        total_unanalyzed += len(unanalyzed)

        # Parse unanalyzed papers
        for arxiv_id in unanalyzed:
            content = papers[arxiv_id]
            parsed = parse_pipe_delimited_paper(content, arxiv_id)

            if parsed:
                parsed['category'] = category
                papers_to_analyze.append(parsed)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
