#!/usr/bin/env python3
"""
Layer 1 Re-tagger — Refresh Methods agent output on existing papers.

Re-runs ONLY the Methods Extractor agent using already-stored Reader output
from the database. No PDF re-download needed — Reader output contains all
the information the Methods agent requires.

Use this after improving src/layer1/prompts/methods.txt to re-tag all
existing papers with better, more specific tags without re-running the
expensive Reader and Positioning agents.

Usage:
    python src/scripts/layer1_retag.py
    python src/scripts/layer1_retag.py --category "LLMs for Algorithm Design"
    python src/scripts/layer1_retag.py --dry-run          # show diffs, no DB writes
    python src/scripts/layer1_retag.py --limit 5          # process only N papers
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer1.schemas import MethodsOutput
from llm_client import create_agent_client

PROMPTS_DIR = Path(__file__).parent.parent / "layer1" / "prompts"


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json_field(value):
    """Parse a JSON string from the DB or return as-is if already a dict/list."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def build_reader_context_from_db(analysis: dict) -> str:
    """
    Reconstruct Reader agent output context from stored DB fields.

    Provides the same information the Methods agent would normally receive
    from the Reader's structured output, without re-running the Reader.
    """
    parts = ["PAPER TITLE: " + analysis.get('title', '(unknown)')]
    parts.append("ABSTRACT: " + (analysis.get('abstract', '') or '')[:600])
    parts.append("")
    parts.append("READER ANALYSIS:")

    for field in ['problem', 'methodology', 'experiments', 'results', 'artifacts']:
        value = _parse_json_field(analysis.get(field))
        if value:
            parts.append(f"  {field}: {json.dumps(value, indent=2)}")

    affiliations = analysis.get('affiliations', '')
    if affiliations:
        parts.append(f"  affiliations: {affiliations}")

    return '\n'.join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Tag diff display
# ──────────────────────────────────────────────────────────────────────────────

def _format_tags(tags: dict) -> str:
    """Format tags dict for display."""
    methods   = tags.get('methods', [])
    problems  = tags.get('problems', [])
    contrib   = tags.get('contribution_type', [])
    lineage   = tags.get('framework_lineage')
    domain    = tags.get('specific_domain')
    coupling  = tags.get('llm_coupling')
    lines = [
        f"  methods:          {', '.join(methods) or '(none)'}",
        f"  problems:         {', '.join(problems) or '(none)'}",
        f"  contribution:     {', '.join(contrib) or '(none)'}",
        f"  framework_lineage: {lineage or '(null)'}",
        f"  specific_domain:   {domain or '(null)'}",
        f"  llm_coupling:      {coupling or '(null)'}",
    ]
    return '\n'.join(lines)


def _show_diff(arxiv_id: str, title: str, old_tags: dict, new_tags: dict) -> None:
    """Print a before/after tag comparison."""
    print(f"\n{'─'*60}")
    print(f"  {arxiv_id}  {title[:55]}...")
    print("  BEFORE:")
    print(_format_tags(old_tags))
    print("  AFTER:")
    print(_format_tags(new_tags))


# ──────────────────────────────────────────────────────────────────────────────
# DB update helper
# ──────────────────────────────────────────────────────────────────────────────

def _update_methods_in_db(db: Database, arxiv_id: str,
                           methods_output: MethodsOutput) -> None:
    """Update only the methods-related columns for an existing paper."""
    tags_dict       = methods_output.tags.model_dump()
    lineage_dict    = methods_output.lineage.model_dump()
    extensions_dict = methods_output.extensions.model_dump()
    conf_dict       = (methods_output.confidence.model_dump()
                       if methods_output.confidence else None)

    db.execute(
        """UPDATE paper_analyses
           SET tags = ?, lineage = ?, extensions = ?, methods_confidence = ?
           WHERE arxiv_id = ?""",
        (
            json.dumps(tags_dict),
            json.dumps(lineage_dict),
            json.dumps(extensions_dict),
            json.dumps(conf_dict) if conf_dict else None,
            arxiv_id,
        )
    )
    db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def retag_papers(category: str = None,
                 dry_run: bool = False,
                 limit: int = None) -> dict:
    """
    Re-tag existing papers with the improved Methods agent.

    Args:
        category: If given, only re-tag papers in this category.
        dry_run:  If True, print diffs but do not write to DB.
        limit:    Maximum number of papers to process.

    Returns:
        Dict with success/failed/skipped counts.
    """
    db = Database()
    methods_client = create_agent_client('methods_extractor')
    methods_prompt = (PROMPTS_DIR / "methods.txt").read_text(encoding='utf-8')

    print(f"\n{'='*70}")
    print("LAYER 1 RE-TAGGER")
    print(f"{'='*70}")
    print(f"  Model:    {methods_client.config.provider} / {methods_client.config.model_name}")
    print(f"  Category: {category or 'ALL'}")
    print(f"  Dry run:  {dry_run}")
    if limit:
        print(f"  Limit:    {limit}")

    with db:
        if category:
            papers = db.get_papers_by_category(category)
        else:
            papers = [dict(r) for r in db.fetchall(
                "SELECT * FROM paper_analyses ORDER BY category, published_date DESC"
            )]

    if limit:
        papers = papers[:limit]

    print(f"\n  Papers to process: {len(papers)}")

    results = {'success': 0, 'failed': 0, 'unchanged': 0}

    for i, analysis in enumerate(papers, 1):
        arxiv_id = analysis['arxiv_id']
        title    = analysis.get('title', '(unknown)')
        cat      = analysis.get('category', '')
        old_tags = _parse_json_field(analysis.get('tags')) or {}

        print(f"\n[{i}/{len(papers)}] {arxiv_id}  ({cat})")
        print(f"  {title[:65]}...")

        try:
            # Build context from stored Reader output
            reader_context = build_reader_context_from_db(analysis)

            # Call Methods agent (no PDF text — Reader analysis only)
            methods_output: MethodsOutput = methods_client.generate_json(
                system_prompt=methods_prompt,
                user_prompt=reader_context,
                output_schema=MethodsOutput,
            )

            new_tags = methods_output.tags.model_dump()

            # Print diff
            _show_diff(arxiv_id, title, old_tags, new_tags)

            if dry_run:
                print("  [DRY RUN] No DB write")
                results['success'] += 1
                continue

            # Write to DB
            with db:
                _update_methods_in_db(db, arxiv_id, methods_output)

            results['success'] += 1
            print(f"  ✓ Updated")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results['failed'] += 1

    print(f"\n{'='*70}")
    print("RE-TAGGER COMPLETE")
    print(f"{'='*70}")
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Re-run Methods Extractor on existing papers with improved prompt'
    )
    parser.add_argument('--category', help='Re-tag only this category')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show tag diffs without writing to DB')
    parser.add_argument('--limit', type=int,
                        help='Process at most N papers (useful for testing)')
    args = parser.parse_args()

    import os
    if not os.getenv("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY not set.")
        print("        Set it with: export GEMINI_API_KEY=your_key_here")
        return 1

    retag_papers(
        category=args.category,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
