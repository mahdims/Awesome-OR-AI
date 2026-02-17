#!/usr/bin/env python3
"""
Re-run Positioning Agent Only

Re-scores all analyzed papers using the updated positioning prompt,
without re-running the Reader or Methods agents. Uses cached PDF text
and existing Reader/Methods outputs from the database.

Usage:
    python src/scripts/rerun_positioning.py                    # All papers
    python src/scripts/rerun_positioning.py --max 5            # First 5
    python src/scripts/rerun_positioning.py --category "LLMs for Algorithm Design"
    python src/scripts/rerun_positioning.py --dry-run           # Preview only
"""

import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from llm_client import create_agent_client
from layer1.schemas import (
    ReaderOutput, MethodsOutput, PositioningOutput,
)
from layer1.pdf_extractor import fetch_arxiv_pdf_text

PROMPTS_DIR = Path(__file__).parent.parent / "layer1" / "prompts"


def rerun_positioning(arxiv_id: str, analysis: dict, positioning_client,
                       positioning_prompt: str, researcher_profile: str,
                       db: Database) -> bool:
    """Re-run positioning agent for a single paper."""
    print(f"\n  [{arxiv_id}] {analysis.get('title', '?')[:60]}...")

    # Reconstruct Reader and Methods outputs from DB
    try:
        def _parse(val):
            if isinstance(val, str):
                return json.loads(val)
            return val if val is not None else {}

        # Fix class/class_ alias: DB stores "class" but Pydantic expects "class" (via alias)
        problem_data = _parse(analysis['problem'])
        if 'class_' in problem_data and 'class' not in problem_data:
            problem_data['class'] = problem_data.pop('class_')

        reader_data = {
            'affiliations': analysis.get('affiliations') or '',
            'problem': problem_data,
            'methodology': _parse(analysis['methodology']),
            'experiments': _parse(analysis['experiments']),
            'results': _parse(analysis['results']),
            'artifacts': _parse(analysis['artifacts']),
        }
        reader_confidence = analysis.get('reader_confidence')
        if reader_confidence:
            reader_data['confidence'] = _parse(reader_confidence)

        methods_data = {
            'lineage': _parse(analysis['lineage']),
            'tags': _parse(analysis['tags']),
            'extensions': _parse(analysis['extensions']),
        }
        methods_confidence = analysis.get('methods_confidence')
        if methods_confidence:
            methods_data['confidence'] = _parse(methods_confidence)

        reader_output = ReaderOutput.model_validate(reader_data)
        methods_output = MethodsOutput.model_validate(methods_data)
    except Exception as e:
        print(f"    [SKIP] Could not reconstruct agent outputs: {e}")
        return False

    # Fetch PDF text (uses cache)
    try:
        pdf_text, _ = fetch_arxiv_pdf_text(arxiv_id)
    except Exception as e:
        print(f"    [SKIP] Could not fetch PDF: {e}")
        return False

    # Build context (same as pipeline.py)
    context = (
        f"PAPER:\n{pdf_text}\n\n"
        f"READER:\n{reader_output.model_dump_json(indent=2)}\n\n"
        f"METHODS:\n{methods_output.model_dump_json(indent=2)}\n\n"
        f"RESEARCHER PROFILE:\n{researcher_profile}"
    )

    # Run positioning agent
    try:
        positioning_output = positioning_client.generate_json(
            system_prompt=positioning_prompt,
            user_prompt=context,
            output_schema=PositioningOutput,
        )
    except Exception as e:
        print(f"    [FAIL] Positioning agent error: {e}")
        return False

    # Update database
    relevance = positioning_output.relevance_scores.model_dump()
    significance = positioning_output.significance.model_dump()
    brief = positioning_output.brief

    db.execute(
        """UPDATE paper_analyses
           SET relevance = ?, significance = ?, brief = ?
           WHERE arxiv_id = ?""",
        (json.dumps(relevance), json.dumps(significance), brief, arxiv_id)
    )
    db.commit()

    m = positioning_output.relevance_scores.methodological
    p = positioning_output.relevance_scores.problem
    i = positioning_output.relevance_scores.inspirational
    must = positioning_output.significance.must_read
    print(f"    M={m} P={p} I={i} must_read={must}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Re-run positioning agent on analyzed papers')
    parser.add_argument('--max', type=int, help='Maximum papers to re-score')
    parser.add_argument('--category', help='Only this category')
    parser.add_argument('--dry-run', action='store_true', help='Preview without running')
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return 1

    db = Database()

    # Get papers to re-score
    with db:
        if args.category:
            papers = db.get_papers_by_category(args.category)
        else:
            papers = db.fetchall("SELECT * FROM paper_analyses ORDER BY published_date DESC")
            papers = [dict(row) for row in papers]

    if args.max:
        papers = papers[:args.max]

    print(f"{'='*70}")
    print(f"RE-RUN POSITIONING AGENT")
    print(f"{'='*70}")
    print(f"Papers to re-score: {len(papers)}")

    if not papers:
        print("[INFO] No papers found.")
        return 0

    if args.dry_run:
        for p in papers:
            rel = p.get('relevance', '{}')
            if isinstance(rel, str):
                rel = json.loads(rel)
            old_m = rel.get('methodological', '?')
            old_p = rel.get('problem', '?')
            old_i = rel.get('inspirational', '?')
            print(f"  {p['arxiv_id']}: M={old_m} P={old_p} I={old_i} — {p.get('title', '?')[:50]}...")
        print(f"\n[DRY RUN] Would re-score {len(papers)} papers")
        return 0

    # Initialize
    positioning_client = create_agent_client('positioning')
    positioning_prompt = (PROMPTS_DIR / "positioning.txt").read_text(encoding='utf-8')
    researcher_profile = (PROMPTS_DIR / "researcher_profile.md").read_text(encoding='utf-8')

    success = 0
    failed = 0

    with db:
        for paper in papers:
            ok = rerun_positioning(
                paper['arxiv_id'], paper,
                positioning_client, positioning_prompt, researcher_profile,
                db
            )
            if ok:
                success += 1
            else:
                failed += 1

    print(f"\n{'='*70}")
    print(f"DONE: {success} re-scored, {failed} failed")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
