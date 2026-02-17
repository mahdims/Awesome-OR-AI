#!/usr/bin/env python3
"""
Layer 3: Living Reviews & Email Reports

Unified CLI for daily, weekly, and monthly operations.

Usage:
    python src/scripts/layer3_run.py daily              # Update reviews + daily email
    python src/scripts/layer3_run.py weekly             # Revise reviews + weekly email with fronts
    python src/scripts/layer3_run.py monthly            # Monthly digest email
    python src/scripts/layer3_run.py daily --preview    # Save HTML locally, don't send
    python src/scripts/layer3_run.py weekly --days 14   # Custom lookback window
    python src/scripts/layer3_run.py weekly --category "LLMs for Algorithm Design"
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database
from layer3.data_collector import get_categories, collect_daily_data, collect_monthly_data
from layer3.enhanced_data_collector import collect_enhanced_weekly_data
from layer3.email_renderer import render_daily_email, render_monthly_email
from layer3.enhanced_email_renderer import render_enhanced_weekly_email
from layer3.email_sender import deliver
from layer3.daily_update import update_living_review
from layer3.weekly_revision import revise_living_review
from layer3.graph_visualizer import visualize_fronts

EMAILS_DIR = Path(__file__).parent.parent.parent / "docs" / "living_reviews" / "emails"
ISSUE_COUNTER_PATH = Path(__file__).parent.parent.parent / "docs" / "bi_daily_issue.json"


# ── Bi-Daily Issue Counter ────────────────────────────────────────────────────

def _load_issue_counter() -> dict:
    """Load the bi-daily issue counter from disk. Returns defaults if missing."""
    if ISSUE_COUNTER_PATH.exists():
        try:
            return json.loads(ISSUE_COUNTER_PATH.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass
    return {"issue": 0, "year": date.today().year, "last_sent": None}


def _save_issue_counter(counter: dict) -> None:
    """Persist the issue counter to disk."""
    ISSUE_COUNTER_PATH.write_text(
        json.dumps(counter, indent=2),
        encoding='utf-8'
    )


def _next_issue_code() -> str:
    """
    Increment and return the next bi-daily issue code.

    Format: "Issue #N of YYYY"
    Counter resets to 1 at the start of each calendar year.
    Persisted in docs/bi_daily_issue.json so missed sends don't inflate counts.
    """
    counter = _load_issue_counter()
    today = date.today()

    # Reset counter at year boundary
    if counter.get('year') != today.year:
        counter['issue'] = 0
        counter['year'] = today.year

    counter['issue'] += 1
    counter['last_sent'] = today.isoformat()
    _save_issue_counter(counter)

    return f"Issue #{counter['issue']} of {counter['year']}"


def run_daily(categories: list, db: Database, days: int = 7,
              preview: bool = False, send: bool = True):
    """Run bi-daily pipeline: update living reviews + send bi-daily email."""
    print(f"\n{'='*70}")
    print(f"BI-DAILY UPDATE — {date.today().isoformat()}")
    print(f"{'='*70}")

    # Step 1: Update living review markdown files
    print(f"\n[STEP 1/2] Updating living reviews...")
    with db:
        for category in categories:
            count = update_living_review(category, db, days=days)
            print(f"  {category}: {count} papers added")

    # Step 2: Generate and deliver bi-daily email
    print(f"\n[STEP 2/2] Generating bi-daily email...")
    categories_data = []
    with db:
        for category in categories:
            data = collect_daily_data(category, db, days=days)
            categories_data.append(data)

    # Get next issue code (increments persistent counter)
    issue_code = _next_issue_code()
    print(f"  {issue_code}")

    html = render_daily_email(categories_data, issue_code=issue_code)

    deliver(
        html_content=html,
        subject=f"Bi-Daily Research Intelligence Briefing — {issue_code} · {date.today().isoformat()}",
        preview_dir=EMAILS_DIR,
        send=send and not preview,
    )

    total = sum(d['stats']['new_count'] for d in categories_data)
    print(f"\n[DONE] Bi-daily update complete: {total} new papers across {len(categories)} categories")


def run_weekly(categories: list, db: Database, days: int = 7,
               preview: bool = False, send: bool = True):
    """
    Run weekly pipeline per category:
      revise living review → graph → enhanced data collection → rich email.

    Each category produces its own email (sent to EMAIL_TO env var at call time).
    """
    print(f"\n{'='*70}")
    print(f"WEEKLY REPORT — {date.today().isoformat()}")
    print(f"{'='*70}")

    for category in categories:
        print(f"\n{'─'*60}")
        print(f"  Category: {category}")
        print(f"{'─'*60}")

        # Step 1: Revise living review with front analysis
        print(f"\n  [1/3] Revising living review...")
        with db:
            revise_living_review(category, db, days=days)

        # Step 2: Generate front network graph
        print(f"\n  [2/3] Generating front network graph...")
        with db:
            graph_b64 = visualize_fronts(category, db)
        if graph_b64:
            print(f"        graph generated")
        else:
            print(f"        no graph (insufficient data)")

        # Step 3: Collect enhanced data + render + deliver
        print(f"\n  [3/3] Collecting data and rendering enhanced email...")
        with db:
            data = collect_enhanced_weekly_data(category, db, days=days)

        # Make graph available to the renderer (stored in data dict)
        if graph_b64:
            data['graph_image_b64'] = graph_b64

        html = render_enhanced_weekly_email(data)

        stats = data['stats']
        print(f"        papers={stats['new_count']}  fronts={stats['fronts_count']}  "
              f"priority={stats.get('top_priority_count', 0)}")

        deliver(
            html_content=html,
            subject=f"Weekly Research Intelligence: {category} — {date.today().isoformat()}",
            preview_dir=EMAILS_DIR,
            filename=f"weekly_{category.lower().replace(' ', '_')[:30]}_{date.today().isoformat()}.html",
            send=send and not preview,
        )

    print(f"\n[DONE] Weekly reports sent for {len(categories)} categories")


def run_monthly(categories: list, db: Database, days: int = 30,
                preview: bool = False, send: bool = True):
    """Run monthly pipeline: full digest email."""
    print(f"\n{'='*70}")
    print(f"MONTHLY DIGEST — {date.today().isoformat()}")
    print(f"{'='*70}")

    # Step 1: Generate graphs
    print(f"\n[STEP 1/2] Generating visualizations...")
    graph_images = {}
    with db:
        for category in categories:
            b64 = visualize_fronts(category, db)
            if b64:
                graph_images[category] = b64

    # Step 2: Collect data and render
    print(f"\n[STEP 2/2] Generating monthly digest...")
    categories_data = []
    with db:
        for category in categories:
            data = collect_monthly_data(category, db, days=days)
            categories_data.append(data)

    html = render_monthly_email(categories_data, graph_images=graph_images)

    deliver(
        html_content=html,
        subject=f"Monthly Research Digest — {date.today().isoformat()}",
        preview_dir=EMAILS_DIR,
        send=send and not preview,
    )

    print(f"\n[DONE] Monthly digest generated for {len(categories)} categories")


def main():
    parser = argparse.ArgumentParser(
        description='Layer 3: Living Reviews & Email Reports'
    )
    parser.add_argument('mode', choices=['daily', 'weekly', 'monthly'],
                        help='Report type to generate')
    parser.add_argument('--days', type=int,
                        help='Lookback window in days (default: 1/7/30 based on mode)')
    parser.add_argument('--category', help='Process only this category')
    parser.add_argument('--preview', action='store_true',
                        help='Save HTML locally, do not send email')
    parser.add_argument('--no-send', action='store_true',
                        help='Skip email sending (still updates reviews)')
    args = parser.parse_args()

    # Default days by mode
    if args.days is None:
        args.days = {'daily': 7, 'weekly': 7, 'monthly': 30}[args.mode]

    db = Database()

    # Resolve categories
    with db:
        if args.category:
            categories = [args.category]
        else:
            categories = get_categories(db)

    if not categories:
        print("[ERROR] No analyzed papers in database. Run Layer 1 first.")
        return 1

    print(f"Categories: {', '.join(categories)}")
    print(f"Mode: {args.mode} | Days: {args.days} | Preview: {args.preview}")

    send = not args.no_send and not args.preview

    if args.mode == 'daily':
        run_daily(categories, db, days=args.days,
                  preview=args.preview, send=send)
    elif args.mode == 'weekly':
        run_weekly(categories, db, days=args.days,
                   preview=args.preview, send=send)
    elif args.mode == 'monthly':
        run_monthly(categories, db, days=args.days,
                    preview=args.preview, send=send)

    return 0


if __name__ == "__main__":
    sys.exit(main())
