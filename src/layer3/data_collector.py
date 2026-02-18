"""
Layer 3: Data Collector

Gathers data from Layer 1 (paper analyses) and Layer 2 (fronts, bridges)
for use in email reports and living review updates.
"""

import json
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import Database


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


def _max_relevance(paper: dict) -> int:
    """Get max relevance score across M/P/I dimensions."""
    rel = _json(paper.get('relevance'), {})
    return max(
        rel.get('methodological', 0),
        rel.get('problem', 0),
        rel.get('inspirational', 0),
    )


def _enrich_paper(paper: dict) -> dict:
    """Parse JSON fields in a paper analysis dict."""
    arxiv_id = paper['arxiv_id']
    return {
        'arxiv_id': arxiv_id,
        'arxiv_url': f'https://arxiv.org/abs/{arxiv_id}',
        'title': paper.get('title', ''),
        'authors': _json(paper.get('authors'), []),
        'abstract': paper.get('abstract', ''),
        'published_date': paper.get('published_date', ''),
        'affiliations': paper.get('affiliations') or '',
        'category': paper.get('category', ''),
        'relevance': _json(paper.get('relevance'), {}),
        'significance': _json(paper.get('significance'), {}),
        'brief': paper.get('brief', ''),
        'methodology': _json(paper.get('methodology'), {}),
        'tags': _json(paper.get('tags'), {}),
        'problem': _json(paper.get('problem'), {}),
        'lineage': _json(paper.get('lineage'), {}),
        'extensions': _json(paper.get('extensions'), {}),
        'artifacts': _json(paper.get('artifacts'), {}),
    }


def _enrich_front(front: dict) -> dict:
    """Parse JSON fields in a research front dict."""
    return {
        'front_id': front['front_id'],
        'category': front['category'],
        'snapshot_date': front.get('snapshot_date', ''),
        'core_papers': _json(front.get('core_papers'), []),
        'size': front.get('size', 0),
        'internal_density': front.get('internal_density', 0),
        'dominant_methods': _json(front.get('dominant_methods'), []),
        'dominant_problems': _json(front.get('dominant_problems'), []),
        'growth_rate': front.get('growth_rate'),
        'stability': front.get('stability'),
        'status': front.get('status', 'unknown'),
        'name': front.get('name', ''),
        'summary': front.get('summary', ''),
        'future_directions': _json(front.get('future_directions'), []),
    }


def _enrich_bridge(bridge: dict) -> dict:
    """Parse JSON fields in a bridge paper dict."""
    return {
        'paper_id': bridge['paper_id'],
        'category': bridge['category'],
        'snapshot_date': bridge.get('snapshot_date', ''),
        'home_front_id': bridge.get('home_front_id', ''),
        'connected_fronts': _json(bridge.get('connected_fronts'), []),
        'bridge_score': bridge.get('bridge_score', 0),
    }


def get_categories(db: Database) -> List[str]:
    """Get all distinct categories from analyzed papers."""
    rows = db.fetchall(
        "SELECT DISTINCT category FROM paper_analyses ORDER BY category"
    )
    return [row['category'] for row in rows]


def collect_daily_data(category: str, db: Database,
                       days: int = 7) -> dict:
    """
    Collect data for daily email: papers published in the last N days.

    Returns dict with:
        - papers: list of enriched paper dicts, sorted by max relevance desc
        - stats: {total, new_count, must_read_count}
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    rows = db.fetchall(
        """SELECT * FROM paper_analyses
           WHERE category = ? AND published_date >= ? AND is_relevant = 1
           ORDER BY published_date DESC""",
        (category, cutoff)
    )
    papers = [_enrich_paper(dict(r)) for r in rows]
    papers.sort(key=_max_relevance, reverse=True)

    total_row = db.fetchone(
        "SELECT COUNT(*) as cnt FROM paper_analyses WHERE category = ? AND is_relevant = 1",
        (category,)
    )

    must_read = sum(1 for p in papers if p['significance'].get('must_read'))

    return {
        'category': category,
        'period_days': days,
        'papers': papers,
        'stats': {
            'total_in_category': total_row['cnt'] if total_row else 0,
            'new_count': len(papers),
            'must_read_count': must_read,
        }
    }


def collect_weekly_data(category: str, db: Database,
                        days: int = 7) -> dict:
    """
    Collect data for weekly email: papers + Layer 2 front analysis.

    Returns dict with:
        - papers: top papers from last N days
        - fronts: latest research fronts with summaries and paper details
        - bridges: top bridge papers with synthesis analysis
        - front_methods: per-front method sets (for overlap matrix)
        - stats: aggregate statistics
    """
    # --- Papers ---
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = db.fetchall(
        """SELECT * FROM paper_analyses
           WHERE category = ? AND analysis_date >= ? AND is_relevant = 1
           ORDER BY published_date DESC""",
        (category, cutoff)
    )
    papers = [_enrich_paper(dict(r)) for r in rows]
    papers.sort(key=_max_relevance, reverse=True)

    # --- Research Fronts ---
    front_rows = db.get_latest_fronts(category)
    fronts = [_enrich_front(f) for f in front_rows]

    # Enrich fronts with per-paper details
    front_methods = {}  # front_id -> set of methods
    for front in fronts:
        methods = set()
        front_papers = []
        for pid in front['core_papers']:
            analysis = db.get_analysis(pid)
            if analysis:
                enriched = _enrich_paper(dict(analysis))
                front_papers.append(enriched)
                methods.update(enriched['tags'].get('methods', []))
        front['papers_detail'] = front_papers
        front_methods[front['front_id']] = methods

    # --- Bridge Papers ---
    snapshot_date = fronts[0]['snapshot_date'] if fronts else None
    bridges = []
    if snapshot_date:
        bridge_rows = db.fetchall(
            """SELECT * FROM bridge_papers
               WHERE category = ? AND snapshot_date = ?
               ORDER BY bridge_score DESC
               LIMIT 5""",
            (category, snapshot_date)
        )
        bridges = [_enrich_bridge(dict(r)) for r in bridge_rows]

        # Enrich bridges with paper analysis + synthesis verdict
        front_lookup = {f['front_id']: f for f in fronts}
        for bridge in bridges:
            analysis = db.get_analysis(bridge['paper_id'])
            if analysis:
                bridge['paper'] = _enrich_paper(dict(analysis))
            else:
                bridge['paper'] = None

            # Synthesis verdict
            paper_methods = set()
            if bridge['paper']:
                paper_methods = set(bridge['paper']['tags'].get('methods', []))

            home_methods = front_methods.get(bridge['home_front_id'], set())
            overlap_home = paper_methods & home_methods

            any_cross = False
            for cid in bridge['connected_fronts']:
                conn_methods = front_methods.get(cid, set())
                if paper_methods & conn_methods:
                    any_cross = True
                    break

            if overlap_home and any_cross:
                bridge['verdict'] = 'TRUE SYNTHESIS'
            elif any_cross:
                bridge['verdict'] = 'CITING BRIDGE'
            elif overlap_home:
                bridge['verdict'] = 'HOME-ANCHORED'
            else:
                bridge['verdict'] = 'STRUCTURAL'

    # --- Method Overlap Matrix ---
    method_overlap = {}
    front_ids = [f['front_id'] for f in fronts]
    for i, fid_a in enumerate(front_ids):
        for j, fid_b in enumerate(front_ids):
            if i < j:
                shared = front_methods.get(fid_a, set()) & front_methods.get(fid_b, set())
                method_overlap[(fid_a, fid_b)] = sorted(shared)

    # --- Stats ---
    total_row = db.fetchone(
        "SELECT COUNT(*) as cnt FROM paper_analyses WHERE category = ? AND is_relevant = 1",
        (category,)
    )
    must_read = sum(1 for p in papers if p['significance'].get('must_read'))

    return {
        'category': category,
        'period_days': days,
        'papers': papers,
        'fronts': fronts,
        'bridges': bridges,
        'front_methods': {k: sorted(v) for k, v in front_methods.items()},
        'method_overlap': {f"{a}|{b}": v for (a, b), v in method_overlap.items()},
        'stats': {
            'total_in_category': total_row['cnt'] if total_row else 0,
            'new_count': len(papers),
            'must_read_count': must_read,
            'fronts_count': len(fronts),
            'bridges_count': len(bridges),
        }
    }


def collect_monthly_data(category: str, db: Database,
                         days: int = 30) -> dict:
    """
    Collect data for monthly review rewrite.
    Same as weekly but with all papers (not just recent).
    """
    weekly = collect_weekly_data(category, db, days=days)

    # Also include ALL relevant papers for the full review
    all_rows = db.fetchall(
        """SELECT * FROM paper_analyses
           WHERE category = ? AND is_relevant = 1
           ORDER BY published_date DESC""",
        (category,)
    )
    weekly['all_papers'] = [_enrich_paper(dict(r)) for r in all_rows]
    weekly['period_days'] = days

    return weekly
